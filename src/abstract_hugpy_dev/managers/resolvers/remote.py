"""Remote execution runners + the worker-provider seam.

resolve() is the single routing authority. When a (model_key, task) should not
run in-process, resolve() swaps the local runner class for one of these:

  * PeerRunner       — static placement.json delegation to another *central*
                       node's POST /api/llm/execute (one-shot). "System A".
  * DelegatingRunner — dynamic offload to a live GPU *worker* from the pool,
                       re-selected on every call, with automatic local
                       fallback. Streams via the worker's /infer/stream and
                       one-shots via /infer. "System B".

Both used to be two unrelated code paths (peers decided inside resolve(), the
worker pool decided in the chat route). Folding the worker pool in here is the
whole point: routing is decided in exactly one place again, and worker offload
now applies to every task and to both run() and stream().

Layering: the worker pool lives in the web layer (it persists next to the model
manifest and is mutated by the /llm/workers routes). To keep this core module
from importing the web layer, the web layer *injects* its selector via
set_worker_provider() at import time. The standalone worker agent never imports
the web layer, so no provider is registered there and DelegatingRunner simply
always runs local — and remote payloads carry _force_local so the far side
never re-delegates (loop guard).
"""
from __future__ import annotations

import os
import json
import base64
import inspect
import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from .imports import (
    TokenEvent, DoneEvent, ErrorEvent, StatusEvent,
)
from .categories import FRAMEWORK_RUNNERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker-provider seam — injected by the web layer (workers.py).
# ---------------------------------------------------------------------------
_worker_provider: Optional[Callable[[str], Optional[dict]]] = None
_spill_provider: Optional[Callable[[str, str], dict]] = None
# Allocator-driven cross-machine placement (web -> core). Returns
# ``{"worker": <lead dict>, "spill": {rpc_servers, tensor_split, n_gpu_layers}}``
# when a model should be SHARDED across the GPU pool, else None to fall through
# to ordinary whole-model selection. None by default ⇒ zero effect on routing.
_placement_provider: Optional[Callable[[str], Optional[dict]]] = None


def set_worker_provider(pick_fn: Callable, spill_fn: Optional[Callable] = None) -> None:
    """Register the live worker selector (web -> core).

    ``pick_fn(model_key) -> worker dict | None`` chooses an online worker
    assigned to the model. ``spill_fn(worker_id, model_key) -> dict`` returns the
    per-assignment GPU/CPU spill override (or {}). Called once, at web-app
    import time.
    """
    global _worker_provider, _spill_provider
    _worker_provider = pick_fn
    _spill_provider = spill_fn
    logger.info("worker provider registered: %s", getattr(pick_fn, "__name__", pick_fn))


def set_placement_provider(place_fn: Optional[Callable]) -> None:
    """Register the allocator-driven shard placement (web -> core), optional."""
    global _placement_provider
    _placement_provider = place_fn
    logger.info("placement provider registered: %s", getattr(place_fn, "__name__", place_fn))


def get_worker_provider() -> Optional[Callable]:
    return _worker_provider


def _pick_worker(model_key: str, pool: Optional[str] = None) -> Optional[dict]:
    if _worker_provider is None:
        return None
    try:
        # The provider may predate the pool arg — fall back to the 1-arg form.
        try:
            return _worker_provider(model_key, pool)
        except TypeError:
            return _worker_provider(model_key)
    except Exception as exc:  # never let pool selection break a request
        logger.warning("worker provider failed for %s: %s", model_key, exc)
        return None


def _select(model_key: str, pool: Optional[str] = None) -> tuple[Optional[dict], Optional[dict]]:
    """Choose where this request runs: ``(worker, spill_override)``.

    ``pool`` (when set) restricts selection to that dedicated worker pool, and a
    general request never lands on a pooled worker — see workers_for_model.

    First ask the placement provider — if it returns a shard plan, the lead
    worker + its rpc/tensor_split spill win. Otherwise fall back to ordinary
    whole-model selection (``spill_override=None`` ⇒ use the per-assignment
    spill). Any failure degrades to normal selection; sharding never breaks a
    request.
    """
    if _placement_provider is not None:
        try:
            plan = _placement_provider(model_key)
        except Exception as exc:
            logger.warning("placement provider failed for %s: %s", model_key, exc)
            plan = None
        if plan and plan.get("worker"):
            logger.info("sharded placement for %s: lead=%s rpc=%s",
                        model_key, plan["worker"].get("id"),
                        (plan.get("spill") or {}).get("rpc_servers"))
            return plan["worker"], (plan.get("spill") or None)
    return _pick_worker(model_key, pool), None


def _spill_for(worker_id: Optional[str], model_key: str) -> dict:
    if _spill_provider is None or not worker_id:
        return {}
    try:
        return _spill_provider(worker_id, model_key) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Worker transport — build the body, inline files, parse the SSE relay.
# ---------------------------------------------------------------------------

# Above this size we don't inline an upload to a worker; the turn runs local.
_MAX_WORKER_FILE_BYTES = 256 * 1024 * 1024

# Request fields that name a local path the worker can't see. We inline whichever
# is present as base64; the worker materializes it back to its own temp path and
# its builder picks it up as "file".
_PATH_KEYS = ("file", "image_path", "audio_path", "file_path")


def _inline_file(payload: dict) -> bool:
    """Replace a local path field with inline bytes the worker can rebuild.

    Returns False (→ run local) if the referenced file is missing or too big.
    True when there was nothing to inline or inlining succeeded.
    """
    key = next((k for k in _PATH_KEYS if payload.get(k)), None)
    if key is None:
        return True
    path = payload[key]
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > _MAX_WORKER_FILE_BYTES:
            return False
        with open(path, "rb") as fh:
            payload["file_b64"] = base64.b64encode(fh.read()).decode("ascii")
        payload["file_name"] = os.path.basename(path)
        payload.pop(key, None)
        return True
    except OSError:
        return False


def _worker_payload(task: str, req, model_key: str, worker_id: Optional[str],
                    spill_override: Optional[dict] = None) -> Optional[dict]:
    """JSON body for a worker /infer[/stream] call, built from a built req.

    A worker re-runs execute_prompt(**body), and req.model_dump() already uses
    prompt_kwargs field names (messages, model_key, image_path, ...). We add the
    resolved task + _force_local (loop guard) and the spill override, then inline
    a local file the worker can't reach. ``spill_override`` (a shard plan's
    rpc_servers/tensor_split) wins over the per-assignment spill when present.
    Returns None to signal "can't offload this turn, run local".
    """
    payload: Dict[str, Any] = {"task": task, "_force_local": True, **req.model_dump()}
    spill = spill_override if spill_override is not None else _spill_for(worker_id, model_key)
    if spill:
        payload["spill"] = spill
    if not _inline_file(payload):
        return None
    return payload


# llama.cpp / OpenAI finish reasons -> DoneEvent's strict Literal.
_WORKER_FINISH_MAP = {
    "length": "max_tokens", "max_tokens": "max_tokens",
    "stop": "stop", "eos": "stop", None: "stop",
    "cancelled": "cancelled", "error": "error",
}


def _event_from_worker_line(d: dict, request_id: str):
    """Map one worker SSE dict to a StreamEvent.

    token/done/error become the typed events; everything else
    (request/status/provision-progress) rides through as a StatusEvent so the
    browser still sees progress.
    """
    t = d.get("type")
    if t == "status" and d.get("stage") == "dispatch":
        # The worker runs the same dispatch engine and announces ITS OWN
        # allocation — "served_by: local" meaning local-to-the-worker. Relayed
        # verbatim it lands AFTER central's true banner and overwrites it, so
        # the console shows "local" while the worker is in fact serving (the
        # great phantom-fallback of 2026-07-02). Central owns the allocation
        # banner; drop the worker's inner one.
        return None
    if t == "token":
        return TokenEvent(request_id=request_id, text=d.get("text", ""))
    if t == "done":
        # Workers emit raw llama.cpp reasons ('length', 'stop', ...); DoneEvent's
        # finish_reason is a strict Literal (stop/max_tokens/cancelled/error), so
        # map first. Without this, a token-capped worker's terminal 'done' fails
        # the Literal and gets silently downgraded to a StatusEvent (no real done).
        finish = _WORKER_FINISH_MAP.get(d.get("finish_reason"), "stop")
        try:
            return DoneEvent(
                request_id=request_id,
                input_tokens=d.get("input_tokens", 0),
                output_chunks=d.get("output_chunks", 1),
                finish_reason=finish,
            )
        except Exception:
            return StatusEvent(**{**d, "request_id": request_id})
    if t == "error":
        return ErrorEvent(request_id=request_id, message=d.get("message", "worker error"))
    return StatusEvent(**{**d, "request_id": d.get("request_id", request_id)})


async def _worker_stream(worker: dict, payload: dict, request_id: str):
    """Relay a worker's POST /infer/stream SSE as StreamEvents.

    Raising before the first event lets the caller fall back to local; a short
    connect timeout makes a dead worker fail over fast, a long read timeout
    leaves room for generation.
    """
    import httpx

    url = worker["url"].rstrip("/") + "/infer/stream"
    timeout = httpx.Timeout(600.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                ev = _event_from_worker_line(d, request_id)
                if ev is None:      # suppressed (worker's inner dispatch banner)
                    continue
                yield ev


async def _worker_run_once(worker: dict, payload: dict, result_type, request_id: str, model_key: str):
    """One-shot worker POST /infer; validate the response into result_type.

    Tolerant of a worker that returns the slim {ok,text,finish_reason} shape by
    filling request_id/model_key defaults before validation.

    Plain httpx like _worker_stream — BYTE-FAITHFUL on purpose. The previous
    abstract_apis transport recursively json-parsed every string field of the
    reply (load_inner_json), so any model answer that happened to be valid JSON
    ("{}", "42", "true", a JSON-formatted reply, …) mutated text:str into a
    dict/int/bool, failed result_type validation here, and silently re-ran the
    whole request on central — a phantom local fallback that looked random
    because it depended on what the model said.
    """
    import httpx

    url = worker["url"].rstrip("/") + "/infer"
    timeout = httpx.Timeout(3600.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if isinstance(data, dict):
        data.setdefault("request_id", request_id)
        data.setdefault("model_key", model_key)
        data.setdefault("ok", True)
    return result_type.model_validate(data)


# ---------------------------------------------------------------------------
# Runner factories — what resolve() swaps in for the local runner class.
# ---------------------------------------------------------------------------

def make_peer_runner(peer, framework: str, task: str):
    """Static placement.json delegation to another central node (one-shot)."""
    local_cls = FRAMEWORK_RUNNERS[(framework, task)]   # borrow request/result types

    class PeerRunner:
        request_type = local_cls.request_type
        result_type = local_cls.result_type

        def __init__(self, cfg):
            self.cfg = cfg
            self.model_key = cfg.model_key

        async def run(self, req):
            # httpx, byte-faithful — see _worker_run_once: abstract_apis'
            # load_inner_json re-parses string fields and corrupts JSON-shaped
            # model replies, failing validation.
            import httpx
            payload = {"delegated": True, "task": task, **req.model_dump()}
            url = peer.base_url.rstrip("/") + "/api/llm/execute"
            timeout = httpx.Timeout(float(self.cfg.timeout_s or 3600), connect=4.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return self.result_type.model_validate(data)

    return PeerRunner


def _alloc_status(request_id: str, worker: Optional[dict]):
    """A status event announcing which allocation served this request — drives
    the chat box's allocation banner. ``served_by`` is "worker" or "local"; for
    a worker we carry its registry name + id. StatusEvent is extra="allow", so
    these fields ride to the browser verbatim via the SSE model_dump(). Emitted
    again as "local" on fallback so the banner reflects the *actual* server, not
    just the intended pick."""
    if worker:
        wid = worker.get("id") or ""
        return StatusEvent(
            request_id=request_id, stage="dispatch", served_by="worker",
            worker_id=wid, worker_name=worker.get("name") or wid,
        )
    return StatusEvent(
        request_id=request_id, stage="dispatch", served_by="local",
        worker_id="", worker_name="local",
    )


def _local_fallback_allowed() -> bool:
    """Whether central may run a WORKER-SELECTED model locally after the
    worker path fails.

    Default NO: an operator who assigned a model to a GPU worker designated
    where it runs — silently re-running a multi-GB model on a (typically
    GPU-less) central burns its CPU/RAM and hides the worker failure (the
    2026-07-02 central-meltdown mode). Models with NO worker selected still
    run local as always. Set HUGPY_LOCAL_FALLBACK=always to restore the old
    degrade-to-local behavior."""
    return (os.environ.get("HUGPY_LOCAL_FALLBACK", "").strip().lower()
            in ("always", "1", "true", "yes", "on"))


def _worker_vision_capable(worker: Optional[dict]) -> bool:
    """True only when the worker AFFIRMATIVELY reports its llama.cpp build can run
    vision (mtmd) — engine.supports_vision. Central does not guess: it trusts what
    the worker says about itself. A worker that doesn't advertise it (older agent)
    or reports it can't is treated as NOT vision-capable, so an image turn never
    lands on a server that would ignore the image and answer from text alone."""
    eng = (worker or {}).get("engine") or {}
    return bool(eng.get("supports_vision"))


def make_delegating_runner(framework: str, task: str):
    """Dynamic worker-pool offload with local fallback, decided per request.

    Cacheable by (model_key, task) because the worker is re-selected on every
    call — the cached instance means "delegate to whatever worker is live for
    this model, otherwise run local". It lazily builds the real local runner so
    the fallback shares dispatch's instance cache semantics.
    """
    local_cls = FRAMEWORK_RUNNERS[(framework, task)]
    _vision_task = (task == "image-text-to-text")

    # Every task delegates to a worker when one is live for this model; the worker
    # owns the GPU. The image rides inline in the worker payload (_worker_payload /
    # _inline_file). For non-vision tasks we do NOT second-guess a live worker —
    # the request goes where it's selected to go; the ONLY fallback is genuine
    # unreachability (no live worker, or it fails BEFORE producing output).
    #
    # Vision is the one exception, and it's CAPABILITY-HONEST, not a guess: a
    # llama.cpp worker only serves an image turn if it AFFIRMATIVELY advertises it
    # can do vision (engine.supports_vision — _worker_vision_capable). A worker
    # that can't run the multimodal projector (older agent, or a build whose mtmd
    # init fails) would silently drop the image and hallucinate from text alone, so
    # we route the turn to a capable server instead — another capable worker if one
    # exists, else the local engine. "The one that does vision is the one assigned
    # to vision": whatever can actually see the image serves it.

    class DelegatingRunner:
        request_type = local_cls.request_type
        result_type = local_cls.result_type

        def __init__(self, cfg):
            self.cfg = cfg
            self.model_key = cfg.model_key
            self._local = None

        def _local_runner(self):
            if self._local is None:
                self._local = local_cls(self.cfg)
            return self._local

        async def run(self, req):
            worker, spill_override = _select(self.model_key, getattr(req, "pool", None))
            if worker and _vision_task and not _worker_vision_capable(worker):
                logger.info("worker %s doesn't advertise vision (engine.supports_vision); "
                            "serving %s where vision actually works instead",
                            worker.get("id"), self.model_key)
                worker = None
            if worker:
                payload = _worker_payload(task, req, self.model_key, worker.get("id"),
                                          spill_override=spill_override)
                if payload is not None:
                    try:
                        return await _worker_run_once(
                            worker, payload, self.result_type,
                            request_id=req.request_id, model_key=self.model_key,
                        )
                    except Exception as exc:
                        if not _local_fallback_allowed():
                            raise RuntimeError(
                                f"worker {worker.get('name') or worker.get('id')} "
                                f"failed for {self.model_key}: {exc} (local "
                                f"fallback disabled for worker-assigned models; "
                                f"set HUGPY_LOCAL_FALLBACK=always to allow)") from exc
                        logger.warning("worker run failed (%s); running %s locally",
                                       exc, self.model_key)
            result = self._local_runner().run(req=req)
            if inspect.isawaitable(result):
                result = await result
            return result

        async def stream(self, req, cancel_event=None):
            worker, spill_override = _select(self.model_key, getattr(req, "pool", None))
            if worker and _vision_task and not _worker_vision_capable(worker):
                logger.info("worker %s doesn't advertise vision (engine.supports_vision); "
                            "serving %s where vision actually works instead",
                            worker.get("id"), self.model_key)
                worker = None
            if worker:
                payload = _worker_payload(task, req, self.model_key, worker.get("id"),
                                          spill_override=spill_override)
                if payload is not None:
                    # Announce the allocation up front — the chat banner shows
                    # this. A pre-output failure falls through and re-announces
                    # "local" below, so the banner ends on the actual server.
                    yield _alloc_status(req.request_id, worker)
                    produced_tokens = False
                    wname = worker.get("name") or worker.get("id") or "worker"
                    try:
                        async for ev in _worker_stream(worker, payload, req.request_id):
                            etype = getattr(ev, "type", None)
                            if etype == "error":
                                # A worker that errors AFTER streaming tokens can't
                                # be replayed locally (would duplicate output) — so
                                # surface it as interrupted. A worker that errors
                                # BEFORE any token used to degrade to local — but a
                                # worker-ASSIGNED model running on central is a
                                # policy violation (and a CPU meltdown for big
                                # models), so by default the worker's error is
                                # surfaced instead (see _local_fallback_allowed).
                                if produced_tokens:
                                    yield ErrorEvent(request_id=req.request_id,
                                                     message=f"worker stream interrupted: {ev.message}")
                                    return
                                if not _local_fallback_allowed():
                                    yield ErrorEvent(
                                        request_id=req.request_id,
                                        message=f"worker {wname} failed before output: "
                                                f"{ev.message} (local fallback disabled "
                                                f"for worker-assigned models)")
                                    return
                                logger.warning("worker %s errored before output (%s); "
                                               "running %s locally", worker.get("id"),
                                               ev.message, self.model_key)
                                break  # -> local fallback below
                            yield ev
                            if etype == "token":
                                produced_tokens = True
                            elif etype == "done":
                                return  # worker completed (even if empty) — terminal
                        else:
                            # Stream ended with no done/error marker.
                            if produced_tokens:
                                return
                            if not _local_fallback_allowed():
                                yield ErrorEvent(
                                    request_id=req.request_id,
                                    message=f"worker {wname} produced no output "
                                            f"(local fallback disabled for "
                                            f"worker-assigned models)")
                                return
                            logger.warning("worker %s produced no output; running %s locally",
                                           worker.get("id"), self.model_key)
                    except Exception as exc:
                        if produced_tokens:
                            # Stream already started — don't replay it locally.
                            yield ErrorEvent(request_id=req.request_id,
                                             message=f"worker stream interrupted: {exc}")
                            return
                        if not _local_fallback_allowed():
                            yield ErrorEvent(
                                request_id=req.request_id,
                                message=f"worker {wname} unreachable/failed: {exc} "
                                        f"(local fallback disabled for worker-"
                                        f"assigned models; set "
                                        f"HUGPY_LOCAL_FALLBACK=always to allow)")
                            return
                        logger.warning("worker offload failed (%s); running %s locally",
                                       exc, self.model_key)
            # Local fallback — reuse dispatch's shared stream-or-wrap primitive
            # (imported lazily to avoid a resolvers<->dispatch import cycle).
            # Re-announce as "local" so the banner reflects this path (covers
            # no worker selected, unbuildable payload, and pre-output failure).
            yield _alloc_status(req.request_id, None)
            from ..dispatch.dispatch import stream_runner
            async for ev in stream_runner(self._local_runner(), req, cancel_event=cancel_event):
                yield ev

    return DelegatingRunner
