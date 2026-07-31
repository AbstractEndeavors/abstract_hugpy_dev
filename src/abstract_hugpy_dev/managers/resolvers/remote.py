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
import time
import base64
import inspect
import asyncio
import logging
import threading
import contextvars
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Cap-aware relay reroute (concurrency hardening 2026-07-11). Returns the ranked
# list of ONLINE workers holding a model, so the in-flight gate can reroute
# around a worker that is at its advertised in-process concurrency cap. None ⇒
# the gate only ever considers the primary pick (older web layer / standalone).
_worker_candidates_provider: Optional[Callable[..., List[dict]]] = None


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


def set_worker_candidates_provider(candidates_fn: Optional[Callable]) -> None:
    """Register the cap-aware reroute list provider (web -> core), optional.

    ``candidates_fn(model_key, pool) -> list[worker dict]`` returns the ranked
    online workers holding the model (no routing side effects). The relay gate
    consults it to find an alternative when the primary pick is at its in-process
    concurrency cap. Unregistered ⇒ the gate degrades to primary-only.
    """
    global _worker_candidates_provider
    _worker_candidates_provider = candidates_fn
    logger.info("worker candidates provider registered: %s",
                getattr(candidates_fn, "__name__", candidates_fn))


def get_worker_provider() -> Optional[Callable]:
    return _worker_provider


# ---------------------------------------------------------------------------
# Serve-metrics sink (web -> core, optional) — operator 2026-07-25:
# "in the end it is about maximizing tok/s ... lets start recording this".
#
# llama-server measures its own decode rate and ships it in a `timings` block on
# every completion; we were discarding it. The relay is the one place central
# learns BOTH which (worker, model) served a request AND how fast it decoded, so
# it is the natural stamping point — but core must not import the web worker
# store, so it goes through the same provider registration every other
# web->core seam here uses. Unregistered (standalone worker / bare central) ⇒
# no-op, byte-identical behaviour to before.
#
# ``sink(worker_id, model_key, tok_s) -> None``. Writes onto the ONE ledger
# (``model_call_stats``), never a second store.
#
# ⚠ RECORDING ONLY. Nothing reads tok/s yet; eviction.sort_key is untouched.
_serve_metrics_sink: Optional[Callable[..., Any]] = None


def set_serve_metrics_sink(sink_fn: Optional[Callable]) -> None:
    """Register the ledger writer for measured serve quality (web -> core)."""
    global _serve_metrics_sink
    _serve_metrics_sink = sink_fn
    logger.info("serve metrics sink registered: %s",
                getattr(sink_fn, "__name__", sink_fn))


def _record_serve_metrics(worker: Optional[dict], model_key: str,
                          payload: Any) -> None:
    """Extract engine tok/s from a worker reply and stamp the ledger.

    TOTALLY FAIL-OPEN, and that is the design constraint, not a nicety: this
    runs on the LIVE serving path, and a relay that raises because a `timings`
    key is missing is a far worse bug than not recording. Every failure mode —
    no sink registered, no worker, no timings block, an old worker that never
    sends one, a store write that fails — returns quietly having done nothing.
    """
    if _serve_metrics_sink is None or not worker:
        return
    try:
        wid = worker.get("id")
        if not wid:
            return
        from ..eviction import tok_s_from_timings
        tok_s = tok_s_from_timings(payload)
        if tok_s is None:
            return
        _serve_metrics_sink(wid, model_key, tok_s)
    except Exception:  # noqa: BLE001 — recording must never fail a request
        logger.debug("serve-metrics recording skipped for %s", model_key,
                     exc_info=True)


# ---------------------------------------------------------------------------
# MODEL GROUPS — member selector (web -> core, optional). OFF BY DEFAULT.
#
# A model group picks WHICH ITERATION of a base model serves a request: the
# transformers repo or the GGUF, and which rung of the GGUF's quant ladder.
# That is a different question from "which box" — it is answered BEFORE box
# selection — so it takes its own seam here rather than colliding with
# ``_select``, which already hosts ``_placement_provider`` for sharding
# (PLACEMENT-SCHEDULER-PLAN: "reconcile with it, don't collide").
#
# ⚠ THE OFF-PATH IS THE CONTRACT (operator directive 2026-07-28). The whole
# feature is behind one settings flag, default FALSE, plus a hard env off
# (HUGPY_MODEL_GROUPS=off). The provider's FIRST LINE is that check, so with
# groups off ``_member_key`` returns None, the consult below is a no-op, and
# resolution is byte-identical to the pre-feature tree. Guarded by
# tests/test_model_groups_offpath.py, which was written and green BEFORE this
# seam existed. Reverting the feature is flipping the flag — not a code change.
#
# Unset on the standalone worker / bare central => None => same thing.
_member_selector: Optional[Callable[..., Optional[str]]] = None

# The EFFECTIVE model key for the request currently in flight, or None for
# "whatever the caller named". See DelegatingRunner.model_key for why this is a
# context value and not an attribute: the runner instance is CACHED AND SHARED
# across every concurrent request for a model, so a group's choice cannot live
# on it. Default None means every off-path request reads through to the base
# key with no context lookup cost worth measuring.
_MEMBER_KEY: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "hugpy_group_member_key", default=None)


def set_member_selector(select_fn: Optional[Callable]) -> None:
    """Register the model-group member selector (web -> core), optional."""
    global _member_selector
    _member_selector = select_fn
    logger.info("model-group member selector registered: %s",
                getattr(select_fn, "__name__", select_fn))


def get_member_selector() -> Optional[Callable]:
    return _member_selector


def _member_key(model_key: str, pool: Optional[str] = None,
                task: Optional[str] = None) -> Optional[str]:
    """The group member to route instead of ``model_key``, or None.

    None means "change nothing" and is the answer in every one of these cases:
    the feature is off, the seam is unregistered, the key is in no group, the
    group chose the key the caller already named, or ANYTHING went wrong. A
    routing decision must never be the thing that raises."""
    if _member_selector is None:
        return None
    try:
        # Arg-count degradation, same convention as every other seam here.
        for _args in ((model_key, pool, task), (model_key, pool), (model_key,)):
            try:
                chosen = _member_selector(*_args)
                break
            except TypeError:
                continue
        else:
            return None
        chosen = (str(chosen).strip() if chosen else "")
        return chosen or None
    except Exception as exc:  # noqa: BLE001 — never break a request over routing
        logger.warning("model-group member selection failed for %s: %s",
                       model_key, exc)
        return None


# ---------------------------------------------------------------------------
# No-worker diagnostic (web -> core, optional).
#
# When selection yields no worker AND this box refuses local serving
# (HUGPY_NO_LOCAL_SERVING), the refused-local error is otherwise opaque ("no
# registered worker is available"). That is exactly the mystery a DESIGNATED-but-
# idle model presents: it is assigned + pinned + on disk, yet the request 500s
# with no hint that its assigned worker was SKIPPED (e.g. a broken llama-cpp / no
# native llama-server binary). This seam, given (model_key, pool, task), returns a
# human-readable reason so the error names the real cause. Unset on the standalone
# worker / bare central ⇒ detail="" ⇒ the message is byte-identical to before.
_no_worker_diag: Optional[Callable[..., str]] = None


def set_no_worker_diagnostic(diag_fn: Optional[Callable]) -> None:
    """Register the assigned-but-excluded explainer (web -> core), optional."""
    global _no_worker_diag
    _no_worker_diag = diag_fn
    logger.info("no-worker diagnostic registered: %s",
                getattr(diag_fn, "__name__", diag_fn))


def _no_worker_detail(model_key: str, pool: Optional[str] = None,
                      task: Optional[str] = None) -> str:
    """Best-effort human reason no worker took a request — the refused-local
    error's ``detail``. "" when the seam is unset or on ANY failure, so it can
    never turn a clean policy refusal into a 500 (advisory only)."""
    if _no_worker_diag is None:
        return ""
    try:
        # Degrade arg-count like the other seams, for a provider on older code.
        for _args in ((model_key, pool, task), (model_key, pool), (model_key,)):
            try:
                return (_no_worker_diag(*_args) or "").strip()
            except TypeError:
                continue
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break a request
        logger.warning("no-worker diagnostic failed for %s: %s", model_key, exc)
    return ""


def _pick_worker(model_key: str, pool: Optional[str] = None,
                 task: Optional[str] = None,
                 require_comfy_id_lock: bool = False) -> Optional[dict]:
    if _worker_provider is None:
        return None
    try:
        # The provider may predate the pool/task/id_lock args (a peer on older
        # code) — widest form first, degrading to narrower ones on an arg-count
        # TypeError. If an OLD provider drops require_comfy_id_lock, the comfy
        # runner's request-time object_info probe is still the honest backstop
        # (it fails as data on a nodeless comfy — never a silent non-locked image).
        for _args in ((model_key, pool, task, require_comfy_id_lock),
                      (model_key, pool, task), (model_key, pool), (model_key,)):
            try:
                return _worker_provider(*_args)
            except TypeError:
                continue
        return None
    except Exception as exc:  # never let pool/task selection break a request
        logger.warning("worker provider failed for %s: %s", model_key, exc)
        return None


def _select(model_key: str, pool: Optional[str] = None,
            task: Optional[str] = None,
            require_comfy_id_lock: bool = False) -> tuple[Optional[dict], Optional[dict]]:
    """Choose where this request runs: ``(worker, spill_override)``.

    ``pool`` (when set) restricts selection to that dedicated worker pool, and a
    general request never lands on a pooled worker — see workers_for_model.
    ``task`` (when set) additionally skips a worker that advertises it can't run
    that task (a missing optional ML dep — the 2026-07-11 request-time-failure
    class); legacy/unknown = capable. ``require_comfy_id_lock`` (set for an
    identity-locked STILL request) restricts to boxes whose ComfyUI advertises
    the IPAdapter nodes (STRICT — id_lock must never route to a nodeless comfy).

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
    return _pick_worker(model_key, pool, task, require_comfy_id_lock), None


def _requested_worker_name(req) -> Optional[str]:
    """The per-request WORKER pin, riding the same ``alloc`` dict as the other
    triggers (operator ask 2026-07-29: "it'll need its worker allocation"):
    ``{"alloc": {"worker": "ae", ...}}``. Not in _REQUEST_ALLOC_KEYS on purpose
    — it steers ROUTING, never the spill wire."""
    a = getattr(req, "alloc", None)
    if isinstance(a, dict):
        w = a.get("worker")
        if isinstance(w, str) and w.strip():
            return w.strip()
    return None


def _resolve_requested_worker(want: str, model_key: str,
                              pool: Optional[str], task: Optional[str]) -> dict:
    """The worker row for an explicitly requested worker, by name or id.

    An explicit pin is a CONTRACT: if the named worker cannot serve, the call
    fails naming why — it is never silently rerouted to a different box or to
    local (an A/B against a named card that quietly ran elsewhere would be a
    lie in the data). The message carries 'requested worker', which
    _PERMANENT_LOAD_MARKERS classifies as permanent: fail fast, no hold."""
    rows = _candidates(model_key, pool, task)
    for w in rows:
        if want in (w.get("name"), w.get("id")):
            return w
    have = ", ".join(str(w.get("name") or w.get("id")) for w in rows) or "none"
    raise RuntimeError(
        f"requested worker '{want}' cannot serve '{model_key}': not among the "
        f"online task-capable workers holding it (eligible: {have})")


def _spill_for(worker_id: Optional[str], model_key: str) -> dict:
    if _spill_provider is None or not worker_id:
        return {}
    try:
        return _spill_provider(worker_id, model_key) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Cap-aware relay admission (concurrency hardening — the central half).
#
# The worker gate (worker_agent.gen_gate) stops a SINGLE box from letting
# concurrent requests race a non-reentrant in-process runner and crash. Central
# does the complementary thing: it never FIRES a relay that would enter a busy
# in-process runner in the first place. It tracks how many relays are in flight
# per (worker, model), respects the worker's advertised in-process concurrency
# cap, reroutes to another online worker holding the model when the primary is
# full, waits briefly for a slot to free, and only then returns an honest busy
# error — so a burst serializes (or degrades honestly) instead of core-dumping a
# worker or piling up unboundedly.
#
# v0 honesty (deliberate): the in-flight counter is per-GUNICORN-PROCESS (a
# module global). With N gunicorn worker processes each gates its own relays, so
# the fleet-wide per-(worker,model) concurrency can exceed the cap by up to N-1.
# That is still a hard bound (never unbounded) AND the per-worker gen_gate is the
# authoritative backstop that actually prevents the crash. Cross-process exact
# accounting can later ride the comms SQLite mirror (the same store jobs use);
# it is intentionally NOT built here. A worker whose model is SLOT-served is not
# gated centrally at all (its llama-server child schedules concurrency itself).
# ---------------------------------------------------------------------------

_INFLIGHT: Dict[Tuple[str, str], int] = {}
_INFLIGHT_LOCK = threading.Lock()


def _gate_disabled() -> bool:
    return os.environ.get("HUGPY_CENTRAL_GATE", "").strip().lower() in (
        "off", "0", "false", "no",
    )


def _gate_wait_s() -> float:
    """Bounded wait for a busy (worker, model) slot to free before giving up."""
    try:
        return max(0.0, float(os.environ.get("HUGPY_CENTRAL_GATE_WAIT_S", "30")))
    except (TypeError, ValueError):
        return 30.0


class WorkerBusyError(RuntimeError):
    """No worker holding the model has in-process capacity within the bounded wait.

    The honest 429/503 the caller surfaces (route maps it to a status). Carries
    the busy worker's name/id, the model, and its in-flight count so the message
    names exactly what is saturated.
    """

    def __init__(self, worker: Optional[dict], model_key: Optional[str], in_flight: int):
        self.worker = worker or {}
        self.model_key = model_key
        self.in_flight = int(in_flight)
        self.worker_name = self.worker.get("name") or self.worker.get("id") or "worker"
        super().__init__(self.stream_message())

    def stream_message(self) -> str:
        return (f"worker_busy: {self.worker_name} is at its in-process concurrency "
                f"limit for {self.model_key} ({self.in_flight} in flight) and no "
                f"other worker holding it is free — retry shortly")

    def as_error(self) -> Dict[str, Any]:
        return {"error": {
            "code": "worker_busy",
            "message": self.stream_message(),
            "worker": self.worker_name,
            "worker_id": self.worker.get("id"),
            "model": self.model_key,
            "in_flight": self.in_flight,
        }}


def _advertised_cap(worker: Optional[dict]) -> int:
    """The worker's advertised safe in-process concurrency for a model.

    Reads ``serving_limits.in_process_max_concurrency``. ABSENT (an older agent
    that predates the field) → 1: a llama.cpp ``Llama`` context and an in-process
    transformers model serialize per model, so 1 is the crash-safe legacy
    assumption. A non-positive advertised value is floored to 1 — 'unlimited'
    in-process concurrency is exactly the crash, never honored.
    """
    lim = (worker or {}).get("serving_limits") or {}
    n = lim.get("in_process_max_concurrency")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 1
    return max(1, n)


def _slot_match_keys(model_key: str) -> set:
    """Alias form-set for ``model_key`` — REUSES workers._match_keys (the Slice A
    ~/-tail unification), so slot classification agrees with routing on which
    spellings name the same model. Lazy, guarded import: remote.py deliberately
    avoids importing the web layer at module load (the standalone worker agent
    imports this module but never calls slot classification — that is central-
    only). On any import failure, degrade to a local mirror of the same
    (raw / lowercased / "/"-tail / "~"-tail) form-set — never a divergent rule,
    just an inlined copy so a broken import can't silently drop ~-matching.
    """
    if not model_key:
        return set()
    try:
        from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import (
            _match_keys as _wk_match_keys)
        return _wk_match_keys(model_key)
    except Exception:  # noqa: BLE001 — inline mirror of workers._match_keys
        raw = str(model_key).strip()
        forms = {raw, raw.lower()}
        tail = raw.split("/")[-1]
        forms.add(tail)
        forms.add(tail.lower())
        if "~" in raw:
            base = raw.split("~", 1)[1]
            if base:
                forms.add(base)
                forms.add(base.lower())
        return forms


def _model_slot_served(worker: Optional[dict], model_key: str) -> bool:
    """True when ``model_key`` is currently seated in a SLOT child on this worker.

    Then the worker's llama-server / llama_cpp.server child schedules concurrency
    itself and central must NOT apply the in-process cap. Best-effort over the
    heartbeat ``slots``/``allocations`` snapshot; any doubt → False (apply the
    cap — over-gating a slot model is a small latency cost, under-gating an
    in-process model is a crash).

    ALIAS-TOLERANT (2026-07-23 incident): a slot seated under the BARE key
    ``Qwen3-Coder-Next-GGUF`` must classify slot-served for a ~-qualified request
    ``Qwen~Qwen3-Coder-Next-GGUF`` and vice versa — otherwise the ~-spelling
    missed the slot and fell to the in-process cap-1 (``worker_busy`` while the
    model was seated and relay-uncapped). Matching goes through the SAME
    ~/-tail unification routing uses (``_slot_match_keys`` → workers._match_keys);
    the cap SEMANTICS are unchanged, only the KEY MATCHING became alias-tolerant.
    """
    if not worker or not model_key:
        return False
    keys = set()
    for s in (worker.get("slots") or []):
        if isinstance(s, dict) and s.get("model_key") and s.get("healthy"):
            keys.add(str(s["model_key"]))
    for a in (worker.get("allocations") or []):
        if isinstance(a, dict) and a.get("kind") == "slot" and a.get("model_key"):
            keys.add(str(a["model_key"]))
    if not keys:
        return False
    if model_key in keys:
        return True
    # Alias match: does any slot key share a normalized form with the request?
    wanted = _slot_match_keys(model_key)
    return any(wanted & _slot_match_keys(k) for k in keys)


def _effective_cap(worker: Optional[dict], model_key: str) -> Optional[int]:
    """The in-process concurrency cap to enforce for (worker, model), or None to
    NOT gate (the model is slot-served — its child schedules itself)."""
    if _model_slot_served(worker, model_key):
        return None
    return _advertised_cap(worker)


def _inflight_try_acquire(worker_id: str, model_key: str, cap: int) -> bool:
    key = (worker_id, model_key)
    with _INFLIGHT_LOCK:
        cur = _INFLIGHT.get(key, 0)
        if cur < cap:
            _INFLIGHT[key] = cur + 1
            return True
        return False


def _inflight_release(worker_id: str, model_key: str) -> None:
    key = (worker_id, model_key)
    with _INFLIGHT_LOCK:
        cur = _INFLIGHT.get(key, 0)
        if cur <= 1:
            _INFLIGHT.pop(key, None)
        else:
            _INFLIGHT[key] = cur - 1


def _inflight_count(worker_id: str, model_key: str) -> int:
    with _INFLIGHT_LOCK:
        return _INFLIGHT.get((worker_id, model_key), 0)


def _candidates(model_key: str, pool: Optional[str] = None,
                task: Optional[str] = None) -> List[dict]:
    """Ranked online workers holding the model (reroute list), or [] if no
    provider / any failure — the gate then considers only the primary. ``task``
    (when set) keeps the reroute list task-capable, same as the primary pick."""
    if _worker_candidates_provider is None:
        return []
    try:
        # Widest form first (see _pick_worker), degrading on an arg-count TypeError.
        for _args in ((model_key, pool, task), (model_key, pool), (model_key,)):
            try:
                return _worker_candidates_provider(*_args) or []
            except TypeError:
                continue
        return []
    except Exception as exc:  # never let reroute break a request
        logger.warning("candidates provider failed for %s: %s", model_key, exc)
        return []


_NOOP_RELEASE = lambda: None  # noqa: E731 — a tiny sentinel is clearer inline

# Per-REQUEST alloc triggers a client may override (operator ask 2026-07-29).
# Exactly the worker's _SPILL_ENV vocabulary that is safe per-call — placement
# levers only. Deliberately excluded: rpc_servers/tensor_split/main_gpu (shard
# topology belongs to the allocator, not a chat request).
_REQUEST_ALLOC_KEYS = frozenset({
    "alloc_mode", "bnb_4bit", "n_cpu_moe", "n_gpu_layers",
    "gpu_mem_gib", "cpu_mem_gib", "threads",
})


class _RelaySlot:
    """A reserved relay admission: which worker to hit, its spill, and the
    release that returns the in-flight permit. ``release()`` is idempotent."""

    __slots__ = ("worker", "spill", "_release", "_released")

    def __init__(self, worker, spill, release):
        self.worker = worker
        self.spill = spill
        self._release = release
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._release()
        except Exception:  # noqa: BLE001 — release must never raise into a finally
            logger.exception("relay slot release failed")


def _try_reserve(worker: Optional[dict], spill, model_key: str,
                 viable: Optional[Callable[[dict], bool]]) -> Optional[_RelaySlot]:
    """Reserve an in-flight relay slot on ``worker`` for ``model_key``.

    Returns a _RelaySlot on success, or None when the worker is ineligible
    (``viable`` predicate — e.g. vision capability) or already at its in-process
    cap. A slot-served model is uncapped (reserved with a no-op release).
    """
    if not worker:
        return None
    if viable is not None and not viable(worker):
        return None
    cap = _effective_cap(worker, model_key)
    if cap is None:                       # slot-served — the child schedules itself
        return _RelaySlot(worker, spill, _NOOP_RELEASE)
    wid = worker.get("id") or ""
    if _inflight_try_acquire(wid, model_key, cap):
        return _RelaySlot(worker, spill, lambda: _inflight_release(wid, model_key))
    return None


def _reserve_once(model_key: str, pool: Optional[str], primary_worker: dict,
                  primary_spill, viable: Optional[Callable[[dict], bool]],
                  task: Optional[str] = None) -> Optional[_RelaySlot]:
    """One admission pass, no wait: the primary pick first, then any other online
    worker holding the model that has room (cap-aware reroute). None if all full.
    Fast on the happy path (primary reserve is lock-only); only a reroute touches
    the candidates provider (a cached registry read). ``task`` keeps the reroute
    list task-capable (same gate as the primary pick)."""
    slot = _try_reserve(primary_worker, primary_spill, model_key, viable)
    if slot is not None:
        return slot
    primary_id = (primary_worker or {}).get("id")
    for alt in _candidates(model_key, pool, task):
        if alt.get("id") == primary_id:
            continue
        slot = _try_reserve(alt, _spill_for(alt.get("id"), model_key),
                            model_key, viable)
        if slot is not None:
            logger.info("relay reroute: %s at cap for %s -> %s (cap-aware)",
                        primary_id, model_key, alt.get("id"))
            return slot
    return None


def _busy(primary_worker: dict, model_key: str) -> "WorkerBusyError":
    return WorkerBusyError(primary_worker, model_key,
                           _inflight_count((primary_worker or {}).get("id") or "",
                                           model_key))


def _acquire_relay_slot(model_key: str, pool: Optional[str], primary_worker: dict,
                        primary_spill, *, viable: Optional[Callable[[dict], bool]] = None,
                        wait_s: Optional[float] = None,
                        task: Optional[str] = None) -> _RelaySlot:
    """SYNC cap-aware admission (tests + any synchronous caller).

    Admit one relay under the cap, rerouting to another holder or WAITING briefly
    (small blocking sleeps) for a slot to free; exhausted → WorkerBusyError. The
    caller MUST ``release()`` the returned slot when the relay (incl. the whole
    stream) finishes. Do NOT call this from the async runner path — a blocking
    sleep would stall the shared event loop; that path uses the async variant.
    See the module note for the v0 per-process honesty caveat.
    """
    if _gate_disabled():
        return _RelaySlot(primary_worker, primary_spill, _NOOP_RELEASE)
    deadline = time.monotonic() + (_gate_wait_s() if wait_s is None else wait_s)
    while True:
        slot = _reserve_once(model_key, pool, primary_worker, primary_spill, viable, task)
        if slot is not None:
            return slot
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _busy(primary_worker, model_key)
        time.sleep(min(0.1, remaining))


async def _acquire_relay_slot_async(model_key: str, pool: Optional[str],
                                    primary_worker: dict, primary_spill, *,
                                    viable: Optional[Callable[[dict], bool]] = None,
                                    wait_s: Optional[float] = None,
                                    task: Optional[str] = None) -> _RelaySlot:
    """ASYNC cap-aware admission for DelegatingRunner.run/stream.

    Identical policy to the sync variant, but the bounded wait YIELDS the shared
    event loop (``await asyncio.sleep``) rather than blocking it: central drives
    every relay on one long-lived loop thread (async_runtime), so a blocking
    sleep here would freeze the request currently HOLDING the slot — it could
    never finish and free the slot, deadlocking the wait. Yielding lets the
    holder keep generating and release, so the waiter is admitted the moment a
    slot frees (or times out honestly).
    """
    if _gate_disabled():
        return _RelaySlot(primary_worker, primary_spill, _NOOP_RELEASE)
    deadline = time.monotonic() + (_gate_wait_s() if wait_s is None else wait_s)
    while True:
        slot = _reserve_once(model_key, pool, primary_worker, primary_spill, viable, task)
        if slot is not None:
            return slot
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _busy(primary_worker, model_key)
        await asyncio.sleep(min(0.1, remaining))


# ---------------------------------------------------------------------------
# Cold-load HOLD (t36) — a call for a FEASIBLE-but-COLD model is a *presumed
# success*, not a fast failure.
#
# When a worker IS selected for a model but the model is not yet loaded and the
# on-demand load/swap trips a TRANSIENT failure (the slot child dropping the
# connection mid-swap — "RemoteProtocolError: Server disconnected", a 503 while
# it warms, a "produced no output" because it was still loading), central used
# to surface that as the caller's error while the load churned on in the
# background. The operator's rule (t36): hold the call, surface load progress,
# and dispatch the instant the model is healthy — fail ONLY when the load
# HONESTLY fails.
#
# Genuine infeasibility is unchanged and still fails FAST: no worker selected
# (the no_local_serving refusal below), or a PERMANENT load error (won't-fit /
# out-of-memory / unknown-model / a capability refusal). The distinction is:
# refusal = never could serve → fail now; cold = will serve → wait.
#
# CENTRAL-ONLY: this reads the worker's existing heartbeat load-state (loaded /
# loading / provisioning / load_reports — the 0.1.190 honest last_load_error) via
# an injected seam; no worker-side change and no new relay-wire field. The
# coalescer is per-gunicorn-process (a module set on the single async_runtime
# loop), the same v0 honesty as the relay in-flight gate above.
# ---------------------------------------------------------------------------

# Load-state seam (web -> core, optional). ``fn(model_key, worker_id, since_ts)``
# returns the worker's live view of the model:
#   {"healthy": bool,        # resident/loaded now (ready to serve)
#    "in_progress": bool,     # weights loading OR still downloading now
#    "progress": float|None,  # download fraction when provisioning
#    "message": str|None,     # human progress line
#    "error": str|None}       # a FRESH (ts>=since_ts) honest load failure
# Unset (standalone worker / bare central) ⇒ None ⇒ the hold degrades to a
# blind bounded retry (still correct, just no progress/early-honest-fail).
_load_state_provider: Optional[Callable[..., Optional[dict]]] = None


def set_load_state_provider(fn: Optional[Callable]) -> None:
    """Register the worker load-state reader (web -> core), optional."""
    global _load_state_provider
    _load_state_provider = fn
    logger.info("load-state provider registered: %s", getattr(fn, "__name__", fn))


def _load_state(model_key: str, worker_id: Optional[str],
                since_ts: float = 0.0) -> Optional[dict]:
    """Best-effort worker load-state; None when unset or on ANY failure (so it can
    never turn a held call into a crash — it is advisory to the hold loop)."""
    if _load_state_provider is None or not worker_id:
        return None
    try:
        for _args in ((model_key, worker_id, since_ts), (model_key, worker_id)):
            try:
                return _load_state_provider(*_args)
            except TypeError:
                continue
    except Exception as exc:  # noqa: BLE001 — load-state must never break a request
        logger.warning("load-state provider failed for %s: %s", model_key, exc)
    return None


def _cold_hold_enabled() -> bool:
    return os.environ.get("HUGPY_COLD_HOLD", "").strip().lower() not in (
        "off", "0", "false", "no",
    )


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _cold_hold_max_s() -> float:
    """HARD CEILING for holding a cold call — the one clock that is NOT
    progress-aware, and deliberately so: it is the guard against a worker that
    keeps 'progressing' pathologically (a re-download loop, a busy signal that
    never resolves) holding a caller forever.

    Default 900s. Raised from 300s on 2026-07-28: with the stall clock now doing
    the real work (a healthy load extends the hold as long as it demonstrably
    moves), the ceiling only has to be longer than the slowest legitimate
    cold provision+load, and 300s was well inside that for a multi-GB pull onto
    a busy box. Operator-tunable (defaults are promises)."""
    return _env_float("HUGPY_COLD_HOLD_MAX_S", 900.0)


def _cold_hold_stall_s() -> float:
    """No-forward-progress bound: if the load makes no movement (not loading, not
    loaded, no fresh progress) for this long, the hold gives up honestly. Default
    90s — mirrors the job store's honest-stall clock."""
    return _env_float("HUGPY_COLD_HOLD_STALL_S", 90.0)


def _cold_hold_poll_s() -> float:
    """Backoff between relay retries / progress emits while holding. Default 2s."""
    return _env_float("HUGPY_COLD_HOLD_POLL_S", 2.0)


def _env_int(name: str, default: int) -> int:
    """Positive-int env knob, same discipline as _env_float (garbage / <=0 ⇒
    default — a knob can misconfigure a deployment, never break it)."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(float(raw))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Cold-hold ADMISSION CAP — the blast-radius bound.
#
# Operator incident 2026-07-27: "what is happening so that the site is getting
# overwhelmed by being called by a single script?" Central stopped answering
# EVERYTHING, /health included, for over a minute at a time and only recovered on
# a unit restart.
#
# The hold above is the right behaviour for ONE cold call and a catastrophe for
# twenty-four of them. Central serves the whole site from `gunicorn --workers 3
# --threads 8` = 24 request slots TOTAL, and a held call pins one of those slots
# for up to HUGPY_COLD_HOLD_MAX_S (1500s on the live unit — 25 minutes). A script
# walking the ~66 text-generation models, or an operator clicking retry two dozen
# times, therefore parks every slot in a hold: /health, /llm/workers, the console
# and the API all share that one pool, so the site dies wholesale while nothing is
# actually broken.
#
# So the hold is CAPPED. Past N simultaneous holds a new arrival is refused
# IMMEDIATELY — an honest 503 in milliseconds — instead of queueing into a slot.
# Bounding the blast radius beats serving every request: a fast 503 is strictly
# better than a dead site, and the caller can retry into a fleet that is still up.
#
# Deliberately NOT a queue. Waiting in a slot for a permit that frees in twenty
# minutes is the outage, restated.
#
# WHAT IS COUNTED — cold holds, not requests. The cap must never refuse healthy
# traffic, so:
#   * a permit is taken OPTIMISTICALLY when a relay begins;
#   * it is released the instant the call proves WARM (the first token), so a
#     long warm generation never occupies a cold-hold permit;
#   * when no permit is free the call is refused ONLY if the worker's own
#     load-state says the model is not loaded. A model that reads HEALTHY is
#     served uncapped;
#   * with no load-state provider (standalone worker / bare central) we cannot
#     tell cold from warm, so we ADMIT. Never refuse on ignorance.
#
# A genuine slow load with a patient client is untouched: it is admitted, holds
# its permit, and runs to completion on the unchanged ceiling/stall clocks. The
# cap changes only how many of them may run AT ONCE.
#
# PER-PROCESS, like _INFLIGHT and _COLD_KICKING (the same v0 honesty): the
# counter is a module global, so with `--workers 3` the fleet-wide ceiling is
# 3 x the cap. The default below already has that multiplication done.
# ---------------------------------------------------------------------------

_HOLDS = 0
_HOLDS_LOCK = threading.Lock()


def _cold_hold_max_concurrent() -> int:
    """How many cold holds THIS process may run at once. Default 4.

    Sized against the measured deployment rather than picked round: central runs
    `gunicorn --workers 3 --threads 8`, so each process owns 8 of the site's 24
    request slots. 4 is half of one process's threads, which means

      * every gunicorn process ALWAYS keeps >=4 threads a hold can never touch,
        so /health, /llm/workers and the console stay answerable however
        saturated the load path is (that is the acceptance test), and
      * the fleet-wide ceiling is 3 x 4 = 12 of 24 slots — the operator's "half".

    Operator-tunable (defaults are promises); garbage or <=0 falls back to 4.
    """
    return _env_int("HUGPY_COLD_HOLD_MAX_CONCURRENT", 4)


def _cold_hold_retry_after_s() -> int:
    """``Retry-After`` on a capacity refusal. Default 20s — long enough that a
    retrying client cannot immediately re-saturate the cap, short enough to be a
    real instruction rather than a brush-off."""
    return _env_int("HUGPY_COLD_HOLD_RETRY_AFTER_S", 20)


class ColdHoldCapacityError(RuntimeError):
    """Central is already holding its maximum number of concurrent model loads.

    NOT a fault, and NOT a refusal of the model: the request was never started,
    so nothing is broken and nothing was lost — the same call succeeds once a
    load in flight finishes. Message discipline (four misattributing diagnostics
    in three days, and the recorded rule that a confidently wrong error is worse
    than a vague one): name the ONE thing that is true — the concurrent-load
    limit — plus what the model is actually doing, and speculate about NOTHING
    else. In particular it never says the box is too small, never says a worker
    is unhealthy, and never says the model failed.
    """

    code = "cold_load_capacity"

    def __init__(self, model_key: Optional[str], worker: Optional[dict],
                 held: int, cap: int, loading: bool,
                 retry_after_s: Optional[int] = None):
        self.model_key = model_key
        self.worker = worker or {}
        self.worker_name = (self.worker.get("name") or self.worker.get("id")
                            or "its worker")
        self.held = int(held)
        self.cap = int(cap)
        self.loading = bool(loading)
        self.retry_after_s = int(retry_after_s if retry_after_s is not None
                                 else _cold_hold_retry_after_s())
        super().__init__(self.stream_message())

    def stream_message(self) -> str:
        state = ("is still loading on" if self.loading
                 else "is not loaded yet on")
        return (f"cold_load_capacity: '{self.model_key}' {state} "
                f"'{self.worker_name}', and central is already holding "
                f"{self.held} concurrent model loads — its limit "
                f"({self.cap} per server process). Nothing is broken and this "
                f"request was not started: it is refused straight away rather "
                f"than queued, so the console and health checks stay "
                f"responsive. Retry in about {self.retry_after_s}s, or wait for "
                f"a load already in flight to finish.")

    def as_error(self) -> Dict[str, Any]:
        return {"error": {
            "code": self.code,
            "message": self.stream_message(),
            "model": self.model_key,
            "worker": self.worker_name,
            "worker_id": self.worker.get("id"),
            "holds_in_flight": self.held,
            "limit": self.cap,
            "retry_after_s": self.retry_after_s,
        }}


class _HoldPermit:
    """One cold-hold admission. ``release()`` is idempotent (mirrors _RelaySlot)."""

    __slots__ = ("_released",)

    def __init__(self):
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        _hold_release()


def _hold_try_acquire() -> Optional[_HoldPermit]:
    """Take a cold-hold permit if one is free, else None. Never blocks."""
    global _HOLDS
    cap = _cold_hold_max_concurrent()
    with _HOLDS_LOCK:
        if _HOLDS >= cap:
            return None
        _HOLDS += 1
    return _HoldPermit()


def _hold_release() -> None:
    global _HOLDS
    with _HOLDS_LOCK:
        _HOLDS = max(0, _HOLDS - 1)


def _hold_count() -> int:
    with _HOLDS_LOCK:
        return _HOLDS


def _admit_cold_hold(model_key: str, worker: Optional[dict],
                     since_ts: float) -> Optional[_HoldPermit]:
    """Admission for a call that may become a cold hold.

    Returns a permit (admitted, counted), or None (admitted UNCOUNTED — the model
    is warm, or we cannot tell). Raises ColdHoldCapacityError when the cap is full
    AND the worker affirmatively reports the model is not loaded — the fast,
    honest refusal that keeps the rest of the site answerable.
    """
    permit = _hold_try_acquire()
    if permit is not None:
        return permit
    ls = _load_state(model_key, (worker or {}).get("id"), since_ts)
    if ls is None:
        return None          # can't tell cold from warm ⇒ never refuse on ignorance
    if ls.get("healthy"):
        return None          # warm: this gate is about LOADS, not about traffic
    raise ColdHoldCapacityError(model_key, worker, _hold_count(),
                                _cold_hold_max_concurrent(),
                                bool(ls.get("in_progress")))


# A load error that is HONEST/PERMANENT — a refusal or a hard load failure that a
# retry cannot fix. These fail the hold immediately (honest refusal preserved).
# Everything ELSE that fails before a token is treated as transient (hold+retry),
# bounded by the stall/ceiling clocks — "predispositioned success until the model
# actually fails to load".
_PERMANENT_LOAD_MARKERS = (
    "won't fit", "wont fit", "won’t fit", "loadrefusal", "budgetrefusal",
    "insufficient storage", "out of memory", "cuda error", "cublas",
    "no capable worker", "no registered worker", "no worker is available",
    "no worker available", "local serving disabled", "hugpy_no_local_serving",
    "unknown model", "vision model loaded in-process", "could not fetch model",
    "not found on central", "unresolvable",
    # Provisioning gave a NAMED cause (2026-07-28). The worker used to flatten
    # every provisioning failure into "could not fetch model X from central or
    # HF"; it now says what actually happened ("could not provision X: disk
    # full (ENOSPC) on /mnt/storage — 0 B free of 938 GB"). That is still the
    # same permanent condition, and it must keep failing fast — retrying a
    # request against a 100%-full drive is exactly the storm this list exists
    # to prevent.
    "could not provision",
    # Operator model BLOCK: a distinct, permanent operator refusal — never held
    # or retried (see comms.blocklist.BLOCKED_MARKER; this string mirrors it).
    "blocked from the serving pool",
    # Per-request worker pin that cannot bind (_resolve_requested_worker): the
    # named box is offline / not holding the model. Retrying cannot conjure it;
    # honest fast failure keeps A/B data truthful.
    "requested worker",
    # STRUCTURALLY INVALID MODEL FILE (incident 2026-07-29). The weights on disk
    # do not match the metadata that describes them, so the loader rejects the
    # file itself — deterministically, in well under a second, identically on
    # every worker and every attempt. Found via the studio-aptitude sweep:
    #   check_tensor_dims: tensor 'token_embd.weight' has wrong shape;
    #   expected 4096, 32005, got 384, 32000
    # which reaches central as "Failed to load model from file: <path>".
    #
    # Absent from this list, that verdict was classified TRANSIENT, so the
    # cold-hold held the call while the worker burned 3 slot attempts + a 120s
    # backoff + an in-process fallback that failed the same way — and central
    # re-requested every ~3s. The caller waited 900s for an answer that was
    # already known at t+0.75s. A corrupt file is not a stall: no retry can
    # reshape a tensor, so fail fast and say so.
    "failed to load model from file", "error loading model",
    "check_tensor_dims", "wrong shape", "missing tensor",
    "unknown model architecture", "unsupported model architecture",
    # NOT-A-LOADABLE-MODEL DIR (incident 2026-07-29). A bare LoRA-adapter
    # directory (no ``model_type`` in config.json) fails transformers'
    # AutoConfig in <1s, deterministically, on every attempt:
    #   ValueError: Unrecognized model in …/qwen3.5-test-stage1-lora. Should
    #   have a `model_type` key in its config.json
    # Classified transient, the hold re-posted it to the worker every ~2.5s
    # for over half an hour (the storm this list exists to prevent). No retry
    # can conjure a base model into an adapter dir.
    "unrecognized model", "model_type",
    # RESPONSE-ENVELOPE FAILURE (incident 2026-07-29). The worker FINISHED the
    # work (ComfyUI generated the image) and then failed to serialize the
    # result ("TypeError: Object of type GeneratedImage is not JSON
    # serializable"). Deterministic per attempt — every retry re-runs a whole
    # generation and dies at the same jsonify — so central held one request
    # for 25 minutes of wasted GPU. A serialization bug is code, not load
    # state: fail fast and name it. (Worker-side fix: agent._jsonable.)
    "not json serializable",
    # MISSING PYTHON DEPENDENCY IN THE WORKER VENV (incident 2026-07-29).
    #   ValueError: Using a `device_map` … requires `accelerate`. You can
    #   install it with `pip install accelerate`
    # Deterministic on every attempt until someone pip-installs the package
    # (the /ops/pip relay exists for exactly that) — holding 92s per request
    # and then hedging "stalled or too large" buried the one actionable line.
    # "no module named" is the same class from the import side.
    "requires `accelerate`", "pip install accelerate", "no module named",
    # WEIGHTS/CONFIG MISMATCH (2026-07-29, Surogate-3.5-2B on ae): transformers
    # raises when checkpoint tensor shapes disagree with the model's config —
    # a broken/partial snapshot or wrong config.json. Deterministic per
    # attempt, the transformers twin of the gguf check_tensor_dims class above.
    "ignore_mismatched_sizes", "size mismatch for",
    # The SLOT path's wording for the same condition: slot_agent now separates a
    # child that DIED (the loader rejected the file) from one that hung, and says
    # "hard load failure" for the former. Previously both read "did not become
    # healthy (stall/hard-cap)" — a stall, i.e. transient — which is precisely
    # how a permanently-broken model kept earning retries.
    "hard load failure",
)


def _is_permanent_load_error(err: Any) -> bool:
    low = str(getattr(err, "message", None) or err or "").lower()
    return any(m in low for m in _PERMANENT_LOAD_MARKERS)


# PERMANENT-FOR-THIS-ATTEMPT vs DETERMINISTIC-UNTIL-REPAIRED. Everything above is
# permanent in the sense the hold cares about: retrying *this* attempt cannot fix
# it, so fail fast (unchanged). But only a subset is permanent in the sense the
# load-verdict CACHE cares about — i.e. still true for the NEXT request.
#
# These markers are STATE-dependent: nothing needs repairing for them to stop
# being true. VRAM frees the moment something is evicted (⭐ eviction is never
# time-vetoed — freshness is rank, not veto); disk frees when the reaper runs; a
# missing model appears when its download lands; a pinned/absent worker comes
# back on its next heartbeat; an operator BLOCK must lift the instant it is
# lifted, not one TTL later. Caching any of those would make central refuse a
# request it could now serve — a timer standing in for a measurement, which the
# operator's residency doctrine rules wrong by default.
#
# So: still an honest FAST refusal, never a cached one. What DOES get cached is
# the deterministic-until-repaired class the cache was built for — a corrupt or
# mismatched weight file, a not-a-model directory, an unknown architecture, a
# missing worker-venv dependency, a serialization bug: conditions that require a
# human/pip/file fix and will fail identically on every attempt until they get it.
_STATE_DEPENDENT_LOAD_MARKERS = (
    # capacity / placement — freed by eviction or a smaller footprint
    "won't fit", "wont fit", "won’t fit", "loadrefusal", "budgetrefusal",
    "out of memory", "insufficient storage",
    # inventory / provisioning — resolved by a download or freed disk
    "could not provision", "could not fetch model", "not found on central",
    # routing / availability — resolved by a heartbeat or a policy flag
    "no capable worker", "no registered worker", "no worker is available",
    "no worker available", "requested worker",
    "local serving disabled", "hugpy_no_local_serving",
    # operator BLOCK — unblocking must take effect immediately, never after a TTL
    "blocked from the serving pool",
)


def _is_cacheable_load_verdict(err: Any) -> bool:
    """True only for a permanent load failure that is ALSO deterministic until
    somebody repairs something — the only class it is honest to answer a LATER
    request from without re-attempting. See _STATE_DEPENDENT_LOAD_MARKERS."""
    low = str(getattr(err, "message", None) or err or "").lower()
    if not low.strip():
        return False
    if not any(m in low for m in _PERMANENT_LOAD_MARKERS):
        return False
    return not any(m in low for m in _STATE_DEPENDENT_LOAD_MARKERS)


# A REQUEST-SHAPE failure — the messages the caller sent are malformed for THIS
# model's chat template (strict user/assistant alternation, an unsupported role,
# a template that demands the last turn be a user turn, ...). It is NOT a load
# problem and NOT a capacity problem, so:
#   * it must never be held/retried — the identical payload fails identically,
#     and holding it for the full cold-hold ceiling is what made a single
#     malformed call hang until the client gave up (HTTP 000, operator report
#     2026-07-27), and
#   * the surfaced message must NEVER speculate about the box being too small.
#     "the model may be too large for the box" was asserted for ANY late failure
#     and cost hours of VRAM investigation on a request-shape bug. A confidently
#     wrong diagnostic is worse than a vague one.
_REQUEST_SHAPE_MARKERS = (
    "templateerror", "template error", "jinja2", "jinja",
    "roles must alternate", "must alternate",
    "chat template",
    "only user and assistant roles are supported",
    "conversation roles must",
)


def _is_request_shape_error(err: Any) -> bool:
    low = str(getattr(err, "message", None) or err or "").lower()
    return any(m in low for m in _REQUEST_SHAPE_MARKERS)


def _request_shape_message(model_key: str, worker: Optional[dict],
                           err: Any) -> str:
    """The honest line for a malformed-request failure.

    Names the actual fault class (request shape / chat template) and the raw
    engine error, and says nothing about box size or load stalls — none of which
    are implicated.
    """
    wname = (worker or {}).get("name") or (worker or {}).get("id") or "worker"
    raw = str(getattr(err, "message", None) or err or "").strip()
    detail = f": {raw}" if raw else ""
    return (f"'{model_key}' on '{wname}' rejected the REQUEST SHAPE{detail} — "
            f"this model's chat template will not render the message sequence "
            f"that was sent (it requires the roles in a particular order, e.g. "
            f"strict user/assistant alternation after an optional system "
            f"message). This is a malformed request, not a capacity or load "
            f"problem: the model is loaded and resending the same messages will "
            f"fail identically. Fix the message list and send again.")


def _blocked_reason(model_key: Optional[str]) -> Optional[str]:
    """Operator BLOCK gate: the honest refusal when ``model_key`` is blocked from
    the serving pool, else None. Block is an operator override that outranks BOTH
    routing selection AND pin — a blocked model is never resolved to a worker AND
    never served locally, so this sits at the TOP of run()/stream(), ahead of
    selection and the local-serving policy. Best-effort (the blocklist lives in
    the stdlib-only comms package); any failure ⇒ None so the gate can never take
    serving down."""
    try:
        from ...comms.blocklist import block_reason
        return block_reason(model_key)
    except Exception:  # noqa: BLE001 — a block read must never break a request
        return None


class _ColdRetry(Exception):
    """A transient pre-token relay failure — the model is (probably) still
    loading/swapping. Caught by the hold loop, which waits and retries."""
    def __init__(self, message: str):
        self.message = str(message or "")
        super().__init__(self.message)


class _LoadFailed(Exception):
    """An HONEST pre-token load failure — surfaced to the caller, no retry."""
    def __init__(self, message: str):
        self.message = str(message or "")
        super().__init__(self.message)


class _RelayUnbuildable(Exception):
    """The relay payload could not be built (oversized inline file) or the
    operator opted into HUGPY_LOCAL_FALLBACK — fall through to local, exactly as
    before the hold existed. Never a load problem, never held."""


# Coalescer: at most ONE cold on-demand load-kick per (worker_id, model_key) in
# flight at a time, so N concurrent calls for the same cold model don't each fire
# a separate on-demand load (the thundering herd). Correct without a lock: every
# holder runs on the single async_runtime loop, so the check-and-add below is
# atomic (no await between them). Per-process, like the relay in-flight gate.
_COLD_KICKING: set = set()

# ---------------------------------------------------------------------------
# DISPATCH QUEUE, half 2 of 2 — the shared VERDICT (operator, 2026-07-29:
# "queues, please, implement a queue"). _COLD_KICKING above is the queue's
# admission half: one attempt drives, everyone else waits in line. This is the
# outcome half: when the driving attempt fails PERMANENTLY, that verdict is
# recorded here so every queued waiter — and every re-submit of the same
# request arriving for the next TTL window — fails fast from the cache instead
# of launching its own doomed attempt against the worker. Without this, a
# client that re-submits after each failure restarts the whole hold from
# scratch and the fleet sees one attempt every poll interval, indefinitely
# (the qwen3.5-test-stage1-lora storm: ~24 attempts/minute for 30+ minutes).
#
# TTL-bounded, never sticky forever: a repaired model (file fixed, base model
# wired, worker updated) serves again one TTL after its last failure. A
# SUCCESSFUL serve clears the verdict immediately. Per-process, like
# _COLD_KICKING and the admission cap (same v0 honesty).
# ---------------------------------------------------------------------------

_LOAD_VERDICTS: Dict[tuple, tuple] = {}   # (worker_id, model_key) -> (expires_ts, message)
_LOAD_VERDICTS_LOCK = threading.Lock()


def _load_verdict_ttl_s() -> float:
    """How long a permanent load failure is answered from the cache. Default
    120s — long enough to absorb a re-submitting client/bench, short enough
    that a genuine fix (file repaired, worker restarted) is picked up without
    operator action. Operator-tunable (defaults are promises)."""
    return _env_float("HUGPY_LOAD_VERDICT_TTL_S", 120.0)


def _record_load_verdict(worker_id, model_key: str, message: str) -> None:
    """Record a verdict, but ONLY for the deterministic-until-repaired class.

    The gate lives here, not at the call sites, so no present or future caller
    can poison the cache by forgetting it. Two things must never be cached:
      * a STATE-dependent refusal (won't fit / no worker / disk full / blocked) —
        it stops being true the moment residency or the fleet changes, so a
        cached copy would refuse a request central could now serve;
      * a CANCEL — a user pulling out is not a load failure. Cancels reach this
        function only as an empty/absent message (the hold loop returns before
        recording), and an empty message is refused below, so one cancelled
        attempt can never mark a model unservable for the whole TTL.
    Both still fail FAST where they are raised; they just leave nothing behind.
    """
    if not _is_cacheable_load_verdict(message):
        return
    with _LOAD_VERDICTS_LOCK:
        _LOAD_VERDICTS[(worker_id or "", model_key)] = (
            time.time() + _load_verdict_ttl_s(), str(message or ""))


def _clear_load_verdict(worker_id, model_key: str) -> None:
    with _LOAD_VERDICTS_LOCK:
        _LOAD_VERDICTS.pop((worker_id or "", model_key), None)


def _active_load_verdict(worker_id, model_key: str) -> "Optional[str]":
    """The cached permanent-failure message for (worker, model), or None.
    Expired entries are dropped on read (self-cleaning; the dict only ever
    holds keys that failed within the last TTL, so it stays tiny)."""
    key = (worker_id or "", model_key)
    with _LOAD_VERDICTS_LOCK:
        entry = _LOAD_VERDICTS.get(key)
        if not entry:
            return None
        expires, message = entry
        if time.time() >= expires:
            _LOAD_VERDICTS.pop(key, None)
            return None
        return message


def _verdict_message(model_key: str, worker: "Optional[dict]", cached: str) -> str:
    wname = (worker or {}).get("name") or (worker or {}).get("id") or "worker"
    return (f"'{model_key}' on '{wname}' failed to load moments ago and the "
            f"failure is permanent (retrying cannot fix it): {cached} — "
            f"answered from the load-verdict cache without re-attempting; "
            f"the verdict expires {int(_load_verdict_ttl_s())}s after the "
            f"failure, sooner if the model serves successfully elsewhere.")


def _retry_backoff_next(current_s: float) -> float:
    """Exponential pacing for hold-loop retries that are making NO progress:
    double up to a cap (default 30s). While a load reports genuine forward
    progress the caller keeps the base poll instead — a loading model deserves
    tight polling; a failing one does not."""
    cap = _env_float("HUGPY_COLD_HOLD_BACKOFF_MAX_S", 30.0)
    return min(max(current_s, _cold_hold_poll_s()) * 2.0, cap)


def _loading_status(request_id: str, model_key: str, worker: Optional[dict],
                    progress: Optional[float], message: Optional[str]) -> "StatusEvent":
    """A held call's progress event. Reuses the SAME wire shape the browser
    already renders for provisioning (``type:"status"`` + message/stage/progress
    — ChatPanel shows ``⏳ {message}{pct}``), so nothing new is invented. ``stage``
    is ``awaiting-load`` so /llm/jobs can show the hold distinctly."""
    wname = (worker or {}).get("name") or (worker or {}).get("id") or "worker"
    msg = message or f"loading {model_key} on {wname}…"
    ev = StatusEvent(request_id=request_id, stage="awaiting-load", message=msg)
    if progress is not None:
        try:
            ev.progress = round(float(progress), 4)
        except (TypeError, ValueError):
            pass
    return ev


def _cold_timeout_message(model_key: str, worker: Optional[dict],
                          last_err: str,
                          last_progress: Optional[str] = None,
                          stalled_for: Optional[float] = None,
                          ceiling: bool = False) -> str:
    """The honest give-up line when a held load never became ready in time.

    The size/stall speculation is only ever appended when the last error is
    actually consistent with a LOAD problem. A request-shape failure that
    somehow reached the ceiling reports itself as what it is — the wrapper must
    never re-label another fault class as a capacity problem.

    ``last_progress`` carries the LAST OBSERVED progress line so the message can
    say WHERE it stopped ("stalled at 12.1 GB after 90s without progress")
    rather than only that it stopped. The operator's 2026-07-28 report of this
    error had no numbers in it at all, which is why it read as an arbitrary
    timeout on a healthy load — which is exactly what it was.
    """
    if last_err and _is_request_shape_error(last_err):
        return _request_shape_message(model_key, worker, last_err)
    wname = (worker or {}).get("name") or (worker or {}).get("id") or "worker"
    if ceiling:
        tail = f" (last: {last_err})" if last_err else ""
        return (f"'{model_key}' did not finish loading on '{wname}' in time"
                f"{tail} — the hold hit its hard ceiling — it kept reporting "
                f"progress but never became ready; try again or assign it "
                f"elsewhere.")
    # SPECIFICITY DISCIPLINE (operator, 2026-07-29: "why is it unsure of what
    # the actual problem was? … this needs to be specific"). When the worker
    # NAMED an error, that error IS the diagnosis — repeating it inside a
    # "stalled, or too large" menu re-labels a known fault as two guesses,
    # both usually wrong (the trigger was a missing `accelerate` dependency
    # reported verbatim and then hedged into a size problem). Speculate ONLY
    # when nothing at all was observed, and say that that's the situation.
    if last_err:
        for_s = (f" after {int(stalled_for)}s with no forward progress"
                 if stalled_for is not None else "")
        # The named error IS the diagnosis (specificity discipline) — but the
        # last observed progress still says WHERE it stopped, and losing it
        # regressed the 2026-07-28 "no numbers at all" report. Both ride.
        at = f" (last observed progress: {last_progress})" if last_progress else ""
        return (f"'{model_key}' failed to become ready on '{wname}'{for_s}{at}. "
                f"The worker's last reported error is the cause: {last_err}")
    where = f" at {last_progress}" if last_progress else ""
    dur = f" for {int(stalled_for)}s" if stalled_for is not None else ""
    return (f"'{model_key}' made no forward progress{where}{dur} on '{wname}' "
            f"and the worker reported no error — the load went silent. Check "
            f"the worker's own logs for the cause (OOM kills and hung IO die "
            f"without reporting); try again or assign it elsewhere.")


# A worker answering "busy" is a worker that is DEMONSTRABLY ALIVE AND WORKING.
#
# This is the second half of the 2026-07-28 incident. ae's per-model gen-gate
# did exactly the right thing — it held the request and returned a structured
# 503 ModelBusy while the weights loaded — and central read that correct
# behaviour as silence, let the 90s stall clock run out, and killed the caller.
# ae finished the load moments later and served the model.
#
# A structured busy/503 from the box we are holding on is therefore FORWARD
# PROGRESS for stall purposes: the worker is up, it recognized the model, and it
# is serializing us behind its own load. It is deliberately NOT unbounded — the
# hard ceiling (HUGPY_COLD_HOLD_MAX_S) is what stops a box that answers "busy"
# forever, which is precisely the pathological-progress case the ceiling exists
# for.
_BUSY_MARKERS = (
    "model_busy", "modelbusy", "is busy:",
    "503", "service unavailable",
    "already in the in-process runner",
)


class _WorkerHTTPError(Exception):
    """A non-2xx from a worker, WITH the worker's own error envelope intact.

    The agent ships every failure AS DATA next to the status code — ModelBusy
    returns 503 with {"ok":false,"error":{"code":"model_busy",...}}, a
    BudgetRefusal returns 507 with {"refused":{...}}, a provisioning failure
    returns its named cause. ``_worker_run_once`` learned to read that body in
    2026 ("a bare raise_for_status() discarded the body and reduced the console
    to 'Server error 500' with no cause"); the STREAM path never did, and kept
    calling bare ``raise_for_status()``.

    That omission is the entry-path defect (operator retest 2026-07-28). With
    the body thrown away, central sees only the opaque httpx string
    ``Server error '503 SERVICE UNAVAILABLE' for url '…/infer/stream'`` and:

      * cannot tell "gen-gate is holding me while the model loads" (hold it)
        from "this box is saturated" from "provisioning failed" (don't) — it was
        matching on the literal text "503", which works only by accident of what
        httpx happens to put in the string;
      * cannot see a genuinely PERMANENT body either — a 507 BudgetRefusal
        arrives as "Client error '507 …'", matches no permanent marker, and gets
        HELD for the full ceiling instead of failing fast. Both directions wrong
        from the same missing parse.

    So the status code AND the parsed body travel together, and classification
    reads both.
    """

    def __init__(self, status: int, body: Any, url: str = ""):
        self.status = int(status)
        self.body = body if isinstance(body, dict) else {}
        err = self.body.get("error")
        if isinstance(err, dict):
            self.code = str(err.get("code") or "")
            self.message = str(err.get("message") or "")
        else:
            self.code = ""
            self.message = str(err or "")
        if not self.message:
            self.message = f"HTTP {self.status} from {url or 'worker'}"
        super().__init__(f"HTTP {self.status}: {self.message}")


# Worker error CODES that mean "this box is working on it, hold" — read from the
# structured envelope rather than sniffed out of prose.
_BUSY_CODES = frozenset({"model_busy", "provisioning", "loading", "warming_up"})


def _is_worker_busy_signal(err: Any) -> bool:
    """Is this failure the worker WORKING (hold) rather than the worker BROKEN?

    Three sources, most authoritative first: the structured error code, the HTTP
    status, then — for transports that lost both — the prose markers.
    """
    if isinstance(err, _WorkerHTTPError):
        if err.code in _BUSY_CODES:
            return True
        # 503 is the agent's "temporarily can't, try shortly" code and is the
        # ONLY status it uses for a gate hold; a 507/5xx-other is a verdict.
        if err.status == 503 and not _is_permanent_load_error(err.message):
            return True
        return False
    low = str(getattr(err, "message", None) or err or "").lower()
    if not low:
        return False
    if _is_permanent_load_error(low) or _is_request_shape_error(low):
        return False          # a named permanent fault is never "still working"
    return any(m in low for m in _BUSY_MARKERS)


def _cold_progress(model_key: str, worker: Optional[dict],
                   since_ts: float) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """Consult worker load-state → (moved, progress, message, honest_error).

    ``moved`` is True when the worker reports the model healthy or actively
    loading/provisioning (forward progress — resets the stall clock). ``honest_error``
    is a FRESH permanent load failure (fail the hold) or None."""
    ls = _load_state(model_key, (worker or {}).get("id"), since_ts)
    if not ls:
        return False, None, None, None
    err = ls.get("error")
    if err and _is_permanent_load_error(err):
        return True, ls.get("progress"), ls.get("message"), str(err)
    moved = bool(ls.get("healthy") or ls.get("in_progress"))
    return moved, ls.get("progress"), ls.get("message"), None


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


def _inline_reference_images(payload: dict) -> bool:
    """Inline id_lock reference stills the worker's comfy (127.0.0.1) can't see.

    Reads each ``reference_images`` path -> base64 into ``reference_images_b64``
    and DROPS the unreachable paths. This is the LIST analogue of _inline_file:
    the single-file _PATH_KEYS inliner + the worker's _materialize_file handle
    exactly one path, and a multi-file rematerializer on the worker is out of
    this slice's agent.py scope — so the reference bytes ride a request FIELD
    (ImageGenRequest.reference_images_b64, like VisionAnalysisRequest.image_b64)
    instead. Returns False (→ run local) if any reference is missing or too big;
    True when there was nothing to inline or inlining succeeded."""
    refs = payload.get("reference_images")
    if not refs:
        return True
    b64s: list[str] = []
    for p in refs:
        try:
            if not os.path.isfile(p) or os.path.getsize(p) > _MAX_WORKER_FILE_BYTES:
                return False
            with open(p, "rb") as fh:
                b64s.append(base64.b64encode(fh.read()).decode("ascii"))
        except OSError:
            return False
    payload["reference_images_b64"] = b64s
    payload.pop("reference_images", None)     # paths the worker can't reach
    return True


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
    payload: Dict[str, Any] = {"_force_local": True, **req.model_dump()}
    # Per-REQUEST alloc triggers (operator ask 2026-07-29). ALWAYS pop the key —
    # even when unset it dumps as alloc=None, and released workers run
    # extra="forbid": an unknown key on the wire rejects ALL relayed chat (the
    # 2026-07-17 None-key incident, same landmine). The whitelisted remainder is
    # merged OVER the assignment spill below — a per-call trigger wins for this
    # call, everything not overridden keeps the designation's value.
    _req_alloc = payload.pop("alloc", None)
    if isinstance(_req_alloc, dict):
        _req_alloc = {k: v for k, v in _req_alloc.items()
                      if k in _REQUEST_ALLOC_KEYS and v is not None}
    else:
        _req_alloc = {}
    # A request type may carry its OWN `task` field (TranscribeRequest.task is
    # whisper's transcribe/translate MODE) — dumped last, it clobbered the
    # DISPATCH task key and every whisper offload died on the worker with
    # "Unknown task='transcribe'". Keep the domain field under its builder
    # alias and let the dispatch key own `task`.
    if payload.get("task") not in (None, task):
        payload["whisper_task"] = payload.pop("task")
    payload["task"] = task
    spill = spill_override if spill_override is not None else _spill_for(worker_id, model_key)
    if _req_alloc:
        spill = {**(spill or {}), **_req_alloc}
        logger.info("per-request alloc override for %s on %s: %s",
                    model_key, worker_id, _req_alloc)
    if spill:
        payload["spill"] = spill
    if not _inline_file(payload):
        return None
    if not _inline_reference_images(payload):     # id_lock references (list)
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
                # Token accounting from a worker on a build that reports it;
                # absent (old workers) -> None, same as before.
                usage=d.get("usage") if isinstance(d.get("usage"), dict) else None,
                # The engine's own decode rate, same additive contract as usage:
                # present from a worker that reports it, None from one that
                # doesn't. RECORDING ONLY — see DoneEvent.timings.
                timings=d.get("timings") if isinstance(d.get("timings"), dict) else None,
            )
        except Exception:
            # ⚠ MIXED-VERSION SAFETY. This reconstruction is EXPLICIT-FIELD (it
            # never splats `d`), so an unknown key a NEWER worker adds cannot
            # reach DoneEvent's extra="forbid" and cannot fail validation here.
            # That matters: the fallback below downgrades to a StatusEvent, and
            # a stream whose terminal `done` silently becomes a status event has
            # NO done at all — the consumer waits forever or finishes without a
            # finish_reason. Keep this constructor field-explicit; never
            # "simplify" it to DoneEvent(**d).
            return StatusEvent(**{**d, "request_id": request_id})
    if t == "error":
        return ErrorEvent(request_id=request_id, message=d.get("message", "worker error"))
    return StatusEvent(**{**d, "request_id": d.get("request_id", request_id)})


async def _worker_stream(worker: dict, payload: dict, request_id: str):
    """Relay a worker's POST /infer/stream SSE as StreamEvents.

    Raising before the first event lets the caller fall back to local; a short
    connect timeout makes a dead worker fail over fast, a long read timeout
    leaves room for generation.

    k59: the timeouts now come from the sanctioned client (call class "relay":
    short connect, a 600 s SILENCE budget between chunks) and the call is
    breaker-gated. This is the single most expensive call central makes — it
    holds a gunicorn thread for the whole generation — so a worker that has
    stopped answering must fail over to local INSTANTLY rather than after
    another connect timeout per attempt.
    """
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import (
        worker_http)

    key = worker_http.breaker_key(worker)
    worker_http.guard(key, url=worker_http.base_url(worker))
    url = worker["url"].rstrip("/") + "/infer/stream"
    try:
        client_cm = worker_http.async_client("relay")
    except worker_http.TRANSPORT_ERRORS as exc:  # pragma: no cover — construction
        worker_http.note_failure(key, exc)
        raise
    try:
        async with client_cm as client:
            async with client.stream("POST", url, json=payload) as resp:
                # The head arrived: whatever the status, this box is REACHABLE.
                # A 503 gen-gate hold is not a breaker event.
                worker_http.note_ok(key)
                if resp.status_code >= 400:
                    # Read the worker's own error envelope before discarding the
                    # response — the streaming twin of the parse _worker_run_once
                    # already does. Without it a gen-gate hold (503 model_busy) and
                    # a capacity verdict (507 refused) are the same opaque string to
                    # the caller, and the cold-hold cannot tell "hold this" from
                    # "fail this". See _WorkerHTTPError.
                    body = None
                    try:
                        await resp.aread()
                        body = resp.json()
                    except Exception:  # noqa: BLE001 — a bodyless 5xx is still a 5xx
                        body = None
                    raise _WorkerHTTPError(resp.status_code, body, url)
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
                    if ev is None:  # suppressed (worker's inner dispatch banner)
                        continue
                    yield ev
    except worker_http.TRANSPORT_ERRORS as exc:
        # Only a TRANSPORT failure counts against the box. A mid-stream read
        # timeout does too: a relay that has gone silent past the budget is
        # indistinguishable from a hung box, and that is exactly the state the
        # breaker exists to stop re-entering.
        worker_http.note_failure(key, exc)
        raise


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
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import (
        worker_http)

    url = worker["url"].rstrip("/") + "/infer"
    # Same discipline as _worker_stream: short connect, long read (call class
    # "relay_long" — the whole generation arrives as one body), breaker-gated.
    with worker_http.breaker_scope(worker):
        async with worker_http.async_client("relay_long") as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                # The agent ships failures AS DATA (ok:false + error + traceback
                # tail) alongside the 4xx/5xx status; a bare raise_for_status()
                # discarded that body and reduced the console to "Server error
                # '500 …'" with no cause. Surface the worker's own reason — the
                # caller (DelegatingRunner) stamps the worker name onto it.
                # Same envelope, same class as the streaming path — so the one-shot
                # runner's hold classifies a 503 gen-gate hold exactly as stream()
                # does instead of matching on prose.
                body = None
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                if body is not None or resp.status_code >= 500:
                    raise _WorkerHTTPError(resp.status_code, body, url)
                resp.raise_for_status()
            data = resp.json()
    if isinstance(data, dict):
        data.setdefault("request_id", request_id)
        data.setdefault("model_key", model_key)
        data.setdefault("ok", True)
    return _stamp_worker_error(result_type.model_validate(data), worker)


def _stamp_worker_error(result, worker: dict):
    """Attribution at the source for errors-as-data.

    A worker that fails with a TYPED {ok: false, error: …} result (HTTP 200 —
    the dispatch plane's errors-as-data contract) used to flow through the
    relay anonymously, so the console showed raw cause frames ("frame 0:
    ModuleNotFoundError: No module named 'torch'") with no hint of WHICH box
    blew up (2026-07-05: ae's torch-less venv). Prefix worker name+id onto the
    error text so every downstream surface (chat, scene frames, job errors)
    carries the attribution."""
    err = getattr(result, "error", None)
    if getattr(result, "ok", True) and not err:
        return result                          # success — nothing to stamp
    if not isinstance(err, str) or not err or err.startswith("on worker "):
        return result                          # nothing stampable / already stamped
    wname = worker.get("name") or worker.get("id") or "worker"
    wid = worker.get("id") or ""
    label = f"{wname} ({wid})" if wid and wid != wname else wname
    stamped = f"on worker {label}: {err}"
    try:
        return result.model_copy(update={"error": stamped})
    except Exception:  # noqa: BLE001 — attribution must never break a result
        try:
            result.error = stamped
        except Exception:  # noqa: BLE001
            pass
        return result


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
            from abstract_hugpy_dev.flask_app.app.functions.imports.utils import (
                worker_http)
            payload = {"delegated": True, "task": task, **req.model_dump()}
            # Same wire scrub as _relay_payload: a peer on a released build
            # forbids unknown keys, and alloc dumps even when None.
            payload.pop("alloc", None)
            url = peer.base_url.rstrip("/") + "/api/llm/execute"
            # A peer is another central, not a worker, but the call shape is
            # identical (long relay over the LAN) so it takes the same short
            # connect + breaker discipline, keyed on the peer's base url.
            with worker_http.breaker_scope(peer.base_url):
                async with worker_http.async_client(
                        "relay_long",
                        read_timeout=float(self.cfg.timeout_s or 3600)) as client:
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


def _humanize_worker_error(wname: str, raw: str) -> str:
    """Turn a raw worker/transport error into a clean, user-safe line.

    A worker's llama-server failure arrives as an httpx status string that leaks
    the worker's INTERNAL loopback URL and HTTP plumbing (e.g. ``Client error
    '400 BAD REQUEST' for url 'http://127.0.0.1:8101/v1/chat/completions'``) —
    noise to a user and an internal-topology leak. This was the /media "raw
    HTML/CSS throws an error" report (2026-07-13): pasting code that OVERFLOWS
    the model's context (code tokenizes ~2x denser than prose, so a modest paste
    overflows) made the slot return 400, which was surfaced verbatim. Translate
    the two common statuses and strip internal URLs for everything else.
    """
    import re
    msg = str(raw or "").strip()
    low = msg.lower()
    if "400" in low and ("bad request" in low or "for url" in low):
        return ("This request was rejected by the model server — most often "
                "because the message is too long for the model's context window. "
                "Code and markup use roughly twice the tokens of plain prose, so "
                "even a modest paste can overflow. Try shortening it, or use a "
                "larger-context model.")
    if "503" in low or "service unavailable" in low:
        return (f"The '{wname}' worker is still loading this model — give it a few "
                f"seconds and send again.")
    # generic: strip the internal loopback URL (and any wrapping quotes) + the
    # MDN hint tail, keep the gist
    msg = re.sub(r"(?:for url\s*)?['\"]?https?://127\.0\.0\.1:\d+\S*?['\"]?(?=\s|$)",
                 "the model server", msg)
    msg = re.sub(r"\s*For more information check:.*$", "", msg, flags=re.S).strip()
    return f"The '{wname}' worker could not complete this request: {msg}".strip()


def _worker_vision_capable(worker: Optional[dict]) -> bool:
    """True only when the worker AFFIRMATIVELY reports its llama.cpp build can run
    vision (mtmd) — engine.supports_vision. Central does not guess: it trusts what
    the worker says about itself. A worker that doesn't advertise it (older agent)
    or reports it can't is treated as NOT vision-capable, so an image turn never
    lands on a server that would ignore the image and answer from text alone."""
    eng = (worker or {}).get("engine") or {}
    return bool(eng.get("supports_vision"))


def _worker_comfy_id_lock_capable(worker: Optional[dict]) -> bool:
    """True only when the worker's ComfyUI AFFIRMATIVELY advertises the IPAdapter
    node pack (comfy.available AND comfy.id_lock). The remote-side twin of
    workers._comfy_id_lock_capable — used as the relay-reroute ``viable`` filter
    so an identity-locked STILL never reroutes onto a nodeless comfy (mirrors how
    vision uses _worker_vision_capable). STRICT: unknown/absent = not capable."""
    comfy = (worker or {}).get("comfy")
    if not isinstance(comfy, dict) or not comfy.get("available"):
        return False
    return bool(comfy.get("id_lock"))


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
            self._base_model_key = cfg.model_key
            self._local = None

        # MODEL GROUPS: the EFFECTIVE key, per request.
        #
        # ⚠ THIS INSTANCE IS SHARED. dispatch caches runners in ``_INSTANCES``
        # keyed by (model_key, task), so there is exactly ONE DelegatingRunner
        # per model on the whole process and every concurrent request for that
        # model is holding it. Assigning ``self.model_key`` to a group's chosen
        # member — the obvious implementation, and the one this started as —
        # would therefore rewrite the cached runner PERMANENTLY (the next
        # request for the original key gets the member without ever consulting
        # the group) and race two concurrent requests against each other.
        #
        # So the member is a per-request CONTEXT value and the instance is
        # never mutated. ContextVar rather than threading.local because run()
        # and stream() are async: several requests interleave on ONE event-loop
        # thread, which a thread-local cannot separate. asyncio copies the
        # context per Task, so a value set inside one request is invisible to
        # every other one, and unset — the default, and the state on every
        # off-path request — reads through to the key the caller named.
        @property
        def model_key(self):
            return _MEMBER_KEY.get() or self._base_model_key

        def _local_runner(self):
            if self._local is None:
                self._local = local_cls(self.cfg)
            return self._local

        async def run(self, req):
            # Operator BLOCK gate — fail fast + honest, ahead of selection and
            # the local-serving policy, so a blocked model refuses on EVERY box
            # (worker-pool central or a local-serving self-host) with the same
            # distinct reason. Not a load error → surfaced as a plain refusal.
            _blk = _blocked_reason(self.model_key)
            if _blk:
                raise RuntimeError(_blk)
            pool = getattr(req, "pool", None)
            # MODEL GROUPS: swap in the group's chosen iteration, if any. Returns
            # None — a no-op — whenever groups are off (the default), so this is
            # the ENTIRE off-path diff at this call site. Deliberately AFTER the
            # block gate: a blocked key refuses as itself, never by silently
            # routing to a sibling.
            _mk = _member_key(self.model_key, pool, task)
            if _mk and _mk != self.model_key:
                _MEMBER_KEY.set(_mk)
            # ID-LOCK: a request carrying reference images (paths, or the b64
            # offload transport) is an identity-locked STILL — it MUST land on a
            # box whose comfy has the IPAdapter nodes. Gate selection + reroute on
            # comfy.id_lock, exactly as vision gates on engine.supports_vision.
            _id_lock = bool(getattr(req, "reference_images", None)
                            or getattr(req, "reference_images_b64", None))
            # Pass the id_lock constraint ONLY when it applies, so a plain request
            # calls _select(mk, pool, task) byte-identically to before this slice
            # (older _select overrides / mocks that predate the kwarg are untouched).
            _sel_kw = {"require_comfy_id_lock": True} if _id_lock else {}
            _viable = (_worker_vision_capable if _vision_task
                       else _worker_comfy_id_lock_capable if _id_lock else None)
            # Per-request worker pin also gates the cap-full REROUTE: without
            # this, _reserve_once could legally move a pinned request onto a
            # different box the moment the named one is at capacity — the exact
            # silent reroute the pin contract forbids.
            _pin = _requested_worker_name(req)
            if _pin:
                _cap_viable = _viable
                _viable = (lambda w, _p=_pin, _v=_cap_viable:
                           (w.get("name") == _p or w.get("id") == _p)
                           and (_v is None or _v(w)))

            # Cold-load HOLD (t36), one-shot flavor: a FEASIBLE-but-COLD model
            # whose on-demand load trips a TRANSIENT failure is HELD and retried
            # (bounded by the ceiling/stall clocks + the worker's honest
            # load-state) instead of failing fast. run() can't stream progress, so
            # it just holds the request through the load. Concurrent one-shots for
            # the same model coalesce at the worker's own gen_gate. Genuine
            # infeasibility (no worker, a PERMANENT load error) still fails fast.
            hold = _cold_hold_enabled() and not _local_fallback_allowed()
            start = time.time()
            deadline = start + _cold_hold_max_s()
            stall_s = _cold_hold_stall_s()
            last_move = start
            last_err = ""
            # Retry pacing: base poll while the load PROGRESSES, exponential
            # backoff (doubling to a cap) while it does not — a failing attempt
            # must not be re-fired at storm rate. See _retry_backoff_next.
            retry_wait = _cold_hold_poll_s()
            # Last progress line we actually OBSERVED, so a terminal
            # message can say where it stopped, not just that it did.
            last_progress = None
            # Cold-hold ADMISSION CAP: taken once, on the first selected worker,
            # and released in the finally below. Full + model-not-loaded raises
            # ColdHoldCapacityError straight out of run() — a fast honest 503,
            # never a queued slot. See the cap block near _hold_try_acquire.
            permit = None
            admitted = False
            _want_worker = _requested_worker_name(req)
            try:
                while True:
                    if _want_worker:
                        # Explicit per-request worker pin: binds routing, or
                        # fails naming why — never silently rerouted.
                        worker, spill_override = _resolve_requested_worker(
                            _want_worker, self.model_key, pool, task), None
                    else:
                        worker, spill_override = _select(self.model_key, pool, task, **_sel_kw)
                    if worker and _vision_task and not _worker_vision_capable(worker):
                        logger.info("worker %s doesn't advertise vision (engine.supports_vision); "
                                    "serving %s where vision actually works instead",
                                    worker.get("id"), self.model_key)
                        worker = None
                    if not worker:
                        break  # no worker selected → refusal / local below (fail fast)
                    # Dispatch-queue verdict: this (worker, model) failed
                    # PERMANENTLY within the TTL — answer from the cache, do
                    # not launch another doomed attempt (see _LOAD_VERDICTS).
                    _cached = _active_load_verdict(worker.get("id"), self.model_key)
                    if _cached and not _local_fallback_allowed():
                        raise RuntimeError(
                            _verdict_message(self.model_key, worker, _cached))
                    if hold and not admitted:
                        admitted = True
                        permit = _admit_cold_hold(self.model_key, worker, start)
                    elif hold and permit is None:
                        # Admitted uncounted (the model read warm) but the call is
                        # holding anyway — top up opportunistically so the counter
                        # tells the truth. Never refuses: we are already committed.
                        permit = _hold_try_acquire()
                    # Cap-aware admission: WorkerBusyError (honest 429/503) propagates
                    # unchanged — concurrency saturation is not a cold load.
                    slot = await _acquire_relay_slot_async(self.model_key, pool, worker,
                                                           spill_override, viable=_viable,
                                                           task=task)
                    worker, spill_override = slot.worker, slot.spill
                    payload = _worker_payload(task, req, self.model_key, worker.get("id"),
                                              spill_override=spill_override)
                    if payload is None:
                        slot.release()
                        break  # unbuildable (oversized inline) → local, as before
                    action = None                       # "local" | "retry" | None(=done)
                    try:
                        _res = await _worker_run_once(
                            worker, payload, self.result_type,
                            request_id=req.request_id, model_key=self.model_key)
                        # ONE-SHOT tok/s. The result schema (TaskResult) is
                        # extra="allow", so a worker's `timings` survives validation
                        # as an extra attribute and needs no wire version bump in
                        # this direction; an older worker simply sends none and this
                        # records nothing. Stamped AFTER a successful relay, so a
                        # failed call never pollutes the history with a rate that
                        # was never achieved.
                        _record_serve_metrics(
                            worker, self.model_key,
                            {"timings": getattr(_res, "timings", None)})
                        _clear_load_verdict(worker.get("id"), self.model_key)
                        return _res
                    except Exception as exc:
                        if _is_request_shape_error(exc) and not _local_fallback_allowed():
                            # Malformed for this model's chat template — fail FAST
                            # and name the real fault. Never held (running it local
                            # would fail the same way, and holding it is the hang).
                            # Deliberately NOT recorded as a load verdict: the
                            # fault is this request's messages, not the model.
                            raise RuntimeError(
                                _request_shape_message(self.model_key, worker, exc)) from exc
                        if _local_fallback_allowed():
                            logger.warning("worker run failed (%s); running %s locally",
                                           exc, self.model_key)
                            action = "local"
                        elif (not hold) or _is_permanent_load_error(exc):
                            if _is_permanent_load_error(exc):
                                _record_load_verdict(worker.get("id"),
                                                     self.model_key, str(exc))
                            raise RuntimeError(
                                f"worker {worker.get('name') or worker.get('id')} "
                                f"failed for {self.model_key}: {exc} (local fallback "
                                f"disabled for worker-assigned models; set "
                                f"HUGPY_LOCAL_FALLBACK=always to allow)") from exc
                        else:
                            last_err = str(exc)
                            action = "retry"
                    finally:
                        slot.release()
                    if action == "local":
                        break
                    # action == "retry": transient hold. Honest-fail / stall / ceiling.
                    moved, _prog, _msg, honest = _cold_progress(self.model_key, worker, start)
                    if honest:
                        # The worker's load-state names a hard failure — record
                        # it so queued/re-submitted calls fail fast (see
                        # _LOAD_VERDICTS) instead of re-driving the same load.
                        _record_load_verdict(worker.get("id"), self.model_key, honest)
                        raise RuntimeError(
                            f"worker {worker.get('name') or worker.get('id')} failed to "
                            f"load {self.model_key}: {honest}")
                    if _msg:
                        last_progress = _msg
                    # A structured busy/503 from this worker is the worker
                    # WORKING, not the worker silent — see _is_worker_busy_signal.
                    if moved or _is_worker_busy_signal(last_err):
                        last_move = time.time()
                    now = time.time()
                    if now > deadline:
                        raise RuntimeError(_cold_timeout_message(
                            self.model_key, worker, last_err,
                            last_progress=last_progress, ceiling=True))
                    if (now - last_move) > stall_s:
                        raise RuntimeError(_cold_timeout_message(
                            self.model_key, worker, last_err,
                            last_progress=last_progress,
                            stalled_for=now - last_move))
                    if moved:
                        retry_wait = _cold_hold_poll_s()    # progressing: poll tight
                    await asyncio.sleep(retry_wait)
                    retry_wait = _retry_backoff_next(retry_wait)
                    continue
            finally:
                # The permit is a cold-hold admission, not a request permit: it is
                # returned the moment this call stops holding — success, refusal,
                # honest failure, or a CancelledError from an abandoned client.
                if permit is not None:
                    permit.release()
            # Per-box "never serve locally" policy: no worker took this request
            # (none selected, or one failed with fallback allowed), and this box
            # hosts no models — refuse with a clear error instead of loading the
            # model into this process. Default off === today's behavior; workers
            # never set the flag. See managers.serve.policy.
            from ..serve.policy import no_local_serving, local_serving_error
            if no_local_serving():
                raise RuntimeError(local_serving_error(
                    self.model_key,
                    detail=_no_worker_detail(self.model_key, pool, task)))
            result = self._local_runner().run(req=req)
            if inspect.isawaitable(result):
                result = await result
            return result

        async def stream(self, req, cancel_event=None):
            # Operator BLOCK gate — the streaming twin of run()'s: yield the
            # honest refusal as an ErrorEvent (the pre-token honest-fail idiom)
            # and stop, before any selection or local-serving fallback.
            _blk = _blocked_reason(self.model_key)
            if _blk:
                yield ErrorEvent(request_id=req.request_id, message=_blk)
                return
            pool = getattr(req, "pool", None)
            # MODEL GROUPS — the streaming twin of run()'s consult. Same no-op
            # when the feature is off (the default).
            _mk = _member_key(self.model_key, pool, task)
            if _mk and _mk != self.model_key:
                _MEMBER_KEY.set(_mk)
            # ID-LOCK parity with run(): a request carrying reference images must
            # land on a comfy-with-IPAdapter box; gate selection + reroute on it.
            _id_lock = bool(getattr(req, "reference_images", None)
                            or getattr(req, "reference_images_b64", None))
            _sel_kw = {"require_comfy_id_lock": True} if _id_lock else {}
            _viable = (_worker_vision_capable if _vision_task
                       else _worker_comfy_id_lock_capable if _id_lock else None)
            # Per-request worker pin also gates the cap-full REROUTE: without
            # this, _reserve_once could legally move a pinned request onto a
            # different box the moment the named one is at capacity — the exact
            # silent reroute the pin contract forbids.
            _pin = _requested_worker_name(req)
            if _pin:
                _cap_viable = _viable
                _viable = (lambda w, _p=_pin, _v=_cap_viable:
                           (w.get("name") == _p or w.get("id") == _p)
                           and (_v is None or _v(w)))

            # -- ONE worker-relay attempt ------------------------------------
            # Yields StreamEvents (allocation banner is emitted by the loop, not
            # here). Returns normally once it produced tokens or a terminal done.
            # For a PRE-TOKEN failure it raises: _LoadFailed (honest → surface),
            # _ColdRetry (transient → the loop holds + retries), or
            # _RelayUnbuildable (oversized payload / operator opted into local
            # fallback → the loop breaks to local). This is the pre-cold-hold
            # relay logic verbatim, with the two pre-token "yield ErrorEvent;
            # return" sites replaced by a classified raise.
            async def _relay_attempt(worker, spill_override):
                payload = _worker_payload(task, req, self.model_key, worker.get("id"),
                                          spill_override=spill_override)
                if payload is None:
                    raise _RelayUnbuildable()
                wname = worker.get("name") or worker.get("id") or "worker"
                produced_tokens = False
                try:
                    async for ev in _worker_stream(worker, payload, req.request_id):
                        etype = getattr(ev, "type", None)
                        if etype == "error":
                            if produced_tokens:
                                # Errored after tokens — can't replay; surface as
                                # interrupted (never retried, never held).
                                yield ErrorEvent(request_id=req.request_id,
                                                 message=f"{_humanize_worker_error(wname, ev.message)} "
                                                         f"(the reply was interrupted partway through)")
                                return
                                # pragma: no cover
                            if _is_request_shape_error(ev.message):
                                # Request-shape (chat template) — honest, never
                                # held, never blamed on box size. Ahead of the
                                # local-fallback branch: local would render the
                                # same template and fail identically.
                                raise _LoadFailed(_request_shape_message(
                                    self.model_key, worker, ev.message))
                            # Busy/loading first, same as the transport branch
                            # below: a worker that reports "still loading" inside
                            # a 200 SSE is a worker to HOLD for, and that must
                            # not be reachable by the local-fallback or the
                            # permanent-error branch.
                            if _is_worker_busy_signal(ev.message):
                                raise _ColdRetry(ev.message)
                            if _local_fallback_allowed():
                                logger.warning("worker %s errored before output (%s); "
                                               "running %s locally", worker.get("id"),
                                               ev.message, self.model_key)
                                raise _RelayUnbuildable()
                            if _is_permanent_load_error(ev.message):
                                _record_load_verdict(worker.get("id"),
                                                     self.model_key, str(ev.message))
                                raise _LoadFailed(_humanize_worker_error(wname, ev.message))
                            raise _ColdRetry(ev.message)   # transient — hold + retry
                        yield ev
                        if etype == "token":
                            produced_tokens = True
                        elif etype == "done":
                            # STREAMING tok/s, stamped on the terminal done —
                            # the streaming twin of run()'s post-relay stamp,
                            # and the only frame that carries a rate. llama.cpp
                            # pushes `timings` onto the FINAL SSE chunk
                            # unconditionally (server-task.cpp,
                            # to_json_oaicompat_chat_stream), so streaming is
                            # NOT a blind spot here — it is measured exactly as
                            # well as the one-shot path. A worker too old to
                            # send it yields None and records nothing.
                            _record_serve_metrics(
                                worker, self.model_key,
                                {"timings": getattr(ev, "timings", None)})
                            return  # terminal (even if empty)
                    else:
                        # Stream ended with no done/error marker.
                        if produced_tokens:
                            return
                        if _local_fallback_allowed():
                            logger.warning("worker %s produced no output; running %s locally",
                                           worker.get("id"), self.model_key)
                            raise _RelayUnbuildable()
                        raise _ColdRetry(f"worker {wname} produced no output "
                                         f"(still loading?)")
                except (_ColdRetry, _LoadFailed, _RelayUnbuildable):
                    raise
                except Exception as exc:
                    if produced_tokens:
                        yield ErrorEvent(request_id=req.request_id,
                                         message=f"worker {wname}: stream interrupted: {exc}")
                        return
                    if _is_request_shape_error(exc):
                        raise _LoadFailed(_request_shape_message(
                            self.model_key, worker, exc))
                    # BUSY IS A HOLD, AND IT IS DECIDED FIRST (operator retest
                    # 2026-07-28: "a cold load STILL errors first, then works on
                    # retry"). A structured 503 on the FIRST attempt means the
                    # worker's gen-gate is holding us while the model loads —
                    # the correct answer is to enter the hold, not to hand the
                    # user an error that tells them to press send again. Ahead
                    # of BOTH the local-fallback branch and the permanent check
                    # so neither can turn a load-in-progress into a terminal
                    # outcome.
                    if _is_worker_busy_signal(exc):
                        logger.info("worker %s is busy/loading %s (%s) — holding",
                                    worker.get("id"), self.model_key, exc)
                        raise _ColdRetry(str(getattr(exc, "message", None) or exc))
                    if _local_fallback_allowed():
                        logger.warning("worker offload failed (%s); running %s locally",
                                       exc, self.model_key)
                        raise _RelayUnbuildable()
                    if _is_permanent_load_error(exc):
                        _record_load_verdict(worker.get("id"),
                                             self.model_key, str(exc))
                        raise _LoadFailed(f"worker {wname} failed for {self.model_key}: {exc}")
                    raise _ColdRetry(str(exc))            # transient — hold + retry

            # -- the HOLD loop -----------------------------------------------
            hold = _cold_hold_enabled() and not _local_fallback_allowed()
            start = time.time()
            deadline = start + _cold_hold_max_s()
            stall_s = _cold_hold_stall_s()
            last_move = start
            last_err = ""
            # Retry pacing — the streaming twin of run()'s: base poll while the
            # load progresses, exponential backoff while it does not.
            retry_wait = _cold_hold_poll_s()
            # Last progress line we actually OBSERVED, so a terminal
            # message can say where it stopped, not just that it did.
            last_progress = None
            announced_wid = None
            # Cold-hold ADMISSION CAP — see the block near _hold_try_acquire.
            # Taken once (on the first selected worker), released on the first
            # token (the call is warm, no longer a hold) and again in the finally
            # that wraps this loop. Cap full + model not loaded ⇒ an immediate,
            # honest ErrorEvent instead of a slot parked for up to the ceiling.
            permit = None
            admitted = False
            _want_worker = _requested_worker_name(req)
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        return  # cancel-while-held: teardown marks the job cancelled
                    if _want_worker:
                        # Explicit per-request worker pin — same contract as the
                        # one-shot path: bind or fail naming why, never reroute.
                        worker, spill_override = _resolve_requested_worker(
                            _want_worker, self.model_key, pool, task), None
                    else:
                        worker, spill_override = _select(self.model_key, pool, task, **_sel_kw)
                    if worker and _vision_task and not _worker_vision_capable(worker):
                        logger.info("worker %s doesn't advertise vision "
                                    "(engine.supports_vision); serving %s where vision "
                                    "actually works instead", worker.get("id"), self.model_key)
                        worker = None
                    if not worker:
                        break  # no worker selected → refusal / local below (fail fast)
                    # Dispatch-queue verdict — the streaming twin of run()'s
                    # check: a permanent failure recorded within the TTL answers
                    # from the cache; no new attempt reaches the worker.
                    _cached = _active_load_verdict(worker.get("id"), self.model_key)
                    if _cached and not _local_fallback_allowed():
                        yield ErrorEvent(request_id=req.request_id,
                                         message=_verdict_message(
                                             self.model_key, worker, _cached))
                        return
                    if hold and not admitted:
                        admitted = True
                        try:
                            permit = _admit_cold_hold(self.model_key, worker, start)
                        except ColdHoldCapacityError as full:
                            yield ErrorEvent(request_id=req.request_id,
                                             message=full.stream_message())
                            return
                    elif hold and permit is None:
                        # Admitted uncounted (the model read warm) but this call is
                        # holding anyway — top up so the counter tells the truth.
                        # Never refuses: we are already committed.
                        permit = _hold_try_acquire()
                    try:
                        slot = await _acquire_relay_slot_async(self.model_key, pool, worker,
                                                               spill_override, viable=_viable,
                                                               task=task)
                    except WorkerBusyError as busy:
                        # Concurrency saturation is its own honest signal (not a cold
                        # load) — surfaced as today, unchanged.
                        yield ErrorEvent(request_id=req.request_id,
                                         message=busy.stream_message())
                        return
                    worker, spill_override = slot.worker, slot.spill
                    wid = worker.get("id") or ""
                    if wid != announced_wid:
                        yield _alloc_status(req.request_id, worker)  # once per worker
                        announced_wid = wid
                    key = (wid, self.model_key)

                    # COALESCE: if another call is already driving this cold load, do
                    # NOT pile a second on-demand load on — release the gate slot and
                    # wait, surfacing progress. (check-and-add is atomic on the one loop.)
                    if hold and key in _COLD_KICKING:
                        slot.release()
                        moved, prog, msg, honest = _cold_progress(self.model_key, worker, start)
                        if honest:
                            yield ErrorEvent(request_id=req.request_id,
                                             message=_humanize_worker_error(
                                                 worker.get("name") or wid, honest))
                            return
                        if msg:
                            last_progress = msg
                        if moved or _is_worker_busy_signal(last_err):
                            last_move = time.time()
                        now = time.time()
                        if now > deadline or (now - last_move) > stall_s:
                            yield ErrorEvent(request_id=req.request_id,
                                             message=_cold_timeout_message(
                                                 self.model_key, worker, last_err,
                                                 last_progress=last_progress,
                                                 stalled_for=(None if now > deadline
                                                              else now - last_move),
                                                 ceiling=now > deadline))
                            return
                        yield _loading_status(req.request_id, self.model_key, worker, prog, msg)
                        await asyncio.sleep(_cold_hold_poll_s())
                        continue

                    if hold:
                        _COLD_KICKING.add(key)
                    action = None                       # "local" | "retry" | None(=done)
                    warm = False
                    try:
                        async for ev in _relay_attempt(worker, spill_override):
                            if hold and not warm and getattr(ev, "type", None) == "token":
                                # First token ⇒ the model is LOADED. Free the cold-kick
                                # key NOW so coalesced waiters dispatch CONCURRENTLY
                                # against the warm model instead of serializing behind
                                # this call's whole generation. (idempotent w/ finally.)
                                _COLD_KICKING.discard(key)
                                # …and give the cold-hold permit back for exactly the
                                # same reason: this call is no longer WAITING FOR A
                                # LOAD, it is generating. A long WARM stream must
                                # never occupy an admission the next cold caller
                                # needs. (release() is idempotent w/ the finally.)
                                if permit is not None:
                                    permit.release()
                                    permit = None
                                warm = True
                                _clear_load_verdict(wid, self.model_key)
                            yield ev
                        return  # attempt completed (tokens/done or interrupted) — terminal
                    except _RelayUnbuildable:
                        action = "local"                # oversized payload / opted-in local
                    except _LoadFailed as lf:
                        yield ErrorEvent(request_id=req.request_id, message=lf.message)
                        return
                    except _ColdRetry as cr:
                        last_err = cr.message
                        if not hold:
                            # Feature disabled → today's behavior: surface, no retry.
                            yield ErrorEvent(request_id=req.request_id,
                                             message=_humanize_worker_error(
                                                 worker.get("name") or wid, cr.message))
                            return
                        action = "retry"
                    finally:
                        # Release the gate slot + free the cold-kick key BEFORE any
                        # wait, so a coalesced waiter proceeds the instant this kick
                        # ends (also releases on client-disconnect GeneratorExit).
                        slot.release()
                        if hold:
                            _COLD_KICKING.discard(key)
                    if action == "local":
                        break  # → local fallback / refusal below
                    # action == "retry": the transient hold. Consult load-state for an
                    # honest fail / progress, emit a loading status, bound by the
                    # stall/ceiling clocks, then retry.
                    moved, prog, msg, honest = _cold_progress(self.model_key, worker, start)
                    if honest:
                        # Hard load failure from the worker's load-state — record
                        # so queued/re-submitted calls answer from the cache.
                        _record_load_verdict(wid, self.model_key, honest)
                        yield ErrorEvent(request_id=req.request_id,
                                         message=_humanize_worker_error(
                                             worker.get("name") or wid, honest))
                        return
                    if msg:
                        last_progress = msg
                    if moved or _is_worker_busy_signal(last_err):
                        last_move = time.time()
                    now = time.time()
                    if now > deadline or (now - last_move) > stall_s:
                        yield ErrorEvent(request_id=req.request_id,
                                         message=_cold_timeout_message(
                                             self.model_key, worker, last_err,
                                             last_progress=last_progress,
                                             stalled_for=(None if now > deadline
                                                          else now - last_move),
                                             ceiling=now > deadline))
                        return
                    yield _loading_status(req.request_id, self.model_key, worker, prog, msg)
                    if moved:
                        retry_wait = _cold_hold_poll_s()    # progressing: poll tight
                    await asyncio.sleep(retry_wait)
                    retry_wait = _retry_backoff_next(retry_wait)
                    continue
            finally:
                # Cold-hold permit returned the moment this call stops holding —
                # terminal event, refusal, or the GeneratorExit a disconnected
                # client's teardown cascades through this generator.
                if permit is not None:
                    permit.release()

            # Per-box "never serve locally" policy: no worker took this request
            # and this box hosts no models — surface a clear error instead of
            # streaming from a locally-loaded model. Default off === today's
            # behavior; workers never set the flag. See managers.serve.policy.
            from ..serve.policy import no_local_serving, local_serving_error
            if no_local_serving():
                yield ErrorEvent(request_id=req.request_id,
                                 message=local_serving_error(
                                     self.model_key,
                                     detail=_no_worker_detail(self.model_key, pool, task)))
                return
            # Local fallback — reuse dispatch's shared stream-or-wrap primitive
            # (imported lazily to avoid a resolvers<->dispatch import cycle).
            # Re-announce as "local" so the banner reflects this path (covers
            # no worker selected, unbuildable payload, and pre-output failure).
            yield _alloc_status(req.request_id, None)
            from ..dispatch.dispatch import stream_runner
            async for ev in stream_runner(self._local_runner(), req, cancel_event=cancel_event):
                yield ev

    return DelegatingRunner
