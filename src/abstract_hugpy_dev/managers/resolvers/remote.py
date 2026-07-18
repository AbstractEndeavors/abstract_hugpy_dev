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


def _model_slot_served(worker: Optional[dict], model_key: str) -> bool:
    """True when ``model_key`` is currently seated in a SLOT child on this worker.

    Then the worker's llama-server / llama_cpp.server child schedules concurrency
    itself and central must NOT apply the in-process cap. Best-effort over the
    heartbeat ``slots``/``allocations`` snapshot; any doubt → False (apply the
    cap — over-gating a slot model is a small latency cost, under-gating an
    in-process model is a crash).
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
    tail = str(model_key).split("/")[-1]
    return any(tail == k.split("/")[-1] for k in keys)


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
    """Hard ceiling for holding a cold call (the 'presumed success' window).
    Default 300s — long enough for a real GPU load/swap, short enough that a
    truly wedged load fails honestly. Operator-tunable (defaults are promises)."""
    return _env_float("HUGPY_COLD_HOLD_MAX_S", 300.0)


def _cold_hold_stall_s() -> float:
    """No-forward-progress bound: if the load makes no movement (not loading, not
    loaded, no fresh progress) for this long, the hold gives up honestly. Default
    90s — mirrors the job store's honest-stall clock."""
    return _env_float("HUGPY_COLD_HOLD_STALL_S", 90.0)


def _cold_hold_poll_s() -> float:
    """Backoff between relay retries / progress emits while holding. Default 2s."""
    return _env_float("HUGPY_COLD_HOLD_POLL_S", 2.0)


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
    # Operator model BLOCK: a distinct, permanent operator refusal — never held
    # or retried (see comms.blocklist.BLOCKED_MARKER; this string mirrors it).
    "blocked from the serving pool",
)


def _is_permanent_load_error(err: Any) -> bool:
    low = str(getattr(err, "message", None) or err or "").lower()
    return any(m in low for m in _PERMANENT_LOAD_MARKERS)


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
                          last_err: str) -> str:
    """The honest give-up line when a held load never became ready in time."""
    wname = (worker or {}).get("name") or (worker or {}).get("id") or "worker"
    tail = f" (last: {last_err})" if last_err else ""
    return (f"'{model_key}' did not finish loading on '{wname}' in time"
            f"{tail} — the model may be too large for the box or the load stalled; "
            f"try again or assign it elsewhere.")


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
    # A request type may carry its OWN `task` field (TranscribeRequest.task is
    # whisper's transcribe/translate MODE) — dumped last, it clobbered the
    # DISPATCH task key and every whisper offload died on the worker with
    # "Unknown task='transcribe'". Keep the domain field under its builder
    # alias and let the dispatch key own `task`.
    if payload.get("task") not in (None, task):
        payload["whisper_task"] = payload.pop("task")
    payload["task"] = task
    spill = spill_override if spill_override is not None else _spill_for(worker_id, model_key)
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
        if resp.status_code >= 400:
            # The agent ships failures AS DATA (ok:false + error + traceback
            # tail) alongside the 4xx/5xx status; a bare raise_for_status()
            # discarded that body and reduced the console to "Server error
            # '500 …'" with no cause. Surface the worker's own reason — the
            # caller (DelegatingRunner) stamps the worker name onto it.
            detail = ""
            try:
                detail = str((resp.json() or {}).get("error") or "")
            except ValueError:
                pass
            if detail:
                raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
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
            self.model_key = cfg.model_key
            self._local = None

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
            while True:
                worker, spill_override = _select(self.model_key, pool, task, **_sel_kw)
                if worker and _vision_task and not _worker_vision_capable(worker):
                    logger.info("worker %s doesn't advertise vision (engine.supports_vision); "
                                "serving %s where vision actually works instead",
                                worker.get("id"), self.model_key)
                    worker = None
                if not worker:
                    break  # no worker selected → refusal / local below (fail fast)
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
                    return await _worker_run_once(
                        worker, payload, self.result_type,
                        request_id=req.request_id, model_key=self.model_key)
                except Exception as exc:
                    if _local_fallback_allowed():
                        logger.warning("worker run failed (%s); running %s locally",
                                       exc, self.model_key)
                        action = "local"
                    elif (not hold) or _is_permanent_load_error(exc):
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
                    raise RuntimeError(
                        f"worker {worker.get('name') or worker.get('id')} failed to "
                        f"load {self.model_key}: {honest}")
                if moved:
                    last_move = time.time()
                now = time.time()
                if now > deadline or (now - last_move) > stall_s:
                    raise RuntimeError(_cold_timeout_message(self.model_key, worker, last_err))
                await asyncio.sleep(_cold_hold_poll_s())
                continue
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
            # ID-LOCK parity with run(): a request carrying reference images must
            # land on a comfy-with-IPAdapter box; gate selection + reroute on it.
            _id_lock = bool(getattr(req, "reference_images", None)
                            or getattr(req, "reference_images_b64", None))
            _sel_kw = {"require_comfy_id_lock": True} if _id_lock else {}
            _viable = (_worker_vision_capable if _vision_task
                       else _worker_comfy_id_lock_capable if _id_lock else None)

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
                            if _local_fallback_allowed():
                                logger.warning("worker %s errored before output (%s); "
                                               "running %s locally", worker.get("id"),
                                               ev.message, self.model_key)
                                raise _RelayUnbuildable()
                            if _is_permanent_load_error(ev.message):
                                raise _LoadFailed(_humanize_worker_error(wname, ev.message))
                            raise _ColdRetry(ev.message)   # transient — hold + retry
                        yield ev
                        if etype == "token":
                            produced_tokens = True
                        elif etype == "done":
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
                    if _local_fallback_allowed():
                        logger.warning("worker offload failed (%s); running %s locally",
                                       exc, self.model_key)
                        raise _RelayUnbuildable()
                    if _is_permanent_load_error(exc):
                        raise _LoadFailed(f"worker {wname} failed for {self.model_key}: {exc}")
                    raise _ColdRetry(str(exc))            # transient — hold + retry

            # -- the HOLD loop -----------------------------------------------
            hold = _cold_hold_enabled() and not _local_fallback_allowed()
            start = time.time()
            deadline = start + _cold_hold_max_s()
            stall_s = _cold_hold_stall_s()
            last_move = start
            last_err = ""
            announced_wid = None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return  # cancel-while-held: teardown marks the job cancelled
                worker, spill_override = _select(self.model_key, pool, task, **_sel_kw)
                if worker and _vision_task and not _worker_vision_capable(worker):
                    logger.info("worker %s doesn't advertise vision "
                                "(engine.supports_vision); serving %s where vision "
                                "actually works instead", worker.get("id"), self.model_key)
                    worker = None
                if not worker:
                    break  # no worker selected → refusal / local below (fail fast)
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
                    if moved:
                        last_move = time.time()
                    now = time.time()
                    if now > deadline or (now - last_move) > stall_s:
                        yield ErrorEvent(request_id=req.request_id,
                                         message=_cold_timeout_message(self.model_key,
                                                                       worker, last_err))
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
                            warm = True
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
                    yield ErrorEvent(request_id=req.request_id,
                                     message=_humanize_worker_error(
                                         worker.get("name") or wid, honest))
                    return
                if moved:
                    last_move = time.time()
                now = time.time()
                if now > deadline or (now - last_move) > stall_s:
                    yield ErrorEvent(request_id=req.request_id,
                                     message=_cold_timeout_message(self.model_key,
                                                                   worker, last_err))
                    return
                yield _loading_status(req.request_id, self.model_key, worker, prog, msg)
                await asyncio.sleep(_cold_hold_poll_s())
                continue

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
