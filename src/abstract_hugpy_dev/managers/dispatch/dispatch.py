"""Runner dispatch — dumb consumer of Resolution.

All routing logic lives in model_resolver.resolve(). This module owns
two things and only two things:

    1. A per-process instance cache keyed by (model_key, task).
    2. An execute_prompt entry point that turns request kwargs into
       a result by handing off to resolve() and the runner.

It does not:
    - Decide which builder to call.
    - Decide which runner class to instantiate.
    - Validate that model+task are compatible.
    - Default task to cfg.primary_task.

If you find yourself adding any of that here, stop and add it to
model_resolver.resolve() instead. That's the whole point.

Why a per-process cache:
    Loading a 14B model takes seconds; doing it on every request is
    obviously wrong. Per-(model_key, task) caching means the same
    model can host two task-runners (e.g. text-generation + code-
    generation on one llama.cpp instance) and each gets its own
    runner wrapper, but inner singletons (REGISTRY for DeepCoder,
    get_llama_runner for llama.cpp) still de-dup the heavy state.
"""

from __future__ import annotations
import inspect
import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from ..resolvers import resolve
from .imports import *

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process instance cache — keyed by (model_key, task) per the contract
# in Resolution.cache_key.
# ---------------------------------------------------------------------------

_INSTANCES: Dict[Tuple[str, str], Runner] = {}
_INSTANCES_LOCK = threading.Lock()

# Models whose runner build (weight load) is IN FLIGHT right now — read by the
# worker heartbeat so the console can attribute "heating" (load in progress)
# distinctly from "serving" (resident) and "cold" (assigned, not loaded).
# Separate lock: heartbeats must read this while a slow build holds
# _INSTANCES_LOCK for minutes.
_BUILDING: set = set()
_BUILDING_LOCK = threading.Lock()


def loading_model_keys() -> List[str]:
    """model_keys whose runner/weights are currently loading in this process."""
    with _BUILDING_LOCK:
        return sorted(_BUILDING)


# ---------------------------------------------------------------------------
# Residency policy — CONTENTION-based LRU (operator doctrine, locked 2026-07-11).
#
# Doctrine: "The likelihood of one model being queried having another
# consecutive query is high — this is the M.O. of hugpy. Resources should
# facilitate keeping models hot; models should spend the least possible time
# 'loading'." So an on-demand model loads on call and STAYS resident until
# another load actually NEEDS its resources; then the least-recently-used
# on-demand resident yields. static/pinned residents NEVER yield.
#
# 2026-07-11 DRIFT CORRECTION: this used to be a CLOCK — the worker's residency
# sweep evicted any on-demand resident after on_demand_ttl_s of no requests
# (default 900s), so a model that had just answered a chat was gone minutes
# later and the next chat paid a full reload. That contradicted the doctrine and
# the console's own residency tooltip ("holds its slot until another model needs
# the seat"). The idle clock is now OPT-IN (worker_agent._residency_sweep_once:
# it runs only when the operator explicitly sets on_demand_ttl_s); the DEFAULT
# trigger is memory contention — ensure_headroom_for_load(), below.
# ---------------------------------------------------------------------------

# Per-model last-request time — the LRU key. touch_model() fires on every served
# request (execute_prompt / execute_prompt_stream); the least-recently touched
# on-demand resident is the first to yield under contention.
_LAST_USED: Dict[str, float] = {}


def touch_model(model_key: str) -> None:
    import time as _time
    _LAST_USED[model_key] = _time.time()


def last_used_snapshot() -> Dict[str, float]:
    return dict(_LAST_USED)


class LoadRefusal(Exception):
    """A GPU load that can't fit even after evicting every permissible resident
    (slice 10). Raised by ensure_headroom_for_load's cross-tier make-room BEFORE
    any CUDA allocation, carrying the typed reason so the caller surfaces an
    honest 'won't fit' instead of an admit-then-OOM crash."""
    def __init__(self, reason: dict):
        self.reason = reason or {}
        super().__init__(self.reason.get("reason") or "won't fit on GPU")


# ---------------------------------------------------------------------------
# Contention hooks — registered by the worker agent; None on bare central.
#
# dispatch is shared code and must not import worker_agent (the dependency runs
# the other way). So, mirroring managers.serve.slots.set_residency_lookup, the
# worker registers the box-specific policy here at startup. Unset -> contention
# eviction is a no-op: central does no local serving, and tests register their
# own fakes. dispatch owns the LRU MECHANISM; the worker owns the fit MEASUREMENT.
# ---------------------------------------------------------------------------
_FIT_CHECK = None        # (model_key) -> bool: does loading it fit in current headroom?
_EVICTABLE = None        # (model_key) -> bool: is it a yieldable on-demand in-process resident?
_POST_EVICT = None       # () -> None: reclaim host RAM / CUDA cache after an eviction
# (model_key) -> dict: CROSS-TIER VRAM make-room (slice 10). The in-process LRU
# yield below only sees _INSTANCES residents; a SLOT CHILD (subprocess) squatting
# the GPU is invisible to it. The worker registers a make-room hook that sees ALL
# residents (in-process + slot + comfy) from the pid-registry measured truth and
# evicts the minimum permissible set through the /ops/evict verb — closing the
# blind admit-then-OOM the incident exposed. None -> the old in-process-only path.
_MAKE_ROOM = None
_CONTENTION_LOCK = threading.Lock()


def set_fit_check(fn) -> None:
    """Register the headroom fit-guard: ``fn(model_key) -> bool``, True when the
    model fits in current headroom without yielding a resident. None disables
    contention eviction (the bare-central / untested default)."""
    global _FIT_CHECK
    _FIT_CHECK = fn


def set_evictable(fn) -> None:
    """Register the yield predicate: ``fn(model_key) -> bool``, True for an
    on-demand in-process resident that MAY yield (not static, not pinned, no
    active gate permits, not slot-backed)."""
    global _EVICTABLE
    _EVICTABLE = fn


def set_post_evict_hook(fn) -> None:
    """Register a post-eviction reclaim (gc + malloc_trim + cuda empty_cache) so
    the next headroom re-check sees the freed memory. None -> skipped."""
    global _POST_EVICT
    _POST_EVICT = fn


def set_make_room(fn) -> None:
    """Register the CROSS-TIER VRAM make-room (slice 10): ``fn(model_key) -> dict``
    with {"action": "proceed"|"evicted"|"refuse", "reason": {...}|None}. Called
    after the in-process LRU yield so a SLOT-CHILD squatter (invisible to the
    in-process path) is also evicted, and an unfittable load is REFUSED before any
    CUDA allocation. None -> the historical in-process-only contention path."""
    global _MAKE_ROOM
    _MAKE_ROOM = fn


def _next_lru_evictable(exclude: str) -> Optional[str]:
    """The least-recently-used yieldable on-demand in-process resident, or None.

    Candidates are the model_keys currently holding a runner in ``_INSTANCES``,
    minus the model being loaded, minus everything the registered ``_EVICTABLE``
    predicate rejects (static, pinned, gate-busy, slot-backed). Ordered by
    ``_LAST_USED`` ascending — the coldest yields first. A never-touched resident
    (warmed by the filler, never requested) sorts oldest and yields first, which
    is exactly right."""
    if _EVICTABLE is None:
        return None
    residents = {mk for (mk, _task) in loaded_model_keys()}
    residents.discard(exclude)
    cands = [mk for mk in residents if _EVICTABLE(mk)]
    if not cands:
        return None
    cands.sort(key=lambda mk: _LAST_USED.get(mk, 0.0))
    return cands[0]


def ensure_headroom_for_load(model_key: str) -> List[str]:
    """CONTENTION eviction — the default (clock-free) residency trigger.

    Called right before a NEW runner is built (see ``_get_or_build_runner``).
    While the registered fit-guard says the incoming model does NOT fit, yield
    the least-recently-used on-demand resident one at a time — re-checking
    headroom after each (the post-evict reclaim hook trims the freed host arena /
    CUDA cache so the re-check actually sees the room) — until it fits or nothing
    is left to yield.

    A model with an in-flight generation (gate permits) is skipped, never ripped
    out from under it — the next LRU candidate is chosen, and the busy one
    becomes evictable only once its permits release. If nothing is yieldable, we
    return and the load proceeds (or fails) EXACTLY as it does today: contention
    only ADDS room, it never changes the too-big error envelope.

    No-op on bare central / when no fit-guard is registered. Returns the list of
    yielded model_keys (for logging + tests).

    CROSS-TIER (slice 10): after the in-process LRU yield, a registered make-room
    hook (set_make_room) runs a SECOND pass that also evicts SLOT-CHILD squatters
    (subprocess VRAM the in-process path cannot see) and REFUSES an unfittable
    load before any CUDA allocation (raising LoadRefusal). Without the hook the
    behavior is byte-identical to before."""
    evicted: List[str] = []
    if _FIT_CHECK is not None:
        with _CONTENTION_LOCK:
            while not _FIT_CHECK(model_key):
                cand = _next_lru_evictable(exclude=model_key)
                if cand is None:
                    break                    # nothing to yield in-process -> below
                logger.info("contention evict: yielding LRU on-demand resident %s to "
                            "make room for %s (doctrine 2026-07-11: keep models hot, "
                            "yield only under memory pressure)", cand, model_key)
                try:
                    evict(cand)
                except Exception:            # noqa: BLE001 — one bad evict must not wedge the load
                    logger.warning("contention evict of %s failed", cand, exc_info=True)
                    break
                evicted.append(cand)
                if _POST_EVICT is not None:
                    try:
                        _POST_EVICT()        # trim so the next fit re-check sees the freed room
                    except Exception:        # noqa: BLE001
                        logger.warning("post-evict reclaim failed", exc_info=True)

    # CROSS-TIER make-room + honest refusal (slice 10). The in-process yield above
    # cannot see a slot child; this hook sees ALL residents and evicts the minimum
    # permissible set (slot children included), then REFUSES if still short.
    if _MAKE_ROOM is not None:
        try:
            verdict = _MAKE_ROOM(model_key)
        except LoadRefusal:
            raise
        except Exception:                    # noqa: BLE001 — a broken hook never blocks a load
            logger.warning("cross-tier make-room failed for %s", model_key,
                           exc_info=True)
            verdict = None
        if isinstance(verdict, dict):
            evicted.extend(v for v in (verdict.get("evicted") or [])
                           if v not in evicted)
            # action == "partial": the full weights don't fit, but the worker
            # admitted an honest GGUF PARTIAL offload (autofit's hybrid). It is an
            # ADMIT, not a refusal — the in-process llama_cpp load then reads the
            # pinned n_gpu_layers via spill.gguf_gpu_layers (set on the served
            # path by the same admission), so we neither raise nor re-price.
            if verdict.get("action") == "refuse":
                raise LoadRefusal(verdict.get("reason") or
                                  {"reason": "won't fit on GPU", "model_key": model_key})
    return evicted


def _get_or_build_runner(res: Resolution) -> Runner:
    """Cache-coherent runner lookup. Double-checked locking under the cache lock."""
    cached = _INSTANCES.get(res.cache_key)
    if cached is not None:
        return cached

    # Contention-based residency (doctrine 2026-07-11): a NEW in-process load may
    # need room. Yield the LRU on-demand resident(s) BEFORE we build — done here,
    # OUTSIDE _INSTANCES_LOCK, because evict() takes that lock too (holding it
    # would deadlock). No-op on bare central (no fit-guard registered).
    ensure_headroom_for_load(res.model_key)

    with _INSTANCES_LOCK:
        cached = _INSTANCES.get(res.cache_key)
        if cached is not None:
            return cached

        logger.info(
            "instantiating runner: model=%s task=%s class=%s framework=%s",
            res.model_key, res.task, res.runner_cls.__name__, res.framework,
        )
        with _BUILDING_LOCK:
            _BUILDING.add(res.model_key)
        try:
            instance = res.runner_cls(res.cfg)
        finally:
            with _BUILDING_LOCK:
                _BUILDING.discard(res.model_key)
        _INSTANCES[res.cache_key] = instance
        return instance


# ---------------------------------------------------------------------------
# Argument normalization — flexible positional input -> kwargs dict.
# ---------------------------------------------------------------------------

def infer_arg_name(arg: Any) -> Optional[str]:
    if arg is None:
        return None
    if isinstance(arg, bool):
        return "do_sample"
    if isinstance(arg, int):
        return "max_new_tokens"
    if isinstance(arg, float):
        return "temperature"
    if isinstance(arg, list):
        return "messages"
    if isinstance(arg, str):
        if os.path.exists(arg):
            return "file"
        lowered = arg.lower()
        looks_like_model = (
            "/" in arg
            or "_gguf" in lowered
            or any(tag in lowered for tag in ("qwen", "llama", "mistral", "gpt"))
        )
        return "model_key" if looks_like_model else "messages"
    return None


def normalize_prompt_kwargs(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convert flexible input into builder-compatible kwargs.

    Explicit kwargs win over inferred positional args. A second float
    becomes top_p (since temperature is already set).
    """
    prompt_kwargs = dict(kwargs)

    for arg in args:
        guessed_key = infer_arg_name(arg)
        if guessed_key is None:
            raise TypeError(f"Could not infer argument type for positional arg: {arg!r}")

        if guessed_key in prompt_kwargs:
            if guessed_key == "temperature" and "top_p" not in prompt_kwargs:
                prompt_kwargs["top_p"] = arg
            continue

        prompt_kwargs[guessed_key] = arg

    return prompt_kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def runner_for(
    model_key: Optional[str] = None,
    *,
    task: Optional[str] = None,
) -> Runner:
    """Get a runner by model_key, task, or both.

    Both are passed through resolve() — so the same (model_key, task)
    pair always lands on the same cached runner, whether you came in
    here or through execute_prompt.
    """
    if model_key is None and task is None:
        raise ValueError("runner_for requires at least one of model_key or task")

    res = resolve({"model_key": model_key, "task": task})
    return _get_or_build_runner(res)


def execute_prompt(*args: Any, **kwargs: Any):
    """One-shot request -> result. Sync entrypoint; awaits inside if needed."""
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)
    res = resolve(prompt_kwargs)
    req = res.builder(prompt_kwargs, res.model_key)
    runner = _get_or_build_runner(res)
    touch_model(res.model_key)   # residency idle clock
    return runner.run(req=req)


async def stream_runner(runner, req, cancel_event=None):
    """Drive any runner to a stream of events — the shared streaming primitive.

    Runners that implement a real async-generator `stream` get streamed through
    (cancel_event forwarded when the signature accepts it). Everyone else is run
    once and emitted as a single token + done. The caller never has to know
    which kind it got.

    This is factored out of execute_prompt_stream so other places that hold a
    runner + built req — notably the DelegatingRunner's local-fallback branch —
    reuse the exact same stream-or-wrap logic instead of duplicating it.
    """
    stream = getattr(runner, "stream", None)
    if stream is not None:
        # Pass cancel_event only if the runner's stream() accepts it.
        try:
            accepts_cancel = "cancel_event" in inspect.signature(stream).parameters
        except (TypeError, ValueError):
            accepts_cancel = False
        produced = stream(req, cancel_event=cancel_event) if (accepts_cancel and cancel_event is not None) else stream(req)
        if hasattr(produced, "__aiter__"):          # real streamer
            async for event in produced:
                yield event
            return
        if inspect.isawaitable(produced):           # coroutine-shaped stream(); don't leak it
            produced.close()

    # one-shot path — the universal verb every runner implements
    result = runner.run(req=req)
    if inspect.isawaitable(result):
        result = await result

    if getattr(result, "ok", True):
        yield TokenEvent(request_id=req.request_id,
                         text=getattr(result, "text", "") or str(result))
        yield DoneEvent(request_id=req.request_id, input_tokens=0,
                        output_chunks=1,
                        finish_reason=getattr(result, "finish_reason", None) or "stop",
                        # ChatResult already defines usage; surface it when the
                        # runner filled it in (None otherwise — additive).
                        usage=getattr(result, "usage", None))
    else:
        yield ErrorEvent(request_id=req.request_id,
                         message=getattr(result, "error", None) or "run failed")


async def execute_prompt_stream(*args, cancel_event=None, **kwargs):
    """Single resolve→builder→runner pass, yielded as events.

    The primitive: resolve the request, build it, stream the runner once.
    Chat continuation/seam handling lives one layer up in execute_chat_stream;
    this stays a single pass so it composes (and so a remote relay sees one
    pass, not a nested continuation loop).

    ``cancel_event`` (an asyncio.Event) is forwarded to runners that accept it
    (llama.cpp, summarizer, DeepCoder), so a caller can stop generation
    mid-stream."""
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)
    res = resolve(prompt_kwargs)
    req = res.builder(prompt_kwargs, res.model_key)
    runner = _get_or_build_runner(res)
    touch_model(res.model_key)   # residency idle clock
    async for event in stream_runner(runner, req, cancel_event=cancel_event):
        yield event
# ---------------------------------------------------------------------------
# Chat continuation engine — one shared implementation.
#
# Hoisted out of the worker agent so local chat and worker chat behave
# identically: both drive this. It wraps the single-pass execute_prompt_stream
# with auto-continuation past the token cap + seam dedup. Because the primitive
# stays single-pass, a remote relay sees a *completed* response (finish=stop)
# and this loop terminates after one pass — no double-continuation.
# ---------------------------------------------------------------------------
from .imports import StatusEvent

# How many continuation passes before giving up (runaway guard). Env-overridable;
# the WORKER_* names are honored too so existing worker deployments keep tuning.
_MAX_CONTINUATIONS = int(os.environ.get("HUGPY_MAX_CONTINUATIONS",
                         os.environ.get("WORKER_MAX_CONTINUATIONS", "20")))
# At a seam the model often re-emits the tail of the previous part; drop an
# overlap up to this many chars.
_SEAM_WINDOW = int(os.environ.get("HUGPY_SEAM_WINDOW",
                   os.environ.get("WORKER_SEAM_WINDOW", "400")))
# finish_reasons that mean "ran out of room" -> continue.
_CONTINUE_ON = {"max_tokens", "length"}
# Per-pass output ceiling. This continuation loop delivers totals larger than
# any single pass, so a runner never needs more than one cap's worth at a time.
# Forwarding a huge max_new_tokens straight through also breaks workers on an
# OLDER abstract_hugpy_dev build that *raises* on over-cap instead of clamping
# (the local coder clamps; old remote workers don't). Clamping the per-pass
# value here makes the gateway resilient to that version skew. Matches the
# DeepCoder default cap; override (lower) if a worker runs a smaller cap.
_PER_PASS_MAX_TOKENS = int(os.environ.get("HUGPY_PER_PASS_MAX_TOKENS", "16000"))


def _overlap_len(prev_tail: str, seg: str) -> int:
    """Longest suffix of prev_tail that is also a prefix of seg (seam dedup).

    Exact match — verbatim repetition is by far the common case at a seam.
    """
    maxk = min(len(prev_tail), len(seg))
    for k in range(maxk, 0, -1):
        if prev_tail.endswith(seg[:k]):
            return k
    return 0


async def execute_chat_stream(*args, cancel_event=None, **kwargs):
    """Chat streaming with auto-continuation + seam-dedup.

    Drives execute_prompt_stream (one resolve→build→run pass) repeatedly: when a
    pass stops because it hit the token cap (finish_reason in _CONTINUE_ON), the
    partial answer is appended as an assistant turn and generation continues, so
    a response longer than any single token allowance still completes.

    Yields StreamEvents: TokenEvent for text, StatusEvent between continuation
    segments (and any provisioning/status passthrough from a worker), and a
    single terminal DoneEvent — or ErrorEvent. ``cancel_event`` stops it between
    and during passes.
    """
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)

    rid = prompt_kwargs.get("request_id")
    if not rid:
        import uuid
        rid = uuid.uuid4().hex
    prompt_kwargs["request_id"] = rid  # stable id across all passes

    # Normalize to a messages list so we can append assistant partials.
    messages = prompt_kwargs.get("messages")
    if not messages:
        messages = [{"role": "user", "content": prompt_kwargs.get("prompt", "")}]
    base = {k: v for k, v in prompt_kwargs.items() if k not in ("messages", "prompt")}

    # Clamp the per-pass output budget before any pass (local or worker relay):
    # continuation below covers totals beyond one pass, and this keeps an
    # over-cap value from ever reaching a worker that would raise on it.
    _mnt = base.get("max_new_tokens")
    if isinstance(_mnt, int) and _mnt > _PER_PASS_MAX_TOKENS:
        base["max_new_tokens"] = _PER_PASS_MAX_TOKENS

    # Caller-supplied continuation budget (ChatRequest.max_chunks). This loop
    # is where "Continue exactly where you left off" passes are minted, so an
    # explicit max_chunks MUST bound the TOTAL number of passes here — the
    # runner-level unbounded loop honoring req.max_chunks isn't enough, because
    # a bounded (max_new_tokens) pass that ends finish=length re-enters THIS
    # loop. max_chunks=1 therefore means: one pass, no auto-continuation (the
    # OpenAI /v1 max_tokens contract). Absent/invalid -> today's ceiling.
    _mc = base.get("max_chunks")
    try:
        _mc = int(_mc) if _mc is not None else None
    except (TypeError, ValueError):
        _mc = None
    total_passes = _MAX_CONTINUATIONS + 1
    if _mc and _mc > 0:
        total_passes = min(total_passes, _mc)

    full_text = ""
    # Aggregate token accounting across continuation passes: each inner pass's
    # done event may carry a usage dict (engine-reported or tokenizer-counted
    # by the runner); the request's real cost is their key-wise sum. Stays None
    # when no pass reported — consumers (/v1 usage object) must degrade.
    usage_totals: Optional[dict] = None

    def _merge_usage(part):
        nonlocal usage_totals
        if not isinstance(part, dict) or not part:
            return
        totals = dict(usage_totals or {})
        for k, v in part.items():
            if isinstance(v, int):
                totals[k] = (totals.get(k) or 0) + v
        usage_totals = totals or None

    for attempt in range(total_passes):
        if cancel_event is not None and cancel_event.is_set():
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason="cancelled")
            return
        if attempt > 0:
            yield StatusEvent(type="status", request_id=rid, stage="generate",
                              message=f"continuing (part {attempt + 1})…",
                              segment=attempt + 1)

        # Seam dedup: on a continuation pass, buffer the head of the segment
        # until we have _SEAM_WINDOW chars (or the pass ends), strip any overlap
        # with what we already emitted, then stream the rest live.
        is_cont = attempt > 0
        prev_tail = full_text[-_SEAM_WINDOW:] if is_cont else ""
        buffering = is_cont
        head = ""
        seg_text = ""
        finish = "stop"
        errored = False

        async for event in execute_prompt_stream(messages=messages,
                                                 cancel_event=cancel_event, **base):
            etype = getattr(event, "type", None)
            if etype == "token":
                text = getattr(event, "text", "") or ""
                seg_text += text
                if buffering:
                    head += text
                    if len(head) < _SEAM_WINDOW:
                        continue
                    k = _overlap_len(prev_tail, head)
                    emit, head, buffering = head[k:], "", False
                    if emit:
                        full_text += emit
                        yield TokenEvent(request_id=rid, text=emit)
                elif text:
                    full_text += text
                    yield TokenEvent(request_id=rid, text=text)
            elif etype == "done":
                finish = getattr(event, "finish_reason", None) or "stop"
                _merge_usage(getattr(event, "usage", None))
            elif etype == "error":
                # A pass that dies after text already streamed shouldn't turn a
                # partially-delivered answer into "[Error: ...]" in the chat.
                # This happens for real: a rambling model (e.g. a text-encoder
                # repack that never stops thinking) trips the engine mid-stream
                # (context overrun, decode assert) and the server aborts the
                # response body. End gracefully: an honest "truncated" status +
                # a normal done, with the failure logged. Only an error with
                # NOTHING delivered is surfaced as an error.
                if full_text.strip():
                    logger.warning(
                        "pass %s failed (%s); ending %s gracefully with %d "
                        "chars already streamed", attempt + 1,
                        getattr(event, "message", None) or "run failed",
                        rid, len(full_text))
                    yield StatusEvent(type="status", request_id=rid,
                                      stage="generate",
                                      message="engine stream ended early — "
                                              "answer truncated")
                    yield DoneEvent(request_id=rid, input_tokens=0,
                                    output_chunks=1, finish_reason="stop",
                                    usage=usage_totals)
                else:
                    yield ErrorEvent(request_id=rid,
                                     message=getattr(event, "message", None) or "run failed")
                errored = True
                break
            else:
                # status / provisioning passthrough (e.g. relayed from a worker)
                yield event

        if errored:
            return

        # Pass ended while still buffering (short segment): flush remainder
        # minus the seam overlap.
        if buffering:
            k = _overlap_len(prev_tail, head)
            emit = head[k:]
            if emit:
                full_text += emit
                yield TokenEvent(request_id=rid, text=emit)

        if finish not in _CONTINUE_ON:
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason=finish, usage=usage_totals)
            return
        if not seg_text.strip():
            # Hit the cap but produced nothing usable — stop to avoid a loop.
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason="stop", usage=usage_totals)
            return

        # Continue: append the partial assistant turn and re-prompt to keep going.
        messages = messages + [
            {"role": "assistant", "content": seg_text},
            {"role": "user", "content": "Continue exactly where you left off. "
                                        "Do not repeat any previous text."},
        ]

    # Exhausted the continuation budget.
    yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                    finish_reason="max_tokens", usage=usage_totals)


# ---------------------------------------------------------------------------
# Inspection / lifecycle — single definition each, no duplicates.
# ---------------------------------------------------------------------------

def loaded_model_keys() -> List[Tuple[str, str]]:
    """Which (model_key, task) pairs currently have a runner instantiated."""
    with _INSTANCES_LOCK:
        return sorted(_INSTANCES.keys())


# Model dirs are immutable once pulled, so walk each once and memoize by path.
_DISK_DETAIL_CACHE: Dict[str, dict] = {}
_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".onnx")


def _dir_size_detail(path: str) -> dict:
    """Recursively size a model dir: total on-disk bytes + weight-file bytes.

    The weight sum (safetensors/bin/…) is a coarse expected-VRAM proxy — what
    the framework will pull into memory, minus tokenizer/config/README noise.
    Cached by path; returns {} for a missing/unreadable dir (caller degrades)."""
    cached = _DISK_DETAIL_CACHE.get(path)
    if cached is not None:
        return cached
    total = weight = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    sz = os.path.getsize(os.path.join(root, fn))
                except OSError:
                    continue
                total += sz
                if fn.lower().endswith(_WEIGHT_EXTS):
                    weight += sz
    except OSError:
        return {}
    out: dict = {}
    if total:
        out["model_bytes"] = total          # frontend renders this as the row's size
    if weight:
        out["weight_bytes"] = weight         # expected-VRAM proxy
    if out:
        _DISK_DETAIL_CACHE[path] = out
    return out


def loaded_disk_detail() -> dict:
    """Per-loaded-model on-disk size for EVERY framework (transformers, diffusers,
    llama_cpp) — keyed by model_key.

    ``loaded_runner_detail`` only sizes in-process GGUF runners, so non-GGUF
    serving rows carried no size at all. This walks each loaded model's dir
    (resolved the same way the puller/loader do, via ``route_destination``) so
    every serving row gets a size server-side — no per-browser computation.
    GGUF rows are refined afterward by ``loaded_runner_detail`` (exact file
    bytes + layer split), which overlays this."""
    out: dict = {}
    try:
        from ...imports import route_destination
        from ...imports.config.main import get_model_config
    except Exception:
        return out
    for (mk, _task) in loaded_model_keys():
        if mk in out:
            continue
        try:
            cfg = get_model_config(mk, dict_return=True)
            path = route_destination(cfg)
        except Exception:
            continue
        d = _dir_size_detail(path)
        if d:
            out[mk] = d
    return out


def evict(model_key: str, task: Optional[str] = None) -> bool:
    """Drop runner(s) from the cache AND free the model's weights.

    If task is None, all task-variants for that model_key are dropped.
    Returns True if anything was evicted.

    Popping the wrapper alone is not enough: the llama.cpp singleton
    (_LLAMA_INSTANCES) holds the loaded weights, so without the cascade the
    VRAM/RAM stayed pinned after "unload" until the process died.
    """
    with _INSTANCES_LOCK:
        if task is not None:
            dropped = _INSTANCES.pop((model_key, task), None) is not None
        else:
            to_drop = [k for k in list(_INSTANCES) if k[0] == model_key]
            for k in to_drop:
                _INSTANCES.pop(k, None)
            dropped = bool(to_drop)
    try:
        from ..llama.runners.get import evict_llama_runner
        heavy = evict_llama_runner(model_key)
    except Exception:
        heavy = False
    return dropped or heavy


def clear() -> None:
    """Drop all cached runners (and their loaded weights)."""
    with _INSTANCES_LOCK:
        _INSTANCES.clear()
    try:
        from ..llama.runners.get import clear_llama_runners
        clear_llama_runners()
    except Exception:
        pass


def supported_task_keys() -> List[Tuple[str, str]]:
    from ..resolvers.model_resolver import _RUNNERS   # was: from .model_resolver import _RUNNERS
    return sorted(_RUNNERS.keys())
