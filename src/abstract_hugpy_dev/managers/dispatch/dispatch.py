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


def _get_or_build_runner(res: Resolution) -> Runner:
    """Cache-coherent runner lookup. Double-checked locking under the cache lock."""
    cached = _INSTANCES.get(res.cache_key)
    if cached is not None:
        return cached

    with _INSTANCES_LOCK:
        cached = _INSTANCES.get(res.cache_key)
        if cached is not None:
            return cached

        logger.info(
            "instantiating runner: model=%s task=%s class=%s framework=%s",
            res.model_key, res.task, res.runner_cls.__name__, res.framework,
        )
        instance = res.runner_cls(res.cfg)
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
                        finish_reason=getattr(result, "finish_reason", None) or "stop")
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

    full_text = ""
    for attempt in range(_MAX_CONTINUATIONS + 1):
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
                                    output_chunks=1, finish_reason="stop")
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
                            finish_reason=finish)
            return
        if not seg_text.strip():
            # Hit the cap but produced nothing usable — stop to avoid a loop.
            yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                            finish_reason="stop")
            return

        # Continue: append the partial assistant turn and re-prompt to keep going.
        messages = messages + [
            {"role": "assistant", "content": seg_text},
            {"role": "user", "content": "Continue exactly where you left off. "
                                        "Do not repeat any previous text."},
        ]

    # Exhausted the continuation budget.
    yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1,
                    finish_reason="max_tokens")


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
