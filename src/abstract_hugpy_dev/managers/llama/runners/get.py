from .src import *


class LocalEngineUnavailable(RuntimeError):
    """Raised when no in-process GGUF engine can be built (llama-cpp-python is
    not installed) and no HTTP slot is up. Carries a user-facing message so the
    chat route can surface actionable guidance instead of a raw import error."""


# ---------------------------------------------------------------------------
# Process-local singleton cache for the heavy GGUF runners.
# Keyed by model_key (str). The adapter wrappers in chat_runner share these.
# ---------------------------------------------------------------------------

_LLAMA_INSTANCES: Dict[str, "LlamaCppBaseRunner"] = {}
_LLAMA_LOCK = threading.Lock()


def get_llama_runner(model_key: str) -> "LlamaCppBaseRunner":
    """Get-or-build the singleton runner for a model_key.

    HTTP runner first (cheap probe); falls back to in-process Python.
    """
    if not isinstance(model_key, str):
        raise TypeError(
            f"get_llama_runner expects model_key: str, got {type(model_key).__name__}"
        )

    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.get(model_key)
        if runner is None:
            runner = _build_runner(model_key)
            _LLAMA_INSTANCES[model_key] = runner
        return runner


def _build_runner(model_key: str) -> "LlamaCppBaseRunner":
    # Cross-machine shard lead: a spill override set HUGPY_RPC_SERVERS, meaning
    # the allocator pooled remote GPUs for this load. The 0.3.x python binding
    # can't shard (no Llama(rpc_servers=…)), so spawn a managed
    # ``llama-server --rpc`` lead and talk to it over HTTP. Any failure falls
    # through to ordinary selection — sharding never breaks a request.
    from ...spill import rpc_servers as _rpc_servers, tensor_split as _tensor_split
    rpc = _rpc_servers()
    if rpc:
        base = ensure_shard_server(model_key, rpc, _tensor_split())
        if base:
            logger.info("get_llama_runner: shard lead (llama-server --rpc %s) for %s",
                        rpc, model_key)
            return LlamaCppRunner(model_key, base_url=base)
        logger.warning("get_llama_runner: shard lead unavailable for %s; "
                       "using ordinary selection", model_key)

    try:
        candidate = LlamaCppRunner(model_key)  # HTTP runner
        # quick probe — if the server isn't up this will throw
        with httpx.Client(timeout=2.0) as client:
            client.get(f"{candidate.base_url}/health").raise_for_status()
        logger.info("get_llama_runner: using HTTP runner for %s", model_key)
        return candidate
    except Exception:
        # Vision GGUFs: the in-process llama-cpp-python multimodal handler fails to
        # load the projector ("Failed to load mtmd context from <mmproj>"). A native
        # llama-server --mmproj loads it C-side and serves images correctly, so spawn/
        # reuse one and talk to it over HTTP. ensure_vision_server returns None for
        # non-vision models (no projector), so text models fall through unchanged.
        try:
            from .src.shard_server import ensure_vision_server
            vbase = ensure_vision_server(model_key)
        except Exception as exc:
            logger.warning("get_llama_runner: native vision server failed for %s: %s",
                           model_key, exc)
            vbase = None
        if vbase:
            logger.info("get_llama_runner: vision model %s -> native --mmproj server %s",
                        model_key, vbase)
            return LlamaCppRunner(model_key, base_url=vbase)
        logger.info(
            "get_llama_runner: HTTP unavailable, falling back to in-process for %s",
            model_key,
        )
        try:
            return LlamaCppPythonRunner(model_key)
        except ImportError as exc:
            # No local GGUF engine (llama-cpp-python missing) AND no HTTP slot.
            # Surface a clean, actionable error rather than letting a raw
            # ModuleNotFoundError escape to the client as a stack-trace string.
            logger.error("get_llama_runner: no local GGUF engine for %s (%s)", model_key, exc)
            raise LocalEngineUnavailable(
                "No local inference engine is available on this central "
                "(llama-cpp-python is not installed) and no model slot is running. "
                "Install the engine (pip install 'hugpy[engine]'), start a model slot, "
                "or bring a worker online for this model."
            ) from exc
