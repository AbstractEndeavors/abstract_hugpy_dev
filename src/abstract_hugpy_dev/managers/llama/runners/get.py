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


def evict_llama_runner(model_key: str) -> bool:
    """Drop the heavy singleton for ``model_key`` and free its weights.

    dispatch.evict only releases the adapter wrapper; the loaded ``Llama``
    handle (VRAM/RAM-resident weights) lives HERE in ``_LLAMA_INSTANCES``, so
    without this cascade the console's "free VRAM" evicted nothing and the
    memory stayed pinned until process death. Close the llama_cpp handle when
    it has one (HTTP runners don't — their memory belongs to the server).
    """
    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.pop(model_key, None)
    if runner is None:
        return False
    llm = getattr(runner, "llm", None)
    try:
        if llm is not None and hasattr(llm, "close"):
            llm.close()
    except Exception:
        logger.warning("evict_llama_runner: close() failed for %s", model_key,
                       exc_info=True)
    try:
        runner.llm = None
    except Exception:
        pass
    import gc
    gc.collect()
    return True


def clear_llama_runners() -> "list[str]":
    """Evict every heavy singleton (the clear() analog of evict_llama_runner)."""
    with _LLAMA_LOCK:
        keys = list(_LLAMA_INSTANCES)
    return [k for k in keys if evict_llama_runner(k)]


def loaded_runner_detail() -> dict:
    """Per-loaded-model load facts (for the worker heartbeat / console serving
    rows): file bytes, layers offloaded vs total, and the resulting GPU share —
    what the load actually DID, not an estimate. HTTP-backed runners (slots,
    shard leads) expose none of this; they report an empty detail."""
    import os as _os

    out: dict = {}
    with _LLAMA_LOCK:
        items = list(_LLAMA_INSTANCES.items())
    for key, r in items:
        d: dict = {}
        path = getattr(r, "model_path", None)
        if path:
            try:
                d["model_bytes"] = _os.path.getsize(path)
            except OSError:
                pass
            try:
                from ...spill import _gguf_layer_count
                d["total_layers"] = _gguf_layer_count(path)
            except Exception:
                pass
        thr = getattr(r, "n_threads", None)
        if thr is not None:
            d["threads"] = thr
        ngl = getattr(r, "n_gpu_layers", None)
        if ngl is not None:
            d["n_gpu_layers"] = ngl
            total = d.get("total_layers")
            if ngl == -1:
                d["gpu_pct"] = 100
            elif total:
                d["gpu_pct"] = round(100 * min(int(ngl), total) / total)
            elif int(ngl) == 0:
                d["gpu_pct"] = 0
        out[key] = d
    return out


def slot_backed_model_keys() -> "set[str]":
    """model_keys whose cached runner is an HTTP proxy (LlamaCppRunner with a
    base_url and no in-process ``llm`` handle) — a slot child or shard-lead
    server, NOT weights resident in this process. Callers use this to keep a
    slot-served model from ALSO being reported as an in-process resident (the
    kind='ram'/'loaded' double-count that flaps with the slot 'serving' row).
    Reads the already-built cache directly, so it never triggers a load."""
    with _LLAMA_LOCK:
        items = list(_LLAMA_INSTANCES.items())
    return {
        key for key, r in items
        if getattr(r, "base_url", None) and getattr(r, "llm", None) is None
    }


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


def _require_profile_ready(model_key: str) -> "dict | None":
    """Env-profiles (stage 1) gate for the slot seat path.

    Returns the profile decision ``{'name','state','bin',...}`` when a dependency
    profile is attributed to ``model_key`` and is READY (the caller ships
    ``opts['profile_bin']`` so the slot child launches from that venv), or None
    when the model has no profile (base behavior, untouched).

    RAISES ``LocalEngineUnavailable`` when a profile is attributed but NOT ready
    (materializing/error) — a profiled model must NEVER fall back to the shared
    venv (that would reintroduce the exact dependency conflict the profile
    isolates). The message is errors-as-data naming the profile + its state.
    """
    try:
        from ...serve import profiles
        resolve = profiles.resolve_model(model_key)
    except Exception:  # noqa: BLE001 — profiles unavailable -> base behavior
        return None
    if not resolve:
        return None
    if resolve.get("state") != "ready":
        detail = f": {resolve.get('error')}" if resolve.get("error") else ""
        raise LocalEngineUnavailable(
            f"model {model_key!r} is attributed to dependency profile "
            f"{resolve.get('name')!r}, which is {resolve.get('state')}{detail} — "
            "the model will seat once the profile finishes materializing; it will "
            "NOT fall back to the shared venv (that would reintroduce the "
            "dependency conflict the profile isolates)")
    return resolve


def _build_runner(model_key: str) -> "LlamaCppBaseRunner":
    # Per-box "never serve locally" policy: every branch below is a LOCAL serve
    # (slot spawn, native --mmproj/--rpc llama-server spawn, or in-process
    # llama-cpp-python weights in this process). A policy box hosts none of them —
    # fail fast with the actionable message instead of spawning/loading. Default
    # off === today's behavior; workers (which serve locally by design) never set
    # the flag. See managers.serve.policy.
    from ...serve.policy import no_local_serving, local_serving_error
    if no_local_serving():
        raise LocalEngineUnavailable(local_serving_error(
            model_key, detail="local GGUF serving is disabled on this box"))

    # Env-profiles (stage 1): if this model is attributed to a dependency profile,
    # resolve it up front. A non-ready profile RAISES here (propagates — never
    # caught by the slot-fallback try below, so a profiled model never silently
    # serves from the shared venv). A ready profile rides into the slot seat as
    # opts['profile_bin'] so the child launches from the profile venv.
    _profile = _require_profile_ready(model_key)

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
        # Slot-first local serving: a local load should land in a SLOT when one
        # is available — the model stays resident past this request (TTL, not
        # process lifetime), shows up in the console's Slots panel, and a load
        # that can't happen carries the slot agent's preflight REASON instead of
        # silently ballooning gunicorn RSS. This is the path dispatch's
        # worker-fallback takes too, so "served locally" now means "in a slot"
        # whenever the pool can take it. The slot's llama-server also loads
        # --mmproj itself, so vision models served this way see images.
        try:
            from ...serve.slots import SlotPool, slots_enabled
            if slots_enabled():
                # Resolve key→GGUF path HERE and hand it to the slot: models
                # registered from central live in THIS process's in-memory
                # registry, which a slot (separate process) never sees.
                opts = None
                try:
                    import os as _os
                    cfg = get_model_config(model_key)
                    mdir = ensure_model(model_key)
                    mpath = None
                    try:
                        from ...serve.overrides import resolve_override_gguf
                        mpath = resolve_override_gguf(model_key, mdir)
                    except Exception:
                        mpath = None
                    mpath = mpath or get_gguf_file(mdir, cfg)
                    if mpath:
                        opts = {"path": _os.fspath(mpath)}
                        # Ship the model's REAL context window too — the slot
                        # can't resolve cfg (separate process) and its bare
                        # default (4096) truncates long chats mid-stream
                        # ("ASGI callable returned without completing
                        # response" → incomplete chunked read up the chain).
                        try:
                            from ...serve.serve import _ctx_for
                            opts["ctx"] = int(_ctx_for(cfg, model_key))
                        except Exception:
                            pass
                    # Explicit per-model budgets (assign spill → env via the
                    # agent's _apply_spill) ride as per-load opts: slot
                    # processes were spawned earlier and never see env changes.
                    for env_name, key in (("HUGPY_GPU_MEM_GIB", "gpu_mem_gib"),
                                          ("HUGPY_CPU_MEM_GIB", "cpu_mem_gib"),
                                          ("HUGPY_N_CPU_MOE", "n_cpu_moe"),
                                          ("DEFAULT_LLAMA_THREADS", "threads")):
                        v = _os.environ.get(env_name)
                        if v:
                            opts = opts or {}
                            opts[key] = v
                    # An EXPLICIT GPU-layer designation (console 'Max GPU' = -1,
                    # 'CPU only' = off) must ride to the slot too: the slot is a
                    # separate process that never sees the agent's
                    # HUGPY_N_GPU_LAYERS, and its _build_cmd autofits (fail-closed
                    # to 0/CPU when it can't read the card), so without this a
                    # 'max GPU' GGUF silently serves on CPU. 'auto' is NOT shipped
                    # so slots keep autofitting from the VRAM free at seat time
                    # (slot 2 takes what slot 1 left).
                    _ngl = (_os.environ.get("HUGPY_N_GPU_LAYERS") or "").strip().lower()
                    if _ngl and _ngl != "auto":
                        try:
                            opts = opts or {}
                            opts["n_gpu_layers"] = 0 if _ngl in ("off", "cpu", "none") else int(_ngl)
                        except ValueError:
                            pass
                except Exception:
                    opts = None
                # Env-profiles (stage 1): a ready profile's venv bin dir rides to
                # the slot so its child launches from that venv (python-child
                # interpreter swap + PATH prefer). The slot is a separate process
                # that can't read the agent's settings, so it arrives as an opt.
                if _profile is not None and _profile.get("bin"):
                    opts = opts or {}
                    opts["profile_bin"] = _profile["bin"]
                    opts["profile"] = _profile.get("name")
                sep = SlotPool().endpoint_for(model_key, opts=opts)
                if sep:
                    logger.info("get_llama_runner: %s -> slot %s (loaded on demand)",
                                model_key, sep)
                    return LlamaCppRunner(model_key, base_url=sep)
                logger.warning("get_llama_runner: every slot is busy with another "
                               "model — %s falls back to in-process", model_key)
        except LocalEngineUnavailable:
            raise                     # profile refusal must never fall back
        except Exception as exc:
            # A profiled model must not silently drop to the shared-venv in-process
            # path when its slot seat fails — surface it as errors-as-data instead.
            if _profile is not None:
                raise LocalEngineUnavailable(
                    f"model {model_key!r} needs dependency profile "
                    f"{_profile.get('name')!r} but the slot seat failed "
                    f"({type(exc).__name__}: {exc}) — refusing the shared-venv "
                    "in-process fallback") from exc
            # endpoint_for surfaces the slot agent's preflight reason verbatim
            # (e.g. "needs ~42 GB RAM (all shards) but only 12 GB available").
            logger.warning("get_llama_runner: slot load refused for %s: %s — "
                           "falling back", model_key, exc)
        # Env-profiles (stage 1): a profiled model must seat in a slot from its
        # profile venv. If we reach here it's ready but unseatable (SLOT_COUNT=0 /
        # slots disabled, or every slot busy) — refuse rather than drop to the
        # shared-venv in-process/vision path. Stage 1 serves profiled models only
        # via slot children; the in-process consumer arrives with stage 3.
        if _profile is not None:
            raise LocalEngineUnavailable(
                f"model {model_key!r} needs dependency profile "
                f"{_profile.get('name')!r} (ready) but no slot could seat it "
                "(SLOT_COUNT=0 / slots disabled, or all slots busy) — stage 1 "
                "serves profiled models only via slot children; refusing the "
                "shared-venv in-process fallback")
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
