"""Slot supervisor — one generic, root-free model 'slot'.

A slot is a long-running service that owns a stable control port and runs ONE
``llama-server`` child for whatever model it's currently assigned. Assigning a
model autofits GPU layers from the VRAM still free at that moment, so a second
slot naturally takes whatever the first one left. The slot proxies the OpenAI
chat API to its child, so its own URL is a stable inference endpoint even across
child reloads.

No root / systemctl at request time: the app drives slots over HTTP (/load,
/unload, /status). You install N slot services ONCE (systemd template in
deploy/, or just run N of these), then the scheduler (:mod:`.slots`) assigns
models to them on demand.

Run one slot:

    SLOT_ID=1 SLOT_PORT=8101 python -m abstract_hugpy_dev.managers.serve.slot_agent

Env:
    SLOT_ID            label for this slot (default "1")
    SLOT_PORT          control + proxy port (default 8101)
    SLOT_CHILD_PORT    the llama-server child's port (default SLOT_PORT + 1000)
    SLOT_HOST          bind address (default 0.0.0.0)
    SLOT_ADVERTISE     host the scheduler should reach this slot on (default 127.0.0.1)
    MAIN_GPU           pin the child to this GPU index (sets CUDA_VISIBLE_DEVICES)
    SLOT_HEALTH_TIMEOUT seconds to wait for the child to come up (default 180)
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger("abstract_hugpy_dev.slot_agent")

SLOT_ID = os.environ.get("SLOT_ID", "1")
SLOT_HOST = os.environ.get("SLOT_HOST", "0.0.0.0")
_PORT_BASE = int(os.environ.get("SLOT_PORT_BASE", "8101"))


def _default_port() -> int:
    # slot N -> base + (N-1), matching slots.slot_urls(); lets a systemd
    # template set only SLOT_ID=%i and derive the port from it.
    try:
        return _PORT_BASE + (int(SLOT_ID) - 1)
    except (TypeError, ValueError):
        return _PORT_BASE


SLOT_PORT = int(os.environ.get("SLOT_PORT", str(_default_port())))
SLOT_CHILD_PORT = int(os.environ.get("SLOT_CHILD_PORT", str(SLOT_PORT + 1000)))
SLOT_ADVERTISE = os.environ.get("SLOT_ADVERTISE", "127.0.0.1")
MAIN_GPU = os.environ.get("MAIN_GPU")
# The FLOOR for the load hard-cap (back-compat: was the whole deadline). A cold
# load gets at LEAST this long regardless of size.
HEALTH_TIMEOUT = float(os.environ.get("SLOT_HEALTH_TIMEOUT", "180"))
# STALL window (slice 12): the honest failure signal is a load making NO forward
# progress (child RSS not growing / VRAM not dropping) for this long — NOT a
# blind clock that ignores size. A 45.9G gguf cold-load off NVMe legitimately
# exceeds 180s; killing it at the old flat deadline re-pages 46G every retry (the
# ae thrash loop). We fail on STALL, not on the clock.
STALL_TIMEOUT = float(os.environ.get("SLOT_LOAD_STALL_TIMEOUT", "60"))
# The GENEROUS-BUT-BOUNDED hard cap: a truly-wedged child that somehow keeps
# nudging RSS must still die eventually. Size-scaled — base + expected_bytes /
# assumed throughput — with the HEALTH_TIMEOUT floor. Assumed effective cold-load
# throughput (bytes/s): NVMe read + CUDA upload + repack; conservative so the cap
# is generous. Override the divisor via env for a slow-disk box.
_LOAD_THROUGHPUT_BPS = float(
    os.environ.get("SLOT_LOAD_THROUGHPUT_MBPS", "200")) * 1024 * 1024  # 200 MB/s
_HARD_CAP_MULT = float(os.environ.get("SLOT_LOAD_HARD_CAP_MULT", "3.0"))
# Bytes-of-progress that count as "real" movement between samples (filter noise).
_PROGRESS_EPSILON = 8 * 1024 * 1024   # 8 MiB
# Repeated-failure backoff (slice 12): after N genuine load failures for a model,
# refuse re-attempts for base × 2^(N-1), capped, so a doomed load doesn't re-page
# 46G on every request. Success clears the counter.
_LOAD_BACKOFF_BASE_S = float(os.environ.get("SLOT_LOAD_BACKOFF_BASE_S", "30"))
_LOAD_BACKOFF_MAX_S = float(os.environ.get("SLOT_LOAD_BACKOFF_MAX_S", "600"))


def _allowed_cpus():
    """Cores this slot's cgroup is confined to via systemd AllowedCPUs (kernel-
    enforced, un-escapable), or None when unconfined. Read-only; no root needed."""
    try:
        import subprocess
        out = subprocess.run(
            ["systemctl", "show", f"abstract-hugpy-slot@{SLOT_ID}.service",
             "-p", "AllowedCPUs", "--value"],
            capture_output=True, text=True, timeout=2)
        v = (out.stdout or "").strip()
        return v or None
    except Exception:
        return None


def _total_gguf_bytes(path):
    """Total on-disk size of a gguf, summing ALL shards when ``path`` is one shard
    of a split model (``…-00001-of-00004.gguf``). The resolver only ever hands us
    the FIRST shard, so a naive getsize under-counts a multi-shard model ~3-4x."""
    try:
        if not path or not os.path.isfile(path):
            return None
        import re
        import glob
        base = os.path.basename(path)
        m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base)
        if m:
            patt = f"{base[:m.start()]}-*-of-{m.group(1)}.gguf"
            shards = [s for s in glob.glob(os.path.join(os.path.dirname(path), patt))
                      if os.path.isfile(s)]
            if shards:
                return sum(os.path.getsize(s) for s in shards)
        return os.path.getsize(path)
    except Exception:
        return None


def _mem_available_bytes():
    """This node's reclaim-inclusive free RAM (MemAvailable)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _model_expected_bytes(model_key):
    """Rough 'total RAM required' for a model ~= its full GGUF size on disk (ALL
    shards summed). Denominator for the load-progress % AND the CPU preflight in
    _build_cmd. Approximate (mmap/repack land resident RAM a bit under file size)."""
    try:
        from .serve import _model_file_for, get_model_config
        cfg = get_model_config(model_key)
        return _total_gguf_bytes(_model_file_for(model_key, cfg))
    except Exception:
        return None


def _proc_rss_bytes(pid):
    """Resident RAM (bytes) of a pid — the slot's llama-server child footprint."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _proc_rss_detail(pid):
    """The HONEST resident-memory split of a pid from /proc/<pid>/status:
    ``{rss_anon_bytes, rss_file_bytes, rss_shmem_bytes}``.

    llama.cpp mmaps the GGUF, so VmRSS counts the FILE-BACKED pages of the
    weights as "resident" — reclaimable page cache, NOT pinned RAM. Measured on
    ae (Qwen3-Coder-Next, 17/48 offload): VmRSS 45.2G but RssAnon only 1.5G,
    RssFile 43.6G — the raw figure overstates true memory pressure ~28x.
    RssAnon is the honest pinned figure; RssFile is the mmap'd/cache share.

    Best-effort + Linux-only: ``{}`` on any read failure (an old kernel without
    the Rss* split, a vanished pid, a non-Linux box) — callers OMIT the fields
    rather than crash the heartbeat. ``rss_bytes`` (VmRSS) keeps its meaning
    unchanged for wire back-compat."""
    out = {}
    keys = {"RssAnon:": "rss_anon_bytes", "RssFile:": "rss_file_bytes",
            "RssShmem:": "rss_shmem_bytes"}
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                for pref, name in keys.items():
                    if line.startswith(pref):
                        out[name] = int(line.split()[1]) * 1024
                        break
    except Exception:
        return {}
    return out


def _cpus_to_hexmask(cpus: str) -> str:
    """Turn a cpu spec like "0-3" or "0,2,4" into llama.cpp's hex --cpu-mask."""
    bits = 0
    for part in str(cpus).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            for c in range(int(lo), int(hi) + 1):
                bits |= 1 << c
        else:
            bits |= 1 << int(part)
    return format(bits, "x") if bits else ""


def _central_url() -> str | None:
    """Where this slot can reach central to learn/pull request-time models.
    Agent-managed slots inherit WORKER_CENTRAL_URL from the agent's env;
    central's own systemd slots serve on-disk models and set neither."""
    return os.environ.get("WORKER_CENTRAL_URL") or os.environ.get("CENTRAL_URL")


def _resolve_cfg(model_key, central_url):
    """``get_model_config``, but if THIS (slot) process's static registry has
    never heard of the model — because central registered it at request time,
    and the slot is a separate process — learn it from central first via the
    SAME ensure-registered path the agent runs. Fixes slots 503'ing on
    request-time-provisioned models (the "op flux" saga)."""
    from .serve import get_model_config
    try:
        return get_model_config(model_key)
    except Exception:
        if not central_url:
            raise
    try:
        from ...worker_agent.provision import ensure_model_registered
        ensure_model_registered(model_key, central_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("slot %s: ensure_model_registered(%s) failed: %s",
                       SLOT_ID, model_key, exc)
    return get_model_config(model_key)   # raise cleanly if still unknown


def _ensure_present(model_key, central_url):
    """Pull a registered-but-absent model's files (request-time model) the same
    way the agent does, before we hand llama.cpp a path. Fast no-op when the
    model is already local (the common case: the agent pre-ensures before /load)."""
    try:
        from ...worker_agent.provision import ensure_model_present
        ensure_model_present(model_key, central_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("slot %s: ensure_model_present(%s) failed: %s",
                       SLOT_ID, model_key, exc)


def _effective_ngl(requested, auto):
    """Override-wins-over-autofit (k14). An EXPLICIT ``n_gpu_layers`` request WINS
    over the autofit — that is the lever the offload speed-cliff sweep (k7) needs:
    seat a GGUF at full offload, then relaunch it DOWN through decreasing layer
    counts. ``None``/absent => autofit, exactly as today.

    NOTE the sentinel: ``None`` is autofit; every integer is an override, INCLUDING
    the live console designations ``-1`` ("Max GPU" — force all layers) and ``0``
    ("CPU only"). ``-1`` is NOT an autofit alias here (managers.llama.runners.get
    ships ``n_gpu_layers=-1`` to the slot precisely to FORCE all layers; aliasing
    it to autofit would silently regress that path). The sweep therefore asks for
    autofit with ``None`` at the top of the ramp and explicit non-negative counts
    below it — it never needs ``-1`` to mean autofit."""
    return auto if requested is None else int(requested)


def _build_cmd(model_key, n_gpu_layers=None, ctx=None, threads=None, cpus=None,
               path=None, gpu_mem_gib=None, cpu_mem_gib=None, profile_bin=None):
    """argv for the child llama-server + the resolved (ngl, ctx, threads, cpus).

    ``profile_bin`` (env-profiles stage 1): when the agent seats a model
    attributed to a dependency profile, it hands the profile venv's bin dir here.
    The PYTHON-launched child (``python -m llama_cpp.server`` — the fallback when
    no native llama-server binary exists) is then spawned from THAT venv's
    interpreter instead of the agent's, isolating the model's extra deps at the
    process seam. The native-binary child is unaffected in argv (its binary is
    resolved by the engine resolver); its PATH still prefers the profile bin via
    the child env (see ``Slot.load``), so a profile-shipped binary would win.
    """
    from .serve import (
        _model_file_for, _ctx_for, get_model_config,
        LLAMA_SERVER_BIN, DEFAULT_LLAMA_THREADS,
    )
    from ..spill import autofit_gpu_layers, vision_projector_bytes

    cfg = None
    if path and os.path.isfile(path):
        # Caller-resolved path (the worker agent registers central's models
        # IN-MEMORY, which a slot — a separate process — never sees; the agent
        # therefore resolves key→file itself and hands us the path).
        pass
    else:
        central_url = _central_url()
        cfg = _resolve_cfg(model_key, central_url)      # ensure-registered fallback
        path = _model_file_for(model_key, cfg)
        # Registered but files absent/partial (request-time model) — pull them
        # the same way the agent does before spawning llama.cpp.
        if central_url and (not path or not os.path.isfile(path)
                            or os.path.getsize(path) == 0):
            _ensure_present(model_key, central_url)
            path = _model_file_for(model_key, cfg)
    # Existence AND non-empty: a 0-byte or truncated GGUF (interrupted pull)
    # passes an isfile check but SIGILLs llama.cpp's native loader on spawn —
    # fail cleanly here instead of core-dumping the child.
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"{model_key}: no usable GGUF on disk (resolved {path!r}) — missing "
            "or empty; refusing to spawn llama.cpp (would SIGILL)")

    # Serve from the box-local NVMe HOT-CACHE tier when this model is warmed there
    # (NVMe-fast); otherwise this kicks a background shared->hot promotion and
    # returns the shared path for this (cold) load, so the next load is fast.
    # Never blocks. The hot_cache tier (HUGPY_HOT_CACHE_ROOT) is the general
    # main-catalog mechanism; the legacy model_cache (HUGPY_MODEL_CACHE) is kept
    # as a fallback so a box still on the old env is not regressed. Neither env
    # set -> path is returned unchanged (byte-identical behaviour).
    try:
        from . import hot_cache
        if hot_cache.enabled():
            path = hot_cache.use(path)
        else:
            from . import model_cache
            path = model_cache.use(path)
    except Exception as exc:
        logger.warning("hot-cache unavailable (%s); loading from %s", exc, path)

    # Autofit from the VRAM free RIGHT NOW, so later slots take what's left.
    # An explicit per-model VRAM budget (gpu_mem_gib) caps what autofit may
    # plan with — the model's contract, not the card's whole remainder.
    free_cap = None
    if gpu_mem_gib not in (None, ""):
        try:
            free_cap = int(float(gpu_mem_gib) * 2**30)
        except (TypeError, ValueError):
            free_cap = None
    # Vision GGUF: the mmproj/CLIP projector loads onto the GPU beside the
    # offloaded layers, so reserve its VRAM BEFORE fitting layers — otherwise on
    # an 8 GB card autofit plans "all layers" against the model file alone and the
    # child then OOMs when the ~1.3 GB projector lands on top. 0 for text models
    # (byte-identical to before).
    _mmproj_reserve = vision_projector_bytes(path)
    if free_cap is not None:
        from ..spill import free_vram_bytes as _fvb
        fv = _fvb()
        auto = autofit_gpu_layers(path, free_vram=min(fv, free_cap) if fv else free_cap,
                                  extra_reserve_bytes=_mmproj_reserve)
    else:
        auto = autofit_gpu_layers(path, extra_reserve_bytes=_mmproj_reserve)
    ngl = _effective_ngl(n_gpu_layers, auto)
    ctx = int(ctx) if ctx else (_ctx_for(cfg, model_key) if cfg is not None else 4096)
    threads = int(threads) if threads else DEFAULT_LLAMA_THREADS
    cpus = str(cpus).strip() if cpus not in (None, "") else None

    # Preflight: when nothing can offload to GPU (auto<=0 — e.g. no GPU on this
    # node) the weights are CPU-RAM-resident, so a model bigger than free RAM will
    # OOM mid-load. Sum ALL shards (the resolved path is only shard 1) and fail
    # fast with a clear message instead of letting RSS climb into an OOM 500.
    # Per-model RAM budget (cpu_mem_gib): the CPU-resident share must fit the
    # model's OWN allowance, not just whatever the box has free.
    if cpu_mem_gib not in (None, ""):
        try:
            from ..spill import cpu_resident_bytes
            ram_budget = float(cpu_mem_gib) * 1e9
            ngl_eff = ngl                      # the already-resolved effective ngl
            need_cpu = cpu_resident_bytes(path, int(ngl_eff)) or 0
            if need_cpu > ram_budget:
                raise RuntimeError(
                    f"{model_key}: CPU-resident share ~{need_cpu / 1e9:.1f} GB exceeds "
                    f"this model's RAM budget ({float(cpu_mem_gib):.1f} GB) — raise the "
                    "budget, offload more layers, or pick a smaller quant")
        except (TypeError, ValueError):
            pass

    if ngl <= 0:                             # effective ngl, not raw autofit —
        # an explicit n_gpu_layers=-1 ("max GPU") that overrode a broken auto=0
        # is GPU-resident and must skip the CPU-RAM refusal below.
        need = _total_gguf_bytes(path)
        avail = _mem_available_bytes()
        if avail:
            # Honor the operator RAM reserve (HUGPY_RAM_RESERVE_GIB) so slot
            # loads leave headroom for processes central can't see.
            from ..spill import ram_reserve_bytes
            avail = max(0, avail - ram_reserve_bytes())
        if need and avail and need > avail * 0.95:
            raise RuntimeError(
                f"{model_key}: needs ~{need / 1e9:.1f} GB RAM (all shards) but only "
                f"{avail / 1e9:.1f} GB budgetable (after reserve) with no GPU offload "
                f"on this node — free RAM (recycle the API worker) or pick a smaller quant")

    import shutil
    server_bin = LLAMA_SERVER_BIN if (LLAMA_SERVER_BIN and (
        os.path.isfile(LLAMA_SERVER_BIN) or shutil.which(LLAMA_SERVER_BIN))) else None

    if server_bin:
        argv = [
            server_bin, "-m", path,
            "--host", "127.0.0.1", "--port", str(SLOT_CHILD_PORT),
            "--n-gpu-layers", str(ngl), "-c", str(ctx), "-t", str(threads),
        ]
        if cpus:
            # Soft pin via llama.cpp's own affinity (taskset is escaped by llama.cpp's
            # per-thread sched_setaffinity). For HARD, kernel-enforced dedication use
            # hugpy-slot-cpus -> cgroup AllowedCPUs. "0-3" / "0,2,4" -> hex mask.
            mask = _cpus_to_hexmask(cpus)
            if mask:
                argv += ["--cpu-mask", mask, "--cpu-strict", "1"]
        # Vision GGUF: load the multimodal projector so /v1/chat/completions accepts
        # image_url content. No-op for text models (no projector beside the model).
        from ...imports.src.utils import find_mmproj
        mmproj = find_mmproj(path)
        if mmproj:
            argv += ["--mmproj", mmproj]
            logger.info("slot %s: vision model — loading projector %s", SLOT_ID, mmproj)
    else:
        # No C++ llama-server on this box (typical for WORKERS): fall back to
        # the OpenAI-compatible server inside llama-cpp-python — the engine
        # every node already has, so there's no binary to distribute or build.
        # Same /v1 surface, so the slot proxy needs no changes. No --cpu-mask
        # equivalent (threads + the unit's cgroup govern CPU); vision models
        # stay on the native/in-process path (no --mmproj here).
        #
        # Vision GGUF: REFUSE rather than seat. llama_cpp.server cannot load
        # the mmproj projector, so a seated vision model would answer every
        # image turn text-blind with no error anywhere. The raised reason
        # propagates verbatim to central's slot-refusal log and load_reports,
        # and get_llama_runner falls through to the native/in-process path.
        from ...imports.src.utils import find_mmproj
        if find_mmproj(path):
            raise RuntimeError(
                f"{model_key}: vision model (mmproj sidecar present) but this "
                "box has no native llama-server (LLAMA_SERVER_BIN) — the "
                "llama_cpp.server fallback cannot load the projector, images "
                "would be silently ignored. Install/point to a llama-server "
                "build (`hugpy install-engine`) to seat vision models.")
        import sys as _sys
        # Env-profiles (stage 1): launch this python child from the profile
        # venv's interpreter when one is attributed (raises errors-as-data if the
        # profile venv python is missing — never a silent shared-venv fallback).
        from . import profiles as _profiles
        child_py = _profiles.child_python(profile_bin, _sys.executable)
        if profile_bin:
            logger.info("slot %s: %s uses dependency profile venv %s (child %s)",
                        SLOT_ID, model_key, profile_bin, child_py)
        argv = [
            child_py, "-m", "llama_cpp.server",
            "--model", path,
            "--host", "127.0.0.1", "--port", str(SLOT_CHILD_PORT),
            "--n_gpu_layers", str(ngl), "--n_ctx", str(ctx),
            "--n_threads", str(threads),
        ]
        if cpus:
            logger.info("slot %s: cpu pin %r ignored in llama_cpp.server mode",
                        SLOT_ID, cpus)
    # The model's TOTAL layer count (GGUF header block_count) — the denominator
    # the console needs to render "17/48 layers" instead of "17/undefined". Read
    # here (the one place the slot holds the resolved file path) via the existing
    # spill reader; best-effort None keeps the field omit-when-unset downstream.
    try:
        from ..spill import _gguf_layer_count
        total_layers = _gguf_layer_count(path)
    except Exception:  # noqa: BLE001 — never block a load on header metadata
        total_layers = None
    return (argv, ngl, ctx, threads, cpus,
            ("binary" if server_bin else "python"), total_layers)


class Slot:
    """Owns at most one llama-server child."""

    def __init__(self):
        self.model_key = None
        self.proc = None
        self.ngl = None
        self.ctx = None
        self.threads = None
        self.cpus = None
        self.gpu = None
        self.profile_bin = None      # env-profiles (stage 1): the profile venv
        # bin dir this model's child launches from (None = shared venv default).
        self.expected_bytes = None
        # GGUF header block_count of the seated model (None = unknown/non-GGUF):
        # the "of 48" in the console's offload readout.
        self.total_layers = None
        self.loaded_at = 0.0
        self.last_used = 0.0
        # Free VRAM sampled at the start of the CURRENT load (slice 12): the
        # baseline the stall-detector measures VRAM-consumed against.
        self._load_free_vram_at_start = None
        # Repeated-failure backoff (slice 12): consecutive genuine load failures
        # for a model_key + when the last one happened, so per-request re-attempts
        # don't hammer a doomed 46G re-page. Plus the last honest failure reason.
        self._load_failures: dict = {}          # model_key -> consecutive count
        self._load_backoff_until: dict = {}      # model_key -> epoch (retry after)
        self.last_load_error: "str | None" = None
        self.lock = threading.Lock()
        # llama_cpp.server (python child) cannot take CONCURRENT streaming
        # requests — overlapping streams kill BOTH with an incomplete chunked
        # read (observed 2026-07-02: the media console's chat + side-calls).
        # The proxy serializes streams through this gate for python children;
        # the C++ llama-server handles parallel slots natively and skips it.
        self.child_kind = None
        self.stream_gate = threading.Semaphore(1)
        # Requests currently streaming through the proxy. The python child is
        # single-threaded: while it GENERATES it cannot answer a health probe,
        # so a probe-only healthy() reported False mid-request and (a) the
        # console flipped the slot to "loading" every time the model was USED,
        # (b) an overlapping proxy call 503'd instead of waiting on the gate.
        # Busy == alive by definition — the request is being served right now.
        self.inflight = 0
        self.child_base = f"http://127.0.0.1:{SLOT_CHILD_PORT}"

    # -- health ------------------------------------------------------------
    def _child_alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def healthy(self) -> bool:
        if not self._child_alive():
            return False
        if self.inflight > 0:
            # Mid-generation: the (python) child won't answer a probe, but it
            # is literally serving a request — that's the healthiest it gets.
            return True
        import httpx
        try:
            if httpx.get(self.child_base + "/health", timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        try:
            # llama_cpp.server (the python fallback child) has no /health;
            # /v1/models answering is its liveness signal.
            return httpx.get(self.child_base + "/v1/models", timeout=2.0).status_code == 200
        except Exception:
            return False

    def _self_heal(self):
        """Clear a WEDGED claim: child dead but model_key still set.

        Without this the slot reports model_key + healthy=False FOREVER — the
        console renders that as a permanent "loading" pill and endpoint_for
        waits its full timeout on a corpse. (Observed 2026-07-03: slot 1's
        reload died and flux showed "loading" indefinitely while chats were
        actually served in-process.) Non-blocking lock: a load() mid-flight
        also has model_key set with the child coming up — its own failure
        path cleans up, so never race it."""
        if self.model_key is None or self._child_alive():
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.model_key is not None and not self._child_alive():
                logger.warning("slot %s: child died while claiming %s — "
                               "clearing the stale claim", SLOT_ID, self.model_key)
                self.model_key = self.ngl = self.ctx = None
                self.threads = self.cpus = self.gpu = self.expected_bytes = None
                self.total_layers = None
                self.profile_bin = None
                self.proc = None
        finally:
            self.lock.release()

    def status(self) -> dict:
        from ..spill import free_vram_bytes
        self._self_heal()
        out = {
            "slot_id": SLOT_ID,
            "control_port": SLOT_PORT,
            "child_port": SLOT_CHILD_PORT,
            "endpoint": f"http://{SLOT_ADVERTISE}:{SLOT_PORT}",
            "model_key": self.model_key,
            "healthy": self.healthy(),
            "busy": self.inflight > 0,
            "n_gpu_layers": self.ngl,
            # GGUF block_count of the seated model — the "of N" for the console's
            # "17/48 layers". None for non-GGUF / an unreadable header (getattr:
            # an instance created before this field existed must not 500 /status).
            "total_layers": getattr(self, "total_layers", None),
            "ctx": self.ctx,
            "threads": self.threads,
            "cpus": self.cpus,
            "gpu": self.gpu,
            "profile_bin": self.profile_bin,   # env-profiles: child's venv, or None
            "allowed_cpus": _allowed_cpus(),   # kernel-enforced dedicated cores
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "free_vram_bytes": free_vram_bytes(),
            "rss_bytes": _proc_rss_bytes(self.proc.pid) if self._child_alive() else 0,
            # The child llama-server/llama_cpp.server PID: this is the process
            # that actually HOLDS the model's VRAM (the slot supervisor python
            # only carries a ~tens-of-MiB CUDA context). The worker agent joins
            # this against nvidia-smi's per-process accounting to report the
            # slot occupant's REAL VRAM (its type/ngl guess is not ground truth).
            "child_pid": self.proc.pid if self._child_alive() else None,
            "expected_bytes": self.expected_bytes,
            # The last honest load-failure reason + backoff (slice 12), so the
            # console can show WHY a model's row is degraded/retrying instead of a
            # silent tight loop. None once a load succeeds.
            "last_load_error": self.last_load_error,
        }
        # Honest RSS split (omit-when-unset): rss_bytes stays VmRSS verbatim for
        # wire back-compat, while rss_anon_bytes is the truly-pinned RAM and
        # rss_file_bytes the mmap'd-GGUF page cache VmRSS also counts (~28x
        # overstatement observed on ae). Absent entirely when /proc can't say.
        if self._child_alive():
            out.update(_proc_rss_detail(self.proc.pid))
        return out

    # -- lifecycle ---------------------------------------------------------
    def load(self, model_key, n_gpu_layers=None, ctx=None, threads=None,
             cpus=None, gpu=None, path=None, gpu_mem_gib=None,
             cpu_mem_gib=None, profile_bin=None, force=False) -> dict:
        with self.lock:
            # ``force`` (k14 relaunch): a relaunch re-seats the SAME model with a
            # NEW spec (e.g. a swept-down n_gpu_layers), so it must bypass the
            # already-serving short-circuit and actually respawn the child —
            # otherwise a same-model relaunch is a silent no-op and the sweep can
            # never change the offload depth.
            if not force and self.model_key == model_key and self.healthy():
                self.last_used = time.time()
                return self.status()

            # BACKOFF (slice 12): after repeated GENUINE load failures for this
            # model, refuse a re-attempt for a growing window instead of hammering
            # a doomed 46G re-page on every incoming request. Cleared on success.
            until = self._load_backoff_until.get(model_key, 0.0)
            if time.time() < until:
                raise RuntimeError(
                    f"slot {SLOT_ID}: {model_key} in load-backoff for "
                    f"{until - time.time():.0f}s after "
                    f"{self._load_failures.get(model_key, 0)} failed attempt(s)"
                    + (f" — {self.last_load_error}" if self.last_load_error else ""))

            self._kill()
            self.profile_bin = profile_bin or None
            (argv, self.ngl, self.ctx, self.threads, self.cpus,
             self.child_kind, self.total_layers) = _build_cmd(
                model_key, n_gpu_layers, ctx, threads, cpus, path=path,
                gpu_mem_gib=gpu_mem_gib, cpu_mem_gib=cpu_mem_gib,
                profile_bin=self.profile_bin)
            # per-load GPU pin overrides the slot's MAIN_GPU default
            self.gpu = gpu if gpu not in (None, "") else MAIN_GPU
            self.expected_bytes = _model_expected_bytes(model_key)
            logger.info("slot %s loading %s (ngl=%s ctx=%s threads=%s cpus=%s gpu=%s): %s",
                        SLOT_ID, model_key, self.ngl, self.ctx, self.threads,
                        self.cpus, self.gpu, " ".join(argv))

            env = dict(os.environ)
            if self.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            # A from-source llama-server links against sibling .so files under the
            # engine dir (libllama/libggml/…). Prepend those to the child's
            # LD_LIBRARY_PATH so it loads without a unit-level env hack (the ae
            # 2026-07-06 manual fix, now derived from HUGPY_ENGINE_DIR in code).
            # Additive + Linux-only + no-op unless the engine dir is overridden.
            try:
                from ...engine.resolve import ld_library_path_with_engine
                _ld = ld_library_path_with_engine(env.get("LD_LIBRARY_PATH"))
                if _ld:
                    env["LD_LIBRARY_PATH"] = _ld
            except Exception:  # noqa: BLE001 — never block a load on lib-path derivation
                pass
            # Env-profiles (stage 1): activate the profile venv for the CHILD only
            # — prepend its bin dir to PATH (a profile-shipped binary wins) and set
            # VIRTUAL_ENV. The agent process is never touched; only this child runs
            # from the profile. No-op without a profile.
            try:
                from . import profiles as _profiles
                env = _profiles.child_env(env, self.profile_bin)
            except Exception:  # noqa: BLE001 — never block a load on env derivation
                pass
            self.proc = subprocess.Popen(argv, env=env)
            self.model_key = model_key
            self.loaded_at = self.last_used = time.time()

            if not self._wait_healthy():
                self._kill()
                self.model_key = None
                # Record the genuine failure and arm exponential backoff so
                # per-request re-attempts don't thrash (slice 12). The message now
                # names STALL vs hard-cap (the honest reason), not a flat clock.
                n = self._load_failures.get(model_key, 0) + 1
                self._load_failures[model_key] = n
                backoff = min(_LOAD_BACKOFF_BASE_S * (2 ** (n - 1)),
                              _LOAD_BACKOFF_MAX_S)
                self._load_backoff_until[model_key] = time.time() + backoff
                self.last_load_error = (
                    f"did not become healthy (stall/hard-cap); attempt {n}, "
                    f"backing off {backoff:.0f}s")
                raise RuntimeError(
                    f"slot {SLOT_ID}: {model_key} {self.last_load_error}")
            # SUCCESS — clear the failure counters + backoff for this model.
            self._load_failures.pop(model_key, None)
            self._load_backoff_until.pop(model_key, None)
            self.last_load_error = None
            logger.info("slot %s ready: %s on %s", SLOT_ID, model_key, self.child_base)
            return self.status()

    def _hard_cap_s(self) -> float:
        """Size-scaled generous-but-bounded hard cap. base (HEALTH_TIMEOUT) +
        expected_bytes / assumed throughput, × a safety multiplier, floored at
        HEALTH_TIMEOUT. A 46G model at 200 MB/s ~= 235s of transfer alone, so the
        cap lands well above a legitimate cold load while still bounding a truly-
        wedged child. Unknown size -> the flat floor (back-compat)."""
        exp = self.expected_bytes
        if not exp:
            return HEALTH_TIMEOUT
        transfer_s = float(exp) / max(_LOAD_THROUGHPUT_BPS, 1.0)
        return max(HEALTH_TIMEOUT, (HEALTH_TIMEOUT + transfer_s) * _HARD_CAP_MULT)

    def _load_progress_bytes(self) -> int:
        """A monotonic-ish PROGRESS signal for an in-flight load: the child's
        resident RAM (weights paging in) PLUS the VRAM consumed since we started
        (free VRAM DROPPING as layers upload). Either one growing == the load is
        moving. Best-effort: 0 on any read failure (a run of 0s reads as a stall,
        which is the safe conservative verdict)."""
        rss = 0
        try:
            if self._child_alive():
                rss = _proc_rss_bytes(self.proc.pid) or 0
        except Exception:  # noqa: BLE001
            rss = 0
        vram_used = 0
        try:
            from ..spill import free_vram_bytes
            fv = free_vram_bytes()
            if fv is not None and self._load_free_vram_at_start is not None:
                vram_used = max(0, self._load_free_vram_at_start - fv)
        except Exception:  # noqa: BLE001
            vram_used = 0
        return int(rss) + int(vram_used)

    def _wait_healthy(self) -> bool:
        """Wait for the child to answer /health, failing on STALL not on a blind
        clock (slice 12). While the child is alive AND making forward progress
        (RSS growing / VRAM filling), keep waiting — a big cold load is SLOW, not
        broken. Kill only when progress stalls for STALL_TIMEOUT, or when the
        generous size-scaled hard cap is blown (a truly-wedged child must die)."""
        try:
            from ..spill import free_vram_bytes
            self._load_free_vram_at_start = free_vram_bytes()
        except Exception:  # noqa: BLE001
            self._load_free_vram_at_start = None
        start = time.time()
        hard_cap = self._hard_cap_s()
        last_progress = self._load_progress_bytes()
        last_progress_ts = start
        while True:
            if not self._child_alive():
                return False                     # child exited -> real failure
            if self.healthy():
                return True                      # up and answering
            now = time.time()
            if now - start >= hard_cap:
                logger.warning("slot %s: load of %s blew the %.0fs hard cap "
                               "(size-scaled) — treating as wedged",
                               SLOT_ID, self.model_key, hard_cap)
                return False
            cur = self._load_progress_bytes()
            if cur - last_progress >= _PROGRESS_EPSILON:
                last_progress = cur              # real movement — reset the stall clock
                last_progress_ts = now
            elif now - last_progress_ts >= STALL_TIMEOUT:
                logger.warning("slot %s: load of %s STALLED — no forward progress "
                               "(RSS+VRAM) for %.0fs (last=%s); killing the wedged "
                               "child", SLOT_ID, self.model_key, STALL_TIMEOUT,
                               cur)
                return False
            time.sleep(1.0)

    def unload(self) -> dict:
        # Interrupt any in-progress load first: killing the child makes a blocking
        # _wait_healthy bail and release the lock, so /unload returns promptly
        # instead of waiting out the load's (up to 180s) health timeout.
        self._kill()
        with self.lock:
            self._kill()
            self.model_key = self.ngl = self.ctx = None
            self.threads = self.cpus = self.gpu = self.expected_bytes = None
            self.total_layers = None
            self.profile_bin = None
            return self.status()

    def relaunch(self, n_gpu_layers=None, ctx=None) -> dict:
        """Re-seat the CURRENTLY-loaded model with a new offload depth / context —
        the lever the k7 offload speed-cliff sweep needs (seat at full offload,
        then relaunch DOWN through decreasing ``n_gpu_layers``, measuring tok/s at
        each step). This is the ONLY way to change a live slot child's ngl: the
        slot child is a spawned process whose ngl is fixed at launch, so a change
        means STOP-then-RESPAWN — which is exactly what this does (via a forced
        load: SIGTERM->wait->SIGKILL of the old child, then a fresh spawn). It also
        answers the ae "slot-child PID never recycles" blocker: relaunch replaces
        the child under a NEW pid every time, no worker restart required.

        The current model_key + its threads/cpus/gpu/profile are preserved; only
        ``n_gpu_layers`` and ``ctx`` are overridden (``None`` for either keeps the
        current value — ctx from the live child, ngl re-autofit). The resulting
        allocation is reported HONESTLY (the echoed ``n_gpu_layers`` is what the
        fresh child actually launched with, i.e. ``self.ngl`` after the respawn —
        not merely what was requested)."""
        mk = self.model_key
        if mk is None:
            raise RuntimeError(
                f"slot {SLOT_ID}: no model loaded — nothing to relaunch")
        requested_ngl = n_gpu_layers
        # A deliberate operator relaunch must not be refused by a stale load
        # backoff armed by an earlier failure of this model — clear it so the
        # forced re-seat actually runs.
        self._load_failures.pop(mk, None)
        self._load_backoff_until.pop(mk, None)
        result = self.load(
            mk, n_gpu_layers=requested_ngl,
            ctx=ctx if ctx is not None else self.ctx,
            threads=self.threads, cpus=self.cpus, gpu=self.gpu,
            gpu_mem_gib=None, cpu_mem_gib=None,
            profile_bin=self.profile_bin, force=True)
        # Surface the request alongside the honest launched value so the caller
        # can see requested-vs-effective at a glance (self.ngl / status carries
        # the measured launch value).
        result = dict(result)
        result["relaunched"] = True
        result["requested_n_gpu_layers"] = requested_ngl
        return result

    def _kill(self):
        if self._child_alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None


def build_app():
    from flask import Flask, request, jsonify, Response

    slot = Slot()
    app = Flask(f"abstract_hugpy_slot_{SLOT_ID}")

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "slot_id": SLOT_ID})

    @app.route("/status")
    def status():
        return jsonify(slot.status())

    @app.route("/load", methods=["POST"])
    def load():
        body = request.get_json(silent=True) or {}
        if not body.get("model_key"):
            return jsonify({"error": "missing model_key"}), 400
        # Per-box "never serve locally" policy: a slot on a policy box must not
        # spawn a llama-server child even on a direct /load (the scheduler
        # already stops routing here, but this self-protects against any direct
        # caller). Set HUGPY_NO_LOCAL_SERVING=true in the slot unit's env to
        # arm it. Default off === today's behavior; workers never set the flag.
        from .policy import no_local_serving, local_serving_error
        if no_local_serving():
            return jsonify({"error": local_serving_error(
                body.get("model_key"),
                detail="slot serving disabled on this box")}), 403
        try:
            return jsonify(slot.load(body["model_key"], body.get("n_gpu_layers"),
                                     body.get("ctx"), body.get("threads"),
                                     body.get("cpus"), body.get("gpu"),
                                     path=body.get("path"),
                                     gpu_mem_gib=body.get("gpu_mem_gib"),
                                     cpu_mem_gib=body.get("cpu_mem_gib"),
                                     profile_bin=body.get("profile_bin")))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/unload", methods=["POST"])
    def unload():
        return jsonify(slot.unload())

    @app.route("/relaunch", methods=["POST"])
    def relaunch():
        # k14: re-seat the CURRENT model with a new offload depth / context so the
        # k7 offload speed-cliff sweep can measure tok/s per n_gpu_layers. Body:
        # {"n_gpu_layers"?: int, "ctx"?: int} — omit either to keep it. A slot with
        # no model loaded is a 409 (nothing to relaunch), never a 500.
        body = request.get_json(silent=True) or {}
        from .policy import no_local_serving, local_serving_error
        if no_local_serving():
            return jsonify({"error": local_serving_error(
                slot.model_key, detail="slot serving disabled on this box")}), 403
        if slot.model_key is None:
            return jsonify({"error": f"slot {SLOT_ID} has no model loaded "
                            "to relaunch"}), 409
        try:
            return jsonify(slot.relaunch(body.get("n_gpu_layers"),
                                         body.get("ctx")))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/v1/<path:sub>", methods=["POST", "GET"])
    def proxy(sub):
        """Forward the OpenAI API to the child, streaming the response back."""
        import httpx
        if not slot.healthy():
            return jsonify({"error": f"slot {SLOT_ID} has no model loaded"}), 503
        slot.last_used = time.time()

        url = f"{slot.child_base}/v1/{sub}"
        body = request.get_data()
        headers = {k: v for k, v in request.headers
                   if k.lower() not in ("host", "content-length")}

        # Serialize python-child requests end-to-end (see stream_gate note):
        # the gate is held for the WHOLE response lifetime and released in the
        # generator's finally, so an overlapping caller waits instead of
        # crashing both streams.
        gated = (slot.child_kind == "python")
        if gated:
            slot.stream_gate.acquire()
        slot.inflight += 1
        try:
            # Bound connect/write/pool so a dead or wedged child fails send()
            # fast instead of hanging forever while holding the stream gate +
            # inflight counter — that leak wedged EVERY later request
            # (busy=True, model=None). Read stays unbounded: a streamed
            # generation legitimately runs for minutes.
            client = httpx.Client(
                timeout=httpx.Timeout(None, connect=10.0, write=30.0, pool=10.0))
            upstream = client.send(
                client.build_request(request.method, url, content=body, headers=headers),
                stream=True,
            )
        except Exception:
            slot.inflight -= 1
            if gated:
                slot.stream_gate.release()
            raise

        def generate():
            try:
                for chunk in upstream.iter_raw():
                    yield chunk
            finally:
                upstream.close()
                client.close()
                slot.inflight -= 1
                if gated:
                    slot.stream_gate.release()

        return Response(generate(), status=upstream.status_code,
                        content_type=upstream.headers.get("content-type", "application/json"))

    return app, slot


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app, _slot = build_app()
    logger.info("slot %s listening on %s:%s (child on :%s, advertise %s)",
                SLOT_ID, SLOT_HOST, SLOT_PORT, SLOT_CHILD_PORT, SLOT_ADVERTISE)
    app.run(host=SLOT_HOST, port=SLOT_PORT, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
