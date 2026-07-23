"""Standalone GPU worker agent for the abstract_hugpy_dev LLM pool.

Run this on any box with a GPU and a working ``abstract_hugpy_dev`` install to
donate that GPU's compute to the central console. The agent:

    1. Detects local GPUs.
    2. Registers with the central node (``/api/llm/workers/register``) and keeps
       a persistent worker id in a local state file so restarts reuse the row.
    3. Serves inference over HTTP for the models the central node assigns to it:
           GET  /health
           POST /infer          {model_key, messages|prompt, ...} -> {text, finish_reason}
           POST /infer/stream   -> SSE token/done/error events
       Inference runs through ``abstract_hugpy_dev.managers.dispatch`` exactly like
       the central node, so the worker loads/serves the model on its own GPU.
    4. Heartbeats every ``--heartbeat`` seconds, reporting live GPU stats and
       which models are currently loaded.

The central node's chat route picks an online, assigned worker for the chosen
model and relays this agent's ``/infer/stream`` back to the browser. If no
worker is assigned (or all are offline) the central node runs the model
locally, so adding workers is purely additive.

Usage
-----
    python -m abstract_hugpy_dev.worker_agent \
        --central https://hugpy.ai \
        --name gpu-box-1 \
        --host 10.0.0.5 --port 9100 \
        --models Qwen_Qwen2.5-7B-Instruct,meta-llama_Llama-3.1-8B-Instruct

Every flag also has an env fallback (WORKER_CENTRAL_URL, WORKER_NAME,
WORKER_HOST, WORKER_PORT, WORKER_MODELS, WORKER_ID_FILE, WORKER_HEARTBEAT).
"""
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import socket
import logging
import argparse
import asyncio
import threading
import subprocess
import urllib.request
import urllib.error
import weakref

# Storage-budget refusal (evict-to-fit path). Safe at module scope: budget.py
# imports back from .agent lazily, inside functions, so there is no cycle.
from .budget import BudgetRefusal

from flask import Flask, request, jsonify, Response, stream_with_context

logger = logging.getLogger("abstract_hugpy_dev.worker_agent")
from .imports import *
from ..central import central_base_url
# Per-model in-process generation gate (concurrency hardening). Light module —
# no heavy deps at import; slot-awareness imports the runner stack lazily. It
# serializes entry into an in-process llama.cpp/transformers runner per model so
# concurrent requests can't race the same non-reentrant native context and SEGV
# the whole worker (the computron 2026-07-11 core-dump class).
from . import gen_gate
# request_id -> asyncio.Event, so POST /infer/cancel can stop an in-flight
# stream mid-generation. Populated by _stream_sync, tripped by the cancel route.
# Cancellation now rides the shared comms JobStore (attach_cancel/cancel) —
# the per-process _CANCELS dict this file used to keep is gone (F1.3: no
# side channels).


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
def detect_gpus() -> list[dict]:
    """Best-effort GPU inventory.

    Tries ``nvidia-smi`` first (no Python deps), then ``torch.cuda``. Returns
    an empty list on a CPU-only box — the worker still registers and serves,
    it just won't be fast. The probe itself lives in :mod:`hugpy._platform.hardware`
    so it stays portable (``nvidia-smi.exe`` on Windows, no probe on Apple silicon).
    """
    from .._platform.hardware import detect_gpus as _detect_gpus

    return _detect_gpus()


def safe_import_torch():
    """Import ``torch``, healing a partially-initialized module first.

    WHY THIS EXISTS (worker ae, 2026-07-05): when this process imports a
    CUDA-built ``llama_cpp`` *before* its first ``import torch``, torch's native
    init trips a circular import and aborts mid-way —
    ``partially initialized module 'torch' has no attribute 'library'``. Python
    then caches that broken half-module in ``sys.modules``, so EVERY later
    ``import torch`` in this process (vision/frame extraction, sd-turbo, whisper)
    hands back the same stale wreck for the process's entire life. One bad import
    ordering silently poisons every torch task on the box until restart. Confirmed
    minimal repro: ``python -c "import torch"`` works; ``python -c "import
    llama_cpp, torch"`` reproduces the abort.

    The durable fix is ordering — import torch before any llama_cpp (see
    :func:`_prime_torch_before_llama`). This helper is the recovery net for a
    race we missed: it (a) returns torch straight from cache when it is already
    fully initialized; (b) otherwise tries a normal import; (c) if the import
    raised OR yielded a half-initialized module (missing ``torch.library``),
    evicts ``torch`` and every ``torch.*`` submodule from ``sys.modules`` and
    retries the import exactly ONCE from a clean slate, logging loudly. A
    still-broken torch (or a genuinely absent one) re-raises so the caller's
    error path reports it.
    """
    import importlib

    def _partial(mod) -> bool:
        # A fully-initialized torch always exposes ``torch.library``; its absence
        # is the fingerprint of the circular-import abort described above.
        return mod is not None and not hasattr(mod, "library")

    cached = sys.modules.get("torch")
    if cached is not None and not _partial(cached):
        return cached  # already fully imported — pure cache hit, no work

    first_error = None
    if cached is None:
        try:
            import torch
            if not _partial(torch):
                return torch
        except Exception as exc:  # noqa: BLE001
            first_error = exc

    # We reach here only when torch is poisoned: the import raised, or the cached
    # / freshly-imported module is half-initialized. Evict the whole torch.*
    # subtree so the retry re-runs torch's init from scratch.
    stale = [name for name in list(sys.modules)
             if name == "torch" or name.startswith("torch.")]
    for name in stale:
        del sys.modules[name]
    logger.warning(
        "safe_import_torch: torch was %s%s — purged %d torch.* module(s) from "
        "sys.modules and retrying import ONCE. This is the llama_cpp/torch CUDA "
        "collision: torch MUST be imported before llama_cpp in this process.",
        "un-importable" if first_error is not None else "partially initialized",
        f" ({type(first_error).__name__}: {first_error})" if first_error else "",
        len(stale),
    )
    importlib.invalidate_caches()
    import torch  # single clean retry; propagates if it still can't init
    return torch


def torch_cuda_status() -> dict:
    """Whether *torch* can actually use CUDA — distinct from nvidia-smi seeing a
    card. Inference runs on the GPU only when ``torch.cuda.is_available()`` is
    True; a CPU-only torch build (or a torch/CUDA-driver mismatch) leaves a
    perfectly good GPU unused. Surfaced in /health so this is diagnosable.

    Goes through :func:`safe_import_torch` so a torch half-poisoned by an earlier
    llama_cpp import is healed here instead of reporting a phantom "no CUDA".
    """
    try:
        torch = safe_import_torch()
        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


# The llama.cpp capability probe runs in a SUBPROCESS (see llama_cpp_cuda_status)
# so the agent process NEVER imports llama_cpp merely to report engine status.
# A CUDA-built llama_cpp imported into this process breaks every later
# ``import torch`` for the process's life (see safe_import_torch), and this probe
# used to fire on every heartbeat — the single most likely way to poison the box.
# The child prints one JSON object describing the engine; the parent parses it.
_LLAMA_PROBE_CODE = r"""
import json, sys
out = {"installed": False}
try:
    import llama_cpp
    out["installed"] = True
    out["version"] = getattr(llama_cpp, "__version__", None)
    try:
        out["supports_gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        out["supports_gpu_offload"] = None
    # supports_vision: a multimodal chat handler (or mtmd) is importable — the
    # same capability the in-process vision runner needs to load an mmproj. Central
    # reads this (engine.supports_vision) and ONLY routes image turns to workers
    # that report it, so an older text-only build is never handed an image.
    supports_vision = False
    try:
        from llama_cpp import llama_chat_format as _cf
        for _name in ("Qwen25VLChatHandler", "Llava16ChatHandler",
                      "Llava15ChatHandler", "MiniCPMv26ChatHandler",
                      "MoondreamChatHandler"):
            if hasattr(_cf, _name):
                supports_vision = True
                break
    except Exception:
        pass
    if not supports_vision:
        try:
            import llama_cpp.mtmd_cpp  # noqa: F401
            supports_vision = True
        except Exception:
            supports_vision = False
    out["supports_vision"] = supports_vision
except Exception as exc:
    out = {"installed": False, "error": "%s: %s" % (type(exc).__name__, exc)}
sys.stdout.write(json.dumps(out))
"""

# Engine build is immutable for a process's life, so the first successful probe
# is cached: no python subprocess (which imports CUDA llama_cpp) spawns on every
# 15s heartbeat / /health hit. A not-installed result is intentionally NOT cached
# — /ops/pip can install the engine at runtime and the next probe must see it.
_LLAMA_PROBE_CACHE: dict | None = None
_LLAMA_PROBE_TIMEOUT = 60.0


def llama_cpp_cuda_status() -> dict:
    """Whether *llama.cpp* (GGUF backend) was built with GPU offload support, and
    whether it can decode images (mtmd) — probed in a SUBPROCESS.

    ``n_gpu_layers`` is silently ignored when llama-cpp-python is the CPU-only
    wheel, so a GGUF model runs entirely on CPU even though autofit picked GPU
    layers; ``llama_supports_gpu_offload()`` is the definitive build check. The
    import runs in a child interpreter (never this process) because a CUDA-built
    llama_cpp imported here poisons every later ``import torch`` — see
    :func:`safe_import_torch` and ``_LLAMA_PROBE_CODE``.
    """
    global _LLAMA_PROBE_CACHE
    if _LLAMA_PROBE_CACHE is not None:
        return _LLAMA_PROBE_CACHE
    result = _probe_llama_cpp_subprocess()
    if result.get("installed"):
        _LLAMA_PROBE_CACHE = result
    return result


def _probe_llama_cpp_subprocess() -> dict:
    """Run the llama_cpp probe in a child interpreter and parse its JSON stdout.

    Every failure mode (no python, timeout, crash, garbage output) degrades to an
    ``installed: False`` dict carrying an ``error`` string — the same shape the
    old in-process except path produced, so callers and heartbeats are unchanged.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LLAMA_PROBE_CODE],
            capture_output=True, text=True, timeout=_LLAMA_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"installed": False,
                "error": f"TimeoutExpired: llama_cpp probe exceeded "
                         f"{_LLAMA_PROBE_TIMEOUT:.0f}s"}
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    out = (proc.stdout or "").strip()
    if not out:
        tail = (proc.stderr or "").strip()[-300:]
        return {"installed": False,
                "error": f"llama_cpp probe produced no output "
                         f"(rc={proc.returncode}): {tail}"}
    try:
        return json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return {"installed": False,
                "error": f"llama_cpp probe output unparseable "
                         f"({type(exc).__name__}): {out[:300]}"}


def _prime_torch_before_llama() -> None:
    """Import torch NOW — before this process can import llama_cpp — when torch is
    installed on this box.

    The agent's in-process GGUF fallback (``execute_prompt`` -> python_runner ->
    ``from llama_cpp import Llama``) and the console's torch tasks (vision,
    sd-turbo, whisper) race to be the first native import. If llama_cpp wins, the
    first ``import torch`` aborts mid-init and stays broken for the process's life
    (see :func:`safe_import_torch`). Importing torch first makes it a complete,
    cached module every later import simply reuses — the ordering fix. Best-effort
    and silent when torch isn't installed (CPU/text-only boxes); a torch that
    genuinely can't import is reported per-request by the torch paths, not here.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return  # no torch on this box — nothing to prime, stay quiet
    except Exception:  # noqa: BLE001
        return
    try:
        safe_import_torch()
        logger.info("primed torch ahead of any llama_cpp import "
                    "(llama_cpp/torch CUDA-collision guard)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch priming skipped (%s: %s); torch paths will report "
                       "per-request if it truly can't import",
                       type(exc).__name__, exc)


def _safe_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _free_ram_bytes() -> int | None:
    """Available RAM in bytes — feeds the allocator's CPU tier. Best-effort.

    Reserve-adjusted (managers.spill honors HUGPY_RAM_RESERVE_GIB) AND, when a
    central RAM ceiling is set, further constrained to the budget-bar spec's
    ``remaining`` (t13/t14) so admission and the console bar can never disagree.
    Wire-compat: this stays the historical ``free_ram`` field; the UNCLAMPED
    reserve-only figure rides alongside as ``free_ram_raw`` (below)."""
    try:
        from ..managers.spill import free_ram_bytes
        return free_ram_bytes()
    except Exception:
        return None


def _free_ram_raw_bytes() -> int | None:
    """Reserve-adjusted budgetable free RAM, UNCLAMPED by the central ceiling
    (t13/t14). Central shows the honest physical-semantics bar and derives the
    ceiling-aware budget from this, so it must ride the heartbeat unclamped —
    distinct from ``free_ram`` (clamped, kept for wire-compat)."""
    try:
        from ..managers.spill import free_ram_raw_bytes
        return free_ram_raw_bytes()
    except Exception:
        return None


def _ram_worker_bytes() -> int | None:
    """RSS of THIS worker's own process tree (agent + slot children) — the
    budget-bar spec's ``worker_usage`` for RAM. Best-effort (spill.py)."""
    try:
        from ..managers.spill import ram_worker_bytes
        return ram_worker_bytes()
    except Exception:
        return None


def _ram_external_bytes() -> int | None:
    """RAM held by everything OUTSIDE this worker's tree (box used − own RSS) —
    the budget-bar spec's ``external_usage`` for RAM. Best-effort (spill.py)."""
    try:
        from ..managers.spill import ram_external_bytes
        return ram_external_bytes()
    except Exception:
        return None


def _trim_host_ram() -> None:
    """Return orphaned host RAM to the OS WITHOUT evicting any model.

    After a model's weights are freed, glibc keeps the freed pages in its
    per-arena free-list, so RSS stays pinned (ae observed at 0 free / 128 GB
    used with nothing loaded). gc.collect() drops Python-side references,
    malloc_trim(0) hands the arena's top free chunks back to the kernel, and
    torch.cuda.empty_cache() releases torch's cached CUDA blocks. Every step is
    best-effort — malloc_trim is glibc/Linux-only (musl/other libc lack it), so
    the whole thing stays defensive. Mirrors the imagegen evict idiom
    (managers/imagegen/imagegen_runner.py ~85-93) plus the malloc_trim."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 — non-glibc/musl: no malloc_trim, skip
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — no torch/cuda: nothing to release
        pass


def _agent_rss_bytes() -> int | None:
    """Resident RAM (bytes) of THIS agent process — for the free-ram deltas.

    NOT the slot child's ``rss_bytes`` in the heartbeat (a different process):
    reads VmRSS via the same /proc helper the slot agent uses, psutil fallback.
    Best-effort (None, never fabricated)."""
    try:
        from ..managers.serve.slot_agent import _proc_rss_bytes
        rss = _proc_rss_bytes(os.getpid())
        if rss is not None:
            return rss
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _ram_total_bytes() -> int | None:
    """RAW physical RAM (MemTotal) in bytes — the pool-budget denominator.

    Unlike _free_ram_bytes (reserve-adjusted + RAM_MAX-capped so central plans
    against budgetable RAM), this is the box's total installed memory, so the
    console can render used-vs-total. Best-effort, mirroring
    _platform/hardware.free_ram_bytes: psutil first, then /proc/meminfo, else
    None (never fabricated)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return None


def _spawn_rpc_server(args):
    """Launch llama.cpp's rpc-server so this box lends its GPU to a shard pool.

    Returns the Popen handle, or None if the binary is missing (the node still
    registers/heartbeats, it just won't be usable as a shard backend until a
    CUDA+RPC llama.cpp build provides ``rpc-server``).
    """
    # Prefer an explicit --rpc-bin/WORKER_RPC_BIN, else whatever the engine
    # resolver finds (a `hugpy install-engine` build ships rpc-server), else the
    # bare name on PATH.
    from .._platform.procutil import popen_detached
    from ..engine.resolve import rpc_bin as _resolve_rpc
    binary = args.rpc_bin if (args.rpc_bin and args.rpc_bin != "rpc-server") else None
    binary = binary or _resolve_rpc() or "rpc-server"
    cmd = [binary, "-H", args.rpc_host, "-p", str(args.rpc_port)]
    try:
        proc = popen_detached(cmd)  # noqa: S603 — operator-controlled args
        logger.info("rpc-server up: %s (pid %s)", " ".join(cmd), proc.pid)
        return proc
    except FileNotFoundError:
        logger.error(
            "rpc-server binary %r not found — run `hugpy install-engine --cuda` or "
            "build a CUDA+RPC llama.cpp (cmake -DGGML_CUDA=on -DGGML_RPC=ON) and set "
            "--rpc-bin/WORKER_RPC_BIN. This node registers but can't serve as a "
            "shard backend.", binary)
        return None
    except OSError as exc:
        logger.error("failed to start rpc-server (%s): %s", " ".join(cmd), exc)
        return None


def _local_ip_toward(central_url: str) -> str | None:
    """The worker's own LAN IP on the route it uses to reach central.

    Opening a UDP socket toward central (no packets are actually sent on
    connect) makes the kernel pick the source address it WOULD use — i.e. the
    worker's real outbound IP (e.g. 192.168.1.128), not loopback/127.0.1.1.

    This is what we advertise, because central can't derive it reliably: when
    the worker reaches central via a public domain, NAT hairpinning makes the
    source IP central sees the router's address (192.168.1.1), not the worker's.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(central_url)
        host = parsed.hostname or central_url
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.0)
            s.connect((host, port))
            ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Central node client (registration + heartbeat)
# ---------------------------------------------------------------------------
class WorkerRejected(Exception):
    """Central refused this worker terminally (401 token / 403 blocked).

    Distinct from a transient error or a 410 (re-register): the operator has
    revoked/blocked us, so the agent should stop rather than retry.
    """
    def __init__(self, code: int, message: str = ""):
        super().__init__(message or f"rejected with HTTP {code}")
        self.code = code


class CentralClient:
    def __init__(self, central_url: str, token: str | None = None):
        # Endpoints live under /api on the central Flask app.
        self.base = central_url.rstrip("/") + "/api/llm/workers"
        self.token = token

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.base + path, data=data, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 401 (bad/revoked/required token) and 403 (blocked) are terminal —
            # the operator decided this worker isn't welcome. Surface them as
            # WorkerRejected so callers stop instead of retrying. Other codes
            # (e.g. 410 "re-register") propagate unchanged.
            if exc.code in (401, 403):
                raise WorkerRejected(exc.code, exc.reason or "") from exc
            raise

    def register(self, payload: dict) -> dict:
        return self._post("/register", payload)

    def heartbeat(self, worker_id: str, payload: dict) -> dict:
        return self._post(f"/{worker_id}/heartbeat", payload)


# ---------------------------------------------------------------------------
# Local inference (reuses the same dispatch the central node uses)
# ---------------------------------------------------------------------------
def _ensure_present(payload: dict, central_url: str | None, state=None) -> None:
    """Provision the requested model before inference (central-first, HF fallback).

    ``state`` opts the pull into the STORAGE BUDGET (evict-to-fit, else refuse).
    A BudgetRefusal PROPAGATES: an unfittable model must fail loudly here rather
    than fall through to a confusing downstream "model not found".
    """
    model_key = payload.get("model_key")
    if not model_key:
        return
    try:
        from .provision import ensure_model_present, ensure_model_registered

        # Learn the model from central if the worker wasn't built with it, then
        # run inference against the canonical local key.
        canonical = ensure_model_registered(model_key, central_url)
        if canonical and canonical != model_key:
            payload["model_key"] = canonical
        # DEMAND: a real inference call is waiting on this model. Central never
        # budget-refuses a demand pull (2026-07-17) — the worker's own fit_plan
        # evicts to fit it; refusing a called model at central would break serving.
        ensure_model_present(payload.get("model_key"), central_url, state=state,
                             purpose="demand")
        if state is not None:
            state.refused.pop(payload.get("model_key"), None)
    except BudgetRefusal as exc:
        if state is not None:
            state.refused[payload.get("model_key") or model_key] = dict(exc.reason)
        logger.error("provisioning of %s REFUSED: %s", model_key,
                     exc.reason.get("reason"))
        raise
    except Exception as exc:
        logger.warning("provisioning check for %s failed: %s", model_key, exc)


def _ensure_present_streaming(payload: dict, central_url: str | None, state=None):
    """Provision the model, yielding SSE 'status' events with download progress.

    Yields encoded SSE lines (status/error). Returns normally once the model is
    present (or was already). Throttled so we don't flood the stream.

    ``state`` opts the pull into the STORAGE BUDGET. A refusal is yielded as an
    SSE 'error' event carrying the structured reason — the stream ends honestly
    ("won't fit: needs X…") instead of showing a progress bar for a download
    that was never going to start.
    """
    model_key = payload.get("model_key")
    if not model_key:
        return
    try:
        from .provision import (
            ensure_model_present, ensure_model_registered, model_is_local,
        )

        # Learn the model from central first, then work the rest of the stream
        # against the canonical local key (so resolution/loading can find it).
        canonical = ensure_model_registered(model_key, central_url)
        if canonical and canonical != model_key:
            payload["model_key"] = canonical
            model_key = canonical

        if model_is_local(model_key):
            return  # nothing to do; go straight to generation

        yield _sse({"type": "status", "stage": "provision",
                    "message": f"fetching {model_key}…", "progress": 0.0})

        # provision runs in a worker thread; it pushes (done,total,fname) onto a
        # queue that we drain into throttled SSE status events from this thread.
        import queue
        import threading

        q: "queue.Queue" = queue.Queue()
        result = {"ok": False, "err": None}

        def _progress(done, total, fname):
            q.put((done, total, fname))

        def _run():
            try:
                # DEMAND: a live chat is streaming; never budget-refused centrally.
                result["ok"] = ensure_model_present(model_key, central_url,
                                                    progress=_progress, state=state,
                                                    purpose="demand")
            except Exception as exc:  # pragma: no cover
                result["err"] = exc
            finally:
                q.put(None)  # sentinel: done

        th = threading.Thread(target=_run, daemon=True)
        th.start()

        last_emit = 0.0
        while True:
            item = q.get()
            if item is None:
                break
            done, total, fname = item
            now = time.time()
            # Emit at most ~3x/sec, but always emit the first/last.
            if now - last_emit < 0.33 and done < (total or 1):
                continue
            last_emit = now
            frac = (done / total) if total else 0.0
            yield _sse({
                "type": "status", "stage": "provision",
                "message": f"downloading {model_key} ({_human(done)}/{_human(total)})",
                "progress": round(frac, 4),
                "done_bytes": done, "total_bytes": total, "file": fname,
            })
        th.join(timeout=1.0)

        if isinstance(result["err"], BudgetRefusal):
            # Storage verdict, not a transfer failure: the pull never started.
            # Carry the structured reason so the UI can show WHY it's missing.
            reason = result["err"].reason
            if state is not None:
                state.refused[model_key] = dict(reason)
            yield _sse({"type": "error", "stage": "provision",
                        "refused": reason,
                        "message": f"{model_key} won't fit: {reason.get('reason')}"})
            return
        if result["err"] is not None:
            yield _sse({"type": "error",
                        "message": f"provisioning failed: {result['err']}"})
            return
        if not result["ok"]:
            yield _sse({"type": "error",
                        "message": f"could not fetch model {model_key} from central or HF"})
            return
        yield _sse({"type": "status", "stage": "provision",
                    "message": "model ready, loading…", "progress": 1.0})
    except Exception as exc:
        logger.warning("streaming provisioning for %s failed: %s", model_key, exc)


def _human(n) -> str:
    if not n:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def _materialize_file(payload: dict) -> str | None:
    """Rebuild an inlined upload (file_b64/file_name) into a local temp file.

    Central ships uploaded files as base64 since the worker can't see central's
    UPLOADS_HOME. We write the bytes to a temp file, point ``payload["file"]``
    at it, and return the temp path so the caller can delete it afterwards.
    Returns None when there's nothing to materialize.
    """
    b64 = payload.pop("file_b64", None)
    name = payload.pop("file_name", None)
    if not b64:
        return None
    import base64
    import tempfile

    suffix = ""
    if name and "." in name:
        suffix = "." + name.rsplit(".", 1)[-1]
    fd, tmp_path = tempfile.mkstemp(prefix="hugpy_worker_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(base64.b64decode(b64))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    payload["file"] = tmp_path
    return tmp_path


def _cleanup_file(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_once(payload: dict) -> dict:
    #from abstract_hugpy_dev.managers.dispatch import execute_prompt

    tmp = _materialize_file(payload)
    try:
        result = execute_prompt(**payload)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        # Return the full result envelope so ANY task (embed, vision, whisper, …)
        # round-trips back to central as its real result_type — not just chat
        # text. Central's DelegatingRunner validates this into result_type.
        if hasattr(result, "model_dump"):
            return result.model_dump()
        # Non-pydantic fallback (shouldn't happen for a registered runner).
        return {
            "ok": getattr(result, "ok", True),
            "text": getattr(result, "text", None) or str(result),
            "finish_reason": getattr(result, "finish_reason", None) or "stop",
        }
    finally:
        _cleanup_file(tmp)


_SPILL_ENV = {
    "n_gpu_layers": "HUGPY_N_GPU_LAYERS",
    "gpu_mem_gib": "HUGPY_GPU_MEM_GIB",
    "cpu_mem_gib": "HUGPY_CPU_MEM_GIB",
    # Explicit per-model core budget (slot loads pass it to the child;
    # in-process loads read DEFAULT_LLAMA_THREADS at build).
    "threads": "DEFAULT_LLAMA_THREADS",
    "tensor_split": "HUGPY_TENSOR_SPLIT",
    "main_gpu": "HUGPY_MAIN_GPU",
    "n_gpu": "HUGPY_N_GPU",
    # Cross-machine RPC sharding: comma-separated "host:port" of llama.cpp
    # rpc-servers to offload layers onto. When central's allocator decides to
    # shard a model, it ships this (+ tensor_split) as a per-request spill
    # override; spill.llama_kwargs() turns it into Llama(rpc_servers=...).
    "rpc_servers": "HUGPY_RPC_SERVERS",
}


# ── operator resource limits (two-tier) ─────────────────────────────────────
# This box's OWN unit config is the hard ceiling; central may set per-worker
# limits but they apply only as a TIGHTENING (min of the two). Originals are
# captured at import so a central limit can be raised again later without
# being mistaken for local config.
_CAP_KNOBS = {
    "ram_max_gib": "HUGPY_RAM_MAX_GIB",
    "gpu_mem_gib": "HUGPY_GPU_MEM_GIB",
    "threads": "DEFAULT_LLAMA_THREADS",
}
_LOCAL_CAP_ENV = {k: os.environ.get(env) for k, env in _CAP_KNOBS.items()}


def _local_caps() -> dict:
    """The operator-configured ceilings from this box's own config, reported to
    central so it can only tighten, never exceed them. Reserves ride along for
    display."""
    out: dict = {}
    for key in _CAP_KNOBS:
        raw = _LOCAL_CAP_ENV.get(key)
        if raw in (None, ""):
            continue
        try:
            out[key] = int(raw) if key == "threads" else float(raw)
        except ValueError:
            continue
    for key, env in (("ram_reserve_gib", "HUGPY_RAM_RESERVE_GIB"),
                     ("vram_reserve_gib", "HUGPY_VRAM_RESERVE_GIB"),
                     # The box's stated STORAGE delegation: how much local disk it
                     # gives the model cache. Reported as a cap so a central
                     # disk_cache_gib limit is clamped to it (_clamp_limits) — the
                     # worker's delegation wins, same rule as RAM. Absent → central
                     # may set any disk_cache_gib (unclamped).
                     ("disk_cache_gib", "HUGPY_DISK_CACHE_MAX_GIB")):
        raw = os.environ.get(env)
        if raw:
            try:
                out[key] = float(raw)
            except ValueError:
                pass
    return out


def _adopt_storage_inputs(state: "WorkerState", worker: dict | None) -> None:
    """Store the STORAGE budget's two central-owned inputs on state.

      * ``limits`` — carries ``disk_cache_gib``, central's storage allocation
        for this box. The auto-evict path is OFF until it is set (budget.cap_bytes).
      * ``model_last_picked`` — central's ``{model_key: epoch}`` LRU clock, the
        FIFO key. The worker can't know it: central routes the calls.
      * ``allocated`` — the ALLOCATION-LEVEL totals (operator, 2026-07-16: "show
        how much is needed based on the total size of all models allocated").
        Sizing the assignment set needs the MANIFEST, which only central holds:
        doing it worker-side would mean one HTTP round-trip PER assigned model,
        inside the single-flight provision lock, on the refusal path. Central
        already computes this per read (storage_proposal.allocated_totals) and
        every heartbeat reply carries it, so the worker just adopts the answer
        and the refusal reason stays a pure, offline computation.

    Never raises into the heartbeat: a malformed reply just leaves the previous
    values in place (and an absent allocation simply keeps the budget unmanaged).
    """
    if not isinstance(worker, dict):
        return
    limits = worker.get("limits")
    if isinstance(limits, dict):
        state.limits = dict(limits)
        # HOT-TIER ALIGNMENT (slice 4): project central's disk_cache_gib into an
        # env var so the hot-cache tier (a different process context that never
        # holds `state`) can fold it into its own min-wins when it shares the
        # store drive. The tier reads this LIVE (hot_cache._store_disk_cap_gib),
        # so a fresh heartbeat's number takes effect without a restart. Absent /
        # cleared -> the tier simply has no central term. This is projection only;
        # the AUTHORITATIVE budget gate stays budget.resolve_effective_cap.
        try:
            dc = limits.get("disk_cache_gib")
            if dc in (None, ""):
                os.environ.pop("_HUGPY_CENTRAL_DISK_CACHE_GIB", None)
            else:
                os.environ["_HUGPY_CENTRAL_DISK_CACHE_GIB"] = str(float(dc))
        except (TypeError, ValueError):
            os.environ.pop("_HUGPY_CENTRAL_DISK_CACHE_GIB", None)
    lp = worker.get("model_last_picked")
    if isinstance(lp, dict):
        state.model_last_picked = dict(lp)
    storage = worker.get("storage")
    if isinstance(storage, dict) and storage.get("allocated_count") is not None:
        state.allocated = {
            "allocated_total_bytes": storage.get("allocated_total_bytes"),
            "allocated_count": storage.get("allocated_count"),
            "allocated_unknown_count": storage.get("allocated_unknown_count"),
        }


def _apply_central_limits(worker: dict | None) -> None:
    """Adopt central's per-worker limits as min(central, local config)."""
    limits = (worker or {}).get("limits") or {}
    for key, env in _CAP_KNOBS.items():
        vals = []
        local_raw = _LOCAL_CAP_ENV.get(key)
        if local_raw not in (None, ""):
            try:
                vals.append(float(local_raw))
            except ValueError:
                pass
        if limits.get(key) is not None:
            try:
                vals.append(float(limits[key]))
            except (TypeError, ValueError):
                pass
        if not vals:
            # Neither side sets it: clear a previously-applied central limit.
            if local_raw in (None, "") and env in os.environ:
                os.environ.pop(env, None)
            continue
        eff = min(vals)
        os.environ[env] = str(int(eff)) if key == "threads" else str(eff)


def _loaded_detail() -> dict:
    # Size EVERY serving row: start with on-disk dir bytes for all frameworks
    # (transformers/diffusers/llama), then let the GGUF runner detail overlay
    # its exact file bytes + layer/GPU split on top. Without the disk base,
    # non-GGUF rows had no size at all.
    detail: dict = {}
    try:
        from ..managers.dispatch import loaded_disk_detail
        detail.update(loaded_disk_detail())
    except Exception:
        pass
    try:
        from ..managers.llama.runners.get import loaded_runner_detail
        for key, facts in loaded_runner_detail().items():
            detail.setdefault(key, {}).update(facts)
    except Exception:
        pass
    return detail


# A resident counts as "serving" if it answered within this window; older ones
# read as idle-resident. Wide enough to keep a warm model lit between bursts,
# short enough that yesterday's test churn shows idle.
_SERVING_WINDOW_S = 180.0


def _model_framework(mk: str) -> "str | None":
    """Framework for a model_key ('gguf'/'transformers'/'comfy'/…) or None.

    Module-level so residency reporting can cheaply tell comfy rows apart from
    real in-pool residents: a comfy checkpoint is served by the EXTERNAL,
    adopted ComfyUI process (out-of-pool) — the worker holds only a thin client
    runner with NO weights, so it must never be counted as an in-RAM resident."""
    try:
        from .imports import get_model_config
        return getattr(get_model_config(mk), "framework", None)
    except Exception:  # noqa: BLE001 — unknown row: treat as non-comfy
        return None


# ── worker-local slot pool (CON-02) ─────────────────────────────────────────
# With SLOT_COUNT > 0 the agent supervises N slot_agent children — the same
# slot machinery central runs, but agent-managed (rootless, no systemd units
# to install). Slot children run llama_cpp.server (no C++ llama-server binary
# needed on workers), and get_llama_runner's slot-first path then serves this
# worker's requests from slots: resident, TTL'd, crash-ISOLATED (a load that
# aborts kills a child, not the agent — the failure mode that took the whole
# agent down on 2026-07-02).

def _slot_statuses() -> list | None:
    try:
        from ..managers.serve.slots import SlotPool, _slot_count
        n = _slot_count()
        if n <= 0:
            # Effective slot count 0 -> report NO slots as an explicit empty
            # list (not None), so central overwrites and CLEARS any stale
            # phantom rows a prior config left behind. Fixes the zero-slot box
            # (e.g. a transformers-only CPU worker) that advertised 2
            # unreachable seats it never actually ran.
            return []
        # Never report more rows than the effective slot count.
        return SlotPool().statuses()[:n]
    except Exception:
        return None


# ── REAL per-process GPU VRAM (nvidia-smi) ──────────────────────────────────
# Type/ngl-based inference was WRONG: a transformers/vision model loads onto
# CUDA but reports n_gpu_layers=null, so the console mislabeled it "host RAM —
# not in VRAM". Ground truth is nvidia-smi's PER-PROCESS accounting, joined with
# what THIS worker knows it launched (slot child PIDs) or holds (in-process
# torch models). Everything here degrades to null on a box with no GPU / no
# nvidia-smi, so such a worker behaves exactly as before.

_MIB = 1024 * 1024
# nvidia-smi is polled at most once per this window and shared across every
# allocation in a heartbeat — never spawned per model.
_GPU_PROC_TTL_S = 8.0
_GPU_PROC_CACHE: dict = {"at": 0.0, "value": {}}


def _gpu_process_vram() -> dict:
    """``{pid: {"name": str, "mib": int}}`` from nvidia-smi's per-process compute
    accounting. Cached ~heartbeat cadence so it runs ONCE per beat, not per model.

    Degrades to ``{}`` (→ callers keep today's behavior) when nvidia-smi is
    absent (no GPU / non-CUDA host), errors, or reports "[N/A]"/"[Not Supported]"
    for a row (no per-process accounting)."""
    now = time.time()
    if now - _GPU_PROC_CACHE["at"] < _GPU_PROC_TTL_S:
        return _GPU_PROC_CACHE["value"]
    out: dict = {}
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                pid_s, name, mem_s = parts[0], parts[1], parts[-1]
                if not pid_s.isdigit():
                    continue
                try:
                    mib = int(float(mem_s))     # "[N/A]"/"[Not Supported]" → skip row
                except ValueError:
                    continue
                out[int(pid_s)] = {"name": name, "mib": mib}
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        out = {}                                # no GPU / no nvidia-smi → today's behavior
    _GPU_PROC_CACHE.update(at=now, value=out)
    return out


def _comfy_process_vram(gpu_procs: "dict | None" = None) -> "int | None":
    """Real VRAM (bytes) of the adopted, EXTERNAL ComfyUI process — from
    nvidia-smi, never from on-disk checkpoint bytes (that all-checkpoints sizing
    is exactly the "37 serving / ~600 GB" bug the 0.1.137 guard fixed). Sums any
    compute proc whose name marks it ComfyUI. ``None`` when nvidia-smi reports no
    such proc (not running / on CPU / no per-proc accounting)."""
    procs = gpu_procs if gpu_procs is not None else _gpu_process_vram()
    mib = 0
    hit = False
    for info in procs.values():
        if "comfyui" in (info.get("name") or "").lower():
            mib += int(info.get("mib") or 0)
            hit = True
    return mib * _MIB if hit else None


def _inprocess_gpu_bytes() -> dict:
    """``{model_key: {"vram_bytes": int, "device": 'cuda'|'cpu'|None}}`` for every
    in-process torch model THIS worker holds — the piece that makes a CUDA-
    resident transformers/vision model stop reading as host RAM.

    The worker python is ONE nvidia-smi process (e.g. ae's 3622 MiB lump holding
    many in-process models at once); torch is the only tool that can split that
    lump per-model. For each model we sum its parameter+buffer bytes that live on
    a cuda device (deduped by storage pointer, so a module shared across two
    task-variants counts once). ``{}`` when torch is missing.

    Reads only ALREADY-materialized state — instance ``__dict__`` and the class-
    level pipeline caches — never a lazy property, so this telemetry pass can
    never trigger a model load."""
    try:
        import torch
    except Exception:
        return {}

    seen_ptrs: set = set()

    def _obj_bytes(obj) -> tuple:
        """(cuda_bytes, cpu_bytes) for a torch nn.Module or a diffusers pipeline
        (walked via its ``.components`` sub-models). Non-torch → (0, 0)."""
        cuda = cpu = 0
        comps = getattr(obj, "components", None)     # diffusers pipeline
        if isinstance(comps, dict):
            for c in comps.values():
                cc, pc = _obj_bytes(c)
                cuda += cc
                cpu += pc
            return cuda, cpu
        if not isinstance(obj, torch.nn.Module):
            return 0, 0
        try:
            tensors = list(obj.parameters()) + list(obj.buffers())
        except Exception:
            return 0, 0
        for t in tensors:
            try:
                ptr = t.data_ptr()
            except Exception:
                continue
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            try:
                nbytes = t.numel() * t.element_size()
            except Exception:
                continue
            if getattr(t, "is_cuda", False):
                cuda += nbytes
            else:
                cpu += nbytes
        return cuda, cpu

    objs_by_key: dict = {}

    def _add(mk, v) -> None:
        if isinstance(v, torch.nn.Module) or isinstance(
                getattr(v, "components", None), dict):
            objs_by_key.setdefault(mk, []).append(v)

    # 1. In-process runner wrappers (transformers/vision/etc.) — the model is an
    #    instance attribute (e.g. vision_coder.self.model, coder.self.model).
    try:
        from ..managers.dispatch.dispatch import _INSTANCES, _INSTANCES_LOCK
        with _INSTANCES_LOCK:
            items = list(_INSTANCES.items())
        for key, runner in items:
            # The cache_key is usually (model_key, task), but some runners key on
            # longer tuples (vision: (key,min,max,dtype); shard: (key,"__vision__",()))
            # or a bare string. The old `for (mk,_task), runner in items` assumed
            # 2-tuples, so ONE non-2-tuple key raised and the outer except silently
            # zeroed EVERY model's VRAM. Extract model_key arity-agnostically and
            # isolate each runner so one bad entry can't abort the whole walk.
            try:
                mk = key[0] if isinstance(key, tuple) and key else key
                attrs = list(vars(runner).values())
            except Exception:
                continue                        # no __dict__ / odd key → skip this one
            for v in attrs:
                _add(mk, v)
                # transformers Pipelines hold the nn.Module at `.model`, not as a
                # direct attr — reach it so pipeline-wrapped models count too
                # (_add ignores non-modules; storage-ptr dedup avoids double count).
                inner = getattr(v, "model", None)
                if inner is not None and inner is not v:
                    _add(mk, inner)
    except Exception:
        pass

    # 2. Diffusers pipelines live in a CLASS-level singleton keyed by model_key,
    #    not on the runner instance — reach them there.
    try:
        from ..managers.imagegen import imagegen_runner as _ig
        for clsname in ("ImageGenRunner", "Img2ImgRunner"):
            cls = getattr(_ig, clsname, None)
            cache = getattr(cls, "_PIPELINES", None)
            if isinstance(cache, dict):
                for mk, pipe in list(cache.items()):
                    _add(mk, pipe)
    except Exception:
        pass

    # 3. Transformers causal-LMs (DeepCoderChatRunner — DAN-Qwen3, DeepCoder, etc.)
    #    DON'T hold their nn.Module on the runner in _INSTANCES: the runner keeps a
    #    cfg and reaches the model lazily via a `coder` property → a SEPARATE
    #    module-level `REGISTRY._instances` of DeepCoder objects. So vars(runner)
    #    never sees the weights, which is why a static in-process transformers model
    #    read device=None. Walk the already-BUILT instances directly (read
    #    `_instances`, never `REGISTRY.get()`, so telemetry can't trigger a load);
    #    each DeepCoder carries its model_key on `.cfg` and its weights on `.model`.
    try:
        from ..managers.generate.coder import REGISTRY as _DC_REGISTRY
        insts = getattr(_DC_REGISTRY, "_instances", None)
        if isinstance(insts, dict):
            for dc in list(insts.values()):
                mk = getattr(getattr(dc, "cfg", None), "model_key", None)
                model = getattr(dc, "model", None)
                if mk and model is not None:
                    _add(mk, model)
    except Exception:
        pass

    # RECONCILIATION: sum(out[*].vram_bytes) is the worker python's model weights
    # on GPU. It runs a bit UNDER that process's nvidia-smi total
    # (_gpu_process_vram()[os.getpid()]) because the CUDA context (~tens of MiB)
    # plus activation/workspace/KV-cache scratch are NOT parameters. That residual
    # is real driver overhead — left as-is, never smeared onto a model, so a
    # model's vram_bytes stays its honest weight footprint.
    out: dict = {}
    for mk, objs in objs_by_key.items():
        cuda = cpu = 0
        for o in objs:
            cc, pc = _obj_bytes(o)
            cuda += cc
            cpu += pc
        device = "cuda" if cuda > 0 else ("cpu" if cpu > 0 else None)
        out[mk] = {"vram_bytes": cuda, "device": device}
    return out


def _vram_split_from_pidlog(pid_log: "dict | None") -> dict:
    """Split the SAME pid_registry snapshot central will store into the two
    VRAM figures the budget-bar spec needs (t13/t14):

      * ``vram_attributed_bytes`` — sum of the attributed MODEL rows (rows with
        a real ``model_key``): the VRAM hugpy's own served models hold. This is
        the spec's VRAM ``worker_usage``. cuda_context (the agent's own CUDA
        context) counts as worker infra too, so it's folded in — it IS the
        worker's usage, just not a served model. ComfyUI and genuinely-foreign/
        unattributed rows are EXTERNAL, excluded here.
      * ``vram_unattributed_bytes`` — sum of the ``unattributed`` squatter rows
        (mib→bytes): genuinely foreign GPU use the console surfaces as overhead.

    Sampled from the pid_log produced the SAME beat as detect_gpus(), so the
    attribution and the driver totals are one snapshot. Degrades to
    ``{"vram_attributed_bytes": None, "vram_unattributed_bytes": None}`` when
    there's no pid_log (no GPU / older agent) — never fabricated."""
    if not isinstance(pid_log, dict):
        return {"vram_attributed_bytes": None, "vram_unattributed_bytes": None}
    attributed = 0
    for row in (pid_log.get("models") or []):
        if not isinstance(row, dict):
            continue
        mode = row.get("host_mode")
        # comfy rows are EXTERNAL (adopted, out-of-pool) — not worker usage.
        if mode == "comfy":
            continue
        vb = row.get("vram_bytes")
        if vb is None:
            continue
        # A worker row counts when it's a served model (has model_key) OR the
        # worker's own cuda_context infra lump. Foreign rows have neither.
        if row.get("model_key") or mode == "cuda_context":
            try:
                attributed += int(vb)
            except (TypeError, ValueError):
                continue
    _MIB_LOCAL = 1024 * 1024
    unattributed = 0
    for row in (pid_log.get("unattributed") or []):
        if not isinstance(row, dict):
            continue
        try:
            unattributed += int(row.get("mib") or 0) * _MIB_LOCAL
        except (TypeError, ValueError):
            continue
    return {"vram_attributed_bytes": attributed,
            "vram_unattributed_bytes": unattributed}


def _slot_total_layers_fallback(model_key: str) -> "int | None":
    """Total GGUF layer count for a SLOT-seated model whose slot build predates
    the ``total_layers`` status field (an adopted stale slot child) — resolved
    via the same geometry reader the offload math uses, and CACHED per model so
    the heartbeat never re-parses a GGUF header every beat. None (also cached)
    for non-GGUF / unresolvable — the allocation row then omits the field."""
    if model_key in _TOTAL_LAYERS_CACHE:
        return _TOTAL_LAYERS_CACHE[model_key]
    tl = None
    try:
        _, tl = _served_gguf_geometry(model_key)
    except Exception:  # noqa: BLE001 — best-effort metadata, never break a beat
        tl = None
    _TOTAL_LAYERS_CACHE[model_key] = tl
    return tl


_TOTAL_LAYERS_CACHE: dict = {}   # model_key -> int | None (misses cached too)


def _allocations(slot_statuses: "list | None" = None) -> list:
    """Unified, engine-agnostic view of every resource allocation on this
    worker — one entry per SLOT-seated model and one per in-RAM (in-process)
    resident model. A slot is a resource allocation to a model regardless of
    engine, so GGUF slot occupants and transformers models held in the agent's
    OWN process are reported side by side. This is a NEW field parallel to (not
    a replacement for) loaded_models/slots, so old central/UI keep working.

    Each entry carries the REAL GPU residency the console consumes:
      ``vram_bytes`` (int bytes | null) — actual VRAM the model occupies now.
      ``device``     ('cuda' | 'cpu' | null) — the device the weights live on.
    SLOT rows join nvidia-smi against the slot's child_pid (exact per-model);
    RAM rows split the worker python's nvidia-smi lump per-model via torch. Both
    are null on a box with no GPU / no nvidia-smi — identical to today. VRAM is
    NEVER written into model_bytes/weight_bytes (those stay on-disk *size*).

    ``slot_statuses`` may be passed in to avoid a second slot round-trip when
    the heartbeat already computed it."""
    out: list = []
    seen: set = set()
    gpu_procs = _gpu_process_vram()            # {} when no GPU / no nvidia-smi
    rows = slot_statuses if slot_statuses is not None else _slot_statuses()
    for s in (rows or []):
        mk = (s or {}).get("model_key")
        if not mk:
            continue                       # empty seats aren't allocations
        seen.add(mk)
        # Join nvidia-smi on the slot's llama-server CHILD pid (the process that
        # actually holds the weights). Absent child_pid (old slot build) or empty
        # gpu_procs (no nvidia-smi) → null, exactly today's shape.
        vram_bytes = None
        device = None
        if gpu_procs:
            cp = s.get("child_pid")
            info = gpu_procs.get(cp) if cp is not None else None
            if info is not None:
                vram_bytes = int(info["mib"]) * _MIB
                device = "cuda" if vram_bytes > 0 else "cpu"
            elif cp is not None:
                # Child is alive but not a GPU compute app → CPU-resident (ngl=0).
                vram_bytes, device = 0, "cpu"
        row = {
            "kind": "slot", "model_key": mk,
            "slot_id": s.get("slot_id"), "healthy": s.get("healthy"),
            "busy": s.get("busy"), "endpoint": s.get("endpoint"),
            "rss_bytes": s.get("rss_bytes"),
            "n_gpu_layers": s.get("n_gpu_layers"), "ctx": s.get("ctx"),
            "vram_bytes": vram_bytes, "device": device,
        }
        # Honest allocation accuracy (2026-07-22), omit-when-unset so the wire
        # shape is unchanged for old slots/central:
        #  * total_layers — GGUF block_count so "17/48" renders instead of
        #    "17/undefined". The slot reports it since this build; for an
        #    ADOPTED older slot child fall back to the agent's own GGUF-header
        #    read (cached — one header parse per model, not per beat).
        #  * rss_anon_bytes / rss_file_bytes — VmRSS counts the mmap'd GGUF's
        #    file-backed pages (reclaimable cache) as resident, overstating true
        #    pinned RAM ~28x on ae; RssAnon is the honest figure. Slot-reported,
        #    else read from /proc/<child_pid>/status here (same box).
        tl = s.get("total_layers")
        if tl is None:
            tl = _slot_total_layers_fallback(mk)
        if tl is not None:
            row["total_layers"] = tl
        if s.get("rss_anon_bytes") is not None:
            for k in ("rss_anon_bytes", "rss_file_bytes", "rss_shmem_bytes"):
                if s.get(k) is not None:
                    row[k] = s[k]
        elif s.get("child_pid") is not None:
            try:
                from ..managers.serve.slot_agent import _proc_rss_detail
                row.update(_proc_rss_detail(s["child_pid"]))
            except Exception:  # noqa: BLE001 — never break the heartbeat on /proc
                pass
        out.append(row)
    detail = _loaded_detail()
    inproc = _inprocess_gpu_bytes()            # {} when torch missing
    try:
        from ..managers.dispatch.dispatch import last_used_snapshot
        last_used = last_used_snapshot() or {}
    except Exception:
        last_used = {}
    now = time.time()
    for mk in loaded_model_keys():
        if mk in seen:
            continue                       # already counted as a slot allocation
        if _model_framework(mk) == "comfy":
            # ComfyUI checkpoints are served by the EXTERNAL, adopted ComfyUI
            # process (out-of-pool): the worker instantiates only a thin client
            # runner that holds NO weights. Counting them as in-RAM residents,
            # sized by on-disk dir bytes, is exactly what made ae read "37
            # serving / ~600 GB". They surface via the `comfy` heartbeat block,
            # not as pool allocations.
            continue
        d = detail.get(mk) or {}
        ip = inproc.get(mk) or {}
        out.append({
            "kind": "ram", "model_key": mk,
            "model_bytes": d.get("model_bytes"),
            "weight_bytes": d.get("weight_bytes"),
            "gpu_pct": d.get("gpu_pct"),
            "n_gpu_layers": d.get("n_gpu_layers"),
            "total_layers": d.get("total_layers"),
            # REAL GPU residency from torch introspection: a cuda-resident
            # transformers/vision model reports vram_bytes>0 + device='cuda' and
            # stops reading as host RAM. None when torch can't see it (e.g. an
            # in-process GGUF Llama handle, not a torch module — its GPU share is
            # still described by n_gpu_layers/gpu_pct above).
            "vram_bytes": ip.get("vram_bytes"),
            "device": ip.get("device"),
            # Idle-vs-serving: the console shows 🔥 only for genuinely-active
            # residents (recently-used in-process), the rest as idle-resident —
            # so a pool of test-churn leftovers never reads as "all serving".
            # Computed worker-side (its own clock vs last_used) to dodge any
            # client/central clock skew. last_used is epoch seconds (None=never).
            "last_used": last_used.get(mk),
            "serving": (last_used.get(mk) is not None
                        and (now - last_used[mk]) < _SERVING_WINDOW_S),
        })
    return out


# Live slot children, module-global so the self-update path can terminate
# them BEFORE re-exec: an orphaned slot survives the update and keeps serving
# OLD code forever (the adoption probe can't tell versions apart) — the
# "adopted stale slot" failure of 2026-07-02.
_SLOT_PROCS: dict[int, subprocess.Popen] = {}


def _kill_slots() -> None:
    for i, p in list(_SLOT_PROCS.items()):
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        _SLOT_PROCS.pop(i, None)


def _supervise_slots() -> None:
    """Spawn and keep alive SLOT_COUNT slot_agent children (no-op when 0)."""
    from ..managers.serve.slots import slots_enabled, _slot_count

    if not slots_enabled():
        return
    top_pkg = __name__.split(".")[0]
    module = f"{top_pkg}.managers.serve.slot_agent"
    n = _slot_count()
    procs = _SLOT_PROCS

    def _slot_answering(i: int) -> bool:
        """A slot from a PREVIOUS agent process may still own the port (agent
        re-exec orphans its children) — adopt it instead of bind-fighting."""
        try:
            from ..managers.serve.slots import slot_urls
            import urllib.request as _url
            with _url.urlopen(slot_urls()[i - 1] + "/health", timeout=2) as r:
                return r.getcode() == 200
        except Exception:
            return False

    def _spawn(i: int) -> None:
        if _slot_answering(i):
            logger.info("slot supervisor: slot %d already serving (adopted)", i)
            procs.pop(i, None)
            return
        env = dict(os.environ)
        env["SLOT_ID"] = str(i)
        procs[i] = subprocess.Popen([sys.executable, "-m", module], env=env)
        logger.info("slot supervisor: started slot %d (pid %s)", i, procs[i].pid)

    def _loop() -> None:
        for i in range(1, n + 1):
            _spawn(i)
        while True:
            time.sleep(20)
            for i in range(1, n + 1):
                p = procs.get(i)
                if p is not None and p.poll() is None:
                    continue                      # our child, alive
                if _slot_answering(i):
                    continue                      # adopted orphan, alive
                if p is not None:
                    logger.warning("slot %d died (rc=%s) — respawning", i, p.returncode)
                _spawn(i)

    threading.Thread(target=_loop, daemon=True, name="slot-supervisor").start()
    logger.info("slot supervisor: managing %d slot(s) via llama_cpp.server children", n)


# ═══════════════════════════════════════════════════════════════════════════
# Restart mechanism (2026-07-12 incident class — CODE_GAPS "2026-07-12" item 3)
# ═══════════════════════════════════════════════════════════════════════════
# Two real incidents (computron restart-loop 160→219; op's 403 dueling-worker
# saga) traced to os.execv under systemd: execv KEEPS this PID but the way the
# agent re-exec'd left systemd believing the service died, so Restart= respawned
# a FRESH process that collided with the still-listening old image on :9100
# ("Address already in use") and restart-looped while the orphan kept heart-
# beating. The fix: UNDER SYSTEMD, never execv — release resources cleanly and
# EXIT with a distinct code so systemd's Restart= respawns exactly ONE properly-
# tracked process. STANDALONE (no systemd), execv in place is still correct and
# is kept as-is.
#
# Exit-code convention: a wanted restart exits _RESTART_EXIT_CODE (a distinct
# NON-ZERO). Non-zero matters because the canonical unit is `Restart=on-failure`
# (install.py) — a zero exit would NOT respawn there; a non-zero one does, and it
# also respawns under the field boxes' hand-rolled `Restart=always`. It is
# deliberately DIFFERENT from _terminal_exit's exit 0 (a 401/403 eviction that
# must STAY stopped under on-failure).
_RESTART_EXIT_CODE = 42
# Bound on how long a restart waits for in-flight generations to drain before it
# stops honoring them and exits anyway (never hangs forever). Read defensively so
# a malformed env value can never break the agent import.
try:
    _RESTART_DRAIN_TIMEOUT_S = max(
        0.0, float(os.environ.get("HUGPY_WORKER_RESTART_DRAIN_S", "30")))
except (TypeError, ValueError):
    _RESTART_DRAIN_TIMEOUT_S = 30.0
# Set the instant a restart is requested, so background loops (heartbeat self-
# update, reconcile, provision kicks) stop scheduling NEW work into a process
# that's about to exit — belt-and-suspenders against the "cannot schedule new
# futures" spam (os._exit already skips the atexit teardown that raises it).
_RESTART_EVENT = threading.Event()
# Long-lived executors (e.g. provision's parallel-transfer pool) register here so
# a restart can shut them down first. WeakSet: a finished pool drops out on GC.
_ACTIVE_EXECUTORS: "weakref.WeakSet" = weakref.WeakSet()


def restart_requested() -> bool:
    """True once a restart is underway — background loops check this to stop
    launching new transfers/updates into a process about to exit."""
    return _RESTART_EVENT.is_set()


def register_executor(ex) -> None:
    """Register a long-lived executor so the restart path shuts it down first.
    Best-effort/defensive: a bad object is simply ignored (never breaks a pull)."""
    try:
        _ACTIVE_EXECUTORS.add(ex)
    except Exception:  # noqa: BLE001 — registration must never break the caller
        pass


def _parent_is_systemd() -> bool:
    """True when this process's PARENT is the systemd manager — i.e. systemd
    fork()+exec()'d us directly, so we are a service's MainPID. PID 1 for a
    system unit; the `systemd --user` process for a user unit (computron/op run
    user units). Reading /proc/<ppid>/comm is Linux-only and best-effort."""
    ppid = os.getppid()
    if ppid == 1:
        return True
    try:
        with open(f"/proc/{ppid}/comm", "r", encoding="utf-8") as fh:
            return fh.read().strip() == "systemd"
    except OSError:
        return False


def _under_systemd() -> bool:
    """True iff THIS process is the MainPID of a systemd service — i.e. exiting
    will make systemd's Restart= respawn a fresh, cgroup-tracked process (so the
    restart path must EXIT, not execv).

    Why not just INVOCATION_ID / NOTIFY_SOCKET (the usual signals): both env vars
    AND the `.service` cgroup are INHERITED by every descendant of a systemd
    service. A worker launched inside another service's tree — a test under
    station-keeper.service, a shell under a login scope — would falsely read
    "systemd" and os._exit() out from under itself. So the signal is confirmed by
    the PARENT: systemd launches a service's MainPID directly, so our parent is
    the manager; a descendant's parent is a shell / the ancestor daemon instead.
    This correctly reads True on the already-deployed field units (no unit-file
    change needed) and False for tests/standalone runs.

    Explicit override: HUGPY_WORKER_SYSTEMD=1/0 forces the decision — the
    canonical unit MAY set =1 to be unambiguous; tests set 0/1 to pin a branch.
    """
    forced = os.environ.get("HUGPY_WORKER_SYSTEMD")
    if forced is not None and forced.strip() != "":
        return forced.strip().lower() in ("1", "true", "yes", "on")
    if not (os.environ.get("INVOCATION_ID") or os.environ.get("NOTIFY_SOCKET")):
        return False
    return _parent_is_systemd()


def _drain_generations(timeout_s: float) -> float:
    """Bounded wait for in-flight in-process generations to finish before a
    restart. Polls the gen-gate's TOTAL active permits; returns the seconds
    waited once they hit 0 or ``timeout_s`` elapses — never hangs, and never
    interrupts a native call (we wait for it to release the gate, up to the
    bound, then proceed). Semantics: honor active generations for up to
    ``timeout_s`` (default 30s), then exit regardless (systemd respawns; a client
    mid-stream sees the connection drop, exactly as any restart)."""
    start = time.monotonic()
    deadline = start + max(0.0, timeout_s)
    while True:
        try:
            active = gen_gate.total_in_flight()
        except Exception:  # noqa: BLE001 — can't measure -> don't block the restart
            active = 0
        if active <= 0 or time.monotonic() >= deadline:
            if active > 0:
                logger.warning("restart drain: %d generation(s) still in flight "
                               "after %.1fs — exiting anyway", active, timeout_s)
            return round(time.monotonic() - start, 3)
        time.sleep(0.2)


def _shutdown_executors() -> None:
    """Shut down registered long-lived executors BEFORE exit, so a still-running
    transfer/reconcile thread can't race into 'cannot schedule new futures'.
    Bounded (wait=False) and best-effort; cancels queued futures where the
    runtime supports it (py>=3.9)."""
    for ex in list(_ACTIVE_EXECUTORS):
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:            # cancel_futures added in 3.9
            try:
                ex.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 — one bad executor must not block the rest
            pass


def _close_http_server(state) -> bool:
    """Release the listening socket (:9100) so the respawned process can bind
    without an 'Address already in use' collision. Returns True if a server
    handle was closed. Best-effort: server_close() only closes the listening
    socket fd (safe whether or not serve_forever is running) — we do NOT call
    shutdown() here, which would block forever if serve_forever never started
    (registration-time self-update, tests). os._exit frees the fd regardless;
    this makes the release explicit and testable."""
    srv = getattr(state, "http_server", None)
    if srv is None:
        return False
    try:
        srv.server_close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _prepare_restart(state, *, reason: str, mode: str,
                     kill_slots: bool, drain_timeout_s: "float | None" = None) -> dict:
    """Perform the clean-shutdown steps for a restart and RETURN the plan.

    This does the WORK (flag, drain, executor shutdown, slot teardown, socket
    release) but never exits/execs — the seam (``_restart``) applies the plan's
    mode. Split out so the shutdown sequence is unit-testable without terminating
    the test process.

    ``mode``: 'exit'  — systemd: caller os._exit(plan['exit_code']); Restart=
                        respawns a fresh, cgroup-tracked process.
              'execv' — standalone: caller execs in place (image replaced).
    """
    if drain_timeout_s is None:
        drain_timeout_s = _RESTART_DRAIN_TIMEOUT_S
    _RESTART_EVENT.set()                                   # 1. stop new work
    plan: dict = {"reason": reason, "mode": mode,
                  "exit_code": _RESTART_EXIT_CODE if mode == "exit" else None,
                  "steps": ["shutdown_flag"]}
    plan["drained_wait_s"] = _drain_generations(drain_timeout_s)   # 2. drain
    plan["steps"].append("drained")
    _shutdown_executors()                                          # 3. executors
    plan["steps"].append("executors")
    # 4. Slot children. Under 'exit' they must go: systemd's default
    # KillMode=control-group tears down the whole cgroup on respawn anyway, so a
    # clean terminate here beats an abrupt SIGKILL, and the fresh agent respawns
    # them. Under 'execv' we only kill when asked (self-update: an orphaned slot
    # would keep serving OLD code) — a plain re-exec ADOPTS live slots to avoid a
    # blip, exactly as today.
    if mode == "exit" or kill_slots:
        _kill_slots()
        plan["steps"].append("slots")
    # 5. Listening socket. Only for 'exit' (execv relies on CLOEXEC to drop it,
    # then re-binds fresh — the standalone path kept as today).
    if mode == "exit":
        plan["socket_closed"] = _close_http_server(state)
        plan["steps"].append("socket")
    return plan


def _restart(state, *, reason: str, reexec_fn, kill_slots: bool = False) -> None:
    """Apply a restart: clean shutdown, then EXIT (systemd) or execv (standalone).

    ``reexec_fn`` is resolved by the caller at SCHEDULE time (see
    ``_schedule_restart``) so a monkeypatched ``procutil.reexec`` is honored and a
    late-firing timer can never call the real ``os.execv`` after a test restored
    it. Under systemd this arg is unused (we os._exit instead)."""
    mode = "exit" if _under_systemd() else "execv"
    plan = _prepare_restart(state, reason=reason, mode=mode, kill_slots=kill_slots)
    if mode == "exit":
        logger.info("restart(%s): clean shutdown done %s — exiting %d for systemd "
                    "respawn", reason, plan["steps"], plan["exit_code"])
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(plan["exit_code"])
    logger.info("restart(%s): standalone re-exec in place %s", reason, plan["steps"])
    # Log the EXACT exec target before the handoff, and survive a raising re-exec.
    # os.execv replaces the image and never returns on success; if control comes
    # back here or it raises, the swap did NOT happen. Left unguarded, that
    # exception bubbles into the heartbeat loop's generic "heartbeat failed"
    # swallow, where a still-OLD image keeps beating — the silent half of the
    # 2026-07-20 ae cosmetic-update path. Log LOUDLY and return instead: the
    # heartbeat now reports the honest (stale) running version, so central's
    # version_ok stays FALSE — a visible, diagnosable skew, not a green lie.
    # (SystemExit from the Windows spawn+exit path is BaseException, so it is not
    # caught here and still propagates.)
    try:
        from .._platform.procutil import _module_argv
        _target_argv = _module_argv() or [sys.executable, *sys.argv]
    except Exception:  # noqa: BLE001 — argv preview is best-effort logging only
        _target_argv = [sys.executable, *sys.argv]
    logger.info("restart(%s): exec target argv=%r (running image %s)",
                reason, _target_argv, _RUNNING_IMAGE_VERSION)
    try:
        reexec_fn()
    except Exception as exc:  # noqa: BLE001 — must not bubble into a silent swallow
        logger.error(
            "restart(%s): RE-EXEC FAILED (%s: %s) — image NOT replaced; staying on "
            "the OLD running version %s. Heartbeat reports the honest (stale) "
            "version, so central shows a version skew, not cosmetic convergence. "
            "An explicit /ops/restart or a real unit restart is required to "
            "converge. exec target was argv=%r.",
            reason, type(exc).__name__, exc, _RUNNING_IMAGE_VERSION, _target_argv)
        return
    # A real os.execv never returns; reaching here means a no-op reexec_fn (test
    # seam) — nothing more to do.


def _schedule_restart(state, reason: str, *, kill_slots: bool = False,
                      delay: float = 0.5) -> None:
    """Ack-first restart used by the /ops handlers: schedule ``_restart`` to run
    AFTER the caller sends its HTTP ack (the drain must not block the response).
    ``procutil.reexec`` is resolved NOW so a monkeypatched no-op is captured in
    the timer closure (test safety) and the standalone path honors it."""
    from .._platform.procutil import reexec
    threading.Timer(
        delay,
        lambda: _restart(state, reason=reason, reexec_fn=reexec, kill_slots=kill_slots),
    ).start()


def _apply_spill(spill: dict | None) -> None:
    """Translate a per-request spill override dict into the env vars the spill
    module reads. Only set keys that were provided; the model loads lazily, so
    setting these before the first request for a model takes effect on load.

    NOTE: changing spill for an ALREADY-loaded model has no effect until it's
    evicted/reloaded — central can force that via a fresh worker process or by
    reassigning before first use. For the common case (assign, then chat) the
    override lands before the model is built.
    """
    if not spill:
        return
    for key, env_name in _SPILL_ENV.items():
        if key not in spill or spill[key] is None:
            continue
        val = spill[key]
        if isinstance(val, (list, tuple)):
            val = ",".join(str(x) for x in val)
        os.environ[env_name] = str(val)


def _sse(payload: dict) -> bytes:
    # werkzeug's WSGI server asserts the app yields bytes, not str — so encode
    # here. (gunicorn is more lenient, but the worker runs the dev server.)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# Continuation passes + seam-dedup now live in the shared core engine
# abstract_hugpy_dev.managers.dispatch.execute_chat_stream (honoring the WORKER_*
# env knobs), so the worker no longer carries its own copy.


def _event_to_dict(ev) -> dict:
    """Map a dispatch StreamEvent to the worker's SSE dict shape.

    token/done/error get the slim browser payloads; status/provisioning/
    continuation passthrough events ride through verbatim via model_dump().
    """
    t = getattr(ev, "type", None)
    if t == "token":
        return {"type": "token", "text": getattr(ev, "text", "")}
    if t == "done":
        out = {"type": "done", "finish_reason": getattr(ev, "finish_reason", "stop")}
        # Token accounting (DoneEvent.usage, additive): forward when the engine
        # reported it so central's /v1 usage object is real for relayed chats.
        usage = getattr(ev, "usage", None)
        if isinstance(usage, dict) and usage:
            out["usage"] = usage
        return out
    if t == "error":
        return {"type": "error", "message": getattr(ev, "message", "run failed")}
    try:
        return ev.model_dump()
    except Exception:
        return {"type": str(t or "status")}


def _stream_sync(payload: dict, request_id: str | None = None):
    """Relay the shared chat engine as SSE from Flask's sync context.

    Auto-continuation + seam-dedup now live in the core
    ``abstract_hugpy_dev.managers.dispatch.execute_chat_stream`` engine — the exact
    same one the central node drives — so worker chat and local chat behave
    identically. This wrapper only materializes an inlined upload, registers the
    request in the shared comms JobStore with a cancel handle (POST
    /infer/cancel trips it — same F5 substrate central uses, no private
    cancel dict), drives the async engine in a sync loop, and encodes each
    StreamEvent as an SSE line.
    """
    from .._platform import async_runtime
    from ..comms import job_store
    tmp = _materialize_file(payload)

    # Register a cancel Event for this request so /infer/cancel can trip it, and
    # thread its id through the engine so all continuation passes share it. The
    # Event binds to the shared runtime loop on first await; cancellation sets
    # it via call_soon_threadsafe (cross-thread set is otherwise unsafe).
    cancel_event = asyncio.Event()
    if request_id:
        try:
            existing = job_store.get(request_id)
            if existing is None or existing.terminal:
                job_store.create(str(payload.get("model_key") or ""),
                                 id=request_id, kind="chat", transport="worker")
            job_store.attach_cancel(
                request_id,
                lambda: async_runtime.call_soon_threadsafe(cancel_event.set))
        except Exception:
            pass
        payload.setdefault("request_id", request_id)

    agen = None
    try:
        agen = execute_chat_stream(cancel_event=cancel_event, **payload)
        # Drive on the process-wide async runtime (one long-lived loop) instead
        # of a fresh per-request loop — fixes "bound to a different event loop"
        # for any cached asyncio primitive. iter_sync owns step-cancel + aclose.
        for event in async_runtime.iter_sync(agen):
            if request_id and getattr(event, "type", None) == "token":
                try:
                    job_store.on_output(request_id)
                except Exception:
                    pass
            yield _sse(_event_to_dict(event))
    except Exception as exc:
        # Last-resort guard: never let an exception escape into the WSGI layer
        # (that aborts the stream with a raw traceback). Emit a clean error.
        logger.warning("stream failed: %s: %s", type(exc).__name__, exc)
        if request_id:
            try:
                job_store.finish(request_id, error=exc)
            except Exception:
                pass
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        if request_id:
            # done, or cancelled if a cancel was requested; no-op if the
            # except above already marked it failed.
            try:
                job_store.finish(request_id)
            except Exception:
                pass
        _cleanup_file(tmp)


def loaded_model_keys() -> list[str]:
    try:
        from ..managers.dispatch import loaded_model_keys as _loaded
        keys = {mk for (mk, _task) in _loaded()}
        # De-dup slot-vs-in-process: a GGUF model seated in a slot leaves a
        # HOLLOW LlamaCppChatRunner in dispatch _INSTANCES whose underlying
        # runner is an HTTP proxy to the slot child (no weights in THIS
        # process). Reporting it as an in-process ('ram'/'loaded') resident is
        # what makes a slot-served model ALSO read 'loaded' and FLAP with its
        # slot 'serving' row. Prefer the slot — report only genuine in-process
        # residents. Slot occupants stay protected via the _slot_occupants()
        # unions at the storage/residency callers and appear as slot rows in
        # allocations. Discriminating on runner TYPE (not the transient per-beat
        # slot snapshot) removes the flap entirely.
        try:
            from ..managers.llama.runners.get import slot_backed_model_keys
            keys -= slot_backed_model_keys()
        except Exception:
            pass
        return sorted(keys)
    except Exception:
        return []


def _loading_model_keys() -> list[str]:
    """Models whose weights are LOADING right now — the console's 'heating'."""
    try:
        from ..managers.dispatch.dispatch import loading_model_keys
        return loading_model_keys()
    except Exception:
        return []


def _path_bytes(path: str) -> int:
    """On-disk bytes of a model path, NOT following symlinks (a symlinked comfy
    checkpoint or shared file costs this box ~nothing to keep)."""
    try:
        if not path or not os.path.exists(path):
            return 0
        if os.path.islink(path):
            return 0
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    except OSError:
        return 0


def _store_root_copy_path(mk: str, cfg) -> str:
    """The path of this model's copy under the WORKER'S OWN store root, when one
    exists and is complete — regardless of what get_model_path's read-through
    prefers.

    THE ae READ-THROUGH GAP (2026-07-17, slice 3). get_model_path resolves via
    _resolved_local, which honours a NAS read-through: on a box like ae whose
    DEFAULT_ROOT is the HOT drive but which ALSO mounts the shared/central NAS
    (carrying the .hugpy-central-catalog sentinel), a model can resolve to its
    NAS copy even though a re-promotable copy sits on the hot store root. That
    NAS path classifies "shared/central — never reaped" -> protected -> the hot
    copy never becomes an eviction candidate and _path_bytes counts the NAS size.

    The reaper must classify the STORE-ROOT copy: resolve the model's dir with
    the resolver PINNED to this worker's own store root (never the NAS), so the
    reap row's path+bytes and its protection verdict are evaluated on the hot
    copy (hot990 -> not shared -> reapable when the box flag is on). Returns ""
    when no complete copy exists under the store root (then the caller falls back
    to get_model_path — a model served straight from the NAS has no hot row).

    Best-effort and never raises: any failure returns "" and the caller uses the
    read-through path as before, so this can only ADD hot-copy candidates, never
    remove an existing protection.
    """
    try:
        root = _models_store_root()          # e.g. /mnt/hot990/hugpy-worker/models
        if not root:
            return ""
        # _models_store_root returns the MODELS dir; resolve_model_dir expects the
        # DEFAULT_ROOT (it joins "models"/ itself), so hand it the parent when the
        # root ends in a models/ component, else the root as-is.
        base = os.path.dirname(root) if os.path.basename(root) == "models" else root
        from ..imports.src.constants.paths import resolve_model_dir
        routing = {
            "hub_id": getattr(cfg, "hub_id", None),
            "framework": getattr(cfg, "framework", None),
            "filename": getattr(cfg, "filename", None),
            "include": getattr(cfg, "include", None),
            "primary_task": getattr(cfg, "primary_task", None),
            "tasks": getattr(cfg, "tasks", None),
            "folder": getattr(cfg, "folder", None),
        }
        # require_complete=True: only a COMPLETE store-root copy is an evictable
        # row. An incomplete/absent hot copy -> "" -> caller uses read-through.
        d = resolve_model_dir(routing, root=base, cfg=cfg, require_complete=True)
        if not d:
            return ""
        # Guard: the resolver is pinned to `base`, but be defensive — only accept
        # a dir that really lives under the store root (never a NAS path that
        # slipped through a symlinked candidate).
        rp = os.path.realpath(d)
        root_rp = os.path.realpath(base)
        if rp == root_rp or rp.startswith(root_rp + os.sep):
            return d
        return ""
    except Exception:  # noqa: BLE001 — never raise into the scan
        return ""


def _reap_scan(state: "WorkerState") -> dict:
    """The reaper's read-only survey: which local model files are RECLAIMABLE
    (on disk, but not static / loaded / loading / provisioning, on a reapable
    non-shared store), and which are PROTECTED (and why). Never touches comfy
    rows — those are symlinks into the operator's ComfyUI, not this worker's
    downloads.

    📌 pin AND assignment are NOT protection reasons here (operator, 2026-07-17):
    both designate only ROUTING/attribution that survives restarts — no bearing on
    eviction. An assigned OR pinned model's files are reclaimable; the assignment/
    pin survives the delete and the bytes re-pull on next call. (Slice 7 aligned
    the bulk path with budget._is_protected and central's proposal chain.)

    This is advisory: /reap re-checks every guard at delete time, because state
    (a load, a pull) can change between preview and reclaim.
    """
    try:
        from .imports import get_models_dict, get_model_config, get_model_path
        from .provision import (model_is_local, _on_shared_model_store,
                                _model_store_reapable)
    except Exception as exc:  # noqa: BLE001
        # HONESTY (slice 3, B): a scan that couldn't even import must NOT read as
        # a clean empty store. Carry the error so _worker_storage/central surface
        # "scan broken", never rows:0 masquerading as "nothing on disk".
        return {"reclaimable": [], "protected": [], "error": str(exc),
                "scan_keys_considered": 0, "scan_rows": 0}

    assigned = set(state.assigned_models or [])
    # HARDEN (reaper guard): fold in slot-seated / answering models.
    # loaded_model_keys() is in-process only and MISSES models seated in the
    # slot pool that are actively serving a request. Union _slot_occupants()
    # so an approved reap can never delete a resident/answering model even if
    # it slipped onto the approved list between preview and reclaim.
    loaded = set(loaded_model_keys()) | _slot_occupants()
    loading = set(_loading_model_keys())

    reclaimable, protected = [], []
    scan_error = ""
    try:
        # Enumerate the UNION, not just get_models_dict(): on a WORKER that dict is
        # the built-in staples + comfy sweep + on-disk discovery report and NEVER
        # includes MODEL_REGISTRY, so models held purely by CENTRAL ASSIGNMENT
        # (discovered rows — a designated gguf like flux2) were surveyed as ABSENT,
        # dropping tens of GB from the storage report though they're on disk, in
        # models_local, and slot-seated. Fold in assigned + loaded + loading; the
        # loop below skips any key that isn't model_is_local, so this only ADDS
        # on-disk models the staple-only dict missed.
        registry_keys = set(get_models_dict().keys())
    except Exception as exc:  # noqa: BLE001
        # DON'T abandon the scan (slice 3, A defense): even if the registry build
        # blew up (e.g. a discovery report unreadable for a process whose $HOME /
        # store root wasn't ready at import), assignment + slot truth still name
        # real on-disk models. Record the error, proceed with the key set we can
        # trust, and fold in _models_local so central-registered copies resolve.
        registry_keys = set()
        scan_error = f"get_models_dict: {exc}"
    try:
        local_keys = set(_models_local(state))       # assigned ∩ on-disk (cached)
    except Exception:  # noqa: BLE001
        local_keys = set()
    keys = registry_keys | assigned | loaded | loading | local_keys

    considered = len(keys)
    row_errors = 0
    # SKIP-REASON HISTOGRAM (slice 5). Cheap, permanent per-key accounting of why
    # a considered key produced NO row. This is what would have named the ae
    # 2026-07-17 incident (74 considered, 0 rows) in a single heartbeat:
    # {"not_local": 74} points straight at presence, distinguishing it from
    # {"no_config": 74} (registry/resolution) or {"comfy": N}. Rows that DO
    # classify are not counted here (they land in reclaimable/protected).
    skip_reasons: dict[str, int] = {}

    def _skip(reason: str):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for mk in keys:
        try:
            cfg = get_model_config(mk)
        except Exception:
            _skip("no_config")
            continue
        if getattr(cfg, "framework", None) == "comfy":
            _skip("comfy")
            continue  # symlinks / operator-owned — reaper stays clear
        try:
            if not model_is_local(mk):
                _skip("not_local")
                continue
        except Exception:
            row_errors += 1
            _skip("locality_error")
            continue
        # STORE-ROOT COPY classification (slice 3, C). Prefer the copy under THIS
        # worker's own store root over get_model_path's read-through, which on ae
        # can hand back a NAS path (shared/central -> protected) even when a
        # re-promotable hot copy exists. Evaluate path+bytes+protection on the hot
        # copy so it becomes a real candidate; fall back to the read-through path
        # when there is no complete store-root copy (a model served straight off
        # the NAS legitimately has no evictable hot row).
        try:
            path = _store_root_copy_path(mk, cfg)
        except Exception:  # noqa: BLE001
            path = ""
        if not path:
            try:
                path = get_model_path(mk) or ""
            except Exception:
                path = ""
        size = _path_bytes(path)
        # SAFE-BY-DEFAULT: a model is reapable only on a box that declared its
        # store local & disposable AND is not shared/central. Everything else is
        # PROTECTED here, so nothing on a shared/unconfigured box is ever proposed
        # — the console still shows usage but offers no deletion. wipe_model
        # re-checks the same gate at delete time.
        rp = os.path.realpath(path) if path else ""
        if not _model_store_reapable(rp):
            why = ("shared/central storage — never reaped"
                   if (not rp or _on_shared_model_store(rp))
                   else "model store not marked reapable")
            protected.append({"model_key": mk, "bytes": size, "why": why})
            continue
        # 📌 pin AND assignment do NOT protect files (operator, 2026-07-17):
        # both designate only ROUTING/attribution that survives restarts — neither
        # has any bearing on eviction/reaping. An assigned OR pinned model's files
        # are reclaimable like any other; the assignment/pin survives the delete
        # and the bytes re-pull on the next call. This is the BULK path catching
        # up to the call-driven path (budget._is_protected dropped `assigned` at
        # f1894b2): the preview and executor must agree with central's proposal
        # chain, which already proposes assigned-but-cold models — otherwise an
        # over-budget box whose cold models are ALL assigned (the normal case: ae
        # designates everything) can never be cleared (slice 7, the "7 refused
        # assigned" incident). ONLY the DURABLE-presence + live-use guards remain:
        # 🔒static, loaded/loading; provisioning is guarded at reclaim time (a
        # mid-pull delete corrupts the fetch). The store-reapable + shared/central
        # sentinel gates above already ran; the path jail runs at wipe time.
        if _residency(mk) == "static":
            protected.append({"model_key": mk, "bytes": size, "why": "static"})
        elif mk in loaded or mk in loading:
            protected.append({"model_key": mk, "bytes": size, "why": "loaded"})
        else:
            # Assigned-but-cold falls through here — a reclaimable candidate.
            # Store-root copy path travels on the row so _reap_reclaim/wipe act on
            # the hot copy — never the NAS (re-proven at delete time).
            reclaimable.append({"model_key": mk, "bytes": size, "path": path})

    reclaimable.sort(key=lambda r: r["bytes"], reverse=True)
    out = {
        "reclaimable": reclaimable,
        "protected": protected,
        "reclaimable_bytes": sum(r["bytes"] for r in reclaimable),
        # DIAGNOSTICS (slice 3, B): make a broken/empty scan self-describing so it
        # can never masquerade as a clean empty store. scan_keys_considered = the
        # full key domain; scan_rows = rows actually classified (reclaimable +
        # protected). considered≫rows with 0 rows is the ae symptom's fingerprint.
        "scan_keys_considered": considered,
        "scan_rows": len(reclaimable) + len(protected),
        "scan_row_errors": row_errors,
        # SKIP-REASON HISTOGRAM (slice 5): why each considered key produced no
        # row — {"not_local": N, "no_config": N, "comfy": N, "locality_error": N}.
        # considered≫rows is now self-explaining in one heartbeat.
        "scan_skip_reasons": skip_reasons,
    }
    if scan_error:
        out["error"] = scan_error
    return out


def _reap_reclaim(state: "WorkerState", model_keys: list[str]) -> dict:
    """Delete the local files of the named models — but ONLY after re-proving,
    per key, that it is still reclaimable (not static/loaded/loading/provisioning,
    not comfy, on a reapable non-shared store). The guard is re-run here, not
    trusted from a stale preview.

    📌 pin AND assignment are deliberately NOT in that list (operator,
    2026-07-17): both designate only ROUTING/attribution that survives restarts —
    no bearing on eviction. A pinned or assigned model's files reap freely; the
    pin/assignment survives the delete and the bytes re-pull on next call. (This
    matches budget._is_protected and central's proposal chain — see slice 7.)"""
    from .imports import get_model_config, get_model_path
    from .provision import model_is_local, wipe_model

    # HARDEN (reaper guard): fold in slot-seated / answering models.
    # loaded_model_keys() is in-process only and MISSES models seated in the
    # slot pool that are actively serving a request. Union _slot_occupants()
    # so an approved reap can never delete a resident/answering model even if
    # it slipped onto the approved list between preview and reclaim. FAIL CLOSED:
    # if the slot probe can't answer, refuse the whole reclaim rather than delete
    # while blind to what's resident.
    try:
        slot_occ = _slot_occupants(strict=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "results": [],
                "error": f"slot occupancy unknown ({exc}) — refusing to reap (fail-closed)"}
    loaded = set(loaded_model_keys()) | slot_occ
    loading = set(_loading_model_keys())
    # Models mid-provision (downloading from central/HF) — deleting under a live
    # pull corrupts the fetch; guard explicitly instead of via assigned-coupling.
    provisioning = set(getattr(state, "_provisioning", None) or [])

    results = []
    for mk in model_keys:
        try:
            cfg = get_model_config(mk)
        except Exception:
            results.append({"model_key": mk, "ok": False, "reason": "unknown model"})
            continue
        if getattr(cfg, "framework", None) == "comfy":
            results.append({"model_key": mk, "ok": False, "reason": "comfy (symlink/operator files) — never reaped"})
            continue
        # 📌 pin AND assignment are NOT reap guards (operator, 2026-07-17): both
        # mean only that ROUTING/attribution survives restarts — no bearing on
        # eviction. A pinned OR assigned model's files are reclaimable; the pin/
        # assignment survives the delete and the bytes re-pull on next call. This
        # mirrors budget._is_protected (which dropped `assigned` at f1894b2) so
        # the executor agrees with central's proposal and the scan preview — the
        # slice-7 fix for "central proposes what the worker refuses". static +
        # loaded/loading + provisioning still guard below; the store-reapable/
        # shared-sentinel gates re-run inside wipe_model's path jail.
        if _residency(mk) == "static":
            results.append({"model_key": mk, "ok": False, "reason": "static"})
            continue
        if mk in provisioning:
            results.append({"model_key": mk, "ok": False, "reason": "provisioning"})
            continue
        if mk in loaded or mk in loading:
            results.append({"model_key": mk, "ok": False, "reason": "loaded/serving/loading"})
            continue
        try:
            if not model_is_local(mk):
                results.append({"model_key": mk, "ok": False, "reason": "no local files"})
                continue
        except Exception:
            results.append({"model_key": mk, "ok": False, "reason": "locality check failed"})
            continue
        # STORE-ROOT COPY (slice 3, C): target the hot copy the scan classified,
        # not get_model_path's read-through (which on ae resolves to the NAS the
        # shared gate correctly refuses). Re-resolve it here — same helper the
        # scan used — so preview and delete can't diverge; fall back to the
        # read-through path when there is no complete store-root copy. wipe_model
        # re-proves the jail + shared gate on whichever realpath this is.
        try:
            target = _store_root_copy_path(mk, cfg)
        except Exception:  # noqa: BLE001
            target = ""
        if not target:
            try:
                target = get_model_path(mk) or ""
            except Exception:
                target = ""
        try:
            freed = _path_bytes(target)
        except Exception:
            freed = 0
        gone = wipe_model(mk, path=target)  # jailed + shared-gate re-proven on target
        _MODELS_LOCAL_CACHE["at"] = 0.0  # force a fresh local walk next beat
        results.append({"model_key": mk, "ok": bool(gone),
                        "freed_bytes": freed if gone else 0,
                        "reason": "" if gone else "delete refused/failed (path jail?)"})
    return {"ok": True, "results": results,
            "freed_bytes": sum(r.get("freed_bytes", 0) for r in results)}


# ── per-worker local-STORAGE survey (heartbeat) ─────────────────────────────
# What model cache this box holds: total on-disk bytes, per-model sizes, and
# which models are PROTECTED (and why). The worker reports the flags only IT can
# know — loaded / slot-seated / loading / provisioning / assigned — plus sizes.
# Central OVERLAYS the two facts the worker cannot know (per-(worker,model)
# last_picked and the disk budget) and derives over_budget + eviction proposals
# in _public_view. This is REUSED from the reaper's own _reap_scan so the
# storage view can never disagree with what the reaper would actually delete.
_STORAGE_CACHE: dict = {"at": 0.0, "value": None}

# RELEASE-BOUND (2026-07-17): measured store-root size, TTL-cached separately so
# the (cheap) scandir walk of the store root is not tied to the 60s model-scan
# cache. This is the AUTHORITATIVE cache_used_bytes — a real filesystem
# measurement of what is on disk under the model store, fixing the 2026-07-16
# discrepancy where the per-model-dir SUM read 128.8GB while `du` measured 81G
# (the sum double-counted / carried stale manifest keys).
_STORE_MEASURE_CACHE: dict = {"at": 0.0, "value": None}


def _models_store_root() -> str | None:
    """The directory the worker's model weights actually live under."""
    try:
        from ..imports.src.constants.constants import MODELS_HOME
        if MODELS_HOME and os.path.isdir(MODELS_HOME):
            return str(MODELS_HOME)
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..imports.src.constants.constants import DEFAULT_ROOT
        cand = os.path.join(str(DEFAULT_ROOT), "models")
        if os.path.isdir(cand):
            return cand
        if os.path.isdir(DEFAULT_ROOT):
            return str(DEFAULT_ROOT)
    except Exception:  # noqa: BLE001
        pass
    return None


def _measured_store_bytes() -> int | None:
    """Real on-disk bytes under the model store root — a scandir walk, TTL-cached
    (120s). Non-following of symlinks so shared/comfy links cost 0 here (same rule
    as _path_bytes). Returns None if the root can't be resolved (caller then falls
    back to the per-model sum). This is the honest cache_used the heartbeat ships.
    """
    now = time.time()
    cached = _STORE_MEASURE_CACHE["value"]
    if cached is not None and now - _STORE_MEASURE_CACHE["at"] < 120.0:
        return cached
    root = _models_store_root()
    if not root:
        return None
    total = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    _STORE_MEASURE_CACHE.update(at=now, value=total)
    return total


# ── ORPHAN (unattributed-on-disk) scan ──────────────────────────────────────
# RELEASE-BOUND (2026-07-17 addendum). The reaper survey (_reap_scan) only ever
# looks at KNOWN/assigned/loaded keys, so on-disk residue that matches NO current
# model — a stalled *.part set from an old eager-era pull, or a whole model dir
# for something no longer assigned — is INVISIBLE to central (computron held 5.7G
# of stalled Qwen2.5-VL-3B .part files that appeared nowhere in the UI). This
# scan walks the store root for that residue and reports it so the console can
# surface "unattributed on disk: X GB". Naming (keeper owns nomenclature): the
# class is "orphaned" in code; the UI labels it "unattributed on disk".
_ORPHAN_CACHE: dict = {"at": 0.0, "value": None}


def _dir_bytes_no_links(path: str) -> int:
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _orphan_scan(state: "WorkerState", known_keys: set) -> dict:
    """On-disk residue attributed to NO current model. TTL-cached (120s).

    Two orphan shapes:
      * a completed model DIR (is_model_dir) whose hub_id/key is in neither the
        manifest nor the assignment/loaded set — a leftover from a prior
        assignment that was never reaped;
      * a STALLED partial: a dir under the store holding *.part staging files
        (a crash/abandoned pull) — is_model_dir() is False for it (no real
        weights), so nothing else sees it.

    ``known_keys`` = every name a live model might match (manifest keys + hub_ids
    + assigned/loaded/loading). Anything on disk NOT matching one of these is
    orphaned. Conservative: an entry we cannot positively tie to a known model is
    reported (visible), never auto-deleted — this scan proposes nothing.

    MATCHING (fixed 2026-07-17, over-report root cause): a dir is compared
    against ``known_keys`` through ONE shared expansion —
    ``provision.known_model_dir_forms`` — the same normalization
    ``model_is_local``/the read-through resolver are built on. The old
    comparison only lowercased+stripped, so a ``~``-qualified assignment key
    (``owner~repo``, minted on an owner collision — see discover_models in
    imports/apis/get_module.py) never textually matched a directory-derived
    ``owner/repo`` name; it happened to still "work" only via an accidental
    bare-repo-name fallback, which breaks the moment two owners share a repo
    basename (exactly the case ``~`` exists to disambiguate). The dir is
    matched BOTH by its relative path (catches a legacy nested-path copy of a
    known model — legacy-path, not orphan) and by its best-effort hub-id guess
    (catches a flat/marker-named dir). ``misc/comfy/**`` is excluded by policy
    (provision.is_doctrine_excluded) — comfy checkpoints are symlinks the
    reaper/storage accounting already treat as never-orphaned; operator
    doctrine: "comfy is excluded from allocations, models can sit on the drive
    unattributed."
    """
    now = time.time()
    cached = _ORPHAN_CACHE["value"]
    if cached is not None and now - _ORPHAN_CACHE["at"] < 120.0:
        return cached

    root = _models_store_root()
    out = {"items": [], "bytes": 0, "count": 0}
    if not root:
        _ORPHAN_CACHE.update(at=now, value=out)
        return out

    try:
        from ..imports.src.constants.paths import (
            is_model_dir, is_directory_excluded, get_hub_id_from_directory,
        )
        from .provision import (
            known_model_dir_forms, dir_is_known_model, is_doctrine_excluded,
            _dir_slug,
        )
    except Exception:  # noqa: BLE001 — never break a heartbeat over this
        _ORPHAN_CACHE.update(at=now, value=out)
        return out

    known_forms = known_model_dir_forms(known_keys)

    items: list[dict] = []
    seen_dirs: set = set()

    def _is_orphan_dir(dirpath: str) -> bool:
        rel = os.path.relpath(dirpath, root)
        if is_doctrine_excluded(rel):
            return False
        if dir_is_known_model(rel, known_forms):
            return False
        # Secondary check: the best-effort hub-id guess (marker-first, then
        # layout-aware path guess) — catches a dir whose relative path doesn't
        # literally match a candidate dir (e.g. a legacy shape the resolver
        # doesn't enumerate) but whose declared/guessed hub_id still resolves.
        hub = get_hub_id_from_directory(dirpath, models_home=root)
        if hub and _dir_slug(hub) in known_forms:
            return False
        if hub:
            tail = str(hub).rsplit("/", 1)[-1]
            if _dir_slug(tail) in known_forms:
                return False
        return True

    # Walk once: catch model dirs (leaves) AND .part-bearing dirs.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(os.path.join(dirpath, d))]
        has_part = any(n.endswith((".part", ".part.state.json")) for n in filenames)
        model_leaf = is_model_dir(dirpath)
        if not (model_leaf or has_part):
            continue
        if dirpath in seen_dirs:
            continue
        if _is_orphan_dir(dirpath):
            b = _dir_bytes_no_links(dirpath)
            if b > 0:
                items.append({
                    "path": os.path.relpath(dirpath, root),
                    "bytes": b,
                    "kind": "partial" if (has_part and not model_leaf) else "stale-dir",
                })
            seen_dirs.add(dirpath)
        if model_leaf:
            dirnames[:] = []   # leaf — don't descend into a model's own files

    items.sort(key=lambda x: x["bytes"], reverse=True)
    out = {"items": items,
           "bytes": sum(i["bytes"] for i in items),
           "count": len(items)}
    _ORPHAN_CACHE.update(at=now, value=out)
    return out


def _storage_model_row(mk: str, size: int, loaded: set, loading: set,
                       provisioning: set, assigned: set,
                       why_hint: str = "") -> dict:
    """One per-model row for the heartbeat storage view: bytes + every
    protection flag + a human `why`. loaded is ALREADY answer-inclusive
    (loaded_model_keys() ∪ _slot_occupants()) at the caller."""
    is_pinned = _pinned(mk)
    is_static = _residency(mk) == "static"
    is_loaded = mk in loaded
    is_loading = mk in loading
    is_provisioning = mk in provisioning
    is_assigned = mk in assigned
    # `why_hint` carries _reap_scan's STORE-GATE verdict ("shared/central storage
    # — never reaped" / "model store not marked reapable"). It is a GENUINE
    # protection reason and the only one this row cannot re-derive from the flags
    # below — it comes from _model_store_reapable(realpath), which the caller
    # already resolved. Honour it.
    #
    # BUG (fixed 2026-07-17): `protected` used to ignore why_hint entirely, so a
    # model protected ONLY by the store gate shipped to central as
    # protected=False, why="" — an UNPROTECTED row with NO reason. On ae that
    # silently mislabelled 101 store-gated models as eviction candidates, and
    # fit_plan's refusal then reported "0 B reclaimable (1 loaded)" — hiding the
    # real cause (the whole store is gated) behind a number that made the FIFO
    # look broken. The policy was always right; only this report lied.
    store_gated = bool((why_hint or "").strip())
    # 📌 pin AND assignment do NOT protect files (operator, 2026-07-17): both are
    # ROUTING/attribution that survives restarts — no bearing on eviction. Both
    # are still reported below as ATTRIBUTION (`pinned`/`assigned` flags + `why`),
    # but NEITHER sets `protected`. This aligns the heartbeat row with _reap_scan,
    # _reap_reclaim, and budget._is_protected (all of which treat assigned as a
    # candidate) so preview, executor, and central agree (slice 7). 🔒static is the
    # durable local-presence guard; loaded/loading/provisioning are live-use
    # guards; the store gate (why_hint) is a hard filesystem fact.
    protected = (is_static or is_loaded or is_loading
                 or is_provisioning or store_gated)
    # Precedence for the human `why` (static > store-gate > loaded > loading >
    # provisioning > assigned > pinned). The store gate outranks the attribution
    # labels DELIBERATELY: those files CANNOT be deleted by the reaper regardless
    # of central's verdict, so a real, enforced reason must beat a routing label.
    # `assigned`/`pinned` read purely as ATTRIBUTION (protected stays False) — a
    # bare-assigned or bare-pinned row is a reclaimable CANDIDATE.
    if is_static:
        why = "static"
    elif store_gated:
        why = why_hint
    elif is_loaded:
        why = "loaded"
    elif is_loading:
        why = "loading"
    elif is_provisioning:
        why = "provisioning"
    elif is_assigned:
        why = "assigned"        # attribution only — protected stays False
    elif is_pinned:
        why = "pinned"          # attribution only — protected stays False
    else:
        why = ""
    return {
        "model_key": mk,
        "bytes": int(size or 0),
        "pinned": is_pinned,
        "loaded": is_loaded,
        "loading": is_loading,
        "provisioning": is_provisioning,
        "assigned": is_assigned,
        "protected": protected,
        "why": why,
    }


def _refused_snapshot(state: "WorkerState") -> dict:
    """A copy of the storage-REFUSED models, pruned of any that since landed.

    A refusal is a point-in-time verdict: once the model's files are on disk
    (the operator raised disk_cache_gib, or a later pull fit after evictions),
    the "missing — won't fit" reason is stale and must not linger in the console.
    """
    out = {}
    for mk, reason in list(getattr(state, "refused", {}).items()):
        try:
            from .provision import model_is_local
            if model_is_local(mk):
                state.refused.pop(mk, None)
                continue
        except Exception:  # noqa: BLE001 — a probe failure keeps the reason
            pass
        out[mk] = dict(reason)
    return out


def _worker_storage(state: "WorkerState") -> dict:
    """Heartbeat STORAGE view for one worker (60s-cached — _reap_scan os.walks
    every local model dir via _path_bytes; running that every beat would slow
    heartbeats on boxes with many large models).

    Shape:
      { cache_used_bytes:int,  # sum of on-disk bytes of ALL local models
                               # (reclaimable + protected); symlinks count 0
        disk_free:int,         # = disk.free_bytes (kept for console convenience)
        models:[ {model_key, bytes, pinned, loaded, loading, provisioning,
                  assigned, protected, why} ] }

    Comfy rows never appear (skipped by _reap_scan — operator symlinks), so they
    neither inflate cache_used_bytes nor get proposed for eviction.
    """
    now = time.time()
    cached = _STORAGE_CACHE["value"]
    if cached is not None and now - _STORAGE_CACHE["at"] < 60.0:
        # The heavy part (per-model disk walk) stays cached, but REFUSALS are
        # cheap and time-critical: a model refused seconds ago must read as
        # missing-with-a-reason on the NEXT beat, not up to 60s later. Refresh
        # just that key on the cached view.
        cached["refused"] = _refused_snapshot(state)
        return cached

    scan = _reap_scan(state)
    # Cheap set-membership truth (no disk walk) for the per-model flags. `loaded`
    # is answer-inclusive: loaded_model_keys() misses slot occupants, so union
    # _slot_occupants() to protect a model that is seated/answering.
    loaded = set(loaded_model_keys()) | _slot_occupants()
    loading = set(_loading_model_keys())
    try:
        provisioning = set(state._provisioning)
    except Exception:  # noqa: BLE001
        provisioning = set()
    assigned = set(state.assigned_models or [])

    models: list[dict] = []
    cache_used = 0
    for row in scan.get("reclaimable", []):
        size = int(row.get("bytes", 0) or 0)
        cache_used += size
        models.append(_storage_model_row(row.get("model_key"), size, loaded,
                                         loading, provisioning, assigned))
    for row in scan.get("protected", []):
        size = int(row.get("bytes", 0) or 0)
        cache_used += size
        models.append(_storage_model_row(row.get("model_key"), size, loaded,
                                         loading, provisioning, assigned,
                                         why_hint=row.get("why", "")))
    models.sort(key=lambda m: m["bytes"], reverse=True)

    disk = _disk_status()
    # AUTHORITATIVE cache_used = MEASURED store-root bytes (release-bound). The
    # per-model dir SUM (`cache_used` here) over-counted in the field (op:
    # 128.8GB summed vs 81G measured), so a real filesystem measurement of the
    # store root is the honest number the gauge should read. Keep the sum as a
    # cross-check diagnostic; fall back to it only if the root can't be measured.
    measured = _measured_store_bytes()
    # ORPHANED (unattributed-on-disk) residue — model dirs / stalled .part sets
    # that match NO current model. Every name a live model might be known by, so
    # the diff doesn't false-flag a resident model as orphaned.
    try:
        known_keys = (set(get_models_dict().keys()) | set(models and [m["model_key"] for m in models] or [])
                      | set(state.assigned_models or []) | loaded | loading | provisioning)
        # fold in each known model's hub_id so a dir named by hub_id matches.
        for _mk in list(known_keys):
            try:
                _c = get_model_config(_mk)
                _h = getattr(_c, "hub_id", None) if _c is not None else None
                if _h:
                    known_keys.add(_h)
            except Exception:  # noqa: BLE001
                continue
        orphans = _orphan_scan(state, known_keys)
    except Exception:  # noqa: BLE001 — a heartbeat must never fail on this
        orphans = {"items": [], "bytes": 0, "count": 0}
    # EFFECTIVE BUDGET (slice 4, min-wins). Resolve the min over {central
    # disk_cache_gib, worker same-drive declarations} for THIS box's store root
    # and report the number + source map so the operator sees WHY a number
    # governs. Not applicable on a shared/central store (the cap is skipped
    # there — slice 2), so we mark it and omit the sources rather than imply a
    # cap. Best-effort: any failure just omits the fields.
    budget_effective_bytes = None
    budget_sources: dict = {}
    budget_not_applicable = False
    try:
        from . import budget as _budget
        if _budget._store_is_shared():
            budget_not_applicable = True
        else:
            store_root = _models_store_root() or ""
            budget_effective_bytes, budget_sources = _budget.resolve_effective_cap(
                getattr(state, "limits", None) or {}, store_root)
    except Exception:  # noqa: BLE001 — a heartbeat must never fail on this
        pass
    out = {
        "cache_used_bytes": measured if measured is not None else cache_used,
        "cache_used_measured_bytes": measured,      # None if root unresolved
        "cache_used_model_sum_bytes": cache_used,   # legacy per-model-dir sum
        # Orphaned residue (release-bound). UI labels it "unattributed on disk".
        "orphaned_bytes": orphans["bytes"],
        "orphaned_count": orphans["count"],
        "orphaned_items": orphans["items"],
        "disk_free": int(disk.get("free_bytes", 0) or 0),
        # EFFECTIVE per-drive budget (slice 4). budget_sources names every term
        # in GiB (central_gib / worker_hot_cache_gib / …) plus effective_gib +
        # effective_source. Shared store -> budget_cap_not_applicable True.
        "budget_effective_bytes": budget_effective_bytes,
        "budget_sources": budget_sources,
        "budget_cap_not_applicable": budget_not_applicable,
        "models": models,
        # Models REFUSED for storage: the pull never started because even a full
        # FIFO couldn't seat them. {model_key: {state:"refused", reason, ...}}.
        # The console renders these as MISSING with the reason on hover — an
        # honest "won't fit", never a phantom "pulling" that can't finish.
        "refused": _refused_snapshot(state),
        # HOT-CACHE tier (box-local NVMe LRU of the main catalog). Honest section
        # so central/console can surface root/budget/used + per-entry last_called.
        # {"enabled": False} when HUGPY_HOT_CACHE_ROOT is unset (no behaviour).
        "hot_cache": _hot_cache_status(),
        # SCAN DIAGNOSTICS (slice 3, B). Carry the reaper survey's own telemetry
        # so a broken/degraded scan can NEVER masquerade as a clean empty store
        # (the ae 2026-07-17 defect: rows:0 while 65 models were on disk, because
        # a swallowed scan error surfaced identically to "nothing here"). Central
        # passes these through verbatim; the console can surface them later.
        #   scan_error            — set when the registry build failed (scan still
        #                           ran on assignment/slot/local keys)
        #   scan_keys_considered  — size of the full key domain the scan walked
        #   scan_rows             — rows actually classified (reclaimable+protected)
        #   scan_row_errors       — per-model probe failures skipped
        # considered≫0 with rows:0 is the fingerprint of the ae failure.
        "scan_error": scan.get("error") or "",
        "scan_keys_considered": int(scan.get("scan_keys_considered") or 0),
        "scan_rows": int(scan.get("scan_rows") or 0),
        "scan_row_errors": int(scan.get("scan_row_errors") or 0),
        # SKIP-REASON HISTOGRAM (slice 5): why considered keys produced no row —
        # names the ae failure class (not_local / no_config / comfy / …) in one
        # heartbeat instead of leaving considered≫rows unexplained.
        "scan_skip_reasons": scan.get("scan_skip_reasons") or {},
        # REGISTRY SOURCES (slice 6): per-origin count of the live registry —
        # {staple, discovered, central, comfy, total}. A dead source is visible
        # in one beat: the ae 2026-07-17 incident was discovered==0 (stale/absent
        # report left the registry staples-only). Pairs with scan_skip_reasons —
        # no_config≫0 WITH discovered==0 points straight at the report/re-walk.
        "registry_sources": _registry_sources(),
    }
    _STORAGE_CACHE.update(at=now, value=out)
    return out


def _hot_cache_status() -> dict:
    """Best-effort hot-cache overview for the heartbeat storage view; never
    raises into a heartbeat (returns {"enabled": False} on any failure)."""
    try:
        from ..managers.serve import hot_cache
        return hot_cache.status()
    except Exception:  # noqa: BLE001
        return {"enabled": False}


def _registry_sources() -> dict:
    """Per-source count of the worker's live model registry (slice 6): how many
    configs came from each origin — {"staple": N, "discovered": N, "central": N,
    "comfy": N, "total": N}.

    Same honesty pattern as the scan skip-reason histogram: a DEAD source is
    visible in one heartbeat. The ae 2026-07-17 incident was 'discovered'==0 (a
    stale/absent discovery report left the registry staples-only); this names
    that directly instead of leaving 63 no_config skips unexplained.

    Classification (best-effort, read-only): a row is `comfy` when framework==
    comfy; `staple` when its key is a curated MODELS entry; `discovered` when it
    carries a `dir` (the ABSOLUTE on-disk path discover_models stamps — staples
    carry only a layout `folder`, never a `dir`); else `central` (adopted from
    central's config row via ensure_model_registered). Never raises."""
    out = {"staple": 0, "discovered": 0, "central": 0, "comfy": 0, "total": 0}
    try:
        from .imports import models_config as mc
        staples = set(getattr(mc, "MODELS", {}).keys())
        reg = getattr(mc, "MODEL_REGISTRY_DICT", None) or {}
        for key, row in reg.items():
            out["total"] += 1
            r = row if isinstance(row, dict) else {}
            if str(r.get("framework") or "") == "comfy":
                out["comfy"] += 1
            elif key in staples:
                out["staple"] += 1
            elif r.get("dir"):
                out["discovered"] += 1
            else:
                out["central"] += 1
    except Exception:  # noqa: BLE001 — a heartbeat must never fail on this
        pass
    return out


def _spill_describe() -> dict:
    try:
        #from abstract_hugpy_dev.managers.spill import describe

        return describe()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def build_app(state: "WorkerState") -> Flask:
    app = Flask("abstract_hugpy_worker")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "ok": True,
                "worker_id": state.worker_id,
                "name": state.name,
                "gpus": detect_gpus(),
                "cuda": torch_cuda_status(),
                "llama_cpp": llama_cpp_cuda_status(),
                "assigned_models": state.assigned_models,
                "provisioning": sorted(state._provisioning),
                "provision_progress": state.provision_snapshot(),
                "loaded_models": loaded_model_keys(),
                "spill": _spill_describe(),
            }
        )

    @app.route("/infer", methods=["POST"])
    def infer():
        payload = request.get_json(silent=True) or {}
        # Errors as DATA, never a raw Flask 500: the raw error page hides the
        # worker-side traceback from central entirely (2026-07-03: three
        # opaque delegation failures in one day were undiagnosable from
        # central). A 500 with a JSON body rides back through the delegating
        # runner's error path, so the console shows the REAL cause.
        try:
            _apply_spill(payload.pop("spill", None))
            _ensure_present(payload, state.central_url, state=state)
            # Per-model generation gate: serialize entry into an in-process
            # (llama.cpp/transformers) runner so concurrent /infer calls can't
            # race the same non-reentrant native context and crash the worker.
            # No-op for a slot-backed model (its child schedules itself). On a
            # bounded-wait timeout this raises ModelBusy -> honest 503 below.
            with gen_gate.gate_for_payload(payload):
                return jsonify(_run_once(payload))
        except gen_gate.ModelBusy as busy:
            # Honest structured busy — the runner is at capacity, not broken.
            return jsonify(busy.as_error(
                {"id": state.worker_id, "name": state.name})), 503
        except BudgetRefusal as exc:
            # The model cannot fit on this box even after a full FIFO. NOT a
            # crash and NOT a traceback: a storage-capacity verdict, so it gets
            # its own honest code (507 Insufficient Storage) and the structured
            # reason. Central can then route elsewhere instead of retrying a
            # box that will never have room.
            return jsonify({
                "ok": False,
                "error": exc.reason.get("reason"),
                "refused": exc.reason,
                "worker": {"id": state.worker_id, "name": state.name},
            }), 507
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            logger.error("infer failed: %s", tb)
            return jsonify({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": tb[-1500:],
                # Attribution at the source: direct API consumers (and any
                # relay that keeps the body) see WHICH box failed without
                # having to know who they called.
                "worker": {"id": state.worker_id, "name": state.name},
            }), 500

    @app.route("/infer/stream", methods=["POST"])
    def infer_stream():
        payload = request.get_json(silent=True) or {}
        _apply_spill(payload.pop("spill", None))
        # Caller-supplied id for cancellation; else generate one. Echo it back
        # as the first SSE event so the client can cancel this exact request.
        req_id = str(payload.pop("request_id", "") or uuid.uuid4().hex)

        # Per-model generation gate, acquired BEFORE the streaming Response so a
        # busy in-process runner is refused with a real HTTP 503 (not a mid-body
        # SSE surprise). The bounded wait blocks here (that IS the honest queue).
        # The token is then held for the WHOLE life of the stream and released in
        # the generator's finally — a streamed response occupies the runner until
        # its last token. No-op token for a slot-backed model. See gen_gate.
        try:
            gate_token = gen_gate.acquire_for_payload(payload)
        except gen_gate.ModelBusy as busy:
            return jsonify(busy.as_error(
                {"id": state.worker_id, "name": state.name})), 503

        def _generate():
            try:
                yield _sse({"type": "request", "request_id": req_id})
                # Stream provisioning progress first (download from central/HF),
                # then generation with auto-continuation. Both emit SSE lines.
                yield from _ensure_present_streaming(payload, state.central_url,
                                                     state=state)
                yield from _stream_sync(payload, request_id=req_id)
            finally:
                # Release on normal end, error, OR client disconnect (Flask closes
                # the generator) — the gate must never leak a permit.
                gate_token.release()

        return Response(
            stream_with_context(_generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
            direct_passthrough=True,
        )

    @app.route("/infer/cancel/<request_id>", methods=["POST"])
    def infer_cancel(request_id):
        # Same substrate as central (F5): the job's attached cancel handle sets
        # the stream's Event on the shared runtime loop via call_soon_threadsafe
        # (a bare cross-thread Event.set() is unsafe — wakes futures on another
        # loop). Wire contract unchanged: 404 for unknown/finished requests.
        from ..comms import job_store
        job = job_store.get(request_id)
        if job is None or job.terminal:
            return jsonify({"cancelled": False, "reason": "unknown or finished request"}), 404
        job_store.cancel(request_id, reason="cancelled via /infer/cancel")
        return jsonify({"cancelled": True, "request_id": request_id})

    # -- privileged ops (F3.4 control agent; CON-05/06 + UTIL-02) ----------
    # Central relays these through its operator gate + audit log. Every
    # response is typed data ({ok, error:{code,message}}), never a traceback.

    @app.route("/ops/restart", methods=["POST"])
    def ops_restart():
        # Respond first, then restart: the caller needs the ack before the
        # process cycles. Under systemd this EXITS (Restart= respawns a fresh,
        # cgroup-tracked process — never the os.execv orphan that squatted :9100
        # and restart-looped); standalone re-execs in place. Persistent worker-id
        # -> same registry row.
        _schedule_restart(state, "ops/restart")
        return jsonify({"ok": True, "restarting": True,
                        "worker_id": state.worker_id})

    @app.route("/ops/free-ram", methods=["POST"])
    def ops_free_ram():
        # NON-destructive host-RAM reclaim: return glibc's orphaned allocator
        # arena (and torch's CUDA cache) to the OS WITHOUT evicting any model.
        # After a model is freed malloc keeps the pages pooled so RSS stays
        # pinned (ae observed at 0 free / 128 GB used, nothing loaded);
        # malloc_trim(0) hands them back. loaded_models is reported UNCHANGED —
        # Unload is the destructive path; this one never touches residency.
        ram_before = _free_ram_bytes()
        rss_before = _agent_rss_bytes()
        _trim_host_ram()
        ram_after = _free_ram_bytes()
        rss_after = _agent_rss_bytes()
        ram_freed = (ram_after - ram_before) if (
            ram_before is not None and ram_after is not None) else None
        return jsonify({
            "ok": True,
            "ram_free_before": ram_before,
            "ram_free_after": ram_after,
            "ram_freed": ram_freed,
            "rss_before": rss_before,
            "rss_after": rss_after,
            "loaded_models": loaded_model_keys(),
        })

    @app.route("/ops/update", methods=["POST"])
    def ops_update():
        # CON-05 on demand: same converge path as the heartbeat handshake
        # (pip install pinned target from PyPI or --pkg-index, then re-exec),
        # minus the wait and the retry backoff — the operator asked NOW.
        args = getattr(state, "args", None)
        if args is None:
            return jsonify({"ok": False, "error": {
                "code": "NoArgs", "message": "agent started without CLI args "
                "context; update unavailable"}}), 501
        body = request.get_json(silent=True) or {}
        target = str(body.get("version") or "").strip()
        if not target:
            return jsonify({"ok": False, "error": {
                "code": "NoVersion",
                "message": 'body must include {"version": "x.y.z"} '
                           '(central sends its required_pkg_version)'}}), 400
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-deps"]
        if args.pkg_index:
            cmd += ["--index-url", args.pkg_index]
        cmd.append(f"{args.pkg_name}=={target}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=560)
            rc, tail = proc.returncode, (proc.stdout + proc.stderr)[-2000:]
        except Exception as exc:
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__, "message": str(exc)}}), 502
        if rc == 0:
            # kill_slots: a fresh version is installed — any orphaned slot child
            # would keep serving the OLD code (the adoption probe can't tell
            # versions apart), so tear them down and let the fresh agent respawn
            # them on the new code. Same discipline as the heartbeat self-update.
            _schedule_restart(state, "ops/update", kill_slots=True)
            return jsonify({"ok": True, "installed": f"{args.pkg_name}=={target}",
                            "restarting": True})
        return jsonify({"ok": False, "error": {
            "code": "PipFailed", "message": f"pip rc={rc}", "detail": tail}}), 502

    @app.route("/ops/pip", methods=["POST"])
    def ops_pip():
        # UTIL-02: install into this worker's env. Argv-list (no shell), rc +
        # output tail returned as data. The operator gate + audit live on
        # central; this endpoint trusts central's relay like every other op.
        body = request.get_json(silent=True) or {}
        pkg = str(body.get("package") or "").strip()
        if not pkg or pkg.startswith("-"):
            return jsonify({"ok": False, "error": {
                "code": "BadPackage",
                "message": 'body must include {"package": "name==ver"} '
                           "(flags are not accepted)"}}), 400
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True, text=True, timeout=560)
            rc, tail = proc.returncode, (proc.stdout + proc.stderr)[-2000:]
        except Exception as exc:
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__, "message": str(exc)}}), 502
        # A llama-cpp-python (re)install changes the engine's GPU-offload
        # capability, but _LLAMA_PROBE_CACHE memoizes the FIRST probe for the
        # whole process life — so /health + heartbeat caps would keep reporting
        # the OLD build's supports_gpu_offload until a full re-exec (/ops/pip
        # never re-execs; /ops/update does). Invalidate the cache so the next
        # probe reflects the freshly-installed build honestly.
        if rc == 0 and "llama" in pkg.lower():
            global _LLAMA_PROBE_CACHE
            _LLAMA_PROBE_CACHE = None
        return (jsonify({"ok": rc == 0, "package": pkg, "rc": rc,
                         "output_tail": tail}), 200 if rc == 0 else 502)

    @app.route("/ops/config", methods=["POST", "GET"])
    def ops_config():
        # Daylight item 3: operator serving-config, persisted in the agent's
        # OWN settings file (beats env/drop-ins — see _apply_settings_env).
        # GET returns current settings + effective values; POST merges the
        # supported keys, persists, and re-execs to apply cleanly (persistent
        # worker-id -> same registry row; ~seconds of blip).
        args = getattr(state, "args", None)
        if args is None:
            return jsonify({"ok": False, "error": {
                "code": "NoArgs", "message": "agent started without CLI args"}}), 501
        if request.method == "GET":
            return jsonify({"ok": True, "settings": _load_settings(args),
                            "effective": _effective_config()})
        body = request.get_json(silent=True) or {}
        unknown = sorted(set(body) - _SETTINGS_KEYS)
        if unknown:
            return jsonify({"ok": False, "error": {
                "code": "UnknownKeys",
                "message": f"unsupported: {unknown}; supported: {sorted(_SETTINGS_KEYS)}"}}), 400
        settings = _load_settings(args)
        if "slot_count" in body:
            try:
                n = int(body["slot_count"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "slot_count must be an integer"}}), 400
            if not 0 <= n <= 16:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "slot_count must be 0..16"}}), 400
            settings["slot_count"] = n
        if "on_demand_ttl_s" in body:
            # OPT-IN idle reclamation (doctrine 2026-07-11). The DEFAULT residency
            # trigger is memory contention, not a clock — so on_demand_ttl_s is
            # ABSENT by default and null/0 CLEARS it (idle sweep off; contention
            # alone governs residency; the heartbeat then reports it as null).
            val = body["on_demand_ttl_s"]
            if val in (None, "", 0, "0"):
                settings.pop("on_demand_ttl_s", None)
            else:
                try:
                    tval = int(val)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": "on_demand_ttl_s must be an integer, or null to disable idle reclamation"}}), 400
                if not 60 <= tval <= 86400:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": "on_demand_ttl_s must be 60..86400 (or null to disable idle reclamation)"}}), 400
                settings["on_demand_ttl_s"] = tval
        if "reconcile_interval_s" in body:
            try:
                tval = int(body["reconcile_interval_s"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "reconcile_interval_s must be an integer"}}), 400
            if not 60 <= tval <= 86400:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": "reconcile_interval_s must be 60..86400"}}), 400
            settings["reconcile_interval_s"] = tval
        if "residency" in body:
            # DEEP-MERGE per model key: {"model": "static"|null}. The default
            # tier is ON-DEMAND and is represented by NO stored entry — null,
            # "", "on-demand" itself, or the legacy synonyms "serving"/"warm"
            # all clear the override. "static" is the only stored value.
            if not isinstance(body["residency"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'residency must be {"<model_key>": "static"|null} — null/"on-demand" (or legacy "serving"/"warm") restores the on-demand default'}}), 400
            merged = dict(settings.get("residency") or {})
            for mk, mode in body["residency"].items():
                if mode in (None, "", "on-demand", "serving", "warm"):
                    # on-demand IS the default — storing it would be noise;
                    # any of these writes clears the override.
                    merged.pop(mk, None)
                elif mode == "static":
                    merged[mk] = mode
                else:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"residency[{mk!r}] must be 'static' or null — on-demand is the default ('on-demand'/'serving'/'warm' also clear the override)"}}), 400
            if merged:
                settings["residency"] = merged
            else:
                settings.pop("residency", None)
        if "pinned" in body:
            # Files-axis pin (tiers v2): {"model": true|null}. Deep-merged.
            if not isinstance(body["pinned"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'pinned must be {"<model_key>": true|null}'}}), 400
            pmerged = dict(settings.get("pinned") or {})
            for mk, val in body["pinned"].items():
                if val in (None, False, ""):
                    pmerged.pop(mk, None)
                elif val is True:
                    pmerged[mk] = True
                else:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"pinned[{mk!r}] must be true or null"}}), 400
            if pmerged:
                settings["pinned"] = pmerged
            else:
                settings.pop("pinned", None)
        if "ctx_pct" in body:
            # Per-model CONTEXT allocation (slice 11 / t27): {"model": 1..100 | null}
            # — percent of the model's max context reserved for KV in fit/serving.
            # Deep-merged, same shape as residency/pinned. null clears (default ctx).
            if not isinstance(body["ctx_pct"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'ctx_pct must be {"<model_key>": 1..100 | null}'}}), 400
            cmerged = dict(settings.get("ctx_pct") or {})
            for mk, val in body["ctx_pct"].items():
                # ONLY an explicit null/"" clears (default ctx). Do NOT treat 0/
                # False as a clear: `0 in (None, "", False)` is True in Python
                # (0 == False), which would silently clear on an out-of-range 0
                # instead of rejecting it — so match None/"" explicitly.
                if val is None or val == "":
                    cmerged.pop(mk, None)
                    continue
                try:
                    pv = int(val)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"ctx_pct[{mk!r}] must be an integer 1..100 or null"}}), 400
                if not (1 <= pv <= 100):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"ctx_pct[{mk!r}]={pv} out of range — must be 1..100"}}), 400
                cmerged[mk] = pv
            if cmerged:
                settings["ctx_pct"] = cmerged
            else:
                settings.pop("ctx_pct", None)
        # t21 tolerance bands + priority (per-model maps, siblings of ctx_pct).
        # A deviation is a percent-of-total tolerance (0..100; 0 == no band);
        # priority is a non-negative integer (0 == normal). Same deep-merge /
        # null-clears shape as ctx_pct. Additive + optional so a released worker
        # that doesn't know these keys simply ignores them (no schema forbid on
        # /ops/config) — the relay-wire landmine does not apply here.
        for _band_key, _lo, _hi, _isint in (
                ("ctx_deviation_pct", 0, 100, False),
                ("vram_deviation_pct", 0, 100, False),
                ("ram_deviation_pct", 0, 100, False),
                ("priority", 0, None, True)):
            if _band_key not in body:
                continue
            if not isinstance(body[_band_key], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": f'{_band_key} must be {{"<model_key>": number | null}}'}}), 400
            _merged = dict(settings.get(_band_key) or {})
            for mk, val in body[_band_key].items():
                # Only explicit null/"" clears (see ctx_pct: 0 == False in Python
                # so match None/"" exactly rather than truthiness).
                if val is None or val == "":
                    _merged.pop(mk, None)
                    continue
                try:
                    pv = int(val) if _isint else float(val)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"{_band_key}[{mk!r}] must be a number or null"}}), 400
                if pv < _lo or (_hi is not None and pv > _hi):
                    hi_txt = _hi if _hi is not None else "∞"
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"{_band_key}[{mk!r}]={pv} out of range — must be {_lo}..{hi_txt}"}}), 400
                _merged[mk] = pv
            if _merged:
                settings[_band_key] = _merged
            else:
                settings.pop(_band_key, None)
        if "comfy_url" in body:
            # Adopted-ComfyUI base URL (probe + job submission read it as the
            # COMFY_URL env). Settable here so central/console can point a worker
            # at its ComfyUI without a systemd drop-in. null/"" clears it (falls
            # back to env / the 127.0.0.1:8188 default).
            cu = body["comfy_url"]
            if cu in (None, ""):
                settings.pop("comfy_url", None)
            elif isinstance(cu, str) and _valid_comfy_url(cu):
                settings["comfy_url"] = cu.strip().rstrip("/")
            else:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": "comfy_url must be an http(s) URL with a host "
                               "(e.g. https://comfy.example.ai), or null"}}), 400
        if "hot_cache_root" in body:
            # Per-worker attribution of the HOT-CACHE tier's root (the box-local
            # NVMe LRU cache of the main catalog — managers/serve/hot_cache.py,
            # which reads HUGPY_HOT_CACHE_ROOT live). This ONLY names WHERE the
            # tier lives on this box (e.g. ae -> /mnt/hot990/hugpy-hot-cache); the
            # tier stays an automatic LRU cache and the SHARED store stays the
            # source of truth. null/"" CLEARS it (revert to the env base, else the
            # tier is off) — same idiom as on_demand_ttl_s. A set value is checked
            # SHAPE-ONLY (must be an absolute path), mirroring comfy_url's URL
            # check: a not-yet-mounted root is accepted here and hot_cache.enabled()
            # disables the tier gracefully until it exists, so config never blocks
            # on a mount that is about to appear.
            hcr = body["hot_cache_root"]
            if hcr in (None, ""):
                settings.pop("hot_cache_root", None)
            elif isinstance(hcr, str) and os.path.isabs(hcr.strip()):
                settings["hot_cache_root"] = hcr.strip().rstrip("/") or "/"
            else:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": "hot_cache_root must be an absolute path "
                               "(e.g. /mnt/hot990/hugpy-hot-cache), or null to clear"}}), 400
        if "profiles" in body:
            # Env-profiles (stage 1): {"<name>": {"packages": [str,...]} | null}.
            # DEEP-MERGE per profile; null/{}/"" clears one. Names slug-safe;
            # packages a NON-EMPTY list of non-empty strings. A profile = a named
            # venv (materialized in the background at boot — see main()) that a
            # profiled model's SLOT CHILD launches from, isolating extra deps from
            # the shared venv. The agent itself never installs into it.
            from ..managers.serve import profiles as _profiles_mod
            if not isinstance(body["profiles"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'profiles must be {"<name>": {"packages": [str,...]}} '
                               "(or null per name to clear)"}}), 400
            pmerged = dict(settings.get("profiles") or {})
            for name, spec in body["profiles"].items():
                if not _profiles_mod.slug_ok(name):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"profile name {name!r} must be slug-safe "
                                   "(letters/digits then . _ - , max 64 chars)"}}), 400
                if spec in (None, {}, ""):
                    pmerged.pop(name, None)
                    continue
                if not isinstance(spec, dict):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"profiles[{name!r}] must be an object with a "
                                   "packages list, or null to clear"}}), 400
                pkgs = spec.get("packages")
                if (not isinstance(pkgs, list) or not pkgs
                        or not all(isinstance(p, str) and p.strip() for p in pkgs)):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"profiles[{name!r}].packages must be a non-empty "
                                   "list of non-empty strings"}}), 400
                pmerged[name] = {"packages": [p.strip() for p in pkgs]}
            if pmerged:
                settings["profiles"] = pmerged
            else:
                settings.pop("profiles", None)
        if "model_profiles" in body:
            # Model->profile ATTRIBUTION (stage 1): {"<model_key>": "<name>" |
            # null}. DEEP-MERGE; null/"" clears an attribution. The name is
            # slug-safe but need NOT already exist (the two keys can arrive in
            # either order across relay calls; a dangling attribution reports
            # honestly and refuses to seat until the profile is declared+ready).
            from ..managers.serve import profiles as _profiles_mod
            if not isinstance(body["model_profiles"], dict):
                return jsonify({"ok": False, "error": {
                    "code": "BadValue",
                    "message": 'model_profiles must be {"<model_key>": "<profile_name>"|null}'}}), 400
            mmerged = dict(settings.get("model_profiles") or {})
            for mk, pname in body["model_profiles"].items():
                if pname in (None, ""):
                    mmerged.pop(mk, None)
                elif _profiles_mod.slug_ok(pname):
                    mmerged[mk] = pname
                else:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue",
                        "message": f"model_profiles[{mk!r}] must be a slug-safe "
                                   "profile name, or null to clear"}}), 400
            if mmerged:
                settings["model_profiles"] = mmerged
            else:
                settings.pop("model_profiles", None)
        _save_settings(args, settings)
        logger.info("ops/config: persisted %s — restarting to apply", settings)
        # Restart to re-project the settings over a clean base. Under systemd this
        # EXITS and the fresh process gets the UNIT's env as the true base (so the
        # env base-sentinels are unnecessary in that path); standalone re-execs
        # and the sentinels carry the pre-projection base across the exec. Both
        # lifecycles are documented at _apply_settings_env.
        _schedule_restart(state, "ops/config apply")
        return jsonify({"ok": True, "settings": settings, "restarting": True})

    @app.route("/probe/<path:model_key>", methods=["POST", "GET"])
    def probe(model_key):
        # Live VRAM-fit check: actually load the model on this worker's GPU and
        # report whether it fit, plus before/after free VRAM. Loading is cached
        # by dispatch, so a probe also warms the model for the first real chat.
        return jsonify(_probe_model(model_key, state))

    @app.route("/models/unload", methods=["POST"])
    def unload():
        # Free GPU VRAM by evicting cached runner(s) from this worker's dispatch
        # cache. Body: {"model_key": ...} drops one model; {} or {"all": true}
        # drops everything loaded. The model stays ASSIGNED (central's registry
        # is untouched) — it just isn't held in VRAM until the next request.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        before = _free_vram_bytes()
        ram_before = _free_ram_bytes()
        err = None
        try:
            from ..managers.dispatch import evict as _evict, clear as _clear
            if model_key and not body.get("all"):
                evicted = bool(_evict(model_key))
            else:
                _clear()
                evicted = True
        except Exception as exc:
            evicted, err = False, f"{type(exc).__name__}: {exc}"
        # The evict/clear above drops references, but glibc keeps the freed
        # weights' host arena pooled (RSS stays pinned) — trim hands it, and
        # torch's CUDA cache, back to the OS so this destructive path returns
        # host RAM too, not just VRAM.
        _trim_host_ram()
        after = _free_vram_bytes()
        ram_after = _free_ram_bytes()
        freed = (after - before) if (before is not None and after is not None) else None
        ram_freed = (ram_after - ram_before) if (
            ram_before is not None and ram_after is not None) else None
        return jsonify({
            "ok": err is None,
            "evicted": evicted,
            "model_key": model_key,
            "error": err,
            "vram_free_before": before,
            "vram_free_after": after,
            "freed": freed,
            "ram_free_before": ram_before,
            "ram_free_after": ram_after,
            "ram_freed": ram_freed,
            "loaded_models": loaded_model_keys(),
        })

    @app.route("/ops/evict", methods=["POST"])
    def ops_evict():
        # Targeted eviction: free ONE model's RAM+VRAM, picking the mechanism by
        # how that model is hosted (comfy /free, slot child kill, or in-process
        # ref-drop). Central sends {"model_key": ..., "force"?: bool} — NEVER a
        # PID (per-box, recycled): the worker resolves the model_key to its live
        # handle here and verifies identity before acting. Fail-safe: an unknown
        # or not-resident model_key is an idempotent no-op at HTTP 200, never a
        # 500. force=true overrides the static/pinned/in-flight gate. Contrast
        # /models/unload (coarse: one key or ALL, in-process/slot-proxy cache
        # only) — this is the surgical, host-mode-aware verb.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        force = bool(body.get("force"))
        try:
            return jsonify({"ok": True, **_evict_model(state, model_key, force)})
        except Exception as exc:  # noqa: BLE001 — evict must never 500 the control plane
            return jsonify({"ok": False, "model_key": model_key,
                            "host_mode": "unknown", "evicted": False,
                            "vram_freed": None, "ram_freed": None,
                            "reason": f"{type(exc).__name__}: {exc}"})

    @app.route("/ops/reap-orphans", methods=["POST"])
    def ops_reap_orphans():
        # p27: kill ORPHANED GPU children this worker itself leaked (own-venv
        # llama-server whose slot claim cleared but whose process kept VRAM —
        # enumerable since c34199e as cuda_context/model_key-None rows, but
        # unevictable because every eviction verb keys on model_key). Central
        # relays this through its operator gate + audit log like every /ops/*;
        # the worker trusts central's relay (same idiom as /ops/pip).
        # Body: {"dry_run"?: bool} — DEFAULTS TO TRUE when absent (preview).
        # OPERATOR/CENTRAL-INVOKED ONLY — never wired to a loop/heartbeat
        # (doctrine, see _reap_gpu_orphans). Never 500s the control plane.
        body = request.get_json(silent=True) or {}
        dry_run = True if "dry_run" not in body else bool(body.get("dry_run"))
        try:
            return jsonify({"ok": True,
                            **_reap_gpu_orphans(state, dry_run=dry_run)})
        except Exception as exc:  # noqa: BLE001 — reap must never 500 the control plane
            return jsonify({"ok": False, "dry_run": dry_run, "results": [],
                            "reaped_count": 0, "term_failed_count": 0,
                            "skipped_count": 0, "reapable_vram_bytes": 0,
                            "error": f"{type(exc).__name__}: {exc}"})

    @app.route("/slots/<slot_id>/relaunch", methods=["POST"])
    def slot_relaunch(slot_id):
        # k14: relaunch ONE of this worker's slot children with a new offload depth
        # (n_gpu_layers) / context, so the k7 offload speed-cliff sweep can seat a
        # GGUF at full offload then sweep it DOWN through layer counts, measuring
        # tok/s at each step. Central relays here with {"n_gpu_layers"?, "ctx"?}.
        # The worker resolves slot_id -> its live control URL, confirms a model is
        # seated, and asks the slot supervisor to STOP->RESPAWN its child (the slot
        # owns the SIGTERM->SIGKILL). This also answers the ae "slot-child PID never
        # recycles" blocker: every relaunch respawns the child under a NEW pid.
        # 404 = no such slot on this worker; 409 = slot empty (nothing to relaunch).
        import httpx
        body = request.get_json(silent=True) or {}
        payload = {k: body[k] for k in ("n_gpu_layers", "ctx")
                   if body.get(k) not in (None, "")}
        try:
            from ..managers.serve.slots import SlotPool
            statuses = SlotPool().statuses()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": {
                "code": type(exc).__name__,
                "message": f"slot pool unavailable: {exc}"}}), 502
        target = None
        for s in (statuses or []):
            if str(s.get("slot_id")) == str(slot_id):
                target = s
                break
        if target is None:
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": "UnknownSlot",
                "message": f"no slot {slot_id} on this worker"}}), 404
        if not target.get("model_key"):
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": "EmptySlot",
                "message": f"slot {slot_id} has no model loaded to relaunch"}}), 409
        control = target.get("_control")
        try:
            # A relaunch respawns a (possibly big) child — allow a cold-load-long
            # window, same order as a /load warm.
            r = httpx.post(control + "/relaunch", json=payload, timeout=900.0)
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "slot_id": slot_id, "error": {
                "code": type(exc).__name__,
                "message": f"relaunch relay to slot failed: {exc}"}}), 502
        if isinstance(data, dict) and data.get("error"):
            return jsonify({"ok": False, "slot_id": slot_id,
                            "error": data.get("error")}), r.status_code
        # Echo the HONEST launched allocation: n_gpu_layers is what the fresh child
        # actually launched with (slot status), not merely what was requested.
        return jsonify({
            "ok": True, "slot_id": slot_id,
            "model_key": (data or {}).get("model_key"),
            "n_gpu_layers": (data or {}).get("n_gpu_layers"),
            "requested_n_gpu_layers": (data or {}).get("requested_n_gpu_layers"),
            "ctx": (data or {}).get("ctx"),
            "child_pid": (data or {}).get("child_pid"),
            "healthy": (data or {}).get("healthy"),
            "allocation": data,
        }), r.status_code

    @app.route("/models/redownload", methods=["POST"])
    def redownload():
        # Force a CLEAN re-pull from central: evict from VRAM, DELETE the model's
        # local files, then re-provision (download) it. Body: {"model_key": ...}.
        # A plain /load only downloads when files are MISSING, so it can't refresh
        # a corrupt/stale on-disk copy — this can.
        body = request.get_json(silent=True) or {}
        model_key = body.get("model_key")
        if not model_key:
            return jsonify({"ok": False, "error": "missing model_key"}), 400
        try:
            from ..managers.dispatch import evict as _evict
            from .provision import (
                wipe_model, ensure_model_present, ensure_model_registered,
            )
            try:
                _evict(model_key)   # drop from VRAM so its files aren't held open
            except Exception:
                pass
            ensure_model_registered(model_key, state.central_url)
            wiped = wipe_model(model_key)
            # Gate the re-pull (Part A, slice 8): a /redownload wipes then re-pulls
            # — it MUST run the storage gate too, or it re-fills an over-budget
            # store the operator just tried to relieve. Pass state explicitly.
            ok = ensure_model_present(model_key, state.central_url, state=state)
            return jsonify({"ok": bool(ok), "wiped": bool(wiped),
                            "redownloaded": bool(ok), "model_key": model_key,
                            "loaded_models": loaded_model_keys()})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/reap", methods=["POST"])
    def reap():
        """Disk reclaim (tiers-v2 slice 4). The bookend to unassign: delete the
        local files of models that are on disk but no longer needed.

        Body:
          {"dry_run": true}          -> PREVIEW only (default): what would be
                                        freed + what's protected and why.
          {"all": true}             -> reclaim every reclaimable model.
          {"model_keys": ["a","b"]} -> reclaim just these (still guard-checked).

        Guards (re-proven at delete time): never assigned, loaded/loading,
        pinned, or comfy (operator symlinks). Deletes are jailed by wipe_model
        against root/home/short paths.
        """
        body = request.get_json(silent=True) or {}
        scan = _reap_scan(state)
        if body.get("dry_run") or (not body.get("all") and not body.get("model_keys")):
            return jsonify({"ok": True, "dry_run": True, **scan})
        if body.get("all"):
            targets = [r["model_key"] for r in scan.get("reclaimable", [])]
        else:
            targets = [str(k) for k in (body.get("model_keys") or [])]
        if not targets:
            return jsonify({"ok": True, "results": [], "freed_bytes": 0,
                            "note": "nothing reclaimable"})
        return jsonify(_reap_reclaim(state, targets))

    # Studio render offload (option a): mount POST /studio/render, GET
    # /studio/render/<job_id>, POST /studio/cancel/<job_id> so central can delegate
    # a REAL-model studio render (produce_clip) to THIS worker's GPU while keeping
    # the control plane. Imported LAZILY (studio_render's own studio-spine imports
    # are lazy inside its render thread, so this never pulls torch/diffusers at
    # boot) and guarded so a mount hiccup can never break the rest of the agent's
    # routes.
    try:
        from .studio_render import register_studio_routes
        register_studio_routes(app, worker_id=state.worker_id, worker_name=state.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("studio render endpoints not mounted: %s", exc)

    return app


def _free_vram_bytes() -> int | None:
    try:
        # Relative import (rename-proof for the prod mirror). This line was
        # once commented out, which left a bare NameError swallowed below —
        # every probe/unload reported vram_free null and fit=false even when
        # the weights landed on the GPU.
        from ..managers.spill import free_vram_bytes
        return free_vram_bytes()
    except Exception:
        return None


def _probe_model(model_key: str, state: "WorkerState") -> dict:
    """Load the model on the GPU and report fit + VRAM deltas.

    Returns {ok, fit, vram_free_before, vram_free_after, vram_used, error}.
    'fit' is a heuristic: ok load AND GPU memory actually decreased (i.e. weights
    landed on the GPU, not spilled entirely to CPU).
    """
    before = _free_vram_bytes()
    result: dict = {"model_key": model_key, "vram_free_before": before}
    try:
        # PROBE DOES NOT DOWNLOAD (operator ruling, ae 1.2TB incident 2026-07-17:
        # "its central that distributed these downloads... it simply needs to
        # abide by the limits set within its own backend"). A probe used to call
        # ensure_model_present() — so probing an ABSENT model WAS a transfer
        # order, and central's warm sweep rode /probe to pull ~700GB onto ae.
        # A probe answers a question ("does this model FIT on my GPU?"); it never
        # provisions. If the files aren't already on THIS box's disk, return an
        # honest non-downloading verdict and let the model arrive on the first
        # REAL call (lazy-download doctrine, 7f0e6e8/2a3baeb).
        #
        # ensure_model_registered is METADATA-only (a small config row from
        # central, no weight bytes) — kept so the locality check + the honest
        # error can name/resolve the model even if this worker wasn't built with
        # it. NO byte transfer happens on ANY path through here.
        from .provision import (
            ensure_model_present, ensure_model_registered, model_is_local,
        )
        canonical = ensure_model_registered(model_key, state.central_url) or model_key

        # Locality gate: the SAME predicate the agent uses everywhere else
        # (model_is_local / _models_local). If the weights aren't already on
        # disk, do not build the runner (which would trigger a load/pull) — the
        # probe reports "not local" and stops here. This is what makes the warm
        # sweep's probe a no-op instead of a distributed 700GB pull order.
        try:
            _local = model_is_local(canonical)
        except Exception:  # noqa: BLE001 — a bad row reads as "not local", never a crash
            _local = False
        if not _local:
            result.update(
                ok=False, fit=False,
                vram_free_after=before, vram_used=0, path="none", local=False,
                error=("not local — probe does not download (lazy doctrine "
                       "2026-07-17); files arrive on first real call"))
            return result
        result["local"] = True

        # Local: safe to build the runner and measure a real fit — no transfer
        # can be triggered because the files are already present.
        #from abstract_hugpy_dev.managers.dispatch import runner_for
        runner = runner_for(model_key=canonical)  # builds the runner WRAPPER
        # runner_for only BUILDS the (lazy) wrapper; for GGUF/in-process runners
        # the weight load + slot seat happen on first .runner access, so a bare
        # build loads NOTHING — the probe would read vram_used=0 / fit=False and
        # seat no slot (exactly the hollow shell that made this model unroutable).
        # Force the underlying runner resident so the probe reflects reality.
        _ensure = getattr(runner, "ensure_loaded", None)
        if callable(_ensure):
            _ensure()

        after = _free_vram_bytes()
        used = (before - after) if (before is not None and after is not None) else None
        # Which path actually took the load: a base_url means an HTTP child
        # (slot or native llama-server); none means in-process llama-cpp-python.
        base_url = (getattr(runner, "base_url", None)
                    or getattr(getattr(runner, "runner", None), "base_url", None))
        result.update(
            ok=True,
            vram_free_after=after,
            vram_used=used,
            path="http" if base_url else "in-process",
            # If GPU free memory dropped meaningfully, weights are on the GPU.
            fit=bool(used and used > 64 * 1024 * 1024),
        )
        # Vision honesty: a vision GGUF served IN-PROCESS cannot decode images
        # (the python binding fails to load the mmproj projector — the reason
        # the native --mmproj server path exists). The load "succeeds" but
        # every image turn silently degrades to text-only, so report the probe
        # as FAILED with the actionable reason instead of ok:true.
        if not base_url:
            try:
                from ..imports.src.utils import find_mmproj
                from .imports import get_model_config, get_model_path
                cfg = get_model_config(canonical)
                tasks = list(getattr(cfg, "tasks", None) or [])
                mpath = None
                try:
                    mpath = get_model_path(canonical)
                except Exception:
                    mpath = getattr(cfg, "dir", None)
                is_vision = ("image-text-to-text" in tasks
                             or bool(mpath and find_mmproj(str(mpath))))
                if is_vision:
                    result.update(
                        ok=False, fit=False,
                        error=("vision model loaded in-process (text-only — the "
                               "python binding cannot load the mmproj projector), "
                               "so images would be silently ignored. Provide a "
                               "native llama-server (LLAMA_SERVER_BIN or `hugpy "
                               "install-engine`) or a healthy slot child so the "
                               "projector loads."))
            except Exception:
                pass  # capability check is advisory — never turn it into a probe crash
    except Exception as exc:
        result.update(ok=False, fit=False, error=f"{type(exc).__name__}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------
class WorkerState:
    def __init__(self, name: str, url: str | None, worker_id: str | None,
                 central_url: str | None = None, port: int | None = None):
        self.name = name
        self.url = url            # None unless operator set --advertise/WORKER_URL
        self.worker_id = worker_id
        self.central_url = central_url
        self.port = port
        # Fleet role: "worker" (whole-model, serves /infer) or "rpc" (lends its
        # GPU to a shard pool via llama.cpp rpc-server). rpc_endpoint is the
        # "host:port" central hands to a lead as an rpc_servers entry.
        self.role = "worker"
        self.rpc_endpoint: str | None = None
        # Models central says we should serve, plus which we've already kicked
        # off a background provision for (so we don't re-trigger every beat).
        self.assigned_models: list[str] = []
        self._provisioning: set[str] = set()
        # Central's per-worker allocations, adopted from the heartbeat reply
        # (_apply_central_limits). The STORAGE budget reads limits
        # ["disk_cache_gib"] from here; unset -> the budget is unmanaged and the
        # auto-evict path stays off (see budget.cap_bytes).
        self.limits: dict = {}
        # Central's LRU clock {model_key: epoch} — when each model was last
        # PICKED to serve on this box. The FIFO key for evict-to-fit; the worker
        # cannot know it (central routes the calls), so central ships it in the
        # heartbeat reply. Missing key -> 0 -> coldest -> evicted first.
        self.model_last_picked: dict = {}
        # Central's ALLOCATION-LEVEL totals for this box's assignment set:
        # {allocated_total_bytes, allocated_count, allocated_unknown_count}.
        # Sizing the set needs the manifest (central-only), so central computes
        # it per read and ships it in the heartbeat reply; the refusal reason
        # reads it from here rather than making N HTTP calls under the pull lock.
        # Empty until the first beat -> the refusal simply omits the structural
        # clause (an unknown total is never reported as a comfortable 0).
        self.allocated: dict = {}
        # Models REFUSED for storage: {model_key: {state:"refused", reason:...}}.
        # Reported in the heartbeat so central/console render the model as
        # MISSING with a hover reason instead of a phantom "pulling".
        self.refused: dict = {}
        # key -> {done_bytes, total_bytes, frac}; populated while a background
        # pre-provision downloads, so central (and the console) can show a %.
        self._provision_progress: dict[str, dict] = {}
        self._provision_lock = threading.Lock()
        # Thundering-herd guard (root-caused live 2026-07-15): a big assignment
        # list used to spawn one _kick_provision background thread PER model
        # simultaneously, each running its own segmented/parallel download —
        # N assigned models meant N concurrent multi-threaded pulls hammering
        # central at once (observed: 30 models, near-constant 503s, no
        # convergence). Cap how many DIFFERENT models may provision at the
        # same time; the rest queue and drain serially. Default 1 (fully
        # serial) — the safe, calm default; raise via env if a box's link to
        # central can actually take it. Read once here (like the other
        # WorkerState fields) rather than re-reading the env on every kick.
        _cap = _safe_int(os.environ.get("WORKER_PROVISION_CONCURRENCY"))
        if _cap is None or _cap < 1:
            _cap = 1
        self.provision_concurrency: int = _cap
        self._provision_semaphore = threading.BoundedSemaphore(self.provision_concurrency)
        # The live werkzeug HTTP server (set in main() once bound). The restart
        # path closes its listening socket to release :9100 cleanly before exit;
        # None until the server is created (and in test clients that never bind).
        self.http_server = None

    def provision_snapshot(self) -> dict:
        """A lock-safe copy of per-model download progress for the heartbeat."""
        with self._provision_lock:
            return {k: dict(v) for k, v in self._provision_progress.items()}


def _eager_pull(model_key: str) -> bool:
    """Should ASSIGNMENT alone pull this model's weights to local disk?

    Lazy-download doctrine (operator, 2026-07-16): "models are attributed to be
    routed to a worker though not immediately downloaded to the worker's drive,
    they should be lazy download instead downloading to the drive only when
    called". Assignment is ATTRIBUTION, not a transfer order — the download
    happens on first CALL, via the inference path's already-working
    _ensure_present / _ensure_present_streaming.

    This is the structural fix for the 2026-07-15 provision storm: assigning N
    models fired N parallel provisions, 503'ing central and leaving four
    truncated GGUFs (~10.7GB) on computron — every one of them "designated" in
    worker_assignments.json.

    Exactly ONE tier pre-pulls, because for it lazy would break a promise the
    tier already makes:

      * static (:_residency) — operator-locked 2026-07-05 as "eager-warmed": a
        locked seat that paid full download latency on first call is a broken
        promise (see the defaults-are-promises doctrine). Static is an
        explicit, deliberately-chosen resident seat — the operator opts INTO
        the download by choosing the tier.

    📌 pin is NOT an eager tier (operator, 2026-07-16): "pinned doesnt mean
    anything aside from: 1) is the model attributed to a worker; if yes, then
    it always will be". Pin is PERMANENT ATTRIBUTION — it answers "does this
    model belong to this worker?", not "when do the bytes arrive". A pinned
    model is still a lazy download, same as any other. Pinning previously
    implied a pre-pull here, which made pin a de-facto transfer order: on ae,
    65/65 assigned models were pinned, so deleting them re-pulled all 65 via
    _reconcile_loop and filled the operator's workstation to 0 bytes free
    (2026-07-16). "none should be pulling at all. they should be lazy."

    Everything else (the on-demand DEFAULT, and now 📌pin) waits to be called.
    NOTE for reconcile: for a non-static model, "assigned but not on disk" is
    the CORRECT resting state, not drift to converge.
    """
    try:
        return _residency(model_key) == "static"
    except Exception:  # noqa: BLE001 — a settings read must not break adoption
        # Fail LAZY: the worst case is one first-call download, whereas failing
        # eager re-creates the storm this function exists to prevent.
        return False


def _sync_assignment(state: "WorkerState", worker: dict) -> None:
    """React to central's worker record: adopt its model list.

    Central owns the assignment (set in the UI). The agent reads it back from
    every register/heartbeat response. Adoption is LAZY (see _eager_pull):
    being assigned a model does NOT download it — only 🔒static models are
    pre-pulled here; every other tier (the on-demand default AND 📌pinned)
    downloads on first call. Pin is permanent ATTRIBUTION, never a transfer
    order. Without this adoption the worker never knew about UI allocation
    changes.

    Seating is a SEPARATE concern from downloading: _fill_empty_slots still
    runs on every assignment change and seats models that are ALREADY LOCAL,
    regardless of tier.
    """
    if not isinstance(worker, dict):
        return
    if "models" not in worker:
        # No authoritative assignment list in this response — adopt nothing,
        # and above all don't treat it as "everything unassigned" below.
        return
    models = worker.get("models") or []
    changed = models != state.assigned_models
    state.assigned_models = list(models)
    # Tiers v3 lazy cleanup: with central's authoritative list in hand, drop
    # residency overrides (static OR on-demand) for models no longer assigned
    # — unless pinned (📌 = permanent attribution). Runs every heartbeat, so
    # it also catches unassigns that happened while this agent was down.
    try:
        _prune_stale_residency(state)
    except Exception as exc:  # noqa: BLE001 — cleanup must not break adoption
        logger.warning("residency prune failed: %s", exc)
    if not changed:
        return
    logger.info("assignment updated: serving %s", models or "(nothing)")

    # Lazy by default: pre-pull ONLY 🔒static, the one tier that promises local
    # presence. Everything else — the on-demand default AND 📌pinned — downloads
    # on first call via _ensure_present. Pin is attribution, not a pre-fetch.
    for model_key in models:
        if _eager_pull(model_key):
            logger.info("pre-provisioning %s (static — eager tier)", model_key)
            _kick_provision(state, model_key, purpose="assign")
    # Slice 9: already-local models can be seated right now — don't wait for
    # the maintenance tick. Background thread: fills block on slot loads.
    # NOT a download: this seats models whose files are ALREADY on disk, so it
    # runs for every tier — an on-demand model that was downloaded by an
    # earlier call still gets its seat back on an assignment change.
    threading.Thread(target=_fill_empty_slots, args=(state,), daemon=True).start()


def _kick_provision(state: "WorkerState", model_key: str,
                    purpose: str = "reconcile") -> None:
    """Provision (and per-policy preload) ONE assigned model in the background.

    Shared by assignment adoption and the UTIL-08 reconcile loop; the
    _provisioning guard makes concurrent kicks a no-op.

    ``purpose`` ("assign" from adoption, "reconcile" from the loop) is a
    BACKGROUND purpose (2026-07-17): central MAY 409 this pull if it would push
    the worker over its storage budget ("central abides by the limits set within
    its own backend"). Contrast the demand path (_ensure_present), which is never
    budget-refused centrally."""
    if restart_requested():
        # A restart is underway — don't spin up a NEW transfer pool into a process
        # about to exit (it would only be torn down by _shutdown_executors).
        return
    # k2: single choke for BOTH callers (assignment adoption's eager pre-pull
    # and the UTIL-08 reconcile loop's re-kick). An operator BLOCK does not
    # auto-unassign, so a blocked static model can still be sitting in
    # state.assigned_models and NOT local — without this, the reconcile loop
    # would re-kick a doomed pull of it every reconcile_interval_s forever.
    # Log once, not every pass; never a request refusal (background-only).
    if _is_blocked_locally(model_key):
        _log_blocked_skip_once(model_key, f"provisioning ({purpose})")
        return
    with state._provision_lock:
        if model_key in state._provisioning:
            return
        state._provisioning.add(model_key)

    if True:  # (indentation shim — keeps the battle-tested _bg body verbatim)
        def _bg(mk=model_key):
            def _prog(done, total, fname=None):
                # Mirrors the inference-time SSE progress, but recorded on state
                # so the heartbeat can report it (the panel polls heartbeats).
                frac = (done / total) if total else 0.0
                with state._provision_lock:
                    entry = state._provision_progress.setdefault(mk, {})
                    entry.update(done_bytes=done, total_bytes=total or 0,
                                 frac=round(frac, 4))
                    # Provenance (item 4): _provision_now streams a "source=…"
                    # pseudo-filename when it picks central vs HF — keep it on
                    # the entry so the console can attribute the pull.
                    if isinstance(fname, str) and fname.startswith("source="):
                        entry["source"] = fname[len("source="):]
            try:
                from .provision import ensure_model_present, model_is_local
                # ComfyUI-backed rows: everything is symlinks — already-
                # loadable / link-from-layout / pull-then-link. ComfyUI owns
                # its own residency, so no runner preload either.
                try:
                    from .provision import (ensure_comfy_checkpoint,
                                            ensure_model_registered)
                    from .imports import get_model_config
                    _ck = ensure_model_registered(mk, state.central_url) or mk
                    if getattr(get_model_config(_ck), "framework", None) == "comfy":
                        ok = ensure_comfy_checkpoint(_ck, state.central_url)
                        logger.info("comfy checkpoint for %s: %s", mk,
                                    "ready" if ok else "NOT available")
                        return
                except Exception:  # noqa: BLE001 — fall through to normal flow
                    pass
                # Hardening: model_is_local RAISES for a key this worker's
                # registry hasn't learned yet — that must trigger the pull
                # (which starts with ensure_model_registered), not abort it.
                try:
                    _has_files = model_is_local(mk)
                except Exception:  # noqa: BLE001
                    _has_files = False
                if not _has_files:
                    logger.info("pre-provisioning assigned model %s…", mk)
                    ensure_model_present(mk, state.central_url, progress=_prog,
                                         state=state, purpose=purpose)
                    logger.info("pre-provisioned %s", mk)
                    state.refused.pop(mk, None)   # it fit after all
                # Warm-up policy (v3 final semantics):
                #   * slots box — seat assignment is the SLOT-FILLER's job
                #     (slice 9, static-first): no in-process preload here, so
                #     nothing double-loads. Files just landed — kick a fill.
                #   * no slots — static always eager-warms in-process; other
                #     models (default on-demand) warm only behind the
                #     WORKER_PRELOAD/WORKER_POOL gate and TTL-yield when idle.
                _preload = os.environ.get(
                    "WORKER_PRELOAD",
                    "1" if os.environ.get("WORKER_POOL", "").strip() else "0",
                ).strip().lower() in ("1", "true", "yes", "on")
                _res = _residency(mk)
                _has_slots = False
                try:
                    from ..managers.serve.slots import slots_enabled
                    _has_slots = slots_enabled()
                except Exception:  # noqa: BLE001
                    pass
                # Slot boxes: seat slot-eligible (GGUF) models — static-first.
                if _has_slots:
                    try:
                        _fill_empty_slots(state)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("post-provision slot fill failed: %s", exc)
                # In-process warm for static (always) or preload models the slot
                # filler does NOT seat — transformers/vision/in-process GGUF. This
                # used to be an `elif _has_slots`, so a STATIC TRANSFORMERS model on
                # a slots box (ae/computron) never loaded: the filler only seats
                # GGUF and this branch was skipped, leaving a hollow shell at 0 VRAM.
                # Warm here whenever the model is not already a live slot occupant
                # (so a seated GGUF model is never double-loaded).
                if _preload or _res == "static":
                    try:
                        from abstract_hugpy_dev.managers.dispatch.dispatch import runner_for
                        if mk not in _slot_occupants():
                            logger.info("preloading (warming) %s…%s", mk,
                                        " [static — forced]" if (_res == "static" and not _preload) else "")
                            runner = runner_for(model_key=mk)   # builds + caches the runner
                            # runner_for only BUILDS the runner; lazy in-process
                            # runners (transformers/DeepCoder) defer the weight load
                            # to first use, so stopping here leaves a hollow shell at
                            # 0 VRAM/RAM that still reads "loaded". static means LIVE
                            # in the resources — force the weights resident now.
                            _ensure = getattr(runner, "ensure_loaded", None)
                            if callable(_ensure):
                                _ensure()
                            logger.info("preloaded %s (resident)", mk)
                    except Exception as exc:
                        logger.warning("preload of %s failed: %s", mk, exc)
            except BudgetRefusal as exc:
                # Not a failure — a DECISION, made before any bytes moved. Record
                # it so the heartbeat reports the model as MISSING with an honest
                # reason (hover text) instead of a pull that never starts.
                state.refused[mk] = dict(exc.reason)
                logger.error("pre-provision of %s REFUSED: %s", mk,
                             exc.reason.get("reason"))
            except Exception as exc:
                logger.warning("pre-provision of %s failed: %s", mk, exc)
            finally:
                with state._provision_lock:
                    state._provisioning.discard(mk)
                    state._provision_progress.pop(mk, None)

        def _bg_gated(mk=model_key):
            # Thundering-herd gate: acquire a slot in the fleet-wide provision
            # semaphore (default 1 = fully serial) BEFORE running the
            # battle-tested _bg body above, release after — win, lose, or
            # exception. This only throttles HOW MANY of these background
            # threads may be doing the heavy ensure_model_present() work at
            # once; it does NOT throttle the inference-triggered path
            # (_ensure_present / _ensure_present_streaming), which calls
            # ensure_model_present() directly and never goes through
            # _kick_provision — so a live chat waiting on a model is never
            # stuck behind a long queue of background assignment pre-fetches.
            # Blocking here (not a timeout/try-acquire) is intentional: every
            # assigned model must EVENTUALLY provision, just not all at once;
            # the per-model _provisioning guard above already prevents the
            # same key from queuing twice, so the wait is bounded by the
            # number of genuinely distinct models still ahead of it.
            state._provision_semaphore.acquire()
            try:
                _bg(mk)
            finally:
                state._provision_semaphore.release()

        threading.Thread(target=_bg_gated, daemon=True).start()


# ── UTIL-08: desired-state reconcile ─────────────────────────────────────────
# Assignment adoption only fires on CHANGE, so a failed pull used to drift
# forever (assigned, files absent, nobody retries until the operator touches
# the assignment). The reconcile loop re-kicks provisioning for any assigned
# model whose files are missing; models_local in the heartbeat gives central
# the disk-truth to SHOW the drift meanwhile.

_MODELS_LOCAL_CACHE: dict = {"at": 0.0, "value": []}


def _models_local(state: "WorkerState") -> list[str]:
    """Assigned models whose files are actually on THIS worker's disk (60s
    cache — model_is_local walks directories; don't pay that every beat)."""
    now = time.time()
    if now - _MODELS_LOCAL_CACHE["at"] < 60.0:
        return _MODELS_LOCAL_CACHE["value"]
    if not state.assigned_models:
        # Startup window: the assignment list arrives with the FIRST heartbeat
        # response — caching an empty walk here made the console show
        # everything '✗ missing' for ~60s after any restart. Don't cache.
        return []
    out: list[str] = []
    try:
        from .provision import model_is_local
        for mk in list(state.assigned_models):
            try:
                if model_is_local(mk):
                    out.append(mk)
            except Exception:  # noqa: BLE001 — one bad row must not hide the rest
                pass
    except Exception:  # noqa: BLE001
        pass
    _MODELS_LOCAL_CACHE.update(at=now, value=out)
    return out


def _reconcile_loop(state: "WorkerState") -> None:
    """Every reconcile_interval_s (default 600): any assigned 🔒static model that
    is NOT local and NOT already provisioning gets its provisioning re-kicked.
    Static is the ONLY tier that promises local presence. Converges failed pulls
    instead of drifting until the next assignment change; the _provisioning guard
    + single-flight lock keep it idempotent.

    Lazy-download doctrine (2026-07-16): a non-static model that is assigned but
    absent is NOT drift — it is the correct resting state, and it stays absent
    until something calls it. Re-kicking it here would silently rebuild the very
    provision storm _sync_assignment stopped, just 10 minutes later.

    That includes 📌pinned: pin is permanent ATTRIBUTION, not a residency
    guarantee. This loop treating pin as eager IS the 2026-07-16 incident — the
    operator deleted ae's models and all 65 (65/65 assigned there were pinned)
    re-pulled from here within 10 minutes, filling his workstation to 0 bytes
    free. A pinned model that is absent is absent on purpose until called."""
    while True:
        time.sleep(max(60, int(_RUNTIME_SETTINGS.get("reconcile_interval_s", 600))))
        if restart_requested():
            return                      # stop scheduling transfers into an exit
        try:
            local = set(_models_local(state))
            for mk in list(state.assigned_models):
                if not _eager_pull(mk):
                    continue      # non-static: absent is correct, not drift
                with state._provision_lock:
                    busy = mk in state._provisioning
                if mk not in local and not busy:
                    logger.warning("reconcile: static model %s promises local "
                                   "presence but is missing on disk — "
                                   "re-kicking provisioning", mk)
                    _MODELS_LOCAL_CACHE["at"] = 0.0   # re-check after the pull
                    _kick_provision(state, mk)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("reconcile iteration failed: %s", exc)


def _load_worker_id(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return (json.load(fh) or {}).get("worker_id")
    except (OSError, ValueError):
        return None


def _save_worker_id(path: str, worker_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"worker_id": worker_id}, fh)
    except OSError:
        logger.warning("could not persist worker id to %s", path)


# ---------------------------------------------------------------------------
# Self-update — track central's required package version (Slice 1).
#
# Central advertises ``required_pkg_version`` in every register/heartbeat
# response. When it differs from the version installed here, we pip-install the
# pinned target from central's own simple index (the one channel every worker
# can already reach outbound) and re-exec the process. The worker-id is
# persisted, so the restarted agent re-registers as the same worker — central
# sees a brief reconnect, not a new worker.
# ---------------------------------------------------------------------------

# Don't re-attempt the same target version more than once per this window. A
# pinned ``==target`` install means success implies an exact version match, so
# the only way we'd retry is a genuinely failed/unavailable build — back that
# off instead of hammering pip every heartbeat.
_UPDATE_RETRY_BACKOFF = 300.0


def _installed_pkg_version(pkg_name: str) -> str | None:
    """Version pip has ON DISK for ``pkg_name`` (installed dist metadata).

    This is what a NOT-YET-EFFECTIVE self-update flips FIRST: ``pip install``
    swaps the site-packages files (and their ``*.dist-info``) before this process
    has re-exec'd, so ``metadata.version()`` reports the NEW version while the
    OLD modules are still the ones executing in memory. So it is the DISK truth,
    NOT the running-image truth — never report it in the heartbeat (see
    ``_running_pkg_version``). It is still the right thing to gate the self-update
    pip on (has the target already been fetched to disk?).
    """
    from importlib import metadata
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return None


# ── Running-image version, snapshotted at import (honest heartbeat source) ────
# The 2026-07-20 ae incident: after the 0.1.196 pip self-update, ae's heartbeat
# reported pkg_version 0.1.196 / version_ok:true while the RUNNING agent still
# served the OLD route set (404 on the new /slots/<id>/relaunch) — because the
# heartbeat sourced its version from ``_installed_pkg_version`` (live DISK
# metadata, already flipped by pip) instead of from the code actually running.
# That is COSMETIC convergence: central believes the fleet is up to date while a
# worker silently serves stale code.
#
# The honest source is ``abstract_hugpy_dev.__version__`` — a source-file literal
# bound when the package was imported at process start. A pip upgrade rewrites
# that file on disk, but the in-memory module object keeps the old value until a
# genuinely fresh process re-imports it. So this constant tells the truth across
# a not-yet-effective upgrade: report OLD until the process really re-execs, and
# central's version_ok stays FALSE (a visible skew) instead of going cosmetically
# green. Captured ONCE here, at import, so nothing can later mutate it.
try:
    from abstract_hugpy_dev import __version__ as _RUNNING_IMAGE_VERSION
except Exception:  # noqa: BLE001 — run-from-copied-file: no package __version__
    _RUNNING_IMAGE_VERSION = None


def _running_pkg_version(pkg_name: str) -> str | None:
    """Version of the CODE THIS PROCESS IS RUNNING — the honest heartbeat source.

    Snapshotted from ``abstract_hugpy_dev.__version__`` at import (above), NOT
    read live from dist metadata, so a self-update that pip-installed new files
    on disk but has not yet re-exec'd keeps reporting the OLD version — the truth
    — rather than the disk's new version. Falls back to disk metadata ONLY when
    there is no package ``__version__`` to trust (a standalone copied agent.py),
    where disk metadata is the best signal available.
    """
    if _RUNNING_IMAGE_VERSION:
        return _RUNNING_IMAGE_VERSION
    return _installed_pkg_version(pkg_name)


def _update_state_path(args) -> str:
    return args.id_file + ".update.json"


# ── operator runtime settings (daylight item 3: console-managed serving) ────
# The agent's OWN config file, set from the console via /ops/config. It is the
# SOURCE OF TRUTH over env/unit drop-ins for the keys it holds — the fix for
# the SLOT_COUNT drop-in ghost (a limits.conf silently resurrecting slots on
# every restart). Precedence: settings file > env > built-in default; the
# heartbeat reports the EFFECTIVE values + their source so the console always
# shows truth.

_SETTINGS_KEYS = {"slot_count", "residency", "on_demand_ttl_s",
                  "reconcile_interval_s", "pinned", "comfy_url",
                  "hot_cache_root", "profiles", "model_profiles",
                  "ctx_pct",   # widen key by key (ctx_pct: per-model context %, slice 11)
                  # t21 tolerance bands (per-model maps, siblings of ctx_pct):
                  # the deviation% each explicit allocation may flex under
                  # contention, plus a compress-others priority. Populated by
                  # central's projection of the spill bands (release-time bridge);
                  # the flex engine (_vram_evict_to_fit) reads them here.
                  "ctx_deviation_pct", "vram_deviation_pct",
                  "ram_deviation_pct", "priority"}
_SETTINGS_SOURCE: dict = {}              # key -> "settings" | "env" | "default"
_RUNTIME_SETTINGS: dict = {}             # the loaded settings, for live readers
# t21 flex: the ctx% the VRAM admission engine COMMITTED a model to under
# contention (compressed toward its ctx band floor to fit without evicting). It
# is read by _ctx_pct so the SERVED -c matches the KV admission reserved — never
# reserve-small-then-serve-large (that OOMs). Cleared when the model is evicted
# or when it later fits at target (uncontended). Empty == no active flex.
_FLEX_CTX_FLOOR: dict = {}               # model_key -> committed compressed ctx%
# t21 stage (2.5): the honest GGUF PARTIAL-offload the VRAM admission COMMITTED a
# model to when its full weights don't fit the GPU even after flex+evict. Value:
# {"path": served_quant_path, "n": n_gpu_layers}. The n rides the slot child
# cmdline (via the verdict -> slot opts) AND pins the in-process llama_cpp load
# (spill.set_ngl_override on the path) so a sharded model never re-OOMs on the
# shard-blind autofit. Cleared when the model is re-admitted (re-decides) or fits
# fully. Empty == no active partial-offload commitment.
_PARTIAL_NGL: dict = {}                   # model_key -> {"path", "n"}
# t28 load-and-learn: the worker's half of the calibration loop.
#   _CALIB_BUFFER      pending calibration_sample dicts, drained into each
#                      heartbeat (worker->central, additive/optional).
#   _CALIB_SAMPLED     model_keys already sampled for their CURRENT residency
#                      episode (dedup — one measured sample per load, re-armed
#                      when the model leaves residency).
#   _CALIB_CORRECTIONS model_key -> clamped learned correction, ADOPTED from the
#                      heartbeat reply (central aggregates + gates + clamps; the
#                      worker just applies it, with a defensive re-clamp). Empty
#                      == no learned number -> the static x1.15 stands.
# All best-effort telemetry: a calibration failure must NEVER break a beat or an
# admission. The whole layer is inert when HUGPY_CALIBRATION=off.
_CALIB_MAXLEN = 64
_CALIB_BUFFER: list = []
_CALIB_SAMPLED: set = set()
_CALIB_CORRECTIONS: dict = {}
_CALIB_LOCK = threading.Lock()
# k2: the worker's adopted view of the operator's model BLOCK set (aa4aea3),
# learned off the heartbeat reply (worker['blocked_models'] = [model_key, ...]),
# the exact same additive/omit-when-empty wire idiom as calibration. Closes the
# gap the 2026-07-18 ae incident exposed: the block primitive covers every
# CENTRAL path already, but a worker's OWN background reconciler loops (slot
# fill, provisioning re-kick) had no way to learn a model was blocked out from
# under an assignment that is still on record (block does not auto-unassign) —
# so they kept retrying a doomed load every ~60s indefinitely.
#   _BLOCKED_MODELS   model_keys blocked as of the last heartbeat. Empty ==
#                     nothing blocked (or an older/central without the
#                     feature; the reply just omits the key).
#   _BLOCKED_LOGGED   model_keys a background loop has already logged a skip
#                     for, so the skip logs ONCE per block episode, not every
#                     tick. Re-armed on unblock so a future re-block logs again.
# ONLY gates the worker's own background warm/load-ahead loops — never an
# explicit relay request (that stays central's honest-refusal job).
_BLOCKED_MODELS: set = set()
_BLOCKED_LOGGED: set = set()
_BLOCKED_LOCK = threading.Lock()


def _adopt_blocked_models(worker: "dict | None") -> None:
    """Adopt central's published model BLOCK set from the heartbeat reply.
    Central only sends the key when the set is non-empty; an older/unblocked
    central omits it -> the local set clears -> nothing reads as blocked.
    Mirrors _adopt_calibration."""
    raw = (worker or {}).get("blocked_models") or []
    parsed = ({str(mk) for mk in raw if mk}
              if isinstance(raw, (list, tuple, set)) else set())
    with _BLOCKED_LOCK:
        _BLOCKED_MODELS.clear()
        _BLOCKED_MODELS.update(parsed)
        _BLOCKED_LOGGED.intersection_update(parsed)  # re-arm anything unblocked


def _is_blocked_locally(model_key: "str | None") -> bool:
    """True iff ``model_key`` is in this worker's adopted BLOCK set. For gating
    the worker's OWN background loops only — see the module note above."""
    if not model_key:
        return False
    with _BLOCKED_LOCK:
        return model_key in _BLOCKED_MODELS


def _log_blocked_skip_once(model_key: str, where: str) -> None:
    """Log a background loop's skip of a blocked model exactly ONCE per block
    episode, not every ~60s tick — the direct fix for the 2026-07-18 ae
    incident (slot-fill reconciler retried a blocked model every beat for
    ~4.75 hours, each attempt failing a full-GPU fit)."""
    with _BLOCKED_LOCK:
        if model_key in _BLOCKED_LOGGED:
            return
        _BLOCKED_LOGGED.add(model_key)
    logger.info("%s: skipping %s — blocked from the serving pool by the "
                "operator (won't retry until unblocked)", where, model_key)


# ── Per-worker KEEP-WARM STAR (boot_prewarm) ────────────────────────────────
# Operator RULINGS 2026-07-23 (these SUPERSEDE the earlier "boot-once" design):
#   RULING 1 — "the star is the ONLY warm source. nothing warms until starred
#               (or static)."
#   RULING 2 — "star = reconcile-kept-warm" (NOT boot-once): a starred model
#               evicted under load COMES BACK on the next reconcile beat. The ⭐
#               star IS the keep-warm designation.
#
# Central publishes this worker's star on every register/heartbeat reply as a
# plain scalar ``worker['boot_prewarm'] = "<model_key>"`` — the exact additive/
# omit-when-unset wire idiom as calibration/reservations/blocked_models, so an
# older released central just omits it and this code tolerates its absence.
#
# SEMANTICS (the ⭐ lever): the star = the operator's keep-warm designation.
# reconcile keeps it warm EVERY BEAT — this adopt runs on each reply and loads
# the star IF NOT CURRENTLY LOADED (an idempotent check against the live
# dispatch/slot residency). So: evicted-under-pressure → next beat reloads it;
# already-loaded → no-op. It stays a NORMALLY EVICTABLE (FIFO) on-demand
# resident — NOT eviction-protected. (For "start here AND stay here", the
# operator promotes the model to 🔒static — that tier, not this one, protects
# residency.) The identifier stays ``boot_prewarm`` (rename churn isn't worth
# it) but it now means "keep-warm star", not "boot once".
#   _STAR_WARMING  model_keys with a warm thread IN FLIGHT right now — an
#                  in-flight guard (NOT a done-latch): it only stops spawning a
#                  SECOND load thread for the same star while the first is still
#                  running, then clears. It never suppresses a re-warm after an
#                  eviction (that would re-break RULING 2). Cleared when the
#                  thread finishes.
_STAR_WARMING: set = set()
_STAR_WARM_LOCK = threading.Lock()


def _star_is_loaded(model_key: str) -> bool:
    """True iff the star model is CURRENTLY resident on this worker — in-process
    OR seated in a slot. The idempotent check that makes keep-warm a no-op when
    the star is already loaded (RULING 2: reload only when absent). Guarded —
    an accessor hiccup reads as 'not loaded' so a beat re-warms rather than
    wrongly skipping."""
    try:
        if model_key in set(loaded_model_keys()):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if model_key in _slot_occupants():
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _load_star_if_absent(state: "WorkerState", model_key: str) -> None:
    """Load the ⭐ star through the NORMAL on-demand path — NOT the static eager
    tier — IF it is not already resident. Provisions it present (downloading if
    absent, DEMAND purpose so the worker's own fit_plan evicts-to-fit rather than
    central budget-refusing), then warms it resident the same way the on-demand
    preload branch does (slot-fill for a GGUF slot occupant; runner_for +
    ensure_loaded otherwise).

    Crucially this does NOT touch residency: it never writes a 'static' override,
    so the loaded model stays a normal FIFO-evictable resident (RULING 2 keeps it
    warm by RELOADING each beat when absent, never by protecting it). Fully
    guarded — a keep-warm failure must NEVER crash the agent or a heartbeat.

    The _STAR_WARMING in-flight guard is cleared in a finally so a later beat
    (e.g. after an eviction) can warm the star again."""
    try:
        if _is_blocked_locally(model_key):
            _log_blocked_skip_once(model_key, "keep-warm star")
            return
        try:
            from .provision import (ensure_model_present, ensure_model_registered,
                                     model_is_local)
            # Learn the model from central if this worker's registry hasn't yet,
            # and work against the canonical local key.
            try:
                canonical = ensure_model_registered(model_key, state.central_url) or model_key
            except Exception:  # noqa: BLE001 — resolution failure -> try the bare key
                canonical = model_key
            # Comfy-backed rows own their own residency (symlinks); just ensure
            # the checkpoint is available and stop (no runner preload, no slot).
            try:
                from .provision import ensure_comfy_checkpoint
                from .imports import get_model_config
                if getattr(get_model_config(canonical), "framework", None) == "comfy":
                    ok = ensure_comfy_checkpoint(canonical, state.central_url)
                    logger.info("keep-warm star: comfy checkpoint for %s: %s",
                                canonical, "ready" if ok else "NOT available")
                    return
            except Exception:  # noqa: BLE001 — fall through to the normal flow
                pass
            # RE-ENTRY doctrine (k35 eviction-re-entry-designated): "reload from
            # disk, NEVER fetch." An evicted star must reload FROM DISK — it must
            # not re-download (re-fetching an evicted model risks re-pulling the
            # exact files a reap just removed — the 2026-07-17 ae pin-storm
            # class). The disk/VRAM distinction is what separates the two cases,
            # and ``model_is_local`` reads DISK PRESENCE ONLY (not VRAM
            # residency):
            #   * ``not _has`` (absent on disk) — a genuine FIRST ACTIVATION of a
            #     not-yet-present star (the operator just starred it, this box
            #     doesn't have it). Fetch it ONCE: that download fulfils the
            #     deliberate star order, and it's ONE model (not the pin-storm's
            #     many), so no storm risk.
            #   * ``_has`` (present on disk, not resident) — the RE-WARM /
            #     RE-ENTRY case. Eviction (_evict_model) frees VRAM/RAM only and
            #     NEVER deletes disk files, so a warmed-then-evicted star stays
            #     on disk → ``_has`` stays True → the fetch below is SKIPPED and
            #     the warm mechanism reloads from disk. Exactly k35's "reload,
            #     never fetch". (Only a REAP — a separate, guarded disk delete —
            #     flips ``_has`` False, at which point re-warm correctly treats it
            #     as a fresh first-activation, not an eviction re-entry.)
            try:
                _has = model_is_local(canonical)
            except Exception:  # noqa: BLE001
                _has = False
            if not _has:
                # Fetch fires ONLY here — genuine disk-absence = first activation.
                # Never on re-warm-after-eviction (that path keeps _has True).
                logger.info("keep-warm star: first-activation fetch of %s "
                            "(absent on disk)…", canonical)
                ensure_model_present(canonical, state.central_url, state=state,
                                     purpose="demand")
            # Warm it resident — the SAME mechanism the on-demand preload branch
            # uses, WITHOUT the _residency 'static' gate. Slot boxes seat a GGUF
            # via the slot filler; everything else warms in-process via runner_for.
            _has_slots = False
            try:
                from ..managers.serve.slots import slots_enabled
                _has_slots = slots_enabled()
            except Exception:  # noqa: BLE001
                pass
            if _has_slots:
                try:
                    _fill_empty_slots(state)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("keep-warm star: slot fill for %s failed: %s",
                                   canonical, exc)
            if canonical not in _slot_occupants():
                try:
                    from abstract_hugpy_dev.managers.dispatch.dispatch import runner_for
                    logger.info("keep-warm star: warming %s (on-demand, evictable)…",
                                canonical)
                    runner = runner_for(model_key=canonical)
                    _ensure = getattr(runner, "ensure_loaded", None)
                    if callable(_ensure):
                        _ensure()
                    logger.info("keep-warm star: warmed %s (resident, FIFO-evictable)",
                                canonical)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("keep-warm star: warm of %s failed: %s", canonical, exc)
        except BudgetRefusal as exc:
            # Won't fit — a DECISION, not a crash. Log and continue.
            logger.warning("keep-warm star %s REFUSED (won't fit): %s",
                           model_key, exc.reason.get("reason"))
        except Exception as exc:  # noqa: BLE001 — keep-warm must never crash the agent
            logger.warning("keep-warm star %s failed: %s", model_key, exc)
    finally:
        # Clear the in-flight guard so a FUTURE beat (e.g. after an eviction) can
        # re-warm the star — RULING 2. This is NOT a done-latch.
        with _STAR_WARM_LOCK:
            _STAR_WARMING.discard(model_key)


def _adopt_boot_prewarm(state: "WorkerState", worker: "dict | None") -> None:
    """Adopt central's per-worker KEEP-WARM STAR from a register/heartbeat reply
    and keep it warm — RULING 2: load it IF NOT CURRENTLY LOADED, every beat.

    Reconcile-kept (NOT boot-once): on EVERY reply carrying a star, if the star
    is not currently resident (evicted, or never loaded), kick a load; if it is
    already resident, no-op. So an eviction-under-pressure is repaired on the
    next beat, and a loaded star costs nothing. Central omits the key when no
    star is set, and an older/released central never sends it, so a reply without
    ``boot_prewarm`` is the normal no-op. Fully guarded (mirrors
    _adopt_blocked_models): a star adoption must NEVER fail a beat.

    Loading runs on a daemon thread so a multi-GB pull never blocks the heartbeat
    loop; the _STAR_WARMING in-flight guard stops a second thread racing the
    first for the same star (it is NOT a permanent latch — it clears when the
    thread finishes, so the next beat after an eviction can warm again)."""
    star = (worker or {}).get("boot_prewarm")
    if not star or not isinstance(star, str):
        return
    # Already resident? Keep-warm is a no-op (RULING 2: reload only when absent).
    if _star_is_loaded(star):
        return
    with _STAR_WARM_LOCK:
        if star in _STAR_WARMING:
            return  # a warm thread for this star is already in flight this beat
        _STAR_WARMING.add(star)
    logger.info("keep-warm star %s not resident — warming (evictable, NOT static)",
                star)
    threading.Thread(target=_load_star_if_absent, args=(state, star),
                     daemon=True).start()


_COMFY_URL_BASE_ENV = "_HUGPY_COMFY_URL_BASE"  # sentinel: the pre-projection
# COMFY_URL (systemd drop-in / env / none), captured once and carried across
# os.execv so clearing the setting reverts to the real base, never the last
# projected value.
_ENV_HOT_CACHE_ROOT = "HUGPY_HOT_CACHE_ROOT"   # the env hot_cache.py reads live
# (managers/serve/hot_cache.py::_root). Projecting the setting onto it is why the
# tier needs NO code change to become a per-worker attributable setting.
_HOT_CACHE_ROOT_BASE_ENV = "_HUGPY_HOT_CACHE_ROOT_BASE"  # sentinel: the pre-
# projection HUGPY_HOT_CACHE_ROOT (drop-in / env / none), captured once and
# carried across os.execv so clearing the setting reverts to the real base — the
# exact same dance as _COMFY_URL_BASE_ENV.


def _valid_comfy_url(cu: str) -> bool:
    """True for an http(s) URL that has a host — rejects scheme-only 'http://'
    and accepts a case-insensitive scheme ('HTTP://host' is fine)."""
    from urllib.parse import urlparse
    try:
        p = urlparse(cu.strip())
    except Exception:  # noqa: BLE001 — unparseable string is not a URL
        return False
    return p.scheme.lower() in ("http", "https") and bool(p.netloc)


def _residency(model_key: str) -> str:
    """Per-model residency POLICY (v3 final semantics, operator-locked
    2026-07-05). Exactly TWO tiers:

      * "on-demand" — the DEFAULT (no stored entry): loads on call; holds a
        slot seat until another model needs it (promotion) — slot occupants
        never TTL-yield; idle IN-PROCESS residents do (frees RAM on
        slot-less boxes). "serving"/"warm" are accepted legacy write-
        synonyms for this default; stored legacy entries read as it too.
      * "static" — the only stored override: locked seat, never swapped out
        or yielded, eager-warmed (the ONLY tier that pre-pulls — see
        _eager_pull). Orthogonal to 📌 pin: pin makes the ATTRIBUTION
        permanent (the override survives unassign-prune), but adds no
        residency or presence promise of its own.

    "Serving" is purely a STATE (a model in a slot), never a policy.
    """
    val = (_RUNTIME_SETTINGS.get("residency") or {}).get(model_key)
    return "static" if val == "static" else "on-demand"


def _pinned(model_key: str) -> bool:
    """📌 pin: the model's ALLOCATION survives restarts — and NOTHING else.

    CANONICAL STATEMENT (operator ruling, 2026-07-17): "the pins only should
    designate that the model allocation survives restarts. the allocation only
    stipulates the routing for that model (to that worker). neither of those
    should have any bearing on the pull or eviction". (Consistent with the
    2026-07-16 answer: "pinned doesnt mean anything aside from: 1) is the model
    attributed to a worker; if yes, then it always will be".)

    So pin answers exactly one question — "does this worker's ALLOCATION
    (routing) for this model survive restarts and unassign attempts?" — with
    "yes, durably". It says NOTHING about when the bytes arrive or whether they
    stay. Concretely, pin DOES:
      * make central refuse unassign while pinned (409),
      * keep residency overrides + the allocation alive across the
        unassign-prune (_prune_stale_residency) and restarts.

    Pin does NOT: pre-fetch, eager-warm, guarantee residency, promise the files
    are on this disk, OR protect the files from eviction/reaping. A pinned model
    is a LAZY download like any other — it arrives on first CALL
    (_ensure_present) — and its files are a normal eviction/reap CANDIDATE
    (budget._is_protected, _reap_scan, workers.storage_proposal all treat pin as
    NON-protecting as of 2026-07-17). Evicting a pinned model's files leaves the
    pin + allocation untouched: routing survives, bytes re-pull on next call.
    Pin's only eviction role is a trivial FIFO tiebreak (unpinned evict first at
    an exact last_picked tie). Do not re-add pin to _eager_pull OR to any disk
    guard: those conflations are the day-one tripwire — the eager one filled the
    operator's workstation to 0 bytes free on 2026-07-16 (ae: 65/65 assigned
    models pinned = every model eager). 🔒static is the ONLY tier that promises
    local presence and protects files.
    """
    return bool((_RUNTIME_SETTINGS.get("pinned") or {}).get(model_key))


def _resolve_model_profile(model_key: str) -> "dict | None":
    """Env-profiles (stage 1) resolver, registered onto managers.serve.profiles
    so the runner spawn seam can decide without reading operator settings itself.

    Reads the live ``model_profiles`` attribution + ``profiles`` manifest from
    _RUNTIME_SETTINGS and returns, for an attributed model,
    ``{'name','state','bin','error'}`` where ``state`` is ready|materializing|
    error and ``bin`` is the profile venv's bin dir ONLY when ready (the value
    the slot child's PATH/interpreter is built from). None when the model has no
    profile — the base serving path is untouched."""
    name = (_RUNTIME_SETTINGS.get("model_profiles") or {}).get(model_key)
    if not name:
        return None
    spec = (_RUNTIME_SETTINGS.get("profiles") or {}).get(name) or {}
    packages = spec.get("packages") or []
    from ..managers.serve import profiles as _profiles
    state = _profiles.state_for(name, packages)
    out = {"name": name, "state": state,
           "bin": _profiles.profile_bin_dir(name) if state == "ready" else None}
    if state == "error":
        out["error"] = (_profiles.read_state(name) or {}).get("error")
    return out


# ---------------------------------------------------------------------------
# Contention-based residency (doctrine 2026-07-11): the worker-side policy
# registered onto dispatch's LRU mechanism (dispatch.set_fit_check /
# set_evictable / set_post_evict_hook). An on-demand model stays hot until a NEW
# load needs its memory; then the LRU on-demand resident yields.
# ---------------------------------------------------------------------------
def _incoming_need_bytes(model_key: str) -> "int | None":
    """Best-effort bytes the incoming model's weights will want (× a small
    headroom factor), resolved from its on-disk size the same way the loader does
    (route_destination). None when the size is unknown — the fit-guard then fails
    OPEN (never blocks an unmeasurable load).

    GGUF landmine (fixed 2026-07-14, mirrors central model_meta): a GGUF repo
    commonly holds several quantizations but only ONE serves, so summing every
    ``.gguf`` in the dir badly overstates the VRAM need — a 24-quant 8B repo read
    as ~94GB (×1.15 ≈ 108GB) and blocked loads on a 7.4GB card even though the
    single served quant is ~5GB. For gguf/llama_cpp we size by the SINGLE
    effective serving quant (+ its mmproj), resolved by the SAME helper central
    uses (``gguf_variants_detail`` → ``effective_bytes``, which honors the
    operator ``gguf_file`` override / ``cfg.filename`` / deterministic auto-rank —
    exactly what the runner loads). We deliberately do NOT fall back to the
    inflated dir-sum for GGUF: an unresolvable effective quant returns None
    (fail-open) rather than re-introducing the very over-count this fixes.
    Non-GGUF frameworks (safetensors/bin) are a single weight set, so the
    dispatch weight-sum stays accurate and is used exactly as before."""
    try:
        from ..imports import route_destination
        from ..imports.config.main import get_model_config
        from ..managers.dispatch.dispatch import _dir_size_detail
        cfg = get_model_config(model_key, dict_return=True)
        path = route_destination(cfg)
        if not path:
            return None
        framework = str((cfg or {}).get("framework") or "").lower()
        if framework in ("gguf", "llama_cpp"):
            # Effective-quant-aware sizing. On any resolution miss, return None
            # (fail open) — never the dir sum, which is the bug this fixes.
            try:
                from ..managers.serve.overrides import gguf_variants_detail
                gguf = gguf_variants_detail(model_key, path, cfg) or {}
            except Exception:  # noqa: BLE001 — best-effort; unresolved -> fail open
                gguf = {}
            eff = gguf.get("effective_bytes")
            return int(eff * 1.15) if eff else None
        detail = _dir_size_detail(path)
        weight = detail.get("weight_bytes") or detail.get("model_bytes")
        return int(weight * 1.15) if weight else None
    except Exception:  # noqa: BLE001 — best-effort; unknown size -> fail open
        return None


# ── Context (KV) as an allocation variable (slice 11 / t27) ─────────────────
def _model_max_ctx(model_key: str, cfg: dict | None = None) -> int | None:
    """The model's MAX context window (tokens) — the 100% ceiling ctx_pct scales.
    Registry model_max_length first (central's truth), else the trained ctx from
    the model's own metadata (gguf .context_length / config max_position_embeddings)."""
    try:
        from ..imports.config.main import get_model_config
        cfg = cfg if cfg is not None else get_model_config(model_key, dict_return=True)
        mml = (cfg or {}).get("model_max_length") or (cfg or {}).get("tokenizer_model_max_length")
        if mml:
            return int(mml)
    except Exception:  # noqa: BLE001
        pass
    try:
        geo = _model_kv_geometry(model_key, cfg)
        if geo.get("ctx_train"):
            return int(geo["ctx_train"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _ctx_pct(model_key: str) -> int | None:
    """Per-model ctx allocation percentage (1-100) from the worker settings map
    (same seam as residency/pinned). None when unset -> today's default ctx
    behavior is byte-identical (no ctx_pct term forced).

    t21: when the VRAM admission engine has COMMITTED this model to a compressed
    ctx to fit under contention (_FLEX_CTX_FLOOR), that floor WINS — the served
    -c must match the KV the admission reserved. The commit is itself within the
    model's ctx band (never below the floor), so it is always a valid 1..100."""
    floored = _FLEX_CTX_FLOOR.get(model_key)
    if floored is not None:
        try:
            return max(1, min(100, int(floored)))
        except (TypeError, ValueError):
            pass
    val = (_RUNTIME_SETTINGS.get("ctx_pct") or {}).get(model_key)
    try:
        v = int(val)
    except (TypeError, ValueError):
        return None
    if v < 1:
        return 1
    if v > 100:
        return 100
    return v


def _ctx_deviation_pct(model_key: str) -> "float | None":
    """Per-model ctx tolerance band (percent points, 0..100) from settings, or
    None when unset. t21: how far the ctx allocation may flex under contention
    (ctx is the cheapest flex). Reads the target ctx_pct from _RUNTIME_SETTINGS,
    NOT the _FLEX_CTX_FLOOR override, so the band is always relative to the
    operator's target, not a prior compression."""
    val = (_RUNTIME_SETTINGS.get("ctx_deviation_pct") or {}).get(model_key)
    try:
        d = float(val)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, d))


def _flex_priority(model_key: str) -> int:
    """Per-model flex priority (0 == normal) from settings, via the ONE seam
    (flex.flex_priority_key). Higher compresses/evicts lower-priority neighbours
    first. The operator may change the priority SOURCE — see flex.py."""
    from .flex import flex_priority_key
    return flex_priority_key(
        {"priority": (_RUNTIME_SETTINGS.get("priority") or {}).get(model_key)})


def _flex_alloc(model_key: str) -> dict:
    """The per-model explicit-allocation view the flex engine consumes:
    ``{"priority", "ctx_deviation_pct"}``. Central projects the spill bands into
    the settings maps this reads (release-time bridge); tests populate them
    directly. Kept tiny + pure-ish so building subject/resident rows is cheap."""
    return {"priority": (_RUNTIME_SETTINGS.get("priority") or {}).get(model_key),
            "ctx_deviation_pct": _ctx_deviation_pct(model_key)}


def _vram_deviation_pct(model_key: str) -> "float | None":
    """Per-model VRAM tolerance band (percent points, 0..100) from settings, or
    None when unset. Symmetric with _ctx_deviation_pct; the stretch input to
    flex.band_ceiling when the partial-offload budget may reach a model's VRAM
    band CEILING (above its gpu_mem_gib target) under its own need. None (today,
    until central projects gpu_mem_gib_deviation_pct into the worker settings)
    collapses band_ceiling to the gpu_mem_gib target — the offload budget is then
    capped at exactly the explicit gpu_mem_gib, byte-identical to autofit's cap.
    Reads the SAME registry key central's _worker_fit mirror uses for the band
    floor, so worker and central agree the moment central populates it."""
    val = (_RUNTIME_SETTINGS.get("gpu_mem_gib_deviation_pct") or {}).get(model_key)
    try:
        d = float(val)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, d))


def _gguf_ngl_intent(model_key: str) -> "tuple[str, int | None]":
    """Decode the effective GGUF placement intent (t26) from HUGPY_N_GPU_LAYERS —
    the SAME wire the console's Autofit / Max GPU / CPU only controls ride, set
    per-model by _apply_spill before the load. Returns ``(intent, requested)``:

      * ``"-1"``            -> ("gpu",  None)  — Max GPU (as many as fit; for an
                                                oversize model == autofit, and it
                                                deliberately does NOT squeeze the
                                                ceiling reserve — see flex.plan_
                                                partial_offload).
      * ``0``/off/cpu/none  -> ("cpu",  None)  — CPU only (n_gpu_layers 0).
      * positive int ``N``  -> ("auto", N)     — honor the explicit layer count,
                                                capped to what fits.
      * unset / ``"auto"``  -> ("auto", None)  — autofit layers-that-fit.

    GGUF-only: for a GGUF a positive int IS a real layer count (unlike the
    transformers reading in spill.n_gpu_layers_intent, which collapses it to
    'auto'), so we decode it here rather than reuse that transformers-shaped
    helper."""
    raw = (os.environ.get("HUGPY_N_GPU_LAYERS") or "").strip().lower()
    if raw in ("", "auto"):
        return "auto", None
    if raw in ("off", "cpu", "none"):
        return "cpu", None
    try:
        n = int(raw)
    except ValueError:
        return "auto", None
    if n < 0:
        return "gpu", None
    if n == 0:
        return "cpu", None
    return "auto", n


def _served_gguf_geometry(model_key: str) -> "tuple[str | None, int | None]":
    """``(served_quant_path, total_layers)`` for a GGUF model — the SERVED quant's
    path and its ``.block_count``, resolved exactly as _model_kv_geometry does (no
    parallel reader). ``(None, None)`` for non-GGUF or on any resolution miss.

    Reused for BOTH the partial-offload layer math and the in-process
    n_gpu_layers override (the same path the in-process runner will load), so the
    plan and the load can never key off different files."""
    try:
        from ..imports import route_destination
        from ..imports.config.main import get_model_config
        cfg = get_model_config(model_key, dict_return=True)
        if str((cfg or {}).get("framework") or "").lower() not in ("gguf", "llama_cpp"):
            return None, None
        path = route_destination(cfg)
        if not path:
            return None, None
        try:
            from ..managers.serve.serve import _model_file_for
            picked = _model_file_for(model_key, get_model_config(model_key))
            if picked:
                path = picked
        except Exception:  # noqa: BLE001 — fall back to the route path
            pass
        from ..managers import spill
        return path, spill._gguf_layer_count(path)
    except Exception:  # noqa: BLE001 — unresolvable geometry -> caller keeps refusal
        return None, None


def _clear_partial_ngl(model_key: str) -> None:
    """Drop any committed partial-offload for ``model_key`` (this admission
    re-decides) and clear the in-process spill override for its path, so a model
    that now fits fully is never forced back onto a stale partial plan."""
    prev = _PARTIAL_NGL.pop(model_key, None)
    if prev and prev.get("path"):
        try:
            from ..managers import spill as _spill
            _spill.clear_ngl_override(prev["path"])
        except Exception:  # noqa: BLE001
            pass


def _resolved_ctx(model_key: str, cfg: dict | None = None) -> "tuple[int | None, int | None, int | None]":
    """Resolve the ctx to plan/serve for this model: (ctx_resolved, pct, max).

    ctx_resolved = pct% × model_max, clamped to the engine/server cap that exists
    today (serve.DEFAULT_LLAMA_CTX for llama.cpp). When ctx_pct is UNSET, returns
    (None, None, max) so callers fall back to today's default ctx path — the
    variable is opt-in and back-compat by construction."""
    pct = _ctx_pct(model_key)
    mx = _model_max_ctx(model_key, cfg)
    if pct is None or not mx:
        return None, pct, mx
    ctx = max(1, int(mx * pct / 100.0))
    # Clamp to the engine/server cap the loader would apply anyway (the existing
    # 'capping -c' logic — enforcement of the RESOLVED value, not a blind cap).
    try:
        from ..managers.serve.serve import DEFAULT_LLAMA_CTX
        framework = str((cfg or {}).get("framework") or "").lower()
        if framework in ("gguf", "llama_cpp"):
            ctx = min(ctx, int(DEFAULT_LLAMA_CTX))
    except Exception:  # noqa: BLE001
        pass
    return ctx, pct, mx


def _model_kv_geometry(model_key: str, cfg: dict | None = None) -> dict:
    """Per-engine KV geometry for a model: {n_layers, n_kv_heads, head_dim,
    ctx_train, dtype}. GGUF reads the served quant's header; transformers reads
    config.json. {} when neither resolves (caller uses the heuristic)."""
    try:
        from ..imports import route_destination
        from ..imports.config.main import get_model_config
        from ..managers import spill
        cfg = cfg if cfg is not None else get_model_config(model_key, dict_return=True)
        framework = str((cfg or {}).get("framework") or "").lower()
        path = route_destination(cfg)
        if not path:
            return {}
        if framework in ("gguf", "llama_cpp"):
            # The served quant file (same resolution the loader/need use).
            gguf_path = path
            try:
                from ..managers.serve.serve import _model_file_for
                picked = _model_file_for(model_key, get_model_config(model_key))
                if picked:
                    gguf_path = picked
            except Exception:  # noqa: BLE001
                pass
            return spill._gguf_kv_geometry(gguf_path)
        # transformers/other: config.json geometry.
        import json
        cfgp = os.path.join(path, "config.json") if os.path.isdir(path) else ""
        if cfgp and os.path.isfile(cfgp):
            with open(cfgp, "r", encoding="utf-8") as fh:
                return spill._transformers_kv_geometry(json.load(fh))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _kv_need_bytes(model_key: str, cfg: dict | None = None) -> "tuple[int, dict]":
    """KV-cache bytes for this model at its RESOLVED ctx, plus a detail dict for
    honest reporting. Returns (0, {...}) when ctx_pct is unset (no ctx term —
    today's behavior). Never silently zero when ctx_pct IS set: geometry-missing
    falls to spill.kv_bytes' conservative heuristic (logged)."""
    from ..imports.config.main import get_model_config
    cfg = cfg if cfg is not None else get_model_config(model_key, dict_return=True)
    ctx, pct, mx = _resolved_ctx(model_key, cfg)
    if ctx is None:
        return 0, {"ctx_pct": None, "ctx_resolved": None, "ctx_max": mx,
                   "geometry_source": None}
    from ..managers import spill
    geo = _model_kv_geometry(model_key, cfg)
    framework = str((cfg or {}).get("framework") or "").lower()
    if framework in ("gguf", "llama_cpp"):
        dtype_bytes = 2.0                    # llama.cpp caches fp16 by default
    else:
        dtype_bytes = spill._kv_dtype_bytes(geo.get("dtype"))
    source = "geometry" if (geo.get("n_layers") and geo.get("n_kv_heads")
                            and geo.get("head_dim")) else "heuristic"
    if source == "heuristic":
        logger.warning("kv: no full geometry for %s — using conservative "
                       "heuristic for the ctx reserve (%s tok @ %s%%)",
                       model_key, ctx, pct)
    kv = spill.kv_bytes(ctx_tokens=ctx, n_layers=geo.get("n_layers"),
                        n_kv_heads=geo.get("n_kv_heads"),
                        head_dim=geo.get("head_dim"), dtype_bytes=dtype_bytes)
    return int(kv or 0), {"ctx_pct": pct, "ctx_resolved": ctx, "ctx_max": mx,
                          "geometry_source": source, "kv_bytes": int(kv or 0)}


def _incoming_need_detail(model_key: str) -> dict:
    """THE authoritative fit-NEED for a model: weights + KV(resolved ctx), with
    the SPLIT for honest reporting. All fit paths (contention, slot ceiling,
    vision-fit, slice-10 admission) compute need through this so no path diverges.

    Returns {total, weights, kv, ...ctx detail}. ``total`` is None only when the
    WEIGHT size is unmeasurable (fail-open, exactly as _incoming_need_bytes did).
    The kv term is 0 when ctx_pct is unset — so a model with no ctx allocation is
    byte-identical to today."""
    weights = _incoming_need_bytes(model_key)
    if not weights:
        return {"total": None, "base_total": None, "calibration_correction": 1.0,
                "weights": weights, "kv": 0,
                "ctx_pct": None, "ctx_resolved": None, "ctx_max": None,
                "geometry_source": None}
    try:
        kv, det = _kv_need_bytes(model_key)
    except Exception:  # noqa: BLE001 — KV is additive; never break a working fit
        kv, det = 0, {"ctx_pct": None, "ctx_resolved": None, "ctx_max": None,
                      "geometry_source": None}
    base_total = int(weights) + int(kv or 0)
    # t28 load-and-learn: consult the learned per-model correction (median
    # measured/predicted from real loads, adopted from central, clamped + gated).
    # `total` — what every fit path prices against — becomes the corrected figure;
    # `base_total` (the UNcorrected prediction, incl. the static x1.15) is carried
    # for honest reporting AND is what a calibration_sample records, so the ratio
    # tracks the true base fudge instead of collapsing to a fixpoint at the
    # current correction. None correction -> total == base_total (byte-identical).
    corr = _calib_correction(model_key)
    total = int(base_total * corr) if corr else base_total
    return {"total": total, "base_total": base_total,
            "calibration_correction": (corr or 1.0),
            "weights": int(weights), "kv": int(kv or 0), **det}


# ── t28 load-and-learn — worker capture + learned-correction application ─────
def _calibration_enabled() -> bool:
    """Master switch (mirrors central's). Default ON; ``off``/``0``/``false``/
    ``no`` makes the whole worker-side layer inert — no capture, no correction."""
    return (os.environ.get("HUGPY_CALIBRATION") or "on").strip().lower() not in (
        "0", "off", "false", "no", "")


def _calib_correction(model_key: str) -> "float | None":
    """The clamped learned correction for a model, or None (the static x1.15
    stands). Read from the map adopted off the heartbeat reply. Central already
    clamps + gates; the [0.8, 1.5] re-clamp here is a defensive safety net so a
    malformed reply can never move need-pricing outside the doctrine band."""
    if not _calibration_enabled():
        return None
    with _CALIB_LOCK:
        c = _CALIB_CORRECTIONS.get(model_key)
    if c is None:
        return None
    try:
        return max(0.8, min(1.5, float(c)))
    except (TypeError, ValueError):
        return None


def _calib_verdict(device: "str | None", ngl, total_layers) -> str:
    """Classify a residency into the placement verdict a calibration sample
    carries: ``full`` (weights+kv fully GPU-resident — the ONLY class that feeds
    the ratio), ``partial`` (some layers on CPU), ``cpu``, or ``unknown``."""
    dev = (device or "").lower() if device else None
    try:
        ngl = int(ngl) if ngl is not None else None
    except (TypeError, ValueError):
        ngl = None
    try:
        tl = int(total_layers) if total_layers is not None else None
    except (TypeError, ValueError):
        tl = None
    if ngl == 0 or dev == "cpu":
        return "cpu"
    if ngl is not None and ngl > 0 and tl and ngl < tl:
        return "partial"
    if dev == "cuda" or (ngl is not None and (ngl == -1 or ngl > 0)):
        return "full"
    return "unknown"


def _build_calibration_success(mk: str, row: dict) -> "dict | None":
    """A measured calibration_sample from an allocation row (post-load): the
    BASE (uncorrected) prediction split paired with the measured VRAM/RSS. None
    when the prediction can't be sized. Omits None fields (additive wire)."""
    try:
        det = _incoming_need_detail(mk)
    except Exception:  # noqa: BLE001 — capture must never raise on the beat
        return None
    base_total = det.get("base_total")
    if not base_total:
        return None
    sample = {
        "model_key": mk,
        "engine": _model_framework(mk),
        "needs_weights_bytes": det.get("weights"),
        "needs_kv_bytes": det.get("kv"),
        "ctx_pct": det.get("ctx_pct"),
        "need_total_bytes": base_total,
        "verdict": _calib_verdict(row.get("device"), row.get("n_gpu_layers"),
                                  row.get("total_layers")),
        "n_gpu_layers": row.get("n_gpu_layers"),
        "total_layers": row.get("total_layers"),
        "vram_bytes": row.get("vram_bytes"),
        "rss_bytes": row.get("rss_bytes"),
        "device": row.get("device"),
        "ok": True,
        "ts": time.time(),
    }
    return {k: v for k, v in sample.items() if v is not None}


def _collect_calibration_from_allocations(allocs: "list | None") -> None:
    """Emit ONE measured sample per residency episode. Keys off the SAME
    allocations view the heartbeat already computes (per-process nvidia-smi
    VRAM), so it captures EVERY load path — on-demand, slot, warm/probe, reconcile
    — uniformly, and dedups via _CALIB_SAMPLED. Samples only once a footprint is
    actually measured (vram_bytes present); departed residents are re-armed so a
    reload re-samples."""
    if not _calibration_enabled():
        return
    resident_now: set = set()
    new_samples: list = []
    for row in (allocs or []):
        mk = (row or {}).get("model_key")
        if not mk:
            continue
        resident_now.add(mk)
        if row.get("vram_bytes") is None:
            continue                        # not measured yet — wait for a beat that does
        with _CALIB_LOCK:
            if mk in _CALIB_SAMPLED:
                continue
        s = _build_calibration_success(mk, row)
        if s:
            new_samples.append(s)
            with _CALIB_LOCK:
                _CALIB_SAMPLED.add(mk)
    with _CALIB_LOCK:
        _CALIB_SAMPLED.intersection_update(resident_now)
        _CALIB_BUFFER.extend(new_samples)
        if len(_CALIB_BUFFER) > _CALIB_MAXLEN:
            del _CALIB_BUFFER[:-_CALIB_MAXLEN]


def _record_calibration_refuse(model_key: str, det: "dict | None") -> None:
    """A load-FAIL sample from the VRAM admission refusal: the prediction with no
    successful measurement (verdict=refuse, ok=False). Stored for telemetry /
    future regression; EXCLUDED from the ratio (it never loaded)."""
    if not _calibration_enabled():
        return
    try:
        det = det or {}
        sample = {
            "model_key": model_key,
            "engine": _model_framework(model_key),
            "needs_weights_bytes": det.get("weights"),
            "needs_kv_bytes": det.get("kv"),
            "ctx_pct": det.get("ctx_pct"),
            "need_total_bytes": det.get("base_total") or det.get("total"),
            "verdict": "refuse",
            "ok": False,
            "ts": time.time(),
        }
        sample = {k: v for k, v in sample.items() if v is not None}
        with _CALIB_LOCK:
            _CALIB_BUFFER.append(sample)
            if len(_CALIB_BUFFER) > _CALIB_MAXLEN:
                del _CALIB_BUFFER[:-_CALIB_MAXLEN]
    except Exception:  # noqa: BLE001 — telemetry must never break admission
        pass


def _drain_calibration_samples() -> list:
    """Snapshot + clear the pending sample buffer for the heartbeat payload.
    Best-effort telemetry: a rare loss on a failed beat is acceptable (the next
    residency re-samples)."""
    with _CALIB_LOCK:
        if not _CALIB_BUFFER:
            return []
        out = list(_CALIB_BUFFER)
        _CALIB_BUFFER.clear()
    return out


def _adopt_calibration(worker: "dict | None") -> None:
    """Adopt central's published per-model corrections from the heartbeat reply
    (``worker['calibration'] = {mk: {"correction", ...}}``). Central only sends
    gate-passing, clamped values; an older/off central simply omits the key ->
    the map clears -> the static x1.15 stands. Mirrors _apply_central_limits."""
    corr = (worker or {}).get("calibration") or {}
    parsed: dict = {}
    if isinstance(corr, dict):
        for mk, info in corr.items():
            try:
                val = info.get("correction") if isinstance(info, dict) else info
                if val is not None:
                    parsed[str(mk)] = float(val)
            except (TypeError, ValueError):
                continue
    with _CALIB_LOCK:
        _CALIB_CORRECTIONS.clear()
        _CALIB_CORRECTIONS.update(parsed)


def _worker_fit_check(model_key: str) -> bool:
    """Contention fit-guard (dispatch.set_fit_check). True when the incoming load
    fits in current headroom WITHOUT yielding a resident; False = memory pressure
    -> yield the LRU on-demand resident.

    GPU box: the newcomer wants to be GPU-resident (hot), so it fits when free
    VRAM holds its weights. If a GPU is present but VRAM can't, that's the
    contention that yields an idle on-demand resident to keep the newcomer on the
    GPU (doctrine: minimize load time, keep models hot); when nothing is left to
    yield the loop stops and the normal autofit path spills to CPU exactly as
    today. CPU-only box: contention is on RAM. Fails OPEN when the size or both
    pools are unmeasurable — an unmeasurable load proceeds exactly as today.

    NEED = weights + KV(resolved ctx) (slice 11): the ctx tax is planned, not
    discovered at OOM. _incoming_need_detail is the ONE authoritative need; kv is
    0 when ctx_pct is unset (byte-identical to today)."""
    need = _incoming_need_detail(model_key).get("total")
    if not need:
        return True
    fv = _free_vram_bytes()
    if fv is not None:
        return fv >= need
    fr = _free_ram_bytes()
    if fr is not None:
        return fr >= need
    return True


def _vram_ceiling_frac() -> float:
    """The real-VRAM ceiling as a fraction of total card VRAM
    (HUGPY_VRAM_CEILING_FRAC, default 0.90). A load may proceed only if it leaves
    the card at/under this fraction full — equivalently, at least (1 - frac) of
    total VRAM free after the weights land. Clamped to a sane (0, 1] so a
    fat-fingered env can never invert the gate."""
    raw = os.environ.get("HUGPY_VRAM_CEILING_FRAC")
    if raw is None or not str(raw).strip():
        return 0.90
    try:
        val = float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric HUGPY_VRAM_CEILING_FRAC=%r; using 0.90",
                       raw)
        return 0.90
    if not (0.0 < val <= 1.0):
        logger.warning("HUGPY_VRAM_CEILING_FRAC=%r out of (0,1]; using 0.90", raw)
        return 0.90
    return val


def _total_vram_bytes() -> "int | None":
    """Total INSTALLED VRAM (spill.total_vram_bytes) — RAW device capacity, or
    None when no GPU / can't measure. Mirrors _free_vram_bytes' import guard so a
    missing torch/nvidia-smi degrades to None, never raises."""
    try:
        from ..managers.spill import total_vram_bytes
        return total_vram_bytes()
    except Exception:  # noqa: BLE001
        return None


def _worker_slot_fit_check(model_key: str) -> bool:
    """Real-VRAM CEILING gate (slots.set_fit_check), Fix A (2026-07-15). True when
    loading ``model_key`` would leave the card at/under the ~90% ceiling given
    REAL current free VRAM — i.e. at least (1 - ceiling) of total VRAM remains
    free AFTER the weights land. False when it would breach the ceiling (the slot
    scheduler then evicts the coldest on-demand occupant(s) and re-checks).

    This is distinct from _worker_fit_check (the in-process contention guard,
    which asks "does it fit WITHOUT yielding a resident"): this gate answers "does
    the WHOLE card stay under the ceiling", so it reacts to OUT-OF-BAND process
    VRAM growth (ComfyUI) that managed-model bookkeeping is blind to — free VRAM
    is the real device read (torch.cuda.mem_get_info, ComfyUI-visible).

    Fails OPEN (True) when free VRAM, total VRAM, or the incoming need is unknown
    (no GPU / can't tell) — NEVER block a load because we couldn't measure. That
    keeps a no-GPU / unmeasurable box byte-identical to today (the gate is a
    no-op there)."""
    total = _total_vram_bytes()
    if not total:
        return True                          # no GPU / can't measure -> allow
    fv = _free_vram_bytes()
    if fv is None:
        return True                          # can't read free VRAM -> allow
    # NEED = weights + KV(resolved ctx) (slice 11) — the ONE authoritative need,
    # so the ceiling gate reserves the ctx tax too. kv=0 when ctx_pct unset.
    need = _incoming_need_detail(model_key).get("total")
    if not need:
        return True                          # unknown weight size -> allow
    headroom = int(total * (1.0 - _vram_ceiling_frac()))
    # Loading consumes ~need; the card is OK if free-after-load still leaves the
    # (1 - ceiling) reserve. Equivalent to "post-load fill <= ceiling".
    return (fv - need) >= headroom


def _worker_evictable(model_key: str) -> bool:
    """Contention yield predicate (dispatch.set_evictable). A model may yield its
    in-process residency ONLY if it is not static, has NO in-flight generation
    (gate permits), and isn't slot-backed (a slot child's weights live in another
    process — dropping the proxy frees nothing here and breaks the seat).

    Tier semantics (operator, 2026-07-15): 📌 pin = the worker is DESIGNATED that
    model (durable assignment across restarts), NOT a resource lock — so a pinned
    model DOES yield to contention (its weights free for a new load; the pin is
    untouched and it reloads on demand). 🔒 static is the only residency lock
    ("static means cannot evict") and never yields. A model mid-generation is
    skipped (the next LRU is chosen) and becomes evictable once its gate permits.
    (Pre-2026-07-15 pinned also never yielded — that conflated designation with a
    resource lock, so pin-bloat could deadlock the make-room evictor.)"""
    if _residency(model_key) == "static":
        return False
    try:
        if gen_gate.in_flight(model_key) > 0:
            return False
    except Exception:  # noqa: BLE001 — can't tell -> don't yield a possibly-busy model
        return False
    try:
        from ..managers.llama.runners.get import slot_backed_model_keys
        if model_key in (slot_backed_model_keys() or set()):
            return False
    except Exception:  # noqa: BLE001 — can't tell slot-backing -> allow (in-process default)
        pass
    return True


# ── targeted eviction (evict <model_key>) ───────────────────────────────────
# Central signals `evict <model_key>` (never a raw PID — PIDs are per-box and get
# recycled). The worker resolves the model_key to its LIVE hosting handle AT
# eviction time, verifies identity, and frees it with the mechanism that matches
# HOW the model is hosted. This is the surgical bookend to /models/unload (which
# is coarse: one model_key or all) — same "stays ASSIGNED, just not resident"
# semantics, but it picks slot-kill vs in-process-drop vs comfy-free per model.

def _evict_gate(model_key: str) -> "tuple[bool, str]":
    """Eviction permission for the destructive evict verb: (allowed, reason).

    Tier semantics RE-CLARIFIED by the operator 2026-07-15: 📌 pin means ONLY
    that this worker is DESIGNATED to serve the model (a durable assignment that
    survives hard restarts) — it is NOT a resource lock and MUST NOT block
    eviction. 🔒 static is the ONLY residency lock ("static means cannot evict").
    So a pinned-but-on-demand model evicts freely: its weights are freed and it
    reloads on the next call, while the pin (designation) is untouched. This is
    why a fully-pinned worker is harmless — designation, not a VRAM hoard.

    Only static (and an in-flight generation) protects here. Slot-backing is NOT
    a blocker — evicting a slot child is the whole point of this verb. ``force``
    (checked by the caller) overrides every clause. A model mid-generation is
    protected unless forced: we never rip weights out from under a running
    request. (Pre-2026-07-15 this also refused pinned models — that conflated
    designation with a resource lock and jammed eviction under pin-bloat.)"""
    if _residency(model_key) == "static":
        return False, "static (locked residency) — pass force to override"
    try:
        if gen_gate.in_flight(model_key) > 0:
            return False, "in-flight generation — pass force to override"
    except Exception:  # noqa: BLE001 — can't tell -> treat as busy (don't rip a maybe-busy model)
        return False, "cannot determine in-flight state — pass force to override"
    return True, ""


def _resolve_slot_handle(model_key: str) -> "dict | None":
    """Resolve model_key -> the slot HANDLE currently serving it, or None.

    Returns {"control_url", "child_pid", "endpoint"} from a LIVE slot-pool
    status read (never a cached/central-supplied value). ``child_pid`` is the
    llama-server/llama_cpp.server child that actually holds the VRAM. Returns
    None when no slot is serving this model_key right now."""
    try:
        from ..managers.serve.slots import SlotPool
        for s in SlotPool().statuses():
            if s.get("model_key") == model_key and s.get("child_pid"):
                return {"control_url": s.get("_control"),
                        "child_pid": s.get("child_pid"),
                        "endpoint": s.get("endpoint")}
    except Exception:  # noqa: BLE001 — no slots / pool error -> not slot-hosted here
        return None
    return None


def _is_inprocess_resident(model_key: str) -> bool:
    """True if this worker holds the model's WEIGHTS in its OWN python process —
    a GGUF llama handle, a dispatch-cached torch runner, a diffusers pipeline, or
    a torch model nvidia-smi attributes to our PID. A slot-backed HTTP proxy
    (base_url, no ``llm``) is NOT a resident (its weights live in the child), so
    the slot branch must be resolved BEFORE this is consulted."""
    # GGUF heavy singletons with a real in-process llm handle.
    try:
        from ..managers.llama.runners.get import _LLAMA_INSTANCES, _LLAMA_LOCK
        with _LLAMA_LOCK:
            for k, r in list(_LLAMA_INSTANCES.items()):
                if k == model_key and getattr(r, "llm", None) is not None:
                    return True
    except Exception:  # noqa: BLE001
        pass
    # dispatch-cached in-process runners (torch/vision/etc.), excluding slot proxies.
    try:
        from ..managers.dispatch import dispatch as _d
        from ..managers.llama.runners.get import slot_backed_model_keys
        slot_keys = slot_backed_model_keys() or set()
        with _d._INSTANCES_LOCK:
            keys = [k[0] if isinstance(k, tuple) and k else k
                    for k in list(_d._INSTANCES)]
        if model_key in keys and model_key not in slot_keys:
            return True
    except Exception:  # noqa: BLE001
        pass
    # diffusers imagegen pipelines (class-level singleton, not on a runner attr).
    try:
        from ..managers.imagegen import imagegen_runner as _ig
        for clsname in ("ImageGenRunner", "Img2ImgRunner"):
            cache = getattr(getattr(_ig, clsname, None), "_PIPELINES", None)
            if isinstance(cache, dict) and model_key in cache:
                return True
    except Exception:  # noqa: BLE001
        pass
    # last resort: torch attributes real VRAM to this model under our PID.
    try:
        if model_key in _inprocess_gpu_bytes():
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _drop_inprocess_model(model_key: str) -> bool:
    """Drop the in-process refs for ``model_key`` and free its weights WITHOUT
    killing the worker PID (siblings share it). dispatch.evict cascades through
    the dispatch adapter cache AND the GGUF heavy singleton; the diffusers
    pipeline lives in a class-level cache that cascade misses, so drop it too.
    _trim_host_ram() then hands the freed arena + torch CUDA cache back."""
    dropped = False
    try:
        from ..managers.dispatch import evict as _evict
        dropped = bool(_evict(model_key)) or dropped
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..managers.imagegen import imagegen_runner as _ig
        for clsname in ("ImageGenRunner", "Img2ImgRunner"):
            cache = getattr(getattr(_ig, clsname, None), "_PIPELINES", None)
            if isinstance(cache, dict) and cache.pop(model_key, None) is not None:
                dropped = True
    except Exception:  # noqa: BLE001
        pass
    _trim_host_ram()
    return dropped


def _comfy_free_models(state: "WorkerState") -> "tuple[bool, str]":
    """Ask the ADOPTED external ComfyUI to release its resident models via its OWN
    HTTP API — never a PID kill (the worker doesn't own comfy's process). ComfyUI
    exposes ``POST /free`` with ``{"unload_models": true, "free_memory": true}``;
    it unloads comfy's currently-loaded checkpoint(s) and hands VRAM back while
    the server stays up for the next job. Returns (freed_ok, note). Degrades
    gracefully (freed_ok=False + reason) when comfy is unreachable / lacks /free."""
    url = _comfy_base_url(state)
    try:
        import httpx
        r = httpx.post(url + "/free",
                       json={"unload_models": True, "free_memory": True},
                       timeout=30.0)
        if r.status_code == 200:
            return True, "comfy /free accepted (unload_models + free_memory)"
        return False, f"comfy /free returned HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001 — comfy down / no /free: degrade, never 500
        return False, f"comfy unreachable at {url}: {type(exc).__name__}: {exc}"


def _comfy_base_url(state: "WorkerState") -> str:
    """The adopted ComfyUI base URL: the operator/comfy_url setting projects onto
    COMFY_URL (see _apply_settings_env); default 127.0.0.1:8188 matches
    managers/comfy/comfy_runner._comfy_url()."""
    return (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


def _evict_model(state: "WorkerState", model_key: str,
                 force: bool = False) -> dict:
    """Resolve ``model_key`` to its LIVE hosting handle and free it with the
    mechanism that matches how it is hosted. Fail-safe: unknown/not-resident is
    an idempotent no-op, never an error. Returns the /ops/evict contract dict.

    Resolution order (comfy first, because a comfy checkpoint is served by an
    EXTERNAL process and never appears in our slot/in-process caches; slot before
    in-process, because a slot-backed model also leaves a thin HTTP proxy in the
    in-process cache that holds no weights):
      1. comfy  — framework == 'comfy'      -> comfy's own /free API
      2. slot   — a live slot serves it     -> verify identity, then slot /unload
                                               (owner does SIGTERM->wait->SIGKILL)
      3. in-process — weights in our PID     -> drop refs + torch empty_cache + trim
      4. not resident                        -> idempotent no-op
    """
    if not isinstance(model_key, str) or not model_key.strip():
        return {"model_key": model_key, "host_mode": "unknown", "evicted": False,
                "vram_freed": None, "ram_freed": None,
                "reason": "missing model_key"}
    model_key = model_key.strip()
    # t21: an evicted model's ctx flex commitment is void — a fresh load
    # re-decides from target. Drop it so _ctx_pct doesn't serve a stale floor.
    _FLEX_CTX_FLOOR.pop(model_key, None)

    vram_before = _free_vram_bytes()
    ram_before = _free_ram_bytes()

    def _result(host_mode, evicted, reason, **extra):
        vram_after = _free_vram_bytes()
        ram_after = _free_ram_bytes()
        vram_freed = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None) else None
        ram_freed = (ram_after - ram_before) if (
            ram_before is not None and ram_after is not None) else None
        out = {"model_key": model_key, "host_mode": host_mode,
               "evicted": bool(evicted), "reason": reason,
               "vram_freed": vram_freed, "ram_freed": ram_freed,
               "vram_free_before": vram_before, "vram_free_after": vram_after,
               "ram_free_before": ram_before, "ram_free_after": ram_after,
               "forced": bool(force), "loaded_models": loaded_model_keys()}
        out.update(extra)
        return out

    # 1. ComfyUI-hosted (external adopted service) — framework says comfy. The
    #    worker never owns comfy's PID; it asks comfy to free via HTTP. The gate
    #    still applies best-effort (a comfy gen in flight is protected unless
    #    forced), but comfy's /free is coarse (releases comfy's resident set).
    if _model_framework(model_key) == "comfy":
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if not allowed:
            return _result("comfy", False, f"eviction gated: {why}")
        freed_ok, note = _comfy_free_models(state)
        return _result("comfy", freed_ok, note)

    # 2. Subprocess-hosted (slot child / worker-spawned llama-server). Resolve the
    #    model_key -> its CURRENT slot handle from a LIVE status read.
    handle = _resolve_slot_handle(model_key)
    if handle is not None:
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if not allowed:
            return _result("slot", False, f"eviction gated: {why}",
                           child_pid=handle.get("child_pid"))
        # RECYCLED-PID GUARD: re-read the slot status right before acting and
        # confirm it STILL maps this model_key to the SAME child_pid we resolved.
        # A slot that has since swapped to another model (or respawned its child
        # under a new pid) must NOT be evicted — that would kill the wrong model.
        pid = handle.get("child_pid")
        control = handle.get("control_url")
        recheck = _resolve_slot_handle(model_key)
        if recheck is None or recheck.get("child_pid") != pid \
                or recheck.get("control_url") != control:
            return _result("slot", False,
                           "slot handle changed before evict (recycled/swapped) "
                           "— not evicted", child_pid=pid)
        # Free via the slot's OWN /unload: the slot supervisor owns the child, so
        # it performs the SIGTERM -> short wait -> SIGKILL itself (Slot._kill:
        # terminate, wait 15s, kill) and clears its own model_key claim atomically
        # — cleaner and safer than the agent os.kill-ing another supervisor's
        # child on a possibly-recycled pid. CUDA context drops on child exit; a
        # host-RAM trim follows to hand the freed arena back.
        err = None
        try:
            from ..managers.serve.slots import SlotPool
            SlotPool().unload(control)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        _trim_host_ram()
        if err is not None:
            return _result("slot", False, f"slot unload failed: {err}",
                           child_pid=pid)
        return _result("slot", True,
                       f"slot child pid={pid} terminated (SIGTERM->SIGKILL) via "
                       "its supervisor", child_pid=pid)

    # 3. In-process torch/GGUF model sharing THIS worker's python PID. Never kill
    #    the PID (that kills the worker + every sibling model) — drop the refs.
    if _is_inprocess_resident(model_key):
        allowed, why = (True, "") if force else _evict_gate(model_key)
        if not allowed:
            return _result("in_process", False, f"eviction gated: {why}")
        dropped = _drop_inprocess_model(model_key)
        return _result("in_process", dropped,
                       "in-process refs dropped + CUDA cache/host arena trimmed"
                       if dropped else "in-process handle already gone")

    # 4. Nothing here holds it. This ALSO covers the foreign/rogue case: a model
    #    that resolves only to a process the agent did not spawn (and isn't comfy)
    #    is OUT OF SCOPE for this slice — we never os.kill an arbitrary PID, so
    #    such a model simply reads as not-resident here. Idempotent no-op, HTTP 200.
    return _result("none", False, "not resident on this worker")


# ── GPU orphan reaper (p27, 2026-07-23) ─────────────────────────────────────
# THE k30 GAP this closes: an orphaned llama-server child — its slot claim
# cleared (slot swapped/respawned) but the child never exited — keeps holding
# VRAM. Since c34199e it is ENUMERABLE (the reconcile second pass tags it
# ``cuda_context`` with model_key None) but UNEVICTABLE: every eviction verb
# keys on model_key, and an orphan has none. This verb kills by PID — the ONLY
# place in the agent that does — so its admission gates are deliberately
# fail-closed and narrow.

# Minimum process age before a pid may be reaped. Closes the mid-spawn race: a
# fresh slot child exists (and holds VRAM) for a beat BEFORE its slot claim
# registers — without this grace a reap racing a spawn would kill a legitimate
# newborn. Read defensively so a malformed env value can never break import.
try:
    _ORPHAN_MIN_AGE_S = max(
        0.0, float(os.environ.get("HUGPY_ORPHAN_MIN_AGE_S", "300") or "300"))
except (TypeError, ValueError):
    _ORPHAN_MIN_AGE_S = 300.0

# How long a SIGTERM'd orphan gets to exit before SIGKILL — same discipline as
# the slot supervisor's own child kill (terminate -> wait -> kill).
_ORPHAN_TERM_WAIT_S = 10.0


def _clock_ticks_per_s() -> float:
    """Kernel clock ticks per second (for /proc starttime -> seconds). 100 on
    every mainstream Linux; read via os.sysconf when available."""
    try:
        return float(os.sysconf("SC_CLK_TCK")) or 100.0
    except (AttributeError, ValueError, OSError):
        return 100.0


def _proc_age_s(pid: int) -> "float | None":
    """Age of process ``pid`` in seconds, or None when it cannot be measured
    (gone / non-Linux / unreadable) — callers treat None as UNVERIFIABLE and
    fail closed. Primary source: /proc/<pid>/stat starttime (ticks since boot)
    against /proc/uptime — the same per-lifetime anchor the recycled-PID guard
    uses, so age and identity come from one number. Fallback: mtime of the
    /proc/<pid> directory (set at process creation)."""
    try:
        from .pid_registry import _default_proc_info
        info = _default_proc_info(int(pid))
    except Exception:  # noqa: BLE001 — probe failure = unverifiable
        info = None
    if info is not None and info.get("starttime") is not None:
        try:
            with open("/proc/uptime", "r") as fh:
                uptime_s = float(fh.read().split()[0])
            started_s = float(info["starttime"]) / _clock_ticks_per_s()
            age = uptime_s - started_s
            if age >= 0:
                return age
        except (OSError, ValueError, IndexError):
            pass
    try:
        st = os.stat("/proc/%d" % int(pid))
        age = time.time() - st.st_mtime
        return age if age >= 0 else None
    except (OSError, ValueError):
        return None


def _reap_own_pids() -> set:
    """The agent's own pid + its direct infra children (slot SUPERVISORS it
    spawned) — never reap targets, by construction."""
    own = {os.getpid()}
    try:
        for p in list(_SLOT_PROCS.values()):
            if p is not None and p.poll() is None and p.pid:
                own.add(int(p.pid))
    except Exception:  # noqa: BLE001 — best-effort; os.getpid() always guards
        pass
    return own


def _self_venv_marker() -> "str | None":
    """The SAME own-venv marker the heartbeat passes pid_registry.reconcile
    (this python's venv root, e.g. ``/opt/hugpy/venv``): a GPU process whose
    name/cmdline contains it runs OUR interpreter/binaries. None when it cannot
    be derived — callers fail closed (nothing is reapable without it)."""
    try:
        marker = os.path.dirname(os.path.dirname(sys.executable)) or None
    except Exception:  # noqa: BLE001
        return None
    # A degenerate marker ("", "/", ".") would match EVERY process name —
    # substring matching makes that a kill-anything wildcard. Refuse it.
    if not marker or marker in ("/", "."):
        return None
    return marker


def _reap_gpu_orphans(state: "WorkerState", dry_run: bool = True) -> dict:
    """Enumerate and (unless ``dry_run``) kill ORPHANED GPU children this worker
    itself leaked: processes from OUR OWN venv that hold VRAM but that no live
    slot claims. Exposed as POST /ops/reap-orphans.

    DOCTRINE — OPERATOR/CENTRAL-INVOKED ONLY. This verb must NEVER be called
    from any loop, heartbeat, sweep, or timer. It is the one place the agent
    kills by raw PID; automation of it is explicitly unsanctioned (operator
    ruling, p27 2026-07-23). Wire it to a button/relay, never a schedule.

    A pid is reapable ONLY when ALL FOUR gates hold — any gate UNVERIFIABLE
    means NOT reapable (fail-closed):

      1. OWN-VENV   — its nvidia-smi process_name or /proc cmdline contains this
                      worker's own venv marker (same marker source the pid
                      registry's cuda_context second pass uses). Comfy
                      (_COMFY_NAME_MARKER in the name) and anything foreign fail
                      by construction; the agent's own pid and its direct infra
                      pids (slot supervisors) are excluded outright.
      2. NO CLAIM   — no current slot status references the pid as child_pid.
                      Slot statuses UNREADABLE -> nothing is reapable this pass.
      3. HOLDS GPU  — the pid appears in the current nvidia-smi snapshot with
                      mib > 0 (a CPU-only stray is not this verb's business).
      4. MIN AGE    — the process is older than HUGPY_ORPHAN_MIN_AGE_S (default
                      300s), closing the mid-spawn race where a slot child
                      exists before its claim registers.

    Kill discipline: SIGTERM -> wait up to 10s -> SIGKILL, with a recycled-PID
    identity re-check (starttime) immediately before each signal. Per-pid result
    rows: {pid, name, vram_bytes, action: reaped|term_failed|skipped, reason}.
    ``dry_run`` (the DEFAULT) only reports what would be reaped."""
    gpu_procs = _gpu_process_vram() or {}
    marker = _self_venv_marker()
    own = _reap_own_pids()
    slots = _slot_statuses()
    claimed: "set | None" = None
    if slots is not None:
        claimed = {s.get("child_pid") for s in slots
                   if isinstance(s, dict) and s.get("child_pid") is not None}
    try:
        from .pid_registry import _COMFY_NAME_MARKER as _comfy_marker
        from .pid_registry import _default_proc_info as _proc_info
    except Exception:  # noqa: BLE001 — no registry module -> nothing verifiable
        _comfy_marker, _proc_info = "comfyui", (lambda _pid: None)

    results: list = []
    reapable_bytes = 0
    for pid, meta in sorted(gpu_procs.items()):
        name = str((meta or {}).get("name") or "")
        mib = int((meta or {}).get("mib") or 0)
        vram_bytes = mib * _MIB

        def _skip(reason: str) -> None:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "skipped", "reason": reason})

        # Gate 3 first (cheap): must actually hold GPU memory.
        if mib <= 0:
            _skip("holds no VRAM (mib<=0)")
            continue
        # Own-pid / infra exclusion — before anything else.
        if pid in own:
            _skip("agent's own pid / direct infra pid — never reapable")
            continue
        # Comfy fails own-venv BY CONSTRUCTION (external adopted service).
        if _comfy_marker in name.lower():
            _skip("comfy process — never reapable (external adopted service)")
            continue
        # Gate 1: OWN-VENV. Marker underivable -> nothing reapable (fail-closed).
        if marker is None:
            _skip("own-venv marker unavailable — cannot prove ownership "
                  "(fail-closed)")
            continue
        info = _proc_info(pid)
        cmdline = str((info or {}).get("cmdline") or "")
        if marker not in name and marker not in cmdline:
            _skip("not from this worker's venv — foreign process, out of scope")
            continue
        # Gate 2: NO LIVE CLAIM. Unreadable slot pool -> unverifiable -> skip.
        if claimed is None:
            _skip("slot statuses unreadable — cannot prove no live claim "
                  "(fail-closed)")
            continue
        if pid in claimed:
            _skip("live slot claims this pid as child_pid — not an orphan")
            continue
        # Gate 4: MIN AGE. Unmeasurable age -> unverifiable -> skip.
        age = _proc_age_s(pid)
        if age is None:
            _skip("process age unmeasurable — cannot rule out mid-spawn race "
                  "(fail-closed)")
            continue
        if age < _ORPHAN_MIN_AGE_S:
            _skip(f"process too young ({age:.0f}s < min age "
                  f"{_ORPHAN_MIN_AGE_S:.0f}s) — mid-spawn race protection")
            continue

        # ALL FOUR GATES HOLD — this pid is a reapable orphan.
        reapable_bytes += vram_bytes
        if dry_run:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "skipped",
                            "reason": "dry_run — would be reaped "
                                      "(all four gates hold)"})
            logger.info(
                "reap-orphans DRY RUN: pid %s (%s, %s) is a reapable orphan "
                "(own-venv, unclaimed, holds GPU, age %.0fs)",
                pid, name, _human_bytes(vram_bytes), age)
            continue

        # Recycled-PID identity anchor: capture starttime NOW; re-verify before
        # each signal so a pid recycled mid-reap is never signalled.
        anchor = (info or {}).get("starttime")

        def _still_same() -> bool:
            cur = _proc_info(pid)
            if cur is None:
                return False                        # gone — nothing to signal
            if anchor is not None and cur.get("starttime") is not None:
                return int(cur["starttime"]) == int(anchor)
            # No starttime anchor available: corroborate via cmdline instead;
            # unverifiable identity -> do NOT signal.
            return bool(cmdline) and cur.get("cmdline") == cmdline

        logger.warning(
            "reap-orphans: KILLING orphaned GPU child pid=%s name=%r vram=%s "
            "age=%.0fs (own-venv match %r, no live slot claim) — SIGTERM",
            pid, name, _human_bytes(vram_bytes), age, marker)
        import signal as _signal
        try:
            if not _still_same():
                results.append({"pid": pid, "name": name,
                                "vram_bytes": vram_bytes, "action": "skipped",
                                "reason": "pid identity changed before SIGTERM "
                                          "(recycled/exited) — not signalled"})
                continue
            os.kill(pid, _signal.SIGTERM)
        except ProcessLookupError:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "reaped",
                            "reason": "already exited at SIGTERM"})
            continue
        except OSError as exc:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "term_failed",
                            "reason": f"SIGTERM failed: {type(exc).__name__}: {exc}"})
            continue
        # Wait up to _ORPHAN_TERM_WAIT_S for a clean exit.
        deadline = time.time() + _ORPHAN_TERM_WAIT_S
        exited = False
        while time.time() < deadline:
            if _proc_info(pid) is None or not _still_same():
                exited = True
                break
            time.sleep(0.25)
        if exited:
            logger.warning("reap-orphans: pid %s exited on SIGTERM", pid)
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "reaped", "reason": "SIGTERM honored"})
            continue
        logger.warning("reap-orphans: pid %s survived SIGTERM %.0fs — SIGKILL",
                       pid, _ORPHAN_TERM_WAIT_S)
        try:
            if _still_same():
                os.kill(pid, _signal.SIGKILL)
                results.append({"pid": pid, "name": name,
                                "vram_bytes": vram_bytes, "action": "reaped",
                                "reason": "SIGKILL after SIGTERM timeout"})
            else:
                results.append({"pid": pid, "name": name,
                                "vram_bytes": vram_bytes, "action": "reaped",
                                "reason": "exited between SIGTERM wait and SIGKILL"})
        except ProcessLookupError:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "reaped",
                            "reason": "exited before SIGKILL"})
        except OSError as exc:
            results.append({"pid": pid, "name": name, "vram_bytes": vram_bytes,
                            "action": "term_failed",
                            "reason": f"SIGKILL failed: {type(exc).__name__}: {exc}"})

    reaped = [r for r in results if r["action"] == "reaped"]
    failed = [r for r in results if r["action"] == "term_failed"]
    if not dry_run and (reaped or failed):
        _trim_host_ram()                # hand back the freed arena, same as evict
    return {
        "dry_run": bool(dry_run),
        "results": results,
        "reaped_count": len(reaped),
        "term_failed_count": len(failed),
        "skipped_count": len(results) - len(reaped) - len(failed),
        "reapable_vram_bytes": reapable_bytes,
        "min_age_s": _ORPHAN_MIN_AGE_S,
    }


# ── VRAM evict-to-fit at admission (slice 10, the VRAM twin of disk evict-to-fit) ─
# The operator's ruling (2026-07-17): "everything is on demand — the process not
# actively replying and not ahead of the subject in the queue, as well as not
# 'static', should be evicted to allow the subject process to proliferate."
#
# THE INCIDENT: a transformers load OOM'd because an IDLE 21.3G coder SLOT CHILD
# squatted the card and NOTHING evicted it first. The in-process contention path
# (dispatch.ensure_headroom_for_load) only ever saw _INSTANCES residents and its
# _worker_evictable predicate REFUSED slot-backed models — so a subprocess
# squatter was invisible to an in-process load's make-room. This choke point sees
# ALL residents (in-process + slot child + comfy) from the pid-registry MEASURED
# truth, applies the protection rules, evicts the minimum LRU set through the SAME
# _evict_model verb the operator proved live via /ops/evict, re-checks, and
# refuses HONESTLY before any CUDA allocation (never admit-then-OOM).

# VRAM eviction counter — the churn the operator watches must now include VRAM
# evictions, not just disk. Surfaced on the heartbeat (see _worker_storage /
# the beat body). A simple monotonic count + the last event, cheap and honest.
_VRAM_EVICTIONS: dict = {"count": 0, "last": None, "last_at": 0.0}


def _note_vram_eviction(victim: str, subject: str, freed: "int | None",
                        host_mode: str) -> None:
    _VRAM_EVICTIONS["count"] += 1
    _VRAM_EVICTIONS["last"] = {
        "victim": victim, "subject": subject, "host_mode": host_mode,
        "vram_freed": freed, "at": time.time()}
    _VRAM_EVICTIONS["last_at"] = time.time()
    logger.info("VRAM evict-to-fit: evicted %s (%s, freed %s) to make room for %s",
                victim, host_mode, _human_bytes(freed), subject)


def _human_bytes(n: "int | None") -> str:
    if not n:
        return "0 B"
    v = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or u == "TB":
            return f"{v:.1f} {u}"
        v /= 1024
    return f"{n} B"


def _need_split_str(det: dict) -> str:
    """The honest weights+kv breakdown for a refusal (slice 11), e.g.
    ' = 21.3 GB weights + 2.8 GB kv@50%ctx'. Empty when there is no ctx (kv=0),
    so a model with no ctx allocation reads exactly as today."""
    kv = int(det.get("kv") or 0)
    if kv <= 0:
        return ""
    pct = det.get("ctx_pct")
    tag = f"@{pct}%ctx" if pct else ""
    return (f" = {_human_bytes(det.get('weights'))} weights + "
            f"{_human_bytes(kv)} kv{tag}")


def _actively_replying(model_key: str, slot_busy: "set | None" = None) -> bool:
    """MEASURED 'actively replying' (operator: protect an in-flight reply), NOT
    inferred from residency. True when the model has an in-flight in-process
    generation (gen_gate) OR its slot is flagged busy this instant. Fail-safe: if
    we can't tell, treat as busy (never rip a possibly-replying model)."""
    try:
        if gen_gate.in_flight(model_key) > 0:
            return True
    except Exception:  # noqa: BLE001 — can't tell -> protect
        return True
    if slot_busy is not None:
        return model_key in slot_busy
    return False


def _busy_slot_models() -> set:
    """Model_keys whose slot is BUSY right now (a live request in the child) —
    the slot-side 'actively replying' signal. Empty on any read failure."""
    try:
        from ..managers.serve.slots import SlotPool
        return {s.get("model_key") for s in SlotPool().statuses()
                if s.get("model_key") and s.get("busy")}
    except Exception:  # noqa: BLE001
        return set()


def _queued_ahead_of(subject: str) -> set:
    """Model_keys with pending work queued AHEAD of the subject's request —
    protected for this pass (operator: 'not ahead of the subject in the queue').

    On-box the worker has no central queue; the honest local signal is the
    provision/warm queue plus any model with WAITING gen-gate entrants (a request
    parked on the gate is queued work targeting that resident). Best-effort — an
    empty set on any failure just means nothing is queue-protected this pass, and
    the in-flight guard still protects an actively-replying model."""
    ahead: set = set()
    try:
        # A model with more gate entrants than are in-flight has requests WAITING
        # on it — queued work the very next release will serve. Protect it.
        for mk, g in list(getattr(gen_gate, "_gates", {}).items()):
            try:
                if g.active() >= g.limit and mk != subject:
                    ahead.add(mk)          # gate saturated -> a waiter is queued
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return ahead


def _vram_residents(state: "WorkerState") -> "list[dict]":
    """Every GPU-resident model this box holds, from the pid-registry MEASURED
    snapshot (in_process + slot subprocess + comfy) UNIONED with the LIVE slot
    statuses, each with its real vram_bytes and host_mode. This is the resident
    TRUTH the eviction planner ranks — it includes the slot child ('max GPU'
    alloc and all: alloc is a sizing preference, not a residency shield) that
    the in-process contention path was blind to. Comfy rows are surfaced but
    EXCLUDED from eviction here (0.1.137: comfy is out of allocations; it has
    its own Fix B headroom path).

    THE k30 INVISIBILITY FIX (2026-07-23): the pid registry is per-process,
    in-memory state repopulated by the heartbeat loop. A slot occupant can be
    plainly visible in the allocations view (live slot status + nvidia-smi)
    while the registry has no record for it yet — a fresh re-exec before the
    first beat, a swept/mis-verified record, or a child whose pid the reconcile
    second pass tagged as an anonymous ``cuda_context`` lump (model_key=None,
    which this function used to skip). The evict planner then enumerated ZERO
    residents on an occupied card and refused with the self-contradictory
    "evicted 0 idle ... 0 protected still hold the card" — a de-facto
    protection class (invisibility) the operator never sanctioned. Fix: the
    planner enumerates the SAME collection the allocations view shows — the
    registry rows PLUS every live slot occupant (model_key set), joining the
    slot child's real VRAM from nvidia-smi when the registry attribution is
    missing. Only static / actively-replying / queued-ahead / comfy protect
    (operator ruling 2026-07-23)."""
    out: list[dict] = []
    seen: set = set()
    try:
        from . import pid_registry as _pidreg
        snap = _pidreg.snapshot_for_heartbeat() or {}
        for row in snap.get("models") or []:
            mk = row.get("model_key")
            if not mk:
                continue                    # cuda_context lump / idle comfy: no model
            seen.add(mk)
            out.append({
                "model_key": mk,
                "vram_bytes": int(row.get("vram_bytes") or 0),
                "host_mode": row.get("host_mode") or "",
                "alive": bool(row.get("alive", True)),
            })
    except Exception:  # noqa: BLE001 — no registry -> nothing to plan against
        pass
    # Union in LIVE slot occupants the registry doesn't know (k30). A slot with
    # a model_key claim is a resource allocation whether or not the registry has
    # caught up; its child holds the VRAM. Join nvidia-smi on child_pid for the
    # honest bytes (0 when unjoinable — candidacy is what matters; _fits() is
    # re-measured from the device after each eviction anyway).
    try:
        gpu_procs = None
        for s in (_slot_statuses() or []):
            mk = (s or {}).get("model_key")
            if not mk or mk in seen:
                continue
            if gpu_procs is None:
                gpu_procs = _gpu_process_vram() or {}
            cp = s.get("child_pid")
            info = gpu_procs.get(cp) if cp is not None else None
            vb = int(info["mib"]) * _MIB if info is not None else 0
            seen.add(mk)
            out.append({
                "model_key": mk,
                "vram_bytes": vb,
                "host_mode": "subprocess",
                "alive": bool(s.get("healthy", True)),
            })
    except Exception:  # noqa: BLE001 — slot pool unreadable -> registry rows stand
        pass
    return out


def _vram_evict_to_fit(state: "WorkerState", model_key: str,
                       need: "int | None" = None) -> dict:
    """THE VRAM admission choke point. Make room for ``model_key`` to land on the
    GPU under the ~90% ceiling by evicting the minimum LRU set of EVICTABLE
    residents, or refuse HONESTLY before any CUDA allocation.

    Protection (operator ruling): NEVER evict a 🔒static resident, a model that is
    ACTIVELY REPLYING (measured: in-flight gen / busy slot), a model with work
    QUEUED AHEAD of the subject, comfy (its own path), or the subject itself.
    EVERYTHING else is a candidate — on-demand idle residents, SLOT CHILDREN
    included ('max GPU' alloc included) — LRU/coldest-first, minimum set to fit.

    Returns a typed verdict:
      {"action": "proceed"|"evicted"|"refuse", "evicted": [mk...],
       "freed_bytes": int, "reason": {...}|None}
    Fails OPEN (proceed) whenever VRAM/need is unmeasurable — an unmeasurable load
    proceeds exactly as today, never blocked because we couldn't measure."""
    # t21: every admission re-decides from TARGET — drop any stale ctx flex
    # commitment for this subject so the fit check below is against the
    # operator's target ctx, not a prior compression (uncontended == target).
    # Same for any prior PARTIAL-offload commitment: a model that now fits fully
    # must serve fully (-1), never stay pinned to a stale layer count.
    _FLEX_CTX_FLOOR.pop(model_key, None)
    _clear_partial_ngl(model_key)
    total = _total_vram_bytes()
    if not total:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                "reason": None, "note": "no GPU / unmeasurable — gate is a no-op"}
    # NEED = weights + KV(resolved ctx) (slice 11): the ctx tax is reserved BEFORE
    # the load, and the split is carried for an honest refusal ("24.1G = 21.3G
    # weights + 2.8G kv@50%ctx"). When the caller passed an explicit `need` (a
    # test / a pre-computed total) it wins; otherwise the authoritative detail.
    _det = _incoming_need_detail(model_key)
    if need is None:
        need = _det.get("total")
    if not need:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                "reason": None, "note": "unknown weight size — fail open"}
    ceiling_reserve = int(total * (1.0 - _vram_ceiling_frac()))

    def _fits() -> "bool | None":
        fv = _free_vram_bytes()
        if fv is None:
            return None                      # can't read -> fail open at the caller
        return (fv - need) >= ceiling_reserve

    ok = _fits()
    if ok is None:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                "reason": None, "note": "can't read free VRAM — fail open"}
    if ok:
        return {"action": "proceed", "evicted": [], "freed_bytes": 0, "reason": None}

    # Over the ceiling — plan the minimum eviction set.
    busy_slots = _busy_slot_models()
    queued_ahead = _queued_ahead_of(model_key)
    try:
        from ..managers.dispatch.dispatch import last_used_snapshot as _lus
        lru = _lus()
    except Exception:  # noqa: BLE001
        lru = {}

    residents = _vram_residents(state)
    candidates: list[dict] = []
    protected: list[dict] = []
    for r in residents:
        mk = r["model_key"]
        if mk == model_key:
            continue
        if str(r.get("host_mode")) == "comfy":
            protected.append({**r, "why": "comfy (own headroom path; excluded "
                                          "from allocations — 0.1.137)"})
            continue
        if _residency(mk) == "static":
            protected.append({**r, "why": "static (locked residency)"})
            continue
        if _actively_replying(mk, busy_slots):
            protected.append({**r, "why": "actively replying (in-flight/busy)"})
            continue
        if mk in queued_ahead:
            protected.append({**r, "why": "queued ahead of the subject"})
            continue
        candidates.append(r)

    # ── t21 tolerance-band FLEX before evict (stage 1) ──────────────────────
    # Try to fit WITHIN bands before evicting anyone. ctx is the CHEAPEST flex,
    # so plan_flex (1) compresses the SUBJECT's own ctx toward its band floor,
    # then (2) — for a strictly higher-priority subject — reclaims resident KV
    # from lower-priority, UNPROTECTED neighbours within THEIR ctx bands.
    # Protection is absolute: only `candidates` (already protection-filtered
    # above) are offered as flex-eligible neighbours. The pure decision lives in
    # flex.plan_flex; here we EXECUTE the piece that is safe worker-side now —
    # the subject's own ctx compression, which lowers `need` so fewer (or no)
    # residents must be evicted. In-place neighbour KV-shrink rides worker
    # enforcement (next cut); until then a higher-priority subject's precedence
    # manifests as the priority-ordered eviction below.
    from .flex import plan_flex, flex_priority_key as _fpk, kv_at_ctx_pct as _kvat
    flex_note = None
    fv_now = _free_vram_bytes()
    if fv_now is not None:
        deficit = ceiling_reserve - (fv_now - need)      # >0 here (ok was False)
        subject = {"weights_bytes": _det.get("weights"), "kv_bytes": _det.get("kv"),
                   "ctx_pct": _det.get("ctx_pct"),
                   "ctx_deviation_pct": _ctx_deviation_pct(model_key),
                   "priority": _flex_priority(model_key)}
        resident_rows = []
        for r in candidates:                             # unprotected only
            mk = r["model_key"]
            try:
                rkv, rdet = _kv_need_bytes(mk)
            except Exception:  # noqa: BLE001 — a pricing gap must not break admission
                rkv, rdet = 0, {}
            resident_rows.append({
                "model_key": mk, "kv_bytes": int(rkv or 0),
                "ctx_pct": (rdet or {}).get("ctx_pct"),
                "ctx_deviation_pct": _ctx_deviation_pct(mk),
                "vram_bytes": int(r.get("vram_bytes") or 0),
                "protected": False, "pinned": bool(r.get("pinned")),
                "alloc": _flex_alloc(mk)})
        plan = plan_flex(subject, resident_rows, deficit)
        if plan.self_ctx_pct is not None and _det.get("kv"):
            # Commit the subject to its compressed ctx so the SERVED -c and the
            # KV admission reserved agree, and re-price `need` at that floor. The
            # captured _fits() closure re-reads `need`, so this shrinks the fit
            # target for both the flex re-check and the eviction loop.
            _FLEX_CTX_FLOOR[model_key] = int(plan.self_ctx_pct)
            new_kv = _kvat(_det.get("kv"), _det.get("ctx_pct"), plan.self_ctx_pct)
            need = int(_det.get("weights") or 0) + int(new_kv or 0)
        if plan.action == "flex" and _fits():
            # Fits WITHIN bands — no eviction. (Neighbour compression in the plan
            # is realised as reduced eviction / priority order until in-place
            # resident shrink lands; self-flex alone already cleared the ceiling.)
            return {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None, "note": f"flex: {plan.note}",
                    "flex": plan.as_dict()}
        flex_note = plan.note

    # ── stage 2: EVICT, lowest flex-priority first, then today's LRU order ───
    # Priority-ascending so a higher-priority subject yields lower-priority
    # neighbours BEFORE higher ones (the operator's "explicit priorities"); then
    # coldest-first (LRU), then largest-first among equal age. With no priorities
    # set (the default, priority==0 everywhere) this is byte-identical to the
    # prior pure-LRU order.
    candidates.sort(key=lambda r: (_fpk(_flex_alloc(r["model_key"])),
                                   lru.get(r["model_key"], 0.0),
                                   -int(r.get("vram_bytes") or 0)))

    evicted: list[str] = []
    evict_failed: list[dict] = []            # attempted but not freed — carried
    freed = 0                                # in the refusal so counts are TRUE
    for r in candidates:
        chk = _fits()
        if chk:
            break
        mk = r["model_key"]
        res = _evict_model(state, mk)        # the SAME verb /ops/evict uses
        if res.get("evicted"):
            fb = res.get("vram_freed")
            freed += int(fb) if fb else 0
            evicted.append(mk)
            _note_vram_eviction(mk, model_key, fb, res.get("host_mode") or "")
            _trim_host_ram()                 # so the next _fits() sees the room
        else:
            # An eviction that resolved to a no-op ("not resident here", a
            # changed slot handle, a failed unload) must not vanish from the
            # story — the old message counted only successes, so a card held by
            # an unevictable-in-practice resident read "evicted 0 ... 0
            # protected", contradicting the visible occupancy (k30).
            evict_failed.append({"model_key": mk,
                                 "host_mode": res.get("host_mode"),
                                 "reason": res.get("reason")})
            logger.warning("VRAM evict-to-fit: eviction of %s did not free it "
                           "(%s: %s)", mk, res.get("host_mode"), res.get("reason"))

    final = _fits()
    if final:
        return {"action": "evicted", "evicted": evicted,
                "freed_bytes": freed, "reason": None}

    fv = _free_vram_bytes()

    # ── stage (2.5): honest GGUF PARTIAL offload — autofit's hybrid contract ──
    # Full GPU offload still doesn't fit after flex + evict. For a GGUF this is
    # NOT a dead end: autofit's PROMISE (empty spill = the default alloc mode) is
    # a hybrid — offload as many layers as safely fit under the ceiling reserve,
    # stream the rest from disk to CPU RAM. This is the regression this restores:
    # a served-many-times brain must not hard-refuse on a card that plainly holds
    # part of it. Priced from the honest, SHARD-AWARE need split (not the shard-
    # blind on-disk autofit), floored against a degenerate offload and against a
    # CPU remainder that would OOM host RAM (never admit-then-OOM). GGUF/slot path
    # only; transformers placement modes are t26 (out of scope).
    partial = None
    ppath, total_layers = _served_gguf_geometry(model_key)
    if fv is not None and total_layers:
        weights = int(_det.get("weights") or 0)
        kv_eff = max(0, int(need) - weights)     # honors any committed ctx flex
        budget = max(0, fv - ceiling_reserve)    # VRAM the offloaded layers may use
        # Cap by the model's explicit VRAM band CEILING when a gpu_mem_gib budget
        # is set (t21) — stretchable to the band ceiling under this model's own
        # need. band_ceiling collapses to the gpu_mem_gib target when no deviation
        # is projected (today), i.e. the same cap autofit already applies.
        gpu_mem_gib = os.environ.get("HUGPY_GPU_MEM_GIB")
        if gpu_mem_gib:
            try:
                from .flex import band_ceiling
                cap = int(band_ceiling(float(gpu_mem_gib) * (2 ** 30),
                                       _vram_deviation_pct(model_key), total))
                budget = min(budget, cap)
            except (TypeError, ValueError):
                pass
        intent, requested = _gguf_ngl_intent(model_key)
        from .flex import plan_partial_offload
        partial = plan_partial_offload(
            weights_bytes=weights, kv_bytes=kv_eff, total_layers=total_layers,
            vram_budget_bytes=budget, ram_free_bytes=_free_ram_bytes(),
            intent=intent, requested_layers=requested)

    if partial is not None and partial.admit:
        # Admit the hybrid. Pin the honest layer count for the in-process
        # llama_cpp load (overriding the shard-blind autofit that re-OOMs a
        # sharded model) AND carry it in the verdict so the slot path launches
        # the child with --n-gpu-layers N instead of -1. Residency is MEASURED
        # post-load (pid-registry) and the slot status reports the real ngl, so
        # the console/bars read the true split with no declared number to drift.
        try:
            from ..managers import spill as _spill
            _spill.set_ngl_override(ppath, partial.n_gpu_layers)
        except Exception:  # noqa: BLE001 — slot opts still carry N; override is a bonus
            pass
        _PARTIAL_NGL[model_key] = {"path": ppath, "n": partial.n_gpu_layers}
        logger.info(
            "partial offload: %s -> %d/%d layers on GPU (%d%%), ~%s VRAM + ~%s RAM "
            "— admitting hybrid instead of refusing (budget %s, ram_free %s)",
            model_key, partial.n_gpu_layers, partial.total_layers, partial.gpu_pct,
            _human_bytes(partial.vram_need_bytes), _human_bytes(partial.ram_need_bytes),
            _human_bytes(partial.vram_budget_bytes), _human_bytes(partial.ram_free_bytes))
        return {"action": "partial", "evicted": evicted, "freed_bytes": freed,
                "reason": None, "n_gpu_layers": partial.n_gpu_layers,
                "gpu_pct": partial.gpu_pct, "partial": partial.as_dict(),
                "note": f"partial GPU offload: {partial.note}"}

    # Still short after eviction AND no admissible partial offload -> HONEST
    # refusal (never admit-then-OOM). Carry what's resident, what's protected +
    # why, what we evicted, the numbers, and — when a partial offload was
    # CONSIDERED and rejected — what it would have been and why (degenerate offload
    # or CPU remainder OOM), in the C4 vision-fit style.
    # t28: a load-FAIL calibration sample (prediction with no successful load).
    _record_calibration_refuse(model_key, _det)
    # A TRUTHFUL account of what holds the card (k30): only claim "protected
    # resident(s) still hold the card" when there ARE protected residents; a
    # failed eviction is reported as such; and when the planner saw NO residents
    # at all on an occupied card, say the occupancy is unattributed instead of
    # the self-contradictory "evicted 0 ... 0 protected still hold the card".
    holders: list[str] = []
    if protected:
        holders.append(f"{len(protected)} protected resident(s) still hold the card")
    if evict_failed:
        holders.append(f"{len(evict_failed)} eviction attempt(s) failed to free "
                       "their resident")
    if not protected and not evict_failed and not candidates and not evicted:
        occupied = None
        if fv is not None and total:
            occupied = max(0, int(total) - int(fv))
        holders.append(
            f"no evictable resident is attributable to a model, yet "
            f"~{_human_bytes(occupied)} of the card is in use — GPU memory is "
            "held by process(es) this worker cannot map to a model_key "
            "(orphaned/adopted child or out-of-band process)")
    reason = {
        "state": "refused",
        "model_key": model_key,
        "reason": (
            f"won't fit on GPU: needs {_human_bytes(need)}{_need_split_str(_det)}, "
            f"{_human_bytes(fv)} free of {_human_bytes(total)} "
            f"({_human_bytes(ceiling_reserve)} ceiling reserve); "
            f"evicted {len(evicted)} idle resident(s) freeing "
            f"{_human_bytes(freed)}"
            + ("; " + "; ".join(holders) if holders else "")
        ),
        "needs_bytes": need,
        # The weights+kv SPLIT (slice 11) — the honest report the operator asked
        # for: what the ctx allocation costs vs the weights.
        "needs_weights_bytes": _det.get("weights"),
        "needs_kv_bytes": _det.get("kv"),
        "ctx_pct": _det.get("ctx_pct"),
        "ctx_resolved": _det.get("ctx_resolved"),
        "ctx_max": _det.get("ctx_max"),
        "kv_geometry_source": _det.get("geometry_source"),
        "free_vram_bytes": fv,
        "total_vram_bytes": total,
        "ceiling_reserve_bytes": ceiling_reserve,
        "evicted": evicted,
        "evicted_freed_bytes": freed,
        "evict_failed": evict_failed,
        "protected": [{"model_key": p["model_key"],
                       "vram_bytes": p.get("vram_bytes"),
                       "host_mode": p.get("host_mode"), "why": p.get("why")}
                      for p in protected],
    }
    if flex_note:
        reason["flex_note"] = flex_note      # what the band flex tried, for hover
    if partial is not None and not partial.admit:
        # A partial offload WAS considered and rejected — say what it would have
        # been and why, so the refusal is honest about the hybrid it declined.
        reason["partial_offload_considered"] = partial.as_dict()
        reason["reason"] = reason["reason"] + "; " + (
            partial.reject_reason or "partial GPU offload not admissible")
    # The load is refused — void any ctx compression / partial-offload commitment
    # we made for it so a future admission of this model re-decides from target.
    _FLEX_CTX_FLOOR.pop(model_key, None)
    _clear_partial_ngl(model_key)
    return {"action": "refuse", "evicted": evicted, "freed_bytes": freed,
            "reason": reason}


# ── Fix B: ensure comfy headroom (evict-to-target-free-VRAM, operator: "always") ─
def _comfy_target_free_bytes() -> int:
    """Target free VRAM to clear before a ComfyUI gen
    (HUGPY_COMFY_TARGET_FREE_GIB, default 7.0 GiB).

    Reasoning for the 7.0 default: recon on ae observed ComfyUI's process VRAM
    growing to ~6.5 GiB (5.5 -> 6.5 G) when it drove a gen — that footprint is
    what topped out the 3090 and evicted nothing. 7.0 GiB is that observed peak
    plus a small margin, so the common still/img2img/id_lock comfy gen has room
    to allocate without OOM/under-offload. It's a knob, not a law: a box running
    heavier SDXL/flux comfy graphs raises it; a tiny-model box lowers it. The
    target is a CEILING on eviction effort, never a guarantee — if nothing is
    evictable we proceed anyway (honest-degrade)."""
    gib = os.environ.get("HUGPY_COMFY_TARGET_FREE_GIB")
    if gib is None or not str(gib).strip():
        val = 7.0
    else:
        try:
            val = float(gib)
        except ValueError:
            logger.warning("ignoring non-numeric HUGPY_COMFY_TARGET_FREE_GIB=%r; "
                           "using 7.0", gib)
            val = 7.0
    return int(max(0.0, val) * 2**30)


def _comfy_headroom_candidates(exclude: str | None) -> list[str]:
    """LRU-ordered (coldest first) on-demand managed model_keys that may be
    evicted to free VRAM for a comfy gen. Union of live SLOT occupants (their
    llama-server children hold real VRAM) and genuine IN-PROCESS residents —
    slot-backed keys are excluded from ``loaded_model_keys`` by design, but they
    are exactly what we must free here, so we add them back from a live slot
    read. Static models are dropped (never evictable); the per-key ``_evict_gate``
    inside ``_evict_model`` still guards in-flight generations. Excludes the comfy
    model_key we're generating FOR. Ordered by dispatch's LRU clock so the coldest
    yields first."""
    keys: set[str] = set()
    try:
        keys.update(loaded_model_keys())         # genuine in-process residents
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..managers.serve.slots import SlotPool
        for s in SlotPool().statuses():
            mk = s.get("model_key")
            if mk:
                keys.add(mk)                     # slot children DO hold VRAM
    except Exception:  # noqa: BLE001 — no slots / pool error -> in-process only
        pass
    if exclude:
        keys.discard(exclude)
    # Drop static (locked) — never a candidate. On-demand (incl. pinned, which
    # yields per 2026-07-15 semantics) stays. The in-flight guard is applied
    # per-key by _evict_model's gate at eviction time.
    cands = [mk for mk in keys if _residency(mk) != "static"]
    try:
        last = _dispatch_last_used()
    except Exception:  # noqa: BLE001
        last = {}
    cands.sort(key=lambda mk: last.get(mk, 0.0))
    return cands


def _dispatch_last_used() -> dict:
    from ..managers.dispatch.dispatch import last_used_snapshot
    return last_used_snapshot()


def _worker_ensure_comfy_headroom(state: "WorkerState", model_key: str,
                                  job_id=None) -> dict:
    """Evict on-demand managed models (LRU, via the SAME _evict_model mechanism
    the evict verb uses) until real free VRAM >= the comfy target, BEFORE a comfy
    gen commits (Fix B). Runs UNCONDITIONALLY per the operator directive ("evict
    down to free target vram always"); a no-op when already above target.

    Honest-degrade at every seam: no GPU / can't read free VRAM -> no-op (return
    early, never block); nothing left to evict but still short -> proceed anyway
    with a logged warning (the comfy gen is NEVER blocked/hung). Returns a small
    telemetry dict (used by the routine's own logging + tests). Best-effort — the
    caller (comfy_runner) swallows any exception, but this stays defensive too."""
    target = _comfy_target_free_bytes()
    fv = _free_vram_bytes()
    if fv is None:
        # No GPU / can't measure: byte-identical to today — do nothing.
        return {"target": target, "free_before": None, "free_after": None,
                "evicted": [], "reached": None, "note": "no GPU / unmeasurable"}
    evicted: list[str] = []
    tried: set[str] = set()
    while fv < target:
        cands = [mk for mk in _comfy_headroom_candidates(exclude=model_key)
                 if mk not in tried]
        if not cands:
            logger.warning(
                "ensure-comfy-headroom: free VRAM %.2fGiB < target %.2fGiB but "
                "nothing on-demand is evictable — proceeding with the comfy gen "
                "anyway (honest-degrade; not blocking the request)",
                fv / 2**30, target / 2**30)
            break
        victim = cands[0]
        tried.add(victim)
        try:
            res = _evict_model(state, victim, force=False)
        except Exception:  # noqa: BLE001 — one bad evict must not wedge the gen
            logger.warning("ensure-comfy-headroom: evict of %s raised; skipping",
                           victim, exc_info=True)
            continue
        if res.get("evicted"):
            evicted.append(victim)
            logger.info("ensure-comfy-headroom: evicted %s (%s) to free VRAM for "
                        "comfy %s", victim, res.get("host_mode"), model_key)
        # Re-read real free VRAM whether or not this one evicted (a gated model
        # frees nothing; we still advanced `tried` so we won't loop on it).
        fv = _free_vram_bytes()
        if fv is None:
            break
    reached = (fv is not None and fv >= target)
    return {"target": target, "free_after": fv, "evicted": evicted,
            "reached": reached}


def _prune_stale_residency(state: "WorkerState") -> None:
    """Tiers v3 lazy cleanup: residency overrides are ASSIGNMENT-scoped unless
    pinned. 🔒 static (and ⏲ on-demand) last while the model stays assigned;
    📌 pin makes the attribution permanent, so pinned overrides survive.

    Drops overrides for models absent from state.assigned_models (the list
    adopted from central's authoritative register/heartbeat response) unless
    pinned. Updates the LIVE settings and persists the file — no re-exec:
    _residency()/the sweep/the slot policy all read _RUNTIME_SETTINGS live."""
    args = getattr(state, "args", None)
    if args is None:                       # startup window before main() wires it
        return
    res = _RUNTIME_SETTINGS.get("residency") or {}
    if not res:
        return
    assigned = set(state.assigned_models)
    stale = [mk for mk in res if mk not in assigned and not _pinned(mk)]
    if not stale:
        return
    for mk in stale:
        logger.info("residency override %r for %s dropped — model unassigned "
                    "and not pinned (static ends at unassign)", res.get(mk), mk)
    settings = _load_settings(args)
    kept = {k: v for k, v in (settings.get("residency") or {}).items()
            if k not in stale}
    if kept:
        settings["residency"] = kept
    else:
        settings.pop("residency", None)
    _save_settings(args, settings)
    live = {k: v for k, v in res.items() if k not in stale}
    if live:
        _RUNTIME_SETTINGS["residency"] = live
    else:
        _RUNTIME_SETTINGS.pop("residency", None)


def _settings_path(args) -> str:
    return args.id_file + ".settings.json"


def _load_settings(args) -> dict:
    try:
        with open(_settings_path(args), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_settings(args, settings: dict) -> None:
    tmp = _settings_path(args) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=1)
    os.replace(tmp, _settings_path(args))


def _apply_settings_env(args) -> dict:
    """Project the settings file onto the env BEFORE anything reads it, so
    every existing consumer (managers.serve.slots._slot_count, …) sees the
    operator's console-set values — and unit drop-ins lose, loudly.

    Runs ONCE at boot (main() calls it before the slot supervisor / any reader)
    and is the ONLY projector — which is what makes the two restart lifecycles
    below work.

    ── Two restart lifecycles for the COMFY_URL / HUGPY_HOT_CACHE_ROOT sentinels ──
    Those two settings are projected onto real env vars that live code reads
    (COMFY_URL; managers/serve/hot_cache.py reads HUGPY_HOT_CACHE_ROOT). To let a
    later CLEAR revert to the true drop-in/env BASE instead of leaking the last
    projected value, the pre-projection base is captured once into a sentinel env
    (_COMFY_URL_BASE_ENV / _HOT_CACHE_ROOT_BASE_ENV). Its lifecycle depends on how
    the agent restarts (see the restart mechanism section):

      * STANDALONE (os.execv): the exec INHERITS os.environ, so the projected
        COMFY_URL and the sentinel both survive into the new image. The sentinel
        is ESSENTIAL here — without it the next boot would recapture the already-
        projected value as the "base" and a clear could never get back to the
        real drop-in/env base. This is the dance the sentinels were built for.

      * SYSTEMD (exit + respawn): the fresh process is started clean by systemd
        with the UNIT's environment, so COMFY_URL/HUGPY_HOT_CACHE_ROOT are back to
        their true base and NO sentinel is inherited. The sentinel is simply
        recaptured from that clean base on this boot — harmless and correct
        (base IS the env). So the sentinel is unnecessary in the systemd path but
        does no harm; the ``if _X not in os.environ`` guards below make both
        lifecycles converge to the same result.
    """
    settings = _load_settings(args)
    _RUNTIME_SETTINGS.clear()
    _RUNTIME_SETTINGS.update(settings)
    if "slot_count" in settings:
        env_was = os.environ.get("SLOT_COUNT")
        os.environ["SLOT_COUNT"] = str(int(settings["slot_count"]))
        _SETTINGS_SOURCE["slot_count"] = "settings"
        if env_was is not None and env_was != os.environ["SLOT_COUNT"]:
            logger.warning("settings override: SLOT_COUNT env/drop-in said %r but "
                           "the operator's runtime settings say %s — settings win",
                           env_was, settings["slot_count"])
    else:
        _SETTINGS_SOURCE["slot_count"] = (
            "env" if os.environ.get("SLOT_COUNT") not in (None, "") else "default")
    # Capture the pre-projection COMFY_URL (drop-in / env / none) ONCE into a
    # sentinel that survives os.execv, so a later clear reverts to the real base
    # instead of leaking the last projected value (execv inherits the live
    # environ; this function is the only projector, run once per boot).
    if _COMFY_URL_BASE_ENV not in os.environ:
        os.environ[_COMFY_URL_BASE_ENV] = os.environ.get("COMFY_URL", "")
    _base = os.environ.get(_COMFY_URL_BASE_ENV, "")
    if settings.get("comfy_url"):
        # Settings win over any env/unit-drop-in COMFY_URL, mirroring slot_count.
        os.environ["COMFY_URL"] = str(settings["comfy_url"])
        _SETTINGS_SOURCE["comfy_url"] = "settings"
        if _base and _base != os.environ["COMFY_URL"]:
            logger.warning("settings override: COMFY_URL env/drop-in said %r but "
                           "the operator's runtime settings say %r — settings win",
                           _base, settings["comfy_url"])
    elif _base:
        os.environ["COMFY_URL"] = _base           # revert to the drop-in/env base
        _SETTINGS_SOURCE["comfy_url"] = "env"
    else:
        os.environ.pop("COMFY_URL", None)         # no base -> 127.0.0.1:8188 default
        _SETTINGS_SOURCE["comfy_url"] = "default"
    # HOT-CACHE ROOT — per-worker attribution of the box-local NVMe LRU tier.
    # managers/serve/hot_cache.py reads HUGPY_HOT_CACHE_ROOT live, so projecting
    # the setting ONTO that env is the whole mechanism (the tier code is
    # untouched): resolution order becomes settings > env base > unset. Same
    # base-sentinel dance as COMFY_URL so a later clear reverts to the true
    # drop-in/env base instead of leaking the last projected value across execv.
    if _HOT_CACHE_ROOT_BASE_ENV not in os.environ:
        os.environ[_HOT_CACHE_ROOT_BASE_ENV] = os.environ.get(_ENV_HOT_CACHE_ROOT, "")
    _hc_base = os.environ.get(_HOT_CACHE_ROOT_BASE_ENV, "")
    if settings.get("hot_cache_root"):
        os.environ[_ENV_HOT_CACHE_ROOT] = str(settings["hot_cache_root"])
        _SETTINGS_SOURCE["hot_cache_root"] = "settings"
        if _hc_base and _hc_base != os.environ[_ENV_HOT_CACHE_ROOT]:
            logger.warning("settings override: HUGPY_HOT_CACHE_ROOT env/drop-in "
                           "said %r but the operator's runtime settings say %r — "
                           "settings win", _hc_base, settings["hot_cache_root"])
        # Best-effort materialization at apply time. NEVER fatal: hot_cache.
        # enabled() re-checks the root live on every use() and disables the tier
        # gracefully if it is (or becomes) uncreatable, so a not-yet-mounted root
        # never breaks the boot — the tier simply activates once it appears.
        try:
            os.makedirs(os.environ[_ENV_HOT_CACHE_ROOT], exist_ok=True)
        except OSError as exc:
            logger.warning("hot_cache_root %r not creatable yet (%s) — the hot "
                           "tier stays off until the path exists",
                           os.environ[_ENV_HOT_CACHE_ROOT], exc)
    elif _hc_base:
        os.environ[_ENV_HOT_CACHE_ROOT] = _hc_base    # revert to the drop-in/env base
        _SETTINGS_SOURCE["hot_cache_root"] = "env"
    else:
        os.environ.pop(_ENV_HOT_CACHE_ROOT, None)     # no base -> tier off (unset)
        _SETTINGS_SOURCE["hot_cache_root"] = "default"
    return settings


def _effective_config() -> dict:
    """What this agent is ACTUALLY running with (for the heartbeat)."""
    try:
        from ..managers.serve.slots import _slot_count
        n = _slot_count()
    except Exception:
        n = None
    # Idle reclamation is OPT-IN (doctrine 2026-07-11): report the TTL as null
    # when the operator hasn't set it, so the console shows the honest "off"
    # (contention-only residency) instead of a phantom 900s clock.
    _ttl_set = "on_demand_ttl_s" in _RUNTIME_SETTINGS
    out = {"slot_count": n,
           "slot_count_source": _SETTINGS_SOURCE.get("slot_count", "default"),
           "on_demand_ttl_s": (int(_RUNTIME_SETTINGS["on_demand_ttl_s"])
                               if _ttl_set else None),
           "on_demand_ttl_s_source": "settings" if _ttl_set else "default"}
    if _RUNTIME_SETTINGS.get("residency"):
        out["residency"] = dict(_RUNTIME_SETTINGS["residency"])
    if _RUNTIME_SETTINGS.get("pinned"):
        out["pinned"] = dict(_RUNTIME_SETTINGS["pinned"])
    if _RUNTIME_SETTINGS.get("ctx_pct"):
        # Per-model context allocation (slice 11) — rides the heartbeat config map
        # to central so the console can show the ctx % the worker will serve.
        out["ctx_pct"] = dict(_RUNTIME_SETTINGS["ctx_pct"])
    out["comfy_url"] = (os.environ.get("COMFY_URL")
                        or "http://127.0.0.1:8188").rstrip("/")
    out["comfy_url_source"] = _SETTINGS_SOURCE.get("comfy_url", "default")
    # HOT-CACHE ROOT: the effective projected root ("" == unset == tier off, the
    # honest reading — hot_cache has no fallback root, unlike comfy_url) + where
    # the value came from, so a /llm/workers row carries the truth exactly as it
    # does for slot_count. This ATTRIBUTES the root per worker; the tier itself is
    # still an automatic LRU cache and the shared store is still the source of truth.
    out["hot_cache_root"] = (os.environ.get(_ENV_HOT_CACHE_ROOT) or "").strip()
    out["hot_cache_root_source"] = _SETTINGS_SOURCE.get("hot_cache_root", "default")
    # Env-profiles (stage 1): the per-profile materialization state
    # (ready|materializing|error) + the model->profile attribution map, so a
    # /llm/workers row carries the truth — central routes a profiled model only
    # once its profile reads ready. Present only when profiles are in play
    # (mirrors residency/pinned). Defensive import: never break a beat.
    if _RUNTIME_SETTINGS.get("profiles"):
        try:
            from ..managers.serve import profiles as _profiles
            out["profiles"] = _profiles.report(_RUNTIME_SETTINGS["profiles"])
        except Exception:  # noqa: BLE001 — heartbeat truth is best-effort
            out["profiles"] = {}
    if _RUNTIME_SETTINGS.get("model_profiles"):
        out["model_profiles"] = dict(_RUNTIME_SETTINGS["model_profiles"])
    return out


def _disk_status() -> dict:
    """Free/total bytes of the volume holding this worker's MODEL ROOT — the
    disk a designation's pull lands on. Central's assign/load preflight uses
    this so a model that won't fit is refused early (409), not mid-pull."""
    try:
        import shutil
        from ..imports.src.constants.constants import DEFAULT_ROOT
        root = DEFAULT_ROOT if os.path.isdir(DEFAULT_ROOT) else os.path.expanduser("~")
        u = shutil.disk_usage(root)
        return {"root": root, "free_bytes": u.free, "total_bytes": u.total}
    except Exception:  # noqa: BLE001
        return {}


# ── ComfyUI presence (slice A of the comfy engine) ──────────────────────────
# The operator installs ComfyUI on the box (own service/venv); the agent
# ADOPTS it: probe the local instance and advertise `comfy` in the heartbeat
# so central can route comfy-templated work here (slice B) and the console
# shows the capability. COMFY_URL overrides the default local port.

_COMFY_CACHE: dict = {"at": 0.0, "value": {"available": False}}


def _comfy_status() -> dict:
    """Probe the local ComfyUI (60s cache): {"available", "url", "version"?,
    "checkpoints"?, "id_lock", "vram_bytes"}. ``vram_bytes`` is ComfyUI's REAL GPU
    footprint from nvidia-smi (per-process), or null — never on-disk checkpoint
    bytes (the 0.1.137 guard). ``id_lock`` is whether this comfy can do
    identity-locked STILLs (the IPAdapter node pack is installed), so central's
    routing gate + the console can see which boxes can do it."""
    now = time.time()
    if now - _COMFY_CACHE["at"] < 60.0:
        out = _COMFY_CACHE["value"]
    else:
        url = (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
        out = {"available": False, "url": url, "id_lock": False}
        try:
            import httpx
            r = httpx.get(url + "/system_stats", timeout=2.0)
            if r.status_code == 200:
                out["available"] = True
                try:
                    sysinfo = (r.json() or {}).get("system") or {}
                    if sysinfo.get("comfyui_version"):
                        out["version"] = sysinfo["comfyui_version"]
                except Exception:  # noqa: BLE001 — version is decoration
                    pass
                # Advertise loadable checkpoints — registry rows' `filename`
                # designations come from this list (slice B).
                try:
                    oi = httpx.get(url + "/object_info/CheckpointLoaderSimple",
                                   timeout=3.0).json()
                    ckpts = oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                    if isinstance(ckpts, list):
                        out["checkpoints"] = ckpts[:50]
                except Exception:  # noqa: BLE001 — list is best-effort
                    pass
                # ID-LOCK capability: probe the SAME object_info API for the
                # IPAdapter node classes, via the comfy runner's own detector so
                # the node-class contract lives in ONE place (never forks from the
                # request-time gate). Rides this 60s presence cache.
                try:
                    from ..managers.comfy.comfy_runner import comfy_has_ipadapter
                    out["id_lock"] = comfy_has_ipadapter(url)
                except Exception:  # noqa: BLE001 — probe/import miss: not capable
                    out["id_lock"] = False
        except Exception:  # noqa: BLE001 — not installed / not running
            pass
        _COMFY_CACHE.update(at=now, value=out)
    # Refresh VRAM every call (cheap — reuses the heartbeat-cached nvidia-smi
    # snapshot), so it isn't frozen for the 60s presence-cache window. Only when
    # ComfyUI is actually up; null otherwise.
    out["vram_bytes"] = _comfy_process_vram() if out.get("available") else None
    return out


def _slot_occupants(strict: bool = False) -> set:
    """Model keys currently seated in this worker's slot pool (empty set when
    slots are disabled/unreachable — callers treat unknown as unoccupied).

    ``strict=True`` re-raises instead of swallowing a probe failure, so a caller
    that must FAIL CLOSED (the reaper) can refuse to delete when it cannot prove
    a model isn't a live slot occupant. The default stays fail-open for telemetry
    callers (heartbeat/survey), where an empty set is harmless."""
    try:
        from ..managers.serve.slots import SlotPool, slots_enabled
        if not slots_enabled():
            return set()
        return {s.get("model_key") for s in SlotPool().statuses()
                if s.get("model_key")}
    except Exception as exc:  # noqa: BLE001 — telemetry, never fatal
        logger.warning("slot occupancy lookup failed: %s", exc)
        if strict:
            raise
        return set()


def _residency_sweep_once(started_at: float) -> None:
    """One pass of the idle TTL sweep — OPT-IN since 2026-07-11 (factored out of
    the loop so it's testable).

    DOCTRINE (operator-locked 2026-07-11): keep models hot. An on-demand model
    stays resident until a NEW load needs its memory — then the LRU on-demand
    resident yields (dispatch.ensure_headroom_for_load). That CONTENTION trigger,
    not a clock, is the default. This idle sweep is the OPT-IN reclamation path:
    it runs ONLY when the operator has explicitly set on_demand_ttl_s (present in
    _RUNTIME_SETTINGS). Absent -> return immediately; contention alone governs
    residency, so a model that just answered a chat is NOT torn down minutes
    later (the drift this correction fixes).

    When enabled, the sweep applies ONLY to IN-PROCESS residents: any non-static
    one idle longer than on_demand_ttl_s is evicted (dispatch.evict cascades to
    the llama singleton — RAM/VRAM actually frees). SLOT occupants are EXEMPT
    (slots stay filled — slice 9 — and a seat changes hands only via LRU
    promotion or explicit unload). Static never yields anywhere."""
    if "on_demand_ttl_s" not in _RUNTIME_SETTINGS:
        return                              # idle reclamation off -> contention only
    ttl = int(_RUNTIME_SETTINGS["on_demand_ttl_s"])
    from ..managers.dispatch.dispatch import (
        last_used_snapshot, evict)
    seated = _slot_occupants()
    last_used = last_used_snapshot()
    now = time.time()
    for mk in loaded_model_keys():
        if _residency(mk) == "static" or mk in seated:
            continue
        idle = now - last_used.get(mk, started_at)
        if idle > ttl:
            logger.info("residency sweep: evicting %s (on-demand in-process, "
                        "idle %.0fs > ttl %ds)", mk, idle, ttl)
            try:
                evict(mk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("residency evict of %s failed: %s", mk, exc)


_SLOT_FILL_LOCK = threading.Lock()


def _fill_empty_slots(state: "WorkerState") -> None:
    """Slice 9: empty slots never sit idle while assigned models exist.

    Runs on startup, after assignment adoption/provisioning, and every
    maintenance tick. Preference order: STATIC first (they must hold seats
    anyway — this subsumes the old static eager-warm on slots boxes), then
    most-recently-used, then any assigned. Candidates must have their files
    local (provisioning re-kicks the fill when a pull lands) and be GGUF
    rows (slots host llama.cpp server children only).

    Each load rides runner_for -> get_llama_runner -> SlotPool.endpoint_for —
    the exact path a live request takes, so per-model opts/ctx resolution,
    same-model reuse and the static-lock guard all apply for free, and each
    load seats itself in an idle slot (never promotes: we only start as many
    loads as there are empty seats). Single-flight."""
    if not _SLOT_FILL_LOCK.acquire(blocking=False):
        return                                   # a fill pass is already running
    try:
        from ..managers.serve.slots import SlotPool, slots_enabled
        if not slots_enabled():
            return
        statuses = SlotPool().statuses()
        empties = [s for s in statuses
                   if "error" not in s and not s.get("model_key")]
        if not empties:
            return
        occupied = {s["model_key"] for s in statuses if s.get("model_key")}
        local = set(_models_local(state))

        def _framework(mk):
            try:
                from .imports import get_model_config
                return getattr(get_model_config(mk), "framework", None)
            except Exception:  # noqa: BLE001 — unknown row: not seatable
                return None

        candidates = []
        for mk in state.assigned_models:
            if mk in occupied or mk not in local or _framework(mk) != "gguf":
                continue
            # k2: an operator BLOCK (aa4aea3) does not auto-unassign — the model
            # can still be sitting in state.assigned_models. Skip it here so this
            # loop stops retrying a doomed seat every ~60s (the 2026-07-18 ae
            # incident); log once, not every tick.
            if _is_blocked_locally(mk):
                _log_blocked_skip_once(mk, "slot fill")
                continue
            candidates.append(mk)
        if not candidates:
            return
        from ..managers.dispatch.dispatch import last_used_snapshot
        last_used = last_used_snapshot()
        candidates.sort(key=lambda mk: (0 if _residency(mk) == "static" else 1,
                                        -last_used.get(mk, 0.0)))
        for mk in candidates[:len(empties)]:
            try:
                logger.info("slot fill: seating %s (%s) in an empty slot",
                            mk, _residency(mk))
                from abstract_hugpy_dev.managers.dispatch.dispatch import runner_for
                runner = runner_for(model_key=mk)   # builds the LAZY wrapper only
                # The seat happens on first .runner access (get_llama_runner ->
                # _build_runner -> SlotPool.endpoint_for). Without forcing it the
                # filler registered a hollow in-process shell and NEVER seated a
                # slot — both slots stayed empty and chat 404'd on the empty slot
                # endpoint. ensure_loaded() materialises the runner = the seat.
                _ensure = getattr(runner, "ensure_loaded", None)
                if callable(_ensure):
                    _ensure()
            except Exception as exc:  # noqa: BLE001 — one seat must not block the rest
                logger.warning("slot fill for %s failed: %s", mk, exc)
    finally:
        _SLOT_FILL_LOCK.release()


def _vram_headroom_sweep(state: "WorkerState") -> None:
    """90% HEADROOM TRIGGER (slice 10 addendum). Admission-time evict-to-fit only
    fires when a NEW load arrives. The second incident had NO new load: ComfyUI
    grew out-of-band while an IDLE non-grower slot child squatted, and the card
    reached 100% and DEADLOCKED — the keeper had to /evict by hand. This closes
    that: on every residency beat, if the card is at/over the ceiling (free VRAM
    below the (1 - ceiling) reserve), evict the coldest EVICTABLE idle resident
    (same protection rules — static / actively-replying / queued-ahead / comfy are
    never touched) to claw back to headroom. No new timer: it rides the existing
    60s residency loop. A no-op when unmeasurable or already under the ceiling.

    Deliberately evicts AT MOST ONE resident per beat: a single reclaim (21G here)
    is enough to break a deadlock, and one-per-beat avoids over-evicting a box
    that's merely near the line — the next beat re-checks and takes another only
    if still pressured."""
    total = _total_vram_bytes()
    if not total:
        return
    fv = _free_vram_bytes()
    if fv is None:
        return
    reserve = int(total * (1.0 - _vram_ceiling_frac()))
    if fv >= reserve:
        return                               # under the ceiling — nothing to do
    # Over the ceiling with no load driving admission. Evict the coldest EVICTABLE
    # idle resident, applying the SAME protection rules as _vram_evict_to_fit.
    busy_slots = _busy_slot_models()
    try:
        from ..managers.dispatch.dispatch import last_used_snapshot as _lus
        lru = _lus()
    except Exception:  # noqa: BLE001
        lru = {}
    residents = _vram_residents(state)
    cands = []
    for r in residents:
        mk = r["model_key"]
        if str(r.get("host_mode")) == "comfy":
            continue                         # comfy has its own path; never here
        if _residency(mk) == "static":
            continue
        if _actively_replying(mk, busy_slots):
            continue
        # No `queued_ahead` here — there is no subject load this pass; a resident
        # with in-flight work is already protected by _actively_replying above.
        cands.append(r)
    if not cands:
        logger.warning("VRAM headroom: card at/over the %.0f%% ceiling (%s free of "
                       "%s) but nothing evictable — every resident is static or "
                       "actively replying; leaving it (autofit/degrade)",
                       _vram_ceiling_frac() * 100, _human_bytes(fv),
                       _human_bytes(total))
        return
    cands.sort(key=lambda r: (lru.get(r["model_key"], 0.0),
                              -int(r.get("vram_bytes") or 0)))
    victim = cands[0]["model_key"]
    logger.info("VRAM headroom: card at/over the %.0f%% ceiling (%s free) with no "
                "load driving admission — evicting coldest idle resident %s "
                "(operator addendum: no human is the eviction policy)",
                _vram_ceiling_frac() * 100, _human_bytes(fv), victim)
    res = _evict_model(state, victim)
    if res.get("evicted"):
        _note_vram_eviction(victim, "headroom-sweep", res.get("vram_freed"),
                            res.get("host_mode") or "")
        _trim_host_ram()


def _residency_sweep_loop(state: "WorkerState") -> None:
    """Residency maintenance every 60s: fill empty slots (slice 9), enforce the
    90% VRAM headroom (slice 10 addendum), then run the idle TTL sweep — which is
    a no-op unless the operator opted into on_demand_ttl_s (contention governs
    residency by default; see _residency_sweep_once + ensure_headroom_for_load)."""
    started_at = time.time()
    while True:
        time.sleep(60.0)
        try:
            _fill_empty_slots(state)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("slot fill pass failed: %s", exc)
        try:
            _vram_headroom_sweep(state)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("VRAM headroom sweep failed: %s", exc)
        try:
            _residency_sweep_once(started_at)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("residency sweep iteration failed: %s", exc)


def _load_update_state(args) -> dict:
    try:
        with open(_update_state_path(args), "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _save_update_state(args, state: dict) -> None:
    try:
        with open(_update_state_path(args), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError:
        pass


def _self_update_if_needed(required: str | None, args, state=None) -> None:
    """Install central's required package version and re-exec, if we're behind.

    Source of the bytes: PyPI by default (where ``sync.trigger`` publishes), since
    workers reach central over the public internet and thus have PyPI access too
    (WireGuard is only the inference callback). ``--pkg-index`` /
    ``WORKER_PKG_INDEX`` overrides to central's own simple index — for a WG-only
    worker with no general egress, or to keep dev builds off public PyPI.

    ``--no-deps``: this is a code hot-swap of an already-provisioned env, so we
    pull ONLY the package and skip dependency resolution. A dev build that adds a
    brand-new dependency needs a one-off full reinstall.
    """
    if not required:
        return  # central isn't managing versions -> never touch the install
    installed = _installed_pkg_version(args.pkg_name)
    if required == installed:
        return

    state = _load_update_state(args)
    if state.get("target") == required and (time.time() - state.get("at", 0)) < _UPDATE_RETRY_BACKOFF:
        return  # already tried this exact target recently; back off

    source = args.pkg_index or "PyPI"
    logger.info("self-update: %s %s -> %s (from %s)",
                args.pkg_name, installed or "(none)", required, source)
    cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-deps"]
    if args.pkg_index:
        cmd += ["--index-url", args.pkg_index]
    cmd.append(f"{args.pkg_name}=={required}")
    try:
        rc = subprocess.call(cmd)
    except Exception as exc:  # noqa: BLE001
        logger.warning("self-update pip invocation failed: %s", exc)
        rc = 1
    _save_update_state(args, {"target": required, "at": time.time(), "rc": rc})

    if rc == 0:
        logger.info("self-update installed %s==%s; restarting agent",
                    args.pkg_name, required)
        # Restart onto the new code. Under systemd this EXITS (Restart= respawns
        # a fresh, properly-tracked process — never the os.execv orphan that
        # squatted :9100). kill_slots=True: an orphaned slot child would keep
        # serving the OLD code forever (the adoption probe can't tell versions
        # apart), so the restart tears them down and the fresh agent respawns
        # them on the new version. Runs on the register/heartbeat thread — no
        # HTTP ack to send, so restart synchronously (does not return).
        from .._platform.procutil import reexec
        _restart(state, reason="self-update", reexec_fn=reexec, kill_slots=True)
    else:
        logger.warning("self-update failed (pip rc=%s); staying on %s",
                       rc, installed or "(none)")


def _terminal_exit(exc: "WorkerRejected") -> None:
    """Stop the agent for good after central refused it (401/403).

    Called from the daemon heartbeat thread too, so it must kill the whole
    process (the main thread is blocked in the inference server) — hence
    os._exit. Exit code 0 so a ``Restart=on-failure`` unit does NOT respawn a
    deliberately-evicted worker (transient crashes still exit non-zero/killed and
    are restarted as before).
    """
    if getattr(exc, "code", None) == 403:
        logger.error("central BLOCKED this worker (403): %s. Stopping — have the "
                     "operator Admit it in the console to rejoin.", exc)
    else:
        logger.error("central refused enrollment (401): %s. Stopping — re-enroll "
                     "with a valid WORKER_ENROLL_TOKEN.", exc)
    os._exit(0)


def env_status() -> dict:
    """Runtime-env capability snapshot: which env TIER this worker serves.

    The tier names the venv this unit runs (WORKER_ENV_TIER, default "stable" —
    the known-good pinned env; "edge" = bleeding-edge libs for models the stable
    env can't load). Library versions are read from the running env itself, so
    central sees the truth rather than a config claim. Central routes a model
    mapped in HUGPY_MODEL_ENV_TIERS only to workers advertising that tier.
    """
    import platform
    tier = (os.environ.get("WORKER_ENV_TIER") or "stable").strip().lower()
    info: dict = {"tier": tier or "stable", "python": platform.python_version()}
    try:
        from importlib.metadata import version
        for pkg in ("llama-cpp-python", "transformers", "torch",
                    "diffusers", "accelerate", "bitsandbytes"):
            try:
                info[pkg] = version(pkg)
            except Exception:  # noqa: BLE001 — absent package: simply unreported
                pass
    except Exception:  # noqa: BLE001
        pass
    return info


# ── install-shape detection (central-side drift detection) ──────────────────
# What SHAPE this worker is installed in, so central can flag boxes that drifted
# from the canonical installer (a hand-rolled unit, a bare process from the wrong
# venv, a stray system unit). Additive heartbeat field — older centrals ignore it.
#
# Canonical unit name is the product-named `hugpy-worker.service`;
# `abstract-hugpy-worker.service` is the recognized LEGACY alias (hand-written
# units the setup doc used to document). Both count as canonical.
_CANONICAL_UNITS = {"hugpy-worker.service", "abstract-hugpy-worker.service"}
_INSTALL_SHAPE: "dict | None" = None


def _detect_systemd_unit() -> "str | None":
    """This process's systemd unit from ``/proc/self/cgroup`` (its last
    ``*.service`` path component), or ``None``. Best-effort — any read/parse
    failure returns ``None`` and never raises."""
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as fh:
            data = fh.read()
    except Exception:  # noqa: BLE001
        return None
    for tok in reversed(data.replace("/", "\n").split("\n")):
        tok = tok.strip()
        if tok.endswith(".service"):
            return tok
    return None


def _compute_install_shape(*, invocation_id, unit, prefix, executable) -> dict:
    """Pure install-shape logic (inputs -> the reported dict). Factored out so
    the canonical truth-table is testable without touching /proc or systemd."""
    via_systemd = bool(invocation_id)
    venv = (prefix or "").rstrip("/")
    canonical = bool(via_systemd
                     and unit in _CANONICAL_UNITS
                     and venv.endswith("hugpy-worker/venv"))
    return {"unit": unit, "via_systemd": via_systemd,
            "venv": prefix, "python": executable, "canonical": canonical}


def _install_shape() -> dict:
    """Cached install-shape for the heartbeat:
    ``{unit, via_systemd, venv, python, canonical}``.

    Computed ONCE (a running process's unit/venv don't change). Fully defensive:
    on any failure it returns a well-formed dict with null/false fields so a
    detection bug can never break the heartbeat.
    """
    global _INSTALL_SHAPE
    if _INSTALL_SHAPE is not None:
        return _INSTALL_SHAPE
    try:
        shape = _compute_install_shape(
            invocation_id=os.environ.get("INVOCATION_ID"),
            unit=_detect_systemd_unit(),
            prefix=sys.prefix,
            executable=sys.executable,
        )
    except Exception:  # noqa: BLE001 — detection must never break the heartbeat
        shape = {"unit": None, "via_systemd": False,
                 "venv": None, "python": None, "canonical": False}
    _INSTALL_SHAPE = shape
    return shape


def _serving_limits() -> dict:
    """Per-worker safe concurrency for IN-PROCESS serving, advertised to central.

    ``in_process_max_concurrency`` is the number of requests that may enter an
    in-process (llama.cpp / transformers) model runner at once — the per-model
    generation gate's limit. Default 1 (native contexts serialize). Central reads
    this to gate its relays: a worker that omits it (older agent) is assumed 1.
    """
    return {"in_process_max_concurrency": gen_gate.concurrency_limit()}


def _slot_capability() -> dict:
    """Whether this box can seat a NATIVE, crash-isolated llama-server slot.

    ``slot_capable`` is the engine-binary truth: a resolvable native
    ``llama-server`` (HUGPY_ENGINE_DIR / LLAMA_SERVER_BIN / PATH). When absent,
    slot seating falls back to the in-process ``llama_cpp.server`` child (text
    only — vision GGUF is refused) or, with SLOT_COUNT=0, to the in-process
    runner outright. Either way the box is serving non-native, which
    central/console must SEE in EVERY heartbeat — that silence is exactly
    computron's 2026-07-11 condition (slots implied, no usable engine binary).
    Fully defensive: any probe failure reports slot_incapable with the reason and
    never breaks the heartbeat.
    """
    try:
        from ..engine.resolve import server_bin
        binpath = server_bin()
    except Exception as exc:  # noqa: BLE001 — capability probe must never break a beat
        return {"slot_capable": False,
                "slot_incapable_reason": f"engine probe failed: "
                                         f"{type(exc).__name__}: {exc}"}
    if binpath:
        return {"slot_capable": True, "slot_incapable_reason": None}
    try:
        from ..managers.serve.slots import slots_enabled, _slot_count
        n = _slot_count()
        slotted = slots_enabled()
    except Exception:  # noqa: BLE001
        n, slotted = 0, False
    reason = ("no native llama-server binary resolvable (set HUGPY_ENGINE_DIR / "
              "LLAMA_SERVER_BIN or run `hugpy install-engine`)")
    if slotted:
        reason += (f"; the {n} configured slot(s) fall back to the in-process "
                   "llama_cpp.server child — text only, vision GGUF is refused")
    else:
        reason += "; SLOT_COUNT=0, so this worker serves models in-process (gated)"
    return {"slot_capable": False, "slot_incapable_reason": reason}


# Per-task capability honesty (2026-07-11). Yesterday three requests reached
# workers whose canonical venv lacks an optional ML dep (sentence-transformers,
# openai-whisper, keybert) and failed AT REQUEST TIME ("sentence-transformers is
# required…", whisper NoneType). Central routes by model assignment alone, so it
# had no way to know a box couldn't run the task. We advertise a per-task map from
# the SAME find_spec probe central's /ml readiness uses, and central skips a worker
# that says False for the request's task (workers_for_model). Legacy agents omit
# the field -> central assumes capable (no regression).

# whisper needs a REAL import probe: find_spec("whisper") can be True yet
# `import whisper` die under numba/numpy>=2.5 (yesterday's third incident), so the
# find_spec-only base map would over-advertise ASR. We do ONE guarded real import,
# TTL-cached so the ~15s heartbeat stays cheap AND an /ops/pip fix is re-detected
# within the TTL instead of needing a worker restart.
_WHISPER_PROBE_TTL_S = 300.0
_WHISPER_PROBE: dict = {"ok": None, "at": 0.0}


def _whisper_importable() -> bool:
    """Whether ``import whisper`` actually SUCCEEDS on this box (TTL-cached).

    Fast path: if whisper isn't even resolvable, return False without importing.
    Otherwise do a guarded real import (the find_spec-insufficient special case)
    and cache the result for ``_WHISPER_PROBE_TTL_S`` so heartbeats stay cheap.
    """
    from ..managers.task_deps import have
    if not have("whisper"):
        return False
    now = time.time()
    cached = _WHISPER_PROBE.get("ok")
    if cached is not None and (now - _WHISPER_PROBE.get("at", 0.0)) < _WHISPER_PROBE_TTL_S:
        return cached
    try:
        import whisper  # noqa: F401 — REAL probe: numba/numpy>=2.5 landmine (2026-07-11)
        ok = True
    except Exception as exc:  # noqa: BLE001 — any import failure = ASR unavailable
        logger.info("whisper is installed but `import whisper` failed (%s: %s); "
                    "advertising automatic-speech-recognition UNAVAILABLE so central "
                    "won't route ASR here", type(exc).__name__, exc)
        ok = False
    _WHISPER_PROBE["ok"] = ok
    _WHISPER_PROBE["at"] = now
    return ok


def _task_capabilities() -> dict:
    """``{task: bool}`` this worker can actually run, advertised to central.

    Built from the shared canonical task->dependency map (managers.task_deps) with
    the SAME find_spec probe central's /ml readiness uses — cheap, no heavy imports
    — then overlaid with the whisper real-import special case. Central gates
    routing on it (workers_for_model): a box missing an optional ML dep never gets
    that task's requests, instead of failing them at request time.
    """
    from ..managers.task_deps import task_capabilities as _base_task_caps
    caps = _base_task_caps()
    caps["automatic-speech-recognition"] = _whisper_importable()
    return caps


def _heartbeat_loop(client: CentralClient, state: WorkerState, args) -> None:
    while True:
        time.sleep(args.heartbeat)
        try:
            # Compute slot statuses ONCE — the unified allocations view reuses it.
            _slots = _slot_statuses()
            # Precision model->PID registry (2026-07-14): populate from data the
            # agent already has THIS beat — slot child_pid (subprocess-hosted),
            # in-process torch keys (share the worker PID), comfy — then reconcile
            # against nvidia-smi ground truth so central gets an honest per-model
            # PID+VRAM log plus any unattributed (foreign/rogue) squatters.
            # Best-effort and fully isolated: a registry error must NEVER skip the
            # beat (a missed heartbeat drops the worker off the fleet), so it
            # degrades to no log exactly like a no-GPU box.
            try:
                from . import pid_registry as _pidreg
                _pidreg.sweep_dead()
                for _s in (_slots or []):
                    if _s.get("model_key") and _s.get("child_pid"):
                        _pidreg.record_launch(_s["model_key"], _s["child_pid"], "subprocess")
                _inproc = _inprocess_gpu_bytes()
                for _mk in _inproc:
                    _pidreg.record_launch(_mk, os.getpid(), "in_process")
                # OWN-PID attribution (2026-07-14): tell reconcile which GPU pids are
                # the worker's own infrastructure so a residual agent / idle-slot CUDA
                # context lump reads as "cuda_context", not an anonymous squatter.
                # os.getpid() is the agent; the venv marker (this python's venv root)
                # catches slot children sharing the venv that aren't a recorded model.
                _own_pids = {os.getpid()}
                _venv_marker = None
                try:
                    _venv_marker = os.path.dirname(os.path.dirname(sys.executable)) or None
                except Exception:  # noqa: BLE001 — marker is best-effort telemetry
                    _venv_marker = None
                _pidreg.reconcile(_gpu_process_vram(), _inproc, _comfy_process_vram(),
                                  own_pids=_own_pids, self_venv_marker=_venv_marker)
                _pid_log = _pidreg.snapshot_for_heartbeat()
            except Exception as _pe:  # noqa: BLE001 — telemetry must never break the beat
                logger.debug("pid_registry snapshot failed: %s", _pe)
                _pid_log = None
            # Honest budget-bar VRAM inputs (t13/t14): split the SAME pid_log into
            # attributed (worker) vs unattributed (foreign) — sampled this beat,
            # alongside detect_gpus() below, so the two are one snapshot.
            _vram_split = _vram_split_from_pidlog(_pid_log)
            # t28: compute the allocations view ONCE, harvest calibration samples
            # off it (measured per-process VRAM per resident), then reuse it for
            # the payload. Fully guarded — capture must never skip a beat.
            _allocs = _allocations(_slots)
            try:
                _collect_calibration_from_allocations(_allocs)
                _calib_samples = _drain_calibration_samples()
            except Exception as _ce:  # noqa: BLE001 — telemetry never breaks a beat
                logger.debug("calibration capture failed: %s", _ce)
                _calib_samples = []
            worker = client.heartbeat(
                state.worker_id,
                {
                    "gpus": detect_gpus(),
                    "loaded_models": loaded_model_keys(),
                    "loading": _loading_model_keys(),
                    "models_local": _models_local(state),
                    "provisioning": sorted(state._provisioning),
                    "provision_progress": state.provision_snapshot(),
                    "spill": _spill_describe(),
                    "url": state.url,     # None -> central keeps source-IP URL
                    "port": state.port,
                    # HONEST version: the running image (import-time snapshot),
                    # never live disk metadata — so a not-yet-effective self-update
                    # shows as skew, not cosmetic convergence (2026-07-20 ae).
                    "pkg_version": _running_pkg_version(args.pkg_name),
                    "role": state.role,
                    "rpc_endpoint": state.rpc_endpoint,
                    "free_ram": _free_ram_bytes(),
                    "ram_total": _ram_total_bytes(),
                    # Honest budget-bar inputs (t13/t14). free_ram stays the
                    # CLAMPED wire-compat field; these are the spec's raw inputs
                    # so central computes bar_used/encroachment/over-limit for RAM
                    # (ram_worker + ram_external) and VRAM (vram_attributed +
                    # vram_unattributed) without guessing. Absent on older agents
                    # -> central degrades to legacy bar semantics.
                    "free_ram_raw": _free_ram_raw_bytes(),
                    "ram_worker_bytes": _ram_worker_bytes(),
                    "ram_external_bytes": _ram_external_bytes(),
                    "vram_attributed_bytes": _vram_split.get("vram_attributed_bytes"),
                    "vram_unattributed_bytes": _vram_split.get("vram_unattributed_bytes"),
                    "disk": _disk_status(),
                    "engine": llama_cpp_cuda_status(),
                    "pool": os.environ.get("WORKER_POOL", ""),
                    "caps": _local_caps(),
                    "env": env_status(),
                    "config": _effective_config(),
                    "comfy": _comfy_status(),
                    "loaded_detail": _loaded_detail(),
                    "slots": _slots,
                    "allocations": _allocs,
                    # t28 load-and-learn: prediction-vs-measured observations
                    # captured this beat (measured successes off `allocations` +
                    # any admission refusals). Additive/optional — None when empty,
                    # omitted for a worker with HUGPY_CALIBRATION=off.
                    "calibration_samples": _calib_samples or None,
                    # Precision model->PID log (2026-07-14): {"models":[{model_key,
                    # pid,host_mode,vram_bytes,alive}], "unattributed":[{pid,name,
                    # mib}]}. None on older/no-GPU boxes -> central just omits it.
                    "pid_registry": _pid_log,
                    "storage": _worker_storage(state),
                    # VRAM eviction churn (slice 10): the eviction the operator was
                    # watching for now surfaces — count + last event {victim,
                    # subject, host_mode, vram_freed, at}. Silent before; a VRAM
                    # evict-to-fit at admission increments this so the churn data
                    # includes GPU evictions, not just disk reaps.
                    "vram_evictions": dict(_VRAM_EVICTIONS),
                    "install": _install_shape(),
                    # Concurrency hardening (2026-07-11): advertise this box's
                    # safe in-process concurrency + whether it can seat a native
                    # crash-isolated slot, so central can gate relays and the
                    # console can badge a worker that's silently serving in-process.
                    "serving_limits": _serving_limits(),
                    **_slot_capability(),
                    # Per-task capability honesty (2026-07-11): which /ml tasks
                    # this box can actually run, so central won't route a task
                    # whose optional dep is missing here (workers_for_model gate).
                    "task_capabilities": _task_capabilities(),
                },
            )
            # Adopt any assignment change made in the UI + pre-provision it.
            _sync_assignment(state, worker)
            # Adopt central's resource limits (min of central + local config).
            _apply_central_limits(worker)
            # t28: adopt central's learned per-model need corrections (if any).
            _adopt_calibration(worker)
            # k2: adopt central's model BLOCK set (if any) — see
            # _adopt_blocked_models. Gates only this worker's own background
            # warm/load-ahead loops (slot fill, provisioning re-kick).
            _adopt_blocked_models(worker)
            # Per-worker KEEP-WARM STAR (operator RULINGS 2026-07-23): if central
            # named a star for this worker, keep it warm — load it IF NOT
            # CURRENTLY LOADED (a normally-evictable on-demand resident, NOT
            # static). Reconcile-kept, NOT boot-once: this runs every beat, so a
            # star evicted under pressure is reloaded on the next beat; an
            # already-loaded star is a no-op (idempotent).
            _adopt_boot_prewarm(state, worker)
            # Keep the STORAGE budget's two central-owned inputs on state: the
            # disk allocation and the LRU clock the FIFO orders by. Both are
            # facts only central holds; the pull path reads them off state.
            _adopt_storage_inputs(state, worker)
            # Converge to central's required package version (restarts on update).
            _self_update_if_needed((worker or {}).get("required_pkg_version"), args, state)
        except WorkerRejected as exc:
            _terminal_exit(exc)   # does not return
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                # Central forgot us (restart / cleared registry) — re-register.
                logger.warning("central returned 410; re-registering")
                _register(client, state, args)
            else:
                logger.warning("heartbeat HTTP %s", exc.code)
        except Exception as exc:
            logger.warning("heartbeat failed: %s", exc)


def _register(client: CentralClient, state: WorkerState, args) -> None:
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    payload = {
        "name": state.name,
        "url": state.url,            # None -> central uses the source IP
        "port": state.port,
        "gpus": detect_gpus(),
        "role": state.role,
        "rpc_endpoint": state.rpc_endpoint,
        "free_ram": _free_ram_bytes(),
        "ram_total": _ram_total_bytes(),
        "models": models or None,
        "worker_id": state.worker_id,
        # HONEST version: running image, not live disk metadata (see heartbeat).
        "pkg_version": _running_pkg_version(args.pkg_name),
        "engine": llama_cpp_cuda_status(),
        "pool": os.environ.get("WORKER_POOL", ""),
        "caps": _local_caps(),
        "env": env_status(),
        # Concurrency hardening (2026-07-11): advertise safe in-process
        # concurrency + native-slot capability from the first contact, so central
        # gates correctly and the console badges an in-process-serving box even
        # before the first heartbeat.
        "serving_limits": _serving_limits(),
        **_slot_capability(),
        # Per-task capability honesty (2026-07-11): advertised from first contact
        # so central's routing gate is correct before the first heartbeat.
        "task_capabilities": _task_capabilities(),
    }
    try:
        worker = client.register(payload)
    except WorkerRejected as exc:
        _terminal_exit(exc)   # does not return — blocked/revoked, don't retry
    state.worker_id = worker.get("id", state.worker_id)
    if state.worker_id:
        _save_worker_id(args.id_file, state.worker_id)
    # Adopt central's view of what we serve (it may already have assignments
    # for this worker_id from a previous session) and pre-provision them.
    _sync_assignment(state, worker)
    _apply_central_limits(worker)
    # Per-worker KEEP-WARM STAR (operator RULINGS 2026-07-23): the register reply
    # may already carry this worker's star — warm it at first contact if not
    # already resident (a normally-evictable on-demand resident, NOT static). The
    # heartbeat loop then keeps it warm every beat (reconcile-kept, not once).
    _adopt_boot_prewarm(state, worker)
    logger.info("registered as worker id=%s serving models=%s", state.worker_id, worker.get("models"))
    # Converge to central's required package version before serving (restarts).
    _self_update_if_needed(worker.get("required_pkg_version"), args, state)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abstract_hugpy_dev.worker_agent")
    p.add_argument("--central", default=central_base_url(default=None),
                   help="Central base URL, e.g. https://hugpy.ai "
                        "(env HUGPY_BASE_URL; legacy WORKER_CENTRAL_URL honoured)")
    p.add_argument("--token", default=os.environ.get("WORKER_ENROLL_TOKEN"),
                   help="Enrollment token issued by the console (hpw_...). Sent as a "
                        "Bearer credential on register/heartbeat. Required once central "
                        "has HUGPY_WORKER_ENROLL_REQUIRED on; recommended otherwise.")
    p.add_argument("--name", default=os.environ.get("WORKER_NAME", socket.gethostname()))
    p.add_argument("--host", default=os.environ.get("WORKER_HOST", "0.0.0.0"),
                   help="Bind address for the worker's inference server")
    p.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", "9100")))
    p.add_argument("--advertise", default=os.environ.get("WORKER_URL"),
                   help="URL the central node should call back on "
                        "(defaults to http://<host>:<port>)")
    p.add_argument("--models", default=os.environ.get("WORKER_MODELS", ""),
                   help="Comma-separated model_keys to self-assign on registration")
    p.add_argument("--heartbeat", type=float, default=float(os.environ.get("WORKER_HEARTBEAT", "15")))
    p.add_argument("--id-file", default=os.environ.get(
        "WORKER_ID_FILE", os.path.expanduser("~/.abstract_hugpy_worker.json")))

    # Self-update: which distribution to track, and where to pull it from.
    # Default source is PyPI (where sync.trigger publishes); set --pkg-index to
    # central's simple index for a WG-only worker with no general egress.
    p.add_argument("--pkg-name", default=os.environ.get("WORKER_PKG_NAME", "abstract_hugpy_dev"),
                   help="Distribution to self-update (default abstract_hugpy_dev). "
                        "Must match the distribution whose version central advertises.")
    p.add_argument("--pkg-index", default=os.environ.get("WORKER_PKG_INDEX"),
                   help="Override pip --index-url for self-update "
                        "(default: PyPI; e.g. https://<central>/api/llm/pip/simple)")

    # Fleet role + RPC shard pool.
    p.add_argument("--role", default=os.environ.get("WORKER_ROLE", "worker"),
                   choices=["worker", "rpc"],
                   help="worker = whole-model (serves /infer); rpc = lends its GPU "
                        "to a shard pool via llama.cpp rpc-server")
    p.add_argument("--rpc-host", default=os.environ.get("WORKER_RPC_HOST", "0.0.0.0"),
                   help="bind address for rpc-server (role=rpc)")
    p.add_argument("--rpc-port", type=int, default=int(os.environ.get("WORKER_RPC_PORT", "50052")),
                   help="port for rpc-server / advertised rpc_endpoint (role=rpc)")
    p.add_argument("--rpc-bin", default=os.environ.get("WORKER_RPC_BIN", "rpc-server"),
                   help="path to the llama.cpp rpc-server binary (CUDA+RPC build)")

    # GPU/CPU spill defaults for this worker. These seed the spill env the
    # inference path reads; per-request overrides from central still win.
    spill = p.add_argument_group("spill (GPU/CPU split)")
    spill.add_argument("--spill", choices=["auto", "off"],
                       default=os.environ.get("WORKER_SPILL", "auto"),
                       help="auto = fit as many layers on GPU as VRAM allows "
                            "(spill rest to CPU); off = CPU only")
    spill.add_argument("--n-gpu-layers", type=int, default=_safe_int(os.environ.get("WORKER_N_GPU_LAYERS")),
                       help="llama.cpp: force N layers on GPU (overrides --spill)")
    spill.add_argument("--gpu-mem", type=float, default=_safe_float(os.environ.get("WORKER_GPU_MEM_GIB")),
                       help="transformers: per-GPU memory budget in GiB")
    spill.add_argument("--cpu-mem", type=float, default=_safe_float(os.environ.get("WORKER_CPU_MEM_GIB")),
                       help="transformers: CPU/RAM budget in GiB for offloaded layers")
    spill.add_argument("--tensor-split", default=os.environ.get("WORKER_TENSOR_SPLIT"),
                       help="multi-GPU split, comma-separated e.g. 0.7,0.3")
    spill.add_argument("--main-gpu", type=int, default=_safe_int(os.environ.get("WORKER_MAIN_GPU")),
                       help="primary GPU index")
    return p


def _safe_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _apply_cli_spill(args) -> None:
    """Seed the spill env from CLI flags (per-request overrides still win)."""
    if args.n_gpu_layers is not None:
        os.environ["HUGPY_N_GPU_LAYERS"] = str(args.n_gpu_layers)
    elif args.spill == "off":
        os.environ["HUGPY_N_GPU_LAYERS"] = "off"
    else:
        os.environ.setdefault("HUGPY_N_GPU_LAYERS", "auto")
    if args.gpu_mem is not None:
        os.environ["HUGPY_GPU_MEM_GIB"] = str(args.gpu_mem)
    if args.cpu_mem is not None:
        os.environ["HUGPY_CPU_MEM_GIB"] = str(args.cpu_mem)
    if args.tensor_split:
        os.environ["HUGPY_TENSOR_SPLIT"] = args.tensor_split
    if args.main_gpu is not None:
        os.environ["HUGPY_MAIN_GPU"] = str(args.main_gpu)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # BOOT DETOX (2026-07-08 ae crash-loop): a 0.1.158 studio render setdefault'ed
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which SURVIVES the agent's
    # re-exec (os.environ is inherited by execv) and this driver/torch combo dies
    # natively under it — poisoning every subsequent CUDA load incl. boot warms.
    # Unless the operator explicitly opted in (HUGPY_CUDA_EXPANDABLE=1), strip the
    # exact leaked value BEFORE any torch import so the box heals on converge.
    if (os.environ.get("HUGPY_CUDA_EXPANDABLE", "").strip() != "1"
            and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        logging.getLogger(__name__).warning(
            "boot detox: removed leaked PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "(opt back in with HUGPY_CUDA_EXPANDABLE=1)")
    args = _build_parser().parse_args(argv)

    if not args.central:
        print("error: --central (or WORKER_CENTRAL_URL) is required", file=sys.stderr)
        return 2

    _apply_cli_spill(args)

    # A worker runs vision models on its own GPU in-process; it has no separate
    # vision server to POST to. Force in-process unless the operator overrode it.
    os.environ.setdefault("HUGPY_VISION_INPROCESS", "1")

    # Torch-first import guard (Bug D): pull torch into sys.modules NOW, before
    # anything in this process can import llama_cpp (the in-process GGUF fallback
    # or a stray probe). A CUDA-built llama_cpp imported first aborts torch's init
    # and leaves a broken half-module cached for the whole process — poisoning
    # every later vision/sd-turbo/whisper request. Priming here makes torch a
    # complete cached module the rest of the run reuses. No-op without torch.
    _prime_torch_before_llama()

    # Only advertise a URL when the operator set one explicitly. Otherwise leave
    # it to central, which derives the reachable address from the request source
    # IP — far more reliable than the worker guessing past 127.0.1.1 / NAT / odd
    # NICs. We still send the listen port so central can build host:port.
    advertise = args.advertise
    if not advertise:
        # Determine the worker's own outbound IP on the route to central. This
        # is reliable even across NAT hairpinning, which fools central's
        # source-IP guess (central would see the router, e.g. 192.168.1.1, not
        # the worker's .128). Falls back to None -> central uses the source IP.
        ip = _local_ip_toward(args.central)
        if ip:
            advertise = f"http://{ip}:{args.port}"
            logger.info("advertising self as %s (local IP toward central)", advertise)
    # Surface GPU usability up front: a worker that can't use CUDA will silently
    # serve every model on CPU. Make that loud so it's not mistaken for "slow".
    _gpus = detect_gpus()
    _cuda = torch_cuda_status()
    _lcpp = llama_cpp_cuda_status()
    if _cuda.get("available"):
        logger.info("torch CUDA ready: %s (torch %s, cuda %s) — transformers models use the GPU",
                    _cuda.get("device_name"), _cuda.get("torch_version"),
                    _cuda.get("cuda_version"))
    elif _gpus:
        logger.warning(
            "GPU(s) detected by nvidia-smi (%s) but torch.cuda.is_available() is "
            "False — transformers inference will run on CPU. This worker's Python "
            "env needs a CUDA build of torch. torch=%s cuda=%s err=%s",
            ", ".join(g.get("name") or "?" for g in _gpus),
            _cuda.get("torch_version"), _cuda.get("cuda_version"), _cuda.get("error"))
    else:
        logger.warning("no usable GPU (nvidia-smi found none and torch has no CUDA); "
                       "inference will run on CPU")

    # GGUF models go through llama.cpp, which needs its OWN CUDA build.
    if _gpus and _lcpp.get("installed") and _lcpp.get("supports_gpu_offload") is False:
        logger.warning(
            "llama-cpp-python is installed WITHOUT GPU offload support — GGUF "
            "models will run on CPU regardless of n_gpu_layers. Reinstall with "
            "CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install --force-reinstall "
            "--no-cache-dir llama-cpp-python  (llama_cpp %s)", _lcpp.get("version"))
    elif _gpus and _lcpp.get("supports_gpu_offload"):
        logger.info("llama.cpp GPU offload available (llama_cpp %s) — GGUF models "
                    "can use the GPU", _lcpp.get("version"))

    state = WorkerState(name=args.name, url=advertise,
                        worker_id=_load_worker_id(args.id_file),
                        central_url=args.central)
    state.port = args.port
    state.role = args.role

    # Operator runtime settings (console-set) project onto the env FIRST, so
    # the slot supervisor + every other reader sees them; drop-ins lose loudly.
    _apply_settings_env(args)

    # v3 final semantics: on-demand is the DEFAULT tier, so every occupant
    # except a static one may be bumped (LRU promotion) when another model
    # needs a seat — exactly the intent of "slots stay filled, seats change
    # hands on demand". The residency lookup lets the scheduler tell an
    # all-STATIC pool apart from a merely-busy one and fail loads with a
    # clear error instead of evicting.
    try:
        from ..managers.serve.slots import (set_eviction_policy,
                                            set_residency_lookup,
                                            set_fit_check as set_slot_fit_check,
                                            set_make_room as set_slot_make_room)
        set_eviction_policy(lambda mk: _residency(mk) == "on-demand")
        set_residency_lookup(_residency)
        # Real-VRAM ceiling gate (Fix A): the slot load/evict path now consults
        # REAL device free VRAM (ComfyUI-visible), not slot-occupancy count, so a
        # card topped out by an out-of-band process (ComfyUI) triggers an LRU
        # on-demand eviction before seating a new model instead of OOM/under-
        # offloading into a "free" seat on a full card. Degrades to allow when
        # unmeasurable (no-GPU / can't read VRAM) — byte-identical to today.
        set_slot_fit_check(_worker_slot_fit_check)
        # CROSS-TIER make-room (slice 10): once slot-side eviction is exhausted,
        # this reclaims VRAM held by an IN-PROCESS resident (invisible to the slot
        # scheduler) so a slot load isn't OOM'd by a sibling transformers model.
        set_slot_make_room(lambda mk: _vram_evict_to_fit(state, mk))
    except Exception as _exc:  # noqa: BLE001
        logger.warning("slot eviction policy not registered: %s", _exc)

    # CONTEXT-as-allocation (slice 11): register the ctx resolver so the served
    # -c honours the per-model ctx_pct — the value fit/admission reserved KV for.
    # Reverse-injection (serve never imports worker_agent). Returns the resolved
    # ctx int, or None (no ctx_pct) -> serve's default ctx path, byte-identical.
    try:
        from ..managers.serve.serve import set_ctx_resolver

        def _ctx_resolver(mk, cfg=None):
            _cfg = None
            try:
                from ..imports.config.main import get_model_config
                _cfg = get_model_config(mk, dict_return=True)
            except Exception:  # noqa: BLE001
                _cfg = None
            ctx, _pct, _mx = _resolved_ctx(mk, _cfg)
            return ctx
        set_ctx_resolver(_ctx_resolver)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("ctx resolver not registered: %s", _exc)

    # Fix B (2026-07-15): ensure headroom before every ComfyUI gen. comfy_runner
    # is package-shared (central imports it), so it must not import worker/GPU
    # internals — instead the worker registers a hook it calls if present (the
    # same None-default indirection as the slot policy). Evicts on-demand managed
    # models (LRU, via _evict_model) until real free VRAM reaches the target, so
    # ComfyUI's demand is a first-class queue entry, not a silent squatter.
    try:
        from ..managers.comfy.comfy_runner import set_comfy_headroom_hook
        set_comfy_headroom_hook(
            lambda mk, job_id=None: _worker_ensure_comfy_headroom(state, mk, job_id))
    except Exception as _exc:  # noqa: BLE001 — headroom prep must never break boot
        logger.warning("comfy headroom hook not registered: %s", _exc)

    # Env-profiles (stage 1): register the model->profile resolver the runner
    # spawn seam consumes, then KICK materialization of every declared profile
    # not yet ready for its manifest. Materialization is slow (pip) so it runs in
    # the background via a registered executor (register_executor) — a restart
    # shuts it down cleanly, and a profiled model only routes/seats once its
    # profile reads ready in the heartbeat. Boot-driven so the restart-based
    # /ops/config apply re-kicks idempotently (a ready profile is a no-op; a
    # changed manifest re-materializes). Fully additive to the boot path.
    # BOOT-TIME REGISTRY RE-WALK (slice 6). The registry is built ONCE at module
    # import from the discovery REPORT FILE (<DEFAULT_ROOT>/projects/
    # model_discovery.json); nothing on the worker ever re-walks the tree, so an
    # ABSENT or STALE report leaves the registry as STAPLES ONLY — every on-disk
    # model then fails get_model_config and dies in the scan's `no_config` bucket
    # (the ae 2026-07-17 incident: 63 no_config, models_local 65->0). The on-disk
    # dirs carry per-dir hugpy.json markers — the source of truth discover_models
    # reads — so a re-walk HERE re-derives their configs regardless of the report
    # file's state. refresh_registry is idempotent, updates in place, and is what
    # its own docstring says to call on startup; the worker just never did.
    # Guarded: a discovery failure must never ground the boot (registry stays
    # whatever import built). This is the honest presence fix — on-disk models
    # resolve configs from their markers, not from a possibly-stale report.
    try:
        from .imports import models_config as _mc
        _before = len(_mc.MODEL_REGISTRY)
        _mc.refresh_registry(run_discovery=True)
        _after = len(_mc.MODEL_REGISTRY)
        logger.info("boot registry re-walk: %d -> %d model configs "
                    "(on-disk markers re-read; report-file staleness bypassed)",
                    _before, _after)
    except Exception as _exc:  # noqa: BLE001 — discovery must never break boot
        logger.warning("boot registry re-walk skipped (%s) — registry stays as "
                       "import-built; on-disk models may read as no_config", _exc)

    try:
        from ..managers.serve import profiles as _profiles
        _profiles.set_model_resolver(_resolve_model_profile)
        _profiles.materialize_all(_RUNTIME_SETTINGS.get("profiles") or {},
                                  register=register_executor)
    except Exception as _exc:  # noqa: BLE001 — profiles must never break boot
        logger.warning("env-profiles not initialized: %s", _exc)

    # Contention-based residency (doctrine 2026-07-11): an on-demand model stays
    # resident until a NEW load needs its memory — then the LRU on-demand
    # resident yields (never static / gate-busy / slot-backed; 📌pinned DOES
    # yield per 2026-07-15 — pin is designation, not a resource lock). dispatch
    # owns the LRU mechanism; the worker registers the box-specific fit-guard +
    # yield predicate + a post-evict trim so each headroom re-check sees the
    # freed memory. See dispatch.ensure_headroom_for_load; the old idle clock
    # (_residency_sweep_once) is now opt-in behind on_demand_ttl_s.
    try:
        from ..managers.dispatch.dispatch import (set_fit_check, set_evictable,
                                                  set_post_evict_hook,
                                                  set_make_room)
        set_fit_check(_worker_fit_check)
        set_evictable(_worker_evictable)
        set_post_evict_hook(_trim_host_ram)
        # CROSS-TIER VRAM make-room (slice 10): the in-process LRU yield is blind
        # to a slot-child squatter — this hook sees ALL residents (pid-registry
        # measured) and evicts the minimum permissible set through the /ops/evict
        # verb, then REFUSES an unfittable load before any CUDA allocation. Bound
        # to the live state so the eviction verb can act.
        set_make_room(lambda mk: _vram_evict_to_fit(state, mk))
    except Exception as _exc:  # noqa: BLE001
        logger.warning("contention residency hooks not registered: %s", _exc)

    # Worker-local slot pool (SLOT_COUNT; settings > env > default).
    _supervise_slots()

    # F1/F5 wiring: control.cancel on this process's bus reaches the shared
    # job store (fires the cancel handle each live stream attached), and job
    # transitions publish back onto the bus. Same substrate as central.
    try:
        from ..comms import wire_cancel, wire_job_events
        wire_cancel()
        wire_job_events(source=f"worker:{args.name or ''}")
    except Exception as _exc:
        logger.warning("comms bus wiring failed: %s", _exc)

    # role=rpc: launch the llama.cpp rpc-server and advertise this box's GPU as a
    # shard backend. The endpoint host is the same outbound IP we advertise for
    # /infer (reachable from the lead); central stores it as an rpc_servers entry.
    rpc_proc = None
    if args.role == "rpc":
        rpc_proc = _spawn_rpc_server(args)
        rpc_host = _local_ip_toward(args.central) or socket.gethostname()
        state.rpc_endpoint = f"{rpc_host}:{args.rpc_port}"
        logger.info("role=rpc — advertising shard endpoint %s", state.rpc_endpoint)

    client = CentralClient(args.central, token=args.token)
    # Present the SAME enrollment token on central-transfer (model-pull) requests
    # in provision.py, so provisioning keeps working once central turns on
    # HUGPY_WORKER_ENROLL_REQUIRED. No-op (tokenless, exactly today's behavior)
    # when args.token is None.
    from .provision import set_enroll_token, set_worker_id, set_budget_state
    set_enroll_token(args.token)
    # Identify this worker on central-transfer requests so central can apply its
    # per-worker storage budget to BACKGROUND pulls (2026-07-17 handshake).
    set_worker_id(state.worker_id)
    # Register the live state so EVERY in-process transfer entry runs the storage
    # gate even when its caller didn't thread `state` (Part A, slice 8): the
    # /redownload route and any ensure_model_present(..., state=None) now gate
    # against the real limits/assigned instead of pulling atop the cap.
    set_budget_state(state)

    try:
        _register(client, state, args)
    except Exception as exc:
        logger.error("initial registration failed: %s", exc)
        # Keep going — the heartbeat loop will retry, and the server can still
        # serve a worker the operator registers manually.

    hb = threading.Thread(target=_heartbeat_loop, args=(client, state, args), daemon=True)
    hb.start()

    # Residency maintenance (v3): fills empty slots (slice 9) + TTL-yields
    # idle IN-PROCESS on-demand residents. First fill lands sooner than the
    # loop's first 60s tick so a restarted agent's slots don't sit empty.
    threading.Thread(target=_residency_sweep_loop, args=(state,), daemon=True).start()
    threading.Timer(20.0, lambda: _fill_empty_slots(state)).start()

    # UTIL-08 reconcile: failed pulls converge instead of drifting forever.
    threading.Thread(target=_reconcile_loop, args=(state,), daemon=True).start()

    logger.info("worker inference server listening on %s (advertising %s)",
                f"{args.host}:{args.port}", state.url)
    state.args = args   # the /ops endpoints need pkg_name/pkg_index/id_file
    # Build the server explicitly (instead of Flask's app.run) so the restart
    # path holds a handle to close the listening socket cleanly before exit —
    # releasing :9100 so systemd's respawned process binds without a collision.
    # make_server binds immediately, so state.http_server is set before we block.
    from werkzeug.serving import make_server
    state.http_server = make_server(args.host, args.port, build_app(state),
                                    threaded=True)
    state.http_server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
