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

from flask import Flask, request, jsonify, Response, stream_with_context

logger = logging.getLogger("abstract_hugpy_dev.worker_agent")
from .imports import *
from ..central import central_base_url
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

    Reserve-adjusted (managers.spill honors HUGPY_RAM_RESERVE_GIB) so central
    plans against RAM this box can actually spare — local processes central
    can't see keep their headroom."""
    try:
        from ..managers.spill import free_ram_bytes
        return free_ram_bytes()
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
def _ensure_present(payload: dict, central_url: str | None) -> None:
    """Provision the requested model before inference (central-first, HF fallback)."""
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
        ensure_model_present(payload.get("model_key"), central_url)
    except Exception as exc:
        logger.warning("provisioning check for %s failed: %s", model_key, exc)


def _ensure_present_streaming(payload: dict, central_url: str | None):
    """Provision the model, yielding SSE 'status' events with download progress.

    Yields encoded SSE lines (status/error). Returns normally once the model is
    present (or was already). Throttled so we don't flood the stream.
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
                result["ok"] = ensure_model_present(model_key, central_url, progress=_progress)
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
        out.append({
            "kind": "slot", "model_key": mk,
            "slot_id": s.get("slot_id"), "healthy": s.get("healthy"),
            "busy": s.get("busy"), "endpoint": s.get("endpoint"),
            "rss_bytes": s.get("rss_bytes"),
            "n_gpu_layers": s.get("n_gpu_layers"), "ctx": s.get("ctx"),
            "vram_bytes": vram_bytes, "device": device,
        })
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
        return {"type": "done", "finish_reason": getattr(ev, "finish_reason", "stop")}
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

        return sorted({mk for (mk, _task) in _loaded()})
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


def _reap_scan(state: "WorkerState") -> dict:
    """The reaper's read-only survey: which local model files are RECLAIMABLE
    (on disk, but not assigned / loaded / loading / pinned), and which are
    PROTECTED (and why). Never touches comfy rows — those are symlinks into the
    operator's ComfyUI, not this worker's downloads.

    This is advisory: /reap re-checks every guard at delete time, because state
    (an assign, a load, a pull) can change between preview and reclaim.
    """
    try:
        from .imports import get_models_dict, get_model_config, get_model_path
        from .provision import model_is_local, _on_shared_model_store, _model_store_reapable
    except Exception as exc:  # noqa: BLE001
        return {"reclaimable": [], "protected": [], "error": str(exc)}

    assigned = set(state.assigned_models or [])
    # HARDEN (reaper guard): fold in slot-seated / answering models.
    # loaded_model_keys() is in-process only and MISSES models seated in the
    # slot pool that are actively serving a request. Union _slot_occupants()
    # so an approved reap can never delete a resident/answering model even if
    # it slipped onto the approved list between preview and reclaim.
    loaded = set(loaded_model_keys()) | _slot_occupants()
    loading = set(_loading_model_keys())

    reclaimable, protected = [], []
    try:
        # Enumerate the UNION, not just get_models_dict(): on a WORKER that dict is
        # the built-in staples + comfy sweep + on-disk discovery report and NEVER
        # includes MODEL_REGISTRY, so models held purely by CENTRAL ASSIGNMENT
        # (discovered rows — a designated gguf like flux2) were surveyed as ABSENT,
        # dropping tens of GB from the storage report though they're on disk, in
        # models_local, and slot-seated. Fold in assigned + loaded + loading; the
        # loop below skips any key that isn't model_is_local, so this only ADDS
        # on-disk models the staple-only dict missed.
        keys = set(get_models_dict().keys()) | assigned | loaded | loading
    except Exception as exc:  # noqa: BLE001
        return {"reclaimable": [], "protected": [], "error": str(exc)}

    for mk in keys:
        try:
            cfg = get_model_config(mk)
        except Exception:
            continue
        if getattr(cfg, "framework", None) == "comfy":
            continue  # symlinks / operator-owned — reaper stays clear
        try:
            if not model_is_local(mk):
                continue
        except Exception:
            continue
        path = ""
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
        if _pinned(mk):
            protected.append({"model_key": mk, "bytes": size, "why": "pinned"})
        elif mk in assigned:
            protected.append({"model_key": mk, "bytes": size, "why": "assigned"})
        elif mk in loaded or mk in loading:
            protected.append({"model_key": mk, "bytes": size, "why": "loaded"})
        else:
            reclaimable.append({"model_key": mk, "bytes": size, "path": path})

    reclaimable.sort(key=lambda r: r["bytes"], reverse=True)
    return {
        "reclaimable": reclaimable,
        "protected": protected,
        "reclaimable_bytes": sum(r["bytes"] for r in reclaimable),
    }


def _reap_reclaim(state: "WorkerState", model_keys: list[str]) -> dict:
    """Delete the local files of the named models — but ONLY after re-proving,
    per key, that it is still reclaimable (not assigned/loaded/loading/pinned,
    not comfy). The guard is re-run here, not trusted from a stale preview."""
    from .imports import get_model_config, get_model_path
    from .provision import model_is_local, wipe_model

    assigned = set(state.assigned_models or [])
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
        if _pinned(mk):
            results.append({"model_key": mk, "ok": False, "reason": "pinned"})
            continue
        if mk in assigned:
            results.append({"model_key": mk, "ok": False, "reason": "assigned"})
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
        try:
            freed = _path_bytes(get_model_path(mk) or "")
        except Exception:
            freed = 0
        gone = wipe_model(mk)  # jailed against root/home/short paths
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


def _storage_model_row(mk: str, size: int, loaded: set, loading: set,
                       provisioning: set, assigned: set,
                       why_hint: str = "") -> dict:
    """One per-model row for the heartbeat storage view: bytes + every
    protection flag + a human `why`. loaded is ALREADY answer-inclusive
    (loaded_model_keys() ∪ _slot_occupants()) at the caller."""
    is_pinned = _pinned(mk)
    is_loaded = mk in loaded
    is_loading = mk in loading
    is_provisioning = mk in provisioning
    is_assigned = mk in assigned
    protected = (is_pinned or is_loaded or is_loading
                 or is_provisioning or is_assigned)
    # Precedence mirrors the reaper's guard order (pinned > assigned > loaded).
    if is_pinned:
        why = "pinned"
    elif is_assigned:
        why = "assigned"
    elif is_loaded:
        why = "loaded"
    elif is_loading:
        why = "loading"
    elif is_provisioning:
        why = "provisioning"
    else:
        why = why_hint or ""
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
    out = {
        "cache_used_bytes": cache_used,
        "disk_free": int(disk.get("free_bytes", 0) or 0),
        "models": models,
    }
    _STORAGE_CACHE.update(at=now, value=out)
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
            _ensure_present(payload, state.central_url)
            return jsonify(_run_once(payload))
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

        def _generate():
            yield _sse({"type": "request", "request_id": req_id})
            # Stream provisioning progress first (download from central/HF), then
            # generation with auto-continuation. Both emit SSE lines already.
            yield from _ensure_present_streaming(payload, state.central_url)
            yield from _stream_sync(payload, request_id=req_id)

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
        # Respond first, then re-exec: the caller needs the ack before the
        # process replaces itself. Persistent worker-id -> same registry row.
        from .._platform.procutil import reexec
        threading.Timer(0.5, reexec).start()
        return jsonify({"ok": True, "restarting": True,
                        "worker_id": state.worker_id})

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
            from .._platform.procutil import reexec
            threading.Timer(0.5, reexec).start()
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
        for _tkey in ("on_demand_ttl_s", "reconcile_interval_s"):
            if _tkey in body:
                try:
                    tval = int(body[_tkey])
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue", "message": f"{_tkey} must be an integer"}}), 400
                if not 60 <= tval <= 86400:
                    return jsonify({"ok": False, "error": {
                        "code": "BadValue", "message": f"{_tkey} must be 60..86400"}}), 400
                settings[_tkey] = tval
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
        _save_settings(args, settings)
        logger.info("ops/config: persisted %s — re-exec to apply", settings)
        from .._platform.procutil import reexec
        threading.Timer(0.5, reexec).start()
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
        after = _free_vram_bytes()
        freed = (after - before) if (before is not None and after is not None) else None
        return jsonify({
            "ok": err is None,
            "evicted": evicted,
            "model_key": model_key,
            "error": err,
            "vram_free_before": before,
            "vram_free_after": after,
            "freed": freed,
            "loaded_models": loaded_model_keys(),
        })

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
            ok = ensure_model_present(model_key, state.central_url)
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
        # Learn the model from central (if needed), make sure its files are
        # present, then build the runner, which loads the model. A tiny run
        # confirms it can actually generate.
        from .provision import ensure_model_present, ensure_model_registered
        canonical = ensure_model_registered(model_key, state.central_url) or model_key
        ensure_model_present(canonical, state.central_url)

        #from abstract_hugpy_dev.managers.dispatch import runner_for
        runner_for(model_key=canonical)  # builds + caches the runner (loads weights)

        after = _free_vram_bytes()
        used = (before - after) if (before is not None and after is not None) else None
        result.update(
            ok=True,
            vram_free_after=after,
            vram_used=used,
            # If GPU free memory dropped meaningfully, weights are on the GPU.
            fit=bool(used and used > 64 * 1024 * 1024),
        )
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
        # key -> {done_bytes, total_bytes, frac}; populated while a background
        # pre-provision downloads, so central (and the console) can show a %.
        self._provision_progress: dict[str, dict] = {}
        self._provision_lock = threading.Lock()

    def provision_snapshot(self) -> dict:
        """A lock-safe copy of per-model download progress for the heartbeat."""
        with self._provision_lock:
            return {k: dict(v) for k, v in self._provision_progress.items()}


def _sync_assignment(state: "WorkerState", worker: dict) -> None:
    """React to central's worker record: adopt its model list and pre-provision.

    Central owns the assignment (set in the UI). The agent reads it back from
    every register/heartbeat response and, for any newly-assigned model it
    doesn't already have, downloads it in the background so the first chat
    doesn't pay the full download latency. Without this the worker never knew
    about UI allocation changes.
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

    for model_key in models:
        _kick_provision(state, model_key)
    # Slice 9: already-local models can be seated right now — don't wait for
    # the maintenance tick. Background thread: fills block on slot loads.
    threading.Thread(target=_fill_empty_slots, args=(state,), daemon=True).start()


def _kick_provision(state: "WorkerState", model_key: str) -> None:
    """Provision (and per-policy preload) ONE assigned model in the background.

    Shared by assignment adoption and the UTIL-08 reconcile loop; the
    _provisioning guard makes concurrent kicks a no-op."""
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
                    ensure_model_present(mk, state.central_url, progress=_prog)
                    logger.info("pre-provisioned %s", mk)
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
                if _has_slots:
                    try:
                        _fill_empty_slots(state)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("post-provision slot fill failed: %s", exc)
                elif _preload or _res == "static":
                    # STATIC is always-on: eager-warm regardless of the
                    # preload gate — it must never wait for a first request.
                    try:
                        from abstract_hugpy_dev.managers.dispatch.dispatch import runner_for
                        logger.info("preloading (warming) %s…%s", mk,
                                    " [static — forced]" if (_res == "static" and not _preload) else "")
                        runner_for(model_key=mk)   # builds + caches -> resident
                        logger.info("preloaded %s (resident)", mk)
                    except Exception as exc:
                        logger.warning("preload of %s failed: %s", mk, exc)
            except Exception as exc:
                logger.warning("pre-provision of %s failed: %s", mk, exc)
            finally:
                with state._provision_lock:
                    state._provisioning.discard(mk)
                    state._provision_progress.pop(mk, None)

        threading.Thread(target=_bg, daemon=True).start()


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
    """Every reconcile_interval_s (default 600): any assigned model that is
    NOT local and NOT already provisioning gets its provisioning re-kicked.
    Converges failed pulls instead of drifting until the next assignment
    change; the _provisioning guard + single-flight lock keep it idempotent."""
    while True:
        time.sleep(max(60, int(_RUNTIME_SETTINGS.get("reconcile_interval_s", 600))))
        try:
            local = set(_models_local(state))
            for mk in list(state.assigned_models):
                with state._provision_lock:
                    busy = mk in state._provisioning
                if mk not in local and not busy:
                    logger.warning("reconcile: assigned model %s is missing on "
                                   "disk — re-kicking provisioning", mk)
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
    from importlib import metadata
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return None


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
                  "reconcile_interval_s", "pinned", "comfy_url"}   # widen key by key
_SETTINGS_SOURCE: dict = {}              # key -> "settings" | "env" | "default"
_RUNTIME_SETTINGS: dict = {}             # the loaded settings, for live readers
_COMFY_URL_BASE_ENV = "_HUGPY_COMFY_URL_BASE"  # sentinel: the pre-projection
# COMFY_URL (systemd drop-in / env / none), captured once and carried across
# os.execv so clearing the setting reverts to the real base, never the last
# projected value.


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
        or yielded, eager-warmed; permanent when combined with 📌 pin.

    "Serving" is purely a STATE (a model in a slot), never a policy.
    """
    val = (_RUNTIME_SETTINGS.get("residency") or {}).get(model_key)
    return "static" if val == "static" else "on-demand"


def _pinned(model_key: str) -> bool:
    """📌 pin (tiers v3 semantics, operator-locked 2026-07-05): PERMANENT
    ATTRIBUTION of the model to this worker — central refuses unassign while
    pinned, files are never reaped, and residency overrides survive. (Reaper
    enforcement pending; advertised in the heartbeat config meanwhile.)"""
    return bool((_RUNTIME_SETTINGS.get("pinned") or {}).get(model_key))


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
    operator's console-set values — and unit drop-ins lose, loudly."""
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
    return settings


def _effective_config() -> dict:
    """What this agent is ACTUALLY running with (for the heartbeat)."""
    try:
        from ..managers.serve.slots import _slot_count
        n = _slot_count()
    except Exception:
        n = None
    out = {"slot_count": n,
           "slot_count_source": _SETTINGS_SOURCE.get("slot_count", "default"),
           "on_demand_ttl_s": int(_RUNTIME_SETTINGS.get("on_demand_ttl_s", 900))}
    if _RUNTIME_SETTINGS.get("residency"):
        out["residency"] = dict(_RUNTIME_SETTINGS["residency"])
    if _RUNTIME_SETTINGS.get("pinned"):
        out["pinned"] = dict(_RUNTIME_SETTINGS["pinned"])
    out["comfy_url"] = (os.environ.get("COMFY_URL")
                        or "http://127.0.0.1:8188").rstrip("/")
    out["comfy_url_source"] = _SETTINGS_SOURCE.get("comfy_url", "default")
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
    "vram_bytes"}. ``vram_bytes`` is ComfyUI's REAL GPU footprint from nvidia-smi
    (per-process), or null — never on-disk checkpoint bytes (the 0.1.137 guard)."""
    now = time.time()
    if now - _COMFY_CACHE["at"] < 60.0:
        out = _COMFY_CACHE["value"]
    else:
        url = (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
        out = {"available": False, "url": url}
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
    """One pass of the TTL sweep (factored out of the loop so it's testable).

    v3 final semantics: the sweep applies ONLY to IN-PROCESS residents. The
    default policy is on-demand, so any non-static in-process model idle
    longer than on_demand_ttl_s is evicted (dispatch.evict cascades to the
    llama singleton — RAM/VRAM actually frees; the op-style slot-less box
    keeps this behavior). SLOT occupants are EXEMPT: slots stay filled
    (slice 9) and a seat changes hands only via LRU promotion or explicit
    unload. Static never yields anywhere."""
    ttl = int(_RUNTIME_SETTINGS.get("on_demand_ttl_s", 900))
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

        candidates = [mk for mk in state.assigned_models
                      if mk not in occupied and mk in local
                      and _framework(mk) == "gguf"]
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
                runner_for(model_key=mk)         # seats itself via endpoint_for
            except Exception as exc:  # noqa: BLE001 — one seat must not block the rest
                logger.warning("slot fill for %s failed: %s", mk, exc)
    finally:
        _SLOT_FILL_LOCK.release()


def _residency_sweep_loop(state: "WorkerState") -> None:
    """Residency maintenance every 60s: fill empty slots (slice 9), then
    TTL-yield idle in-process on-demand residents."""
    started_at = time.time()
    while True:
        time.sleep(60.0)
        try:
            _fill_empty_slots(state)
        except Exception as exc:  # noqa: BLE001 — the loop must never die
            logger.warning("slot fill pass failed: %s", exc)
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


def _self_update_if_needed(required: str | None, args) -> None:
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
        # Terminate supervised slots BEFORE re-exec: orphaned slots would be
        # adopted by the new agent and keep serving the OLD code forever.
        # The fresh agent respawns them on the new version.
        _kill_slots()
        from .._platform.procutil import reexec
        reexec()
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


def _heartbeat_loop(client: CentralClient, state: WorkerState, args) -> None:
    while True:
        time.sleep(args.heartbeat)
        try:
            # Compute slot statuses ONCE — the unified allocations view reuses it.
            _slots = _slot_statuses()
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
                    "pkg_version": _installed_pkg_version(args.pkg_name),
                    "role": state.role,
                    "rpc_endpoint": state.rpc_endpoint,
                    "free_ram": _free_ram_bytes(),
                    "ram_total": _ram_total_bytes(),
                    "disk": _disk_status(),
                    "engine": llama_cpp_cuda_status(),
                    "pool": os.environ.get("WORKER_POOL", ""),
                    "caps": _local_caps(),
                    "env": env_status(),
                    "config": _effective_config(),
                    "comfy": _comfy_status(),
                    "loaded_detail": _loaded_detail(),
                    "slots": _slots,
                    "allocations": _allocations(_slots),
                    "storage": _worker_storage(state),
                },
            )
            # Adopt any assignment change made in the UI + pre-provision it.
            _sync_assignment(state, worker)
            # Adopt central's resource limits (min of central + local config).
            _apply_central_limits(worker)
            # Converge to central's required package version (re-execs on update).
            _self_update_if_needed((worker or {}).get("required_pkg_version"), args)
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
        "pkg_version": _installed_pkg_version(args.pkg_name),
        "engine": llama_cpp_cuda_status(),
        "pool": os.environ.get("WORKER_POOL", ""),
        "caps": _local_caps(),
        "env": env_status(),
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
    logger.info("registered as worker id=%s serving models=%s", state.worker_id, worker.get("models"))
    # Converge to central's required package version before serving (re-execs).
    _self_update_if_needed(worker.get("required_pkg_version"), args)


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
                                            set_residency_lookup)
        set_eviction_policy(lambda mk: _residency(mk) == "on-demand")
        set_residency_lookup(_residency)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("slot eviction policy not registered: %s", _exc)

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
    build_app(state).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
