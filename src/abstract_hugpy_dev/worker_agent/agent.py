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


def torch_cuda_status() -> dict:
    """Whether *torch* can actually use CUDA — distinct from nvidia-smi seeing a
    card. Inference runs on the GPU only when ``torch.cuda.is_available()`` is
    True; a CPU-only torch build (or a torch/CUDA-driver mismatch) leaves a
    perfectly good GPU unused. Surfaced in /health so this is diagnosable.
    """
    try:
        import torch
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


def _llama_cpp_supports_vision() -> bool:
    """Whether this worker's llama.cpp can actually decode images (mtmd).

    True when a multimodal chat handler is importable — the same capability the
    in-process vision runner needs to load an mmproj projector. Central reads this
    (engine.supports_vision) and ONLY routes image turns to workers that report it,
    so an older text-only build is never handed an image to guess at. Without this
    field a worker is treated as text-only and every vision turn falls back to the
    central's local engine — even when the worker has the VL model loaded.
    """
    try:
        from llama_cpp import llama_chat_format as _cf
        for _name in ("Qwen25VLChatHandler", "Llava16ChatHandler",
                      "Llava15ChatHandler", "MiniCPMv26ChatHandler",
                      "MoondreamChatHandler"):
            if hasattr(_cf, _name):
                return True
    except Exception:
        pass
    try:
        import llama_cpp.mtmd_cpp  # noqa: F401
        return True
    except Exception:
        return False


def llama_cpp_cuda_status() -> dict:
    """Whether *llama.cpp* (GGUF backend) was built with GPU offload support.

    ``n_gpu_layers`` is silently ignored when llama-cpp-python is the CPU-only
    wheel, so a GGUF model runs entirely on CPU even though autofit picked GPU
    layers. ``llama_supports_gpu_offload()`` is the definitive build check.
    """
    try:
        import llama_cpp
        supports = None
        try:
            supports = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            pass
        return {
            "installed": True,
            "version": getattr(llama_cpp, "__version__", None),
            "supports_gpu_offload": supports,
            "supports_vision": _llama_cpp_supports_vision(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}


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
                     ("vram_reserve_gib", "HUGPY_VRAM_RESERVE_GIB")):
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
        from ..managers.serve.slots import SlotPool, slots_enabled
        if not slots_enabled():
            return None
        return SlotPool().statuses()
    except Exception:
        return None


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
    models = worker.get("models") or []
    if models == state.assigned_models:
        return
    state.assigned_models = list(models)
    logger.info("assignment updated: serving %s", models or "(nothing)")

    for model_key in models:
        with state._provision_lock:
            if model_key in state._provisioning:
                continue
            state._provisioning.add(model_key)

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
                if not model_is_local(mk):
                    logger.info("pre-provisioning assigned model %s…", mk)
                    ensure_model_present(mk, state.central_url, progress=_prog)
                    logger.info("pre-provisioned %s", mk)
                # Preload / "dedicate": eagerly WARM the model (build+cache the
                # runner so it stays resident) ahead of the first request — so a
                # dedicated/pooled worker answers with no cold-load latency.
                # Default ON when this worker has a pool; force with WORKER_PRELOAD.
                _preload = os.environ.get(
                    "WORKER_PRELOAD",
                    "1" if os.environ.get("WORKER_POOL", "").strip() else "0",
                ).strip().lower() in ("1", "true", "yes", "on")
                if _preload:
                    try:
                        from abstract_hugpy_dev.managers.dispatch.dispatch import runner_for
                        logger.info("preloading (warming) %s…", mk)
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

_SETTINGS_KEYS = {"slot_count"}          # widen deliberately, key by key
_SETTINGS_SOURCE: dict = {}              # key -> "settings" | "env" | "default"


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
    return settings


def _effective_config() -> dict:
    """What this agent is ACTUALLY running with (for the heartbeat)."""
    try:
        from ..managers.serve.slots import _slot_count
        n = _slot_count()
    except Exception:
        n = None
    return {"slot_count": n,
            "slot_count_source": _SETTINGS_SOURCE.get("slot_count", "default")}


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
        for pkg in ("llama-cpp-python", "transformers", "torch"):
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
            worker = client.heartbeat(
                state.worker_id,
                {
                    "gpus": detect_gpus(),
                    "loaded_models": loaded_model_keys(),
                    "loading": _loading_model_keys(),
                    "provisioning": sorted(state._provisioning),
                    "provision_progress": state.provision_snapshot(),
                    "spill": _spill_describe(),
                    "url": state.url,     # None -> central keeps source-IP URL
                    "port": state.port,
                    "pkg_version": _installed_pkg_version(args.pkg_name),
                    "role": state.role,
                    "rpc_endpoint": state.rpc_endpoint,
                    "free_ram": _free_ram_bytes(),
                    "engine": llama_cpp_cuda_status(),
                    "pool": os.environ.get("WORKER_POOL", ""),
                    "caps": _local_caps(),
                    "env": env_status(),
                    "config": _effective_config(),
                    "loaded_detail": _loaded_detail(),
                    "slots": _slot_statuses(),
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

    logger.info("worker inference server listening on %s (advertising %s)",
                f"{args.host}:{args.port}", state.url)
    state.args = args   # the /ops endpoints need pkg_name/pkg_index/id_file
    build_app(state).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
