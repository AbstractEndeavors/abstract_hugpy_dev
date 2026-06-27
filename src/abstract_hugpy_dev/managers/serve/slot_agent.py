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
HEALTH_TIMEOUT = float(os.environ.get("SLOT_HEALTH_TIMEOUT", "180"))


def _build_cmd(model_key, n_gpu_layers=None, ctx=None, threads=None):
    """argv for the child llama-server + the resolved (ngl, ctx) actually used."""
    from .serve import (
        _model_file_for, _ctx_for, get_model_config,
        LLAMA_SERVER_BIN, DEFAULT_LLAMA_THREADS,
    )
    from ..spill import autofit_gpu_layers

    cfg = get_model_config(model_key)
    path = _model_file_for(model_key, cfg)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{model_key}: no GGUF on disk (resolved {path!r})")

    # Autofit from the VRAM free RIGHT NOW, so later slots take what's left.
    ngl = n_gpu_layers if n_gpu_layers is not None else autofit_gpu_layers(path)
    ctx = int(ctx) if ctx else _ctx_for(cfg, model_key)
    threads = int(threads) if threads else DEFAULT_LLAMA_THREADS

    argv = [
        LLAMA_SERVER_BIN, "-m", path,
        "--host", "127.0.0.1", "--port", str(SLOT_CHILD_PORT),
        "--n-gpu-layers", str(ngl), "-c", str(ctx), "-t", str(threads),
    ]
    # Vision GGUF: load the multimodal projector so /v1/chat/completions accepts
    # image_url content. No-op for text models (no projector beside the model).
    from ...imports.src.utils import find_mmproj
    mmproj = find_mmproj(path)
    if mmproj:
        argv += ["--mmproj", mmproj]
        logger.info("slot %s: vision model — loading projector %s", SLOT_ID, mmproj)
    return argv, ngl, ctx


class Slot:
    """Owns at most one llama-server child."""

    def __init__(self):
        self.model_key = None
        self.proc = None
        self.ngl = None
        self.ctx = None
        self.loaded_at = 0.0
        self.last_used = 0.0
        self.lock = threading.Lock()
        self.child_base = f"http://127.0.0.1:{SLOT_CHILD_PORT}"

    # -- health ------------------------------------------------------------
    def _child_alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def healthy(self) -> bool:
        if not self._child_alive():
            return False
        try:
            import httpx
            return httpx.get(self.child_base + "/health", timeout=2.0).status_code == 200
        except Exception:
            return False

    def status(self) -> dict:
        from ..spill import free_vram_bytes
        return {
            "slot_id": SLOT_ID,
            "control_port": SLOT_PORT,
            "child_port": SLOT_CHILD_PORT,
            "endpoint": f"http://{SLOT_ADVERTISE}:{SLOT_PORT}",
            "model_key": self.model_key,
            "healthy": self.healthy(),
            "n_gpu_layers": self.ngl,
            "ctx": self.ctx,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "free_vram_bytes": free_vram_bytes(),
        }

    # -- lifecycle ---------------------------------------------------------
    def load(self, model_key, n_gpu_layers=None, ctx=None, threads=None) -> dict:
        with self.lock:
            if self.model_key == model_key and self.healthy():
                self.last_used = time.time()
                return self.status()

            self._kill()
            argv, self.ngl, self.ctx = _build_cmd(model_key, n_gpu_layers, ctx, threads)
            logger.info("slot %s loading %s (ngl=%s ctx=%s): %s",
                        SLOT_ID, model_key, self.ngl, self.ctx, " ".join(argv))

            env = dict(os.environ)
            if MAIN_GPU is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(MAIN_GPU)
            self.proc = subprocess.Popen(argv, env=env)
            self.model_key = model_key
            self.loaded_at = self.last_used = time.time()

            if not self._wait_healthy():
                self._kill()
                self.model_key = None
                raise RuntimeError(
                    f"slot {SLOT_ID}: {model_key} did not become healthy in "
                    f"{HEALTH_TIMEOUT:.0f}s")
            logger.info("slot %s ready: %s on %s", SLOT_ID, model_key, self.child_base)
            return self.status()

    def _wait_healthy(self) -> bool:
        deadline = time.time() + HEALTH_TIMEOUT
        while time.time() < deadline:
            if not self._child_alive():
                return False
            if self.healthy():
                return True
            time.sleep(1.0)
        return False

    def unload(self) -> dict:
        with self.lock:
            self._kill()
            self.model_key = self.ngl = self.ctx = None
            return self.status()

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
        try:
            return jsonify(slot.load(body["model_key"], body.get("n_gpu_layers"),
                                     body.get("ctx"), body.get("threads")))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/unload", methods=["POST"])
    def unload():
        return jsonify(slot.unload())

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

        client = httpx.Client(timeout=None)
        upstream = client.send(
            client.build_request(request.method, url, content=body, headers=headers),
            stream=True,
        )

        def generate():
            try:
                for chunk in upstream.iter_raw():
                    yield chunk
            finally:
                upstream.close()
                client.close()

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
