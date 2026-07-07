"""Worker-side studio render endpoints (GPU offload, option a).

A studio render (``produce_clip`` -> content-addressed clip) runs HERE, on a GPU
worker, while the central VM keeps the whole control plane (media_jobs.db stays
single-writer on central). Central's studio_i2v bus adapter
(``video_intel/runners/studio_i2v.py``) delegates a REAL-model render to
``POST /studio/render``, polls ``GET /studio/render/<job_id>``, and — on done —
``ingest``s the clip from the SHARED content-addressed path (central + worker share
``/mnt/llm_storage``), so there is no b64 round-trip.

This module is import-light on purpose: Flask + stdlib at module top, and the
studio spine (numpy/PIL/torch via ``produce_clip``) is imported LAZILY inside the
render thread. It does NOT import ``worker_agent.agent``, so it can be mounted on a
bare Flask app (the tree's E2E test does exactly that) without the agent's heavy
boot.

Three routes (registered by :func:`register_studio_routes`):
    POST /studio/render            {job_id, spec, central_version?} -> 202 | 409
    GET  /studio/render/<job_id>   -> {status, progress?, result?, pkg_version}
    POST /studio/cancel/<job_id>   -> sets the render's local cancel flag

Concurrency: ONE render at a time. A second ``/studio/render`` (a DIFFERENT job)
while one is running returns 409 (central maps that to a retryable JobError); a
re-POST of the SAME job_id is idempotent. Cancel is cooperative: the local flag
feeds the render thread's ``should_cancel()``, the exact probe ``produce_clip``
already accepts (T1 semantics) — a cancel makes the runner abort BEFORE writing a
clip and return ``Err(CANCELLED)``.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import logging
import threading
import time

from flask import jsonify, request

logger = logging.getLogger(__name__)


def _pkg_version() -> str:
    """This worker's package version — echoed on every response so central can log
    a version skew against its own. Best-effort; 'unknown' if unresolved."""
    try:
        from abstract_hugpy_dev import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


class _RenderJob:
    __slots__ = ("job_id", "status", "result", "progress", "cancel",
                 "started_at", "thread")

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "running"     # running | done | error
        self.result = None         # produce_clip payload dict once settled
        self.progress = None       # coarse live progress blob (or None)
        self.cancel = threading.Event()
        self.started_at = time.time()
        self.thread = None


class StudioRenderManager:
    """Thread-safe, one-render-at-a-time studio render table for a worker.

    In-memory by design: the DURABLE job ledger stays single-writer on central
    (media_jobs.db). This table only tracks the in-flight render so central can
    poll it; a worker restart drops it, which central reads as ``worker_lost``."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, _RenderJob] = {}
        self._active: str | None = None   # in-flight render's job_id, or None

    def start(self, job_id: str, spec_dict: dict) -> tuple[bool, str]:
        """Begin a render in a background thread. Returns ``(ok, reason)``:
        ``(True, "started")`` new render kicked off; ``(True, "exists")`` a re-POST
        of a known job (idempotent); ``(False, "busy")`` a DIFFERENT render is in
        flight (caller returns 409)."""
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return True, "exists"          # idempotent re-POST of a known job
            if self._active is not None:
                active = self._jobs.get(self._active)
                if active is not None and active.status == "running":
                    return False, "busy"        # a different render is in flight
                self._active = None             # stale pointer (prior settled)
            job = _RenderJob(job_id)
            self._jobs[job_id] = job
            self._active = job_id
            t = threading.Thread(
                target=self._run, args=(job, spec_dict),
                name=f"studio-render-{job_id[:8]}", daemon=True)
            job.thread = t
            t.start()
            return True, "started"

    def _run(self, job: "_RenderJob", spec_dict: dict) -> None:
        """Render thread: rebuild the spec through the SAME validating deserializer
        the bus uses, run the SHARED ``run_produce_clip`` (identical to central's
        in-process path), and record the JSON-safe result payload."""
        try:
            # Lazy imports (torch/diffusers/numpy pulled only here, never at mount).
            from ..video_intel.runners.studio_i2v import (
                artifact_result_to_payload,
                run_produce_clip,
            )
            from ..video_intel.studio.job import studio_i2v_from_dict

            spec = studio_i2v_from_dict(spec_dict)
            with self._lock:
                job.progress = {"phase": "rendering", "started_at": job.started_at}
            should_cancel = lambda: job.cancel.is_set()  # noqa: E731
            result = run_produce_clip(spec, should_cancel)
            payload = artifact_result_to_payload(result)
            with self._lock:
                job.result = payload
                job.status = "done"
                job.progress = None
        except Exception as exc:  # noqa: BLE001 — a crash is errors-as-data too
            logger.exception("studio render %s crashed", job.job_id)
            with self._lock:
                job.result = {"ok": False, "error": {
                    "code": "internal",
                    "message": f"{type(exc).__name__}: {exc}",
                    "retryable": False}}
                job.status = "error"
                job.progress = None
        finally:
            with self._lock:
                if self._active == job.job_id:
                    self._active = None

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "status": "unknown",
                        "progress": None, "result": None}
            return {"job_id": job_id, "status": job.status,
                    "progress": dict(job.progress) if job.progress else None,
                    "result": job.result}

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "cancelled": False, "status": "unknown"}
            if job.status != "running":
                return {"job_id": job_id, "cancelled": False, "status": job.status}
            job.cancel.set()
            return {"job_id": job_id, "cancelled": True, "status": "cancelling"}


def register_studio_routes(app, *, manager=None, worker_id=None, worker_name=None):
    """Mount the studio render endpoints on ``app``.

    ``manager`` lets a test inject its own :class:`StudioRenderManager`; production
    (``build_app``) passes ``None`` (a fresh one). ``worker_id`` / ``worker_name``
    are echoed for attribution. Returns the manager so a caller/test can inspect
    it."""
    mgr = manager or StudioRenderManager()
    ver = _pkg_version()

    @app.route("/studio/render", methods=["POST"])
    def studio_render():
        body = request.get_json(silent=True) or {}
        job_id = body.get("job_id")
        spec_dict = body.get("spec")
        if not job_id or not isinstance(spec_dict, dict):
            return jsonify({"ok": False, "error": {
                "code": "BadRequest",
                "message": 'body must include {"job_id": str, "spec": {...}}'},
                "pkg_version": ver}), 400
        central_version = str(body.get("central_version") or "")
        if central_version and central_version != ver:
            logger.warning("studio offload VERSION SKEW: central=%s worker=%s "
                           "(job %s) — running anyway; behavior may differ",
                           central_version, ver, job_id)
        ok, reason = mgr.start(job_id, spec_dict)
        if not ok:
            return jsonify({"ok": False, "error": {
                "code": "Busy",
                "message": "another studio render is in flight (one at a time)"},
                "pkg_version": ver, "worker_id": worker_id}), 409
        return jsonify({"ok": True, "job_id": job_id, "accepted": reason,
                        "pkg_version": ver, "worker_id": worker_id,
                        "worker_name": worker_name}), 202

    @app.route("/studio/render/<job_id>", methods=["GET"])
    def studio_render_status(job_id):
        st = mgr.status(job_id)
        st["pkg_version"] = ver
        st["worker_id"] = worker_id
        return jsonify(st), 200

    @app.route("/studio/cancel/<job_id>", methods=["POST"])
    def studio_render_cancel(job_id):
        out = mgr.cancel(job_id)
        out["pkg_version"] = ver
        return jsonify(out), 200

    logger.info("studio render endpoints mounted (worker=%s, pkg=%s)",
                worker_name or worker_id or "?", ver)
    return mgr
