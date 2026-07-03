### routes/video_routes.py
"""HTTP surface (Phase 3a) for the Video Intelligence crop feature.

Four additive, all-JSON routes over the already-verified headless backbone
(`abstract_hugpy_dev.video_intel`). This module only translates HTTP <-> the
backbone; every invariant (metadata resolution, axis validity, single-writer
job state) lives in the backbone and is reused here, never re-implemented.

Frozen contract (a frontend is being built to the same contract in parallel):
    POST /video/ingest        {"path": "<abspath under /uploads>"} -> MediaRef
    POST /video/jobs/crop     {"source": <MediaRef>, "spatial"?, "temporal"?}
                              -> {"job_id": ...}
    GET  /video/jobs/<job_id> -> {"job_id","status","result"}
    GET  /video/media?handle= -> raw file bytes (source image OR crop result)

Mirrors upload_routes' blueprint idiom: `get_bp(...)` mints the Blueprint (the
same shared helper the other *_bp modules use, re-exported through
flask_app.app.functions). Imported directly here so this module stays
self-contained and can be registered on a minimal standalone app for headless
verification without booting the full wsgi stack.
"""
from __future__ import annotations

import dataclasses
import mimetypes
import os

from flask import request, jsonify, send_file

from abstract_flask import get_bp

from abstract_hugpy_dev.imports.src.constants.constants import (
    UPLOADS_HOME,
    DEFAULT_ROOT,
)
from abstract_hugpy_dev.video_intel import media_store, media_bus
from abstract_hugpy_dev.video_intel.media_schema import make_media_ref
from abstract_hugpy_dev.video_intel.crop_schema import (
    SpatialRegion,
    TemporalRegion,
    make_crop,
)

video_bp, logger = get_bp("video_bp", __name__)


# --------------------------------------------------------------------------- #
# storage jail — same realpath-under-roots check as media_store._is_within,
# replicated here so a route never touches a path outside the storage roots.
# --------------------------------------------------------------------------- #
def _is_within(path: str, root: str) -> bool:
    if not root:
        return False
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    try:
        return os.path.commonpath([rp, rr]) == rr
    except ValueError:
        return False


def _jail_resolve(handle):
    """Resolve a caller-supplied path to a realpath under UPLOADS_HOME or
    DEFAULT_ROOT. Returns the resolved realpath, or None if it escapes the jail
    (or is missing/ill-typed) — the single seam that keeps these routes from
    becoming an arbitrary-file-read/write."""
    if not handle or not isinstance(handle, str):
        return None
    rp = os.path.realpath(handle)
    if _is_within(rp, UPLOADS_HOME) or _is_within(rp, DEFAULT_ROOT):
        return rp
    return None


# --------------------------------------------------------------------------- #
# 1) POST /video/ingest — resolve metadata ONCE, mint a MediaRef
# --------------------------------------------------------------------------- #
@video_bp.route("/video/ingest", methods=["POST"])
def video_ingest():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    resolved = _jail_resolve(path)
    if resolved is None:
        return jsonify({"error": f"path missing or outside storage jail: {path!r}"}), 400
    try:
        ref = media_store.ingest(resolved)
    except Exception as exc:  # ingest raises locally (FileNotFound/Value/Runtime)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    return jsonify(dataclasses.asdict(ref)), 200


# --------------------------------------------------------------------------- #
# 2) POST /video/jobs/crop — validate + enqueue a crop job
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/crop", methods=["POST"])
def video_crop():
    body = request.get_json(silent=True) or {}
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef"}), 400
    sp = body.get("spatial")
    tp = body.get("temporal")
    try:
        source = make_media_ref(**source_d)
        spatial = SpatialRegion(**sp) if sp is not None else None
        temporal = TemporalRegion(**tp) if tp is not None else None
        spec = make_crop(source=source, spatial=spatial, temporal=temporal)
    except (ValueError, TypeError) as exc:  # invalid axis combo / bad fields = 400
        return jsonify({"error": str(exc)}), 400
    job_id = media_bus.enqueue("crop", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 3) GET /video/jobs/<job_id> — read-only job view (unknown id -> null view)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>", methods=["GET"])
def video_job_status(job_id):
    # media_bus.get returns {"job_id","status":null,"result":null} for an unknown
    # id, so the poller can distinguish "not yet / unknown" from a real status.
    return jsonify(media_bus.get(job_id)), 200


# --------------------------------------------------------------------------- #
# 4) GET /video/media?handle=<abspath> — serve raw bytes for a MediaRef uri
# --------------------------------------------------------------------------- #
@video_bp.route("/video/media", methods=["GET"])
def video_media():
    handle = request.args.get("handle")
    resolved = _jail_resolve(handle)
    if resolved is None or not os.path.isfile(resolved):
        return jsonify({"error": "not found"}), 404
    mime, _ = mimetypes.guess_type(resolved)
    return send_file(
        resolved,
        mimetype=mime or "application/octet-stream",
        conditional=True,
    )
