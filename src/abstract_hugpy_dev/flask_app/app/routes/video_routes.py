### routes/video_routes.py
"""HTTP surface (Phase 3a) for the Video Intelligence crop feature.

Additive, all-JSON routes over the already-verified headless backbone
(`abstract_hugpy_dev.video_intel`). This module only translates HTTP <-> the
backbone; every invariant (metadata resolution, axis validity, single-writer
job state) lives in the backbone and is reused here, never re-implemented.

Frozen contract (a frontend is being built to the same contract in parallel):
    POST /video/ingest              {"path": "<abspath under /uploads>"} -> MediaRef
    POST /video/jobs/crop           {"source": <MediaRef>, "spatial"?, "temporal"?}
                                    -> {"job_id": ...}
    POST /video/jobs/frame_extract  {"source": <MediaRef>, "fps", "quality", "fmt", ...}
                                    -> {"job_id": ...}
    POST /video/jobs/audio_extract  {"source": <MediaRef>, "fmt"?: "wav"}
                                    -> {"job_id": ...}
    POST /video/jobs/generate_image {"parts": [...], "model_id", ...} -> {"job_id": ...}
    GET  /video/presets             -> {"presets": [ {"id","name","description",
                                    "mode","model_key","defaults":{...},
                                    "recommended"} ]}
    POST /video/presets/<id>/apply  -> auto-pick a GPU worker + assign + warm the
                                    preset's model -> {"ok","worker","model_key",
                                    "mode","defaults":{...},"warming"}
    GET  /video/jobs/<job_id>       -> {"job_id","status","result"}
    GET  /video/media?handle=       -> raw file bytes (source image OR job result)

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
from abstract_hugpy_dev.video_intel.frame_schema import make_frame_extract
from abstract_hugpy_dev.video_intel.audio_schema import make_audio_extract
from abstract_hugpy_dev.video_intel.gen_schema import (
    GenPromptPart,
    make_generate_image,
)
from abstract_hugpy_dev.video_intel.scene_schema import make_generate_scene
from abstract_hugpy_dev.video_intel.movie_schema import GoalInterval, make_movie
from abstract_hugpy_dev.video_intel.chains import (
    resolve_video_parts,
    resolve_video_parts_scene,
    resolve_video_parts_movie,
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
# 2b) POST /video/jobs/frame_extract — validate + enqueue a frame-extract job
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/frame_extract", methods=["POST"])
def video_frame_extract():
    body = request.get_json(silent=True) or {}
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef"}), 400
    win = body.get("window")
    try:
        source = make_media_ref(**source_d)
        window = TemporalRegion(**win) if win is not None else None
        spec = make_frame_extract(
            source=source,
            fps=body.get("fps"),
            quality=body.get("quality"),
            fmt=body.get("fmt"),
            window=window,
            max_frames=body.get("max_frames"),
        )
    except (ValueError, TypeError) as exc:  # bad fields / axis combo = 400
        return jsonify({"error": str(exc)}), 400
    job_id = media_bus.enqueue("frame_extract", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2b') POST /video/jobs/audio_extract — validate + enqueue an audio-extract job
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/audio_extract", methods=["POST"])
def video_audio_extract():
    body = request.get_json(silent=True) or {}
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef"}), 400
    try:
        source = make_media_ref(**source_d)
        spec = make_audio_extract(
            source=source,
            fmt=body.get("fmt", "wav"),
        )
    except (ValueError, TypeError) as exc:  # bad fields / non-video source = 400
        return jsonify({"error": str(exc)}), 400
    job_id = media_bus.enqueue("audio_extract", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2c) POST /video/jobs/generate_image — validate + resolve video parts + enqueue
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/generate_image", methods=["POST"])
def video_generate_image():
    body = request.get_json(silent=True) or {}
    parts_in = body.get("parts")
    if not isinstance(parts_in, list) or not parts_in:
        return jsonify({"error": "missing or empty 'parts' list"}), 400
    try:
        parts = []
        for pd in parts_in:
            if not isinstance(pd, dict):
                raise ValueError("each part must be an object")
            media_d = pd.get("media")
            media = make_media_ref(**media_d) if isinstance(media_d, dict) else None
            parts.append(GenPromptPart(
                kind=pd.get("kind"),
                text=pd.get("text"),
                media=media,
            ))
        spec = make_generate_image(
            parts=tuple(parts),
            model_id=body.get("model_id"),
            width=body.get("width"),
            height=body.get("height"),
            steps=body.get("steps"),
            guidance=body.get("guidance"),
            seed=body.get("seed"),
            negative=body.get("negative"),
            strength=body.get("strength"),   # img2img (additive, optional)
            project=body.get("project"),     # auto-archive NAME (optional)
        )
    except (ValueError, TypeError) as exc:  # bad fields / part combo = 400
        return jsonify({"error": str(exc)}), 400

    # Resolve any VIDEO parts to image frames BEFORE enqueue (Phase 7 chain), so
    # the runner never sees a video part. Extraction failure -> 400 (see chains).
    try:
        resolved = resolve_video_parts(spec)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = media_bus.enqueue("generate_image", resolved)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2d) POST /video/jobs/generate_scene — one query -> N frames (+ optional mp4)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/generate_scene", methods=["POST"])
def video_generate_scene():
    body = request.get_json(silent=True) or {}
    parts_in = body.get("parts")
    if not isinstance(parts_in, list) or not parts_in:
        return jsonify({"error": "missing or empty 'parts' list"}), 400
    try:
        parts = []
        for pd in parts_in:
            if not isinstance(pd, dict):
                raise ValueError("each part must be an object")
            media_d = pd.get("media")
            media = make_media_ref(**media_d) if isinstance(media_d, dict) else None
            parts.append(GenPromptPart(
                kind=pd.get("kind"),
                text=pd.get("text"),
                media=media,
            ))
        spec = make_generate_scene(
            parts=tuple(parts),
            model_id=body.get("model_id"),
            width=body.get("width"),
            height=body.get("height"),
            steps=body.get("steps"),
            guidance=body.get("guidance"),
            n_frames=body.get("n_frames"),
            fps=body.get("fps"),
            assemble=body.get("assemble"),
            seed=body.get("seed"),
            motion=body.get("motion"),
            negative=body.get("negative"),
            # img2img additive knobs (optional; absent -> factory defaults:
            # strength None -> runner 0.45; chain defaults True).
            strength=body.get("strength"),
            chain=body.get("chain", True),
            project=body.get("project"),     # auto-archive NAME (optional)
        )
    except (ValueError, TypeError) as exc:  # bad fields / part combo / frame_cap = 400
        return jsonify({"error": str(exc)}), 400

    # Resolve any VIDEO parts to image frames BEFORE enqueue (chain twin), so the
    # runner never sees a video part. Extraction failure -> 400 (see chains).
    try:
        resolved = resolve_video_parts_scene(spec)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = media_bus.enqueue("generate_scene", resolved)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2d') POST /video/jobs/generate_movie — a GOAL TIMELINE -> a stitched movie
# --------------------------------------------------------------------------- #
# Mirrors generate_scene: parse body -> build GoalIntervals -> make_movie
# (validates contiguity/ranges) -> resolve any video goal refs -> enqueue. The
# scene-template fields (model/size/steps/…) are shared by every segment; `goals`
# is the ordered, contiguous, non-overlapping timeline; the director knobs turn on
# optional per-segment vision scoring + retry.
@video_bp.route("/video/jobs/generate_movie", methods=["POST"])
def video_generate_movie():
    body = request.get_json(silent=True) or {}
    goals_in = body.get("goals")
    if not isinstance(goals_in, list) or not goals_in:
        return jsonify({"error": "missing or empty 'goals' list"}), 400
    try:
        goals = []
        for gd in goals_in:
            if not isinstance(gd, dict):
                raise ValueError("each goal must be an object")
            ref_d = gd.get("ref")
            ref = make_media_ref(**ref_d) if isinstance(ref_d, dict) else None
            goals.append(GoalInterval(
                start_frame=gd.get("start_frame"),
                end_frame=gd.get("end_frame"),
                prompt=gd.get("prompt"),
                ref=ref,
            ))
        spec = make_movie(
            goals=tuple(goals),
            model_id=body.get("model_id"),
            width=body.get("width"),
            height=body.get("height"),
            steps=body.get("steps"),
            guidance=body.get("guidance"),
            fps=body.get("fps"),
            assemble=body.get("assemble"),
            seed=body.get("seed"),
            negative=body.get("negative"),
            strength=body.get("strength"),
            chain=body.get("chain", True),
            project=body.get("project"),
            # director knobs (optional; absent -> factory defaults)
            vision_enabled=body.get("vision_enabled", False),
            score_threshold=body.get("score_threshold", 60),
            max_attempts_per_segment=body.get("max_attempts_per_segment", 1),
            judge_model_id=body.get("judge_model_id"),
            time_budget_s=body.get("time_budget_s"),
        )
    except (ValueError, TypeError) as exc:  # bad fields / contiguity / frame_cap = 400
        return jsonify({"error": str(exc)}), 400

    # Resolve any VIDEO goal refs to a representative still BEFORE enqueue, so the
    # runner never sees a video ref. Extraction failure -> 400 (see chains).
    try:
        resolved = resolve_video_parts_movie(spec)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = media_bus.enqueue("generate_movie", resolved)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2e) GET /video/presets — curated "ideal default loads" for scene generation
# --------------------------------------------------------------------------- #
# Thin idiom (mirrors prompt_routes' GET /prompt/tasks): import the static
# registry, dump it as JSON. No side effects — a preset is just a named bundle
# of a model_key + coherence mode + per-frame defaults the UI pre-fills.
@video_bp.route("/video/presets", methods=["GET"])
def video_presets():
    from abstract_hugpy_dev.video_intel.presets import available_presets
    return jsonify({"presets": [p.to_dict() for p in available_presets()]}), 200


# --------------------------------------------------------------------------- #
# GPU-worker auto-pick — reuses the worker registry helpers (workers_for_model /
# _has_usable_gpu / _worker_fit) that the /llm/workers/<id>/assign path uses, so
# a preset lands on the same class of worker an operator would pick by hand.
# --------------------------------------------------------------------------- #
def _pick_gpu_worker(model_key):
    """Choose an online, approved, GPU-capable worker to warm ``model_key`` on.

    Preference order (best first): a worker that already carries the model
    (assigned or loaded — no reload), then one where it fits VRAM outright, then
    any where it at least fits (VRAM+RAM), then the most free VRAM. Returns None
    when no GPU worker is eligible (the caller maps that to a 409 NoGpuWorker).
    """
    from ..functions.imports.utils.workers import (
        worker_store, _has_usable_gpu, _engine_unusable,
    )
    from .worker_routes import _worker_fit

    # Workers already serving this model (assigned OR loaded), stale beats ok —
    # reusing one avoids a multi-GB reload. online_only=False so a briefly-stale
    # assignee still counts as "already has it".
    warm_ids = {w["id"] for w in
                worker_store.workers_for_model(model_key, online_only=False)}

    eligible = []
    for w in worker_store.all():
        # Same admission/engine/liveness gates workers_for_model applies, plus a
        # hard GPU requirement (a preset's "recommended: gpu" is load-bearing).
        if w.get("admission") != "approved":
            continue
        if _engine_unusable(w):
            continue
        if w.get("status") != "online":
            continue
        if not _has_usable_gpu(w):
            continue
        eligible.append(w)
    if not eligible:
        return None

    def _rank(w):
        fit = _worker_fit(model_key, w)   # fit/gpu_resident None for unsizable models
        return (
            0 if w["id"] in warm_ids else 1,
            0 if fit.get("gpu_resident") else 1,
            0 if fit.get("fit") is not False else 1,
            -(fit.get("vram_free") or 0),
            w.get("id", ""),
        )

    eligible.sort(key=_rank)
    return eligible[0]


# --------------------------------------------------------------------------- #
# 2f) POST /video/presets/<preset_id>/apply — validate + auto-pick a GPU worker,
#     assign + background-warm the model, return the gen defaults for the UI.
#     Guards mirror workers_assign: catalog membership (404) + central-holds-
#     files (409); adds preset-exists (404) and no-GPU-worker (409) on top.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/presets/<preset_id>/apply", methods=["POST"])
def video_preset_apply(preset_id):
    from abstract_hugpy_dev.video_intel.presets import get_preset

    preset = get_preset(preset_id)
    if preset is None:
        return jsonify({"ok": False, "error": {
            "code": "UnknownPreset",
            "message": f"no video preset {preset_id!r}"}}), 404

    model_key = preset.model_key

    # Catalog membership — same source (get_models_dict) workers_assign checks.
    from abstract_hugpy_dev.imports.config.models.models_config import get_models_dict
    if model_key not in get_models_dict(dict_return=True):
        return jsonify({"ok": False, "error": {
            "code": "UnknownModel",
            "message": f"preset model {model_key!r} is not in the catalog"}}), 404

    # Item-4 invariant: central must hold the files, or the worker silently pulls
    # from HF at internet speed. Reuse the exact guard workers_assign uses.
    from .worker_routes import _central_missing_reason, _kick_warm
    missing = _central_missing_reason(model_key)
    if missing:
        return jsonify({"ok": False, "error": {
            "code": "CentralMissing",
            "message": (f"central does not have {model_key!r} on disk ({missing}) "
                        "— download it on the Models tab first; workers provision "
                        "from central")}}), 409

    # Auto-pick a GPU-capable worker (presets are "recommended: gpu").
    worker = _pick_gpu_worker(model_key)
    if worker is None:
        return jsonify({"ok": False, "error": {
            "code": "NoGpuWorker",
            "message": ("no online GPU-capable worker is available to warm this "
                        "preset — bring a GPU worker online or assign manually")}}), 409

    # Designate = ready: assign then background-warm (never wait on the load).
    from ..functions.imports.utils.workers import assign_model
    assigned = assign_model(worker["id"], model_key)
    if assigned is None:
        # Raced: the worker vanished between pick and assign.
        return jsonify({"ok": False, "error": {
            "code": "NoGpuWorker",
            "message": "the selected worker is no longer available — retry"}}), 409
    _kick_warm(assigned, [model_key], "video-preset")

    return jsonify({
        "ok": True,
        "worker": {"name": assigned.get("name"), "id": assigned.get("id")},
        "model_key": model_key,
        "mode": preset.mode,
        "defaults": preset.defaults(),
        "warming": True,
    }), 200


# --------------------------------------------------------------------------- #
# 2g) GET /movie/presets — curated MOVIE TEMPLATES for the Movie Maker tab
# --------------------------------------------------------------------------- #
# Movie twin of GET /video/presets: import the static registry, dump it as JSON.
# No side effects — a movie template is just a named bundle of a model_key + the
# scene-template settings + a goal timeline (contiguous half-open intervals that
# tile [0, total)) the Movie Maker tab pre-fills into its goal editor.
@video_bp.route("/movie/presets", methods=["GET"])
def movie_presets():
    from abstract_hugpy_dev.video_intel.presets import available_movie_presets
    return jsonify({"presets": [p.to_dict() for p in available_movie_presets()]}), 200


# --------------------------------------------------------------------------- #
# 2h) POST /movie/presets/<preset_id>/apply — return the directly-POSTable
#     generate_movie body for this template (unknown id -> 404). Read-only/open,
#     same auth posture as GET /movie/presets: unlike the video apply this does
#     NOT touch the worker plane — a movie template just pre-fills the goal editor
#     (its `request` sub-object is a curated /video/jobs/generate_movie body).
# --------------------------------------------------------------------------- #
@video_bp.route("/movie/presets/<preset_id>/apply", methods=["POST"])
def movie_preset_apply(preset_id):
    from abstract_hugpy_dev.video_intel.presets import get_movie_preset

    preset = get_movie_preset(preset_id)
    if preset is None:
        return jsonify({"ok": False, "error": {
            "code": "UnknownPreset",
            "message": f"no movie preset {preset_id!r}"}}), 404

    return jsonify(preset.apply()), 200


# --------------------------------------------------------------------------- #
# 3) GET /video/jobs/<job_id> — read-only job view (unknown id -> null view)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>", methods=["GET"])
def video_job_status(job_id):
    # media_bus.get returns {"job_id","status":null,"result":null} for an unknown
    # id, so the poller can distinguish "not yet / unknown" from a real status.
    return jsonify(media_bus.get(job_id)), 200


# --------------------------------------------------------------------------- #
# 3b) POST /video/jobs/<job_id>/cancel — cooperative cancel
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>/cancel", methods=["POST"])
def video_job_cancel(job_id):
    # queued jobs die outright; a running scene stops BETWEEN frames (mid-frame
    # inference is never interrupted). Idempotent — cancelling a terminal or
    # unknown job reports cancelled=False.
    return jsonify(media_bus.cancel(job_id)), 200


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
