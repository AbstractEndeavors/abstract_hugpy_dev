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
import secrets

from flask import request, jsonify, send_file

from abstract_flask import get_bp

from abstract_hugpy_dev.imports.src.constants.constants import (
    UPLOADS_HOME,
    DEFAULT_ROOT,
)
from abstract_hugpy_dev.video_intel import media_store, media_bus, identity_profiles
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
from abstract_hugpy_dev.video_intel.studio.job import make_studio_i2v
from abstract_hugpy_dev.video_intel.studio_movie_schema import (
    _DEFAULT_CONTEXT_FRAMES as _DEFAULT_MOVIE_CONTEXT_FRAMES,
    StudioMovieGoal,
    make_studio_movie,
)
from abstract_hugpy_dev.video_intel.identity_reconstruction_schema import (
    DEFAULT_VIEWS as _DEFAULT_RECON_VIEWS,
    make_identity_reconstruction,
)
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


def _resolve_asset_uri(asset_id):
    """Resolve a media-catalog ``asset_id`` to the ``uri`` (abs path) of the produced
    ref that carries it — the B2 chain lookup so the console can hand the studio a
    prior tier's output BY ID (``source_asset_id``) rather than a path. Scans the
    media-bus job store (the same durable catalog ``/video/studio/clips`` reads) for a
    job result whose ``outputs[*].asset_id`` matches, newest first, and returns that
    output's uri. Read-only ``mode=ro`` connection (mirrors the clips-list route); the
    subsequent jail + ffprobe validation in the caller still guards the returned path.
    Returns the uri string, or None (unknown id / no catalog / transient lock)."""
    if not asset_id or not isinstance(asset_id, str):
        return None
    import json as _json
    import sqlite3
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT result_json FROM media_jobs "
                "WHERE result_json IS NOT NULL ORDER BY updated DESC LIMIT 1000"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    for (result_json,) in rows:
        if not result_json:
            continue
        try:
            res = _json.loads(result_json)
        except (ValueError, TypeError):
            continue
        for o in (res.get("outputs") or []):
            if (isinstance(o, dict) and o.get("asset_id") == asset_id
                    and isinstance(o.get("uri"), str) and o["uri"]):
                return o["uri"]
    return None


def _autofit_vram_budget(raw):
    """A BLANK/absent/null ``vram_budget_gb`` means AUTOFIT (return ``None``): the studio
    render sizes the routing budget to the SERVING WORKER's measured free VRAM at render
    time, rather than a low guess that is guaranteed to fail (operator doctrine 2026-07-12:
    "if a model needs 14GB and it's blank, just do 14, otherwise a fail is 100% likely").
    An EXPLICIT number is the manual override — passed through untouched (a bad value still
    400s in the validating factory). None threads through the spec to render_clip, which
    resolves it; no worker/VRAM data degrades to the historical synthetic default."""
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return raw


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
# 2d'') POST /video/studio/i2v — a studio image-to-video clip via the cinema
#        studio spine (B2). Mirrors the movie/scene routes: parse body -> build
#        the validated StudioI2VSpec -> media_bus.enqueue -> {job_id}. The job
#        runs through the studio's own router->manifest->runner->content-addressed
#        clip path (produce_clip) and its output is cataloged in the media store.
#        Query it exactly like any other media job: GET /video/jobs/<job_id>.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/i2v", methods=["POST"])
def video_studio_i2v():
    body = request.get_json(silent=True) or {}
    # resolution may arrive nested ({"resolution": {"width","height","fps"}}) or as
    # flat top-level keys — accept both (nested wins, mirrors the frontend contract).
    res = body.get("resolution") if isinstance(body.get("resolution"), dict) else {}
    width = res.get("width", body.get("width"))
    height = res.get("height", body.get("height"))
    fps = res.get("fps", body.get("fps"))
    # sane studio default so an empty POST still produces a clip (synthetic spine).
    width = 512 if width is None else width
    height = 512 if height is None else height
    fps = 24 if fps is None else fps
    # capability defaults to "i2v" (backward-compat); "t2v" (text-to-video) and any
    # other Capability value are accepted and validated inside make_studio_i2v.
    capability = body.get("capability", "i2v")
    # a start_image, if supplied, must resolve inside the storage jail (never an
    # arbitrary-file read) — same seam as /video/ingest. T2V is TEXT-ONLY, so a
    # start_image is meaningless for it: we DELIBERATELY IGNORE it (drop to None),
    # never jail-resolve or reject it — a t2v clip is a pure function of prompt +
    # seed + geometry. i2v (the default) is unaffected.
    start_image = body.get("start_image")
    if capability == "t2v":
        start_image = None
    elif start_image is not None:
        start_image = _jail_resolve(start_image)
        if start_image is None:
            return jsonify({"error": "start_image outside storage jail"}), 400
    # source_video (B2 movie->studio chain): the prior tier's clip this studio job
    # extends. Accept EITHER an absolute "source_video" path (jail-resolved like
    # start_image) OR a "source_asset_id" resolved to its uri via the media catalog.
    # An i2v job with a source but no start_image extends the clip from its LAST FRAME
    # (the runner does the extraction); t2v is text-only, so a source is meaningless
    # and DELIBERATELY DROPPED. A non-video / nonexistent / jail-escaping target is a
    # clean 4xx here rather than a deferred runner failure.
    source_video = body.get("source_video")
    source_asset_id = body.get("source_asset_id")
    if capability == "t2v":
        source_video = None
    else:
        if source_video is None and source_asset_id:
            source_video = _resolve_asset_uri(source_asset_id)
            if source_video is None:
                return jsonify(
                    {"error": f"source_asset_id not found in catalog: {source_asset_id!r}"}), 404
        if source_video is not None:
            resolved_sv = _jail_resolve(source_video)
            if resolved_sv is None:
                return jsonify({"error": "source_video outside storage jail"}), 400
            if not os.path.isfile(resolved_sv):
                return jsonify({"error": "source_video not found"}), 404
            # Authoritative video check: ffprobe-classify via media_store (probe wins).
            try:
                sref = media_store.ingest(resolved_sv, kind_hint="video")
            except Exception as exc:  # unreadable / no A/V stream / jail = bad input
                return jsonify(
                    {"error": f"source_video is not a readable media file: {exc}"}), 400
            if sref.kind != "video":
                return jsonify(
                    {"error": f"source_video is not a video (classified as {sref.kind})"}), 400
            source_video = sref.uri
    # IDENTITY LOCK (id_lock): reference image(s) of the subject preserved across the
    # render (Wan VACE reference-to-video). Each is jail-resolved + ffprobe/PIL-classified
    # as an IMAGE (a non-image / jail-escaping / missing target is a clean 4xx here, not a
    # deferred runner failure). Up to 4 (all consumed by diffusers 0.39). Permitted for
    # the VACE capabilities (id_lock / v2v — the runners that consume them); rejected for
    # any other capability so a non-VACE runner can never silently ignore them.
    _REF_CAPS = {"id_lock", "v2v"}
    # UNIFIED IDENTITY (2026-07-12): an enqueue may name a saved identity profile
    # instead of raw reference_images; the profile's curated set is canonical. Its
    # OWN bounded block (a concurrent vram_budget_gb edit lives elsewhere in this
    # route — keep these seams from entangling).
    reference_images_in, _prof_err = _reference_images_from_body(body)
    if _prof_err is not None:
        _pl, _st = _prof_err
        return jsonify(_pl), _st
    resolved_refs: list = []
    if reference_images_in is not None:
        if not isinstance(reference_images_in, list):
            return jsonify({"error": "reference_images must be a list of paths"}), 400
        if capability not in _REF_CAPS:
            return jsonify({"error": "reference_images require capability id_lock or v2v; "
                                     f"got {capability!r}"}), 400
        if len(reference_images_in) > 4:
            return jsonify({"error": "at most 4 reference_images are accepted"}), 400
        for raw in reference_images_in:
            if not isinstance(raw, str) or not raw.strip():
                return jsonify({"error": "each reference_image must be a non-empty path"}), 400
            rp = _jail_resolve(raw)
            if rp is None:
                return jsonify({"error": "reference_image outside storage jail"}), 400
            if not os.path.isfile(rp):
                return jsonify({"error": "reference_image not found"}), 404
            try:
                iref = media_store.ingest(rp, kind_hint="image")
            except Exception as exc:  # unreadable / not a real image = bad input
                return jsonify(
                    {"error": f"reference_image is not a readable media file: {exc}"}), 400
            if iref.kind != "image":
                return jsonify(
                    {"error": f"reference_image is not an image (classified as {iref.kind})"}), 400
            resolved_refs.append(iref.uri)
    # ROUTE RULE: capability id_lock REQUIRES >=1 reference image.
    if capability == "id_lock" and not resolved_refs:
        return jsonify(
            {"error": "capability id_lock requires at least one reference_image"}), 400

    # OPTIONAL VACE control still (composition blocking) — ONLY valid with id_lock. A
    # single image + its kind (pose|depth|sketch), jail-resolved + image-classified.
    control_image = body.get("control_image")
    control_kind = body.get("control_kind")
    if (control_image is not None or control_kind is not None) and capability != "id_lock":
        return jsonify(
            {"error": "control_image/control_kind are only valid with capability id_lock"}), 400
    if control_image is not None:
        if not isinstance(control_image, str) or not control_image.strip():
            return jsonify({"error": "control_image must be a non-empty path"}), 400
        if control_kind not in ("pose", "depth", "sketch"):
            return jsonify({"error": "control_kind must be one of pose|depth|sketch"}), 400
        cp = _jail_resolve(control_image)
        if cp is None:
            return jsonify({"error": "control_image outside storage jail"}), 400
        if not os.path.isfile(cp):
            return jsonify({"error": "control_image not found"}), 404
        try:
            cref = media_store.ingest(cp, kind_hint="image")
        except Exception as exc:
            return jsonify(
                {"error": f"control_image is not a readable media file: {exc}"}), 400
        if cref.kind != "image":
            return jsonify(
                {"error": f"control_image is not an image (classified as {cref.kind})"}), 400
        control_image = cref.uri
    elif control_kind is not None:
        return jsonify({"error": "control_kind requires a control_image"}), 400
    # SAMPLER OVERRIDES (route passthrough): optional "steps"/"cfg" numbers that PIN the
    # denoise settings (explicit values ALWAYS win over the bound model's family
    # default). Validate ranges HERE for a clean 400 with a precise message (steps 1-100,
    # cfg 0-20); make_studio_i2v re-checks the same bounds for non-route callers. An
    # integral float steps (e.g. 30.0 from a JS number input) is accepted as 30.
    steps = body.get("steps")
    if isinstance(steps, float) and steps.is_integer():
        steps = int(steps)
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool)
                              or not (1 <= steps <= 100)):
        return jsonify({"error": "steps must be an integer in [1, 100]"}), 400
    cfg = body.get("cfg")
    if cfg is not None and (not isinstance(cfg, (int, float)) or isinstance(cfg, bool)
                            or not (0 <= cfg <= 20)):
        return jsonify({"error": "cfg must be a number in [0, 20]"}), 400
    # DIRECT MODEL CHOICE (pin): optional "model_id". Threaded into the spec -> the
    # CapabilityRequest pin. The router binds THAT model or returns a clear Err-as-data
    # (PINNED_MODEL_UNAVAILABLE / a sharpened gate reason) that rides back on the job —
    # never a silent fallback. Shape (non-empty string) is checked in make_studio_i2v.
    model_id = body.get("model_id")
    try:
        spec = make_studio_i2v(
            capability=capability,
            width=width,
            height=height,
            fps=fps,
            # AUTOFIT: blank/absent/null -> None (size to the serving worker's free VRAM);
            # an explicit number is the manual override (unchanged).
            vram_budget_gb=_autofit_vram_budget(body.get("vram_budget_gb")),
            seed=body.get("seed", 0),
            out_root=body.get("out_root"),
            start_image=start_image,
            # C-prompt: accept "negative_prompt" (canonical) with backward-compat to
            # the older "negative" key; "prompt" carries the positive text prompt.
            negative=body.get("negative_prompt", body.get("negative")),
            prompt=body.get("prompt"),
            project=body.get("project"),     # auto-archive NAME (optional)
            # B2 chain: the validated abs path of the prior tier's clip (or None).
            source_video=source_video,
            # Sampler overrides + model pin (validated above / shape-checked in factory).
            steps=steps,
            cfg=cfg,
            model_id=model_id,
            # IDENTITY LOCK (id_lock): the validated reference image uris + optional VACE
            # control still (both already jail-resolved + image-classified above).
            reference_images=tuple(resolved_refs),
            control_image=control_image,
            control_kind=control_kind,
        )
    except (ValueError, TypeError) as exc:  # bad geometry / capability / overrides = 400
        return jsonify({"error": str(exc)}), 400
    job_id = media_bus.enqueue("studio_i2v", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2d''') POST /video/studio/movie — a STUDIO MOVIE (an ordered strip of studio
#        clips conjoined at splice points, like an NLE row) via the studio spine.
#        Mirrors /video/studio/i2v: parse body -> build the validated
#        StudioMovieSpec (a take-tree of segment NODES) -> media_bus.enqueue ->
#        {job_id}. The job runs through runners/studio_movie.py, which renders each
#        segment INLINE through the same produce_clip spine and stitches the strip
#        (non-destructive trims honored at concat). Query it exactly like any other
#        media job: GET /video/jobs/<job_id>.
#
#        Ergonomics: a goal's segment_id auto-fills to "seg_NN" and its
#        parent_segment_id auto-chains to the previous node when omitted, so a
#        minimal body ({"goals":[{"prompt":...},{"prompt":..., "branch_frame":10}]})
#        forms a valid LINEAR chain. An explicitly-supplied id/parent is passed
#        through and re-validated by make_studio_movie (a broken chain -> 400).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/movie", methods=["POST"])
def video_studio_movie():
    body = request.get_json(silent=True) or {}
    # resolution nested ({"resolution": {...}}) or flat top-level keys (nested wins),
    # mirroring the i2v route. Sane studio defaults so a minimal POST still renders.
    res = body.get("resolution") if isinstance(body.get("resolution"), dict) else {}
    width = res.get("width", body.get("width"))
    height = res.get("height", body.get("height"))
    fps = res.get("fps", body.get("fps"))
    width = 512 if width is None else width
    height = 512 if height is None else height
    fps = 24 if fps is None else fps

    goals_in = body.get("goals")
    if not isinstance(goals_in, list) or not goals_in:
        return jsonify({"error": "goals must be a non-empty list of segment nodes"}), 400

    # Build the take-tree nodes, auto-filling segment_id + the linear parent chain
    # when omitted (an explicit value is passed through + re-validated in the factory).
    goals = []
    prev_id = None
    for i, g in enumerate(goals_in):
        if not isinstance(g, dict):
            return jsonify({"error": f"goals[{i}] must be an object"}), 400
        seg_id = g.get("segment_id") or f"seg_{i:02d}"
        if i == 0:
            parent = g.get("parent_segment_id")  # None expected; factory enforces root
        else:
            parent = g.get("parent_segment_id", prev_id)
        goals.append(StudioMovieGoal(
            segment_id=seg_id,
            prompt=g.get("prompt"),
            parent_segment_id=parent,
            branch_frame=g.get("branch_frame"),
            negative=g.get("negative"),
            seed=g.get("seed"),
            model_id=g.get("model_id"),
            steps=g.get("steps"),
            cfg=g.get("cfg"),
            # JOINT MODE + context frames (VACE-extend splice motion-carry). joint_mode
            # defaults to "still" (absent/null/"" -> "still"); a bad non-empty value / a
            # root vace_extend / an out-of-range context_frames is a clean 400 via
            # make_studio_movie's validation below.
            joint_mode=(g.get("joint_mode") or "still"),
            context_frames=g.get("context_frames"),
        ))
        prev_id = seg_id

    # SEGMENT 0 conditioning still (optional): a jail-resolved + image-classified
    # start_image path, OR a start_image_asset_id resolved via the media catalog.
    # A jail-escaping / nonexistent / non-image target is a clean 4xx here rather
    # than a deferred runner failure. When absent, segment 0 renders t2v.
    start_ref = None
    start_image = body.get("start_image")
    start_asset_id = body.get("start_image_asset_id")
    if start_image is None and start_asset_id:
        start_image = _resolve_asset_uri(start_asset_id)
        if start_image is None:
            return jsonify(
                {"error": f"start_image_asset_id not found in catalog: {start_asset_id!r}"}), 404
    if start_image is not None:
        rp = _jail_resolve(start_image)
        if rp is None:
            return jsonify({"error": "start_image outside storage jail"}), 400
        if not os.path.isfile(rp):
            return jsonify({"error": "start_image not found"}), 404
        try:
            start_ref = media_store.ingest(rp, kind_hint="image")
        except Exception as exc:  # unreadable / not a real image = bad input
            return jsonify(
                {"error": f"start_image is not a readable media file: {exc}"}), 400
        if start_ref.kind != "image":
            return jsonify(
                {"error": f"start_image is not an image (classified as {start_ref.kind})"}), 400

    # IDENTITY LOCK (id_lock): movie-level subject reference image(s). When present the movie
    # is an IDENTITY MOVIE — the runner renders EVERY segment capability id_lock (Wan-VACE
    # reference-to-video) so the locked subject carries across scene changes. Accept EITHER a
    # list of jailed "reference_images" paths OR "reference_image_asset_ids" (resolved via the
    # media catalog). Each is jail-resolved + ffprobe/PIL-classified as an IMAGE (a non-image /
    # jail-escaping / missing target is a clean 4xx here, not a deferred runner failure). Up to
    # 4 (all consumed by diffusers 0.39). Mirrors /video/studio/i2v's reference handling.
    # UNIFIED IDENTITY (2026-07-12): accept identity_profile:<slug> (canonical) as an
    # alternative to raw reference_images / reference_image_asset_ids. Own bounded block.
    reference_images_in, _prof_err = _reference_images_from_body(body)
    if _prof_err is not None:
        _pl, _st = _prof_err
        return jsonify(_pl), _st
    reference_asset_ids = body.get("reference_image_asset_ids")
    if reference_images_in is None and isinstance(reference_asset_ids, list):
        reference_images_in = []
        for aid in reference_asset_ids:
            uri = _resolve_asset_uri(aid)
            if uri is None:
                return jsonify(
                    {"error": f"reference_image_asset_id not found in catalog: {aid!r}"}), 404
            reference_images_in.append(uri)
    resolved_refs: list = []
    if reference_images_in is not None:
        if not isinstance(reference_images_in, list):
            return jsonify({"error": "reference_images must be a list of paths"}), 400
        if len(reference_images_in) > 4:
            return jsonify({"error": "at most 4 reference_images are accepted"}), 400
        for raw in reference_images_in:
            if not isinstance(raw, str) or not raw.strip():
                return jsonify({"error": "each reference_image must be a non-empty path"}), 400
            rp = _jail_resolve(raw)
            if rp is None:
                return jsonify({"error": "reference_image outside storage jail"}), 400
            if not os.path.isfile(rp):
                return jsonify({"error": "reference_image not found"}), 404
            try:
                iref = media_store.ingest(rp, kind_hint="image")
            except Exception as exc:  # unreadable / not a real image = bad input
                return jsonify(
                    {"error": f"reference_image is not a readable media file: {exc}"}), 400
            if iref.kind != "image":
                return jsonify(
                    {"error": f"reference_image is not an image (classified as {iref.kind})"}), 400
            resolved_refs.append(iref.uri)

    try:
        spec = make_studio_movie(
            goals=tuple(goals),
            width=width,
            height=height,
            fps=fps,
            # AUTOFIT: blank/absent/null -> None (each segment sizes to the serving worker's
            # free VRAM at render time); an explicit number is the manual override.
            vram_budget_gb=_autofit_vram_budget(body.get("vram_budget_gb")),
            seed=body.get("seed", 0),
            # C-prompt: "negative_prompt" (canonical) with back-compat to "negative".
            negative=body.get("negative_prompt", body.get("negative")),
            model_id=body.get("model_id"),
            steps=body.get("steps"),
            cfg=body.get("cfg"),
            project=body.get("project"),
            out_root=body.get("out_root"),
            start_image=start_ref,
            time_budget_s=body.get("time_budget_s"),
            # Movie-level default trailing-frame count for vace_extend splices (a node may
            # override via its own context_frames). Absent -> the schema default (8).
            context_frames=body.get("context_frames", _DEFAULT_MOVIE_CONTEXT_FRAMES),
            # IDENTITY LOCK: the validated reference image uris (jail-resolved + image-classified
            # above). Non-empty -> an identity movie (every segment renders id_lock).
            reference_images=tuple(resolved_refs),
        )
    except (ValueError, TypeError) as exc:  # bad node / geometry / chain = 400
        return jsonify({"error": str(exc)}), 400
    job_id = media_bus.enqueue("generate_studio_movie", spec)
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
# 2i) GET /video/studio/presets — curated STUDIO clip presets for the Studio Clips
#     station. Studio twin of GET /video/presets + GET /movie/presets: import the
#     static registry, dump it as JSON. No side effects — a studio preset is a named
#     bundle of a capability ("i2v"/"t2v") + geometry + a routing vram_budget_gb the
#     station pre-fills into its generate affordance. The model is NOT pinned here;
#     the studio router resolves capability + resolution + budget at run time.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/presets", methods=["GET"])
def studio_presets():
    from abstract_hugpy_dev.video_intel.studio_presets import available_studio_presets
    return jsonify({"presets": [p.to_dict() for p in available_studio_presets()]}), 200


# --------------------------------------------------------------------------- #
# 2j) POST /video/studio/presets/<preset_id>/apply — return the directly-POSTable
#     /video/studio/i2v body for this preset (unknown id -> 404). Read-only/open,
#     same posture as the movie apply: unlike the video-preset apply this does NOT
#     touch the worker plane — a studio preset just pre-fills the generate affordance
#     (its `request` sub-object is a curated /video/studio/i2v body).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/presets/<preset_id>/apply", methods=["POST"])
def studio_preset_apply(preset_id):
    from abstract_hugpy_dev.video_intel.studio_presets import get_studio_preset

    preset = get_studio_preset(preset_id)
    if preset is None:
        return jsonify({"ok": False, "error": {
            "code": "UnknownPreset",
            "message": f"no studio preset {preset_id!r}"}}), 404

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
    if mime is None:
        # A studio clip can be served here BY URI (cross-station library playback) from
        # an extensionless media-store path — guess_type then returns None and the
        # generic octet-stream fallback makes the console <video> show a gray unknown
        # mime. A file under the studio clips dir is always an mp4, so prefer that.
        from abstract_hugpy_dev.video_intel.studio.job import DEFAULT_CLIPS_ROOT
        if _is_within(os.path.realpath(resolved), DEFAULT_CLIPS_ROOT):
            mime = "video/mp4"
    return send_file(
        resolved,
        mimetype=mime or "application/octet-stream",
        conditional=True,
    )


# --------------------------------------------------------------------------- #
# 5) GET /video/studio/clip/<job_id> — STREAM a produced studio clip (slice #3).
#     Convenience twin of /video/media for studio i2v: resolve the clip path from
#     the media-bus job result (the content-addressed clip cataloged by the studio
#     runner) BY JOB ID, so the console viewer plays a clip without ever handling a
#     filesystem path. Range-aware (send_file conditional=True) so an HTML5 <video>
#     can seek. Path-traversal guard: the resolved realpath MUST live under the
#     studio clips dir — a job whose output escapes that tree (or any non-studio /
#     non-done job) is refused, so this can only ever serve a cataloged studio clip.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clips", methods=["GET"])
def video_studio_clips():
    # DURABLE recent-clips list for the console viewer (slice #3). Sources the media
    # CATALOG (the media-bus job store) rather than the comms /llm/jobs view, which
    # only retains terminal rows for ~600s — so a clip produced an hour ago still
    # lists here. Read-only projection: job_id + status + the first output's display
    # metadata (asset_id/geometry/duration). The clip bytes are NEVER referenced by
    # path in the response — playback is by job_id through /video/studio/clip/<id>,
    # which owns the jail. Reads media_bus's own DB_PATH via a read-only connection
    # (mirrors media_bus.get's read); it does not mutate the bus or its schema.
    #
    # `archived_at IS NULL` excludes ARCHIVED clips (POST .../archive) — this is
    # the fix for "removed clips just reappear": this list IS the catalog (it is
    # DB-driven, not a filesystem walk), so a clip hidden here via media_bus.archive
    # stays hidden across every poll, unlike the old client-only "remove" the UI
    # used to do (which this same query's next 6s tick silently resurrected).
    import json as _json
    import sqlite3

    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))

    clips = []
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT job_id, status, result_json, created, updated "
                "FROM media_jobs WHERE name='studio_i2v' AND archived_at IS NULL "
                "ORDER BY updated DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        # No DB yet / transient lock (or a not-yet-migrated archived_at column on
        # a DB no write path has touched this process) -> an empty list is the
        # honest answer, same posture as before this feature.
        rows = []

    for job_id, status, result_json, created, updated in rows:
        out = None
        if result_json:
            try:
                res = _json.loads(result_json)
                outputs = res.get("outputs") or []
                if outputs and isinstance(outputs[0], dict):
                    o = outputs[0]
                    out = {
                        "asset_id": o.get("asset_id"),
                        # The clip's abs path — so the console can build a full video
                        # MediaRef for the Session Library + "Send to Studio" chain
                        # (B2). It is the same uri the movie/scene job results already
                        # expose on /video/jobs/<id>; playback still goes by job_id
                        # through /video/studio/clip/<id>, which owns the jail.
                        "uri": o.get("uri"),
                        "mime": o.get("mime"),
                        "width": o.get("width"),
                        "height": o.get("height"),
                        "duration_s": o.get("duration_s"),
                    }
            except (ValueError, TypeError):
                out = None
        clips.append({
            "job_id": job_id,
            "status": status,
            "playable": bool(status == "done" and out),
            "created": created,
            "updated": updated,
            "output": out,
        })

    return jsonify({"clips": clips}), 200


@video_bp.route("/video/studio/clip/<job_id>", methods=["GET"])
def video_studio_clip(job_id):
    # Canonical clips root — imported LAZILY (mirrors the runner's lazy studio-spine
    # imports) so this module stays app-boot cheap and drift-free with studio.job.
    from abstract_hugpy_dev.video_intel.studio.job import DEFAULT_CLIPS_ROOT

    # Archived clips are HIDDEN from GET /video/studio/clips but their bytes are
    # NEVER deleted (never-delete doctrine) — a direct fetch by id gets an HONEST
    # 410 naming "archived", not a bare 404 that reads as "this never existed".
    if media_bus.is_archived(job_id):
        return jsonify({"error": "clip archived", "archived": True}), 410

    view = media_bus.get(job_id)              # {"status","result",...} — unknown -> nulls
    result = view.get("result") if isinstance(view, dict) else None
    if not (isinstance(result, dict) and result.get("ok")):
        # unknown / queued / running / failed / cancelled — no playable clip yet
        return jsonify({"error": "no completed studio clip for that job"}), 404

    outputs = result.get("outputs") or []
    first = outputs[0] if (outputs and isinstance(outputs[0], dict)) else {}
    uri = first.get("uri")
    if not uri or not isinstance(uri, str):
        return jsonify({"error": "job result carries no clip uri"}), 404

    # Jail: only serve a file that really lives under the studio clips tree. This is
    # the single seam that keeps a crafted/rehomed uri from becoming an arbitrary
    # file read — it is checked on the REALPATH, so symlinks/.. can't escape.
    resolved = os.path.realpath(uri)
    if not _is_within(resolved, DEFAULT_CLIPS_ROOT) or not os.path.isfile(resolved):
        return jsonify({"error": "clip not found"}), 404

    # EXPLICIT Content-Type — NEVER filename guessing. A content-addressed studio clip
    # can be served from a uri WITHOUT a .mp4 extension (media-store ingest names some
    # assets by id), and send_file's extension guess then yields no/an unknown type, so
    # the console <video> shows a gray "unknown mime" error. Source the mime from the
    # catalog record (outputs[].mime, already "video/mp4"); fall back to "video/mp4" for
    # anything under the studio clips dir (every file the jail above admits is a clip).
    mime = first.get("mime")
    if not (isinstance(mime, str) and mime.strip()):
        mime = "video/mp4"
    return send_file(
        resolved,
        mimetype=mime,
        conditional=True,   # HTTP Range + conditional requests => <video> can seek
    )


# --------------------------------------------------------------------------- #
# 5b) GET /video/studio/clip/<job_id>/detail — the exact CREATION PARAMETERS of a
#     studio render, for a list-row "why did this pass/fail" expander. LAZY (fetched
#     on expand, not on the 6s list poll) so the list stays lean, and JAILED like the
#     stream route (the manifest.json is read only from beside a clip that really lives
#     under the studio clips tree). Two data sources, used-not-invented:
#       * DONE  -> the content-addressed manifest.json beside the clip (the TRUE render
#                  params: model_id, precision, resolution, seed, sampler steps/cfg/shift,
#                  prompt/negative, source_video, content_hash) + the output geometry.
#       * FAILED/CANCELLED (or any non-done) -> the job's requested spec + the error
#                  {code, message, retryable} from the job result. No clip, so no manifest.
#     The requested SPEC (from the bus row's spec_json) rides along in every case so the
#     UI can show "asked for X, got Y".
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clip/<job_id>/detail", methods=["GET"])
def video_studio_clip_detail(job_id):
    import json as _json
    import sqlite3
    from abstract_hugpy_dev.video_intel.studio.job import DEFAULT_CLIPS_ROOT

    # Same honest 410 as the stream route: an archived clip's row/manifest still
    # exist (never-delete), but the expander should say "archived", not 404.
    if media_bus.is_archived(job_id):
        return jsonify({"error": "clip archived", "archived": True}), 410

    # Read the bus row read-only (mirrors /video/studio/clips) — spec_json carries the
    # REQUESTED params, result_json the outcome. Unknown id -> 404 (nothing to detail).
    row = None
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT status, spec_json, result_json, created, updated FROM media_jobs "
                "WHERE job_id=? AND name='studio_i2v'",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        row = None
    if row is None:
        return jsonify({"error": "no studio job for that id"}), 404
    status, spec_json, result_json, created, updated = row

    # Curated view of the REQUESTED spec (drop out_root — an internal path). Everything
    # else is a creation parameter worth showing.
    spec_view = None
    if spec_json:
        try:
            s = _json.loads(spec_json)
            spec_view = {k: s.get(k) for k in (
                "capability", "width", "height", "fps", "vram_budget_gb", "seed",
                "prompt", "negative", "steps", "cfg", "model_id",
                "start_image", "source_video")}
        except (ValueError, TypeError):
            spec_view = None

    result = None
    if result_json:
        try:
            result = _json.loads(result_json)
        except (ValueError, TypeError):
            result = None

    error_view = None
    manifest_view = None
    if isinstance(result, dict):
        if result.get("ok"):
            outputs = result.get("outputs") or []
            first = outputs[0] if (outputs and isinstance(outputs[0], dict)) else {}
            uri = first.get("uri")
            if isinstance(uri, str) and uri:
                resolved = os.path.realpath(uri)
                # Same jail as the stream route: only read a manifest beside a clip that
                # really lives under the studio clips tree (checked on the realpath).
                if _is_within(resolved, DEFAULT_CLIPS_ROOT) and os.path.isfile(resolved):
                    mpath = os.path.join(os.path.dirname(resolved), "manifest.json")
                    if _is_within(os.path.realpath(mpath), DEFAULT_CLIPS_ROOT) \
                            and os.path.isfile(mpath):
                        try:
                            with open(mpath, "r", encoding="utf-8") as fh:
                                m = _json.load(fh)
                            manifest_view = _studio_manifest_view(m, resolved, first)
                        except (OSError, ValueError, TypeError):
                            manifest_view = None
        else:
            err = result.get("error") or {}
            if isinstance(err, dict):
                error_view = {
                    "code": err.get("code"),
                    "message": err.get("message"),
                    "retryable": bool(err.get("retryable", False)),
                }

    # SOURCE discriminator (coordinator addendum): which record the render params come
    # from. "manifest" = the content-addressed manifest.json beside a produced clip — the
    # TRUE, RESOLVED params (model bound, sampler steps/cfg actually used). "job_record" =
    # only the media-bus row exists: a FAILED / CANCELLED / still-running job wrote NO
    # manifest (only successful renders write the clip dir), so `spec` carries the
    # REQUESTED params (unresolved — e.g. `steps` null means "model default", which was
    # never bound) and `error` carries the failure {code,message,retryable}. The UI reads
    # this to label job_record params as REQUESTED and to explain the row that most needs
    # it. We surface only what the record HOLDS — no resolved sampler value is invented
    # for a job that failed before it bound a model.
    source = "manifest" if manifest_view is not None else "job_record"
    return jsonify({
        "job_id": job_id,
        "status": status,
        "source": source,
        "playable": bool(status == "done" and manifest_view is not None),
        "spec": spec_view,
        "manifest": manifest_view,
        "error": error_view,
        # Bus row timestamps (epoch seconds) — when the job was enqueued / last updated.
        # Present in every case (the row always carries them); the UI shows them on a
        # failed/cancelled row alongside the requested params + error.
        "created": created,
        "updated": updated,
    }), 200


# --------------------------------------------------------------------------- #
# 5c) POST /video/studio/clip/<job_id>/archive + /unarchive — the fix for
#     "removed clips just reappear". GET /video/studio/clips (2j... above) is
#     DB-driven, not a filesystem walk, so a real "remove" has to be a mark THAT
#     query excludes — which is exactly what media_bus.archive does (see its
#     header note). The clip's row and bytes on disk are NEVER touched; archiving
#     only flips archived_at, so the never-delete doctrine holds trivially (there
#     is no delete path to guard against). Idempotent by choice, not 409: a
#     retried/double-clicked archive (the UI archives optimistically, see
#     StudioPlane's Session-Library remove) must read as "already gone", never as
#     a failure that would restore the row and show an error for nothing.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clip/<job_id>/archive", methods=["POST"])
def video_studio_clip_archive(job_id):
    result = media_bus.archive(job_id)
    if not result["found"]:
        return jsonify({"ok": False, "error": "no studio job for that id"}), 404
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "archived": True,
        "already": result["already"],
        "archived_at": result["archived_at"],
    }), 200


@video_bp.route("/video/studio/clip/<job_id>/unarchive", methods=["POST"])
def video_studio_clip_unarchive(job_id):
    # The honest counterpart — cheap to add, and it makes archive a REVERSIBLE
    # hide rather than a one-way trapdoor.
    result = media_bus.unarchive(job_id)
    if not result["found"]:
        return jsonify({"ok": False, "error": "no studio job for that id"}), 404
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "archived": False,
        "already": result["already"],
    }), 200


def _studio_manifest_view(m: dict, clip_path: str, output: dict) -> dict:
    """Compact projection of a render manifest.json for the clip-detail expander. The
    clip DIR name IS the content_hash (content-addressed storage), so we surface it from
    the path rather than recomputing. ``output`` (the cataloged MediaRef) supplies the
    real duration; frames is DERIVED (duration * fps) for a CFR clip — not invented."""
    ladder = m.get("resolution_ladder") or []
    res = ladder[0] if (ladder and isinstance(ladder[0], (list, tuple))
                        and len(ladder[0]) == 3) else None
    resolution = ({"width": res[0], "height": res[1], "fps": res[2]}
                  if res is not None else None)
    seeds = m.get("seeds") or {}
    duration_s = output.get("duration_s") if isinstance(output, dict) else None
    frames = None
    if resolution is not None and isinstance(duration_s, (int, float)):
        frames = int(round(duration_s * resolution["fps"]))
    return {
        "content_hash": os.path.basename(os.path.dirname(clip_path)),
        "model_id": m.get("model_id"),
        "framework": m.get("framework"),
        "task": m.get("task"),
        "capability": m.get("capability"),
        "precision": m.get("precision"),
        "determinism_class": m.get("determinism_class"),
        "resolution": resolution,
        "duration_s": duration_s,
        "frames": frames,
        "seed": seeds.get("global_seed"),
        "sampler": m.get("sampler"),   # {sampler, scheduler, steps, cfg, shift, sigmas}
        "prompt": m.get("prompt"),
        "negative_prompt": m.get("negative_prompt"),
        "source_video": m.get("source_video"),
    }


# --------------------------------------------------------------------------- #
# 5c) GET /video/projects — the distinct known auto-archive PROJECT names.
#     Read-only projection over the media-bus job store: scans EVERY job's stored
#     spec_json (studio_i2v, generate_movie, generate_scene, generate_image) for
#     distinct non-empty "project" values — the optional human archive NAME threaded
#     through each enqueue route — and returns them sorted case-insensitively. Uses
#     the same jailed mode=ro connection idiom as /video/studio/clips (no writes, no
#     schema touch). A frontend project-picker reads this to offer known names; a
#     job that carried no project contributes nothing (no empty entry).
# --------------------------------------------------------------------------- #
@video_bp.route("/video/projects", methods=["GET"])
def video_projects():
    import json as _json
    import sqlite3

    # Distinct EXACT names (a set); the case-insensitive ordering is the SORT, not the
    # de-dup, so "Alpha" and "alpha" would both list (they are distinct strings) — the
    # frontend contract only pins the shape {"projects": [...]} and the sort.
    names: set = set()
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT spec_json FROM media_jobs WHERE spec_json IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        # No DB yet / transient lock -> an empty list is the honest answer.
        rows = []

    for (spec_json,) in rows:
        if not spec_json:
            continue
        try:
            spec = _json.loads(spec_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(spec, dict):
            continue
        p = spec.get("project")
        if isinstance(p, str):
            p = p.strip()
            if p:
                names.add(p)

    return jsonify({"projects": sorted(names, key=str.lower)}), 200


# --------------------------------------------------------------------------- #
# IDENTITY PROFILES (studio stage (a)) — a NAMED, DURABLE library item:
#   {name, reference_images (1..4), created_at, notes?}
# DOCTRINE (STUDIO-ROADMAP.md "IDENTITY PROFILES"): a profile is the durable form
# of "the reference set IS the identity" — a character's curated reference DNA
# saved ONCE and associated anywhere (single clips, movies, stills) instead of
# being re-supplied per request. This is the LIBRARY-ITEM surface; stage (b)
# (turnaround generation from a profile + the re-edit loop that promotes an
# approved rendering to the canonical reference) layers on top of this store next.
#
# These routes only translate HTTP <-> video_intel.identity_profiles (the store
# owns the single-writer/atomic-write invariant, reused from api_keys). Reference
# images are validated HERE with the studio movie/i2v route's EXACT jail + ingest
# + image-classify pass, then stored as given (durable uploads/store paths).
# Errors-as-data throughout — a bad path is a clean 4xx, never a deferred failure.
# --------------------------------------------------------------------------- #
def _validate_profile_reference_images(raws):
    """Jail-resolve + image-classify a list of reference-image paths EXACTLY as the
    studio routes do (``_jail_resolve`` -> ``media_store.ingest(kind_hint="image")``
    -> ``kind == "image"``). Returns ``(resolved_abs_paths, None)`` on success or
    ``(None, (error_payload, status))`` so the caller returns the clean 4xx. 1..4
    images required — the same envelope the movie route enforces."""
    if not isinstance(raws, list) or not raws:
        return None, ({"error": "reference_images must be a non-empty list of paths"}, 400)
    if len(raws) > identity_profiles.MAX_REFERENCE_IMAGES:
        return None, ({
            "error": f"at most {identity_profiles.MAX_REFERENCE_IMAGES} reference_images are accepted"
        }, 400)
    resolved: list = []
    for raw in raws:
        if not isinstance(raw, str) or not raw.strip():
            return None, ({"error": "each reference_image must be a non-empty path"}, 400)
        rp = _jail_resolve(raw)
        if rp is None:
            return None, ({"error": "reference_image outside storage jail"}, 400)
        if not os.path.isfile(rp):
            return None, ({"error": "reference_image not found"}, 404)
        try:
            iref = media_store.ingest(rp, kind_hint="image")
        except Exception as exc:  # unreadable / not a real image = bad input
            return None, ({"error": f"reference_image is not a readable media file: {exc}"}, 400)
        if iref.kind != "image":
            return None, ({
                "error": f"reference_image is not an image (classified as {iref.kind})"
            }, 400)
        resolved.append(iref.uri)
    return resolved, None


def _reference_images_from_body(body):
    """Resolve an optional ``identity_profile`` slug to its canonical reference set,
    for the studio i2v / movie enqueue bodies.

    UNIFIED IDENTITY (operator 2026-07-12): an identity profile IS the identity. A
    studio enqueue may carry ``identity_profile: "<slug>"`` instead of a raw
    ``reference_images`` list. The saved profile's reference set is CANONICAL: when
    BOTH are present the profile WINS — a named, curated identity outranks ad-hoc
    paths (and a later edit to the profile re-resolves through the same slug). Raw
    ``reference_images`` stays accepted for the unsaved / backward-compat case.

    PROMOTED CANONICAL preference (stage (b)): once the operator promotes reconstruction
    views to the profile's ``canonical`` ref set (POST .../canonical), that APPROVED DNA
    outranks the raw uploaded ``reference_images`` — the profile resolves to ``canonical``
    when it is non-empty, falling back to ``reference_images`` otherwise. Same single
    code path either way.

    Returns ``(reference_images_or_None, None)`` on success, or
    ``(None, (error_payload, status))`` when a given slug names no profile — a stale
    slug is a clean 4xx, never a silent empty identity. The returned list still flows
    through the SAME jail + ingest + image-classify validation the routes already run
    (the profile's stored paths are re-checked, one code path)."""
    slug = body.get("identity_profile")
    if slug is None:
        return body.get("reference_images"), None
    if not isinstance(slug, str) or not slug.strip():
        return None, ({"error": "identity_profile must be a slug string"}, 400)
    profile = identity_profiles.get_profile(slug.strip())
    if profile is None:
        return None, ({"error": f"identity_profile {slug!r} not found"}, 404)
    canonical = profile.get("canonical") or []
    chosen = canonical if canonical else list(profile.get("reference_images") or [])
    return list(chosen), None


@video_bp.route("/video/identity-profiles", methods=["GET"])
def video_identity_profiles_list():
    # Active (non-archived) profiles, newest first. The store's projection already
    # folds the slug into each row, so a list row is self-describing.
    return jsonify({"profiles": identity_profiles.list_profiles()}), 200


@video_bp.route("/video/identity-profiles", methods=["POST"])
def video_identity_profiles_create():
    # {name, reference_images[1..4], notes?} -> create. Slug derives from name; a
    # collision with an existing ACTIVE profile is a 409 (never a silent overwrite).
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return jsonify({"error": "name is required"}), 400
    resolved, err = _validate_profile_reference_images(body.get("reference_images"))
    if err is not None:
        payload, status = err
        return jsonify(payload), status
    notes = body.get("notes")
    if notes is not None and not isinstance(notes, str):
        return jsonify({"error": "notes must be a string"}), 400
    try:
        profile = identity_profiles.create_profile(name, resolved, notes=notes or "")
    except identity_profiles.ProfileError as exc:  # dup slug / bad shape = errors-as-data
        status = 409 if exc.code == "duplicate" else 400
        return jsonify({"error": str(exc), "code": exc.code}), status
    return jsonify({"profile": profile}), 201


@video_bp.route("/video/identity-profiles/<slug>", methods=["GET"])
def video_identity_profile_get(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200


@video_bp.route("/video/identity-profiles/<slug>", methods=["DELETE"])
def video_identity_profile_delete(slug):
    # ARCHIVE semantics (never-delete doctrine): the store moves the entry under a
    # `_deleted` key with a timestamp rather than erasing it. Idempotent — deleting
    # an unknown/already-archived slug is a clean 404 no-op.
    archived = identity_profiles.delete_profile(slug)
    if archived is None:
        return jsonify({"error": "identity profile not found", "archived": False}), 404
    return jsonify({"ok": True, "archived": True, "slug": slug}), 200


# --------------------------------------------------------------------------- #
# PATCH /video/identity-profiles/<slug> — edit an existing profile's DISPLAY
# fields + reference set. {name?, notes?, reference_images?}, all optional (a
# true partial update — an omitted key is left untouched, matching
# identity_profiles.update_profile's **kwargs contract below). Any given
# `reference_images` runs the SAME jail + ingest + image-classify validation as
# POST create (`_validate_profile_reference_images`), so a PATCH can never leave
# a profile pointing at an unreadable/non-image/jail-escaping path.
#
# RENAMING KEEPS THE SLUG STABLE — `name` only ever changes the stored display
# string, never the `<slug>` this route (and every identity_profile:<slug>
# reference in a saved template/spec/enqueue body) keys on. Re-slugging on
# rename would silently break every one of those references; see
# identity_profiles.update_profile's docstring for the full rationale.
#
# An identity keeps >=1 reference image always: a given `reference_images` must
# be non-empty (the store rejects an empty list exactly like create does) — omit
# the key entirely to leave the current set untouched. Existence is checked
# BEFORE any (potentially expensive) reference validation runs, so an unknown
# slug 404s fast without jail-resolving/ingesting images that will never be
# stored; the store's own None-return is still honored afterward as a defensive
# recheck against a concurrent archive racing this same request.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>", methods=["PATCH"])
def video_identity_profile_update(slug):
    if identity_profiles.get_profile(slug) is None:
        return jsonify({"error": "identity profile not found"}), 404

    body = request.get_json(silent=True) or {}
    kwargs: dict = {}

    if "name" in body:
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            return jsonify({"error": "name is required"}), 400
        kwargs["name"] = name

    if "notes" in body:
        notes = body.get("notes")
        if notes is not None and not isinstance(notes, str):
            return jsonify({"error": "notes must be a string"}), 400
        kwargs["notes"] = notes or ""

    if "reference_images" in body:
        resolved, err = _validate_profile_reference_images(body.get("reference_images"))
        if err is not None:
            payload, status = err
            return jsonify(payload), status
        kwargs["reference_images"] = resolved

    try:
        profile = identity_profiles.update_profile(slug, **kwargs)
    except identity_profiles.ProfileError as exc:  # errors-as-data, never a 500
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# POST /video/identity-profiles/<slug>/reconstruction — STAGE (b): generate an
# identity-locked TURNAROUND of the character (one still per view) from its reference
# images + description, stored in the identity's dir for approval.
#
# Body (all optional): {prompt?: str, views?: [str], seed?: int, mode?: "sheet"|"turntable"}
#   prompt  extra description woven into every view prompt; default = the profile's notes.
#   views   the view names to render; default ["front","three_quarter","profile","back"].
#           A SINGLE-view request (["front"]) is valid — the cheap one-render check.
#           IGNORED in turntable mode (the orbit clip's frames define the set).
#   seed    base render seed (default 0; all views share it — the view differs by prompt).
#   mode    "sheet" (default — N independent view-stills) or "turntable" (ONE 360° orbit
#           clip, every frame kept as an angular degree-view the UI scrubs to rotate).
#
# Enqueues ONE orchestrator job (identity_reconstruction) and returns {job_id, recon_id};
# its runner renders each view behind the swap seam, then attaches the produced stills to
# the profile. Poll GET /video/jobs/<job_id>; on done, re-read the profile (GET .../<slug>)
# for the new ``reconstructions`` entry keyed by recon_id.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction", methods=["POST"])
def video_identity_profile_reconstruction(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    body = request.get_json(silent=True) or {}

    # Resolve the profile's reference set through the SAME resolver the studio enqueue
    # uses (canonical-preferred) + the SAME jail + ingest + image-classify validation
    # (single code path). A promoted canonical set outranks the raw uploads here too.
    refs_in, perr = _reference_images_from_body({"identity_profile": slug})
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    resolved_refs, verr = _validate_profile_reference_images(refs_in)
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    # prompt: default to the profile's notes (the description the profile carries).
    prompt = body.get("prompt")
    if prompt is None:
        prompt = profile.get("notes") or ""
    elif not isinstance(prompt, str):
        return jsonify({"error": "prompt must be a string"}), 400

    # views: default to the canonical turnaround set; a non-empty list of non-empty
    # strings (a single view is valid — verify one render cheaply first).
    views = body.get("views")
    if views is None:
        views = list(_DEFAULT_RECON_VIEWS)
    if not isinstance(views, list) or not views:
        return jsonify({"error": "views must be a non-empty list of view names"}), 400
    for v in views:
        if not isinstance(v, str) or not v.strip():
            return jsonify({"error": "each view must be a non-empty string"}), 400

    seed = body.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        return jsonify({"error": "seed must be an int"}), 400

    # mode: default "sheet" (the existing N-independent-view-stills path). "turntable"
    # renders ONE 360° orbit clip and keeps every frame as a scrubbable degree-view
    # (``views`` from the body are ignored in turntable mode — the orbit defines the set).
    mode = body.get("mode", "sheet")
    if mode not in ("sheet", "turntable"):
        return jsonify({"error": 'mode must be "sheet" or "turntable"'}), 400

    recon_id = "recon_" + secrets.token_hex(8)
    try:
        spec = make_identity_reconstruction(
            slug=slug,
            recon_id=recon_id,
            reference_images=tuple(resolved_refs),
            views=tuple(views),
            base_prompt=prompt,
            seed=seed,
            mode=mode,
            # geometry (<=480p id_lock ceiling) + autofit VRAM are the schema defaults.
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400
    job_id = media_bus.enqueue("identity_reconstruction", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# POST /video/identity-profiles/<slug>/canonical — STAGE (b) approve: promote chosen
# reconstruction views into the profile's ``canonical`` reference set (the approved
# character DNA the resolver then PREFERS over the raw uploads). Body:
# {recon_id: str, views: [int]} — the view indices of that reconstruction to promote
# (at most 4; canonical feeds the id_lock reference channel). Returns {profile} with
# ``canonical`` populated. An unknown recon_id / out-of-range index is a clean 400.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/canonical", methods=["POST"])
def video_identity_profile_canonical(slug):
    if identity_profiles.get_profile(slug) is None:
        return jsonify({"error": "identity profile not found"}), 404
    body = request.get_json(silent=True) or {}
    recon_id = body.get("recon_id")
    if not isinstance(recon_id, str) or not recon_id.strip():
        return jsonify({"error": "recon_id is required"}), 400
    views = body.get("views")
    if not isinstance(views, list) or not views:
        return jsonify({"error": "views must be a non-empty list of view indices"}), 400
    for v in views:
        if not isinstance(v, int) or isinstance(v, bool):
            return jsonify({"error": "each view must be an integer index"}), 400
    try:
        profile = identity_profiles.promote_reconstruction_views(slug, recon_id, views)
    except identity_profiles.ProfileError as exc:  # bad index / unknown recon = 400
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200
