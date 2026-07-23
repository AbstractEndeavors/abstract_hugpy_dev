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
from abstract_hugpy_dev.video_intel import media_store, media_bus, identity_profiles, shot_intent
from abstract_hugpy_dev.video_intel.placement import job_placement
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
    make_identity_mesh,
    MESH_VIEW_NAMES as _MESH_VIEW_NAMES,
)
from abstract_hugpy_dev.video_intel.identity_video_extract_schema import (
    make_identity_video_extract,
)
from abstract_hugpy_dev.video_intel.chains import (
    resolve_video_parts,
    resolve_video_parts_scene,
    resolve_video_parts_movie,
)

video_bp, logger = get_bp("video_bp", __name__)


# --------------------------------------------------------------------------- #
# k9 attribution — WHO is enqueueing this video job. Resolved HERE, in the Flask
# request context (media_bus is Flask-free), and threaded onto the job so it
# surfaces in /llm/jobs. Mirrors chat's streaming._resolve_request_principal:
# operator session first (the stronger, first-party identity), then a video-share
# principal (share:<key_id>) for an outside party on a share link, else None
# (unattributed — every self-hosted/open-mode call). Best-effort: attribution
# must never fail an enqueue.
# --------------------------------------------------------------------------- #
def _request_principal():
    try:
        from ..operator_auth import operator_authenticated
        if operator_authenticated():
            return "operator"
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..video_auth import _video_share_principal
        p = _video_share_principal(request)
        if p:
            return p
    except Exception:  # noqa: BLE001
        pass
    return None


def _video_enqueue(name, spec):
    """media_bus.enqueue with the request principal stamped for attribution (k9).
    Every /video enqueue route funnels through this so a job's origin (operator
    vs a share link) rides onto /llm/jobs uniformly."""
    return media_bus.enqueue(name, spec, principal=_request_principal())


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


def _ingest_image_references(raws):
    """Jail-resolve + ``media_store.ingest(kind_hint="image")``-classify a list of raw
    reference-image paths into media-store URIs — the ONE code path a studio spec's id_lock
    references (movie-level OR the S2-movie per-goal view refs) must pass through, because
    the renderer consumes media_store URIs, not raw paths. Errors-as-data: returns
    ``(uris, None)`` on success, or ``(None, (payload, status))`` on the FIRST bad path (a
    non-string, a jail escape -> 400, a missing file -> 404, an unreadable/non-image file ->
    400) so the caller returns the 4xx verbatim. Factored out of the movie-level reference
    loop so the per-goal bank frames get IDENTICAL jail + classify treatment (no duplication,
    no drift)."""
    uris = []
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
            return None, ({"error": f"reference_image is not an image (classified as {iref.kind})"}, 400)
        uris.append(iref.uri)
    return uris, None


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
    job_id = _video_enqueue("crop", spec)
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
    job_id = _video_enqueue("frame_extract", spec)
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
    job_id = _video_enqueue("audio_extract", spec)
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

    job_id = _video_enqueue("generate_image", resolved)
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

    job_id = _video_enqueue("generate_scene", resolved)
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

    job_id = _video_enqueue("generate_movie", resolved)
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
    # NOTE (2026-07-16): canonical may now hold 8 views, but _reference_images_from_body
    # already narrows a profile-resolved set to the RENDER cap (4) before returning, so the
    # >4 check below can never fire on an identity's own DNA. It still guards a caller's RAW
    # reference_images list, where >4 stays a clean caller error.
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
    job_id = _video_enqueue("studio_i2v", spec)
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
    # NOTE (2026-07-16): an 8-view canonical is already narrowed to the RENDER cap (4) by
    # _reference_images_from_body, so the >4 check below only ever guards a raw caller list.
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
        # jail-resolve + image-classify -> media_store URIs (the ONE ingest path; see
        # _ingest_image_references). A bad path is a clean 4xx here, not a runner failure.
        resolved_refs, _ref_err = _ingest_image_references(reference_images_in)
        if _ref_err is not None:
            _pl, _st = _ref_err
            return jsonify(_pl), _st

    # PER-GOAL VIEW (IDENTITY-3D-CONTINUITY-PLAN.md S2-movie + S3): let EACH segment of an
    # identity movie condition on a DIFFERENT turntable VIEW of the SAME identity, so a
    # ``cut`` into a new scene ("beach" -> "volleyball") holds the character while turning
    # the camera per shot. The MOVIE-LEVEL DNA (``resolved_refs``) stays canonical (resolved
    # above, unchanged) — this only OVERRIDES a goal's own reference set when a view resolves
    # to ring frames; a goal with no view is left untouched (its schema ``reference_images``
    # stays None -> the runner inherits the movie-level set, byte-identical to today).
    #
    # Per goal the view is chosen by precedence:
    #   1. explicit ``view`` on the goal (semantic name or {azimuth_deg})  -> "explicit" ;
    #   2. else DERIVED from the goal's prompt text (S3 keyword pass)       -> "derived"  ;
    #   3. else no view                                                     -> "none" (inherit).
    # DEGRADE-TO-INHERIT (defaults-are-promises): only an INVALID *explicit* view is a clean
    # 400 (naming the segment). A valid view on an identity with NO turntable ring, or on a
    # NON-identity movie (no slug / no movie-level refs), simply inherits — never an error.
    # The per-goal bank paths go through the SAME _ingest_image_references media_store path
    # the movie-level refs use (the renderer consumes URIs, not raw paths).
    slug = body.get("identity_profile")
    is_identity_movie = bool(resolved_refs) and isinstance(slug, str) and bool(slug.strip())
    if is_identity_movie:
        profile = identity_profiles.get_profile(slug.strip())  # re-fetch; validated above
        # bank_views resolves the version itself (id-or-name, else active), so mirror the
        # movie-level precedence by handing it the body's identity_version straight through
        # (already validated by _reference_images_from_body). Compute the ring ONCE.
        bank = identity_profiles.bank_views(
            profile, version_id=body.get("identity_version")) if profile else []
        for gi, (g_in, goal) in enumerate(zip(goals_in, goals)):
            view_hint = g_in.get("view")
            view_source = "none"
            azimuth_deg = None
            if view_hint is not None:
                azimuth_deg, view_err = identity_profiles.azimuth_for_view(view_hint)
                if view_err is not None:
                    return jsonify(
                        {"error": f"goal {goal.segment_id!r}: {view_err}"}), 400
                view_source = "explicit"
            else:
                derived = shot_intent.derive_view_from_prompt(goal.prompt)
                if derived is not None:
                    azimuth_deg, _ = identity_profiles.azimuth_for_view(derived)
                    view_source = "derived"
            if azimuth_deg is None:
                logger.info("movie per-goal view: segment=%s view_source=none", goal.segment_id)
                continue
            if not bank:
                # identity has no turntable ring -> inherit the movie-level DNA (never an error)
                logger.info("movie per-goal view: segment=%s view_source=%s no-ring -> inherit",
                            goal.segment_id, view_source)
                continue
            # K is the RENDER cap (4), not the canonical/storage cap: these frames go
            # straight into ONE segment's id_lock conditioning. Was MAX_CANONICAL_IMAGES
            # back when both numbers were 4; pinned to MAX_RENDER_REFS on 2026-07-16 when
            # canonical widened to 8, so a per-goal view still conditions on 4 frames.
            picked = identity_profiles.nearest_bank_views(
                bank, azimuth_deg, identity_profiles.MAX_RENDER_REFS)
            goal_uris, _g_err = _ingest_image_references([b["path"] for b in picked])
            if _g_err is not None:
                _pl, _st = _g_err
                return jsonify(_pl), _st
            goals[gi] = dataclasses.replace(goal, reference_images=tuple(goal_uris))
            logger.info("movie per-goal view: segment=%s view_source=%s azimuth=%.1f n_refs=%d",
                        goal.segment_id, view_source, azimuth_deg, len(goal_uris))

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
    job_id = _video_enqueue("generate_studio_movie", spec)
    return jsonify({"job_id": job_id}), 200


# --------------------------------------------------------------------------- #
# 2d'''') POST /video/mlt/render — headless Kdenlive/MLT render (k22).
#     The operator authors a project in Kdenlive on Windows against the Samba studio
#     share, saves the .kdenlive into the WRITABLE edits/ subtree, and this route
#     enqueues an ``mlt_render`` media_bus job that path-maps the project + renders it
#     server-side with melt, writing the output back under edits/renders/. Query it
#     exactly like any other media job: GET /video/jobs/<job_id>.
#
#     CONSOLE-OPERATOR ONLY: the blanket /video gate already admitted an operator OR a
#     share-link guest, but this WRITES INTO the operator's real editing tree (and reads
#     an arbitrary project file), so — like /to-editor — the body re-checks
#     operator_authenticated() and 403s a share guest. Defense in depth.
#
#     JAIL: the project MUST live under the studio tree (the runner re-checks); the output
#     is always resolved under edits/renders/ by the runner. A path outside the jail 400s
#     here for a fast, honest rejection.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/mlt/render", methods=["POST"])
def video_mlt_render():
    from ..operator_auth import operator_authenticated
    if not operator_authenticated():
        return jsonify({"error": "operator session required"}), 403

    from abstract_hugpy_dev.video_intel.mlt_render_schema import make_mlt_render
    from abstract_hugpy_dev.video_intel.runners.mlt_render import STUDIO_ROOT

    body = request.get_json(silent=True) or {}
    project_path = body.get("project_path")
    if not isinstance(project_path, str) or not project_path.strip():
        return jsonify({"error": "project_path is required (absolute path to a "
                        ".kdenlive/.mlt project under the studio share)"}), 400

    # Jail the project under the studio tree (fast 400; the runner re-checks authoritatively).
    rp = os.path.realpath(project_path.strip())
    if not _is_within(rp, STUDIO_ROOT):
        return jsonify({"error": "project_path is outside the studio jail"}), 400
    if not os.path.isfile(rp):
        return jsonify({"error": "project_path not found"}), 404

    try:
        spec = make_mlt_render(
            project_path=rp,
            output_rel=body.get("output_rel"),
            width=body.get("width"),
            height=body.get("height"),
            fps=body.get("fps"),
            profile=body.get("profile"),
            vcodec=body.get("vcodec", "libx264"),
            acodec=body.get("acodec", "aac"),
            container=body.get("container", "mp4"),
            vb=body.get("vb"),
            drive_letter=body.get("drive_letter"),
        )
    except (ValueError, TypeError) as exc:  # structural spec error = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("mlt_render", spec)
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
# 2e-bis) POST /video/prompt/assist — LLM-backed helper for the studio's prompt
#     input. Two modes:
#       detail    expand/enrich the caller's DRAFT image prompt into a richer,
#                 more vivid diffusion prompt (preserves the draft's subject
#                 and intent — "draft" is required).
#       generate  write a full, original image-generation prompt from
#                 scratch; "draft", if given, is used only as a loose theme.
#
#     Routes through the exact SAME internal chat plane as /chat/stream and
#     /prompt — managers.dispatch.execute_prompt via resolve() (see
#     ../functions/chat/streaming.py:94 execute_chat_stream and
#     ../routes/discord_routes.py:326 _generate_candidate for the two other
#     callers of this same one-shot pattern). On THIS central
#     (HUGPY_NO_LOCAL_SERVING=true, see managers/serve/policy.py) resolve()'s
#     DelegatingRunner sends the completion to a live GPU worker; it never
#     loads a model in-process here. No live worker / unreachable -> a clean
#     502 via the same _friendly_stream_error mapping /chat/stream uses on a
#     stream failure — never a 500, never a silent local load.
# --------------------------------------------------------------------------- #
# Default assist model. The operator asked for flux2-klein (2026-07-13), BUT on
# THIS dev central flux2 does not resolve for inference: execute_prompt (and even
# /v1/chat/completions) only know the 11 serve-configured models — the other ~100
# manifest/discovered models (flux2-klein, the HunyuanVideo rewriter, etc.) are
# catalog-only and 404 with "Unknown model_key". Defaulting to flux2 would make
# every default call fail, so per the defaults-are-promises doctrine the default
# is the best chat model that ACTUALLY resolves here: Qwen2.5-3B-Instruct-GGUF
# (verified 200 + good prompt enrichment). A caller can still pass any key via
# body["model"] — and once flux2 is serve-configured on the fleet this should
# switch back to it. keeper 2026-07-13 (flux2 non-resolution flagged to operator).
_DEFAULT_PROMPT_ASSIST_MODEL = "Qwen2.5-3B-Instruct-GGUF"

_PROMPT_ASSIST_SYSTEM = (
    "You are an expert image-prompt engineer. Return ONLY the final prompt "
    "text — no preamble, no quotes, no explanation. Make it vivid, specific, "
    "and suitable for a diffusion image model."
)

# Context-aware framing (operator 2026-07-13 "yes" to context-aware generation).
# The caller may pass context.kind so a MOVIE/CLIP (video) gets motion/camera
# phrasing while an IMAGE/SCENE (still) keeps diffusion phrasing. No context ->
# the original still-image behavior, so old callers are unaffected.
_PROMPT_ASSIST_KINDS = ("image", "scene", "movie", "clip")
_PROMPT_ASSIST_VIDEO_KINDS = frozenset({"movie", "clip"})


def _assist_framing(kind):
    """Return (system_prompt, medium_noun) for the requested kind. Video kinds
    ask for motion/camera; everything else (incl. None) keeps still-image
    phrasing identical to the pre-context behavior."""
    if kind in _PROMPT_ASSIST_VIDEO_KINDS:
        return (
            "You are an expert video-prompt engineer. Return ONLY the final "
            "prompt text — no preamble, no quotes, no explanation. Make it "
            "vivid and specific, describing subject, motion, camera movement, "
            "and mood, suitable for a text-to-video model.",
            "video-generation prompt",
        )
    return (_PROMPT_ASSIST_SYSTEM, "image-generation prompt")


def _await_sync(value):
    """Drive execute_prompt's (possibly) awaitable result from WSGI — the exact
    same idiom prompt_routes._await_sync / discord_routes._await_sync use (each
    module keeps its own copy rather than sharing a private helper cross-file).
    Uses the process-wide async runtime (one long-lived loop), not a fresh
    per-request loop — see _platform/async_runtime."""
    import inspect
    if not inspect.isawaitable(value):
        return value
    from abstract_hugpy_dev._platform import async_runtime
    return async_runtime.run(value)


def _prompt_assist_result_text(result) -> str:
    """Best-effort text extraction — mirrors discord_routes._result_text so a
    worker-relay dict, a pydantic ChatResult, or any other TaskResult-shaped
    object all yield the same plain string."""
    if isinstance(result, dict):
        return result.get("text") or ""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d.get("text") or ""
            except TypeError:
                continue
    return getattr(result, "text", "") or ""


@video_bp.route("/video/prompt/assist", methods=["POST"])
def video_prompt_assist():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if mode not in ("detail", "generate"):
        return jsonify({"error": 'mode must be "detail" or "generate"'}), 400

    draft = body.get("draft")
    if draft is not None and not isinstance(draft, str):
        return jsonify({"error": "draft must be a string"}), 400
    draft = draft.strip() if draft else ""
    if mode == "detail" and not draft:
        return jsonify({"error": 'draft is required for mode "detail"'}), 400

    model_key = body.get("model") or _DEFAULT_PROMPT_ASSIST_MODEL
    if not isinstance(model_key, str) or not model_key.strip():
        return jsonify({"error": "model must be a non-empty string"}), 400

    # Optional context so the assist is aware of WHAT is being generated
    # (still image vs. a video clip/movie). Absent/empty -> still-image
    # phrasing identical to the pre-context behavior (back-compat).
    context = body.get("context") or {}
    if not isinstance(context, dict):
        return jsonify({"error": "context must be an object"}), 400
    kind = context.get("kind")
    if kind is not None and kind not in _PROMPT_ASSIST_KINDS:
        return jsonify({
            "error": "context.kind must be one of " + "|".join(_PROMPT_ASSIST_KINDS)
        }), 400
    hint = context.get("hint")
    if hint is not None and not isinstance(hint, str):
        return jsonify({"error": "context.hint must be a string"}), 400
    hint = hint.strip() if hint else ""

    system_prompt, medium = _assist_framing(kind)

    if mode == "detail":
        user = (
            f"Expand this draft {medium} into a richer, more vivid, more "
            "specific prompt. Preserve the subject and intent of the draft "
            f"— add detail, don't replace it.\n\nDraft prompt: {draft}"
        )
    elif draft:
        user = (f'Write one compelling, original {medium} using '
                f'"{draft}" as a loose theme.')
    else:
        user = f"Write one compelling, original {medium} of your choosing."

    if hint:
        user += f"\n\nAdditional context to honor: {hint}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    # Late imports (mirrors prompt_routes/discord_routes) — dodges circulars
    # and keeps this module app-boot cheap when chat's plane isn't touched.
    from ..functions.imports import execute_prompt
    from ..functions.chat.streaming import _friendly_stream_error
    try:
        result = _await_sync(execute_prompt(
            model_key=model_key,
            messages=messages,
            task="text-generation",
            max_new_tokens=200,
        ))
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        # resolve()/builder validation errors (e.g. unknown model_key, or the
        # model doesn't support text-generation) — the caller's to fix, same
        # envelope /prompt uses for the identical exception set.
        return jsonify({"error": str(exc).strip("'\"")}), 400
    except Exception as exc:
        # No live worker for this model / worker unreachable / no local engine
        # — an actionable message via the same mapper /chat/stream uses,
        # never a raw traceback, never a 500.
        logger.exception("prompt/assist failed")
        return jsonify({"error": _friendly_stream_error(exc)}), 502

    ok = result.get("ok", True) if isinstance(result, dict) else getattr(result, "ok", True)
    text = _prompt_assist_result_text(result).strip()
    if not ok or not text:
        err = result.get("error") if isinstance(result, dict) else getattr(result, "error", None)
        return jsonify({"error": err or "assist produced no text"}), 502

    return jsonify({"prompt": text, "model": model_key, "kind": kind}), 200


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
    # assignee still counts as "already has it". Wildcard catches are excluded:
    # a "take all comers" box (worker_wildcard opt-in) is ELIGIBLE for the
    # model but does not HAVE it, so counting it "warm" would fake the
    # avoid-a-reload preference this set exists for.
    warm_ids = {w["id"] for w in
                worker_store.workers_for_model(model_key, online_only=False)
                if not w.get("_wildcard_catch")}

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
# 2z) GET /video/jobs — bus-wide LISTING for the console-wide "Active Processes"
#     view. In-flight media-bus jobs by default (queued/claimed/running/
#     cancelling); ?all=1 appends recent terminal rows (bounded). Each row carries
#     its parsed `progress` (incl. the awaiting_capacity HOLD marker) and a
#     `placement` object — {source:"reservation"|"template", host, worker_id, gpu,
#     process, reserved_bytes} (omit-when-unset) — so the console can show WHERE a
#     run physically executes (e.g. "ae · cuda:0 · P-studio"). Read-only; never
#     5xxes (a bus/placement hiccup degrades to fewer rows / no placement).
#
#     NOTE: this bare-path route MUST be registered on the blueprint so it wins
#     over the SPA catch-all (`@app.route("/<path:asset>")`): Werkzeug ranks a
#     static rule above a <path:> converter regardless of registration order, so
#     GET /video/jobs resolves here, not to index.html. (The per-id sibling
#     /video/jobs/<job_id> below is a distinct, more-specific rule.)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs", methods=["GET"])
def video_jobs_list():
    include_terminal = (request.args.get("all") or "").strip().lower() in (
        "1", "true", "yes", "on")
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    jobs = []
    try:
        rows = media_bus.list_jobs(include_terminal=include_terminal, limit=limit)
    except Exception:  # noqa: BLE001 — the listing never 5xxes
        logger.debug("video jobs listing failed", exc_info=True)
        rows = []
    for row in rows:
        try:
            pl = job_placement(row.get("job_id"), row.get("name"))
            if pl:
                row["placement"] = pl
        except Exception:  # noqa: BLE001 — placement is best-effort per row
            pass
        jobs.append(row)
    return jsonify({"jobs": jobs}), 200


# --------------------------------------------------------------------------- #
# 3) GET /video/jobs/<job_id> — read-only job view (unknown id -> null view)
# --------------------------------------------------------------------------- #
@video_bp.route("/video/jobs/<job_id>", methods=["GET"])
def video_job_status(job_id):
    # media_bus.get returns {"job_id","name":null,"status":null,"result":null,
    # "progress":null} for an unknown id, so the poller can distinguish
    # "not yet / unknown" from a real status. Enriched with a `placement` object
    # (same helper as GET /video/jobs / /video/studio/clips) when known — WHERE the
    # run executes — set only when present, so a light/unknown job is unaffected.
    view = media_bus.get(job_id)
    try:
        pl = job_placement(job_id, view.get("name") if isinstance(view, dict) else None)
        if pl and isinstance(view, dict):
            view["placement"] = pl
    except Exception:  # noqa: BLE001 — enrichment never breaks the status read
        pass
    return jsonify(view), 200


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
                "SELECT job_id, status, result_json, created, updated, progress_json "
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

    for job_id, status, result_json, created, updated, progress_json in rows:
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
        # Additive honesty (Active Processes): the live progress blob (incl. the
        # awaiting_capacity HOLD marker) + a placement object (WHERE the render
        # executes). Existing fields are untouched. `progress` is null unless the
        # runner has written one; `placement` is omitted unless known.
        progress = None
        if progress_json:
            try:
                progress = _json.loads(progress_json)
            except (ValueError, TypeError):
                progress = None
        clip = {
            "job_id": job_id,
            "status": status,
            "playable": bool(status == "done" and out),
            "created": created,
            "updated": updated,
            "output": out,
            "progress": progress,
        }
        try:
            pl = job_placement(job_id, "studio_i2v")
            if pl:
                clip["placement"] = pl
        except Exception:  # noqa: BLE001 — placement is best-effort per clip
            pass
        clips.append(clip)

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


# --------------------------------------------------------------------------- #
# 5d) POST /video/studio/clip/<job_id>/to-editor — Studio "Send to Editor" (k12).
#     A ONE-CLICK handoff of a produced clip, in a Filmora-native MP4, into a
#     stable inbox (studio.job.EDITOR_INBOX_ROOT) the operator's LAN Windows
#     workstation (Filmora desktop) picks up over a host-side Samba export.
#
#     CONSOLE-OPERATOR ONLY. The blanket /video gate (video_auth.install_video_gate)
#     already admitted EITHER an operator session OR a video-share credential by the
#     time this body runs — but this action WRITES INTO THE OPERATOR'S REAL EDITING
#     FOLDER, so a share-link guest (who CAN pass that gate) must not reach it. The
#     body re-checks operator_authenticated() and 403s otherwise: defense in depth,
#     independent of the blanket gate (mirrors _request_principal's lazy import).
#
#     The clip is resolved BY JOB ID exactly like GET /video/studio/clip/<id>
#     (archived -> 410, no completed clip -> 404, jail on the realpath under the
#     studio clips tree). The FORMAT GUARANTEE (ffprobe -> remux-or-transcode) lives
#     in the backbone (studio.editor_handoff), never inline here, per this module's
#     house rule. RE-SEND is NON-DESTRUCTIVE: unique_path appends _1, _2 … rather
#     than clobbering a prior copy the operator may have open in Filmora mid-edit.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/studio/clip/<job_id>/to-editor", methods=["POST"])
def video_studio_clip_to_editor(job_id):
    # Cheapest check first: CONSOLE OPERATOR ONLY (a share guest is refused here
    # even though it passed the blanket /video gate). Same lazy import idiom as
    # _request_principal() at the top of this module.
    from ..operator_auth import operator_authenticated
    if not operator_authenticated():
        return jsonify({"error": "operator session required"}), 403

    import json as _json
    import sqlite3
    from abstract_hugpy_dev.video_intel.studio.job import (
        DEFAULT_CLIPS_ROOT,
        EDITOR_INBOX_ROOT,
    )
    from abstract_hugpy_dev.video_intel.studio.editor_handoff import (
        editor_filename,
        send_to_editor,
    )
    from abstract_hugpy_dev.imports.src.utils import unique_path

    # Resolve the clip exactly like GET /video/studio/clip/<job_id>: an archived
    # clip's bytes still exist (never-delete) -> honest 410, not a bare 404; any
    # non-done/failed job -> no playable clip; the jail on the realpath keeps a
    # crafted/rehomed uri from becoming an arbitrary file read.
    if media_bus.is_archived(job_id):
        return jsonify({"error": "clip archived", "archived": True}), 410

    view = media_bus.get(job_id)
    result = view.get("result") if isinstance(view, dict) else None
    if not (isinstance(result, dict) and result.get("ok")):
        return jsonify({"error": "no completed studio clip for that job"}), 404

    outputs = result.get("outputs") or []
    first = outputs[0] if (outputs and isinstance(outputs[0], dict)) else {}
    uri = first.get("uri")
    if not uri or not isinstance(uri, str):
        return jsonify({"error": "job result carries no clip uri"}), 404

    resolved = os.path.realpath(uri)
    if not _is_within(resolved, DEFAULT_CLIPS_ROOT) or not os.path.isfile(resolved):
        return jsonify({"error": "clip not found"}), 404

    # Filename slug from the render's prompt/project. spec_json carries them; there
    # is no dedicated media_bus getter, so read the bus row read-only, exactly like
    # GET /video/studio/clip/<id>/detail. Best-effort — a slug of "clip" is a fine
    # fallback if the row/spec can't be read.
    title = None
    try:
        conn = sqlite3.connect(
            f"file:{media_bus.DB_PATH}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT spec_json FROM media_jobs WHERE job_id=? AND name='studio_i2v'",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            s = _json.loads(row[0])
            title = s.get("prompt") or s.get("project")
    except (sqlite3.Error, ValueError, TypeError):
        title = None

    filename = editor_filename(title, job_id)
    os.makedirs(EDITOR_INBOX_ROOT, exist_ok=True)
    dest = unique_path(os.path.join(EDITOR_INBOX_ROOT, filename))

    outcome = send_to_editor(resolved, dest)
    if not outcome.ok:
        return jsonify({
            "error": "could not prepare the clip for the editor",
            "code": outcome.code,
            "message": outcome.message,
        }), 500

    return jsonify({
        "ok": True,
        "filename": os.path.basename(outcome.dest),
        "path": outcome.dest,
        # "copy" (remux) vs "transcode" — surfaced so the console can say which ran.
        "mode": outcome.mode,
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
    if len(raws) > identity_profiles.MAX_SOURCE_IMAGES:
        return None, ({
            "error": f"at most {identity_profiles.MAX_SOURCE_IMAGES} reference_images are accepted"
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

    VERSION-AWARE DNA (VERSIONS slice): an identity now holds N named versions, each with
    its own promoted ``canonical``. The resolved DNA is the ACTIVE version's canonical by
    default; an optional ``identity_version`` (a version_id OR its name, e.g.
    "textured-01") names a specific version. Resolution precedence is:
    active/named version's canonical -> profile-level ``canonical`` -> ``reference_images``
    — so an un-versioned profile, an un-versioned caller, or an empty-canonical version all
    degrade to exactly the behavior above. A stale/unknown/archived ``identity_version`` is
    a clean 404.

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

    # VERSION-AWARE DNA (IDENTITY-VERSIONS-SLICE.md slice 2): the id_lock reference set is
    # the ACTIVE version's canonical by default; an explicit ``identity_version`` names a
    # specific one — matched by its version_id OR its name (e.g. "textured-01"). A stale /
    # unknown / archived version is a clean 404, never a silent wrong-identity (the public
    # ``versions`` list already omits archived versions, so those are unreachable here).
    # Precedence for the resolved DNA:
    #     chosen version's canonical -> profile-level canonical -> reference_images
    # so a pre-versions profile, an un-versioned caller, or a version with an empty
    # canonical all degrade to EXACTLY today's behavior (no regression on first load).
    versions = [v for v in (profile.get("versions") or []) if isinstance(v, dict)]
    req_version = body.get("identity_version")
    chosen_version = None
    if req_version is not None:
        if not isinstance(req_version, str) or not req_version.strip():
            return None, ({"error": "identity_version must be a version id or name string"}, 400)
        needle = req_version.strip()
        chosen_version = next(
            (v for v in versions
             if v.get("version_id") == needle or v.get("name") == needle),
            None,
        )
        if chosen_version is None:
            return None, ({"error": f"identity_version {req_version!r} not found for "
                                    f"identity_profile {slug!r}"}, 404)
    else:
        active_id = profile.get("active_version")
        if active_id:
            chosen_version = next(
                (v for v in versions if v.get("version_id") == active_id), None)

    version_canonical = (
        [p for p in (chosen_version.get("canonical") or []) if isinstance(p, str)]
        if chosen_version else []
    )
    profile_canonical = [p for p in (profile.get("canonical") or []) if isinstance(p, str)]
    canonical = version_canonical or profile_canonical
    canonical_default = canonical if canonical else list(profile.get("reference_images") or [])
    # Provenance: did the DNA come from the canonical RING (angle-ordered frames), or from
    # the raw ``reference_images`` uploads (unordered photos)? Only a ring may be angle-
    # strided down to the render cap — see the narrowing block at the end of this function.
    from_canonical_ring = bool(canonical)

    # VIEW-AWARE DNA (IDENTITY-3D-CONTINUITY-PLAN.md S2): an optional ``identity_view`` hint
    # — a semantic name ("back", "left-profile", …) OR an ``{azimuth_deg, elevation_deg?}``
    # object — selects the K angle-nearest frames from the identity's turntable RING instead
    # of the flat cardinals, so a "from behind" shot conditions on back-view frames. The bank
    # is a pure computed read over the chosen version's turntable reconstruction (the ring the
    # identity already rendered — no new state). Precedence + graceful degrade:
    #   * NO hint                       -> canonical_default (byte-identical to before) ;
    #   * hint + a turntable bank exists -> the K angle-nearest bank frames (angle-spread) ;
    #   * hint but NO bank (versionless / clay-only / legacy profile) -> canonical_default.
    # So a hintless call is zero-regression and a hinted call on an identity without a ring
    # still yields the working canonical set (defaults-are-promises). K matches the RENDER cap
    # (``MAX_RENDER_REFS``, i.e. up to 4 — what one id_lock render can consume; the canonical
    # STORAGE cap is 8 since 2026-07-16). An invalid hint is a clean 400.
    view_hint = body.get("identity_view")
    view_source = "canonical-default"
    chosen = canonical_default
    if view_hint is not None:
        azimuth_deg, view_err = identity_profiles.azimuth_for_view(view_hint)
        if view_err is not None:
            return None, ({"error": view_err}, 400)
        chosen_version_id = chosen_version.get("version_id") if chosen_version else None
        bank = identity_profiles.bank_views(profile, version_id=chosen_version_id)
        if bank:
            # K = the RENDER cap (4). These frames become one render's id_lock refs, so this
            # tracks MAX_RENDER_REFS, NOT the canonical/storage cap (8 since 2026-07-16).
            # The two were the same number (4) when this was written.
            picked = identity_profiles.nearest_bank_views(
                bank, azimuth_deg, identity_profiles.MAX_RENDER_REFS)
            chosen = [b["path"] for b in picked]
            view_source = "explicit-view"
        # else: no turntable ring for this version -> fall through on canonical_default.

    # A saved profile keeps positional slots for sources that never materialized
    # (recorded in missing_references) or that have since gone stale on disk. Drop
    # any reference that no longer exists so ONE bad path can't 404 the whole
    # identity downstream — the profile IS the identity, complete or not
    # (operator 2026-07-12). If this empties the set, the caller's non-empty
    # validation returns a clean 400 rather than a per-path "not found".
    chosen = [r for r in chosen if isinstance(r, str) and os.path.isfile(r)]

    # CANONICAL 8 vs RENDER 4 (2026-07-16) — narrowed HERE, once, for every caller.
    # The canonical STORAGE cap widened to 8 (the 45° ring) but ONE id_lock/VACE render
    # still consumes at most 4 refs (a model constraint: each ref becomes a VACE reference
    # latent). Every consumer of this resolver feeds a render channel that hard-rejects >4
    # (/video/enqueue and /video/studio/movie 400; identity_reconstruction_schema and
    # studio.job RAISE), so an 8-view identity would otherwise fail against its OWN approved
    # DNA. This function is the ONE place that knows the refs came from a canonical RING, so
    # it is the honest place to narrow: callers stay unchanged and can't forget.
    # Striding (not [:4]) keeps the sample angle-spread — an 8-view set yields the 4
    # cardinals (0/90/180/270), byte-identical DNA to a 4-view profile today, instead of the
    # lopsided front+right half-turn [:4] would take. <=4 sets pass through untouched, so
    # every profile on disk right now resolves EXACTLY as before (zero regression).
    # The explicit-view path already asked the bank for MAX_RENDER_REFS, so it is a no-op
    # there; this backstops the canonical-default path.
    # SCOPED TO A CANONICAL RING ON PURPOSE: when a profile has NO canonical, this resolver
    # falls back to the raw ``reference_images`` uploads (up to 12) — unordered photos, not
    # ring frames. Those must NOT be strided (there is no angle to spread across, and their
    # >4 handling is each caller's existing [:4]/400 contract). Only narrow what actually
    # came from the canonical ring, so the legacy upload path is untouched.
    n_before = len(chosen)
    if from_canonical_ring:
        chosen = identity_profiles.render_refs_from_canonical(chosen)
    # Record which DNA path served the resolve so a live enqueue is auditable (the return
    # SHAPE is unchanged; this is observability only). ``view_source`` is the S2 honesty
    # flag: explicit-view == the turntable ring served it; canonical-default == today's set.
    logger.info("identity DNA resolved: slug=%s view_source=%s n_refs=%d (from %d canonical)",
                slug.strip(), view_source, len(chosen), n_before)
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
    # ANGLE BANK summary (IDENTITY-3D-CONTINUITY-PLAN.md S1): surface, per version, WHAT
    # angles the identity actually carries (count + degrees_per_frame + azimuth range) so
    # the operator/UI can see the ring a view hint can select from. Purely ADDITIVE — a
    # new ``views`` key on each version object, computed read-only from the turntable
    # reconstruction; no existing key is removed or renamed. Scoped to this single-profile
    # GET (not the list) so the cheap list endpoint stays untaxed.
    for v in profile.get("versions") or []:
        if isinstance(v, dict):
            v["views"] = identity_profiles.views_summary(profile, version_id=v.get("version_id"))
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
        # Wire key stays ``reference_images`` (UI + tests read it); the store's
        # param is ``source_images`` (internal rename). Map here, or update_profile
        # raises "unexpected keyword argument 'reference_images'" (the PATCH 500).
        kwargs["source_images"] = resolved

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
# NOTE (keeper 2026-07-14): the OLD sheet/turntable-only reconstruction handler was
# CONSOLIDATED into the single angle-ring-aware handler below
# (video_identity_profile_reconstruction). Its @video_bp.route decorator is removed so
# exactly ONE rule serves this path — Werkzeug matched the first-registered rule, which
# shadowed the newer handler and rejected mode="angle-ring". Body retained (never-delete);
# full prior file at video_routes.py.bak-keeper-20260714.
def _retired_video_identity_profile_reconstructions(slug):
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
            # builder param is source_images (internal rename); passing
            # reference_images= raises "unexpected keyword argument".
            source_images=tuple(resolved_refs),
            views=tuple(views),
            base_prompt=prompt,
            seed=seed,
            mode=mode,
            # geometry (<=480p id_lock ceiling) + autofit VRAM are the schema defaults.
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400
    job_id = _video_enqueue("identity_reconstruction", spec)
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

# --------------------------------------------------------------------------- #
# 5d) POST /video/identity-profiles/<slug>/reconstruction — STAGE (b) update:
#     Support "mode": "angle-ring" in the base reconstruction handler.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction", methods=["POST"])
def video_identity_profile_reconstruction(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
    body = request.get_json(silent=True) or {}

    refs_in, perr = _reference_images_from_body({"identity_profile": slug})
    if perr is not None:
        payload, status = perr
        return jsonify(payload), status
    # id_lock reference channel accepts at most 4 (_MAX_CANONICAL_IMAGES). A profile may
    # hold up to 12 SOURCE images (and, since 2026-07-16, up to 8 CANONICAL views), so
    # narrow the resolver's canonical-preferred, existence-filtered set to 4 rather than
    # 400-ing with "at most 4 ... accepted".
    # A canonical RING is already narrowed to 4 (ring-strided, so the 4 cardinals rather
    # than a lopsided half-turn) inside _reference_images_from_body. This [:4] therefore
    # only ever trims a raw 12-image UPLOAD set — unordered photos with no angle to spread
    # across — so it keeps its original first-4 behavior, unchanged.
    refs_in = list(refs_in)[:4]
    if not refs_in:
        return jsonify({"error": "Profile has no valid reference images"}), 400
    resolved_refs, verr = _validate_profile_reference_images(refs_in)
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    prompt = body.get("prompt")
    if prompt is None:
        prompt = profile.get("notes") or ""
    elif not isinstance(prompt, str):
        return jsonify({"error": "prompt must be a string"}), 400

    # mode: "sheet" | "turntable" | "angle-ring"
    mode = body.get("mode", "sheet")
    if mode not in ("sheet", "turntable", "angle-ring"):
        return jsonify({"error": 'mode must be "sheet", "turntable", or "angle-ring"'}), 400

    # Extract angle-ring parameters when active
    angle_step_deg = body.get("angle_step_deg", 10)
    elevations_deg = body.get("elevations_deg", [0])

    if mode == "angle-ring":
        # 36 views at 10 deg step is default
        if not isinstance(angle_step_deg, int) or angle_step_deg <= 0:
            return jsonify({"error": "angle_step_deg must be a positive integer"}), 400
        if not isinstance(elevations_deg, list) or not elevations_deg:
            return jsonify({"error": "elevations_deg must be a non-empty list of integers"}), 400
        views = [f"angle_{deg}" for deg in range(0, 360, angle_step_deg)]
    else:
        views = body.get("views")
        if views is None:
            views = list(_DEFAULT_RECON_VIEWS)
        if not isinstance(views, list) or not views:
            return jsonify({"error": "views must be a non-empty list"}), 400

    seed = body.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        return jsonify({"error": "seed must be an int"}), 400

    # CLEANUP-PROMPT slice (C4 — the reachability wire): same body-or-gen_settings
    # precedence as the /generate route's vision_model/cleanup_prompt resolution — an
    # EXPLICIT request-body ``negative_prompt`` wins; else the identity's PERSISTED
    # gen_settings.negative_prompt (the Advanced-panel field); else "" (today's exact
    # call, defaults-are-promises). NOTE: make_identity_reconstruction is currently the
    # bare ``**kwargs`` passthrough (identity_reconstruction_schema.py:570), so this
    # value reaches IdentityReconstructionSpec.negative_prompt directly via the
    # dataclass constructor, unvalidated by the dead factory at :119 — acceptable here
    # since the field is a plain string with default "" (out of scope to fix the shadow).
    _gen_settings = profile.get("gen_settings") or {}
    negative_prompt = body.get("negative_prompt")
    if negative_prompt in (None, ""):
        negative_prompt = _gen_settings.get("negative_prompt", "")

    recon_id = "recon_" + secrets.token_hex(8)
    try:
        spec = make_identity_reconstruction(
            slug=slug,
            recon_id=recon_id,
            # builder param is source_images (internal rename); passing
            # reference_images= raises "unexpected keyword argument".
            source_images=tuple(resolved_refs),
            views=tuple(views),
            base_prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            mode=mode,
            angle_step_deg=angle_step_deg,
            elevations_deg=elevations_deg,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
        
    job_id = _video_enqueue("identity_reconstruction", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# 5e) PATCH /video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>
#     Approve or reject a specific angle tile. Updates the status locally in the
#     identity profile record.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>", methods=["PATCH"])
def video_identity_profile_view_status(slug, recon_id, view_id):
    if identity_profiles.get_profile(slug) is None:
        return jsonify({"error": "identity profile not found"}), 404
        
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    if status not in ("approved", "rejected"):
        return jsonify({"error": "status must be 'approved' or 'rejected'"}), 400

    try:
        profile = identity_profiles.update_reconstruction_view_status(
            slug=slug,
            recon_id=recon_id,
            view_id=view_id,
            status=status
        )
    except identity_profiles.ProfileError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), 400
        
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5f) POST /video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>/regenerate
#     Regenerate a single angle conditioned on nearby approved angle neighbors.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/views/<view_id>/regenerate", methods=["POST"])
def video_identity_profile_view_regenerate(slug, recon_id, view_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
        
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt") or (profile.get("notes") or "")
    seed = body.get("seed", secrets.randbelow(1000000))
    use_neighbors = bool(body.get("use_nearest_approved_neighbors", True))

    try:
        # Fetch the active reconstruction configuration to preserve overall context
        spec = identity_profiles.make_single_view_regeneration_spec(
            slug=slug,
            recon_id=recon_id,
            view_id=view_id,
            prompt=prompt,
            seed=seed,
            use_neighbors=use_neighbors
        )
    except identity_profiles.ProfileError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), 400

    job_id = _video_enqueue("identity_view_regenerate", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# Shared: build the cardinal-view -> path map for a mesh build, JAILED to the
# profile's OWN reference/canonical images. Central has no GPU — the relay runner
# reads exactly these paths — so the "default front = canonical[0]-on-disk else the
# first existing reference" rule + the path jail live in ONE place, shared by the
# per-reconstruction mesh route (5g) and the one-click /generate template.
#   Returns (view_map, candidates, None) on success, or (None, None, (payload, status))
#   when an explicit view path is not one of the profile's own images (a clean 400) —
#   the caller then returns ``jsonify(payload), status``.
#   ``candidates`` is the ordered list of ALL existing-on-disk source reference images
#   — populated ONLY when the caller did NOT explicitly assign ``views.front`` (an
#   explicit front assignment disables fleet-VLM auto-selection: candidates == []).
#   The relay runner (bus job context, never the request handler — a vision call is
#   ~5-10s/image and there can be up to 12) uses this to ask the fleet vision amenity
#   which candidate shows the character's FULL BODY and swap it in as front (keeper
#   2026-07-14: luigi ref_00 was a cropped waist-up portrait -> cut-off-legs mesh).
# --------------------------------------------------------------------------- #
def _resolve_profile_mesh_views(profile, body_views):
    canonical = [p for p in (profile.get("canonical") or []) if isinstance(p, str)]
    references = [p for p in (profile.get("reference_images") or []) if isinstance(p, str)]
    allowed = set(canonical) | set(references)  # the jail: only the profile's own images

    def _first_existing(paths):
        for p in paths:
            if os.path.isfile(p):
                return p
        return None

    # Default FRONT = the first existing SOURCE reference image — NEVER canonical.
    # canonical is GENERATION DNA and (via auto-promote) usually holds renders of the
    # PREVIOUS mesh's turntable; feeding it back into mesh reconstruction creates a
    # feedback loop that faithfully re-meshes the prior mesh, artifacts and all (bit
    # live 2026-07-14: luigi's 2nd generate re-meshed his 1st mesh's background slab
    # because front defaulted to canonical[0] — rembg 'applied' but the subject WAS the
    # old render). Reconstruction eats ORIGINAL photos; canonical is only reachable via
    # an EXPLICIT body views assignment below.
    existing_references = [p for p in references if os.path.isfile(p)]
    front = existing_references[0] if existing_references else None
    view_map: dict = {}
    if front is not None:
        view_map["front"] = front
    # Candidates for fleet-VLM auto-selection: every OTHER existing source reference,
    # in order (the current default front stays first-tried / the fallback). Cleared
    # below the moment an explicit front is assigned.
    candidates: list = list(existing_references)

    # Optional explicit assignment: {views: {front|right|back|left: <path>}}. Each path
    # must be one of the profile's own images (the jail) — an arbitrary path is a clean
    # 400, never accepted.
    if isinstance(body_views, dict):
        for vname, vpath in body_views.items():
            if vname not in _MESH_VIEW_NAMES:
                return None, None, ({"error": f"unknown view {vname!r}; expected one of "
                                        f"{list(_MESH_VIEW_NAMES)}"}, 400)
            if not isinstance(vpath, str) or vpath not in allowed:
                return None, None, ({"error": f"view {vname!r} must reference one of this "
                                        "profile's own reference/canonical images"}, 400)
            view_map[vname] = vpath
            if vname == "front":
                # An EXPLICIT front assignment is an operator override — auto-selection
                # never second-guesses it.
                candidates = []

    if "front" not in view_map:
        return None, None, ({"error": "profile has no usable reference image for the mesh "
                                "front view"}, 400)
    return view_map, candidates, None


# --------------------------------------------------------------------------- #
# 5g) POST /video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh
#     Trigger a ComfyUI mesh pipeline job (Hunyuan3D) utilizing selected, approved
#     anchor views.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh", methods=["POST"])
def video_identity_profile_build_mesh(slug, recon_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404

    body = request.get_json(silent=True) or {}

    # The mesh (+ optional turntable) is built from the identity's OWN reference/canonical
    # images — NEVER arbitrary paths (the shared jail below). (The old contract's
    # `views: [viewIds]` list is IGNORED, not rejected — the default front mapping applies
    # — so an older client never hard-400s here.)
    view_map, view_candidates, verr = _resolve_profile_mesh_views(profile, body.get("views"))
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    # Optional knobs — malformed optional params fall back to their defaults (a bad tuning
    # value should not hard-fail the build; make_identity_mesh re-validates regardless).
    def _pos_int(d, name, default):
        v = d.get(name, default)
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else default

    seed = body.get("seed", 12345)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        seed = 12345
    tt = body.get("turntable") if isinstance(body.get("turntable"), dict) else {}
    elev = tt.get("elevation_deg", 8.0)
    if not isinstance(elev, (int, float)) or isinstance(elev, bool):
        elev = 8.0

    # PER-IDENTITY VISION MODEL — same precedence as the one-click /generate route
    # (request-body ``vision_model`` > the identity's persisted gen_settings.vision_model
    # > None == fleet default). A surgical per-reconstruction mesh build runs the SAME
    # front-select step, so it honors the identity's chosen VL model too. Blank/None here
    # keeps today's behavior exactly (no ``model`` sent -> the 3B).
    _gen_settings = profile.get("gen_settings") or {}
    vision_model = body.get("vision_model")
    if vision_model in (None, ""):
        vision_model = _gen_settings.get("vision_model")
    if vision_model in (None, ""):
        # AUTO default (operator 2026-07-15): prefer a 7B VL when the fleet has one —
        # the 3B mislabeled a waist-up ref as full-body. None when no 7B is installed
        # (== fleet-default 3B); the relay degrades a failing 7B call back to the 3B.
        vision_model = identity_profiles.preferred_identity_vision_model()

    # CLEANUP / NEGATIVE prompt — SAME body-or-gen_settings precedence as /generate
    # (C4). This surgical per-reconstruction re-mesh runs the SAME T-pose front render
    # + studio path, so it must honor the identity's persisted cleanup/negative too —
    # else a re-mesh silently drops the "no object on her back" instruction. Blank =
    # today's exact behavior.
    cleanup_prompt = body.get("cleanup_prompt")
    if cleanup_prompt in (None, ""):
        cleanup_prompt = _gen_settings.get("cleanup_prompt", "")
    negative_prompt = body.get("negative_prompt")
    if negative_prompt in (None, ""):
        negative_prompt = _gen_settings.get("negative_prompt", "")

    try:
        spec = make_identity_mesh(
            slug=slug,
            recon_id=recon_id,
            view_sources=tuple(view_map.items()),
            seed=seed,
            num_inference_steps=_pos_int(body, "num_inference_steps", 30),
            octree_resolution=_pos_int(body, "octree_resolution", 380),
            texture=bool(body.get("texture", False)),
            chain_turntable=bool(body.get("chain_turntable", True)),
            frame_count=_pos_int(tt, "frame_count", 72),
            fps=_pos_int(tt, "fps", 24),
            width=_pos_int(tt, "width", 768),
            height=_pos_int(tt, "height", 768),
            elevation_deg=elev,
            transparent=bool(tt.get("transparent", False)),
            view_candidates=view_candidates,
            vision_model=vision_model,
            cleanup_prompt=cleanup_prompt,
            negative_prompt=negative_prompt,
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400

    # Seed the mesh state to "queued" so the GET mesh-status route + UI reflect the
    # in-flight build immediately (best-effort — a clean no-op if this recon_id has no
    # attached reconstruction yet; the relay attaches + records the terminal state).
    try:
        identity_profiles.set_mesh_state(slug, recon_id, {"status": "queued", "error": None})
    except identity_profiles.ProfileError:
        pass

    job_id = _video_enqueue("identity_mesh_build", spec)
    return jsonify({"job_id": job_id, "recon_id": recon_id}), 200


# --------------------------------------------------------------------------- #
# 5h) GET /video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh
#     Retrieve the generation status, active GLB path, and preview assets.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/reconstruction/<recon_id>/mesh", methods=["GET"])
def video_identity_profile_mesh_status(slug, recon_id):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404
        
    mesh_state = identity_profiles.get_mesh_state(slug, recon_id)
    if mesh_state is None:
        return jsonify({"status": "none"}), 200

    return jsonify(mesh_state), 200


# --------------------------------------------------------------------------- #
# 5g-vx) POST /video/identity-profiles/video-extract
#     char360 VIDEO -> per-character 360° view-sets, written back into identity profiles
#     (CHAR360-FEATURE-PLAN S3). This is its OWN relay job (identity_video_extract, mirrors
#     identity_mesh_build) — it relays a source clip to the standalone GPU render service
#     (which grew the video_extract kind in S2), polls it, downloads the per-character
#     view-sets, and writes them back. It is NOT the local identity_reconstruction job.
#
#     Body:
#       source          a MediaRef-shaped dict for the source VIDEO clip (the same shape
#                       the frame_extract route accepts — rehydrated via make_media_ref).
#                       Its uri must be an absolute path inside the storage jail; the runner
#                       forwards it to the service as video_path (ae + central share the
#                       mount, so a large clip is never base64-inflated through the body).
#       target          "create" (mint a NEW profile per detected character), "review"
#                       (CHARACTER-GROUPS-PLAN S1 — run char360 and RETURN the grouped views
#                       for curation, writing NO profile), or an EXISTING profile slug
#                       (append each character's view-set to it). Required.
#       char360_params? optional passthrough knobs for the service's Char360Params
#                       (stride / yolo_model / min_h_frac / cluster_dist / min_faces);
#                       unknown keys are dropped by the spec factory.
#     Returns {job_id, target} 200; a bad source/target is a clean 400; an unknown target
#     slug is a 404 (checked up front, mirroring the mesh route's profile guard).
#
#     REVIEW-mode RESULT CONTRACT (S1): the terminal result carries the grouped manifest —
#     GET /video/jobs/<job_id> -> result.groups =
#       {"n_characters": int,
#        "groups": [{"char": str, "face_centroid": [float]|null,
#                    "views": [{"url": <abs jailed path handle>, "yaw": float|null,
#                               "bin": int|null, "score": float|null}]}]}
#     Each view "url" is the persisted crop's media HANDLE (an absolute path under the
#     storage jail); the UI renders it via mediaBytesUrl(url) — the SAME GET /video/media
#     ?handle= route the profile canonical views use. No profile is created or appended.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/video-extract", methods=["POST"])
def video_identity_profile_video_extract():
    body = request.get_json(silent=True) or {}

    # source: a MediaRef-shaped dict (mirrors the frame_extract route — the console builds
    # the ref via POST /video/ingest, then hands us the ref dict). Rehydrate + validate it,
    # then jail its uri so the runner can never be pointed at an arbitrary file to relay.
    source_d = body.get("source")
    if not isinstance(source_d, dict):
        return jsonify({"error": "missing or invalid 'source' MediaRef (a video)"}), 400
    try:
        source = make_media_ref(**source_d)
    except (ValueError, TypeError) as exc:  # bad ref shape / non-abs uri = 400
        return jsonify({"error": f"invalid source MediaRef: {exc}"}), 400
    if source.kind != "video":
        return jsonify({"error": f"source must be a video; got kind={source.kind!r}"}), 400
    if _jail_resolve(source.uri) is None:
        return jsonify({"error": "source video is outside the storage jail"}), 400

    target = body.get("target")
    if not isinstance(target, str) or not target.strip():
        return jsonify({"error": "target is required ('create' or an existing profile slug)"}), 400
    target = target.strip()

    # An ADD target must name a LIVE profile — a clean 404 up front (rather than after a
    # full extract), mirroring the mesh route's get_profile guard. The correlation id handed
    # to the service is the slug (add) or a synthesized id (create/review — the runner
    # synthesizes one when identity_id is blank, but pass an explicit honest one here too).
    #
    # "review" (CHARACTER-GROUPS-PLAN S1) is NON-COMMITTING: like "create" it names no
    # existing profile, so it is NOT profile-checked here — it runs char360 and returns the
    # grouped views for curation WITHOUT writing anything. The grouped manifest rides the
    # job's terminal result and is read via GET /video/jobs/<job_id> -> result.groups.
    if target in ("create", "review"):
        identity_id = None  # the runner synthesizes videoextract-<job_id>
    else:
        if identity_profiles.get_profile(target) is None:
            return jsonify({"error": f"identity profile {target!r} not found"}), 404
        identity_id = target

    char360_params = body.get("char360_params")
    if char360_params is not None and not isinstance(char360_params, dict):
        return jsonify({"error": "char360_params must be an object"}), 400

    try:
        spec = make_identity_video_extract(
            source=source,
            target=target,
            char360_params=char360_params,
            identity_id=identity_id,
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400

    job_id = _video_enqueue("identity_video_extract", spec)
    return jsonify({"job_id": job_id, "target": target}), 200


# --------------------------------------------------------------------------- #
# 5h) POST /video/identity-profiles/from-groups
#     CHARACTER-GROUPS-PLAN S3 — commit the curated char360 groups (S1's REVIEW
#     manifest, edited client-side by S2) into identity profiles. ONE profile is
#     created per submitted group, through the EXACT SAME validation + copy path
#     as the single-profile create route above
#     (_validate_profile_reference_images -> identity_profiles.create_profile):
#     no re-invented staging, no new jail rule. The reference-image entries are
#     the jailed crop handles S1 persisted under
#     <IDENTITIES_HOME>/_char360_extracts/<job_id>/char_NN/<file> (servable via
#     GET /video/media?handle=), but any jail-valid image path is accepted —
#     this route does not care how a handle was produced.
#
#     Body: {"groups": [{"name"?: str, "reference_images": [<handle>, ...]}]}
#     A missing/blank name is derived as "Character N" (1-based index over the
#     submitted list) so every group still gets a stable, url-safe default slug.
#
#     Returns 200 ALWAYS (a bad group is errors-as-data, never a batch failure) —
#       {"results": [{"name": str, "ok": bool, "slug"?: str, "error"?: str}]}
#     — one result per input group, ORDER-PRESERVING. A validation failure (bad/
#     missing/non-image reference, jail escape, dup slug, ...) records
#     ok:false + error for THAT group only; the rest of the batch still commits.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/from-groups", methods=["POST"])
def video_identity_profiles_from_groups():
    body = request.get_json(silent=True) or {}
    groups = body.get("groups")
    if not isinstance(groups, list) or not groups:
        return jsonify({"error": "groups must be a non-empty list"}), 400

    results = []
    for idx, group in enumerate(groups):
        default_name = f"Character {idx + 1}"
        if not isinstance(group, dict):
            results.append({"name": default_name, "ok": False, "error": "group must be an object"})
            continue
        raw_name = group.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else default_name

        resolved, err = _validate_profile_reference_images(group.get("reference_images"))
        if err is not None:
            payload, _status = err
            results.append({"name": name, "ok": False, "error": payload.get("error", "invalid reference_images")})
            continue

        try:
            profile = identity_profiles.create_profile(name, resolved)
        except identity_profiles.ProfileError as exc:  # dup slug / bad shape = errors-as-data
            results.append({"name": name, "ok": False, "error": str(exc)})
            continue

        results.append({"name": name, "ok": True, "slug": profile["slug"]})

    return jsonify({"results": results}), 200


# --------------------------------------------------------------------------- #
# 5i) POST /video/identity-profiles/<slug>/generate
#     ONE-CLICK FULL IDENTITY GENERATION — the template that turns a saved identity
#     PROFILE into a complete 3D identity in a single action: a Hunyuan3D mesh, a
#     true-360° Blender turntable, and (by default) auto-promoted CANONICAL reference
#     angles. Usable on ANY profile that carries 1..12 reference images — NO prior
#     reconstruction / character-sheet / angle-ring approval is required (it mints a
#     fresh recon_id and drives the whole chain off the profile's own refs).
#
#     Body — all optional; the DEFAULTS are the happy path (a bare {} is the intended
#     call):
#       views?      {front|right|back|left: <path>} — each an EXPLICIT override; every
#                   path must be one of THIS profile's own reference/canonical images
#                   (the shared jail — an arbitrary path is a clean 400). Omitted views
#                   fall back to the default front (canonical[0]-on-disk else the first
#                   existing reference image), exactly like the 5g mesh route.
#       texture?    bool (default TRUE) — bake a texture onto the mesh (the template's happy path).
#       turntable?  {frame_count?, fps?, width?, height?, elevation_deg?, transparent?}
#                   — orbit render knobs (defaults 72 / 24 / 768 / 768 / 8.0 / False).
#       auto_promote? bool (default True) — promote the 4 cardinal turntable frames to
#                   canonical AFTER the render, but ONLY when canonical is still empty
#                   (a curated canonical set is never clobbered; the relay re-reads it).
#     Always chains the turntable (the "full identity" always renders mesh -> 360°).
#     Returns {job_id, recon_id} 200; 404 for an unknown profile.
# --------------------------------------------------------------------------- #
_POSE_CHOICES = ("none", "t-pose")

# T-pose render geometry — the Wan-VACE id_lock ceiling is 480p (a 512² id_lock render
# fails ``no_capable_model``); a square 480×480 portrait matches the movie/reconstruction
# id_lock default, so the T-pose still snaps to it. The capability probe builds a spec at
# THIS geometry so the routing decision it makes matches the render the relay will run.
# Shared with runners/identity_render_relay._render_pose_front (kept identical there).
_TPOSE_RENDER_W, _TPOSE_RENDER_H, _TPOSE_RENDER_FPS = 480, 480, 16


def _pose_stage_capable(slug: str) -> bool:
    """Whether the T-pose pose-normalization RENDER STAGE can actually run on this
    deployment RIGHT NOW.

    Slice 5 (IDENTITY-VERSIONS-SLICE.md) is the render stage: the relay renders ONE
    id_lock T-pose STILL on the Wan-VACE path (studio worker on ae, ~6GB free on the
    3090) and meshes THAT instead of the crossed-arm source photo. Whether that render
    will land on a GPU worker — versus silently falling to central's GPU-less in-process
    path, where an id_lock render can only fail — is the EXACT same question the studio
    delegation layer already answers for every VACE render: ``should_delegate(spec)``
    returns True iff (a) a studio worker is RESOLVABLE (``HUGPY_STUDIO_WORKER`` set, or
    the registry-based resolver once studio models are first-class rows) AND (b) the
    request binds a REAL (non-synthetic) VACE model at the worker's autofit budget. That
    predicate reads the in-process worker registry + the capability router — no network
    probe — so it is cheap enough to run on the request path (defaults-are-promises: we
    only advertise "capable" when the render will really delegate to a GPU that owns the
    model).

    We reuse that decision instead of hand-rolling a ``comfy.id_lock`` capability check:
    the T-pose still is a Wan-VACE reference-to-video render, NOT a ComfyUI-IPAdapter
    still, so ``comfy.id_lock`` is the wrong signal — the right signal is "will an
    id_lock studio render delegate to a worker that serves the VACE model". We build a
    REPRESENTATIVE id_lock ``studio_i2v`` spec at the reconstruction id_lock ceiling
    (480×480, the same geometry the relay's T-pose render will use) and ask
    ``should_delegate`` about it.

    FAIL-CLOSED: any exception — an import failure, a registry read hiccup, a router
    error — degrades to False so a ``pose: "t-pose"`` request falls back to the normal
    front (honest not-capable notice) rather than enqueuing a build that would fail its
    pose render. ``slug`` is accepted for signature stability (a future per-identity gate
    could consult it) but the capability is deployment-wide today.

    Kept a FUNCTION so the probe lives in ONE place and a test can exercise the capable
    branch by monkeypatching it (or by pointing HUGPY_STUDIO_WORKER at a real registry
    row)."""
    try:
        # Lazy imports — the studio spine + registry are heavy and must not load at
        # module import time (this route module imports at app boot). Mirrors the
        # relay's lazy-import discipline.
        from ...video_intel.runners.studio_i2v import should_delegate
        from ...video_intel.studio.job import make_studio_i2v

        # A representative id_lock still spec at the VACE 480p id_lock ceiling — the SAME
        # geometry the relay's T-pose render uses (see _render_pose_front). A single
        # placeholder reference image is enough for the routing decision (should_delegate
        # reads the worker registry + the capability router; it does not open the file).
        probe = make_studio_i2v(
            capability="id_lock",
            width=_TPOSE_RENDER_W,
            height=_TPOSE_RENDER_H,
            fps=_TPOSE_RENDER_FPS,
            vram_budget_gb=None,        # autofit — exactly what the real render uses
            seed=0,
            prompt="capability probe",
            reference_images=("__probe__",),
        )
        return bool(should_delegate(probe))
    except Exception:  # noqa: BLE001 — fail-closed: any probe failure => not capable
        logger.debug("pose-stage capability probe failed for %s", slug, exc_info=True)
        return False


@video_bp.route("/video/identity-profiles/<slug>/generate", methods=["POST"])
def video_identity_profile_generate(slug):
    profile = identity_profiles.get_profile(slug)
    if profile is None:
        return jsonify({"error": "identity profile not found"}), 404

    body = request.get_json(silent=True) or {}

    # Cardinal view map, JAILED to the profile's own images (shared with the 5g route).
    view_map, view_candidates, verr = _resolve_profile_mesh_views(profile, body.get("views"))
    if verr is not None:
        payload, status = verr
        return jsonify(payload), status

    # POSE NORMALIZATION (IDENTITY-VERSIONS-SLICE.md slice 3): optional pose, validated
    # none|t-pose. The t-pose RENDER STAGE (render an id_lock T-pose still and mesh THAT to
    # clear crossed-arm occlusion) is slice 5 — gated here behind a capability check.
    #   * pose="none" (or absent) -> today's behavior EXACTLY (no extra response fields).
    #   * pose="t-pose" + capable  -> the spec carries pose so the relay renders the T-pose
    #                                 front; response notes it applied.
    #   * pose="t-pose" + NOT capable (today) -> the build STILL proceeds off the normal
    #     front (honest fallback), and the response carries a structured not-capable notice
    #     so the caller knows the normalization was not applied. A malformed pose is a 400.
    pose_req = body.get("pose", "none")
    if pose_req is None:
        pose_req = "none"
    if not isinstance(pose_req, str) or pose_req not in _POSE_CHOICES:
        return jsonify({"error": f"pose must be one of {list(_POSE_CHOICES)}"}), 400
    effective_pose = "none"
    pose_notice = None
    if pose_req == "t-pose":
        if _pose_stage_capable(slug):
            effective_pose = "t-pose"
            pose_notice = {"requested": "t-pose", "applied": True, "capable": True}
        else:
            pose_notice = {
                "requested": "t-pose",
                "applied": False,
                "capable": False,
                "code": "pose_stage_unavailable",
                "message": ("the T-pose normalization render stage is not yet available "
                            "on this deployment; generated from the source pose instead"),
            }

    # A fresh, self-describing reconstruction id — the whole chain (mesh + turntable +
    # canonical) hangs off it; no prior reconstruction is needed.
    recon_id = "identity_" + secrets.token_hex(8)

    # Optional knobs — a malformed optional value falls back to its default (a bad tuning
    # value should not hard-fail the build; make_identity_mesh re-validates regardless).
    def _pos_int(d, name, default):
        v = d.get(name, default)
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else default

    tt = body.get("turntable") if isinstance(body.get("turntable"), dict) else {}
    elev = tt.get("elevation_deg", 8.0)
    if not isinstance(elev, (int, float)) or isinstance(elev, bool):
        elev = 8.0

    # PER-IDENTITY VISION MODEL (operator-requested): the VL model the mesh relay's
    # FRONT-SELECT step uses to pick the full-body reference before meshing. Precedence:
    # an EXPLICIT request-body ``vision_model`` wins; else the identity's PERSISTED
    # ``gen_settings.vision_model`` (what the Settings tab saves); else None (== the
    # fleet-default VL model — the relay sends no ``model`` field, byte-identical to before
    # this setting existed). ``profile`` is already the PUBLIC shape here, so its
    # ``gen_settings`` is the full defaulted block; ``make_identity_mesh`` normalizes
    # ""/whitespace to None. This is why a BARE one-click still honors the identity's saved
    # choice even when the UI omits the field from the body.
    _gen_settings = profile.get("gen_settings") or {}
    vision_model = body.get("vision_model")
    if vision_model in (None, ""):
        vision_model = _gen_settings.get("vision_model")
    if vision_model in (None, ""):
        # AUTO default (operator 2026-07-15): prefer a 7B VL when the fleet has one —
        # the 3B mislabeled a waist-up ref as full-body. None when no 7B is installed
        # (== fleet-default 3B); the relay degrades a failing 7B call back to the 3B.
        vision_model = identity_profiles.preferred_identity_vision_model()

    # CLEANUP-PROMPT slice (C4 — the reachability wire): same precedence pattern as
    # vision_model above — an EXPLICIT request-body value wins; else the identity's
    # PERSISTED gen_settings value (what the new Advanced-panel field saves); else ""
    # (== today's exact render, defaults-are-promises). make_identity_mesh coerces
    # None -> "" regardless, so a bare one-click still honors the profile's saved
    # cleanup/negative steer even when the UI body omits the fields.
    cleanup_prompt = body.get("cleanup_prompt")
    if cleanup_prompt in (None, ""):
        cleanup_prompt = _gen_settings.get("cleanup_prompt", "")
    negative_prompt = body.get("negative_prompt")
    if negative_prompt in (None, ""):
        negative_prompt = _gen_settings.get("negative_prompt", "")

    try:
        spec = make_identity_mesh(
            slug=slug,
            recon_id=recon_id,
            view_sources=tuple(view_map.items()),
            # TEXTURE DEFAULTS TRUE on the one-click template (operator 2026-07-14 night:
            # only the explicitly-textured run "is textured correctly of the 3" — the UI
            # sends a bare body, so the default IS the promise; the paint path is proven
            # and costs ~1-2 extra minutes). ``"texture": false`` opts a run out; the
            # per-reconstruction 5g mesh route keeps texture opt-in for surgical builds.
            texture=bool(body.get("texture", True)),
            chain_turntable=True,   # the full-identity template always renders the 360°
            auto_promote=bool(body.get("auto_promote", True)),
            frame_count=_pos_int(tt, "frame_count", 72),
            fps=_pos_int(tt, "fps", 24),
            width=_pos_int(tt, "width", 768),
            height=_pos_int(tt, "height", 768),
            elevation_deg=elev,
            transparent=bool(tt.get("transparent", False)),
            view_candidates=view_candidates,
            pose=effective_pose,
            vision_model=vision_model,
            cleanup_prompt=cleanup_prompt,
            negative_prompt=negative_prompt,
        )
    except (ValueError, TypeError) as exc:  # bad fields = 400
        return jsonify({"error": str(exc)}), 400

    # Seed mesh state to "queued" so GET .../reconstruction/<recon_id>/mesh + the UI
    # reflect the in-flight build immediately (set_mesh_state creates the record for this
    # brand-new recon_id — the mesh-first flow — then the relay records the terminal state).
    try:
        identity_profiles.set_mesh_state(slug, recon_id, {"status": "queued", "error": None})
    except identity_profiles.ProfileError:
        pass

    job_id = _video_enqueue("identity_mesh_build", spec)
    resp = {"job_id": job_id, "recon_id": recon_id}
    # Only surface a ``pose`` block when t-pose was explicitly requested — a bare click
    # (pose="none") keeps the exact {job_id, recon_id} shape it has today.
    if pose_notice is not None:
        resp["pose"] = pose_notice
    return jsonify(resp), 200


# --------------------------------------------------------------------------- #
# 5j) PATCH /video/identity-profiles/<slug>/settings — VERSIONS slice: persist the
#     per-identity generation defaults the left-column Settings tab edits and a
#     bare /generate click honors. Body IS the partial gen_settings object itself
#     (NOT nested under a "gen_settings" key — matches identityProfileSettingsUrl's
#     client, which PATCHes `fields` directly). A true partial merge — an omitted
#     key is left untouched (identity_profiles.set_gen_settings's contract); the
#     store rejects an unknown key, a wrong-typed value, an out-of-enum pose, or a
#     front_ref outside the profile's own reference images, all as a clean 400
#     (ProfileError, never a 500). Unknown slug -> 404, same message/shape as the
#     other identity-profile routes.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/settings", methods=["PATCH"])
def video_identity_profile_settings(slug):
    if identity_profiles.get_profile(slug) is None:
        return jsonify({"error": "identity profile not found"}), 404

    body = request.get_json(silent=True) or {}
    try:
        profile = identity_profiles.set_gen_settings(slug, body)
    except identity_profiles.ProfileError as exc:  # unknown key / bad type = clean 400
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:  # lost a race with a concurrent archive
        return jsonify({"error": "identity profile not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5k) POST /video/identity-profiles/<slug>/versions/<version_id>/activate —
#     VERSIONS slice: point the identity's ACTIVE version at <version_id> (the
#     id_lock DNA source future generations resolve to by default, via
#     _reference_images_from_body). No body. 404 when the slug is unknown OR
#     version_id names no active (non-archived) version of it —
#     set_active_version returns None for either case; the slug is checked first
#     so an unknown slug gets the same message every other identity-profile route
#     gives it, and an unknown/archived version_id is called out by name.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/versions/<version_id>/activate", methods=["POST"])
def video_identity_profile_activate_version(slug, version_id):
    if identity_profiles.get_profile(slug) is None:
        return jsonify({"error": "identity profile not found"}), 404

    profile = identity_profiles.set_active_version(slug, version_id)
    if profile is None:
        return jsonify({"error": f"version {version_id!r} not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5l) PATCH /video/identity-profiles/<slug>/versions/<version_id> — VERSIONS
#     slice: rename/annotate one version. Body {name?, notes?}, both optional (a
#     true partial update — an omitted key is left untouched, mirrors the
#     PATCH /<slug> profile-edit route's **kwargs idiom above). A blank name is a
#     clean 400 (checked here exactly like the profile-edit route, before ever
#     calling the store — update_version also guards it, belt-and-suspenders).
#     404 when the slug or version_id is unknown/archived.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/versions/<version_id>", methods=["PATCH"])
def video_identity_profile_update_version(slug, version_id):
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

    try:
        profile = identity_profiles.update_version(slug, version_id, **kwargs)
    except identity_profiles.ProfileError as exc:  # errors-as-data, never a 500
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:
        return jsonify({"error": f"version {version_id!r} not found"}), 404
    return jsonify({"profile": profile}), 200


# --------------------------------------------------------------------------- #
# 5m) DELETE /video/identity-profiles/<slug>/versions/<version_id> — VERSIONS
#     slice: ARCHIVE a version (never-delete: flagged, bytes kept, dropped from
#     the wire list). The store REFUSES the clay base (the geometric ground
#     truth) and the currently ACTIVE version — either surfaces here as the
#     store's own ProfileError message/code turned into a clean 400 (the check
#     lives in identity_profiles.archive_version; this route does not duplicate
#     it). 404 when the slug or version_id is unknown/already archived.
# --------------------------------------------------------------------------- #
@video_bp.route("/video/identity-profiles/<slug>/versions/<version_id>", methods=["DELETE"])
def video_identity_profile_archive_version(slug, version_id):
    if identity_profiles.get_profile(slug) is None:
        return jsonify({"error": "identity profile not found"}), 404

    try:
        profile = identity_profiles.archive_version(slug, version_id)
    except identity_profiles.ProfileError as exc:  # base/active refusal = clean 400
        return jsonify({"error": str(exc), "code": exc.code}), 400
    if profile is None:
        return jsonify({"error": f"version {version_id!r} not found"}), 404
    return jsonify({"profile": profile}), 200
