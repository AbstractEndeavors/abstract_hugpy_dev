"""Pure ``(studio, generate_studio_movie)`` runner — an ordered strip of REAL studio
clips conjoined at splice points, like a single NLE timeline ROW.

``run_generate_studio_movie(spec, job_id) -> JobResult``. A FAT orchestrator that
sequences the movie's segments INLINE (NOT an orchestrator-of-child-jobs, which
would deadlock the single-daemon bus — it follows ``runners.movie``'s inline
pattern). media_jobs.db stays single-writer in the bus; this runner is pure
``(spec, job_id) -> JobResult``.

NLE-ROW SEMANTICS. A studio movie is ``[segment 0 | segment 1 | …]``: an ordered
strip of studio ``produce_clip`` outputs conjoined at splice points. Per segment,
in timeline order:
  a. Segment 0 renders t2v (or i2v when a movie-level ``start_image`` is given).
     Each LATER segment renders i2v, conditioned on ONE still — the **branch
     frame** of the PREVIOUS segment's clip (``branch_frame: int | null``; null ⇒
     the parent's LAST frame). The still is extracted with ffmpeg (a frame-accurate
     ``select=eq(n\\,B)`` pluck) and handed to ``produce_clip`` as the i2v
     ``start_image``.
  b. The render goes through the SAME studio boundary the single-clip bus job uses:
     ``runners.studio_i2v.run_produce_clip`` (router -> manifest -> runner ->
     content-addressed clip). RESUME is content-addressed INSIDE ``produce_clip``:
     an identical segment spec re-run returns the existing clip (``resumed=True``),
     no regeneration — so a re-enqueue of the same movie skips/reuses every segment.
     Each segment renders under its OWN ``out_root`` subtree
     (``<movie_root>/segment_NN``) because ``start_image`` is NOT part of the studio
     content_hash (only prompt/seed/geometry/source_video/… are); isolating the
     out_root guarantees two segments never collide on a shared hash and each
     resumes independently.
  c. Between segments: honor ``is_cancelling(job_id)`` AND ``spec.time_budget_s``
     (this runner owns its OWN wall-clock — the bus has no timeout/reaper). The
     cancel probe is also threaded DOWN into ``produce_clip`` so a mid-render cancel
     aborts before a clip is written (Err(CANCELLED), errors-as-data).
  d. Emit NESTED movie progress via ``media_bus.set_progress``.

NON-DESTRUCTIVE TRIM (metadata, honored at ASSEMBLY only). The per-segment clip
files stay WHOLE — they are content-addressed and never modified. A mid-frame
branch means the assembled movie uses the PARENT clip only UP TO the branch frame;
that trim is applied only when building ``movie.mp4`` (by re-encoding a trimmed
COPY of the parent into a work dir), never by re-rendering. The retained parent
length is ``trim_frames = branch_frame + 1`` (frames ``[0, branch_frame]``
inclusive); a null branch resolves to the parent's LAST frame, so the parent plays
in FULL. The LEAF segment (no child) plays in full. Concat is ffmpeg's concat
DEMUXER over the (uniformly re-encoded) contribution clips.

``movie.json`` sidecar records the full node list + per-joint
``{branch_frame, trim_frames}`` + a DRIFT note: i2v-still conditioning carries a
frame, NOT motion — VACE-extend is the planned upgrade.

DRIFT / v0 simplifications (also in ``studio_movie_schema``'s header + the report):
  * one movie-level tier (``vram_budget_gb``) applied to every segment;
  * geometry (w/h/fps) uniform across segments (they concat into one row);
  * per-segment renders run IN-PROCESS through ``run_produce_clip`` (no
    HUGPY_STUDIO_WORKER delegation, unlike the single-clip ``run_studio_i2v``) —
    per-segment GPU-worker offload is planned growth. On this GPU-less box the
    synthetic tier renders every segment fine.

Pure discipline (map §6): EXPECTED failures are returned as
``JobResult(ok=False, JobError(...))`` — DATA, never a raise. The studio-spine
import stays LAZY (``run_produce_clip`` does its own lazy studio imports); nothing
heavy is pulled at this module's import time, so it can never break app boot.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, replace

from abstract_hugpy_dev._platform.binaries import resolve_bin
from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT
from abstract_hugpy_dev.imports.src.utils import slugify

from ..media_store import ingest
from ..result_schema import JobError, JobResult
from ..studio.job import make_studio_i2v
from ..studio_movie_schema import StudioMovieSpec
# The studio-spine boundary. studio_i2v's module top is dependency-light (its
# studio/numpy imports are lazy INSIDE its functions), so importing these here —
# and thus at app boot via runners/__init__ — can never break boot. run_produce_clip
# builds env+seeds from a StudioI2VSpec and calls produce_clip (content-addressed
# resume inside); _stage_error_to_job_error is the ONE StageError -> JobError adapter
# (reused so a movie segment's Err classifies identically to a single-clip job).
from .studio_i2v import _stage_error_to_job_error, run_produce_clip

logger = logging.getLogger(__name__)

# Studio movies land under the media-store root (inside ingest's storage jail) so
# every segment clip + the final movie.mp4 is cataloged like any other media output.
STUDIO_MOVIE_ROOT = os.path.join(DEFAULT_ROOT, "video_intel", "studio_movies")

_DRIFT_NOTE = ("i2v-still conditioning: each segment is conditioned on ONE frame of "
               "its parent's clip — motion is NOT carried across the splice; "
               "VACE-extend is the planned upgrade.")


# --------------------------------------------------------------------------- #
# ffmpeg helpers — a frame-accurate branch pluck, a frame-accurate trim, and the
# concat-demux stitch. Each is errors-as-data (returns (ok, stderr_tail)) except
# the concat, which RAISES so the caller wraps it into a movie_assembly_failed
# JobError (never a raise across the job boundary).
# --------------------------------------------------------------------------- #
def _extract_frame_at(clip_path: str, frame_index: int, dest_png: str) -> "tuple[bool, str]":
    """Pluck the ``frame_index``-th (0-based) frame of ``clip_path`` to ``dest_png``.

    Frame-accurate via the ``select=eq(n\\,IDX)`` filter + ``-frames:v 1`` (the
    existing ``ffmpeg_frames`` helper is an fps RESAMPLE, not an index pluck, so it
    is unsuitable — this is the small dedicated extractor the header points at).
    Never raises on a plain ffmpeg failure (errors-as-data). Returns
    (ok, stderr_tail)."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(os.path.dirname(dest_png), exist_ok=True)
    # -vsync 0 / -frames:v 1 with a select that keeps only frame IDX yields exactly
    # that one still. -q:v 2 = high-quality mjpeg/png quantizer.
    cmd = [
        ffmpeg, "-y",
        "-i", clip_path,
        "-vf", f"select=eq(n\\,{int(frame_index)})",
        "-vsync", "0",
        "-frames:v", "1",
        "-q:v", "2",
        dest_png,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (result.returncode == 0 and os.path.isfile(dest_png)
          and os.path.getsize(dest_png) > 0)
    return ok, (result.stderr or "")[-500:]


def _trim_clip(src_mp4: str, dst_mp4: str, n_frames: int, fps: int) -> "tuple[bool, str]":
    """Re-encode the FIRST ``n_frames`` frames of ``src_mp4`` into ``dst_mp4``.

    Frame-accurate via ``-frames:v n`` on decoded output. Re-encodes with the SAME
    house H.264/yuv420p invocation the synthetic runner uses, so EVERY contribution
    clip (trimmed parents AND the untrimmed leaf, which is also routed through here
    with n = its full length) carries identical codec params — the concat DEMUXER's
    ``-c copy`` is then valid. Never raises on a plain ffmpeg failure. Returns
    (ok, stderr_tail)."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", src_mp4,
        "-frames:v", str(int(n_frames)),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(int(fps)),
        "-movflags", "+faststart",
        dst_mp4,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (result.returncode == 0 and os.path.isfile(dst_mp4)
          and os.path.getsize(dst_mp4) > 0)
    return ok, (result.stderr or "")[-500:]


def _concat_clips(contribution_mp4s: "list[str]", movie_mp4: str, work_dir: str) -> None:
    """Stitch the contribution mp4s into ``movie_mp4`` via ffmpeg's concat DEMUXER
    (stream-copy — all inputs went through ``_trim_clip`` so they share codec params,
    making ``-c copy`` valid + fast). Mirrors ``runners.movie._concat_movie``. RAISES
    on failure — the caller wraps it into a movie_assembly_failed JobError."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w") as fh:
        for p in contribution_mp4s:
            safe = p.replace("'", "'\\''")   # concat-demux single-quote escaping
            fh.write(f"file '{safe}'\n")
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        movie_mp4,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(movie_mp4):
        raise RuntimeError(
            f"ffmpeg concat failed rc={result.returncode}: {(result.stderr or '')[-500:]}")


# --------------------------------------------------------------------------- #
# assembly — trim each parent at its child's branch point (metadata honored at
# concat time only), leaf plays full, concat -> movie.mp4. Best-effort / never
# raises across the job boundary (a stitch hiccup can only lose the partial, never
# mask the caller's real error).
# --------------------------------------------------------------------------- #
def _assemble_movie(movie_root: str, work_dir: str, seg_records: "list[dict]",
                    fps: int, job_id: str) -> "dict":
    """(Re)stitch whatever segments are complete SO FAR into
    ``<movie_root>/movie.mp4``, honoring each parent's TRIM at its child's branch
    point. Returns an ``assembly`` dict ``{movie, total_frames, joints}`` (movie is
    None when nothing could be stitched). NEVER raises — a stitch failure is logged
    and swallowed so a partial save is additive only."""
    completed = [r for r in seg_records if r["status"] in ("done", "resumed")]
    assembly = {"movie": None, "total_frames": 0, "joints": []}
    if not completed:
        return assembly

    # Per-joint trim record + each segment's contribution frame count. A segment
    # with a NEXT completed segment (its child, linear chain) is trimmed to
    # child.resolved_branch + 1; the last completed segment plays FULL.
    contributions: "list[tuple[str, int]]" = []   # (segment clip path, n frames)
    joints: "list[dict]" = []
    for p, rec in enumerate(completed):
        if p + 1 < len(completed):
            child = completed[p + 1]
            rb = child["resolved_branch"]          # branch INTO this segment
            trim_frames = int(rb) + 1
            joints.append({
                "parent_segment_id": rec["segment_id"],
                "child_segment_id": child["segment_id"],
                "branch_frame": int(rb),
                "trim_frames": trim_frames,
            })
        else:
            trim_frames = int(rec["frames"])       # leaf: full clip
        contributions.append((rec["clip_path"], trim_frames))

    # Materialize each contribution as a uniformly re-encoded clip in the work dir,
    # then concat. Non-destructive: the source clips are never touched.
    os.makedirs(work_dir, exist_ok=True)
    contrib_paths: "list[str]" = []
    total = 0
    try:
        for i, (src, n) in enumerate(contributions):
            dst = os.path.join(work_dir, f"contrib_{i:02d}.mp4")
            ok, tail = _trim_clip(src, dst, n, fps)
            if not ok:
                logger.warning("studio movie %s: contribution trim %d FAILED "
                               "(non-fatal): %s", job_id, i, tail)
                # keep whatever we could stitch before this failure
                break
            contrib_paths.append(dst)
            total += n
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        if len(contrib_paths) == 1:
            shutil.copyfile(contrib_paths[0], movie_mp4)
            assembly["movie"] = "movie.mp4"
            assembly["total_frames"] = total
        elif len(contrib_paths) >= 2:
            _concat_clips(contrib_paths, movie_mp4, work_dir)
            assembly["movie"] = "movie.mp4"
            assembly["total_frames"] = total
    except Exception as exc:  # a stitch failure must NEVER mask the caller's error
        logger.warning("studio movie %s: assembly FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        if os.path.isfile(movie_mp4):
            assembly["movie"] = "movie.mp4"     # keep an earlier good stitch if any
    assembly["joints"] = joints
    return assembly


def _write_movie_json(movie_root: str, spec: StudioMovieSpec, seg_records: "list[dict]",
                      assembly: "dict", job_id: str, partial: bool) -> "dict":
    """(Over)write ``<movie_root>/movie.json`` — the full node list + per-joint
    ``{branch_frame, trim_frames}`` + assembly + drift note. Returns the manifest
    dict (for ``JobResult.movie``). Best-effort — never raises across the job
    boundary."""
    manifest = {
        "kind": "studio_movie",
        "drift": _DRIFT_NOTE,
        "fps": spec.fps,
        "width": spec.width,
        "height": spec.height,
        "vram_budget_gb": spec.vram_budget_gb,
        "segments": seg_records,
        "joints": assembly.get("joints", []),
        "assembly": {
            "movie": assembly.get("movie"),
            "total_frames": assembly.get("total_frames", 0),
        },
        "partial": partial,
        "segments_completed": len([r for r in seg_records if r["status"] in ("done", "resumed")]),
        "segments_total": len(spec.goals),
    }
    try:
        with open(os.path.join(movie_root, "movie.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
    except Exception as exc:  # best-effort — never raise across the job boundary
        logger.warning("studio movie %s: movie.json write FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)
    return manifest


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #
def run_generate_studio_movie(spec: StudioMovieSpec, job_id: str) -> JobResult:
    started_at = time.time()
    from ..media_bus import is_cancelling, set_progress

    seg_total = len(spec.goals)
    projectmeta = slugify(spec.project) if spec.project else job_id
    movie_root = os.path.join(
        os.path.abspath(spec.out_root) if spec.out_root else STUDIO_MOVIE_ROOT,
        projectmeta)
    work_dir = os.path.join(movie_root, "_assembly")
    os.makedirs(movie_root, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    logger.info("studio movie %s: %d segment(s), %dx%d @ %dfps, tier vram<=%.2fGB",
                job_id, seg_total, spec.width, spec.height, spec.fps, spec.vram_budget_gb)

    seg_records: "list[dict]" = []   # movie.json / JobResult.movie segment nodes
    seg_refs: "list" = []            # per-segment clip MediaRefs, in order
    prev_clip_path: "str | None" = None   # the parent clip the next segment branches from
    prev_frames: int = 0                   # the parent clip's frame count

    # live per-segment meta (for the nested progress blob)
    segments_meta = [
        {"index": i, "segment_id": g.segment_id, "prompt": g.prompt,
         "status": "pending", "resumed": None}
        for i, g in enumerate(spec.goals)
    ]

    def _emit(stage: str, current: "dict | None" = None) -> None:
        """Build + persist the NESTED movie progress blob (best-effort)."""
        seg_done = sum(1 for s in segments_meta if s["status"] in ("done", "resumed"))
        elapsed = time.time() - started_at
        eta = round((elapsed / seg_done) * (seg_total - seg_done), 2) if seg_done > 0 else None
        blob = {
            "stage": stage, "segment_done": seg_done, "segment_total": seg_total,
            "segments": segments_meta, "current": current,
            "started_at": started_at, "eta_s": eta,
        }
        try:
            set_progress(job_id, blob)
        except Exception:
            logger.debug("studio movie %s: set_progress failed (non-fatal)", job_id, exc_info=True)

    def _save(partial: bool) -> "dict":
        """Best-effort iterative finalize: (re)stitch completed segments + rewrite
        movie.json, so an unfinishable movie still leaves a watchable partial +
        fresh manifest on disk."""
        assembly = _assemble_movie(movie_root, work_dir, seg_records, spec.fps, job_id)
        return _write_movie_json(movie_root, spec, seg_records, assembly, job_id, partial)

    def _partial_return(result: JobResult) -> JobResult:
        """Iterative partial save, THEN attach project + the partial manifest to a
        failure/cancel JobResult so the operator learns WHERE the partial landed.
        The ORIGINAL error is preserved verbatim (additive only)."""
        manifest = _save(partial=True)
        return replace(
            result,
            project={"name": spec.project, "uuid": job_id, "dir": movie_root},
            movie=manifest)

    should_cancel = lambda: is_cancelling(job_id)  # noqa: E731

    _emit("loading", None)

    for seg_i, goal in enumerate(spec.goals):
        # ---- between-segment cancel + time-budget checks (the bus won't) ----
        if is_cancelling(job_id):
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code="cancelled",
                message=f"cancelled after {seg_i} of {seg_total} segment(s)",
                retryable=False)))
        if spec.time_budget_s is not None and (time.time() - started_at) > spec.time_budget_s:
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code="time_budget_exceeded",
                message=(f"studio movie time budget {spec.time_budget_s}s exceeded after "
                         f"{seg_i} of {seg_total} segment(s)"),
                retryable=True)))

        seg_out_root = os.path.join(movie_root, f"segment_{seg_i:02d}")

        # ---- decide this segment's conditioning still + capability ----
        # segment 0: i2v from the movie start_image if given, else t2v. Later
        # segments: i2v from the parent clip's BRANCH FRAME (null -> last frame).
        resolved_branch = None
        if seg_i == 0:
            start_image = spec.start_image.uri if spec.start_image is not None else None
            capability = "i2v" if start_image else "t2v"
        else:
            # branch_frame null -> the parent's LAST frame (prev_frames - 1).
            raw = goal.branch_frame
            resolved_branch = (prev_frames - 1) if raw is None else int(raw)
            # RUN-time bound check (the schema can't know the parent's real length):
            # a branch past the parent's frames is errors-as-data, never a crash.
            if resolved_branch < 0 or resolved_branch >= prev_frames:
                return _partial_return(JobResult(job_id, ok=False, error=JobError(
                    code="branch_frame_out_of_range",
                    message=(f"segment {seg_i} ({goal.segment_id!r}) branch_frame "
                             f"{raw!r} -> resolved {resolved_branch} is outside the "
                             f"parent clip's [0, {prev_frames}) frames"),
                    retryable=False)))
            branch_png = os.path.join(seg_out_root, "branch.png")
            _emit("branching", {"segment_id": goal.segment_id, "branch_frame": resolved_branch})
            ok, tail = _extract_frame_at(prev_clip_path, resolved_branch, branch_png)
            if not ok:
                return _partial_return(JobResult(job_id, ok=False, error=JobError(
                    code="branch_frame_extract_failed",
                    message=(f"segment {seg_i} ({goal.segment_id!r}): could not extract "
                             f"branch frame {resolved_branch} from the parent clip: {tail}"),
                    retryable=False)))
            start_image = branch_png
            capability = "i2v"

        # ---- deterministic per-segment seed (node override wins) ----
        seg_seed = goal.seed if goal.seed is not None else (spec.seed + seg_i)

        # ---- build the per-segment studio spec + render through the SAME spine ----
        # (validate-at-construction; a bad geometry/override raises LOCALLY here, which
        # is a programmer error since the movie spec was already validated — geometry
        # is movie-level and in range.)
        seg_spec = make_studio_i2v(
            capability=capability,
            width=spec.width, height=spec.height, fps=spec.fps,
            vram_budget_gb=spec.vram_budget_gb,
            seed=seg_seed,
            out_root=seg_out_root,
            start_image=start_image,
            negative=(goal.negative if goal.negative is not None else spec.negative),
            prompt=goal.prompt,
            project=spec.project,
            steps=(goal.steps if goal.steps is not None else spec.steps),
            cfg=(goal.cfg if goal.cfg is not None else spec.cfg),
            model_id=(goal.model_id if goal.model_id is not None else spec.model_id),
        )

        segments_meta[seg_i].update(status="generating")
        _emit("generating", {"segment_id": goal.segment_id, "index": seg_i,
                             "prompt": goal.prompt, "capability": capability})

        result = run_produce_clip(seg_spec, should_cancel)
        if result.is_err():
            # An expected per-segment failure (unroutable, mid-render CANCELLED, IO)
            # fails the whole movie — DATA, never a raise. Save the completed segments
            # so far + carry the partial pointer (original error preserved).
            segments_meta[seg_i].update(status="failed")
            return _partial_return(JobResult(
                job_id, ok=False, error=_stage_error_to_job_error(result.error)))

        art = result.unwrap()
        # Catalog the WHOLE clip (never trimmed) as a video MediaRef on outputs.
        ref = ingest(art.path, kind_hint="video")
        seg_refs.append(ref)

        seg_records.append({
            "index": seg_i,
            "segment_id": goal.segment_id,
            "parent_segment_id": goal.parent_segment_id,
            "prompt": goal.prompt,
            "capability": capability,
            "seed": seg_seed,
            "branch_frame": goal.branch_frame,          # the AUTHORED value (may be null)
            "resolved_branch": resolved_branch,          # frame index into the PARENT (None for root)
            "clip_path": art.path,
            "clip_uri": ref.uri,
            "frames": art.frames,
            "width": art.width,
            "height": art.height,
            "duration_s": art.duration_s,
            "content_hash": art.content_hash,
            "resumed": art.resumed,
            "status": "resumed" if art.resumed else "done",
        })
        segments_meta[seg_i].update(status=("resumed" if art.resumed else "done"),
                                    resumed=art.resumed)

        prev_clip_path = art.path
        prev_frames = art.frames
        logger.info("studio movie %s: segment %d (%s) %s — %d frames @ %s",
                    job_id, seg_i, goal.segment_id,
                    "RESUMED" if art.resumed else "rendered", art.frames, art.path)

        _emit("generating", None)
        # ITERATIVE save after every segment (watchable partial + fresh manifest).
        _save(partial=True)

    # ---- FINAL assembly + manifest ----
    _emit("assembling", None)
    movie_manifest = _save(partial=False)

    # Ingest the stitched movie.mp4 LAST so outputs[-1] is the assembled video.
    all_refs = list(seg_refs)
    if movie_manifest.get("assembly", {}).get("movie"):
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        if os.path.isfile(movie_mp4):
            all_refs.append(ingest(movie_mp4, kind_hint="video"))

    _emit("archiving", None)
    return JobResult(
        job_id, ok=True, outputs=tuple(all_refs),
        project={"name": spec.project, "uuid": job_id, "dir": movie_root},
        movie=movie_manifest)
