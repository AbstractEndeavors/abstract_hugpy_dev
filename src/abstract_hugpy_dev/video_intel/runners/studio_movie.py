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
     Each LATER segment is spliced onto its parent per its ``joint_mode``:
       * "still" (default, backward-compatible): render i2v, conditioned on ONE still
         — the **branch frame** of the PREVIOUS segment's clip (``branch_frame:
         int | null``; null ⇒ the parent's LAST frame). The still is extracted with
         ffmpeg (a frame-accurate ``select=eq(n\\,B)`` pluck) and handed to
         ``produce_clip`` as the i2v ``start_image``. Motion is NOT carried.
       * "vace_extend": carry MOTION across the splice — extract the parent's TRAILING
         ``context_frames`` frames (``[branch-K+1 .. branch]``, clamped) and route the
         render through the VACE path (capability "v2v" -> Task.VACE_CONTROL) with those
         frames as the temporal conditioning. The VACE runner builds the diffusers
         video+mask extend idiom (kept context prefix + generated tail), so the segment
         CONTINUES the parent's motion instead of restarting from one frame. The child's
         first K output frames RECONSTRUCT the context and are DROPPED at assembly (see
         ASSEMBLY below) so no frame double-plays. A vace_extend segment routes to a real
         VACE model, so its per-segment vram budget is raised to the VACE floor
         (``_VACE_MIN_BUDGET_GB``); on a GPU-less box it returns the VACE runner's
         graceful Err (NO_GPU/DEPS_MISSING/WEIGHTS_MISSING) — an HONEST per-segment error,
         NEVER a silent fallback to still-mode.
       * "cut": a HARD SCENE CUT — NO frame carry at all. No branch still / context window
         is extracted; the child is a FRESH render of its own prompt. The parent is NOT
         trimmed (it plays in FULL) and assembly records the joint as ``{mode:"cut"}``.

     IDENTITY MOVIE (movie-level ``reference_images`` set) — the operator's "take that id
     and use it for a video: her on the beach, then playing volleyball". When the movie
     carries reference image(s), EVERY segment (segment 0 included) renders capability
     ``id_lock`` (Wan-VACE reference-to-video) with those references, so the locked SUBJECT
     carries across every scene change. The per-segment budget is raised to the shared
     ``_VACE_MIN_BUDGET_GB`` floor (id_lock routes through the VACE path, exactly like
     vace_extend). The joint behavior is UNCHANGED per mode — a "still" joint still extracts
     the branch frame and trims the parent, a "cut" joint still carries no frame — but the
     RENDER of each segment is now reference-conditioned. On the VACE path the i2v
     ``start_image`` (a "still" joint's branch frame) is ACCEPTED but UNUSED (the runner
     conditions on the references, not a single still) — so for an identity movie the
     REFERENCES win the conditioning; the branch frame governs only the parent TRIM at
     assembly. There is no synthetic id_lock tier, so an identity movie on a GPU-less box
     surfaces the VACE runner's graceful per-segment Err (see ``_VACE_MIN_BUDGET_GB``).
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

VACE-EXTEND OVERLAP (assembly). The parent-trim math above is UNCHANGED: the parent
still plays ``[0 .. branch]`` and the child still "starts at the branch frame". But a
``vace_extend`` child's output INCLUDES its first K frames as a RECONSTRUCTION of the
parent's trailing context (the kept mask=0 prefix). Those K frames overlap the parent's
tail, so they are DROPPED from the CHILD's head at concat (``context_drop = K`` frames):
the child contributes ``[K .. end]``, its first NEWLY-generated frame (index K) splices
directly onto the parent's branch frame — no frame double-plays. (A "still" child has
``context_drop = 0`` — nothing dropped, today's behavior byte-identical.)

``movie.json`` sidecar records the full node list + per-joint
``{branch_frame, trim_frames, mode, context_frames}`` (``mode`` labels each splice
"still" vs "vace_extend" so the UI can show it honestly) + a DRIFT note.

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

_DRIFT_NOTE = ("still-mode splices condition each segment on ONE frame of its parent — "
               "motion is NOT carried across the splice. A joint's "
               "joint_mode='vace_extend' carries motion via VACE-extend (conditioning on "
               "the parent's trailing context_frames through the diffusers video+mask "
               "extend idiom) instead of a single still. A joint_mode='cut' is a HARD "
               "scene cut: no frame carry, the parent plays in full. With movie-level "
               "reference_images set (an identity movie) every segment renders id_lock so "
               "the subject carries across scene changes even though no pixels do — on the "
               "VACE path the references win the conditioning (an i2v branch still is "
               "accepted but unused; it governs only the parent trim).")

# The SHARED Wan-VACE budget floor. Two segment kinds route through the VACE path
# (Task.VACE_CONTROL), which is served ONLY by real Wan-VACE models — the cheapest,
# wan2.1-vace-1.3b, needs ~6GB (INT8 @ <=480p):
#   * a ``vace_extend`` splice (motion-carry across a join), and
#   * EVERY segment of an IDENTITY MOVIE (movie-level ``reference_images`` -> capability
#     ``id_lock`` -> Task.VACE_CONTROL reference-to-video).
# A plain still/i2v/t2v segment stays on the movie's (often tiny/synthetic) budget, so a
# VACE-bound segment RAISES its own per-segment budget to this floor to actually REACH the
# VACE model — never a silent downgrade (that dishonesty is banned). On a GPU-less box the
# render then returns the VACE runner's graceful DEPS_MISSING/NO_GPU/WEIGHTS_MISSING (an
# honest per-segment Err), not a synthetic clip — IDENTICAL to the single-clip id_lock
# path. There is NO synthetic id_lock/VACE tier BY DESIGN (no synthetic model declares
# id_lock), so an identity movie is a real-box render; the honest GPU-less result is a
# graceful Err. NOTE: vace-1.3b tops out at 480p, so a movie wider/taller than 832x480
# surfaces an honest VRAM_EXCEEDED (a bigger VACE model needs a bigger movie budget).
_VACE_MIN_BUDGET_GB = 6.0


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


def _extract_context_frames(
    clip_path: str, branch_index: int, k: int, dest_dir: str
) -> "tuple[list[str] | None, str, list[int]]":
    """Pluck the parent clip's TRAILING context window for a ``vace_extend`` splice:
    frames ``[branch_index-k+1 .. branch_index]`` (inclusive, oldest -> newest), CLAMPED
    at 0. Writes them as ordered PNGs (``ctx_000.png`` = oldest) into ``dest_dir`` via the
    same frame-accurate ``select=eq(n,IDX)`` pluck the branch still uses.

    The number extracted is ``min(k, branch_index + 1)`` — a branch too early to have k
    frames behind it yields the fewer frames actually available (never a crash). Returns
    ``(paths | None, stderr_tail, indices)``; None signals an errors-as-data ffmpeg
    failure. ``indices`` is the exact 0-based source frame indices extracted (in order),
    so the caller/manifest records precisely which parent frames carried the motion."""
    start = max(0, int(branch_index) - int(k) + 1)
    indices = list(range(start, int(branch_index) + 1))   # inclusive, oldest -> newest
    os.makedirs(dest_dir, exist_ok=True)
    paths: "list[str]" = []
    for pos, idx in enumerate(indices):
        dest = os.path.join(dest_dir, f"ctx_{pos:03d}.png")
        ok, tail = _extract_frame_at(clip_path, idx, dest)
        if not ok:
            return None, tail, indices
        paths.append(dest)
    return paths, "", indices


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


def _slice_clip(src_mp4: str, dst_mp4: str, start_frame: int, n_frames: int,
                fps: int) -> "tuple[bool, str]":
    """Re-encode a WINDOW ``[start_frame, start_frame+n_frames)`` of ``src_mp4`` into
    ``dst_mp4`` — the vace_extend head-drop path (drop a child's reconstructed context
    prefix). Frame-accurate via a ``select='gte(n,START)'`` filter + ``setpts`` reset,
    then ``-frames:v n`` on the kept stream. Same house H.264/yuv420p invocation as
    ``_trim_clip`` so every contribution shares codec params (concat ``-c copy`` stays
    valid). For ``start_frame <= 0`` it is exactly ``_trim_clip`` (first-n), so the
    still-mode path never changes. Never raises on a plain ffmpeg failure."""
    if int(start_frame) <= 0:
        return _trim_clip(src_mp4, dst_mp4, n_frames, fps)
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    s = int(start_frame)
    # select drops the leading window, setpts rebases timestamps to 0; -frames:v then
    # counts the KEPT frames. NOTE: no ``-vsync 0`` here — it conflicts with the CFR
    # ``-r`` on this ffmpeg (6.1) and aborts ("Invalid argument"); the default vsync +
    # ``-r`` gives a clean CFR clip whose codec params match _trim_clip's (concat-safe).
    vf = (f"select='gte(n\\,{s})',setpts=PTS-STARTPTS,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd = [
        ffmpeg, "-y",
        "-i", src_mp4,
        "-vf", vf,
        "-frames:v", str(int(n_frames)),
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

    # Per-joint trim record + each segment's contribution WINDOW [head_drop, tail_end).
    #   * tail_end (the PARENT trim): a segment with a NEXT completed segment (its child)
    #     plays to child.resolved_branch + 1 for a still/vace_extend child; a "cut" child
    #     carries NO frame, so its parent plays in FULL (tail_end = the parent's frames — no
    #     trim). The last (leaf) segment plays FULL.
    #   * head_drop (vace_extend only): a segment RENDERED via vace_extend has its first
    #     ``context_drop`` frames as a RECONSTRUCTION of its parent's context (the kept
    #     mask=0 prefix), which overlaps the parent's tail — so drop them from this segment's
    #     head (context_drop=0 for a still/cut segment, so it plays [0, tail_end) exactly as
    #     before). The joint records the CHILD's splice ``mode`` so the UI can label it
    #     "still" / "vace_extend" / "cut" honestly.
    # (segment clip path, head_drop, n_frames)
    contributions: "list[tuple[str, int, int]]" = []
    joints: "list[dict]" = []
    for p, rec in enumerate(completed):
        head_drop = int(rec.get("context_drop") or 0)   # vace_extend reconstructed prefix
        if p + 1 < len(completed):
            child = completed[p + 1]
            child_mode = child.get("joint_mode", "still")
            if child_mode == "cut":
                # SCENE CUT: no frame carry -> the parent plays in FULL (no trim). The joint
                # records mode="cut" with branch_frame=None (a cut conditions on no frame) and
                # trim_frames = the parent's full length (the spliced-row math reads a joint
                # whose trim == full as an untrimmed block; see movieTimeline.deriveRow).
                tail_end = int(rec["frames"])
                joints.append({
                    "parent_segment_id": rec["segment_id"],
                    "child_segment_id": child["segment_id"],
                    "branch_frame": None,
                    "trim_frames": tail_end,
                    "mode": "cut",
                    "context_frames": None,
                })
            else:
                rb = child["resolved_branch"]          # branch INTO this segment
                tail_end = int(rb) + 1
                joints.append({
                    "parent_segment_id": rec["segment_id"],
                    "child_segment_id": child["segment_id"],
                    "branch_frame": int(rb),
                    "trim_frames": tail_end,
                    # SPLICE MODE (child's): "still" or "vace_extend"; context_frames is the
                    # child's kept-context length (None for a still splice). The UI labels the
                    # splice from these — motion-carry is never silent.
                    "mode": child_mode,
                    "context_frames": (child.get("context_frames")
                                       if child_mode == "vace_extend" else None),
                })
        else:
            tail_end = int(rec["frames"])          # leaf: full clip
        contributions.append((rec["clip_path"], head_drop, tail_end - head_drop))

    # Materialize each contribution as a uniformly re-encoded WINDOW in the work dir,
    # then concat. Non-destructive: the source clips are never touched. A head_drop>0
    # (vace_extend) window goes through _slice_clip; head_drop==0 is the historical
    # first-n _trim_clip (byte-identical still-mode path).
    os.makedirs(work_dir, exist_ok=True)
    contrib_paths: "list[str]" = []
    total = 0
    try:
        for i, (src, head_drop, n) in enumerate(contributions):
            dst = os.path.join(work_dir, f"contrib_{i:02d}.mp4")
            ok, tail = _slice_clip(src, dst, head_drop, n, fps)
            if not ok:
                logger.warning("studio movie %s: contribution slice %d FAILED "
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
    ``{branch_frame, trim_frames, mode, context_frames}`` (``mode`` labels each splice
    "still" vs "vace_extend"; ``context_frames`` is the vace kept-context length) +
    assembly + drift note. Returns the manifest dict (for ``JobResult.movie``).
    Best-effort — never raises across the job boundary."""
    manifest = {
        "kind": "studio_movie",
        "drift": _DRIFT_NOTE,
        "fps": spec.fps,
        "width": spec.width,
        "height": spec.height,
        "vram_budget_gb": spec.vram_budget_gb,
        # IDENTITY LOCK: the movie-level subject references (empty for a plain movie). When
        # non-empty every segment rendered capability id_lock (see each segment's capability).
        "id_lock": bool(spec.reference_images),
        "reference_images": list(spec.reference_images),
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

        # ---- decide this segment's conditioning + capability + joint mode ----
        # IDENTITY MOVIE (movie-level reference_images set): EVERY segment renders capability
        # id_lock (Wan-VACE reference-to-video) so the locked SUBJECT carries across scene
        # changes; the per-segment budget is raised to the shared VACE floor (id_lock routes
        # through the VACE path, exactly like vace_extend). The per-mode JOINT behavior is
        # UNCHANGED (a still joint still extracts + trims, a cut carries no frame) — only the
        # RENDER is now reference-conditioned. On the VACE path an i2v start_image is ACCEPTED
        # but UNUSED (the runner conditions on the references), so for an id-movie the
        # REFERENCES win; a still joint's branch frame governs only the parent TRIM at assembly.
        # Otherwise (PLAIN movie): segment 0 is i2v (movie start_image) else t2v; a later
        # segment splices onto its parent per goal.joint_mode:
        #   * "still": i2v conditioned on ONE branch frame (start_image). No motion carry.
        #   * "vace_extend": v2v (VACE) conditioned on the parent's TRAILING context frames.
        #   * "cut": a HARD scene cut — no frame carry; a FRESH render, parent plays in FULL.
        id_refs = tuple(spec.reference_images or ())
        is_id_movie = bool(id_refs)
        resolved_branch = None
        start_image = None
        vace_context_frames = None
        seg_joint_mode = "still"
        seg_context_frames = 0          # how many parent frames carried the motion (K)
        seg_context_drop = 0            # frames DROPPED from THIS segment's head at assembly
        # An id-movie segment (id_lock) routes through VACE -> raise to the shared VACE floor.
        seg_budget = (max(spec.vram_budget_gb, _VACE_MIN_BUDGET_GB)
                      if is_id_movie else spec.vram_budget_gb)
        if seg_i == 0:
            # Root: id_lock in an id-movie (the references define the render); else i2v from
            # the movie start_image, else t2v. In an id-movie the movie start_image is
            # ACCEPTED but the VACE runner ignores it (references win) — carried for provenance.
            start_image = spec.start_image.uri if spec.start_image is not None else None
            capability = "id_lock" if is_id_movie else ("i2v" if start_image else "t2v")
        elif goal.joint_mode == "cut":
            # SCENE CUT: no frame carry at all — no branch resolve / extraction. The child is a
            # FRESH render (id_lock in an id-movie so the subject carries, else t2v). The parent
            # plays in FULL: resolved_branch stays None so assembly does NOT trim it.
            seg_joint_mode = "cut"
            _emit("branching", {"segment_id": goal.segment_id, "mode": "cut"})
            capability = "id_lock" if is_id_movie else "t2v"
        else:
            # still / vace_extend splice onto the parent at a branch frame.
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
            seg_joint_mode = goal.joint_mode
            if seg_joint_mode == "vace_extend":
                # VACE-EXTEND: extract the parent's trailing K frames [branch-K+1 .. branch]
                # (clamped) and route the segment through the VACE path (capability "v2v" ->
                # Task.VACE_CONTROL). The child's first K output frames RECONSTRUCT this
                # context and are dropped at assembly (context_drop) so no frame double-plays.
                # RAISE the per-segment budget to the VACE floor so a real VACE model
                # actually binds (a still segment stays on the movie's tiny/synthetic
                # budget) — this is REQUIRED to reach the VACE path, NOT a silent downgrade.
                # In an id-movie the references ALSO ride along (identity + motion-carry both).
                k = goal.context_frames if goal.context_frames is not None else spec.context_frames
                ctx_dir = os.path.join(seg_out_root, "context")
                _emit("branching", {"segment_id": goal.segment_id, "branch_frame": resolved_branch,
                                    "mode": "vace_extend", "context_frames": k})
                paths, tail, idxs = _extract_context_frames(
                    prev_clip_path, resolved_branch, k, ctx_dir)
                if paths is None:
                    return _partial_return(JobResult(job_id, ok=False, error=JobError(
                        code="context_frame_extract_failed",
                        message=(f"segment {seg_i} ({goal.segment_id!r}, joint_mode=vace_extend): "
                                 f"could not extract context frames {idxs} from the parent "
                                 f"clip: {tail}"),
                        retryable=False)))
                vace_context_frames = tuple(paths)
                seg_context_frames = len(paths)     # actual K extracted = min(k, branch+1)
                seg_context_drop = len(paths)        # the child reconstructs these -> drop at assembly
                seg_budget = max(spec.vram_budget_gb, _VACE_MIN_BUDGET_GB)
                capability = "v2v"
            else:
                # STILL (default, backward-compatible): condition on ONE branch frame. In an
                # id-movie the render is id_lock (the references) + this branch still, which the
                # VACE runner ACCEPTS but IGNORES (references win) — the still governs only the
                # parent TRIM at assembly. In a plain movie it is the historical i2v.
                branch_png = os.path.join(seg_out_root, "branch.png")
                _emit("branching", {"segment_id": goal.segment_id,
                                    "branch_frame": resolved_branch, "mode": "still"})
                ok, tail = _extract_frame_at(prev_clip_path, resolved_branch, branch_png)
                if not ok:
                    return _partial_return(JobResult(job_id, ok=False, error=JobError(
                        code="branch_frame_extract_failed",
                        message=(f"segment {seg_i} ({goal.segment_id!r}): could not extract "
                                 f"branch frame {resolved_branch} from the parent clip: {tail}"),
                        retryable=False)))
                start_image = branch_png
                capability = "id_lock" if is_id_movie else "i2v"

        # ---- deterministic per-segment seed (node override wins) ----
        seg_seed = goal.seed if goal.seed is not None else (spec.seed + seg_i)

        # ---- build the per-segment studio spec + render through the SAME spine ----
        # (validate-at-construction; a bad geometry/override raises LOCALLY here, which
        # is a programmer error since the movie spec was already validated — geometry
        # is movie-level and in range.)
        seg_spec = make_studio_i2v(
            capability=capability,
            width=spec.width, height=spec.height, fps=spec.fps,
            vram_budget_gb=seg_budget,   # bumped to the VACE floor for a vace_extend joint
            seed=seg_seed,
            out_root=seg_out_root,
            start_image=start_image,
            negative=(goal.negative if goal.negative is not None else spec.negative),
            prompt=goal.prompt,
            project=spec.project,
            steps=(goal.steps if goal.steps is not None else spec.steps),
            cfg=(goal.cfg if goal.cfg is not None else spec.cfg),
            model_id=(goal.model_id if goal.model_id is not None else spec.model_id),
            # VACE-EXTEND temporal conditioning (None for a still/i2v/t2v segment).
            vace_context_frames=vace_context_frames,
            # IDENTITY LOCK: the movie-level subject references, passed on EVERY segment of an
            # id-movie (capability id_lock) so the locked subject carries across scene changes.
            # None for a plain movie.
            reference_images=(id_refs if is_id_movie else None),
        )

        segments_meta[seg_i].update(status="generating")
        _emit("generating", {"segment_id": goal.segment_id, "index": seg_i,
                             "prompt": goal.prompt, "capability": capability})

        result = run_produce_clip(seg_spec, should_cancel)
        if result.is_err():
            # An expected per-segment failure (unroutable, mid-render CANCELLED, IO, or a
            # vace_extend segment's graceful NO_GPU/DEPS_MISSING/WEIGHTS_MISSING on a
            # GPU-less box) fails the whole movie — DATA, never a raise, NEVER a silent
            # fallback to still-mode. The JobError is ENRICHED with WHICH segment + joint
            # mode failed, and the failed node is recorded in movie.json (status="failed").
            segments_meta[seg_i].update(status="failed")
            je = _stage_error_to_job_error(result.error)
            seg_records.append({
                "index": seg_i,
                "segment_id": goal.segment_id,
                "parent_segment_id": goal.parent_segment_id,
                "prompt": goal.prompt,
                "capability": capability,
                "joint_mode": seg_joint_mode,
                "context_frames": seg_context_frames,
                "resolved_branch": resolved_branch,
                "vram_budget_gb": seg_budget,
                "status": "failed",
                "error": {"code": je.code, "message": je.message},
            })
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code=je.code,
                message=(f"segment {seg_i} ({goal.segment_id!r}, joint_mode={seg_joint_mode}, "
                         f"capability={capability}): {je.message}"),
                retryable=je.retryable)))

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
            # JOINT MODE honesty: how this segment was spliced onto its parent + the
            # motion-carry conditioning. context_frames = parent frames KEPT as the VACE
            # extend prefix (0 for still); context_drop = frames the assembler drops from
            # THIS segment's head so the reconstructed context never double-plays (0 for
            # still). vram_budget_gb = the EFFECTIVE per-segment budget (bumped for vace).
            "joint_mode": seg_joint_mode,
            "context_frames": seg_context_frames,
            "context_drop": seg_context_drop,
            "vram_budget_gb": seg_budget,
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
