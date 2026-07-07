"""Pure `(diffusers, generate_movie)` runner — a GOAL TIMELINE -> a stitched movie.

`run_generate_movie(spec, job_id) -> JobResult`. A FAT orchestrator that sequences
the movie's segments INLINE (NOT an orchestrator-of-child-jobs, which would
deadlock the single-daemon bus — it follows run_generate_scene's inline pattern).

Per segment (goal), in timeline order:
  a. Render the segment's frames (n = end-start) via the EXTRACTED scene core
     `runners.scene.render_scene_frames`, base prompt = goal.prompt. Cross-segment
     DRIFT: the next segment's start image = the previous segment's LAST frame
     (carried), so goals transition smoothly. Feasible only when the fleet can
     serve image-to-image for this model; when it CANNOT, the movie falls back to
     INDEPENDENT segments (no carry) + concat and SAYS SO (movie.json `drift`).
  b. Capture the segment's frames (MediaRefs) — the caller owns the refs list.
  c. If `spec.vision_enabled`, score the KEY frame (the last) via the vision plane
     (execute_prompt task=image-text-to-text) and parse VERDICT/SCORE/WHY.
  d. Retry: vision on AND score < threshold AND attempt < max -> re-render with a
     bumped, DETERMINISTIC seed (base + segment_index*1000 + attempt); else keep
     the best-scoring take and proceed.
  e. Write the segment bundle INCREMENTALLY under assets/<projectmeta>/segment_NN/
     (reuse scene._write_bundle). RESUME: a segment whose bundle already exists is
     SKIPPED (a re-enqueue continues where it left off).
  f. Emit NESTED movie progress via media_bus.set_progress.
  g. Between segments: honor is_cancelling(job_id) AND spec.time_budget_s (the
     runner owns its OWN wall-clock — the bus has no timeout/reaper).
  h. After all segments: ffmpeg CONCAT-DEMUX the segment mp4s -> movie.mp4 (under
     scene._SCENE_SEM) and write movie.json.
  i. Return JobResult(outputs=[all frames + movie.mp4], project=..., movie=...).

Pure discipline (map §6): EXPECTED failures are returned as JobResult(ok=False,
JobError(...)) — DATA, never a raise. The inference-plane import stays LAZY.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, replace

from abstract_hugpy_dev._platform.binaries import resolve_bin
from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT
from abstract_hugpy_dev.imports.src.utils import slugify

from ..gen_schema import text_part
from ..media_store import ingest
from ..movie_schema import MovieSpec, total_frames
from ..result_schema import JobError, JobResult
from ..scene_schema import make_generate_scene
from ._img2img import img2img_available
from .scene import (
    _SCENE_SEM,
    _assemble_scene_mp4,
    _write_bundle,
    render_scene_frames,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# vision verdict parsing — a small, isolated, unit-testable helper
# --------------------------------------------------------------------------- #
def parse_vision_verdict(text: str) -> dict:
    """Parse the judge's reply into ``{"verdict","score","why"}``.

    The judge is asked to reply exactly ``VERDICT=YES|NO; SCORE=0-100;
    WHY=<one sentence>`` but real models drift, so this is TOLERANT:
      * VERDICT: the ``VERDICT=YES|NO`` field, else a bare YES/NO word; upper-cased
        ("YES"/"NO") or None if neither is present.
      * SCORE: the ``SCORE=<int>`` field (clamped to 0..100), else None.
      * WHY: the ``WHY=<...>`` tail (trailing period stripped), else "".

    Returns DATA only — never raises (a judge that produces garbage yields
    verdict=None/score=None, which the orchestrator treats as "unscored, keep")."""
    t = text or ""
    verdict = None
    m = re.search(r"VERDICT\s*[=:]\s*(YES|NO)", t, re.I)
    if m:
        verdict = m.group(1).upper()

    score = None
    m = re.search(r"SCORE\s*[=:]\s*(\d{1,3})", t, re.I)
    if m:
        score = max(0, min(100, int(m.group(1))))

    why = ""
    m = re.search(r"WHY\s*[=:]\s*(.+)", t, re.I | re.S)
    if m:
        why = m.group(1).strip().splitlines()[0].strip().rstrip(".").strip()

    if verdict is None:
        if re.search(r"\bYES\b", t, re.I):
            verdict = "YES"
        elif re.search(r"\bNO\b", t, re.I):
            verdict = "NO"

    return {"verdict": verdict, "score": score, "why": why}


def _vision_text(res) -> str:
    """Best-effort extract the reply text from an execute_prompt result object."""
    txt = getattr(res, "text", None)
    if txt:
        return txt
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(res, attr, None)
        if callable(fn):
            try:
                d = fn()
            except TypeError:
                continue
            if isinstance(d, dict) and d.get("text"):
                return d["text"]
    return str(res)


def _score_keyframe(goal: str, keyframe_uri: str, judge_model_id) -> dict:
    """Score one KEY frame against ``goal`` via the vision plane. Best-effort:
    a plane raise / not-ok result degrades to an UNSCORED verdict (score=None) so
    the movie still ships (the segment's take is simply kept). Returns the
    parse_vision_verdict dict plus a ``raw`` field carrying the model's reply."""
    prompt = (
        f"GOAL: {goal}.\n"
        "Does the image achieve this goal? Reply exactly: "
        "VERDICT=YES|NO; SCORE=0-100; WHY=<one sentence>."
    )
    kwargs = dict(
        task="image-text-to-text",
        file=keyframe_uri,
        prompt=prompt,
        max_new_tokens=80,
    )
    if judge_model_id:
        kwargs["model_key"] = judge_model_id
    try:
        from abstract_hugpy_dev.managers.dispatch import execute_prompt
        from abstract_hugpy_dev._platform.async_runtime import run
        res = run(execute_prompt(**kwargs))
    except Exception as exc:  # plane raised -> unscored (keep the take)
        logger.info("movie vision judge raised (%s: %s); leaving segment unscored",
                    type(exc).__name__, exc)
        return {"verdict": None, "score": None, "why": f"judge unavailable: {exc}", "raw": ""}
    if not getattr(res, "ok", True):
        err = getattr(res, "error", None)
        return {"verdict": None, "score": None,
                "why": f"judge not-ok: {err}", "raw": ""}
    raw = _vision_text(res)
    verdict = parse_vision_verdict(raw)
    verdict["raw"] = raw
    return verdict


# --------------------------------------------------------------------------- #
# ffmpeg concat-demux stitch (segment mp4s -> one movie.mp4)
# --------------------------------------------------------------------------- #
def _concat_movie(segment_mp4s: "list[str]", movie_mp4: str, work_dir: str) -> None:
    """Stitch the segment mp4s into a single movie.mp4 via ffmpeg's concat DEMUXER
    (stream-copy — the segments are all encoded identically by
    _assemble_scene_mp4, so ``-c copy`` is valid and fast). Serialized behind
    scene._SCENE_SEM like the scene assembly. RAISES on failure — the caller wraps
    it into a movie_assembly_failed JobError (never a raise across the boundary)."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w") as fh:
        for p in segment_mp4s:
            # concat-demux single-quote escaping: ' -> '\''
            safe = p.replace("'", "'\\''")
            fh.write(f"file '{safe}'\n")
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        movie_mp4,
    ]
    with _SCENE_SEM:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(movie_mp4):
        raise RuntimeError(
            f"ffmpeg concat failed rc={result.returncode}: {(result.stderr or '')[-500:]}"
        )


# --------------------------------------------------------------------------- #
# ITERATIVE / RESILIENT finalize — always leaves a watchable partial movie +
# an up-to-date manifest on disk
# --------------------------------------------------------------------------- #
def _finalize_movie(assets_root: str, projectmeta: str, segment_mp4s: "list[str]",
                    segment_records: "list[dict]", segments_meta: "list[dict]",
                    spec: MovieSpec, job_id: str, n_frames_total: int,
                    drift_mode: str, started_at: float, partial: bool) -> dict:
    """(Re)stitch the segment mp4s rendered SO FAR into
    ``assets/<projectmeta>/movie.mp4`` and (over)write
    ``assets/<projectmeta>/movie.json`` from the CURRENT ``segment_records``.
    Returns the movie manifest dict (for ``JobResult.movie``).

    Called BOTH after every completed segment and before every early
    return with ``partial=True`` (the ITERATIVE, resilient save that keeps a
    watchable partial movie + a fresh manifest on disk even when the job is
    cancelled / times out / a segment fails), AND once at the very end with
    ``partial=False`` (the full movie).

    PURE BEST-EFFORT — NEVER raises: a stitch or a write hiccup is logged and
    swallowed so a partial save is ADDITIVE only, can never mask the caller's
    real error, and never fails an otherwise-successful movie across the job
    boundary."""
    # ---- (re)stitch whatever segment mp4s exist SO FAR -> movie.mp4 ----
    # 0 mp4s -> nothing to stitch; 1 -> copy it (concat-demux is a needless
    # re-mux of a single input); 2+ -> ffmpeg concat-demux. Wrapped so a stitch
    # failure never masks the caller's real error (nor a failing movie's error).
    movie_rel = None
    if spec.assemble:
        movie_mp4 = os.path.join(assets_root, "movie.mp4")
        existing = [p for p in segment_mp4s if os.path.isfile(p)]
        try:
            if len(existing) == 1:
                shutil.copyfile(existing[0], movie_mp4)
                movie_rel = "movie.mp4"
            elif len(existing) >= 2:
                work_dir = os.path.join(DEFAULT_ROOT, "video_intel", "movies", job_id)
                _concat_movie(existing, movie_mp4, work_dir)
                movie_rel = "movie.mp4"
            # 0 existing segment mp4s -> nothing to stitch (movie_rel stays None)
        except Exception as exc:  # a stitch failure must NEVER mask the real error
            logger.warning("movie %s: partial stitch FAILED (non-fatal): %s: %s",
                           job_id, type(exc).__name__, exc)
            # keep pointing at an earlier GOOD stitch if one is already on disk
            if os.path.isfile(movie_mp4):
                movie_rel = "movie.mp4"

    # ---- movie.json manifest (superset of the original keys + partial flags) ----
    movie_manifest = {
        "goals": [
            {"start_frame": g.start_frame, "end_frame": g.end_frame, "prompt": g.prompt}
            for g in spec.goals
        ],
        "drift": drift_mode,
        "vision_enabled": spec.vision_enabled,
        "score_threshold": spec.score_threshold,
        "n_frames_total": n_frames_total,
        "segments": segment_records,
        "movie": movie_rel,
        "partial": partial,
        "segments_completed": len(segment_records),
        "segments_total": len(spec.goals),
    }
    try:
        with open(os.path.join(assets_root, "movie.json"), "w") as fh:
            json.dump(movie_manifest, fh, indent=2)
    except Exception as exc:  # best-effort — never raise across the job boundary
        logger.warning("movie %s: movie.json write FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)
    return movie_manifest


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #
def run_generate_movie(spec: MovieSpec, job_id: str) -> JobResult:
    started_at = time.time()
    from ..media_bus import is_cancelling, set_progress

    seg_total = len(spec.goals)
    n_frames_total = total_frames(spec)
    projectmeta = slugify(spec.project) if spec.project else job_id
    assets_root = os.path.join(DEFAULT_ROOT, "assets", projectmeta)
    work_dir = os.path.join(DEFAULT_ROOT, "video_intel", "movies", job_id)
    os.makedirs(assets_root, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Cross-segment DRIFT feasibility, probed ONCE: carry the previous last frame
    # as the next segment's start image only when the fleet can serve img2img for
    # this model; else fall back to INDEPENDENT segments (v1 text-to-image) + concat.
    can_carry = img2img_available(spec.model_id)
    drift_mode = "carry" if can_carry else "independent (image-to-image unavailable on the fleet)"
    logger.info("movie %s: %d segment(s), %d total frame(s), drift=%s",
                job_id, seg_total, n_frames_total, drift_mode)

    # live per-segment meta (mutated in place; emitted in the nested progress blob)
    segments_meta = [
        {"index": i, "goal": g.prompt, "prompt": g.prompt, "attempt": 0,
         "score": None, "status": "pending", "frames": []}
        for i, g in enumerate(spec.goals)
    ]
    segment_records: "list[dict]" = []   # movie.json / JobResult.movie segments
    segment_mp4s: "list[str]" = []        # bundle video.mp4 paths for the final concat
    all_refs: "list" = []                 # every chosen-take frame ref, in order
    prev_last_frame: "str | None" = None  # carried start image for the next segment

    def _emit(stage: str, current: "dict | None" = None) -> None:
        """Build + persist the NESTED movie progress blob (best-effort)."""
        elapsed = time.time() - started_at
        seg_done = sum(1 for s in segments_meta if s["status"] in ("done", "skipped"))
        eta = round((elapsed / seg_done) * (seg_total - seg_done), 2) if seg_done > 0 else None
        blob = {
            "stage": stage,
            "segment_done": seg_done,
            "segment_total": seg_total,
            "segments": segments_meta,
            "current": current,
            "started_at": started_at,
            "eta_s": eta,
        }
        try:
            set_progress(job_id, blob)
        except Exception:
            logger.debug("movie %s: set_progress failed (non-fatal)", job_id, exc_info=True)

    def _save(partial: bool) -> dict:
        """Best-effort ITERATIVE finalize with the current segment state."""
        return _finalize_movie(
            assets_root, projectmeta, segment_mp4s, segment_records, segments_meta,
            spec, job_id, n_frames_total, drift_mode, started_at, partial=partial)

    def _partial_return(result: JobResult) -> JobResult:
        """Do the ITERATIVE partial save, THEN attach project + the partial movie
        manifest to a failure/cancel JobResult so the operator learns WHERE the
        partial output landed (these error returns carried no pointer before).
        The ORIGINAL error is preserved verbatim via dataclasses.replace —
        additive only, never changes the error semantics."""
        manifest = _save(partial=True)
        return replace(
            result,
            project={"name": spec.project, "uuid": job_id, "dir": f"assets/{projectmeta}"},
            movie=manifest,
        )

    _emit("loading", None)

    for seg_i, goal in enumerate(spec.goals):
        # ---- between-segment cancel + time-budget checks (the bus won't) ----
        # Each early return first does the best-effort ITERATIVE save, then
        # carries the project + partial-movie pointer so the operator learns
        # WHERE the partial output landed.
        if is_cancelling(job_id):
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code="cancelled",
                message=f"cancelled after {seg_i} of {seg_total} segment(s)",
                retryable=False)))
        if spec.time_budget_s is not None and (time.time() - started_at) > spec.time_budget_s:
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code="time_budget_exceeded",
                message=(f"movie time budget {spec.time_budget_s}s exceeded after "
                         f"{seg_i} of {seg_total} segment(s)"),
                retryable=True)))

        seg_name = f"segment_{seg_i:02d}"
        seg_bundle = os.path.join(assets_root, seg_name)
        seg_n = goal.end_frame - goal.start_frame

        # ---- RESUME: a completed segment bundle short-circuits the render ----
        resumed = _resume_segment(seg_bundle)
        if resumed is not None:
            frame_refs, mp4_path, seed_val = resumed
            all_refs.extend(frame_refs)
            if mp4_path:
                segment_mp4s.append(mp4_path)
            if frame_refs:
                prev_last_frame = frame_refs[-1].uri
            segments_meta[seg_i].update(
                status="skipped", score=None, attempt=0,
                frames=[asdict(r) for r in frame_refs if r.kind == "image"])
            segment_records.append({
                "index": seg_i, "goal": goal.prompt, "prompt": goal.prompt,
                "seed": seed_val, "attempts": 0, "scores": [], "chosen_take": None,
                "status": "resumed", "mp4": (f"{seg_name}/video.mp4" if mp4_path else None),
            })
            _emit("generating", None)
            logger.info("movie %s: segment %d RESUMED from existing bundle", job_id, seg_i)
            # iterative save: a resumed segment contributes its mp4 to the running
            # partial stitch + keeps movie.json current
            _save(partial=True)
            continue

        # ---- decide this segment's start image (explicit ref, else drift carry) ----
        explicit = goal.ref.uri if goal.ref is not None else None
        carry = prev_last_frame if (can_carry and seg_i > 0) else None
        seg_start_frame = explicit if explicit is not None else carry

        label_prompt = goal.prompt if len(goal.prompt) <= 60 else goal.prompt[:59] + "…"

        # ---- attempt loop (retry the WEAK takes when vision is on) ----
        best = None
        best_rank = None
        attempt_scores: "list" = []
        for attempt in range(spec.max_attempts_per_segment):
            # deterministic bumped seed: base + segment_index*1000 + attempt
            seg_seed = None if spec.seed is None else (spec.seed + seg_i * 1000 + attempt)
            seg_out = os.path.join(work_dir, f"seg_{seg_i:02d}_att_{attempt}")

            seg_refs: "list" = []
            seg_frame_paths: "list[str]" = []
            seg_frame_secs: "list[float]" = []
            _clock = [time.time()]

            def _on_frame_done(frame_path: str, i: int,
                               _refs=seg_refs, _paths=seg_frame_paths,
                               _secs=seg_frame_secs, _clk=_clock, _att=attempt) -> None:
                now = time.time()
                _secs.append(round(now - _clk[0], 3))
                _clk[0] = now
                ref = ingest(frame_path)
                _refs.append(ref)
                _paths.append(frame_path)
                current = {
                    "done": len(_paths),
                    "total": seg_n,
                    "stage": "generating",
                    "label": (f"segment {seg_i + 1}/{seg_total} · attempt {_att + 1} · "
                              f"frame {len(_paths)}/{seg_n} — {label_prompt}"),
                    "model": spec.model_id,
                    "frames": [asdict(r) for r in _refs if r.kind == "image"],
                }
                segments_meta[seg_i].update(
                    status="generating", attempt=_att,
                    frames=current["frames"])
                _emit("generating", current)

            err = render_scene_frames(
                model_id=spec.model_id,
                base_prompt=goal.prompt,
                n_frames=seg_n,
                width=spec.width,
                height=spec.height,
                steps=spec.steps,
                guidance=spec.guidance,
                seed=seg_seed,
                motion=None,
                negative=spec.negative,
                strength=spec.strength,
                chain=spec.chain,
                start_frame=seg_start_frame,
                out_dir=seg_out,
                job_id=job_id,
                on_frame_done=_on_frame_done,
            )
            if err is not None:
                # An expected per-segment failure (honest img2img_unavailable on an
                # explicit ref, cancel, plane error, ...) fails the whole movie —
                # DATA, never a raise. Save the completed segments so far + carry
                # the partial pointer (original error preserved).
                return _partial_return(err)
            if not seg_frame_paths:
                return _partial_return(JobResult(job_id, ok=False, error=JobError(
                    code="segment_no_frames",
                    message=f"segment {seg_i} produced no frames",
                    retryable=True)))

            # ---- score the KEY frame (last) when vision is enabled ----
            score = None
            verdict = None
            if spec.vision_enabled:
                segments_meta[seg_i].update(status="scoring")
                _emit("scoring", None)
                verdict = _score_keyframe(goal.prompt, seg_frame_paths[-1], spec.judge_model_id)
                score = verdict.get("score")
            attempt_scores.append(score)

            # keep the best-scoring take (None score ranks below any real score,
            # but a take is ALWAYS kept even when unscored)
            rank = score if score is not None else -1
            if best is None or rank > best_rank:
                best = {
                    "frame_paths": list(seg_frame_paths),
                    "refs": list(seg_refs),
                    "seed": seg_seed,
                    "out_dir": seg_out,
                    "score": score,
                    "verdict": verdict,
                    "attempt": attempt,
                    "per_frame_secs": list(seg_frame_secs),
                }
                best_rank = rank
            segments_meta[seg_i].update(score=score)

            # ---- retry decision ----
            if not spec.vision_enabled:
                break
            if score is None:                       # unscored -> keep, no retry
                break
            if score >= spec.score_threshold:       # good enough
                break
            if attempt + 1 >= spec.max_attempts_per_segment:
                break
            logger.info("movie %s: segment %d take %d scored %s < %s — retrying "
                        "with bumped seed", job_id, seg_i, attempt, score, spec.score_threshold)

        # ---- assemble this segment's mp4 + write its bundle INCREMENTALLY ----
        seg_mp4 = None
        if spec.assemble:
            _emit("assembling", None)
            seg_mp4 = os.path.join(best["out_dir"], f"{seg_name}.mp4")
            try:
                _assemble_scene_mp4(best["out_dir"], seg_mp4, spec.fps)
            except Exception as exc:
                return JobResult(job_id, ok=False, error=JobError(
                    code="segment_assembly_failed",
                    message=f"segment {seg_i}: {exc}",
                    retryable=False))

        seg_spec = make_generate_scene(
            parts=(text_part(goal.prompt),),
            model_id=spec.model_id, width=spec.width, height=spec.height,
            steps=spec.steps, guidance=spec.guidance,
            n_frames=seg_n, fps=spec.fps, assemble=spec.assemble,
            seed=best["seed"], negative=spec.negative,
            chain=spec.chain, strength=spec.strength, project=spec.project,
        )
        try:
            _write_bundle(
                spec=seg_spec, job_id=job_id,
                projectmeta=os.path.join(projectmeta, seg_name),
                frame_paths=best["frame_paths"], mp4_path=seg_mp4,
                base_prompt=goal.prompt, started_at=started_at,
                finished_at=time.time(), per_frame_secs=best["per_frame_secs"],
            )
        except Exception as exc:  # best-effort bundle (mirrors scene) — do NOT raise
            logger.warning("movie %s: segment %d bundle FAILED (non-fatal): %s: %s",
                           job_id, seg_i, type(exc).__name__, exc)

        # the bundle's copied video.mp4 is what the final concat consumes
        if seg_mp4 is not None:
            segment_mp4s.append(os.path.join(seg_bundle, "video.mp4"))

        all_refs.extend(best["refs"])
        prev_last_frame = best["frame_paths"][-1]

        segments_meta[seg_i].update(
            status="done", score=best["score"], attempt=best["attempt"],
            frames=[asdict(r) for r in best["refs"] if r.kind == "image"])
        segment_records.append({
            "index": seg_i, "goal": goal.prompt, "prompt": goal.prompt,
            "seed": best["seed"], "attempts": len(attempt_scores),
            "scores": attempt_scores, "chosen_take": best["attempt"],
            "status": "done",
            "why": (best["verdict"] or {}).get("why") if best["verdict"] else None,
            "mp4": (f"{seg_name}/video.mp4" if seg_mp4 is not None else None),
        })
        _emit("generating", None)
        # ITERATIVE save: after EVERY completed segment, refresh movie.json + the
        # running movie.mp4 of completed segments so an unfinishable movie always
        # leaves a watchable partial + up-to-date manifest on disk.
        _save(partial=True)

    # ---- FINAL best-effort finalize: full stitch + movie.json (partial=False) ----
    if spec.assemble and segment_mp4s:
        _emit("assembling", None)
    movie_manifest = _save(partial=False)

    # the stitched movie.mp4 is ingested LAST so outputs[-1] classifies as the video
    if movie_manifest.get("movie"):
        movie_mp4 = os.path.join(assets_root, "movie.mp4")
        if os.path.isfile(movie_mp4):
            all_refs.append(ingest(movie_mp4))

    _emit("archiving", None)
    return JobResult(
        job_id, ok=True, outputs=tuple(all_refs),
        project={"name": spec.project, "uuid": job_id, "dir": f"assets/{projectmeta}"},
        movie=movie_manifest,
    )


def _resume_segment(seg_bundle: str):
    """RESUME probe: if ``seg_bundle`` already holds a completed segment
    (project.json present), re-ingest its frames + locate its video.mp4 so the
    orchestrator can SKIP re-rendering it. Returns ``(frame_refs, mp4_path|None,
    seed)`` or None when the segment must (still) be rendered. Best-effort — any
    read/ingest hiccup returns None (re-render rather than trust a partial)."""
    pj = os.path.join(seg_bundle, "project.json")
    if not os.path.isfile(pj):
        return None
    try:
        with open(pj) as fh:
            manifest = json.load(fh)
        frame_names = manifest.get("frames") or []
        frame_refs = []
        for name in frame_names:
            fp = os.path.join(seg_bundle, name)
            if not os.path.isfile(fp):
                return None  # partial bundle -> re-render
            frame_refs.append(ingest(fp))
        if not frame_refs:
            return None
        mp4_path = None
        cand = os.path.join(seg_bundle, "video.mp4")
        if manifest.get("mp4") and os.path.isfile(cand):
            mp4_path = cand
        seeds = manifest.get("seeds")
        seed_val = seeds[0] if isinstance(seeds, list) and seeds else (
            seeds if isinstance(seeds, int) else None)
        return frame_refs, mp4_path, seed_val
    except Exception:
        return None
