"""Pure `(diffusers, generate_scene)` runner — one query -> N consecutive frames.

`run_generate_scene(spec, job_id) -> JobResult`. Two coherence modes, chosen by
whether the spec carries a START-FRAME image part (the FIRST image part):

  * NO start frame -> v1: seed + prompt-schedule. The runner walks n_frames
    SEQUENTIALLY, derives a per-frame prompt (base + a positional tag + an
    optional motion schedule) and a per-frame seed (base_seed + i), and drives
    the SAME text-to-image plane (managers.dispatch.execute_prompt) once/frame.

  * START-FRAME present -> IMG2IMG (image-to-image plane). If img2img is not
    available on the fleet (the sd-turbo advertisement flip is HELD), returns a
    retryable `image_to_image_unavailable` JobError — an HONEST failure, never a
    silent fall back to text-to-image. When available:
      - chain=True  (default): frame 0 conditions on the start frame; each later
        frame conditions on the PREVIOUS frame's saved output (true sequential
        chaining) with base + motion step i.
      - chain=False: every frame conditions on the start frame (no drift) with
        base + motion step i.

Each frame is materialized to a padded frame_%05d.png under DEFAULT_ROOT and
re-ingested. When spec.assemble, the frames are muxed into a browser-playable
H.264 mp4 (yuv420p + faststart) that is ingested LAST (so it classifies as
kind="video").

Pure discipline (map §6): EXPECTED failures (no prompt, an unresolved video part,
the plane raising / returning not-ok / no usable output, assembly failing) are
returned as JobResult(ok=False, JobError(...)) — DATA, never a raise. Only the
worker loop may catch an UNEXPECTED raise.

Import/purity discipline mirrors imagegen.py: the inference-plane import is LAZY
(inside the loop) so importing `video_intel.runners` never couples to the health
of the managers/dispatch plane.

CONCURRENCY BOUND: only the mp4 ASSEMBLY subprocess is serialized behind a
module-level BoundedSemaphore(1) (like ffmpeg_frames' fan-out). Generation is
remote-GPU-bound and is intentionally NOT gated here.
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import threading

from abstract_hugpy_dev._platform.binaries import resolve_bin
from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT

from ..media_store import ingest
from ..result_schema import JobError, JobResult
from ..scene_schema import GenerateSceneSpec, FRAME_CAP
from ._gpu_guard import guard_gpu_worker
from ._img2img import img2img_available

logger = logging.getLogger(__name__)

# Serialize ONLY the mp4 assembly subprocess (see module docstring). Generation
# is remote-GPU-bound and is not gated here.
_SCENE_SEM = threading.BoundedSemaphore(1)


def _assemble_scene_mp4(frame_dir: str, mp4_path: str, fps: int) -> None:
    """Mux frame_%05d.png in `frame_dir` into a browser-playable H.264 mp4.

    Raises RuntimeError on ffmpeg failure; the runner converts it into a
    scene_assembly_failed JobError (never a raise across the job boundary). The
    scale filter forces even dimensions (libx264 + yuv420p require them).
    """
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path,
    ]
    with _SCENE_SEM:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(mp4_path):
        raise RuntimeError(
            f"ffmpeg assembly failed rc={result.returncode}: {(result.stderr or '')[-500:]}"
        )


def _scene_concurrency(model_id: str, n_frames: int) -> int:
    """How many frames to generate at once. HUGPY_SCENE_CONCURRENCY pins it;
    otherwise self-tune to the number of ONLINE workers assigned this model
    (the delegating plane's least-recently-picked routing spreads concurrent
    requests across them, and each worker serializes same-model generates via
    the runners' per-model lock — so extra in-flight frames just queue).
    Floor 1 = today's sequential behavior on a one-worker fleet."""
    env = (os.environ.get("HUGPY_SCENE_CONCURRENCY") or "").strip()
    if env:
        try:
            return max(1, min(int(env), n_frames))
        except ValueError:
            pass
    try:
        from ...flask_app.app.functions.imports.utils.workers import list_workers
        live = sum(
            1 for w in list_workers()
            if w.get("status") == "online" and model_id in (w.get("models") or [])
        )
        return max(1, min(live, n_frames))
    except Exception:
        return 1  # registry unavailable (e.g. worker-side import) — sequential


def _generate_frames_parallel(frame_kwargs: "list[dict]", out_dir: str,
                              job_id: str, model_id: str):
    """Tier-1 scene fan-out: generate INDEPENDENT frames concurrently.

    Only for orders where frames don't feed each other — v1 (seed-schedule
    text-to-image) and img2img chain=False (every frame off the START frame).
    chain=True stays strictly sequential at the call site. Cooperative cancel:
    frames not yet started become no-ops once 'cancelling' is set; in-flight
    frames finish (same between-frames contract as the sequential path).
    Returns (ordered_frame_paths, None) or (None, JobResult error)."""
    from concurrent.futures import ThreadPoolExecutor
    from ..media_bus import is_cancelling

    n = len(frame_kwargs)
    workers = _scene_concurrency(model_id, n)
    if workers <= 1:
        paths: list = []
        for i, kw in enumerate(frame_kwargs):
            if is_cancelling(job_id):
                return None, JobResult(job_id, ok=False, error=JobError(
                    code="cancelled",
                    message=f"cancelled after {i} of {n} frame(s)",
                    retryable=False))
            frame_path, err = _generate_one_frame(kw, out_dir, i, job_id)
            if err is not None:
                return None, err
            paths.append(frame_path)
        return paths, None

    logger.info("scene %s: fan-out %d frames across up to %d concurrent "
                "generates (model=%s)", job_id, n, workers, model_id)

    def _task(i: int, kw: dict):
        if is_cancelling(job_id):
            return i, None, "cancelled"
        frame_path, err = _generate_one_frame(kw, out_dir, i, job_id)
        return i, frame_path, err

    results: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame_path, err in pool.map(
                lambda args: _task(*args), enumerate(frame_kwargs)):
            results[i] = (frame_path, err)

    cancelled = sum(1 for p, e in results.values() if e == "cancelled")
    if cancelled:
        done = sum(1 for p, e in results.values() if e is None)
        return None, JobResult(job_id, ok=False, error=JobError(
            code="cancelled",
            message=f"cancelled after {done} of {n} frame(s)",
            retryable=False))
    for i in range(n):  # first REAL error by frame order, deterministic
        _path, err = results[i]
        if err is not None:
            return None, err
    return [results[i][0] for i in range(n)], None


def _generate_one_frame(kwargs: dict, out_dir: str, i: int, job_id: str):
    """Drive the EXISTING inference plane once and materialize the result to a
    SEQUENTIAL padded frame_%05d.png. Returns (frame_path, None) on success, or
    (None, JobResult) carrying a JobError on any expected failure.

    Shared by BOTH the v1 (text-to-image) and img2img loops so the plane-drive +
    materialize logic (and its DATA error shapes) exist in exactly one place. The
    inference-plane import stays LAZY (keeps this runner pure)."""
    try:
        from abstract_hugpy_dev.managers.dispatch import execute_prompt
        from abstract_hugpy_dev._platform.async_runtime import run
        res = run(execute_prompt(**kwargs))
    except Exception as exc:  # unknown model / registry / plane error -> DATA
        return None, JobResult(job_id, ok=False, error=JobError(
            code="generation_failed",
            message=f"inference plane raised on frame {i}: {type(exc).__name__}: {exc}",
            retryable=True,
        ))

    if not getattr(res, "ok", False):
        return None, JobResult(job_id, ok=False, error=JobError(
            code="generation_failed",
            message=f"frame {i}: {str(getattr(res, 'error', None) or 'unknown')}",
            retryable=True,
        ))

    images = getattr(res, "images", None) or ()
    if not images:
        return None, JobResult(job_id, ok=False, error=JobError(
            code="generation_no_output",
            message=f"plane returned ok but produced no image for frame {i}",
            retryable=True,
        ))

    frame_path = os.path.join(out_dir, f"frame_{i:05d}.png")
    img = images[0]
    src_path = getattr(img, "path", None)
    if src_path and os.path.isfile(src_path):
        shutil.copyfile(src_path, frame_path)
    else:
        # Plane served this remotely: its `path` lives on THAT machine; the
        # inline b64 bytes are how the artifact reaches this machine.
        b64 = getattr(img, "b64", None)
        if not b64:
            return None, JobResult(job_id, ok=False, error=JobError(
                code="generation_no_output",
                message=(f"frame {i}: generated image has no usable file path "
                         f"on disk and no inline bytes: {src_path!r}"),
                retryable=False,
            ))
        with open(frame_path, "wb") as fh:
            fh.write(base64.b64decode(b64))
    return frame_path, None


def run_generate_scene(spec: GenerateSceneSpec, job_id: str) -> JobResult:
    # ---- defensive frame cap (belt-and-suspenders; the factory already guards) ----
    if spec.n_frames > FRAME_CAP:
        return JobResult(job_id, ok=False, error=JobError(
            code="frame_cap_exceeded",
            message=f"n_frames={spec.n_frames} exceeds cap {FRAME_CAP}",
            retryable=False,
        ))

    # ---- safety net: video parts MUST be resolved to image parts upstream ----
    # (chains.resolve_video_parts_scene runs in the enqueue path). If a raw video
    # part reaches the runner, refuse loudly rather than silently drop it.
    for i, p in enumerate(spec.parts):
        if p.kind == "video":
            return JobResult(job_id, ok=False, error=JobError(
                code="unresolved_video_part",
                message=(
                    f"part[{i}] is a raw video part; it must be resolved to "
                    "image frames via chains.resolve_video_parts_scene before enqueue"
                ),
                retryable=False,
            ))

    # ---- resolve the base prompt: join text parts (ordered) ----
    texts = [p.text.strip() for p in spec.parts
             if p.kind == "text" and p.text and p.text.strip()]
    base_prompt = "\n".join(texts).strip()
    if not base_prompt:
        return JobResult(job_id, ok=False, error=JobError(
            code="no_prompt",
            message="generate_scene has no text part to build a prompt from",
            retryable=False,
        ))

    # ---- start-frame image (img2img seam) ----
    # The FIRST image part is the init/start frame. Reuse the existing parts list
    # (no new init_image field); its local path is media.uri (parts materialize
    # exactly as in generate_image).
    start_frame = next(
        (p.media.uri for p in spec.parts if p.kind == "image" and p.media is not None),
        None,
    )
    # Runner-applied default when None (LOCKED CONTRACT): 0.45.
    strength = spec.strength if spec.strength is not None else 0.45

    # A start-frame image REQUIRES a servable img2img pair. The sd-turbo
    # advertisement flip is HELD, so on live central this is FALSE -> honest,
    # retryable failure. NEVER silently fall back to text-to-image.
    if start_frame is not None and not img2img_available(spec.model_id):
        logger.info(
            "scene %s: start-frame image present but image-to-image is not "
            "available on the fleet (model=%s); returning honest failure",
            job_id, spec.model_id,
        )
        return JobResult(job_id, ok=False, error=JobError(
            code="image_to_image_unavailable",
            message="image-to-image not available on the fleet",
            retryable=True,
        ))

    # ---- GPU-worker guard ONCE before the loop (shared refusal policy) ----
    # The img2img probe above is ADDITIONAL to this guard, not a replacement.
    _refusal = guard_gpu_worker(spec.model_id, job_id)
    if _refusal is not None:
        return _refusal

    out_dir = os.path.join(DEFAULT_ROOT, "video_intel", "scenes", job_id)
    os.makedirs(out_dir, exist_ok=True)

    refs = []

    if start_frame is not None:
        # ---- img2img path: condition each frame on an init image ----
        mode = "chain" if spec.chain else "parallel"
        logger.info(
            "scene %s: IMG2IMG-%s path (model=%s start_frame=%s strength=%s n=%d)",
            job_id, mode, spec.model_id, os.path.basename(start_frame),
            strength, spec.n_frames,
        )
        def _img2img_kwargs(i: int, cond_path: str) -> dict:
            # chain:    frame 0 = base prompt; frame i>0 = base + motion step i.
            # parallel: every frame = base + motion step i (all off the start frame).
            include_motion = bool(spec.motion) and (spec.chain is False or i > 0)
            prompt = base_prompt
            if include_motion:
                # .replace (NOT .format): user strings may contain stray braces.
                prompt += ", " + spec.motion.replace("{i}", str(i + 1)).replace("{n}", str(spec.n_frames))
            kwargs = dict(
                task="image-to-image",
                image_path=cond_path,
                strength=strength,
                prompt=prompt,
                model_key=spec.model_id,
                width=spec.width,
                height=spec.height,
                num_inference_steps=spec.steps,
                guidance_scale=spec.guidance,
                num_images=1,
                return_b64=True,
            )
            if spec.seed is not None:
                kwargs["seed"] = spec.seed + i
            if spec.negative is not None:
                kwargs["negative_prompt"] = spec.negative
            return kwargs

        if spec.chain:
            # TRUE sequential chaining — frame i+1 consumes frame i's output;
            # inherently unparallelizable.
            prev_output = start_frame
            for i in range(spec.n_frames):
                # Cooperative cancel — honored BETWEEN frames (mid-frame
                # inference is never interrupted).
                from ..media_bus import is_cancelling
                if is_cancelling(job_id):
                    return JobResult(job_id, ok=False, error=JobError(
                        code="cancelled",
                        message=f"cancelled after {i} of {spec.n_frames} frame(s)",
                        retryable=False))
                frame_path, err = _generate_one_frame(
                    _img2img_kwargs(i, prev_output), out_dir, i, job_id)
                if err is not None:
                    return err
                prev_output = frame_path
                # Re-ingest so dims/mime are authoritatively resolved (§9.2).
                refs.append(ingest(frame_path))
        else:
            # parallel mode: every frame conditions on the START frame — the
            # frames are independent, so fan them out across the fleet (the
            # plane's least-recently-picked routing spreads concurrent calls).
            frame_paths, err = _generate_frames_parallel(
                [_img2img_kwargs(i, start_frame) for i in range(spec.n_frames)],
                out_dir, job_id, spec.model_id)
            if err is not None:
                return err
            for frame_path in frame_paths:
                refs.append(ingest(frame_path))
    else:
        # ---- v1 path: seed + prompt-schedule (NO img2img) ----
        # Frames are seed/prompt-scheduled but INDEPENDENT (frame i never sees
        # frame i-1), so they fan out like img2img parallel mode.
        logger.info(
            "scene %s: V1 text-to-image path (seed+prompt-schedule; model=%s n=%d)",
            job_id, spec.model_id, spec.n_frames,
        )

        def _v1_kwargs(i: int) -> dict:
            # ---- per-frame prompt schedule (v1): positional tag + optional motion ----
            prompt = base_prompt + f", frame {i + 1} of {spec.n_frames}"
            if spec.motion:
                # .replace (NOT .format): user strings may contain stray braces.
                prompt += ", " + spec.motion.replace("{i}", str(i + 1)).replace("{n}", str(spec.n_frames))
            # ---- per-frame kwargs — SAME shape as imagegen ----
            kwargs = dict(
                task="text-to-image",
                prompt=prompt,
                model_key=spec.model_id,
                width=spec.width,
                height=spec.height,
                num_inference_steps=spec.steps,
                guidance_scale=spec.guidance,
                num_images=1,
                return_b64=True,
            )
            # per-frame seed: base_seed + i for coherence; omit -> random per frame
            if spec.seed is not None:
                kwargs["seed"] = spec.seed + i
            if spec.negative is not None:
                kwargs["negative_prompt"] = spec.negative
            return kwargs

        frame_paths, err = _generate_frames_parallel(
            [_v1_kwargs(i) for i in range(spec.n_frames)],
            out_dir, job_id, spec.model_id)
        if err is not None:
            return err
        for frame_path in frame_paths:
            # Re-ingest so the frame's dims/mime are authoritatively resolved (§9.2).
            refs.append(ingest(frame_path))

    # ---- optional assembly: frames -> browser-playable mp4 (ingested LAST) ----
    if spec.assemble:
        mp4_path = os.path.join(out_dir, f"{job_id}.mp4")
        try:
            _assemble_scene_mp4(out_dir, mp4_path, spec.fps)
        except Exception as exc:  # ffmpeg mux failure -> DATA (not retryable)
            return JobResult(job_id, ok=False, error=JobError(
                code="scene_assembly_failed",
                message=str(exc),
                retryable=False,
            ))
        # ingest LAST so the video ref is outputs[-1] (classifies as kind="video").
        refs.append(ingest(mp4_path))

    return JobResult(job_id, ok=True, outputs=tuple(refs))
