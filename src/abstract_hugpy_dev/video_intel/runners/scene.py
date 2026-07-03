"""Pure `(diffusers, generate_scene)` runner — one query -> N consecutive frames.

`run_generate_scene(spec, job_id) -> JobResult`. Coherence mode is seed +
prompt-schedule v1 (NO img2img — the managers plane has no img2img pair wired;
confirmed). The runner walks n_frames SEQUENTIALLY, derives a per-frame prompt
(base + a positional tag + an optional motion schedule) and a per-frame seed
(base_seed + i when a seed is set), and drives the SAME inference plane as
generate_image (managers.dispatch.execute_prompt) once per frame. Each frame is
materialized to a padded frame_%05d.png under DEFAULT_ROOT and re-ingested. When
spec.assemble, the frames are muxed into a browser-playable H.264 mp4 (yuv420p +
faststart) that is ingested LAST (so it classifies as kind="video").

An img2img chain can drop in later WITHOUT reshaping the loop: swap the per-frame
execute_prompt call for an img2img call that also conditions on the previous
frame path (the sequential loop + padded frame paths already exist for it).

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

    # ---- GPU-worker guard ONCE before the loop (shared refusal policy) ----
    _refusal = guard_gpu_worker(spec.model_id, job_id)
    if _refusal is not None:
        return _refusal

    out_dir = os.path.join(DEFAULT_ROOT, "video_intel", "scenes", job_id)
    os.makedirs(out_dir, exist_ok=True)

    refs = []
    for i in range(spec.n_frames):
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

        # ---- drive the EXISTING plane once (lazy import; keep the runner pure) ----
        try:
            from abstract_hugpy_dev.managers.dispatch import execute_prompt
            from abstract_hugpy_dev._platform.async_runtime import run
            res = run(execute_prompt(**kwargs))
        except Exception as exc:  # unknown model / registry / plane error -> DATA
            return JobResult(job_id, ok=False, error=JobError(
                code="generation_failed",
                message=f"inference plane raised on frame {i}: {type(exc).__name__}: {exc}",
                retryable=True,
            ))

        if not getattr(res, "ok", False):
            return JobResult(job_id, ok=False, error=JobError(
                code="generation_failed",
                message=f"frame {i}: {str(getattr(res, 'error', None) or 'unknown')}",
                retryable=True,
            ))

        images = getattr(res, "images", None) or ()
        if not images:
            return JobResult(job_id, ok=False, error=JobError(
                code="generation_no_output",
                message=f"plane returned ok but produced no image for frame {i}",
                retryable=True,
            ))

        # ---- materialize the frame to a SEQUENTIAL padded path ----
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
                return JobResult(job_id, ok=False, error=JobError(
                    code="generation_no_output",
                    message=(f"frame {i}: generated image has no usable file path "
                             f"on disk and no inline bytes: {src_path!r}"),
                    retryable=False,
                ))
            with open(frame_path, "wb") as fh:
                fh.write(base64.b64decode(b64))

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
