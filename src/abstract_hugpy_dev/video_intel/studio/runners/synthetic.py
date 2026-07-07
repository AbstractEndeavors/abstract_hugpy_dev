"""SYNTHETIC i2v runner (P0-B1) — the no-model executor that proves the spine.

Given a ``RenderManifest`` it produces a real, playable H.264 clip with NO GPU and
NO weights: deterministic frames synthesized from ``manifest.seeds.global_seed`` +
the top ``Resolution`` in the ladder, assembled to mp4 via ffmpeg (the house
invocation mirrored from ``video_intel/runners/scene.py``).

Invariants honored:
  * INV-3  errors as data      — expected failures return ``Err(StageError)``;
                                 only genuine programmer error would raise.
  * INV-6  content-addressed    — output at ``<out_root>/<content_hash>/clip.mp4``
           + atomic + resumable   with ``manifest.json`` / ``provenance.json``
                                   sidecars; writes go temp -> ``os.replace``; an
                                   existing non-empty clip SKIPS regeneration.
  * Determinism  — same manifest ⇒ identical frame bytes (``synthesize_frame`` is
                   a pure function of seed + geometry + frame index).

Clip length is a function of the MANIFEST only (``fps * 2`` seconds, capped by the
bound model's ``max_frames``) so identical manifests yield identical-length clips
— a request's ``min_frames`` is honored upstream by the ROUTER (it rejects models
whose ``max_frames`` is below the floor), never by silently lengthening the clip
here, which would divorce clip length from the content hash (INV-6).

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Callable

import numpy as np
from PIL import Image

from ..artifacts import Artifact
from ..errors import Err, ErrorCode, Ok, Result, StageError
from ..manifest import render_manifest_to_dict
from ..registry import MODEL_REGISTRY
from ..schemas import ProvenanceStub, RenderManifest
from ..storage import atomic_write_text

# Serialize ONLY the mp4 assembly subprocess (house convention, mirrors
# scene.py's _SCENE_SEM). Frame synthesis is pure-python/numpy and not gated.
_ASSEMBLY_SEM = threading.BoundedSemaphore(1)

_CLIP_NAME = "clip.mp4"
_MANIFEST_NAME = "manifest.json"
_PROVENANCE_NAME = "provenance.json"


# --------------------------------------------------------------------------- #
# Pure, deterministic frame synthesis (testable in isolation)
# --------------------------------------------------------------------------- #
def _frame_params(seed: int) -> dict:
    """Seed-derived, resolution-independent pattern parameters. A fixed RNG stream
    off ``global_seed`` so the same seed always yields the same look."""
    rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    return {
        "fx": float(rng.uniform(2.0, 6.0)),
        "fy": float(rng.uniform(2.0, 6.0)),
        "fd": float(rng.uniform(2.0, 6.0)),
        "px": float(rng.uniform(0.0, 2.0 * math.pi)),
        "py": float(rng.uniform(0.0, 2.0 * math.pi)),
        "pd": float(rng.uniform(0.0, 2.0 * math.pi)),
        # per-channel colour phase offsets
        "cr": float(rng.uniform(0.0, 2.0 * math.pi)),
        "cg": float(rng.uniform(0.0, 2.0 * math.pi)),
        "cb": float(rng.uniform(0.0, 2.0 * math.pi)),
        # start-image tint gains + pan direction
        "gr": float(rng.uniform(0.85, 1.15)),
        "gg": float(rng.uniform(0.85, 1.15)),
        "gb": float(rng.uniform(0.85, 1.15)),
        "panx": float(rng.uniform(-1.0, 1.0)),
        "pany": float(rng.uniform(-1.0, 1.0)),
    }


def synthesize_frame(
    seed: int,
    width: int,
    height: int,
    frame_idx: int,
    n_frames: int,
    start_arr: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Deterministic HxWx3 uint8 frame. Pure function of its arguments — same
    inputs ⇒ byte-identical output. With ``start_arr`` it does a seeded tint +
    slow zoom-pan of the still; otherwise a seed-driven procedural plasma."""
    p = _frame_params(seed)
    denom = max(1, n_frames - 1)
    t = frame_idx / denom                     # 0..1 progress
    phase = 2.0 * math.pi * frame_idx / max(1, n_frames)  # loops smoothly

    if start_arr is not None:
        # --- seeded tint + slow zoom-pan of a still image ---
        img = Image.fromarray(start_arr)
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        zoom = 1.0 + 0.12 * t                 # zoom in up to 12%
        cw = max(2, int(round(width / zoom)))
        ch = max(2, int(round(height / zoom)))
        max_dx = width - cw
        max_dy = height - ch
        # deterministic pan sweeping from center outward along the seeded vector
        cx = (width - cw) / 2.0 + p["panx"] * (max_dx / 2.0) * t
        cy = (height - ch) / 2.0 + p["pany"] * (max_dy / 2.0) * t
        left = int(min(max(0, round(cx)), max_dx))
        top = int(min(max(0, round(cy)), max_dy))
        crop = img.crop((left, top, left + cw, top + ch)).resize(
            (width, height), Image.LANCZOS)
        arr = np.asarray(crop, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr[:, :, :3]
        # slow oscillating tint so motion is visible even on a flat still
        osc = 0.15 * math.sin(phase)
        gains = np.array([p["gr"] + osc, p["gg"] - osc, p["gb"] + osc], np.float32)
        out = np.clip(arr * gains, 0.0, 255.0).astype(np.uint8)
        return out

    # --- procedural plasma (no start image) ---
    yy, xx = np.mgrid[0:height, 0:width]
    xn = xx.astype(np.float32) / float(width)
    yn = yy.astype(np.float32) / float(height)
    v1 = np.sin(p["fx"] * xn * 2.0 * math.pi + phase + p["px"])
    v2 = np.sin(p["fy"] * yn * 2.0 * math.pi + phase * 1.3 + p["py"])
    v3 = np.sin(p["fd"] * (xn + yn) * math.pi * 2.0 + phase * 0.7 + p["pd"])
    base = (v1 + v2 + v3) / 3.0               # -1..1
    r = 0.5 + 0.5 * np.sin(base * math.pi + p["cr"])
    g = 0.5 + 0.5 * np.sin(base * math.pi + p["cg"])
    b = 0.5 + 0.5 * np.sin(base * math.pi + p["cb"])
    frame = np.stack([r, g, b], axis=-1)
    return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Manifest-derived geometry (clip length is a pure function of the manifest)
# --------------------------------------------------------------------------- #
def _geometry(manifest: RenderManifest) -> tuple[int, int, int, int]:
    """(width, height, fps, n_frames) from the top ladder rung + bound model cap."""
    top = manifest.resolution_ladder[0]
    fps = top.fps
    n = fps * 2                               # a short ~2s clip
    cfg = MODEL_REGISTRY.get(manifest.model_id)
    if cfg is not None and cfg.max_frames:
        n = min(n, cfg.max_frames)
    return top.width, top.height, fps, max(1, n)


def _assemble_mp4(frame_dir: str, tmp_mp4: str, fps: int) -> tuple[bool, str]:
    """Mux frame_%05d.png -> H.264 mp4 (house invocation from scene.py). Returns
    (ok, stderr_tail); never raises on a plain ffmpeg failure."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-start_number", "0",
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_mp4,
    ]
    with _ASSEMBLY_SEM:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = result.returncode == 0 and os.path.isfile(tmp_mp4) and os.path.getsize(tmp_mp4) > 0
    return ok, (result.stderr or "")[-500:]


def _provenance_dict(manifest: RenderManifest) -> dict:
    prov = manifest.provenance
    if prov is None:
        from datetime import datetime, timezone
        prov = ProvenanceStub(
            operator="synthetic-runner",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    return {
        "operator": prov.operator,
        "created_at": prov.created_at,
        "tool": prov.tool,
        "c2pa_pending": prov.c2pa_pending,
    }


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_synthetic_i2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a synthetic clip for ``manifest`` under ``out_root``.

    Returns ``Ok(Artifact)`` on success, ``Err(StageError)`` on any expected
    failure (unwritable output tree, bad start image, ffmpeg mux failure). Only a
    genuine programmer error (e.g. a non-RenderManifest) raises.

    ``should_cancel`` is an OPTIONAL cooperative-cancel probe (Task 1): a zero-arg
    callable polled at the TOP of every frame. When it returns True the loop aborts
    BEFORE the atomic ``os.replace`` — so no clip.mp4 lands at the content-addressed
    path — and returns ``Err(StageError(CANCELLED, ...))``. The temp frame dir is
    cleaned by the existing ``finally``. None (default) = never cancel."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    content_hash = manifest.content_hash()
    width, height, fps, n_frames = _geometry(manifest)
    duration_s = n_frames / float(fps)
    out_dir = os.path.join(os.path.abspath(out_root), content_hash)
    clip_path = os.path.join(out_dir, _CLIP_NAME)

    # INV-6 resume: an existing non-empty clip is returned as-is, no regeneration.
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return Ok(Artifact(
            path=clip_path, content_hash=content_hash, frames=n_frames,
            width=width, height=height, duration_s=duration_s, resumed=True))

    # Load the start image up front (an unreadable one is DATA, not a crash).
    start_arr = None
    if start_image is not None:
        try:
            with Image.open(start_image) as im:
                start_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception as exc:  # unreadable/corrupt still -> errors-as-data
            return Err(StageError(
                ErrorCode.IO_ERROR,
                f"could not read start_image: {exc}",
                (("start_image", str(start_image)),),
            ))

    frame_dir = None
    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)
        frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i in range(n_frames):
            # Cooperative mid-render cancel: honored BETWEEN frames. Aborting here
            # (before the atomic os.replace below) guarantees NO clip lands at the
            # content-addressed path; the finally cleans up frame_dir (and there is
            # no tmp_mp4 yet). Mirrors scene.py's per-frame is_cancelling() poll.
            if should_cancel is not None and should_cancel():
                return Err(StageError(
                    ErrorCode.CANCELLED,
                    f"cancelled mid-render after {i} of {n_frames} frame(s)",
                    (("content_hash", content_hash), ("frames", str(n_frames))),
                ))
            frame = synthesize_frame(
                manifest.seeds.global_seed, width, height, i, n_frames, start_arr)
            Image.fromarray(frame).save(
                os.path.join(frame_dir, f"frame_{i:05d}.png"), "PNG")

        # NOTE: the temp name keeps a .mp4 extension so ffmpeg infers the mp4
        # muxer (it keys off the extension); it stays in out_dir for an atomic
        # same-filesystem os.replace onto clip.mp4.
        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _assemble_mp4(frame_dir, tmp_mp4, fps)
        if not ok:
            return Err(StageError(
                ErrorCode.ASSEMBLY_FAILED,
                f"ffmpeg mux failed: {stderr_tail}",
                (("content_hash", content_hash), ("frames", str(n_frames))),
            ))

        os.replace(tmp_mp4, clip_path)        # atomic promotion of the clip
        tmp_mp4 = None

        # Sidecars (INV-1/INV-7): the manifest that defines this render + the
        # provenance stub, both written atomically alongside the pixels.
        atomic_write_text(
            os.path.join(out_dir, _MANIFEST_NAME),
            json.dumps(render_manifest_to_dict(manifest), indent=2, sort_keys=True))
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(_provenance_dict(manifest), indent=2, sort_keys=True))
    except OSError as exc:                     # unwritable out_root, disk full, ...
        return Err(StageError(
            ErrorCode.IO_ERROR,
            f"synthetic runner IO failure: {exc}",
            (("out_dir", out_dir),),
        ))
    finally:
        if tmp_mp4 is not None and os.path.isfile(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except OSError:
                pass
        if frame_dir is not None and os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)

    return Ok(Artifact(
        path=clip_path, content_hash=content_hash, frames=n_frames,
        width=width, height=height, duration_s=duration_s, resumed=False))


# --------------------------------------------------------------------------- #
# SYNTHETIC t2v (Task 3b) — the no-model TEXT-to-video prover
# --------------------------------------------------------------------------- #
def run_synthetic_t2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a synthetic TEXT-to-video clip for ``manifest``.

    A thin, DETERMINISTIC delegation to ``run_synthetic_i2v`` with
    ``start_image`` forced to None: text-to-video has no conditioning still, so a
    supplied start_image is DELIBERATELY IGNORED (never tints/pans a frame). The
    frames are a pure function of ``manifest.seeds.global_seed`` + geometry — the
    PROMPT rides in the manifest (and thus the content_hash + ``manifest.json``
    sidecar) for provenance, but never alters a pixel, so t2v stays
    byte-deterministic. Identical content-addressed atomic layout / resume /
    errors-as-data as the i2v path (it IS the i2v path). Only a genuine programmer
    error (a non-RenderManifest) raises — inherited from ``run_synthetic_i2v``."""
    return run_synthetic_i2v(
        manifest, out_root, start_image=None, should_cancel=should_cancel)
