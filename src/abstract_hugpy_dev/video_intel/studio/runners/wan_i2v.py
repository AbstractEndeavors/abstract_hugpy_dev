"""REAL Wan i2v runner (P0-6) — the first weight-backed executor behind the
studio's runner contract, structurally complete and plug-in ready for the
4x3090 box (bitsandbytes int8/nf4), IMPORT-SAFE and GRACEFULLY-DEGRADING now.

It mirrors ``synthetic.run_synthetic_i2v`` exactly:

    run_wan_i2v(manifest, out_root, start_image=None) -> Result[Artifact, StageError]

Same content-addressed atomic layout (``<out_root>/<content_hash>/clip.mp4`` +
``manifest.json`` + ``provenance.json``), same resume-on-hash, same errors-as-data
discipline (INV-3/INV-6). The ffmpeg assembly + sidecar helpers are REUSED from
``synthetic`` so the on-disk shape is byte-for-byte the same contract.

IMPORT SAFETY (hard requirement): torch / diffusers / transformers /
bitsandbytes are NEVER imported at module top — only lazily INSIDE the runner,
after preflight passes. Importing this module (or the studio package, or the
Flask app) pulls only stdlib + the studio's own light modules, so app boot never
drags in the heavy GPU stack and never fails on a box without it.

GRACEFUL DEGRADATION (this dev VM has NO GPU / NO weights): preflight returns
``Err(StageError(...))`` as DATA, never raises:
  * missing torch/diffusers/transformers/bitsandbytes/accelerate -> DEPS_MISSING
  * no CUDA device                                               -> NO_GPU
  * model weights not on disk under the weights root             -> WEIGHTS_MISSING
Only genuine programmer error (a non-RenderManifest) raises.

REAL PATH (runs only when preflight passes, i.e. on the box): loads the Wan i2v
pipeline via diffusers with a bitsandbytes quantized transformer (operator
directive: "utilize bitsandbytes"), runs i2v from ``start_image`` (or t2v when
None) at the manifest's resolution / frame-count / seed / sampler, writes the
frames, and ffmpeg-assembles them into the same atomic content-addressed clip.
Diffusers pipeline classes used: ``WanImageToVideoPipeline`` (i2v) /
``WanPipeline`` (t2v), ``WanTransformer3DModel`` (bnb-quantized),
``AutoencoderKLWan`` (fp32 VAE), ``diffusers.BitsAndBytesConfig``.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Callable

from ..artifacts import Artifact
from ..enums import Precision
from ..errors import Err, ErrorCode, Ok, Result, StageError
from ..manifest import render_manifest_to_dict
from ..registry import MODEL_REGISTRY
from ..schemas import RenderManifest
from ..storage import atomic_write_text
# Reuse the synthetic runner's atomic/content-addressed plumbing so the Wan clip
# lands in the IDENTICAL on-disk layout. These pull numpy/PIL (already house
# deps, present everywhere) — NOT the heavy torch/diffusers stack, which stays
# lazy inside run_wan_i2v.
from .synthetic import (
    _CLIP_NAME,
    _MANIFEST_NAME,
    _PROVENANCE_NAME,
    _assemble_mp4,
    _geometry,
    _provenance_dict,
)

# Python deps the REAL inference path needs. Preflight reports any that are
# absent as DEPS_MISSING data (never an ImportError at module import).
_REQUIRED_DEPS = ("torch", "diffusers", "transformers", "bitsandbytes", "accelerate")


# --------------------------------------------------------------------------- #
# Weights / geometry resolution (pure, no heavy deps)
# --------------------------------------------------------------------------- #
def _weights_root(manifest: RenderManifest) -> str | None:
    """The weights root, sourced FIRST from the manifest's captured env_snapshot
    (``STUDIO_WEIGHTS_ROOT`` was threaded there by ``env.to_snapshot()`` at build
    time, INV-5), falling back to the live process env. None if neither is set."""
    snap = dict(manifest.env_snapshot)
    return snap.get("STUDIO_WEIGHTS_ROOT") or os.environ.get("STUDIO_WEIGHTS_ROOT")


def _local_model_dir(weights_root: str, weight_uri: str) -> str:
    """Local on-disk dir for an HF-style ``org/name`` weight_uri, mirrored under
    the weights root (``<weights_root>/<org>/<name>``)."""
    parts = [p for p in weight_uri.split("/") if p]
    return os.path.join(weights_root, *parts)


def _wan_geometry(manifest: RenderManifest) -> tuple[int, int, int, int]:
    """(width, height, fps, n_frames) mirroring synthetic's ``_geometry`` but
    snapped to Wan's temporal cadence: the latent VAE compresses time 4:1, so the
    pipeline requires ``num_frames == 4*k + 1`` (e.g. 81). Snapping here (not in
    the real path) keeps the resume check and the generation call agreeing on the
    exact frame count."""
    width, height, fps, n = _geometry(manifest)
    n = max(1, n)
    n = ((n - 1) // 4) * 4 + 1        # nearest 4k+1 <= n
    return width, height, fps, n


def _missing_deps() -> list[str]:
    """Which of the heavy inference deps are absent — checked via find_spec so we
    never actually import (and thus never fail-loud) at preflight."""
    import importlib.util
    missing: list[str] = []
    for mod in _REQUIRED_DEPS:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing


# --------------------------------------------------------------------------- #
# Preflight — errors as data (returns a StageError to raise-as-Err, or None)
# --------------------------------------------------------------------------- #
def _preflight(manifest: RenderManifest) -> StageError | None:
    """Gate the real path. Returns a ``StageError`` (the caller wraps it in
    ``Err``) when the box can't run Wan i2v yet, or None when everything the real
    path needs is present. Order: deps -> GPU -> weights (each needs the prior)."""
    missing = _missing_deps()
    if missing:
        return StageError(
            ErrorCode.DEPS_MISSING,
            "Wan i2v needs GPU inference deps that are not installed: "
            + ", ".join(missing)
            + ". Install: pip install torch (CUDA build) diffusers transformers "
              "bitsandbytes accelerate",
            (("missing", ",".join(missing)),),
        )

    import torch  # lazy — only reached once torch is importable
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        return StageError(
            ErrorCode.NO_GPU,
            "no CUDA device available; Wan i2v requires a CUDA GPU (the 4x3090 "
            "box) for bitsandbytes int8/nf4 inference",
            (("cuda", "unavailable"), ("model_id", manifest.model_id)),
        )

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    if cfg is None:
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            f"model_id {manifest.model_id!r} is not in the studio registry",
            (("model_id", manifest.model_id),),
        )

    weights_root = _weights_root(manifest)
    if not weights_root:
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "STUDIO_WEIGHTS_ROOT is not set — no weights root to resolve the Wan "
            "weights against",
            (("model_id", manifest.model_id),),
        )

    model_dir = _local_model_dir(weights_root, cfg.weight_uri)
    if not (os.path.isdir(model_dir)
            and os.path.isfile(os.path.join(model_dir, "model_index.json"))):
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            f"Wan weights not found on disk at {model_dir}; download with "
            f"`huggingface-cli download {cfg.weight_uri} --local-dir {model_dir}`",
            (("model_dir", model_dir), ("weight_uri", cfg.weight_uri)),
        )
    return None


# --------------------------------------------------------------------------- #
# Precision -> bitsandbytes quantization (operator directive: int8 / nf4)
# --------------------------------------------------------------------------- #
def _bnb_config(precision: Precision, BitsAndBytesConfig, torch):
    """Map the router-selected precision to a bitsandbytes quant config:
      * INT8      -> load_in_8bit  (bnb int8)
      * FP8       -> load_in_4bit + nf4  (the tightest bnb path, ~4bit)
      * BF16/FP16 -> None (caller has the VRAM; load unquantized in bf16)
    Returns None to mean "no bnb quantization"."""
    if precision == Precision.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == Precision.FP8:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_wan_i2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a Wan i2v clip for ``manifest`` under ``out_root``.

    Returns ``Ok(Artifact)`` on a real render (on the box), or ``Err(StageError)``
    on any expected failure — including the preflight failures that make this a
    graceful no-op on a GPU-less / weight-less box (DEPS_MISSING / NO_GPU /
    WEIGHTS_MISSING). Only a genuine programmer error (a non-RenderManifest)
    raises.

    ``should_cancel`` is an OPTIONAL cooperative-cancel probe (Task 1): a zero-arg
    callable polled at the natural checkpoints (before load, between load and
    render, after render) and — during denoise — wired into the pipeline via
    diffusers' ``callback_on_step_end`` (the callback sets ``pipe._interrupt=True``
    so the loop breaks at the next step boundary). A cancel at any checked point
    returns ``Err(StageError(CANCELLED, ...))`` BEFORE any clip is written. NOTE:
    TRUE mid-denoise interruption is BOX-ONLY — this GPU-less VM short-circuits at
    preflight below, so the callback path only ever executes on the real box. None
    (default) = never cancel."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    content_hash = manifest.content_hash()
    width, height, fps, n_frames = _wan_geometry(manifest)
    out_dir = os.path.join(os.path.abspath(out_root), content_hash)
    clip_path = os.path.join(out_dir, _CLIP_NAME)

    # INV-6 resume FIRST: an existing non-empty clip is served as-is, with NO GPU
    # and NO reload — a box that rendered it can return it later even offline.
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return Ok(Artifact(
            path=clip_path, content_hash=content_hash, frames=n_frames,
            width=width, height=height, duration_s=n_frames / float(fps),
            resumed=True))

    # PREFLIGHT: everything below the real path returns as DATA, never raises.
    pf = _preflight(manifest)
    if pf is not None:
        return Err(pf)

    # ----------------------------------------------------------------------- #
    # REAL PATH — only reached on a box with deps + CUDA + weights on disk.
    # Never executes on the dev VM (preflight short-circuits above). Written
    # complete enough to run once the 4x3090 box is live.
    # ----------------------------------------------------------------------- #
    import torch
    from diffusers import (
        AutoencoderKLWan,
        BitsAndBytesConfig,
        WanImageToVideoPipeline,
        WanPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import load_image

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    model_dir = _local_model_dir(_weights_root(manifest), cfg.weight_uri)
    compute_dtype = torch.bfloat16
    quant_config = _bnb_config(manifest.precision, BitsAndBytesConfig, torch)
    seed = manifest.seeds.global_seed
    steps = manifest.sampler.steps
    cfg_scale = manifest.sampler.cfg
    # C-prompt: text conditioning from the manifest (part of its content_hash). An
    # empty prompt is valid (image-conditioned i2v); an empty negative maps to None
    # so the pipeline uses its own default rather than an explicit "" negative.
    prompt = manifest.prompt
    negative_prompt = manifest.negative_prompt or None

    # Cooperative mid-render cancel wiring (Task 1). diffusers 0.39's
    # WanImageToVideoPipeline.__call__ supports `callback_on_step_end`; the callback
    # sets `pipe._interrupt=True` so the denoise loop breaks at the next step
    # boundary. We ALSO re-check should_cancel() around the call so a cancel is
    # still honored if a box's diffusers lacks the callback param. This whole path
    # is BOX-ONLY (preflight short-circuits the GPU-less VM above).
    def _cancel_step_cb(pipe_ref, step_index, timestep, cb_kwargs):
        if should_cancel is not None and should_cancel():
            pipe_ref._interrupt = True   # diffusers checks self.interrupt each step
        return cb_kwargs

    call_extra: dict = {}
    if should_cancel is not None:
        call_extra["callback_on_step_end"] = _cancel_step_cb

    frame_dir = None
    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Cooperative cancel — BEFORE load (no weights touched yet if we bail).
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled before wan load",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # bitsandbytes-quantized DiT transformer (int8 / nf4 per precision).
        tf_kwargs = {"subfolder": "transformer", "torch_dtype": compute_dtype}
        if quant_config is not None:
            tf_kwargs["quantization_config"] = quant_config
        transformer = WanTransformer3DModel.from_pretrained(model_dir, **tf_kwargs)
        # Wan's VAE is numerically sensitive; the diffusers Wan reference loads it
        # in fp32 (it is small relative to the DiT, so this is affordable).
        vae = AutoencoderKLWan.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float32)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Cooperative cancel — BETWEEN load and render (weights loaded, nothing
        # rendered/written yet). Per-step interruption during render is handled by
        # the callback below.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled after wan load, before render",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        if start_image is not None:
            # --- i2v ---
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_dir, transformer=transformer, vae=vae,
                torch_dtype=compute_dtype)
            # bnb-quantized weights stay put; offload the rest to fit 3090-class
            # VRAM. (Do NOT call .to("cuda") on an 8bit/4bit model.)
            pipe.enable_model_cpu_offload()
            # C-prompt: the manifest's text prompt (+ negative) drives conditioning.
            # i2v is image-conditioned, so an empty prompt is still valid.
            result = pipe(
                image=load_image(start_image),
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=n_frames,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                **call_extra,
            )
        else:
            # --- t2v (start_image is None) ---
            pipe = WanPipeline.from_pretrained(
                model_dir, transformer=transformer, vae=vae,
                torch_dtype=compute_dtype)
            pipe.enable_model_cpu_offload()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=n_frames,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                **call_extra,
            )

        # Cooperative cancel — AFTER render: if the callback interrupted the denoise
        # loop (pipe._interrupt), the pipeline still returns partial frames. Abort
        # here, BEFORE assembling/writing, so no clip lands at the addressed path.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled mid-denoise (interrupted)",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # diffusers video pipelines return frames as result.frames[0] — a list of
        # PIL.Image, one per output frame.
        frames = result.frames[0]
        actual_frames = len(frames)

        frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i, fr in enumerate(frames):
            fr.save(os.path.join(frame_dir, f"frame_{i:05d}.png"), "PNG")

        # Same atomic ffmpeg assembly + promotion as the synthetic runner.
        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _assemble_mp4(frame_dir, tmp_mp4, fps)
        if not ok:
            return Err(StageError(
                ErrorCode.ASSEMBLY_FAILED,
                f"ffmpeg mux failed: {stderr_tail}",
                (("content_hash", content_hash), ("frames", str(actual_frames))),
            ))

        os.replace(tmp_mp4, clip_path)        # atomic promotion of the clip
        tmp_mp4 = None

        atomic_write_text(
            os.path.join(out_dir, _MANIFEST_NAME),
            json.dumps(render_manifest_to_dict(manifest), indent=2, sort_keys=True))
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(_provenance_dict(manifest), indent=2, sort_keys=True))
    except Exception as exc:  # inference/IO failure rides back as data (INV-3)
        name = type(exc).__name__
        is_oom = "OutOfMemory" in name or "out of memory" in str(exc).lower()
        return Err(StageError(
            ErrorCode.OOM if is_oom else ErrorCode.IO_ERROR,
            f"wan i2v {'ran out of VRAM' if is_oom else 'inference failed'}: {exc}",
            (("content_hash", content_hash), ("model_id", manifest.model_id)),
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
        path=clip_path, content_hash=content_hash, frames=actual_frames,
        width=width, height=height, duration_s=actual_frames / float(fps),
        resumed=False))
