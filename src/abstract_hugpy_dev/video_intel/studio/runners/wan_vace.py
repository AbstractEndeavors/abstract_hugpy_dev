"""REAL Wan VACE v2v runner (B-3) — the studio's FIRST weight-backed ENHANCEMENT
executor: it restyles/enhances an existing clip (e.g. a movie-tier output) via
diffusers' unified VACE control path, IMPORT-SAFE and GRACEFULLY-DEGRADING now.

    run_wan_vace(manifest, out_root, start_image=None, should_cancel=None)
        -> Result[Artifact, StageError]

It mirrors ``run_wan_i2v`` exactly — same signature, same content-addressed atomic
layout (``<out_root>/<content_hash>/clip.mp4`` + ``manifest.json`` +
``provenance.json``), same resume-on-hash (INV-6), same errors-as-data discipline
(INV-3), same bitsandbytes-quantized DiT (operator directive: "utilize
bitsandbytes"), same T1 cooperative-cancel wiring. The pure geometry / weights /
quant / deps helpers are REUSED from ``wan_i2v`` and the ffmpeg-assembly + sidecar
helpers from ``synthetic``, so the on-disk shape is byte-for-byte the same contract.

IMPORT SAFETY (hard requirement): torch / diffusers / transformers /
bitsandbytes are NEVER imported at module top — only lazily INSIDE the runner,
after preflight passes. Importing this module (or the studio package, or the
Flask app) pulls only stdlib + the studio's own light modules (numpy/PIL via the
reused ``synthetic``/``wan_i2v`` helpers), never the heavy GPU stack.

GRACEFUL DEGRADATION + errors-as-data preflight, in THIS order (returns
``Err(StageError(...))`` as DATA, never raises):
  * v2v render carries no / a nonexistent source_video       -> SOURCE_MISSING
  * missing torch/diffusers/transformers/bitsandbytes/accel  -> DEPS_MISSING
  * no CUDA device                                           -> NO_GPU
  * model weights not on disk under the weights root         -> WEIGHTS_MISSING
Only genuine programmer error (a non-RenderManifest) raises.

The SOURCE_MISSING check runs FIRST — a v2v render is DEFINED by the clip it
enhances, so a source-less request is a SPEC error that is malformed on ANY box
(GPU or not). Checking it before deps/GPU/weights means a source-less v2v is
reported as SOURCE_MISSING even on this GPU-less / bitsandbytes-less dev VM,
rather than masked by the box's DEPS_MISSING. (On this dev box a v2v render WITH a
real source degrades to DEPS_MISSING — bitsandbytes is absent — which is the
intended dev behavior: there is NO synthetic v2v stand-in, because enhancing a
real clip has no meaningful no-model equivalent; the graceful Err IS the answer.)

REAL PATH (runs only when preflight passes, i.e. on the 4x3090 box): loads the
diffusers ``WanVACEPipeline`` with a bitsandbytes-quantized
``WanVACETransformer3DModel`` (int8/nf4 per precision) + fp32 ``AutoencoderKLWan``
VAE, decodes the FULL source clip to control frames (ffmpeg, house style; resampled
to the manifest's frame count), runs VACE control (``video=`` conditioning) at the
manifest's resolution / frame-count / seed / sampler with the prompt/negative, and
ffmpeg-assembles the result into the same atomic content-addressed clip.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Callable

from ..artifacts import Artifact
from ..enums import Precision
from ..errors import Err, ErrorCode, Ok, Result, StageError
from ..manifest import render_manifest_to_dict
from ..registry import MODEL_REGISTRY
from ..schemas import RenderManifest
from ..storage import atomic_write_text
# Reuse the synthetic runner's atomic/content-addressed plumbing so the VACE clip
# lands in the IDENTICAL on-disk layout (numpy/PIL house deps, NOT the heavy
# torch/diffusers stack — that stays lazy inside run_wan_vace).
from .synthetic import (
    _CLIP_NAME,
    _MANIFEST_NAME,
    _PROVENANCE_NAME,
    _assemble_mp4,
    _provenance_dict,
)
# Reuse the Wan i2v runner's PURE helpers (no heavy deps): weights-root resolution,
# the 4k+1 temporal-cadence geometry snap, the precision->bitsandbytes quant map,
# and the find_spec-based dep probe. DRY — both Wan runners resolve weights /
# geometry / quant identically.
from .wan_i2v import (
    _REQUIRED_DEPS,
    _bnb_config,
    _frame_to_pil,
    _local_model_dir,
    _max_vram_gb,
    _missing_deps,
    _should_place_whole_on_gpu,
    _wan_geometry,
    _weights_root,
)


# --------------------------------------------------------------------------- #
# Source-clip resolution (pure, no heavy deps) — the v2v INPUT
# --------------------------------------------------------------------------- #
def _resolve_source(manifest: RenderManifest) -> str | None:
    """The absolute path of the clip this v2v render enhances, or None if the
    manifest carries none. ``source_video`` is part of the content_hash (B-2), so a
    v2v render is deterministically keyed on the clip it restyles."""
    src = getattr(manifest, "source_video", "") or ""
    return src or None


# --------------------------------------------------------------------------- #
# Preflight — errors as data (returns a StageError to raise-as-Err, or None)
# --------------------------------------------------------------------------- #
def _preflight(manifest: RenderManifest) -> StageError | None:
    """Gate the real path. Returns a ``StageError`` (the caller wraps it in ``Err``)
    when this box can't run Wan VACE v2v, or None when everything is present.

    ORDER: source -> deps -> GPU -> weights. The SOURCE check is FIRST because a
    v2v render with no source clip is a SPEC error (malformed on any box), and it
    must surface as SOURCE_MISSING rather than be masked by a GPU-less box's
    DEPS_MISSING. The deps/GPU/weights checks then mirror ``wan_i2v._preflight``."""
    # --- SPEC: a v2v render is DEFINED by the clip it enhances (source first) ---
    source = _resolve_source(manifest)
    if source is None:
        return StageError(
            ErrorCode.SOURCE_MISSING,
            "v2v (VACE) render carries no source_video — a video-to-video "
            "enhancement is defined by the clip it restyles; supply source_video",
            (("model_id", manifest.model_id), ("capability", manifest.capability.value)),
        )
    if not os.path.isfile(source):
        return StageError(
            ErrorCode.SOURCE_MISSING,
            f"v2v (VACE) source_video not found on disk: {source}",
            (("source_video", source), ("model_id", manifest.model_id)),
        )

    # --- ENV: deps -> GPU -> weights (identical discipline to wan_i2v) ---
    missing = _missing_deps()
    if missing:
        return StageError(
            ErrorCode.DEPS_MISSING,
            "Wan VACE v2v needs GPU inference deps that are not installed: "
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
            "no CUDA device available; Wan VACE v2v requires a CUDA GPU (the 4x3090 "
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
            "VACE weights against",
            (("model_id", manifest.model_id),),
        )

    model_dir = _local_model_dir(weights_root, cfg.weight_uri)
    if not (os.path.isdir(model_dir)
            and os.path.isfile(os.path.join(model_dir, "model_index.json"))):
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            f"Wan VACE weights not found on disk at {model_dir}; download with "
            f"`huggingface-cli download {cfg.weight_uri} --local-dir {model_dir}`",
            (("model_dir", model_dir), ("weight_uri", cfg.weight_uri)),
        )
    return None


# --------------------------------------------------------------------------- #
# Full-clip control-frame decode (box-only; ffmpeg house style) — the VACE input
# --------------------------------------------------------------------------- #
def _read_control_frames(
    source_video: str, width: int, height: int, n_frames: int, frame_dir: str
) -> "tuple[list | None, str]":
    """Decode ``source_video``'s frames to PNGs (ffmpeg, mirroring
    ``synthetic._assemble_mp4`` / ``_extract_last_frame``: shutil.which, PIPE,
    returncode check — never raises on a plain ffmpeg failure), load them as RGB
    PIL images resized to ``(width, height)``, and RESAMPLE by nearest index to
    EXACTLY ``n_frames``.

    VACE's control video length must equal ``num_frames``, and ``num_frames`` is a
    pure function of the manifest (``_wan_geometry``, the 4k+1 snap) computed BEFORE
    any decode — so resume-on-hash stays valid regardless of the source clip's own
    frame count (unlike i2v, VACE consumes the WHOLE clip, not just its last frame).
    Returns ``(frames | None, stderr_tail)``; None signals an errors-as-data failure.
    """
    from PIL import Image  # house dep; lazy to honor the module's import discipline

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-i", source_video,
        "-vsync", "0",
        os.path.join(frame_dir, "src_%05d.png"),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None, (result.stderr or "")[-500:]

    names = sorted(
        n for n in os.listdir(frame_dir)
        if n.startswith("src_") and n.endswith(".png"))
    if not names:
        return None, "ffmpeg decoded no frames from source_video"

    imgs: list = []
    for nm in names:
        with Image.open(os.path.join(frame_dir, nm)) as im:
            imgs.append(im.convert("RGB").resize((width, height), Image.LANCZOS))

    total = len(imgs)
    denom = max(1, n_frames - 1)
    picked = [imgs[min(total - 1, round(i * (total - 1) / denom))]
              for i in range(n_frames)]
    return picked, ""


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_wan_vace(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a Wan VACE v2v clip for ``manifest`` under ``out_root``.

    Returns ``Ok(Artifact)`` on a real render (on the box), or ``Err(StageError)``
    on any expected failure — including the preflight failures that make this a
    graceful no-op on a source-less / GPU-less / weight-less box (SOURCE_MISSING /
    DEPS_MISSING / NO_GPU / WEIGHTS_MISSING). Only a genuine programmer error (a
    non-RenderManifest) raises.

    ``start_image`` is part of the shared runner contract (i2v conditioning still)
    but is UNUSED by VACE v2v — the conditioning here is the FULL source clip
    (``manifest.source_video``), decoded to a control video. It is accepted (and
    ignored) so the (framework, task) dispatch signature stays uniform.

    ``should_cancel`` is the OPTIONAL cooperative-cancel probe (T1): a zero-arg
    callable polled at the natural checkpoints (before load, between load and
    render, after render) and — during denoise — wired into the pipeline via
    diffusers' ``callback_on_step_end`` (the callback sets ``pipe._interrupt=True``
    so the loop breaks at the next step boundary). A cancel at any checked point
    returns ``Err(StageError(CANCELLED, ...))`` BEFORE any clip is written. TRUE
    mid-denoise interruption is BOX-ONLY — this GPU-less VM short-circuits at
    preflight, so the callback path only ever executes on the real box. None
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
    # Order: source (spec) -> deps -> GPU -> weights (see _preflight).
    pf = _preflight(manifest)
    if pf is not None:
        return Err(pf)

    # ----------------------------------------------------------------------- #
    # REAL PATH — only reached on a box with a source clip + deps + CUDA + weights.
    # Never executes on the dev VM (preflight short-circuits above). Written
    # complete enough to run once the 4x3090 box is live.
    # ----------------------------------------------------------------------- #
    import torch
    from diffusers import (
        AutoencoderKLWan,
        BitsAndBytesConfig,
        UniPCMultistepScheduler,
        WanVACEPipeline,
        WanVACETransformer3DModel,
    )

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    model_dir = _local_model_dir(_weights_root(manifest), cfg.weight_uri)
    source_video = _resolve_source(manifest)      # preflight proved it exists
    compute_dtype = torch.bfloat16
    quant_config = _bnb_config(manifest.precision, BitsAndBytesConfig, torch)
    seed = manifest.seeds.global_seed
    steps = manifest.sampler.steps
    cfg_scale = manifest.sampler.cfg
    # C-prompt: text conditioning from the manifest (part of its content_hash). An
    # empty prompt is valid; an empty negative maps to None so the pipeline uses its
    # own default rather than an explicit "" negative.
    prompt = manifest.prompt
    negative_prompt = manifest.negative_prompt or None

    # PLACEMENT + SHIFT (shared decision with wan_i2v). Put a sub-envelope UNQUANTIZED
    # model wholly on the GPU; a bnb-quantized (INT8/FP8) VACE precision offloads (never
    # .to() a quantized pipeline). flow_shift is the manifest's recorded scheduler shift.
    model_gb = cfg.vram.as_map().get(manifest.precision)
    place_whole = _should_place_whole_on_gpu(
        manifest.precision, model_gb, _max_vram_gb(manifest))
    flow_shift = manifest.sampler.shift

    # Cooperative mid-render cancel wiring (T1). diffusers 0.39's
    # WanVACEPipeline.__call__ supports `callback_on_step_end`; the callback sets
    # `pipe._interrupt=True` so the denoise loop breaks at the next step boundary.
    # We ALSO re-check should_cancel() around the call. BOX-ONLY (preflight
    # short-circuits the GPU-less VM above).
    def _cancel_step_cb(pipe_ref, step_index, timestep, cb_kwargs):
        if should_cancel is not None and should_cancel():
            pipe_ref._interrupt = True   # diffusers checks self.interrupt each step
        return cb_kwargs

    call_extra: dict = {}
    if should_cancel is not None:
        call_extra["callback_on_step_end"] = _cancel_step_cb

    src_frame_dir = None
    out_frame_dir = None
    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Cooperative cancel — BEFORE load (no weights touched yet if we bail).
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled before wan vace load",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # Decode the FULL source clip to VACE control frames BEFORE loading multi-GB
        # weights, so a bad source fails fast (errors-as-data). source_video is in the
        # manifest (content_hash), so this is deterministic + resume-safe.
        src_frame_dir = tempfile.mkdtemp(prefix=".srcframes-", dir=out_dir)
        control_frames, stderr_tail = _read_control_frames(
            source_video, width, height, n_frames, src_frame_dir)
        if control_frames is None:
            return Err(StageError(
                ErrorCode.IO_ERROR,
                f"could not decode control frames from source_video: {stderr_tail}",
                (("source_video", source_video),)))

        # bitsandbytes-quantized VACE DiT transformer (int8 / nf4 per precision).
        tf_kwargs = {"subfolder": "transformer", "torch_dtype": compute_dtype}
        if quant_config is not None:
            tf_kwargs["quantization_config"] = quant_config
        transformer = WanVACETransformer3DModel.from_pretrained(model_dir, **tf_kwargs)
        # Wan's VAE is numerically sensitive; the diffusers Wan reference loads it in
        # fp32 (small relative to the DiT, so this is affordable).
        vae = AutoencoderKLWan.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float32)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Cooperative cancel — BETWEEN load and render (weights loaded, nothing
        # rendered/written yet). Per-step interruption is handled by the callback.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled after wan vace load, before render",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        pipe = WanVACEPipeline.from_pretrained(
            model_dir, transformer=transformer, vae=vae, torch_dtype=compute_dtype)
        # SHIFT: wire the manifest's flow-match/UniPC shift into the scheduler so the
        # denoise uses exactly the recorded value (INV-1). None leaves the default.
        if flow_shift is not None:
            try:
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=flow_shift)
            except Exception:
                pass  # diffusers build without flow_shift on UniPC — keep the default
        # PLACEMENT: whole pipeline to GPU when it fits unquantized, else offload. A
        # bnb-quantized (INT8/FP8) precision ALWAYS offloads (never .to() a quantized
        # pipeline — _should_place_whole_on_gpu returns False for INT8/FP8).
        if place_whole:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()

        # VACE unified control: `video=` conditions the generation on the source
        # clip's frames (no mask / reference_images => pure v2v restyle/enhance).
        # output_type="pil" so result.frames[0] is a list of PIL.Image for the save
        # loop below (the VACE pipeline otherwise defaults to numpy output).
        result = pipe(
            video=control_frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="pil",
            **call_extra,
        )

        # Cooperative cancel — AFTER render: if the callback interrupted the denoise
        # loop (pipe._interrupt), the pipeline still returns partial frames. Abort
        # here, BEFORE assembling/writing, so no clip lands at the addressed path.
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, "cancelled mid-denoise (interrupted)",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        # diffusers video pipelines return frames as result.frames[0]. We request
        # output_type="pil" but the per-frame type varies by pipeline/version
        # (wan_i2v got ndarray on ae 2026-07-07 and PIL-only .save() failed after
        # a full denoise) — normalize per-frame, same belt as wan_i2v.
        frames = result.frames[0]
        actual_frames = len(frames)

        out_frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i, fr in enumerate(frames):
            _frame_to_pil(fr).save(
                os.path.join(out_frame_dir, f"frame_{i:05d}.png"), "PNG")

        # Same atomic ffmpeg assembly + promotion as the synthetic / i2v runners.
        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _assemble_mp4(out_frame_dir, tmp_mp4, fps)
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
            f"wan vace v2v {'ran out of VRAM' if is_oom else 'inference failed'}: {exc}",
            (("content_hash", content_hash), ("model_id", manifest.model_id)),
        ))
    finally:
        if tmp_mp4 is not None and os.path.isfile(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except OSError:
                pass
        for d in (src_frame_dir, out_frame_dir):
            if d is not None and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    return Ok(Artifact(
        path=clip_path, content_hash=content_hash, frames=actual_frames,
        width=width, height=height, duration_s=actual_frames / float(fps),
        resumed=False))
