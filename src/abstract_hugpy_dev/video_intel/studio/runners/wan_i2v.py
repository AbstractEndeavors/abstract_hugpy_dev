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
import logging
import os
import shutil
import sys
import tempfile
from typing import Callable

from ..artifacts import Artifact
from ..enums import Precision, Task
from ..errors import Err, ErrorCode, Ok, Result, StageError
from ..manifest import render_manifest_to_dict
from ..registry import MODEL_REGISTRY
from ..schemas import RenderManifest
from ..storage import atomic_write_text
# Reuse the synthetic runner's atomic/content-addressed plumbing so the Wan clip
# lands in the IDENTICAL on-disk layout. These pull numpy/PIL (already house
# deps, present everywhere) — NOT the heavy torch/diffusers stack, which stays
# lazy inside run_wan_i2v. ``_extract_last_frame`` + ``_SOURCE_LASTFRAME_NAME`` are
# the B2 "extend the movie" helpers, shared so Wan extracts the source clip's last
# frame byte-identically to the synthetic prover.
from .synthetic import (
    _CLIP_NAME,
    _MANIFEST_NAME,
    _PROVENANCE_NAME,
    _SOURCE_LASTFRAME_NAME,
    _assemble_mp4,
    _extract_last_frame,
    _geometry,
    _provenance_dict,
)

logger = logging.getLogger(__name__)

# Python deps the REAL inference path needs. Preflight reports any that are
# absent as DEPS_MISSING data (never an ImportError at module import).
# ``ftfy`` is required because the Wan pipelines' prompt-clean path imports it (the
# i2v/t2v denoise calls ``ftfy.fix_text`` on the prompt); a box missing it OOM-free
# used to surface it as a mid-load IO_ERROR AFTER loading ~14GB of weights (live
# 2026-07-07) — listing it here makes it an honest DEPS_MISSING at PREFLIGHT instead.
_REQUIRED_DEPS = (
    "torch", "diffusers", "transformers", "bitsandbytes", "accelerate", "ftfy")


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


# --------------------------------------------------------------------------- #
# HOT weights root (item 5) — a per-box NVMe copy that loads faster than the shared
# /mnt/llm_storage mount. Box-local ONLY; NEVER a canonical input.
# --------------------------------------------------------------------------- #
_HOT_WEIGHTS_ROOT_ENV = "STUDIO_WEIGHTS_HOT_ROOT"


def _hot_weights_root() -> str | None:
    """A per-box NVMe hot-copy weights root, read from the LOCAL PROCESS ENV ONLY
    (``STUDIO_WEIGHTS_HOT_ROOT``) — deliberately NOT from the manifest env_snapshot,
    which is captured on central and must not dictate a box-local path. None if unset
    or empty.

    The hot copy is a faster LOAD SOURCE only: it is never written into the manifest /
    env_snapshot, so it CANNOT change a clip's content_hash (weights come from the
    same bytes wherever they load from). Central builds the manifest without ever
    seeing this var; the worker resolves it here at render time."""
    root = os.environ.get(_HOT_WEIGHTS_ROOT_ENV)
    return root or None


def _resolve_model_dir(manifest: RenderManifest, weight_uri: str) -> tuple[str | None, str]:
    """Resolve the on-disk model dir for ``weight_uri`` PLUS a tag for WHICH root
    served it. Order (box-local NVMe hot copy first, then the shared/snapshot root):

      1. ``STUDIO_WEIGHTS_HOT_ROOT`` set AND
         ``<hot>/<org>/<name>/model_index.json`` present -> (``<hot_dir>``, "hot");
      2. else -> (``<shared_dir>`` | None, "shared") — ``_local_model_dir`` over the
         shared weights root from the manifest snapshot (or process env), UNCHANGED;
         None when no shared root is configured.

    The hot presence gate is ``model_index.json`` (the same completeness gate the
    shared preflight uses), so a partial / in-flight hot copy transparently falls back
    to the shared store rather than loading half a model."""
    hot = _hot_weights_root()
    if hot:
        hot_dir = _local_model_dir(hot, weight_uri)
        if os.path.isfile(os.path.join(hot_dir, "model_index.json")):
            return hot_dir, "hot"
    shared_root = _weights_root(manifest)
    if not shared_root:
        return None, "shared"
    return _local_model_dir(shared_root, weight_uri), "shared"


def _weights_missing_msg(weight_uri: str, hot: str | None, shared_root: str | None) -> str:
    """A WEIGHTS_MISSING message that names BOTH roots that were tried (item 5), so a
    box operator can see whether the hot NVMe copy, the shared mount, or both are
    absent."""
    tried: list[str] = []
    if hot:
        tried.append("hot NVMe " + _local_model_dir(hot, weight_uri))
    if shared_root:
        tried.append("shared " + _local_model_dir(shared_root, weight_uri))
    where = "; ".join(tried) if tried else "no weights root configured"
    dl_root = shared_root or hot or "<weights_root>"
    return (f"weights not found on disk for {weight_uri} (looked in: {where}); "
            f"download with `huggingface-cli download {weight_uri} "
            f"--local-dir {_local_model_dir(dl_root, weight_uri)}`")


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


def _frame_to_pil(frame):
    """Normalize ONE pipeline output frame to a PIL.Image.

    Diffusers video pipelines vary in what ``result.frames[0]`` yields per frame
    even under ``output_type="pil"`` (proven on ae 2026-07-07: WanPipeline handed
    back numpy and the PIL-only save failed after a full denoise). Handles: PIL
    passthrough, torch-like tensors (``.cpu()`` duck-typed — torch never imported
    here), numpy HWC float [0,1] / uint8, CHW transposed, single-channel. Raises
    TypeError on anything else (rides back as errors-as-data)."""
    import numpy as np
    from PIL import Image
    if isinstance(frame, Image.Image):
        return frame
    if hasattr(frame, "cpu"):                      # torch tensor, duck-typed
        frame = frame.cpu().numpy()
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = (np.clip(arr.astype("float32"), 0.0, 1.0)
                   * 255.0).round().astype(np.uint8)
        if (arr.ndim == 3 and arr.shape[0] in (1, 3, 4)
                and arr.shape[-1] not in (1, 3, 4)):
            arr = np.transpose(arr, (1, 2, 0))     # CHW -> HWC
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Image.fromarray(arr)
    raise TypeError(f"unsupported pipeline frame type: {type(frame).__name__}")


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
              "bitsandbytes accelerate ftfy",
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

    # WEIGHTS root resolution honors the box-local HOT NVMe copy first (item 5), then
    # the shared/snapshot root — see _resolve_model_dir. Neither configured is the
    # "no weights root" error; configured-but-model-absent names BOTH roots tried.
    hot = _hot_weights_root()
    shared_root = _weights_root(manifest)
    if not hot and not shared_root:
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "no weights root set — neither STUDIO_WEIGHTS_HOT_ROOT (box-local NVMe) "
            "nor STUDIO_WEIGHTS_ROOT is configured to resolve the Wan weights against",
            (("model_id", manifest.model_id),),
        )

    model_dir, _root_used = _resolve_model_dir(manifest, cfg.weight_uri)
    if not model_dir or not (os.path.isdir(model_dir)
            and os.path.isfile(os.path.join(model_dir, "model_index.json"))):
        return StageError(
            ErrorCode.WEIGHTS_MISSING,
            "Wan " + _weights_missing_msg(cfg.weight_uri, hot, shared_root),
            (("weight_uri", cfg.weight_uri),
             ("hot_root", hot or ""), ("shared_root", shared_root or "")),
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
# GPU PLACEMENT decision (pure, no heavy deps — unit-tested without a GPU)
# --------------------------------------------------------------------------- #
# LEGACY: the flat headroom this decision used until 2026-07-27. Kept ONLY as the
# fallback for a model with no measured footprint (see _placement_need_gib), and as
# the record of why it existed. Its own incident note:
#
#   ae 3090, 2026-07-07: wan2.1-t2v-1.3b "8.2GB" placed whole-on-GPU actually
#   allocated 19.6GB and OOM'd at 832x480x29f next to comfy's 512MB.
#
# That 19.6 GiB was a MEASUREMENT and was never an argument against whole-GPU
# placement: it fits a 23.56 GiB card with 3.96 GiB spare. The bug was arithmetic:
# the registry envelope (8.2) counts the DiT ONLY, while the real resident is
# DiT + UMT5-XXL + VAE + activations. Comparing a DiT-only number against a flat
# 16 GB fudge made "8.2 + 16 = 24.2 > 23.56" — a refusal by 0.64 GB, produced by
# adding two numbers that measure different things.
#
# ⚠ THE DERIVED MODEL BELOW DOES NOT REPRODUCE 19.6 — it returns 17.897 GiB for that
# exact tuple. See the "UNCLOSED RESIDUAL" note on _placement_need_gib. Do not read
# the 19.6 here as something this file computes; it is a number a card once reported.
_PLACEMENT_MARGIN_GB = 16.0

# ── MEASURED COMPONENT FOOTPRINTS ───────────────────────────────────────────
# Parameter counts read from the safetensors HEADERS under
# /mnt/llm_storage/video_intel/studio/weights/Wan-AI on 2026-07-27 (sum of every
# tensor's shape product, de-duplicated across shards) and cross-checked against each
# transformer/config.json. These are facts about the weights, not estimates.
#
#   wan2.1-t2v-1.3b       DiT  1,418,996,800  = 1.4190e9  (dim 1536, ffn 8960, L30)
#   wan2.1-vace-1.3b      DiT  2,153,972,032  = 2.1540e9  (1.419e9 F32 base
#                                                + 0.735e9 BF16 / 15 VACE blocks)
#   wan2.1-i2v-14b-720p   DiT 16,395,083,584  = 16.3951e9 (dim 5120, ffn 13824, L40)
#   wan2.1-vace-14b       DiT 17,337,592,896  = 17.3376e9
#   wan2.2-t2v-a14b       DiT 14,288,491,584  = 14.2885e9 x2 (transformer + _2)
#   wan2.2-i2v-a14b       DiT 14,288,901,184  = 14.2889e9 x2 (transformer + _2)
#
# ⚠ CORRECTED 2026-07-27 (adversarial review): wan2.1-vace-14b was carried here as
# 16.3951e9 with the gloss "same geometry as the i2v 14B". The DIM geometry is indeed
# identical (both 5120/13824/40 layers, so _WAN_U below is shared), but the PARAMETER
# COUNTS are not, because the two models bolt different things onto the same trunk.
# Both start from the plain 14B T2V trunk of 14,288,491,584 params, then:
#   i2v-14b   + 2,106,592,000  image cross-attention (config image_dim 1280,
#                              in_channels 36)            -> 16,395,083,584
#   vace-14b  + 3,049,101,312  BF16 VACE control blocks
#                              (config vace_layers = 8)   -> 17,337,592,896
# The old number under-counted vace-14b by 0.942e9 params = 1.76 GiB at bf16.
#
# Shared sidecars, every Wan pipeline:
#   UMT5-XXL text encoder 5,680,910,336 = 5.6809e9 — loaded bf16 => 10.582 GiB. THE
#                                     LARGEST SINGLE RESIDENT, larger than an nf4 14B
#                                     DiT, and it has no registry row anywhere.
#   AutoencoderKLWan        126,892,531 = 0.1269e9 — forced fp32 => 0.473 GiB
#   CLIP ViT-H (i2v only)   632,076,800 = 0.6321e9 — bf16 => 1.177 GiB
_UMT5_PARAMS = 5.6809e9
_VAE_PARAMS = 0.1269e9
_CLIP_PARAMS = 0.6321e9

# params, has_image_encoder, extra_dit_params (MoE second expert, NEVER quantized)
#
# ⚠ has_image_encoder IS A DISK FACT, NOT A CAPABILITY FACT (corrected 2026-07-27).
# "It is an i2v model" does NOT imply a CLIP image encoder: wan2.2-i2v-a14b's
# model_index.json declares  "image_encoder": [null, null]  and there is no
# image_encoder/ directory under its weights at all — Wan 2.2 conditions i2v through
# the DiT's own 36 in_channels instead. Carrying True here added a phantom +1.177 GiB
# to every wan2.2-i2v-a14b estimate. Verified per row against model_index.json +
# the on-disk directory listing; only wan2.1-i2v-14b-720p actually has one
# (CLIPVisionModelWithProjection, 632,076,800 params).
_WAN_FOOTPRINTS: dict[str, tuple[float, bool, float]] = {
    "wan2.1-t2v-1.3b":     (1.4190e9, False, 0.0),
    "wan2.1-vace-1.3b":    (2.1540e9, False, 0.0),
    "wan2.1-i2v-14b-720p": (16.3951e9, True, 0.0),
    "wan2.1-vace-14b":     (17.3376e9, False, 0.0),
    "wan2.2-t2v-a14b":     (14.2885e9, False, 14.2885e9),
    "wan2.2-i2v-a14b":     (14.2889e9, False, 14.2889e9),
}

# Bytes per parameter by precision.
#
#   bf16/fp16  2.0     — the runner computes in torch.bfloat16 either way.
#   fp32       2.0     — NOT 4.0. ``compute_dtype`` is HARDCODED to torch.bfloat16 in
#                        run_wan_i2v and passed as ``torch_dtype`` to both
#                        WanTransformer3DModel.from_pretrained and the pipeline, so an
#                        fp32 manifest would still land in VRAM at 2 bytes/param. (No
#                        registry row declares FP32 today, so this is unreachable — but
#                        an unreachable row that lies is still a lie: it read as
#                        "fp32 costs double", which is what a reader would carry away.)
#   int8       1.003   — CORRECTED from 1.06 on 2026-07-27. A bitsandbytes
#                        Linear8bitLt stores CB (int8, 1 byte/param) plus SCB (one
#                        float32 PER OUTPUT ROW), i.e. 1 + 4/in_features bytes/param
#                        for that layer — 1.0026 at in_features 1536, 1.00029 at
#                        13824. Weighting every 2-D weight in each DiT by its own
#                        in_features and charging the residual non-Linear params
#                        (0.10% of the 1.3B, 0.04% of the 14B: norms, the 3-D patch
#                        conv, embeddings — bnb leaves these at compute dtype) at bf16
#                        gives a whole-DiT effective 1.00293 (1.3B) / 1.00110 (14B).
#                        1.003 is the conservative end of that range.
#                        The old gloss — "8 bits + bnb's fp16 outlier bookkeeping" —
#                        was also WRONG about the mechanism, not just the number:
#                        LLM.int8()'s outlier columns are extracted from the
#                        ACTIVATION at matmul time, not stored alongside the weight,
#                        so they belong in the workspace term below, never here.
#                        Consequence of the old 1.06: the 14B DiT was over-priced by
#                        0.87 GiB at int8 (16.185 vs 15.315).
#   fp8        0.5625  — nf4: 4 bits + an fp32 absmax per 64-element block
#                        (double-quant is not enabled here) = 0.5 + 0.0625.
_BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp32": 2.0, "int8": 1.003, "fp8": 0.5625}

# ── ACTIVATION WORKSPACE ────────────────────────────────────────────────────
# Calibrated on ae from two measured points at 832x480 for the 1.3B:
#   45f -> 18,720 latent tokens -> 4.6 GiB
#   81f -> 32,760 latent tokens -> 5.5 GiB
# i.e. +0.90 GiB for +14,040 tokens.
#
# ⚠ THE INTERCEPT IS DERIVED FROM THOSE POINTS, NOT TYPED IN (fixed 2026-07-27).
# The shipped constants were slope 6.41e-5 with intercept 0.76 — the slope is the
# two-point fit, but 0.76 is NOT the intercept that fit produces:
#     slope     = 0.90 / 14,040        = 6.410256e-5 GiB/token
#     intercept = 4.6 - slope*18,720   = 3.400 GiB   (and 5.5 - slope*32,760 = 3.400)
# The line the code drew therefore missed BOTH of its own calibration points by
# 2.64 GiB, and since the intercept is a constant, EVERY estimate in this module was
# 2.64 GiB light. That is the unsafe direction: it is what let
# wan2.1-i2v-14b-720p @fp8 832x480x29f read as 23.476 (a "fit" by 0.08 GiB) when the
# same calibration says 26.116 — a MISS by 2.56 — and with the INT8/FP8 early return
# retired that wrong verdict is a bare pipe.to("cuda") with no offload fallback.
# Computing the line from the two points below makes that class of drift impossible:
# there is no second place for the numbers to disagree.
#
# A 3.4 GiB constant term is large but not implausible for this box: the 2026-07-08
# allocator note in _prime_cuda_allocator records ~2.71 GiB RESERVED-BUT-UNALLOCATED
# on a failing render, and a CUDA context plus cuBLAS/cuDNN workspaces account for
# several hundred MB more. The intercept is where that lives.
#
# Cross-check on the SLOPE against structure: 10 live inner-dim buffers + 2 ffn
# buffers at bf16 predicts (10*1536 + 2*8960)*2 = 65.0 KiB/token vs the measured 68.8
# — within 6%, which is why extrapolating the slope to the 14B by its own
# (inner_dim, ffn_dim) is sound. (The structural cross-check speaks ONLY to the
# slope; it says nothing about the intercept, which is why the intercept has to come
# from the measured points.)
_WS_CAL_LO = (18_720, 4.6)   # 832x480x45f on ae
_WS_CAL_HI = (32_760, 5.5)   # 832x480x81f on ae
_WS_SLOPE_GIB_PER_TOKEN = ((_WS_CAL_HI[1] - _WS_CAL_LO[1])
                           / (_WS_CAL_HI[0] - _WS_CAL_LO[0]))     # 6.410256e-5
_WS_INTERCEPT_GIB = _WS_CAL_LO[1] - _WS_SLOPE_GIB_PER_TOKEN * _WS_CAL_LO[0]   # 3.400

# VAE decode workspace. ⚠ UNMEASURED ASSUMPTION — labelled as one (2026-07-27).
# It shipped as ``_WS_INTERCEPT_GIB + 0.60``; the 0.60 has no measurement anywhere in
# this tree, and coupling it to the DENOISE intercept was a category error besides —
# it made a decode estimate move whenever the denoise line was recalibrated. Pinned
# here to the numeric value it has always had (0.76 + 0.60 = 1.36) so this correction
# changes nothing about decode, and named so the assumption is visible.
#
# It is currently INERT and provably so: _placement_need_gib takes
# max(denoise_ws, decode_ws), and the denoise term is never below the intercept
# (3.400 GiB at one single token), so 1.36 can never govern at any geometry. Treat it
# as a placeholder for a measurement nobody has taken rather than as a live bound —
# and note that it is only defensible AT ALL because _place_pipe engages
# vae.enable_tiling() on both branches (see that docstring): untiled,
# AutoencoderKLWan._decode is a resolution-squared fp32 spike, not a constant.
_DECODE_WS_GIB = 1.36

_WS_REF_U = 33_280          # 10*1536 + 2*8960 for the 1.3B
_WAN_U = {                  # 10*inner_dim + 2*ffn_dim
    "wan2.1-t2v-1.3b": 33_280, "wan2.1-vace-1.3b": 33_280,
    "wan2.1-i2v-14b-720p": 78_848, "wan2.1-vace-14b": 78_848,
    "wan2.2-t2v-a14b": 78_848, "wan2.2-i2v-a14b": 78_848,
}


def _latent_tokens(width: int, height: int, n_frames: int) -> int:
    """Wan latent token count: patch_size [1,2,2] over an 8x spatially / 4:1
    temporally compressed latent = 16 px per token, 4 frames per latent frame."""
    return max(1, (width // 16) * (height // 16) * (((max(1, n_frames) - 1) // 4) + 1))


def _placement_need_gib(model_id: str, precision: "Precision", width: int,
                        height: int, n_frames: int) -> float | None:
    """TOTAL VRAM this render needs whole-on-GPU: DiT + sidecars + workspace.

    None when the model has no measured footprint — the caller then falls back to
    the legacy flat-margin test rather than guessing.

    This replaces comparing a DiT-only registry number against a flat 16 GB fudge.
    For wan2.1-t2v-1.3b @fp16 832x480x29f — the exact tuple that lands a clip on ae —
    it returns **17.897 GiB**, decomposed as:

        DiT      1.4190e9 @ bf16   =  2.6431
        UMT5-XXL 5.6809e9 @ bf16   = 10.5815
        VAE      0.1269e9 @ fp32   =  0.4727      static subtotal 13.6973
        denoise workspace, 12,480 latent tokens
                 3.400 + 6.410256e-5*12,480 =  4.2000
        ------------------------------------------------------
        total                                    17.8973   (fits 23.56 by 5.663)

    ⚠ UNCLOSED RESIDUAL — 1.703 GiB, RECORDED RATHER THAN PAPERED OVER (2026-07-27).
    The 2026-07-07 ae incident measured **19.6 GiB** resident for that same tuple, and
    CAPABILITY-VIABILITY-MAP.md repeats it in three places. This function returns
    17.897. Earlier revisions of this docstring asserted the two were the same number;
    they never have been, at any intercept this file has shipped. The honest state:

      * The two figures cannot BOTH be right. The map decomposes 19.6 as
        13.70 static + ~5.90 activations at 12,480 tokens (29f). The calibration this
        module is built on says 4.60 at 18,720 tokens (45f). More activation at FEWER
        tokens is impossible for any monotone-in-tokens model, so at least one of the
        three data points is mislabelled — and none of them has a provenance anywhere
        in this tree beyond the comment that cites it. We cannot tell which from here.
      * The "maybe the incident's GEOMETRY label is wrong" hypothesis was checked and
        DOES NOT HOLD. 1280x720x81f (75,600 tokens) prices at 21.944 GiB, which is
        2.344 off 19.6 — FURTHER away than 832x480x29f's 1.703. (That hypothesis was
        computed against the broken 0.76 intercept, where 720p81f gave 19.31 and did
        look like a match. Correcting the intercept dissolves it.) The 1.3B does pass
        THROUGH 19.6 between 832x480x93f (37,440 tokens -> 19.497) and 832x480x97f
        (39,000 -> 19.597), and again at 1280x720x41f (39,600 -> 19.636); but 93f/97f
        are both above the row's max_frames=81, the registry declares exactly ONE
        resolution for this row (832x480), and 1280x720 is therefore not a geometry it
        can be asked for at all. Those are coincidences of a monotone line crossing a
        value, not an explanation.
      * The residual points the UNSAFE way (we under-price by 1.7 GiB against the one
        directly measured whole-GPU resident this fleet has). It is not currently an
        OOM hazard because the thinnest surviving fit has more headroom than that:
        wan2.1-vace-1.3b @fp16 832x480x81f, the tightest row that still fits, clears
        23.56 by 2.994 GiB. test_placement_need.py pins that inequality, so if a
        future edit ever narrows a fit to less than the unexplained residual, the
        suite fails and says exactly this.

    CLOSING IT needs a measurement, not more arithmetic: torch.cuda.max_memory_-
    allocated() around a whole-GPU 832x480x29f run on ae, split denoise vs decode."""
    fp = _WAN_FOOTPRINTS.get(model_id)
    if fp is None:
        return None
    dit_params, has_image_encoder, extra_dit = fp
    prec = str(getattr(precision, "value", precision)).lower()
    bpp = _BYTES_PER_PARAM.get(prec)
    if bpp is None:
        return None
    gib = 1024.0 ** 3
    need = dit_params * bpp / gib
    # A MoE second expert is loaded UNQUANTIZED at bf16 — quantizing `transformer`
    # does not touch `transformer_2`. This is why the A14B rows do not fit.
    if extra_dit:
        need += extra_dit * 2.0 / gib
    need += _UMT5_PARAMS * 2.0 / gib          # bf16
    need += _VAE_PARAMS * 4.0 / gib           # forced fp32 by the runner
    if has_image_encoder:
        need += _CLIP_PARAMS * 2.0 / gib      # bf16
    u = _WAN_U.get(model_id, _WS_REF_U)
    tokens = _latent_tokens(width, height, n_frames)
    denoise_ws = _WS_INTERCEPT_GIB + _WS_SLOPE_GIB_PER_TOKEN * tokens * (u / _WS_REF_U)
    # VAE decode with tiling enabled is bounded; without tiling it is the spike that
    # most plausibly caused the 07-07 OOM, which is why _place_pipe now engages the
    # memory savers on BOTH branches. See _DECODE_WS_GIB: unmeasured, and currently
    # inert (it can never exceed the denoise term at any geometry).
    return need + max(denoise_ws, _DECODE_WS_GIB)


def _placement_budget_gib(*candidates: float | None) -> float | None:
    """The placement budget = the MINIMUM of every ceiling that applies (2026-07-27).

    ⚠ SEMANTICS, DECIDED AND WRITTEN DOWN. A VRAM budget is an ADMISSION CONSTRAINT,
    not a hint: whoever set it is entitled to have the render stay inside it. A
    physical card is a HARD ceiling. Neither ever licenses exceeding the other, so
    the only sound combination is the min — and because min() can only ever LOWER the
    budget, adopting it cannot turn a previously-safe offload into a new whole-GPU
    placement. It fails toward offload, which is the recoverable direction.

    WHY THIS EXISTS. A reviewer measured, 2026-07-27:
    ``POST /video/studio/i2v {"vram_budget_gb": 6.0, ...}`` routes to INT8 (the
    registry envelope 5.0 fits 6.0), and then placement compared the derived need
    (15.3 GiB at the time) against the CARD (24.0) and answered whole-GPU=True. The
    caller asked for 6 GB and got a 15 GiB placement — the budget governed WHICH
    MODEL was admitted and then stopped governing anything.

    ⚠ HALF OF THAT IS STILL OPEN, AND HERE IS EXACTLY WHERE. This helper closes the
    ceilings the runner can SEE: the live device and the manifest's declared
    ``STUDIO_MAX_VRAM_GB``. It cannot close the reviewer's 6.0, because that number is
    ``CapabilityRequest.vram_budget_gb`` and it is NEVER PLUMBED TO A RUNNER —
    ``video_intel/runners/studio_i2v.py::run_produce_clip`` calls
    ``resolve_studio_env(spec.out_root, master_fps=spec.fps)`` and lets ``max_vram_gb``
    fall to its 24.0 default, and ``RenderManifest`` has no budget field at all. Two
    files this module does not own, so the seam is named rather than cut.

    ⚠ AND WHEN IT IS CUT, DO NOT ROUTE IT THROUGH ``env_snapshot``. ``schemas.py`` puts
    ``env_snapshot`` inside the clip ``content_hash``, and an AUTOFIT budget is sized
    from the worker's MEASURED FREE VRAM (``studio_i2v._resolve_autofit``) — i.e. it
    is not stable for a fixed spec. Hashing it would re-address a clip on every render
    and destroy resume. Thread it as a non-hashed runner argument into this function
    instead; that is what the varargs shape here is for.

    ``None`` candidates are ignored (unknown ≠ zero). All-None returns None, which the
    caller reads as "no budget resolved" and answers offload."""
    known = [float(c) for c in candidates if c is not None]
    return min(known) if known else None


def _max_vram_gb(manifest: RenderManifest) -> float | None:
    """The GPU VRAM budget in GB for the PLACEMENT decision.

    THE LIVE DEVICE AND THE DECLARED CEILING BOTH BIND (2026-07-27) — this returns the
    MIN of them via ``_placement_budget_gib``, not the first one that resolves. The
    device comes from ``_platform.hardware.total_vram_bytes()`` (the same probe pair,
    torch ``mem_get_info`` → ``nvidia-smi``, the worker's own VRAM ceiling uses); the
    declared ceiling from the manifest's captured ``env_snapshot``
    ``STUDIO_MAX_VRAM_GB``, else the live process env. None if nothing resolves, which
    keeps the conservative offload behaviour.

    On today's fleet the min is a no-op in the safe direction (ae: device 23.56 vs
    declared 24.0 -> 23.56; computron: 8.0 vs 24.0 -> 8.0). It starts mattering the
    moment an operator declares a ceiling BELOW the card, which is precisely when
    "the card fits it" must stop being the answer.

    WHY READ THE CARD HERE AND NOT IN ``resolve_studio_env``: the env value rides
    ``env_snapshot``, and ``schemas.py`` puts ``env_snapshot`` INSIDE the clip
    ``content_hash``. Making the SNAPSHOT device-derived would re-address every
    clip the moment two boxes disagree (ae reads 23.56 GB, computron 8.0), so an
    identical spec would re-render instead of resuming. The placement decision is
    not part of the manifest, so reading the real card HERE is free of that.
    Deliberately the same split ``_hot_weights_root()`` uses: process-local truth
    for behaviour, snapshot for addressing.

    ⚠ WHAT THIS FIXES: ``resolve_studio_env`` defaults ``max_vram_gb`` to 24.0 and
    NEITHER CALLER EVER OVERRODE IT, so every placement decision the fleet ever
    made was measured against 24.0 — including on computron, an 8 GiB 4060."""
    device_gb: float | None = None
    try:
        from ...._platform.hardware import total_vram_bytes
        total = total_vram_bytes()
        if total and total > 0:
            device_gb = float(total) / (1024 ** 3)
    except Exception:  # noqa: BLE001 — a probe must never fail a render
        pass
    snap = dict(manifest.env_snapshot)
    raw = snap.get("STUDIO_MAX_VRAM_GB") or os.environ.get("STUDIO_MAX_VRAM_GB")
    declared_gb: float | None = None
    if raw not in (None, ""):
        try:
            declared_gb = float(raw)
        except (TypeError, ValueError):
            declared_gb = None
    # ⚠ A DEAF PROBE MUST FAIL CLOSED (2026-07-27, round-2 review). If the device probe
    # returns nothing we do NOT fall back to the declared ceiling, because the declared
    # value is almost never a declaration: ``resolve_studio_env`` DEFAULTS max_vram_gb to
    # 24.0 and no caller overrides it, so "24.0" overwhelmingly means "nobody said". Under
    # the retired flat margin that fabricated 24.0 was inert (8.2 + 16.0 = 24.2 > 24.0 ->
    # offload). Against derived need it is PERMISSIVE: wan2.1-t2v-1.3b fp16 needs 17.897,
    # which "fits" 24.0 -> a bare ``pipe.to("cuda")`` — on computron's 8 GiB 4060 that is
    # an OOM caused by a probe failure, not by a real budget. Unknown card => offload.
    if device_gb is None:
        return None
    # BOTH bind — see _placement_budget_gib. A malformed / absent declaration is
    # "unknown", not "zero", so it drops out of the min rather than forcing offload.
    return _placement_budget_gib(device_gb, declared_gb)


# --------------------------------------------------------------------------- #
# QUANTIZED-MOVE CAPABILITY GUARD (2026-07-27) — the runtime half of retiring the
# INT8/FP8 blanket rule. PURE + version-string driven, so it is unit-testable on a
# box with no bitsandbytes at all (this dev VM has none).
# --------------------------------------------------------------------------- #
# The exact gates in the INSTALLED diffusers 0.39.0, read out of
# site-packages/diffusers/pipelines/pipeline_utils.py (line numbers re-read from the
# venv 2026-07-27, inside DiffusionPipeline.to's per-module loop):
#
#   :541  if is_loaded_in_8bit_bnb and device is not None
#             and is_bitsandbytes_version("<", "0.48.0"):   logger.warning(...)
#   :559  if is_loaded_in_4bit_bnb and device is not None
#             and is_transformers_version(">", "4.44.0"):   module.to(device=device)
#   :562  if (is_loaded_in_8bit_bnb and device is not None
#             and is_transformers_version(">", "4.58.0")
#             and is_bitsandbytes_version(">=", "0.48.0")): module.to(device=device)
#   :569  elif not is_loaded_in_4bit_bnb and not is_loaded_in_8bit_bnb ...:
#                                                          module.to(device, dtype)
#
# ⚠ READ THE CONTROL FLOW, NOT THE WARNING. When the 8-bit gate at :562 fails, nothing
# moves the module and the ``elif`` at :569 is skipped too (its first clause is
# ``not is_loaded_in_8bit_bnb``, and the module IS 8-bit). The UNQUANTIZED components
# take that ``elif`` and DO go to CUDA. The pipeline therefore ends up SPLIT: VAE +
# UMT5 on the GPU, the quantized DiT stranded on the CPU, and the render dies with a
# device-mismatch RuntimeError at the FIRST denoise step — after paying a multi-GB
# load.
#
# ⚠ AND IN ONE HALF OF THE FAILURE SPACE IT IS COMPLETELY SILENT — verified by reading
# :541 rather than assuming. That warning ("…moving it to cuda via `.to()` is not
# supported") is conditioned ONLY on ``bitsandbytes < 0.48.0``. So:
#   * bnb < 0.48.0                      -> warned, then stranded.
#   * bnb >= 0.48.0, transformers <= 4.58.0 -> :541 does not fire, :562 still fails on
#     the transformers clause, and the DiT is stranded with NO log line at all.
# The second case is exactly a box that took the raised pin below but is behind on
# transformers. A warning nobody reads is not a guard; NO warning at all is worse,
# which is why this is a decision-time gate and not a log-grep.
# The 4-bit path fails the same way, one gate earlier: transformers <= 4.44.0 skips
# :559, the ``elif`` is skipped as well, and nf4 strands identically — also silently,
# since :541 is 8-bit-only.
# (bitsandbytes < 0.43.2 on 4-bit is the one LOUD variant — :559 runs and
# ModelMixin.to raises ValueError. Still a failure after the load; still guarded.)
#
# So whole-GPU placement of a bnb-quantized pipeline is only ever CHOSEN when the
# installed stack can actually perform the move. Everything else offloads, which is
# what the fleet did for its whole life and what always works.
#
# Versions verified in this venv 2026-07-27: diffusers 0.39.0, transformers 5.12.1,
# accelerate 1.14.0, bitsandbytes ABSENT. ae runs bitsandbytes 0.49.2.
_BNB_MIN_FOR_MOVE = {"int8": (0, 48, 0), "fp8": (0, 43, 2)}
_TRANSFORMERS_MIN_FOR_MOVE = {"int8": (4, 58, 0), "fp8": (4, 44, 0)}  # STRICT >


def _version_tuple(raw: str | None) -> tuple[int, ...] | None:
    """A PEP440-ish version string -> comparable int tuple, or None when unusable.

    Deliberately dumb: split the leading numeric-dot run and drop any suffix
    (``0.49.2.dev0``, ``5.12.1+cu128`` -> (0,49,2) / (5,12,1)). We only ever ask
    coarse "is it at least X" questions, and an UNPARSEABLE version must read as
    unknown -> guard closed, never as "probably fine"."""
    if not raw:
        return None
    head = str(raw).strip().split("+", 1)[0]
    parts: list[int] = []
    for chunk in head.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _quantized_move_supported(precision: "Precision", bnb_version: str | None,
                              transformers_version: str | None) -> bool:
    """PURE: can THIS stack move a bnb-quantized pipeline to CUDA via ``pipe.to()``?

    True only for INT8/FP8 with both versions present AND satisfying the diffusers
    0.39 gates above (bnb ``>=``, transformers strictly ``>``). Any non-quantized
    precision returns True — there is nothing to guard, ``.to()`` is unconditional.
    A missing/unparseable version returns False: we cannot verify the move, so we do
    not risk the strand."""
    prec = str(getattr(precision, "value", precision)).lower()
    bnb_min = _BNB_MIN_FOR_MOVE.get(prec)
    if bnb_min is None:
        return True                        # bf16/fp16/fp32 — not quantized, no gate
    bnb = _version_tuple(bnb_version)
    tfm = _version_tuple(transformers_version)
    if bnb is None or tfm is None:
        return False
    return bnb >= bnb_min and tfm > _TRANSFORMERS_MIN_FOR_MOVE[prec]


def _installed_quantized_move_ok(precision: "Precision") -> bool:
    """``_quantized_move_supported`` against the versions actually installed here.

    LAZY + TOTAL: reads ``importlib.metadata`` (never imports torch/bitsandbytes), so
    this module stays app-boot-safe and this function is callable on a box with no GPU
    stack. Any probe failure is a False — unverifiable means offload."""
    import importlib.metadata as md

    def _v(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001 — absent/broken dist metadata == unknown
            return None

    return _quantized_move_supported(precision, _v("bitsandbytes"), _v("transformers"))


def _should_place_whole_on_gpu(
    precision: Precision,
    model_gb: float | None,
    max_vram_gb: float | None,
    margin: float = _PLACEMENT_MARGIN_GB,
    *,
    model_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    n_frames: int | None = None,
    quantized_move_ok: bool | None = None,
) -> bool:
    """PURE placement decision (no GPU — unit-testable). True => move the WHOLE pipeline
    to CUDA (``pipe.to("cuda")``); False => ``enable_model_cpu_offload()``.

    DERIVED, NOT FUDGED (2026-07-27). When the model has a measured footprint and the
    geometry is known, this compares the REAL total —
    ``DiT(precision) + UMT5 + VAE [+ CLIP] [+ MoE expert] + activations(tokens)`` —
    against the placement budget (``_placement_budget_gib``: the min of the live card
    and any declared ceiling). Before, it compared the registry's **DiT-only** number
    against a flat 16 GB margin, which is adding two quantities that measure different
    things: ``8.2 + 16.0 = 24.2 > 23.56`` refused wan2.1-t2v-1.3b by **0.64 GB** on a
    card where the derived need is 17.897 GiB with 5.663 GiB to spare (the 2026-07-07
    incident MEASURED 19.6 there; see the unclosed-residual note on
    ``_placement_need_gib`` — this function does not reproduce that number and no
    longer claims to). Every 480p render on this fleet took the slow offload branch
    because of that arithmetic.

    ⚠ THE INT8/FP8 BLANKET EARLY RETURN IS GONE — REPLACED BY A CAPABILITY CHECK, NOT
    BY NOTHING (corrected 2026-07-27). Its rationale ("calling ``.to()`` on an
    8-/4-bit model is unsupported") was stale AS A BLANKET RULE but is exactly right
    on an under-versioned stack, and dropping it outright opened a new failure: on
    diffusers 0.39 an 8-bit pipeline whose stack fails the version gates is not moved
    and not refused, it is SPLIT — unquantized parts to CUDA, the DiT left on the CPU
    — and dies at the first denoise after a multi-GB load. So a quantized precision
    is placed whole-on-GPU only when it BOTH fits and the installed
    bitsandbytes/transformers can actually perform the move
    (``_installed_quantized_move_ok``; see the gate table above). ``quantized_move_ok``
    overrides that probe for tests and for a caller that has already resolved it.

    Falls back to the legacy ``model_gb + margin`` test whenever the footprint or the
    geometry is unknown, so an unmeasured model keeps exactly today's conservative
    behaviour. A missing budget => False (offload)."""
    if max_vram_gb is None:
        return False
    if model_id and width and height and n_frames:
        need = _placement_need_gib(model_id, precision, width, height, n_frames)
        if need is not None:
            fits = need <= max_vram_gb
            # The move must be POSSIBLE, not just affordable. Consulted lazily and ONLY
            # for a bnb-quantized precision that would otherwise be placed — a bf16
            # render never pays for the probe, and an override passed by a caller who
            # is thinking about int8 can never accidentally offload an unquantized one.
            movable = True
            prec_key = str(getattr(precision, "value", precision)).lower()
            if fits and prec_key in _BNB_MIN_FOR_MOVE:
                movable = (_installed_quantized_move_ok(precision)
                           if quantized_move_ok is None else bool(quantized_move_ok))
            logger.info("wan placement: %s @%s %dx%dx%df needs %.2f GiB, budget "
                        "%.2f GiB -> %s", model_id,
                        getattr(precision, "value", precision), width, height,
                        n_frames, need, max_vram_gb,
                        "WHOLE-GPU" if fits and movable
                        else ("offload (fits, but the installed bitsandbytes/"
                              "transformers cannot move a quantized pipeline)"
                              if fits else "offload"))
            return fits and movable
    # Legacy path: unmeasured model or unknown geometry.
    if precision in (Precision.INT8, Precision.FP8):
        return False
    if model_gb is None:
        return False
    return (model_gb + margin) <= max_vram_gb


# --------------------------------------------------------------------------- #
# CUDA allocator defragmentation (item 7) — must run BEFORE torch imports
# --------------------------------------------------------------------------- #
def _prime_cuda_allocator() -> bool:
    """OPT-IN ONLY (HUGPY_CUDA_EXPANDABLE=1): set
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before torch import to
    defragment the CUDA allocator (evidence: the 14B-int8 OOM failed an 80MB
    allocation with ~2.71 GiB reserved-but-unallocated).

    WHY OPT-IN (2026-07-08 ae incident): the original setdefault-on-by-default
    variant CRASH-LOOPED ae — os.environ survives the agent's re-exec, so one
    render primed the flag into the process lineage forever, and this
    driver/torch combo dies natively under expandable_segments (renders died in
    load ~30-40s; then even boot warm-ups crashed). The flag is only applied
    when the operator explicitly sets HUGPY_CUDA_EXPANDABLE=1 on a box.

    HONESTY: the setting only takes effect if the CUDA allocator has NOT already
    initialized in this process. On a worker whose agent avoids torch at boot that is
    normally true for the first studio render, but a prior in-process
    torch/transformers load may have initialized it — so we log whether ``torch`` was
    already imported at this point (sys.modules probe) so it is clear whether this line
    or the process env is doing the work. Returns True iff torch was already imported
    (the line is then likely a no-op for this process)."""
    already = "torch" in sys.modules
    if os.environ.get("HUGPY_CUDA_EXPANDABLE", "").strip() == "1":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        # DETOX: if a prior (0.1.158) prime leaked this exact value into the
        # re-exec-surviving environ, remove it so the box heals on converge.
        if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True":
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    logger.info("wan cuda allocator: PYTORCH_CUDA_ALLOC_CONF=%s (torch already imported: %s)",
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), already)
    return already


# --------------------------------------------------------------------------- #
# OFFLOAD-branch VRAM levers (item 4) — pure/duck-testable, no heavy deps here
# --------------------------------------------------------------------------- #
def _engage_memory_savers(pipe) -> list[str]:
    """On the OFFLOAD placement branch, engage diffusers' peak-VRAM levers so a
    14B-int8 i2v @480p fits next to comfy on a shared 24GB card (live 2026-07-07 it
    OOM'd by ~0.5GB). Each lever is best-effort: a diffusers build / pipeline / VAE
    that lacks it raises ``AttributeError``, which is caught and the lever skipped —
    never fail a render over a memory hint. Returns the names that engaged (also
    logged), so a duck-typed pipe can unit-test the wiring with no GPU.

    diffusers 0.39 surface (verified in-tree): ``DiffusionPipeline`` (base of all
    three Wan pipelines) has ``enable_attention_slicing``; ``AutoencoderKLWan`` has
    ``enable_tiling`` but NOT ``enable_slicing`` — so ``vae.enable_slicing()`` is
    the AttributeError-guarded lever that legitimately no-ops on the Wan VAE."""
    engaged: list[str] = []
    try:
        pipe.enable_attention_slicing()
        engaged.append("attention_slicing")
    except Exception:  # a memory HINT must never fail a render (keeper hardening)
        pass
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        try:
            vae.enable_tiling()
            engaged.append("vae_tiling")
        except Exception:  # a memory HINT must never fail a render (keeper hardening)
            pass
        try:
            vae.enable_slicing()
            engaged.append("vae_slicing")
        except Exception:  # a memory HINT must never fail a render (keeper hardening)
            pass
    if engaged:
        logger.info("wan offload VRAM levers engaged: %s", ", ".join(engaged))
    return engaged


def _place_pipe(pipe, place_whole: bool) -> list[str]:
    """Apply the placement decision to a loaded pipe and engage the VRAM levers on
    BOTH branches. Module-level (shared by wan_i2v + wan_vace) and duck-testable
    with no GPU. Returns the list of engaged levers.

    ⚠ THE SAVERS NOW RUN ON THE WHOLE-GPU BRANCH TOO (2026-07-27), and this is
    load-bearing rather than tidiness. ``vae.enable_tiling()`` is what makes VAE
    decode a BOUNDED constant instead of a resolution-squared spike: untiled,
    ``AutoencoderKLWan._decode`` runs full-frame fp32 convolutions and accumulates
    the whole clip via ``torch.cat``. That untiled decode — not the denoise — is
    the most plausible cause of the 2026-07-07 OOM the old flat margin was built
    to avoid.
    
    So it would have been exactly wrong to flip placement onto measured need while
    leaving this branch unguarded: ``_placement_need_gib`` prices decode as a bounded
    constant (``_DECODE_WS_GIB`` = 1.36 GiB), and that price is only honest if tiling
    is actually on. A calculation that assumes a lever nobody pulled is how you
    reproduce the incident you were trying to prevent.

    ⚠ AND THE PRICE IS AN ASSUMPTION, NOT A MEASUREMENT (2026-07-27). 1.36 GiB has no
    measured provenance in this tree, and it is currently inert anyway: the need
    calculation takes ``max(denoise_ws, _DECODE_WS_GIB)`` and the denoise term never
    drops below the 3.400 GiB intercept, so decode never governs at any geometry.
    Tiling is therefore load-bearing for a bound this file cannot yet defend with a
    number — keep it engaged on both branches until someone measures decode."""
    if place_whole:
        pipe.to("cuda")                        # whole pipeline on GPU (it fits)
        return _engage_memory_savers(pipe)     # tiling keeps decode bounded
    pipe.enable_model_cpu_offload()            # too big -> stream modules
    return _engage_memory_savers(pipe)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_wan_i2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
    on_step: "Callable[[int, int], None] | None" = None,
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
    (default) = never cancel.

    ``on_step`` is an OPTIONAL denoise-progress sink (k57): ``on_step(step, steps)``
    is called at each ``callback_on_step_end`` boundary with the 1-based step and
    the total. It rides the SAME diffusers callback the cancel probe uses, so it
    costs nothing extra, and it is the ONLY honest source of within-clip progress —
    without it a single-segment render has no measurable movement between "started"
    and "done" and the console's bar sits at 0 for the whole render. Best-effort:
    the sink is wrapped, so a slow/throwing consumer can never break a render.
    None (default) = report nothing (unchanged behaviour)."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    # ── IDENTITY GUARD: refuse what this runner cannot honour ────────────────
    # THIS RUNNER NEVER READS reference_images (grep it: zero occurrences). VACE
    # does. So an id_lock request that lands HERE silently produces a plausible
    # clip of the WRONG PERSON, with no error — the worst failure shape there is.
    #
    # It is reachable: CAPABILITY_TASKS[ID_LOCK] = (VACE_CONTROL, I2V), and the
    # I2V fallback binds whenever no VACE model fits the geometry — INCLUDING the
    # studio routes' own 512x512 default, which is outside vace-1.3b's 480p
    # envelope. models_seed's comment claimed "in practice VACE always wins ... so
    # id_lock never silently routes to a runner that would ignore the references";
    # that was true only inside the 480p envelope and is corrected there now.
    #
    # Fail LOUD instead of dropping the identity. This is the point of harm, so
    # the guard lives here — it holds no matter how routing later changes.
    if getattr(manifest, "reference_images", None):
        return Err(StageError(
            ErrorCode.NO_CAPABLE_MODEL,
            f"identity lock requested ({len(manifest.reference_images)} reference "
            f"image(s)) but this render bound the i2v runner, which cannot consume "
            f"them — the identity would be silently dropped. Wan-VACE is the "
            f"reference-to-video path; it maxes at 480p, so request a geometry "
            f"within 832x480 (the studio default) for id_lock."))

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

    # CUDA allocator defragmentation (item 7): set PYTORCH_CUDA_ALLOC_CONF BEFORE any
    # torch import (preflight below imports torch to probe CUDA), so it precedes the
    # very first import in this render flow and the sys.modules log stays honest. No-op
    # + harmless on this GPU-less box (preflight then returns DEPS_MISSING before torch).
    _prime_cuda_allocator()

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
        UniPCMultistepScheduler,
        WanImageToVideoPipeline,
        WanPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import load_image

    cfg = MODEL_REGISTRY.get(manifest.model_id)
    # WEIGHTS SOURCE (item 5): prefer the box-local hot NVMe copy if it holds the
    # model, else the shared root — a faster LOAD only; does not affect content_hash.
    model_dir, weights_root_used = _resolve_model_dir(manifest, cfg.weight_uri)
    logger.info("wan i2v: loading %s from %s (%s weights root)",
                cfg.weight_uri, model_dir, weights_root_used)
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

    # PLACEMENT decision (operator directive: put sub-envelope models WHOLLY on the GPU
    # instead of parking ~15GB in worker RAM via offload). Pure + precomputed here;
    # applied per pipe below. A bnb-quantized (INT8/FP8) precision is placed only when
    # it BOTH fits and the installed bitsandbytes/transformers can actually perform the
    # move (_installed_quantized_move_ok) — otherwise the historical offload path.
    model_gb = cfg.vram.as_map().get(manifest.precision)
    place_whole = _should_place_whole_on_gpu(
        manifest.precision, model_gb, _max_vram_gb(manifest),
        # Pass the identity + geometry so the decision uses MEASURED component bytes
        # (DiT + UMT5 + VAE [+ CLIP] + token-scaled activations) instead of the
        # registry's DiT-only number plus a flat fudge. Without these it silently
        # falls back to the legacy test.
        model_id=manifest.model_id, width=width, height=height, n_frames=n_frames)
    # SHIFT: the flow-match/UniPC scheduler shift RECORDED in the manifest (set by
    # resolve_sampler from the resolution: 3.0 @480p, 5.0 @720p+). None (unset) leaves
    # the pipeline's own default scheduler untouched.
    flow_shift = manifest.sampler.shift

    def _prepare_pipe(pipe):
        """Apply the manifest's scheduler shift + the placement decision to a loaded
        pipe. Wiring shift here (not just recording it) closes the gap where
        manifest.sampler.shift existed but was never consumed — the denoise now uses
        EXACTLY the value in the manifest (INV-1)."""
        if flow_shift is not None:
            try:
                # Wan denoises with a flow-prediction UniPC scheduler; from_config keeps
                # the model's own scheduler config and only overrides flow_shift.
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=flow_shift)
            except Exception:
                # A diffusers build whose UniPC lacks flow_shift: keep the default
                # scheduler rather than fail the render (shift is still in the manifest).
                pass
        # Placement + the VRAM levers (item 4), engaged on BOTH branches. _place_pipe
        # offloads an over-budget (or unmovably-quantized) pipe and engages attention
        # slicing + VAE tiling/slicing; a model that fits goes wholly to CUDA.
        _place_pipe(pipe, place_whole)

    # Cooperative mid-render cancel wiring (Task 1). diffusers 0.39's
    # WanImageToVideoPipeline.__call__ supports `callback_on_step_end`; the callback
    # sets `pipe._interrupt=True` so the denoise loop breaks at the next step
    # boundary. We ALSO re-check should_cancel() around the call so a cancel is
    # still honored if a box's diffusers lacks the callback param. This whole path
    # is BOX-ONLY (preflight short-circuits the GPU-less VM above).
    def _cancel_step_cb(pipe_ref, step_index, timestep, cb_kwargs):
        if should_cancel is not None and should_cancel():
            pipe_ref._interrupt = True   # diffusers checks self.interrupt each step
        if on_step is not None:
            try:
                on_step(int(step_index) + 1, int(steps))
            except Exception:  # noqa: BLE001 — telemetry never breaks a render
                pass
        return cb_kwargs

    call_extra: dict = {}
    if should_cancel is not None or on_step is not None:
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

        # B2 chain — "extend the movie": condition an i2v render on the source clip's
        # LAST FRAME when no start_image was given, BEFORE loading multi-GB weights so
        # a bad source fails fast (errors-as-data). source_video is in the manifest
        # (content_hash), so the extend is deterministic + resume-safe. t2v is
        # text-only (task != I2V) -> the source is carried, never used. BOX-ONLY like
        # the rest of this real path (the GPU-less VM short-circuits at preflight).
        if (start_image is None and (manifest.source_video or "")
                and manifest.task == Task.I2V):
            last_frame = os.path.join(out_dir, _SOURCE_LASTFRAME_NAME)
            ok, stderr_tail = _extract_last_frame(manifest.source_video, last_frame)
            if not ok:
                return Err(StageError(
                    ErrorCode.IO_ERROR,
                    f"could not extract last frame from source_video: {stderr_tail}",
                    (("source_video", manifest.source_video),)))
            start_image = last_frame

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
            # Placement + scheduler shift (see _prepare_pipe). A model that fits the
            # budget goes wholly to CUDA — including a bnb-quantized one, but ONLY when
            # the installed stack can move it (see _should_place_whole_on_gpu).
            _prepare_pipe(pipe)
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
                output_type="pil",
                **call_extra,
            )
        else:
            # --- t2v (start_image is None) ---
            pipe = WanPipeline.from_pretrained(
                model_dir, transformer=transformer, vae=vae,
                torch_dtype=compute_dtype)
            # Placement + scheduler shift (see _prepare_pipe).
            _prepare_pipe(pipe)
            result = pipe(
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
        # output_type="pil", but the actual per-frame type varies by pipeline/
        # version (list of PIL, ndarray (T,H,W,C) float [0,1], torch tensor) —
        # the FIRST real render on ae (2026-07-07) got ndarray and PIL-only
        # .save() failed AFTER a full denoise. Normalize per-frame.
        frames = result.frames[0]
        actual_frames = len(frames)

        frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i, fr in enumerate(frames):
            _frame_to_pil(fr).save(
                os.path.join(frame_dir, f"frame_{i:05d}.png"), "PNG")

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
        # Provenance records WHICH weights root served (hot NVMe vs shared) — a
        # sidecar-only field (item 5); it is NOT a canonical input, so it never
        # participates in the content_hash.
        prov = _provenance_dict(manifest)
        prov["weights_root_used"] = weights_root_used
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(prov, indent=2, sort_keys=True))
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
