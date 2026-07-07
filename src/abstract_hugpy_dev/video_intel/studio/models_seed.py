"""The zoo, as data (§2). Registering this module populates all three registries.

VRAM figures are PLANNING ESTIMATES in GB - they move with resolution, frame
count, precision, and offload strategy. They exist so the router can reason about
fit, not as a promise. Pin real weight hashes before production (INV-1/INV-2);
until then models are `unpinned=True` and boot requires STUDIO_ALLOW_UNPINNED=1.

`verify_uri=True` marks a repo path NOT directly confirmed in the source thread -
treat those as "check before you `huggingface-cli download`."
"""

from __future__ import annotations

from .enums import (
    AdapterKind,
    Capability,
    DeterminismClass,
    Framework,
    LicenseClass,
    PathClass,
    Precision,
    Task,
)
from .registry import RunnerSpec, register_model, register_runner, set_capability_tasks
from .schemas import ModelConfig, Resolution, VramEnvelope

# --- reusable resolution constants ----------------------------------------
R_256 = Resolution(256, 256, 24)
R_512 = Resolution(512, 512, 8)
R_COG = Resolution(720, 480, 8)
R_MOCHI = Resolution(848, 480, 30)
R_480P = Resolution(832, 480, 16)
R_768 = Resolution(768, 768, 24)
R_720P = Resolution(1280, 720, 24)
R_720P_V = Resolution(720, 1280, 24)
R_1080P = Resolution(1920, 1080, 24)
R_4K = Resolution(3840, 2160, 50)


def E(*pairs: tuple[Precision, float]) -> VramEnvelope:
    return VramEnvelope(tuple(pairs))


HF = "https://huggingface.co/"

# --------------------------------------------------------------------------
# Runners: (framework, task) -> how to execute it. Entrypoints wired explicitly.
# --------------------------------------------------------------------------
_RUNNERS = (
    # Task 3b: WAN T2V is now WIRED to the real runner — run_wan_t2v is a thin DRY
    # delegation to run_wan_i2v's WanPipeline (t2v) branch with start_image forced
    # None, so it inherits the identical preflight / bnb / cancel / atomic path.
    RunnerSpec(Framework.WAN, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.wan_t2v:run_wan_t2v", Precision.INT8),
    # P0-6: WAN i2v is WIRED to the real runner (import-safe, graceful-degrading,
    # bitsandbytes int8/nf4 on the box). VACE stays a dormant placeholder until its
    # bet lands (validate_registry only checks a runner is registered, not
    # importable). run_wan_i2v itself also serves t2v when start_image is None.
    RunnerSpec(Framework.WAN, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.wan_i2v:run_wan_i2v", Precision.INT8),
    RunnerSpec(Framework.WAN, Task.VACE_CONTROL, "abstract_hugpy_dev.video_intel.studio.runners.wan:vace", Precision.INT8),
    RunnerSpec(Framework.LTX, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.ltx:t2v", Precision.FP8),
    RunnerSpec(Framework.LTX, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.ltx:i2v", Precision.FP8),
    RunnerSpec(Framework.LTX, Task.AUDIO_VIDEO, "abstract_hugpy_dev.video_intel.studio.runners.ltx:av", Precision.FP8),
    RunnerSpec(Framework.LTX, Task.UPSCALE, "abstract_hugpy_dev.video_intel.studio.runners.ltx:upscale", Precision.INT8),
    RunnerSpec(Framework.HUNYUAN, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.hunyuan:t2v", Precision.INT8),
    RunnerSpec(Framework.HUNYUAN, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.hunyuan:i2v", Precision.INT8),
    RunnerSpec(Framework.HUNYUAN, Task.AVATAR_LIPSYNC, "abstract_hugpy_dev.video_intel.studio.runners.hunyuan:avatar", Precision.FP8),
    RunnerSpec(Framework.COGVIDEOX, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.cog:t2v", Precision.INT8),
    RunnerSpec(Framework.COGVIDEOX, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.cog:i2v", Precision.INT8),
    RunnerSpec(Framework.MOCHI, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.mochi:t2v", Precision.FP8),
    RunnerSpec(Framework.OPEN_SORA, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.opensora:t2v", Precision.FP8),
    RunnerSpec(Framework.OPEN_SORA, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.opensora:i2v", Precision.FP8),
    RunnerSpec(Framework.SKYREELS, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.skyreels:i2v", Precision.INT8),
    RunnerSpec(Framework.ANIMATEDIFF, Task.MOTION_MODULE, "abstract_hugpy_dev.video_intel.studio.runners.animatediff:motion", Precision.INT8),
    RunnerSpec(Framework.FRAMEPACK, Task.STREAM_I2V, "abstract_hugpy_dev.video_intel.studio.runners.framepack:stream", Precision.FP16, supports_streaming=True),
    RunnerSpec(Framework.LTX, Task.UPSCALE, "abstract_hugpy_dev.video_intel.studio.runners.ltx:upscale2", Precision.INT8) if False else
    RunnerSpec(Framework.RIFE, Task.INTERPOLATE, "abstract_hugpy_dev.video_intel.studio.runners.rife:interp", Precision.FP16),
    RunnerSpec(Framework.CODEFORMER, Task.RESTORE_FACE, "abstract_hugpy_dev.video_intel.studio.runners.codeformer:restore", Precision.FP16),
    # SYNTHETIC: no-model procedural runner (P0-B1). Proves the whole spine
    # (capability -> binding -> manifest -> frames -> ffmpeg -> mp4) end-to-end
    # with NO GPU/weights. min_precision=INT8 (the floor) so any precision binds.
    RunnerSpec(Framework.SYNTHETIC, Task.I2V, "abstract_hugpy_dev.video_intel.studio.runners.synthetic:run_synthetic_i2v", Precision.INT8),
    # Task 3b: SYNTHETIC t2v prover — no GPU/weights, deterministic, TEXT-AGNOSTIC
    # frames (the prompt rides in the manifest for provenance but never alters a
    # pixel). Thin wrapper over run_synthetic_i2v with start_image forced None.
    RunnerSpec(Framework.SYNTHETIC, Task.T2V, "abstract_hugpy_dev.video_intel.studio.runners.synthetic:run_synthetic_t2v", Precision.INT8),
)
for _r in _RUNNERS:
    register_runner(_r)


# --------------------------------------------------------------------------
# Capability -> satisfying tasks, in preference order (the join).
# --------------------------------------------------------------------------
set_capability_tasks({
    Capability.T2V: (Task.T2V, Task.AUDIO_VIDEO),
    Capability.I2V: (Task.I2V, Task.AUDIO_VIDEO, Task.STREAM_I2V),
    Capability.ID_LOCK: (Task.I2V,),
    Capability.KEYFRAME: (Task.I2V,),
    Capability.MOTION: (Task.VACE_CONTROL, Task.MOTION_MODULE),
    Capability.V2V: (Task.VACE_CONTROL,),
    Capability.INPAINT: (Task.VACE_CONTROL,),
    Capability.OUTPAINT: (Task.VACE_CONTROL,),
    Capability.RETAKE: (Task.VACE_CONTROL,),
    Capability.STREAM: (Task.STREAM_I2V,),
    Capability.AUDIO: (Task.AUDIO_VIDEO,),
    Capability.LIPSYNC: (Task.AVATAR_LIPSYNC, Task.AUDIO_VIDEO),
    Capability.UPRES: (Task.UPSCALE,),
    Capability.INTERP: (Task.INTERPOLATE,),
    Capability.RESTORE: (Task.RESTORE_FACE,),
    Capability.ASSEMBLE: (),   # orchestration stage; PLANNED, no model
})


_ID_ADAPTERS = frozenset({
    AdapterKind.IDENTITY_LORA, AdapterKind.IP_ADAPTER,
    AdapterKind.CANONICAL_MULTIVIEW, AdapterKind.CAMERA_CTRL,
})

# --------------------------------------------------------------------------
# Base generative models
# --------------------------------------------------------------------------
_MODELS = (
    # ---- Wan family (Alibaba, Apache-2.0) --------------------------------
    ModelConfig(
        model_id="wan2.1-t2v-1.3b", family=Framework.WAN,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.FP16, 8.2), (Precision.INT8, 5.0)),
        resolutions=(R_480P,), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-T2V-1.3B", source_url=HF + "Wan-AI/Wan2.1-T2V-1.3B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA, AdapterKind.CAMERA_CTRL}),
        notes="Consumer entry point; ~8GB fp16, fits a 3090 comfortably. 480p base.",
    ),
    ModelConfig(
        model_id="wan2.1-i2v-14b-720p", family=Framework.WAN,
        tasks=(Task.I2V,),
        capabilities=(Capability.I2V, Capability.ID_LOCK, Capability.KEYFRAME),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 18.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_720P_V, R_480P), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-I2V-14B-720P", source_url=HF + "Wan-AI/Wan2.1-I2V-14B-720P",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=_ID_ADAPTERS,
        notes="Identity-lock workhorse (ID-4: lock a still, then animate). FP8/INT8 to fit 3090.",
    ),
    ModelConfig(
        model_id="wan2.2-t2v-a14b", family=Framework.WAN,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.BF16, 42.0), (Precision.FP8, 20.0), (Precision.INT8, 16.0)),
        resolutions=(R_720P, R_480P), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.2-T2V-A14B", source_url=HF + "Wan-AI/Wan2.2-T2V-A14B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA, AdapterKind.CAMERA_CTRL}),
        notes="MoE ~27B total / ~14B active (high-noise + low-noise experts).",
    ),
    ModelConfig(
        model_id="wan2.2-i2v-a14b", family=Framework.WAN,
        tasks=(Task.I2V,), capabilities=(Capability.I2V, Capability.ID_LOCK),
        vram=E((Precision.BF16, 42.0), (Precision.FP8, 20.0), (Precision.INT8, 16.0)),
        resolutions=(R_720P, R_720P_V, R_480P), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.2-I2V-A14B", source_url=HF + "Wan-AI/Wan2.2-I2V-A14B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=_ID_ADAPTERS,
        notes="Best first-frame motion quality of the open i2v models per mid-2026 comparisons.",
    ),
    ModelConfig(
        model_id="wan2.1-vace-1.3b", family=Framework.WAN,
        tasks=(Task.VACE_CONTROL,),
        capabilities=(Capability.MOTION, Capability.V2V, Capability.INPAINT,
                      Capability.OUTPAINT, Capability.RETAKE),
        vram=E((Precision.FP16, 10.0), (Precision.INT8, 6.0)),
        resolutions=(R_480P,), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-VACE-1.3B", source_url=HF + "Wan-AI/Wan2.1-VACE-1.3B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.VACE, AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="Unified control: reference/depth/pose/inpaint/extend + RETAKE (MOT-5). Fits 3090.",
    ),
    ModelConfig(
        model_id="wan2.1-vace-14b", family=Framework.WAN,
        tasks=(Task.VACE_CONTROL,),
        capabilities=(Capability.MOTION, Capability.V2V, Capability.INPAINT,
                      Capability.OUTPAINT, Capability.RETAKE),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 20.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_480P), max_frames=81, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="Wan-AI/Wan2.1-VACE-14B", source_url=HF + "Wan-AI/Wan2.1-VACE-14B",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.VACE, AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="720p unified control. Multi-condition (depth + DWPose + mask) chaining supported.",
    ),

    # ---- LTX family (Lightricks, commercial agreement required) ----------
    ModelConfig(
        model_id="ltx-video-0.9.8-dev", family=Framework.LTX,
        tasks=(Task.T2V, Task.I2V),
        capabilities=(Capability.T2V, Capability.I2V, Capability.KEYFRAME),
        vram=E((Precision.FP16, 16.0), (Precision.FP8, 10.0), (Precision.INT8, 8.0)),
        resolutions=(R_1080P, R_720P, R_480P), max_frames=257, max_duration_s=10.0,
        license=LicenseClass.LTX_COMMERCIAL,
        weight_uri="Lightricks/LTX-Video-0.9.8-dev", source_url=HF + "Lightricks/LTX-Video",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="Fastest open path; great for storyboard/preview (PIPE-4). Diffusers: LTXConditionPipeline.",
    ),
    ModelConfig(
        model_id="ltx-2.3", family=Framework.LTX,
        tasks=(Task.AUDIO_VIDEO,),
        capabilities=(Capability.T2V, Capability.I2V, Capability.AUDIO, Capability.LIPSYNC),
        vram=E((Precision.BF16, 32.0), (Precision.FP8, 16.0)),
        resolutions=(R_4K, R_1080P, R_720P), max_frames=1000, max_duration_s=20.0,
        license=LicenseClass.LTX_COMMERCIAL, native_audio=True,
        weight_uri="Lightricks/LTX-2 (verify 2.3 tag)", source_url=HF + "Lightricks",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        accepts_adapters=frozenset({AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="Single-pass synced 4K audio+video incl. lip-sync (AUD-1). 32GB official; "
              "community GGUF Q4~15GB/Q3~12GB unofficial. Verify exact 2.3 repo id.",
    ),

    # ---- HunyuanVideo family (Tencent, community license) ----------------
    ModelConfig(
        model_id="hunyuanvideo", family=Framework.HUNYUAN,
        tasks=(Task.T2V, Task.I2V), capabilities=(Capability.T2V, Capability.I2V),
        vram=E((Precision.BF16, 80.0), (Precision.FP8, 24.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_720P_V), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.TENCENT_COMMUNITY,
        weight_uri="tencent/HunyuanVideo", source_url=HF + "tencent/HunyuanVideo",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA, AdapterKind.CONTROLNET}),
        notes="13B cinematic; strong motion/texture. Diffusers 4bit+tiling+offload ~6.5GB. "
              "License permits commercial use up to a MAU ceiling - verify for your scale. "
              "Text encoders: Kijai/HunyuanVideo_comfy.",
    ),
    ModelConfig(
        model_id="hunyuanvideo-avatar", family=Framework.HUNYUAN,
        tasks=(Task.AVATAR_LIPSYNC,), capabilities=(Capability.LIPSYNC,),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 24.0)),
        resolutions=(R_720P, R_720P_V), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.TENCENT_COMMUNITY,
        weight_uri="tencent/HunyuanVideo-Avatar", source_url=HF + "tencent",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        notes="Audio-driven human animation (AUD-3 lip-sync path). Verify exact repo id.",
    ),

    # ---- Other open bases -------------------------------------------------
    ModelConfig(
        model_id="cogvideox-5b", family=Framework.COGVIDEOX,
        tasks=(Task.T2V, Task.I2V), capabilities=(Capability.T2V, Capability.I2V),
        vram=E((Precision.FP16, 16.0), (Precision.INT8, 12.0)),
        resolutions=(R_COG,), max_frames=49, max_duration_s=6.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="THUDM/CogVideoX-5b", source_url=HF + "THUDM/CogVideoX-5b",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        accepts_adapters=frozenset({AdapterKind.IDENTITY_LORA}),
        notes="720x480 @ 8fps, 6s. Verify org (THUDM vs zai-org) and 5B license terms "
              "(2B is Apache-2.0; 5B ships a custom CogVideoX License).",
    ),
    ModelConfig(
        model_id="mochi-1-preview", family=Framework.MOCHI,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.BF16, 24.0), (Precision.FP8, 20.0)),
        resolutions=(R_MOCHI,), max_frames=163, max_duration_s=5.4,
        license=LicenseClass.APACHE_2_0,
        weight_uri="genmo/mochi-1-preview", source_url=HF + "genmo/mochi-1-preview",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        notes="10B, photoreal, slow (8+ min/clip on a 4090). Open training code for fine-tune.",
    ),
    ModelConfig(
        model_id="open-sora-v2", family=Framework.OPEN_SORA,
        tasks=(Task.T2V, Task.I2V), capabilities=(Capability.T2V, Capability.I2V),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 24.0)),
        resolutions=(R_768, R_256), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="hpcai-tech/Open-Sora-v2", source_url=HF + "hpcai-tech/Open-Sora-v2",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        notes="11B, 256/768px, fully open training pipeline (train on proprietary data). "
              "License reported as both Apache-2.0 and MIT across sources - verify.",
    ),
    ModelConfig(
        model_id="skyreels-v1", family=Framework.SKYREELS,
        tasks=(Task.I2V,), capabilities=(Capability.I2V, Capability.ID_LOCK),
        vram=E((Precision.BF16, 40.0), (Precision.FP8, 24.0), (Precision.INT8, 14.0)),
        resolutions=(R_720P, R_720P_V), max_frames=129, max_duration_s=5.0,
        license=LicenseClass.TENCENT_COMMUNITY,
        weight_uri="Skywork/SkyReels-V1", source_url=HF + "Skywork/SkyReels-V1",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        accepts_adapters=_ID_ADAPTERS,
        notes="HunyuanVideo fine-tune on film/TV; human-centric, convincing faces. "
              "Inherits Tencent community license. Verify exact repo (V1 vs V2).",
    ),

    # ---- Motion control (legacy SD path) ---------------------------------
    ModelConfig(
        model_id="animatediff-lightning", family=Framework.ANIMATEDIFF,
        tasks=(Task.MOTION_MODULE,), capabilities=(Capability.MOTION,),
        vram=E((Precision.FP16, 8.0), (Precision.INT8, 6.0)),
        resolutions=(R_512,), max_frames=64, max_duration_s=4.0,
        license=LicenseClass.OPENRAIL_M,
        weight_uri="ByteDance/AnimateDiff-Lightning", source_url=HF + "ByteDance/AnimateDiff-Lightning",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        accepts_adapters=frozenset({AdapterKind.MOTION_LORA, AdapterKind.CONTROLNET, AdapterKind.CAMERA_CTRL}),
        notes="SD1.5-era path: AnimateDiff (temporal) + ControlNet (structure). Stylized/anime, "
              "huge existing LoRA ecosystem. OpenRAIL-M use restrictions apply.",
    ),

    # ---- Streaming (autoregressive, constant-memory) ---------------------
    ModelConfig(
        model_id="framepack-i2v-hy", family=Framework.FRAMEPACK,
        tasks=(Task.STREAM_I2V,), capabilities=(Capability.STREAM,),
        vram=E((Precision.FP16, 6.0),),
        resolutions=(R_720P, R_480P), max_frames=3600, max_duration_s=120.0,
        license=LicenseClass.TENCENT_COMMUNITY, path_class=PathClass.STREAMING,
        weight_uri="lllyasviel/FramePackI2V_HY", source_url=HF + "lllyasviel/FramePackI2V_HY",
        default_determinism=DeterminismClass.DRIFTING, unpinned=True, verify_uri=True,
        notes="Constant-memory long i2v (~6GB!) on a HunyuanVideo base - ideal for the 3090 "
              "streaming path (STR-3). Code Apache (github lllyasviel/FramePack); weights HY-derived. "
              "Note STR-6: not for frame-perfect identity lock.",
    ),

    # ---- Quality ladder ---------------------------------------------------
    ModelConfig(
        model_id="ltxv-spatial-upscaler-0.9.8", family=Framework.LTX,
        tasks=(Task.UPSCALE,), capabilities=(Capability.UPRES,),
        vram=E((Precision.FP16, 12.0), (Precision.INT8, 8.0)),
        resolutions=(R_4K, R_1080P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.LTX_COMMERCIAL,
        weight_uri="Lightricks/ltxv-spatial-upscaler-0.9.8",
        source_url=HF + "Lightricks/ltxv-spatial-upscaler-0.9.8",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True,
        notes="Latent spatial upscaler (QLD-2). Pairs with LTX base.",
    ),
    ModelConfig(
        model_id="rife-practical", family=Framework.RIFE,
        tasks=(Task.INTERPOLATE,), capabilities=(Capability.INTERP,),
        vram=E((Precision.FP16, 3.0),),
        resolutions=(R_4K, R_1080P, R_720P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.MIT,
        weight_uri="hzwer/Practical-RIFE", source_url="https://github.com/hzwer/Practical-RIFE",
        default_determinism=DeterminismClass.EXACT, unpinned=True, verify_uri=True,
        notes="Frame interpolation 24->48/60 and AR-seam smoothing (QLD-3). Deterministic. "
              "GitHub weights; verify HF mirror if you want one.",
    ),
    ModelConfig(
        model_id="codeformer", family=Framework.CODEFORMER,
        tasks=(Task.RESTORE_FACE,), capabilities=(Capability.RESTORE,),
        vram=E((Precision.FP16, 3.0),),
        resolutions=(R_4K, R_1080P, R_720P), max_frames=100000, max_duration_s=100000.0,
        license=LicenseClass.PROPRIETARY,   # S-Lab 1.0: NON-COMMERCIAL
        weight_uri="sczhou/CodeFormer", source_url="https://github.com/sczhou/CodeFormer",
        default_determinism=DeterminismClass.SEEDED_APPROX, unpinned=True, verify_uri=True,
        notes="Face/detail restore (QLD-4). NTU S-Lab 1.0 license = NON-COMMERCIAL: will not "
              "auto-route for commercial_use. Use GFPGAN (TencentARC/GFPGAN, Apache) for commercial.",
    ),

    # ---- SYNTHETIC (no-model) --------------------------------------------
    # P0-B1: a deterministic procedural i2v runner with NO weights/GPU. It exists
    # to prove the full studio spine end-to-end (capability -> binding -> manifest
    # -> frames -> ffmpeg -> mp4 + sidecars) before any real model bet. It is
    # PINNED with a fixed pseudo weight_hash so it is production-clean (needs no
    # STUDIO_ALLOW_UNPINNED) and DeterminismClass.EXACT (synthetic IS exact). It
    # only wins the router when the VRAM budget is too small for any real model
    # (min real i2v footprint is 8GB), so it never shadows a genuine binding.
    ModelConfig(
        model_id="synthetic-i2v", family=Framework.SYNTHETIC,
        tasks=(Task.I2V,), capabilities=(Capability.I2V,),
        vram=E((Precision.INT8, 0.1)),
        resolutions=(R_720P, R_512, R_256), max_frames=240, max_duration_s=10.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="synthetic://procedural-i2v",
        source_url="synthetic://procedural-i2v",
        default_determinism=DeterminismClass.EXACT, path_class=PathClass.OFFLINE,
        weight_hash="synthetic-i2v-v1-0000000000000000000000000000000000000000000000000000000000000000",
        synthetic=True,   # LAST-RESORT: any real model that fits ALWAYS outranks it
                          # in router scoring, so it can never shadow a real binding
                          # (not even when its tiny 0.1GB envelope also fits).
        notes="No-model procedural runner (P0-B1). Deterministic frames from the "
              "manifest seed + top resolution, assembled to H.264 via ffmpeg. Tiny "
              "0.1GB envelope so it only binds when no real model fits the budget.",
    ),
    # Task 3b: the SYNTHETIC T2V twin — the no-model text-to-video prover. Same
    # last-resort discipline as synthetic-i2v (synthetic=True => any real Wan t2v
    # model ALWAYS outranks it), same pinned/EXACT posture. Deliberately capped at
    # 512x512 (R_512/R_256, NO 720p): a t2v demo is tiny by definition, and the cap
    # means synthetic can never even be a CANDIDATE against a real model at the
    # larger formats real Wan t2v owns (it also keeps the "0.5GB @ 720p is
    # unroutable" router invariant intact). The PROMPT is recorded in the manifest
    # (content_hash + sidecar) for provenance but never touches a pixel — synthetic
    # frames are a pure function of seed + geometry, so t2v stays byte-deterministic.
    ModelConfig(
        model_id="synthetic-t2v", family=Framework.SYNTHETIC,
        tasks=(Task.T2V,), capabilities=(Capability.T2V,),
        vram=E((Precision.INT8, 0.1)),
        resolutions=(R_512, R_256), max_frames=240, max_duration_s=10.0,
        license=LicenseClass.APACHE_2_0,
        weight_uri="synthetic://procedural-t2v",
        source_url="synthetic://procedural-t2v",
        default_determinism=DeterminismClass.EXACT, path_class=PathClass.OFFLINE,
        weight_hash="synthetic-t2v-v1-0000000000000000000000000000000000000000000000000000000000000000",
        synthetic=True,   # LAST-RESORT: any real t2v model that fits ALWAYS outranks
                          # it in router scoring, so it never shadows a real binding.
        notes="No-model procedural T2V prover (Task 3b). Deterministic frames from "
              "the manifest seed + top resolution (prompt-agnostic — the prompt is "
              "recorded for provenance, never alters pixels), assembled to H.264 via "
              "ffmpeg. Tiny 0.1GB envelope, capped at 512x512 (no 720p) so it only "
              "binds a tiny-budget T2V demo and never shadows a real Wan t2v model.",
    ),
)
for _m in _MODELS:
    register_model(_m)
