"""Text-to-image runner.

Serves ("transformers", "text-to-image"). One diffusers pipeline per
model_key (class-level singleton cache — same pattern as
FeatureExtractionRunner), generation runs in a worker thread.

diffusers/torch are imported lazily inside the .pipeline property, so
importing this module doesn't require either library to be installed.
Only callers that actually generate pay the import cost — and if it
fails, the error fires at first use, not at dispatch import time.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
from typing import Any, Dict

from .imports import *           # ensure_model, UPLOADS_HOME, TokenEvent, DoneEvent, …
from .schemas import GeneratedImage, ImageGenRequest, ImageGenResult

logger = logging.getLogger(__name__)

# Per-model GENERATE serialization, shared by both runners. A diffusers
# pipeline object is NOT safe under concurrent __call__ (scheduler state
# races) — and the scene fan-out (video_intel/runners/scene.py) deliberately
# issues concurrent frame requests that may land on the same worker. Different
# models still generate in parallel; same-model calls queue.
_GEN_LOCKS: Dict[str, threading.Lock] = {}
_GEN_LOCKS_GUARD = threading.Lock()


def _generate_lock(model_key: str) -> threading.Lock:
    with _GEN_LOCKS_GUARD:
        lock = _GEN_LOCKS.get(model_key)
        if lock is None:
            lock = _GEN_LOCKS[model_key] = threading.Lock()
        return lock


class ImageGenRunner:
    """Runner for diffusers text-to-image pipelines.

    Per-process singleton cache (_PIPELINES) means many runner instances
    for the same model_key share one loaded pipeline. The Runner wrapper
    itself is cheap; the pipeline isn't.
    """

    request_type = ImageGenRequest
    result_type = ImageGenResult

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs

    # --- pipeline loading (lazy, singleton) ---------------------------------

    @property
    def pipeline(self):
        cached = self._PIPELINES.get(self.model_key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._PIPELINES.get(self.model_key)
            if cached is not None:
                return cached

            try:
                import torch
                from diffusers import AutoPipelineForText2Image
            except ImportError as exc:
                raise RuntimeError(
                    "diffusers + torch are required for text-to-image tasks "
                    "but are not installed. `pip install diffusers torch`."
                ) from exc

            model_dir = ensure_model(self.model_key)
            cuda = torch.cuda.is_available()
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if cuda else torch.float32,
            )
            pipe = pipe.to("cuda" if cuda else "cpu")

            logger.info(
                "ImageGenRunner: loaded model=%s dir=%s device=%s",
                self.model_key, model_dir, "cuda" if cuda else "cpu",
            )
            self._PIPELINES[self.model_key] = pipe
            return pipe

    # --- generation ---------------------------------------------------------

    def _generate(self, req: ImageGenRequest) -> list[GeneratedImage]:
        """Blocking generate. Called from a worker thread by .run().

        Only explicitly-set request fields reach the pipeline call, so the
        pipeline's per-model defaults govern everything the caller left out.
        """
        import torch

        call_kwargs: Dict[str, Any] = {
            "prompt": req.prompt,
            "num_images_per_prompt": req.num_images,
        }
        for field in ("negative_prompt", "width", "height",
                      "num_inference_steps", "guidance_scale"):
            value = getattr(req, field)
            if value is not None:
                call_kwargs[field] = value
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            call_kwargs["generator"] = torch.Generator(device).manual_seed(req.seed)

        with _generate_lock(self.model_key):
            output = self.pipeline(**call_kwargs)

        out_dir = os.path.join(UPLOADS_HOME, "generated")
        os.makedirs(out_dir, exist_ok=True)

        images: list[GeneratedImage] = []
        for index, image in enumerate(output.images):
            path = os.path.join(out_dir, f"{req.request_id}_{index}.png")
            image.save(path, format="PNG")
            b64 = None
            if req.return_b64:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(GeneratedImage(
                path=path, b64=b64,
                width=image.width, height=image.height,
                seed=req.seed,
            ))
        return images

    # --- public API ---------------------------------------------------------

    async def run(self, req: ImageGenRequest) -> ImageGenResult:
        try:
            images = await asyncio.to_thread(self._generate, req)
            return ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                images=images,
                text=(f"generated {len(images)} image(s): "
                      + ", ".join(img.path for img in images)),
            )
        except Exception as exc:
            logger.exception(
                "ImageGenRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, req: ImageGenRequest, cancel_event=None):
        """One-shot wrapped as a stream, mirroring VisionRunner."""
        result = await self.run(req)
        if result.ok:
            yield TokenEvent(request_id=req.request_id, text=result.text)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                            output_chunks=1, finish_reason="stop")
        else:
            yield ErrorEvent(request_id=req.request_id,
                             message=result.error or "image generation failed")


class Img2ImgRunner:
    """Runner for diffusers image-to-image (img2img) pipelines.

    SIBLING of ImageGenRunner — same lazy/singleton/thread-offload pattern, but
    it drives ``AutoPipelineForImage2Image`` and conditions generation on an init
    image (req.image_path) with an optional denoising ``strength``. It REUSES
    ImageGenRequest/ImageGenResult (the remote factory wrappers copy request/
    result types straight off FRAMEWORK_RUNNERS, so reusing them means the worker
    offload path needs zero changes).

    Its pipeline cache is its OWN (_PIPELINES) — the img2img pipeline object is a
    different class from text2img's, so it must not share the text2img cache.

    INERT until a model advertises ("transformers","image-to-image"): the sd-turbo
    advertisement flip is HELD (see models_config.py) so live central never routes
    img2img to the old-wheel GPU worker.
    """

    request_type = ImageGenRequest
    result_type = ImageGenResult

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs

    # --- pipeline loading (lazy, singleton) ---------------------------------

    @property
    def pipeline(self):
        cached = self._PIPELINES.get(self.model_key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._PIPELINES.get(self.model_key)
            if cached is not None:
                return cached

            try:
                import torch
                from diffusers import AutoPipelineForImage2Image, DiffusionPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "diffusers + torch are required for image-to-image tasks "
                    "but are not installed. `pip install diffusers torch`."
                ) from exc

            model_dir = ensure_model(self.model_key)
            # GPU guard block — VERBATIM from ImageGenRunner: fp16 on cuda, fp32
            # on cpu, move the whole pipe to the resolved device.
            cuda = torch.cuda.is_available()
            dtype = torch.float16 if cuda else torch.float32

            # ---- fit ladder (cuda only) -------------------------------------
            # Weights far bigger than free VRAM (Qwen-Image-Edit: ~55GB bf16 vs
            # a 24GB card) can still serve: quantize ON LOAD to bnb 4-bit
            # (~4.5x smaller — the 20B transformer lands ~11GB) and let
            # cpu-offload spill the rest to host RAM. Opt out / force via
            # HUGPY_IMG2IMG_QUANTIZE = "auto" (default) | "always" | "never".
            quant_config = None
            if cuda:
                mode = (os.environ.get("HUGPY_IMG2IMG_QUANTIZE") or "auto").lower()
                weight_bytes = 0
                for root, _dirs, files in os.walk(model_dir):
                    for fn in files:
                        if fn.endswith((".safetensors", ".bin")):
                            try:
                                weight_bytes += os.path.getsize(os.path.join(root, fn))
                            except OSError:
                                pass
                free_vram = torch.cuda.mem_get_info()[0]
                need_quant = (mode == "always" or
                              (mode == "auto" and weight_bytes > free_vram * 0.85))
                if need_quant and mode != "never":
                    try:
                        import bitsandbytes  # noqa: F401 — availability probe
                        from diffusers import PipelineQuantizationConfig
                        quant_config = PipelineQuantizationConfig(
                            quant_backend="bitsandbytes_4bit",
                            quant_kwargs={
                                "load_in_4bit": True,
                                "bnb_4bit_quant_type": "nf4",
                                "bnb_4bit_compute_dtype": torch.bfloat16,
                            },
                            components_to_quantize=["transformer", "text_encoder"],
                        )
                        logger.warning(
                            "Img2ImgRunner: model=%s weights ~%.1fGB vs %.1fGB free "
                            "VRAM — loading in bnb 4-bit (nf4) to fit",
                            self.model_key, weight_bytes / 1e9, free_vram / 1e9,
                        )
                    except Exception as exc:  # noqa: BLE001 — no bnb/API: plain load
                        logger.warning(
                            "Img2ImgRunner: wanted 4-bit load for %s but the "
                            "quantization stack is unavailable (%s) — plain load "
                            "may OOM", self.model_key, exc,
                        )
                        quant_config = None
            load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
            # AutoPipelineForImage2Image only maps the classic families (SD /
            # SDXL / flux1 …); natively image-conditioned EDIT pipelines
            # (QwenImageEditPlusPipeline, Flux2KleinPipeline, …) are absent from
            # its mapping and it raises "can't find a pipeline linked to <cls>".
            # Those classes are img2img by construction, so fall back to
            # DiffusionPipeline, which instantiates the concrete class straight
            # from model_index.json.
            fallback = False
            try:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_dir, **load_kwargs,
                )
            except ValueError as exc:
                logger.info(
                    "Img2ImgRunner: AutoPipeline has no img2img mapping for "
                    "model=%s (%s); falling back to the concrete pipeline class",
                    self.model_key, exc,
                )
                pipe = DiffusionPipeline.from_pretrained(model_dir, **load_kwargs)
                fallback = True
            if cuda and (fallback or quant_config is not None):
                # Edit pipelines are typically far larger than the classic SD
                # families — component-wise CPU offload spills non-active
                # components to host RAM so a single consumer GPU serves them
                # instead of OOMing at .to("cuda"). (bnb-quantized components
                # are already device-placed; offload handles the rest.)
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    try:
                        pipe = pipe.to("cuda")
                    except Exception:
                        pass  # quantized components may already sit on device
            else:
                pipe = pipe.to("cuda" if cuda else "cpu")

            logger.info(
                "Img2ImgRunner: loaded model=%s dir=%s device=%s class=%s%s",
                self.model_key, model_dir, "cuda" if cuda else "cpu",
                type(pipe).__name__, " (cpu-offload)" if cuda and fallback else "",
            )
            self._PIPELINES[self.model_key] = pipe
            return pipe

    # --- input helpers -------------------------------------------------------

    def _load_init_image(self, req: ImageGenRequest):
        """Load the init image (mirrors VisionAnalysisRunner._load_image). A
        clean error (raised here, caught by run() into an ok=False result) when
        no init image was provided — img2img has nothing to condition on."""
        from PIL import Image
        if not req.image_path:
            raise ValueError(
                "image-to-image requires an init image (image_path); none provided"
            )
        return Image.open(req.image_path).convert("RGB")

    # --- generation ---------------------------------------------------------

    def _generate(self, req: ImageGenRequest) -> list[GeneratedImage]:
        """Blocking img2img generate. Called from a worker thread by .run().

        Mirrors ImageGenRunner._generate but conditions on an init image. Only
        explicitly-set request fields reach the pipeline call.
        """
        import torch

        init_img = self._load_init_image(req)
        # The SD img2img pipeline derives the output size from the init image
        # (its __call__ takes no width/height), so honor the requested dims by
        # RESIZING the init here. This also keeps every chained scene frame the
        # same size, which the mp4 mux requires.
        if req.width is not None and req.height is not None:
            init_img = init_img.resize((req.width, req.height))

        call_kwargs: Dict[str, Any] = {
            "prompt": req.prompt,
            "num_images_per_prompt": req.num_images,
            "image": init_img,
        }
        # width/height are handled via the resize above (the pipeline ignores
        # them), so they are intentionally NOT forwarded here.
        for field in ("negative_prompt", "num_inference_steps", "guidance_scale"):
            value = getattr(req, field)
            if value is not None:
                call_kwargs[field] = value
        if req.strength is not None:
            call_kwargs["strength"] = req.strength

        # sd-turbo numeric edge: diffusers computes effective steps as
        # int(num_inference_steps * strength) and RAISES when that is 0. sd-turbo
        # runs 1-4 steps, so a low strength (e.g. steps=2 * strength=0.3 -> 0
        # effective) detonates. Bump steps so int(steps*strength) >= 1 and log
        # LOUDLY, rather than letting the pipeline raise.
        steps = call_kwargs.get("num_inference_steps")
        strength = call_kwargs.get("strength")
        if (steps is not None and strength is not None
                and strength > 0 and int(steps * strength) < 1):
            import math
            bumped = int(math.ceil(1.0 / strength))
            logger.warning(
                "Img2ImgRunner: num_inference_steps=%s * strength=%s -> %d "
                "effective steps (0 raises in diffusers); bumping steps %s -> %d "
                "for model=%s", steps, strength, int(steps * strength),
                steps, bumped, self.model_key,
            )
            call_kwargs["num_inference_steps"] = bumped

        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            call_kwargs["generator"] = torch.Generator(device).manual_seed(req.seed)

        # Concrete edit pipelines (QwenImageEditPlus, Flux2Klein, …) don't share
        # the SD img2img signature — e.g. no `strength`, `true_cfg_scale` in
        # place of guidance. Filter kwargs to what THIS pipeline's __call__
        # actually accepts (a **kwargs pipeline keeps everything) and log the
        # drops, instead of detonating on an unexpected-keyword TypeError.
        import inspect
        pipe = self.pipeline
        try:
            sig = inspect.signature(pipe.__call__)
            has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())
            if not has_var_kw:
                accepted = set(sig.parameters)
                dropped = [k for k in call_kwargs if k not in accepted]
                if dropped:
                    logger.warning(
                        "Img2ImgRunner: %s.__call__ does not accept %s — "
                        "dropping for model=%s",
                        type(pipe).__name__, dropped, self.model_key,
                    )
                    call_kwargs = {k: v for k, v in call_kwargs.items()
                                   if k in accepted}
        except (TypeError, ValueError):
            pass  # unsignaturable callable — send as-is

        with _generate_lock(self.model_key):
            output = pipe(**call_kwargs)

        out_dir = os.path.join(UPLOADS_HOME, "generated")
        os.makedirs(out_dir, exist_ok=True)

        images: list[GeneratedImage] = []
        for index, image in enumerate(output.images):
            path = os.path.join(out_dir, f"{req.request_id}_{index}.png")
            image.save(path, format="PNG")
            b64 = None
            if req.return_b64:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(GeneratedImage(
                path=path, b64=b64,
                width=image.width, height=image.height,
                seed=req.seed,
            ))
        return images

    # --- public API ---------------------------------------------------------

    async def run(self, req: ImageGenRequest) -> ImageGenResult:
        try:
            images = await asyncio.to_thread(self._generate, req)
            return ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                images=images,
                text=(f"generated {len(images)} image(s): "
                      + ", ".join(img.path for img in images)),
            )
        except Exception as exc:
            logger.exception(
                "Img2ImgRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, req: ImageGenRequest, cancel_event=None):
        """One-shot wrapped as a stream, mirroring ImageGenRunner."""
        result = await self.run(req)
        if result.ok:
            yield TokenEvent(request_id=req.request_id, text=result.text)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                            output_chunks=1, finish_reason="stop")
        else:
            yield ErrorEvent(request_id=req.request_id,
                             message=result.error or "image-to-image generation failed")
