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
