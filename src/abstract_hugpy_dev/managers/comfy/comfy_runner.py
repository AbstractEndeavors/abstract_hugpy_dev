"""ComfyRunner — drive a LOCAL ComfyUI instance as a hugpy engine (slice B).

The operator installs ComfyUI on the worker box (slice A adopts + advertises
it); this runner turns it into a registry engine: `("comfy", task)` rows route
here, a VANILLA-nodes workflow template is built per request (no custom nodes
in the base templates), submitted to ComfyUI's HTTP API, and the artifacts are
returned as ImageGenResult — so the remote/delegation factories, the b64
artifact seam, the video arm, and the scene fan-out all work unchanged.

Request/result REUSE ImageGenRequest/ImageGenResult (the Img2ImgRunner
precedent): the checkpoint is the registry row's ``filename`` (a file inside
ComfyUI's own models/checkpoints — hugpy holds no files for comfy models, by
design).

API surface used (all vanilla ComfyUI):
    POST /prompt {"prompt": workflow, "client_id"}   -> {"prompt_id"}
    GET  /history/<prompt_id>                        -> outputs when done
    GET  /view?filename&subfolder&type=output        -> image bytes
    POST /upload/image (multipart)                   -> init image for img2img
    POST /interrupt                                  -> cancel (future wiring)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from typing import Any, Dict

from ..imagegen.schemas import GeneratedImage, ImageGenRequest, ImageGenResult
from ...imports.src.constants.constants import UPLOADS_HOME

logger = logging.getLogger(__name__)

_POLL_S = 1.0
_TIMEOUT_S = float(os.environ.get("COMFY_TIMEOUT_S", "600"))


def _comfy_url() -> str:
    return (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


def _t2i_workflow(ckpt: str, req: ImageGenRequest, seed: int) -> Dict[str, Any]:
    """Vanilla text-to-image graph: checkpoint -> CLIP pos/neg -> KSampler ->
    VAEDecode -> SaveImage."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": req.prompt or ""}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1],
                         "text": req.negative_prompt or ""}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": req.width or 512, "height": req.height or 512,
                         "batch_size": req.num_images or 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "seed": seed,
                         "steps": req.num_inference_steps or 20,
                         "cfg": req.guidance_scale if req.guidance_scale is not None else 7.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "hugpy"}},
    }


def _i2i_workflow(ckpt: str, req: ImageGenRequest, seed: int,
                  uploaded_name: str) -> Dict[str, Any]:
    """Vanilla img2img graph: LoadImage -> VAEEncode -> KSampler(denoise=
    strength) — same tail as t2i."""
    wf = _t2i_workflow(ckpt, req, seed)
    strength = req.strength if req.strength is not None else 0.6
    wf["8"] = {"class_type": "LoadImage", "inputs": {"image": uploaded_name}}
    wf["9"] = {"class_type": "VAEEncode",
               "inputs": {"pixels": ["8", 0], "vae": ["1", 2]}}
    wf["5"]["inputs"]["latent_image"] = ["9", 0]
    wf["5"]["inputs"]["denoise"] = max(0.05, min(1.0, float(strength)))
    del wf["4"]  # EmptyLatentImage replaced by the encoded init image
    return wf


class ComfyRunner:
    """Runner for ComfyUI-backed image tasks — text-to-image + image-to-image."""

    request_type = ImageGenRequest
    result_type = ImageGenResult

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        # The checkpoint FILE NAME inside ComfyUI's models/checkpoints — the
        # registry row's `filename` is the designation.
        self.checkpoint = getattr(cfg, "filename", None)
        if not self.checkpoint:
            raise ValueError(
                f"{self.model_key}: comfy model rows must set `filename` to the "
                "checkpoint name inside ComfyUI's models/checkpoints")

    # -- HTTP helpers --------------------------------------------------------
    def _upload_init_image(self, client, path: str) -> str:
        with open(path, "rb") as fh:
            files = {"image": (os.path.basename(path), fh, "image/png")}
            r = client.post(_comfy_url() + "/upload/image",
                            files=files, data={"overwrite": "true"})
        r.raise_for_status()
        return r.json()["name"]

    def _generate(self, req: ImageGenRequest) -> "list[GeneratedImage]":
        import httpx

        seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big")
        with httpx.Client(timeout=30.0) as client:
            if req.image_path:
                uploaded = self._upload_init_image(client, req.image_path)
                workflow = _i2i_workflow(self.checkpoint, req, seed, uploaded)
            else:
                workflow = _t2i_workflow(self.checkpoint, req, seed)

            r = client.post(_comfy_url() + "/prompt", json={
                "prompt": workflow, "client_id": f"hugpy-{uuid.uuid4().hex[:8]}"})
            if r.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI rejected the workflow ({r.status_code}): "
                    f"{r.text[:400]}")
            prompt_id = r.json()["prompt_id"]
            logger.info("ComfyRunner: %s submitted prompt %s (ckpt=%s, %s)",
                        self.model_key, prompt_id, self.checkpoint,
                        "img2img" if req.image_path else "text2img")

            # Poll history until outputs arrive (generation runs server-side).
            deadline = time.time() + _TIMEOUT_S
            outputs = None
            while time.time() < deadline:
                h = client.get(_comfy_url() + f"/history/{prompt_id}").json()
                entry = h.get(prompt_id)
                if entry:
                    status = (entry.get("status") or {})
                    if status.get("status_str") == "error":
                        msgs = [m for m in (status.get("messages") or [])
                                if m and m[0] == "execution_error"]
                        detail = json.dumps(msgs[-1][1] if msgs else status)[:400]
                        raise RuntimeError(f"ComfyUI execution error: {detail}")
                    if entry.get("outputs"):
                        outputs = entry["outputs"]
                        break
                time.sleep(_POLL_S)
            if outputs is None:
                raise TimeoutError(
                    f"ComfyUI did not finish prompt {prompt_id} within "
                    f"{_TIMEOUT_S:.0f}s")

            out_dir = os.path.join(UPLOADS_HOME, "generated")
            os.makedirs(out_dir, exist_ok=True)
            images: list[GeneratedImage] = []
            index = 0
            for node_out in outputs.values():
                for im in node_out.get("images", []):
                    r = client.get(_comfy_url() + "/view", params={
                        "filename": im["filename"],
                        "subfolder": im.get("subfolder", ""),
                        "type": im.get("type", "output")})
                    r.raise_for_status()
                    data = r.content
                    path = os.path.join(out_dir, f"{req.request_id}_{index}.png")
                    with open(path, "wb") as fh:
                        fh.write(data)
                    b64 = (base64.b64encode(data).decode("ascii")
                           if req.return_b64 else None)
                    images.append(GeneratedImage(
                        path=path, b64=b64, width=req.width, height=req.height,
                        seed=seed))
                    index += 1
            if not images:
                raise RuntimeError("ComfyUI finished but produced no images")
            return images

    # -- public API (mirrors Img2ImgRunner) ----------------------------------
    async def run(self, req: ImageGenRequest) -> ImageGenResult:
        try:
            images = await asyncio.to_thread(self._generate, req)
            return ImageGenResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=True, images=images,
                text=f"generated {len(images)} image(s) via ComfyUI")
        except Exception as exc:  # noqa: BLE001 — errors are data
            logger.warning("ComfyRunner %s failed: %s", self.model_key, exc)
            return ImageGenResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, images=[], error=f"{type(exc).__name__}: {exc}")
