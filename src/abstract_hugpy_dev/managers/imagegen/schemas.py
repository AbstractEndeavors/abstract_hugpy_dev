"""Schemas for the text-to-image task.

Generation parameters are all optional: anything the caller doesn't set is
omitted from the diffusers pipeline call, so the pipeline's own per-model
defaults apply (sdxl-turbo wants 1-4 steps and guidance 0.0; SD 1.5 wants
~50 steps and guidance 7.5 — neither needs the caller to know that).
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeneratedImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    b64: Optional[str] = None       # base64 PNG bytes, omitted when return_b64=False
    width: int
    height: int
    seed: Optional[int] = None


class ImageGenRequest(BaseModel):
    """One unit of text-to-image work. Built per call."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general
    prompt: str = Field(min_length=1)

    negative_prompt: Optional[str] = None
    width: Optional[int] = Field(default=None, ge=64, le=4096, multiple_of=8)
    height: Optional[int] = Field(default=None, ge=64, le=4096, multiple_of=8)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    # --- ComfyUI sampler plumbing (additive) ---
    # Read only by the comfy runner (managers/comfy/comfy_runner.py); the
    # diffusers runners ignore them. None -> the runner's historical
    # euler/normal defaults, so requests without them behave exactly as before.
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    # --- img2img (image-to-image) additive fields ---
    # text-to-image callers never set these (extra="ignore" on the frozen model
    # keeps them absent for the text2img path); the img2img runner + builder read
    # them. image_path rides remote._PATH_KEYS so the worker inliner rebuilds it.
    image_path: Optional[str] = None       # init image; rides remote._PATH_KEYS inliner
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    num_images: int = Field(default=1, ge=1, le=4)
    # b64 in the result is what lets HTTP callers (the discord bot) fetch the
    # bytes without a second round-trip; set False for in-process callers that
    # only want the saved paths.
    return_b64: bool = True


class ImageGenResult(BaseModel):
    request_id: str
    model_key: str
    ok: bool = True
    images: List[GeneratedImage] = Field(default_factory=list)
    # Human-readable summary so chat-stream wrapping (stream_runner's one-shot
    # path reads result.text) degrades to something sensible.
    text: str = ""
    error: Optional[str] = None
