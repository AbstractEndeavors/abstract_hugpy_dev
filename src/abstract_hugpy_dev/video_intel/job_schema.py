"""Job envelope + registry — map §5.

Registry over globals: JOB_REGISTRY maps a job `name` to its frozen JobSpec
(spec_type, runner_key = (framework, task), queue, timeout_s). The worker looks
up runner_key here and dispatches through the runner DISPATCH table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Type

from .audio_schema import AudioExtractSpec
from .crop_schema import CropSpec
from .frame_schema import FrameExtractSpec
from .gen_schema import GenerateImageSpec
from .movie_schema import MovieSpec
from .scene_schema import GenerateSceneSpec


@dataclass(frozen=True)
class JobSpec:
    name: str
    spec_type: Type                       # one of the frozen specs
    runner_key: Tuple[str, str]           # (framework, task)
    queue: str
    timeout_s: int


# Registry, not globals. Phase 1/2 registered "crop"; Phase 4 landed
# frame_extract + generate_image; Phase 5 landed audio_extract. Every job's
# spec type must exist as an import above before it can be registered here.
JOB_REGISTRY = {
    "crop": JobSpec("crop", CropSpec, ("ffmpeg", "crop"), "media", 300),  # runner branches spatial/temporal
    "frame_extract": JobSpec("frame_extract", FrameExtractSpec, ("ffmpeg", "frame_extract"), "media", 600),
    "audio_extract": JobSpec("audio_extract", AudioExtractSpec, ("ffmpeg", "audio_extract"), "media", 300),
    "generate_image": JobSpec("generate_image", GenerateImageSpec, ("diffusers", "generate_image"), "gpu", 900),
    "generate_scene": JobSpec("generate_scene", GenerateSceneSpec, ("diffusers", "generate_scene"), "gpu", 3600),
    # Movie = a SEQUENCE of scene segments; the fat orchestrator sequences them
    # inline, so its wall-clock is (segments × per-scene) — a longer timeout.
    "generate_movie": JobSpec("generate_movie", MovieSpec, ("diffusers", "generate_movie"), "gpu", 14400),
}
