"""Runner dispatch table — map §6/§8: (framework, task) -> pure runner fn.

Registries over globals. The worker looks up JOB_REGISTRY[name].runner_key and
calls DISPATCH[runner_key](spec, job_id). Runners are pure `spec -> JobResult`
(they also take the job_id so the JobResult can carry it).

Phase 4 landed frame_extract + generate_image; Phase 5 landed audio_extract.
All runner keys are wired below:
    ("ffmpeg", "crop"): run_crop,
    ("ffmpeg", "frame_extract"): run_frame_extract,
    ("ffmpeg", "audio_extract"): run_audio_extract,
    ("diffusers", "generate_image"): run_generate_image,
"""
from __future__ import annotations

from .ffmpeg_audio import run_audio_extract
from .ffmpeg_crop import run_crop
from .ffmpeg_frames import run_frame_extract
from .imagegen import run_generate_image

DISPATCH = {
    ("ffmpeg", "crop"): run_crop,
    ("ffmpeg", "frame_extract"): run_frame_extract,
    ("ffmpeg", "audio_extract"): run_audio_extract,
    ("diffusers", "generate_image"): run_generate_image,
}
