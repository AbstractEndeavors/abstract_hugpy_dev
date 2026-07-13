"""Runner dispatch table — map §6/§8: (framework, task) -> pure runner fn.

Registries over globals. The worker looks up JOB_REGISTRY[name].runner_key and
calls DISPATCH[runner_key](spec, job_id). Runners are pure `spec -> JobResult`
(they also take the job_id so the JobResult can carry it).

Phase 4 landed frame_extract + generate_image; Phase 5 landed audio_extract;
generate_scene (one query -> N consecutive frames + optional mp4) is wired below.
All runner keys are wired below:
    ("ffmpeg", "crop"): run_crop,
    ("ffmpeg", "frame_extract"): run_frame_extract,
    ("ffmpeg", "audio_extract"): run_audio_extract,
    ("diffusers", "generate_image"): run_generate_image,
    ("diffusers", "generate_scene"): run_generate_scene,
"""
from __future__ import annotations

from .ffmpeg_audio import run_audio_extract
from .ffmpeg_crop import run_crop
from .ffmpeg_frames import run_frame_extract
from .imagegen import run_generate_image
from .movie import run_generate_movie
from .scene import run_generate_scene
# B2: studio i2v — the media bus's seam to the studio spine (produce_clip). Its
# module top is dependency-light (studio/numpy imports are lazy inside the runner),
# so this import can never break app boot.
from .studio_i2v import run_studio_i2v
# Studio movie — the fat orchestrator that renders an ordered strip of studio clips
# INLINE through the produce_clip spine. Import-safe like studio_i2v (studio/numpy
# imports stay lazy inside the runner), so this never breaks app boot.
from .studio_movie import run_generate_studio_movie
# Identity reconstruction (studio stage (b)) — the orchestrator that renders an
# identity-locked turnaround set from a profile + description. Import-safe like the
# studio runners (studio/media_store imports stay lazy inside the runner).
from .identity_reconstruction import run_identity_reconstruction

DISPATCH = {
    ("ffmpeg", "crop"): run_crop,
    ("ffmpeg", "frame_extract"): run_frame_extract,
    ("ffmpeg", "audio_extract"): run_audio_extract,
    ("diffusers", "generate_image"): run_generate_image,
    ("diffusers", "generate_scene"): run_generate_scene,
    ("diffusers", "generate_movie"): run_generate_movie,
    ("studio", "i2v"): run_studio_i2v,
    ("studio", "movie"): run_generate_studio_movie,
    ("identity", "reconstruction"): run_identity_reconstruction,
}
