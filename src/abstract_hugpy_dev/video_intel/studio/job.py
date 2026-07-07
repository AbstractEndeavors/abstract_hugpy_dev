"""Studio i2v JOB SPEC — the durable, bus-carried intent for a studio clip (B2).

This is the media-bus currency for a studio image-to-video job: a small frozen,
validate-at-construction spec (house style mirrored from ``movie_schema.make_movie``
/ ``crop_schema.make_crop``) that carries everything a studio i2v job needs and
nothing that isn't JSON-safe. The bus serializes it via ``asdict`` -> ``json`` and
rehydrates it through ``studio_i2v_from_dict`` (reconstruct + RE-VALIDATE), exactly
like every other media job.

Deliberately DECOUPLED from the studio spine's rich value objects: the spec holds
plain primitives (capability as a string, geometry as ints), so it round-trips
through JSON with zero enum/dataclass ceremony. The bus RUNNER
(``video_intel/runners/studio_i2v.py``) is the one place those primitives are
lifted into ``CapabilityRequest`` / ``Resolution`` / ``SeedBundle`` and handed to
``produce_clip`` — keeping the studio spine importable-but-dormant at module top
(no numpy/PIL pulled into app boot from here).

``resolve_studio_env`` satisfies INV-5 by RESOLVING concrete paths (under the
media-store root) rather than demanding the operator set STUDIO_* env vars — a
worker enqueues a studio job with no environment wiring at all.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT

from .enums import Capability
from .env import StudioEnv

# Where studio clips land by default: a content-addressed tree UNDER the media
# store root, so the produced clip.mp4 is inside media_store.ingest's storage
# jail (DEFAULT_ROOT) and can be cataloged like any other media output.
STUDIO_ROOT = os.path.join(DEFAULT_ROOT, "video_intel", "studio")
DEFAULT_CLIPS_ROOT = os.path.join(STUDIO_ROOT, "clips")

# This slice wires exactly one studio runner path (SYNTHETIC i2v — the no-model
# spine prover). A budget below the smallest real i2v footprint (8GB) makes the
# router deterministically bind synthetic-i2v; that is the produce-a-clip default.
# A larger budget routes to a real model whose runner is not wired yet, which
# comes back as JobResult(ok=False, RUNNER_MISSING) — errors as data, never a raise.
_DEFAULT_VRAM_BUDGET_GB = 0.5

_VALID_CAPABILITIES = frozenset(c.value for c in Capability)


@dataclass(frozen=True)
class StudioI2VSpec:
    """Frozen currency of a studio i2v bus job. Built ONLY via ``make_studio_i2v``
    (validate-at-construction); the bus rehydrates it via ``studio_i2v_from_dict``,
    which re-validates through the same factory. All fields are JSON-safe
    primitives so ``asdict`` -> ``json.dumps`` round-trips cleanly."""
    capability: str          # a Capability value, e.g. "i2v"
    width: int
    height: int
    fps: int
    vram_budget_gb: float
    seed: int
    out_root: str            # output location (a dir under the media-store root)
    start_image: Optional[str] = None   # abs path to a still (i2v conditioning)
    negative: Optional[str] = None      # carried; synthetic ignores it
    prompt: Optional[str] = None        # carried; synthetic ignores it


def make_studio_i2v(
    *,
    capability: str = Capability.I2V.value,
    width: int,
    height: int,
    fps: int,
    vram_budget_gb: float = _DEFAULT_VRAM_BUDGET_GB,
    seed: int = 0,
    out_root: Optional[str] = None,
    start_image: Optional[str] = None,
    negative: Optional[str] = None,
    prompt: Optional[str] = None,
) -> StudioI2VSpec:
    """Validate every field and build the frozen ``StudioI2VSpec``. Raises
    ``ValueError``/``TypeError`` LOCALLY on any structural violation (house
    discipline: a structurally-invalid spec is programmer/caller error caught at
    the boundary, never carried across the bus). Runtime policy failures — an
    unroutable request, an unreadable start image — are NOT validated here; they
    surface as errors-as-data from ``produce_clip`` at run time."""
    if not (isinstance(capability, str) and capability in _VALID_CAPABILITIES):
        raise ValueError(
            f"capability must be one of {sorted(_VALID_CAPABILITIES)}; got {capability!r}")
    for name, val in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive int; got {val!r}")
    if not isinstance(vram_budget_gb, (int, float)) or isinstance(vram_budget_gb, bool) \
            or vram_budget_gb <= 0:
        raise ValueError(f"vram_budget_gb must be a positive number; got {vram_budget_gb!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {seed!r}")
    if start_image is not None and not (isinstance(start_image, str) and start_image.strip()):
        raise ValueError(f"start_image must be a non-empty string or None; got {start_image!r}")
    if negative is not None and not isinstance(negative, str):
        raise ValueError(f"negative must be a string or None; got {negative!r}")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError(f"prompt must be a string or None; got {prompt!r}")

    resolved_out = out_root if (isinstance(out_root, str) and out_root.strip()) \
        else DEFAULT_CLIPS_ROOT

    return StudioI2VSpec(
        capability=capability,
        width=width,
        height=height,
        fps=fps,
        vram_budget_gb=float(vram_budget_gb),
        seed=seed,
        out_root=os.path.abspath(resolved_out),
        start_image=start_image,
        negative=negative,
        prompt=prompt,
    )


def studio_i2v_from_dict(d: dict) -> StudioI2VSpec:
    """Rebuild a ``StudioI2VSpec`` from its ``asdict`` form, THROUGH the validating
    factory (mirrors ``movie_schema``'s deserialize-then-revalidate) so a rehydrated
    spec is re-checked, never trusted blind. Registered in
    ``media_bus.SPEC_DESERIALIZERS`` under the name ``"studio_i2v"``."""
    return make_studio_i2v(
        capability=d.get("capability", Capability.I2V.value),
        width=d["width"],
        height=d["height"],
        fps=d["fps"],
        vram_budget_gb=d.get("vram_budget_gb", _DEFAULT_VRAM_BUDGET_GB),
        seed=d.get("seed", 0),
        out_root=d.get("out_root"),
        start_image=d.get("start_image"),
        negative=d.get("negative"),
        prompt=d.get("prompt"),
    )


def resolve_studio_env(out_root: str, *, master_fps: int, max_vram_gb: float = 24.0) -> StudioEnv:
    """Resolve a concrete ``StudioEnv`` from sensible worker defaults (INV-5) WITHOUT
    requiring any STUDIO_* environment variable: every required field is filled with
    a resolved value (paths under the media-store root, house mastering defaults),
    so a bus runner constructs a complete env with no operator wiring. ``master_fps``
    is threaded from the job's target so the recorded env matches the render intent.
    ``allow_unpinned=True`` (dev posture); the synthetic runner is pinned regardless,
    and ``produce_clip`` never calls ``validate_registry``, so this only affects the
    recorded env snapshot, never routing."""
    return StudioEnv(
        output_root=os.path.abspath(out_root),
        weights_root=os.path.join(STUDIO_ROOT, "weights"),
        manifest_root=os.path.join(STUDIO_ROOT, "manifests"),
        master_colorspace="rec709",
        master_fps=int(master_fps),
        max_vram_gb=float(max_vram_gb),
        loudness_target_lufs=-14.0,
        allow_unpinned=True,
    )
