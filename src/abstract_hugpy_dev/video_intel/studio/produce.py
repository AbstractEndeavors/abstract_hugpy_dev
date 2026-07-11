"""End-to-end studio production (P0-B1): a ``CapabilityRequest`` in, a playable
content-addressed clip out, entirely through the studio's own spine.

``produce_clip`` is the thin conductor that wires the pieces already built:

    request --router.resolve--> ModelBinding
            --make_render_manifest(binding, env, seeds, sampler, ladder)--> RenderManifest
            --dispatch on (framework, task)--> runner(manifest, out_root, start_image)
            --> Result[Artifact, StageError]

Errors are data end to end (INV-3): an unroutable request propagates the router's
``Err`` verbatim; a missing runner returns ``Err(RUNNER_MISSING)``; the runner's
own IO/assembly failures ride back as ``Err``. Nothing here raises on a runtime
policy failure. This module is deliberately NOT imported by ``studio/__init__``
or the media bus — it is the studio-internal proof path for this slice.

No pathlib anywhere. os.path only (there is none here — pure orchestration).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from .artifacts import Artifact
from .enums import Framework, Task
from .env import StudioEnv
from .errors import Err, ErrorCode, Result, StageError
from .manifest import make_render_manifest
from .models_seed import FAMILY_SAMPLER_DEFAULTS, PLACEHOLDER_SAMPLER_DEFAULTS
from .router import CapabilityRouter
from .runners.ffmpeg_enhance import run_ffmpeg_interpolate, run_ffmpeg_upscale
from .runners.ltx_upscale import run_ltx_upscale
from .runners.rife_interpolate import run_rife_interpolate
from .runners.synthetic import run_synthetic_i2v, run_synthetic_t2v
from .runners.wan_i2v import run_wan_i2v
from .runners.wan_t2v import run_wan_t2v
from .runners.wan_vace import run_wan_vace
from .schemas import (
    CapabilityRequest,
    ProvenanceStub,
    Resolution,
    SamplerConfig,
    SeedBundle,
)

# Explicit dispatch table: (framework, task) -> runner callable. SYNTHETIC proves
# the spine with no GPU/weights (i2v AND t2v — Task 3b); WAN i2v/t2v are the real-
# model runners — IMPORT-SAFE (torch/diffusers are lazy inside them) and returning
# Err-as-data (DEPS_MISSING / NO_GPU / WEIGHTS_MISSING) on a box that can't run
# them yet, so wiring them here never breaks a GPU-less path. The t2v runners are
# thin DRY delegations (start_image forced None) to their i2v siblings' text
# branches. A binding whose (framework, task) is absent returns Err(RUNNER_MISSING)
# rather than importing an unwired path.
_DISPATCH = {
    (Framework.SYNTHETIC, Task.I2V): run_synthetic_i2v,
    (Framework.SYNTHETIC, Task.T2V): run_synthetic_t2v,
    (Framework.WAN, Task.I2V): run_wan_i2v,
    (Framework.WAN, Task.T2V): run_wan_t2v,
    # B-3: WAN VACE v2v — the studio's first REAL enhancement path. A v2v binding
    # (restyle/enhance an existing clip) dispatches here; run_wan_vace is import-safe
    # (torch/diffusers lazy) and returns Err-as-data on a box that can't run it yet.
    (Framework.WAN, Task.VACE_CONTROL): run_wan_vace,
    # slice b: INTERP + UPRES enhancement paths. FFMPEG is the REAL last-resort
    # (minterpolate / scale=lanczos via the system binary — works on this GPU-less
    # box today); RIFE / LTX are the premium weight-backed runners, import-safe and
    # returning Err-as-data (DEPS_MISSING / WEIGHTS_MISSING) until their assets are
    # staged. All four are dispatched here so any interp/upres binding is executable.
    (Framework.FFMPEG, Task.INTERPOLATE): run_ffmpeg_interpolate,
    (Framework.FFMPEG, Task.UPSCALE): run_ffmpeg_upscale,
    (Framework.RIFE, Task.INTERPOLATE): run_rife_interpolate,
    (Framework.LTX, Task.UPSCALE): run_ltx_upscale,
}

_DEFAULT_SEED = 0


def _default_seeds() -> SeedBundle:
    return SeedBundle(global_seed=_DEFAULT_SEED, stage_seeds=(("base", _DEFAULT_SEED),))


# Flow-match / UniPC scheduler SHIFT is resolution-dependent (the Wan reference:
# 3.0 @ 480p, 5.0 @ 720p+). Threshold on the LONG edge so both landscape 720p
# (1280x720) and portrait 720p (720x1280) trip 720p — a simple threshold, not a
# per-format table (operator: "simple thresholds, no over-engineering").
_SHIFT_720P = 5.0
_SHIFT_480P = 3.0
_720P_LONG_EDGE = 1280


def _default_sampler() -> SamplerConfig:
    # Historical placeholder (synthetic / ffmpeg-rife enhancers never sample). RETAINED
    # as the base ``resolve_sampler`` builds on for families with no real defaults; a
    # real family (FAMILY_SAMPLER_DEFAULTS) overrides steps/cfg/shift. Built FROM the
    # models_seed data table so the placeholder has ONE source of truth. steps=1/cfg=1.0
    # = a no-op denoise — honest for a runner that does not sample, and byte-identical
    # to every prior synthetic/enhancer content-addressed clip.
    p = PLACEHOLDER_SAMPLER_DEFAULTS
    return SamplerConfig(
        sampler=p["sampler"], scheduler=p["scheduler"], steps=p["steps"], cfg=p["cfg"])


def resolve_sampler(
    framework: Framework,
    target_resolution: Resolution,
    *,
    steps: int | None = None,
    cfg: float | None = None,
) -> SamplerConfig:
    """The SamplerConfig a render uses when the spec did NOT pass a full SamplerConfig.

    Data-over-code: the per-family denoise defaults live in ``models_seed``'s
    ``FAMILY_SAMPLER_DEFAULTS`` (a REAL family like Wan declares steps 32 / cfg 5.0 /
    UniPC flow-match; a family ABSENT from the table — synthetic, ffmpeg, rife — falls
    back to the steps=1 / cfg=1.0 placeholder, unchanged). ``shift`` is derived from the
    target resolution (720p+ -> 5.0, else 3.0) for real families only; placeholder
    families keep shift=None (their runners ignore the sampler entirely).

    Explicit ``steps`` / ``cfg`` (from the /video/studio/i2v route, threaded via the
    spec) ALWAYS win over the model default — None means "unset, use the default". The
    returned SamplerConfig is RECORDED verbatim in the manifest (content_hash keys on
    it), so the runner denoises with EXACTLY these values — never a value that differs
    from the manifest (the studio's INV-1 hard rule)."""
    fam = FAMILY_SAMPLER_DEFAULTS.get(framework)
    if fam is None:
        base = _default_sampler()          # placeholder family (synthetic / enhancers)
    else:
        long_edge = max(target_resolution.width, target_resolution.height)
        shift = _SHIFT_720P if long_edge >= _720P_LONG_EDGE else _SHIFT_480P
        base = SamplerConfig(
            sampler=fam["sampler"], scheduler=fam["scheduler"],
            steps=fam["steps"], cfg=fam["cfg"], shift=shift)
    return SamplerConfig(
        sampler=base.sampler,
        scheduler=base.scheduler,
        steps=steps if steps is not None else base.steps,
        cfg=cfg if cfg is not None else base.cfg,
        shift=base.shift,
    )


def produce_clip(
    request: CapabilityRequest,
    *,
    env: StudioEnv,
    out_root: str,
    seeds: SeedBundle | None = None,
    sampler: SamplerConfig | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    start_image: str | None = None,
    prompt: str = "",
    negative_prompt: str = "",
    source_video: str | None = None,
    reference_images: tuple[str, ...] | None = None,
    control_image: str | None = None,
    control_kind: str | None = None,
    vace_context_frames: tuple[str, ...] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Result[Artifact, StageError]:
    """Resolve ``request``, build its manifest, and run the bound runner.

    Returns ``Ok(Artifact)`` on success or ``Err(StageError)`` on any expected
    failure (unroutable request, missing runner, runner IO/assembly failure).

    SAMPLER precedence (see the resolve_sampler call below): an explicit ``sampler``
    (a whole SamplerConfig) is used verbatim; otherwise the bound model's family
    default fills it, with the scalar ``steps`` / ``cfg`` overrides applied on top
    (None = unset). A pinned model is carried on ``request.pinned_model_id`` and honored
    (or clearly refused) by the router — produce_clip does not special-case it.

    ``should_cancel`` is an OPTIONAL cooperative-cancel probe threaded DOWN to the
    runner: a zero-arg callable returning True once the job should stop. The studio
    spine never sources it (that would couple the spine to the media bus) — the bus
    adapter (``video_intel/runners/studio_i2v.py``) supplies it as
    ``lambda: media_bus.is_cancelling(job_id)``. When it fires mid-render the runner
    aborts BEFORE writing a clip and returns ``Err(StageError(CANCELLED, ...))``, so
    resume/idempotency stay intact (a cancelled run leaves no clip -> a re-run
    regenerates). None (the default) is the historical no-cancel behavior."""
    binding_res = CapabilityRouter().resolve(request)
    if binding_res.is_err():
        return binding_res            # propagate the router's Err verbatim (INV-3)
    binding = binding_res.unwrap()

    # MODEL-AWARE SAMPLER DEFAULTS. The router has now BOUND a concrete model, so we can
    # fill the denoise settings from the bound model's FAMILY when the caller didn't set
    # them — the fix for "synthetic-era steps=1/cfg=1.0 reaching the REAL runner" (gray
    # mush: one unguided step). Precedence, highest first:
    #   1. an explicit full ``sampler`` (a whole SamplerConfig) — used verbatim (the
    #      studio-internal tests' full-control path; scalar steps/cfg are ignored then).
    #   2. otherwise resolve_sampler(bound family, resolution) with the scalar steps/cfg
    #      overrides applied on top (explicit route values ALWAYS win over the default).
    # A synthetic / ffmpeg / rife binding keeps steps=1/cfg=1.0 (its family is absent
    # from the defaults table) — so synthetic renders are unchanged (back-compat), while
    # a Wan binding gets real steps 32 / cfg 5.0 / flow-shift. Whatever is chosen is the
    # SamplerConfig recorded in the manifest and denoised with (INV-1: no divergence).
    resolved_sampler = (
        sampler if sampler is not None
        else resolve_sampler(
            binding.framework, request.target_resolution, steps=steps, cfg=cfg)
    )

    manifest = make_render_manifest(
        render_id=uuid.uuid4().hex,
        capability=request.capability,
        binding=binding,
        seeds=seeds if seeds is not None else _default_seeds(),
        sampler=resolved_sampler,
        resolution_ladder=(request.target_resolution,),
        env=env,
        provenance=ProvenanceStub(
            operator="hugpy-studio",
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        # C-prompt: text conditioning threaded into the manifest (and thus its
        # content_hash). None (an Optional spec field / absent JSON key) -> "" is
        # normalized in the manifest factory.
        prompt=prompt,
        negative_prompt=negative_prompt,
        # B2 chain: the source clip this render extends, threaded into the manifest
        # (and thus its content_hash + the manifest.json sidecar). CARRIED for every
        # capability; the i2v runners CONSUME it (extend from its last frame when no
        # start_image is given) — see run_synthetic_i2v / run_wan_i2v. None -> "".
        source_video=source_video or "",
        # IDENTITY LOCK (id_lock): reference image paths + optional VACE control still,
        # threaded into the manifest (canonical inputs). CARRIED for every capability;
        # the VACE runner CONSUMES them (reference-to-video / control channel). None -> ()/"".
        reference_images=tuple(reference_images or ()),
        control_image=control_image or "",
        control_kind=control_kind or "",
        # VACE-EXTEND temporal conditioning (studio-movie splice motion-carry): the
        # parent clip's trailing context frames, threaded into the manifest for the VACE
        # runner to build the diffusers video+mask extend idiom. CARRIED for every
        # capability (the VACE runner CONSUMES it; others ignore it), but — unlike the
        # other conditioning inputs — NOT part of the content_hash (see the field docstring
        # on RenderManifest). None -> ().
        vace_context_frames=tuple(vace_context_frames or ()),
    )

    runner = _DISPATCH.get((binding.framework, binding.task))
    if runner is None:
        return Err(StageError(
            ErrorCode.RUNNER_MISSING,
            f"no wired runner for ({binding.framework.value}, {binding.task.value})",
            (("model_id", binding.model_id),),
        ))

    # Thread the cancel probe only when a caller supplied one, so the runner
    # contract stays backward-compatible with any 3-arg dispatch shim that predates
    # cooperative cancel (tests monkeypatch _DISPATCH). The wired runners all accept
    # should_cancel=None; this just avoids forwarding it to shims that don't.
    runner_kwargs = {"start_image": start_image}
    if should_cancel is not None:
        runner_kwargs["should_cancel"] = should_cancel
    return runner(manifest, out_root, **runner_kwargs)
