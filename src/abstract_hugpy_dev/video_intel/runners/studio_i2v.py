"""Studio i2v BUS RUNNER (B2) — the boundary between the media bus and the studio
spine. Pure ``(StudioI2VSpec, job_id) -> JobResult`` (map §6): expected failures
are DATA (a ``JobResult(ok=False, JobError(...))``), never a raise; only a genuine
programmer error would raise, and ``media_bus.run_claimed`` is the one place that
catches that.

It does four things and nothing else:
  1. Resolve a concrete ``StudioEnv`` from worker defaults (INV-5; no operator env).
  2. Lift the JSON-safe spec into a studio ``CapabilityRequest`` and call
     ``produce_clip`` — the studio's own spine (router -> manifest -> runner ->
     content-addressed clip). Resume (INV-6) is handled inside ``produce_clip``: an
     identical spec re-run reuses the existing clip, no regeneration.
  3. Translate the studio Result at THIS boundary — and only here — via
     ``_stage_error_to_job_error``: studio's ``StageError`` becomes the bus's own
     ``JobError``. Since the Task 2 collapse there is ONE JobError class
     (result_schema.JobError IS comms.jobs.JobError); studio's ``StageError`` is a
     SEPARATE studio-layer vocabulary that this seam ADAPTS into that unified
     JobError — a translation at the single boundary, not a merge, leaving studio's
     errors.py self-contained (StageError reconciliation is TODO(P0-1)).
  4. Ingest the produced ``clip.mp4`` into the media store so the clip is cataloged
     as a ``MediaRef`` (kind="video"), carried out on ``JobResult.outputs``.

Heavy studio imports (which transitively pull numpy/PIL via the synthetic runner)
are LAZY — done inside ``run_studio_i2v`` — so importing this module (which the bus
does at boot, via ``runners/__init__``) stays cheap and can never break app boot.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

from ..media_store import ingest
from ..result_schema import JobError, JobResult

# StageError codes worth a retry (transient/resource), vs. policy/routing failures
# where the SAME spec would deterministically fail again. Kept as the string values
# so this module needs no studio-enum import at top.
_RETRYABLE_CODES = frozenset({"oom", "nan_in_vae", "assembly_failed", "io_error"})


def _stage_error_to_job_error(stage_error) -> JobError:
    """BOUNDARY adapter: studio ``StageError`` -> bus ``JobError``. The ONE place the
    two error vocabularies meet; a translation, not a merge."""
    code = getattr(stage_error.code, "value", str(stage_error.code))
    return JobError(
        code=code,
        message=str(stage_error),
        retryable=code in _RETRYABLE_CODES,
    )


def run_studio_i2v(spec, job_id: str) -> JobResult:
    """Run a studio i2v job through ``produce_clip`` and return a ``JobResult``.

    ``Ok(Artifact)`` -> ``JobResult(ok=True, outputs=(clip MediaRef,))``; the ref
    carries the clip path (uri) + resolved geometry/duration + a minted asset id.
    ``Err(StageError)`` -> ``JobResult(ok=False, error=JobError(...))``. Nothing here
    raises on an expected failure."""
    # --- lazy studio-spine imports (keep module-top / app-boot dependency-free) ---
    from ..studio.job import resolve_studio_env
    from ..studio.produce import produce_clip
    from ..studio.schemas import CapabilityRequest, Resolution, SeedBundle
    from ..studio.enums import Capability
    # media_bus is the ONE video_intel dep this adapter is allowed to hold (the
    # studio spine must stay bus-free). Lazy-imported like scene.py/movie.py.
    from .. import media_bus

    request = CapabilityRequest(
        capability=Capability(spec.capability),
        target_resolution=Resolution(spec.width, spec.height, spec.fps),
        vram_budget_gb=spec.vram_budget_gb,
        # B2 chain: carry the source clip on the request (routing does not key on it;
        # produce_clip reads spec.source_video for the manifest + extend). None -> None.
        source_video=getattr(spec, "source_video", None),
    )
    env = resolve_studio_env(spec.out_root, master_fps=spec.fps)
    seeds = SeedBundle(global_seed=spec.seed, stage_seeds=(("base", spec.seed),))

    # Cooperative mid-render cancel (Task 1): thread the bus's is_cancelling poll
    # DOWN into the studio spine as a pure zero-arg probe. The studio never imports
    # media_bus — only this adapter does. A cancel makes produce_clip's runner abort
    # BEFORE writing a clip and return Err(StageError(CANCELLED)), which
    # _stage_error_to_job_error maps to JobError(code="cancelled", retryable=False).
    should_cancel = lambda: media_bus.is_cancelling(job_id)  # noqa: E731

    result = produce_clip(
        request,
        env=env,
        out_root=spec.out_root,
        seeds=seeds,
        start_image=spec.start_image,
        prompt=getattr(spec, "prompt", "") or "",
        negative_prompt=getattr(spec, "negative", "") or "",
        # B2 chain: the prior-tier clip (movie/scene mp4) this job extends. Carried
        # into the manifest; the i2v runner extends from its last frame when there is
        # no start_image. None -> "" (no source) inside produce_clip.
        source_video=getattr(spec, "source_video", None),
        should_cancel=should_cancel,
    )

    if result.is_err():
        return JobResult(
            job_id=job_id,
            ok=False,
            error=_stage_error_to_job_error(result.error),
        )

    # Ok(Artifact): the clip exists (produce_clip only returns Ok with a written,
    # non-empty clip). Catalog it exactly as the movie/scene runners do — ingest
    # probes it once and mints an immutable video MediaRef carried on outputs.
    artifact = result.unwrap()
    ref = ingest(artifact.path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))
