"""Shared GPU-worker guard — extracted VERBATIM from runners/imagegen.py.

`guard_gpu_worker(model_id, job_id)` centralizes the 2026-07-03 central-meltdown
fix so every generation runner (generate_image, generate_scene, ...) shares ONE
policy. DelegatingRunner runs LOCAL unconditionally when the worker provider
returns no live worker (the HUGPY_LOCAL_FALLBACK gate only covers
selected-then-failed). During the worker's warm-up window that loaded a multi-GB
diffusion model onto central's CPU inside gunicorn (13 GB RSS, box-wide
timeouts). Policy here mirrors the plane's own: when a fleet EXISTS (provider
registered) but no live worker serves this model, refuse with retryable data
instead of melting central. A standalone single-box deploy (no provider
registered) keeps local generation — that posture is the product. Override with
HUGPY_VIDEOGEN_LOCAL=always.

The block below is byte-identical to imagegen's prior inline guard; only
`spec.model_id` became the `model_id` parameter.
"""
from __future__ import annotations

import os
from typing import Optional

from ..result_schema import JobError, JobResult


def guard_gpu_worker(model_id: str, job_id: str) -> Optional[JobResult]:
    """Return a refusal JobResult when a fleet exists but no live worker serves
    `model_id` (and the local override is off); else None (proceed)."""
    # Per-box "never serve locally" policy: this box runs no in-process
    # generation at all, even in a standalone/no-provider posture. Refuse before
    # a multi-GB diffusion model can land on this CPU. HUGPY_VIDEOGEN_LOCAL=always
    # still overrides for a deliberate local run. Default off === today's
    # behavior. See managers.serve.policy.
    try:
        from abstract_hugpy_dev.managers.serve.policy import no_local_serving
        _policy_off = not no_local_serving()
    except Exception:
        _policy_off = True
    _videogen_local = (
        os.environ.get("HUGPY_VIDEOGEN_LOCAL", "").strip().lower()
        in ("always", "1", "true", "yes", "on"))
    if not _policy_off and not _videogen_local:
        return JobResult(job_id, ok=False, error=JobError(
            code="local_serving_disabled",
            message=(
                f"local model serving is disabled on this box "
                f"(HUGPY_NO_LOCAL_SERVING); refusing in-process generation of "
                f"{model_id!r}. Bring a GPU worker online, or set "
                f"HUGPY_VIDEOGEN_LOCAL=always to permit local generation here."
            ),
            retryable=True,
        ))
    try:
        from abstract_hugpy_dev.managers.resolvers.remote import get_worker_provider
        provider = get_worker_provider()
    except ImportError:
        provider = None
    if provider is not None:
        try:
            try:
                live_worker = provider(model_id, None)
            except TypeError:   # provider may predate the pool arg
                live_worker = provider(model_id)
        except Exception:
            live_worker = None  # a broken provider must not crash the runner
        if live_worker is None and (
            os.environ.get("HUGPY_VIDEOGEN_LOCAL", "").strip().lower()
            not in ("always", "1", "true", "yes", "on")
        ):
            return JobResult(job_id, ok=False, error=JobError(
                code="no_live_gpu_worker",
                message=(
                    f"no live GPU worker is serving {model_id!r} (worker "
                    "offline or still warming); refusing local CPU generation "
                    "on central. Retry shortly, or set HUGPY_VIDEOGEN_LOCAL="
                    "always to permit in-process generation."
                ),
                retryable=True,
            ))
    return None
