"""Boundary results — map §6 exactly.

A runner's outcome crosses a module boundary (bus <-> runner), so it is DATA,
never a raise. Expected failures become a JobError carried inside a JobResult.
Only the worker loop (media_bus.run_claimed) is allowed to catch an UNEXPECTED
raise and convert it to a JobResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .media_schema import MediaRef


@dataclass(frozen=True)
class JobError:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class JobResult:
    job_id: str
    ok: bool
    outputs: Tuple[MediaRef, ...] = ()    # frame_extract -> many; crop/gen -> one
    error: Optional[JobError] = None      # data across the boundary, never a raise
