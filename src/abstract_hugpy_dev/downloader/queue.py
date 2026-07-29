"""ENQUEUE / CANCEL / RETRY — everything the API is allowed to do to a download.

The console API's entire relationship with downloading is now this module. It
creates a queued job and returns; it never starts a transfer, never spawns a
child, never touches the network on a request path, and never runs a monitor
thread. The daemon (daemon.py) is the only process that executes work.

Flask-free on purpose: the daemon imports the same helpers for its own
bookkeeping, and the route layer keeps only thin shims that translate to HTTP.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..comms.jobs import Job, job_store, normalize_status, to_legacy
from .engine import DOWNLOAD_KIND, invalidate_model_status_cache
from .presence import downloader_alive

# How long a job may sit unclaimed before the view says so out loud. Short
# enough that an operator learns within one poll cycle that no daemon is
# running; long enough that a normal claim (well under a second) never trips it.
WAITING_GRACE_SECONDS = 30.0

_NO_DAEMON_MESSAGE = (
    "Queued — waiting for the downloader service (hugpy-downloader-dev is not "
    "running).")
_WAITING_MESSAGE = "Queued — waiting for downloader…"


def enqueue_download(model_key: str, model: dict,
                     total_bytes: Optional[int] = None,
                     transport: str = "web") -> Job:
    """Create a QUEUED download job and hand it to the daemon. Returns
    immediately — the only work done here is one mirror row.

    The model spec rides in ``payload`` because the daemon is a different
    process: ``Job._model`` is runtime-only and would arrive empty. ``payload``
    is also what makes RETRY work after an API restart — the spec is persisted,
    not held in some thread's closure.
    """
    payload: dict[str, Any] = {"model": model}
    if total_bytes:
        payload["total_bytes"] = int(total_bytes)
    return job_store.enqueue(
        model_key, kind=DOWNLOAD_KIND, transport=transport,
        status="pending", message=_WAITING_MESSAGE,
        total_bytes=total_bytes, payload=payload,
        model_name=(model or {}).get("name") or model_key,
    )


def annotate_waiting(d: dict) -> dict:
    """Make a queued-but-unstarted job SAY SO.

    Graceful degradation is the point (there is deliberately no in-process
    fallback — falling back would resurrect the very bug this separation fixes).
    A job that has been queued past the grace window gets an honest message, and
    when the daemon's heartbeat is stale it names the service that is missing.
    Read-time only: nothing is written, so an operator starting the daemon makes
    the message disappear on the next poll."""
    try:
        if normalize_status(d.get("status")) != "pending":
            return d
        age = time.time() - float(d.get("progressed_at") or 0)
        if age < WAITING_GRACE_SECONDS:
            return d
        d = dict(d)
        d["message"] = (_WAITING_MESSAGE if downloader_alive()
                        else _NO_DAEMON_MESSAGE)
    except (TypeError, ValueError):
        pass
    return d


def list_downloads() -> list[dict]:
    """The /jobs view: every download row, MIRROR-MERGED, legacy wire shape.

    Two things this must get right, both of which the old local-only read got
    for free by owning everything:
      * live rows now belong to the DAEMON, so they only exist in the mirror;
      * TERMINAL rows must stay visible — snapshot() hides cross-process
        terminals except for media kinds, and a download that vanished at 100%
        instead of showing "completed" would be a worse UI than before.
    """
    rows = job_store.snapshot(kinds={DOWNLOAD_KIND}, live_only=False,
                              terminal_kinds=(DOWNLOAD_KIND,))
    return [to_legacy(annotate_waiting(d)) for d in rows]


def get_download(job_id: str) -> Optional[dict]:
    """One download row, mirror-merged, legacy wire shape (None if unknown)."""
    d = job_store.get_dict(job_id)
    if d is None:
        return None
    return to_legacy(annotate_waiting(d))


def cancel_download(job_id: str) -> dict:
    """Cancel a download WHEREVER it runs.

    ``cancel_authoritative`` already does exactly the right thing across
    processes: with no live owner in THIS process it raises the shared cancel
    flag (which the daemon's store watcher turns into a real teardown) AND
    force-marks the row terminal, so a job nobody owns can never stay immortal.
    """
    d = job_store.get_dict(job_id)
    if d is None:
        return {"cancelled": False, "reason": "unknown job"}
    if normalize_status(d.get("status")) in ("done", "cancelled", "failed",
                                             "expired"):
        return {"cancelled": False, "reason": f"job is {d.get('status')}"}
    res = job_store.cancel_authoritative(job_id, reason="Cancelled by user.")
    if res.get("cancelled"):
        invalidate_model_status_cache(
            f"download cancelled: {d.get('model_key')}",
            model_key=d.get("model_key") or None)
    return {"cancelled": bool(res.get("cancelled")), "mode": res.get("mode")}


def retry_download(job_id: str) -> dict:
    """Re-queue a failed/cancelled download so the daemon picks it up again.

    Same job id and the SAME persisted payload, so partial files already on disk
    are resumed (HF resume + staging adoption), not re-fetched."""
    d = job_store.get_dict(job_id)
    if d is None:
        return {"retried": False, "reason": "unknown job"}
    if normalize_status(d.get("status")) not in ("done", "cancelled", "failed",
                                                 "expired"):
        return {"retried": False, "reason": f"job is already {d.get('status')}"}
    if not (d.get("payload") or {}).get("model"):
        return {"retried": False, "reason": "no model context to resume from"}
    ok = job_store.requeue(job_id, message=_WAITING_MESSAGE,
                           kinds=(DOWNLOAD_KIND,))
    if not ok:
        return {"retried": False, "reason": "job row is gone"}
    return {"retried": True, "id": job_id}
