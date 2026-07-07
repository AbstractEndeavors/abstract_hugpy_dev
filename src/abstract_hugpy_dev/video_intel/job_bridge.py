"""One-directional bridge: media_bus job lifecycle -> comms.JobStore (A/P0-2).

The roadmap ratified a BRIDGE, not a collapse:

  * comms.JobStore  = attribution / queue / cancel source of truth. It is what
    ``GET /llm/jobs`` reads. Before this bridge NO media_bus job (crop / scene /
    movie / studio_i2v) surfaced there — media jobs were queryable ONLY at
    ``GET /video/jobs/<id>``.
  * media_bus       = execution source of truth (its own durable media_jobs.db).

This module mirrors media-bus transitions INTO the JobStore, and does so
STRICTLY one-directionally (bus -> JobStore). It NEVER writes back into the bus,
so there is no distributed-write deadlock: cancel/attribution/queue stay the
JobStore's job, execution stays the bus's job.

Best-effort by construction: every function swallows and logs on failure, so a
bridge hiccup can NEVER fail a media job (the mirror plumbing in comms.jobs has
the same discipline). media_bus calls these at exactly three points:
``enqueue`` -> ``on_enqueue`` (queued), ``run_claimed`` -> ``on_running``
(running) then ``on_terminal`` (done | failed | cancelled).

Cross-process shape (the dev service runs gunicorn --workers 3 sharing ONE comms
mirror via HUGPY_COMMS_DB): the media daemon that RUNS a job is frequently a
different process than the one that enqueued it (each process runs a daemon
thread and the bus's atomic sqlite claim hands the job to whichever wins). So:

  * ``on_enqueue`` writes the QUEUED snapshot straight to the shared MIRROR and
    does NOT anchor a local JobStore record. A local "pending" record in the
    enqueue process would be a record that process never advances (a sibling
    runs the job) — it would shadow the fresher cross-process mirror view and
    stick at "pending". Mirror-only avoids that: every process still shows the
    job as queued (snapshot merges live mirror rows), and there is no stale
    local shadow.
  * ``on_running`` / ``on_terminal`` run in the process that actually owns the
    job. They anchor a LOCAL JobStore record and mark it terminal there. The
    JobStore retains terminal jobs (~600s), and ``GET /llm/jobs?live=0`` reads
    that retained representation — so a completed media job SHOWS, it does not
    vanish. Terminal is deliberately carried by local retention, NOT by the
    mirror's live_rows() (which excludes terminal rows — the exact gap a prior
    investigation flagged as the naive-bridge trap).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# media_bus status vocab -> comms canonical status vocab. media_bus speaks
# queued/claimed/running/cancelling/done/failed/cancelled; comms speaks
# pending/processing/streaming/done/cancelled/failed. Running maps to
# "processing" (a generic in-flight state) rather than "streaming" on purpose:
# "streaming" carries token-stream semantics (tokens/sec, on_output) that are
# meaningless for a frame-rendering media job — it would render oddly in the
# activity view. A media job is fully visible in /llm/jobs either way.
_STATUS_MAP = {
    "queued": "pending",
    "claimed": "processing",
    "running": "processing",
    "cancelling": "processing",
    "done": "done",
    "failed": "failed",
    "cancelled": "cancelled",
}

# Attribution transport tag for every bridged media job. Parallels the
# chat/web/discord/worker transports the JobStore already carries; lets the
# console filter /llm/jobs?transport=media.
_TRANSPORT = "media"


def _store():
    """The comms JobStore singleton. Imported lazily (never at media_bus import
    time) so media_bus stays importable even if comms is unhappy, and so there
    is zero import-time coupling / cycle risk."""
    from abstract_hugpy_dev.comms import job_store
    return job_store


def _result_uri(result) -> str | None:
    """Best-effort clip/artifact uri from a JobResult's first output — the
    back-reference to what the media job produced. Tolerates both a live
    JobResult (outputs = tuple[MediaRef]) and a plain dict (already asdict'd)."""
    try:
        outputs = getattr(result, "outputs", None)
        if outputs is None and isinstance(result, dict):
            outputs = result.get("outputs")
        if not outputs:
            return None
        first = outputs[0]
        uri = getattr(first, "uri", None)
        if uri is None and isinstance(first, dict):
            uri = first.get("uri")
        return uri or None
    except Exception:
        return None


def _error_dict(result):
    """Best-effort {code, message, retryable} from a failed JobResult's JobError.

    Since the Task 2 collapse there is ONE JobError class — ``result_schema.JobError``
    IS ``comms.jobs.JobError`` — so this is now a thin ``coerce -> to_dict`` of that
    single class rather than a hand-rolled cross-vocabulary conversion. The emitted
    shape is unchanged for the /llm/jobs contract: ``to_dict`` emits ``retryable``
    ALWAYS (None when absent) and ``detail`` only when truthy — bridged media errors
    carry ``detail=None``, so they still serialize as exactly {code, message,
    retryable}. Tolerates a live JobError, a plain dict, or an already-asdict'd
    result. The comms import stays LAZY (bridge discipline: zero import-time
    coupling)."""
    try:
        err = getattr(result, "error", None)
        if err is None and isinstance(result, dict):
            err = result.get("error")
        from abstract_hugpy_dev.comms.jobs import JobError
        job_err = JobError.coerce(err)   # JobError | dict | None -> JobError | None
        return job_err.to_dict() if job_err is not None else None
    except Exception:
        return None


def on_enqueue(job_id: str, name: str) -> None:
    """queued -> pending. Mirror-ONLY (no local anchor): every gunicorn process
    then shows the job as queued via the shared mirror, and the enqueue process
    does not pin a record it will never advance. No-op when the mirror is off
    (single-process installs still get running/terminal from the local store)."""
    try:
        store = _store()
        mirror = getattr(store, "mirror", None)
        if mirror is None:
            return
        # Reuse Job.to_dict() so the mirrored row is always shape-correct for
        # snapshot()/live_rows(); building the Job locally does NOT insert it
        # into the store (no stale local shadow) — we only upsert its dict.
        from abstract_hugpy_dev.comms.jobs import Job
        row = Job(id=job_id, kind=name, status="pending",
                  transport=_TRANSPORT, model_name=name).to_dict()
        mirror.upsert(row)
    except Exception:
        logger.warning("job_bridge.on_enqueue failed for %s (%s)",
                       job_id, name, exc_info=True)


def on_running(job_id: str, name: str, worker: str | None = None) -> None:
    """claimed/running -> processing, in the process that actually runs the job.
    Anchors a LOCAL JobStore record (create-if-missing, else advance) so this
    process owns the row for its visible life and mirrors the live status to
    siblings."""
    try:
        store = _store()
        if store.get(job_id) is None:
            store.create(id=job_id, kind=name, status="processing",
                         worker=worker, transport=_TRANSPORT, model_name=name)
        else:
            store.update(job_id, status="processing", worker=worker,
                         transport=_TRANSPORT, model_name=name)
    except Exception:
        logger.warning("job_bridge.on_running failed for %s (%s)",
                       job_id, name, exc_info=True)


def on_terminal(job_id: str, name: str, status: str, result=None,
                worker: str | None = None) -> None:
    """done|failed|cancelled -> terminal, in the process that ran the job. Marks
    the local record terminal (retained ~600s) so GET /llm/jobs?live=0 shows the
    completed job — the JobStore's own terminal representation, NOT the mirror's
    live_rows() (which excludes terminal). On success the produced clip uri rides
    along as the job message (back-reference to the artifact)."""
    try:
        store = _store()
        comms_status = _STATUS_MAP.get(status, "failed")
        # Ensure a local record exists to mark terminal (defensive: normally
        # on_running already anchored it in this same process).
        if store.get(job_id) is None:
            store.create(id=job_id, kind=name, status="processing",
                         worker=worker, transport=_TRANSPORT, model_name=name)
        error = _error_dict(result) if comms_status == "failed" else None
        store.finish(job_id, status=comms_status, error=error)
        if comms_status == "done":
            uri = _result_uri(result)
            if uri:
                # Non-status update on a terminal job is allowed (finish/update
                # keep the first terminal status; message is free-form).
                store.update(job_id, message=uri)
    except Exception:
        logger.warning("job_bridge.on_terminal failed for %s (%s -> %s)",
                       job_id, name, status, exc_info=True)
