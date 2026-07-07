"""F5 — one Job schema + JobStore for every unit of work, every transport.

Before this module the tree had TWO half-job systems:

    - flask_app .../schemas/job_schemas.py — download jobs only (progress bytes,
      retry telemetry), invisible to chat.
    - managers/dispatch/activity.py — in-flight chat requests only, ephemeral
      (popped on stream end), no principal/transport/worker metadata, no cancel.

This store unifies them. Lifecycle is frozen:

    pending -> processing -> streaming -> (done | cancelled | failed)

Legacy names still arrive from old callers and old UI expectations
(queued/running/completed); normalize_status() maps them on write and read so
nothing breaks while consumers migrate.

Cancellation lives ON the job: the owning stream attaches a zero-arg cancel
handle (e.g. one that sets its asyncio.Event on the shared runtime loop), and
anyone — an HTTP route, a bus control message — cancels through the store.
The race that matters (DISC-03): cancel-vs-just-finished. Rules:

    - cancel() only *requests*: fires the handle, flags cancel_requested. It
      never force-marks the job cancelled while the stream is live.
    - the first terminal status wins; finish() on an already-terminal job is a
      no-op. A job that completed a microsecond before the cancel arrives
      stays "done".
    - the stream's own teardown converts cancel_requested into status
      "cancelled" (via finish(job_id, "cancelled") or the finish() default
      described below), so "cancelled" always means the resources were
      actually released by the code that held them.

Like its predecessors this is per-process (one gunicorn process, many
threads) and thread-safe. Terminal jobs are retained briefly so UIs can show
"just finished", then pruned — the store never grows without bound.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .shared import SqliteMirror

CANONICAL_STATUSES = ("pending", "processing", "streaming",
                      "done", "cancelled", "failed")
TERMINAL_STATUSES = frozenset(("done", "cancelled", "failed"))

# The media_bus job kinds bridged in via video_intel.job_bridge (transport
# "media"). snapshot(live_only=False) surfaces a sibling process's *terminal*
# rows for THESE kinds only (via mirror.terminal_rows), so a finished media job
# is visible on every process — not just the one that ran it. Strictly gated to
# media so chat/download terminal cross-process behavior is unchanged (their
# terminal rows still show only via each process's local ~600s retention).
MEDIA_KINDS = ("crop", "frame_extract", "generate_scene",
               "generate_movie", "studio_i2v")

# Old JOBSTATUS names (and activity.py's view states) -> canonical.
_LEGACY_STATUS = {
    "queued": "pending",
    "waiting": "pending",
    "running": "processing",
    "active": "streaming",
    "completed": "done",
}
# Canonical -> the old JOBSTATUS wire names, for surfaces (the download UI's
# /jobs contract) that still speak them. streaming has no legacy analogue;
# it reads as running there.
LEGACY_FOR_CANONICAL = {
    "pending": "queued",
    "processing": "running",
    "streaming": "running",
    "done": "completed",
}


def normalize_status(status: Any) -> str:
    s = str(status or "pending").strip().lower()
    s = _LEGACY_STATUS.get(s, s)
    return s if s in CANONICAL_STATUSES else "pending"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobError:
    """Error-as-data. A failed job carries one of these, never a bare string
    (str(exc) leaks internals and can't be routed on)."""
    code: str = "error"
    message: str = ""
    detail: Optional[dict] = None
    # Field-aligned with video_intel.result_schema.JobError (which serializes on
    # /video/jobs and REQUIRES retryable). Here it is nullable and additive: chat
    # /download rows serialize "retryable": null; bridged media rows carry the
    # real bool. Deliberately NOT required — never break the non-media callers.
    retryable: Optional[bool] = None

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        # Emitted ALWAYS (nullable) so every /llm/jobs error object carries the
        # field — a real bool for media, null for chat/download.
        out["retryable"] = self.retryable
        return out

    @classmethod
    def coerce(cls, err: Any) -> "JobError | None":
        if err is None or isinstance(err, JobError):
            return err
        if isinstance(err, dict):
            return cls(code=str(err.get("code") or "error"),
                       message=str(err.get("message") or ""),
                       detail=err.get("detail"),
                       retryable=err.get("retryable"))
        if isinstance(err, BaseException):
            return cls(code=type(err).__name__, message=str(err))
        return cls(message=str(err))


@dataclass
class Job:
    id: str
    model_key: str = ""
    status: str = "pending"
    kind: str = "chat"                    # chat | v1 | discord | cli | download | ...
    # Addressing / attribution (F5.2). All optional until F2 lands a real
    # Principal — carry them now so nothing has to re-plumb later.
    principal: Optional[str] = None       # who asked
    transport: Optional[str] = None       # web | v1 | discord | cli | worker
    channel: Optional[str] = None         # discord channel id, web session, ...
    worker: Optional[str] = None          # worker name/id serving it
    slot: Optional[str] = None            # slot id on that worker
    model_name: Optional[str] = None      # display name (model_key is the key)
    message: str = ""
    error: Optional[JobError] = None
    # Live-stream telemetry (was activity.py).
    tokens: int = 0
    started_ts: float = field(default_factory=time.time)
    first_output_ts: Optional[float] = None
    ended_ts: Optional[float] = None
    cancel_requested: bool = False
    # Download telemetry (was the flask job_schemas Job) — unused for chat.
    progress: float = 0.0                 # 0.0–1.0
    total_bytes: Optional[int] = None
    downloaded_bytes: Optional[int] = None
    attempt: int = 0
    max_attempts: int = 0
    stalled: bool = False
    bytes_per_second: Optional[float] = None
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    # Runtime-only, never serialized: download subprocess, resolved model dict
    # (kept so a manual retry can resume without re-resolving), cancel handle,
    # and the last time this job synced to the cross-process mirror.
    _proc: Any = field(default=None, repr=False, compare=False)
    _model: Optional[dict] = field(default=None, repr=False, compare=False)
    _cancel: Optional[Callable[[], None]] = field(default=None, repr=False,
                                                  compare=False)
    _last_sync: float = field(default=0.0, repr=False, compare=False)

    @property
    def terminal(self) -> bool:
        return normalize_status(self.status) in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot — superset of both predecessors' shapes."""
        now = time.time()
        ended = self.ended_ts or now
        return {
            "id": self.id,
            "model_key": self.model_key,
            "status": normalize_status(self.status),
            "kind": self.kind,
            "principal": self.principal,
            "transport": self.transport,
            "channel": self.channel,
            "worker": self.worker,
            "slot": self.slot,
            "model": self.model_name or self.model_key or "?",
            "message": self.message,
            "error": self.error.to_dict() if self.error else None,
            "tokens": self.tokens,
            "elapsed": round(ended - self.started_ts, 1),
            # seconds spent waiting before the first output (queue time).
            "wait": round((self.first_output_ts or ended) - self.started_ts, 1),
            "cancel_requested": self.cancel_requested,
            "progress": round(self.progress, 4),
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "stalled": self.stalled,
            "bytes_per_second": self.bytes_per_second,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """The pre-comms /jobs wire shape: legacy status names, error as a
        plain string. The download UI (ModelTable) reads exactly this; new
        surfaces should read to_dict() and canonical states instead."""
        d = self.to_dict()
        d["status"] = LEGACY_FOR_CANONICAL.get(d["status"], d["status"])
        if isinstance(d.get("error"), dict):
            d["error"] = d["error"].get("message") or d["error"].get("code")
        return d


def _default_mirror() -> Optional[SqliteMirror]:
    """Cross-process mirror for the module singleton. Off-switch:
    HUGPY_COMMS_DB=off. Bare JobStore() instances (tests, embedders) stay
    purely in-process unless given a mirror explicitly."""
    env = (os.environ.get("HUGPY_COMMS_DB") or "").strip().lower()
    if env in ("off", "none", "0", "disabled"):
        return None
    try:
        return SqliteMirror()
    except Exception:
        return None


class JobStore:
    """Thread-safe, per-process, with an optional cross-process mirror.

    ``on_change`` (set once at wiring time) is called with (job, prior_status)
    after every status transition, outside no locks the caller can see — the
    bus adapter uses it to publish job.* events without this module importing
    the bus (keeps comms.jobs importable by absolutely everything).

    ``mirror`` (comms.shared.SqliteMirror) makes cancel and queue views
    correct when the service runs multiple processes (gunicorn --workers N):
    transitions are mirrored, a cancel for a job another process owns raises
    a flag on the shared row, and the owner notices it on its next token.
    Everything mirror-related is best-effort — no mirror, or a broken one,
    degrades to per-process behavior.
    """

    def __init__(self, *, retain_terminal: int = 100,
                 retain_secs: float = 600.0,
                 mirror: Optional[SqliteMirror] = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._retain_terminal = retain_terminal
        self._retain_secs = retain_secs
        self.on_change: Optional[Callable[[Job, str], None]] = None
        self.mirror = mirror
        self._last_mirror_prune = 0.0
        self._watcher: Optional[threading.Thread] = None

    # -- creation ----------------------------------------------------------
    def create(self, model_key: str = "", *, id: Optional[str] = None,
               kind: str = "chat", **meta: Any) -> Job:
        """Old signature preserved: download callers do create(model_key).
        New callers pass id=request_id so the job id IS the request id and
        every layer (SSE, bus, cancel route) correlates on one key."""
        job = Job(id=id or str(uuid.uuid4()), model_key=model_key or "",
                  kind=kind)
        for k, v in meta.items():
            if hasattr(job, k):
                setattr(job, k, v)
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job
        self._emit(job, "")
        self._mirror_upsert(job)
        self._maybe_prune_mirror()
        self._ensure_watcher()
        return job

    # -- mutation ----------------------------------------------------------
    def update(self, job_id: str, **changes: Any) -> Optional[Job]:
        prior = None
        resurrected = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            prior = normalize_status(job.status)
            if "status" in changes:
                new = normalize_status(changes["status"])
                if job.terminal and new in TERMINAL_STATUSES and new != prior:
                    # First terminal state wins (cancel-vs-just-finished race):
                    # done never becomes cancelled after the fact.
                    changes = {k: v for k, v in changes.items() if k != "status"}
                else:
                    changes["status"] = new
                    if new in TERMINAL_STATUSES:
                        if job.ended_ts is None:
                            job.ended_ts = time.time()
                    elif job.terminal:
                        # Deliberate terminal->live resurrection (download
                        # retry reuses the job id): reset the run telemetry.
                        job.ended_ts = None
                        job.cancel_requested = False
                        job.first_output_ts = None
                        resurrected = True
            if "error" in changes:
                changes["error"] = JobError.coerce(changes["error"])
            for k, v in changes.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = _utcnow_iso()
        if normalize_status(job.status) != prior:
            self._emit(job, prior)
        if resurrected and self.mirror is not None:
            # upsert never lowers the shared cancel flag; a fresh run must.
            try:
                self.mirror.clear_cancel(job_id)
            except Exception:
                pass
        self._mirror_upsert(job)
        return job

    def on_output(self, job_id: str, n: int = 1) -> None:
        """First output flips the job to streaming; every output bumps the
        counter. One lock acquisition per token — same cost activity.py paid.

        The mirror sync here is write-only and throttled (~1/sec per job) so
        sibling queue views stay fresh without taxing the token hot path;
        noticing a sibling's cancel flag is the watcher thread's job."""
        emit = False
        sync = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.terminal:
                return
            if job.first_output_ts is None:
                job.first_output_ts = time.time()
            if normalize_status(job.status) != "streaming":
                prior = normalize_status(job.status)
                job.status = "streaming"
                emit = True
            job.tokens += n
            if self.mirror is not None and \
                    time.time() - job._last_sync >= 1.0:
                sync = True
        if emit:
            self._emit(job, prior)
        if sync:
            self._mirror_upsert(job)

    def finish(self, job_id: str, status: Optional[str] = None,
               error: Any = None) -> Optional[Job]:
        """Terminal marking, from the code that actually owned the stream.
        With no explicit status: failed if an error is given, cancelled if a
        cancel was requested, else done. No-op if already terminal."""
        job = self.get(job_id)
        if job is None:
            return None
        if status is None:
            if error is not None:
                status = "failed"
            elif job.cancel_requested:
                status = "cancelled"
            else:
                status = "done"
        changes: dict[str, Any] = {"status": status}
        if error is not None:
            changes["error"] = error
        return self.update(job_id, **changes)

    # -- cancel plane ------------------------------------------------------
    def attach_cancel(self, job_id: str, handle: Callable[[], None]) -> None:
        fire = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job._cancel = handle
                # A cancel that raced in before the stream attached its handle
                # must still take effect — fire immediately.
                if job.cancel_requested:
                    fire, job._cancel = job._cancel, None
        # Same race, cross-process flavor: a sibling flagged the mirror row
        # before we attached. Fires the handle we just stored via cancel().
        if (fire is None and job is not None and not job.cancel_requested
                and self.mirror is not None):
            try:
                if self.mirror.cancel_requested(job_id):
                    self.cancel(job_id, reason="cancelled by sibling process")
                    return
            except Exception:
                pass
        self._fire(fire)

    def cancel(self, job_id: str, reason: str = "") -> bool:
        """Request cancellation. Returns True if the job was live anywhere —
        here, or (via the mirror) in a sibling process, which notices the
        shared flag on its next token. Never force-marks cancelled — the
        owning stream's teardown does (first terminal state wins keeps the
        just-finished case honest)."""
        fire = None
        local_live = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and not job.terminal:
                local_live = True
                job.cancel_requested = True
                if reason:
                    job.message = reason
                job.updated_at = _utcnow_iso()
                fire, job._cancel = job._cancel, None   # fire at most once
        self._fire(fire)
        remote_live = False
        if self.mirror is not None:
            try:
                remote_live = self.mirror.request_cancel(job_id)
            except Exception:
                pass
            if local_live:
                self._mirror_upsert(job)
        return local_live or remote_live

    @staticmethod
    def _fire(handle: Optional[Callable[[], None]]) -> None:
        # Always outside the store lock: a handle may touch the store.
        if handle is not None:
            try:
                handle()
            except Exception:
                pass

    # -- views -------------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def snapshot(self, *, kinds: Optional[set[str]] = None,
                 live_only: bool = True) -> list[dict]:
        """JSON-safe view for queue UIs. Waiting first, then longest-running —
        the exact ordering the console's activity view has always shown.
        With a mirror, live jobs owned by sibling processes are merged in
        (their snapshots are at most ~1s stale); local records win on id."""
        with self._lock:
            jobs = [j for j in self._jobs.values()
                    if (kinds is None or j.kind in kinds)
                    and (not live_only or not j.terminal)]
            local_ids = set(self._jobs)
        out = [j.to_dict() for j in jobs]
        if self.mirror is not None:
            try:
                for d in self.mirror.live_rows():
                    # Skip our own rows — the in-memory record is the truth
                    # for them (including ones the mirror hasn't caught up on).
                    if d.get("id") in local_ids:
                        continue
                    if kinds is not None and d.get("kind") not in kinds:
                        continue
                    out.append(d)
            except Exception:
                pass
            # Cross-process TERMINAL media rows (media-gated). live_rows() hides
            # terminal rows by design; this additive merge surfaces a sibling's
            # finished media job on the full (live_only=False) view — and ONLY
            # for MEDIA_KINDS, so chat/download terminal behavior is unchanged.
            if not live_only:
                try:
                    seen = {d.get("id") for d in out}   # local + live-merged ids
                    for d in self.mirror.terminal_rows(MEDIA_KINDS):
                        if d.get("id") in seen:
                            continue
                        if kinds is not None and d.get("kind") not in kinds:
                            continue
                        out.append(d)
                        seen.add(d.get("id"))
                except Exception:
                    pass
        out.sort(key=lambda d: (d["status"] not in ("pending", "processing"),
                                -d.get("elapsed", 0)))
        return out

    def counts(self, *, kinds: Optional[set[str]] = None) -> dict:
        snap = self.snapshot(kinds=kinds)
        waiting = sum(1 for d in snap
                      if d["status"] in ("pending", "processing"))
        active = sum(1 for d in snap if d["status"] == "streaming")
        return {"waiting": waiting, "active": active, "total": len(snap)}

    # -- retention ---------------------------------------------------------
    def _prune_locked(self) -> None:
        now = time.time()
        terminal = [(j.ended_ts or now, jid) for jid, j in self._jobs.items()
                    if j.terminal]
        stale = [jid for ended, jid in terminal
                 if now - ended > self._retain_secs]
        terminal.sort()
        overflow = [jid for _, jid in
                    terminal[:max(0, len(terminal) - self._retain_terminal)]]
        for jid in {*stale, *overflow}:
            self._jobs.pop(jid, None)

    def _emit(self, job: Job, prior: str) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(job, prior)
        except Exception:
            pass

    # -- mirror plumbing (all best-effort) -----------------------------------
    def _mirror_upsert(self, job: Optional[Job]) -> None:
        if self.mirror is None or job is None:
            return
        try:
            job._last_sync = time.time()
            self.mirror.upsert(job.to_dict())
        except Exception:
            pass

    def _maybe_prune_mirror(self) -> None:
        if self.mirror is None:
            return
        now = time.time()
        if now - self._last_mirror_prune < 60.0:
            return
        self._last_mirror_prune = now
        try:
            self.mirror.prune()
        except Exception:
            pass

    def _ensure_watcher(self) -> None:
        """Owner-side cancel watcher: a sibling process can only raise the
        shared flag; the process holding the stream's cancel handle must fire
        it. A 1s tick covers the waiting/provisioning phase too — exactly
        when users cancel most, and when no tokens flow to piggyback on."""
        if self.mirror is None or self._watcher is not None:
            return
        with self._lock:
            if self._watcher is not None:
                return
            t = threading.Thread(target=self._watch_mirror,
                                 name="comms-mirror-watch", daemon=True)
            self._watcher = t
        t.start()

    def _watch_mirror(self) -> None:
        while True:
            time.sleep(1.0)
            try:
                with self._lock:
                    candidates = [j.id for j in self._jobs.values()
                                  if not j.terminal and not j.cancel_requested]
                if not candidates:
                    continue
                flagged = self.mirror.flagged_ids()
                for jid in candidates:
                    if jid in flagged:
                        self.cancel(jid, reason="cancelled by sibling process")
            except Exception:
                pass


job_store = JobStore(mirror=_default_mirror())
