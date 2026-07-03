"""Durable, multi-process-safe job bus (stdlib sqlite3, WAL).

State machine (map §6):  queued -> claimed -> running -> done | failed

Discipline:
  * Enqueue path:  spec -> serialize -> insert 'queued' -> return job_id.
  * Worker path:   claim (atomic cross-process) -> run pure runner -> write once.
  * Single writer: every state write is gated on `WHERE claim_token=?`, so only
    the claiming worker mutates a given job_id.
  * Errors are DATA: a runner returns JobError inside JobResult; only this loop
    (run_claimed) catches an UNEXPECTED raise and converts it to a JobResult.

Spec (de)serialization is keyed by JobSpec.name so Phase 3 routes reuse it.
`start_worker_daemon()` is DEFINED but never called at import — Phase 3 wires it
at app init.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict
from typing import Callable, Dict, Optional
from uuid import uuid4

from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT

from .audio_schema import make_audio_extract
from .crop_schema import CropSpec, SpatialRegion, TemporalRegion, make_crop
from .frame_schema import make_frame_extract
from .gen_schema import GenPromptPart, make_generate_image
from .job_schema import JOB_REGISTRY
from .media_schema import make_media_ref
from .result_schema import JobResult
from .runners import DISPATCH

DB_PATH = os.path.join(DEFAULT_ROOT, "video_intel", "media_jobs.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS media_jobs (
    job_id      TEXT PRIMARY KEY,
    name        TEXT,
    status      TEXT,
    spec_json   TEXT,
    result_json TEXT,
    claim_token TEXT,
    created     REAL,
    updated     REAL
)
"""

_init_lock = threading.Lock()
_initialized = False


# --------------------------------------------------------------------------- #
# spec (de)serialization — keyed by JobSpec.name (Phase 3 reuses these)
# --------------------------------------------------------------------------- #
def _crop_from_dict(d: dict) -> CropSpec:
    """Rebuild a CropSpec from its asdict() form, through the validating
    factories (make_media_ref + make_crop) so invariants are re-checked."""
    src_d = d["source"]
    source = make_media_ref(**src_d)
    sp = d.get("spatial")
    tp = d.get("temporal")
    spatial = SpatialRegion(**sp) if sp is not None else None
    temporal = TemporalRegion(**tp) if tp is not None else None
    return make_crop(source=source, spatial=spatial, temporal=temporal)


def _frame_extract_from_dict(d: dict):
    """Rebuild a FrameExtractSpec from its asdict() form, through the validating
    factories (make_media_ref + optional TemporalRegion + make_frame_extract)."""
    source = make_media_ref(**d["source"])
    win = d.get("window")
    window = TemporalRegion(**win) if win is not None else None
    return make_frame_extract(
        source=source,
        fps=d["fps"],
        quality=d["quality"],
        fmt=d["fmt"],
        window=window,
        max_frames=d.get("max_frames"),
    )


def _audio_extract_from_dict(d: dict):
    """Rebuild an AudioExtractSpec from its asdict() form, through the validating
    factories (make_media_ref + make_audio_extract)."""
    source = make_media_ref(**d["source"])
    return make_audio_extract(source=source, fmt=d["fmt"])


def _generate_image_from_dict(d: dict):
    """Rebuild a GenerateImageSpec from its asdict() form, through the validating
    factories. Each part's media (if any) round-trips via make_media_ref."""
    parts = []
    for pd in d["parts"]:
        media_d = pd.get("media")
        media = make_media_ref(**media_d) if media_d is not None else None
        parts.append(GenPromptPart(kind=pd["kind"], text=pd.get("text"), media=media))
    return make_generate_image(
        parts=tuple(parts),
        model_id=d["model_id"],
        width=d["width"],
        height=d["height"],
        steps=d["steps"],
        guidance=d["guidance"],
        seed=d.get("seed"),
        negative=d.get("negative"),
    )


# name -> (dict -> spec). Grows as Phase 4+ specs land.
SPEC_DESERIALIZERS: Dict[str, Callable[[dict], object]] = {
    "crop": _crop_from_dict,
    "frame_extract": _frame_extract_from_dict,
    "audio_extract": _audio_extract_from_dict,
    "generate_image": _generate_image_from_dict,
}


def serialize_spec(name: str, spec) -> str:
    """Frozen spec -> json string. Nested MediaRef/regions serialize via asdict.

    `name` is accepted for symmetry with deserialize_spec / Phase 3 routing;
    asdict is generic so it is not needed to encode, but keeping the signature
    keyed by name keeps the (de)serialize pair a matched, registry-driven set.
    """
    return json.dumps(asdict(spec))


def deserialize_spec(name: str, d: dict):
    """json-dict -> frozen spec, via the per-name registry (registry-driven)."""
    try:
        builder = SPEC_DESERIALIZERS[name]
    except KeyError:
        raise KeyError(f"no spec deserializer registered for job name {name!r}")
    return builder(d)


def serialize_result(result: JobResult) -> str:
    return json.dumps(asdict(result))


# --------------------------------------------------------------------------- #
# connection / schema
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_db() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        conn = _connect()
        try:
            conn.execute(_CREATE_SQL)
        finally:
            conn.close()
        _initialized = True


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def enqueue(name: str, spec) -> str:
    """Mint a job_id, serialize the spec, insert status='queued'. Returns job_id."""
    if name not in JOB_REGISTRY:
        raise KeyError(f"unknown job name {name!r}; registered: {sorted(JOB_REGISTRY)}")
    _ensure_db()
    job_id = uuid4().hex
    spec_json = serialize_spec(name, spec)
    now = time.time()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO media_jobs "
            "(job_id, name, status, spec_json, result_json, claim_token, created, updated) "
            "VALUES (?, ?, 'queued', ?, NULL, NULL, ?, ?)",
            (job_id, name, spec_json, now, now),
        )
    finally:
        conn.close()
    return job_id


def claim(worker_token: str) -> Optional[str]:
    """Atomically claim the oldest queued job across processes. Returns job_id
    or None. Uses BEGIN IMMEDIATE + a conditional UPDATE so exactly one worker
    can transition a given row out of 'queued'."""
    _ensure_db()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT job_id FROM media_jobs WHERE status='queued' "
            "ORDER BY created LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        job_id = row[0]
        cur = conn.execute(
            "UPDATE media_jobs SET status='claimed', claim_token=?, updated=? "
            "WHERE job_id=? AND status='queued'",
            (worker_token, time.time(), job_id),
        )
        conn.execute("COMMIT")
        return job_id if cur.rowcount == 1 else None
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def run_claimed(job_id: str, worker_token: str) -> Optional[JobResult]:
    """Load + deserialize the spec (registry-driven), mark 'running' (single
    writer via claim_token), dispatch the pure runner, and write the JobResult
    ONCE. This is the ONLY place allowed to catch an UNEXPECTED raise and turn
    it into JobResult(ok=False, JobError('internal', ...))."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT name, spec_json, claim_token FROM media_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        name, spec_json, claim_token = row
        if claim_token != worker_token:
            # not our claim — refuse to write (single writer invariant)
            return None

        # transition to running (gated on our claim_token)
        conn.execute(
            "UPDATE media_jobs SET status='running', updated=? "
            "WHERE job_id=? AND claim_token=?",
            (time.time(), job_id, worker_token),
        )
    finally:
        conn.close()

    # ---- run outside the DB connection; the runner is pure & may block ----
    try:
        spec = deserialize_spec(name, json.loads(spec_json))
        job_spec = JOB_REGISTRY[name]
        runner = DISPATCH[job_spec.runner_key]
        result = runner(spec, job_id)
        if not isinstance(result, JobResult):
            raise TypeError(
                f"runner {job_spec.runner_key} returned {type(result).__name__}, "
                "expected JobResult"
            )
    except Exception as exc:  # the one sanctioned catch/convert point
        from .result_schema import JobError
        result = JobResult(
            job_id=job_id,
            ok=False,
            error=JobError(
                code="internal",
                message=f"{type(exc).__name__}: {exc}",
                retryable=False,
            ),
        )

    # ---- write the terminal state ONCE (single writer via claim_token) ----
    status = "done" if result.ok else "failed"
    result_json = serialize_result(result)
    conn = _connect()
    try:
        conn.execute(
            "UPDATE media_jobs SET status=?, result_json=?, updated=? "
            "WHERE job_id=? AND claim_token=?",
            (status, result_json, time.time(), job_id, worker_token),
        )
    finally:
        conn.close()
    return result


def work_once(worker_token: Optional[str] = None) -> Optional[str]:
    """Claim one job and run it. Returns the processed job_id or None if the
    queue was empty. This is what the headless self-test and the daemon call."""
    token = worker_token or f"worker-{os.getpid()}-{uuid4().hex[:8]}"
    job_id = claim(token)
    if job_id is None:
        return None
    run_claimed(job_id, token)
    return job_id


def get(job_id: str) -> dict:
    """Read-only view: {"job_id", "status", "result": <JobResult dict|None>}."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT status, result_json FROM media_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"job_id": job_id, "status": None, "result": None}
    status, result_json = row
    result = json.loads(result_json) if result_json else None
    return {"job_id": job_id, "status": status, "result": result}


def start_worker_daemon(worker_token: Optional[str] = None,
                        idle_sleep_s: float = 0.25) -> threading.Thread:
    """Background thread looping work_once() with a short sleep when idle.

    DEFINED but NOT called at import — Phase 3 wires it at app init.
    """
    token = worker_token or f"daemon-{os.getpid()}-{uuid4().hex[:8]}"

    def _loop() -> None:
        while True:
            try:
                processed = work_once(token)
            except Exception:
                # never let the daemon thread die on a transient DB error
                processed = None
            if processed is None:
                time.sleep(idle_sleep_s)

    thread = threading.Thread(target=_loop, name="media_bus_worker", daemon=True)
    thread.start()
    return thread
