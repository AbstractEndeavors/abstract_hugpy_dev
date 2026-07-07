"""Cross-process mirror for the comms control plane (SQLite, stdlib-only).

The JobStore and Bus are per-process — correct for one gunicorn worker, but
the dev service runs ``--workers 3``: a cancel POST lands on process B while
process A serves the stream, and B's store has never heard of the job. This
mirror is the smallest primitive that fixes that without a broker or an
external dependency:

    - every job transition is upserted here (JSON snapshot + cancel flag),
    - a cancel for a job we don't hold locally sets the flag on the row,
    - the process that owns the stream notices the flag on its next token
      (throttled read on the hot path) and fires its local cancel handle,
    - queue views merge live rows from sibling processes into their snapshot.

SQLite in WAL mode handles concurrent single-row writes from a handful of
processes easily. Every operation opens a short-lived connection — no shared
handles across threads, no pooling to get wrong. All methods are best-effort:
a mirror failure degrades to per-process behavior (exactly today's world),
never breaks a chat. After MAX_FAILURES consecutive errors the mirror
disables itself loudly rather than taxing every token with a doomed write.

The db path must be shared by all processes of one service: HUGPY_COMMS_DB
if set, else a per-user file under XDG_RUNTIME_DIR (or /tmp). Central and a
worker agent on the same box sharing one file is harmless — job ids are
uuids and each process still only cancels what it holds a handle for.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    data             TEXT NOT NULL,
    status           TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'chat',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    updated          REAL NOT NULL
);
"""

MAX_FAILURES = 5


def default_db_path() -> str:
    env = (os.environ.get("HUGPY_COMMS_DB") or "").strip()
    if env:
        return env
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return os.path.join(base, f"hugpy-comms-{os.getuid()}.db")


class SqliteMirror:
    def __init__(self, path: Optional[str] = None,
                 retain_secs: float = 600.0) -> None:
        self.path = path or default_db_path()
        self.retain_secs = retain_secs
        self._failures = 0
        self._disabled = False
        self._init_lock = threading.Lock()
        self._initialized = False

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=2.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=2000")
        return conn

    def _ensure(self) -> bool:
        if self._disabled:
            return False
        if self._initialized:
            return True
        with self._init_lock:
            if self._initialized:
                return True
            try:
                with self._connect() as conn:
                    conn.execute(_SCHEMA)
                self._initialized = True
                return True
            except Exception as exc:
                self._note_failure("init", exc)
                return False

    def _note_failure(self, op: str, exc: Exception) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES and not self._disabled:
            self._disabled = True
            logger.error("comms mirror DISABLED after %d failures "
                         "(last: %s during %s) — cross-process cancel/queue "
                         "degrade to per-process until restart",
                         self._failures, exc, op)
        else:
            logger.warning("comms mirror %s failed: %s", op, exc)

    def _ok(self) -> None:
        self._failures = 0

    # -- writes --------------------------------------------------------------
    def upsert(self, job_dict: dict[str, Any]) -> None:
        """Mirror a job snapshot. cancel_requested is only ever raised here,
        never lowered (a resurrection lowers it explicitly via clear_cancel) —
        so a racing sibling's cancel can't be lost under a stale local write."""
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO jobs (id, data, status, kind, cancel_requested, updated) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "  data=excluded.data, status=excluded.status, "
                    "  kind=excluded.kind, "
                    "  cancel_requested=MAX(jobs.cancel_requested, excluded.cancel_requested), "
                    "  updated=excluded.updated",
                    (str(job_dict.get("id")), json.dumps(job_dict),
                     str(job_dict.get("status") or "pending"),
                     str(job_dict.get("kind") or "chat"),
                     1 if job_dict.get("cancel_requested") else 0,
                     time.time()))
            self._ok()
        except Exception as exc:
            self._note_failure("upsert", exc)

    def request_cancel(self, job_id: str) -> bool:
        """Raise the cancel flag on a row we may not hold locally. Returns
        True if a live (non-terminal) row existed to flag."""
        if not self._ensure():
            return False
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE jobs SET cancel_requested=1, updated=? "
                    "WHERE id=? AND status NOT IN ('done','cancelled','failed')",
                    (time.time(), job_id))
            self._ok()
            return cur.rowcount > 0
        except Exception as exc:
            self._note_failure("request_cancel", exc)
            return False

    def clear_cancel(self, job_id: str) -> None:
        """Deliberate resurrection (download retry) starts a fresh run."""
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute("UPDATE jobs SET cancel_requested=0, updated=? "
                             "WHERE id=?", (time.time(), job_id))
            self._ok()
        except Exception as exc:
            self._note_failure("clear_cancel", exc)

    def prune(self) -> None:
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM jobs WHERE "
                    "status IN ('done','cancelled','failed') AND updated < ?",
                    (time.time() - self.retain_secs,))
                # Rows whose owner process died mid-stream never go terminal;
                # sweep anything untouched for much longer than any stream.
                conn.execute("DELETE FROM jobs WHERE updated < ?",
                             (time.time() - max(self.retain_secs * 6, 3600.0),))
            self._ok()
        except Exception as exc:
            self._note_failure("prune", exc)

    # -- reads ---------------------------------------------------------------
    def cancel_requested(self, job_id: str) -> bool:
        if not self._ensure():
            return False
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT cancel_requested FROM jobs WHERE id=?",
                    (job_id,)).fetchone()
            self._ok()
            return bool(row and row[0])
        except Exception as exc:
            self._note_failure("cancel_requested", exc)
            return False

    def flagged_ids(self) -> set:
        """Ids with the cancel flag raised — the owner-side watcher intersects
        these with its local live jobs."""
        if not self._ensure():
            return set()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id FROM jobs WHERE cancel_requested=1").fetchall()
            self._ok()
            return {r[0] for r in rows}
        except Exception as exc:
            self._note_failure("flagged_ids", exc)
            return set()

    def live_rows(self) -> list[dict[str, Any]]:
        """Snapshots of every live job any process mirrored here."""
        if not self._ensure():
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data FROM jobs WHERE "
                    "status NOT IN ('done','cancelled','failed')").fetchall()
            self._ok()
        except Exception as exc:
            self._note_failure("live_rows", exc)
            return []
        out = []
        for (data,) in rows:
            try:
                out.append(json.loads(data))
            except Exception:
                continue
        return out

    def terminal_rows(self, kinds: tuple) -> list[dict[str, Any]]:
        """Snapshots of TERMINAL jobs of the given kinds — the additive
        complement to live_rows() (which excludes terminal by design). Callers
        pass a strict kind allowlist (MEDIA_KINDS) so this surfaces a sibling's
        finished media job cross-process WITHOUT touching chat/download terminal
        behavior. Empty kinds -> nothing (never a full-table terminal scan)."""
        kinds = tuple(kinds or ())
        if not kinds:
            return []
        if not self._ensure():
            return []
        try:
            # Placeholders are '?' derived from the count only — the kind values
            # themselves are always bound parameters (no SQL injection surface).
            placeholders = ",".join("?" for _ in kinds)
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data FROM jobs WHERE "
                    "status IN ('done','cancelled','failed') "
                    f"AND kind IN ({placeholders})", kinds).fetchall()
            self._ok()
        except Exception as exc:
            self._note_failure("terminal_rows", exc)
            return []
        out = []
        for (data,) in rows:
            try:
                out.append(json.loads(data))
            except Exception:
                continue
        return out
