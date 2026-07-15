"""Orphan-sweep in SqliteMirror.prune() must age on real forward-progress
silence (progressed_at), NOT on `updated`.

Regression: identity_mesh_build render jobs wedge mid-render (ae drops during
Hunyuan3D texture load). They read stalled=true but never go terminal, and they
were IMMORTAL because upsert() bumps `updated`=now on EVERY write — including
snapshot re-upserts triggered merely by VIEWING /llm/jobs and the API-restart
re-read. The orphan-sweep keyed on `updated`, so its deadline never elapsed.

Fix: the orphan-sweep keys on progressed_at (the movement-only clock). These
tests are script-style (module-level asserts run at pytest collection), matching
the sibling test_comms_mirror.py convention.
"""
import sys, tempfile, os, time, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.comms.shared import SqliteMirror

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def _row_count(mirror, job_id):
    with sqlite3.connect(mirror.path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE id=?", (job_id,)).fetchone()[0]


# retain_secs=600 -> orphan window = max(600*6, 3600) = 3600s (1h).
db = os.path.join(tempfile.mkdtemp(prefix="comms-orphan-"), "comms.db")
mirror = SqliteMirror(db, retain_secs=600.0)
now = time.time()
ORPHAN = max(600.0 * 6, 3600.0)   # 3600s

# --- THE BUG: wedged render, ANCIENT progressed_at, FRESH updated ---
# Simulate the immortal render: it last made real progress 2h ago (well past
# the 1h orphan window) but a view/recompute re-upserted it 1s ago so `updated`
# is fresh. Under the old (updated-keyed) sweep this row lived forever.
mirror.upsert({
    "id": "wedged-1", "status": "processing", "kind": "video",
    "progressed_at": now - 2 * 3600,   # ancient real progress
})
check("wedged row present before prune", _row_count(mirror, "wedged-1") == 1)
# upsert just stamped updated=now (the immortality mechanism) — confirm it.
with sqlite3.connect(db) as _c:
    upd = _c.execute("SELECT updated FROM jobs WHERE id=?", ("wedged-1",)).fetchone()[0]
check("wedged row has FRESH updated (immortality mechanism)", (now - upd) < 5)

mirror.prune()
check("BUGFIX: wedged render reaped by progress-sweep",
      _row_count(mirror, "wedged-1") == 0)

# --- NEGATIVE: healthy render, FRESH progressed_at, OLD updated ---
# A progressing stream must NEVER be reaped. Force an old `updated` directly so
# we're purely testing that the sweep ignores `updated` for active rows.
mirror.upsert({
    "id": "healthy-1", "status": "processing", "kind": "video",
    "progressed_at": now - 10,   # made progress 10s ago -> alive
})
with sqlite3.connect(db) as conn:
    conn.execute("UPDATE jobs SET updated=? WHERE id=?",
                 (now - 10 * 3600, "healthy-1"))   # ancient updated
mirror.prune()
check("healthy progressing render NOT reaped despite ancient `updated`",
      _row_count(mirror, "healthy-1") == 1)

# --- NEGATIVE: queued/pending job with ancient progressed_at NOT reaped ---
# A job waiting its turn is starved, not wedged. pending is not in _ORPHAN_ACTIVE.
mirror.upsert({
    "id": "queued-1", "status": "pending", "kind": "video",
    "progressed_at": now - 5 * 3600,
})
mirror.prune()
check("pending/queued job NOT reaped by progress-sweep",
      _row_count(mirror, "queued-1") == 1)

# --- FAIL-OPEN: active row with NULL progressed_at NOT reaped ---
# A NULL/garbage progress clock must not be reaped — we only reap when we
# positively know it's ancient.
mirror.upsert({
    "id": "noprog-1", "status": "processing", "kind": "video",
    # no progressed_at key at all
})
with sqlite3.connect(db) as conn:
    conn.execute("UPDATE jobs SET updated=? WHERE id=?",
                 (now - 10 * 3600, "noprog-1"))
mirror.prune()
check("active row with NULL progressed_at NOT reaped (fail-open)",
      _row_count(mirror, "noprog-1") == 1)

# --- terminal cleanup path still keyed on `updated` (unchanged) ---
mirror.upsert({
    "id": "done-1", "status": "done", "kind": "chat",
    "progressed_at": now,   # progressed recently, but terminal + old `updated`
})
with sqlite3.connect(db) as conn:
    conn.execute("UPDATE jobs SET updated=? WHERE id=?",
                 (now - 2 * 600, "done-1"))   # past retain_secs=600
mirror.prune()
check("terminal row past retain window still reaped (via `updated`)",
      _row_count(mirror, "done-1") == 0)

# --- MIGRATION: an existing populated DB WITHOUT the column migrates cleanly ---
legacy_db = os.path.join(tempfile.mkdtemp(prefix="comms-legacy-"), "comms.db")
with sqlite3.connect(legacy_db) as conn:
    # Old schema: no progressed_at column.
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL, "
        "status TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'chat', "
        "cancel_requested INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)")
    conn.execute(
        "INSERT INTO jobs (id, data, status, kind, cancel_requested, updated) "
        "VALUES (?,?,?,?,?,?)",
        ("legacy-1", "{}", "processing", "video", 0, now))

legacy = SqliteMirror(legacy_db, retain_secs=600.0)
# First touch triggers _ensure() -> _migrate(); must not raise.
legacy.prune()
check("legacy DB migrated: progressed_at column added", True)
with sqlite3.connect(legacy_db) as conn:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
check("migrated schema carries progressed_at", "progressed_at" in cols)
# Legacy active row has NULL progressed_at -> fail-open, survives.
check("legacy active row (NULL progressed_at) survives prune",
      _row_count(legacy, "legacy-1") == 1)
# A second _ensure()/migrate is a harmless no-op (idempotent).
legacy2 = SqliteMirror(legacy_db, retain_secs=600.0)
legacy2.prune()
check("re-migration is idempotent (no raise)", True)

print(f"\nALL {ok} CHECKS PASSED")
