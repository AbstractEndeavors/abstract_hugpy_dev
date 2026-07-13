"""Download progress-honesty + staging-orphan-reaper regression (2026-07-12).

Yesterday's atomic-provisions change (imports/apis/download_models.py) lands
an in-flight download in a per-pid staging sibling ``<dest>.tmp-<pid>`` and
only renames it onto the final ``<dest>`` on completion (integrity fix — a
partial pull can never sit at a resolvable model path). The download-progress
reader (flask_app/.../downloads/cancelable_downloads.py) measured bytes at
``dest`` ONLY, so every in-flight download showed 0% until the finishing
rename.

This file exercises, without touching the real model store:
  * progress honesty  — the fixed `_progress_bytes` reads staging bytes while
    in flight and final-dest bytes once promoted, never both (rename-safe);
  * the orphan reaper — dead-pid + stale staging is removed, live-pid staging
    is left alone, young dead-pid staging survives the grace window;
  * adopt-on-resume   — a fresh run's staging dir adopts (renames onto
    itself) the newest dead-pid orphan for the SAME dest instead of
    re-fetching from zero, and ages out any other orphans past grace;
  * the discover-walk hook actually calls the reaper (wiring check, no real
    filesystem/network walk).

Runs like the other tests here: venv/bin/python tests/test_download_progress.py
"""
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import time
import shutil
import tempfile
import subprocess
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

dm = importlib.import_module("abstract_hugpy_dev.imports.apis.download_models")
gm = importlib.import_module("abstract_hugpy_dev.imports.apis.get_module")

# cancelable_downloads.py lives under flask_app.app.functions — importing it
# as a bare dotted path BEFORE the flask app has booted trips a pre-existing,
# unrelated import-order landmine (abstract_hugpy_dev.flask_app's `app`
# attribute gets shadowed by the third-party `flask.app` submodule partway
# through flask_app/__init__.py). Booting the app once first populates
# sys.modules with the correctly-resolved submodule, and importlib.import_module
# hits that sys.modules entry directly on the fast path. Reported separately —
# out of scope for this slice (progress path + staging reaper only).
importlib.import_module("abstract_hugpy_dev.flask_app.wsgi_app").get_hugpy_flask()
cd = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.downloads.cancelable_downloads")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

MB = 1024 * 1024


def wfile(path, mb=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * (mb * MB))


def dead_pid() -> int:
    """A pid that is GUARANTEED not alive: fork, exit immediately, reap."""
    child = os.fork()
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    return child


def age(path, seconds_ago):
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


tmp = tempfile.mkdtemp(prefix="hugpy-download-progress-")

# ===========================================================================
# PROGRESS HONESTY — the actual regression site (cancelable_downloads._progress_bytes)
# ===========================================================================
print("[progress honesty]")
dest = os.path.join(tmp, "models", "transformers", "own", "in-flight-model")
staged = dest + f".tmp-{os.getpid()}"
wfile(os.path.join(staged, "model-00001.safetensors"), 3)
wfile(os.path.join(staged, "model-00002.safetensors"), 2)
staged_total = 5 * MB

check("dest does not exist yet (still mid-download)", not os.path.exists(dest))
check("OLD read path (_dir_bytes(dest) alone) is the live 0% bug",
      cd._dir_bytes(dest) == 0)
check("FIXED read path (_progress_bytes) reports the staging bytes",
      cd._progress_bytes(dest) == staged_total)
check("download_models.staged_bytes agrees",
      dm.staged_bytes(dest) == staged_total)

# simulate _promote_staged's common-path rename (dest didn't exist -> os.rename)
os.rename(staged, dest)
check("post-promote: staging sibling is gone", not os.path.exists(staged))
check("post-promote: progress reads the final dest, same total (no double count)",
      cd._progress_bytes(dest) == staged_total)
check("post-promote: no staging bytes remain",
      dm.staged_bytes(dest) == 0)

# a second, freshly-arriving staging dir for a DIFFERENT in-flight attempt of
# the same dest (e.g. a resume after promote already happened) must not be
# double-counted against the now-complete dest
staged2 = dest + ".tmp-999999"
wfile(os.path.join(staged2, "extra.safetensors"), 1)
check("dest + a stray sibling sum correctly (no double count, no dedup needed)",
      cd._progress_bytes(dest) == staged_total + 1 * MB)
shutil.rmtree(staged2, ignore_errors=True)


# ===========================================================================
# ORPHAN REAPER — reap_orphaned_staging (store-wide sweep, hooked at discover)
# ===========================================================================
print("[reaper]")
root = os.path.join(tmp, "reaper-store")
models = os.path.join(root, "models", "transformers", "own")

dead1 = dead_pid()
old_dead_dir = os.path.join(models, "repo-a").rstrip("/") + f".tmp-{dead1}"
wfile(os.path.join(old_dead_dir, "half.bin"), 1)
age(old_dead_dir, 3600)                              # 1h old — well past grace

live_dir = os.path.join(models, "repo-b").rstrip("/") + f".tmp-{os.getpid()}"
wfile(os.path.join(live_dir, "half.bin"), 1)
age(live_dir, 3600)                                  # old mtime but LIVE pid

dead2 = dead_pid()
young_dead_dir = os.path.join(models, "repo-c").rstrip("/") + f".tmp-{dead2}"
wfile(os.path.join(young_dead_dir, "half.bin"), 1)
# fresh mtime (just written) — within the grace window

removed = dm.reap_orphaned_staging(root=root, grace_seconds=600)

check("dead-pid + stale (past grace) staging IS removed",
      old_dead_dir in removed and not os.path.exists(old_dead_dir))
check("live-pid staging is NEVER touched (even with an old mtime)",
      live_dir not in removed and os.path.exists(live_dir))
check("dead-pid + young (within grace) staging is KEPT",
      young_dead_dir not in removed and os.path.exists(young_dead_dir))

# age the young one out and re-sweep with a short grace -> now it goes too
age(young_dead_dir, 5)
removed2 = dm.reap_orphaned_staging(root=root, grace_seconds=1)
check("previously-young dead-pid staging is reaped once it clears grace",
      young_dead_dir in removed2 and not os.path.exists(young_dead_dir))
check("live-pid staging still untouched after the second sweep",
      os.path.exists(live_dir))
shutil.rmtree(live_dir, ignore_errors=True)


# ===========================================================================
# ADOPT-ON-RESUME — _adopt_or_reap_staging (per-dest hook before a new pull)
# ===========================================================================
print("[adopt-on-resume]")
adest = os.path.join(tmp, "models", "transformers", "own", "resumable-model")

dead_old = dead_pid()
orphan_old = adest + f".tmp-{dead_old}"
wfile(os.path.join(orphan_old, "shard1.bin"), 1)
# left with a FRESH mtime here on purpose: it must survive the upcoming
# grace_seconds=600 adopt call below (aged out only in the later, separate
# short-grace sweep) — orphan_new is created after it so it's naturally the
# newer of the two, no manual aging needed to pick the adoption winner.

dead_new = dead_pid()
orphan_new = adest + f".tmp-{dead_new}"
wfile(os.path.join(orphan_new, "shard1.bin"), 4)     # further along -> newest
# fresh mtime (newest by construction AND by mtime)

fresh_staged = dm._staging_dir(adest)                # this run's own pid
result = dm._adopt_or_reap_staging(adest, fresh_staged, grace_seconds=600)

check("adopt returns the caller's own staged path",
      result == fresh_staged)
check("the NEWEST dead orphan was renamed onto the fresh staged path (adopted)",
      os.path.isdir(fresh_staged)
      and os.path.exists(os.path.join(fresh_staged, "shard1.bin"))
      and os.path.getsize(os.path.join(fresh_staged, "shard1.bin")) == 4 * MB)
check("the adopted orphan's old path is gone (it WAS the rename, not a copy)",
      not os.path.exists(orphan_new))
check("the OLDER orphan is untouched (still within its own grace check here)",
      os.path.exists(orphan_old))

# re-run with a short grace: the older, non-adopted orphan should now age out
age(orphan_old, 3600)
result2 = dm._adopt_or_reap_staging(adest + "-other", adest + "-other.tmp-nope",
                                     grace_seconds=1)
check("adopt-or-reap is a no-op when there are no siblings for that dest",
      result2 == adest + "-other.tmp-nope")
# direct grace check on the untouched older orphan via the store-wide reaper
removed4 = dm.reap_orphaned_staging(root=tmp, grace_seconds=1)
check("the older, non-adopted orphan is reaped by a subsequent sweep",
      orphan_old in removed4 and not os.path.exists(orphan_old))
shutil.rmtree(fresh_staged, ignore_errors=True)

# a LIVE sibling must never be adopted or reaped, even if it's the "newest"
print("[adopt-on-resume: live sibling is untouchable]")
ldest = os.path.join(tmp, "models", "transformers", "own", "live-contended-model")
liveproc = subprocess.Popen(["sleep", "20"])
try:
    live_sibling = ldest + f".tmp-{liveproc.pid}"
    wfile(os.path.join(live_sibling, "shard1.bin"), 2)
    fresh2 = dm._staging_dir(ldest)
    result3 = dm._adopt_or_reap_staging(ldest, fresh2, grace_seconds=0)
    check("a live pid's staging sibling is left alone entirely",
          not os.path.exists(fresh2) and os.path.exists(live_sibling))
    check("adopt-or-reap still returns the caller's staged path",
          result3 == fresh2)
finally:
    liveproc.terminate()
    liveproc.wait()
    shutil.rmtree(ldest + f".tmp-{liveproc.pid}", ignore_errors=True)


# ===========================================================================
# DISCOVER-WALK HOOK WIRING — reaper actually fires from discover_model(s)
# ===========================================================================
print("[discover-walk hook wiring]")
calls = []
_orig = dm.reap_orphaned_staging
dm.reap_orphaned_staging = lambda *a, **k: (calls.append((a, k)) or [])
try:
    gm._reap_orphaned_staging_quiet()
finally:
    dm.reap_orphaned_staging = _orig
check("the discover-walk hook calls the staging reaper", len(calls) == 1)


shutil.rmtree(tmp, ignore_errors=True)
print(f"\nALL {ok} CHECKS PASSED")
