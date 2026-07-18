"""k3: proves the WorkerStore test-isolation landmine is closed.

Background: WorkerStore's default path (and its assignment-memory sidecar)
resolves through ``schemas.settings.manifest_path`` -> ``PROJECTS_HOME``,
computed ONCE at import time via ``abstract_essentials.get_env_value()`` -- a
``.env``-FILE reader that never consults ``os.environ``. The
``os.environ["PROJECTS_HOME"] = tempfile.mkdtemp()`` idiom used across this
suite is therefore a NO-OP for WorkerStore: it silently falls through to the
REAL ``/mnt/llm_storage/projects/`` registry this box's live fleet uses.
Proven incident: k2's first ``test_block_propagation.py`` run registered a
real ``wk-prop`` row into the LIVE registry.

This test is deliberately conservative about the live file: it NEVER opens it
for writing, only ``os.stat``s it and reads its worker-id set, both before and
after exercising the fix. The actual mutating exercise (register / assign /
unassign / heartbeat -- the exact call classes that leak, per
``worker_store_isolation.py``'s writeup) runs ONLY against the isolated
tmpdir-backed store from ``swap_worker_store`` / ``isolated_worker_store``.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_worker_store_isolation.py -v
     (or plain: venv/bin/python tests/test_worker_store_isolation.py)
"""
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib as _il  # noqa: E402

from worker_store_isolation import isolated_worker_store, swap_worker_store  # noqa: E402

W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print("  ok  ", name)
    else:
        fail += 1
        print("  FAIL", name)


# ── sentinel: snapshot the REAL live paths, read-only ───────────────────────
# Whatever schemas.settings.manifest_path resolves to AT THIS MOMENT (before
# any test in this process has redirected it) is the live path. os.stat only
# -- this file is never opened for writing by this test.
_real_workers_path = W._default_workers_path()
_real_assign_mem_path = W._assign_memory_path()


def _snapshot(path):
    """(exists, mtime_ns, size, sorted-worker-ids-or-None) -- read-only."""
    try:
        st = os.stat(path)
    except OSError:
        return (False, None, None, None)
    ids = None
    if path == _real_workers_path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ids = sorted(w.get("id") for w in data.get("workers", []) if w.get("id"))
        except (OSError, ValueError):
            ids = None
    return (True, st.st_mtime_ns, st.st_size, ids)


before_workers = _snapshot(_real_workers_path)
before_assign_mem = _snapshot(_real_assign_mem_path)

print(f"real workers.json:          {_real_workers_path}  {before_workers[:3]}")
print(f"real worker_assignments.json: {_real_assign_mem_path}  {before_assign_mem[:3]}")


# ── [1] the OLD (broken) idiom is provably a no-op for WorkerStore ──────────
print("\n[1] the PROJECTS_HOME-only idiom does not isolate WorkerStore")

_bogus = tempfile.mkdtemp(prefix="hugpy-isolation-proof-bogus-")
_orig_environ_ph = os.environ.get("PROJECTS_HOME")
try:
    os.environ["PROJECTS_HOME"] = _bogus
    # get_env_value() (what PROJECTS_HOME/MODELS_DICT_PATH/settings.manifest_path
    # were computed from, once, at import time) never reads os.environ -- so
    # the resolved default path is UNCHANGED by this assignment. Pure read
    # (string join), no I/O against either path.
    check("old idiom: _default_workers_path() ignores the just-set env var "
          "(still the real, pre-existing path)",
          W._default_workers_path() == _real_workers_path)
    check("old idiom: _assign_memory_path() ignores it too (same landmine)",
          W._assign_memory_path() == _real_assign_mem_path)
finally:
    if _orig_environ_ph is None:
        os.environ.pop("PROJECTS_HOME", None)
    else:
        os.environ["PROJECTS_HOME"] = _orig_environ_ph


# ── [2] isolated_worker_store() actually redirects both paths ───────────────
print("\n[2] isolated_worker_store() redirects WorkerStore + the sidecar")

store, tmp = isolated_worker_store(prefix="hugpy-isolation-proof-")
check("isolated store's own path is inside the tmpdir, not the live path",
      store._path.startswith(tmp) and store._path != _real_workers_path)
check("assignment-memory sidecar now resolves inside the SAME tmpdir "
      "(the second landmine -- path= alone does not fix this)",
      W._assign_memory_path().startswith(tmp))
check("...and is provably NOT the real path",
      W._assign_memory_path() != _real_assign_mem_path)


# ── [3] a full mutating cycle -- the exact classes that leaked in k2/k3 ─────
print("\n[3] register/assign/unassign/heartbeat through the isolated store")

v1 = store.register(name="ghost-a", url="http://192.0.2.50:9100",
                     worker_id="ghost-a", models=["Some~Model"])
check("register (new worker) succeeds against the isolated store", v1 is not None)
# Re-register the SAME id with models -- the "existing" branch that fires
# _remember_assignments (WRITE class #1).
v1b = store.register(name="ghost-a", url="http://192.0.2.50:9100",
                      worker_id="ghost-a", models=["Some~Model", "Other~Model"])
check("re-register (existing branch, triggers _remember_assignments) "
      "still lands in the tmpdir", "Other~Model" in v1b["models"])

store.register(name="ghost-b", url="http://192.0.2.51:9100", worker_id="ghost-b")
av = store.assign_model("ghost-b", "Assigned~Model")
check("assign_model (WRITE class #2) succeeds against the isolated store",
      av is not None and "Assigned~Model" in av["models"])
uv = store.unassign_model("ghost-b", "Assigned~Model")
check("unassign_model (WRITE class #2, the exact test_storage_budget.py call) "
      "succeeds against the isolated store",
      uv is not None and "Assigned~Model" not in uv["models"])
hv = store.heartbeat("ghost-b", loaded_models=["Assigned~Model"])
check("heartbeat succeeds against the isolated store", hv is not None)

# The sidecar file the mutating calls above wrote to -- confirm it landed in
# the tmpdir and actually contains our synthetic id (proves the write really
# happened, just not where it used to).
assign_mem_path = W._assign_memory_path()
check("assignment-memory sidecar file was actually written, inside the tmpdir",
      os.path.isfile(assign_mem_path) and assign_mem_path.startswith(tmp))
with open(assign_mem_path, "r", encoding="utf-8") as f:
    mem = json.load(f)
check("...and contains the synthetic worker id (real write, real proof)",
      "ghost-a" in mem)


# ── [4] swap_worker_store() restores the module singleton + sidecar path ────
print("\n[4] swap_worker_store() drives module-level fns, then restores")

_orig_store_obj = W.worker_store
_orig_manifest_path = W.settings.manifest_path
with swap_worker_store(prefix="hugpy-isolation-proof-swap-") as swapped:
    check("inside the context: W.worker_store IS the swapped instance",
          W.worker_store is swapped)
    check("inside the context: manifest_path points into a fresh tmpdir",
          W.settings.manifest_path != _orig_manifest_path)
    # Drive it through the MODULE-LEVEL wrapper functions (what Flask routes
    # call) -- this is precisely the path test_model_block.py's /assign route
    # exercises after an unblock.
    W.worker_store.register(name="w1", url="http://192.0.2.60:9100", worker_id="w1")
    view = W.assign_model("w1", "Routed~Model")
    check("module-level assign_model() (route path) writes through the "
          "swapped singleton", view is not None and "Routed~Model" in view["models"])

check("after the context: W.worker_store is restored to the original object",
      W.worker_store is _orig_store_obj)
check("after the context: manifest_path is restored",
      W.settings.manifest_path == _orig_manifest_path)
# NB: compare against _orig_manifest_path (what it was just before THIS
# context), not _real_workers_path from the top of the file -- step [2]'s
# isolated_worker_store() call deliberately does not restore (that's its
# documented, by-design difference from swap_worker_store), so by this point
# in the script settings.manifest_path was already redirected before we ever
# entered this section. swap_worker_store's contract is "restores to
# whatever it was on entry", not "restores to the process's original value".
check("after the context: _default_workers_path() resolves back to what it "
      "was just before this swap (swap_worker_store's actual restore contract)",
      W._default_workers_path() == os.path.join(
          os.path.dirname(_orig_manifest_path), "workers.json"))


# ── [5] THE PROOF: the real live files were never touched ───────────────────
print("\n[5] the real registry + sidecar are untouched by any of the above")

after_workers = _snapshot(_real_workers_path)
after_assign_mem = _snapshot(_real_assign_mem_path)

check("real workers.json: exists before == exists after",
      before_workers[0] == after_workers[0])
if before_workers[0]:
    # mtime/size are ADVISORY only -- this is a live API process's registry,
    # so a real worker heartbeat landing in this window would legitimately
    # bump mtime without any test involvement. The AUTHORITATIVE check is the
    # worker-id set: none of this test's synthetic ids ("ghost-a", "ghost-b",
    # "w1", ...) may appear, and the real id set must be byte-identical.
    check("real workers.json: worker-id SET is byte-identical before/after "
          "(the authoritative check -- no ghost row, none missing)",
          before_workers[3] == after_workers[3] and after_workers[3] is not None)
    _synthetic = {"ghost-a", "ghost-b", "w1"}
    check("real workers.json: none of this test's synthetic ids leaked in",
          not (_synthetic & set(after_workers[3] or [])))
    if before_workers[1] == after_workers[1]:
        print("      (bonus: mtime also identical -- no write landed in this "
              "window at all, not even a legitimate heartbeat)")

check("real worker_assignments.json: exists before == exists after",
      before_assign_mem[0] == after_assign_mem[0])
if before_assign_mem[0]:
    check("real worker_assignments.json: byte size unchanged",
          before_assign_mem[2] == after_assign_mem[2])
    if before_assign_mem[1] == after_assign_mem[1]:
        print("      (mtime also identical)")
    else:
        # size-unchanged but mtime moved would still be surprising for a file
        # only ever written by _remember_assignments under a real assign/
        # unassign/re-register -- flag it loudly rather than pass silently.
        print("      NOTE: mtime moved with size unchanged -- re-verify manually "
              "if this repeats (could be an atomic rewrite of identical content "
              "by live traffic, but worth a second look).")

print(f"\n{ok} passed, {fail} failed")
assert fail == 0, f"{fail} check(s) failed — see FAIL lines above"
if __name__ == "__main__":
    # Only exit-code here, not under pytest collection (an unconditional
    # sys.exit() at module scope aborts pytest's collector with an
    # INTERNALERROR instead of a normal per-file failure — the assert above
    # already fails the file under pytest; this just keeps `python
    # tests/test_worker_store_isolation.py` script-runnable with a real exit
    # code the way every sibling test in this suite is).
    sys.exit(1 if fail else 0)
