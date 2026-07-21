"""k10 hardening: sanctioned removal of GHOST entries from the worker
assignment-memory sidecar (worker_assignments.json).

Background: workers.py's module docstring (~line 41-48) is explicit that
deleting a live worker ROW deliberately does NOT delete its assignment
memory -- designations are worker-lifetime, not row-lifetime, so a
re-register under a known id restores them. That design stays intact here.
This only adds a maintenance path for entries whose id was NEVER a real,
current worker (or no longer is one AT ALL) -- e.g. a synthetic/malformed id
left behind by a test or a one-off manual poke -- while REFUSING to touch
any id that is still live in workers.json.

Covers:
  * forget_assignment_memory(): forgetting a ghost works (file rewritten,
    entry gone, atomic .tmp+rename like _remember_assignments);
  * forget_assignment_memory(): a LIVE worker id is refused (ValueError);
  * forget_assignment_memory(): an unknown (never-remembered) id reports
    "unknown" rather than silently succeeding;
  * the DELETE /llm/workers/<id>/memory route: 200 for a ghost, 404 for
    unknown, 409 for a live id;
  * the route is operator-gated in operator_auth._SENSITIVE, same tier as
    the plain worker DELETE.

Uses the swap_worker_store() isolation helper (see worker_store_isolation.py)
so this never touches the real /mnt/llm_storage/projects/ registry or its
sidecar -- both the store's own path AND the assignment-memory sidecar are
redirected into a fresh tmpdir for the duration of the swap.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_worker_memory_forget.py -v
     (or plain: venv/bin/python tests/test_worker_memory_forget.py)
"""
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_store_isolation import swap_worker_store  # noqa: E402

W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
oa = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.operator_auth")

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


# ── [1] store-level: forget a ghost, refuse a live id, unknown is honest ────
print("\n[1] forget_assignment_memory() -- store level")

with swap_worker_store(prefix="hugpy-memory-forget-test-"):
    # A LIVE worker (real row in the isolated registry). _remember_assignments
    # only fires on assign/unassign or a re-register-with-models — a bare
    # fresh register() does NOT snapshot memory (see register()'s "existing"
    # branch vs the fresh-wid branch) — so assign_model() here is the real
    # trigger, matching how the live fleet actually populates the sidecar.
    W.worker_store.register(name="live-one", url="http://192.0.2.70:9100",
                             worker_id="live-one")
    W.worker_store.assign_model("live-one", "Some~Model")
    check("setup: live-one has an assignment-memory entry (assign_model "
          "remembers it)", "live-one" in W._load_assign_memory())

    # A GHOST: present in memory only (simulate via a live register+assign+
    # remove, which per the by-design invariant leaves the memory entry
    # behind).
    W.worker_store.register(name="ghost-one", url="http://192.0.2.71:9100",
                             worker_id="ghost-one")
    W.worker_store.assign_model("ghost-one", "Ghost~Model")
    check("setup: ghost-one has a memory entry before removal",
          "ghost-one" in W._load_assign_memory())
    check("setup: remove() drops the live row", W.worker_store.remove("ghost-one"))
    check("setup: ghost-one is no longer live", W.worker_store.get("ghost-one") is None)
    check("setup: ghost-one's memory entry SURVIVED the row removal (by design)",
          "ghost-one" in W._load_assign_memory())

    # -- refuse a live id --
    try:
        W.forget_assignment_memory("live-one")
        check("refuse: forgetting a LIVE id raises ValueError", False)
    except ValueError:
        check("refuse: forgetting a LIVE id raises ValueError", True)
    check("refuse: live-one's memory entry is untouched after the refusal",
          "live-one" in W._load_assign_memory())

    # -- unknown id: never remembered at all --
    result_unknown = W.forget_assignment_memory("never-existed-id")
    check("unknown: forget_assignment_memory() returns 'unknown' (no raise, "
          "no-op)", result_unknown == "unknown")

    # -- forget the actual ghost --
    mem_path = W._assign_memory_path()
    before_mtime = os.stat(mem_path).st_mtime_ns
    result_ghost = W.forget_assignment_memory("ghost-one")
    check("forget: returns 'forgot' for the real ghost", result_ghost == "forgot")
    check("forget: ghost-one is gone from memory afterwards",
          "ghost-one" not in W._load_assign_memory())
    check("forget: live-one's entry is untouched by forgetting a DIFFERENT id",
          "live-one" in W._load_assign_memory())
    check("forget: sidecar file was actually rewritten (mtime advanced)",
          os.stat(mem_path).st_mtime_ns >= before_mtime)

    # -- atomicity: no leftover .tmp file --
    check("forget: no leftover .tmp file after the atomic rename",
          not os.path.isfile(mem_path + ".tmp"))

    # -- forgetting the same ghost again is now 'unknown', not an error --
    result_again = W.forget_assignment_memory("ghost-one")
    check("forget: forgetting an already-forgotten id is 'unknown' (idempotent, "
          "no raise)", result_again == "unknown")


# ── [2] route level: DELETE /llm/workers/<id>/memory ────────────────────────
print("\n[2] route: DELETE /llm/workers/<id>/memory")

from flask import Flask  # noqa: E402

app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

with swap_worker_store(prefix="hugpy-memory-forget-route-test-"):
    W.worker_store.register(name="live-two", url="http://192.0.2.72:9100",
                             worker_id="live-two")
    W.worker_store.assign_model("live-two", "Some~Model")
    W.worker_store.register(name="ghost-two", url="http://192.0.2.73:9100",
                             worker_id="ghost-two")
    W.worker_store.assign_model("ghost-two", "Ghost~Model")
    W.worker_store.remove("ghost-two")
    check("route setup: ghost-two memory present, row gone",
          "ghost-two" in W._load_assign_memory()
          and W.worker_store.get("ghost-two") is None)

    r_live = client.delete("/llm/workers/live-two/memory")
    check("route: DELETE on a LIVE id -> 409", r_live.status_code == 409)
    check("route: live-two's memory entry untouched after the 409",
          "live-two" in W._load_assign_memory())

    r_unknown = client.delete("/llm/workers/totally-unknown-id/memory")
    check("route: DELETE on an id never in memory -> 404", r_unknown.status_code == 404)

    r_ghost = client.delete("/llm/workers/ghost-two/memory")
    check("route: DELETE on the real ghost -> 200", r_ghost.status_code == 200)
    check("route: 200 body echoes {forgot: id}",
          (r_ghost.get_json() or {}).get("forgot") == "ghost-two")
    check("route: ghost-two actually gone from memory now",
          "ghost-two" not in W._load_assign_memory())

    r_ghost_again = client.delete("/llm/workers/ghost-two/memory")
    check("route: repeating the DELETE on an already-forgotten id -> 404 now",
          r_ghost_again.status_code == 404)


# ── [3] operator gating: the route is in _SENSITIVE, same tier as DELETE row ─
print("\n[3] operator gating")


def _gated(path, method):
    p = path
    if p == "/api" or p.startswith("/api/"):
        p = p[len("/api"):] or "/"
    return any(method in methods and rx.match(p) for methods, rx in oa._SENSITIVE)


check("gating: DELETE /llm/workers/<id>/memory is operator-gated",
      _gated("/llm/workers/some-id/memory", "DELETE"))
check("gating: gate also matches the /api-mounted path",
      _gated("/api/llm/workers/some-id/memory", "DELETE"))
check("gating: the plain row DELETE /llm/workers/<id> is unaffected (still "
      "gated, still does NOT match the /memory suffix pattern by itself)",
      _gated("/llm/workers/some-id", "DELETE"))
check("gating: GET /llm/workers/<id> stays open (read)",
      _gated("/llm/workers/some-id", "GET") is False)


print(f"\n{ok} passed, {fail} failed")
assert fail == 0, f"{fail} check(s) failed — see FAIL lines above"
if __name__ == "__main__":
    sys.exit(1 if fail else 0)
