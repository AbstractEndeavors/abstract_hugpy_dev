"""POST /llm/workers/<id>/reap-approve — central second-guard route regression.

Verifies the operator-approved eviction route wiring WITHOUT a live worker:
  * intersects approved keys with the freshly-recomputed proposal (drops keys no
    longer proposed — the render->approve race guard);
  * delegates the SURVIVORS to the same guarded reaper relay (/reap) that
    workers_reap uses, and folds approved/reaped/dropped into the result;
  * all-dropped -> early return (relay NEVER called, nothing deleted);
  * bad body -> 400; unknown worker -> 404.

Runs like the other tests here: venv/bin/python tests/test_reap_approve_route.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep audit writes out of the real projects tree.
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-reap-approve-test-")

import importlib
from flask import Flask, jsonify

wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

# --- collaborators stubbed at the module level (the route calls bare names) ---
_orig = (wr.get_worker, wr.worker_storage_view, wr._relay_worker_op)
relay_calls = []

def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
    relay_calls.append({"op_path": op_path, "body": dict(body), "action": action})
    return jsonify({
        "ok": True, "freed_bytes": 123,
        "results": [{"model_key": k, "ok": True, "freed_bytes": 1}
                    for k in body["model_keys"]],
    }), 200

PROPOSAL = {
    "over_budget": True,
    "proposed_evictions": [
        {"model_key": "a", "bytes": 10, "last_picked": 100.0},
        {"model_key": "b", "bytes": 20, "last_picked": 200.0},
    ],
}
try:
    wr.get_worker = lambda wid: {"id": wid, "name": "box"} if wid == "wid" else None
    wr.worker_storage_view = lambda wid: dict(PROPOSAL)
    wr._relay_worker_op = _fake_relay

    # 1) happy path: 'a' is proposed (kept), 'z' is not (dropped) -----------
    relay_calls.clear()
    r = client.post("/llm/workers/wid/reap-approve", json={"model_keys": ["a", "z"]})
    body = r.get_json()
    check("happy: 200", r.status_code == 200)
    check("happy: relay called exactly once", len(relay_calls) == 1)
    check("happy: relay hits the SAME guarded /reap path",
          relay_calls[0]["op_path"] == "/reap")
    check("happy: relay action is reap-approve",
          relay_calls[0]["action"] == "reap-approve")
    check("happy: only the intersected key is relayed (z dropped centrally)",
          relay_calls[0]["body"]["model_keys"] == ["a"])
    check("happy: result folds approved set", body["approved"] == ["a", "z"])
    check("happy: result folds reaped=intersected", body["reaped"] == ["a"])
    check("happy: result folds dropped (render->approve race guard)",
          body["dropped"] == ["z"])
    check("happy: reaper's typed freed_bytes passed through",
          body["freed_bytes"] == 123)

    # 2) all approved keys no longer proposed -> early return, NO relay -----
    relay_calls.clear()
    r = client.post("/llm/workers/wid/reap-approve", json={"model_keys": ["z", "q"]})
    body = r.get_json()
    check("all-dropped: 200", r.status_code == 200)
    check("all-dropped: relay NEVER called (nothing deleted)", relay_calls == [])
    check("all-dropped: freed_bytes 0", body["freed_bytes"] == 0)
    check("all-dropped: dropped carries both keys",
          set(body["dropped"]) == {"z", "q"} and body["reaped"] == [])
    check("all-dropped: explanatory note present", "note" in body)

    # 3) bad body -> 400 ----------------------------------------------------
    check("missing model_keys -> 400",
          client.post("/llm/workers/wid/reap-approve", json={}).status_code == 400)
    check("empty model_keys -> 400",
          client.post("/llm/workers/wid/reap-approve",
                      json={"model_keys": []}).status_code == 400)

    # 4) unknown worker -> 404 ---------------------------------------------
    check("unknown worker -> 404",
          client.post("/llm/workers/nope/reap-approve",
                      json={"model_keys": ["a"]}).status_code == 404)

    # 5) proposal recomputed empty (worker back under budget) -> early return
    relay_calls.clear()
    wr.worker_storage_view = lambda wid: {"over_budget": False,
                                          "proposed_evictions": []}
    r = client.post("/llm/workers/wid/reap-approve", json={"model_keys": ["a"]})
    check("under-budget-at-approve: nothing relayed",
          relay_calls == [] and r.get_json()["freed_bytes"] == 0)
finally:
    (wr.get_worker, wr.worker_storage_view, wr._relay_worker_op) = _orig

print(f"\nall {ok} checks passed")
