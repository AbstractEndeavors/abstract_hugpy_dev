"""POST /llm/workers/<id>/residency-all — bulk-residency route (todo t12).

The console's Serving table lets the operator multi-select models and change
their RESIDENCY tier in ONE action. That action must ride ONE /ops/config relay
(one agent re-exec), NOT N single-model POSTs — the same batching _relay_pin_all
does for pins. This regresses the route/relay wiring WITHOUT a live worker:

  * one relay call carrying the WHOLE selection as a single residency map;
  * on-demand normalizes to the null wire value (the agent's clears-the-override
    convention); "static" is stored verbatim; residency ONLY — never `pinned`;
  * off-worker keys in the selection are dropped (render->click staleness),
    reported as `skipped`, never written into the settings map;
  * per-model results / counts / mode / restarting surfaced like pin-all;
  * a relay failure still returns the full per-model map (never a bare 5xx);
  * bad body -> 400; bad mode -> 400; unknown worker -> 404.

Runs like the other tests here: venv/bin/python tests/test_bulk_residency_route.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep audit writes out of the real projects tree.
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-bulk-residency-test-")

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

# The worker has models a,b,c designated; z is NOT designated to it.
WORKER = {"id": "wid", "name": "box", "models": ["a", "b", "c"]}

_orig = (wr.get_worker, wr._relay_worker_op)
relay_calls = []

def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
    relay_calls.append({"op_path": op_path, "body": dict(body),
                        "action": action, "retry": retry_on_connect})
    # The agent replies {ok, settings, restarting} to /ops/config.
    return jsonify({"ok": True, "settings": {}, "restarting": True}), 200

try:
    wr.get_worker = lambda wid: dict(WORKER) if wid == "wid" else None
    wr._relay_worker_op = _fake_relay

    # 1) static on a subset: ONE relay, one residency map, residency-only ------
    relay_calls.clear()
    r = client.post("/llm/workers/wid/residency-all",
                    json={"model_keys": ["a", "b"], "mode": "static"})
    body = r.get_json()
    check("static: 200", r.status_code == 200)
    check("static: exactly ONE relay call (not one per model)", len(relay_calls) == 1)
    check("static: relay hits /ops/config", relay_calls[0]["op_path"] == "/ops/config")
    check("static: config-style relay gets the apply-blip retry",
          relay_calls[0]["retry"] is True)
    check("static: whole selection in ONE residency map",
          relay_calls[0]["body"] == {"residency": {"a": "static", "b": "static"}})
    check("static: residency ONLY — never touches pinned",
          "pinned" not in relay_calls[0]["body"])
    check("static: mode echoed", body["mode"] == "static")
    check("static: per-model results all ok",
          body["results"] == {"a": "ok", "b": "ok"})
    check("static: counts", body["counts"] == {"ok": 2, "error": 0, "total": 2})
    check("static: restarting flag passed through", body["restarting"] is True)

    # 2) on-demand normalizes to the null wire value (clears the override) ------
    relay_calls.clear()
    r = client.post("/llm/workers/wid/residency-all",
                    json={"model_keys": ["a", "c"], "mode": "on-demand"})
    body = r.get_json()
    check("on-demand: 200", r.status_code == 200)
    check("on-demand: null wire value for every selected key",
          relay_calls[0]["body"] == {"residency": {"a": None, "c": None}})
    check("on-demand: mode label", body["mode"] == "on-demand")
    check("on-demand: counts", body["counts"] == {"ok": 2, "error": 0, "total": 2})

    # 3) off-worker key in the selection is dropped (render->click staleness) ---
    relay_calls.clear()
    r = client.post("/llm/workers/wid/residency-all",
                    json={"model_keys": ["a", "z"], "mode": "static"})
    body = r.get_json()
    check("stale: only designated key relayed (z dropped)",
          relay_calls[0]["body"] == {"residency": {"a": "static"}})
    check("stale: dropped key reported as skipped", body["skipped"] == ["z"])
    check("stale: results cover only the designated key",
          body["results"] == {"a": "ok"})

    # 4) all keys off-worker -> no relay, honest note --------------------------
    relay_calls.clear()
    r = client.post("/llm/workers/wid/residency-all",
                    json={"model_keys": ["z", "q"], "mode": "static"})
    body = r.get_json()
    check("all-stale: 200", r.status_code == 200)
    check("all-stale: relay NEVER called", relay_calls == [])
    check("all-stale: counts zero", body["counts"] == {"ok": 0, "error": 0, "total": 0})
    check("all-stale: explanatory note", "note" in body)
    check("all-stale: both keys skipped", set(body["skipped"]) == {"z", "q"})

    # 5) relay FAILS -> still returns the full per-model map, every key errored -
    relay_calls.clear()
    def _fail_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
        return jsonify({"ok": False, "error": {
            "code": "AgentRestarting", "message": "worker agent is restarting"}}), 503
    wr._relay_worker_op = _fail_relay
    r = client.post("/llm/workers/wid/residency-all",
                    json={"model_keys": ["a", "b"], "mode": "static"})
    body = r.get_json()
    check("relay-fail: still 200 (structured body carries the failure)",
          r.status_code == 200)
    check("relay-fail: ok is False", body["ok"] is False)
    check("relay-fail: every selected model errored with the relay message",
          all("restarting" in v for v in body["results"].values())
          and set(body["results"]) == {"a", "b"})
    check("relay-fail: counts all error",
          body["counts"] == {"ok": 0, "error": 2, "total": 2})
    check("relay-fail: error surfaced for fetchJson too",
          body.get("error", {}).get("code") == "AgentRestarting")
    wr._relay_worker_op = _fake_relay

    # 6) bad body / bad mode / unknown worker ----------------------------------
    check("missing model_keys -> 400",
          client.post("/llm/workers/wid/residency-all",
                      json={"mode": "static"}).status_code == 400)
    check("empty model_keys -> 400",
          client.post("/llm/workers/wid/residency-all",
                      json={"model_keys": [], "mode": "static"}).status_code == 400)
    check("bad mode -> 400 (never silently clears)",
          client.post("/llm/workers/wid/residency-all",
                      json={"model_keys": ["a"], "mode": "bogus"}).status_code == 400)
    check("unknown worker -> 404",
          client.post("/llm/workers/nope/residency-all",
                      json={"model_keys": ["a"], "mode": "static"}).status_code == 404)

    # 7) the normalizer maps every accepted on-demand synonym to None ----------
    for m in (None, "", "on-demand", "on_demand", "serving", "warm"):
        check(f"normalize {m!r} -> None (on-demand default)",
              wr._normalize_residency(m) is None)
    check("normalize 'static' -> 'static'",
          wr._normalize_residency("static") == "static")
    check("normalize garbage -> sentinel (route turns it into a 400)",
          wr._normalize_residency("bogus") == "__invalid__")

    # 8) the route is operator-gated (same tier as config/pin-all) -------------
    oa = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.operator_auth")
    def _gated(path, method):
        return any(method in methods and rx.match(path)
                   for methods, rx in oa._SENSITIVE)
    check("residency-all POST is in _SENSITIVE (operator-gated)",
          _gated("/llm/workers/w1/residency-all", "POST"))
    check("residency-all GET is NOT gated (no such route, but never open-write)",
          not _gated("/llm/workers/w1/residency-all", "GET"))
finally:
    (wr.get_worker, wr._relay_worker_op) = _orig

print(f"\nall {ok} checks passed")
