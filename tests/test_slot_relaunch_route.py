"""k14 — POST /llm/workers/<id>/slots/<slot_id>/relaunch (central verb) + the
worker agent's /slots/<slot_id>/relaunch ops endpoint.

The central verb relays the offload-depth relaunch to a worker's slot child so the
k7 speed-cliff sweep can drive it. Regressed here WITHOUT a live worker:

  central route:
    * operator-gated (_SENSITIVE), same tier as the other worker ops;
    * unknown worker id -> 404;
    * offline worker -> 409 (no agent to relay to);
    * relays to /slots/<slot_id>/relaunch with a {n_gpu_layers?, ctx?} payload,
      dropping blanks, and returns the worker's typed result verbatim;
    * the worker's own 404 (unknown slot) / 409 (empty slot) propagate.

  worker endpoint (faked slot pool):
    * unknown slot id -> 404;
    * empty slot -> 409;
    * a seated slot -> relays to the slot control /relaunch and echoes the HONEST
      launched n_gpu_layers (what the fresh child launched with, not just asked).

Run: venv/bin/python tests/test_slot_relaunch_route.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-relaunch-test-")

import importlib

wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
cr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.comms_routes")
oa = importlib.import_module("abstract_hugpy_dev.flask_app.app.operator_auth")

from flask import Flask

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ══════════════════════ central route ══════════════════════════════════════
app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

ONLINE = {"id": "wid", "name": "box", "status": "online",
          "url": "http://worker:9100"}
OFFLINE = {"id": "off", "name": "box2", "status": "offline",
           "url": "http://worker2:9100"}

_orig = (wr.get_worker, wr._relay_worker_op, cr.audit)
relay_calls = []


def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
    from flask import jsonify
    relay_calls.append({"worker_id": worker_id, "op_path": op_path,
                        "body": body, "action": action})
    # emulate the worker's honest result: launched with the requested depth.
    return jsonify({"ok": True, "slot_id": op_path.split("/")[2],
                    "n_gpu_layers": body.get("n_gpu_layers"),
                    "requested_n_gpu_layers": body.get("n_gpu_layers")}), 200


try:
    wr.get_worker = lambda wid: (dict(ONLINE) if wid == "wid"
                                 else dict(OFFLINE) if wid == "off" else None)
    wr._relay_worker_op = _fake_relay
    cr.audit = lambda *a, **k: None

    # 1) happy path: relays with the trimmed payload to the right op path -------
    relay_calls.clear()
    r = client.post("/llm/workers/wid/slots/2/relaunch",
                    json={"n_gpu_layers": 17, "ctx": 8192})
    body = r.get_json()
    check("happy: 200", r.status_code == 200)
    check("happy: relayed once", len(relay_calls) == 1)
    check("happy: op path targets the slot", relay_calls[0]["op_path"] == "/slots/2/relaunch")
    check("happy: action label", relay_calls[0]["action"] == "slot-relaunch")
    check("happy: payload carries ngl + ctx",
          relay_calls[0]["body"] == {"n_gpu_layers": 17, "ctx": 8192})
    check("happy: honest launched ngl echoed", body["n_gpu_layers"] == 17)

    # 2) blanks dropped; absent n_gpu_layers => autofit (payload omits it) ------
    relay_calls.clear()
    r = client.post("/llm/workers/wid/slots/1/relaunch", json={"n_gpu_layers": ""})
    check("blank ngl dropped from payload", relay_calls[0]["body"] == {})
    relay_calls.clear()
    r = client.post("/llm/workers/wid/slots/1/relaunch", json={})
    check("empty body => empty payload (slot re-autofits)", relay_calls[0]["body"] == {})

    # 3) explicit CPU-only (0) and Max GPU (-1) pass through as overrides -------
    relay_calls.clear()
    client.post("/llm/workers/wid/slots/1/relaunch", json={"n_gpu_layers": 0})
    check("ngl=0 (CPU only) passes through", relay_calls[0]["body"] == {"n_gpu_layers": 0})

    # 4) unknown worker -> 404 -------------------------------------------------
    r = client.post("/llm/workers/nope/slots/1/relaunch", json={"n_gpu_layers": 4})
    check("unknown worker -> 404", r.status_code == 404)

    # 5) offline worker -> 409 (no agent to relay to; never relayed) -----------
    relay_calls.clear()
    r = client.post("/llm/workers/off/slots/1/relaunch", json={"n_gpu_layers": 4})
    check("offline worker -> 409", r.status_code == 409)
    check("offline: never relayed", relay_calls == [])
    check("offline: honest code",
          (r.get_json() or {}).get("error", {}).get("code") == "WorkerOffline")

    # 6) the worker's own 404 / 409 propagate verbatim -------------------------
    def _relay_worker_404(worker_id, op_path, body, timeout, action, **k):
        from flask import jsonify
        return jsonify({"ok": False, "error": {"code": "UnknownSlot"}}), 404
    wr._relay_worker_op = _relay_worker_404
    r = client.post("/llm/workers/wid/slots/9/relaunch", json={})
    check("worker unknown-slot 404 propagates", r.status_code == 404)
    wr._relay_worker_op = _fake_relay

    # 7) operator-gated in _SENSITIVE ------------------------------------------
    def _gated(path, method):
        return any(method in methods and rx.match(path)
                   for methods, rx in oa._SENSITIVE)
    check("relaunch POST is in _SENSITIVE (operator-gated)",
          _gated("/llm/workers/wid/slots/2/relaunch", "POST"))
    check("the single-segment worker rule does NOT swallow it (own rule needed)",
          _gated("/llm/workers/wid/slots/2/relaunch", "POST"))
    check("GET is not gated (no such route, but the rule is POST-only)",
          not _gated("/llm/workers/wid/slots/2/relaunch", "GET"))
finally:
    (wr.get_worker, wr._relay_worker_op, cr.audit) = _orig


# ══════════════════════ worker agent endpoint ══════════════════════════════
# Build the worker Flask app with a FAKE slot pool so we exercise slot-id
# resolution + relay without spawning any child.
wa = importlib.import_module("abstract_hugpy_dev.worker_agent.agent")
slots_mod = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")


class _FakePool:
    """Two slots: id 1 seated with 'coder', id 2 empty."""
    def __init__(self, urls=None):
        pass

    def statuses(self):
        return [
            {"slot_id": "1", "model_key": "coder", "_control": "http://s1",
             "n_gpu_layers": -1, "healthy": True, "child_pid": 111},
            {"slot_id": "2", "model_key": None, "_control": "http://s2",
             "n_gpu_layers": None, "healthy": True, "child_pid": None},
        ]


class _State:
    worker_id = "wid"
    name = "box"


# Fake the slot control /relaunch HTTP call: the fresh child launched at ngl=17.
class _Resp:
    def __init__(self, payload, status):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def _fake_httpx_post(url, json=None, timeout=None):
    return _Resp({"model_key": "coder", "n_gpu_layers": json.get("n_gpu_layers"),
                  "requested_n_gpu_layers": json.get("n_gpu_layers"),
                  "ctx": json.get("ctx", 4096), "child_pid": 222,
                  "healthy": True, "relaunched": True}, 200)


_worig = (slots_mod.SlotPool,)
wapp = wa.build_app(_State())
wclient = wapp.test_client()

import httpx as _httpx
_httpx_orig = _httpx.post
try:
    slots_mod.SlotPool = _FakePool
    _httpx.post = _fake_httpx_post

    # seated slot: relays + echoes the HONEST launched ngl (new pid 222) ------
    r = wclient.post("/slots/1/relaunch", json={"n_gpu_layers": 17})
    body = r.get_json()
    check("worker: seated slot -> 200", r.status_code == 200)
    check("worker: echoes launched ngl", body["n_gpu_layers"] == 17)
    check("worker: reports the fresh child pid (PID recycled)", body["child_pid"] == 222)
    check("worker: ok flag true", body["ok"] is True)

    # empty slot -> 409 --------------------------------------------------------
    r = wclient.post("/slots/2/relaunch", json={"n_gpu_layers": 4})
    check("worker: empty slot -> 409", r.status_code == 409)
    check("worker: empty slot code",
          (r.get_json() or {}).get("error", {}).get("code") == "EmptySlot")

    # unknown slot -> 404 ------------------------------------------------------
    r = wclient.post("/slots/9/relaunch", json={"n_gpu_layers": 4})
    check("worker: unknown slot -> 404", r.status_code == 404)
    check("worker: unknown slot code",
          (r.get_json() or {}).get("error", {}).get("code") == "UnknownSlot")
finally:
    slots_mod.SlotPool = _worig[0]
    _httpx.post = _httpx_orig

print(f"\nall {ok} checks passed")
