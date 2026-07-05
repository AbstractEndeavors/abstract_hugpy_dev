"""_relay_worker_op retry-on-connect (apply-blip fix): config-style ops get
ONE retry after a 3s pause when the worker agent's socket is dead (it re-execs
~0.5s after ACKing a config change), and an honest 503 "agent is restarting"
when the retry also can't connect. Non-config ops keep the single-shot 502.

Runs like the other tests here: venv/bin/python tests/test_relay_retry.py
"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from flask import Flask

wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
cr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.comms_routes")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


class _FakeResp:
    status_code = 200
    def json(self):
        return {"ok": True, "restarting": True}


app = Flask("relay-retry-test")

_orig = (wr.get_worker, cr.audit, httpx.post, time.sleep)
_posts, _sleeps = [], []
_outcomes = []          # per-call: "ok" | exception instance to raise

def _fake_post(url, json=None, timeout=None):
    _posts.append(url)
    out = _outcomes.pop(0)
    if out == "ok":
        return _FakeResp()
    raise out

def _run(retry):
    with app.app_context():
        resp, status = wr._relay_worker_op(
            "w1", "/ops/config", {"pinned": {"m": True}},
            timeout=15.0, action="config", retry_on_connect=retry)
        return resp.get_json(), status

try:
    wr.get_worker = lambda wid: {"name": "t", "url": "http://worker:9999"}
    cr.audit = lambda *a, **k: None
    httpx.post = _fake_post
    time.sleep = lambda s: _sleeps.append(s)   # _relay imports time as _time

    # 1. blip then recovery: connect error -> 3s pause -> retry succeeds
    _outcomes[:] = [httpx.ConnectError("refused"), "ok"]
    body, status = _run(retry=True)
    check("retry after connect error succeeds", status == 200 and body["ok"] is True)
    check("exactly one 3s pause", _sleeps == [3.0])
    check("two POST attempts", len(_posts) == 2)

    # 2. blip persists: both attempts fail -> honest 503, not a bare 502
    _posts.clear(); _sleeps.clear()
    _outcomes[:] = [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow")]
    body, status = _run(retry=True)
    check("persistent blip -> 503", status == 503)
    check("503 message says agent is restarting",
          "restarting" in body["error"]["message"])
    check("503 code is AgentRestarting", body["error"]["code"] == "AgentRestarting")

    # 3. non-config ops: single shot, generic 502, no sleeping (historical)
    _posts.clear(); _sleeps.clear()
    _outcomes[:] = [httpx.ConnectError("refused")]
    body, status = _run(retry=False)
    check("no retry without the flag -> 502", status == 502)
    check("generic error shape preserved", body["error"]["code"] == "ConnectError")
    check("no pause, single POST", _sleeps == [] and len(_posts) == 1)

    # 4. non-connect failures never retry, even for config ops
    _posts.clear(); _sleeps.clear()
    _outcomes[:] = [ValueError("bad json")]
    body, status = _run(retry=True)
    check("non-connect error -> immediate 502", status == 502)
    check("non-connect error not retried", _sleeps == [] and len(_posts) == 1)
finally:
    wr.get_worker, cr.audit, httpx.post, time.sleep = _orig

print(f"\nall {ok} checks passed")
