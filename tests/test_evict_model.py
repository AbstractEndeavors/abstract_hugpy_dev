"""Targeted eviction — `evict <model_key>` (worker /ops/evict).

Central signals `evict <model_key>` (+ optional force), NEVER a raw PID (PIDs are
per-box and recycled). The worker resolves the model_key to its LIVE hosting
handle at eviction time, verifies identity, and frees it with the mechanism that
matches HOW it's hosted:

  * comfy      — framework == 'comfy'  -> comfy's OWN POST /free (no PID kill)
  * slot       — a live slot serves it -> verify identity, then slot /unload
                                          (owner does SIGTERM -> wait -> SIGKILL)
  * in-process — weights in our PID     -> drop refs + CUDA empty_cache + trim
  * not resident / foreign proc         -> idempotent no-op, never a kill

Covered here:
  (1) model_key -> handle resolution picks the right host_mode per mode;
  (2) recycled-PID guard: if the slot handle changed before we act (swapped model
      or respawned child under a new pid), we do NOT evict — no slot /unload fires;
  (3) the static / in-flight gate is honored UNLESS force=true (📌pin does NOT
      gate the evict verb — pin is designation, not a VRAM lock; 2026-07-15);
  (4) the comfy path calls comfy's /free with the documented body (mocked httpx);
  (5) a foreign/non-owned model_key resolves to "not resident" and is REFUSED,
      never killed (no in-process drop, no slot unload);
  (6) an unknown/missing model_key is an idempotent no-op at HTTP 200 (never 500);
  (7) central relay POST /llm/workers/<id>/evict forwards to /ops/evict verbatim.

Runs like the other tests here: venv/bin/python tests/test_evict_model.py
"""
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# See test_residency_contention.py: managers/__init__ star-imports shadow the
# subpackage attrs, so import_module to bind the REAL modules the agent uses.
from abstract_hugpy_dev.worker_agent import agent
slots = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def _fixed_mem():
    """Patch the before/after VRAM+RAM readers to a deterministic cycle so
    vram_freed/ram_freed are computable without a GPU: each _evict_model reads
    before (1000/2000) then after (5000/9000) -> freed 4000/7000, every call.
    Returns a restore()."""
    import itertools
    _fv, _fr = agent._free_vram_bytes, agent._free_ram_bytes
    vcyc, rcyc = itertools.cycle([1000, 5000]), itertools.cycle([2000, 9000])
    agent._free_vram_bytes = lambda: next(vcyc)
    agent._free_ram_bytes = lambda: next(rcyc)
    def restore():
        agent._free_vram_bytes, agent._free_ram_bytes = _fv, _fr
    return restore


def new_client():
    state = agent.WorkerState(name="t", url=None, worker_id="w-e")
    return agent.build_app(state).test_client(), state


# ---------------------------------------------------------------------------
# Save the module-level seams we monkeypatch; restore at the very end.
# ---------------------------------------------------------------------------
_SAVE = {
    "framework": agent._model_framework,
    "resolve_slot": agent._resolve_slot_handle,
    "inproc_resident": agent._is_inprocess_resident,
    "drop_inproc": agent._drop_inprocess_model,
    "comfy_free": agent._comfy_free_models,
    "trim": agent._trim_host_ram,
    "SlotPool": slots.SlotPool,
    "in_flight": agent.gen_gate.in_flight,
    "settings": dict(agent._RUNTIME_SETTINGS),
    "loaded": agent.loaded_model_keys,
}

# Neutralize side-effecting globals for the whole run.
agent._trim_host_ram = lambda: None
agent.loaded_model_keys = lambda: []
agent._RUNTIME_SETTINGS.clear()
agent.gen_gate.in_flight = lambda mk: 0
# Default: nothing is comfy / slot / in-process unless a test says so.
agent._model_framework = lambda mk: None
agent._resolve_slot_handle = lambda mk: None
agent._is_inprocess_resident = lambda mk: False


class FakeSlotPool:
    """Records every .unload(control_url) so a test can assert whether the slot
    child was actually torn down (the recycled-PID guard must PREVENT it)."""
    calls = []

    def __init__(self, urls=None):
        pass

    def unload(self, control_url, **kw):
        FakeSlotPool.calls.append(control_url)
        return {"ok": True}

slots.SlotPool = FakeSlotPool


try:
    _restore_mem = _fixed_mem()

    # --- (1a) comfy host-mode: framework==comfy -> comfy branch -----------------
    agent._model_framework = lambda mk: "comfy" if mk == "cmfy" else None
    _comfy_called = []
    def _fake_comfy_free(state):
        _comfy_called.append(True)
        return True, "comfy /free accepted (unload_models + free_memory)"
    agent._comfy_free_models = _fake_comfy_free
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "cmfy"})
    b = r.get_json()
    check("(1a) comfy model -> host_mode 'comfy', evicted true",
          r.status_code == 200 and b["host_mode"] == "comfy" and b["evicted"] is True)
    check("(1a) comfy path went through comfy's /free API (no PID kill)",
          _comfy_called == [True])
    check("(1a) vram_freed reported from before/after delta",
          b["vram_freed"] == 4000 and b["ram_freed"] == 7000)
    agent._model_framework = lambda mk: None      # reset

    # --- (1b) slot host-mode: live slot serves it -> slot /unload ---------------
    FakeSlotPool.calls.clear()
    handle = {"control_url": "http://127.0.0.1:8101", "child_pid": 4242,
              "endpoint": "http://127.0.0.1:8101"}
    agent._resolve_slot_handle = lambda mk: dict(handle) if mk == "slotmodel" else None
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "slotmodel"})
    b = r.get_json()
    check("(1b) slot model -> host_mode 'slot', evicted true, child_pid reported",
          b["host_mode"] == "slot" and b["evicted"] is True and b["child_pid"] == 4242)
    check("(1b) slot /unload fired on the resolved control url",
          FakeSlotPool.calls == ["http://127.0.0.1:8101"])

    # --- (2) recycled-PID guard: handle changes before we act -> NO unload ------
    FakeSlotPool.calls.clear()
    # first call (resolve) returns pid 4242; recheck returns a DIFFERENT pid.
    _seq = iter([
        {"control_url": "http://127.0.0.1:8101", "child_pid": 4242},   # resolve
        {"control_url": "http://127.0.0.1:8101", "child_pid": 9999},   # recheck (swapped)
    ])
    agent._resolve_slot_handle = lambda mk: next(_seq)
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "slotmodel"})
    b = r.get_json()
    check("(2) recycled/swapped slot handle -> evicted false, reason flags it",
          b["host_mode"] == "slot" and b["evicted"] is False
          and "recycled" in b["reason"].lower())
    check("(2) recycled-PID guard PREVENTED the slot /unload (no kill fired)",
          FakeSlotPool.calls == [])
    agent._resolve_slot_handle = lambda mk: None  # reset

    # --- (3) in-process host-mode + gate honored unless force -------------------
    FakeSlotPool.calls.clear()
    _dropped = []
    agent._is_inprocess_resident = lambda mk: mk == "ip"
    agent._drop_inprocess_model = lambda mk: (_dropped.append(mk) or True)

    # 3a: plain on-demand in-process -> evicted
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "ip"})
    b = r.get_json()
    check("(3a) in-process on-demand -> host_mode 'in_process', evicted true",
          b["host_mode"] == "in_process" and b["evicted"] is True)
    check("(3a) in-process drop actually ran (no PID kill)", _dropped == ["ip"])

    # 3b: static residency -> GATED without force
    _dropped.clear()
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update({"residency": {"ip": "static"}})
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "ip"})
    b = r.get_json()
    check("(3b) static model without force -> gated, evicted false",
          b["evicted"] is False and "static" in b["reason"].lower())
    check("(3b) gated eviction did NOT drop the model", _dropped == [])

    # 3c: force overrides the static gate
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "ip", "force": True})
    b = r.get_json()
    check("(3c) static model WITH force -> evicted true (gate overridden)",
          b["evicted"] is True and b["forced"] is True and _dropped == ["ip"])

    # 3d: in-flight generation gate (no force) -> protected
    _dropped.clear()
    agent._RUNTIME_SETTINGS.clear()
    agent.gen_gate.in_flight = lambda mk: 1 if mk == "ip" else 0
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "ip"})
    b = r.get_json()
    check("(3d) mid-generation model without force -> gated (never ripped)",
          b["evicted"] is False and "in-flight" in b["reason"].lower()
          and _dropped == [])
    agent.gen_gate.in_flight = lambda mk: 0
    agent._is_inprocess_resident = lambda mk: False   # reset

    # --- (5) foreign / non-owned model_key -> not resident, REFUSED not killed --
    FakeSlotPool.calls.clear()
    _dropped.clear()
    client, _ = new_client()
    r = client.post("/ops/evict", json={"model_key": "who-owns-this"})
    b = r.get_json()
    check("(5) foreign/non-owned model -> host_mode 'none', evicted false",
          r.status_code == 200 and b["host_mode"] == "none" and b["evicted"] is False)
    check("(5) not-resident reason surfaced", "not resident" in b["reason"].lower())
    check("(5) nothing was killed/dropped for a non-owned model",
          FakeSlotPool.calls == [] and _dropped == [])

    # --- (6) unknown/missing model_key -> idempotent 200 no-op, never 500 -------
    client, _ = new_client()
    r = client.post("/ops/evict", json={})
    b = r.get_json()
    check("(6) missing model_key -> 200 no-op (not a 500), evicted false",
          r.status_code == 200 and b["evicted"] is False)
    r = client.post("/ops/evict", json={"model_key": "   "})
    check("(6) blank model_key -> 200 no-op", r.status_code == 200
          and r.get_json()["evicted"] is False)

    _restore_mem()

    # --- (4) _comfy_free_models unit: calls comfy /free with the documented body -
    # restore the REAL fn (test 1a swapped in a recorder) before unit-testing it.
    agent._comfy_free_models = _SAVE["comfy_free"]
    # import httpx inside the fn -> swap sys.modules['httpx'] for a recorder.
    import os as _os
    captured = {}
    fake_httpx = types.ModuleType("httpx")
    class _Resp:
        status_code = 200
    def _post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()
    fake_httpx.post = _post
    _real_httpx = sys.modules.get("httpx")
    sys.modules["httpx"] = fake_httpx
    _env_was = _os.environ.get("COMFY_URL")
    _os.environ["COMFY_URL"] = "http://comfy.local:8188"
    try:
        state = agent.WorkerState(name="t", url=None, worker_id="w-e")
        freed_ok, note = agent._comfy_free_models(state)
        check("(4) comfy /free -> reports freed ok on HTTP 200", freed_ok is True)
        check("(4) comfy /free hit COMFY_URL + /free",
              captured["url"] == "http://comfy.local:8188/free")
        check("(4) comfy /free body = {unload_models:true, free_memory:true}",
              captured["json"] == {"unload_models": True, "free_memory": True})
    finally:
        if _real_httpx is not None:
            sys.modules["httpx"] = _real_httpx
        else:
            sys.modules.pop("httpx", None)
        if _env_was is None:
            _os.environ.pop("COMFY_URL", None)
        else:
            _os.environ["COMFY_URL"] = _env_was

    # --- (7) central relay: POST /llm/workers/<id>/evict -> /ops/evict verbatim -
    wr = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
    relayed = {}
    def _fake_relay(worker_id, op_path, body, timeout, action, **kw):
        relayed.update(worker_id=worker_id, op_path=op_path, body=body,
                       action=action)
        return ({"ok": True, "relayed": True}, 200)
    _relay_was = wr._relay_worker_op
    wr._relay_worker_op = _fake_relay
    try:
        # Drive the view function directly (its body only touches request +
        # _relay_worker_op); a minimal request context supplies the JSON body.
        app = wr.worker_bp
        from flask import Flask
        _a = Flask(__name__)
        _a.register_blueprint(wr.worker_bp)
        c = _a.test_client()
        # operator gate isn't mounted on this bare app, so the route runs raw.
        r = c.post("/llm/workers/ae/evict",
                   json={"model_key": "m", "force": True})
        check("(7) relay route forwards to worker /ops/evict",
              relayed.get("op_path") == "/ops/evict" and relayed.get("action") == "evict")
        check("(7) relay passes model_key + force through unchanged",
              relayed.get("body") == {"model_key": "m", "force": True}
              and relayed.get("worker_id") == "ae")
    finally:
        wr._relay_worker_op = _relay_was

finally:
    agent._model_framework = _SAVE["framework"]
    agent._resolve_slot_handle = _SAVE["resolve_slot"]
    agent._is_inprocess_resident = _SAVE["inproc_resident"]
    agent._drop_inprocess_model = _SAVE["drop_inproc"]
    agent._comfy_free_models = _SAVE["comfy_free"]
    agent._trim_host_ram = _SAVE["trim"]
    slots.SlotPool = _SAVE["SlotPool"]
    agent.gen_gate.in_flight = _SAVE["in_flight"]
    agent.loaded_model_keys = _SAVE["loaded"]
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update(_SAVE["settings"])

print(f"\nall {ok} checks passed")
