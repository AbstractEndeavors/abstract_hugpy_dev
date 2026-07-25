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
  (8) HONEST accounting (2026-07-25 fix — the ae 44GB/35.6MB lie): the new
      ``freed`` breakdown is measured PER-MODEL from ground truth (nvidia-smi
      joined on child_pid + /proc rss split for slot; torch tensor bytes /
      GGUF file size for in-process), never a box-wide MemAvailable delta.
      comfy reports freed=None with a reason (no per-model attribution is
      possible); not-resident reports zeros (nothing to free); the legacy
      vram_freed/ram_freed keys stay present for wire back-compat.
  (9) central relay POST /llm/workers/<id>/reap-orphans (k32 gap) forwards to
      /ops/reap-orphans verbatim, and is operator-gated the same as evict.
  (10) the stranded-slot fix: POST /slots/<slot_id>/unload unconditionally
      tears down a slot's child regardless of its model_key claim (including
      None/stale), and its central relay + operator gate exist.

Runs like the other tests here: venv/bin/python tests/test_evict_model.py
"""
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker_store_isolation import swap_worker_store

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

    # --- (8) HONEST per-model accounting (fixes the ae 44GB/35.6MB lie) ---------
    print("\n[8] honest _model_footprint_before_evict per host_mode")
    _fp_save = {
        "gpu": agent._gpu_process_vram,
        "inproc": agent._inprocess_gpu_bytes,
        "detail": agent._loaded_detail,
    }
    try:
        # 8a: slot — nvidia-smi joined on child_pid + /proc rss split.
        agent._gpu_process_vram = lambda: {4242: {"name": "llama-server", "mib": 42000}}
        _sa = importlib.import_module("abstract_hugpy_dev.managers.serve.slot_agent")
        _proc_rss_save = _sa._proc_rss_detail
        _sa._proc_rss_detail = lambda pid: (
            {"rss_anon_bytes": 1_500_000_000, "rss_file_bytes": 43_600_000_000}
            if pid == 4242 else {})
        try:
            fp = agent._model_footprint_before_evict(
                "slotmodel", "slot", {"child_pid": 4242, "control_url": "x"})
        finally:
            _sa._proc_rss_detail = _proc_rss_save
        check("(8a) slot vram_bytes = nvidia-smi mib joined on child_pid",
              fp["vram_bytes"] == 42000 * agent._MIB)
        check("(8a) slot ram_anon_bytes = the HONEST pinned figure (rss_anon)",
              fp["ram_anon_bytes"] == 1_500_000_000)
        check("(8a) slot ram_file_bytes carried separately, never folded into anon",
              fp["ram_file_bytes"] == 43_600_000_000)
        check("(8a) measured_from names the real source", "nvidia-smi" in fp["measured_from"])

        # 8b: slot — pid absent from nvidia-smi / /proc unreadable -> nulls, not 0s.
        agent._gpu_process_vram = lambda: {}
        _sa._proc_rss_detail = lambda pid: {}
        fp = agent._model_footprint_before_evict(
            "slotmodel", "slot", {"child_pid": 9999, "control_url": "x"})
        check("(8b) unmeasurable slot vram -> None, never a fabricated number",
              fp["vram_bytes"] is None)
        check("(8b) unmeasurable slot ram -> None", fp["ram_anon_bytes"] is None
              and fp["ram_file_bytes"] is None)

        # 8c: in_process GGUF — file-backed bytes labeled, NOT claimed as anon RAM.
        agent._loaded_detail = lambda: {"gguf-model": {"model_bytes": 44_000_000_000}}
        agent._inprocess_gpu_bytes = lambda: {}
        fp = agent._model_footprint_before_evict("gguf-model", "in_process")
        check("(8c) in-process GGUF -> ram_file_bytes = the on-disk/mmap'd size",
              fp["ram_file_bytes"] == 44_000_000_000)
        check("(8c) in-process GGUF -> vram_bytes stays None (no GPU claim made)",
              fp["vram_bytes"] is None)
        check("(8c) measured_from documents it's file-backed, not pinned anon RAM",
              "NOT pinned anon RAM" in fp["measured_from"])

        # 8d: in_process torch — REAL per-model tensor sum, not a delta.
        agent._loaded_detail = lambda: {}
        agent._inprocess_gpu_bytes = lambda: {
            "torch-model": {"vram_bytes": 8_000_000_000, "device": "cuda"}}
        fp = agent._model_footprint_before_evict("torch-model", "in_process")
        check("(8d) in-process torch -> vram_bytes = real per-model tensor sum",
              fp["vram_bytes"] == 8_000_000_000)
        check("(8d) measured_from cites torch tensor introspection, not a delta",
              "not a delta" in fp["measured_from"])

        # 8e: comfy — no per-model attribution is possible; None + reason, never a guess.
        fp = agent._model_footprint_before_evict("comfy-model", "comfy")
        check("(8e) comfy -> every byte field stays None (no fabricated box-wide guess)",
              fp["vram_bytes"] is None and fp["ram_anon_bytes"] is None
              and fp["ram_file_bytes"] is None)
        check("(8e) comfy reason explains WHY (no per-model_key attribution possible)",
              "no per-model_key attribution" in fp["measured_from"])

        # 8f: not-resident -> zeros (genuinely nothing to free), not None.
        fp = agent._model_footprint_before_evict("ghost", "none")
        check("(8f) not-resident -> zeros, not null (there IS a definite answer: none)",
              fp == {"vram_bytes": 0, "ram_anon_bytes": 0, "ram_file_bytes": 0,
                     "measured_from": "not resident — nothing to free"})
    finally:
        agent._gpu_process_vram = _fp_save["gpu"]
        agent._inprocess_gpu_bytes = _fp_save["inproc"]
        agent._loaded_detail = _fp_save["detail"]

    # --- (8g) end-to-end: /ops/evict's slot branch carries "freed" + legacy keys -
    print("\n[8g] /ops/evict response carries BOTH the new 'freed' block and the "
          "legacy vram_freed/ram_freed (wire back-compat)")
    _restore_mem2 = _fixed_mem()
    try:
        agent._gpu_process_vram = lambda: {4242: {"name": "llama-server", "mib": 42000}}
        _sa2 = importlib.import_module("abstract_hugpy_dev.managers.serve.slot_agent")
        _sa2._proc_rss_detail = lambda pid: {"rss_anon_bytes": 1_500_000_000,
                                             "rss_file_bytes": 43_600_000_000}
        FakeSlotPool.calls.clear()
        handle = {"control_url": "http://127.0.0.1:8101", "child_pid": 4242,
                  "endpoint": "http://127.0.0.1:8101"}
        agent._resolve_slot_handle = lambda mk: dict(handle) if mk == "slotmodel" else None
        client, _ = new_client()
        r = client.post("/ops/evict", json={"model_key": "slotmodel"})
        b = r.get_json()
        check("(8g) legacy vram_freed/ram_freed keys still present (wire back-compat)",
              b["vram_freed"] == 4000 and b["ram_freed"] == 7000)
        check("(8g) new 'freed' block present with the honest per-model figures",
              b["freed"]["vram_bytes"] == 42000 * agent._MIB
              and b["freed"]["ram_anon_bytes"] == 1_500_000_000)
        agent._resolve_slot_handle = lambda mk: None
    finally:
        _restore_mem2()
        agent._gpu_process_vram = _fp_save["gpu"]

    # --- (9) central relay: POST /llm/workers/<id>/reap-orphans (k32 gap) -------
    print("\n[9] central relay for reap-orphans (k32) + operator gate")
    relayed.clear()
    wr._relay_worker_op = _fake_relay
    try:
        _a = Flask(__name__)
        _a.register_blueprint(wr.worker_bp)
        c = _a.test_client()
        r = c.post("/llm/workers/ae/reap-orphans", json={"dry_run": False})
        check("(9a) relay route forwards to worker /ops/reap-orphans",
              relayed.get("op_path") == "/ops/reap-orphans"
              and relayed.get("action") == "reap-orphans")
        check("(9a) relay passes dry_run through unchanged",
              relayed.get("body") == {"dry_run": False}
              and relayed.get("worker_id") == "ae")
    finally:
        wr._relay_worker_op = _relay_was

    oa = importlib.import_module("abstract_hugpy_dev.flask_app.app.operator_auth")
    import os as _os2
    _auth_mode_was = _os2.environ.get("HUGPY_AUTH_MODE")
    _token_was = _os2.environ.get("HUGPY_OPERATOR_TOKEN")
    _os2.environ["HUGPY_AUTH_MODE"] = "open"
    _os2.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
    try:
        _gate_app = Flask(__name__)
        _gate_app.register_blueprint(wr.worker_bp)
        oa.install_operator_gate(_gate_app)
        gc = _gate_app.test_client()
        r = gc.post("/llm/workers/ae/reap-orphans", json={})
        check("(9b) reap-orphans WITHOUT operator token -> 401 (operator-gated)",
              r.status_code == 401)
    finally:
        if _auth_mode_was is None:
            _os2.environ.pop("HUGPY_AUTH_MODE", None)
        else:
            _os2.environ["HUGPY_AUTH_MODE"] = _auth_mode_was
        if _token_was is None:
            _os2.environ.pop("HUGPY_OPERATOR_TOKEN", None)
        else:
            _os2.environ["HUGPY_OPERATOR_TOKEN"] = _token_was

    # --- (10) stranded-slot fix: unconditional /slots/<id>/unload ---------------
    print("\n[10] stranded-slot fix: POST /slots/<slot_id>/unload (model_key-independent)")
    FakeSlotPool.calls.clear()
    _statuses_save = None
    try:
        class _StatusSlotPool(FakeSlotPool):
            def statuses(self):
                return [{"slot_id": "1", "model_key": None, "child_pid": 7777,
                         "_control": "http://127.0.0.1:8101"}]
        slots.SlotPool = _StatusSlotPool
        client, _ = new_client()
        r = client.post("/slots/1/unload", json={})
        b = r.get_json()
        check("(10a) unload fires even though model_key is None (the stranding case)",
              r.status_code == 200 and b["ok"] is True
              and b["model_key_before"] is None and b["child_pid_before"] == 7777)
        check("(10a) the slot's control url actually got .unload()'d",
              FakeSlotPool.calls == ["http://127.0.0.1:8101"])

        # unknown slot id -> 404, never a silent no-op
        client, _ = new_client()
        r = client.post("/slots/99/unload", json={})
        check("(10b) unknown slot id -> 404", r.status_code == 404)
    finally:
        slots.SlotPool = FakeSlotPool

    # central relay + operator gate for the new route
    relayed.clear()
    wr._relay_worker_op = _fake_relay
    try:
        with swap_worker_store(prefix="hugpy-evict-workers-"):
            Wm = importlib.import_module(
                "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")
            Wm.worker_store.register(name="ae", url="http://192.0.2.9:9100",
                                     worker_id="ae")
            _a = Flask(__name__)
            _a.register_blueprint(wr.worker_bp)
            c = _a.test_client()
            r = c.post("/llm/workers/ae/slots/1/unload", json={})
            check("(10c) relay route forwards to worker /slots/<id>/unload",
                  relayed.get("op_path") == "/slots/1/unload"
                  and relayed.get("action") == "slot-unload")
    finally:
        wr._relay_worker_op = _relay_was

    _os2.environ["HUGPY_AUTH_MODE"] = "open"
    _os2.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
    try:
        _gate_app2 = Flask(__name__)
        _gate_app2.register_blueprint(wr.worker_bp)
        oa.install_operator_gate(_gate_app2)
        gc2 = _gate_app2.test_client()
        r = gc2.post("/llm/workers/ae/slots/1/unload", json={})
        check("(10d) slot unload WITHOUT operator token -> 401 (operator-gated)",
              r.status_code == 401)
    finally:
        if _auth_mode_was is None:
            _os2.environ.pop("HUGPY_AUTH_MODE", None)
        else:
            _os2.environ["HUGPY_AUTH_MODE"] = _auth_mode_was
        if _token_was is None:
            _os2.environ.pop("HUGPY_OPERATOR_TOKEN", None)
        else:
            _os2.environ["HUGPY_OPERATOR_TOKEN"] = _token_was

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
