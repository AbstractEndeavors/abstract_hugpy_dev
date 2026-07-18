"""k2: worker-side slot-fill/prewarm reconciler honors the operator model BLOCK.

Journal-proven defect (ae, 2026-07-18): once ``unsloth~Qwen3-Coder-Next-GGUF``
was operator-blocked (models.blocked, aa4aea3), ae's worker-side slot-fill
reconciler (``_fill_empty_slots``, ridden every 60s off ``_residency_sweep_loop``)
kept retrying warm-loads of the blocked model for ~4.75 hours, each attempt
failing a full-GPU fit. Block already covers every CENTRAL path (routing,
assign-409, warm sweeps, provisioning kick, the brain ladder) — a block does
NOT auto-unassign, so a still-assigned blocked model just sat in
``state.assigned_models`` forever, invisible to the worker's own background
loops.

Two-halves fix, covered here:
  1. Propagation — the block set rides the heartbeat REPLY (a plain, additive,
     omit-when-empty list), the EXACT wire idiom t28 calibration already
     established (fed4ae8): published on a COPY of the reply, never persisted
     onto the stored worker record. ``workers_heartbeat`` in worker_routes.py.
  2. Worker-side skip — every worker-local warm/load-ahead loop that reads
     ``state.assigned_models`` learns to skip a blocked model_key, logging the
     skip ONCE (not every ~60s tick):
       * ``_fill_empty_slots`` (the slot-fill reconciler — the incident itself)
       * ``_kick_provision`` (the single choke for both the eager-pull-on-
         assign path AND the UTIL-08 reconcile-loop re-kick)

Explicitly OUT of scope: refusing an EXPLICIT relay request for a blocked
model. That stays central's honest-refusal job (comms.blocklist / remote.py) —
this is about the worker's OWN background loops only.

Sections:
  [1] central: workers_heartbeat reply carries blocked_models (+ landmine
      proof: never persisted onto the stored record; omitted when empty).
  [2] worker: _adopt_blocked_models (present/absent/empty/malformed) +
      released-worker tolerance (a reply without the key).
  [3] worker: _fill_empty_slots skips a blocked candidate, logs once, resumes
      on unblock, re-arms on re-block.
  [4] worker: _kick_provision (the single choke) skips a blocked model, logs
      once, an unblocked model still provisions normally.
  [5] worker: _reconcile_loop still calls _kick_provision for a blocked static
      model (the guard lives INSIDE _kick_provision, not at the call site) —
      locks in the single-choke design.

Run: cd .../abstract_hugpy_dev && venv/bin/python tests/test_block_propagation.py
"""
import os
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-block-prop-test-"))

import importlib

from flask import Flask

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


bl = importlib.import_module("abstract_hugpy_dev.comms.blocklist")
wr = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.worker_routes")
W = importlib.import_module("abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")
A = importlib.import_module("abstract_hugpy_dev.worker_agent.agent")
# managers/__init__ star-imports shadow serve/dispatch attributes (see
# test_residency_static.py) — go through import_module for the real modules.
slots = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")
dispatch = importlib.import_module("abstract_hugpy_dev.managers.dispatch.dispatch")
provision = importlib.import_module("abstract_hugpy_dev.worker_agent.provision")
agent_imports = importlib.import_module("abstract_hugpy_dev.worker_agent.imports")


def _reset_blocklist():
    for k in list(bl.blocked_keys()):
        bl.unblock(k)


def _reset_worker_block_state():
    with A._BLOCKED_LOCK:
        A._BLOCKED_MODELS.clear()
        A._BLOCKED_LOGGED.clear()


# ── [1] central: workers_heartbeat reply carries blocked_models ─────────────
print("\n[1] central propagation (workers_heartbeat reply)")

# ⚠ ISOLATION LANDMINE (discovered while writing this test — not a k2 defect,
# flagged in the delivery report): WorkerStore's default path comes from
# abstract_hugpy_dev.imports.src.constants.constants.PROJECTS_HOME, which is
# resolved via abstract_essentials.get_env_value() — a ``.env``-FILE reader
# (cwd / home / ~/.envy_all), NOT os.environ. So the
# ``os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(...)`` idiom used elsewhere
# in this suite (test_model_block.py et al.) does NOT isolate the worker
# registry — it silently falls through to the REAL /mnt/llm_storage/projects/
# workers.json. (comms.settings.SettingsStore — what blocklist.py rides — DOES
# read PROJECTS_HOME from real os.environ, so blocklist tests ARE isolated;
# only the worker registry has this gap.) Registering a real worker here for
# real would leak a ghost row into the LIVE fleet the operator's console
# shows — so we inject an explicit tempfile-backed WorkerStore instead of
# touching the module singleton.
_wk_tmp_store = W.WorkerStore(path=os.path.join(
    tempfile.mkdtemp(prefix="hugpy-block-prop-workers-"), "workers.json"))
_worker_store_orig = W.worker_store
W.worker_store = _wk_tmp_store

app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

try:
    _reset_blocklist()
    W.worker_store.register(name="box", url="http://192.0.2.9:9100", worker_id="wk-prop")

    r = client.post("/llm/workers/wk-prop/heartbeat", json={})
    check("unblocked: heartbeat 200", r.status_code == 200)
    check("unblocked: reply omits blocked_models (nothing to publish, additive/omit-when-empty)",
          "blocked_models" not in (r.get_json() or {}))

    bl.block("M~blocked-1")
    r = client.post("/llm/workers/wk-prop/heartbeat", json={})
    check("blocked (1 model): reply carries blocked_models",
          (r.get_json() or {}).get("blocked_models") == ["M~blocked-1"])

    bl.block("A~blocked-2")
    r = client.post("/llm/workers/wk-prop/heartbeat", json={})
    check("blocked (2 models): reply is the sorted full set",
          (r.get_json() or {}).get("blocked_models") == ["A~blocked-2", "M~blocked-1"])

    # landmine proof: published on a COPY, never persisted onto the stored record
    # (the exact same idiom as calibration/reservations — worker_routes.py comment)
    stored = client.get("/llm/workers/wk-prop").get_json()
    check("landmine proof: blocked_models NEVER persists onto the stored worker record",
          "blocked_models" not in (stored or {}))

    _reset_blocklist()
    r = client.post("/llm/workers/wk-prop/heartbeat", json={})
    check("unblock reverts: reply omits blocked_models again",
          "blocked_models" not in (r.get_json() or {}))

    # a heartbeat never 5xxes even if the blocklist read is unhealthy
    _bk_orig = wr._blocked_keys
    try:
        wr._blocked_keys = lambda: (_ for _ in ()).throw(RuntimeError("settings store down"))
        r = client.post("/llm/workers/wk-prop/heartbeat", json={})
        check("blocklist read failure: heartbeat still 200 (fully guarded)",
              r.status_code == 200)
    finally:
        wr._blocked_keys = _bk_orig
finally:
    W.worker_store = _worker_store_orig


# ── [2] worker: _adopt_blocked_models ────────────────────────────────────────
print("\n[2] worker adoption")

_reset_worker_block_state()
A._adopt_blocked_models({"blocked_models": ["m1", "m2"]})
check("adopt: present list -> set adopted", A._BLOCKED_MODELS == {"m1", "m2"})
check("adopt: _is_blocked_locally True for a blocked key",
      A._is_blocked_locally("m1") is True)
check("adopt: _is_blocked_locally False for an unblocked key",
      A._is_blocked_locally("m3") is False)
check("adopt: _is_blocked_locally False for None/falsy input",
      A._is_blocked_locally(None) is False and A._is_blocked_locally("") is False)

A._adopt_blocked_models({"blocked_models": []})
check("adopt: explicit empty list clears", A._BLOCKED_MODELS == set())

A._adopt_blocked_models({"blocked_models": ["m1"]})
A._adopt_blocked_models({})
check("adopt: reply with no blocked_models key clears stale state",
      A._BLOCKED_MODELS == set())

# released-worker tolerance: a plain dict reply missing the key entirely
# (older/pre-k2 central) must not raise and must not leave a stale block.
A._adopt_blocked_models({"blocked_models": ["m1"]})
A._adopt_blocked_models({"limits": {}, "required_pkg_version": "0.1.191"})
check("released-worker tolerance: reply without the key -> no error, clears",
      A._BLOCKED_MODELS == set())
check("released-worker tolerance: None worker -> no error",
      A._adopt_blocked_models(None) is None and A._BLOCKED_MODELS == set())

# malformed shapes degrade gracefully — a heartbeat adoption must never raise
A._adopt_blocked_models({"blocked_models": ["m1"]})
A._adopt_blocked_models({"blocked_models": "not-a-list"})
check("adopt: malformed (string, not list) -> ignored, empty set",
      A._BLOCKED_MODELS == set())
A._adopt_blocked_models({"blocked_models": ["m1", None, "", 123]})
check("adopt: falsy entries (None/'') dropped, truthy entries stringified",
      A._BLOCKED_MODELS == {"m1", "123"})

_reset_worker_block_state()


# ── [3] worker: _fill_empty_slots skips a blocked candidate (log-once) ──────
print("\n[3] slot-fill reconciler skip")


class _FillPoolBlock:
    def __init__(self, urls=None):
        pass

    def statuses(self):
        return [{"_control": "u1", "model_key": None},
                {"_control": "u2", "model_key": None}]


_seated = []
_logged3 = []
_orig3 = (slots.slots_enabled, slots.SlotPool, A._models_local,
          agent_imports.get_model_config, dispatch.last_used_snapshot,
          dispatch.runner_for, A.logger.info)


def _skip_count(logged, mk):
    """Count only the SKIP log line for ``mk`` — not the (also mk-mentioning)
    'seating m-blocked...' line a later resumed pass emits."""
    return sum(1 for a in logged if "skipping" in str(a) and mk in str(a))


try:
    slots.slots_enabled = lambda: True
    slots.SlotPool = _FillPoolBlock
    A._models_local = lambda st: ["m-ok", "m-blocked"]
    agent_imports.get_model_config = lambda mk: types.SimpleNamespace(framework="gguf")
    dispatch.last_used_snapshot = lambda: {}
    dispatch.runner_for = lambda model_key=None, **kw: _seated.append(model_key)
    A.logger.info = lambda *a, **kw: _logged3.append(a)

    _reset_worker_block_state()
    A._adopt_blocked_models({"blocked_models": ["m-blocked"]})

    st = A.WorkerState(name="t", url=None, worker_id="w-fill-block")
    st.assigned_models = ["m-ok", "m-blocked"]

    A._fill_empty_slots(st)
    check("blocked candidate is never seated (THE incident, fixed)",
          "m-blocked" not in _seated)
    check("unblocked candidate still seats normally", "m-ok" in _seated)
    check("skip logged exactly once on first pass",
          _skip_count(_logged3, "m-blocked") == 1)
    check("skip log names the loop", any("slot fill" in str(a) for a in _logged3))

    # this is the direct fix for the 60s-forever spam: repeated passes must
    # not add more log lines.
    _seated.clear()
    for _ in range(3):
        A._fill_empty_slots(st)
    check("3 more passes (simulating ~3 more residency-sweep ticks): "
          "still logged only once, no spam",
          _skip_count(_logged3, "m-blocked") == 1)
    check("still never seated across repeated passes", "m-blocked" not in _seated)

    # unblock -> resumes
    A._adopt_blocked_models({})
    check("unblock: no longer blocked locally", A._is_blocked_locally("m-blocked") is False)
    _seated.clear()
    A._fill_empty_slots(st)
    check("unblocked model resumes normal seating", "m-blocked" in _seated)

    # re-block -> re-arms (a fresh block episode logs again, not suppressed forever)
    A._adopt_blocked_models({"blocked_models": ["m-blocked"]})
    _seated.clear()
    A._fill_empty_slots(st)
    check("re-block after unblock: skip is logged again (re-armed)",
          _skip_count(_logged3, "m-blocked") == 2)
finally:
    (slots.slots_enabled, slots.SlotPool, A._models_local,
     agent_imports.get_model_config, dispatch.last_used_snapshot,
     dispatch.runner_for, A.logger.info) = _orig3


# ── [4] worker: _kick_provision (the single choke) skips a blocked model ────
print("\n[4] provisioning-kick skip")

_pulled = []
_logged4 = []
_orig4 = (provision.ensure_model_present, provision.model_is_local,
          provision.ensure_model_registered, A.restart_requested, A.logger.info)


def _wait_done(st, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with st._provision_lock:
            if not st._provisioning:
                return True
        time.sleep(0.02)
    return False


try:
    provision.ensure_model_present = (
        lambda mk, url, progress=None, **kw: _pulled.append(mk))
    provision.model_is_local = lambda mk: False
    provision.ensure_model_registered = lambda mk, url: mk
    A.restart_requested = lambda: False
    A.logger.info = lambda *a, **kw: _logged4.append(a)

    _reset_worker_block_state()
    A._adopt_blocked_models({"blocked_models": ["m-blocked-static"]})

    st = A.WorkerState(name="t", url=None, worker_id="w-kick-block")
    A._kick_provision(st, "m-blocked-static")
    check("blocked model: _kick_provision is a no-op (never enters _provisioning)",
          "m-blocked-static" not in st._provisioning)
    time.sleep(0.1)
    check("blocked model: ensure_model_present is NEVER called (no doomed pull)",
          _pulled == [])
    check("blocked model: skip logged once",
          sum(1 for a in _logged4 if "m-blocked-static" in str(a)) == 1)

    # an unblocked model still provisions normally through the SAME function
    A._kick_provision(st, "m-ok-static")
    check("unblocked model: provision thread runs to completion", _wait_done(st))
    check("unblocked model: ensure_model_present WAS called", _pulled == ["m-ok-static"])

    # a second kick for the same still-blocked model adds no second log line
    A._kick_provision(st, "m-blocked-static")
    check("second kick, same blocked model: still only ONE log line (no spam)",
          sum(1 for a in _logged4 if "m-blocked-static" in str(a)) == 1)
finally:
    (provision.ensure_model_present, provision.model_is_local,
     provision.ensure_model_registered, A.restart_requested, A.logger.info) = _orig4


# ── [5] worker: _reconcile_loop routes a blocked static model through the ───
#      SAME choke (_kick_provision) — locks in the single-choke design ───────
print("\n[5] reconcile loop -> _kick_provision (single-choke design)")

_kicked_via_reconcile = []
_orig5 = (A._kick_provision, A.time.sleep, A.restart_requested, A._models_local)
try:
    A._kick_provision = (
        lambda state, mk, purpose="reconcile": _kicked_via_reconcile.append(mk))
    _sleep_calls = {"n": 0}

    def _sleep(_secs):
        _sleep_calls["n"] += 1

    A.time.sleep = _sleep
    flags = iter([False, True])   # one body pass, then stop the loop
    A.restart_requested = lambda: next(flags, True)
    A._models_local = lambda state: []   # nothing on disk -> both look missing

    A._RUNTIME_SETTINGS.clear()
    A._RUNTIME_SETTINGS.update({"residency": {
        "m-blocked-static": "static", "m-ok-static": "static"}})

    _reset_worker_block_state()
    A._adopt_blocked_models({"blocked_models": ["m-blocked-static"]})

    st = A.WorkerState(name="t", url=None, worker_id="w-reconcile-block")
    st.assigned_models = ["m-blocked-static", "m-ok-static"]
    A._reconcile_loop(st)
    check("reconcile loop still calls _kick_provision for a blocked static model "
          "(the guard lives INSIDE _kick_provision, not at this call site)",
          "m-blocked-static" in _kicked_via_reconcile)
    check("reconcile loop calls it for the unblocked static model too",
          "m-ok-static" in _kicked_via_reconcile)
finally:
    (A._kick_provision, A.time.sleep, A.restart_requested, A._models_local) = _orig5
    A._RUNTIME_SETTINGS.clear()

_reset_blocklist()
_reset_worker_block_state()

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
