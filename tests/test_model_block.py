"""Model-level BLOCK from the serving pool (operator pool primitive).

An operator can BLOCK a model_key (and UNBLOCK it). A blocked model is removed
from the serving pool GLOBALLY: never a routing candidate, never (re)assigned,
never warmed/provisioned, never a fallback default — while its files and existing
designation rows are left untouched (inert). Block outranks pin (pin is routing
persistence; block is an operator override) and a direct call fails fast with a
distinct honest refusal. Persisted in the F4 settings store (survives restart).

This regresses every enforcement point WITHOUT a live worker/fleet:
  * the blocklist registry (block/is_blocked/blocked_keys/reason/info/unblock);
  * remote.py honest-refusal vocabulary (permanent marker + _blocked_reason);
  * routing candidate selection (workers_for_model / pick / candidates → []);
  * the no-worker diagnostic (explain_no_worker → distinct blocked reason);
  * reconcile warm-set ∩ not-blocked + _kick_warm provisioning filter;
  * the /assign route → 409; the /block + /unblock routes; operator-gating;
  * the placement/feasibility preview → "blocked" reason, not fake-infeasible;
  * /models + /v1/models listings mark blocked:true;
  * the brain/default fallback ladder (_reconcile_default) skips a blocked one.

Runs like the other tests here: venv/bin/python tests/test_model_block.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-model-block-test-")

import importlib
from flask import Flask

# ⚠ ISOLATION LANDMINE (k2 incident, 2026-07-18): the PROJECTS_HOME env var set
# above does NOT isolate the worker registry — WorkerStore's default path
# resolves through schemas.settings.manifest_path, a module-level singleton
# read from a .env FILE at import time (abstract_essentials.get_env_value),
# never from os.environ. Section 6 below drives real Flask routes, including
# an /assign call that (once the model is unblocked) reaches the real,
# UNSTUBBED module-level assign_model() — which writes through the module
# singleton W.worker_store straight to the LIVE /mnt/llm_storage/projects/
# registry (proven: it opens/locks/rewrites that file even for an unknown
# worker id). See tests/worker_store_isolation.py for the full writeup.
from worker_store_isolation import swap_worker_store

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

bl = importlib.import_module("abstract_hugpy_dev.comms.blocklist")


def _reset():
    for k in list(bl.blocked_keys()):
        bl.unblock(k)


# ── 1) the blocklist registry ────────────────────────────────────────────────
_reset()
check("registry: not blocked initially", bl.is_blocked("M~1") is False)
check("registry: block_reason None when unblocked", bl.block_reason("M~1") is None)
rec = bl.block("M~1", note="dueling coder-next")
check("registry: is_blocked after block", bl.is_blocked("M~1") is True)
check("registry: record shape {blocked,by,ts,note} (extensible)",
      rec["blocked"] is True and rec["by"] == "operator"
      and isinstance(rec["ts"], float) and rec["note"] == "dueling coder-next")
check("registry: blocked_keys lists it", bl.blocked_keys() == {"M~1"})
check("registry: block_info carries the record", (bl.block_info("M~1") or {}).get("note") == "dueling coder-next")
check("registry: block_reason is distinct + carries the marker",
      bl.BLOCKED_MARKER in (bl.block_reason("M~1") or "")
      and "by the operator" in bl.block_reason("M~1"))
check("registry: block is idempotent (still one key)", (bl.block("M~1") or bl.is_blocked("M~1")) and bl.blocked_keys() == {"M~1"})
check("registry: None/empty key never blocked", bl.is_blocked("") is False and bl.is_blocked(None) is False)
check("registry: unblock returns was_blocked True", bl.unblock("M~1") is True)
check("registry: is_blocked False after unblock", bl.is_blocked("M~1") is False)
check("registry: block_info None after unblock", bl.block_info("M~1") is None)
check("registry: unblock again is a no-op (was_blocked False)", bl.unblock("M~1") is False)

# persistence survives a fresh SettingsStore instance (== a central restart) ---
bl.block("Persist~Me")
from abstract_hugpy_dev.comms.settings import SettingsStore
fresh = SettingsStore()
check("registry: block persists across a fresh store (restart)",
      bool(fresh.get(bl.NS, "Persist~Me", {}).get("blocked")))
bl.unblock("Persist~Me")

# ── 2) remote.py honest-refusal vocabulary ───────────────────────────────────
remote = importlib.import_module("abstract_hugpy_dev.managers.resolvers.remote")
check("remote: marker present in _PERMANENT_LOAD_MARKERS",
      any("blocked from the serving pool" in m for m in remote._PERMANENT_LOAD_MARKERS))
bl.block("M~2")
check("remote: block_reason classified as PERMANENT (fail fast, no retry/hold)",
      remote._is_permanent_load_error(bl.block_reason("M~2")) is True)
check("remote: _blocked_reason returns the reason when blocked",
      bool(remote._blocked_reason("M~2")))
check("remote: _blocked_reason None when not blocked", remote._blocked_reason("M~unblocked") is None)
bl.unblock("M~2")

# ── 3) routing candidate selection + 4) no-worker diagnostic ─────────────────
W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")

# A worker that WOULD serve "M" — gate helpers stubbed to pass so the ONLY
# variable under test is the operator block.
SERVER = {"id": "w1", "name": "box", "status": "online", "admission": "approved",
          "models": ["M"], "loaded_models": [], "grants": {}, "pool": ""}
_wsave = (W.worker_store.all, W._engine_unusable, W._worker_env_tier,
          W.env_tier_for_model, W._task_capable)
try:
    W.worker_store.all = lambda: [dict(SERVER)]
    W._engine_unusable = lambda w: False
    W._worker_env_tier = lambda w: "stable"
    W.env_tier_for_model = lambda mk: "stable"
    W._task_capable = lambda w, t: True

    _reset()
    check("routing: worker IS a candidate when unblocked",
          [w["id"] for w in W.worker_store.workers_for_model("M")] == ["w1"])
    check("routing: pick_for_model returns the worker when unblocked",
          (W.worker_store.pick_for_model("M") or {}).get("id") == "w1")

    bl.block("M")
    check("routing: BLOCK removes ALL candidates (workers_for_model → [])",
          W.worker_store.workers_for_model("M") == [])
    check("routing: pick_for_model → None when blocked (→ honest refusal)",
          W.worker_store.pick_for_model("M") is None)
    check("routing: candidates_for_model → [] when blocked (relay reroute)",
          W.worker_store.candidates_for_model("M") == [])
    check("diagnostic: explain_no_worker returns the DISTINCT blocked reason",
          bl.BLOCKED_MARKER in W.explain_no_worker("M"))

    bl.unblock("M")
    check("routing: unblock REVERTS — worker is a candidate again",
          [w["id"] for w in W.worker_store.workers_for_model("M")] == ["w1"])
    check("diagnostic: explain_no_worker empty again after unblock (no manufactured reason)",
          W.explain_no_worker("M") == "")
finally:
    (W.worker_store.all, W._engine_unusable, W._worker_env_tier,
     W.env_tier_for_model, W._task_capable) = _wsave

# ── 5) reconcile warm-set ∩ not-blocked + _kick_warm provisioning filter ─────
wr = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.worker_routes")

WK = {"id": "wk", "name": "box", "url": "http://192.0.2.1:9100",
      "models_local": ["A", "B"],
      "config": {"warm_whitelist": ["A", "B"], "residency": {}}}
_reset()
check("reconcile: warm set holds whitelisted on-disk models when unblocked",
      wr._reconcile_warm_set(WK) == ["A", "B"])
bl.block("A")
check("reconcile: BLOCK removes A from the curated warm set (∩ not-blocked)",
      wr._reconcile_warm_set(WK) == ["B"])
check("provision: _kick_warm filters blocked keys — all-blocked schedules NOTHING",
      wr._kick_warm({"id": "wk", "url": "http://192.0.2.1:9100"}, ["A"], "test") == [])
bl.block("B")
check("provision: _kick_warm with a fully-blocked set → [] (no probe/pull thread)",
      wr._kick_warm({"id": "wk", "url": "http://192.0.2.1:9100"}, ["A", "B"], "test") == [])
_reset()
check("reconcile: unblock REVERTS the warm set", wr._reconcile_warm_set(WK) == ["A", "B"])

# ── 6) the routes: assign 409, block/unblock, placement, gating ──────────────
cr = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.comms_routes")
app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

_orig = (wr.get_models_dict, wr._transfer_authorized, cr.audit,
         wr._central_missing_reason, wr._disk_preflight_reason, wr.get_worker)
try:
    wr.get_models_dict = lambda dict_return=False: {"M": {}}
    wr._transfer_authorized = lambda: True
    cr.audit = lambda *a, **k: None
    wr._central_missing_reason = lambda mk: None
    wr._disk_preflight_reason = lambda w, mk: None
    wr.get_worker = lambda wid: {"id": wid, "name": "box", "models_local": []}

    # The /assign route below calls the REAL, unstubbed module-level
    # assign_model() once "M" is unblocked (nothing else in this section
    # stubs it) — swap the module singleton for an isolated store so that
    # write lands on a tmpdir, never the live registry.
    with swap_worker_store():
        _reset()
        # /block sets state; unknown key 404s; /unblock reverts
        r = client.post("/llm/models/M/block", json={"note": "op says"})
        check("route /block: 200 + blocked:true", r.status_code == 200 and r.get_json()["blocked"] is True)
        check("route /block: model is actually blocked now", bl.is_blocked("M") is True)
        check("route /block: unknown key → 404",
              client.post("/llm/models/Nope~Key/block", json={}).status_code == 404)

        # assign of a blocked model → 409 with a clear blocked reason
        ra = client.post("/llm/workers/w1/assign", json={"model_key": "M"})
        check("route /assign: blocked model → 409",
              ra.status_code == 409 and "blocked from the serving pool" in (ra.get_json() or {}).get("error", ""))

        # placement/feasibility preview reports blocked (not fake-infeasible)
        rp = client.get("/llm/models/M/placement")
        pj = rp.get_json()
        check("route /placement: blocked:true + blocked winner_reason (not fake-infeasible)",
              rp.status_code == 200 and pj.get("blocked") is True
              and "blocked from the serving pool" in (pj.get("winner_reason") or "")
              and pj.get("workers") == [])

        # /unblock reverts everything
        ru = client.post("/llm/models/M/unblock", json={})
        check("route /unblock: 200 + was_blocked:true", ru.status_code == 200 and ru.get_json()["was_blocked"] is True)
        check("route /unblock: model no longer blocked", bl.is_blocked("M") is False)
        # This is the confirmed live-write path (k3): once unblocked, the route
        # falls through to the REAL assign_model() — now safely inside the
        # swapped, tmpdir-backed W.worker_store.
        check("route /assign: unblocked model no longer 409 on the block gate",
              client.post("/llm/workers/w1/assign", json={"model_key": "M"}).status_code != 409)
finally:
    (wr.get_models_dict, wr._transfer_authorized, cr.audit,
     wr._central_missing_reason, wr._disk_preflight_reason, wr.get_worker) = _orig

# operator-gating: both verbs must be in _SENSITIVE (assign tier)
oa = importlib.import_module("abstract_hugpy_dev.flask_app.app.operator_auth")
def _gated(path, method):
    p = path
    if p == "/api" or p.startswith("/api/"):
        p = p[len("/api"):] or "/"
    return any(method in methods and rx.match(p) for methods, rx in oa._SENSITIVE)
check("gating: POST /llm/models/<k>/block is operator-gated", _gated("/llm/models/Some~Key/block", "POST"))
check("gating: POST /llm/models/<k>/unblock is operator-gated", _gated("/llm/models/Some~Key/unblock", "POST"))
check("gating: gate also matches the /api-mounted path", _gated("/api/llm/models/o/r/g~k/block", "POST"))
check("gating: GET /llm/models/<k>/placement stays OPEN (read)", _gated("/llm/models/k/placement", "GET") is False)
check("gating: slashed model_key still gated (path converter)", _gated("/llm/models/owner/repo~q/block", "POST"))

# ── 7) listings mark blocked:true (/models + /v1/models) ─────────────────────
lsr = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.llm_storage_routes")
lapp = Flask(__name__); lapp.register_blueprint(lsr.llm_bp); lclient = lapp.test_client()
_lsave = (lsr.get_models_dict, lsr.update_model_status, lsr.media_default_state,
          lsr.media_state, lsr._annotate_gguf_size, lsr._annotate_size)
try:
    # A SHARED manifest dict (mutated in place across calls) — mirrors the real
    # cached get_models_dict, so the stale-`block`-key-after-unblock regression
    # is actually reachable here.
    _MANIFEST = {"M": {"model_key": "M", "status": "installed"}}
    lsr.get_models_dict = lambda dict_return=False: _MANIFEST
    lsr.update_model_status = lambda m: m
    lsr.media_default_state = lambda: None
    lsr.media_state = lambda mk: False
    lsr._annotate_gguf_size = lambda m, mk: None
    lsr._annotate_size = lambda m, mk: None
    _reset()
    row = (lclient.get("/models").get_json() or [{}])[0]
    check("/models: blocked:false when not blocked", row.get("blocked") is False)
    bl.block("M")
    row = (lclient.get("/models").get_json() or [{}])[0]
    check("/models: blocked:true + block record when blocked",
          row.get("blocked") is True and (row.get("block") or {}).get("blocked") is True)
    bl.unblock("M")
    row = (lclient.get("/models").get_json() or [{}])[0]
    check("/models: unblock clears blocked AND the stale block record (cached dict)",
          row.get("blocked") is False and "block" not in row)
finally:
    (lsr.get_models_dict, lsr.update_model_status, lsr.media_default_state,
     lsr.media_state, lsr._annotate_gguf_size, lsr._annotate_size) = _lsave

v1 = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.v1_routes")
vapp = Flask(__name__); vapp.register_blueprint(v1.v1_bp); vclient = vapp.test_client()
_vsave = (v1.get_models_dict, v1.update_model_status, v1.media_default_state, v1.api_key_required)
try:
    v1.get_models_dict = lambda dict_return=False: {"M": {"status": "installed", "primary_task": "text-generation"}}
    v1.update_model_status = lambda m: m
    v1.media_default_state = lambda: None
    v1.api_key_required = lambda: False
    bl.block("M")
    d = (v1client_json := vclient.get("/v1/models").get_json())["data"][0]
    check("/v1/models: additive blocked:true", d.get("blocked") is True and d["id"] == "M")
    bl.unblock("M")
    d = vclient.get("/v1/models").get_json()["data"][0]
    check("/v1/models: blocked:false after unblock", d.get("blocked") is False)
finally:
    (v1.get_models_dict, v1.update_model_status, v1.media_default_state, v1.api_key_required) = _vsave

# ── 8) the brain / default fallback ladder skips a blocked candidate ─────────
md = importlib.import_module("abstract_hugpy_dev.imports.config.models.models_default")
_mdsave = md._on_disk
try:
    md._on_disk = lambda cfg: True     # both stand-ins count as installed
    reg = {"BRAIN": object(), "ALT": object()}
    _reset()
    check("brain: configured brain kept when NOT blocked",
          md._reconcile_default("BRAIN", reg) == "BRAIN")
    bl.block("BRAIN")
    check("brain: BLOCKED brain is SKIPPED — stands in an installed non-blocked model",
          md._reconcile_default("BRAIN", reg) == "ALT")
    bl.block("ALT")
    check("brain: both blocked → stand-in set empty → configured returned unchanged (accuracy preserved)",
          md._reconcile_default("BRAIN", reg) == "BRAIN")
    _reset()
    check("brain: unblock REVERTS to the configured brain",
          md._reconcile_default("BRAIN", reg) == "BRAIN")
finally:
    md._on_disk = _mdsave

_reset()
print(f"\nall {ok} checks passed")
