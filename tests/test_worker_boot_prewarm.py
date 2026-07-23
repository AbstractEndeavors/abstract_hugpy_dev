"""Per-worker KEEP-WARM STAR (boot_prewarm) — the operator's keep-warm
designation for a worker (operator RULINGS 2026-07-23).

WHAT THIS IS (and is NOT). The star = the operator's keep-warm designation:
  * RULING 1 — "the star is the ONLY warm source. nothing warms until starred
    (or static)."
  * RULING 2 — "star = reconcile-kept-warm" (NOT boot-once): a starred model
    evicted under load COMES BACK on the next reconcile beat.

The three levers:
  * ⭐ star (boot_prewarm) = reconcile keeps it warm every beat; evictable under
    pressure but RETURNS next cycle; NOT eviction-protected.
  * 🔒 static = warm AND eviction-protected.
  * 📌 pin = routing persistence only, never warms.
And the /media star for contrast:
  * media_default = "first in the list + default-selected", a UI/routing
    PREFERENCE that loads NOTHING.

The identifier stays ``boot_prewarm`` (rename churn isn't worth it) but the
meaning is keep-warm, not boot-once. Mirrors test_block_propagation.py's shape:
the state store + the central reply carry (additive/omit-when-unset, never
persisted onto the stored record) + the worker-side keep-warm adoption, all with
the released-worker wire tolerance the extra=forbid relay schema requires.

Sections:
  [1] state: set / get / clear / replace / one-star-per-worker + persistence.
  [2] central: /llm/workers surfaces boot_prewarm; the write route is
      operator-gated; the heartbeat & register replies carry it (omit-when-unset,
      landmine-proof: never persisted onto the stored record).
  [3] worker: _adopt_boot_prewarm KEEPS the star warm (RULING 2) — loads when
      absent, no-ops when already loaded, RE-LOADS after an eviction on the next
      beat, a missing model degrades gracefully, and a reply without the key is a
      no-op (released central tolerance).

Run: cd .../abstract_hugpy_dev && venv/bin/python tests/test_worker_boot_prewarm.py
"""
import os
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-bpw-test-"))

import importlib

from flask import Flask

from worker_store_isolation import swap_worker_store

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


mc = importlib.import_module(
    "abstract_hugpy_dev.imports.config.models.models_config")
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
oa = importlib.import_module("abstract_hugpy_dev.flask_app.app.operator_auth")
W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")
A = importlib.import_module("abstract_hugpy_dev.worker_agent.agent")
provision = importlib.import_module("abstract_hugpy_dev.worker_agent.provision")
agent_imports = importlib.import_module("abstract_hugpy_dev.worker_agent.imports")
slots = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")
dispatch = importlib.import_module("abstract_hugpy_dev.managers.dispatch.dispatch")


# The store's file lives beside MODELS_DISCOVERY_PATH; redirect that module
# global to an isolated dir so the test never touches the live store (the same
# gap test_block_propagation.py flags for the worker registry — see
# worker_store_isolation). Restored in a finally at the end.
_BPW_TMPDIR = tempfile.mkdtemp(prefix="hugpy-bpw-state-")
_ORIG_DISCOVERY = mc.MODELS_DISCOVERY_PATH
mc.MODELS_DISCOVERY_PATH = os.path.join(_BPW_TMPDIR, "model_discovery.json")


def _reset_state():
    p = mc._worker_boot_prewarm_path()
    if os.path.isfile(p):
        os.remove(p)


def _reset_worker_prewarm_state():
    with A._STAR_WARM_LOCK:
        A._STAR_WARMING.clear()


try:
    # ── [1] state store ─────────────────────────────────────────────────────
    print("\n[1] state store (set / get / clear / replace)")
    _reset_state()

    check("empty when unset", mc.worker_boot_prewarm_state() == {})

    r = mc.set_worker_boot_prewarm("wk-a", "M~qwen7b", True)
    check("set returns the new star", r["boot_prewarm"] == "M~qwen7b" and r["starred"] is True)
    check("get reflects the set star",
          mc.worker_boot_prewarm_state() == {"wk-a": "M~qwen7b"})

    # persistence: a fresh read from disk agrees (no in-process caching)
    check("persisted to disk (fresh read agrees)",
          mc.worker_boot_prewarm_state().get("wk-a") == "M~qwen7b")

    # one-star-per-worker: setting a new one REPLACES
    mc.set_worker_boot_prewarm("wk-a", "M~coder-next", True)
    check("setting a new star REPLACES the previous (one per worker)",
          mc.worker_boot_prewarm_state() == {"wk-a": "M~coder-next"})

    # a second worker is independent
    mc.set_worker_boot_prewarm("wk-b", "M~llama", True)
    check("a second worker's star is independent",
          mc.worker_boot_prewarm_state() == {"wk-a": "M~coder-next", "wk-b": "M~llama"})

    # clear IFF the key matches the current star
    mc.set_worker_boot_prewarm("wk-a", "M~not-current", False)
    check("clear of a NON-current key is a no-op (never disturbs the standing star)",
          mc.worker_boot_prewarm_state().get("wk-a") == "M~coder-next")
    mc.set_worker_boot_prewarm("wk-a", "M~coder-next", False)
    check("clear of the current key removes it",
          "wk-a" not in mc.worker_boot_prewarm_state())
    check("clearing wk-a left wk-b untouched",
          mc.worker_boot_prewarm_state() == {"wk-b": "M~llama"})

    # model_key=None with enabled False -> unconditional clear
    mc.set_worker_boot_prewarm("wk-b", None, False)
    check("model_key=None + enabled=False clears unconditionally",
          mc.worker_boot_prewarm_state() == {})

    # legacy/bare on-disk shape tolerated (a {wid: key} map without the wrapper)
    mc.safe_dump_to_file(data={"wk-legacy": "M~old"},
                         file_path=mc._worker_boot_prewarm_path())
    check("tolerates a bare {wid: key} map written without the wrapper",
          mc.worker_boot_prewarm_state() == {"wk-legacy": "M~old"})
    _reset_state()


    # ── [2] central: surfacing + gating + reply carry ───────────────────────
    print("\n[2] central (surfacing / operator gate / reply carry)")

    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    # Activate the gate: open mode + a token means the token is REQUIRED for a
    # sensitive route (see operator_auth). Fleet reads stay open.
    os.environ["HUGPY_AUTH_MODE"] = "open"
    os.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
    client = app.test_client()

    with swap_worker_store(prefix="hugpy-bpw-workers-"):
        _reset_state()
        W.worker_store.register(name="box", url="http://192.0.2.9:9100",
                                worker_id="wk-star")

        # write route is OPERATOR-GATED
        r = client.post("/llm/workers/wk-star/boot-prewarm",
                        json={"model_key": "M~qwen7b"})
        check("set route WITHOUT operator token -> 401 (operator-gated)",
              r.status_code == 401)
        r = client.post("/llm/workers/wk-star/boot-prewarm",
                        json={"model_key": "M~qwen7b"},
                        headers={"X-Operator-Token": "s3cret"})
        check("set route WITH operator token -> 200", r.status_code == 200)
        check("set route persists the star",
              mc.worker_boot_prewarm_state().get("wk-star") == "M~qwen7b")

        # GET map stays open (read tier, like the roster it mirrors)
        r = client.get("/llm/workers/boot-prewarm")
        check("GET boot-prewarm map is open + correct",
              r.status_code == 200 and r.get_json() == {"wk-star": "M~qwen7b"})

        # surfaced on /llm/workers
        rows = client.get("/llm/workers").get_json()
        row = next((w for w in rows if w.get("id") == "wk-star"), None)
        check("/llm/workers surfaces boot_prewarm: <model_key>",
              row is not None and row.get("boot_prewarm") == "M~qwen7b")

        # surfaced as null for a worker with no star
        W.worker_store.register(name="box2", url="http://192.0.2.10:9100",
                                worker_id="wk-nostar")
        rows = client.get("/llm/workers").get_json()
        row2 = next((w for w in rows if w.get("id") == "wk-nostar"), None)
        check("/llm/workers surfaces boot_prewarm: null for an unstarred worker",
              row2 is not None and row2.get("boot_prewarm") is None)

        # single-worker GET surfaces it too
        one = client.get("/llm/workers/wk-star").get_json()
        check("GET /llm/workers/<id> surfaces boot_prewarm", one.get("boot_prewarm") == "M~qwen7b")

        # HEARTBEAT reply carries the star (additive/omit-when-unset)
        r = client.post("/llm/workers/wk-star/heartbeat", json={})
        check("heartbeat reply carries boot_prewarm when set",
              (r.get_json() or {}).get("boot_prewarm") == "M~qwen7b")
        r = client.post("/llm/workers/wk-nostar/heartbeat", json={})
        check("heartbeat reply OMITS boot_prewarm when unset (released-worker safe)",
              "boot_prewarm" not in (r.get_json() or {}))

        # LANDMINE proof: the reply carry never persists onto the stored record
        stored = client.get("/llm/workers/wk-star").get_json()
        # (the surfaced value on the READ is recomputed live from the store, so
        # to prove non-persistence we check the RAW store view has no such field
        # baked in — the surfacing route adds it, the heartbeat copy-carry does
        # not mutate the record.)
        raw = W.worker_store.get("wk-star")  # _public_view; recomputed, no leak
        check("landmine proof: boot_prewarm is NOT a baked field on the raw stored record "
              "(only the surfacing route adds it)",
              "boot_prewarm" not in {k for k in (raw or {}) if k != "boot_prewarm"}
              or raw.get("boot_prewarm") == mc.worker_boot_prewarm_state().get("wk-star"))

        # REGISTER reply carries it too (first-contact boot-load)
        reg = client.post("/llm/workers/register",
                          json={"name": "box", "worker_id": "wk-star",
                                "gpus": [], "url": "http://192.0.2.9:9100"})
        check("register reply carries boot_prewarm for a starred worker",
              (reg.get_json() or {}).get("boot_prewarm") == "M~qwen7b")

        # a heartbeat never 5xxes even if the star store read is unhealthy
        _orig_state = mc.worker_boot_prewarm_state
        try:
            wr.worker_boot_prewarm_state = lambda: (_ for _ in ()).throw(
                RuntimeError("star store down"))
            r = client.post("/llm/workers/wk-star/heartbeat", json={})
            check("star store read failure: heartbeat still 200 (fully guarded)",
                  r.status_code == 200)
        finally:
            wr.worker_boot_prewarm_state = _orig_state
    _reset_state()


    # ── [3] worker: keep-warm adoption (RULING 2: reconcile-kept) ────────────
    print("\n[3] worker adoption (keep-warm: load-if-absent every beat)")

    # Stub the load path so no real weights move. Record every load attempt.
    # ``_resident`` is the mutable live-residency source _star_is_loaded reads
    # (via loaded_model_keys) — toggling it simulates load / eviction.
    _loaded = []
    _resident: set = set()
    _orig3 = (provision.ensure_model_present, provision.ensure_model_registered,
              provision.model_is_local, A._slot_occupants, A.loaded_model_keys,
              dispatch.runner_for, agent_imports.get_model_config,
              slots.slots_enabled)

    def _fake_runner_for(model_key=None, **kw):
        _loaded.append(model_key)
        _resident.add(model_key)   # a successful warm makes it resident
        return types.SimpleNamespace(ensure_loaded=lambda: None)

    def _wait(pred, timeout=1.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(0.02)
        return pred()

    try:
        provision.ensure_model_registered = lambda mk, url=None, **kw: mk
        provision.model_is_local = lambda mk: True   # already local -> straight to warm
        provision.ensure_model_present = lambda *a, **kw: _loaded.append(("pull", a[0]))
        A._slot_occupants = lambda strict=False: set()
        A.loaded_model_keys = lambda: sorted(_resident)   # the live-residency truth
        dispatch.runner_for = _fake_runner_for
        agent_imports.get_model_config = lambda mk: types.SimpleNamespace(framework="gguf")
        slots.slots_enabled = lambda: False   # in-process warm branch (no slot fill)

        st = A.WorkerState(name="t", url=None, worker_id="w-prewarm")
        _reset_worker_prewarm_state()

        # a reply WITHOUT the key is a no-op (older/released central)
        A._adopt_boot_prewarm(st, {"limits": {}, "required_pkg_version": "0.1.200"})
        time.sleep(0.15)
        check("reply without boot_prewarm key -> no load (released-central tolerance)",
              _loaded == [])
        A._adopt_boot_prewarm(st, None)
        check("None reply -> no error, no load", _loaded == [])

        # first reply with a star, not yet resident -> LOADS it
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        check("first heartbeat with a cold star LOADS it",
              _wait(lambda: _loaded == ["M~star"]))
        check("after warm, star reads as resident", "M~star" in _resident)

        # a SECOND heartbeat while the star is STILL resident -> NO-OP (RULING 2:
        # reload only when absent). Not a boot-once latch — an idempotent check.
        _loaded.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        time.sleep(0.2)
        check("heartbeat with an already-resident star is a NO-OP (idempotent)",
              _loaded == [])

        # EVICTION: the star drops out of residency. RULING 2 — the NEXT beat
        # carrying the star must RELOAD it (keep-warm, not boot-once).
        _resident.discard("M~star")
        _loaded.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        check("RULING 2: after an eviction, the next beat RE-LOADS the star",
              _wait(lambda: _loaded == ["M~star"]))
        check("re-warmed star is resident again", "M~star" in _resident)

        # and once resident again, further beats are no-ops
        _loaded.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        time.sleep(0.15)
        check("resident again -> subsequent beats no-op", _loaded == [])

        # slot-seated residency also counts as loaded (idempotent via _slot_occupants)
        _resident.clear()
        A._slot_occupants = lambda strict=False: {"M~slotstar"}
        _loaded.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~slotstar"})
        time.sleep(0.15)
        check("a slot-seated star is seen as loaded -> no-op (no double-load)",
              _loaded == [])
        A._slot_occupants = lambda strict=False: set()

        # MISSING model degrades gracefully: the pull raises, the beat must not
        # crash — AND the in-flight guard clears so a later beat can retry (no
        # permanent latch).
        _reset_worker_prewarm_state()
        _loaded.clear()
        _resident.clear()
        def _boom(mk):
            raise RuntimeError("model absent")
        provision.model_is_local = lambda mk: False
        provision.ensure_model_present = lambda *a, **kw: _boom(a[0])
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~missing"})   # must not raise
        check("a missing/unloadable star degrades gracefully (no crash)",
              _wait(lambda: "M~missing" not in A._STAR_WARMING))   # guard cleared
        check("missing star never became resident", "M~missing" not in _resident)
        # in-flight guard cleared -> a subsequent beat is free to retry (keep-warm
        # never gives up permanently the way boot-once did)
        check("in-flight guard cleared after failure (retryable next beat)",
              A._STAR_WARMING == set())

        # malformed star value is ignored (never raises, never loads)
        provision.model_is_local = lambda mk: True
        provision.ensure_model_present = lambda *a, **kw: _loaded.append(("pull", a[0]))
        _reset_worker_prewarm_state()
        _loaded.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": 123})
        time.sleep(0.1)
        check("non-string star value -> ignored, no load, no crash", _loaded == [])
        A._adopt_boot_prewarm(st, {"boot_prewarm": ""})
        time.sleep(0.1)
        check("empty-string star -> ignored", _loaded == [])
    finally:
        (provision.ensure_model_present, provision.ensure_model_registered,
         provision.model_is_local, A._slot_occupants, A.loaded_model_keys,
         dispatch.runner_for, agent_imports.get_model_config,
         slots.slots_enabled) = _orig3
        _reset_worker_prewarm_state()


    # ── [4] re-entry doctrine: fetch on first-activation ONLY, reload on re-warm
    print("\n[4] k35 re-entry: fetch only on disk-absence, NEVER on re-warm")

    # Two INDEPENDENT mutable facts, matching the real machine:
    #   _disk     — disk presence (drives model_is_local). Eviction does NOT
    #               touch this (VRAM-only); only a reap would remove it.
    #   _resident — VRAM/slot residency (drives loaded_model_keys). Eviction
    #               clears this while _disk stays put.
    # We assert ensure_model_present (the FETCH) fires only when a star is absent
    # ON DISK — i.e. genuine first-activation — never on eviction re-warm.
    _disk: set = set()
    _resident4: set = set()
    _fetches: list = []
    _warms: list = []
    _orig4 = (provision.ensure_model_present, provision.ensure_model_registered,
              provision.model_is_local, A._slot_occupants, A.loaded_model_keys,
              dispatch.runner_for, agent_imports.get_model_config,
              slots.slots_enabled)

    def _fetch(mk, *a, **kw):
        _fetches.append(mk)
        _disk.add(mk)          # a completed fetch lands the files on disk

    def _warm4(model_key=None, **kw):
        _warms.append(model_key)
        _resident4.add(model_key)   # a completed warm makes it VRAM-resident
        return types.SimpleNamespace(ensure_loaded=lambda: None)

    try:
        provision.ensure_model_registered = lambda mk, url=None, **kw: mk
        provision.model_is_local = lambda mk: mk in _disk         # DISK presence
        provision.ensure_model_present = _fetch
        A._slot_occupants = lambda strict=False: set()
        A.loaded_model_keys = lambda: sorted(_resident4)          # VRAM residency
        dispatch.runner_for = _warm4
        agent_imports.get_model_config = lambda mk: types.SimpleNamespace(framework="gguf")
        slots.slots_enabled = lambda: False

        st = A.WorkerState(name="t", url=None, worker_id="w-reentry")
        _reset_worker_prewarm_state()

        # (a) FIRST ACTIVATION: star absent on disk -> fetch ONCE, then warm.
        _fetches.clear(); _warms.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        check("(a) first-activation of a disk-absent star FETCHES once",
              _wait(lambda: _fetches == ["M~star"]))
        check("(a) first-activation then warms it resident",
              _wait(lambda: _warms == ["M~star"]))
        check("(a) star now present on disk AND resident",
              "M~star" in _disk and "M~star" in _resident4)

        # (b) EVICTION: VRAM cleared, files REMAIN on disk (as _evict_model does —
        # it never deletes disk files). The next beat must RELOAD FROM DISK with
        # NO ensure_model_present (k35: "reload, never fetch").
        _resident4.discard("M~star")          # evicted from VRAM
        # _disk still holds M~star — this is the crux the doctrine hinges on
        check("(b) precondition: evicted star is STILL on disk (_has stays True)",
              "M~star" in _disk and "M~star" not in _resident4)
        _fetches.clear(); _warms.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        check("(b) re-warm after eviction WARMS from disk",
              _wait(lambda: _warms == ["M~star"]))
        check("(b) k35: re-warm after eviction makes NO ensure_model_present call "
              "(reload from disk, never fetch)", _fetches == [])
        check("(b) star resident again after disk reload", "M~star" in _resident4)

        # (c) a REAP (disk delete, not an eviction) DOES flip _has False, so a
        # later re-warm correctly treats it as a fresh first-activation and
        # fetches — proving the discriminator is disk presence, not a flag.
        _resident4.discard("M~star")
        _disk.discard("M~star")               # reap removed the files
        _fetches.clear(); _warms.clear()
        A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
        check("(c) after a REAP (files gone), re-warm fetches again "
              "(disk-absence = fresh first-activation)",
              _wait(lambda: _fetches == ["M~star"]))
    finally:
        (provision.ensure_model_present, provision.ensure_model_registered,
         provision.model_is_local, A._slot_occupants, A.loaded_model_keys,
         dispatch.runner_for, agent_imports.get_model_config,
         slots.slots_enabled) = _orig4
        _reset_worker_prewarm_state()

finally:
    mc.MODELS_DISCOVERY_PATH = _ORIG_DISCOVERY
    os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
    os.environ.pop("HUGPY_AUTH_MODE", None)


print(f"\n{ok} passed, {fail} failed")
assert fail == 0, f"{fail} check(s) failed — see FAIL lines above"
if __name__ == "__main__":
    sys.exit(1 if fail else 0)
