"""WILDCARD PLACEMENT + DESIGNATION SCOPE + KEY-MATCH UNIFICATION (Slice A,
operator doctrine 2026-07-23).

THE DOCTRINE. Worker designations are a HARD routing scope: they seal where
designated models CAN route. An UNDESIGNATED model "gets in where it fits in" —
but ONLY on workers that opted in as WILDCARD ("a worker can be designated to
take all comers while adhering to its allocated model list as priority, or it
can not be selected as a wildcard and adhere only to its own allocated
models"). A designated model whose home workers are ALL unavailable overflows
onto wildcard workers; it busts only when neither can serve. Once resident,
NORMAL eviction rules apply — the flag is routing only ("you don't want random
evictions simply because you have a verbose model registry" — hence an explicit
per-worker opt-in, DEFAULT FALSE: no flags set == today's routing exactly).

Binding keeper decisions regressed here:
  1. RESIDENT = DE FACTO DESIGNATION — a box holding the model (loaded_models /
     grants) is always eligible for it, wildcard or not.
  2. Default wildcard = FALSE for every worker (defaults are promises).
  3. "~"-tail unification in _match_keys ships WITH this slice, guarded by the
     blocked-sibling check (an alias match must never launder a BLOCKED model).

Mirrors test_worker_boot_prewarm.py's harness (state store + Flask routes +
operator gate) and test_central_task_gating.py's isolated-WorkerStore routing
checks.

Sections:
  [1] state: set / get / clear / absent-is-False + legacy bare shape.
  [2] central routes: GET map open, POST operator-gated, ``wildcard`` field
      surfaced on /llm/workers and /llm/workers/<id> (response-copy only).
  [3] _match_keys "~"-tail unification (qualified <-> bare, both directions).
  [4] eligibility: sealed scope preserved by default; wildcard catch; hard
      gates still apply; resident-de-facto regression guard; no store leak.
  [5] ranking: home above wildcard catch; overflow-by-ordering when home
      workers are all excluded.
  [6] blocked-sibling guard at the match site.
  [7] ⭐ star ranking priority (ambiguity tie-break, post-incident 2026-07-23):
      warm > star; home > star; alias-tolerant. Rank key order
      (home/wildcard, warm, star, gpu, last_picked, id).

Run: cd .../abstract_hugpy_dev && venv/bin/python tests/test_worker_wildcard.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-wc-test-"))

import importlib

from flask import Flask

from worker_store_isolation import swap_worker_store, isolated_worker_store

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
bl = importlib.import_module("abstract_hugpy_dev.comms.blocklist")


# Redirect the flag store's directory (it lives beside MODELS_DISCOVERY_PATH)
# to an isolated dir so the test never touches the live store — same isolation
# the boot_prewarm test uses. Restored in the finally at the end.
_WC_TMPDIR = tempfile.mkdtemp(prefix="hugpy-wc-state-")
_ORIG_DISCOVERY = mc.MODELS_DISCOVERY_PATH
mc.MODELS_DISCOVERY_PATH = os.path.join(_WC_TMPDIR, "model_discovery.json")


def _reset_state():
    p = mc._worker_wildcard_path()
    if os.path.isfile(p):
        os.remove(p)


# Targeted cleanup ONLY (never a blanket unblock of blocked_keys(): if the
# settings store ever resolves to a live deployment's file, a sweep would
# silently undo REAL operator blocks — the k2 isolation incident class).
_TEST_BLOCK_KEY = "B~X"


try:
    # ── [1] state store ─────────────────────────────────────────────────────
    print("\n[1] state store (set / get / clear / absent-is-False)")
    _reset_state()

    check("empty when unset", mc.worker_wildcard_state() == {})
    check("absent worker id reads False (default-false promise)",
          mc.worker_wildcard_state().get("wk-nobody", False) is False)

    r = mc.set_worker_wildcard("wk-a", True)
    check("set returns the new state", r == {"worker_id": "wk-a", "wildcard": True})
    check("get reflects the opt-in", mc.worker_wildcard_state() == {"wk-a": True})
    check("persisted to disk (fresh read agrees)",
          mc.worker_wildcard_state().get("wk-a") is True)

    mc.set_worker_wildcard("wk-b", True)
    check("a second worker's flag is independent",
          mc.worker_wildcard_state() == {"wk-a": True, "wk-b": True})

    r = mc.set_worker_wildcard("wk-a", False)
    check("clear returns wildcard False", r == {"worker_id": "wk-a", "wildcard": False})
    check("cleared worker drops OUT of the map (absent, not stored-False)",
          mc.worker_wildcard_state() == {"wk-b": True})
    mc.set_worker_wildcard("wk-never", False)
    check("clearing a never-set worker is a no-op (idempotent)",
          mc.worker_wildcard_state() == {"wk-b": True})

    # legacy/bare on-disk shape tolerated (a {wid: bool} map without the wrapper)
    mc.safe_dump_to_file(data={"wk-legacy": True, "wk-off": False},
                         file_path=mc._worker_wildcard_path())
    check("tolerates a bare {wid: bool} map; falsy entries read absent",
          mc.worker_wildcard_state() == {"wk-legacy": True})
    _reset_state()

    # ── [2] central routes: gating + surfacing ──────────────────────────────
    print("\n[2] central routes (GET open / POST gated / surfacing)")

    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    # Activate the gate: open mode + a token means the token is REQUIRED for a
    # sensitive route (see operator_auth). Fleet reads stay open.
    os.environ["HUGPY_AUTH_MODE"] = "open"
    os.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
    client = app.test_client()

    with swap_worker_store(prefix="hugpy-wc-workers-"):
        _reset_state()
        W.worker_store.register(name="box", url="http://192.0.2.9:9100",
                                worker_id="wk-wild")
        W.worker_store.register(name="box2", url="http://192.0.2.10:9100",
                                worker_id="wk-plain")

        r = client.post("/llm/workers/wk-wild/wildcard", json={"enabled": True})
        check("set route WITHOUT operator token -> 401 (operator-gated)",
              r.status_code == 401)
        r = client.post("/llm/workers/wk-wild/wildcard", json={"enabled": True},
                        headers={"X-Operator-Token": "s3cret"})
        check("set route WITH operator token -> 200 and persists",
              r.status_code == 200
              and mc.worker_wildcard_state() == {"wk-wild": True})

        r = client.get("/llm/workers/wildcard")
        check("GET wildcard map is open + correct",
              r.status_code == 200 and r.get_json() == {"wk-wild": True})

        rows = client.get("/llm/workers").get_json()
        by_id = {w.get("id"): w for w in rows}
        check("/llm/workers surfaces wildcard: true for the opted-in worker",
              by_id.get("wk-wild", {}).get("wildcard") is True)
        check("/llm/workers surfaces wildcard: false for an un-flagged worker",
              by_id.get("wk-plain", {}).get("wildcard") is False)

        one = client.get("/llm/workers/wk-wild").get_json()
        check("GET /llm/workers/<id> surfaces wildcard", one.get("wildcard") is True)

        # response-copy only: the flag is never baked onto the stored record
        raw = W.worker_store._load().get("wk-wild") or {}
        check("landmine proof: 'wildcard' is NOT persisted on the raw stored record",
              "wildcard" not in raw)

        # clear via the route
        r = client.post("/llm/workers/wk-wild/wildcard", json={"enabled": False},
                        headers={"X-Operator-Token": "s3cret"})
        check("clear via route -> 200 + map empties",
              r.status_code == 200 and mc.worker_wildcard_state() == {})
    _reset_state()

    # ── [3] _match_keys "~"-tail unification ────────────────────────────────
    print("\n[3] _match_keys (~-tail unification)")

    check('Qwen~X form set == {"Qwen~X","qwen~x","X","x"}',
          W._match_keys("Qwen~X") == {"Qwen~X", "qwen~x", "X", "x"})
    check('bare X form set == {"X","x"}', W._match_keys("X") == {"X", "x"})
    check("qualified request matches bare advertisement (Qwen~X -> X)",
          bool(W._match_keys("Qwen~X") & W._match_keys("X")))
    check("bare request matches qualified advertisement (X -> Qwen~X)",
          bool(W._match_keys("X") & W._match_keys("Qwen~X")))
    check("unsloth~X <-> X intersect",
          bool(W._match_keys("unsloth~X") & W._match_keys("X")))
    check("same base, different owners intersect (Qwen~X <-> unsloth~X)",
          bool(W._match_keys("Qwen~X") & W._match_keys("unsloth~X")))
    check("different bases do NOT intersect (Qwen~X vs unsloth~Y)",
          not (W._match_keys("Qwen~X") & W._match_keys("unsloth~Y")))
    check('"/"-tail form still works (owner/name <-> name)',
          bool(W._match_keys("Qwen/Qwen2.5-Coder") & W._match_keys("qwen2.5-coder")))

    # ── [4] eligibility: sealed by default; wildcard catch; hard gates ──────
    print("\n[4] eligibility (sealed default / wildcard catch / hard gates)")

    ASSIGNED = "org~Assigned-Model"
    STRAY = "org~Stray-Model"      # designated NOWHERE
    EMBED = "feature-extraction"

    _reset_state()
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-route-")

    def _add(wid, models, **kw):
        store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid,
                       models=models, **kw)
        store.set_admission(wid, "approved")

    _add("sealed", [ASSIGNED])
    check("today's behavior: designated model routes to its designated worker",
          {w["id"] for w in store.workers_for_model(ASSIGNED)} == {"sealed"})
    check("today's behavior: undesignated model + non-wildcard worker -> NO candidate",
          store.workers_for_model(STRAY) == [])
    check("pick_for_model: undesignated model -> None (local-fallback signal preserved)",
          store.pick_for_model(STRAY) is None)

    # opt the worker in -> it catches the stray
    mc.set_worker_wildcard("sealed", True)
    cands = store.workers_for_model(STRAY)
    check("wildcard worker CATCHES an undesignated model",
          [w["id"] for w in cands] == ["sealed"])
    check("the catch is marked _wildcard_catch (say-why/ranking marker)",
          cands and cands[0].get("_wildcard_catch") is True)
    home = store.workers_for_model(ASSIGNED)
    check("the same worker's DESIGNATED model is NOT marked as a catch",
          home and not home[0].get("_wildcard_catch"))
    raw = store._load().get("sealed") or {}
    check("marker never leaks into the persisted store record",
          "_wildcard_catch" not in raw)

    # hard gates still apply to a wildcard catch: task incapability drops it
    _add("wild-incap", [], task_capabilities={EMBED: False})
    mc.set_worker_wildcard("wild-incap", True)
    check("wildcard catch is still dropped by task-incapability (hard gates hold)",
          "wild-incap" not in
          {w["id"] for w in store.workers_for_model(STRAY, task=EMBED)})
    check("without the task, the same wildcard worker IS a candidate",
          "wild-incap" in {w["id"] for w in store.workers_for_model(STRAY)})

    # RESIDENT = DE FACTO DESIGNATION (keeper decision 1): a NON-wildcard box
    # holding the model in loaded_models (not in its models list) stays eligible.
    _reset_state()
    _add("holder", [])
    store.heartbeat("holder", loaded_models=["org~Resident-Model"])
    res = store.workers_for_model("org~Resident-Model")
    check("resident-de-facto: loaded_models-only + non-wildcard -> STILL eligible",
          [w["id"] for w in res] == ["holder"])
    check("a resident match is HOME (not a wildcard catch)",
          res and not res[0].get("_wildcard_catch"))

    # flag-store failure can never break selection (hot-path guard)
    _orig_state = mc.worker_wildcard_state
    try:
        mc.worker_wildcard_state = lambda: (_ for _ in ()).throw(
            RuntimeError("flag store down"))
        check("flag-store read failure degrades to sealed routing (no raise)",
              {w["id"] for w in store.workers_for_model("org~Resident-Model")}
              == {"holder"})
    finally:
        mc.worker_wildcard_state = _orig_state

    # ── [5] ranking: home above wildcard; overflow-by-ordering ──────────────
    print("\n[5] ranking (home first; overflow when home is out)")

    _reset_state()
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-rank-")
    # The wildcard box gets a GPU so that WITHOUT the home sort key it would
    # rank FIRST (gpu beats no-gpu) — proving the home key is load-bearing.
    _add("home-box", [ASSIGNED])
    _add("wild-box", [],
         gpus=[{"name": "rtx", "memory_free": 8 * 2**30, "memory_total": 24 * 2**30}])
    mc.set_worker_wildcard("wild-box", True)

    ids = [w["id"] for w in store.candidates_for_model(ASSIGNED)]
    check("candidates_for_model ranks HOME above the (GPU-better) wildcard catch",
          ids == ["home-box", "wild-box"])
    pick = store.pick_for_model(ASSIGNED)
    check("pick_for_model chooses the home worker while home is available",
          pick and pick["id"] == "home-box")

    # Home workers all unavailable (engine affirmatively unusable) -> the same
    # ranked walk lands on the wildcard box: overflow IS the ordering.
    store.register(name="home-box", url="http://home-box:9100",
                   worker_id="home-box", models=[ASSIGNED],
                   engine={"installed": False})
    pick = store.pick_for_model(ASSIGNED)
    check("overflow-by-ordering: home excluded -> wildcard worker is chosen",
          pick and pick["id"] == "wild-box")
    check("the overflow candidate is visibly a wildcard catch",
          [w.get("_wildcard_catch") for w in store.workers_for_model(ASSIGNED)]
          == [True])

    # ── [6] blocked-sibling guard ───────────────────────────────────────────
    print("\n[6] blocked-sibling guard (alias match never launders a block)")

    _reset_state()
    bl.unblock(_TEST_BLOCK_KEY)   # targeted pre-clean (idempotent)
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-block-")
    _add("sib", ["B~X"])
    bl.block("B~X", note="test sibling block")
    check("request A~X vs advertised BLOCKED B~X -> alias match refused",
          store.workers_for_model("A~X") == [])
    check("the blocked key itself still yields no candidates (requested-key gate)",
          store.workers_for_model("B~X") == [])
    _add("bare", ["X"])
    check("request A~X vs advertised UNBLOCKED bare X -> matched",
          {w["id"] for w in store.workers_for_model("A~X")} == {"bare"})
    bl.unblock("B~X")
    check("after unblock, the sibling advertisement matches again",
          {w["id"] for w in store.workers_for_model("A~X")} == {"sib", "bare"})

    # ── [7] ⭐ star ranking priority (ambiguity tie-break, post-incident) ──────
    # Operator RULING 2026-07-23: the star "shouldn't effect anything but priority
    # for ambiguous model calls." Rank key order is
    #   (home/wildcard, warm, star, gpu, last_picked, id)
    # so: home outranks everything; warm outranks star; a star breaks the tie when
    # neither box is warm. Alias-tolerant (star recorded under ~-key matches a
    # bare-key request and vice versa, via _match_keys).
    print("\n[7] star ranking priority (warm > star; home > star; alias-tolerant)")

    RANKED = "org~Ranked-Model"

    def _clear_stars():
        for _wid in list(mc.worker_boot_prewarm_state().keys()):
            mc.set_worker_boot_prewarm(_wid, None, False)

    # (a) neither warm: the STARRED worker outranks the non-starred one.
    _reset_state()
    _clear_stars()
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-star-a-")
    _add("plain-box", [RANKED])
    _add("star-box", [RANKED])
    mc.set_worker_boot_prewarm("star-box", RANKED, True)
    ids = [w["id"] for w in store.candidates_for_model(RANKED)]
    check("neither warm: the ⭐ starred worker outranks the non-starred one",
          ids[0] == "star-box")
    check("pick_for_model chooses the starred worker when nothing is warm",
          (store.pick_for_model(RANKED) or {}).get("id") == "star-box")

    # (b) WARM outranks STAR: a warm non-starred box beats a starred cold box.
    _reset_state()
    _clear_stars()
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-star-b-")
    _add("warm-box", [RANKED])
    _add("star-cold", [RANKED])
    store.heartbeat("warm-box", loaded_models=[RANKED])   # warm, NOT starred
    mc.set_worker_boot_prewarm("star-cold", RANKED, True)  # starred, cold
    ids = [w["id"] for w in store.candidates_for_model(RANKED)]
    check("WARM outranks STAR: warm-but-unstarred beats starred-but-cold",
          ids[0] == "warm-box")

    # (c) HOME (Slice A) outranks STAR: a wildcard-catch box that is STARRED still
    # loses to a plain home box. The star box is a wildcard catch for a model it
    # is NOT designated for; the home box is designated. Home wins.
    _reset_state()
    _clear_stars()
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-star-c-")
    _add("home-plain", [RANKED])                           # designated (home)
    _add("wild-star", [],                                  # undesignated…
         gpus=[{"name": "rtx", "memory_free": 8 * 2**30, "memory_total": 24 * 2**30}])
    mc.set_worker_wildcard("wild-star", True)              # …wildcard catch…
    mc.set_worker_boot_prewarm("wild-star", RANKED, True)  # …AND starred (+GPU)
    ids = [w["id"] for w in store.candidates_for_model(RANKED)]
    check("HOME outranks STAR: plain home box beats a starred (GPU) wildcard catch",
          ids == ["home-plain", "wild-star"])

    # (d) ALIAS-TOLERANT: a star recorded under the ~-qualified key matches a
    # bare-key request (and the reverse), via _match_keys — same unification as
    # Slice A. Star set as "Owner~Base"; request the bare "Base".
    _reset_state()
    _clear_stars()
    store, _tmp = isolated_worker_store(prefix="hugpy-wc-star-d-")
    _add("plainer", ["Base"])
    _add("starrer", ["Base"])
    mc.set_worker_boot_prewarm("starrer", "Owner~Base", True)   # ~-qualified star
    ids = [w["id"] for w in store.candidates_for_model("Base")]  # bare request
    check("alias-tolerant: ~-qualified star matches a bare-key request",
          ids[0] == "starrer")
    _clear_stars()

finally:
    try:
        bl.unblock(_TEST_BLOCK_KEY)   # targeted cleanup, never a blanket sweep
    except Exception:  # noqa: BLE001 — cleanup must not mask a real failure
        pass
    mc.MODELS_DISCOVERY_PATH = _ORIG_DISCOVERY
    os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
    os.environ.pop("HUGPY_AUTH_MODE", None)


print(f"\n{ok} passed, {fail} failed")
assert fail == 0, f"{fail} check(s) failed — see FAIL lines above"
if __name__ == "__main__":
    sys.exit(1 if fail else 0)
