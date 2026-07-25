"""CASE A auto-block + CASE B manifest-orphan surfacing (operator ruling 2026-07-25).

  "models that aren't in the manifest and models that simply will not fit on a
   worker no matter what, if allocated, should be blocked. the user will be
   forced to acknowledge it and act or not"

The approved shape treats the two as DIFFERENT problems:

CASE A — fits NO worker in ANY mode -> AUTO-BLOCK (arithmetic, not judgement),
reversible via the existing /unblock. Covered here:
  * alloc_modes.worker_fit_verdict — three-valued (True/False/None);
  * alloc_modes.fleet_fit_verdict — blockable IFF >=1 confident refusal and
    ZERO "fits"; never on missing data; never when ONE worker can hold it;
  * blocklist.auto_block — by="auto", declines on an operator unblock and on an
    operator-authored block; refreshes its own record;
  * blocklist.unblock stickiness — an operator unblock leaves the
    operator_unblocked tombstone; an auto unblock deletes without one;
  * workers.fleet_fit_for_model / maybe_auto_block — online-only, degrade-safe;
  * the /assign hook — an infeasible model is auto-blocked and refused with the
    machine's own reasoning (blocked_by:"auto"), and an operator unblock is NOT
    undone by a subsequent /assign.

CASE B — NOT in central's manifest -> a FAULT, never a block (/block 404s on a
non-manifest key by design). Covered here:
  * the hugpy-fleet-triage orphan fault fires, names the worker + count;
  * it stays SILENT when the manifest read fails (degrade-not-guess);
  * the /assign clear path: an empty spill on an ALREADY-DESIGNATED orphan key
    clears the row instead of 404ing, while a genuinely unknown key still 404s
    and a NON-empty spill on an orphan key still 404s.

Run:  venv/bin/python tests/test_auto_block_infeasible.py
"""
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-auto-block-test-")

import importlib
from flask import Flask

# See tests/worker_store_isolation.py — PROJECTS_HOME does NOT isolate the
# worker registry; section 5 drives real /assign routes that reach the real
# module-level assign_model().
from worker_store_isolation import swap_worker_store

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

GIB = 2 ** 30

am = importlib.import_module("abstract_hugpy_dev.managers.alloc_modes")
bl = importlib.import_module("abstract_hugpy_dev.comms.blocklist")


def _reset():
    for k in list(bl.blocked_keys()):
        bl.unblock(k, by="auto")          # auto-unblock leaves no tombstone
    # also clear any tombstones a prior section left
    try:
        from abstract_hugpy_dev.comms.settings import settings_store
        for k in list((settings_store.all(bl.NS) or {}).keys()):
            settings_store.delete(bl.NS, k)
    except Exception:
        pass


# ── 1) worker_fit_verdict: three-valued, loose-by-design ─────────────────────
check("fit: fits the GPU outright -> True",
      am.worker_fit_verdict("gguf", 10 * GIB, 24 * GIB, 15 * GIB) is True)
check("fit: too big for GPU but fits RAM -> True",
      am.worker_fit_verdict("gguf", 14 * GIB, 8 * GIB, 15 * GIB) is True)
check("fit: fits only GPU+RAM COMBINED -> True (max-gpu/max-ram spill across "
      "both; combined is the real physical ceiling)",
      am.worker_fit_verdict("gguf", 30 * GIB, 24 * GIB, 15 * GIB) is True)
check("fit: exceeds GPU AND RAM AND combined -> False (confident refusal)",
      am.worker_fit_verdict("gguf", 68 * GIB, 24 * GIB, 15 * GIB) is False)
check("fit: exactly combined capacity -> True (<=, never a fencepost block)",
      am.worker_fit_verdict("gguf", 39 * GIB, 24 * GIB, 15 * GIB) is True)
check("fit: NO headroom factor is applied here — a fudge is right for picking a "
      "default that must succeed, wrong for taking a model OUT of the pool",
      am.worker_fit_verdict("gguf", int(0.99 * 24 * GIB), 24 * GIB, None) is True)
check("fit: transformers gets the SAME combined ceiling (accelerate offloads too)",
      am.worker_fit_verdict("transformers", 30 * GIB, 24 * GIB, 15 * GIB) is True)

# degrade-not-guess
check("fit: unknown model size -> None (no vote)",
      am.worker_fit_verdict("gguf", None, 24 * GIB, 15 * GIB) is None)
check("fit: zero/garbage model size -> None",
      am.worker_fit_verdict("gguf", 0, 24 * GIB, 15 * GIB) is None
      and am.worker_fit_verdict("gguf", "big", 24 * GIB, 15 * GIB) is None)
check("fit: BOTH totals unknown -> None (unmeasured box has no opinion)",
      am.worker_fit_verdict("gguf", 68 * GIB, None, None) is None)
check("fit: ONE known total is enough to vote (GPU known, RAM unknown)",
      am.worker_fit_verdict("gguf", 68 * GIB, 24 * GIB, None) is False
      and am.worker_fit_verdict("gguf", 10 * GIB, 24 * GIB, None) is True)

# ── 2) fleet_fit_verdict: the roll-up, and the asymmetry that keeps it safe ──
BIG, SMALL = 68 * GIB, 4 * GIB
def _box(name, gpu, ram, size=BIG, engine="gguf"):
    return {"name": name, "engine": engine, "model_bytes": size,
            "gpu_total_bytes": gpu, "ram_total_bytes": ram}

v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB),
                          _box("op", None, 30 * GIB),
                          _box("computron", 8 * GIB, 16 * GIB)])
check("fleet: fits NO worker -> blockable True", v["blockable"] is True)
check("fleet: fits_somewhere False + every refuser named",
      v["fits_somewhere"] is False
      and set(v["refused_by"]) == {"ae", "op", "computron"})
check("fleet: the why is the operator-facing reasoning (names the size, the "
      "biggest GPU/RAM, and the escape hatch)",
      "68.0 GiB exceeds every worker" in v["why"]
      and "24.0" in v["why"] and "unblock to override" in v["why"]
      and v["why"].startswith("auto:"))

v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB),
                          _box("bigbox", 80 * GIB, 256 * GIB)])
check("fleet: fits ONE worker -> NEVER blockable ('no matter what' means the "
      "whole fleet, not one box)",
      v["blockable"] is False and v["fits_somewhere"] is True
      and v["fits_on"] == ["bigbox"] and v["refused_by"] == ["ae"])

v = am.fleet_fit_verdict([_box("ae", None, None), _box("op", None, None)])
check("fleet: ALL workers unknown -> not blockable, fits_somewhere None",
      v["blockable"] is False and v["fits_somewhere"] is None
      and set(v["unknown"]) == {"ae", "op"} and "degrade-not-guess" in v["why"])

v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB), _box("op", None, None)])
check("fleet: one confident refusal + one unknown -> STILL blockable, and the "
      "why discloses the box that had no data",
      v["blockable"] is True and "1 worker(s) had no data" in v["why"])

v = am.fleet_fit_verdict([_box("ae", 24 * GIB, 15 * GIB, size=None)])
check("fleet: unsizable model -> nobody votes -> not blockable",
      v["blockable"] is False and v["fits_somewhere"] is None)

check("fleet: EMPTY fleet -> not blockable (an absent fleet is not evidence)",
      am.fleet_fit_verdict([])["blockable"] is False
      and am.fleet_fit_verdict(None)["blockable"] is False)
check("fleet: junk rows are skipped, never crash",
      am.fleet_fit_verdict(["nope", None, 7])["blockable"] is False)

# ── 3) blocklist: auto authorship + operator-unblock STICKINESS ──────────────
_reset()
rec = bl.auto_block("A~1", "auto: 68.0 GiB exceeds every worker — unblock to override")
check("auto_block: writes by='auto' (machine-authored, auditable apart from "
      "an operator block)",
      rec is not None and rec["by"] == "auto" and rec["blocked"] is True)
check("auto_block: the note carries the reasoning the console shows",
      "unblock to override" in rec["note"])
check("auto_block: the model is actually blocked", bl.is_blocked("A~1") is True)
check("auto_block: re-stamping its OWN record is allowed (refreshes numbers)",
      (bl.auto_block("A~1", "auto: refreshed") or {}).get("note") == "auto: refreshed")

check("stickiness: operator_unblocked False while merely blocked",
      bl.operator_unblocked("A~1") is False)
check("unblock(operator): reports was_blocked True",
      bl.unblock("A~1", by="operator") is True)
check("unblock(operator): model is released", bl.is_blocked("A~1") is False)
check("unblock(operator): leaves the sticky operator_unblocked marker",
      bl.operator_unblocked("A~1") is True)
check("unblock(operator): the inert tombstone is NOT in blocked_keys "
      "(nothing downstream can see it)",
      "A~1" not in bl.blocked_keys())
check("stickiness: THE POINT — auto_block DECLINES a key the operator released",
      bl.auto_block("A~1", "auto: still too big") is None
      and bl.is_blocked("A~1") is False)
check("stickiness: an OPERATOR block still works on that key (a human may "
      "always change their mind; only the MACHINE is suppressed)",
      bl.block("A~1", by="operator")["by"] == "operator"
      and bl.is_blocked("A~1") is True)
check("stickiness: an operator block CLEARS the tombstone (whole-record write)",
      bl.operator_unblocked("A~1") is False)

_reset()
bl.block("A~2", by="operator", note="operator's own reason")
check("authorship: auto_block DECLINES to overwrite an operator-authored block "
      "(a machine must not rewrite a human's note)",
      bl.auto_block("A~2", "auto: whatever") is None
      and (bl.block_info("A~2") or {}).get("note") == "operator's own reason")

_reset()
bl.auto_block("A~3", "auto: too big")
check("unblock(auto): the machine retracting its OWN block deletes without a "
      "tombstone — only a human's override earns stickiness",
      bl.unblock("A~3", by="auto") is True
      and bl.operator_unblocked("A~3") is False)
check("unblock(auto): so a LATER, genuinely-correct auto-block still lands",
      bl.auto_block("A~3", "auto: too big again") is not None)
_reset()

# ── 4) workers.py glue: online-only, degrade-safe ────────────────────────────
W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")

_wo = (W._model_engine, W._model_size_bytes)
try:
    W._model_engine = lambda mk: "gguf"
    W._model_size_bytes = lambda mk: BIG

    rows = [{"name": "ae", "status": "online", "gpu_total": 24 * GIB,
             "ram_total": 15 * GIB},
            {"name": "op", "status": "online", "gpu_total": None,
             "ram_total": 30 * GIB}]
    r = W.fleet_fit_for_model("M~big", rows)
    check("glue: resolves each online worker's totals and blocks a fleet-wide "
          "misfit", r["blockable"] is True and set(r["refused_by"]) == {"ae", "op"})

    rows2 = rows + [{"name": "bigbox", "status": "online",
                     "gpu_total": 80 * GIB, "ram_total": 256 * GIB}]
    check("glue: one box that CAN hold it -> not blockable",
          W.fleet_fit_for_model("M~big", rows2)["blockable"] is False)

    rows3 = [dict(rows[0]), {"name": "bigbox", "status": "offline",
                             "gpu_total": 80 * GIB, "ram_total": 256 * GIB}]
    r3 = W.fleet_fit_for_model("M~big", rows3)
    check("glue: an OFFLINE box neither refuses nor rescues — it is simply not "
          "a candidate (a rebooting box must not vote a model out)",
          "bigbox" not in r3["refused_by"] and "bigbox" not in r3["fits_on"]
          and "bigbox" not in r3["unknown"])
    check("glue: an ALL-offline fleet is unblockable",
          W.fleet_fit_for_model("M~big", [rows3[1]])["blockable"] is False)

    W._model_size_bytes = lambda mk: None
    check("glue: unsizable model -> not blockable (degrade-not-guess)",
          W.fleet_fit_for_model("M~big", rows)["blockable"] is False)

    def _boom(mk):
        raise RuntimeError("manifest exploded")
    W._model_size_bytes = _boom
    check("glue: an EXCEPTION never manufactures a block",
          W.fleet_fit_for_model("M~big", rows)["blockable"] is False)

    # maybe_auto_block composes the two
    W._model_size_bytes = lambda mk: BIG
    _reset()
    check("maybe_auto_block: blocks the fleet-wide misfit",
          W.maybe_auto_block("M~big", rows) is not None
          and bl.is_blocked("M~big") is True)
    _reset()
    check("maybe_auto_block: returns None (and blocks nothing) when it fits",
          W.maybe_auto_block("M~big", rows2) is None
          and bl.is_blocked("M~big") is False)
    bl.auto_block("M~big", "auto: prior")
    bl.unblock("M~big", by="operator")
    check("maybe_auto_block: honors the operator-unblock tombstone",
          W.maybe_auto_block("M~big", rows) is None
          and bl.is_blocked("M~big") is False)
    _reset()
finally:
    (W._model_engine, W._model_size_bytes) = _wo

# ── 5) the /assign hook + the CASE-B clear path ──────────────────────────────
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
cr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.comms_routes")
app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

_orig = (wr.get_models_dict, cr.audit, wr._central_missing_reason,
         wr._disk_preflight_reason, wr.get_worker, W._model_engine,
         W._model_size_bytes, W.list_workers)
try:
    wr.get_models_dict = lambda dict_return=False: {"M~big": {}, "M~ok": {}}
    cr.audit = lambda *a, **k: None
    wr._central_missing_reason = lambda mk: None
    wr._disk_preflight_reason = lambda w, mk: None
    W._model_engine = lambda mk: "gguf"
    W._model_size_bytes = lambda mk: BIG if mk == "M~big" else SMALL
    W.list_workers = lambda: [
        {"name": "ae", "status": "online", "gpu_total": 24 * GIB,
         "ram_total": 15 * GIB}]

    with swap_worker_store():
        # ae designates an ORPHAN key (not in the stubbed manifest) with a spill
        # — exactly the live ae rows (comfy-dreamshaper-8-pruned-1 & co).
        W.worker_store.register(worker_id="w1", name="ae", url="http://x")
        W.assign_model("w1", "orphan~row", spill={"n_gpu_layers": 5})
        wr.get_worker = lambda wid: W.worker_store._load().get(wid)

        _reset()
        # ── CASE A at the route ──────────────────────────────────────────
        ra = client.post("/llm/workers/w1/assign", json={"model_key": "M~big"})
        rj = ra.get_json() or {}
        check("route /assign: an infeasible model is AUTO-BLOCKED on allocation "
              "(the operator's 'if allocated' trigger)",
              bl.is_blocked("M~big") is True)
        check("route /assign: and refused 409 in the same breath",
              ra.status_code == 409)
        check("route /assign: the refusal is HONEST about authorship "
              "(blocked_by 'auto', never 'by the operator')",
              rj.get("blocked_by") == "auto"
              and "by the operator" not in rj.get("error", ""))
        check("route /assign: the refusal carries the machine's own numbers so "
              "the operator can act",
              "exceeds every worker" in rj.get("error", "")
              and "unblock to override" in rj.get("error", ""))
        check("route /assign: the block record is machine-authored",
              (bl.block_info("M~big") or {}).get("by") == "auto")

        # ── THE STICKINESS PROOF ─────────────────────────────────────────
        ru = client.post("/llm/models/M~big/unblock", json={})
        check("route /unblock: releases the auto-blocked model",
              ru.status_code == 200 and bl.is_blocked("M~big") is False)
        ra2 = client.post("/llm/workers/w1/assign", json={"model_key": "M~big"})
        check("route: THE POINT — a re-/assign does NOT re-block what a human "
              "just released (otherwise the unblock button reads as broken)",
              bl.is_blocked("M~big") is False)
        check("route: and that assign is no longer refused on the block gate",
              ra2.status_code != 409)

        _reset()
        # a model that FITS is never touched
        rok = client.post("/llm/workers/w1/assign", json={"model_key": "M~ok"})
        check("route /assign: a model that fits is never auto-blocked",
              bl.is_blocked("M~ok") is False and rok.status_code == 200)

        # ── CASE B: the clear path ───────────────────────────────────────
        before = (W.worker_store._load()["w1"].get("spill_by_model") or {})
        check("case B setup: the orphan row exists and is not in the manifest",
              "orphan~row" in before
              and "orphan~row" not in wr.get_models_dict(dict_return=True))
        rc = client.post("/llm/workers/w1/assign",
                         json={"model_key": "orphan~row", "spill": {}})
        after = (W.worker_store._load()["w1"].get("spill_by_model") or {})
        check("route /assign: an EMPTY spill on an already-designated orphan "
              "CLEARS the row instead of 404ing (cleanup, never a new "
              "assignment)",
              rc.status_code == 200 and "orphan~row" not in after)

        check("route /assign: a genuinely UNKNOWN key still 404s with the "
              "original manifest message (the name-vs-key slip this gate exists "
              "for is unaffected)",
              client.post("/llm/workers/w1/assign",
                          json={"model_key": "never~seen", "spill": {}}
                          ).status_code == 404)
        check("route /assign: a NON-EMPTY spill on an orphan key still 404s — "
              "the relaxation can only remove a row, never write a contract",
              client.post("/llm/workers/w1/assign",
                          json={"model_key": "orphan~row2",
                                "spill": {"n_gpu_layers": 3}}).status_code == 404)
        check("route /unassign: the full-removal path clears an orphan too "
              "(it has no manifest gate — the inconsistency the relaxation fixed)",
              client.post("/llm/workers/w1/unassign",
                          json={"model_key": "orphan~row"}).status_code == 200
              and "orphan~row" not in
              (W.worker_store._load()["w1"].get("models") or []))
finally:
    (wr.get_models_dict, cr.audit, wr._central_missing_reason,
     wr._disk_preflight_reason, wr.get_worker, W._model_engine,
     W._model_size_bytes, W.list_workers) = _orig
_reset()

# ── 6) CASE B: the hugpy-fleet-triage orphan FAULT ───────────────────────────
_triage_path = (Path(__file__).resolve().parents[2] / "bin" / "hugpy-fleet-triage")
spec = importlib.util.spec_from_loader(
    "hugpy_fleet_triage",
    importlib.machinery.SourceFileLoader("hugpy_fleet_triage", str(_triage_path)))
tri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tri)

_FLEET = [{"name": "ae", "status": "online", "pkg_version": "0.1.208",
           "version_ok": True, "slot_capable": True,
           "gpus": [{"memory_free": 20 * GIB, "memory_total": 24 * GIB}],
           "slots": [], "storage": {},
           "models": ["Qwen~Real", "comfy-dreamshaper-8-pruned-1",
                      "comfy-qwen-3-4b", "comfy-z-image-turbo-bf16-1"],
           "spill_by_model": {"comfy-qwen-3-4b": {"n_gpu_layers": 1}}},
          {"name": "op", "status": "online", "pkg_version": "0.1.208",
           "version_ok": True, "slot_capable": False, "gpus": [], "slots": [],
           "storage": {}, "models": ["Qwen~Real"], "spill_by_model": {}}]
_MANIFEST = [{"model_key": "Qwen~Real"}]


def _run_triage(fleet, manifest):
    """Drive collect() with only the two HTTP reads it needs stubbed."""
    def _get(url, timeout=20.0):
        if url.endswith("/llm/workers"):
            return fleet, None
        if url.endswith("/models"):
            return (manifest, None) if manifest is not None else (None, "boom")
        return None, "not stubbed"
    _o = (tri._get, tri._code, tri.subprocess.run, tri.shutil.disk_usage)
    try:
        tri._get = _get
        tri._code = lambda url, timeout=12.0: 200
        tri.subprocess.run = lambda *a, **k: type(
            "R", (), {"stdout": "active"})()
        tri.shutil.disk_usage = lambda p: type(
            "D", (), {"free": 100 * GIB, "total": 0, "used": 0})()
        return tri.collect()
    finally:
        (tri._get, tri._code, tri.subprocess.run, tri.shutil.disk_usage) = _o


d = _run_triage(_FLEET, _MANIFEST)
faults = [f for f in d["faults"] if "not in central's manifest" in f]
check("triage: the orphan fault FIRES", len(faults) == 1)
check("triage: it NAMES the worker and the count (the operator's wording)",
      faults[0].startswith("worker ae holds 3 allocation row(s) for models not "
                           "in central's manifest"))
check("triage: it says why it matters and what to do",
      "routing can never reach them" in faults[0]
      and "clear the rows or re-add the models" in faults[0])
check("triage: it lists the actual keys",
      all(k in faults[0] for k in ("comfy-dreamshaper-8-pruned-1",
                                   "comfy-qwen-3-4b",
                                   "comfy-z-image-turbo-bf16-1")))
check("triage: a CLEAN worker raises no orphan fault", "op" not in faults[0])
check("triage: the machine view carries the per-worker orphan list",
      {w["name"]: w.get("orphan_designations") for w in d["workers"]}
      == {"ae": ["comfy-dreamshaper-8-pruned-1", "comfy-qwen-3-4b",
                 "comfy-z-image-turbo-bf16-1"], "op": []})
check("triage: a spill-only row (no models entry) is caught too",
      "comfy-qwen-3-4b" in faults[0])

d2 = _run_triage(_FLEET, None)
check("triage: manifest UNREADABLE -> NO orphan fault (degrade-not-guess; the "
      "alternative is reporting every designation on the fleet as an orphan)",
      not [f for f in d2["faults"] if "not in central's manifest" in f])
check("triage: and it says so in notes, so the gap is visible not silent",
      any("skipped the orphan-designation check" in n for n in d2["notes"]))

d3 = _run_triage([dict(_FLEET[1])], _MANIFEST)
check("triage: no orphans anywhere -> silent",
      not [f for f in d3["faults"] if "not in central's manifest" in f])

check("triage: still READ-ONLY by construction (GETs only — no mutating verb "
      "anywhere in the collector)",
      not any(s in _triage_path.read_text()
              for s in ("urlopen(Request", "method=\"POST\"", "method='POST'")))

print(f"\nALL {ok} auto-block / manifest-orphan checks passed")
