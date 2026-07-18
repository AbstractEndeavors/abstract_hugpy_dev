"""POST /llm/workers/<id>/alloc-all — bulk-alloc route (todo t15; per-model % t48).

The operator clarified the multi-select was for ALLOC (GPU allocation), residency
kept too. The single-model alloc editor applies via POST /assign {model_key,
spill} — a CENTRAL REGISTRY write to spill_by_model (assign_model), NOT the
/ops/config relay family — so, unlike residency/pin, it does NOT restart the
agent. The bulk action must match that EXACTLY: N registry writes in one request,
per-model results like residency-all, and a constant restarting:False.

This regresses the route WITHOUT a live worker:
  * one assign_model call per selected key carrying the SAME spill contract
    (body: {"spill": {...}}) — for autofit / max GPU / CPU only / an operator-
    typed absolute GiB budget, every selected model IS meant to get the same
    contract;
  * OR one assign_model call per key with its OWN spill (body:
    {"spills": {model_key: {...}}}) — t48: a PERCENT VRAM/RAM budget must
    resolve against each model's OWN size, so two differently-sized models in
    the same bulk selection land two DIFFERENT absolute GiB numbers instead of
    one flat number (resolved once against the worker's capacity) stamped on
    both — that was the bug ("...not the total for the particular model that
    happened to be the first in the list's actual ram alloc");
  * the spill value set mirrors the editor (autofit {} / max GPU {n_gpu_layers:-1}
    / CPU only {n_gpu_layers:"off"} / custom budgets);
  * NEVER a relay / restart (restarting is always False, no /ops/config touched);
  * off-worker keys dropped (render->click staleness) → skipped, never assigned;
  * per-model results / counts / alloc label surfaced like residency-all;
  * one bad key errors that key only, the rest still apply;
  * the engine gate is evaluated per-key against THAT key's own spill in the
    `spills` path (a per-model map can carry different explicit-budget keys per
    member, unlike the single shared `spill`);
  * bad body -> 400; bad spill shape -> 400; bad spills shape -> 400; unknown
    worker -> 404.

Runs like the other tests here: venv/bin/python tests/test_bulk_alloc_route.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-bulk-alloc-test-")

import importlib
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
cr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.comms_routes")

from flask import Flask

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

app = Flask(__name__)
app.register_blueprint(wr.worker_bp)
client = app.test_client()

# Worker has a,b,c designated; a,b have files local (warmable), c does not; z is
# NOT designated.
WORKER = {"id": "wid", "name": "box", "models": ["a", "b", "c"],
          "models_local": ["a", "b"]}

_orig = (wr.get_worker, wr.assign_model, wr._kick_warm, cr.audit,
         wr._relay_worker_op, wr._model_framework)
assign_calls = []      # (worker_id, model_key, spill)
warm_calls = []        # (models,) passed to _kick_warm
relay_seen = {"n": 0}
# Engine table for the gate: base tests treat a,b,c as GGUF so the historical
# behavior (max GPU / custom apply to all) is preserved; the engine-gating block
# below overrides this with a MIXED map.
FRAMEWORKS = {"a": "gguf", "b": "gguf", "c": "gguf"}


def _fake_assign(worker_id, model_key, spill=None):
    assign_calls.append((worker_id, model_key, spill))
    return dict(WORKER)   # assign_model returns the public worker view


def _fake_warm(worker, model_keys, source):
    warm_calls.append(tuple(model_keys))
    return list(model_keys)


# Guard: the alloc path must NEVER relay to the worker (no restart). Any call to
# _relay_worker_op is a failure of the "no restart" contract.
def _relay_tripwire(*a, **k):
    relay_seen["n"] += 1
    raise AssertionError("alloc-all must NOT relay to the worker (no restart)")


try:
    wr.get_worker = lambda wid: dict(WORKER) if wid == "wid" else None
    wr.assign_model = _fake_assign
    wr._kick_warm = _fake_warm
    wr._relay_worker_op = _relay_tripwire
    cr.audit = lambda *a, **k: None
    wr._model_framework = lambda mk: FRAMEWORKS.get(mk)   # engine gate stub

    # 1) max GPU on a subset: one assign per key, SAME spill, no relay ----------
    assign_calls.clear(); warm_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a", "b"], "spill": {"n_gpu_layers": -1}})
    body = r.get_json()
    check("maxgpu: 200", r.status_code == 200)
    check("maxgpu: one assign_model per selected key", len(assign_calls) == 2)
    check("maxgpu: every assign carries the SAME spill",
          all(c[2] == {"n_gpu_layers": -1} for c in assign_calls))
    check("maxgpu: NEVER relayed (no restart)", relay_seen["n"] == 0)
    check("maxgpu: restarting is always False", body["restarting"] is False)
    check("maxgpu: alloc label", body["alloc"] == "max GPU")
    check("maxgpu: per-model results all ok",
          body["results"] == {"a": "ok", "b": "ok"})
    check("maxgpu: counts", body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2})
    check("maxgpu: only LOCAL models re-seated (a,b local; warm once)",
          warm_calls == [("a", "b")])

    # 2) autofit = empty spill (clears the override) ---------------------------
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a"], "spill": {}})
    body = r.get_json()
    check("autofit: empty spill written (clears override)",
          assign_calls == [("wid", "a", {})])
    check("autofit: label", body["alloc"] == "autofit")
    # null spill is ALSO autofit
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a"], "spill": None})
    check("autofit: null spill == {} (autofit)",
          assign_calls == [("wid", "a", {})])

    # 3) CPU only + custom budgets label correctly -----------------------------
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a"], "spill": {"n_gpu_layers": "off"}})
    check("cpu-only: label", r.get_json()["alloc"] == "CPU only")
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a"],
                          "spill": {"gpu_mem_gib": 8, "cpu_mem_gib": 16, "threads": 4}})
    check("custom: label lists budgets",
          r.get_json()["alloc"] == "8G VRAM · 16G RAM · 4 cores")

    # 4) off-worker key dropped (render->click staleness) → skipped ------------
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a", "z"], "spill": {"n_gpu_layers": -1}})
    body = r.get_json()
    check("stale: only the designated key assigned (z dropped)",
          [c[1] for c in assign_calls] == ["a"])
    check("stale: dropped off-worker key reported (distinct from engine skips)",
          body["off_worker"] == ["z"])
    check("stale: results cover only the designated key",
          body["results"] == {"a": "ok"})

    # 5) all keys off-worker → no assign, honest note --------------------------
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["z", "q"], "spill": {}})
    body = r.get_json()
    check("all-stale: 200 + no assign", r.status_code == 200 and assign_calls == [])
    check("all-stale: counts zero", body["counts"] == {"ok": 0, "error": 0, "skipped": 0, "total": 0})
    check("all-stale: note + off_worker", "note" in body and set(body["off_worker"]) == {"z", "q"})

    # 6) one bad key errors that key only; the rest still apply ----------------
    assign_calls.clear()
    def _assign_b_raises(worker_id, model_key, spill=None):
        assign_calls.append((worker_id, model_key, spill))
        if model_key == "b":
            raise RuntimeError("boom")
        return dict(WORKER)
    wr.assign_model = _assign_b_raises
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a", "b", "c"], "spill": {"n_gpu_layers": -1}})
    body = r.get_json()
    check("partial: all three attempted", len(assign_calls) == 3)
    check("partial: a,c ok; b errored", body["results"]["a"] == "ok"
          and body["results"]["c"] == "ok" and "boom" in body["results"]["b"])
    check("partial: counts 2 ok / 1 error", body["counts"] == {"ok": 2, "error": 1, "skipped": 0, "total": 3})
    check("partial: ok flag False when any errored", body["ok"] is False)
    wr.assign_model = _fake_assign

    # 7) bad body / bad spill / unknown worker ---------------------------------
    check("missing model_keys -> 400",
          client.post("/llm/workers/wid/alloc-all", json={"spill": {}}).status_code == 400)
    check("empty model_keys -> 400",
          client.post("/llm/workers/wid/alloc-all",
                      json={"model_keys": [], "spill": {}}).status_code == 400)
    check("unknown spill key -> 400",
          client.post("/llm/workers/wid/alloc-all",
                      json={"model_keys": ["a"], "spill": {"bogus": 1}}).status_code == 400)
    check("bad n_gpu_layers -> 400",
          client.post("/llm/workers/wid/alloc-all",
                      json={"model_keys": ["a"], "spill": {"n_gpu_layers": "lots"}}).status_code == 400)
    check("non-dict spill -> 400",
          client.post("/llm/workers/wid/alloc-all",
                      json={"model_keys": ["a"], "spill": 5}).status_code == 400)
    check("unknown worker -> 404",
          client.post("/llm/workers/nope/alloc-all",
                      json={"model_keys": ["a"], "spill": {}}).status_code == 404)

    # 8) the spill validator + label helpers (unit) ----------------------------
    check("validate: None -> autofit {}", wr._validate_alloc_spill(None) == ({}, None))
    check("validate: max GPU passes",
          wr._validate_alloc_spill({"n_gpu_layers": -1}) == ({"n_gpu_layers": -1}, None))
    clean, reason = wr._validate_alloc_spill({"bad": 1})
    check("validate: unknown key rejected", clean is None and "unsupported" in reason)
    check("label: empty -> autofit", wr._alloc_label({}) == "autofit")
    check("label: -1 -> max GPU", wr._alloc_label({"n_gpu_layers": -1}) == "max GPU")
    check("label: off -> CPU only", wr._alloc_label({"n_gpu_layers": "off"}) == "CPU only")

    # 9) the route is operator-gated (assign-family tier) ----------------------
    oa = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.operator_auth")
    def _gated(path, method):
        return any(method in methods and rx.match(path)
                   for methods, rx in oa._SENSITIVE)
    check("alloc-all POST is in _SENSITIVE (operator-gated)",
          _gated("/llm/workers/w1/alloc-all", "POST"))

    # ── ENGINE GATING (t15 refinement — GGUF-only alloc) ─────────────────────
    # Mixed selection: g1 gguf, t1/t2 transformers, cf comfy.
    MIXED = {"id": "wid", "name": "box",
             "models": ["g1", "t1", "t2", "cf"], "models_local": []}
    MIXED_FW = {"g1": "gguf", "t1": "transformers", "t2": "transformers", "cf": "comfy"}
    wr.get_worker = lambda wid: dict(MIXED) if wid == "wid" else None
    wr._model_framework = lambda mk: MIXED_FW.get(mk)

    # (a) mixed + explicit budget (gguf-only) → gguf applied, others skipped-with-reason
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "t1", "t2", "cf"],
                          "spill": {"gpu_mem_gib": 8}})
    body = r.get_json()
    check("mixed/custom: ONLY the gguf model assigned",
          [c[1] for c in assign_calls] == ["g1"])
    check("mixed/custom: g1 ok", body["results"]["g1"] == "ok")
    check("mixed/custom: transformers skipped-with-reason",
          "skipped" in body["results"]["t1"] and "GGUF-only" in body["results"]["t1"]
          and "transformers" in body["results"]["t1"])
    check("mixed/custom: comfy skipped-with-reason",
          "skipped" in body["results"]["cf"] and "comfy" in body["results"]["cf"])
    check("mixed/custom: counts 1 ok / 3 skipped",
          body["counts"] == {"ok": 1, "error": 0, "skipped": 3, "total": 4})
    check("mixed/custom: ok:true (an engine skip is not a failure)", body["ok"] is True)

    # (b) mixed + max GPU (n_gpu_layers only — PLACEMENT INTENT, t26) → applies
    # to EVERYONE now (Max GPU is engine-agnostic; the worker maps n_gpu_layers to
    # transformers placement). No skips.
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "t1"], "spill": {"n_gpu_layers": -1}})
    body = r.get_json()
    check("mixed/maxgpu: applies to BOTH gguf and transformers (t26)",
          sorted(c[1] for c in assign_calls) == ["g1", "t1"])
    check("mixed/maxgpu: every assign carries the max-GPU spill",
          all(c[2] == {"n_gpu_layers": -1} for c in assign_calls))
    check("mixed/maxgpu: nothing skipped",
          body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2})

    # (b2) mixed + CPU only (n_gpu_layers:"off") → also applies to everyone
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "t1", "cf"], "spill": {"n_gpu_layers": "off"}})
    body = r.get_json()
    check("mixed/cpu-only: applies to all engines (t26)",
          sorted(c[1] for c in assign_calls) == ["cf", "g1", "t1"])
    check("mixed/cpu-only: nothing skipped",
          body["counts"] == {"ok": 3, "error": 0, "skipped": 0, "total": 3})

    # (c) ALL-transformers + MAX GPU → all APPLY (placement intent, not gguf-only)
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["t1", "t2"], "spill": {"n_gpu_layers": -1}})
    body = r.get_json()
    check("all-transformers/maxgpu: all written (placement intent applies)",
          sorted(c[1] for c in assign_calls) == ["t1", "t2"])
    check("all-transformers/maxgpu: none skipped",
          body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2})

    # (c2) ALL-transformers + EXPLICIT budget → still all skipped (gguf-only class)
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["t1", "t2"], "spill": {"gpu_mem_gib": 8}})
    body = r.get_json()
    check("all-transformers/explicit: nothing written", assign_calls == [])
    check("all-transformers/explicit: ok:true", body["ok"] is True)
    check("all-transformers/explicit: all skipped",
          body["counts"] == {"ok": 0, "error": 0, "skipped": 2, "total": 2})

    # (d) AUTOFIT ({}) applies to EVERYONE regardless of engine (engine-agnostic)
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "t1", "t2", "cf"], "spill": {}})
    body = r.get_json()
    check("autofit: applied to every engine",
          sorted(c[1] for c in assign_calls) == ["cf", "g1", "t1", "t2"])
    check("autofit: counts all ok, none skipped",
          body["counts"] == {"ok": 4, "error": 0, "skipped": 0, "total": 4})

    # (e) unknown engine: only the EXPLICIT class fails safe (skipped); a
    # placement-intent mode (Max GPU) applies even to an unknown engine.
    wr._model_framework = lambda mk: None
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1"], "spill": {"gpu_mem_gib": 8}})
    body = r.get_json()
    check("unknown-engine/explicit: skipped (fail safe)",
          assign_calls == [] and "skipped" in body["results"]["g1"]
          and "unknown engine" in body["results"]["g1"])
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1"], "spill": {"n_gpu_layers": -1}})
    check("unknown-engine/maxgpu: APPLIES (placement intent, not gated)",
          [c[1] for c in assign_calls] == ["g1"])
    wr._model_framework = lambda mk: MIXED_FW.get(mk)

    # (f) the helpers (unit): narrowed _alloc_is_gguf_only + _alloc_spill_ok_for_engine
    check("gguf_only: autofit is NOT gguf-only", wr._alloc_is_gguf_only({}) is False)
    check("gguf_only: None is NOT gguf-only", wr._alloc_is_gguf_only(None) is False)
    check("gguf_only: Max GPU (n_gpu_layers only) is NOT gguf-only (t26)",
          wr._alloc_is_gguf_only({"n_gpu_layers": -1}) is False)
    check("gguf_only: CPU only (n_gpu_layers only) is NOT gguf-only (t26)",
          wr._alloc_is_gguf_only({"n_gpu_layers": "off"}) is False)
    check("gguf_only: gpu_mem_gib budget IS gguf-only", wr._alloc_is_gguf_only({"gpu_mem_gib": 8}) is True)
    check("gguf_only: cpu_mem_gib budget IS gguf-only", wr._alloc_is_gguf_only({"cpu_mem_gib": 8}) is True)
    check("gguf_only: threads IS gguf-only", wr._alloc_is_gguf_only({"threads": 4}) is True)
    check("gguf_only: tensor_split IS gguf-only", wr._alloc_is_gguf_only({"tensor_split": [0.5, 0.5]}) is True)
    check("gguf_only: a budget ALONGSIDE n_gpu_layers is still gguf-only",
          wr._alloc_is_gguf_only({"n_gpu_layers": -1, "gpu_mem_gib": 8}) is True)
    ok1, r1 = wr._alloc_spill_ok_for_engine({}, "t1")
    check("single-gate: autofit ok on transformers", ok1 is True and r1 is None)
    okp, rp = wr._alloc_spill_ok_for_engine({"n_gpu_layers": -1}, "t1")
    check("single-gate: Max GPU OK on transformers now (t26)", okp is True and rp is None)
    okc, rc = wr._alloc_spill_ok_for_engine({"n_gpu_layers": "off"}, "t1")
    check("single-gate: CPU only OK on transformers now (t26)", okc is True and rc is None)
    ok2, r2 = wr._alloc_spill_ok_for_engine({"gpu_mem_gib": 8}, "t1")
    check("single-gate: explicit budget REJECTED on transformers",
          ok2 is False and "GGUF-only" in r2 and "transformers" in r2)
    ok3, r3 = wr._alloc_spill_ok_for_engine({"gpu_mem_gib": 8}, "g1")
    check("single-gate: explicit budget ok on a gguf model", ok3 is True and r3 is None)

    # (g) SINGLE-MODEL /assign route enforces the gate too (defense-in-depth).
    # A gguf-only spill on a transformers key -> 409; autofit -> allowed. (`wr`
    # IS the worker_routes module — patch its collaborators in place.)
    _orig_missing = wr._central_missing_reason
    _orig_disk = wr._disk_preflight_reason
    _orig_models = wr.get_models_dict
    try:
        wr._central_missing_reason = lambda mk: None          # central has it
        wr._disk_preflight_reason = lambda w, mk: None         # fits
        wr.get_models_dict = lambda dict_return=False: {"t1": {}, "g1": {}}
        rr = client.post("/llm/workers/wid/assign",
                         json={"model_key": "t1", "spill": {"gpu_mem_gib": 8}})
        check("single /assign: gguf-only budget on transformers -> 409",
              rr.status_code == 409 and "GGUF-only" in (rr.get_json() or {}).get("error", ""))
        assign_calls.clear()
        rr = client.post("/llm/workers/wid/assign",
                         json={"model_key": "t1", "spill": {}})
        check("single /assign: autofit on transformers allowed",
              rr.status_code == 200 and assign_calls == [("wid", "t1", {})])
        rr = client.post("/llm/workers/wid/assign",
                         json={"model_key": "g1", "spill": {"n_gpu_layers": -1}})
        check("single /assign: gguf-only on a gguf model allowed",
              rr.status_code == 200)
    finally:
        wr._central_missing_reason = _orig_missing
        wr._disk_preflight_reason = _orig_disk
        wr.get_models_dict = _orig_models

    # ── PERCENTAGE → GiB resolution is a UI-side concern (BudgetInput/AllocControl
    # resolve % against the worker's effective capacity AT APPLY TIME; the wire
    # carries concrete gpu_mem_gib/cpu_mem_gib). The backend sees only the resolved
    # GiB, and the validator accepts them as before — verified here that a resolved
    # budget spill is a normal gguf-only custom alloc. The %→GiB math + clamp is
    # exercised in the UI; there is no server route to test for it (no schema
    # change). This asserts the contract the UI resolves TO stays valid. ───────
    clean, reason = wr._validate_alloc_spill({"gpu_mem_gib": 6.0, "cpu_mem_gib": 12.0})
    check("resolved %→GiB budget validates as a normal custom spill",
          reason is None and clean == {"gpu_mem_gib": 6.0, "cpu_mem_gib": 12.0})
    check("resolved budget is gguf-only", wr._alloc_is_gguf_only(clean) is True)

    # ── PER-MODEL alloc (t48): `spills: {model_key: spill}` instead of one
    # `spill` broadcast to every key. This is how a bulk PERCENT VRAM/RAM budget
    # rides the wire now: the UI resolves the percent against EACH model's own
    # size client-side (still no percent concept on the wire) and sends the
    # resulting per-model absolutes here in one request, so a 40% budget on a
    # big model and a 40% budget on a small model land as two DIFFERENT GiB
    # numbers instead of one flat number stamped on both. ───────────────────
    wr.get_worker = lambda wid: dict(WORKER) if wid == "wid" else None
    wr._model_framework = lambda mk: FRAMEWORKS.get(mk)

    # (h1) two gguf models, two DIFFERENT resolved budgets in one request.
    assign_calls.clear(); warm_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a", "b"],
                          "spills": {"a": {"gpu_mem_gib": 9.6, "cpu_mem_gib": 4.0},
                                     "b": {"gpu_mem_gib": 2.4, "cpu_mem_gib": 1.0}}})
    body = r.get_json()
    check("per-model: 200", r.status_code == 200)
    check("per-model: one assign_model per key, EACH its OWN spill",
          {c[1]: c[2] for c in assign_calls} ==
          {"a": {"gpu_mem_gib": 9.6, "cpu_mem_gib": 4.0},
           "b": {"gpu_mem_gib": 2.4, "cpu_mem_gib": 1.0}})
    a_spill = next(c[2] for c in assign_calls if c[1] == "a")
    b_spill = next(c[2] for c in assign_calls if c[1] == "b")
    check("per-model: distinct per-model values actually landed", a_spill != b_spill)
    check("per-model: NEVER relayed (no restart)", relay_seen["n"] == 0)
    check("per-model: restarting is always False", body["restarting"] is False)
    check("per-model: alloc label", body["alloc"] == "per-model")
    check("per-model: per-model results all ok",
          body["results"] == {"a": "ok", "b": "ok"})
    check("per-model: counts", body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2})
    check("per-model: only LOCAL models re-seated (a,b local; warm once)",
          warm_calls == [("a", "b")])

    # (h2) a key missing from `spills` (caller only sent a subset) -> autofit
    # for that key (same {}-is-autofit convention as the broadcast path).
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["a", "b"],
                          "spills": {"a": {"n_gpu_layers": -1}}})
    body = r.get_json()
    check("per-model: key absent from spills resolves to autofit",
          dict((c[1], c[2]) for c in assign_calls) == {"a": {"n_gpu_layers": -1}, "b": {}})

    # (h3) engine gate is evaluated PER KEY's OWN spill (mixed selection: g1
    # gets an explicit gguf-only budget, t1 gets autofit — only g1 should be
    # gated on, and it should NOT be skipped since it IS gguf).
    wr.get_worker = lambda wid: dict(MIXED) if wid == "wid" else None
    wr._model_framework = lambda mk: MIXED_FW.get(mk)
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "t1"],
                          "spills": {"g1": {"gpu_mem_gib": 8}, "t1": {"gpu_mem_gib": 4}}})
    body = r.get_json()
    check("per-model engine gate: gguf member applies",
          body["results"]["g1"] == "ok")
    check("per-model engine gate: transformers member skipped (its OWN spill is gguf-only)",
          "skipped" in body["results"]["t1"] and "GGUF-only" in body["results"]["t1"])
    check("per-model engine gate: counts 1 ok / 1 skipped",
          body["counts"] == {"ok": 1, "error": 0, "skipped": 1, "total": 2})
    check("per-model engine gate: only g1 actually written",
          [c[1] for c in assign_calls] == ["g1"])

    # (h4) off-worker key + stale note behave the same as the broadcast path.
    assign_calls.clear()
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "zz"], "spills": {"g1": {}}})
    body = r.get_json()
    check("per-model: off-worker key dropped, reported separately",
          body["off_worker"] == ["zz"] and body["results"] == {"g1": "ok"})

    # (h5) bad shapes -> 400.
    check("spills: non-dict -> 400",
          client.post("/llm/workers/wid/alloc-all",
                      json={"model_keys": ["g1"], "spills": "nope"}).status_code == 400)
    check("spills: bad per-key spill -> 400",
          client.post("/llm/workers/wid/alloc-all",
                      json={"model_keys": ["g1"], "spills": {"g1": {"bogus": 1}}}).status_code == 400)

    wr.get_worker = lambda wid: dict(WORKER) if wid == "wid" else None
    wr._model_framework = lambda mk: FRAMEWORKS.get(mk)
finally:
    (wr.get_worker, wr.assign_model, wr._kick_warm, cr.audit,
     wr._relay_worker_op, wr._model_framework) = _orig

print(f"\nall {ok} checks passed")
