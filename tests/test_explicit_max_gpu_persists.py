"""EXPLICIT max-gpu must be distinguishable from CLEAR (bugfix 2026-07-25).

THE BUG (reproduced live before this fix): max-gpu was the ONE allocation mode
that could not be saved. Its wire encoding was ``{}``, but ``{}`` is ALSO the
"clear this override" signal in ``assign_model`` — so selecting max-gpu in the
console DELETED the row instead of writing it, and ``/assign`` returned a
success either way. The operator could not tell their choice had no effect.

It went unnoticed for a long time because a cleared row falls through to the
DERIVED default, which for most models IS max-gpu — so it did the right thing by
accident. That stopped when derived defaults landed: Qwen2.5-VL-7B-Instruct-GGUF
(12.29 GiB) on computron's ~7.6 GiB card derives **max-ram**, so the operator's
explicit max-gpu silently became max-ram and the load refused.

THE FIX: an explicit max-gpu persists as ``{"alloc_mode": "max-gpu"}`` (non-empty
-> writable), while ``{}`` keeps meaning CLEAR. The persisted key is central-side
bookkeeping and is STRIPPED at the emission seam (``spill_for``), so the wire
stays byte-identical to a blank max-gpu on every worker version.

Covers, in order:
  1. explicit max-gpu round-trips: write -> read back -> still max-gpu;
  2. ``{}`` still CLEARS (the console's "↺ Auto — derived" control);
  3. the manifest-orphan cleanup path's safety argument still holds — an empty
     spill remains structurally incapable of writing a contract;
  4. a DERIVED max-gpu still does NOT persist (it must re-derive);
  5. the worker wire is unchanged, incl. a gated-DOWN (pre-0.1.203) worker;
  6. the response reflects what was actually persisted.

Run:  venv/bin/python tests/test_explicit_max_gpu_persists.py
"""
import os
import sys

import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-maxgpu-test-")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

GIB = 2 ** 30

from abstract_hugpy_dev.managers.alloc_modes import (
    mode_to_spill, normalize_spill, derive_alloc_mode, gate_spill_for_worker,
    default_allocation)

# ── 1. the encoding is distinguishable at the pure layer ────────────────────
check("an explicit max-gpu is NON-EMPTY (so assign_model writes it)",
      mode_to_spill("max-gpu", explicit_pick=True) == {"alloc_mode": "max-gpu"})
check("a derived max-gpu is still {} (so it stays unpersisted)",
      mode_to_spill("max-gpu") == {})
check("the two encodings are distinguishable — THE WHOLE BUG",
      mode_to_spill("max-gpu", explicit_pick=True) != mode_to_spill("max-gpu"))
check("both still DERIVE back to max-gpu (same mode, different provenance)",
      derive_alloc_mode({"alloc_mode": "max-gpu"}) == "max-gpu"
      and derive_alloc_mode({}) == "max-gpu")
check("a console-sent max-gpu survives normalization instead of collapsing",
      normalize_spill({"alloc_mode": "max-gpu"})[0] == {"alloc_mode": "max-gpu"})
check("the legacy alias 'autofit' resolves to the same canonical encoding",
      normalize_spill({"alloc_mode": "autofit"})[0] == {"alloc_mode": "max-gpu"})

# ── the store seam ───────────────────────────────────────────────────────────
from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers as W
from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import WorkerStore

# THE LIVE REPRO's shape: a 12.29 GiB GGUF on a ~7.6 GiB card. Its DERIVED
# default is max-ram (too big for the card, fits RAM) — so a lost max-gpu does
# NOT silently land on max-gpu here. That is exactly what made the bug visible.
_SIZES = {"vl7b": int(12.29 * GIB), "small": 2 * GIB}
_ENGINES = {"vl7b": "gguf", "small": "gguf"}
_REAL = (W._model_size_bytes, W._model_engine, W._model_moe_detail)
W._model_size_bytes = lambda mk: _SIZES.get(mk)
W._model_engine = lambda mk: _ENGINES.get(mk)
W._model_moe_detail = lambda mk: None



W.worker_store = WorkerStore(
    path=os.path.join(os.environ["PROJECTS_HOME"], "wk.json"))
store = W.worker_store
W._assign_memory_path = lambda: os.path.join(
    os.environ["PROJECTS_HOME"], "worker_assignments.json")

w = store.register(name="computron", url="http://c:9100", pkg_version="0.1.209")
wid = w["id"]
with store._transaction() as wk:
    wk[wid]["gpus"] = [{"name": "RTX 4060", "memory_total": int(7.6 * GIB),
                        "memory_free": int(7.0 * GIB)}]
    wk[wid]["ram_total"] = 64 * GIB
store.assign_model(wid, "vl7b")

# ── 4. the DERIVED default must NOT persist ─────────────────────────────────
# Asserted against derived_allocation_for (the ALLOCATION view — the one
# spill_for actually emits from), not derived_default_for (the NAME view). The
# two DISAGREE for a large dense GGUF: the name view short-circuits every dense
# GGUF to "max-gpu" while the allocation view walks the tree and returns
# "max-ram". That divergence is PRE-EXISTING and outside this bugfix (reported,
# not changed here) — the emitting seam is what governs behavior, so that is
# what this test pins.
check("PRECONDITION (the live repro): this model's DERIVED allocation is "
      "max-ram, NOT max-gpu — so a dropped max-gpu really does change behavior",
      W.derived_allocation_for(wid, "vl7b")["mode"] == "max-ram")
check("a blank assign persists NOTHING (a derived default is not a choice)",
      not (store._load()[wid].get("spill_by_model") or {}).get("vl7b"))
check("derived max-gpu on a SMALL model also persists nothing",
      (store.assign_model(wid, "small") is not None
       and not (store._load()[wid].get("spill_by_model") or {}).get("small")))
check("default_allocation still encodes a derived max-gpu as {} (unpersisted); "
      "unknown size is the degrade-not-guess path that derives max-gpu since "
      "fits-whole now derives gpu-only (operator default order 2026-07-31)",
      default_allocation("gguf", None, 24 * GIB, 64 * GIB)["spill"] == {})

# ── 1. explicit max-gpu ROUND-TRIPS ─────────────────────────────────────────
store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
row = (store._load()[wid].get("spill_by_model") or {}).get("vl7b")
check("THE FIX: an explicit max-gpu is actually WRITTEN to the registry",
      row == {"alloc_mode": "max-gpu"})
check("it reads back as max-gpu (round-trip)", derive_alloc_mode(row) == "max-gpu")
check("it BEATS the derivation — the operator's choice is no longer overwritten "
      "by the max-ram default that broke the live case",
      derive_alloc_mode(row) != W.derived_allocation_for(wid, "vl7b")["mode"])

# ── 5. the WIRE is unchanged: the worker must not see the bookkeeping key ───
# Sending a literal HUGPY_ALLOC_MODE=max-gpu would SUPPRESS the worker's auto MoE
# split (slot_agent bails on any k37 alloc_mode), making an explicit max-gpu
# behave WORSE than a blank one. So the key is stripped at emission.
check("EMISSION: the worker receives {} — byte-identical to a blank max-gpu",
      store.spill_for(wid, "vl7b") == {})
check("the strip does not disturb a real mode contract (max-ram still rides)",
      (store.assign_model(wid, "small", spill={"alloc_mode": "max-ram"})
       is not None)
      and store.spill_for(wid, "small") == {"alloc_mode": "max-ram"})

# gated-DOWN worker: the downgrade target IS max-gpu, so behavior is identical.
old = store.register(name="oldbox", url="http://o:9100", pkg_version="0.1.150")
oid = old["id"]
with store._transaction() as wk:
    wk[oid]["gpus"] = [{"name": "RTX 4060", "memory_total": int(7.6 * GIB),
                        "memory_free": int(7.0 * GIB)}]
    wk[oid]["ram_total"] = 64 * GIB
store.assign_model(oid, "vl7b", spill={"alloc_mode": "max-gpu"})
check("GATED-DOWN worker (pkg 0.1.150) still gets max-gpu BEHAVIOUR ({})",
      store.spill_for(oid, "vl7b") == {})
check("...and its persisted row is untouched (it applies on update)",
      (store._load()[oid].get("spill_by_model") or {}).get("vl7b")
      == {"alloc_mode": "max-gpu"})
# The strip runs BEFORE the version gate, so the gate never fires on a key that
# carries no instruction — an old worker must not log a fictional
# "max-gpu downgraded to max-gpu" note.
check("the version gate is not engaged by the stripped key (no fictional "
      "downgrade note)",
      gate_spill_for_worker({}, "0.1.150", "oldbox") == ({}, None))

# ── 2. {} still CLEARS ──────────────────────────────────────────────────────
store.assign_model(wid, "vl7b", spill={})
check("{} still CLEARS the override (the console's '↺ Auto — derived')",
      not (store._load()[wid].get("spill_by_model") or {}).get("vl7b"))
check("after the clear the model tracks the DERIVATION again (max-ram here)",
      W.derived_allocation_for(wid, "vl7b")["mode"] == "max-ram"
      and store.spill_for(wid, "vl7b") == {"alloc_mode": "max-ram"})
store.assign_model(wid, "small", spill={})
check("clearing a max-ram contract works the same way",
      not (store._load()[wid].get("spill_by_model") or {}).get("small"))

# ── 3. the manifest-orphan cleanup path's SAFETY ARGUMENT still holds ───────
# worker_routes (~1331) relaxes the manifest gate for a CLEAR of an already-
# designated key. Its safety rests on: an empty spill cannot write a contract.
store.assign_model(wid, "orphan", spill={"alloc_mode": "explicit",
                                         "gpu_mem_gib": 4.0})
check("orphan cleanup: a stale contract exists to be cleaned",
      (store._load()[wid].get("spill_by_model") or {}).get("orphan"))
store.assign_model(wid, "orphan", spill={})
check("orphan cleanup: the empty-spill clear still removes the row",
      not (store._load()[wid].get("spill_by_model") or {}).get("orphan"))
check("orphan cleanup: an empty spill remains STRUCTURALLY incapable of "
      "writing a contract (the relaxation's whole safety argument)",
      all(not (store._load()[wid].get("spill_by_model") or {}).get("phantom")
          for _ in [store.assign_model(wid, "phantom", spill={})]))

# ── 6. the response reflects what was PERSISTED ─────────────────────────────
from abstract_hugpy_dev.flask_app.app.routes import worker_routes as wr

store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
wv = store.get(wid)
res = wr._assign_allocation_result(wv, "vl7b", {"alloc_mode": "max-gpu"})
check("response: an explicit max-gpu reports persisted=True",
      res["persisted"] is True and res["mode"] == "max-gpu")
check("response: it is attributed to the OPERATOR, not the derivation",
      res["source"] == "operator")
check("response: honored=True — asked for max-gpu, got max-gpu",
      res["honored"] is True and res["requested_mode"] == "max-gpu")
check("response: the note says the choice will survive derivation changes",
      "persisted" in res["note"])

store.assign_model(wid, "vl7b", spill={})
res_clear = wr._assign_allocation_result(store.get(wid), "vl7b", {})
check("response: after a CLEAR it reports persisted=False / source=derived — "
      "the API no longer claims a write that did not happen",
      res_clear["persisted"] is False and res_clear["source"] == "derived")
check("response: a clear reports the DERIVED mode it now tracks (max-ram)",
      res_clear["mode"] == "max-ram")
check("response: a clear is 'honored' (no mode was requested)",
      res_clear["honored"] is True and res_clear["requested_mode"] is None)

# The dishonesty detector itself: if a write were ever silently dropped again,
# the response must say so rather than report success.
res_lie = wr._allocation_state(None, {"alloc_mode": "max-gpu"})
check("response: a DROPPED write is reported as NOT honored (this is what the "
      "old {'admission':'approved'} hid)",
      res_lie["honored"] is False and res_lie["persisted"] is False
      and "NOT applied" in res_lie["note"])

print(f"\nALL {ok} explicit-max-gpu checks passed")

# ── RESTORE the module-scope stubs (this file is a SCRIPT, not pytest funcs) ──
# The patches above are applied at IMPORT time and would otherwise outlive this
# file, poisoning every later test in the same pytest process: test_storage_
# budget_fifo's real-manifest check passes alone and fails whenever this file
# runs first, because _model_size_bytes still returns the stub's None for every
# real manifest key. A module-scoped fixture cannot help here — pytest collects
# no test functions from a script-style harness, so nothing would ever tear it
# down. Restoring at the end of the module body is the only hook that runs.
(W._model_size_bytes, W._model_engine, W._model_moe_detail) = _REAL
