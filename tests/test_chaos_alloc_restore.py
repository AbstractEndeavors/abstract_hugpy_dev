"""chaos alloc: snapshot / apply / restore round-trip + verification, the
engine-gate (non-200) abort, and partial-apply restore.

Run:  venv/bin/python tests/test_chaos_alloc_restore.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstract_hugpy_dev.chaos import alloc
from abstract_hugpy_dev.chaos.assortment import worker_index
from chaos_fakes import FakeClient

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

# ── snapshot captures the CURRENT spill (incl. None==autofit) ────────────────
c = FakeClient()
widx = worker_index(c.workers())
snap = alloc.snapshot(c, "small-gguf", ["computron", "ae"], widx)
check("snapshot reads computron's existing spill",
      snap["computron"]["before"] == {"n_gpu_layers": -1})
check("snapshot reads ae's absent spill as None (autofit)",
      snap["ae"]["before"] is None)
check("snapshot carries worker ids for the write-back",
      snap["computron"]["worker_id"] == "wid-comp")

# ── apply changes the spill, restore puts the EXACT prior back + verifies ────
chaos_spill = {"gpu_mem_gib": 4.0, "ctx_pct": 50}
res = alloc.apply(c, "small-gguf", chaos_spill, snap)
check("apply reports ok", res["ok"] is True)
# state changed on both workers
w = {x["name"]: x for x in c.workers()}
check("computron spill changed to chaos spill",
      w["computron"]["spill_by_model"]["small-gguf"] == chaos_spill)
check("ae spill materialised to chaos spill",
      w["ae"]["spill_by_model"]["small-gguf"] == chaos_spill)

rest = alloc.restore(c, "small-gguf", snap)
check("restore ok (verified against fresh read)", rest["ok"] is True)
w2 = {x["name"]: x for x in c.workers()}
check("computron restored to prior n_gpu_layers:-1",
      w2["computron"]["spill_by_model"]["small-gguf"] == {"n_gpu_layers": -1})
check("ae restored to autofit (None before -> {} clears override)",
      "small-gguf" not in w2["ae"]["spill_by_model"])
check("restore per_worker matches flags all true",
      all(v["matches"] for v in rest["per_worker"].values()))

# ── engine gate: a gguf-only spill on a transformers model is refused (409) ──
c2 = FakeClient()
widx2 = worker_index(c2.workers())
snap2 = alloc.snapshot(c2, "tf-model", ["computron"], widx2)
bad = alloc.apply(c2, "tf-model", {"n_gpu_layers": "off"}, snap2)
check("apply not ok when engine gate refuses (409)", bad["ok"] is False)
check("apply surfaces the 409 status + verbatim error",
      bad["results"]["computron"]["status"] == 409
      and "not GGUF" in bad["results"]["computron"]["error"])
# nothing was written -> restore is a clean no-op that still verifies clean
rest2 = alloc.restore(c2, "tf-model", snap2)
check("restore after a refused apply verifies clean (no drift)",
      rest2["ok"] is True)
w3 = {x["name"]: x for x in c2.workers()}
check("tf-model spill unchanged (never had one)",
      "tf-model" not in w3["computron"].get("spill_by_model", {}))

# ── partial apply (one worker 404s) still restores the one that changed ──────
c3 = FakeClient()
widx3 = worker_index(c3.workers())
snap3 = alloc.snapshot(c3, "small-gguf", ["computron", "ae"], widx3)
# corrupt one snapshot's worker id to force a failed write on ae
snap3["ae"]["worker_id"] = "wid-does-not-exist"
res3 = alloc.apply(c3, "small-gguf", {"gpu_mem_gib": 2.0, "ctx_pct": 25}, snap3)
check("apply reports not-ok on a partial (one worker unresolvable)",
      res3["ok"] is False)
check("computron still received the chaos spill",
      c3.workers()[0]["spill_by_model"]["small-gguf"] == {"gpu_mem_gib": 2.0, "ctx_pct": 25})
rest3 = alloc.restore(c3, "small-gguf", snap3)
check("restore puts computron back even after a partial apply",
      {x["name"]: x for x in c3.workers()}["computron"]
      ["spill_by_model"]["small-gguf"] == {"n_gpu_layers": -1})

print(f"\nALL {ok} alloc/restore checks passed")
