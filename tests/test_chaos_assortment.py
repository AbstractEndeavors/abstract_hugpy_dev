"""chaos assortment: enumeration, blocked-skip, framework-gated modes,
seeded-draw determinism, and hybrid feasibility (predicted-infeasible).

Run:  venv/bin/python tests/test_chaos_assortment.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstract_hugpy_dev.chaos import assortment as A
from chaos_fakes import make_models, make_workers, GIB

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

models = make_models()
workers = make_workers()

# ── enumeration + blocked-skip ──────────────────────────────────────────────
enum = A.enumerate_assortment(models, workers)
check("blocked model excluded from servable",
      "blocked-model" not in [r["model_key"] for r in enum["models"]])
check("blocked model reported in blocked_excluded",
      enum["blocked_excluded"] == ["blocked-model"])
check("image-only (non-chat) excluded",
      "image-only" not in [r["model_key"] for r in enum["models"]])
check("n_servable counts only chat, non-blocked",
      enum["n_servable"] == 4)  # small-gguf, tf-model, huge-gguf, unassigned-gguf
check("unassigned-gguf is servable but NOT exercisable (no candidate worker)",
      any(r["model_key"] == "unassigned-gguf" and not r["exercisable"]
          for r in enum["models"]))
check("n_exercisable excludes the unassigned model",
      enum["n_exercisable"] == 3)

# ── framework-gated alloc modes ─────────────────────────────────────────────
sm = {m["model_key"]: m for m in A.servable_models(models)}
check("gguf model gets all 5 alloc modes",
      set(A.modes_for(sm["small-gguf"]["framework"])) == set(A.ALLOC_MODES))
check("transformers model gets the four non-explicit modes (max-ram opened for "
      "non-GGUF 2026-07-24; only explicit stays GGUF-only)",
      A.modes_for(sm["tf-model"]["framework"])
      == ("gpu-only", "ram-only", "max-gpu", "max-ram"))

# ── candidate workers = already-assigned online ─────────────────────────────
widx = A.worker_index(workers)
check("small-gguf candidates = both workers",
      A.candidate_workers("small-gguf", widx) == ["ae", "computron"])
check("huge-gguf candidate = ae only",
      A.candidate_workers("huge-gguf", widx) == ["ae"])
check("unassigned-gguf has no candidates",
      A.candidate_workers("unassigned-gguf", widx) == [])

# ── seeded-draw determinism ─────────────────────────────────────────────────
seq1 = [A.draw_combo(random.Random(999), models, workers) for _ in range(1)]
# same seed -> identical sequence
r_a, r_b = random.Random(42), random.Random(42)
draws_a = [A.draw_combo(r_a, models, workers) for _ in range(20)]
draws_b = [A.draw_combo(r_b, models, workers) for _ in range(20)]
check("same seed -> identical combo sequence (reproducible)",
      [d["model_key"] for d in draws_a] == [d["model_key"] for d in draws_b]
      and [d["spill"] for d in draws_a] == [d["spill"] for d in draws_b])
r_c = random.Random(43)
draws_c = [A.draw_combo(r_c, models, workers) for _ in range(20)]
check("different seed -> different sequence (chaotic)",
      [d["model_key"] for d in draws_a] != [d["model_key"] for d in draws_c])
check("every draw targets an assigned worker",
      all(d["target_workers"] and set(d["target_workers"]) <=
          set(A.candidate_workers(d["model_key"], widx)) for d in draws_a))
check("no draw ever selects the blocked model",
      all(d["model_key"] != "blocked-model" for d in draws_a + draws_c))
check("combo.ctx_pct matches the spill's ctx_pct (or None)",
      all(d["ctx_pct"] == (d["spill"].get("ctx_pct")) for d in draws_a))
check("ram-only draws carry no ctx_pct",
      all(d["ctx_pct"] is None for d in draws_a if d["alloc_mode"] == "ram-only"))

# ── spill construction stays within recognised keys ─────────────────────────
from abstract_hugpy_dev.chaos.schema import SPILL_KEYS
check("every drawn spill uses only recognised /assign keys",
      all(set(d["spill"]) <= SPILL_KEYS for d in draws_a + draws_c))

# ── hybrid feasibility (predicted-infeasible) ───────────────────────────────
# huge-gguf (400 GiB) exceeds ae's 24+128=152 GiB hybrid -> infeasible.
feas_huge = A.feasibility(400 * GIB, ["ae"], workers)
check("400GiB need on ae (152GiB hybrid) is predicted-infeasible",
      feas_huge["feasible"] is False and feas_huge["infeasible_reason"])
# small-gguf (2 GiB) fits ae's hybrid easily.
feas_small = A.feasibility(2 * GIB, ["ae", "computron"], workers)
check("2GiB need is feasible on the fleet",
      feas_small["feasible"] is True and feas_small["infeasible_reason"] is None)
# unknown need fails OPEN (feasible) — admission gate does the same.
feas_unknown = A.feasibility(None, ["ae"], workers)
check("unknown need fails open (feasible)", feas_unknown["feasible"] is True)
check("feasibility reports per-worker hybrid totals",
      feas_small["per_worker"]["ae"]["hybrid_total"] == (24 + 128) * GIB)

print(f"\nALL {ok} assortment checks passed")
