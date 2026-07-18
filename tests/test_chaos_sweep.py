"""chaos sweep (k7): the sweep-point math (grid / ceiling / budget clamp / cliff
detection), the top-N selection + ranking evidence, and the snapshot→restore
zero-drift discipline of a whole-model sweep on the in-memory fake fleet.

Run:  venv/bin/python tests/test_chaos_sweep.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstract_hugpy_dev.chaos import sweep
from abstract_hugpy_dev.chaos.schema import validate_observation
from chaos_fakes import FakeClient, GIB

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ── median (median of 2 == mean; robust to Nones) ────────────────────────────
check("median of one value", sweep.median([12.0]) == 12.0)
check("median of two == their mean", sweep.median([10.0, 20.0]) == 15.0)
check("median of three", sweep.median([5.0, 100.0, 9.0]) == 9.0)
check("median ignores Nones", sweep.median([None, 8.0, None]) == 8.0)
check("median of empty is None", sweep.median([]) is None)

# ── ceiling_pct: fits -> 100; oversize -> card share ─────────────────────────
check("ceiling 100 when model fully fits", sweep.ceiling_pct(2.3, 5.0) == 100)
check("ceiling 100 when cap == need", sweep.ceiling_pct(5.0, 5.0) == 100)
check("ceiling ~35 for a 45GiB model on a 16GiB safe cap",
      sweep.ceiling_pct(45.0, 16.0) == int(100 * 16 / 45))

# ── sweep_points: fine / coarse / oversize scaling ───────────────────────────
fine = sweep.sweep_points(3.0, 6.0, big_model_gib=20.0)
check("fits model uses the full fine grid", fine == [100, 85, 70, 55, 40, 25])
coarse = sweep.sweep_points(30.0, 40.0, big_model_gib=20.0)
check("big model that fits uses the coarse grid", coarse == [100, 70, 40])
overs = sweep.sweep_points(45.0, 16.0, big_model_gib=20.0)
check("oversize model sweeps DOWN from the achievable ceiling (multi-point)",
      len(overs) >= 2 and overs[0] == sweep.ceiling_pct(45.0, 16.0)
      and overs == sorted(overs, reverse=True))
check("oversize points are all <= ceiling and strictly descending & distinct",
      overs[0] <= sweep.ceiling_pct(45.0, 16.0)
      and all(a > b for a, b in zip(overs, overs[1:])))

# ── point_budget_gib: pct of need, clamped to safe cap ───────────────────────
b, clamped = sweep.point_budget_gib(70, 4.0, 10.0)
check("budget = pct*need when it fits the cap", b == 2.8 and clamped is False)
b2, clamped2 = sweep.point_budget_gib(100, 20.0, 12.0)
check("budget clamps to the safe cap (OOM guard)", b2 == 12.0 and clamped2 is True)

# ── detect_cliff: largest relative drop; None -> collapse ────────────────────
curve = [{"vram_share_pct": 100, "tokens_per_s": 40.0},
         {"vram_share_pct": 85, "tokens_per_s": 38.0},
         {"vram_share_pct": 70, "tokens_per_s": 36.0},
         {"vram_share_pct": 55, "tokens_per_s": 6.0},   # cliff here
         {"vram_share_pct": 40, "tokens_per_s": 5.0}]
cl = sweep.detect_cliff(curve)
check("cliff located at the 70->55 transition",
      cl["cliff"] and cl["from_pct"] == 70 and cl["to_pct"] == 55)
check("cliff relative drop ~0.83", abs(cl["rel_drop"] - (36 - 6) / 36) < 1e-3)
none_curve = [{"vram_share_pct": 100, "tokens_per_s": 30.0},
              {"vram_share_pct": 50, "tokens_per_s": None}]  # refused/CPU-fell
cl2 = sweep.detect_cliff(none_curve)
check("a None tok/s point is a full collapse (cliff)",
      cl2["cliff"] and cl2["to_pct"] == 50 and cl2["rel_drop"] == 1.0)
flat = [{"vram_share_pct": 100, "tokens_per_s": 20.0},
        {"vram_share_pct": 40, "tokens_per_s": 19.5}]
cl3 = sweep.detect_cliff(flat)
check("a flat curve reports only a tiny drop", cl3["rel_drop"] < 0.1)

# ── rank_targets: GPU-only, usage ranking, pinning, exclusions ───────────────
c = FakeClient()
sel = sweep.rank_targets(c.models(), c.workers(), top_n=10)
chosen_keys = [t["model_key"] for t in sel["chosen"]]
check("only GGUF chat models are chosen (small-gguf)", chosen_keys == ["small-gguf"])
sg = sel["chosen"][0]
check("small-gguf pinned to the card it was last picked on (computron 1000>100)",
      sg["worker"] == "computron")
check("chosen carries ranking evidence (rank + last_picked)",
      sg["rank"] == 1 and sg["last_picked"] == 1000.0)
excl = {e["model_key"]: e["reason"] for e in sel["excluded"]}
check("huge-gguf excluded: weights exceed the box hybrid",
      excl.get("huge-gguf") == "weights-exceed-hybrid")
check("tf-model excluded: not gguf", excl.get("tf-model") == "not-gguf")
check("op (CPU box, no vram_total) contributes no targets",
      all(t["worker"] != "op" for t in sel["chosen"]))

# usage ordering on a purpose-built two-model card
models2 = [
    {"model_key": "hot", "framework": "gguf", "effective_bytes": 3 * GIB,
     "size_bytes": 3 * GIB, "model_max_length": 8192,
     "primary_task": "text-generation", "tasks": ["text-generation"], "blocked": False},
    {"model_key": "cold", "framework": "gguf", "effective_bytes": 3 * GIB,
     "size_bytes": 3 * GIB, "model_max_length": 8192,
     "primary_task": "text-generation", "tasks": ["text-generation"], "blocked": False}]
workers2 = [{"id": "wid-ae", "name": "ae", "status": "online",
             "vram_total": 24 * GIB, "ram_total": 128 * GIB, "vram_free": 20 * GIB,
             "models": ["hot", "cold"], "loaded_models": [],
             "model_last_picked": {"hot": 5000.0, "cold": 1000.0},
             "spill_by_model": {}}]
sel2 = sweep.rank_targets(models2, workers2, top_n=10)
check("most-recently-picked model ranks first",
      [t["model_key"] for t in sel2["chosen"]] == ["hot", "cold"])

# ── whole-model sweep: snapshot -> points -> restore (zero drift) ────────────
tmp = tempfile.mkdtemp(prefix="sweep-test-")
c2 = FakeClient(materialize_alloc={"computron": {
    "kind": "slot", "vram_bytes": 2 * GIB, "rss_bytes": 3 * GIB,
    "n_gpu_layers": -1, "total_layers": 28, "ctx": 4096, "serving": False}})
runner = sweep.SweepRunner(
    c2, top_n=10, ctx_pct=50, budget_minutes=90, out_dir=tmp,
    max_new_tokens=8, warmup_tokens=4, timed_runs=2, chat_ceiling_s=30,
    headroom_gib=1.0, vram_safety_frac=1.0, assign_settle_s=0.0, settle_s=0.0,
    big_model_gib=20.0, worker_filter=None)
runner._started = sweep.time.time()
runner._budget_hit = False
target = {"model_key": "small-gguf", "framework": "gguf",
          "effective_bytes": 2 * GIB, "worker": "computron",
          "worker_id": "wid-comp", "vram_total": 8 * GIB, "ram_total": 16 * GIB,
          "last_picked": 1000.0, "rank": 1}
res = runner.sweep_model(target)
check("sweep fired at least one point", res["points_fired"] >= 1)
check("every emitted observation validates clean (schema-complete + sweep block)",
      all(validate_observation(o) == [] for o in runner.point_records))
fired = [o for o in runner.point_records if o.get("kind") != "skip"]
check("fired points carry a populated sweep block",
      fired and all(o["sweep"]["vram_share_pct"] is not None
                    and o["sweep"]["requested_gib"] is not None for o in fired))
check("measured offload read back onto the sweep block",
      fired[0]["sweep"]["actual_n_gpu_layers"] == -1
      and fired[0]["sweep"]["actual_gpu_pct"] == 100)
# the model was forced cold before each point AND once more at restore
check("model was unloaded (forced cold) at least once per point",
      len(c2.unload_calls) >= len(fired))
# RESTORE: original spill written back byte-identical, verified
w_after = {x["name"]: x for x in c2.workers()}
check("computron spill restored to its ORIGINAL value (n_gpu_layers:-1)",
      w_after["computron"]["spill_by_model"]["small-gguf"] == {"n_gpu_layers": -1})
check("restore ledger records a verified byte-identical restore",
      runner.restore_ledger and runner.restore_ledger[0]["ok"] is True)

# ── the observations file was written and is one JSON object per line ────────
import json as _json
obs_lines = Path(runner.obs_path).read_text().strip().splitlines()
check("one JSONL observation appended per emitted point",
      len(obs_lines) == len(runner.point_records))
check("each JSONL line is a valid chaos-obs with a sweep block",
      all(_json.loads(ln).get("sweep") is not None for ln in obs_lines))

print(f"\nALL {ok} sweep checks passed")
