"""chaos schema: observation completeness + verdict inference + verbatim
refusal capture (the predicted-vs-measured contract the learner joins on).

Run:  venv/bin/python tests/test_chaos_schema.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstract_hugpy_dev.chaos import observe
from abstract_hugpy_dev.chaos.schema import (
    blank_observation, validate_observation, SCHEMA_VERSION,
    REQUIRED_TOP_KEYS, REQUIRED_MEASURED_KEYS, REQUIRED_ADMISSION_KEYS)
from chaos_fakes import GIB

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

# ── blank observation is schema-complete ────────────────────────────────────
obs = blank_observation()
check("blank observation validates clean", validate_observation(obs) == [])
check("blank carries the schema version", obs["schema_version"] == SCHEMA_VERSION)
for k in REQUIRED_TOP_KEYS:
    check(f"blank has top key {k}", k in obs)
for k in REQUIRED_MEASURED_KEYS:
    check(f"blank has measured key {k}", k in obs["measured"])
for k in REQUIRED_ADMISSION_KEYS:
    check(f"blank has admission key {k}", k in obs["measured"]["admission"])

# ── validator catches a dropped key ─────────────────────────────────────────
broken = blank_observation()
del broken["measured"]["allocation"]
check("validator flags a missing measured key",
      any("allocation" in p for p in validate_observation(broken)))
broken2 = blank_observation()
broken2["kind"] = "skip"          # skip with no reason is invalid
check("validator flags skip without a reason",
      any("skip_reason" in p for p in validate_observation(broken2)))
broken3 = blank_observation()
broken3["skip_reason"] = "made-up-reason"
check("validator flags an unknown skip_reason",
      any("unknown skip_reason" in p for p in validate_observation(broken3)))

# ── verdict inference from the measured allocation ──────────────────────────
def measured(term, workers, model_key="small-gguf", ev_before=None, jobs=None):
    return observe.build_measured(term, workers, model_key,
                                  ev_before or {}, jobs)

# proceed: full GPU slot, no evictions
w_proceed = [{"name": "ae", "vram_evictions": 3, "vram_total": 24 * GIB,
              "gpus": [{"memory_free": 6 * GIB}],
              "allocations": [{"model_key": "small-gguf", "kind": "slot",
                               "vram_bytes": 2 * GIB, "rss_bytes": 3 * GIB,
                               "n_gpu_layers": -1, "total_layers": 29,
                               "ctx": 16384, "serving": True}]}]
m = measured({"outcome": "done", "served_worker": "ae"}, w_proceed,
             ev_before={"ae": 3})
check("full GPU slot -> verdict proceed", m["admission"]["verdict"] == "proceed")
check("measured captures real vram_bytes from the serving contract",
      m["allocation"]["vram_bytes"] == 2 * GIB)
check("proceed reports zero eviction delta",
      m["admission"]["vram_evictions_delta"] == 0)

# partial: 12/29 layers on GPU
w_partial = [{"name": "ae", "vram_evictions": 3,
              "allocations": [{"model_key": "small-gguf", "kind": "slot",
                               "vram_bytes": GIB, "rss_bytes": 5 * GIB,
                               "n_gpu_layers": 12, "total_layers": 29,
                               "ctx": 8192}]}]
mp = measured({"outcome": "done", "served_worker": "ae"}, w_partial,
              ev_before={"ae": 3})
check("12/29 layers -> verdict partial", mp["admission"]["verdict"] == "partial")

# cpu: ram allocation
w_cpu = [{"name": "op", "vram_evictions": 0,
          "allocations": [{"model_key": "small-gguf", "kind": "ram",
                           "rss_bytes": 4 * GIB}]}]
mc = measured({"outcome": "done", "served_worker": "op"}, w_cpu,
              ev_before={"op": 0})
check("ram allocation -> verdict cpu", mc["admission"]["verdict"] == "cpu")

# evicted: slot load after an eviction delta
w_evict = [{"name": "ae", "vram_evictions": 5,
            "allocations": [{"model_key": "small-gguf", "kind": "slot",
                             "vram_bytes": 2 * GIB, "n_gpu_layers": -1,
                             "total_layers": 29}]}]
me = measured({"outcome": "done", "served_worker": "ae"}, w_evict,
              ev_before={"ae": 3})
check("eviction delta>0 on full load -> verdict evicted",
      me["admission"]["verdict"] == "evicted"
      and me["admission"]["vram_evictions_delta"] == 2)

# refuse: verbatim reason dict surfaces via last_load_error
refusal = {"state": "refused", "model_key": "huge-gguf",
           "needs_bytes": 400 * GIB, "needs_weights_bytes": 380 * GIB,
           "needs_kv_bytes": 20 * GIB, "ctx_pct": 50,
           "partial_offload_considered": {"admit": False,
                                          "reject_reason": "CPU remainder OOM"},
           "protected": [{"model_key": "sd-turbo", "why": "actively replying"}],
           "evicted": []}
w_refuse = [{"name": "ae", "vram_evictions": 3, "last_load_error": refusal,
             "allocations": []}]
mr = measured({"outcome": "refused", "served_worker": "ae",
               "error": "won't fit on GPU: needs 400.0G"},
              w_refuse, ev_before={"ae": 3})
check("refusal -> verdict refuse", mr["admission"]["verdict"] == "refuse")
check("refusal_reason captured VERBATIM (needs split preserved)",
      mr["admission"]["refusal_reason"]["needs_weights_bytes"] == 380 * GIB
      and mr["admission"]["refusal_reason"]["needs_kv_bytes"] == 20 * GIB)
check("partial_offload_considered surfaced from the refusal",
      mr["admission"]["partial_offload_considered"]["reject_reason"]
      == "CPU remainder OOM")
check("verbatim error text retained",
      "won't fit" in mr["error"])

# refusal reason embedded as JSON inside an error string is still parsed
import json as _json
err_str = "load failed: " + _json.dumps({"state": "refused",
                                         "needs_bytes": 123})
w_err = [{"name": "ae", "allocations": [], "last_load_error": None}]
mj = measured({"outcome": "error", "served_worker": "ae", "error": err_str},
              w_err, ev_before={})
check("refusal JSON embedded in an error string is parsed",
      mj["admission"]["refusal_reason"]["needs_bytes"] == 123)

# served_worker falls back to the job row when the stream didn't name one
mfb = measured({"outcome": "done", "served_worker": None}, w_proceed,
               ev_before={"ae": 3},
               jobs={"jobs": [{"id": "x", "worker": "ae", "status": "done"}]})
check("served_worker falls back to the job row",
      mfb["served_worker"] == "ae")

# ── predicted side prices from meta and pairs with measured ─────────────────
from chaos_fakes import FakeClient
c = FakeClient()
combo = {"model_key": "small-gguf", "framework": "gguf",
         "effective_bytes": 2 * GIB, "alloc_mode": "budget",
         "spill": {"gpu_mem_gib": 4.0, "ctx_pct": 50}, "ctx_pct": 50,
         "target_workers": ["computron", "ae"]}
pred = observe.build_predicted(c, combo, c.workers())
check("predicted need_bytes priced from meta", isinstance(pred["need_bytes"], int))
check("predicted splits weights vs kv",
      pred["needs_weights_bytes"] == 2 * GIB and pred["needs_kv_bytes"] > 0)
check("predicted per-worker advice populated with offload advice",
      pred["per_worker"]["computron"]["advice"] is not None
      and "n_gpu_layers" in pred["per_worker"]["computron"]["advice"])
check("predicted feasibility computed", pred["feasible"] is True)

# a full skip observation validates clean
skip = blank_observation()
skip.update({"run_id": "r", "trial_id": "t", "seed": 1, "round": 0,
             "ts_start": 1.0, "ts_end": 2.0, "duration_s": 1.0,
             "kind": "skip", "skip_reason": "predicted-infeasible"})
check("a filled skip observation validates clean",
      validate_observation(skip) == [])

print(f"\nALL {ok} schema checks passed")
