"""chaos runner: end-to-end trial behaviour with an in-memory fleet —
back-off on foreign jobs, health-degraded stop, predicted-infeasible skip, the
per-worker sequential gate, a full happy-path trial with verified restore, and
observation append + schema validity.

Run:  venv/bin/python tests/test_chaos_backoff.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstract_hugpy_dev.chaos.runner import ChaosRunner, WorkerGate, load_operator_token
from abstract_hugpy_dev.chaos.schema import validate_observation
from chaos_fakes import FakeClient, GIB

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

def mk_runner(client, seed=7, out=None):
    return ChaosRunner(client, seed=seed, rounds=1, budget_minutes=45,
                       out_dir=out or tempfile.mkdtemp(prefix="chaos-test-"),
                       max_new_tokens=8, chat_ceiling_s=5, settle_s=0)

# ── per-worker sequential gate ──────────────────────────────────────────────
g = WorkerGate()
check("gate claims free workers", g.claim(["ae", "computron"]) is True)
check("gate refuses a second concurrent claim on a busy worker",
      g.claim(["ae"]) is False)
g.release(["ae", "computron"])
check("gate re-claims after release", g.claim(["ae"]) is True)

# ── foreign-jobs back-off ───────────────────────────────────────────────────
foreign = {"jobs": [{"id": "v1-realtraffic", "status": "active"}],
           "counts": {"active": 1}}
c = FakeClient(jobs=foreign)
r = mk_runner(c)
obs = r.run_trial(0)
check("foreign active job -> back-off skip",
      obs["kind"] == "skip" and obs["skip_reason"] == "back-off-foreign-jobs")
check("back-off flag set", obs["back_off"] is True)
check("back-off did NOT fire a generation", c.chat_calls == [])
check("back-off did NOT write any spill", c.assign_calls == [])

# a chaos-owned active job is NOT treated as foreign
own = {"jobs": [{"id": f"{r.run_id}-r0", "status": "active"}]}
check("own chaos job not counted as foreign traffic",
      r._foreign_active(own) is False)

# ── health-degraded skip ────────────────────────────────────────────────────
c_bad = FakeClient(health_code=503)
r_bad = mk_runner(c_bad)
obs_h = r_bad.run_trial(0)
check("health != 200 -> health-degraded skip",
      obs_h["kind"] == "skip" and obs_h["skip_reason"] == "health-degraded")
check("no fire on degraded health", c_bad.chat_calls == [])

# ── predicted-infeasible skip (huge-gguf) — force the draw ──────────────────
class OnlyHuge(FakeClient):
    def models(self):
        return [m for m in super().models() if m["model_key"] == "huge-gguf"]
c_huge = OnlyHuge()
r_huge = mk_runner(c_huge)
obs_inf = r_huge.run_trial(0)
check("400GiB model -> predicted-infeasible skip",
      obs_inf["kind"] == "skip" and obs_inf["skip_reason"] == "predicted-infeasible")
check("predicted-infeasible did NOT fire", c_huge.chat_calls == [])
check("predicted-infeasible did NOT leave a spill written", c_huge.assign_calls == [])
check("infeasible observation carries the predicted need + reason",
      obs_inf["predicted"]["need_bytes"] and obs_inf["predicted"]["infeasible_reason"])

# ── happy path: apply -> fire -> measure -> restore (verified) ──────────────
class OnlySmall(FakeClient):
    def models(self):
        return [m for m in super().models() if m["model_key"] == "small-gguf"]
alloc_row = {"kind": "slot", "vram_bytes": 2 * GIB, "rss_bytes": 3 * GIB,
             "n_gpu_layers": -1, "total_layers": 29, "ctx": 16384,
             "serving": True}
c_ok = OnlySmall(chat_terminal={"outcome": "done", "served_worker": "computron",
                                "error": None, "finish_reason": "stop",
                                "ttft_s": 0.3, "load_duration_s": 1.1,
                                "wall_s": 1.8, "tokens": 2, "stages": []},
                 materialize_alloc={"computron": alloc_row})
out_dir = tempfile.mkdtemp(prefix="chaos-happy-")
r_ok = mk_runner(c_ok, seed=1, out=out_dir)
# capture pre-trial spill for the restore-diff proof
pre = {w["name"]: dict(w.get("spill_by_model", {})) for w in c_ok.workers()}
obs_t = r_ok.run_trial(0)
post = {w["name"]: dict(w.get("spill_by_model", {})) for w in c_ok.workers()}
check("happy path is a trial (not a skip)", obs_t["kind"] == "trial")
check("happy path fired exactly one generation", len(c_ok.chat_calls) == 1)
check("measured captured the real allocation vram_bytes",
      obs_t["measured"]["allocation"]["vram_bytes"] == 2 * GIB)
check("verdict inferred proceed", obs_t["measured"]["admission"]["verdict"] == "proceed")
check("restore verified ok", obs_t["restore"]["ok"] is True)
check("spill state fully restored after the trial (diff before==after)",
      pre == post)
check("observation validates clean against the schema",
      validate_observation(obs_t) == [])

# ── the observation was appended to the JSONL ───────────────────────────────
obs_file = Path(out_dir) / "observations.jsonl"
check("observations.jsonl written", obs_file.is_file())
lines = obs_file.read_text().strip().splitlines()
check("exactly one observation line appended", len(lines) == 1)
parsed = json.loads(lines[0])
check("appended line round-trips as valid JSON + schema",
      validate_observation(parsed) == [])
# manifest written
manifest = Path(out_dir) / "runs" / f"{r_ok.run_id}.json"
# run() writes the manifest; run_trial alone doesn't — write one to prove it
r_ok._write_manifest("test", 1.0, 2.0,
                     {"n_models_total": 1, "workers": []})
check("run manifest written", manifest.is_file())

# ── operator-token resolution precedence ────────────────────────────────────
tf = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
tf.write('HUGPY_OPERATOR_TOKEN="tok-from-file"\n')
tf.close()
check("explicit token wins",
      load_operator_token("explicit-tok", tf.name) == "explicit-tok")
check("env-file token read when no explicit/env",
      load_operator_token(None, tf.name) == "tok-from-file")

print(f"\nALL {ok} runner/back-off checks passed")
