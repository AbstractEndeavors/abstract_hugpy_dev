"""t28 load-and-learn — the calibration layer that closes the predict/measure loop.

Covers all four moving parts with the seams stubbed (no GPU, no live central):

  1. Central store + aggregation — median measured/predicted ratio, the spread
     gate, the sample floor, the clamp band, partial/refuse exclusion, and the
     HUGPY_CALIBRATION kill-switch.
  2. Worker capture — sample SHAPE (omit-when-None), verdict classification,
     dedup-per-residency + re-arm, the load-fail (refuse) sample, and the drain.
  3. Application — need-pricing consults the learned correction ELSE the static
     x1.15, defensive clamp, kill-switch.
  4. End-to-end — real store -> real aggregate -> adopt -> real _incoming_need_
     detail applies the learned number.
  5. Wire-landmine proof — HeartbeatRequest is additive-safe (extra ignored), and
     the heartbeat reply is consumed as a plain dict (unknown key tolerated).

Run: venv/bin/python tests/test_calibration.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Deterministic env for the store gates BEFORE importing it.
os.environ.pop("HUGPY_CALIBRATION", None)
os.environ.pop("HUGPY_CALIBRATION_MIN_SAMPLES", None)
os.environ.pop("HUGPY_CALIBRATION_MAX_SPREAD", None)
os.environ.pop("HUGPY_CALIBRATION_CLAMP_LO", None)
os.environ.pop("HUGPY_CALIBRATION_CLAMP_HI", None)

from abstract_hugpy_dev.comms.calibration import CalibrationStore  # noqa: E402
from abstract_hugpy_dev.worker_agent import agent as A             # noqa: E402

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


def _store():
    return CalibrationStore(os.path.join(tempfile.mkdtemp(prefix="calib-"), "c.db"))


def _sample(model="m", need=1000, vram=900, verdict="full", ok_=True,
            ngl=None, total_layers=None, device="cuda"):
    s = {"model_key": model, "engine": "gguf", "needs_weights_bytes": int(need * 0.9),
         "needs_kv_bytes": int(need * 0.1), "ctx_pct": 50, "need_total_bytes": need,
         "verdict": verdict, "vram_bytes": vram, "device": device, "ok": ok_,
         "ts": 1000.0}
    if ngl is not None:
        s["n_gpu_layers"] = ngl
    if total_layers is not None:
        s["total_layers"] = total_layers
    return s


# ── 1) central store + aggregation ──────────────────────────────────────────
print("\n[1] central store + aggregation")

st = _store()
for _ in range(3):
    st.record(None, _sample(need=1000, vram=900))          # ratio 0.9 x3
agg = st.aggregate("m")
check("usable_count counts only full/ok/measured rows", agg["usable_count"] == 3)
check("median_ratio == 0.9", agg["median_ratio"] == 0.9)
check("spread 0 for identical ratios", agg["spread"] == 0.0)
check("gated True at >= floor with tight spread", agg["gated"] is True)
check("correction == median (in band)", agg["correction"] == 0.9)

# sample floor: 2 samples < default floor 3 -> not gated, no correction
st2 = _store()
for _ in range(2):
    st2.record(None, _sample(need=1000, vram=950))
agg2 = st2.aggregate("m")
check("below floor -> gated False", agg2["gated"] is False)
check("below floor -> correction None (static stands)", agg2["correction"] is None)
check("below floor still reports median for the table", agg2["median_ratio"] == 0.95)

# spread gate: wildly inconsistent ratios -> not trusted
st3 = _store()
for v in (500, 1500, 500, 1500):                            # ratios .5/1.5 -> huge spread
    st3.record(None, _sample(need=1000, vram=v))
agg3 = st3.aggregate("m")
check("high spread -> gated False", agg3["gated"] is False and agg3["correction"] is None)
check("high spread measured (> max 0.35)", agg3["spread"] and agg3["spread"] > 0.35)

# clamp band: median below 0.8 clamps up to 0.8; above 1.5 clamps down to 1.5
st_lo = _store()
for _ in range(3):
    st_lo.record(None, _sample(need=1000, vram=500))        # ratio 0.5
check("median below band clamps to 0.8", st_lo.aggregate("m")["correction"] == 0.8)
st_hi = _store()
for _ in range(3):
    st_hi.record(None, _sample(need=1000, vram=2000))       # ratio 2.0
check("median above band clamps to 1.5", st_hi.aggregate("m")["correction"] == 1.5)

# partial / cpu / refuse rows are EXCLUDED from the ratio
st_x = _store()
for _ in range(3):
    st_x.record(None, _sample(need=1000, vram=900))         # full, usable
st_x.record(None, _sample(need=1000, vram=300, verdict="partial"))
st_x.record(None, _sample(need=1000, vram=0, verdict="cpu", device="cpu"))
st_x.record(None, _sample(need=1000, vram=None, verdict="refuse", ok_=False))
agg_x = st_x.aggregate("m")
check("partial/cpu/refuse excluded from usable", agg_x["usable_count"] == 3)
check("total sample_count counts ALL rows", agg_x["sample_count"] == 6)
check("correction unaffected by non-full rows", agg_x["correction"] == 0.9)

# corrections() publishes only gated models; correction_for
st_c = _store()
for _ in range(3):
    st_c.record(None, _sample(model="hot", need=1000, vram=1200))   # ratio 1.2
st_c.record(None, _sample(model="cold", need=1000, vram=900))       # 1 sample -> ungated
corr = st_c.corrections()
check("corrections() includes the gated model", corr.get("hot", {}).get("correction") == 1.2)
check("corrections() omits the ungated model", "cold" not in corr)
check("correction_for gated == 1.2", st_c.correction_for("hot") == 1.2)
check("correction_for ungated == None", st_c.correction_for("cold") is None)
check("table lists every model with samples", {r["model_key"] for r in st_c.table()} == {"hot", "cold"})

# kill-switch: corrections() inert when off, table still introspectable
os.environ["HUGPY_CALIBRATION"] = "off"
check("kill-switch: corrections() empty when off", st_c.corrections() == {})
check("kill-switch: correction_for None when off", st_c.correction_for("hot") is None)
check("kill-switch: table still shows what WOULD be learned",
      any(r["model_key"] == "hot" for r in st_c.table()))
os.environ.pop("HUGPY_CALIBRATION")

# env-tunable floor: raising the floor above the sample count un-gates
os.environ["HUGPY_CALIBRATION_MIN_SAMPLES"] = "5"
check("raising the floor un-gates a 3-sample model", st.aggregate("m")["gated"] is False)
os.environ.pop("HUGPY_CALIBRATION_MIN_SAMPLES")


# ── 2) worker capture ───────────────────────────────────────────────────────
print("\n[2] worker capture")

# stub the prediction so the sample is deterministic without a real model/GPU.
A._incoming_need_bytes = lambda mk: 1000
A._model_framework = lambda mk: "gguf"


def _reset_worker():
    with A._CALIB_LOCK:
        A._CALIB_BUFFER.clear()
        A._CALIB_SAMPLED.clear()
        A._CALIB_CORRECTIONS.clear()


_reset_worker()
row = {"model_key": "m", "device": "cuda", "n_gpu_layers": 32,
       "total_layers": 32, "vram_bytes": 900, "rss_bytes": 500}
s = A._build_calibration_success("m", row)
check("success sample carries base (uncorrected) prediction", s["need_total_bytes"] == 1000)
check("success sample verdict == full", s["verdict"] == "full")
check("success sample measured vram", s["vram_bytes"] == 900)
check("success sample ok True", s["ok"] is True)
check("omit-when-None: no null keys on the wire", all(v is not None for v in s.values()))

# verdict classification
check("verdict partial when ngl < total", A._calib_verdict("cuda", 10, 32) == "partial")
check("verdict cpu when ngl 0", A._calib_verdict("cuda", 0, 32) == "cpu")
check("verdict cpu when device cpu", A._calib_verdict("cpu", None, None) == "cpu")
check("verdict full when in-process cuda (ngl None)", A._calib_verdict("cuda", None, None) == "full")
check("verdict full at max-gpu (ngl -1)", A._calib_verdict("cuda", -1, 32) == "full")

# dedup per residency + re-arm on departure
_reset_worker()
A._collect_calibration_from_allocations([row])
check("first residency -> one sample buffered", len(A._CALIB_BUFFER) == 1)
A._collect_calibration_from_allocations([row])
check("same residency -> no duplicate sample", len(A._CALIB_BUFFER) == 1)
A._collect_calibration_from_allocations([{"model_key": "m", "vram_bytes": None}])
check("unmeasured row -> not sampled", len(A._CALIB_BUFFER) == 1)
A._collect_calibration_from_allocations([])                 # model left residency
A._collect_calibration_from_allocations([row])              # reload -> re-sampled
check("re-arm on departure -> reload re-samples", len(A._CALIB_BUFFER) == 2)

# load-fail (refuse) sample
_reset_worker()
A._record_calibration_refuse("m", {"weights": 900, "kv": 100, "base_total": 1000, "ctx_pct": 50})
check("refuse sample buffered", len(A._CALIB_BUFFER) == 1)
check("refuse sample ok False", A._CALIB_BUFFER[0]["ok"] is False)
check("refuse sample verdict refuse", A._CALIB_BUFFER[0]["verdict"] == "refuse")

# drain clears the buffer
drained = A._drain_calibration_samples()
check("drain returns the buffered samples", len(drained) == 1)
check("drain clears the buffer", len(A._CALIB_BUFFER) == 0)

# kill-switch stops capture
_reset_worker()
os.environ["HUGPY_CALIBRATION"] = "off"
A._collect_calibration_from_allocations([row])
A._record_calibration_refuse("m", {"base_total": 1000})
check("kill-switch: no capture when off", len(A._CALIB_BUFFER) == 0)
os.environ.pop("HUGPY_CALIBRATION")


# ── 3) application: consults-learned-ELSE-static ────────────────────────────
print("\n[3] application")

_reset_worker()
check("no correction adopted -> static (total == base)",
      A._incoming_need_detail("m")["total"] == 1000)
A._adopt_calibration({"calibration": {"m": {"correction": 1.2}}})
det = A._incoming_need_detail("m")
check("learned correction applied to total", det["total"] == 1200)
check("base_total stays the UNcorrected prediction", det["base_total"] == 1000)
check("detail carries the correction for provenance", det["calibration_correction"] == 1.2)
A._adopt_calibration({"calibration": {"m": {"correction": 5.0}}})
check("defensive re-clamp on the worker (5.0 -> 1.5)",
      A._incoming_need_detail("m")["total"] == 1500)
os.environ["HUGPY_CALIBRATION"] = "off"
check("kill-switch: correction not applied when off",
      A._incoming_need_detail("m")["total"] == 1000)
os.environ.pop("HUGPY_CALIBRATION")
# adopting an empty/absent reply clears prior corrections (older/off central)
A._adopt_calibration({"calibration": {"m": {"correction": 1.2}}})
A._adopt_calibration({})
check("absent reply key clears corrections -> static",
      A._incoming_need_detail("m")["total"] == 1000)


# ── 4) end-to-end: store -> aggregate -> adopt -> apply ─────────────────────
print("\n[4] end-to-end")

_reset_worker()
e2e = _store()
for _ in range(4):
    e2e.record("worker-1", _sample(model="m", need=1000, vram=1150))   # ratio 1.15
pub = e2e.corrections(["m"])
check("e2e: store publishes a gated correction", pub.get("m", {}).get("correction") == 1.15)
A._adopt_calibration({"calibration": pub})
check("e2e: worker adopts + applies the learned number",
      A._incoming_need_detail("m")["total"] == 1150)


# ── 5) wire-landmine proof ──────────────────────────────────────────────────
print("\n[5] wire-landmine proof")

from abstract_hugpy_dev.flask_app.app.routes.worker_routes import HeartbeatRequest  # noqa: E402
hb = HeartbeatRequest(**{"calibration_samples": [_sample()],
                         "some_future_field_old_central_never_saw": 123})
check("HeartbeatRequest accepts calibration_samples (additive)",
      hb.calibration_samples and len(hb.calibration_samples) == 1)
check("HeartbeatRequest IGNORES unknown fields (extra='ignore', not forbid)",
      not hasattr(hb, "some_future_field_old_central_never_saw"))
# a heartbeat reply is a plain dict the worker reads with .get() — an OLD worker
# ignores the calibration key; a reply WITHOUT it clears cleanly (no exception).
A._adopt_calibration({"limits": {}, "required_pkg_version": "0.1.191"})
check("reply without calibration key -> no error, corrections cleared",
      A._incoming_need_detail("m")["total"] == 1000)


print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
