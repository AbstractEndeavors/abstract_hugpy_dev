"""k37 Slice B — the FIVE allocation modes + honest rename.

Covers, per the approved plan (ALLOCATION-PLACEMENT-PLAN.md) and the operator
spec (allocation-modes-spec):
  * build_spill materializes all 5 flat modes onto the wire encoding
    (gpu-only -1 / ram-only "off" / max-gpu {} zero-knob / max-ram + explicit
    on the NEW version-gated keys) and accepts legacy names via the alias
    table (autofit/cpu-only/budget/bands — resolved, never emitted);
  * read-time DERIVATION matrix (n_gpu_layers -1 -> gpu-only, 0/"off" ->
    ram-only, budgets/bands -> explicit, unset -> max-gpu) — the migration is
    derivation, no file rewrite;
  * DEFAULT is max-gpu (a blank model fits-and-spills, never OOMs —
    defaults-are-promises);
  * leniency floor math: 100% target + 30% leniency -> 70/30 floor, asserted
    against flex.band_floor (whole = THE MODEL); degrade-within-band admits,
    bust past the floor refuses naming mode + floor;
  * engine gating: transformers may pick gpu-only/ram-only/max-gpu only;
    max-ram/explicit on a transformers key -> honest refusal naming the mode;
  * the version gate: a spill carrying the new keys is emitted verbatim to a
    >= 0.1.203 worker and downgraded to {} (max-gpu) + logged for an older one
    (WorkerStore.spill_for end-to-end).

Run:  venv/bin/python tests/test_alloc_modes.py
"""
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-alloc-modes-test-")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

GIB = 2 ** 30

# ── the vocabulary + alias table ─────────────────────────────────────────────
from abstract_hugpy_dev.managers.alloc_modes import (
    ALLOC_MODES, LEGACY_ALLOC_ALIASES, resolve_alloc_mode, derive_alloc_mode,
    mode_to_spill, normalize_spill, gate_spill_for_worker,
    worker_honors_mode_keys, MODE_MIN_PKG_VERSION, NONGGUF_ALLOWED_MODES)

check("the five flat modes, exactly",
      ALLOC_MODES == ("gpu-only", "ram-only", "max-gpu", "max-ram", "explicit"))
check("legacy autofit -> max-gpu (alias, logged)",
      resolve_alloc_mode("autofit") == ("max-gpu", True))
check("legacy cpu-only -> ram-only",
      resolve_alloc_mode("cpu-only") == ("ram-only", True))
check("legacy budget -> explicit",
      resolve_alloc_mode("budget") == ("explicit", True))
check("legacy bands -> explicit",
      resolve_alloc_mode("bands") == ("explicit", True))
check("canonical-first: the STRING 'max-gpu' is the NEW max-gpu, never the "
      "old -1 meaning", resolve_alloc_mode("max-gpu") == ("max-gpu", False))
check("the historical old-max-gpu(-1) mapping is recorded in the alias table "
      "(unreachable at runtime by design)",
      LEGACY_ALLOC_ALIASES["max-gpu"] == "gpu-only")
check("unknown mode resolves to None (caller degrades, never 500s)",
      resolve_alloc_mode("warp-drive") == (None, False))

# ── build_spill: all 5 modes + legacy aliases ────────────────────────────────
from abstract_hugpy_dev.chaos import assortment as A
from abstract_hugpy_dev.chaos.schema import ALLOC_MODES as CHAOS_MODES, SPILL_KEYS

check("chaos draws from the same five-mode vocabulary", CHAOS_MODES == ALLOC_MODES)
rng = random.Random(7)
check("gpu-only -> n_gpu_layers -1 (the old 'Max GPU' all-or-bust, honestly named)",
      A.build_spill("gpu-only", 50, 4.0, rng)["n_gpu_layers"] == -1)
check("ram-only -> n_gpu_layers 'off'",
      A.build_spill("ram-only", 50, 4.0, rng) == {"n_gpu_layers": "off"})
check("max-gpu -> {} (autofit, ZERO knobs — the cut-and-dry pick)",
      A.build_spill("max-gpu", 50, 4.0, rng) == {})
mr = A.build_spill("max-ram", 50, 4.0, rng)
check("max-ram rides the new alloc_mode key + ctx",
      mr["alloc_mode"] == "max-ram" and mr["ctx_pct"] == 50)
ex = A.build_spill("explicit", 75, 6.0, rng)
check("explicit carries target + leniency + priority band keys",
      ex["alloc_mode"] == "explicit" and ex["gpu_mem_gib"] == 6.0
      and 0 < ex["leniency_pct"] <= 100 and "priority" in ex
      and ex["ctx_pct"] == 75)
check("every built spill uses only recognised /assign keys",
      all(set(A.build_spill(m, 50, 4.0, random.Random(1))) <= SPILL_KEYS
          for m in ALLOC_MODES))
check("legacy 'autofit' still builds (resolved to max-gpu, {})",
      A.build_spill("autofit", 50, 4.0, rng) == {})
check("legacy 'cpu-only' still builds (resolved to ram-only)",
      A.build_spill("cpu-only", 50, 4.0, rng) == {"n_gpu_layers": "off"})
check("legacy 'budget' still builds (resolved to explicit)",
      A.build_spill("budget", 50, 4.0, random.Random(3))["alloc_mode"] == "explicit")
check("transformers models get exactly the three coarse modes",
      A.NONGGUF_MODES == ("gpu-only", "ram-only", "max-gpu") ==
      NONGGUF_ALLOWED_MODES)
check("gguf keeps the full five", A.modes_for("gguf") == ALLOC_MODES)

# ── mode_to_spill (the central materialization helper) ───────────────────────
check("mode_to_spill gpu-only", mode_to_spill("gpu-only") == {"n_gpu_layers": -1})
check("mode_to_spill ram-only", mode_to_spill("ram-only") == {"n_gpu_layers": "off"})
check("mode_to_spill max-gpu is {}", mode_to_spill("max-gpu") == {})
check("mode_to_spill max-ram", mode_to_spill("max-ram") == {"alloc_mode": "max-ram"})
check("mode_to_spill explicit carries knobs",
      mode_to_spill("explicit", gpu_mem_gib=8, leniency_pct=30, priority=1)
      == {"alloc_mode": "explicit", "gpu_mem_gib": 8.0, "leniency_pct": 30.0,
          "priority": 1})
check("mode_to_spill unknown degrades to max-gpu ({})",
      mode_to_spill("warp-drive") == {})

# ── normalize_spill: alias resolution + coarse-trio rewrite ─────────────────
check("normalize: alloc_mode 'autofit' rewrites to {} (never emitted back)",
      normalize_spill({"alloc_mode": "autofit"})[0] == {})
check("normalize: alloc_mode 'gpu-only' rewrites onto the legacy wire",
      normalize_spill({"alloc_mode": "gpu-only"})[0] == {"n_gpu_layers": -1})
check("normalize: alloc_mode 'cpu-only' -> ram-only wire",
      normalize_spill({"alloc_mode": "cpu-only"})[0] == {"n_gpu_layers": "off"})
check("normalize: max-ram keeps the key (version-gated pair only)",
      normalize_spill({"alloc_mode": "max-ram"})[0] == {"alloc_mode": "max-ram"})
s, note = normalize_spill({"alloc_mode": "bogus", "ctx_pct": 25})
check("normalize: unknown mode dropped with a note, rest of spill survives",
      s == {"ctx_pct": 25} and "bogus" in note)

# ── derivation matrix + default-is-max-gpu ───────────────────────────────────
check("derive: n_gpu_layers -1 -> gpu-only",
      derive_alloc_mode({"n_gpu_layers": -1}) == "gpu-only")
check("derive: n_gpu_layers 0 -> ram-only",
      derive_alloc_mode({"n_gpu_layers": 0}) == "ram-only")
check("derive: n_gpu_layers 'off' -> ram-only",
      derive_alloc_mode({"n_gpu_layers": "off"}) == "ram-only")
check("derive: unset -> max-gpu (DEFAULT: serves-and-spills, never OOMs)",
      derive_alloc_mode({}) == "max-gpu" == derive_alloc_mode(None))
check("derive: a positive layer count stays a max-gpu flavor",
      derive_alloc_mode({"n_gpu_layers": 17}) == "max-gpu")
check("derive: an explicit budget contract reads as explicit",
      derive_alloc_mode({"gpu_mem_gib": 6.0}) == "explicit")
check("derive: a band contract reads as explicit",
      derive_alloc_mode({"gpu_mem_gib": 6.0, "gpu_mem_gib_deviation_pct": 25})
      == "explicit")
check("derive: persisted alloc_mode wins over legacy knobs",
      derive_alloc_mode({"alloc_mode": "max-ram", "n_gpu_layers": -1}) == "max-ram")
check("derive: persisted LEGACY name resolves canonical",
      derive_alloc_mode({"alloc_mode": "autofit"}) == "max-gpu")

# ── persisted overrides: write-resolution + read-time derivation ─────────────
from abstract_hugpy_dev.managers.serve import overrides as OV
OV.set_override("m-legacy", {"alloc_mode": "cpu-only"})
check("set_override stores the CANONICAL name (legacy accepted, never stored)",
      OV.get_override("m-legacy").get("alloc_mode") == "ram-only")
OV.set_override("m-typo", {"alloc_mode": "warp-drive"})
check("set_override drops an unknown alloc_mode (degrade, never 500)",
      "alloc_mode" not in OV.get_override("m-typo"))
OV.set_override("m-explicit", {"alloc_mode": "explicit", "leniency_pct": "30",
                               "priority": "2", "priority_device": "ram"})
o = OV.get_override("m-explicit")
check("explicit knobs persist typed (leniency float, priority int, device)",
      o == {"alloc_mode": "explicit", "leniency_pct": 30.0, "priority": 2,
            "priority_device": "ram"})
OV.set_override("m-old-wire", {"n_gpu_layers": -1})
check("effective_alloc_mode derives gpu-only from a legacy -1 override",
      OV.effective_alloc_mode("m-old-wire") == "gpu-only")
check("effective_alloc_mode: blank model == max-gpu (the default promise)",
      OV.effective_alloc_mode("m-never-touched") == "max-gpu")

# ── leniency floor math vs the flex band engine ──────────────────────────────
from abstract_hugpy_dev.worker_agent.flex import (
    band_floor, leniency_floor_pct, plan_explicit_offload)

check("100% target + 30% leniency -> 70% floor",
      leniency_floor_pct(100, 30) == 70.0)
check("leniency floor never below 0", leniency_floor_pct(50, 80) == 0.0)
# The conversion IS the band engine's: whole = THE MODEL, deviation = leniency.
model = 10 * GIB
check("band_floor(model, 30%, whole=model) == 7 GiB (70/30 floor in bytes)",
      band_floor(model, 30.0, model) == 7 * GIB)

# degrade WITHIN the band: 8 GiB budget over a 7 GiB floor -> admit a partial
p = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                          vram_budget_bytes=8 * GIB, ram_free_bytes=64 * GIB,
                          mode="explicit", priority_device="gpu",
                          leniency_pct=30.0)
check("explicit degrades within the band (admit between floor and target)",
      p is not None and p.admit and 23 <= p.n_gpu_layers < 32)
check("degraded plan still prices the RAM remainder",
      p.ram_need_bytes > 0 and p.vram_need_bytes <= 8 * GIB)
# bust PAST the floor: 5 GiB budget under the 7 GiB floor -> honest refusal
p2 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                           vram_budget_bytes=5 * GIB, ram_free_bytes=64 * GIB,
                           mode="explicit", priority_device="gpu",
                           leniency_pct=30.0)
check("bust past the floor refuses", p2 is not None and not p2.admit)
check("the refusal names the MODE and the FLOOR",
      "explicit" in p2.reject_reason and "70%" in p2.reject_reason
      and "floor" in p2.reject_reason)
# full fit at target: budget >= model -> every layer (no needless degrade)
p3 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                           vram_budget_bytes=12 * GIB, ram_free_bytes=64 * GIB,
                           leniency_pct=30.0)
check("fits at target -> all layers (degrade only under pressure)",
      p3.admit and p3.n_gpu_layers == 32)
# zero leniency = a point band: under target -> bust (no undeclared tolerance)
p4 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                           vram_budget_bytes=8 * GIB, ram_free_bytes=64 * GIB,
                           leniency_pct=0.0)
check("no declared leniency -> exact target or bust", not p4.admit)

# max-ram = explicit(ram priority, 100% target, generous leniency): RAM first,
# ONLY the overflow to GPU; bust only when RAM+GPU can't satisfy.
p5 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                           vram_budget_bytes=8 * GIB,
                           ram_free_bytes=int(6.5 * GIB), mode="max-ram",
                           priority_device="ram", leniency_pct=100.0)
check("max-ram: overflow (not the RAM-resident share) rides the GPU",
      p5.admit and 0 < p5.n_gpu_layers < 32)
p6 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                           vram_budget_bytes=2 * GIB, ram_free_bytes=2 * GIB,
                           mode="max-ram", priority_device="ram",
                           leniency_pct=100.0)
check("max-ram busts honestly when RAM+GPU can't satisfy",
      not p6.admit and "max-ram" in p6.reject_reason
      and "can't satisfy" in p6.reject_reason)
p7 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                           vram_budget_bytes=8 * GIB, ram_free_bytes=64 * GIB,
                           mode="max-ram", priority_device="ram",
                           leniency_pct=100.0)
check("max-ram with roomy RAM -> pure RAM (0 GPU layers)",
      p7.admit and p7.n_gpu_layers == 0)

# ── worker loader semantics: maxram_gpu_layers is autofit INVERTED ───────────
from abstract_hugpy_dev.managers import spill as SP
with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
    f.write(b"x" * (1 << 20))                 # 1 MiB; layer count -> assumed 32
    gguf_path = f.name
check("maxram: whole model fits RAM -> 0 GPU layers",
      SP.maxram_gpu_layers(gguf_path, free_ram=4 * (1 << 20)) == 0)
n_over = SP.maxram_gpu_layers(gguf_path, free_ram=512 * 1024)
check("maxram: only the RAM overflow goes to GPU (inverted fill)",
      0 < n_over < 32)
os.environ["HUGPY_ALLOC_MODE"] = "max-ram"
os.environ["HUGPY_CPU_MEM_GIB"] = str(512 * 1024 / GIB)
check("gguf_gpu_layers honors HUGPY_ALLOC_MODE=max-ram",
      SP.gguf_gpu_layers(gguf_path) == n_over)
os.environ.pop("HUGPY_CPU_MEM_GIB", None)   # let the CPU budget be generous
mm = SP.transformers_max_memory(model_need_bytes=20 * GIB)
check("transformers max-ram: generous CPU + remainder GPU",
      mm is not None and mm["cpu"] != "0.00GiB")
mm2 = SP.transformers_max_memory()
check("transformers max-ram without a size: GPU budget 0 (never a silent "
      "GPU fill against a RAM priority)", mm2 is not None and mm2[0] == "0.00GiB")
for k in ("HUGPY_ALLOC_MODE", "HUGPY_CPU_MEM_GIB"):
    os.environ.pop(k, None)
os.unlink(gguf_path)

# ── the version gate (no dead knobs) ─────────────────────────────────────────
check(f"mode keys need >= {MODE_MIN_PKG_VERSION}",
      worker_honors_mode_keys(MODE_MIN_PKG_VERSION)
      and worker_honors_mode_keys("0.2.0")
      and not worker_honors_mode_keys("0.1.202")
      and not worker_honors_mode_keys(None)
      and not worker_honors_mode_keys("garbage"))
gated, note = gate_spill_for_worker({"alloc_mode": "max-ram"}, "0.1.202", "op")
check("old worker: max-ram downgrades to {} (max-gpu) with an honest note",
      gated == {} and "max-ram" in note and "0.1.202" in note)
gated2, note2 = gate_spill_for_worker({"alloc_mode": "explicit",
                                       "leniency_pct": 30}, "0.1.203", "ae")
check("new worker: mode spill emitted verbatim",
      gated2 == {"alloc_mode": "explicit", "leniency_pct": 30} and note2 is None)
gated3, note3 = gate_spill_for_worker({"n_gpu_layers": -1}, "0.1.150", "op")
check("legacy-only spill passes any version untouched (no gate)",
      gated3 == {"n_gpu_layers": -1} and note3 is None)

# end-to-end through the store seam central's relay actually reads
from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import WorkerStore
store = WorkerStore(path=os.path.join(os.environ["PROJECTS_HOME"], "wk.json"))
old = store.register(name="old-box", url="http://x:9100", pkg_version="0.1.202")
new = store.register(name="new-box", url="http://y:9100", pkg_version="0.1.203")
store.assign_model(old["id"], "m1", spill={"alloc_mode": "max-ram"})
store.assign_model(new["id"], "m1", spill={"alloc_mode": "max-ram"})
check("spill_for downgrades the OLD worker to {} (request-time, honest log)",
      store.spill_for(old["id"], "m1") == {})
check("spill_for emits the mode verbatim to the NEW worker",
      store.spill_for(new["id"], "m1") == {"alloc_mode": "max-ram"})
raw = (store._load().get(old["id"]) or {}).get("spill_by_model", {}).get("m1")
check("the PERSISTED contract is untouched (applies once the worker updates)",
      raw == {"alloc_mode": "max-ram"})

# ── engine gating at the /assign seam ────────────────────────────────────────
import importlib
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")

_orig_fw = wr._model_framework
try:
    wr._model_framework = lambda mk: ("transformers" if mk.startswith("tf")
                                      else "gguf")
    ok1, r1 = wr._alloc_spill_ok_for_engine({"alloc_mode": "max-ram"}, "tf-m")
    check("transformers + max-ram -> refused, naming the mode",
          ok1 is False and "max-ram" in r1 and "GGUF-only" in r1)
    ok2, r2 = wr._alloc_spill_ok_for_engine(
        {"alloc_mode": "explicit", "leniency_pct": 30}, "tf-m")
    check("transformers + explicit -> refused, naming the mode",
          ok2 is False and "explicit" in r2)
    check("transformers + the coarse trio -> allowed (engine-agnostic intent)",
          wr._alloc_spill_ok_for_engine({}, "tf-m") == (True, None)
          and wr._alloc_spill_ok_for_engine({"n_gpu_layers": -1}, "tf-m") == (True, None)
          and wr._alloc_spill_ok_for_engine({"n_gpu_layers": "off"}, "tf-m") == (True, None))
    check("gguf + max-ram/explicit -> allowed",
          wr._alloc_spill_ok_for_engine({"alloc_mode": "max-ram"}, "g-m") == (True, None)
          and wr._alloc_spill_ok_for_engine(
              {"alloc_mode": "explicit", "leniency_pct": 10}, "g-m") == (True, None))
finally:
    wr._model_framework = _orig_fw

# validation seam accepts + normalizes the new keys (bulk path)
clean, reason = wr._validate_alloc_spill({"alloc_mode": "autofit"})
check("bulk validate: legacy name accepted, resolved, never written back",
      reason is None and clean == {})
clean2, reason2 = wr._validate_alloc_spill({"alloc_mode": "max-ram", "ctx_pct": 50})
check("bulk validate: max-ram survives normalization",
      reason2 is None and clean2 == {"alloc_mode": "max-ram", "ctx_pct": 50})
check("bulk validate: unknown alloc_mode rejected at the door with the list",
      wr._validate_alloc_spill({"alloc_mode": "warp"})[1] is not None)
check("bulk validate: leniency_pct range-checked",
      wr._validate_alloc_spill({"leniency_pct": 130})[1] is not None
      and wr._validate_alloc_spill({"alloc_mode": "explicit",
                                    "leniency_pct": 30})[1] is None)
check("bulk validate: priority_device gpu|ram only",
      wr._validate_alloc_spill({"priority_device": "tpu"})[1] is not None)

# labels use the honest flat names
check("label: {} == max-gpu", wr._alloc_label({}) == "max-gpu")
check("label: -1 == gpu-only", wr._alloc_label({"n_gpu_layers": -1}) == "gpu-only")
check("label: off == ram-only", wr._alloc_label({"n_gpu_layers": "off"}) == "ram-only")
check("label: mode spill names the mode",
      wr._alloc_label({"alloc_mode": "max-ram"}) == "max-ram")

print(f"\nALL {ok} alloc-mode checks passed")
