"""k37 CAPABILITY-AWARE DEFAULT ALLOC MODE (operator ruling 2026-07-24).

Two halves of the operator ask:
  1. the BLANK default is derived per (model x worker) by FEASIBILITY, engine-
     aware: GGUF -> max-gpu always; a transformers model too big for the box's
     GPU but fitting RAM -> ram-only ON THAT WORKER (the only feasible option, so
     it IS the default, and it must SERVE — defaults-are-promises);
  2. the SELECTABLE set is bounded to what's feasible (feasible_modes) so an
     infeasible mode is never offered — enforced at /assign (and the bulk path)
     with an honest numbers-naming 409/skip, surfaced on the serving row.

Covers:
  * feasible_default_mode matrix (gguf any-size, transformers 68/24/124 ->
    ram-only, 5/24 -> max-gpu, 200/24/124 -> max-gpu (fits neither), unknown ->
    max-gpu);
  * feasible_modes matrix per engine (incl. the 68/24/124 case -> exactly
    (ram-only, max-ram); max-gpu eliminated for oversized transformers; unknown
    data -> coarse trio + max-ram feasible; max-ram is engine-AGNOSTIC as of
    2026-07-24, only explicit stays engine-gated for non-GGUF);
  * an explicit persisted mode wins over derivation;
  * the emission seam: spill_for emits {"n_gpu_layers":"off"} for the oversized-
    transformers blank case, {} for the GGUF/fitting blank case, and leaves a
    persisted spill untouched;
  * the /assign feasibility gate refuses an infeasible mode (max-gpu on the
    68/24 case) with a numbers-naming reason; the bulk path skips it.

Run:  venv/bin/python tests/test_alloc_defaults.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-alloc-defaults-test-")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

GIB = 2 ** 30

from abstract_hugpy_dev.managers.alloc_modes import (
    feasible_default_mode, feasible_modes, is_gguf_engine, ALLOC_MODES,
    derive_alloc_mode, _GPU_FIT_HEADROOM)

# ── engine classification ────────────────────────────────────────────────────
check("gguf / llama_cpp are the GGUF family",
      is_gguf_engine("gguf") and is_gguf_engine("llama_cpp")
      and is_gguf_engine("GGUF"))
check("transformers / comfy / None are NOT GGUF",
      not is_gguf_engine("transformers") and not is_gguf_engine("comfy")
      and not is_gguf_engine(None))

# ── feasible_default_mode matrix ─────────────────────────────────────────────
check("gguf any size -> max-gpu (partial offload universal), even oversized",
      feasible_default_mode("gguf", 200 * GIB, 24 * GIB, 124 * GIB) == "max-gpu"
      and feasible_default_mode("llama_cpp", 500 * GIB, 8 * GIB, 16 * GIB)
      == "max-gpu")
# the operator's headline case: 68 GB transformers, 24 GB GPU, 124 GB RAM
check("transformers 68GB / 24GB GPU / 124GB RAM -> ram-only (only feasible)",
      feasible_default_mode("transformers", 68 * GIB, 24 * GIB, 124 * GIB)
      == "ram-only")
check("transformers 5GB / 24GB GPU -> max-gpu (plausibly fits GPU)",
      feasible_default_mode("transformers", 5 * GIB, 24 * GIB, 124 * GIB)
      == "max-gpu")
check("transformers 200GB / 24GB GPU / 124GB RAM -> max-gpu (fits NEITHER, "
      "honest refusal downstream — no invented fourth state)",
      feasible_default_mode("transformers", 200 * GIB, 24 * GIB, 124 * GIB)
      == "max-gpu")
# headroom boundary: exactly at 0.9x GPU still fits; just over falls to ram-only
check("transformers at exactly 0.9x GPU total -> max-gpu (within headroom)",
      feasible_default_mode("transformers", int(_GPU_FIT_HEADROOM * 24 * GIB),
                            24 * GIB, 124 * GIB) == "max-gpu")
check("transformers just over 0.9x GPU total (but fits RAM) -> ram-only",
      feasible_default_mode("transformers", int(_GPU_FIT_HEADROOM * 24 * GIB) + 1,
                            24 * GIB, 124 * GIB) == "ram-only")
# degrade-not-guess: any missing input -> max-gpu (today's behavior)
check("unknown size -> max-gpu (never derive from a guess)",
      feasible_default_mode("transformers", None, 24 * GIB, 124 * GIB) == "max-gpu")
check("unknown gpu total -> max-gpu",
      feasible_default_mode("transformers", 68 * GIB, None, 124 * GIB) == "max-gpu")
check("oversized-for-GPU but unknown RAM -> max-gpu (can't justify ram-only "
      "on a guess)",
      feasible_default_mode("transformers", 68 * GIB, 24 * GIB, None) == "max-gpu")

# ── feasible_modes matrix ────────────────────────────────────────────────────
# gguf: max-gpu/max-ram/explicit universal; gpu-only bounded by GPU fit;
# ram-only bounded by RAM fit.
fg = feasible_modes("gguf", 68 * GIB, 24 * GIB, 124 * GIB)
check("gguf 68/24/124: max-gpu, max-ram, explicit, ram-only feasible; gpu-only "
      "eliminated (won't fit GPU alone)",
      "max-gpu" in fg and "max-ram" in fg and "explicit" in fg
      and "ram-only" in fg and "gpu-only" not in fg)
fg2 = feasible_modes("gguf", 5 * GIB, 24 * GIB, 124 * GIB)
check("gguf 5/24: every mode feasible (fits GPU, RAM, and combined)",
      fg2 == ALLOC_MODES)
# THE headline: transformers 68/24/124 -> (ram-only, max-ram). max-ram is now
# engine-agnostic (2026-07-24): 68 <= 24+124 GiB combined, so it is feasible for
# a non-GGUF model too. explicit stays engine-gated off (banded leniency floor
# has no transformers analogue).
ft = feasible_modes("transformers", 68 * GIB, 24 * GIB, 124 * GIB)
check("transformers 68/24/124 -> feasible EXACTLY (ram-only, max-ram) — gpu-only "
      "& max-gpu eliminated (won't fit GPU alone); max-ram fits combined; "
      "explicit engine-gated off",
      ft == ("ram-only", "max-ram"))
check("=> max-gpu is NOT an offered mode for the 68/24 transformers case",
      "max-gpu" not in ft)
check("=> explicit is NEVER offered for a non-GGUF model (stays engine-gated)",
      "explicit" not in ft)
ft2 = feasible_modes("transformers", 5 * GIB, 24 * GIB, 124 * GIB)
check("transformers 5/24: gpu-only, ram-only, max-gpu, max-ram feasible; "
      "explicit engine-gated off",
      set(ft2) == {"gpu-only", "ram-only", "max-gpu", "max-ram"})
# fits neither GPU nor RAM nor combined: transformers 200/24/124
ft3 = feasible_modes("transformers", 200 * GIB, 24 * GIB, 124 * GIB)
check("transformers 200/24/124 (fits neither, nor combined 148): nothing "
      "physically lands, so the set falls back to (max-gpu,) — honest refusal "
      "downstream, never empty",
      ft3 == ("max-gpu",))
# unknown data -> never eliminate on missing data. max-ram no longer engine-gated
# so it rides through on unknown size; only explicit is dropped.
check("unknown size -> coarse trio + max-ram feasible (fail-open); only explicit "
      "engine-gated off",
      feasible_modes("transformers", None, 24 * GIB, 124 * GIB) ==
      ("gpu-only", "ram-only", "max-gpu", "max-ram"))
check("unknown size on GGUF -> literally every mode",
      feasible_modes("gguf", None, None, None) == ALLOC_MODES)
check("unknown totals on transformers -> coarse trio + max-ram feasible",
      feasible_modes("transformers", 68 * GIB, None, None) ==
      ("gpu-only", "ram-only", "max-gpu", "max-ram"))
# explicit is ENGINE-GATED regardless of numbers (a capability fact); max-ram is
# NOT — it is offered for non-GGUF whenever the numbers fit.
check("non-GGUF never offers explicit (engine gate) even with room; max-ram IS "
      "offered when it fits combined",
      "explicit" not in feasible_modes("transformers", 1 * GIB, 24 * GIB, 124 * GIB)
      and "max-ram" in feasible_modes("transformers", 1 * GIB, 24 * GIB, 124 * GIB))
check("the default is the BEST feasible member (default in feasible set)",
      feasible_default_mode("transformers", 68 * GIB, 24 * GIB, 124 * GIB)
      in feasible_modes("transformers", 68 * GIB, 24 * GIB, 124 * GIB))

# ── explicit persisted mode wins over derivation ─────────────────────────────
check("a persisted alloc_mode is NOT blank -> derivation never overrides it",
      derive_alloc_mode({"alloc_mode": "gpu-only"}) == "gpu-only")

# ── the emission seam (spill_for) ────────────────────────────────────────────
from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers as W
from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import WorkerStore

# Stub central's engine + size resolution so the seam is deterministic without a
# real registry/manifest. "tf-big" = 68 GiB transformers; "tf-small" = 5 GiB
# transformers; "g-big" = 200 GiB gguf; "unk" = unknown size.
_SIZES = {"tf-big": 68 * GIB, "tf-small": 5 * GIB, "g-big": 200 * GIB, "unk": None}
_ENGINES = {"tf-big": "transformers", "tf-small": "transformers",
            "g-big": "gguf", "unk": "transformers"}
W._model_size_bytes = lambda mk: _SIZES.get(mk)
W._model_engine = lambda mk: _ENGINES.get(mk)

# Point the MODULE-GLOBAL store at an ISOLATED tmp registry so this test never
# touches the real workers.json (the default path sits next to the manifest) and
# is deterministic run-to-run. The wrappers (derived_default_for,
# feasible_modes_for, feasibility_context) and the route gate all read
# W.worker_store, so reassigning it here routes them at our isolated store.
W.worker_store = WorkerStore(
    path=os.path.join(os.environ["PROJECTS_HOME"], "wk.json"))
store = W.worker_store
# Isolate the assignment-memory sidecar too (durable hardware totals ride it) —
# its default path sits next to the real manifest.
_MEM_PATH = os.path.join(os.environ["PROJECTS_HOME"], "worker_assignments.json")
W._assign_memory_path = lambda: _MEM_PATH
# a 24 GiB GPU / 124 GiB RAM worker on a mode-honoring version
w = store.register(name="box24", url="http://z:9100", pkg_version="0.1.203")
# supply the totals the way a heartbeat does (raw record), via a transaction.
with store._transaction() as wk:
    wk[w["id"]]["gpus"] = [{"name": "RTX 3090", "memory_total": 24 * GIB,
                            "memory_free": 20 * GIB}]
    wk[w["id"]]["ram_total"] = 124 * GIB

for mk in ("tf-big", "tf-small", "g-big", "unk"):
    store.assign_model(w["id"], mk)          # blank spill (no alloc contract)

check("seam: oversized transformers blank default SERVES as ram-only "
      "({'n_gpu_layers':'off'})",
      store.spill_for(w["id"], "tf-big") == {"n_gpu_layers": "off"})
check("seam: fitting transformers blank default -> {} (max-gpu, unchanged)",
      store.spill_for(w["id"], "tf-small") == {})
check("seam: GGUF blank default -> {} (max-gpu always, any size)",
      store.spill_for(w["id"], "g-big") == {})
check("seam: unknown-size transformers blank default -> {} (max-gpu, degrade)",
      store.spill_for(w["id"], "unk") == {})

# an explicit persisted spill is left untouched (the blank derivation only ever
# fires when NOTHING placement-affecting is persisted).
store.assign_model(w["id"], "tf-big", spill={"n_gpu_layers": -1})
check("seam: a persisted placement spill wins — derivation never overrides it",
      store.spill_for(w["id"], "tf-big") == {"n_gpu_layers": -1})
store.assign_model(w["id"], "tf-big", spill={})   # back to blank
check("seam: clearing back to blank restores the ram-only feasible default",
      store.spill_for(w["id"], "tf-big") == {"n_gpu_layers": "off"})

# a worker with NO reported totals -> fail-open (max-gpu blank), even oversized
w2 = store.register(name="box-fresh", url="http://q:9100", pkg_version="0.1.203")
store.assign_model(w2["id"], "tf-big")
check("seam: a worker missing totals fails open to {} (max-gpu) — never a "
      "ram-only guess",
      store.spill_for(w2["id"], "tf-big") == {})

# module wrappers used by the surface
check("derived_default_for reads the RAW record -> ram-only for the 68/24 case",
      W.derived_default_for(w["id"], "tf-big") == "ram-only")
check("feasible_modes_for -> (ram-only, max-ram) for the 68/24 transformers case "
      "(max-ram opened for non-GGUF 2026-07-24; fits combined 148 GiB)",
      W.feasible_modes_for(w["id"], "tf-big") == ("ram-only", "max-ram"))
check("feasibility_context surfaces the raw numbers for an honest 409",
      W.feasibility_context(w["id"], "tf-big") ==
      {"engine": "transformers", "model_bytes": 68 * GIB,
       "gpu_total_bytes": 24 * GIB, "ram_total_bytes": 124 * GIB,
       # MoE (2026-07-24): the expert-split GPU need; None for non-MoE.
       "moe_split_gpu_bytes": None})

# ── the /assign feasibility gate + bulk skip ─────────────────────────────────
import importlib
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")

# max-gpu on the 68/24 transformers case is INFEASIBLE -> refused, numbers named
okf, reason = wr._alloc_mode_feasible_for_worker({}, w["id"], "tf-big")
check("gate: max-gpu (blank derive) on 68/24 transformers -> refused",
      okf is False)
check("gate: the refusal NAMES the mode, the numbers, and the feasible set",
      "max-gpu" in reason and "68.0GiB" in reason and "24.0GiB" in reason
      and "ram-only" in reason)
# the feasible pick is allowed
okf2, reason2 = wr._alloc_mode_feasible_for_worker(
    {"n_gpu_layers": "off"}, w["id"], "tf-big")
check("gate: ram-only on the same case -> allowed", okf2 is True and reason2 is None)
# a fitting model allows max-gpu
okf3, _ = wr._alloc_mode_feasible_for_worker({}, w["id"], "tf-small")
check("gate: max-gpu on a fitting transformers model -> allowed", okf3 is True)
# unknown worker / unknown data fails open (allow)
okf4, _ = wr._alloc_mode_feasible_for_worker({}, "no-such-worker", "tf-big")
check("gate: unknown worker fails open (allow — assign 404s later)", okf4 is True)
okf5, _ = wr._alloc_mode_feasible_for_worker({}, w["id"], "unk")
check("gate: unknown model size fails open (allow — never eliminate on a guess)",
      okf5 is True)

# ── durable hardware totals (operator addendum 2026-07-24) ───────────────────
# A box's GPU/RAM CAPACITY is a physical FACT, not session state. A re-register
# (or a transient empty gpu probe) must NOT erase totals central already learned,
# so feasibility keeps working across the re-register window.
GPUS_24 = [{"name": "RTX 3090", "memory_total": 24 * GIB, "memory_free": 20 * GIB}]

# fresh box: first register WITH totals learns them and persists them durably.
d1 = store.register(name="durable-box", url="http://d:9100",
                    pkg_version="0.1.203", gpus=GPUS_24, ram_total=124 * GIB)
did = d1["id"]
store.assign_model(did, "tf-big")            # designate the oversized transformers
check("durable: first register with totals -> feasibility resolves (ram-only, max-ram)",
      W.feasible_modes_for(did, "tf-big") == ("ram-only", "max-ram"))
rec = store._load()[did]
check("durable: register persisted the last-known GPU total as a durable fact",
      rec.get("gpu_total_bytes_known") == 24 * GIB
      and rec.get("ram_total_bytes_known") == 124 * GIB)

# RE-REGISTER with an EMPTY gpu probe (driver not ready) + no ram_total: the live
# gpus[] is wiped, but the durable fact must survive so feasibility still works.
store.register(name="durable-box", url="http://d:9100", pkg_version="0.1.203",
               gpus=[], ram_total=None)
rec2 = store._load()[did]
check("durable: an empty-probe re-register did NOT erase the durable totals",
      rec2.get("gpu_total_bytes_known") == 24 * GIB
      and rec2.get("ram_total_bytes_known") == 124 * GIB)
check("durable: live gpus[] IS now empty (the transient reading), proving the "
      "fallback — not stale live data — carries feasibility",
      not rec2.get("gpus"))
check("durable: feasibility STILL resolves (ram-only, max-ram) across the re-register window",
      W.feasible_modes_for(did, "tf-big") == ("ram-only", "max-ram"))
check("durable: the derived default STILL serves as ram-only across the window",
      W.derived_default_for(did, "tf-big") == "ram-only"
      and store.spill_for(did, "tf-big") == {"n_gpu_layers": "off"})

# a beat that carries fresh totals confirms/updates them (advance path).
store.heartbeat(did, gpus=GPUS_24, ram_total=124 * GIB)
check("durable: a heartbeat re-confirms the totals (advance-only update)",
      store._load()[did].get("gpu_total_bytes_known") == 24 * GIB)

# survive a FULL registry loss: the durable totals ride the assignment-memory
# sidecar, so a returning worker id inherits them on a brand-new row.
_remembered = W._load_assign_memory().get(did) or {}
check("durable: the totals were snapshotted into the assignment-memory sidecar",
      _remembered.get("gpu_total_bytes_known") == 24 * GIB
      and _remembered.get("ram_total_bytes_known") == 124 * GIB)
store.remove(did)                            # registry row gone (row swept)
d3 = store.register(name="durable-box", url="http://d2:9100",
                    pkg_version="0.1.203", gpus=[], ram_total=None,
                    worker_id=did)           # returning id, no live totals yet
check("durable: a returning id with a LOST row inherits totals from memory",
      store._load()[did].get("gpu_total_bytes_known") == 24 * GIB)
check("durable: feasibility works immediately on the restored row (pre-first-beat)",
      W.feasible_modes_for(did, "tf-big") == ("ram-only", "max-ram"))

# a GENUINELY-NEW worker id (never seen, no memory) still fails open + self-logs.
import logging as _logging
_caplog = []
class _H(_logging.Handler):
    def emit(self, r):
        _caplog.append(r.getMessage())
_h = _H()
W.logger.addHandler(_h)
try:
    dn = store.register(name="brand-new", url="http://new:9100",
                        pkg_version="0.1.203", gpus=[], ram_total=None)
    store.assign_model(dn["id"], "tf-big")
    modes = W.feasible_modes_for(dn["id"], "tf-big")
finally:
    W.logger.removeHandler(_h)
check("first-contact: a brand-new worker with NO totals fails open (all coarse "
      "modes + max-ram feasible — never eliminate on missing data; only explicit "
      "stays engine-gated)",
      set(modes) == {"gpu-only", "ram-only", "max-gpu", "max-ram"})
check("first-contact: the self-policing fail-open WARNING fired (drift signal), "
      "naming the missing data",
      any("fail-open" in m and "no gpu/ram totals" in m for m in _caplog))
# rate-limited: a second identical resolution does NOT re-log.
_caplog.clear()
W.logger.addHandler(_h)
try:
    W.feasible_modes_for(dn["id"], "tf-big")
finally:
    W.logger.removeHandler(_h)
check("first-contact: the fail-open log is once-per-(model,worker) (no spam)",
      not any("fail-open" in m for m in _caplog))

print(f"\nALL {ok} capability-aware-default checks passed")
