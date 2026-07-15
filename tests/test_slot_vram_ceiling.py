"""Real-VRAM ~90% ceiling gate on the slot load/evict path (Fix A, 2026-07-15).

ae's 3090 got topped out by a SEPARATE ComfyUI process (~5.5->6.5G) and the slot
scheduler never reacted, because SlotPool.endpoint_for decided "is there room?"
by slot-OCCUPANCY count, never by real device VRAM — so a 95%-full card with an
idle slot loaded happily, then the child under-offloaded or OOMed.

This slice adds a real-VRAM ceiling gate:
  * agent._worker_slot_fit_check(model_key) -> bool: True when loading leaves the
    card at/under the ~90% ceiling (>= (1-ceiling) of total VRAM free after the
    weights land), False when it would breach; degrades to True when free/total/
    need is unknown (no GPU / can't measure) — NEVER blocks a load because we
    couldn't read the card. HUGPY_VRAM_CEILING_FRAC overrides the 0.90 default.
  * slots.SlotPool.endpoint_for: with a registered fit-check that says "over
    ceiling", it evicts the LRU idle on-demand occupant (via the SAME mechanism
    the all-busy promotion branch uses) BEFORE loading, re-checking each round;
    nothing evictable + still over ceiling -> proceeds anyway (honest-degrade).
  * No fit-check registered (bare central / no-GPU) -> occupancy-only routing,
    byte-identical to before.

Runs like the other tests here: venv/bin/python tests/test_slot_vram_ceiling.py
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.worker_agent import agent
slots = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


GIB = 2**30

# ===========================================================================
# Part 1 — agent._worker_slot_fit_check semantics (mock free/total/need)
# ===========================================================================
_fv_orig = agent._free_vram_bytes
_tv_orig = agent._total_vram_bytes
_need_orig = agent._incoming_need_bytes
try:
    # 24 GiB card, ceiling 0.90 -> headroom reserve = 2.4 GiB must stay free.
    agent._total_vram_bytes = lambda: 24 * GIB
    agent._incoming_need_bytes = lambda mk: 4 * GIB          # needs 4 GiB

    # 10 GiB free now: after a 4 GiB load, 6 GiB free >= 2.4 GiB reserve -> fits.
    agent._free_vram_bytes = lambda: 10 * GIB
    check("fit_check: True when loading leaves >= (1-ceiling) headroom",
          agent._worker_slot_fit_check("m") is True)

    # 5 GiB free now: after a 4 GiB load, 1 GiB free < 2.4 GiB reserve -> breach.
    agent._free_vram_bytes = lambda: 5 * GIB
    check("fit_check: False when loading would breach the ~90% ceiling",
          agent._worker_slot_fit_check("m") is False)

    # Exactly at the boundary: free=6.4 GiB, need=4 -> 2.4 GiB free after == reserve.
    agent._free_vram_bytes = lambda: int(6.4 * GIB)
    check("fit_check: True exactly at the ceiling boundary (>= is inclusive)",
          agent._worker_slot_fit_check("m") is True)

    # --- honest-degrade: unmeasurable -> True (never block) ---
    agent._free_vram_bytes = lambda: 5 * GIB                 # would breach if measured
    agent._total_vram_bytes = lambda: None                  # can't read total
    check("fit_check: degrades to True when total VRAM unknown (no GPU)",
          agent._worker_slot_fit_check("m") is True)

    agent._total_vram_bytes = lambda: 24 * GIB
    agent._free_vram_bytes = lambda: None                   # can't read free
    check("fit_check: degrades to True when free VRAM unknown",
          agent._worker_slot_fit_check("m") is True)

    agent._free_vram_bytes = lambda: 5 * GIB
    agent._incoming_need_bytes = lambda mk: None            # unknown weight size
    check("fit_check: degrades to True when the incoming need is unknown",
          agent._worker_slot_fit_check("m") is True)

    # --- ceiling env override respected ---
    agent._incoming_need_bytes = lambda mk: 4 * GIB
    agent._free_vram_bytes = lambda: 5 * GIB                # breaches at 0.90 (see above)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "0.99"          # reserve = 0.24 GiB only
    # after a 4 GiB load, 1 GiB free >= 0.24 GiB reserve -> now fits.
    check("fit_check: HUGPY_VRAM_CEILING_FRAC override respected (0.99 -> fits)",
          agent._worker_slot_fit_check("m") is True)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "0.50"          # reserve = 12 GiB
    # 5 GiB free < 12 GiB reserve even before the load -> breach.
    check("fit_check: a stricter ceiling (0.50) breaches",
          agent._worker_slot_fit_check("m") is False)
    check("_vram_ceiling_frac reads the env override", agent._vram_ceiling_frac() == 0.50)
    os.environ.pop("HUGPY_VRAM_CEILING_FRAC", None)
    check("_vram_ceiling_frac defaults to 0.90", agent._vram_ceiling_frac() == 0.90)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "banana"
    check("_vram_ceiling_frac ignores garbage -> 0.90", agent._vram_ceiling_frac() == 0.90)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "1.5"           # out of (0,1]
    check("_vram_ceiling_frac clamps out-of-range -> 0.90", agent._vram_ceiling_frac() == 0.90)
    os.environ.pop("HUGPY_VRAM_CEILING_FRAC", None)
finally:
    agent._free_vram_bytes = _fv_orig
    agent._total_vram_bytes = _tv_orig
    agent._incoming_need_bytes = _need_orig
    os.environ.pop("HUGPY_VRAM_CEILING_FRAC", None)


# ===========================================================================
# Part 2 — slots.SlotPool.endpoint_for ceiling eviction (mock the pool I/O)
# ===========================================================================
# A fake pool: two slots, one holding an idle on-demand model ("A"), one idle.
# The ceiling gate says "over ceiling" until "A" is evicted (its /unload frees
# the card). We assert endpoint_for evicts A THEN loads the new model.
class FakePool(slots.SlotPool):
    def __init__(self, statuses):
        super().__init__(urls=[s["_control"] for s in statuses])
        self._statuses = statuses
        self.unloaded = []
        self.loaded = []

    def statuses(self):
        # return a shallow copy list of the live dicts (endpoint_for mutates none)
        return [dict(s) for s in self._statuses]

    def unload(self, control_url):
        self.unloaded.append(control_url)
        # free the seat: the occupant is gone
        for s in self._statuses:
            if s["_control"] == control_url:
                s["model_key"] = None
                s["healthy"] = True
        return {"ok": True}


def _post_recorder(pool):
    def fake_post(url, body, timeout):
        pool.loaded.append((url, body.get("model_key")))
        return {"endpoint": url.replace("/load", "") + "/infer"}
    return fake_post


_ep_saved = (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
             slots._post, slots._get)
try:
    # on-demand eviction policy (as the worker registers): A is on-demand.
    slots.set_eviction_policy(lambda mk: True)      # every occupant is on-demand
    slots.set_residency_lookup(lambda mk: "on-demand")
    slots._get = lambda url, timeout=3.0: {}        # unused (we override statuses)

    # (i) OVER CEILING then FITS after evicting A ---------------------------
    st = [
        {"_control": "http://s0", "model_key": "A", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)

    # gate: over ceiling while A is resident; fits once A is gone.
    def gate_needs_A_gone(mk):
        a_resident = any(s.get("model_key") == "A" for s in pool._statuses)
        return not a_resident      # False (over ceiling) while A resident
    slots.set_fit_check(gate_needs_A_gone)

    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(i) endpoint_for evicted the LRU on-demand occupant A before loading",
          pool.unloaded == ["http://s0"])
    check("(i) then loaded NEW into a freed idle slot",
          any(mk == "NEW" for (_u, mk) in pool.loaded))
    check("(i) returned a usable endpoint", isinstance(ep, str) and ep)

    # (ii) ALREADY UNDER CEILING -> no eviction, just load into the idle slot ---
    st = [
        {"_control": "http://s0", "model_key": "A", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    slots.set_fit_check(lambda mk: True)            # always fits
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(ii) under ceiling: NO eviction",
          pool.unloaded == [])
    check("(ii) under ceiling: loaded NEW into the pre-existing idle slot",
          pool.loaded and pool.loaded[0][1] == "NEW")

    # (iii) HONEST-DEGRADE: over ceiling, nothing evictable -> proceed anyway ---
    # Only a static occupant + an idle slot; eviction policy rejects static, so
    # nothing is evictable, yet the load must still proceed (never hang).
    st = [
        {"_control": "http://s0", "model_key": "STAT", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    slots.set_eviction_policy(lambda mk: mk != "STAT")   # STAT is not on-demand
    slots.set_fit_check(lambda mk: False)                # ALWAYS over ceiling
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(iii) honest-degrade: nothing evictable -> STAT never evicted",
          pool.unloaded == [])
    check("(iii) honest-degrade: the load STILL proceeds (never hangs)",
          pool.loaded and pool.loaded[0][1] == "NEW" and isinstance(ep, str))

    # (iv) NO fit-check registered -> occupancy-only, byte-identical to before ---
    st = [
        {"_control": "http://s0", "model_key": "A", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    slots.set_fit_check(None)                       # no ceiling gate
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(iv) no fit-check: NO ceiling eviction (occupancy-only path)",
          pool.unloaded == [])
    check("(iv) no fit-check: loaded straight into the idle slot",
          pool.loaded and pool.loaded[0][1] == "NEW")
finally:
    (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
     slots._post, slots._get) = _ep_saved

print(f"\nall {ok} checks passed")
