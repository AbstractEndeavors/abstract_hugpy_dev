"""Slice 10 — VRAM evict-to-fit at admission + the 90% headroom sweep.

Field incident (ae, 2026-07-17): a transformers load OOM'd — "31.69 MiB free of
23.56 GiB. Process 2586405 has 21.26 GiB" = an IDLE coder SLOT CHILD squatting
the card. Nothing evicted it first: the in-process contention path only ever saw
_INSTANCES residents and refused slot-backed models. The addendum incident: comfy
GREW out-of-band + the idle non-grower squatted → 100% + deadlock; the keeper
had to /evict by hand (proving the machinery, not the policy).

The operator's ruling: "everything is on demand — the process not actively
replying and not ahead of the subject in the queue, as well as not 'static',
should be evicted to allow the subject process to proliferate."

These drive _vram_evict_to_fit / _vram_headroom_sweep with the seams stubbed
(VRAM readers, residents, evict verb) so behavior is asserted without a GPU.

Run: venv/bin/python -m pytest tests/test_vram_evict_to_fit.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.worker_agent import agent as A          # noqa: E402
from abstract_hugpy_dev.worker_agent import gen_gate            # noqa: E402
# managers/__init__ star-imports shadow the subpackage attrs — bind the REAL
# module the agent uses via import_module (same landmine the sibling tests note).
D = importlib.import_module("abstract_hugpy_dev.managers.dispatch.dispatch")

GIB = 1 << 30

# The real function object, captured BEFORE any fixture monkeypatches the module
# attribute — the k30 end-to-end test restores it to drive the slot union.
_REAL_VRAM_RESIDENTS = A._vram_residents


class _State:
    pass


@pytest.fixture
def rig(monkeypatch):
    """A GPU with a mutable free-VRAM cell and a resident set. Evicting a model
    removes it from residents AND adds its bytes back to free (the reclaim)."""
    card = {"total": 24 * GIB, "free": 0, "need": 0}
    residents = {}          # model_key -> {vram_bytes, host_mode}
    lru = {}                # model_key -> last_used epoch
    static = set()
    replying = set()
    busy_slots = set()
    evicted_calls = []

    # ENV HYGIENE. These knobs are read straight from os.environ deep inside the
    # admission (spill.alloc_mode_env, the HUGPY_GPU_MEM_GIB band cap, the 4-bit
    # re-price, the thrash floor), so a sibling suite that setenv's one of them
    # in the same process silently re-plans every test here. That is exactly the
    # cross-file pollution that made four tests in this file pass alone and fail
    # in a full run: test_fleet_templates leaks HUGPY_GPU_MEM_GIB="0.0", which is
    # a TRUTHY string, so `budget = min(budget, band_ceiling(0.0, ...))` collapsed
    # the partial-offload budget to 0 and every hybrid degenerated into a refusal.
    # Clearing them here makes the rig mean what it says: an unconfigured box.
    for _leak in ("HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_ALLOC_MODE",
                  "HUGPY_LENIENCY_PCT", "HUGPY_PRIORITY_DEVICE", "HUGPY_BNB_4BIT",
                  "HUGPY_N_GPU_LAYERS", "HUGPY_VRAM_CEILING_FRAC",
                  "HUGPY_EVICT_MIN_RESIDENCY_S", "HUGPY_EVICT_LEAST_REAPING"):
        monkeypatch.delenv(_leak, raising=False)

    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: card["need"])
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v["vram_bytes"],
                                    "host_mode": v["host_mode"], "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency",
                        lambda mk: "static" if mk in static else "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set(busy_slots))
    monkeypatch.setattr(gen_gate, "in_flight",
                        lambda mk: 1 if mk in replying else 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    # last_used_snapshot lives in dispatch; patch it there.
    monkeypatch.setattr(D, "last_used_snapshot", lambda: dict(lru))

    def _fake_evict(state, mk, force=False):
        evicted_calls.append(mk)
        row = residents.pop(mk, None)
        freed = row["vram_bytes"] if row else 0
        card["free"] += freed
        return {"model_key": mk, "evicted": bool(row),
                "vram_freed": freed if row else None,
                "host_mode": row["host_mode"] if row else "none"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)

    # reset the module counter for a clean assertion each test
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)

    return type("Rig", (), {
        "card": card, "residents": residents, "lru": lru, "static": static,
        "replying": replying, "busy_slots": busy_slots,
        "evicted": evicted_calls})()


# ── THE ae SHAPE: idle 21.3G slot child evicted for a small transformers load ──
def test_idle_slot_child_evicted_for_a_new_load(rig):
    rig.card["free"] = 32 * 1024 * 1024          # 31.69 MiB free (the incident)
    rig.card["need"] = 500 * 1024 * 1024         # a small transformers subject
    rig.residents["Qwen~Qwen3-Coder-Next-GGUF"] = {
        "vram_bytes": int(21.26 * GIB), "host_mode": "subprocess"}   # the squatter
    rig.lru["Qwen~Qwen3-Coder-Next-GGUF"] = 100.0                    # idle, cold

    plan = A._vram_evict_to_fit(_State(), "identity-vl-subject")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["Qwen~Qwen3-Coder-Next-GGUF"]         # slot child GONE
    assert A._VRAM_EVICTIONS["count"] == 1
    assert A._VRAM_EVICTIONS["last"]["victim"] == "Qwen~Qwen3-Coder-Next-GGUF"


def test_already_fits_evicts_nothing(rig):
    rig.card["free"] = 10 * GIB
    rig.card["need"] = 2 * GIB
    rig.residents["idle"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "proceed"
    assert plan["evicted"] == []


# ── static is refused-around, never evicted ────────────────────────────────
def test_static_resident_is_protected_subject_refuses_honestly(rig):
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["static_big"] = {"vram_bytes": 20 * GIB, "host_mode": "in_process"}
    rig.static.add("static_big")
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert plan["evicted"] == []                 # static never evicted
    reason = plan["reason"]
    assert any(p["model_key"] == "static_big" and "static" in p["why"]
               for p in reason["protected"])
    assert "won't fit on GPU" in reason["reason"]


# ── actively replying is protected (measured, not inferred) ────────────────
def test_actively_replying_resident_is_protected(rig):
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["busy"] = {"vram_bytes": 20 * GIB, "host_mode": "in_process"}
    rig.replying.add("busy")                     # in-flight generation
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert "busy" not in rig.evicted
    assert any("actively replying" in p["why"] for p in plan["reason"]["protected"])


def test_busy_slot_is_protected(rig):
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["slotbusy"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    rig.busy_slots.add("slotbusy")               # slot-side busy flag
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert "slotbusy" not in rig.evicted


# ── queue-ahead is protected ───────────────────────────────────────────────
def test_queued_ahead_resident_is_protected(rig, monkeypatch):
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["ahead"] = {"vram_bytes": 20 * GIB, "host_mode": "in_process"}
    # A resident with pending work queued ahead of the subject.
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subj: {"ahead"})
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert "ahead" not in rig.evicted
    assert any("queued ahead" in p["why"] for p in plan["reason"]["protected"])


# ── comfy is never evicted here (0.1.137 exclusion; its own headroom path) ──
def test_comfy_resident_is_never_evicted(rig):
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["comfy-sdxl"] = {"vram_bytes": 20 * GIB, "host_mode": "comfy"}
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert "comfy-sdxl" not in rig.evicted
    assert any(p["host_mode"] == "comfy" for p in plan["reason"]["protected"])


# ── minimum LRU set: coldest first, stop as soon as it fits ────────────────
def test_evicts_the_minimum_lru_set(rig):
    rig.card["free"] = 0
    rig.card["need"] = 6 * GIB
    # Three idle residents; need 6G. Coldest is 'c1' (5G) — evicting it + 'c2'
    # (5G) yields 10G >= 6G, but c1 alone (5G) is short, so it takes c1 then c2
    # and stops (does NOT touch the warmest c3).
    rig.residents["c1"] = {"vram_bytes": 5 * GIB, "host_mode": "in_process"}
    rig.residents["c2"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.residents["c3"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.lru.update(c1=100.0, c2=200.0, c3=300.0)   # c1 coldest, c3 warmest
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["c1", "c2"]         # coldest two, in order
    assert "c3" not in plan["evicted"]             # warmest untouched


# ── full permissible eviction still short → honest refusal ──────────────────
def test_full_evict_still_short_refuses_with_reasons(rig):
    rig.card["free"] = 0
    rig.card["need"] = 30 * GIB                    # bigger than the whole card
    rig.residents["idle1"] = {"vram_bytes": 5 * GIB, "host_mode": "in_process"}
    rig.residents["idle2"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), "huge")
    assert plan["action"] == "refuse"
    # It DID evict everything permissible (honest effort) but still short.
    assert set(plan["evicted"]) == {"idle1", "idle2"}
    r = plan["reason"]
    assert r["needs_bytes"] == 30 * GIB
    assert r["evicted_freed_bytes"] == 10 * GIB
    assert "won't fit on GPU" in r["reason"]


# ── fail-open: unmeasurable never blocks a load ────────────────────────────
def test_no_gpu_is_a_noop(rig):
    rig.card["total"] = 0                          # no GPU
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "proceed"
    assert rig.evicted == []


def test_unknown_need_fails_open(rig):
    rig.card["free"] = 0
    rig.card["need"] = 0                           # size unknown
    rig.residents["idle"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "proceed"
    assert rig.evicted == []                       # nothing evicted on fail-open


# ═══════════ the 90% headroom sweep (addendum incident) ════════════════════
def test_headroom_sweep_evicts_when_over_ceiling(rig):
    # Card at ~100% (comfy grew): free below the (1 - 0.90) reserve of 24G = 2.4G.
    rig.card["free"] = 100 * 1024 * 1024          # 100 MiB free (deadlock shape)
    rig.residents["idle_coder"] = {"vram_bytes": int(21.3 * GIB),
                                   "host_mode": "subprocess"}
    rig.lru["idle_coder"] = 50.0
    A._vram_headroom_sweep(_State())
    assert rig.evicted == ["idle_coder"]           # coldest idle reclaimed
    assert A._VRAM_EVICTIONS["last"]["subject"] == "headroom-sweep"


def test_headroom_sweep_noop_under_ceiling(rig):
    rig.card["free"] = 10 * GIB                    # plenty free
    rig.residents["idle"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    A._vram_headroom_sweep(_State())
    assert rig.evicted == []


def test_headroom_sweep_protects_static_and_replying(rig):
    rig.card["free"] = 100 * 1024 * 1024
    rig.residents["stat"] = {"vram_bytes": 12 * GIB, "host_mode": "in_process"}
    rig.residents["busy"] = {"vram_bytes": 12 * GIB, "host_mode": "subprocess"}
    rig.static.add("stat")
    rig.replying.add("busy")
    A._vram_headroom_sweep(_State())
    assert rig.evicted == []                        # both protected -> nothing evicted


def test_headroom_sweep_never_evicts_comfy(rig):
    rig.card["free"] = 100 * 1024 * 1024
    rig.residents["comfy-x"] = {"vram_bytes": 12 * GIB, "host_mode": "comfy"}
    rig.residents["idle"] = {"vram_bytes": 12 * GIB, "host_mode": "subprocess"}
    rig.lru.update(**{"comfy-x": 10.0, "idle": 20.0})
    A._vram_headroom_sweep(_State())
    assert rig.evicted == ["idle"]                  # comfy skipped, idle taken
    assert "comfy-x" not in rig.evicted


# ═══════════ dispatch wiring: refusal raises LoadRefusal ═══════════════════
def test_ensure_headroom_raises_loadrefusal_on_make_room_refuse(monkeypatch):
    
    monkeypatch.setattr(D, "_FIT_CHECK", None)      # skip the in-process path
    monkeypatch.setattr(D, "_MAKE_ROOM",
                        lambda mk: {"action": "refuse", "evicted": [],
                                    "reason": {"reason": "won't fit on GPU",
                                               "model_key": mk}})
    with pytest.raises(D.LoadRefusal):
        D.ensure_headroom_for_load("subject")


def test_ensure_headroom_returns_evicted_from_make_room(monkeypatch):

    monkeypatch.setattr(D, "_FIT_CHECK", None)
    monkeypatch.setattr(D, "_MAKE_ROOM",
                        lambda mk: {"action": "evicted", "evicted": ["idle_slot"],
                                    "reason": None})
    out = D.ensure_headroom_for_load("subject")
    assert out == ["idle_slot"]


def test_ensure_headroom_does_not_raise_on_partial_admit(monkeypatch):
    # A PARTIAL verdict is an ADMIT (honest hybrid), never a refusal — the
    # in-process load proceeds and reads the pinned n_gpu_layers via spill.
    monkeypatch.setattr(D, "_FIT_CHECK", None)
    monkeypatch.setattr(D, "_MAKE_ROOM",
                        lambda mk: {"action": "partial", "evicted": [],
                                    "n_gpu_layers": 17, "gpu_pct": 35,
                                    "reason": None})
    out = D.ensure_headroom_for_load("subject")   # must NOT raise LoadRefusal
    assert out == []


# ═══════════ stage (2.5): honest GGUF partial offload at admission ══════════
# The oversize agent-brain incident: a GGUF whose FULL weights exceed the card
# even on an EMPTY card was hard-refused. Autofit's promise is a hybrid — offload
# the layers that fit, stream the rest to CPU RAM — so admission now DEGRADES to
# a partial offload instead of refusing (GGUF/slot path only).
@pytest.fixture
def gguf_rig(rig, monkeypatch):
    """Extend the base rig for the GGUF partial path: a served-quant geometry and
    a controllable host-RAM reading."""
    geo = {"path": "/models/coder-next/q4.gguf", "layers": 48}
    ram = {"free": 200 * GIB}
    monkeypatch.setattr(A, "_served_gguf_geometry",
                        lambda mk: (geo["path"], geo["layers"]))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: ram["free"])
    # Clear any leftover ngl pin (module globals) so each test starts clean.
    A._PARTIAL_NGL.clear()
    from abstract_hugpy_dev.managers import spill as _spill
    _spill._NGL_OVERRIDE.clear()
    return type("GgufRig", (), {"geo": geo, "ram": ram})()


def test_oversize_gguf_admits_as_partial_offload(rig, gguf_rig):
    # coder-next shape: 24 GiB card, ~21 GiB free (empty-ish), 52 GiB need.
    rig.card["free"] = 21 * GIB
    rig.card["need"] = 52 * GIB
    plan = A._vram_evict_to_fit(_State(), "Qwen~Qwen3-Coder-Next-GGUF")
    assert plan["action"] == "partial"
    assert plan["n_gpu_layers"] > 0
    assert 0 < plan["gpu_pct"] < 100
    # budget = 21 - 2.4 = 18.6 GiB; per-layer = 52/48 GiB -> 17 layers fit.
    assert plan["n_gpu_layers"] == 17
    # The in-process load is pinned to the honest count (overrides shard-blind
    # autofit) on the served path.
    from abstract_hugpy_dev.managers import spill
    assert spill._NGL_OVERRIDE.get(gguf_rig.geo["path"]) == 17
    assert A._PARTIAL_NGL["Qwen~Qwen3-Coder-Next-GGUF"]["n"] == 17


def test_partial_refused_when_ram_cannot_hold_remainder(rig, gguf_rig):
    rig.card["free"] = 21 * GIB
    rig.card["need"] = 52 * GIB
    gguf_rig.ram["free"] = 4 * GIB                 # can't hold the ~33 GiB CPU share
    plan = A._vram_evict_to_fit(_State(), "coder")
    assert plan["action"] == "refuse"
    considered = plan["reason"]["partial_offload_considered"]
    assert considered["admit"] is False
    assert "host RAM" in considered["reject_reason"]
    assert "host RAM" in plan["reason"]["reason"]  # extended honest message
    # A refused hybrid pins nothing.
    from abstract_hugpy_dev.managers import spill
    assert gguf_rig.geo["path"] not in spill._NGL_OVERRIDE


def test_partial_refused_when_offload_is_degenerate(rig, gguf_rig):
    # Barely over the ceiling: tiny budget -> ~0 layers -> below the floor.
    rig.card["free"] = 3 * GIB                      # 3 - 2.4 = 0.6 GiB budget
    rig.card["need"] = 52 * GIB
    plan = A._vram_evict_to_fit(_State(), "coder")
    assert plan["action"] == "refuse"
    assert plan["reason"]["partial_offload_considered"]["admit"] is False
    assert "degenerate" in plan["reason"]["reason"]


def test_non_gguf_oversize_still_refuses_unchanged(rig, monkeypatch):
    # No GGUF geometry -> no partial attempted -> the honest refusal, unchanged.
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 30 * GIB
    plan = A._vram_evict_to_fit(_State(), "some-transformers-model")
    assert plan["action"] == "refuse"
    assert "partial_offload_considered" not in plan["reason"]


def test_full_fit_never_reaches_partial_and_pin_is_cleared(rig, gguf_rig):
    # Fits outright -> proceed, no partial, and any stale pin is cleared at entry.
    from abstract_hugpy_dev.managers import spill
    spill.set_ngl_override(gguf_rig.geo["path"], 5)          # stale from a prior load
    A._PARTIAL_NGL["subject"] = {"path": gguf_rig.geo["path"], "n": 5}
    rig.card["free"] = 30 * GIB
    rig.card["need"] = 2 * GIB
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "proceed"
    assert spill._NGL_OVERRIDE.get(gguf_rig.geo["path"]) is None   # re-decided full
    assert "subject" not in A._PARTIAL_NGL


# ═══════════ k30 (2026-07-23): the invisibility protection class is closed ═══
# Incident (ae): a chat load for the 51.8G coder was fast-refused with
# "evicted 0 idle resident(s) freeing 0 B, but 0 protected resident(s) still
# hold the card" while an IDLE 18.85G Fable slot occupant plainly held it. Root
# cause: the evict planner enumerated ONLY the in-memory pid registry; a slot
# occupant the registry hadn't (re)recorded — fresh re-exec, swept record, or a
# child tagged as an anonymous cuda_context lump (model_key=None) — was
# invisible, i.e. immune to eviction: a de-facto third protection class beyond
# the operator's ruling (only 🔒static and actively-answering protect). Fix:
# _vram_residents unions LIVE slot occupants with the registry, and the refusal
# only claims what is true (failed evictions counted; unattributed occupancy
# named instead of "0 protected still hold the card").

def test_vram_residents_unions_live_slot_occupant_missing_from_registry(monkeypatch):
    """A slot occupant with NO pid-registry record is still an enumerable
    resident (the same collection the allocations view shows)."""
    from abstract_hugpy_dev.worker_agent import pid_registry as PR

    class _EmptyReg:
        @staticmethod
        def snapshot_for_heartbeat():
            return {"models": [], "unattributed": []}     # registry knows nothing
    monkeypatch.setattr(A, "_slot_statuses", lambda: [
        {"slot_id": "1", "model_key": "Fable-Distill", "child_pid": 4242,
         "busy": False, "healthy": True},
        {"slot_id": "2", "model_key": None, "child_pid": None},
    ])
    monkeypatch.setattr(A, "_gpu_process_vram",
                        lambda: {4242: {"name": "llama-server", "mib": 17977}})
    monkeypatch.setattr(PR, "snapshot_for_heartbeat", _EmptyReg.snapshot_for_heartbeat)

    rows = A._vram_residents(_State())
    assert [r["model_key"] for r in rows] == ["Fable-Distill"]
    assert rows[0]["host_mode"] == "subprocess"
    assert rows[0]["vram_bytes"] == 17977 * (1 << 20)     # joined from nvidia-smi


def test_vram_residents_does_not_duplicate_registry_backed_slot(monkeypatch):
    from abstract_hugpy_dev.worker_agent import pid_registry as PR
    monkeypatch.setattr(PR, "snapshot_for_heartbeat", lambda: {"models": [
        {"model_key": "Fable-Distill", "pid": 4242, "host_mode": "subprocess",
         "vram_bytes": 5, "alive": True},
        {"model_key": None, "pid": 999, "host_mode": "cuda_context",
         "vram_bytes": 1, "alive": True},                 # anonymous lump: skipped
    ], "unattributed": []})
    monkeypatch.setattr(A, "_slot_statuses", lambda: [
        {"slot_id": "1", "model_key": "Fable-Distill", "child_pid": 4242}])
    rows = A._vram_residents(_State())
    assert [r["model_key"] for r in rows] == ["Fable-Distill"]   # once, not twice


def test_k30_idle_slot_invisible_to_registry_is_evicted_not_refused(
        rig, monkeypatch):
    """THE k30 SHAPE end-to-end: registry-blind idle 18.85G slot occupant, a
    51.8G subject. The planner must see it via the slot union and evict it —
    then the 48-layer hybrid becomes viable (18/48 >= the 3-layer floor)."""
    # Un-stub _vram_residents: use the REAL union against fake registry+slots
    # (the rig fixture replaced it; _REAL_VRAM_RESIDENTS was captured at import).
    monkeypatch.setattr(A, "_vram_residents", _REAL_VRAM_RESIDENTS)
    from abstract_hugpy_dev.worker_agent import pid_registry as PR
    monkeypatch.setattr(PR, "snapshot_for_heartbeat",
                        lambda: {"models": [], "unattributed": []})
    fable_vram = int(18851299328)
    slot_rows = [{"slot_id": "1", "model_key": "Fable-Distill",
                  "child_pid": 4242, "busy": False, "healthy": True}]
    monkeypatch.setattr(A, "_slot_statuses", lambda: list(slot_rows))
    monkeypatch.setattr(A, "_gpu_process_vram",
                        lambda: {4242: {"name": "llama-server",
                                        "mib": fable_vram // (1 << 20)}})
    # 23.6G card, 4.0G free, subject needs 51.8G (the incident numbers).
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(4.0 * GIB)
    rig.card["need"] = int(51.8 * GIB)
    # The fake evictor must free the slot occupant's bytes when asked.
    rig.residents["Fable-Distill"] = {"vram_bytes": fable_vram,
                                      "host_mode": "subprocess"}
    # Hybrid geometry: coder-next 48 layers, plenty of host RAM.
    monkeypatch.setattr(A, "_served_gguf_geometry",
                        lambda mk: ("/models/coder-next/q4.gguf", 48))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: 200 * GIB)
    A._PARTIAL_NGL.clear()
    from abstract_hugpy_dev.managers import spill as _spill
    _spill._NGL_OVERRIDE.clear()

    plan = A._vram_evict_to_fit(_State(), "Qwen~Qwen3-Coder-Next-GGUF")
    assert "Fable-Distill" in plan["evicted"]             # the squatter yielded
    # Full 51.8G still can't fit a 23.6G card — but the hybrid now admits:
    # budget = (4.0 + 18.85 hmm freed) - 2.36 reserve ≈ 20.4G; per-layer =
    # 51.8/48 ≈ 1.079G -> 18 layers ≥ the 3-layer floor.
    assert plan["action"] == "partial"
    assert plan["n_gpu_layers"] >= 17


def test_k30_refusal_message_is_truthful_when_nothing_enumerable(rig, monkeypatch):
    """Occupied card, ZERO enumerable residents: the refusal must NOT claim
    'N protected resident(s) still hold the card' — it names the unattributed
    occupancy instead."""
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(4.0 * GIB)
    rig.card["need"] = int(51.8 * GIB)
    # residents dict left EMPTY -> the (stubbed) planner sees nothing.
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    plan = A._vram_evict_to_fit(_State(), "coder")
    assert plan["action"] == "refuse"
    msg = plan["reason"]["reason"]
    assert "protected resident(s) still hold the card" not in msg
    assert "cannot map to a model_key" in msg
    assert plan["reason"]["evict_failed"] == []


def test_k30_failed_eviction_is_counted_in_the_refusal(rig, monkeypatch):
    """An eviction attempt that frees nothing must surface in the refusal
    (evict_failed), never silently read as 'evicted 0 ... 0 protected'."""
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 30 * GIB
    rig.residents["stuck"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    monkeypatch.setattr(
        A, "_evict_model",
        lambda state, mk, force=False: {"model_key": mk, "evicted": False,
                                        "vram_freed": None, "host_mode": "slot",
                                        "reason": "slot unload failed: boom"})
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert plan["evicted"] == []
    ef = plan["reason"]["evict_failed"]
    assert len(ef) == 1 and ef[0]["model_key"] == "stuck"
    assert "eviction attempt(s) failed" in plan["reason"]["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# THE EVICTION FLOW SPEC ON THE REAL VRAM PATH (assets/evictionflow.html).
#
# tests/test_eviction_flow.py asserts the shared function in isolation. These
# assert it is genuinely WIRED INTO _vram_evict_to_fit — the admission choke
# point — because a correct helper nobody calls is the failure mode this repo
# has shipped before.
# ─────────────────────────────────────────────────────────────────────────────
def test_cliff_order_a_ram_preferring_resident_yields_before_a_gpu_one(rig, monkeypatch):
    """KEY ①, on the device where it bites. The RAM-preferring resident is
    HOTTER and MORE CALLED, and still goes first: it is on this card only
    opportunistically (already off the cliff by design), whereas the max-gpu
    resident losing residency is the measured 135->36 tok/s drop."""
    rig.card["free"] = 0
    rig.card["need"] = 4 * GIB
    rig.residents["wants-ram"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.residents["wants-gpu"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.lru.update({"wants-ram": 9999.0, "wants-gpu": 1.0})   # ram one is HOTTER
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "alloc_mode",
                        {"wants-ram": "max-ram", "wants-gpu": "max-gpu"})
    plan = A._vram_evict_to_fit(_State(), "subject")
    # ORDER is the assertion (the ~90% ceiling reserve means the need exceeds
    # either resident alone, so both go — but the SEQUENCE is the cliff order).
    assert plan["evicted"][0] == "wants-ram", (
        "cliff order: the mismatched (RAM-preferring) resident must yield "
        "before the one whose preference names this card")


def test_without_a_known_mode_the_order_is_todays_idle_first(rig, monkeypatch):
    """DEGRADE-NOT-GUESS: with no persisted alloc_mode every resident degrades
    to the blank max-gpu default, key ① becomes a constant, and the order is
    the honest idle-first one — byte-identical to today. An unknown preference
    must never invent a cliff-order verdict."""
    rig.card["free"] = 0
    rig.card["need"] = 4 * GIB
    rig.residents["hot"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.residents["cold"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.lru.update({"hot": 9999.0, "cold": 1.0})
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "alloc_mode", {})
    assert A._vram_evict_to_fit(_State(), "subject")["evicted"][0] == "cold"


def test_least_reaping_on_the_vram_path_spares_the_redundant_victim(rig):
    """Walk-then-drop, on the real admission. need 12: the walk takes
    small(5) then big(20) because 5 was short; the drop pass then removes
    'small' since 'big' alone covers 12. ONE model unloaded, not two."""
    rig.card["free"] = 0
    rig.card["need"] = 12 * GIB
    rig.residents["small"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.residents["big"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    rig.lru.update({"small": 1.0, "big": 2.0})     # small is colder -> walked first
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["evicted"] == ["big"]
    assert "small" not in plan["evicted"], "least reaping: y1 spared"


def test_the_frontier_rule_holds_on_the_vram_path(rig):
    """A hot resident that is a PERFECT fit for the need is never pulled in:
    the walk stops before it, and the drop pass only removes."""
    rig.card["free"] = 0
    rig.card["need"] = 12 * GIB
    rig.residents["c1"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.residents["c2"] = {"vram_bytes": 10 * GIB, "host_mode": "subprocess"}
    rig.residents["perfect-but-hot"] = {"vram_bytes": 12 * GIB,
                                        "host_mode": "subprocess"}
    rig.lru.update({"c1": 1.0, "c2": 2.0, "perfect-but-hot": 9999.0})
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert "perfect-but-hot" not in plan["evicted"]
    assert set(plan["evicted"]) == {"c1", "c2"}


def test_the_thrash_floor_protects_a_freshly_loaded_resident(rig, monkeypatch):
    """ENACTED PROPOSAL 2, on the real path. 'fresh' is the coldest thing on
    the card by idle anchor (loaded 10s ago, never called) and would be evicted
    immediately — load, evict, reload. The floor REMOVES it from the pool."""
    import time as _t
    rig.card["free"] = 0
    rig.card["need"] = 4 * GIB
    now = _t.time()
    rig.residents["fresh"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess",
                              "resident_since": now - 10}
    rig.residents["settled"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess",
                                "resident_since": now - 99_999}
    rig.lru.update({"settled": now - 5})           # settled answered 5s ago
    monkeypatch.setattr(
        A, "_vram_residents",
        lambda s: [{"model_key": k, **v, "alive": True}
                   for k, v in rig.residents.items()])
    monkeypatch.setenv("HUGPY_EVICT_MIN_RESIDENCY_S", "300")
    assert A._vram_evict_to_fit(_State(), "subject")["evicted"][0] == "settled", (
        "the floored fresh load must not be the FIRST thing taken")
    # Floor OFF -> the thrash reproduces (the fresh load is taken).
    monkeypatch.setenv("HUGPY_EVICT_MIN_RESIDENCY_S", "0")
    rig.residents["fresh"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess",
                              "resident_since": now - 10}
    rig.residents["settled"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess",
                                "resident_since": now - 99_999}
    rig.card["free"] = 0
    assert A._vram_evict_to_fit(_State(), "subject")["evicted"][0] == "fresh", (
        "with the floor off the thrash reproduces: the fresh load leads")


def test_static_still_outranks_everything_the_spec_says(rig, monkeypatch):
    """The operator's protection ruling is applied UPSTREAM of the shared pool
    (in _partition_residents) and is not weakened by any of this: a 🔒static
    resident is not a victim even when it is the only thing that could fit."""
    rig.card["free"] = 0
    rig.card["need"] = 4 * GIB
    rig.residents["locked"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    rig.static.add("locked")
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse" and plan["evicted"] == []


# ═══════ 2026-07-27: THE SUBJECT IS ITS OWN BLOCKER (ae, 0.1.216) ═══════════
# Operator, verbatim: "it says its serving it, it gets the call, it loads the
# model, it serves it then says it cannot because it's not loaded and has no
# room."
#
#   LoadRefusal: won't fit on GPU: needs 12.0 GB, 8.1 GB free of 23.6 GB
#   (2.4 GB ceiling reserve); evicted 0 idle resident(s) freeing 0 B;
#   no evictable resident is attributable to a model, yet ~15.4 GB of the card
#   is in use — GPU memory is held by process(es) this worker cannot map to a
#   model_key (orphaned/adopted child or out-of-band process)
#
# …emitted while the console showed the SAME model resident and serving:
#   MN-GRAND-23.5B-Gutenberg : 13.3 GiB attributed + 1.6 GiB KV = 14.8 GiB used
#   pid registry: MN-GRAND -> pid 1071915 -> in_process -> 12.8 GiB (alive)
#   vram_attributed_bytes = 13.3 GiB     vram_unattributed_bytes = 0
#
# TWO defects, both asserted here:
#  A. `_partition_residents` drops the subject from BOTH halves ("never evict
#     yourself", correct) and nothing ever credited the subject's own footprint
#     as headroom — so the only resident on the card was invisible to the fit
#     math AND unavailable as a victim. 8.1 free + 12.8 held = 20.9 >= 12.0.
#  B. the "cannot map to a model_key / orphaned / out-of-band" sentence fired on
#     an empty candidates+protected pool, which means "nothing EVICTABLE", not
#     "nothing ATTRIBUTABLE". It was false, and provably so.
MNG = "TheDrummer~MN-GRAND-Gutenberg-Lyra4-Lyra-23.5B-v4.0-GGUF"


@pytest.fixture
def ae_1216(rig):
    """The live ae numbers, exactly: 23.6 GiB card, 8.1 GB free, 12.0 GB need,
    subject already resident in-process holding 12.8 GiB."""
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = 8_100_000_000               # 8.1 GB free (the refusal)
    rig.card["need"] = 12_000_000_000              # 12.0 GB need (the refusal)
    rig.residents[MNG] = {"vram_bytes": int(12.8 * GIB),   # pid 1071915, alive
                          "host_mode": "in_process"}
    return rig


def test_resident_subject_is_credited_its_own_footprint_and_admits(ae_1216):
    """THE REGRESSION. The subject is already on the card; crediting its own
    bytes makes its own need fit trivially, so admission proceeds instead of
    refusing — and it evicts NOTHING to do it."""
    plan = A._vram_evict_to_fit(_State(), MNG)
    assert plan["action"] == "proceed", plan.get("reason")
    assert plan["evicted"] == []
    assert ae_1216.evicted == []                   # nobody was disturbed


def test_the_subject_is_never_a_victim_of_its_own_admission(ae_1216):
    """The credit must not be implemented by evicting the subject: the operator's
    'never the subject itself' protection is inviolable and the whole point is
    that the bytes are ALREADY there."""
    A._vram_evict_to_fit(_State(), MNG)
    assert MNG not in ae_1216.evicted
    assert MNG in ae_1216.residents                # still resident afterwards


def test_credit_is_exactly_the_measured_footprint(ae_1216):
    """The credited figure is the MEASURED pid-registry footprint, not a
    declared/derived one — a guessed credit is how admit-then-OOM happens."""
    assert A._subject_resident_vram_bytes(_State(), MNG) == int(12.8 * GIB)


def test_credit_is_zero_for_a_subject_that_is_not_resident(ae_1216):
    """Non-resident subject -> no credit -> today's arithmetic, untouched."""
    assert A._subject_resident_vram_bytes(_State(), "some-other-model") == 0


def test_non_resident_subject_path_is_byte_identical(rig):
    """The regression guard for everyone else: with the subject absent from the
    resident set the verdict is exactly what it was before the credit existed
    (this is the ae shape with the subject's row removed -> still refuses)."""
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = 8_100_000_000
    rig.card["need"] = 12_000_000_000
    rig.residents["someone-else"] = {"vram_bytes": int(12.8 * GIB),
                                     "host_mode": "in_process"}
    # someone-else is evictable, so this admits by eviction — the subject gets
    # no credit at all and the neighbour pays, exactly as before.
    plan = A._vram_evict_to_fit(_State(), MNG)
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["someone-else"]


def test_resident_subject_still_too_big_refuses_honestly(rig, monkeypatch):
    """The credit is a fit INPUT, not a bypass. A re-seat that wants more than
    even (free + its own bytes) can give still refuses — and the refusal shows
    the credit so the arithmetic is checkable."""
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 40 * GIB                    # bigger than the whole card
    rig.residents["big"] = {"vram_bytes": int(12.8 * GIB), "host_mode": "in_process"}
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    plan = A._vram_evict_to_fit(_State(), "big")
    assert plan["action"] == "refuse"
    assert plan["evicted"] == []                   # never itself
    r = plan["reason"]
    assert r["subject_resident_bytes"] == int(12.8 * GIB)
    assert r["free_vram_bytes"] == 1 * GIB         # RAW device read, unchanged
    assert r["free_vram_effective_bytes"] == 1 * GIB + int(12.8 * GIB)
    assert "the subject itself already holds" in r["reason"]


def test_refusal_names_the_subject_instead_of_inventing_an_orphan(rig, monkeypatch):
    """DEFECT B. When the subject is the holder, the refusal must SAY SO — never
    'GPU memory is held by process(es) this worker cannot map to a model_key'.
    That sentence sent the operator hunting external PIDs that did not exist."""
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 40 * GIB
    rig.residents["big"] = {"vram_bytes": int(12.8 * GIB), "host_mode": "in_process"}
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    msg = A._vram_evict_to_fit(_State(), "big")["reason"]["reason"]
    assert "cannot map to a model_key" not in msg
    assert "orphaned" not in msg
    assert "the SUBJECT ITSELF" in msg
    assert "big" in msg


def test_orphan_message_only_fires_on_genuinely_unattributed_memory(
        rig, monkeypatch):
    """The orphan sentence survives — but only where it is TRUE: an occupied
    card whose pid log accounts for none of it (the k30 shape)."""
    from abstract_hugpy_dev.worker_agent import pid_registry as PR
    monkeypatch.setattr(PR, "snapshot_for_heartbeat",
                        lambda: {"models": [], "unattributed": []})
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(4.0 * GIB)              # 19.6 GiB in use, 0 attributed
    rig.card["need"] = int(51.8 * GIB)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    msg = A._vram_evict_to_fit(_State(), "coder")["reason"]["reason"]
    assert "cannot map to a model_key" in msg


def test_orphan_message_cites_the_measured_unattributed_figure(rig, monkeypatch):
    """When memory really is unattributed, the refusal quotes the measured
    figure rather than asserting the whole occupancy is foreign."""
    from abstract_hugpy_dev.worker_agent import pid_registry as PR
    monkeypatch.setattr(PR, "snapshot_for_heartbeat", lambda: {
        "models": [], "unattributed": [{"pid": 9001, "name": "python", "mib": 6000}]})
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(4.0 * GIB)
    rig.card["need"] = int(51.8 * GIB)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    r = A._vram_evict_to_fit(_State(), "coder")["reason"]
    assert "cannot map to a model_key" in r["reason"]
    assert "measured UNATTRIBUTED" in r["reason"]
    assert r["vram_unattributed_bytes"] == 6000 * (1 << 20)


def test_attributed_occupancy_is_not_reported_as_an_orphan(rig, monkeypatch):
    """The ae contradiction, generalised: attribution says the card is fully
    accounted for (unattributed = 0), so the refusal must not claim a squatter.
    Here the attributed bytes are a model_key-less cuda_context lump, which
    `_vram_residents` skips — so candidates AND protected are empty without any
    orphan being involved."""
    from abstract_hugpy_dev.worker_agent import pid_registry as PR
    monkeypatch.setattr(PR, "snapshot_for_heartbeat", lambda: {
        "models": [{"model_key": None, "pid": 4242, "host_mode": "cuda_context",
                    "vram_bytes": int(19.6 * GIB), "alive": True}],
        "unattributed": []})
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(4.0 * GIB)
    rig.card["need"] = int(51.8 * GIB)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    r = A._vram_evict_to_fit(_State(), "coder")["reason"]
    assert "cannot map to a model_key" not in r["reason"]
    assert "nothing foreign is squatting" in r["reason"]
    assert r["vram_attributed_bytes"] == int(19.6 * GIB)
    assert r["vram_unattributed_bytes"] == 0


def test_unmeasurable_attribution_degrades_instead_of_naming_a_culprit(
        rig, monkeypatch):
    """DEGRADE-NOT-GUESS: an unreadable pid log must not be reported as either
    an orphan or a clean bill of health."""
    from abstract_hugpy_dev.worker_agent import pid_registry as PR
    monkeypatch.setattr(PR, "snapshot_for_heartbeat", lambda: None)
    rig.card["total"] = int(23.6 * GIB)
    rig.card["free"] = int(4.0 * GIB)
    rig.card["need"] = int(51.8 * GIB)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    r = A._vram_evict_to_fit(_State(), "coder")["reason"]
    assert "UNMEASURABLE" in r["reason"]
    assert "cannot map to a model_key" not in r["reason"]
    assert "vram_unattributed_bytes" not in r     # omitted, never fabricated


# ── anti-double-count: the credit must be memory that is REALLY available ────
def test_a_dead_resident_row_is_not_credited(rig, monkeypatch):
    """`vram-admission-no-evict` guard. A reaped/dead row's bytes are ALREADY
    back in the device's free figure; crediting them would count them twice and
    admit-then-OOM. Only live rows count."""
    monkeypatch.setattr(A, "_vram_residents", lambda s: [
        {"model_key": "ghost", "vram_bytes": 20 * GIB,
         "host_mode": "in_process", "alive": False}])
    assert A._subject_resident_vram_bytes(_State(), "ghost") == 0


def test_an_unjoinable_resident_row_credits_nothing(rig, monkeypatch):
    """A row we could not join to nvidia-smi (vram_bytes 0) is an occupant of
    UNKNOWN size — degrade-not-guess, credit nothing rather than invent."""
    monkeypatch.setattr(A, "_vram_residents", lambda s: [
        {"model_key": "unjoined", "vram_bytes": 0,
         "host_mode": "subprocess", "alive": True}])
    assert A._subject_resident_vram_bytes(_State(), "unjoined") == 0


def test_comfy_is_never_credited_to_a_subject(rig, monkeypatch):
    """comfy is out of allocations (0.1.137) and has its own headroom path — its
    bytes are never headroom for a model admission."""
    monkeypatch.setattr(A, "_vram_residents", lambda s: [
        {"model_key": "comfy-sdxl", "vram_bytes": 12 * GIB,
         "host_mode": "comfy", "alive": True}])
    assert A._subject_resident_vram_bytes(_State(), "comfy-sdxl") == 0


def test_credit_is_not_double_counted_against_the_eviction_need(rig):
    """The credit must shrink the deficit the eviction planner works to, not sit
    alongside it: with the subject holding 10G and 0 free, a 12G need is 2G
    short — the ceiling reserve pushes it to ~4.4G, which the 5G neighbour
    covers ALONE. Crediting twice (22G apparent) would evict nobody and OOM;
    not crediting at all would evict BOTH neighbours."""
    rig.card["total"] = 24 * GIB
    rig.card["free"] = 0
    rig.card["need"] = 12 * GIB
    rig.residents["subj"] = {"vram_bytes": 10 * GIB, "host_mode": "in_process"}
    rig.residents["n1"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.residents["n2"] = {"vram_bytes": 5 * GIB, "host_mode": "subprocess"}
    rig.lru.update(n1=1.0, n2=2.0)
    plan = A._vram_evict_to_fit(_State(), "subj")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["n1"]               # the MINIMUM set, still minimum
    assert "n2" not in plan["evicted"]
    assert "subj" not in rig.evicted


def test_partial_offload_budget_includes_the_subjects_own_bytes(rig, gguf_rig):
    """A GGUF re-seat that can't fit whole still gets the honest hybrid, sized
    against free + its own released bytes — not against free alone."""
    rig.card["total"] = 24 * GIB
    rig.card["free"] = 1 * GIB                     # 1 - 2.4 reserve => 0 budget
    rig.card["need"] = 52 * GIB
    rig.residents["coder"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), "coder")
    # Without the credit the budget is 0 -> degenerate -> refuse. With it the
    # budget is (1 + 20) - 2.4 = 18.6 GiB -> the same 17/48 hybrid as ever.
    assert plan["action"] == "partial"
    assert plan["n_gpu_layers"] == 17


# ── all five protection classes still hold WITH a resident subject ──────────
@pytest.mark.parametrize("klass", ["static", "replying", "busy_slot",
                                   "queued_ahead", "comfy"])
def test_every_protection_class_still_holds_with_a_credited_subject(
        rig, monkeypatch, klass):
    """The credit must not buy its way past any protection. Subject resident and
    credited, need still unsatisfiable, one protected neighbour each time: the
    neighbour is never evicted and the refusal still names why."""
    rig.card["total"] = 24 * GIB
    rig.card["free"] = 0
    rig.card["need"] = 40 * GIB                    # unsatisfiable either way
    rig.residents["subj"] = {"vram_bytes": 2 * GIB, "host_mode": "in_process"}
    rig.residents["neighbour"] = {
        "vram_bytes": 20 * GIB,
        "host_mode": "comfy" if klass == "comfy" else "subprocess"}
    if klass == "static":
        rig.static.add("neighbour")
    elif klass == "replying":
        rig.replying.add("neighbour")
    elif klass == "busy_slot":
        rig.busy_slots.add("neighbour")
    elif klass == "queued_ahead":
        monkeypatch.setattr(A, "_queued_ahead_of", lambda subj: {"neighbour"})
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    plan = A._vram_evict_to_fit(_State(), "subj")
    assert plan["action"] == "refuse"
    assert "neighbour" not in rig.evicted, f"{klass} protection was weakened"
    assert "subj" not in rig.evicted, "the subject protected itself"
    assert plan["reason"]["protected"], "the protected row must be reported"
