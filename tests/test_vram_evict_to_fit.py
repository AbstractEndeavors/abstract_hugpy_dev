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
