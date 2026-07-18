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
