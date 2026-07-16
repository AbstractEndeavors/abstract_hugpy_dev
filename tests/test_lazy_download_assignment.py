"""Lazy download: assignment is ATTRIBUTION, not a transfer order (2026-07-16).

Operator ruling: "models are attributed to be routed to a worker though not
immediately downloaded to the workers drive, they should be lazy download
instead downloading to the drive only when called".

Before this slice, ``_sync_assignment`` kicked a provision for EVERY assigned
model, and ``_reconcile_loop`` re-kicked any assigned model missing on disk
every 600s. That equated assignment with download and produced the 2026-07-15
storm: N assigned models = N parallel provisions, central 503'ing, and four
truncated GGUFs (~10.7GB) left on computron — all "designated" in
worker_assignments.json.

Now the default (on-demand) tier downloads only when CALLED, via the inference
path's already-working ``_ensure_present``. Exactly two tiers still pre-pull,
because lazy would break a promise they already make:
  * static  — operator-locked 2026-07-05 as "eager-warmed"
  * pinned  — 📌 permanent attribution

These tests patch ``_kick_provision`` at the agent module and assert on WHO
gets kicked; the download machinery itself is out of scope here (covered by
tests/test_provision_concurrency_gate.py). ``_fill_empty_slots`` is patched
too — it is a SEATING concern, not a download, and must keep running for
already-local models of every tier.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_lazy_download_assignment.py -q
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from abstract_hugpy_dev.worker_agent import agent as A


@pytest.fixture
def kicks(monkeypatch):
    """Record _kick_provision calls instead of downloading anything."""
    seen: list[str] = []
    monkeypatch.setattr(A, "_kick_provision", lambda state, mk: seen.append(mk))
    return seen


@pytest.fixture
def fills(monkeypatch):
    """Record _fill_empty_slots calls (seating, NOT downloading)."""
    seen: list[bool] = []
    monkeypatch.setattr(A, "_fill_empty_slots", lambda state: seen.append(True))
    return seen


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    """Isolated residency/pin settings; restored after each test."""
    monkeypatch.setattr(A, "_RUNTIME_SETTINGS", {
        "residency": {"m-static": "static"},
        "pinned": {"m-pinned": True},
    })
    # Prune reads/writes the settings file via state.args — keep it inert so
    # these tests exercise adoption only.
    monkeypatch.setattr(A, "_prune_stale_residency", lambda state: None)


def _state(assigned=None) -> A.WorkerState:
    st = A.WorkerState(name="test-worker", url=None, worker_id="w-lazy",
                       central_url=None)
    st.assigned_models = list(assigned or [])
    return st


def _wait_until(cond, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# ── the tier predicate ──────────────────────────────────────────────────────
def test_eager_pull_only_for_static_and_pinned():
    assert A._eager_pull("m-static") is True
    assert A._eager_pull("m-pinned") is True
    assert A._eager_pull("m-ondemand") is False   # the DEFAULT


def test_eager_pull_fails_lazy_when_settings_read_raises(monkeypatch):
    """A broken settings read must not resurrect the storm."""
    def _boom(_mk):
        raise RuntimeError("settings unreadable")
    monkeypatch.setattr(A, "_residency", _boom)
    assert A._eager_pull("m-whatever") is False


# ── (1) assignment of an on-demand model does NOT download ──────────────────
def test_assigning_on_demand_model_does_not_kick_provision(kicks, fills):
    A._sync_assignment(_state(), {"models": ["m-ondemand"]})
    assert kicks == [], "assignment alone must not download an on-demand model"


def test_assigning_many_on_demand_models_kicks_nothing(kicks, fills):
    """The storm shape: a big list must produce ZERO provisions."""
    models = [f"m-bulk-{i}" for i in range(30)]
    A._sync_assignment(_state(), {"models": models})
    assert kicks == []


# ── (2)/(3) static and pinned still pre-pull ────────────────────────────────
def test_assigning_static_model_kicks_provision(kicks, fills):
    A._sync_assignment(_state(), {"models": ["m-static"]})
    assert kicks == ["m-static"]


def test_assigning_pinned_model_kicks_provision(kicks, fills):
    A._sync_assignment(_state(), {"models": ["m-pinned"]})
    assert kicks == ["m-pinned"]


def test_mixed_assignment_pulls_only_the_eager_tiers(kicks, fills):
    A._sync_assignment(
        _state(), {"models": ["m-ondemand", "m-static", "m-other", "m-pinned"]})
    assert kicks == ["m-static", "m-pinned"]


# ── (5) seating still runs — it is not a download ───────────────────────────
def test_fill_empty_slots_runs_on_assignment_change(kicks, fills):
    """Already-local models must still get seated, on-demand included."""
    A._sync_assignment(_state(), {"models": ["m-ondemand"]})
    assert _wait_until(lambda: fills == [True]), \
        "_fill_empty_slots must still run on an assignment change"


def test_unchanged_assignment_does_not_refill_or_kick(kicks, fills):
    st = _state(assigned=["m-static"])
    A._sync_assignment(st, {"models": ["m-static"]})   # same list -> no change
    assert kicks == []
    assert fills == []


def test_response_without_models_list_is_not_read_as_unassign(kicks, fills):
    """The partial-response guard: no 'models' key -> adopt nothing."""
    st = _state(assigned=["m-static"])
    A._sync_assignment(st, {"status": "ok"})
    assert st.assigned_models == ["m-static"], "must not clear the assignment"
    assert kicks == []


# ── (4) reconcile: absent on-demand is the resting state, not drift ─────────
@pytest.fixture
def _fast_reconcile(monkeypatch):
    """Run exactly one reconcile iteration, then break the loop.

    _reconcile_loop sleeps >=60s and never returns, so drive it by stubbing
    sleep and raising out of the second pass via restart_requested().
    """
    calls = {"n": 0}

    def _sleep(_secs):
        calls["n"] += 1

    monkeypatch.setattr(A.time, "sleep", _sleep)
    # first check (after sleep) False -> body runs; second pass True -> return
    flags = iter([False, True])
    monkeypatch.setattr(A, "restart_requested", lambda: next(flags, True))
    return calls


def test_reconcile_does_not_rekick_missing_on_demand_model(
        kicks, _fast_reconcile, monkeypatch):
    monkeypatch.setattr(A, "_models_local", lambda state: [])   # nothing on disk
    A._reconcile_loop(_state(assigned=["m-ondemand"]))
    assert kicks == [], \
        "an assigned-but-absent on-demand model is the CORRECT resting state"


def test_reconcile_rekicks_missing_static_and_pinned(
        kicks, _fast_reconcile, monkeypatch):
    monkeypatch.setattr(A, "_models_local", lambda state: [])
    A._reconcile_loop(_state(assigned=["m-ondemand", "m-static", "m-pinned"]))
    assert kicks == ["m-static", "m-pinned"]


def test_reconcile_does_not_rekick_a_model_already_on_disk(
        kicks, _fast_reconcile, monkeypatch):
    monkeypatch.setattr(A, "_models_local", lambda state: ["m-static"])
    A._reconcile_loop(_state(assigned=["m-static"]))
    assert kicks == []


def test_reconcile_respects_the_in_flight_provisioning_guard(
        kicks, _fast_reconcile, monkeypatch):
    monkeypatch.setattr(A, "_models_local", lambda state: [])
    st = _state(assigned=["m-static"])
    st._provisioning.add("m-static")     # already being pulled
    A._reconcile_loop(st)
    assert kicks == []
