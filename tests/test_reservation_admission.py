"""Reservation-GATED ADMISSION — the non-destructive fit PROBE (engine.can_admit /
force_admit_safe) and the media_bus scheduler that gates claiming on it (holds a
head that can't fit, overtakes it with a later job that fits, bounds the overtake
so the head can't starve, and is a transparent FIFO no-op when the layer is off).

No GPU, no network: the fleet read + registry are stubbed with a mutable fake
worker (engine layer), and the scheduler layer stubs the probe directly so the
overtake/starvation ALGORITHM is asserted deterministically.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_reservation_admission.py -q
"""
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Isolate the registry singleton + force estimates before import (all read live).
_D = tempfile.mkdtemp(prefix="hugpy-resv-adm-")
os.environ["HUGPY_RESERVATIONS_DB"] = os.path.join(_D, "resv.db")
os.environ["HUGPY_RESERVATIONS_MEASURED"] = os.path.join(_D, "nope.json")
os.environ.pop("HUGPY_STUDIO_WORKER", None)
os.environ.pop("IDENTITY_RENDER_URL", None)
os.environ.pop("HUGPY_RESERVATIONS", None)          # default ON
os.environ.pop("HUGPY_MEDIA_BUS_LOOKAHEAD", None)
os.environ.pop("HUGPY_MEDIA_BUS_MAX_OVERTAKE", None)

import pytest  # noqa: E402

from abstract_hugpy_dev.video_intel.reservation import engine as E  # noqa: E402
from abstract_hugpy_dev.video_intel.reservation.registry import (  # noqa: E402
    reservation_registry as REG)
from abstract_hugpy_dev.video_intel import media_bus as MB  # noqa: E402
from abstract_hugpy_dev.video_intel.job_schema import JobSpec  # noqa: E402
from abstract_hugpy_dev.video_intel.result_schema import JobResult  # noqa: E402

# Repoint the process-wide registry singleton at our scratch DB regardless of
# import order (a sibling test file may have constructed it already).
REG.path = os.environ["HUGPY_RESERVATIONS_DB"]
REG._initialized = False

_GIB = 1024 ** 3


# ─────────────────────────── engine.can_admit probe ──────────────────────────
def _worker(free_gib, residents):
    return {
        "id": "ae", "name": "ae", "status": "online", "url": "http://ae:9100",
        "gpus": [{"memory_total": 24 * _GIB, "memory_free": int(free_gib * _GIB)}],
        "pid_registry": {"models": [dict(alive=True, **r) for r in residents]},
    }


def test_probe_admits_when_free_plus_makeroom_reaches_peak(monkeypatch):
    # Card mostly free — a Wan (20 GB) run's peak fits without touching a reservation.
    w = _worker(free_gib=22, residents=[])
    monkeypatch.setattr(E, "_list_workers", lambda: [w])
    admit, reason = E.can_admit("studio_i2v", None, run_id="probe-fit")
    assert admit is True and reason is None


def test_probe_counts_make_room_headroom_optimistically(monkeypatch):
    # Only 6 GB physically free, but an 18 GB evictable brain — make-room WOULD
    # reach the 20 GB peak, so the probe ADMITS (never HOLDS a render make-room
    # could clear).
    w = _worker(free_gib=6, residents=[
        {"model_key": "Qwen~brain", "host_mode": "subprocess", "vram_bytes": 18 * _GIB}])
    monkeypatch.setattr(E, "_list_workers", lambda: [w])
    admit, reason = E.can_admit("studio_i2v", None)
    assert admit is True and reason is None


def test_probe_holds_when_capacity_is_reserved_by_an_active_run(monkeypatch):
    # A first heavy run holds an ACTIVE reservation (20 GB). While it renders the
    # card is occupied (free low, nothing evictable) → a SECOND heavy run's probe
    # HOLDS (would collide) with an honest reason.
    monkeypatch.setattr(E, "_list_workers",
                        lambda: [_worker(free_gib=3, residents=[])])
    REG.claim("run-active", "ae", "ae", "studio_i2v", 20 * _GIB)
    try:
        admit, reason = E.can_admit("generate_studio_movie", None, run_id="probe-2nd")
        assert admit is False
        assert reason["peak_bytes"] == 20 * _GIB
        assert reason["reserved_bytes"] == 20 * _GIB
        assert reason["short_by_bytes"] > 0
    finally:
        REG.release("run-active")


def test_probe_fails_open_for_light_task_disabled_and_blind_fleet(monkeypatch):
    # Light task (no template) → admit.
    assert E.can_admit("generate_image", None) == (True, None)
    # Unresolvable fleet → fail open.
    monkeypatch.setattr(E, "_list_workers", lambda: [])
    assert E.can_admit("studio_i2v", None) == (True, None)
    # Layer disabled → transparent admit.
    monkeypatch.setenv("HUGPY_RESERVATIONS", "off")
    assert E.can_admit("studio_i2v", None) == (True, None)


def test_force_admit_safe_light_always_heavy_only_when_unreserved(monkeypatch):
    monkeypatch.setattr(E, "_list_workers",
                        lambda: [_worker(free_gib=3, residents=[])])
    assert E.force_admit_safe("generate_image") is True     # light — always safe
    assert E.force_admit_safe("studio_i2v") is True         # heavy, nothing reserved
    REG.claim("run-hot", "ae", "ae", "studio_i2v", 20 * _GIB)
    try:
        assert E.force_admit_safe("studio_i2v") is False    # heavy, card reserved
    finally:
        REG.release("run-hot")


# ─────────────────────── media_bus scheduler (algorithm) ──────────────────────
@dataclass(frozen=True)
class _FakeSpec:
    tag: str = "x"


@pytest.fixture
def bus(tmp_path, monkeypatch):
    """A private media bus DB + two throwaway job kinds: a 'fake_heavy' (reservable-
    like, gated by the stubbed probe) and a 'fake_light' (always admits)."""
    monkeypatch.setattr(MB, "DB_PATH", str(tmp_path / "media_jobs.db"))
    monkeypatch.setattr(MB, "_initialized", False)
    for kind in ("fake_heavy", "fake_light"):
        monkeypatch.setitem(MB.JOB_REGISTRY, kind,
                            JobSpec(kind, _FakeSpec, (kind,), "run", 60))
        monkeypatch.setitem(MB.SPEC_DESERIALIZERS, kind, lambda d: _FakeSpec(**d))
    # Neutralize the JobStore bridge so scheduler tests don't touch comms.
    monkeypatch.setattr(MB, "_bridge", lambda *a, **k: None)
    yield


def _gate(monkeypatch, admissible):
    """Stub the admission probe: names in ``admissible`` admit, others HOLD."""
    monkeypatch.setattr(MB, "_admission_enabled", lambda: True)

    def _probe(name, job_id):
        if name in admissible:
            return True, None
        return False, {"reason": "held (test)", "peak_bytes": 20 * _GIB}

    monkeypatch.setattr(MB, "_probe_admission", _probe)


def test_disabled_layer_is_transparent_fifo(bus, monkeypatch):
    # Layer OFF → claim_admissible is pure FIFO (oldest first), no probe at all.
    monkeypatch.setattr(MB, "_admission_enabled", lambda: False)
    called = {"probe": 0}
    monkeypatch.setattr(MB, "_probe_admission",
                        lambda *a: (called.__setitem__("probe", called["probe"] + 1), (True, None))[1])
    a = MB.enqueue("fake_light", _FakeSpec(tag="a"))
    b = MB.enqueue("fake_light", _FakeSpec(tag="b"))
    assert MB.claim_admissible("w1") == a          # oldest first
    assert MB.claim_admissible("w2") == b
    assert called["probe"] == 0                    # gate never consulted when OFF


def test_head_holds_and_later_job_overtakes(bus, monkeypatch):
    # Head is a heavy job that can't fit; a later LIGHT job can → the claimer holds
    # the head (marks awaiting_capacity) and claims the light job past it.
    _gate(monkeypatch, admissible={"fake_light"})
    monkeypatch.setattr(MB, "_force_admit_safe", lambda name: False)  # heavy: unsafe
    head = MB.enqueue("fake_heavy", _FakeSpec(tag="head"))
    light = MB.enqueue("fake_light", _FakeSpec(tag="light"))
    got = MB.claim_admissible("w1")
    assert got == light                            # overtook the held head
    view = MB.get(head)
    assert view["status"] == "queued"              # head still queued (cancellable)
    assert view["progress"]["phase"] == "awaiting_capacity"
    assert view["progress"]["overtaken"] == 1
    assert view["progress"]["reason"]["reason"] == "held (test)"


def test_held_head_stays_cancellable(bus, monkeypatch):
    _gate(monkeypatch, admissible={"fake_light"})
    monkeypatch.setattr(MB, "_force_admit_safe", lambda name: False)
    head = MB.enqueue("fake_heavy", _FakeSpec(tag="head"))
    MB.enqueue("fake_light", _FakeSpec(tag="light"))
    MB.claim_admissible("w1")                       # holds head, runs light
    res = MB.cancel(head)                           # a held (queued) job cancels cleanly
    assert res["cancelled"] is True and res["status"] == "cancelled"
    assert MB.get(head)["status"] == "cancelled"


def test_lone_unfittable_head_force_admits_when_safe(bus, monkeypatch):
    # A lone head that can't fit but whose force-admit is SAFE (no active
    # reservation to collide with — it will offload) is force-admitted best-effort
    # rather than held forever.
    _gate(monkeypatch, admissible=set())            # nothing admits via the probe
    monkeypatch.setattr(MB, "_force_admit_safe", lambda name: True)
    head = MB.enqueue("fake_heavy", _FakeSpec(tag="lonely"))
    assert MB.claim_admissible("w1") == head        # force-admitted best-effort


def test_head_holds_and_idles_when_force_admit_unsafe(bus, monkeypatch):
    # Head can't fit, no later job fits, and force-admit is UNSAFE (an active
    # reservation occupies the card) → the claimer IDLES (returns None) and keeps
    # the head held, waiting for the reservation to release. Never OOMs a 2nd heavy.
    _gate(monkeypatch, admissible=set())
    monkeypatch.setattr(MB, "_force_admit_safe", lambda name: False)
    head = MB.enqueue("fake_heavy", _FakeSpec(tag="head"))
    assert MB.claim_admissible("w1") is None         # idle — nothing safe to start
    assert MB.get(head)["status"] == "queued"        # still held, still cancellable


def test_overtake_is_bounded_then_claimer_idles(bus, monkeypatch):
    # Anti-starvation: once the head has been overtaken MAX times, the claimer
    # STOPS overtaking (idles) so throughput can't jump the head forever.
    monkeypatch.setenv("HUGPY_MEDIA_BUS_MAX_OVERTAKE", "2")
    monkeypatch.setenv("HUGPY_MEDIA_BUS_LOOKAHEAD", "20")
    _gate(monkeypatch, admissible={"fake_light"})
    monkeypatch.setattr(MB, "_force_admit_safe", lambda name: False)
    head = MB.enqueue("fake_heavy", _FakeSpec(tag="head"))
    lights = [MB.enqueue("fake_light", _FakeSpec(tag=f"l{i}")) for i in range(5)]
    # Two overtakes are allowed...
    g1 = MB.claim_admissible("w1")
    g2 = MB.claim_admissible("w2")
    assert g1 == lights[0] and g2 == lights[1]
    assert MB.get(head)["progress"]["overtaken"] == 2
    # ...the third claim finds the head at the cap → refuses to overtake, idles.
    assert MB.claim_admissible("w3") is None
    # Head is still queued; the remaining lights were NOT drained past the cap.
    assert MB.get(head)["status"] == "queued"
    assert MB.get(lights[2])["status"] == "queued"


# ───────────────────────────── pool concurrency ──────────────────────────────
def test_pool_runs_two_light_jobs_concurrently(bus, monkeypatch):
    # start_worker_daemon spawns N runner threads (default 2); two light jobs whose
    # runners block on a barrier must be IN-FLIGHT simultaneously — proving the pool
    # is concurrent, not the old single serial daemon.
    _gate(monkeypatch, admissible={"fake_light"})
    monkeypatch.setenv("HUGPY_MEDIA_BUS_RUNNERS", "2")
    barrier = threading.Barrier(2, timeout=5)

    def runner(spec, job_id):
        barrier.wait()                     # both must arrive → proves concurrency
        return JobResult(job_id=job_id, ok=True)

    monkeypatch.setitem(MB.DISPATCH, ("fake_light",), runner)
    a = MB.enqueue("fake_light", _FakeSpec(tag="a"))
    b = MB.enqueue("fake_light", _FakeSpec(tag="b"))
    stop = threading.Event()
    threads = MB.start_worker_daemon(idle_sleep_s=0.02, stop_event=stop)
    try:
        assert len(threads) == 2
        deadline = time.time() + 6
        while time.time() < deadline:
            if MB.get(a)["status"] == "done" and MB.get(b)["status"] == "done":
                break
            time.sleep(0.05)
        assert MB.get(a)["status"] == "done"
        assert MB.get(b)["status"] == "done"   # both done → barrier released → concurrent
    finally:
        # Stop the pool + join so no zombie runner thread survives to hammer a
        # later test's DB (daemon threads outlive the test otherwise).
        stop.set()
        for t in threads:
            t.join(timeout=3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
