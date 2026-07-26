"""The two eviction knobs, on real switches (operator, 2026-07-25).

Both shipped env-only and were only reachable by editing a systemd drop-in.
This file covers putting them on settings/console switches:

  1. ANTI-THRASH FLOOR  ``evict_min_residency_s``  — PER-WORKER.
  2. LEAST REAPING      ``evict_least_reaping``    — FLEET-WIDE.

The scope split is the load-bearing decision and is asserted here, not just
documented: the floor is a VRAM-residency concept with no central counterpart
(both storage sites hardcode it to 0), while least-reaping gates the DROP PASS
that central's ``storage_proposal`` runs too — so a per-worker value would
break Parity. ``test_parity_holds_under_the_fleet_switch`` is the guard.

Run: venv/bin/python -m pytest tests/test_evict_policy_knobs.py -v
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.managers import eviction as ev  # noqa: E402
from abstract_hugpy_dev.worker_agent import agent, budget  # noqa: E402

GIB = 1 << 30
NOW = 1_000_000.0

_ENV_FLOOR = "HUGPY_EVICT_MIN_RESIDENCY_S"
_ENV_REAP = "HUGPY_EVICT_LEAST_REAPING"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Both knobs read the ENV, so every test starts from a known-absent one."""
    monkeypatch.delenv(_ENV_FLOOR, raising=False)
    monkeypatch.delenv(_ENV_REAP, raising=False)
    agent._SETTINGS_SOURCE.clear()
    agent._RUNTIME_SETTINGS.clear()
    yield


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE DROP PASS — the switch must actually change it, and OFF must be
#    byte-identical to the pre-drop-pass greedy walk.
# ─────────────────────────────────────────────────────────────────────────────
# Sized so the drop pass BITES: the walk takes small+big, but big alone covers
# the need, so least-reaping spares the small one. Without a fixture where the
# two answers differ, a broken switch would pass silently.
_DROP_RESIDENTS = [
    ev.Resident(model_key="small", bytes=5 * GIB, last_call=NOW - 9000, calls=0),
    ev.Resident(model_key="big", bytes=35 * GIB, last_call=NOW - 8000, calls=0),
]
_DROP_NEED = 15 * GIB


def test_least_reaping_on_drops_the_covered_victim():
    """ON (today's default): one 35 GiB unload satisfies a 15 GiB need."""
    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, min_residency_s=0.0, least_reaping=True)
    assert plan.victims == ["big"]
    assert plan.spared == ["small"]
    assert plan.freed == 35 * GIB


def test_least_reaping_off_keeps_the_whole_walk():
    """OFF: the greedy walk, nothing spared — MORE headroom, more unloads."""
    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, min_residency_s=0.0, least_reaping=False)
    assert plan.victims == ["small", "big"]
    assert plan.spared == []
    assert plan.freed == 40 * GIB          # the extra headroom the operator wanted


def test_off_is_byte_identical_to_the_pre_drop_pass_walk():
    """The OFF state must be the OLD behaviour EXACTLY, not merely similar.

    Recomputes the pre-drop-pass algorithm independently (sort by the spec key,
    walk until covered, keep everything walked) and asserts the switch reproduces
    it — victims, order, and freed bytes. This is the regression that would catch
    the switch accidentally changing the WALK or the SORT rather than only
    skipping the drop.
    """
    walkable = sorted(_DROP_RESIDENTS, key=lambda r: ev.sort_key(r, ev.VRAM, NOW))
    expected, freed = [], 0
    for r in walkable:
        if freed >= _DROP_NEED:
            break
        expected.append(r.model_key)
        freed += r.bytes

    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, min_residency_s=0.0, least_reaping=False)
    assert plan.victims == expected
    assert plan.freed == freed


def test_default_is_least_reaping_on():
    """Absent argument == today's behaviour. Defaults must not change behaviour."""
    assert ev.DEFAULT_LEAST_REAPING is True
    plan = ev.evict_plan(ev.VRAM, _DROP_NEED, _DROP_RESIDENTS,
                         now=NOW, min_residency_s=0.0)
    assert plan.victims == ["big"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. THE ANTI-THRASH FLOOR — and 0 genuinely disabling it.
# ─────────────────────────────────────────────────────────────────────────────
def _fresh_resident():
    """One model, resident 10s — inside any non-zero floor."""
    return [ev.Resident(model_key="fresh", bytes=10 * GIB, last_call=None,
                        calls=0, resident_since=NOW - 10)]


def test_floor_removes_a_fresh_model_from_the_pool():
    plan = ev.evict_plan(ev.VRAM, 5 * GIB, _fresh_resident(),
                         now=NOW, min_residency_s=300.0)
    assert plan.victims == []
    assert not plan.enough                    # the admission REFUSES
    assert any("minimum residency" in b["why"] for b in plan.blocking)


def test_floor_zero_genuinely_disables_it():
    """0 is the ESCAPE HATCH — it must restore the pre-2026-07-25 behaviour."""
    plan = ev.evict_plan(ev.VRAM, 5 * GIB, _fresh_resident(),
                         now=NOW, min_residency_s=0.0)
    assert plan.victims == ["fresh"]
    assert plan.enough


def test_floor_reader_env_and_default():
    assert agent._evict_min_residency_s() == ev.DEFAULT_MIN_RESIDENCY_S
    os.environ[_ENV_FLOOR] = "42"
    assert agent._evict_min_residency_s() == 42.0
    os.environ[_ENV_FLOOR] = "0"
    assert agent._evict_min_residency_s() == 0.0       # 0 survives as 0
    os.environ[_ENV_FLOOR] = "nonsense"
    assert agent._evict_min_residency_s() == ev.DEFAULT_MIN_RESIDENCY_S


def test_least_reaping_reader_env_and_default():
    assert agent._evict_least_reaping() is True
    for falsy in ("0", "false", "FALSE", "no", "off"):
        os.environ[_ENV_REAP] = falsy
        assert agent._evict_least_reaping() is False, falsy
    for truthy in ("1", "true", "yes", "on"):
        os.environ[_ENV_REAP] = truthy
        assert agent._evict_least_reaping() is True, truthy


def test_budget_reader_matches_the_agent_reader():
    """budget.py must parse the shared env EXACTLY as agent.py does — they are
    two readers of one contract, and a divergence would split the fleet's
    storage half from its VRAM half."""
    for val in ("0", "false", "off", "1", "true", ""):
        os.environ[_ENV_REAP] = val
        assert budget._least_reaping() == agent._evict_least_reaping(), val
    del os.environ[_ENV_REAP]
    assert budget._least_reaping() == agent._evict_least_reaping()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SETTINGS ROUND-TRIP + SOURCE PRECEDENCE (settings > env > default).
# ─────────────────────────────────────────────────────────────────────────────
class _Args:
    def __init__(self, path):
        self.settings_file = str(path)


def _apply(tmp_path, settings, monkeypatch, env=None):
    """Write a settings file, project it, and return (floor, least_reaping)."""
    import json
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    # The base-sentinels are captured once per boot; clear them so each call is
    # an independent "boot" rather than inheriting the previous test's base.
    for k in list(os.environ):
        if k.startswith("_HUGPY_BASE_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(agent, "_load_settings", lambda _a: dict(settings))
    agent._apply_settings_env(_Args(p))
    return agent._evict_min_residency_s(), agent._evict_least_reaping()


def test_settings_roundtrip_through_apply(tmp_path, monkeypatch):
    floor, reap = _apply(tmp_path, {"evict_min_residency_s": 120.0,
                                    "evict_least_reaping": False}, monkeypatch)
    assert floor == 120.0
    assert reap is False
    assert agent._SETTINGS_SOURCE["evict_min_residency_s"] == "settings"
    assert agent._SETTINGS_SOURCE["evict_least_reaping"] == "settings"


def test_setting_zero_survives_projection(tmp_path, monkeypatch):
    """The escape hatch must survive the projector.

    ``0`` and ``False`` are the two legitimately-FALSY projected values in the
    whole settings file. A truthiness guard in _apply_settings_env would drop
    exactly the two the operator most needs — hence this test.
    """
    floor, reap = _apply(tmp_path, {"evict_min_residency_s": 0.0,
                                    "evict_least_reaping": False}, monkeypatch)
    assert floor == 0.0
    assert reap is False
    assert agent._SETTINGS_SOURCE["evict_min_residency_s"] == "settings"


def test_settings_beat_the_env_dropin(tmp_path, monkeypatch):
    floor, reap = _apply(tmp_path, {"evict_min_residency_s": 77.0,
                                    "evict_least_reaping": False},
                         monkeypatch, env={_ENV_FLOOR: "900", _ENV_REAP: "1"})
    assert floor == 77.0                     # setting, not the 900s drop-in
    assert reap is False


def test_env_wins_when_no_setting(tmp_path, monkeypatch):
    floor, _ = _apply(tmp_path, {}, monkeypatch, env={_ENV_FLOOR: "900"})
    assert floor == 900.0
    assert agent._SETTINGS_SOURCE["evict_min_residency_s"] == "env"


def test_default_when_neither(tmp_path, monkeypatch):
    floor, reap = _apply(tmp_path, {}, monkeypatch)
    assert floor == ev.DEFAULT_MIN_RESIDENCY_S
    assert reap is ev.DEFAULT_LEAST_REAPING
    assert agent._SETTINGS_SOURCE["evict_min_residency_s"] == "default"
    assert agent._SETTINGS_SOURCE["evict_least_reaping"] == "default"


def test_cleared_setting_reverts_to_the_dropin_base(tmp_path, monkeypatch):
    """A CLEAR must revert to the true drop-in base, not leak the last
    projected value across the re-exec (the COMFY_URL sentinel dance)."""
    monkeypatch.setenv(_ENV_FLOOR, "900")
    _apply(tmp_path, {"evict_min_residency_s": 77.0}, monkeypatch)
    assert agent._evict_min_residency_s() == 77.0
    # Second "boot" with the setting gone, sentinels intact (no monkeypatch
    # clearing this time) — the base must come back.
    monkeypatch.setattr(agent, "_load_settings", lambda _a: {})
    agent._apply_settings_env(_Args(tmp_path / "settings.json"))
    assert agent._evict_min_residency_s() == 900.0
    assert agent._SETTINGS_SOURCE["evict_min_residency_s"] == "env"


def test_only_the_per_worker_key_is_in_the_whitelist():
    """RENAMED + INVERTED (operator ruling 2026-07-25: "yes reject it").

    This previously asserted BOTH keys were accepted, which encoded the
    accept-then-silently-override behaviour the ruling removed: a worker stored
    evict_least_reaping, reported it back as saved, and central's heartbeat
    overwrote it on the next beat. The floor stays — it is genuinely per-worker.
    """
    assert "evict_min_residency_s" in agent._SETTINGS_KEYS
    assert "evict_least_reaping" not in agent._SETTINGS_KEYS


def test_unknown_key_is_still_rejected():
    """The strict whitelist must not have been loosened to admit the new keys."""
    assert "evict_min_residency" not in agent._SETTINGS_KEYS   # near-miss typo
    assert "totally_made_up" not in agent._SETTINGS_KEYS


# ─────────────────────────────────────────────────────────────────────────────
# 4. FLEET ADOPTION — the heartbeat path for the fleet-wide knob.
# ─────────────────────────────────────────────────────────────────────────────
def test_adopt_least_reaping_from_the_heartbeat():
    agent._adopt_least_reaping({"evict_least_reaping": False})
    assert agent._evict_least_reaping() is False
    agent._adopt_least_reaping({"evict_least_reaping": True})
    assert agent._evict_least_reaping() is True


def test_absent_key_does_not_clobber_a_local_dropin(monkeypatch):
    """Central with NO opinion must leave a local drop-in alone — absence means
    'no ruling', never 'force the default'."""
    monkeypatch.setenv(_ENV_REAP, "0")
    monkeypatch.setenv(f"_HUGPY_BASE_{_ENV_REAP}", "0")
    agent._adopt_least_reaping({})                     # no key on the reply
    assert agent._evict_least_reaping() is False       # drop-in survived


def test_adoption_never_raises_on_junk():
    """A heartbeat must never die on policy adoption."""
    for junk in (None, {}, {"evict_least_reaping": object()}):
        agent._adopt_least_reaping(junk)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PARITY — the invariant the scope decision exists to protect.
# ─────────────────────────────────────────────────────────────────────────────
def test_parity_holds_under_the_fleet_switch(monkeypatch):
    """Central's preview and the worker's storage auto-evict must name the SAME
    victims in BOTH switch states.

    This is the assertion that justifies making least-reaping fleet-wide: the
    two sites are driven over ONE fixture with the policy flipped, and the
    victim lists must match each time. If the knob were per-worker, this is the
    test that would fail the moment two boxes disagreed.
    """
    from test_eviction_parity import _central_victims, _worker_victims
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import (
        workers as central)

    for state in ("1", "0"):
        monkeypatch.setenv(_ENV_REAP, state)
        # Patch the module OBJECT, not a dotted string: the package has a
        # `functions.functions` name collision that breaks string resolution.
        monkeypatch.setattr(central, "_fleet_least_reaping",
                            (lambda s=state: s == "1"))
        assert _worker_victims() == _central_victims(), (
            f"victim sets diverged with least_reaping={state}")


def test_the_floor_is_zero_on_both_storage_sites():
    """The scope decision's OTHER half, asserted rather than assumed.

    The anti-thrash floor is safe to make per-worker precisely BECAUSE neither
    storage site consults it — both hardcode 0, so a per-worker floor can never
    desynchronise central's preview from a worker's execution. If someone later
    wires the floor into a storage site, this fails and the scope decision must
    be revisited.
    """
    import inspect
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers
    for src in (inspect.getsource(budget.fit_plan),
                inspect.getsource(workers.storage_proposal)):
        assert "min_residency_s=0.0" in src


# ── the operator's ruling: a fleet key is REJECTED, and says where to go ─────
def test_fleet_only_key_is_rejected_by_ops_config_with_the_right_door():
    """evict_least_reaping must NOT be a per-worker setting.

    Operator ruling 2026-07-25 ("yes reject it... best to have these decisions
    explicit in proof of action"). It was briefly accepted so the relay stayed
    uniform — but it is FLEET policy, so a worker would store it, report it back
    as saved, and have central's heartbeat overwrite it on the next beat: a
    setting you can write, that reads back correct, and that silently stops
    mattering.

    That is the same shape as the max-gpu bug found earlier the SAME DAY, where
    /assign returned "admission: approved" for a value it never persisted. So
    the rejection is not a bare "unsupported" — it names the route that DOES own
    the knob, because the operator asked for a real thing at the wrong door.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from abstract_hugpy_dev.worker_agent import agent as A

    assert "evict_least_reaping" not in A._SETTINGS_KEYS, (
        "fleet policy must not be a per-worker setting key")
    assert "evict_min_residency_s" in A._SETTINGS_KEYS, (
        "the anti-thrash floor IS genuinely per-worker and must stay accepted")

    # The variable is the declaration — one place, extensible, no toggle.
    assert "evict_least_reaping" in A._FLEET_ONLY_SETTINGS
    where = A._FLEET_ONLY_SETTINGS["evict_least_reaping"]
    assert "/llm/evict-policy" in where, (
        "the rejection must name the route that owns the knob, not just refuse")
