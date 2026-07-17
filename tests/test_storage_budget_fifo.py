"""Storage budget: evict-to-fit (FIFO), or REFUSE the pull before it starts.

The incident (2026-07-16): the operator's workstation "op" filled to 0 bytes
free. provision.py had NO disk checks — it downloaded until [Errno 28].

The operator's design, which these tests hold to:
  1. over the central-allocated budget -> FIFO the models
  2. "remove an existing model and install the one that is being called"
     (the CALLER always wins)
  3. "refuse it if it wont, show it as missing, hover info why"

⚠ This codebase has TWICE shipped tests that asserted the bug. So these tests
assert BEHAVIOR, not implementation: eviction ORDER is asserted explicitly, the
refusal asserts the download NEVER STARTED (by spying on the actual transfer
function), and the no-thrash case asserts NOTHING is evicted.

Run: venv/bin/python -m pytest tests/test_storage_budget_fifo.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.worker_agent import budget  # noqa: E402

GIB = 1 << 30


def _model(key, gib, last_picked=None, **flags):
    """One storage-view row, as the worker's heartbeat reports it."""
    row = {"model_key": key, "bytes": int(gib * GIB), "protected": False,
           "why": "", "pinned": False, "loaded": False, "loading": False,
           "provisioning": False, "assigned": False}
    row.update(flags)
    return row


def _static(key, gib, last_picked=None, **flags):
    """A 🔒static row — the one tier that blocks disk eviction. Used where a test
    needs an UNRECLAIMABLE model to force a refusal (📌pin no longer protects
    files as of 2026-07-17, so pinned rows are eviction candidates)."""
    flags.setdefault("protected", True)
    flags.setdefault("why", "static")
    return _model(key, gib, last_picked=last_picked, **flags)


def _storage(models):
    return {"cache_used_bytes": sum(m["bytes"] for m in models),
            "models": models, "disk_free": 10 * GIB}


# ── 1. fits under budget -> no eviction, proceeds ───────────────────────────
def test_pull_that_fits_evicts_nothing_and_proceeds():
    storage = _storage([_model("old", 10), _model("warm", 10)])
    plan = budget.fit_plan("newbie", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {"old": 1, "warm": 100})
    assert plan["action"] == "proceed"
    assert plan["evict"] == []


def test_exactly_filling_the_budget_still_proceeds():
    """Boundary: used + need == cap must FIT (<=), not trip an off-by-one."""
    storage = _storage([_model("a", 30)])
    plan = budget.fit_plan("newbie", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {"a": 1})
    assert plan["action"] == "proceed"


# ── 2. doesn't fit, enough reclaimable -> FIFO oldest-first, then install ───
def test_fifo_evicts_oldest_first_and_only_as_many_as_needed():
    # 45 GiB used of a 50 GiB budget; a 20 GiB pull needs 15 GiB freed.
    storage = _storage([
        _model("oldest", 10),   # last_picked 100 -> goes 1st
        _model("middle", 10),   # last_picked 200 -> goes 2nd
        _model("newest", 25),   # last_picked 300 -> must SURVIVE
    ])
    last_picked = {"oldest": 100, "middle": 200, "newest": 300}
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, last_picked)

    assert plan["action"] == "evict"
    # ORDER is the assertion: oldest-first, and it STOPS once it fits.
    assert plan["evict"] == ["oldest", "middle"]
    assert "newest" not in plan["evict"]
    assert plan["freed_bytes"] >= plan["must_free_bytes"]


def test_never_served_models_are_coldest_and_go_first():
    """A model central never served has no last_picked -> 0 -> evicted first.
    Exactly right for never-called test-churn leftovers."""
    storage = _storage([_model("served", 20), _model("never_served", 20)])
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {"served": 500})
    assert plan["action"] == "evict"
    assert plan["evict"][0] == "never_served"


def test_equally_cold_models_evict_largest_first():
    """Tiebreak among all-zero last_picked: biggest first, so the budget clears
    in the fewest deletes. Mirrors storage_proposal's sort exactly."""
    storage = _storage([_model("small", 5), _model("big", 30),
                        _model("mid", 15)])
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {})
    assert plan["action"] == "evict"
    assert plan["evict"][0] == "big"


# ── 3. never evicts protected models or the keep-target ────────────────────
# NOTE (operator ruling 2026-07-17): 📌pin is DELIBERATELY absent from this
# list — pin designates only that the allocation/routing survives restarts and
# has NO bearing on eviction. A pinned model's files are a normal candidate (see
# test_pinned_model_is_a_candidate_and_evicts below). Only 🔒static and the
# live-use guards (loaded/loading/provisioning) protect files here.
@pytest.mark.parametrize("flag", ["loaded", "loading", "provisioning"])
def test_protected_models_are_never_evicted(flag):
    storage = _storage([_model("protected_one", 40, **{flag: True}),
                        _model("cold", 20)])
    plan = budget.fit_plan("caller", 10 * GIB, storage,
                           {"disk_cache_gib": 50}, {"protected_one": 1,
                                                    "cold": 999})
    # protected_one is the OLDEST, so a policy ignoring guards would take it.
    assert "protected_one" not in plan["evict"]
    assert plan["evict"] == ["cold"]


def test_static_residency_is_never_evicted():
    """🔒static is the one tier that blocks eviction (pin-vs-static semantics)."""
    storage = _storage([_model("stat", 40, protected=True, why="static"),
                        _model("cold", 20)])
    plan = budget.fit_plan("caller", 10 * GIB, storage,
                           {"disk_cache_gib": 50}, {"stat": 1, "cold": 999})
    assert "stat" not in plan["evict"]


def test_pinned_model_is_a_candidate_and_evicts_when_fifo_reaches_it():
    """📌pin has NO bearing on eviction (operator, 2026-07-17): a pinned model's
    FILES are a normal FIFO candidate. Here the pinned model is the OLDEST, so it
    is evicted first — its pin/allocation are runtime state the fit_plan never
    touches, so nothing in the returned plan disturbs them."""
    storage = _storage([_model("pinned_old", 40, pinned=True),
                        _model("warm", 20)])
    plan = budget.fit_plan("caller", 10 * GIB, storage,
                           {"disk_cache_gib": 50},
                           {"pinned_old": 1, "warm": 999})
    assert plan["action"] == "evict"
    assert plan["evict"] == ["pinned_old"]   # pinned + oldest -> goes first
    # fit_plan is PURE — it names files to delete, never mutates pin/allocation.
    assert plan["reason"] is None


def test_pin_is_only_a_trivial_tiebreak_unpinned_evicts_first():
    """Pin's ONLY eviction role (operator called it "trivial and likely
    unnecessary"): at an EXACT last_picked tie, the UNPINNED candidate is evicted
    before the pinned one. Same size + same last_picked isolates the tiebreak."""
    storage = _storage([_model("pinned", 20, pinned=True),
                        _model("plain", 20, pinned=False)])
    # identical last_picked -> only the pin flag breaks the tie.
    plan = budget.fit_plan("caller", 15 * GIB, storage,
                           {"disk_cache_gib": 50},
                           {"pinned": 100, "plain": 100})
    assert plan["action"] == "evict"
    assert plan["evict"] == ["plain"]        # unpinned goes first
    assert "pinned" not in plan["evict"]


def test_the_model_being_provisioned_is_never_evicted():
    """model_cache.evict_for's keep_dir exclusion, by model_key: never evict
    the very thing we are making room for."""
    storage = _storage([_model("caller", 30, last_picked=1),
                        _model("other", 20)])
    plan = budget.fit_plan("caller", 30 * GIB, storage,
                           {"disk_cache_gib": 50}, {"caller": 1, "other": 999})
    assert "caller" not in plan["evict"]


def test_partial_bytes_of_the_keep_target_count_as_headroom():
    """A resumed pull already has bytes on disk; only the DELTA is new. Without
    this the check double-counts and refuses a pull that actually fits."""
    storage = _storage([_model("caller", 18), _model("other", 30)])
    # cap 50, used 48. A 20 GiB model with 18 already there needs only 2 more.
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {})
    assert plan["action"] == "proceed"


# ── 4. can't fit even after a full FIFO -> REFUSE, never download ──────────
def test_refuses_when_even_a_full_fifo_cannot_free_enough():
    # 🔒static (not pin) is the blocker now: pin no longer protects files, so a
    # pinned big model would simply be evicted and the pull would fit.
    storage = _storage([_static("static_big", 40),
                        _model("cold", 5)])
    plan = budget.fit_plan("huge", 30 * GIB, storage,
                           {"disk_cache_gib": 50}, {})
    assert plan["action"] == "refuse"
    assert plan["evict"] == []          # refuse -> delete NOTHING

    reason = plan["reason"]
    assert reason["state"] == "refused"
    # Honest + specific: needed, budget, reclaimable, and what blocked.
    assert "won't fit" in reason["reason"]
    assert reason["needs_bytes"] == 30 * GIB
    assert reason["budget_bytes"] == 50 * GIB
    assert reason["reclaimable_bytes"] == 5 * GIB
    assert reason["blocked"] == {"static": 1}
    assert "1 static" in reason["reason"]
    assert reason["shortfall_bytes"] > 0


def test_refusal_reason_is_human_readable_with_real_numbers():
    storage = _storage([_static("p1", 30),
                        _model("p2", 20, loaded=True)])
    plan = budget.fit_plan("big", 25 * GIB, storage, {"disk_cache_gib": 50}, {})
    assert plan["action"] == "refuse"
    text = plan["reason"]["reason"]
    assert "25.0 GB" in text and "50.0 GB" in text   # needs + budget
    assert "0 B reclaimable" in text
    assert "1 static" in text and "1 loaded" in text


def test_refused_pull_never_starts_the_download():
    """THE POINT OF THE FEATURE: no more 7%-wedged pulls that fill a disk.
    Spies on the real transfer functions — they must NEVER be called."""
    from abstract_hugpy_dev.worker_agent import provision

    calls = []

    class _FakeState:
        limits = {"disk_cache_gib": 50}
        model_last_picked = {}
        central_url = "http://central"
        refused: dict = {}

    # 🔒static blocker (pin no longer protects files) so the pull is REFUSED.
    storage = _storage([_static("static_big", 48)])

    orig_fetch = provision.fetch_from_central
    orig_arch = provision.fetch_archive_from_central
    orig_hf = provision.fetch_from_hf
    orig_local = provision.model_is_local
    orig_reg = provision.ensure_model_registered
    orig_size = provision.central_total_bytes
    orig_evict = budget.evict_to_fit
    try:
        provision.fetch_from_central = lambda *a, **k: calls.append("central")
        provision.fetch_archive_from_central = lambda *a, **k: calls.append("archive")
        provision.fetch_from_hf = lambda *a, **k: calls.append("hf")
        provision.model_is_local = lambda mk: False
        provision.ensure_model_registered = lambda mk, url: mk
        provision.central_total_bytes = lambda url, mk: 30 * GIB
        # Drive the REAL fit_plan through evict_to_fit's decision, without a disk.
        def _fake_evict(state, mk, need):
            plan = budget.fit_plan(mk, need, storage, state.limits,
                                   state.model_last_picked)
            if plan["action"] == "refuse":
                raise budget.BudgetRefusal(plan["reason"])
        budget.evict_to_fit = _fake_evict

        with pytest.raises(budget.BudgetRefusal) as err:
            provision.ensure_model_present("huge", "http://central",
                                           state=_FakeState())
        assert err.value.reason["state"] == "refused"
    finally:
        provision.fetch_from_central = orig_fetch
        provision.fetch_archive_from_central = orig_arch
        provision.fetch_from_hf = orig_hf
        provision.model_is_local = orig_local
        provision.ensure_model_registered = orig_reg
        provision.central_total_bytes = orig_size
        budget.evict_to_fit = orig_evict

    # The assertion that matters: NOT ONE BYTE was transferred.
    assert calls == []


def test_no_state_means_no_budget_check_pure_superset():
    """Standalone/CLI provisions (state=None) behave byte-for-byte as before."""
    from abstract_hugpy_dev.worker_agent import provision

    seen = []
    orig_local = provision.model_is_local
    orig_reg = provision.ensure_model_registered
    orig_now = provision._provision_now
    try:
        provision.model_is_local = lambda mk: False
        provision.ensure_model_registered = lambda mk, url: mk
        provision._provision_now = lambda *a, **k: (seen.append("pulled"), True)[1]
        assert provision.ensure_model_present("x", "http://central") is True
    finally:
        provision.model_is_local = orig_local
        provision.ensure_model_registered = orig_reg
        provision._provision_now = orig_now
    assert seen == ["pulled"]


# ── 5. disk_cache_gib unset -> sane fallback, NO thrash ────────────────────
def test_unset_disk_cache_gib_does_not_evict_anything():
    """D: op has limits {"ram_max_gib":30,"threads":3} — no disk_cache_gib. On a
    100%-full drive a free-disk-reserve basis makes EVERYTHING over-budget, and
    an auto path on that basis would evict model after model on every call.
    Unset must mean 'unmanaged', never 'evict everything'."""
    storage = {"cache_used_bytes": 900 * GIB, "disk_free": 0,   # drive FULL
               "models": [_model("a", 300), _model("b", 300), _model("c", 300)]}
    for limits in ({"ram_max_gib": 30.0, "threads": 3}, {}, None,
                   {"disk_cache_gib": None}, {"disk_cache_gib": ""},
                   {"disk_cache_gib": "not-a-number"}, {"disk_cache_gib": 0}):
        plan = budget.fit_plan("caller", 50 * GIB, storage, limits, {})
        assert plan["action"] == "proceed", f"thrashed on limits={limits!r}"
        assert plan["evict"] == []


def test_cap_bytes_parses_real_values():
    assert budget.cap_bytes({"disk_cache_gib": 50}) == 50 * GIB
    assert budget.cap_bytes({"disk_cache_gib": "50"}) == 50 * GIB
    assert budget.cap_bytes({"disk_cache_gib": 0.5}) == GIB // 2
    assert budget.cap_bytes({"disk_cache_gib": -5}) is None


# ── the EXECUTOR: evict_to_fit must drive the guarded delete path ──────────
def test_evict_to_fit_executes_evictions_through_the_guarded_reaper():
    """fit_plan only PLANS. This asserts the executor actually calls the single
    guarded delete path (_reap_reclaim, which re-proves every guard per key),
    in FIFO order — not some second, divergent delete of its own."""
    from abstract_hugpy_dev.worker_agent import agent

    reaped = []
    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {"oldest": 1, "newest": 900}
    state.refused = {}

    orig_storage, orig_reap = agent._worker_storage, agent._reap_reclaim
    try:
        agent._worker_storage = lambda s: {
            "cache_used_bytes": 45 * GIB, "disk_free": 0,
            "models": [_model("oldest", 10), _model("newest", 35)]}
        agent._reap_reclaim = lambda s, keys: (
            reaped.extend(keys), {"ok": True, "freed_bytes": 10 * GIB})[1]
        budget.evict_to_fit(state, "caller", 20 * GIB)
    finally:
        agent._worker_storage, agent._reap_reclaim = orig_storage, orig_reap

    assert reaped[0] == "oldest"        # FIFO order reaches the real deleter


def test_evict_to_fit_raises_and_deletes_nothing_when_it_cannot_fit():
    from abstract_hugpy_dev.worker_agent import agent

    reaped = []
    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {}
    state.refused = {}

    orig_storage, orig_reap = agent._worker_storage, agent._reap_reclaim
    try:
        agent._worker_storage = lambda s: {
            "cache_used_bytes": 48 * GIB, "disk_free": 0,
            "models": [_static("big", 48)]}   # static blocks; pin would not
        agent._reap_reclaim = lambda s, keys: (reaped.extend(keys), {})[1]
        with pytest.raises(budget.BudgetRefusal):
            budget.evict_to_fit(state, "huge", 30 * GIB)
    finally:
        agent._worker_storage, agent._reap_reclaim = orig_storage, orig_reap

    assert reaped == []                 # a refusal deletes NOTHING


def test_a_bookkeeping_failure_never_breaks_a_working_pull():
    """If the budget check itself blows up, the pull proceeds as it did before
    this feature — a storage-accounting bug must not ground the fleet."""
    from abstract_hugpy_dev.worker_agent import agent

    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {}
    state.refused = {}
    orig = agent._worker_storage
    try:
        def _boom(s):
            raise RuntimeError("storage survey exploded")
        agent._worker_storage = _boom
        budget.evict_to_fit(state, "caller", 20 * GIB)   # must NOT raise
    finally:
        agent._worker_storage = orig


# ── C: the refusal must actually REACH the console ─────────────────────────
def test_refused_survives_centrals_storage_proposal():
    """storage_proposal REBUILDS the storage dict field-by-field, so a new
    worker-reported key is dropped unless it is explicitly carried. The console
    reads THIS dict — without the passthrough the reason never renders and the
    model is invisible rather than missing-with-a-reason."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import (
        storage_proposal)

    reason = {"state": "refused", "reason": "won't fit: needs 30.0 GB…",
              "needs_bytes": 30 * GIB}
    out = storage_proposal({
        "storage": {"cache_used_bytes": 48 * GIB, "disk_free": 2 * GIB,
                    "models": [_model("big", 48, protected=True,
                                      why="static")],
                    "refused": {"huge": reason}},
        "disk": {"free_bytes": 2 * GIB, "total_bytes": 100 * GIB},
        "limits": {"disk_cache_gib": 50},
    })
    assert out["refused"] == {"huge": reason}
    # A refused model has no files, so it must never be in models/proposals.
    assert out["proposed_evictions"] == []


def test_storage_proposal_degrades_for_a_pre_feature_worker():
    """An older agent reports no `refused` key; central must yield {} — never
    a KeyError and never a phantom refusal."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import (
        storage_proposal)

    assert storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": GIB, "models": []},
        "disk": {}, "limits": {},
    })["refused"] == {}
    assert storage_proposal({"disk": {}})["refused"] == {}     # no survey at all


# ── the caller always wins (the operator's rule 2), end-to-end on the plan ──
def test_caller_wins_over_many_cold_models():
    """A big pull sweeps as many cold models as it takes — the caller installs."""
    models = [_model(f"cold{i}", 10) for i in range(6)]      # 60 GiB used
    storage = _storage(models)
    lp = {f"cold{i}": i + 1 for i in range(6)}               # cold0 oldest
    plan = budget.fit_plan("caller", 30 * GIB, storage,
                           {"disk_cache_gib": 60}, lp)
    assert plan["action"] == "evict"
    assert plan["evict"] == ["cold0", "cold1", "cold2"]      # strict FIFO order


# ══ ALLOCATION-LEVEL view (operator, 2026-07-16) ═══════════════════════════
# "it should also show how much is needed based on the total size of all models
# allocated". The per-pull numbers answer "can THIS pull fit". These assert the
# STRUCTURAL question: can the ASSIGNED SET fit at all?

def _alloc(total_gib, count, unknown=0):
    """Central's allocation totals as the heartbeat reply ships them."""
    return {"allocated_total_bytes": int(total_gib * GIB),
            "allocated_count": count, "allocated_unknown_count": unknown}


def test_refusal_reports_the_allocated_set_total_and_overage():
    """1+3: the refusal carries the ASSIGNMENT-set total and how far over the
    budget it is — the operator's "how much is needed" for the WHOLE set."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(180, 12))
    assert plan["action"] == "refuse"
    r = plan["reason"]
    assert r["allocated_total_bytes"] == 180 * GIB
    assert r["allocated_count"] == 12
    assert r["allocated_unknown_count"] == 0
    # 180 assigned - 50 budget = 130 over. The STRUCTURAL deficit.
    assert r["allocated_over_budget_bytes"] == 130 * GIB


def test_allocated_total_is_the_assignment_set_not_what_is_on_disk():
    """1: THE point. Lazy download means an assigned model often has NO files.
    Sizing only on-disk models would UNDER-report and make an over-subscribed
    set look fine — hiding exactly what this feature exists to show."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers

    # 5 models ASSIGNED; only ONE has landed on disk.
    sizes = {"a": 40 * GIB, "b": 40 * GIB, "c": 40 * GIB, "d": 30 * GIB,
             "e": 30 * GIB}
    orig = workers._model_size_bytes
    try:
        workers._model_size_bytes = lambda mk: sizes.get(mk)
        out = workers.allocated_totals({"models": ["a", "b", "c", "d", "e"]})
    finally:
        workers._model_size_bytes = orig

    assert out["allocated_count"] == 5           # the ASSIGNMENT set, not 1
    assert out["allocated_total_bytes"] == 180 * GIB
    assert out["allocated_unknown_count"] == 0


def test_model_size_bytes_really_resolves_against_the_real_manifest():
    """NO MOCK. Every other allocation test stubs _model_size_bytes, so all of
    them passed while the REAL function returned None for every model in the
    manifest (a wrong relative-import depth, swallowed by a broad except). That
    is this repo's recurring failure: a green test asserting a dead function.
    This drives the real registry and demands a real number."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers
    from abstract_hugpy_dev.imports.config.models.models_config import (
        get_models_dict)

    manifest = get_models_dict(dict_return=True) or {}
    on_disk = [k for k in manifest if workers._model_size_bytes(k)]
    assert on_disk, ("_model_size_bytes returned None for EVERY manifest model "
                     "— the sizing path is dead (import depth?), not merely cold")
    # A real, sane size — not a 0 dressed up as an answer.
    assert workers._model_size_bytes(on_disk[0]) > 0
    # An unknowable model is None (-> counted as unknown), never 0.
    assert workers._model_size_bytes("definitely/not-a-real-model") is None


def test_unknown_sizes_are_counted_never_silently_zeroed():
    """2: a model central can't size must be COUNTED and REPORTED. Silently
    treating it as 0 makes an over-subscribed set read as comfortable."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers

    orig = workers._model_size_bytes
    try:
        # 2 sizable, 2 unknowable (not in the manifest / not on disk).
        workers._model_size_bytes = lambda mk: (40 * GIB if mk in ("a", "b")
                                                else None)
        out = workers.allocated_totals({"models": ["a", "b", "ghost1", "ghost2"]})
    finally:
        workers._model_size_bytes = orig

    assert out["allocated_count"] == 4
    assert out["allocated_unknown_count"] == 2   # surfaced, not swallowed
    assert out["allocated_total_bytes"] == 80 * GIB   # a FLOOR, honestly


def test_unknown_sizes_are_named_in_the_human_reason_as_a_floor():
    """2: the hover must SAY the total is incomplete — "≥" + an unknown count —
    rather than present a floor as a precise number."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(180, 12, unknown=3))
    text = plan["reason"]["reason"]
    assert "≥" in text and "3 unknown" in text


def test_allocated_over_budget_is_zero_when_the_assigned_set_fits():
    """3: a set that fits reports 0 overage — a refusal is then honestly about
    THIS pull, not a structural over-subscription."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(30, 2))
    r = plan["reason"]
    assert r["allocated_over_budget_bytes"] == 0
    assert "over budget" not in r["reason"].split("assigned set")[1]


def test_human_reason_states_the_allocation_total_plainly():
    """5: the operator reads the HOVER, not the JSON."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(180, 12))
    text = plan["reason"]["reason"]
    assert "won't fit" in text                    # the per-pull half survives
    assert "180.0 GB" in text                     # the allocation total
    assert "130.0 GB over budget" in text         # the structural deficit
    assert "assigned set (12)" in text


def test_allocation_clause_is_omitted_when_central_has_not_said():
    """Honesty: before the first heartbeat (or against an older central) there
    are NO totals. Say nothing — never claim a 0 GiB allocation, which would
    read as "nothing is assigned" and is a lie."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    for allocated in (None, {}, {"allocated_count": None}):
        plan = budget.fit_plan("huge", 30 * GIB, storage,
                               {"disk_cache_gib": 50}, {}, allocated)
        text = plan["reason"]["reason"]
        assert "assigned set" not in text
        assert "0 B" not in text.split("reclaimable")[1]
        assert "won't fit" in text               # the pull verdict still lands


def test_allocation_totals_never_change_the_fit_verdict():
    """Additive by construction: the decision is a function of REAL bytes on
    REAL disk. A wildly over-subscribed assignment must not refuse a pull that
    genuinely fits — that would ground a working fleet on bookkeeping."""
    storage = _storage([_model("a", 10)])
    plan = budget.fit_plan("newbie", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {}, _alloc(900, 30))
    assert plan["action"] == "proceed"


# ── the new fields must REACH the console (the drop-on-rebuild trap) ────────
def test_allocation_fields_survive_centrals_storage_proposal():
    """4: storage_proposal REBUILDS the dict field-by-field — the previous slice
    found it silently DROPPED `refused`. Assert the new fields make it out, or
    the console renders nothing and the operator is told nothing."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import workers

    orig = workers._model_size_bytes
    try:
        workers._model_size_bytes = lambda mk: 60 * GIB      # 3 x 60 = 180
        out = workers.storage_proposal({
            "models": ["a", "b", "c"],           # the ASSIGNMENT set
            "storage": {"cache_used_bytes": 10 * GIB, "disk_free": 40 * GIB,
                        "models": [_model("a", 10)]},        # only ONE landed
            "disk": {"free_bytes": 40 * GIB, "total_bytes": 500 * GIB},
            "limits": {"disk_cache_gib": 50},
        })
    finally:
        workers._model_size_bytes = orig

    assert out["allocated_total_bytes"] == 180 * GIB
    assert out["allocated_count"] == 3
    assert out["allocated_unknown_count"] == 0
    assert out["allocated_over_budget_bytes"] == 130 * GIB
    # Structural truth with NO pull happening and the disk comfortably under.
    assert out["over_budget"] is False


def test_storage_proposal_allocation_degrades_for_an_unassigned_worker():
    """A worker with no assignments reports a real, honest zero — and never a
    phantom overage."""
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import (
        storage_proposal)

    out = storage_proposal({"storage": {"cache_used_bytes": 0, "disk_free": GIB,
                                        "models": []},
                            "disk": {}, "limits": {"disk_cache_gib": 50}})
    assert out["allocated_count"] == 0
    assert out["allocated_total_bytes"] == 0
    assert out["allocated_over_budget_bytes"] == 0


def test_worker_adopts_allocation_totals_from_the_heartbeat_reply():
    """The wire: sizing needs the manifest (central-only), so central ships the
    totals and the worker adopts them — no HTTP under the provision lock."""
    from abstract_hugpy_dev.worker_agent import agent

    state = type("S", (), {})()
    state.limits = {}
    state.model_last_picked = {}
    state.allocated = {}
    agent._adopt_storage_inputs(state, {
        "limits": {"disk_cache_gib": 50},
        "storage": {"allocated_total_bytes": 180 * GIB, "allocated_count": 12,
                    "allocated_unknown_count": 1},
    })
    assert state.allocated["allocated_total_bytes"] == 180 * GIB
    assert state.allocated["allocated_count"] == 12
    assert state.allocated["allocated_unknown_count"] == 1


def test_adopting_a_pre_feature_reply_leaves_allocation_untouched():
    """An older central sends no allocation totals: keep what we had, never
    clobber it with a phantom zero."""
    from abstract_hugpy_dev.worker_agent import agent

    state = type("S", (), {})()
    state.limits = {}
    state.model_last_picked = {}
    state.allocated = {"allocated_total_bytes": 180 * GIB,
                       "allocated_count": 12, "allocated_unknown_count": 0}
    agent._adopt_storage_inputs(state, {"limits": {"disk_cache_gib": 50},
                                        "storage": {"cache_used_bytes": 0}})
    assert state.allocated["allocated_count"] == 12       # preserved
