"""Reconcile deduces fit instead of load-probing every assigned model.

Background (PLAN-reconcile-deduce-fit.md, 2026-07-15): heartbeat reconcile used
to build ``cold`` = every assigned-but-not-loaded model and blanket-fire a live
``/probe/<mk>`` (an ACTUAL GPU load) at each one. A worker with dozens of
assignments would sequentially load/evict-churn a card that physically holds a
handful, just to answer a question central already had the numbers for:
``_worker_fit`` computes fit from effective GGUF bytes vs the worker's reported
vram_free/free_ram, no load required.

``_warmable_subset(worker, cold)`` is the new prefilter reconcile runs before
kicking `_kick_warm`:
  * ``fit is False`` (can't fit VRAM+RAM combined) -> DROPPED, never probed.
  * ``fit is None`` (can't be sized) -> KEPT unconditionally (the honest
    live-probe fallback — no numbers exist to deduce from).
  * otherwise -> co-residency capped: greedy-packed GPU-resident-first,
    smallest-need-first, until the running total would exceed the worker's
    current vram_free. The rest stays assigned-but-cold (lazy-loads fine on
    real demand).

Runs both ways:
    cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
    venv/bin/python -m pytest tests/test_reconcile_deduce_fit.py -q
    venv/bin/python tests/test_reconcile_deduce_fit.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-reconcile-fit-test-"))

wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")

GIB = 2 ** 30

# --- shared fixture: a worker with a known VRAM budget + a model size table --
# free_ram is deliberately modest (not 64 GiB) so huge-d (50 GiB) genuinely
# fails combined VRAM+RAM capacity (10 + 8 = 18 GiB) instead of "fitting" by
# spilling entirely into a huge RAM pool — that would defeat the fixture's
# purpose (an always-un-fittable model to exercise the drop path).
WORKER = {"id": "w1", "name": "testbox", "gpu": "RTX TEST",
          "vram_free": int(10 * GIB), "free_ram": int(8 * GIB)}

# model_key -> raw GGUF bytes (None = unsizable)
SIZES = {
    "small-a": int(2 * GIB),     # gpu_resident, cheap
    "small-b": int(2 * GIB),     # gpu_resident, cheap
    "mid-c":   int(5 * GIB),     # gpu_resident, bigger
    "huge-d":  int(50 * GIB),    # won't fit VRAM+RAM at all (fit False)
    "unsized-e": None,           # can't be sized (fit None) -> always kept
}


class _patched_sizes:
    """Swap wr._model_gguf_bytes for a fixed lookup table; restore on exit."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.orig = None

    def __enter__(self):
        self.orig = wr._model_gguf_bytes
        wr._model_gguf_bytes = lambda mk: self.sizes.get(mk)
        return self

    def __exit__(self, *exc):
        wr._model_gguf_bytes = self.orig


def _clear_skip_log():
    wr._fit_skip_last.clear()


# --------------------------------------------------------------------------- #
# fit-based drop / keep
# --------------------------------------------------------------------------- #
def test_unfittable_dropped_unsizable_kept():
    with _patched_sizes(SIZES):
        _clear_skip_log()
        warm = wr._warmable_subset(WORKER, ["huge-d", "unsized-e"])
    assert "huge-d" not in warm, warm
    assert "unsized-e" in warm, warm


def test_fittable_set_kept_within_budget_unfittable_still_dropped():
    with _patched_sizes(SIZES):
        _clear_skip_log()
        # small-a(2) + small-b(2) + mid-c(5) = 9 GiB raw, fits a 10 GiB vram_free
        # budget even after headroom on the smaller two; huge-d never belongs.
        warm = wr._warmable_subset(
            WORKER, ["small-a", "small-b", "mid-c", "huge-d"])
    assert "huge-d" not in warm, warm
    assert set(warm) <= {"small-a", "small-b", "mid-c"}
    assert "small-a" in warm and "small-b" in warm


# --------------------------------------------------------------------------- #
# co-residency cap: don't warm more than can actually co-reside
# --------------------------------------------------------------------------- #
def test_tight_budget_caps_the_warm_set():
    tight_worker = dict(WORKER, vram_free=int(3 * GIB))
    with _patched_sizes(SIZES):
        _clear_skip_log()
        warm = wr._warmable_subset(tight_worker, ["small-a", "small-b", "mid-c"])
    # need = raw * VRAM_HEADROOM (1.15): small-a/b need ~2.3 GiB each, mid-c
    # ~5.75 GiB — a 3 GiB budget can hold at most ONE small model, never mid-c.
    assert len(warm) < 3, warm
    assert "mid-c" not in warm, warm
    assert set(warm) <= {"small-a", "small-b", "mid-c"}


def test_gpu_resident_preferred_when_packing():
    # Budget only large enough for one of two equally-sized candidates: the
    # smaller/gpu_resident-preferring pack must still pick deterministically
    # (smallest-need-first tie-break) rather than warming neither.
    spill_worker = {"id": "w2", "vram_free": int(1 * GIB), "free_ram": int(64 * GIB)}
    sizes = {"resident-small": int(1 * GIB) // 2, "spill-small": int(1 * GIB) // 2}
    with _patched_sizes(sizes):
        _clear_skip_log()
        warm = wr._warmable_subset(spill_worker, ["resident-small", "spill-small"])
    assert "resident-small" in warm, warm


def test_no_vram_free_reported_skips_the_cap():
    # A worker with no vram_free number (e.g. hasn't reported a GPU) can't be
    # capped against — the fit-based drop still runs, but nothing is withheld
    # for co-residency reasons (no data to invent a cap from).
    no_vram_worker = {"id": "w3", "vram_free": None, "free_ram": int(8 * GIB)}
    with _patched_sizes(SIZES):
        _clear_skip_log()
        warm = wr._warmable_subset(
            no_vram_worker, ["small-a", "small-b", "mid-c", "huge-d"])
    assert set(warm) == {"small-a", "small-b", "mid-c"}, warm


# --------------------------------------------------------------------------- #
# skip-log throttling: no per-beat spam
# --------------------------------------------------------------------------- #
def test_unfittable_skip_logged_once_not_per_call():
    logged = []
    orig_info = wr.logger.info
    wr.logger.info = lambda *a, **kw: logged.append((a, kw))
    try:
        with _patched_sizes(SIZES):
            _clear_skip_log()
            wr._warmable_subset(WORKER, ["huge-d"])
            wr._warmable_subset(WORKER, ["huge-d"])
            wr._warmable_subset(WORKER, ["huge-d"])
    finally:
        wr.logger.info = orig_info
    assert len(logged) == 1, logged


# --------------------------------------------------------------------------- #
# trivial edge case
# --------------------------------------------------------------------------- #
def test_empty_cold_list_is_a_noop():
    with _patched_sizes(SIZES):
        _clear_skip_log()
        assert wr._warmable_subset(WORKER, []) == []


# --------------------------------------------------------------------------- #
# curated keep-warm set: warm the designated subset, NOT the whole inventory
# (operator, 2026-07-15: "defaults immutable, others customizable, lazy-load
#  the rest"). _reconcile_warm_set = (immutable defaults ∪ pins ∪ static ∪
#  warm_whitelist) ∩ on-disk. Everything else lazy-loads.
# --------------------------------------------------------------------------- #
class _patched_defaults:
    """Pin wr._immutable_warm_defaults() to a known set; restore on exit.

    Sets the module cache directly so the real TASK_DEFAULTS import is bypassed
    (the test asserts the *selection logic*, not the fleet's actual defaults)."""

    def __init__(self, keys):
        self.keys = frozenset(keys)
        self.orig = None

    def __enter__(self):
        self.orig = wr._IMMUTABLE_WARM_DEFAULTS
        wr._IMMUTABLE_WARM_DEFAULTS = self.keys
        return self

    def __exit__(self, *exc):
        wr._IMMUTABLE_WARM_DEFAULTS = self.orig


# an ae-like box: a big inventory, a couple defaults on disk, a few pins.
_INV = ["junk-%d" % i for i in range(40)]
INV_WORKER = {
    "id": "wbig", "name": "aebox",
    "models": ["def-chat", "def-vl"] + _INV + ["pin-x", "pin-off"],
    "config": {"pinned": {"pin-x": True, "pin-off": False}},
}


def test_curated_set_is_defaults_plus_pins_present_not_inventory():
    with _patched_defaults({"def-chat", "def-vl", "def-not-on-disk"}):
        warm = wr._reconcile_warm_set(INV_WORKER)
    # defaults present + pinned-true present — nothing from the 40-item inventory,
    # not the absent default, not the pinned-false entry.
    assert warm == ["def-chat", "def-vl", "pin-x"], warm


def test_curated_set_drops_the_whole_inventory():
    with _patched_defaults({"def-chat", "def-vl"}):
        warm = set(wr._reconcile_warm_set(INV_WORKER))
    assert not (warm & set(_INV)), warm            # zero junk warmed
    assert len(warm) <= 3, warm                    # only the curated handful


def test_curated_set_honors_absent_default_is_not_invented():
    # A fleet default the box does NOT have on disk is never warmed (you can't
    # keep warm what isn't there) — guards against warming a missing model_key.
    w = {"id": "w", "models": ["only-this"], "config": {}}
    with _patched_defaults({"some-default", "another"}):
        assert wr._reconcile_warm_set(w) == []


def test_curated_set_forward_compat_warm_whitelist_and_static():
    # config schema has neither today; honor them if a future writer sets them.
    w = {"id": "w", "models": ["a", "b", "c", "d"],
         "config": {"warm_whitelist": ["a"], "residency": {"b": "static", "c": "on_demand"}}}
    with _patched_defaults(set()):
        warm = wr._reconcile_warm_set(w)
    assert warm == ["a", "b"], warm                # whitelist:a + static:b; c/d lazy


def test_curated_set_empty_models_is_noop():
    with _patched_defaults({"def-chat"}):
        assert wr._reconcile_warm_set({"id": "w", "models": [], "config": {}}) == []
        assert wr._reconcile_warm_set({"id": "w"}) == []


# --------------------------------------------------------------------------- #
# plain-script runner (pytest not required)
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
