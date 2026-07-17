"""Call-driven storage budget for the worker: evict-to-fit, or REFUSE the pull.

WHY THIS EXISTS (incident, 2026-07-16): the operator's workstation ("op") filled
to 0 bytes free. ``provision.py`` had NO disk checks at all — it downloaded until
the kernel returned [Errno 28], leaving a wedged partial pull on a full disk.

THE OPERATOR'S DESIGN (verbatim):
  1. "the total space from those modules, if the total space of allocated drive
     space from central is taken up, then fifo the models"
  2. "yes fifo it, remove an existing model and install the one that is being
     called"
  3. "and yes, refuse it if it wont, show it as missing, hover info why"

So: **the model being CALLED always wins.** On a pull, if the worker's model
total would exceed its central-allocated budget, evict oldest-first (FIFO) to
make room, then install the caller. ONLY if a full FIFO of every reclaimable
candidate still can't free enough do we REFUSE — before a single byte is
downloaded — and report the model as MISSING with an honest reason.

── HOW THIS RELATES TO THE OPERATOR-GATED REAPER (read before changing) ────────
This is a SEPARATE, NARROW path. It is NOT a second eviction policy and NOT a
background sweep:

  * ``/reap`` + ``/reap-approve`` (flask_app/.../worker_routes.py) remain the
    BULK, operator-in-the-loop path: a human reads the proposal and approves a
    subset. That route's guarantee — "nothing deletes except through this
    explicit, operator-gated, audited call" — was scoped to THAT route's own
    bulk-reclaim flow, and it still holds for it: this module never calls it,
    never auto-approves it, and never widens what it may delete.
  * THIS path fires ONLY to make room for a model ACTIVELY BEING PROVISIONED
    (call-driven), never on a timer, never as a sweep, and it evicts the MINIMUM
    prefix of the FIFO order needed to seat the caller. No call -> no delete.

Both paths funnel into the SAME single delete choke point (``wipe_model``, which
is path-jailed and re-proves the shared/central-store gate), so neither can
delete something the other wouldn't.

── ORDERING + GUARDS ARE REUSED, NOT REINVENTED ────────────────────────────────
The FIFO order and the candidate domain deliberately mirror the two proven
implementations already in the tree, so a third divergent policy never exists:

  * ``utils/workers.py:storage_proposal`` — central's read-only preview: the
    candidate domain is UNPROTECTED models only, sorted ASCENDING by
    ``last_picked`` (oldest-first), greedily accumulated until ``need`` is
    covered. This module produces the SAME order over the SAME domain, so what
    the console previews is what an auto-evict would actually take.
  * ``managers/serve/model_cache.py:evict_for(need_bytes, keep_dir)`` — the hot
    cache's LRU-evict-until-fits loop, including the ``keep_dir`` exclusion that
    stops a warm from evicting the very entry it is warming.

We MIRROR rather than import ``evict_for``: it is bound to the hot-cache tier
(its own CACHE_DIR/CACHE_MAX_BYTES globals, mtime as the LRU key, whole-dir
rmtree). This tier is different in every one of those inputs — the model root,
a central-allocated budget, central's ``last_picked`` as the FIFO key, and
``wipe_model`` as the only permitted delete. Reusing it would mean rewiring the
hot cache around parameters it doesn't have. The SEMANTICS are copied exactly:
oldest-first, stop as soon as it fits, never touch the keep target.

── FIFO KEY ────────────────────────────────────────────────────────────────────
``last_picked`` (when central was last asked to serve this model on this box) is
the operator's "oldest" — a model nobody has called in weeks is the right thing
to drop for one being called RIGHT NOW. Central owns that clock and ships it in
the assignment payload; a model central has never served has no entry and sorts
as 0 = coldest = evicted first (exactly right for never-served test leftovers).
When central hasn't shipped the map at all, we fall back to on-disk mtime so the
order is still oldest-first rather than arbitrary.

── 📌 PIN + ALLOCATION HAVE NO BEARING ON EVICTION (operator, 2026-07-17) ───────
The canonical statement (verbatim): "the pins only should designate that the
model allocation survives restarts. the allocation only stipulates the routing
for that model (to that worker). neither of those should have any bearing on the
pull or eviction, unless its to do with priority, then a pinned model should
take higher precidence than unpinned, but even that is trivial".

So in THIS module: a pinned or assigned model is a normal eviction CANDIDATE
(see _is_protected). Evicting its files leaves the pin + allocation untouched —
routing survives and the bytes re-pull on the next call. Pin's ONLY eviction
role is the trivial FIFO tiebreak in fit_plan: among equally-stale candidates,
unpinned evict first. Only 🔒static promises local presence and is protected;
loaded/loading/provisioning are protected as live-use guards. This is the
day-one tripwire the operator called out — conflating attribution/routing with a
disk shield filled his workstation to 0 bytes free on 2026-07-16.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger("abstract_hugpy_dev.worker_agent.budget")

# Protection reasons that make a model INELIGIBLE for auto-eviction. Mirrors
# storage_proposal's chain (utils/workers.py) minus `assigned` AND `pinned` —
# see _is_protected for why NEITHER protects here.
#   * `assigned` = attribution/routing only (lazy-download doctrine).
#   * `pinned`   = the ALLOCATION survives restarts, nothing else (operator,
#     2026-07-17). Pin has NO bearing on eviction — a pinned model's files are a
#     normal LRU candidate; evicting them leaves pin + routing untouched and the
#     bytes re-pull on next call.
# Only 🔒static (durable local-presence promise) and the live-use guards
# (loaded/loading/provisioning — deleting under a live pull corrupts the fetch)
# stay protected.
_PROTECTED_REASONS = ("static", "loaded", "loading", "provisioning")


class BudgetRefusal(Exception):
    """A pull that cannot fit even after a full permissible FIFO.

    Raised BEFORE any bytes are downloaded. Carries the machine-readable
    ``reason`` dict the heartbeat/console render on hover, so the model reads as
    MISSING-with-a-reason rather than a stalled pull.
    """

    def __init__(self, reason: dict):
        self.reason = reason
        super().__init__(reason.get("reason") or "won't fit")


def _human(n) -> str:
    """Bytes -> a short human string. Mirrors provision._human's units so the
    refusal reason reads the same as the transfer logs."""
    if not n:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def cap_bytes(limits: dict | None) -> int | None:
    """The worker's EXPLICIT storage allocation in bytes, or None if unset.

    ``limits['disk_cache_gib']`` is central's per-worker allocation (the
    operator sets it; the box may tighten it via HUGPY_DISK_CACHE_MAX_GIB, which
    central already clamps against in _clamp_limits). Returns None when it is
    absent/blank/unparseable/non-positive — i.e. "no allocation declared".

    D (budget must be real): None is a FIRST-CLASS answer, not a zero. An unset
    allocation means this box has no declared model-cache ceiling, and the
    auto-evict path treats that as "don't evict" rather than inventing one. See
    fit_plan for why the free-disk reserve is NOT used as a fallback budget here.
    """
    if not isinstance(limits, dict):
        return None
    raw = limits.get("disk_cache_gib")
    if raw in (None, ""):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    return int(val * (1 << 30))


def _is_protected(row: dict) -> str:
    """The reason ``row`` may NOT be auto-evicted, or "" if it is a candidate.

    Domain mirrors storage_proposal's guard chain with TWO deliberate
    differences: NEITHER ``assigned`` NOR ``pinned`` protects here.

    CANONICAL STATEMENT (operator ruling, 2026-07-17): "the pins only should
    designate that the model allocation survives restarts. the allocation only
    stipulates the routing for that model (to that worker). neither of those
    should have any bearing on the pull or eviction".
      * 📌 pin = the model's ALLOCATION survives restarts (and unassign
        attempts). Nothing else.
      * Allocation = ROUTING: which worker answers for that model.
      * NEITHER has any bearing on the pull (already true — lazy download,
        7f0e6e8/2a3baeb) NOR on eviction (this rule). Evicting a pinned or
        assigned model's FILES never touches its pin or allocation — routing
        survives, bytes re-pull on next call. A row whose ONLY claim is pinned
        (or assigned) is a CANDIDATE.

    This closes the day-one tripwire the operator called out: assignment and pin
    are attribution/routing, not a disk shield. On a box whose models are all
    assigned (the normal case, and op's actual case) they must all be reclaimable
    or the disk fills — exactly the 2026-07-16 incident.

    Only the DURABLE local-presence promise and the live-use guards protect:
    🔒static (the ONE tier that promises the files stay local), plus
    loaded/loading/provisioning (deleting under a live pull corrupts the fetch;
    deleting a loaded model breaks serving). A pinned/assigned model that is ALSO
    static/loaded keeps protection through THAT flag — never through pin.
    """
    if row.get("protected") and (row.get("why") or "") not in ("assigned", ""):
        # Worker-side flag already decided (e.g. "shared/central storage —
        # never reaped", "model store not marked reapable"). Trust it, except
        # for a bare `assigned`, which is attribution only (see above).
        return str(row.get("why") or "protected")
    for flag in _PROTECTED_REASONS:
        if row.get(flag):
            return flag
    return ""


def _allocation_clause(allocated: dict | None, cap: int) -> tuple[str, dict]:
    """The ALLOCATION-LEVEL half of a refusal: is the ASSIGNED SET itself too big?

    OPERATOR (2026-07-16): "it should also show how much is needed based on the
    total size of all models allocated". The per-pull numbers answer "can THIS
    pull fit". This answers the more useful question: the deficit is STRUCTURAL
    — the assignment set cannot fit at ANY eviction order, so no call will ever
    be lucky. Without it the operator reads a refusal as this-pull-was-unlucky
    and re-tries forever.

    Returns ``(text, fields)``. Central sizes the set (it owns the manifest) and
    ships the totals in the heartbeat reply; this is a pure read of that answer.

    HONESTY RULES (a silent 0 makes an over-subscribed set look fine):
      * no totals yet (pre-first-beat / older central) -> ("", {}). SAY NOTHING
        rather than claim a 0 GiB allocation.
      * unknown-size models are COUNTED and NAMED in the text ("N unknown"), and
        the total is then a FLOOR — "≥" says so rather than implying precision.
    """
    if not isinstance(allocated, dict) or allocated.get("allocated_count") is None:
        return "", {}
    total = int(allocated.get("allocated_total_bytes") or 0)
    count = int(allocated.get("allocated_count") or 0)
    unknown = int(allocated.get("allocated_unknown_count") or 0)
    over = max(0, total - cap)
    fields = {
        "allocated_total_bytes": total,
        "allocated_count": count,
        "allocated_unknown_count": unknown,
        "allocated_over_budget_bytes": over,
    }
    if not count:
        return "", fields
    approx = "≥" if unknown else ""
    text = f" — assigned set ({count}) totals {approx}{_human(total)}"
    if unknown:
        text += f", {unknown} unknown"
    if over:
        text += f" ({_human(over)} over budget)"
    return text, fields


def fit_plan(model_key: str, need_bytes: int, storage: dict,
             limits: dict | None, last_picked: dict | None = None,
             allocated: dict | None = None) -> dict:
    """Decide how to seat ``need_bytes`` of ``model_key`` under the budget.

    PURE — computes, never deletes. The caller (evict_to_fit) executes it. Being
    pure is what lets the tests assert the ORDER and the REFUSAL without a disk.

    ``allocated`` — central's ALLOCATION-LEVEL totals for this worker's
    assignment set (``{allocated_total_bytes, allocated_count,
    allocated_unknown_count}``, adopted from the heartbeat reply). Optional and
    purely ADDITIVE: it changes no decision, only what a refusal REPORTS. The
    fit verdict stays a function of real bytes on real disk.

    Returns::

        {"action": "proceed"|"evict"|"refuse",
         "evict": [model_key, ...],        # FIFO order, oldest-first
         "reason": {...} | None}           # machine-readable, only on refuse

    Decision table:
      * no explicit cap            -> "proceed" (D: unset != evict everything)
      * fits under the cap as-is   -> "proceed", nothing evicted
      * fits after evicting a FIFO prefix of reclaimable candidates -> "evict"
      * even a FULL FIFO can't free enough -> "refuse" (+ an honest reason)
    """
    cap = cap_bytes(limits)
    rows = [r for r in (storage.get("models") or [])
            if isinstance(r, dict) and r.get("model_key")]
    used = int(storage.get("cache_used_bytes") or 0)
    need = max(0, int(need_bytes or 0))

    # The model being provisioned is the KEEP TARGET (model_cache.evict_for's
    # keep_dir exclusion, by key rather than by path): never evict the thing we
    # are making room for. Bytes it ALREADY has on disk (a resumed/partial pull)
    # are counted as headroom it doesn't need to re-take.
    have = 0
    for r in rows:
        if r["model_key"] == model_key:
            have = int(r.get("bytes") or 0)
            break
    delta = max(0, need - have)          # NEW bytes this pull will add

    # ── D: no explicit allocation -> no auto-eviction ───────────────────────
    # A worker with no disk_cache_gib has no declared ceiling, so there is no
    # honest way to say what is "over". Deliberately NOT falling back to the
    # free-disk reserve (which storage_proposal uses for its DISPLAY budget):
    # on a 100%-full drive, disk_free < reserve makes EVERYTHING read as
    # over-budget, and an auto path on that basis would evict model after model
    # on every call — thrash, and exactly the damage this fix exists to stop.
    # Unset therefore means "don't manage this box's storage": pull as before.
    # The operator sets disk_cache_gib to turn self-maintenance ON.
    if cap is None:
        return {"action": "proceed", "evict": [], "reason": None,
                "note": "no disk_cache_gib allocation set — budget unmanaged"}

    if used + delta <= cap:
        return {"action": "proceed", "evict": [], "reason": None}

    # ── over budget: FIFO the reclaimable candidates, oldest first ──────────
    lp_map = last_picked or {}

    def _lp(mk, row):
        v = lp_map.get(mk)
        try:
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
        # No central clock for this key: fall back to the worker's own on-disk
        # mtime so the order stays oldest-first instead of arbitrary. 0.0 (never
        # served, no mtime) sorts coldest — evicted first, which is right for
        # never-called leftovers.
        try:
            v2 = row.get("mtime")
            return float(v2) if v2 is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    candidates = []
    blocked: dict[str, int] = {}
    for r in rows:
        mk = r["model_key"]
        if mk == model_key:
            continue                       # the keep target — never a candidate
        why = _is_protected(r)
        if why:
            blocked[why] = blocked.get(why, 0) + 1
            continue
        candidates.append((_lp(mk, r), bool(r.get("pinned")),
                           int(r.get("bytes") or 0), mk))

    # Oldest-first (primary key = central last_picked). Then a TRIVIAL pin
    # tiebreak: among equally-stale candidates, evict UNPINNED before PINNED
    # (False sorts before True). The operator called this "trivial and likely
    # unnecessary" (2026-07-17) — implemented only because it costs nothing and
    # gives a pinned model a hair of extra precedence at an exact last_picked
    # tie. Then largest-first among the rest so the budget clears in the fewest
    # deletes; stable key tiebreak. IDENTICAL primary order to storage_proposal.
    candidates.sort(key=lambda c: (c[0], c[1], -c[2], c[3]))

    must_free = used + delta - cap
    reclaimable_total = sum(b for _lp_, _pin_, b, _mk in candidates)

    evict: list[str] = []
    freed = 0
    for _lp_, _pin_, b, mk in candidates:
        if freed >= must_free:
            break
        evict.append(mk)
        freed += b

    if freed < must_free:
        # ── B: REFUSE. Not even a full FIFO can seat this model. ────────────
        # Return BEFORE any download starts — the whole point: no more
        # 7%-wedged pulls that fill a disk.
        blocked_str = ", ".join(f"{n} {why}" for why, n in
                                sorted(blocked.items(), key=lambda kv: -kv[1]))
        # The ALLOCATION-LEVEL clause: per-pull numbers say "this pull won't
        # fit"; this says WHY it never will if the assigned set is itself
        # over-subscribed. Additive — it never changes the verdict above.
        alloc_text, alloc_fields = _allocation_clause(allocated, cap)
        reason = {
            "state": "refused",
            "model_key": model_key,
            "reason": (
                f"won't fit: needs {_human(delta)}, budget {_human(cap)}, "
                f"{_human(reclaimable_total)} reclaimable"
                + (f" ({blocked_str})" if blocked_str else "")
                + alloc_text
            ),
            **alloc_fields,
            "needs_bytes": delta,
            "budget_bytes": cap,
            "used_bytes": used,
            "must_free_bytes": must_free,
            "reclaimable_bytes": reclaimable_total,
            "reclaimable_count": len(candidates),
            "blocked": blocked,
            "shortfall_bytes": must_free - reclaimable_total,
        }
        return {"action": "refuse", "evict": [], "reason": reason}

    return {"action": "evict", "evict": evict, "reason": None,
            "freed_bytes": freed, "must_free_bytes": must_free}


def evict_to_fit(state, model_key: str, need_bytes: int) -> None:
    """Make room under the budget for ``model_key``, or raise BudgetRefusal.

    The IMPURE bookend to fit_plan: gathers live inputs, runs the plan, and
    executes any evictions through the worker's single guarded delete path
    (``_reap_reclaim`` — which re-proves EVERY guard per key at delete time and
    whose ``wipe_model`` is path-jailed and refuses shared/central storage).

    Called from provision.ensure_model_present BEFORE a pull starts. Best-effort
    by construction: any failure to COMPUTE a plan lets the pull proceed exactly
    as it did before this feature (never break a working pull on a bookkeeping
    error) — but a REFUSAL is a decision, not a failure, and always propagates.
    """
    try:
        from .agent import _worker_storage, _reap_reclaim
        storage = _worker_storage(state)
        limits = getattr(state, "limits", None) or {}
        last_picked = getattr(state, "model_last_picked", None) or {}
        # Central-computed allocation totals (heartbeat reply). Absent before the
        # first beat -> the refusal simply omits the structural clause.
        allocated = getattr(state, "allocated", None) or {}
        plan = fit_plan(model_key, need_bytes, storage, limits, last_picked,
                        allocated)
    except BudgetRefusal:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("budget check for %s failed (%s) — proceeding with the "
                       "pull as before", model_key, exc)
        return

    if plan["action"] == "refuse":
        logger.error("REFUSING pull of %s — %s", model_key,
                     plan["reason"]["reason"])
        raise BudgetRefusal(plan["reason"])

    if plan["action"] != "evict" or not plan["evict"]:
        return

    logger.info("budget: %s needs room — FIFO-evicting %d model(s) oldest-first: %s",
                model_key, len(plan["evict"]), ", ".join(plan["evict"]))
    try:
        result = _reap_reclaim(state, plan["evict"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("budget: eviction for %s failed (%s) — proceeding; the "
                       "pull may still hit a full disk", model_key, exc)
        return
    freed = result.get("freed_bytes", 0) if isinstance(result, dict) else 0
    logger.info("budget: freed %s for %s", _human(freed), model_key)
    # Force a fresh storage walk so the next check sees the deletions (the
    # 60s _STORAGE_CACHE would otherwise re-report the evicted models).
    try:
        from .agent import _STORAGE_CACHE
        _STORAGE_CACHE["at"] = 0.0
    except Exception:  # noqa: BLE001
        pass
