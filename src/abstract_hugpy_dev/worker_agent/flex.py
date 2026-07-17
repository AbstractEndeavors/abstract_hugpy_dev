"""Tolerance-band secondary allocation layer (t21) — the pure band math + the
flex-before-evict decision, and the ONE priority comparator seam.

WHY THIS EXISTS (operator spec, recorded 2026-07-17, verbatim intent):
  "a threshold should be made as a secondary allocation layer, in which the
   percentage can deviate within a certain percent total from the desired
   explicit allocation for edge cases and explicit priorities."
Extended the same day: context (ctx%) joins VRAM% and RAM% as the THIRD banded
variable — the SAME system, one engine.

── THE MODEL ────────────────────────────────────────────────────────────────
Every EXPLICIT allocation (the spill contract — GGUF-only per 98f5056/363d0ed)
gains, per banded variable, a ``target`` and a ``deviation_pct``. The deviation
is PERCENT-OF-TOTAL (of the honest whole — the encroachment-model denominators
from the honest-bars work, commit e0bd9e4), so a band is the HARD interval

    [ target − deviation_pct%·whole , target + deviation_pct%·whole ]

clamped to the variable's own domain ([0, whole] for bytes; [1, 100] for ctx%).

THREE banded variables, ONE engine:
  * VRAM%  — target = spill ``gpu_mem_gib`` (bytes); whole = the box's honest
             VRAM denominator.
  * RAM%   — target = spill ``cpu_mem_gib`` (bytes); whole = the honest RAM
             denominator.
  * CTX%   — target = ``ctx_pct`` (already a percent 1..100 of the model's max
             context); the band is ± deviation percentage-points of that scale
             (i.e. "whole" is the 100-point ctx scale, so a percent-of-total
             deviation is just ± that many points).

── RESOLUTION ORDER AT CONTENTION (the operator's three stages) ─────────────
  (1) FLEX  — an edge-case load may stretch WITHIN its own band; a higher-
              priority load may compress neighbors WITHIN THEIR bands (never
              below the band FLOOR). ctx-KV is the CHEAPEST flex, so this module
              prefers compressing ctx before touching weight placement.
  (2) EVICT — only when flexing cannot fit (idle on-demand first, LRU, with the
              admission-doctrine protections — static/replying/queue-ahead never
              yield). This module does NOT evict; it hands the caller a plan and
              the caller runs the existing eviction engine (agent._vram_evict_to
              _fit) for stage (2).
  (3) REFUSE honestly — the existing honest-refusal path, unchanged.

UNCONTENDED == everyone at target. Bands ONLY matter under contention: if the
subject already fits at its target, no band is consulted and behaviour is
byte-identical to before this module existed.

── PROTECTION OUTRANKS PRIORITY ─────────────────────────────────────────────
The admission-doctrine protected tiers (🔒static / actively-replying / queued-
ahead) are NEVER flex-compressed by ANY priority. Protection is absolute;
priority only orders the UNPROTECTED, flex-eligible neighbours.

This module is PURE (no I/O, no globals, no eviction). That is what lets the
tests assert floors/ceilings, the compress-by-priority ordering, protection
immunity, uncontended==target, and the ctx-first ordering without a GPU.
"""
from __future__ import annotations

from typing import Optional


# ── THE priority comparator seam ─────────────────────────────────────────────
# Keeper decision (confirm-pending with the operator, 2026-07-17): priority is an
# optional explicit per-model integer on the explicit allocation (unset == 0 ==
# normal; higher compresses lower within bands). Ties are broken by pin (the
# existing pin-as-tiebreak doctrine) — the caller supplies that tiebreak; this
# function answers ONLY the primary priority key. Protected tiers are handled by
# the caller and never reach a priority comparison.
#
# THE OPERATOR MAY ADJUST THE SOURCE of priority (e.g. residency tier, a queue
# signal, an SLA class). Keep that change to THIS ONE function: everything else
# consumes ``flex_priority_key`` and never re-reads ``alloc['priority']`` itself,
# so a different answer plugs in here without re-plumbing the engine.
def flex_priority_key(alloc: Optional[dict]) -> int:
    """The primary priority key for an explicit allocation (higher == more
    important, i.e. compresses/evicts lower-priority neighbours first). Unset,
    non-dict, or unparseable == 0 == normal. THE single source of the priority
    number — see the module note before changing where it reads from."""
    if not isinstance(alloc, dict):
        return 0
    try:
        return int(alloc.get("priority") or 0)
    except (TypeError, ValueError):
        return 0


# ── band math (bytes-domain: VRAM / RAM) ─────────────────────────────────────
def band_bounds(target: float, deviation_pct: Optional[float],
                whole: Optional[float]) -> "tuple[float, float]":
    """The hard band interval ``(floor, ceiling)`` for a bytes-domain variable.

    ``deviation_pct`` is percent-of-``whole`` (the honest denominator). No band
    (deviation unset/<=0 or ``whole`` unknown) collapses to the point
    ``(target, target)`` — a model with no declared tolerance can neither be
    stretched nor compressed, so it behaves exactly as it did pre-t21.

    Clamped to ``[0, whole]`` when ``whole`` is known (never propose a negative
    or over-total allocation). ``target`` itself is clamped into the same domain
    first so a mis-entered target can't produce a floor above the ceiling."""
    t = max(0.0, float(target or 0.0))
    if whole is not None and whole > 0:
        t = min(t, float(whole))
    dev = float(deviation_pct or 0.0)
    if dev <= 0 or not whole or whole <= 0:
        return t, t
    span = (dev / 100.0) * float(whole)
    floor = max(0.0, t - span)
    ceil = min(float(whole), t + span)
    if ceil < floor:                      # degenerate (target clamped past ceil)
        floor = ceil = t
    return floor, ceil


def band_floor(target: float, deviation_pct: Optional[float],
               whole: Optional[float]) -> float:
    """The most-compressed admissible allocation for a bytes-domain band — the
    minimum a higher-priority neighbour may squeeze this one to (stage 1). Never
    below the floor (operator: 'never below band floor')."""
    return band_bounds(target, deviation_pct, whole)[0]


def band_ceiling(target: float, deviation_pct: Optional[float],
                 whole: Optional[float]) -> float:
    """The most-stretched admissible allocation for a bytes-domain band — the
    maximum an edge-case load may stretch ITSELF to (stage 1, self-flex up)."""
    return band_bounds(target, deviation_pct, whole)[1]


# ── band math (ctx%: the 1..100 point scale) ─────────────────────────────────
def ctx_band_bounds(ctx_pct: Optional[int],
                    deviation_pct: Optional[float]) -> "tuple[int, int] | None":
    """The hard ctx band ``(floor_pct, ceil_pct)`` in whole percent points, or
    None when there is no ctx target (ctx band is opt-in — no target == today's
    default ctx, byte-identical).

    ctx% is ALREADY a percent of the model's max context, so a percent-of-total
    deviation is simply ± that many points, clamped to the ctx domain [1, 100].
    ctx is the CHEAPEST flex (KV bytes are linear in ctx tokens), so the engine
    reaches for this band first."""
    if ctx_pct is None:
        return None
    try:
        c = int(ctx_pct)
    except (TypeError, ValueError):
        return None
    c = max(1, min(100, c))
    dev = float(deviation_pct or 0.0)
    if dev <= 0:
        return c, c
    floor = max(1, int(round(c - dev)))
    ceil = min(100, int(round(c + dev)))
    if ceil < floor:
        floor = ceil = c
    return floor, ceil


def kv_at_ctx_pct(kv_at_target: Optional[int], target_pct: Optional[int],
                  new_pct: Optional[int]) -> int:
    """Scale a KV-cache byte figure measured at ``target_pct`` to ``new_pct``.

    KV bytes are LINEAR in resolved ctx tokens and ctx tokens are linear in the
    percent, so ``kv(new) = kv(target) · new_pct / target_pct``. Used to price a
    ctx flex (compress the subject's ctx toward its band floor) WITHOUT a re-load
    or re-measure. Degenerate inputs (missing/zero) return the input unchanged so
    a pricing gap never fabricates savings."""
    if not kv_at_target or not target_pct or not new_pct:
        return int(kv_at_target or 0)
    try:
        return max(0, int(kv_at_target * float(new_pct) / float(target_pct)))
    except (TypeError, ValueError, ZeroDivisionError):
        return int(kv_at_target or 0)


# ── the flex-before-evict decision ───────────────────────────────────────────
class FlexPlan:
    """The result of :func:`plan_flex`.

    ``action``:
      * ``"proceed"`` — fits at target; bands untouched (uncontended path).
      * ``"flex"``    — fits AFTER flexing; ``self_ctx_pct`` is the compressed
                        ctx the subject should serve at (or None if unchanged),
                        ``compress`` is the ordered neighbour-compression plan
                        (list of ``{"model_key","from_ctx_pct","to_ctx_pct",
                        "kv_freed"}``), and ``freed_bytes`` is the total the flex
                        yields. The caller applies the self-ctx compression and
                        (worker-enforcement, next cut) the neighbour compression.
      * ``"evict"``   — flexing cannot fit; hand off to the eviction engine.
                        ``priority_order`` is the flex-priority-ascending +
                        pin-tiebreak order the evictor should yield candidates in
                        (lowest priority / most-compressible first).
    ``deficit_bytes`` is what still had to be found after self-flex; ``note`` is
    a short human trace of which stage resolved it."""

    __slots__ = ("action", "self_ctx_pct", "compress", "freed_bytes",
                 "priority_order", "deficit_bytes", "note")

    def __init__(self, action, self_ctx_pct=None, compress=None, freed_bytes=0,
                 priority_order=None, deficit_bytes=0, note=""):
        self.action = action
        self.self_ctx_pct = self_ctx_pct
        self.compress = compress or []
        self.freed_bytes = int(freed_bytes or 0)
        self.priority_order = priority_order or []
        self.deficit_bytes = int(deficit_bytes or 0)
        self.note = note

    def as_dict(self) -> dict:
        return {"action": self.action, "self_ctx_pct": self.self_ctx_pct,
                "compress": self.compress, "freed_bytes": self.freed_bytes,
                "priority_order": self.priority_order,
                "deficit_bytes": self.deficit_bytes, "note": self.note}


def _neighbour_sort_key(n: dict) -> tuple:
    """Yield/compress order for a flex-eligible neighbour: LOWEST priority first
    (a higher-priority subject compresses lower-priority neighbours), then
    UNPINNED before pinned (the existing pin-as-tiebreak doctrine — pin buys a
    hair of precedence at an exact priority tie, nothing more), then the largest
    compressible headroom first so the fewest neighbours are disturbed."""
    return (flex_priority_key(n.get("alloc")),
            bool(n.get("pinned")),
            -int(n.get("flex_headroom_bytes") or 0))


def plan_flex(subject: dict, residents: list, deficit_bytes: int) -> FlexPlan:
    """Decide whether ``deficit_bytes`` of VRAM pressure for ``subject`` can be
    absorbed by FLEXING within bands instead of evicting — stage (1).

    PURE. Never evicts, never mutates. The caller (agent._vram_evict_to_fit)
    runs this BEFORE its LRU eviction planner: a ``flex`` verdict means the load
    fits without evicting anyone; an ``evict`` verdict hands back the priority
    order the existing evictor should use.

    ``subject`` = ``{"weights_bytes", "kv_bytes", "ctx_pct", "ctx_deviation_pct",
        "priority"}`` — the incoming load's need split + its ctx band + priority.
    ``residents`` = list of ``{"model_key", "kv_bytes", "ctx_pct",
        "ctx_deviation_pct", "vram_bytes", "protected"(bool), "pinned"(bool),
        "alloc": {"priority": int}}`` — the current GPU residents.
    ``deficit_bytes`` = free_needed − free_have (>0; the shortfall to clear the
    ceiling). Callers only invoke this when the target does NOT fit, so a caller
    that passes deficit<=0 gets an immediate ``proceed``.

    Order of operations (ctx-first, self before neighbours, floor-respecting):
      1. SELF-FLEX: compress the subject's OWN ctx toward its band floor,
         re-pricing KV linearly. This shrinks the deficit at zero disruption.
      2. NEIGHBOUR-FLEX: only a higher-priority subject may reclaim resident KV
         by compressing lower-priority, UNPROTECTED neighbours toward THEIR ctx
         floors (priority-ascending, pin-tiebroken). Sum their headroom until the
         (post-self) deficit is covered.
      3. If self+neighbour flex covers the deficit -> ``flex``. Else -> ``evict``
         with the priority order for the eviction engine.
    """
    if deficit_bytes <= 0:
        return FlexPlan("proceed", note="fits at target — bands untouched")

    remaining = int(deficit_bytes)
    note_parts = []

    # ── stage 1: SELF-FLEX (compress the subject's own ctx to its band floor) ──
    self_ctx_pct = None
    subj_ctx = subject.get("ctx_pct")
    subj_dev = subject.get("ctx_deviation_pct")
    subj_kv = int(subject.get("kv_bytes") or 0)
    band = ctx_band_bounds(subj_ctx, subj_dev)
    if band is not None and subj_kv > 0:
        floor_pct, _ceil = band
        if floor_pct < int(subj_ctx):
            kv_floor = kv_at_ctx_pct(subj_kv, int(subj_ctx), floor_pct)
            saved = max(0, subj_kv - kv_floor)
            if saved > 0:
                self_ctx_pct = floor_pct
                remaining -= saved
                note_parts.append(
                    f"self ctx {subj_ctx}%->{floor_pct}% freed {saved}B")

    if remaining <= 0:
        return FlexPlan("flex", self_ctx_pct=self_ctx_pct,
                        freed_bytes=deficit_bytes - remaining,
                        deficit_bytes=0,
                        note="; ".join(note_parts) or "self-flex fit")

    # ── stage 2: NEIGHBOUR-FLEX (higher-priority subject compresses lowers) ────
    subj_prio = flex_priority_key(
        {"priority": subject.get("priority")})
    eligible = []
    for r in residents:
        if r.get("protected"):
            continue                       # protection outranks priority, always
        rb = ctx_band_bounds(r.get("ctx_pct"), r.get("ctx_deviation_pct"))
        rkv = int(r.get("kv_bytes") or 0)
        if rb is None or rkv <= 0:
            continue                       # nothing to compress
        r_floor, _ = rb
        if r_floor >= int(r.get("ctx_pct")):
            continue                       # no compressible headroom
        # ONLY a strictly-higher-priority subject may compress a neighbour.
        if subj_prio <= flex_priority_key(r.get("alloc")):
            continue
        kv_floor = kv_at_ctx_pct(rkv, int(r["ctx_pct"]), r_floor)
        headroom = max(0, rkv - kv_floor)
        if headroom <= 0:
            continue
        eligible.append({**r, "flex_headroom_bytes": headroom,
                         "to_ctx_pct": r_floor})

    eligible.sort(key=_neighbour_sort_key)

    compress = []
    for n in eligible:
        if remaining <= 0:
            break
        compress.append({"model_key": n["model_key"],
                         "from_ctx_pct": int(n["ctx_pct"]),
                         "to_ctx_pct": int(n["to_ctx_pct"]),
                         "kv_freed": int(n["flex_headroom_bytes"])})
        remaining -= int(n["flex_headroom_bytes"])

    if remaining <= 0:
        if compress:
            note_parts.append(
                f"compressed {len(compress)} lower-priority neighbour(s)")
        return FlexPlan("flex", self_ctx_pct=self_ctx_pct, compress=compress,
                        freed_bytes=deficit_bytes - remaining,
                        deficit_bytes=0,
                        note="; ".join(note_parts) or "neighbour-flex fit")

    # ── stage 3: flex cannot fit -> hand the evictor a priority order ──────────
    # Lowest priority / most-compressible first (unprotected only; the evictor
    # re-applies protection itself). This is the bridge until in-place resident
    # KV-shrink is enforced worker-side (next cut): a higher-priority subject's
    # eviction still prefers the lower-priority neighbours first.
    order = [r["model_key"] for r in
             sorted((r for r in residents if not r.get("protected")),
                    key=_neighbour_sort_key)]
    if note_parts:
        note_parts.append("flex insufficient — evict")
    return FlexPlan("evict", self_ctx_pct=self_ctx_pct, compress=compress,
                    freed_bytes=deficit_bytes - remaining,
                    priority_order=order, deficit_bytes=remaining,
                    note="; ".join(note_parts) or "no flex headroom — evict")
