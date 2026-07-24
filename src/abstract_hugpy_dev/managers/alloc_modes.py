"""The FIVE allocation modes — the operator-facing placement vocabulary (k37).

Operator spec 2026-07-23 (allocation-modes-spec): every model gets ONE of five
FLAT modes. Each mode = placement preference + bust condition:

    gpu-only   all layers on the GPU, no spill; won't fit GPU (after evict) -> bust
    ram-only   all in RAM, never the GPU (binds CPU even with a GPU present);
               won't fit RAM -> bust
    max-gpu    as much GPU as available+needed, spill the rest to RAM;
               can't satisfy -> bust                      (THE DEFAULT — a blank
               model serves-and-spills, never OOMs: defaults-are-promises)
    max-ram    as much RAM as available+needed, spill the rest to GPU;
               can't satisfy -> bust
    explicit   target VRAM/RAM budgets + a leniency%% + a device priority
               (gpu default); can't fit even at the loosened floor -> bust

Internally max-gpu / max-ram are explicit(priority-device, ~100%% target,
generous leniency) — but they stay FIVE FLAT NAMES on every surface. The split
is a COGNITIVE-LOAD boundary, not an implementation one (operator ruling):
"use my GPU, spill the rest" must stay a zero-knob pick; the moment you tune
(%%/leniency/priority) you reach for explicit. Never collapse max-* into
explicit in any UI/API surface.

LENIENCY MATH (operator-confirmed): N%% leniency = up to N%% OF THE MODEL may
land off its ideal device before bust. Ideal 100%% GPU + 30%% leniency ->
degrade step-by-step down to the FLOOR 70%% GPU / 30%% RAM; only bust when even
the floor won't fit. The conversion onto the tolerance-band engine
(worker_agent/flex.py) is: whole = the MODEL's bytes, deviation = leniency%%,
so ``band_floor(target_bytes, leniency_pct, model_bytes)`` IS the floor.

THE HONEST RENAME (keeper owns nomenclature): today's console "Max GPU"
(n_gpu_layers=-1, all-or-OOM) is really **gpu-only**; today's "autofit"
(as-much-GPU-as-fits, spill rest) is really **max-gpu**. Legacy names are
accepted on INPUT (resolved + logged), never emitted back. NOTE the collision:
the STRING "max-gpu" exists in both vocabularies with different meanings —
resolution is canonical-first, so "max-gpu" always reads as the NEW max-gpu
(fit-and-spill); the historical -1 meaning is only reachable as "gpu-only"
(nothing persisted ever stored the old string — the old wire encoding was the
n_gpu_layers value itself, which read-time DERIVATION maps correctly).

WIRE ENCODING (unchanged for the three legacy-expressible modes — keeper
amendment 3: n_gpu_layers semantics NEVER change on the wire):
    gpu-only -> {"n_gpu_layers": -1}
    ram-only -> {"n_gpu_layers": "off"}
    max-gpu  -> {}                        (autofit, zero knobs)
    max-ram  -> {"alloc_mode": "max-ram"}                       (NEW keys)
    explicit -> {"alloc_mode": "explicit", gpu_mem_gib?, cpu_mem_gib?,
                 "leniency_pct"?, "priority_device"?, priority?} (NEW keys)

VERSION GATE (no dead knobs): the NEW spill keys are only emitted to workers
whose pkg_version honors them (>= MODE_MIN_PKG_VERSION). Released workers
IGNORE unknown spill keys (verified safe), but a selected mode must never be a
silent no-op — an old worker's request is downgraded to max-gpu ({} autofit)
and the downgrade is logged/surfaced (gate_spill_for_worker).

This module is PURE (stdlib only) so chaos, the routes, overrides, and the
worker can all share ONE vocabulary without import weight.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The five flat operator-facing modes. Order matters only for display.
ALLOC_MODES = ("gpu-only", "ram-only", "max-gpu", "max-ram", "explicit")

# Legacy name -> canonical mode. Accepted on input, resolved + logged, never
# emitted back. "max-gpu" is listed for the HISTORICAL record (old "Max GPU"
# = -1 = new gpu-only) but is UNREACHABLE at runtime: resolution is
# canonical-first and "max-gpu" is a canonical name of the new vocabulary.
LEGACY_ALLOC_ALIASES = {
    "autofit": "max-gpu",     # old fit-and-spill default -> the new honest name
    "cpu-only": "ram-only",   # old CPU-only -> RAM is what it actually binds
    "cpu_only": "ram-only",
    "max-gpu": "gpu-only",    # HISTORICAL ONLY (all-or-OOM); see module note
    "max_gpu": "gpu-only",    # (underscore form has no canonical collision)
    "gpu_only": "gpu-only",
    "ram_only": "ram-only",
    "max_ram": "max-ram",
    "budget": "explicit",     # old chaos budget draw = an explicit gpu budget
    "bands": "explicit",      # old chaos band draw = explicit with a band
}

# Spill keys that only a mode-aware worker understands. Presence of ANY of
# these on a spill makes it version-gated (gate_spill_for_worker).
NEW_SPILL_KEYS = frozenset({"alloc_mode", "leniency_pct", "priority_device"})

# First worker package version whose spill/env plumbing honors the new keys
# (Slice B2 ships in this cut). Anything older gets the max-gpu fallback.
MODE_MIN_PKG_VERSION = "0.1.203"

# Modes a non-GGUF (transformers/comfy) model may select: accelerate handles
# the three coarse placement intents through the wired loaders; max-ram and
# explicit need fine-grained placement the gap loaders don't wire yet
# (Slice C) — they are refused honestly at /assign for non-GGUF models.
NONGGUF_ALLOWED_MODES = ("gpu-only", "ram-only", "max-gpu")


def resolve_alloc_mode(name: Any) -> "tuple[Optional[str], bool]":
    """``(canonical_mode, was_alias)`` for a mode name, canonical-first.

    Canonical names pass through untouched. A legacy alias resolves to its
    canonical mode with ``was_alias=True`` (callers log; never emit the alias
    back). Unknown/empty -> ``(None, False)`` — the caller degrades (keeps its
    default) rather than raising."""
    if name is None:
        return None, False
    s = str(name).strip().lower()
    if not s:
        return None, False
    if s in ALLOC_MODES:
        return s, False
    alias = LEGACY_ALLOC_ALIASES.get(s)
    if alias:
        return alias, True
    return None, False


def derive_alloc_mode(override: "Optional[dict]") -> str:
    """The model's EFFECTIVE mode from a persisted override/spill dict —
    read-time derivation, the migration (no file rewrite ever needed).

      * an explicit ``alloc_mode`` wins (aliases resolved);
      * else the legacy wire value: n_gpu_layers -1 -> gpu-only,
        0/"off"/"cpu"/"none" -> ram-only;
      * else an explicit-budget/band/leniency contract -> explicit
        (today's "budget"/"bands" spills ARE explicit allocations);
      * else -> max-gpu (THE DEFAULT: a blank model fits-and-spills, never
        OOMs — defaults-are-promises).
    """
    ov = override or {}
    got, was_alias = resolve_alloc_mode(ov.get("alloc_mode"))
    if got:
        if was_alias:
            logger.info("alloc_mode legacy name %r resolved to %r",
                        ov.get("alloc_mode"), got)
        return got
    ngl = ov.get("n_gpu_layers")
    if ngl is not None:
        s = str(ngl).strip().lower()
        if s == "-1":
            return "gpu-only"
        if s in ("0", "off", "cpu", "none"):
            return "ram-only"
        # positive layer count / "auto": a fit-and-spill flavor -> max-gpu
        # (the explicit layer count still rides the wire untouched).
    for k in ("leniency_pct", "gpu_mem_gib", "cpu_mem_gib",
              "gpu_mem_gib_deviation_pct", "cpu_mem_gib_deviation_pct"):
        if ov.get(k) is not None:
            return "explicit"
    return "max-gpu"


def mode_to_spill(mode: Any, *, ctx_pct: "Optional[int]" = None,
                  gpu_mem_gib: "Optional[float]" = None,
                  cpu_mem_gib: "Optional[float]" = None,
                  leniency_pct: "Optional[float]" = None,
                  priority: "Optional[int]" = None,
                  priority_device: "Optional[str]" = None) -> dict:
    """Materialize a mode (+ optional explicit knobs) into the /assign spill
    contract. Legacy aliases are resolved first. Unknown mode -> {} (max-gpu),
    logged — degrade, never raise."""
    canonical, was_alias = resolve_alloc_mode(mode)
    if canonical is None:
        if mode not in (None, ""):
            logger.warning("unknown alloc mode %r -> defaulting to max-gpu", mode)
        canonical = "max-gpu"
    elif was_alias:
        logger.info("alloc_mode legacy name %r resolved to %r", mode, canonical)
    spill: dict = {}
    if canonical == "gpu-only":
        spill = {"n_gpu_layers": -1}
    elif canonical == "ram-only":
        return {"n_gpu_layers": "off"}          # ctx irrelevant off-GPU
    elif canonical == "max-gpu":
        spill = {}
    elif canonical == "max-ram":
        spill = {"alloc_mode": "max-ram"}
    elif canonical == "explicit":
        spill = {"alloc_mode": "explicit"}
        if gpu_mem_gib is not None:
            spill["gpu_mem_gib"] = float(gpu_mem_gib)
        if cpu_mem_gib is not None:
            spill["cpu_mem_gib"] = float(cpu_mem_gib)
        if leniency_pct is not None:
            spill["leniency_pct"] = float(leniency_pct)
        if priority is not None:
            spill["priority"] = int(priority)
        if priority_device is not None:
            spill["priority_device"] = str(priority_device)
    if ctx_pct is not None and canonical != "max-gpu":
        spill["ctx_pct"] = int(ctx_pct)
    return spill


def normalize_spill(spill: "Optional[dict]") -> "tuple[dict, Optional[str]]":
    """Normalize a client-supplied spill's ``alloc_mode`` value IN PLACE of the
    wire encoding: legacy aliases resolve to canonical; the three legacy-
    expressible modes (gpu-only / ram-only / max-gpu) are REWRITTEN onto the
    unchanged legacy wire (n_gpu_layers / {}), so ``alloc_mode`` only ever
    survives on the wire for max-ram / explicit (the version-gated pair).

    Returns ``(normalized_spill, note)`` — note is a human line when something
    was resolved/rewritten (for logs / say-why), None when untouched. Unknown
    mode values are DROPPED with a note (degrade-not-500; the rest of the
    spill still applies)."""
    if not isinstance(spill, dict) or "alloc_mode" not in spill:
        return (dict(spill) if isinstance(spill, dict) else {}), None
    out = dict(spill)
    raw = out.pop("alloc_mode")
    canonical, was_alias = resolve_alloc_mode(raw)
    if canonical is None:
        note = (f"unknown alloc_mode {raw!r} dropped (recognized: "
                f"{', '.join(ALLOC_MODES)}); rest of the spill still applies")
        logger.warning("normalize_spill: %s", note)
        return out, note
    note = (f"alloc_mode {raw!r} resolved to {canonical!r}" if was_alias else None)
    if canonical == "gpu-only":
        out["n_gpu_layers"] = -1
    elif canonical == "ram-only":
        out["n_gpu_layers"] = "off"
    elif canonical == "max-gpu":
        out.pop("n_gpu_layers", None)       # {} / no layer knob IS max-gpu
    else:                                   # max-ram / explicit keep the key
        out["alloc_mode"] = canonical
    if note:
        logger.info("normalize_spill: %s", note)
    return out, note


def _ver_tuple(v: Any) -> "Optional[tuple]":
    try:
        parts = str(v).strip().split(".")
        return tuple(int(p) for p in parts) if parts else None
    except (TypeError, ValueError):
        return None


def worker_honors_mode_keys(pkg_version: Any) -> bool:
    """True when a worker's reported package version understands the NEW spill
    keys (>= MODE_MIN_PKG_VERSION). Unknown/unparseable -> False (fail SAFE:
    never ship a knob we can't prove the worker reads)."""
    have = _ver_tuple(pkg_version)
    need = _ver_tuple(MODE_MIN_PKG_VERSION)
    return have is not None and need is not None and have >= need


def gate_spill_for_worker(spill: "Optional[dict]", pkg_version: Any,
                          worker_name: str = "") -> "tuple[dict, Optional[str]]":
    """THE version gate at emission: a spill carrying NEW mode keys is only
    sent to a worker that honors them; an older worker gets max-gpu ({}
    autofit) for that request, with a note the caller logs/surfaces.

    Returns ``(spill_to_emit, downgrade_note)``. A spill with no new keys
    passes through untouched (None note) regardless of version."""
    s = dict(spill or {})
    if not (set(s) & NEW_SPILL_KEYS):
        return s, None
    if worker_honors_mode_keys(pkg_version):
        return s, None
    mode = s.get("alloc_mode") or "explicit"
    note = (f"worker {worker_name or '?'} (pkg {pkg_version or 'unknown'}) "
            f"predates allocation-mode spill keys (needs >= "
            f"{MODE_MIN_PKG_VERSION}); '{mode}' downgraded to max-gpu "
            f"(autofit) for this request — update the worker to honor it")
    return {}, note
