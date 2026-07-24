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
# these on a spill makes it version-gated (gate_spill_for_worker). n_cpu_moe
# (the MoE expert-split knob, 2026-07-24) ships in the SAME cut as the mode
# keys, so the one MODE_MIN_PKG_VERSION gate covers it — an older worker would
# silently drop the knob (unknown spill keys are ignored), and a selected knob
# must never be a silent no-op.
NEW_SPILL_KEYS = frozenset({"alloc_mode", "leniency_pct", "priority_device",
                            "n_cpu_moe"})

# First worker package version whose spill/env plumbing honors the new keys
# (Slice B2 ships in this cut). Anything older gets the max-gpu fallback.
MODE_MIN_PKG_VERSION = "0.1.203"

# Modes a non-GGUF (transformers/comfy) model may select: accelerate handles
# the three coarse placement intents through the wired loaders; max-ram and
# explicit need fine-grained placement the gap loaders don't wire yet
# (Slice C) — they are refused honestly at /assign for non-GGUF models.
NONGGUF_ALLOWED_MODES = ("gpu-only", "ram-only", "max-gpu")

# GGUF family: the HF-canonical 'gguf' plus the llama_cpp synonym. A GGUF model
# is ALWAYS max-gpu by default (partial offload makes any size feasible — the
# runner spills whatever won't fit to RAM), so its feasible default never
# depends on the box's totals. Everything else (transformers / comfy) loads
# whole-tensor via accelerate, so an oversized model genuinely cannot land on
# the GPU and its feasible default is worker-dependent.
GGUF_ENGINES = frozenset({"gguf", "llama_cpp"})

# Headroom factor for the transformers GPU-fit test (feasible_default_mode).
# A transformers model whose effective footprint exceeds this fraction of the
# box's TOTAL GPU capacity "clearly cannot fit the GPU" and defaults to
# ram-only on that worker. 0.9 leaves ~10% for the CUDA context, activation
# working set, and KV/attention scratch that ride alongside the weights — a
# model at 0.95× total VRAM would OOM the moment it allocated its first
# forward-pass buffer, so 0.9 is the defensible "clearly cannot fit" line
# (not a fit PREDICTION — the real autofit/accelerate placement still runs; this
# only picks the read-time DEFAULT so a doomed max-gpu is never the blank
# promise). Above 0.9× GPU but within RAM -> ram-only; above RAM too -> leave
# max-gpu and let the worker refuse honestly (no invented fourth state).
_GPU_FIT_HEADROOM = 0.9


def is_gguf_engine(engine: Any) -> bool:
    """True for the GGUF family (gguf / llama_cpp), case-insensitive. Anything
    else — including None/unknown — is treated as non-GGUF by the caller, but
    the feasible-default logic degrades unknown engines to max-gpu regardless."""
    return str(engine or "").strip().lower() in GGUF_ENGINES


def _as_int(v: Any) -> "Optional[int]":
    try:
        return int(v) if v else None
    except (TypeError, ValueError):
        return None


def feasible_modes(engine: Any,
                   model_bytes: "Optional[int]",
                   gpu_total_bytes: "Optional[int]",
                   ram_total_bytes: "Optional[int]",
                   moe_split_gpu_bytes: "Optional[int]" = None) -> tuple:
    """The allocation modes that are FEASIBLE for one (model x worker), in
    ALLOC_MODES display order (operator ruling 2026-07-24 scope-extension: "the
    user shouldn't be able to select an option that implies something it cannot
    do"). The blank default is the best member of this set (feasible_default_mode
    returns exactly that). PURE — the enforcement/surface glue lives at the
    routes.

    Feasibility matrix (each rule is a hard can-it-physically-land test, not a
    fit prediction — the real placement still runs; this only bounds what may be
    SELECTED so an impossible mode is never offered):

      * ``gpu-only`` — the model fits the GPU total within the headroom factor
        (``model <= _GPU_FIT_HEADROOM * gpu_total``). All-or-bust on the GPU, so
        it must plausibly fit the GPU alone.
      * ``ram-only`` — the model fits RAM total (``model <= ram_total``). Binds
        the CPU; never touches the GPU.
      * ``max-gpu`` — GGUF: ALWAYS (partial offload spills whatever won't fit to
        RAM, so it is universally feasible). Transformers/comfy: ONLY if the
        model fits the GPU total (same headroom test as gpu-only) — the gap
        loaders place whole-tensor, so an oversized transformers model genuinely
        cannot use the GPU and max-gpu must NOT be offered (the operator's
        68 GB-on-24 GB case).
      * ``max-ram`` — a split exists: the model fits RAM+GPU COMBINED (its
        overflow rides the GPU). ENGINE-GATED: non-GGUF stays refused (409) until
        Slice C wires fine-grained transformers placement, so it is dropped for
        non-GGUF here regardless of the numbers.
      * ``explicit`` — some split exists (``model <= gpu_total + ram_total``);
        ENGINE-GATED the same as max-ram (non-GGUF dropped until Slice C).

    UNKNOWN size or unknown totals -> ALL modes feasible: never eliminate on
    missing data (degrade to today's permissiveness). Specifically, if
    ``model_bytes`` is unknown, or the total a rule needs is unknown, that rule
    does NOT eliminate its mode. The engine gate on max-ram/explicit is the ONLY
    elimination that fires without size/totals (it is a capability fact, not a
    measurement).

    ``moe_split_gpu_bytes`` (MoE, 2026-07-24): for a detected-MoE GGUF the
    caller passes the GPU-side need of the expert split (non-expert bytes —
    surfaced by gguf_variants_detail's ``moe`` at enrichment). GPU-fit tests
    (gpu-only / a non-GGUF-style max-gpu check) then price THAT instead of the
    full file: under the auto policy (and/or an operator ``n_cpu_moe``) the
    card only ever holds the non-expert share, so eliminating gpu-only against
    the full 41.6GB would wrongly bar a mode the split makes serveable. Dense
    models pass None — byte-identical."""
    size = _as_int(model_bytes)
    gpu_total = _as_int(gpu_total_bytes)
    ram_total = _as_int(ram_total_bytes)
    gguf = is_gguf_engine(engine)
    unknown_size = size is None
    # The GPU-side footprint used for GPU-fit tests: the MoE split's non-expert
    # share when known (never larger than the full size), else the full size.
    moe_gpu = _as_int(moe_split_gpu_bytes)
    gpu_size = min(size, moe_gpu) if (size is not None and moe_gpu) else size

    out = []
    for mode in ALLOC_MODES:
        if mode == "gpu-only":
            # Fits GPU (headroom). Unknown size/gpu_total -> don't eliminate.
            feasible = (unknown_size or gpu_total is None
                        or gpu_size <= _GPU_FIT_HEADROOM * gpu_total)
        elif mode == "ram-only":
            feasible = (unknown_size or ram_total is None
                        or size <= ram_total)
        elif mode == "max-gpu":
            if gguf:
                feasible = True                  # partial offload: universal
            else:
                feasible = (unknown_size or gpu_total is None
                            or size <= _GPU_FIT_HEADROOM * gpu_total)
        elif mode in ("max-ram", "explicit"):
            # Engine gate (a capability fact, fires even on missing data).
            if not gguf:
                feasible = False
            else:
                combined = None
                if gpu_total is not None or ram_total is not None:
                    combined = (gpu_total or 0) + (ram_total or 0)
                feasible = (unknown_size or combined is None
                            or size <= combined)
        else:  # pragma: no cover — ALLOC_MODES is closed
            feasible = True
        if feasible:
            out.append(mode)
    # Never return an empty set — a model must always have SOMETHING selectable
    # (defaults-are-promises). If the numbers eliminated everything (e.g. a
    # transformers model bigger than RAM and GPU), fall back to max-gpu so the
    # worker can refuse HONESTLY downstream rather than the UI offering nothing.
    return tuple(out) if out else ("max-gpu",)


def feasible_default_mode(engine: Any,
                          model_bytes: "Optional[int]",
                          gpu_total_bytes: "Optional[int]",
                          ram_total_bytes: "Optional[int]") -> str:
    """The BLANK default alloc mode derived by FEASIBILITY for one (model x
    worker), engine-aware (operator ruling 2026-07-24). This ONLY supplies the
    default when NOTHING is persisted — an explicit alloc_mode always wins
    upstream; this is never consulted for a model that has one.

      * GGUF (any size) -> ``max-gpu`` ALWAYS. Partial offload makes every size
        feasible on any GPU (spill the rest to RAM), so this is today's blank
        default, unchanged, and independent of the box totals.
      * transformers/comfy:
          - if the footprint CLEARLY cannot fit the GPU
            (``model_bytes > _GPU_FIT_HEADROOM * gpu_total_bytes``) but DOES fit
            RAM (``model_bytes <= ram_total_bytes``) -> ``ram-only`` (emits the
            legacy ``{"n_gpu_layers": "off"}`` — works on ANY worker version, no
            gate). This is the "68 GB model, 24 GB GPU, 124 GB RAM -> RAM-only"
            case: the only feasible option, so it IS the default.
          - if it plausibly fits the GPU -> ``max-gpu`` (today's default;
            autofit/accelerate handles it).
          - if it fits NEITHER (bigger than RAM too) -> ``max-gpu`` and let the
            worker refuse honestly (no invented fourth state).

    DEGRADE-NOT-GUESS: any missing input (unknown size, unknown GPU total, or —
    for the ram-only decision — unknown RAM total) falls back to ``max-gpu``,
    today's behavior. A default is never derived from a guessed number."""
    if is_gguf_engine(engine):
        return "max-gpu"
    # Non-GGUF: need a real size and a real GPU total to say anything.
    if not model_bytes or not gpu_total_bytes:
        return "max-gpu"
    try:
        size = int(model_bytes)
        gpu_total = int(gpu_total_bytes)
    except (TypeError, ValueError):
        return "max-gpu"
    if size <= _GPU_FIT_HEADROOM * gpu_total:
        return "max-gpu"                       # plausibly fits the GPU
    # Clearly can't fit the GPU. RAM-only only if it actually fits RAM AND we
    # know RAM (an unknown RAM total can't justify ram-only -> leave max-gpu).
    if not ram_total_bytes:
        return "max-gpu"
    try:
        ram_total = int(ram_total_bytes)
    except (TypeError, ValueError):
        return "max-gpu"
    if size <= ram_total:
        return "ram-only"                      # the only feasible landing
    return "max-gpu"                           # fits neither -> honest refusal downstream


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
