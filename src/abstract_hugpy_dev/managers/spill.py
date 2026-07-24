"""GPU/CPU spill configuration — how much of a model lives on the GPU.

Two backends, two knobs:

  * llama.cpp (GGUF): ``n_gpu_layers`` passed to ``Llama(...)``.
        -1 = every layer on GPU, 0 = pure CPU, N = first N transformer layers
        on GPU (the rest spill to CPU/RAM). This is the ONLY thing that lights
        up the GPU for GGUF models — without it llama.cpp runs CPU-only.
  * transformers: ``max_memory`` passed to ``from_pretrained(device_map="auto")``,
        e.g. ``{0: "7GiB", "cpu": "32GiB"}`` — accelerate then shards layers to
        fit the per-device budget, spilling the overflow to CPU.

Config comes from environment variables so the worker agent (which owns the
process) can set it per model load without threading new fields through the
resolver/dispatch chain:

    HUGPY_N_GPU_LAYERS   "auto" | "off" | int   llama.cpp layers on GPU
    HUGPY_TENSOR_SPLIT   csv floats             multi-GPU split e.g. "0.7,0.3"
    HUGPY_MAIN_GPU       int                    primary GPU index (llama.cpp)
    HUGPY_GPU_MEM_GIB    float                  per-GPU budget (transformers)
    HUGPY_CPU_MEM_GIB    float                  CPU/RAM budget (transformers)
    HUGPY_N_GPU          int                    #GPUs to spread across

Everything is optional. Default mode is "auto": detect free VRAM, estimate the
model's size, and fit as many layers as will hold — falling back to CPU (0
layers) when no GPU is visible, so a CPU-only host behaves exactly as before.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Optional

logger = logging.getLogger("abstract_hugpy_dev.spill")

# Keep some VRAM headroom for the KV-cache + activations; never budget 100%.
_VRAM_SAFETY = 0.85
# When we can't read a GGUF's real layer count, assume a 7B-ish 32-layer model.
_ASSUMED_LAYERS = 32


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _env_float(name: str) -> Optional[float]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return None


def _env_int(name: str) -> Optional[int]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return None


# ---------------------------------------------------------------------------
# hardware probes (best-effort, never raise)
# ---------------------------------------------------------------------------
def vram_reserve_bytes() -> int:
    """VRAM kept out of every budget (HUGPY_VRAM_RESERVE_GIB, default 1.0).

    The box may have GPU consumers central knows nothing about (a desktop
    session, another app). Reserving a slice at the probe layer means autofit,
    preflights, and heartbeat-fed budgets all leave it alone, while the raw
    per-GPU numbers shown in the console stay truthful."""
    gib = _env_float("HUGPY_VRAM_RESERVE_GIB")
    return int((1.0 if gib is None else gib) * 2**30)


def ram_reserve_bytes() -> int:
    """RAM kept out of every budget (HUGPY_RAM_RESERVE_GIB, default 4.0).

    Same idea as the VRAM reserve: local processes central can't see need
    room, and a load that consumes MemAvailable to the floor gets the whole
    agent OOM-killed mid-request."""
    gib = _env_float("HUGPY_RAM_RESERVE_GIB")
    return int((4.0 if gib is None else gib) * 2**30)


# ---------------------------------------------------------------------------
# Honest budget-bar semantics (t13/t14, operator spec 2026-07-17, REFINED)
# ---------------------------------------------------------------------------
# The console resource bars used to draw a numerator and denominator from
# different universes (physical-derived "used" vs central-limit "total"), so on
# any under-budget box the bar collapsed to physical_total − central_limit — an
# ARTIFACT, not usage. The operator specified the honest model, adopted as
# doctrine. It is IDENTICAL for RAM and VRAM, so it lives here once and is used
# by BOTH the central summary (the bar) and the worker allocator (budgetable
# free) — bar and admission can then never disagree.
#
# Final formula (both clamps mandatory — operator refinement 2026-07-17,
# "the limit can never lead into negative, but the limit should not be
#  encroached by ram unless it exceeds the difference in worker process"):
#
#   external_headroom = physical_total − central_limit    # never the worker's
#   encroachment      = max(0, external_usage − external_headroom)
#   bar_used          = min(central_limit, worker_usage + encroachment)  # ≤ limit
#   remaining         = max(0, central_limit − worker_usage − encroachment)  # ≥ 0
#
# The central_limit is the WORKER'S budget. External consumers (a desktop
# session, ComfyUI, another app) first spend their OWN headroom (physical above
# the limit); only what they use BEYOND that headroom encroaches on the worker's
# budget. Worked example (operator): 128 physical / 90 limit / 20 worker / 10
# external → headroom 38, encroachment 0, bar 20/90, 70 to go. External grows to
# 50 → encroachment 12 → bar 32/90, 58 to go.
#
# OVER-LIMIT HONESTY (operator, 2026-07-17): the CLAMPS are a RENDER/admission
# rule — the display never goes negative and the fill never overflows — but a
# genuine overrun (raw worker_usage + encroachment > central_limit) is never
# hidden behind a clean full bar. So the payload carries BOTH the clamped
# figures (bar_used/remaining, for the fill + admission) AND the RAW ones
# (raw_used, over_limit, over_by) so central/console can pin the chip at 100%
# and surface an explicit over-limit warning. The allocator floors remaining at
# 0 the same way: an over-limit box admits nothing new until it drains.
def budget_bar(physical_total: Optional[int],
               central_limit: Optional[int],
               worker_usage: Optional[int],
               external_usage: Optional[int]) -> dict:
    """Compute the honest bar (t13/t14 spec, refined) from the four measured
    inputs.

    All arguments are bytes (or None where unmeasured). Returns a dict:
      * ``semantics="central"`` when a central_limit is set: bar_used/remaining
        follow the clamped spec; ``total`` is the limit; ``raw_used`` is the
        UNCLAMPED worker+encroachment; ``over_limit`` / ``over_by`` flag a true
        overrun.
      * ``semantics="physical"`` when NO central_limit is set: headroom is
        undefined, so the bar shows plain measured usage (worker+external)
        against the physical total; no encroachment, never over-limit.
    ``bar_used``/``remaining`` are None only when the necessary inputs are
    missing (never fabricated)."""
    w = worker_usage if worker_usage is not None else None
    x = external_usage if external_usage is not None else None
    # No central limit -> physical-total semantics (plain measured usage).
    if not central_limit or central_limit <= 0:
        parts = [v for v in (w, x) if v is not None]
        bar_used = sum(parts) if parts else None
        total = physical_total
        remaining = (max(0, total - bar_used)
                     if (total is not None and bar_used is not None) else None)
        return {"semantics": "physical", "total": total,
                "bar_used": bar_used, "remaining": remaining,
                "raw_used": bar_used, "over_limit": False, "over_by": 0,
                "encroachment": 0, "worker_usage": w, "external_usage": x,
                "external_headroom": None}
    # Central-limit semantics (the spec).
    headroom = None
    if physical_total is not None:
        headroom = max(0, physical_total - central_limit)
    encroachment = 0
    if x is not None and headroom is not None:
        encroachment = max(0, x - headroom)
    elif x is not None and headroom is None:
        # No physical read to derive headroom from — the safe, non-fabricating
        # choice is to treat all external usage as encroachment (the limit is
        # the only denominator we trust). Rare: a box that reports a limit but
        # no physical total.
        encroachment = x
    # RAW (unclamped) worker+encroachment — the truth central/console must keep.
    raw_used = None
    if w is not None:
        raw_used = w + encroachment
    elif encroachment:
        raw_used = encroachment
    # CLAMPED fill: never overflows the limit (the ≤ clamp).
    bar_used = min(central_limit, raw_used) if raw_used is not None else None
    # CLAMPED remaining: never negative (the ≥0 clamp). Derived from the raw
    # worker+encroachment, floored at 0 — the figure admission also uses.
    remaining = (max(0, central_limit - raw_used)
                 if raw_used is not None else None)
    over_by = (max(0, raw_used - central_limit) if raw_used is not None else 0)
    over_limit = over_by > 0
    return {"semantics": "central", "total": central_limit,
            "bar_used": bar_used, "remaining": remaining,
            "raw_used": raw_used, "over_limit": over_limit, "over_by": over_by,
            "encroachment": encroachment, "worker_usage": w,
            "external_usage": x, "external_headroom": headroom}


def free_vram_bytes() -> Optional[int]:
    """Budgetable free VRAM on the primary GPU (raw minus the operator
    reserve), or None if no GPU / can't tell."""
    from .._platform.hardware import free_vram_bytes as _free_vram

    raw = _free_vram(_env_int("HUGPY_MAIN_GPU") or 0)
    if raw is None:
        return None
    return max(0, raw - vram_reserve_bytes())


def total_vram_bytes() -> Optional[int]:
    """Total INSTALLED VRAM on the primary GPU in bytes, or None if no GPU /
    can't tell.

    Unlike ``free_vram_bytes`` this is RAW — the operator reserve is NOT
    subtracted, because total is a fixed physical property of the card (its
    capacity), while the reserve is a slice held out of the FREE budget. The
    VRAM-ceiling gate (Fix A) uses it as the denominator for the ~90% ceiling —
    "keep the physical card at/under N% full" is a statement about the whole
    card, so it must be the whole card. Same probe family as ``free_vram_bytes``
    (torch.cuda.mem_get_info total, then nvidia-smi), so ceiling and free reads
    come from the same ComfyUI-visible device truth. Degrades to None (never 0)
    so the ceiling gate can tell "unmeasurable" from "no capacity" and fail
    OPEN."""
    from .._platform.hardware import total_vram_bytes as _total_vram

    return _total_vram(_env_int("HUGPY_MAIN_GPU") or 0)


def ram_worker_bytes() -> Optional[int]:
    """RSS of THIS worker's own process tree (the agent + every slot child it
    spawned), in bytes. This is ``worker_usage`` for the budget-bar spec: the
    RAM hugpy itself holds, distinct from external processes central can't see.
    Best-effort via psutil (children(recursive=True)); None if unmeasurable."""
    try:
        import psutil
        me = psutil.Process()
        total = me.memory_info().rss
        for child in me.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:  # noqa: BLE001 — a child may exit mid-walk
                continue
        return int(total)
    except Exception:  # noqa: BLE001 — no psutil / permission: don't fabricate
        return None


def ram_external_bytes() -> Optional[int]:
    """RAM used by everything OUTSIDE this worker's process tree, in bytes:
    (box used) − (worker own RSS). ``external_usage`` for the budget-bar spec.

    Box-used = physical total − MemAvailable, read against the SAME psutil
    snapshot as the total so the two can't skew. Clamped ≥0. None when either
    side is unmeasurable (never fabricated)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        box_used = int(vm.total) - int(vm.available)
    except Exception:  # noqa: BLE001
        return None
    own = ram_worker_bytes()
    if own is None:
        return None
    return max(0, box_used - own)


def ram_max_bytes() -> Optional[int]:
    """The central/local RAM CEILING in bytes (HUGPY_RAM_MAX_GIB), or None if
    unset. Set by _apply_central_limits from central's limits.ram_max_gib; the
    budget-bar spec's ``central_limit`` for RAM."""
    cap = _env_float("HUGPY_RAM_MAX_GIB")
    return int(cap * 2**30) if cap is not None else None


def free_ram_raw_bytes() -> Optional[int]:
    """Reserve-adjusted budgetable free RAM, UNCLAMPED by the RAM ceiling:
    ``max(0, MemAvailable − reserve)``. This is the honest "free after reserve"
    the console needs to show physical-semantics bars and that the ceiling-aware
    budget below is derived from. None if the raw read fails."""
    from .._platform.hardware import free_ram_bytes as _free_ram

    raw = _free_ram()
    if raw is None:
        return None
    return max(0, raw - ram_reserve_bytes())


def free_ram_bytes() -> Optional[int]:
    """Budgetable free RAM the allocator's fit decisions consume.

    Reworked for the t13/t14 budget-bar spec so the bar and admission can NEVER
    disagree: the budgetable free is the SPEC's ``remaining`` for RAM, floored
    by the reserve-only free —

        budgetable = min(free_after_reserve, limit − worker_usage − encroachment)

    Interaction with HUGPY_RAM_RESERVE_GIB: the reserve is applied FIRST (in
    free_ram_raw_bytes) so a load can never consume MemAvailable to the OOM
    floor — that floor is independent of the central ceiling and always binds.
    The ceiling term then further constrains it to the WORKER'S budget: the
    limit minus what the worker already uses minus any external ENCROACHMENT
    (external usage that has spilled past the physical headroom into the
    worker's budget — spill.budget_bar). Where no ceiling is set, this is the
    old reserve-only behavior verbatim (limit term absent → min() is a no-op)."""
    raw = free_ram_raw_bytes()
    if raw is None:
        return None
    limit = ram_max_bytes()
    if limit is None:
        # No ceiling -> reserve-only behavior, exactly as before.
        return raw
    from .._platform.hardware import free_ram_bytes as _free_ram
    physical = None
    try:
        import psutil
        physical = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        physical = None
    bar = budget_bar(physical_total=physical, central_limit=limit,
                     worker_usage=ram_worker_bytes(),
                     external_usage=ram_external_bytes())
    remaining = bar.get("remaining")
    if remaining is None:
        # Couldn't compute the spec remaining (missing worker/external reads) —
        # degrade to the historical hard cap so behavior never gets LOOSER than
        # before: min(free_after_reserve, limit).
        return min(raw, limit)
    # The allocator's free is the tighter of the reserve floor and the spec
    # remaining — the bar's number, so admission and the console agree.
    return min(raw, remaining)


def cpu_resident_bytes(model_path: str, n_gpu_layers: int) -> Optional[int]:
    """Rough RAM footprint of a GGUF load: the weight share NOT offloaded."""
    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return None
    if n_gpu_layers == -1:
        return 0
    if n_gpu_layers <= 0:
        return file_bytes
    total = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    frac = min(1.0, max(0.0, 1.0 - (n_gpu_layers / max(total, 1))))
    return int(file_bytes * frac)


def _gguf_metadata(model_path: str, want_suffixes: tuple) -> dict:
    """Scan a GGUF header's KV table and return the values whose keys END WITH any
    of ``want_suffixes`` (e.g. ``.block_count``, ``.attention.head_count_kv``,
    ``.embedding_length``, ``.context_length``). Best-effort; {} on any issue.

    The GGUF geometry keys are namespaced by architecture (``qwen2.block_count``,
    ``llama.attention.head_count_kv``, …), so suffix-matching is arch-agnostic —
    verified 2026-07-17 against a real Qwen2.5-Coder-3B q4 gguf:
      qwen2.block_count=36, qwen2.attention.head_count=16,
      qwen2.attention.head_count_kv=2, qwen2.embedding_length=2048,
      qwen2.context_length=32768.
    (GGUF spec: github.com/ggml-org/ggml/blob/master/docs/gguf.md — the
    general/architecture KVs are the canonical model geometry.)"""
    out: dict = {}
    try:
        import struct

        with open(model_path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return out
            version = struct.unpack("<I", fh.read(4))[0]
            if version < 2:
                return out
            struct.unpack("<Q", fh.read(8))[0]              # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n).decode("utf-8", "replace")

            # Minimal GGUF value-type reader — enough to scan the KV table.
            def read_val(t: int):
                simple = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i",
                          5: "<I", 6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
                if t in simple:
                    fmt = simple[t]
                    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
                if t == 8:                                   # string
                    return read_str()
                if t == 9:                                   # array
                    et = struct.unpack("<I", fh.read(4))[0]
                    n = struct.unpack("<Q", fh.read(8))[0]
                    return [read_val(et) for _ in range(n)]
                raise ValueError(f"unknown gguf type {t}")

            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack("<I", fh.read(4))[0]
                val = read_val(vtype)
                for suf in want_suffixes:
                    if key.endswith(suf):
                        out[suf] = val
                        break
    except Exception:
        return out
    return out


def _gguf_layer_count(model_path: str) -> Optional[int]:
    """Read ``*.block_count`` from a GGUF header. Best-effort; None on any issue."""
    bc = _gguf_metadata(model_path, (".block_count",)).get(".block_count")
    try:
        return int(bc) if bc is not None else None
    except (TypeError, ValueError):
        return None


# ── MoE detection + expert/non-expert byte split (2026-07-24 measured win) ──
# Measured on ae/3090 (Qwen3-Coder-Next, an 80B-A3B-class MoE): the naive layer
# split (autofit 17/48) gave ~15.2 tok/s @ 16.6 GiB VRAM; the MoE-aware split
# (--n-cpu-moe 999 + n_gpu_layers=-1: ALL attention/shared/KV on GPU, expert FFN
# tensors on CPU) gave ~24.1 tok/s @ 3.2 GiB VRAM — +59% AND 5x less VRAM.
# Mechanism: MoE bytes are mostly expert FFN tensors and each token touches only
# a few experts, so keeping the always-hot non-expert tensors on the GPU beats
# splitting whole layers. Dense models have no expert tensors -> the flag is a
# no-op and everything below reads "not MoE" (byte-identical behavior).
#
# Detection (operator-grounded 2026-07-24; verified against the real coder-next
# GGUF header — qwen3next.expert_count = 512, expert_used_count = 10):
#   * KV ``{arch}.expert_count`` — the arch-agnostic suffix ``.expert_count``.
#     ``expert_count == 0`` (or absent) IS the definition of dense: detection is
#     this ONE key, no heuristics. ``.expert_used_count`` rides for reporting.
#   * Tensor names literally mark the experts: ``blk.<i>.ffn_(gate|up|down)_exps.*``
#     — the ``_exps`` suffix is the per-tensor is_expert bit, with the layer
#     attribution ``<i>`` built in. The router (``ffn_gate_inp``) and the
#     shared experts (``ffn_*_shexp``) do NOT carry the suffix and stay on the
#     GPU, exactly as llama-server's --n-cpu-moe keeps them.
#   * The name is a CONVENTION, so it is only the fast/primary path — a
#     name-INDEPENDENT SHAPE backstop guards against a converter that names
#     experts differently: an expert tensor is the STACKED one, ``n_dims >= 3``
#     with the header's expert_count in its last dims slot (``dims[-1]``).
#     Verified 2026-07-24 against the real coder-next shards (expert_count=512):
#     all 144 ``_exps`` tensors are 3-D with dims[-1]==512, and name & shape
#     select the IDENTICAL set. The ``nd>=3`` guard matters — 168 *2-D*
#     non-expert tensors (router/shexp/attn_k/v) also carry 512 in dims[-1], so
#     only the stacked-ness tells a real expert weight from a coincidental dim.
#     is_expert = name-match OR shape-match; metadata (expert_count) is the gate;
#     when the two methods disagree that is drift worth a log line, never silent.
# Tensor bytes come from the header's tensor-info table (name + data offset),
# sized by offset-difference within the data section — exact (padding included)
# without a GGML type-size table. Parsed ONCE per file (cache keyed by
# path+size+mtime, mirroring how block_count is only read at plan time — never
# re-parsed per beat).
#
# THE PRINCIPLE (why this exists as a decision input, not a nicety): autofit's
# defect was reducing a TYPED tensor list to an opaque byte-bag at one decision
# step — asking "how many whole layers fit" instead of "what KIND of bytes are
# these". The GGUF header already says which bytes are cold expert weights and
# which are always-hot attention/shared/KV; the fix is the decision function
# CONSUMING metadata it already has. Any future placement decision should start
# from this typed view, never re-flatten it to a single size.
_MOE_EXPERT_TENSOR_RE = None                     # compiled lazily (re import below)
_MOE_DETAIL_CACHE: dict = {}                     # abspath -> {"sig": (sz, mt), "detail": {...}}
# n_cpu_moe value meaning "ALL expert layers on CPU" (llama-server caps it to
# the model's layer count, so any large sentinel works; 999 matches the ae
# deployment's proven LLAMA_ARG_N_CPU_MOE=999).
MOE_ALL_LAYERS = 999


def _expert_tensor_re():
    """The is_expert bit: a name segment ending in ``_exps`` (with the layer
    index captured for per-layer attribution). ``ffn_gate_inp`` (router) and
    ``ffn_*_shexp`` (shared experts) never match — GPU-resident by design."""
    global _MOE_EXPERT_TENSOR_RE
    if _MOE_EXPERT_TENSOR_RE is None:
        import re
        _MOE_EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\..*_exps(\.|$)")
    return _MOE_EXPERT_TENSOR_RE


def _layer_index(name: str) -> Optional[int]:
    """The ``<i>`` of a ``blk.<i>.…`` tensor name, for per-layer attribution, or
    None. Used to attribute a SHAPE-matched expert tensor to its block even when
    the name doesn't carry the ``_exps`` suffix (a nonstandard converter)."""
    import re
    m = re.match(r"^blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _gguf_scan_moe(model_path: str, expert_count_hint: Optional[int] = None) -> dict:
    """One-file GGUF header scan for the MoE split: KV expert counts + the
    expert/non-expert tensor byte split, computed BOTH ways —

      * NAME match: the ``_exps`` suffix (the fast path / primary bit), and
      * SHAPE match: a STACKED tensor (``n_dims >= 3``) carrying the header's
        ``expert_count`` in its last dims slot (``dims[-1]``) — name-independent.

    The dims rule is empirical, verified 2026-07-24 against the real coder-next
    Q4_K_M shards (expert_count=512): every one of the 144 ``ffn_*_exps`` tensors
    is 3-D with ``dims[-1] == 512``, and the two methods select the IDENTICAL set.
    The ``n_dims >= 3`` guard is load-bearing: on those shards 168 *2-D* tensors
    (router ``ffn_gate_inp``, shared experts ``ffn_*_shexp``, ``attn_k/v``) also
    happen to hold 512 in their last slot — only the STACKED expert weights are
    3-D, so the stacked-ness is what distinguishes a real expert tensor from a
    coincidental dimension. (GGUF stores dims reversed vs the logical shape, so
    the stacked-expert axis lands in the last stored slot, ``dims[-1]``.)

    Returns per-shard byte splits for BOTH methods plus the hit counts the
    caller (gguf_moe_detail) needs for the cross-method consistency check.
    {} on any parse issue (dense path)."""
    out: dict = {}
    try:
        import struct

        with open(model_path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return out
            version = struct.unpack("<I", fh.read(4))[0]
            if version < 2:
                return out
            n_tensors = struct.unpack("<Q", fh.read(8))[0]
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n).decode("utf-8", "replace")

            def read_val(t: int):
                simple = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i",
                          5: "<I", 6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
                if t in simple:
                    fmt = simple[t]
                    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
                if t == 8:
                    return read_str()
                if t == 9:
                    et = struct.unpack("<I", fh.read(4))[0]
                    n = struct.unpack("<Q", fh.read(8))[0]
                    return [read_val(et) for _ in range(n)]
                raise ValueError(f"unknown gguf type {t}")

            alignment = 32
            # Split GGUFs carry the full KV metadata only in shard 1; later
            # shards have no expert_count of their own, so the hint (the count
            # discovered from shard 1) lets the SHAPE backstop still fire on
            # them. A shard's OWN header always wins if it has one.
            expert_count = expert_count_hint
            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack("<I", fh.read(4))[0]
                val = read_val(vtype)
                if key.endswith(".expert_count"):
                    out["expert_count"] = val
                    try:
                        expert_count = int(val)
                    except (TypeError, ValueError):
                        expert_count = None
                elif key.endswith(".expert_used_count"):
                    out["expert_used_count"] = val
                elif key == "general.alignment":
                    try:
                        alignment = int(val) or 32
                    except (TypeError, ValueError):
                        pass

            infos = []                                   # (name, offset, dims)
            for _ in range(n_tensors):
                name = read_str()
                nd = struct.unpack("<I", fh.read(4))[0]
                dims = struct.unpack(f"<{nd}Q", fh.read(8 * nd)) if nd else ()
                fh.read(4)                               # ggml type
                off = struct.unpack("<Q", fh.read(8))[0]
                infos.append((name, off, dims))
            header_end = fh.tell()

        data_start = (header_end + alignment - 1) // alignment * alignment
        data_bytes = os.path.getsize(model_path) - data_start
        if data_bytes < 0 or not infos:
            # A header with no tensor table (or a truncated file) can't be split.
            out["expert_bytes"] = 0
            out["non_expert_bytes"] = 0
            out["expert_bytes_shape"] = 0
            out["non_expert_bytes_shape"] = 0
            out["name_expert_hits"] = 0
            out["shape_expert_hits"] = 0
            return out
        infos.sort(key=lambda t: t[1])
        rx = _expert_tensor_re()
        # NAME method (primary): the _exps suffix.
        exp = nexp = 0
        by_layer: dict = {}
        # SHAPE method (backstop): stacked (nd>=3) tensor with expert_count in the
        # last dims slot. Off when the header carries no expert_count.
        exp_s = nexp_s = 0
        by_layer_s: dict = {}
        name_hits = shape_hits = 0
        for i, (name, off, dims) in enumerate(infos):
            end = infos[i + 1][1] if i + 1 < len(infos) else data_bytes
            size = max(0, end - off)
            m = rx.match(name)
            if m:
                name_hits += 1
                exp += size
                layer = int(m.group(1))
                by_layer[layer] = by_layer.get(layer, 0) + size
            else:
                nexp += size
            is_shape_expert = bool(
                expert_count and expert_count > 0
                and len(dims) >= 3 and dims[-1] == expert_count)
            if is_shape_expert:
                shape_hits += 1
                exp_s += size
                sl = _layer_index(name)
                if sl is not None:
                    by_layer_s[sl] = by_layer_s.get(sl, 0) + size
            else:
                nexp_s += size
        out["expert_bytes"] = int(exp)
        out["non_expert_bytes"] = int(nexp)
        out["expert_bytes_by_layer"] = by_layer
        out["expert_bytes_shape"] = int(exp_s)
        out["non_expert_bytes_shape"] = int(nexp_s)
        out["expert_bytes_by_layer_shape"] = by_layer_s
        out["name_expert_hits"] = int(name_hits)
        out["shape_expert_hits"] = int(shape_hits)
    except Exception:  # noqa: BLE001 — unreadable header == dense path, never raise
        return {}
    return out


def _gguf_shard_paths(model_path: str) -> list:
    """All shard files of a split GGUF (``…-00001-of-0000N.gguf``), or just
    ``[model_path]`` for a single-file model. Mirrors the slot supervisor's
    shard-summing (_total_gguf_bytes) so the MoE split is shard-aware too."""
    try:
        import glob
        import re
        base = os.path.basename(model_path)
        m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base)
        if m:
            patt = f"{base[:m.start()]}-*-of-{m.group(1)}.gguf"
            shards = sorted(
                s for s in glob.glob(os.path.join(os.path.dirname(model_path), patt))
                if os.path.isfile(s))
            if shards:
                return shards
    except Exception:  # noqa: BLE001
        pass
    return [model_path]


def gguf_moe_detail(model_path) -> dict:
    """THE MoE reader: ``{is_moe, expert_count, expert_used_count, expert_bytes,
    non_expert_bytes, files}`` for a GGUF (shard-aware: byte splits summed across
    all shards of a split model). ``{"is_moe": False}`` for a dense model, a
    non-GGUF path, or ANY read failure — missing metadata degrades to the dense
    path, never raises. Cached per path by (size, mtime): the header is parsed
    once per file version, never per beat/request.

    DOCTRINE — the name is convention; the shape is ground truth; metadata is the
    gate. Which tensors are experts is decided by NAME (the ``_exps`` suffix, the
    fast primary bit) OR by SHAPE (a stacked ``nd>=3`` tensor carrying the
    header's ``expert_count`` in its last dims slot — the name-independent
    backstop, so a converter that renames experts can't silently zero the split).
    The header's ``expert_count`` KV is the GATE: no positive count, no MoE, no
    matter how the tensors are named — names alone never activate the split. When
    the two methods DISAGREE that is real drift in the file, and drift is worth a
    log line, never a silent misprice: a nonstandard-naming file logs a WARNING
    and is served by shape; a file whose header claims MoE but shows no expert
    tensors at all (by either method) logs a WARNING and falls back to the dense
    split (a safe plain layer split, never a mispriced one); a file whose names
    look like experts but whose header has no count is treated as dense with a
    WARNING (no false MoE). The (path,size,mtime) cache makes every such warning
    fire once per file version."""
    try:
        path = os.path.abspath(str(model_path))
        st = os.stat(path)
        sig = (int(st.st_size), int(st.st_mtime))
    except (TypeError, ValueError, OSError):
        return {"is_moe": False}
    cached = _MOE_DETAIL_CACHE.get(path)
    if cached is not None and cached.get("sig") == sig:
        return cached["detail"]
    expert = nexpert = 0
    expert_s = nexpert_s = 0
    name_hits = shape_hits = 0
    expert_count = expert_used = None
    by_layer: dict = {}
    by_layer_s: dict = {}
    shards = _gguf_shard_paths(path)
    for shard in shards:
        # Thread the expert_count discovered so far (shard 1 carries it; later
        # shards don't) so the SHAPE backstop can fire on every shard.
        scan = _gguf_scan_moe(shard, expert_count_hint=expert_count)
        if not scan:
            continue
        expert += int(scan.get("expert_bytes") or 0)
        nexpert += int(scan.get("non_expert_bytes") or 0)
        expert_s += int(scan.get("expert_bytes_shape") or 0)
        nexpert_s += int(scan.get("non_expert_bytes_shape") or 0)
        name_hits += int(scan.get("name_expert_hits") or 0)
        shape_hits += int(scan.get("shape_expert_hits") or 0)
        for layer, b in (scan.get("expert_bytes_by_layer") or {}).items():
            by_layer[layer] = by_layer.get(layer, 0) + int(b)
        for layer, b in (scan.get("expert_bytes_by_layer_shape") or {}).items():
            by_layer_s[layer] = by_layer_s.get(layer, 0) + int(b)
        if expert_count is None and scan.get("expert_count") is not None:
            try:
                expert_count = int(scan["expert_count"])
            except (TypeError, ValueError):
                pass
        if expert_used is None and scan.get("expert_used_count") is not None:
            try:
                expert_used = int(scan["expert_used_count"])
            except (TypeError, ValueError):
                pass

    # ── Cross-method consistency: name (primary) vs shape (backstop) ──────────
    # The header's positive expert_count is the GATE for MoE. Given the gate,
    # reconcile the two expert-selection methods and log any disagreement ONCE
    # (the cache below makes this per-(path,size,mtime)).
    has_count = bool(expert_count and expert_count > 0)
    if not has_count and name_hits > 0:
        # REVERSE inconsistency: tensors named like experts but no metadata gate.
        # Names alone never activate the split — treat as dense, no false MoE.
        logger.warning(
            "gguf MoE: %s has %d _exps-named tensor(s) but no positive "
            "expert_count in the header — metadata is the gate, treating as "
            "dense (no split).", path, name_hits)
        # Fold the name-matched bytes back into non-expert so no downstream
        # consumer that reads expert_bytes directly can misprice a dense file.
        nexpert += expert
        expert = 0
        by_layer = {}
    elif has_count and name_hits == 0:
        if shape_hits > 0:
            # Nonstandard converter: experts present by SHAPE, not by name.
            # Use the shape results so the split is still priced correctly.
            logger.warning(
                "gguf MoE: %s — expert tensors present by shape (%d stacked "
                "tensors with dims[-1]==expert_count=%d) but not by _exps "
                "naming — nonstandard converter? Using shape-derived split.",
                path, shape_hits, expert_count)
            expert, nexpert = expert_s, nexpert_s
            by_layer = by_layer_s
        else:
            # Header claims MoE but NEITHER method finds expert tensors — the
            # safe fallback is the dense split (plain layer split), never a
            # mispriced one. expert stays 0 -> is_moe False below.
            logger.warning(
                "gguf MoE: %s header claims MoE (expert_count=%d) but no expert "
                "tensors identifiable by name or shape — treating as dense.",
                path, expert_count)

    # expert_count == 0 or absent IS the definition of dense (operator
    # grounding); expert bytes must also exist for a split to mean anything.
    is_moe = bool(expert_count and expert_count > 0 and expert > 0)
    # Sparsity (expert_used_count / expert_count): the fraction of expert bytes
    # a token actually touches — it predicts the per-token CPU traffic of a
    # split (coder-next: 43.59 GiB x 10/512 ~= 0.85 GiB/token, which against
    # RAM bandwidth reproduces the measured ~24 tok/s). Carried for the
    # placement evaluator; None when either count is unreadable.
    sparsity = None
    if expert_count and expert_used:
        try:
            sparsity = float(expert_used) / float(expert_count)
        except (TypeError, ValueError, ZeroDivisionError):
            sparsity = None
    detail = {"is_moe": is_moe, "expert_count": expert_count,
              "expert_used_count": expert_used, "sparsity": sparsity,
              "expert_bytes": int(expert), "non_expert_bytes": int(nexpert),
              "expert_bytes_by_layer": by_layer,
              "files": len(shards)}
    _MOE_DETAIL_CACHE[path] = {"sig": sig, "detail": detail}
    return detail


def moe_split_need(detail: dict, n_cpu_moe: Optional[int] = None) -> "Optional[dict]":
    """Per-layer-aware pricing of a MoE split: what a load with ``--n-cpu-moe N``
    puts where. ``{"cpu_bytes", "gpu_bytes", "layers_on_cpu"}`` or None for a
    dense/unreadable detail (caller keeps opaque-size pricing).

    llama-server moves the expert tensors of the FIRST N block indices to CPU
    (everything else — attention, router, shared experts, embeddings, output,
    and the expert tensors of layers >= N — stays GPU-side). ``n_cpu_moe`` of
    None or >= the attributed layer count means ALL experts on CPU (the 999
    sentinel). The per-layer map keeps a future partial split precisely
    priceable instead of re-flattening the typed tensor list to one number."""
    if not isinstance(detail, dict) or not detail.get("is_moe"):
        return None
    expert = int(detail.get("expert_bytes") or 0)
    nexpert = int(detail.get("non_expert_bytes") or 0)
    by_layer = detail.get("expert_bytes_by_layer") or {}
    layers = sorted(by_layer)
    if n_cpu_moe is None or not layers or int(n_cpu_moe) >= len(layers):
        return {"cpu_bytes": expert, "gpu_bytes": nexpert,
                "layers_on_cpu": len(layers)}
    n = max(0, int(n_cpu_moe))
    cpu = sum(int(by_layer[i]) for i in layers[:n])
    return {"cpu_bytes": int(cpu), "gpu_bytes": int(nexpert + (expert - cpu)),
            "layers_on_cpu": n}


def n_cpu_moe_env() -> Optional[int]:
    """Explicit per-request/per-model n_cpu_moe from HUGPY_N_CPU_MOE (the spill
    wire, set by the worker's _apply_spill), or None when unset. The number of
    MoE layers whose EXPERT tensors stay on CPU (MOE_ALL_LAYERS/999 = all);
    explicit always wins over the auto policy."""
    return _env_int("HUGPY_N_CPU_MOE")


# ── KV-cache quantification (slice 11 / t27) ────────────────────────────────
# Operator (2026-07-17): "the context can necessarily be quantified into ram
# needed correct? ... this should be a variable as well based on percentage max."
#
# The KV cache is the attention key/value tensors held for every token in the
# context window — the RAM/VRAM tax that fit/admission ignored (weights-only).
# The exact cache size is a function of the model's real geometry and the ctx:
#
#   kv_bytes = 2 (K and V) × n_layers × ctx_tokens × n_kv_heads × head_dim
#              × dtype_bytes
#
# n_kv_heads (NOT n_attention_heads) is what modern GQA/MQA models actually
# cache — Qwen2.5-Coder-3B has 16 attention heads but only 2 KV heads, an 8×
# reduction, so using attention-heads would over-count KV by 8×. head_dim =
# embedding_length / attention_head_count when not stated explicitly.
#
# dtype: llama.cpp caches fp16 by default (2 bytes); a quantized-KV config
# (-ctk/-ctv q8_0 / q4_0) lowers it. transformers caches in the model's compute
# dtype (torch_dtype: bf16/fp16 = 2, fp32 = 4) unless a cache override says else.
_KV_DTYPE_BYTES = {
    "f32": 4.0, "float32": 4.0, "fp32": 4.0,
    "f16": 2.0, "float16": 2.0, "fp16": 2.0, "bf16": 2.0, "bfloat16": 2.0,
    "q8_0": 1.0, "q8": 1.0, "int8": 1.0,
    "q5_0": 0.65, "q5_1": 0.69,
    "q4_0": 0.5, "q4_1": 0.56, "q4": 0.5, "int4": 0.5,
}
# When geometry is unavailable we NEVER silently return zero (that reintroduces
# the unplanned tax). A stated conservative heuristic: bytes per token per layer
# for a typical mid-size GQA model, cross-checked against the exact formula for
# Qwen2.5-Coder-3B (36L × 2kv × 128hd × 2B × 2 = ~256 KiB/token total ≈
# 7.3 KiB/token/layer → round UP to be conservative). Used only as a floor when
# real geometry can't be read; a WARN says so at the call site.
_KV_HEURISTIC_BYTES_PER_TOKEN_PER_LAYER = 8 * 1024  # 8 KiB, deliberately generous


def _kv_dtype_bytes(name: Optional[str], default: float = 2.0) -> float:
    if not name:
        return default
    return _KV_DTYPE_BYTES.get(str(name).strip().lower(), default)


def _gguf_kv_geometry(model_path: str) -> dict:
    """Layers / kv-heads / head-dim / trained ctx from a GGUF header, or {}.
    head_dim falls back to embedding_length / attention.head_count (llama.cpp's
    own derivation) when a *.attention.key_length is absent."""
    md = _gguf_metadata(model_path, (
        ".block_count", ".attention.head_count", ".attention.head_count_kv",
        ".embedding_length", ".attention.key_length", ".context_length"))
    if not md:
        return {}
    n_layers = md.get(".block_count")
    n_heads = md.get(".attention.head_count")
    n_kv = md.get(".attention.head_count_kv") or n_heads      # MHA: kv == heads
    emb = md.get(".embedding_length")
    head_dim = md.get(".attention.key_length")
    if not head_dim and emb and n_heads:
        try:
            head_dim = int(emb) // int(n_heads)
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    out = {}
    for k, v in (("n_layers", n_layers), ("n_kv_heads", n_kv),
                 ("head_dim", head_dim), ("ctx_train", md.get(".context_length"))):
        try:
            if v is not None:
                out[k] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def _transformers_kv_geometry(config: dict) -> dict:
    """Layers / kv-heads / head-dim / dtype from a transformers config.json dict.
    Mirrors HF conventions: num_key_value_heads defaults to num_attention_heads
    (MHA) when absent; head_dim defaults to hidden_size / num_attention_heads."""
    if not isinstance(config, dict):
        return {}
    n_layers = config.get("num_hidden_layers")
    n_heads = config.get("num_attention_heads")
    n_kv = config.get("num_key_value_heads") or n_heads      # MHA fallback
    head_dim = config.get("head_dim")
    if not head_dim and config.get("hidden_size") and n_heads:
        try:
            head_dim = int(config["hidden_size"]) // int(n_heads)
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    out: dict = {"dtype": config.get("torch_dtype")}
    for k, v in (("n_layers", n_layers), ("n_kv_heads", n_kv),
                 ("head_dim", head_dim),
                 ("ctx_train", config.get("max_position_embeddings"))):
        try:
            if v is not None:
                out[k] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def kv_bytes(*, ctx_tokens: int, n_layers: Optional[int] = None,
             n_kv_heads: Optional[int] = None, head_dim: Optional[int] = None,
             dtype_bytes: float = 2.0) -> Optional[int]:
    """KV-cache bytes for ``ctx_tokens`` given the model geometry, or a stated
    conservative HEURISTIC when geometry is missing (never silently zero).

    kv = 2 × n_layers × ctx × n_kv_heads × head_dim × dtype_bytes. Returns None
    only when ctx is non-positive (no cache). When n_layers is known but the
    per-head geometry is not, falls back to the bytes-per-token-per-layer
    heuristic (× n_layers × ctx); when even n_layers is unknown, uses an assumed
    layer count so the caller still gets a non-zero, conservative reserve."""
    try:
        ctx = int(ctx_tokens)
    except (TypeError, ValueError):
        return None
    if ctx <= 0:
        return None
    if n_layers and n_kv_heads and head_dim:
        return int(2 * int(n_layers) * ctx * int(n_kv_heads) * int(head_dim)
                   * float(dtype_bytes))
    # Geometry incomplete — conservative heuristic, never zero.
    layers = int(n_layers) if n_layers else _ASSUMED_LAYERS
    return int(layers * ctx * _KV_HEURISTIC_BYTES_PER_TOKEN_PER_LAYER)


# ---------------------------------------------------------------------------
# llama.cpp (GGUF)
# ---------------------------------------------------------------------------
def vision_projector_bytes(model_path: str) -> int:
    """Bytes of the mmproj / CLIP projector sidecar beside a vision GGUF (0 if none).

    A vision GGUF is a PAIR: the language-model quant + a separate ``mmproj-*.gguf``
    (the image encoder / projector). llama.cpp loads the projector ONTO THE GPU
    alongside the offloaded layers, so its VRAM must be reserved BEFORE we fit
    language-model layers — otherwise an 8 GB card computes "offload all layers"
    against the model file alone, then OOMs when the ~1.3 GB projector lands on
    top (Qwen2.5-VL-7B ships a 1.35 GB mmproj). Text models have no projector, so
    this returns 0 and the fit math is byte-identical to before.

    Self-contained (no import of the imports package) so the fit math stays
    offline-testable and can never be broken by a heavy import chain. Best-effort:
    any filesystem error yields 0 (fail open — never inflate the reserve)."""
    _hints = ("mmproj", "mm-proj", "mm_proj", "projector")
    try:
        directory = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        if not directory or not os.path.isdir(directory):
            return 0
        main_abs = os.path.abspath(model_path) if os.path.isfile(model_path) else None
        for fn in os.listdir(directory):
            low = fn.lower()
            if not low.endswith(".gguf"):
                continue
            if not any(h in low for h in _hints):
                continue
            cand = os.path.join(directory, fn)
            if main_abs is not None and os.path.abspath(cand) == main_abs:
                continue                        # never count the main file itself
            try:
                return int(os.path.getsize(cand))
            except OSError:
                return 0
    except OSError:
        return 0
    return 0


def autofit_gpu_layers(model_path: str,
                       free_vram: Optional[int] = None,
                       extra_reserve_bytes: int = 0) -> int:
    """How many GGUF layers fit in free VRAM. -1 (all) when the whole file fits.

    ``extra_reserve_bytes`` is VRAM that must be held OUT of the layer budget
    because something else lands on the GPU next to the offloaded layers — for a
    vision GGUF this is the mmproj/CLIP projector (see ``vision_projector_bytes``).
    It is subtracted from the budget alongside the KV/context allowance, so a
    partial split leaves honest room for the projector and we only return -1 (all
    layers) when the whole model AND the projector AND the context all fit. 0
    (the default) reproduces the historical text-model behaviour exactly."""
    if free_vram is None:
        free_vram = free_vram_bytes()
        # Operator VRAM budget (the console's "VRAM budget…" mode / spill
        # gpu_mem_gib) caps the autofit. GGUF loads ignored this before — the
        # knob only reached the transformers path, so a per-model budget on a
        # GGUF worker silently did nothing.
        gpu_gib = _env_float("HUGPY_GPU_MEM_GIB")
        if gpu_gib is not None:
            cap = int(gpu_gib * 2**30)
            free_vram = min(free_vram, cap) if free_vram else cap
    if not free_vram:
        # Fail OPEN, not closed. free_vram is unknown here, but if the box HAS a
        # GPU (detect_gpus finds a card even when the free-VRAM probe on the
        # primary index came back None — e.g. the slot supervisor whose
        # torch/nvidia-smi view differs from the agent's) an offload-capable
        # llama.cpp must put every layer on the GPU, not drop the model to CPU.
        # Only a genuinely GPU-less host stays on CPU (0). Gate on detect_gpus()
        # (hardware truth) rather than importing llama_cpp here — a CUDA
        # llama_cpp import would poison a later torch import in the in-process
        # (agent/central) caller of autofit.
        try:
            from .._platform.hardware import detect_gpus
            if detect_gpus():
                return -1
        except Exception:
            pass
        return 0

    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return 0
    if file_bytes <= 0:
        return 0

    # Weights are not the whole story: llama_context still needs VRAM AFTER
    # the weights land (KV cache scales with n_ctx, plus compute-graph
    # buffers). The flat safety factor leaves only ~3.5 GB on a 24 GB card —
    # a 70B uploads ~20 GB of weights and then dies with "Failed to create
    # llama_context". Reserve an explicit context allowance under the margin.
    ctx_reserve = int((_env_float("HUGPY_VRAM_CTX_RESERVE_GIB") or 2.5) * 2**30)
    reserve = ctx_reserve + max(0, int(extra_reserve_bytes))
    budget = int(free_vram * _VRAM_SAFETY) - reserve
    if budget <= 0:
        return 0
    if budget >= file_bytes:
        return -1                           # everything fits on the GPU
                                            # (model + projector + context)

    total_layers = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    # Weights dominate the file; approximate per-layer cost as an even split.
    per_layer = file_bytes / max(total_layers, 1)
    fit = int(budget // per_layer)
    fit = max(0, min(fit, total_layers))
    logger.info(
        "autofit gguf: file=%.1fGiB free=%.1fGiB -> %d/%d layers on GPU",
        file_bytes / 2**30, free_vram / 2**30, fit, total_layers,
    )
    return fit


# The worker's honest partial-offload admission (agent._vram_evict_to_fit) pins an
# EXPLICIT n_gpu_layers per SERVED-quant path here, so the in-process llama_cpp
# load offloads exactly the admitted layer count. Without it the in-process path
# would fall to autofit_gpu_layers, which sizes off the on-disk file (shard-1
# only) and UNDER-counts a sharded model -> over-offloads -> the very OOM this
# fixes. Path-keyed (not model_key) because gguf_gpu_layers only sees a path;
# absent/None -> historical env+autofit behaviour, byte-identical.
_NGL_OVERRIDE: dict[str, int] = {}


def set_ngl_override(model_path, n_gpu_layers) -> None:
    """Pin an explicit n_gpu_layers for a resolved GGUF path (the worker's honest
    partial-offload plan). Consulted by gguf_gpu_layers BEFORE env/autofit."""
    try:
        _NGL_OVERRIDE[os.path.abspath(str(model_path))] = int(n_gpu_layers)
    except (TypeError, ValueError, OSError):
        pass


def clear_ngl_override(model_path) -> None:
    """Drop the partial-offload pin for a path (re-admission / full-fit re-decide)."""
    try:
        _NGL_OVERRIDE.pop(os.path.abspath(str(model_path)), None)
    except (TypeError, ValueError, OSError):
        pass


# ── k37 allocation modes (worker-side wire: HUGPY_ALLOC_MODE etc.) ──────────
# The five-mode selector's NEW spill keys land here as env (set per request by
# agent._apply_spill): alloc_mode -> HUGPY_ALLOC_MODE, leniency_pct ->
# HUGPY_LENIENCY_PCT, priority_device -> HUGPY_PRIORITY_DEVICE. Only max-ram /
# explicit ever ride this wire — gpu-only/ram-only/max-gpu keep the unchanged
# legacy n_gpu_layers encoding. Central version-gates emission so an old
# worker never receives these; on a mode-aware worker they must never be a
# dead knob, so both engine paths below consult them.
def alloc_mode_env() -> Optional[str]:
    """The request's allocation mode from HUGPY_ALLOC_MODE ("max-ram" |
    "explicit"), or None (legacy encoding / no mode — behave exactly as
    before)."""
    raw = _env("HUGPY_ALLOC_MODE")
    if raw is None:
        return None
    low = raw.strip().lower()
    return low or None


def leniency_pct_env() -> Optional[float]:
    """explicit mode's leniency (percent OF THE MODEL that may land off its
    ideal device before bust), 0..100, or None when unset."""
    v = _env_float("HUGPY_LENIENCY_PCT")
    if v is None:
        return None
    return max(0.0, min(100.0, v))


def priority_device_env() -> str:
    """explicit mode's priority device ("gpu" default | "ram")."""
    raw = (_env("HUGPY_PRIORITY_DEVICE") or "gpu").strip().lower()
    return "ram" if raw == "ram" else "gpu"


# RAM safety factor for the max-ram fill (mirror of _VRAM_SAFETY: never budget
# every last byte of MemAvailable for weights).
_RAM_FILL_SAFETY = 0.95


def maxram_gpu_layers(model_path: str, free_ram: Optional[int] = None) -> int:
    """n_gpu_layers for the **max-ram** mode: fill the RAM budget FIRST, and
    only the OVERFLOW layers go to the GPU — autofit's per-layer pricing,
    inverted. 0 when the whole model fits RAM (pure RAM residency; the GPU is
    only touched when RAM genuinely can't hold everything).

    The RAM budget = budgetable free RAM (reserve- and ceiling-aware) capped by
    an explicit HUGPY_CPU_MEM_GIB when set, with a safety factor so the fill
    never rides MemAvailable to the OOM floor. Whether the GPU can actually
    hold the overflow is the ADMISSION engine's question
    (flex.plan_explicit_offload, ram priority) — this is the loader-level
    intent, exactly like autofit_gpu_layers is for max-gpu."""
    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return 0
    if file_bytes <= 0:
        return 0
    if free_ram is None:
        free_ram = free_ram_bytes()
        cpu_gib = _env_float("HUGPY_CPU_MEM_GIB")
        if cpu_gib is not None:
            cap = int(cpu_gib * 2**30)
            free_ram = min(free_ram, cap) if free_ram else cap
    total_layers = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    per_layer = file_bytes / max(total_layers, 1)
    ram_budget = int((free_ram or 0) * _RAM_FILL_SAFETY)
    in_ram = min(total_layers, int(ram_budget // per_layer)) if per_layer > 0 else total_layers
    overflow = max(0, total_layers - in_ram)
    logger.info(
        "max-ram gguf: file=%.1fGiB ram_budget=%.1fGiB -> %d/%d layers in RAM, "
        "%d overflow to GPU",
        file_bytes / 2**30, ram_budget / 2**30, in_ram, total_layers, overflow)
    return overflow


def gguf_gpu_layers(model_path: str) -> int:
    """Resolve n_gpu_layers for a GGUF model: the worker's partial-offload pin
    first (honest layers-that-fit), else the k37 allocation mode (max-ram's
    inverted fill), else env (+autofit)."""
    try:
        pinned = _NGL_OVERRIDE.get(os.path.abspath(model_path))
    except (TypeError, OSError):
        pinned = None
    if pinned is not None:
        return int(pinned)
    # k37: max-ram inverts the fill (RAM first, overflow to GPU). explicit
    # falls through to autofit — its VRAM target rides HUGPY_GPU_MEM_GIB which
    # already caps autofit; the leniency FLOOR is enforced at admission
    # (flex.plan_explicit_offload), not here.
    if alloc_mode_env() == "max-ram":
        return maxram_gpu_layers(model_path)
    raw = _env("HUGPY_N_GPU_LAYERS")
    if raw is None or raw.lower() == "auto":
        return autofit_gpu_layers(model_path)
    if raw.lower() in ("off", "cpu", "none"):
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("bad HUGPY_N_GPU_LAYERS=%r; using autofit", raw)
        return autofit_gpu_layers(model_path)


def tensor_split() -> Optional[list[float]]:
    raw = _env("HUGPY_TENSOR_SPLIT")
    if not raw:
        return None
    try:
        parts = [float(x) for x in raw.split(",") if x.strip()]
        return parts or None
    except ValueError:
        logger.warning("bad HUGPY_TENSOR_SPLIT=%r; ignoring", raw)
        return None


def main_gpu() -> Optional[int]:
    return _env_int("HUGPY_MAIN_GPU")


def rpc_servers() -> Optional[str]:
    """Comma-separated "host:port" of llama.cpp rpc-servers to shard onto, or None.

    Set (as a per-request spill override) by central's allocator when it decides
    to shard a model across multiple GPUs on different machines.
    """
    raw = _env("HUGPY_RPC_SERVERS")
    return raw.strip() if raw and raw.strip() else None


def _binding_supports_rpc() -> bool:
    """Whether this llama-cpp-python build accepts ``Llama(rpc_servers=…)``.

    The param existed in 0.2.78–0.2.90 and was dropped in the 0.3.x rewrite —
    passing it there raises TypeError. Shard leads on a >=0.3 binding must be
    served via ``llama-server --rpc …`` (the serve layer's extra_args) instead
    of the in-process runner.
    """
    try:
        import inspect
        from llama_cpp import Llama
        return "rpc_servers" in inspect.signature(Llama.__init__).parameters
    except Exception:
        return False


def llama_kwargs(model_path: str) -> dict[str, Any]:
    """Spill kwargs for ``Llama(...)``. Always includes n_gpu_layers.

    When ``HUGPY_RPC_SERVERS`` is set we're the LEAD of a cross-machine shard:
    pass ``rpc_servers`` and force ``n_gpu_layers=-1`` (offload ALL layers across
    the pooled GPUs — the whole point is to never touch CPU). ``tensor_split``
    (also supplied by the allocator) then weights layers across [local, *rpc].
    """
    rpc = rpc_servers()
    if rpc and not _binding_supports_rpc():
        logger.warning(
            "HUGPY_RPC_SERVERS=%s set but this llama-cpp-python has no "
            "rpc_servers param (dropped in 0.3.x) — ignoring the shard plan "
            "and loading locally. Serve shard leads via llama-server --rpc "
            "(cfg.extra['llama_extra_args']) instead.", rpc,
        )
        rpc = None
    if rpc:
        kwargs: dict[str, Any] = {"n_gpu_layers": -1, "rpc_servers": rpc}
    else:
        kwargs = {"n_gpu_layers": gguf_gpu_layers(model_path)}
    ts = tensor_split()
    if ts is not None:
        kwargs["tensor_split"] = ts
    mg = main_gpu()
    if mg is not None:
        kwargs["main_gpu"] = mg
    return kwargs


# ---------------------------------------------------------------------------
# placement intent — one wire field (n_gpu_layers), interpreted PER ENGINE
# ---------------------------------------------------------------------------
# The console's Autofit / Max GPU / CPU only controls are ENGINE-AGNOSTIC
# PLACEMENT INTENT (operator ruling 2026-07-17: "the autofit, maxgpu and cpu only
# should still be on the table as its easy to infer what those would indicate for
# a transformer"). They ride the SAME wire field the GGUF path uses —
# HUGPY_N_GPU_LAYERS (-1 / "off" / "auto") — but that field NAMES llama.cpp layer
# counts. For a GGUF model it IS a layer count (gguf_gpu_layers). For a
# transformers model there are no "gpu layers" to count, so the field is read as
# PLACEMENT INTENT and mapped onto the transformers device_map/max_memory world:
#   * -1  ("Max GPU")  -> put the WHOLE model on the GPU (no CPU budget)
#   * 0/"off" ("CPU only") -> keep it entirely on CPU (gpu budget 0)
#   * unset/"auto"     -> today's autofit (shard to fit VRAM, spill to CPU)
# Same wire, per-engine interpretation — placement intent, NOT layer counts.
def n_gpu_layers_intent() -> str:
    """Decode HUGPY_N_GPU_LAYERS into an engine-agnostic PLACEMENT class:
    ``"gpu"`` (all-on-GPU, from -1), ``"cpu"`` (CPU-only, from 0/"off"/"cpu"/
    "none"), or ``"auto"`` (fit-and-spill, from unset/"auto"/a positive int).

    A positive int is a llama.cpp partial-offload count with no transformers
    analogue, so for the transformers path it reads as ``"auto"`` (fit as much as
    fits) — the honest 'some on GPU' behavior — rather than inventing a split."""
    raw = _env("HUGPY_N_GPU_LAYERS")
    if raw is None:
        return "auto"
    low = raw.strip().lower()
    if low in ("auto", ""):
        return "auto"
    if low in ("off", "cpu", "none"):
        return "cpu"
    if low == "-1":
        return "gpu"
    try:
        return "cpu" if int(low) == 0 else "auto"
    except ValueError:
        return "auto"


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------
def _gib(n: float) -> str:
    return f"{n:.2f}GiB"


def transformers_max_memory(model_need_bytes: Optional[int] = None) -> Optional[dict]:
    """Build a ``max_memory`` map for device_map='auto', or None to skip.

    Explicit env budgets win; otherwise autofit from detected free VRAM/RAM.
    Returns None when no GPU is visible (let transformers stay on CPU).

    k37 **max-ram** (HUGPY_ALLOC_MODE=max-ram): RAM-priority placement —
    generous CPU budget + only the REMAINDER on the GPU. accelerate fills GPUs
    first, so RAM priority is expressed by capping the GPU budget at what the
    CPU budget cannot hold: gpu = max(0, need − cpu). ``model_need_bytes``
    (optional, from a loader that knows its size — Slice C wires the gap
    loaders) makes that remainder honest; without it the GPU budget is 0 (pure
    RAM — safe, never a silent GPU fill against an explicit RAM priority).
    NOTE central engine-gates max-ram/explicit to GGUF models today, so this
    branch is defense-in-depth + the Slice C seam, not a live central path.

    PLACEMENT INTENT (t26): HUGPY_N_GPU_LAYERS is honored here as engine-agnostic
    placement, NOT a layer count (see n_gpu_layers_intent):
      * "cpu"  (n_gpu_layers 0/"off") -> gpu budget 0 GiB so device_map='auto'
        places the whole model on CPU. Returned as a map (not None) so the
        intent BINDS even when a GPU is present — the operator asked for CPU.
      * "gpu"  (n_gpu_layers -1)      -> no CPU budget: the model is forced onto
        the card (accelerate raises honestly if it truly can't fit, rather than
        us silently spilling against an explicit 'all on GPU').
      * "auto"                        -> today's fit-and-spill behavior, unchanged.
    An EXPLICIT gpu_mem_gib/cpu_mem_gib budget still wins over the intent-derived
    default for that axis (the GGUF-only explicit class; when present here it is
    simply honored)."""
    intent = n_gpu_layers_intent()
    gpu_gib = _env_float("HUGPY_GPU_MEM_GIB")
    cpu_gib = _env_float("HUGPY_CPU_MEM_GIB")
    n_gpu = _env_int("HUGPY_N_GPU") or 1

    # k37 max-ram: RAM-priority — generous CPU, remainder (if known) on GPU.
    if alloc_mode_env() == "max-ram":
        if cpu_gib is None:
            fr = free_ram_bytes()
            cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0
        if gpu_gib is None:
            if model_need_bytes:
                gpu_gib = max(0.0, (int(model_need_bytes) / 2**30) - cpu_gib)
            else:
                gpu_gib = 0.0          # unknown need: never silently fill the GPU
        mm_ram: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
        mm_ram["cpu"] = _gib(cpu_gib)
        logger.info("transformers placement=max-ram max_memory=%s", mm_ram)
        return mm_ram

    # CPU-only intent: force everything off the GPU. Bind even with a GPU present
    # (an explicit CPU-only placement, not autofit) — gpu budget 0, generous CPU.
    if intent == "cpu":
        if cpu_gib is None:
            fr = free_ram_bytes()
            cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0
        mm_cpu: dict[Any, str] = {i: _gib(0.0) for i in range(max(n_gpu, 1))}
        mm_cpu["cpu"] = _gib(cpu_gib)
        logger.info("transformers placement=cpu-only max_memory=%s", mm_cpu)
        return mm_cpu

    if gpu_gib is None:
        fv = free_vram_bytes()
        if not fv:
            return None                     # no GPU -> no spill map
        gpu_gib = (fv * _VRAM_SAFETY) / 2**30

    # All-on-GPU intent: no CPU budget so accelerate keeps the whole model on the
    # card. Skip the CPU-spill fallback below (a bare gpu-only max_memory). An
    # explicit cpu_mem_gib still wins if the operator set one alongside.
    if intent == "gpu":
        mm_gpu: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
        if cpu_gib is not None:
            mm_gpu["cpu"] = _gib(cpu_gib)
        logger.info("transformers placement=all-gpu max_memory=%s", mm_gpu)
        return mm_gpu

    if cpu_gib is None:
        fr = free_ram_bytes()
        cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0

    mm: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
    mm["cpu"] = _gib(cpu_gib)
    logger.info("transformers placement=auto max_memory=%s", mm)
    return mm


def describe() -> dict[str, Any]:
    """Human-readable snapshot of the current spill config (for heartbeats/UI)."""
    return {
        "mode": (_env("HUGPY_N_GPU_LAYERS") or "auto"),
        "alloc_mode": alloc_mode_env(),          # k37: max-ram|explicit|None
        "leniency_pct": leniency_pct_env(),
        "priority_device": (priority_device_env()
                            if alloc_mode_env() == "explicit" else None),
        "n_gpu_layers_env": _env("HUGPY_N_GPU_LAYERS"),
        "n_cpu_moe": n_cpu_moe_env(),            # MoE expert-split knob (env wire)
        "gpu_mem_gib": _env_float("HUGPY_GPU_MEM_GIB"),
        "cpu_mem_gib": _env_float("HUGPY_CPU_MEM_GIB"),
        "tensor_split": tensor_split(),
        "free_vram_bytes": free_vram_bytes(),
    }
