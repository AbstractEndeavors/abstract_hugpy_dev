"""Persisted, per-model serving overrides — the UI-writable layer.

The registry (MODELS + discovery) gives each model its baseline serving config
in ``cfg.extra``; this overlay lets the console change it per model at runtime
without rebuilding the registry or editing code. Stored as one JSON file keyed
by model_key:

    {"DAN-L3-R1-8B-i1-GGUF": {"serve_mode": "systemd", "n_gpu_layers": -1,
                              "threads": 8, "llama_ctx": 8192}}

:func:`serve_spec_for` merges this over ``cfg.extra`` (override wins), so the
systemd unit, the swap config, and the HTTP runner endpoint all reflect it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

try:
    from ...imports.src.constants.constants import PROJECTS_HOME
except Exception:  # pragma: no cover - fall back if layout differs
    from ..._platform.paths import models_root
    PROJECTS_HOME = os.environ.get("PROJECTS_HOME") or os.path.join(
        os.environ.get("DEFAULT_ROOT") or models_root(), "projects")

_OVERRIDES_PATH = os.environ.get("SERVE_OVERRIDES_PATH") or os.path.join(
    PROJECTS_HOME, "serve_overrides.json")
_LOCK = threading.Lock()

# Fields the console may set per model. Anything else is ignored.
ALLOWED_FIELDS = {
    "serve_mode",     # off | systemd | swap
    "n_gpu_layers",   # GPU offload (-1 all, 0 cpu, N layers)
    "n_cpu_moe",      # MoE expert split: N MoE layers whose EXPERT tensors stay
                      # on CPU (999 = all — spill.MOE_ALL_LAYERS). Rides
                      # llama-server --n-cpu-moe; measured on ae 2026-07-24:
                      # -1 layers + 999 beat the 17/48 layer split by +59% tok/s
                      # at 5x less VRAM on an 80B-A3B MoE. Absent -> auto policy
                      # (MoE hybrid auto-splits; dense/fits-whole unchanged).
    "threads",        # CPU threads
    "llama_ctx",      # context window
    "gpu_mem_gib",    # transformers per-GPU budget / explicit-mode VRAM target
    "cpu_mem_gib",    # transformers CPU/RAM budget / explicit-mode RAM target
    "always_on",      # systemd always-on vs swap on-demand
    "ttl_seconds",    # swap idle-unload TTL
    "gguf_file",      # which downloaded .gguf to serve (basename; "" = auto/default)
    # k37 — the five-mode allocation selector (gpu-only|ram-only|max-gpu|
    # max-ram|explicit; legacy names resolved on write, never stored back).
    "alloc_mode",
    "leniency_pct",   # explicit mode: N% OF THE MODEL may land off its ideal
                      # device before bust (100% GPU + 30% -> floor 70/30)
    "priority",       # explicit mode: flex priority (0 = normal; higher
                      # compresses lower-priority neighbours within bands)
    "priority_device",  # explicit mode: which device the target favors
                        # ("gpu" default | "ram")
}
_INT_FIELDS = {"n_gpu_layers", "n_cpu_moe", "threads", "llama_ctx", "ttl_seconds",
               "priority"}
_FLOAT_FIELDS = {"gpu_mem_gib", "cpu_mem_gib", "leniency_pct"}
_BOOL_FIELDS = {"always_on"}


def _load() -> dict:
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def all_overrides() -> dict:
    return _load()


def get_override(model_key: str) -> dict:
    return _load().get(model_key, {}) or {}


def _gguf_basenames(model_dir: str) -> list:
    """Sorted basenames of the model's downloaded .gguf files, excluding the
    mmproj projector (which is not a servable language model)."""
    try:
        from ...imports.src.utils import is_mmproj_file
    except Exception:
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
    out = []
    try:
        for fn in os.listdir(model_dir):
            if fn.lower().endswith(".gguf") and not is_mmproj_file(fn):
                out.append(fn)
    except OSError:
        return []
    return sorted(out)


def available_gguf_files(model_dir: str) -> list:
    """Public: the .gguf variants the operator may pick for this model."""
    return _gguf_basenames(model_dir)


def resolve_override_gguf(model_key: str, model_dir: str):
    """Absolute path of the operator-selected .gguf for this model, IF the
    ``gguf_file`` override is set and that file exists under ``model_dir``; else
    None (caller falls back to the registry/auto resolution). Honored by both the
    in-process runner and the systemd/swap serve spec, so the choice is global."""
    fn = (get_override(model_key) or {}).get("gguf_file")
    if not fn:
        return None
    cand = os.path.join(model_dir, os.path.basename(str(fn)))
    return cand if os.path.isfile(cand) else None


def _file_bytes(model_dir: str, fn: str) -> int:
    try:
        return int(os.path.getsize(os.path.join(model_dir, os.path.basename(str(fn)))))
    except OSError:
        return 0


def _mmproj_bytes(model_dir: str) -> int:
    """Total bytes of the mmproj projector(s) beside the model. A vision GGUF is
    a PAIR (quant + mmproj-*.gguf), so the projector is part of what actually
    loads — it belongs in the model's effective size even though it isn't a
    servable variant on its own. Recursive: a projector can be nested in a
    subdir just like split shards are (see ``_servable_gguf_files``)."""
    try:
        from ...imports.src.utils import is_mmproj_file
    except Exception:  # noqa: BLE001
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
    total = 0
    try:
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                if fn.lower().endswith(".gguf") and is_mmproj_file(fn):
                    try:
                        total += int(os.path.getsize(os.path.join(root, fn)))
                    except OSError:
                        pass
    except OSError:
        pass
    return total


# A split/sharded GGUF ships as N files ``<stem>-<NNNNN>-of-<MMMMM>.gguf``; they
# are ONE logical model that llama.cpp loads from the first shard.
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


def _servable_gguf_files(model_dir: str) -> list:
    """Recursively list servable .gguf files (relative paths + sizes), excluding
    the mmproj projector.

    RECURSIVE, unlike :func:`_gguf_basenames`: a split/sharded GGUF nests its
    shards in a subdir (e.g. ``<quant>/<quant>-00001-of-00004.gguf``), and a
    shallow ``os.listdir`` misses them entirely — the model then resolved to NO
    servable variant and no ``effective_bytes`` at all (the sharded-GGUF
    effective-size blind spot, t33). Mirrors ``get_gguf_file``'s recursive glob
    so the two agree on what is servable.

    Returns ``[(relpath, bytes), …]`` sorted by relpath.
    """
    try:
        from ...imports.src.utils import is_mmproj_file
    except Exception:  # noqa: BLE001
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
    out = []
    try:
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                if not fn.lower().endswith(".gguf") or is_mmproj_file(fn):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, model_dir)
                try:
                    sz = int(os.path.getsize(full))
                except OSError:
                    sz = 0
                out.append((rel, sz))
    except OSError:
        return []
    return sorted(out)


def _gguf_variant_groups(files: list) -> list:
    """Collapse a recursive servable-gguf listing into pickable VARIANTS,
    shard-aware.

    A split GGUF presents as N shard files (``-00001-of-0000N.gguf`` …) that are
    ONE logical model; they fold into a single variant whose ``bytes`` = SUM of
    all shards and whose entrypoint/canonical filename is the ``-00001`` shard
    (what ``get_gguf_file``/llama-server is actually pointed at). Every non-shard
    .gguf is its own variant. Multiple quant families that are each sharded stay
    separate (grouped by containing dir + shard stem), so they rank as distinct
    variants exactly like single-file quants do.

    ``files``: ``[(relpath, bytes), …]``. Returns ``[{filename, bytes, members}]``
    where ``filename`` is the entrypoint basename and ``members`` are the
    relpaths that make up the variant (one for a single file, N for shards).
    """
    groups: dict = {}
    variants = []
    for rel, sz in files:
        base = os.path.basename(rel)
        m = _SHARD_RE.match(base)
        if not m:
            variants.append({"filename": base, "bytes": int(sz),
                             "members": [rel]})
            continue
        idx = int(m.group("idx"))
        # Group by (containing dir, shard stem) so two sharded quant families in
        # their own subdirs never merge.
        key = (os.path.dirname(rel), m.group("stem").lower())
        g = groups.setdefault(key, {"bytes": 0, "members": [],
                                    "entry_rel": None, "entry_idx": None})
        g["bytes"] += int(sz)
        g["members"].append(rel)
        if g["entry_idx"] is None or idx < g["entry_idx"]:
            g["entry_idx"] = idx
            g["entry_rel"] = rel
    for g in groups.values():
        variants.append({"filename": os.path.basename(g["entry_rel"]),
                         "bytes": g["bytes"], "members": sorted(g["members"])})
    variants.sort(key=lambda v: v["filename"].lower())
    return variants


def gguf_variants_detail(model_key: str, model_dir: str, cfg=None) -> dict:
    """Per-variant sizes + the EFFECTIVE (resolved) quant for a GGUF model.

    A GGUF repo commonly holds several quantizations (Q4_K_M, Q5_K_M, Q8_0…) but
    only ONE is served, so summing the whole directory badly overstates "the
    model". This resolves the single quant the runner will actually load — exactly
    as ``get_gguf_file`` does (operator ``gguf_file`` override → ``cfg.filename``
    → deterministic auto-rank) — and reports its size plus each pickable variant's
    size. Model-level and worker-agnostic (a ``.gguf`` is identical bytes on every
    box), so the console reuses this one number wherever a GGUF's size is shown.

    Returns ``{}`` for a dir with no servable .gguf (e.g. a transformers model or
    a not-yet-downloaded repo), so callers fall back to their existing size.

    Shard-aware: a split GGUF is N shard files that are ONE model — they collapse
    into a single variant whose bytes SUM the shards (see ``_gguf_variant_groups``),
    so a sharded model resolves a real ``effective_bytes`` instead of ``None``
    (t33: a ~48GB sharded coder model that read as no size at all).
    """
    files = _servable_gguf_files(model_dir)         # recursive; catches nested shards
    if not files:
        return {}
    mmproj = _mmproj_bytes(model_dir)
    variants = _gguf_variant_groups(files)          # shard sets folded into one variant
    # Resolve the effective entrypoint exactly as the runner does (operator
    # gguf_file override -> cfg.filename -> deterministic auto-rank), then find
    # which variant that resolved file belongs to (a shard maps to its group).
    eff_full = None
    try:
        from ...imports.config.main import get_gguf_file
        prefer = (get_override(model_key) or {}).get("gguf_file") or None
        p = get_gguf_file(model_dir, cfg, prefer=prefer)
        eff_full = os.path.abspath(p) if p else None
    except Exception:  # noqa: BLE001 — resolution is best-effort
        eff_full = None
    eff_variant = None
    if eff_full:
        eff_base = os.path.basename(eff_full).lower()
        for v in variants:
            member_fulls = {os.path.abspath(os.path.join(model_dir, m))
                            for m in v["members"]}
            if eff_full in member_fulls or any(
                    os.path.basename(m).lower() == eff_base for m in v["members"]):
                eff_variant = v
                break
    if eff_variant is None and len(variants) == 1:
        eff_variant = variants[0]                    # single variant is unambiguously it
    eff = eff_variant["filename"] if eff_variant else None
    eff_quant = eff_variant["bytes"] if eff_variant else 0
    out_variants = [{"filename": v["filename"], "bytes": v["bytes"],
                     "is_effective": (v is eff_variant)} for v in variants]
    # MoE detection + expert/non-expert byte split of the EFFECTIVE quant
    # (spill.gguf_moe_detail — cached per file, so this rides the same
    # discovery/enrichment reads effective_bytes does at no recurring cost).
    # Central feasibility uses non_expert_bytes as the GPU-side need under the
    # MoE-split plan; a dense model or any read failure simply omits the key.
    moe = None
    try:
        moe_path = eff_full
        if not moe_path and eff_variant:
            moe_path = os.path.join(model_dir, eff_variant["members"][0])
        if moe_path and os.path.isfile(moe_path):
            from ..spill import gguf_moe_detail
            d_moe = gguf_moe_detail(moe_path)
            if d_moe.get("is_moe"):
                # Compact wire view (the per-layer map stays worker-side in the
                # spill cache; central sizing needs only the byte totals).
                moe = {"is_moe": True,
                       "expert_count": d_moe.get("expert_count"),
                       "expert_used_count": d_moe.get("expert_used_count"),
                       "sparsity": d_moe.get("sparsity"),
                       "expert_bytes": d_moe.get("expert_bytes"),
                       "non_expert_bytes": d_moe.get("non_expert_bytes")}
    except Exception:  # noqa: BLE001 — MoE detail is additive; never break sizing
        moe = None
    return {
        **({"moe": moe} if moe else {}),
        "variants": out_variants,                   # [{filename, bytes, is_effective}]
        "mmproj_bytes": mmproj,
        "effective_gguf": eff,
        "effective_quant_bytes": eff_quant,
        # What the model actually is on disk when served: the one quant (SUMMED
        # across shards for a split GGUF) + its projector (0 for text-only). None
        # only when no effective variant can be resolved among many — caller
        # keeps its own size.
        "effective_bytes": (eff_quant + mmproj) if eff_quant else None,
    }


def _coerce(field: str, value):
    if value is None or value == "":
        return None  # signals "clear this field"
    if field == "alloc_mode":
        # Legacy names (autofit/cpu-only/...) resolve to canonical on WRITE, so
        # the stored value is always one of the five flat modes — accepted on
        # input, never stored/emitted back. Unknown -> clear + log (degrade,
        # never 500 a serving-settings POST over a typo).
        from ..alloc_modes import resolve_alloc_mode, ALLOC_MODES
        canonical, was_alias = resolve_alloc_mode(value)
        if canonical is None:
            logger.warning("ignoring unknown alloc_mode %r (recognized: %s)",
                           value, ", ".join(ALLOC_MODES))
            return None
        if was_alias:
            logger.info("alloc_mode legacy name %r stored as %r", value, canonical)
        return canonical
    if field == "priority_device":
        v = str(value).strip().lower()
        if v not in ("gpu", "ram"):
            logger.warning("ignoring unknown priority_device %r (gpu|ram)", value)
            return None
        return v
    if field in _INT_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    if field in _BOOL_FIELDS:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return str(value)


def set_override(model_key: str, fields: dict) -> dict:
    """Merge ``fields`` into the model's override; a None/"" value clears a key.

    Returns the model's full override after the update.
    """
    with _LOCK:
        data = _load()
        current = dict(data.get(model_key, {}) or {})
        for key, raw in (fields or {}).items():
            if key not in ALLOWED_FIELDS:
                continue
            coerced = _coerce(key, raw)
            if coerced is None:
                current.pop(key, None)
            else:
                current[key] = coerced
        if current:
            data[model_key] = current
        else:
            data.pop(model_key, None)
        _save(data)
        return current


def effective_alloc_mode(model_key: str) -> str:
    """The model's EFFECTIVE allocation mode (k37) — the persisted
    ``alloc_mode`` when set, else READ-TIME DERIVATION from the legacy knobs
    (n_gpu_layers -1 -> gpu-only, 0/"off" -> ram-only, explicit budgets/bands
    -> explicit, unset -> max-gpu). Derivation IS the migration: no override
    file is ever rewritten for the rename, and a blank model reads max-gpu
    (fit-and-spill, never OOM — defaults-are-promises)."""
    from ..alloc_modes import derive_alloc_mode
    return derive_alloc_mode(get_override(model_key))


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_OVERRIDES_PATH) or ".", exist_ok=True)
    tmp = _OVERRIDES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, _OVERRIDES_PATH)


def _bare_key(k: str) -> str:
    """Model name without the 'owner~' collision qualifier."""
    return k.split("~", 1)[1] if "~" in k else k


def migrate_overrides(registry) -> dict:
    """Heal overrides orphaned when a model key got collision-qualified
    (``name`` -> ``owner~name``). For each override key absent from the registry,
    re-key it to the qualified key sharing its bare suffix. If several owners
    match (a real collision), disambiguate by the override's ``gguf_file`` vs each
    variant's ``filename``; skip + log when still ambiguous so a human picks the
    owner. Idempotent; never clobbers an existing override on the target.

    ``registry``: ``{model_key: cfg-dict-or-ModelConfig}``. Returns ``{old: new}``.
    """
    moved: dict = {}
    with _LOCK:
        ov = _load()
        if not ov:
            return moved
        keys = list(registry)

        def _fname(k):
            cfg = registry.get(k)
            fn = cfg.get("filename") if isinstance(cfg, dict) else getattr(cfg, "filename", None)
            return os.path.basename(str(fn or ""))

        for okey in list(ov):
            if okey in registry:
                continue                                   # still a valid key
            cands = [k for k in keys if _bare_key(k) == okey]
            if not cands:
                continue                                   # model gone — leave override
            target = cands[0] if len(cands) == 1 else None
            if target is None:                             # multi-owner: use gguf_file hint
                gf = os.path.basename(str((ov[okey] or {}).get("gguf_file") or ""))
                matched = [k for k in cands if gf and _fname(k) == gf]
                target = matched[0] if len(matched) == 1 else None
            if target and target not in ov:
                ov[target] = ov.pop(okey)
                moved[okey] = target
                logger.info("serve override migrated: %r -> %r", okey, target)
            elif not target:
                logger.warning("serve override %r orphaned + ambiguous across %s; "
                               "re-key manually", okey, cands)
        if moved:
            _save(ov)
    return moved
