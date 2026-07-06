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
    "threads",        # CPU threads
    "llama_ctx",      # context window
    "gpu_mem_gib",    # transformers per-GPU budget
    "cpu_mem_gib",    # transformers CPU/RAM budget
    "always_on",      # systemd always-on vs swap on-demand
    "ttl_seconds",    # swap idle-unload TTL
    "gguf_file",      # which downloaded .gguf to serve (basename; "" = auto/default)
}
_INT_FIELDS = {"n_gpu_layers", "threads", "llama_ctx", "ttl_seconds"}
_FLOAT_FIELDS = {"gpu_mem_gib", "cpu_mem_gib"}
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
    servable variant on its own."""
    try:
        from ...imports.src.utils import is_mmproj_file
    except Exception:  # noqa: BLE001
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
    total = 0
    try:
        for fn in os.listdir(model_dir):
            if fn.lower().endswith(".gguf") and is_mmproj_file(fn):
                total += _file_bytes(model_dir, fn)
    except OSError:
        pass
    return total


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
    """
    names = _gguf_basenames(model_dir)              # servable variants (no mmproj)
    if not names:
        return {}
    mmproj = _mmproj_bytes(model_dir)
    eff = None
    try:
        from ...imports.config.main import get_gguf_file
        prefer = (get_override(model_key) or {}).get("gguf_file") or None
        p = get_gguf_file(model_dir, cfg, prefer=prefer)
        eff = os.path.basename(p) if p else None
    except Exception:  # noqa: BLE001 — resolution is best-effort
        eff = None
    if eff not in names:
        eff = names[0] if len(names) == 1 else None
    variants = [{"filename": n, "bytes": _file_bytes(model_dir, n),
                 "is_effective": (n == eff)} for n in names]
    eff_quant = next((v["bytes"] for v in variants if v["is_effective"]), 0)
    return {
        "variants": variants,                       # [{filename, bytes, is_effective}]
        "mmproj_bytes": mmproj,
        "effective_gguf": eff,
        "effective_quant_bytes": eff_quant,
        # What the model actually is on disk when served: the one quant + its
        # projector (0 for text-only). None when the effective quant can't be
        # resolved (multi-variant, no choice) — caller keeps its own size.
        "effective_bytes": (eff_quant + mmproj) if eff_quant else None,
    }


def _coerce(field: str, value):
    if value is None or value == "":
        return None  # signals "clear this field"
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
