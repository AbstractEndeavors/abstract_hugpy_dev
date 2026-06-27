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
import os
import threading

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
        os.makedirs(os.path.dirname(_OVERRIDES_PATH) or ".", exist_ok=True)
        tmp = _OVERRIDES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, _OVERRIDES_PATH)
        return current
