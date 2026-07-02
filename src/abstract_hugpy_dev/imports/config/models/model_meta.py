"""F3.1 — model metadata + recommended settings, from ONE place (CON-03).

Every model-selection surface (HugPy Console ModelTable, media-intelligence
ComposerControls, Discord embeds) reads this — no UI hardcodes a model fact.
The registry (ModelConfig) stays the identity source; this module derives
the operational metadata the roadmap wants surfaced everywhere:

    size_bytes    real on-disk footprint (all shards; cached by dir mtime)
    quant         parsed from the GGUF filename (q4_k_m, iq3_xs, f16, ...)
    params_b      billions of parameters, parsed from name/hub_id (7B, 0.5b)
    ctx_max       the model's context window (model_max_length)
    recommended   settings annotated against a target's VRAM — not static
                  constants: ctx, n_gpu_layers, threads, and the reason.

Honesty rule: when we don't know, fields are None and the reason says why —
a picker that shows "?" beats one that shows a confident wrong number.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Optional

# quant tokens as they appear in GGUF filenames, longest-match first.
_QUANT_RE = re.compile(
    r"(?i)[-._](i?q[0-9]+(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|fp32)(?=[-._]|$)")
# parameter counts: 7B, 0.5b, 14B, 1.5B — bounded so '2024' never matches.
_PARAMS_RE = re.compile(r"(?i)(?<![0-9.])([0-9]{1,3}(?:\.[0-9]+)?)\s?b(?![a-z0-9])")

# Rough default VRAM headroom multiplier (weights + KV cache + overhead).
_VRAM_HEADROOM = 1.15

_SIZE_CACHE: dict[str, tuple[float, int]] = {}
_SIZE_LOCK = threading.Lock()


def parse_quant(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    m = _QUANT_RE.search(filename)
    return m.group(1).lower() if m else None


def parse_params_b(*names: Optional[str]) -> Optional[float]:
    for name in names:
        if not name:
            continue
        m = _PARAMS_RE.search(name)
        if m:
            try:
                v = float(m.group(1))
                if 0.05 <= v <= 700:
                    return v
            except ValueError:
                continue
    return None


def dir_size_bytes(path: Optional[str]) -> Optional[int]:
    """On-disk footprint, cached by directory mtime so /models stays cheap."""
    if not path or not os.path.isdir(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _SIZE_LOCK:
        hit = _SIZE_CACHE.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    with _SIZE_LOCK:
        _SIZE_CACHE[path] = (mtime, total)
    return total


def _model_dir(cfg: Any) -> Optional[str]:
    """Absolute model directory from a ModelConfig(-ish) object or dict."""
    folder = None
    if isinstance(cfg, dict):
        folder = cfg.get("folder")
    else:
        folder = getattr(cfg, "folder", None)
    if not folder:
        return None
    if os.path.isabs(folder):
        return folder
    try:
        from ...src.constants.constants import MODELS_HOME
        return os.path.join(str(MODELS_HOME), folder)
    except Exception:
        return None


def recommended_settings(*, size_bytes: Optional[int], ctx_max: Optional[int],
                         framework: str = "",
                         vram_bytes: Optional[int] = None) -> dict:
    """Recommendations annotated against a target's VRAM (CON-03: not static
    constants). vram_bytes None = no target known; recommendations degrade
    honestly and the reason explains what's missing."""
    rec: dict[str, Any] = {
        "ctx": min(int(ctx_max), 32768) if ctx_max else None,
        "n_gpu_layers": None,
        "threads": max(2, min(8, (os.cpu_count() or 4) // 2)),
        "fits_vram": None,
        "reason": "",
    }
    if size_bytes is None:
        rec["reason"] = "model size unknown (not downloaded?) — no offload advice"
        return rec
    need = int(size_bytes * _VRAM_HEADROOM)
    rec["need_bytes"] = need
    if vram_bytes is None:
        rec["reason"] = "no target VRAM given — pass ?vram_gib= for offload advice"
        return rec
    if need <= vram_bytes:
        rec["n_gpu_layers"] = -1
        rec["fits_vram"] = True
        rec["reason"] = "fits fully in VRAM (size×%.2f ≤ free)" % _VRAM_HEADROOM
    else:
        frac = max(0.0, min(1.0, vram_bytes / need))
        rec["fits_vram"] = False
        rec["gpu_fraction"] = round(frac, 2)
        rec["reason"] = ("~%d%% of weights fit — partial offload; exact "
                         "n_gpu_layers depends on layer count (use probe)"
                         % int(frac * 100))
    return rec


def model_meta(cfg: Any, *, vram_bytes: Optional[int] = None) -> dict:
    """The one metadata dict every picker renders. cfg = ModelConfig or its
    to_dict()."""
    d = cfg if isinstance(cfg, dict) else cfg.to_dict()
    name = d.get("name") or ""
    hub_id = d.get("hub_id") or ""
    filename = d.get("filename")
    ctx_max = d.get("model_max_length")
    framework = d.get("framework") or ""
    size = dir_size_bytes(_model_dir(d))
    return {
        "model_key": d.get("model_key"),
        "size_bytes": size,
        "quant": parse_quant(filename),
        "params_b": parse_params_b(name, hub_id, filename),
        "ctx_max": ctx_max,
        "framework": framework,
        "recommended": recommended_settings(
            size_bytes=size, ctx_max=ctx_max, framework=framework,
            vram_bytes=vram_bytes),
    }
