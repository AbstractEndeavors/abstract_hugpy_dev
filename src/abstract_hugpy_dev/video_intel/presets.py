"""Curated video-generation presets — "ideal default loads".

A tiny, frozen registry the UI reads to offer a dropdown of ready-made
scene-generation setups. Each preset names a catalog model_key, a coherence
``mode`` and a bundle of per-frame generation defaults, so selecting one is
enough to warm the right model on a GPU worker and pre-fill the generate_scene
form.

Deliberately mirrors the keyword-preset pattern in
``managers/keywords/keybert_model.py`` (a frozen ``*Preset`` dataclass + a plain
dict registry + ``available_presets()``/``get_preset()`` accessors) so the two
read the same way. Nothing here restarts a service or touches the serve/worker
control plane — it is a static table the HTTP surface dumps and looks up.

``mode`` maps to generate_scene coherence semantics (the runner + UI already
understand ``chain``):
    "edit-chain"     -> chain=True  (start-frame edit chain; each frame
                        conditions on the previous frame's output)
    "img2img"        -> chain=False (off-start img2img; every frame conditions
                        on the same start frame — no drift)
    "text-to-image"  -> v1 no-start path (pure prompt-scheduled generation)
The preset keeps ``mode`` as a plain string; translating it to ``chain`` is the
caller's job (generate_scene defaults chain=True).

Quant note (do NOT plumb): ``HUGPY_IMG2IMG_QUANTIZE`` is ENV-ONLY (read once in
``managers/imagegen/imagegen_runner.py``), NOT per-request. Its default "auto"
already 4-bit-quantizes any model that would exceed free VRAM (so
Qwen-Image-Edit auto-quants on a GPU worker). The advisory ``quant`` field below
is DISPLAY-ONLY — it changes nothing at runtime; there is intentionally no
per-request quant plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema — one frozen preset row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoGenPreset:
    """Immutable "ideal default load" for scene generation.

    ``model_key``  a real catalog key (validated at apply time against
                   get_models_dict — the same source workers_assign uses).
    ``mode``       "edit-chain" | "img2img" | "text-to-image" (see module docs).
    The remaining fields are the per-frame generation defaults the UI pre-fills;
    ``defaults()`` bundles exactly the pinned contract's ``defaults`` sub-shape.
    ``quant``      advisory display string only — see the module quant note; it
                   is NOT plumbed into any generation request.
    """

    id: str
    name: str
    description: str
    mode: str
    model_key: str
    # per-frame generation defaults
    strength: float
    steps: int
    guidance: float
    width: int
    height: int
    n_frames: int
    fps: int
    negative: str = ""
    recommended: str = "gpu"
    quant: str = "auto"          # advisory/display only — see module quant note

    def defaults(self) -> Dict[str, Any]:
        """The pinned ``defaults`` sub-object (order/keys frozen for the UI)."""
        return {
            "strength": self.strength,
            "steps": self.steps,
            "guidance": self.guidance,
            "width": self.width,
            "height": self.height,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "negative": self.negative,
        }

    def to_dict(self) -> Dict[str, Any]:
        """The pinned per-preset wire shape for GET /video/presets."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "model_key": self.model_key,
            "defaults": self.defaults(),
            "recommended": self.recommended,
            # advisory, display-only (see module quant note) — additive to the
            # pinned shape; the UI is free to ignore it.
            "quant": self.quant,
        }


# ---------------------------------------------------------------------------
# Registry — insertion order is the dropdown order
# ---------------------------------------------------------------------------
VIDEO_PRESETS: Dict[str, VideoGenPreset] = {}


def register_preset(preset: VideoGenPreset) -> None:
    if preset.id in VIDEO_PRESETS:
        raise KeyError(f"Video preset {preset.id!r} already registered")
    VIDEO_PRESETS[preset.id] = preset


def available_presets() -> List[VideoGenPreset]:
    """All presets in registration (dropdown) order."""
    return list(VIDEO_PRESETS.values())


def get_preset(preset_id: str) -> Optional[VideoGenPreset]:
    """The preset for ``preset_id`` or None (the apply route maps None -> 404)."""
    return VIDEO_PRESETS.get(preset_id)


# ---- built-in seed presets ------------------------------------------------
# model_key values validated against the live catalog (get_models_dict) on
# 2026-07-06; see the module/route docs for the mode->chain mapping.

register_preset(VideoGenPreset(
    id="realistic-edit-chain",
    name="Realistic edit-chain",
    description=("Instruction-driven frame evolution; best coherence; "
                 "auto-4bit on GPU."),
    mode="edit-chain",
    # Qwen-Image-Edit-2509 (transformers, image-to-image) — exact catalog key.
    model_key="a3527183~Qwen-Image-Edit-2509",
    strength=0.5,
    steps=28,
    guidance=4.0,
    width=1024,
    height=1024,
    n_frames=8,
    fps=8,
    negative="",
))

register_preset(VideoGenPreset(
    id="realistic-img2img",
    name="Realistic img2img sequence",
    description=("Off-start img2img: every frame conditions on the same start "
                 "frame (no drift); photoreal SDXL."),
    mode="img2img",
    # Photoreal SDXL — Juggernaut XL (comfy, text-to-image + image-to-image).
    # The bare candidate keys resolve only under the comfy- prefix in the live
    # catalog; Juggernaut XL is the true SDXL matching the 1024-native defaults.
    model_key="comfy-juggernautxl-ragnarok",
    strength=0.5,
    steps=30,
    guidance=6.0,
    width=1024,
    height=1024,
    n_frames=8,
    fps=8,
    negative="",
))

register_preset(VideoGenPreset(
    id="fast-draft",
    name="Fast draft",
    description=("Quick low-step turbo preview; text-to-image, no start "
                 "frame."),
    mode="text-to-image",
    # SDXL-Turbo (transformers, text-to-image) — exact catalog key.
    model_key="sdxl-turbo",
    strength=0.6,
    steps=6,
    guidance=1.5,
    width=768,
    height=768,
    n_frames=6,
    fps=8,
    negative="",
))
