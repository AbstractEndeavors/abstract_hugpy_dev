"""Identity RECONSTRUCTION job schema (studio stage (b)) — the durable, bus-carried
intent for "generate an identity-locked turnaround of a character from its reference
images + description, for approval".

A reconstruction job renders ONE identity-locked still per requested VIEW
(``front`` / ``three_quarter`` / ``profile`` / ``back`` by default), then hands the
produced stills to ``identity_profiles.attach_reconstruction`` so they land in the
identity's own directory awaiting the operator's approval. It is an ORCHESTRATOR job
(like ``generate_studio_movie``): the route enqueues ONE of these and the runner
(``runners/identity_reconstruction.py``) drives the per-view renders INLINE through
the shared render primitive, so a single ``job_id`` covers the whole set.

House style mirrors ``studio.job`` / ``studio_movie_schema``: a frozen, JSON-safe,
validate-at-construction spec built ONLY via ``make_identity_reconstruction``; the
bus rehydrates it through ``identity_reconstruction_from_dict`` (reconstruct +
RE-VALIDATE). All fields are primitives / string tuples so ``asdict`` -> ``json``
round-trips cleanly with zero enum/dataclass ceremony.

Geometry defaults to the Wan-VACE id_lock ceiling (<=480p — a 512² id_lock fails
``no_capable_model``): a square 480×480 @ 16fps is a sensible portrait default. The
budget defaults to ``None`` = AUTOFIT (blank budget sizes to the serving worker's
measured free VRAM at render time; never a guaranteed-fail low guess).

No pathlib anywhere. os.path only (none needed here — pure data).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# The subject reference images cap — an id_lock render consumes at most this many
# (diffusers 0.39 prepends each as a VACE reference latent). Mirrors
# ``studio.job._MAX_REFERENCE_IMAGES`` / ``identity_profiles.MAX_REFERENCE_IMAGES``.
_MAX_REFERENCE_IMAGES = 4

# The default set of turnaround views (order preserved end-to-end). A single-view
# request (e.g. ``["front"]``) is valid — the cheap way to verify one render first.
DEFAULT_VIEWS: Tuple[str, ...] = ("front", "three_quarter", "profile", "back")

# Wan-VACE id_lock geometry ceiling is 480p; a square portrait still default @ 16fps.
_DEFAULT_WIDTH, _DEFAULT_HEIGHT, _DEFAULT_FPS = 480, 480, 16

# Reconstruction MODE (shared contract):
#   "sheet"     — the EXISTING behaviour: N INDEPENDENT view-stills (one id_lock clip
#                 per named view, only frame 0 kept). The default — wire shape unchanged.
#   "turntable" — NEW: ONE id_lock ORBIT clip whose EVERY frame is kept, each frame a
#                 degree-view; the record's ordered ``views`` hold the frames in angular
#                 order. The runner branches on this.
RECON_MODES: Tuple[str, ...] = ("sheet", "turntable")
_DEFAULT_MODE = "sheet"

# Turntable frame cap: a studio id_lock clip is spine-derived (~2s, model-capped), so at
# <=16fps it yields well under this. A HIGH cap (never a truncator on the happy path) so
# the frame-extract runner keeps EVERY frame of the orbit clip as a dense angular sequence.
_DEFAULT_TURNTABLE_MAX_FRAMES = 240


@dataclass(frozen=True)
class IdentityReconstructionSpec:
    """Frozen currency of an ``identity_reconstruction`` bus job.

        slug             the identity profile this reconstruction belongs to (the
                         store key ``attach_reconstruction`` writes back to).
        recon_id         the minted id of THIS turnaround set (the route mints it and
                         returns it alongside the job_id so the UI can correlate).
        reference_images the ORDERED, jailed abs paths of the subject reference
                         image(s) the id_lock render conditions on (1..4). CANONICAL —
                         they define the identity. The route resolves + validates them
                         (canonical set preferred over the raw uploads) before enqueue.
        base_prompt      the extra description woven into every view prompt (the
                         profile's notes by default; may be empty).
        views            the ORDERED tuple of view names to render (>=1). A view-
                         specific prompt is derived per view in the runner.
        seed             base render seed (all views share it; the view differs by
                         prompt, not seed, so the identity stays put).
        width/height/fps id_lock render geometry (<=480p ceiling — see module header).
        vram_budget_gb   AUTOFIT tier. ``None`` (default) = size to the serving
                         worker's measured free VRAM at render time; a number PINS it.
        mode             "sheet" (default — N independent view-stills, existing path)
                         or "turntable" (one 360° orbit clip, every frame kept). The
                         runner branches on this; absent => "sheet" (backward-compat).
        turntable_max_frames  turntable-only hint: the frame-extract cap (every frame
                         of the orbit clip is kept up to this). Ignored in "sheet" mode.
    """
    slug: str
    recon_id: str
    reference_images: Tuple[str, ...]
    views: Tuple[str, ...] = DEFAULT_VIEWS
    base_prompt: str = ""
    seed: int = 0
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    fps: int = _DEFAULT_FPS
    vram_budget_gb: Optional[float] = None
    mode: str = _DEFAULT_MODE
    turntable_max_frames: int = _DEFAULT_TURNTABLE_MAX_FRAMES


def make_identity_reconstruction(
    *,
    slug: str,
    recon_id: str,
    reference_images: Tuple[str, ...],
    views: Optional[Tuple[str, ...]] = None,
    base_prompt: str = "",
    seed: int = 0,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    fps: int = _DEFAULT_FPS,
    vram_budget_gb: Optional[float] = None,
    mode: str = _DEFAULT_MODE,
    turntable_max_frames: int = _DEFAULT_TURNTABLE_MAX_FRAMES,
) -> IdentityReconstructionSpec:
    """Validate every field and build the frozen ``IdentityReconstructionSpec``.
    Raises ``ValueError``/``TypeError`` LOCALLY on any structural violation (house
    discipline: a structurally-invalid spec is caller error caught at the boundary,
    never carried across the bus). Runtime policy failures (an unroutable render, an
    archived profile) are NOT validated here — they surface as errors-as-data from the
    runner / store."""
    if not (isinstance(slug, str) and slug.strip()):
        raise ValueError(f"slug must be a non-empty string; got {slug!r}")
    if not (isinstance(recon_id, str) and recon_id.strip()):
        raise ValueError(f"recon_id must be a non-empty string; got {recon_id!r}")

    # reference images: coerce list/tuple -> tuple; each a non-empty string; 1..4.
    if isinstance(reference_images, (list, tuple)):
        reference_images = tuple(reference_images)
    else:
        raise ValueError(
            f"reference_images must be a list/tuple of paths; got {reference_images!r}")
    if not reference_images:
        raise ValueError("at least one reference_image is required")
    for i, r in enumerate(reference_images):
        if not (isinstance(r, str) and r.strip()):
            raise ValueError(f"reference_images[{i}] must be a non-empty string; got {r!r}")
    if len(reference_images) > _MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"at most {_MAX_REFERENCE_IMAGES} reference_images are accepted; "
            f"got {len(reference_images)}")

    # views: None -> the default turnaround set; coerce list/tuple -> tuple; >=1;
    # each a non-empty string.
    if views is None:
        views = DEFAULT_VIEWS
    if isinstance(views, (list, tuple)):
        views = tuple(views)
    else:
        raise ValueError(f"views must be a list/tuple of view names; got {views!r}")
    if not views:
        raise ValueError("at least one view is required")
    for i, v in enumerate(views):
        if not (isinstance(v, str) and v.strip()):
            raise ValueError(f"views[{i}] must be a non-empty string; got {v!r}")

    if base_prompt is not None and not isinstance(base_prompt, str):
        raise ValueError(f"base_prompt must be a string or None; got {base_prompt!r}")
    base_prompt = base_prompt or ""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {seed!r}")
    for name, val in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive int; got {val!r}")
    # AUTOFIT: None is a LEGAL value (blank -> fit to the worker's free VRAM at render
    # time). A NUMBER is the manual override and must be positive.
    if vram_budget_gb is not None:
        if not isinstance(vram_budget_gb, (int, float)) or isinstance(vram_budget_gb, bool) \
                or vram_budget_gb <= 0:
            raise ValueError(
                f"vram_budget_gb must be a positive number or None (autofit); "
                f"got {vram_budget_gb!r}")

    # mode: absent/None -> the default "sheet" (backward-compat); else one of RECON_MODES.
    if mode is None:
        mode = _DEFAULT_MODE
    if not (isinstance(mode, str) and mode in RECON_MODES):
        raise ValueError(f"mode must be one of {list(RECON_MODES)}; got {mode!r}")

    # turntable_max_frames: a positive int cap (only consulted in turntable mode).
    if not isinstance(turntable_max_frames, int) or isinstance(turntable_max_frames, bool) \
            or turntable_max_frames <= 0:
        raise ValueError(
            f"turntable_max_frames must be a positive int; got {turntable_max_frames!r}")

    return IdentityReconstructionSpec(
        slug=slug,
        recon_id=recon_id,
        reference_images=reference_images,
        views=views,
        base_prompt=base_prompt,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        vram_budget_gb=(float(vram_budget_gb) if vram_budget_gb is not None else None),
        mode=mode,
        turntable_max_frames=turntable_max_frames,
    )


def identity_reconstruction_from_dict(d: dict) -> IdentityReconstructionSpec:
    """Rebuild an ``IdentityReconstructionSpec`` from its ``asdict`` form THROUGH the
    validating factory (deserialize-then-revalidate, like every other studio spec).
    Registered in ``media_bus.SPEC_DESERIALIZERS`` under ``"identity_reconstruction"``."""
    return make_identity_reconstruction(
        slug=d["slug"],
        recon_id=d["recon_id"],
        reference_images=d.get("reference_images") or (),
        views=d.get("views"),
        base_prompt=d.get("base_prompt", ""),
        seed=d.get("seed", 0),
        width=d.get("width", _DEFAULT_WIDTH),
        height=d.get("height", _DEFAULT_HEIGHT),
        fps=d.get("fps", _DEFAULT_FPS),
        vram_budget_gb=d.get("vram_budget_gb"),
        # Backward-compat: an old serialized spec has no ``mode`` -> "sheet".
        mode=d.get("mode") or _DEFAULT_MODE,
        turntable_max_frames=d.get("turntable_max_frames", _DEFAULT_TURNTABLE_MAX_FRAMES),
    )
