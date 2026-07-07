"""Schemas over ad-hoc objects. Every one is a frozen, slotted dataclass.

The load-bearing type is `RenderManifest` (INV-1): a render is *defined* by its
manifest, and the pixels are a cache of it. `canonical_inputs()` / `content_hash()`
give a stable reproducibility + dedup + resume key (INV-6) that excludes metadata
(render_id, timestamps) and includes everything that changes the output.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .enums import (
    AdapterKind,
    Capability,
    ControlKind,
    DeterminismClass,
    Framework,
    LicenseClass,
    LICENSE_AUTO_COMMERCIAL,
    PathClass,
    Precision,
    RiskFlag,
    Task,
)

# ---------------------------------------------------------------------------
# Value schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Resolution:
    width: int
    height: int
    fps: int  # nominal target fps for the model; real cadence is per-render

    def __post_init__(self) -> None:
        # Structurally-invalid geometry is programmer error, not runtime data.
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError(f"invalid Resolution {self.width}x{self.height}@{self.fps}")

    @property
    def area(self) -> int:
        return self.width * self.height

    def covers(self, target: "Resolution") -> bool:
        """True if this resolution is at least as large as `target` in both dims."""
        return self.width >= target.width and self.height >= target.height


@dataclass(frozen=True, slots=True)
class VramEnvelope:
    """VRAM cost per precision, in GB. Stored as a sorted tuple of pairs so the
    schema stays frozen/hashable. Values are PLANNING ESTIMATES (they move with
    resolution/frames/offload) - the router uses them for fit, not as a promise."""
    per_precision: tuple[tuple[Precision, float], ...]

    def __post_init__(self) -> None:
        if not self.per_precision:
            raise ValueError("VramEnvelope needs at least one precision")
        for prec, gb in self.per_precision:
            if gb <= 0:
                raise ValueError(f"non-positive VRAM for {prec}: {gb}")

    def as_map(self) -> dict[Precision, float]:
        return {p: gb for p, gb in self.per_precision}

    def min_gb(self) -> float:
        return min(gb for _, gb in self.per_precision)

    def fits(self, budget_gb: float) -> tuple[Precision, ...]:
        return tuple(p for p, gb in self.per_precision if gb <= budget_gb)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """All seeds captured, never implicit (INV-2)."""
    global_seed: int
    stage_seeds: tuple[tuple[str, int], ...] = ()   # (stage_name, seed)
    chunk_seed_base: int | None = None               # for autoregressive chunks


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    shift: float | None = None
    sigmas: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ControlRef:
    """A control signal referenced by content hash (INV-1). The pixels of the
    depth map / pose skeleton live in the store; the manifest carries the hash."""
    kind: ControlKind
    content_hash: str
    weight: float = 1.0
    target_frames: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AdapterRef:
    kind: AdapterKind
    adapter_id: str
    weight: float
    weight_hash: str


@dataclass(frozen=True, slots=True)
class ProvenanceStub:
    """INV-7. C2PA is filled in at mastering; this is the internal stub."""
    operator: str
    created_at: str        # ISO-8601
    tool: str = "hugpy-studio"
    c2pa_pending: bool = True


# ---------------------------------------------------------------------------
# RenderManifest - the source of truth for a render (INV-1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RenderManifest:
    render_id: str
    capability: Capability
    model_id: str
    weight_hash: str | None          # None only if the bound model is unpinned
    framework: Framework
    task: Task
    precision: Precision             # FIX-1: router-selected precision changes the
                                     # output (fp8 vs bf16), so it MUST be in the hash
    seeds: SeedBundle
    sampler: SamplerConfig
    resolution_ladder: tuple[Resolution, ...]
    controls: tuple[ControlRef, ...] = ()
    adapters: tuple[AdapterRef, ...] = ()
    identity_ids: tuple[str, ...] = ()           # character stable IDs (§3)
    identity_view_hashes: tuple[str, ...] = ()   # canonical multi-view refs (ID-3)
    determinism_class: DeterminismClass = DeterminismClass.SEEDED_APPROX
    env_snapshot: tuple[tuple[str, str], ...] = ()
    provenance: ProvenanceStub | None = None
    # Text conditioning (C-prompt): the prompt genuinely changes the output, so it
    # is part of the reproducibility key (canonical_inputs -> content_hash). "" is a
    # valid empty prompt (image-conditioned i2v). Appended (not inserted) so no
    # positional field shifts for existing construction sites.
    prompt: str = ""
    negative_prompt: str = ""

    def canonical_inputs(self) -> dict:
        """Everything that changes the output; nothing that is mere metadata.
        Excludes render_id + provenance so two identical intents hash equal."""
        return {
            "capability": self.capability.value,
            "model_id": self.model_id,
            "weight_hash": self.weight_hash,
            "framework": self.framework.value,
            "task": self.task.value,
            "precision": self.precision.value,   # FIX-1: fp8 vs bf16 must not collide
            "seeds": {
                "global": self.seeds.global_seed,
                "stage": sorted(self.seeds.stage_seeds),
                "chunk_base": self.seeds.chunk_seed_base,
            },
            "sampler": {
                "sampler": self.sampler.sampler,
                "scheduler": self.sampler.scheduler,
                "steps": self.sampler.steps,
                "cfg": self.sampler.cfg,
                "shift": self.sampler.shift,
                "sigmas": list(self.sampler.sigmas),
            },
            "resolution_ladder": [
                [r.width, r.height, r.fps] for r in self.resolution_ladder
            ],
            "controls": sorted(
                [c.kind.value, c.content_hash, c.weight, list(c.target_frames)]
                for c in self.controls
            ),
            "adapters": sorted(
                [a.kind.value, a.adapter_id, a.weight, a.weight_hash]
                for a in self.adapters
            ),
            "identity_ids": sorted(self.identity_ids),
            "identity_view_hashes": sorted(self.identity_view_hashes),
            "determinism_class": self.determinism_class.value,
            "env_snapshot": sorted(self.env_snapshot),
            # C-prompt: text conditioning changes the pixels, so it is in the hash.
            # Empty prompt still participates (its value is just ""), which re-addresses
            # ALL prior content-addressed clips once — correct + acceptable for dev.
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
        }

    def content_hash(self) -> str:
        blob = json.dumps(self.canonical_inputs(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ModelConfig - one row of the zoo, as data (§2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    family: Framework
    tasks: tuple[Task, ...]
    capabilities: tuple[Capability, ...]
    vram: VramEnvelope
    resolutions: tuple[Resolution, ...]
    max_frames: int
    max_duration_s: float
    license: LicenseClass
    weight_uri: str                       # HF repo / GitHub / local
    source_url: str
    default_determinism: DeterminismClass
    path_class: PathClass = PathClass.OFFLINE
    native_audio: bool = False
    accepts_adapters: frozenset[AdapterKind] = frozenset()
    weight_hash: str | None = None        # pin before production
    unpinned: bool = False                # must be True if weight_hash is None
    verify_uri: bool = False              # repo path not confirmed this session
    synthetic: bool = False               # LAST-RESORT placeholder (no-model
                                          # procedural runner). The router ranks any
                                          # REAL model strictly above a synthetic
                                          # one, so it binds only when no real model
                                          # fits the request. Set True on synthetic
                                          # rows ONLY (see models_seed synthetic-i2v).
    notes: str = ""

    @property
    def commercial_auto(self) -> bool:
        return LICENSE_AUTO_COMMERCIAL[self.license]

    def supports_resolution(self, target: Resolution) -> bool:
        return any(r.covers(target) for r in self.resolutions)

    def best_native_area(self) -> int:
        return max(r.area for r in self.resolutions)


# ---------------------------------------------------------------------------
# Request / binding / job
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    """What a shot asks for. The router turns this into a ModelBinding or an Err."""
    capability: Capability
    target_resolution: Resolution
    vram_budget_gb: float
    commercial_use: bool = False
    allowed_licenses: frozenset[LicenseClass] = frozenset()  # empty = any
    latency_budget_ms: int | None = None    # set => streaming path required (STR-6)
    require_native_audio: bool = False
    preferred_framework: Framework | None = None
    risk_flags: frozenset[RiskFlag] = frozenset()
    min_frames: int = 0


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """The router's resolved answer: which model, which runner, which precision."""
    model_id: str
    framework: Framework
    task: Task
    precision: Precision
    path_class: PathClass
    weight_uri: str
    weight_hash: str | None
    determinism_class: DeterminismClass   # FIX-3: carried from the bound model's
                                          # default_determinism so a manifest built
                                          # from this binding reflects the real class
                                          # (EXACT/SEEDED_APPROX/DRIFTING), not a
                                          # hardcoded literal.


@dataclass(frozen=True, slots=True)
class Job:
    """Frozen currency of the queue (ORCH-1). Lifecycle STATE is not stored here -
    it lives in the append-only ledger (ORCH-4); the Job itself never mutates."""
    job_id: str
    request: CapabilityRequest
    binding: ModelBinding
    manifest: RenderManifest
    priority: int = 100
    retake_budget: int = 2
    risk_flags: frozenset[RiskFlag] = frozenset()
    provenance: ProvenanceStub | None = None


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One append-only transition (ORCH-4/ORCH-6). Errors ride along as data."""
    job_id: str
    state: "JobStateT"
    at: str                       # ISO-8601
    detail: str = ""
    error: "StageErrorT | None" = None


# Late imports for annotations only (avoid a hard import cycle with errors/enums
# at module top while keeping the names available for tooling).
from .enums import JobState as JobStateT  # noqa: E402
from .errors import StageError as StageErrorT  # noqa: E402
