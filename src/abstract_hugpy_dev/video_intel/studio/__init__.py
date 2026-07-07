"""hugpy-studio: capability layer over a video-generation model zoo.

Importing this package registers the zoo (models_seed) so the registries are
populated. Validation is EXPLICIT, not at import: call ``validate_registry()``
yourself (fail-loud, comprehensive) — the embedded package must never abort a
bare ``import abstract_hugpy_dev.video_intel.studio`` just because the seeded
models are still unpinned (they all are, in this slice). Tests and the eventual
serve path call ``validate_registry()`` before dispatch.

NOTE (P0-6): the ``RunnerSpec.entrypoint`` dotted paths
(``abstract_hugpy_dev.video_intel.studio.runners.*``) are declared but UNWIRED —
the ``runners`` subpackage does not exist yet (gated P0-6 slice).
``validate_registry()`` only checks a runner is *registered*, not importable, so
this dormant state is intentional and does not break import or validation.
"""

from __future__ import annotations

from . import models_seed  # noqa: F401  (side effect: populates registries)
from .enums import (
    AdapterKind,
    Capability,
    ControlKind,
    DeterminismClass,
    Framework,
    JobState,
    LicenseClass,
    PathClass,
    Precision,
    RiskFlag,
    Task,
)
from .errors import ConfigError, Err, ErrorCode, Ok, RegistryError, Result, StageError
from .registry import (
    CAPABILITY_TASKS,
    MODEL_REGISTRY,
    PLANNED_CAPABILITIES,
    RUNNER_REGISTRY,
    RunnerSpec,
    unpinned_models,
    validate_registry,
)
from .manifest import (
    make_render_manifest,
    render_manifest_from_dict,
    render_manifest_to_dict,
)
from .router import CapabilityRouter
from .schemas import (
    AdapterRef,
    CapabilityRequest,
    ControlRef,
    Job,
    LedgerEvent,
    ModelBinding,
    ModelConfig,
    ProvenanceStub,
    RenderManifest,
    Resolution,
    SamplerConfig,
    SeedBundle,
    VramEnvelope,
)

# NOTE: unlike the frozen prototype, we do NOT call validate_registry() at import
# time. In the dev tree that would raise RegistryError (all seeded models are
# unpinned) and break `import abstract_hugpy_dev.video_intel.studio`. Validation
# stays exported and is invoked explicitly by callers/tests (see docstring).

__all__ = [
    "AdapterKind", "Capability", "ControlKind", "DeterminismClass", "Framework",
    "JobState", "LicenseClass", "PathClass", "Precision", "RiskFlag", "Task",
    "ConfigError", "Err", "ErrorCode", "Ok", "RegistryError", "Result", "StageError",
    "CAPABILITY_TASKS", "MODEL_REGISTRY", "PLANNED_CAPABILITIES", "RUNNER_REGISTRY",
    "RunnerSpec", "unpinned_models", "validate_registry",
    "CapabilityRouter",
    "make_render_manifest", "render_manifest_from_dict", "render_manifest_to_dict",
    "AdapterRef", "CapabilityRequest", "ControlRef", "Job", "LedgerEvent",
    "ModelBinding", "ModelConfig", "ProvenanceStub", "RenderManifest", "Resolution",
    "SamplerConfig", "SeedBundle", "VramEnvelope",
]
