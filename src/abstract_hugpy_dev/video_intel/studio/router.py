"""Capability router (ORCH-3 / INV-8). Resolves a CapabilityRequest into a
concrete ModelBinding given live constraints, or returns an Err *with the reason
each candidate was rejected* - a NO_CAPABLE_MODEL that tells you why beats one
that doesn't (the whole point of not being betrayed later).

The one hard structural rule enforced here is STR-6: locked-identity work plus a
real-time latency budget is a forbidden combination, because causal attention
degrades reference fidelity. The router refuses it rather than silently shipping
a drifting face at low latency.
"""

from __future__ import annotations

from .enums import (
    Capability,
    LICENSE_PREFERENCE,
    PathClass,
    Precision,
    PRECISION_QUALITY,
)
from .errors import Err, ErrorCode, Ok, Result, StageError
from .registry import CAPABILITY_TASKS, MODEL_REGISTRY, runner_for
from .schemas import CapabilityRequest, ModelBinding, ModelConfig


def _pick_precision(
    cfg: ModelConfig, budget_gb: float, min_precision: Precision
) -> Precision | None:
    """Highest-quality precision whose VRAM cost fits the budget AND meets the
    chosen runner's quality floor (FIX-4). A precision below ``min_precision``
    (e.g. INT8 under an FP8 floor) is never a valid selection even if it fits the
    VRAM budget: it would silently ship below the runner's supported quality."""
    floor = PRECISION_QUALITY[min_precision]
    fitting = [p for p in cfg.vram.fits(budget_gb) if PRECISION_QUALITY[p] >= floor]
    if not fitting:
        return None
    return max(fitting, key=lambda p: PRECISION_QUALITY[p])


def _pick_task(cfg: ModelConfig, capability: Capability):
    """First task that both satisfies the capability and has a runner."""
    for task in CAPABILITY_TASKS.get(capability, ()):  # ordered by preference
        if task in cfg.tasks and runner_for(cfg.family, task) is not None:
            return task
    return None


class CapabilityRouter:
    def resolve(self, req: CapabilityRequest) -> Result[ModelBinding, StageError]:
        # STR-6: refuse locked-identity under a real-time latency budget.
        streaming_required = req.latency_budget_ms is not None
        if streaming_required and req.capability in (Capability.ID_LOCK, Capability.KEYFRAME):
            return Err(StageError(
                ErrorCode.CAPABILITY_STREAMING_CONFLICT,
                "locked-identity work cannot run under a real-time latency budget; "
                "causal attention degrades reference fidelity (STR-6). Route this "
                "shot offline, or drop the latency budget.",
                (("capability", req.capability.value),
                 ("latency_budget_ms", str(req.latency_budget_ms))),
            ))

        candidates = [m for m in MODEL_REGISTRY.values()
                      if req.capability in m.capabilities]
        if not candidates:
            return Err(StageError(
                ErrorCode.NO_CAPABLE_MODEL,
                f"no model declares capability {req.capability.value!r}",
            ))

        rejected: list[str] = []
        survivors: list[tuple[ModelConfig, object, Precision]] = []

        for cfg in candidates:
            # path class must match the streaming requirement
            if streaming_required and cfg.path_class != PathClass.STREAMING:
                rejected.append(f"{cfg.model_id}: not a streaming model")
                continue

            # license / commercial gating
            if req.commercial_use:
                # Additive rule: a model routes commercially if it is auto-commercial
                # OR the caller asserts they hold the agreement (license in
                # allowed_licenses). commercial_auto must NOT be nullified.
                allowed = cfg.commercial_auto or cfg.license in req.allowed_licenses
                if not allowed:
                    rejected.append(
                        f"{cfg.model_id}: license {cfg.license.value} not "
                        f"auto-commercial and not in allowed_licenses")
                    continue
            elif req.allowed_licenses and cfg.license not in req.allowed_licenses:
                # FIX-2: the strict whitelist only applies on the non-commercial
                # path. Running it unconditionally turned allowed_licenses into a
                # hard whitelist that nullified commercial_auto above.
                rejected.append(
                    f"{cfg.model_id}: license {cfg.license.value} not in allowed set")
                continue

            # native audio requirement
            if req.require_native_audio and not cfg.native_audio:
                rejected.append(f"{cfg.model_id}: no native audio")
                continue

            # resolution
            if not cfg.supports_resolution(req.target_resolution):
                rejected.append(
                    f"{cfg.model_id}: max res < "
                    f"{req.target_resolution.width}x{req.target_resolution.height}")
                continue

            # frame budget
            if req.min_frames and cfg.max_frames < req.min_frames:
                rejected.append(
                    f"{cfg.model_id}: max_frames {cfg.max_frames} < {req.min_frames}")
                continue

            # capability -> task -> runner
            task = _pick_task(cfg, req.capability)
            if task is None:
                rejected.append(f"{cfg.model_id}: no runnable task for capability")
                continue

            # precision / VRAM fit — bounded below by the chosen runner's floor (FIX-4)
            spec = runner_for(cfg.family, task)
            precision = _pick_precision(cfg, req.vram_budget_gb, spec.min_precision)
            if precision is None:
                if not cfg.vram.fits(req.vram_budget_gb):
                    rejected.append(
                        f"{cfg.model_id}: min {cfg.vram.min_gb():.0f}GB > "
                        f"budget {req.vram_budget_gb:.0f}GB")
                else:
                    # Fits the budget, but only below the runner's precision floor.
                    rejected.append(
                        f"{cfg.model_id}: no precision >= runner floor "
                        f"{spec.min_precision.value} fits budget "
                        f"{req.vram_budget_gb:.0f}GB")
                continue

            survivors.append((cfg, task, precision))

        if not survivors:
            code = ErrorCode.NO_CAPABLE_MODEL
            # Sharpen the code when the rejection was unanimous on one axis.
            if all("GB" in r for r in rejected):
                code = ErrorCode.VRAM_EXCEEDED
            elif all("license" in r for r in rejected):
                code = ErrorCode.LICENSE_VIOLATION
            elif all("max res" in r for r in rejected):
                code = ErrorCode.RESOLUTION_UNSUPPORTED
            return Err(StageError(
                code,
                f"no model satisfied capability {req.capability.value!r} under the "
                f"given constraints",
                tuple(("rejected", r) for r in rejected),
            ))

        cfg, task, precision = max(survivors, key=lambda s: self._score(req, *s))
        return Ok(ModelBinding(
            model_id=cfg.model_id,
            framework=cfg.family,
            task=task,             # type: ignore[arg-type]
            precision=precision,
            path_class=cfg.path_class,
            weight_uri=cfg.weight_uri,
            weight_hash=cfg.weight_hash,
            determinism_class=cfg.default_determinism,   # FIX-3: propagate the class
        ))

    @staticmethod
    def _score(req: CapabilityRequest, cfg: ModelConfig, task, precision: Precision):
        # LAST-RESORT rule: the placeholder synthetic model must NEVER shadow a real
        # generative model. This is the TOP-priority score dimension, so any real
        # survivor (real_first=1) strictly outranks any synthetic one (0) whatever
        # the lower dimensions say. Synthetic wins ONLY when it is the sole survivor
        # (no real model fit — e.g. a sub-GB VRAM budget), which preserves the tiny
        # demo path without letting it steal a genuine binding.
        real_first = 0 if cfg.synthetic else 1
        offline_pref = 0 if req.latency_budget_ms is not None else (
            1 if cfg.path_class == PathClass.OFFLINE else 0)
        framework_pref = 1 if (req.preferred_framework and
                               cfg.family == req.preferred_framework) else 0
        return (
            real_first,
            offline_pref,
            framework_pref,
            LICENSE_PREFERENCE[cfg.license],
            PRECISION_QUALITY[precision],
            cfg.best_native_area(),
            -cfg.vram.min_gb(),   # tie-break: prefer the tighter footprint to pack
        )
