"""Registries over globals (INV-4 / INV-8).

Three tables, all module-level, all populated via `register_*` and frozen in
practice after import:

  MODEL_REGISTRY     : model_id                -> ModelConfig      (the zoo)
  RUNNER_REGISTRY    : (Framework, Task)       -> RunnerSpec       (dispatch)
  CAPABILITY_TASKS   : Capability              -> tuple[Task, ...] (the join)

`validate_registry()` runs at import (see __init__) and proves the join is total:
every capability a model claims is routable, every task has a runner, every
declared Capability is either served or explicitly PLANNED. It collects EVERY
problem and raises once - you see the whole failure set, not just the first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .enums import Capability, Framework, Precision, PRECISION_QUALITY, Task
from .errors import RegistryError
from .schemas import ModelConfig


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    """How to execute one (framework, task). `entrypoint` is wired explicitly -
    a dotted path to the callable, no auto-discovery (explicit env wiring)."""
    framework: Framework
    task: Task
    entrypoint: str
    min_precision: Precision
    supports_streaming: bool = False


MODEL_REGISTRY: dict[str, ModelConfig] = {}
RUNNER_REGISTRY: dict[tuple[Framework, Task], RunnerSpec] = {}
CAPABILITY_TASKS: dict[Capability, tuple[Task, ...]] = {}

# Capabilities intentionally not backed by a generative model (orchestration
# stages, e.g. multi-shot assembly). Declared so validate_registry() does not
# flag them as orphans, but tracked loudly rather than silently ignored.
PLANNED_CAPABILITIES: frozenset[Capability] = frozenset({Capability.ASSEMBLE})


def register_model(cfg: ModelConfig) -> None:
    if cfg.model_id in MODEL_REGISTRY:
        raise RegistryError(f"duplicate model_id: {cfg.model_id!r}")
    MODEL_REGISTRY[cfg.model_id] = cfg


def register_runner(spec: RunnerSpec) -> None:
    key = (spec.framework, spec.task)
    if key in RUNNER_REGISTRY:
        raise RegistryError(f"duplicate runner: {key}")
    RUNNER_REGISTRY[key] = spec


def set_capability_tasks(mapping: dict[Capability, tuple[Task, ...]]) -> None:
    CAPABILITY_TASKS.clear()
    CAPABILITY_TASKS.update(mapping)


def runner_for(framework: Framework, task: Task) -> RunnerSpec | None:
    return RUNNER_REGISTRY.get((framework, task))


def _allow_unpinned() -> bool:
    # Explicit env gate, loud by default. Seed/dev may run unpinned; production
    # must pin weight hashes or set this deliberately.
    return os.environ.get("STUDIO_ALLOW_UNPINNED") == "1"


def validate_registry() -> None:
    """Fail-loud, comprehensive. Collects every problem, then raises once."""
    problems: list[str] = []

    # Every task referenced anywhere must have a runner.
    tasks_with_runner = {task for (_fw, task) in RUNNER_REGISTRY}

    for cap, tasks in CAPABILITY_TASKS.items():
        if not tasks and cap not in PLANNED_CAPABILITIES:
            problems.append(f"capability {cap.value!r} maps to no tasks and is not PLANNED")
        for task in tasks:
            if task not in tasks_with_runner:
                problems.append(
                    f"capability {cap.value!r} references task {task.value!r} "
                    f"with no runner registered for any framework"
                )

    served_caps: set[Capability] = set()

    for cfg in MODEL_REGISTRY.values():
        mid = cfg.model_id

        if not cfg.capabilities:
            problems.append(f"{mid}: declares no capabilities")
        if not cfg.tasks:
            problems.append(f"{mid}: declares no tasks")
        if not cfg.resolutions:
            problems.append(f"{mid}: declares no resolutions")

        # weight pinning (INV-1/INV-2)
        if cfg.weight_hash is None and not cfg.unpinned:
            problems.append(f"{mid}: no weight_hash and not marked unpinned")
        if cfg.weight_hash is None and cfg.unpinned and not _allow_unpinned():
            problems.append(
                f"{mid}: unpinned weights but STUDIO_ALLOW_UNPINNED != '1' "
                f"(pin the weight hash or set the env gate deliberately)"
            )

        # every (family, task) the model can run must have a runner
        for task in cfg.tasks:
            spec = runner_for(cfg.family, task)
            if spec is None:
                problems.append(
                    f"{mid}: no runner for ({cfg.family.value}, {task.value})"
                )
                continue
            # FIX-4: the model must expose >=1 VRAM precision that meets this
            # runner's min_precision floor, else the router can never bind it at a
            # supported quality (fail loud at boot, not at dispatch).
            floor = PRECISION_QUALITY[spec.min_precision]
            if not any(PRECISION_QUALITY[p] >= floor
                       for p, _gb in cfg.vram.per_precision):
                problems.append(
                    f"{mid}: no VRAM precision >= runner floor "
                    f"{spec.min_precision.value} for ({cfg.family.value}, {task.value})"
                )

        # every capability the model claims must be routable via one of its tasks
        for cap in cfg.capabilities:
            served_caps.add(cap)
            candidate_tasks = CAPABILITY_TASKS.get(cap)
            if candidate_tasks is None:
                problems.append(f"{mid}: claims capability {cap.value!r} absent from CAPABILITY_TASKS")
                continue
            if not (set(candidate_tasks) & set(cfg.tasks)):
                problems.append(
                    f"{mid}: claims capability {cap.value!r} but has none of its "
                    f"satisfying tasks {[t.value for t in candidate_tasks]}"
                )

        # streaming models must be backed by streaming-capable runners (STR path)
        if cfg.path_class.value == "streaming":
            for task in cfg.tasks:
                spec = runner_for(cfg.family, task)
                if spec is not None and not spec.supports_streaming:
                    problems.append(
                        f"{mid}: path_class=streaming but runner "
                        f"({cfg.family.value}, {task.value}) is not streaming-capable"
                    )

    # every declared Capability must be served by a model or explicitly PLANNED
    for cap in Capability:
        if cap not in served_caps and cap not in PLANNED_CAPABILITIES:
            problems.append(f"capability {cap.value!r} is served by no model and not PLANNED")

    if problems:
        raise RegistryError(
            "registry validation failed with "
            f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems)
        )


def unpinned_models() -> tuple[str, ...]:
    """Reporting hook: which models still need a pinned weight hash."""
    return tuple(sorted(m.model_id for m in MODEL_REGISTRY.values() if m.weight_hash is None))
