"""Studio runners — the executors a ModelBinding dispatches to (INV-8).

Each ``RunnerSpec.entrypoint`` in the zoo is a dotted path into this subpackage.
Only the SYNTHETIC runner is wired in this slice (P0-B1); the real-model runner
modules (wan, ltx, hunyuan, ...) are declared in the registry but intentionally
UNWIRED until their model bets land. ``validate_registry()`` only checks a runner
is *registered*, not importable, so this partial state is by design.

Importing this package is cheap: it does NOT eagerly import the runner modules
(which pull PIL/numpy). Callers import a concrete runner by name, e.g.
``from ...studio.runners.synthetic import run_synthetic_i2v``.
"""

from __future__ import annotations
