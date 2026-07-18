"""Shared test-isolation helper for the GPU worker registry (WorkerStore) and
its assignment-memory sidecar.

THE LANDMINE (k2 incident, 2026-07-18): WorkerStore's default path resolves
through ``schemas.settings.manifest_path`` -> ``MODELS_DICT_PATH`` ->
``PROJECTS_HOME``, all computed ONCE at import time via
``abstract_essentials.get_env_value()`` -- a ``.env``-FILE reader (cwd /
``$HOME`` / ``~/.envy_all``) that NEVER consults ``os.environ`` (see
``imports/src/constants/constants.py`` and the ``abstract_essentials.utils``
docstring: "Searches, in order: the supplied path, the current working
directory, the user's home directory, and ~/.envy_all"). So the
``os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(...)`` idiom used to isolate
most of this suite is a NO-OP for anything keyed off ``settings.manifest_path``
-- it silently falls through to the REAL ``/mnt/llm_storage/projects/``
directory this box's live fleet registry lives in.

Two DISTINCT things share this landmine and both must be redirected:

  1. ``WorkerStore``'s OWN path (``workers.json``) -- already has a ``path=``
     constructor seam; most of this suite uses it correctly.
  2. The assignment-memory SIDECAR (``worker_assignments.json``, written by
     ``_remember_assignments`` / read by ``_load_assign_memory`` in
     ``workers.py``) -- this does NOT go through ``WorkerStore.__init__`` at
     all. It independently derives its directory from the SAME frozen
     ``schemas.settings`` singleton (``os.path.dirname(settings.manifest_path)``)
     every time it's called, so a bare ``WorkerStore(path=tmp)`` isolates the
     worker rows but NOT this sidecar. It fires on ``.register()`` of an
     ALREADY-registered worker_id (re-register with models), and
     unconditionally on every ``.assign_model()`` / ``.unassign_model()`` --
     even when the target worker id doesn't exist in the (isolated) store, the
     underlying ``_transaction()`` still opens/locks/rewrites whatever real
     file ``settings.manifest_path`` points at. A FIRST-time
     ``.register(worker_id=...)`` also READS this sidecar (harmless on its
     own, but still touches live storage from a test).

Proven incident: k2's first ``test_block_propagation.py`` run registered a
real ``wk-prop`` row into the LIVE fleet registry the operator's console
shows (cleaned up since). ``test_model_block.py`` reaches the real
module-level ``assign_model()`` (unstubbed) via a Flask route after an
unblock, and ``test_storage_budget.py`` calls ``store.unassign_model(...)``
directly -- both hit the sidecar WRITE path for real, not just in theory.

Use ``isolated_worker_store()`` for a one-off, standalone-script-style test
(the tmp redirect is never restored -- fine, the process exits right after)
or ``swap_worker_store()`` as a context manager when the module-level
``W.worker_store`` singleton itself needs to be swapped (e.g. a test drives
Flask routes / the module-level ``assign_model``/``unassign_model``/
``grant_model`` wrapper functions) and/or when running under pytest inside a
shared process with other test files, where leaving global state redirected
forever is sloppier than it needs to be.

Neither helper touches PROJECTS_HOME/os.environ at all -- the whole point is
that env var does NOT reach these paths. Do not rely on it for WorkerStore
isolation; use these helpers instead.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")


def _isolated_paths(prefix: str) -> Tuple[str, str, str]:
    tmp = tempfile.mkdtemp(prefix=prefix)
    return tmp, os.path.join(tmp, "workers.json"), os.path.join(tmp, "model_manifest.json")


def isolated_worker_store(prefix: str = "hugpy-test-workers-"):
    """Build a fresh, tmpdir-backed ``WorkerStore`` AND redirect the
    assignment-memory sidecar into the SAME tmpdir, so ``.register()`` /
    ``.assign_model()`` / ``.unassign_model()`` on the returned store can
    never reach the live ``/mnt/llm_storage/projects/`` registry.

    ``schemas.settings`` is a process-wide singleton shared by every module
    that does ``from .schemas import settings`` (workers.py,
    enrollment_tokens.py, discord_bindings.py, phone_brick_store.py) -- this
    redirects it for the life of the process. That's exactly right for a
    standalone-script test (``python tests/test_x.py``, one process, exits
    right after) but NOT auto-restored; if you need restoration (pytest,
    multiple test files sharing one process, or driving the module-level
    ``W.worker_store`` singleton through routes), use ``swap_worker_store``
    instead.

    Returns ``(store, tmp_dir)``.
    """
    tmp, workers_path, manifest_path = _isolated_paths(prefix)
    W.settings.manifest_path = manifest_path
    return W.WorkerStore(path=workers_path), tmp


@contextmanager
def swap_worker_store(prefix: str = "hugpy-test-workers-") -> Iterator["W.WorkerStore"]:
    """Context manager: isolate AND swap the MODULE-LEVEL ``W.worker_store``
    singleton -- what Flask routes and the module-level ``assign_model`` /
    ``unassign_model`` / ``grant_model`` / ``heartbeat`` wrapper functions
    call through -- plus the assignment-memory sidecar. Restores both on
    exit, so it's safe to use repeatedly in one process (pytest running many
    test files, or several ``with`` blocks in one file).

    Yields the isolated store (also reachable as ``W.worker_store`` for the
    duration of the ``with`` block, which is what makes route-driven tests
    safe).
    """
    tmp, workers_path, manifest_path = _isolated_paths(prefix)
    orig_store = W.worker_store
    orig_manifest_path = W.settings.manifest_path
    W.settings.manifest_path = manifest_path
    W.worker_store = W.WorkerStore(path=workers_path)
    try:
        yield W.worker_store
    finally:
        W.worker_store = orig_store
        W.settings.manifest_path = orig_manifest_path
