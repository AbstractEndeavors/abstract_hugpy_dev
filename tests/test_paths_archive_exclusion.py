"""Archive-exclusion regression for resolve_model_dir() / route_destination().

Confirmed live bug: candidate_model_dirs() legitimately SURFACES a model's
entry under <root>/models/_archive/dedupes/... (reconcile's archive/de-dupe
area — reconcile IS the reconcile survey set and needs to see these), but the
weight file there is often a SYMLINK back into /checkpoints. resolve_model_dir()
must NEVER hand that path out as a live serve/transfer source, even when it is
listed first (e.g. because the registry's ``folder`` field still points at it)
and exists on disk ahead of the real flat dir.

Runs like the other tests here: venv/bin/python tests/test_paths_archive_exclusion.py
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

paths = importlib.import_module("abstract_hugpy_dev.imports.src.constants.paths")

ok = 0
def check(name, cond):
    global ok
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise SystemExit(f"CHECK FAILED: {name}")
    ok += 1


def wfile(path, size=1024 * 1024 + 1):
    # model_looks_downloaded's generic (no config.json) branch requires each
    # *.safetensors file to exceed 1 MiB to distinguish a real weight from a
    # Git-LFS pointer stub; default to just over that floor so "complete"
    # checks in this test genuinely exercise the completeness gate.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)


tmp = tempfile.mkdtemp(prefix="hugpy-archive-exclusion-")
root = tmp

# ---------------------------------------------------------------------------
# Case 1: mirrors the confirmed live bug — a registry entry whose ``folder``
# points into models/_archive/dedupes/..., AND the real flat dir also exists.
# resolve_model_dir(require_complete=False) must pick the REAL dir, not the
# archive one, even though the archive candidate is listed first (candidate 0)
# by candidate_model_dirs (folder-recorded dir wins the survey-list race).
# ---------------------------------------------------------------------------
archive_dir = os.path.join(root, "models", "_archive", "dedupes", "misc", "comfy", "foo")
real_dir = os.path.join(root, "models", "misc", "comfy", "foo")
wfile(os.path.join(archive_dir, "model.safetensors"))
wfile(os.path.join(real_dir, "model.safetensors"))

model = {
    "hub_id": "comfy/foo",
    "framework": "misc",
    "primary_task": "misc",
    "tasks": ["misc"],
    "folder": "_archive/dedupes/misc/comfy/foo",
}

cands = paths.candidate_model_dirs(model, root)
check("sanity: candidate_model_dirs still surfaces the archive dir (survey set unchanged)",
      archive_dir in cands)
check("sanity: archive candidate 0 is listed before the real dir (mirrors the live bug)",
      cands.index(archive_dir) < cands.index(real_dir))

resolved = paths.resolve_model_dir(model, root, require_complete=False)
check("resolve_model_dir (require_complete=False) returns the REAL dir, not the archive copy",
      resolved == real_dir)

resolved_complete = paths.resolve_model_dir(model, root, require_complete=True)
check("resolve_model_dir (require_complete=True) also returns the REAL dir",
      resolved_complete == real_dir)

routed = paths.route_destination(model, root)
check("route_destination returns the REAL dir for the archived-folder model",
      routed == real_dir)

# ---------------------------------------------------------------------------
# Case 2: ONLY the archive copy exists on disk (no live dir anywhere). Even
# though it is the sole EXISTING candidate, resolve_model_dir must not return
# it — it should fall through past it to flat_destination() (the safe write
# target for "act on this" callers), never surfacing the archived symlink copy.
# ---------------------------------------------------------------------------
shutil.rmtree(real_dir)
only_archive_model = {
    "hub_id": "comfy/bar",
    "framework": "misc",
    "primary_task": "misc",
    "tasks": ["misc"],
    "folder": "_archive/dedupes/misc/comfy/bar",
}
only_archive_dir = os.path.join(root, "models", "_archive", "dedupes", "misc", "comfy", "bar")
wfile(os.path.join(only_archive_dir, "model.safetensors"))

resolved_only_archive = paths.resolve_model_dir(only_archive_model, root, require_complete=False)
expected_flat = paths.flat_destination(only_archive_model, root)
check("archive-only case: does NOT return the archive path",
      resolved_only_archive != only_archive_dir)
check("archive-only case: falls through to flat_destination() instead",
      resolved_only_archive == expected_flat)

resolved_only_archive_strict = paths.resolve_model_dir(only_archive_model, root, require_complete=True)
check("archive-only case (require_complete=True): returns None (no complete archive fallback)",
      resolved_only_archive_strict is None)

# ---------------------------------------------------------------------------
# Case 3: a plain, non-archive model is completely unaffected — byte-identical
# behavior to before this change.
# ---------------------------------------------------------------------------
plain_dir = os.path.join(root, "models", "misc", "comfy", "baz")
wfile(os.path.join(plain_dir, "model.safetensors"))
plain_model = {
    "hub_id": "comfy/baz",
    "framework": "misc",
    "primary_task": "misc",
    "tasks": ["misc"],
}
resolved_plain = paths.resolve_model_dir(plain_model, root, require_complete=False)
check("non-archive model resolves to its real dir, unchanged",
      resolved_plain == plain_dir)
routed_plain = paths.route_destination(plain_model, root)
check("route_destination unchanged for a non-archive model",
      routed_plain == plain_dir)

# ---------------------------------------------------------------------------
# _is_archived_path helper: component match, not substring match.
# ---------------------------------------------------------------------------
check("_is_archived_path: true for a real _archive component",
      paths._is_archived_path("/root/models/_archive/dedupes/misc/comfy/foo"))
check("_is_archived_path: false for a repo merely named with _archive as a substring",
      not paths._is_archived_path("/root/models/misc/comfy/my_archive_tool"))
check("_is_archived_path: false for a clean path",
      not paths._is_archived_path("/root/models/misc/comfy/foo"))
check("_is_archived_path: false for empty/None-ish input",
      not paths._is_archived_path(""))

shutil.rmtree(tmp, ignore_errors=True)
print(f"\nALL {ok} CHECKS PASSED")
