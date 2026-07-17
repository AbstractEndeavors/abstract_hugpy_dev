"""Worker-side orphan (unattributed-on-disk) scan (2026-07-17 addendum).

computron held 5.7G of a STALLED partial download (Qwen2.5-VL-3B, all *.part
files) for a model NOT allocated to it — and it appeared NOWHERE in the UI
because the reaper survey only looks at KNOWN/assigned keys. _orphan_scan walks
the store root for exactly this residue: leftover model dirs + stalled *.part
sets that match no current model. This locks its behavior against a temp store.

Runs under pytest: venv/bin/python -m pytest tests/test_orphan_scan.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import the agent module; the flat store layout is family/owner/repo under root.
from abstract_hugpy_dev.worker_agent import agent as A          # noqa: E402


class _State:
    assigned_models = []


def _write(path, size=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


def _make_store():
    root = tempfile.mkdtemp()
    tr = os.path.join(root, "transformers")
    # ATTRIBUTED + resident: a real model dir for an assigned model.
    _write(os.path.join(tr, "google", "flan-t5-xl", "model.safetensors"), 4000)
    _write(os.path.join(tr, "google", "flan-t5-xl", "config.json"), 100)
    # ORPHAN 1: stalled partial — only .part files, no real weights.
    _write(os.path.join(tr, "Qwen", "Qwen2.5-VL-3B", "model.safetensors.part"), 5700)
    _write(os.path.join(tr, "Qwen", "Qwen2.5-VL-3B", "config.json.part"), 50)
    # ORPHAN 2: a completed but unassigned leftover model dir.
    _write(os.path.join(tr, "old", "leftover", "model.safetensors"), 900)
    _write(os.path.join(tr, "old", "leftover", "config.json"), 100)
    return root


def _run(root, known_hub_ids):
    # Point the scan's root resolver at our temp store.
    orig = A._models_store_root
    A._models_store_root = lambda: root
    A._ORPHAN_CACHE["value"] = None      # bypass the TTL cache between runs
    try:
        return A._orphan_scan(_State(), set(known_hub_ids))
    finally:
        A._models_store_root = orig
        A._ORPHAN_CACHE["value"] = None


def test_stalled_part_and_leftover_are_orphaned():
    root = _make_store()
    # flan-t5-xl IS known/assigned; the other two are not.
    out = _run(root, {"google/flan-t5-xl", "flan-t5-xl"})
    paths = {i["path"]: i for i in out["items"]}
    # the assigned model is NOT flagged
    assert not any("flan-t5-xl" in p for p in paths)
    # both orphans ARE flagged, with the right kind
    qwen = next((v for p, v in paths.items() if "Qwen2.5-VL-3B" in p), None)
    leftover = next((v for p, v in paths.items() if "leftover" in p), None)
    assert qwen is not None and qwen["kind"] == "partial"
    assert leftover is not None and leftover["kind"] == "stale-dir"
    assert out["count"] == 2
    assert out["bytes"] == qwen["bytes"] + leftover["bytes"]


def test_all_known_no_orphans():
    root = _make_store()
    out = _run(root, {"google/flan-t5-xl", "flan-t5-xl",
                      "Qwen/Qwen2.5-VL-3B", "Qwen2.5-VL-3B",
                      "old/leftover", "leftover"})
    assert out["count"] == 0 and out["bytes"] == 0


def test_no_store_root_is_safe():
    orig = A._models_store_root
    A._models_store_root = lambda: None
    A._ORPHAN_CACHE["value"] = None
    try:
        out = A._orphan_scan(_State(), set())
        assert out == {"items": [], "bytes": 0, "count": 0}
    finally:
        A._models_store_root = orig
        A._ORPHAN_CACHE["value"] = None
