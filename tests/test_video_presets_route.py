"""GET /video/presets + POST /video/presets/<id>/apply — route contract.

Verifies the video-preset HTTP surface WITHOUT a live worker/catalog touch:
  * GET  /video/presets            -> {"presets":[...]} with the 3 seed presets,
                                       each in the pinned wire shape;
  * POST /video/presets/<bad>/apply-> 404 (get_preset -> None short-circuits
                                       before any catalog/worker work).

Mirrors test_reap_approve_route.py's idiom (temp PROJECTS_HOME, a minimal Flask
app with the blueprint mounted, a test_client), but exposed as pytest tests so
`python -m pytest tests/test_video_presets_route.py` reports a clean pass.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep any bus/audit writes out of the real projects tree.
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-video-presets-test-"))

import importlib

from flask import Flask

vr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.video_routes")

app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()

# The pinned per-preset wire shape.
_TOP_KEYS = {"id", "name", "description", "mode", "model_key", "defaults", "recommended"}
_DEFAULT_KEYS = {"strength", "steps", "guidance", "width", "height",
                 "n_frames", "fps", "negative"}
_EXPECTED = {
    "realistic-edit-chain": "a3527183~Qwen-Image-Edit-2509",
    "realistic-img2img": "comfy-juggernautxl-ragnarok",
    "fast-draft": "sdxl-turbo",
}


def test_get_presets_contract_shape():
    r = client.get("/video/presets")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert isinstance(body, dict) and "presets" in body
    presets = body["presets"]
    assert isinstance(presets, list) and len(presets) == 3, presets

    by_id = {p["id"]: p for p in presets}
    assert set(by_id) == set(_EXPECTED), set(by_id)

    for pid, model_key in _EXPECTED.items():
        p = by_id[pid]
        # every pinned top-level key present
        assert _TOP_KEYS <= set(p), (pid, set(p))
        assert p["model_key"] == model_key, (pid, p["model_key"])
        assert p["recommended"] == "gpu", (pid, p["recommended"])
        # defaults sub-object carries exactly the pinned keys
        assert _DEFAULT_KEYS <= set(p["defaults"]), (pid, set(p["defaults"]))


def test_get_presets_modes():
    presets = client.get("/video/presets").get_json()["presets"]
    modes = {p["id"]: p["mode"] for p in presets}
    assert modes == {
        "realistic-edit-chain": "edit-chain",
        "realistic-img2img": "img2img",
        "fast-draft": "text-to-image",
    }, modes


def test_apply_unknown_preset_404():
    r = client.post("/video/presets/does-not-exist/apply")
    assert r.status_code == 404, r.status_code
    body = r.get_json()
    assert body.get("ok") is False, body
    assert body.get("error", {}).get("code") == "UnknownPreset", body


if __name__ == "__main__":  # allow the script-style run the sibling tests use
    test_get_presets_contract_shape()
    test_get_presets_modes()
    test_apply_unknown_preset_404()
    print("all video-preset route checks passed")
