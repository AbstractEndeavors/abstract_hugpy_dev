"""GET /video/presets + POST /video/presets/<id>/apply — route contract.

Verifies the video-preset HTTP surface WITHOUT a live worker/catalog touch:
  * GET  /video/presets            -> {"presets":[...]} carrying (at least) the
                                       known seed presets, each in the pinned
                                       wire shape (incl. the advisory sampler);
  * POST /video/presets/<bad>/apply-> 404 (get_preset -> None short-circuits
                                       before any catalog/worker work).

The known-id checks are subset assertions (``<=``) so that adding presets to the
registry does not break this contract test — only removing/renaming a known one
or dropping a pinned wire key does.

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
# The known preset ids -> catalog model_key (the 3 seeds + the 5 ComfyUI Field
# Guide presets). Subset-checked, so future additions won't break this test.
_EXPECTED = {
    "realistic-edit-chain": "a3527183~Qwen-Image-Edit-2509",
    "realistic-img2img": "comfy-juggernautxl-ragnarok",
    "fast-draft": "sdxl-turbo",
    "photoreal-portrait-sd15": "comfy-epicrealism-naturalsinrc1vae",
    "photoreal-sdxl": "comfy-juggernautxl-ragnarok",
    "anime-stylized": "comfy-neverendingdreamned-v122bakedvae",
    "painterly-art": "comfy-dreamshaper-8",
    "sdxl-lightning": "comfy-dreamshaperxl-lightningdpmsde",
}
# The known modes (subset-checked alongside _EXPECTED).
_EXPECTED_MODES = {
    "realistic-edit-chain": "edit-chain",
    "realistic-img2img": "img2img",
    "fast-draft": "text-to-image",
    "photoreal-portrait-sd15": "text-to-image",
    "photoreal-sdxl": "text-to-image",
    "anime-stylized": "text-to-image",
    "painterly-art": "text-to-image",
    "sdxl-lightning": "text-to-image",
}


def test_get_presets_contract_shape():
    r = client.get("/video/presets")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert isinstance(body, dict) and "presets" in body
    presets = body["presets"]
    assert isinstance(presets, list) and presets, presets

    by_id = {p["id"]: p for p in presets}
    # every known preset id is present (subset — additions are fine)
    assert set(_EXPECTED) <= set(by_id), set(by_id)

    for pid, model_key in _EXPECTED.items():
        p = by_id[pid]
        # every pinned top-level key present
        assert _TOP_KEYS <= set(p), (pid, set(p))
        assert p["model_key"] == model_key, (pid, p["model_key"])
        assert p["recommended"] == "gpu", (pid, p["recommended"])
        # defaults sub-object carries exactly the pinned keys
        assert _DEFAULT_KEYS <= set(p["defaults"]), (pid, set(p["defaults"]))

    # the advisory sampler field is present in every preset's wire shape
    for p in presets:
        assert "sampler" in p, (p.get("id"), set(p))


def test_get_presets_modes():
    presets = client.get("/video/presets").get_json()["presets"]
    modes = {p["id"]: p["mode"] for p in presets}
    for pid, mode in _EXPECTED_MODES.items():
        assert modes.get(pid) == mode, (pid, modes.get(pid))


def test_apply_unknown_preset_404():
    r = client.post("/video/presets/does-not-exist/apply")
    assert r.status_code == 404, r.status_code
    body = r.get_json()
    assert body.get("ok") is False, body
    assert body.get("error", {}).get("code") == "UnknownPreset", body


# --------------------------------------------------------------------------- #
# MOVIE templates — the img2img-drift "street-walk" preset (strength+negative)
# --------------------------------------------------------------------------- #
# The empirically-tuned street-walk template is the first movie preset to carry a
# strength + negative; assert both reach the directly-POSTable generate_movie body
# (apply().request) and that its 4-goal timeline tiles [0, 12). storm-front is the
# regression: an existing preset still constructs + applies unchanged.
_STREET_WALK_NEGATIVE = ("different person, face change, identity change, "
                         "deformed face, extra limbs, warped body, morphing, blurry")


def _assert_tiles(goals, total):
    """The goal timeline is contiguous, non-overlapping and tiles [0, total)."""
    assert goals and goals[0]["start_frame"] == 0, goals
    cursor = 0
    for g in goals:
        assert g["start_frame"] == cursor, (cursor, g)
        assert g["end_frame"] > g["start_frame"], g
        cursor = g["end_frame"]
    assert cursor == total, (cursor, total)


def test_movie_presets_list_includes_street_walk():
    r = client.get("/movie/presets")
    assert r.status_code == 200, r.status_code
    presets = r.get_json()["presets"]
    by_id = {p["id"]: p for p in presets}
    assert "street-walk" in by_id, sorted(by_id)
    sw = by_id["street-walk"]
    assert sw["model_key"] == "comfy-juggernautxl-ragnarok", sw["model_key"]
    # strength + negative are surfaced at the top level AND in the settings bundle
    assert sw["strength"] == 0.45, sw["strength"]
    assert sw["negative"] == _STREET_WALK_NEGATIVE, sw["negative"]
    assert sw["settings"]["strength"] == 0.45, sw["settings"]
    assert sw["settings"]["negative"] == _STREET_WALK_NEGATIVE, sw["settings"]
    _assert_tiles(sw["goals"], 12)


def test_movie_preset_street_walk_apply_carries_strength_negative():
    r = client.post("/movie/presets/street-walk/apply")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body.get("ok") is True, body
    assert body.get("id") == "street-walk", body
    req = body["request"]
    # the POSTable generate_movie body MUST carry strength + negative
    assert req["strength"] == 0.45, req["strength"]
    assert req["negative"] == _STREET_WALK_NEGATIVE, req["negative"]
    assert req["model_id"] == "comfy-juggernautxl-ragnarok", req["model_id"]
    # 4 contiguous goals tiling [0, 12)
    assert len(req["goals"]) == 4, req["goals"]
    _assert_tiles(req["goals"], 12)


def test_movie_preset_storm_front_regression():
    """An existing preset (no strength/negative set) still applies — defaults ride
    through: strength None, negative "" — proving the 6 seeds are unaffected."""
    r = client.post("/movie/presets/storm-front/apply")
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body.get("ok") is True, body
    req = body["request"]
    assert req["model_id"] == "comfy-juggernautxl-ragnarok", req["model_id"]
    assert req["strength"] is None, req["strength"]
    assert req["negative"] == "", req["negative"]
    _assert_tiles(req["goals"], 12)


if __name__ == "__main__":  # allow the script-style run the sibling tests use
    test_get_presets_contract_shape()
    test_get_presets_modes()
    test_apply_unknown_preset_404()
    test_movie_presets_list_includes_street_walk()
    test_movie_preset_street_walk_apply_carries_strength_negative()
    test_movie_preset_storm_front_regression()
    print("all video-preset route checks passed")
