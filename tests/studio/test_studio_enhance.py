"""slice b — INTERP + UPRES enhancement capabilities (§6).

Locks the frame-interpolation and spatial-upscale runner slices as executable
checks, in the same script style as ``test_studio_vace.py`` (plain python,
``__main__`` guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any
check FAILED, every check independently run so a failing one never masks the rest).
pytest is NOT installed in this venv, so there are no fixtures.

Invariants under test:
  * The FFMPEG last-resort runners are REAL on this GPU-less box: minterpolate
    (motion-compensated) interpolation to a target fps, and lanczos spatial upscale
    to a target resolution — content-addressed, atomic, resume-on-hash.
  * The router ranks the PREMIUM models (rife-practical / ltxv-spatial-upscaler)
    strictly above the ffmpeg last-resort (synthetic=True): ffmpeg binds only when
    no premium model fits the budget; a fitting premium model outranks it.
  * The premium runners degrade gracefully (errors-as-data) on this box:
    rife -> DEPS_MISSING (Practical-RIFE not vendored), ltx -> WEIGHTS_MISSING (HF
    license-gated 401 weights not staged). Never a raise.
  * A source-less enhance is a SPEC error (SOURCE_MISSING), checked BEFORE the box
    capability, for both capabilities.
  * The prompt rides in the content_hash for provenance but NEVER alters output bits
    (deterministic ffmpeg transform, -threads 1).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_enhance.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

logging.disable(logging.INFO)  # silence the models_config registry chatter

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT  # noqa: E402
from abstract_hugpy_dev.video_intel import media_bus  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.enums import (  # noqa: E402
    Capability,
    Framework,
    Precision,
    Task,
)
from abstract_hugpy_dev.video_intel.studio.env import StudioEnv  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.produce import produce_clip  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.registry import (  # noqa: E402
    runner_for,
    validate_registry,
)
from abstract_hugpy_dev.video_intel.studio.router import CapabilityRouter  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.schemas import (  # noqa: E402
    CapabilityRequest,
    Resolution,
)

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

# The tiny source clip: 8 frames @ 8fps, 160x90.
_SRC_W, _SRC_H, _SRC_FPS, _SRC_DUR = 160, 90, 8, 1
_SRC_FRAMES = _SRC_FPS * _SRC_DUR  # 8

# INTERP target: same geometry, DOUBLE the fps (8 -> 16). UPRES target: 2x spatial.
R_INTERP = Resolution(_SRC_W, _SRC_H, 16)
R_UPRES = Resolution(_SRC_W * 2, _SRC_H * 2, _SRC_FPS)  # 320x180 @ 8

# --------------------------------------------------------------------------- #
# Isolation: point the media bus at a TEMP DB so the route enqueue never touches
# the real media_jobs.db. A real work dir UNDER DEFAULT_ROOT holds the tiny mp4
# fixture (the route jails source_video to it).
# --------------------------------------------------------------------------- #
_TMP_DB = tempfile.mkstemp(prefix="enhance-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False  # force _ensure_db to re-init against the temp DB
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

_WORK = tempfile.mkdtemp(prefix="studio-enhance-", dir=DEFAULT_ROOT)  # inside the jail
_SRC_MP4 = os.path.join(_WORK, "source_clip.mp4")

vr = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _make_src(path: str) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={_SRC_DUR}:size={_SRC_W}x{_SRC_H}:rate={_SRC_FPS}",
         "-pix_fmt", "yuv420p", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _probe(video: str) -> tuple[int, int, float, int]:
    """(width, height, fps, nb_frames) of a clip via ffprobe (decode-counted frames)."""
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-of", "json", video],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    s = json.loads(r.stdout)["streams"][0]
    rate = str(s.get("r_frame_rate") or "0")
    fps = (float(rate.split("/")[0]) / float(rate.split("/")[1])) if "/" in rate else float(rate)
    return int(s["width"]), int(s["height"]), fps, int(s.get("nb_read_frames") or 0)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_fixtures() -> None:
    if _FFMPEG:
        _make_src(_SRC_MP4)


def _teardown_fixtures() -> None:
    shutil.rmtree(_WORK, ignore_errors=True)
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


def _studio_env(master_fps: int = 12) -> StudioEnv:
    return StudioEnv(
        output_root="/out", weights_root="/weights", manifest_root="/manifests",
        master_colorspace="rec709", master_fps=master_fps, max_vram_gb=24.0,
        loudness_target_lufs=-14.0, allow_unpinned=True)


def _req(capability: Capability, res: Resolution, budget_gb: float) -> CapabilityRequest:
    return CapabilityRequest(
        capability=capability, target_resolution=res, vram_budget_gb=budget_gb)


# --------------------------------------------------------------------------- #
# [1] Router: interp @ 0.5GB -> ffmpeg-minterpolate (framework=ffmpeg, task=interpolate).
#     No premium interp model (rife 3GB+) fits, so the last resort binds.
# --------------------------------------------------------------------------- #
def test_router_interp_tiny_binds_ffmpeg():
    b = CapabilityRouter().resolve(_req(Capability.INTERP, R_INTERP, 0.5)).unwrap()
    assert b.model_id == "ffmpeg-minterpolate", b.model_id
    assert b.framework == Framework.FFMPEG, b.framework
    assert b.task == Task.INTERPOLATE, b.task


# --------------------------------------------------------------------------- #
# [2] Router: interp @ 6GB -> rife-practical. The REAL premium model fits and
#     strictly OUTRANKS the ffmpeg last-resort (synthetic=True) via real_first.
# --------------------------------------------------------------------------- #
def test_router_interp_big_binds_rife_real_outranks_ffmpeg():
    b = CapabilityRouter().resolve(_req(Capability.INTERP, R_INTERP, 6.0)).unwrap()
    assert b.model_id == "rife-practical", (
        f"a fitting premium interp model must outrank the ffmpeg last-resort; got {b.model_id}")
    assert b.framework == Framework.RIFE, b.framework
    assert b.task == Task.INTERPOLATE, b.task


# --------------------------------------------------------------------------- #
# [3] Router: upres @ 0.5GB -> ffmpeg-lanczos-upscale. No premium upres model
#     (ltx 8GB+) fits, so the last resort binds.
# --------------------------------------------------------------------------- #
def test_router_upres_tiny_binds_ffmpeg():
    b = CapabilityRouter().resolve(_req(Capability.UPRES, R_UPRES, 0.5)).unwrap()
    assert b.model_id == "ffmpeg-lanczos-upscale", b.model_id
    assert b.framework == Framework.FFMPEG, b.framework
    assert b.task == Task.UPSCALE, b.task


# --------------------------------------------------------------------------- #
# [4] Router: upres @ 12GB -> ltxv-spatial-upscaler-0.9.7. The REAL premium model
#     fits (FP16 @ 12GB) and outranks the ffmpeg last-resort.
# --------------------------------------------------------------------------- #
def test_router_upres_big_binds_ltx_real_outranks_ffmpeg():
    b = CapabilityRouter().resolve(_req(Capability.UPRES, R_UPRES, 12.0)).unwrap()
    assert b.model_id == "ltxv-spatial-upscaler-0.9.7", (
        f"a fitting premium upres model must outrank the ffmpeg last-resort; got {b.model_id}")
    assert b.framework == Framework.LTX, b.framework
    assert b.task == Task.UPSCALE, b.task


# --------------------------------------------------------------------------- #
# [5] Registry: validate_registry() passes AND all four enhancement entrypoints are
#     wired at their real modules (ffmpeg x2 new; rife + ltx RE-POINTED off the old
#     unwired runners.rife:interp / runners.ltx:upscale placeholders).
# --------------------------------------------------------------------------- #
def test_registry_valid_and_entrypoints_wired():
    validate_registry()  # raises RegistryError on any problem
    wired = {
        (Framework.FFMPEG, Task.INTERPOLATE):
            "abstract_hugpy_dev.video_intel.studio.runners.ffmpeg_enhance:run_ffmpeg_interpolate",
        (Framework.FFMPEG, Task.UPSCALE):
            "abstract_hugpy_dev.video_intel.studio.runners.ffmpeg_enhance:run_ffmpeg_upscale",
        (Framework.RIFE, Task.INTERPOLATE):
            "abstract_hugpy_dev.video_intel.studio.runners.rife_interpolate:run_rife_interpolate",
        (Framework.LTX, Task.UPSCALE):
            "abstract_hugpy_dev.video_intel.studio.runners.ltx_upscale:run_ltx_upscale",
    }
    for key, ep in wired.items():
        spec = runner_for(*key)
        assert spec is not None, f"no runner registered for {key}"
        assert spec.entrypoint == ep, (
            f"{key} entrypoint must be {ep!r}; got {spec.entrypoint!r}")


# --------------------------------------------------------------------------- #
# [6] REAL interpolation: produce_clip interp @ 0.5GB with a tiny 8f@8fps source,
#     targeting 16fps -> a real mp4 whose ffprobe fps ~= 16 and whose frame count is
#     STRICTLY GREATER than the source (motion-interpolated, not a passthrough) and
#     near-doubled (>= 1.5x) — proving mci genuinely synthesized in-between frames.
# --------------------------------------------------------------------------- #
def test_real_interpolation_doubles_fps_and_frames():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping real interpolation check)")
        return
    sw, sh, sfps, sframes = _probe(_SRC_MP4)
    assert (sw, sh, round(sfps), sframes) == (_SRC_W, _SRC_H, _SRC_FPS, _SRC_FRAMES), \
        f"source fixture geometry unexpected: {(sw, sh, sfps, sframes)}"
    out_root = tempfile.mkdtemp(prefix="studio-enh-interp-")
    try:
        res = produce_clip(_req(Capability.INTERP, R_INTERP, 0.5),
                           env=_studio_env(), out_root=out_root, source_video=_SRC_MP4)
        assert res.is_ok(), f"interp must succeed on this box; got {res}"
        a = res.unwrap()
        assert os.path.isfile(a.path) and os.path.getsize(a.path) > 0, a.path
        ow, oh, ofps, oframes = _probe(a.path)
        assert abs(ofps - 16.0) < 1.0, f"interpolated fps must be ~16; got {ofps}"
        assert (ow, oh) == (_SRC_W, _SRC_H), f"interp keeps source geometry; got {(ow, oh)}"
        assert oframes > sframes, (
            f"interpolation must ADD frames (motion-interpolated, not dup/passthrough); "
            f"got {oframes} vs source {sframes}")
        assert oframes >= int(sframes * 1.5), (
            f"interpolated frame count must be near-doubled (~2x); got {oframes} "
            f"vs source {sframes} (minterpolate drops a few trailing un-interpolatable "
            f"frames, so it lands ~13 not exactly 16)")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [7] REAL upscale: produce_clip upres @ 0.5GB with the tiny 160x90 source, targeting
#     320x180 -> a real mp4 whose ffprobe geometry is EXACTLY 320x180.
# --------------------------------------------------------------------------- #
def test_real_upscale_hits_target_geometry():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping real upscale check)")
        return
    out_root = tempfile.mkdtemp(prefix="studio-enh-upres-")
    try:
        res = produce_clip(_req(Capability.UPRES, R_UPRES, 0.5),
                           env=_studio_env(), out_root=out_root, source_video=_SRC_MP4)
        assert res.is_ok(), f"upres must succeed on this box; got {res}"
        a = res.unwrap()
        assert os.path.isfile(a.path) and os.path.getsize(a.path) > 0, a.path
        ow, oh, _ofps, _oframes = _probe(a.path)
        assert (ow, oh) == (R_UPRES.width, R_UPRES.height), (
            f"upscaled geometry must be {R_UPRES.width}x{R_UPRES.height}; got {ow}x{oh}")
        assert (a.width, a.height) == (R_UPRES.width, R_UPRES.height), (
            f"artifact geometry must match; got {a.width}x{a.height}")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [8] Determinism / prompt-invariance: two interp renders that differ ONLY in the
#     prompt address DIFFERENT content_hashes (prompt is in the hash) but produce
#     BYTE-IDENTICAL output (the prompt never reaches ffmpeg; -threads 1 fixes bits).
# --------------------------------------------------------------------------- #
def test_prompt_in_hash_but_not_in_pixels():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg unavailable — skipping prompt-invariance check)")
        return
    o1 = tempfile.mkdtemp(prefix="studio-enh-p1-")
    o2 = tempfile.mkdtemp(prefix="studio-enh-p2-")
    try:
        a = produce_clip(_req(Capability.INTERP, R_INTERP, 0.5), env=_studio_env(),
                         out_root=o1, source_video=_SRC_MP4, prompt="neon noir").unwrap()
        b = produce_clip(_req(Capability.INTERP, R_INTERP, 0.5), env=_studio_env(),
                         out_root=o2, source_video=_SRC_MP4, prompt="soft watercolor").unwrap()
        assert a.content_hash != b.content_hash, "a different prompt must re-address the clip"
        assert _sha256(a.path) == _sha256(b.path), (
            "the prompt must NOT alter output bits (deterministic ffmpeg, -threads 1)")
    finally:
        shutil.rmtree(o1, ignore_errors=True)
        shutil.rmtree(o2, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [9] Resume-on-hash (INV-6): a second identical interp produce serves the existing
#     clip as-is (resumed=True), same path, without re-running ffmpeg.
# --------------------------------------------------------------------------- #
def test_resume_on_hash():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg unavailable — skipping resume check)")
        return
    out_root = tempfile.mkdtemp(prefix="studio-enh-resume-")
    try:
        a = produce_clip(_req(Capability.INTERP, R_INTERP, 0.5), env=_studio_env(),
                         out_root=out_root, source_video=_SRC_MP4).unwrap()
        assert a.resumed is False, "first render is a fresh render"
        b = produce_clip(_req(Capability.INTERP, R_INTERP, 0.5), env=_studio_env(),
                         out_root=out_root, source_video=_SRC_MP4).unwrap()
        assert b.resumed is True, "second identical render must resume the existing clip"
        assert b.path == a.path, "resume serves the SAME content-addressed clip"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [10] Premium graceful (interp): interp @ 6GB with a REAL source binds rife-practical
#      and dispatches to run_rife_interpolate, which degrades to Err(DEPS_MISSING) on
#      this box (Practical-RIFE not vendored). Never raises.
# --------------------------------------------------------------------------- #
def test_premium_rife_graceful_deps_missing():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping premium rife check)")
        return
    b = CapabilityRouter().resolve(_req(Capability.INTERP, R_INTERP, 6.0)).unwrap()
    assert b.model_id == "rife-practical", b.model_id  # sanity: the Err below IS from it
    out_root = tempfile.mkdtemp(prefix="studio-enh-rife-")
    try:
        res = produce_clip(_req(Capability.INTERP, R_INTERP, 6.0),
                           env=_studio_env(), out_root=out_root, source_video=_SRC_MP4)
        assert res.is_err(), f"rife on this box must be Err (not Ok); got {res}"
        assert res.error.code.value == "deps_missing", (
            f"rife must degrade to deps_missing (Practical-RIFE not vendored); "
            f"got {res.error.code.value}")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [11] Premium graceful (upres): upres @ 12GB with a REAL source binds
#      ltxv-spatial-upscaler and dispatches to run_ltx_upscale, which degrades to
#      Err(WEIGHTS_MISSING) on this box (HF license-gated 401 weights). Never raises.
# --------------------------------------------------------------------------- #
def test_premium_ltx_graceful_weights_missing():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping premium ltx check)")
        return
    b = CapabilityRouter().resolve(_req(Capability.UPRES, R_UPRES, 12.0)).unwrap()
    assert b.model_id == "ltxv-spatial-upscaler-0.9.7", b.model_id
    out_root = tempfile.mkdtemp(prefix="studio-enh-ltx-")
    try:
        res = produce_clip(_req(Capability.UPRES, R_UPRES, 12.0),
                           env=_studio_env(), out_root=out_root, source_video=_SRC_MP4)
        assert res.is_err(), f"ltx on this box must be Err (not Ok); got {res}"
        assert res.error.code.value == "weights_missing", (
            f"ltx must degrade to weights_missing (HF-gated weights not staged); "
            f"got {res.error.code.value}")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [12] Source-first spec error (interp): interp @ 0.5GB with NO source_video ->
#      Err(SOURCE_MISSING), checked BEFORE the ffmpeg box-capability check.
# --------------------------------------------------------------------------- #
def test_interp_no_source_is_source_missing():
    out_root = tempfile.mkdtemp(prefix="studio-enh-nosrc-i-")
    try:
        res = produce_clip(_req(Capability.INTERP, R_INTERP, 0.5),
                           env=_studio_env(), out_root=out_root)
        assert res.is_err(), f"interp with no source must be Err; got {res}"
        assert res.error.code.value == "source_missing", res.error.code.value
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [13] Source-first spec error (upres): upres @ 0.5GB with NO source_video ->
#      Err(SOURCE_MISSING).
# --------------------------------------------------------------------------- #
def test_upres_no_source_is_source_missing():
    out_root = tempfile.mkdtemp(prefix="studio-enh-nosrc-u-")
    try:
        res = produce_clip(_req(Capability.UPRES, R_UPRES, 0.5),
                           env=_studio_env(), out_root=out_root)
        assert res.is_err(), f"upres with no source must be Err; got {res}"
        assert res.error.code.value == "source_missing", res.error.code.value
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [14] Route: POST /video/studio/i2v {capability:"interp", source_video:<real mp4>}
#      -> 200 {job_id} (capability passes through; source validated + enqueued).
# --------------------------------------------------------------------------- #
def test_route_interp_source_video_200():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping route interp check)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "interp",
        "resolution": {"width": _SRC_W, "height": _SRC_H, "fps": 16},
        "vram_budget_gb": 0.5, "seed": 0,
        "source_video": _SRC_MP4,
        "prompt": "smooth the motion"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


# --------------------------------------------------------------------------- #
# [15] Route: POST /video/studio/i2v {capability:"upres", source_video:<real mp4>}
#      -> 200 {job_id}.
# --------------------------------------------------------------------------- #
def test_route_upres_source_video_200():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping route upres check)")
        return
    r = client.post("/video/studio/i2v", json={
        "capability": "upres",
        "resolution": {"width": R_UPRES.width, "height": R_UPRES.height, "fps": _SRC_FPS},
        "vram_budget_gb": 0.5, "seed": 0,
        "source_video": _SRC_MP4,
        "prompt": "upscale to HD"})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()


# --------------------------------------------------------------------------- #
# [16] Import safety: importing the three new runner modules pulls NONE of the heavy
#      GPU stack at module top — the app-boot invariant that lets them land in the
#      dispatch table without dragging torch/diffusers in.
# --------------------------------------------------------------------------- #
def test_enhance_imports_are_gpu_stack_free():
    import importlib as _il
    for mod in ("ffmpeg_enhance", "rife_interpolate", "ltx_upscale"):
        _il.import_module(f"abstract_hugpy_dev.video_intel.studio.runners.{mod}")
    heavy = [m for m in ("torch", "diffusers", "transformers", "bitsandbytes")
             if m in sys.modules]
    assert not heavy, f"importing the enhance runners must not pull the GPU stack; pulled {heavy}"


CHECKS = [
    ("router: interp@0.5GB -> ffmpeg-minterpolate (ffmpeg, interpolate)",
     test_router_interp_tiny_binds_ffmpeg),
    ("router: interp@6GB -> rife-practical (REAL premium outranks ffmpeg last-resort)",
     test_router_interp_big_binds_rife_real_outranks_ffmpeg),
    ("router: upres@0.5GB -> ffmpeg-lanczos-upscale (ffmpeg, upscale)",
     test_router_upres_tiny_binds_ffmpeg),
    ("router: upres@12GB -> ltxv-spatial-upscaler (REAL premium outranks ffmpeg last-resort)",
     test_router_upres_big_binds_ltx_real_outranks_ffmpeg),
    ("registry: validate_registry() passes + 4 enhancement entrypoints wired (ffmpeg x2, rife+ltx re-pointed)",
     test_registry_valid_and_entrypoints_wired),
    ("REAL interp: 8f@8fps -> 16fps, ffprobe fps~=16 + frames>source + near-2x (mci, not dup)",
     test_real_interpolation_doubles_fps_and_frames),
    ("REAL upres: 160x90 -> 320x180, ffprobe geometry exactly 320x180",
     test_real_upscale_hits_target_geometry),
    ("determinism: prompt in content_hash but NOT in pixels (byte-identical output)",
     test_prompt_in_hash_but_not_in_pixels),
    ("resume-on-hash: identical interp re-run serves the existing clip (resumed=True)",
     test_resume_on_hash),
    ("premium graceful: interp@6GB (rife) -> Err(deps_missing), never raises",
     test_premium_rife_graceful_deps_missing),
    ("premium graceful: upres@12GB (ltx) -> Err(weights_missing), never raises",
     test_premium_ltx_graceful_weights_missing),
    ("source-first: interp with no source_video -> Err(source_missing)",
     test_interp_no_source_is_source_missing),
    ("source-first: upres with no source_video -> Err(source_missing)",
     test_upres_no_source_is_source_missing),
    ("route: POST {capability:interp, source_video} -> 200 {job_id}",
     test_route_interp_source_video_200),
    ("route: POST {capability:upres, source_video} -> 200 {job_id}",
     test_route_upres_source_video_200),
    ("import safety: importing ffmpeg_enhance/rife_interpolate/ltx_upscale pulls no GPU stack",
     test_enhance_imports_are_gpu_stack_free),
]


def main() -> int:
    _setup_fixtures()
    passed = 0
    failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            try:
                fn()
            except Exception as exc:  # surface EVERY divergence, not just the first
                failed += 1
                print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"[{i}] PASS  {name}")
    finally:
        _teardown_fixtures()
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
