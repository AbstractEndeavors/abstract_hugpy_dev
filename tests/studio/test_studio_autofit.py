"""Studio AUTOFIT budget — conformance.

Operator doctrine (2026-07-12): a BLANK vram budget must NOT be a guaranteed-fail low
guess. "Why can it not default to what's needed? If a model needs 14GB and it's blank,
why would it fail trying 6GB — just do 14, otherwise a fail is 100% likely." A blank
budget means AUTOFIT: size the routing budget to the SERVING WORKER's measured free VRAM.

This locks the autofit behavior as executable checks in the same script style as
``test_studio_offload.py`` / ``test_studio_movie_offload.py`` (plain python, ``__main__``
guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check FAILED,
every check independent). pytest is NOT installed in this venv.

What is under test:
  * ROUTE sentinel (_autofit_vram_budget): blank/absent/null -> None (autofit); an explicit
    number is the manual override (passthrough); a bad value still 400s in the factory.
  * SPEC threading: make_studio_i2v / make_studio_movie accept None (autofit) as legal,
    still reject bad numbers, and None round-trips through asdict -> from_dict.
  * AUTOFIT RESOLUTION (_resolve_autofit / _autofit_from_worker): a fake registry row with
    known free VRAM -> effective = free - margin (10% or 2GB, whichever larger); host-only
    URL match; no worker / no VRAM data -> the historical fallback (0.5); an EXPLICIT budget
    bypasses the worker lookup entirely.
  * render_clip STAMPS the resolved (effective_budget_gb, budget_source) — including the
    IN-PROCESS GPU-less central path (autofit -> fallback -> synthetic).
  * MOVIE: an id-movie with a None budget -> every segment carries the autofit budget
    (>= the VACE floor) + budget_source in movie.json; an EXPLICIT budget is byte-identical
    to today (floored, budget_source "explicit").

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_autofit.py
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
os.environ.pop("HUGPY_STUDIO_WORKER", None)
os.environ.pop("HUGPY_STUDIO_FORCE_REMOTE", None)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT  # noqa: E402
from abstract_hugpy_dev.video_intel import media_bus  # noqa: E402
from abstract_hugpy_dev.video_intel.runners import studio_i2v as S  # noqa: E402
from abstract_hugpy_dev.video_intel.runners.studio_movie import (  # noqa: E402
    run_generate_studio_movie,
)
from abstract_hugpy_dev.video_intel.studio.job import make_studio_i2v, studio_i2v_from_dict  # noqa: E402
from abstract_hugpy_dev.video_intel.studio_movie_schema import (  # noqa: E402
    StudioMovieGoal,
    make_studio_movie,
    studio_movie_from_dict,
)
from abstract_hugpy_dev.flask_app.app.routes.video_routes import _autofit_vram_budget  # noqa: E402

# The worker store the autofit resolver reads (patched per check). The ``import ... as``
# form trips a package-init quirk in this tree, so trigger the import via ``from ... import``
# and grab the module object out of sys.modules to patch its ``list_workers``.
from abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers import list_workers  # noqa: E402,F401
_WK = sys.modules["abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers"]

try:
    from PIL import Image  # noqa: E402
    _PIL = True
except Exception:  # noqa: BLE001
    _PIL = False

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

# LIVE-DB SAFETY (mirrors the offload/movie suites): repoint the bus at a throwaway db so
# is_cancelling / set_progress / enqueue never touch the running dev central's live db.
_TMP_DB_DIR = tempfile.mkdtemp(prefix="hugpy_autofit_test_")
media_bus.DB_PATH = os.path.join(_TMP_DB_DIR, "media_jobs.db")
media_bus._initialized = False

_GIB = 1024 ** 3
_ENV_KEYS = ("HUGPY_STUDIO_WORKER", "HUGPY_STUDIO_FORCE_REMOTE",
             "HUGPY_STUDIO_POLL_INTERVAL_S", "HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S",
             "HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S")


def _clear_env() -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _fake_workers(rows):
    """Install a fake list_workers returning ``rows``; return a restore callable."""
    orig = _WK.list_workers
    _WK.list_workers = lambda: list(rows)
    return lambda: setattr(_WK, "list_workers", orig)


def _worker_row(name, url, free_gib):
    return {"name": name, "url": url,
            "gpus": [{"index": 0, "memory_total": 24 * _GIB,
                      "memory_free": int(free_gib * _GIB)}]}


# --------------------------------------------------------------------------- #
# check runner (script idiom)
# --------------------------------------------------------------------------- #
_n = 0
_fail = 0


def check(desc, fn):
    global _n, _fail
    _n += 1
    try:
        fn()
        print(f"[{_n}] PASS  {desc}")
    except AssertionError as exc:
        _fail += 1
        print(f"[{_n}] FAIL  {desc}\n        {exc}")
    except Exception as exc:  # noqa: BLE001
        _fail += 1
        print(f"[{_n}] FAIL  {desc}\n        (unexpected {type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
# (A) ROUTE SENTINEL + factory
# --------------------------------------------------------------------------- #
def _route_sentinel():
    # blank / absent / null -> None (autofit)
    assert _autofit_vram_budget(None) is None
    assert _autofit_vram_budget("") is None
    assert _autofit_vram_budget("   ") is None
    # explicit number -> passthrough (manual override)
    assert _autofit_vram_budget(8.0) == 8.0
    assert _autofit_vram_budget(6) == 6
    # a bad non-empty value passes through so the FACTORY 400s it (not silently coerced)
    assert _autofit_vram_budget("bad") == "bad"


def _factory_accepts_none_i2v():
    sp = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                         vram_budget_gb=None, seed=0)
    assert sp.vram_budget_gb is None, sp.vram_budget_gb
    # explicit still works + coerces to float
    sp2 = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                          vram_budget_gb=8, seed=0)
    assert sp2.vram_budget_gb == 8.0 and isinstance(sp2.vram_budget_gb, float)


def _factory_rejects_bad_i2v():
    for bad in ("bad", 0, -1, [1]):
        try:
            make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                            vram_budget_gb=bad, seed=0)
            raise AssertionError(f"vram_budget_gb={bad!r} must be rejected (route 400)")
        except (ValueError, TypeError):
            pass


def _route_composition_i2v():
    # the exact route composition: make_studio_i2v(vram_budget_gb=_autofit_vram_budget(raw))
    sp = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                         vram_budget_gb=_autofit_vram_budget(""), seed=0)
    assert sp.vram_budget_gb is None
    try:
        make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                        vram_budget_gb=_autofit_vram_budget("bad"), seed=0)
        raise AssertionError("a bad budget string must still 400 through the factory")
    except (ValueError, TypeError):
        pass


def _factory_none_movie():
    g = (StudioMovieGoal(segment_id="s0", prompt="a"),)
    sp = make_studio_movie(goals=g, width=256, height=256, fps=8, vram_budget_gb=None)
    assert sp.vram_budget_gb is None
    for bad in ("bad", 0, -1):
        try:
            make_studio_movie(goals=g, width=256, height=256, fps=8, vram_budget_gb=bad)
            raise AssertionError(f"movie vram_budget_gb={bad!r} must be rejected")
        except (ValueError, TypeError):
            pass


def _none_roundtrips():
    sp = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                         vram_budget_gb=None, seed=0)
    import json
    back = studio_i2v_from_dict(json.loads(json.dumps(asdict(sp))))
    assert back.vram_budget_gb is None, back.vram_budget_gb
    # explicit round-trips too
    sp2 = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                          vram_budget_gb=8, seed=0)
    assert studio_i2v_from_dict(json.loads(json.dumps(asdict(sp2)))).vram_budget_gb == 8.0
    # movie round-trip
    g = (StudioMovieGoal(segment_id="s0", prompt="a"),)
    msp = make_studio_movie(goals=g, width=256, height=256, fps=8, vram_budget_gb=None)
    mback = studio_movie_from_dict(json.loads(json.dumps(asdict(msp))))
    assert mback.vram_budget_gb is None, mback.vram_budget_gb


# --------------------------------------------------------------------------- #
# (B) AUTOFIT RESOLUTION
# --------------------------------------------------------------------------- #
def _i2v(budget):
    return make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                           vram_budget_gb=budget, seed=0)


def _explicit_bypasses_lookup():
    # An explicit budget must NEVER touch the worker store. Patch list_workers to EXPLODE.
    def _boom():
        raise AssertionError("explicit budget must not read the worker store")
    restore = _fake_workers([])
    _WK.list_workers = _boom
    try:
        os.environ["HUGPY_STUDIO_WORKER"] = "http://10.9.9.9:9100"
        spec, eff, src = S._resolve_autofit(_i2v(8.0))
        assert src == "explicit" and eff == 8.0 and spec.vram_budget_gb == 8.0
    finally:
        restore()
        _clear_env()


def _autofit_sizes_to_worker():
    restore = _fake_workers([_worker_row("ae", "http://10.9.9.9:7003", 24.0)])
    try:
        # HUGPY_STUDIO_WORKER port (9100) differs from the registry row port (7003) —
        # host-only match must still resolve.
        os.environ["HUGPY_STUDIO_WORKER"] = "http://10.9.9.9:9100"
        spec, eff, src = S._resolve_autofit(_i2v(None))
        # 24 GiB free, margin = max(2.4, 2.0) = 2.4 -> 21.6
        assert abs(eff - 21.6) < 1e-6, eff
        assert src == "autofit:ae", src
        assert abs(spec.vram_budget_gb - 21.6) < 1e-6, spec.vram_budget_gb
    finally:
        restore()
        _clear_env()


def _margin_math():
    # 40 GiB: 10% (4.0) dominates -> 36.0
    restore = _fake_workers([_worker_row("big", "http://1.1.1.1:9100", 40.0)])
    try:
        os.environ["HUGPY_STUDIO_WORKER"] = "http://1.1.1.1:9100"
        _sp, eff, _src = S._resolve_autofit(_i2v(None))
        assert abs(eff - 36.0) < 1e-6, f"10% margin expected 36.0; got {eff}"
    finally:
        restore()
        _clear_env()
    # 12 GiB: 2GB floor (2.0 > 1.2) dominates -> 10.0
    restore = _fake_workers([_worker_row("small", "http://1.1.1.2:9100", 12.0)])
    try:
        os.environ["HUGPY_STUDIO_WORKER"] = "http://1.1.1.2:9100"
        _sp, eff, _src = S._resolve_autofit(_i2v(None))
        assert abs(eff - 10.0) < 1e-6, f"2GB floor expected 10.0; got {eff}"
    finally:
        restore()
        _clear_env()


def _no_match_fallback():
    # worker row exists but a DIFFERENT host -> no match -> fallback
    restore = _fake_workers([_worker_row("ae", "http://10.0.0.1:9100", 24.0)])
    try:
        os.environ["HUGPY_STUDIO_WORKER"] = "http://192.168.5.5:9100"
        spec, eff, src = S._resolve_autofit(_i2v(None))
        assert eff == 0.5 and src == "autofit:fallback", (eff, src)
        assert spec.vram_budget_gb == 0.5
    finally:
        restore()
        _clear_env()


def _no_worker_env_fallback():
    restore = _fake_workers([_worker_row("ae", "http://10.0.0.1:9100", 24.0)])
    try:
        _clear_env()  # no HUGPY_STUDIO_WORKER
        _sp, eff, src = S._resolve_autofit(_i2v(None))
        assert eff == 0.5 and src == "autofit:fallback", (eff, src)
    finally:
        restore()
        _clear_env()


def _no_vram_data_fallback():
    # matched box, but it reports no gpus / no memory_free -> fallback
    restore = _fake_workers([{"name": "gpuless", "url": "http://10.9.9.9:9100", "gpus": []}])
    try:
        os.environ["HUGPY_STUDIO_WORKER"] = "http://10.9.9.9:9100"
        _sp, eff, src = S._resolve_autofit(_i2v(None))
        assert eff == 0.5 and src == "autofit:fallback", (eff, src)
    finally:
        restore()
        _clear_env()


# --------------------------------------------------------------------------- #
# (C) render_clip STAMPS the resolved budget (incl. in-process GPU-less central)
# --------------------------------------------------------------------------- #
def _render_clip_inprocess_fallback():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="autofit-inproc-", dir=DEFAULT_ROOT)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    _clear_env()  # no worker -> in-process on this GPU-less box
    try:
        spec = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                               vram_budget_gb=None, seed=0, out_root=work)
        outcome = S.render_clip(spec, render_id="autofit-inproc")
        assert outcome.ok is True, f"synthetic fallback render must succeed; got {outcome.error}"
        assert outcome.budget_source == "autofit:fallback", outcome.budget_source
        assert outcome.effective_budget_gb == 0.5, outcome.effective_budget_gb
        assert os.path.isfile(outcome.path), outcome.path
    finally:
        media_bus.is_cancelling = orig_cx
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


def _render_clip_explicit_stamp():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="autofit-explicit-", dir=DEFAULT_ROOT)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    _clear_env()
    try:
        spec = make_studio_i2v(capability="i2v", width=256, height=256, fps=8,
                               vram_budget_gb=0.5, seed=1, out_root=work)
        outcome = S.render_clip(spec, render_id="autofit-explicit")
        assert outcome.ok is True, outcome.error
        assert outcome.budget_source == "explicit", outcome.budget_source
        assert outcome.effective_budget_gb == 0.5, outcome.effective_budget_gb
    finally:
        media_bus.is_cancelling = orig_cx
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (D) MOVIE autofit — a compact fake worker so an id-movie completes on a fake GPU box
# --------------------------------------------------------------------------- #
_FPS, _W, _H = 12, 320, 180
_SEG_FRAMES = _FPS * 2


class _FakeWorker:
    """Minimal studio-worker HTTP mock (mirrors test_studio_movie_offload): captures each
    delegated spec, renders a solid-gray clip on the SHARED store, scripts a running->done
    poll. Content-addresses by spec so a re-run resumes."""

    def __init__(self):
        self.posts = []
        self.renders = {}
        self._seen = {}

    @staticmethod
    def _key(spec):
        return (spec["out_root"], spec.get("prompt"), spec["seed"],
                spec["width"], spec["height"], spec["fps"], spec["capability"])

    def post(self, url, payload, timeout):
        if url.endswith("/studio/render"):
            rid, spec = payload["job_id"], payload["spec"]
            self.posts.append((rid, spec))
            key = self._key(spec)
            resumed = key in self._seen
            if resumed:
                clip = self._seen[key]
            else:
                clip = os.path.join(spec["out_root"], "_worker", "clip.mp4")
                _build_gray_clip(clip, _SEG_FRAMES, spec["width"], spec["height"], spec["fps"])
                self._seen[key] = clip
            self.renders[rid] = {"polls": 0, "done": {
                "ok": True, "path": clip, "content_hash": f"wc-{abs(hash(key))}",
                "frames": _SEG_FRAMES, "width": spec["width"], "height": spec["height"],
                "duration_s": _SEG_FRAMES / spec["fps"], "resumed": resumed}}
            return 202, {"ok": True, "accepted": "started", "pkg_version": S._pkg_version()}
        return 200, {"cancelled": True}

    def get(self, url, timeout):
        rid = url.rsplit("/", 1)[-1]
        r = self.renders.get(rid)
        if r is None:
            return 200, {"status": "unknown", "result": None}
        r["polls"] += 1
        frames = [{"status": "running", "progress": {"phase": "r"}},
                  {"status": "done", "result": r["done"]}]
        return 200, dict(frames[min(r["polls"] - 1, len(frames) - 1)])


def _build_gray_clip(dst, n_frames, w, h, fps):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fdir = tempfile.mkdtemp(prefix=".wframes-", dir=os.path.dirname(dst))
    try:
        for n in range(n_frames):
            Image.new("RGB", (w, h), (128, 128, 128)).save(os.path.join(fdir, f"f_{n:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fdir, "f_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        shutil.rmtree(fdir, ignore_errors=True)


def _install_http(fake):
    op, og = S._http_post_json, S._http_get_json
    S._http_post_json, S._http_get_json = fake.post, fake.get
    return op, og


def _fast_env(base):
    _clear_env()
    os.environ["HUGPY_STUDIO_WORKER"] = base
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.01"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S"] = "0.2"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"] = "0.02"


def _id_movie_autofit():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="autofit-idmovie-", dir=DEFAULT_ROOT)
    refs = []
    for i in range(2):
        rp = os.path.join(work, f"ref_{i}.png")
        Image.new("RGB", (64, 64), (30 + i * 40, 20, 10)).save(rp)
        refs.append(rp)
    fake = _FakeWorker()
    op, og = _install_http(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    restore = _fake_workers([_worker_row("ae", "http://10.9.9.9:7003", 24.0)])
    _fast_env("http://10.9.9.9:9100")   # host matches the fake registry row
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="her on the beach"),
                 StudioMovieGoal(segment_id="s1", prompt="playing volleyball",
                                 parent_segment_id="s0", joint_mode="cut", seed=22))
        spec = make_studio_movie(goals=goals, width=_W, height=_H, fps=_FPS,
                                 vram_budget_gb=None,   # BLANK -> autofit
                                 seed=21, out_root=work, reference_images=tuple(refs))
        res = run_generate_studio_movie(spec, job_id="idm-autofit")
        assert res.ok is True, f"autofit id-movie must complete; got {res.error}"
        # Every delegated segment carried the AUTOFIT budget (21.6), well above the VACE floor.
        assert len(fake.posts) == 2, f"both segments must delegate; got {len(fake.posts)}"
        for rid, dspec in fake.posts:
            assert dspec["capability"] == "id_lock", rid
            assert abs(dspec["vram_budget_gb"] - 21.6) < 1e-6, (
                f"{rid}: autofit budget expected 21.6; got {dspec['vram_budget_gb']}")
            assert dspec["vram_budget_gb"] >= 6.0, "autofit budget must clear the VACE floor"
        # movie.json / seg records carry the RESOLVED budget + budget_source.
        for seg in res.movie["segments"]:
            assert seg["budget_source"] == "autofit:ae", seg.get("budget_source")
            assert abs(seg["vram_budget_gb"] - 21.6) < 1e-6, seg["vram_budget_gb"]
            assert seg["vram_budget_gb"] >= 6.0, "autofit budget must clear the VACE floor"
    finally:
        media_bus.is_cancelling = orig_cx
        S._http_post_json, S._http_get_json = op, og
        restore()
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


def _id_movie_explicit_regression():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="autofit-idmovie-expl-", dir=DEFAULT_ROOT)
    refs = [os.path.join(work, "ref.png")]
    Image.new("RGB", (64, 64), (30, 20, 10)).save(refs[0])
    fake = _FakeWorker()
    op, og = _install_http(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    # A live worker row exists, but an EXPLICIT budget must BYPASS the lookup entirely.
    restore = _fake_workers([_worker_row("ae", "http://10.9.9.9:7003", 24.0)])
    _fast_env("http://10.9.9.9:9100")
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="her on the beach"),)
        spec = make_studio_movie(goals=goals, width=_W, height=_H, fps=_FPS,
                                 vram_budget_gb=8.0,    # EXPLICIT -> unchanged (floored to >=6)
                                 seed=7, out_root=work, reference_images=tuple(refs))
        res = run_generate_studio_movie(spec, job_id="idm-explicit")
        assert res.ok is True, res.error
        (_rid, dspec), = fake.posts
        # Byte-identical to today: explicit 8.0 floored to max(8,6)=8.0, source "explicit".
        assert dspec["vram_budget_gb"] == 8.0, dspec["vram_budget_gb"]
        seg = res.movie["segments"][0]
        assert seg["budget_source"] == "explicit", seg.get("budget_source")
        assert seg["vram_budget_gb"] == 8.0, seg["vram_budget_gb"]
    finally:
        media_bus.is_cancelling = orig_cx
        S._http_post_json, S._http_get_json = op, og
        restore()
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    # (A) route sentinel + spec threading
    check("route sentinel: blank/absent/null -> None; explicit passthrough; bad passthrough",
          _route_sentinel)
    check("factory: make_studio_i2v accepts None (autofit); explicit -> float", _factory_accepts_none_i2v)
    check("factory: make_studio_i2v still rejects bad budgets (route 400)", _factory_rejects_bad_i2v)
    check("route composition: blank -> None spec; bad -> 400 in factory", _route_composition_i2v)
    check("factory: make_studio_movie accepts None; rejects bad", _factory_none_movie)
    check("round-trip: None budget survives asdict -> from_dict (i2v + movie)", _none_roundtrips)
    # (B) autofit resolution
    check("resolve: EXPLICIT budget bypasses the worker lookup entirely", _explicit_bypasses_lookup)
    check("resolve: None budget sizes to worker free VRAM (host-only match)", _autofit_sizes_to_worker)
    check("resolve: margin = max(10%, 2GB) (40GiB->36.0, 12GiB->10.0)", _margin_math)
    check("resolve: no host match -> fallback 0.5 / autofit:fallback", _no_match_fallback)
    check("resolve: no HUGPY_STUDIO_WORKER -> fallback", _no_worker_env_fallback)
    check("resolve: matched box with no VRAM data -> fallback", _no_vram_data_fallback)
    # (C) render_clip stamping (incl. in-process GPU-less central)
    check("render_clip: in-process fallback (GPU-less central) -> synthetic + autofit:fallback stamp",
          _render_clip_inprocess_fallback)
    check("render_clip: explicit budget -> 'explicit' stamp", _render_clip_explicit_stamp)
    # (D) movie autofit + movie.json budget_source
    check("movie: id-movie None budget -> segments carry autofit budget >= floor + budget_source",
          _id_movie_autofit)
    check("movie: id-movie EXPLICIT budget -> floored, byte-identical, budget_source 'explicit'",
          _id_movie_explicit_regression)

    print(f"\n{_n - _fail} passed, {_fail} failed of {_n}")
    sys.exit(1 if _fail else 0)
