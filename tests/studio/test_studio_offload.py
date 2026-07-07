"""Studio render OFFLOAD to a GPU worker (option a) — conformance.

Locks the offload seam as executable checks, in the same script style as
``test_studio_cancel.py`` / ``test_studio_t2v.py`` (plain python, ``__main__``
guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check
FAILED, every check independently run so a failing one never masks the rest).
pytest is NOT installed in this venv, so there are no fixtures.

What is under test:
  * DECISION rule (video_intel/runners/studio_i2v.should_delegate): synthetic ->
    local, real -> delegate WHEN a worker is set, worker unset -> local, and the
    TEST-ONLY HUGPY_STUDIO_FORCE_REMOTE=1 override -> delegate even synthetic.
  * PAYLOAD round-trip: a spec's asdict -> studio_i2v_from_dict -> the SAME
    CapabilityRequest the in-process path builds (identical delegated construction).
  * artifact_result_to_payload / _payload_to_job_result: Ok -> shared-path ingest,
    Err -> the SAME JobError mapping (incl. retryable classification), worker
    error -> JobError, malformed payload -> retryable internal error.
  * StudioRenderManager state machine: one render at a time (busy), idempotent
    re-POST, cancel/status of an unknown job.
  * E2E with NO GPU and NO release: an EPHEMERAL worker agent spun from THIS tree
    on a loopback port (werkzeug, in-process, synthetic render only — no model
    load, no systemd, not registered as a real worker). Prove enqueue -> delegate
    -> worker thread renders synthetic -> shared-path result -> central ingests ->
    job done in the media store; and a cancel mid-render settles as 'cancelled'
    with NO clip left behind. The ephemeral agent is torn down afterwards.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_offload.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT  # noqa: E402
from abstract_hugpy_dev.video_intel import media_bus  # noqa: E402
from abstract_hugpy_dev.video_intel.runners import studio_i2v as S  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.artifacts import Artifact  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.enums import Capability  # noqa: E402
from abstract_hugpy_dev.video_intel.studio.errors import (  # noqa: E402
    Err,
    ErrorCode,
    Ok,
    StageError,
)
from abstract_hugpy_dev.video_intel.studio.job import (  # noqa: E402
    make_studio_i2v,
    studio_i2v_from_dict,
)
from abstract_hugpy_dev.video_intel.studio.schemas import (  # noqa: E402
    CapabilityRequest,
    Resolution,
)
from abstract_hugpy_dev.worker_agent.studio_render import (  # noqa: E402
    StudioRenderManager,
    register_studio_routes,
)

# LIVE-DB SAFETY + ISOLATION (mirrors test_invariants_conformance): media_bus.DB_PATH
# is derived from DEFAULT_ROOT at import and points at the LIVE media_jobs.db, which
# the running dev central's worker DAEMON continuously drains. If our E2E enqueued
# there, that daemon could CLAIM our job first (and run it in-process, without our
# ephemeral worker), stealing the render. Repoint the bus at a throwaway db BEFORE
# any db op so our jobs are ours alone and the live queue is never touched. (Clip
# storage still lands under DEFAULT_ROOT — only the job ledger moves.)
import tempfile  # noqa: E402
_TMP_DB_DIR = tempfile.mkdtemp(prefix="hugpy_offload_test_")
media_bus.DB_PATH = os.path.join(_TMP_DB_DIR, "media_jobs.db")
media_bus._initialized = False

_FFMPEG = shutil.which("ffmpeg") is not None

# Env keys we mutate — captured so every check restores the process env it found.
_ENV_KEYS = (
    "HUGPY_STUDIO_WORKER",
    "HUGPY_STUDIO_FORCE_REMOTE",
    "HUGPY_STUDIO_POLL_INTERVAL_S",
    "HUGPY_STUDIO_DELEGATE_TIMEOUT_S",
)


def _clear_offload_env() -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _clip_files_under(root: str) -> list:
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn == "clip.mp4":
                found.append(os.path.join(dirpath, fn))
    return found


def _synth_spec(out_root=None, *, cap="i2v", w=256, h=256, fps=24, seed=0):
    """A spec whose VRAM budget (0.5GB) is below every real model's floor, so the
    router binds a SYNTHETIC runner — a real render with NO GPU/weights."""
    return make_studio_i2v(
        capability=cap, width=w, height=h, fps=fps, vram_budget_gb=0.5,
        seed=seed, out_root=out_root)


def _real_spec():
    """A spec that resolves to a REAL model (wan2.1-t2v-1.3b @ 832x480, 8GB)."""
    return make_studio_i2v(
        capability="t2v", width=832, height=480, fps=16, vram_budget_gb=8.0, seed=0)


# --------------------------------------------------------------------------- #
# (1) DECISION rule matrix
# --------------------------------------------------------------------------- #
def test_decision_rule_matrix():
    _clear_offload_env()
    synth = _synth_spec()
    real = _real_spec()
    try:
        # No worker configured -> never delegate (in-process, unchanged).
        assert S.should_delegate(synth) is False, "no worker: synthetic must be local"
        assert S.should_delegate(real) is False, "no worker: real must be local"

        # Worker set -> synthetic local, real delegated (the core decision rule).
        os.environ["HUGPY_STUDIO_WORKER"] = "http://127.0.0.1:9"
        assert S.should_delegate(synth) is False, "worker set: synthetic must stay local"
        assert S.should_delegate(real) is True, "worker set: real must delegate"

        # Test-only force-remote -> delegate even a synthetic render.
        os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"
        assert S.should_delegate(synth) is True, "force-remote: synthetic must delegate"
        assert S.should_delegate(real) is True, "force-remote: real must delegate"

        # resolves_to_real_model is the pure discriminator (worker-independent).
        assert S.resolves_to_real_model(synth) is False
        assert S.resolves_to_real_model(real) is True
    finally:
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (2) PAYLOAD round-trip: spec -> asdict -> from_dict -> identical request
# --------------------------------------------------------------------------- #
def test_payload_request_round_trip():
    spec = make_studio_i2v(
        capability="i2v", width=512, height=384, fps=12, vram_budget_gb=8.0,
        seed=7, prompt="a cat", negative="blurry", source_video=None)
    d = asdict(spec)                         # the exact wire form central POSTs
    rebuilt = studio_i2v_from_dict(d)        # the exact spec the worker rebuilds

    # The request the worker builds must equal the request the in-process path
    # builds from the original spec — byte-for-byte (CapabilityRequest is frozen).
    req_direct = S.build_capability_request(spec)
    req_wire = S.build_capability_request(rebuilt)
    assert req_direct == req_wire, (
        f"delegated request must equal in-process request; "
        f"{req_direct} != {req_wire}")

    # And it must equal a hand-built CapabilityRequest with the spec's fields.
    expected = CapabilityRequest(
        capability=Capability("i2v"),
        target_resolution=Resolution(512, 384, 12),
        vram_budget_gb=8.0,
        source_video=None)
    assert req_wire == expected, f"request fields drifted: {req_wire} != {expected}"


# --------------------------------------------------------------------------- #
# (3) artifact_result_to_payload — Ok + Err (+ retryable classification)
# --------------------------------------------------------------------------- #
def test_artifact_result_to_payload():
    art = Artifact(path="/mnt/llm_storage/video_intel/studio/clips/abc/clip.mp4",
                   content_hash="abc", frames=48, width=256, height=256,
                   duration_s=2.0, resumed=False)
    ok = S.artifact_result_to_payload(Ok(art))
    assert ok["ok"] is True and ok["path"] == art.path, f"ok payload wrong: {ok}"
    assert ok["frames"] == 48 and ok["content_hash"] == "abc"

    # CANCELLED -> not retryable (intentional).
    canc = S.artifact_result_to_payload(
        Err(StageError(ErrorCode.CANCELLED, "stopped")))
    assert canc["ok"] is False, f"cancel payload must be ok=False: {canc}"
    assert canc["error"]["code"] == "cancelled", f"code: {canc}"
    assert canc["error"]["retryable"] is False, "cancel must not be retryable"

    # OOM -> retryable (transient/resource).
    oom = S.artifact_result_to_payload(Err(StageError(ErrorCode.OOM, "boom")))
    assert oom["error"]["code"] == "oom" and oom["error"]["retryable"] is True, oom

    # NO_GPU -> not retryable (policy/routing on this box).
    nogpu = S.artifact_result_to_payload(Err(StageError(ErrorCode.NO_GPU, "no cuda")))
    assert nogpu["error"]["code"] == "no_gpu" and nogpu["error"]["retryable"] is False, nogpu


# --------------------------------------------------------------------------- #
# (4) _payload_to_job_result — worker error -> JobError; malformed -> retryable
# --------------------------------------------------------------------------- #
def test_payload_to_job_result_errors():
    # A worker-side error payload rebuilds a JobError verbatim.
    jr = S._payload_to_job_result(
        {"ok": False, "error": {"code": "worker_lost", "message": "gone",
                                "retryable": True}}, "job-x")
    assert jr.ok is False and jr.error.code == "worker_lost", f"{jr}"
    assert jr.error.retryable is True and jr.error.message == "gone"
    assert jr.outputs == ()

    # A cancelled worker render maps to a 'cancelled' JobError (run_claimed will
    # then record a 'cancelled' terminal status, not 'failed').
    jc = S._payload_to_job_result(
        {"ok": False, "error": {"code": "cancelled", "message": "stopped",
                                "retryable": False}}, "job-c")
    assert jc.error.code == "cancelled" and jc.error.retryable is False, f"{jc}"

    # A malformed/empty payload is itself errors-as-data (retryable internal).
    jm = S._payload_to_job_result({}, "job-m")
    assert jm.ok is False and jm.error.retryable is True, f"{jm}"

    # ok=True but no path -> retryable internal (never a false success).
    jp = S._payload_to_job_result({"ok": True}, "job-p")
    assert jp.ok is False and jp.error.code == "internal", f"{jp}"


# --------------------------------------------------------------------------- #
# (5) StudioRenderManager — one at a time (busy), idempotent, unknown cancel
# --------------------------------------------------------------------------- #
def test_render_manager_state_machine():
    # A stub manager whose render thread never settles, so we can observe the
    # in-flight state deterministically (no dependence on render timing).
    class _StubMgr(StudioRenderManager):
        def _run(self, job, spec_dict):
            return  # leave status 'running' + _active set (simulated in-flight)

    mgr = _StubMgr()
    ok_a, why_a = mgr.start("job-A", {"width": 1})
    assert ok_a and why_a == "started", (ok_a, why_a)

    # A DIFFERENT job while one is in flight -> busy (one render at a time).
    ok_b, why_b = mgr.start("job-B", {"width": 1})
    assert ok_b is False and why_b == "busy", (ok_b, why_b)

    # A re-POST of the SAME job -> idempotent (exists), not busy.
    ok_a2, why_a2 = mgr.start("job-A", {"width": 1})
    assert ok_a2 is True and why_a2 == "exists", (ok_a2, why_a2)

    # cancel of a running job -> cancelling; unknown job -> not cancelled.
    c = mgr.cancel("job-A")
    assert c["cancelled"] is True and c["status"] == "cancelling", c
    cu = mgr.cancel("nope")
    assert cu["cancelled"] is False and cu["status"] == "unknown", cu

    # status of an unknown job -> 'unknown' (central reads this as worker_lost).
    su = mgr.status("nope")
    assert su["status"] == "unknown" and su["result"] is None, su


# --------------------------------------------------------------------------- #
# ephemeral worker plumbing for the E2E checks
# --------------------------------------------------------------------------- #
class _EphemeralWorker:
    """A studio worker agent spun from THIS tree on a loopback port: a bare Flask
    app with ONLY the studio render routes (no central registration, no systemd,
    no model load). Torn down via stop()."""

    def __init__(self):
        from flask import Flask
        from werkzeug.serving import make_server

        app = Flask("ephemeral-studio-worker")
        self.manager = register_studio_routes(
            app, worker_id="ephemeral-test", worker_name="ephemeral")
        self._srv = make_server("127.0.0.1", 0, app, threaded=True)
        self.port = self._srv.server_port
        self._thread = threading.Thread(
            target=self._srv.serve_forever, name="ephemeral-worker", daemon=True)
        self._thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        try:
            self._srv.shutdown()
        except Exception:
            pass


def _wait_until(pred, timeout_s: float, interval_s: float = 0.05) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval_s)
    return False


# --------------------------------------------------------------------------- #
# (6) E2E happy path: enqueue -> delegate -> worker renders -> central ingests
# --------------------------------------------------------------------------- #
def test_e2e_delegate_synthetic_render():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping E2E synthetic render)")
        return
    _clear_offload_env()
    worker = _EphemeralWorker()
    out_root = os.path.join(DEFAULT_ROOT, "video_intel", "studio", "clips",
                            f"offload-e2e-{os.getpid()}")
    os.environ["HUGPY_STUDIO_WORKER"] = worker.base
    os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"     # delegate the synthetic render
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.1"
    try:
        spec = _synth_spec(out_root=out_root, seed=101)
        job_id = media_bus.enqueue("studio_i2v", spec)

        # work_once claims + run_claimed -> run_studio_i2v -> DELEGATE (blocks in
        # the poll loop until the remote synthetic render settles), then INGESTS
        # the clip from the shared path. Synchronous by design.
        processed = media_bus.work_once()
        assert processed == job_id, f"work_once should process our job; got {processed}"

        view = media_bus.get(job_id)
        assert view["status"] == "done", f"job must be done; got {view['status']}"
        result = view["result"]
        assert result and result["ok"] is True, f"delegated render must be ok: {result}"
        outs = result.get("outputs") or []
        assert len(outs) == 1, f"one video MediaRef expected; got {outs}"
        ref = outs[0]
        assert ref["kind"] == "video", f"output must be kind=video; got {ref['kind']}"
        uri = ref["uri"]
        # The clip was written by the worker under the SHARED media-store root and
        # ingested by central directly (no b64) — prove the file is really there.
        assert os.path.isfile(uri) and os.path.getsize(uri) > 0, (
            f"ingested clip must exist under the shared root: {uri}")
        assert os.path.commonpath([os.path.realpath(uri),
                                    os.path.realpath(DEFAULT_ROOT)]) == \
            os.path.realpath(DEFAULT_ROOT), f"clip must be under the shared root: {uri}"
    finally:
        worker.stop()
        _clear_offload_env()
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (7) E2E cancel mid-render: media_bus.cancel -> forwarded -> settles 'cancelled'
# --------------------------------------------------------------------------- #
def test_e2e_cancel_mid_render():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping E2E cancel)")
        return
    _clear_offload_env()

    # Widen the mid-render window DETERMINISTICALLY (test-only): a tiny per-frame
    # sleep in the SAME-PROCESS worker render thread, so the render is reliably
    # in flight when central forwards the cancel. This does NOT change production
    # behavior — it only slows THIS test's frame loop.
    from abstract_hugpy_dev.video_intel.studio.runners import synthetic as _syn
    _orig_synth = _syn.synthesize_frame

    def _slow_synth(*a, **k):
        time.sleep(0.03)
        return _orig_synth(*a, **k)

    _syn.synthesize_frame = _slow_synth

    worker = _EphemeralWorker()
    out_root = os.path.join(DEFAULT_ROOT, "video_intel", "studio", "clips",
                            f"offload-cancel-{os.getpid()}")
    os.environ["HUGPY_STUDIO_WORKER"] = worker.base
    os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.1"
    try:
        spec = _synth_spec(out_root=out_root, seed=202)
        job_id = media_bus.enqueue("studio_i2v", spec)

        # Run the full bus cycle in the background (it blocks until settle).
        done = {}

        def _work():
            done["job"] = media_bus.work_once()

        wt = threading.Thread(target=_work, name="e2e-cancel-work", daemon=True)
        wt.start()

        # Wait until the render is genuinely in flight (central flipped 'running'
        # and the worker has started rendering).
        assert _wait_until(lambda: media_bus.get(job_id)["status"] == "running", 10.0), (
            "job never reached 'running'")
        assert _wait_until(
            lambda: worker.manager.status(job_id).get("status") == "running", 10.0), (
            "worker never started the render")
        time.sleep(0.15)  # let a few frames render so this is truly mid-render

        # Cancel via the bus (the ONLY cancel channel) — central's poll loop must
        # detect is_cancelling and forward POST /studio/cancel to the worker.
        res = media_bus.cancel(job_id)
        assert res["cancelled"] is True, f"cancel should engage; got {res}"

        wt.join(timeout=15.0)
        assert not wt.is_alive(), "work thread did not settle after cancel"

        view = media_bus.get(job_id)
        assert view["status"] == "cancelled", (
            f"a cancelled delegated render must settle 'cancelled'; got {view['status']}")
        result = view["result"]
        assert result and result["ok"] is False, f"cancelled result ok=False: {result}"
        assert result["error"]["code"] == "cancelled", f"error code: {result}"
        # abort-before-replace guarantee holds across the offload seam too.
        leftovers = _clip_files_under(out_root)
        assert leftovers == [], f"a cancelled render must leave NO clip.mp4; {leftovers}"
    finally:
        worker.stop()
        _syn.synthesize_frame = _orig_synth
        _clear_offload_env()
        shutil.rmtree(out_root, ignore_errors=True)


CHECKS = [
    ("decision rule: synthetic->local, real->delegate, unset->local, force->delegate",
     test_decision_rule_matrix),
    ("payload round-trip: spec asdict -> from_dict -> identical CapabilityRequest",
     test_payload_request_round_trip),
    ("artifact_result_to_payload: Ok path + Err retryable classification",
     test_artifact_result_to_payload),
    ("_payload_to_job_result: worker error/cancel/malformed -> JobError",
     test_payload_to_job_result_errors),
    ("StudioRenderManager: one-at-a-time busy, idempotent re-POST, unknown cancel",
     test_render_manager_state_machine),
    ("E2E: enqueue -> delegate -> worker synthetic render -> shared ingest -> done",
     test_e2e_delegate_synthetic_render),
    ("E2E: cancel mid-render -> forwarded to worker -> settles 'cancelled', no clip",
     test_e2e_cancel_mid_render),
]


def main() -> int:
    passed = 0
    failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            fn()
        except Exception as exc:  # surface EVERY divergence, not just the first
            failed += 1
            import traceback
            print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[{i}] PASS  {name}")
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
