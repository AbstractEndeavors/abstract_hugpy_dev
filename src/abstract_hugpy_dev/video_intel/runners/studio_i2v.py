"""Studio i2v BUS RUNNER (B2) — the boundary between the media bus and the studio
spine. Pure ``(StudioI2VSpec, job_id) -> JobResult`` (map §6): expected failures
are DATA (a ``JobResult(ok=False, JobError(...))``), never a raise; only a genuine
programmer error would raise, and ``media_bus.run_claimed`` is the one place that
catches that.

It does four things and nothing else:
  1. Resolve a concrete ``StudioEnv`` from worker defaults (INV-5; no operator env).
  2. Lift the JSON-safe spec into a studio ``CapabilityRequest`` and call
     ``produce_clip`` — the studio's own spine (router -> manifest -> runner ->
     content-addressed clip). Resume (INV-6) is handled inside ``produce_clip``: an
     identical spec re-run reuses the existing clip, no regeneration.
  3. Translate the studio Result at THIS boundary — and only here — via
     ``_stage_error_to_job_error``: studio's ``StageError`` becomes the bus's own
     ``JobError``. Since the Task 2 collapse there is ONE JobError class
     (result_schema.JobError IS comms.jobs.JobError); studio's ``StageError`` is a
     SEPARATE studio-layer vocabulary that this seam ADAPTS into that unified
     JobError — a translation at the single boundary, not a merge, leaving studio's
     errors.py self-contained (StageError reconciliation is TODO(P0-1)).
  4. Ingest the produced ``clip.mp4`` into the media store so the clip is cataloged
     as a ``MediaRef`` (kind="video"), carried out on ``JobResult.outputs``.

STUDIO RENDER OFFLOAD (option a). A REAL-model render (e.g. wan2.1-t2v-1.3b) does
not fit central's GPU-less control-plane box, so this adapter can DELEGATE the
``produce_clip`` execution to a studio GPU worker (``worker_agent/studio_render.py``)
while central keeps the whole control plane and media_jobs.db stays single-writer
here. The decision is data-driven and reversible:

  * ``HUGPY_STUDIO_WORKER`` unset            -> in-process (today's behavior).
  * resolves to a SYNTHETIC / ffmpeg / unroutable binding -> in-process (central
    renders those fine, no model load).
  * resolves to a REAL model AND a worker is set -> delegate: POST /studio/render,
    poll GET status, forward progress + cancel, then INGEST the clip from the
    SHARED content-addressed path (central + worker share /mnt/llm_storage — no b64
    round-trip). The post-render handling (ingest on Ok, JobError on Err) is
    IDENTICAL to the in-process path — the same ``run_produce_clip`` builds the
    request on both sides and the same ``_stage_error_to_job_error`` classifies an
    Err (executed on the worker, rebuilt verbatim here).
  * ``HUGPY_STUDIO_FORCE_REMOTE=1`` is a TEST-ONLY override that delegates even a
    synthetic render (so the offload path is provable with no GPU / no release).

Failure modes are errors-as-data, never a hang or a stuck 'running' row: a worker
unreachable AT KICK-OFF falls back to the in-process path (which on central is the
graceful NO_GPU/DEPS preflight); a worker that dies or times out AFTER accepting
the job returns a retryable ``JobError`` (worker_lost / delegation_timeout).

Heavy studio imports (which transitively pull numpy/PIL via the synthetic runner)
are LAZY — done inside the functions — so importing this module (which the bus does
at boot, via ``runners/__init__``) stays cheap and can never break app boot.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict

from ..media_store import ingest
from ..result_schema import JobError, JobResult

logger = logging.getLogger(__name__)

# StageError codes worth a retry (transient/resource), vs. policy/routing failures
# where the SAME spec would deterministically fail again. Kept as the string values
# so this module needs no studio-enum import at top.
_RETRYABLE_CODES = frozenset({"oom", "nan_in_vae", "assembly_failed", "io_error"})

# --- offload env knobs (all optional; unset => today's in-process behavior) ---
_WORKER_ENV = "HUGPY_STUDIO_WORKER"            # base URL of the studio GPU worker
_FORCE_REMOTE_ENV = "HUGPY_STUDIO_FORCE_REMOTE"  # TEST-ONLY: delegate even synthetic
_POLL_ENV = "HUGPY_STUDIO_POLL_INTERVAL_S"     # status poll cadence (s)
_TIMEOUT_ENV = "HUGPY_STUDIO_DELEGATE_TIMEOUT_S"  # RENDER budget once RUNNING (s)
_OVERALL_CAP_ENV = "HUGPY_STUDIO_OVERALL_CAP_S"   # overall wall-clock cap incl. queue wait (s)
# Kick-off retry (item 2): a ConnectionReset/refused at kick-off (the post-restart /
# converge socket window) retries within this window before falling back in-process.
_KICKOFF_RETRY_WINDOW_ENV = "HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S"
_KICKOFF_RETRY_INTERVAL_ENV = "HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"
_DEFAULT_POLL_S = 2.0
# TWO clocks (item 1): the RENDER budget bounds a render once it is actually RUNNING
# on the worker (it does NOT penalize time a job spent WAITING in the worker's queue);
# the OVERALL cap is a separate wall-clock ceiling over the whole delegation (queue
# wait + kick-off-409 retries + render) so a wedged worker can never hang a job forever.
_DEFAULT_TIMEOUT_S = 1800.0                     # 30 min: a real Wan render fits well under
_DEFAULT_OVERALL_CAP_S = 7200.0                # 2 h: render budget + a few queued jobs ahead
_DEFAULT_KICKOFF_RETRY_WINDOW_S = 30.0         # retry a reset/refused kick-off for ~30s
_DEFAULT_KICKOFF_RETRY_INTERVAL_S = 5.0        # ...every ~5s (also the queue-full retry cadence)
_MAX_POLL_ERRORS = 5                            # consecutive poll failures => worker_lost
_KICKOFF_TIMEOUT_S = 30.0                       # POST /studio/render connect budget
_CANCEL_TIMEOUT_S = 15.0
_STATUS_TIMEOUT_S = 20.0


def _pkg_version() -> str:
    """This tree's package version — echoed to the worker so a version skew is
    LOGGED (never fatal). Best-effort; 'unknown' if the import can't resolve."""
    try:
        from abstract_hugpy_dev import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _stage_error_to_job_error(stage_error) -> JobError:
    """BOUNDARY adapter: studio ``StageError`` -> bus ``JobError``. The ONE place the
    two error vocabularies meet; a translation, not a merge."""
    code = getattr(stage_error.code, "value", str(stage_error.code))
    return JobError(
        code=code,
        message=str(stage_error),
        retryable=code in _RETRYABLE_CODES,
    )


# --------------------------------------------------------------------------- #
# Shared spine helpers — used by BOTH the central in-process path (below) and the
# worker studio-render thread (worker_agent/studio_render.py imports these), so a
# delegated render is constructed and executed IDENTICALLY to a local one.
# --------------------------------------------------------------------------- #
def build_capability_request(spec):
    """Lift a ``StudioI2VSpec`` into a studio ``CapabilityRequest``. The SINGLE
    source of the request construction, so the in-process render, the remote
    worker render, and the delegation DECISION rule all agree exactly (this is the
    request the router resolves and the manifest is built from)."""
    from ..studio.enums import Capability
    from ..studio.schemas import CapabilityRequest, Resolution

    return CapabilityRequest(
        capability=Capability(spec.capability),
        target_resolution=Resolution(spec.width, spec.height, spec.fps),
        vram_budget_gb=spec.vram_budget_gb,
        # B2 chain: carry the source clip on the request (routing does not key on it;
        # produce_clip reads spec.source_video for the manifest + extend). None -> None.
        source_video=getattr(spec, "source_video", None),
        # DIRECT MODEL CHOICE: thread the pin so the router binds the requested model or
        # returns a clear Err. Also steers the delegation DECISION (a pinned real model
        # delegates to the GPU worker), since should_delegate/resolves_to_real_model both
        # build the request through here. getattr keeps older spec dicts (no model_id)
        # working. None -> auto-pick.
        pinned_model_id=getattr(spec, "model_id", None),
    )


def run_produce_clip(spec, should_cancel):
    """Build env + seeds from ``spec`` and run ``produce_clip``; return the studio
    ``Result[Artifact, StageError]`` verbatim.

    The ONE place a ``StudioI2VSpec`` is turned into a ``produce_clip`` call —
    shared so the central in-process path and the worker render thread render the
    same pixels for the same spec. ``should_cancel`` is the cooperative-cancel
    probe threaded down to the runner (Task 1)."""
    from ..studio.job import resolve_studio_env
    from ..studio.produce import produce_clip
    from ..studio.schemas import SeedBundle

    request = build_capability_request(spec)
    env = resolve_studio_env(spec.out_root, master_fps=spec.fps)
    seeds = SeedBundle(global_seed=spec.seed, stage_seeds=(("base", spec.seed),))
    return produce_clip(
        request,
        env=env,
        out_root=spec.out_root,
        seeds=seeds,
        # SAMPLER OVERRIDES: pass the spec's optional steps/cfg (None = unset -> the bound
        # model's family default fills them in produce_clip). getattr keeps older spec
        # dicts (no steps/cfg fields) working.
        steps=getattr(spec, "steps", None),
        cfg=getattr(spec, "cfg", None),
        start_image=spec.start_image,
        prompt=getattr(spec, "prompt", "") or "",
        negative_prompt=getattr(spec, "negative", "") or "",
        # B2 chain: the prior-tier clip (movie/scene mp4) this job extends. Carried
        # into the manifest; the i2v runner extends from its last frame when there is
        # no start_image. None -> "" (no source) inside produce_clip.
        source_video=getattr(spec, "source_video", None),
        # IDENTITY LOCK (id_lock): reference image paths + optional VACE control still,
        # carried into the manifest (canonical inputs). The VACE runner consumes them
        # (reference-to-video / control channel). getattr keeps older spec dicts working.
        reference_images=getattr(spec, "reference_images", None),
        control_image=getattr(spec, "control_image", None),
        control_kind=getattr(spec, "control_kind", None),
        should_cancel=should_cancel,
    )


def artifact_result_to_payload(result) -> dict:
    """Serialize a ``produce_clip`` ``Result`` into the JSON-safe payload the worker
    returns over HTTP and central consumes.

    ``Ok(Artifact)`` -> ``{"ok": True, "path": <shared clip path>, ...geometry}``;
    central re-``ingest``s that path (which re-probes geometry). ``Err(StageError)``
    -> ``{"ok": False, "error": {...JobError fields}}`` — the SAME
    ``_stage_error_to_job_error`` mapping the in-process path uses runs HERE (on
    the worker), so the code + ``retryable`` classification lives in exactly one
    place and central rebuilds the JobError verbatim."""
    if result.is_err():
        je = _stage_error_to_job_error(result.error)
        return {"ok": False, "error": {
            "code": je.code, "message": je.message, "retryable": je.retryable}}
    art = result.unwrap()
    return {
        "ok": True,
        "path": art.path,
        "content_hash": art.content_hash,
        "frames": art.frames,
        "width": art.width,
        "height": art.height,
        "duration_s": art.duration_s,
        "resumed": art.resumed,
    }


def _payload_to_job_result(payload: dict, job_id: str) -> JobResult:
    """Turn a worker render payload (from ``artifact_result_to_payload``) into a
    ``JobResult``, applying the EXACT post-``produce_clip`` handling the in-process
    path uses: Ok -> ``ingest`` the SHARED clip path into the media store (mints the
    video MediaRef); Err -> rebuild the JobError. Central-only (it does the
    ingest). A malformed/empty payload is itself errors-as-data."""
    if not isinstance(payload, dict) or "ok" not in payload:
        return JobResult(job_id=job_id, ok=False, error=JobError(
            code="internal",
            message="studio worker returned no render result payload",
            retryable=True))
    if not payload.get("ok"):
        err = payload.get("error") or {}
        return JobResult(job_id=job_id, ok=False, error=JobError(
            code=err.get("code", "internal"),
            message=err.get("message", "studio worker render failed"),
            retryable=bool(err.get("retryable", False))))
    path = payload.get("path")
    if not path:
        return JobResult(job_id=job_id, ok=False, error=JobError(
            code="internal",
            message="studio worker reported ok but no clip path",
            retryable=True))
    # SHARED filesystem: central + worker both see /mnt/llm_storage, so the clip the
    # worker wrote is ingestable here directly — identical to the in-process path.
    ref = ingest(path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))


# --------------------------------------------------------------------------- #
# Delegation decision + HTTP loop (central side)
# --------------------------------------------------------------------------- #
def _studio_worker_base() -> str:
    return (os.environ.get(_WORKER_ENV) or "").strip().rstrip("/")


def resolves_to_real_model(spec) -> bool:
    """Read-only router probe: does ``spec`` bind a REAL (non-synthetic) model at
    its vram budget? A synthetic / ffmpeg last-resort binding, or an unroutable
    request, -> False (central handles those in-process; an unroutable request
    surfaces its router Err-as-data on the in-process path unchanged). Never
    raises — a probe failure conservatively keeps the render local."""
    try:
        from ..studio.registry import MODEL_REGISTRY
        from ..studio.router import CapabilityRouter

        res = CapabilityRouter().resolve(build_capability_request(spec))
        if res.is_err():
            return False
        cfg = MODEL_REGISTRY.get(res.unwrap().model_id)
        return bool(cfg is not None and not cfg.synthetic)
    except Exception:  # noqa: BLE001
        return False


def should_delegate(spec) -> bool:
    """True iff this render should be sent to a studio GPU worker: a worker base
    URL is configured AND (the force-remote test override is on OR the spec binds a
    real model). No worker configured => never delegate (in-process, unchanged)."""
    if not _studio_worker_base():
        return False
    if os.environ.get(_FORCE_REMOTE_ENV) == "1":
        return True
    return resolves_to_real_model(spec)


def _http_post_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.getcode(), (json.loads(body) if body else {})


def _http_get_json(url: str, timeout: float):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.getcode(), (json.loads(body) if body else {})


def _timeout_result(job_id: str, base: str, cancel_url: str) -> JobResult:
    """Overall wall-clock budget exceeded: best-effort tell the worker to stop,
    then return a retryable JobError (never a stuck 'running' row)."""
    try:
        _http_post_json(cancel_url, {}, timeout=_CANCEL_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        pass
    logger.warning("studio job %s exceeded delegation timeout on %s", job_id, base)
    return JobResult(job_id=job_id, ok=False, error=JobError(
        code="delegation_timeout",
        message=f"studio render on {base} exceeded the delegation timeout",
        retryable=True))


def _delegate_to_worker(base: str, spec, job_id: str):
    """Delegate the render to the studio worker at ``base`` and settle the job.

    Returns a ``JobResult`` once the remote render settles (done/error/cancelled/
    timeout/worker-lost/queue-full), OR ``None`` to signal "kick-off failed, fall
    back to the in-process path" — used ONLY before the worker has accepted the job
    (an unreachable worker at kick-off is exactly the scout's "worker unreachable ->
    current behavior"). Once the worker returns 202 we never fall back (that would
    risk a double render); a later failure becomes a retryable JobError.

    QUEUE (item 1): a busy worker now QUEUES a second render (202 accepted="queued"
    + position) rather than 409ing, so the delegation just polls it through. A 409
    means the worker's bounded queue is FULL — worker_busy now means WAIT, so we
    retry kick-off with backoff inside the OVERALL cap, only failing worker_busy when
    that window expires. KICK-OFF RESET RETRY (item 2): a ConnectionReset/refused at
    kick-off (post-restart socket window) retries for ~30s before falling back."""
    from .. import media_bus

    spec_dict = asdict(spec)
    central_version = _pkg_version()
    start_payload = {"job_id": job_id, "spec": spec_dict,
                     "central_version": central_version}

    poll_s = _float_env(_POLL_ENV, _DEFAULT_POLL_S)
    render_budget = _float_env(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_S)
    overall_cap = _float_env(_OVERALL_CAP_ENV, _DEFAULT_OVERALL_CAP_S)
    reset_window = _float_env(_KICKOFF_RETRY_WINDOW_ENV, _DEFAULT_KICKOFF_RETRY_WINDOW_S)
    retry_interval = _float_env(_KICKOFF_RETRY_INTERVAL_ENV, _DEFAULT_KICKOFF_RETRY_INTERVAL_S)

    started_at = time.time()
    overall_deadline = started_at + overall_cap    # whole-delegation ceiling
    reset_deadline = started_at + reset_window      # connection-reset retry window
    cancel_url = base + "/studio/cancel/" + job_id

    # ---- kick off the remote render (retry: 409-full within overall cap; a reset/
    #      refused within the reset window; other errors -> fall back in-process) --
    body = None
    while True:
        try:
            _code, body = _http_post_json(
                base + "/studio/render", start_payload, timeout=_KICKOFF_TIMEOUT_S)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                # Worker's bounded queue is FULL. worker_busy means WAIT: retry within
                # the overall cap, only failing worker_busy when the window expires.
                if time.time() >= overall_deadline:
                    logger.warning("studio worker %s queue full past overall cap — "
                                   "job %s", base, job_id)
                    return JobResult(job_id=job_id, ok=False, error=JobError(
                        code="worker_busy",
                        message=f"studio worker queue full past the delegation "
                                f"window: {base}",
                        retryable=True))
                logger.info("studio worker %s queue full (409) — retrying kick-off "
                            "in %.1fs (job %s)", base, retry_interval, job_id)
                time.sleep(retry_interval)
                continue
            # Any other HTTP error at kick-off: the worker never started this render,
            # so fall back to the in-process path (graceful NO_GPU/DEPS on central).
            logger.warning("studio worker %s rejected /studio/render (HTTP %s) — "
                           "falling back in-process for job %s", base, exc.code, job_id)
            return None
        except (urllib.error.URLError, OSError) as exc:
            # ConnectionReset/refused/unreachable (e.g. the post-restart converge
            # socket window, seen twice live): retry within the reset window, then
            # fall back in-process (the scout's "worker unreachable" behavior).
            if time.time() >= reset_deadline:
                logger.warning("studio worker %s unreachable at kick-off past retry "
                               "window (%s: %s) — falling back in-process for job %s",
                               base, type(exc).__name__, exc, job_id)
                return None
            logger.info("studio worker %s kick-off connection error (%s: %s) — "
                        "retrying in %.1fs (job %s)",
                        base, type(exc).__name__, exc, retry_interval, job_id)
            time.sleep(retry_interval)
            continue
        except ValueError as exc:
            # Malformed response body — not transient; fall back in-process.
            logger.warning("studio worker %s returned an unparseable kick-off body "
                           "(%s) — falling back in-process for job %s",
                           base, exc, job_id)
            return None

    worker_version = str(body.get("pkg_version") or "")
    if worker_version and worker_version != central_version:
        logger.warning("studio offload VERSION SKEW: central=%s worker=%s (job %s) "
                       "— delegating anyway; behavior may differ across versions",
                       central_version, worker_version, job_id)
    logger.info("studio job %s delegated to %s (accepted=%s, position=%s, worker_pkg=%s)",
                job_id, base, body.get("accepted"), body.get("position"),
                worker_version or "?")

    # ---- poll to settlement ----------------------------------------------------
    # TWO clocks: the RENDER budget clock starts only at the RUNNING transition (a
    # queued job is not charged for its wait); the OVERALL cap is the absolute ceiling
    # over queue wait + render.
    status_url = base + "/studio/render/" + job_id
    cancel_sent = False
    consecutive_errors = 0
    render_deadline = None            # set when we first observe status == "running"

    def _past_budget() -> bool:
        now = time.time()
        if now > overall_deadline:
            return True
        return render_deadline is not None and now > render_deadline

    while True:
        time.sleep(poll_s)

        # Forward a cancel INTENT once (then keep polling until the worker settles).
        if not cancel_sent:
            try:
                cancelling = media_bus.is_cancelling(job_id)
            except Exception:  # noqa: BLE001
                cancelling = False
            if cancelling:
                try:
                    _http_post_json(cancel_url, {}, timeout=_CANCEL_TIMEOUT_S)
                    logger.info("studio job %s: forwarded cancel to %s", job_id, base)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("studio job %s: cancel POST failed: %s", job_id, exc)
                cancel_sent = True

        # Poll status.
        try:
            _code, st = _http_get_json(status_url, timeout=_STATUS_TIMEOUT_S)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning("studio job %s: status poll failed (%d/%d): %s",
                           job_id, consecutive_errors, _MAX_POLL_ERRORS, exc)
            if consecutive_errors >= _MAX_POLL_ERRORS:
                return JobResult(job_id=job_id, ok=False, error=JobError(
                    code="worker_lost",
                    message=f"studio worker {base} unreachable after "
                            f"{consecutive_errors} status polls",
                    retryable=True))
            if _past_budget():
                return _timeout_result(job_id, base, cancel_url)
            continue

        status = st.get("status")

        # QUEUED (item 1): the render is WAITING behind another on the worker. Keep
        # polling, forward the queue position as progress so the console shows it, and
        # bound the wait by the OVERALL cap only (the render budget hasn't started).
        if status == "queued":
            position = st.get("position")
            try:
                media_bus.set_progress(
                    job_id, {"phase": "queued", "position": position})
            except Exception:  # noqa: BLE001
                pass
            if time.time() > overall_deadline:
                logger.warning("studio job %s queued past overall cap on %s",
                               job_id, base)
                return _timeout_result(job_id, base, cancel_url)
            continue

        # RUNNING transition: start the render budget clock (not charged for the wait).
        if status == "running" and render_deadline is None:
            render_deadline = time.time() + render_budget

        # Forward live progress best-effort (whatever the worker exposes).
        prog = st.get("progress")
        if prog is not None:
            try:
                media_bus.set_progress(job_id, prog)
            except Exception:  # noqa: BLE001
                pass

        if status in ("done", "error"):
            # Both carry a result payload; error-as-data (incl. a cancelled render,
            # which is an Err(CANCELLED) produce_clip result -> code "cancelled").
            return _payload_to_job_result(st.get("result") or {}, job_id)
        if status in (None, "unknown"):
            # The worker forgot this job (restarted between accept and now).
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="worker_lost",
                message=f"studio worker {base} no longer knows job {job_id} "
                        f"(worker restarted?)",
                retryable=True))
        # status == "running" -> keep polling until settled or timed out.
        if _past_budget():
            return _timeout_result(job_id, base, cancel_url)


def run_studio_i2v(spec, job_id: str) -> JobResult:
    """Run a studio i2v job through ``produce_clip`` and return a ``JobResult``.

    ``Ok(Artifact)`` -> ``JobResult(ok=True, outputs=(clip MediaRef,))``; the ref
    carries the clip path (uri) + resolved geometry/duration + a minted asset id.
    ``Err(StageError)`` -> ``JobResult(ok=False, error=JobError(...))``. Nothing here
    raises on an expected failure.

    When a studio GPU worker is configured (``HUGPY_STUDIO_WORKER``) and this render
    binds a real model, the ``produce_clip`` execution is DELEGATED to that worker
    (see module docstring); the ingest/error handling below is then applied to the
    worker's SHARED-path result, identical to the in-process path. A worker
    unreachable at kick-off falls back to in-process here."""
    from .. import media_bus

    # --- studio render offload (option a) --------------------------------------
    if should_delegate(spec):
        base = _studio_worker_base()
        delegated = _delegate_to_worker(base, spec, job_id)
        if delegated is not None:
            return delegated
        logger.info("studio job %s: in-process fallback (worker kick-off failed)",
                    job_id)

    # --- in-process render (historical path; unchanged semantics) --------------
    # Cooperative mid-render cancel (Task 1): thread the bus's is_cancelling poll
    # DOWN into the studio spine as a pure zero-arg probe. The studio never imports
    # media_bus — only this adapter does. A cancel makes produce_clip's runner abort
    # BEFORE writing a clip and return Err(StageError(CANCELLED)), which
    # _stage_error_to_job_error maps to JobError(code="cancelled", retryable=False).
    should_cancel = lambda: media_bus.is_cancelling(job_id)  # noqa: E731

    result = run_produce_clip(spec, should_cancel)

    if result.is_err():
        return JobResult(
            job_id=job_id,
            ok=False,
            error=_stage_error_to_job_error(result.error),
        )

    # Ok(Artifact): the clip exists (produce_clip only returns Ok with a written,
    # non-empty clip). Catalog it exactly as the movie/scene runners do — ingest
    # probes it once and mints an immutable video MediaRef carried on outputs.
    artifact = result.unwrap()
    ref = ingest(artifact.path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))
