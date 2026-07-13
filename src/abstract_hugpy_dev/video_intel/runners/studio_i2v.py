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
from dataclasses import asdict, dataclass, replace

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

# --- AUTOFIT (blank budget -> fit to the serving worker's free VRAM) ------------------
# Operator doctrine (2026-07-12): a BLANK vram budget must NOT be a guaranteed-fail low
# guess ("if a model needs 14GB and it's blank, just do 14, otherwise a fail is 100%
# likely"). When a spec's ``vram_budget_gb`` is None, ``render_clip`` resolves the target
# worker via the pluggable seam and sizes the routing budget to that worker's MEASURED
# free VRAM (freshest heartbeat), minus a safety margin.
#
# SAFETY MARGIN: a render's real footprint runs a bit OVER the router's planning estimate
# (activations, allocator fragmentation, other procs / the OS on the card), so hold back
# the LARGER of 10% or 2GB — 2GB dominates on small cards, 10% dominates on big ones.
_AUTOFIT_MARGIN_FRACTION = 0.10
_AUTOFIT_MARGIN_FLOOR_GB = 2.0
# ``gpus[].memory_free`` is stored in BYTES (nvidia-smi MiB * 1024*1024, see
# _platform.hardware.detect_gpus), so convert with GiB (1024**3) — the honest inverse of
# the MiB source and the more conservative GB number (won't over-state the budget).
_BYTES_PER_GIB = 1024 ** 3
# No worker / no VRAM data -> today's default (0.5 => synthetic-if-available). Mirrors
# studio.job._DEFAULT_VRAM_BUDGET_GB so an unresolvable autofit degrades EXACTLY to the
# historical blank behavior (the UI banner already warns loudly), never a new breakage.
_AUTOFIT_FALLBACK_BUDGET_GB = 0.5


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
# ClipOutcome — the SHARED render currency of ``render_clip`` (below), consumed by
# BOTH the single-clip bus adapter (``run_studio_i2v``) and each studio-movie segment
# (``runners.studio_movie``). A studio render settles as EITHER a produced clip (its
# content-addressed SHARED path + geometry) OR an already-translated bus ``JobError``.
# The in-process ``produce_clip`` Result AND a delegated worker payload both normalize
# INTO this one type, so the two callers read an identical shape and each does its own
# catalog/record: the single-clip adapter ``ingest``s the path into a JobResult; the
# movie reads ``frames`` for its splice/trim math, records the node, and ingests too.
# On failure ``error`` is ALWAYS a bus JobError (the StageError->JobError translation
# happened at the single boundary already: in-process via ``_stage_error_to_job_error``,
# delegated on the WORKER via ``artifact_result_to_payload``), never a studio StageError.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClipOutcome:
    ok: bool
    # produced-clip fields (ok=True; None on failure). ``path`` is the SHARED
    # content-addressed clip the caller ingests; the geometry is what a caller needs
    # without re-probing (the movie's branch/trim math reads ``frames``).
    path: "str | None" = None
    content_hash: "str | None" = None
    frames: "int | None" = None
    width: "int | None" = None
    height: "int | None" = None
    duration_s: "float | None" = None
    resumed: bool = False
    # failure field (ok=False): the already-translated bus JobError (retryable set).
    error: "JobError | None" = None
    # AUTOFIT provenance (honesty in the artifact). ``effective_budget_gb`` is the vram
    # budget the router actually resolved for this render (== the explicit budget, OR the
    # worker's measured-free-minus-margin when the spec budget was blank/None), and
    # ``budget_source`` labels HOW it was chosen: ``"explicit"`` (a pinned number),
    # ``"autofit:<worker>"`` (sized to a named worker's free VRAM), or ``"autofit:fallback"``
    # (no worker / no VRAM data -> today's synthetic default). Stamped by ``render_clip`` on
    # BOTH ok and err outcomes; the movie runner records them per segment in movie.json.
    # Defaulted so every other construction site (worker-payload normalize, in-process) is
    # unchanged — render_clip fills them via dataclasses.replace.
    effective_budget_gb: "float | None" = None
    budget_source: "str | None" = None


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
        # VACE-EXTEND: the parent clip's trailing context frames (studio-movie splice
        # motion-carry). Carried into the manifest; the VACE runner builds the video+mask
        # extend idiom from them. getattr keeps older spec dicts (no field) working.
        vace_context_frames=getattr(spec, "vace_context_frames", None),
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


def _payload_to_clip_outcome(payload: dict) -> ClipOutcome:
    """Normalize a worker render payload (from ``artifact_result_to_payload``) into a
    ``ClipOutcome`` WITHOUT ingesting — the caller (single-clip or movie) does its own
    catalog. Ok -> the SHARED clip path + geometry; Err -> the rebuilt JobError (the
    worker already ran ``_stage_error_to_job_error``, so this is a verbatim rebuild, no
    re-translation). A malformed/empty payload is itself errors-as-data (retryable
    internal), and an ok-without-path never becomes a false success."""
    if not isinstance(payload, dict) or "ok" not in payload:
        return ClipOutcome(ok=False, error=JobError(
            code="internal",
            message="studio worker returned no render result payload",
            retryable=True))
    if not payload.get("ok"):
        err = payload.get("error") or {}
        return ClipOutcome(ok=False, error=JobError(
            code=err.get("code", "internal"),
            message=err.get("message", "studio worker render failed"),
            retryable=bool(err.get("retryable", False))))
    path = payload.get("path")
    if not path:
        return ClipOutcome(ok=False, error=JobError(
            code="internal",
            message="studio worker reported ok but no clip path",
            retryable=True))
    return ClipOutcome(
        ok=True,
        path=path,
        content_hash=payload.get("content_hash"),
        frames=payload.get("frames"),
        width=payload.get("width"),
        height=payload.get("height"),
        duration_s=payload.get("duration_s"),
        resumed=bool(payload.get("resumed", False)))


def _payload_to_job_result(payload: dict, job_id: str) -> JobResult:
    """Turn a worker render payload (from ``artifact_result_to_payload``) into a
    ``JobResult``, applying the EXACT post-``produce_clip`` handling the in-process
    path uses: Ok -> ``ingest`` the SHARED clip path into the media store (mints the
    video MediaRef); Err -> rebuild the JobError. Central-only (it does the
    ingest). A malformed/empty payload is itself errors-as-data. Thin wrapper over
    ``_payload_to_clip_outcome`` so the payload->JobError classification lives in ONE
    place (this shape is still exercised directly by the offload conformance test)."""
    outcome = _payload_to_clip_outcome(payload)
    if not outcome.ok:
        return JobResult(job_id=job_id, ok=False, error=outcome.error)
    # SHARED filesystem: central + worker both see /mnt/llm_storage, so the clip the
    # worker wrote is ingestable here directly — identical to the in-process path.
    ref = ingest(outcome.path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))


# --------------------------------------------------------------------------- #
# Delegation decision + HTTP loop (central side)
# --------------------------------------------------------------------------- #
def _studio_worker_base() -> str:
    return (os.environ.get(_WORKER_ENV) or "").strip().rstrip("/")


def resolve_studio_worker(spec) -> str:
    """PLUGGABLE worker-RESOLUTION seam: the base URL of the studio GPU worker THIS
    render should delegate to (``""`` = none -> in-process). The single place the
    "which worker" question is answered, so both ``should_delegate`` and ``render_clip``
    agree, and a future resolver DROPS IN here without reshaping the delegation helper.

    TODAY: returns the ``HUGPY_STUDIO_WORKER`` env target (one global studio worker).

    OPERATOR-DIRECTED TARGET ARCHITECTURE (2026-07-12): delegation targeting is NOT a
    studio-specific concern — it belongs to the STANDARD model-routing layer. Central
    has no studio models ASSIGNED to it, so a render that binds a real studio model
    (wan / vace / ltx) should route to the worker that OWNS that model, via the registry
    (``workers_for_model(bound_model_id, capability)``, honoring the capability + the
    workers' in-flight gates). Those studio models are not yet first-class registry
    entries (a FOLLOW-UP slice registers them); once they are, the registry-based
    resolver replaces the body HERE and ``HUGPY_STUDIO_WORKER`` DEMOTES to an override /
    fallback (e.g. ``return _studio_worker_base() or _registry_worker(spec)``). Kept a
    pure function OF THE SPEC so that resolver can read the bound model + capability off
    it with no signature change to this seam or its callers."""
    return _studio_worker_base()


def _url_host(url: str) -> str:
    """Lowercased host of ``url`` (no port/scheme). The studio worker base
    (``HUGPY_STUDIO_WORKER``, a studio-render URL) and a registry row's ``url`` (the
    agent URL) share a HOST but may differ in port/scheme, so autofit maps the studio
    worker to its registry row by host alone."""
    from urllib.parse import urlparse
    if not url:
        return ""
    u = url if "://" in url else "http://" + url
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _autofit_from_worker(base: str) -> "tuple[float, str] | None":
    """``(effective_budget_gb, worker_name)`` for the registry worker whose ``url`` host
    matches ``base`` — its MEASURED free VRAM minus the safety margin — or ``None`` when no
    worker matches / it reports no usable VRAM (autofit then falls back to the default).

    The free-VRAM source is the worker record's per-GPU ``gpus[].memory_free`` (bytes,
    freshest heartbeat). A single render binds ONE device, so we size to the LARGEST single
    GPU's free VRAM (== ``gpus[0]`` on the single-GPU boxes today), never the multi-GPU sum
    (which would over-state what one render can use). Lazy import of central's worker store:
    the studio spine stays boot-cheap and ``runners.scene`` already crosses this exact
    boundary; any import failure (e.g. a pure worker-side context) degrades to None."""
    host = _url_host(base)
    if not host:
        return None
    try:
        from ...flask_app.app.functions.imports.utils.workers import list_workers
    except Exception:  # noqa: BLE001
        return None
    try:
        workers = list_workers()
    except Exception:  # noqa: BLE001
        return None
    for w in workers:
        if _url_host(w.get("url") or "") != host:
            continue
        gpus = [g for g in (w.get("gpus") or []) if isinstance(g, dict)]
        frees = [g.get("memory_free") for g in gpus
                 if isinstance(g.get("memory_free"), (int, float)) and g.get("memory_free") > 0]
        if not frees:
            return None      # matched the box but it reports no VRAM -> fall back
        free_gib = max(frees) / _BYTES_PER_GIB
        margin = max(free_gib * _AUTOFIT_MARGIN_FRACTION, _AUTOFIT_MARGIN_FLOOR_GB)
        effective = free_gib - margin
        if effective <= 0:
            return None      # a card with less free than the margin -> fall back
        return effective, str(w.get("name") or w.get("id") or "worker")
    return None              # no registry row for this worker URL -> fall back


def _resolve_autofit(spec) -> "tuple[object, float, str]":
    """Resolve a BLANK (None) ``vram_budget_gb`` to a concrete routing budget, returning
    ``(spec, effective_budget_gb, budget_source)``.

    * An EXPLICIT budget is a passthrough — the spec is returned UNCHANGED (byte-identical
      delegation payload + routing), ``budget_source == "explicit"``.
    * A None budget is AUTOFIT: resolve the target worker via the pluggable seam and size
      the budget to its measured free VRAM (minus margin). No worker resolvable / no VRAM
      data (including an IN-PROCESS render on the GPU-less control-plane central, where the
      seam yields no worker) -> the historical default (0.5 => synthetic-if-available),
      ``budget_source == "autofit:fallback"``. Otherwise ``"autofit:<worker>"``.

    The None is resolved to a NUMBER here, BEFORE any router probe (``resolves_to_real_model``
    / ``produce_clip`` both key on ``vram_budget_gb``), via ``dataclasses.replace`` so the
    frozen spec carried onward (and delegated to the worker) holds the concrete budget."""
    if spec.vram_budget_gb is not None:
        return spec, float(spec.vram_budget_gb), "explicit"
    base = resolve_studio_worker(spec)
    resolved = _autofit_from_worker(base) if base else None
    if resolved is None:
        eff = _AUTOFIT_FALLBACK_BUDGET_GB
        logger.info("studio autofit: no worker VRAM for %r -> fallback budget %.2fGB "
                    "(synthetic-if-available)", base or "(no worker)", eff)
        return replace(spec, vram_budget_gb=eff), eff, "autofit:fallback"
    eff, name = resolved
    logger.info("studio autofit: sized budget to %.2fGB from worker %s free VRAM (- margin)",
                eff, name)
    return replace(spec, vram_budget_gb=eff), eff, f"autofit:{name}"


def _wants_remote(spec) -> bool:
    """The render-side half of the delegation decision (worker-INDEPENDENT): the
    TEST-ONLY force-remote override, else whether the spec binds a REAL (non-synthetic)
    model. Factored so ``should_delegate`` and ``render_clip`` share ONE rule."""
    if os.environ.get(_FORCE_REMOTE_ENV) == "1":
        return True
    return resolves_to_real_model(spec)


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
    """True iff this render should be sent to a studio GPU worker: the resolver yields
    a worker target AND (the force-remote test override is on OR the spec binds a real
    model). No worker resolved => never delegate (in-process, unchanged). ``render_clip``
    resolves the worker ONCE itself (this predicate is the public/tested decision rule).

    AUTOFIT: a blank (None) budget is resolved to a concrete number FIRST so the real-model
    probe sees a routable budget (an explicit budget is unchanged — the passthrough)."""
    resolved, _eff, _src = _resolve_autofit(spec)
    return bool(resolve_studio_worker(resolved)) and _wants_remote(resolved)


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


def _timeout_outcome(render_id: str, base: str, cancel_url: str) -> ClipOutcome:
    """Overall wall-clock budget exceeded: best-effort tell the worker to stop, then
    return a retryable delegation_timeout ``ClipOutcome`` (never a stuck 'running' row).
    The delegation loop returns this so BOTH callers settle it their own way."""
    try:
        _http_post_json(cancel_url, {}, timeout=_CANCEL_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        pass
    logger.warning("studio render %s exceeded delegation timeout on %s", render_id, base)
    return ClipOutcome(ok=False, error=JobError(
        code="delegation_timeout",
        message=f"studio render on {base} exceeded the delegation timeout",
        retryable=True))


def _timeout_result(job_id: str, base: str, cancel_url: str) -> JobResult:
    """JobResult wrapper over ``_timeout_outcome`` (retained public shape). Overall
    wall-clock budget exceeded -> best-effort stop the worker + a retryable JobError."""
    outcome = _timeout_outcome(job_id, base, cancel_url)
    return JobResult(job_id=job_id, ok=False, error=outcome.error)


def _delegate_to_worker(base: str, spec, render_id: str, *,
                        should_cancel=None, progress_sink=None):
    """Delegate the render to the studio worker at ``base`` and settle it.

    Returns a ``ClipOutcome`` once the remote render settles (done/error/cancelled/
    timeout/worker-lost/queue-full), OR ``None`` to signal "kick-off failed, fall
    back to the in-process path" — used ONLY before the worker has accepted the job
    (an unreachable worker at kick-off is exactly the scout's "worker unreachable ->
    current behavior"). Once the worker returns 202 we never fall back (that would
    risk a double render); a later failure becomes a retryable JobError.

    ``render_id`` is the WORKER-SIDE render key (in the POST body + status/cancel URLs)
    — for a single-clip job it IS the bus job_id; for a MOVIE SEGMENT it is a per-
    segment id (a movie posts many renders under its one bus job, and the worker keys
    by this id, so each segment must be distinct or the worker would dedup them as one
    "exists" job). ``should_cancel`` (the bus is_cancelling probe on the OWNING job)
    and ``progress_sink`` (where a queued-position / live-progress blob is emitted) are
    INJECTED so the single-clip path forwards to ``media_bus`` on its own job while the
    movie relays cancel from the MOVIE job and nests each segment's progress. Both
    default to ``media_bus`` on ``render_id`` (the historical single-clip behavior),
    so an existing 3-arg caller is unchanged.

    QUEUE (item 1): a busy worker now QUEUES a second render (202 accepted="queued"
    + position) rather than 409ing, so the delegation just polls it through. A 409
    means the worker's bounded queue is FULL — worker_busy now means WAIT, so we
    retry kick-off with backoff inside the OVERALL cap, only failing worker_busy when
    that window expires. KICK-OFF RESET RETRY (item 2): a ConnectionReset/refused at
    kick-off (post-restart socket window) retries for ~30s before falling back."""
    from .. import media_bus

    # Injected cancel/progress hooks; default to media_bus on render_id (single-clip).
    if should_cancel is None:
        should_cancel = lambda: media_bus.is_cancelling(render_id)  # noqa: E731
    if progress_sink is None:
        progress_sink = lambda blob: media_bus.set_progress(render_id, blob)  # noqa: E731

    spec_dict = asdict(spec)
    central_version = _pkg_version()
    start_payload = {"job_id": render_id, "spec": spec_dict,
                     "central_version": central_version}

    poll_s = _float_env(_POLL_ENV, _DEFAULT_POLL_S)
    render_budget = _float_env(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_S)
    overall_cap = _float_env(_OVERALL_CAP_ENV, _DEFAULT_OVERALL_CAP_S)
    reset_window = _float_env(_KICKOFF_RETRY_WINDOW_ENV, _DEFAULT_KICKOFF_RETRY_WINDOW_S)
    retry_interval = _float_env(_KICKOFF_RETRY_INTERVAL_ENV, _DEFAULT_KICKOFF_RETRY_INTERVAL_S)

    started_at = time.time()
    overall_deadline = started_at + overall_cap    # whole-delegation ceiling
    reset_deadline = started_at + reset_window      # connection-reset retry window
    cancel_url = base + "/studio/cancel/" + render_id

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
                                   "render %s", base, render_id)
                    return ClipOutcome(ok=False, error=JobError(
                        code="worker_busy",
                        message=f"studio worker queue full past the delegation "
                                f"window: {base}",
                        retryable=True))
                logger.info("studio worker %s queue full (409) — retrying kick-off "
                            "in %.1fs (render %s)", base, retry_interval, render_id)
                time.sleep(retry_interval)
                continue
            # Any other HTTP error at kick-off: the worker never started this render,
            # so fall back to the in-process path (graceful NO_GPU/DEPS on central).
            logger.warning("studio worker %s rejected /studio/render (HTTP %s) — "
                           "falling back in-process for render %s", base, exc.code, render_id)
            return None
        except (urllib.error.URLError, OSError) as exc:
            # ConnectionReset/refused/unreachable (e.g. the post-restart converge
            # socket window, seen twice live): retry within the reset window, then
            # fall back in-process (the scout's "worker unreachable" behavior).
            if time.time() >= reset_deadline:
                logger.warning("studio worker %s unreachable at kick-off past retry "
                               "window (%s: %s) — falling back in-process for render %s",
                               base, type(exc).__name__, exc, render_id)
                return None
            logger.info("studio worker %s kick-off connection error (%s: %s) — "
                        "retrying in %.1fs (render %s)",
                        base, type(exc).__name__, exc, retry_interval, render_id)
            time.sleep(retry_interval)
            continue
        except ValueError as exc:
            # Malformed response body — not transient; fall back in-process.
            logger.warning("studio worker %s returned an unparseable kick-off body "
                           "(%s) — falling back in-process for render %s",
                           base, exc, render_id)
            return None

    worker_version = str(body.get("pkg_version") or "")
    if worker_version and worker_version != central_version:
        logger.warning("studio offload VERSION SKEW: central=%s worker=%s (render %s) "
                       "— delegating anyway; behavior may differ across versions",
                       central_version, worker_version, render_id)
    logger.info("studio render %s delegated to %s (accepted=%s, position=%s, worker_pkg=%s)",
                render_id, base, body.get("accepted"), body.get("position"),
                worker_version or "?")

    # ---- poll to settlement ----------------------------------------------------
    # TWO clocks: the RENDER budget clock starts only at the RUNNING transition (a
    # queued job is not charged for its wait); the OVERALL cap is the absolute ceiling
    # over queue wait + render.
    status_url = base + "/studio/render/" + render_id
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
        # ``should_cancel`` probes the OWNING bus job (the movie relays the MOVIE job's
        # cancel to the per-segment worker render; single-clip probes its own job).
        if not cancel_sent:
            try:
                cancelling = should_cancel()
            except Exception:  # noqa: BLE001
                cancelling = False
            if cancelling:
                try:
                    _http_post_json(cancel_url, {}, timeout=_CANCEL_TIMEOUT_S)
                    logger.info("studio render %s: forwarded cancel to %s", render_id, base)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("studio render %s: cancel POST failed: %s", render_id, exc)
                cancel_sent = True

        # Poll status.
        try:
            _code, st = _http_get_json(status_url, timeout=_STATUS_TIMEOUT_S)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning("studio render %s: status poll failed (%d/%d): %s",
                           render_id, consecutive_errors, _MAX_POLL_ERRORS, exc)
            if consecutive_errors >= _MAX_POLL_ERRORS:
                return ClipOutcome(ok=False, error=JobError(
                    code="worker_lost",
                    message=f"studio worker {base} unreachable after "
                            f"{consecutive_errors} status polls",
                    retryable=True))
            if _past_budget():
                return _timeout_outcome(render_id, base, cancel_url)
            continue

        status = st.get("status")

        # QUEUED (item 1): the render is WAITING behind another on the worker. Keep
        # polling, forward the queue position as progress so the console shows it, and
        # bound the wait by the OVERALL cap only (the render budget hasn't started).
        if status == "queued":
            position = st.get("position")
            try:
                progress_sink({"phase": "queued", "position": position})
            except Exception:  # noqa: BLE001
                pass
            if time.time() > overall_deadline:
                logger.warning("studio render %s queued past overall cap on %s",
                               render_id, base)
                return _timeout_outcome(render_id, base, cancel_url)
            continue

        # RUNNING transition: start the render budget clock (not charged for the wait).
        if status == "running" and render_deadline is None:
            render_deadline = time.time() + render_budget

        # Forward live progress best-effort (whatever the worker exposes).
        prog = st.get("progress")
        if prog is not None:
            try:
                progress_sink(prog)
            except Exception:  # noqa: BLE001
                pass

        if status in ("done", "error"):
            # Both carry a result payload; error-as-data (incl. a cancelled render,
            # which is an Err(CANCELLED) produce_clip result -> code "cancelled").
            return _payload_to_clip_outcome(st.get("result") or {})
        if status in (None, "unknown"):
            # The worker forgot this render (restarted between accept and now).
            return ClipOutcome(ok=False, error=JobError(
                code="worker_lost",
                message=f"studio worker {base} no longer knows render {render_id} "
                        f"(worker restarted?)",
                retryable=True))
        # status == "running" -> keep polling until settled or timed out.
        if _past_budget():
            return _timeout_outcome(render_id, base, cancel_url)


def render_clip(spec, *, render_id: str, should_cancel=None, progress_sink=None,
                produce=None) -> ClipOutcome:
    """Render ONE studio clip for ``spec`` — DELEGATED to a studio GPU worker when the
    resolver yields a target AND the spec binds a real model, else IN-PROCESS — and
    return a normalized ``ClipOutcome``. THE shared render primitive: both the single-
    clip bus adapter (``run_studio_i2v``) and each studio-movie segment call this so the
    delegate-or-inline decision LADDER, the delegation loop, and the error translation
    live in exactly one place.

    The decision ladder (identical to the single-clip path's historical rule):
      * resolver yields no worker (``HUGPY_STUDIO_WORKER`` unset today) -> in-process.
      * spec binds a SYNTHETIC / ffmpeg / unroutable model -> in-process (central
        renders those fine; force-remote test override still delegates a synthetic).
      * spec binds a REAL model AND a worker resolves -> DELEGATE (poll, forward
        progress, relay cancel, ingest-from-shared-path via the SHARED filesystem).
      * worker unreachable AT KICK-OFF -> in-process fallback (the graceful NO_GPU/DEPS
        preflight on central). A worker that dies AFTER accepting -> a retryable
        ClipOutcome error (worker_lost / delegation_timeout), never a double render.

    ``render_id`` keys the render on the WORKER (single-clip: the bus job_id; a movie
    segment: a distinct per-segment id). ``should_cancel`` is the cooperative-cancel
    probe (threaded into ``produce_clip`` in-process AND relayed to the worker when
    delegating). ``progress_sink`` receives queued-position / live-progress blobs (the
    movie nests these per segment). ``produce`` is the in-process render function
    (defaults to ``run_produce_clip``); it is injectable so the movie's module-level
    render seam — which its tests patch — stays the inline execution point. All three
    default to the single-clip behavior (media_bus on ``render_id`` / ``run_produce_clip``)."""
    from .. import media_bus

    if should_cancel is None:
        should_cancel = lambda: media_bus.is_cancelling(render_id)  # noqa: E731
    if progress_sink is None:
        progress_sink = lambda blob: media_bus.set_progress(render_id, blob)  # noqa: E731
    if produce is None:
        produce = run_produce_clip

    # --- AUTOFIT: resolve a BLANK (None) budget to a concrete one BEFORE any router
    #     probe / produce_clip (both key on vram_budget_gb). An explicit budget is a
    #     passthrough (spec unchanged). The resolved (budget, source) is stamped onto the
    #     returned ClipOutcome so the movie runner can record it honestly in movie.json.
    spec, eff_gb, budget_source = _resolve_autofit(spec)

    def _stamp(outcome: ClipOutcome) -> ClipOutcome:
        return replace(outcome, effective_budget_gb=eff_gb, budget_source=budget_source)

    # --- studio render offload (option a): resolve the worker ONCE, then decide. -----
    base = resolve_studio_worker(spec)
    if base and _wants_remote(spec):
        outcome = _delegate_to_worker(
            base, spec, render_id,
            should_cancel=should_cancel, progress_sink=progress_sink)
        if outcome is not None:
            return _stamp(outcome)    # settled remotely (never fall back after 202)
        logger.info("studio render %s: in-process fallback (worker kick-off failed)",
                    render_id)

    # --- in-process render (historical path; unchanged semantics) --------------
    # Cooperative mid-render cancel (Task 1): the studio never imports media_bus — only
    # this adapter does — so the zero-arg ``should_cancel`` probe is threaded DOWN into
    # the spine. A cancel makes produce_clip's runner abort BEFORE writing a clip and
    # return Err(StageError(CANCELLED)) -> JobError(code="cancelled", retryable=False).
    # NOTE: on the GPU-less control-plane central this is where autofit's fallback budget
    # (0.5) lands — the synthetic path, unchanged (a real model there returns NO_GPU as data).
    result = produce(spec, should_cancel)
    if result.is_err():
        # ONE boundary: studio StageError -> bus JobError (the delegated path did the
        # identical translation on the worker, so ClipOutcome.error is always a JobError).
        return _stamp(ClipOutcome(ok=False, error=_stage_error_to_job_error(result.error)))
    art = result.unwrap()
    return _stamp(ClipOutcome(
        ok=True, path=art.path, content_hash=art.content_hash, frames=art.frames,
        width=art.width, height=art.height, duration_s=art.duration_s,
        resumed=art.resumed))


def run_studio_i2v(spec, job_id: str) -> JobResult:
    """Run a studio i2v job through ``produce_clip`` and return a ``JobResult``.

    ``Ok(Artifact)`` -> ``JobResult(ok=True, outputs=(clip MediaRef,))``; the ref
    carries the clip path (uri) + resolved geometry/duration + a minted asset id.
    ``Err(StageError)`` -> ``JobResult(ok=False, error=JobError(...))``. Nothing here
    raises on an expected failure.

    Delegation, the in-process fallback, cancel + progress forwarding, and the
    StageError->JobError translation all live in the shared ``render_clip`` (the bus
    job_id is the worker render key); this adapter just settles the ``ClipOutcome``
    into a ``JobResult`` — ``ingest``ing the SHARED clip path on Ok exactly as before."""
    outcome = render_clip(spec, render_id=job_id)
    if not outcome.ok:
        return JobResult(job_id=job_id, ok=False, error=outcome.error)
    # Ok: the clip exists on the SHARED store (in-process or worker-written). Catalog it
    # exactly as the movie/scene runners do — ingest probes it once and mints a video
    # MediaRef carried on outputs.
    ref = ingest(outcome.path, kind_hint="video")
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))
