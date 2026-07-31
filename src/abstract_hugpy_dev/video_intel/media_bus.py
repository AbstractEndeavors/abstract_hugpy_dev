"""Durable, multi-process-safe job bus (stdlib sqlite3, WAL).

State machine (map §6):  queued -> claimed -> running -> done | failed

Discipline:
  * Enqueue path:  spec -> serialize -> insert 'queued' -> return job_id.
  * Worker path:   claim (atomic cross-process) -> run pure runner -> write once.
  * Single writer: every state write is gated on `WHERE claim_token=?`, so only
    the claiming worker mutates a given job_id.
  * Errors are DATA: a runner returns JobError inside JobResult; only this loop
    (run_claimed) catches an UNEXPECTED raise and converts it to a JobResult.

Spec (de)serialization is keyed by JobSpec.name so Phase 3 routes reuse it.
`start_worker_daemon()` is DEFINED but never called at import — Phase 3 wires it
at app init.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT

from .audio_schema import make_audio_extract
from .crop_schema import CropSpec, SpatialRegion, TemporalRegion, make_crop
from .frame_schema import make_frame_extract
from .gen_schema import GenPromptPart, make_generate_image
from .job_schema import JOB_REGISTRY
from .media_schema import make_media_ref
from .movie_schema import GoalInterval, make_movie
from .scene_schema import make_generate_scene
from .studio.job import studio_i2v_from_dict
from .studio_movie_schema import studio_movie_from_dict
from .identity_reconstruction_schema import (
    identity_reconstruction_from_dict,
    identity_mesh_from_dict,
)
from .identity_video_extract_schema import identity_video_extract_from_dict
from .mlt_render_schema import mlt_render_from_dict
from .result_schema import JobResult
from .runners import DISPATCH

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(DEFAULT_ROOT, "video_intel", "media_jobs.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS media_jobs (
    job_id       TEXT PRIMARY KEY,
    name         TEXT,
    status       TEXT,
    spec_json    TEXT,
    result_json  TEXT,
    claim_token  TEXT,
    created      REAL,
    updated      REAL,
    progress_json TEXT,
    stage_log_json TEXT
)
"""

_init_lock = threading.Lock()
_initialized = False


# --------------------------------------------------------------------------- #
# spec (de)serialization — keyed by JobSpec.name (Phase 3 reuses these)
# --------------------------------------------------------------------------- #
def _crop_from_dict(d: dict) -> CropSpec:
    """Rebuild a CropSpec from its asdict() form, through the validating
    factories (make_media_ref + make_crop) so invariants are re-checked."""
    src_d = d["source"]
    source = make_media_ref(**src_d)
    sp = d.get("spatial")
    tp = d.get("temporal")
    spatial = SpatialRegion(**sp) if sp is not None else None
    temporal = TemporalRegion(**tp) if tp is not None else None
    return make_crop(source=source, spatial=spatial, temporal=temporal)


def _frame_extract_from_dict(d: dict):
    """Rebuild a FrameExtractSpec from its asdict() form, through the validating
    factories (make_media_ref + optional TemporalRegion + make_frame_extract)."""
    source = make_media_ref(**d["source"])
    win = d.get("window")
    window = TemporalRegion(**win) if win is not None else None
    return make_frame_extract(
        source=source,
        fps=d["fps"],
        quality=d["quality"],
        fmt=d["fmt"],
        window=window,
        max_frames=d.get("max_frames"),
    )


def _audio_extract_from_dict(d: dict):
    """Rebuild an AudioExtractSpec from its asdict() form, through the validating
    factories (make_media_ref + make_audio_extract)."""
    source = make_media_ref(**d["source"])
    return make_audio_extract(source=source, fmt=d["fmt"])


def _generate_image_from_dict(d: dict):
    """Rebuild a GenerateImageSpec from its asdict() form, through the validating
    factories. Each part's media (if any) round-trips via make_media_ref."""
    parts = []
    for pd in d["parts"]:
        media_d = pd.get("media")
        media = make_media_ref(**media_d) if media_d is not None else None
        parts.append(GenPromptPart(kind=pd["kind"], text=pd.get("text"), media=media))
    return make_generate_image(
        parts=tuple(parts),
        model_id=d["model_id"],
        width=d["width"],
        height=d["height"],
        steps=d["steps"],
        guidance=d["guidance"],
        seed=d.get("seed"),
        negative=d.get("negative"),
        strength=d.get("strength"),   # img2img (additive; v1 payloads omit it)
        project=d.get("project"),     # auto-archive NAME (additive; may be absent)
    )


def _generate_scene_from_dict(d: dict):
    """Rebuild a GenerateSceneSpec from its asdict() form, through the validating
    factories. Each part's media (if any) round-trips via make_media_ref; scene
    fields (n_frames/fps/assemble/seed/motion/negative) are carried through."""
    parts = []
    for pd in d["parts"]:
        media_d = pd.get("media")
        media = make_media_ref(**media_d) if media_d is not None else None
        parts.append(GenPromptPart(kind=pd["kind"], text=pd.get("text"), media=media))
    return make_generate_scene(
        parts=tuple(parts),
        model_id=d["model_id"],
        width=d["width"],
        height=d["height"],
        steps=d["steps"],
        guidance=d["guidance"],
        n_frames=d["n_frames"],
        fps=d["fps"],
        assemble=d["assemble"],
        seed=d.get("seed"),
        motion=d.get("motion"),
        negative=d.get("negative"),
        # img2img additive knobs (v1 payloads omit them -> factory defaults:
        # strength=None -> runner applies 0.45; chain defaults True).
        strength=d.get("strength"),
        chain=d.get("chain", True),
        project=d.get("project"),     # auto-archive NAME (additive; may be absent)
    )


def _generate_movie_from_dict(d: dict):
    """Rebuild a MovieSpec from its asdict() form, through the validating factory.
    Each goal round-trips into a GoalInterval (its optional ref through
    make_media_ref); the scene-template fields + director knobs (vision_enabled,
    score_threshold, max_attempts_per_segment, judge_model_id, time_budget_s) are
    carried through so a re-enqueue (RESUME) rebuilds an identical spec."""
    goals = []
    for gd in d["goals"]:
        ref_d = gd.get("ref")
        ref = make_media_ref(**ref_d) if ref_d is not None else None
        goals.append(GoalInterval(
            start_frame=gd["start_frame"],
            end_frame=gd["end_frame"],
            prompt=gd["prompt"],
            ref=ref,
        ))
    return make_movie(
        goals=tuple(goals),
        model_id=d["model_id"],
        width=d["width"],
        height=d["height"],
        steps=d["steps"],
        guidance=d["guidance"],
        fps=d["fps"],
        assemble=d["assemble"],
        seed=d.get("seed"),
        negative=d.get("negative"),
        strength=d.get("strength"),
        chain=d.get("chain", True),
        project=d.get("project"),
        vision_enabled=d.get("vision_enabled", False),
        score_threshold=d.get("score_threshold", 60),
        max_attempts_per_segment=d.get("max_attempts_per_segment", 1),
        judge_model_id=d.get("judge_model_id"),
        time_budget_s=d.get("time_budget_s"),
    )


# name -> (dict -> spec). Grows as Phase 4+ specs land.
SPEC_DESERIALIZERS: Dict[str, Callable[[dict], object]] = {
    "crop": _crop_from_dict,
    "frame_extract": _frame_extract_from_dict,
    "audio_extract": _audio_extract_from_dict,
    "generate_image": _generate_image_from_dict,
    "generate_scene": _generate_scene_from_dict,
    "generate_movie": _generate_movie_from_dict,
    # B2 (closes manifest.py TODO(P0-3)): the studio i2v spec rehydrates through
    # its own validate-at-construction factory (studio.job.studio_i2v_from_dict).
    "studio_i2v": studio_i2v_from_dict,
    # Studio movie — the take-tree spec rehydrates through its own
    # validate-at-construction factory (studio_movie_schema.studio_movie_from_dict).
    "generate_studio_movie": studio_movie_from_dict,
    # Identity reconstruction (studio stage (b)) — rehydrates through its own
    # validate-at-construction factory (identity_reconstruction_from_dict).
    "identity_reconstruction": identity_reconstruction_from_dict,
    # Identity 3D mesh build (+ turntable) RELAY — rehydrates through its own
    # validate-at-construction factory (identity_mesh_from_dict). The runner relays it
    # to the remote GPU render service (central has no GPU).
    "identity_mesh_build": identity_mesh_from_dict,
    # Identity VIDEO-EXTRACT (char360) RELAY — rehydrates through its own validate-at-
    # construction factory (identity_video_extract_from_dict). The runner relays the source
    # video to the remote GPU render service, then writes the per-character view-sets back
    # into identity profiles (central has no GPU + never runs char360).
    "identity_video_extract": identity_video_extract_from_dict,
    # MLT/Kdenlive headless render (k22) — rehydrates through its own validate-at-construction
    # factory (mlt_render_from_dict). The runner path-maps the project + renders it with melt.
    "mlt_render": mlt_render_from_dict,
}


def serialize_spec(name: str, spec) -> str:
    """Frozen spec -> json string. Nested MediaRef/regions serialize via asdict.

    `name` is accepted for symmetry with deserialize_spec / Phase 3 routing;
    asdict is generic so it is not needed to encode, but keeping the signature
    keyed by name keeps the (de)serialize pair a matched, registry-driven set.
    """
    return json.dumps(asdict(spec))


def deserialize_spec(name: str, d: dict):
    """json-dict -> frozen spec, via the per-name registry (registry-driven)."""
    try:
        builder = SPEC_DESERIALIZERS[name]
    except KeyError:
        raise KeyError(f"no spec deserializer registered for job name {name!r}")
    return builder(d)


def serialize_result(result: JobResult) -> str:
    return json.dumps(asdict(result))


# --------------------------------------------------------------------------- #
# connection / schema
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _connect_ro() -> sqlite3.Connection:
    """A READ-ONLY handle for the observability reads (get / list_jobs) — k57.

    Opened ``mode=ro`` so the connection physically cannot take a write lock, with
    a SHORT busy timeout: a panel poll must degrade to "no rows this tick" rather
    than sit for 30s behind a renderer's write. WAL lets this reader run fully
    concurrently with the heartbeating writers — it never blocks them and they
    never block it. Falls back to the read/write handle when the DB file does not
    exist yet (a first-boot read before any enqueue)."""
    try:
        conn = sqlite3.connect("file:" + DB_PATH + "?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    except sqlite3.OperationalError:
        return _connect()


def _ensure_db() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        conn = _connect()
        try:
            conn.execute(_CREATE_SQL)
            # Idempotent migration: DBs created before the live-progress feature
            # lack progress_json. ADD COLUMN is a no-op-or-raise on re-run, so we
            # swallow the "duplicate column name" OperationalError.
            try:
                conn.execute("ALTER TABLE media_jobs ADD COLUMN progress_json TEXT")
            except sqlite3.OperationalError:
                pass  # column already present (fresh CREATE or prior migration)
            # Same idempotent migration, for the studio-clip ARCHIVE feature: DBs
            # created before it lack archived_at. NULL = active (listed by GET
            # /video/studio/clips); a REAL epoch timestamp = archived (hidden from
            # that list, honest 410 from the per-id serve/detail routes). The clip's
            # row and its bytes on disk are never touched by archiving — only this
            # one column flips — so a pre-feature DB just starts every existing job
            # off as active, which is the correct/only sane default.
            try:
                conn.execute("ALTER TABLE media_jobs ADD COLUMN archived_at REAL")
            except sqlite3.OperationalError:
                pass  # column already present (fresh CREATE or prior migration)
            # k9 attribution: who enqueued this job (a comms principal string —
            # "operator", "apikey:<id>", "share:<id>", …). NULL = unattributed
            # (every pre-k9 job). Resolved in the REQUEST context at enqueue and
            # persisted here so the bus->JobStore bridge can carry it to /llm/jobs
            # even though the process that RUNS the job (and fires on_running/
            # on_terminal) is frequently a different one than enqueued it.
            try:
                conn.execute("ALTER TABLE media_jobs ADD COLUMN principal TEXT")
            except sqlite3.OperationalError:
                pass  # column already present (fresh CREATE or prior migration)
            # Idempotent migration for the STAGE TIMELINE (the "what is it doing /
            # where did it fail" feature): an append-only per-job JSON list of coarse
            # stage entries ({stage, ts, ts_last, count, detail?, + terminal outcome}).
            # Unlike progress_json (the latest live blob, OVERWRITTEN each call and
            # NULLED at terminal), this is RETAINED through a done/failed/cancelled
            # terminal so the panel can show the full sequence a render DID + the exact
            # failing stage. A pre-feature DB just starts every existing row at an empty
            # (NULL) timeline, which reads as "no history recorded" — the correct default.
            try:
                conn.execute("ALTER TABLE media_jobs ADD COLUMN stage_log_json TEXT")
            except sqlite3.OperationalError:
                pass  # column already present (fresh CREATE or prior migration)
            # k57 LISTING INDEXES. The catalog table has carried only its implicit
            # PRIMARY KEY since day one, so every GET /video/jobs query was a FULL
            # SCAN + sort over the whole history — on a live central that is ~1.5k
            # rows on shared storage, several of them holding multi-MB result blobs,
            # which measured 6s (in-flight) and 21s (terminal) per call. These two
            # indexes turn both listing queries into a bounded range scan in the
            # exact (status, order-by) shape list_jobs uses. Idempotent + additive;
            # writers are unaffected beyond the usual index maintenance.
            for _idx_sql in (
                "CREATE INDEX IF NOT EXISTS idx_media_jobs_status_created "
                "ON media_jobs(status, created)",
                "CREATE INDEX IF NOT EXISTS idx_media_jobs_status_updated "
                "ON media_jobs(status, updated)",
            ):
                try:
                    conn.execute(_idx_sql)
                except sqlite3.OperationalError:
                    pass  # a locked/older DB just runs unindexed, as it did before
        finally:
            conn.close()
        _initialized = True


# --------------------------------------------------------------------------- #
# comms.JobStore bridge (A/P0-2) — one-directional (bus -> JobStore) mirror of
# media-job lifecycle so media jobs surface in GET /llm/jobs. Kept as a thin,
# lazily-imported, exception-swallowing dispatcher so this module has ZERO
# import-time coupling to comms and a bridge failure can never break a media
# job. All bridge policy lives in video_intel/job_bridge.py.
# --------------------------------------------------------------------------- #
def _bridge(fn_name: str, *args, **kwargs) -> None:
    try:
        from . import job_bridge
        getattr(job_bridge, fn_name)(*args, **kwargs)
    except Exception:
        pass  # bus -> JobStore mirror is best-effort; execution is unaffected


# --------------------------------------------------------------------------- #
# p6 GPU RESERVATION seam — heavy video tasks pre-claim the (single) video GPU
# BEFORE dispatch so a Wan/Hunyuan render doesn't collide mid-run with the LLM
# agent-brain squatting the card. Lazily imported + fully guarded so the media
# bus keeps ZERO import-time coupling to the reservation engine and a reservation
# failure can NEVER break a media job. Policy lives in video_intel/reservation/.
# --------------------------------------------------------------------------- #
def _acquire_reservation(name: str, spec, job_id: str):
    """Returns (handle, refusal_result). ``handle`` None ⇒ proceed unreserved
    (a light task, the layer off, or an infra hiccup — fail open). ``refusal_result``
    is a terminal JobResult when the engine HONESTLY refused (a measured shortfall it
    could not clear) so the run terminals as gpu_unavailable instead of OOM'ing."""
    try:
        from .reservation import acquire, ReservationRefused
    except Exception:  # noqa: BLE001 — no engine present ⇒ unreserved, exactly as before
        return None, None
    try:
        return acquire(name, spec, job_id), None
    except ReservationRefused as rr:
        from .result_schema import JobError
        return None, JobResult(
            job_id=job_id, ok=False,
            error=JobError(code="gpu_unavailable", message=str(rr), retryable=True))
    except Exception:  # noqa: BLE001 — an engine bug proceeds unreserved, never wedges dispatch
        logger.debug("reservation acquire failed — proceeding unreserved",
                     exc_info=True)
        return None, None


def _release_reservation(job_id: str) -> None:
    """Release a run's GPU claim on ANY terminal path (done/failed/cancelled/abort).
    Idempotent + best-effort — a lease TTL is the backstop for a crash."""
    try:
        from .reservation import release
        release(job_id, reason="run terminal")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# RESERVATION-GATED ADMISSION seam — the claim path probes the reservation
# engine's non-destructive fit PROBE before flipping a queued row to 'claimed'.
# A head that can't fit HOLDS (stays queued, marked awaiting_capacity) and the
# claimer may look PAST it to a later job that fits (bounded overtake). Like the
# reservation acquire/release seam above, this is lazily imported + fully guarded
# so the bus keeps ZERO import-time coupling to the reservation engine and any
# probe failure FAILS OPEN (admit) — an admission bug can never wedge dispatch.
# When the reservation layer is OFF the whole gate is a transparent no-op (pure
# FIFO — see claim_admissible's fast-path).
# --------------------------------------------------------------------------- #
def _admission_enabled() -> bool:
    """True when the reservation layer is on (so the admission gate is active).
    Fail-CLOSED to False (pure FIFO) on any import/infra problem — the gate is
    never allowed to be the thing that breaks claiming."""
    try:
        from .reservation import admission_enabled
        return bool(admission_enabled())
    except Exception:  # noqa: BLE001
        return False


def _probe_admission(name: str, job_id: str) -> Tuple[bool, Optional[dict]]:
    """(admit, reason) from the engine's non-destructive probe. Fail-OPEN
    (True, None) on any error — a probe hiccup must never HOLD a render."""
    try:
        from .reservation import can_admit
        return can_admit(name, None, run_id=job_id)
    except Exception:  # noqa: BLE001
        return True, None


def _force_admit_safe(name: str) -> bool:
    """Whether the scheduler's starvation/deadlock guard may force-admit a held
    head best-effort without colliding with an active reservation. Fail-OPEN
    True."""
    try:
        from .reservation import force_admit_safe
        return bool(force_admit_safe(name))
    except Exception:  # noqa: BLE001
        return True


# ── admission knobs (env-overridable; defaults are today's-fleet success paths) ─
def _runner_count() -> int:
    """Concurrent claim->run threads in the pool (HUGPY_MEDIA_BUS_RUNNERS, def 2).
    Heavy GPU tasks still serialize naturally via exclusive reservations; the win
    is light/CPU tasks + multi-worker fleets no longer queuing behind a render."""
    try:
        return max(1, int(os.environ.get("HUGPY_MEDIA_BUS_RUNNERS", "2")))
    except (TypeError, ValueError):
        return 2


def _lookahead() -> int:
    """How many oldest queued rows the claimer may consider — the head plus the
    overtake window (HUGPY_MEDIA_BUS_LOOKAHEAD, def 5)."""
    try:
        return max(1, int(os.environ.get("HUGPY_MEDIA_BUS_LOOKAHEAD", "5")))
    except (TypeError, ValueError):
        return 5


def _max_overtake() -> int:
    """Max times a held HEAD may be overtaken by later jobs before the claimer
    STOPS overtaking and idles until the head's capacity frees — the anti-
    starvation bound (HUGPY_MEDIA_BUS_MAX_OVERTAKE, def 8). The head still runs the
    instant its reservation fits; this only caps how long throughput may jump it."""
    try:
        return max(0, int(os.environ.get("HUGPY_MEDIA_BUS_MAX_OVERTAKE", "8")))
    except (TypeError, ValueError):
        return 8


def _load_progress(progress_json: Optional[str]) -> Optional[dict]:
    if not progress_json:
        return None
    try:
        return json.loads(progress_json)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# STAGE TIMELINE — an append-only, per-job record of the COARSE stages a render
# moved through, RETAINED through the terminal write (progress_json is the latest
# live frame blob and is nulled at terminal; this is the durable "what it did /
# where it failed" history the Active-Processes expandable renders). Each entry:
#   {stage, ts (first-seen), ts_last (last-updated), count (# set_progress hits in
#    this stage — so a 48-frame render loop is ONE 'rendering' row, not 48),
#    detail? (a short human line), + terminal outcome on the final entry}.
# Everything here is BEST-EFFORT / never-fatal (mirrors set_progress): a timeline
# hiccup can never fail a render — every call site wraps these in try/except.
# --------------------------------------------------------------------------- #
_STAGE_LOG_MAX = 60   # cap so a pathological/looping job can't grow it unbounded


def _fmt_gib(nbytes) -> Optional[str]:
    try:
        return f"{float(nbytes) / (1024 ** 3):.1f} GiB"
    except (TypeError, ValueError):
        return None


def _coarse_stage(progress) -> Optional[str]:
    """The COARSE phase key a live progress blob is in — the granularity the timeline
    dedups on. Prefers an explicit hold ``phase`` (awaiting_capacity), then the
    runner's ``stage`` (loading / branching / generating / assembling / archiving),
    else a generic 'working'. None for a non-dict/empty blob."""
    if not isinstance(progress, dict):
        return None
    ph = progress.get("phase")
    if isinstance(ph, str) and ph.strip():
        return ph
    st = progress.get("stage")
    if isinstance(st, str) and st.strip():
        return st
    return "working"


def _stage_detail(progress) -> Optional[str]:
    """A short, HONEST 'what it's doing' line for a live blob — built ONLY from fields
    the blob actually carries (never fabricated). Covers the reservation HOLD marker,
    the studio-movie nested segment blob, and the scene/frame-loop blob."""
    if not isinstance(progress, dict):
        return None
    # Reservation HOLD (awaiting_capacity) — surface the measured shortfall + overtakes.
    if progress.get("phase") == "awaiting_capacity":
        reason = progress.get("reason") if isinstance(progress.get("reason"), dict) else {}
        line = "awaiting GPU capacity"
        sb = reason.get("short_by_bytes")
        g = _fmt_gib(sb) if sb is not None else None
        if g:
            line += f" — short by {g}"
        ot = progress.get("overtaken")
        if isinstance(ot, int) and ot > 0:
            line += f" · jumped {ot}×"
        return line
    bits: List[str] = []
    sd, stot = progress.get("segment_done"), progress.get("segment_total")
    if isinstance(sd, int) and isinstance(stot, int):
        bits.append(f"segment {sd}/{stot}")
    cur = progress.get("current")
    if isinstance(cur, dict):
        cap = cur.get("capability")
        if isinstance(cap, str) and cap:
            bits.append(cap)
        # MODEL ATTRIBUTION (k58): a movie segment whose capability the movie-level pin
        # cannot serve resolves its own model, and that substitution has to be legible
        # in the TIMELINE, not just in movie.json after the render. Only shown when the
        # blob carries it (never fabricated).
        mid = cur.get("model_id")
        if isinstance(mid, str) and mid:
            bits.append(f"model {mid}")
        if cur.get("model_source") == "capability_fallback":
            pin = cur.get("pinned_model_id")
            bits.append(f"capability fallback (pin {pin} does not serve {cap})"
                        if isinstance(pin, str) and pin else "capability fallback")
        w = cur.get("worker")
        if isinstance(w, dict):
            wstage = w.get("stage") or w.get("phase")
            if isinstance(wstage, str) and wstage:
                bits.append(f"worker {wstage}")
        prm = cur.get("prompt")
        if isinstance(prm, str) and prm.strip() and not bits:
            bits.append(prm.strip()[:60])
    done, total = progress.get("done"), progress.get("total")
    if isinstance(done, int) and isinstance(total, int):
        bits.append(f"frame {done}/{total}")
    lbl = progress.get("label")
    if isinstance(lbl, str) and lbl.strip() and not bits:
        bits.append(lbl.strip()[:80])
    return " · ".join(bits) if bits else None


def _as_pair(a, b):
    """(done, total) as ints when BOTH are honest non-negative numbers with total>0,
    else None. Guards every fraction below — a bar is never drawn from half a pair."""
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if b <= 0 or a < 0:
        return None
    return (float(a), float(b))


def _within_segment_fraction(progress: dict):
    """0..1 through the CURRENT unit of work, or None when nothing reports it.

    Sources, in order: an explicit ``fraction``/``percent``; the runner's denoise
    STEP counter (``step``/``steps`` — what diffusers' callback_on_step_end feeds,
    forwarded verbatim from the worker's render blob); a frame counter. Read from
    the live blob's ``current.worker`` sub-object first (a movie segment delegated
    to the ae worker nests the worker's own blob there) and then from the top level
    (an in-process render writes it flat). NEVER fabricated: a render whose runner
    reports nothing yields None and the bar stays at the segment granularity."""
    scopes = []
    cur = progress.get("current")
    if isinstance(cur, dict):
        w = cur.get("worker")
        if isinstance(w, dict):
            scopes.append(w)
        scopes.append(cur)
    scopes.append(progress)
    for sc in scopes:
        frac = sc.get("fraction")
        if isinstance(frac, (int, float)) and not isinstance(frac, bool) and 0 <= frac <= 1:
            return float(frac)
        pct = sc.get("percent")
        if isinstance(pct, (int, float)) and not isinstance(pct, bool) and 0 <= pct <= 100:
            return float(pct) / 100.0
        for done_key, total_key in (("step", "steps"), ("done", "total"),
                                    ("frame", "frames")):
            pair = _as_pair(sc.get(done_key), sc.get(total_key))
            if pair:
                return min(1.0, pair[0] / pair[1])
    return None


def _progress_ratio(progress) -> Optional[float]:
    """The job's overall 0..1 completion, or None when the runner reports nothing
    measurable (the panel then shows no bar rather than a fake one).

    A segmented render (studio movie / scene) is ``(segments_done + within-segment
    fraction) / segment_total`` — so the bar MOVES during a long single segment
    instead of sitting at 0 until the segment lands, which is exactly the "progress
    bar never updates" the operator reported. A flat render is just its own
    within-unit fraction. A held (awaiting_capacity) job has made no progress by
    definition -> 0.0, so the bar reads honestly as "not started"."""
    if not isinstance(progress, dict):
        return None
    if progress.get("phase") == "awaiting_capacity":
        return 0.0
    seg = _as_pair(progress.get("segment_done"), progress.get("segment_total"))
    inner = _within_segment_fraction(progress)
    if seg:
        done, total = seg
        if inner is not None and done < total:
            done += inner
        return max(0.0, min(1.0, done / total))
    return inner


def _progress_detail(progress) -> Optional[dict]:
    """The NUMERIC bits behind the bar, for a panel that wants to label it:
    ``{segment_done, segment_total, step, steps, fraction}`` (omit-when-unset).
    None when the blob carries no numbers at all."""
    if not isinstance(progress, dict):
        return None
    out: dict = {}
    seg = _as_pair(progress.get("segment_done"), progress.get("segment_total"))
    if seg:
        out["segment_done"] = int(seg[0])
        out["segment_total"] = int(seg[1])
    for sc in ((progress.get("current") or {}).get("worker")
               if isinstance(progress.get("current"), dict) else None,
               progress):
        if not isinstance(sc, dict):
            continue
        pair = _as_pair(sc.get("step"), sc.get("steps"))
        if pair and "step" not in out:
            out["step"] = int(pair[0])
            out["steps"] = int(pair[1])
    frac = _within_segment_fraction(progress)
    if frac is not None:
        out["fraction"] = round(frac, 4)
    return out or None


def _load_stage_log(raw) -> list:
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except Exception:  # noqa: BLE001
        return []


def _append_stage_log(job_id: str, stage: Optional[str], detail: Optional[str],
                      terminal_extra: Optional[dict] = None) -> None:
    """Append (or COALESCE) a timeline entry for ``job_id``. Same coarse ``stage`` as
    the last entry (and NOT a terminal entry) -> bump its ``count``/``ts_last`` and
    refresh ``detail`` (a frame loop stays one row). A different stage, or a terminal
    entry, -> a new row. Un-gated (the runner owns its own job_id, like set_progress).
    Best-effort — the ONE caller-side contract is to wrap this so a DB hiccup never
    fails a render."""
    if not stage:
        return
    now = time.time()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT stage_log_json FROM media_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
        if row is None:
            return  # unknown id — nothing to record against
        log = _load_stage_log(row[0])
        if log and log[-1].get("stage") == stage and not terminal_extra:
            last = log[-1]
            last["count"] = int(last.get("count", 1)) + 1
            last["ts_last"] = now
            if detail:
                last["detail"] = detail
        else:
            entry = {"stage": stage, "ts": now, "ts_last": now, "count": 1}
            if detail:
                entry["detail"] = detail
            if terminal_extra:
                entry.update(terminal_extra)
            log.append(entry)
        if len(log) > _STAGE_LOG_MAX:
            log = log[-_STAGE_LOG_MAX:]
        conn.execute(
            "UPDATE media_jobs SET stage_log_json=? WHERE job_id=?",
            (json.dumps(log), job_id),
        )
    finally:
        conn.close()


def _record_terminal_stage(job_id: str, status: str, result) -> None:
    """Write the FINAL timeline entry carrying the outcome — RETAINED through terminal
    (progress_json is nulled, this is not). For a failure it carries the exact
    ``code``/``message``/``retryable`` AND ``failed_at_stage`` (the live stage the job
    was in when it broke — the operator's "where is it failing"). Best-effort."""
    extra: dict = {}
    detail: Optional[str] = None
    err = getattr(result, "error", None)
    if status == "done":
        detail = "completed"
    elif err is not None:
        code = getattr(err, "code", None)
        msg = getattr(err, "message", None)
        retry = getattr(err, "retryable", None)
        detail = msg or status
        extra = {"code": code, "message": msg,
                 "retryable": bool(retry) if retry is not None else None}
    else:
        detail = status
    # Which live stage were we in when the terminal hit? The last non-terminal row.
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT stage_log_json FROM media_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
    finally:
        conn.close()
    prior = [e for e in _load_stage_log(row[0] if row else None)
             if e.get("stage") not in _TERMINAL_STATES]
    if prior:
        extra["failed_at_stage"] = prior[-1].get("stage")
    _append_stage_log(job_id, status, detail, terminal_extra=extra)


def build_failure_summary(result, stage_log) -> Optional[dict]:
    """Terminal FAILURE detail for the panel: ``{stage, code, message, retryable}`` —
    the whole point of the expandable on a broken render. ``result`` is the parsed
    JobResult dict (or None); ``stage_log`` the parsed timeline list. None unless the
    job terminated NOT-ok with an error object. ``stage`` = the recorded
    ``failed_at_stage`` (where it broke), else None."""
    if not isinstance(result, dict) or result.get("ok"):
        return None
    err = result.get("error")
    if not isinstance(err, dict):
        return None
    stage = None
    for e in reversed(stage_log or []):
        if e.get("failed_at_stage"):
            stage = e.get("failed_at_stage")
            break
    retry = err.get("retryable")
    return {
        "stage": stage,
        "code": err.get("code"),
        "message": err.get("message"),
        "retryable": bool(retry) if retry is not None else None,
    }


def _last_movement_ts(stage_log, updated) -> Optional[float]:
    """The epoch ts of the most recent timeline movement — the HONEST "time since last
    movement" basis, tied to the CURRENT stage rather than a bare row clock. Falls back
    to the row's ``updated`` when the timeline is empty."""
    ts = None
    for e in (stage_log or []):
        for k in ("ts_last", "ts"):
            v = e.get(k)
            if isinstance(v, (int, float)):
                ts = v if ts is None else max(ts, v)
    return ts if ts is not None else updated


def _current_stage(stage_log) -> Optional[str]:
    """The stage a job is CURRENTLY in — the last non-terminal timeline row, or None."""
    for e in reversed(stage_log or []):
        s = e.get("stage")
        if s and s not in _TERMINAL_STATES:
            return s
    return None


def _mark_awaiting_capacity(job_id: str, reason: Optional[dict],
                            prev: Optional[dict], overtaken: int) -> None:
    """Set (or refresh) a held job's ``awaiting_capacity`` progress marker so the
    hold + reason are VISIBLE via GET /video/jobs/<id> and (through the bridge)
    /llm/jobs — mirroring the cold-hold/awaiting-load pattern. The job stays
    status='queued' (so cancel() still cancels it exactly as any queued job). To
    avoid spamming set_progress (and its bridge) every idle tick, we only write
    when the marker actually CHANGES (first hold, or the overtaken count ticks)."""
    already = bool(prev) and prev.get("phase") == "awaiting_capacity"
    if already and int(prev.get("overtaken", 0)) == int(overtaken):
        return  # unchanged — don't re-emit
    held_since = prev.get("held_since") if already else time.time()
    marker = {
        "phase": "awaiting_capacity",
        "reason": reason or {},
        "held_since": held_since,
        "overtaken": int(overtaken),
    }
    try:
        set_progress(job_id, marker)
    except Exception:  # noqa: BLE001 — the hold marker is best-effort observability
        pass


def _candidate_queued(limit: int):
    """The oldest ``limit`` queued rows (job_id, name, progress_json), FIFO order.
    A plain read (no write lock) so probing — which may hit the fleet read — never
    holds the sqlite write lock across network I/O."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT job_id, name, progress_json FROM media_jobs "
            "WHERE status='queued' ORDER BY created LIMIT ?", (int(limit),)
        ).fetchall()
    finally:
        conn.close()


def _try_claim_specific(job_id: str, worker_token: str) -> bool:
    """Atomically claim ONE specific queued job (the admission-chosen row). Same
    cross-process guarantee as claim(): BEGIN IMMEDIATE + a conditional UPDATE
    gated on status='queued', so exactly one claimer wins. Clears progress_json
    so a stale awaiting_capacity marker doesn't linger once the job starts. Returns
    True iff we won the row."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "UPDATE media_jobs SET status='claimed', claim_token=?, updated=?, "
            "progress_json=NULL WHERE job_id=? AND status='queued'",
            (worker_token, time.time(), job_id),
        )
        conn.execute("COMMIT")
        return cur.rowcount == 1
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def claim_admissible(worker_token: str) -> Optional[str]:
    """Reservation-GATED claim. Probes the reservation engine's non-destructive
    fit PROBE before flipping a queued row to 'claimed':

      * layer OFF  → transparent no-op: pure FIFO claim() (current behavior).
      * head fits  → claim the head (FIFO).
      * head can't → HOLD it (mark awaiting_capacity) and look PAST it, within a
                     bounded window, for a later job that fits (overtake). The
                     overtake count is capped (anti-starvation): once exhausted
                     the claimer idles until the head's capacity frees.
      * no later job fits AND force-admitting the head can't collide with an
                     active reservation → force-admit the head best-effort (its
                     envelope only fails because the render will OFFLOAD — §7.4),
                     exactly today's FIFO behavior. Otherwise return None (idle).

    Returns the claimed job_id, or None (queue empty / everything held). The
    reservation acquire() in run_claimed remains the AUTHORITY; this is advisory
    admission (handle the probe→acquire race via acquire's best-effort path)."""
    _ensure_db()
    if not _admission_enabled():
        return claim(worker_token)            # transparent FIFO — no fleet reads

    cands = _candidate_queued(_lookahead())
    if not cands:
        return None
    head_id, head_name, head_pj = cands[0]
    admit, reason = _probe_admission(head_name, head_id)
    if admit:
        if _try_claim_specific(head_id, worker_token):
            return head_id
        return None                           # lost the race — rescan next tick

    # Head can't be admitted now — HOLD it and account the overtake.
    head_prev = _load_progress(head_pj)
    overtaken = (int(head_prev.get("overtaken", 0))
                 if head_prev and head_prev.get("phase") == "awaiting_capacity"
                 else 0)
    _mark_awaiting_capacity(head_id, reason, head_prev, overtaken)

    # Overtake with a later admissible job (bounded by the anti-starvation cap).
    if overtaken < _max_overtake():
        for cid, cname, _cpj in cands[1:]:
            adc, _r = _probe_admission(cname, cid)
            if adc and _try_claim_specific(cid, worker_token):
                _mark_awaiting_capacity(head_id, reason, head_prev, overtaken + 1)
                logger.info("media_bus: %s (%s) overtook held head %s (%s) "
                            "[overtaken=%d/%d]", cid, cname, head_id, head_name,
                            overtaken + 1, _max_overtake())
                return cid

    # No later job fit (or the overtake budget is spent). Starvation/deadlock
    # guard: force-admit the head best-effort ONLY when that can't collide with an
    # active reservation (a lone head whose only failing is the offload envelope);
    # otherwise idle and wait for the in-flight reservation to release.
    if _force_admit_safe(head_name) and _try_claim_specific(head_id, worker_token):
        logger.info("media_bus: force-admitting held head %s (%s) best-effort — "
                    "no active-reservation collision (render offloads / worker gate "
                    "is the fit authority)", head_id, head_name)
        return head_id
    return None


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def enqueue(name: str, spec, principal: Optional[str] = None) -> str:
    """Mint a job_id, serialize the spec, insert status='queued'. Returns job_id.

    ``principal`` (k9) is the comms attribution string for whoever enqueued this
    job — resolved by the caller in the REQUEST context (this module has no Flask
    coupling). It is persisted on the row so the bus->JobStore bridge can stamp it
    onto /llm/jobs from the (possibly different) process that runs the job."""
    if name not in JOB_REGISTRY:
        raise KeyError(f"unknown job name {name!r}; registered: {sorted(JOB_REGISTRY)}")
    _ensure_db()
    job_id = uuid4().hex
    spec_json = serialize_spec(name, spec)
    now = time.time()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO media_jobs "
            "(job_id, name, status, spec_json, result_json, claim_token, created, updated, principal) "
            "VALUES (?, ?, 'queued', ?, NULL, NULL, ?, ?, ?)",
            (job_id, name, spec_json, now, now, principal),
        )
    finally:
        conn.close()
    # One-directional bridge (A/P0-2): surface this queued job in comms.JobStore
    # (GET /llm/jobs), carrying its attribution. Best-effort — never fails enqueue.
    _bridge("on_enqueue", job_id, name, principal=principal)
    return job_id


def claim(worker_token: str) -> Optional[str]:
    """Atomically claim the oldest queued job across processes. Returns job_id
    or None. Uses BEGIN IMMEDIATE + a conditional UPDATE so exactly one worker
    can transition a given row out of 'queued'."""
    _ensure_db()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT job_id FROM media_jobs WHERE status='queued' "
            "ORDER BY created LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        job_id = row[0]
        cur = conn.execute(
            "UPDATE media_jobs SET status='claimed', claim_token=?, updated=? "
            "WHERE job_id=? AND status='queued'",
            (worker_token, time.time(), job_id),
        )
        conn.execute("COMMIT")
        return job_id if cur.rowcount == 1 else None
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def run_claimed(job_id: str, worker_token: str) -> Optional[JobResult]:
    """Load + deserialize the spec (registry-driven), mark 'running' (single
    writer via claim_token), dispatch the pure runner, and write the JobResult
    ONCE. This is the ONLY place allowed to catch an UNEXPECTED raise and turn
    it into JobResult(ok=False, JobError('internal', ...))."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT name, spec_json, claim_token, principal FROM media_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        name, spec_json, claim_token, principal = row
        if claim_token != worker_token:
            # not our claim — refuse to write (single writer invariant)
            return None

        # transition to running (gated on our claim_token). The status guard
        # keeps a concurrent cancel visible: a job flipped to 'cancelling'
        # between claim and here must NOT be overwritten back to 'running'
        # (the runners poll is_cancelling between frames).
        conn.execute(
            "UPDATE media_jobs SET status='running', updated=? "
            "WHERE job_id=? AND claim_token=? AND status='claimed'",
            (time.time(), job_id, worker_token),
        )
    finally:
        conn.close()

    # One-directional bridge (A/P0-2): mark this job running in comms.JobStore
    # (GET /llm/jobs), in the process that actually owns the run — carrying the
    # attribution read from the row so it survives across the enqueue/run process
    # split (k9). Best-effort.
    _bridge("on_running", job_id, name, worker=worker_token, principal=principal)

    # ---- run outside the DB connection; the runner is pure & may block ----
    # p6: a heavy GPU video task pre-claims the card here (make-room via the
    # eviction verbs) BEFORE its runner touches the GPU. A non-heavy task / infra
    # hiccup proceeds unreserved; a measured, unclearable shortfall short-circuits
    # to a gpu_unavailable terminal instead of dispatching a render that would OOM.
    # The claim is released on EVERY terminal path in the finally below (incl.
    # abort/cancel + the exception path); a crashed run's claim self-expires.
    reservation_handle = None
    try:
        spec = deserialize_spec(name, json.loads(spec_json))
        job_spec = JOB_REGISTRY[name]
        runner = DISPATCH[job_spec.runner_key]
        reservation_handle, reservation_refusal = _acquire_reservation(
            name, spec, job_id)
        if reservation_refusal is not None:
            result = reservation_refusal
        else:
            result = runner(spec, job_id)
            if not isinstance(result, JobResult):
                raise TypeError(
                    f"runner {job_spec.runner_key} returned {type(result).__name__}, "
                    "expected JobResult"
                )
    except Exception as exc:  # the one sanctioned catch/convert point
        from .result_schema import JobError
        result = JobResult(
            job_id=job_id,
            ok=False,
            error=JobError(
                code="internal",
                message=f"{type(exc).__name__}: {exc}",
                retryable=False,
            ),
        )
    finally:
        # Release the GPU claim on ANY terminal path — success, failure, cancel,
        # or an internal raise. No-op when nothing was claimed.
        if reservation_handle is not None:
            _release_reservation(job_id)

    # ---- write the terminal state ONCE (single writer via claim_token) ----
    # A runner that honored a cancel returns error.code='cancelled' — record
    # that as its own terminal status so the console can tell "you stopped it"
    # from "it broke".
    if result.ok:
        status = "done"
    elif result.error is not None and getattr(result.error, "code", None) == "cancelled":
        status = "cancelled"
    else:
        status = "failed"
    result_json = serialize_result(result)
    conn = _connect()
    try:
        conn.execute(
            "UPDATE media_jobs SET status=?, result_json=?, progress_json=NULL, "
            "updated=? WHERE job_id=? AND claim_token=?",
            (status, result_json, time.time(), job_id, worker_token),
        )
    finally:
        conn.close()
    # STAGE TIMELINE terminal entry — RETAINED (the progress_json blob above was
    # nulled; this is not). Carries the outcome: the exact code/message/retryable +
    # failed_at_stage on a failure. Best-effort — a timeline hiccup never masks the
    # real terminal write above.
    try:
        _record_terminal_stage(job_id, status, result)
    except Exception:  # noqa: BLE001
        logger.debug("media_bus: terminal stage record failed (non-fatal) for %s",
                     job_id, exc_info=True)
    # One-directional bridge (A/P0-2): mark the terminal state in comms.JobStore
    # (done | failed | cancelled), carrying the clip uri on success + the
    # attribution (k9). Best-effort.
    _bridge("on_terminal", job_id, name, status, result=result,
            worker=worker_token, principal=principal)
    return result


def cancel(job_id: str) -> dict:
    """Cooperative cancel. queued → 'cancelled' outright (claim() only picks
    'queued', so it never runs); claimed/running → 'cancelling', a flag the
    frame-loop runners poll via is_cancelling() and honor between frames
    (mid-frame inference is never interrupted); terminal states are untouched.
    Returns {"job_id", "status", "cancelled": bool} — cancelled=False means
    there was nothing to stop (unknown id or already terminal)."""
    _ensure_db()
    from .result_schema import JobError, JobResult as _JR
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT status FROM media_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
        if row is None:
            return {"job_id": job_id, "status": None, "cancelled": False}
        status = row[0]
        if status == "queued":
            result_json = serialize_result(_JR(
                job_id=job_id, ok=False,
                error=JobError(code="cancelled",
                               message="cancelled before it started",
                               retryable=False)))
            conn.execute(
                "UPDATE media_jobs SET status='cancelled', result_json=?, "
                "updated=? WHERE job_id=? AND status='queued'",
                (result_json, time.time(), job_id),
            )
            # STAGE TIMELINE: record the pre-start cancel as a retained terminal
            # entry too (best-effort). Uses this connection's own write above; the
            # append opens its own connection (WAL, single-writer per row is fine).
            try:
                _append_stage_log(job_id, "cancelled", "cancelled before it started",
                                  terminal_extra={"code": "cancelled",
                                                  "message": "cancelled before it started",
                                                  "retryable": False})
            except Exception:  # noqa: BLE001
                pass
            # Bridge the immediate (pre-start) terminal so a cancel-before-run
            # shows as cancelled in GET /llm/jobs too. Best-effort; the name is
            # read from the row so no signature change is needed.
            nrow = conn.execute(
                "SELECT name FROM media_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            _bridge("on_terminal", job_id, (nrow[0] if nrow else "media"),
                    "cancelled")
            return {"job_id": job_id, "status": "cancelled", "cancelled": True}
        if status in ("claimed", "running"):
            conn.execute(
                "UPDATE media_jobs SET status='cancelling', updated=? "
                "WHERE job_id=? AND status IN ('claimed','running')",
                (time.time(), job_id),
            )
            return {"job_id": job_id, "status": "cancelling", "cancelled": True}
        return {"job_id": job_id, "status": status, "cancelled": False}
    finally:
        conn.close()


def is_cancelling(job_id: str) -> bool:
    """True while a cancel is pending for a claimed/running job — polled by the
    frame-loop runners between frames."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT status FROM media_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
        return bool(row) and row[0] == "cancelling"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Studio-clip ARCHIVE (never-delete doctrine) — fixes "removed clips just
# reappear": GET /video/studio/clips is DB-driven (a media_jobs SELECT, not a
# filesystem walk — see that route's header note), so "remove from the library"
# is a mark this bus owns, not a client-only mutation the next ~6s poll undoes.
# The row and the clip's bytes on disk are NEVER touched — archived_at is the
# only thing that changes. Mirrors cancel()/is_cancelling()'s idiom: idempotent,
# reports what happened via a dict rather than raising, single writer per call.
# --------------------------------------------------------------------------- #
def archive(job_id: str) -> dict:
    """Archive a studio clip: sets archived_at (once) so the list query excludes
    it. Scoped to name='studio_i2v' — this bus also carries every other job kind,
    and archive is a studio-clips-library concept only. Idempotent: archiving an
    already-archived clip is a clean NO-OP that reports the ORIGINAL archived_at
    (never bumped) via `already=True`, not an error — a UI retry or a double
    click must never surface as a failure. Returns {"job_id","found","archived",
    "already","archived_at"}; found=False means no studio_i2v row exists for that
    id (the route's 404)."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT archived_at FROM media_jobs WHERE job_id=? AND name='studio_i2v'",
            (job_id,),
        ).fetchone()
        if row is None:
            return {"job_id": job_id, "found": False, "archived": False,
                    "already": False, "archived_at": None}
        existing = row[0]
        if existing is not None:
            return {"job_id": job_id, "found": True, "archived": True,
                    "already": True, "archived_at": existing}
        now = time.time()
        conn.execute(
            "UPDATE media_jobs SET archived_at=? "
            "WHERE job_id=? AND name='studio_i2v' AND archived_at IS NULL",
            (now, job_id),
        )
        return {"job_id": job_id, "found": True, "archived": True,
                "already": False, "archived_at": now}
    finally:
        conn.close()


def unarchive(job_id: str) -> dict:
    """The honest counterpart to archive() — cheap to add, and it keeps the
    archive a REVERSIBLE hide rather than a one-way trapdoor. Same idempotent
    shape: unarchiving a clip that was never archived (or already unarchived) is
    a clean no-op (`already=True`), not an error."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT archived_at FROM media_jobs WHERE job_id=? AND name='studio_i2v'",
            (job_id,),
        ).fetchone()
        if row is None:
            return {"job_id": job_id, "found": False, "archived": False, "already": False}
        existing = row[0]
        if existing is None:
            return {"job_id": job_id, "found": True, "archived": False, "already": True}
        conn.execute(
            "UPDATE media_jobs SET archived_at=NULL "
            "WHERE job_id=? AND name='studio_i2v' AND archived_at IS NOT NULL",
            (job_id,),
        )
        return {"job_id": job_id, "found": True, "archived": False, "already": False}
    finally:
        conn.close()


def is_archived(job_id: str) -> bool:
    """True if the studio clip carries an archived_at mark — checked by the
    per-id serve/detail routes so a direct fetch of an archived clip answers
    HONESTLY (410 'archived') instead of the generic 404 'never existed'.
    Un-gated read (like is_cancelling); NOT scoped to name='studio_i2v' since
    job_id is globally unique and archived_at is only ever set by archive()
    above, which already enforces that scope."""
    _ensure_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT archived_at FROM media_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
        return bool(row) and row[0] is not None
    finally:
        conn.close()


def set_progress(job_id: str, progress: dict) -> None:
    """Record the live per-frame progress blob for a running job — written by the
    frame-loop runners as frames land and read by the poller via get().

    Un-gated (like is_cancelling): the runner owns its own job_id, so no
    claim_token guard is needed. Overwrites progress_json each call. Best-effort
    at the call site — the runners wrap this so a transient DB hiccup never fails
    a generation."""
    _ensure_db()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE media_jobs SET progress_json=?, updated=? WHERE job_id=?",
            (json.dumps(progress), time.time(), job_id),
        )
    finally:
        conn.close()
    # STAGE TIMELINE: append/coalesce a coarse-stage entry so the panel has the full
    # sequence a render moved through (deduped — a 48-frame loop stays ONE 'rendering'
    # row with a running count, not 48). Best-effort; a timeline hiccup never fails a
    # generation (mirrors the whole set_progress contract).
    try:
        _append_stage_log(job_id, _coarse_stage(progress), _stage_detail(progress))
    except Exception:  # noqa: BLE001
        logger.debug("media_bus: stage-log append failed (non-fatal) for %s",
                     job_id, exc_info=True)
    # One-directional bridge (A/P0-2): mirror this live progress blob into the
    # comms.JobStore so GET /llm/jobs carries the running job's stage + log tail +
    # honest stall clock (not green-but-empty). Best-effort by construction (the
    # dispatcher swallows) — a bridge hiccup never fails the generation.
    _bridge("on_progress", job_id, progress)


def work_once(worker_token: Optional[str] = None) -> Optional[str]:
    """Claim one job and run it. Returns the processed job_id or None if the
    queue was empty. This is what the headless self-test and the daemon call."""
    token = worker_token or f"worker-{os.getpid()}-{uuid4().hex[:8]}"
    job_id = claim(token)
    if job_id is None:
        return None
    run_claimed(job_id, token)
    return job_id


def get(job_id: str) -> dict:
    """Read-only view: {"job_id", "name", "status", "result": <JobResult dict|None>,
    "progress": <blob dict|None>}. `progress` is an object WHILE running (the
    live per-frame blob) and null at a terminal state (run_claimed nulls it on
    the terminal write). `name` (the bus job kind) is additive — callers that
    ignore it are unaffected, and the placement projection reads it. Unknown id ->
    all-null view."""
    _ensure_db()
    conn = _connect_ro()
    try:
        row = conn.execute(
            "SELECT name, status, result_json, progress_json, stage_log_json, updated "
            "FROM media_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"job_id": job_id, "name": None, "status": None,
                "result": None, "progress": None,
                # Additive telemetry fields (empty for an unknown id) so a consumer
                # can read them unconditionally without a shape check.
                "stage_log": [], "failure": None,
                "last_movement_ts": None, "current_stage": None}
    name, status, result_json, progress_json, stage_log_json, updated = row
    result = json.loads(result_json) if result_json else None
    progress = json.loads(progress_json) if progress_json else None
    stage_log = _load_stage_log(stage_log_json)
    return {"job_id": job_id, "name": name, "status": status,
            "result": result, "progress": progress,
            # ADDITIVE (the exhaustive per-process telemetry): the retained stage
            # TIMELINE (survives terminal), the terminal FAILURE summary (stage/code/
            # message/retryable — "where it's failing"), the last-movement ts (honest
            # stall basis tied to the current stage), and the current stage. Existing
            # keys are unchanged.
            "stage_log": stage_log,
            "failure": build_failure_summary(result, stage_log),
            "last_movement_ts": _last_movement_ts(stage_log, updated),
            "current_stage": _current_stage(stage_log),
            # The numeric bar (k57) — 0..1 overall, plus the numbers behind it.
            # None when the runner reports nothing measurable (no fabricated bar).
            "progress_ratio": _progress_ratio(progress),
            "progress_detail": _progress_detail(progress)}


# --------------------------------------------------------------------------- #
# Bus-wide LISTING — feeds GET /video/jobs (the console-wide "Active Processes"
# view). Read-only projection over the media_jobs catalog (like
# /video/studio/clips), NOT the comms /llm/jobs view (which drops terminal rows
# after ~600s). In-flight rows by default; ``include_terminal`` appends recent
# terminal rows (bounded). The route enriches each row with a placement object.
# --------------------------------------------------------------------------- #
_INFLIGHT_STATES = ("queued", "claimed", "running", "cancelling")
_TERMINAL_STATES = ("done", "failed", "cancelled")

# How many timeline entries a LIVE row carries into the listing. The panel wants
# "the last few events", not the whole 60-entry history, and the feed is polled
# every couple of seconds — so live rows ship a TAIL and the row's
# ``stage_log_total`` says how much was elided (the per-id GET has the rest).
# TERMINAL rows are never trimmed: a failed render must show its full sequence.
_STAGE_TAIL = 6

# Stale-in-flight cutoff (k57 item 5). A bus row still claiming an in-flight
# status whose PROGRESSED_AT (last real movement — the stage-timeline clock, not
# the row's `updated`, which any rewrite bumps) is older than this was abandoned
# by a process that died without a terminal write: six studio_i2v daemon rows from
# 2026-07-29 were still rendering as "running" in the Active panel two days later.
# They are HIDDEN from the default listing rather than deleted — this is a read
# path, and it must never write (a renderer's own DB is not a view's to mutate).
# The window is deliberately long: a job legitimately HELD for GPU capacity only
# re-writes its marker when the hold CHANGES, so its movement clock can idle for
# hours while it is perfectly alive. Env-overridable; default 6h.
def _stale_inflight_seconds() -> float:
    raw = (os.environ.get("HUGPY_MEDIA_BUS_STALE_SECONDS") or "").strip()
    if not raw:
        return 21600.0
    try:
        v = float(raw)
        return v if v > 0 else 21600.0
    except ValueError:
        return 21600.0


def _project_job_row(r, *, tail: bool = False, now: Optional[float] = None,
                     stale_cutoff: Optional[float] = None) -> dict:
    (job_id, name, status, created, updated, principal, progress_json,
     stage_log_json, result_json) = r
    stage_log = _load_stage_log(stage_log_json)
    result = json.loads(result_json) if result_json else None
    progress = _load_progress(progress_json)
    progressed_at = _last_movement_ts(stage_log, updated)
    row = {
        "job_id": job_id,
        "name": name,
        "status": status,
        "created": created,
        "updated": updated,
        "principal": principal,
        "progress": progress,
        # ADDITIVE per-process telemetry (identical to media_bus.get()'s): the
        # retained stage TIMELINE, the terminal FAILURE summary, the last-movement
        # ts (stall basis), and the current stage. Terminal rows (include_terminal)
        # carry their full timeline + failure so a failed render stays inspectable.
        "stage_log": stage_log[-_STAGE_TAIL:] if tail else stage_log,
        "stage_log_total": len(stage_log),
        "failure": build_failure_summary(result, stage_log),
        "last_movement_ts": progressed_at,
        "current_stage": _current_stage(stage_log),
        # k57: the numeric bar + the numbers behind it, and the movement clock the
        # stale filter ages on (surfaced so the panel can SAY why a row is hidden).
        "progress_ratio": _progress_ratio(progress),
        "progress_detail": _progress_detail(progress),
        "progressed_at": progressed_at,
    }
    if stale_cutoff is not None:
        row["stale"] = bool(progressed_at is not None and progressed_at < stale_cutoff)
        if row["stale"] and now is not None and progressed_at is not None:
            row["stale_for_s"] = round(now - progressed_at, 1)
    return row


_INFLIGHT_SELECT = (
    "SELECT job_id, name, status, created, updated, principal, progress_json, "
    "stage_log_json, NULL "
    "FROM media_jobs WHERE status IN (?,?,?,?) ORDER BY created ASC LIMIT ?")

# The same query with the stale rows dropped in SQL. `updated` is bumped by every
# write that also moves the timeline, so movement_ts <= updated always: an
# `updated` older than the cutoff is stale BEYOND DOUBT and can be excluded before
# LIMIT (so a pile of abandoned rows can never crowd live work out of the page).
# The exact (timeline-based) test still runs per row afterwards for the rest.
_INFLIGHT_SELECT_FRESH = (
    "SELECT job_id, name, status, created, updated, principal, progress_json, "
    "stage_log_json, NULL "
    "FROM media_jobs WHERE status IN (?,?,?,?) AND updated >= ? "
    "ORDER BY created ASC LIMIT ?")

# Terminal rows need result_json for the failure envelope — but ONLY a FAILED row
# has one, and a DONE movie's result blob runs to megabytes (2.7 MB on the live
# central). Reading those blobs to throw them away was a second-order cost of the
# ?all=1 listing, so the CASE keeps sqlite from touching the overflow pages of the
# rows whose result we would discard anyway.
_TERMINAL_SELECT = (
    "SELECT job_id, name, status, created, updated, principal, progress_json, "
    "stage_log_json, CASE WHEN status='failed' THEN result_json END "
    "FROM media_jobs WHERE status IN (?,?,?) ORDER BY updated DESC LIMIT ?")


def list_jobs(include_terminal: bool = False, limit: int = 50,
              include_stale: bool = False) -> List[dict]:
    """Bus-wide job listing for GET /video/jobs.

    In-flight rows (queued/claimed/running/cancelling) in FIFO order by ``created``
    by default; ``include_terminal`` appends up to ``limit`` recent terminal rows
    (done/failed/cancelled), newest-updated first. Each row is a dict:
    {job_id, name, status, created, updated, principal, progress(parsed|None),
     stage_log(tail on live rows)/stage_log_total, failure, progress_ratio,
     progress_detail, progressed_at, stale} — ``progress`` carries the live
    per-frame blob AND the ``awaiting_capacity`` hold marker verbatim.

    STALE in-flight rows (no movement for ``_stale_inflight_seconds``, i.e. a job
    orphaned by a dead process) are hidden unless ``include_stale``; they are still
    reachable per-id and via the terminal/stale filters, and are never mutated here.

    READ-ONLY BY CONSTRUCTION (k57): opened ``mode=ro``, so this path cannot take a
    write lock even by accident, and it does NO per-row work beyond parsing the
    columns it already selected — no network call, no second store, no lock a
    renderer holds. ``limit`` is clamped to 1..200. The caller enriches the rows
    with placement from a single ``PlacementSnapshot``."""
    _ensure_db()
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))
    now = time.time()
    stale_cutoff = now - _stale_inflight_seconds()
    conn = _connect_ro()
    try:
        if include_stale:
            rows = conn.execute(_INFLIGHT_SELECT,
                                (*_INFLIGHT_STATES, limit)).fetchall()
        else:
            rows = conn.execute(_INFLIGHT_SELECT_FRESH,
                                (*_INFLIGHT_STATES, stale_cutoff, limit)).fetchall()
        out = [_project_job_row(r, tail=True, now=now, stale_cutoff=stale_cutoff)
               for r in rows]
        if not include_stale:
            out = [row for row in out if not row.get("stale")]
        if include_terminal:
            trows = conn.execute(_TERMINAL_SELECT,
                                 (*_TERMINAL_STATES, limit)).fetchall()
            out.extend(_project_job_row(r) for r in trows)
        return out
    finally:
        conn.close()


def _runner_loop(worker_token: str, idle_sleep_s: float,
                 stop_event: Optional[threading.Event] = None) -> None:
    """One pool thread: reservation-gated claim -> run, forever. Each pass claims
    at most one job (heavy runs serialize naturally via exclusive reservations;
    light/CPU jobs run concurrently across the pool). Never dies on a transient
    error — a bad tick just idles. ``stop_event`` (optional) lets a caller stop the
    loop gracefully (used by tests; production runs it forever as a daemon)."""
    while not (stop_event is not None and stop_event.is_set()):
        try:
            job_id = claim_admissible(worker_token)
        except Exception:  # noqa: BLE001 — never let a runner die on a transient error
            logger.debug("media_bus runner: claim_admissible raised", exc_info=True)
            job_id = None
        if job_id is None:
            if stop_event is not None and stop_event.wait(idle_sleep_s):
                return
            elif stop_event is None:
                time.sleep(idle_sleep_s)
            continue
        try:
            run_claimed(job_id, worker_token)
        except Exception:  # noqa: BLE001 — run_claimed already converts runner raises;
            # this guards ONLY an unexpected bus-level error so the pool survives.
            logger.warning("media_bus runner: run_claimed raised for %s",
                           job_id, exc_info=True)


def start_worker_daemon(worker_token: Optional[str] = None,
                        idle_sleep_s: float = 0.25,
                        stop_event: Optional[threading.Event] = None
                        ) -> List[threading.Thread]:
    """Start the media-bus RUNNER POOL — ``HUGPY_MEDIA_BUS_RUNNERS`` (default 2)
    threads, each doing reservation-gated claim -> run. Replaces the old single
    serial daemon so light/CPU tasks and multi-worker fleets no longer queue
    behind a heavy render; heavy GPU tasks still serialize via exclusive
    reservations (the admission gate + the worker gate keep the single-3090 fleet
    a success path — nothing OOMs that wouldn't today). Returns the thread list.

    DEFINED but NOT called at import — wsgi wires it once per process at app init.
    The bus's atomic cross-process claim still guarantees exactly one runner (of
    the N*processes total) transitions any given job."""
    base = worker_token or f"daemon-{os.getpid()}"
    n = _runner_count()
    threads: List[threading.Thread] = []
    for i in range(n):
        token = f"{base}-r{i}-{uuid4().hex[:6]}"
        t = threading.Thread(target=_runner_loop,
                             args=(token, idle_sleep_s, stop_event),
                             name=f"media_bus_worker_{i}", daemon=True)
        t.start()
        threads.append(t)
    logger.info("media_bus: started %d runner thread(s)", n)
    return threads
