"""The reservation engine — acquire / hold / release around a heavy video run.

``acquire(job_name, spec, run_id)`` is called by ``media_bus.run_claimed`` right
before a heavy runner dispatches. It:

  1. loads the task template (measured overlay applied) → the run's PEAK GPU need;
  2. resolves the TARGET worker/card (today: ``ae``, the one video GPU);
  3. records an ACTIVE claim in the registry BEFORE making room, so central
     admission-respect (``fleet_snapshot``) immediately treats the reserved bytes
     as not-free for other placements;
  4. drives PROACTIVE make-room through the EXISTING worker verbs — a ComfyUI
     flush first (``/ops/evict`` a comfy-attributed resident → the worker's own
     ``_comfy_free_models`` / ``set_comfy_headroom_hook`` path), then the eviction
     engine (``/ops/evict`` the on-demand residents largest-first, ``force=false``
     so the WORKER's gate keeps 🔒static / actively-replying / queued-ahead
     residents safe) — polling live free-VRAM within a bounded deadline;
  5. on success, starts a lease REFRESHER (heartbeat) and returns a handle held
     for the whole run; on timeout-while-short it RELEASES the claim and raises
     ``ReservationRefused`` (honest refusal — never an admit-then-OOM, never a
     new protected tier, never a deadlock against one it can't clear).

``release(run_id)`` (called on ANY terminal run path — done/failed/cancelled/
abort/crash-via-lease-expiry) stops the refresher and terminals the claim.

Everything is BEST-EFFORT and fail-OPEN on infrastructure problems: if the store
is down, the fleet is unreadable, or the peak is unknown, a render PROCEEDS
UNRESERVED exactly as it does today. The engine only ever REFUSES when it can
measure a real shortfall it could not clear — the one case where proceeding
would OOM.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .registry import reservation_registry
from .templates import ReservationTemplate, load_template

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3


class ReservationRefused(Exception):
    """The card could not be cleared to the run's peak within the deadline (the
    shortfall is all protected residents, or make-room stalled). Carries a typed
    reason so the dispatch path surfaces an honest 'GPU unavailable' terminal
    instead of dispatching a render that would OOM."""
    def __init__(self, reason: Dict[str, Any]):
        self.reason = reason or {}
        super().__init__(self.reason.get("reason") or "GPU reservation refused")


# ── tunables (env-overridable) ───────────────────────────────────────────────
def _makeroom_timeout_s() -> float:
    try:
        return max(0.0, float(os.environ.get("HUGPY_RESERVATION_MAKEROOM_TIMEOUT_S", "90")))
    except ValueError:
        return 90.0


def _poll_s() -> float:
    try:
        return max(0.2, float(os.environ.get("HUGPY_RESERVATION_POLL_S", "3")))
    except ValueError:
        return 3.0


def _settle_s() -> float:
    """Pause after an evict so CUDA/host frees settle before the next fit re-read."""
    try:
        return max(0.0, float(os.environ.get("HUGPY_RESERVATION_SETTLE_S", "1.5")))
    except ValueError:
        return 1.5


def _enabled() -> bool:
    return (os.environ.get("HUGPY_RESERVATIONS") or "on").strip().lower() not in (
        "0", "off", "false", "no", "")


def _refuse_enabled() -> bool:
    """Whether a make-room shortfall HARD-REFUSES the run (gpu_unavailable) vs.
    proceeds best-effort.

    DEFAULT OFF — a safety promise ([[defaults-are-promises]]). The seeded peaks
    are the WHOLE-GPU envelope (e.g. Wan ~20 GB), but a studio render with a blank
    budget AUTOFITS and OFFLOADS to whatever VRAM is free (§3.4 stage 1), so it
    does NOT strictly need the envelope — refusing on it would block a render that
    would have succeeded offloaded. So by default the engine does the VALUABLE
    part (proactive make-room + honest accounting) and PROCEEDS, leaving the actual
    fit to the render's autofit + the WORKER's own admission gate (the authority,
    which already handles offload). Turn ON once p7's ``measured.json`` supplies
    real per-(model,geometry,precision) peaks — then an honest refusal is a
    refusal against a TRUE need, not the envelope."""
    return (os.environ.get("HUGPY_RESERVATION_REFUSE") or "off").strip().lower() in (
        "1", "on", "true", "yes")


# ── fleet reads (lazy — the engine stays boot-cheap; a bare/worker context degrades) ─
def _list_workers() -> List[Dict[str, Any]]:
    try:
        from ...flask_app.app.functions.imports.utils.workers import list_workers
        return list(list_workers() or [])
    except Exception:  # noqa: BLE001
        return []


def _url_host(url: str) -> str:
    from urllib.parse import urlparse
    if not url:
        return ""
    u = url if "://" in url else "http://" + url
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _delegation_base(template: ReservationTemplate) -> str:
    """The delegation target URL for this template's task (host-matched to a
    registry worker below). Studio tasks → HUGPY_STUDIO_WORKER; identity tasks →
    IDENTITY_RENDER_URL; dispatch-plane tasks have no fixed base."""
    if template.delegation == "studio_worker":
        return (os.environ.get("HUGPY_STUDIO_WORKER") or "").strip()
    if template.delegation == "identity_render":
        return (os.environ.get("IDENTITY_RENDER_URL") or "").strip()
    return ""


def _has_gpu(w: Dict[str, Any]) -> Optional[int]:
    gpus = [g for g in (w.get("gpus") or []) if isinstance(g, dict)]
    totals = [g.get("memory_total") for g in gpus
              if isinstance(g.get("memory_total"), (int, float)) and g.get("memory_total") > 0]
    return int(max(totals)) if totals else None


def _resolve_target(template: ReservationTemplate
                    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """(worker_id, worker_dict) for the card this run reserves, or (None, None).

    Prefer a host-match to the delegation target (studio/identity env); else the
    single online GPU-bearing worker with the most VRAM (today: ``ae``, the one
    video GPU §0). Only ONLINE workers are eligible."""
    workers = [w for w in _list_workers() if w.get("status") == "online"]
    if not workers:
        return None, None
    base = _delegation_base(template)
    host = _url_host(base)
    if host:
        for w in workers:
            if _url_host(w.get("url") or "") == host and w.get("id"):
                return w["id"], w
    # Fallback: the biggest online GPU box (the video GPU).
    gpu_workers = [(w, _has_gpu(w)) for w in workers]
    gpu_workers = [(w, t) for (w, t) in gpu_workers if t]
    if not gpu_workers:
        return None, None
    gpu_workers.sort(key=lambda wt: -wt[1])
    w = gpu_workers[0][0]
    return w.get("id"), w


def _refresh_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    for w in _list_workers():
        if w.get("id") == worker_id:
            return w
    return None


def _free_vram(worker: Optional[Dict[str, Any]]) -> Optional[int]:
    """Physical free VRAM on the target card — the LARGEST single GPU's free bytes
    (a render binds ONE device; never the multi-GPU sum). None when unmeasurable."""
    if not worker:
        return None
    gpus = [g for g in (worker.get("gpus") or []) if isinstance(g, dict)]
    frees = [g.get("memory_free") for g in gpus
             if isinstance(g.get("memory_free"), (int, float)) and g.get("memory_free") > 0]
    return int(max(frees)) if frees else None


def _pid_models(worker: Dict[str, Any]) -> List[Dict[str, Any]]:
    pr = worker.get("pid_registry")
    if isinstance(pr, dict):
        models = pr.get("models")
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict)]
    return []


def _comfy_resident_keys(worker: Dict[str, Any]) -> List[str]:
    """Comfy-attributed residents central can flush via /ops/evict (→ comfy /free).
    An IDLE comfy holds a null model_key (nothing central can target by key — that
    residual case is the worker-side set_comfy_headroom_hook, release-bound)."""
    out = []
    for m in _pid_models(worker):
        if str(m.get("host_mode")) == "comfy" and m.get("model_key") \
                and int(m.get("vram_bytes") or 0) > 0:
            out.append(str(m["model_key"]))
    return out


def _evict_candidates(worker: Dict[str, Any], tried: set) -> List[str]:
    """On-demand LLM/diffusers residents central may ASK the worker to evict,
    largest-first (frees the most, fewest calls). ``force=false`` on the relay
    means the WORKER is the authority on what is actually permissible — a static /
    replying / queued-ahead resident is refused there, never here. cuda_context /
    comfy(null-key) entries carry no model_key and are skipped."""
    rows = []
    for m in _pid_models(worker):
        mk = m.get("model_key")
        if not mk or mk in tried:
            continue
        if str(m.get("host_mode")) == "comfy":
            continue  # handled by the comfy-flush pass
        vb = int(m.get("vram_bytes") or 0)
        if vb <= 0:
            continue
        rows.append((str(mk), vb))
    if not rows:
        # Fallback for a worker that doesn't report pid_registry: loaded_detail.
        ld = worker.get("loaded_detail")
        if isinstance(ld, dict):
            for mk, det in ld.items():
                if mk in tried or not isinstance(det, dict):
                    continue
                vb = int(det.get("model_bytes") or det.get("weight_bytes") or 0)
                rows.append((str(mk), vb))
    rows.sort(key=lambda r: -r[1])
    return [mk for mk, _ in rows]


def _evict(worker: Dict[str, Any], model_key: str, force: bool = False) -> Dict[str, Any]:
    """Relay a targeted eviction to the worker's control agent (/ops/evict). The
    worker picks the mechanism by host_mode (comfy /free, slot SIGTERM, in-process
    ref-drop) and enforces the protection gate when force=false. Best-effort:
    a transport error reads as 'not evicted' and the loop moves on."""
    url = (worker.get("url") or "").rstrip("/") + "/ops/evict"
    try:
        import httpx
        r = httpx.post(url, json={"model_key": model_key, "force": bool(force)},
                       timeout=45.0)
        if r.status_code == 200:
            return r.json()
        return {"evicted": False, "reason": f"HTTP {r.status_code}"}
    except Exception as exc:  # noqa: BLE001
        return {"evicted": False, "reason": f"{type(exc).__name__}: {exc}"}


def _refusal_reason(worker: Optional[Dict[str, Any]], peak: int,
                    free: Optional[int], evicted: List[str]) -> Dict[str, Any]:
    remaining = []
    if worker:
        for m in _pid_models(worker):
            mk = m.get("model_key")
            if mk and mk not in evicted and int(m.get("vram_bytes") or 0) > 0:
                remaining.append({"model_key": str(mk),
                                  "vram_bytes": int(m.get("vram_bytes") or 0),
                                  "host_mode": str(m.get("host_mode"))})
    return {
        "reason": "GPU reservation could not clear the card to the run's peak "
                  "(remaining residents are protected: static / actively replying "
                  "/ queued ahead)",
        "peak_bytes": int(peak),
        "free_bytes": (int(free) if free is not None else None),
        "short_by_bytes": (int(peak - free) if free is not None else None),
        "worker_id": (worker or {}).get("id"),
        "evicted": list(evicted),
        "remaining_residents": remaining,
    }


# ── make-room orchestration ──────────────────────────────────────────────────
def _ensure_headroom(worker_id: str, worker: Dict[str, Any], peak: Optional[int]
                     ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Drive the card to ``peak`` free bytes via the existing verbs. Returns
    (ok, evicted, refusal_reason). Fails OPEN (ok=True) when peak/free are
    unmeasurable — never blocks a render we cannot size."""
    evicted: List[str] = []
    if peak is None:
        return True, evicted, None
    free = _free_vram(worker)
    if free is None:
        return True, evicted, None            # unmeasurable → proceed best-effort
    if free >= peak:
        return True, evicted, None            # already fits — no eviction (shared card)

    deadline = time.time() + _makeroom_timeout_s()
    tried: set = set()

    # 1) ComfyUI flush FIRST — the cheap 2-7 GB out-of-band win.
    for mk in _comfy_resident_keys(worker):
        res = _evict(worker, mk, force=False)
        if res.get("evicted"):
            evicted.append(mk)
        tried.add(mk)
        fa = res.get("vram_free_after")
        if isinstance(fa, (int, float)) and fa >= peak:
            return True, evicted, None
    worker = _refresh_worker(worker_id) or worker
    free = _free_vram(worker)
    if free is not None and free >= peak:
        return True, evicted, None

    # 2) Eviction engine — on-demand residents largest-first, force=false so the
    #    worker's own gate protects static/replying/queued-ahead. Bounded wait.
    while time.time() < deadline:
        worker = _refresh_worker(worker_id) or worker
        free = _free_vram(worker)
        if free is not None and free >= peak:
            return True, evicted, None
        cands = [c for c in _evict_candidates(worker, tried)]
        if not cands:
            # Nothing left we may try. Give an in-flight resident a moment to free
            # (it may finish replying), then re-check; if still short → the whole
            # shortfall is protected → refuse honestly (never deadlock).
            time.sleep(_poll_s())
            worker = _refresh_worker(worker_id) or worker
            free = _free_vram(worker)
            if free is not None and free >= peak:
                return True, evicted, None
            return False, evicted, _refusal_reason(worker, peak, free, evicted)
        cand = cands[0]
        res = _evict(worker, cand, force=False)
        tried.add(cand)
        if res.get("evicted"):
            evicted.append(cand)
            fa = res.get("vram_free_after")
            if isinstance(fa, (int, float)) and fa >= peak:
                return True, evicted, None
        # else: gated (protected) or no-op — 'tried' keeps us from spinning on it.
        time.sleep(_settle_s())

    # Deadline hit — one last honest re-read.
    worker = _refresh_worker(worker_id) or worker
    free = _free_vram(worker)
    if free is not None and free >= peak:
        return True, evicted, None
    return False, evicted, _refusal_reason(worker, peak, free, evicted)


# ── lease refresher ──────────────────────────────────────────────────────────
class _Claim:
    """Handle for a held reservation. Owns a daemon refresher thread that heartbeats
    the lease until released; a crash that never releases lets the lease lapse and
    the registry self-expires the claim (orphan safety)."""
    def __init__(self, run_id: str, worker_id: Optional[str], peak: Optional[int]):
        self.run_id = run_id
        self.worker_id = worker_id
        self.peak = peak
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_refresher(self) -> None:
        interval = max(5.0, reservation_registry.lease_ttl_s / 3.0)

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    reservation_registry.refresh(self.run_id)
                except Exception:  # noqa: BLE001 — a refresh miss just shortens the lease
                    pass

        t = threading.Thread(target=_loop,
                             name=f"reservation-refresh-{self.run_id[:8]}",
                             daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()


# Live refreshers by run_id, so release(run_id) can stop the thread even without
# the handle in hand (belt-and-suspenders against a lost handle).
_ACTIVE: Dict[str, _Claim] = {}
_ACTIVE_LOCK = threading.Lock()


# ── public API (dispatch path only) ──────────────────────────────────────────
def acquire(job_name: str, spec: Any, run_id: str) -> Optional[_Claim]:
    """Pre-claim the card for a heavy video run. Returns a held claim, or None when
    the task is not reservable / the layer is disabled / infra is unreadable
    (proceed unreserved). Raises ReservationRefused when a real, measured shortfall
    could not be cleared within the deadline (surface an honest terminal)."""
    if not _enabled():
        return None
    template = load_template(job_name)
    if template is None:
        return None                       # light task — no reservation
    worker_id, worker = _resolve_target(template)
    peak = template.peak_bytes()
    if worker_id is None or worker is None:
        # Can't see the fleet — fail open (proceed unreserved), don't block a render
        # because central momentarily can't resolve the target.
        logger.info("reservation: no target worker resolved for %s (run %s) — "
                    "proceeding unreserved", job_name, run_id)
        return None

    gpu = template.gpu_affinity
    # Claim BEFORE make-room so admission-respect sees the reserved bytes immediately.
    reservation_registry.claim(run_id, worker_id, gpu, job_name, peak)
    try:
        ok, evicted, refusal = _ensure_headroom(worker_id, worker, peak)
    except Exception as exc:  # noqa: BLE001 — an engine bug must not wedge the render
        logger.warning("reservation make-room raised for %s (run %s) — proceeding "
                       "unreserved: %s", job_name, run_id, exc, exc_info=True)
        # Keep the claim (accounting) but don't refuse on our own bug; the worker's
        # own admission gate remains the backstop.
        claim = _Claim(run_id, worker_id, peak)
        claim.start_refresher()
        with _ACTIVE_LOCK:
            _ACTIVE[run_id] = claim
        return claim

    if evicted:
        reservation_registry.note_make_room(run_id, evicted)
    if not ok:
        if _refuse_enabled():
            # Honest refusal (opt-in) — release the claim so we don't hold phantom
            # bytes, then raise so the dispatch path terminals the run as
            # gpu_unavailable. Only sound once the peak is a MEASURED true need.
            reservation_registry.release(run_id, reason=(refusal or {}).get("reason"),
                                         state="released")
            logger.info("reservation REFUSED for %s (run %s): peak=%s free=%s "
                        "evicted=%s", job_name, run_id, peak,
                        (refusal or {}).get("free_bytes"), evicted)
            raise ReservationRefused(refusal or {"reason": "GPU reservation refused"})
        # DEFAULT (best-effort): make-room did what it safely could; the peak is
        # only the whole-GPU ENVELOPE, and the render autofits/offloads to the
        # remaining VRAM. PROCEED — hold the claim (accounting) and let the render's
        # autofit + the worker admission gate decide the real fit. Never blocks a
        # render on the envelope; never OOMs (the worker gate is the backstop).
        logger.info("reservation best-effort for %s (run %s): could not reach "
                    "envelope peak=%s (free=%s, evicted=%s) — proceeding; the "
                    "render autofits + the worker gate is the fit authority",
                    job_name, run_id, peak, (refusal or {}).get("free_bytes"),
                    evicted or "none")

    claim = _Claim(run_id, worker_id, peak)
    claim.start_refresher()
    with _ACTIVE_LOCK:
        _ACTIVE[run_id] = claim
    logger.info("reservation HELD for %s (run %s) on %s: peak=%s evicted=%s",
                job_name, run_id, worker_id, peak, evicted or "none")
    return claim


def release(run_id: str, reason: Optional[str] = None) -> None:
    """Release a run's claim on ANY terminal path (done/failed/cancelled/abort).
    Idempotent + best-effort — a double release or an unknown run is a clean no-op.
    Stops the refresher so the lease stops heartbeating."""
    if not run_id:
        return
    with _ACTIVE_LOCK:
        claim = _ACTIVE.pop(run_id, None)
    if claim is not None:
        try:
            claim.stop()
        except Exception:  # noqa: BLE001
            pass
    try:
        reservation_registry.release(run_id, reason=reason or "run terminal")
    except Exception:  # noqa: BLE001 — release is best-effort; the lease TTL is the backstop
        pass
