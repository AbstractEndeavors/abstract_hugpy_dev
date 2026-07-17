"""GPU worker registry.

A *worker* is a remote box that runs the standalone worker agent
(``abstract_hugpy_dev.worker_agent``), exposes an HTTP inference endpoint, and
joins this central node so its GPU(s) can serve one or more models from the
manifest.

This module is the single source of truth for the pool. It owns:

    - persistence of the worker list to a JSON file beside the model manifest
      (so the pool survives restarts),
    - registration / heartbeat / removal,
    - model assignment (which worker may serve which model_key),
    - liveness (a worker is ``online`` only if it has heartbeat-ed recently),
    - selection (pick an online worker that is assigned + ready for a model).

Routing (chat/streaming) and the ``/llm/workers`` routes are dumb consumers of
the functions exported here.
"""
from __future__ import annotations

import os
import json
import time
import uuid
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import fcntl  # POSIX advisory file locks — cross-process coordination.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from .schemas import settings

import logging
logger = logging.getLogger(__name__)


# ── assignment memory (daylight 4b) ─────────────────────────────────────────
# Operator designations are WORKER-lifetime, not row-lifetime: a console
# dead-worker sweep DELETE (or a registry loss) used to wipe a worker's
# models + per-model spill, so a re-register came back empty (2026-07-03:
# computron lost 4 of 7 designations). Every assign/unassign snapshots the
# worker's designations here, keyed by its persistent worker id; a fresh-row
# re-register with a known id restores them. Deleting a row deliberately does
# NOT delete its memory — that's the point.

def _assign_memory_path() -> str:
    return os.path.join(os.path.dirname(settings.manifest_path),
                        "worker_assignments.json")


def _load_assign_memory() -> Dict[str, Any]:
    try:
        with open(_assign_memory_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember_assignments(worker: Dict[str, Any]) -> None:
    """Snapshot one worker's designations (models + spill) into the memory."""
    wid = worker.get("id")
    if not wid:
        return
    try:
        mem = _load_assign_memory()
        mem[wid] = {
            "name": worker.get("name"),
            "models": list(worker.get("models") or []),
            "spill_by_model": dict(worker.get("spill_by_model") or {}),
            "remembered_at": _now(),
        }
        tmp = _assign_memory_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(mem, fh, indent=1)
        os.replace(tmp, _assign_memory_path())
    except Exception as exc:  # noqa: BLE001 — memory is best-effort, never fatal
        logger.warning("assignment memory write failed for %s: %s", wid, exc)


def _default_workers_path() -> str:
    """Sit the worker registry next to the model manifest (…/projects/)."""
    return os.path.join(os.path.dirname(settings.manifest_path), "workers.json")


# A worker that hasn't checked in within this window is considered offline.
HEARTBEAT_TIMEOUT_SECONDS = 45.0


def _provision_stall_seconds() -> float:
    """Forward-progress silence (seconds) after which a provisioning entry stops
    reading as an in-flight pull.

    Same shape + rationale as comms.jobs._stall_seconds (the orphan-job fix):
    read at COMPUTE time so an operator can retune without a restart, and a bad
    value degrades to the default rather than raising into a read path.

    Default 600s. Deliberately MUCH larger than the 45s heartbeat timeout: a
    real pull can legitimately go quiet for minutes (a slow segment, a stalled
    HF mirror, a big file's final flush), and calling a live transfer "dead"
    would strip its eviction guard mid-write — the one truly destructive
    mistake available here. Offline workers are already caught by the cheaper
    liveness gate, so this window only has to cover the ONLINE-but-wedged case.
    """
    raw = (os.environ.get("HUGPY_PROVISION_STALL_SECONDS") or "").strip()
    if not raw:
        return 600.0
    try:
        v = float(raw)
        return v if v > 0 else 600.0
    except ValueError:
        return 600.0


def _live_provisioning(worker: Dict[str, Any]) -> set:
    """The subset of ``worker['provisioning']`` that is a GENUINELY LIVE pull.

    Defect (operator, 2026-07-16): a ``provisioning`` entry was immortal. The
    worker announces the list in its heartbeat and removes an entry in a
    ``finally``; if the process dies mid-pull, that ``finally`` never runs, so
    central reported "provisioning" forever (observed: op offline 2h+, still 4
    entries; ae online with 63 entries and ZERO bytes moving).

    This is the orphan-job defect class — state that ages on writes which STOP
    ARRIVING when the writer dies. The recorded lesson: age on PROGRESS, not on
    presence-in-a-list. So an entry is live only when BOTH hold:

      1. the worker is alive       -> REUSES ``_is_online`` (the single existing
                                      staleness notion; no second rule invented)
      2. its bytes are moving      -> a ``provision_progress`` entry whose
                                      ``done_bytes`` advanced within the stall
                                      window, per the central-stamped
                                      ``progressed_at`` clock (see ``heartbeat``)

    Why (2) needs a central clock rather than ``frac > 0``: op's dead pull is
    frozen at ``frac=0.0722`` with 1.8GB done. A truthy frac only proves bytes
    moved ONCE — never that they are moving NOW. Only elapsed-time-since-advance
    can tell a live 7% from a corpse stuck at 7%.

    QUEUED-NOT-STALLED (why absence of an entry is not evidence of death): the
    worker adds a key to ``_provisioning`` at KICK time but only creates a
    ``_provision_progress`` entry once its download callback fires, and
    ``WORKER_PROVISION_CONCURRENCY`` defaults to 1. So ae's 63 progress-less
    entries are models QUEUED behind the semaphore, not wedged ones — correctly
    NOT in-flight (nothing is transferring), and equally correctly NOT
    eviction-protected (they have no bytes on disk to protect).

    Fail-SAFE toward the live case: if the clock is missing/garbage on an ONLINE
    worker with a progress entry, treat it as live. A false "live" costs a
    delayed console pill; a false "dead" could unprotect a real in-flight write.
    """
    prov = set(worker.get("provisioning") or [])
    if not prov or not _is_online(worker):
        # Offline/stale worker: nothing it last claimed is in flight, because
        # nothing of it is running. This is the op case.
        return set()
    progress = worker.get("provision_progress") or {}
    if not isinstance(progress, dict):
        return set()
    now = _now()
    window = _provision_stall_seconds()
    live = set()
    for mk in prov:
        entry = progress.get(mk)
        if not isinstance(entry, dict):
            continue          # queued behind the concurrency semaphore (ae case)
        ts = entry.get("progressed_at")
        if ts is None:
            live.add(mk)      # fail-safe: pre-clock/legacy entry on a live worker
            continue
        try:
            if (now - float(ts)) <= window:
                live.add(mk)
        except (TypeError, ValueError):
            live.add(mk)      # fail-safe: never unprotect on a garbage clock
    return live


def tracked_pkg_name() -> str:
    """Distribution name workers track + central reports its version of.

    Must match the worker's ``--pkg-name`` (``WORKER_PKG_NAME``). Default is the
    dev distribution.
    """
    return os.environ.get("HUGPY_PKG_NAME", "abstract_hugpy_dev")


def required_pkg_version() -> Optional[str]:
    """The dev package version central wants every worker to be running.

    Advertised back to workers in every register/heartbeat response. Resolution
    order:
      1. ``HUGPY_REQUIRED_PKG_VERSION`` env (explicit pin), then
      2. a ``required_pkg_version`` file beside the manifest, then
      3. **central's own installed version** of the tracked dist.

    (3) is the zero-config path: the existing deploy (`pip install -U
    <dist>` on central) becomes the signal — workers converge to whatever
    version central is itself running. ``None`` (dist not installed, no override)
    means "not managing versions" and workers never self-update.
    """
    env = os.environ.get("HUGPY_REQUIRED_PKG_VERSION")
    if env and env.strip():
        # Explicit operator pin — honored verbatim, INCLUDING a PEP 440 local
        # ("+build") version, for a fleet deliberately set up to install from
        # central's private --pkg-index.
        return env.strip()

    # The file/installed fallbacks must resolve to a PUBLICLY installable version.
    # A local version (contains "+", e.g. "0.1.51+c8b13590d") only exists on
    # central's private index; advertising one to the common PyPI-based worker
    # makes its self-update fail on a version pip can't find (rc=1 every
    # heartbeat) and would force a downgrade off a newer public release. So a
    # local fallback version means "not managing versions" → workers stay put.
    def _public(v: Optional[str]) -> Optional[str]:
        return v if (v and "+" not in v) else None

    path = os.environ.get("HUGPY_REQUIRED_PKG_VERSION_FILE") or \
        os.path.join(os.path.dirname(settings.manifest_path), "required_pkg_version")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            pinned = fh.read().strip()
        if pinned:
            return _public(pinned)
    except OSError:
        pass
    # Do NOT auto-derive a pin from central's own installed version. That dev
    # build is frequently a local "+build" (e.g. 0.1.41+phone5) that PyPI workers
    # can't install, and even a clean value would silently downgrade a worker on
    # a newer public release. Version management is therefore OPT-IN: set a clean
    # public version via HUGPY_REQUIRED_PKG_VERSION or the required_pkg_version
    # file. Otherwise central does not manage worker versions (workers stay put).
    return None


def pkg_index_dir() -> str:
    """Directory of built wheels that central serves as a PEP-503 simple index.

    The ``sync.trigger`` build drops the freshly-built dev wheel here. Override
    with ``HUGPY_PKG_INDEX_DIR``; defaults to a ``pip_index`` dir beside the
    model manifest.
    """
    return os.environ.get("HUGPY_PKG_INDEX_DIR") or \
        os.path.join(os.path.dirname(settings.manifest_path), "pip_index")


def _now() -> float:
    return time.time()


def _is_online(worker: Dict[str, Any]) -> bool:
    last = worker.get("last_seen") or 0
    return (_now() - last) <= HEARTBEAT_TIMEOUT_SECONDS


def _public_view(worker: Dict[str, Any]) -> Dict[str, Any]:
    """The shape returned to API callers — derived ``status`` included.

    ``status`` is *liveness* (online/offline from last_seen). ``admission`` is the
    operator gate (pending/approved/blocked) and is independent of liveness. Rows
    written before the admission feature have no ``admission`` key; they are
    grandfathered to ``approved`` here so an existing fleet keeps serving.
    """
    return {
        **worker,
        **_vram_summary(worker),
        **_ram_summary(worker),
        # Derived local-storage view + guarded LRU eviction proposal, recomputed
        # on every read from already-stored fields (same pure-function pattern as
        # the vram/ram summaries above; no daemon, no auto-fire — nothing deletes
        # here). Overwrites the raw ``storage`` heartbeat field with the enriched
        # console-facing shape (over_budget + proposed_evictions[]).
        "storage": storage_proposal(worker),
        # IN-FLIGHT PULLS ONLY (2026-07-16). The raw record keeps whatever the
        # worker last announced; the PUBLIC view reports only pulls that are
        # actually moving, so a dead/stalled entry can never render as an
        # active transfer ("defaults are promises" — a row that says "working
        # on it" when nothing is working is a lie). Derived here, on every read,
        # like status/storage above — no daemon, no sweep. The console's
        # ⏳ pulling pill and its provision_progress % both key off this list,
        # so an assigned-but-absent model correctly falls through to "missing".
        "provisioning": sorted(_live_provisioning(worker)),
        "status": "online" if _is_online(worker) else "offline",
        "admission": worker.get("admission", "approved"),
        # SYSTEM-authored placement grants (Phase 1 item 2) — separate from the
        # operator-designated ``models`` list. Never treat a missing key as
        # absence-of-feature; always surface the (possibly empty) dict so
        # console/tests can see grants land and clear.
        "grants": dict(worker.get("grants") or {}),
    }


def _clamp_limits(limits: Dict[str, Any], caps: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp operator limits to the worker's own configured caps.

    The worker's unit config is authoritative ("central shall be forced to
    view that as its max") — central can set anything LESS, never more."""
    out: Dict[str, Any] = {}
    for k, v in limits.items():
        cap = caps.get(k)
        if cap is not None:
            try:
                v = min(float(v), float(cap))
                if k == "threads":
                    v = int(v)
            except (TypeError, ValueError):
                continue
        out[k] = v
    return out


# Default runtime-environment tier. A worker that doesn't report its env (older
# agent) and a model with no explicit requirement both resolve to this, so a
# pre-feature fleet keeps matching exactly as before the tier gate existed.
DEFAULT_ENV_TIER = "stable"


def _model_env_tiers() -> Dict[str, str]:
    """Operator map of model -> REQUIRED env tier.

    Parsed from ``HUGPY_MODEL_ENV_TIERS`` = ``"key:tier,key2:tier"`` (e.g.
    ``"Qwen3.6-27B-AEON:edge"``). A model not listed requires the default tier,
    so this whole gate is a no-op until the operator maps a model.
    """
    out: Dict[str, str] = {}
    for part in os.environ.get("HUGPY_MODEL_ENV_TIERS", "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, tier = part.rpartition(":")
        key, tier = key.strip(), tier.strip().lower()
        if key and tier:
            out[key] = tier
    return out


def env_tier_for_model(model_key: str) -> str:
    """The runtime-env tier ``model_key`` requires (alias-tolerant lookup)."""
    tiers = _model_env_tiers()
    if not tiers:
        return DEFAULT_ENV_TIER
    wanted = _match_keys(model_key)
    for key, tier in tiers.items():
        if key == model_key or (_match_keys(key) & wanted):
            return tier
    return DEFAULT_ENV_TIER


def _worker_env_tier(worker: Dict[str, Any]) -> str:
    """The env tier a worker ADVERTISES (from its own venv, via register/heartbeat).

    Workers that don't report an env (older agents) are treated as serving the
    default tier — the grandfather rule that keeps a pre-feature fleet routing
    unchanged. An edge model therefore only lands on a worker that AFFIRMATIVELY
    advertises the edge env.
    """
    env = worker.get("env")
    tier = env.get("tier") if isinstance(env, dict) else None
    if tier is None:
        return DEFAULT_ENV_TIER
    return str(tier).strip().lower() or DEFAULT_ENV_TIER


def _engine_unusable(worker: Dict[str, Any]) -> bool:
    """True only when a worker EXPLICITLY reports it has no inference engine.

    Workers that don't report engine status (older agents) are assumed capable,
    so this never excludes a pre-feature fleet — it only skips a worker that
    affirmatively says ``engine.installed == False`` (e.g. llama-cpp missing),
    which would otherwise be picked and fail every request.
    """
    eng = worker.get("engine")
    return isinstance(eng, dict) and eng.get("installed") is False


# Tasks whose worker capability is authoritatively gated ELSEWHERE and must NOT be
# re-filtered by the find_spec-derived task_capabilities map:
#   image-text-to-text — vision uses the stricter engine.supports_vision truth
#     (llama.cpp mtmd build) enforced in resolvers.remote; a transformers-VL worker
#     without llama_cpp would advertise this False and be wrongly skipped here.
_TASK_CAP_GATE_EXCLUDE = {"image-text-to-text"}


def _task_capable(worker: Dict[str, Any], task: Optional[str]) -> bool:
    """Whether ``worker`` can run ``task`` per its advertised ``task_capabilities``.

    Capability-honest, exactly like the engine/vision/tier gates: a worker is
    skipped ONLY when it AFFIRMATIVELY advertises the task as unavailable — the
    2026-07-11 request-time-failure class (a canonical venv missing
    sentence-transformers / whisper / keybert). LEGACY workers (no
    ``task_capabilities`` field) and tasks a worker doesn't enumerate are assumed
    capable, so a pre-feature fleet routes exactly as before. A ``None`` task
    (non-ML routing, e.g. video auto-pick) never gates, and vision defers to the
    stricter engine gate (``_TASK_CAP_GATE_EXCLUDE``).
    """
    if not task or task in _TASK_CAP_GATE_EXCLUDE:
        return True
    caps = worker.get("task_capabilities")
    if not isinstance(caps, dict) or task not in caps:
        return True
    return bool(caps.get(task))


def _comfy_id_lock_capable(worker: Dict[str, Any]) -> bool:
    """Whether ``worker``'s ComfyUI can do identity-locked STILLs — the
    IPAdapter node pack is installed (``comfy.id_lock`` advertised True from the
    agent's object_info probe).

    STRICT / affirmative-only, DELIBERATELY UNLIKE ``_task_capable``'s
    legacy-permissive default: an id_lock request must land on a box that PROVABLY
    has the nodes, because silently degrading to a NON-locked image is forbidden
    (WORKER-SETUP §5b / comfy_runner). A worker with no ``comfy`` block, comfy
    unavailable, or ``id_lock`` != True does NOT qualify. There's no legacy fleet
    to preserve here — id_lock is a brand-new capability, so "unknown" means "not
    yet", not "assume yes".
    """
    comfy = worker.get("comfy")
    if not isinstance(comfy, dict) or not comfy.get("available"):
        return False
    return bool(comfy.get("id_lock"))


def _has_usable_gpu(worker: Dict[str, Any]) -> bool:
    """Whether the worker advertises a GPU with free VRAM (for efficiency ranking).

    Capability-honest, like vision routing: a worker whose llama.cpp build
    AFFIRMATIVELY reports it cannot offload (engine.supports_gpu_offload is
    False) ranks as GPU-less no matter what nvidia-smi shows — n_gpu_layers is
    silently ignored by a CPU-only wheel, so its "GPU" would never be used.
    Older agents that don't report the flag keep their GPU credit (no guessing).
    """
    if (worker.get("engine") or {}).get("supports_gpu_offload") is False:
        return False
    return any((g.get("memory_free") or 0) > 0 for g in (worker.get("gpus") or []))


def _vram_summary(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Flat, normalized GPU/VRAM rollup derived from the per-GPU ``gpus[]`` list.

    The registry stores VRAM only nested inside ``gpus[]`` (refreshed every
    heartbeat), so each consumer had to dig — and the worker summary read as
    "gpu: None" even for a box with a real card. This surfaces a single primary
    GPU name plus summed totals so the console and a VRAM-fit ("worker slot")
    allocator can read flat fields, exactly the way the local slot pool reads a
    flat RAM number. All counts are bytes; ``None`` where unknown (never
    fabricated — an empty/again-unreported ``gpus`` yields None, not 0).
    """
    gpus = [g for g in (worker.get("gpus") or []) if isinstance(g, dict)]
    if not gpus:
        return {"gpu": None, "gpu_count": 0, "vram_total": None, "vram_free": None, "vram_used": None}
    name   = next((g.get("name") for g in gpus if g.get("name")), None)
    totals = [g.get("memory_total") for g in gpus if g.get("memory_total")]
    frees  = [g.get("memory_free")  for g in gpus if g.get("memory_free") is not None]
    vram_total = sum(totals) if totals else None
    vram_free  = sum(frees)  if frees  else None
    vram_used  = (vram_total - vram_free) if (vram_total is not None and vram_free is not None) else None
    return {
        "gpu": name,
        "gpu_count": len(gpus),
        "vram_total": vram_total,
        "vram_free": vram_free,
        "vram_used": vram_used,
    }


def _ram_summary(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Flat RAM rollup — the CPU-tier mirror of _vram_summary.

    ``ram_total`` is the box's RAW installed memory (MemTotal, reported by the
    agent). ``ram_used`` is derived as ``ram_total - free_ram`` — but note
    free_ram is reserve-adjusted AND HUGPY_RAM_MAX_GIB-capped, so this reads as
    "used incl. reserve/headroom", NOT pure model RSS (the console labels it so).
    None where unknown (never fabricated — mirrors _vram_summary's discipline);
    clamped to >=0 so reserve accounting can't yield a negative width.
    """
    ram_total = worker.get("ram_total")
    free_ram = worker.get("free_ram")
    ram_used = (
        max(0, ram_total - free_ram)
        if (ram_total is not None and free_ram is not None)
        else None
    )
    return {"ram_total": ram_total, "ram_used": ram_used}


def _disk_reserve_bytes() -> int:
    """Free-space reserve (bytes) kept on a worker's MODEL-ROOT volume.

    Below this reserve a worker is "over budget" and its COLD local models become
    eviction candidates. Sized to comfortably exceed the largest single model
    pull (~45 GiB per the model_cache header note) so a provision can always land
    after one eviction. Override with ``HUGPY_WORKER_DISK_RESERVE_GIB`` (default
    50). This is DISTINCT from ``HUGPY_MODEL_CACHE_MAX_GIB`` (=450, the separate
    SSD hot-cache bound in managers/serve/model_cache.py) — do not conflate: this
    reserve is on the model-root disk, not the SSD cache.
    """
    try:
        gib = float(os.environ.get("HUGPY_WORKER_DISK_RESERVE_GIB", "50"))
    except (TypeError, ValueError):
        gib = 50.0
    if gib < 0:
        gib = 0.0
    return int(gib * (1 << 30))


def _model_size_bytes(model_key: str) -> Optional[int]:
    """One ASSIGNED model's size per central's manifest, or None if unknowable.

    The same source ``worker_agent/provision.central_total_bytes`` resolves for a
    single pull, read LOCALLY here (central owns the manifest and the model dirs,
    so this needs no HTTP — see allocated_totals for why that matters).

    GGUF honesty: a GGUF dir holds SEVERAL quants, so its directory sum is NOT
    what serving costs. ``effective_bytes`` (gguf_variants_detail) is the quant
    that actually serves — the same number the Models tab shows. Falls back to
    the mtime-cached directory footprint for transformers/comfy.

    None is a FIRST-CLASS answer meaning "central cannot say" (not in the
    manifest / not on disk / sizing raised). Callers MUST count it as unknown and
    report it — never coerce it to 0, which would make an over-subscribed
    assignment set read as comfortably fitting (the exact dishonesty this
    feature exists to remove).
    """
    if not model_key:
        return None
    try:
        # Import depths differ and are NOT interchangeable: this module sits at
        # flask_app/app/functions/imports/utils/, so `routes` is 4 up while the
        # TOP-LEVEL `imports` package (abstract_hugpy_dev.imports — a different
        # tree from this one's own `imports` parent) is 6. Getting this wrong
        # raises ModuleNotFoundError, which an over-broad except would swallow
        # into a permanent "size unknown" — every model silently unsized, an
        # over-subscribed set reading as empty. Logged loudly for that reason.
        from ....routes.llm_storage_routes import _annotate_gguf_size, _annotate_size
        from ......imports.config.models.models_config import get_models_dict
    except Exception as exc:  # noqa: BLE001 — sizing must never break a read
        logger.warning("allocation sizing unavailable (%s) — assigned-set totals "
                       "will report as unknown", exc)
        return None
    try:
        manifest = get_models_dict(dict_return=True) or {}
        entry = manifest.get(model_key)
        if not isinstance(entry, dict):
            return None
        # Work on a COPY: the annotators mutate the dict they are handed, and the
        # registry entry is the cached, shared MODEL_REGISTRY_DICT row.
        model = dict(entry)
        _annotate_gguf_size(model, model_key)   # -> effective_bytes (GGUF quant)
        _annotate_size(model, model_key)        # -> size_bytes (eff or dir sum)
        size = model.get("size_bytes")
        return int(size) if size else None
    except Exception:  # noqa: BLE001 — unknown size is a valid answer here
        return None


def allocated_totals(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Size the worker's ASSIGNMENT SET against its budget — the STRUCTURAL view.

    OPERATOR (2026-07-16): "it should also show how much is needed based on the
    total size of all models allocated". The per-pull refusal ("this 23.5 GiB
    pull won't fit") answers a different, smaller question. This answers: can the
    ASSIGNED SET fit AT ALL? A worker assigned 12 models totalling 180 GiB
    against a 50 GiB budget is over-subscribed BY CONSTRUCTION — no eviction
    order rescues it, and it will wedge on some future call no matter which model
    is unlucky enough to be the one that asks.

    Domain = ``worker['models']`` — the OPERATOR DESIGNATION set written by
    assign_model/unassign_model. NOT the on-disk inventory (lazy-download means
    an assigned model routinely has no files yet — sizing only what landed would
    UNDER-report an over-subscribed set, hiding the very thing this shows) and
    NOT ``grants`` (system-authored, freely evictable, never operator intent).

    Returns::

        {"allocated_total_bytes": int,      # sum of the KNOWN-size models
         "allocated_count": int,            # models in the assignment set
         "allocated_unknown_count": int,    # sizes central couldn't resolve
         "allocated_over_budget_bytes": int}  # total - budget, 0 when it fits

    ``allocated_total_bytes`` is a FLOOR when allocated_unknown_count > 0: the
    unknowns are counted and surfaced, never silently zeroed, so a reader can see
    the number is incomplete rather than trust a comfortable-looking lie.
    """
    models = [m for m in (worker.get("models") or []) if m]
    total = 0
    unknown = 0
    for mk in models:
        size = _model_size_bytes(mk)
        if size:
            total += size
        else:
            unknown += 1
    return {"allocated_total_bytes": total,
            "allocated_count": len(models),
            "allocated_unknown_count": unknown}


def storage_proposal(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a worker's local-STORAGE view + a guarded LRU eviction PROPOSAL.

    A PURE read-time computation over already-stored heartbeat fields — no
    daemon, no background loop, no persistent toggle, and THIS FUNCTION never
    deletes anything (it returns a proposal; a caller must act on it). It is
    spread into every worker read by ``_public_view`` (the always-on storage
    monitoring depiction) and re-run by the ``/reap-approve`` route as its
    central second guard, so the console preview and the approval share one
    source of truth.

    NOTE (2026-07-16): "no auto-fire" describes THIS central preview and the
    operator-gated bulk reaper it feeds — it is NOT a fleet-wide claim. The
    worker's ``worker_agent/budget.py`` auto-evicts on the PROVISION path
    (call-driven only) to seat a model being pulled. That path deliberately
    reuses THIS function's ordering + guard semantics (unprotected candidates,
    ascending last_picked, largest-first among equally-cold) so the console's
    preview and an auto-evict can never disagree about what would go.

    Inputs (all raw worker-record fields):
      - ``worker['storage']``   the worker-reported survey
            ``{cache_used_bytes, disk_free, models:[{model_key, bytes, pinned,
            loaded, loading, provisioning, assigned, protected, why}]}``.
            ABSENT on a pre-feature agent -> a monitoring-only view with an empty
            proposal (the worker must ship this field for the proposal to have a
            per-model inventory).
      - ``worker['disk']``      ``{free_bytes, total_bytes}`` of the model root.
      - ``worker['model_last_picked']`` central LRU signal ``{model_key: epoch}``
            stamped in ``pick_for_model``. A missing entry defaults to 0 (coldest
            -> proposed for eviction first — exactly right for never-served
            test-churn leftovers).
      - ``worker['limits']['disk_cache_gib']`` optional explicit per-worker cap;
            WINS over the free-disk reserve when set.
      - ``worker['loaded_models']`` / ``['loading']`` / ``['provisioning']``
            central slot-merged live truth — the redundant central guard that
            closes the worker reaper's in-process-only loaded gap.
      - ``worker['config']['residency']`` / ``['pinned']`` static/pin attribution.

    Budget (two modes, cheap; explicit cap wins):
      * explicit cap  -> over_budget ``cache_used > cap``;  need ``cache_used-cap``
      * else reserve  -> over_budget ``disk_free < reserve``; need ``reserve-disk_free``

    Proposal (mirrors ``model_cache.evict_for``): domain = RECLAIMABLE candidates
    only (unprotected), sorted ASCENDING by ``last_picked`` (LRU oldest-first),
    greedily accumulating bytes until ``need`` is covered — that subset (possibly
    several models) is ``proposed_evictions``. The console renders it; it computes
    nothing.

    📌 PIN + ALLOCATION HAVE NO BEARING ON EVICTION (operator ruling,
    2026-07-17, verbatim): "the pins only should designate that the model
    allocation survives restarts. the allocation only stipulates the routing for
    that model (to that worker). neither of those should have any bearing on the
    pull or eviction, unless its to do with priority, then a pinned model should
    take higher precidence than unpinned, but even that is trivial".
      * 📌 pin = the model's ALLOCATION survives restarts (and unassign — the
        409). Nothing else. Allocation = ROUTING (which worker answers).
      * A pinned or assigned model's FILES are a normal LRU eviction candidate
        here — ``proposed_evictions`` MAY include pinned files. Evicting them
        leaves pin + allocation untouched; the bytes re-pull on next call.
      * Pin's only eviction role is the trivial FIFO tiebreak below (unpinned
        proposed first at an exact last_picked tie).
      * The ``pinned``/``why`` fields stay HONEST as ATTRIBUTION info (a row can
        read pinned:true / why:"pinned" while protected:false). 🔒static is the
        ONLY durable local-presence guard; loaded/loading/provisioning are
        live-use guards. This removed the day-one tripwire (attribution/routing
        masquerading as a disk shield). unassign-409 is UNTOUCHED — that IS pin.
    """
    storage = worker.get("storage")
    reported = isinstance(storage, dict)
    disk = worker.get("disk") or {}

    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    disk_free = None
    if reported and storage.get("disk_free") is not None:
        disk_free = _as_int(storage.get("disk_free"))
    if disk_free is None and disk.get("free_bytes") is not None:
        disk_free = _as_int(disk.get("free_bytes"))
    disk_total = _as_int(disk.get("total_bytes"))

    cache_used = _as_int(storage.get("cache_used_bytes")) if reported else None
    # ORPHANED (unattributed-on-disk) residue reported by the worker (release-
    # bound field). Passed through verbatim: model dirs / stalled .part sets on
    # disk that match NO current assignment (computron's 5.7G Qwen2.5-VL-3B
    # .part junk). Absent on a pre-2026-07-17 agent -> zeros (feature-off).
    orphaned_bytes = (_as_int(storage.get("orphaned_bytes")) or 0) if reported else 0
    orphaned_count = (_as_int(storage.get("orphaned_count")) or 0) if reported else 0
    orphaned_items = (storage.get("orphaned_items") or []) if reported else []
    last_picked_map = worker.get("model_last_picked") or {}
    limits = worker.get("limits") or {}
    cfg = worker.get("config") or {}
    residency = cfg.get("residency") or {}
    pinned_cfg = cfg.get("pinned") or {}
    # Central slot-merged live truth — closes the reaper's in-process-only
    # loaded_model_keys() gap (it misses slot occupants / answering models).
    loaded_now = set(worker.get("loaded_models") or [])
    loading_now = set(worker.get("loading") or [])
    # LIVE pulls only — a stale/dead-owner entry is neither reported as
    # in-flight nor granted eviction protection. See _live_provisioning.
    provisioning_now = _live_provisioning(worker)

    reserve = _disk_reserve_bytes()

    # ── budget: explicit per-worker cap wins over the free-disk reserve ──────
    cap_gib = limits.get("disk_cache_gib")
    budget_basis = "reserve"
    budget = None
    over_budget = False
    need_bytes = 0
    if cap_gib not in (None, ""):
        cap_bytes = None
        try:
            cap_bytes = int(float(cap_gib) * (1 << 30))
        except (TypeError, ValueError):
            cap_bytes = None
        if cap_bytes is not None:
            budget_basis = "cap"
            budget = cap_bytes
            if cache_used is not None and cache_used > cap_bytes:
                over_budget = True
                need_bytes = cache_used - cap_bytes
    if budget_basis == "reserve":
        # Express the cache-ceiling budget so the console bar (cache_used vs
        # budget) is consistent with the flag: over_budget <=> cache_used > budget
        # <=> disk_free < reserve. need is the free-disk shortfall.
        if disk_free is not None and cache_used is not None:
            budget = cache_used + disk_free - reserve
        if disk_free is not None and disk_free < reserve:
            over_budget = True
            need_bytes = reserve - disk_free

    # ── per-model view + reclaimable candidate domain ───────────────────────
    def _lp(mk):
        v = last_picked_map.get(mk)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    grants_now = worker.get("grants") or {}
    models_out: List[Dict[str, Any]] = []
    candidates: List[tuple] = []   # (last_picked, bytes, model_key)
    raw_models = storage.get("models") if reported else None
    for m in (raw_models or []):
        if not isinstance(m, dict) or not m.get("model_key"):
            continue
        mk = m["model_key"]
        b = _as_int(m.get("bytes")) or 0
        lp = _lp(mk)
        # Central-final protection = worker's own flag OR any redundant central
        # guard (slot-merged loaded/loading, provisioning, static, pin). A model
        # is a candidate ONLY if unprotected on BOTH sides.
        #
        # PROVISIONING is the ONE guard in this chain that is liveness-gated
        # (2026-07-16). The worker's own per-row ``m['provisioning']`` flag is
        # NOT consulted here, unlike loaded/loading: it comes from the same dead
        # heartbeat snapshot as the stale list, so honouring it would re-admit
        # exactly the phantom protection this fix removes (op's dead pull still
        # flags its row). Central instead trusts only _live_provisioning.
        #
        # This LOSES no protection for a real pull: an ONLINE worker with a
        # moving pull is live by construction. And the worker keeps its OWN
        # authoritative guard locally (worker_agent/budget.py's
        # _PROTECTED_REASONS) — a box never deletes under its own live write on
        # central's say-so. What it REMOVES is permanent phantom protection: a
        # dead entry used to make a real, cold, reclaimable file un-evictable
        # forever, silently shrinking the reclaimable pool on a full disk.
        #
        # NOTE (Phase 1 item 2, grant markers): a SYSTEM grant is DELIBERATELY
        # ABSENT from this chain — the opposite of an operator "assigned"
        # designation. A model that is ONLY granted (not assigned/static/loaded)
        # gets no protection here and remains a normal LRU eviction candidate;
        # grants are reclaimable by construction, never a residency guarantee.
        # If a model happens to be BOTH granted and assigned/static/etc., that
        # FILE-PROTECTING designation still protects it as before — the grant
        # itself contributes nothing. (📌pin is NOT file-protecting as of
        # 2026-07-17, so granted+pinned is a candidate — see below.)
        why = m.get("why") or ""
        protected = bool(m.get("protected"))
        is_pinned = bool(m.get("pinned") or pinned_cfg.get(mk))
        # Worker-reported protection is trusted ONLY for reasons that are not
        # pure attribution ("shared/central storage — never reaped", "model
        # store not marked reapable", live-use guards). A released worker
        # (<=0.1.183) still stamps protected/why="pinned" or "assigned" from the
        # old doctrine — strip those two here and let the chain below recompute,
        # or the fleet's stale flags keep the day-one tripwire alive centrally
        # until the next release (keeper, 2026-07-17: ae reported 21/21 pinned
        # rows protected with zero live-use flags — reclaimable pool read as
        # empty on a 700G/429G box).
        if protected and why in ("pinned", "assigned"):
            # Clear the stale label too: if a live-use guard re-protects below,
            # the chain stamps the HONEST reason (loaded/loading/…) instead of
            # leaving attribution vocabulary on a protection flag.
            protected, why = False, ""
        # 📌 pin does NOT protect files (operator ruling, 2026-07-17): "the pins
        # only should designate that the model allocation survives restarts. the
        # allocation only stipulates the routing... neither of those should have
        # any bearing on the pull or eviction". So pin is DELIBERATELY absent
        # from this protection chain — a pinned model's files are a normal LRU
        # candidate; the pin + allocation survive the eviction and the bytes
        # re-pull on next call. `pinned` is still reported below as ATTRIBUTION
        # (m['pinned']) but never sets `protected`/`why`. 🔒static is the ONLY
        # durable local-presence guard; loaded/loading/provisioning are live-use
        # guards. This removes the day-one tripwire that let attribution/routing
        # masquerade as a disk shield.
        if not protected:
            if str(residency.get(mk) or "").lower() == "static":
                protected, why = True, why or "static"
            # NO `assigned` branch (operator ruling 2026-07-17): "the allocation
            # only stipulates the routing for that model... neither of those
            # should have any bearing on the pull or eviction." Assignment is
            # attribution, same as pin — worker-side budget._is_protected has
            # said so since f1894b2; this chain protecting `assigned` was the
            # central half of the same day-one tripwire (on a box whose on-disk
            # models are all assigned — ae, op — the reclaimable pool read as
            # permanently empty).
            elif mk in loaded_now or m.get("loaded"):
                protected, why = True, why or "loaded"
            elif mk in loading_now or m.get("loading"):
                protected, why = True, why or "loading"
            elif mk in provisioning_now:
                protected, why = True, why or "provisioning"
            elif is_pinned and not why:
                # ATTRIBUTION-only annotation: honest `why` for a pinned model
                # that has no other protecting flag, while `protected` stays
                # False (it remains a candidate). A bare pinned row therefore
                # shows why="pinned" but IS eligible for the proposal below.
                why = "pinned"
        models_out.append({
            "model_key": mk,
            "bytes": b,
            "last_picked": lp or None,     # None = never served through central
            "protected": protected,
            "why": why,
            "granted": mk in grants_now,   # SYSTEM marker only — confers no protection
            "pinned": is_pinned,           # ATTRIBUTION only — confers no eviction protection (2026-07-17)
            "loaded": bool(m.get("loaded") or mk in loaded_now),
            "loading": bool(m.get("loading") or mk in loading_now),
            # LIVE pulls only (not the worker's stale per-row flag) — this is
            # what the console renders as "⏳ pulling"; a dead pull must read
            # as missing, never as an active transfer.
            "provisioning": mk in provisioning_now,
            "assigned": bool(m.get("assigned")),
        })
        if not protected:
            candidates.append((lp, is_pinned, b, mk))

    proposed: List[Dict[str, Any]] = []
    proposed_free = 0
    if over_budget and need_bytes > 0 and candidates:
        # LRU oldest-first (ascending last_picked). Then a TRIVIAL 📌pin tiebreak
        # (operator called it "trivial and likely unnecessary", 2026-07-17):
        # among equally-stale candidates, propose UNPINNED before PINNED (False
        # sorts first) — a pinned model gets a hair of extra precedence only at
        # an exact last_picked tie. Then largest-first so the budget clears in
        # the fewest deletes; stable key tiebreak. IDENTICAL to budget.fit_plan,
        # so central's preview and the worker's auto-evict agree on what goes.
        candidates.sort(key=lambda c: (c[0], c[1], -c[2], c[3]))
        for lp, _pin, b, mk in candidates:
            if proposed_free >= need_bytes:
                break
            proposed.append({"model_key": mk, "bytes": b,
                             "last_picked": lp or None})
            proposed_free += b

    # ── ALLOCATION-LEVEL view (operator, 2026-07-16) ────────────────────────
    # Structural, and TRUE EVEN WHEN NO PULL IS HAPPENING: if the assigned set
    # itself exceeds the budget, the worker is over-subscribed now — the console
    # can surface that BEFORE some unlucky call wedges. Computed on every read
    # (like the vram/ram summaries) and cheap: sizes come from the cached
    # registry + mtime-cached dir walks, not per-model HTTP.
    alloc = allocated_totals(worker)
    alloc_over = 0
    if budget is not None and alloc["allocated_total_bytes"] > budget:
        alloc_over = alloc["allocated_total_bytes"] - budget
    alloc["allocated_over_budget_bytes"] = alloc_over

    # ── ATTRIBUTED vs RESIDENT (2026-07-17) ─────────────────────────────────
    # The operator scare: assignment/pin ATTRIBUTES a model to a worker without
    # putting bytes on disk (lazy download, 7f0e6e8/2a3baeb). The fleet gauge
    # read cache_used/budget, and an over-subscribed ATTRIBUTION set made a box
    # with nothing transferring look like a runaway download storm. Split the two
    # so attribution can NEVER masquerade as disk pressure:
    #   * attributed = the assignment/pin SET's effective size (may exceed disk;
    #     "assigned but not on disk" is a CORRECT resting state, not pressure).
    #   * resident   = bytes ACTUALLY on disk. The worker's measured cache_used is
    #     the authority; the per-model on-disk sum is the fallback/cross-check.
    # The disk-pressure GAUGE is derived from RESIDENT only.
    resident_from_models = sum(int(m.get("bytes") or 0) for m in models_out)
    resident_bytes = cache_used if cache_used is not None else (
        resident_from_models if reported else None)
    attributed = {
        "attributed_total_bytes": alloc["allocated_total_bytes"],
        "attributed_count": alloc["allocated_count"],
        "attributed_unknown_count": alloc["allocated_unknown_count"],
        "attributed_over_budget_bytes": alloc_over,
    }
    resident = {
        # bytes on disk NOW. `resident_bytes` is the number the gauge must use.
        "resident_bytes": resident_bytes,
        "resident_model_bytes": resident_from_models,
        # measured vs summed can disagree (heartbeat lag / non-model files); both
        # surfaced so the console shows the truth instead of averaging a lie.
        "resident_source": ("measured" if cache_used is not None
                            else ("summed" if reported else "unknown")),
        # ORPHANED = on disk but attributed to NO model (leftover dirs + stalled
        # .part sets). A THIRD class distinct from attributed and
        # resident-attributed: junk eating the drive that the allocation ledger
        # never showed. UI label: "unattributed on disk".
        "orphaned_bytes": orphaned_bytes,
        "orphaned_count": orphaned_count,
        "orphaned_items": orphaned_items,
    }
    # The disk-pressure gauge: RESIDENT over budget. Attribution is deliberately
    # excluded — an over-subscribed assignment set is surfaced via
    # attributed_over_budget_bytes (structural), never as a full-disk reading.
    gauge = {
        "gauge_used_bytes": resident_bytes,   # <-- what the UI bar fills to
        "gauge_budget_bytes": budget,
        "gauge_basis": "resident",
        "gauge_over_budget": over_budget,     # already computed from cache_used/disk_free
    }

    return {
        **alloc,
        **attributed,
        **resident,
        **gauge,
        "reported": reported,
        "cache_used_bytes": cache_used,
        "disk_free": disk_free,
        "disk_total": disk_total,
        "reserve": reserve,
        "budget": budget,
        "budget_basis": budget_basis,
        "over_budget": over_budget,
        "need_bytes": need_bytes if over_budget else 0,
        "proposed_free_bytes": proposed_free,
        "proposed_evictions": proposed,
        "models": models_out,
        # Storage REFUSALS reported by the worker: {model_key: {state:"refused",
        # reason, needs_bytes, budget_bytes, reclaimable_bytes, blocked, ...}}.
        # Models whose pull was refused BEFORE it started because even a full
        # FIFO of the reclaimable models couldn't seat them. Passed through
        # VERBATIM — this is the worker's own verdict about its own disk, and
        # central has no better information to second-guess it with. They have
        # no files on disk, so they are deliberately absent from `models` and
        # never appear in a proposal; the console renders them as MISSING with
        # the reason on hover.
        "refused": (storage.get("refused") or {}) if reported else {},
        # SCAN DIAGNOSTICS (slice 3, B) — passed through VERBATIM from the worker
        # survey so a broken/degraded reap scan can never masquerade as a clean
        # empty store (the ae 2026-07-17 defect: rows:0 while 65 models were on
        # disk). The console can surface scan_error / considered≫rows. Absent on a
        # pre-slice-3 worker -> falsy defaults (feature simply off).
        "scan_error": (storage.get("scan_error") or "") if reported else "",
        "scan_keys_considered": (_as_int(storage.get("scan_keys_considered")) or 0) if reported else 0,
        "scan_rows": (_as_int(storage.get("scan_rows")) or 0) if reported else 0,
        "scan_row_errors": (_as_int(storage.get("scan_row_errors")) or 0) if reported else 0,
        # EFFECTIVE BUDGET (slice 4, min-wins) — the worker's own resolved
        # min(central disk_cache_gib, worker same-drive declarations) + the source
        # map, passed through VERBATIM so the console can show WHY a number
        # governs (e.g. central 400 wins over worker hot 1500). Absent on a
        # pre-slice-4 worker -> None/{}/False (feature simply off). This is the
        # WORKER's own resolution; central's `budget`/`over_budget` above are its
        # own view and unchanged.
        "budget_effective_bytes": (_as_int(storage.get("budget_effective_bytes"))) if reported else None,
        "budget_sources": (storage.get("budget_sources") or {}) if reported else {},
        "budget_cap_not_applicable": bool(storage.get("budget_cap_not_applicable")) if reported else False,
    }


def _match_keys(model_key: str) -> set:
    """Normalized aliases a model might be named by, for tolerant matching.

    A model can be referenced as its registry key, its hub_id (owner/name), or
    just the trailing name — and with different case. We compare on the set of
    these forms so an assignment made via one spelling still routes a chat that
    uses another. Example: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "Qwen2.5-Coder-3B-Instruct-GGUF" and the lowercased variants all match.
    """
    if not model_key:
        return set()
    raw = str(model_key).strip()
    forms = {raw, raw.lower()}
    tail = raw.split("/")[-1]
    forms.add(tail)
    forms.add(tail.lower())
    return forms


class WorkerStore:
    """Disk-authoritative, multi-process-safe registry of GPU workers.

    Under gunicorn/uwsgi the API runs as several processes, so an in-memory
    dict would split-brain: a worker registered in process A would be invisible
    to a heartbeat or chat request handled by process B (the classic symptom is
    "registers + shows in the UI, but heartbeats 410 and chats never offload").

    To avoid that, ``workers.json`` is the single source of truth: every read
    re-loads it, and every mutation takes an exclusive ``fcntl`` lock, reloads,
    mutates, and writes back atomically. A short-lived in-process RLock just
    keeps threads within one process from racing the same fd.
    """

    # Read-cache TTL: the console polls /llm/workers every ~10s; without this
    # every poll does an open+flock+read of workers.json, which BLOCKS on a
    # degraded mount and stalls the API. Reads serve from cache within the TTL;
    # writes always go to disk and refresh the cache, so liveness stays correct.
    _READ_TTL = 3.0

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _default_workers_path()
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_at = 0.0
        self._ensure_parent()

    # -- persistence (disk-authoritative) ----------------------------------
    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

    def _read_unlocked(self, fh=None) -> Dict[str, Dict[str, Any]]:
        """Parse the workers map from an open fh, or from disk if none given.

        A non-empty file that fails to parse is treated as CORRUPTION, not as an
        empty registry: we log and re-raise rather than return {}. Otherwise a
        torn write (this unit restarts often) would be silently 'healed' into an
        empty fleet, and the next write would persist that empty set — wiping
        every worker. Absent/empty files still return {} (normal cold start).
        """
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            elif os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = f.read()
            else:
                return {}
        except OSError:
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("workers registry root is not a JSON object")
            return {w["id"]: w for w in data.get("workers", []) if w.get("id")}
        except (ValueError, KeyError) as exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "workers registry %s is unparseable (%d bytes) — refusing to treat "
                "as empty; leaving the file intact for recovery (%s)",
                self._path, len(raw), exc,
            )
            raise

    def _write_unlocked(self, fh, workers: Dict[str, Dict[str, Any]]) -> None:
        """Overwrite the open, locked fh with the workers map."""
        payload = json.dumps({"workers": list(workers.values())}, indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Read-only snapshot of the registry, cached for a few seconds.

        Polls (list/get/pick) hit this; the cache keeps a hung/slow mount from
        blocking every request. Writes refresh the cache, so freshly-registered
        or reassigned workers are visible immediately to the writing process.
        """
        now = time.time()
        with self._lock:
            if self._cache is not None and (now - self._cache_at) < self._READ_TTL:
                return self._cache
            try:
                data = self._read_unlocked()
            except (ValueError, KeyError):
                # Corrupt on-disk file: don't crash polls — serve the last good
                # snapshot if we have one (the error is already logged).
                if self._cache is not None:
                    return self._cache
                raise
            self._cache = data
            self._cache_at = now
            return data

    @contextmanager
    def _transaction(self):
        """Yield the on-disk workers map under an exclusive cross-process lock.

        Reload -> mutate (caller) -> persist. The yielded dict is written back
        when the block exits without raising. Falls back to a plain in-process
        critical section when ``fcntl`` is unavailable.
        """
        with self._lock:
            self._ensure_parent()
            # Open r+ (create if missing) so we hold one fd for lock+read+write.
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            fh = os.fdopen(fd, "r+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                workers = self._read_unlocked(fh)
                yield workers
                self._write_unlocked(fh, workers)
                # Refresh the read-cache so this process sees its own write
                # immediately (and other processes within the TTL).
                self._cache = workers
                self._cache_at = time.time()
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

    # -- registration / lifecycle ------------------------------------------
    def register(
        self,
        *,
        name: str,
        url: str,
        gpus: Optional[List[Dict[str, Any]]] = None,
        role: str = "worker",
        models: Optional[List[str]] = None,
        worker_id: Optional[str] = None,
        pkg_version: Optional[str] = None,
        rpc_endpoint: Optional[str] = None,
        free_ram: Optional[int] = None,
        ram_total: Optional[int] = None,
        engine: Optional[Dict[str, Any]] = None,
        pool: Optional[str] = None,
        caps: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, Any]] = None,
        serving_limits: Optional[Dict[str, Any]] = None,
        slot_capable: Optional[bool] = None,
        slot_incapable_reason: Optional[str] = None,
        task_capabilities: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Add a worker (or re-register an existing one by id/url).

        Re-registration is keyed first on the supplied ``worker_id``, then on
        ``url`` — so an agent that restarts and advertises the same URL keeps
        its assignments instead of creating a duplicate row.
        """
        url = (url or "").rstrip("/")
        with self._transaction() as workers:
            existing = None
            if worker_id and worker_id in workers:
                existing = workers[worker_id]
            else:
                for w in workers.values():
                    if w.get("url") == url:
                        existing = w
                        break

            if existing is not None:
                # Grandfather pre-feature rows to approved; never silently revive a
                # blocked worker (the route refuses it, but don't let a re-register
                # flip it back to serving).
                existing.setdefault("admission", "approved")
                existing.update(
                    name=name or existing.get("name"),
                    url=url or existing.get("url"),
                    gpus=gpus if gpus is not None else existing.get("gpus", []),
                    role=role or existing.get("role", "worker"),
                    last_seen=_now(),
                )
                if models is not None:
                    existing["models"] = sorted(set(models))
                if pkg_version is not None:
                    existing["pkg_version"] = pkg_version
                if rpc_endpoint is not None:
                    existing["rpc_endpoint"] = rpc_endpoint
                if free_ram is not None:
                    existing["free_ram"] = free_ram
                if ram_total is not None:
                    existing["ram_total"] = ram_total
                if engine is not None:
                    existing["engine"] = engine
                if caps is not None:
                    existing["caps"] = caps
                if env is not None:
                    existing["env"] = env
                # Concurrency-hardening capability (2026-07-11). Stored verbatim;
                # _public_view spreads them onto /llm/workers rows. A None from an
                # older agent leaves the field untouched (legacy-safe).
                if serving_limits is not None:
                    existing["serving_limits"] = serving_limits
                if slot_capable is not None:
                    existing["slot_capable"] = slot_capable
                    existing["slot_incapable_reason"] = slot_incapable_reason
                # Per-task capability honesty (2026-07-11) — stored verbatim, same
                # legacy-safe idiom: a None from an older agent leaves any prior
                # value untouched. Central's workers_for_model gate reads it.
                if task_capabilities is not None:
                    existing["task_capabilities"] = task_capabilities
                # Only a NON-EMPTY declared pool re-asserts on re-register, so an
                # operator-set pool isn't wiped by a worker that doesn't declare
                # WORKER_POOL (which sends ""). Declaring workers still win.
                if pool and pool.strip():
                    existing["pool"] = pool.strip()
                # 4b organic backfill: every re-register refreshes the
                # assignment memory, so designations that predate the memory
                # feature become durable without an explicit assign.
                if existing.get("models"):
                    _remember_assignments(existing)
                return _public_view(existing)

            wid = worker_id or uuid.uuid4().hex
            # 4b: a fresh row for a KNOWN worker id (its old row was swept /
            # the registry was lost) restores the operator's designations from
            # the assignment memory — designations are worker-lifetime.
            remembered = _load_assign_memory().get(wid) if worker_id else None
            restored_models: List[str] = []
            restored_spill: Dict[str, Any] = {}
            if remembered:
                restored_models = list(remembered.get("models") or [])
                restored_spill = dict(remembered.get("spill_by_model") or {})
                if restored_models:
                    logger.warning(
                        "register: restoring %d remembered designation(s) for "
                        "returning worker %s (%s): %s", len(restored_models),
                        name or wid, wid, restored_models)
            worker = {
                "id": wid,
                "name": name or wid,
                "url": url,
                "role": role or "worker",
                "gpus": gpus or [],
                "models": sorted(set(models or []) | set(restored_models)),
                "spill_by_model": restored_spill,
                "pkg_version": pkg_version,
                "rpc_endpoint": rpc_endpoint,
                "free_ram": free_ram,
                "ram_total": ram_total,
                "engine": engine,
                "caps": caps,
                # Concurrency-hardening capability (2026-07-11): safe in-process
                # concurrency + whether the box can seat a native crash-isolated
                # slot. None on a pre-feature agent -> central assumes cap 1 and
                # shows no slot badge. See remote._advertised_cap / _public_view.
                "serving_limits": serving_limits,
                "slot_capable": slot_capable,
                "slot_incapable_reason": slot_incapable_reason,
                # Per-task capability honesty (2026-07-11): {task: bool} of the /ml
                # tasks this box can actually run (find_spec probe + a real whisper
                # import). None on a pre-feature agent -> central assumes capable so
                # a legacy fleet routes unchanged. See workers_for_model / _task_capable.
                "task_capabilities": task_capabilities,
                # Runtime-env capability: {"tier": "stable"|"edge"|..., versions}.
                # Read from the worker's own venv, so it's truth not config claim.
                "env": env,
                # Dedicated-pool label. "" = general pool. A pooled worker serves
                # ONLY requests tagged for its pool (reserved capacity); general
                # traffic never lands on it. See workers_for_model.
                "pool": (pool or "").strip(),
                # New workers land pending: they appear in the console but do not
                # serve traffic until an operator admits them (approval-required).
                "admission": "pending",
                "created_at": _now(),
                "last_seen": _now(),
            }
            workers[wid] = worker
            return _public_view(worker)

    def heartbeat(
        self,
        worker_id: str,
        *,
        gpus: Optional[List[Dict[str, Any]]] = None,
        loaded_models: Optional[List[str]] = None,
        loading: Optional[List[str]] = None,
        models_local: Optional[List[str]] = None,
        provisioning: Optional[List[str]] = None,
        provision_progress: Optional[Dict[str, Any]] = None,
        spill: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
        pkg_version: Optional[str] = None,
        role: Optional[str] = None,
        rpc_endpoint: Optional[str] = None,
        free_ram: Optional[int] = None,
        ram_total: Optional[int] = None,
        disk: Optional[Dict[str, Any]] = None,
        engine: Optional[Dict[str, Any]] = None,
        pool: Optional[str] = None,
        caps: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        comfy: Optional[Dict[str, Any]] = None,
        loaded_detail: Optional[Dict[str, Any]] = None,
        slots: Optional[List[Dict[str, Any]]] = None,
        allocations: Optional[List[Dict[str, Any]]] = None,
        pid_registry: Optional[Dict[str, Any]] = None,
        storage: Optional[Dict[str, Any]] = None,
        install: Optional[Dict[str, Any]] = None,
        serving_limits: Optional[Dict[str, Any]] = None,
        slot_capable: Optional[bool] = None,
        slot_incapable_reason: Optional[str] = None,
        task_capabilities: Optional[Dict[str, bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a worker alive and refresh its live GPU / loaded-model stats."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["last_seen"] = _now()
            if url:
                worker["url"] = url.rstrip("/")
            if gpus is not None:
                worker["gpus"] = gpus
            if loaded_models is not None:
                # TRUTHFUL residency: the agent's loaded_models only covers its
                # in-process dispatch cache — a model resident in a SLOT child
                # (llama_cpp.server / llama-server it spawned or adopted) is
                # invisible to it, so the console showed "Serving, nothing
                # loaded" while GBs sat on the GPU, and the warm reconcile
                # would re-probe already-warm models. Union in what the slots
                # report about themselves.
                merged = list(loaded_models)
                for s in (slots if slots is not None
                          else worker.get("slots") or []):
                    mk = (s or {}).get("model_key")
                    if mk and s.get("healthy") and mk not in merged:
                        merged.append(mk)
                worker["loaded_models"] = merged
            if loading is not None:
                worker["loading"] = loading   # weights load in flight ("heating")
            if models_local is not None:
                worker["models_local"] = models_local   # disk-truth (UTIL-08)
            if provisioning is not None:
                worker["provisioning"] = provisioning
            if provision_progress is not None:
                # PROGRESS CLOCK (orphan-job lesson: age on PROGRESS, not on any
                # write). The worker re-sends its whole progress map every
                # heartbeat, so the ARRIVAL of this field proves only that the
                # agent is alive — not that any pull is moving. A dead pull's
                # last snapshot keeps replaying verbatim (op sat frozen at
                # frac=0.0722 for 2h+). Carry a central ``progressed_at`` per
                # model, bumped ONLY when done_bytes actually ADVANCES, so
                # _live_provisioning can tell a live 7% from a corpse at 7%.
                prev = worker.get("provision_progress") or {}
                stamped: Dict[str, Any] = {}
                for mk, entry in (provision_progress or {}).items():
                    if not isinstance(entry, dict):
                        stamped[mk] = entry
                        continue
                    entry = dict(entry)
                    old = prev.get(mk) if isinstance(prev, dict) else None
                    old = old if isinstance(old, dict) else {}

                    def _done(e):
                        try:
                            return float(e.get("done_bytes") or 0)
                        except (TypeError, ValueError):
                            return 0.0

                    advanced = _done(entry) > _done(old)
                    carried = old.get("progressed_at")
                    if advanced or carried is None:
                        # First sighting counts as progress: a pull that just
                        # started has moved no bytes yet and must not be born
                        # already-stale.
                        entry["progressed_at"] = _now()
                    else:
                        entry["progressed_at"] = carried
                    stamped[mk] = entry
                worker["provision_progress"] = stamped
            if spill is not None:
                worker["spill"] = spill
            if pkg_version is not None:
                worker["pkg_version"] = pkg_version
            if role is not None:
                worker["role"] = role
            if rpc_endpoint is not None:
                worker["rpc_endpoint"] = rpc_endpoint
            if free_ram is not None:
                worker["free_ram"] = free_ram
            if ram_total is not None:
                worker["ram_total"] = ram_total
            if disk is not None:
                worker["disk"] = disk   # model-root volume free/total (preflight)
            if engine is not None:
                worker["engine"] = engine
            if env is not None:
                worker["env"] = env
            if config is not None:
                worker["config"] = config   # effective serving-config + source
            if comfy is not None:
                worker["comfy"] = comfy     # ComfyUI presence (slice A)
            if loaded_detail is not None:
                worker["loaded_detail"] = loaded_detail
            if slots is not None:
                worker["slots"] = slots
            if allocations is not None:
                # Unified engine-agnostic allocation view (slot-seated + in-RAM
                # residents). Stored verbatim; _public_view spreads it through.
                worker["allocations"] = allocations
            if pid_registry is not None:
                # Precision model->PID log (2026-07-14): per-model pid/host_mode/
                # vram + unattributed foreign squatters. Stored verbatim;
                # _public_view spreads it so the console renders it per worker.
                worker["pid_registry"] = pid_registry
            if storage is not None:
                # Worker-reported local-storage survey (per-model on-disk bytes +
                # protection flags + cache_used_bytes). Stored verbatim; the
                # over_budget flag + LRU eviction proposal are derived centrally
                # in _public_view via storage_proposal (which overlays the fields
                # the worker can't know: last_picked + the budget).
                worker["storage"] = storage
            if install is not None:
                # Install-shape (uniform-install drift detection): {unit,
                # via_systemd, venv, python, canonical}. Stored verbatim and
                # spread through _public_view (via **worker); the console badges
                # a non-canonical install off it.
                worker["install"] = install
            if caps is not None:
                worker["caps"] = caps
                # Worker-side config is the hard ceiling: if its caps tightened
                # below an operator limit, re-clamp the stored limit now.
                if worker.get("limits"):
                    worker["limits"] = _clamp_limits(worker["limits"], caps)
            # Concurrency-hardening capability (2026-07-11) — refreshed every beat
            # so the console/gate see live truth (a worker that installs the engine
            # binary flips slot_capable within one heartbeat). Legacy-safe: a None
            # from an older agent leaves the fields absent (central assumes cap 1).
            if serving_limits is not None:
                worker["serving_limits"] = serving_limits
            if slot_capable is not None:
                worker["slot_capable"] = slot_capable
                worker["slot_incapable_reason"] = slot_incapable_reason
            # Per-task capability honesty (2026-07-11) — refreshed every beat so an
            # /ops/pip that adds a missing dep flips the task True within one beat.
            # Legacy-safe: a None from an older agent leaves the field absent.
            if task_capabilities is not None:
                worker["task_capabilities"] = task_capabilities
            if pool and pool.strip():   # non-empty only — see register() note
                worker["pool"] = pool.strip()
            return _public_view(worker)

    def remove(self, worker_id: str) -> bool:
        with self._transaction() as workers:
            return workers.pop(worker_id, None) is not None

    # Operator-settable per-worker resource limits. Central may only TIGHTEN:
    # a worker's own configured caps (reported in its heartbeat as ``caps``)
    # are the hard ceiling, so every write is clamped against them.
    #
    # ``disk_cache_gib`` is the OPTIONAL explicit per-worker storage cap (GiB):
    # when set it drives the over-budget flag off cache_used vs the cap (WINS over
    # the free-disk reserve default in storage_proposal), and — unlike the others
    # — has no worker-reported cap, so _clamp_limits passes it through unclamped.
    # More robust than the free-disk reserve against non-model disk pressure.
    _LIMIT_KEYS = ("ram_max_gib", "gpu_mem_gib", "threads", "disk_cache_gib")

    def set_limits(self, worker_id: str,
                   limits: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Set (or clear, with None/{}) central's resource limits for a worker.

        Values are clamped to the worker's self-reported caps — the box's own
        config always wins. Unknown keys are dropped; non-numeric values raise.
        """
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            if not limits:
                worker.pop("limits", None)
                return _public_view(worker)
            clean: Dict[str, Any] = {}
            for k in self._LIMIT_KEYS:
                if k not in limits or limits[k] in (None, ""):
                    continue
                try:
                    clean[k] = float(limits[k]) if k != "threads" else int(limits[k])
                except (TypeError, ValueError):
                    raise ValueError(f"limit {k} must be numeric")
            worker["limits"] = _clamp_limits(clean, worker.get("caps") or {})
            return _public_view(worker)

    def set_pool(self, worker_id: str, pool: str) -> Optional[Dict[str, Any]]:
        """Operator override of a worker's dedicated pool ("" clears). Survives
        heartbeats from workers that don't declare WORKER_POOL (they send "",
        which the register/heartbeat guards ignore)."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["pool"] = (pool or "").strip()
            return _public_view(worker)

    _ADMISSION_STATES = ("pending", "approved", "blocked")

    def set_admission(self, worker_id: str, state: str) -> Optional[Dict[str, Any]]:
        """Set a worker's admission gate (pending/approved/blocked).

        ``approved`` lets it serve; ``pending`` parks it (visible, idle);
        ``blocked`` evicts it — the register/heartbeat routes refuse a blocked
        worker so its agent stops instead of respawning. Persisted, so the gate
        survives the worker's next heartbeat (unlike ``remove``, which a heartbeat
        would undo).
        """
        if state not in self._ADMISSION_STATES:
            raise ValueError(f"admission must be one of {self._ADMISSION_STATES}")
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["admission"] = state
            return _public_view(worker)

    # -- model assignment ---------------------------------------------------
    def assign_model(
        self,
        worker_id: str,
        model_key: str,
        spill: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assign a model to a worker, with optional per-assignment spill config.

        ``spill`` is an opaque dict of GPU/CPU knobs (e.g. n_gpu_layers,
        gpu_mem_gib, cpu_mem_gib) the worker applies when it loads the model.
        Omitted / None means "use the worker's autofit default."
        """
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            models = set(worker.get("models", []))
            models.add(model_key)
            worker["models"] = sorted(models)
            if spill is not None:
                by_model = worker.setdefault("spill_by_model", {})
                # An empty dict clears any override back to autofit.
                if spill:
                    by_model[model_key] = spill
                else:
                    by_model.pop(model_key, None)
            _remember_assignments(worker)   # 4b: designations survive row loss
            return _public_view(worker)

    def unassign_model(self, worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["models"] = sorted(set(worker.get("models", [])) - {model_key})
            worker.get("spill_by_model", {}).pop(model_key, None)
            # Hygiene: drop the per-model LRU stamp too, so the model_last_picked
            # map doesn't grow unbounded with unassigned models. Harmless for the
            # eviction proposal — a missing entry defaults to 0 (coldest), which
            # is correct for a now-unassigned leftover.
            worker.get("model_last_picked", {}).pop(model_key, None)
            _remember_assignments(worker)   # 4b: an explicit unassign IS forgotten
            return _public_view(worker)

    # -- placement grants (Phase 1 item 2) -----------------------------------
    # A GRANT is a SYSTEM-authored designation — born from a future
    # capacity-aware placement decision, NOT an operator assign/pin. Stored
    # separately from ``worker["models"]`` so it can never masquerade as
    # operator intent: assign/unassign, storage protection's "assigned" branch,
    # and the assignment-memory snapshot all stay blind to it. A grant is
    # freely LRU-evictable (see storage_proposal) and dies with the live
    # worker row — it is deliberately NOT written to the assign-memory file
    # (_remember_assignments), so a row-loss restore never resurrects it. This
    # method only touches ``worker["grants"]``; ``worker["models"]`` is
    # untouched (orthogonal to assign_model/unassign_model).
    def grant_model(self, worker_id: str, model_key: str,
                    job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            grants = worker.setdefault("grants", {})
            grants[model_key] = {
                "ts": _now(),
                "job_id": job_id,
                "origin": "system",
            }
            return _public_view(worker)

    def ungrant_model(self, worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
        """Remove one grant. Idempotent — a missing key is a no-op, not an error."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker.get("grants", {}).pop(model_key, None)
            return _public_view(worker)

    def spill_for(self, worker_id: str, model_key: str) -> Dict[str, Any]:
        """Per-assignment spill override for (worker, model), or {} for autofit."""
        worker = self._load().get(worker_id)
        if worker is None:
            return {}
        return dict(worker.get("spill_by_model", {}).get(model_key, {}))

    def set_load_report(self, worker_id: str, model_key: str,
                        report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Record the outcome of a warm/probe attempt for (worker, model).

        ``report`` is the worker's /probe response ({ok, fit, vram_used, error, …})
        plus a ``ts`` stamp, or a synthesized {ok: False, error} when the probe
        HTTP call itself failed. ``None`` clears the entry (e.g. on unassign).
        Stored under ``load_reports[model_key]`` on the worker record so the
        console can say WHY a model stayed cold instead of showing a silent
        no-op activate."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            reports = worker.setdefault("load_reports", {})
            if report is None:
                reports.pop(model_key, None)
            else:
                reports[model_key] = report
            return _public_view(worker)

    # -- queries ------------------------------------------------------------
    def get(self, worker_id: str) -> Optional[Dict[str, Any]]:
        worker = self._load().get(worker_id)
        return _public_view(worker) if worker else None

    def all(self) -> List[Dict[str, Any]]:
        return [_public_view(w) for w in self._load().values()]

    def storage_view(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """The derived storage view + LRU eviction proposal for one worker,
        computed from its RAW record (NOT the _public_view output, whose
        ``storage`` key is already the derived shape). This is the /reap-approve
        route's second-guard recompute: it must read the raw worker-reported
        ``storage`` survey to re-derive the CURRENT proposal at approve time.
        ``None`` if the worker is unknown."""
        worker = self._load().get(worker_id)
        return storage_proposal(worker) if worker else None

    def workers_for_model(self, model_key: str, *, online_only: bool = True,
                          pool: Optional[str] = None,
                          task: Optional[str] = None,
                          require_comfy_id_lock: bool = False) -> List[Dict[str, Any]]:
        wanted = _match_keys(model_key)
        want_pool = (pool or "").strip()
        need_tier = env_tier_for_model(model_key)
        tier_skipped = 0
        task_skipped = 0
        id_lock_skipped = 0
        engine_skipped = 0
        out = []
        for w in self.all():
            # Only admitted workers serve. Pending (awaiting operator approval) and
            # blocked workers are never picked for inference, even if assigned.
            if w.get("admission") != "approved":
                continue
            # Capability guard: skip a worker that reports no inference engine —
            # it would accept the dispatch and fail, wasting a hop before the
            # local fallback. (Workers not reporting engine status are kept.)
            if _engine_unusable(w):
                # Say-why parity with the tier/task/id_lock gates below: count a
                # worker only when it is otherwise ASSIGNED to this model, so an
                # empty result's log names the real cause (a DESIGNATED worker
                # whose engine can't serve — the "assigned+pinned but 500s"
                # mystery) instead of every engine-broken box on the fleet.
                _serveable = (list(w.get("models", [])) + list(w.get("loaded_models", []))
                              + list(w.get("grants", {}).keys()))
                if (model_key in _serveable
                        or wanted & {a for m in _serveable for a in _match_keys(m)}):
                    engine_skipped += 1
                continue
            # Dedicated-pool reservation: a request for pool P uses ONLY pool-P
            # workers; a general request (no pool) uses ONLY un-pooled workers.
            # So dedicated capacity is reserved for its app and never consumed by
            # general traffic — and a pool request that finds no pool worker
            # falls back to local (caller's None handling), not to the shared pool.
            if (w.get("pool") or "").strip() != want_pool:
                continue
            # Candidates = models this worker is ASSIGNED **or currently reports
            # LOADED**. A worker holding the model warm (loaded via probe, or
            # left resident after an unassign) is the best possible server —
            # ignoring it sent the request to a cold local fallback while a
            # GPU sat there with the weights already up. Loaded-ness is
            # heartbeat-fresh; if it evicts between beats the relay fails
            # pre-token and the caller falls back as always.
            # Grants (SYSTEM-authored placement, Phase 1 item 2) are serveable
            # exactly like an operator assignment or a live-loaded model —
            # once a granted model is actually held by the worker it must
            # route, or the grant is pointless. Grants confer NO eviction
            # protection (see storage_proposal) — this is purely "can serve",
            # not "may not be reclaimed".
            serveable = (list(w.get("models", [])) + list(w.get("loaded_models", []))
                         + list(w.get("grants", {}).keys()))
            # Match on the raw key OR any normalized alias (hub_id vs key vs
            # case), so an assignment made via one form still routes a chat that
            # names the model a slightly different way.
            if not (model_key in serveable or wanted & {a for m in serveable for a in _match_keys(m)}):
                continue
            if online_only and w["status"] != "online":
                continue
            # Runtime-env tier gate: the model runs ONLY on a worker whose venv
            # tier matches (strict both ways — an edge env can regress stable
            # models just as a stable env can't load edge architectures). Both
            # sides default to "stable", so an unmapped model on an unreporting
            # fleet routes exactly as before this gate existed.
            if _worker_env_tier(w) != need_tier:
                tier_skipped += 1
                continue
            # Per-task capability gate (2026-07-11): skip a worker that
            # AFFIRMATIVELY advertises it can't run this task (a canonical venv
            # missing an optional ML dep — sentence-transformers / whisper /
            # keybert). Legacy/unknown = capable, so a pre-feature fleet is
            # untouched; a None task never gates. Same say-why idiom as the tier
            # gate below.
            if not _task_capable(w, task):
                task_skipped += 1
                continue
            # ID-LOCK routing gate (identity-locked STILLs): an id_lock image
            # request must land on a box whose ComfyUI PROVABLY has the IPAdapter
            # nodes (comfy.id_lock). Affirmative-only — never route id_lock to a
            # comfy-less / nodeless worker where it would fail at request time (or
            # worse, tempt a silent non-locked fallback). Off (False) for every
            # other request, so ordinary routing is untouched.
            if require_comfy_id_lock and not _comfy_id_lock_capable(w):
                id_lock_skipped += 1
                continue
            out.append(w)
        if not out and engine_skipped:
            # The model HAS designated servers — every one was excluded because it
            # AFFIRMATIVELY reports its inference engine is unusable (llama-cpp not
            # loadable AND no native llama-server binary; engine.installed=False).
            # Name the cause so the operator repairs the box (`hugpy install-engine`
            # / reinstall llama-cpp-python) instead of seeing only the downstream
            # "no worker available / local serving disabled" 500 with no reason.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s: %d assigned worker(s) skipped — inference engine "
                "unusable (llama-cpp not loadable AND no native llama-server "
                "binary). Repair the engine on those boxes or assign the model to "
                "a healthy worker.", model_key, engine_skipped)
        if not out and tier_skipped:
            # The model HAS servers — they were excluded on env tier alone. Say
            # so, or the operator sees only the downstream "no worker / local
            # fallback disabled" error with no cause.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s requires env tier %r; %d otherwise-eligible worker(s) "
                "skipped (none advertise that tier)",
                model_key, need_tier, tier_skipped)
        if not out and task_skipped:
            # The model HAS servers — they were excluded on task capability alone
            # (they advertise they can't run this task). Name the reason, or the
            # operator sees only the downstream no-worker error with no cause.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s task %r: %d otherwise-eligible worker(s) skipped "
                "(task unavailable — missing optional ML dependency on those boxes)",
                model_key, task, task_skipped)
        if not out and id_lock_skipped:
            # The model HAS servers — every one was excluded because its ComfyUI
            # lacks the IPAdapter node pack (comfy.id_lock False/absent). Name the
            # cause so the operator installs it (WORKER-SETUP §5b) instead of
            # seeing only the downstream no-worker error.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "model %s id_lock: %d otherwise-eligible worker(s) skipped — no "
                "box advertises comfy.id_lock (install ComfyUI_IPAdapter_plus + "
                "weights per WORKER-SETUP §5b)", model_key, id_lock_skipped)
        return out

    def pick_for_model(self, model_key: str, pool: Optional[str] = None,
                       task: Optional[str] = None,
                       require_comfy_id_lock: bool = False) -> Optional[Dict[str, Any]]:
        """Choose an online worker to serve ``model_key`` (optionally within a
        dedicated ``pool``, and — when set — one that can run ``task``).

        ``require_comfy_id_lock`` (set for identity-locked STILL requests) further
        restricts to boxes whose ComfyUI advertises the IPAdapter nodes.

        Preference order:
            1. workers that already report the model as loaded (warm),
            2. otherwise the least-recently-picked online assignee.

        Returns ``None`` when no online worker (in the requested pool) is assigned
        to the model, which signals the caller to fall back to local execution.
        """
        candidates = self.workers_for_model(
            model_key, online_only=True, pool=pool, task=task,
            require_comfy_id_lock=require_comfy_id_lock)
        if not candidates:
            # Fall back to assigned workers even with a stale heartbeat. Heartbeat
            # (worker->central) can time out when central is briefly slow, while
            # offload (central->worker) still works — so an assigned worker that
            # looks "offline" is often still serviceable. The stream proxy fails
            # fast to local if the worker is genuinely unreachable.
            candidates = self.workers_for_model(
                model_key, online_only=False, pool=pool, task=task,
                require_comfy_id_lock=require_comfy_id_lock)
        if not candidates and (pool or "").strip():
            # PHANTOM-POOL RESCUE: a pool restriction only means something when the
            # pool exists. If NO registered worker carries this pool tag at all
            # (e.g. a client still sending the old default pool="ml" on a fleet
            # that never tagged one), honoring it would silently strand the request
            # on central-local even though a general worker serves the model. That
            # is the exact bug the un-pooled client default fixed — cover stale
            # clients here too. A pool with members but none available keeps the
            # reservation semantics: no crossover, local fallback.
            want_pool = pool.strip()
            pool_exists = any((w.get("pool") or "").strip() == want_pool
                              for w in self.all())
            if not pool_exists:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "pool %r has no registered workers; treating request "
                    "for %s as general (un-pooled)", want_pool, model_key)
                return self.pick_for_model(
                    model_key, pool=None, task=task,
                    require_comfy_id_lock=require_comfy_id_lock)
        if not candidates:
            return None

        # Version gate (soft): prefer workers running central's required package
        # version, so a chat doesn't land on a worker mid-rollout that's still on
        # old code. Soft — if NONE have converged yet, we still serve from the
        # (stale-but-working) assignees rather than forcing a local-only outage
        # during the ~heartbeat-long update window.
        required = required_pkg_version()
        if required:
            matched = [w for w in candidates if w.get("pkg_version") == required]
            if matched:
                candidates = matched

        # Efficiency-aware ranking (capability already filtered above). Prefer,
        # in order: a worker that already has the model warm (avoids a multi-GB
        # reload), then one with a usable GPU over CPU-only, then the
        # least-recently-picked (spreads load). Stable id tiebreak so the order
        # never wobbles. (Full need-vs-capacity placement is the allocator's job;
        # this is the lightweight default pick.)
        def _rank(w: Dict[str, Any]):
            warm = model_key in (w.get("loaded_models") or [])
            return (0 if warm else 1,
                    0 if _has_usable_gpu(w) else 1,
                    w.get("last_picked", 0),
                    w.get("id", ""))
        candidates.sort(key=_rank)
        chosen = candidates[0]

        # Persist the pick so round-robin survives across processes.
        with self._transaction() as workers:
            stored = workers.get(chosen["id"])
            if stored is not None:
                now = _now()
                stored["last_picked"] = now
                # Per-(worker,model) LRU signal for the storage eviction proposal.
                # ``last_picked`` above is a SINGLE per-WORKER round-robin scalar,
                # stamped on EVERY pick regardless of model — it spreads load, it
                # can't key an LRU-per-model eviction. This map records the last
                # time THIS model was routed to THIS worker (the authoritative
                # "central served (worker, model)" event), so storage_proposal can
                # sort candidates oldest-first; a model never served through
                # central has no entry -> defaults to 0 -> proposed first.
                stored.setdefault("model_last_picked", {})[model_key] = now
                chosen = stored
        return _public_view(chosen)

    def candidates_for_model(self, model_key: str,
                             pool: Optional[str] = None,
                             task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ranked ONLINE workers that can serve ``model_key`` — the cap-aware
        relay router's alternatives list (concurrency hardening 2026-07-11).

        Same eligibility + ranking as ``pick_for_model`` (warm, then GPU, then
        least-recently-picked), but WITHOUT the ``last_picked`` write: central's
        in-flight gate iterates this to reroute around a worker that is at its
        advertised in-process concurrency cap, and re-stamping every candidate on
        each probe would corrupt the round-robin. Online only — a reroute target
        must be live right now (the stale-heartbeat fallback pick_for_model does
        is for last-resort primary selection, not for spreading concurrent load).
        """
        candidates = self.workers_for_model(model_key, online_only=True, pool=pool, task=task)
        if not candidates:
            return []
        required = required_pkg_version()
        if required:
            matched = [w for w in candidates if w.get("pkg_version") == required]
            if matched:
                candidates = matched

        def _rank(w: Dict[str, Any]):
            warm = model_key in (w.get("loaded_models") or [])
            return (0 if warm else 1,
                    0 if _has_usable_gpu(w) else 1,
                    w.get("last_picked", 0),
                    w.get("id", ""))

        return sorted(candidates, key=_rank)


worker_store = WorkerStore()


# Module-level convenience wrappers (mirrors the manifest.py / peers.py style of
# exposing plain functions for routes to import).
def register_worker(**kwargs) -> Dict[str, Any]:
    return worker_store.register(**kwargs)


def heartbeat_worker(worker_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    # kwargs: gpus, loaded_models, spill — all optional, passed straight through.
    return worker_store.heartbeat(worker_id, **kwargs)


def remove_worker(worker_id: str) -> bool:
    return worker_store.remove(worker_id)


def set_worker_admission(worker_id: str, state: str) -> Optional[Dict[str, Any]]:
    return worker_store.set_admission(worker_id, state)


def set_worker_pool(worker_id: str, pool: str) -> Optional[Dict[str, Any]]:
    return worker_store.set_pool(worker_id, pool)


def set_worker_limits(worker_id: str, limits) -> Optional[Dict[str, Any]]:
    return worker_store.set_limits(worker_id, limits)


def enroll_required() -> bool:
    """Whether a valid enrollment token is mandatory to register/heartbeat.

    Default OFF (gradual rollout): tokenless workers may still register, but land
    ``pending`` like everyone else. Flip ``HUGPY_WORKER_ENROLL_REQUIRED`` truthy
    once the fleet is re-enrolled to refuse tokenless / revoked workers outright.
    """
    return os.environ.get("HUGPY_WORKER_ENROLL_REQUIRED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def assign_model(worker_id: str, model_key: str,
                 spill: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return worker_store.assign_model(worker_id, model_key, spill=spill)


def unassign_model(worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    worker_store.set_load_report(worker_id, model_key, None)
    return worker_store.unassign_model(worker_id, model_key)


def grant_model(worker_id: str, model_key: str,
                job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """SYSTEM-authored placement grant (Phase 1 item 2) — see WorkerStore.grant_model."""
    return worker_store.grant_model(worker_id, model_key, job_id=job_id)


def ungrant_model(worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    return worker_store.ungrant_model(worker_id, model_key)


def set_load_report(worker_id: str, model_key: str,
                    report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return worker_store.set_load_report(worker_id, model_key, report)


def spill_for(worker_id: str, model_key: str) -> Dict[str, Any]:
    return worker_store.spill_for(worker_id, model_key)


def list_workers() -> List[Dict[str, Any]]:
    return worker_store.all()


def get_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    return worker_store.get(worker_id)


def worker_storage_view(worker_id: str) -> Optional[Dict[str, Any]]:
    """Freshly-recomputed storage view + eviction proposal for a worker (from its
    RAW record). The /reap-approve route's central second guard. None if unknown."""
    return worker_store.storage_view(worker_id)


def pick_worker_for_model(model_key: str, pool: Optional[str] = None,
                          task: Optional[str] = None,
                          require_comfy_id_lock: bool = False) -> Optional[Dict[str, Any]]:
    return worker_store.pick_for_model(
        model_key, pool=pool, task=task,
        require_comfy_id_lock=require_comfy_id_lock)


def candidates_for_model(model_key: str, pool: Optional[str] = None,
                         task: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ranked online workers holding ``model_key`` — the relay gate's reroute
    list (see WorkerStore.candidates_for_model). No routing side effects."""
    return worker_store.candidates_for_model(model_key, pool=pool, task=task)


def explain_no_worker(model_key: str, pool: Optional[str] = None,
                      task: Optional[str] = None) -> str:
    """Human reason no worker took a request for ``model_key`` — the ``detail`` the
    refused-local error (HUGPY_NO_LOCAL_SERVING) surfaces so a DESIGNATED-but-idle
    model's failure is actionable instead of an opaque "no worker available".

    Walks the workers ASSIGNED to (or holding) the model and names why each was
    excluded from selection by a HARD static gate (admission, engine usability,
    dedicated-pool reservation, env tier, task capability) — the same gates
    ``workers_for_model`` applies. Returns "" when the model has no assigned worker
    at all (the caller's generic message already covers "assign it somewhere"),
    when every assigned worker actually passed the static gates (so the miss was
    transient — a stale beat or momentary cap, not a designation problem), or on
    ANY error: this is advisory and must never raise into a request.
    """
    try:
        wanted = _match_keys(model_key)
        want_pool = (pool or "").strip()
        need_tier = env_tier_for_model(model_key)
        reasons: List[str] = []
        for w in worker_store.all():
            serveable = list(w.get("models", [])) + list(w.get("loaded_models", []))
            if not (model_key in serveable
                    or wanted & {a for m in serveable for a in _match_keys(m)}):
                continue                          # not designated for this model
            name = w.get("name") or w.get("id") or "worker"
            if w.get("admission") != "approved":
                reasons.append(f"{name}: not approved (admission={w.get('admission')!r})")
                continue
            if _engine_unusable(w):
                eng = w.get("engine") or {}
                sr = w.get("slot_incapable_reason")
                err = str(eng.get("error") or "").strip()
                if w.get("slot_capable") is False and sr:
                    why = str(sr)
                elif err:
                    why = f"llama-cpp not loadable: {err}"
                else:
                    why = "inference engine reports installed=False"
                reasons.append(f"{name}: engine unusable ({why[:400]})")
                continue
            if (w.get("pool") or "").strip() != want_pool:
                reasons.append(f"{name}: reserved for pool {w.get('pool')!r} "
                               f"(request pool {want_pool!r})")
                continue
            if _worker_env_tier(w) != need_tier:
                reasons.append(f"{name}: env tier {_worker_env_tier(w)!r} != "
                               f"required {need_tier!r}")
                continue
            if not _task_capable(w, task):
                reasons.append(f"{name}: cannot run task {task!r} "
                               f"(missing optional dependency)")
                continue
            # Passed every HARD static gate — its miss was runtime/transient, not a
            # designation problem; don't manufacture a reason for it.
        if not reasons:
            return ""
        return (f"{model_key} is assigned but no worker could serve it — "
                + "; ".join(reasons[:4])
                + ". Repair the worker (e.g. `hugpy install-engine` / reinstall "
                  "llama-cpp-python) or assign the model to a healthy worker.")
    except Exception:  # noqa: BLE001 — advisory only; never raise into a request
        return ""


def fleet_snapshot() -> list:
    """The deterministic allocator's view of the fleet, from the live registry.

    Each worker → a Node with summed free VRAM (across its GPUs), free RAM,
    rpc_endpoint, and online flag. This snapshot + a task's Need is all the
    allocator looks at, so the same registry state yields the same placement.
    """
    from ......managers.resolvers.allocator import Node
    nodes = []
    for w in worker_store.all():
        gpus = w.get("gpus") or []
        free_vram = sum(int(g.get("memory_free") or 0) for g in gpus)
        nodes.append(Node(
            id=w["id"],
            free_vram=free_vram,
            free_ram=int(w.get("free_ram") or 0),
            rpc_endpoint=w.get("rpc_endpoint"),
            can_lead=(w.get("role") != "rpc"),   # rpc nodes are backends, not leads
            online=(w.get("status") == "online"),
            env_tier=_worker_env_tier(w),
        ))
    return nodes


def plan_placement(bytes_needed: int, *, cpu_ok: bool = False, headroom: float = 1.15,
                   env_tier: Optional[str] = None):
    """Deterministically place a task needing ``bytes_needed`` on the live fleet.

    Returns the allocator's Placement (whole / shard / cpu / none). For a 'shard'
    result, ``placement.rpc_servers`` + ``placement.tensor_split`` are what the
    lead is handed as a spill override. ``env_tier`` (when set) restricts the
    snapshot to workers serving that runtime-env tier — the allocator stays
    env-agnostic; we filter its input.
    """
    from ......managers.resolvers.allocator import Need, allocate
    nodes = fleet_snapshot()
    if env_tier:
        nodes = [n for n in nodes if n.env_tier == env_tier]
    return allocate(
        Need(bytes_needed=int(bytes_needed), cpu_ok=cpu_ok, headroom=headroom),
        nodes,
    )


def _shard_eligible() -> Dict[str, int]:
    """Models the operator allows to shard, with a VRAM byte estimate.

    Parsed from ``HUGPY_SHARD_MODELS`` = ``"key:bytes,key2:bytes"`` (bytes may use
    a ``g``/``gb`` suffix, e.g. ``BigModel:140gb``). A model NOT listed never
    shards — so this whole path is a no-op until the operator opts a model in.
    """
    out: Dict[str, int] = {}
    for part in os.environ.get("HUGPY_SHARD_MODELS", "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, raw = part.rpartition(":")
        raw = raw.strip().lower()
        mult = 2**30 if raw.endswith(("g", "gb")) else 1
        num = raw.rstrip("gb").strip()
        try:
            out[key.strip()] = int(float(num) * mult)
        except ValueError:
            continue
    return out


def placement_for_model(model_key: str) -> Optional[Dict[str, Any]]:
    """Allocator-driven shard placement for the remote.py seam.

    Returns ``{"worker": <lead dict>, "spill": {...}}`` only when ``model_key`` is
    shard-eligible AND the allocator decides it must shard across the pool; else
    ``None`` so the caller uses ordinary whole-model routing. The spill carries
    ``rpc_servers`` + a VRAM-proportional ``tensor_split`` + ``n_gpu_layers=-1``.
    """
    elig = _shard_eligible()
    need = elig.get(model_key) or elig.get(str(model_key).split("/")[-1])
    if not need:
        return None
    placement = plan_placement(need, cpu_ok=False,
                               env_tier=env_tier_for_model(model_key))
    if placement.kind != "shard":
        return None
    lead = get_worker(placement.lead_id)
    if not lead or not lead.get("url"):
        return None
    return {
        "worker": lead,
        "spill": {
            "rpc_servers": ",".join(placement.rpc_servers),
            "tensor_split": list(placement.tensor_split),
            "n_gpu_layers": -1,
        },
    }


# Register this pool's selector with the core router (web -> core — the correct
# dependency direction). resolve() consults it to offload a (model, task) to a
# live GPU worker, falling back to local. This module is imported at web-app
# startup; the standalone worker agent never imports it, so the core router
# simply runs everything local there (and delegated requests carry _force_local).
try:
    from ......managers.resolvers import (
        set_worker_provider as _set_worker_provider,
        set_placement_provider as _set_placement_provider,
    )
    _set_worker_provider(pick_worker_for_model, spill_for)
    # Allocator-driven sharding. No-op until a model is opted in via
    # HUGPY_SHARD_MODELS, so it never affects ordinary routing by default.
    _set_placement_provider(placement_for_model)
    # Cap-aware relay reroute (concurrency hardening 2026-07-11): the core gate
    # asks this for alternative online workers when the primary is at its
    # advertised in-process concurrency cap. Optional in older cores — guarded.
    try:
        from ......managers.resolvers.remote import set_worker_candidates_provider
        set_worker_candidates_provider(candidates_for_model)
    except Exception as _exc2:  # older core without the seam — gate degrades to primary-only
        import logging as _logging
        _logging.getLogger(__name__).info(
            "candidates provider not registered (older core): %s", _exc2)
    # No-worker diagnostic (2026-07-15): when selection yields no worker and this
    # box refuses local serving, the refused-local error names the DESIGNATED-but-
    # excluded worker(s) + reason (broken engine / no llama-server binary), turning
    # the opaque "assigned+pinned but 500s" mystery into an actionable message.
    # Optional in older cores — guarded; unset ⇒ the message is byte-identical.
    try:
        from ......managers.resolvers.remote import set_no_worker_diagnostic
        set_no_worker_diagnostic(explain_no_worker)
    except Exception as _exc3:  # older core without the seam — message unchanged
        import logging as _logging
        _logging.getLogger(__name__).info(
            "no-worker diagnostic not registered (older core): %s", _exc3)
except Exception as _exc:  # never let registration break importing the pool
    import logging as _logging
    _logging.getLogger(__name__).warning("worker provider registration failed: %s", _exc)
