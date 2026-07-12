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
        "status": "online" if _is_online(worker) else "offline",
        "admission": worker.get("admission", "approved"),
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


def storage_proposal(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a worker's local-STORAGE view + a guarded LRU eviction PROPOSAL.

    A PURE read-time computation over already-stored heartbeat fields — no
    daemon, no background loop, no persistent toggle, and it NEVER deletes
    anything. It is spread into every worker read by ``_public_view`` (the
    always-on storage monitoring depiction) and re-run by the ``/reap-approve``
    route as its central second guard, so the console preview and the approval
    share one source of truth.

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
    last_picked_map = worker.get("model_last_picked") or {}
    limits = worker.get("limits") or {}
    cfg = worker.get("config") or {}
    residency = cfg.get("residency") or {}
    pinned_cfg = cfg.get("pinned") or {}
    # Central slot-merged live truth — closes the reaper's in-process-only
    # loaded_model_keys() gap (it misses slot occupants / answering models).
    loaded_now = set(worker.get("loaded_models") or [])
    loading_now = set(worker.get("loading") or [])
    provisioning_now = set(worker.get("provisioning") or [])

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
        why = m.get("why") or ""
        protected = bool(m.get("protected"))
        if not protected:
            if m.get("pinned") or pinned_cfg.get(mk):
                protected, why = True, why or "pinned"
            elif str(residency.get(mk) or "").lower() == "static":
                protected, why = True, why or "static"
            elif m.get("assigned"):
                protected, why = True, why or "assigned"
            elif mk in loaded_now or m.get("loaded"):
                protected, why = True, why or "loaded"
            elif mk in loading_now or m.get("loading"):
                protected, why = True, why or "loading"
            elif mk in provisioning_now or m.get("provisioning"):
                protected, why = True, why or "provisioning"
        models_out.append({
            "model_key": mk,
            "bytes": b,
            "last_picked": lp or None,     # None = never served through central
            "protected": protected,
            "why": why,
            "pinned": bool(m.get("pinned") or pinned_cfg.get(mk)),
            "loaded": bool(m.get("loaded") or mk in loaded_now),
            "loading": bool(m.get("loading") or mk in loading_now),
            "provisioning": bool(m.get("provisioning") or mk in provisioning_now),
            "assigned": bool(m.get("assigned")),
        })
        if not protected:
            candidates.append((lp, b, mk))

    proposed: List[Dict[str, Any]] = []
    proposed_free = 0
    if over_budget and need_bytes > 0 and candidates:
        # LRU oldest-first (ascending last_picked); among equally-cold entries
        # (e.g. never-served leftovers, all last_picked=0) evict the LARGEST
        # first so the budget clears in the fewest deletes; stable key tiebreak.
        candidates.sort(key=lambda c: (c[0], -c[1], c[2]))
        for lp, b, mk in candidates:
            if proposed_free >= need_bytes:
                break
            proposed.append({"model_key": mk, "bytes": b,
                             "last_picked": lp or None})
            proposed_free += b

    return {
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
                worker["provision_progress"] = provision_progress
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
            serveable = list(w.get("models", [])) + list(w.get("loaded_models", []))
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
except Exception as _exc:  # never let registration break importing the pool
    import logging as _logging
    _logging.getLogger(__name__).warning("worker provider registration failed: %s", _exc)
