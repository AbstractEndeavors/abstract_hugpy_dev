"""HTTP surface for the GPU worker pool.

These endpoints serve two audiences:

  * the worker agent (machine-to-machine):
        POST /llm/workers/register
        POST /llm/workers/<id>/heartbeat
  * the console UI (human-driven):
        GET    /llm/workers
        GET    /llm/workers/<id>
        DELETE /llm/workers/<id>
        POST   /llm/workers/<id>/assign      {"model_key": ..., "spill": {...}?}
        POST   /llm/workers/<id>/unassign    {"model_key": ...}
  * model provisioning (worker pulls files from central over WireGuard):
        GET    /llm/models/<model_key>/manifest      file list + sizes + meta
        GET    /llm/models/<model_key>/file?path=..  stream one file (Range ok)

All registry state lives in functions.imports.utils.workers; this module only
translates HTTP <-> that store. get_bp, the worker_store helpers,
get_models_dict, get_model_config and route_destination are all re-exported
through functions/__init__ (imports → utils), mirroring how
llm_storage_routes.py pulls its registry helpers.
"""
import os
import re
import mimetypes

from pydantic import BaseModel, Field, ValidationError
from flask import request, jsonify, abort, send_file, Response

from .imports import *
from ....managers.serve.overrides import get_override, set_override, available_gguf_files, all_overrides, gguf_variants_detail
# Serving/slot drivers: imported at module scope (the route handlers below call
# these by bare name). They are not on the functions star re-export chain.
from ....managers.serve.serve import (
    install_serving, apply_plan, serving_overview, serve_spec_for, spec_row,
)
from ....managers.serve.slots import SlotPool, slots_enabled, slot_install_steps, _slot_count
# Self-update plumbing: the version central wants workers on, and the wheel dir
# it serves as a simple index. Imported directly (not via the functions star) so
# this doesn't depend on the re-export chain picking up the new names.
from ..functions.imports.utils.workers import (
    required_pkg_version, pkg_index_dir, set_worker_admission, set_worker_pool,
    set_worker_limits, enroll_required, worker_storage_view,
    set_worker_auto_reap, record_worker_auto_reap, forget_assignment_memory,
)
from ..functions.imports.utils.enrollment_tokens import (
    create_enrollment_token, verify_enrollment_token,
    revoke_enrollment_token, list_enrollment_tokens,
)
# Per-worker KEEP-WARM STAR (boot_prewarm) — the operator's keep-warm
# designation (operator RULINGS 2026-07-23): reconcile keeps it warm every beat,
# so a star evicted under pressure returns next cycle. "nothing warms until
# starred or static." DISTINCT from media_default (a UI/routing preference that
# loads nothing) and from 🔒static (which is eviction-protected). The identifier
# stays ``boot_prewarm`` but the meaning is keep-warm, not boot-once. Persisted
# server-side (worker_boot_prewarm.json) beside the media stores; surfaced on the
# worker record and carried to the worker on the heartbeat reply (omit-when-unset,
# so an older worker just ignores it).
# Per-worker WILDCARD flag — the "take all comers" ROUTING opt-in (operator
# doctrine 2026-07-23): designations are a hard routing scope; an undesignated
# model routes onto a worker ONLY if that worker opted in here. Routing
# eligibility ONLY (default False; never a warm source, never eviction
# protection). Persisted server-side (worker_wildcard.json) beside the star
# store; surfaced on the worker record the same response-copy way.
from ....imports.config.models.models_config import (
    worker_boot_prewarm_state, set_worker_boot_prewarm,
    worker_wildcard_state, set_worker_wildcard,
)

worker_bp, logger = get_bp("worker_bp", __name__)


# ── assigned = ready: background warm + heartbeat reconcile ─────────────────
# An operator's assignment is a READINESS contract, not a routing hint: a model
# designated to a worker is loaded there proactively — assign kicks a warm, and
# every heartbeat re-converges assigned-vs-loaded (covers worker reboots, agent
# restarts and evictions) — so the first real request never pays a multi-GB
# lazy load. Best-effort and rate-limited: a model that doesn't fit simply
# fails its probe on the worker (reported honestly there) and is retried only
# after the cooldown; probes run SEQUENTIALLY per worker so two multi-GB loads
# never race for the same VRAM.
import time as _time
import threading as _threading

_WARM_COOLDOWN_S = float(os.environ.get("HUGPY_WARM_COOLDOWN_S", "600"))
_warm_last: dict = {}          # (worker_id, model_key) -> monotonic ts
_warm_busy: set = set()        # worker_ids with a warm thread in flight
_warm_lock = _threading.Lock()


def _model_blocked(model_key: str) -> bool:
    """Operator BLOCK check — guarded (fail-open = not blocked). See
    comms.blocklist. A blocked model is never warmed/probed/provisioned."""
    try:
        from abstract_hugpy_dev.comms.blocklist import is_blocked
        return is_blocked(model_key)
    except Exception:  # noqa: BLE001 — never let the block gate break a sweep
        return False


def _blocked_keys() -> set:
    try:
        from abstract_hugpy_dev.comms.blocklist import blocked_keys
        return blocked_keys()
    except Exception:  # noqa: BLE001
        return set()


def _kick_warm(worker, model_keys, source: str) -> list:
    """Probe the given models on the worker in ONE background thread.

    Returns the models actually scheduled (cooldown/busy-filtered). Safe to
    call from any request: never blocks, never raises."""
    import httpx
    wid = (worker or {}).get("id") or ""
    base = ((worker or {}).get("url") or "").rstrip("/")
    if not wid or not base:
        return []
    # Operator BLOCK is the single choke on central-initiated PROVISIONING: a
    # warm probe DOWNLOADS an absent model onto the worker (ensure_model_present),
    # so filtering here stops every central pull path (reconcile-warm AND
    # assign-warm AND any future caller) for a blocked model — never a routing
    # candidate, never a transfer target.
    blocked = _blocked_keys()
    model_keys = [mk for mk in (model_keys or []) if mk not in blocked]
    if not model_keys:
        return []
    now = _time.monotonic()
    with _warm_lock:
        if wid in _warm_busy:
            return []
        due = [mk for mk in model_keys
               if now - _warm_last.get((wid, mk), 0.0) >= _WARM_COOLDOWN_S]
        if not due:
            return []
        for mk in due:
            _warm_last[(wid, mk)] = now
        _warm_busy.add(wid)
    logger.info("warming %s on worker %s (%s)", due, wid[:8], source)

    def _run():
        try:
            for mk in due:
                try:
                    httpx.post(base + "/probe/" + mk, timeout=900.0)
                except Exception:
                    pass   # best-effort; the next reconcile retries post-cooldown
        finally:
            with _warm_lock:
                _warm_busy.discard(wid)

    _threading.Thread(target=_run, name=f"warm-{source}-{wid[:8]}",
                      daemon=True).start()
    return due


# Reconcile's "deduce, don't count the world" prefilter: cooldown for the
# skip-log below, same cadence as _kick_warm's own cooldown so an un-fittable
# assigned model gets ONE note per window instead of per-beat spam.
_fit_skip_last: dict = {}      # (worker_id, model_key) -> monotonic ts of last skip-log


def _warmable_subset(worker, cold: list) -> list:
    """Filter reconcile's ``cold`` (assigned-but-not-loaded) model keys down to
    the subset actually worth paying a live load-probe for.

    ``_worker_fit`` (below, ~line 1235) already computes fit from numbers
    central has every heartbeat (effective GGUF bytes vs vram_free/free_ram) —
    no load required. Reconcile used to skip this and blanket-probe every cold
    assignment, which on a worker with dozens of assignments meant sequentially
    loading each one onto a card that physically holds a handful — a load/evict
    churn to answer a question already answerable from data on hand.

    Rules (see PLAN-reconcile-deduce-fit.md):
      * ``fit is False`` (can't fit VRAM+RAM combined) -> DROP. Probing it only
        confirms what we already know. Logged once per cooldown window, not
        every beat.
      * ``fit is None`` (can't be sized) -> KEEP unconditionally. This is the
        genuinely-ambiguous case where a real load is the only honest answer —
        the live-probe fallback stays in play for it.
      * otherwise -> co-residency cap: greedy-pack GPU-resident-first,
        smallest-``need``-first, until the running total would exceed the
        worker's current ``vram_free``. The rest stays assigned-but-cold; they
        still lazy-load correctly the moment real demand hits them.

    Pure function of (worker, cold) plus the module-level cooldown dict — no
    I/O, no network, safe to unit test directly.
    """
    wid = (worker or {}).get("id") or ""
    vram_free = (worker or {}).get("vram_free")
    unsizable = []
    fittable = []   # (need, gpu_resident, model_key)
    now = _time.monotonic()
    for mk in (cold or []):
        verdict = _worker_fit(mk, worker)
        if verdict.get("fit") is False:
            key = (wid, mk)
            if now - _fit_skip_last.get(key, 0.0) >= _WARM_COOLDOWN_S:
                _fit_skip_last[key] = now
                logger.info(
                    "reconcile: skipping load-probe for %s on worker %s "
                    "(assigned but won't fit) — %s",
                    mk, wid[:8] if wid else wid, verdict.get("reason") or "won't fit")
            continue
        if verdict.get("fit") is None:
            unsizable.append(mk)
            continue
        fittable.append((verdict.get("need") or 0, bool(verdict.get("gpu_resident")), mk))

    if vram_free is None:
        # No VRAM number to cap against (e.g. a worker that hasn't reported a
        # GPU yet) — the fit-based drop above already did the useful work;
        # don't invent a cap out of missing data.
        return unsizable + [mk for _, _, mk in fittable]

    fittable.sort(key=lambda t: (not t[1], t[0]))   # gpu_resident first, then smallest need
    capped = []
    used = 0
    for need, _resident, mk in fittable:
        if used + need <= vram_free:
            capped.append(mk)
            used += need
        # else: leave assigned-but-cold — the existing lazy-load-on-demand path
        # already handles first real request correctly.
    return unsizable + capped


# ── curated keep-warm set: 🔒static ONLY (the star does NOT reconcile-warm) ───
# Operator rulings 2026-07-23 (post-incident — SUPERSEDES both the 2026-07-15
# "immutable task-default floor" below AND the 0.1.201 star-in-warm-set design):
#   "the star is only supposed to indicate load that model on boot."
#   "it shouldn't effect anything but priority for ambiguous model calls."
#
# So central's reconcile keep-warm set is exactly ONE tier — the only one that
# PROMISES central-kept presence:
#   * 🔒 static — warm AND eviction-protected (unchanged). This is THE keep-warm
#     tier.
# The ⭐ star (boot_prewarm) is deliberately NOT here. The star's warm happens
# exactly ONCE, on the WORKER'S boot (agent._adopt_boot_prewarm, boot-once); it
# is NOT re-probed warm by this reconcile loop. The 0.1.201 build put the star in
# this set (RULING-2 "reconcile-kept-warm"); that re-warm fought active inference
# on ae today (star re-warm of coder-next → slot child stalled → zombie seat →
# agent freeze). Reverted: a starred model evicted under pressure STAYS cold
# until the worker restarts. Co-fit-gated re-entry is future work (Slice D).
#
# Everything else — the fleet TASK_DEFAULTS (sd-turbo et al.), 📌 pins, the ⭐
# star, and the whole ``models`` inventory — is NOT kept warm here and lazy-loads
# on first real request. "Nothing warms until starred (boot-load) or static":
# the task-defaults floor stays DEAD (do not resurrect); a task default is a
# ROUTING fallback (model_resolver.TASK_DEFAULTS still resolves "task named alone
# → default model"), never a warm designation. Pure central-side — no worker
# release.
#
# NOTE: ``_immutable_warm_defaults`` / ``_IMMUTABLE_WARM_DEFAULTS`` below are now
# UNUSED by the warm set (dropped from _reconcile_warm_set per RULING 1). Left in
# place — harmless and cheap — rather than churn; TASK_DEFAULTS itself is
# untouched and still drives routing (model_resolver.py) and the /prompt defaults
# feed (prompt_routes.py). If a future cleanup wants them gone, this pair (and
# only this pair) is safe to delete.
_IMMUTABLE_WARM_DEFAULTS: frozenset | None = None


def _immutable_warm_defaults() -> frozenset:
    """Fleet default model_keys (one per task), sourced from ``TASK_DEFAULTS``.

    ⚠ NO LONGER part of the keep-warm set (operator RULING 1, 2026-07-23: only
    the ⭐ star and 🔒static warm). Retained as a harmless helper; TASK_DEFAULTS
    stays the ROUTING fallback table (model_resolver.py). Guarded so a refactor
    of the constant's home can never raise on any caller.
    """
    global _IMMUTABLE_WARM_DEFAULTS
    if _IMMUTABLE_WARM_DEFAULTS is None:
        try:
            from abstract_hugpy_dev.imports.src.constants.categories import (
                TASK_DEFAULTS as _TASK_DEFAULTS)
            _IMMUTABLE_WARM_DEFAULTS = frozenset(
                str(v) for v in _TASK_DEFAULTS.values() if v)
        except Exception:  # noqa: BLE001 — never fail over a default lookup
            _IMMUTABLE_WARM_DEFAULTS = frozenset()
    return _IMMUTABLE_WARM_DEFAULTS


def _reconcile_warm_set(worker) -> list:
    """The curated set of a worker's on-disk models to keep warm on reconcile.

    = 🔒static ∩ on-disk − blocked (operator RULING 2026-07-23, post-incident).
    NOTHING warms here but 🔒static: the ⭐ star, the fleet TASK_DEFAULTS, 📌 pins,
    and the whole ``models`` inventory are all excluded and lazy-load on demand
    (the star additionally boot-loads ONCE on the worker side). The result still
    passes through ``_warmable_subset`` (fit-cap) before any real load, so this
    only ever *narrows* what reconcile touches.

      * 🔒 static — the one eviction-protected, central-kept-warm local-presence
        tier. THE keep-warm tier.

    ⚠ The ⭐ star (boot_prewarm) is deliberately NOT reconcile-warmed. It warms
    exactly once, on the worker's boot (agent._adopt_boot_prewarm, boot-once) —
    re-probing it warm here is what fought active inference on ae 2026-07-23
    (0.1.201's RULING-2 re-warm → coder-next slot child stalled → zombie seat →
    freeze). A starred model evicted under pressure STAYS cold until the worker
    restarts; co-fit-gated re-entry is future work (Slice D).

    ⚠ "on disk" = ``models_local`` (the worker's heartbeat disk-truth, UTIL-08),
    NEVER ``worker["models"]`` — that is the operator DESIGNATION set. Reading
    designations here made the ∩ a no-op, so every unloaded designation probed
    "cold" and the worker's /probe downloads absent models — central re-creating
    the eager-pull storm through the probe side door (2026-07-17: ae ground
    toward its full 1.2TB at ~5GB/min until this was fixed). A worker that
    reports no models_local warms NOTHING — a missed warm costs one first-call
    load; the designation fallback costs a terabyte.
    """
    present = set(worker.get("models_local") or [])
    if not present:
        return []
    cfg = worker.get("config") or {}
    # 🔒 STATIC — warm AND eviction-protected. The ONLY tier central keeps warm
    # (operator RULING 2026-07-23). The ⭐ star is NOT here (it boot-loads once,
    # worker-side, and is never reconcile-re-warmed); 📌 pins are routing
    # persistence only; the fleet TASK_DEFAULTS are a routing fallback
    # (model_resolver.TASK_DEFAULTS) — none of them warm.
    static = {k for k, v in (cfg.get("residency") or {}).items() if v == "static"}
    curated = static & present
    # Operator BLOCK outranks warm: a blocked model is never kept warm, even if
    # it is static. Intersect the curated set with not-blocked.
    curated -= _blocked_keys()
    return sorted(curated)


def _bearer_token() -> str | None:
    """Extract a Bearer token from the Authorization header (worker enrollment)."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def _enrollment_ok() -> bool:
    """Gate worker register/heartbeat on the enrollment token.

    Rules (so revocation always bites, but rollout stays gradual):
      * a VALID token  -> allow.
      * a token that is present but invalid/revoked -> deny ALWAYS (this is how
        revoke evicts a worker for good — even while enroll isn't yet required).
      * NO token at all -> allow only when HUGPY_WORKER_ENROLL_REQUIRED is off
        (gradual rollout); deny once it's on.
    """
    tok = _bearer_token()
    if tok is not None:
        return verify_enrollment_token(tok)
    return not enroll_required()


def _transfer_authorized() -> bool:
    """Credential gate for the worker file-transfer endpoints (manifest / file /
    chunksums). These stream multi-GB model weights to workers over WireGuard and
    were previously UNAUTHENTICATED on an internet-facing central — anyone who
    could reach the origin could exfiltrate every weight (and, before the
    seek-based Range fix below, peg central's CPU).

    A caller is authorized when EITHER:
      * it presents operator credentials (console session / ``HUGPY_OPERATOR_TOKEN``)
        — the same gate that guards the privileged console routes; OR
      * it presents a valid, un-revoked worker enrollment bearer token — the same
        machine-to-machine credential the agent already sends on register/heartbeat.

    Tokenless callers follow the SAME gradual-rollout rule as register/heartbeat
    (``_enrollment_ok`` → ``enroll_required``): allowed only while
    ``HUGPY_WORKER_ENROLL_REQUIRED`` is off. A present-but-invalid/revoked token
    is ALWAYS denied. No new auth scheme — this is ``operator_authenticated()``
    OR the existing enrollment gate.
    """
    try:
        from ..operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:
        # Operator gate unavailable for any reason -> fall back to the worker
        # enrollment gate rather than failing open.
        pass
    return _enrollment_ok()


class GpuInfo(BaseModel):
    index: int | None = None
    name: str | None = None
    memory_total: int | None = None
    memory_free: int | None = None


class RegisterRequest(BaseModel):
    name: str
    # Optional: the worker may advertise its own callback URL, but central will
    # override it with the request's real source IP when the worker can't tell
    # what address is actually reachable (loopback / 127.0.1.1 / NAT / bad NIC).
    url: str | None = Field(default=None, examples=["http://10.0.0.5:9100"])
    port: int | None = 9100
    gpus: list[GpuInfo] = Field(default_factory=list)
    role: str = "worker"
    models: list[str] | None = None
    worker_id: str | None = None
    # The dev package version this worker currently has installed (for the
    # self-update handshake). None from older agents that don't report it.
    pkg_version: str | None = None
    # Shard pool: "host:port" of this node's llama.cpp rpc-server (role=rpc), and
    # available RAM for the allocator's CPU tier.
    rpc_endpoint: str | None = None
    free_ram: int | None = None
    # RAW physical RAM (MemTotal) in bytes — the pool-budget denominator, unlike
    # reserve-adjusted free_ram. None from older agents that don't report it.
    ram_total: int | None = None
    # Inference-engine capability snapshot, e.g. {"installed": bool, "version":
    # str, "supports_gpu_offload": bool}. Lets central skip a worker that can't
    # actually serve (no engine) instead of routing a request that will fail.
    engine: dict | None = None
    # Dedicated-pool label (WORKER_POOL). "" = general. A pooled worker serves
    # ONLY requests tagged for its pool — reserved capacity for an external app.
    pool: str | None = None
    # The box's OWN configured resource ceilings (unit env: ram_max_gib,
    # gpu_mem_gib, threads, reserves). Central-set limits are clamped to these.
    caps: dict | None = None
    # Runtime-env capability snapshot: {"tier": "stable"|"edge"|..., "python":
    # ..., "transformers": ...}. The tier names which venv the unit runs
    # (WORKER_ENV_TIER); versions are read from the env itself. Models mapped to
    # a tier (HUGPY_MODEL_ENV_TIERS) route only to workers advertising it.
    env: dict | None = None
    # Concurrency-hardening capability (2026-07-11). serving_limits =
    # {"in_process_max_concurrency": N} — safe concurrent entrants into an
    # in-process runner (central gates relays to it; absent -> assume 1).
    # slot_capable = a native crash-isolated llama-server is resolvable;
    # slot_incapable_reason explains a False (e.g. no engine binary, in-process
    # fallback). None on older agents -> the field is simply absent on the row.
    serving_limits: dict | None = None
    slot_capable: bool | None = None
    slot_incapable_reason: str | None = None
    # Per-task capability honesty (2026-07-11): {task: bool} of the /ml tasks this
    # box can actually run (find_spec probe + a real whisper import). Central skips
    # a worker that says False for the request's task (workers_for_model). None on
    # older agents -> the field is absent on the row (assumed capable, no regression).
    task_capabilities: dict | None = None


# Hostnames/IPs a worker might self-report that are NOT reachable from central.
_UNREACHABLE_HOSTS = {"127.0.0.1", "127.0.1.1", "localhost", "0.0.0.0", "::1", ""}


def _is_dangerous_callback_host(host: str) -> bool:
    """Block self-advertised callback hosts that would turn central into an SSRF
    proxy: cloud-metadata / link-local (169.254.0.0/16 incl. 169.254.169.254,
    fe80::/10). LAN/private and routable (e.g. WireGuard) worker IPs are fine —
    workers are reached on their real address. A bare hostname is allowed.
    """
    h = (host or "").lower().strip("[]")
    if h in {"metadata.google.internal", "metadata"}:
        return True
    try:
        import ipaddress
        ip = ipaddress.ip_address(h)
        return ip.is_link_local or ip.is_unspecified
    except ValueError:
        return False


def _client_ip() -> str:
    """The worker's real source IP as seen by central.

    Honors X-Forwarded-For (left-most) when behind nginx/a proxy, else the raw
    socket peer. This is the address central can actually call back on.
    """
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or ""


def _host_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _resolve_worker_url(advertised: str | None, port: int | None) -> str:
    """Pick the callback URL central will store for a worker.

    If the worker advertised a usable host, trust it. Otherwise (no URL, or a
    loopback/bogus host) build one from the request's source IP + the port.
    """
    if advertised:
        host = _host_of(advertised)
        if host and host not in _UNREACHABLE_HOSTS and not _is_dangerous_callback_host(host):
            return advertised.rstrip("/")
    ip = _client_ip()
    p = port or 9100
    # IPv6 literal needs brackets.
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{p}"


class HeartbeatRequest(BaseModel):
    gpus: list[GpuInfo] | None = None
    loaded_models: list[str] | None = None
    # Models whose weights are LOADING on the worker right now — the console's
    # "heating" attribution (distinct from provisioning=pulling files and
    # loaded_models=resident/serving).
    loading: list[str] | None = None
    # UTIL-08 disk-truth: which ASSIGNED models actually have files on the
    # worker's disk — lets the console show "assigned but missing" drift.
    models_local: list[str] | None = None
    # Models the worker is currently pulling from central/HF in the background.
    provisioning: list[str] | None = None
    # Per-model download progress for the models in `provisioning`:
    # {model_key: {done_bytes, total_bytes, frac}}. Lets the console show a
    # percentage next to "pulling" instead of a bare spinner.
    provision_progress: dict | None = None
    spill: dict | None = None
    url: str | None = None
    port: int | None = None
    pkg_version: str | None = None
    role: str | None = None
    rpc_endpoint: str | None = None
    free_ram: int | None = None
    # RAW physical RAM (MemTotal) in bytes — see RegisterRequest.ram_total.
    ram_total: int | None = None
    # Honest budget-bar inputs (t13/t14). free_ram stays the CLAMPED wire-compat
    # figure; these carry the spec's raw inputs so central computes the honest
    # bar (bar_used / encroachment / over-limit) instead of the physical_total −
    # central_limit artifact. free_ram_raw = reserve-adjusted but UNCLAMPED by the
    # ceiling; ram_worker/ram_external = the worker's own process-tree RSS vs
    # everything else; vram_attributed/vram_unattributed = pid_registry model-row
    # sum vs foreign squatter sum, sampled the same beat as gpus. None on older
    # agents -> central degrades to legacy bar semantics (bar_semantics="legacy").
    free_ram_raw: int | None = None
    ram_worker_bytes: int | None = None
    ram_external_bytes: int | None = None
    vram_attributed_bytes: int | None = None
    vram_unattributed_bytes: int | None = None
    # Free/total bytes of the worker's model-root volume — the assign/load
    # preflight refuses pulls that won't fit (disk-aware allocation).
    disk: dict | None = None
    engine: dict | None = None
    pool: str | None = None
    caps: dict | None = None
    # Runtime-env capability snapshot — see RegisterRequest.env.
    env: dict | None = None
    # Effective operator serving-config (settings > env > default) — e.g.
    # {slot_count, slot_count_source}; set via POST /llm/workers/<id>/config.
    config: dict | None = None
    # ComfyUI presence on the worker (slice A): {"available", "url", "version"?}.
    comfy: dict | None = None
    # Per-loaded-model load facts: {key: {model_bytes, n_gpu_layers,
    # total_layers, gpu_pct}} — drives the console's serving rows.
    loaded_detail: dict | None = None
    # This worker's OWN slot pool statuses (agent-supervised llama_cpp.server
    # children) — rendered in its Compute-tab row like central's slots.
    slots: list | None = None
    # Unified, engine-agnostic resource-allocation view: one entry per
    # slot-seated model and one per in-RAM (in-process) resident model
    # ({kind: "slot"|"ram", model_key, ...}). A NEW field parallel to
    # slots/loaded_models — the console renders the worker card from it when
    # present, and falls back to slots+loaded_models for older agents.
    allocations: list | None = None
    # Precision model->PID log: {"models":[{model_key,pid,host_mode,vram_bytes,
    # alive}], "unattributed":[{pid,name,mib}]}. Stored verbatim + spread through
    # _public_view. Absent on older agents -> field simply absent.
    pid_registry: dict | None = None
    # Local-STORAGE survey (model-cache footprint on the model-root disk):
    # {cache_used_bytes, disk_free, models:[{model_key, bytes, pinned, loaded,
    # loading, provisioning, assigned, protected, why}]}. Central stores it
    # verbatim and overlays last_picked + the budget to derive the over_budget
    # flag + LRU eviction proposal in _public_view (storage_proposal). Absent on
    # pre-feature agents -> the proposal has no per-model inventory (monitoring
    # only). NB: model-root disk, DISTINCT from the SSD hot-cache.
    storage: dict | None = None
    # Install-shape (uniform-install drift detection): {unit, via_systemd, venv,
    # python, canonical}. Computed once on the worker. Absent on older agents ->
    # the console simply shows no install badge. Stored verbatim; the console
    # flags a non-canonical install from it.
    install: dict | None = None
    # Concurrency-hardening capability (2026-07-11) — see RegisterRequest.
    # serving_limits.in_process_max_concurrency gates central's relays;
    # slot_capable/slot_incapable_reason surface a box silently serving
    # in-process. None on older agents -> legacy-safe (cap 1, no badge).
    serving_limits: dict | None = None
    slot_capable: bool | None = None
    slot_incapable_reason: str | None = None
    # Per-task capability honesty (2026-07-11) — see RegisterRequest. Refreshed
    # every beat so an /ops/pip that adds a missing dep flips the task within one
    # heartbeat. None on older agents -> field absent (assumed capable).
    task_capabilities: dict | None = None
    # VRAM eviction churn (slice 10): {count, last:{victim,subject,host_mode,
    # vram_freed,at}, last_at} — the GPU evict-to-fit churn the operator watches,
    # surfaced beside the disk reaps. None on a pre-slice-10 worker.
    vram_evictions: dict | None = None
    # t28 load-and-learn: compact prediction-vs-measured observations the worker
    # emits on load-success (measured VRAM/RSS) and load-fail (refusal). Central
    # persists + aggregates them into per-model correction factors. Additive +
    # optional (extra='ignore' drops it for older workers); worker->central is
    # the safe direction. Each row: {model_key, engine, needs_weights_bytes,
    # needs_kv_bytes, ctx_pct, need_total_bytes, verdict, n_gpu_layers,
    # total_layers, vram_bytes, rss_bytes, load_seconds, device, ok, ts}.
    calibration_samples: list | None = None


class AssignRequest(BaseModel):
    model_key: str
    # Optional per-assignment GPU/CPU spill override. Empty/omitted = autofit.
    # Recognized keys: n_gpu_layers (int|"auto"|"off"), gpu_mem_gib (float),
    # cpu_mem_gib (float), tensor_split (list[float]).
    spill: dict | None = None


@worker_bp.route("/llm/workers", methods=["GET"])
def workers_list():
    """Worker registry (F3.2): running module version surfaced prominently —
    control-plane/worker version skew is silent behavior drift, so every row
    carries version_ok against central's required_pkg_version."""
    required = required_pkg_version()
    rows = list_workers()
    # Per-worker KEEP-WARM STAR: the model this worker keeps warm (reconcile-kept
    # every beat; evictable but returns next cycle; NOT static). Surfaced as
    # ``boot_prewarm: <model_key>|null`` so the console can render the star. Read
    # once for the whole list (never 5xxes the roster over it).
    try:
        _stars = worker_boot_prewarm_state()
    except Exception:  # noqa: BLE001 — the star map must never break /llm/workers
        _stars = {}
    # Per-worker WILDCARD flag: whether this worker opted in as "take all
    # comers" (routing eligibility only — see workers.py). Same response-copy
    # surfacing discipline as the star: read once for the whole list, stamped on
    # the row, never persisted onto the stored record, never 5xxes the roster.
    try:
        _wildcards = worker_wildcard_state()
    except Exception:  # noqa: BLE001 — the flag map must never break /llm/workers
        _wildcards = {}
    for w in rows:
        w["required_pkg_version"] = required
        w["version_ok"] = (required is None
                           or w.get("pkg_version") == required)
        w["boot_prewarm"] = _stars.get(w.get("id")) or None
        w["wildcard"] = bool(_wildcards.get(w.get("id")))
    # Call-time attribution (2026-07-14): stamp each worker's pid_registry
    # unattributed entries that are a RELAY-dispatched foreign GPU service
    # (identity-render) with the identity slug + job_id of the active
    # identity_mesh_build job central dispatched — so hy3dgen's ~5GB reads as the
    # identity it was called for, not an anonymous squatter. Best-effort: never
    # breaks the worker list (see pid_attribution.enrich_workers_pid_registry).
    try:
        from ..functions.imports.utils.pid_attribution import (
            enrich_workers_pid_registry)
        enrich_workers_pid_registry(rows)
    except Exception:  # noqa: BLE001 — attribution enrichment never 5xxes /llm/workers
        logger.debug("pid_registry relay attribution hook failed", exc_info=True)
    return jsonify(rows)


@worker_bp.route("/llm/workers/required-version", methods=["GET"])
def workers_required_version():
    """Public: the package version central wants workers to converge to.

    Unauthenticated by design — the bootstrap queries this BEFORE a worker
    exists (to pick which pip version to install) and it leaks nothing: the same
    value already rides every register/heartbeat reply. ``null`` when central
    pins no version (workers then track latest). Static path, so it takes routing
    priority over ``/llm/workers/<worker_id>``.
    """
    return jsonify({"required_pkg_version": required_pkg_version()})


@worker_bp.route("/llm/calibration", methods=["GET"])
def llm_calibration():
    """t28 load-and-learn: the per-model calibration table — for each model with
    observations, the sample counts, the median measured/predicted VRAM ratio,
    the spread, and the gate-passing clamped correction (null until enough sane
    samples). ``?model=<key>`` narrows to one model. Read-only JSON; degrades to
    an empty table when the store is unavailable (never 5xxes).

    ``enabled`` reflects the ``HUGPY_CALIBRATION`` master switch — when off the
    table still shows what WOULD be learned, but no correction is published to a
    worker reply or consulted by the preflight."""
    from abstract_hugpy_dev.comms import calibration as _calib
    model = (request.args.get("model") or "").strip()
    try:
        if model:
            agg = _calib.calibration_store.aggregate(model)
            rows = [agg] if agg else []
        else:
            rows = _calib.calibration_table()
    except Exception:  # noqa: BLE001 — introspection never 5xxes
        logger.debug("calibration table failed", exc_info=True)
        rows = []
    return jsonify({"enabled": _calib._enabled(),
                    "min_samples": _calib._min_samples(),
                    "max_spread": _calib._max_spread(),
                    "clamp": list(_calib._clamp_band()),
                    "models": rows})


@worker_bp.route("/llm/reservations", methods=["GET"])
def llm_reservations():
    """p6: the GPU-reservation listing — the heavy video runs that have PRE-CLAIMED
    a card (with the peak bytes reserved, the lease countdown, and what make-room
    yielded). Read-only; the operator's console SEES claims here but this slice
    exposes no create route (claims are minted only by the video dispatch path).

    Active claims by default; ``?all=1`` also lists recent released/expired rows
    (so the console can show what just finished / self-expired). ``?templates=1``
    adds the loaded per-task templates (measured overlay applied) for introspection.
    Never 5xxes — a store hiccup degrades to an empty list."""
    include_terminal = (request.args.get("all") or "").strip().lower() in (
        "1", "true", "yes", "on")
    out: dict = {"reservations": []}
    try:
        from abstract_hugpy_dev.video_intel.reservation.registry import (
            reservation_registry as _rr)
        out["reservations"] = _rr.listing(include_terminal=include_terminal)
    except Exception:  # noqa: BLE001 — introspection never 5xxes
        logger.debug("reservation listing failed", exc_info=True)
    if (request.args.get("templates") or "").strip().lower() in (
            "1", "true", "yes", "on"):
        try:
            from abstract_hugpy_dev.video_intel.reservation.templates import (
                reservable_tasks, load_template)
            out["templates"] = {t: load_template(t).as_dict()
                                 for t in reservable_tasks()}
        except Exception:  # noqa: BLE001
            logger.debug("reservation templates view failed", exc_info=True)
    return jsonify(out)


@worker_bp.route("/llm/workers/install.sh", methods=["GET"])
@worker_bp.route("/llm/workers/bootstrap.sh", methods=["GET"])
def workers_install_sh():
    """Public: serve the packaged worker bootstrap so a bare box enrolls with

        curl -fsSL https://dev.hugpy.ai/api/llm/workers/install.sh \
            | bash -s -- --name <box> --token <enroll-token>

    (--central defaults to THIS central via the sed below.) Unauthenticated by
    design, same rationale as required-version: it runs BEFORE a worker exists,
    and the script is public code straight out of the PyPI package — the enroll
    token, not the script, is the credential. The operator reached for exactly
    this URL on 2026-07-10 before it existed; now it does.
    """
    import re
    from importlib import resources
    script = (resources.files("abstract_hugpy_dev.worker_agent")
              .joinpath("bootstrap.sh").read_text(encoding="utf-8"))
    # Default --central to the central actually serving this script, so the
    # curl|bash one-liner needs only --name and --token.
    base = (request.host_url or "").rstrip("/")
    if base:
        script = re.sub(r'^CENTRAL="[^"]*"', f'CENTRAL="{base}"',
                        script, count=1, flags=re.M)
    return Response(script, mimetype="text/x-shellscript")


@worker_bp.route("/llm/queue", methods=["GET"])
def llm_queue():
    """Live in-flight chat queue (waiting/active) for the console activity view."""
    from abstract_hugpy_dev.managers.dispatch import activity
    return jsonify({"active": activity.snapshot(), "counts": activity.counts()})


@worker_bp.route("/llm/workers/register", methods=["POST"])
def workers_register():
    if not _enrollment_ok():
        # Bad/revoked token, or tokenless when enrollment is required. The agent
        # treats 401 as terminal and exits (no respawn).
        abort(401, description="Worker enrollment token invalid or required.")
    # A malformed/empty body is a client error (400), not a server fault (500):
    # RegisterRequest requires at least ``name`` and would otherwise raise an
    # unhandled ValidationError → 500. The enrollment 401 above still takes
    # precedence, so a tokenless caller (when enrollment is required) never
    # reaches body parsing. Valid bodies are unaffected.
    try:
        body = RegisterRequest(**(request.get_json(silent=True) or {}))
    except ValidationError:
        abort(400, description="Malformed worker registration body.")
    # Central decides the reachable callback URL from the request source IP when
    # the worker can't self-report a usable address.
    url = _resolve_worker_url(body.url, body.port)
    worker = register_worker(
        name=body.name,
        url=url,
        gpus=[g.model_dump() for g in body.gpus],
        role=body.role,
        models=body.models,
        worker_id=body.worker_id,
        pkg_version=body.pkg_version,
        rpc_endpoint=body.rpc_endpoint,
        free_ram=body.free_ram,
        ram_total=body.ram_total,
        engine=body.engine,
        pool=body.pool,
        caps=body.caps,
        env=body.env,
        serving_limits=body.serving_limits,
        slot_capable=body.slot_capable,
        slot_incapable_reason=body.slot_incapable_reason,
        task_capabilities=body.task_capabilities,
    )
    if worker.get("admission") == "blocked":
        # Operator evicted this worker; 403 tells the agent to stop, not respawn.
        abort(403, description="Worker is blocked by the operator.")
    # Tell the agent which package version to converge to (self-update handshake).
    worker["required_pkg_version"] = required_pkg_version()
    # Per-worker KEEP-WARM STAR (operator RULINGS 2026-07-23): carry this
    # worker's star from FIRST contact so the agent can warm it immediately
    # (thereafter the heartbeat keeps it warm every beat). Additive/omit-when-
    # unset (a released worker without the feature just ignores the extra key),
    # and this is the register reply (never persisted onto the stored record via
    # a mutation). Fully guarded — registration must never 5xx over the star store.
    try:
        _star = worker_boot_prewarm_state().get(worker.get("id"))
        if _star:
            worker["boot_prewarm"] = _star
    except Exception:  # noqa: BLE001 — star lookup must never break registration
        logger.debug("boot-prewarm register hook failed", exc_info=True)
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>", methods=["GET"])
def workers_get(worker_id):
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    # Surface the boot-prewarm star here too (mirrors workers_list).
    try:
        worker["boot_prewarm"] = worker_boot_prewarm_state().get(worker_id) or None
    except Exception:  # noqa: BLE001 — never 5xx a worker read over the star store
        worker["boot_prewarm"] = None
    # And the wildcard routing opt-in (mirrors workers_list; response-copy only).
    try:
        worker["wildcard"] = bool(worker_wildcard_state().get(worker_id))
    except Exception:  # noqa: BLE001 — never 5xx a worker read over the flag store
        worker["wildcard"] = False
    return jsonify(worker)


@worker_bp.route("/llm/workers/boot-prewarm", methods=["GET"])
def worker_boot_prewarm_list():
    """The full per-worker KEEP-WARM STAR map: {worker_id: model_key}.

    The ⭐ star = the operator's keep-warm designation (operator RULINGS
    2026-07-23) — reconcile keeps it warm every beat, so a star evicted under
    pressure returns next cycle. It is a NORMALLY EVICTABLE resident, NOT
    eviction-protected (that's 🔒static), and NOT the media_default UI
    preference. "nothing warms until starred or static." Read-only; unauthed by
    design (same tier as the /llm/workers roster it mirrors). (The path/field
    keep the ``boot-prewarm``/``boot_prewarm`` identifier — rename churn isn't
    worth it — but the meaning is keep-warm, not boot-once.)"""
    return jsonify(worker_boot_prewarm_state())


@worker_bp.route("/llm/workers/<worker_id>/boot-prewarm", methods=["POST"])
def set_worker_boot_prewarm_route(worker_id):
    """Set (or clear) this worker's ⭐ KEEP-WARM STAR — the ONE model this worker
    keeps warm.

    Body: {"model_key": "<key>", "enabled": bool}. enabled=True (default) makes
    ``model_key`` this worker's star, REPLACING any previous one. enabled=False
    clears it (only if ``model_key`` matches the current star, or if model_key is
    omitted/null — an unconditional clear). Single value per worker, persisted
    server-side (worker_boot_prewarm.json) so every client agrees.

    Operator-gated (operator_auth._SENSITIVE). The star = the operator's
    keep-warm designation (operator RULINGS 2026-07-23): reconcile keeps it warm
    EVERY beat, so a star evicted under pressure returns next cycle. It is NOT
    the media_default UI preference and NOT 🔒static — it does NOT mark the model
    static, does NOT protect it from eviction (evictable under pressure, but it
    returns next reconcile beat), and does NOT require the model to be
    present/allocated. "nothing warms until starred or static." For "start here
    AND stay here" with eviction protection, promote the model to 🔒static
    instead — that tier is what protects residency."""
    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key", body.get("model"))
    enabled = body.get("enabled", body.get("starred", True))
    return jsonify(set_worker_boot_prewarm(worker_id, model_key, enabled))


@worker_bp.route("/llm/workers/wildcard", methods=["GET"])
def worker_wildcard_list():
    """The full per-worker WILDCARD opt-in map: {worker_id: true}.

    A wildcard worker "takes all comers": it may catch UNDESIGNATED models and
    the overflow of designated models whose home workers can't serve, while its
    own designated models stay its priority (ranking sorts home matches first —
    see workers.py). Routing eligibility ONLY — never a warm source, never
    eviction protection, never a bypass of block/admission/pool/engine/task
    gates. An ABSENT worker id reads False (default-false promise: with no
    flags set the fleet routes exactly as before this feature existed).
    Read-only; unauthed by design (same tier as the /llm/workers roster and the
    boot-prewarm map it mirrors)."""
    return jsonify(worker_wildcard_state())


@worker_bp.route("/llm/workers/<worker_id>/wildcard", methods=["POST"])
def set_worker_wildcard_route(worker_id):
    """Set (or clear) this worker's WILDCARD ("take all comers") routing opt-in.

    Body: {"enabled": true|false} (default true). enabled=true opts the worker
    in: undesignated models may route here, and designated models overflow here
    when their home workers are all refused/at-cap — the worker's OWN designated
    models keep priority (home ranks above wildcard-catch; overflow is pure
    ordering, no separate machinery). enabled=false restores the default sealed
    scope: only its own designated / resident / granted models route here.

    Operator-gated (operator_auth._SENSITIVE — same routing-registry-write tier
    as assign/boot-prewarm). ROUTING ONLY (operator doctrine 2026-07-23): once
    resident, normal eviction rules apply — "you don't want random evictions
    simply because you have a verbose model registry" is exactly why all-comers
    is an explicit per-worker opt-in, default False. Persisted server-side
    (worker_wildcard.json) so every client agrees; stamped on worker payloads as
    ``wildcard: bool``, never persisted onto the stored worker record."""
    body = request.get_json(silent=True) or {}
    enabled = body.get("enabled", body.get("wildcard", True))
    return jsonify(set_worker_wildcard(worker_id, enabled))


@worker_bp.route("/llm/workers/<worker_id>/health", methods=["GET"])
def workers_health(worker_id):
    """Probe the worker's own HTTP server (not just its heartbeat).

    Heartbeat liveness tells you the agent process is alive and can REACH
    central. This instead has central call the worker's /health, which confirms
    central -> worker connectivity (the direction chat offload actually uses)
    and returns the worker's live GPU/loaded-model/spill snapshot.
    """
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")

    url = (worker.get("url") or "").rstrip("/") + "/health"
    try:
        import httpx

        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        return jsonify({"reachable": True, "url": url, "health": resp.json()})
    except Exception as exc:
        return jsonify({"reachable": False, "url": url, "error": f"{type(exc).__name__}: {exc}"})


@worker_bp.route("/llm/workers/<worker_id>/heartbeat", methods=["POST"])
def workers_heartbeat(worker_id):
    if not _enrollment_ok():
        abort(401, description="Worker enrollment token invalid or required.")
    body = HeartbeatRequest(**(request.get_json(silent=True) or {}))
    # Keep the callback URL correct as the network sees it — fixes workers that
    # first registered (in an older agent) with a loopback/bogus address.
    url = _resolve_worker_url(body.url, body.port)
    worker = heartbeat_worker(
        worker_id,
        gpus=[g.model_dump() for g in body.gpus] if body.gpus is not None else None,
        loaded_models=body.loaded_models,
        loading=body.loading,
        models_local=body.models_local,
        provisioning=body.provisioning,
        provision_progress=body.provision_progress,
        spill=body.spill,
        url=url,
        pkg_version=body.pkg_version,
        role=body.role,
        rpc_endpoint=body.rpc_endpoint,
        free_ram=body.free_ram,
        ram_total=body.ram_total,
        free_ram_raw=body.free_ram_raw,
        ram_worker_bytes=body.ram_worker_bytes,
        ram_external_bytes=body.ram_external_bytes,
        vram_attributed_bytes=body.vram_attributed_bytes,
        vram_unattributed_bytes=body.vram_unattributed_bytes,
        disk=body.disk,
        engine=body.engine,
        pool=body.pool,
        caps=body.caps,
        env=body.env,
        config=body.config,
        comfy=body.comfy,
        loaded_detail=body.loaded_detail,
        slots=body.slots,
        allocations=body.allocations,
        pid_registry=body.pid_registry,
        storage=body.storage,
        install=body.install,
        serving_limits=body.serving_limits,
        slot_capable=body.slot_capable,
        slot_incapable_reason=body.slot_incapable_reason,
        task_capabilities=body.task_capabilities,
        vram_evictions=body.vram_evictions,
    )
    if worker is None:
        # The agent thinks it's registered but central forgot it (restart,
        # cleared registry). 410 tells the agent to re-register.
        abort(410, description="Unknown worker id; please re-register.")
    if worker.get("admission") == "blocked":
        # Persistent eviction: 403 stops the agent instead of letting it limp on.
        abort(403, description="Worker is blocked by the operator.")
    # Designated = ready: re-converge a CURATED keep-warm set on every beat —
    # NOT the box's whole ``models`` inventory. The curated set is the immutable
    # task-defaults present on disk plus the operator's pins (_reconcile_warm_set);
    # everything else on disk lazy-loads on first real request. A cold member of
    # that set (worker rebooted, agent restarted, weights evicted) gets a
    # background warm — rate-limited by _WARM_COOLDOWN_S so an un-fittable model
    # doesn't probe-spin. It then passes _warmable_subset to deduce fit from
    # numbers already on hand before paying a live load-probe: never probe a
    # model that can't fit, and never warm more than can co-reside at once.
    try:
        loaded = set(worker.get("loaded_models") or [])
        cold = [mk for mk in _reconcile_warm_set(worker) if mk not in loaded]
        if cold:
            warm_now = _warmable_subset(worker, cold)
            if warm_now:
                _kick_warm(worker, warm_now, "reconcile")
    except Exception:
        pass  # readiness convergence must never fail a heartbeat
    # AUTO-REAP (slice 8, Part B): event-driven — this beat is the trigger, no
    # timer/daemon. Fires the guarded reap-approve flow ONLY when the worker
    # opted in AND is over budget with a proposal AND the cooldown elapsed. Its
    # own try/except means it can never fail a heartbeat.
    _maybe_auto_reap(worker_id, worker)
    # PENDING-ORPHAN EXPIRY (slice 9, defect 2): also event-driven off the beat
    # (no timer thread). Retires never-dispatched pending jobs so a stuck row
    # doesn't wait for someone to open /llm/jobs. Best-effort, never fails a beat.
    try:
        from abstract_hugpy_dev.comms import job_store
        job_store.expire_pending_orphans()
    except Exception:
        pass
    # Advertise the target version every beat, so a worker converges within one
    # heartbeat of central's required version changing.
    worker["required_pkg_version"] = required_pkg_version()
    # t28 load-and-learn: persist any calibration observations the worker shipped
    # this beat, then publish the gate-passing per-model corrections back in the
    # reply (a plain dict the worker reads with .get() — additive, an older
    # worker just ignores it). Fully guarded: the learned loop must NEVER fail a
    # heartbeat (a missed beat drops the worker off the fleet). Sent on a COPY so
    # the ephemeral corrections never persist onto the stored worker record (they
    # must not ride the relay wire built from it).
    reply_extra: dict = {}
    try:
        from abstract_hugpy_dev.comms import calibration as _calib
        if body.calibration_samples:
            _calib.record_samples(worker_id, body.calibration_samples)
        relevant = sorted(set(worker.get("loaded_models") or [])
                          | set(worker.get("models") or []))
        corr = _calib.corrections_for(relevant or None)
        if corr:
            reply_extra["calibration"] = corr
    except Exception:  # noqa: BLE001 — calibration is best-effort; never 5xx a beat
        logger.debug("calibration heartbeat hook failed", exc_info=True)
    # p6: publish this worker's ACTIVE GPU reservations back on the reply (additive,
    # omit-when-unset — same wire idiom as calibration: a plain list the worker reads
    # with .get(), an older worker just ignores it). The WORKER-side hard admission
    # gate (release-bound; see report) consumes this to hold the reserved bytes out
    # of its OWN evict-to-fit headroom, so the LLM plane never reclaims VRAM a heavy
    # video render is about to occupy. Fully guarded — never fails a beat.
    try:
        from abstract_hugpy_dev.video_intel.reservation.registry import (
            reservation_registry as _rr)
        resv = _rr.active(worker_id)
        if resv:
            reply_extra["reservations"] = [
                {"run_id": r.get("run_id"), "task": r.get("task"),
                 "gpu": r.get("gpu"), "peak_bytes": r.get("peak_bytes")}
                for r in resv]
    except Exception:  # noqa: BLE001 — reservation hook is best-effort; never 5xx a beat
        logger.debug("reservation heartbeat hook failed", exc_info=True)
    # k2: publish the operator's model BLOCK set on the reply (additive,
    # omit-when-empty — same wire idiom as calibration/reservations: a plain
    # list the worker reads with .get(), an older worker just ignores it). The
    # block primitive (aa4aea3) already covers every CENTRAL path — routing,
    # assign-409, warm sweeps, provisioning kick, the agent-brain ladder — but
    # the worker's OWN background reconciler loops (slot fill, etc.) have no
    # other way to learn a model was blocked out from under an assignment that
    # is still on record (block deliberately does not auto-unassign). This
    # closes that gap. Fully guarded — never fails a beat.
    try:
        blocked = sorted(_blocked_keys())
        if blocked:
            reply_extra["blocked_models"] = blocked
    except Exception:  # noqa: BLE001 — block propagation is best-effort; never 5xx a beat
        logger.debug("blocklist heartbeat hook failed", exc_info=True)
    # Per-worker KEEP-WARM STAR ("star", operator RULINGS 2026-07-23): publish
    # THIS worker's star on the reply (additive, OMIT-WHEN-UNSET — same wire idiom
    # as calibration/reservations/blocked_models: a plain scalar the worker reads
    # with .get(), an older released worker just ignores it, so the extra=forbid
    # relay schema is never broken). The worker KEEPS IT WARM: on every beat it
    # loads the star if not currently resident (a NORMALLY EVICTABLE on-demand
    # resident, NOT static, NOT the media_default UI preference), so an eviction
    # under pressure is repaired next beat. Only sent when a star is set for this
    # worker. Fully guarded — never fails a beat.
    try:
        star = worker_boot_prewarm_state().get(worker_id)
        if star:
            reply_extra["boot_prewarm"] = star
    except Exception:  # noqa: BLE001 — star propagation is best-effort; never 5xx a beat
        logger.debug("boot-prewarm heartbeat hook failed", exc_info=True)
    # Sent on a COPY so the ephemeral corrections/reservations/blocked set never
    # persist onto the stored worker record (they must not ride the relay wire
    # built from it).
    if reply_extra:
        reply = dict(worker)
        reply.update(reply_extra)
        return jsonify(reply)
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>", methods=["DELETE"])
def workers_remove(worker_id):
    if not remove_worker(worker_id):
        abort(404, description="Unknown worker id.")
    return jsonify({"removed": True, "id": worker_id})


@worker_bp.route("/llm/workers/<worker_id>/memory", methods=["DELETE"])
def workers_forget_memory(worker_id):
    """k10 hardening: sanctioned removal of a GHOST assignment-memory entry
    (worker_assignments.json) — an id with no live row in workers.json at all.

    The by-design durability (a live row's designations survive row loss —
    see the module docstring in workers.py) is untouched: this 409s if
    ``worker_id`` is still live, and 404s if it was never in memory. Only a
    truly stray id (already absent from the live registry) can be forgotten.
    """
    try:
        result = forget_assignment_memory(worker_id)
    except ValueError as exc:
        abort(409, description=str(exc))
    if result == "unknown":
        abort(404, description="Unknown worker id in assignment memory.")
    return jsonify({"forgot": worker_id})


# -- admission gate (the console "switch") ---------------------------------
# Unlike DELETE (which a heartbeat undoes), these set a PERSISTENT admission
# state so the decision sticks across the worker's next contact.
def _set_admission_or_404(worker_id, state):
    worker = set_worker_admission(worker_id, state)
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>/admit", methods=["POST"])
def workers_admit(worker_id):
    """Approve a pending/blocked worker so it may serve traffic."""
    return _set_admission_or_404(worker_id, "approved")


@worker_bp.route("/llm/workers/<worker_id>/block", methods=["POST"])
def workers_block(worker_id):
    """Block a worker: it stops serving and its agent exits on next contact."""
    return _set_admission_or_404(worker_id, "blocked")


@worker_bp.route("/llm/workers/<worker_id>/pool", methods=["POST"])
def workers_set_pool(worker_id):
    """Operator override of a worker's dedicated pool ({"pool": "..."}; "" clears).
    A worker that declares WORKER_POOL still re-asserts its own on next heartbeat."""
    body = request.get_json(silent=True) or {}
    worker = set_worker_pool(worker_id, body.get("pool", ""))
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>/limits", methods=["POST"])
def workers_set_limits(worker_id):
    """Operator resource limits for a worker ({"ram_max_gib", "gpu_mem_gib",
    "threads"}; {} clears). Clamped server-side to the worker's self-reported
    caps — the box's own config is the hard ceiling, central may only tighten.
    The worker adopts the effective limits on its next heartbeat."""
    body = request.get_json(silent=True) or {}
    try:
        worker = set_worker_limits(worker_id, body.get("limits", body))
    except ValueError as exc:
        abort(400, description=str(exc))
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>/admission", methods=["POST"])
def workers_admission(worker_id):
    """Set admission to an explicit state (pending|approved|blocked)."""
    body = request.get_json(silent=True) or {}
    state = body.get("state")
    try:
        return _set_admission_or_404(worker_id, state)
    except ValueError as exc:
        abort(400, description=str(exc))


# -- enrollment tokens (the single funnel to admit a machine at all) -------
@worker_bp.route("/llm/enroll-tokens", methods=["GET"])
def enroll_tokens_list():
    return jsonify(list_enrollment_tokens())


@worker_bp.route("/llm/enroll-tokens", methods=["POST"])
def enroll_tokens_create():
    """Mint an enrollment token. The plaintext ``token`` is returned ONCE here."""
    body = request.get_json(silent=True) or {}
    return jsonify(create_enrollment_token(label=str(body.get("label") or "")))


@worker_bp.route("/llm/enroll-tokens/<token_id>", methods=["DELETE"])
def enroll_tokens_revoke(token_id):
    """Revoke a token — its workers are refused (401) and their agents stop."""
    if not revoke_enrollment_token(token_id):
        abort(404, description="Unknown token id.")
    return jsonify({"revoked": True, "id": token_id})


def _central_missing_reason(model_key: str) -> str | None:
    """Daylight item 4 invariant: workers pull FROM CENTRAL. Return why central
    can't provide this model's files, or None when it can (present on disk)."""
    try:
        from ....imports import route_destination
        from ....imports.config.main import get_model_config, model_looks_downloaded
        cfg = get_model_config(model_key)
        # ComfyUI rows go through the SAME guard now: central symlinks its
        # /checkpoints store into the manifest layout, so holding the file is
        # a real requirement (workers pull + symlink it into their ComfyUI).
        path = route_destination(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(
            hub_id=getattr(cfg, "hub_id", None), framework=getattr(cfg, "framework", None),
            tasks=getattr(cfg, "tasks", None), primary_task=getattr(cfg, "primary_task", None),
            filename=getattr(cfg, "filename", None), include=getattr(cfg, "include", None),
            name=getattr(cfg, "name", None), folder=getattr(cfg, "folder", None)))
        if not os.path.isdir(path):
            return "no model directory"
        if not model_looks_downloaded(path, cfg):
            return "directory present but files incomplete"
        return None
    except Exception as exc:  # noqa: BLE001 — resolution failure = can't provide
        return f"{type(exc).__name__}: {exc}"


def _disk_preflight_reason(worker: dict, model_key: str) -> str | None:
    """Disk-aware allocation: a designation triggers a pull onto the worker's
    model-root volume — refuse EARLY (clear 409) when it won't fit, instead of
    failing mid-pull with a full disk. None = fits / already local / unknown
    (older agents don't report disk — never block on absent telemetry)."""
    if model_key in (worker.get("models_local") or []):
        return None                          # files already there — no pull
    free = ((worker.get("disk") or {}).get("free_bytes"))
    if not free:
        return None                          # no disk telemetry — don't block
    try:
        from ....managers.dispatch.dispatch import _dir_size_detail
        from ....imports import route_destination
        from ....imports.config.main import get_model_config
        cfg = get_model_config(model_key)
        need = (_dir_size_detail(route_destination(
            cfg.to_dict() if hasattr(cfg, "to_dict") else {
                "hub_id": getattr(cfg, "hub_id", None),
                "framework": getattr(cfg, "framework", None),
                "primary_task": getattr(cfg, "primary_task", None),
                "name": getattr(cfg, "name", None),
                "folder": getattr(cfg, "folder", None),
                "filename": getattr(cfg, "filename", None)})) or {}).get("model_bytes")
    except Exception:  # noqa: BLE001 — unsizable: don't block
        return None
    if need and free < need * 1.1:           # 10% headroom for temp/partial files
        return (f"needs ~{need/1e9:.1f}GB but the worker's model volume has "
                f"only {free/1e9:.1f}GB free")
    return None


@worker_bp.route("/llm/central-provisioning", methods=["GET"])
def central_provisioning():
    """Per-model central-disk readiness — the SAME authoritative check the worker
    load/assign guard uses (``_central_missing_reason``), exposed so the picker can
    show each model's provisioning state BEFORE an allocation is attempted, and
    offer a download for the ones central doesn't hold complete.

    Why not the manifest ``status`` field: it is NOT trustworthy for this — a dir
    holding only an ``mmproj-*.gguf`` (no main quant) is still marked 'installed',
    and archived rows can read 'partial'. This endpoint reflects what the loader
    actually enforces, so the UI and the guard never disagree.

    Returns ``{ model_key: {state, reason} }`` where state is one of
    ``ready`` (allocatable) / ``incomplete`` (dir present, files missing → offer
    download) / ``absent`` (no dir → offer download) / ``error`` (couldn't
    resolve). Optional ``?keys=k1,k2`` scans only those manifest keys (the picker
    passes what it's showing); default = every manifest model. Read-only."""
    manifest = get_models_dict(dict_return=True)
    keys_param = (request.args.get("keys") or "").strip()
    if keys_param:
        keys = [k for k in (s.strip() for s in keys_param.split(",")) if k in manifest]
    else:
        keys = list(manifest.keys())
    out = {}
    for k in keys:
        reason = _central_missing_reason(k)
        if reason is None:
            state = "ready"
        elif reason == "no model directory":
            state = "absent"
        elif reason == "directory present but files incomplete":
            state = "incomplete"
        else:
            state = "error"
        out[k] = {"state": state, "reason": reason}
    return jsonify(out)


@worker_bp.route("/llm/workers/<worker_id>/assign", methods=["POST"])
def workers_assign(worker_id):
    body = AssignRequest(**(request.get_json(silent=True) or {}))
    # t21: range-check any tolerance-band keys on the spill (same guard the bulk
    # path applies). The spill is a free-form dict on AssignRequest, so this is
    # the single path's only band validation.
    band_reason = _validate_band_values(body.spill)
    if band_reason is not None:
        return jsonify({"error": band_reason}), 400
    if body.model_key not in get_models_dict(dict_return=True):
        # JSON (not abort's HTML) so the UI surfaces a clean reason instead of a
        # raw 404 page. A key that isn't in the manifest is usually a name-vs-key
        # slip (e.g. "Wan2.1-VACE-1.3B" the display name vs the "Wan-AI~..." key).
        return jsonify({"error": f"unknown model key '{body.model_key}' — it is "
                        "not in central's manifest"}), 404
    # Operator BLOCK gate: a blocked model may not be (re)designated to any
    # worker. Clear 409 with the fix, same shape as the disk/engine refusals
    # below. Existing designations are left recorded (inert) — block does not
    # auto-unassign — so this only stops NEW assignments while blocked.
    if _model_blocked(body.model_key):
        return jsonify({"error": f"'{body.model_key}' is blocked from the serving "
                        "pool by the operator — unblock it (Models tab) before "
                        "assigning it to a worker"}), 409
    # Item 4 guard: a model can't be designated unless central itself holds the
    # files — otherwise the worker silently pulls ~50GB from HF at internet
    # speed (the 2026-07-03 sdxl-turbo saga). Clear 409 with the fix.
    missing = _central_missing_reason(body.model_key)
    if missing:
        return jsonify({"error": f"central does not have '{body.model_key}' on "
                        f"disk ({missing}) — download it on the Models tab first; "
                        "workers provision from central"}), 409
    # Disk-aware allocation: refuse a pull the worker's disk can't hold.
    _w = get_worker(worker_id)
    if _w is not None:
        disk_no = _disk_preflight_reason(_w, body.model_key)
        if disk_no:
            return jsonify({"error": f"'{body.model_key}' won't fit on "
                            f"{_w.get('name') or worker_id}: {disk_no} — free "
                            "space or pick another worker"}), 409
    # Engine gate (operator ruling 2026-07-17), defense-in-depth behind the UI:
    # a GGUF-only spill (explicit budget / layer offload) must not be written onto
    # a resolvably non-GGUF model. Autofit ({} / no spill) is engine-agnostic and
    # always allowed, so ordinary assignment (with or without an autofit spill) is
    # unaffected — only an explicit GGUF-only contract on a transformers/comfy key
    # is refused (clear 409), the same class the bulk route skips.
    ok, reason = _alloc_spill_ok_for_engine(body.spill, body.model_key)
    if not ok:
        return jsonify({"error": reason}), 409
    worker = assign_model(worker_id, body.model_key, spill=body.spill)
    if worker is None:
        abort(404, description="Unknown worker id.")
    # Lazy doctrine (operator 2026-07-16/17): assignment is ATTRIBUTION, never
    # a transfer order — and the worker's /probe DOWNLOADS an absent model, so
    # an unconditional assign-warm was a hidden pull. Warm only when the files
    # are already on the box (seat-now); an absent model waits for first call.
    if body.model_key in (worker.get("models_local") or []):
        _kick_warm(worker, [body.model_key], "assign")
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>/unassign", methods=["POST"])
def workers_unassign(worker_id):
    body = AssignRequest(**(request.get_json(silent=True) or {}))
    # Tiers v3 (operator decision 2026-07-05): 📌 pin = PERMANENT ATTRIBUTION
    # of the model to this worker, so a pinned designation cannot be removed
    # — unpin first. The pin lives in the agent's own settings and rides back
    # in every heartbeat's `config`, so the registry row is the truth here.
    _w = get_worker(worker_id)
    if _w is not None and ((_w.get("config") or {}).get("pinned") or {}).get(body.model_key):
        return jsonify({"ok": False, "error": {
            "code": "Pinned",
            "message": (f"{body.model_key} is pinned to "
                        f"{_w.get('name') or worker_id} — unpin first")}}), 409
    worker = unassign_model(worker_id, body.model_key)
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/chat/cancel/<request_id>", methods=["POST"])
def chat_cancel(request_id):
    """Cancel an in-flight chat, wherever it runs.

    Local-first (F1.3): publish control.cancel on the comms bus — the wired
    consumer calls job_store.cancel, which fires the cancel handle the local
    stream attached, so a generation served by THIS process actually stops
    (pre-comms, only worker-relayed requests were cancellable).

    Then the legacy fan-out: a request relayed to a GPU worker executes there,
    so its cancel event lives in that worker's process. We don't track which
    worker owns it; every online worker gets the cancel, the owner stops, the
    rest 404 harmlessly.
    """
    import httpx
    from abstract_hugpy_dev.comms import bus, job_store, TOPIC_CONTROL_CANCEL

    # Direct store cancel first — its return is cross-process truth (live
    # here, or flagged on the shared mirror for the sibling gunicorn worker
    # that owns the stream). The bus publish keeps control observable to any
    # subscriber; wire_cancel's redundant second cancel is a no-op.
    cancelled = job_store.cancel(request_id,
                                 reason="cancelled via /llm/chat/cancel")
    bus.publish(TOPIC_CONTROL_CANCEL, job_id=request_id, source="web",
                payload={"reason": "cancelled via /llm/chat/cancel"})
    for w in list_workers():
        if w.get("status") != "online":
            continue
        url = (w.get("url") or "").rstrip("/") + f"/infer/cancel/{request_id}"
        try:
            r = httpx.post(url, timeout=4.0)
            if r.status_code == 200:
                cancelled = True
        except Exception:
            continue
    return jsonify({"cancelled": cancelled, "request_id": request_id})


@worker_bp.route("/llm/workers/<worker_id>/unload", methods=["POST"])
def workers_unload(worker_id):
    """Free GPU VRAM on a worker by evicting cached model(s).

    Body: {"model_key": ...} unloads one model; {} or {"all": true} unloads
    everything the worker has in VRAM. The model stays ASSIGNED (we don't touch
    the registry) — this only drops it from the worker's live cache so the VRAM
    is reclaimed. Relays to the worker's /models/unload.
    """
    import httpx

    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    body = request.get_json(silent=True) or {}
    url = (worker.get("url") or "").rstrip("/") + "/models/unload"
    try:
        r = httpx.post(url, json=body, timeout=30.0)
        return jsonify(r.json())
    except Exception as exc:
        return jsonify({"ok": False, "evicted": False,
                        "error": f"{type(exc).__name__}: {exc}"})


def _relay_worker_op(worker_id: str, op_path: str, body: dict,
                     timeout: float, action: str,
                     retry_on_connect: bool = False) -> "tuple":
    """CON-05/06 + UTIL-02 relay: forward a privileged op to the worker's
    control agent and return its TYPED result verbatim (F3.4: errors are
    data, not exceptions). Operator-gated in _SENSITIVE; every call audited.

    retry_on_connect (config-style ops only): a /ops/config POST is ACKed and
    then the agent re-execs ~0.5s later, so a second config click during the
    ~5s blip hits a dead socket — which used to come back as a bare 502
    ("pin is broken"). For those ops a connect-class failure gets ONE retry
    after a 3s pause; if that also can't connect, the answer is an honest
    503 "agent is restarting" instead of the generic 502. Every other op
    keeps the historical single-shot behavior exactly."""
    import time as _time

    import httpx
    from .comms_routes import audit

    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    audit(f"worker.{action}", {"worker_id": worker_id,
                               "worker": worker.get("name"), "body": body})
    url = (worker.get("url") or "").rstrip("/") + op_path

    def _fail(exc):
        return jsonify({"ok": False,
                        "error": {"code": type(exc).__name__,
                                  "message": str(exc)}}), 502

    _connect_errors = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError)
    try:
        r = httpx.post(url, json=body, timeout=timeout)
        return jsonify(r.json()), r.status_code
    except _connect_errors as exc:
        if not retry_on_connect:
            return _fail(exc)
        _time.sleep(3.0)
        try:
            r = httpx.post(url, json=body, timeout=timeout)
            return jsonify(r.json()), r.status_code
        except _connect_errors:
            return jsonify({"ok": False, "error": {
                "code": "AgentRestarting",
                "message": ("worker agent is restarting to apply a previous "
                            "change — retry in a few seconds")}}), 503
        except Exception as exc2:  # noqa: BLE001 — non-connect retry failure
            return _fail(exc2)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@worker_bp.route("/llm/workers/<worker_id>/restart", methods=["POST"])
def workers_restart(worker_id):
    """CON-06: restart the worker agent process. The agent re-execs itself
    (rootless — central can't reach its systemctl --user); its persistent
    worker id means it re-registers as the same row. Availability is
    heartbeat-driven, so the row goes offline->online as it comes back."""
    return _relay_worker_op(worker_id, "/ops/restart",
                            request.get_json(silent=True) or {},
                            timeout=15.0, action="restart")


@worker_bp.route("/llm/workers/<worker_id>/free-ram", methods=["POST"])
def workers_free_ram(worker_id):
    """Non-destructive host-RAM reclaim: relay to the worker agent's
    /ops/free-ram, which runs gc + malloc_trim(0) + torch.empty_cache() to hand
    glibc's orphaned allocator arena back to the OS WITHOUT evicting any model
    (after a free glibc keeps the pages pooled, so RSS stays pinned otherwise).
    Returns the worker JSON verbatim; loaded models are left untouched."""
    return _relay_worker_op(worker_id, "/ops/free-ram",
                            request.get_json(silent=True) or {},
                            timeout=30.0, action="free-ram")


@worker_bp.route("/llm/workers/<worker_id>/evict", methods=["POST"])
def workers_evict(worker_id):
    """Targeted eviction: free ONE model's RAM+VRAM on a worker, letting the
    worker pick the mechanism by how the model is hosted (comfy /free, slot child
    SIGTERM->SIGKILL, or in-process ref-drop). Relays to the worker's /ops/evict.

    Body: {"model_key": ..., "force"?: bool}. We send the model_key, NEVER a PID
    (PIDs are per-box and recycled) — the worker resolves the model_key to its
    live handle and verifies identity at eviction time. force=true overrides the
    static/pinned/in-flight gate. The model stays ASSIGNED (registry untouched);
    this only drops it from residency. Unknown/not-resident is an idempotent
    no-op. Returns the worker JSON verbatim (host_mode, evicted, vram/ram freed).
    """
    return _relay_worker_op(worker_id, "/ops/evict",
                            request.get_json(silent=True) or {},
                            timeout=45.0, action="evict")


@worker_bp.route("/llm/workers/<worker_id>/slots/<slot_id>/relaunch",
                 methods=["POST"])
def workers_slot_relaunch(worker_id, slot_id):
    """k14: relaunch a worker's slot child with a new GPU-offload depth / context.

    The lever the k7 offload speed-cliff sweep needs: seat a GGUF at full offload,
    then relaunch it DOWN through decreasing n_gpu_layers, measuring tok/s at each
    step. Body: {"n_gpu_layers"?: int, "ctx"?: int} — omit either to keep it
    (n_gpu_layers absent => the slot re-autofits; an explicit count WINS). Relays
    to the worker agent's /slots/<slot_id>/relaunch, which asks the slot supervisor
    to STOP->RESPAWN its child (SIGTERM->SIGKILL) under a NEW pid — so this also
    addresses the ae "slot-child PID never recycles" blocker without a worker
    restart. Returns the worker's honest result: the echoed n_gpu_layers is the
    value the fresh child actually LAUNCHED with, not merely what was requested.

    404 unknown worker id; 409 when the worker is offline (can't relay); the
    worker itself answers 404 for an unknown slot and 409 for an empty slot, both
    propagated verbatim. Operator-gated in operator_auth._SENSITIVE, audited like
    every other worker op."""
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    if worker.get("status") != "online":
        # Offline worker: there is no agent to relay to. Refuse cleanly (409)
        # rather than let the relay time out into a generic 502.
        return jsonify({"ok": False, "error": {
            "code": "WorkerOffline",
            "message": f"worker {worker_id} is offline — cannot relaunch its "
                       "slot until it is back online"}}), 409
    body = request.get_json(silent=True) or {}
    payload = {k: body[k] for k in ("n_gpu_layers", "ctx")
               if body.get(k) not in (None, "")}
    return _relay_worker_op(worker_id, f"/slots/{slot_id}/relaunch", payload,
                            timeout=900.0, action="slot-relaunch")


@worker_bp.route("/llm/workers/<worker_id>/config", methods=["POST"])
def workers_config(worker_id):
    """Daylight item 3: set a worker's serving config from the console — e.g.
    {"slot_count": 1}. Persisted in the AGENT's own settings file (beats env
    drop-ins), applied via agent re-exec; the next heartbeat reports the
    effective values, so the row shows truth."""
    return _relay_worker_op(worker_id, "/ops/config",
                            request.get_json(silent=True) or {},
                            timeout=15.0, action="config",
                            retry_on_connect=True)


def _relay_pin_all(worker_id, pin: bool):
    """Pin (pin=True) or unpin (pin=False) EVERY model currently designated to
    this worker, reusing the SAME code path the single-model pin uses.

    The single pin (UI togglePin → workers_config) is one /ops/config POST with
    ``{"pinned": {model_key: true|null}}``; the worker agent's /ops/config
    ITERATES that map and applies it as one atomic settings-write + one re-exec.
    So the whole-worker action is that exact relay with EVERY key in the dict
    instead of one — no duplicated pin logic, and one agent restart rather than
    one per model (which would stack restarts and spuriously fail later pins).

    Returns a Flask (json, 200): per-model ``results`` ({model_key: "ok"|error
    message}), summary ``counts`` and the relay's ``restarting`` flag. Resilient
    by design — a relay failure marks every model errored and STILL returns the
    full map (never a bare 5xx that would abort the caller before it sees which
    models were affected)."""
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    keys = list(worker.get("models") or [])
    action = "pin_all" if pin else "unpin_all"
    if not keys:
        return jsonify({"ok": True, "pinned": pin, "results": {},
                        "counts": {"ok": 0, "error": 0, "total": 0},
                        "restarting": False,
                        "note": "no models are designated to this worker"})
    # Same payload shape as the single pin (workers_config), just every key.
    payload = {"pinned": {mk: (True if pin else None) for mk in keys}}
    resp, status = _relay_worker_op(worker_id, "/ops/config", payload,
                                    timeout=15.0, action=action,
                                    retry_on_connect=True)
    data = resp.get_json(silent=True) or {}
    ok = (200 <= status < 300) and data.get("ok", True) is not False
    if ok:
        results = {mk: "ok" for mk in keys}
        counts = {"ok": len(keys), "error": 0, "total": len(keys)}
    else:
        err = data.get("error")
        msg = ((err.get("message") if isinstance(err, dict) else err)
               or data.get("reason") or f"config relay failed (HTTP {status})")
        results = {mk: msg for mk in keys}
        counts = {"ok": 0, "error": len(keys), "total": len(keys)}
    out = {"ok": ok, "pinned": pin, "results": results, "counts": counts,
           "restarting": bool(data.get("restarting"))}
    if not ok and data.get("error"):
        out["error"] = data["error"]      # let the UI's fetchJson surface it too
    # Always 200: the structured body (ok + counts) carries success/failure, so
    # the caller sees the per-model map even when the underlying relay failed.
    return jsonify(out)


@worker_bp.route("/llm/workers/<worker_id>/pin-all", methods=["POST"])
def workers_pin_all(worker_id):
    """Tiers v3 bulk action: 📌 pin EVERY model currently designated to this
    worker, in ONE settings-write via the single-pin relay (see _relay_pin_all).
    Sticky by design — each pinned model then refuses unassign (409 'unpin
    first') until /unpin-all (or a per-model unpin) reverses it. Operator-gated
    + audited like every other /ops/config relay."""
    return _relay_pin_all(worker_id, pin=True)


@worker_bp.route("/llm/workers/<worker_id>/unpin-all", methods=["POST"])
def workers_unpin_all(worker_id):
    """Inverse of /pin-all — the undo. Unpins EVERY model designated to this
    worker in one /ops/config write (a `pinned` map of nulls), same relay/code
    path; afterward the models can be unassigned again."""
    return _relay_pin_all(worker_id, pin=False)


# ── bulk residency (todo t12) ────────────────────────────────────────────────
# The console's per-worker Serving table lets the operator multi-SELECT models
# and change their RESIDENCY tier in ONE action, instead of clicking the per-
# model residency control N times. N single-model /config POSTs would stack N
# agent re-execs (each /ops/config schedules a restart) and spuriously fail the
# later ones during the ~5s blip — the EXACT problem _relay_pin_all already
# solved for pins: relay ONE /ops/config with the whole map = one atomic
# settings-write + one re-exec.
#
# ⚠ RESIDENCY ONLY — this is NOT pin. Residency is the two-tier serving policy
#   (on-demand default | 🔒static, the one tier that promises local presence /
#   blocks eviction); 📌pin is the SEPARATE routing-persistence axis. This route
#   never touches `pinned`.
#
# The worker agent's /ops/config already DEEP-MERGES a `residency` map
# (agent.py ops_config: {"<model_key>": "static"|null}, one merge + one
# _schedule_restart) — exactly like `pinned` — so the whole-selection action is
# that same single relay with every selected key in the dict. No new worker
# release, no duplicated policy: on-demand is the default and is stored as NO
# entry, so the normalized wire value for on-demand is ``null`` (the single-
# model setResidency uses the same null-clears convention).
def _normalize_residency(mode) -> "str | None":
    """Map a caller-supplied residency mode onto the wire value the agent stores.

    "static" is the only stored tier; on-demand IS the default and clears the
    override — so null / "" / "on-demand" and the agent's legacy synonyms
    ("serving"/"warm") all normalize to None. Anything else is rejected by the
    route (a bad tier must 400, never silently clear)."""
    if mode == "static":
        return "static"
    if mode in (None, "", "on-demand", "on_demand", "serving", "warm"):
        return None
    return "__invalid__"


def _relay_residency_map(worker_id, model_keys, mode):
    """Set the residency tier of MANY designated models to ``mode`` in ONE
    settings-write, reusing the SAME /ops/config relay the single-model
    setResidency uses (workers_config).

    ``model_keys`` — the operator's selection (a subset of this worker's
    designated models; NOT necessarily every model, unlike pin-all). ``mode`` —
    "static" or on-demand (null/"on-demand"). One /ops/config POST carries
    ``{"residency": {mk: <wire>, ...}}`` for every selected key, and the agent
    applies it as one atomic merge + one re-exec — never N restarts.

    Returns a Flask (json, 200) with the SAME shape /pin-all returns so the UI
    can surface it identically: per-model ``results`` ({model_key: "ok"|error}),
    summary ``counts``, ``mode``, and the relay's ``restarting`` flag. Resilient:
    a relay failure marks every selected model errored and STILL returns the full
    map (never a bare 5xx that would abort the caller before it sees the outcome).
    """
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    wire = _normalize_residency(mode)
    if wire == "__invalid__":
        return jsonify({"ok": False, "error": {
            "code": "BadValue",
            "message": (f"residency mode {mode!r} must be 'static' or "
                        "'on-demand' (null/'on-demand' restores the default)")}}), 400
    # Only act on keys ACTUALLY designated to this worker — a selection can go
    # stale between render and click (an unassign in another tab). Silently
    # ignoring off-worker keys keeps the settings-write honest (the agent would
    # merge a residency entry for a model it isn't assigned otherwise).
    designated = set(worker.get("models") or [])
    keys = [mk for mk in (model_keys or []) if mk in designated]
    skipped = [mk for mk in (model_keys or []) if mk not in designated]
    label = "static" if wire == "static" else "on-demand"
    action = f"residency_all:{label}"
    if not keys:
        out = {"ok": True, "mode": label, "results": {},
               "counts": {"ok": 0, "error": 0, "total": 0},
               "restarting": False,
               "note": "none of the selected models are designated to this worker"}
        if skipped:
            out["skipped"] = skipped
        return jsonify(out)
    # Same payload shape as the single residency change (workers_config), just
    # every selected key at once — the agent deep-merges them in one write.
    payload = {"residency": {mk: wire for mk in keys}}
    resp, status = _relay_worker_op(worker_id, "/ops/config", payload,
                                    timeout=15.0, action=action,
                                    retry_on_connect=True)
    data = resp.get_json(silent=True) or {}
    ok = (200 <= status < 300) and data.get("ok", True) is not False
    if ok:
        results = {mk: "ok" for mk in keys}
        counts = {"ok": len(keys), "error": 0, "total": len(keys)}
    else:
        err = data.get("error")
        msg = ((err.get("message") if isinstance(err, dict) else err)
               or data.get("reason") or f"config relay failed (HTTP {status})")
        results = {mk: msg for mk in keys}
        counts = {"ok": 0, "error": len(keys), "total": len(keys)}
    out = {"ok": ok, "mode": label, "results": results, "counts": counts,
           "restarting": bool(data.get("restarting"))}
    if skipped:
        out["skipped"] = skipped
    if not ok and data.get("error"):
        out["error"] = data["error"]      # let the UI's fetchJson surface it too
    # Always 200: the structured body (ok + counts) carries success/failure, so
    # the caller sees the per-model map even when the underlying relay failed.
    return jsonify(out)


@worker_bp.route("/llm/workers/<worker_id>/residency-all", methods=["POST"])
def workers_residency_all(worker_id):
    """Bulk-residency (todo t12): set the RESIDENCY tier of a SELECTED set of
    this worker's designated models in ONE settings-write via the single-model
    residency relay (see _relay_residency_map). Body:
    ``{"model_keys": [...], "mode": "static"|"on-demand"}``. One agent re-exec,
    not one per model. Residency only — never touches 📌pin. Operator-gated +
    audited like every other /ops/config relay (config/pin-all family)."""
    body = request.get_json(silent=True) or {}
    model_keys = body.get("model_keys")
    if not isinstance(model_keys, list) or not model_keys:
        return jsonify({"ok": False, "error": {
            "code": "BadValue",
            "message": 'model_keys must be a non-empty list of model keys'}}), 400
    return _relay_residency_map(worker_id, model_keys, body.get("mode"))


# ── bulk alloc (todo t15) ────────────────────────────────────────────────────
# The operator's clarification of what the multi-select was FOR: "i meant ALLOC
# for the group set, but residency can remain as well." So the same Serving-table
# selection that drives bulk residency also drives a bulk GPU-ALLOCATION change:
# set the per-model spill (autofit / max GPU / CPU only / custom budgets) for the
# whole selection in ONE action.
#
# ⚠ ALLOC IS NOT RESIDENCY, AND NOT A RELAY. The single-model alloc editor
#   applies via POST /llm/workers/<id>/assign {model_key, spill} — a CENTRAL
#   REGISTRY write to spill_by_model[model_key] (assign_model), NOT the
#   /ops/config family. So, unlike residency/pin, an alloc change does NOT restart
#   the agent: the spill is the model's resource CONTRACT, applied by the worker
#   the next time it loads the model. This route therefore does NOT relay to the
#   worker and NEVER carries a `restarting` flag — it writes N registry entries in
#   one request and returns per-model results the same shape residency-all uses.
#
# The spill value set MIRRORS the single editor's modeToSpill exactly:
#   autofit  -> {}                     (clears any override back to autofit)
#   max GPU  -> {"n_gpu_layers": -1}   (all layers on GPU)
#   CPU only -> {"n_gpu_layers": "off"}
#   custom   -> {"gpu_mem_gib"?, "cpu_mem_gib"?, "threads"?}  explicit budgets
# t21 tolerance-band keys ride the SAME spill contract (the explicit allocation),
# additive + optional. gpu/cpu deviations are percent-of-honest-whole bands around
# the gpu_mem_gib/cpu_mem_gib targets; ctx_pct is the CTX target (percent of the
# model's max context) and ctx_deviation_pct its band; priority (int >=0, 0=normal)
# lets a higher-priority allocation compress lower ones within their bands first.
#
# t48 addendum: the ONE-spill-broadcast shape above is correct when the operator
# picks an absolute (or autofit/max-GPU/CPU-only) — every member is meant to get
# the SAME contract. It is WRONG for a PERCENT VRAM/RAM budget: "40%" must mean
# 40% of each model's OWN size, not one absolute GiB number (resolved once
# against the worker's capacity) stamped identically on every member regardless
# of that member's actual size. `workers_alloc_all` therefore also accepts
# `spills: {model_key: spill}` — a per-model contract map, applied by
# `_apply_alloc_map_multi` — for exactly this case; the UI picks whichever shape
# fits (`spill` when every member truly wants the same contract, `spills` when a
# percent budget was resolved per model).
_BAND_SPILL_KEYS = {"gpu_mem_gib_deviation_pct", "cpu_mem_gib_deviation_pct",
                    "ctx_pct", "ctx_deviation_pct", "priority"}
_ALLOC_SPILL_KEYS = {"n_gpu_layers", "gpu_mem_gib", "cpu_mem_gib",
                     "threads", "tensor_split"} | _BAND_SPILL_KEYS


def _validate_alloc_spill(spill):
    """Validate a bulk-alloc spill dict against the recognized AssignRequest
    knobs. Returns (clean_dict, None) on success or (None, reason) on a bad shape.

    ``None`` / ``{}`` are BOTH valid and mean autofit (clear the override) — the
    same convention modeToSpill/assign_model use. Unknown keys are rejected so a
    typo can't silently write a no-op contract; n_gpu_layers accepts the same
    int|"auto"|"off" the single path does."""
    if spill is None:
        return {}, None
    if not isinstance(spill, dict):
        return None, "spill must be an object (or null/{} for autofit)"
    unknown = sorted(set(spill) - _ALLOC_SPILL_KEYS)
    if unknown:
        return None, (f"unsupported spill keys {unknown}; recognized: "
                      f"{sorted(_ALLOC_SPILL_KEYS)}")
    ngl = spill.get("n_gpu_layers")
    if ngl is not None and not (
            isinstance(ngl, int) or ngl in ("auto", "off")
            or (isinstance(ngl, str) and ngl.lstrip("-").isdigit())):
        return None, 'n_gpu_layers must be an int, "auto", or "off"'
    band_reason = _validate_band_values(spill)
    if band_reason is not None:
        return None, band_reason
    return dict(spill), None


def _validate_band_values(spill) -> "str | None":
    """Range-check the t21 tolerance-band keys on a spill, or None if clean.

    Consumption clamps defensively (flex.band_bounds / ctx_band_bounds), so this
    is a UX guard that rejects an obviously-wrong number at the door rather than a
    safety gate. Shared by the single (/assign) and bulk (/alloc-all) paths so both
    reject identically. Absent keys are fine (the bands are opt-in)."""
    if not isinstance(spill, dict):
        return None
    for k in ("gpu_mem_gib_deviation_pct", "cpu_mem_gib_deviation_pct",
              "ctx_deviation_pct"):
        if k in spill and spill[k] is not None:
            try:
                v = float(spill[k])
            except (TypeError, ValueError):
                return f"{k} must be a number 0..100 (percent-of-total band)"
            if not (0.0 <= v <= 100.0):
                return f"{k}={spill[k]} out of range — must be 0..100"
    if "ctx_pct" in spill and spill["ctx_pct"] is not None:
        try:
            v = int(spill["ctx_pct"])
        except (TypeError, ValueError):
            return "ctx_pct must be an integer 1..100 (percent of max context)"
        if not (1 <= v <= 100):
            return f"ctx_pct={spill['ctx_pct']} out of range — must be 1..100"
    if "priority" in spill and spill["priority"] is not None:
        try:
            v = int(spill["priority"])
        except (TypeError, ValueError):
            return "priority must be a non-negative integer (0 = normal)"
        if v < 0:
            return f"priority={spill['priority']} out of range — must be >= 0"
    return None


# ── engine gating (operator ruling 2026-07-17, NARROWED by t26) ─────────────
# Original ruling: "explicit budget … should only be an option if the model is a
# gguf file." Refined (t26): "the non ggufs should still have options, just not
# explicit — the autofit, maxgpu and cpu only should still be on the table as its
# easy to infer what those would indicate for a transformer."
#
# So the engine-exclusive set NARROWS to ONLY the EXPLICIT-BUDGET class —
# gpu_mem_gib / cpu_mem_gib / threads / tensor_split. These are per-model
# resource-contract numbers with true meaning only in the llama.cpp world.
#
# Autofit / Max GPU / CPU only are ENGINE-AGNOSTIC PLACEMENT INTENT that ride the
# n_gpu_layers wire field (autofit={}, max GPU={n_gpu_layers:-1}, CPU only=
# {n_gpu_layers:"off"|0}). The GGUF loader reads them as layer counts; the
# transformers loader now interprets the SAME field as placement (spill.py
# transformers_max_memory / n_gpu_layers_intent — all-on-GPU / CPU-only /
# fit-and-spill), so they are NO LONGER dead knobs for transformers and apply to
# every engine. Only the explicit class skips/409s on non-GGUF.
# t21 bands ride the explicit allocation and are GGUF-only just like the budgets
# they band (spec: "explicit budgets are GGUF-only"; the UI gates them the same
# way). So a spill carrying ONLY a band/priority is still classified GGUF-only.
_EXPLICIT_BUDGET_KEYS = ({"gpu_mem_gib", "cpu_mem_gib", "threads", "tensor_split"}
                         | _BAND_SPILL_KEYS)


def _alloc_is_gguf_only(spill) -> bool:
    """True ONLY when this spill carries an EXPLICIT-BUDGET knob (gpu_mem_gib /
    cpu_mem_gib / threads / tensor_split) — the sole GGUF-exclusive class (t26).
    Autofit ({}/None) AND the placement-intent modes (Max GPU / CPU only, which
    carry ONLY n_gpu_layers) are engine-agnostic and apply to every engine."""
    if not spill:
        return False
    return any(k in spill for k in _EXPLICIT_BUDGET_KEYS)


def _model_framework(model_key: str) -> "str | None":
    """The model's engine/framework from central's registry ('gguf'|'llama_cpp'
    |'transformers'|'comfy'|…), lowercased, or None if it can't be resolved.
    Central knows this authoritatively (get_model_config), so the engine gate is
    enforced server-side — the UI gating is a courtesy, not the enforcement."""
    try:
        from ....imports.config.main import get_model_config
        cfg = get_model_config(model_key)
        fw = getattr(cfg, "framework", None)
        return str(fw).lower() if fw else None
    except Exception:  # noqa: BLE001 — unresolvable engine: caller treats as unknown
        return None


def _is_gguf_framework(fw: "str | None") -> bool:
    """GGUF family: the HF-canonical 'gguf' plus the llama_cpp synonym."""
    return fw in ("gguf", "llama_cpp")


def _alloc_spill_ok_for_engine(spill, model_key) -> "tuple[bool, str | None]":
    """Single-model engine gate: is this spill allowed for ``model_key``'s engine?

    Returns (True, None) when allowed, (False, reason) when a GGUF-only spill is
    aimed at a resolvably non-GGUF model. Autofit ({}/None) is always allowed
    (engine-agnostic), so this only ever refuses an EXPLICIT GGUF-only contract on
    a transformers/comfy key — the same class the bulk route skips. An
    unresolvable engine fails SAFE (refuse the GGUF-only write). Pure + directly
    unit-testable (mirrors _apply_alloc_map's per-key gate for the single path)."""
    if not _alloc_is_gguf_only(spill):
        return True, None
    fw = _model_framework(model_key)
    if _is_gguf_framework(fw):
        return True, None
    shown = fw or "unknown engine"
    return False, (f"{_alloc_label(spill)} is a GGUF-only allocation "
                   f"(explicit budget); '{model_key}' is {shown} — use autofit, "
                   "Max GPU, or CPU only for a non-GGUF model")


def _apply_alloc_map(worker_id, model_keys, spill):
    """Set the GPU-ALLOCATION (spill) of a SELECTED set of a worker's designated
    models in ONE request, reusing the SAME registry write the single-model alloc
    uses (assign_model, POST /assign). No relay, no restart — a per-model spill is
    a central registry contract applied on the model's next load.

    ENGINE GATE (operator ruling): when ``spill`` is a GGUF-only allocation
    (_alloc_is_gguf_only), non-GGUF members are SKIPPED with an honest per-model
    reason and counted under ``counts.skipped`` — never written. Autofit applies
    to every engine. This is defense-in-depth behind the UI gating.

    ``model_keys`` — the operator's selection (a subset). ``spill`` — the alloc
    contract to write (already validated). Returns a Flask (json, 200) with the
    SAME shape residency-all/pin-all return so the UI can surface it identically:
    per-model ``results`` ({model_key: "ok"|"skipped — …"|error}), ``counts``, an
    ``alloc`` label, and (constant) ``restarting: False`` — alloc never re-execs
    the agent. Audited like the single assign (operator-gated)."""
    from .comms_routes import audit

    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    # Only act on keys ACTUALLY designated to this worker — a selection can go
    # stale between render and click. Off-worker keys are reported as skipped and
    # never assigned (assign_model would otherwise ADD them, silently designating
    # a model the operator only meant to re-allocate).
    designated = set(worker.get("models") or [])
    keys = [mk for mk in (model_keys or []) if mk in designated]
    off_worker = [mk for mk in (model_keys or []) if mk not in designated]
    label = _alloc_label(spill)
    gguf_only = _alloc_is_gguf_only(spill)
    audit("worker.alloc_all", {"worker_id": worker_id,
                               "worker": worker.get("name"),
                               "model_keys": keys, "spill": spill,
                               "alloc": label, "gguf_only": gguf_only})
    if not keys:
        out = {"ok": True, "alloc": label, "results": {},
               "counts": {"ok": 0, "error": 0, "skipped": 0, "total": 0},
               "restarting": False,
               "note": "none of the selected models are designated to this worker"}
        if off_worker:
            out["off_worker"] = off_worker
        return jsonify(out)
    results = {}
    okN = 0
    skipN = 0
    for mk in keys:
        # ENGINE GATE: a GGUF-only spill must not touch a non-GGUF model. Skip it
        # with an honest reason (registry untouched) instead of writing a knob
        # the transformers loader ignores. Unknown engine -> treat as non-GGUF
        # (fail safe: don't write a GGUF-only contract onto an unresolvable model).
        if gguf_only:
            fw = _model_framework(mk)
            if not _is_gguf_framework(fw):
                shown = fw or "unknown engine"
                results[mk] = (f"skipped — {label} is a GGUF-only allocation; "
                               f"this model is {shown}")
                skipN += 1
                continue
        try:
            # assign_model writes spill_by_model[mk] = spill ({} clears it). The
            # model is already designated (we filtered to `designated`), so this
            # only rewrites the contract — it never newly-adds a model.
            w = assign_model(worker_id, mk, spill=spill)
            if w is None:
                results[mk] = "worker vanished mid-apply"
            else:
                results[mk] = "ok"
                okN += 1
        except Exception as exc:  # noqa: BLE001 — one bad key must not abort the rest
            results[mk] = f"{type(exc).__name__}: {exc}"
    # Re-seat the models whose files are already on the box so the new contract
    # takes effect without waiting for the next organic call — same seat-now
    # policy the single /assign uses (warm only local files; absent ones wait).
    try:
        w2 = get_worker(worker_id) or worker
        local = set(w2.get("models_local") or [])
        seat = [mk for mk in keys if mk in local and results.get(mk) == "ok"]
        if seat:
            _kick_warm(w2, seat, "alloc_all")
    except Exception:  # noqa: BLE001 — the warm is best-effort, never fails the write
        pass
    errN = len(keys) - okN - skipN
    # ok=True when nothing HARD-errored: an engine skip is an expected, correct
    # outcome (not a failure), so an all-transformers gguf-only apply is ok:true
    # with everything skipped and nothing written.
    out = {"ok": errN == 0, "alloc": label, "results": results,
           "counts": {"ok": okN, "error": errN, "skipped": skipN, "total": len(keys)},
           "restarting": False}
    if off_worker:
        # off-worker staleness is surfaced separately from the engine skips (which
        # live in results with their reason) so the two never conflate.
        out["off_worker"] = off_worker
    return jsonify(out)


def _apply_alloc_map_multi(worker_id, model_keys, spills_by_key):
    """Per-model variant of _apply_alloc_map (t48): each selected model gets its
    OWN spill contract in the SAME one-request bulk write, instead of one shared
    dict stamped on every key.

    Root cause this exists to fix: the bulk editor's PERCENT budgets (VRAM/RAM)
    used to resolve once client-side against the WORKER's capacity into a single
    absolute GiB number, then that one number rode as `spill` and was applied
    identically to every selected model — correct (at best) for whichever model
    that absolute actually matched, wrong for every other member of a
    differently-sized group (operator, t48: "...not the total for the
    particular model that happened to be the first in the list's actual ram
    alloc"). The fix resolves the percent PER MODEL (against that model's own
    effective size) client-side, then sends the resulting per-model absolutes
    here in one request via `spills: {model_key: spill}` instead of one shared
    `spill` — still no percent concept on the wire, just fanned per key.

    Same registry write, same engine gate (computed per key off that key's OWN
    spill, since different members can carry different explicit-budget keys),
    same warm-reseat, same response shape as _apply_alloc_map."""
    from .comms_routes import audit

    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    # Only act on keys ACTUALLY designated to this worker — same staleness guard
    # as _apply_alloc_map.
    designated = set(worker.get("models") or [])
    keys = [mk for mk in (model_keys or []) if mk in designated]
    off_worker = [mk for mk in (model_keys or []) if mk not in designated]
    audit("worker.alloc_all", {"worker_id": worker_id,
                               "worker": worker.get("name"),
                               "model_keys": keys,
                               "spills": {mk: (spills_by_key.get(mk) or {}) for mk in keys},
                               "alloc": "per-model", "per_model": True})
    if not keys:
        out = {"ok": True, "alloc": "per-model", "results": {},
               "counts": {"ok": 0, "error": 0, "skipped": 0, "total": 0},
               "restarting": False,
               "note": "none of the selected models are designated to this worker"}
        if off_worker:
            out["off_worker"] = off_worker
        return jsonify(out)
    results = {}
    okN = 0
    skipN = 0
    for mk in keys:
        spill = spills_by_key.get(mk) or {}
        # ENGINE GATE: same rule as the broadcast path, evaluated against THIS
        # key's own spill (a per-model map can carry different explicit-budget
        # keys per member, unlike the single shared `spill`).
        if _alloc_is_gguf_only(spill):
            fw = _model_framework(mk)
            if not _is_gguf_framework(fw):
                shown = fw or "unknown engine"
                results[mk] = (f"skipped — {_alloc_label(spill)} is a GGUF-only "
                               f"allocation; this model is {shown}")
                skipN += 1
                continue
        try:
            # assign_model writes spill_by_model[mk] = spill ({} clears it).
            w = assign_model(worker_id, mk, spill=spill)
            if w is None:
                results[mk] = "worker vanished mid-apply"
            else:
                results[mk] = "ok"
                okN += 1
        except Exception as exc:  # noqa: BLE001 — one bad key must not abort the rest
            results[mk] = f"{type(exc).__name__}: {exc}"
    # Re-seat the models whose files are already on the box, same seat-now
    # policy _apply_alloc_map uses.
    try:
        w2 = get_worker(worker_id) or worker
        local = set(w2.get("models_local") or [])
        seat = [mk for mk in keys if mk in local and results.get(mk) == "ok"]
        if seat:
            _kick_warm(w2, seat, "alloc_all")
    except Exception:  # noqa: BLE001 — the warm is best-effort, never fails the write
        pass
    errN = len(keys) - okN - skipN
    out = {"ok": errN == 0, "alloc": "per-model", "results": results,
           "counts": {"ok": okN, "error": errN, "skipped": skipN, "total": len(keys)},
           "restarting": False}
    if off_worker:
        out["off_worker"] = off_worker
    return jsonify(out)


def _alloc_label(spill) -> str:
    """Human label for a spill contract (mirrors the UI's spillLabel)."""
    if not spill:
        return "autofit"
    ngl = spill.get("n_gpu_layers")
    if ngl in (-1, "-1"):
        return "max GPU"
    if ngl in (0, "0", "off"):
        return "CPU only"
    parts = []
    if spill.get("gpu_mem_gib") is not None:
        parts.append(f"{spill['gpu_mem_gib']}G VRAM")
    if spill.get("cpu_mem_gib") is not None:
        parts.append(f"{spill['cpu_mem_gib']}G RAM")
    if spill.get("threads") is not None:
        parts.append(f"{spill['threads']} cores")
    return " · ".join(parts) or "custom"


@worker_bp.route("/llm/workers/<worker_id>/alloc-all", methods=["POST"])
def workers_alloc_all(worker_id):
    """Bulk-alloc (todo t15; per-model % t48): set the GPU ALLOCATION (spill
    contract) of a SELECTED set of this worker's designated models in ONE
    request, via the same registry write the single-model alloc uses
    (assign_model). Body is EITHER:
      ``{"model_keys": [...], "spill": {...}|{}|null}`` — ONE contract
        BROADCAST to every selected key (autofit={} / max GPU={n_gpu_layers:-1}
        / CPU only={n_gpu_layers:"off"} / custom budgets) — unchanged since
        t15; OR
      ``{"model_keys": [...], "spills": {model_key: {...}|{}|null, ...}}`` —
        (t48) a PER-MODEL contract, one entry per selected key. This is how the
        bulk editor's PERCENT VRAM/RAM budgets ride the wire: the UI resolves
        each model's percent against ITS OWN effective size client-side (no
        percent concept lands here either — same "no schema change for the
        percent itself" posture as t15, the wire still only ever sees resolved
        GiB) and sends the resulting per-model absolutes in one request instead
        of one flat number broadcast to every member regardless of its size.
    NO agent restart — a spill is a registry contract applied on next load.
    Operator-gated + audited like the single /assign."""
    body = request.get_json(silent=True) or {}
    model_keys = body.get("model_keys")
    if not isinstance(model_keys, list) or not model_keys:
        return jsonify({"ok": False, "error": {
            "code": "BadValue",
            "message": 'model_keys must be a non-empty list of model keys'}}), 400
    spills_in = body.get("spills")
    if spills_in is not None:
        if not isinstance(spills_in, dict):
            return jsonify({"ok": False, "error": {
                "code": "BadValue",
                "message": "spills must be an object of {model_key: spill}"}}), 400
        per_key = {}
        for mk in model_keys:
            clean, reason = _validate_alloc_spill(spills_in.get(mk))
            if reason is not None:
                return jsonify({"ok": False, "error": {
                    "code": "BadValue", "message": f"spills[{mk}]: {reason}"}}), 400
            per_key[mk] = clean
        return _apply_alloc_map_multi(worker_id, model_keys, per_key)
    clean, reason = _validate_alloc_spill(body.get("spill"))
    if reason is not None:
        return jsonify({"ok": False, "error": {
            "code": "BadValue", "message": reason}}), 400
    return _apply_alloc_map(worker_id, model_keys, clean)


@worker_bp.route("/llm/workers/<worker_id>/reap", methods=["POST"])
def workers_reap(worker_id):
    """Tiers-v2 slice 4: reclaim a worker's disk. The bookend to unassign —
    delete local files of models that are on disk but no longer needed.
    Body: {"dry_run": true} previews (default); {"all": true} or
    {"model_keys":[...]} reclaims. The worker re-proves every guard
    (assigned/loaded/pinned/comfy) at delete time. Operator-gated + audited."""
    return _relay_worker_op(worker_id, "/reap",
                            request.get_json(silent=True) or {},
                            timeout=120.0, action="reap")


# ── AUTO-REAP (slice 8, Part B — opt-in, heartbeat-driven, guarded) ─────────
# Operator ask 2026-07-17: "there needs to be a way to auto approve this". This
# RETIRES the 'central never auto-approves' absolutism — but only as an OPT-IN
# (auto_reap default false; the hand-approve flow stays the default posture).
# When a worker has opted in, central's heartbeat ingest fires EXACTLY the
# operator reap-approve flow (recompute → intersect-with-itself → audit →
# guarded relay). NO new timer/daemon — it is driven only by the beats the
# worker already sends. The worker-side re-prove chain (_reap_reclaim) is
# untouched, so nothing loaded/static/provisioning/gated is ever deletable, and
# an auto-fire reclaims at most the proposal's own need (never more).
def _auto_reap_cooldown_s() -> float:
    """Per-worker minimum seconds between auto-fires. Constant with an env
    override so a wedged proposal can't hammer the relay every beat."""
    try:
        return float(os.environ.get("HUGPY_AUTO_REAP_COOLDOWN_S", "300"))
    except (TypeError, ValueError):
        return 300.0


def _maybe_auto_reap(worker_id: str, worker: dict) -> None:
    """Event-driven auto-reap check, run on heartbeat ingest. Fires the guarded
    reap-approve flow ONCE when, and only when, ALL hold:
      * the worker opted in (worker['auto_reap'] truthy);
      * it is over budget with a NON-EMPTY proposal (the same read-only proposal
        the console renders — proposed only when actually over budget);
      * the per-worker cooldown has elapsed since the last auto-fire.
    Best-effort: any failure here must never fail the heartbeat. Blast radius is
    bounded to the proposal's own keys — central proposes exactly the `need`, and
    the worker re-proves every guard at delete time."""
    try:
        if not worker.get("auto_reap"):
            return                                    # default posture — hand-approve
        proposal = worker_storage_view(worker_id) or {}
        if not proposal.get("over_budget"):
            return                                    # nothing to do
        evictions = proposal.get("proposed_evictions") or []
        if not evictions:
            return                                    # over budget but nothing eligible
        now = time.time()
        last = worker.get("last_auto_reap_at")
        try:
            if last is not None and (now - float(last)) < _auto_reap_cooldown_s():
                return                                # cooldown suppresses re-fire
        except (TypeError, ValueError):
            pass
        keys = [e["model_key"] for e in evictions if e.get("model_key")]
        if not keys:
            return
        # Stamp BEFORE firing so a slow relay can't let the next beat double-fire.
        record_worker_auto_reap(worker_id, now)
        _execute_reap(worker_id, worker, keys, trigger="auto")
    except Exception as exc:  # noqa: BLE001 — a heartbeat must never fail on this
        logger.warning("auto-reap check for %s skipped: %s", worker_id, exc)


def _execute_reap(worker_id: str, worker: dict, approved_keys: list,
                  trigger: str = "operator"):
    """The SHARED reap-approve core used by BOTH the operator route and the
    heartbeat auto-fire (no duplicated policy): recompute the proposal off live
    state, INTERSECT the approved keys with it (defense against a stale render or
    a since-protected model), AUDIT (the event name distinguishes operator vs
    auto), and relay to the SAME guarded worker executor (/reap) where
    _reap_reclaim re-proves every guard per key. Returns (data_dict, status).

    ``trigger`` — "operator" (hand-approved) or "auto" (auto_reap fired). It only
    changes the AUDIT event name + a marker in the result, never the guards."""
    from .comms_routes import audit
    proposal = worker_storage_view(worker_id) or {}
    ev_by_key = {e["model_key"]: e
                 for e in (proposal.get("proposed_evictions") or [])}
    intersected = [k for k in approved_keys if k in ev_by_key]
    dropped = [k for k in approved_keys if k not in ev_by_key]
    approved_evictions = [ev_by_key[k] for k in intersected]

    event = "worker.auto-reap" if trigger == "auto" else "worker.reap-approve"
    audit(event, {
        "worker_id": worker_id, "worker": worker.get("name"),
        "trigger": trigger,
        "approved": approved_keys, "dropped": dropped,
        "evictions": approved_evictions,
        "freed_estimate_bytes": sum(e.get("bytes") or 0
                                    for e in approved_evictions),
    })

    if not intersected:
        return {
            "ok": True, "freed_bytes": 0, "results": [],
            "approved": approved_keys, "reaped": [], "dropped": dropped,
            "trigger": trigger,
            "note": ("approved models are no longer eligible for reaping "
                     "(loaded/assigned/pinned/provisioning, or the worker is back "
                     "under budget) — nothing deleted"),
        }, 200

    resp, status = _relay_worker_op(
        worker_id, "/reap", {"model_keys": intersected},
        timeout=120.0, action=event.split(".", 1)[1])
    data = resp.get_json(silent=True)
    if isinstance(data, dict):
        data.setdefault("approved", approved_keys)
        data["reaped"] = intersected
        data["dropped"] = dropped
        data["trigger"] = trigger
        return data, status
    return {"ok": False, "trigger": trigger, "raw_status": status}, status


@worker_bp.route("/llm/workers/<worker_id>/auto-reap", methods=["POST"])
def workers_set_auto_reap(worker_id):
    """Operator-gated opt-in for AUTO-REAP (slice 8, Part B). Body:
    ``{"enabled": true|false}``. Default OFF — the hand-approve flow is the
    default posture. When on, central's heartbeat ingest fires the guarded
    reap-approve flow when the worker is over budget with a proposal (cooldown-
    limited). Same operator-gated route family as limits/pool/admission."""
    body = request.get_json(silent=True) or {}
    enabled = bool(body.get("enabled"))
    worker = set_worker_auto_reap(worker_id, enabled)
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>/reap-approve", methods=["POST"])
def workers_reap_approve(worker_id):
    """Operator-approved, LRU-guarded eviction of COLD local models when a worker
    is over its storage budget — the human-in-the-loop bookend to the read-only
    proposal.

    Flow: the console renders ``storage.over_budget`` + ``storage.proposed_evictions``
    (computed read-side in _public_view/storage_proposal — a PURE, no-daemon,
    no-auto-fire preview). The operator approves a subset. This route then, as
    the CENTRAL SECOND GUARD:
      (1) re-derives the CURRENT proposal from live state (worker_storage_view —
          the same helper _public_view uses, off the RAW record), because state
          may have changed since the render;
      (2) INTERSECTS the approved keys with the freshly-proposed set, dropping
          anything that since became loaded/assigned/pinned/provisioning or is no
          longer proposed (over budget cleared);
      (3) delegates the survivors to the SAME guarded reaper relay ``workers_reap``
          uses (POST /reap) — where the worker's ``_reap_reclaim`` re-proves EVERY
          guard per key at delete time and ``wipe_model`` is path-jailed.
    This BULK reclaim path deletes ONLY through this explicit, operator-gated,
    audited call: no background monitoring, no timer, and no auto-approval —
    central never widens or self-approves an operator's approved set.

    SCOPE NOTE (amended 2026-07-16, storage-budget incident — op filled to 0
    bytes free): this docstring used to open "NOTHING deletes except through
    this call". That is no longer true FLEET-WIDE, and saying so would be a lie
    (this codebase has a history of docs asserting behavior the code no longer
    has), so the claim is now scoped to THIS route. A second, deliberately
    NARROW delete path exists worker-side: ``worker_agent/budget.py``
    (evict_to_fit) FIFO-evicts cold, unprotected models to make room for a model
    that is ACTIVELY BEING PROVISIONED — the operator's rule "remove an existing
    model and install the one that is being called". It is call-driven ONLY
    (no call -> no delete), never a sweep or a timer, evicts the minimum needed,
    and refuses the pull outright rather than over-delete. It changes NOTHING
    about this route: the bulk operator-gated flow above still behaves exactly
    as before, and BOTH paths funnel into the same single guarded delete choke
    point (``_reap_reclaim`` -> ``wipe_model``), so neither can delete anything
    the other's guards would protect.

    Body: ``{"model_keys": [...approved keys the console rendered...]}``.
    Returns the reaper's typed result (freed_bytes + per-key results) plus the
    central ``approved``/``reaped``/``dropped`` key sets so the console can show
    both what the worker deleted-vs-skipped AND what central filtered pre-relay.
    """
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    body = request.get_json(silent=True) or {}
    approved = body.get("model_keys")
    if not isinstance(approved, list) or not approved:
        abort(400, description='body must include a non-empty {"model_keys": [...]}')
    approved_keys = [str(k) for k in approved if k]

    # The recompute → intersect → audit → guarded relay core is SHARED with the
    # heartbeat auto-fire (_execute_reap) so the two can never diverge; here the
    # trigger is "operator" (hand-approved), which only changes the audit event.
    data, status = _execute_reap(worker_id, worker, approved_keys,
                                 trigger="operator")
    return jsonify(data), status


@worker_bp.route("/llm/workers/<worker_id>/update", methods=["POST"])
def workers_update(worker_id):
    """CON-05: trigger the worker's module self-update NOW (same converge
    path as the heartbeat's required_pkg_version handshake, minus the wait).
    Body: {"version": "..."} to pin; default = central's required version.
    Confirm afterward via the worker registry's pkg_version."""
    return _relay_worker_op(worker_id, "/ops/update",
                            request.get_json(silent=True) or {},
                            timeout=30.0, action="update")


@worker_bp.route("/llm/workers/<worker_id>/pip", methods=["POST"])
def workers_pip(worker_id):
    """UTIL-02: pip install into the worker's env, through the operator gate.
    Body: {"package": "name==ver"}. The gate + audit ARE the feature; the
    install is trivial. Long timeout — pip resolves are slow."""
    body = request.get_json(silent=True) or {}
    pkg = str(body.get("package") or "").strip()
    if not pkg:
        abort(400, description='body must include {"package": "..."}')
    return _relay_worker_op(worker_id, "/ops/pip", {"package": pkg},
                            timeout=600.0, action="pip")


@worker_bp.route("/llm/workers/<worker_id>/probe", methods=["POST"])
def workers_probe(worker_id):
    """Live VRAM-fit probe: ask the worker to load the model and report fit.

    Body: {"model_key": ...}. Relays to the worker's /probe, which loads the
    model on its GPU and returns {fit, vram_free_before/after, vram_used}.
    """
    import httpx

    body = AssignRequest(**(request.get_json(silent=True) or {}))
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    url = (worker.get("url") or "").rstrip("/") + "/probe/" + body.model_key
    try:
        # Loading can be slow (download + load), so allow generous time.
        r = httpx.post(url, timeout=900.0)
        return jsonify(r.json())
    except Exception as exc:
        return jsonify({"ok": False, "fit": False,
                        "error": f"{type(exc).__name__}: {exc}"})


# ── worker slots: VRAM-fit preflight + load (the GPU analog of /llm/slots) ──
#
# The local slot pool refuses a model that won't fit RAM before it OOMs the box
# (slot_agent._build_cmd preflight). Worker slots do the same against a worker's
# free VRAM: a cheap, instant central-side reject so a too-big model never even
# reaches the card. The flat vram_free now on every worker record (_vram_summary)
# is what makes this a one-line comparison instead of a heavy live probe.

# VRAM need over the raw GGUF size: weights resident on the GPU plus kv-cache /
# context and CUDA runtime overhead. A need-side multiplier (overhead scales with
# the model), the VRAM counterpart of the local preflight's 0.95 fill guard.
VRAM_HEADROOM = float(os.environ.get("HUGPY_VRAM_HEADROOM", "1.15"))


def _model_gguf_bytes(model_key):
    """Total on-disk GGUF size for a model (sums all shards), or None if unknown.

    ``_model_file_for`` and ``_total_gguf_bytes`` are underscore-private, so they
    are NOT pulled in by ``from .imports import *`` — import them explicitly here
    (a bare reference would NameError and, caught below, silently read as None)."""
    try:
        from ....managers.serve.serve import _model_file_for
        from ....managers.serve.slot_agent import _total_gguf_bytes
        src = _model_file_for(model_key, get_model_config(model_key))
    except Exception:
        return None
    if not src:
        return None
    try:
        return _total_gguf_bytes(src)
    except Exception:
        return None


def _worker_fit(model_key, worker):
    """Capacity preflight for placing a model on a worker — the GPU analog of the
    local RAM preflight, but DUAL. A GPU worker holds weights in VRAM and can
    spill the remainder to host RAM, so a load only truly *fails* when the model
    exceeds BOTH combined — that's what we block on. Separately we flag whether it
    fits VRAM outright (``gpu_resident``) vs would partially offload to CPU
    (slower), so the UI can warn without refusing.

    Why dual: a worker like an 8 GB card on a 16 GB box can't be judged on VRAM
    alone — a model bigger than VRAM may still load (spilling to RAM), and one
    bigger than VRAM+RAM can't load at all. ``fit=None`` when the model can't be
    sized (then we don't block — defer to the live load). All byte counts."""
    need_raw = _model_gguf_bytes(model_key)
    vram = worker.get("vram_free")
    ram = worker.get("free_ram")
    gib = float(2 ** 30)
    if need_raw is None:
        return {"fit": None, "gpu_resident": None, "need": None, "need_raw": None,
                "vram_free": vram, "ram_free": ram, "reason": "model size unknown — not preflighted"}
    need = int(need_raw * VRAM_HEADROOM)
    # t28 load-and-learn: refine the VRAM-residency estimate with the learned,
    # per-model correction (median measured/predicted from real loads), clamped +
    # gated central-side. Applied to `need` (drives gpu_resident + the human hint)
    # so the preflight agrees with the worker's own corrected admission; the hard
    # combined-capacity block below stays on the raw size, so calibration can
    # only make the residency hint MORE accurate, never invent a new refusal.
    calibration_correction = None
    try:
        from abstract_hugpy_dev.comms.calibration import calibration_store as _cal
        _c = _cal.correction_for(model_key)
        if _c:
            need = int(need * float(_c))
            calibration_correction = float(_c)
    except Exception:  # noqa: BLE001 — learned pricing is additive; never break fit
        calibration_correction = None
    capacity = (vram or 0) + (ram or 0)
    fit = (need_raw <= capacity) if capacity else None
    gpu_resident = (vram is not None) and (need <= vram)
    where = worker.get("gpu") or "this worker"
    # ── t21 central mirror: a model carrying an explicit VRAM tolerance band may,
    # UNDER CONTENTION, be seated at its band FLOOR (a smaller gpu_mem_gib =
    # fewer GPU layers, more CPU spill) rather than its target. So the feasibility
    # math accepts the band floor as admissible even when the target doesn't fit
    # free VRAM. Pure read of central's spill registry via the ONE band-math seam;
    # additive fields, never changes the base fit/gpu_resident verdict.
    band_floor_bytes = None
    band_floor_admissible = None
    spill = (worker.get("spill_by_model") or {}).get(model_key) or {}
    gpu_target_gib = spill.get("gpu_mem_gib")
    gpu_dev = spill.get("gpu_mem_gib_deviation_pct")
    vram_total = worker.get("vram_total")
    if gpu_target_gib is not None and gpu_dev and vram_total:
        try:
            from ....worker_agent.flex import band_floor as _band_floor
            band_floor_bytes = int(_band_floor(
                float(gpu_target_gib) * gib, gpu_dev, vram_total))
            if vram is not None:
                band_floor_admissible = band_floor_bytes <= vram
        except Exception:  # noqa: BLE001 — band math is additive; never break fit
            band_floor_bytes = None
    # ── partial-offload mirror (t21 stage 2.5): the worker now DEGRADES an
    # oversize GGUF to an honest hybrid (offload the layers that fit, stream the
    # rest to CPU RAM) instead of hard-refusing. Central's feasibility must AGREE:
    # a model that fits combined VRAM+RAM but not VRAM outright is admissible as a
    # partial offload. Additive boolean; never changes the base fit/gpu_resident
    # verdict (the human `reason` for this case already reads "would partially
    # offload to CPU"). The worker makes the real, geometry-aware call at load.
    partial_offload_admissible = bool(fit and not gpu_resident)
    if fit is False:
        reason = (f"won't fit {where}: model is {need_raw/gib:.1f} GiB but only "
                  f"{(vram or 0)/gib:.1f} GiB VRAM + {(ram or 0)/gib:.1f} GiB RAM free "
                  f"({capacity/gib:.1f} GiB total)")
    elif fit and not gpu_resident and band_floor_admissible:
        reason = (f"fits GPU-resident at its VRAM band floor ({band_floor_bytes/gib:.1f} "
                  f"GiB) under contention — target would offload to CPU; "
                  f"{(vram or 0)/gib:.1f} GiB VRAM free on {where}")
    elif fit and not gpu_resident:
        reason = (f"fits but would partially offload to CPU: needs ~{need/gib:.1f} GiB, "
                  f"only {(vram or 0)/gib:.1f} GiB VRAM free on {where} — slower than GPU-resident")
    else:
        reason = None
    return {"fit": fit, "gpu_resident": gpu_resident, "need": need, "need_raw": need_raw,
            "vram_free": vram, "ram_free": ram, "capacity": capacity,
            "headroom": VRAM_HEADROOM, "reason": reason,
            "calibration_correction": calibration_correction,
            "band_floor_bytes": band_floor_bytes,
            "band_floor_admissible": band_floor_admissible,
            "partial_offload_admissible": partial_offload_admissible}


def _worker_already_has(worker: dict, model_key: str) -> bool:
    """Whether ``worker`` already holds ``model_key`` in any serveable sense —
    operator-assigned (``models``), heartbeat-resident (``loaded_models``), or a
    SYSTEM placement grant (``grants``). Mirrors the union ``workers_for_model``
    uses to decide "can this worker serve it" (workers.py), so the placement
    preview's "already_has" reads consistently with real routing. Read-only:
    just a membership check over fields already on the worker record."""
    if model_key in (worker.get("models") or []):
        return True
    if model_key in (worker.get("loaded_models") or []):
        return True
    if model_key in (worker.get("grants") or {}):
        return True
    return False


@worker_bp.route("/llm/models/<path:model_key>/block", methods=["POST"])
def model_block(model_key):
    """Operator BLOCK a model from the SERVING POOL (global).

    A blocked model is never routed to, never (re)designated/assigned, never
    warmed/provisioned by a sweep, and never resolved by a fallback ladder — its
    files stay on disk and its existing designation rows stay recorded (inert).
    Block is a ROUTING override that outranks pin (pin is routing persistence;
    block is an operator override) — this does NOT auto-unassign and does NOT
    fight the pin's unassign-409. Reversible via /unblock.

    Operator-gated in operator_auth._SENSITIVE (same tier as assign). Body
    (optional): {"note": str}. Idempotent; returns the block record + the
    manifest membership so the console can reflect state immediately."""
    if model_key not in get_models_dict(dict_return=True):
        return jsonify({"error": f"unknown model key '{model_key}' — it is not in "
                        "central's manifest"}), 404
    body = request.get_json(silent=True) or {}
    from abstract_hugpy_dev.comms.blocklist import block as _block
    rec = _block(model_key, by="operator", note=body.get("note"))
    try:
        from .comms_routes import audit
        audit("model.block", {"model_key": model_key, "note": body.get("note")})
    except Exception:  # noqa: BLE001 — audit is best-effort, never fatal
        pass
    return jsonify({"ok": True, "model_key": model_key, "blocked": True,
                    "block": rec})


@worker_bp.route("/llm/models/<path:model_key>/unblock", methods=["POST"])
def model_unblock(model_key):
    """Operator UNBLOCK a model — return it to the serving pool. The undo for
    /block. Idempotent (unblocking a non-blocked model is a no-op that reports
    ``was_blocked: false``). Operator-gated like /block."""
    if model_key not in get_models_dict(dict_return=True):
        return jsonify({"error": f"unknown model key '{model_key}' — it is not in "
                        "central's manifest"}), 404
    from abstract_hugpy_dev.comms.blocklist import unblock as _unblock
    was = _unblock(model_key)
    try:
        from .comms_routes import audit
        audit("model.unblock", {"model_key": model_key, "was_blocked": was})
    except Exception:  # noqa: BLE001
        pass
    return jsonify({"ok": True, "model_key": model_key, "blocked": False,
                    "was_blocked": was})


@worker_bp.route("/llm/models/<path:model_key>/placement", methods=["GET"])
def model_placement_preview(model_key):
    """OBSERVE-ONLY placement feasibility preview (Phase 1 item 3).

    Answers "if a FLOATING call for ``model_key`` arrived with no worker
    assigned, which workers could feasibly hold it, and which would win?" —
    pure read, proves the fit/feasibility logic before any auto-placement is
    built. Takes NO action: no assignment, no grant, no queue mutation, no
    probe. Reuses ``_worker_fit`` and ``_disk_preflight_reason`` VERBATIM —
    this endpoint does not compute sizing or fit itself, only surfaces it.

    Gated like the other fleet-capacity/file-adjacent reads in this module
    (``_transfer_authorized``): operator token OR a valid worker enrollment
    bearer — the least-privileged gate that still fits a read of per-worker
    VRAM/RAM/disk capacity.
    """
    if not _transfer_authorized():
        abort(401, description="Worker enrollment or operator token required.")

    # Operator BLOCK: report "blocked" as the honest winner_reason rather than
    # walking the fleet and claiming fake-infeasibility. Additive `blocked:true`
    # so the console can render the distinct state; no worker is a candidate.
    if _model_blocked(model_key):
        return jsonify({
            "model_key": model_key, "size_bytes": None, "workers": [],
            "feasible_workers": [], "winner": None,
            "winner_reason": (f"'{model_key}' is blocked from the serving pool by "
                              "the operator — unblock it to place it"),
            "blocked": True,
        })

    size_bytes = None
    try:
        size_bytes = _model_gguf_bytes(model_key)
    except Exception:  # noqa: BLE001 — degrade, never 500 the whole preview
        size_bytes = None

    workers_out = []
    feasible_online = []   # (worker_dict, verdict, already_has) candidates for winner
    for worker in (list_workers() or []):
        wid = worker.get("id")
        name = worker.get("name") or wid
        online = (worker.get("status") == "online")
        already_has = _worker_already_has(worker, model_key)

        fit_errored = False
        try:
            verdict = _worker_fit(model_key, worker)
        except Exception as exc:  # noqa: BLE001 — one bad worker must not sink the preview
            fit_errored = True
            verdict = {"fit": None, "gpu_resident": None, "vram_free": worker.get("vram_free"),
                       "ram_free": worker.get("free_ram"),
                       "reason": f"size unknown ({type(exc).__name__}: {exc})"}

        try:
            disk_reason = _disk_preflight_reason(worker, model_key)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully per worker
            disk_reason = f"disk check failed ({type(exc).__name__}: {exc})"
        disk_ok = disk_reason is None

        fit = verdict.get("fit")
        if fit_errored:
            # A genuine failure to evaluate this worker — never claim feasible.
            feasible = None
        else:
            feasible = (fit is not False)   # None (unsizable) treated as feasible/defer

        workers_out.append({
            "id": wid,
            "name": name,
            "feasible": feasible,
            "gpu_resident": verdict.get("gpu_resident"),
            "vram_free": verdict.get("vram_free"),
            "ram_free": verdict.get("ram_free"),
            "reason": verdict.get("reason"),
            "disk_ok": disk_ok,
            "disk_reason": disk_reason,
            "online": online,
            "already_has": already_has,
        })

        if online and feasible and disk_ok:
            feasible_online.append((worker, verdict, already_has))

    feasible_names = [w.get("name") or w.get("id") for w, _v, _a in feasible_online]

    winner = None
    winner_reason = None
    if not feasible_online:
        if not any(w["online"] for w in workers_out):
            winner_reason = "no online worker"
        elif size_bytes is not None:
            winner_reason = f"no online worker can fit {size_bytes / (2**30):.1f} GiB"
        else:
            winner_reason = "no online worker passes disk/fit preflight"
    else:
        # Policy (observe-only — mirrors what a real scheduler would pick,
        # commits nothing): 1) already holds it, 2) GPU-resident outright over
        # spill-only, 3) most vram_free.
        def _rank(item):
            _w, v, already = item
            return (
                0 if already else 1,
                0 if v.get("gpu_resident") else 1,
                -(v.get("vram_free") or 0),
            )
        feasible_online.sort(key=_rank)
        best_w, best_v, best_already = feasible_online[0]
        winner = best_w.get("name") or best_w.get("id")
        if best_already:
            winner_reason = "already holds the model"
        elif best_v.get("gpu_resident"):
            winner_reason = "fits VRAM outright with the most free VRAM among candidates"
        else:
            winner_reason = "best available fit (spill to RAM) among candidates"

    return jsonify({
        "model_key": model_key,
        "size_bytes": size_bytes,
        "workers": workers_out,
        "feasible_workers": feasible_names,
        "winner": winner,
        "winner_reason": winner_reason,
    })


@worker_bp.route("/llm/workers/<worker_id>/load", methods=["POST"])
def workers_load(worker_id):
    """Place a model on a GPU worker: VRAM preflight → assign → load into VRAM.

    The worker-slot analog of /llm/slots/load. Refuses cleanly (409) when the
    model won't fit the worker's free VRAM, instead of letting it OOM the card.
    On a pass it assigns the model (registry) and kicks a best-effort warm on the
    worker so it becomes GPU-resident. The warm is BACKGROUND: loading a model can
    take minutes (the worker's /probe loads it synchronously), and the request
    must not block — the local slot loader is async for the same reason. The UI
    reflects residency from the worker's heartbeat (loaded_models), polling
    /llm/workers; if the warm fails, the next inference lazy-loads it anyway.

    Body: {model_key, spill?, force?, redownload?}. force=true skips the preflight
    (still bounded by the worker's own limits). redownload=true first wipes the
    model's files on the worker and re-pulls them from central before warming —
    for a corrupt/stale on-disk copy (a plain load only downloads when MISSING)."""
    import httpx
    import threading

    raw = request.get_json(silent=True) or {}
    body = AssignRequest(**raw)
    force = bool(raw.get("force"))
    redownload = bool(raw.get("redownload"))
    if body.model_key not in get_models_dict(dict_return=True):
        # JSON (not abort's HTML) so allocateMany's refusal note is a clean line,
        # never a raw <!doctype> 404 page dumped into the UI.
        return jsonify({"error": f"unknown model key '{body.model_key}' — it is "
                        "not in central's manifest"}), 404
    # Item 4 guard — same invariant as /assign: central must hold the files.
    missing = _central_missing_reason(body.model_key)
    if missing:
        return jsonify({"error": f"central does not have '{body.model_key}' on "
                        f"disk ({missing}) — download it on the Models tab first; "
                        "workers provision from central"}), 409
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    # Disk-aware allocation — refuse a pull the worker's model volume can't
    # hold (force does NOT bypass this: a full disk mid-pull helps nobody).
    disk_no = _disk_preflight_reason(worker, body.model_key)
    if disk_no:
        return jsonify({"error": f"'{body.model_key}' won't fit on "
                        f"{worker.get('name') or worker_id}: {disk_no} — free "
                        "space or pick another worker"}), 409

    verdict = _worker_fit(body.model_key, worker)
    if verdict.get("fit") is False and not force:
        # `error` so the shared fetchJson surfaces the human reason (it only reads
        # error/detail/message); `reason`+`preflight` kept for programmatic use.
        return jsonify({"loaded": False, "error": verdict["reason"],
                        "reason": verdict["reason"], "preflight": verdict}), 409

    # passed (or forced/undecided) → assign, then warm in the background
    assign_model(worker_id, body.model_key, spill=body.spill)
    base = (worker.get("url") or "").rstrip("/")
    url = base + "/probe/" + body.model_key

    def _warm():
        # Best-effort warm, but NEVER silent: the probe outcome (the worker's
        # {ok, fit, error, …} — or the transport failure reaching it) lands in
        # the registry as load_reports[model_key], where the console shows WHY
        # a model stayed cold. Before this, every failure mode here looked
        # identical to success ("activate does nothing").
        import time as _time
        from ..functions.imports.utils.workers import set_load_report
        report: dict = {"ts": _time.time()}
        try:
            if redownload:
                # Wipe the model's files on the worker + re-pull from central BEFORE
                # warming. A full download can take minutes, so it rides this same
                # background thread — the request never blocks.
                httpx.post(base + "/models/redownload",
                           json={"model_key": body.model_key}, timeout=3600.0)
            r = httpx.post(url, timeout=900.0)  # worker loads synchronously; can be slow
            try:
                report.update(r.json())
            except Exception:
                report.update(ok=r.is_success,
                              error=None if r.is_success else f"HTTP {r.status_code}")
        except Exception as exc:
            report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        try:
            set_load_report(worker_id, body.model_key, report)
        except Exception:
            logger.exception("warm: could not record load report for %s on %s",
                             body.model_key, worker_id)

    threading.Thread(target=_warm, name=f"warm-{worker_id[:8]}", daemon=True).start()
    note = ("redownloading (wipe + re-pull from central) then warming on the worker"
            if redownload else
            "assigned + warming on the worker — watch loaded_models for residency")
    return jsonify({"loaded": "loading", "assigned": True, "redownload": redownload,
                    "preflight": verdict, "worker": get_worker(worker_id), "note": note})


# ──────────────────────────────────────────────────────────────────────────
# Package distribution — central serves the dev wheel as a PEP-503 simple index.
#
# The sync.trigger build drops freshly-built wheels into pkg_index_dir(); workers
# self-update via `pip install --index-url https://<central>/api/llm/pip/simple
# <pkg>==<version>`. This reuses the one channel every worker already reaches
# outbound (the same node it heartbeats to), so external/WireGuard workers need
# no PyPI access and nothing extra exposed.
# ──────────────────────────────────────────────────────────────────────────
_PKG_NORMALIZE = re.compile(r"[-_.]+")


def _normalize_project(name: str) -> str:
    """PEP 503 normalized project name (lowercase, runs of -_. -> single -)."""
    return _PKG_NORMALIZE.sub("-", name).lower()


@worker_bp.route("/llm/pip/simple/<project>/", methods=["GET"])
def pip_simple_project(project):
    """PEP-503 page for one project: links to every matching wheel/sdist."""
    norm = _normalize_project(project)
    d = pkg_index_dir()
    links = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if not (fn.endswith(".whl") or fn.endswith(".tar.gz")):
                continue
            # Distribution name is the segment before the first '-' in the
            # filename (e.g. abstract_hugpy_dev-0.1.401.dev1-...whl).
            dist = fn.split("-", 1)[0]
            if _normalize_project(dist) == norm:
                links.append(fn)
    body = ("<!DOCTYPE html><html><head>"
            f"<meta name=\"pypi:repository-version\" content=\"1.0\"></head><body>\n"
            + "\n".join(f'<a href="{fn}">{fn}</a><br>' for fn in links)
            + "\n</body></html>\n")
    return Response(body, mimetype="text/html")


@worker_bp.route("/llm/pip/simple/<project>/<path:filename>", methods=["GET"])
def pip_simple_file(project, filename):
    """Stream one wheel/sdist from the index dir, path-confined."""
    d = os.path.realpath(pkg_index_dir())
    target = os.path.realpath(os.path.join(d, filename))
    if target != d and not target.startswith(d + os.sep):
        abort(403, description="Path escapes index directory.")
    if not os.path.isfile(target):
        abort(404, description="No such distribution file.")
    return send_file(target, as_attachment=True,
                     download_name=os.path.basename(target), conditional=True)


# ──────────────────────────────────────────────────────────────────────────
# Model provisioning — workers pull missing model files from central.
#
# A worker that lacks a model calls /manifest to learn the file list + the
# routing metadata (framework/task/hub_id) it needs to place the files under
# its OWN storage root, then GETs each file via /file. Streaming with send_file
# means large GGUF/safetensors transfers don't buffer in memory and support
# HTTP Range (resumable). Both routes are read-only and confined to the model's
# own destination directory.
# ──────────────────────────────────────────────────────────────────────────
def _resolve_manifest_key(manifest: dict, model_key: str) -> str | None:
    """Resolve a requested key to a manifest key, tolerantly.

    Workers reference a model by its hub_id (``Qwen/Qwen2.5-Coder-3B-Instruct-GGUF``)
    while the manifest is keyed by the short name (``Qwen2.5-Coder-3B-Instruct-GGUF``)
    and carries the hub_id as a field. Match on the registry key, ``hub_id``,
    ``name``, or the trailing path segment — case-insensitively — so either
    spelling resolves.
    """
    if model_key in manifest:
        return model_key
    want = {model_key, model_key.lower(),
            model_key.split("/")[-1], model_key.split("/")[-1].lower()}
    for key, row in manifest.items():
        if not isinstance(row, dict):
            continue
        forms = {key, row.get("hub_id"), row.get("key"), row.get("name")}
        forms = {str(f) for f in forms if f}
        forms |= {f.lower() for f in forms}
        forms |= {f.split("/")[-1] for f in forms}
        forms |= {f.split("/")[-1].lower() for f in forms}
        if want & forms:
            return key
    return None


def _model_dir_or_404(model_key: str):
    manifest = get_models_dict(dict_return=True)
    key = _resolve_manifest_key(manifest, model_key)
    if key is None:
        abort(404, description="Unknown model key.")
    model = manifest[key]
    dest = route_destination(model)
    if not os.path.isdir(dest):
        abort(409, description="Model is not installed on central.")
    return model, os.path.realpath(dest)


# ── central storage-budget gate on BACKGROUND transfers (ae 1.2TB 2026-07-17) ──
# Operator ruling: "its central that distributed these downloads, it has autonomy
# over this... it simply needs to abide by the limits set within its own backend."
# Central refuses a BACKGROUND pull (purpose reconcile/assign) from a worker whose
# ledger says resident+incoming would exceed its budget. DEMAND pulls (a called
# model) are NEVER budget-refused here — the worker's own fit_plan evict-to-fit
# seats them; refusing a called model centrally would break serving. A request
# with NO purpose header (an old agent mid-convergence) is treated as DEMAND —
# permissive during rollout; this leniency EXPIRES once the fleet converges on
# the purpose-aware agent (2026-07-17 release).
_TRANSFER_PURPOSE_HEADER = "X-Transfer-Purpose"
_TRANSFER_WORKER_HEADER = "X-Worker-Id"
# Background purposes = the budget-refusable class. Everything else (demand,
# probe, or an absent header) is served.
_BACKGROUND_TRANSFER_PURPOSES = frozenset({"reconcile", "assign"})


def _budget_refusal_for_transfer(model, incoming_bytes):
    """Return a machine-readable 409 reason dict if this transfer REQUEST must be
    refused for the declaring worker's storage budget, else None.

    Reads the purpose + worker id off the request headers. Only BACKGROUND pulls
    are candidates for refusal; demand / probe / no-header are always served
    (None). Uses the SAME budget/resident numbers storage_proposal already
    computes (via worker_storage_view) — no second accounting path. Refuses only
    when the worker is KNOWN, has a real budget, and resident+incoming > budget.
    Any ambiguity (unknown worker, no budget, unknown incoming size) -> serve.
    """
    purpose = (request.headers.get(_TRANSFER_PURPOSE_HEADER) or "").strip().lower()
    if purpose not in _BACKGROUND_TRANSFER_PURPOSES:
        return None  # demand / probe / absent -> never budget-refused centrally
    worker_id = (request.headers.get(_TRANSFER_WORKER_HEADER) or "").strip()
    if not worker_id:
        return None  # background but anonymous (shouldn't happen) -> serve
    try:
        from ..functions.imports.utils.workers import worker_storage_view
        view = worker_storage_view(worker_id)
    except Exception:  # noqa: BLE001 — accounting failure must not block serving
        return None
    if not isinstance(view, dict):
        return None  # unknown worker -> serve (don't refuse what we can't size)
    budget = view.get("budget")
    resident = view.get("resident_bytes")
    if budget in (None, "") or resident is None:
        return None  # no managed budget -> serve (unmanaged = pre-feature behavior)
    try:
        budget = int(budget)
        resident = int(resident)
        incoming = int(incoming_bytes or 0)
    except (TypeError, ValueError):
        return None
    # If the model is already (partly) resident, its bytes are counted in
    # `resident`; a background top-up of an already-present model is a no-op the
    # worker's file-resume handles, so only refuse when the FULL incoming size on
    # top of current residency would bust the budget. Conservative: this is the
    # same "resident + need" the worker's own budget check uses.
    if resident + incoming <= budget:
        return None
    return {
        "code": "storage_budget_exceeded",
        "purpose": purpose,
        "worker_id": worker_id,
        "model_key": (model.get("key") or model.get("name")
                      if isinstance(model, dict) else None),
        "budget_bytes": budget,
        "resident_bytes": resident,
        "incoming_bytes": incoming,
        "would_use_bytes": resident + incoming,
        "reason": (f"background pull ({purpose}) refused: resident {resident} + "
                   f"incoming {incoming} = {resident + incoming} bytes would "
                   f"exceed this worker's storage budget of {budget} bytes. "
                   "Central abides by the limits set within its own backend "
                   "(2026-07-17). The model pulls on a real call (demand), which "
                   "the worker's own evict-to-fit seats."),
    }


@worker_bp.route("/llm/models/<path:model_key>/manifest", methods=["GET"])
def model_file_manifest(model_key):
    if not _transfer_authorized():
        abort(401, description="Worker enrollment or operator token required.")
    model, dest = _model_dir_or_404(model_key)

    # Walk the whole model dir first (skipping transfer-machinery artifacts —
    # chunk-hash sidecars, .part/.state staging, and dot-directories like
    # .cache/.git, see walk_listing's docstring), then SINGLE-FORMAT filter it.
    # Shared with /archive via format_select.walk_listing — one walk, one skip
    # list, not a hand-copied mirror.
    from ..functions.imports.utils.format_select import select_files, walk_listing
    raw = walk_listing(dest)       # [(rel, size)]
    raw_total = sum(s for (_r, s) in raw)

    # Central mirrors WHOLE HF snapshots — the same weights in several formats,
    # often an fp32 duplicate too. A worker only needs ONE usable weight format +
    # the sidecars. Offer exactly that (degrade-to-correct: an unrecognized layout
    # falls back to the whole listing). GGUF is untouched — its effective quant is
    # resolved elsewhere. The worker's per-file puller drives off this `files`
    # list, so this is what actually lands on the worker's disk.
    framework = model.get("framework")
    selected = select_files(raw, framework=framework)
    files = [{"path": r, "size": s} for (r, s) in selected]
    total = sum(s for (_r, s) in selected)

    # BUDGET GATE (2026-07-17): the manifest is the FIRST thing both the per-file
    # and the archive transports fetch, so refusing a background-over-budget pull
    # HERE stops it before a single weight byte moves — on either transport. The
    # single-format `total` above is exactly the incoming size the worker would
    # land. Demand/probe/no-purpose pass through untouched (see helper).
    refusal = _budget_refusal_for_transfer(model, total)
    if refusal is not None:
        logger.info("transfer of %s REFUSED (budget): %s", model_key,
                    refusal.get("reason"))
        return jsonify({"error": refusal["reason"], **refusal}), 409

    return jsonify({
        "model_key": model_key,
        "hub_id": model.get("hub_id"),
        "name": model.get("name"),
        "framework": framework,
        "task": model.get("task") or model.get("primary_task"),
        "filename": model.get("filename"),
        "include": model.get("include"),
        "total_bytes": total,          # single-format effective size
        "files": files,                # single usable format + sidecars
        # Whole-snapshot footprint, for diagnostics only (never the transfer set).
        "dir_total_bytes": raw_total,
        "dir_file_count": len(raw),
    })


# Fixed streaming window for ranged serving: seek once, then copy the requested
# bytes in 1 MiB reads. This is the DoS fix. werkzeug's send_file(conditional=True)
# reaches a Range's start offset by CONSUMING the response iterable chunk-by-chunk
# (``_RangeWrapper._first_iteration``: ``while read_length <= start_byte:
# _next_chunk()``) — O(offset) per request, O(offset^2) across a chunked multi-GB
# pull. A handful of large-offset Range GETs then peg all of central's CPU: a
# trivial remote DoS on an internet-facing origin. A real ``file.seek(start)`` is
# O(1) to any offset and reads only the requested window.
_FILE_STREAM_CHUNK = 1024 * 1024  # 1 MiB

# Single-range header: "bytes=<first>-<last>" with either side optionally empty.
_RANGE_RE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.IGNORECASE)


def _parse_single_range(range_header: str, size: int):
    """Interpret a ``Range`` header against a file of ``size`` bytes.

    Returns one of:
      * ``(start, end)`` — inclusive offsets of a satisfiable single range (206);
      * ``None``          — serve the WHOLE file (200): no/blank/unparseable
        header, or a multi-range request we deliberately answer whole rather than
        emit ``multipart/byteranges`` (the worker puller only asks single ranges);
      * ``"unsatisfiable"`` — well-formed but unmeetable range (416).

    Never loops or scans — pure arithmetic on the offsets.
    """
    if not range_header:
        return None
    # Multi-range ("bytes=0-1,4-5"): don't build multipart — hand back the whole
    # file (200). Correct and cheap; crucially it never spins.
    if "," in range_header:
        return None
    m = _RANGE_RE.match(range_header)
    if not m:
        return None  # malformed -> ignore Range, serve full (200)
    first, last = m.group(1), m.group(2)
    if first == "" and last == "":
        return None  # "bytes=-" is malformed -> serve full
    if first == "":
        # Suffix range: "bytes=-N" -> the final N bytes. N==0 is unsatisfiable.
        n = int(last)
        if n <= 0:
            return "unsatisfiable"
        start = max(0, size - n)
        end = size - 1
    else:
        start = int(first)
        end = int(last) if last != "" else size - 1
        if end >= size:
            end = size - 1  # clamp an over-long end to the last byte (RFC 7233)
    if size == 0 or start >= size or start > end:
        return "unsatisfiable"
    return (start, end)


def _stream_file_window(path: str, start: int, end: int):
    """Yield ``path[start..end]`` inclusive in fixed 1 MiB reads after a single
    ``seek`` — O(1) to reach ``start``, and reads ONLY the requested window."""
    remaining = end - start + 1
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            buf = fh.read(min(_FILE_STREAM_CHUNK, remaining))
            if not buf:
                break  # file truncated under us; stop rather than loop
            remaining -= len(buf)
            yield buf


# ── global central transfer cap (the "pin-30 survives" protection) ──────────
# Each puller (provision.py) opens up to HUGPY_PULL_CONCURRENCY (default 8)
# concurrent segmented Range-GETs PER WORKER. A big assignment/pin or a
# reconcile storm across K workers therefore lands up to ~8*K simultaneous
# byte streams on central's two weight-serving endpoints (/file and
# /archive) with NO shared bound between them — a handful of pins was enough
# to saturate central's link/CPU (the live 2026-07-15 incident). This is a
# GLOBAL bound across both endpoints, independent of per-worker concurrency.
#
# The cap is read ONCE at import time into a BoundedSemaphore, same pattern
# as _warm_lock/_warm_busy above: a fixed module-level guard, not re-read
# per request.
def _transfer_cap() -> int:
    raw = os.environ.get("HUGPY_CENTRAL_TRANSFER_MAX")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 3
    return n if n >= 1 else 3


_TRANSFER_CAP = _transfer_cap()          # fixed at module load, see above
_transfer_sem = _threading.BoundedSemaphore(_TRANSFER_CAP)

# How long a request waits for a free permit before giving up and telling the
# worker to back off. Workers already retry failed segments
# (provision.py:_download_segment_with_retry), so a prompt 503 is safe,
# cheap backpressure — far better than blocking a gunicorn worker thread
# indefinitely (which just turns the transfer cap into a request-thread
# exhaustion bug instead of a fix).
_TRANSFER_WAIT_S = float(os.environ.get("HUGPY_CENTRAL_TRANSFER_WAIT_S", "5"))


def _transfer_busy_response() -> Response:
    """503 + Retry-After for a request that couldn't get a transfer permit."""
    resp = Response(
        "Central weight-transfer capacity is saturated; retry shortly.",
        status=503,
        mimetype="text/plain",
    )
    resp.headers["Retry-After"] = "2"
    return resp


class _TransferPermit:
    """One acquired slot on ``_transfer_sem``, released EXACTLY ONCE.

    The tricky part of this cap is lifetime: both endpoints below return a
    streaming Response/generator, so the route FUNCTION returns long before
    the bytes finish flowing. Acquiring in the handler and releasing with a
    plain ``with _transfer_sem:`` around the handler body would release the
    permit the instant the Response object is constructed — i.e. before a
    single byte reaches the worker — which caps nothing. The permit must
    therefore be released when the STREAM ends: normal completion, a client
    disconnect, or a write error — never on handler return.

    This wrapper makes that release idempotent (BoundedSemaphore.release()
    raises if called more times than acquire(), so every call site — a
    generator's ``finally`` and a wrapped-iterable ``close()`` (see
    ``_ReleaseOnCloseIter`` below) — can all call ``.release()``
    unconditionally without risking a double-release).
    """

    __slots__ = ("_sem", "_released", "_lock")

    def __init__(self, sem: "_threading.BoundedSemaphore"):
        self._sem = sem
        self._released = False
        self._lock = _threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._sem.release()


def _acquire_transfer_permit() -> "_TransferPermit | None":
    """Try to reserve one of the global transfer slots.

    Returns a ``_TransferPermit`` to release when the stream ends, or
    ``None`` if none became free within ``_TRANSFER_WAIT_S`` (caller should
    respond 503). Blocking with a bounded timeout — never forever — so a
    saturated central sheds load instead of piling up stuck request threads.
    """
    if _transfer_sem.acquire(blocking=True, timeout=_TRANSFER_WAIT_S):
        return _TransferPermit(_transfer_sem)
    return None


class _ReleaseOnCloseIter:
    """Wrap a WSGI response iterable so ``.close()`` also releases a transfer
    permit — for the ``send_file`` / full-file branch of ``model_file``.

    LANDMINE (why this exists instead of ``Response.call_on_close``):
    ``send_file`` returns a response with ``direct_passthrough=True`` and
    ``resp.response`` set to a ``werkzeug.wsgi.FileWrapper``. Werkzeug's
    ``Response.get_app_iter`` special-cases ``direct_passthrough`` by handing
    the WSGI server ``self.response`` (the FileWrapper) DIRECTLY, bypassing
    the ``ClosingIterator(iterable, self.close)`` wrapper it uses in every
    other case. Concretely: the WSGI server (werkzeug's own dev server,
    gunicorn — both follow the WSGI spec of calling ``.close()`` on whatever
    iterable the app returned) ends up calling ``FileWrapper.close()``, never
    ``Response.close()`` — so anything registered via
    ``resp.call_on_close(...)`` is simply never invoked on this path. (Proven
    empirically while building this cap — a first attempt using
    ``call_on_close`` silently leaked every full-file permit.) Wrapping the
    iterable itself, instead of relying on the Response object's own close
    hook, sidesteps that entirely: whatever consumes/closes the returned
    iterable — real WSGI server or Flask's test client — reaches OUR
    ``close()``, which forwards to the real FileWrapper's close (still closes
    the underlying fh) and then releases the permit exactly once.
    """

    def __init__(self, inner, on_close):
        self._inner = inner
        self._on_close = on_close

    def __iter__(self):
        return iter(self._inner)

    def close(self):
        try:
            if hasattr(self._inner, "close"):
                self._inner.close()
        finally:
            self._on_close()


@worker_bp.route("/llm/models/<path:model_key>/file", methods=["GET"])
def model_file(model_key):
    if not _transfer_authorized():
        abort(401, description="Worker enrollment or operator token required.")
    _model, dest = _model_dir_or_404(model_key)

    rel = request.args.get("path", "")
    if not rel:
        abort(400, description="Missing ?path=")

    # ── path jail (UNCHANGED — do not weaken) ─────────────────────────────
    # Confine LEXICALLY first (catches ../ and absolute paths WITHOUT following
    # a symlink at the target — comfy checkpoints in the model dir ARE symlinks
    # into the shared /checkpoints store, and realpath-first falsely 403'd them,
    # breaking every comfy per-file transfer).
    target = os.path.normpath(os.path.join(dest, rel))
    if target != dest and not target.startswith(dest + os.sep):
        abort(403, description="Path escapes model directory.")
    # Then follow the link, but ONLY within the model storage root — a stray
    # symlink to /etc/passwd still 403s.
    real = os.path.realpath(target)
    from ....imports.src.constants.constants import DEFAULT_ROOT as _DR
    root_real = os.path.realpath(_DR)
    if real != target and not (real == root_real or real.startswith(root_real + os.sep)):
        abort(403, description="Symlink escapes the model storage root.")
    target = real
    if not os.path.isfile(target):
        abort(404, description="No such file.")
    if not os.access(target, os.R_OK):
        # Exists on disk but the API process can't read it (permission-
        # restricted — e.g. an HF cache metadata file dropped 0600 by a
        # different uid; live 2026-07-18 on a comfy checkpoint's .cache/
        # tree json). Degrade to a clean 404 BEFORE any response headers
        # commit, instead of a raw PermissionError 500 from inside
        # send_file()'s open() — matches the house degrade-not-500 pattern.
        # Belt-and-suspenders: the real fix is the manifest/archive walkers
        # no longer OFFERING dot-directory files at all (see
        # format_select.walk_listing) — this guards any other path to an
        # unreadable file too (a direct ?path= guess, a legit weight with
        # bad perms, …).
        logger.warning("model_file: %s exists but is not readable by the API "
                       "process; refusing (permission denied on central).",
                       target)
        abort(404, description="File exists on central but is not readable "
                               "(permission denied); treat as unavailable.")

    # ── serve the bytes ───────────────────────────────────────────────────
    # KEEPER FOLLOW-UP (X-Accel-Redirect): everything above RESOLVES the target;
    # everything below SERVES it. That split is the seam for the nginx sendfile
    # offload — replace the body below with a Response carrying
    # ``X-Accel-Redirect: <internal-location-mapping-to target>`` so nginx serves
    # the bytes (Range included) and Python never touches them. Left explicit and
    # self-contained for exactly that swap. (NOT implemented here on purpose.)
    size = os.path.getsize(target)
    download_name = os.path.basename(target)
    ctype = mimetypes.guess_type(download_name)[0] or "application/octet-stream"

    rng = _parse_single_range(request.headers.get("Range", ""), size)

    if rng == "unsatisfiable":
        resp = Response(status=416)
        resp.headers["Content-Range"] = f"bytes */{size}"
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    # ── global transfer cap ────────────────────────────────────────────────
    # Acquire AFTER auth + path-jail + 404/416 (never spend a permit on a
    # request that was never going to stream bytes), but BEFORE the first
    # byte is served. See _TransferPermit above for why this can't be a
    # plain "with" around the handler: both branches below return a
    # streaming response and the permit must outlive this function's return.
    permit = _acquire_transfer_permit()
    if permit is None:
        return _transfer_busy_response()

    if rng is None:
        # Full-file GET. send_file streams via wsgi.file_wrapper (sendfile) and
        # sets Content-Length/Content-Type/Last-Modified. conditional=False so
        # werkzeug's O(offset) _RangeWrapper is NEVER engaged — every Range is
        # handled by the seek path above, so this only ever serves a whole file.
        resp = send_file(target, as_attachment=True, download_name=download_name,
                         conditional=False)
        resp.headers["Accept-Ranges"] = "bytes"  # advertise range support
        # send_file's response is direct_passthrough with a bare FileWrapper
        # iterable — Response.call_on_close does NOT fire here (see
        # _ReleaseOnCloseIter's docstring for why). Wrap the iterable itself
        # so the permit releases exactly once when the WSGI server (or a
        # dropped connection) closes it — stream finished OR aborted.
        resp.response = _ReleaseOnCloseIter(resp.response, permit.release)
        return resp

    # Satisfiable single range -> 206 with a real seek (O(1) to the offset).
    start, end = rng
    length = end - start + 1

    def _ranged_body():
        try:
            yield from _stream_file_window(target, start, end)
        finally:
            # Released on normal completion AND on client disconnect/error:
            # a generator's `finally` still runs when werkzeug closes it via
            # GeneratorExit (dropped connection) or an exception propagates.
            permit.release()

    resp = Response(_ranged_body(), status=206,
                    mimetype=ctype, direct_passthrough=True)
    resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(length)
    resp.headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return resp


# Per-chunk hashing for verified transfers. 32 MiB chunks: big enough that a
# 40 GB file is ~1280 sums (tiny JSON), small enough that a failed/corrupt
# chunk re-fetch is cheap. Clamped so a caller can't request degenerate sizes.
_CHUNKSUM_DEFAULT = 32 * 1024 * 1024
_CHUNKSUM_MIN, _CHUNKSUM_MAX = 4 * 1024 * 1024, 256 * 1024 * 1024


def _chunk_sums(path: str, chunk_bytes: int) -> list[str]:
    """SHA-256 of each ``chunk_bytes`` slice of ``path``, sidecar-cached.

    Hashing a 40 GB file off HDD takes minutes — it must happen once per
    (file, chunk size), not per worker pull. The sidecar is keyed by size +
    mtime so a re-uploaded file re-hashes; failure to WRITE the sidecar is
    tolerated (read-only mounts) at the cost of re-hashing next time.
    """
    import hashlib
    import json

    st = os.stat(path)
    side = f"{path}.chunksums-{chunk_bytes}.json"
    try:
        with open(side, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("size") == st.st_size and cached.get("mtime") == int(st.st_mtime):
            return cached["sums"]
    except Exception:
        pass
    sums = []
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            sums.append(hashlib.sha256(buf).hexdigest())
    try:
        tmp = side + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"size": st.st_size, "mtime": int(st.st_mtime),
                       "chunk_bytes": chunk_bytes, "sums": sums}, fh)
        os.replace(tmp, side)
    except OSError:
        pass
    return sums


@worker_bp.route("/llm/models/<path:model_key>/chunksums", methods=["GET"])
def model_file_chunksums(model_key):
    """Per-chunk SHA-256 manifest for one file — the verify half of a chunked,
    resumable transfer. The worker downloads chunk-aligned ranges, hashes each
    as it lands, and re-fetches ONLY chunks that fail — so completeness is
    proven content, not a size that a preallocated-then-crashed pull can fake.
    """
    if not _transfer_authorized():
        abort(401, description="Worker enrollment or operator token required.")
    _model, dest = _model_dir_or_404(model_key)
    rel = request.args.get("path", "")
    if not rel:
        abort(400, description="Missing ?path=")
    target = os.path.realpath(os.path.join(dest, rel))
    if target != dest and not target.startswith(dest + os.sep):
        abort(403, description="Path escapes model directory.")
    if not os.path.isfile(target):
        abort(404, description="No such file.")
    try:
        chunk = int(request.args.get("chunk") or _CHUNKSUM_DEFAULT)
    except ValueError:
        chunk = _CHUNKSUM_DEFAULT
    chunk = max(_CHUNKSUM_MIN, min(chunk, _CHUNKSUM_MAX))
    return jsonify({
        "path": rel,
        "size": os.path.getsize(target),
        "chunk_bytes": chunk,
        "algo": "sha256",
        "sums": _chunk_sums(target, chunk),
    })


@worker_bp.route("/llm/models/<path:model_key>/archive", methods=["GET"])
def model_archive(model_key):
    """Stream the model's ENTIRE directory as one uncompressed tar.

    This is the most reliable way to hand a worker a whole model: a single
    sequential stream instead of N per-file GETs that can drop files. The tar is
    produced on the fly through an OS pipe driven by a writer thread, so central
    never buffers the model (which can be many GB) in memory or stages it on
    disk. Members are stored at paths relative to the model dir, so the worker
    extracts straight into its own destination.

    Uncompressed (``w|``) on purpose: model weights are incompressible, so gzip
    would only burn CPU on both ends.
    """
    import tarfile
    import threading

    # Same exfil surface as /file and /manifest (streams the WHOLE model dir),
    # so it wears the same credential gate. NOT in the keeper's enumerated three
    # (file/manifest/chunksums) — flagged in the change report; revert this one
    # check if strict scoping is preferred.
    if not _transfer_authorized():
        abort(401, description="Worker enrollment or operator token required.")
    _model, dest = _model_dir_or_404(model_key)

    # Deterministic file list (same walk as the manifest), newest layout intact.
    # Symlinked members (comfy checkpoints -> /checkpoints store) are packed as
    # their REAL bytes: tar.add on a symlink stores a link member by default,
    # which the worker-side extractor (0.1.129+) rightly refuses. Dereference
    # here — but only links that stay within the model storage root.
    from ....imports.src.constants.constants import DEFAULT_ROOT as _DR
    from ..functions.imports.utils.format_select import select_files
    _root_real = os.path.realpath(_DR)
    # Whole-dir walk (skipping transfer-machinery sidecars, same as /manifest),
    # then SINGLE-FORMAT filter so the archive tar carries exactly the same file
    # set the per-file transport does — the fallback must not silently re-ship
    # every format the /manifest path excludes.
    walked = []            # [(rel, size)] for the format filter
    real_by_rel = {}       # rel -> real path to pack
    for root, dirs, names in os.walk(dest):
        # Same dot-directory prune as format_select.walk_listing (.cache/.git/…
        # are never servable weights — and HF cache metadata can be
        # permission-restricted, see that helper's docstring).
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(names):
            if ".chunksums-" in name or name.endswith((".part", ".part.state.json")):
                continue
            full = os.path.join(root, name)
            if not os.path.isfile(full):   # follows links: target must exist
                continue
            real = os.path.realpath(full)
            if real != full and not (real == _root_real or real.startswith(_root_real + os.sep)):
                logger.warning("archive %s: skipping %s (link escapes storage root)",
                               model_key, name)
                continue
            rel = os.path.relpath(full, dest)
            try:
                sz = os.path.getsize(full)
            except OSError:
                # Unreadable entry: degrade by skipping, never let one bad
                # entry break the whole archive walk.
                logger.warning("archive %s: skipping unreadable %s", model_key, name)
                continue
            walked.append((rel, sz))
            real_by_rel[rel] = real
    _model_fw = _model.get("framework") if isinstance(_model, dict) else None
    entries = [(real_by_rel[rel], rel)
               for (rel, _sz) in select_files(walked, framework=_model_fw)]

    # ── global transfer cap ────────────────────────────────────────────────
    # Acquire AFTER auth + the (cheap-ish) directory walk above, but BEFORE
    # generate() starts the pipe/writer-thread that actually streams bytes.
    # An archive pull is the biggest single amplifier of the 8*K problem (a
    # whole model dir over one connection), so it shares the same cap as
    # /file. See _TransferPermit above: release happens in generate()'s
    # existing try/finally below, which already runs on normal completion,
    # a dropped client connection (GeneratorExit), or a writer exception —
    # exactly the "stream lifetime, not handler return" semantics this cap
    # needs.
    permit = _acquire_transfer_permit()
    if permit is None:
        return _transfer_busy_response()

    def generate():
        r_fd, w_fd = os.pipe()

        def _writer():
            try:
                with os.fdopen(w_fd, "wb") as wf:
                    with tarfile.open(fileobj=wf, mode="w|") as tar:
                        for full, rel in entries:
                            try:
                                tar.add(full, arcname=rel, recursive=False)
                            except FileNotFoundError:
                                continue  # file vanished mid-stream; skip it
                            except OSError as exc:
                                # Unreadable entry (permission-restricted, …):
                                # degrade by skipping — never let one bad
                                # member abort an otherwise-good tar stream.
                                logger.warning(
                                    "archive %s: skipping unreadable member %s (%s)",
                                    model_key, rel, exc)
                                continue
            except Exception:
                logger.exception("archive writer failed for %s", model_key)

        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()
        try:
            with os.fdopen(r_fd, "rb") as rf:
                while True:
                    chunk = rf.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            thread.join()
            permit.release()

    return Response(
        generate(),
        mimetype="application/x-tar",
        headers={
            "Content-Disposition": f'attachment; filename="{model_key}.tar"',
            "X-Accel-Buffering": "no",
        },
        direct_passthrough=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Per-model serving control — what the console edits (mode + GPU/CPU/ctx).
#
# GET  /api/llm/serving                 overview rows for every model
# GET  /api/llm/serving/<key>           one model's effective serving + override
# POST /api/llm/serving/<key>           set override fields; {"apply": true} to
#                                       also (re)write + restart the unit
#
# The override is persisted (serve_overrides.json) and merged into the spec, so
# it drives the systemd unit, the swap config, and the HTTP runner endpoint.
# Applying systemd changes needs root; when the API isn't root we return the
# exact commands to run with sudo instead of failing.
# ──────────────────────────────────────────────────────────────────────────
def _apply_serving(model_key):
    import subprocess

    plan = install_serving(only=[model_key])
    if not plan.steps:
        return {"applied": False, "reason": "nothing to apply (mode=off)"}
    # geteuid() is POSIX-only; on Windows there's no uid-0 concept, so treat the
    # privilege gate as "not elevated" and surface the commands to run by hand.
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return {"applied": False, "reason": "API is not root; run with sudo",
                "commands": plan.describe()}

    def _run(argv):
        subprocess.run(list(argv), check=True)
        return " ".join(argv)

    def _write(path, content):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    apply_plan(plan, run=_run, write=_write)
    return {"applied": True, "commands": plan.describe()}


def _parse_systemd_ts(s):
    """systemd ActiveEnterTimestamp ('Sat 2026-06-27 08:00:00 UTC') -> epoch secs."""
    import time, calendar
    s = (s or "").strip()
    if not s or s.lower().startswith("n/a"):
        return None
    try:
        p = s.split()                       # [wday, date, time, tz]
        st = time.strptime(p[1] + " " + p[2], "%Y-%m-%d %H:%M:%S")
        return int(calendar.timegm(st))     # the timestamps here are UTC
    except Exception:
        return None


def _unit_live_state(model_key):
    """Live systemd-unit state for an explicitly-pinned model: is its always-on
    unit actually running, its cgroup RAM, and since when. Read-only `systemctl
    show` (no root). {active:None} if the unit/spec can't be resolved."""
    import subprocess
    try:
        unit = serve_spec_for(model_key).unit_name + ".service"
    except Exception:
        return {"active": None}
    try:
        out = subprocess.run(
            ["systemctl", "show", unit,
             "--property=ActiveState,SubState,MemoryCurrent,ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=3)
        kv = {}
        for line in (out.stdout or "").splitlines():
            k, _, v = line.partition("=")
            kv[k] = v
    except Exception as exc:
        return {"unit": unit, "active": None, "error": str(exc)}
    state = kv.get("ActiveState", "") or ""
    mem = kv.get("MemoryCurrent", "")
    return {
        "unit": unit,
        "state": state,                                  # active|inactive|failed|…
        "sub_state": kv.get("SubState", "") or "",
        "active": state == "active",
        "rss_bytes": int(mem) if mem.isdigit() and int(mem) < (1 << 63) else None,
        "since": _parse_systemd_ts(kv.get("ActiveEnterTimestamp", "")),
    }


@worker_bp.route("/llm/serving", methods=["GET"])
def serving_list():
    # Attach the EXPLICIT override overlay per row so the console can tell a
    # deliberate pin (operator set serve_mode in serve_overrides.json) from the
    # registry/env default (DEFAULT_SERVE_MODE=systemd makes every model look
    # "systemd" otherwise). For pinned rows, also attach live unit state so the
    # console can show whether the always-on unit is actually running + its RAM.
    rows = serving_overview()
    ov = all_overrides()
    for r in rows:
        r["override"] = ov.get(r.get("key"), {})
        if (r.get("override") or {}).get("serve_mode") == "systemd":
            r["unit"] = _unit_live_state(r.get("key"))
    return jsonify(rows)


def _gguf_choices(model_key):
    """(available .gguf basenames, currently-selected override) for the UI."""
    avail = []
    try:
        from abstract_hugpy_dev.imports.config.main import get_model_path
        avail = available_gguf_files(get_model_path(model_key))
    except Exception:
        avail = []
    selected = (get_override(model_key) or {}).get("gguf_file") or ""
    return avail, selected


def _gguf_detail(model_key):
    """Per-variant sizes + the effective (resolved) quant for the serving editor,
    so the variant dropdown can label each option with its size and mark the one
    that actually loads. Best-effort — {} for a non-GGUF or absent dir.

    Resolves the dir via route_destination (the authoritative install path the
    manifest records) so it matches /models — a discovered model's cfg.folder can
    be stale and point at a near-empty stub dir."""
    try:
        from abstract_hugpy_dev.imports.config.main import get_model_path, get_model_config
        cfg = None
        try:
            cfg = get_model_config(model_key)
        except Exception:  # noqa: BLE001
            cfg = None
        model_dir = None
        try:
            from abstract_hugpy_dev.imports.src.constants.paths import route_destination
            d = cfg.to_dict() if (cfg is not None and hasattr(cfg, "to_dict")) \
                else (cfg if isinstance(cfg, dict) else {})
            dest = route_destination(d) if d else None
            if dest and os.path.isdir(dest):
                model_dir = dest
        except Exception:  # noqa: BLE001
            model_dir = None
        if not model_dir:
            model_dir = get_model_path(model_key)
        return gguf_variants_detail(model_key, model_dir, cfg) or {}
    except Exception:  # noqa: BLE001
        return {}


def _with_gguf(row, model_key):
    """Attach the variant choices + sizes to a serving row (shared by GET/POST)."""
    row["available_gguf"], row["gguf_file"] = _gguf_choices(model_key)
    d = _gguf_detail(model_key)
    row["available_gguf_detail"] = d.get("variants") or []
    row["effective_gguf"] = d.get("effective_gguf")
    row["effective_bytes"] = d.get("effective_bytes")
    row["mmproj_bytes"] = d.get("mmproj_bytes")
    return row


@worker_bp.route("/llm/serving/<model_key>", methods=["GET"])
def serving_get(model_key):
    # A serving STATUS poll must degrade, never 500. An unknown/stale model_key
    # (e.g. a poller still asking about a deleted model like "Qwythos-…") made
    # serve_spec_for -> get_model_config raise KeyError, which became an
    # unhandled 500 on every poll; the console surfaced those 500s as repeated
    # "blips". Answer a clean, cheap 404 instead (and never open a store for an
    # unknown key). keeper 2026-07-13. [Note: an FD/EMFILE cascade was ruled out
    # — journal showed 0 EMFILE that day, virtiofsd was at 1M FDs.]
    try:
        spec = serve_spec_for(model_key)
    except KeyError:
        return jsonify({"model_key": model_key, "known": False, "serving": False,
                        "error": f"Unknown model: {model_key}"}), 404
    row = spec_row(spec)
    row["override"] = get_override(model_key)
    _with_gguf(row, model_key)
    return jsonify(row)


@worker_bp.route("/llm/serving/<model_key>", methods=["POST"])
def serving_set(model_key):
    body = request.get_json(silent=True) or {}
    do_apply = bool(body.pop("apply", False))
    set_override(model_key, body)

    row = spec_row(serve_spec_for(model_key))
    row["override"] = get_override(model_key)
    _with_gguf(row, model_key)
    if do_apply:
        row["apply"] = _apply_serving(model_key)
    else:
        try:
            row["plan"] = install_serving(only=[model_key]).describe()
        except Exception as exc:  # plan preview is best-effort
            row["plan_error"] = f"{type(exc).__name__}: {exc}"
    return jsonify(row)


# ──────────────────────────────────────────────────────────────────────────
# Model slots — the live pool of generic slot supervisors.
#
# GET  /api/llm/slots                what each slot is serving + free VRAM
# POST /api/llm/slots/load           {"model_key": ...}  load into a free slot
# POST /api/llm/slots/unload         {"control": "http://...:8101"}  free a slot
# GET  /api/llm/slots/install        one-time install steps (dry run; sudo to do)
# ──────────────────────────────────────────────────────────────────────────
def _sys_resources():
    """System RAM (with the used / reclaimable-cache / free split) + cores, read
    from this VM. `free` semantics: buff/cache = Buffers+Cached+SReclaimable (all
    reclaimable on demand, already counted in `available`); used = total-free-cache."""
    import os as _os
    mem = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable", "MemFree",
                         "Buffers", "Cached", "SReclaimable"):
                    mem[k] = int(v.split()[0]) * 1024
    except Exception:
        pass
    total = mem.get("MemTotal")
    free = mem.get("MemFree")
    cache = (mem.get("Buffers") or 0) + (mem.get("Cached") or 0) + (mem.get("SReclaimable") or 0)
    used = (total - free - cache) if (total is not None and free is not None) else None
    return {"total_bytes": total,
            "available_bytes": mem.get("MemAvailable"),
            "free_bytes": free,
            "cache_bytes": cache or None,
            "used_bytes": used,
            "cpu_count": _os.cpu_count()}


@worker_bp.route("/llm/slots", methods=["GET"])
def slots_overview():
    # Surface the EFFECTIVE slot count + raw SLOT_COUNT env so a systemd drop-in
    # silently overriding it (the resurrection ghost) is visible in the data
    # plane, not just guessable from how many slot rows appear.
    import os as _os
    meta = {"slot_count": _slot_count(), "slot_count_env": _os.environ.get("SLOT_COUNT")}
    if not slots_enabled():
        return jsonify({"enabled": False, "slots": [], "resources": _sys_resources(), **meta})
    return jsonify({"enabled": True, "slots": SlotPool().overview(),
                    "resources": _sys_resources(), **meta})


@worker_bp.route("/llm/free-worker", methods=["POST"])
def free_worker():
    """Recycle the API gunicorn worker to release its accumulated in-process RAM
    (anon memory — e.g. in-process embed/media models, registry caches). Sends a
    graceful SIGHUP to the gunicorn master: a fresh worker spawns and the old one
    drains in-flight requests then exits — no root, no dropped service. Slot
    agents are separate processes and are unaffected. (Reclaimable page cache is
    NOT this — the kernel frees that on demand; drop_caches isn't needed.)"""
    import os, signal
    ppid = os.getppid()
    try:
        os.kill(ppid, signal.SIGHUP)
        return jsonify({"ok": True, "signaled_pid": ppid,
                        "note": "API worker recycling — reconnect in a few seconds"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@worker_bp.route("/llm/slots/load", methods=["POST"])
def slots_load():
    body = request.get_json(silent=True) or {}
    if not body.get("model_key"):
        return jsonify({"error": "missing model_key"}), 400
    # optional per-load compute knobs (blank/omitted = autofit/default)
    opts = {k: body[k] for k in ("n_gpu_layers", "ctx", "threads", "cpus", "gpu")
            if body.get(k) not in (None, "")}
    try:
        endpoint = SlotPool().endpoint_for(body["model_key"], opts=opts)
    except RuntimeError as exc:
        # Scheduler refusals are DATA (e.g. "all slots are static-locked",
        # slot-agent preflight reasons) — a 409 with the reason, not a raw 500.
        return jsonify({"loaded": False, "reason": str(exc),
                        "slots": SlotPool().overview()}), 409
    if endpoint is None:
        return jsonify({"loaded": False, "reason": "all slots busy",
                        "slots": SlotPool().overview()}), 409
    return jsonify({"loaded": True, "endpoint": endpoint,
                    "slots": SlotPool().overview()})


@worker_bp.route("/llm/cache", methods=["GET"])
def cache_status():
    """SSD hot-cache overview (used/budget/free + cached entries + what's warming)."""
    from ....managers.serve import model_cache
    return jsonify(model_cache.status())


@worker_bp.route("/llm/cache/warm", methods=["POST"])
def cache_warm():
    """Background-warm a model's GGUF onto the SSD cache so its next load is fast.
    Returns immediately; the copy runs detached (idempotent, single-flight)."""
    import os as _os
    from ....managers.serve import model_cache
    body = request.get_json(silent=True) or {}
    key = body.get("model_key")
    if not key:
        return jsonify({"ok": False, "error": "model_key required"}), 400
    if not model_cache.enabled():
        return jsonify({"ok": False, "error": "model cache not enabled on this node"}), 503
    try:
        src = _model_file_for(key, get_model_config(key))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    if not src or not _os.path.isfile(src):
        return jsonify({"ok": False, "error": f"no gguf on disk for {key}"}), 404
    if model_cache.is_complete(src):
        return jsonify({"ok": True, "already_warm": True, "model_key": key})
    model_cache._warm_async(src)
    return jsonify({"ok": True, "warming": True, "model_key": key})


@worker_bp.route("/llm/slots/unload", methods=["POST"])
def slots_unload():
    body = request.get_json(silent=True) or {}
    control = body.get("control")
    if not control:
        return jsonify({"error": "missing control url"}), 400
    try:
        return jsonify(SlotPool().unload(control))
    except Exception as exc:  # slot unreachable / mid-load timeout — don't 500 the UI
        return jsonify({"unloaded": False,
                        "error": f"{type(exc).__name__}: {exc}"}), 502


@worker_bp.route("/llm/slots/install", methods=["GET"])
def slots_install():
    return jsonify({"steps": [{"kind": k, "payload": p}
                              for k, p in slot_install_steps()]})


# ──────────────────────────────────────────────────────────────────────────
# Central-driven worker install.
#
# An operator on a GPU box runs ONE command; everything else (where to find the
# agent, which central to call back, the port) is supplied by central here, so
# the worker doesn't need to be pre-configured:
#
#     curl -fsSL https://api.hugpy.ai/llm/workers/install.sh | bash
#
# The script makes sure abstract_hugpy is importable, then launches the agent
# pointed at THIS central (derived from the request host). Override port/name
# with env vars before the pipe, e.g.  WORKER_PORT=9101 WORKER_NAME=gpu2 bash.
# ──────────────────────────────────────────────────────────────────────────
def _central_base_url() -> str:
    """The externally-visible base URL of this central node, from the request."""
    # Honor proxy headers so we emit the public https URL, not the gunicorn host.
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host") or request.host
    return f"{proto}://{host}"


def _primary_lan_ip() -> str:
    """This box's primary outbound-interface IP, via the UDP-connect trick
    (connect() on a datagram socket sends nothing; it just resolves the route).
    Empty string when the box has no route at all."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


@worker_bp.route("/llm/workers/central-address", methods=["GET"])
def workers_central_address():
    """A worker-reachable origin for the console's install command.

    The console renders the one-line installer from the browser's origin; when
    the operator browses central on the box itself (http://localhost:7002) that
    origin is loopback — useless pasted on a remote GPU box. base_url is the
    request origin with a loopback host swapped for this box's primary
    interface IP (scheme and port preserved), so the copied command both
    fetches the script and bakes a central URL the worker can actually reach.
    """
    from urllib.parse import urlsplit
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host") or request.host
    hostname = (urlsplit(f"//{host}").hostname or "").lower()
    loopback = hostname in _UNREACHABLE_HOSTS
    lan_ip = _primary_lan_ip()
    if loopback and lan_ip:
        port = urlsplit(f"//{host}").port
        host = f"{lan_ip}:{port}" if port else lan_ip
    return jsonify({"base_url": f"{proto}://{host}", "lan_ip": lan_ip,
                    "request_host_loopback": loopback})


@worker_bp.route("/llm/workers/install.sh", methods=["GET"])
def worker_install_script():
    central = _central_base_url()
    script = r"""#!/usr/bin/env bash
# abstract_hugpy_dev GPU worker — one-line installer (served by central).
set -euo pipefail

CENTRAL="${WORKER_CENTRAL_URL:-__CENTRAL__}"
PORT="${WORKER_PORT:-9100}"
NAME="${WORKER_NAME:-$(hostname)}"
# Enrollment token (issued by the console). Required once central has
# HUGPY_WORKER_ENROLL_REQUIRED on; during gradual rollout it's optional but
# recommended. Supplied via env, NOT baked into this script (it's served openly):
#   WORKER_ENROLL_TOKEN=hpw_... curl -fsSL $CENTRAL/api/llm/workers/install.sh | bash
TOKEN="${WORKER_ENROLL_TOKEN:-}"
# WORKER_PYTHON forces a specific interpreter; otherwise we auto-detect one that
# already has abstract_hugpy_dev installed.
PY="${WORKER_PYTHON:-}"
# SYSTEMD=1 installs+enables a user service (auto-start on boot); default just
# runs in the foreground. SYSTEMD=0 to force foreground.
SYSTEMD="${SYSTEMD:-ask}"
# Where the worker stores models it pulls from central. A worker does NOT need
# central's /mnt mount: it downloads each model once over HTTP (resumable) and
# caches it locally, which is faster than serving weights live over sshfs/NFS.
# Default to a local dir so a missing/broken /mnt never matters; override with
# DEFAULT_ROOT.
export DEFAULT_ROOT="${DEFAULT_ROOT:-$HOME/.abstract_hugpy/storage}"

echo "abstract_hugpy_dev worker installer"
echo "  central : $CENTRAL"
echo "  name    : $NAME"
echo "  port    : $PORT"
echo "  storage : $DEFAULT_ROOT"
[[ -n "$TOKEN" ]] && echo "  token   : (enrollment token supplied)" || echo "  token   : (none — relying on gradual enrollment)"

has_hugpy() { "$1" -c "import abstract_hugpy_dev" >/dev/null 2>&1; }

# 1. Find a python that can import abstract_hugpy_dev.
if [[ -n "$PY" ]]; then
  if ! has_hugpy "$PY"; then
    echo "error: WORKER_PYTHON=$PY cannot import abstract_hugpy_dev. Details:" >&2
    "$PY" -c "import abstract_hugpy_dev" || true
    exit 1
  fi
else
  echo "Searching for a python with abstract_hugpy_dev…"
  CANDIDATES=()
  # the currently-active env first (you ran this from inside it)
  [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python3" ]] && CANDIDATES+=("$CONDA_PREFIX/bin/python3")
  [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]] && CANDIDATES+=("$VIRTUAL_ENV/bin/python3")
  # current PATH pythons
  for c in python3 python; do command -v "$c" >/dev/null 2>&1 && CANDIDATES+=("$(command -v "$c")"); done
  # conda envs
  for base in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" \
              /opt/*/miniconda3 /opt/*/miniforge3 /opt/conda; do
    for p in "$base"/bin/python3 "$base"/envs/*/bin/python3; do
      [[ -x "$p" ]] && CANDIDATES+=("$p")
    done
  done
  # common venv locations
  for p in /opt/*/venv/bin/python3 "$HOME"/.virtualenvs/*/bin/python3 \
           /srv/*/venv/bin/python3; do
    [[ -x "$p" ]] && CANDIDATES+=("$p")
  done

  # De-duplicate while preserving order.
  declare -A SEEN=()
  UNIQ=()
  for c in "${CANDIDATES[@]}"; do
    [[ -n "${SEEN[$c]:-}" ]] && continue
    SEEN[$c]=1; UNIQ+=("$c")
  done

  FIRST_ERR=""
  for cand in "${UNIQ[@]}"; do
    if has_hugpy "$cand"; then PY="$cand"; break; fi
    # Capture the first real import error so we can show WHY (not just "not found").
    if [[ -z "$FIRST_ERR" ]]; then
      FIRST_ERR="$("$cand" -c "import abstract_hugpy_dev" 2>&1 || true)"
      [[ -n "$FIRST_ERR" ]] && FIRST_ERR="[$cand] $FIRST_ERR"
    fi
  done

  if [[ -z "$PY" ]]; then
    echo "error: no python could import abstract_hugpy_dev." >&2
    echo "Checked: ${UNIQ[*]:-<none>}" >&2
    if [[ -n "$FIRST_ERR" ]]; then
      echo "First import error was:" >&2
      echo "$FIRST_ERR" >&2
    fi
    echo "If the package is installed but import fails above, that error is the" >&2
    echo "real problem (e.g. a missing dependency). Otherwise install it, or run:" >&2
    echo "  WORKER_PYTHON=/path/to/python curl -fsSL $CENTRAL/api/llm/workers/install.sh | bash" >&2
    exit 1
  fi
fi
echo "  python  : $PY"

RUN_CMD=("$PY" -m abstract_hugpy_dev.worker_agent --central "$CENTRAL" --name "$NAME" --port "$PORT")
[[ -n "$TOKEN" ]] && RUN_CMD+=(--token "$TOKEN")

# 2. Optionally install a systemd --user service so it auto-starts on boot.
maybe_systemd() {
  command -v systemctl >/dev/null 2>&1 || { echo "systemctl not found; running foreground."; return 1; }
  if [[ "$SYSTEMD" == "ask" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "Install a systemd --user service so it auto-starts on boot? [y/N] " ans
      [[ "$ans" =~ ^[Yy] ]] || return 1
    else
      # piped (curl|bash) with no TTY: default to foreground unless SYSTEMD=1.
      return 1
    fi
  elif [[ "$SYSTEMD" != "1" ]]; then
    return 1
  fi
  return 0
}

if maybe_systemd; then
  UDIR="$HOME/.config/systemd/user"
  mkdir -p "$UDIR"
  cat > "$UDIR/abstract-hugpy-worker.service" <<UNIT
[Unit]
Description=abstract_hugpy_dev GPU worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=WORKER_CENTRAL_URL=$CENTRAL
Environment=WORKER_NAME=$NAME
Environment=WORKER_PORT=$PORT
Environment=DEFAULT_ROOT=$DEFAULT_ROOT
${TOKEN:+Environment=WORKER_ENROLL_TOKEN=$TOKEN}
ExecStart=$PY -m abstract_hugpy_dev.worker_agent --central $CENTRAL --name $NAME --port $PORT ${TOKEN:+--token $TOKEN}
# on-failure (not always): a deliberate block/revoke makes the agent exit 0, so
# systemd leaves it stopped; transient crashes exit non-zero and are restarted.
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now abstract-hugpy-worker.service
  # Let the service keep running after logout.
  command -v loginctl >/dev/null 2>&1 && loginctl enable-linger "$USER" 2>/dev/null || true
  echo "✓ Installed user service. Logs: journalctl --user -u abstract-hugpy-worker -f"
  exit 0
fi

# 3. Foreground run.
# Termux/Android: hold a wakelock so Android Doze doesn't throttle the worker's
# background network/CPU when the screen sleeps. Without it the periodic OUTBOUND
# heartbeat to central stalls ("read operation timed out") and central marks the
# worker offline — even though INBOUND /infer still works while it's being hit.
# This is the same fix phone_brick/bootstrap.sh already applies for the vision pool.
if [[ "${PREFIX:-}" == *com.termux* ]] || command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock 2>/dev/null || true
  echo "  termux : wake-lock held (keeps heartbeats alive under Doze)"
  trap 'termux-wake-unlock 2>/dev/null || true' EXIT
  echo "Starting worker agent in the foreground (Ctrl-C to stop)…"
  "${RUN_CMD[@]}"
else
  echo "Starting worker agent in the foreground (Ctrl-C to stop)…"
  exec "${RUN_CMD[@]}"
fi
"""
    script = script.replace("__CENTRAL__", central)
    return Response(script, mimetype="text/x-shellscript")

