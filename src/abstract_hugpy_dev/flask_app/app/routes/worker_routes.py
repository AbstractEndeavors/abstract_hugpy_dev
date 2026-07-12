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
)
from ..functions.imports.utils.enrollment_tokens import (
    create_enrollment_token, verify_enrollment_token,
    revoke_enrollment_token, list_enrollment_tokens,
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


def _kick_warm(worker, model_keys, source: str) -> list:
    """Probe the given models on the worker in ONE background thread.

    Returns the models actually scheduled (cooldown/busy-filtered). Safe to
    call from any request: never blocks, never raises."""
    import httpx
    wid = (worker or {}).get("id") or ""
    base = ((worker or {}).get("url") or "").rstrip("/")
    if not wid or not base:
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
    for w in rows:
        w["required_pkg_version"] = required
        w["version_ok"] = (required is None
                           or w.get("pkg_version") == required)
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
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>", methods=["GET"])
def workers_get(worker_id):
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


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
        storage=body.storage,
        install=body.install,
        serving_limits=body.serving_limits,
        slot_capable=body.slot_capable,
        slot_incapable_reason=body.slot_incapable_reason,
        task_capabilities=body.task_capabilities,
    )
    if worker is None:
        # The agent thinks it's registered but central forgot it (restart,
        # cleared registry). 410 tells the agent to re-register.
        abort(410, description="Unknown worker id; please re-register.")
    if worker.get("admission") == "blocked":
        # Persistent eviction: 403 stops the agent instead of letting it limp on.
        abort(403, description="Worker is blocked by the operator.")
    # Designated = ready: re-converge assigned-vs-loaded on every beat. A cold
    # assigned model (worker rebooted, agent restarted, weights evicted) gets a
    # background warm — rate-limited by _WARM_COOLDOWN_S so an un-fittable
    # model doesn't probe-spin.
    try:
        cold = [mk for mk in (worker.get("models") or [])
                if mk not in set(worker.get("loaded_models") or [])]
        if cold:
            _kick_warm(worker, cold, "reconcile")
    except Exception:
        pass  # readiness convergence must never fail a heartbeat
    # Advertise the target version every beat, so a worker converges within one
    # heartbeat of central's required version changing.
    worker["required_pkg_version"] = required_pkg_version()
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>", methods=["DELETE"])
def workers_remove(worker_id):
    if not remove_worker(worker_id):
        abort(404, description="Unknown worker id.")
    return jsonify({"removed": True, "id": worker_id})


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


@worker_bp.route("/llm/workers/<worker_id>/assign", methods=["POST"])
def workers_assign(worker_id):
    body = AssignRequest(**(request.get_json(silent=True) or {}))
    if body.model_key not in get_models_dict(dict_return=True):
        abort(404, description="Unknown model key.")
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
    worker = assign_model(worker_id, body.model_key, spill=body.spill)
    if worker is None:
        abort(404, description="Unknown worker id.")
    # Designated = ready: load it on the worker NOW (background), don't wait
    # for the first inference to pay the lazy load.
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
    NOTHING deletes except through this explicit, operator-gated, audited call;
    there is no background monitoring and no auto-approval.

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

    # (1)+(2) recompute + intersect — defense-in-depth against a render->approve
    # race. The proposal is derived from the RAW record (not the public view).
    proposal = worker_storage_view(worker_id) or {}
    ev_by_key = {e["model_key"]: e
                 for e in (proposal.get("proposed_evictions") or [])}
    intersected = [k for k in approved_keys if k in ev_by_key]
    dropped = [k for k in approved_keys if k not in ev_by_key]

    # Audit the approved eviction with per-model bytes + last_picked (the exact
    # {model_key, bytes, last_picked} rows the console rendered), plus what the
    # central intersection dropped.
    from .comms_routes import audit
    approved_evictions = [ev_by_key[k] for k in intersected]
    audit("worker.reap-approve", {
        "worker_id": worker_id, "worker": worker.get("name"),
        "approved": approved_keys, "dropped": dropped,
        "evictions": approved_evictions,
        "freed_estimate_bytes": sum(e.get("bytes") or 0
                                    for e in approved_evictions),
    })

    if not intersected:
        # Everything approved has since become protected or is no longer over
        # budget. Report it as data (not a 5xx) so the console explains the drop.
        return jsonify({
            "ok": True, "freed_bytes": 0, "results": [],
            "approved": approved_keys, "reaped": [], "dropped": dropped,
            "note": ("approved models are no longer eligible for reaping "
                     "(loaded/assigned/pinned/provisioning, or the worker is back "
                     "under budget) — nothing deleted"),
        })

    # (3) delegate to the SAME guarded reaper relay; the worker re-proves every
    # guard per key at delete time (the single delete path — never trusts a
    # preview). _relay_worker_op audits worker.reap-approve too.
    resp, status = _relay_worker_op(
        worker_id, "/reap", {"model_keys": intersected},
        timeout=120.0, action="reap-approve")
    # Fold the central intersection outcome into the reaper's typed result so the
    # console shows deleted-vs-skipped AND anything central dropped pre-relay.
    data = resp.get_json(silent=True)
    if isinstance(data, dict):
        data.setdefault("approved", approved_keys)
        data["reaped"] = intersected
        data["dropped"] = dropped
        return jsonify(data), status
    return resp, status


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
    capacity = (vram or 0) + (ram or 0)
    fit = (need_raw <= capacity) if capacity else None
    gpu_resident = (vram is not None) and (need <= vram)
    where = worker.get("gpu") or "this worker"
    if fit is False:
        reason = (f"won't fit {where}: model is {need_raw/gib:.1f} GiB but only "
                  f"{(vram or 0)/gib:.1f} GiB VRAM + {(ram or 0)/gib:.1f} GiB RAM free "
                  f"({capacity/gib:.1f} GiB total)")
    elif fit and not gpu_resident:
        reason = (f"fits but would partially offload to CPU: needs ~{need/gib:.1f} GiB, "
                  f"only {(vram or 0)/gib:.1f} GiB VRAM free on {where} — slower than GPU-resident")
    else:
        reason = None
    return {"fit": fit, "gpu_resident": gpu_resident, "need": need, "need_raw": need_raw,
            "vram_free": vram, "ram_free": ram, "capacity": capacity,
            "headroom": VRAM_HEADROOM, "reason": reason}


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
        abort(404, description="Unknown model key.")
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


@worker_bp.route("/llm/models/<path:model_key>/manifest", methods=["GET"])
def model_file_manifest(model_key):
    if not _transfer_authorized():
        abort(401, description="Worker enrollment or operator token required.")
    model, dest = _model_dir_or_404(model_key)

    files = []
    total = 0
    for root, _dirs, names in os.walk(dest):
        for name in names:
            # Transfer machinery artifacts are not model content: chunk-hash
            # sidecars (server-side cache) and .part/.state staging files
            # (worker-side, in case a worker dir is ever served back out).
            if ".chunksums-" in name or name.endswith((".part", ".part.state.json")):
                continue
            full = os.path.join(root, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            rel = os.path.relpath(full, dest)
            files.append({"path": rel, "size": size})
            total += size

    return jsonify({
        "model_key": model_key,
        "hub_id": model.get("hub_id"),
        "name": model.get("name"),
        "framework": model.get("framework"),
        "task": model.get("task") or model.get("primary_task"),
        "filename": model.get("filename"),
        "include": model.get("include"),
        "total_bytes": total,
        "files": files,
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

    if rng is None:
        # Full-file GET. send_file streams via wsgi.file_wrapper (sendfile) and
        # sets Content-Length/Content-Type/Last-Modified. conditional=False so
        # werkzeug's O(offset) _RangeWrapper is NEVER engaged — every Range is
        # handled by the seek path above, so this only ever serves a whole file.
        resp = send_file(target, as_attachment=True, download_name=download_name,
                         conditional=False)
        resp.headers["Accept-Ranges"] = "bytes"  # advertise range support
        return resp

    # Satisfiable single range -> 206 with a real seek (O(1) to the offset).
    start, end = rng
    length = end - start + 1
    resp = Response(_stream_file_window(target, start, end), status=206,
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
    _root_real = os.path.realpath(_DR)
    entries = []
    for root, _dirs, names in os.walk(dest):
        for name in sorted(names):
            full = os.path.join(root, name)
            if not os.path.isfile(full):   # follows links: target must exist
                continue
            real = os.path.realpath(full)
            if real != full and not (real == _root_real or real.startswith(_root_real + os.sep)):
                logger.warning("archive %s: skipping %s (link escapes storage root)",
                               model_key, name)
                continue
            entries.append((real, os.path.relpath(full, dest)))

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
    row = spec_row(serve_spec_for(model_key))
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

