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

from pydantic import BaseModel, Field
from flask import request, jsonify, abort, send_file, Response

from .imports import *
from ....managers.serve.overrides import get_override, set_override, available_gguf_files, all_overrides
# Serving/slot drivers: imported at module scope (the route handlers below call
# these by bare name). They are not on the functions star re-export chain.
from ....managers.serve.serve import (
    install_serving, apply_plan, serving_overview, serve_spec_for, spec_row,
)
from ....managers.serve.slots import SlotPool, slots_enabled, slot_install_steps
# Self-update plumbing: the version central wants workers on, and the wheel dir
# it serves as a simple index. Imported directly (not via the functions star) so
# this doesn't depend on the re-export chain picking up the new names.
from ..functions.imports.utils.workers import (
    required_pkg_version, pkg_index_dir, set_worker_admission, set_worker_pool, enroll_required,
)
from ..functions.imports.utils.enrollment_tokens import (
    create_enrollment_token, verify_enrollment_token,
    revoke_enrollment_token, list_enrollment_tokens,
)

worker_bp, logger = get_bp("worker_bp", __name__)


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
    # Inference-engine capability snapshot, e.g. {"installed": bool, "version":
    # str, "supports_gpu_offload": bool}. Lets central skip a worker that can't
    # actually serve (no engine) instead of routing a request that will fail.
    engine: dict | None = None
    # Dedicated-pool label (WORKER_POOL). "" = general. A pooled worker serves
    # ONLY requests tagged for its pool — reserved capacity for an external app.
    pool: str | None = None


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
    engine: dict | None = None
    pool: str | None = None


class AssignRequest(BaseModel):
    model_key: str
    # Optional per-assignment GPU/CPU spill override. Empty/omitted = autofit.
    # Recognized keys: n_gpu_layers (int|"auto"|"off"), gpu_mem_gib (float),
    # cpu_mem_gib (float), tensor_split (list[float]).
    spill: dict | None = None


@worker_bp.route("/llm/workers", methods=["GET"])
def workers_list():
    return jsonify(list_workers())


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
    body = RegisterRequest(**(request.get_json(silent=True) or {}))
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
        engine=body.engine,
        pool=body.pool,
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
        provisioning=body.provisioning,
        provision_progress=body.provision_progress,
        spill=body.spill,
        url=url,
        pkg_version=body.pkg_version,
        role=body.role,
        rpc_endpoint=body.rpc_endpoint,
        free_ram=body.free_ram,
        engine=body.engine,
        pool=body.pool,
    )
    if worker is None:
        # The agent thinks it's registered but central forgot it (restart,
        # cleared registry). 410 tells the agent to re-register.
        abort(410, description="Unknown worker id; please re-register.")
    if worker.get("admission") == "blocked":
        # Persistent eviction: 403 stops the agent instead of letting it limp on.
        abort(403, description="Worker is blocked by the operator.")
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


@worker_bp.route("/llm/workers/<worker_id>/assign", methods=["POST"])
def workers_assign(worker_id):
    body = AssignRequest(**(request.get_json(silent=True) or {}))
    if body.model_key not in get_models_dict(dict_return=True):
        abort(404, description="Unknown model key.")
    worker = assign_model(worker_id, body.model_key, spill=body.spill)
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/workers/<worker_id>/unassign", methods=["POST"])
def workers_unassign(worker_id):
    body = AssignRequest(**(request.get_json(silent=True) or {}))
    worker = unassign_model(worker_id, body.model_key)
    if worker is None:
        abort(404, description="Unknown worker id.")
    return jsonify(worker)


@worker_bp.route("/llm/chat/cancel/<request_id>", methods=["POST"])
def chat_cancel(request_id):
    """Cancel an in-flight chat by relaying to whichever worker is running it.

    The browser knows the request_id (echoed in the SSE 'request' event). We
    don't track which worker owns it, so we fan the cancel out to every online
    worker; the one running it stops, the rest 404 harmlessly.
    """
    import httpx

    cancelled = False
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

    Body: {model_key, spill?, force?}. force=true skips the preflight (still
    bounded by the worker's own limits)."""
    import httpx
    import threading

    raw = request.get_json(silent=True) or {}
    body = AssignRequest(**raw)
    force = bool(raw.get("force"))
    if body.model_key not in get_models_dict(dict_return=True):
        abort(404, description="Unknown model key.")
    worker = get_worker(worker_id)
    if worker is None:
        abort(404, description="Unknown worker id.")

    verdict = _worker_fit(body.model_key, worker)
    if verdict.get("fit") is False and not force:
        # `error` so the shared fetchJson surfaces the human reason (it only reads
        # error/detail/message); `reason`+`preflight` kept for programmatic use.
        return jsonify({"loaded": False, "error": verdict["reason"],
                        "reason": verdict["reason"], "preflight": verdict}), 409

    # passed (or forced/undecided) → assign, then warm in the background
    assign_model(worker_id, body.model_key, spill=body.spill)
    url = (worker.get("url") or "").rstrip("/") + "/probe/" + body.model_key

    def _warm():
        try:
            httpx.post(url, timeout=900.0)  # worker loads synchronously; can be slow
        except Exception:
            pass  # best-effort — lazy-load on first inference covers a warm failure

    threading.Thread(target=_warm, name=f"warm-{worker_id[:8]}", daemon=True).start()
    return jsonify({"loaded": "loading", "assigned": True, "preflight": verdict,
                    "worker": get_worker(worker_id),
                    "note": "assigned + warming on the worker — watch loaded_models for residency"})


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
    model, dest = _model_dir_or_404(model_key)

    files = []
    total = 0
    for root, _dirs, names in os.walk(dest):
        for name in names:
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


@worker_bp.route("/llm/models/<path:model_key>/file", methods=["GET"])
def model_file(model_key):
    _model, dest = _model_dir_or_404(model_key)

    rel = request.args.get("path", "")
    if not rel:
        abort(400, description="Missing ?path=")

    # Resolve and confine: the final real path must stay inside dest.
    target = os.path.realpath(os.path.join(dest, rel))
    if target != dest and not target.startswith(dest + os.sep):
        abort(403, description="Path escapes model directory.")
    if not os.path.isfile(target):
        abort(404, description="No such file.")

    # conditional/Range handling is provided by send_file.
    return send_file(target, as_attachment=True,
                     download_name=os.path.basename(target),
                     conditional=True)


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

    _model, dest = _model_dir_or_404(model_key)

    # Deterministic file list (same walk as the manifest), newest layout intact.
    entries = []
    for root, _dirs, names in os.walk(dest):
        for name in sorted(names):
            full = os.path.join(root, name)
            if os.path.isfile(full):
                entries.append((full, os.path.relpath(full, dest)))

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


@worker_bp.route("/llm/serving/<model_key>", methods=["GET"])
def serving_get(model_key):
    row = spec_row(serve_spec_for(model_key))
    row["override"] = get_override(model_key)
    row["available_gguf"], row["gguf_file"] = _gguf_choices(model_key)
    return jsonify(row)


@worker_bp.route("/llm/serving/<model_key>", methods=["POST"])
def serving_set(model_key):
    body = request.get_json(silent=True) or {}
    do_apply = bool(body.pop("apply", False))
    set_override(model_key, body)

    row = spec_row(serve_spec_for(model_key))
    row["override"] = get_override(model_key)
    row["available_gguf"], row["gguf_file"] = _gguf_choices(model_key)
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
    if not slots_enabled():
        return jsonify({"enabled": False, "slots": [], "resources": _sys_resources()})
    return jsonify({"enabled": True, "slots": SlotPool().overview(),
                    "resources": _sys_resources()})


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
    endpoint = SlotPool().endpoint_for(body["model_key"], opts=opts)
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

