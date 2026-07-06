import threading
import time as _time

from ..functions import *
# Registry prune (hide a not-installed "ghost" model) + media-chat allow-flag.
# Explicit imports so they work regardless of the functions star-export.
from ....imports.config.models.models_config import (
    prune_model, set_model_media, media_state,
    media_default_state, set_media_default, refresh_registry,
)

llm_bp, logger = get_bp("llm_bp", __name__)

for name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(name).setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────
@llm_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "storage_root": str(settings.storage_root),
        "manifest_path": str(settings.manifest_path),
    })


@llm_bp.route("/llm/peers", methods=["GET"])
def peers():
    return jsonify(list_peers())


def _annotate_gguf_size(model: dict, mk: str) -> None:
    """For a GGUF model, attach the EFFECTIVE-quant size — the single quant that
    actually serves (operator ``gguf_file`` override → ``cfg.filename`` → auto),
    plus its mmproj projector — so the console shows the model's real size instead
    of the whole-directory / whole-repo sum (a GGUF repo holds many quants; only
    one is served). No-op for transformers models and for GGUF dirs not downloaded
    here. Model-level + worker-agnostic, so the Models tab AND the worker-card
    strip both read this one number (same /models feed)."""
    fw = (model.get("framework") or "").lower()
    if fw not in ("gguf", "llama_cpp"):
        return
    try:
        from ....managers.serve.overrides import gguf_variants_detail
        from ....imports.config.main import get_model_config, get_model_path
        model_dir = model.get("destination") or get_model_path(mk)
        try:
            cfg = get_model_config(mk)
        except Exception:  # noqa: BLE001
            cfg = None
        d = gguf_variants_detail(mk, model_dir, cfg)
    except Exception:  # noqa: BLE001 — never break the models list over sizing
        d = {}
    if not d:
        return
    if d.get("effective_bytes"):
        model["effective_bytes"] = d["effective_bytes"]
    model["effective_gguf"] = d.get("effective_gguf")
    model["gguf_variants"] = d.get("variants") or []
    model["mmproj_bytes"] = d.get("mmproj_bytes")


@llm_bp.route("/models", methods=["GET"])
def list_models():
    manifest = get_models_dict(dict_return=True)
    media_default = media_default_state()
    output = []
    for key, model in manifest.items():
        model = update_model_status(model)
        mk = model.get("model_key") or key
        # Whether this model is offered in the media-intelligence chat dropdown.
        model["media"] = media_state(mk)
        # Whether this model is THE preselected default for the media chat.
        # Exactly one model carries media_default=True (or none, if unset).
        model["media_default"] = (mk == media_default)
        # GGUF: the model's real size = the one quant that serves, not the dir sum.
        _annotate_gguf_size(model, mk)
        output.append(model)

    return jsonify(output)


# ── Disk discovery (the console's "Discover models" button) ───────────────
# The discovery report (MODELS_DISCOVERY_PATH) is the persisted half of the
# registry; a walk taken while the storage mount was degraded can shrink it,
# so models "disappear" from /models while their files still sit on disk
# (observed 2026-07-04: report at 2 entries vs 108 model dirs). Nothing
# re-walks at runtime — download completion only re-READS the report — so
# this route is the recovery path. The walk enriches from hub metadata and
# can take minutes, hence the background thread + poll shape: POST to start,
# GET for state, re-fetch /models when running goes false.
#
# State lives in a FILE next to the discovery report, not a module global:
# the API runs gunicorn --workers 3, so per-process state answers the poll
# wrong 2/3 of the time (the comms-mirror lesson). A stale "running" left by
# a killed worker expires via _DISCOVER_STALE_S.
_discover_lock = threading.Lock()
_DISCOVER_STALE_S = 30 * 60


def _discover_state_path() -> str:
    from ....imports.src.constants.constants import MODELS_DISCOVERY_PATH
    return str(MODELS_DISCOVERY_PATH) + ".state.json"


def _read_discover_state() -> dict:
    state = {"running": False, "started_at": None, "finished_at": None,
             "found": None, "error": None}
    try:
        with open(_discover_state_path(), "r", encoding="utf-8") as fh:
            state.update(json.load(fh))
    except (OSError, ValueError):
        pass
    # Self-heal: a worker that died mid-sweep leaves running=true forever.
    if state.get("running") and state.get("started_at") and \
            _time.time() - state["started_at"] > _DISCOVER_STALE_S:
        state.update(running=False,
                     error="sweep did not finish (worker restarted?)")
    return state


def _write_discover_state(state: dict) -> None:
    path = _discover_state_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def _run_discovery(state: dict):
    try:
        refresh_registry(run_discovery=True)   # walk + save report + in-place update
        state["found"] = len(get_models_dict(dict_return=True))
    except Exception as exc:  # noqa: BLE001 — state must always resolve
        logger.warning("model discovery sweep failed: %s", exc)
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        state.update(running=False, finished_at=_time.time())
        try:
            _write_discover_state(state)
        except OSError as exc:
            logger.warning("could not persist discovery state: %s", exc)


@llm_bp.route("/models/discover", methods=["POST"])
def discover_models_start():
    with _discover_lock:
        state = _read_discover_state()
        if state["running"]:
            return jsonify({**state, "started": False}), 409
        state.update(running=True, started_at=_time.time(),
                     finished_at=None, found=None, error=None)
        _write_discover_state(state)
    threading.Thread(target=_run_discovery, args=(state,),
                     name="models-discover", daemon=True).start()
    return jsonify({**state, "started": True}), 202


@llm_bp.route("/models/discover", methods=["GET"])
def discover_models_state():
    return jsonify(_read_discover_state())


@llm_bp.route("/models/<model_key>", methods=["GET"])
def get_model(model_key):
    manifest = get_models_dict(dict_return=True)
    logger.info(manifest)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    model = manifest[model_key]
    return jsonify({"key": model_key, **model, **model_status(model)})


@llm_bp.route("/models/<model_key>/download", methods=["POST"])
def start_download(model_key):
    model = get_model_config(model_key,dict_return=True)
    if not model:
        abort(404, description="Unknown model key.")
    logger.info(model)
    body = request.get_json(silent=True) or {}
    job = job_store.create(model_key, kind="download", transport="web")
    start_cancellable_download(job, model, total_bytes=body.get("total_bytes"))
    return jsonify(job.to_legacy_dict())


@llm_bp.route("/jobs", methods=["GET"])
def list_jobs():
    # The store is shared with chat/inference jobs now — this surface is the
    # download manager's, so only download jobs belong on it (legacy wire
    # shape: queued/running/completed, error as string — ModelTable reads it).
    return jsonify([job.to_legacy_dict() for job in job_store.all()
                    if job.kind == "download"])


@llm_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = job_store.get(job_id)
    if not job:
        abort(404, description="Unknown job ID.")
    return jsonify(job.to_legacy_dict())


@llm_bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    return jsonify(cancel_download(job_id))


@llm_bp.route("/jobs/<job_id>/retry", methods=["POST"])
def retry_job(job_id):
    """Resume a failed/cancelled download from its partial files on disk."""
    return jsonify(retry_download(job_id))


@llm_bp.route("/llm/repos/download", methods=["POST"])
def download_repo():
    """Acquire any Hugging Face repo by hub_id without a pre-registered manifest entry.

    If register=True, the model is added to the manifest so it appears in the
    registry browser on the next refresh.
    """
    body = HFRepoDownloadRequest(**(request.get_json(silent=True) or {}))
    model = {
        "name": body.name or body.hub_id.split("/")[-1],
        "hub_id": body.hub_id,
        "framework": body.framework,
        "task": body.task,
        "filename": body.filename,
        "include": body.include,
    }

    if body.register:
        model_key, _ = upsert_model(settings.manifest_path, model)
    else:
        from ..functions.imports.utils.manifest import key_for_hub_id
        model_key = key_for_hub_id(body.hub_id)

    job = job_store.create(model_key, kind="download", transport="web")
    start_cancellable_download(job, model, total_bytes=body.total_bytes)
    return jsonify({**job.to_legacy_dict(), "model_key": model_key})


@llm_bp.route("/models/<model_key>", methods=["DELETE"])
def delete_model(model_key):
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")

    destination = route_destination(manifest.get(model_key))
    if not os.path.exists(destination):
        return jsonify({
            "deleted": False,
            "message": "Model is not installed.",
            "destination": str(destination),
        })

    shutil.rmtree(destination)
    return jsonify({"deleted": True, "destination": str(destination)})


@llm_bp.route("/models/<model_key>/prune", methods=["POST"])
def prune_model_route(model_key):
    """Remove a NOT-installed model's registry entry (a "ghost" row).

    Distinct from DELETE, which only removes downloaded files. Prune hides the
    catalog row itself (persisted in pruned_models.json) so it stops cluttering
    the listing. Refuses to prune a model that still has files on disk — Delete
    those first, so prune never silently orphans real data."""
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")

    destination = route_destination(manifest.get(model_key))
    if destination and os.path.exists(destination):
        return jsonify({
            "pruned": False,
            "message": "Model has files on disk — delete them before pruning.",
            "destination": str(destination),
        }), 409

    result = prune_model(model_key)
    return jsonify(result)


@llm_bp.route("/models/<model_key>/media", methods=["POST"])
def set_model_media_route(model_key):
    """Toggle whether a model is offered in the media-intelligence chat dropdown.

    Body: {"enabled": bool}. Curated default models start enabled; the store only
    keeps deviations from that default (see set_model_media)."""
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    body = request.get_json(silent=True) or {}
    enabled = body.get("enabled", body.get("media", True))
    return jsonify(set_model_media(model_key, enabled))


@llm_bp.route("/models/<model_key>/media-default", methods=["POST"])
def set_model_media_default_route(model_key):
    """Set (or clear) the single default media-chat model — the one the media
    chat dropdown preselects.

    Body: {"default": bool} (defaults to True). default=True makes this model THE
    default, replacing any previous one; default=False clears it only if this
    model is the current default. Single global value, persisted server-side
    (media_default.json) so every client agrees.

    Setting a model as default does NOT require it to be media-enabled."""
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    body = request.get_json(silent=True) or {}
    is_default = body.get("default", body.get("enabled", True))
    return jsonify(set_media_default(model_key, is_default))
