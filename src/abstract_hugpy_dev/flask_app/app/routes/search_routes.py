### routes/search_routes.py

from ..functions import *
search_bp, logger = get_bp("search_bp", __name__)

# Single declared token source for ALL Hugging Face calls in this module.
# token=False FORCES anonymous (HF public search/metadata needs no auth); a
# stale ~/.cache/huggingface/token cannot poison the call. Set HF_TOKEN only
# for gated/private repos or higher rate limits. Everything below goes through
# this `api` object so there is exactly one place a token can come from.
api = HfApi(token=os.getenv("HF_TOKEN") or False)


# ── helpers ───────────────────────────────────────────────────────────────
def _free_bytes() -> int | None:
    """Headroom on the filesystem where downloads actually land (MODELS_DIR)."""
    try:
        probe = MODELS_DIR if os.path.exists(MODELS_DIR) else "/"
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _context_length(hub_id: str, files) -> int | None:
    if not any(f.path == "config.json" for f in files):
        return None
    try:
        cfg_path = api.hf_hub_download(hub_id, "config.json")   # tiny, cached
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg.get("max_position_embeddings") or cfg.get("n_positions")
    except Exception:
        return None


def _license_of(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    try:
        return card.to_dict().get("license")
    except Exception:
        return getattr(card, "license", None)


# ── search: list available HF repos ────────────────────────────────────────
@search_bp.route("/search", methods=["GET"])
def search_models():
    q = request.args.get("q", "").strip()
    limit = request.args.get("limit", default=20, type=int)
    author = request.args.get("author")
    task = request.args.get("task")          # -> pipeline_tag
    library = request.args.get("library")    # -> filter
    sort = request.args.get("sort", default="last_modified")
    direction = request.args.get("direction", default=-1, type=int)  # client-side only now
    with_size = request.args.get("with_size", default="1") != "0"

    if sort not in ("last_modified", "downloads", "likes", "created_at"):
        sort = "last_modified"

    try:
        models = api.list_models(
            search=q or None,
            author=author,
            pipeline_tag=task or None,        # task is its own param
            filter=library or None,           # filter = library/tag
            sort=sort,
            limit=limit,
            full=False,
        )
    except Exception as exc:
        abort(502, description=f"Hugging Face request failed: {exc}")

    results = []
    for model in models:
        hub_id = model.modelId
        results.append(
            ModelSearchResult(
                hub_id=hub_id,
                author=getattr(model, "author", None),
                downloads=getattr(model, "downloads", None),
                likes=getattr(model, "likes", None),
                tags=getattr(model, "tags", []) or [],
                pipeline_tag=getattr(model, "pipeline_tag", None),
                library_name=getattr(model, "library_name", None),
                private=getattr(model, "private", None),
                total_bytes=model_size(hub_id) if with_size else None,   # #2
                last_modified=str(getattr(model, "last_modified", "")) or None,
                created_at=str(getattr(model, "created_at", "")) or None,
            ).model_dump()
        )

    return jsonify(results)



# ── spec: per-repo detail + install options (lazy, on row expand) ───────────
@search_bp.route("/hf/spec", methods=["GET"])
def hf_spec():
    hub_id = request.args.get("hub_id")
    if not hub_id:
        abort(400, description="hub_id is required.")

    try:
        info = api.model_info(hub_id, files_metadata=True)
    except Exception as exc:
        abort(502, description=f"Hugging Face request failed: {exc}")

    files = [FileSpec(path=s.rfilename, size=s.size) for s in info.siblings]
    total = sum(f.size for f in files if f.size) or None
    free = _free_bytes()

    task = getattr(info, "pipeline_tag", None) or "text-generation"
    options = resolve_options(hub_id, task, files, free)

    num_params = getattr(getattr(info, "safetensors", None), "total", None)

    spec = ModelSpec(
        hub_id=hub_id,
        license=_license_of(info),
        gated=getattr(info, "gated", None),
        last_modified=str(getattr(info, "last_modified", "")) or None,
        total_bytes=total,
        num_params=num_params,
        context_length=_context_length(hub_id, files),
        gguf_quants=[o.id.split(":", 1)[1] for o in options.options
                     if o.id.startswith("gguf:")],
        files=files,
    )

    return jsonify({"spec": spec.model_dump(), "options": options.model_dump()})


# ── Civitai — the checkpoint habitat, wired to the comfy drop-a-file flow ────
# Search is anonymous (their public API); DOWNLOAD streams the single-file
# checkpoint straight into <root>/checkpoints, where the registry sweep
# self-registers it as a comfy-<slug> model on the next read. Some files
# require a Civitai account token: set CIVITAI_API_TOKEN in the env (d-env).

_CIVITAI = "https://civitai.com/api/v1"
_CIVITAI_DL: dict = {}   # filename -> {done_bytes, total_bytes, status, error?}


def _checkpoints_dir() -> str:
    from ...imports.src.constants.constants import DEFAULT_ROOT
    d = os.path.join(DEFAULT_ROOT, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


@search_bp.route("/civitai/search", methods=["GET"])
def civitai_search():
    """Proxy Civitai model search into rows the console table can render."""
    import httpx
    params = {
        "types": "Checkpoint",
        "limit": max(1, min(int(request.args.get("limit", 20)), 50)),
        "sort": request.args.get("sort") or "Highest Rated",
        "nsfw": "false",
    }
    q = (request.args.get("query") or "").strip()
    if q:
        params["query"] = q
    base = (request.args.get("base") or "").strip()
    if base:
        params["baseModels"] = base
    try:
        r = httpx.get(_CIVITAI + "/models", params=params, timeout=20.0)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"civitai search failed: {exc}"}), 502

    rows = []
    for it in payload.get("items", []):
        v = (it.get("modelVersions") or [{}])[0]
        f = next((x for x in (v.get("files") or [])
                  if str(x.get("name", "")).endswith((".safetensors", ".ckpt"))
                  and (x.get("type") == "Model" or x.get("type") is None)), None)
        if not f or not f.get("downloadUrl"):
            continue
        stats = it.get("stats") or {}
        rows.append({
            "civitai_id": it.get("id"),
            "name": it.get("name"),
            "base_model": v.get("baseModel"),
            "version": v.get("name"),
            "filename": f.get("name"),
            "total_bytes": int((f.get("sizeKB") or 0) * 1024),
            "downloads": stats.get("downloadCount"),
            "likes": stats.get("thumbsUpCount"),
            "published": (v.get("publishedAt") or it.get("createdAt") or "")[:10],
            "download_url": f.get("downloadUrl"),
            "page_url": f"https://civitai.com/models/{it.get('id')}",
            # comfy-compat hint: our vanilla template speaks SD1.x/2.x/SDXL.
            "comfy_ready": str(v.get("baseModel") or "").startswith(("SD 1", "SD 2", "SDXL")),
        })
    return jsonify(rows)


@search_bp.route("/civitai/download", methods=["POST"])
def civitai_download():
    """Stream ONE checkpoint into <root>/checkpoints (background). The comfy
    sweep registers it automatically once the file lands — the whole install
    is this download. Operator-gated (writes to central storage)."""
    import threading
    body = request.get_json(silent=True) or {}
    url = str(body.get("download_url") or "").strip()
    filename = os.path.basename(str(body.get("filename") or "").strip())
    if not url.startswith("https://civitai.com/"):
        return jsonify({"error": "download_url must be a civitai.com URL"}), 400
    if not filename.endswith((".safetensors", ".ckpt")):
        return jsonify({"error": "filename must end in .safetensors/.ckpt"}), 400
    dest = os.path.join(_checkpoints_dir(), filename)
    if os.path.exists(dest):
        return jsonify({"ok": True, "already": True, "filename": filename})
    if _CIVITAI_DL.get(filename, {}).get("status") == "downloading":
        return jsonify({"ok": True, "already": False, "filename": filename,
                        "note": "download already in progress"})

    free = _free_bytes()
    want = int(body.get("total_bytes") or 0)
    if free is not None and want and free < want * 1.1:
        return jsonify({"error": f"not enough space: needs ~{want/1e9:.1f}GB, "
                        f"{free/1e9:.1f}GB free on central"}), 409

    def _pull():
        import httpx
        tmp = dest + ".part"
        headers = {}
        token = os.getenv("CIVITAI_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        st = _CIVITAI_DL[filename] = {"done_bytes": 0, "total_bytes": want,
                                      "status": "downloading"}
        try:
            with httpx.stream("GET", url, headers=headers, timeout=60.0,
                              follow_redirects=True) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(
                        "civitai requires an account token for this file — set "
                        "CIVITAI_API_TOKEN in d-env/env and restart")
                r.raise_for_status()
                st["total_bytes"] = int(r.headers.get("content-length") or want or 0)
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        st["done_bytes"] += len(chunk)
            os.replace(tmp, dest)
            st["status"] = "done"
            logger.info("civitai: %s landed in /checkpoints — the sweep "
                        "registers it on the next registry read", filename)
        except Exception as exc:  # noqa: BLE001
            st["status"] = "failed"
            st["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("civitai download of %s failed: %s", filename, exc)
            try:
                os.remove(tmp)
            except OSError:
                pass

    threading.Thread(target=_pull, daemon=True).start()
    return jsonify({"ok": True, "started": True, "filename": filename,
                    "note": "lands in /checkpoints; registers automatically"})


@search_bp.route("/civitai/downloads", methods=["GET"])
def civitai_downloads():
    """Live progress for in-flight civitai pulls (the UI polls this)."""
    return jsonify(_CIVITAI_DL)
