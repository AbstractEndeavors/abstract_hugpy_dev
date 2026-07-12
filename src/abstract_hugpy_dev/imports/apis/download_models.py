#!/usr/bin/env python3
from .imports import *
import threading
import shutil as _shutil
_REPORT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Atomic provisioning (operator-locked 2026-07-11): a download lands in a
# per-pid STAGING sibling and is only renamed onto its final path once complete,
# so a partially-transferred dir can NEVER sit at a resolvable model path (the
# "redownload / phantom-partial" class of bug). Temp cleanup is EXEMPT from the
# never-delete rule — a staging temp is our own in-flight scratch, never catalog
# data.
# ---------------------------------------------------------------------------
def _staging_dir(dest: str) -> str:
    """The atomic staging path for a download: a per-pid sibling of the final
    dest. Same-pid retries reuse it (HF resume); a crashed pid leaves a
    ``.tmp-<pid>`` the next run discards."""
    return f"{dest}.tmp-{os.getpid()}"


def _promote_staged(staged: str, dest: str) -> str:
    """Atomically move a COMPLETED staging dir onto its final path. Same
    filesystem -> os.rename (instant). If dest already exists (resume/merge),
    move staged's entries in (os.replace) and drop the emptied staging dir."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        os.rename(staged, dest)
        return dest
    for name in os.listdir(staged):
        src = os.path.join(staged, name)
        dst = os.path.join(dest, name)
        if os.path.isdir(src) and os.path.isdir(dst):
            for sub in os.listdir(src):
                os.replace(os.path.join(src, sub), os.path.join(dst, sub))
        else:
            os.replace(src, dst)
    _shutil.rmtree(staged, ignore_errors=True)   # temp cleanup: EXEMPT from never-delete
    return dest


def _discard_staged(staged: str) -> None:
    """Remove a failed/partial staging dir. Temp cleanup — EXEMPT from the
    never-delete rule (in-flight scratch, never catalog data)."""
    _shutil.rmtree(staged, ignore_errors=True)

def record_downloaded_model(model: dict, dest: str = None, root: str = DEFAULT_ROOT,
                            report_path: str = None) -> dict:
    """Write one downloaded model into the discovery report so the registry —
    which derives from that report — shows it immediately, no full re-walk.

    Records the real on-disk dir (dest) so folder resolution is exact instead
    of a routed guess. Locked: monitor threads for concurrent downloads all
    read-modify-write the same file."""
    report_path = report_path or MODELS_DISCOVERY_PATH
    dest = dest or route_destination(model, root)
    hub_id = model.get("hub_id")
    name = model.get("name") or (hub_id.split("/")[-1] if hub_id else None)
    if not name:
        return {}

    primary = model.get("primary_task") or model.get("task")
    tasks = model.get("tasks") or ([primary] if primary else None)
    record = {
        "hub_id": hub_id,
        "framework": model.get("framework"),
        "primary_task": primary,
        "tasks": tasks,
        "filename": model.get("filename"),
        "include": model.get("include"),
        "model_max_length": model.get("model_max_length"),
        "dir": dest,
        "folder": os.path.relpath(dest, MODELS_HOME) if dest.startswith(MODELS_HOME) else dest,
    }
    with _REPORT_LOCK:
        report = safe_load_from_json(report_path) or {}
        report[name] = {**report.get(name, {}),
                        **{k: v for k, v in record.items() if v is not None}}
        safe_dump_to_file(data=report, file_path=report_path)
    return report[name]






def _clean_repo_id(hub_id: str) -> str:
    """Strip storage-path routing (family/task/...) back to owner/repo.

    cfg.hub_id sometimes carries the full storage path
    (gguf/text-generation/owner/repo) because discovery recorded the path as
    the id. HF wants just owner/repo. Drop leading family/task segments.
    """
    parts = hub_id.strip("/").split("/")
    FAMILIES = {"gguf", "transformers", "misc", "datasets", "models"}
    while len(parts) > 2 and parts[0] in FAMILIES:
        parts = parts[1:]                       # drop family
        if parts and parts[0] not in FAMILIES:
            parts = parts[1:]                   # drop the task that followed
    return "/".join(parts)

def _stamp(destination: str, key: str, model: dict[str, Any]) -> None:
    """Write hugpy.json into the destination so the model self-describes for
    discovery — identity no longer has to be inferred from the path later."""
    try:
        write_hugpy_marker(
            destination,
            hub_id=model.get("hub_id"),
            name=model.get("name") or key,
            framework=model.get("framework"),
            tasks=model.get("tasks"),
            primary_task=model.get("primary_task"),
            filename=model.get("filename"),
            include=model.get("include"),
            source="download",
        )
    except OSError as exc:
        print(f"  [warn] could not write hugpy.json: {exc}")


def _fetch_mmproj_sidecars(repo_id: str, destination: str) -> None:
    """Best-effort pull of any mmproj/projector GGUF sidecars from ``repo_id``.

    Vision GGUFs are useless for images without their CLIP projector, and the
    sidecar is routinely overlooked when a model is registered by single
    ``filename`` — the model then loads fine but silently answers text-blind.
    So every GGUF download tries these patterns; repos without sidecars (the
    common, text-only case) match nothing and this is a no-op. Never raises:
    a missing projector should surface at serve time (probe/vision honesty),
    not break the weights download that just succeeded."""
    from ..src.utils import find_mmproj
    try:
        if find_mmproj(destination):
            return  # already have one beside the weights
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*mmproj*.gguf", "*mm-proj*.gguf",
                            "*mm_proj*.gguf", "*projector*.gguf"],
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
        got = find_mmproj(destination)
        if got:
            print(f"  auto-downloaded vision projector sidecar: {os.path.basename(got)}")
    except Exception as exc:
        print(f"  mmproj sidecar check skipped ({type(exc).__name__}: {exc})")


def download_one(model: dict[str, Any],root: str=None,model_key=None, dry_run: bool = False) -> None:
    
    hub_id = model.get("hub_id")
    framework = model.get("framework")
    filename = model.get("filename")
    include = model.get("include")
    if hub_id and not model_key:
        model_key = hub_id.split('/')[-1]
    root = root or DEFAULT_ROOT
    if not hub_id:
        print(f"[SKIP] {model_key}: missing hub_id")
        return

    # New work lands FLAT (task segment retired). If a COMPLETE copy already
    # exists anywhere (flat OR a legacy task path), don't re-download it — the
    # read-through resolver is what stops "redownload models that are ready".
    destination = flat_destination(model, root)
    existing = resolve_model_dir(model, root)
    repo_id, subfolder_from_hub = split_hub_id(hub_id)

    print()
    print(f"[MODEL] {model_key}")
    print(f"  framework: {framework}")
    print(f"  task:      {model.get('primary_task')}")
    print(f"  repo_id:   {repo_id}")
    if subfolder_from_hub:
        print(f"  subfolder: {subfolder_from_hub}")
    print(f"  dest:      {destination}")

    if dry_run:
        return
    if existing:
        print(f"  already present (complete) at {existing} — skipping")
        return

    # ATOMIC: download into a per-pid staging sibling, promote on success only.
    staged = _staging_dir(destination)
    os.makedirs(staged, exist_ok=True)
    try:
        # GGUF / single-file
        if framework == "gguf":
            if filename:
                hf_hub_download(
                    repo_id=repo_id, filename=filename, subfolder=subfolder_from_hub,
                    local_dir=staged, local_dir_use_symlinks=False)
                print(f"  downloaded file: {filename}")
                _fetch_mmproj_sidecars(repo_id, staged)
            elif include:
                snapshot_download(
                    repo_id=repo_id, allow_patterns=include,
                    local_dir=staged, local_dir_use_symlinks=False)
                print(f"  downloaded pattern: {include}")
                _fetch_mmproj_sidecars(repo_id, staged)
            else:
                snapshot_download(
                    repo_id=repo_id, local_dir=staged, local_dir_use_symlinks=False)
                print("  downloaded full snapshot")
        elif model.get("task") == "dataset" or model.get("primary_task") == "dataset":
            snapshot_download(
                repo_id=repo_id, repo_type="dataset",
                local_dir=staged, local_dir_use_symlinks=False)
            print("  downloaded dataset snapshot")
        else:
            # Transformers / full repo
            allow_patterns = include if include else None
            snapshot_download(
                repo_id=repo_id, allow_patterns=allow_patterns,
                local_dir=staged, local_dir_use_symlinks=False)
            print(f"  downloaded snapshot with pattern: {allow_patterns}"
                  if allow_patterns else "  downloaded full snapshot")
        _stamp(staged, model_key, model)
        _promote_staged(staged, destination)
    except BaseException:
        # A partial/aborted pull must never sit at a resolvable path — discard
        # the staging temp (exempt from never-delete) and re-raise.
        _discard_staged(staged)
        raise


def download_dict_models() -> None:
    parser = argparse.ArgumentParser(description="Auto-distribute LLM downloads by framework/task.")
    parser.add_argument("manifest", type=str, help="Path to model manifest JSON.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="LLM storage root.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional model keys to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print destinations without downloading.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(args.root, "cache", "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.root, "cache", "huggingface", "hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(args.root, "cache", "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(args.root, "cache", "pip"))

    # args.manifest is a str -> use open(), not args.manifest.open()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key, model in manifest.items():
        if args.only and key not in args.only:
            continue
        try:
            download_one(key, model, args.root, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[ERROR] {key}: {exc}")




def _fp16_ignore_patterns(repo_id: str) -> "list[str] | None":
    """Item 4 variant filter: for a DIFFUSERS pipeline repo whose weights ship
    fp16 twins (``x.fp16.safetensors`` beside ``x.safetensors``), return
    ignore_patterns that skip the full-precision twins and the alt runtimes
    (onnx/openvino/flax) — sdxl-turbo drops from 55GB to ~7GB. Returns None
    (no filtering) for non-diffusers repos or repos without fp16 variants, so
    a model that only ships full precision still downloads completely."""
    from huggingface_hub import HfApi
    files = set(HfApi().list_repo_files(repo_id))
    if "model_index.json" not in files:
        return None                      # not a diffusers pipeline — untouched
    twins = [f for f in files
             if f.endswith(".safetensors") and ".fp16." not in f
             and f.replace(".safetensors", ".fp16.safetensors") in files]
    if not twins:
        return None                      # no fp16 variant — pull everything
    ignore = sorted(twins)
    # Alt runtimes ride along in these repos and are never used by our loader.
    ignore += ["*.onnx", "*.onnx_data", "*.msgpack", "*openvino*", "*.ckpt"]
    # .bin twins of safetensors weights (legacy format duplicates).
    ignore += [f for f in files
               if f.endswith(".bin")
               and f.replace(".bin", ".safetensors") in files]
    return ignore


def ensure_model(key: str, root: str = DEFAULT_ROOT) -> str:
    cfg = get_model_config(key)

    # Clean identity FIRST — both the repo HF pulls from and the path we
    # route to must use owner/repo, not the storage path.
    repo_id = _clean_repo_id(cfg.hub_id)

    # Route to framework/primary_task/hub_id using route_destination, the same
    # function download_one uses — single source of truth for layout. Build a
    # routing dict whose hub_id is the CLEAN id so we don't re-nest the prefix.
    routing = {
        "hub_id":       repo_id,
        "name":         getattr(cfg, "name", None) or key,
        "framework":    getattr(cfg, "framework", None),
        "primary_task": getattr(cfg, "primary_task", None) or "misc",
    }
    # Read-through: a COMPLETE copy anywhere (flat OR any legacy task path)
    # short-circuits — this is what stops re-downloading ready models.
    existing = resolve_model_dir(routing, root, cfg=cfg)
    if existing:
        # Read-through the box-local NVMe HOT-CACHE tier for the steady-state
        # load: a complete hot copy is served, else the shared path unchanged
        # (a background promotion is scheduled -> the NEXT call is NVMe-hot).
        # Env-gated (HUGPY_HOT_CACHE_ROOT); byte-identical when unset. This is
        # the transformers/diffusers dir chokepoint that resolve_model_source
        # does NOT see (imagegen/embed/keywords/summarizers load via
        # ensure_model). A just-downloaded model (below) is intentionally NOT
        # promoted inline — it promotes on its next call, so a fresh multi-GB
        # pull is never immediately re-copied. Never raises into a load.
        try:
            from ...managers.serve import hot_cache
            return hot_cache.use(existing)
        except Exception:  # noqa: BLE001
            return existing

    # New work lands FLAT. ATOMIC: download into a per-pid staging sibling and
    # promote onto the final path only once complete — a partial pull can never
    # sit at a resolvable model dir.
    path = flat_destination(routing, root)
    staged = _staging_dir(path)
    os.makedirs(staged, exist_ok=True)

    repo_id, subfolder = split_hub_id(repo_id)  # handle owner/repo/subfolder
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": staged,
        "local_dir_use_symlinks": False,
    }
    if subfolder:
        download_kwargs["subfolder"] = subfolder
    if getattr(cfg, "include", None):
        download_kwargs["allow_patterns"] = cfg.include
    elif getattr(cfg, "filename", None):
        # Single-file model (GGUF quants live N-per-repo): pull just that file,
        # mirroring download_one's llama_cpp branch — never the whole repo.
        download_kwargs["allow_patterns"] = [cfg.filename]
    else:
        # Daylight item 4: diffusers pipelines often ship EVERY precision +
        # runtime (sdxl-turbo: a 55GB repo whose fp16 pipeline is ~7GB). When
        # the repo IS a diffusers pipeline and fp16 twins exist, skip the
        # full-precision twins and the alt runtimes. Introspection failure ->
        # full snapshot (correct, just big) — never a broken model.
        try:
            ignore = _fp16_ignore_patterns(repo_id)
            if ignore:
                download_kwargs["ignore_patterns"] = ignore
                print(f"  fp16 variant filter: skipping {len(ignore)} "
                      f"full-precision/alt-runtime pattern(s)")
        except Exception:
            pass
    try:
        snapshot_download(**download_kwargs)

        # Vision GGUFs need their mmproj projector beside the weights; a
        # filename/include entry that omits it produces a text-blind model.
        # No-op for repos without sidecars (the common text-only case).
        if getattr(cfg, "framework", None) == "gguf":
            _fetch_mmproj_sidecars(repo_id, staged)

        write_hugpy_marker(
            staged,
            hub_id=repo_id if not subfolder else f"{repo_id}/{subfolder}",
            name=getattr(cfg, "name", None) or key,
            framework=getattr(cfg, "framework", None),
            tasks=getattr(cfg, "tasks", None),
            primary_task=getattr(cfg, "primary_task", None),
            filename=getattr(cfg, "filename", None),
            include=getattr(cfg, "include", None),
            source="download",
        )
        _promote_staged(staged, path)
        return path
    except BaseException as e:
        # A partial/aborted pull must never sit at a resolvable path — discard
        # the staging temp (exempt from never-delete) and report the failure.
        _discard_staged(staged)
        logger.info(f"Download Failed: {e}")
def ensure_models(models_dict_path=None):
    models_dict = get_models_dict(models_dict_path=models_dict_path)
    for model,values in models_dict.items():
        ensure_model(model)
