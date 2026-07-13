"""Model config discovery: the hugpy.json marker is the source of truth for
identity; disk contents fill descriptive fields; registry is fallback.
"""

from .huggingface_api import *
from .call_api import *
from .imports import *
from huggingface_hub.errors import HFValidationError


# ---------------------------------------------------------------------------
# Clean hub_id resolution — one rule, used everywhere. Strips leading slashes
# and rejects ids that aren't owner/repo so they never reach the HF validator.
# ---------------------------------------------------------------------------
def clean_hub_id(directory: str, fallback: str = "") -> str:
    hub_id = hub_id_for(directory, fallback) or ""
    return hub_id.strip("/")


def is_valid_repo_id(hub_id: str) -> bool:
    hub_id = (hub_id or "").strip("/")
    return bool(hub_id) and "/" in hub_id

def base_present(base_model: str) -> bool:
    if not base_model:
        return True   # not an adapter; nothing to require
    base_dir = route_destination(
        {"hub_id": base_model, "framework": "transformers",
         "primary_task": "text-generation"}
    )
    return os.path.isdir(base_dir) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(base_dir)
    )
# ---------------------------------------------------------------------------
# Registry lookup — exact / canonical match only
# ---------------------------------------------------------------------------
def get_config_from_folder(folder: str) -> Optional[dict]:
    target = _norm_folder(folder)
    if not target:
        return None
    for name, cfg in MODEL_REGISTRY.items():
        if normalize_folder(cfg.folder) == target:
            return cfg.to_dict()
    return None


# ---------------------------------------------------------------------------
# Main discovery walk
# ---------------------------------------------------------------------------
def get_all_configs(verbose: bool = False, get_code: bool = False,
                    get_files: bool = False, save_variables: bool = True
                    ) -> Dict[str, "ModelConfig"]:
    ALLCONFIGS: Dict[str, "ModelConfig"] = {}

    tasks_path = os.path.join(MODELS_HOME, "tasks.json")
    tasks_data = {}
    if os.path.isfile(tasks_path):
        tasks_data = safe_load_from_json(tasks_path) or {}

    model_dirs = exclude_dirs(get_model_dirs())
    shortnames = [d.replace(MODELS_HOME, "") for d in model_dirs]

    for directory in model_dirs:
        shortname = directory.replace(MODELS_HOME, "")
        folder = eatAll(shortname, "/")
        max_model_length = get_max_model_length(folder) or DEFAULT_MAX_TOKENS

        response_dir = get_response_dir(folder)
        _, resp_files = get_files_and_dirs(response_dir, allowed_exts=['.py'])
        if get_code and any(f.endswith('python_0.py') for f in resp_files):
            continue

        registry_cfg = get_config_from_folder(folder)
        marker = read_hugpy_marker(directory)

        guffs = get_guffs_in_dir(directory)
        configs = get_config_in_dir(directory)
        config_json = configs[0] if configs else None
        name = (marker or {}).get("name") or get_target_name(shortname, shortnames)

        hub_id = clean_hub_id(directory, folder)
        framework = (marker or {}).get("framework") or infer_framework(directory)
        tasks = (marker or {}).get("tasks") or tasks_data.get(name)
        primary_task = (marker or {}).get("primary_task") or (tasks[0] if tasks else None)

        discovered = {
            "name":             name,
            "hub_id":           hub_id,
            "folder":           folder,
            "framework":        framework,
            "filename":         (marker or {}).get("filename") or extract_gguf_filename(guffs, directory),
            "tasks":            tasks,
            "primary_task":     primary_task,
            "include":          (marker or {}).get("include"),
            "model_max_length": max_model_length,
            "port":             get_port(name),
            "host":             get_host(name),
        }

        merged, provenance = merge_disk_over_registry(discovered, registry_cfg)

        if merged.get("tasks") is None:
            merged["tasks"] = ["text-generation"]
            merged["primary_task"] = "text-generation"
            provenance["primary_task"] = "default"

        if verbose:
            print(f"[discover] {merged['name']}: {provenance}")

        name = merged["name"]
        if name and name not in ALLCONFIGS:
            ALLCONFIGS[name] = merged
            if get_code:
                call_and_code(model_config=merged, config_json=config_json, directory=directory)
        elif verbose:
            print(f"DUPLICATE NAME:\n{merged}")

        if get_files and name in ALLCONFIGS:
            ALLCONFIGS[name]['files'] = os.listdir(directory)

    if save_variables:
        if verbose:
            print(f"SAVING VARIABLES: {MODELS_DICT_PATH}")
        safe_dump_to_json(data=ALLCONFIGS, file_path=MODELS_DICT_PATH, indent=2)

    return ALLCONFIGS


# ---------------------------------------------------------------------------
# Resolvers — each independent; failure returns {} not an exception
# ---------------------------------------------------------------------------
def resolve_local_config(directory: str, hub_id: str) -> dict:
    path = os.path.join(directory, "config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "architectures":           cfg.get("architectures"),
        "model_type":              cfg.get("model_type"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "sliding_window":          cfg.get("sliding_window"),
        "rope_scaling":            cfg.get("rope_scaling"),
        "torch_dtype":             cfg.get("torch_dtype"),
        "vocab_size":              cfg.get("vocab_size"),
        # peft markers — present iff this is an adapter
        "peft_type":               cfg.get("peft_type"),
        "base_model":              cfg.get("base_model_name_or_path"),
    }


def resolve_local_tokenizer(directory: str, hub_id: str) -> dict:
    path = os.path.join(directory, "tokenizer_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            tk = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    raw = tk.get("model_max_length")
    if isinstance(raw, (int, float)) and 0 < raw < TOKENIZER_SENTINEL_THRESHOLD:
        return {"tokenizer_model_max_length": int(raw)}
    return {}


def resolve_hugpy_marker(directory: str, hub_id: str) -> dict:
    """The hugpy.json stamp is the DECLARED identity — authoritative. Without
    this resolver, a stamped dir with no config.json (e.g. the comfy sweep's
    layout dirs, GGUF single-file dirs) enriched to nothing and the registry
    minted a row with DEFAULT framework/task (the sd-turbo phantom rows)."""
    from ..src.constants.hugpy_marker import read_hugpy_marker
    marker = read_hugpy_marker(directory) or {}
    tasks = marker.get("tasks")
    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]
    return {k: v for k, v in {
        "name":         marker.get("name"),
        "framework":    marker.get("framework"),
        "tasks":        tasks,
        "primary_task": marker.get("primary_task"),
    }.items() if v is not None}


# Storage families that imply a framework. "misc" implies nothing — comfy and
# every other odd runtime shares it, so only the task segment is trusted there.
_LAYOUT_FAMILY_FRAMEWORK = {"gguf": "gguf", "transformers": "transformers"}


def resolve_layout_path(directory: str, hub_id: str) -> dict:
    """Last-resort attribution from the routed storage layout itself:
    MODELS_HOME/<family>/<task>/<owner>/<repo>. The path was WRITTEN from the
    model's real attribution at download/route time, so it beats guessing
    defaults — but it runs LAST, after marker/config/tokenizer/hub."""
    try:
        rel = os.path.relpath(directory, MODELS_HOME)
    except ValueError:
        return {}
    parts = rel.split(os.sep)
    if rel.startswith("..") or len(parts) < 3:
        return {}
    family, task = parts[0], parts[1]
    from ..src.constants.categories import HF_TASK_TO_TASKS, RUNNER_PAIRS
    known_tasks = {t for _fw, t in RUNNER_PAIRS}
    for _hf, _ts in HF_TASK_TO_TASKS.items():
        known_tasks.add(_hf); known_tasks.update(_ts)
    out: dict = {}
    if task in known_tasks:
        out["tasks"] = [task]
        out["primary_task"] = task
    fw = _LAYOUT_FAMILY_FRAMEWORK.get(family)
    if fw:
        out["framework"] = fw
    return out


def resolve_hub_model_info(directory: str, hub_id: str, api: HfApi) -> dict:
    # Skip ids that aren't owner/repo — the HF validator raises on these,
    # and a malformed id can't resolve anything anyway.
    if not is_valid_repo_id(hub_id):
        return {}

    try:
        info = api.model_info(hub_id, files_metadata=False)
    except (HfHubHTTPError, HFValidationError) as exc:
        logger.warning("hub_model_info skipped for %r: %s", hub_id, exc)
        return {}

    params = getattr(info.safetensors, "total", None) if info.safetensors else None
    auto_model = getattr(info.transformers_info, "auto_model", None) if info.transformers_info else None

    card = info.card_data
    def _card(key):
        if card is None:
            return None
        return card.get(key) if isinstance(card, dict) else getattr(card, key, None)

    languages = _card("language")
    if isinstance(languages, str):
        languages = [languages]

    return {
        "pipeline_tag":     info.pipeline_tag,
        "library_name":     info.library_name,
        "auto_model_class": auto_model,
        "parameter_count":  params,
        "license":          _card("license"),
        "gated":            bool(info.gated) if info.gated is not None else None,
        "languages":        languages,
        "tags":             info.tags,
    }


# ---------------------------------------------------------------------------
# Resolver chain
# ---------------------------------------------------------------------------
ResolverFn = Callable[[str, str], dict]


def build_resolver_chain(*, api: Optional[HfApi] = None,
                         use_hub: bool = True) -> List[Tuple[str, ResolverFn]]:
    # Order = priority (first non-None wins): declared identity (hugpy.json)
    # beats disk contents beats the Hub; the routed-path layout is the floor —
    # anything is better than minting default framework/task attribution.
    chain: List[Tuple[str, ResolverFn]] = [
        ("hugpy_marker",    resolve_hugpy_marker),
        ("local_config",    resolve_local_config),
        ("local_tokenizer", resolve_local_tokenizer),
    ]
    if use_hub:
        hub_api = api or HfApi()
        chain.append(("hub_model_info", lambda d, h: resolve_hub_model_info(d, h, hub_api)))
    chain.append(("layout_path", resolve_layout_path))
    return chain


def enrich(directory: str, hub_id: str,
           chain: List[Tuple[str, ResolverFn]]) -> Tuple[ModelMetadata, Dict[str, str]]:
    valid_fields = {f.name for f in dataclass_fields(ModelMetadata)}
    merged: Dict[str, Any] = {"hub_id": hub_id}
    sources: Dict[str, str] = {"hub_id": "input"}

    for source_name, fn in chain:
        try:
            partial = fn(directory, hub_id)
        except Exception as exc:
            logger.warning("resolver %s failed for %r: %s", source_name, hub_id, exc)
            continue
        for field_name, value in partial.items():
            if value is None or field_name not in valid_fields:
                continue
            if merged.get(field_name) is None:
                merged[field_name] = value
                sources[field_name] = source_name

    return ModelMetadata(**merged), sources


def _reap_orphaned_staging_quiet() -> None:
    """Sweep MODELS_HOME for dead-pid `.tmp-<pid>` download staging orphans
    (see imports/apis/download_models.py) before/while walking the tree — the
    discovery walk already visits every dir, so this is the natural hook and
    needs no new daemon. Never raises into discovery over a reaper hiccup."""
    try:
        from .download_models import reap_orphaned_staging
        removed = reap_orphaned_staging()
        if removed:
            logger.info("discovery: reaped %d orphaned download staging dir(s)", len(removed))
    except Exception as exc:  # noqa: BLE001
        logger.warning("discovery: staging reaper skipped (%s)", exc)


def discover_model(save_json: bool = True, verbose: bool = True, use_hub: bool = True):
    _reap_orphaned_staging_quiet()
    discovered = {}
    chain = build_resolver_chain(use_hub=use_hub)

    for directory in get_model_dirs():
        file_parts = get_file_parts(directory)
        folder = directory[len(MODELS_HOME):]
        name = file_parts.get("basename")
        parent_dirname = file_parts.get("parent_dirname")

        hub_id = clean_hub_id(directory, directory[len(parent_dirname) + 1:])

        meta, meta_sources = enrich(directory, hub_id, chain)
        discovered[name] = {"name": name, "hub_id": hub_id, "folder": folder}
        discovered[name].update({
            "pipeline_tag":            meta.pipeline_tag,
            "library_name":            meta.library_name,
            "auto_model_class":        meta.auto_model_class,
            "architectures":           meta.architectures,
            "model_type":              meta.model_type,
            "max_position_embeddings": meta.max_position_embeddings,
            "model_max_length":        meta.tokenizer_model_max_length
                                       or meta.max_position_embeddings
                                       or DEFAULT_MAX_TOKENS,
            "parameter_count":         meta.parameter_count,
            "license":                 meta.license,
            "gated":                   meta.gated,
            "languages":               meta.languages,
        })
        if verbose:
            print(f"[enrich] {hub_id}: {meta_sources}")

    if save_json:
        safe_dump_to_file(data=discovered, file_path=MODELS_DICT_PATH)
    return discovered


def discover_models(save_json: bool = True, verbose: bool = True, use_hub: bool = True):
    """Walk the model tree and enrich each dir. Records the real on-disk
    location — discovery already walked it; don't throw it away.

    Model keys are the bare directory basename (e.g. ``Qwen2.5-3B-Instruct-GGUF``)
    UNLESS two different owners ship a repo of the same name — e.g. both
    ``unsloth/Qwen3-Coder-Next-GGUF`` and ``Qwen/Qwen3-Coder-Next-GGUF``. Only
    the *colliding* names are qualified with their one-dir-up owner
    (``<owner>/<name>``) so each physical copy survives as its own registry row;
    unique names stay bare so the common case needs no owner prefix. Without
    this, two distinct models with the same basename collapse onto one key and
    one silently overwrites the other — taking its dir/filename/size with it
    (the "confused local map": only one shows, with the wrong gguf size)."""
    _reap_orphaned_staging_quiet()
    chain = build_resolver_chain(use_hub=use_hub)

    rows = []
    for directory in get_model_dirs():
        file_parts = get_file_parts(directory)
        name = file_parts.get("basename")
        parent_dirname = file_parts.get("parent_dirname")
        owner = os.path.basename(os.path.dirname(directory))    # the one dir up

        hub_id = clean_hub_id(directory, directory[len(parent_dirname) + 1:])
        meta, meta_sources = enrich(directory, hub_id, chain)

        row = meta.to_dict()
        # Declared name (hugpy.json) wins; the bare basename is the fallback —
        # a marker-stamped "comfy-sd-turbo" must not degrade to "sd-turbo" and
        # collide with the staple's display name.
        row["name"] = row.get("name") or name
        row["dir"] = directory                                  # absolute, ground truth
        try:
            row["folder"] = os.path.relpath(directory, MODELS_HOME)   # MODELS_HOME-relative
        except ValueError:
            row["folder"] = directory
        # Resolve the model file via the ONE canonical, shard-aware finder
        # (recursive; prefers the 00001-of shard, excludes the mmproj projector)
        # so split-gguf models nested in a subdir get a real, loadable filename.
        from ..config.main import get_gguf_file
        gguf = get_gguf_file(directory, None)
        if gguf:
            row["filename"] = os.path.relpath(gguf, directory)

        # De-tasking (operator-locked 2026-07-11): task is DERIVED metadata that
        # lives in the marker/registry/hub — never baked as truth from the path.
        # In the flat layout there IS no task segment, so a dir with no marker
        # and no hub/config task yields task=None here. Flag such a dir
        # `needs_classification` (rather than silently committing a path guess)
        # so the console can prompt for a real task. A task sourced ONLY from the
        # legacy layout path (a not-yet-migrated task dir) is likewise flagged —
        # its path-derived task is provisional until the marker confirms it.
        task_src = meta_sources.get("primary_task") or meta_sources.get("tasks")
        if row.get("primary_task") is None or task_src in (None, "layout_path"):
            row["needs_classification"] = True

        rows.append((name, owner, row))
        if verbose:
            print(f"[enrich] {hub_id}: {meta_sources}")

    # qualify-on-collision: bare name when unique, else "<owner>~<name>".
    # The delimiter is `~` (URL-unreserved, not %-encoded, systemd-unit safe) NOT
    # `/` — a slash in a model_key falls through every `<model_key>`-in-path route
    # to the SPA catch-all (returns index.html -> the UI's "JSON.parse: unexpected
    # character at line 1 column 1").
    name_counts = {}
    for name, _owner, _row in rows:
        name_counts[name] = name_counts.get(name, 0) + 1
    discovered = {}
    for name, owner, row in rows:
        key = name if name_counts[name] == 1 else f"{owner}~{name}"
        if key in discovered:        # same owner+name twice (rare) — keep both, distinct
            suffix = 2
            while f"{key}~{suffix}" in discovered:
                suffix += 1
            key = f"{key}~{suffix}"
        discovered[key] = row

    if save_json:
        # Shrink guard: a walk taken while the storage mount is degraded finds
        # a fraction of the catalog; saving that would "disappear" models whose
        # files still exist (2026-07-04: report collapsed to 2 entries vs 108
        # dirs on disk). Keep the outgoing report as .prev whenever this walk
        # found less than half of what the last one did — recovery is then a
        # re-walk on healthy disk (POST /models/discover) or restoring .prev.
        try:
            import json as _json, shutil as _shutil
            with open(MODELS_DISCOVERY_PATH, "r", encoding="utf-8") as _fh:
                _prior = _json.load(_fh)
            if len(_prior) > 4 and len(discovered) < 0.5 * len(_prior):
                _shutil.copy2(MODELS_DISCOVERY_PATH, MODELS_DISCOVERY_PATH + ".prev")
                print(f"[discover] WARNING: walk found {len(discovered)} models "
                      f"but the prior report had {len(_prior)} — storage mount "
                      f"degraded? Prior report kept at "
                      f"{MODELS_DISCOVERY_PATH}.prev")
        except (OSError, ValueError):
            pass
        safe_dump_to_file(data=discovered, file_path=MODELS_DISCOVERY_PATH)
    return discovered
