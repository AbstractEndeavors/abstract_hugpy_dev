"""models_config.py — the registry.

MODELS is the authoritative base: the real, curated models. Discovery finds
everything else on disk (test downloads) and appends it at build time. Staples
are never overwritten — a discovered row is skipped if its model_key OR its
cleaned hub_id already belongs to a staple (prevents same-path collisions like
Falconsai-text-summarization vs a discovered text_summarization).

Build order:
    MODELS (curated)  +  discovery report (derived)  ->  ModelConfig registry

Import is cheap: it merges MODELS with whatever discovery report already exists
on disk. To re-walk the model tree (HF metadata, network), call
refresh_registry() explicitly — e.g. on hugpy module startup.
"""

from .imports import *

logger = get_logFile(__name__)


# ===========================================================================
# Base registry — the real models. Authoritative.
# ===========================================================================
# Stock fleet: exactly one default model per task (a model may serve its whole
# task group). Efficiency-first, non-gargantuan picks — ~17GB all-in. Bigger
# siblings (flan-t5-xl, whisper-large-v3, sdxl-turbo, Qwen2.5-VL-7B, LED-16384,
# gte-large) stay available as opt-in installs via discovery; they're just not
# staples. Every DEFAULT_* constant in constants.py points at a key below.
MODELS = {
    "Qwen2.5-3B-Instruct-GGUF": {
        "model_max_length": 32768, "include": None, "name": "Qwen2.5-3B-Instruct-GGUF",
        "framework": "llama_cpp", "hub_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "folder": "Qwen/Qwen2.5-3B-Instruct-GGUF", "tasks": ["text-generation"],
        "primary_task": "text-generation", "port": None,
    },

    # Torch-free vision: llama.cpp serves the LM gguf with the mmproj CLIP
    # projector via --mmproj (find_mmproj auto-discovers it beside the model).
    # `include` pulls BOTH files; `filename` is the LM so get_gguf_file resolves
    # it and skips the projector. This is DEFAULT_VISION_MODEL — preferred over
    # the transformers variant on CPU/phone workers that can't install torch.
    "Qwen2.5-VL-3B-Instruct-GGUF": {
        "model_max_length": 32768,
        "include": ["Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                    "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"],
        "name": "Qwen2.5-VL-3B-Instruct-GGUF",
        "framework": "llama_cpp", "hub_id": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "folder": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        "tasks": ["image-text-to-text", "text-generation"],
        "primary_task": "image-text-to-text", "port": None,
    },
    "whisper-large-v3-turbo": {
        "model_max_length": 448, "include": None, "name": "whisper-large-v3-turbo",
        "framework": "transformers", "hub_id": "openai/whisper-large-v3-turbo", "filename": None,
        "folder": "openai/whisper-large-v3-turbo", "tasks": ["automatic-speech-recognition"],
        "primary_task": "automatic-speech-recognition", "port": None,
    },
    "flan-t5-large": {
        "model_max_length": 1024, "include": None, "name": "flan-t5-large",
        "framework": "transformers", "hub_id": "google/flan-t5-large", "filename": None,
        "folder": "google/flan-t5-large", "tasks": ["text-summarization", "text2text-generation"],
        "primary_task": "text-summarization", "port": None,
    },
    "all-minilm-l6-v2": {
        "model_max_length": 512, "include": None, "name": "all-minilm-l6-v2",
        "framework": "transformers", "hub_id": "sentence-transformers/all-minilm-l6-v2",
        "filename": None, "folder": "sentence-transformers/all-minilm-l6-v2",
        "tasks": ["feature-extraction", "sentence-similarity", "keyword-extraction"],
        "primary_task": "feature-extraction", "port": None,
    },
    "sd-turbo": {
        "model_max_length": 77, "include": None, "name": "sd-turbo",
        "framework": "transformers", "hub_id": "stabilityai/sd-turbo", "filename": None,
        "folder": "stabilityai/sd-turbo", "tasks": ["text-to-image"],
        "primary_task": "text-to-image", "port": None,
    },
}


# ===========================================================================
# Derivation — discovery/manifest row -> ModelConfig-ready dict.
# Pure; no torch, no runner-stack import (so building the registry never drags
### the inference stack in). RUNNER_PAIRS mirrors FRAMEWORK_RUNNERS statically.
# ===========================================================================
DEFAULT_MAX_TOKENS_LOCAL = DEFAULT_MAX_TOKENS

_FAMILIES = {"gguf", "transformers", "misc", "datasets", "models"}


def _clean_repo_id(hub_id):
    """Strip storage-path leakage (gguf/text-generation/owner/repo, leading
    slashes) back to owner/repo — the only shape HF and routing accept."""
    parts = (hub_id or "").strip("/").split("/")
    while len(parts) > 2 and parts[0] in _FAMILIES:
        parts = parts[1:]
        if parts and parts[0] not in _FAMILIES:
            parts = parts[1:]
    return "/".join(parts)

def base_present(base_model: str) -> bool:
    """True if a PEFT adapter's base model is actually on disk.

    Non-adapters (base_model falsy) pass trivially. An adapter passes only
    if route_destination's base dir exists AND holds real weights — a bare
    dir with just a config doesn't count.
    """
    if not base_model:
        return True
    base_dir = route_destination(
        {"hub_id": base_model,
         "framework": "transformers",
         "primary_task": "text-generation"}
    )
    if not os.path.isdir(base_dir):
        return False
    try:
        return any(
            f.endswith(".safetensors") or f.endswith(".bin")
            for f in os.listdir(base_dir)
        )
    except OSError:
        return False


_SEQ2SEQ = {"t5", "led", "bart", "pegasus", "mbart", "mt5", "longt5"}
_EMBED   = {"bert", "new", "roberta", "mpnet", "nomic_bert"}
_ASR     = {"whisper"}
_VISION  = {"qwen2_5_vl", "minicpmv4_6", "mllama", "idefics3", "internvl"}

def _safe_path_part(value):
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")

def _runtime_folder(framework, hub_id, include=None, filename=None):
    framework = (framework or "").lower().strip()
    if framework == "llama_cpp": return "gguf"
    if filename and filename.lower().endswith(".gguf"): return "gguf"
    if include:
        pats = include if isinstance(include, list) else [include]
        if any("gguf" in p.lower() for p in pats): return "gguf"
    return "transformers" if framework == "transformers" else "misc"

def _routed_folder(framework, task, hub_id, filename=None, include=None):
    """Predicted MODELS_HOME-relative folder — only used when the model isn't
    on disk yet, so there's no real dir to record."""
    if task == "dataset": return None
    return f"{_runtime_folder(framework, hub_id, include, filename)}/{_safe_path_part(task)}/{_safe_path_part(hub_id)}"

def _resolve_folder(row, framework, task, hub_id, filename, include):
    """Real dir wins; then an already-routed folder; then a prediction."""
    abs_dir = row.get("dir")
    if abs_dir and MODELS_HOME:
        rel = os.path.relpath(abs_dir, MODELS_HOME)
        if not rel.startswith(".."):
            return rel
    f = row.get("folder")
    if f and len(f.strip("/").split("/")) >= 3:   # looks like runtime/task/owner/repo already
        return f.strip("/")
    return _routed_folder(framework, task, hub_id, filename, include) or hub_id
def _derive_framework(name, hub_id, row):
    if row.get("framework"):
        return row["framework"]
    blob = f"{name} {hub_id}".lower()
    tags = [t.lower() for t in (row.get("tags") or [])]
    return "llama_cpp" if ("gguf" in blob or "gguf" in tags) else "transformers"


def _derive_tasks(framework, row):
    tasks = row.get("tasks")
    if tasks:
        return tasks if isinstance(tasks, list) else [tasks]
    if framework == "llama_cpp":
        # Vision GGUFs (a VL model + its mmproj sidecar) ride the chat path but
        # MUST advertise image-text-to-text, or they're dropped from that task
        # ("has no runner") and only ever answer text. Detect by model_type or a
        # "VL" marker in the id/folder/filename.
        _blob = " ".join(str(row.get(k) or "") for k in ("hub_id", "folder", "filename", "name")).lower()
        _mtype = (row.get("model_type") or "").lower()
        if _mtype in _VISION or any(s in _blob for s in ("-vl-", "-vl.", "vl-instruct", "qwen2.5-vl", "qwen2_5_vl")):
            return ["image-text-to-text", "text-generation"]
        return ["text-generation"]                      # all the gguf runner serves
    pt = row.get("pipeline_tag") or row.get("primary_task") or row.get("task")
    if pt in HF_TASK_TO_TASKS:
        return list(HF_TASK_TO_TASKS[pt])
    m = (row.get("model_type") or "").lower()
    if m in _SEQ2SEQ: return ["text-summarization", "text2text-generation"]
    if m in _EMBED:   return ["feature-extraction", "sentence-similarity"]
    if m in _ASR:     return ["automatic-speech-recognition"]
    if m in _VISION:  return ["image-text-to-text", "text-generation"]
    return ["text-generation"]                          # conservative floor


def derive_model_config_row(name, row):
    """One discovery/manifest row -> ModelConfig-ready dict, or (None, reason)."""
    hub_id = _clean_repo_id(row.get("hub_id") or row.get("folder") or name)
    if not hub_id or "/" not in hub_id:
        return None, f"unusable hub_id {row.get('hub_id')!r}"

    # PEFT adapter gate: an adapter is a delta on a base model. It needs
    # base_model_name_or_path, and that base must be on disk to serve.
    # base-less or base-absent adapters are dropped here so they never
    # enter the registry and detonate inside from_pretrained on first use.
    peft_base = row.get("base_model")
    if peft_base and not base_present(peft_base):
        return None, f"peft adapter base {peft_base!r} not on disk; acquire it first"

    framework = _derive_framework(name, hub_id, row)
    tasks = _derive_tasks(framework, row)
    primary = row.get("primary_task") if row.get("primary_task") in tasks else tasks[0]
    no_runner = [t for t in tasks if (framework, t) not in RUNNER_PAIRS]
    on_disk = bool(row.get("dir"))
    if no_runner and not on_disk:
        # Nothing can serve it AND it isn't downloaded → don't surface a dead
        # entry that can neither run nor be opened.
        return None, f"({framework},{no_runner}) has no runner"
    # A downloaded model ALWAYS appears on the models tab, even when no runner
    # can serve its task(s) — it's kept and flagged `serveable: False` (with the
    # offending tasks) so the UI can show it as present-but-unservable and the
    # serve path can refuse cleanly, instead of the model silently vanishing.
    folder = _resolve_folder(row, framework, primary, hub_id,
                             row.get("filename"), row.get("include"))
    return {
        "name": row.get("name") or name, "model_key": name,
        "hub_id": hub_id, "folder": folder,
        "dir": row.get("dir"),
        "framework": framework, "tasks": tasks, "primary_task": primary,
        "base_model": peft_base,                 # None for ordinary models
        "model_max_length": row.get("model_max_length")
            or row.get("tokenizer_model_max_length")
            or row.get("max_position_embeddings") or DEFAULT_MAX_TOKENS_LOCAL,
        "filename": row.get("filename"), "include": row.get("include"),
        "port": row.get("port"), "host": row.get("host"),
        "serveable": not no_runner,
        "unserveable_tasks": no_runner,
    }, None

def _absorb_disk(staple, disc):
    """Disk facts from a discovered row override a staple's hand-written guesses."""
    for k in ("dir", "folder", "filename"):
        if disc.get(k):
            staple[k] = disc[k]

def merge_discovery_into_models(discovery, base=None):
    base = base if base is not None else MODELS
    merged = {k: dict(v) for k, v in base.items()}
    hub_to_key = {_clean_repo_id(v.get("hub_id")): k for k, v in base.items()}
    dropped = []
    for name, row in (discovery or {}).items():
        hub = _clean_repo_id(row.get("hub_id") or row.get("folder") or name)
        if name in merged:                       # same key as a staple
            _absorb_disk(merged[name], row); continue
        if hub in hub_to_key:                     # same hub_id as a staple
            _absorb_disk(merged[hub_to_key[hub]], row)
            dropped.append((name, f"merged into staple {hub_to_key[hub]} (same hub_id)"))
            continue
        cfg, why = derive_model_config_row(name, row)
        if cfg is None:
            dropped.append((name, why)); continue
        merged[name] = row if "dir" in row else dict(row)
        merged[name].setdefault("model_key", name)
        hub_to_key[hub] = name
    # derive every merged row (staples now carry absorbed disk facts)
    out, drops2 = {}, []
    for name, row in merged.items():
        cfg, why = derive_model_config_row(name, row)
        (out.__setitem__(name, cfg) if cfg else drops2.append((name, why)))
    return out, dropped + drops2


# ===========================================================================
# ModelConfig assembly — identical validation path as before.
# ===========================================================================
def assess_config(cls, values):
    """Build cls if values can form a valid instance, else False. Never raises."""
    flds = {f.name: f for f in fields(cls)}
    for f in flds.values():
        required = f.default is MISSING and f.default_factory is MISSING
        if required and values.get(f.name) in (None, "", []):
            return False
    out = {}
    for name, f in flds.items():
        if name in values:
            out[name] = values[name]
        elif f.default is not MISSING:
            out[name] = f.default
        else:
            out[name] = f.default_factory()
    return cls(**out)


def get_model_values(config, dict_return=False, return_dict=False):
    if dict_return or return_dict:
        return config.to_dict()
    return config


def get_assessed_model_config(values, dict_return=False, return_dict=False):
    assessed = assess_config(ModelConfig, values)
    if assessed is False:
        return False
    return get_model_values(assessed, dict_return=dict_return, return_dict=return_dict)


def update_model_config_dict(model_key=None, values=None, dict_obj=None,
                             dict_return=False, return_dict=False, key=None):
    dict_obj = dict_obj if dict_obj is not None else {}
    model_key = model_key or key
    values = dict(values or {})
    values["model_key"] = model_key
    config = get_assessed_model_config(values, dict_return=dict_return, return_dict=return_dict)
    if config is False:
        logger.warning("registry: %s failed ModelConfig assessment, skipped", model_key)
        return dict_obj
    dict_obj[model_key] = config
    return dict_obj


def _load_discovery_report(path=None):
    """Read the on-disk discovery report. Prefer the descriptive report; fall
    back to the registry-shaped manifest. Either shape works."""
    for candidate in (path, MODELS_DISCOVERY_PATH, MODELS_DICT_PATH):
        if candidate and os.path.isfile(candidate):
            data = safe_load_from_json(candidate)
            if data:
                return data
    return {}


# ---------------------------------------------------------------------------
# Pruned-models list — user-hidden "ghost" registry rows.
# ---------------------------------------------------------------------------
# A model can show as "missing" forever: a curated staple that was never
# downloaded, or a discovery row whose files were deleted. The UI lets an
# operator prune such a row. Pruning is persisted here (a plain JSON list of
# model_keys beside the discovery report) and applied as a final filter in
# get_models_dict, so it hides the row from EVERY listing path (curated or
# discovered) without mutating the curated MODELS source in code.
def _pruned_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "pruned_models.json")


def _load_pruned():
    p = _pruned_path()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, list):
            return set(data)
        if isinstance(data, dict):
            return set(data.get("pruned") or [])
    return set()


def _save_pruned(keys):
    safe_dump_to_file(data=sorted(keys), file_path=_pruned_path())


def prune_model(model_key):
    """Hide a not-installed model from the registry listing.

    NON-DESTRUCTIVE: it only adds ``model_key`` to the persisted prune-list, which
    get_models_dict filters out. It does NOT mutate the discovery report or the
    curated MODELS code — so the action is fully reversible via unprune_model
    (and works for a curated staple, which never lives in the report). Returns a
    small status dict."""
    pruned = _load_pruned()
    was_present = model_key in pruned
    pruned.add(model_key)
    _save_pruned(pruned)
    return {
        "pruned": True,
        "model_key": model_key,
        "already_pruned": was_present,
    }


def unprune_model(model_key):
    """Reverse a prune: drop ``model_key`` from the prune-list (the curated staple
    or a re-discovered row then reappears in listings)."""
    pruned = _load_pruned()
    existed = model_key in pruned
    pruned.discard(model_key)
    _save_pruned(pruned)
    return {"unpruned": existed, "model_key": model_key}


# ---------------------------------------------------------------------------
# Media-chat allow-flag — which models the media-intelligence chat dropdown offers.
# ---------------------------------------------------------------------------
# The admin console's Models tab has a "Media" checkbox per model; the media arm's
# chat model picker only offers media-flagged + chat-capable models. DEFAULT: a
# curated staple (a key in MODELS) starts ENABLED, everything else (discovered
# rows) starts disabled — so the picker ships sane without any curation. The store
# records only DEVIATIONS from that default (a staple the operator turned off, or a
# discovered model they turned on), keeping it minimal and self-healing if MODELS
# changes. Shape on disk: {"enabled": [...], "disabled": [...]}.
def _media_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "media_models.json")


def _load_media():
    p = _media_path()
    enabled, disabled = set(), set()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, dict):
            enabled = set(data.get("enabled") or [])
            disabled = set(data.get("disabled") or [])
        elif isinstance(data, list):       # legacy plain allow-list
            enabled = set(data)
    return {"enabled": enabled, "disabled": disabled}


def _save_media(state):
    safe_dump_to_file(
        data={"enabled": sorted(state["enabled"]), "disabled": sorted(state["disabled"])},
        file_path=_media_path(),
    )


def _default_media(model_key):
    """Default media-flag for a model with no explicit override: curated staples
    (the MODELS defaults) start ON, discovered models start OFF."""
    return model_key in MODELS


def media_state(model_key):
    """Effective media-chat flag for ``model_key`` (override wins over default)."""
    ov = _load_media()
    if model_key in ov["disabled"]:
        return False
    if model_key in ov["enabled"]:
        return True
    return _default_media(model_key)


def set_model_media(model_key, enabled):
    """Set the media-chat flag. Stores only a deviation from the default — toggling
    a model back to its default state clears the override (no dead entries)."""
    enabled = bool(enabled)
    ov = _load_media()
    ov["enabled"].discard(model_key)
    ov["disabled"].discard(model_key)
    if enabled != _default_media(model_key):
        (ov["enabled"] if enabled else ov["disabled"]).add(model_key)
    _save_media(ov)
    return {"model_key": model_key, "media": enabled}


# ---------------------------------------------------------------------------
# Default media-chat model — the ONE model the media-intelligence chat dropdown
# preselects. Single global value (a model_key), not per-model. Persisted beside
# the media allow-flag store (same mechanism: a plain JSON file next to the
# discovery report) so every client agrees deterministically and it survives a
# restart. Shape on disk: {"default": "<model_key>"} or {"default": null}.
# Kept in its OWN file (not merged into media_models.json) so the whole-file
# rewrites of _save_media / _save_media_default never clobber each other.
def _media_default_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "media_default.json")


def media_default_state():
    """The currently-stored default media model_key, or None if unset/cleared."""
    p = _media_default_path()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, dict):
            val = data.get("default")
            return val or None
        if isinstance(data, str):       # tolerate a bare key on disk
            return data or None
    return None


def set_media_default(model_key, enabled):
    """Set or clear the single default media model (single-default semantics).

    enabled True  -> make ``model_key`` the default, REPLACING any previous one.
    enabled False -> clear the default IFF ``model_key`` is the current default
                     (clearing a non-default key is a no-op, never disturbs the
                     standing default).

    NOTE: setting a default does NOT require the model to be media-enabled — the
    caller may flag a model as default independently of its media allow-flag.
    Returns the resulting state for that key."""
    enabled = bool(enabled)
    current = media_default_state()
    if enabled:
        new_default = model_key
    else:
        new_default = None if current == model_key else current
    safe_dump_to_file(data={"default": new_default}, file_path=_media_default_path())
    return {
        "model_key": model_key,
        "media_default": new_default == model_key,
        "default": new_default,
    }


def get_models_dict(models_dict_path=None, dict_return=False, return_dict=False,
                    discovery=None):
    """Build the registry: MODELS + discovery (test downloads).

    discovery=None -> read the report on disk. Pass a dict to merge an
    in-memory discovery result (e.g. straight from a fresh walk).

    Operator-pruned model_keys (pruned_models.json) are filtered out as a final
    step so a hidden "ghost" row never shows in any listing."""
    dict_return = dict_return or return_dict
    report = discovery if discovery is not None else _load_discovery_report(models_dict_path)
    merged, dropped = merge_discovery_into_models(report)

    for model_key, why in dropped:
        logger.info("registry: dropped %s (%s)", model_key, why)

    pruned = _load_pruned()
    nudict = {}
    for model_key, values in merged.items():
        if model_key in pruned:
            continue
        nudict = update_model_config_dict(
            model_key=model_key, values=values, dict_obj=nudict, dict_return=dict_return
        )
    return nudict


# ===========================================================================
# Registry — built at import from MODELS + existing discovery report.
# ===========================================================================
MODEL_REGISTRY: Dict[str, ModelConfig] = get_models_dict()
MODEL_REGISTRY_DICT: Dict[str, dict] = get_models_dict(dict_return=True)


def get_model_registry(dict_return=False, return_dict=False):
    dict_return = dict_return or return_dict
    return MODEL_REGISTRY_DICT if dict_return else MODEL_REGISTRY


def refresh_registry(run_discovery=True):
    """Re-walk the model tree and rebuild MODEL_REGISTRY in place. Call this on
    hugpy startup. run_discovery=False just re-reads the existing report.

    Late import of discover_models avoids a circular import at module load.

    The update is IN PLACE (update-then-prune, never rebind): other modules
    import these dicts by reference (`from ... import MODEL_REGISTRY`), so
    rebinding the names here would leave every importer holding the stale
    dict. Update-then-prune also means a concurrent reader on the threaded
    server never catches the dict momentarily empty."""
    report = None
    if run_discovery:
        try:
            from ...apis.get_module import discover_models
            report = discover_models(save_json=True, verbose=False, use_hub=True)
        except Exception as exc:
            logger.warning("refresh_registry: discovery walk failed (%s); "
                           "falling back to on-disk report", exc)
    fresh = get_models_dict(discovery=report)
    fresh_dict = get_models_dict(dict_return=True, discovery=report)
    MODEL_REGISTRY.update(fresh)
    for stale in [k for k in MODEL_REGISTRY if k not in fresh]:
        MODEL_REGISTRY.pop(stale, None)
    MODEL_REGISTRY_DICT.update(fresh_dict)
    for stale in [k for k in MODEL_REGISTRY_DICT if k not in fresh_dict]:
        MODEL_REGISTRY_DICT.pop(stale, None)
    try:
        from .models_default import refresh_task_registries
        refresh_task_registries()
    except Exception as exc:
        logger.warning("refresh_registry: task registry refresh failed (%s)", exc)
    # Self-heal serve overrides orphaned by collision-qualification of keys
    # (bare `name` -> `owner~name`). Runs at discovery so a re-walk re-homes them.
    try:
        from ....managers.serve.overrides import migrate_overrides
        moved = migrate_overrides(fresh_dict)
        if moved:
            logger.info("refresh_registry: migrated %d orphaned serve override(s): %s",
                        len(moved), moved)
    except Exception as exc:
        logger.warning("refresh_registry: serve-override migration skipped (%s)", exc)
    return MODEL_REGISTRY
