from .imports import *
from .constants import *
from .hugpy_marker import *
def safe_path_part(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")
def safe_name(value: str) -> str:
    value = value.strip()
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_/")
def split_hub_id(hub_id: str) -> tuple[str, str | None]:
    parts = hub_id.strip("/").split("/")
    if len(parts) <= 2:
        return hub_id, None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return repo_id, subfolder
def runtime_folder(framework: str, hub_id: str, include: Any = None, filename: str | None = None) -> str:
    framework = framework.lower().strip()

    if framework == "gguf":
        return "gguf"

    if filename and filename.lower().endswith(".gguf"):
        return "gguf"

    if include:
        patterns = include if isinstance(include, list) else [include]
        if any("gguf" in pattern.lower() for pattern in patterns):
            return "gguf"

    if framework == "transformers":
        return "transformers"

    return "misc"

# ---------------------------------------------------------------------------
# Model Paths — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------
def is_model_dir(directory: str) -> bool:
    """A model dir declares itself with a hugpy.json. Fall back to weight
    markers for dirs not yet stamped (legacy / first-run before backfill).

    ``model_index.json`` marks a diffusers PIPELINE root — the model itself,
    whose weights live in per-component subdirs (text_encoder/, transformer/,
    vae/, …). It MUST count as a model dir so the walk stops here (a leaf) and
    never descends to register those components as phantom standalone models
    (the bare `text_encoder`/`transformer` rows). Without this, an unstamped
    pipeline (freshly downloaded, before its marker is backfilled) has no
    top-level config.json/weights, so the walk fell through into its parts."""
    if os.path.isfile(os.path.join(directory, HUGPY_MARKER)):
        return True
    try:
        entries = os.listdir(directory)
    except (OSError, NotADirectoryError):
        return False
    return any(e == "config.json" or e == "model_index.json"
               or e.lower().endswith((".gguf", ".safetensors"))
               for e in entries)

def get_model_dirs(models_home=None):
    root = get_model_home(models_home=models_home)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(join_path(dirpath, d))]
        if is_model_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []     # model dir is a leaf; don't descend
    return found

def hub_id_for(directory: str, fallback: str | None = None) -> str | None:
    """Repo id from the declared marker; path slice only if unstamped."""
    marker = read_hugpy_marker(directory)
    if marker and marker.get("hub_id"):
        return marker["hub_id"]
    return fallback


def get_model_dirs(models_home=None):
    """Walk MODELS_HOME and return every directory that actually holds a model,
    regardless of how deep it sits. Skips excluded dirs and never lists files."""
    root = get_model_home(models_home=models_home)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded subtrees in-place so os.walk doesn't descend them
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(join_path(dirpath, d))]
        if is_model_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []          # don't descend into a model's own files
    return found

def get_hub_id_from_directory(directory, models_home=None):
    """Repo id = the path of the model dir relative to its task dir.
    (directory is the longer path; strip the task-dir prefix off the FRONT.)"""
    root = get_model_home(models_home=models_home)
    rel = directory[len(root):].strip("/")
    parts = rel.split("/")
    # rel = family/task/owner/repo[/subfolder]; drop family+task, keep the rest
    return "/".join(parts[2:]) if len(parts) > 2 else rel

def is_directory_excluded(directory):
    base = os.path.basename(directory.rstrip("/"))
    if base in EXCLUDE_DIR_NAMES:
        return True
    return any(base.startswith(p) for p in EXCLUDE_DIR_PREFIXES)

def exclude_dirs(directories):
    return [d for d in directories if not is_directory_excluded(d)]
def resolve_task(model: dict) -> str:
    """Effective task across all dict shapes:
       ModelConfig.to_dict() -> primary_task (+ tasks list)
       /repos/download dict   -> primary_task
       legacy manifest        -> task (singular)
    First non-empty wins; misc is the floor."""
    pt = model.get("primary_task")
    if pt:
        return pt
    tasks = model.get("tasks")
    if tasks:
        return tasks[0] if isinstance(tasks, list) else tasks
    return model.get("task") or "misc"
def _existing_sibling_task_dir(root: str, runtime: str, hub_path: str) -> str | None:
    """Same runtime + hub_id under a DIFFERENT task folder — where a model's files
    physically landed at download time, which can diverge from a later
    content-corrected task (e.g. a text gguf mis-routed under image-text-to-text,
    then re-derived to text-generation once we saw it has no mmproj). Returns the
    first existing such dir, else None."""
    base = os.path.join(root, "models", runtime)
    try:
        entries = os.listdir(base)
    except OSError:
        return None
    for task_dir in entries:
        cand = os.path.join(base, task_dir, hub_path)
        if os.path.isdir(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# Storage layout — operator-locked 2026-07-11: FLAT.
#
#   models/<runtime>/<owner>/<repo>        (datasets keep datasets/<owner>/<repo>)
#
# The task segment DIED. It baked derived, mutable, PLURAL metadata (a model
# advertises many tasks) into an IMMUTABLE path — which produced task-twin dirs
# (the same repo under text-generation AND image-text-to-text), sticky wrong-task
# discovery, empty re-routed dirs with the weights stranded in legacy misc/, and
# the "redownload models that are ready" complaints. Task/framework now live
# ONLY in the registry + the per-dir hugpy.json marker. route_destination() emits
# the flat path for ALL new work; resolve_model_dir() reads THROUGH every
# historical layout so a model already on disk under an old task path is never
# re-downloaded, mis-flagged, or 404'd — during OR after the migration.
# ---------------------------------------------------------------------------

# Runtime families = the top storage segment. "misc" is the catch-all runtime
# (comfy + any odd loader); "safetensors" is a historical family kept for reads.
RUNTIME_FAMILIES = ("gguf", "transformers", "misc", "safetensors")


def _hub_path_of(model: dict) -> str:
    return safe_path_part(
        model.get("hub_id") or model.get("name") or model.get("folder") or "")


def flat_destination(model: dict, root: str = DEFAULT_ROOT) -> str:
    """The FLAT write target — models/<runtime>/<owner>/<repo> — where ALL new
    downloads land. No task segment. Single source of truth for the new layout;
    datasets keep their own top-level home (unchanged)."""
    hub_path = _hub_path_of(model)
    if resolve_task(model) == "dataset":
        return os.path.join(root, "datasets", hub_path)
    runtime = runtime_folder(model.get("framework") or "", hub_path,
                             include=model.get("include"),
                             filename=model.get("filename"))
    return os.path.join(root, "models", runtime, hub_path)


def _routing_as_cfg(model: dict):
    """A minimal cfg shim for model_looks_downloaded from a bare routing dict
    (which carries no ModelConfig). Only the fields the completeness gate reads
    matter: framework (the gguf branch), filename/include (pin + vision), and
    primary_task/tasks (the vision-needs-mmproj gate)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        framework=model.get("framework"),
        filename=model.get("filename"),
        include=model.get("include"),
        primary_task=model.get("primary_task") or model.get("task"),
        tasks=model.get("tasks"),
    )


def legacy_task_dirs(hub_path: str, runtime: str, root: str = DEFAULT_ROOT) -> list:
    """Every task-based legacy dir on disk holding this repo under ``runtime``:
    models/<runtime>/<task>/<owner>/<repo>. The task segment is globbed, so this
    finds task-twins (the same repo under several task folders) without knowing
    the task set in advance. Sorted for a deterministic order."""
    import glob as _glob
    pattern = os.path.join(root, "models", runtime, "*", hub_path)
    return sorted(d for d in _glob.glob(pattern) if os.path.isdir(d))


def candidate_model_dirs(model: dict, root: str = DEFAULT_ROOT) -> list:
    """ORDERED list of every dir this model's files might occupy, best-layout
    first — the read-through search order AND the reconcile survey set:

      1. the entry's recorded on-disk dir (discovery ground truth): ``dir``,
         then MODELS_HOME/``folder``;
      2. the FLAT path models/<runtime>/<owner>/<repo>;
      3. legacy task dirs models/<runtime>/<task>/<owner>/<repo> — the model's
         advertised primary_task first, then the rest, deterministically;
      4. the same across the OTHER runtime families (a repo mis-filed under a
         different family, or a text gguf vs. its vision twin).

    De-duplicated, order-preserving. Dirs need NOT exist — callers test."""
    hub_path = _hub_path_of(model)
    task = resolve_task(model)
    out: list = []
    seen: set = set()

    def _add(d):
        if d and d not in seen:
            seen.add(d)
            out.append(d)

    if task == "dataset":
        _add(os.path.join(root, "datasets", hub_path))
        return out

    rec = model.get("dir")
    if rec:
        _add(rec)
    folder = model.get("folder")
    if folder:
        _add(folder if os.path.isabs(folder)
             else os.path.join(root, "models", folder))

    primary_runtime = runtime_folder(model.get("framework") or "", hub_path,
                                     include=model.get("include"),
                                     filename=model.get("filename"))
    families = [primary_runtime] + [f for f in RUNTIME_FAMILIES
                                    if f != primary_runtime]
    for fam in families:
        _add(os.path.join(root, "models", fam, hub_path))        # flat
        legacy = legacy_task_dirs(hub_path, fam, root)           # task-based
        legacy.sort(key=lambda d: (0 if os.sep + task + os.sep in d else 1, d))
        for d in legacy:
            _add(d)
    return out


def _safe_complete(directory: str, cfg) -> bool:
    """``model_looks_downloaded``, guarded and re-entrancy-safe.

    route_destination is also called DURING import-time registry construction
    (models_config's ``MODEL_REGISTRY = get_models_dict()`` -> comfy sweep ->
    route_destination), at which point ``config.main`` is only PARTIALLY
    initialized on the current import stack. Importing it there would re-enter a
    half-built module. So we use the rich ``model_looks_downloaded`` ONLY when
    ``config.main`` is ALREADY fully loaded in sys.modules (always true at
    runtime — it defines DEFAULT_PATHS, imported by the whole package), and fall
    back to a weak "exists and non-empty" signal otherwise. Every REAL
    resolution (loads, /models, reconcile) runs at runtime and gets the rich
    verdict; the weak fallback only ever serves the import-time sweep, which
    doesn't need read-through. Never raises into a resolve."""
    try:
        import sys as _sys
        main_name = __name__.rsplit(".", 3)[0] + ".config.main"  # imports.config.main
        main_mod = _sys.modules.get(main_name)
        mld = getattr(main_mod, "model_looks_downloaded", None) if main_mod is not None else None
        if mld is not None:
            return bool(mld(directory, cfg))
    except Exception:
        pass
    try:
        return os.path.isdir(directory) and bool(os.listdir(directory))
    except OSError:
        return False


def resolve_model_dir(model: dict, root: str = DEFAULT_ROOT, cfg=None,
                      require_complete: bool = True):
    """Read-through resolver — the ONE place that turns a routing/config into the
    real on-disk dir, checking the FLAT layout first, then EVERY legacy layout
    (see candidate_model_dirs). Returns the first dir that passes
    ``model_looks_downloaded``. This is the guarantee that a model downloaded
    under an OLD task path is never re-downloaded or 404'd during (or after) the
    migration. Loaders/provisioners route through it.

      require_complete=True  -> first COMPLETE dir, else None.
      require_complete=False -> first COMPLETE dir, else the first EXISTING dir
                                (so resume/delete/status act on the real partial
                                files, never orphaning them), else the flat write
                                target for a genuinely-new download.
    """
    _cfg = cfg if cfg is not None else _routing_as_cfg(model)
    cands = candidate_model_dirs(model, root)
    for d in cands:
        if os.path.isdir(d) and _safe_complete(d, _cfg):
            return d
    if require_complete:
        return None
    for d in cands:
        if os.path.isdir(d):
            return d
    return flat_destination(model, root)


def route_destination(model: dict, root: str = DEFAULT_ROOT) -> str:
    """THE single path chokepoint. Reads resolve through every historical layout
    (flat + legacy task dirs + misc/other families); a genuinely-new download
    with nothing on disk gets the FLAT target models/<runtime>/<owner>/<repo>.
    Kept single-positional-arg compatible (root optional) — every call site and
    the worker-side re-export depend on that shape."""
    return resolve_model_dir(model, root, require_complete=False)
