from .imports import *
from .models import *
import logging
logger = logging.getLogger(__name__)



def resolve_hf_model_dir(base_dir: str) -> str:

    if config_exists(base_dir):
        return base_dir

    snapshots = join_path(base,"snapshots")
    if is_dir(snapshots):
        candidates = [
            p for p in itter_dir(snapshots)
            if is_dir(p) and config_exists(p)
        ]

        if candidates:
            return max(candidates, key=lambda p: st_mtime(p))

    raise FileNotFoundError(f"No usable Hugging Face model dir found under: {base}")

# ---------------------------------------------------------------------
# Registry utilities
# ---------------------------------------------------------------------

def list_models():
    return list(MODEL_REGISTRY.keys())


def _resolve_model_key(model_key, registry, prefer=None):
    """Map a possibly-bare model_key to a concrete registry key.

    Keys are bare basenames unless an owner collision forced an
    ``<owner>/<name>`` qualifier (see discover_models). Resolution order:
      1. exact key — covers unique bare keys AND fully-qualified keys;
      2. a single qualified key whose bare suffix matches;
      3. ambiguous bare key -> prefer a variant already allocated to a
         slot/worker (``prefer``), else the first qualified key in sorted order.
    Returns the resolved key, or None when nothing matches."""
    if model_key in registry:
        return model_key
    candidates = sorted(k for k in registry if k.rsplit("/", 1)[-1] == model_key)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for p in (prefer or []):                       # slot/worker allocation wins
        if p in candidates:
            return p
        for c in candidates:                       # prefer may itself be bare
            if c.rsplit("/", 1)[-1] == p:
                return c
    logger.info("ambiguous model_key %r -> %s (candidates: %s)",
                model_key, candidates[0], candidates)
    return candidates[0]                            # stable first-in-list


def get_model_config(model_key: str=None,dict_return=False,return_dict=False,key: str=None,prefer=None) -> ModelConfig or dict:
    model_key = model_key or key
    model_registry = get_model_registry(dict_return=dict_return,return_dict=return_dict)

    resolved = _resolve_model_key(model_key, model_registry, prefer=prefer)
    if resolved is None:
        raise KeyError(f"Unknown model: {model_key}")
    return model_registry[resolved]


def list_model_options():
    return {
        key: {
            "name": cfg.name,
            "hub_id": cfg.hub_id,
            "folder": cfg.folder,
            "tasks": cfg.tasks,
            "framework": cfg.framework,
            "filename": cfg.filename,
            "max_new_tokens": cfg.max_new_tokens,
            "port": cfg.port,
        
        }
        for key, cfg in MODEL_REGISTRY.items()
    }

# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

def get_model_path(key: str):
    env_override = get_env_value(f"MODEL_{key.upper()}")
    if env_override:
        return env_override
    cfg = get_model_config(key)
    path = os.path.join(MODELS_HOME,cfg.folder)
    return path


def get_gguf_file(path: str, cfg: ModelConfig, prefer: Optional[str] = None) -> Optional[str]:
    # The multimodal projector (mmproj-*.gguf) is NOT the model — exclude it so a
    # vision GGUF dir resolves to the language model, not the CLIP projector.
    from ..src.utils import is_mmproj_file
    # Recursive: split-gguf models nest their shards in a subdir; the shallow
    # glob missed them and resolved to None (the "filename fiasco").
    ggufs = [g for g in (get_glob(path, "*.gguf", recursive=True)
                         + get_glob(path, "*.GGUF", recursive=True))
             if not is_mmproj_file(g)]
    if not ggufs:
        return None

    # 1) Designation wins, and it auto-grabs the file that exists. `prefer` is a
    #    per-request user choice; cfg.filename is the config-level default. Each
    #    matches by exact basename OR by substring (so "Q4_K_M" selects the right
    #    quant without naming the whole file).
    for want in (prefer, getattr(cfg, "filename", None)):
        if not want or is_mmproj_file(want):
            continue
        base = os.path.basename(want).lower()
        for g in ggufs:                                   # exact basename
            if os.path.basename(g).lower() == base:
                return g
        hits = sorted(g for g in ggufs if base in os.path.basename(g).lower())
        if hits:                                          # substring / quant tag
            return hits[0]

    # 2) Exactly one model gguf: unambiguous, grab it.
    if len(ggufs) == 1:
        return ggufs[0]

    # 3) Multiple, no designation: pick DETERMINISTICALLY (not arbitrary glob
    #    order). Prefer the first shard of a split gguf, else a common quant.
    shard0 = sorted(g for g in ggufs if "00001-of-" in os.path.basename(g).lower())
    if shard0:
        return shard0[0]

    def _rank(g: str):
        b = os.path.basename(g).lower()
        for i, q in enumerate(("q4_k_m", "q4_k_s", "q5_k_m", "q6_k", "q8_0",
                               "q4_0", "q5_0", "f16", "bf16", "f32")):
            if q in b:
                return (i, b)
        return (99, b)

    return sorted(ggufs, key=_rank)[0]


def model_looks_downloaded(path: str, cfg: Optional[ModelConfig] = None) -> bool:
    """
    Lightweight check to avoid treating partial Hugging Face / Git-LFS
    pointer directories as usable model directories.

    Supports both:
      - transformers model dirs
      - GGUF model dirs for llama.cpp
    """
    if not exists(path) or not is_dir(path):
        return False

    if cfg and cfg.framework == "llama_cpp":
        gguf = get_gguf_file(path, cfg)
        if not (gguf and exists(gguf) and st_size(gguf) > 1024 * 1024):
            return False
        # A vision GGUF needs its mmproj projector beside the model, or llama.cpp
        # loads it text-only and silently ignores images. Treat a vision model
        # dir WITHOUT a projector as incomplete so ensure_model fetches the
        # projector instead of short-circuiting on the main quant alone.
        is_vision = (getattr(cfg, "primary_task", None) == "image-text-to-text"
                     or "image-text-to-text" in (getattr(cfg, "tasks", None) or []))
        if not is_vision:
            inc = getattr(cfg, "include", None) or []
            try:
                from ..src.utils import is_mmproj_file
                is_vision = any(is_mmproj_file(x) for x in inc)
            except Exception:
                is_vision = False
        if is_vision:
            try:
                from ..src.utils import find_mmproj
                if not find_mmproj(path):
                    return False
            except Exception:
                pass
        return True

    if not config_exists(path):
        return False

    safetensor_files = list(get_glob(path,"*.safetensors"))

    if safetensor_files:
        for file_path in safetensor_files:
            if st_size(file_path) < 1024 * 1024:
                return False
        return True

    expected_any = [
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]

    return any(exists(join_path(path,name)) for name in expected_any)

# ---------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------




def resolve_model_source(key: str) -> str:
    cfg = get_model_config(key)
    local = get_model_path(key)
    env_override = get_env_value(f"MODEL_{key.upper()}")

    if env_override and not exists(local):
        raise FileNotFoundError(
            f"MODEL_{key.upper()}={env_override} was set but path does not exist"
        )

    if cfg.framework == "llama_cpp":
        if not model_looks_downloaded(local, cfg):
            return cfg.hub_id

        gguf = get_gguf_file(local, cfg)
        if not gguf:
            raise FileNotFoundError(f"No GGUF file found in {local}")

        return str(gguf)

    if model_looks_downloaded(local, cfg):
        return str(local)

    return cfg.hub_id


# ---------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------

class _LazyModelPaths:
    """
    Dict-like that resolves on access, not import.

    Managers that cache DEFAULT_PATHS["foo"] in __init__ get the
    correct value at construction time — even if the model was
    downloaded or deleted after the module was first imported.
    """

    def __getitem__(self, key: str) -> str:
        return resolve_model_source(key)

    def get(self, key: str, default=None) -> str:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in MODEL_REGISTRY


DEFAULT_PATHS: _LazyModelPaths = _LazyModelPaths()
