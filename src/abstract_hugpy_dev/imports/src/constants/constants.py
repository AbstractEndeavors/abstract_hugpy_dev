import os
from ..standalone_utils import get_env_value
from typing import Literal, Optional
from .imports import make_list,HfApi,re,safe_dump_to_file

# Tokenizers set this as a sentinel for "no enforced limit". It's never a real window.

# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------
HUGGINGFACE_DOMAIN = "https://huggingface.co"

HF_TOKEN = get_env_value("HF_TOKEN") or False


def env_bool(key: str, default: bool = False) -> bool:
    """Env flag -> bool. `get_env_value(...) or default` can never yield False
    (the string \"false\" is truthy, unset -> default), so flags parsed that way
    are stuck at their default forever. This coerces properly.

    Process environment wins over the .env file: get_env_value only reads the
    .env file, but operational flags must also respond to a systemd
    `Environment=` line or a `FLAG=false cmd` shell prefix."""
    value = os.environ.get(key)
    if value is None:
        value = get_env_value(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")

hfApi = HfApi(token=HF_TOKEN)

def _resolve_default_root():
    # Single source of truth for the storage root, per-OS. Honours DEFAULT_ROOT,
    # keeps the historical /mnt/llm_storage mount when present and writable, and
    # otherwise lands in a per-user data dir (XDG / AppData / Library) so the API
    # works out of the box on a fresh box, Windows, or macOS.
    from ...._platform.paths import models_root

    return models_root()

DEFAULT_ROOT = _resolve_default_root()

MODELS_HOME = MODELS_DIR =  get_env_value("MODELS_HOME") or os.path.join(DEFAULT_ROOT,"models")

UPLOADS_HOME = CHAT_UPLOAD_DIR =  get_env_value("UPLOADS_HOME") or os.path.join(DEFAULT_ROOT,"uploads")

PROJECTS_HOME = PROJECTS_DIR =  get_env_value("PROJECTS_HOME") or os.path.join(DEFAULT_ROOT,"projects")

PROJECTS_PLACEMENT_PATH = get_env_value("PROJECTS_PLACEMENT_PATH") or os.path.join(PROJECTS_HOME,"placement.json")

DATASETS_HOME = DATASETS_DIR =  get_env_value("DATASETS_HOME") or os.path.join(DEFAULT_ROOT,"datasets")

MODELS_DISCOVERY_PATH = get_env_value("MODELS_DISCOVERY_PATH") or os.path.join(PROJECTS_HOME,"model_discovery.json")

MODELS_DICT_PATH = get_env_value("MODELS_DICT_PATH") or os.path.join(PROJECTS_HOME,"model_manifest.json")

HF_CACHE = get_env_value("HF_CACHE") or os.path.join(MODELS_HOME,"cache")

HF_HOME = get_env_value("HF_HOME") or os.path.join(HF_CACHE,"huggingface")

HF_HUB_CACHE = get_env_value("HF_HUB_CACHE") or os.path.join(HF_HOME,"hub")

TORCH_HOME = get_env_value("TORCH_HOME") or os.path.join(HF_CACHE,"torch")

PIP_CACHE_DIR = get_env_value("PIP_CACHE_DIR") or os.path.join(HF_CACHE,"pip")

PATHS = [
    MODELS_HOME,
    UPLOADS_HOME,
    PROJECTS_HOME,
    DATASETS_HOME,
    HF_HUB_CACHE,
    HF_CACHE,
    HF_HUB_CACHE,
    TORCH_HOME,
    PIP_CACHE_DIR
]



def _ensure_dirs(paths):
    """Best-effort create the storage dirs.

    Importing abstract_hugpy must never hard-crash just because a storage path
    can't be made — e.g. on a worker box where DEFAULT_ROOT (/mnt/llm_storage)
    is a broken/stale mount (OSError errno 5) or simply not present. Each dir is
    created independently; failures are warned about, not fatal. Set
    DEFAULT_ROOT to a local, writable path on such boxes.
    """
    import logging
    failed = []
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            failed.append((path, exc))
    if failed:
        logging.getLogger("abstract_hugpy").warning(
            "could not create %d storage dir(s); continuing. "
            "Set DEFAULT_ROOT to a writable path to silence this. Details: %s",
            len(failed),
            "; ".join(f"{p} ({e.__class__.__name__}: {e})" for p, e in failed),
        )


_ensure_dirs(PATHS)
if not os.path.isfile(PROJECTS_PLACEMENT_PATH):
    safe_dump_to_file(file_path=PROJECTS_PLACEMENT_PATH,data={})
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("TORCH_HOME", TORCH_HOME)
os.environ.setdefault("PIP_CACHE_DIR", PIP_CACHE_DIR)

HUGPY_MARKER= get_env_value("HUGPY_MARKER") or "hugpy.json"

LLAMA_HOST= get_env_value("LLAMA_HOST") or "http://127.0.0.1"
VISION_HOST= get_env_value("VISION_HOST") or "http://127.0.0.1"

# `legacy` is the operator's data-hoarding store — kept on disk but never
# walked, indexed, or counted by any metric/appendage in these systems.
EXCLUDE_DIR_NAMES = make_list(get_env_value("EXCLUDE_DIR_NAMES") or ".cache,.git,.locks,snapshots,blobs,refs,1_Pooling,2_Normalize,onnx,legacy")
EXCLUDE_DIR_NAMES = frozenset(EXCLUDE_DIR_NAMES)

EXCLUDE_DIR_PREFIXES = make_list(get_env_value("EXCLUDE_DIR_PREFIXES") or "models--")
EXCLUDE_DIR_PREFIXES = tuple(EXCLUDE_DIR_PREFIXES)  # HF cache root naming

TOKENIZER_SENTINEL_THRESHOLD = float(get_env_value("TOKENIZER_SENTINEL_THRESHOLD") or 10**9)
DEFAULT_TIMEOUT= float(get_env_value("DEFAULT_TIMEOUT") or 3600.0)
DEFAULT_MAX_TOKENS= int(get_env_value("DEFAULT_MAX_TOKENS") or 32768)
MIN_INPUT_WORDS_DEFAULT = get_env_value("MIN_INPUT_WORDS_DEFAULT") or 10
  # whatever key resolve_qwen_vl_path expects
  
SOURCEKIND = make_list(get_env_value("SOURCEKIND") or "text,url,file,image")
SOURCEKIND =Literal[*SOURCEKIND]

JOBSTATUS = make_list(get_env_value("JOBSTATUS") or "queued,running,completed,failed,cancelled")
JOBSTATUS =Literal[*JOBSTATUS]

DEFAULT_TEMPERATURE = float(get_env_value("DEFAULT_TEMPERATURE") or 0.1)
DEFAULT_TOP_P = float(get_env_value("DEFAULT_TOP_P") or 1)

FINISH_REASONS = make_list(get_env_value("FINISH_REASONS") or "stop,max_tokens,cancelled,error")
FINISH_REASONS =Literal[*FINISH_REASONS]

ROLES = make_list(get_env_value("ROLES") or "system,user,assistant")
ROLES = Literal[*ROLES]

# Stock defaults: one model per task, every one a curated staple in
# models_config.MODELS — efficiency-first picks (whole fleet ~17GB).
DEFAULT_CHAT_MODEL = get_env_value("DEFAULT_CHAT_MODEL") or "Qwen2.5-3B-Instruct-GGUF"
DEFAULT_VISION_MODEL = get_env_value("DEFAULT_VISION_MODEL") or "Qwen2.5-VL-3B-Instruct-GGUF"
DEFAULT_WHISPER_MODEL = get_env_value("DEFAULT_WHISPER_MODEL") or "whisper-large-v3-turbo"
DEFAULT_SUMMARIZE_MODEL = get_env_value("DEFAULT_SUMMARIZE_MODEL") or "flan-t5-large"
DEFAULT_EMBED_MODEL = get_env_value("DEFAULT_EMBED_MODEL") or "all-minilm-l6-v2"
DEFAULT_IMAGEGEN_MODEL = get_env_value("DEFAULT_IMAGEGEN_MODEL") or "sd-turbo"
DEFAULT_KEYWORDS_MODEL = get_env_value("DEFAULT_KEYWORDS_MODEL") or "all-minilm-l6-v2"
# Vision-analysis family (generic transformers-pipeline runner).
DEFAULT_DEPTH_MODEL = get_env_value("DEFAULT_DEPTH_MODEL") or "depth-anything-v2-small"
DEFAULT_DETECT_MODEL = get_env_value("DEFAULT_DETECT_MODEL") or "detr-resnet-50"
DEFAULT_IMG_CLASSIFY_MODEL = get_env_value("DEFAULT_IMG_CLASSIFY_MODEL") or "vit-base-patch16-224"
DEFAULT_SEGMENT_MODEL = get_env_value("DEFAULT_SEGMENT_MODEL") or "segformer-b0-ade"

DISK_AUTHORITATIVE = make_list(get_env_value("DISK_AUTHORITATIVE") or "name,folder,framework,filename")
OVERLAY_ALLOWED = set(make_list(get_env_value("OVERLAY_ALLOWED") or "port, host, timeout_s, include"))

GGUF_QUANT = re.compile(r"(Q\d+_[A-Z0-9_]+|F16|BF16|F32)", re.I)

DEFAULT_LOCAL_FILES_ONLY = env_bool("DEFAULT_LOCAL_FILES_ONLY", True)

# Kill switch for resolve()-time staple downloads. Default on: a fresh install
# pulls the curated MODELS fleet on first use. Set HUGPY_AUTO_DOWNLOAD=false on
# air-gapped boxes / workers that must never touch the network.
HUGPY_AUTO_DOWNLOAD = env_bool("HUGPY_AUTO_DOWNLOAD", True)
