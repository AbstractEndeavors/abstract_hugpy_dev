from .imports import *
import re
from pathlib import PurePosixPath


def _normalize_model_path(value: str) -> str:
    """
    Normalize a registry folder/model path for suffix comparison.
    Works for Unix-style paths and Hugging Face-style repo ids.
    """
    return str(PurePosixPath(str(value).strip().rstrip("/")))


def _path_suffix_matches(folder: str, model_key: str) -> bool:
    """
    Return True when `model_key` matches the trailing path parts of `folder`.

    Examples:
        folder:    /mnt/llm_storage/models/Qwen/Qwen2.5-7B
        model_key: Qwen/Qwen2.5-7B        -> True
        model_key: Qwen2.5-7B             -> True
        model_key: other/Qwen2.5-7B       -> False
    """
    folder_parts = _normalize_model_path(folder).split("/")
    model_parts = _normalize_model_path(model_key).split("/")

    if len(model_parts) > len(folder_parts):
        return False

    return folder_parts[-len(model_parts):] == model_parts


def _slugify(value: str) -> str:
    """Collapse a key/hub_id to the manifest slug form for comparison.

    The manifest keys models as key_for_hub_id("C10X/Qwen2.5-1.5B-Instruct")
    -> "C10X_Qwen2.5-1.5B-Instruct", while the registry keys by folder tail
    ("Qwen2.5-1.5B-Instruct"). Comparing slugs (case-insensitive, separators
    collapsed to "_") lets either form resolve to the canonical registry key.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_").lower()


def assure_model_key(model_key):
    """
    Resolve a user-provided model key, repo id, manifest slug, folder name,
    or folder suffix into the canonical key from MODEL_REGISTRY.
    """
    if not model_key:
        return None

    model_key = str(model_key).strip().rstrip("/")

    if model_key in MODEL_REGISTRY:
        return model_key

    slug = _slugify(model_key)

    for key, values in MODEL_REGISTRY.items():
        if _slugify(key) == slug:
            return key

        hub_id = getattr(values, "hub_id", None)
        if hub_id and _slugify(hub_id) == slug:
            return key

        folder = getattr(values, "folder", None)
        if folder and _path_suffix_matches(folder, model_key):
            return key

    return None
