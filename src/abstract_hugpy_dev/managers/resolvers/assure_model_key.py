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


def _bare_tail(value: str) -> str:
    """The model name without its ``publisher~`` collision qualifier.

    Central keys a name-collision family as ``Owner~Repo``; the bare ``Repo`` is
    the SAME model. Routing already treats the two as one (``_match_keys`` adds
    the "~"-tail), but the canonical registry resolver did not, so a bare key
    slug-compared against an ``Owner~Repo`` key never matched and central kept
    offering the bare twin as if it were a distinct (routable) model. k67.
    """
    s = str(value)
    return s.split("~", 1)[1] if "~" in s else s


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
    bare_slug = _slugify(_bare_tail(model_key))

    # A last-resort bare-tail match, tried only after every stronger signal
    # below fails, so an exact / org-qualified / hub_id / folder hit always
    # wins first. Recorded here and applied after the loop so a precise match
    # later in the registry is never pre-empted by a looser bare-tail one.
    bare_fallback = None

    for key, values in MODEL_REGISTRY.items():
        if _slugify(key) == slug:
            return key

        hub_id = getattr(values, "hub_id", None)
        if hub_id and _slugify(hub_id) == slug:
            return key

        folder = getattr(values, "folder", None)
        if folder and _path_suffix_matches(folder, model_key):
            return key

        # Bare key ↔ Owner~Repo: the input's bare tail slug-equals this key's
        # bare tail (so "Qwen3-Coder-Next-GGUF" resolves to the canonical
        # "Qwen~Qwen3-Coder-Next-GGUF" row). Held, not returned, so it only
        # applies when nothing stronger matched anywhere in the registry.
        if bare_fallback is None and _slugify(_bare_tail(key)) == bare_slug:
            bare_fallback = key

    return bare_fallback
