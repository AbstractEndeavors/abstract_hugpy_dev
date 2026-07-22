"""BACK-COMPAT SHIM — this module moved to ``comms/model_metadata.py``
(2026-07-22 rename: the store widened from HF-only to the external model
metadata store, covering HF per-repo AND Civitai per-model facts).

Old names keep working forever-ish:

    HfMetadataStore   -> ModelMetadataStore
    hf_metadata_store -> model_metadata_store (SAME singleton object)
    HUGPY_HF_CACHE_DB -> HUGPY_MODEL_METADATA_DB (old var honored if set)
    hf_metadata.db    -> model_metadata.db (one-shot rename on store init)

New code imports from ``abstract_hugpy_dev.comms.model_metadata``.
"""
from .model_metadata import *  # noqa: F401,F403
from .model_metadata import (  # noqa: F401 — explicit for tooling/star-safety
    MAX_FAILURES,
    ModelMetadataStore,
    checkpoint_stem,
    default_db_path,
    fetch_civitai_meta,
    fetch_repo_info,
    model_metadata_store,
    serialize_civitai_model,
    serialize_model_info,
    sum_sibling_sizes,
)

# Legacy aliases.
HfMetadataStore = ModelMetadataStore
hf_metadata_store = model_metadata_store
