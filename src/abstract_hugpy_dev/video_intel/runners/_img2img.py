"""img2img availability probe for the video arm.

`img2img_available(model_id)` is ADDITIONAL to (never a replacement for) the GPU
guard in `_gpu_guard.py`. It answers ONE question: can the managers inference
plane actually serve ("transformers","image-to-image") for this model right now?

True iff BOTH:
  (a) the managers registry has ("transformers","image-to-image") wired — a
      runner AND a request builder are registered; AND
  (b) it is SERVABLE for `model_id` — the model advertises the task, i.e. a
      resolve for (model_id, task="image-to-image") would succeed. `resolve()`
      raises when the model's cfg.tasks does not list the task.

(b) is SERVABLE when the model's cfg.tasks lists "image-to-image". The config
layer now advertises image-to-image for every image-generation checkpoint
(SD/SDXL/flux-class diffusers, comfy SD-lineage checkpoints — see
models_config._augment_img2img), so image models return True here, while a
genuinely-incapable model (a text LLM, whisper, embeddings, …) returns the honest
FALSE -> "not available on the fleet".

Import is LAZY inside the function (mirrors the runners' discipline): merely
importing this module never couples to the health of the managers/dispatch plane.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_IMG2IMG_KEY = ("transformers", "image-to-image")


def img2img_available(model_id: str) -> bool:
    """Return True iff the fleet can serve image-to-image for `model_id`."""
    # (a) registry has the pair wired (runner + builder).
    try:
        from abstract_hugpy_dev.managers.resolvers.categories.frameworks import (
            FRAMEWORK_RUNNERS,
        )
        from abstract_hugpy_dev.managers.resolvers.categories.builders import (
            MODEL_REQUEST_BUILDERS,
        )
    except Exception as exc:  # managers plane unhealthy -> not available
        logger.info("img2img_available: managers registry import failed (%s)", exc)
        return False
    if _IMG2IMG_KEY not in FRAMEWORK_RUNNERS or _IMG2IMG_KEY not in MODEL_REQUEST_BUILDERS:
        logger.info("img2img_available: ('transformers','image-to-image') not registered")
        return False

    # (b) servable for this model — resolve_model_key validates the model
    # advertises the task (task in cfg.tasks). Image-generation checkpoints now
    # advertise image-to-image (models_config._augment_img2img); a model that
    # genuinely can't (text LLM, whisper, embeddings, …) raises here => False.
    try:
        from abstract_hugpy_dev.managers.resolvers.model_resolver import resolve_model_key
        resolve_model_key(model_key=model_id, task="image-to-image")
    except Exception as exc:
        logger.info("img2img_available: %s cannot serve image-to-image (%s)",
                    model_id, exc)
        return False
    return True
