"""Generic execute_prompt passthrough — the whole dispatch surface over HTTP.

/chat/stream covers streamed text generation; this route covers everything
else with one verb. The JSON body is the same kwargs surface execute_prompt
takes in-process (task, model_key, prompt/messages/text/texts/other_texts,
file/image_b64, generation params, task-specific params), and the response is
the TaskResult dumped to JSON — so an HTTP client can drive every registered
task category (embeddings, similarity, summarize-with-presets, whisper with
language/translate, vision, text-to-image) without a dedicated route each.

    POST /prompt        {"task": "sentence-similarity", "texts": [...], "other_texts": [...]}
    GET  /prompt/tasks  -> {"tasks": [...], "defaults": {task: model_key}}

Explicit values win; anything omitted falls to resolve()'s default chain
(model_key > task > media-type-of-file > chat default).
"""
from __future__ import annotations

import asyncio
import inspect

from ..functions import *

prompt_bp, logger = get_bp("prompt_bp", __name__)


def _await_sync(value):
    """Drive execute_prompt's (possibly) awaitable result from WSGI.

    Uses the process-wide async runtime (one long-lived loop) rather than a
    fresh per-request loop — see _platform/async_runtime.
    """
    if not inspect.isawaitable(value):
        return value
    from abstract_hugpy_dev._platform import async_runtime
    return async_runtime.run(value)


def _result_payload(result) -> dict:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    return {"text": str(result)}


@prompt_bp.route("/prompt", methods=["POST"])
def prompt_execute():
    body = request.get_json(silent=True) or {}
    # Underscore-prefixed keys are internal routing controls (_force_local);
    # never client-settable.
    kwargs = {
        k: v for k, v in body.items()
        if not k.startswith("_") and v is not None
    }
    if not kwargs:
        return jsonify({"ok": False, "error": "empty request body"}), 400

    # Resolve the dedicated worker pool (API key's bound pool + explicit override)
    # in request context, then thread it so non-chat tasks route to it too.
    try:
        from ..functions.chat.streaming import _resolve_request_pool
        eff_pool = _resolve_request_pool(kwargs.get("pool"))
        if eff_pool:
            kwargs["pool"] = eff_pool
        else:
            kwargs.pop("pool", None)
    except Exception:
        pass

    from ..functions.imports import execute_prompt  # late import dodges circulars

    try:
        result = _await_sync(execute_prompt(**kwargs))
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        # resolve()/builder validation errors — the caller's to fix.
        return jsonify({"ok": False, "error": str(exc).strip("'\"")}), 400
    except Exception as exc:
        logger.exception("execute_prompt failed")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    payload = _result_payload(result)
    payload.setdefault("ok", not payload.get("error"))
    return jsonify(payload)


@prompt_bp.route("/prompt/tasks", methods=["GET"])
def prompt_tasks():
    from ..functions.imports import KNOWN_TASKS_REGISTRY, TASK_DEFAULTS
    return jsonify({
        "tasks": sorted(KNOWN_TASKS_REGISTRY),
        "defaults": dict(TASK_DEFAULTS),
    })
