"""Public, OpenAI-compatible inference API (/v1) + API-key management.

The UI is just one client; this is the programmatic surface. Any OpenAI SDK
or plain curl works against it:

    client = OpenAI(base_url="https://dev.hugpy.ai/api/v1", api_key="hp_…")
    client.chat.completions.create(model="Qwen2.5-1.5B-Instruct",
                                   messages=[...], stream=True)

Auth is optional by design: a site-level `require_key` flag (toggled from
the UI) decides whether /v1 calls must present `Authorization: Bearer hp_…`.
Keys themselves are minted/revoked from the UI via the /keys routes below,
which are part of the site (same origin as the console), not part of /v1.

Model names accept any form assure_model_key resolves — registry key,
hub id (org/name), or manifest slug.
"""
from __future__ import annotations

import json
import time
import uuid
from functools import wraps

from flask import Response, jsonify, request, stream_with_context

from ..functions import *  # get_bp, api-key store, chat_iter_sync, get_models_dict, update_model_status, …

v1_bp, logger = get_bp("v1_bp", __name__)


# ──────────────────────────────────────────────────────────────────────────
# auth
# ──────────────────────────────────────────────────────────────────────────
def _bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # OpenAI SDKs send Bearer; allow ?api_key= for quick curl tests too.
    return request.args.get("api_key")


def _openai_error(message: str, status: int, err_type: str = "invalid_request_error"):
    return jsonify({"error": {"message": message, "type": err_type, "code": status}}), status


def v1_auth(fn):
    """Enforce the site's key policy: open unless require_key is on."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if api_key_required() and not verify_api_key(_bearer_token()):
            return _openai_error(
                "Missing or invalid API key. Pass 'Authorization: Bearer <key>' "
                "(create keys in the console under API access).",
                401, "authentication_error",
            )
        return fn(*args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────────────────────────────────
# /v1/models
# ──────────────────────────────────────────────────────────────────────────
@v1_bp.route("/v1/models", methods=["GET"])
@v1_auth
def v1_models():
    manifest = get_models_dict(dict_return=True)
    data = []
    for key, model in manifest.items():
        model = update_model_status(model)
        if model.get("status") != "installed":
            continue
        data.append({
            "id": key,
            "object": "model",
            "created": 0,
            "owned_by": "hugpy",
            "hub_id": model.get("hub_id"),
            "task": model.get("primary_task") or model.get("task"),
            "context_length": model.get("model_max_length"),
        })
    return jsonify({"object": "list", "data": data})


# ──────────────────────────────────────────────────────────────────────────
# /v1/chat/completions
# ──────────────────────────────────────────────────────────────────────────
def _completion_kwargs(payload: dict) -> dict:
    messages = payload.get("messages")
    if not messages:
        raise ValueError("'messages' is required")
    # OpenAI clients must send *something* as model; "default" (and empty)
    # mean "no preference" — leave model_key unset so resolve() falls through
    # to the reconciled chat default instead of 404ing on a literal "default".
    model = payload.get("model")
    if isinstance(model, str) and model.strip().lower() in ("", "default"):
        model = None
    kwargs = {
        "messages": [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ],
        "model_key": model,
        "request_id": f"v1-{uuid.uuid4().hex}",
    }
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if max_tokens:
        # Explicit client cap → bounded; omitted → engine runs unbounded with
        # auto-continuation, same as the console.
        kwargs["max_new_tokens"] = int(max_tokens)
    if payload.get("temperature") is not None:
        kwargs["temperature"] = float(payload["temperature"])
        kwargs["do_sample"] = float(payload["temperature"]) > 0
    return kwargs


async def _v1_events(prompt_kwargs: dict):
    """Raw StreamEvents from the chat engine (late import dodges circulars)."""
    from ..functions.imports import execute_chat_stream
    async for event in execute_chat_stream(**prompt_kwargs):
        yield event


def _finish_reason(reason: str | None) -> str:
    return {"max_tokens": "length"}.get(reason or "stop", reason or "stop")


@v1_bp.route("/v1/chat/completions", methods=["POST"])
@v1_auth
def v1_chat_completions():
    payload = request.get_json(silent=True) or {}
    try:
        prompt_kwargs = _completion_kwargs(payload)
    except (ValueError, TypeError) as exc:
        return _openai_error(str(exc), 400)

    model = payload.get("model") or "default"
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if payload.get("stream"):
        def chunk(delta: dict, finish=None) -> bytes:
            return (
                "data: " + json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }, ensure_ascii=False) + "\n\n"
            ).encode("utf-8")

        async def sse():
            yield chunk({"role": "assistant", "content": ""})
            try:
                async for ev in _v1_events(prompt_kwargs):
                    t = getattr(ev, "type", None)
                    if t == "token":
                        yield chunk({"content": ev.text})
                    elif t == "done":
                        yield chunk({}, finish=_finish_reason(ev.finish_reason))
                    elif t == "error":
                        yield chunk({"content": f"\n[error: {ev.message}]"}, finish="stop")
            except Exception as exc:
                logger.exception("v1 stream failed")
                yield chunk({"content": f"\n[error: {exc}]"}, finish="stop")
            yield b"data: [DONE]\n\n"

        return Response(
            # heartbeat keeps a slow stream alive past an upstream proxy's read
            # timeout (the non-streaming drain below stays heartbeat-free).
            stream_with_context(chat_iter_sync(sse(), heartbeat=b": keepalive\n\n")),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # non-streaming: drain the same event stream and assemble one body
    text_parts: list[str] = []
    finish = "stop"
    error_message = None
    try:
        for ev in chat_iter_sync(_v1_events(prompt_kwargs)):
            t = getattr(ev, "type", None)
            if t == "token":
                text_parts.append(ev.text)
            elif t == "done":
                finish = _finish_reason(ev.finish_reason)
            elif t == "error":
                error_message = ev.message
    except KeyError as exc:
        # resolve() raises before the stream starts, e.g. unknown model
        return _openai_error(str(exc).strip("'\""), 404, "invalid_request_error")
    except Exception as exc:
        logger.exception("v1 completion failed")
        return _openai_error(f"{type(exc).__name__}: {exc}", 500, "api_error")

    if error_message and not text_parts:
        status = 404 if "Unknown model" in error_message else 500
        return _openai_error(error_message, status, "api_error")

    return jsonify({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(text_parts)},
            "finish_reason": finish,
        }],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    })


# ──────────────────────────────────────────────────────────────────────────
# auth config (console-side) — tells the UI how to authenticate users.
# mode "external": delegate to a separate login service (HUGPY_AUTH_BASE).
# mode "open":     single-operator instance, no login wall (distribution
#                  default; the /v1 key system still gates programmatic use).
# ──────────────────────────────────────────────────────────────────────────
@v1_bp.route("/auth/config", methods=["GET"])
def auth_config():
    import os as _os
    try:
        from abstract_hugpy_dev.imports.src.standalone_utils import get_env_value as _gev
    except Exception:
        _gev = lambda *_a, **_k: None
    mode = (_os.environ.get("HUGPY_AUTH_MODE") or _gev("HUGPY_AUTH_MODE") or "external").lower()
    if mode not in ("open", "external"):
        mode = "external"
    if mode != "external":
        return jsonify({"mode": mode, "base": None})
    # external mode: by default advertise a SAME-ORIGIN base so the UI talks to
    # our auth proxy (auth_proxy_routes.py) instead of the upstream auth service
    # directly. That keeps the session cookie first-party → Safari/Firefox stop
    # dropping it (the cross-site third-party-cookie login loop). Set
    # HUGPY_AUTH_PROXY=0 to fall back to advertising the upstream directly.
    from .auth_proxy_routes import proxy_enabled, public_base
    if proxy_enabled():
        base = _os.environ.get("HUGPY_AUTH_PUBLIC_BASE") or public_base()
    else:
        base = (_os.environ.get("HUGPY_AUTH_BASE") or _gev("HUGPY_AUTH_BASE")
                or "https://api.abstractendeavors.com")
    return jsonify({"mode": mode, "base": base})


# ──────────────────────────────────────────────────────────────────────────
# key management (console-side, same-origin; not part of the /v1 surface)
# ──────────────────────────────────────────────────────────────────────────
@v1_bp.route("/keys", methods=["GET"])
def keys_list():
    return jsonify({"require_key": api_key_required(), "keys": list_api_keys()})


@v1_bp.route("/keys", methods=["POST"])
def keys_create():
    body = request.get_json(silent=True) or {}
    # Optional `pool` binds the key to a dedicated worker pool so the app's
    # requests route to its reserved workers from the key alone.
    return jsonify(create_api_key(body.get("name", ""), pool=body.get("pool", "")))


@v1_bp.route("/keys/<key_id>", methods=["DELETE"])
def keys_revoke(key_id):
    if not revoke_api_key(key_id):
        return jsonify({"ok": False, "error": "unknown key id"}), 404
    return jsonify({"ok": True})


@v1_bp.route("/keys/require", methods=["PUT"])
def keys_require():
    body = request.get_json(silent=True) or {}
    return jsonify({"require_key": set_api_key_required(bool(body.get("require")))})
