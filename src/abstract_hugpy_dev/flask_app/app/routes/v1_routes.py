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
# Default media-chat model_key (single global value); explicit import — not in the functions star-export.
from ....imports.config.models.models_config import media_default_state
# k9 video-share key store (its OWN category/store — never the /v1 api_keys store).
from ..functions.imports.utils.video_share_keys import (
    create_share_key,
    list_share_keys,
    revoke_share_key,
)
# Pure request/response plumbing lives in v1_helpers (stdlib-only, no Flask)
# so it unit-tests offline; see that module's docstring.
from .v1_helpers import (
    _build_tools_preamble,
    _completion_kwargs,
    _inject_tools_preamble,
    _parse_tool_calls,
    _usage_block,
)

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
    media_default = media_default_state()
    # Operator BLOCK set (guarded — a listing must never 500 over the blocklist).
    try:
        from abstract_hugpy_dev.comms.blocklist import blocked_keys
        _blocked = blocked_keys()
    except Exception:  # noqa: BLE001
        _blocked = set()
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
            # FULL capability list — `task` (primary) alone hid secondary
            # capabilities from every task-filtered UI (e.g. a dual
            # text-to-image + image-to-image model looked t2i-only).
            "tasks": model.get("tasks") or ([model.get("primary_task")] if model.get("primary_task") else []),
            "context_length": model.get("model_max_length"),
            "media_default": (key == media_default),
            # Additive: ⛔ blocked from the serving pool by the operator. A call
            # naming a blocked model fails fast with the distinct blocked reason.
            "blocked": (key in _blocked),
        })
    return jsonify({"object": "list", "data": data})


# ──────────────────────────────────────────────────────────────────────────
# /v1/chat/completions
# (payload -> prompt_kwargs translation is _completion_kwargs in v1_helpers)
# ──────────────────────────────────────────────────────────────────────────
async def _v1_events(prompt_kwargs: dict):
    """Raw StreamEvents from the chat engine (late import dodges circulars).

    Registered in the live queue (same as the console /chat/stream path) so /v1
    (OpenAI-compatible) traffic shows in the activity view too. Best-effort —
    queue bookkeeping must never break a completion."""
    from ..functions.imports import execute_chat_stream
    from abstract_hugpy_dev.managers.dispatch import activity
    rid = prompt_kwargs.get("request_id")
    mk = prompt_kwargs.get("model_key")
    name = mk
    try:
        from ..functions.imports import get_model_config
        if mk:
            name = getattr(get_model_config(mk), "name", None) or mk
    except Exception:
        pass
    activity.begin(rid, mk, name, kind="v1")
    try:
        async for event in execute_chat_stream(**prompt_kwargs):
            if getattr(event, "type", None) == "token":
                activity.on_token(rid)
            yield event
    finally:
        activity.end(rid)


def _finish_reason(reason: str | None) -> str:
    return {"max_tokens": "length"}.get(reason or "stop", reason or "stop")


@v1_bp.route("/v1/chat/completions", methods=["POST"])
@v1_auth
def v1_chat_completions():
    payload = request.get_json(silent=True) or {}

    # Central-side tools shim (see v1_helpers): the frozen engine schema can't
    # carry `tools`, so tool-calling is prompt-injected here and parsed back
    # out of the reply — every GGUF model gains it with no engine change.
    # tool_choice "none" (or no usable tool entries) leaves tools_preamble
    # None and the request behaves exactly as today.
    tools_preamble = _build_tools_preamble(payload.get("tools"),
                                           payload.get("tool_choice"))
    if tools_preamble and payload.get("messages"):
        payload = dict(payload)
        payload["messages"] = _inject_tools_preamble(payload["messages"],
                                                     tools_preamble)

    try:
        prompt_kwargs = _completion_kwargs(payload)
    except (ValueError, TypeError) as exc:
        return _openai_error(str(exc), 400)

    # REJECT-AT-INTAKE (slice 9, defect 3): a request naming a model that resolves
    # to NOTHING (not in the registry/catalog, no worker designation) must be
    # rejected NOW with a 4xx + known-keys hint — never accepted into a
    # pending-forever job (the operator's immortal flux2-klein row). We reuse the
    # DISPATCHER'S OWN resolution (resolve_model_key -> assure_model_key), so the
    # boundary is exact: a model that IS known but merely not local/loaded still
    # RESOLVES (registry membership, not disk presence) and queues — lazy download
    # is the design. Only a truly-unresolvable explicit key rejects. A None/
    # "default" model_key (no preference) is left for the engine to default.
    _mk = prompt_kwargs.get("model_key")
    if _mk:
        try:
            from abstract_hugpy_dev.managers.resolvers.model_resolver import (
                resolve_model_key)
            resolve_model_key(model_key=_mk)
        except KeyError as exc:
            # KeyError carries the "Unknown model_key=...; known: [...]" hint.
            return _openai_error(str(exc).strip("'\""), 400)
        except Exception:
            # A non-resolution error (registry probe failed) must NOT reject a
            # legitimate request — fall through and let the engine try.
            pass

    # A tool call is one short, bounded turn — never auto-continue it. A
    # continuation pass is exactly what rambled the captured 2026-07-14
    # incident and would splice "Continue…" text into the JSON block. An
    # explicit client max_chunks still wins.
    if tools_preamble and "max_chunks" not in prompt_kwargs:
        prompt_kwargs["max_chunks"] = 1

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

        # OpenAI semantics: usage rides in ONE extra final chunk (choices: []),
        # and only when the client opted in via stream_options.include_usage.
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

        def usage_chunk(usage) -> bytes:
            return (
                "data: " + json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": _usage_block(usage),
                }, ensure_ascii=False) + "\n\n"
            ).encode("utf-8")

        async def sse():
            usage = None
            yield chunk({"role": "assistant", "content": ""})
            # Streaming with tools buffers the whole reply and parses at done —
            # the simplest CORRECT behavior: a <tool_call> block is only
            # recognizable once complete, and OpenAI SDKs accept the final
            # burst. A fully-incremental tool-call stream (deltas per argument
            # fragment) is a later refinement. Non-tool requests stream
            # token-by-token exactly as before.
            buffered: list = []
            try:
                async for ev in _v1_events(prompt_kwargs):
                    t = getattr(ev, "type", None)
                    if t == "token":
                        if tools_preamble:
                            buffered.append(ev.text)
                        else:
                            yield chunk({"content": ev.text})
                    elif t == "done":
                        usage = getattr(ev, "usage", None)
                        finish = _finish_reason(ev.finish_reason)
                        if tools_preamble:
                            clean_text, tool_calls = _parse_tool_calls("".join(buffered))
                            buffered = []
                            if tool_calls:
                                yield chunk({"tool_calls": [
                                    {**tc, "index": i}
                                    for i, tc in enumerate(tool_calls)
                                ]})
                                finish = "tool_calls"
                            elif clean_text:
                                # No call — the buffered reply is plain content.
                                yield chunk({"content": clean_text})
                        yield chunk({}, finish=finish)
                    elif t == "error":
                        if buffered:
                            yield chunk({"content": "".join(buffered)})
                            buffered = []
                        yield chunk({"content": f"\n[error: {ev.message}]"}, finish="stop")
            except Exception as exc:
                logger.exception("v1 stream failed")
                if buffered:
                    yield chunk({"content": "".join(buffered)})
                yield chunk({"content": f"\n[error: {exc}]"}, finish="stop")
            if include_usage:
                yield usage_chunk(usage)
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
    usage = None
    try:
        for ev in chat_iter_sync(_v1_events(prompt_kwargs)):
            t = getattr(ev, "type", None)
            if t == "token":
                text_parts.append(ev.text)
            elif t == "done":
                finish = _finish_reason(ev.finish_reason)
                usage = getattr(ev, "usage", None)
            elif t == "error":
                error_message = ev.message
    except KeyError as exc:
        # resolve() raises before the stream starts, e.g. unknown model
        return _openai_error(str(exc).strip("'\""), 404, "invalid_request_error")
    except Exception as exc:
        logger.exception("v1 completion failed")
        return _openai_error(f"{type(exc).__name__}: {exc}", 500, "api_error")

    if error_message and not text_parts:
        # Cap-aware relay gate (concurrency hardening): a busy in-process runner
        # is not a fault — every holder of the model is momentarily at its safe
        # concurrency limit. Answer 503 (retryable) so a batch client backs off
        # instead of treating it as a hard error.
        if "worker_busy" in error_message or "model_busy" in error_message:
            return _openai_error(error_message, 503, "server_busy")
        status = 404 if "Unknown model" in error_message else 500
        return _openai_error(error_message, status, "api_error")

    content = "".join(text_parts)
    message = {"role": "assistant", "content": content}
    if tools_preamble:
        # Errors-as-data: _parse_tool_calls returns (original text, None) on
        # no/malformed calls, so the worst case is a plain content answer —
        # a shim parse failure can never 500 the route.
        clean_text, tool_calls = _parse_tool_calls(content)
        if tool_calls:
            message = {"role": "assistant", "content": clean_text or None,
                       "tool_calls": tool_calls}
            finish = "tool_calls"

    return jsonify({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        # Real token accounting threaded up from the runner via the done
        # event; all-None only when genuinely unavailable (never a crash).
        "usage": _usage_block(usage),
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


# ──────────────────────────────────────────────────────────────────────────
# k9 — VIDEO-SHARE links. Mint a video-scoped share credential (a NEW key
# category, its own store — see functions/.../video_share_keys.py) and hand its
# link to an outside party so they can drive the /video features WITHOUT a
# console login. These routes are OPERATOR-ONLY (operator_auth._SENSITIVE gates
# ^/keys/video-share) and are deliberately NOT on the /video surface, so a
# share principal (which can pass the /video gate) can never reach them —
# structurally "no key-minting-by-key". The GET doubles as the SPA's auth probe.
# ──────────────────────────────────────────────────────────────────────────
def _public_base() -> str:
    """Best-effort public origin for building a share URL. Prefers an explicit
    HUGPY_PUBLIC_BASE; else reconstructs from the forwarded host/proto (the SPA
    also rebuilds the link from window.location.origin, so this is the
    curl/programmatic path)."""
    import os as _os
    base = (_os.environ.get("HUGPY_PUBLIC_BASE") or "").strip()
    if base:
        return base.rstrip("/")
    proto = (request.headers.get("X-Forwarded-Proto") or request.scheme or "https").split(",")[0].strip()
    host = (request.headers.get("X-Forwarded-Host") or request.host or "").split(",")[0].strip()
    return f"{proto}://{host}".rstrip("/") if host else ""


@v1_bp.route("/keys/video-share", methods=["GET"])
def video_share_list():
    return jsonify({"keys": list_share_keys()})


@v1_bp.route("/keys/video-share", methods=["POST"])
def video_share_create():
    body = request.get_json(silent=True) or {}
    ttl = body.get("ttl_days", None)
    minted = create_share_key(label=body.get("label", ""),
                              ttl_days=(ttl if ttl is not None else 30))
    base = _public_base()
    minted["url"] = f"{base}/video/?share={minted['key']}" if base else \
        f"/video/?share={minted['key']}"
    return jsonify(minted)


@v1_bp.route("/keys/video-share/<key_id>", methods=["DELETE"])
def video_share_revoke(key_id):
    if not revoke_share_key(key_id):
        return jsonify({"ok": False, "error": "unknown key id"}), 404
    return jsonify({"ok": True})
