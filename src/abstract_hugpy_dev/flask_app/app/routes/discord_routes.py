"""HTTP surface for model <-> Discord bindings (the bot's "mobile arm").

Two audiences, like worker_routes / phone_brick_routes:

  * the console UI (human-driven):
        GET    /discord/bindings
        POST   /discord/bindings        {"model_key", "channel_id"?, "user_id"?, "label"?}
        DELETE /discord/bindings/<id>
        POST   /discord/outbox          {"content", "channel_id"?|"user_id"?|"binding_id"?}
  * the hugpy bot (machine-to-machine, polls central):
        GET    /discord/resolve?channel_id=&user_id=   -> {"model_key": ... | null}
        POST   /discord/outbox/drain                   -> {"messages": [...]}
        POST   /discord/channels   {"channels":[{id,name,guild,guild_id}]}  (bot reports)
        GET    /discord/channels   -> {"channels":[...], "channels_at": ts}  (UI dropdown)

All state lives in functions.imports.utils.discord_bindings; this module only
translates HTTP <-> that store. The blueprint is discovered + mounted bare
(/discord/...) via routes/__init__, and dual-mounted under /api in
wsgi_app.get_hugpy_flask (the /api prefix is stripped by nginx in prod and by
ApiPrefixMiddleware on bare gunicorn), exactly like the GPU worker routes.
"""
from pydantic import BaseModel
from flask import request, jsonify, abort

from .imports import *  # get_bp + the functions star (get_models_dict, …)
from ..functions.imports.utils.discord_bindings import (
    list_bindings,
    add_binding,
    remove_binding,
    resolve_model,
    enqueue_outbound,
    drain_outbound,
    set_channels,
    get_channels,
    set_users,
    get_users,
    add_bridge,
    list_bridges,
    get_bridge,
    remove_bridge,
    bridge_for_channel,
    append_bridge_message,
    get_bridge_messages,
    update_bridge_message,
    add_session,
    list_sessions,
    revoke_session,
    session_by_token,
)
import asyncio
import inspect

discord_bp, logger = get_bp("discord_bp", __name__)


class BindingRequest(BaseModel):
    model_key: str
    channel_id: str | None = None
    user_id: str | None = None
    label: str | None = None


class OutboundRequest(BaseModel):
    content: str
    channel_id: str | None = None
    user_id: str | None = None
    binding_id: str | None = None


@discord_bp.route("/discord/bindings", methods=["GET"])
def discord_bindings_list():
    return jsonify({"bindings": list_bindings()})


@discord_bp.route("/discord/bindings", methods=["POST"])
def discord_bindings_create():
    body = BindingRequest(**(request.get_json(silent=True) or {}))
    if body.model_key not in get_models_dict(dict_return=True):
        abort(404, description="Unknown model key.")
    if not body.channel_id and not body.user_id:
        abort(400, description="Provide a channel_id and/or a user_id.")
    try:
        binding = add_binding(
            model_key=body.model_key,
            channel_id=body.channel_id,
            user_id=body.user_id,
            label=body.label,
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(binding), 201


@discord_bp.route("/discord/bindings/<binding_id>", methods=["DELETE"])
def discord_bindings_delete(binding_id):
    if not remove_binding(binding_id):
        abort(404, description="Unknown binding id.")
    return jsonify({"ok": True, "id": binding_id})


@discord_bp.route("/discord/resolve", methods=["GET"])
def discord_resolve():
    """The bot asks central which model an inbound (channel, user) should hit."""
    model_key = resolve_model(
        channel_id=request.args.get("channel_id"),
        user_id=request.args.get("user_id"),
    )
    return jsonify({"model_key": model_key})


@discord_bp.route("/discord/outbox", methods=["POST"])
def discord_outbox_enqueue():
    """Queue a model-originated message to be pushed into its Discord target."""
    body = OutboundRequest(**(request.get_json(silent=True) or {}))
    if not (body.content or "").strip():
        abort(400, description="content is required.")
    try:
        msg = enqueue_outbound(
            content=body.content,
            channel_id=body.channel_id,
            user_id=body.user_id,
            binding_id=body.binding_id,
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(msg), 201


@discord_bp.route("/discord/outbox/drain", methods=["POST"])
def discord_outbox_drain():
    """The bot polls this and delivers each returned message to Discord."""
    return jsonify({"messages": drain_outbound()})


@discord_bp.route("/discord/channels", methods=["GET"])
def discord_channels_list():
    """The console UI reads this to populate the channel dropdown."""
    return jsonify(get_channels())


@discord_bp.route("/discord/channels", methods=["POST"])
def discord_channels_report():
    """The bot reports the text channels it can currently see."""
    body = request.get_json(silent=True) or {}
    channels = body.get("channels")
    if not isinstance(channels, list):
        abort(400, description="channels must be a list.")
    return jsonify(set_channels(channels))


@discord_bp.route("/discord/users", methods=["GET"])
def discord_users_list():
    """The console UI reads this to populate the user dropdown."""
    return jsonify(get_users())


@discord_bp.route("/discord/users", methods=["POST"])
def discord_users_report():
    """The bot reports the guild members it can see (needs members intent)."""
    body = request.get_json(silent=True) or {}
    users = body.get("users")
    if not isinstance(users, list):
        abort(400, description="users must be a list.")
    return jsonify(set_users(users))


# ── bridges: a console session <-> a Discord channel ──────────────────────
class BridgeRequest(BaseModel):
    binding_id: str | None = None         # resolve channel/model from an existing binding
    channel_id: str | None = None         # …or specify directly
    model_key: str | None = None
    user_id: str | None = None
    directive: str | None = None
    defer_mode: str = "auto"              # auto | defer | directive
    brain: str = "model"                  # model (auto-reply) | keeper (keeper drives)
    keeper_target: str | None = None      # informational: which keeper is wired here


@discord_bp.route("/discord/bridges", methods=["GET"])
def discord_bridges_list():
    return jsonify({"bridges": list_bridges()})


@discord_bp.route("/discord/bridges", methods=["POST"])
def discord_bridges_create():
    body = BridgeRequest(**(request.get_json(silent=True) or {}))
    channel_id, model_key, user_id = body.channel_id, body.model_key, body.user_id
    if body.binding_id:
        match = next((b for b in list_bindings() if b.get("id") == body.binding_id), None)
        if not match:
            abort(404, description="Unknown binding id.")
        channel_id = channel_id or match.get("channel_id")
        user_id = user_id or match.get("user_id")
        model_key = model_key or match.get("model_key")
    if not channel_id:
        abort(400, description="Provide a binding_id or a channel_id.")
    try:
        bridge = add_bridge(channel_id=channel_id, model_key=model_key, user_id=user_id,
                            directive=body.directive, defer_mode=body.defer_mode,
                            brain=body.brain, keeper_target=body.keeper_target)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(bridge), 201


@discord_bp.route("/discord/bridges/<bridge_id>", methods=["DELETE"])
def discord_bridges_delete(bridge_id):
    if not remove_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    return jsonify({"ok": True, "id": bridge_id})


@discord_bp.route("/discord/bridges/<bridge_id>/messages", methods=["GET"])
def discord_bridge_messages(bridge_id):
    """The console polls this for the merged transcript (since a timestamp)."""
    if not get_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    try:
        since = float(request.args.get("since", "0") or 0)
    except ValueError:
        since = 0.0
    return jsonify({"messages": get_bridge_messages(bridge_id, since)})


@discord_bp.route("/discord/bridges/<bridge_id>/send", methods=["POST"])
def discord_bridge_send(bridge_id):
    """Console output for a bridge: record it AND push it into the Discord channel."""
    bridge = get_bridge(bridge_id)
    if not bridge:
        abort(404, description="Unknown bridge id.")
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        abort(400, description="content is required.")
    source = body.get("source") or "console"
    msg = append_bridge_message(bridge_id, direction="out", source=source,
                                content=content, author=body.get("author"))
    enqueue_outbound(content=content, channel_id=bridge.get("channel_id"),
                     user_id=bridge.get("user_id"))
    return jsonify(msg or {}), 201


# ── candidate generation (the bridge's "brain") ───────────────────────────
def _await_sync(value):
    """Drive a (possibly) awaitable execute_prompt result from WSGI.

    Uses the process-wide async runtime (one long-lived loop) rather than a
    fresh per-request loop — see _platform/async_runtime.
    """
    if not inspect.isawaitable(value):
        return value
    from abstract_hugpy_dev._platform import async_runtime
    return async_runtime.run(value)


def _result_text(result) -> str:
    if isinstance(result, dict):
        return result.get("text") or ""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d.get("text") or ""
            except TypeError:
                continue
    return getattr(result, "text", "") or ""


_DIRECTIVE_DECIDE = (
    "\n\nWhen you have drafted your reply: if the guidance above indicates you "
    "should let the operator review it before it is sent (e.g. uncertainty, or a "
    "sensitive / out-of-scope topic), begin the reply with 'DEFER: '. Otherwise "
    "reply normally."
)


def _generate_candidate(bridge: dict) -> str:
    """A model reply built from the bridge's directive + recent sent transcript.
    Returns '' on any failure so the caller falls back to manual operation."""
    model_key = bridge.get("model_key")
    if not model_key:
        return ""
    directive = (bridge.get("directive") or
                 "You are relaying a Discord channel on the operator's behalf. Reply concisely.")
    if bridge.get("defer_mode") == "directive":
        directive = directive + _DIRECTIVE_DECIDE
    msgs = [{"role": "system", "content": directive}]
    for m in get_bridge_messages(bridge["id"]):
        if m.get("status") != "sent":
            continue  # skip pending/rejected — not part of the real conversation
        role = "user" if m.get("direction") == "in" else "assistant"
        msgs.append({"role": role, "content": m.get("content") or ""})
    if len(msgs) > 13:                       # cap history (system + last 12 turns)
        msgs = [msgs[0]] + msgs[-12:]
    from ..functions.imports import execute_prompt
    result = _await_sync(execute_prompt(model_key=model_key, messages=msgs,
                                        task="text-generation"))
    return (_result_text(result) or "").strip()


@discord_bp.route("/discord/inbox", methods=["POST"])
def discord_inbox():
    """The bot posts inbound channel messages here. For a bridged channel we
    record the message, generate a candidate reply from the bridge's directive,
    then apply defer_mode: auto-send, hold for the operator, or let the model
    decide (directive mode)."""
    body = request.get_json(silent=True) or {}
    channel_id = body.get("channel_id")
    content = (body.get("content") or "").strip()
    bridge = bridge_for_channel(channel_id) if channel_id else None
    if not bridge:
        return jsonify({"bridged": False})
    if not content:
        return jsonify({"bridged": True, "bridge_id": bridge["id"],
                        "defer_mode": bridge.get("defer_mode"), "action": "none"})

    append_bridge_message(bridge["id"], direction="in", source="discord",
                          content=content, author=body.get("author"))

    # keeper-brained bridges: an attached keeper process polls the transcript
    # and drives replies itself (via /keeper-reply), so central only records the
    # inbound turn here — no auto-candidate.
    if bridge.get("brain", "model") == "keeper":
        return jsonify({"bridged": True, "bridge_id": bridge["id"],
                        "brain": "keeper", "defer_mode": bridge.get("defer_mode"),
                        "action": "await_keeper"})

    action = "none"
    try:
        candidate = _generate_candidate(bridge)
    except Exception:
        logger.exception("bridge candidate generation failed")
        candidate = ""

    action = _route_bridge_reply(bridge, candidate, source="model")
    return jsonify({"bridged": True, "bridge_id": bridge["id"],
                    "defer_mode": bridge.get("defer_mode"), "action": action})


def _route_bridge_reply(bridge: dict, candidate: str, *, source: str) -> str:
    """Apply a bridge's defer_mode to an outbound reply: hold it for operator
    approval (pending) or send it (enqueue to Discord). Shared by the model
    auto-candidate path and the keeper-reply path so both gate identically.
    Returns the action taken: 'none' | 'pending' | 'sent'."""
    candidate = (candidate or "").strip()
    if not candidate:
        return "none"
    mode = bridge.get("defer_mode", "auto")
    defer = (mode == "defer")
    if mode == "directive" and candidate.upper().startswith("DEFER"):
        defer = True
        candidate = candidate.split(":", 1)[1].strip() if ":" in candidate else candidate
    if not candidate:
        return "none"
    if defer:
        append_bridge_message(bridge["id"], direction="out", source=source,
                              content=candidate, status="pending")
        return "pending"
    append_bridge_message(bridge["id"], direction="out", source=source,
                          content=candidate, status="sent")
    enqueue_outbound(content=candidate, channel_id=bridge.get("channel_id"),
                     user_id=bridge.get("user_id"))
    return "sent"


@discord_bp.route("/discord/bridges/<bridge_id>/keeper-reply", methods=["POST"])
def discord_bridge_keeper_reply(bridge_id):
    """A keeper submits a drafted reply for a keeper-brained bridge. The bridge's
    defer_mode decides what happens: user-strict (defer) holds it as a pending
    candidate for operator approval in the console; keeper-choice (directive)
    lets the keeper send directly or escalate with a 'DEFER:' prefix; auto sends.
    The keeper never reaches Discord unreviewed under user-strict."""
    bridge = get_bridge(bridge_id)
    if not bridge:
        abort(404, description="Unknown bridge id.")
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        abort(400, description="content is required.")
    action = _route_bridge_reply(bridge, content, source="keeper")
    return jsonify({"bridge_id": bridge_id, "defer_mode": bridge.get("defer_mode"),
                    "action": action}), 201


@discord_bp.route("/discord/bridges/<bridge_id>/approve", methods=["POST"])
def discord_bridge_approve(bridge_id):
    """Approve (optionally edit) a pending candidate and send it to Discord."""
    bridge = get_bridge(bridge_id)
    if not bridge:
        abort(404, description="Unknown bridge id.")
    body = request.get_json(silent=True) or {}
    msg_id = body.get("message_id")
    if not msg_id:
        abort(400, description="message_id is required.")
    edited = body.get("content")
    msg = update_bridge_message(bridge_id, msg_id, status="sent",
                                content=(edited.strip() if isinstance(edited, str) and edited.strip() else None))
    if not msg:
        abort(404, description="Unknown message id.")
    enqueue_outbound(content=msg["content"], channel_id=bridge.get("channel_id"),
                     user_id=bridge.get("user_id"))
    return jsonify(msg)


@discord_bp.route("/discord/bridges/<bridge_id>/reject", methods=["POST"])
def discord_bridge_reject(bridge_id):
    """Discard a pending candidate without sending it."""
    if not get_bridge(bridge_id):
        abort(404, description="Unknown bridge id.")
    msg_id = (request.get_json(silent=True) or {}).get("message_id")
    if not msg_id:
        abort(400, description="message_id is required.")
    msg = update_bridge_message(bridge_id, msg_id, status="rejected")
    if not msg:
        abort(404, description="Unknown message id.")
    return jsonify(msg)


# ── comms sessions: scoped bearer endpoints for terminal agents ────────────
# The operator mints a session bound to one channel (POST /discord/sessions —
# operator-gated) and hands the returned token to a terminal/agent session.
# The holder gets exactly two verbs on /discord/session/<token>/…: read the
# channel transcript and send into the channel. Revocable, optionally
# expiring, and the store only keeps the token's sha256. NOTE: the token
# rides in the URL path for paste-ability — it will appear in front-proxy
# access logs, which are operator-controlled here; rotate via revoke+mint.

class SessionMintRequest(BaseModel):
    channel_id: str
    label: str | None = None
    ttl_hours: float | None = None
    author: str | None = None


# Discord hard-caps a message at 2000 chars; refuse instead of letting the
# bot's delivery fail silently later.
_SESSION_MSG_LIMIT = 1900


@discord_bp.route("/discord/sessions", methods=["POST"])   # operator-gated
def discord_sessions_mint():
    body = SessionMintRequest(**(request.get_json(silent=True) or {}))
    # Inbound needs the channel bridged; reuse the existing bridge or create a
    # pure relay (keeper brain — never auto-generates a model reply).
    bridge = bridge_for_channel(body.channel_id)
    if not bridge:
        bridge = add_bridge(channel_id=body.channel_id, brain="keeper",
                            keeper_target=f"session:{(body.label or '').strip() or 'terminal'}")
    try:
        token, session = add_session(channel_id=body.channel_id,
                                     label=body.label or "",
                                     ttl_hours=body.ttl_hours,
                                     author=body.author)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify({"token": token,           # shown exactly once
                    "session": session,
                    "bridge_id": bridge["id"],
                    "endpoint": f"/discord/session/{token}"}), 201


@discord_bp.route("/discord/sessions", methods=["GET"])    # operator-gated
def discord_sessions_list():
    return jsonify({"sessions": list_sessions()})


@discord_bp.route("/discord/sessions/<session_id>", methods=["DELETE"])  # operator-gated
def discord_sessions_revoke(session_id):
    if not revoke_session(session_id):
        abort(404, description="Unknown or already-revoked session id.")
    return jsonify({"ok": True, "id": session_id})


def _session_or_404(token: str) -> dict:
    s = session_by_token(token)
    if not s:
        abort(404)  # bad, revoked and expired are indistinguishable — no oracle
    return s


@discord_bp.route("/discord/session/<token>", methods=["GET"])
def discord_session_info(token):
    """Self-describing so the endpoint alone is enough to hand to an agent."""
    s = _session_or_404(token)
    chans = (get_channels() or {}).get("channels") or []
    name = next((c.get("name") for c in chans
                 if str(c.get("id")) == str(s["channel_id"])), None)
    return jsonify({
        "channel_id": s["channel_id"],
        "channel": name,
        "label": s.get("label") or "",
        "bridged": bool(bridge_for_channel(s["channel_id"])),
        "usage": {
            "poll": "GET …/messages?since=<ts float from last message> → {\"messages\":[…]}",
            "send": f"POST …/send {{\"content\":\"…\"}} (≤{_SESSION_MSG_LIMIT} chars) → bot delivers within ~8s",
        },
    })


@discord_bp.route("/discord/session/<token>/messages", methods=["GET"])
def discord_session_messages(token):
    s = _session_or_404(token)
    bridge = bridge_for_channel(s["channel_id"])
    if not bridge:
        return jsonify({"messages": [],
                        "warning": "channel bridge removed — inbound relay is off"})
    try:
        since = float(request.args.get("since", "0") or 0)
    except ValueError:
        since = 0.0
    return jsonify({"messages": get_bridge_messages(bridge["id"], since)})


@discord_bp.route("/discord/session/<token>/send", methods=["POST"])
def discord_session_send(token):
    s = _session_or_404(token)
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        abort(400, description="content is required.")
    if len(content) > _SESSION_MSG_LIMIT:
        abort(413, description=f"content exceeds {_SESSION_MSG_LIMIT} chars — split it.")
    author = (body.get("author") or s.get("author") or s.get("label") or "session").strip()
    bridge = bridge_for_channel(s["channel_id"])
    msg = None
    if bridge:  # record in the transcript so other session holders see it
        msg = append_bridge_message(bridge["id"], direction="out", source="session",
                                    content=content, author=author)
    enqueue_outbound(content=content, channel_id=s["channel_id"])
    return jsonify({"ok": True, "message": msg}), 201
