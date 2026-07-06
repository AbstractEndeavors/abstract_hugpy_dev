"""Server-side operator authentication gate for console-side management routes.

WHY
---
Authentication for the console used to be enforced ONLY in the React UI. The
Flask layer was wide open, so anyone who could reach the API origin could mint
API keys, mint/revoke worker enrollment tokens, admit/assign workers (→ prompt
exfiltration + SSRF), and drive the Discord console — all unauthenticated. This
gate closes that by validating the operator at the server.

DESIGN
------
A single ``before_request`` matches the request against an explicit allowlist of
SENSITIVE routes (operator-only, mutating, or secret-bearing) and requires
operator auth for those. Everything else — health/readiness, model reads,
inference (``/v1`` is gated by the API-key system), and the machine-to-machine
endpoints the worker / bot / phone arms depend on — is left untouched, so this
never breaks those flows.

Auth modes (resolved from ``HUGPY_AUTH_MODE``, same as ``/auth/config``):
  * ``external`` — validate the first-party session cookie by forwarding it to
    the upstream auth service's ``/me`` (the same session the React UI uses),
    with a short positive/negative cache. A configured ``HUGPY_OPERATOR_TOKEN``
    is also accepted (CLI/automation). Fails CLOSED (deny) if the auth service
    is unreachable — a sensitive route must not open up during an auth outage.
  * ``open`` — the self-hosted single-operator default (``pip install hugpy``).
    Permissive UNLESS ``HUGPY_OPERATOR_TOKEN`` is set, in which case that token
    is required. So the localhost product keeps its no-login UX, while a public
    open deployment can still lock the management surface with one env var.

The gate only ENFORCES in external mode (or when an operator token is set), so
installing it changes nothing until the operator flips ``HUGPY_AUTH_MODE`` —
making rollout safe to deploy and verify before activation.
"""
from __future__ import annotations

import os
import re
import time
import hashlib
import logging

from flask import request, abort

logger = logging.getLogger(__name__)

# Sensitive routes: (allowed-methods-that-require-auth, normalized-path regex).
# Paths are matched AFTER stripping a leading "/api" (gunicorn dual-mounts the
# worker/discord/phone blueprints under /api as well as bare). Only the listed
# methods are gated, so e.g. GET /discord/bridges (bot M2M) stays open while
# POST /discord/bridges (operator) is gated.
_SENSITIVE = [
    # API key management (key minting was anonymously reachable — CRITICAL)
    ({"GET", "POST"},            re.compile(r"^/keys$")),
    ({"DELETE"},                 re.compile(r"^/keys/[^/]+$")),
    ({"PUT"},                    re.compile(r"^/keys/require$")),
    # Worker enrollment tokens (minting/revoking enrollment — CRITICAL)
    ({"GET", "POST"},            re.compile(r"^/llm/enroll-tokens$")),
    ({"DELETE"},                 re.compile(r"^/llm/enroll-tokens/[^/]+$")),
    # Worker admission / control — operator actions (register & heartbeat are
    # M2M and deliberately NOT here). Admission is what makes a worker
    # dispatch-eligible, so gating it closes anonymous self-admission → SSRF.
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/(admit|block|admission|assign|unassign|unload|probe|pool|limits)$")),
    ({"DELETE"},                 re.compile(r"^/llm/workers/[^/]+$")),
    # Serving / slot control (operator) — the GET status reads stay open.
    ({"POST"},                   re.compile(r"^/llm/serving/[^/]+$")),
    ({"POST"},                   re.compile(r"^/llm/slots/(load|unload)$")),
    # File uploads are intentionally NOT operator-gated: the media-intelligence
    # arm needs them for any authenticated user (upload -> /ml/vision|/ml/extract).
    # Same exposure tier as /chat/stream and /ml/* — the user-facing product routes.
    # Discord HUMAN console routes. The bot's M2M calls (GET /discord/resolve,
    # POST /discord/outbox/drain, POST /discord/channels, POST /discord/users,
    # POST /discord/inbox, GET /discord/bridges) are intentionally excluded.
    ({"GET", "POST", "DELETE"},  re.compile(r"^/discord/bindings(/[^/]+)?$")),
    ({"POST"},                   re.compile(r"^/discord/bridges$")),
    ({"DELETE"},                 re.compile(r"^/discord/bridges/[^/]+$")),
    ({"POST"},                   re.compile(r"^/discord/bridges/[^/]+/(send|keeper-reply|approve|reject)$")),
    ({"GET"},                    re.compile(r"^/discord/bridges/[^/]+/messages$")),
    # Comms sessions: minting/listing/revoking the scoped bearer tokens is
    # operator-only. The /discord/session/<token>/… verbs are deliberately NOT
    # here — the session token IS their credential (same rationale as
    # principal tokens below).
    ({"GET", "POST"},            re.compile(r"^/discord/sessions$")),
    ({"DELETE"},                 re.compile(r"^/discord/sessions/[^/]+$")),
    # F2 principals: minting identities/tokens is operator-only. The
    # /auth/discord-link handshake and /auth/whoami stay open — the principal
    # token IS their credential.
    ({"GET", "POST"},            re.compile(r"^/auth/principals$")),
    ({"DELETE"},                 re.compile(r"^/auth/principals/[^/]+$")),
    ({"POST"},                   re.compile(r"^/auth/principals/[^/]+/token$")),
    # F4 settings: reads stay open (UIs render from them); writes are the
    # console's authoritative control plane (CON-08) -> operator-only.
    ({"POST", "PUT", "DELETE"},  re.compile(r"^/settings/.+$")),
    # Worker ops (CON-05/06, UTIL-02): restart / module update / pip install /
    # serving-config are privileged executor actions on a worker —
    # operator-only, audited. (config added 2026-07-03: it re-execs the agent
    # and rewrites its runtime settings — same privilege tier as update.)
    # pin-all/unpin-all relay the SAME /ops/config write in bulk (see
    # worker_routes._relay_pin_all) — same privilege tier as config.
    # reap-approve = operator-approved eviction of cold local models (drives the
    # same guarded reaper as /reap, with a central intersection second guard).
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/(restart|update|pip|config|reap|reap-approve|pin-all|unpin-all)$")),
    # Civitai checkpoint download — writes multi-GB files into central's
    # /checkpoints store (which self-registers models) — operator-only.
    ({"POST"},                   re.compile(r"^/civitai/download$")),
    # Disk discovery sweep — rebuilds the discovery report (walks the whole
    # model tree + hub enrichment); the GET state poll stays open.
    ({"POST"},                   re.compile(r"^/models/discover$")),
]

_SESSION_CACHE: dict[str, tuple[bool, float]] = {}
_CACHE_TTL = 30.0


def _auth_mode() -> str:
    mode = (os.environ.get("HUGPY_AUTH_MODE") or "external").lower()
    return mode if mode in ("open", "external") else "external"


def _operator_token() -> str:
    return (os.environ.get("HUGPY_OPERATOR_TOKEN") or "").strip()


def _provided_token() -> str:
    t = request.headers.get("X-Operator-Token")
    if t:
        return t.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _validate_session_external() -> bool:
    """True iff the request's cookies authenticate against the upstream /me."""
    cookie_hdr = request.headers.get("Cookie", "")
    if not cookie_hdr:
        return False
    key = hashlib.sha256(cookie_hdr.encode("utf-8")).hexdigest()
    now = time.time()
    cached = _SESSION_CACHE.get(key)
    if cached and cached[1] > now:
        return cached[0]
    ok = False
    try:
        import requests
        from .routes.auth_proxy_routes import upstream_base
        resp = requests.get(f"{upstream_base()}/me", cookies=request.cookies, timeout=8)
        if resp.status_code == 200:
            try:
                data = resp.json()
                ok = bool(data) and not (isinstance(data, dict) and data.get("error"))
            except Exception:
                ok = True
    except Exception as exc:
        # Fail closed: a sensitive route must not open up if auth is unreachable.
        logger.warning("operator session validation failed (auth service): %s", exc)
        return False
    _SESSION_CACHE[key] = (ok, now + _CACHE_TTL)
    return ok


def operator_authenticated() -> bool:
    mode = _auth_mode()
    tok = _operator_token()
    if tok and _provided_token() and _provided_token() == tok:
        return True
    if mode == "open":
        # Self-hosted single-operator default: permissive unless a token is set.
        return not tok
    return _validate_session_external()


def _path_is_sensitive() -> bool:
    path = request.path or "/"
    if path == "/api" or path.startswith("/api/"):
        path = path[len("/api"):] or "/"
    method = request.method
    for methods, rx in _SENSITIVE:
        if method in methods and rx.match(path):
            return True
    return False


def install_operator_gate(app) -> None:
    """Register the before_request gate on a Flask app (idempotent)."""
    if getattr(app, "_operator_gate_installed", False):
        return
    app._operator_gate_installed = True

    @app.before_request
    def _operator_gate():
        if request.method == "OPTIONS":
            return None  # never block CORS preflight
        if not _path_is_sensitive():
            return None
        if not operator_authenticated():
            abort(401, description="Operator authentication required for this route.")
        return None

    logger.info("operator auth gate installed (mode=%s, token_set=%s)",
                _auth_mode(), bool(_operator_token()))
