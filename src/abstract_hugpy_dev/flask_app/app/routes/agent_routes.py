"""P3.1 — the `/agent/*` agent-node registry + dispatch blueprint.

Phase 3 turns central into the head of a *fleet of agents*: remote P2.7
daemons enroll and heartbeat like GPU workers / phone bricks, the operator
dispatches a task to a node and watches it, and the node pulls its queue.

Like the worker and phone-brick blueprints, this serves two audiences — and
the split (public machine-to-machine vs operator-only) is deliberate, the same
public-vs-internal curation the /endpoints inspector surfaces:

  * the nodes (machine-to-machine):
        POST /agent/register              enroll -> {id, token, ...}   (bootstrap; open)
        POST /agent/<id>/heartbeat        {status, current_task, version}   (node token)
        GET  /agent/<id>/tasks?since=     pull queued tasks            (node token)
        POST /agent/<id>/tasks/<seq>/result  {status, result} -> finalize  (node token; P3.1b)
  * the console UI (operator, human-driven):
        GET  /agent/nodes                 list every node + live status  (operator)
        GET  /agent/<id>/tasks/<seq>      one task's full row incl result (operator; P3.1b)
        POST /agent/<id>/dispatch         {task} -> queue it for a node  (operator)
  * the terminal (operator, human-driven, no browser):
        GET  /agent/client.sh             serve the bash dispatch client   (open)
  * secure one-time install links (2026-07-23):
        POST   /agent/install-links            mint scoped key + link   (operator, strict)
        GET    /agent/install-links            list links + status      (operator, strict)
        DELETE /agent/install-links/<id>       revoke link AND its key  (operator, strict)
        GET    /agent/install/<link_id>        one-time templated .py download (link capability)
        GET    /agent/install/<link_id>.sh     POSIX wrapper (free fetch; .py is the use)
        GET    /agent/install/<link_id>.ps1    Windows wrapper (free fetch; .py is the use)

Gates, all fail-closed:
  * ``register`` is the unauthenticated bootstrap (a node has no credential
    yet) — it MINTS the node's enroll token, returned exactly once. (A future
    slice can front it with a pre-shared enrollment secret, exactly like the
    GPU workers' HUGPY_WORKER_ENROLL_REQUIRED gate; the spec issues the
    credential here, so today it is open like /phone-brick/register.)
  * ``heartbeat`` and ``tasks`` are node-authenticated: the caller must present
    THIS node's enroll token (``Authorization: Bearer <token>`` or
    ``X-Agent-Token``). Missing/mismatched -> 401; a node central has forgotten
    -> 410 (re-register); a revoked node -> 403.
  * ``nodes`` and ``dispatch`` are operator-only, via ``operator_authenticated``
    — the exact gate the console-management routes use (and additionally listed
    in operator_auth._SENSITIVE so the central before_request gate also covers
    them). It fails closed in external mode / whenever an operator token is set.
  * ``client.sh`` is unauthenticated, same rationale as the worker/phone-brick
    bootstrap scripts: it is plain client code with no embedded secret. The
    dispatch/nodes calls IT makes are still operator-gated as above — the
    script reads the operator token from the caller's own environment.

All node state lives in comms.agent_nodes (the shared comms SQLite db — cross
-process, gunicorn 3-worker safe), never per-process memory. Nodes may reach
these endpoints over nginx (/api stripped) or directly over the VPN; the /api
dual-mount in wsgi_app.py makes both resolve, exactly like the GPU workers.
"""
import json
import os

from flask import request, jsonify, abort

from .imports import *  # get_bp + the functions star
from ....comms.agent_nodes import agent_node_store

agent_bp, logger = get_bp("agent_bp", __name__)


# ── credential extraction ──────────────────────────────────────────────────
def _agent_token() -> str | None:
    """The node's enroll token, from Authorization: Bearer <token> (the M2M
    credential the workers already use) or the X-Agent-Token convenience
    header. Never logged."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok:
            return tok
    tok = (request.headers.get("X-Agent-Token") or "").strip()
    return tok or None


def _require_node_auth(node_id: str) -> dict:
    """Authenticate a node-facing request and return the node's public view.

    Fail-closed with honest codes: unknown node -> 410 (re-register), revoked
    node -> 403, missing/mismatched token -> 401. Aborts on any failure; on
    success returns the node dict."""
    node = agent_node_store.get(node_id)
    if node is None:
        abort(410, description="Unknown agent node id; please re-register.")
    if node.get("revoked"):
        abort(403, description="Agent node revoked by the operator.")
    if not agent_node_store.authenticate(node_id, _agent_token()):
        abort(401, description="Agent node token invalid or required.")
    return node


def _agent_gates_open() -> bool:
    """OPERATOR-DIRECTED open mode (2026-07-15: "can the agents feature be ungated
    entirely for now?"): ``HUGPY_AGENT_OPEN`` truthy waives the OPERATOR gate on
    ``/agent/nodes`` + ``/agent/<id>/dispatch`` for local testing.

    ⚠️ SCOPE NARROWED 2026-07-16 (operator: *"if the hugpy_agent_open is bypassing
    gating then rework the method so that it abides to the gate"*). It used to waive
    the ``/agent/register`` API-key gate TOO — see ``_require_api_key``, which is now
    PERMANENT and does NOT consult this function. That combination was the same
    silent-reopen as the old sitewide-toggle coupling, just behind a different flag:
    flip open mode for a test and the credential-minting bootstrap door reopened to
    the internet. **Open mode can no longer waive the register key. Ever.**

    What it still waives (deliberate, testing-only):
      * the operator gate on ``nodes`` / ``dispatch`` / ``tasks/<seq>``.
    What it NEVER waives:
      * the ``/agent/register`` API key — the one endpoint that MINTS credentials;
      * node-TOKEN auth on heartbeat/tasks/result — that's a node's identity, not a
        human gate; without it any caller could read/claim another node's queue.

    ⚠ These routes are reachable from the PUBLIC INTERNET on this deployment (the
    host front → :7001 → :7002 ``/api`` chain bypasses VM nginx allow-deny), so open
    mode still means anyone can LIST nodes and DISPATCH tasks to them. It remains a
    deliberate env flag defaulting CLOSED — unset the var + restart to restore the
    operator gate. It is a local-testing convenience, not a deployment posture."""
    return (os.getenv("HUGPY_AGENT_OPEN", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _require_operator() -> None:
    """Operator gate for the console-facing routes. Fails closed if the gate
    module is unavailable for any reason (never fail open) — unless the operator
    has explicitly opened the agents feature (``HUGPY_AGENT_OPEN``)."""
    if _agent_gates_open():
        return
    try:
        from ..operator_auth import operator_authenticated
    except Exception:
        abort(401, description="Operator authentication required for this route.")
    if not operator_authenticated():
        abort(401, description="Operator authentication required for this route.")


def _api_key_bearer() -> "str | None":
    """A console API key from ``Authorization: Bearer <key>`` (or ``?api_key=``
    for curl), the same extraction ``/v1`` and ``/ml`` use for their key gate.
    Distinct from ``_agent_token()`` (that is a NODE'S enroll token; this is a
    console API key)."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.args.get("api_key")


def _require_api_key() -> None:
    """PERMANENT API-key gate for the bootstrap ``/agent/register``.

    OPERATOR RULING 2026-07-16 (verbatim): *"this agent key should be a separate
    api category entirely"* — *"unlike the api key for v1 that gates or ungates the
    entire schema sitewide with a click, [agent calls] should be gated permanently."*

    THE FLAW THIS FIXES. Until now this gate read::

        if api_key_required() and not verify_api_key(...):   # ← coupled

    i.e. agent enrollment inherited the SITEWIDE key policy: flip ``/v1`` keyless
    for a demo and ``/agent/register`` silently reopened too. Those two are
    categorically different asks:

      * ``/v1`` open  = a deliberate POSTURE choice ("this deployment is keyless"),
        toggled from the console by design.
      * ``/agent/register`` open = never intended. It is the fleet BOOTSTRAP — the
        one endpoint that MINTS node credentials — and it is reachable from the
        PUBLIC INTERNET here (host front → :7001 devServer → :7002 forwards
        ``/api/agent/register`` from localhost, which no connection-IP / nginx
        allow-deny can gate). Verified open live on 2026-07-16: a bare public POST
        returned 201 and minted a node.

    So the gate no longer consults ``api_key_required()``: a valid console-minted
    key is required ALWAYS, independent of the sitewide toggle. ``verify_api_key``
    already validates on the key's own merits (hash + revocation) and never read
    that flag, so no key-system change was needed — only the decoupling.

    Fails closed: an unloadable key module refuses rather than admits.

    NOR does it consult ``_agent_gates_open()`` (operator 2026-07-16: *"if the
    hugpy_agent_open is bypassing gating then rework the method so that it abides to
    the gate"*). Open mode waiving this key was the SAME silent-reopen as the
    sitewide coupling, just behind a different flag — flip it for a test and the
    public bootstrap door swings open again. Open mode may still waive the OPERATOR
    gate on nodes/dispatch (its testing purpose); it can never waive this one.

    CONSEQUENCE (intended): every agent daemon must be handed a console API key to
    enroll — including this VM's own todo-keeper node. Bootstrap is the hardest
    door, not the softest. Keys are minted in the console under API access
    (``POST /keys``) and are individually revocable (``DELETE /keys/<id>``)."""
    try:
        from ..functions.imports.utils.api_keys import verify_api_key
    except Exception:
        # fail closed: if the key module can't load we cannot verify -> refuse
        abort(401, description="Agent registration key gate unavailable.")
    # required_scope="agent-register" (2026-07-23): the key must carry that
    # scope or "full". Legacy keys (no scopes field) read as ["full"] and keep
    # passing; a narrowly scoped install-link key (e.g. ["v1"]) cannot enroll
    # a node unless the operator granted it "agent-register" at mint time.
    if not verify_api_key(_api_key_bearer(), required_scope="agent-register"):
        abort(401, description=(
            "Agent registration requires a valid API key with the "
            "'agent-register' scope. Pass 'Authorization: Bearer <key>' "
            "(create keys in the console under API access)."))


# ── machine-to-machine: nodes enroll + heartbeat ───────────────────────────
@agent_bp.route("/agent/register", methods=["POST"])
def agent_register():
    """Bootstrap enrollment. {name, host, capabilities} -> node id + one-time
    enroll token. Gated by the console API-key policy (``_require_api_key`` — the
    same gate as the general ``/v1`` and media ``/ml`` endpoints); the node token
    it receives is what authenticates every subsequent call."""
    _require_api_key()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, description="Agent registration requires a 'name'.")
    caps = body.get("capabilities")
    if caps is None:
        caps = []
    if not isinstance(caps, list):
        abort(400, description="'capabilities' must be a list.")
    node = agent_node_store.register(
        name=name,
        host=(body.get("host") or "").strip(),
        capabilities=caps,
    )
    return jsonify(node), 201


@agent_bp.route("/agent/<node_id>/heartbeat", methods=["POST"])
def agent_heartbeat(node_id):
    """A node reports it is alive. {status, current_task, version}. Node-token
    authenticated; only the provided fields are written."""
    _require_node_auth(node_id)
    body = request.get_json(silent=True) or {}

    def _s(v):
        return str(v) if v is not None else None

    node = agent_node_store.heartbeat(
        node_id,
        status=_s(body.get("status")),
        current_task=_s(body.get("current_task")),
        version=_s(body.get("version")),
    )
    if node is None:
        # Raced with a delete between the auth check and the write.
        abort(410, description="Unknown agent node id; please re-register.")
    return jsonify(node)


# ── machine-to-machine: a node pulls its dispatched tasks ──────────────────
@agent_bp.route("/agent/<node_id>/tasks", methods=["GET"])
def agent_tasks(node_id):
    """The node's queue. ``?since=<seq>`` pulls only tasks newer than the
    cursor the node last saw (the pull is idempotent). Returns the tasks plus a
    ``cursor`` to pass as ``since`` next time. Node-token authenticated."""
    _require_node_auth(node_id)
    try:
        since = int(request.args.get("since", 0) or 0)
    except (TypeError, ValueError):
        since = 0
    tasks = agent_node_store.tasks_since(node_id, since=since)
    cursor = tasks[-1]["seq"] if tasks else since
    return jsonify({"node_id": node_id, "since": since,
                    "cursor": cursor, "tasks": tasks})


# ── machine-to-machine: a node reports a task's result (P3.1b) ─────────────
_RESULT_STATUS = {"done", "error"}


@agent_bp.route("/agent/<node_id>/tasks/<seq>/result", methods=["POST"])
def agent_task_result(node_id, seq):
    """A node reports the OUTCOME of a dispatched task. Node-token authenticated
    (the SAME gate as heartbeat/tasks — unknown node → 410, revoked → 403,
    missing/bad token → 401). Body: ``{status: "done"|"error", result: <string>}``.

    Transitions the task row queued → done/error and stores the (size-capped)
    result + finished_at. Fail-closed on the task itself: a seq that is unknown
    or belongs to ANOTHER node → 404. Idempotent-safe: re-posting an already
    finalized task does NOT overwrite it → 409 with the stored view (first report
    wins; a node re-posting after a crash treats BOTH 200 and 409 as 'recorded')."""
    _require_node_auth(node_id)
    body = request.get_json(silent=True) or {}
    status = (body.get("status") or "").strip().lower()
    if status not in _RESULT_STATUS:
        abort(400, description="Result 'status' must be 'done' or 'error'.")
    result = body.get("result")
    if result is not None and not isinstance(result, str):
        result = json.dumps(result)      # structured output round-trips as text
    outcome = agent_node_store.complete_task(
        node_id, seq, status=status, result=result)
    if outcome.get("ok"):
        return jsonify(outcome["task"]), 200
    if outcome.get("reason") == "conflict":
        return jsonify(dict(outcome["task"], already_finalized=True)), 409
    abort(404, description="Unknown task for this node.")


# ── public: serve the terminal dispatch client ──────────────────────────────
def _find_dispatch_client() -> "str | None":
    """Locate ``hugpy_agent/bin/hugpy-dispatch`` on disk.

    ``HUGPY_AGENT_CLIENT_SH`` overrides the path outright (ops convenience /
    testing) and is AUTHORITATIVE when set — a misconfigured override (points
    at nothing) surfaces as 404 rather than silently falling back to
    auto-discovery, so a bad env var is visible instead of masked. Only when
    the var is unset do we walk up from THIS file (not cwd — the service runs
    with ``chdir ~/station/dev/abstract_hugpy_dev``, one level short of the
    repo root that actually holds ``hugpy_agent/``) until a directory
    containing ``hugpy_agent/bin/hugpy-dispatch`` is found. Returns None if
    nothing resolves."""
    override = os.getenv("HUGPY_AGENT_CLIENT_SH")
    if override:
        return override if os.path.isfile(override) else None
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        candidate = os.path.join(d, "hugpy_agent", "bin", "hugpy-dispatch")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


@agent_bp.route("/agent/client.sh", methods=["GET"])
def agent_client_sh():
    """Public: serve the terminal dispatch client so any box can install it
    with one line::

        curl -fsSL https://dev.hugpy.ai/api/agent/client.sh | bash -s install

    Unauthenticated by design, same rationale as the worker/phone-brick
    bootstrap scripts (``workers_install_sh`` / ``phone_install_script``):
    the script is plain client code with no embedded secret — it reads the
    operator token from the caller's environment or config file at RUN time,
    never from this response. Piping to ``bash -s install`` means the
    script's ``install`` subcommand cannot copy itself from ``$0`` (stdin has
    no file), so it re-downloads its own source from
    ``$HUGPY_CENTRAL/agent/client.sh`` in that case — see the install
    subcommand's stdin-fallback branch.

    404s with a one-line explanation if the script cannot be located (e.g. a
    deployment layout where ``hugpy_agent/`` isn't a sibling of the repo this
    file ships from) rather than 500ing.
    """
    from flask import Response
    path = _find_dispatch_client()
    if path is None:
        abort(404, description=(
            "Dispatch client script not found on this deployment "
            "(hugpy_agent/bin/hugpy-dispatch missing)."))
    with open(path, encoding="utf-8") as fh:
        script = fh.read()
    return Response(script, mimetype="text/x-shellscript")


# ── secure one-time install links (2026-07-23, operator-approved) ──────────
# The console owner mints a labeled/scoped ONE-TIME download link for the
# hugpy-agent installer. The download templates a freshly minted scoped key
# into the installer's EMBEDDED_API_KEY slot — the operator never sees or
# handles the raw key (it is NEVER in the mint response; it exists only inside
# the download). Mint/list/revoke are OPERATOR-gated (the operator token /
# session — a mere api key can never mint a key: "no key-minting-by-key",
# same structural rule as the video-share links). The download GET itself is
# gated by the link_id (an unguessable secrets.token_urlsafe capability).
#
# Use counting: only the .py fetch consumes a use. The .sh/.ps1 wrappers are
# free (audited but not decremented) so the one-liner
#     curl -fsSL <base>/agent/install/<link_id>.sh | bash
# — which fetches the wrapper AND then the .py — costs exactly ONE use.

def _install_links_mod():
    from ..functions.imports.utils import install_links
    return install_links


def _find_installer_py() -> "str | None":
    """Locate ``hugpy_agent/install/install_hugpy_agent.py`` on disk.

    Same discovery idiom as ``_find_dispatch_client`` (the served client.sh):
    ``HUGPY_AGENT_INSTALLER_PY`` overrides outright and is AUTHORITATIVE when
    set (a bad override 404s visibly instead of silently falling back); else
    walk up from THIS file until a directory containing the installer is found.
    SINGLE SOURCE by design: the installer ships in the hugpy_agent repo (its
    tests import it from there) and central serves that same file — no
    build-time copy to drift."""
    override = os.getenv("HUGPY_AGENT_INSTALLER_PY")
    if override:
        return override if os.path.isfile(override) else None
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        candidate = os.path.join(
            d, "hugpy_agent", "install", "install_hugpy_agent.py")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


_EMBED_LINE = 'EMBEDDED_API_KEY = ""'


def _template_installer(source: str, raw_key: str) -> "str | None":
    """Replace the installer's EMBEDDED_API_KEY slot with the raw key.
    Returns None if the slot line is missing (installer drifted — refuse to
    serve an un-keyed download from a one-time link)."""
    if _EMBED_LINE not in source:
        return None
    # repr() the key so any quoting is safe; the slot is a plain assignment.
    return source.replace(_EMBED_LINE, f"EMBEDDED_API_KEY = {raw_key!r}", 1)


@agent_bp.route("/agent/install-links", methods=["POST"])
def install_link_create():
    """OPERATOR: mint a scoped key + its one-time install link.
    Body: {label (required), scopes (default ["v1"]), key_expires_at? (epoch s
    or ISO-8601), link_ttl_s (default 86400), max_uses (default 1)}.
    Returns {url, link_id, label, scopes, expires_at, max_uses, uses_left,
    key_id, status} — NEVER the raw key."""
    _require_operator_strict()
    body = request.get_json(silent=True) or {}
    label = (body.get("label") or "").strip()
    if not label:
        abort(400, description="An install link requires a 'label'.")
    scopes = body.get("scopes")
    if scopes is not None and not isinstance(scopes, list):
        abort(400, description="'scopes' must be a list.")
    key_expires_at = body.get("key_expires_at")
    if isinstance(key_expires_at, str) and key_expires_at.strip():
        # Accept ISO-8601 too (the spec says "optional ISO"); epoch also fine.
        from datetime import datetime
        try:
            key_expires_at = datetime.fromisoformat(
                key_expires_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            abort(400, description="'key_expires_at' must be epoch seconds or ISO-8601.")
    elif isinstance(key_expires_at, str):
        key_expires_at = None
    try:
        link = _install_links_mod().create_install_link(
            label=label,
            scopes=scopes,
            key_expires_at=key_expires_at,
            link_ttl_s=body.get("link_ttl_s"),
            max_uses=body.get("max_uses"),
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    link["url"] = f"{_install_public_base()}/agent/install/{link['link_id']}"
    return jsonify(link), 201


def _install_public_base() -> str:
    """Public base for building install URLs — same idiom as the video-share
    ``_public_base``: an explicit ``HUGPY_PUBLIC_BASE`` wins; else reconstruct
    from the forwarded proto/host. The public entry is the ``/api`` mount
    (host front → :7001 → :7002), so ``/api`` is appended unless the base
    already ends with it or the request itself arrived bare."""
    base = (os.getenv("HUGPY_PUBLIC_BASE") or "").strip().rstrip("/")
    if not base:
        proto = (request.headers.get("X-Forwarded-Proto") or request.scheme
                 or "https").split(",")[0].strip()
        host = (request.headers.get("X-Forwarded-Host") or request.host
                or "").split(",")[0].strip()
        base = f"{proto}://{host}".rstrip("/") if host else ""
    if base and not base.endswith("/api") and (
            request.path == "/api" or request.path.startswith("/api/")):
        base += "/api"
    return base


@agent_bp.route("/agent/install-links", methods=["GET"])
def install_link_list():
    """OPERATOR: every link with computed status (active/exhausted/expired/
    revoked) + use counts. Raw keys never appear (scrubbed store-side)."""
    _require_operator_strict()
    return jsonify({"links": _install_links_mod().list_install_links()})


@agent_bp.route("/agent/install-links/<link_id>", methods=["DELETE"])
def install_link_revoke(link_id):
    """OPERATOR: revoke the link AND the key it minted."""
    _require_operator_strict()
    if not _install_links_mod().revoke_install_link(link_id):
        abort(404, description="Unknown install link.")
    return jsonify({"ok": True})


def _require_operator_strict() -> None:
    """The operator gate for install-link management — WITHOUT the
    ``HUGPY_AGENT_OPEN`` testing waiver ``_require_operator`` honors. These
    routes MINT credentials (the same category as ``/agent/register``'s
    permanent key gate): open mode may waive the fleet-view gates, never a
    credential-minting one. Fails closed if the gate module is unavailable."""
    try:
        from ..operator_auth import operator_authenticated
    except Exception:
        abort(401, description="Operator authentication required for this route.")
    if not operator_authenticated():
        abort(401, description="Operator authentication required for this route.")


def _serve_install_py(link_id: str):
    """The one-time download itself: template the raw key in, consume a use."""
    from flask import Response
    mod = _install_links_mod()
    path = _find_installer_py()
    if path is None:
        abort(404, description=(
            "Installer source not found on this deployment "
            "(hugpy_agent/install/install_hugpy_agent.py missing)."))
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    raw_key = mod.consume_download(link_id, remote_addr=remote)
    if raw_key is None:
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    body = _template_installer(source, raw_key)
    if body is None:
        # The slot line drifted out of the installer: refuse rather than serve
        # an un-keyed installer from a link that just consumed a use.
        logger.error("install link %s…: EMBEDDED_API_KEY slot missing in %s",
                     link_id[:8], path)
        abort(500, description="Installer template slot missing on this deployment.")
    logger.info("install link %s… served install_hugpy_agent.py to %s",
                link_id[:8], remote or "?")
    resp = Response(body, mimetype="text/x-python")
    resp.headers["Content-Disposition"] = (
        'attachment; filename="install_hugpy_agent.py"')
    resp.headers["Cache-Control"] = "no-store"
    return resp


_SH_WRAPPER = """#!/bin/sh
# hugpy-agent one-time installer bootstrap (POSIX).
# Fetches the python installer from the SAME one-time link and runs it.
# This wrapper fetch does NOT consume the link — only the .py fetch does.
set -e
PY_URL="{py_url}"
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "hugpy-agent installer: python3 is required but was not found on PATH." >&2
  echo "Install python3 (e.g. 'sudo apt install python3' / 'brew install python3') and re-run." >&2
  exit 1
fi
TMP="$(mktemp /tmp/install_hugpy_agent.XXXXXX.py)"
trap 'rm -f "$TMP"' EXIT
if ! curl -fsSL "$PY_URL" -o "$TMP"; then
  echo "hugpy-agent installer: download failed — the link may be used up, expired, or revoked." >&2
  exit 1
fi
exec "$PY" "$TMP" "$@"
"""

_PS1_WRAPPER = """# hugpy-agent one-time installer bootstrap (Windows PowerShell).
# Fetches the python installer from the SAME one-time link and runs it.
# This wrapper fetch does NOT consume the link — only the .py fetch does.
$ErrorActionPreference = 'Stop'
$PyUrl = '{py_url}'
$Py = $null
foreach ($c in @('py', 'python3', 'python')) {{
  if (Get-Command $c -ErrorAction SilentlyContinue) {{ $Py = $c; break }}
}}
if (-not $Py) {{
  Write-Error 'hugpy-agent installer: python is required but was not found on PATH. Install it from https://python.org and re-run.'
  exit 1
}}
$Tmp = Join-Path $env:TEMP ("install_hugpy_agent_" + [System.Guid]::NewGuid().ToString('N') + '.py')
try {{
  Invoke-WebRequest -UseBasicParsing -Uri $PyUrl -OutFile $Tmp
  & $Py $Tmp @args
}} finally {{
  Remove-Item -Force -ErrorAction SilentlyContinue $Tmp
}}
"""


def _serve_install_wrapper(link_id: str, kind: str):
    """The .sh / .ps1 convenience wrappers. Validity-checked (a dead link 410s
    here too, honestly) but NEVER decrements — only the .py fetch counts."""
    from flask import Response
    mod = _install_links_mod()
    if not mod.peek_active(link_id):
        abort(410, description=(
            "This install link is no longer valid — it was used up, expired, "
            "or revoked. Ask the console owner to mint a fresh one."))
    remote = (request.headers.get("X-Forwarded-For") or
              request.remote_addr or "").split(",")[0].strip()
    mod.note_wrapper_fetch(link_id, remote_addr=remote, kind=kind)
    # The wrapper fetches the .py from the SAME path the caller just used,
    # minus the extension — so whatever base/mount reached us keeps working.
    py_url = f"{_install_public_base()}/agent/install/{link_id}"
    if kind == "sh":
        body = _SH_WRAPPER.format(py_url=py_url)
        mime = "text/x-shellscript"
        fname = "install_hugpy_agent.sh"
    else:
        body = _PS1_WRAPPER.format(py_url=py_url)
        mime = "text/plain"
        fname = "install_hugpy_agent.ps1"
    resp = Response(body, mimetype=mime)
    resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    resp.headers["Cache-Control"] = "no-store"
    return resp


@agent_bp.route("/agent/install/<link_id>", methods=["GET"])
def install_download(link_id):
    """The one-time download. ``<link_id>`` bare serves the templated .py
    (consumes a use); ``<link_id>.sh`` / ``<link_id>.ps1`` serve the platform
    wrappers (free — they fetch the .py themselves, which is the one use)."""
    if link_id.endswith(".sh"):
        return _serve_install_wrapper(link_id[:-3], "sh")
    if link_id.endswith(".ps1"):
        return _serve_install_wrapper(link_id[:-4], "ps1")
    return _serve_install_py(link_id)


# ── console UI: operator lists nodes + dispatches tasks ────────────────────
@agent_bp.route("/agent/nodes", methods=["GET"])
def agent_nodes():
    """Every enrolled node with its live status — operator-only."""
    _require_operator()
    return jsonify(agent_node_store.all())


@agent_bp.route("/agent/<node_id>/tasks/<seq>", methods=["GET"])
def agent_task_detail(node_id, seq):
    """One task's full row incl. its ``result`` — operator-only (what the P3.3
    console panel reads to render a run's final report). Gated exactly like
    ``GET /agent/nodes``. Distinct from the node-token pull ``GET
    /agent/<id>/tasks`` (no ``<seq>``, returns the whole queue)."""
    _require_operator()
    task = agent_node_store.get_task(node_id, seq)
    if task is None:
        abort(404, description="Unknown task for this node.")
    return jsonify(task)


@agent_bp.route("/agent/<node_id>/dispatch", methods=["POST"])
def agent_dispatch(node_id):
    """Queue a task for a node — operator-only. Body: {task}. The node picks it
    up on its next GET /agent/<id>/tasks."""
    _require_operator()
    body = request.get_json(silent=True) or {}
    if "task" not in body or body.get("task") is None:
        abort(400, description="Dispatch requires a 'task'.")
    queued = agent_node_store.dispatch(node_id, body["task"])
    if queued is None:
        abort(404, description="Unknown or revoked agent node id.")
    return jsonify(queued), 201
