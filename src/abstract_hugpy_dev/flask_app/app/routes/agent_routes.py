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
    entirely for now?"): ``HUGPY_AGENT_OPEN`` truthy disables BOTH the register
    API-key gate and the operator gate on nodes/dispatch. Node-TOKEN auth on
    heartbeat/tasks stays — that's a node's identity, not a human gate; without it
    any caller could read/claim another node's queue.

    ⚠ These routes are reachable from the PUBLIC INTERNET on this deployment (the
    host front → :7001 → :7002 ``/api`` chain bypasses VM nginx allow-deny), so
    open mode means anyone can enroll nodes and dispatch tasks. Deliberately an
    env flag, defaulting CLOSED: unset the var + restart to restore the gates."""
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
    """API-key gate for the OPEN bootstrap ``/agent/register`` — the SAME policy
    the general (``/v1``) and media (``/ml``) endpoints use (operator-directed
    2026-07-14: "add an api gate to it just like the media and general endpoint
    calling").

    ``/agent/register`` is reachable from the public internet on this deployment
    (host front → :7001 devServer → :7002 forwards ``/api/agent/register`` from
    localhost, which no connection-IP / nginx allow-deny can gate), and it minted
    node tokens to anyone — demonstrated live by the operator. Requiring a valid
    console API key closes that with the existing credential system: when the
    site key policy is on (``api_key_required``), an enrolling node must present a
    console-minted key, exactly like a ``/v1`` caller. Fails closed (401). When
    the policy is off (an open/LAN-only deployment) register stays open, matching
    ``/v1``'s own behaviour — so this never breaks a deployment that runs keyless.

    NB register is the fleet BOOTSTRAP endpoint (a node has no credential yet), so
    with the policy on, agent daemons must be handed a console API key to enroll."""
    if _agent_gates_open():
        return
    try:
        from ..functions.imports.utils.api_keys import api_key_required, verify_api_key
    except Exception:
        # fail closed: if the key module can't load we cannot verify -> refuse
        abort(401, description="Agent registration key gate unavailable.")
    if api_key_required() and not verify_api_key(_api_key_bearer()):
        abort(401, description=(
            "Agent registration requires a valid API key. Pass "
            "'Authorization: Bearer <key>' (create keys in the console under API access)."))


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
