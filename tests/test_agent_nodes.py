"""P3.1 — offline tests for the /agent/* node registry + dispatch.

Three layers, NO network:
  * pure store (comms.agent_nodes.AgentNodeStore): enroll, token hashing +
    fail-closed auth, heartbeat, the dispatch queue's monotonic pull cursor,
    revocation, and the cross-process (shared-db) property that makes it
    gunicorn 3-worker safe.
  * the Flask blueprint via test_client: the M2M contract (register/heartbeat/
    tasks) and its node-token gate (401/403/410), plus the operator gate on
    /agent/nodes and /agent/<id>/dispatch — and the full register -> heartbeat
    -> dispatch -> pull acceptance flow.
  * the central operator allowlist (operator_auth._SENSITIVE) covers exactly
    the two operator routes and none of the M2M ones.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_agent_nodes.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Isolate any incidental writes; point the comms db at a scratch file so the
# imported module singletons never touch a real one.
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-agent-test-"))
os.environ["HUGPY_COMMS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="hugpy-agent-comms-"), "comms.db")

import pytest
from flask import Flask

from abstract_hugpy_dev.comms.agent_nodes import AgentNodeStore
from abstract_hugpy_dev.flask_app.app.routes import agent_routes
from abstract_hugpy_dev.flask_app.app import operator_auth


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def store(tmp_path):
    return AgentNodeStore(path=str(tmp_path / "agent.db"))


@pytest.fixture
def client(store, monkeypatch):
    """A bare app with ONLY the agent blueprint — so the inline gates are what's
    under test (no central before_request gate installed), and the store is a
    fresh scratch db per test.

    The site key policy (``api_key_required``) is forced OFF here so /agent/register
    behaves as the open bootstrap these M2M-contract tests were written against —
    otherwise the real dev box's key config (``require_key`` on) would leak into the
    test process and 401 every enroll. The register KEY GATE itself is exercised
    explicitly by ``_key_policy`` below."""
    monkeypatch.setattr(agent_routes, "agent_node_store", store)
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import api_keys as _ak
    monkeypatch.setattr(_ak, "api_key_required", lambda: False)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app.test_client()


@pytest.fixture
def keyed_client(store, monkeypatch):
    """Like ``client`` but with the register API-key gate turned ON (site key
    policy required) and key verification stubbed, so the gate on /agent/register
    can be exercised without a real key store. Yields ``(test_client, good_key)``."""
    monkeypatch.setattr(agent_routes, "agent_node_store", store)
    from abstract_hugpy_dev.flask_app.app.functions.imports.utils import api_keys as _ak
    good = "hp_test_valid_key"
    monkeypatch.setattr(_ak, "api_key_required", lambda: True)
    monkeypatch.setattr(_ak, "verify_api_key", lambda tok: tok == good)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app.test_client(), good


def _enroll(client, name="luigi", host="10.0.0.9", caps=("chat",)):
    r = client.post("/agent/register",
                    json={"name": name, "host": host,
                          "capabilities": list(caps)})
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── pure store ──────────────────────────────────────────────────────────────
def test_register_mints_prefixed_id_and_token(store):
    node = store.register(name="luigi", host="h", capabilities=["chat", "tools"])
    assert node["id"].startswith("agn_")
    assert node["token"].startswith("agt_")
    assert node["name"] == "luigi"
    assert node["capabilities"] == ["chat", "tools"]
    assert node["status"] == "enrolled"


def test_register_public_view_never_leaks_secret(store):
    node = store.register(name="n")
    # get()/all() are the API-facing views: no plaintext, no hash, ever.
    got = store.get(node["id"])
    assert "token" not in got and "token_hash" not in got
    assert all("token" not in n and "token_hash" not in n for n in store.all())


def test_authenticate_is_fail_closed(store):
    node = store.register(name="n")
    nid, tok = node["id"], node["token"]
    assert store.authenticate(nid, tok) is True
    assert store.authenticate(nid, "agt_wrong") is False
    assert store.authenticate(nid, "no-prefix") is False
    assert store.authenticate(nid, None) is False
    assert store.authenticate(nid, "") is False
    assert store.authenticate("agn_unknown", tok) is False


def test_token_is_hashed_at_rest(store):
    node = store.register(name="n")
    import sqlite3
    with sqlite3.connect(store.path) as conn:
        row = conn.execute("SELECT token_hash FROM agent_nodes WHERE id=?",
                           (node["id"],)).fetchone()
    stored = row[0]
    assert stored != node["token"]           # never the plaintext
    assert len(stored) == 64                 # sha256 hex
    import hashlib
    assert stored == hashlib.sha256(node["token"].encode()).hexdigest()


def test_heartbeat_updates_reported_fields(store):
    node = store.register(name="n")
    assert store.get(node["id"])["last_seen"] is not None  # set at enroll
    hb = store.heartbeat(node["id"], status="busy",
                         current_task="atsk_1", version="0.1.0")
    assert hb["status"] == "busy"
    assert hb["current_task"] == "atsk_1"
    assert hb["version"] == "0.1.0"
    assert hb["last_seen"] is not None


def test_heartbeat_only_writes_provided_fields(store):
    node = store.register(name="n")
    store.heartbeat(node["id"], status="idle", version="0.1.0")
    store.heartbeat(node["id"], current_task="atsk_9")  # status/version untouched
    got = store.get(node["id"])
    assert got["status"] == "idle"
    assert got["version"] == "0.1.0"
    assert got["current_task"] == "atsk_9"


def test_heartbeat_unknown_node_returns_none(store):
    assert store.heartbeat("agn_nope", status="busy") is None


def test_dispatch_and_pull_cursor_is_monotonic(store):
    node = store.register(name="n")
    nid = node["id"]
    t1 = store.dispatch(nid, {"kind": "chat", "prompt": "a"})
    t2 = store.dispatch(nid, {"kind": "chat", "prompt": "b"})
    assert t1["id"].startswith("atsk_") and t2["seq"] > t1["seq"]
    assert t1["task"] == {"kind": "chat", "prompt": "a"}
    all_ = store.tasks_since(nid, 0)
    assert [t["seq"] for t in all_] == [t1["seq"], t2["seq"]]
    # advancing the cursor past t1 yields only t2 (idempotent pull)
    assert [t["seq"] for t in store.tasks_since(nid, t1["seq"])] == [t2["seq"]]
    # cursor at the tail yields nothing
    assert store.tasks_since(nid, t2["seq"]) == []


def test_dispatch_is_scoped_per_node(store):
    a = store.register(name="a")["id"]
    b = store.register(name="b")["id"]
    ta = store.dispatch(a, {"x": 1})
    store.dispatch(b, {"y": 2})
    # a only ever sees its own task
    got = store.tasks_since(a, 0)
    assert [t["seq"] for t in got] == [ta["seq"]]
    assert got[0]["node_id"] == a


def test_dispatch_unknown_node_returns_none(store):
    assert store.dispatch("agn_nope", {"x": 1}) is None


def test_revoke_blocks_auth_and_dispatch(store):
    node = store.register(name="n")
    nid, tok = node["id"], node["token"]
    assert store.revoke(nid) is True
    assert store.authenticate(nid, tok) is False       # revoked -> no auth
    assert store.dispatch(nid, {"x": 1}) is None        # revoked -> no queue
    assert store.get(nid)["revoked"] is True


def test_cross_process_shared_db(tmp_path):
    """Gunicorn 3-worker safety: a second store on the SAME db file sees a
    node registered by the first, and can pull tasks dispatched through it."""
    path = str(tmp_path / "shared.db")
    s1 = AgentNodeStore(path=path)
    s2 = AgentNodeStore(path=path)
    node = s1.register(name="n", capabilities=["chat"])
    nid, tok = node["id"], node["token"]
    assert s2.get(nid) is not None
    assert s2.authenticate(nid, tok) is True
    s1.dispatch(nid, {"kind": "chat"})
    assert len(s2.tasks_since(nid, 0)) == 1


# ── Flask blueprint: M2M contract ───────────────────────────────────────────
def test_route_register_returns_201_with_token(client):
    node = _enroll(client)
    assert node["id"].startswith("agn_")
    assert node["token"].startswith("agt_")


def test_route_register_requires_name(client):
    assert client.post("/agent/register", json={}).status_code == 400
    assert client.post("/agent/register",
                       json={"name": "  "}).status_code == 400


def test_route_register_rejects_non_list_capabilities(client):
    r = client.post("/agent/register",
                    json={"name": "n", "capabilities": "chat"})
    assert r.status_code == 400


# ── register API-key gate (keeper 2026-07-14; operator-directed "just like the
#    media and general endpoint calling") ─────────────────────────────────────
def test_register_gate_open_when_key_policy_off(client):
    """Site key policy OFF -> register stays the open bootstrap (matches /v1)."""
    assert client.post("/agent/register", json={"name": "n"}).status_code == 201


def test_register_gate_refuses_without_key_when_policy_on(keyed_client):
    """Site key policy ON + no key -> 401. This is the demonstrated internet-open
    hole closed: a keyless register no longer mints a token."""
    kc, _good = keyed_client
    assert kc.post("/agent/register", json={"name": "n"}).status_code == 401


def test_register_gate_rejects_bad_key_when_policy_on(keyed_client):
    kc, _good = keyed_client
    r = kc.post("/agent/register", json={"name": "n"},
                headers={"Authorization": "Bearer hp_wrong"})
    assert r.status_code == 401


def test_register_gate_allows_valid_key_when_policy_on(keyed_client):
    """A legit node presenting a valid console key still enrolls (bootstrap works)."""
    kc, good = keyed_client
    r = kc.post("/agent/register", json={"name": "n"},
                headers={"Authorization": f"Bearer {good}"})
    assert r.status_code == 201, r.get_json()
    assert r.get_json()["token"].startswith("agt_")


def test_register_gate_accepts_key_via_query_param(keyed_client):
    """?api_key= is accepted too (same curl affordance /v1 and /ml give)."""
    kc, good = keyed_client
    r = kc.post(f"/agent/register?api_key={good}", json={"name": "n"})
    assert r.status_code == 201, r.get_json()


def test_route_heartbeat_requires_valid_token(client):
    node = _enroll(client)
    nid = node["id"]
    # no token
    assert client.post(f"/agent/{nid}/heartbeat", json={}).status_code == 401
    # wrong token
    assert client.post(f"/agent/{nid}/heartbeat", json={},
                       headers=_auth("agt_wrong")).status_code == 401


def test_route_heartbeat_ok_with_token(client):
    node = _enroll(client)
    r = client.post(f"/agent/{node['id']}/heartbeat",
                    json={"status": "busy", "current_task": "atsk_1",
                          "version": "0.1.0"},
                    headers=_auth(node["token"]))
    assert r.status_code == 200
    assert r.get_json()["status"] == "busy"


def test_route_heartbeat_accepts_x_agent_token_header(client):
    node = _enroll(client)
    r = client.post(f"/agent/{node['id']}/heartbeat", json={"status": "idle"},
                    headers={"X-Agent-Token": node["token"]})
    assert r.status_code == 200


def test_route_heartbeat_unknown_node_410(client):
    r = client.post("/agent/agn_nope/heartbeat", json={},
                    headers=_auth("agt_whatever"))
    assert r.status_code == 410


def test_route_heartbeat_revoked_node_403(client, store):
    node = _enroll(client)
    store.revoke(node["id"])
    r = client.post(f"/agent/{node['id']}/heartbeat", json={},
                    headers=_auth(node["token"]))
    assert r.status_code == 403


def test_route_tasks_requires_valid_token(client):
    node = _enroll(client)
    assert client.get(f"/agent/{node['id']}/tasks").status_code == 401
    assert client.get(f"/agent/{node['id']}/tasks",
                      headers=_auth("agt_wrong")).status_code == 401


def test_route_tasks_pull_with_cursor(client, store):
    node = _enroll(client)
    nid = node["id"]
    store.dispatch(nid, {"n": 1})
    store.dispatch(nid, {"n": 2})
    r = client.get(f"/agent/{nid}/tasks", headers=_auth(node["token"]))
    body = r.get_json()
    assert r.status_code == 200
    assert len(body["tasks"]) == 2
    cursor = body["cursor"]
    # re-pull from the cursor: nothing new
    r2 = client.get(f"/agent/{nid}/tasks?since={cursor}",
                    headers=_auth(node["token"]))
    assert r2.get_json()["tasks"] == []


# ── Flask blueprint: operator gate ──────────────────────────────────────────
def _operator_env(monkeypatch, token="op-secret"):
    """Force the operator gate to actually bite, no network: 'open' mode with a
    token set means the token is REQUIRED (external-session path never taken)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", token)
    return {"X-Operator-Token": token}


def test_route_nodes_operator_gated(client, monkeypatch):
    hdr = _operator_env(monkeypatch)
    assert client.get("/agent/nodes").status_code == 401           # no operator
    r = client.get("/agent/nodes", headers=hdr)
    assert r.status_code == 200 and isinstance(r.get_json(), list)


def test_route_nodes_open_when_no_token(client, monkeypatch):
    # Self-hosted default: open mode, no operator token -> permissive.
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    assert client.get("/agent/nodes").status_code == 200


def test_route_dispatch_operator_gated(client, monkeypatch):
    node = _enroll(client)
    _operator_env(monkeypatch)
    # gate runs BEFORE body validation -> 401 even with a valid task
    r = client.post(f"/agent/{node['id']}/dispatch", json={"task": {"k": 1}})
    assert r.status_code == 401


def test_route_dispatch_ok_and_queues(client, monkeypatch):
    node = _enroll(client)
    hdr = _operator_env(monkeypatch)
    r = client.post(f"/agent/{node['id']}/dispatch",
                    json={"task": {"kind": "chat", "prompt": "hi"}},
                    headers=hdr)
    assert r.status_code == 201
    queued = r.get_json()
    assert queued["task"] == {"kind": "chat", "prompt": "hi"}
    # and the node can pull it
    pull = client.get(f"/agent/{node['id']}/tasks",
                      headers=_auth(node["token"]))
    assert len(pull.get_json()["tasks"]) == 1


def test_route_dispatch_requires_task(client, monkeypatch):
    node = _enroll(client)
    hdr = _operator_env(monkeypatch)
    assert client.post(f"/agent/{node['id']}/dispatch", json={},
                       headers=hdr).status_code == 400


def test_route_dispatch_unknown_node_404(client, monkeypatch):
    hdr = _operator_env(monkeypatch)
    r = client.post("/agent/agn_nope/dispatch", json={"task": {"k": 1}},
                    headers=hdr)
    assert r.status_code == 404


def test_end_to_end_register_heartbeat_dispatch_pull(client, monkeypatch):
    """The P3.1 acceptance flow, offline."""
    node = _enroll(client, name="acceptance")
    nid, tok = node["id"], node["token"]
    # heartbeat
    assert client.post(f"/agent/{nid}/heartbeat", json={"status": "idle"},
                       headers=_auth(tok)).status_code == 200
    # operator sees it live
    hdr = _operator_env(monkeypatch)
    roster = client.get("/agent/nodes", headers=hdr).get_json()
    assert any(n["id"] == nid and n["status"] == "idle" for n in roster)
    # operator dispatches; node pulls exactly that task
    client.post(f"/agent/{nid}/dispatch", json={"task": {"do": "thing"}},
                headers=hdr)
    tasks = client.get(f"/agent/{nid}/tasks", headers=_auth(tok)).get_json()
    assert [t["task"] for t in tasks["tasks"]] == [{"do": "thing"}]


# ── central operator allowlist (operator_auth._SENSITIVE) ───────────────────
def _is_sensitive(method, path):
    app = Flask(__name__)
    with app.test_request_context(path=path, method=method):
        return operator_auth._path_is_sensitive()


def test_sensitive_covers_operator_agent_routes():
    assert _is_sensitive("GET", "/agent/nodes")
    assert _is_sensitive("POST", "/agent/agn_abc/dispatch")
    # and via the /api-prefixed form (gate strips /api first)
    assert _is_sensitive("GET", "/api/agent/nodes")
    assert _is_sensitive("POST", "/api/agent/agn_abc/dispatch")


def test_sensitive_excludes_m2m_agent_routes():
    assert not _is_sensitive("POST", "/agent/register")
    assert not _is_sensitive("POST", "/agent/agn_abc/heartbeat")
    assert not _is_sensitive("GET", "/agent/agn_abc/tasks")
