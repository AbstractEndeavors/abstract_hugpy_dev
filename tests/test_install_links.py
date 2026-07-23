"""Secure one-time INSTALL LINKS (2026-07-23) — offline tests, NO network.

Four layers:
  * api_keys scope extension: the verification matrix (a scoped key passes its
    scope and fails others, "full" passes everything, LEGACY rows — no new
    fields — behave exactly as before), expiry/disabled, additive list view.
  * install_links store: mint (raw key never in the public view), download
    consumption + scrubbing, wrapper fetches free, revoke kills link AND key.
  * the Flask routes via test_client: operator gate on mint/list/revoke
    (strict — HUGPY_AGENT_OPEN does NOT waive it), the link lifecycle
    (mint → .sh → .py → exhausted 410), the templated download containing the
    key and py_compile-ing, ttl expiry, raw key never in any operator-facing
    response.
  * the central operator allowlist (operator_auth._SENSITIVE) covers exactly
    the management routes and never the download GET.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_install_links.py -q
"""
import json
import os
import py_compile
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-install-links-test-"))
os.environ.setdefault("HUGPY_COMMS_DB", os.path.join(
    tempfile.mkdtemp(prefix="hugpy-install-links-comms-"), "comms.db"))
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

import pytest
from flask import Flask

from abstract_hugpy_dev.flask_app.app.functions.imports.utils import api_keys as ak
from abstract_hugpy_dev.flask_app.app.functions.imports.utils import install_links as il
from abstract_hugpy_dev.flask_app.app.routes import agent_routes
from abstract_hugpy_dev.flask_app.app import operator_auth


# ── fixtures: throwaway store files per test ────────────────────────────────
@pytest.fixture(autouse=True)
def scratch_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ak, "_store_path",
                        lambda: str(tmp_path / "api_keys.json"))
    monkeypatch.setattr(il, "_store_path",
                        lambda: str(tmp_path / "install_links.json"))
    yield


@pytest.fixture
def installer_py(tmp_path, monkeypatch):
    """A minimal stand-in installer WITH the EMBEDDED_API_KEY slot, plus the
    REAL repo installer for the compile test."""
    src = tmp_path / "install_hugpy_agent.py"
    src.write_text('#!/usr/bin/env python3\n'
                   'EMBEDDED_API_KEY = ""\n'
                   'if __name__ == "__main__":\n'
                   '    print(bool(EMBEDDED_API_KEY))\n')
    monkeypatch.setenv("HUGPY_AGENT_INSTALLER_PY", str(src))
    return str(src)


OP_TOKEN = "op-secret-token"


@pytest.fixture
def client(monkeypatch, installer_py):
    """Bare app with only the agent blueprint. Operator auth via
    HUGPY_OPERATOR_TOKEN in open mode (the operator_auth contract: open mode +
    token set => the token is required)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", OP_TOKEN)
    monkeypatch.delenv("HUGPY_AGENT_OPEN", raising=False)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app.test_client()


def _op():
    return {"X-Operator-Token": OP_TOKEN}


def _mint(client, **over):
    body = {"label": "test box"}
    body.update(over)
    r = client.post("/agent/install-links", json=body, headers=_op())
    assert r.status_code == 201, r.get_json()
    return r.get_json()


# ═══════════════════════════════════════════════════════════════════════════
# 1) api_keys scope matrix
# ═══════════════════════════════════════════════════════════════════════════
def test_scope_matrix_scoped_key():
    k = ak.create_api_key(name="n", scopes=["v1"])
    tok = k["key"]
    assert ak.verify_api_key(tok) is True                       # unscoped call: as today
    assert ak.verify_api_key(tok, required_scope="v1") is True  # its scope
    assert ak.verify_api_key(tok, required_scope="ml") is False # others refused
    assert ak.verify_api_key(tok, required_scope="agent-register") is False


def test_scope_matrix_full_passes_all():
    tok = ak.create_api_key(name="n", scopes=["full"])["key"]
    for scope in (None, "v1", "ml", "agent-register", "full"):
        assert ak.verify_api_key(tok, required_scope=scope) is True


def test_scope_matrix_multi_scope():
    tok = ak.create_api_key(name="n", scopes=["v1", "ml"])["key"]
    assert ak.verify_api_key(tok, required_scope="v1") is True
    assert ak.verify_api_key(tok, required_scope="ml") is True
    assert ak.verify_api_key(tok, required_scope="agent-register") is False


def test_legacy_rows_read_as_full_scope():
    """A pre-scope row (no label/scopes/created_by/expires_at/disabled) must
    verify against ANY required_scope and list with the defaults — the lazy
    additive migration."""
    tok = ak.create_api_key(name="old")["key"]
    key_id = ak.key_id_for_token(tok)
    # Strip the new fields to fabricate a genuine legacy row on disk.
    data = ak._load()
    rec = data["keys"][key_id]
    for f in ("label", "scopes", "created_by", "expires_at", "disabled"):
        rec.pop(f, None)
    ak._save(data)
    assert ak.verify_api_key(tok) is True
    assert ak.verify_api_key(tok, required_scope="v1") is True
    assert ak.verify_api_key(tok, required_scope="agent-register") is True
    listed = [k for k in ak.list_api_keys() if k["id"] == key_id][0]
    assert listed["scopes"] == ["full"]
    assert listed["label"] == ""
    assert listed["created_by"] == "operator"
    assert listed["expires_at"] is None and listed["disabled"] is False


def test_unknown_scope_refused_at_mint():
    with pytest.raises(ValueError):
        ak.create_api_key(name="n", scopes=["v2-typo"])


def test_key_expiry_and_disabled():
    tok = ak.create_api_key(name="n", scopes=["v1"],
                            expires_at=time.time() + 3600)["key"]
    assert ak.verify_api_key(tok) is True
    key_id = ak.key_id_for_token(tok)
    data = ak._load()
    data["keys"][key_id]["expires_at"] = time.time() - 5
    ak._save(data)
    assert ak.verify_api_key(tok) is False          # expired
    data = ak._load()
    data["keys"][key_id]["expires_at"] = None
    data["keys"][key_id]["disabled"] = True
    ak._save(data)
    assert ak.verify_api_key(tok) is False          # disabled


def test_mint_response_carries_new_fields():
    k = ak.create_api_key(name="n", label="the label", scopes=["ml"],
                          created_by="install-link")
    assert k["label"] == "the label"
    assert k["scopes"] == ["ml"]
    assert k["created_by"] == "install-link"
    assert "hash" not in k


# ═══════════════════════════════════════════════════════════════════════════
# 2) install_links store
# ═══════════════════════════════════════════════════════════════════════════
def test_store_mint_never_returns_raw_key():
    link = il.create_install_link(label="box A")
    assert "raw_key" not in link
    assert link["status"] == "active"
    assert link["uses_left"] == 1 and link["max_uses"] == 1
    assert link["scopes"] == ["v1"]                 # spec default
    # the key it minted exists, labeled, created_by install-link:
    keys = ak.list_api_keys()
    assert any(k["id"] == link["key_id"]
               and k["created_by"] == "install-link"
               and k["label"] == "box A" for k in keys)
    # listing never leaks the raw key either:
    assert all("raw_key" not in row for row in il.list_install_links())


def test_store_consume_returns_key_then_exhausts_and_scrubs():
    link = il.create_install_link(label="one shot")
    raw = il.consume_download(link["link_id"], remote_addr="1.2.3.4")
    assert raw and raw.startswith("hp_")
    assert ak.verify_api_key(raw, required_scope="v1") is True
    # exhausted now:
    assert il.consume_download(link["link_id"]) is None
    row = il.get_link(link["link_id"])
    assert row["status"] == "exhausted" and row["uses_left"] == 0
    # the raw key is scrubbed from the store file itself:
    with open(il._store_path()) as fh:
        assert raw not in fh.read()
    # audit row recorded:
    assert row["downloads"][0]["remote_addr"] == "1.2.3.4"
    assert row["downloads"][0]["kind"] == "py"


def test_store_multi_use_counts_down():
    link = il.create_install_link(label="triple", max_uses=3)
    for left in (2, 1, 0):
        assert il.consume_download(link["link_id"]) is not None
        assert il.get_link(link["link_id"])["uses_left"] == left
    assert il.consume_download(link["link_id"]) is None


def test_store_wrapper_fetch_does_not_decrement():
    link = il.create_install_link(label="wrapped")
    assert il.peek_active(link["link_id"]) is True
    il.note_wrapper_fetch(link["link_id"], remote_addr="9.9.9.9", kind="sh")
    row = il.get_link(link["link_id"])
    assert row["uses_left"] == 1                    # untouched
    assert any(d["kind"] == "sh" for d in row["downloads"])


def test_store_ttl_expiry_refuses_and_scrubs():
    link = il.create_install_link(label="stale", link_ttl_s=1)
    data = il._load()
    data["links"][link["link_id"]]["expires_at"] = time.time() - 5
    il._save(data)
    assert il.peek_active(link["link_id"]) is False
    assert il.consume_download(link["link_id"]) is None
    assert il.get_link(link["link_id"])["status"] == "expired"
    with open(il._store_path()) as fh:
        stored = json.load(fh)
    assert stored["links"][link["link_id"]]["raw_key"] == ""


def test_store_revoke_kills_link_and_key():
    link = il.create_install_link(label="doomed")
    key_id = link["key_id"]
    assert il.revoke_install_link(link["link_id"]) is True
    assert il.get_link(link["link_id"])["status"] == "revoked"
    assert il.consume_download(link["link_id"]) is None
    # the KEY is revoked too — it verifies for nothing:
    assert all(k["id"] != key_id for k in ak.list_api_keys())
    # and the raw key is gone from disk:
    with open(il._store_path()) as fh:
        stored = json.load(fh)
    assert stored["links"][link["link_id"]]["raw_key"] == ""


def test_store_key_revoked_out_of_band_refuses_download():
    """Operator revokes the KEY directly (from the key list) — the link must
    stop serving even while nominally active."""
    link = il.create_install_link(label="key pulled")
    ak.revoke_api_key(link["key_id"])
    assert il.consume_download(link["link_id"]) is None


def test_store_blank_label_refused():
    with pytest.raises(ValueError):
        il.create_install_link(label="   ")


# ═══════════════════════════════════════════════════════════════════════════
# 3) the Flask routes
# ═══════════════════════════════════════════════════════════════════════════
def test_route_mint_requires_operator(client):
    r = client.post("/agent/install-links", json={"label": "x"})
    assert r.status_code == 401
    r = client.post("/agent/install-links", json={"label": "x"},
                    headers={"X-Operator-Token": "wrong"})
    assert r.status_code == 401


def test_route_list_and_revoke_require_operator(client):
    assert client.get("/agent/install-links").status_code == 401
    assert client.delete("/agent/install-links/abc").status_code == 401


def test_route_agent_open_does_not_waive_install_link_gate(client, monkeypatch):
    """HUGPY_AGENT_OPEN waives the fleet-view operator gates — it must NEVER
    waive a credential-minting surface (the 2026-07-16 register ruling)."""
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "true")
    assert client.post("/agent/install-links",
                       json={"label": "x"}).status_code == 401
    assert client.get("/agent/install-links").status_code == 401


def test_route_mint_response_never_contains_raw_key(client):
    import re
    link = _mint(client, label="no leak")
    blob = json.dumps(link)
    assert "raw_key" not in blob
    # no key material of any shape (hp_ + 40 hex — the actual token format;
    # a bare "hp_" substring could occur by chance inside a token_urlsafe id):
    assert not re.search(r"hp_[0-9a-f]{40}", blob)
    assert link["url"].endswith(f"/agent/install/{link['link_id']}")
    assert link["label"] == "no leak"
    assert link["scopes"] == ["v1"]


def test_route_mint_validates_body(client):
    assert client.post("/agent/install-links", json={},
                       headers=_op()).status_code == 400          # no label
    assert client.post("/agent/install-links",
                       json={"label": "x", "scopes": "v1"},
                       headers=_op()).status_code == 400          # not a list
    assert client.post("/agent/install-links",
                       json={"label": "x", "scopes": ["nope"]},
                       headers=_op()).status_code == 400          # bad scope
    assert client.post("/agent/install-links",
                       json={"label": "x", "key_expires_at": "not-a-date"},
                       headers=_op()).status_code == 400


def test_route_mint_accepts_iso_key_expiry(client):
    link = _mint(client, label="iso", key_expires_at="2027-01-01T00:00:00Z")
    keys = ak.list_api_keys()
    rec = [k for k in keys if k["id"] == link["key_id"]][0]
    assert rec["expires_at"] and rec["expires_at"] > time.time()


def test_route_download_lifecycle_and_410(client):
    link = _mint(client, label="lifecycle")
    lid = link["link_id"]

    # .sh wrapper: free, keyless, points at the .py path
    r = client.get(f"/agent/install/{lid}.sh")
    assert r.status_code == 200
    sh = r.get_data(as_text=True)
    import re
    key_rx = re.compile(r"hp_[0-9a-f]{40}")
    assert f"/agent/install/{lid}" in sh
    assert "python3" in sh
    assert not key_rx.search(sh)                    # wrapper has NO key

    # .ps1 wrapper too
    r = client.get(f"/agent/install/{lid}.ps1")
    assert r.status_code == 200
    assert not key_rx.search(r.get_data(as_text=True))

    # wrappers consumed nothing:
    rows = client.get("/agent/install-links", headers=_op()).get_json()["links"]
    row = [x for x in rows if x["link_id"] == lid][0]
    assert row["uses_left"] == 1

    # the .py download: templated key, attachment, compiles
    r = client.get(f"/agent/install/{lid}")
    assert r.status_code == 200
    assert "attachment" in (r.headers.get("Content-Disposition") or "")
    body = r.get_data(as_text=True)
    assert 'EMBEDDED_API_KEY = ""' not in body      # slot was filled
    import re
    m = re.search(r"EMBEDDED_API_KEY = '(hp_[0-9a-f]+)'", body)
    assert m, "templated key missing from the download"
    raw = m.group(1)
    assert ak.verify_api_key(raw, required_scope="v1") is True
    # the served bytes are valid python:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        served = fh.name
    py_compile.compile(served, doraise=True)
    os.unlink(served)

    # exhausted now: the .py AND the wrappers all 410 with a human message
    for path in (f"/agent/install/{lid}",
                 f"/agent/install/{lid}.sh",
                 f"/agent/install/{lid}.ps1"):
        r = client.get(path)
        assert r.status_code == 410
        assert "no longer valid" in r.get_data(as_text=True)


def test_route_download_real_installer_compiles(client, monkeypatch):
    """Serve the ACTUAL repo installer (hugpy_agent/install/) templated — it
    must contain the key and py_compile. Skips honestly if the repo layout
    doesn't carry hugpy_agent/ (e.g. a packaged-only deployment)."""
    monkeypatch.delenv("HUGPY_AGENT_INSTALLER_PY", raising=False)
    real = agent_routes._find_installer_py()
    if real is None:
        pytest.skip("hugpy_agent/install/install_hugpy_agent.py not in this tree")
    link = _mint(client, label="real installer")
    r = client.get(f"/agent/install/{link['link_id']}")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "EMBEDDED_API_KEY = 'hp_" in body
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        served = fh.name
    py_compile.compile(served, doraise=True)
    os.unlink(served)


def test_route_expired_link_410(client):
    link = _mint(client, label="ttl")
    data = il._load()
    data["links"][link["link_id"]]["expires_at"] = time.time() - 5
    il._save(data)
    assert client.get(f"/agent/install/{link['link_id']}").status_code == 410
    assert client.get(f"/agent/install/{link['link_id']}.sh").status_code == 410


def test_route_revoke_kills_key_and_download(client):
    link = _mint(client, label="revoked")
    r = client.delete(f"/agent/install-links/{link['link_id']}", headers=_op())
    assert r.status_code == 200
    assert client.get(f"/agent/install/{link['link_id']}").status_code == 410
    assert all(k["id"] != link["key_id"] for k in ak.list_api_keys())
    # unknown id -> 404
    assert client.delete("/agent/install-links/nope",
                         headers=_op()).status_code == 404


def test_route_unknown_link_410(client):
    assert client.get("/agent/install/definitely-not-a-link").status_code == 410


def test_route_list_shows_status_and_counts_never_keys(client):
    a = _mint(client, label="a", max_uses=3)
    client.get(f"/agent/install/{a['link_id']}")     # consume one
    b = _mint(client, label="b")
    client.delete(f"/agent/install-links/{b['link_id']}", headers=_op())
    r = client.get("/agent/install-links", headers=_op())
    assert r.status_code == 200
    blob = r.get_data(as_text=True)
    import re
    assert not re.search(r"hp_[0-9a-f]{40}", blob) and "raw_key" not in blob
    rows = {x["link_id"]: x for x in r.get_json()["links"]}
    assert rows[a["link_id"]]["status"] == "active"
    assert rows[a["link_id"]]["uses_left"] == 2
    assert rows[b["link_id"]]["status"] == "revoked"


# ═══════════════════════════════════════════════════════════════════════════
# 4) the central operator allowlist
# ═══════════════════════════════════════════════════════════════════════════
def _sensitive(method, path):
    return any(method in methods and rx.match(path)
               for methods, rx in operator_auth._SENSITIVE)


def test_allowlist_covers_management_not_download():
    assert _sensitive("POST", "/agent/install-links")
    assert _sensitive("GET", "/agent/install-links")
    assert _sensitive("DELETE", "/agent/install-links/abc123")
    # the download GET is capability-gated by the link id, NOT operator-gated:
    assert not _sensitive("GET", "/agent/install/abc123")
    assert not _sensitive("GET", "/agent/install/abc123.sh")


def test_allowlist_agent_open_does_not_waive_install_links(monkeypatch):
    """The central gate's HUGPY_AGENT_OPEN waiver must skip /agent/install-links
    (credential-minting) while still waiving the fleet-view rules."""
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "true")
    app = Flask(__name__)
    with app.test_request_context("/agent/install-links", method="POST"):
        assert operator_auth._path_is_sensitive() is True
    with app.test_request_context("/agent/nodes", method="GET"):
        assert operator_auth._path_is_sensitive() is False   # waived, as before


# ═══════════════════════════════════════════════════════════════════════════
# 5) the scope-activated gates (v1/ml/agent-register call sites)
# ═══════════════════════════════════════════════════════════════════════════
def test_v1_scoped_install_key_cannot_register_an_agent(monkeypatch):
    """An install-link key minted with the default ["v1"] scope must NOT pass
    the /agent/register gate — the structural point of scoping."""
    link = il.create_install_link(label="v1 only")
    raw = il.consume_download(link["link_id"])
    assert ak.verify_api_key(raw, required_scope="v1") is True
    assert ak.verify_api_key(raw, required_scope="agent-register") is False

    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    c = app.test_client()
    r = c.post("/agent/register", json={"name": "sneaky"},
               headers={"Authorization": f"Bearer {raw}"})
    assert r.status_code == 401


def test_agent_register_scope_passes(monkeypatch):
    link = il.create_install_link(label="enroller",
                                  scopes=["agent-register"])
    raw = il.consume_download(link["link_id"])
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    c = app.test_client()
    r = c.post("/agent/register", json={"name": "legit"},
               headers={"Authorization": f"Bearer {raw}"})
    # 201 (the store is the real comms sqlite — scratch env) or 500 if the
    # comms db can't init here; the GATE decision is what we assert:
    assert r.status_code != 401
