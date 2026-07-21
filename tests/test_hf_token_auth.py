"""k29 — Hugging Face credentials store + routes.

Covers: token-store round-trip (0600, precedence, delete), whoami validation
(mocked — no live HF), the GET/POST/DELETE routes end-to-end through the REAL
operator gate, invalid-token rejection (400), and that the routes are in the
operator _SENSITIVE allowlist.

Runs like the other tests here:  venv/bin/python tests/test_hf_token_auth.py
"""
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-hf-token-test-")
# Clean env so source detection is deterministic (no ambient HF_TOKEN).
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
# Operator gate ON: open mode + a token means the gate enforces.
os.environ["HUGPY_AUTH_MODE"] = "open"
os.environ["HUGPY_OPERATOR_TOKEN"] = "op-secret-xyz"

import importlib

# Isolate the token file to a throwaway path. PROJECTS_HOME resolves through the
# secure-store get_env_value (NOT os.environ), so it points at the REAL state
# root — we must override HF_TOKEN_PATH on both the constants layer (which the
# reader keys off) and the store module so the test never touches a real token.
_C = importlib.import_module("abstract_hugpy_dev.imports.src.constants.constants")
_TOKEN_DIR = tempfile.mkdtemp(prefix="hugpy-hf-token-file-")
_TOKEN_PATH = os.path.join(_TOKEN_DIR, "hf_token")
_C.HF_TOKEN_PATH = _TOKEN_PATH

hf = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.hf_token")
hf.HF_TOKEN_PATH = _TOKEN_PATH
# constants seeded os.environ["HF_TOKEN"] from the box's real stored token at
# import — clear it so this test starts from a clean, deterministic slate (the
# real token file is untouched; we only ever read/write _TOKEN_PATH here).
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ── whoami mock: swap requests.get so no network is touched ──────────────────
import requests

class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

_VALID = "hf_valid_token_abcd1234"
_ROUTE = {"status": 200, "payload": {"name": "octocat"}}  # mutable per-test

_orig_get = requests.get
def _fake_get(url, headers=None, timeout=None, **kw):
    assert "whoami-v2" in url
    sent = (headers or {}).get("Authorization", "")
    if sent == f"Bearer {_VALID}":
        return _Resp(200, {"name": "octocat", "fullname": "The Octocat"})
    return _Resp(401, {"error": "Invalid credentials in Authorization header"})
requests.get = _fake_get

try:
    # ── 1) validation ────────────────────────────────────────────────────────
    st, user, err = hf.validate_hf_token(_VALID)
    check("validate: valid token -> ok + username", st == "ok" and user == "octocat")
    st, user, err = hf.validate_hf_token("hf_bogus")
    check("validate: bad token -> invalid + HF message", st == "invalid" and "Invalid" in (err or ""))

    # ── 2) store round-trip + 0600 + precedence ──────────────────────────────
    check("get: none stored, no env -> None", hf.get_hf_token() is None)
    check("source: none -> None", hf.token_source() is None)

    hf.store_hf_token(_VALID)
    check("store: get_hf_token returns the stored token", hf.get_hf_token() == _VALID)
    check("store: source == stored", hf.token_source() == "stored")
    mode = stat.S_IMODE(os.stat(hf.HF_TOKEN_PATH).st_mode)
    check("store: file is 0600", mode == 0o600)
    # stored file lives at the isolated state-root path (outside any git tree)
    check("store: path is the isolated state-root token file",
          hf.HF_TOKEN_PATH == _TOKEN_PATH and os.path.exists(_TOKEN_PATH))
    # apply pushed it into the env seam
    check("apply: env HF_TOKEN set for implicit call sites",
          os.environ.get("HF_TOKEN") == _VALID)

    # stored wins over env (env source = the genuine captured constants.HF_TOKEN_ENV)
    _C.HF_TOKEN_ENV = "env-fallback-token"
    check("precedence: stored token wins over env", hf.get_hf_token() == _VALID)
    check("precedence: source still stored while a file exists", hf.token_source() == "stored")

    # ── 3) delete -> falls back to the genuine env token ─────────────────────
    removed = hf.delete_hf_token()
    check("delete: removed True", removed is True)
    check("delete: file gone", not os.path.exists(hf.HF_TOKEN_PATH))
    check("delete: env token is the fallback", hf.get_hf_token() == "env-fallback-token")
    check("delete: source == env", hf.token_source() == "env")
    _C.HF_TOKEN_ENV = False   # back to a truly clean slate
    hf.apply_hf_token_to_env()

    # ── 4) routes end-to-end through the REAL operator gate ──────────────────
    from flask import Flask
    sr = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.routes.search_routes")
    oa = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.operator_auth")

    app = Flask(__name__)
    app.register_blueprint(sr.search_bp)
    oa.install_operator_gate(app)
    client = app.test_client()
    H = {"X-Operator-Token": "op-secret-xyz"}

    # gate: unauthenticated is refused on every verb
    check("gate: GET without token -> 401", client.get("/llm/hf/auth").status_code == 401)
    check("gate: POST without token -> 401",
          client.post("/llm/hf/auth", json={"token": _VALID}).status_code == 401)
    check("gate: DELETE without token -> 401", client.delete("/llm/hf/auth").status_code == 401)

    # GET anonymous shape (no token stored)
    r = client.get("/llm/hf/auth", headers=H)
    j = r.get_json()
    check("GET: 200 anonymous shape",
          r.status_code == 200 and j["authenticated"] is False
          and j["username"] is None and j["token_last4"] is None and j["source"] is None)

    # POST invalid token -> 400 with HF's message, nothing stored
    r = client.post("/llm/hf/auth", headers=H, json={"token": "hf_bogus"})
    check("POST invalid: 400", r.status_code == 400)
    check("POST invalid: nothing persisted", hf.get_hf_token() is None)

    # POST missing token -> 400
    check("POST empty: 400", client.post("/llm/hf/auth", headers=H, json={}).status_code == 400)

    # POST valid -> 200, stored, authenticated, last4, token never echoed
    r = client.post("/llm/hf/auth", headers=H, json={"token": _VALID})
    j = r.get_json()
    check("POST valid: 200 + authenticated + username",
          r.status_code == 200 and j["authenticated"] is True and j["username"] == "octocat")
    check("POST valid: last4 only, never the token",
          j["token_last4"] == _VALID[-4:] and _VALID not in (r.get_data(as_text=True)))
    check("POST valid: source stored", j["source"] == "stored")

    # DELETE -> 200, removed, back to anonymous
    r = client.delete("/llm/hf/auth", headers=H)
    j = r.get_json()
    check("DELETE: 200 + removed + anonymous",
          r.status_code == 200 and j.get("removed") is True and j["source"] is None)

    # ── 5) gating declared in _SENSITIVE (bare + /api-mounted) ───────────────
    def _gated(path, method):
        p = path
        if p == "/api" or p.startswith("/api/"):
            p = p[len("/api"):] or "/"
        return any(method in m and rx.match(p) for m, rx in oa._SENSITIVE)
    for verb in ("GET", "POST", "DELETE"):
        check(f"gating: {verb} /llm/hf/auth operator-gated", _gated("/llm/hf/auth", verb))
    check("gating: /api-mounted path also gated", _gated("/api/llm/hf/auth", "POST"))

finally:
    requests.get = _orig_get

print(f"\nALL {ok} CHECKS PASSED")
