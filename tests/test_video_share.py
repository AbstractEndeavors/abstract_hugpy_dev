"""k9 — video SHARE-LINK feature: the key store, the /video gate's share seam,
the STRUCTURAL hardening (a share key opens /video but can NEVER mint another key
or reach any console/operator route), and job attribution.

This is the SIBLING of test_video_gate.py (which pins the k8 gate + its 25
checks). Those 25 stay green untouched — the k8 test overrides the share seam in
every case, so this real implementation never perturbs it. Here we exercise the
real implementation instead.

Runs like the other tests here:  venv/bin/python tests/test_video_share.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-video-share-test-")
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

import importlib

from flask import Flask

vsk = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.video_share_keys")
va = importlib.import_module("abstract_hugpy_dev.flask_app.app.video_auth")
oa = importlib.import_module("abstract_hugpy_dev.flask_app.app.operator_auth")

# Point the share-key store at a throwaway temp file (no dependency on the real
# manifest/settings resolution — the store idiom is what we're testing).
_STORE_FILE = os.path.join(
    tempfile.mkdtemp(prefix="hugpy-share-store-"), "video_share_keys.json")
vsk._store_path = lambda: _STORE_FILE

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# --------------------------------------------------------------------------- #
# 1) the key store: mint / verify / expiry / revoke / wrong-category
# --------------------------------------------------------------------------- #
minted = vsk.create_share_key(label="Alex — review", ttl_days=30)
check("mint: returns the full key ONCE (hpv_ prefix)",
      minted["key"].startswith("hpv_") and len(minted["key"]) > 20)
check("mint: hash is NOT returned", "hash" not in minted)
check("mint: carries an id + label + expires_at",
      minted["id"] and minted["label"] == "Alex — review" and minted["expires_at"])

good = minted["key"]
check("verify: a good key resolves to its key_id",
      vsk.verify_share_key(good) == minted["id"])
check("principal: a good key => share:<id>",
      vsk.share_principal(good) == f"share:{minted['id']}")

check("verify: garbage => None", vsk.verify_share_key("hpv_deadbeef") is None)
check("verify: empty => None", vsk.verify_share_key("") is None)
check("verify: None => None", vsk.verify_share_key(None) is None)
# Wrong category: a /v1-style api key (hp_) is NOT in the share store.
check("verify: a foreign (hp_) token => None (separate category/store)",
      vsk.verify_share_key("hp_" + "a" * 40) is None)

# Expired key: mint with a ttl and force expires_at into the past.
exp = vsk.create_share_key(label="stale", ttl_days=1)
data = vsk._load()
data["keys"][exp["id"]]["expires_at"] = time.time() - 10
vsk._save(data)
check("verify: an EXPIRED key => None", vsk.verify_share_key(exp["key"]) is None)
check("principal: an expired key => None", vsk.share_principal(exp["key"]) is None)

# Non-expiring link (ttl_days<=0).
forever = vsk.create_share_key(label="forever", ttl_days=0)
check("mint: ttl_days<=0 => no expiry", forever["expires_at"] is None)
check("verify: a non-expiring key resolves",
      vsk.verify_share_key(forever["key"]) == forever["id"])

# Revoke.
check("revoke: known id => True", vsk.revoke_share_key(minted["id"]) is True)
check("verify: a REVOKED key => None", vsk.verify_share_key(good) is None)
check("revoke: unknown id => False", vsk.revoke_share_key("nope") is False)

# List: non-revoked only, newest first, expired flagged.
listed = vsk.list_share_keys()
listed_ids = {k["id"] for k in listed}
check("list: revoked key is excluded", minted["id"] not in listed_ids)
check("list: active keys included", forever["id"] in listed_ids)
check("list: never leaks the hash", all("hash" not in k for k in listed))
check("list: the expired key is present but flagged expired",
      any(k["id"] == exp["id"] and k["expired"] for k in listed))

# The two categories live in DIFFERENT files (structural isolation, not a tag).
ak = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.api_keys")
check("stores: api_keys.json and video_share_keys.json are DISTINCT files",
      os.path.basename(ak._store_path()) == "api_keys.json"
      and os.path.basename(vsk._store_path()) == "video_share_keys.json")


# --------------------------------------------------------------------------- #
# 2) the /video gate's share seam — the three credential carriers
# --------------------------------------------------------------------------- #
live = vsk.create_share_key(label="seam", ttl_days=30)
app = Flask(__name__)

with app.test_request_context(f"/video/x?share={live['key']}"):
    from flask import request as _rq
    check("seam: ?share= query => share:<id>",
          va._video_share_principal(_rq) == f"share:{live['id']}")
with app.test_request_context("/video/x", headers={"X-Video-Share": live["key"]}):
    from flask import request as _rq
    check("seam: X-Video-Share header => share:<id>",
          va._video_share_principal(_rq) == f"share:{live['id']}")
with app.test_request_context(
        "/video/x", headers={"Authorization": f"Bearer {live['key']}"}):
    from flask import request as _rq
    check("seam: Authorization Bearer hpv_ => share:<id>",
          va._video_share_principal(_rq) == f"share:{live['id']}")
with app.test_request_context("/video/x"):
    from flask import request as _rq
    check("seam: no credential => None", va._video_share_principal(_rq) is None)
with app.test_request_context("/video/x", headers={"X-Video-Share": "hpv_bogus"}):
    from flask import request as _rq
    check("seam: an invalid share key => None",
          va._video_share_principal(_rq) is None)


# --------------------------------------------------------------------------- #
# 3) THE STRUCTURAL PIN (extended): a share key opens /video, but can NEVER
#    reach the mint route (no key-minting-by-key) or any console/operator route.
# --------------------------------------------------------------------------- #
class _ApiPrefixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app
    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


def build_app():
    """Mirrors production wiring: /api-strip -> operator gate -> video gate, with
    the console-gated /keys, the OPERATOR-gated /keys/video-share mint route, a
    /video data route, and the SPA shell catch-all."""
    a = Flask(__name__)

    @a.route("/keys", methods=["GET"])
    def _keys():
        return "keys", 200

    @a.route("/keys/video-share", methods=["GET", "POST"])
    def _mint():
        return "minted", 200  # only reachable past the OPERATOR gate

    @a.route("/video/studio/clips", methods=["GET"])
    def _clips():
        return "clips", 200

    def _shell(asset=""):
        return "<html>shell</html>", 200
    a.add_url_rule("/", endpoint="_hugpy_ui", view_func=_shell, defaults={"asset": ""})
    a.add_url_rule("/<path:asset>", endpoint="_hugpy_ui", view_func=_shell)

    oa.install_operator_gate(a)
    va.install_video_gate(a)
    a.wsgi_app = _ApiPrefixMiddleware(a.wsgi_app)
    return a


os.environ["HUGPY_AUTH_MODE"] = "external"
oa._SESSION_CACHE.clear()
_orig_session = oa._validate_session_external
try:
    oa._validate_session_external = lambda: False  # NO console session anywhere

    a = build_app()
    c = a.test_client()
    hdr = {"X-Video-Share": live["key"]}

    # The share key opens the /video surface.
    check("share-key: /api/video/studio/clips -> 200 (video gate satisfied)",
          c.get("/api/video/studio/clips", headers=hdr).status_code == 200)
    # ...but is powerless against the mint route (operator-gated + off the /video
    # surface) — this is the "no key-minting-by-key" hardening.
    check("share-key: GET /api/keys/video-share -> 401 (cannot list/mint)",
          c.get("/api/keys/video-share", headers=hdr).status_code == 401)
    check("share-key: POST /api/keys/video-share -> 401 (cannot mint another key)",
          c.post("/api/keys/video-share", headers=hdr).status_code == 401)
    # ...and never satisfies the console gate at large.
    check("share-key: /api/keys -> 401 (share cannot satisfy console gate)",
          c.get("/api/keys", headers=hdr).status_code == 401)

    # A bare anon caller is denied the mint route too (baseline).
    check("anon: POST /api/keys/video-share -> 401 (operator-only mint)",
          c.post("/api/keys/video-share").status_code == 401)

    # A console session opens BOTH the video surface AND the mint route.
    oa._validate_session_external = lambda: True
    oa._SESSION_CACHE.clear()
    a = build_app()
    c = a.test_client()
    check("session: POST /api/keys/video-share -> 200 (operator can mint)",
          c.post("/api/keys/video-share").status_code == 200)
    check("session: /api/video/studio/clips -> 200",
          c.get("/api/video/studio/clips").status_code == 200)
finally:
    oa._validate_session_external = _orig_session


# --------------------------------------------------------------------------- #
# 4) attribution: a share principal lands on the job, end to end.
# --------------------------------------------------------------------------- #
# (a) media_bus persists the principal on the job row + carries it to the bridge.
import abstract_hugpy_dev.video_intel.media_bus as mb
mb.DB_PATH = os.path.join(tempfile.mkdtemp(prefix="hugpy-mediabus-"), "media_jobs.db")
mb._initialized = False
mb.serialize_spec = lambda name, spec: "{}"  # bypass real spec serialization
jid = mb.enqueue("crop", object(), principal="share:xyz789")
conn = mb._connect()
try:
    row = conn.execute(
        "SELECT principal FROM media_jobs WHERE job_id=?", (jid,)).fetchone()
finally:
    conn.close()
check("attribution: media_bus stores the principal on the job row",
      row is not None and row[0] == "share:xyz789")

# (b) the bus->JobStore bridge carries the principal onto the comms Job that
#     GET /llm/jobs reads (running + terminal, in the process that runs the job).
os.environ.pop("HUGPY_COMMS_DB", None)  # in-process store, no cross-proc mirror
from abstract_hugpy_dev.comms import job_store
job_bridge = importlib.import_module("abstract_hugpy_dev.video_intel.job_bridge")

job_bridge.on_running("attrib-job-1", "studio_i2v", worker="w", principal="share:xyz789")
j = job_store.get("attrib-job-1")
check("attribution: on_running stamps principal onto the comms Job",
      j is not None and j.principal == "share:xyz789")
check("attribution: the principal is on the /llm/jobs wire dict",
      j.to_dict().get("principal") == "share:xyz789")

job_bridge.on_terminal("attrib-job-1", "studio_i2v", "done", principal="share:xyz789")
j2 = job_store.get("attrib-job-1")
check("attribution: it survives to the terminal record",
      j2 is not None and j2.principal == "share:xyz789")

# A None principal (unattributed job) must never blank an existing attribution.
job_bridge.on_running("attrib-job-1", "studio_i2v", principal=None)
check("attribution: a later None never clobbers an existing principal",
      job_store.get("attrib-job-1").principal == "share:xyz789")


print(f"\nall {ok} checks passed")
