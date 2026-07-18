"""The /video auth gate (video_auth.install_video_gate) + its structural
separation from the console/operator gate (operator_auth).

k8 (2026-07-18): the video arm went public/ungated 2026-07-15; the operator
directed it behind the SAME boundary as the console. This regresses:

  * the /video + /movie surface is DENIED for an anonymous caller (external
    mode): data/media/XHR -> 401, a browser shell navigation -> 302 to "/";
  * a valid console session (or operator token, or open mode) is ALLOWED through;
  * the SHARE SEAM: a video-scoped share principal satisfies the VIDEO gate but
    NEVER the console gate — share-principal present => /api/video ok while
    /api/keys is STILL 401. This is the load-bearing structural property (the
    share credential is video-scoped by construction);
  * the surface matcher covers /video, /api/video, /movie (and NOT /keys,
    /console, /workers).

Runs like the other tests here: venv/bin/python tests/test_video_gate.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-video-gate-test-")
# Deterministic: no operator token in the env for these cases.
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

import importlib

va = importlib.import_module("abstract_hugpy_dev.flask_app.app.video_auth")
oa = importlib.import_module("abstract_hugpy_dev.flask_app.app.operator_auth")

from flask import Flask

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


class _ApiPrefixMiddleware:
    """Mirror wsgi_app.ApiPrefixMiddleware: strip a leading /api BEFORE routing,
    exactly as production does (nginx/this middleware), so /api/video/... routes
    to the bare /video/... rule (not the SPA catch-all)."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app
    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


def build_app():
    """A minimal app that mirrors production wiring: the /api-strip middleware,
    the operator gate, then the video gate; a console-gated route (/keys, in
    operator_auth._SENSITIVE), a video data route, a movie route, an open route,
    and the SPA shell catch-all (endpoint _hugpy_ui) the video gate keys off for
    the redirect branch."""
    app = Flask(__name__)

    @app.route("/keys", methods=["GET"])
    def _keys():
        return "keys", 200

    @app.route("/video/studio/clips", methods=["GET"])
    def _clips():
        return "clips", 200

    @app.route("/movie/presets", methods=["GET"])
    def _movie():
        return "movie", 200

    @app.route("/llm/workers", methods=["GET"])
    def _workers():
        return "workers", 200

    def _shell(asset=""):
        return "<html>shell</html>", 200
    app.add_url_rule("/", endpoint="_hugpy_ui", view_func=_shell, defaults={"asset": ""})
    app.add_url_rule("/<path:asset>", endpoint="_hugpy_ui", view_func=_shell)

    oa.install_operator_gate(app)
    va.install_video_gate(app)
    app.wsgi_app = _ApiPrefixMiddleware(app.wsgi_app)
    return app


# --- unit: the surface matcher -------------------------------------------------
with build_app().test_request_context("/video/studio/clips"):
    check("surface: /video is video", va._is_video_surface() is True)
with build_app().test_request_context("/api/video/studio/clips"):
    check("surface: /api/video is video (after /api strip)", va._is_video_surface() is True)
with build_app().test_request_context("/movie/presets"):
    check("surface: /movie is video surface", va._is_video_surface() is True)
with build_app().test_request_context("/video"):
    check("surface: bare /video is video", va._is_video_surface() is True)
with build_app().test_request_context("/keys"):
    check("surface: /keys is NOT video", va._is_video_surface() is False)
with build_app().test_request_context("/videofoo"):
    check("surface: /videofoo is NOT video (word-boundary)", va._is_video_surface() is False)
with build_app().test_request_context("/llm/workers"):
    check("surface: /llm/workers is NOT video", va._is_video_surface() is False)

# --- EXTERNAL mode: anonymous is denied on video, and on console --------------
os.environ["HUGPY_AUTH_MODE"] = "external"
oa._SESSION_CACHE.clear()

_orig_session = oa._validate_session_external
_orig_share = va._video_share_principal
try:
    # No session, no token, no share credential.
    oa._validate_session_external = lambda: False
    va._video_share_principal = lambda request: None

    app = build_app()
    c = app.test_client()

    check("anon: /api/video/studio/clips -> 401 (data)",
          c.get("/api/video/studio/clips").status_code == 401)
    check("anon: /video/studio/clips -> 401 (data)",
          c.get("/video/studio/clips").status_code == 401)
    check("anon: /movie/presets -> 401",
          c.get("/movie/presets").status_code == 401)
    # Browser shell navigation -> redirect to the console login at "/".
    r = c.get("/video/", headers={"Sec-Fetch-Dest": "document"})
    check("anon: /video/ browser shell -> 302 redirect", r.status_code == 302)
    check("anon: shell redirect targets console root '/'",
          (r.headers.get("Location") or "").endswith("/"))
    # Console boundary still works exactly as before (operator gate).
    check("anon: /api/keys -> 401 (console gate)",
          c.get("/api/keys").status_code == 401)
    # Non-gated reads stay open.
    check("anon: /llm/workers stays open (200)",
          c.get("/llm/workers").status_code == 200)

    # --- THE STRUCTURAL PROPERTY: a video-share principal opens /video ONLY ----
    va._video_share_principal = lambda request: {"scope": "video", "share": "tok123"}
    oa._validate_session_external = lambda: False  # still NO console session
    oa._SESSION_CACHE.clear()

    app = build_app()
    c = app.test_client()
    check("share-principal: /api/video/studio/clips -> 200 (video gate satisfied)",
          c.get("/api/video/studio/clips").status_code == 200)
    check("share-principal: /movie/presets -> 200",
          c.get("/movie/presets").status_code == 200)
    check("share-principal: /api/keys STILL 401 (share cannot satisfy console gate)",
          c.get("/api/keys").status_code == 401)
    check("share-principal: /video/ shell -> 200 (served)",
          c.get("/video/", headers={"Sec-Fetch-Dest": "document"}).status_code == 200)

    # --- a valid console session opens BOTH surfaces --------------------------
    va._video_share_principal = lambda request: None
    oa._validate_session_external = lambda: True
    oa._SESSION_CACHE.clear()

    app = build_app()
    c = app.test_client()
    check("session: /api/video/studio/clips -> 200", c.get("/api/video/studio/clips").status_code == 200)
    check("session: /api/keys -> 200 (console session valid)", c.get("/api/keys").status_code == 200)
    check("session: /video/ shell -> 200", c.get("/video/").status_code == 200)
finally:
    oa._validate_session_external = _orig_session
    va._video_share_principal = _orig_share

# --- OPEN mode (self-hosted product, no token): the arm is open, no login -----
os.environ["HUGPY_AUTH_MODE"] = "open"
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
oa._SESSION_CACHE.clear()
app = build_app()
c = app.test_client()
check("open-mode: /api/video/studio/clips -> 200 (no login in self-hosted)",
      c.get("/api/video/studio/clips").status_code == 200)
check("open-mode: /api/keys -> 200 (open)", c.get("/api/keys").status_code == 200)

# --- OPEN mode WITH an operator token: video requires the token too -----------
os.environ["HUGPY_OPERATOR_TOKEN"] = "s3cret"
oa._SESSION_CACHE.clear()
app = build_app()
c = app.test_client()
check("open+token: anon /api/video -> 401 (locked management surface)",
      c.get("/api/video/studio/clips").status_code == 401)
check("open+token: with token -> 200",
      c.get("/api/video/studio/clips", headers={"X-Operator-Token": "s3cret"}).status_code == 200)
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

print(f"\nall {ok} checks passed")
