### routes/upload_routes.py

from ..functions import *
import os
import re
import time
import shutil

upload_bp, logger = get_bp("upload_bp", __name__)

# ── per-session upload lifecycle ─────────────────────────────────────────────
# Uploads are tagged with the browser's per-tab session id (sessionStorage), sent
# as the `X-Hugpy-Session` header. Each session's saved paths + a last-seen marker
# live under UPLOADS_HOME/.sessions/<sid>/. The browser heartbeats (/session/ping)
# to keep its session alive; on tab close it beacons (/session/end?sid=) to wipe
# immediately. A throttled, request-driven sweep wipes any session idle past
# SESSION_TTL — the 1h safety net for missed beacons. No cron / background thread:
# the sweep piggybacks on ping/upload, so it self-cleans whenever anyone is around.

SESSIONS_DIR = ".sessions"
SESSION_TTL = 3600          # wipe a session's uploads 1h after its last heartbeat
SWEEP_EVERY = 600           # at most one sweep per 10 min (race-tolerant across workers)
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _sessions_root():
    return os.path.join(UPLOADS_HOME, SESSIONS_DIR)


def _valid_sid(sid):
    return isinstance(sid, str) and bool(_SID_RE.match(sid))


def _read_sid():
    # sendBeacon uses the query string (?sid=); fetch heartbeats use a header/JSON.
    sid = request.headers.get("X-Hugpy-Session") or request.args.get("sid")
    if not sid:
        body = request.get_json(silent=True)
        if isinstance(body, dict):
            sid = body.get("sid")
    if not sid:
        sid = request.form.get("sid")
    return sid if _valid_sid(sid) else None


def _touch_session(sid, path=None):
    d = os.path.join(_sessions_root(), sid)
    os.makedirs(d, exist_ok=True)
    if path:
        try:
            with open(os.path.join(d, "paths"), "a") as fh:
                fh.write(path + "\n")
        except OSError:
            pass
    # `seen` mtime is the session's last-seen timestamp.
    open(os.path.join(d, "seen"), "w").close()


def _within_uploads(path):
    root = os.path.realpath(UPLOADS_HOME)
    rp = os.path.realpath(path)
    return rp == root or rp.startswith(root + os.sep)


def _wipe_session(sid):
    d = os.path.join(_sessions_root(), sid)
    try:
        with open(os.path.join(d, "paths")) as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        paths = []
    for p in paths:
        try:
            if _within_uploads(p) and os.path.isfile(p):
                os.unlink(p)
        except OSError:
            pass
    shutil.rmtree(d, ignore_errors=True)


def _maybe_sweep():
    root = _sessions_root()
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        return
    marker = os.path.join(root, ".last_sweep")
    now = time.time()
    try:
        if now - os.path.getmtime(marker) < SWEEP_EVERY:
            return                       # swept recently — skip
    except OSError:
        pass                             # marker missing → run
    try:
        open(marker, "w").close()        # claim this sweep window
    except OSError:
        pass
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        if name.startswith("."):
            continue
        d = os.path.join(root, name)
        try:
            idle = now - os.path.getmtime(os.path.join(d, "seen"))
        except OSError:
            try:
                idle = now - os.path.getmtime(d)
            except OSError:
                continue
        if idle > SESSION_TTL:
            _wipe_session(name)


@upload_bp.route("/uploads", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, description="no file provided")
    os.makedirs(UPLOADS_HOME, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
    dest = os.path.join(UPLOADS_HOME, name)
    f.save(dest)
    sid = _read_sid()
    if sid:
        _touch_session(sid, dest)
        _maybe_sweep()
    return jsonify({"path": dest, "name": f.filename, "size": os.path.getsize(dest)})


@upload_bp.route("/session/ping", methods=["POST"])
def session_ping():
    """Heartbeat: keep this session's uploads alive; opportunistically sweep idle ones."""
    sid = _read_sid()
    if sid:
        _touch_session(sid)
        _maybe_sweep()
    return jsonify({"ok": True})


@upload_bp.route("/session/end", methods=["POST"])
def session_end():
    """Tab-close beacon: wipe this session's uploads immediately (best-effort)."""
    sid = _read_sid()
    if sid:
        _wipe_session(sid)
    return jsonify({"ok": True})


def _forget_path(sid, base):
    """Drop any path whose basename == `base` from this session's registry."""
    pf = os.path.join(_sessions_root(), sid, "paths")
    try:
        with open(pf) as fh:
            kept = [ln.strip() for ln in fh
                    if ln.strip() and os.path.basename(ln.strip()) != base]
    except OSError:
        return
    try:
        with open(pf, "w") as fh:
            for p in kept:
                fh.write(p + "\n")
    except OSError:
        pass


@upload_bp.route("/session/file", methods=["DELETE", "POST"])
def session_file_delete():
    """User-initiated delete of ONE uploaded file from the store. Accepts the
    file id (the /uploads basename, or a full path). Only the basename under
    UPLOADS_HOME is ever touched — no traversal — and it's dropped from the
    session registry so the sweep/wipe won't trip over a stale entry."""
    fid = (request.args.get("id")
           or (request.get_json(silent=True) or {}).get("id")
           or request.form.get("id"))
    if not fid:
        return jsonify({"ok": False, "error": "missing id"}), 400
    base = os.path.basename(str(fid).strip())
    if not base or base in (".", ".."):
        return jsonify({"ok": False, "error": "bad id"}), 400
    target = os.path.join(UPLOADS_HOME, base)
    deleted = False
    try:
        if _within_uploads(target) and os.path.isfile(target):
            os.unlink(target)
            deleted = True
    except OSError:
        pass
    sid = _read_sid()
    if sid:
        _forget_path(sid, base)
    return jsonify({"ok": True, "deleted": deleted})
