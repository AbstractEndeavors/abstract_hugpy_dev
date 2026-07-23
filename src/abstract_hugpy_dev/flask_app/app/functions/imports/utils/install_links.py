"""One-time secure INSTALL LINKS for the hugpy-agent installer (2026-07-23).

WHAT THIS IS
------------
The operator mints a labeled, scoped, revocable install link from the console's
API tab. Fetching the link serves ``install_hugpy_agent.py`` with a freshly
minted API key baked into its ``EMBEDDED_API_KEY`` slot — so a new box installs
and enrolls with ONE line and the operator never handles (or sees) the raw key.

THE KEY IS NEVER A STANDING OPERATOR KEY. Each link mints its OWN key via
``api_keys.create_api_key`` (created_by="install-link", the operator's label,
the operator's chosen scopes — default ["v1"]) so it is individually
scoped/labeled/revocable like any other console key.

WHERE THE RAW KEY LIVES (read before touching)
----------------------------------------------
``api_keys`` stores only the sha256 of a key — by design it can never re-reveal
one. But an install link must template the RAW key into a download that happens
LATER than the mint. So the raw key is held here, in the link record, for the
link's lifetime only, and is SCRUBBED (overwritten with "") the moment the link
can no longer serve it:
  * on the download that exhausts the last use,
  * on revoke,
  * lazily on first touch after expiry.
This store file therefore holds live secrets while links are active — the same
exposure class as api_keys.json holding hashes plus the manifest dir generally
(server-side, storage-root, never served). The mint response NEVER contains the
raw key; the ONLY place it ever leaves the server is inside the templated
installer download itself.

USE COUNTING (wrapper vs payload)
---------------------------------
``/agent/install/<link_id>.sh`` and ``.ps1`` are convenience wrappers that
locate a python and curl the ``.py`` from the same link path. Only the ``.py``
fetch decrements ``uses_left`` — the wrapper fetch does NOT — so the canonical
one-liner (wrapper fetch + the wrapper's own .py fetch) counts as ONE use, not
two. Documented contract; the route layer enforces it by calling
``consume_download`` only from the ``.py`` branch.

Storage idiom mirrors ``video_share_keys.py``: a small JSON file next to the
model manifest, process-wide lock, unique-per-write temp + os.replace.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from typing import Any, Optional

from .schemas import settings
from . import api_keys

_LOCK = threading.Lock()

DEFAULT_LINK_TTL_S = 86400          # 24h
DEFAULT_MAX_USES = 1
DEFAULT_SCOPES = ["v1"]


def _store_path() -> str:
    return os.path.join(os.path.dirname(settings.manifest_path), "install_links.json")


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        return {"links": {}, "audit": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"links": {}, "audit": []}
    data.setdefault("links", {})
    data.setdefault("audit", [])
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Unique temp name per write (pid+token) — same multi-process atomicity
    # rationale api_keys.py documents; os.replace stays the atomicity point.
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _now() -> float:
    return time.time()


def _is_expired(rec: dict[str, Any], now: Optional[float] = None) -> bool:
    exp = rec.get("expires_at")
    if not exp:
        return False
    try:
        return float(exp) <= (now if now is not None else _now())
    except (TypeError, ValueError):
        return False


def _status(rec: dict[str, Any]) -> str:
    if rec.get("revoked"):
        return "revoked"
    if _is_expired(rec):
        return "expired"
    if int(rec.get("uses_left") or 0) <= 0:
        return "exhausted"
    return "active"


def _scrub(rec: dict[str, Any]) -> None:
    """Drop the raw key from a record that can no longer serve a download."""
    rec["raw_key"] = ""


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    """A link record WITHOUT the raw key, plus its computed status."""
    out = {k: v for k, v in rec.items() if k != "raw_key"}
    out["status"] = _status(rec)
    return out


def create_install_link(label: str,
                        scopes: Optional[list[str]] = None,
                        key_expires_at: Optional[float] = None,
                        link_ttl_s: Optional[float] = None,
                        max_uses: Optional[int] = None) -> dict[str, Any]:
    """Mint a fresh scoped key + its one-time link. Returns the PUBLIC view
    (link_id, key_id, label, scopes, expires_at, uses…) — NEVER the raw key.

    Raises ValueError on a blank label or unknown scope (api_keys validates
    the vocabulary)."""
    label = (label or "").strip()
    if not label:
        raise ValueError("an install link requires a label")
    scopes = list(scopes) if scopes else list(DEFAULT_SCOPES)
    try:
        ttl = float(link_ttl_s) if link_ttl_s is not None else float(DEFAULT_LINK_TTL_S)
    except (TypeError, ValueError):
        ttl = float(DEFAULT_LINK_TTL_S)
    try:
        uses = int(max_uses) if max_uses is not None else DEFAULT_MAX_USES
    except (TypeError, ValueError):
        uses = DEFAULT_MAX_USES
    uses = max(1, uses)

    # Mint the key FIRST (api_keys raises on a bad scope before anything is
    # persisted here). name = the label so the key list reads sensibly.
    minted = api_keys.create_api_key(
        name=label, label=label, scopes=scopes,
        created_by="install-link", expires_at=key_expires_at)
    raw_key = minted["key"]

    link_id = secrets.token_urlsafe(24)
    now = _now()
    rec = {
        "link_id": link_id,
        "key_id": minted["id"],
        "label": label,
        "scopes": scopes,
        "created_at": now,
        "expires_at": (now + ttl) if ttl > 0 else None,
        "key_expires_at": float(key_expires_at) if key_expires_at else None,
        "max_uses": uses,
        "uses_left": uses,
        "downloads": [],       # audit rows: {ts, remote_addr, kind}
        "revoked": False,
        "raw_key": raw_key,    # scrubbed on exhaustion/revoke/expiry
    }
    with _LOCK:
        data = _load()
        data["links"][link_id] = rec
        _save(data)
    return _public(rec)


def get_link(link_id: str) -> Optional[dict[str, Any]]:
    """Public view of one link (no raw key), or None."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        return _public(rec) if rec else None


def list_install_links() -> list[dict[str, Any]]:
    """Every link (incl. exhausted/expired/revoked — the operator sees the full
    ledger), newest first, never the raw key."""
    with _LOCK:
        data = _load()
    out = [_public(rec) for rec in data["links"].values()]
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


def consume_download(link_id: str, remote_addr: str = "") -> Optional[str]:
    """The .py download path: if the link is active AND its key still verifies
    as un-revoked, return the RAW KEY, decrement uses_left, audit the download,
    and scrub the raw key if that was the last use. Returns None (and scrubs
    where appropriate) for exhausted/expired/revoked links or a revoked key.

    This is the ONLY function that ever returns the raw key."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return None
        if rec.get("revoked") or _is_expired(rec) or int(rec.get("uses_left") or 0) <= 0:
            if rec.get("raw_key"):
                _scrub(rec)
                _save(data)
            return None
        raw = rec.get("raw_key") or ""
        if not raw:
            return None  # already scrubbed (shouldn't happen while active)
        rec["uses_left"] = int(rec["uses_left"]) - 1
        rec["downloads"].append({"ts": _now(),
                                 "remote_addr": remote_addr or "",
                                 "kind": "py"})
        if rec["uses_left"] <= 0:
            _scrub(rec)
        _save(data)
    # Key revocation check OUTSIDE our lock (api_keys has its own): a link whose
    # key the operator already revoked must not hand out a dead — or worse,
    # resurrected-looking — credential.
    if not api_keys.verify_api_key(raw):
        return None
    return raw


def peek_active(link_id: str) -> bool:
    """True iff the link could currently serve a download — used by the .sh/.ps1
    wrapper routes so a dead link 410s at the wrapper too, WITHOUT consuming a
    use (only the .py fetch decrements)."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return False
        return _status(rec) == "active" and bool(rec.get("raw_key"))


def note_wrapper_fetch(link_id: str, remote_addr: str = "", kind: str = "sh") -> None:
    """Audit a wrapper (.sh/.ps1) fetch. Does NOT decrement uses_left."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return
        rec.setdefault("downloads", []).append(
            {"ts": _now(), "remote_addr": remote_addr or "", "kind": kind})
        _save(data)


def revoke_install_link(link_id: str) -> bool:
    """Revoke a link AND the key behind it. Idempotent-ish: revoking an already
    revoked link still ensures the key is revoked. False only for unknown ids."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return False
        rec["revoked"] = True
        _scrub(rec)
        _save(data)
        key_id = rec.get("key_id")
    if key_id:
        api_keys.revoke_api_key(key_id)
    return True
