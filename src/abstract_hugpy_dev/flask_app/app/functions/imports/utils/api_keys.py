"""API-key store for the public /v1 inference API.

Keys are created from the UI, shown to the user exactly once, and stored
hashed (sha256) — the store can verify a presented key but never reveal it
again. A single `require_key` flag decides whether /v1 calls must present a
key at all; with the flag off the API is open and keys are optional.

Storage is a small JSON file next to the model manifest
(PROJECTS_HOME/api_keys.json) so it lives under the storage root with
everything else and survives restarts. Guarded by a process-wide lock;
gunicorn runs a single worker here, matching how job_store is handled.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from typing import Any, Optional

from .schemas import settings

_LOCK = threading.Lock()
_KEY_PREFIX = "hp_"


def _store_path() -> str:
    return os.path.join(os.path.dirname(settings.manifest_path), "api_keys.json")


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        return {"require_key": False, "keys": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"require_key": False, "keys": {}}
    data.setdefault("require_key", False)
    data.setdefault("keys", {})
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_api_key(name: str = "", pool: str = "") -> dict[str, Any]:
    """Mint a key. The returned dict includes the FULL key — the only time
    it is ever available; persist only its hash.

    ``pool`` binds the key to a dedicated worker pool: requests authenticated
    with it route to that pool by default (the app needs no per-request flag).
    """
    token = _KEY_PREFIX + secrets.token_hex(20)
    key_id = secrets.token_hex(8)
    record = {
        "id": key_id,
        "name": (name or "").strip() or "unnamed",
        "pool": (pool or "").strip(),
        "prefix": token[:11],
        "hash": _hash(token),
        "created_at": time.time(),
        "last_used": None,
        "revoked": False,
    }
    with _LOCK:
        data = _load()
        data["keys"][key_id] = record
        _save(data)
    public = {k: v for k, v in record.items() if k != "hash"}
    public["key"] = token
    return public


def list_api_keys() -> list[dict[str, Any]]:
    with _LOCK:
        data = _load()
    out = []
    for rec in data["keys"].values():
        if rec.get("revoked"):
            continue
        out.append({k: v for k, v in rec.items() if k != "hash"})
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


def revoke_api_key(key_id: str) -> bool:
    with _LOCK:
        data = _load()
        rec = data["keys"].get(key_id)
        if not rec:
            return False
        rec["revoked"] = True
        _save(data)
    return True


def verify_api_key(token: Optional[str]) -> bool:
    if not token:
        return False
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                rec["last_used"] = time.time()
                _save(data)
                return True
    return False


def pool_for_key(token: Optional[str]) -> Optional[str]:
    """The dedicated pool bound to this key (or None if no key / no pool / revoked).
    Used to route an app's traffic to its reserved workers from its key alone."""
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                p = (rec.get("pool") or "").strip()
                return p or None
    return None


def key_id_for_token(token: Optional[str]) -> Optional[str]:
    """The key record id behind a plaintext token (or None). Lets the
    principal layer (comms.principals) attribute a request to a stable key
    identity without ever storing the plaintext."""
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for key_id, rec in data["keys"].items():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                return key_id
    return None


def key_name_for_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                return rec.get("name") or None
    return None


def api_key_required() -> bool:
    with _LOCK:
        return bool(_load().get("require_key"))


def set_api_key_required(required: bool) -> bool:
    with _LOCK:
        data = _load()
        data["require_key"] = bool(required)
        _save(data)
    return bool(required)


# Media-intelligence access gate — a SEPARATE flag from `require_key` (the /v1
# gate) and from the console's own login. When on, the media-intelligence /ml/*
# inference endpoints require a valid Bearer key (the same keys minted above),
# letting the operator flip media between "open within the console" and
# "key-gated" without touching the console auth.
def media_key_required() -> bool:
    with _LOCK:
        return bool(_load().get("media_require_key"))


def set_media_key_required(required: bool) -> bool:
    with _LOCK:
        data = _load()
        data["media_require_key"] = bool(required)
        _save(data)
    return bool(required)
