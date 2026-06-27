"""Process-wide registry of in-flight chat requests, for the console's live
queue view.

Generation is serialized (per-model generate_lock / max_concurrent_generations),
so at any moment some requests are GENERATING and others are WAITING for the
slot. This tracks that lifecycle thread-safely (one gunicorn process, many
threads) and exposes a snapshot the UI polls via ``GET /api/llm/queue``.

States: ``waiting`` (accepted, no token yet — queued or provisioning) →
``active`` (producing tokens). Entries are removed when the stream ends (success,
error, or client disconnect — the stream's ``finally`` always calls ``end``).
"""
from __future__ import annotations

import time
import threading

_lock = threading.Lock()
_entries: "dict[str, dict]" = {}


def begin(request_id, model_key=None, model_name=None, kind="chat") -> None:
    if not request_id:
        return
    with _lock:
        # Preserve an existing started_at if the same id re-registers.
        prior = _entries.get(request_id)
        _entries[request_id] = {
            "request_id": request_id,
            "model_key": model_key,
            "model": model_name or model_key or "?",
            "kind": kind,
            "state": "waiting",
            "started_at": prior["started_at"] if prior else time.time(),
            "first_token_at": None,
            "tokens": 0,
        }


def mark_active(request_id) -> None:
    with _lock:
        e = _entries.get(request_id)
        if e and e["state"] != "active":
            e["state"] = "active"
            e["first_token_at"] = time.time()


def on_token(request_id) -> None:
    """First token flips the entry active; every token bumps the counter. One
    lock acquisition per token (cheap; only a few streams run at once)."""
    with _lock:
        e = _entries.get(request_id)
        if not e:
            return
        if e["state"] != "active":
            e["state"] = "active"
            e["first_token_at"] = time.time()
        e["tokens"] += 1


def end(request_id) -> None:
    if not request_id:
        return
    with _lock:
        _entries.pop(request_id, None)


def snapshot() -> list:
    """Public, JSON-safe view. Waiting first, then longest-running."""
    now = time.time()
    with _lock:
        out = []
        for e in _entries.values():
            started = e["started_at"]
            ft = e["first_token_at"]
            out.append({
                "request_id": e["request_id"],
                "model_key": e["model_key"],
                "model": e["model"],
                "kind": e["kind"],
                "state": e["state"],
                "elapsed": round(now - started, 1),
                # seconds spent WAITING before the first token (queue time).
                "wait": round((ft or now) - started, 1),
                "tokens": e["tokens"],
            })
    out.sort(key=lambda x: (x["state"] != "waiting", -x["elapsed"]))
    return out


def counts() -> dict:
    with _lock:
        waiting = sum(1 for e in _entries.values() if e["state"] == "waiting")
        active = sum(1 for e in _entries.values() if e["state"] == "active")
    return {"waiting": waiting, "active": active, "total": waiting + active}
