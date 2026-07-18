"""Model BLOCK list — the operator's model-level serving-pool primitive.

WHAT
----
An operator can **block** a ``model_key`` (and unblock it). A blocked model is
removed from the SERVING POOL: central never routes to it, never
designates/assigns it, never warms/provisions/transfers it on a sweep, and never
resolves it through a fallback ladder (incl. the agent-brain default). Its files
stay on disk untouched and its existing designation rows stay VISIBLE (inert +
labeled) — block is a ROUTING/pool override, engine-agnostic (a sibling of the
WORKER-level block verb), reversible with one click.

Block outranks pin at routing time: pin is routing *persistence* (the allocation
survives restarts), block is an operator *override*. Blocking does NOT
auto-unassign and does NOT fight the pin's unassign-409 — the designation stays
recorded, routing simply refuses to use it.

SCOPE (today)
-------------
GLOBAL only: a blocked model is blocked EVERYWHERE. The per-worker case is
already covered by *unassign*; global is the missing primitive. The stored value
is an extensible dict (``{blocked, by, ts, note?}``) so a per-worker ``scope``
field can be added later without a migration.

PERSISTENCE
-----------
The F4 runtime settings store (``comms.settings.settings_store``), namespace
``models.blocked`` keyed by ``model_key``. That is the runtime source-of-truth
idiom the console's control plane already uses (fcntl-locked read-modify-write,
atomic replace, a short read cache) — it survives a central restart and rides the
existing operator-gated ``/settings`` surface. No new storage mechanism is
introduced here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .settings import settings_store

logger = logging.getLogger(__name__)

# Settings namespace for the block registry (model_key -> {blocked, by, ts, note?}).
NS = "models.blocked"

# The canonical honest-refusal phrase. Kept here so the routing gate, the
# no-worker diagnostic, and remote.py's _PERMANENT_LOAD_MARKERS agree on ONE
# distinct string (never softened / reused for an unrelated reason).
BLOCKED_MARKER = "blocked from the serving pool"


def _record(model_key: str) -> dict:
    """The raw stored record for ``model_key`` (``{}`` when never blocked).

    Normalizes legacy/degenerate shapes into a dict so ``is_blocked`` has one
    truth: a bare truthy scalar is treated as ``{"blocked": True}``.
    """
    try:
        v = settings_store.get(NS, str(model_key))
    except Exception as exc:  # noqa: BLE001 — a read must never break a caller
        logger.warning("blocklist read failed for %s: %s", model_key, exc)
        return {}
    if isinstance(v, dict):
        return v
    return {"blocked": True} if v else {}


def is_blocked(model_key: Optional[str]) -> bool:
    """True iff ``model_key`` is currently blocked from the serving pool."""
    if not model_key:
        return False
    return bool(_record(model_key).get("blocked"))


def block_info(model_key: Optional[str]) -> Optional[dict]:
    """The block record (``{blocked, by, ts, note?}``) when blocked, else None."""
    if not model_key:
        return None
    rec = _record(model_key)
    return dict(rec) if rec.get("blocked") else None


def blocked_keys() -> set:
    """The set of all currently-blocked model_keys."""
    try:
        allv = settings_store.all(NS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("blocklist enumerate failed: %s", exc)
        return set()
    out = set()
    for k, v in (allv or {}).items():
        if (isinstance(v, dict) and v.get("blocked")) or (not isinstance(v, dict) and v):
            out.add(k)
    return out


def block_reason(model_key: Optional[str]) -> Optional[str]:
    """The honest refusal reason when ``model_key`` is blocked, else None.

    Distinct, operator-authored, and NEVER reused for an unrelated cause — the
    string carries ``BLOCKED_MARKER`` so the cold-hold classifier in
    resolvers.remote treats it as a PERMANENT refusal (fail fast, no retry)."""
    if not is_blocked(model_key):
        return None
    return (f"'{model_key}' is {BLOCKED_MARKER} by the operator "
            f"— unblock it to route to it again")


def block(model_key: str, *, by: Optional[str] = None,
          note: Optional[str] = None) -> dict:
    """Block ``model_key`` from the serving pool (idempotent). Returns the record."""
    rec: dict[str, Any] = {"blocked": True, "by": (by or "operator"),
                           "ts": time.time()}
    if note:
        rec["note"] = str(note)
    settings_store.set(NS, str(model_key), rec)
    logger.info("model BLOCKED from serving pool: %s (by=%s)", model_key, rec["by"])
    return rec


def unblock(model_key: str) -> bool:
    """Unblock ``model_key`` (delete its record). Returns whether it was blocked."""
    existed = settings_store.delete(NS, str(model_key))
    if existed:
        logger.info("model UNBLOCKED (returned to serving pool): %s", model_key)
    return existed
