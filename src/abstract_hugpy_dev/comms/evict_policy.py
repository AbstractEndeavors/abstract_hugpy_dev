"""FLEET-WIDE eviction policy — the knobs that must be the same everywhere.

WHY THIS EXISTS (and why it is not a per-worker setting)
--------------------------------------------------------
The eviction spec (``assets/evictionflow.html``, 2026-07-25) has Parity as its
first invariant: central's preview and the worker's auto-evict must name the
SAME victims. ``managers/eviction.py`` upholds that by being PURE — every input
is passed in — but purity only guarantees parity if the two sides pass the same
inputs.

``least_reaping`` gates the DROP PASS, and the drop pass runs inside
``evict_plan``, which BOTH sides call:

    central   flask_app/.../utils/workers.py  storage_proposal  (the preview)
    worker    worker_agent/budget.py          fit_plan          (the storage half)
    worker    worker_agent/agent.py           the VRAM auto-evict

If worker A had it ON and central previewed it OFF, the console would show the
operator one victim list and the fleet would execute another — exactly the
divergence ``tests/test_eviction_parity.py`` exists to catch. So this knob is
owned CENTRALLY, stored once, and shipped to every worker on the heartbeat
reply (the ``blocked_models`` idiom: additive, omit-when-default, so an older
worker simply ignores it).

CONTRAST — what is correctly PER-WORKER
---------------------------------------
``evict_min_residency_s`` (the anti-thrash floor) is NOT here, deliberately. It
is a VRAM-RESIDENCY concept, and central has no VRAM preview to diverge from:
its single eviction call site (``storage_proposal``) is the DISK and hardcodes
``min_residency_s=0.0``, as does the worker's own storage half
(``budget.fit_plan``) — both with a standing comment explaining that a freshly
DOWNLOADED file has no load clock. The floor therefore only ever affects the
worker's own VRAM auto-evict, where per-box hardware differences make a
per-worker value the right answer. It lives in the worker's own settings file.

PERSISTENCE
-----------
The F4 runtime settings store (``comms.settings.settings_store``), namespace
``fleet.evict`` — the same idiom ``comms.blocklist`` uses (fcntl-locked
read-modify-write, atomic replace, short read cache), reachable through the
already-operator-gated ``/settings`` surface. No new storage mechanism.
"""
from __future__ import annotations

import logging
from typing import Optional

from .settings import settings_store

logger = logging.getLogger(__name__)

# Settings namespace for fleet-wide eviction policy.
NS = "fleet.evict"

# The stored key for the drop-pass switch.
KEY_LEAST_REAPING = "least_reaping"


def least_reaping() -> bool:
    """The fleet's drop-pass policy — True == today's shipped behaviour.

    Falls back to ``managers.eviction.DEFAULT_LEAST_REAPING`` (True) when the
    operator has never set it, so an untouched fleet behaves exactly as it does
    today. A read must NEVER break a caller (a heartbeat runs through here), so
    a broken store degrades to the default rather than raising.
    """
    from ..managers.eviction import DEFAULT_LEAST_REAPING
    try:
        v = settings_store.get(NS, KEY_LEAST_REAPING)
    except Exception as exc:  # noqa: BLE001 — a read must never break a beat
        logger.warning("fleet evict-policy read failed: %s", exc)
        return DEFAULT_LEAST_REAPING
    if v is None:
        return DEFAULT_LEAST_REAPING
    if isinstance(v, bool):
        return v
    # Tolerate a scalar written through the generic /settings surface.
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def is_set() -> bool:
    """True iff the operator has explicitly set the policy (vs. defaulting).

    Drives the omit-when-default wire rule: central publishes the key on the
    heartbeat ONLY when it has a real opinion, so a worker with a local drop-in
    keeps it until an operator actually rules.
    """
    try:
        return settings_store.get(NS, KEY_LEAST_REAPING) is not None
    except Exception:  # noqa: BLE001
        return False


def set_least_reaping(value: Optional[bool], by: str = "operator") -> bool:
    """Set (or CLEAR, with ``None``) the fleet drop-pass policy.

    Returns the value now in force. Clearing reverts every worker to its own
    base on the next beat — central stops publishing the key, and
    ``agent._adopt_least_reaping`` restores the captured drop-in base.
    """
    if value is None:
        try:
            settings_store.delete(NS, KEY_LEAST_REAPING)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fleet evict-policy clear failed: %s", exc)
        return least_reaping()
    settings_store.set(NS, KEY_LEAST_REAPING, bool(value))
    logger.info("fleet eviction policy: least_reaping=%s (by %s)", bool(value), by)
    return bool(value)
