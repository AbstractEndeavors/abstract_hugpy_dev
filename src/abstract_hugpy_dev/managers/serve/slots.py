"""Slot scheduler — assigns models to the root-free slot supervisors.

The pool is a fixed set of slot control URLs (``SLOT_COUNT`` slots starting at
``SLOT_PORT_BASE``). On demand it routes a model to a slot:

    1. a slot already serving that model      -> reuse it
    2. an idle slot                            -> load the model there (the slot
                                                  autofits GPU layers from the
                                                  VRAM still free, so slots fill
                                                  the card in order)
    3. all slots busy                          -> return None; the caller routes
                                                  the overflow through swap

Pure HTTP to the supervisors — no systemctl, no root.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("abstract_hugpy_dev.slots")


_LOGGED_COUNT = False

# Tiers-v2 eviction policy hook: a callable mk -> bool ("is this model
# on-demand, i.e. may it be bumped from its slot for a caller?"). Registered
# by the WORKER agent from its runtime settings; None (e.g. bare central)
# keeps the historical behavior: all-busy -> None -> swap fallback.
_EVICTION_POLICY = None


def set_eviction_policy(fn) -> None:
    global _EVICTION_POLICY
    _EVICTION_POLICY = fn


def _slot_count() -> int:
    global _LOGGED_COUNT
    raw = os.environ.get("SLOT_COUNT")
    try:
        n = max(0, int(raw)) if raw is not None else 2
    except ValueError:
        n = 2
    if not _LOGGED_COUNT:
        _LOGGED_COUNT = True
        # De-silence the env-layering ghost: a systemd drop-in can override the
        # unit's SLOT_COUNT and silently resurrect slots every restart (op's
        # limits.conf SLOT_COUNT=2). Record the EFFECTIVE value + raw env once
        # per process so the journal shows the truth.
        logger.info("SLOT_COUNT effective=%d (env=%r) — if this differs from the "
                    "unit file, a drop-in is overriding it", n, raw)
    return n


def slots_enabled() -> bool:
    return _slot_count() > 0


def _slot_host() -> str:
    return os.environ.get("SLOT_ADVERTISE") or os.environ.get("SLOT_HOST_ADDR") or "127.0.0.1"


def _slot_port_base() -> int:
    try:
        return int(os.environ.get("SLOT_PORT_BASE", "8101"))
    except ValueError:
        return 8101


def slot_urls() -> list[str]:
    host, base = _slot_host(), _slot_port_base()
    return [f"http://{host}:{base + i}" for i in range(_slot_count())]


def _get(url: str, timeout: float = 3.0) -> dict:
    import httpx
    return httpx.get(url, timeout=timeout).json()


def _post(url: str, body: dict, timeout: float) -> dict:
    import httpx
    return httpx.post(url, json=body, timeout=timeout).json()


class SlotPool:
    def __init__(self, urls: list[str] | None = None):
        self.urls = urls if urls is not None else slot_urls()

    def statuses(self) -> list[dict]:
        out = []
        for url in self.urls:
            try:
                status = _get(url + "/status")
                status["_control"] = url
            except Exception as exc:  # a down slot shouldn't break scheduling
                status = {"_control": url, "healthy": False, "model_key": None,
                          "error": str(exc)}
            out.append(status)
        return out

    def endpoint_for(self, model_key: str, *, load_timeout: float = 900.0,
                     opts: dict | None = None) -> str | None:
        """Resolve (and if needed load) the slot serving ``model_key``.

        ``opts`` is an optional per-load compute dict (n_gpu_layers, ctx,
        threads, cpus, gpu) applied when the model is loaded into a free slot.
        Returns the slot's inference endpoint, or None when every slot is busy
        with a different model (caller should fall back to swap).
        """
        statuses = self.statuses()

        # 1. already serving OR currently loading it — reuse, never load a 2nd
        #    copy. A slot mid-load has model_key set but healthy=False; if we
        #    returned that endpoint immediately the caller would proxy to a child
        #    that isn't up yet and get a 503. So WAIT for the loading slot to go
        #    healthy (coalescing concurrent requests onto the one load), just like
        #    loading into a fresh slot blocks. A down slot reports model_key=None.
        for s in statuses:
            if s.get("model_key") != model_key:
                continue
            ep = s.get("endpoint") or s["_control"]
            if s.get("healthy"):
                return ep
            deadline = time.time() + load_timeout
            while time.time() < deadline:
                time.sleep(2.0)
                try:
                    st = _get(s["_control"] + "/status")
                except Exception:
                    break                       # slot went away — reload below
                if st.get("healthy"):
                    return st.get("endpoint") or ep
                if not st.get("model_key"):
                    break                       # load aborted/failed — reload below
            break                               # not ready in time — fall through

        # 2. an idle slot (reachable, nothing loaded)
        for s in statuses:
            if "error" in s:
                continue
            if not s.get("model_key"):
                body = {"model_key": model_key, **(opts or {})}
                resp = _post(s["_control"] + "/load", body, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(f"slot load failed: {resp['error']}")
                return resp.get("endpoint") or s["_control"]

        # 3. everything busy — tiers-v2 promotion: bump the LRU *idle*
        #    occupant whose model is itself on-demand (policy hook). Never a
        #    busy slot, never a serving/pinned model (the policy returns False
        #    for those), never for a model already handled above.
        if _EVICTION_POLICY is not None:
            candidates = [s for s in statuses
                          if s.get("model_key") and s.get("healthy")
                          and not s.get("busy")
                          and s.get("model_key") != model_key]
            try:
                candidates = [s for s in candidates
                              if _EVICTION_POLICY(s["model_key"])]
            except Exception:  # noqa: BLE001 — a broken policy must not crash serving
                candidates = []
            candidates.sort(key=lambda s: s.get("last_used") or 0)
            for victim in candidates:
                logger.info(
                    "slot promotion: evicting idle on-demand %s from %s to "
                    "load %s", victim["model_key"], victim["_control"], model_key)
                try:
                    self.unload(victim["_control"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("promotion evict failed on %s: %s",
                                   victim["_control"], exc)
                    continue
                body = {"model_key": model_key, **(opts or {})}
                resp = _post(victim["_control"] + "/load", body, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(f"slot load failed: {resp['error']}")
                return resp.get("endpoint") or victim["_control"]
        return None

    def load(self, model_key: str, control_url: str, *, timeout: float = 900.0) -> dict:
        return _post(control_url + "/load", {"model_key": model_key}, timeout)

    def unload(self, control_url: str) -> dict:
        return _post(control_url + "/unload", {}, 30.0)

    def overview(self) -> list[dict]:
        return self.statuses()


# --------------------------------------------------------------------------- #
# one-time install: N generic slot services (systemd template)                #
# --------------------------------------------------------------------------- #
def render_slot_unit(python_bin: str | None = None, user: str | None = None,
                     group: str | None = None, main_gpu: str | None = None) -> str:
    """A systemd TEMPLATE unit: ``abstract-hugpy-slot@.service``.

    The instance number is the slot id (``systemctl enable --now
    abstract-hugpy-slot@1``); the agent derives its port from SLOT_ID. Installed
    ONCE; thereafter the app drives slots over HTTP — no per-model units, no
    sudo at request time.
    """
    import sys
    python_bin = python_bin or sys.executable
    user = user or os.environ.get("LLAMA_SERVICE_USER", "solcatcher")
    group = group or os.environ.get("LLAMA_SERVICE_GROUP", "web")
    env_lines = [
        "Environment=SLOT_ID=%i",
        f"Environment=SLOT_PORT_BASE={_slot_port_base()}",
    ]
    if main_gpu is not None:
        env_lines.append(f"Environment=MAIN_GPU={main_gpu}")
    return "\n".join((
        "[Unit]",
        "Description=abstract_hugpy_dev model slot %i",
        "After=network.target",
        "StartLimitIntervalSec=120",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "Type=simple",
        f"User={user}",
        f"Group={group}",
        *env_lines,
        f"ExecStart={python_bin} -m abstract_hugpy_dev.managers.serve.slot_agent",
        "Restart=always",
        "RestartSec=5",
        "TimeoutStopSec=60",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ))


def slot_install_steps(unit_dir: str = "/etc/systemd/system",
                       main_gpu: str | None = None) -> list[tuple[str, str]]:
    """Return [(kind, payload)] describing the one-time install:

    ('write', unit_text) for the template, then ('cmd', shell) lines to enable
    each slot. The caller writes/executes (with sudo); this stays side-effect
    free so it can be shown as a dry run.
    """
    import os.path as osp
    unit_path = osp.join(unit_dir, "abstract-hugpy-slot@.service")
    steps = [("write:" + unit_path, render_slot_unit(main_gpu=main_gpu)),
             ("cmd", "systemctl daemon-reload")]
    for i in range(_slot_count()):
        steps.append(("cmd", f"systemctl enable --now abstract-hugpy-slot@{i + 1}"))
    return steps
