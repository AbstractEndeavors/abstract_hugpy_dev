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

logger = logging.getLogger("abstract_hugpy_dev.slots")


def _slot_count() -> int:
    try:
        return max(0, int(os.environ.get("SLOT_COUNT", "2")))
    except ValueError:
        return 2


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

    def endpoint_for(self, model_key: str, *, load_timeout: float = 900.0) -> str | None:
        """Resolve (and if needed load) the slot serving ``model_key``.

        Returns the slot's inference endpoint, or None when every slot is busy
        with a different model (caller should fall back to swap).
        """
        statuses = self.statuses()

        # 1. already serving it
        for s in statuses:
            if s.get("model_key") == model_key and s.get("healthy"):
                return s.get("endpoint") or s["_control"]

        # 2. an idle slot (reachable, nothing loaded)
        for s in statuses:
            if "error" in s:
                continue
            if not s.get("model_key"):
                resp = _post(s["_control"] + "/load", {"model_key": model_key}, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(f"slot load failed: {resp['error']}")
                return resp.get("endpoint") or s["_control"]

        # 3. everything busy
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
