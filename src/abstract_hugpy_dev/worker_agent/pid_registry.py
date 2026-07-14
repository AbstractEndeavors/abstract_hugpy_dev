"""Precision model->PID registry — the authoritative "worker-module -> pid log".

A process-local, thread-safe registry that tracks, per active model, WHICH OS
process holds it, HOW it's hosted, and — after a reconcile against nvidia-smi
ground truth — how much VRAM it occupies. This is the DATA/telemetry layer:

  * a separate ``evict <model_key>`` action reads it to find the exact PID to
    signal (built concurrently, not here), and
  * central's worker row displays the ``snapshot_for_heartbeat()`` log.

Design constraints honored here:

  * SELF-CONTAINED. This module imports nothing from ``agent.py`` (whose import
    pulls torch and the whole runner stack). It defines its own ``_MIB`` and its
    own ``/proc`` probe. The caller passes the nvidia-smi maps into
    ``reconcile`` — this module never shells out to nvidia-smi.

  * DEGRADE-TO-EMPTY. No records / empty inputs / no ``/proc`` (non-Linux) all
    yield empty attributions and an empty snapshot — never an exception. Same
    convention as ``agent._gpu_process_vram`` returning ``{}`` with no GPU.

  * RECYCLED-PID GUARD (the "precision" the operator asked for). A PID number is
    reused by the OS after a process exits. Before we trust a stored PID we
    re-check its IDENTITY: the Linux process START TIME (field 22 of
    ``/proc/<pid>/stat``, in clock ticks since boot) is a per-process-lifetime
    constant — a new process reusing the number has a DIFFERENT start time. So
    "same PID number, different start time" => the model's process is GONE and
    the number now belongs to a stranger => ``verify`` returns ``None``. The
    command line (``/proc/<pid>/cmdline``) is a secondary anchor used only when
    the start time couldn't be read at record time.

  * The ``/proc`` probe is INJECTABLE (constructor arg / ``set_proc_info_probe``)
    so every path is unit-testable with fake processes — no real subprocess or
    GPU needed.

Host modes (``host_mode``):
  * ``subprocess``  — a slot child (llama-server): PID known at spawn, its VRAM
                      is that PID's nvidia-smi ``mib``.
  * ``in_process``  — a torch model sharing the worker python PID: its VRAM is
                      the per-model torch split (nvidia-smi can't split the lump).
  * ``comfy``       — the external, adopted ComfyUI process: VRAM by process name.
  * ``external``    — an adopted foreign process the worker chose to track by PID.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

# Bytes per MiB — defined locally so this module stays dependency-free (importing
# it must never drag in torch / the runner stack via agent.py).
_MIB = 1024 * 1024

# Field index of ``starttime`` in /proc/<pid>/stat AFTER the "pid (comm)" prefix
# is split off. The stat line is: `pid (comm) state ppid ... starttime ...`;
# starttime is field 22 (1-indexed). Dropping fields 1-2 (pid, comm) leaves
# `state` at index 0, so starttime (field 22) sits at index 22 - 3 = 19.
_STAT_STARTTIME_IDX = 19

ProcInfo = Dict[str, object]        # {"starttime": int|None, "cmdline": str, "name": str}
ProcInfoProbe = Callable[[int], Optional[ProcInfo]]
HostMode = str                      # "subprocess" | "in_process" | "comfy" | "external"


def _default_proc_info(pid: int) -> Optional[ProcInfo]:
    """Read ``{starttime, cmdline, name}`` for ``pid`` from ``/proc``, or ``None``
    if the PID is gone / unreadable. Non-Linux (no ``/proc``) -> ``None`` for
    every PID, which makes ``verify`` degrade to "can't confirm -> don't trust".

    ``starttime`` is the recycled-PID anchor (see module docstring). ``comm`` and
    the full cmdline are captured as secondary identity hints.
    """
    if pid is None:
        return None
    try:
        with open("/proc/%d/stat" % int(pid), "rb") as fh:
            data = fh.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        return None
    # comm (field 2) is wrapped in parens and may itself contain spaces/parens,
    # so slice on the LAST ')' rather than naively splitting on whitespace.
    rparen = data.rfind(b")")
    lparen = data.find(b"(")
    starttime: Optional[int] = None
    name = ""
    if 0 <= lparen < rparen:
        name = data[lparen + 1:rparen].decode("utf-8", "replace")
        rest = data[rparen + 2:].split()
        if len(rest) > _STAT_STARTTIME_IDX:
            try:
                starttime = int(rest[_STAT_STARTTIME_IDX])
            except (ValueError, IndexError):
                starttime = None
    cmdline = ""
    try:
        with open("/proc/%d/cmdline" % int(pid), "rb") as fh:
            cmdline = fh.read().replace(b"\x00", b" ").strip().decode("utf-8", "replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        cmdline = ""
    return {"starttime": starttime, "cmdline": cmdline, "name": name}


class PidRegistry:
    """Thread-safe model_key -> process record store with a recycled-PID guard.

    A record is ``{model_key, pid, host_mode, launched_at, start_tick,
    cmdline_hint, last_vram_bytes, alive}``. ``start_tick`` is the process
    start time captured at record time — the identity anchor.
    """

    def __init__(self, proc_info: Optional[ProcInfoProbe] = None) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, dict] = {}
        self._last_unattributed: List[dict] = []
        self._proc_info: ProcInfoProbe = proc_info or _default_proc_info

    # -- probe injection (tests) ------------------------------------------------
    def set_proc_info_probe(self, probe: ProcInfoProbe) -> None:
        """Swap the ``/proc`` probe (tests inject fake processes)."""
        with self._lock:
            self._proc_info = probe

    # -- lifecycle --------------------------------------------------------------
    def record_launch(self, model_key: str, pid: Optional[int],
                      host_mode: HostMode, cmdline_hint: Optional[str] = None) -> dict:
        """Record (or refresh) that ``model_key`` is hosted by ``pid``.

        Captures the process start time NOW as the recycled-PID anchor. Idempotent
        for the heartbeat-driven population path: re-recording the SAME pid with
        the SAME start time preserves ``launched_at`` (so re-observing an already
        known child every beat doesn't reset its clock); a changed pid/start time
        replaces the record (a genuinely new launch, e.g. a reloaded slot).
        """
        info = self._proc_info(pid) if pid is not None else None
        start_tick = info.get("starttime") if info else None
        cmd = cmdline_hint or (info.get("cmdline") if info else None) or None
        with self._lock:
            prev = self._records.get(model_key)
            if (prev is not None and prev.get("pid") == pid
                    and prev.get("start_tick") == start_tick
                    and start_tick is not None):
                # Same live process re-observed — refresh nothing that would
                # perturb identity/age; just keep it.
                prev["cmdline_hint"] = prev.get("cmdline_hint") or cmd
                return dict(prev)
            rec = {
                "model_key": model_key,
                "pid": pid,
                "host_mode": host_mode,
                "launched_at": time.time(),
                "start_tick": start_tick,
                "cmdline_hint": cmd,
                "last_vram_bytes": None,
                "alive": pid is not None and (info is not None),
            }
            self._records[model_key] = rec
            return dict(rec)

    def forget(self, model_key: str) -> bool:
        """Drop ``model_key``'s record. Returns True if one was present."""
        with self._lock:
            return self._records.pop(model_key, None) is not None

    def sweep_dead(self) -> List[str]:
        """Drop every record whose process is gone OR whose PID was recycled.
        Returns the list of forgotten model_keys."""
        dropped: List[str] = []
        with self._lock:
            for mk in list(self._records.keys()):
                if self._verify_locked(mk) is None:
                    self._records.pop(mk, None)
                    dropped.append(mk)
        return dropped

    # -- identity ---------------------------------------------------------------
    def _verify_locked(self, model_key: str) -> Optional[int]:
        """``verify`` core; caller holds the lock."""
        rec = self._records.get(model_key)
        if rec is None:
            return None
        pid = rec.get("pid")
        if pid is None:
            return None
        info = self._proc_info(pid)
        if info is None:
            rec["alive"] = False        # process gone
            return None
        # RECYCLED-PID GUARD: start time is a per-lifetime constant. A mismatch
        # means the number was reused by an unrelated process => not ours.
        anchor = rec.get("start_tick")
        cur = info.get("starttime")
        if anchor is not None and cur is not None:
            if int(cur) != int(anchor):
                rec["alive"] = False
                return None
        elif anchor is None:
            # Start time unavailable at record time -> fall back to the cmdline
            # anchor. If we can't corroborate identity at all, don't trust it.
            hint = rec.get("cmdline_hint")
            cmd = (info.get("cmdline") or "") if info else ""
            if not hint or hint not in cmd:
                rec["alive"] = False
                return None
        rec["alive"] = True
        return pid

    def verify(self, model_key: str) -> Optional[int]:
        """Return ``model_key``'s PID ONLY if it's still alive AND still the same
        process we recorded (recycled-PID guard). Otherwise ``None``."""
        with self._lock:
            return self._verify_locked(model_key)

    # -- reconcile against nvidia-smi ground truth ------------------------------
    def reconcile(self, gpu_procs: Optional[dict],
                  inprocess_bytes: Optional[dict],
                  comfy_bytes: Optional[int]) -> dict:
        """Join the registry against nvidia-smi ground truth. PURE of its inputs
        (no ``/proc``, no nvidia-smi call) so it's directly unit-testable.

        Inputs mirror ``agent.py``:
          * ``gpu_procs``       -> ``{pid: {"name", "mib"}}`` (``_gpu_process_vram``)
          * ``inprocess_bytes`` -> ``{model_key: {"vram_bytes", "device"}}``
                                   (``_inprocess_gpu_bytes``)
          * ``comfy_bytes``     -> ``int`` bytes | ``None`` (``_comfy_process_vram``)

        Attribution rules:
          * ``subprocess``/``external`` -> the record's PID's ``mib`` (bytes);
            an alive child that isn't a GPU compute app attributes 0 (CPU).
          * ``in_process`` -> the per-model torch split; the shared worker-python
            PID lump is marked explained so it isn't flagged as a squatter.
          * ``comfy`` -> ``comfy_bytes``; comfyui-named GPU procs are explained.

        Any GPU proc the registry can't explain is surfaced as ``unattributed`` —
        this is how central learns about FOREIGN / ROGUE VRAM squatters.

        Returns ``{"attributed": {model_key: vram_bytes}, "unattributed":
        [{pid, name, mib}]}`` and stores ``last_vram_bytes`` per record +
        ``unattributed`` for the next ``snapshot_for_heartbeat``.
        """
        gpu_procs = gpu_procs or {}
        inprocess_bytes = inprocess_bytes or {}
        attributed: Dict[str, int] = {}
        explained: set = set()
        with self._lock:
            for rec in self._records.values():
                mk = rec["model_key"]
                mode = rec.get("host_mode")
                pid = rec.get("pid")
                if mode == "in_process":
                    ip = inprocess_bytes.get(mk) or {}
                    vb = int(ip.get("vram_bytes") or 0)
                    if pid is not None and pid in gpu_procs:
                        explained.add(pid)      # the shared worker-python lump
                elif mode == "comfy":
                    vb = int(comfy_bytes or 0)
                    for p, meta in gpu_procs.items():
                        if "comfyui" in (str(meta.get("name") or "")).lower():
                            explained.add(p)
                else:                           # subprocess / external
                    vb = 0
                    if pid is not None and pid in gpu_procs:
                        vb = int(gpu_procs[pid].get("mib") or 0) * _MIB
                        explained.add(pid)
                    elif pid is not None:
                        explained.add(pid)      # alive child, CPU-resident
                attributed[mk] = vb
                rec["last_vram_bytes"] = vb
            unattributed = [
                {"pid": p, "name": meta.get("name"), "mib": meta.get("mib")}
                for p, meta in gpu_procs.items() if p not in explained
            ]
            self._last_unattributed = unattributed
        return {"attributed": attributed, "unattributed": unattributed}

    # -- heartbeat telemetry ----------------------------------------------------
    def snapshot_for_heartbeat(self) -> dict:
        """The "worker-module-pid log" central displays.

        Returns ``{"models": [{model_key, pid, host_mode, vram_bytes, alive}],
        "unattributed": [{pid, name, mib}]}``. ``alive`` is a LIVE recycled-PID-
        guarded ``verify`` (not the last-cached flag); ``vram_bytes`` is the most
        recent ``reconcile`` attribution. Empty registry -> both lists empty
        (degrade-safe).

        NB: the contract sketch said "-> list"; a dict carrying BOTH the per-model
        log and the unattributed squatters is strictly better for the single
        heartbeat field (central needs both), so this returns one dict. The
        per-model log is under ``["models"]``.
        """
        with self._lock:
            models = []
            for mk in list(self._records.keys()):
                rec = self._records[mk]
                alive = self._verify_locked(mk) is not None
                models.append({
                    "model_key": mk,
                    "pid": rec.get("pid"),
                    "host_mode": rec.get("host_mode"),
                    "vram_bytes": rec.get("last_vram_bytes"),
                    "alive": alive,
                })
            return {"models": models, "unattributed": list(self._last_unattributed)}


# ── module-level default registry + free-function facade ─────────────────────
# The worker holds ONE registry; these delegate to it so callers use
# ``pid_registry.record_launch(...)`` etc. Tests that want isolation construct
# their own ``PidRegistry(proc_info=fake)``.
_REGISTRY = PidRegistry()


def registry() -> PidRegistry:
    """The process-wide default registry."""
    return _REGISTRY


def set_proc_info_probe(probe: ProcInfoProbe) -> None:
    _REGISTRY.set_proc_info_probe(probe)


def record_launch(model_key: str, pid: Optional[int], host_mode: HostMode,
                  cmdline_hint: Optional[str] = None) -> dict:
    return _REGISTRY.record_launch(model_key, pid, host_mode, cmdline_hint)


def forget(model_key: str) -> bool:
    return _REGISTRY.forget(model_key)


def sweep_dead() -> List[str]:
    return _REGISTRY.sweep_dead()


def verify(model_key: str) -> Optional[int]:
    return _REGISTRY.verify(model_key)


def reconcile(gpu_procs: Optional[dict], inprocess_bytes: Optional[dict],
              comfy_bytes: Optional[int]) -> dict:
    return _REGISTRY.reconcile(gpu_procs, inprocess_bytes, comfy_bytes)


def snapshot_for_heartbeat() -> dict:
    return _REGISTRY.snapshot_for_heartbeat()
