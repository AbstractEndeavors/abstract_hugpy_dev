"""Portable process spawn / terminate / re-exec.

The Linux build leaned on POSIX-only primitives:

  * ``subprocess.Popen(..., start_new_session=True)`` to detach a child so it
    survives a gunicorn worker reload — Windows rejects that kwarg.
  * ``os.killpg(os.getpgid(pid), SIGTERM)`` to take down a child and anything it
    spawned — ``os.getpgid``/``killpg`` don't exist on Windows.
  * ``os.execv(sys.executable, ...)`` to restart in place — no Windows analogue.

This module wraps each so call sites stay one-liners and the OS branch lives in
exactly one place.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Optional, Sequence

from . import IS_WINDOWS


def popen_detached(argv: Sequence[str], **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` that starts in its own session/process group.

    On POSIX the child gets a new session (``start_new_session=True``) so it
    outlives a parent reload and can later be torn down as a group. On Windows we
    use ``CREATE_NEW_PROCESS_GROUP`` (the closest analogue) instead of the
    POSIX-only kwarg, which would otherwise raise ``ValueError``.
    """
    if IS_WINDOWS:
        flags = kwargs.pop("creationflags", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(list(argv), **kwargs)


def terminate_tree(proc, sig: int = signal.SIGTERM) -> None:
    """Terminate ``proc`` and the children it spawned, best-effort.

    Accepts a :class:`subprocess.Popen` or a :class:`multiprocessing.Process`
    (anything exposing ``pid`` and, ideally, ``terminate``). Never raises if the
    process is already gone or we lack permission.

    POSIX: signal the whole process group (matches the old
    ``killpg(getpgid(pid), SIGTERM)`` behaviour). Windows: ``taskkill /T`` to walk
    the child tree, falling back to ``proc.terminate()``.
    """
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            return
        except Exception:
            pass
        _safe_terminate(proc)
        return
    # POSIX: try the process group first, then the bare pid.
    try:
        os.killpg(os.getpgid(pid), sig)
        return
    except (ProcessLookupError, PermissionError):
        return
    except (AttributeError, OSError):
        pass
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _safe_terminate(proc) -> None:
    term = getattr(proc, "terminate", None)
    if callable(term):
        try:
            term()
        except Exception:
            pass


def reexec(argv: Optional[Sequence[str]] = None) -> "int | None":
    """Restart the current interpreter with ``argv`` (default: this process).

    POSIX replaces the image via ``os.execv`` and never returns. Windows has no
    exec; we spawn a fresh process and exit, which is what callers (the worker
    self-update) want anyway. Returns nothing on POSIX; calls ``sys.exit`` on
    Windows.
    """
    args = list(argv) if argv is not None else ([sys.executable] + sys.argv)
    if IS_WINDOWS:
        proc = subprocess.Popen(args)
        sys.exit(proc.wait())
    os.execv(args[0], args)
