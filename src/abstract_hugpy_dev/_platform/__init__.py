"""Cross-platform foundation for hugpy.

hugpy began life as a Linux-only service and grew a lot of POSIX assumptions:
systemd units, ``/srv`` and ``/etc`` paths, ``nvidia-smi`` + ``/proc/meminfo``
probing, ``os.killpg`` process groups. This package is the seam that lets the
same code run on Windows, macOS, and Linux:

    paths     — per-OS data/config/cache/engine dirs (XDG / AppData / Library)
    binaries  — resolve an executable name to a path, ``.exe`` on Windows
    procutil  — spawn detached, terminate a process tree, re-exec — portably
    hardware  — RAM + GPU probes that degrade to ``None`` off Linux/NVIDIA

Everything honours the existing env-var overrides (via ``env_value`` here, which
prefers ``abstract_security.get_env_value`` and falls back to ``os.environ``) so
deployments that already set ``LLAMA_CPP_DIR``/``DEFAULT_ROOT``/etc. keep working.
"""
from __future__ import annotations

import os
import sys

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


def env_value(name: str):
    """Resolve a config override the way the rest of hugpy does.

    Prefer ``abstract_security.get_env_value`` (which can read the project's
    secrets store), falling back to the process environment. Returns ``None``
    when unset/empty so callers can ``or`` in a default.
    """
    val = None
    try:
        from abstract_hugpy_dev.imports.src.standalone_utils import get_env_value

        val = get_env_value(name)
    except Exception:
        val = None
    if val is None:
        val = os.environ.get(name)
    if isinstance(val, str):
        val = val.strip()
        return val or None
    return val
