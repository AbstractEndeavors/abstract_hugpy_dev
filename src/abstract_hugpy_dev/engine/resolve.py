"""Locate the native llama.cpp executables — one resolver for the whole package.

Replaces the per-module ``LLAMA_CPP_DIR`` / ``LLAMA_SERVER_BIN`` defaults that
used to disagree (``/srv/abstractendeavors/...`` vs ``~/.local/share/hugpy/...``).
Resolution order, for each of ``llama-server`` / ``rpc-server`` / ``llama-cli``:

    1. explicit env override (e.g. ``LLAMA_SERVER_BIN``)
    2. an engine fetched/built into ``paths.engine_dir()`` (incl. a ``build/bin``
       subdir, matching the cmake layout)
    3. ``PATH`` (``shutil.which``, ``.exe`` on Windows)

Returns ``None`` when absent — callers fall back to the in-process runner and/or
tell the user to run ``hugpy install-engine``.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .._platform import env_value
from .._platform.binaries import resolve_bin
from .._platform.paths import engine_dir


def _engine_search_dirs() -> List[str]:
    root = engine_dir()
    # Prebuilt release zips unpack their binaries at the top level or under bin/;
    # a from-source cmake build puts them in build/bin/. Search all three.
    return [
        root,
        os.path.join(root, "bin"),
        os.path.join(root, "build", "bin"),
    ]


def _resolve(name: str, *env_keys: str) -> Optional[str]:
    for key in env_keys:
        override = env_value(key)
        if override and os.path.isfile(override):
            return override
    return resolve_bin(name, extra_dirs=_engine_search_dirs())


def server_bin() -> Optional[str]:
    """Path to ``llama-server`` (the HTTP inference server), or ``None``."""
    return _resolve("llama-server", "LLAMA_SERVER_BIN")


def rpc_bin() -> Optional[str]:
    """Path to ``rpc-server`` (the cross-machine shard backend), or ``None``."""
    return _resolve("rpc-server", "WORKER_RPC_BIN", "LLAMA_RPC_BIN")


def cli_bin() -> Optional[str]:
    """Path to ``llama-cli``, or ``None``."""
    return _resolve("llama-cli", "LLAMA_CLI_BIN")


def have_native_engine() -> bool:
    return server_bin() is not None


# --------------------------------------------------------------------------- #
# Shared-library path for spawned native binaries                             #
# --------------------------------------------------------------------------- #
# A from-source ``llama-server`` links against sibling ``.so`` files (libllama,
# libggml, libggml-cuda …) that live NEXT to the binary — not in a system lib
# dir. When the agent spawns a slot child (or a native --mmproj/--rpc server)
# the child must find those on its loader path or it dies with
# "libggml.so: cannot open shared object file". On ae (2026-07-06) this was
# patched by hand as a unit-level LD_LIBRARY_PATH; deriving it from the engine
# dir in code makes the fix travel with the package instead.
def engine_lib_dirs() -> List[str]:
    """Existing directories that may hold the native llama.cpp shared libs,
    derived from the engine dir (``HUGPY_ENGINE_DIR``/``LLAMA_CPP_DIR``).

    Returns ``[]`` when no engine-dir override is set — we do NOT guess the
    per-OS default, so a box with a system/PATH llama-server (its libs already
    resolvable) is left untouched. Only dirs that exist are returned.
    """
    if not (env_value("HUGPY_ENGINE_DIR") or env_value("LLAMA_CPP_DIR")):
        return []
    root = engine_dir()
    # The engine dir itself + the usual binary/lib locations: a prebuilt release
    # zip unpacks .so at the top level or under lib/; a from-source cmake build
    # co-locates them with the binaries under build/bin (and sometimes build/lib).
    cands = [
        root,
        os.path.join(root, "lib"),
        os.path.join(root, "bin"),
        os.path.join(root, "build", "bin"),
        os.path.join(root, "build", "lib"),
    ]
    # Any other ``lib`` dir shallowly under root or build/ (cmake variants).
    for base in (root, os.path.join(root, "build")):
        try:
            for name in os.listdir(base):
                cands.append(os.path.join(base, name, "lib"))
        except OSError:
            pass
    out: List[str] = []
    seen = set()
    for d in cands:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def ld_library_path_with_engine(current: Optional[str] = None) -> Optional[str]:
    """Prepend the engine lib dirs (:func:`engine_lib_dirs`) to an
    ``LD_LIBRARY_PATH`` value, skipping any already present.

    Returns ``current`` unchanged when there is nothing to add (no engine-dir
    override, dirs already present, or non-Linux where LD_LIBRARY_PATH is inert).
    None-safe: a ``None`` input with dirs to add yields just the new dirs joined.
    """
    import sys
    if not sys.platform.startswith("linux"):
        return current
    dirs = engine_lib_dirs()
    if not dirs:
        return current
    parts = [p for p in (current or "").split(os.pathsep) if p]
    have = set(parts)
    new = [d for d in dirs if d not in have]
    if not new:
        return current
    return os.pathsep.join(new + parts)
