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
