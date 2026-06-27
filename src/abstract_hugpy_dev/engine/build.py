"""Build llama.cpp from source via cmake — the fallback when no prebuilt asset
matches (exotic arch, or a CUDA/RPC build the release channel doesn't carry).

Requires ``git`` and ``cmake`` on PATH (and a CUDA toolkit for ``cuda=True``).
Clones into ``engine_dir()/src`` and builds into ``engine_dir()/build`` so the
resolver's ``build/bin`` search picks up ``llama-server`` / ``rpc-server``.

``-DGGML_RPC=ON`` is always set so the cross-machine shard backend
(``rpc-server``) is produced — the worker fleet needs it. This mirrors the build
hint the worker agent already prints (``cmake -DGGML_CUDA=on -DGGML_RPC=ON``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from .._platform import env_value
from .._platform.binaries import resolve_bin
from .._platform.paths import engine_dir
from . import resolve

_DEFAULT_GIT_URL = "https://github.com/ggml-org/llama.cpp.git"


def _require(tool: str) -> str:
    path = resolve_bin(tool)
    if not path:
        raise RuntimeError(f"`{tool}` not found on PATH — needed to build the engine from source.")
    return path


def build_from_source(*, cuda: bool = False, tag: Optional[str] = None,
                      git_url: Optional[str] = None, jobs: Optional[int] = None) -> dict:
    git = _require("git")
    cmake = _require("cmake")
    git_url = git_url or env_value("HUGPY_ENGINE_GIT_URL") or _DEFAULT_GIT_URL
    tag = tag or env_value("HUGPY_ENGINE_TAG")

    root = engine_dir()
    src = os.path.join(root, "src")
    build = os.path.join(root, "build")

    if not os.path.isdir(os.path.join(src, ".git")):
        if os.path.exists(src):
            shutil.rmtree(src, ignore_errors=True)
        subprocess.run([git, "clone", "--depth", "1",
                        *(["--branch", tag] if tag else []), git_url, src], check=True)
    else:
        subprocess.run([git, "-C", src, "fetch", "--depth", "1", "origin",
                        *( [tag] if tag else [] )], check=False)

    configure = [cmake, "-S", src, "-B", build, "-DLLAMA_BUILD_SERVER=ON", "-DGGML_RPC=ON"]
    if cuda:
        configure.append("-DGGML_CUDA=on")
    subprocess.run(configure, check=True)

    build_cmd = [cmake, "--build", build, "--config", "Release",
                 "--target", "llama-server", "rpc-server"]
    if jobs:
        build_cmd += ["-j", str(jobs)]
    subprocess.run(build_cmd, check=True)

    server = resolve.server_bin()
    if not server:
        raise RuntimeError(f"build completed but no llama-server found under {build}")
    return {
        "note": "built from source",
        "engine_dir": root,
        "server_bin": server,
        "rpc_bin": resolve.rpc_bin(),
        "cli_bin": resolve.cli_bin(),
    }
