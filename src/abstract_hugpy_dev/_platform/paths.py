"""Per-OS application directories — one source of truth.

Replaces the scattered hardcoded ``/srv/abstractendeavors/...``,
``~/.local/share/hugpy``, ``/etc/llama-swap``, and ``/mnt/llm_storage`` literals.
Every path is overridable by the same env vars the code already honoured, so
existing Linux deployments are unaffected; only the *defaults* become per-OS:

    data_dir()    Linux ~/.local/share/hugpy   macOS ~/Library/Application Support/hugpy   Windows %LOCALAPPDATA%\\hugpy
    config_dir()  Linux ~/.config/hugpy        macOS ~/Library/Application Support/hugpy   Windows %LOCALAPPDATA%\\hugpy
    cache_dir()   Linux ~/.cache/hugpy         macOS ~/Library/Caches/hugpy                Windows %LOCALAPPDATA%\\hugpy\\Cache
    engine_dir()  data_dir()/engine            — where the fetched llama.cpp binary lands
    models_root() DEFAULT_ROOT or data_dir()/llm_storage

We use ``platformdirs`` when available (added to base deps) and fall back to a
hand-rolled per-OS layout so this module never hard-fails on import.
"""
from __future__ import annotations

import os

from . import IS_MACOS, IS_WINDOWS, env_value

_APP = "hugpy"


def _home(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def _fallback_data() -> str:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, _APP)
    if IS_MACOS:
        return _home("Library", "Application Support", _APP)
    return os.path.join(os.environ.get("XDG_DATA_HOME") or _home(".local", "share"), _APP)


def _fallback_config() -> str:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, _APP)
    if IS_MACOS:
        return _home("Library", "Application Support", _APP)
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or _home(".config"), _APP)


def _fallback_cache() -> str:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, _APP, "Cache")
    if IS_MACOS:
        return _home("Library", "Caches", _APP)
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or _home(".cache"), _APP)


def _dirs():
    try:
        import platformdirs

        return platformdirs.PlatformDirs(_APP, appauthor=False)
    except Exception:
        return None


def data_dir() -> str:
    override = env_value("HUGPY_DATA_DIR")
    if override:
        return _ensure(override)
    d = _dirs()
    return _ensure(d.user_data_dir if d else _fallback_data())


def config_dir() -> str:
    override = env_value("HUGPY_CONFIG_DIR")
    if override:
        return _ensure(override)
    d = _dirs()
    return _ensure(d.user_config_dir if d else _fallback_config())


def cache_dir() -> str:
    override = env_value("HUGPY_CACHE_DIR")
    if override:
        return _ensure(override)
    d = _dirs()
    return _ensure(d.user_cache_dir if d else _fallback_cache())


def engine_dir() -> str:
    """Where ``hugpy install-engine`` unpacks the native llama.cpp binaries."""
    override = env_value("HUGPY_ENGINE_DIR") or env_value("LLAMA_CPP_DIR")
    if override:
        return _ensure(override)
    return _ensure(os.path.join(data_dir(), "engine"))


def models_root() -> str:
    """Model/upload/dataset storage root.

    Honours the legacy ``DEFAULT_ROOT``/``MODELS_HOME`` env vars first — but only
    if that path can actually be created and written. A stale/un-writable override
    (e.g. ``DEFAULT_ROOT=/mnt/llm_storage`` carried in a server ``.env`` onto a
    worker or a phone where ``/mnt`` is read-only) is ignored in favour of a
    per-user dir under ``data_dir()``, so storage never lands on a dead path.
    """
    override = env_value("DEFAULT_ROOT")
    if override and _usable(override):
        return override
    # Preserve the historical Linux mount when it exists and is writable.
    legacy = "/mnt/llm_storage"
    try:
        if os.path.isdir(legacy) and os.access(legacy, os.W_OK):
            return legacy
    except OSError:
        pass
    return _ensure(os.path.join(data_dir(), "llm_storage"))


def _usable(path: str) -> bool:
    """True only if *path* exists (or can be created) AND is writable."""
    try:
        os.makedirs(path, exist_ok=True)
        return os.path.isdir(path) and os.access(path, os.W_OK)
    except OSError:
        return False


def _ensure(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path
