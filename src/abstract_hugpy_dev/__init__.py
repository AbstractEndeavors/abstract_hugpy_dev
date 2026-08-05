# Must run before anything imports pydantic: on platforms without pydantic_core
# (e.g. Termux/Android, where it has no wheel and needs Rust to build), install a
# pure-Python pydantic shim so the package still imports. No-op where the real
# pydantic is available. See _compat_pydantic.py.
from ._compat_pydantic import ensure_pydantic as _ensure_pydantic
_ensure_pydantic()

# THE version number for the package. Single-sourced: pyproject.toml reads this
# attribute (`[tool.setuptools.dynamic] version = {attr = ...}`), so the wheel's
# metadata and the running module can never disagree. That desync is exactly what
# shipped in 0.1.224 — dist metadata 0.1.224, this literal 0.1.223 — which made
# every worker on the required version report version_ok:false forever.
# Running-source remains authoritative (2026-07-20 skew-honesty design).
# Exposed over HTTP at GET /version.
__version__ = "0.1.230"

from .imports import *
from .managers import *
from .utils import *
