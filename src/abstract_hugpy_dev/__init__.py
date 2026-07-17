# Must run before anything imports pydantic: on platforms without pydantic_core
# (e.g. Termux/Android, where it has no wheel and needs Rust to build), install a
# pure-Python pydantic shim so the package still imports. No-op where the real
# pydantic is available. See _compat_pydantic.py.
from ._compat_pydantic import ensure_pydantic as _ensure_pydantic
_ensure_pydantic()

# Running-source version (authoritative even when the installed metadata is
# stale, e.g. a run-from-source dev box). Keep in sync with pyproject `version`
# — keeper/stock_pip_index.sh stamps both. Exposed over HTTP at GET /version.
__version__ = "0.1.186"

from .imports import *
from .managers import *
from .utils import *
