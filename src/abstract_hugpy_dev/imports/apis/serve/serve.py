"""Deprecated location — re-export of the canonical serve module.

This used to be a near-byte-for-byte duplicate of
``hugpy.managers.serve.serve`` carrying its own (Linux-only, hardcoded)
``LLAMA_CPP_DIR``/systemd defaults. Those have been unified and made
cross-platform in the canonical module; this shim re-exports it so any lingering
import path keeps working while there is exactly one implementation.
"""
from ....managers.serve.serve import *  # noqa: F401,F403
