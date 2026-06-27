"""Deprecated location — re-export of the canonical serve CLI.

The implementation now lives in ``hugpy.managers.serve.serve_cli``; this shim
keeps the old import path valid.
"""
from ....managers.serve.serve_cli import *  # noqa: F401,F403
