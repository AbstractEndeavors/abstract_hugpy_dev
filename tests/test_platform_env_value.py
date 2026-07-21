"""Regression test for the computron garbage-path incident (k30).

``abstract_hugpy_dev._platform.env_value`` prefers a file-based reader
(``abstract_essentials.get_env_value``, re-exported as
``abstract_hugpy_dev.imports.src.standalone_utils.get_env_value``) that reads
a ``.env`` file line by line and hands back everything after the ``=``
VERBATIM — including a trailing inline ``# comment``. On computron,
``HUGPY_ENGINE_DIR=/mnt/storage/hugpy-worker/engine          # native llama.cpp
engine build dir`` in a real ``.env`` produced a literal directory whose name
was the comment text (via ``paths.engine_dir()``/``models_root()``
``os.makedirs`` calls), plus a 182GB duplicate store.

The fix lives in OUR seam (``_platform.__init__._sanitize_env_str`` /
``env_value``), not in the ``abstract_security``/``abstract_essentials``
site-package (which reinstalls on every worker and would silently drop any
local patch).

This test drives the REAL ``env_value()`` end-to-end against a real ``.env``
file. ``abstract_essentials.get_env_value`` resolves its search path from
``os.getcwd()`` (no path/file_name args are passed at our call site), so we
``chdir`` into a throwaway directory holding the ``.env`` — the same
resolution mechanism the incident happened through — rather than
monkeypatching the reader, so the test exercises the real precedence and
real file-parsing behavior end to end.

Cases:
  (a) inline-commented path value  -> comment stripped, path clean
  (b) quoted value containing '#'  -> quotes unwrapped, '#' preserved as data
  (c) clean value                  -> passed through unchanged
  (d) key missing from .env        -> falls back to sanitized os.environ

Runs like the other tests here: venv/bin/python tests/test_platform_env_value.py
"""
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev._platform import env_value, _sanitize_env_str  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ── snapshot everything we mutate so the file leaves no residue ─────────────
_CWD_BACKUP = os.getcwd()
_ENV_BACKUP = dict(os.environ)

_TMP = tempfile.mkdtemp(prefix="hugpy-env-value-")

try:
    env_path = os.path.join(_TMP, ".env")
    with open(env_path, "w") as fh:
        fh.write(
            "HUGPY_ENGINE_DIR=/mnt/storage/hugpy-worker/engine          "
            "# native llama.cpp engine build dir\n"
        )
        fh.write('HUGPY_QUOTED=\"a#b/weird path\"\n')
        fh.write("HUGPY_CLEAN=/mnt/storage/hugpy-worker/models\n")
        # HUGPY_FALLBACK deliberately absent from the file.

    # scrub anything that could collide, then set the fallback-only var and
    # a couple of process-env decoys that must NOT shadow the file values.
    for k in ("HUGPY_ENGINE_DIR", "HUGPY_QUOTED", "HUGPY_CLEAN", "HUGPY_FALLBACK"):
        os.environ.pop(k, None)
    os.environ["HUGPY_FALLBACK"] = "  /mnt/storage/fallback   # from os.environ  "
    os.environ["HUGPY_ENGINE_DIR"] = "/should/not/win/over/file"

    os.chdir(_TMP)

    # (a) inline-commented path value -> comment stripped
    check(
        "inline comment stripped from file value",
        env_value("HUGPY_ENGINE_DIR") == "/mnt/storage/hugpy-worker/engine",
    )

    # (b) quoted value containing '#' -> unwrapped, '#' preserved as data
    check(
        "quoted value keeps internal '#' and drops the quotes",
        env_value("HUGPY_QUOTED") == "a#b/weird path",
    )

    # (c) clean value passed through unchanged
    check(
        "clean value unchanged",
        env_value("HUGPY_CLEAN") == "/mnt/storage/hugpy-worker/models",
    )

    # (d) missing key falls back to os.environ, sanitized the same way
    check(
        "missing-from-file key falls back to sanitized os.environ",
        env_value("HUGPY_FALLBACK") == "/mnt/storage/fallback",
    )

    # unit-level checks on the sanitizer itself, unquoted '#' with no
    # preceding whitespace is left alone (not treated as a comment start)
    check(
        "unquoted '#' glued to text is not treated as a comment",
        _sanitize_env_str("http://host/path#fragment") == "http://host/path#fragment",
    )
    check(
        "single-quoted value keeps internal '#'",
        _sanitize_env_str("'a#b'") == "a#b",
    )
    check(
        "unquoted value with no comment is untouched",
        _sanitize_env_str("plain/value") == "plain/value",
    )

finally:
    os.chdir(_CWD_BACKUP)
    os.environ.clear()
    os.environ.update(_ENV_BACKUP)

print(f"\nall {ok} checks passed")
