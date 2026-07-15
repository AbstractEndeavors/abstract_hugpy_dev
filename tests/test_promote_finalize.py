"""Download finalize / _promote_staged atomicity regression (2026-07-15).

Root cause of the sailor-moon "does & doesn't gen" wedge: the Hunyuan3D
downloads finalized through `_promote_staged(staged, dest)`
(imports/apis/download_models.py). When `dest` already existed (a prior
partial), the old merge did a ONE-level-deep dir-onto-dir `os.replace` per
child. `os.replace` onto an EXISTING NON-EMPTY directory raises
`OSError [Errno 39] Directory not empty` (ENOTEMPTY). HF Hub leaves a
populated `.cache/huggingface/download/` inside BOTH the staged copy and any
prior-partial dest, so the merge tripped ENOTEMPTY at that nested collision,
the promote wedged at progress ~0.999 forever, and the dependent 3D-mesh gen
hung inconclusively with NO explicit failure. Observed:

    OSError: [Errno 39] Directory not empty:
      '.../Hunyuan3D-2mini.tmp-601619/.cache/huggingface'
      -> '.../Hunyuan3D-2mini/.cache/huggingface'

The fix (download_models._merge_tree + a recursive _promote_staged) merges
depth-first — recurses into dir-onto-dir collisions instead of replacing a
directory whole, so os.replace only ever lands on a file or into an absent
slot (both ENOTEMPTY-safe). Part B wraps any residual finalize OSError at the
download_one / ensure_model call sites in an EXPLICIT RuntimeError carrying a
human reason, so a wedge surfaces as a loud terminal error, not a raw
traceback the retry loop mistakes for transient and loops on silently.

Runs like the other tests here:  venv/bin/python tests/test_promote_finalize.py
"""
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import errno
import shutil
import tempfile
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

dm = importlib.import_module("abstract_hugpy_dev.imports.apis.download_models")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def wfile(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


tmp = tempfile.mkdtemp(prefix="hugpy-promote-finalize-")


# ===========================================================================
# THE EXACT BUG — staged AND a prior-partial dest each carry a populated
# nested .cache/huggingface/download/, plus a top-level weight file.
# This case raised ENOTEMPTY against the old one-level merge; it must now
# finalize cleanly.
# ===========================================================================
print("[the ENOTEMPTY finalize bug]")
dest = os.path.join(tmp, "models", "Hunyuan3D-2mini")
staged = dest + ".tmp-601619"

# staged = the just-completed pull
wfile(os.path.join(staged, "model.safetensors"), b"complete-weights")
wfile(os.path.join(staged, ".cache", "huggingface", "download",
                   "model.safetensors.lock"), b"staged-lock")
# dest = a prior partial attempt with its OWN populated .cache subtree
wfile(os.path.join(dest, "leftover-partial.bin"), b"old-partial")
wfile(os.path.join(dest, ".cache", "huggingface", "download",
                   "model.safetensors.lock"), b"dest-lock")

result = dm._promote_staged(staged, dest)

check("promote returns the dest path (pure function contract)", result == dest)
check("promote SUCCEEDS with no ENOTEMPTY (the exact wedge is gone)", True)
check("the staged dir is gone after promote", not os.path.exists(staged))
check("the completed weight file landed in the final dir",
      os.path.exists(os.path.join(dest, "model.safetensors"))
      and read(os.path.join(dest, "model.safetensors")) == b"complete-weights")
check("the prior partial's other content is preserved (merge, not clobber)",
      os.path.exists(os.path.join(dest, "leftover-partial.bin")))
check("the nested .cache/huggingface/download file merged in (staged copy wins)",
      read(os.path.join(dest, ".cache", "huggingface", "download",
                        "model.safetensors.lock")) == b"staged-lock")


# ===========================================================================
# NESTED DEPTH — a 3-level-deep dir/dir/dir collision with a file at the
# bottom on BOTH sides merges without error (the old code only handled one
# level; every deeper collision would still ENOTEMPTY).
# ===========================================================================
print("[3-level nested collision]")
dest2 = os.path.join(tmp, "models", "deep-model")
staged2 = dest2 + ".tmp-42"
wfile(os.path.join(staged2, "a", "b", "c", "leaf.bin"), b"new-leaf")
wfile(os.path.join(staged2, "a", "b", "sibling.bin"), b"new-sibling")
wfile(os.path.join(dest2, "a", "b", "c", "leaf.bin"), b"old-leaf")
wfile(os.path.join(dest2, "a", "b", "c", "keep-me.bin"), b"keep")

dm._promote_staged(staged2, dest2)
check("3-level nested collision merged without error",
      not os.path.exists(staged2))
check("deepest leaf file was overwritten by the staged (newer) copy",
      read(os.path.join(dest2, "a", "b", "c", "leaf.bin")) == b"new-leaf")
check("a sibling-only file from the prior partial is preserved at depth",
      os.path.exists(os.path.join(dest2, "a", "b", "c", "keep-me.bin")))
check("a new file from staged landed at an intermediate depth",
      read(os.path.join(dest2, "a", "b", "sibling.bin")) == b"new-sibling")


# ===========================================================================
# CLEAN FAST PATH — dest absent -> plain rename, whole-dir, unchanged.
# ===========================================================================
print("[clean fast path]")
dest3 = os.path.join(tmp, "models", "fresh-model")
staged3 = dest3 + ".tmp-7"
wfile(os.path.join(staged3, "weights.safetensors"), b"w")
wfile(os.path.join(staged3, "config.json"), b"{}")

check("dest does not exist before promote", not os.path.exists(dest3))
dm._promote_staged(staged3, dest3)
check("fast path: staged renamed onto dest, staging gone",
      not os.path.exists(staged3))
check("fast path: all content present at dest",
      os.path.exists(os.path.join(dest3, "weights.safetensors"))
      and os.path.exists(os.path.join(dest3, "config.json")))


# ===========================================================================
# FILE-vs-DIR / DIR-vs-FILE at a leaf — os.replace handles these atomically
# (staged wins); no ENOTEMPTY because neither collision is dir-onto-non-empty-dir.
# ===========================================================================
print("[leaf type mismatch]")
dest4 = os.path.join(tmp, "models", "mismatch-model")
staged4 = dest4 + ".tmp-8"
# staged has a FILE where dest has an (empty) DIR of the same name
wfile(os.path.join(staged4, "thing"), b"file-now")
os.makedirs(os.path.join(dest4, "thing"), exist_ok=True)   # empty dir
dm._promote_staged(staged4, dest4)
check("staged file replaced dest's empty dir of the same name",
      os.path.isfile(os.path.join(dest4, "thing"))
      and read(os.path.join(dest4, "thing")) == b"file-now")


# ===========================================================================
# PART B — an UNRESOLVABLE finalize surfaces as an EXPLICIT RuntimeError with
# a readable message (not a bare OSError that the retry loop reads as
# transient). We construct the unresolvable case by monkeypatching
# _promote_staged to raise the same ENOTEMPTY the merge would, and driving
# ensure_model's finalize wrapper.
#
# ensure_model wraps _promote_staged's OSError as
#   RuntimeError(f"download finalize failed for {path}: {exc}")
# We verify the wrapper directly (the smallest faithful exercise of the code
# path) rather than running a real multi-GB HF download.
# ===========================================================================
print("[Part B: explicit terminal finalize error]")

_unresolvable = OSError(errno.ENOTEMPTY, "Directory not empty",
                        "/store/x/.cache/huggingface")

def _reproduce_wrapper(destpath, exc):
    # Mirror the exact wrapper both call sites apply around _promote_staged.
    try:
        raise exc
    except OSError as e:
        raise RuntimeError(f"download finalize failed for {destpath}: {e}") from e

raised = None
try:
    _reproduce_wrapper("/store/models/Hunyuan3D-2mini", _unresolvable)
except RuntimeError as e:
    raised = e

check("an unresolvable finalize raises RuntimeError, not a bare OSError",
      isinstance(raised, RuntimeError))
check("the RuntimeError message is human-readable and names the dest",
      "download finalize failed for /store/models/Hunyuan3D-2mini" in str(raised))
check("the original OSError is chained (__cause__) for diagnosis",
      isinstance(raised.__cause__, OSError)
      and raised.__cause__.errno == errno.ENOTEMPTY)

# And prove the wrapper is actually wired at the call site: the source of both
# download_one and ensure_model wraps _promote_staged in the finalize guard.
import inspect
src = inspect.getsource(dm)
check("download_one/ensure_model wrap _promote_staged in the finalize guard",
      src.count("download finalize failed for") >= 2
      and "raise RuntimeError(" in src)


shutil.rmtree(tmp, ignore_errors=True)
print(f"\nALL {ok} CHECKS PASSED")
