"""HOT-CACHE TIER — an automatic LRU cache of the MAIN model catalog on a
worker's fast box-local NVMe.

Operator doctrine (verbatim): "anything that is called should be on the hot
drive; when space gets tight and one is needed then the fifo should go by the
time since called and space afforded by deletion."

Shape of the mechanism
----------------------
The canonical models live on the big/shared array under ``MODELS_HOME``
(e.g. ``/mnt/llm_storage/models`` — the fleet's source of truth). This tier
promotes the *hot set* — the models actually being called — onto a box-local
NVMe (``HUGPY_HOT_CACHE_ROOT``) and serves loads from there. The shared array
stays the SOURCE OF TRUTH and is NEVER written or deleted by this module.

Read-through, never blocking
    ``use(shared_path)`` returns the path a loader should open. If a COMPLETE
    hot copy exists it returns the hot path (and stamps "last called"); else it
    kicks a *background* promotion and returns the shared path UNCHANGED for
    this (cold) load. A cold ~100 GiB HDD->NVMe copy is minutes of sustained IO
    and can NEVER sit inside a synchronous load request, so promotion is always
    async. Warm a model once and subsequent calls hit NVMe.

Env-gated, zero behaviour change when unset
    Unset ``HUGPY_HOT_CACHE_ROOT`` -> ``use()`` returns its argument byte-for-
    byte. A box without the env, or central, behaves exactly as before.

Concurrency / correctness
    A SINGLE promoter thread drains a queue (one big copy at a time — the shared
    array can't sustain many). Each file copies to a ``.part`` temp then
    atomically renames, and the completeness gate requires EVERY file present
    with a matching size, so a partial/in-flight copy is never resolvable.

Eviction = the operator's FIFO-by-time-since-called
    When a promotion needs room beyond the budget, evict least-recently-CALLED
    hot entries first, weighing the space each frees, until the incoming model
    fits. If it cannot fit even after evicting every eligible entry, SKIP the
    promotion (log it) — the actual load already ran off the shared array, so a
    load NEVER fails for lack of hot space.

Anti-thrash (churn is the normal mode here)
    Most models the 3090 serves are large relative to the budget (two or three
    fill the drive) and rotate in and out constantly. A model that JUST served
    must not be displaced by a first-time/stale caller — otherwise two big
    models alternating would copy+evict ~100 GiB per call. So an entry idle for
    less than ``HUGPY_HOT_CACHE_MIN_RESIDENCY_S`` is NOT an eviction candidate;
    if the only room is behind such a fresh entry, the promotion is skipped and
    that model keeps serving from the shared array until the activity pattern
    genuinely shifts.

This is the GENERAL mechanism for the main catalog. The studio render path has
a sibling ``STUDIO_WEIGHTS_HOT_ROOT`` (passive: use a hot dir if an operator
placed one there); see the report note on unifying the two.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import queue
import re
import shutil
import threading
import time

from . import chunksum_verify as _cv

logger = logging.getLogger(__name__)


def _verify_staged(src: str, staged: str) -> tuple[str, str]:
    """Content-verify a staged .part against its shared-store source's sidecar.

    Split out from _promote so it is directly drivable in tests and so a
    verification bug can never take down the promoter thread: an unexpected
    exception degrades to UNVERIFIED (copy proceeds, as it did before this
    gate existed) rather than turning the hot tier into a hard dependency on
    the sidecar machinery. Only a POSITIVE corruption proof stops a promote.
    """
    try:
        return _cv.verify_against_source(staged, src)
    except Exception as exc:  # noqa: BLE001 — never fail a promote on our own bug
        logger.warning("hot_cache: verification error for %s (%s) — treating as "
                       "unverified", src, exc)
        return _cv.UNVERIFIED, f"verifier error: {exc}"

# --------------------------------------------------------------------------- #
# Env knobs (read live so a systemd drop-in edit + restart takes effect; unset
# ROOT == disabled == byte-identical behaviour).
# --------------------------------------------------------------------------- #
_ENV_ROOT = "HUGPY_HOT_CACHE_ROOT"
_ENV_GIB = "HUGPY_HOT_CACHE_GIB"
_ENV_MIN_RESIDENCY = "HUGPY_HOT_CACHE_MIN_RESIDENCY_S"

_DEFAULT_GIB = 225.0
_DEFAULT_MIN_RESIDENCY_S = 1800.0
_INDEX_NAME = ".hot_cache_index.json"

GiB = 1 << 30


def _root() -> str:
    return (os.environ.get(_ENV_ROOT) or "").strip()


def _budget_bytes() -> int:
    try:
        return int(float(os.environ.get(_ENV_GIB, _DEFAULT_GIB)) * GiB)
    except (TypeError, ValueError):
        return int(_DEFAULT_GIB * GiB)


def _min_residency_s() -> float:
    try:
        return float(os.environ.get(_ENV_MIN_RESIDENCY, _DEFAULT_MIN_RESIDENCY_S))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_RESIDENCY_S


def enabled() -> bool:
    """True only when a hot root is configured AND usable (exists + writable).
    A misconfigured root disables the tier rather than raising into a load."""
    root = _root()
    if not root:
        return False
    try:
        os.makedirs(root, exist_ok=True)
        return os.path.isdir(root) and os.access(root, os.W_OK)
    except OSError:
        return False


def _models_home() -> str:
    try:
        from ...imports.src.constants.constants import MODELS_HOME
        return str(MODELS_HOME)
    except Exception:  # noqa: BLE001
        return "/mnt/llm_storage/models"


# --------------------------------------------------------------------------- #
# Path mapping. rel() mirrors the shared layout under the hot root so a model
# lands at <root>/<family>/<task>/<owner>/<repo>/... — identical to its shared
# subtree, which makes the mapping stable and human-legible.
# --------------------------------------------------------------------------- #
def _rel(path: str) -> str:
    home = _models_home()
    try:
        rel = os.path.relpath(path, home)
        if rel.startswith(".."):
            raise ValueError
        return rel
    except (ValueError, OSError):
        return os.path.basename(path.rstrip(os.sep))


def hot_path(shared_path: str) -> str:
    """The hot-root mirror of a shared file OR dir."""
    return os.path.join(_root(), _rel(shared_path))


def _under_root(path: str) -> bool:
    root = _root()
    if not root:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except (ValueError, OSError):
        return False


def _entry_src_dir(shared_path: str) -> str:
    """The shared dir that is the EVICTION UNIT for this model. For a resolved
    GGUF file it's the containing dir (holds the quant + shards + mmproj); for a
    transformers/diffusers model it's the model dir itself."""
    if os.path.isdir(shared_path):
        return os.path.abspath(shared_path)
    return os.path.abspath(os.path.dirname(shared_path))


def _entry_key(shared_path: str) -> str:
    """Stable index key + display name = the MODELS_HOME-relative entry dir."""
    return _rel(_entry_src_dir(shared_path))


# --------------------------------------------------------------------------- #
# File set: the exact bytes a call needs on the hot drive.
#   * GGUF file input  -> the quant, every shard of a split model, + mmproj.
#   * model-dir input  -> every real file under the dir (transformers/diffusers).
# --------------------------------------------------------------------------- #
def _gguf_file_set(src: str) -> list[str]:
    out = [src]
    d = os.path.dirname(src)
    base = os.path.basename(src)
    m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base, re.IGNORECASE)
    if m:
        stem = base[: m.start()]
        out = glob.glob(os.path.join(d, f"{stem}-*-of-{m.group(1)}.gguf")) + \
            glob.glob(os.path.join(d, f"{stem}-*-of-{m.group(1)}.GGUF"))
    out += glob.glob(os.path.join(d, "*mmproj*.gguf")) + glob.glob(os.path.join(d, "*mmproj*.GGUF"))
    return sorted({f for f in out if os.path.isfile(f)})


def _is_bookkeeping(name: str) -> bool:
    """Transfer bookkeeping / staging remnants — NOT model content.

    ``.chunksums-*.json`` sidecars are verification metadata that belong beside
    the SOURCE; copying them to the hot drive spends the weight budget on
    bookkeeping and makes them look like files needing verification (they have
    no sidecar of their own -> a pointless "UNVERIFIED" line per promote).
    ``.part``/``.state.json`` are a crashed pull's leftovers: promoting them
    would carry a wedge onto the hot drive — the exact class of artifact that
    misled this incident for 32h. Mirrors central's own exclusion list
    (worker_routes.py: ``".chunksums-" in name or name.endswith((".part", …))``)
    and reconcile.py's, so the three agree on what counts as real weight.
    """
    low = name.lower()
    return ".chunksums-" in low or low.endswith((".part", ".state.json"))


def _dir_file_set(d: str) -> list[str]:
    out: list[str] = []
    for root, _sub, files in os.walk(d):
        for f in files:
            if _is_bookkeeping(f):
                continue
            p = os.path.join(root, f)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(p)
    return sorted(out)


def _file_set(shared_path: str) -> list[str]:
    if os.path.isdir(shared_path):
        return _dir_file_set(shared_path)
    return _gguf_file_set(shared_path)


def _sizes(paths: list[str]) -> int:
    total = 0
    for p in paths:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


def is_complete(shared_path: str) -> bool:
    """True iff every file the call needs is on the hot drive with a matching
    size. The size match is the completeness gate: a truncated / in-flight copy
    (the ``.part`` never renames until it verifies) reads as incomplete and
    transparently falls back to the shared array.

    NOTE the size match here is a RESOLUTION gate, not a trust gate — content is
    proven at promote time (_promote -> _verify_staged), which is the only place
    that has the source beside the copy. Re-hashing every file on every call
    would put minutes of IO inside a load."""
    return not incomplete_reason(shared_path)


def incomplete_reason(shared_path: str) -> str:
    """Why the hot copy isn't usable — "" when it IS complete.

    Exists because the incident's cost was diagnostic, not mechanical: a hot
    tree of ``.part`` files read as a bare False, the loader silently fell back,
    and the failure eventually surfaced as diffusers hunting a legacy ``.bin``
    that was never the problem. Naming the real state (staging file present,
    size mismatch, absent) is what turns 32h into 32 seconds.
    """
    files = _file_set(shared_path)
    if not files:
        return "no source files in the shared store"
    for f in files:
        hp = hot_path(f)
        try:
            if os.path.isfile(hp):
                if os.path.getsize(hp) == os.path.getsize(f):
                    continue
                return (f"{os.path.basename(f)}: hot copy is "
                        f"{os.path.getsize(hp)}B vs {os.path.getsize(f)}B in the "
                        f"shared store (incomplete copy)")
            if os.path.isfile(hp + ".part"):
                st = os.stat(hp + ".part")
                return (f"{os.path.basename(f)}: transfer never finished — a "
                        f"staging .part ({st.st_size}B, {(time.time() - st.st_mtime) / 3600:.1f}h "
                        f"old) is present but was never promoted. This model is "
                        f"NOT usable from the hot cache; serving from the shared "
                        f"store instead")
            return f"{os.path.basename(f)}: absent from the hot cache"
        except OSError as exc:
            return f"{os.path.basename(f)}: {exc}"
    return ""


# --------------------------------------------------------------------------- #
# Persistent index. entries[<rel_key>] = {model_key, bytes, last_called,
# promoted_at, kind}. Written atomically; rebuilt by scanning the hot root if
# missing/corrupt.
# --------------------------------------------------------------------------- #
_INDEX: dict = {"version": 1, "entries": {}}
_INDEX_LOADED = False
_INDEX_LOCK = threading.RLock()


def _index_file() -> str:
    return os.path.join(_root(), _INDEX_NAME)


def _load_index() -> None:
    global _INDEX, _INDEX_LOADED
    with _INDEX_LOCK:
        if _INDEX_LOADED:
            return
        path = _index_file()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                _INDEX = {"version": 1, "entries": data["entries"]}
                _INDEX_LOADED = True
                return
            raise ValueError("index shape")
        except (OSError, ValueError, json.JSONDecodeError):
            _INDEX = {"version": 1, "entries": {}}
            _rebuild_index_locked()
            _INDEX_LOADED = True


def _rebuild_index_locked() -> None:
    """Reconstruct the index from what is on the hot root. An entry = a leaf dir
    holding real weight files; last_called/promoted_at seed from newest mtime
    (best available truth) so recency survives an index loss."""
    root = _root()
    if not root or not os.path.isdir(root):
        return
    entries: dict = {}
    for dirpath, _sub, files in os.walk(root):
        real = [f for f in files if not f.startswith(".") and
                os.path.isfile(os.path.join(dirpath, f)) and
                not os.path.islink(os.path.join(dirpath, f))]
        if not real:
            continue
        # Only leaf-ish dirs that directly hold files become entries; a parent
        # that merely contains subdirs is not itself an entry.
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            continue
        try:
            newest = max(os.path.getmtime(os.path.join(dirpath, f)) for f in real)
            sz = sum(os.path.getsize(os.path.join(dirpath, f)) for f in real)
        except OSError:
            continue
        entries[rel] = {"model_key": rel, "bytes": int(sz),
                        "last_called": float(newest), "promoted_at": float(newest),
                        "kind": "dir"}
    _INDEX["entries"] = entries
    _save_index_locked()
    if entries:
        logger.info("hot_cache: rebuilt index from disk (%d entries, %.1f GiB)",
                    len(entries), sum(e["bytes"] for e in entries.values()) / GiB)


def _save_index_locked() -> None:
    path = _index_file()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_INDEX, fh)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("hot_cache: index save failed: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _index_used_bytes() -> int:
    with _INDEX_LOCK:
        return sum(int(e.get("bytes", 0) or 0) for e in _INDEX["entries"].values())


def _stamp_called(key: str, bytes_hint: int = 0, kind: str = "dir") -> None:
    now = time.time()
    with _INDEX_LOCK:
        e = _INDEX["entries"].get(key)
        if e is None:
            e = {"model_key": key, "bytes": int(bytes_hint or 0),
                 "last_called": now, "promoted_at": now, "kind": kind}
            _INDEX["entries"][key] = e
        else:
            e["last_called"] = now
            if bytes_hint:
                e["bytes"] = int(bytes_hint)
        _save_index_locked()


# --------------------------------------------------------------------------- #
# Free space + pin awareness.
# --------------------------------------------------------------------------- #
def _free_bytes() -> int:
    try:
        st = os.statvfs(_root())
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _is_pinned(key: str) -> bool:
    """Best-effort: a pinned model is evictable from HOT only LAST (its shared
    copy is safe, so hot eviction is safe too — just wasteful). Maps the entry's
    repo basename to the worker's pinned settings; any failure -> not pinned."""
    try:
        from ...worker_agent.agent import _pinned  # lazy: worker-side only
    except Exception:  # noqa: BLE001
        return False
    base = os.path.basename(key.rstrip(os.sep))
    try:
        return bool(_pinned(key) or _pinned(base))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Eviction — the operator's FIFO-by-time-since-called, with the anti-thrash
# residency guard. Runs ONLY inside the single promoter thread, so it never
# races a concurrent promotion of the same entry.
# --------------------------------------------------------------------------- #
def _evict_locked(key: str) -> int:
    """Delete a hot entry dir + drop its index row. Returns bytes freed. NEVER
    touches the shared store (only paths under the hot root)."""
    hp = os.path.join(_root(), key)
    freed = int(_INDEX["entries"].get(key, {}).get("bytes", 0) or 0)
    try:
        if _under_root(hp) and os.path.isdir(hp):
            shutil.rmtree(hp)
    except OSError as exc:
        logger.warning("hot_cache: evict failed for %s: %s", key, exc)
    _INDEX["entries"].pop(key, None)
    return freed


def _make_room(need: int, keep_key: str) -> bool:
    """Evict least-recently-CALLED entries until `need` fits under the budget AND
    on the filesystem. Anti-thrash: an entry idle < min_residency is protected
    (not a candidate). Pinned entries evict last. Returns True if room was made,
    False if the model can't fit even after evicting every eligible entry (then
    the caller SKIPS promotion)."""
    budget = _budget_bytes()
    residency = _min_residency_s()
    now = time.time()

    def fits() -> bool:
        return (_index_used_bytes() + need <= budget) and (_free_bytes() >= need)

    with _INDEX_LOCK:
        if fits():
            return True
        candidates = [(k, v) for k, v in _INDEX["entries"].items() if k != keep_key]
        # Anti-thrash: exclude anything that served within the residency window.
        eligible = [(k, v) for (k, v) in candidates
                    if (now - float(v.get("last_called", 0) or 0)) >= residency]
        fresh = len(candidates) - len(eligible)
        # Order: unpinned before pinned, then least-recently-called first.
        eligible.sort(key=lambda kv: (_is_pinned(kv[0]),
                                      float(kv[1].get("last_called", 0) or 0)))
        for k, v in eligible:
            idle = now - float(v.get("last_called", 0) or 0)
            freed = _evict_locked(k)
            logger.info("hot_cache: evicted %s (%.1f GiB, idle %.0f min) to make "
                        "room for %s", k, freed / GiB, idle / 60.0, keep_key)
            if fits():
                _save_index_locked()
                return True
        _save_index_locked()
        if not fits():
            if fresh:
                logger.info("hot_cache: SKIP promote %s — need %.1f GiB but only "
                            "%.1f GiB freeable; %d recent entr%s within the "
                            "%.0f-min residency window are protected (anti-thrash)",
                            keep_key, need / GiB, _free_bytes() / GiB, fresh,
                            "y" if fresh == 1 else "ies", residency / 60.0)
            else:
                logger.info("hot_cache: SKIP promote %s — need %.1f GiB, does not "
                            "fit even after evicting all evictable entries",
                            keep_key, need / GiB)
            return False
        return True


# --------------------------------------------------------------------------- #
# Promotion — single promoter thread draining a queue. Atomic per file
# (.part -> rename); completeness requires every file, so partials never serve.
# --------------------------------------------------------------------------- #
_QUEUE: "queue.Queue[str]" = queue.Queue()
_QUEUED: set = set()          # entry keys queued or in-flight (dedup)
_INFLIGHT: str | None = None
_STATE_LOCK = threading.Lock()
_PROMOTER: threading.Thread | None = None


def _promote(shared_path: str) -> None:
    """Copy a model's file set into the hot cache, evicting LRU first. Returns
    nothing; logs outcomes. Abandons cleanly (removes the .part) if the source
    changes/vanishes mid-copy."""
    key = _entry_key(shared_path)
    files = _file_set(shared_path)
    if not files:
        return
    if is_complete(shared_path):
        _stamp_called(key, bytes_hint=_sizes(files))
        return
    need = _sizes(files)
    if need <= 0:
        return
    budget = _budget_bytes()
    if need > budget:
        logger.info("hot_cache: SKIP promote %s — %.1f GiB exceeds the whole "
                    "%.1f GiB budget", key, need / GiB, budget / GiB)
        return
    if not _make_room(need, keep_key=key):
        return
    try:
        for f in files:
            dst = hot_path(f)
            try:
                if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(f):
                    continue                                   # already present
            except OSError:
                pass
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            src_size = os.path.getsize(f)
            logger.info("hot_cache: promoting %s -> %s (%.1f GiB)",
                        f, dst, src_size / GiB)
            shutil.copyfile(f, tmp)
            if os.path.getsize(tmp) != src_size:
                os.remove(tmp)
                raise IOError(f"size mismatch copying {f} (source changed mid-copy?)")
            # CONTENT gate, not a length gate. A size check is exactly what let
            # the 2026-07-15 sd-turbo corruption through: the staged vae was
            # 167335342 bytes — the CORRECT size — but the bytes were wrong, so
            # it promoted, then surfaced 32h later as diffusers complaining
            # about a missing legacy .bin (an error naming the wrong file
            # entirely). Verify against the shared store's chunksums sidecar.
            verdict, detail = _verify_staged(f, tmp)
            if verdict == _cv.CORRUPT:
                # Do NOT promote, and do NOT keep the bad bytes: leaving the
                # .part is what wedged ae for 32h and misled the next reader.
                # Dropping it makes the next promote a clean, succeeding retry.
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise IOError(
                    f"CORRUPT TRANSFER of {f} -> {dst}: {detail}. Refusing to "
                    f"promote; staged copy discarded so the next call re-copies. "
                    f"The shared-store source is unchanged.")
            if verdict == _cv.UNVERIFIED:
                # Absent/stale sidecar = no evidence either way. Copy proceeds
                # (see chunksum_verify: blocking here would break every file
                # that has no sidecar), but we say so rather than implying the
                # bytes were checked.
                logger.info("hot_cache: %s promoted UNVERIFIED (%s)",
                            os.path.basename(f), detail)
            os.replace(tmp, dst)
        if is_complete(shared_path):
            _stamp_called(key, bytes_hint=need)
            logger.info("hot_cache: promoted %s (%.1f GiB) — next call is NVMe-hot",
                        key, need / GiB)
        else:
            logger.warning("hot_cache: promote %s finished incomplete — leaving "
                           "shared-served", key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hot_cache: promote failed for %s: %s", key, exc)


def _promoter_loop() -> None:
    global _INFLIGHT
    while True:
        shared_path = _QUEUE.get()
        key = _entry_key(shared_path)
        with _STATE_LOCK:
            _INFLIGHT = key
        try:
            _promote(shared_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hot_cache: promoter error for %s: %s", key, exc)
        finally:
            with _STATE_LOCK:
                _INFLIGHT = None
                _QUEUED.discard(key)
            _QUEUE.task_done()


def _ensure_promoter() -> None:
    global _PROMOTER
    with _STATE_LOCK:
        if _PROMOTER is not None and _PROMOTER.is_alive():
            return
        _PROMOTER = threading.Thread(target=_promoter_loop, name="hot-cache-promoter",
                                     daemon=True)
        _PROMOTER.start()


def _enqueue(shared_path: str) -> None:
    key = _entry_key(shared_path)
    with _STATE_LOCK:
        if key in _QUEUED:
            return                                            # already queued/in-flight
        _QUEUED.add(key)
    _ensure_promoter()
    _QUEUE.put(shared_path)


# --------------------------------------------------------------------------- #
# Public read-through resolution.
# --------------------------------------------------------------------------- #
def use(shared_path: str, promote: bool = True) -> str:
    """Resolve the path a loader should open. Returns the HOT copy when complete
    (and stamps it "called" for LRU); otherwise schedules an async promotion and
    returns ``shared_path`` UNCHANGED so the cold load runs off the shared array
    while the hot copy fills for next time. NEVER blocks, NEVER raises, NEVER
    writes/deletes the shared store. Disabled (env unset) -> returns its
    argument unchanged."""
    if not shared_path or not enabled():
        return shared_path
    if _under_root(shared_path):
        return shared_path                                    # already a hot path
    try:
        _load_index()
        why = incomplete_reason(shared_path)
        if not why:
            _stamp_called(_entry_key(shared_path), bytes_hint=_sizes(_file_set(shared_path)))
            return hot_path(shared_path)
        # Say WHY we're serving cold. A wedged .part used to be indistinguishable
        # from "not promoted yet" in the logs, which is how a 32h-stale staging
        # file stayed invisible until a loader misreported it.
        logger.info("hot_cache: serving %s from the shared store — %s",
                    _entry_key(shared_path), why)
        if promote:
            _enqueue(shared_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hot_cache: use() fell back to shared for %s: %s", shared_path, exc)
    return shared_path


# --------------------------------------------------------------------------- #
# Stale .part surfacing. A wedged staging file is junk that MISLEADS the next
# reader — the sd-turbo tree sat as .part for 32h and the eventual error blamed
# a missing .bin. We SURFACE them (heartbeat/status) rather than auto-deleting:
# a .part younger than the threshold may be an in-flight copy on another
# thread/process, and this module must never race a live transfer. Reaping is
# left to the promoter, which already discards its own .part on a failed verify
# and re-copies from scratch — so a surfaced stale .part is diagnostic, not a
# leak. Scanning is confined to the hot root (_under_root); the shared store's
# real weights are never touched.
# --------------------------------------------------------------------------- #
_STALE_PART_S = 3600.0        # 1h: far beyond any legitimate single-file copy


def stale_parts(older_than_s: float = _STALE_PART_S) -> list[dict]:
    """Wedged ``.part`` staging files under the hot root, oldest first."""
    root = _root()
    if not root or not os.path.isdir(root):
        return []
    now = time.time()
    out: list[dict] = []
    for dirpath, _sub, files in os.walk(root):
        for f in files:
            if not f.endswith(".part"):
                continue
            p = os.path.join(dirpath, f)
            if not _under_root(p):
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            age = now - st.st_mtime
            if age < older_than_s:
                continue
            out.append({"path": p, "rel": os.path.relpath(p, root),
                        "bytes": int(st.st_size), "age_s": round(age, 1)})
    out.sort(key=lambda e: e["age_s"], reverse=True)
    return out


def status() -> dict:
    """Honest hot-cache overview for the worker storage view / console."""
    if not enabled():
        return {"enabled": False}
    try:
        _load_index()
        with _INDEX_LOCK:
            entries = [
                {"model_key": e.get("model_key", k), "bytes": int(e.get("bytes", 0) or 0),
                 "last_called": float(e.get("last_called", 0) or 0),
                 "promoted_at": float(e.get("promoted_at", 0) or 0),
                 "kind": e.get("kind", "dir")}
                for k, e in _INDEX["entries"].items()
            ]
        entries.sort(key=lambda m: m["last_called"], reverse=True)
        with _STATE_LOCK:
            promoting = _INFLIGHT
            queued = sorted(_QUEUED - ({_INFLIGHT} if _INFLIGHT else set()))
        # Surfaced so a wedged staging file is VISIBLE on the heartbeat instead
        # of waiting 32h to reappear as a misleading loader error.
        try:
            stale = stale_parts()
        except OSError:
            stale = []
        return {
            "enabled": True,
            "stale_parts": stale,
            "root": _root(),
            "budget_bytes": _budget_bytes(),
            "used_bytes": _index_used_bytes(),
            "free_bytes": _free_bytes(),
            "min_residency_s": _min_residency_s(),
            "promoting": promoting,
            "queued": queued,
            "entries": entries,
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "root": _root(), "error": str(exc)}
