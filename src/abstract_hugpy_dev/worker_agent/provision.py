"""Model provisioning for the worker — central-first, Hugging Face fallback.

The worker ships with only a small curated model registry; CENTRAL is the
source of truth for what models exist. So before any files are fetched, the
worker first makes sure it KNOWS the model — pulling the model's config row
from central and registering it into the worker's own in-memory registry
(:func:`ensure_model_registered`). Without that step a model central assigns
but the worker wasn't built with fails to even resolve ("Unknown model_key=
None") and to provision ("Unknown model").

Once the model is known, its files are fetched in this order:

    1. From the CENTRAL node, over WireGuard, using the read-only endpoints
       /api/llm/models/<key>/manifest and /api/llm/models/<key>/file. This needs
       no Hugging Face token on the worker and reuses whatever central already
       downloaded.
    2. If central doesn't have it (409) or is unreachable, fall back to the
       normal Hugging Face download via abstract_hugpy_dev.ensure_model — which the
       inference path would call anyway.

Files are placed under the worker's OWN storage root using the same
route_destination() layout central uses, so the existing loader/`ensure_model`
finds them with no further config.
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
import urllib.parse
import urllib.request
import urllib.error

logger = logging.getLogger("abstract_hugpy_dev.worker_agent.provision")

_CHUNK = 8 * 1024 * 1024  # 8 MiB streaming chunks

# Single-flight provisioning: one download per model_key at a time. Without this
# every concurrent /infer/stream (plus the pre-provision) kicks off its own full
# multi-GB transfer into the SAME directory, and the parallel writers stomp each
# other (the symptom: a transfer "stuck" partway). Waiters block, then find the
# model already present and return immediately.
_PROVISION_LOCKS: dict[str, threading.Lock] = {}
_PROVISION_LOCKS_GUARD = threading.Lock()


def _provision_lock(model_key: str) -> threading.Lock:
    with _PROVISION_LOCKS_GUARD:
        lock = _PROVISION_LOCKS.get(model_key)
        if lock is None:
            lock = _PROVISION_LOCKS[model_key] = threading.Lock()
        return lock


# ---------------------------------------------------------------------------
# Registry sync — teach the worker about a model it wasn't built with.
# ---------------------------------------------------------------------------
def _clean_hub(value) -> str:
    """Normalise a hub_id for comparison (strip storage-path leakage)."""
    try:
        from .imports import models_config as mc
        return mc._clean_repo_id(value)
    except Exception:
        return str(value or "").strip("/")


def _assure_local_key(model_key: str):
    """Canonical local registry key for model_key (key/hub_id/suffix), or None."""
    try:
        from .imports import assure_model_key
        return assure_model_key(model_key)
    except Exception:
        return None


def _central_manifest(central_url: str, model_key: str) -> dict:
    """GET central's file manifest + routing meta for a model key."""
    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)
    return _get_json(base + "/manifest")


# Central's model list/config lives under a different prefix than the worker
# file-share routes, and that prefix has moved between builds. Try the known
# candidates rather than hard-coding one (which 404'd in the field).
_MODEL_LIST_PATHS = ("/api/models", "/api/llm/models", "/models")


def list_central_models(central_url: str) -> list[dict]:
    """Return central's model rows, trying each known list endpoint in turn.

    Each row is a model config dict (carrying at least ``model_key``/``key`` and
    ``hub_id``). Returns [] if none of the endpoints answer with a usable list.
    """
    for path in _MODEL_LIST_PATHS:
        url = central_url.rstrip("/") + path
        try:
            listing = _get_json(url)
        except Exception:
            continue
        if isinstance(listing, dict):
            return [r for r in listing.values() if isinstance(r, dict)]
        if isinstance(listing, list):
            return [r for r in listing if isinstance(r, dict)]
    return []


def _fetch_central_model_row(central_url: str, model_key: str) -> dict | None:
    """Pull one model's config row from central, however possible.

    Order:
      1. the worker file-share **manifest** endpoint
         (/api/llm/models/<key>/manifest) — proven to work and it also carries
         the routing meta we need to register. Used when model_key is central's
         own single-segment key (e.g. the assignment key ``DAN-Qwen3-1.7B``).
      2. central's model list/config, trying each known prefix, matching on key
         OR cleaned hub_id (handles a request that carries the hub_id while
         central keys the model by a short name).
    """
    # 1) manifest endpoint — reliable, and confirms central actually has files.
    if "/" not in model_key:
        try:
            meta = _central_manifest(central_url, model_key)
            if isinstance(meta, dict) and meta.get("hub_id"):
                row = dict(meta)
                row.setdefault("key", model_key)
                return row
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 409):
                logger.warning("central manifest for %s: HTTP %s", model_key, exc.code)
        except Exception as exc:
            logger.warning("central manifest for %s failed: %s", model_key, exc)

    # 2) resolve via the model list (handles hub_id + unknown prefix).
    want = _clean_hub(model_key)
    for path in _MODEL_LIST_PATHS:
        url = central_url.rstrip("/") + path
        try:
            listing = _get_json(url)
        except Exception:
            continue
        if isinstance(listing, dict):
            rows = list(listing.values())
        elif isinstance(listing, list):
            rows = listing
        else:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if (row.get("key") or row.get("model_key")) == model_key:
                return row
            if want and _clean_hub(row.get("hub_id")) == want:
                return row
    logger.warning("central has no resolvable config for %s", model_key)
    return None


def _register_local_model(model_key: str, row: dict) -> bool:
    """Insert a central-provided model row into the worker's live registry.

    Mutates MODEL_REGISTRY / MODEL_REGISTRY_DICT in place so every holder of
    those dicts (resolver, loader, get_model_config) immediately sees the model.
    """
    try:
        from .imports import models_config as mc
    except Exception as exc:
        logger.warning("cannot access local registry to register %s: %s", model_key, exc)
        return False
    try:
        cfg, why = mc.derive_model_config_row(model_key, dict(row))
        if cfg is None:
            logger.warning("central row for %s not directly usable (%s); "
                           "registering raw", model_key, why)
            cfg = dict(row)
            cfg.setdefault("model_key", model_key)
        mc.update_model_config_dict(model_key=model_key, values=cfg,
                                    dict_obj=mc.MODEL_REGISTRY)
        mc.update_model_config_dict(model_key=model_key, values=cfg,
                                    dict_obj=mc.MODEL_REGISTRY_DICT, dict_return=True)
        if model_key in mc.MODEL_REGISTRY:
            logger.info("registered model %s from central into local registry", model_key)
            return True
        logger.warning("registration of %s did not stick (failed assessment)", model_key)
        return False
    except Exception as exc:
        logger.warning("failed to register %s locally: %s", model_key, exc)
        return False


def ensure_model_registered(model_key: str, central_url: str | None) -> str | None:
    """Make sure ``model_key`` exists in the worker's LOCAL registry.

    Accepts a registry key OR a hub_id. If the worker already knows it,
    returns the canonical local key. Otherwise pulls the config row from
    central and registers it. Returns the canonical local key, or None if the
    model can't be learned (no central / central doesn't have it).
    """
    local = _assure_local_key(model_key)
    if local:
        return local
    if not central_url:
        return None

    row = _fetch_central_model_row(central_url, model_key)
    if not row:
        logger.warning("central has no config for %s; cannot register", model_key)
        return None

    key = row.get("key") or row.get("model_key") or model_key
    if _register_local_model(key, row):
        return _assure_local_key(key) or key
    return None


def _comfy_checkpoints_dir() -> str:
    """Where THIS box's ComfyUI loads checkpoints from. Override with
    COMFY_CHECKPOINTS_DIR; default matches the standard install."""
    return os.environ.get("COMFY_CHECKPOINTS_DIR") or os.path.expanduser(
        "~/ComfyUI/models/checkpoints")


def ensure_comfy_checkpoint(model_key: str, central_url: str | None) -> bool:
    """Comfy provisioning: the checkpoint must end up INSIDE ComfyUI's
    models/checkpoints — everything is symlinks, never copies.

      1. already in ComfyUI's dir            -> done (operator hand-placed)
      2. present in the hugpy model layout   -> symlink it in
      3. neither -> pull from central via the NORMAL file machinery (central
         symlinks its /checkpoints store into the manifest layout), then
         symlink into ComfyUI's dir.
    """
    try:
        model_key = ensure_model_registered(model_key, central_url) or model_key
        from .imports import get_model_config
        cfg = get_model_config(model_key)
        filename = getattr(cfg, "filename", None)
        if not filename:
            return False
        dest = os.path.join(_comfy_checkpoints_dir(), filename)
        if os.path.exists(dest):
            return True
        # in the hugpy layout already?
        from .imports import route_destination
        src_dir = route_destination({
            "hub_id": getattr(cfg, "hub_id", None), "framework": "comfy",
            "primary_task": getattr(cfg, "primary_task", None),
            "name": getattr(cfg, "name", None),
            "folder": getattr(cfg, "folder", None), "filename": filename})
        src = os.path.join(src_dir, filename)
        if not os.path.isfile(src) and central_url:
            logger.info("comfy checkpoint %s missing locally — pulling from "
                        "central", filename)
            if not ensure_model_present(model_key, central_url):
                return False
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(os.path.realpath(src), dest)
            logger.info("comfy checkpoint linked: %s -> %s", dest, src)
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_comfy_checkpoint(%s) failed: %s", model_key, exc)
        return False


def model_is_local(model_key: str) -> bool:
    """True if the model already looks downloaded under the worker's storage."""
    try:
        from .imports import (
            get_model_config, model_looks_downloaded, get_model_path,
        )
        cfg = get_model_config(model_key)
        # Comfy rows: "local" means the checkpoint is loadable by THIS box's
        # ComfyUI (its own dir counts — operator hand-placed files included).
        if getattr(cfg, "framework", None) == "comfy":
            filename = getattr(cfg, "filename", "") or ""
            if filename and os.path.exists(
                    os.path.join(_comfy_checkpoints_dir(), filename)):
                return True
            return bool(model_looks_downloaded(get_model_path(model_key), cfg))
        return bool(model_looks_downloaded(get_model_path(model_key), cfg))
    except Exception:
        return False


def _on_shared_model_store(rp: str) -> bool:
    """True when `rp` lives on SHARED/central model storage — the canonical
    catalog other fleet nodes read — so it must NEVER be deleted from here.

    Operator invariant: "nothing should delete from the central drive." A box
    that mounts the shared catalog (e.g. ae on the USB-C NAS) serves the fleet's
    source-of-truth copies; a reap there would remove them for everyone. Two
    independent signals, either one trips the guard (fail-safe):
      1. ``HUGPY_SHARED_MODEL_STORE`` truthy — explicit per-box opt-out of ALL
         deletes; set on every box whose model root is the shared/central volume.
      2. A ``.hugpy-central-catalog`` sentinel at or above ``rp`` — the central
         node drops it at MODELS_HOME root, so a shared mount auto-detects even
         if the env flag was forgotten.
    """
    if os.environ.get("HUGPY_SHARED_MODEL_STORE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    d = rp
    for _ in range(64):                       # walk up to the mount root
        try:
            if os.path.exists(os.path.join(d, ".hugpy-central-catalog")):
                return True
        except OSError:
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return False


def _model_store_reapable(rp: str) -> bool:
    """SAFE-BY-DEFAULT gate for deleting a model file. Returns True ONLY when the
    box has explicitly declared its model store LOCAL & disposable
    (``HUGPY_MODEL_STORE_REAPABLE`` truthy) AND ``rp`` is not on shared/central
    storage. Every other state — the flag unset, a shared-store flag, a central
    sentinel — returns False, so an UNCONFIGURED or shared box never deletes a
    model file. This makes "nothing deletes from the central drive" hold even
    with zero per-box setup: reaping is opt-in, not opt-out."""
    if _on_shared_model_store(rp):
        return False
    return os.environ.get("HUGPY_MODEL_STORE_REAPABLE", "").strip().lower() in ("1", "true", "yes", "on")


def wipe_model(model_key: str) -> bool:
    """Delete the model's local files so the next provision re-downloads it.

    Used by the `redownload` path: a plain provision only fetches when the model
    is MISSING (see model_is_local), so refreshing a corrupt/stale copy requires
    removing it first. Returns True if the path is gone afterwards. Jailed against
    obviously-wrong targets (root/home/short paths) AND against shared/central
    model storage — the operator invariant that no reap may touch the central
    drive."""
    import os
    import shutil
    try:
        from .imports import get_model_path
        path = get_model_path(model_key)
    except Exception:
        return False
    if not path:
        return False
    rp = os.path.realpath(path)
    if len(rp) < 6 or rp in ("/", os.path.expanduser("~")):
        return False  # refuse a dangerous target
    # HARD INVARIANT (single choke point for the reaper AND the redownload path):
    # delete ONLY when the box declared its model store reapable AND the target is
    # not shared/central; never follow a symlink to delete its (shared/operator-
    # managed) target. Safe-by-default: unconfigured or shared boxes refuse.
    if os.path.islink(path) or not _model_store_reapable(rp):
        logger.warning(
            "wipe_model REFUSED for %s (%s): model store not reapable / shared / "
            "symlink — never deleted (set HUGPY_MODEL_STORE_REAPABLE=1 only on "
            "boxes with local disposable model storage).",
            model_key, rp)
        return False
    try:
        if os.path.isdir(rp):
            shutil.rmtree(rp, ignore_errors=True)
        elif os.path.exists(rp):
            os.unlink(rp)
    except OSError:
        pass
    return not os.path.exists(rp)


def _local_destination(meta: dict) -> str:
    """Where this file-set should live on the worker (same layout as central)."""
    from .imports import route_destination

    return route_destination({
        "hub_id": meta.get("hub_id"),
        "name": meta.get("name"),
        "framework": meta.get("framework"),
        "task": meta.get("task"),
        "primary_task": meta.get("task"),
        "filename": meta.get("filename"),
        "include": meta.get("include"),
    })


def _get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        import json
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest_path: str, expected_size: int | None,
                   on_bytes=None) -> None:
    """Stream one file to dest_path, resuming if a partial is already present.

    ``on_bytes(n)`` is called with the number of newly-written bytes per chunk
    so the caller can report download progress.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    have = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    if expected_size is not None and have == expected_size:
        if on_bytes:
            on_bytes(have)   # count the already-present bytes toward progress
        return  # already complete

    req = urllib.request.Request(url)
    if have and expected_size and have < expected_size:
        req.add_header("Range", f"bytes={have}-")
        mode = "ab"
        if on_bytes:
            on_bytes(have)   # resumed: pre-existing bytes already on disk
    else:
        have = 0
        mode = "wb"

    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, mode) as fh:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            if on_bytes:
                on_bytes(len(chunk))


def _download_with_retry(url: str, dest_path: str, expected_size: int | None,
                         on_bytes=None, attempts: int = 4) -> None:
    """Download one file, retrying transient failures with backoff.

    Verifies the on-disk size against ``expected_size`` (when known) and retries
    until it matches, so a truncated/short file never passes as complete. Raises
    the last error if every attempt fails. Progress (``on_bytes``) is reported
    only on the first attempt to avoid double-counting on retry.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            _download_file(url, dest_path, expected_size,
                           on_bytes=on_bytes if i == 0 else None)
            size_ok = (expected_size is None
                       or (os.path.exists(dest_path)
                           and os.path.getsize(dest_path) == expected_size))
            if size_ok and _gguf_header_ok(dest_path):
                return
            if size_ok:
                # Right size, garbage content (preallocated-then-crashed pull).
                # Size alone lies here — and the resumer treats full-size files
                # as complete, so remove the shell before retrying.
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
                last_exc = RuntimeError(
                    f"corrupt GGUF header for {os.path.basename(dest_path)} — "
                    "removed, re-downloading")
            else:
                last_exc = RuntimeError(
                    f"size mismatch for {os.path.basename(dest_path)}: "
                    f"{os.path.getsize(dest_path) if os.path.exists(dest_path) else 0}"
                    f"/{expected_size}")
        except Exception as exc:  # noqa: BLE001 — retry transient network errors
            last_exc = exc
        time.sleep(min(2 ** i, 8))
    raise last_exc or RuntimeError(f"failed to download {url}")


def _gguf_header_ok(path: str) -> bool:
    """Cheap validity check: GGUF files must start with the b'GGUF' magic.

    A crashed multi-connection pull leaves a PREALLOCATED full-size file of
    zeros — it passes the size check forever, and llama.cpp only reveals the
    truth at load time ('invalid magic characters'). Non-.gguf files pass
    unchecked (no cheap magic for them)."""
    if not path.lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def _missing_or_short(dest: str, files: list[dict]) -> list[tuple]:
    """Return [(rel, expected_size, reason)] for files not fully present.

    Side effect: a full-size file with a corrupt GGUF header is UNLINKED here
    (and reported as "corrupt") — the resume logic downstream treats
    right-sized files as complete, so the only way to force a clean re-pull
    is to remove the shell before the download pass sees it."""
    out = []
    for entry in files:
        rel = entry.get("path")
        if not rel:
            continue
        size = entry.get("size")
        target = os.path.join(dest, rel)
        if not os.path.exists(target):
            out.append((rel, size, "absent"))
        elif size is not None and os.path.getsize(target) != size:
            out.append((rel, size, "short"))
        elif not _gguf_header_ok(target):
            try:
                os.remove(target)
            except OSError:
                pass
            out.append((rel, size, "corrupt"))
    return out


def _pull_concurrency() -> int:
    """Max simultaneous connections for a transfer (env HUGPY_PULL_CONCURRENCY)."""
    try:
        return max(1, int(os.environ.get("HUGPY_PULL_CONCURRENCY", "8")))
    except ValueError:
        return 8


# Files bigger than this are split into byte-range segments fetched in parallel,
# so a single multi-GB weights file isn't stuck on one connection.
_SEGMENT_MIN_BYTES = 64 * 1024 * 1024
_SEGMENT_BYTES = 64 * 1024 * 1024


def _supports_range(url: str) -> bool:
    """True if central honours HTTP Range (returns 206) for this file URL."""
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.getcode() == 206
    except Exception:
        return False


def _download_segment(url: str, dest_path: str, start: int, end: int,
                      on_bytes=None) -> None:
    """Fetch one inclusive byte range [start, end] into dest_path at its offset."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.getcode() != 206:
            raise RuntimeError("server ignored Range request")
        remaining = end - start + 1
        with open(dest_path, "r+b") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = resp.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
                if on_bytes:
                    on_bytes(len(chunk))
    if remaining > 0:
        raise RuntimeError(f"short segment {start}-{end} of {dest_path}")


def _download_segment_with_retry(url, dest_path, start, end, on_bytes=None,
                                 attempts: int = 4) -> None:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            _download_segment(url, dest_path, start, end,
                              on_bytes=on_bytes if i == 0 else None)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2 ** i, 8))
    raise last_exc or RuntimeError(f"failed segment {start}-{end} of {url}")


def _segment_ranges(size: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) ranges covering a file, ~_SEGMENT_BYTES each."""
    nseg = max(1, min(_pull_concurrency() * 4, -(-size // _SEGMENT_BYTES)))
    step = -(-size // nseg)  # ceil
    ranges = []
    start = 0
    while start < size:
        end = min(start + step, size) - 1
        ranges.append((start, end))
        start += step
    return ranges


# ── verified chunked transfer ────────────────────────────────────────────────
# Content-verified, crash-safe pulls: central serves per-chunk SHA-256 sums
# (GET .../chunksums), the worker fetches chunk-aligned ranges into a .part
# staging file, hashes each chunk AS IT LANDS, records verified chunks in a
# .state sidecar (so a crash resumes from proven content, not a byte offset),
# and only os.replace()s onto the final name once every chunk verified.
# "Exists under its final name" therefore MEANS complete — the invariant the
# old preallocate-then-size-check scheme couldn't give (a crashed pull left a
# full-size zero shell that passed as present forever).

def _chunk_bytes() -> int:
    try:
        return max(4 * 2**20, min(int(os.environ.get("HUGPY_CHUNK_BYTES", 32 * 2**20)),
                                  256 * 2**20))
    except ValueError:
        return 32 * 2**20


def _fetch_chunksums(base: str, rel: str, chunk_bytes: int) -> list[str] | None:
    """Per-chunk sums from central, or None (older central / any failure —
    the caller falls back to the size+magic scheme)."""
    try:
        d = _get_json(base + "/chunksums?path=" + urllib.parse.quote(rel)
                      + f"&chunk={chunk_bytes}", timeout=590)
        sums = d.get("sums")
        return list(sums) if sums else None
    except Exception as exc:  # noqa: BLE001
        logger.info("no chunksums for %s (%s) — size+magic mode", rel, exc)
        return None


def _load_chunk_state(state_path: str, nchunks: int) -> set[int]:
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            got = json.load(fh).get("verified") or []
        return {i for i in got if isinstance(i, int) and 0 <= i < nchunks}
    except Exception:
        return set()


def _save_chunk_state(state_path: str, verified: set[int]) -> None:
    try:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"verified": sorted(verified)}, fh)
        os.replace(tmp, state_path)
    except OSError:
        pass  # state is an optimization; losing it only costs re-verification


def _fetch_chunk_verified(url: str, part_path: str, index: int, size: int,
                          chunk_bytes: int, want_sha: str, on_bytes=None,
                          attempts: int = 4) -> None:
    """Fetch chunk ``index``, verify its SHA-256, write it at its offset."""
    import hashlib

    start = index * chunk_bytes
    end = min(start + chunk_bytes, size) - 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.getcode() != 206:
                    raise RuntimeError("server ignored Range request")
                buf = resp.read(end - start + 1)
            if len(buf) != end - start + 1:
                raise RuntimeError(f"short chunk {index}: {len(buf)}/{end - start + 1}")
            got = hashlib.sha256(buf).hexdigest()
            if got != want_sha:
                raise RuntimeError(f"chunk {index} hash mismatch")
            with open(part_path, "r+b") as fh:
                fh.seek(start)
                fh.write(buf)
            if on_bytes and i == 0:
                on_bytes(len(buf))
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2 ** i, 8))
    raise last_exc or RuntimeError(f"failed chunk {index} of {url}")


def _ensure_part(part_path: str, size: int) -> None:
    """Preallocate the staging file — only if absent or wrong-sized, so a
    resumed pull never clobbers already-verified chunks."""
    os.makedirs(os.path.dirname(part_path) or ".", exist_ok=True)
    if not os.path.exists(part_path) or os.path.getsize(part_path) != size:
        with open(part_path, "wb") as fh:
            fh.truncate(size)


def fetch_from_central(central_url: str, model_key: str, progress=None) -> bool:
    """Pull a model's ENTIRE directory from central — parallel and segmented.

    Speed comes from two kinds of parallelism over central's existing ``/file``
    endpoint (which supports HTTP Range): small files download concurrently, and
    each large file is split into byte-range segments fetched concurrently, so a
    single multi-GB weights file isn't bottlenecked on one connection. Total
    simultaneous connections are capped at ``HUGPY_PULL_CONCURRENCY`` (default 8).

    Only files not already complete on disk are fetched (file-level resume).
    ``progress(done_bytes, total_bytes, name)`` reports aggregate bytes. Returns
    True only once every manifest file is present at its expected size (verified,
    with re-fetch of any gap); False if central lacks the model (404/409); raises
    if central has it but the transfer can't be completed.
    """
    from concurrent.futures import ThreadPoolExecutor

    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)
    try:
        manifest = _get_json(base + "/manifest")
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 409):
            logger.info("central has no copy of %s (HTTP %s)", model_key, exc.code)
            return False
        raise
    except urllib.error.URLError as exc:
        logger.warning("central unreachable for %s (%s)", model_key, exc)
        return False

    dest = _local_destination(manifest)
    files = manifest.get("files") or []
    total = manifest.get("total_bytes") or sum((e.get("size") or 0) for e in files)
    concurrency = _pull_concurrency()

    # File-level resume: only fetch what isn't already complete. If nothing is
    # pending, this is a pure no-op (not even a Range probe).
    pending = [{"path": r, "size": s} for r, s, _w in _missing_or_short(dest, files)]
    if not pending:
        logger.info("%s already complete on disk (%d files)", model_key, len(files))
        return True

    # Range support lets us segment big files; probe once on the largest pending.
    ranged_ok = False
    biggest = max(pending, key=lambda e: e.get("size") or 0)
    if (biggest.get("size") or 0) >= _SEGMENT_MIN_BYTES:
        ranged_ok = _supports_range(
            base + "/file?path=" + urllib.parse.quote(biggest["path"]))

    logger.info("provisioning %s from central: %d files (%s), %d-way parallel"
                "%s -> %s", model_key, len(files), _human(total), concurrency,
                " (segmented)" if ranged_ok else "", dest)

    done_lock = threading.Lock()
    pstate = {"done": 0, "last": 0.0}

    def _on_bytes(n):
        if not progress:
            return
        with done_lock:
            pstate["done"] += n
            now = time.time()
            if now - pstate["last"] < 0.3 and pstate["done"] < total:
                return
            pstate["last"] = now
            done = pstate["done"]
        progress(min(done, total) if total else done, total, "files")

    # Per-file transfer context for the staged/verified scheme. Everything
    # downloads into <target>.part and is promoted by _finalize only after
    # verification — a crash mid-pull leaves a .part (+ chunk state), never a
    # plausible-looking final file.
    cb = _chunk_bytes()
    ctx: dict = {}   # rel -> {size, part, state, sums, verified, lock}

    def _build_units(entries):
        """Flatten entries into download units, capping total connections.

        Preferred unit is a VERIFIED CHUNK (central supplied per-chunk sums:
        fetch chunk-aligned range → hash → write → record). Fallbacks: plain
        byte-range segments (Range but no sums), then whole-file streaming —
        both still staged to .part and sealed by size+magic at finalize.
        """
        units = []
        for entry in entries:
            rel, size = entry["path"], entry.get("size")
            part = os.path.join(dest, rel) + ".part"
            c = ctx.setdefault(rel, {"lock": threading.Lock()})
            c.update(size=size, part=part, state=None, sums=None, verified=None)
            sums = None
            if ranged_ok and size and size >= _SEGMENT_MIN_BYTES:
                sums = _fetch_chunksums(base, rel, cb)
                if sums is not None and len(sums) != -(-size // cb):
                    logger.warning("chunksums length mismatch for %s (%d vs %d) — "
                                   "falling back to size+magic", rel, len(sums),
                                   -(-size // cb))
                    sums = None
            if sums is not None:
                _ensure_part(part, size)
                state_path = part + ".state.json"
                verified = _load_chunk_state(state_path, len(sums))
                c.update(state=state_path, sums=sums, verified=verified)
                units += [(rel, size, "chunk", i)
                          for i in range(len(sums)) if i not in verified]
            elif ranged_ok and size and size >= _SEGMENT_MIN_BYTES:
                _ensure_part(part, size)
                units += [(rel, size, "segment", se) for se in _segment_ranges(size)]
            else:
                units.append((rel, size, "whole", None))
        return units

    def _run_unit(unit):
        rel, size, kind, arg = unit
        url = base + "/file?path=" + urllib.parse.quote(rel)
        c = ctx[rel]
        try:
            if kind == "whole":
                _download_with_retry(url, c["part"], size, on_bytes=_on_bytes)
            elif kind == "segment":
                _download_segment_with_retry(url, c["part"], arg[0], arg[1],
                                             on_bytes=_on_bytes)
            else:  # verified chunk
                _fetch_chunk_verified(url, c["part"], arg, size, cb,
                                      c["sums"][arg], on_bytes=_on_bytes)
                with c["lock"]:
                    c["verified"].add(arg)
                    _save_chunk_state(c["state"], c["verified"])
        except Exception as exc:  # noqa: BLE001 — gate below re-fetches/decides
            logger.warning("download of %s (%s %s) failed: %s; will re-try in "
                           "verify pass", rel, kind, arg, exc)

    def _finalize(entries):
        """Promote fully-landed .part files onto their final names (atomic)."""
        for entry in entries:
            rel, size = entry["path"], entry.get("size")
            c = ctx.get(rel)
            if not c or not os.path.exists(c["part"]):
                continue
            part = c["part"]
            if size is not None and os.path.getsize(part) != size:
                continue                     # still incomplete — next pass resumes
            if c["sums"] is not None:
                with c["lock"]:
                    if len(c["verified"]) < len(c["sums"]):
                        continue             # unverified chunks remain
            if not _gguf_header_ok(part):
                # Chunk-verified content with a bad header means CENTRAL's copy
                # is corrupt — re-pulling would reproduce it. Fail loud.
                logger.error("%s: staged file fails the GGUF magic check — "
                             "central's copy may itself be corrupt", rel)
                for p in (part, c.get("state")):
                    if p:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                continue
            os.replace(part, os.path.join(dest, rel))
            if c.get("state"):
                try:
                    os.remove(c["state"])
                except OSError:
                    pass

    def _parallel(entries):
        units = _build_units(entries)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(_run_unit, units))
        _finalize(entries)

    # Initial pass over the pending files (computed above).
    _parallel(pending)

    # Completeness gate: re-verify and re-fetch anything that didn't fully land.
    for _ in range(3):
        missing = _missing_or_short(dest, files)
        if not missing:
            break
        logger.warning("central transfer of %s incomplete: %d/%d files "
                       "missing/short; re-fetching", model_key, len(missing), len(files))
        _parallel([{"path": rel, "size": size} for rel, size, _why in missing])

    missing = _missing_or_short(dest, files)
    if missing:
        raise RuntimeError(
            f"central transfer of {model_key} incomplete: "
            f"{len(missing)}/{len(files)} files still missing/short "
            f"(e.g. {missing[0][0]} [{missing[0][2]}]) under {dest}")

    logger.info("provisioned %s from central in full (%d files, %s)",
                model_key, len(files), _human(total))
    return True


class _CountingReader:
    """Wrap a byte stream, reporting bytes read as download progress.

    tarfile in streaming mode pulls from this via ``read``; every pull advances
    a byte counter that reflects the actual network transfer (not file
    boundaries). Emits a throttled progress callback and a throttled journal log
    so a slow-but-moving transfer is visibly distinct from a real stall.
    """

    def __init__(self, fileobj, total, model_key, on_progress=None):
        self._f = fileobj
        self._total = int(total or 0)
        self._model_key = model_key
        self._on_progress = on_progress
        self._done = 0
        self._last_emit = 0.0
        self._last_log = 0.0

    def read(self, size=-1):
        chunk = self._f.read(size)
        if chunk:
            self._done += len(chunk)
            now = time.time()
            done = min(self._done, self._total) if self._total else self._done
            if self._on_progress and now - self._last_emit > 0.5:
                self._last_emit = now
                try:
                    self._on_progress(done, self._total, "archive")
                except Exception:  # progress is best-effort; never break the read
                    pass
            if now - self._last_log > 5.0:
                self._last_log = now
                pct = (100.0 * done / self._total) if self._total else 0.0
                logger.info("downloading %s archive: %s / %s (%.0f%%)",
                            self._model_key, _human(self._done),
                            _human(self._total), pct)
        return chunk


def fetch_archive_from_central(central_url: str, model_key: str, progress=None) -> bool:
    """Pull the model's ENTIRE directory from central as one streamed tar.

    Downloads central's ``/archive`` endpoint and extracts it on the fly (no
    temp tar on disk, bounded memory), confining every member to the model's
    destination, then verifies the result against central's manifest. This is
    the primary transport: one sequential stream can't "drop" files the way N
    independent GETs can.

    Returns True once the directory is present in full. Returns False if central
    can't serve an archive — the endpoint is missing on an older central
    (404/405) or central doesn't have the model (404/409) — so the caller can
    fall back to the per-file transfer. Raises if the archive arrived but
    couldn't be completed (so a partial directory is never reported as success).
    """
    import tarfile

    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)

    # Manifest gives us the destination + the file set to verify against.
    try:
        manifest = _get_json(base + "/manifest")
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 409):
            logger.info("central has no copy of %s (HTTP %s)", model_key, exc.code)
            return False
        raise
    except urllib.error.URLError as exc:
        logger.warning("central unreachable for %s (%s)", model_key, exc)
        return False

    dest = _local_destination(manifest)
    files = manifest.get("files") or []
    total = manifest.get("total_bytes") or sum((e.get("size") or 0) for e in files)
    dest_real = os.path.realpath(dest)
    os.makedirs(dest, exist_ok=True)

    logger.info("provisioning %s from central archive: %d files (%s) -> %s",
                model_key, len(files), _human(total), dest)

    try:
        resp = urllib.request.urlopen(base + "/archive", timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405, 409):
            logger.info("central has no archive endpoint for %s (HTTP %s); "
                        "will use per-file transfer", model_key, exc.code)
            return False
        raise
    except urllib.error.URLError as exc:
        logger.warning("central archive unreachable for %s (%s)", model_key, exc)
        return False

    # Wrap the stream so progress reflects BYTES read off the socket — the true
    # download rate — rather than ticking only when a whole file finishes
    # extracting (which sat at 0% while a multi-GB file streamed). Also logs
    # throughput to the journal so a real stall is visible vs. a slow file.
    reader = _CountingReader(resp, total, model_key, on_progress=progress)

    # mode "r|" = sequential streaming read, matching central's "w|" writer.
    with resp, tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            # LEXICAL containment check (normpath, NOT realpath). realpath
            # follows an EXISTING symlink already sitting at the target — and
            # comfy checkpoints ARE symlinks — so a prior link made realpath
            # resolve outside dest and false-flagged safe bare filenames like
            # 'model.safetensors' as "unsafe path", blocking every comfy pull
            # and driving the reconcile storm. normpath catches real traversal
            # ('..'/absolute) without chasing a link at the destination.
            target = os.path.normpath(os.path.join(dest_real, member.name))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise RuntimeError(f"unsafe path in archive member: {member.name!r}")
            # A symlink/hardlink/device MEMBER *inside* the tar is the genuine
            # traversal threat (central packs only regular files + dirs).
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing non-regular archive member: {member.name!r}")
            # A symlink already at the target (a prior comfy link) must be
            # removed first, or tar.extract writes THROUGH it to its
            # outside-dest target. Replace it with the real bytes.
            if os.path.islink(target) or os.path.isfile(target):
                try:
                    os.unlink(target)
                except OSError:
                    pass
            tar.extract(member, dest)

    if progress:
        progress(total, total, "archive")  # final 100%

    # Completeness gate against the manifest.
    missing = _missing_or_short(dest, files)
    if missing:
        raise RuntimeError(
            f"central archive of {model_key} incomplete after extract: "
            f"{len(missing)}/{len(files)} files missing/short "
            f"(e.g. {missing[0][0]} [{missing[0][2]}]) under {dest}")

    logger.info("provisioned %s from central archive in full (%d files, %s)",
                model_key, len(files), _human(total))
    return True


def _human(n) -> str:
    if not n:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def fetch_from_hf(model_key: str) -> str:
    """Last-resort: pull from Hugging Face via the normal code path."""
    from .imports import ensure_model

    logger.info("provisioning %s from Hugging Face", model_key)
    return ensure_model(model_key)


def ensure_model_present(model_key: str, central_url: str | None, progress=None) -> bool:
    """Make sure model_key is on local disk. Central-first, then HF fallback.

    ``progress(done_bytes, total_bytes, filename)`` is forwarded to the central
    download so callers can stream provisioning status. Returns True if the
    model is present (or already was), False if it could not be provisioned.
    """
    # Teach the worker about the model first (central is the source of truth),
    # then provision against the canonical local key. This is what lets the
    # worker serve a model it wasn't built with.
    canonical = ensure_model_registered(model_key, central_url) or model_key

    if model_is_local(canonical):
        return True

    # Single-flight: serialize provisioning of this model so concurrent callers
    # (multiple infer requests + the pre-provision) don't each download it in
    # parallel into the same directory.
    lock = _provision_lock(canonical)
    if not lock.acquire(blocking=False):
        logger.info("provisioning of %s already in progress; waiting for it",
                    canonical)
        lock.acquire()
    try:
        # Another thread may have finished the download while we waited.
        if model_is_local(canonical):
            logger.info("%s became available while waiting; using it", canonical)
            return True
        return _provision_now(canonical, central_url, progress=progress)
    finally:
        lock.release()


def _provision_now(canonical: str, central_url: str | None, progress=None) -> bool:
    """Do the actual fetch (central archive -> per-file -> HF). Caller holds the
    per-model provisioning lock."""
    # Priority: get the model FILES from CENTRAL first — it's the source of
    # truth and needs no HF token. Hugging Face is only a fallback, used when
    # central can't provide the files (no central URL, central unreachable, or
    # central doesn't have them on disk).
    # Daylight item 4: the chosen SOURCE must be loud — logged AND streamed
    # through the progress callback (a "source=…" marker the agent stores on
    # provision_progress), so a silent 55GB HF pull while central held the
    # files can never happen unnoticed again.
    central_reason = "no central URL configured"
    if central_url:
        # 1) parallel + segmented per-file transfer — fastest (saturates the
        #    link; a big weights file is split across many connections).
        if progress:
            progress(0, 0, "source=central")
        try:
            if fetch_from_central(central_url, canonical, progress=progress):
                logger.info("PROVENANCE: %s provisioned from CENTRAL (parallel)",
                            canonical)
                return True
            central_reason = "central does not have the files (per-file 409/empty)"
        except Exception as exc:
            central_reason = f"parallel transfer failed: {type(exc).__name__}: {exc}"
            logger.warning("central parallel transfer of %s failed: %s; "
                           "trying archive", canonical, exc)
        # 2) whole-directory tar stream — single-connection fallback.
        try:
            if fetch_archive_from_central(central_url, canonical, progress=progress):
                logger.info("PROVENANCE: %s provisioned from CENTRAL (archive)",
                            canonical)
                return True
            central_reason = "central cannot provide the files (archive refused)"
        except Exception as exc:
            central_reason = f"archive transfer failed: {type(exc).__name__}: {exc}"

    logger.warning("PROVENANCE: %s falling back to HUGGING FACE — central "
                   "rejected: %s", canonical, central_reason)
    try:
        if progress:
            progress(0, 0, f"source=hf ({central_reason[:120]})")
        fetch_from_hf(canonical)
        logger.warning("PROVENANCE: %s provisioned from HUGGING FACE (central "
                       "rejected: %s)", canonical, central_reason)
        return True
    except Exception as exc:
        logger.error("could not provision %s from central or HF: %s "
                     "(central rejected: %s)", canonical, exc, central_reason)
        return False

