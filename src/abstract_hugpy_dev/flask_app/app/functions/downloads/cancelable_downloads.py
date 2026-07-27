import multiprocessing as mp
import tempfile
from datetime import datetime, timezone
from flask import jsonify, abort
from .imports import *
from .downloader import *
# ──────────────────────────────────────────────────────────────────────────
# Tunables (env-overridable). A download that writes no new bytes for
# STALL_SECONDS is considered stalled and gets killed + resumed. Each download
# is attempted up to MAX_ATTEMPTS times; HF keeps partial files on disk so a
# resume picks up where the previous attempt stopped.
# ──────────────────────────────────────────────────────────────────────────
STALL_SECONDS = int(os.environ.get("HUGPY_DOWNLOAD_STALL_SECONDS", "180"))
MAX_ATTEMPTS  = int(os.environ.get("HUGPY_DOWNLOAD_MAX_ATTEMPTS", "4"))


# ──────────────────────────────────────────────────────────────────────────
# Error hand-off across the process boundary — the download runs in a child
# process, so it writes its failure reason to a temp file the monitor reads.
# ──────────────────────────────────────────────────────────────────────────
def _error_path(job_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"hugpy-download-{job_id}.err")


def _write_error(job_id: str, msg: str) -> None:
    try:
        with open(_error_path(job_id), "w", encoding="utf-8") as fh:
            fh.write(msg[:2000])
    except OSError:
        pass


def _read_error(job_id: str) -> str | None:
    try:
        with open(_error_path(job_id), "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _clear_error(job_id: str) -> None:
    try:
        os.remove(_error_path(job_id))
    except OSError:
        pass


def update_model_status(model: dict) -> dict:
    model.update(model_status(model))
    return model


def _estimate_total_bytes(model: dict) -> int | None:
    """Sum the sizes of exactly the files this download will fetch, so the
    progress bar can show a real percentage. Respects filename (single GGUF),
    include patterns, or full repo. Returns None on any failure -> the bar
    falls back to indeterminate, which still works."""
    hub_id = model.get("hub_id")
    if not hub_id:
        return None
    repo_id, _ = split_hub_id(hub_id)
    # Per-repo metadata rides the permanent central HF cache (fetch-once —
    # comms/model_metadata.py): only the first estimate of a repo ever pings HF.
    from abstract_hugpy_dev.comms.model_metadata import fetch_repo_info
    try:
        payload = fetch_repo_info(repo_id, files_metadata=True, api=hfApi)
    except Exception as exc:
        logger.info("size estimate failed for %s: %s", hub_id, exc)
        return None
    if payload is None:
        return None

    filename = model.get("filename")
    include = model.get("include")

    def will_download(path: str) -> bool:
        if filename:
            return path == filename or path.endswith("/" + filename)
        if include:
            pats = include if isinstance(include, list) else [include]
            return any(fnmatch.fnmatch(path, p) for p in pats)
        return True

    total = sum((s.get("size") or 0) for s in (payload.get("siblings") or [])
                if will_download(s.get("rfilename") or ""))
    return total or None


# ──────────────────────────────────────────────────────────────────────────
# Subprocess worker — module-level so it's spawn-safe. Captures the real
# failure reason (HF errors propagate out of download_one) into the error file,
# then re-raises so the process exits non-zero and the monitor sees the failure.
# ──────────────────────────────────────────────────────────────────────────
# How long the (network) size estimate may take before the download proceeds
# with an indeterminate progress bar. Deliberately short: the estimate is a
# NICETY (it only turns the bar from indeterminate into a percentage), while the
# thread it occupies is shared with heartbeat handling.
_ESTIMATE_TIMEOUT_S = float(os.environ.get("HUGPY_HF_ESTIMATE_TIMEOUT_S", "8") or 8)


def _estimate_total_bytes_bounded(model: dict) -> "int | None":
    """_estimate_total_bytes with a hard wall-clock bound.

    The underlying HfApi call cannot be interrupted, so the worker thread it
    occupies is left to finish on its own (daemon, so it can never block
    shutdown) — what we bound is how long WE wait for it. Returns None on
    timeout: an indeterminate bar, which is strictly better than a stalled API.
    """
    # A bare daemon thread, NOT ThreadPoolExecutor: the executor's context
    # manager JOINS its workers on __exit__, so `with … as ex` waits for the hung
    # call anyway and the timeout buys nothing. Verified the hard way — a 2s
    # bound over a 60s call still returned at 60s. Simply ceasing to wait on a
    # daemon thread is what actually releases us; it dies with the process.
    import threading as _th
    box = {}

    def _run():
        try:
            box["v"] = _estimate_total_bytes(model)
        except Exception as exc:  # noqa: BLE001
            box["e"] = exc

    t = _th.Thread(target=_run, name="hf-estimate", daemon=True)
    t.start()
    t.join(_ESTIMATE_TIMEOUT_S)
    if t.is_alive():
        logger.warning(
            "HF size estimate exceeded %.0fs — continuing with an indeterminate "
            "progress bar (the download itself is unaffected)",
            _ESTIMATE_TIMEOUT_S)
        return None
    if "e" in box:
        logger.info("HF size estimate unavailable (%s)", box["e"])
        return None
    return box.get("v")


def _download_worker(job_id: str, model_key: str, model: dict) -> None:
    os.setpgrp()
    try:
        download_one(model=model, model_key=model_key)   # writes hugpy.json via _stamp
        _clear_error(job_id)
    except Exception as exc:
        _write_error(job_id, f"{type(exc).__name__}: {exc}")
        raise


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _progress_bytes(dest: str) -> int:
    """Bytes on disk for a download IN PROGRESS: the final `dest` (once
    promoted) PLUS any still-live `.tmp-<pid>` staging sibling(s) — atomic
    provisioning (imports/apis/download_models.py) lands new work in staging
    and only renames it onto `dest` on completion, so `_dir_bytes(dest)` alone
    always read 0 (0%) for the entire in-flight window (operator-felt
    regression, fixed 2026-07-12: a staging dir was seen growing 3.7GB->4.4GB
    in 3s while the console showed 0%). Safe to sum both — see
    download_models.staged_bytes' docstring for why this never double-counts."""
    from .....imports.apis.download_models import staged_bytes
    return _dir_bytes(dest) + staged_bytes(dest)


def _is_cancelled(job_id: str) -> bool:
    cur = job_store.get(job_id)
    return bool(cur and cur.status == "cancelled")


def _watch(proc, job_id: str, dest: str, total_bytes: int | None) -> bool:
    """Sample progress every second while ``proc`` runs.

    Reports bytes/sec and percentage. Returns True if the transfer STALLED
    (no new bytes for STALL_SECONDS) — in which case the process group is
    killed so it can be resumed — or False if the process exited on its own.
    """
    last_bytes = _progress_bytes(dest)
    last_change = time.time()
    prev_bytes, prev_t = last_bytes, last_change

    while proc.is_alive():
        time.sleep(1.0)
        if _is_cancelled(job_id):
            return False
        now = time.time()
        got = _progress_bytes(dest)
        bps = max(got - prev_bytes, 0) / max(now - prev_t, 1e-6)
        prev_bytes, prev_t = got, now
        if got > last_bytes:
            last_bytes, last_change = got, now
        pct = (got / total_bytes) if total_bytes else 0.0
        job_store.update(job_id, progress=min(pct, 0.999),
                         downloaded_bytes=got, bytes_per_second=bps, stalled=False)

        if (now - last_change) >= STALL_SECONDS:
            job_store.update(job_id, stalled=True)
            from ....._platform.procutil import terminate_tree
            terminate_tree(proc)
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Launch: spawn the worker under a monitor that auto-resumes a stalled/failed
# transfer with backoff, surfaces the real error, and resolves the terminal
# state. A user cancel at any point (status -> cancelled) stops the loop.
# ──────────────────────────────────────────────────────────────────────────
def start_cancellable_download(job: Job, model: dict, total_bytes: int | None = None) -> None:
    dest = route_destination(model=model)
    logger.info("download -> %s", dest)

    job_store.update(
        job.id, status="running", message="Downloading…",
        total_bytes=total_bytes, attempt=1, max_attempts=MAX_ATTEMPTS,
        stalled=False, error=None, _model=model,
    )

    def _spawn():
        _clear_error(job.id)
        # SPAWN, NOT FORK (operator report 2026-07-27: "a download from hf in the
        # add models tab seems to push off all of the workers and makes the api
        # unstable").
        #
        # mp.Process defaults to FORK on Linux, and this runs inside a gunicorn
        # worker configured `--workers 3 --threads 8`. Forking a multi-threaded
        # process copies the memory image but only the CALLING thread, so any
        # lock another thread happened to hold is inherited LOCKED and can never
        # be released. The worker registry is exactly such a lock:
        # workers.json is guarded by fcntl.flock(LOCK_EX) and EVERY heartbeat
        # takes it. Fork mid-transaction and heartbeat handling stalls; past the
        # 45s HEARTBEAT_TIMEOUT_SECONDS every worker reads `offline` — the whole
        # fleet "pushed off" by one download. The child also inherits the parent's
        # HTTP/CUDA/SSL state, which is its own class of instability.
        #
        # `spawn` starts a clean interpreter: no inherited locks, no shared fds,
        # no half-copied thread state. It costs an interpreter start (~0.3s) per
        # download, which is irrelevant next to the transfer itself, and the child
        # already takes only picklable args (job id, model_key, a plain dict), so
        # nothing here depended on inherited memory.
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_download_worker,
                        args=(job.id, job.model_key, model), daemon=True)
        p.start()
        job_store.update(job.id, _proc=p)
        return p

    def monitor() -> None:
        nonlocal total_bytes
        if total_bytes is None:
            # SIZE ESTIMATE IS A NETWORK CALL — keep it OFF the request path and
            # bounded (operator, 2026-07-27: "a download from hf in the add
            # models tab seems to push off all of the workers and makes the api
            # unstable ... it just needs to take a path to download that doesn't
            # affect the api").
            #
            # _estimate_total_bytes -> comms.model_metadata.fetch_repo_info ->
            # HfApi.model_info(files_metadata=True): a blocking HTTPS round-trip
            # to huggingface.co, with no timeout of its own, for the FULL file
            # manifest of the repo. It runs here purely to make the progress bar
            # show a percentage. When HF is slow or unreachable it parks this
            # thread indefinitely — and with gunicorn at --workers 3 --threads 8
            # a handful of parked threads starve the pool that also serves
            # /llm/workers/<id>/heartbeat. Miss 45s of heartbeats
            # (HEARTBEAT_TIMEOUT_SECONDS) and every worker reads `offline`:
            # exactly "pushes off all of the workers".
            #
            # It is already on a daemon thread rather than the request itself, so
            # the download STARTS promptly; the hazard is the thread it holds.
            # Bounded by a watchdog: if the estimate has not landed in
            # _ESTIMATE_TIMEOUT_S we proceed WITHOUT a total (an indeterminate
            # bar, which the UI already handles) rather than let one slow HF call
            # hold a thread for the life of the download.
            total_bytes = _estimate_total_bytes_bounded(model)
            if total_bytes:
                job_store.update(job.id, total_bytes=total_bytes)

        attempt = 1
        while True:
            if attempt > 1:
                job_store.update(
                    job.id, attempt=attempt, status="running", stalled=False,
                    message=f"Resuming (attempt {attempt}/{MAX_ATTEMPTS})…",
                )
            proc = _spawn()
            stalled = _watch(proc, job.id, dest, total_bytes)
            proc.join()

            if _is_cancelled(job.id):
                return

            if not stalled and proc.exitcode == 0:
                job_store.update(
                    job.id, status="completed", progress=1.0, stalled=False,
                    downloaded_bytes=_progress_bytes(dest), error=None,
                    bytes_per_second=None, message=f"Installed at {dest}",
                )
                try:
                    record_downloaded_model(model, dest)
                    refresh_registry(run_discovery=False)
                except Exception as exc:
                    logger.warning("post-download registry refresh failed: %s", exc)
                return

            # Failed or stalled — figure out why, then resume or give up.
            detail = _read_error(job.id) or (
                f"stalled: no new data for {STALL_SECONDS}s"
                if stalled else f"worker exited with code {proc.exitcode}"
            )
            if attempt >= MAX_ATTEMPTS:
                job_store.update(
                    job.id, status="failed", stalled=stalled, bytes_per_second=None,
                    message="Download stalled." if stalled else "Download failed.",
                    error=detail,
                )
                return

            backoff = min(2 ** attempt, 30)
            job_store.update(
                job.id, status="running", stalled=stalled, error=detail,
                message=(f"{'Stalled' if stalled else 'Error'}; retrying in {backoff}s "
                         f"(attempt {attempt + 1}/{MAX_ATTEMPTS})…"),
            )
            for _ in range(backoff):
                if _is_cancelled(job.id):
                    return
                time.sleep(1.0)
            attempt += 1

    threading.Thread(target=monitor, daemon=True).start()


def cancel_download(job_id: str) -> dict:
    job = job_store.get(job_id)
    if not job:
        abort(404, description="Unknown job ID.")
    if job.terminal:
        return {"cancelled": False, "reason": f"job is {job.status}"}

    # Set status FIRST so the monitor's auto-resume loop sees the cancel and
    # won't relaunch after we kill the current attempt.
    job_store.update(job_id, status="cancelled", message="Cancelled by user.",
                     stalled=False, bytes_per_second=None)

    proc = getattr(job, "_proc", None)
    if proc is not None and proc.is_alive():
        from ....._platform.procutil import terminate_tree
        terminate_tree(proc)
    return {"cancelled": True}


def retry_download(job_id: str) -> dict:
    """Resume a failed/cancelled download from where it stopped.

    Reuses the same job id and the model context captured at first launch, so
    partial files already on disk are continued (HF resumes), not re-fetched.
    """
    job = job_store.get(job_id)
    if not job:
        abort(404, description="Unknown job ID.")
    if not job.terminal:
        return {"retried": False, "reason": f"job is already {job.status}"}
    model = getattr(job, "_model", None)
    if not model:
        return {"retried": False, "reason": "no model context to resume from"}
    start_cancellable_download(job, model, total_bytes=job.total_bytes)
    return {"retried": True, "id": job_id}
