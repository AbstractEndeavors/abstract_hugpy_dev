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
    try:
        info = hfApi.model_info(repo_id, files_metadata=True)
    except Exception as exc:
        logger.info("size estimate failed for %s: %s", hub_id, exc)
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

    total = sum((s.size or 0) for s in (info.siblings or []) if will_download(s.rfilename))
    return total or None


# ──────────────────────────────────────────────────────────────────────────
# Subprocess worker — module-level so it's spawn-safe. Captures the real
# failure reason (HF errors propagate out of download_one) into the error file,
# then re-raises so the process exits non-zero and the monitor sees the failure.
# ──────────────────────────────────────────────────────────────────────────
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


def _is_cancelled(job_id: str) -> bool:
    cur = job_store.get(job_id)
    return bool(cur and cur.status == "cancelled")


def _watch(proc, job_id: str, dest: str, total_bytes: int | None) -> bool:
    """Sample progress every second while ``proc`` runs.

    Reports bytes/sec and percentage. Returns True if the transfer STALLED
    (no new bytes for STALL_SECONDS) — in which case the process group is
    killed so it can be resumed — or False if the process exited on its own.
    """
    last_bytes = _dir_bytes(dest)
    last_change = time.time()
    prev_bytes, prev_t = last_bytes, last_change

    while proc.is_alive():
        time.sleep(1.0)
        if _is_cancelled(job_id):
            return False
        now = time.time()
        got = _dir_bytes(dest)
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
        p = mp.Process(target=_download_worker, args=(job.id, job.model_key, model), daemon=True)
        p.start()
        job_store.update(job.id, _proc=p)
        return p

    def monitor() -> None:
        nonlocal total_bytes
        if total_bytes is None:
            total_bytes = _estimate_total_bytes(model)
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
                    downloaded_bytes=_dir_bytes(dest), error=None,
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
    if job.status not in ("queued", "running"):
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
    if job.status in ("queued", "running"):
        return {"retried": False, "reason": f"job is already {job.status}"}
    model = getattr(job, "_model", None)
    if not model:
        return {"retried": False, "reason": "no model context to resume from"}
    start_cancellable_download(job, model, total_bytes=job.total_bytes)
    return {"retried": True, "id": job_id}
