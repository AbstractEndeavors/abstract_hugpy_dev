"""Process-wide async runtime — ONE long-lived event loop in a daemon thread.

Replaces the per-request ``asyncio.new_event_loop()`` pattern every SSE / one-shot
endpoint used. That pattern caused two problems:

  * Loop-binding crashes — an asyncio sync primitive (Semaphore/Lock/Event)
    cached on a process singleton binds to the FIRST request's loop and then
    raises "bound to a different event loop" on the next request. With one
    persistent loop, cached primitives stay valid for the life of the process.
  * Per-request loop churn — creating/closing a loop per request and pinning a
    thread in run_until_complete for the whole stream. The shared loop interleaves
    many streams cooperatively; blocking model work runs in the default executor
    (asyncio.to_thread), so the loop stays responsive.

Sync callers submit coroutines via ``run()`` / ``iter_sync()``; the loop runs them
and the caller blocks on a ``concurrent.futures.Future``. All entry points are
thread-safe (``run_coroutine_threadsafe``). Usable from both central (gunicorn
threads) and the worker agent (its request threads).
"""
from __future__ import annotations

import asyncio
import threading
import logging
import concurrent.futures as _cf

logger = logging.getLogger(__name__)

_loop: "asyncio.AbstractEventLoop | None" = None
_thread: "threading.Thread | None" = None
_start_lock = threading.Lock()


def loop() -> "asyncio.AbstractEventLoop":
    """The shared event loop, starting its daemon thread on first use."""
    global _loop, _thread
    lp = _loop
    if lp is not None and lp.is_running():
        return lp
    with _start_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        lp = asyncio.new_event_loop()
        ready = threading.Event()

        def _run():
            asyncio.set_event_loop(lp)
            ready.set()
            lp.run_forever()

        t = threading.Thread(target=_run, name="hugpy-async-runtime", daemon=True)
        t.start()
        ready.wait(5)
        _loop, _thread = lp, t
        logger.info("async runtime started (thread=%s)", t.name)
        return _loop


def submit(coro) -> "_cf.Future":
    """Schedule a coroutine on the shared loop; return its concurrent Future."""
    return asyncio.run_coroutine_threadsafe(coro, loop())


def run(coro):
    """Run a coroutine on the shared loop from a sync thread; block for its result."""
    if not asyncio.iscoroutine(coro):
        # Tolerate already-resolved values (callers that may pass a plain result).
        return coro
    return submit(coro).result()


def call_soon_threadsafe(callback, *args) -> None:
    """Schedule a plain callback on the shared loop (e.g. ``Event.set`` from
    another thread, which is otherwise unsafe to call cross-loop)."""
    loop().call_soon_threadsafe(callback, *args)


def iter_sync(agen, heartbeat: "bytes | None" = None, heartbeat_secs: float = 15.0):
    """Drive an async generator from a sync (WSGI) thread on the SHARED loop.

    Mirrors the old per-request driver semantics:
      * With ``heartbeat`` bytes, each step waits at most ``heartbeat_secs`` and
        yields the keepalive on timeout while the SAME step keeps running — so a
        slow first token can't trip an upstream proxy, and every keepalive write
        lets the WSGI server notice a dead client and trigger teardown.
      * ``heartbeat=None`` blocks for each real event (internal/worker drains).
      * On teardown the in-flight step is cancelled, then ``aclose()`` cascades
        GeneratorExit through every ``async for`` / ``async with`` so a relayed
        worker's httpx stream is released rather than leaked.
    """
    lp = loop()
    fut = None
    try:
        while True:
            if fut is None:
                fut = asyncio.run_coroutine_threadsafe(agen.__anext__(), lp)
            try:
                item = fut.result(heartbeat_secs if heartbeat is not None else None)
                fut = None
            except _cf.TimeoutError:
                # Next event still cooking — keep the connection warm, keep
                # awaiting the SAME step (it's still running on the loop).
                yield heartbeat
                continue
            except StopAsyncIteration:
                fut = None
                break
            if isinstance(item, str):
                item = item.encode("utf-8")
            yield item
    finally:
        try:
            if fut is not None and not fut.done():
                # Cancel the in-flight step first: CancelledError unwinds the
                # chain's `async with` blocks (closing the worker httpx stream),
                # after which aclose() can finalize without "already running".
                fut.cancel()
                try:
                    fut.result(5)
                except BaseException:
                    pass
            closer = asyncio.run_coroutine_threadsafe(agen.aclose(), lp)
            try:
                closer.result(10)
            except BaseException:
                pass
        except Exception:
            pass
