"""Per-model install-status memo — the /v1/models availability fix (2026-07-27).

Live incident: central was unusable. ``/health`` answered in 0.0009s and
``/llm/workers`` in 2.0s, but ``/v1/models`` took 18.5s, then 25.7s, then 55.3s
— degrading — while established connections to :7002 climbed 47 -> 140 against
only 24 gunicorn slots (``--workers 3 --threads 8``). Thread wchans across all
three workers read ``request_wait_answer`` (FUSE/virtiofs), ``folio_wait_bit_common``
(disk) and ``locks_lock_inode_wait``: the API was not computing, it was queued
on I/O.

Cause: ``v1_models`` (and ``list_models``) loop the whole manifest and call
``update_model_status(model)`` per model. That delegates to ``model_status``,
which walks the store — ``route_destination`` globs four runtime families'
legacy task dirs and stats every candidate, then ``model_looks_downloaded``
globs the winner. ~10^2 filesystem calls per model, ~107 models, every call a
virtiofs round-trip. The manifest itself was already cached; the per-model
status stat was not. Installation status only changes on download / delete /
prune / reconcile / discovery, so re-walking per request was pure waste.

This suite locks the fix (``comms/model_status_cache.py`` + its call sites):

  * same VALUES cached vs uncached — the listing content never changes shape;
  * a second listing does far fewer status reads (counted stub, not wall time);
  * TTL expiry re-reads;
  * every invalidation event forces a re-read — download completed / failed /
    cancelled, model delete, prune, reconcile apply, and refresh_registry (the
    chokepoint that covers discovery too);
  * concurrent callers single-flight instead of stampeding the walk;
  * a failure in the cache machinery degrades to the LIVE stat, and an error
    raised by the live stat still propagates (never an invented status);
  * a sibling PROCESS's invalidation is picked up (the gunicorn --workers 3
    case) via the local epoch file.

Runs under pytest:
    venv/bin/python -m pytest tests/test_model_status_cache.py -q
"""
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The memo publishes a cross-process invalidation token to a file. Point it at a
# per-run temp path BEFORE anything imports/uses the module, so a test run can
# never stomp (or be stomped by) the epoch a live central is sharing between its
# gunicorn workers.
_EPOCH_DIR = tempfile.mkdtemp(prefix="hugpy-status-cache-test-")
os.environ["HUGPY_MODEL_STATUS_EPOCH_PATH"] = os.path.join(_EPOCH_DIR, "epoch")
os.environ.pop("HUGPY_MODEL_STATUS_TTL_S", None)
os.environ.pop("HUGPY_MODEL_STATUS_EPOCH_POLL_S", None)

cache = importlib.import_module("abstract_hugpy_dev.comms.model_status_cache")
from abstract_hugpy_dev.comms.jobs import normalize_status  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _model(key="repo-0", **over):
    m = {
        "model_key": key,
        "hub_id": f"org/{key}",
        "name": key,
        "framework": "gguf",
        "primary_task": "text-generation",
        "tasks": ["text-generation"],
    }
    m.update(over)
    return m


class CountingStat:
    """Stand-in for ``model_status`` that counts how often the store is read."""

    def __init__(self, status="installed", destination="/store/x"):
        self.calls = 0
        self.status = status
        self.destination = destination
        self.lock = threading.Lock()
        self.delay = 0.0

    def __call__(self, model):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {"status": self.status,
                "destination": f"{self.destination}/{model.get('model_key')}",
                "installed_marker": f"{self.destination}/{model.get('model_key')}/hugpy.json"}


@pytest.fixture(autouse=True)
def _clean_cache():
    cache.reset_model_status_cache()
    for var in ("HUGPY_MODEL_STATUS_TTL_S", "HUGPY_MODEL_STATUS_EPOCH_POLL_S",
                "HUGPY_MODEL_STATUS_LOCK_WAIT_S"):
        os.environ.pop(var, None)
    yield
    cache.reset_model_status_cache()
    for var in ("HUGPY_MODEL_STATUS_TTL_S", "HUGPY_MODEL_STATUS_EPOCH_POLL_S",
                "HUGPY_MODEL_STATUS_LOCK_WAIT_S"):
        os.environ.pop(var, None)


# ──────────────────────────────────────────────────────────────────────────
# 1. the memo itself
# ──────────────────────────────────────────────────────────────────────────
def test_same_value_cached_and_uncached():
    """The cached answer is byte-identical to the live one."""
    live = CountingStat()
    m = _model()
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "0"          # cache off
    uncached = cache.cached_model_status(m, live)
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "30"
    cold = cache.cached_model_status(m, live)
    warm = cache.cached_model_status(m, live)
    assert uncached == cold == warm
    assert live.calls == 2                                 # off + cold, not warm


def test_second_listing_does_far_fewer_status_reads():
    """A whole-manifest walk pays once; the next one pays nothing."""
    live = CountingStat()
    manifest = [_model(f"repo-{i}") for i in range(107)]

    first = [cache.cached_model_status(m, live) for m in manifest]
    after_first = live.calls
    second = [cache.cached_model_status(m, live) for m in manifest]
    after_second = live.calls

    assert after_first == 107, "cold walk must read every model exactly once"
    assert after_second == 107, "a warm listing must read the store ZERO times"
    assert first == second


def test_returned_dict_is_a_copy_callers_cannot_poison():
    """Listings do ``model.update(status)`` and mutate rows in place."""
    live = CountingStat()
    m = _model()
    got = cache.cached_model_status(m, live)
    got["status"] = "not_installed"
    got["injected"] = True
    again = cache.cached_model_status(m, live)
    assert again["status"] == "installed"
    assert "injected" not in again


def test_ttl_expiry_re_reads():
    live = CountingStat()
    m = _model()
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "0.25"
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1
    time.sleep(0.35)
    cache.cached_model_status(m, live)
    assert live.calls == 2, "an expired entry must be re-read from the store"


def test_ttl_zero_disables_the_cache_entirely():
    live = CountingStat()
    m = _model()
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "0"
    for _ in range(5):
        cache.cached_model_status(m, live)
    assert live.calls == 5


def test_a_routing_change_is_a_different_entry():
    """Re-key a model (new hub_id/filename/framework) and the old memo can't
    answer for it — the key IS the routing identity the live stat reads."""
    live = CountingStat()
    base = _model()
    cache.cached_model_status(base, live)
    assert live.calls == 1
    for field, value in (("hub_id", "org/other"), ("filename", "x.gguf"),
                         ("framework", "transformers"), ("dir", "/elsewhere"),
                         ("include", ["*.gguf"]), ("tasks", ["image-to-image"])):
        before = live.calls
        cache.cached_model_status(_model(**{field: value}), live)
        assert live.calls == before + 1, f"{field} must change the cache key"


def test_scopes_do_not_collide():
    """Two questions about the SAME model must not share an entry."""
    status = CountingStat(status="installed")
    holdings = CountingStat(status="ready")
    m = _model()
    a = cache.cached_model_status(m, status)
    b = cache.cached_model_status(m, holdings, scope="central-holdings")
    assert a["status"] == "installed"
    assert b["status"] == "ready"
    assert status.calls == 1 and holdings.calls == 1
    assert cache.cached_model_status(m, status)["status"] == "installed"
    assert cache.cached_model_status(
        m, holdings, scope="central-holdings")["status"] == "ready"
    assert status.calls == 1 and holdings.calls == 1


def test_invalidation_forces_a_re_read():
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1
    cache.invalidate_model_status("test")
    cache.cached_model_status(m, live)
    assert live.calls == 2


def test_refresh_always_reads_live_and_seeds_the_memo():
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    assert live.calls == 1
    cache.refresh_model_status(m, live)                    # explicit refresh
    assert live.calls == 2, "an explicit refresh must not be served from cache"
    cache.cached_model_status(m, live)
    assert live.calls == 2, "…and it must leave the memo repaired"


def test_concurrent_callers_do_not_stampede():
    """32 threads asking for the same model share ONE walk."""
    live = CountingStat()
    live.delay = 0.15                                      # a slow store read
    m = _model()
    results, errors = [], []

    def worker():
        try:
            results.append(cache.cached_model_status(dict(m), live))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors
    assert len(results) == 32
    assert all(r == results[0] for r in results)
    assert live.calls == 1, (
        f"single-flight broken: {live.calls} threads each walked the store")


def test_concurrent_callers_across_models_still_walk_each_model_once():
    live = CountingStat()
    live.delay = 0.01
    manifest = [_model(f"repo-{i}") for i in range(20)]
    errors = []

    def worker():
        try:
            for m in manifest:
                cache.cached_model_status(dict(m), live)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    assert live.calls == 20, f"expected one walk per model, got {live.calls}"


# ──────────────────────────────────────────────────────────────────────────
# 2. degrade-not-guess
# ──────────────────────────────────────────────────────────────────────────
def test_cache_machinery_failure_falls_back_to_the_live_stat(monkeypatch):
    live = CountingStat()
    m = _model()

    def boom(*_a, **_k):
        raise RuntimeError("key computation exploded")

    monkeypatch.setattr(cache, "status_key", boom)
    got = cache.cached_model_status(m, live)
    assert got["status"] == "installed"
    assert live.calls == 1
    # and it keeps working — never a poisoned entry, just no caching
    cache.cached_model_status(m, live)
    assert live.calls == 2
    assert cache.cache_stats()["errors"] >= 1


def test_a_store_read_error_propagates_exactly_as_today():
    """The live stat raising is TODAY's behaviour; the memo must not swallow it
    and must never invent a status in its place."""
    calls = {"n": 0}

    def exploding(_model):
        calls["n"] += 1
        raise OSError("virtiofs went away")

    with pytest.raises(OSError):
        cache.cached_model_status(_model(), exploding)
    assert calls["n"] == 1
    # nothing was memoized, so the next caller retries the store
    with pytest.raises(OSError):
        cache.cached_model_status(_model(), exploding)
    assert calls["n"] == 2


def test_a_non_dict_status_is_never_memoized():
    calls = {"n": 0}

    def weird(_model):
        calls["n"] += 1
        return None

    assert cache.cached_model_status(_model(), weird) is None
    assert cache.cached_model_status(_model(), weird) is None
    assert calls["n"] == 2


def test_unwritable_epoch_degrades_to_ttl_only(monkeypatch):
    """A /tmp we cannot write must not break invalidation locally."""
    monkeypatch.setenv("HUGPY_MODEL_STATUS_EPOCH_PATH",
                       "/proc/definitely/not/writable/epoch")
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.invalidate_model_status("unwritable-epoch")       # must not raise
    cache.cached_model_status(m, live)
    assert live.calls == 2


# ──────────────────────────────────────────────────────────────────────────
# 3. cross-process (gunicorn --workers 3)
# ──────────────────────────────────────────────────────────────────────────
def test_a_sibling_process_invalidation_is_picked_up():
    """Process A memoizes; process B deletes a model and invalidates; A must
    stop serving the stale entry — inside the epoch poll, not the TTL."""
    os.environ["HUGPY_MODEL_STATUS_EPOCH_POLL_S"] = "0"     # read every lookup
    os.environ["HUGPY_MODEL_STATUS_TTL_S"] = "3600"         # TTL can't be the reason
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1

    code = (
        "import os,sys;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r});"
        "import importlib;"
        "c=importlib.import_module('abstract_hugpy_dev.comms.model_status_cache');"
        "c.invalidate_model_status('sibling-process')"
    )
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    cache.cached_model_status(m, live)
    assert live.calls == 2, (
        "a sibling gunicorn worker's invalidation was not observed")


def test_first_epoch_read_does_not_spuriously_clear():
    """Adopting a token we have never seen must not throw away a warm memo for
    no reason (a fresh process starts empty anyway)."""
    os.environ["HUGPY_MODEL_STATUS_EPOCH_POLL_S"] = "0"
    cache.invalidate_model_status("seed")                   # publish a token
    cache.reset_model_status_cache()                        # forget we saw it
    live = CountingStat()
    m = _model()
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    cache.cached_model_status(m, live)
    assert live.calls == 1
    assert cache.cache_stats()["epoch_clears"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 4. wiring — the real call sites
# ──────────────────────────────────────────────────────────────────────────
# cancelable_downloads.py lives under flask_app.app.functions; importing it as a
# bare dotted path before the flask app has booted trips the known import-order
# landmine (flask_app's `app` attribute shadowed by flask's own `app` submodule).
# Booting once first populates sys.modules correctly — same preamble as
# tests/test_download_progress.py.
importlib.import_module("abstract_hugpy_dev.flask_app.wsgi_app").get_hugpy_flask()
dl = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.downloads.downloader")
cd = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.downloads.cancelable_downloads")
routes = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.llm_storage_routes")
v1 = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes.v1_routes")
mc = importlib.import_module("abstract_hugpy_dev.imports.config.models.models_config")


def _manifest(n=25):
    return {f"repo-{i}": _model(f"repo-{i}") for i in range(n)}


def _flask_app():
    return importlib.import_module(
        "abstract_hugpy_dev.flask_app.wsgi_app").get_hugpy_flask()


def test_update_model_status_reads_through_the_memo(monkeypatch):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    rows = [_model(f"repo-{i}") for i in range(30)]
    first = [dict(cd.update_model_status(dict(r))) for r in rows]
    second = [dict(cd.update_model_status(dict(r))) for r in rows]
    assert live.calls == 30, "the second manifest walk must not touch the store"
    assert first == second
    assert first[0]["status"] == "installed"
    assert first[0]["destination"].endswith("repo-0")


def test_v1_models_identical_cached_vs_uncached(monkeypatch):
    """Response CONTENT is unchanged; only the number of store reads moves."""
    live = CountingStat()
    manifest = _manifest()
    monkeypatch.setattr(dl, "model_status", live)
    monkeypatch.setattr(v1, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(v1, "api_key_required", lambda: False)
    app = _flask_app()

    monkeypatch.setenv("HUGPY_MODEL_STATUS_TTL_S", "0")     # today's behaviour
    with app.test_client() as client:
        uncached = client.get("/v1/models")
        assert uncached.status_code == 200
        uncached_body = uncached.get_json()
    reads_uncached = live.calls

    cache.reset_model_status_cache()
    monkeypatch.setenv("HUGPY_MODEL_STATUS_TTL_S", "30")
    live.calls = 0
    with app.test_client() as client:
        cold = client.get("/v1/models").get_json()
        reads_cold = live.calls
        warm = client.get("/v1/models").get_json()
        reads_warm = live.calls

    assert uncached_body == cold == warm
    assert len(cold["data"]) == len(manifest)
    assert reads_uncached == len(manifest)
    assert reads_cold == len(manifest)
    assert reads_warm == reads_cold, (
        "a second /v1/models must add ZERO store reads")


def test_models_listing_shares_the_same_memo(monkeypatch):
    """/models and /v1/models must not each keep their own idea of status."""
    live = CountingStat()
    manifest = _manifest(10)
    monkeypatch.setattr(dl, "model_status", live)
    monkeypatch.setattr(v1, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(v1, "api_key_required", lambda: False)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)
    app = _flask_app()
    with app.test_client() as client:
        first = client.get("/models")
        assert first.status_code == 200
        assert live.calls == len(manifest)
        client.get("/v1/models")
        assert live.calls == len(manifest), (
            "/v1/models re-walked what /models had already read")
        body = client.get("/models").get_json()
    assert len(body) == len(manifest)
    assert live.calls == len(manifest)


def test_single_model_route_is_a_live_refresh(monkeypatch):
    live = CountingStat()
    manifest = _manifest(3)
    monkeypatch.setattr(dl, "model_status", live)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)
    app = _flask_app()
    with app.test_client() as client:
        client.get("/models/repo-0")
        client.get("/models/repo-0")
    assert live.calls == 2, "the detail route must always read the store"


def test_central_provisioning_poll_shares_the_memo(monkeypatch):
    """/llm/central-provisioning is polled every 10s by the console and loops
    the WHOLE manifest through the same route_destination + model_looks_downloaded
    walk. The listing must memoize; the single-model GUARD must stay live."""
    wr = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.routes.worker_routes")
    manifest = _manifest(12)
    probes = {"n": 0}

    def fake_reason(model_key):
        probes["n"] += 1
        return None if model_key.endswith("0") else "no model directory"

    class _Cfg:
        def __init__(self, row):
            self._row = row

        def to_dict(self):
            return dict(self._row)

    monkeypatch.setattr(wr, "_central_missing_reason", fake_reason)
    monkeypatch.setattr(wr, "get_models_dict", lambda **_k: manifest)
    main = importlib.import_module("abstract_hugpy_dev.imports.config.main")
    monkeypatch.setattr(main, "get_model_config",
                        lambda mk, **_k: _Cfg(manifest[mk]))
    app = _flask_app()

    with app.test_client() as client:
        first = client.get("/llm/central-provisioning")
        assert first.status_code == 200
        assert probes["n"] == len(manifest)
        second = client.get("/llm/central-provisioning").get_json()
        assert probes["n"] == len(manifest), (
            "a second poll re-walked the store")
    assert first.get_json() == second
    assert second["repo-0"] == {"state": "ready", "reason": None}
    assert second["repo-1"]["state"] == "absent"

    # the guard path is deliberately NOT memoized
    before = probes["n"]
    wr._central_missing_reason("repo-1")
    wr._central_missing_reason("repo-1")
    assert probes["n"] == before + 2


# ── invalidation events ───────────────────────────────────────────────────
def _spy_invalidation(monkeypatch):
    seen = []
    real = cache.invalidate_model_status

    def spy(reason=""):
        seen.append(reason)
        real(reason)

    monkeypatch.setattr(cache, "invalidate_model_status", spy)
    return seen


def test_delete_route_invalidates_and_forces_a_re_read(monkeypatch, tmp_path):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    dest = tmp_path / "models" / "gguf" / "org" / "repo-0"
    dest.mkdir(parents=True)
    (dest / "w.gguf").write_bytes(b"x")
    manifest = _manifest(3)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(routes, "route_destination", lambda _m: str(dest))
    seen = _spy_invalidation(monkeypatch)
    app = _flask_app()

    with app.test_client() as client:
        cd.update_model_status(_model("repo-0"))
        assert live.calls == 1
        cd.update_model_status(_model("repo-0"))
        assert live.calls == 1                     # memoized
        resp = client.delete("/models/repo-0")
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

    assert any("deleted" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2, "delete must force the listings to re-read"


def test_prune_route_invalidates(monkeypatch, tmp_path):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    manifest = _manifest(3)
    monkeypatch.setattr(routes, "get_models_dict", lambda **_k: manifest)
    monkeypatch.setattr(routes, "route_destination",
                        lambda _m: str(tmp_path / "gone"))
    monkeypatch.setattr(routes, "prune_model", lambda k: {"pruned": True, "key": k})
    seen = _spy_invalidation(monkeypatch)
    app = _flask_app()

    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1
    with app.test_client() as client:
        resp = client.post("/models/repo-0/prune")
        assert resp.status_code == 200
    assert any("pruned" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2


def test_reconcile_apply_invalidates_but_a_dry_run_does_not(monkeypatch):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    reconcile = importlib.import_module("abstract_hugpy_dev.imports.apis.reconcile")
    monkeypatch.setattr(reconcile, "reconcile_store",
                        lambda **_k: {"actions": [], "warnings": []})
    seen = _spy_invalidation(monkeypatch)
    app = _flask_app()

    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1
    # /models/reconcile is operator-token gated (operator_auth._SENSITIVE), so
    # drive the view function inside a request context — we are testing the
    # invalidation wiring, not the gate.
    with app.test_request_context("/models/reconcile", method="POST",
                                  json={"apply": False}):
        _body, status = routes.reconcile_store_route()
        assert status == 202
    assert not seen, "a dry run touches nothing and must not invalidate"
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1

    with app.test_request_context("/models/reconcile", method="POST",
                                  json={"apply": True}):
        _body, status = routes.reconcile_store_route()
        assert status == 200
    assert any("reconcile" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2


def test_refresh_registry_invalidates(monkeypatch):
    """THE chokepoint: download completion, the discovery sweep and reconcile's
    registry write all land here, so one hook covers them."""
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    md = importlib.import_module(
        "abstract_hugpy_dev.imports.config.models.models_default")
    ov = importlib.import_module("abstract_hugpy_dev.managers.serve.overrides")
    # refresh_registry(run_discovery=False) is idempotent on the live registry
    # (get_models_dict returns the cached object, so update+prune are no-ops);
    # stub only the two expensive follow-ups so this test never walks the store.
    monkeypatch.setattr(md, "refresh_task_registries", lambda *a, **k: None)
    monkeypatch.setattr(ov, "migrate_overrides", lambda *a, **k: [])
    seen = _spy_invalidation(monkeypatch)

    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1
    mc.refresh_registry(run_discovery=False)
    assert any("refresh_registry" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2


def test_download_cancel_invalidates(monkeypatch):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    seen = _spy_invalidation(monkeypatch)
    job = cd.job_store.create("repo-0", kind="download", transport="test")

    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1
    assert cd.cancel_download(job.id)["cancelled"] is True
    assert any("cancelled" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2


class _FakeProc:
    """A download child that finishes immediately with ``exitcode``."""

    def __init__(self, exitcode=0):
        self.exitcode = exitcode
        self._alive = False

    def start(self):
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, *_a, **_k):
        return None


class _FakeCtx:
    def __init__(self, exitcode=0):
        self.exitcode = exitcode

    def Process(self, *_a, **_k):
        return _FakeProc(self.exitcode)


def _drive_download(monkeypatch, tmp_path, exitcode, max_attempts=1):
    """Run start_cancellable_download's monitor to a terminal state with no
    real subprocess, network or store walk."""
    dest = tmp_path / "dest"
    dest.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cd, "route_destination", lambda **_k: str(dest))
    monkeypatch.setattr(cd, "_estimate_total_bytes_bounded", lambda _m: None)
    monkeypatch.setattr(cd, "_watch", lambda *_a, **_k: False)
    monkeypatch.setattr(cd, "_read_error", lambda _j: "synthetic failure")
    monkeypatch.setattr(cd, "record_downloaded_model", lambda *a, **k: None)
    monkeypatch.setattr(cd, "refresh_registry", lambda *a, **k: None)
    monkeypatch.setattr(cd, "MAX_ATTEMPTS", max_attempts)
    monkeypatch.setattr(cd.mp, "get_context", lambda _n: _FakeCtx(exitcode))

    job = cd.job_store.create("repo-0", kind="download", transport="test")
    cd.start_cancellable_download(job, _model("repo-0"))
    deadline = time.time() + 30
    while time.time() < deadline:
        cur = cd.job_store.get(job.id)
        if cur is not None and cur.terminal:
            return cur
        time.sleep(0.05)
    raise AssertionError("download monitor never reached a terminal state")


def test_download_completion_invalidates(monkeypatch, tmp_path):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    seen = _spy_invalidation(monkeypatch)
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1

    # "completed" canonicalizes to "done" in the shared job store.
    job = _drive_download(monkeypatch, tmp_path, exitcode=0)
    assert normalize_status(job.status) == "done"
    assert any("completed" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2, "a finished download must show up in the listings"


def test_download_failure_invalidates(monkeypatch, tmp_path):
    live = CountingStat()
    monkeypatch.setattr(dl, "model_status", live)
    seen = _spy_invalidation(monkeypatch)
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 1

    job = _drive_download(monkeypatch, tmp_path, exitcode=1)
    assert normalize_status(job.status) == "failed"
    assert any("failed" in r for r in seen), seen
    cd.update_model_status(_model("repo-0"))
    assert live.calls == 2, "a give-up leaves partial files — re-read"


# ──────────────────────────────────────────────────────────────────────────
# 5. the real walk, measured against a synthetic store
# ──────────────────────────────────────────────────────────────────────────
def test_real_model_status_walk_is_eliminated_on_the_second_pass(tmp_path,
                                                                 monkeypatch):
    """End-to-end with the REAL ``model_status``: count actual os.* calls.

    Uses the real resolver against a throwaway store (never the live one), so
    this measures the thing that was killing central rather than a stub."""
    paths = importlib.import_module(
        "abstract_hugpy_dev.imports.src.constants.paths")
    root = tmp_path / "store"
    for i in range(20):
        d = root / "models" / "gguf" / f"org{i % 4}" / f"repo-{i}"
        d.mkdir(parents=True)
        (d / f"repo-{i}.Q4_K_M.gguf").write_bytes(b"\0" * (2 * 1024 * 1024))
    for task in ("text-generation", "image-text-to-text"):
        (root / "models" / "gguf" / task).mkdir(parents=True, exist_ok=True)

    real_route = paths.route_destination
    monkeypatch.setattr(
        dl, "route_destination",
        lambda m, _r=str(root): real_route(m, _r))

    counter = {"n": 0}
    originals = {name: getattr(os, name)
                 for name in ("stat", "lstat", "listdir", "scandir")}

    def wrap(fn):
        def inner(*a, **k):
            counter["n"] += 1
            return fn(*a, **k)
        return inner

    rows = [_model(f"repo-{i}") for i in range(20)]
    try:
        for name, fn in originals.items():
            monkeypatch.setattr(os, name, wrap(fn))
        [cd.update_model_status(dict(r)) for r in rows]      # warm the dentries
        cache.reset_model_status_cache()

        counter["n"] = 0
        first = [dict(cd.update_model_status(dict(r))) for r in rows]
        cold_calls = counter["n"]

        counter["n"] = 0
        second = [dict(cd.update_model_status(dict(r))) for r in rows]
        warm_calls = counter["n"]
    finally:
        for name, fn in originals.items():
            setattr(os, name, fn)

    assert first == second, "the memo must not change what the listing reports"
    assert cold_calls > 100, f"expected a real walk, saw {cold_calls} os calls"
    assert warm_calls <= 4, (
        f"a warm listing still made {warm_calls} filesystem calls "
        f"(cold pass was {cold_calls})")
    print(json.dumps({"models": len(rows), "cold_os_calls": cold_calls,
                      "warm_os_calls": warm_calls}))
