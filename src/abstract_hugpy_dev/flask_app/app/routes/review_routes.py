### routes/review_routes.py
"""Model review over HTTP — the on-demand half of the reviewer.

  GET  /llm/review/criteria            saved criteria
  PUT  /llm/review/criteria/<name>     create/update one
  POST /llm/review/screen              {hub_ids:[...]} metadata verdict, no download
  POST /llm/review/run                 {criteria|hub_ids} full pipeline (background)
  GET  /llm/review/runs                run history
  GET  /llm/review/results             recorded reviews (?criteria=&best=1)
  POST /llm/review/ingest              a worker box pushes a finished run in

CENTRAL'S DB IS THE SOURCE OF TRUTH. The pipeline runs where the GPU is (ae),
writing its rows to that box's local sqlite; without /ingest those rows never
reach the DB these read routes serve and the console shows an empty
leaderboard. /ingest is the landing zone — see review/push.py for the sending
half. Flow is one-way, worker -> central, always.

Screening is synchronous — it's metadata only and returns in seconds. A full
run downloads weights and loads them on the GPU, so it goes to a background
thread and the caller polls /runs; holding a request open for the length of a
20 GB download is how you get a gateway timeout and an orphaned staging dir.
"""
from ..functions import *

review_bp, logger = get_bp("review_bp", __name__)

_RUNNING: dict = {}          # criteria name -> {"run_id":…, "started":…}


def _crit(name_or_payload):
    from abstract_hugpy_dev.review.criteria import ReviewCriteria, load_criteria
    if isinstance(name_or_payload, str):
        return load_criteria(name_or_payload)
    return ReviewCriteria.from_dict(name_or_payload)


@review_bp.route("/llm/review/criteria", methods=["GET"])
def review_criteria_list():
    from abstract_hugpy_dev.review.criteria import list_criteria, load_criteria
    out = []
    for name in list_criteria():
        try:
            out.append(load_criteria(name).to_dict())
        except Exception as exc:
            out.append({"name": name, "error": str(exc)})
    return jsonify(out)


@review_bp.route("/llm/review/criteria/<name>", methods=["PUT"])
def review_criteria_put(name):
    from abstract_hugpy_dev.review.criteria import ReviewCriteria, save_criteria
    payload = request.get_json(silent=True) or {}
    payload["name"] = name
    crit = ReviewCriteria.from_dict(payload)
    save_criteria(crit)
    return jsonify(crit.to_dict())


@review_bp.route("/llm/review/screen", methods=["POST"])
def review_screen():
    """Metadata-only verdict for specific repos. Fast, fetches no weights."""
    payload = request.get_json(silent=True) or {}
    hub_ids = payload.get("hub_ids") or ([payload["hub_id"]]
                                         if payload.get("hub_id") else [])
    if not hub_ids:
        abort(400, description="hub_ids is required")
    if len(hub_ids) > 50:
        abort(400, description="at most 50 hub_ids per request")

    crit = _crit(payload.get("criteria") or payload.get("criteria_name") or {"name": "adhoc"})
    from abstract_hugpy_dev.review.screen import screen, _hf_api
    api = _hf_api()
    results = []
    for hub_id in hub_ids:
        try:
            results.append(screen(hub_id, crit, api=api).to_dict())
        except Exception as exc:
            results.append({"hub_id": hub_id, "passed": False,
                            "reasons": [f"{type(exc).__name__}: {exc}"]})
    return jsonify({"criteria": crit.name, "results": results})


@review_bp.route("/llm/review/run", methods=["POST"])
def review_run():
    """Full pipeline in the background. One run per criteria at a time — two
    concurrent runs would fight over the GPU and the disk cap."""
    import threading
    import time as _time

    payload = request.get_json(silent=True) or {}
    name = payload.get("criteria") or payload.get("criteria_name")
    if not name:
        abort(400, description="criteria is required")
    crit = _crit(name)
    hub_ids = payload.get("hub_ids") or None
    force = bool(payload.get("force"))

    live = _RUNNING.get(crit.name)
    if live and live.get("thread") and live["thread"].is_alive():
        return jsonify({"status": "already_running", **{
            k: v for k, v in live.items() if k != "thread"}}), 409

    from abstract_hugpy_dev.review import pipeline

    def _work():
        try:
            pipeline.run(crit, hub_ids=hub_ids, force=force,
                         log=lambda m: logger.info("[review] %s", m))
        except Exception as exc:
            logger.exception("review run for %s failed: %s", crit.name, exc)

    t = threading.Thread(target=_work, name=f"review-{crit.name}", daemon=True)
    _RUNNING[crit.name] = {"started": _time.time(), "thread": t}
    t.start()
    return jsonify({"status": "started", "criteria": crit.name}), 202


@review_bp.route("/llm/review/runs", methods=["GET"])
def review_runs():
    from abstract_hugpy_dev.review import store
    return jsonify(store.runs(request.args.get("criteria"),
                              limit=request.args.get("limit", 20, type=int)))


# ── ingest: a worker box hands central its finished run ───────────────────
# Cap one POST. push.py chunks at 250; anything far above that is a confused or
# hostile caller, and the run header rides every chunk so truncating the tail
# would silently lose results rather than fail honestly.
MAX_INGEST_RESULTS = 1000


def _ingest_authorized() -> bool:
    """Ingest gate: the SAME credential /llm/evictions/ingest uses.

    Modelled on eviction_routes._worker_authorized — delegates to
    worker_routes._enrollment_ok, so a worker presents the enrollment bearer it
    already holds for register/heartbeat and revoking that worker stops its
    review pushes at the same instant it stops its heartbeat. No new secret to
    rotate. An operator token is accepted too (manual replay from a laptop,
    `python -m abstract_hugpy_dev.review push`).

    Fails CLOSED if the gate cannot be imported: an unauthenticated write
    endpoint is not an acceptable degradation."""
    try:
        from ..operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:                       # noqa: BLE001
        pass                                # fall through to the worker gate
    # STRICTER than register/heartbeat on purpose (keeper, 2026-07-29): the
    # gradual-rollout "no token -> allow" that _enrollment_ok grants would make
    # this WRITE endpoint publicly writable through the internet-facing origin
    # (a probe proved it). A present, valid enrollment token is required
    # always; the fleet holds per-box tokens since the same day.
    try:
        from .worker_routes import _bearer_token
        from ..functions.imports.utils.enrollment_tokens import (
            verify_enrollment_token)
        tok = _bearer_token()
        return tok is not None and bool(verify_enrollment_token(tok))
    except Exception:                       # noqa: BLE001
        logger.warning("review ingest: enrollment gate unavailable — refusing")
        return False


@review_bp.route("/llm/review/ingest", methods=["POST"])
def review_ingest():
    """Accept one pushed run + its results from the box that produced them.

    Body::

        {"host": "ae",
         "criteria": "nightly",
         "run": {"run_id": 12, "criteria": "nightly", "started_at": …,
                 "finished_at": …, "screened": …, "passed": …,
                 "downloaded": …, "smoked": …, "error": null},
         "results": [{"run_id": 12, "criteria": …, "hub_id": …, "stage": …,
                      "passed": …, "score": …, "verdict": …,
                      "payload": {…}, "reviewed_at": …}, …]}

    IDEMPOTENT. Rows are keyed on (host, run_id) and
    (host, run_id, hub_id, stage), so a retried or chunk-replayed push updates
    in place — a worker that never sees our 200 can resend forever without
    duplicating a single row.

    NEVER 5xx OVER DATA. A malformed row is counted in ``rejected`` and the
    rest of the batch still lands; a store fault answers 200 with everything
    counted rejected. The only non-2xx answers are 401 (unauthenticated) and
    400 (the envelope itself is not the documented shape) — because a worker
    that reads a 5xx as "central is broken, retry harder" is how a telemetry
    channel turns into a storm."""
    if not _ingest_authorized():
        return jsonify({"error": "Worker enrollment or operator token required."}), 401
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    host = (body.get("host") or "").strip() if isinstance(body.get("host"), str) else ""
    if not host:
        return jsonify({"error": "host is required — rows are keyed on it"}), 400
    run = body.get("run")
    results = body.get("results")
    if run is not None and not isinstance(run, dict):
        return jsonify({"error": "run must be an object"}), 400
    if results is None:
        results = []
    if not isinstance(results, list):
        return jsonify({"error": "results must be a list"}), 400
    if len(results) > MAX_INGEST_RESULTS:
        return jsonify({"error": f"at most {MAX_INGEST_RESULTS} results per "
                                 f"request; send more chunks"}), 400

    from abstract_hugpy_dev.review import store

    run_row = None
    run_ok = False
    if run:
        try:
            run_row = store.ingest_run(host, run)
            run_ok = run_row is not None
        except Exception as exc:            # noqa: BLE001 — never 5xx at a worker
            logger.warning("review ingest: run upsert failed for %s: %s", host, exc)

    accepted = rejected = 0
    try:
        accepted, rejected = store.ingest_results(host, results)
    except Exception as exc:                # noqa: BLE001
        logger.warning("review ingest: result upsert failed for %s: %s", host, exc)
        rejected = len(results)

    if not run_ok and run:
        rejected += 1                       # the run header itself was unusable
    return jsonify({"ok": True, "host": host, "run_id": run_row,
                    "accepted": accepted, "rejected": rejected})


@review_bp.route("/llm/review/results", methods=["GET"])
def review_results():
    from abstract_hugpy_dev.review import store
    criteria = request.args.get("criteria")
    limit = request.args.get("limit", 50, type=int)
    if request.args.get("best") not in (None, "", "0"):
        if not criteria:
            abort(400, description="best=1 requires criteria")
        return jsonify(store.leaderboard(criteria, limit=limit))
    return jsonify(store.recent(criteria, limit=limit,
                                stage=request.args.get("stage")))
