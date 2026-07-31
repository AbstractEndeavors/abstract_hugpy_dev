"""STUDIO-ASSIST LIVE LOG (operator directive, 2026-07-31).

The operator wants a live log in the studio UI showing what each prompt-generate
attempt actually returned — the raw model reply, what was stripped, and the
outcome — so failures ("returned only reasoning and no output", "did not return
the JSON object the spread contract requires") are self-diagnosable.

This proves the two halves that make that log trustworthy:

  1. THE STORE (comms/studio_assist_log.py) — append/recent/max_id, the ring
     bound, and the sqlite mirror keyed by rowid (the cross-gunicorn-worker
     cursor). A store fault must cost nothing; ``raw`` is kept UNTRUNCATED.

  2. THE INSTRUMENTATION (flask_app/.../video_routes.py) — the four outcomes the
     operator actually sees, driven through the real route with a stubbed
     executor (the tests/test_prompt_spread.py harness):
        * a served spread          (JSON parsed ok      -> outcome=served)
        * a spread parse failure   (SpreadParseError    -> outcome=parse_error,
                                     the FULL raw reply captured)
        * a reasoning-only reply   (<think>-only        -> from_reasoning True)
        * a worker error           (executor raised     -> outcome=worker_error)

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_studio_assist_log.py -q
    venv/bin/python tests/test_studio_assist_log.py
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.disable(logging.INFO)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


from abstract_hugpy_dev.comms import studio_assist_log as SAL  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. THE STORE
# --------------------------------------------------------------------------- #
def test_store_append_recent_maxid():
    with tempfile.TemporaryDirectory() as d:
        store = SAL.StudioAssistStore(path=os.path.join(d, "c.db"))
        SAL.set_store(store)
        SAL.reset_for_tests()

        check("empty store recent is []", store.recent() == [])
        check("empty store max_id is 0", store.max_id() == 0)

        r1 = SAL.append(run_id="run-1", mode="spread", outcome=SAL.OUTCOME_SERVED,
                        raw="A" * 10, text="A prompt", model_requested="m",
                        model_resolved="m-x", elapsed_ms=12)
        r2 = SAL.append(run_id="run-2", mode="negative", outcome=SAL.OUTCOME_EMPTY,
                        raw="", error="nothing came back")
        check("append returns a record", isinstance(r1, dict) and r1["run_id"] == "run-1")

        got = store.recent()
        check("both rows persisted", len(got) == 2)
        check("recent is oldest-first", got[0]["run_id"] == "run-1" and got[1]["run_id"] == "run-2")
        check("each row carries a store _id", all(isinstance(g["_id"], int) for g in got))
        check("max_id tracks the head", store.max_id() == got[-1]["_id"])

        # after_id is the stream cursor — returns only what is newer.
        tail = store.recent(after_id=got[0]["_id"])
        check("after_id tails by rowid", len(tail) == 1 and tail[0]["run_id"] == "run-2")

        # raw is stored UNTRUNCATED; raw_cap only bounds the returned copy.
        SAL.append(run_id="big", mode="spread", outcome=SAL.OUTCOME_PARSE_ERROR,
                   raw="Z" * 5000, error="bad json")
        full = [g for g in store.recent() if g["run_id"] == "big"][0]
        check("store keeps raw untruncated", len(full["raw"]) == 5000)
        capped = [g for g in store.recent(raw_cap=100) if g["run_id"] == "big"][0]
        check("raw_cap truncates the returned copy", len(capped["raw"]) == 100)
        check("raw_cap flags truncation", capped.get("raw_truncated") is True)
        SAL.set_store(None)


def test_ring_is_bounded():
    SAL.set_store(SAL.StudioAssistStore(path="off"))   # disable sqlite side
    SAL.reset_for_tests()
    orig = SAL.RING_MAX
    try:
        SAL.RING_MAX = 5
        SAL._RING.clear()
        SAL._RING = __import__("collections").deque(maxlen=SAL.RING_MAX)
        for i in range(20):
            SAL.append(run_id=f"r{i}", mode="spread", outcome=SAL.OUTCOME_SERVED)
        ring = SAL.recent(limit=999)
        check("ring is bounded to RING_MAX", len(ring) == 5)
        check("ring keeps the NEWEST", ring[-1]["run_id"] == "r19")
    finally:
        SAL.RING_MAX = orig
        SAL._RING = __import__("collections").deque(maxlen=orig)
    SAL.set_store(None)


def test_disabled_store_is_silent():
    store = SAL.StudioAssistStore(path="off")   # HUGPY_COMMS_DB=off sentinel
    check("off sentinel disables the store", store._disabled is True)
    check("disabled append is a no-op", store.append([{"run_id": "x", "stage": "s"}]) == 0)
    check("disabled recent is []", store.recent() == [])
    check("disabled max_id is 0", store.max_id() == 0)
    # A store pointed at an unwritable path must degrade, never raise.
    bad = SAL.StudioAssistStore(path="/proc/nonexistent/dir/c.db")
    check("unwritable store append degrades to 0",
          bad.append([{"run_id": "y"}]) == 0)


def test_outcome_classification():
    C = SAL.classify_execute_error
    check("400 -> resolve_error", C(400, "unknown model") == SAL.OUTCOME_RESOLVE_ERROR)
    check("502 dead worker -> worker_error",
          C(502, "no live worker for this model") == SAL.OUTCOME_WORKER_ERROR)
    check("502 produced no text -> empty",
          C(502, "assist produced no text") == SAL.OUTCOME_EMPTY)
    check("502 returned nothing -> empty",
          C(502, "the assistant returned nothing — neither a prompt nor reasoning")
          == SAL.OUTCOME_EMPTY)


# --------------------------------------------------------------------------- #
# 2. THE INSTRUMENTATION — driven through the real route, stubbed executor.
#
# Reuses the tests/test_prompt_spread.py harness (_patch_executor patches the
# ONE execute_prompt the route reaches through functions.imports; patching
# managers.dispatch would silently run live inference — see that file's landmine
# note).
# --------------------------------------------------------------------------- #
def _load_route_harness():
    import json  # noqa: F401 — used by the spread reply builder
    from flask import Flask

    vr = importlib.import_module(
        "abstract_hugpy_dev.flask_app.app.routes.video_routes")
    imports_mod = sys.modules[
        "abstract_hugpy_dev.flask_app.app.functions.imports"]
    app = Flask(__name__)
    app.register_blueprint(vr.video_bp)
    return vr, imports_mod, app.test_client()


def _patch_executor(imports_mod, reply_text, ok=True, error=None, raises=None):
    import types
    def fake_execute_prompt(*a, **kw):
        if raises is not None:
            raise raises
        return {"ok": ok, "text": reply_text, "error": error,
                "model_key": "resolved-model-x"}
    imports_mod.execute_prompt = fake_execute_prompt
    return types.SimpleNamespace()


def _spread_body(ids=("segment-0", "segment-1")):
    return {
        "mode": "spread",
        "movie_query": "a diver finds something under the ice",
        "style_bible": {"world": "arctic station", "subject": "a lone diver",
                        "visual_style": "grainy 16mm"},
        "fixed_segments": [],
        "target_segments": [
            {"segment_id": sid, "direction": "make it colder",
             "joint_mode": "vace_extend", "index": i}
            for i, sid in enumerate(ids)],
        "global_negative": ["watermark"],
        "steering_seed": 184392,
    }


def _segments_reply(ids):
    import json
    return json.dumps({"segments": [
        {"segment_id": sid, "operation": "generate_from_direction",
         "prompt": f"A shot for {sid}.", "negative": "blurry",
         "continuity_note": f"after {sid}", "directions_used": [0]}
        for sid in ids], "invented_identity_attributes": [], "warnings": []})


def _fresh_store():
    d = tempfile.mkdtemp()
    SAL.set_store(SAL.StudioAssistStore(path=os.path.join(d, "c.db")))
    SAL.reset_for_tests()
    return SAL.get_store()


def test_served_spread_records_served():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    ids = ("segment-0", "segment-1")
    _patch_executor(imports_mod, _segments_reply(ids))
    r = client.post("/video/prompt/assist", json=_spread_body(ids))
    check("served spread returns 200", r.status_code == 200)
    rows = store.recent()
    check("served spread logged exactly one record", len(rows) == 1)
    rec = rows[0]
    check("served spread outcome=served", rec["outcome"] == SAL.OUTCOME_SERVED)
    check("served spread captured raw", "segment-0" in rec.get("raw", ""))
    check("served spread labelled mode=spread", rec["mode"] == "spread")
    check("served spread records model_resolved", rec["model_resolved"] == "resolved-model-x")
    SAL.set_store(None)


def test_spread_parse_error_records_raw():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    # A reply that is NOT the spread JSON contract -> SpreadParseError.
    _patch_executor(imports_mod, "here are some nice shots for your movie, enjoy!")
    r = client.post("/video/prompt/assist", json=_spread_body())
    check("parse failure returns 502", r.status_code == 502)
    rows = store.recent()
    check("parse failure logged one record", len(rows) == 1)
    rec = rows[0]
    check("parse failure outcome=parse_error", rec["outcome"] == SAL.OUTCOME_PARSE_ERROR)
    check("parse failure captured the FULL raw reply",
          rec.get("raw") == "here are some nice shots for your movie, enjoy!")
    check("parse failure recorded the contract error", bool(rec.get("error")))
    SAL.set_store(None)


def test_reasoning_only_reply_flags_from_reasoning():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    # The whole answer lives inside <think>…</think>: _studio_no_think salvages
    # the reasoning as the prompt and flags from_reasoning. Drive it through the
    # detail/generate inline handler (mode=generate), which serves prose.
    _patch_executor(imports_mod, "<think>A neon city at dusk, rain-slicked streets</think>")
    r = client.post("/video/prompt/assist",
                    json={"mode": "generate", "context": {"kind": "image"}})
    check("reasoning-only generate returns 200", r.status_code == 200)
    body = r.get_json()
    check("reasoning-only served the reasoning as prompt", bool(body.get("prompt")))
    rows = store.recent()
    check("reasoning-only logged one record", len(rows) == 1)
    rec = rows[0]
    check("reasoning-only outcome=served", rec["outcome"] == SAL.OUTCOME_SERVED)
    check("reasoning-only flags from_reasoning True", rec.get("from_reasoning") is True)
    check("reasoning-only captured the raw <think> reply", "<think>" in rec.get("raw", ""))
    check("reasoning-only stripped text present", bool(rec.get("text")))
    SAL.set_store(None)


def test_worker_error_records_worker_error():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    _patch_executor(imports_mod, "", raises=RuntimeError("no live worker for this model"))
    r = client.post("/video/prompt/assist", json=_spread_body())
    check("worker error returns 502", r.status_code == 502)
    rows = store.recent()
    check("worker error logged one record", len(rows) == 1)
    rec = rows[0]
    check("worker error outcome=worker_error", rec["outcome"] == SAL.OUTCOME_WORKER_ERROR)
    check("worker error recorded the message", bool(rec.get("error")))
    SAL.set_store(None)


def test_resolve_error_records_resolve_error():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    _patch_executor(imports_mod, "", raises=ValueError("unknown model_key 'nope'"))
    r = client.post("/video/prompt/assist", json=_spread_body())
    check("resolve error returns 400", r.status_code == 400)
    rows = store.recent()
    rec = rows[0]
    check("resolve error outcome=resolve_error", rec["outcome"] == SAL.OUTCOME_RESOLVE_ERROR)
    SAL.set_store(None)


def test_backfill_and_stream_routes_exist():
    vr, imports_mod, client = _load_route_harness()
    _fresh_store()
    # No auth gate installed on this bare test app, so the routes resolve; we only
    # assert they are wired and shaped (the video gate provides auth in prod).
    r = client.get("/video/prompt/assist/log?limit=10")
    check("backfill route resolves", r.status_code == 200)
    body = r.get_json()
    check("backfill returns events+cursor", "events" in body and "cursor" in body)
    SAL.set_store(None)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print(f"\n{ok} checks passed")


if __name__ == "__main__":
    _run_all()
