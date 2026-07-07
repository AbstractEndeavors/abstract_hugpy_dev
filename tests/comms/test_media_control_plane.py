"""Slice B — media jobs as first-class citizens of the /llm control plane.

Locks the THREE additive changes and, above all, PROVES chat/download behavior
is unchanged:

  B1  cancel fan-out    POST /llm/jobs/<id>/cancel reaches media_bus.cancel for a
                        media job, and stays purely comms-side for chat.
  B2  terminal_rows     SqliteMirror.terminal_rows(kinds) is media-gated; a
                        sibling process sees a finished MEDIA job but NEVER a
                        finished chat job (chat terminal stays local-retention).
  B3  JobError.retryable  every /llm/jobs error object carries "retryable"
                        (null for chat/download, real bool for media), coerce()
                        round-trips it, and the bridge preserves it.

House style mirrors tests/studio/test_studio_conformance.py: a plain python
script with a __main__ guard, run via
``venv/bin/python tests/comms/test_media_control_plane.py``. pytest is NOT
installed in this venv. Each check prints a numbered ``[n] PASS`` / ``[n] FAIL``
line; the driver keeps going on a failing check so EVERY divergence surfaces; the
process exits nonzero iff any check FAILED.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_n = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _n, _failed
    _n += 1
    if cond:
        print(f"[{_n}] PASS  {name}")
    else:
        _failed += 1
        print(f"[{_n}] FAIL  {name}")


from abstract_hugpy_dev.comms.jobs import Job, JobError, JobStore, MEDIA_KINDS
from abstract_hugpy_dev.comms.shared import SqliteMirror

# ---------------------------------------------------------------------------
# B2 — SqliteMirror.terminal_rows is media-gated (READ-side).
# ---------------------------------------------------------------------------
db1 = os.path.join(tempfile.mkdtemp(prefix="b-mirror-"), "comms.db")
m = SqliteMirror(db1)
m.upsert(Job(id="t-med", kind="frame_extract", status="done",
             transport="media").to_dict())
m.upsert(Job(id="t-chat", kind="chat", status="done").to_dict())
m.upsert(Job(id="t-med-live", kind="crop", status="processing",
             transport="media").to_dict())          # live -> must be excluded
trows = {d["id"]: d for d in m.terminal_rows(MEDIA_KINDS)}
check("terminal_rows returns a finished media-kind row", "t-med" in trows)
check("terminal_rows EXCLUDES a finished chat-kind row", "t-chat" not in trows)
check("terminal_rows EXCLUDES a live media row", "t-med-live" not in trows)
check("terminal_rows(()) is empty (never a full terminal scan)",
      m.terminal_rows(()) == [])
check("live_rows() still excludes terminal (unchanged)",
      "t-med" not in {d["id"] for d in m.live_rows()})

# ---------------------------------------------------------------------------
# B2 — snapshot(live_only=False) merges a SIBLING's terminal MEDIA row, and
# leaves chat terminal cross-process behavior UNCHANGED.
# ---------------------------------------------------------------------------
db2 = os.path.join(tempfile.mkdtemp(prefix="b-snap-"), "comms.db")
A = JobStore(mirror=SqliteMirror(db2))   # "worker A" — actually runs the jobs
B = JobStore(mirror=SqliteMirror(db2))   # "worker B" — a sibling, ran nothing

A.create("scene", id="m-1", kind="generate_scene", transport="media")
A.finish("m-1")                          # terminal + mirrored
A.create("m", id="c-1", kind="chat", transport="web")
A.finish("c-1")                          # terminal + mirrored

bsnap_full = {d["id"] for d in B.snapshot(live_only=False)}
check("B (sibling) SEES the finished media job via the new merge",
      "m-1" in bsnap_full)
check("B (sibling) does NOT see the finished chat job (chat UNCHANGED)",
      "c-1" not in bsnap_full)

bsnap_live = {d["id"] for d in B.snapshot(live_only=True)}
check("B live view excludes the terminal media row (live path unchanged)",
      "m-1" not in bsnap_live)

asnap_full = {d["id"] for d in A.snapshot(live_only=False)}
check("A (owner) still sees its chat terminal via LOCAL retention (unchanged)",
      "c-1" in asnap_full)
check("A (owner) sees its media terminal locally too", "m-1" in asnap_full)

# kind filter is honored by the terminal merge too: filtering to chat surfaces
# no media terminal row on the sibling.
bsnap_chat = {d["id"] for d in B.snapshot(kinds={"chat"}, live_only=False)}
check("kind=chat filter surfaces no media terminal on sibling",
      "m-1" not in bsnap_chat and "c-1" not in bsnap_chat)

# ---------------------------------------------------------------------------
# B3 — JobError.retryable: emitted ALWAYS (nullable), coerce round-trips it,
# detail still works.
# ---------------------------------------------------------------------------
d_true = JobError(code="oom", message="no vram", retryable=True).to_dict()
check("JobError(retryable=True).to_dict() carries retryable: true",
      d_true.get("retryable") is True)
d_def = JobError(code="e", message="boom").to_dict()
check("default JobError serializes retryable: null",
      "retryable" in d_def and d_def["retryable"] is None)
c_false = JobError.coerce({"code": "x", "message": "y", "retryable": False})
check("coerce() pulls retryable from a dict (False round-trips)",
      c_false.retryable is False and c_false.to_dict()["retryable"] is False)
d_detail = JobError(code="e", message="m", detail={"k": 1}).to_dict()
check("detail still works alongside retryable",
      d_detail.get("detail") == {"k": 1} and d_detail["retryable"] is None)
c_detail = JobError.coerce({"code": "e", "detail": {"z": 2}})
check("coerce() still preserves detail", c_detail.detail == {"z": 2})

# ---------------------------------------------------------------------------
# B3 — bridge preserves retryable end to end (result_schema.JobError -> dict ->
# comms.JobError via coerce -> to_dict emits the real bool). _error_dict reads
# result.error, so the realistic shape is a JobResult wrapping the JobError —
# exactly what media_bus.run_claimed hands to on_terminal.
# ---------------------------------------------------------------------------
from abstract_hugpy_dev.video_intel import job_bridge
from abstract_hugpy_dev.video_intel import result_schema as _rs

res = _rs.JobResult(job_id="j", ok=False,
                    error=_rs.JobError(code="oom", message="no vram",
                                       retryable=True))
ed = job_bridge._error_dict(res)
check("_error_dict preserves retryable: true from result_schema.JobError",
      ed is not None and ed.get("retryable") is True and ed.get("code") == "oom")
res_f = _rs.JobResult(job_id="j", ok=False,
                      error=_rs.JobError(code="bad", message="x",
                                         retryable=False))
check("_error_dict preserves retryable: false",
      job_bridge._error_dict(res_f).get("retryable") is False)
check("bridge dict -> comms.JobError.coerce -> to_dict keeps the real bool",
      JobError.coerce(ed).to_dict()["retryable"] is True)
check("_error_dict tolerates a plain-dict result too",
      job_bridge._error_dict({"error": {"code": "x", "retryable": False}})
      .get("retryable") is False)

# ---------------------------------------------------------------------------
# B1 — the real flask app builds and the new cancel rule is registered.
# ---------------------------------------------------------------------------
from abstract_hugpy_dev.flask_app.wsgi_app import get_hugpy_flask
app = get_hugpy_flask()
_rules = {r.rule: r.methods for r in app.url_map.iter_rules()}
check("app builds and POST /llm/jobs/<job_id>/cancel is registered",
      "/llm/jobs/<job_id>/cancel" in _rules
      and "POST" in _rules["/llm/jobs/<job_id>/cancel"])

# ---------------------------------------------------------------------------
# B1 — cancel fan-out semantics over the live app (test client).
# ---------------------------------------------------------------------------
from abstract_hugpy_dev.comms import job_store
c = app.test_client()

# media_bus.cancel is patched with a smart spy for ALL cases: it returns a media
# result only for known media ids and a safe no-op otherwise, mirroring the real
# media_bus.cancel (which no-ops on a non-media id). The route now calls it
# UNCONDITIONALLY so a queued media job with no local record still fans out; every
# case records the call, and chat/unknown just get the no-op.
import abstract_hugpy_dev.video_intel.media_bus as _media_bus
_calls = []
_orig = _media_bus.cancel
_media_ids = {"mc-1", "qm-1"}


def _spy(jid):
    _calls.append(jid)
    if jid in _media_ids:
        return {"job_id": jid, "status": "cancelling", "cancelled": True}
    return {"job_id": jid, "status": None, "cancelled": False}


_media_bus.cancel = _spy
try:
    # (a) chat job: comms-side cancel fires the local handle; media_bus is called
    # but no-ops, so the response stays transport=web / cancelled=true.
    fired = []
    job_store.create("m", id="nc-1", kind="chat", transport="web")
    job_store.attach_cancel("nc-1", lambda: fired.append(1))
    rb = c.post("/llm/jobs/nc-1/cancel", json={"reason": "user stop"}).get_json()
    check("chat cancel via /llm/jobs -> cancelled true, transport web (media no-op)",
          rb.get("cancelled") is True and rb.get("transport") == "web")
    check("chat cancel actually fired the local handle", fired == [1])
    job_store.finish("nc-1")

    # (b) unknown id: nothing anywhere -> cancelled false, null status/transport.
    ru = c.post("/llm/jobs/nope/cancel").get_json()
    check("unknown id -> cancelled false, status/transport null",
          ru.get("cancelled") is False and ru.get("status") is None
          and ru.get("transport") is None)

    # (c) running media job (has a local transport=media record): fan-out reaches
    # media_bus.cancel and merges both planes.
    job_store.create("wan", id="mc-1", kind="studio_i2v", transport="media",
                     status="processing")
    rm = c.post("/llm/jobs/mc-1/cancel", json={}).get_json()
    check("media cancel fanned out to media_bus.cancel(job_id)", "mc-1" in _calls)
    check("media cancel merges planes (cancelled + media status + transport)",
          rm.get("cancelled") is True and rm.get("status") == "cancelling"
          and rm.get("transport") == "media")
    job_store.finish("mc-1")

    # (d) QUEUED media job with NO local JobStore record (on_enqueue is mirror-only):
    # job_store.get -> None, so transport can't be read locally. The unconditional
    # media_bus.cancel still fans out, and the route infers transport=media from it.
    rq = c.post("/llm/jobs/qm-1/cancel", json={}).get_json()
    check("queued media (no local record) still fans out to media_bus.cancel",
          "qm-1" in _calls)
    check("queued media cancel -> cancelled true, status cancelling, transport media",
          rq.get("cancelled") is True and rq.get("status") == "cancelling"
          and rq.get("transport") == "media")
finally:
    _media_bus.cancel = _orig

# ---------------------------------------------------------------------------
print(f"\n{_n - _failed} passed, {_failed} failed of {_n}")
sys.exit(1 if _failed else 0)
