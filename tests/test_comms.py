"""Sanity: comms.jobs + comms.bus behavior, incl. the DISC-03 races."""
import sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.comms import (
    Job, JobStore, JobError, normalize_status, Bus, BusMessage,
    wire_cancel, wire_job_events, TOPIC_CONTROL_CANCEL,
)

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

# --- status normalization / legacy aliases ---
check("queued->pending", normalize_status("queued") == "pending")
check("running->processing", normalize_status("running") == "processing")
check("completed->done", normalize_status("completed") == "done")
check("garbage->pending", normalize_status("wat") == "pending")
check("canonical passthrough", normalize_status("streaming") == "streaming")

# --- old download-caller API shape ---
s = JobStore()
j = s.create("qwen-7b")   # old positional signature
check("create(model_key) works", j.model_key == "qwen-7b" and j.status == "pending")
s.update(j.id, status="running", progress=0.5)   # old status name
check("legacy status write normalized", s.get(j.id).to_dict()["status"] == "processing")
d = s.get(j.id).to_dict()
for k in ("progress", "total_bytes", "attempt", "stalled", "created_at"):
    assert k in d, k
check("download dict keys preserved", True)

# --- chat lifecycle with request_id as job id ---
j2 = s.create("m", id="rid-1", kind="chat", transport="web", channel="sess-9")
check("job id is request id", j2.id == "rid-1")
s.on_output("rid-1")
s.on_output("rid-1")
snap = s.snapshot()
mine = [x for x in snap if x["id"] == "rid-1"][0]
check("streaming after first output", mine["status"] == "streaming")
check("token count", mine["tokens"] == 2)
check("wait <= elapsed", mine["wait"] <= mine["elapsed"])
s.finish("rid-1")
check("finish -> done", s.get("rid-1").to_dict()["status"] == "done")
check("terminal excluded from live snapshot",
      not [x for x in s.snapshot() if x["id"] == "rid-1"])

# --- cancel: handle fires, teardown converts, first-terminal-wins ---
fired = []
j3 = s.create("m", id="rid-2")
s.attach_cancel("rid-2", lambda: fired.append(1))
check("cancel returns True on live job", s.cancel("rid-2", reason="user stop"))
check("handle fired once", fired == [1])
check("not force-marked cancelled", s.get("rid-2").to_dict()["status"] == "pending")
s.finish("rid-2")   # stream teardown, no explicit status
check("teardown converts to cancelled", s.get("rid-2").to_dict()["status"] == "cancelled")

j4 = s.create("m", id="rid-3")
s.finish("rid-3")                       # finished just before cancel arrives
check("cancel on finished job is False", not s.cancel("rid-3"))
check("stays done (first terminal wins)", s.get("rid-3").to_dict()["status"] == "done")
s.update("rid-3", status="failed")      # late overwrite attempt
check("terminal not overwritten", s.get("rid-3").to_dict()["status"] == "done")

# cancel BEFORE handle attach -> fires on attach
j5 = s.create("m", id="rid-4")
s.cancel("rid-4")
late = []
s.attach_cancel("rid-4", lambda: late.append(1))
check("late-attached handle fires immediately", late == [1])

# --- error-as-data ---
j6 = s.create("m", id="rid-5")
s.finish("rid-5", error=RuntimeError("boom"))
d6 = s.get("rid-5").to_dict()
check("failed with typed error",
      d6["status"] == "failed" and d6["error"]["code"] == "RuntimeError"
      and d6["error"]["message"] == "boom")

# --- bus: topics, wildcard, addressing, serialization ---
b = Bus()
all_sub = b.subscribe("job.*")
addr_sub = b.subscribe("control.cancel", target="worker-a")
b.publish("job.created", job_id="x", payload={"kind": "chat"})
m = all_sub.get(timeout=1)
check("wildcard receives", m is not None and m.topic == "job.created")
b.publish("control.cancel", job_id="y", target="worker-b")
check("addressed msg not delivered to other target", addr_sub.get(timeout=0.1) is None)
b.publish("control.cancel", job_id="z", target="worker-a")
check("addressed msg delivered to its target", addr_sub.get(timeout=1).job_id == "z")
b.publish("control.cancel", job_id="w")   # broadcast reaches targeted sub too
check("broadcast reaches targeted sub", addr_sub.get(timeout=1).job_id == "w")
rt = BusMessage.from_dict(m.to_dict())
check("envelope round-trips", rt.topic == m.topic and rt.job_id == m.job_id)

# --- wire_cancel: control message -> store.cancel -> handle fires ---
s2 = JobStore()
b2 = Bus()
th = wire_cancel(b2, s2)
check("wire_cancel idempotent", wire_cancel(b2, s2) is th)
fired2 = threading.Event()
s2.create("m", id="rid-6")
s2.attach_cancel("rid-6", fired2.set)
b2.publish(TOPIC_CONTROL_CANCEL, job_id="rid-6", payload={"reason": "stop"})
check("bus cancel fires handle", fired2.wait(timeout=2))
check("cancel_requested set", s2.get("rid-6").cancel_requested)

# --- wire_job_events: store transitions publish on the bus ---
s3 = JobStore()
b3 = Bus()
wire_job_events(b3, s3, source="test")
ev_sub = b3.subscribe("job.*")
s3.create("m", id="rid-7", kind="chat")
s3.on_output("rid-7")
s3.finish("rid-7")
topics = [ev_sub.get(timeout=1).topic for _ in range(3)]
check("created/status/done published", topics == ["job.created", "job.status", "job.done"])

# --- retention: terminal jobs pruned ---
s4 = JobStore(retain_terminal=2, retain_secs=9999)
for i in range(6):
    s4.create("m", id=f"t-{i}")
    s4.finish(f"t-{i}")
s4.create("m", id="live")
terminals = [j for j in s4.all() if j.terminal]
check("terminal overflow pruned", len(terminals) <= 3)  # 2 retained + latest batch edge

print(f"\nALL {ok} CHECKS PASSED")

# --- resurrection: download retry reuses the job id (cancelled -> running) ---
s5 = JobStore()
s5.create("m", id="r-1", kind="download")
s5.update("r-1", status="cancelled")
s5.update("r-1", status="running")   # retry_download path
check("terminal->live resurrection allowed",
      s5.get("r-1").to_dict()["status"] == "processing")
check("resurrection resets cancel_requested", not s5.get("r-1").cancel_requested)
check("resurrection clears ended_ts", s5.get("r-1").ended_ts is None)
s5.update("r-1", status="completed")
check("resurrected job can re-terminate", s5.get("r-1").to_dict()["status"] == "done")
check("legacy wire mapping", s5.get("r-1").to_legacy_dict()["status"] == "completed")

print(f"\nALL {ok} CHECKS PASSED (incl. resurrection)")
