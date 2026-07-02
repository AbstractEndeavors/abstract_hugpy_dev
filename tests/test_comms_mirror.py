"""Cross-process mirror: two JobStores sharing one SQLite file simulate two
gunicorn workers — cancel lands on the wrong one, queue views merge."""
import sys, tempfile, os, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.comms.jobs import JobStore
from abstract_hugpy_dev.comms.shared import SqliteMirror

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

db = os.path.join(tempfile.mkdtemp(prefix="comms-mirror-"), "comms.db")
A = JobStore(mirror=SqliteMirror(db))   # "gunicorn worker A" — owns the stream
B = JobStore(mirror=SqliteMirror(db))   # "gunicorn worker B" — gets the POST

# A serves a stream; B has never heard of the job
fired = threading.Event()
A.create("qwen", id="x-1", kind="chat", transport="web")
A.attach_cancel("x-1", fired.set)
check("B sees no local job", B.get("x-1") is None)

# queue merge: B's snapshot shows A's live job via the mirror
snapB = B.snapshot()
check("B snapshot merges A's job",
      any(d["id"] == "x-1" for d in snapB))
check("B counts include it", B.counts()["total"] >= 1)

# cancel lands on B -> returns True (remote live), flag raised on the mirror
check("cancel on B returns True", B.cancel("x-1", reason="user stop"))
check("A's handle not fired yet (flag only)", not fired.is_set())

# A's watcher notices within ~2s and fires the local handle
check("A's watcher fires the handle", fired.wait(timeout=4))
check("A's job flagged", A.get("x-1").cancel_requested)
A.finish("x-1")
check("A teardown -> cancelled",
      A.get("x-1").to_dict()["status"] == "cancelled")

# terminal state propagates: B's view no longer shows it (after A's upsert)
time.sleep(0.1)
check("gone from B's snapshot once terminal",
      not any(d["id"] == "x-1" for d in B.snapshot()))

# cancel-before-attach, cross-process: flag first, attach later on A
B2 = JobStore(mirror=SqliteMirror(db))
A2 = JobStore(mirror=SqliteMirror(db))
A2.create("m", id="x-2", kind="chat")
check("B2 flags unattached job", B2.cancel("x-2"))
late = threading.Event()
A2.attach_cancel("x-2", late.set)   # attach checks the mirror flag directly
check("late attach fires from mirror flag", late.wait(timeout=4))

# unknown id: nothing anywhere -> False
check("unknown id cancels False", not B.cancel("nope"))

# resurrection clears the shared flag
A3 = JobStore(mirror=SqliteMirror(db))
A3.create("m", id="x-3", kind="download")
A3.cancel("x-3")
A3.finish("x-3")
A3.update("x-3", status="running")          # retry
check("flag cleared on resurrection",
      not A3.mirror.cancel_requested("x-3"))

print(f"\nALL {ok} CHECKS PASSED")
