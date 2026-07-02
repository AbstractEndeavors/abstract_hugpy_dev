"""Integration: boot the real flask app, exercise queue/jobs/cancel surfaces."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.flask_app.wsgi_app import get_hugpy_flask
app = get_hugpy_flask()
c = app.test_client()

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

from abstract_hugpy_dev.comms import job_store, bus, TOPIC_CONTROL_CANCEL

# wire_cancel/wire_job_events ran in the factory
check("store->bus adapter wired", job_store.on_change is not None)

# --- queue view (activity shim) ---
r = c.get("/llm/queue")
check("GET /llm/queue 200", r.status_code == 200)
q = r.get_json()
check("queue shape", "active" in q and "counts" in q and
      set(q["counts"]) == {"waiting", "active", "total"})

# simulate a chat stream lifecycle through the shim (what /chat/stream does)
from abstract_hugpy_dev.managers.dispatch import activity
activity.begin("it-1", "some-model", "Some Model")
q = c.get("/llm/queue").get_json()
mine = [e for e in q["active"] if e["request_id"] == "it-1"]
check("begun request visible as waiting",
      len(mine) == 1 and mine[0]["state"] == "waiting")
check("old snapshot keys intact",
      set(mine[0]) == {"request_id", "model_key", "model", "kind", "state",
                       "elapsed", "wait", "tokens"})
activity.on_token("it-1"); activity.on_token("it-1")
q = c.get("/llm/queue").get_json()
mine = [e for e in q["active"] if e["request_id"] == "it-1"][0]
check("active after tokens", mine["state"] == "active" and mine["tokens"] == 2)
activity.end("it-1")
q = c.get("/llm/queue").get_json()
check("gone from queue after end",
      not [e for e in q["active"] if e["request_id"] == "it-1"])

# --- downloads /jobs surface: legacy wire, download-only ---
job_store.create("some-model", id="dl-1", kind="download", transport="web")
job_store.update("dl-1", status="running", progress=0.4)
job_store.create("m", id="chat-x", kind="chat")   # must NOT appear on /jobs
r = c.get("/jobs")
jobs = {j["id"]: j for j in r.get_json()}
check("download job on /jobs with legacy status",
      jobs.get("dl-1", {}).get("status") == "running")
check("chat job not on /jobs", "chat-x" not in jobs)
r = c.get("/jobs/dl-1")
check("GET /jobs/<id> legacy shape", r.get_json()["status"] == "running")
job_store.finish("chat-x")

# --- chat cancel route: local job gets cancel_requested via the bus ---
fired = []
job_store.create("m", id="cx-1", kind="chat", transport="web")
job_store.attach_cancel("cx-1", lambda: fired.append(1))
r = c.post("/llm/chat/cancel/cx-1")
check("cancel route 200 + cancelled:true",
      r.status_code == 200 and r.get_json()["cancelled"] is True)
deadline = time.time() + 2
while not fired and time.time() < deadline:
    time.sleep(0.02)
check("cancel handle fired via bus", fired == [1])
check("job flagged cancel_requested", job_store.get("cx-1").cancel_requested)
job_store.finish("cx-1")
check("teardown resolves cancelled",
      job_store.get("cx-1").to_dict()["status"] == "cancelled")

# unknown request: no local job, no workers -> cancelled false
r = c.post("/llm/chat/cancel/nope")
check("unknown cancel -> false", r.get_json()["cancelled"] is False)

# --- bus saw the lifecycle events ---
sub = bus.subscribe("job.*")
job_store.create("m", id="ev-1", kind="chat")
m = sub.get(timeout=1)
check("job.created published on app bus", m is not None and m.job_id == "ev-1")
job_store.finish("ev-1")
m2 = sub.get(timeout=1)
check("job.done published", m2 is not None and m2.topic == "job.done")
sub.close()

print(f"\nALL {ok} CHECKS PASSED")

# ---------------- Phase-0 F2/F3/F4 route surfaces ----------------
import tempfile as _tf, os as _os
_tmp = _tf.mkdtemp(prefix="wiring-found-")
from abstract_hugpy_dev.comms import principals as _pr, settings as _se
_pr.principal_store._path = _os.path.join(_tmp, "principals.json")
_se.settings_store._path = _os.path.join(_tmp, "settings.json")
_se.settings_store._cache = None

# F4 settings control API
r = c.post("/settings/discord.channels/42", json={"value": {"respond": "all"}})
check("settings write 200", r.status_code == 200)
r = c.get("/settings/discord.channels")
check("settings ns read", r.get_json()["values"].get("42") == {"respond": "all"})
r = c.post("/settings/discord.channels/42", json={"merge": {"personality": "pirate"}, "value": None})
check("settings merge", r.get_json()["value"]["personality"] == "pirate")
r = c.get("/settings")
check("namespaces listed", "discord.channels" in r.get_json()["namespaces"])

# bot M2M pref write lands in the settings store
r = c.post("/discord/prefs", json={"user_id": "777", "model_key": "qwen"})
check("discord prefs M2M write", r.status_code == 200)
r = c.get("/settings/discord.users/777")
check("pref visible in settings", r.get_json()["value"] == {"model": "qwen"})

# F2 principals lifecycle over HTTP
r = c.post("/auth/principals", json={"kind": "user", "name": "bob", "groups": ["media"]})
check("principal create 200", r.status_code == 200)
body = r.get_json()
check("token returned once", (body.get("token") or "").startswith("hpp_"))
tok = body["token"]
r = c.get("/auth/whoami", headers={"Authorization": f"Bearer {tok}"})
check("whoami resolves principal", (r.get_json()["principal"] or {}).get("name") == "bob")
r = c.post("/auth/discord-link", json={"token": tok, "discord_user_id": "555"})
check("discord-link 200", r.status_code == 200 and r.get_json()["linked"])
r = c.post("/auth/discord-link", json={"token": "hpp_bad", "discord_user_id": "556"})
check("discord-link bad token 401", r.status_code == 401)

# F5/CON-01 unified jobs endpoint
job_store.create("m", id="uj-1", kind="chat", transport="cli", principal="pr_x")
r = c.get("/llm/jobs?transport=cli")
rows = r.get_json()["jobs"]
check("unified jobs filters by transport",
      any(j["id"] == "uj-1" and j["principal"] == "pr_x" for j in rows))
job_store.finish("uj-1")

# F3.1 model meta endpoint
r = c.get("/models/Qwen2.5-3B-Instruct-GGUF/meta")
if r.status_code == 200:
    m = r.get_json()
    check("meta has quant + recommended",
          "recommended" in m and m.get("params_b") == 3.0)
else:
    check("meta 404 for unknown key only", r.status_code == 404)

# F3.2 worker registry surfacing
r = c.get("/llm/workers")
check("workers rows carry version_ok",
      all("version_ok" in w for w in r.get_json()))

print(f"\nALL {ok} CHECKS PASSED (incl. foundations)")
