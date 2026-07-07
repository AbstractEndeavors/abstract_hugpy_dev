"""Per-box "never serve locally" policy gate (HUGPY_NO_LOCAL_SERVING).

The central API/UI/dev station must never host or serve a model in its own
process (the path that spawned an OOM'ing llama-server on the 11 GiB VM). The
gate is a PER-BOX opt-in, default OFF, so the identical package still serves on
the worker boxes (ae/computron/op) — a hardcoded-off would kill the fleet on the
next release.

This asserts the gate at each choke point AND that default-off is a no-op:
  * policy.no_local_serving() flips only on the env flag; off by default.
  * slots.slots_enabled() -> False under policy, regardless of SLOT_COUNT.
  * serve.serve_endpoint() -> None under policy (no local slot/swap endpoint).
  * resolvers.remote DelegatingRunner refuses the local fallback under policy
    (both run() raise and stream() ErrorEvent), for every task uniformly.
  * video_intel guard_gpu_worker refuses in-process generation under policy even
    with no worker provider registered (standalone posture).

Runs like the other tests here:
    venv/bin/python tests/test_no_local_serving.py
"""
import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def _set_policy(on: bool):
    if on:
        os.environ["HUGPY_NO_LOCAL_SERVING"] = "true"
    else:
        os.environ.pop("HUGPY_NO_LOCAL_SERVING", None)


# --- policy primitive ------------------------------------------------------
policy = importlib.import_module("abstract_hugpy_dev.managers.serve.policy")

_set_policy(False)
check("default OFF (unset)", policy.no_local_serving() is False)
os.environ["HUGPY_NO_LOCAL_SERVING"] = "false"
check("explicit false stays OFF", policy.no_local_serving() is False)
for v in ("1", "true", "yes", "on", "TRUE", " On "):
    os.environ["HUGPY_NO_LOCAL_SERVING"] = v
    check(f"flag {v!r} turns policy ON", policy.no_local_serving() is True)
msg = policy.local_serving_error("Qwen2.5-VL-3B-Instruct-GGUF")
check("error names the model + the flag + a fix",
      "Qwen2.5-VL-3B-Instruct-GGUF" in msg
      and "HUGPY_NO_LOCAL_SERVING" in msg and "worker" in msg)


# --- slots.slots_enabled ---------------------------------------------------
slots = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")

os.environ["SLOT_COUNT"] = "2"           # a pool WOULD exist absent the policy
_set_policy(False)
check("SLOT_COUNT=2 + policy off -> slots enabled", slots.slots_enabled() is True)
check("SLOT_COUNT=2 -> _slot_count()==2 (0 does NOT fall through to a default)",
      slots._slot_count() == 2)
os.environ["SLOT_COUNT"] = "0"
check("SLOT_COUNT=0 honored (no default fallthrough)", slots._slot_count() == 0)
check("SLOT_COUNT=0 -> slots disabled", slots.slots_enabled() is False)
os.environ["SLOT_COUNT"] = "2"
_set_policy(True)
check("policy ON forces slots OFF even with SLOT_COUNT=2",
      slots.slots_enabled() is False)
check("policy ON -> slot_urls() empty (nothing to route to)",
      slots.slot_urls() == [])


# --- serve.serve_endpoint --------------------------------------------------
serve = importlib.import_module("abstract_hugpy_dev.managers.serve.serve")

_set_policy(True)
# Short-circuits BEFORE any registry resolution, so a dummy key is safe.
check("policy ON -> serve_endpoint() returns None (no local endpoint)",
      serve.serve_endpoint("any-model-key") is None)


# --- resolvers.remote DelegatingRunner ------------------------------------
remote = importlib.import_module("abstract_hugpy_dev.managers.resolvers.remote")

fw_runners = remote.FRAMEWORK_RUNNERS
framework, task = next(iter(fw_runners))          # any registered (fw, task)
Runner = remote.make_delegating_runner(framework, task)
runner = Runner(types.SimpleNamespace(model_key="test-model"))

# Force "no worker selected" without touching the provider seam.
remote._select = lambda mk, pool=None: (None, None)

req = types.SimpleNamespace(request_id="rid-1", pool=None)

_set_policy(True)
raised = None
try:
    asyncio.run(runner.run(req))
except RuntimeError as exc:
    raised = str(exc)
check("policy ON -> run() refuses local fallback",
      raised is not None and "HUGPY_NO_LOCAL_SERVING" in raised)

async def _collect_stream():
    evs = []
    async for ev in runner.stream(req):
        evs.append(ev)
    return evs
evs = asyncio.run(_collect_stream())
check("policy ON -> stream() yields exactly one error event",
      len(evs) == 1 and getattr(evs[0], "type", None) == "error")
check("policy ON -> stream() error carries the policy message",
      "HUGPY_NO_LOCAL_SERVING" in getattr(evs[0], "message", ""))

# policy OFF: the runner must proceed to the local runner, not refuse.
_set_policy(False)
sentinel = object()
runner._local = types.SimpleNamespace(run=lambda req: sentinel)
got = asyncio.run(runner.run(req))
check("policy OFF -> run() proceeds to the local runner (no refusal)",
      got is sentinel)


# --- video_intel guard_gpu_worker -----------------------------------------
guard_mod = importlib.import_module(
    "abstract_hugpy_dev.video_intel.runners._gpu_guard")

# No worker provider registered (standalone posture): historically this PROCEEDS.
remote.set_worker_provider(None) if False else None   # leave provider as-is (None)
_set_policy(False)
os.environ.pop("HUGPY_VIDEOGEN_LOCAL", None)
res = guard_mod.guard_gpu_worker("some-diffusion-model", "job-1")
check("policy OFF + no provider -> generation proceeds (res None)", res is None)

_set_policy(True)
res = guard_mod.guard_gpu_worker("some-diffusion-model", "job-1")
check("policy ON -> generation refused even with no provider",
      res is not None and res.ok is False
      and res.error.code == "local_serving_disabled")

os.environ["HUGPY_VIDEOGEN_LOCAL"] = "always"
res = guard_mod.guard_gpu_worker("some-diffusion-model", "job-1")
check("policy ON + HUGPY_VIDEOGEN_LOCAL=always -> still allowed", res is None)
os.environ.pop("HUGPY_VIDEOGEN_LOCAL", None)


# --- get._build_runner (gguf choke) — heavy import, guard/skip on ImportError
try:
    get_mod = importlib.import_module(
        "abstract_hugpy_dev.managers.llama.runners.get")
except Exception as exc:                          # llama_cpp/torch may be absent
    print(f"  skip - get._build_runner import unavailable ({type(exc).__name__})")
else:
    _set_policy(True)
    raised = None
    try:
        get_mod._build_runner("test-model")
    except get_mod.LocalEngineUnavailable as exc:
        raised = str(exc)
    except Exception as exc:                       # any other error = gate leaked
        raised = f"WRONG:{type(exc).__name__}:{exc}"
    check("policy ON -> _build_runner raises LocalEngineUnavailable",
          raised is not None and raised.startswith("local model serving is disabled"))


_set_policy(False)
os.environ.pop("SLOT_COUNT", None)
print(f"\nall {ok} checks passed")
