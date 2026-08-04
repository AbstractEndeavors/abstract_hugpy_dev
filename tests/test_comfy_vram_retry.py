"""One-shot VRAM-class retry across the comfy dispatch seam (k71).

ComfyUI generations intermittently fail on the FIRST attempt with an
OOM/allocation error caused by stale pre-eviction VRAM/pool state; an identical
second try succeeds. The seams now retry ONCE — and ONLY for that class.

Covered here:
  * vram_retry.is_retryable_vram_failure — positive-marker classification:
    the VRAM/eviction/OOM/allocation class retries; workflow-validation,
    missing-model/checkpoint, timeout, no-images, connection errors do NOT.
  * ComfyRunner.run — a retryable first failure triggers EXACTLY one retry
    (cancel first submission -> re-drive headroom hook -> settle -> resubmit);
    a second failure surfaces exactly as today; non-retryable never retries;
    a success needs no retry machinery at all.
  * ImageGenRunner.run — the in-process twin (evict-idle + release + settle).
  * identity_mesh.run — the video_intel comfy station, same contract.

Runs like the other tests here: venv/bin/python tests/test_comfy_vram_retry.py
"""
import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["HUGPY_VRAM_RETRY_SETTLE_S"] = "0"   # no real sleeping in tests

from abstract_hugpy_dev.managers.imagegen import vram_retry
from abstract_hugpy_dev.managers.imagegen.schemas import (
    GeneratedImage, ImageGenRequest)
comfy_runner = importlib.import_module(
    "abstract_hugpy_dev.managers.comfy.comfy_runner")
imagegen_runner = importlib.import_module(
    "abstract_hugpy_dev.managers.imagegen.imagegen_runner")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ===========================================================================
# Part 1 — classification: retryable ONLY for the VRAM/OOM/allocation class
# ===========================================================================
is_retryable = vram_retry.is_retryable_vram_failure

# The class as it actually reaches the seams:
retryable = [
    # comfy history execution_error, torch >= 2.4 wording (exception_type +
    # exception_message ride the JSON detail):
    RuntimeError('ComfyUI execution error: {"exception_type": '
                 '"torch.OutOfMemoryError", "exception_message": "Allocation '
                 'on device 0 would exceed allowed memory. (out of memory)"}'),
    # torch < 2.4 wording (also what the wan runners' is_oom matches):
    RuntimeError("ComfyUI execution error: CUDA out of memory. Tried to "
                 "allocate 2.50 GiB (GPU 0; 23.56 GiB total capacity)"),
    # in-process exception TYPE name carries the class even with a terse message
    type("OutOfMemoryError", (RuntimeError,), {})("CUDA error"),
    # allocator-starved library calls one step later:
    RuntimeError("cuBLAS failure: CUBLAS_STATUS_ALLOC_FAILED"),
    RuntimeError("RuntimeError: CUDA error: out of memory"),
    # comfy model_management / host-side wording for the same condition:
    RuntimeError("Not enough memory to load the model"),
    RuntimeError("Unable to allocate 512.0 MiB for output tensor"),
]
for exc in retryable:
    check(f"retryable: {str(exc)[:60]!r}", is_retryable(exc))

not_retryable = [
    # workflow validation (the /prompt 400)
    RuntimeError("ComfyUI rejected the workflow (400): {\"error\": {\"type\": "
                 "\"prompt_outputs_failed_validation\"}}"),
    # missing model/checkpoint
    RuntimeError("ComfyUI rejected the workflow (400): value not in list: "
                 "ckpt_name: 'sd15.safetensors' not in []"),
    # timeout — excluded by TYPE even though a message could be anything
    TimeoutError("ComfyUI did not finish prompt abc within 600s"),
    # finished-but-empty
    RuntimeError("ComfyUI finished but produced no images"),
    # transport failure
    ConnectionError("connection refused: 127.0.0.1:8188"),
    # generic execution error with no VRAM wording
    RuntimeError("ComfyUI execution error: mat1 and mat2 shapes cannot be "
                 "multiplied"),
]
for exc in not_retryable:
    check(f"NOT retryable: {str(exc)[:60]!r}", not is_retryable(exc))

# a timeout whose text happens to mention memory is STILL excluded by type
check("TimeoutError is excluded by type even with OOM wording",
      not is_retryable(TimeoutError("out of memory while waiting")))

# settle knob: env-clamped, never a hang
os.environ["HUGPY_VRAM_RETRY_SETTLE_S"] = "9999"
check("settle delay clamps to 10s max", vram_retry.settle_delay_s() == 10.0)
os.environ["HUGPY_VRAM_RETRY_SETTLE_S"] = "garbage"
check("settle delay ignores garbage -> 3.0s default",
      vram_retry.settle_delay_s() == 3.0)
os.environ["HUGPY_VRAM_RETRY_SETTLE_S"] = "0"


# ===========================================================================
# Part 2 — ComfyRunner.run: exactly ONE retry, and only for the VRAM class
# ===========================================================================
REQ = ImageGenRequest(request_id="req-1", model_key="comfy-ckpt", prompt="a cat")
IMAGES = [GeneratedImage(path="/tmp/x.png", width=64, height=64)]

OOM = RuntimeError('ComfyUI execution error: {"exception_type": '
                   '"torch.OutOfMemoryError", "exception_message": '
                   '"Allocation on device 0"}')
OOM.comfy_prompt_id = "prompt-first"


def make_runner(script):
    """A ComfyRunner whose _generate pops outcomes off `script` (an exception
    -> raise, anything else -> return); records the attempt count."""
    cfg = types.SimpleNamespace(model_key="comfy-ckpt", filename="ckpt.safetensors")
    runner = comfy_runner.ComfyRunner(cfg)
    calls = {"generate": 0, "cancel": []}
    def fake_generate(req):
        calls["generate"] += 1
        outcome = script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    runner._generate = fake_generate
    runner._cancel_prompt = lambda pid: calls["cancel"].append(pid)
    return runner, calls


hook_calls = []
_saved_hook = comfy_runner._COMFY_HEADROOM_HOOK
comfy_runner.set_comfy_headroom_hook(
    lambda mk, job_id=None: hook_calls.append((mk, job_id)))
try:
    # (a) retryable first failure, then success -> ok, EXACTLY one retry,
    # first submission cancelled by its prompt id, headroom hook re-driven.
    runner, calls = make_runner([OOM, IMAGES])
    res = asyncio.run(runner.run(REQ))
    check("(a) retryable-then-success returns ok=True", res.ok is True)
    check("(a) exactly one retry (two _generate calls)", calls["generate"] == 2)
    check("(a) first submission cancelled with its tagged prompt id",
          calls["cancel"] == ["prompt-first"])
    check("(a) headroom/eviction hook re-driven before the retry",
          hook_calls == [("comfy-ckpt", "req-1")])

    # (b) retryable failure TWICE -> exactly two attempts, second failure
    # surfaces exactly as today (ok=False, same error envelope).
    hook_calls.clear()
    OOM2 = RuntimeError("ComfyUI execution error: CUDA out of memory. Tried "
                        "to allocate 2.50 GiB")
    runner, calls = make_runner([OOM, OOM2])
    res = asyncio.run(runner.run(REQ))
    check("(b) second failure surfaces as ok=False", res.ok is False)
    check("(b) never a third attempt", calls["generate"] == 2)
    check("(b) the SECOND failure's wording is the surfaced error",
          "Tried to allocate" in (res.error or ""))

    # (c) non-retryable classes NEVER retry: one attempt, error as today.
    for exc, label in [
        (TimeoutError("ComfyUI did not finish prompt p1 within 600s"), "timeout"),
        (RuntimeError("ComfyUI rejected the workflow (400): value not in "
                      "list: ckpt_name"), "validation/missing-checkpoint"),
        (RuntimeError("ComfyUI finished but produced no images"), "no-images"),
    ]:
        hook_calls.clear()
        runner, calls = make_runner([exc, IMAGES])
        res = asyncio.run(runner.run(REQ))
        check(f"(c) {label}: no retry (one attempt)", calls["generate"] == 1)
        check(f"(c) {label}: surfaces ok=False", res.ok is False)
        check(f"(c) {label}: no cancel, no headroom re-drive",
              calls["cancel"] == [] and hook_calls == [])

    # (d) plain success touches none of the retry machinery.
    hook_calls.clear()
    runner, calls = make_runner([IMAGES])
    res = asyncio.run(runner.run(REQ))
    check("(d) success: one attempt, no cancel, no hook",
          res.ok and calls["generate"] == 1 and calls["cancel"] == []
          and hook_calls == [])
finally:
    comfy_runner._COMFY_HEADROOM_HOOK = _saved_hook


# ===========================================================================
# Part 3 — ImageGenRunner.run: the in-process twin of the same seam
# ===========================================================================
def make_imagegen_runner(script):
    cfg = types.SimpleNamespace(model_key="sd15")
    runner = imagegen_runner.ImageGenRunner(cfg)
    calls = {"generate": 0}
    def fake_generate(req):
        calls["generate"] += 1
        outcome = script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    runner._generate = fake_generate
    return runner, calls


settles = []
_saved_settle = imagegen_runner._settle_for_vram_retry
imagegen_runner._settle_for_vram_retry = (
    lambda cache, lock, keep: settles.append(keep))
try:
    torch_oom = type("OutOfMemoryError", (RuntimeError,), {})(
        "CUDA out of memory. Tried to allocate 20.00 MiB")
    runner, calls = make_imagegen_runner([torch_oom, IMAGES])
    res = asyncio.run(runner.run(
        ImageGenRequest(request_id="r2", model_key="sd15", prompt="a dog")))
    check("(e) imagegen retryable-then-success -> ok, exactly one retry",
          res.ok is True and calls["generate"] == 2)
    check("(e) settle re-drove eviction for the generating model",
          settles == ["sd15"])

    settles.clear()
    runner, calls = make_imagegen_runner(
        [ValueError("image-to-image requires an init image"), IMAGES])
    res = asyncio.run(runner.run(
        ImageGenRequest(request_id="r3", model_key="sd15", prompt="a dog")))
    check("(f) imagegen non-retryable: one attempt, ok=False, no settle",
          res.ok is False and calls["generate"] == 1 and settles == [])
finally:
    imagegen_runner._settle_for_vram_retry = _saved_settle

# The REAL settle helper is safe on a no-GPU box (empty cache, settle=0).
imagegen_runner._settle_for_vram_retry({}, imagegen_runner.threading.Lock(),
                                       "sd15")
check("(g) real _settle_for_vram_retry no-ops cleanly with no GPU/pipelines",
      True)


# ===========================================================================
# Part 4 — identity_mesh.run: the video_intel comfy station
# ===========================================================================
identity_mesh = importlib.import_module(
    "abstract_hugpy_dev.video_intel.runners.identity_mesh")

SPEC = types.SimpleNamespace(slug="hero", recon_id="rc1", view_ids=("v0",))

_saved = (identity_mesh._resolve_view_uris, identity_mesh._submit_and_wait,
          identity_mesh._cancel_and_settle)
try:
    identity_mesh._resolve_view_uris = lambda slug, rid, vids: {"v0": "/tmp/v0.png"}
    mesh_calls = {"submit": 0, "cancel": []}
    mesh_oom = RuntimeError('ComfyUI execution error: {"exception_type": '
                            '"torch.OutOfMemoryError"}')
    mesh_oom.comfy_prompt_id = "mesh-first"
    script = [mesh_oom,
              RuntimeError("ComfyUI execution error: missing node "
                           "Hunyuan3DShapeGenerator")]
    def fake_submit(payload):
        mesh_calls["submit"] += 1
        raise script.pop(0)
    identity_mesh._submit_and_wait = fake_submit
    identity_mesh._cancel_and_settle = (
        lambda pid, job: mesh_calls["cancel"].append(pid))

    out = identity_mesh.run(SPEC)
    check("(h) mesh: retryable first failure -> exactly one retry",
          mesh_calls["submit"] == 2)
    check("(h) mesh: first submission cancelled by its prompt id",
          mesh_calls["cancel"] == ["mesh-first"])
    check("(h) mesh: second (non-VRAM) failure surfaces as error data",
          out["ok"] is False and "missing node" in out["error"]["message"])

    mesh_calls = {"submit": 0, "cancel": []}
    def fake_submit_bad(payload):
        mesh_calls["submit"] += 1
        raise RuntimeError("ComfyUI execution error: value not in list")
    identity_mesh._submit_and_wait = fake_submit_bad
    identity_mesh._cancel_and_settle = (
        lambda pid, job: mesh_calls["cancel"].append(pid))
    out = identity_mesh.run(SPEC)
    check("(i) mesh: non-retryable does NOT retry (one attempt, no cancel)",
          mesh_calls["submit"] == 1 and mesh_calls["cancel"] == []
          and out["ok"] is False)
finally:
    (identity_mesh._resolve_view_uris, identity_mesh._submit_and_wait,
     identity_mesh._cancel_and_settle) = _saved

os.environ.pop("HUGPY_VRAM_RETRY_SETTLE_S", None)

print(f"\nall {ok} checks passed")
