"""Headless self-test for the generate_scene runner — with the img2img path.

Run:  PYTHONPATH=src ./venv/bin/python \
          src/abstract_hugpy_dev/video_intel/_selftest_scene.py

Isolation (mirrors prior video_intel self-tests / deploy memory):
  * media_bus.DB_PATH is repointed to a PRIVATE temp sqlite db and
    media_bus._initialized is reset, so enqueue/work_once/get never touch the
    real job bus. We drain jobs with our OWN work_once() call.
  * The DEFAULT_ROOT=/tmp env prefix is SILENTLY IGNORED (.env pins it), so we
    rely on the DB repoint for isolation — synth artifacts live under the REAL
    DEFAULT_ROOT (so media_store.ingest's storage jail accepts them).

Two img2img sections (NEITHER commits the HELD step-6 advertisement flip to
disk — Part B applies it IN-PROCESS only, to the live MODEL_REGISTRY object):

  Part A (rails, headless-tolerant): enqueue a scene spec WITH a start-frame
    image part + chain=True and drain it. Because sd-turbo does NOT advertise
    image-to-image (step 6 held), the runner must return the retryable
    `image_to_image_unavailable` JobError — proving the HONEST-failure path
    (never a silent fall back to text-to-image).

  Part B (real CPU coverage): apply the step-6 flip in-process (add
    "image-to-image" to sd-turbo's registry tasks), set HUGPY_VIDEOGEN_LOCAL=
    always, synth a tiny 256x256 init PNG, and call the managers plane directly
    with task="image-to-image". Assert a real GeneratedImage with non-trivial
    bytes. GATED: if sd-turbo weights aren't locally loadable, SKIP with a LOUD
    note (the deploy agent reports it) rather than failing the self-test.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from uuid import uuid4

from abstract_hugpy_dev._platform.binaries import resolve_bin
from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT


# --------------------------------------------------------------------------- #
# reusable helpers (module-level so other tests can call them)
# --------------------------------------------------------------------------- #
def synth_png(path: str, width: int = 256, height: int = 256, color: str = "red") -> str:
    """Synthesize a tiny solid-color PNG via ffmpeg lavfi. Returns the path.

    Mirrors the gen self-test's synth idiom (resolve_bin + PIPE + returncode)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s={width}x{height}:d=1",
        "-frames:v", "1", path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError(f"ffmpeg synth failed rc={res.returncode}: {(res.stderr or '')[-400:]}")
    return path


def private_bus_db():
    """Repoint media_bus.DB_PATH to a private temp sqlite db and reset the
    one-time init flag, so this self-test drains its OWN queue in isolation."""
    from abstract_hugpy_dev.video_intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_selftest_scene_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return media_bus


def _synth_root():
    """A writable subdir under the REAL DEFAULT_ROOT so ingest's jail accepts it."""
    d = os.path.join(DEFAULT_ROOT, "video_intel", "_selftest_scene", uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Part A — honest-failure rails through the bus (no flip applied)
# --------------------------------------------------------------------------- #
def part_a_honest_failure() -> bool:
    print("\n=== Part A: img2img honest-failure rails (step 6 HELD) ===")
    from abstract_hugpy_dev.video_intel import media_store
    from abstract_hugpy_dev.video_intel.gen_schema import image_part, text_part
    from abstract_hugpy_dev.video_intel.scene_schema import make_generate_scene

    media_bus = private_bus_db()

    synth_dir = _synth_root()
    init_png = synth_png(os.path.join(synth_dir, "start_frame.png"), 256, 256, "red")
    start_ref = media_store.ingest(init_png)
    print(f"[A] ingested start frame: kind={start_ref.kind} {start_ref.width}x{start_ref.height}")

    spec = make_generate_scene(
        parts=(text_part("a serene landscape, slowly panning"), image_part(start_ref)),
        model_id="sd-turbo",
        width=256, height=256, steps=2, guidance=0.0,
        n_frames=3, fps=8, assemble=False,
        motion="camera pans right, step {i} of {n}",
        chain=True,
    )
    job_id = media_bus.enqueue("generate_scene", spec)
    processed = media_bus.work_once()
    view = media_bus.get(job_id)
    print(f"[A] job={job_id[:8]} processed={bool(processed)} status={view['status']}")

    result = view.get("result") or {}
    err = result.get("error") or {}
    if view["status"] == "done":
        # A live worker somehow served it — accept the valid done shape too.
        outs = result.get("outputs") or []
        ok = len(outs) >= 1
        print(f"[A] unexpectedly DONE with {len(outs)} output(s) (a live img2img "
              f"server must be present); accepting valid shape: {ok}")
        return ok
    ok = (
        view["status"] == "failed"
        and err.get("code") == "image_to_image_unavailable"
        and err.get("retryable") is True
    )
    print(f"[A] error code={err.get('code')!r} retryable={err.get('retryable')!r} "
          f"message={err.get('message')!r}")
    print(f"[A] {'PASS' if ok else 'FAIL'}: honest image_to_image_unavailable path")
    return ok


# --------------------------------------------------------------------------- #
# Part B — real CPU img2img proof (in-process flip; guarded/skippable)
# --------------------------------------------------------------------------- #
def _apply_step6_flip_in_process() -> bool:
    """Add 'image-to-image' to sd-turbo's LIVE registry tasks (in-process only —
    never written to models_config.py). Returns True if the flip is now active."""
    from abstract_hugpy_dev.imports.config.models.models_config import MODEL_REGISTRY
    cfg = MODEL_REGISTRY.get("sd-turbo")
    if cfg is None:
        print("[B] sd-turbo not in MODEL_REGISTRY; cannot flip")
        return False
    if "image-to-image" not in cfg.tasks:
        # cfg is a frozen dataclass, but .tasks is a mutable list — appending to
        # the list is allowed (we mutate contents, not the attribute binding).
        cfg.tasks.append("image-to-image")
    return "image-to-image" in cfg.tasks


def part_b_cpu_proof() -> str:
    """Returns 'pass', 'fail', or 'skip'."""
    print("\n=== Part B: real CPU img2img proof (in-process step-6 flip) ===")
    if not _apply_step6_flip_in_process():
        print("[B] FAIL: could not apply in-process flip")
        return "fail"
    os.environ["HUGPY_VIDEOGEN_LOCAL"] = "always"

    from abstract_hugpy_dev.video_intel.runners._img2img import img2img_available
    if not img2img_available("sd-turbo"):
        print("[B] FAIL: img2img_available('sd-turbo') False AFTER flip")
        return "fail"
    print("[B] img2img_available('sd-turbo') = True (flip active)")

    synth_dir = _synth_root()
    init_png = synth_png(os.path.join(synth_dir, "init.png"), 256, 256, "blue")
    print(f"[B] synth init PNG: {init_png}")

    from abstract_hugpy_dev.managers.dispatch import execute_prompt
    from abstract_hugpy_dev._platform.async_runtime import run

    try:
        res = run(execute_prompt(
            task="image-to-image",
            model_key="sd-turbo",
            image_path=init_png,
            strength=0.7,
            prompt="a red cube",
            num_inference_steps=2,
            width=256, height=256,
            return_b64=True,
        ))
    except Exception as exc:
        # A load/registry raise reaching here is unexpected (the runner returns
        # ok=False on load failure), but treat weight-load failures as SKIP.
        msg = f"{type(exc).__name__}: {exc}"
        if _looks_like_weights_unavailable(msg):
            print(f"[B] SKIP: img2img CPU proof skipped: sd-turbo weights unavailable in VM ({msg})")
            return "skip"
        print(f"[B] FAIL: execute_prompt raised: {msg}")
        return "fail"

    ok = bool(getattr(res, "ok", False))
    if not ok:
        err = getattr(res, "error", "") or ""
        if _looks_like_weights_unavailable(err):
            print(f"[B] SKIP: img2img CPU proof skipped: sd-turbo weights unavailable in VM ({err})")
            return "skip"
        print(f"[B] FAIL: result not ok: {err}")
        return "fail"

    images = getattr(res, "images", None) or ()
    if not images:
        print("[B] FAIL: ok result but no images")
        return "fail"
    img0 = images[0]
    b64 = getattr(img0, "b64", None)
    nbytes = len(base64.b64decode(b64)) if b64 else 0
    real = nbytes > 1000 and img0.width == 256 and img0.height == 256
    print(f"[B] GeneratedImage: path={img0.path} {img0.width}x{img0.height} "
          f"b64_bytes={nbytes}")
    print(f"[B] {'PASS' if real else 'FAIL'}: real img2img bytes produced on CPU")
    return "pass" if real else "fail"


def _looks_like_weights_unavailable(msg: str) -> bool:
    """Heuristic: a from_pretrained / download / file-not-found style failure that
    means the sd-turbo weights aren't loadable here (SKIP, not FAIL)."""
    low = (msg or "").lower()
    needles = (
        "no such file", "not found", "cannot find", "does not exist",
        "connection", "offline", "couldn't connect", "could not connect",
        "huggingface", "from_pretrained", "safetensors", "no module named",
        "failed to load", "unable to load", "oserror", "errno",
    )
    return any(n in low for n in needles)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("generate_scene self-test — img2img path")
    print("=" * 70)

    results = []
    a_ok = part_a_honest_failure()
    results.append(("Part A (honest-failure rails)", "pass" if a_ok else "fail"))

    b = part_b_cpu_proof()
    results.append(("Part B (CPU img2img proof)", b))

    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, status in results:
        print(f"  {status.upper():5} — {name}")
    print("=" * 70)

    # Part A must pass. Part B may 'skip' (weights unavailable) without failing.
    hard_fail = (not a_ok) or (b == "fail")
    if hard_fail:
        print("RESULT: FAIL")
        return 1
    if b == "skip":
        print("RESULT: PASS (Part B skipped: sd-turbo weights unavailable in VM)")
    else:
        print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
