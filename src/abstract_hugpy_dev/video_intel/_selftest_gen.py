"""Headless self-test for Phase 4 (frame_extract) + Phase 6 (generate_image) +
Phase 7 (video->frames chain). ADDITIVE — does NOT modify _selftest.py.

Run in the hugpy VM venv AS UBUNTU with prod DEFAULT_ROOT (pre-restart gate; no
root-owned files):

  lxc exec hugpy -- runuser -u ubuntu -- env DEFAULT_ROOT=/mnt/llm_storage \
    PYTHONPATH=/home/ubuntu/station/dev/abstract_hugpy_dev/src \
    /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
    /home/ubuntu/station/dev/abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/_selftest_gen.py

ISOLATED BUS DB: the LIVE service runs media_bus worker daemons against the
shared DB_PATH with the PRE-DEPLOY code (which lacks the new job names and would
claim+fail our jobs). So we repoint media_bus at a PRIVATE scratch DB before any
DB use and process jobs only with our own work_once() — fully deterministic, no
race with the live daemons.
"""
from __future__ import annotations

import logging
logging.disable(logging.INFO)

import json
import os
import subprocess
import time
from dataclasses import asdict

from abstract_hugpy_dev._platform.binaries import resolve_bin
from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT

from abstract_hugpy_dev.video_intel.media_store import ingest
from abstract_hugpy_dev.video_intel.frame_schema import make_frame_extract
from abstract_hugpy_dev.video_intel.gen_schema import (
    text_part, video_part, make_generate_image,
)
from abstract_hugpy_dev.video_intel.chains import resolve_video_parts
from abstract_hugpy_dev.video_intel import media_bus


SCRATCH = os.path.join(DEFAULT_ROOT, "video_intel", "_scratch")
FFMPEG = resolve_bin("ffmpeg") or "ffmpeg"

# --- isolate the bus from the live daemons (see module docstring) ---
os.makedirs(SCRATCH, exist_ok=True)
media_bus.DB_PATH = os.path.join(SCRATCH, "selftest_gen_jobs.db")
media_bus._initialized = False


def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"asset gen failed: {' '.join(cmd)}\n{r.stderr}")


def _gen_video():
    vid = os.path.join(SCRATCH, "gen_test.mp4")
    # 3s @ 10fps, 320x240 -> plenty of frames to sample from
    _run([FFMPEG, "-y", "-f", "lavfi",
          "-i", "testsrc=size=320x240:rate=10:duration=3", vid])
    return vid


def _drain(jid, timeout_s):
    """Process our private queue until `jid` reaches a terminal state (our own
    work_once is the only worker on this DB)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        got = media_bus.get(jid)
        if got["status"] in ("done", "failed"):
            return got
        processed = media_bus.work_once()
        if processed is None:
            time.sleep(0.1)
    return media_bus.get(jid)


def main() -> int:
    print("DEFAULT_ROOT =", DEFAULT_ROOT)
    print("scratch      =", SCRATCH)
    print("ffmpeg       =", FFMPEG)
    print("bus db       =", media_bus.DB_PATH, "(private, isolated from live daemons)")
    print("=" * 70)

    vid = _gen_video()
    vid_ref = ingest(vid)
    print("VIDEO MediaRef:", json.dumps(asdict(vid_ref)))
    assert vid_ref.kind == "video", f"video kind={vid_ref.kind}"

    # ---------------- Phase 4: frame_extract (many outputs) ----------------
    print("\n--- PHASE 4: frame_extract (fps=2 over 3s -> ~6 frames) ---")
    spec = make_frame_extract(vid_ref, fps=2.0, quality=90, fmt="jpg")
    jid = media_bus.enqueue("frame_extract", spec)
    got = _drain(jid, 120)
    print("status:", got["status"], "nframes:",
          len(got["result"]["outputs"]) if got["result"] else None)
    assert got["status"] == "done", f"status={got['status']} result={got['result']}"
    outs = got["result"]["outputs"]
    assert len(outs) > 1, f"expected >1 frame, got {len(outs)}"
    for o in outs:
        assert o["kind"] == "image", f"frame kind={o['kind']}"
        assert os.path.isfile(o["uri"]), f"missing frame file {o['uri']}"
    print("frame_extract OK ->", len(outs), "frames, first:", outs[0]["uri"])

    # ---------------- Phase 4: LOUD max_frames cap ----------------
    print("\n--- PHASE 4: LOUD cap (fps=5 over 3s => ~15 > cap 1) ---")
    spec_cap = make_frame_extract(vid_ref, fps=5.0, quality=90, fmt="jpg", max_frames=1)
    jid_cap = media_bus.enqueue("frame_extract", spec_cap)
    got_cap = _drain(jid_cap, 60)
    print("status:", got_cap["status"], "error:",
          got_cap["result"]["error"] if got_cap["result"] else None)
    assert got_cap["status"] == "failed", f"status={got_cap['status']}"
    assert got_cap["result"]["ok"] is False
    assert got_cap["result"]["error"]["code"] == "frame_cap_exceeded", \
        f"code={got_cap['result']['error']['code']}"
    print("frame_cap_exceeded OK (refused, not truncated)")

    # ---------------- Phase 6: generate_image (text only) ----------------
    print("\n--- PHASE 6: generate_image (text only, sd-turbo) ---")
    gspec = make_generate_image(
        parts=(text_part("a red cube on green grass, studio lighting"),),
        model_id="sd-turbo", width=512, height=512, steps=2, guidance=0.0, seed=0,
    )
    jid_g = media_bus.enqueue("generate_image", gspec)
    got_g = _drain(jid_g, 600)  # actually runs sd-turbo; allow generous time
    print("status:", got_g["status"])
    print("result:", json.dumps(got_g["result"]))
    gen_ran = False
    if got_g["status"] == "done":
        assert got_g["result"]["ok"] is True
        out_g = got_g["result"]["outputs"][0]
        assert out_g["kind"] == "image", f"gen kind={out_g['kind']}"
        assert os.path.isfile(out_g["uri"]), f"missing gen file {out_g['uri']}"
        gen_ran = True
        print("generate_image OK ->", out_g["uri"],
              f"({out_g['width']}x{out_g['height']})")
    else:
        # generation didn't reach a model: must be CLEAN JobError data, not a crash
        err = got_g["result"]["error"]
        assert isinstance(err, dict) and set(err) == {"code", "message", "retryable"}, \
            f"generate_image failed but not clean JobError: {got_g['result']}"
        print("generate_image did NOT reach a model; CLEAN JobError:",
              err["code"], "-", err["message"][:160])

    # ---------------- Phase 7: video part -> chain -> generate_image ----------------
    print("\n--- PHASE 7: generate_image with a VIDEO part via resolve_video_parts ---")
    gspec_v = make_generate_image(
        parts=(text_part("reimagine this scene as an oil painting"), video_part(vid_ref)),
        model_id="sd-turbo", width=512, height=512, steps=2, guidance=0.0, seed=1,
    )
    resolved = resolve_video_parts(gspec_v, uniform_n=3)
    kinds = [p.kind for p in resolved.parts]
    print("resolved part kinds:", kinds)
    assert "video" not in kinds, f"resolved spec still has a video part: {kinds}"
    assert kinds.count("image") >= 1, f"chain produced no image parts: {kinds}"
    jid_v = media_bus.enqueue("generate_image", resolved)
    got_v = _drain(jid_v, 600)
    print("status:", got_v["status"])
    print("result:", json.dumps(got_v["result"]))
    if got_v["status"] == "failed":
        # whatever the outcome, the runner must NEVER have seen a video part
        assert got_v["result"]["error"]["code"] != "unresolved_video_part", \
            "runner saw a raw video part — the chain did not resolve it!"
        err = got_v["result"]["error"]
        assert isinstance(err, dict) and set(err) == {"code", "message", "retryable"}
        print("chain resolved video (no unresolved_video_part); gen JobError:",
              err["code"])
    else:
        out_v = got_v["result"]["outputs"][0]
        assert out_v["kind"] == "image"
        assert os.path.isfile(out_v["uri"])
        print("Phase 7 generate_image OK ->", out_v["uri"])

    print("\n" + "=" * 70)
    if gen_ran:
        print("ALL PASS (frame_extract + cap + generate_image REAL image + chain)")
    else:
        print("ALL PASS (frame_extract + cap + chain OK; generate_image returned "
              "CLEAN JobError data — no real image, see above)")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - self-test reporter
        import traceback
        traceback.print_exc()
        print(f"FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)
