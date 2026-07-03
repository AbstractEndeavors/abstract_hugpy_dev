"""Headless self-test for the Video Intelligence backbone (Phases 1 & 2).

Run in the hugpy VM venv:

  lxc exec hugpy -- env DEFAULT_ROOT=/mnt/llm_storage \
    PYTHONPATH=/home/ubuntu/station/dev/abstract_hugpy_dev/src \
    /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
    /home/ubuntu/station/dev/abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/_selftest.py

Generates KNOWN test assets with ffmpeg under DEFAULT_ROOT/video_intel/_scratch
(makedirs only — deletes nothing else), exercises ingest + crop through the bus,
and prints a final ALL PASS / FAIL line.
"""
from __future__ import annotations

# Silence the noisy INFO logging emitted while the constants/model registry
# imports, so the evidence below is readable.
import logging
logging.disable(logging.INFO)

import json
import os
import subprocess
from dataclasses import asdict

from abstract_hugpy_dev._platform.binaries import resolve_bin
from abstract_hugpy_dev.imports.src.constants.constants import DEFAULT_ROOT

from abstract_hugpy_dev.video_intel.media_store import ingest
from abstract_hugpy_dev.video_intel.crop_schema import (
    SpatialRegion, TemporalRegion, make_crop,
)
from abstract_hugpy_dev.video_intel import media_bus


SCRATCH = os.path.join(DEFAULT_ROOT, "video_intel", "_scratch")
FFMPEG = resolve_bin("ffmpeg") or "ffmpeg"


def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"asset gen failed: {' '.join(cmd)}\n{r.stderr}")


def _gen_assets():
    os.makedirs(SCRATCH, exist_ok=True)
    img = os.path.join(SCRATCH, "test.png")
    aud = os.path.join(SCRATCH, "test.wav")
    vid = os.path.join(SCRATCH, "test.mp4")
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=1",
          "-frames:v", "1", img])
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
          "-ar", "16000", "-ac", "1", aud])
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
          vid])
    return img, aud, vid


def _approx(a, b, tol=0.15):
    return a is not None and abs(a - b) <= tol


def main() -> int:
    print("DEFAULT_ROOT =", DEFAULT_ROOT)
    print("scratch      =", SCRATCH)
    print("ffmpeg       =", FFMPEG)
    print("bus db       =", media_bus.DB_PATH)
    print("=" * 70)

    img, aud, vid = _gen_assets()

    # -------------------- Phase 1: ingest --------------------
    print("\n--- PHASE 1: ingest ---")
    img_ref = ingest(img)
    aud_ref = ingest(aud)
    vid_ref = ingest(vid)
    print("IMAGE MediaRef:", json.dumps(asdict(img_ref)))
    print("AUDIO MediaRef:", json.dumps(asdict(aud_ref)))
    print("VIDEO MediaRef:", json.dumps(asdict(vid_ref)))

    assert img_ref.kind == "image", f"image kind={img_ref.kind}"
    assert img_ref.width == 640 and img_ref.height == 480, f"image dims={img_ref.width}x{img_ref.height}"

    assert aud_ref.kind == "audio", f"audio kind={aud_ref.kind}"
    assert _approx(aud_ref.duration_s, 3.0), f"audio duration={aud_ref.duration_s}"
    assert aud_ref.sample_rate == 16000, f"audio sample_rate={aud_ref.sample_rate}"
    assert aud_ref.channels == 1, f"audio channels={aud_ref.channels}"

    assert vid_ref.kind == "video", f"video kind={vid_ref.kind}"
    assert vid_ref.width == 320 and vid_ref.height == 240, f"video dims={vid_ref.width}x{vid_ref.height}"
    assert _approx(vid_ref.duration_s, 2.0), f"video duration={vid_ref.duration_s}"
    assert _approx(vid_ref.fps_native, 25.0, tol=0.5), f"video fps={vid_ref.fps_native}"
    print("PHASE 1 asserts OK")

    # -------------------- Phase 2: spatial crop (image) --------------------
    print("\n--- PHASE 2: spatial crop (image) ---")
    spec = make_crop(img_ref, spatial=SpatialRegion(x=100, y=50, w=200, h=150))
    jid = media_bus.enqueue("crop", spec)
    processed = media_bus.work_once()
    got = media_bus.get(jid)
    print("job_id:", jid, "processed:", processed)
    print("round-trip:", json.dumps(got))
    assert got["status"] == "done", f"status={got['status']}"
    assert got["result"]["ok"] is True, f"ok={got['result']['ok']}"
    out0 = got["result"]["outputs"][0]
    assert out0["width"] == 200 and out0["height"] == 150, f"out dims={out0['width']}x{out0['height']}"
    assert os.path.isfile(out0["uri"]), f"missing output file {out0['uri']}"
    print("spatial crop OK ->", out0["uri"])

    # -------------------- Phase 2: temporal crop (audio) --------------------
    print("\n--- PHASE 2: temporal crop (audio) ---")
    spec_t = make_crop(aud_ref, temporal=TemporalRegion(start_s=0.5, end_s=2.0))
    jid_t = media_bus.enqueue("crop", spec_t)
    media_bus.work_once()
    got_t = media_bus.get(jid_t)
    print("job_id:", jid_t)
    print("round-trip:", json.dumps(got_t))
    assert got_t["status"] == "done", f"status={got_t['status']}"
    assert got_t["result"]["ok"] is True
    out_t = got_t["result"]["outputs"][0]
    assert _approx(out_t["duration_s"], 1.5), f"audio-crop duration={out_t['duration_s']}"
    print("temporal crop OK -> duration", out_t["duration_s"])

    # -------------------- Negative 1: local raise --------------------
    print("\n--- NEGATIVE 1: image + temporal must raise (local validation) ---")
    raised = False
    try:
        make_crop(img_ref, temporal=TemporalRegion(start_s=0.0, end_s=1.0))
    except ValueError as e:
        raised = True
        print("got expected ValueError:", e)
    assert raised, "make_crop(image, temporal) did NOT raise"

    # -------------------- Negative 2: error as data --------------------
    print("\n--- NEGATIVE 2: out-of-bounds bbox -> JobError data (no raise) ---")
    spec_bad = make_crop(img_ref, spatial=SpatialRegion(x=9999, y=50, w=200, h=150))
    jid_bad = media_bus.enqueue("crop", spec_bad)
    media_bus.work_once()
    got_bad = media_bus.get(jid_bad)
    print("job_id:", jid_bad)
    print("round-trip:", json.dumps(got_bad))
    assert got_bad["status"] == "failed", f"status={got_bad['status']}"
    assert got_bad["result"]["ok"] is False, f"ok={got_bad['result']['ok']}"
    err = got_bad["result"]["error"]
    assert isinstance(err, dict), f"error is not a dict: {type(err)}"
    assert set(err.keys()) == {"code", "message", "retryable"}, f"error keys={set(err.keys())}"
    print("error-as-data OK:", err["code"])

    print("\n" + "=" * 70)
    print("ALL PASS")
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
