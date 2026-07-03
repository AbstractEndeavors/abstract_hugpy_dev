"""Headless self-test for Phase 5 (audio_extract). ADDITIVE — does NOT modify
_selftest.py or _selftest_gen.py.

Run in the hugpy VM venv AS UBUNTU with prod DEFAULT_ROOT (pre-restart gate; no
root-owned files):

  lxc exec hugpy -- runuser -u ubuntu -- env DEFAULT_ROOT=/mnt/llm_storage \
    PYTHONPATH=/home/ubuntu/station/dev/abstract_hugpy_dev/src \
    /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
    /home/ubuntu/station/dev/abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/_selftest_audio.py

ISOLATED BUS DB: the LIVE service runs media_bus worker daemons against the
shared DB_PATH with the PRE-DEPLOY code (which lacks the new job name and would
claim+fail our jobs). So we repoint media_bus at a PRIVATE scratch DB before any
DB use and process jobs only with our own work_once() — fully deterministic, no
race with the live daemons.

Covers: (1) video WITH audio -> audio_extract -> one audio MediaRef on disk;
(2) temporal crop of that extracted audio (reuses the existing crop job);
(3) a SILENT video -> audio_extract -> failed with code `no_audio_track`.
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
from abstract_hugpy_dev.video_intel.audio_schema import make_audio_extract
from abstract_hugpy_dev.video_intel.crop_schema import TemporalRegion, make_crop
from abstract_hugpy_dev.video_intel import media_bus


SCRATCH = os.path.join(DEFAULT_ROOT, "video_intel", "_scratch")
FFMPEG = resolve_bin("ffmpeg") or "ffmpeg"

# --- isolate the bus from the live daemons (see module docstring) ---
os.makedirs(SCRATCH, exist_ok=True)
media_bus.DB_PATH = os.path.join(SCRATCH, "selftest_audio_jobs.db")
media_bus._initialized = False


def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"asset gen failed: {' '.join(cmd)}\n{r.stderr}")


def _gen_video_with_audio():
    vid = os.path.join(SCRATCH, "audio_test_withaudio.mp4")
    # 3s @ 25fps, 320x240 video + a 440Hz sine audio track
    _run([FFMPEG, "-y",
          "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
          "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
          vid])
    return vid


def _gen_video_silent():
    vid = os.path.join(SCRATCH, "audio_test_silent.mp4")
    # video-only (no sine input) -> a silent video with no audio stream
    _run([FFMPEG, "-y",
          "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          vid])
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

    vid = _gen_video_with_audio()
    vid_ref = ingest(vid)
    print("VIDEO MediaRef:", json.dumps(asdict(vid_ref)))
    assert vid_ref.kind == "video", f"video kind={vid_ref.kind}"
    assert not (vid_ref.sample_rate is None and vid_ref.channels is None), \
        f"expected an audio track; sample_rate={vid_ref.sample_rate} channels={vid_ref.channels}"
    print("source video has audio -> sample_rate=%s channels=%s"
          % (vid_ref.sample_rate, vid_ref.channels))

    # ---------------- Phase 5: audio_extract (one audio output) ----------------
    print("\n--- PHASE 5: audio_extract (fmt=wav) ---")
    spec = make_audio_extract(vid_ref, fmt="wav")
    jid = media_bus.enqueue("audio_extract", spec)
    got = _drain(jid, 120)
    print("status:", got["status"])
    print("result:", json.dumps(got["result"]))
    assert got["status"] == "done", f"status={got['status']} result={got['result']}"
    assert got["result"]["ok"] is True
    outs = got["result"]["outputs"]
    assert len(outs) == 1, f"expected exactly 1 audio output, got {len(outs)}"
    audio_out = outs[0]
    assert audio_out["kind"] == "audio", f"output kind={audio_out['kind']}"
    assert os.path.isfile(audio_out["uri"]), f"missing audio file {audio_out['uri']}"
    print("audio_extract OK ->", audio_out["uri"],
          f"(sr={audio_out['sample_rate']} ch={audio_out['channels']} "
          f"dur={audio_out['duration_s']})")

    # ---------------- Phase 5: temporal crop of the extracted audio ----------------
    print("\n--- PHASE 5: temporal crop of the extracted audio [0.5, 1.5) ---")
    audio_ref = media_store_ref(audio_out)
    cspec = make_crop(source=audio_ref, temporal=TemporalRegion(0.5, 1.5))
    jid_c = media_bus.enqueue("crop", cspec)
    got_c = _drain(jid_c, 120)
    print("status:", got_c["status"])
    print("result:", json.dumps(got_c["result"]))
    assert got_c["status"] == "done", f"status={got_c['status']} result={got_c['result']}"
    assert got_c["result"]["ok"] is True
    couts = got_c["result"]["outputs"]
    assert len(couts) == 1, f"expected exactly 1 cropped audio output, got {len(couts)}"
    crop_out = couts[0]
    assert crop_out["kind"] == "audio", f"cropped kind={crop_out['kind']}"
    assert os.path.isfile(crop_out["uri"]), f"missing cropped file {crop_out['uri']}"
    print("audio crop OK ->", crop_out["uri"], f"(dur={crop_out['duration_s']})")

    # ---------------- Phase 5: no_audio_track path (silent video) ----------------
    print("\n--- PHASE 5: audio_extract on a SILENT video -> no_audio_track ---")
    svid = _gen_video_silent()
    svid_ref = ingest(svid)
    print("SILENT VIDEO MediaRef:", json.dumps(asdict(svid_ref)))
    assert svid_ref.kind == "video", f"silent video kind={svid_ref.kind}"
    assert svid_ref.sample_rate is None and svid_ref.channels is None, \
        f"expected NO audio track; sample_rate={svid_ref.sample_rate} channels={svid_ref.channels}"
    sspec = make_audio_extract(svid_ref, fmt="wav")
    jid_s = media_bus.enqueue("audio_extract", sspec)
    got_s = _drain(jid_s, 60)
    print("status:", got_s["status"])
    print("result:", json.dumps(got_s["result"]))
    assert got_s["status"] == "failed", f"status={got_s['status']}"
    assert got_s["result"]["ok"] is False
    assert got_s["result"]["error"]["code"] == "no_audio_track", \
        f"code={got_s['result']['error']['code']}"
    print("no_audio_track OK (refused a silent video, clean JobError data)")

    print("\n" + "=" * 70)
    print("ALL PASS (audio_extract + audio temporal crop + no_audio_track)")
    return 0


def media_store_ref(out_dict):
    """Re-hydrate a MediaRef from a JobResult output dict through the validating
    factory (the bus round-trips specs the same way)."""
    from abstract_hugpy_dev.video_intel.media_schema import make_media_ref
    return make_media_ref(**out_dict)


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - self-test reporter
        import traceback
        traceback.print_exc()
        print(f"FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)
