#!/usr/bin/env python
"""Phase 3a headless verification for the Video Intelligence HTTP surface.

Builds a MINIMAL standalone Flask app registering ONLY video_bp (does NOT boot
the full wsgi_app / live stack) and drives the full crop loop with Flask's
test_client. Run with an ISOLATED throwaway DEFAULT_ROOT so nothing is written
into /mnt/llm_storage:

  lxc exec hugpy -- env DEFAULT_ROOT=/tmp/vi_w2_test \
      PYTHONPATH=/home/ubuntu/station/dev/abstract_hugpy_dev/src \
      /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
      /home/ubuntu/station/dev/abstract_hugpy_dev/_vi_phase3a_verify.py
"""
import os
import subprocess
import sys

# DEFAULT_ROOT must be set in the env BEFORE importing the backbone (media_bus
# computes DB_PATH from the constants module at import time).
DEFAULT_ROOT = os.environ.get("DEFAULT_ROOT", "/tmp/vi_w2_test")
UPLOADS_DIR = os.path.join(DEFAULT_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

from flask import Flask

from abstract_hugpy_dev.flask_app.app.routes.video_routes import video_bp
from abstract_hugpy_dev.video_intel import media_bus

_failures = []


def check(cond, label):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        _failures.append(label)


def main():
    print(f"DEFAULT_ROOT = {DEFAULT_ROOT}")
    print(f"media_bus.DB_PATH = {media_bus.DB_PATH}")

    # ---- minimal standalone app: ONLY video_bp -----------------------------
    app = Flask("vi_phase3a_verify")
    app.register_blueprint(video_bp)
    client = app.test_client()

    # ---- 1) generate a known 640x480 test image ----------------------------
    img_path = os.path.join(UPLOADS_DIR, "t.png")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=1",
         "-frames:v", "1", img_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    print(f"\n[step 1] generated test image: {img_path} "
          f"({os.path.getsize(img_path)} bytes)")

    # ---- 2) POST /video/ingest --------------------------------------------
    r = client.post("/video/ingest", json={"path": img_path})
    print(f"\n[step 2] POST /video/ingest -> {r.status_code}")
    ref = r.get_json()
    print(f"         MediaRef = {ref}")
    check(r.status_code == 200, "ingest returns 200")
    check(isinstance(ref, dict) and ref.get("width") == 640, "MediaRef width == 640")
    check(isinstance(ref, dict) and ref.get("height") == 480, "MediaRef height == 480")
    check(isinstance(ref, dict) and ref.get("kind") == "image", "MediaRef kind == image")

    # ---- 3) POST /video/jobs/crop -----------------------------------------
    r = client.post("/video/jobs/crop", json={
        "source": ref,
        "spatial": {"x": 100, "y": 50, "w": 200, "h": 150},
    })
    print(f"\n[step 3] POST /video/jobs/crop -> {r.status_code}")
    body = r.get_json()
    print(f"         body = {body}")
    check(r.status_code == 200, "crop enqueue returns 200")
    job_id = (body or {}).get("job_id")
    check(bool(job_id), "crop returns a job_id")

    # ---- 4) advance the job (daemon isn't running under test_client) -------
    processed = media_bus.work_once()
    print(f"\n[step 4] media_bus.work_once() processed job_id = {processed}")
    r = client.get(f"/video/jobs/{job_id}")
    view = r.get_json()
    print(f"         GET /video/jobs/{job_id} -> {r.status_code}")
    print(f"         view = {view}")
    check(r.status_code == 200, "job status returns 200")
    check((view or {}).get("status") == "done", "job status == done")
    result = (view or {}).get("result") or {}
    check(result.get("ok") is True, "result.ok is True")
    outputs = result.get("outputs") or []
    out0 = outputs[0] if outputs else {}
    check(out0.get("width") == 200, "outputs[0].width == 200")
    check(out0.get("height") == 150, "outputs[0].height == 150")
    out_uri = out0.get("uri")
    print(f"         crop output uri = {out_uri}")

    # ---- 5) GET /video/media?handle=<crop uri> ----------------------------
    r = client.get("/video/media", query_string={"handle": out_uri})
    ctype = r.headers.get("Content-Type", "")
    print(f"\n[step 5] GET /video/media -> {r.status_code}  Content-Type={ctype!r}"
          f"  body_len={len(r.data)}")
    check(r.status_code == 200, "media serve returns 200")
    check(ctype.startswith("image/png"), "media Content-Type is image/png")
    check(len(r.data) > 0, "media body is non-empty")

    # ---- 6) negative: image source + temporal region -> 400 ---------------
    r = client.post("/video/jobs/crop", json={
        "source": ref,
        "temporal": {"start_s": 0.0, "end_s": 1.0},
    })
    body = r.get_json()
    print(f"\n[step 6] POST /video/jobs/crop (image + temporal) -> {r.status_code}")
    print(f"         body = {body}")
    check(r.status_code == 400, "image+temporal crop returns 400")
    check(bool((body or {}).get("error")), "400 carries an error message")

    print()
    if _failures:
        print(f"FAIL ({len(_failures)} check(s) failed): {_failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
