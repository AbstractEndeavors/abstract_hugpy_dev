---
name: vi-backend-deploy-gates
description: video_intel backend job deploy/verify — VM selftest pre-gate, one batched restart, live e2e, and the DEFAULT_ROOT-isolation + remote-GPU-proof gotchas
metadata:
  type: reference
---

Deploy/verify recipe for a NEW `video_intel` backend job (rails: frozen Spec +
pure runner + JOB_REGISTRY/DISPATCH/SPEC_DESERIALIZERS + a `POST /video/jobs/<name>`
route). Backend package: `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/`.
Route file: `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/flask_app/app/routes/video_routes.py`
(auto-mounted at BOTH `/video/...` and, via wsgi_app.py, `/api/video/...`).

**Pre-gate in the VM as ubuntu (the share is mounted at
`/home/ubuntu/station/dev/abstract_hugpy_dev` inside the `hugpy` VM):**
`lxc exec hugpy -- sudo -u ubuntu bash -lc 'cd /home/ubuntu/station/dev/abstract_hugpy_dev && PYTHONPATH=src ./venv/bin/python src/abstract_hugpy_dev/video_intel/_selftest_<x>.py'`
Run the base `_selftest.py` + `_selftest_gen.py` (regression) + the new job's selftest.
NOTE (2026-07-03): on the current share, only `_selftest_scene.py` actually EXISTS under
`video_intel/` — the historical `_selftest.py` / `_selftest_gen.py` referenced here are NOT
present. When they're absent, substitute regression coverage with a clean combined
`import abstract_hugpy_dev.managers` + `video_intel` import, a v1 spec round-trip, and
old-payload tolerance. Always `find video_intel -name '_selftest*.py'` and run what exists.
`sudo -u ubuntu` from the host as `solcatcher` works.

**GOTCHA — the `DEFAULT_ROOT=/tmp/... ` isolation prefix in the pre-gate command is
SILENTLY IGNORED.** `_platform.env_value()` reads the project `.env` FIRST (which pins
`DEFAULT_ROOT=/mnt/llm_storage`) and only falls back to process env, so every selftest
runs under `/mnt/llm_storage` regardless of the exported var. Isolation from the LIVE
daemon (which would steal claims) is instead guaranteed at the DB layer: the gen/scene
selftests repoint `media_bus.DB_PATH` to a private `selftest_*_jobs.db` + set
`media_bus._initialized=False` and drain via their own `work_once()`. So: don't rely on
the `/tmp` DEFAULT_ROOT for isolation — rely on the DB repoint being present in the
selftest. (The user's written deploy protocol assumes `/tmp` isolation; it's the DB
repoint that actually protects you.)

**Selftest shape that works headless:** Part A = rails smoke (enqueue → work_once →
assert `done` with the right output shape, but TOLERATE a clean JobError dict
`{code,message,retryable}` because headless has no live worker / no local model). Part B
= exercise any ffmpeg/CPU step directly (e.g. synth 2 tiny PNGs via ffmpeg lavfi, call
the assembly helper) so you get REAL coverage of the non-GPU path independent of the
fleet. Factor such steps into a module-level helper so the selftest can call them.
For a GPU-model step you can still prove on CPU: guard it (`HUGPY_VIDEOGEN_LOCAL=always`,
tiny dims/steps) and SKIP-with-loud-note if the model weights aren't loadable in the VM —
see [[vi-img2img-fleet-release]] (the sd-turbo img2img CPU proof RAN green in the VM).

**Restart + regression:** `lxc exec hugpy -- systemctl is-active hugpy-api-dev`
(active) → ONE `systemctl restart hugpy-api-dev` (batched, only after pre-gate passes)
→ is-active again → curl `/api/version` `/media/` `/video/` `/` all 200 against
`https://dev.hugpy.ai`.

**Live e2e:** extend the host harness `/tmp/vi_e2e.py` (read it first to reuse its base
URL + POST/poll helpers + assertion style), add a gate that POSTs the new job and polls
`GET /video/jobs/<id>` to a terminal state, asserting the output shape + that each output
is servable via `GET /video/media?handle=<uri>` (200, non-trivial body). Then run the
FULL e2e (all gates) — no regressions. Model `sd-turbo` is the fast text-to-image model
on worker `op` (192.168.1.113, RTX 3090).

**REMOTE-GPU PROOF (how to prove generation didn't fall back to local CPU):** the runner
guard refuses local when a fleet exists but no live worker, UNLESS `HUGPY_VIDEOGEN_LOCAL`
is set. The live daemon's Environment has it UNSET → a `done` result with real image
bytes ⟹ a live worker served the model remotely (else the job would've failed
`no_live_gpu_worker`). Corroborate via the daemon journal:
`lxc exec hugpy -- journalctl -u hugpy-api-dev --since ...` shows
`POST http://192.168.1.113:9100/infer "HTTP/1.1 200 OK"` — one per generated frame.
(Note `/api/workers` `/api/fleet` return the SPA catch-all HTML, not JSON — use the
journal for fleet state, not those endpoints.)

See [[vi-ui-frontend-deploy]] for the frontend rsync/archive half and
[[vi-ui-workbench-decomposition]] for the full-stack decomposition shape.
