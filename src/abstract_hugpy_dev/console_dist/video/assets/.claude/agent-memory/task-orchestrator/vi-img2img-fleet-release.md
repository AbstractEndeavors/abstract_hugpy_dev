---
name: vi-img2img-fleet-release
description: The pinned-wheel fleet-release STOP boundary + hold-the-advertisement-flip pattern for hugpy inference-plane task additions that need the GPU worker
metadata:
  type: reference
---

Adding a NEW inference TASK (e.g. `image-to-image`) that a GPU worker must execute hits a
split-brain deploy: **central runs from the share (edits go live on restart, no release),
but the GPU worker `op` runs a PINNED PyPI wheel** (`abstract_hugpy_dev==0.1.92` as of
2026-07-03, actively bumped by a concurrent ServeMode release workstream). The worker
re-runs `execute_prompt(**body)` on ITS OWN installed engine, so new task code on central
does NOTHING on the worker until the wheel is updated — which is a **fleet-wide PyPI
release** (edit share → publish → `POST /ops/update {"version"}` through central, or
heartbeat auto-converge). That release is the mission's declared STOP boundary: do NOT
improvise it, especially while another terminal is mid-release on the same package.

**The safe posture that lets you ship everything EXCEPT the release:**
- Deploy the additive engine code + video wiring + UI to live central/frontend (all
  regression-safe). Register the runner/builder/registry rows — they are INERT until a
  model advertises the task.
- **HOLD the "advertisement flip"** — the one line that adds the task to a curated model's
  `tasks` list (`imports/config/models/models_config.py`, e.g. `sd-turbo → tasks`). Leave
  it as a documented one-liner. WHY: the moment central advertises the task, it will route
  those requests to a live worker that has the MODEL loaded (`op` has `sd-turbo`) — but the
  worker's OLD wheel can't serve the new task → ugly misroute-then-fail. The worker
  advertises MODELS, not tasks; central maps model→tasks from ITS config, so central
  cannot tell "new wheel" from "old wheel." Holding the flip keeps routing honest.
- **Make the runner's availability probe key on real servability, not registry presence.**
  `video_intel/runners/_img2img.py::img2img_available(model_id)` = pair registered AND
  `resolve` succeeds for `(model_id, task)` (i.e. the model actually advertises it). With
  the flip held, this returns FALSE on live central → the runner returns a retryable
  `JobError{code:"image_to_image_unavailable"}` (honest, not silent fallback). This is the
  mission-preferred behavior AND it's what makes "ship dark, fail honestly" work.
- **Prove the generation LOCALLY** to satisfy the fallback: the scene selftest flips the
  advertisement IN-PROCESS (like it repoints `media_bus.DB_PATH`) + sets
  `HUGPY_VIDEOGEN_LOCAL=always` + synths a tiny init PNG + runs a bounded CPU sd-turbo
  img2img (256×256, steps 2). This RAN green (real bytes) — proves the full
  `execute_prompt(task="image-to-image")` path without touching the worker or the live flip.
- **Live e2e asserts the HONEST error**, not a `done` (since the fleet can't serve it yet):
  GATE 9 posts a start-frame scene + chain=true and asserts the terminal
  `image_to_image_unavailable`. v1 (no start frame) still `done` = remote-GPU regression proof.

**RUNNER_PAIRS static-mirror landmine (mandatory even when the flip is held):**
`imports/src/constants/categories.py` `RUNNER_PAIRS` is a STATIC mirror of
`FRAMEWORK_RUNNERS` (does NOT auto-derive). `derive_model_config_row` drops any model
(staples included) whose advertised task isn't in `RUNNER_PAIRS` and isn't on disk — so the
new `(framework,task)` pair MUST be added there BEFORE the flip, or flipping later silently
drops the whole staple row (and the default that rides it). Add the pair; keep it.

**HF_TASK_TO_TASKS is a DIFFERENT knob — usually leave the new task OUT.** Adding the task
to `HF_TASK_TO_TASKS` makes model DISCOVERY classify externally-downloaded HF models with
that `pipeline_tag` and — because the pair now has a runner — advertise them as SERVABLE.
On this fleet that would auto-advertise unvetted flux / Qwen-Image-Edit img2img models
(exactly the "don't force flux" line). Omit it; scope the task to the ONE curated model.

**Verification lesson (burned once here):** a subagent's ATTRIBUTION of an anomaly can be
wrong. The backend verify agent blamed my registry rows for a multi-GB `Qwen-Image-Edit`
prefetch; a focused diagnostic agent proved it EXTERNAL (another terminal's explicit
`/llm/repos/download` campaign — download routing keys on the request `task`, not
`HF_TASK_TO_TASKS`; a TTS download predated my edit by 8h). Always diagnose an anomaly to
root cause (find the trigger file:line or the external process id) before "fixing" it.

**op access changed (2026-07-03):** reachable by SSH PUBKEY (`solcatcher` `id_rsa`
authorized), NOT the recorded fleet password (rotated/stale). Rollback baseline for the
future release lives at `video_intel/_op_pip_snapshot_2026-07-03.txt` on the share.

See [[vi-backend-deploy-gates]] for the base deploy protocol and
[[vi-ui-workbench-decomposition]] for the full-stack decomposition this extends.
