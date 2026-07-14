"""Identity 3D MESH (+ turntable) bus runner — a RELAY to a remote GPU render service.

Central has NO GPU: it must never import torch / hy3dgen / a diffusion pipeline for a
mesh build. This runner is a thin HTTP CLIENT to the ``IDENTITY_RENDER_URL`` service
(a sibling agent builds that service to the FIXED contract mirrored below). It:

  1. Reads the identity's cardinal reference images (jailed by the route to the
     profile's OWN refs) and base64-encodes them.
  2. POSTs a job to ``<IDENTITY_RENDER_URL>/jobs`` (kind ``mesh_and_turntable`` when the
     spec chains a turntable, else ``mesh_build``), authenticating with the
     ``X-Identity-Render-Token`` header.
  3. POLLs ``GET /jobs/<id>`` every ~5s until done / error / a generous deadline, honoring
     a cooperative cancel (relayed as a best-effort ``DELETE /jobs/<id>``).
  4. On done, DOWNLOADS every produced file and PERSISTS it under the identity's own dir
     (``<IDENTITIES_HOME>/<slug>/mesh/<recon_id>/``) via ``identity_profiles`` path
     helpers — the GLB + mesh json at the mesh root, the turntable frames + mp4 under
     ``turntable/``.
  5. ATTACHES the turntable frames as a ``reconstructions`` entry (reusing
     ``attach_reconstruction`` with ``replace=True`` so the EXISTING promote route works
     on the turntable output) and RECORDS mesh state (``set_mesh_state``) so the GET
     mesh-status route surfaces status + GLB + mp4.

Pure ``(IdentityMeshSpec, job_id) -> JobResult`` (map §6): EVERY expected failure — an
unconfigured service, an unreachable host, a 401, a render error, a timeout — is DATA
(``JobResult(ok=False, JobError(...))``), never a raise. Only a genuine programmer error
raises, and ``media_bus.run_claimed`` is the one place that catches that.

Heavy imports (requests, identity_profiles, media_schema, media_bus) are LAZY — done
inside the runner — so importing this module (which ``runners/__init__`` does at boot)
stays cheap and can never break app boot. No pathlib anywhere; os.path only.

Remote service contract (FIXED — the service is built to exactly this):
  * ``GET  /health``                     -> 200 {ok, service, capabilities}
  * ``POST /jobs``                       -> 202 {job_id}
  * ``GET  /jobs/<id>``                  -> {job_id, status, error?, files?}
  * ``GET  /jobs/<id>/files/<path>``     -> raw bytes
  * ``DELETE /jobs/<id>``                -> best-effort cleanup
Auth: header ``X-Identity-Render-Token: <IDENTITY_RENDER_TOKEN>`` (missing/wrong -> 401).

FLEET-VLM FRONT AUTO-SELECTION (keeper 2026-07-14): the route only ever defaults
``front`` to the profile's FIRST source reference image — if that photo is a cropped
waist-up portrait, the mesh comes out with cut-off legs (happened live: luigi ref_00).
When the route did NOT see an explicit ``views.front`` override, it hands this runner
``spec.view_candidates`` (every existing source ref, in order) instead of guessing.
HERE — inside the bus job, never the HTTP handler, because each vision call is
~5-10s and there can be up to 12 candidates — ``_select_front_view`` asks hugpy's own
``/ml/vision`` amenity, in order, "does this show the character's full body?" and the
first "yes" becomes front. Any failure (network, non-200, ambiguous text, the
``IDENTITY_FRONT_AUTOSELECT`` kill-switch) is a soft skip — the default front from the
route is always a safe fallback, so selection NEVER fails the job.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import secrets
import time

from ..result_schema import JobError, JobResult

logger = logging.getLogger(__name__)

# Per-request HTTP budget as a (connect, read) TUPLE: the read side stays generous (a
# POST carries the b64 reference images — a few MB) but the CONNECT side is short, so a
# down/firewalled service is detected in seconds, not a 120s hang per attempt (ae's
# firewall DROPs unknown ports — no RST — so a flat timeout eats the whole budget just
# discovering "nobody home"; keeper 2026-07-14).
_HTTP_TIMEOUT_S = (10.0, 120.0)
# Poll cadence + whole-job deadline. The bus registers identity_mesh_build with a
# 14400s timeout; poll a touch under it so the runner returns clean errors-as-data
# rather than being killed mid-poll. Overridable via env for tests / tuning.
_POLL_INTERVAL_S = float(os.getenv("IDENTITY_RENDER_POLL_INTERVAL_S", "5") or "5")
_POLL_DEADLINE_S = float(os.getenv("IDENTITY_RENDER_DEADLINE_S", "14100") or "14100")

# ---- fleet-VLM front auto-selection (module docstring) ------------------------- #
# Per-candidate HTTP budget as a (connect, read) tuple — a vision call is ~5-10s, so
# the read side stays generous; the connect side stays short (localhost — a down
# amenity fails fast, not a hang). Overridable via env for tests / tuning.
_VISION_TIMEOUT_S = (
    float(os.getenv("IDENTITY_FRONT_VISION_CONNECT_TIMEOUT_S", "10") or "10"),
    float(os.getenv("IDENTITY_FRONT_VISION_READ_TIMEOUT_S", "60") or "60"),
)
_VISION_MAX_CANDIDATES = 12
_VISION_PROMPT = (
    "Look at the character in this image. Does the image show the character's ENTIRE "
    "body from head to feet, with the feet fully visible (not cropped by the image "
    "edge)? Answer with exactly one word: yes or no."
)


def _autoselect_enabled() -> bool:
    """The ``IDENTITY_FRONT_AUTOSELECT`` kill-switch — default ON; "off"/"0"/"false"
    (any case) disables fleet-VLM front selection entirely."""
    v = (os.getenv("IDENTITY_FRONT_AUTOSELECT", "") or "").strip().lower()
    return v not in ("off", "0", "false")


def _reply_says_yes(text) -> bool:
    """A clear, unambiguous "yes" as the FIRST word of the vision reply (tolerating
    leading punctuation/markdown like ``**Yes.**`` or ``"Yes,"``) — a longer hedge that
    merely mentions "yes" later ("no, but yes the head is visible...") does not count."""
    if not isinstance(text, str) or not text.strip():
        return False
    cleaned = re.sub(r"^[^a-zA-Z]+", "", text.strip())
    if not cleaned:
        return False
    first_word = cleaned.split(None, 1)[0].rstrip(".,!:;\"'*_")
    return first_word.lower() == "yes"


def _select_front_view(candidates, slug: str, requests_mod):
    """Ask hugpy's own fleet vision amenity (``POST /ml/vision``), IN ORDER (capped at
    ``_VISION_MAX_CANDIDATES``), which candidate reference image shows the character's
    FULL BODY. Returns ``(chosen_path_or_None, checked_count)`` — the first candidate
    with a clear "yes" wins. EVERY failure (unreadable file, unreachable amenity, a
    non-200, a non-JSON/not-ok reply, an ambiguous answer) is a soft skip logged at
    info: this never raises, so a flaky vision call can never fail the mesh job."""
    base = (os.getenv("HUGPY_CENTRAL_URL", "") or "http://127.0.0.1:7002").strip().rstrip("/")
    url = f"{base}/ml/vision"
    checked = 0
    for path in candidates[:_VISION_MAX_CANDIDATES]:
        checked += 1
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            logger.info("identity mesh front-select: candidate unreadable (%s)", path)
            continue
        b64 = base64.b64encode(raw).decode("ascii")
        try:
            resp = requests_mod.post(url, json={"image_b64": b64, "prompt": _VISION_PROMPT},
                                     timeout=_VISION_TIMEOUT_S)
        except requests_mod.RequestException as exc:
            logger.info("identity mesh front-select: /ml/vision unreachable for %s (%s)",
                        path, exc)
            continue
        if resp.status_code != 200:
            logger.info("identity mesh front-select: /ml/vision HTTP %s for %s",
                        resp.status_code, path)
            continue
        try:
            body = resp.json()
        except ValueError:
            logger.info("identity mesh front-select: non-JSON /ml/vision reply for %s", path)
            continue
        if not isinstance(body, dict) or body.get("ok") is False:
            logger.info("identity mesh front-select: /ml/vision reported not-ok for %s", path)
            continue
        if _reply_says_yes(body.get("text")):
            logger.info("identity mesh front-select: %s qualifies as full-body (profile %s)",
                        path, slug)
            return path, checked
    return None, checked


def _dest_for(mesh_dir: str, turntable_dir: str, fpath: str) -> str:
    """Map a service file name to its durable destination under the identity dir.

    ``identity.glb`` / ``*.glb`` and ``*.json`` land at the mesh root; ``*.mp4`` and any
    ``frames/…`` land under ``turntable/`` (frames keep the ``frames/`` subdir). Every
    component is reduced to ``os.path.basename`` so a hostile ``../`` in a service-supplied
    name can never escape the mesh dir (defense-in-depth even though the service is trusted)."""
    norm = (fpath or "").replace("\\", "/").lstrip("/")
    base = os.path.basename(norm)
    low = norm.lower()
    if low.endswith(".glb"):
        return os.path.join(mesh_dir, base)
    if low.endswith(".json"):
        return os.path.join(mesh_dir, base)
    if norm.startswith("frames/") or (low.endswith(".png") and "frame" in low):
        return os.path.join(turntable_dir, "frames", base)
    # mp4 (turntable video) and anything else -> the turntable bucket.
    return os.path.join(turntable_dir, base)


def _atomic_write_bytes(dest: str, data: bytes) -> None:
    """Write *data* to *dest* atomically (unique temp in the dest dir + os.replace),
    mirroring identity_profiles' copy idiom so a crashed download never leaves a
    half-written artifact at the final name."""
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def run_identity_mesh_build(spec, job_id: str) -> JobResult:
    """Relay ``spec`` to the remote render service and persist the result. Returns
    ``JobResult(ok=True, outputs=(turntable mp4,))`` on success (the GLB is recorded in
    the profile's mesh state, which the GET mesh route reads — a GLB is not a MediaRef
    kind). Every expected failure is a clean error-as-data."""
    # Lazy imports keep this module boot-cheap (runners/__init__ imports it at boot).
    from .. import identity_profiles
    from ..media_bus import is_cancelling
    from ..media_schema import make_media_ref

    slug = spec.slug
    recon_id = spec.recon_id

    def _set_state(patch: dict) -> None:
        try:
            identity_profiles.set_mesh_state(slug, recon_id, patch)
        except Exception:  # noqa: BLE001 — mesh-state is a best-effort mirror; never fail the job on it
            logger.debug("set_mesh_state failed for %s/%s", slug, recon_id, exc_info=True)

    def _fail(code: str, message: str, retryable: bool) -> JobResult:
        _set_state({"status": "error", "error": message})
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code=code, message=message, retryable=retryable))

    url = (os.getenv("IDENTITY_RENDER_URL", "") or "").strip().rstrip("/")
    token = (os.getenv("IDENTITY_RENDER_TOKEN", "") or "").strip()
    if not url or not token:
        return _fail(
            "not_configured",
            "the identity 3D render service is not configured on this host — set "
            "IDENTITY_RENDER_URL and IDENTITY_RENDER_TOKEN (central has no GPU; mesh "
            "builds are relayed to a remote GPU render service).",
            retryable=False)

    import requests  # lazy — present (2.34.2); keeps the module boot-cheap

    headers = {"X-Identity-Render-Token": token}

    # ---- FLEET-VLM FRONT AUTO-SELECTION (module docstring) --------------------------- #
    # The route only hands >=2 view_candidates when the caller did NOT explicitly assign
    # a front — an explicit assignment is an operator override that auto-selection never
    # second-guesses. Every branch records an honest ``front_selection`` outcome; a VLM
    # miss/error NEVER fails the job — it just keeps the route's default front.
    view_sources = spec.view_sources or ()
    candidates = tuple(getattr(spec, "view_candidates", ()) or ())
    default_front = next((p for n, p in view_sources if n == "front"), None)
    front_selection = {"mode": "explicit", "chosen": default_front, "checked": 0,
                       "full_body": None}
    if not candidates:
        front_selection["mode"] = "explicit"
    elif len(candidates) < 2:
        # Only one usable reference on the whole profile — nothing to choose between.
        front_selection["mode"] = "default"
    elif not _autoselect_enabled():
        front_selection["mode"] = "disabled"
        logger.info("identity mesh front-select: IDENTITY_FRONT_AUTOSELECT disabled for %s", slug)
    else:
        chosen, checked = _select_front_view(candidates, slug, requests)
        front_selection["mode"] = "vlm"
        front_selection["checked"] = checked
        if chosen is not None:
            front_selection["chosen"] = chosen
            front_selection["full_body"] = True
            view_sources = tuple(
                (n, chosen) if n == "front" else (n, p) for n, p in view_sources)
        else:
            front_selection["full_body"] = False
            logger.info("identity mesh front-select: no full-body candidate found for %s "
                        "(checked %d of %d) — keeping default front",
                        slug, checked, len(candidates))

    # ---- read + base64 the assigned view images (jailed to the profile's own refs) ----
    views_b64: dict[str, str] = {}
    for pair in (view_sources or ()):
        try:
            name, path = pair[0], pair[1]
        except (IndexError, TypeError):
            continue
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            logger.warning("identity mesh: view %r image unreadable (%s)", name, path)
            continue
        views_b64[name] = base64.b64encode(raw).decode("ascii")
    if "front" not in views_b64:
        return _fail(
            "bad_views",
            f"no readable 'front' reference image for the mesh build of profile {slug!r}",
            retryable=False)

    kind = "mesh_and_turntable" if spec.chain_turntable else "mesh_build"
    payload = {
        "kind": kind,
        "identity_id": slug,
        "views": views_b64,
        "mesh_params": {
            "seed": spec.seed,
            "num_inference_steps": spec.num_inference_steps,
            "octree_resolution": spec.octree_resolution,
            "texture": bool(spec.texture),
        },
        "turntable_params": {
            "frame_count": spec.frame_count,
            "fps": spec.fps,
            "width": spec.width,
            "height": spec.height,
            "elevation_deg": spec.elevation_deg,
            "transparent": bool(spec.transparent),
        },
    }

    _set_state({"status": "running", "error": None, "job_id": job_id,
               "front_selection": front_selection})

    # ---- POST the job ----
    try:
        resp = requests.post(f"{url}/jobs", json=payload, headers=headers,
                             timeout=_HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        return _fail("render_unreachable",
                     f"could not reach the identity render service at {url}: {exc}",
                     retryable=True)
    if resp.status_code == 401:
        return _fail("render_unauthorized",
                     "the identity render service rejected the token (HTTP 401)",
                     retryable=False)
    if resp.status_code != 202:
        body = (resp.text or "")[:300]
        return _fail("render_rejected",
                     f"the render service rejected the job (HTTP {resp.status_code}): {body}",
                     retryable=True)
    try:
        remote_id = resp.json()["job_id"]
    except (ValueError, KeyError, TypeError):
        return _fail("render_bad_response",
                     "the render service accepted the job but returned no job_id",
                     retryable=True)
    if not isinstance(remote_id, str) or not remote_id:
        return _fail("render_bad_response",
                     "the render service returned an empty job_id", retryable=True)

    def _delete_remote() -> None:
        try:
            requests.delete(f"{url}/jobs/{remote_id}", headers=headers, timeout=30.0)
        except requests.RequestException:
            pass  # best-effort cleanup; never fail the job on it

    # ---- poll until done / error / cancel / deadline ----
    deadline = time.time() + _POLL_DEADLINE_S
    files: list = []
    while True:
        if is_cancelling(job_id):
            _delete_remote()
            _set_state({"status": "cancelled", "error": None})
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="cancelled",
                message=f"identity mesh build for profile {slug!r} cancelled by user",
                retryable=False))
        if time.time() > deadline:
            _delete_remote()
            return _fail("render_timeout",
                         f"identity mesh render for profile {slug!r} did not finish within "
                         f"the deadline ({int(_POLL_DEADLINE_S)}s)",
                         retryable=True)
        try:
            pr = requests.get(f"{url}/jobs/{remote_id}", headers=headers,
                              timeout=_HTTP_TIMEOUT_S)
            if pr.status_code == 200:
                pbody = pr.json()
                status = pbody.get("status")
                if status == "done":
                    files = pbody.get("files") or []
                    break
                if status == "error":
                    _delete_remote()
                    msg = pbody.get("error") or "the render service reported an error"
                    return _fail("render_failed",
                                 f"identity mesh render failed for profile {slug!r}: {msg}",
                                 retryable=True)
                # queued / running -> keep polling
        except requests.RequestException:
            pass  # transient poll hiccup — keep polling until the deadline
        except ValueError:
            pass  # non-JSON status body — treat as transient
        time.sleep(_POLL_INTERVAL_S)

    # ---- download every produced file + persist under the identity dir ----
    mesh_dir = os.path.join(identity_profiles._identity_dir(slug), "mesh", recon_id)
    turntable_dir = os.path.join(mesh_dir, "turntable")
    glb_path = mesh_json_path = video_path = None
    frame_paths: list[str] = []
    for f in files:
        if not isinstance(f, str) or not f.strip():
            continue
        try:
            fr = requests.get(f"{url}/jobs/{remote_id}/files/{f}", headers=headers,
                              timeout=_HTTP_TIMEOUT_S)
        except requests.RequestException:
            logger.warning("identity mesh: failed to download %r", f)
            continue
        if fr.status_code != 200:
            logger.warning("identity mesh: file %r -> HTTP %s", f, fr.status_code)
            continue
        dest = _dest_for(mesh_dir, turntable_dir, f)
        try:
            _atomic_write_bytes(dest, fr.content)
        except OSError:
            logger.warning("identity mesh: could not persist %r -> %s", f, dest)
            continue
        low = dest.lower()
        if low.endswith(".glb"):
            glb_path = dest
        elif low.endswith(".json"):
            mesh_json_path = dest
        elif low.endswith(".mp4"):
            video_path = dest
        elif low.endswith(".png"):
            frame_paths.append(dest)

    if glb_path is None:
        return _fail("render_no_glb",
                     f"the render service reported done for profile {slug!r} but produced "
                     "no .glb mesh file",
                     retryable=True)

    # ---- attach the turntable frames as a reconstruction entry (promotable) ----
    # Reuse attach_reconstruction with replace=True so the promote route — which finds a
    # record by recon_id — sees THESE frames (and the list never grows a duplicate id).
    # The frames carry angular order via degrees_per_frame + their sorted order so the UI
    # can scrub. attach failure never fails the job; mesh state is the durable record.
    attached_rec = None
    if frame_paths:
        frame_paths.sort()  # frame_0000.png, frame_0001.png … == angular order
        n = len(frame_paths)
        try:
            attached_rec = identity_profiles.attach_reconstruction(
                slug, recon_id, frame_paths,
                spec={"job_id": job_id, "mode": "turntable", "frame_count": n,
                      "degrees_per_frame": (round(360.0 / n, 2) if n else None),
                      "source": "identity_render_relay", "glb_path": glb_path},
                replace=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning("identity mesh: attach_reconstruction failed for %s/%s",
                           slug, recon_id, exc_info=True)

    # ---- optional AUTO-PROMOTE: seed canonical from the cardinal turntable frames ----
    # The ONE-CLICK full-identity template (POST .../generate) sets spec.auto_promote so a
    # single action goes mesh -> turntable -> CANONICAL. Guardrails (all deliberate):
    #   * only when a turntable actually attached (a record with N views exists);
    #   * only when N >= 4 — a partial canonical would be a false promise, so skip entirely;
    #   * promote the 4 CARDINAL indices [0, N//4, N//2, (3*N)//4] (deduped + clamped),
    #     indexing the attached reconstruction's own view list (what promote validates).
    # LATEST-WINS (operator RESCINDED the never-clobber rule, 2026-07-14 night: "it is
    # less important now that we have a real base project as a whole to go from"):
    # auto-promote REPLACES any existing canonical with the newest generation's cardinals.
    # This is provenance-safe because mesh reconstruction never reads canonical (the
    # feedback-loop fix); the manual Approve button still allows picking other angles,
    # and ``auto_promote: false`` on the request opts a run out entirely.
    # A promotion failure NEVER fails the job (the mesh + turntable already succeeded) — it
    # is recorded in mesh state as ``auto_promote_error`` and the job still returns ok.
    auto_promote_extra: dict = {}
    if getattr(spec, "auto_promote", False) and attached_rec is not None:
        try:
            rec_views = list(attached_rec.get("views") or [])
            nv = len(rec_views)
            if nv < 4:
                logger.info("identity mesh: auto-promote skipped for %s/%s — only %d "
                            "turntable frames (need >=4 for a cardinal canonical set)",
                            slug, recon_id, nv)
            else:
                cardinal: list[int] = []
                for i in (0, nv // 4, nv // 2, (3 * nv) // 4):
                    i = min(i, nv - 1)  # clamp (defensive; all < nv for nv>=4)
                    if i not in cardinal:
                        cardinal.append(i)  # dedupe, preserve angular order
                identity_profiles.promote_reconstruction_views(slug, recon_id, cardinal)
                auto_promote_extra["auto_promoted"] = True
        except Exception as exc:  # noqa: BLE001 — a promote failure never fails the job
            logger.warning("identity mesh: auto-promote failed for %s/%s",
                           slug, recon_id, exc_info=True)
            auto_promote_extra["auto_promote_error"] = str(exc)

    # ---- record terminal mesh state (the GET mesh-status route reads this) ----
    # front_selection is re-included here (not just in the "running" patch above)
    # because attach_reconstruction(replace=True), just above, REPLACES the whole
    # reconstruction record when a turntable attached — including its "mesh" sub-dict —
    # so anything set only before that attach would otherwise vanish from the terminal
    # state a caller actually reads.
    _set_state({
        "status": "done",
        "error": None,
        "glb_path": glb_path,
        "video_path": video_path,
        "mesh_json_path": mesh_json_path,
        "frame_count": len(frame_paths),
        "textured": bool(spec.texture),
        "job_id": job_id,
        "front_selection": front_selection,
        **auto_promote_extra,
    })

    _delete_remote()  # best-effort remote cleanup after a successful download

    # ---- JobResult outputs: the turntable mp4 (a video MediaRef). The GLB is NOT a
    # MediaRef kind (image/audio/video only), so it is surfaced via mesh state above. ----
    outputs: list = []
    if video_path:
        try:
            outputs.append(make_media_ref(
                asset_id=secrets.token_hex(8), kind="video", uri=video_path,
                mime="video/mp4"))
        except Exception:  # noqa: BLE001 — a MediaRef we can't build never fails the job
            logger.debug("identity mesh: could not build mp4 MediaRef", exc_info=True)
    return JobResult(job_id=job_id, ok=True, outputs=tuple(outputs))
