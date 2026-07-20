"""Identity VIDEO-EXTRACT (char360) bus runner — a RELAY to a remote GPU render service.

Central has NO GPU and NEVER runs char360: it must never import cv2 / ultralytics /
insightface / scenedetect. This runner is a thin HTTP CLIENT to the ``IDENTITY_RENDER_URL``
service (which grew the ``video_extract`` job kind in slice S2 — see
CHAR360-FEATURE-PLAN.md). It mirrors ``runners/identity_render_relay.py`` verbatim where
the patterns apply. It:

  1. POSTs a ``video_extract`` job to ``<IDENTITY_RENDER_URL>/jobs`` authenticating with the
     ``X-Identity-Render-Token`` header. The clip is passed as ``video_path`` (the S2
     escape hatch — ae + central share the ``/mnt/llm_storage`` mount, so a hundreds-of-MB
     video need not be base64-inflated through the request body).
  2. POLLs ``GET /jobs/<id>`` every ~5s until done / error / a generous deadline, honoring a
     cooperative cancel (relayed as a best-effort ``DELETE /jobs/<id>``), mirroring live
     stage/progress/log_tail into the media bus.
  3. On done, DOWNLOADS the manifest ``char360_result.json`` (it is in the ``files`` list),
     then, for EACH detected character, downloads that character's binned view files and
     PERSISTS them locally (atomic writes).
  4. WRITES BACK per character via ``identity_profiles.attach_reconstruction`` — branching on
     ``spec.target``: ``"create"`` mints a NEW profile from that character's views;
     otherwise the views APPEND (``replace=False``) to the existing profile named by the
     slug. The per-character ``face_centroid`` is carried into the recon spec so S3b's
     face-descriptor match path can read it.

REVIEW MODE (``spec.target == "review"`` — CHARACTER-GROUPS-PLAN S1): a THIRD, NON-COMMITTING
branch. Steps 1-3 are IDENTICAL (POST, poll, download the manifest AND every per-character
crop into the storage jail), but step 4 is REPLACED: NO identity profile is created or
appended. Instead the runner returns the per-character grouped views to the caller so the
edit/move/merge UI can curate the partition before anything is committed. The grouped
manifest rides the terminal ``JobResult.groups`` (a plain dict, like ``project``/``movie``),
retrievable via GET /video/jobs/<id> -> ``result.groups``:

    {"n_characters": int,
     "groups": [{"char": str, "face_centroid": [float]|null,
                 "views": [{"url": <abs jailed path handle>, "yaw": float|null,
                            "bin": int|null, "score": float|null}]}]}

Each view ``url`` is the persisted crop's media HANDLE — an absolute path under the storage
jail (the crops land under ``IDENTITIES_HOME/_char360_extracts/<job_id>/``, itself under
DEFAULT_ROOT, so the ``GET /video/media?handle=`` route serves them). The UI renders it via
``mediaBytesUrl(url)`` — the SAME media-byte route the profile canonical views use (the
canonical wire likewise carries bare jailed PATHS, wrapped client-side; central holds no URL
literal). Groups + views are emitted in the manifest's order (bin-ascending per character).

Pure ``(IdentityVideoExtractSpec, job_id) -> JobResult`` (map §6): EVERY expected failure —
an unconfigured service, an unreachable host, a 401, a render error, a timeout, a missing
target profile — is DATA (``JobResult(ok=False, JobError(...))``), never a raise. Only a
genuine programmer error raises, and ``media_bus.run_claimed`` is the one place that catches
that.

Heavy imports (requests, identity_profiles, media_schema, media_bus) are LAZY — done inside
the runner — so importing this module (which ``runners/__init__`` does at boot) stays cheap
and can never break app boot, AND central never pulls in a char360/cv2 dependency. No
pathlib anywhere; os.path only.

Remote service contract (FIXED — the S2 service is built to exactly this):
  * ``POST /jobs``                       -> 202 {job_id}
       body for kind ``video_extract``: {kind, identity_id, video_path XOR video_b64,
       char360_params:{stride?,yolo_model?,min_h_frac?,cluster_dist?,min_faces?}}
  * ``GET  /jobs/<id>``                  -> {job_id, status, error?, files?,
                                           stage?, progress?, log_tail?, updated?}
  * ``GET  /jobs/<id>/files/<path>``     -> raw bytes
  * ``DELETE /jobs/<id>``                -> best-effort cleanup
Auth: header ``X-Identity-Render-Token: <IDENTITY_RENDER_TOKEN>`` (missing/wrong -> 401).

The done manifest ``char360_result.json`` shape (VERIFIED, S2 service.py):
  {ok, kind, identity_id, source_video, n_characters, bins_deg,
   characters:[ {char, views:[{bin,file,yaw,yaw_source,score}], bins_filled,
                 bins_missing, face_centroid} ]}
Each ``views[].file`` is a job-relative posix name like ``char_00/view_00_....png``.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time

from ..result_schema import JobError, JobResult

logger = logging.getLogger(__name__)

# The manifest file the service writes at the job-dir root (in the done ``files`` list).
_MANIFEST_NAME = "char360_result.json"

# Per-request HTTP budget as a (connect, read) TUPLE — mirrors identity_render_relay: a
# short connect side detects a down/firewalled service in seconds (ae's firewall DROPs
# unknown ports — no RST — so a flat timeout eats the whole budget just discovering "nobody
# home"), a generous read side tolerates a large manifest / view download.
_HTTP_TIMEOUT_S = (10.0, 120.0)
# Poll cadence + whole-job deadline. The bus registers identity_video_extract with a 14400s
# timeout; poll a touch under it so the runner returns clean errors-as-data rather than
# being killed mid-poll. Overridable via env (SHARED with the mesh relay's knobs — same
# service, same cadence — so tuning/tests move them together).
_POLL_INTERVAL_S = float(os.getenv("IDENTITY_RENDER_POLL_INTERVAL_S", "5") or "5")
_POLL_DEADLINE_S = float(os.getenv("IDENTITY_RENDER_DEADLINE_S", "14100") or "14100")


def _atomic_write_bytes(dest: str, data: bytes) -> None:
    """Write *data* to *dest* atomically (unique temp in the dest dir + os.replace),
    mirroring identity_render_relay / identity_profiles' copy idiom so a crashed download
    never leaves a half-written artifact at the final name."""
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def _short_token(job_id: str, char_id: str) -> str:
    """A short, human-ish, UNIQUE token for a minted profile name — derived from the
    job_id + char_id plus a little randomness so two ``create`` runs of the same clip never
    collide on a slug. Ordinary Python randomness is fine here (this is a store write, not a
    determinism-sensitive workflow script)."""
    base = "".join(ch for ch in f"{job_id}{char_id}" if ch.isalnum())[-6:]
    return f"{base}{secrets.token_hex(2)}" if base else secrets.token_hex(4)


def _download_file(requests_mod, url: str, headers: dict, remote_id: str,
                   rel_name: str) -> bytes | None:
    """GET one job-relative file from the service; return its bytes, or None on any
    non-200 / transport error (logged, never raised — a missing view is skipped, not fatal)."""
    try:
        fr = requests_mod.get(f"{url}/jobs/{remote_id}/files/{rel_name}",
                              headers=headers, timeout=_HTTP_TIMEOUT_S)
    except requests_mod.RequestException:
        logger.warning("identity video-extract: failed to download %r", rel_name)
        return None
    if fr.status_code != 200:
        logger.warning("identity video-extract: file %r -> HTTP %s", rel_name, fr.status_code)
        return None
    return fr.content


def run_identity_video_extract(spec, job_id: str) -> JobResult:
    """Relay ``spec`` (a source video + a create|add target) to the remote render service,
    poll it, download the per-character view-sets, and write them back into identity
    profiles. Returns ``JobResult(ok=True)`` on success (the created/updated profiles are
    the durable record; there is no MediaRef output kind for a view-set). Every expected
    failure is a clean error-as-data."""
    # Lazy imports keep this module boot-cheap (runners/__init__ imports it at boot) AND
    # keep char360/cv2 off the central side entirely — this runner only ever RELAYS.
    from .. import identity_profiles
    from ..media_bus import is_cancelling, set_progress

    def _fail(code: str, message: str, retryable: bool) -> JobResult:
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code=code, message=message, retryable=retryable))

    url = (os.getenv("IDENTITY_RENDER_URL", "") or "").strip().rstrip("/")
    token = (os.getenv("IDENTITY_RENDER_TOKEN", "") or "").strip()
    if not url or not token:
        return _fail(
            "not_configured",
            "the identity render service is not configured on this host — set "
            "IDENTITY_RENDER_URL and IDENTITY_RENDER_TOKEN (central has no GPU; char360 "
            "video-extracts are relayed to a remote GPU render service).",
            retryable=False)

    import requests  # lazy — present (2.34.2); keeps the module boot-cheap

    headers = {"X-Identity-Render-Token": token}

    target = spec.target
    is_create = (target == "create")
    # REVIEW (CHARACTER-GROUPS-PLAN S1): run char360 + download crops, then RETURN the grouped
    # views WITHOUT writing any profile. Like "create" it names no existing slug, so it skips
    # the ADD profile-existence guard below and synthesizes a correlation id.
    is_review = (target == "review")

    # If this is an ADD to an existing slug, verify the profile EXISTS up front so we fail
    # fast + cleanly (rather than after a full extract) — mirrors the mesh relay's early
    # guard style. A create target is validated per-character at write-back (create_profile
    # raises on a dup slug, which we catch). Review writes nothing, so it too is skipped.
    if not is_create and not is_review:
        try:
            if identity_profiles.get_profile(target) is None:
                return _fail(
                    "no_such_profile",
                    f"identity_video_extract target profile {target!r} does not exist "
                    "(create it first, or use target='create')",
                    retryable=False)
        except Exception:  # noqa: BLE001 — a store read hiccup is transient, not a bad target
            logger.debug("identity video-extract: get_profile(%r) raised (treating as "
                         "present; write-back will surface a real miss)", target, exc_info=True)

    # The service's JobCreateRequest REQUIRES an identity_id. Prefer the spec's (the route
    # passes the slug when adding, or a synthesized id when creating); synthesize a safe
    # fallback so the POST never fails on a missing correlation id. Keep it to the service's
    # safe-id charset (letters/digits/_-.) — a slug already is, and the fallback is hex.
    identity_id = (getattr(spec, "identity_id", None) or "").strip()
    if not identity_id:
        # Only an ADD (a real slug target) reuses the target as the correlation id; create
        # AND review synthesize a fresh one (review names no profile, create mints its own).
        identity_id = (target if (not is_create and not is_review and target)
                       else f"videoextract-{job_id}")

    char360_params = dict(getattr(spec, "char360_params", {}) or {})

    payload = {
        "kind": "video_extract",
        "identity_id": identity_id,
        # video_path (NOT video_b64): the S2 escape hatch for the shared mount — a clip may
        # be hundreds of MB, so base64 through the body is wasteful. spec.source.uri is an
        # absolute path (make_media_ref guarantees it).
        "video_path": spec.source.uri,
        "char360_params": char360_params,
    }

    # ---- POST the job (identical handling to the mesh relay) ----
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

    # ---- poll until done / error / cancel / deadline (mirrors the mesh relay) ----
    deadline = time.time() + _POLL_DEADLINE_S
    files: list = []
    while True:
        if is_cancelling(job_id):
            _delete_remote()
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="cancelled",
                message=f"identity video-extract for {target!r} cancelled by user",
                retryable=False))
        if time.time() > deadline:
            _delete_remote()
            return _fail("render_timeout",
                         f"identity video-extract for {target!r} did not finish within the "
                         f"deadline ({int(_POLL_DEADLINE_S)}s)",
                         retryable=True)
        try:
            pr = requests.get(f"{url}/jobs/{remote_id}", headers=headers,
                              timeout=_HTTP_TIMEOUT_S)
            if pr.status_code == 200:
                pbody = pr.json()
                # ---- live progress relay (additive; older services omit these) ----
                # Mirror ae's per-stage progress + rolling log tail into the media bus (and
                # thence GET /llm/jobs) so a long — or WEDGED — extract surfaces its stage +
                # live log instead of reading progress 0. Best-effort + wrapped; a DB hiccup
                # here NEVER fails the extract (the poll/done/error logic below is untouched).
                if isinstance(pbody, dict) and any(
                        k in pbody for k in ("stage", "progress", "log_tail")):
                    try:
                        blob = {"source": "identity_video_extract",
                                "remote_updated": pbody.get("updated")}
                        for k in ("stage", "progress", "log_tail"):
                            if k in pbody:
                                blob[k] = pbody.get(k)
                        set_progress(job_id, blob)
                    except Exception:  # noqa: BLE001 — progress mirror is best-effort only
                        logger.debug("identity video-extract: progress stamp failed for %s",
                                     job_id, exc_info=True)
                status = pbody.get("status")
                if status == "done":
                    files = pbody.get("files") or []
                    break
                if status == "error":
                    _delete_remote()
                    msg = pbody.get("error") or "the render service reported an error"
                    return _fail("render_failed",
                                 f"identity video-extract failed for {target!r}: {msg}",
                                 retryable=True)
                # queued / running -> keep polling
        except requests.RequestException:
            pass  # transient poll hiccup — keep polling until the deadline
        except ValueError:
            pass  # non-JSON status body — treat as transient
        time.sleep(_POLL_INTERVAL_S)

    # ---- download + parse the manifest (char360_result.json) ----
    if _MANIFEST_NAME not in files:
        _delete_remote()
        return _fail("render_no_manifest",
                     f"the render service reported done for {target!r} but produced no "
                     f"{_MANIFEST_NAME} manifest",
                     retryable=True)
    raw = _download_file(requests, url, headers, remote_id, _MANIFEST_NAME)
    if raw is None:
        _delete_remote()
        return _fail("render_no_manifest",
                     f"could not download the {_MANIFEST_NAME} manifest for {target!r}",
                     retryable=True)
    try:
        manifest = json.loads(raw)
    except (ValueError, TypeError):
        _delete_remote()
        return _fail("render_bad_manifest",
                     f"the render service returned a non-JSON {_MANIFEST_NAME} for {target!r}",
                     retryable=True)
    if not isinstance(manifest, dict):
        _delete_remote()
        return _fail("render_bad_manifest",
                     f"the {_MANIFEST_NAME} manifest for {target!r} was not an object",
                     retryable=True)

    characters = manifest.get("characters")
    if not isinstance(characters, list) or not characters:
        # A clean, honest terminal: the extract ran but found no characters (e.g. <2 faces).
        # This is error-as-data, not a raise — the caller surfaces it; nothing is written.
        _delete_remote()
        n = manifest.get("n_characters")
        return _fail("no_characters",
                     f"the char360 extract found no characters in the source video "
                     f"(n_characters={n!r})",
                     retryable=False)

    # A stable local staging root for the downloaded per-character view files. It lands under
    # IDENTITIES_HOME (a dir the store already owns + creates, itself under DEFAULT_ROOT so it
    # stays inside the media_store jail) in a per-job scratch subdir. attach_reconstruction
    # re-materializes these into each identity's own reconstruction dir (a byte-copy), so this
    # staging copy is just the download landing zone — it only needs to be a readable path.
    stage_root = os.path.join(
        identity_profiles.IDENTITIES_HOME, "_char360_extracts", job_id)

    # ---- per-character: download views + write back (create | add) OR group (review) ----
    created_slugs: list[str] = []
    updated_slugs: list[str] = []
    attached: list[dict] = []
    per_char_errors: list[dict] = []
    # REVIEW accumulator (CHARACTER-GROUPS-PLAN S1): one grouped entry per detected character,
    # emitted in manifest order; unused (stays empty) for create/add.
    groups: list[dict] = []

    for idx, ch in enumerate(characters):
        if is_cancelling(job_id):  # cooperative cancel between characters
            _delete_remote()
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="cancelled",
                message=f"identity video-extract for {target!r} cancelled by user",
                retryable=False))
        if not isinstance(ch, dict):
            per_char_errors.append({"index": idx, "error": "character entry was not an object"})
            continue
        char_id = ch.get("char")
        if not isinstance(char_id, str) or not char_id.strip():
            char_id = f"char_{idx:02d}"
        views = ch.get("views")
        if not isinstance(views, list) or not views:
            per_char_errors.append({"char": char_id, "error": "no views for this character"})
            continue

        # Download each view file into the staging root, preserving the service's job-relative
        # subpath (char_NN/<file>) so distinct characters never collide. Keep angular order:
        # the manifest's per-character views are already emitted bin-ascending by the service.
        # ``view_records`` carries the persisted dest path PLUS each view's yaw/bin/score (the
        # review contract needs them; create/add reads only the paths via ``view_paths``).
        view_records: list[dict] = []
        for v in views:
            if not isinstance(v, dict):
                continue
            rel = v.get("file")
            if not isinstance(rel, str) or not rel.strip():
                continue
            # Defense-in-depth: basename every component so a hostile "../" in a
            # service-supplied name can never escape the staging dir (the service is
            # trusted, but the write-back path must be un-escapable).
            safe_rel = "/".join(
                os.path.basename(part) for part in rel.replace("\\", "/").split("/") if part)
            if not safe_rel:
                continue
            data = _download_file(requests, url, headers, remote_id, rel)
            if data is None:
                continue  # a missing view is skipped, never fatal (attach records the rest)
            dest = os.path.join(stage_root, safe_rel)
            try:
                _atomic_write_bytes(dest, data)
            except OSError:
                logger.warning("identity video-extract: could not persist %r -> %s", rel, dest)
                continue
            view_records.append({
                "url": dest,               # the media HANDLE (abs jailed path); UI wraps it
                "yaw": v.get("yaw"),       # the view's yaw in degrees (float) or null
                "bin": v.get("bin"),       # the yaw-bin index (int) or null
                "score": v.get("score"),   # the detection/pose score (float) or null
            })

        view_paths = [r["url"] for r in view_records]
        if not view_paths:
            per_char_errors.append({"char": char_id, "error": "no downloadable views"})
            continue

        # REVIEW (CHARACTER-GROUPS-PLAN S1): DO NOT write a profile — accumulate the grouped
        # views for the curation UI and move on. face_centroid rides through so the UI (and a
        # later commit) can key on it. Nothing about the store is touched in this branch.
        if is_review:
            groups.append({
                "char": char_id,
                "face_centroid": ch.get("face_centroid"),
                "views": view_records,
            })
            continue

        n = len(view_paths)
        # The provenance spec carried into attach_reconstruction. The mode is the NEW
        # "video_extract" (the write-back mode gate is widened to keep it, not downgrade it
        # to "sheet"). ``char`` + ``face_centroid`` ride through the (widened) allow-list so
        # S3b's face-descriptor match path can read the per-character centroid off the record.
        face_centroid = ch.get("face_centroid")  # a list[float] or null (faceless cluster)
        recon_spec = {
            "source": "identity_video_extract_relay",
            "mode": "video_extract",
            "frame_count": n,
            "degrees_per_frame": (round(360.0 / n, 2) if n else None),
            "job_id": job_id,
            "char": char_id,
            "face_centroid": face_centroid,
        }

        # A recon_id unique per (job, character) so a re-run of THIS extract updates in place
        # (replace not needed — we always append for add; for create the profile itself is
        # fresh) and two characters never share a recon bundle dir.
        recon_id = f"videoextract_{job_id}_{char_id}"

        try:
            if is_create:
                # CREATE a new profile from this character's binned views, then attach the
                # same views as a video_extract reconstruction. Name it honestly + uniquely.
                name = f"video-char-{char_id}-{_short_token(job_id, char_id)}"
                prof = identity_profiles.create_profile(
                    name, list(view_paths[:identity_profiles.MAX_SOURCE_IMAGES]), notes="")
                new_slug = prof.get("slug") if isinstance(prof, dict) else None
                if not new_slug:
                    per_char_errors.append(
                        {"char": char_id, "error": "create_profile returned no slug"})
                    continue
                rec = identity_profiles.attach_reconstruction(
                    new_slug, recon_id, view_paths, spec=recon_spec, replace=False)
                if rec is None:
                    per_char_errors.append(
                        {"char": char_id, "slug": new_slug,
                         "error": "attach_reconstruction found no active profile after create"})
                    continue
                created_slugs.append(new_slug)
                attached.append({"char": char_id, "slug": new_slug, "recon_id": recon_id,
                                 "frame_count": n})
            else:
                # ADD to the named slug: APPEND (replace=False) — these video-derived sets are
                # additive; do NOT clobber existing reconstructions.
                rec = identity_profiles.attach_reconstruction(
                    target, recon_id, view_paths, spec=recon_spec, replace=False)
                if rec is None:
                    # The profile vanished (archived) between the up-front check and here —
                    # surface it as a clean per-character error, keep going for the rest.
                    per_char_errors.append(
                        {"char": char_id, "slug": target,
                         "error": "attach_reconstruction found no active profile (archived?)"})
                    continue
                if target not in updated_slugs:
                    updated_slugs.append(target)
                attached.append({"char": char_id, "slug": target, "recon_id": recon_id,
                                 "frame_count": n})
        except identity_profiles.ProfileError as exc:
            # A store-contract failure for THIS character (e.g. a duplicate slug on create) is
            # per-character error-as-data — record it and keep processing the others.
            per_char_errors.append({"char": char_id, "error": f"{exc}"})
            continue
        except Exception as exc:  # noqa: BLE001 — never let one character's write-back kill the job
            logger.warning("identity video-extract: write-back raised for char %s",
                           char_id, exc_info=True)
            per_char_errors.append({"char": char_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

    _delete_remote()  # best-effort remote cleanup after a successful download

    # ---- REVIEW terminal (CHARACTER-GROUPS-PLAN S1) ----
    # Return the grouped views WITHOUT having written any profile. If not a single character
    # yielded a downloadable view, that is an honest failure (nothing to curate) — errors-as-
    # data, mirroring the write-back guard below. Otherwise ok=True with the grouped manifest
    # on JobResult.groups (GET /video/jobs/<id> -> result.groups). n_characters is the count
    # of GROUPS actually built (characters with >=1 downloaded crop), not the raw manifest
    # count, so the UI never renders an empty partition.
    if is_review:
        if not groups:
            detail = "; ".join(f"{e.get('char', '?')}: {e.get('error')}"
                               for e in per_char_errors)
            return _fail("no_review_groups",
                         "identity video-extract review produced no groups"
                         + (f" ({detail})" if detail else ""),
                         retryable=False)
        logger.info("identity video-extract REVIEW done: %d group(s), %d char error(s)",
                    len(groups), len(per_char_errors))
        return JobResult(job_id=job_id, ok=True,
                         groups={"n_characters": len(groups), "groups": groups})

    # If NOT a single character wrote back, that is a genuine failure (nothing landed) — an
    # honest error-as-data rather than a hollow ok. Otherwise the job succeeds even if some
    # characters errored (the successes are the durable record; the per-char errors are
    # logged and surfaced in the JobResult is not a field, so we just return ok=True).
    if not attached:
        detail = "; ".join(f"{e.get('char', '?')}: {e.get('error')}" for e in per_char_errors)
        return _fail("write_back_failed",
                     f"identity video-extract for {target!r} wrote back no characters"
                     + (f" ({detail})" if detail else ""),
                     retryable=False)

    logger.info(
        "identity video-extract done for target=%r: created=%s updated=%s attached=%d errors=%d",
        target, created_slugs, updated_slugs, len(attached), len(per_char_errors))

    # No MediaRef output kind for a view-set (image/audio/video only, and these are stills
    # already owned by the profiles) — the created/updated profiles + their reconstructions
    # are the durable record a caller reads via GET /video/identity-profiles.
    return JobResult(job_id=job_id, ok=True)
