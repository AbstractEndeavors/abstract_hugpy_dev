"""Identity-profile store — the DURABLE form of "the reference set IS the identity".

An identity profile is a NAMED library item: ``{name, source_images (1..4),
created_at, notes?}``. The operator's vision (STUDIO-ROADMAP.md, "IDENTITY
PROFILES") is that a character's DNA — its curated reference set — is created
ONCE and associated anywhere (single clips, movies, stills) instead of being
re-supplied per request. This module is stage (a): the library item itself.
Stage (b) — turnaround generation from a profile + the re-edit loop that promotes
an approved rendering to the canonical reference — comes next and layers on top
of this store; nothing here forecloses it (a future ``canonical`` ref set is just
another key on the entry).

Storage is a first-class directory of its own — the Identities feature OWNS its
reference images instead of merely pointing at ephemeral ``uploads/`` paths that
the session-scoped upload reaper (upload_routes._wipe_session) is free to erase::

    <IDENTITIES_HOME>/
      identity_profiles.json          # the registry (source of truth), {profiles,_deleted}
      <slug>/
        profile.json                  # human-readable denormalized MIRROR of the entry
        ref_00.<ext>  ref_01.<ext> …  # the identity's reference images, COPIED IN
        _superseded/<ts>/…            # refs replaced by an update (never erased)
      _deleted/
        <slug>@<ts>/                  # archived identity dirs (moved here on delete)

PERSISTENCE INVARIANT (operator 2026-07-13 — "identities are persistent, their
reference images must never be reaped"): IDENTITIES_HOME is a SIBLING of
UPLOADS_HOME (both under DEFAULT_ROOT), NEVER a child of it. The upload reaper is
jailed to UPLOADS_HOME (upload_routes._within_uploads), so it structurally cannot
reach an identity's copied images. This module also NEVER registers a copied ref
with any upload-session tracking (it never calls _touch_session, never writes
under UPLOADS_HOME/.sessions/): an identity's pixels live ONLY under
``IDENTITIES_HOME/<slug>/`` and answer to nothing but this store.

On create the validated source paths (which the route has already jail-resolved +
image-classified) are COPIED into the identity's dir; the entry's
``source_images`` then point at the identity-owned copies (still under
DEFAULT_ROOT, so the media_store jail still accepts them). The registry lives with
the other durable registries and survives restarts; the copies survive the reaper.

Single-writer discipline is the api_keys._save idiom REUSED VERBATIM: a
process-wide lock plus a unique-per-write temp file (pid + token) renamed onto
the target with os.replace as the sole atomicity point — so two concurrent
writers never race between open() and replace() (that exact race bit a batch of
/v1 auths on 2026-07-11). The same temp-name+os.replace atomicity guards every
copied ref and every profile.json write. Never-delete doctrine: a delete ARCHIVES
the entry under ``_deleted`` AND MOVES the identity's dir under ``_deleted/`` —
bytes are never erased, only relocated; an update MOVES superseded refs under
``_superseded/`` rather than overwriting them.

Backward-compat: the FIRST load after this change, when the new registry is absent
but the legacy ``PROJECTS_HOME/identity_profiles.json`` exists, migrates each
active profile into its own dir (COPYING referenced images, never moving), leaving
the legacy registry and the original uploads UNTOUCHED (reversible). A missing
source never crashes and never drops the identity — the entry is kept, the source
recorded in ``missing_references``. Guarded by new-registry-absence, so it runs at
most once and is idempotent.
"""
from __future__ import annotations

import os
import re
import json
import secrets
import shutil
import threading
import time
import unicodedata
from typing import Any, Optional

from abstract_hugpy_dev.imports.src.constants.constants import (
    IDENTITIES_HOME,
    PROJECTS_HOME,
    UPLOADS_HOME,
)

# json is imported lazily inside _load/_save to keep the module import cheap and
# to mirror how the route modules defer json/sqlite until first use.

# video_intel/identity_profiles.py (Append to existing file)

from .identity_reconstruction_schema import IdentitySingleViewRegenSpec, IdentityMeshSpec

_LOCK = threading.Lock()

MAX_SOURCE_IMAGES = 12
MAX_CANONICAL_IMAGES = 4  # Canonical anchors stay capped at 4

# --------------------------------------------------------------------------- #
# VERSIONS slice (operator-directed 2026-07-14; IDENTITY-VERSIONS-SLICE.md).
#
# An identity holds ONE base (the clay mesh minted on the first clay generate,
# name "base") + N append-only VERSIONS — named render-sets minted by the mesh
# RELAY on each successful generate. Never-delete lives here too: a version is
# ARCHIVED (flagged, dropped from the wire list), its pixels never erased.
#
# ``gen_settings`` are the per-identity generation defaults the left-column
# Settings tab persists and the Generate click prefills. ALWAYS present on the
# wire (defaulted) so the UI can rely on the shape. Additive keys only — nothing
# below removes or renames an existing profile field.
# --------------------------------------------------------------------------- #
VERSION_KINDS = ("clay", "textured", "styled")

# The canonical happy-path generation settings (defaults-are-promises: a bare
# Generate click == these). Merged over any stored partial in ``_public`` so the
# wire shape is complete even for a profile that has never had settings PATCHed.
_DEFAULT_GEN_SETTINGS: dict[str, Any] = {
    "texture": True,
    "pose": "none",          # "none" | "t-pose"
    "frame_count": 72,
    "fps": 24,
    "width": 768,
    "height": 768,
    "auto_promote": True,
    "front_ref": None,       # abs path (one of the profile's own refs) | null
    "remove_background": True,
}
_POSE_CHOICES = ("none", "t-pose")

class ProfileError(ValueError):
    """Bad-input on the store contract (empty name, no/too-many refs, dup slug).

    A ValueError subclass so the route's existing ``except (ValueError, TypeError)``
    idiom catches it as a clean 4xx; ``code`` lets the route pick the precise
    status (409 for a duplicate, 400 for the rest) — errors-as-data, never a raw
    500 crossing the HTTP boundary."""

    def __init__(self, message: str, code: str = "invalid_profile") -> None:
        super().__init__(message)
        self.code = code


def _store_path() -> str:
    # Call-time resolution (not import-time) mirrors api_keys._store_path: it reads
    # the module global IDENTITIES_HOME so a test that rebinds it to a temp dir lands
    # the whole store there, never in the real identities tree. (Env isolation does
    # not work — constants' get_env_value reads the .env file — so the direct module
    # rebind is the honest lever, exactly as the store test already does.)
    return os.path.join(IDENTITIES_HOME, "identity_profiles.json")


def _legacy_store_path() -> str:
    # The pre-migration location: a single buried JSON under PROJECTS_HOME whose
    # entries merely pointed at ephemeral uploads/ paths. Read-only from here on —
    # the migration COPIES out of it and never mutates or deletes it (reversibility).
    return os.path.join(PROJECTS_HOME, "identity_profiles.json")


def _identity_dir(slug: str) -> str:
    # The per-identity folder that OWNS this profile's reference images. A sibling
    # of the registry under IDENTITIES_HOME, itself a sibling of UPLOADS_HOME — see
    # the module docstring's PERSISTENCE INVARIANT: the upload reaper cannot reach here.
    return os.path.join(IDENTITIES_HOME, slug)


def _ext_for(src: str) -> str:
    """The lowercased extension to give a copied ref (source ext preserved). A
    source with no extension gets ``.img`` so the copy always has a stable name."""
    ext = os.path.splitext(src)[1].lower()
    return ext if ext else ".img"


def _atomic_copy(src: str, dest: str) -> None:
    """Copy *src* -> *dest* with the store's atomicity idiom: copy to a unique temp
    name IN the destination dir, then os.replace onto the final name (the sole
    atomicity point). Raises on a missing/unreadable source (callers tolerate it)."""
    tmp = f"{dest}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def _materialize_refs(dir_path: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """COPY each source in order into ``dir_path/ref_NN.<ext>`` and return
    ``(source_images, missing_references)``.

    Order is preserved and the ref number is the list POSITION, so a rendered set
    stays aligned even when a middle source is absent. A missing/unreadable source
    NEVER crashes: nothing is copied for it, its ORIGINAL path is retained in the
    returned reference set, and it is recorded in ``missing_references`` (so a UI
    can flag a broken ref and a future re-copy has the original handle). The
    never-delete doctrine holds — this only ever writes new ``ref_NN`` files."""
    os.makedirs(dir_path, exist_ok=True)
    refs: list[str] = []
    missing: list[str] = []
    for i, src in enumerate(sources):
        dest = os.path.join(dir_path, f"ref_{i:02d}{_ext_for(src)}")
        if os.path.isfile(src):
            try:
                _atomic_copy(src, dest)
                refs.append(dest)
                continue
            except OSError:
                pass  # fall through to the missing-source path
        refs.append(src)      # keep the original handle for a broken/absent source
        missing.append(src)
    return refs, missing


def _write_profile_json(slug: str, entry: dict[str, Any]) -> None:
    """(Re)write the human-readable denormalized MIRROR of an entry at
    ``<slug>/profile.json``. The registry is the source of truth; this is a
    convenience view regenerated on every create/update. Same atomic temp+replace."""
    dir_path = _identity_dir(slug)
    os.makedirs(dir_path, exist_ok=True)
    doc = {
        "slug": slug,
        "name": entry.get("name", ""),
        "created_at": entry.get("created_at"),
        "notes": entry.get("notes", ""),
        "source_images": list(entry.get("source_images") or []),
        "missing_references": list(entry.get("missing_references") or []),
        # stage (b) additive keys: the generated turnaround renderings awaiting
        # approval, and any views the operator has promoted to the canonical set.
        "reconstructions": list(entry.get("reconstructions") or []),
        "canonical": list(entry.get("canonical") or []),
    }
    path = os.path.join(dir_path, "profile.json")
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _supersede_existing_refs(dir_path: str) -> None:
    """MOVE the identity's current ``ref_NN.*`` files aside into
    ``<dir>/_superseded/<ts>/`` before a new set is copied in — never-delete: a
    replaced reference set is relocated, never overwritten or erased. No-op when the
    dir is absent or holds no ref files. (Only the top-level ``ref_*`` files move;
    prior ``_superseded`` archives and ``profile.json`` stay put.)"""
    if not os.path.isdir(dir_path):
        return
    existing = [
        n for n in os.listdir(dir_path)
        if n.startswith("ref_") and not n.endswith(".tmp")
        and os.path.isfile(os.path.join(dir_path, n))
    ]
    if not existing:
        return
    dest_dir = os.path.join(dir_path, "_superseded", f"{time.time()}")
    os.makedirs(dest_dir, exist_ok=True)
    for n in existing:
        shutil.move(os.path.join(dir_path, n), os.path.join(dest_dir, n))


def _materialize_refs_staged(dir_path: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """Two-stage COPY-commit variant of ``_materialize_refs`` for UPDATES, where an
    incoming path may BE one of the identity's OWN current ``ref_NN`` files (the UI reads
    the owned copies back and re-submits them when editing the set). The old update flow
    superseded the current refs FIRST and then copied from ``sources`` — so a
    self-referential source had already been MOVED under ``_superseded/`` and was recorded
    as MISSING, desyncing the registry (it named ``ref_01..ref_11`` while disk held
    ``ref_00..ref_10``, and every path failed ``os.path.isfile`` downstream).

    Here every readable source is COPIED to a ``_stage_NN`` name (extension preserved)
    BEFORE anything is superseded — so a self-reference is captured while it still exists;
    only THEN is the current set relocated (never-delete, via ``_supersede_existing_refs``)
    and the staged files committed to their final position-numbered ``ref_NN`` names. A
    missing/unreadable source behaves exactly as in ``_materialize_refs`` (its ORIGINAL
    path is retained in the returned set and recorded in ``missing``)."""
    os.makedirs(dir_path, exist_ok=True)
    # STAGE 1 — copy each readable source to a staging name (distinct from ref_*), so
    # NOTHING is superseded until every source is safely captured off its origin.
    staged: dict[int, tuple[str, str]] = {}   # position -> (stage_path, ext)
    originals: dict[int, str] = {}            # position -> original path (missing source)
    missing: list[str] = []
    for i, src in enumerate(sources):
        ext = _ext_for(src)
        if os.path.isfile(src):
            stage = os.path.join(dir_path, f"_stage_{i:02d}{ext}")
            try:
                _atomic_copy(src, stage)
                staged[i] = (stage, ext)
                continue
            except OSError:
                pass
        originals[i] = src
        missing.append(src)
    # STAGE 2 — relocate the current ref_* set now that every new source is staged.
    _supersede_existing_refs(dir_path)
    # STAGE 3 — commit staged files to their final ref_NN names (position preserved).
    refs: list[str] = []
    for i in range(len(sources)):
        if i in staged:
            stage, ext = staged[i]
            dest = os.path.join(dir_path, f"ref_{i:02d}{ext}")
            os.replace(stage, dest)
            refs.append(dest)
        else:
            refs.append(originals[i])
    return refs, missing


# --------------------------------------------------------------------------- #
# stage (b) — reconstruction (turnaround) + canonical helpers. Reuse the store's
# atomic-copy / supersede / atomic-json idioms VERBATIM so the generated stills
# and the promoted canonical set answer to nothing but this store, exactly like a
# profile's reference images do.
# --------------------------------------------------------------------------- #
def _reconstruction_root(slug: str) -> str:
    """``<slug>/reconstruction`` — the folder holding every recon bundle for an
    identity (one ``<recon_id>/`` subdir per generated turnaround set)."""
    return os.path.join(_identity_dir(slug), "reconstruction")


def _reconstruction_dir(slug: str, recon_id: str) -> str:
    """``<slug>/reconstruction/<recon_id>`` — one generated turnaround set's bundle
    (its ``views/`` stills + ``manifest.json``)."""
    return os.path.join(_reconstruction_root(slug), recon_id)


def _canonical_dir(slug: str) -> str:
    """``<slug>/canonical`` — the promoted canonical reference set (``ref_NN.<ext>``),
    a SIBLING key of the identity's own ``ref_NN`` uploads. The module docstring
    (stage (b)) anticipates this: "a future ``canonical`` ref set is just another key
    on the entry"."""
    return os.path.join(_identity_dir(slug), "canonical")


def _atomic_write_json(path: str, doc: dict[str, Any]) -> None:
    """Write *doc* as pretty JSON to *path* with the store's atomicity idiom (unique
    temp name in the dest dir + os.replace). Mirrors ``_write_profile_json``'s write."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _supersede_existing_dir(target_dir: str) -> None:
    """MOVE an existing *target_dir* aside under ``<parent>/_superseded/<ts>/<base>``
    before a fresh set is written in its place — never-delete: a replaced recon
    bundle or canonical set is relocated, never overwritten or erased. No-op when the
    dir is absent. Mirrors ``_supersede_existing_refs`` (which moves individual ref
    files) but at whole-directory granularity."""
    if not os.path.isdir(target_dir):
        return
    parent = os.path.dirname(target_dir)
    base = os.path.basename(target_dir)
    dest = os.path.join(parent, "_superseded", f"{time.time()}", base)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(target_dir, dest)


def _materialize_views(dir_path: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """COPY each produced still in order into ``dir_path/view_NN.png`` and return
    ``(views, missing)``. Order is preserved and the view number is the list POSITION.
    A missing/unreadable source NEVER crashes (mirrors ``_materialize_refs``): nothing
    is copied for it, its ORIGINAL path is retained in the returned list, and it is
    recorded in ``missing``. Stills come out of the frame-extract job as PNG, so every
    owned copy is normalized to ``.png``."""
    os.makedirs(dir_path, exist_ok=True)
    views: list[str] = []
    missing: list[str] = []
    for i, src in enumerate(sources):
        dest = os.path.join(dir_path, f"view_{i:02d}.png")
        if isinstance(src, str) and os.path.isfile(src):
            try:
                _atomic_copy(src, dest)
                views.append(dest)
                continue
            except OSError:
                pass  # fall through to the missing-source path
        views.append(src)     # keep the original handle for a broken/absent source
        missing.append(src)
    return views, missing


def _archive_identity_dir(slug: str, deleted_at: float) -> None:
    """MOVE ``<slug>/`` under ``_deleted/<slug>@<deleted_at>/`` (correlating with the
    registry archive key) so a delete relocates the identity's owned pixels rather
    than erasing them. Best-effort and tolerant: a missing dir or an FS error must
    never fail the delete (the registry archive is the durable record of record)."""
    src = _identity_dir(slug)
    if not os.path.isdir(src):
        return
    try:
        graveyard = os.path.join(IDENTITIES_HOME, "_deleted")
        os.makedirs(graveyard, exist_ok=True)
        shutil.move(src, os.path.join(graveyard, f"{slug}@{deleted_at}"))
    except OSError:
        pass  # never erase, and never let a relocation hiccup fail the archive


def _empty() -> dict[str, Any]:
    return {"profiles": {}, "_deleted": {}}


def _migrate_legacy() -> Optional[dict[str, Any]]:
    """One-time, idempotent, reversible migration from the legacy single-file store.

    Runs ONLY when the new registry is absent AND the legacy
    ``PROJECTS_HOME/identity_profiles.json`` exists (the caller guards on
    new-registry-absence, so this fires at most once). For each ACTIVE profile it
    creates ``<slug>/`` and COPIES (never moves) each referenced image into
    ``ref_NN.<ext>``, rewriting source_images to the copies and recording any
    missing sources in ``missing_references`` — a missing source neither crashes
    nor drops the identity (an all-missing entry is kept with an empty-copied set,
    its original paths retained, and the full missing list). ``_deleted`` entries
    carry over as-is (archived dirs are not materialized). The legacy registry and
    every original uploads/ file are left UNTOUCHED (reversibility). Returns the new
    registry (already written atomically), or None when there is nothing to migrate."""
    if os.path.exists(_store_path()):
        return None  # already migrated — never run twice
    legacy_path = _legacy_store_path()
    if not os.path.exists(legacy_path):
        return None
    try:
        with open(legacy_path, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except (OSError, ValueError):
        return None  # a corrupt legacy file is not worth crashing the feature over
    if not isinstance(legacy, dict):
        return None

    profiles = legacy.get("profiles") if isinstance(legacy.get("profiles"), dict) else {}
    deleted = legacy.get("_deleted") if isinstance(legacy.get("_deleted"), dict) else {}
    new_data: dict[str, Any] = {"profiles": {}, "_deleted": dict(deleted)}
    for slug, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        sources = [r for r in (entry.get("source_images") or []) if isinstance(r, str)]
        refs, missing = _materialize_refs(_identity_dir(slug), sources)
        new_entry["source_images"] = refs
        if missing:
            new_entry["missing_references"] = missing
        else:
            new_entry.pop("missing_references", None)
        new_data["profiles"][slug] = new_entry
        _write_profile_json(slug, new_entry)
    _save(new_data)
    return new_data


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        migrated = _migrate_legacy()
        if migrated is not None:
            return migrated
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        # A corrupt/half-written file must not wedge the feature — the honest
        # answer is an empty store (the next _save re-writes it cleanly).
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data.setdefault("profiles", {})
    data.setdefault("_deleted", {})
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Temp name UNIQUE PER WRITE (pid + token): several gunicorn processes may
    # write here, and two writers sharing one "<path>.tmp" race between open()
    # and os.replace() — the loser's replace() dies FileNotFoundError. pid+token
    # keeps every write atomic AND collision-free; os.replace is the atomicity
    # point. (Lifted from api_keys._save, which learned this the hard way.)
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def slugify(name: str) -> str:
    """A stable, filesystem-safe, url-safe slug from a display name. NFKD-fold to
    ascii, lowercase, non-alphanumerics -> single hyphens, trim. The slug is the
    profile's identity in the store + the DELETE route path segment."""
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", norm).strip("-").lower()
    return slug


def _public(slug: str, entry: dict[str, Any]) -> dict[str, Any]:
    """The wire shape for one profile — the slug folded in alongside the stored
    fields so a list row is self-describing (the caller keys deletes on it)."""
    out = {
        "slug": slug,
        "name": entry.get("name", ""),
        # WIRE key is ``reference_images`` (what the UI schema + video_routes'
        # identity resolver + tests read), sourced from the internal
        # ``source_images`` storage field. Keep these two names in sync: emitting
        # ``source_images`` on the wire silently blanks the UI's reference set and
        # makes the movie/i2v identity resolver see an empty identity.
        "reference_images": list(entry.get("source_images") or []),
        "created_at": entry.get("created_at"),
        "notes": entry.get("notes", ""),
    }
    # Additive: only surfaced when some source could not be copied in, so a broken
    # reference is honestly visible (a UI can show a badge). Absent on the happy path.
    missing = entry.get("missing_references")
    if missing:
        out["missing_references"] = list(missing)
    # Stage (b) additive keys — ALWAYS present (default empty) so a UI can rely on the
    # shape: ``reconstructions`` is the ordered list of generated turnaround sets
    # awaiting approval; ``canonical`` is the promoted reference set (empty until the
    # operator promotes recon views). Neither removes/renames an existing key.
    out["reconstructions"] = [
        _recon_manifest(r) for r in (entry.get("reconstructions") or [])
        if isinstance(r, dict)
    ]
    out["canonical"] = list(entry.get("canonical") or [])
    canonical_missing = entry.get("canonical_missing")
    if canonical_missing:
        out["canonical_missing"] = list(canonical_missing)
    # VERSIONS slice additive keys — ALWAYS present (defaulted) so the UI can rely
    # on the shape. ``versions`` omits archived entries (never-delete: they stay in
    # storage, just off the wire); ``active_version`` is the id_lock DNA pointer;
    # ``gen_settings`` is the full happy-path defaults merged over any stored partial.
    out["versions"] = [
        _public_version(v) for v in (entry.get("versions") or [])
        if isinstance(v, dict) and not v.get("archived")
    ]
    out["active_version"] = entry.get("active_version")
    out["gen_settings"] = _merged_gen_settings(entry.get("gen_settings"))
    return out


def _public_version(v: dict[str, Any]) -> dict[str, Any]:
    """The FIXED wire shape for one version — exactly the seven contract keys (a
    sibling UI is built to these). The internal-only ``base``/``archived`` flags stay
    OFF the wire; the base version is identifiable by ``name == "base"`` + ``kind ==
    "clay"``, and archived versions are omitted from the list entirely."""
    return {
        "version_id": v.get("version_id"),
        "name": v.get("name", ""),
        "kind": v.get("kind", "clay"),
        "recon_id": v.get("recon_id"),
        "created_at": v.get("created_at"),
        "canonical": list(v.get("canonical") or []),
        "notes": v.get("notes", ""),
    }


def _merged_gen_settings(stored: Any) -> dict[str, Any]:
    """The full happy-path defaults with any stored partial layered on top — only
    KNOWN keys survive (a stale/unknown stored key is dropped from the wire), so the
    shape is always exactly the contract's nine keys."""
    gs = dict(_DEFAULT_GEN_SETTINGS)
    if isinstance(stored, dict):
        for k in _DEFAULT_GEN_SETTINGS:
            if k in stored:
                gs[k] = stored[k]
    return gs


def list_profiles() -> list[dict[str, Any]]:
    """Active (non-archived) profiles, newest first."""
    with _LOCK:
        data = _load()
    out = [_public(slug, entry) for slug, entry in data["profiles"].items()]
    out.sort(key=lambda p: p.get("created_at") or 0, reverse=True)
    return out


def get_profile(slug: str) -> Optional[dict[str, Any]]:
    """One active profile by slug, or None if unknown/archived."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
    if entry is None:
        return None
    return _public(slug, entry)


def create_profile(
    name: str,
    source_images: list[str],
    notes: str = "",
) -> dict[str, Any]:
    """Create a profile from a display name + a set of ALREADY-VALIDATED reference
    image paths (the route jail-resolves + ffprobe-classifies them as images
    first, exactly like the movie route; this store trusts what it is handed and
    persists it). Slug is derived from the name; a collision with an existing
    ACTIVE slug raises ProfileError(code="duplicate") -> the route's 409. The
    validated sources are COPIED into the identity's own dir (``<slug>/ref_NN.<ext>``)
    and the entry then points at those identity-owned copies, 1..4 in order (order
    is meaningful — an id_lock hash keys on it). A source gone missing does NOT crash
    the create: it is skipped, its original path retained, recorded in
    ``missing_references``. This is the escape from the upload reaper — the pixels
    now live under IDENTITIES_HOME, not the ephemeral uploads/ tree."""
    display = (name or "").strip()
    if not display:
        raise ProfileError("name is required", code="invalid_profile")
    if not isinstance(source_images, list) or not source_images:
        raise ProfileError("at least one reference_image is required", code="invalid_profile")
    # Change MAX_source_images to MAX_SOURCE_IMAGES here:
    if len(source_images) > MAX_SOURCE_IMAGES:
        raise ProfileError(
            f"at most {MAX_SOURCE_IMAGES} source_images are accepted",
            code="invalid_profile",
        )
    refs: list[str] = []
    for raw in source_images:
        if not isinstance(raw, str) or not raw.strip():
            raise ProfileError("each reference_image must be a non-empty path", code="invalid_profile")
        refs.append(raw)
    slug = slugify(display)
    if not slug:
        raise ProfileError("name has no url-safe characters", code="invalid_profile")

    entry: dict[str, Any] = {
        "name": display,
        "source_images": refs,  # replaced by the identity-owned copies below
        "created_at": time.time(),
        "notes": (notes or "").strip(),
    }
    with _LOCK:
        data = _load()
        if slug in data["profiles"]:
            raise ProfileError(f"a profile named {display!r} already exists", code="duplicate")
        # COPY the validated sources into the identity's own dir; the entry now
        # points at the identity-owned copies (under IDENTITIES_HOME, still within
        # DEFAULT_ROOT -> media_store jail OK) and so survives the upload reaper.
        owned, missing = _materialize_refs(_identity_dir(slug), refs)
        entry["source_images"] = owned
        if missing:
            entry["missing_references"] = missing
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def update_profile(
    slug: str,
    *,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    source_images: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    """Edit an EXISTING active profile IN PLACE. A true partial update: each of
    ``name``/``notes``/``source_images`` left at its default (``None``) is a
    no-op for that field — the route only forwards the keys actually present in
    the PATCH body, so this never needs a sentinel to distinguish "omitted" from
    "cleared".

    SLUG STABILITY (why a rename never re-slugs): the slug is this profile's
    identity everywhere OUTSIDE this store — it is the ``<slug>`` segment in the
    route path, and every ``identity_profile:<slug>`` reference a saved template,
    movie spec, or enqueue body carries. Re-deriving the slug from a new display
    name on rename would silently strand every one of those references (a
    template built against "mira" would 404 the moment someone renamed the
    display name to "Mira Prime"). So ``name`` is DISPLAY-ONLY here: the dict key
    in ``data["profiles"]`` never changes, only the stored ``name`` string does.

    ``source_images``, when given, must be a non-empty 1..MAX_source_images
    list (the route has already jail-resolved + image-classified it exactly like
    POST create) — an identity is never left pointing at zero references. Pass it
    as ``None`` (the default) to leave the current reference set untouched. When a
    new set IS given it is COPIED into the identity's dir renumbered from ref_00,
    and the superseded ref files are MOVED under ``<slug>/_superseded/<ts>/`` — the
    prior pixels are never erased, only relocated (never-delete doctrine). The
    ``profile.json`` mirror is regenerated on every update.

    Returns the updated public shape, or ``None`` if the slug names no ACTIVE
    profile (unknown or archived) — the route turns that into the same 404
    get/delete already give an unknown slug. Archives nothing: this mutates the
    entry in place, so ``created_at`` and the archive history are untouched."""
    if not slug or not isinstance(slug, str):
        return None

    new_name: Optional[str] = None
    if name is not None:
        new_name = name.strip()
        if not new_name:
            raise ProfileError("name is required", code="invalid_profile")

    new_refs: Optional[list[str]] = None
    if source_images is not None:
        if not isinstance(source_images, list) or not source_images:
            raise ProfileError("at least one reference_image is required", code="invalid_profile")
        # Change MAX_source_images to MAX_SOURCE_IMAGES here:
        if len(source_images) > MAX_SOURCE_IMAGES:
            raise ProfileError(
                f"at most {MAX_SOURCE_IMAGES} source_images are accepted",
                code="invalid_profile",
            )
        new_refs = []
        for raw in source_images:
            if not isinstance(raw, str) or not raw.strip():
                raise ProfileError("each reference_image must be a non-empty path", code="invalid_profile")
            new_refs.append(raw)

    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        entry = dict(entry)  # edit a copy — nothing is written until _save
        if new_name is not None:
            entry["name"] = new_name
        if notes is not None:
            entry["notes"] = notes.strip()
        if new_refs is not None:
            # Two-stage COPY-commit: stage the new set to _stage_NN BEFORE superseding the
            # old ref_* files, so re-submitting the identity's OWN owned paths (the UI reads
            # them back on edit) can't desync — a self-referential source is copied while it
            # still exists. _materialize_refs_staged supersedes then commits internally.
            # (Was: supersede-then-copy, which moved the source out from under the copy and
            # recorded every ref as missing → registry named ref_01..ref_11, disk had ref_00..)
            owned, missing = _materialize_refs_staged(_identity_dir(slug), new_refs)
            entry["source_images"] = owned
            if missing:
                entry["missing_references"] = missing
            else:
                entry.pop("missing_references", None)
        _write_profile_json(slug, entry)  # regenerate the mirror on every update
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def delete_profile(slug: str) -> Optional[dict[str, Any]]:
    """ARCHIVE (never erase) a profile. The registry entry moves under ``_deleted``
    keyed ``<slug>@<deleted_at>`` with a ``deleted_at`` stamp AND the identity's own
    dir is MOVED under ``_deleted/<slug>@<deleted_at>/`` — so a name can be reused,
    the history is preserved, and the reference pixels are relocated rather than
    erased (never-delete doctrine). Returns the archived public shape, or None if the
    slug was not an active profile (idempotent — deleting an unknown/already-archived
    slug is a clean no-op)."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].pop(slug, None)
        if entry is None:
            return None
        deleted_at = time.time()
        entry = dict(entry)
        entry["deleted_at"] = deleted_at
        data["_deleted"][f"{slug}@{deleted_at}"] = entry
        _save(data)
        # Relocate (never rm) the identity's owned bytes under _deleted/, correlated
        # with the registry archive key by the same deleted_at stamp. Best-effort.
        _archive_identity_dir(slug, deleted_at)
    return _public(slug, entry)


# --------------------------------------------------------------------------- #
# STAGE (b) — identity RECONSTRUCTION (turnaround generation) + canonical promote.
#
# The module docstring's stage (b): "turnaround generation from a profile + the
# re-edit loop that promotes an approved rendering to the canonical reference".
# These store functions own the DURABLE side of that loop; the actual rendering
# (the Wan VACE id_lock render + frame-extract, behind the runner's swap seam)
# hands the produced stills to ``attach_reconstruction`` on completion.
#
#   attach_reconstruction     — persist a generated turnaround set (its stills copied
#                               in, a manifest written, the entry's ``reconstructions``
#                               list appended) for approval.
#   promote_reconstruction_views — copy chosen recon views into the ``canonical`` ref
#                               set on the entry (the approved reference DNA).
#   list_reconstructions / get_reconstruction — UI read helpers.
#
# All follow the store's never-delete + atomic idioms: an existing recon bundle or
# canonical set is SUPERSEDED (moved aside), never overwritten; every copy + json
# write is atomic; the registry mutation is single-writer under ``_LOCK``.
# --------------------------------------------------------------------------- #
def _recon_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize ONE stored reconstruction record into the wire/manifest shape the UI
    reads. Additive + backward-compat only: an OLD record (attached before turntables)
    carries no ``mode`` -> defaulted to ``"sheet"`` so the shape is always self-describing.
    A turntable record's ``frame_count``/``degrees_per_frame`` (and the ordered ``views``
    holding the orbit frames in angular order) ride through untouched. Never drops or
    renames a key — the sheet path's shape is unchanged."""
    out = dict(record)
    if out.get("mode") not in ("sheet", "turntable"):
        out["mode"] = "sheet"
    return out


def attach_reconstruction(
    slug: str,
    recon_id: str,
    view_paths: list[str],
    *,
    spec: Optional[dict[str, Any]] = None,
    replace: bool = False,
) -> Optional[dict[str, Any]]:
    """Persist a generated turnaround set for *slug* under ``recon_id``.

    Creates ``<slug>/reconstruction/<recon_id>/views/`` and ``_atomic_copy``s each
    produced still (order preserved) into ``view_NN.png``, writes ``manifest.json``
    (recon_id, created_at, ordered views, plus the render provenance carried in
    *spec*: job_id / prompt / seed / per-view prompts / view names), and APPENDS the
    record to the entry's ``reconstructions`` list under ``_LOCK`` (regenerating
    ``profile.json``). Never overwrites an existing recon dir — it is SUPERSEDED
    (moved aside) first, so a re-run under a colliding id never erases prior pixels.

    A missing/unreadable produced still does NOT crash: it is skipped, its original
    path retained in the record, and recorded under ``missing_views`` (mirrors
    ``_materialize_refs``). Returns the stored reconstruction record, or ``None`` if
    the slug names no ACTIVE profile (the caller — a bus runner — surfaces that as a
    clean error-as-data; the profile may have been archived mid-render)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str) or not recon_id.strip():
        raise ProfileError("recon_id is required", code="invalid_profile")
    if not isinstance(view_paths, list) or not view_paths:
        raise ProfileError("view_paths must be a non-empty list of still paths",
                           code="invalid_profile")

    recon_dir = _reconstruction_dir(slug, recon_id)
    views_dir = os.path.join(recon_dir, "views")
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        # Never overwrite an existing recon bundle — relocate it (never-delete).
        _supersede_existing_dir(recon_dir)
        owned, missing = _materialize_views(views_dir, view_paths)
        record: dict[str, Any] = {
            "recon_id": recon_id,
            "created_at": time.time(),
            "views": owned,
        }
        if missing:
            record["missing_views"] = missing
        meta = spec if isinstance(spec, dict) else {}
        for key in ("job_id", "prompt", "seed", "prompts", "view_names",
                    # turntable additive provenance (sheet records omit these):
                    "frame_count", "degrees_per_frame", "orbit_prompt"):
            if key in meta:
                record[key] = meta[key]
        # MODE always stored so the manifest is self-describing; absent meta => "sheet"
        # (the existing N-independent-view-stills path). For a turntable, ``views`` holds
        # the orbit clip's frames in ANGULAR order and ``frame_count`` its length.
        mode = meta.get("mode")
        record["mode"] = mode if mode in ("sheet", "turntable") else "sheet"
        # manifest.json: the on-disk denormalized mirror of the record (same atomic write).
        _atomic_write_json(os.path.join(recon_dir, "manifest.json"), record)

        entry = dict(entry)  # edit a copy — nothing is written until _save
        recons = list(entry.get("reconstructions") or [])
        # replace=True (used by the mesh RELAY, which re-attaches under an EXISTING
        # recon_id): overwrite the first record sharing this recon_id IN PLACE so the
        # promote route — which finds a record by recon_id — sees THIS set, and the
        # list never grows a second same-id record. Default (append) is unchanged for
        # the reconstruction runner, which always mints a fresh recon_id.
        replaced = False
        if replace:
            for i, r in enumerate(recons):
                if isinstance(r, dict) and r.get("recon_id") == recon_id:
                    recons[i] = record
                    replaced = True
                    break
        if not replaced:
            recons.append(record)
        entry["reconstructions"] = recons
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return dict(record)


def promote_reconstruction_views(
    slug: str,
    recon_id: str,
    chosen_indices: list[int],
) -> Optional[dict[str, Any]]:
    """Promote chosen views of reconstruction ``recon_id`` into the entry's
    ``canonical`` reference set — the approved character DNA.

    Copies the views at ``chosen_indices`` (into ``<slug>/canonical/ref_NN.<ext>`` via
    ``_materialize_refs``, so a promoted set is renumbered from ``ref_00`` exactly like
    an uploaded set) and points the entry's ``canonical`` key at those identity-owned
    copies. Any existing canonical set is SUPERSEDED first (never-delete). At most
    ``MAX_source_images`` may be promoted (canonical feeds the id_lock reference
    channel, capped at 4). ``profile.json`` is regenerated.

    Errors-as-data on the store contract (``ProfileError`` -> the route's 4xx): a bad
    index shape, an out-of-range index, an unknown ``recon_id``, or too many views.
    Returns the updated public profile shape (its ``canonical`` now populated), or
    ``None`` if the slug names no ACTIVE profile (the route's 404)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str) or not recon_id.strip():
        raise ProfileError("recon_id is required", code="invalid_profile")
    if not isinstance(chosen_indices, list) or not chosen_indices:
        raise ProfileError("views must be a non-empty list of view indices",
                           code="invalid_profile")
    idxs: list[int] = []
    for raw in chosen_indices:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ProfileError("each view index must be a non-negative int",
                               code="invalid_profile")
        idxs.append(raw)
    if len(idxs) > MAX_CANONICAL_IMAGES:
        raise ProfileError(
            f"at most {MAX_CANONICAL_IMAGES} views may be promoted to canonical",
            code="invalid_profile")

    canon_dir = _canonical_dir(slug)
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        record = next(
            (r for r in (entry.get("reconstructions") or [])
             if isinstance(r, dict) and r.get("recon_id") == recon_id),
            None,
        )
        if record is None:
            raise ProfileError(f"reconstruction {recon_id!r} not found",
                               code="invalid_profile")
        views = list(record.get("views") or [])
        chosen_sources: list[str] = []
        for i in idxs:
            if i >= len(views):
                raise ProfileError(
                    f"view index {i} is out of range (this reconstruction has "
                    f"{len(views)} views)", code="invalid_profile")
            chosen_sources.append(views[i])
        # Never overwrite an existing canonical set — relocate it (never-delete).
        _supersede_existing_dir(canon_dir)
        owned, missing = _materialize_refs(canon_dir, chosen_sources)
        entry = dict(entry)  # edit a copy — nothing is written until _save
        entry["canonical"] = owned
        if missing:
            entry["canonical_missing"] = missing
        else:
            entry.pop("canonical_missing", None)
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def list_reconstructions(slug: str) -> Optional[list[dict[str, Any]]]:
    """The ordered list of generated turnaround sets for *slug* (newest last, as
    attached), or ``None`` if the slug names no active profile."""
    if not slug or not isinstance(slug, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
    if entry is None:
        return None
    return [_recon_manifest(r) for r in (entry.get("reconstructions") or [])
            if isinstance(r, dict)]


def get_reconstruction(slug: str, recon_id: str) -> Optional[dict[str, Any]]:
    """One reconstruction record by id, or ``None`` if the slug/recon_id is unknown."""
    recons = list_reconstructions(slug)
    if recons is None:
        return None
    return next((r for r in recons if r.get("recon_id") == recon_id), None)


def update_reconstruction_view_status(slug: str, recon_id: str, view_id: str, status: str) -> dict:
    """Updates the explicit approval status of a single angle tile."""
    if status not in ("approved", "rejected"):
        raise ProfileError("validation", "Status must be 'approved' or 'rejected'")
        
    profile = get_profile(slug)
    if not profile:
        raise ProfileError("not_found", f"Profile {slug} not found")

    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    if not target_recon:
        raise ProfileError("not_found", f"Reconstruction {recon_id} not found")

    views = target_recon.get("views", [])
    target_view = next((v for v in views if isinstance(v, dict) and v.get("viewId") == view_id), None)
    if not target_view:
        raise ProfileError("not_found", f"View {view_id} not found")

    target_view["status"] = status
    
    # Save the updated profile
    _save_profile(slug, profile)
    return profile

def make_single_view_regeneration_spec(slug: str, recon_id: str, view_id: str, prompt: str, seed: int, use_neighbors: bool) -> IdentitySingleViewRegenSpec:
    """Prepares the regeneration spec, extracting nearest approved neighbors for conditioning."""
    profile = get_profile(slug)
    if not profile:
        raise ProfileError("not_found", f"Profile {slug} not found")
        
    # (In a full implementation, you would scan target_recon["views"] to find 
    # the closest azimuthDeg neighbors with status == "approved" and extract their imageUris)
    neighbor_uris = [] 
    
    return IdentitySingleViewRegenSpec(
        slug=slug,
        recon_id=recon_id,
        view_id=view_id,
        prompt=prompt,
        seed=seed,
        use_neighbors=use_neighbors,
        neighbor_images=tuple(neighbor_uris)
    )

def make_mesh_reconstruction_spec(slug: str, recon_id: str, view_ids: list, backend: str, workflow: str, output_format: str) -> IdentityMeshSpec:
    """Locks the active views and queues the mesh build job."""
    profile = get_profile(slug)
    if not profile:
        raise ProfileError("not_found", f"Profile {slug} not found")
        
    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    if not target_recon:
        raise ProfileError("not_found", f"Reconstruction {recon_id} not found")

    # Initialize the mesh block to "queued" so the UI disables the build button
    if "mesh" not in target_recon:
        target_recon["mesh"] = {}
    target_recon["mesh"]["status"] = "queued"
    target_recon["mesh"]["error"] = None
    
    _save_profile(slug, profile)
    
    return IdentityMeshSpec(
        slug=slug,
        recon_id=recon_id,
        view_ids=tuple(view_ids),
        backend=backend,
        workflow=workflow,
        output_format=output_format
    )

def get_mesh_state(slug: str, recon_id: str) -> dict:
    """Read-only fetch of the mesh build status."""
    profile = get_profile(slug)
    if not profile:
        return None

    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    if not target_recon:
        return None

    return target_recon.get("mesh")


def set_mesh_state(slug: str, recon_id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Merge *patch* into reconstruction ``recon_id``'s ``mesh`` state block (created if
    absent) and persist. This is the DURABLE WRITE side of the mesh-build lifecycle that
    ``get_mesh_state`` reads: the route seeds ``{"status": "queued"}`` on enqueue, and the
    relay runner records ``{"status": "running"}`` → ``{"status": "done", "glb_path": …,
    "video_path": …, "mesh_json_path": …}`` (or ``{"status": "error", "error": …}``).

    Uses the store's REAL single-writer + atomic idiom (``_LOCK`` / ``_load`` / ``_save`` /
    ``_write_profile_json``) — NOT the unwired ``_save_profile`` stub the older
    ``make_mesh_reconstruction_spec`` references. Returns the updated ``mesh`` dict, or
    ``None`` if the slug/recon_id names no active reconstruction (a best-effort caller — a
    bus runner — tolerates that; the profile may have been archived or the reconstruction
    not yet attached)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str):
        return None
    if not isinstance(patch, dict):
        raise ProfileError("mesh state patch must be a dict", code="invalid_profile")
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        recons = list(entry.get("reconstructions") or [])
        target_idx = next(
            (i for i, r in enumerate(recons)
             if isinstance(r, dict) and r.get("recon_id") == recon_id),
            None,
        )
        if target_idx is None:
            # MESH-FIRST flow (v0 photos→mesh): there IS no prior reconstruction to hang
            # the mesh state on — the old silent ``return None`` made a successful build
            # invisible to GET .../mesh even though the GLB persisted (keeper 2026-07-14,
            # first live build). Create a minimal record in attach_reconstruction's shape
            # (views empty until a turntable attach replaces it) instead of no-opping.
            recons.append({
                "recon_id": recon_id,
                "created_at": time.time(),
                "views": [],
                "mode": "mesh",
            })
            target_idx = len(recons) - 1
        record = dict(recons[target_idx])
        mesh = dict(record.get("mesh") or {})
        mesh.update(patch)
        record["mesh"] = mesh
        recons[target_idx] = record
        entry = dict(entry)  # edit a copy — nothing is written until _save
        entry["reconstructions"] = recons
        _write_profile_json(slug, entry)  # regenerate the mirror
        data["profiles"][slug] = entry
        _save(data)
    return mesh


# --------------------------------------------------------------------------- #
# VERSIONS slice — append-only version list + active pointer + gen_settings.
#
# The mesh RELAY (runners/identity_render_relay.py) owns the mint moment: on a
# successful build it calls ``mint_version`` with the auto-promoted cardinals as the
# version's canonical, and the new version becomes ACTIVE (latest-wins). The routes
# own activate / rename / archive + the Settings tab (``set_gen_settings``). Every
# mutation is single-writer under ``_LOCK`` with the store's atomic-write idiom, and
# never-delete holds — an archived version is flagged, never dropped from storage.
# --------------------------------------------------------------------------- #
def _auto_version_name(versions: list, kind: str) -> tuple[str, bool]:
    """The auto-name for a freshly minted version + whether it is the BASE.

    The FIRST ``clay`` version ever minted for a profile is the base (name ``"base"``,
    the geometric ground truth). Every other version is ``"<kind>-NN"`` where NN counts
    ALL prior versions of that kind (archived included, so a name is never reused).
    Returns ``(name, is_base)``."""
    has_base = any(isinstance(v, dict) and v.get("base") for v in versions)
    if kind == "clay" and not has_base:
        return "base", True
    nn = 1 + sum(1 for v in versions if isinstance(v, dict) and v.get("kind") == kind)
    return f"{kind}-{nn:02d}", False


def mint_version(
    slug: str,
    recon_id: str,
    kind: str,
    canonical: list[str],
    name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Mint (or, on a re-run of the SAME ``recon_id``, update in place) a VERSION and
    make it ACTIVE (latest-wins).

    ``kind`` is one of ``VERSION_KINDS`` (``clay``/``textured``/``styled``). ``canonical``
    is the version's own promoted cardinal set (the id_lock DNA a caller gets when this
    version is active). When ``name`` is ``None`` an auto-name is derived
    (``_auto_version_name``) — the first clay is pinned as the ``base``. Dedupe by
    ``recon_id``: a version already carrying this recon_id (e.g. a bus retry) is updated
    in place rather than duplicated, preserving its ``version_id``/``name``/base flag.

    Append-only + never-delete: an existing version is never removed here. Returns the
    minted/updated version's PUBLIC shape, or ``None`` if the slug names no active
    profile (a best-effort caller — the relay — tolerates that)."""
    if not slug or not isinstance(slug, str):
        return None
    if not recon_id or not isinstance(recon_id, str) or not recon_id.strip():
        raise ProfileError("recon_id is required", code="invalid_profile")
    if kind not in VERSION_KINDS:
        raise ProfileError(f"kind must be one of {list(VERSION_KINDS)}", code="invalid_profile")
    canon = [p for p in (canonical or []) if isinstance(p, str)]
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        entry = dict(entry)  # edit a copy — nothing is written until _save
        versions = list(entry.get("versions") or [])
        existing_idx = next(
            (i for i, v in enumerate(versions)
             if isinstance(v, dict) and v.get("recon_id") == recon_id and not v.get("archived")),
            None,
        )
        if existing_idx is not None:
            version = dict(versions[existing_idx])
            version["kind"] = kind
            version["canonical"] = canon
            if name is not None:
                version["name"] = name
            versions[existing_idx] = version
        else:
            auto_name, is_base = _auto_version_name(versions, kind)
            version = {
                "version_id": "ver_" + secrets.token_hex(8),
                "name": name if name is not None else auto_name,
                "kind": kind,
                "recon_id": recon_id,
                "created_at": time.time(),
                "canonical": canon,
                "notes": "",
            }
            if is_base and name is None:
                version["base"] = True  # the geometric ground truth — never archivable
            versions.append(version)
        entry["versions"] = versions
        entry["active_version"] = version["version_id"]  # latest-wins
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public_version(version)


def set_active_version(slug: str, version_id: str) -> Optional[dict[str, Any]]:
    """Point the identity's ACTIVE version at ``version_id`` (the id_lock DNA source).
    Returns the updated PUBLIC profile, or ``None`` if the slug names no active profile
    OR ``version_id`` names no active (non-archived) version of it — the route turns
    either into a 404."""
    if not slug or not isinstance(slug, str) or not version_id or not isinstance(version_id, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        versions = list(entry.get("versions") or [])
        match = next(
            (v for v in versions
             if isinstance(v, dict) and v.get("version_id") == version_id and not v.get("archived")),
            None,
        )
        if match is None:
            return None
        entry = dict(entry)
        entry["active_version"] = version_id
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def update_version(
    slug: str,
    version_id: str,
    *,
    name: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Partial edit of a version's DISPLAY fields (``name`` / ``notes``) in place — an
    argument left at ``None`` is a no-op for that field (the route only forwards keys
    actually present). Returns the updated PUBLIC profile, or ``None`` if the slug/version
    is unknown or archived (route 404). A blank ``name`` is a ProfileError -> the route's
    400."""
    if not slug or not isinstance(slug, str) or not version_id or not isinstance(version_id, str):
        return None
    new_name: Optional[str] = None
    if name is not None:
        new_name = name.strip()
        if not new_name:
            raise ProfileError("name is required", code="invalid_profile")
    new_notes: Optional[str] = None
    if notes is not None:
        if not isinstance(notes, str):
            raise ProfileError("notes must be a string", code="invalid_profile")
        new_notes = notes.strip()
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        versions = list(entry.get("versions") or [])
        idx = next(
            (i for i, v in enumerate(versions)
             if isinstance(v, dict) and v.get("version_id") == version_id and not v.get("archived")),
            None,
        )
        if idx is None:
            return None
        version = dict(versions[idx])
        if new_name is not None:
            version["name"] = new_name
        if new_notes is not None:
            version["notes"] = new_notes
        versions[idx] = version
        entry = dict(entry)
        entry["versions"] = versions
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def archive_version(slug: str, version_id: str) -> Optional[dict[str, Any]]:
    """ARCHIVE a version (never-delete: flag it ``archived``, keep its bytes, drop it
    from the wire list). REFUSED for the clay BASE (the geometric ground truth) and for
    the currently ACTIVE version — either is a ProfileError -> the route's 400 with a
    clear message. Returns the updated PUBLIC profile, or ``None`` if the slug/version is
    unknown / already archived (route 404)."""
    if not slug or not isinstance(slug, str) or not version_id or not isinstance(version_id, str):
        return None
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        versions = list(entry.get("versions") or [])
        idx = next(
            (i for i, v in enumerate(versions)
             if isinstance(v, dict) and v.get("version_id") == version_id and not v.get("archived")),
            None,
        )
        if idx is None:
            return None
        if versions[idx].get("base"):
            raise ProfileError(
                "the clay base version cannot be archived (it is the identity's geometric "
                "ground truth)", code="invalid_profile")
        if entry.get("active_version") == version_id:
            raise ProfileError(
                "the active version cannot be archived — activate another version first",
                code="invalid_profile")
        version = dict(versions[idx])
        version["archived"] = True
        version["archived_at"] = time.time()
        versions[idx] = version
        entry = dict(entry)
        entry["versions"] = versions
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)


def set_gen_settings(slug: str, partial: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Merge a PARTIAL gen_settings update into the identity's stored settings. Only the
    contract's known keys are accepted (an unknown key is a ProfileError -> the route's
    400); every value is type-checked, ``pose`` is enum-checked, and ``front_ref`` (when
    non-null) is JAILED to the profile's OWN reference images. Returns the updated PUBLIC
    profile (its ``gen_settings`` reflecting the merge), or ``None`` if the slug names no
    active profile (route 404)."""
    if not isinstance(partial, dict):
        raise ProfileError("gen_settings must be an object", code="invalid_profile")
    with _LOCK:
        data = _load()
        entry = data["profiles"].get(slug)
        if entry is None:
            return None
        own = {p for p in (entry.get("source_images") or []) if isinstance(p, str)}
        cleaned: dict[str, Any] = {}
        for k, v in partial.items():
            if k not in _DEFAULT_GEN_SETTINGS:
                raise ProfileError(f"unknown gen_settings key {k!r}", code="invalid_profile")
            if k in ("texture", "auto_promote", "remove_background"):
                if not isinstance(v, bool):
                    raise ProfileError(f"{k} must be a bool", code="invalid_profile")
            elif k == "pose":
                if v not in _POSE_CHOICES:
                    raise ProfileError(
                        f"pose must be one of {list(_POSE_CHOICES)}", code="invalid_profile")
            elif k in ("frame_count", "fps", "width", "height"):
                if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                    raise ProfileError(f"{k} must be a positive int", code="invalid_profile")
            elif k == "front_ref":
                if v is not None and (not isinstance(v, str) or v not in own):
                    raise ProfileError(
                        "front_ref must be one of the profile's own reference images",
                        code="invalid_profile")
            cleaned[k] = v
        entry = dict(entry)
        gs = dict(entry.get("gen_settings") or {})
        gs.update(cleaned)
        entry["gen_settings"] = gs
        _write_profile_json(slug, entry)
        data["profiles"][slug] = entry
        _save(data)
    return _public(slug, entry)
