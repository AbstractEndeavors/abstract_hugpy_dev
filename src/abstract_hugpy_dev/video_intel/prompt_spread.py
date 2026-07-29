"""STUDIO SPREAD — the whole-movie prompt generator (STUDIO-SPREAD-SPEC §1).

WHAT A SPREAD IS
----------------
ONE generator call that writes (or rewrites) the prompts for an ENTIRE movie at
once, holding the rows the user did not select as LOCKED CONTEXT. It is not a
convenience wrapper around N single-row calls — it is the opposite of that, and
the difference is the whole feature:

    per-row Generate (today)      spread (here)
    ------------------------      ---------------------------------
    N calls                       1 call
    N independent steering sets   ONE shared steering set
    each row blind to the others  every row sees the full timeline
    six rows -> six worlds        six rows -> six shots of one film

``prompt_seeds.steering_axes`` randomizes per call by design (a single Generate
click must not return the same prompt twice). Run that N times for a movie and
coherence is not merely absent, it is actively destroyed. So a spread draws one
``spread_axes`` set for the whole movie and varies only the BEAT per segment
(``prompt_seeds.beat_for_index``) — the inverse of the per-call randomization.

WHY THE MODEL IS TOLD THE JOINT MODES IN PLAIN LANGUAGE
-------------------------------------------------------
``joint_mode`` is a RENDER-TIME fact with dramatic consequences the writer must
respect: a ``still`` join carries a single frame and NO motion, so a segment that
opens mid-sprint will stutter; a ``vace_extend`` join carries the previous shot's
motion, so the segment must continue that movement rather than restart it; a
``cut`` carries nothing at all, so the segment is free to relocate. A model shown
the raw token ``vace_extend`` guesses. A model shown the sentence does not.

⚠ NOMENCLATURE NOTE (keeper, 2026-07-29): the spec's illustrative gloss for
``vace_extend`` — *"continues from one frame of the previous shot; motion is not
carried"* — actually describes ``still``. ``studio_movie_schema`` is explicit
that ``vace_extend`` exists precisely TO carry motion across the splice (it
conditions on the parent's trailing ``context_frames``). The plain language below
follows the SCHEMA, because a preface that misdescribes the join would teach the
model to write exactly the wrong continuation. The spec's INTENT — plain language,
never a raw token — is honoured in full.

OUTPUT DISCIPLINE (§1e)
-----------------------
Constrained JSON out, parsed by the package scavenger
(``utils/json_scavenge``) after ``<think>`` is stripped
(``utils/no_think``). A reply that will not parse is an HONEST 502 carrying the
raw (think-stripped) text — never an invented segment. And a segment the model
returns for a row that was NOT selected is DROPPED, not applied: a locked row is
the user's work, and a generator that could overwrite it would make selection a
lie.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .prompt_seeds import beat_for_index, spread_axes, spread_steering_clause

__all__ = [
    "SpreadError",
    "SpreadParseError",
    "JOINT_MODE_PLAIN",
    "VALID_JOINT_MODES",
    "VALID_OPERATIONS",
    "SPREAD_SYSTEM",
    "NEGATIVE_SYSTEM",
    "SpreadRequest",
    "validate_context",
    "render_context_preface",
    "render_identity_block",
    "build_spread_request",
    "build_spread_messages",
    "build_negative_messages",
    "parse_spread_reply",
]


class SpreadError(ValueError):
    """A structurally invalid request — the caller's to fix (maps to 400)."""


class SpreadParseError(ValueError):
    """The model's reply could not be read as the contracted JSON (maps to 502).

    Carries the raw (think-stripped) text so the failure is diagnosable and the
    caller can be told the truth instead of handed fabricated segments.
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw or ""


# Mirrors studio_movie_schema._VALID_JOINT_MODES exactly — keep in sync.
VALID_JOINT_MODES = ("still", "vace_extend", "cut")

#: Joint mode -> the sentence the MODEL sees. Never the raw token.
JOINT_MODE_PLAIN: Dict[str, str] = {
    "still": ("this shot begins from a single still frame of the previous shot; "
              "no motion is carried across the join, so it must start from rest "
              "rather than mid-movement"),
    "vace_extend": ("this shot continues the previous shot, carrying its motion "
                    "across the join; it must continue that movement rather than "
                    "restart or reverse it"),
    "cut": ("this shot is a hard cut — nothing is carried over from the previous "
            "shot, so it may change location or framing freely, but the subject "
            "and the film's world must stay the same"),
}

#: Operations a spread may return for a row. ``keep`` is how a model declines to
#: change a row it was asked about — accepted so it never has to invent instead.
VALID_OPERATIONS = ("generate_from_direction", "enhance_scene", "generate", "keep")

_DEFAULT_DO_NOT_INVENT = ("age", "gender", "clothing", "ethnicity")


# --------------------------------------------------------------------------- #
# SYSTEM PROMPTS
# --------------------------------------------------------------------------- #
SPREAD_SYSTEM = (
    "You are a film director writing the shot prompts for ONE short film that "
    "will be rendered by a text-to-video model.\n"
    "\n"
    "RULES:\n"
    "1. Every shot belongs to the SAME film — same subject, same world, same "
    "visual style. Continuity is the point.\n"
    "2. Rewrite ONLY the segments marked REGENERATE. Segments marked LOCKED are "
    "the user's own work: use them as context, never restate or replace them.\n"
    "3. Write one chronological cinematic paragraph per segment. Plain "
    "renderable description: subject, action, camera, light. No headings, no "
    "markdown, no lists, no reasoning, no commentary.\n"
    "4. Incorporate EVERY direction given for a segment.\n"
    "5. NEVER invent identity attributes (age, gender, clothing, ethnicity, "
    "hair, build) for a named character. If it is not in the locked identity "
    "data, do not state it.\n"
    "6. Respect each segment's join description — it says what is carried over "
    "from the shot before it.\n"
    "\n"
    "Return ONLY a JSON object, no fence and no prose around it:\n"
    '{"segments": [{"segment_id": "...", "operation": '
    '"generate_from_direction|enhance_scene|generate|keep", "prompt": "...", '
    '"negative": "...", "continuity_note": "...", "directions_used": [0], '
    '"warnings": []}], "invented_identity_attributes": [], "warnings": []}\n'
    "One entry per REGENERATE segment and nothing else. "
    '"continuity_note" is one short sentence on how the shot connects to its '
    "neighbours."
)

NEGATIVE_SYSTEM = (
    "You write NEGATIVE prompts for image and video generation models.\n"
    "\n"
    "A negative prompt is an EXCLUSION LIST of artifacts and quality failures — "
    "it is not prose, not a scene, and not a description of what should happen. "
    "Never describe the shot; only name what must not appear in it.\n"
    "\n"
    "RULES:\n"
    "1. Output short comma-separated phrases, lowercase, no sentences.\n"
    "2. Name rendering artifacts and quality failures: deformed hands, extra "
    "fingers, extra limbs, fused faces, warped anatomy, flicker, temporal "
    "instability, morphing, blur, low resolution, jpeg artifacts, watermark, "
    "text overlay, logo, oversaturation, banding.\n"
    "3. Add exclusions specific to what is being generated when the shot makes "
    "them relevant (e.g. duplicate subject, changing wardrobe, identity drift, "
    "camera shake).\n"
    "4. Never repeat a phrase. No prose, no headings, no explanation, no "
    "reasoning, no markdown.\n"
    "\n"
    "Return ONLY the comma-separated list."
)


# --------------------------------------------------------------------------- #
# TYPED CONTEXT (§1c) — hint stays free-form; structured row state is TYPED.
# --------------------------------------------------------------------------- #
def _validate_segment_ref(value: Any, label: str) -> Optional[Dict[str, Any]]:
    """Validate one typed segment reference. Field names mirror
    ``studio_movie_schema.StudioMovieGoal`` EXACTLY (prompt / seed /
    branch_frame / joint_mode / negative) so the UI can hand a row straight
    through without a translation layer to get wrong."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SpreadError(f"context.{label} must be an object")
    out: Dict[str, Any] = {}
    for key in ("segment_id", "prompt", "negative", "direction"):
        v = value.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            raise SpreadError(f"context.{label}.{key} must be a string")
        v = v.strip()
        if v:
            out[key] = v
    jm = value.get("joint_mode")
    if jm is not None:
        if jm not in VALID_JOINT_MODES:
            raise SpreadError(
                f"context.{label}.joint_mode must be one of "
                + "|".join(VALID_JOINT_MODES))
        out["joint_mode"] = jm
    for key in ("branch_frame", "seed", "index"):
        v = value.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            raise SpreadError(f"context.{label}.{key} must be an integer")
        if v < 0:
            raise SpreadError(f"context.{label}.{key} must be >= 0")
        out[key] = v
    return out


def _validate_identity(value: Any, label: str = "context.identity_profile"):
    """Validate a LOCKED identity block (§1e).

    Accepts the wire shape of a stored identity profile
    (``/video/identity-profiles`` -> ``{slug, name, reference_images, notes,
    ...}``) as well as the spec's ``{identity_id, name, reference_asset_ids,
    locked_description, do_not_invent}``, because the UI has the profile row in
    hand and should not have to rebuild it. Unknown extra keys are IGNORED
    rather than rejected — a profile row carries reconstructions/versions the
    generator has no use for.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [_validate_identity(v, label) for v in value if v is not None]
    if not isinstance(value, dict):
        raise SpreadError(f"{label} must be an object")
    ident = value.get("identity_id") or value.get("slug") or value.get("id")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpreadError(f"{label}.name is required")
    refs = value.get("reference_asset_ids")
    if refs is None:
        refs = value.get("reference_images")
    if refs is not None and not isinstance(refs, list):
        raise SpreadError(f"{label}.reference_asset_ids must be a list")
    locked = (value.get("locked_description") or value.get("notes") or "")
    if not isinstance(locked, str):
        raise SpreadError(f"{label}.locked_description must be a string")
    dni = value.get("do_not_invent")
    if dni is None:
        dni = list(_DEFAULT_DO_NOT_INVENT)
    if not isinstance(dni, list) or any(not isinstance(x, str) for x in dni):
        raise SpreadError(f"{label}.do_not_invent must be a list of strings")
    return {
        "identity_id": ident if isinstance(ident, str) else None,
        "name": name.strip(),
        "reference_asset_ids": [str(r) for r in (refs or [])],
        "locked_description": locked.strip(),
        "do_not_invent": list(dni),
    }


def validate_context(context: Any) -> Dict[str, Any]:
    """Validate the typed ``context`` block; returns a normalized copy.

    ``hint`` stays FREE-FORM and untouched (§1c ruling: do NOT make hint the
    carrier for structured row state). Everything structured is typed and
    validated here, so a malformed row is a clean 400 instead of garbage
    silently rendered into the model preface.
    """
    if context is None:
        return {}
    if not isinstance(context, dict):
        raise SpreadError("context must be an object")
    out: Dict[str, Any] = {}
    for label in ("segment", "previous_segment", "next_segment"):
        ref = _validate_segment_ref(context.get(label), label)
        if ref:
            out[label] = ref
    ident = _validate_identity(context.get("identity_profile"))
    if ident:
        out["identity_profile"] = ident
    return out


def render_identity_block(identity) -> str:
    """The LOCKED identity preface (§1e).

    Both benchmarked models invented ages and wardrobes when handed a bare
    character NAME, so the name is never sent alone: it always arrives with what
    is known and an explicit list of what must not be filled in.
    """
    if not identity:
        return ""
    people = identity if isinstance(identity, list) else [identity]
    lines = ["LOCKED IDENTITY — these people already exist. Use them as given."]
    for p in people:
        if not p:
            continue
        bits = [f'- {p.get("name")}']
        if p.get("identity_id"):
            bits.append(f'(identity {p["identity_id"]})')
        lines.append(" ".join(bits))
        if p.get("locked_description"):
            lines.append(f'    known: {p["locked_description"]}')
        refs = p.get("reference_asset_ids") or []
        if refs:
            lines.append(f"    reference images on file: {len(refs)}")
        dni = p.get("do_not_invent") or []
        if dni:
            lines.append("    DO NOT INVENT OR STATE: " + ", ".join(dni)
                         + " — if it is not written above, do not describe it")
    return "\n".join(lines)


def render_context_preface(ctx: Dict[str, Any]) -> str:
    """Render the validated typed context into the model preface.

    Joint modes become SENTENCES (see the module docstring) and a branch frame
    becomes "conditioned on frame N of the previous shot" — the model has no
    idea what ``branch_frame: 37`` means, and inventing a meaning for it is
    worse than not sending it.
    """
    if not ctx:
        return ""
    blocks: List[str] = []
    labels = (
        ("previous_segment", "THE SHOT BEFORE THIS ONE"),
        ("segment", "THIS SHOT (current state)"),
        ("next_segment", "THE SHOT AFTER THIS ONE"),
    )
    for key, title in labels:
        ref = ctx.get(key)
        if not ref:
            continue
        lines = [title + ":"]
        if ref.get("prompt"):
            lines.append(f'    prompt: {ref["prompt"]}')
        if ref.get("negative"):
            lines.append(f'    negative: {ref["negative"]}')
        if ref.get("direction"):
            lines.append(f'    direction from the user: {ref["direction"]}')
        jm = ref.get("joint_mode")
        if jm:
            lines.append(f"    join: {JOINT_MODE_PLAIN[jm]}")
        if ref.get("branch_frame") is not None and jm != "cut":
            lines.append(
                f'    it starts from frame {ref["branch_frame"]} of the shot '
                "before it, not from that shot's end")
        blocks.append("\n".join(lines))
    ident = render_identity_block(ctx.get("identity_profile"))
    if ident:
        blocks.append(ident)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# SPREAD REQUEST (§1a)
# --------------------------------------------------------------------------- #
_STYLE_BIBLE_KEYS = ("world", "subject", "visual_style",
                     "camera_language", "color_language")


@dataclass(frozen=True)
class SpreadRequest:
    movie_query: str
    style_bible: Dict[str, str]
    fixed_segments: Tuple[Dict[str, Any], ...]
    target_segments: Tuple[Dict[str, Any], ...]
    global_negative: Tuple[str, ...]
    steering_seed: Optional[int]
    steering: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    hint: str = ""
    kind: str = "movie"

    @property
    def target_ids(self) -> Tuple[str, ...]:
        return tuple(s["segment_id"] for s in self.target_segments)


def _validate_segment_list(value: Any, label: str, locked: bool) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SpreadError(f"{label} must be a list")
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise SpreadError(f"{label}[{i}] must be an object")
        sid = raw.get("segment_id")
        if not isinstance(sid, str) or not sid.strip():
            raise SpreadError(f"{label}[{i}].segment_id is required")
        ref = _validate_segment_ref(raw, f"{label}[{i}]") or {}
        ref["segment_id"] = sid.strip()
        ref["locked"] = locked
        # Timeline position: an explicit index wins (rows can arrive in any
        # order once a UI mixes selected + unselected); list order backs it up.
        ref.setdefault("index", i if locked else i)
        out.append(ref)
    return out


def build_spread_request(body: Dict[str, Any]) -> SpreadRequest:
    """Validate a ``mode="spread"`` body into a SpreadRequest. Raises SpreadError."""
    movie_query = body.get("movie_query")
    if movie_query is not None and not isinstance(movie_query, str):
        raise SpreadError("movie_query must be a string")
    movie_query = (movie_query or "").strip()

    sb_raw = body.get("style_bible") or {}
    if not isinstance(sb_raw, dict):
        raise SpreadError("style_bible must be an object")
    style_bible = {}
    for k in _STYLE_BIBLE_KEYS:
        v = sb_raw.get(k)
        if v is None:
            continue
        if not isinstance(v, str):
            raise SpreadError(f"style_bible.{k} must be a string")
        if v.strip():
            style_bible[k] = v.strip()

    fixed = _validate_segment_list(body.get("fixed_segments"), "fixed_segments", True)
    targets = _validate_segment_list(body.get("target_segments"), "target_segments", False)
    if not targets:
        raise SpreadError('target_segments must contain at least one segment '
                          'for mode "spread"')
    seen = set()
    for s in list(fixed) + list(targets):
        if s["segment_id"] in seen:
            raise SpreadError(f'duplicate segment_id {s["segment_id"]!r}')
        seen.add(s["segment_id"])

    gn = body.get("global_negative")
    if gn is None:
        global_negative: Tuple[str, ...] = ()
    elif isinstance(gn, str):
        global_negative = tuple(p.strip() for p in gn.split(",") if p.strip())
    elif isinstance(gn, list):
        if any(not isinstance(x, str) for x in gn):
            raise SpreadError("global_negative must be a list of strings")
        global_negative = tuple(x.strip() for x in gn if x.strip())
    else:
        raise SpreadError("global_negative must be a list of strings or a string")

    seed = body.get("steering_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise SpreadError("steering_seed must be an integer")

    ctx_raw = body.get("context") or {}
    if not isinstance(ctx_raw, dict):
        raise SpreadError("context must be an object")
    hint = ctx_raw.get("hint")
    if hint is not None and not isinstance(hint, str):
        raise SpreadError("context.hint must be a string")
    kind = ctx_raw.get("kind") or "movie"
    ctx = validate_context(ctx_raw)

    # ONE steering set for the whole spread — the coherence mechanism (§1a).
    axes = spread_axes(kind, seed=seed)

    return SpreadRequest(
        movie_query=movie_query,
        style_bible=style_bible,
        fixed_segments=tuple(fixed),
        target_segments=tuple(targets),
        global_negative=global_negative,
        steering_seed=seed,
        steering=axes,
        context=ctx,
        hint=(hint or "").strip(),
        kind=kind,
    )


def _render_timeline(req: SpreadRequest) -> str:
    """Every segment, in TIMELINE ORDER, locked ones included.

    The locked rows are what make this a spread rather than N rolls: the model
    can only write a segment that continues from its actual neighbour if it can
    SEE that neighbour, even when that neighbour is not being rewritten.
    """
    rows = sorted(list(req.fixed_segments) + list(req.target_segments),
                  key=lambda s: (s.get("index", 0), s["segment_id"]))
    total = len(rows)
    out: List[str] = []
    for position, s in enumerate(rows):
        mark = "LOCKED — do not rewrite" if s.get("locked") else "REGENERATE"
        lines = [f'SEGMENT {position + 1} of {total} [{s["segment_id"]}] — {mark}']
        if s.get("prompt"):
            lines.append(f'    current prompt: {s["prompt"]}')
        elif not s.get("locked"):
            lines.append("    current prompt: (empty — write it)")
        if s.get("direction"):
            lines.append(f'    the user asks for: {s["direction"]}')
        if s.get("negative"):
            lines.append(f'    negative: {s["negative"]}')
        jm = s.get("joint_mode")
        if position > 0 and jm:
            lines.append(f"    join: {JOINT_MODE_PLAIN[jm]}")
        if position > 0 and s.get("branch_frame") is not None and jm != "cut":
            lines.append(
                f'    it starts from frame {s["branch_frame"]} of the previous '
                "shot, not from that shot's end")
        if not s.get("locked"):
            lines.append(f"    beat for this shot: {beat_for_index(position, total)}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def build_spread_messages(req: SpreadRequest) -> List[Dict[str, str]]:
    """The ONE generator call's messages. Never called per segment."""
    parts: List[str] = []
    if req.movie_query:
        parts.append(f"THE FILM THE USER ASKED FOR:\n{req.movie_query}")
    if req.style_bible:
        parts.append("STYLE BIBLE (binding for every shot):\n" + "\n".join(
            f"- {k.replace('_', ' ')}: {v}" for k, v in req.style_bible.items()))
    parts.append(spread_steering_clause(req.steering))
    ident = render_identity_block(req.context.get("identity_profile"))
    if ident:
        parts.append(ident)
    if req.global_negative:
        parts.append("EXCLUDE FROM EVERY SHOT (the movie's negative prompt):\n"
                     + ", ".join(req.global_negative))
    parts.append("THE TIMELINE:\n\n" + _render_timeline(req))
    if req.hint:
        parts.append(f"Additional context to honour: {req.hint}")
    ids = ", ".join(req.target_ids)
    parts.append(
        f"Write ONLY these {len(req.target_ids)} segments: {ids}. "
        "Return the JSON object described in your instructions and nothing else."
    )
    return [
        {"role": "system", "content": SPREAD_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_negative_messages(draft: str, ctx: Dict[str, Any], hint: str = "",
                            subject: str = "") -> List[Dict[str, str]]:
    """Messages for ``mode="negative"`` (§1b).

    Deliberately its OWN framing. Reusing the scene-prose system prompt here
    produces a poem: the assist system prompt asks for a vivid description, and a
    negative asked for vividly is a description of the thing you were trying to
    exclude.
    """
    parts: List[str] = []
    if subject:
        parts.append(f"The shot being generated:\n{subject}")
    preface = render_context_preface(ctx)
    if preface:
        parts.append(preface)
    if draft:
        parts.append("The existing negative prompt — keep what is useful, remove "
                     "duplicates, and extend it:\n" + draft)
    if hint:
        parts.append(f"Additional context to honour: {hint}")
    parts.append("Write the negative prompt as comma-separated phrases only.")
    return [
        {"role": "system", "content": NEGATIVE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# --------------------------------------------------------------------------- #
# PARSING THE REPLY — honest failure, and locked rows are UNTOUCHABLE
# --------------------------------------------------------------------------- #
def _clean_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def parse_spread_reply(text: str, target_ids) -> Dict[str, Any]:
    """Parse the generator's JSON into replacements for the TARGET rows only.

    Raises :class:`SpreadParseError` (carrying the raw text) when nothing
    parseable arrives, or when the reply parses but contains no usable segment —
    those are 502s that say so. NEVER fabricates a segment: a target the model
    skipped comes back as a WARNING and an absent row, so the UI leaves the
    user's existing text alone.
    """
    from ..utils.json_scavenge import extract_json_array, extract_json_object

    raw = (text or "").strip()
    wanted = list(target_ids)
    wanted_set = set(wanted)

    parsed: Any = extract_json_object(raw)
    if parsed is None:
        arr = extract_json_array(raw, accept_lone_object=False)
        if arr is not None:
            parsed = {"segments": arr}
    if parsed is None:
        raise SpreadParseError(
            "the generator did not return the JSON object the spread contract "
            "requires", raw)

    rows = parsed.get("segments")
    if not isinstance(rows, list):
        raise SpreadParseError(
            "the generator's reply contained no \"segments\" list", raw)

    warnings: List[str] = [w for w in (parsed.get("warnings") or [])
                           if isinstance(w, str)]
    invented: List[str] = [w for w in (parsed.get("invented_identity_attributes") or [])
                           if isinstance(w, str)]

    segments: List[Dict[str, Any]] = []
    seen: set = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = _clean_str(row.get("segment_id"))
        if not sid:
            continue
        if sid not in wanted_set:
            # A LOCKED (or unknown) row. Dropped, loudly. The user did not select
            # it, so applying it would overwrite work they chose to keep — that
            # would make the selection checkbox a lie.
            warnings.append(
                f"the generator returned segment {sid!r}, which was not selected "
                "— it was discarded and that row is unchanged")
            continue
        if sid in seen:
            warnings.append(f"the generator returned segment {sid!r} twice — "
                            "the first version was kept")
            continue
        prompt = _clean_str(row.get("prompt"))
        if not prompt:
            warnings.append(f"the generator returned an empty prompt for {sid!r} "
                            "— that row is unchanged")
            continue
        seen.add(sid)
        op = row.get("operation")
        if op not in VALID_OPERATIONS:
            op = "generate"
        row_invented = [x for x in (row.get("invented_identity_attributes") or [])
                        if isinstance(x, str)]
        invented.extend(row_invented)
        directions = row.get("directions_used")
        if not isinstance(directions, list):
            directions = []
        segments.append({
            "segment_id": sid,
            "operation": op,
            "prompt": prompt,
            "negative": _clean_str(row.get("negative")),
            "continuity_note": _clean_str(row.get("continuity_note")),
            "directions_used": directions,
            "warnings": [w for w in (row.get("warnings") or [])
                         if isinstance(w, str)],
        })

    if not segments:
        raise SpreadParseError(
            "the generator returned no usable segments for the rows that were "
            "selected", raw)

    missing = [sid for sid in wanted if sid not in seen]
    if missing:
        warnings.append(
            "the generator did not write " + ", ".join(repr(m) for m in missing)
            + " — those rows are unchanged")

    return {
        "segments": segments,
        "warnings": warnings,
        "invented_identity_attributes": invented,
        "missing_segments": missing,
    }


def spread_debug_payload(req: SpreadRequest) -> str:
    """The rendered brief, for logs/tests. Not sent anywhere on its own."""
    return json.dumps({
        "target_ids": list(req.target_ids),
        "steering": req.steering,
        "steering_seed": req.steering_seed,
    }, ensure_ascii=False)
