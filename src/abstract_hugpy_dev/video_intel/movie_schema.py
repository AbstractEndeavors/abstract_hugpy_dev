"""Movie-generation schema — a GOAL TIMELINE rendered as contiguous SEGMENTS.

A movie is a SEQUENCE OF SEGMENTS. Each segment is one scene render (via the
extracted `runners.scene.render_scene_frames` core) that covers a half-open
``[start_frame, end_frame)`` slice of the movie's frame timeline and is driven by
its own goal prompt. The segments TILE the timeline exactly — contiguous and
non-overlapping — so ``total n_frames == max(end_frame)``.

A `GoalInterval` is one entry on that timeline: a frame range + the prompt the
segment should achieve, plus an optional `ref` MediaRef (an explicit start image
for that segment). The `MovieSpec` bundles the scene-template generation fields
(model/size/steps/…) shared by every segment, the ordered `goals`, and the
DIRECTOR knobs that turn on optional vision scoring + retry.

Mirrors scene_schema.py exactly: a frozen spec + a validating factory whose
raises are LOCAL to construction (never across a boundary), and asdict-friendly
serialization (nested MediaRef/GoalInterval round-trip through the bus).

Per-segment frame count is capped by the SAME FRAME_CAP as a scene (each segment
IS a scene render); the movie TOTAL is unbounded by design (movies are long).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .media_schema import MediaRef
from .scene_schema import FRAME_CAP


@dataclass(frozen=True)
class GoalInterval:
    """One entry on the movie's goal timeline.

        start_frame  inclusive frame index (>= 0)
        end_frame    EXCLUSIVE frame index (> start_frame) — half-open range
        prompt       the goal this segment should achieve (non-empty)
        ref          optional explicit start image (MediaRef, kind='image') for
                     this segment; when absent the orchestrator carries the
                     previous segment's LAST frame for cross-segment drift.
    """
    start_frame: int
    end_frame: int
    prompt: str
    ref: Optional[MediaRef] = None


@dataclass(frozen=True)
class MovieSpec:
    """Render a GOAL TIMELINE as a sequence of contiguous scene segments.

    Scene-template fields (shared by every segment) mirror GenerateSceneSpec:
        model_id, width, height, steps, guidance, fps, assemble, seed, negative,
        strength, chain, project.

    goals   the ORDERED, contiguous, non-overlapping tuple of GoalInterval that
            tiles ``[0, total)`` (total == max(end_frame)).

    Director knobs (optional vision scoring + retry):
        vision_enabled            score each segment's KEY frame before proceeding
        score_threshold           0..100; a take below it is "weak"
        max_attempts_per_segment  retry budget per segment (>=1)
        judge_model_id            optional model_key for the vision judge (else the
                                  plane's default image-text-to-text model)
        time_budget_s             optional wall-clock budget the runner owns itself
                                  (the single-daemon bus has no timeout/reaper).
    """
    # --- scene-template fields (shared by every segment) ---
    model_id: str
    width: int
    height: int
    steps: int
    guidance: float
    fps: int
    assemble: bool
    goals: Tuple[GoalInterval, ...]
    seed: Optional[int] = None
    negative: Optional[str] = None
    strength: Optional[float] = None
    chain: bool = True
    project: Optional[str] = None
    # --- director knobs ---
    vision_enabled: bool = False
    score_threshold: int = 60
    max_attempts_per_segment: int = 1
    judge_model_id: Optional[str] = None
    time_budget_s: Optional[int] = None


def make_movie(
    goals: Tuple[GoalInterval, ...],
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    fps: int,
    assemble: bool,
    seed: Optional[int] = None,
    negative: Optional[str] = None,
    strength: Optional[float] = None,
    chain: bool = True,
    project: Optional[str] = None,
    vision_enabled: bool = False,
    score_threshold: int = 60,
    max_attempts_per_segment: int = 1,
    judge_model_id: Optional[str] = None,
    time_budget_s: Optional[int] = None,
) -> MovieSpec:
    """Validate + build a MovieSpec. Raises are LOCAL to construction.

    Goal-timeline invariants (the load-bearing ones):
      * goals non-empty;
      * each goal.start_frame is an int >= 0 and goal.end_frame > start_frame
        (half-open, non-empty range);
      * each goal.prompt is a non-empty string;
      * each goal.ref (when present) is a MediaRef of kind 'image';
      * each segment's frame count (end-start) is 1..FRAME_CAP (a segment IS a
        scene render, so it shares the scene cap);
      * the goals — IN THE GIVEN ORDER — are CONTIGUOUS and non-overlapping and
        tile ``[0, total)``: the first starts at 0, each next starts exactly where
        the previous ended, so total == max(end_frame).

    Scene-template invariants mirror make_generate_scene: truthy model_id;
    positive width/height/steps; fps >= 1; assemble bool; strength in [0,1] or
    None; chain bool. Director-knob invariants: score_threshold int in 0..100;
    max_attempts_per_segment int >= 1; vision_enabled bool; time_budget_s None or
    a positive int; judge_model_id None or a non-empty str.

    Also the reconstruction path used by the bus deserializer — goals are rebuilt
    into GoalInterval (their ref through make_media_ref) before this is called.
    """
    goals = tuple(goals)
    if not goals:
        raise ValueError("make_movie requires at least one GoalInterval")

    # ---- scene-template fields ----
    if not model_id:
        raise ValueError(f"model_id must be a non-empty model key; got {model_id!r}")
    if not (isinstance(width, int) and width > 0):
        raise ValueError(f"width must be a positive int; got {width!r}")
    if not (isinstance(height, int) and height > 0):
        raise ValueError(f"height must be a positive int; got {height!r}")
    if not (isinstance(steps, int) and steps > 0):
        raise ValueError(f"steps must be a positive int; got {steps!r}")
    if not (isinstance(fps, int) and fps >= 1):
        raise ValueError(f"fps must be an int >= 1; got {fps!r}")
    if not isinstance(assemble, bool):
        raise ValueError(f"assemble must be a bool; got {assemble!r}")
    if strength is not None and not (isinstance(strength, (int, float))
                                     and 0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be a float in [0, 1] or None; got {strength!r}")
    if not isinstance(chain, bool):
        raise ValueError(f"chain must be a bool; got {chain!r}")

    # ---- director knobs ----
    if not isinstance(vision_enabled, bool):
        raise ValueError(f"vision_enabled must be a bool; got {vision_enabled!r}")
    if not (isinstance(score_threshold, int) and 0 <= score_threshold <= 100):
        raise ValueError(
            f"score_threshold must be an int in 0..100; got {score_threshold!r}")
    if not (isinstance(max_attempts_per_segment, int) and max_attempts_per_segment >= 1):
        raise ValueError(
            f"max_attempts_per_segment must be an int >= 1; got {max_attempts_per_segment!r}")
    if judge_model_id is not None and not (isinstance(judge_model_id, str) and judge_model_id):
        raise ValueError(
            f"judge_model_id must be a non-empty str or None; got {judge_model_id!r}")
    if time_budget_s is not None and not (isinstance(time_budget_s, int)
                                          and not isinstance(time_budget_s, bool)
                                          and time_budget_s > 0):
        raise ValueError(
            f"time_budget_s must be a positive int or None; got {time_budget_s!r}")

    # ---- goal-timeline invariants (contiguity in the GIVEN order) ----
    cursor = 0
    for gi, g in enumerate(goals):
        if not isinstance(g, GoalInterval):
            raise ValueError(f"goals[{gi}] must be a GoalInterval; got {type(g).__name__}")
        if not (isinstance(g.start_frame, int) and not isinstance(g.start_frame, bool)
                and g.start_frame >= 0):
            raise ValueError(
                f"goals[{gi}].start_frame must be an int >= 0; got {g.start_frame!r}")
        if not (isinstance(g.end_frame, int) and not isinstance(g.end_frame, bool)
                and g.end_frame > g.start_frame):
            raise ValueError(
                f"goals[{gi}].end_frame must be an int > start_frame "
                f"({g.start_frame}); got {g.end_frame!r}")
        if not (isinstance(g.prompt, str) and g.prompt.strip()):
            raise ValueError(f"goals[{gi}].prompt must be a non-empty string")
        if g.ref is not None:
            if not isinstance(g.ref, MediaRef):
                raise ValueError(
                    f"goals[{gi}].ref must be a MediaRef or None; got {type(g.ref).__name__}")
            if g.ref.kind != "image":
                raise ValueError(
                    f"goals[{gi}].ref must be an image MediaRef; got kind={g.ref.kind!r}")
        seg_frames = g.end_frame - g.start_frame
        if seg_frames > FRAME_CAP:
            raise ValueError(
                f"frame_cap_exceeded: goals[{gi}] spans {seg_frames} frames "
                f"({g.start_frame}..{g.end_frame}) which exceeds the per-segment "
                f"cap {FRAME_CAP}")
        # contiguity / non-overlap in the GIVEN order (must tile [0, total))
        if g.start_frame != cursor:
            raise ValueError(
                f"goals must be CONTIGUOUS + non-overlapping starting at 0: "
                f"goals[{gi}].start_frame={g.start_frame} but the previous goal "
                f"ended at {cursor} (gap or overlap)")
        cursor = g.end_frame

    return MovieSpec(
        model_id=model_id,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        fps=fps,
        assemble=assemble,
        goals=goals,
        seed=seed,
        negative=negative,
        strength=(float(strength) if strength is not None else None),
        chain=chain,
        project=(project or None),
        vision_enabled=vision_enabled,
        score_threshold=score_threshold,
        max_attempts_per_segment=max_attempts_per_segment,
        judge_model_id=(judge_model_id or None),
        time_budget_s=time_budget_s,
    )


def total_frames(spec: MovieSpec) -> int:
    """The movie's total frame count == max(end_frame) (goals tile [0, total))."""
    return max(g.end_frame for g in spec.goals)
