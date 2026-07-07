"""Headless self-test for the MOVIE TEMPLATES (curated goal-timeline presets).

Run (GPU-free, no service, drains nothing — pure in-process validation):
    PYTHONPATH=src /home/ubuntu/station/dev/abstract_hugpy_dev/venv/bin/python \
        src/abstract_hugpy_dev/video_intel/_selftest_movie_presets.py

Proves the movie-template registry (video_intel/presets.py) is internally sound
and its wire shapes are what the Movie Maker tab is built to:

  1) every MOVIE_PRESETS id is unique (dict keys + registration order);
  2) each template's goal timeline is CONTIGUOUS and TILES [0, total) with
     non-empty prompts — proven by feeding the goals through movie_schema.make_movie
     (the same factory the /video/jobs/generate_movie route uses), which raises
     locally unless the timeline is valid, then asserting total_frames() lines up;
  3) each model_key is a non-empty string (catalog membership is the apply-time
     job of the video plane; here we only assert the field is well-formed);
  4) the /apply dict ROUND-TRIPS: apply()["request"] is a directly-POSTable
     generate_movie body whose goals rebuild into GoalIntervals and reconstruct
     the SAME MovieSpec (model/dims/steps/guidance/fps/chain + the goal timeline).

Nothing here loads weights, touches a GPU, or hits the job bus — it is a static
table validated in-process, so it runs anywhere the package imports.
"""
from __future__ import annotations

import sys

from abstract_hugpy_dev.video_intel.presets import (
    MOVIE_PRESETS,
    MoviePreset,
    available_movie_presets,
    get_movie_preset,
)
from abstract_hugpy_dev.video_intel.movie_schema import (
    GoalInterval,
    make_movie,
    total_frames as spec_total_frames,
)


def _fail(msg: str) -> bool:
    print(f"  FAIL: {msg}")
    return False


def check_unique_ids() -> bool:
    print("\n=== 1) unique ids ===")
    presets = available_movie_presets()
    ids = [p.id for p in presets]
    ok = True
    if len(ids) != len(set(ids)):
        ok = _fail(f"duplicate ids in registration order: {ids}")
    # The registry dict and the ordered accessor must agree (register_movie_preset
    # refuses a dup id, so this also asserts every registration landed).
    if set(ids) != set(MOVIE_PRESETS.keys()):
        ok = _fail(f"accessor ids {set(ids)} != registry keys {set(MOVIE_PRESETS.keys())}")
    if len(ids) != len(MOVIE_PRESETS):
        ok = _fail(f"accessor len {len(ids)} != registry len {len(MOVIE_PRESETS)}")
    print(f"  {len(ids)} templates, ids={ids}")
    print(f"  {'PASS' if ok else 'FAIL'}: ids unique and registry/accessor agree")
    return ok


def _make_movie_from(fields: dict, goals: list) -> object:
    """Build a MovieSpec from a request-body-shaped dict + rebuilt GoalIntervals —
    exactly what the /video/jobs/generate_movie route does (route lines 302-334)."""
    intervals = tuple(
        GoalInterval(
            start_frame=g["start_frame"],
            end_frame=g["end_frame"],
            prompt=g["prompt"],
        )
        for g in goals
    )
    return make_movie(
        goals=intervals,
        model_id=fields["model_id"],
        width=fields["width"],
        height=fields["height"],
        steps=fields["steps"],
        guidance=fields["guidance"],
        fps=fields["fps"],
        assemble=fields["assemble"],
        chain=fields["chain"],
        vision_enabled=fields["vision_enabled"],
        score_threshold=fields["score_threshold"],
    )


def check_preset(preset: MoviePreset) -> bool:
    print(f"\n--- template {preset.id!r} ({preset.name}) ---")
    ok = True

    # 3) model_key is a non-empty string.
    if not (isinstance(preset.model_key, str) and preset.model_key.strip()):
        ok = _fail(f"model_key must be a non-empty string; got {preset.model_key!r}")
    else:
        print(f"  model_key = {preset.model_key!r}")

    # non-empty prompts on every goal (make_movie also enforces this, but assert
    # it here so a blank prompt is a named failure not a generic ValueError).
    for i, g in enumerate(preset.goals):
        if not (isinstance(g.get("prompt"), str) and g["prompt"].strip()):
            ok = _fail(f"goals[{i}].prompt must be a non-empty string")

    # 2) goals contiguous + tile [0, total) — proven by make_movie (raises locally
    # on any gap/overlap/bad range) built from the SAME goal timeline.
    try:
        spec_direct = make_movie(
            goals=tuple(
                GoalInterval(g["start_frame"], g["end_frame"], g["prompt"])
                for g in preset.goals
            ),
            model_id=preset.model_key,
            width=preset.width,
            height=preset.height,
            steps=preset.steps,
            guidance=preset.guidance,
            fps=preset.fps,
            assemble=True,
            chain=preset.chain,
            vision_enabled=preset.vision_enabled,
            score_threshold=preset.score_threshold,
        )
    except (ValueError, TypeError) as exc:
        return _fail(f"goals did not form a valid MovieSpec: {type(exc).__name__}: {exc}")

    total = spec_total_frames(spec_direct)
    if total != preset.total_frames():
        ok = _fail(f"total_frames mismatch: spec={total} preset={preset.total_frames()}")
    # explicit contiguity/tiling check (belt-and-suspenders on top of make_movie):
    cursor = 0
    for i, g in enumerate(preset.goals):
        if g["start_frame"] != cursor:
            ok = _fail(f"goals[{i}] starts at {g['start_frame']}, expected {cursor} "
                       f"(gap/overlap — not contiguous)")
        cursor = g["end_frame"]
    if cursor != total:
        ok = _fail(f"goals do not tile [0, {total}); last end_frame={cursor}")
    print(f"  {len(preset.goals)} goals tile [0, {total}) contiguously")

    # 4) the /apply dict round-trips: apply()["request"] is a directly-POSTable
    # generate_movie body; rebuilding a MovieSpec from it must match spec_direct.
    apply_env = preset.apply()
    if apply_env.get("ok") is not True:
        ok = _fail(f"apply() envelope ok != True; got {apply_env.get('ok')!r}")
    if apply_env.get("id") != preset.id:
        ok = _fail(f"apply() id {apply_env.get('id')!r} != {preset.id!r}")
    req = apply_env.get("request")
    if not isinstance(req, dict):
        return _fail("apply()['request'] missing or not an object")
    if req.get("model_id") != preset.model_key:
        ok = _fail(f"apply request model_id {req.get('model_id')!r} != {preset.model_key!r}")
    try:
        spec_round = _make_movie_from(req, req["goals"])
    except (KeyError, ValueError, TypeError) as exc:
        return _fail(f"apply request did not round-trip through make_movie: "
                     f"{type(exc).__name__}: {exc}")

    # The reconstructed spec must equal the direct one (MovieSpec is a frozen
    # dataclass — value equality over every field incl. the goal tuple).
    if spec_round != spec_direct:
        ok = _fail("round-tripped MovieSpec differs from the direct build")
    else:
        print(f"  /apply round-trips → MovieSpec(model={spec_round.model_id!r} "
              f"{spec_round.width}x{spec_round.height} steps={spec_round.steps} "
              f"guidance={spec_round.guidance} fps={spec_round.fps} "
              f"chain={spec_round.chain}, {len(spec_round.goals)} goals)")

    # get_movie_preset(id) must return this exact object (apply-route lookup).
    if get_movie_preset(preset.id) is not preset:
        ok = _fail(f"get_movie_preset({preset.id!r}) did not return this preset")

    print(f"  {'PASS' if ok else 'FAIL'}: {preset.id}")
    return ok


def main() -> int:
    print("=" * 70)
    print("MOVIE TEMPLATES self-test — registry + goal-timeline + /apply round-trip")
    print("=" * 70)

    results = [("unique ids", check_unique_ids())]
    for preset in available_movie_presets():
        results.append((preset.id, check_preset(preset)))

    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':4} — {name}")
    print("=" * 70)

    all_ok = all(ok for _, ok in results)
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'} "
          f"({len(MOVIE_PRESETS)} movie templates)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
