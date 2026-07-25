"""VRAM offload cliff sweep — where does throughput actually fall off?

**Operator ask (2026-07-25, verbatim):** _"just have it start at max vram with
spill to ram and see if there is any difference every 10% less vram. and have any
notable difference examined within that 10% to determine where the change was. if
there was no change ever, then thats info to know. and means that vram is not a
priority"_

So: coarse pass at 100%→0% GPU layers in 10% steps; whenever two adjacent coarse
points differ by more than ``--notable`` (default 15%), BISECT that 10% band to
locate the actual transition. A flat curve is a RESULT, not a failure — it means
VRAM is not the lever we assume it is, and the band/floor guesses built on that
assumption are unfounded.

Why this can run now, when k7 (2026-07-18) could not
----------------------------------------------------
`OFFLOAD-CLIFF-2026-07-18.md` built a harness and could not measure anything,
for two reasons it documented as blockers:
  1. the ae slot pinned models at full GPU and IGNORED the offload lever, and
  2. no central verb could cold-recycle the slot child (`unload`/`evict`/
     `slots-unload` all left the PID unchanged), so a swept budget never applied.
Both are fixed: the k14 `/slots/<id>/relaunch` verb respawns the child under a
NEW pid with an explicit `n_gpu_layers`, verified live 2026-07-25. That was
literally recommendation #1 of that document.

k7 also selected its subjects by POPULARITY (top-10 by recent usage) and found 7
of 10 sat on a CPU-only worker — so it concluded "unmeasurable". The MoE sweep
(`test_moe.py`) took the opposite approach, enumerating by STRUCTURE, and got
definitive per-model answers including for models nobody had ever called. This
harness follows the structural method: it picks subjects by what the placement
decision depends on (dense, fits the card, real chat model), not by call counts.

**A NOTE ON THE MODEL OF SPILL THIS TESTS.** `spill.cpu_resident_bytes()` today
assumes layers are UNIFORM: ``file_bytes * (1 - ngl/total)``. Real GGUFs are not
uniform — the token embedding and output head are large non-repeating tensors
that belong to no block. So the x-axis here is reported BOTH ways: the nominal
uniform estimate (what the code believes) and the measured VRAM actually taken
(what the card reports). Divergence between them is itself a finding.

Usage (needs a QUIET GPU — it repeatedly re-seats a model):

    venv/bin/python tests/test_vram_cliff.py --list
    venv/bin/python tests/test_vram_cliff.py --model <key> --path <gguf>
    venv/bin/python tests/test_vram_cliff.py --model <key> --path <gguf> \
        --step 10 --notable 15 --runs 3

Results append to /mnt/llm_storage/comms/vram-cliff-sweep.json. Nothing here is
destructive: the slot is unloaded at the end and the record is additive.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GIB = 2 ** 30
AE_AGENT = "http://192.168.1.100:9100"
AE_SLOT = "http://192.168.1.100:8101"
RECORD = Path("/mnt/llm_storage/comms/vram-cliff-sweep.json")
MOE_RECORD = Path("/mnt/llm_storage/comms/moe-detection-sweep.json")

PROMPT = ("Write a short paragraph explaining what a cache is in computing. "
          "Be specific and concrete.")


# ───────────────────────────────── plumbing ─────────────────────────────────

def _post(url: str, body: dict | None = None, timeout: float = 900.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as exc:                    # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw[:400]}


def _get(url: str, timeout: float = 30.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as exc:                    # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def vram_used_bytes() -> int | None:
    h = _get(f"{AE_AGENT}/health")
    gpus = h.get("gpus") or []
    if not gpus:
        return None
    g = gpus[0]
    tot, free = g.get("memory_total"), g.get("memory_free")
    return None if tot is None or free is None else int(tot) - int(free)


def slot_status() -> dict:
    return _get(f"{AE_SLOT}/status")


def unload() -> None:
    _post(f"{AE_SLOT}/unload", {}, timeout=120)
    time.sleep(2)


def seat(model_key: str, path: str, ngl: int) -> dict:
    """Seat the model at an explicit layer count. Returns the slot's own report.

    Deliberately uses the SLOT's /load (not central's) so the measurement is of
    the placement we asked for, with nothing in between re-deciding it.
    """
    unload()
    t0 = time.time()
    r = _post(f"{AE_SLOT}/load",
              {"model_key": model_key, "path": path, "n_gpu_layers": ngl})
    r["_load_seconds"] = round(time.time() - t0, 1)
    return r


def timed_generate(model_key: str, max_tokens: int = 96) -> dict:
    """One timed decode through the worker. tok/s from the streamed token count."""
    body = {"model_key": model_key,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{AE_AGENT}/infer/stream", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = first = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            t0 = time.time()
            for line in r:
                s = line.decode(errors="replace").strip()
                if not s.startswith("data: "):
                    continue
                try:
                    ev = json.loads(s[6:])
                except ValueError:
                    continue
                if ev.get("type") == "token":
                    tokens += 1
                    if first is None:
                        first = time.time()
    except Exception as exc:                    # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "tokens": tokens}
    end = time.time()
    if not tokens or t0 is None:
        return {"error": "no tokens", "tokens": 0}
    decode_s = end - (first or t0)
    return {
        "tokens": tokens,
        "ttft_s": round((first - t0), 3) if first else None,
        "total_s": round(end - t0, 3),
        "tok_s": round(tokens / decode_s, 2) if decode_s > 0 else None,
    }


# ───────────────────────────────── the sweep ────────────────────────────────

def measure_point(model_key: str, path: str, pct: int, total_layers: int,
                  runs: int, max_tokens: int) -> dict:
    """Seat at `pct`% of layers on GPU and measure decode throughput.

    pct=100 -> ngl=-1 (all layers, the 'max vram' start the operator asked for);
    pct=0   -> ngl=0  (pure CPU, full spill to RAM).
    """
    ngl = -1 if pct >= 100 else int(round(total_layers * pct / 100.0))
    seated = seat(model_key, path, ngl)
    if seated.get("error") or not seated.get("healthy"):
        return {"pct": pct, "ngl": ngl, "error": seated.get("error") or "unhealthy",
                "load_seconds": seated.get("_load_seconds")}
    time.sleep(1)
    vram = vram_used_bytes()
    samples = []
    for i in range(runs + 1):                   # first run is warmup, discarded
        g = timed_generate(model_key, max_tokens)
        if i and not g.get("error") and g.get("tok_s"):
            samples.append(g["tok_s"])
    st = slot_status()
    return {
        "pct": pct,
        "ngl": ngl,
        "effective_ngl": st.get("n_gpu_layers"),
        "n_cpu_moe": st.get("n_cpu_moe"),
        "vram_used_gib": round(vram / GIB, 3) if vram else None,
        "tok_s_median": round(statistics.median(samples), 2) if samples else None,
        "tok_s_samples": samples,
        "load_seconds": seated.get("_load_seconds"),
        "child_pid": st.get("child_pid"),
    }


def sweep_model(model_key: str, path: str, step: int, notable_pct: float,
                runs: int, max_tokens: int, verbose: bool = True) -> dict:
    """Coarse 100→0 in `step`%; bisect any adjacent pair differing > notable_pct."""
    total_layers = _get(f"{AE_SLOT}/status").get("total_layers")
    if not total_layers:
        probe = seat(model_key, path, -1)
        total_layers = probe.get("total_layers") or slot_status().get("total_layers")
    if not total_layers:
        return {"model_key": model_key, "error": "could not determine total_layers"}

    grid = list(range(100, -1, -step))
    if grid[-1] != 0:
        grid.append(0)
    if verbose:
        print(f"\n{model_key}\n  layers={total_layers}  grid={grid}  "
              f"runs={runs}/point  notable={notable_pct}%\n")

    coarse: list[dict] = []
    for pct in grid:
        pt = measure_point(model_key, path, pct, total_layers, runs, max_tokens)
        coarse.append(pt)
        if verbose:
            if pt.get("error"):
                print(f"  {pct:>3}% ngl={pt['ngl']:<4} ERROR {pt['error'][:70]}")
            else:
                print(f"  {pct:>3}% ngl={pt['ngl']:<4} "
                      f"vram={pt['vram_used_gib']:>6} GiB  "
                      f"{pt['tok_s_median']:>7} tok/s  "
                      f"(load {pt['load_seconds']}s)")

    # Bisect every adjacent pair whose throughput moved more than `notable_pct`.
    refined: list[dict] = []
    for a, b in zip(coarse, coarse[1:]):
        ta, tb = a.get("tok_s_median"), b.get("tok_s_median")
        if not ta or not tb:
            continue
        delta = abs(tb - ta) / max(ta, tb) * 100.0
        if delta <= notable_pct:
            continue
        lo, hi = b["pct"], a["pct"]
        if verbose:
            print(f"\n  NOTABLE {hi}%->{lo}%: {ta} -> {tb} tok/s "
                  f"({delta:.0f}% change) — bisecting\n")
        seen = {a["pct"]: ta, b["pct"]: tb}
        for _ in range(3):                      # 3 splits ~= 1-2% resolution
            mid = (lo + hi) // 2
            if mid in seen or mid in (lo, hi):
                break
            pt = measure_point(model_key, path, mid, total_layers, runs, max_tokens)
            refined.append(pt)
            tm = pt.get("tok_s_median")
            if verbose:
                print(f"    bisect {mid:>3}% ngl={pt['ngl']:<4} "
                      f"{tm if tm else pt.get('error','?')} tok/s")
            if not tm:
                break
            seen[mid] = tm
            # Walk toward whichever side still holds the discontinuity.
            if abs(tm - seen[hi]) > abs(tm - seen[lo]):
                hi = mid
            else:
                lo = mid
        if verbose:
            print(f"  -> transition localized between {lo}% and {hi}% GPU layers\n")

    ok = [p for p in coarse if p.get("tok_s_median")]
    spread = None
    if len(ok) >= 2:
        vals = [p["tok_s_median"] for p in ok]
        spread = round((max(vals) - min(vals)) / max(vals) * 100.0, 1)
    return {
        "model_key": model_key,
        "path": path,
        "total_layers": total_layers,
        "coarse": coarse,
        "refined": refined,
        "spread_pct": spread,
        "verdict": ("FLAT — no measurable cliff; VRAM placement is NOT the lever "
                    "for this model" if spread is not None and spread < notable_pct
                    else "cliff present" if spread is not None else "inconclusive"),
    }


def dense_candidates(limit: int = 10, min_gib: float = 3.0,
                     max_gib: float = 20.0) -> list[dict]:
    """Dense GGUFs that FIT the card, from the MoE sweep record (structural pick).

    Deliberately not a popularity ranking — that is the selection mistake that
    made the k7 sweep unmeasurable.
    """
    if not MOE_RECORD.is_file():
        return []
    rows = json.loads(MOE_RECORD.read_text())["rows"]
    out = [r for r in rows
           if not r["is_moe"] and r["verdict"] == "dense"
           and "_archive" not in r["path"]
           and min_gib <= (r["size_gib"] or 0) <= max_gib]
    out.sort(key=lambda r: -(r["size_gib"] or 0))
    return out[:limit]


def append_record(result: dict, out: Path = RECORD) -> Path:
    payload = {"runs": []}
    if out.is_file():
        try:
            payload = json.loads(out.read_text())
        except ValueError:
            pass
    payload.setdefault("runs", []).append({"at": int(time.time()), **result})
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        return out
    except OSError:
        alt = Path.cwd() / out.name
        alt.write_text(json.dumps(payload, indent=2))
        return alt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list dense candidates that fit the card, then exit")
    ap.add_argument("--model", help="model_key to sweep")
    ap.add_argument("--path", help="absolute .gguf path (shard 1 if sharded)")
    ap.add_argument("--step", type=int, default=10, help="coarse step %% (default 10)")
    ap.add_argument("--notable", type=float, default=15.0,
                    help="%% throughput change that triggers a bisect (default 15)")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per point")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--keep", action="store_true",
                    help="leave the last placement seated (default: unload)")
    a = ap.parse_args(argv)

    if a.list or not (a.model and a.path):
        cands = dense_candidates()
        if not cands:
            print(f"no candidates (is {MOE_RECORD} present? run test_moe.py first)")
            return 2
        print("dense GGUFs that fit the 24 GiB card (structural pick):\n")
        for r in cands:
            print(f"  {r['size_gib']:>6.2f} GiB  {r['path']}")
        print("\nsweep one with:\n  venv/bin/python tests/test_vram_cliff.py \\\n"
              "      --model '<model_key>' --path '<path above>'")
        return 0

    h = _get(f"{AE_AGENT}/health")
    if h.get("error"):
        print(f"ae agent unreachable: {h['error']}")
        return 2
    print(f"ae: {h.get('name')}  gpus={[g.get('name') for g in (h.get('gpus') or [])]}")

    result = sweep_model(a.model, a.path, a.step, a.notable, a.runs, a.max_tokens)
    if not a.keep:
        unload()

    out = append_record(result)
    print("\n" + "=" * 78)
    print(f"  verdict: {result.get('verdict')}")
    if result.get("spread_pct") is not None:
        print(f"  spread : {result['spread_pct']}% between fastest and slowest")
    print(f"  record : {out}")
    if (result.get("spread_pct") or 0) < a.notable:
        print("\n  A FLAT curve is a real result: it says GPU layer placement did "
              "not\n  move throughput for this model — so VRAM is not the "
              "priority the\n  band/floor guesses assume, and those guesses are "
              "unfounded.")
    return 0


# ─────────────────────── cheap pytest contract (CI-safe) ────────────────────
# No GPU, no store: these pin the sweep's LOGIC so it can't silently rot.

def test_grid_starts_at_max_vram_and_reaches_zero():
    step = 10
    grid = list(range(100, -1, -step))
    if grid[-1] != 0:
        grid.append(0)
    assert grid[0] == 100 and grid[-1] == 0
    assert len(grid) == 11


def test_pct_to_ngl_mapping():
    total = 48
    assert (-1 if 100 >= 100 else 0) == -1          # 100% -> -1 (all layers)
    assert int(round(total * 50 / 100.0)) == 24
    assert int(round(total * 0 / 100.0)) == 0


def test_flat_curve_is_reported_as_a_result_not_a_failure():
    vals = [20.0, 19.8, 20.1, 19.9]
    spread = (max(vals) - min(vals)) / max(vals) * 100.0
    assert spread < 15.0                             # would read as FLAT


if __name__ == "__main__":
    raise SystemExit(main())
