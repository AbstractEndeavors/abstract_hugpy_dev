"""MoE detection sweep over EVERY GGUF in the store — and a record of it.

Why (2026-07-25): ``gguf_moe_detail()`` answers a path it cannot read with a
perfectly plausible ``is_moe: False`` + zeroed byte counts — indistinguishable
from an honest "this is a dense model". In the serving path a false negative
silently means NO EXPERT SPLIT: the model loads all-layers-on-GPU and either
stalls or hogs the card. That is how coder-next ran unsplit.

Found the same day: pointing detection at a model's REPO DIRECTORY rather than a
shard file gives exactly that false negative —
    .../Qwen/Qwen3-Coder-Next-GGUF                      -> is_moe False, files 1
    .../Qwen3-Coder-Next-Q4_K_M/...-00001-of-00004.gguf -> is_moe True, 512
                                                           experts, 43.59/1.49 GiB
So: sweep every GGUF, classify each answer, and WRITE THE RECORD. The
``is_moe: False`` rows are the audit target — each must be explainable as a
genuinely dense model rather than a read that quietly produced nothing.

    venv/bin/python tests/test_moe.py            # sweep + write the record
    venv/bin/python tests/test_moe.py --quiet    # record only, no per-file lines

The pytest tests at the bottom are cheap, read no store, and DO run in CI: they
pin the false-negative contract this sweep exists to police.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.managers.spill import gguf_moe_detail  # noqa: E402

GIB = 2 ** 30
STORE = Path("/mnt/llm_storage/models")
RECORD = Path("/mnt/llm_storage/comms/moe-detection-sweep.json")
_SUSPECT_MIN_BYTES = 4 * 2 ** 20      # a >4MiB gguf should attribute SOMETHING


def _is_entrypoint(p: Path) -> bool:
    """Shard 1, or an unsharded file — never a later shard, never an mmproj.

    ``gguf_moe_detail`` is handed shard 1 and resolves the rest itself (that is
    what its ``files`` count reports), so sweeping later shards would both waste
    time and double-count bytes.
    """
    n = p.name
    if "mmproj" in n.lower():
        return False
    if "-of-" in n:
        idx = n.rsplit("-of-", 1)[0].rsplit("-", 1)[-1]
        return idx.strip("0") in ("1", "")
    return True


def discover(root: Path = STORE) -> list[Path]:
    return sorted(p for p in root.rglob("*.gguf") if _is_entrypoint(p))


def classify(path: Path, detail: dict) -> str:
    """Separate an HONEST dense answer from a FAILED READ.

    Both return ``is_moe: False`` with zero expert bytes, so the only signal is
    corroborating evidence: a real read attributes SOME bytes. Zero total bytes
    on a non-trivial file means the parse produced nothing — suspect, not dense.
    """
    if detail.get("is_moe"):
        return "moe"
    total = int(detail.get("expert_bytes") or 0) + int(detail.get("non_expert_bytes") or 0)
    if total > 0:
        return "dense"                     # real read, simply no expert tensors
    try:
        size = path.stat().st_size
    except OSError:
        return "unreadable"
    return "suspect" if size > _SUSPECT_MIN_BYTES else "dense"


def sweep(paths: list[Path], verbose: bool = True) -> list[dict]:
    rows: list[dict] = []
    for i, p in enumerate(paths, 1):
        t0 = time.time()
        err = None
        try:
            d = gguf_moe_detail(str(p))
        except Exception as exc:           # noqa: BLE001 — record it, never abort
            d, err = {}, f"{type(exc).__name__}: {exc}"
        verdict = "error" if err else classify(p, d)
        try:
            size_gib = round(p.stat().st_size / GIB, 3)
        except OSError:
            size_gib = None
        rows.append({
            "path": str(p),
            "size_gib": size_gib,
            "verdict": verdict,
            "is_moe": bool(d.get("is_moe")),
            "expert_count": d.get("expert_count"),
            "expert_used_count": d.get("expert_used_count"),
            "expert_gib": round((d.get("expert_bytes") or 0) / GIB, 3),
            "non_expert_gib": round((d.get("non_expert_bytes") or 0) / GIB, 3),
            "layers": len(d.get("expert_bytes_by_layer") or {}),
            "files": d.get("files"),
            "error": err,
            "seconds": round(time.time() - t0, 2),
        })
        if verbose:
            r = rows[-1]
            tag = {"moe": "MoE", "dense": "dense", "suspect": "SUSPECT",
                   "unreadable": "UNREADABLE", "error": "ERROR"}[verdict]
            if verdict == "moe":
                extra = (f"  {r['expert_count']}x exp, {r['expert_gib']}"
                         f"/{r['non_expert_gib']} GiB, {r['layers']}L,"
                         f" files={r['files']}")
            else:
                extra = f"  {err}" if err else f"  files={r['files']}"
            print(f"[{i:>3}/{len(paths)}] {tag:<10} {p.name[:58]:<58}{extra}")
    return rows


def tally(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["verdict"]] = out.get(r["verdict"], 0) + 1
    return out


def write_record(rows: list[dict], out: Path = RECORD) -> Path:
    payload = {
        "generated_at": int(time.time()),
        "store": str(STORE),
        "total": len(rows),
        "counts": tally(rows),
        "note": ("is_moe:false rows are the audit target — a failed READ returns "
                 "the same shape as a dense model. verdict 'suspect' = a "
                 "non-trivial gguf that attributed ZERO bytes, i.e. the parse "
                 "produced nothing."),
        "rows": rows,
    }
    blob = json.dumps(payload, indent=2)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob)
        return out
    except OSError as exc:                 # noqa: BLE001 — never lose the sweep
        alt = Path.cwd() / out.name
        alt.write_text(blob)
        print(f"  (could not write {out}: {exc}; wrote {alt})")
        return alt


def main() -> int:
    if not STORE.is_dir():
        print(f"store not present: {STORE}")
        return 2
    verbose = "--quiet" not in sys.argv
    paths = discover()
    print(f"MoE detection sweep — {len(paths)} entrypoint GGUF(s) under {STORE}\n")
    rows = sweep(paths, verbose=verbose)
    out = write_record(rows)

    counts = tally(rows)
    print("\n" + "=" * 78)
    for k in ("moe", "dense", "suspect", "unreadable", "error"):
        if counts.get(k):
            print(f"  {k:<11} {counts[k]}")
    print(f"\nrecord: {out}")

    flagged = [r for r in rows if r["verdict"] in ("suspect", "unreadable", "error")]
    if flagged:
        print(f"\n{len(flagged)} row(s) returned is_moe:false with NO evidence of a "
              "real read — these need a human verdict:")
        for r in flagged:
            print(f"  - [{r['verdict']}] {r['path']}"
                  f"{'  ' + r['error'] if r['error'] else ''}")
    return 0


# ─────────────────────── cheap pytest contract (CI-safe) ────────────────────

def test_unparseable_gguf_is_not_reported_as_a_dense_model(tmp_path):
    """The exact shape that started the coder-next hunt must be FLAGGED.

    Detection handed something it cannot parse returns is_moe:false + zero
    bytes, which a caller cannot tell apart from a real dense model — so no
    split is applied. classify() draws that line; it must say 'suspect'.
    """
    fake = tmp_path / "model.gguf"
    fake.write_bytes(b"\0" * (8 * 2 ** 20))
    try:
        d = gguf_moe_detail(str(fake))
    except Exception:                      # noqa: BLE001 — a raise is also fine
        d = {}
    assert not d.get("is_moe")
    assert classify(fake, d) == "suspect"


def test_classify_separates_real_dense_from_failed_read(tmp_path):
    f = tmp_path / "x.gguf"
    f.write_bytes(b"\0" * (8 * 2 ** 20))
    assert classify(f, {"is_moe": False, "expert_bytes": 0,
                        "non_expert_bytes": 7 * GIB, "files": 1}) == "dense"
    assert classify(f, {"is_moe": False, "expert_bytes": 0,
                        "non_expert_bytes": 0, "files": 1}) == "suspect"
    assert classify(f, {"is_moe": True, "expert_bytes": 43 * GIB,
                        "non_expert_bytes": GIB, "files": 4}) == "moe"


def test_entrypoint_filter_takes_shard_one_only():
    assert _is_entrypoint(Path("/x/Coder-Next-Q4_K_M-00001-of-00004.gguf"))
    for n in (2, 3, 4):
        assert not _is_entrypoint(
            Path(f"/x/Coder-Next-Q4_K_M-0000{n}-of-00004.gguf"))
    assert _is_entrypoint(Path("/x/plain-model.gguf"))
    assert not _is_entrypoint(Path("/x/mmproj-F16.gguf"))


if __name__ == "__main__":
    raise SystemExit(main())
