"""Worker fit-guard sizes a multi-quant GGUF by its SINGLE effective quant, not
the whole-dir sum (regression for the 2026-07-14 VRAM-fit over-count).

A GGUF repo commonly holds many quantizations (imatrix repos ship 20+), but only
ONE serves. The worker's contention fit-guard used to size the incoming model by
summing EVERY .gguf in the dir × 1.15 — so an 8B repo with 24 quants read as
~108GB and blocked loads on a 7.4GB card even though the served quant is ~5GB.
The fix makes ``agent._incoming_need_bytes`` GGUF-effective-quant-aware (mirrors
central ``model_meta`` via ``gguf_variants_detail``): for gguf/llama_cpp it sizes
by the single effective quant (+ its mmproj); non-GGUF frameworks keep the
weight-file sum (a single weight set, accurate).

Runs like the other tests here:
    venv/bin/python tests/test_incoming_need_gguf_effective.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import the agent module FIRST so sys.modules['abstract_hugpy_dev.imports'] is
# the real package (a codebase name-collision otherwise clobbers it).
from abstract_hugpy_dev.worker_agent import agent as A          # noqa: E402
from abstract_hugpy_dev.managers.serve.overrides import (        # noqa: E402
    gguf_variants_detail,
)

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print("  ok  ", name)
    else:
        fail += 1
        print("  FAIL", name)


def _mkgguf(sizes):
    d = tempfile.mkdtemp()
    for name, sz in sizes.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\0" * sz)
    return d


def _patch_cfg(framework, dest):
    """Point the agent's config/route resolution at our fixture dir."""
    import abstract_hugpy_dev.imports.config.main as CM
    CM.get_model_config = lambda mk, dict_return=False: {
        "framework": framework, "model_key": mk}
    sys.modules["abstract_hugpy_dev.imports"].route_destination = (
        lambda cfg: dest)


# --- 1) the helper itself: effective == single quant (+mmproj), NOT the sum ---
SIZES = {
    "m.i1-IQ1_S.gguf": 1000,
    "m.i1-Q2_K.gguf": 2000,
    "m.i1-Q4_K_M.gguf": 4000,   # deterministic auto-rank winner
    "m.i1-Q8_0.gguf": 8000,
    "mmproj-m.gguf": 500,       # projector: part of effective size, not a variant
}
d_multi = _mkgguf(SIZES)
g = gguf_variants_detail("x", d_multi, {"framework": "gguf"}) or {}
dir_sum = sum(SIZES.values())                       # 15500
variant_sum = sum(SIZES.values()) - SIZES["mmproj-m.gguf"]   # 15000
eff = g.get("effective_gguf")
eff_file_bytes = SIZES.get(eff)

check("resolves a single effective quant", eff in SIZES and eff != "mmproj-m.gguf")
check("effective_quant_bytes == that one file",
      g.get("effective_quant_bytes") == eff_file_bytes)
check("mmproj folded into effective_bytes",
      g.get("effective_bytes") == eff_file_bytes + SIZES["mmproj-m.gguf"])
check("effective_bytes << dir sum (the bug)",
      g.get("effective_bytes") < dir_sum)
check("mmproj excluded from servable variants",
      len(g.get("variants") or []) == 4)
check("variants would sum to the inflated number",
      sum(v["bytes"] for v in g["variants"]) == variant_sum)

# --- 2) the fit-guard: GGUF need == effective quant × 1.15, NOT the sum -------
_patch_cfg("gguf", d_multi)
need = A._incoming_need_bytes("dan-multi")
expect = int(g["effective_bytes"] * 1.15)
check("GGUF _incoming_need_bytes == effective_bytes × 1.15", need == expect)
check("GGUF need is NOT the whole-dir sum × 1.15",
      need != int(dir_sum * 1.15) and need < int(dir_sum * 1.15))

# --- 3) non-GGUF unchanged: still the weight-file sum × 1.15 ------------------
d_st = _mkgguf({"a.safetensors": 3000, "b.safetensors": 3000, "tokenizer.json": 10})
_patch_cfg("transformers", d_st)
need_st = A._incoming_need_bytes("some-transformers")
check("non-GGUF need == weight-sum × 1.15 (readme/tokenizer excluded)",
      need_st == int(6000 * 1.15))

# --- 4) fail-open: unresolved size -> None (guard never blocks) ---------------
_patch_cfg("gguf", tempfile.mkdtemp())      # empty dir: no servable .gguf
check("empty GGUF dir -> None (fail open)", A._incoming_need_bytes("empty") is None)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
