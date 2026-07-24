"""MoE-PLACEMENT productization (2026-07-24 measured win, operator-grounded).

Measured on ae/3090 (Qwen3-Coder-Next, 80B-A3B MoE): the naive 17/48 layer
split gave ~15.2 tok/s @ 16.6 GiB VRAM; the MoE-aware split (n_gpu_layers=-1 +
llama-server --n-cpu-moe 999: experts to CPU, everything else + KV on GPU) gave
~24.1 tok/s @ 3.2 GiB — +59% AND 5x less VRAM. This suite proves the feature:

  * GGUF MoE detection from synthetic headers the REAL reader parses
    (expert_count KV = the definition; `_exps` tensor suffix = the per-tensor
    is_expert bit; router/shexp never match), shard-aware, cached;
  * ground-truth reconciliation against the real coder-next shards when local
    (expert 43.59 GiB / non-expert 1.49 GiB / ec 512 / euc 10 — keeper-parsed);
  * per-layer-aware split pricing (spill.moe_split_need) incl. partial N;
  * the _build_cmd AUTO policy matrix: MoE+hybrid -> -1 + --n-cpu-moe 999;
    MoE+fits-whole -> --n-cpu-moe 0 (env-hack-proof); dense -> unchanged;
    explicit n_gpu_layers / n_cpu_moe always win; engine-degrade paths;
  * relaunch accepts n_cpu_moe; slot admission verdict threads it;
  * expert-aware need: the fit checks pass the empty-card 41.6GB-MoE case;
    _vram_evict_to_fit re-targets to the split instead of an impossible full
    fit; calibration verdict honesty; feasibility with MoE sizing;
  * the n_cpu_moe knob: overrides coercion + spill env wire.

Run: venv/bin/python -m pytest tests/test_moe_placement.py -q
"""
import importlib
import os
import struct
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import the agent module FIRST (codebase name-collision landmine — see the
# sibling tests) and bind the real dispatch module via import_module.
from abstract_hugpy_dev.worker_agent import agent as A            # noqa: E402
from abstract_hugpy_dev.managers import spill                     # noqa: E402
from abstract_hugpy_dev.managers import alloc_modes as AM         # noqa: E402

sa = importlib.import_module("abstract_hugpy_dev.managers.serve.slot_agent")
SL = importlib.import_module("abstract_hugpy_dev.managers.serve.slots")
OV = importlib.import_module("abstract_hugpy_dev.managers.serve.overrides")
D = importlib.import_module("abstract_hugpy_dev.managers.dispatch.dispatch")

GIB = 1 << 30


# ═══════════ synthetic GGUF fixtures (the real reader parses these) ═════════
def _mk_gguf(path, *, tensors=(), block_count=None, expert_count=None,
             expert_used=None, dims_by_name=None):
    """A minimal-but-real GGUF v3: magic/version/counts, a KV table (uint32
    values), a tensor-info table (name/dims/type/offset), aligned data section.
    Tensor sizes are realized by consecutive offsets + real file padding, which
    is exactly how the reader prices them (offset deltas, no type table).

    ``dims_by_name`` optionally supplies a real dims tuple per tensor name (e.g.
    ``(2048, 512, 512)`` for a stacked expert weight), so the SHAPE backstop can
    be exercised; tensors absent from the map keep the historical 1-D ``(1,)``
    (which the shape method — needing nd>=3 — correctly ignores)."""
    import io
    buf = io.BytesIO()
    dims_by_name = dims_by_name or {}

    def ws(s):
        b = s.encode()
        buf.write(struct.pack("<Q", len(b)))
        buf.write(b)

    kvs = []
    if block_count is not None:
        kvs.append(("fake.block_count", block_count))
    if expert_count is not None:
        kvs.append(("fake.expert_count", expert_count))
    if expert_used is not None:
        kvs.append(("fake.expert_used_count", expert_used))
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))
    buf.write(struct.pack("<Q", len(tensors)))
    buf.write(struct.pack("<Q", len(kvs)))
    for key, val in kvs:
        ws(key)
        buf.write(struct.pack("<I", 5))          # uint32
        buf.write(struct.pack("<I", val))
    off = 0
    for name, size in tensors:
        ws(name)
        dims = dims_by_name.get(name, (1,))      # default 1-D (unused by name path)
        buf.write(struct.pack("<I", len(dims)))  # n_dims
        for d in dims:
            buf.write(struct.pack("<Q", d))
        buf.write(struct.pack("<I", 0))          # ggml type (unused)
        buf.write(struct.pack("<Q", off))
        off += size
    header = buf.getvalue()
    data_start = (len(header) + 31) // 32 * 32   # general.alignment default 32
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(b"\0" * (data_start - len(header)))
        fh.write(b"\0" * off)
    return str(path)


_MOE_TENSORS = (
    ("token_embd.weight", 512),
    ("blk.0.attn_q.weight", 1000),
    ("blk.0.ffn_gate_inp.weight", 96),           # router — NOT an expert
    ("blk.0.ffn_up_shexp.weight", 224),          # shared expert — NOT an expert
    ("blk.0.ffn_up_exps.weight", 4000),
    ("blk.0.ffn_down_exps.weight", 4000),
    ("blk.1.attn_q.weight", 1000),
    ("blk.1.ffn_gate_exps.weight", 8000),
    ("output.weight", 500),
)
# expert = 16000; non-expert = 512+1000+96+224+1000+500 = 3332


@pytest.fixture
def moe_gguf(tmp_path):
    return _mk_gguf(tmp_path / "moe.gguf", tensors=_MOE_TENSORS,
                    block_count=48, expert_count=512, expert_used=10)


@pytest.fixture
def dense_gguf(tmp_path):
    tensors = tuple((n, s) for n, s in _MOE_TENSORS if "_exps" not in n)
    return _mk_gguf(tmp_path / "dense.gguf", tensors=tensors, block_count=48)


# ═══════════ detection + byte split ═════════════════════════════════════════
def test_detects_moe_and_splits_bytes_by_kind(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    assert d["is_moe"] is True
    assert d["expert_count"] == 512
    assert d["expert_used_count"] == 10
    assert d["sparsity"] == pytest.approx(10 / 512)
    assert d["expert_bytes"] == 16000
    assert d["non_expert_bytes"] == 3332
    # per-layer attribution (the future-partial pricing input)
    assert d["expert_bytes_by_layer"] == {0: 8000, 1: 8000}


def test_router_and_shared_experts_are_not_experts(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    # ffn_gate_inp (router) + ffn_up_shexp (shared) priced GPU-side: their
    # bytes are inside non_expert_bytes, never expert_bytes.
    assert d["expert_bytes"] == 16000            # only the _exps tensors


def test_absent_expert_count_is_dense(dense_gguf):
    d = spill.gguf_moe_detail(dense_gguf)
    assert d["is_moe"] is False
    assert d["expert_bytes"] == 0


def test_zero_expert_count_is_dense(tmp_path):
    p = _mk_gguf(tmp_path / "zero.gguf", tensors=_MOE_TENSORS,
                 block_count=48, expert_count=0)
    assert spill.gguf_moe_detail(p)["is_moe"] is False


def test_missing_or_garbage_file_degrades_to_dense(tmp_path):
    assert spill.gguf_moe_detail("/nope/never.gguf") == {"is_moe": False}
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    assert spill.gguf_moe_detail(str(junk))["is_moe"] is False


def test_shard_aware_sums_across_shards(tmp_path):
    s1 = _mk_gguf(tmp_path / "m-00001-of-00002.gguf",
                  tensors=_MOE_TENSORS[:5], block_count=48,
                  expert_count=512, expert_used=10)
    _mk_gguf(tmp_path / "m-00002-of-00002.gguf", tensors=_MOE_TENSORS[5:])
    d = spill.gguf_moe_detail(s1)
    assert d["files"] == 2
    assert d["is_moe"] is True
    assert d["expert_bytes"] == 16000            # summed across both shards
    assert d["non_expert_bytes"] == 3332
    assert d["expert_bytes_by_layer"] == {0: 8000, 1: 8000}


def test_detail_is_cached_by_path_signature(moe_gguf, monkeypatch):
    first = spill.gguf_moe_detail(moe_gguf)

    def _boom(_p):
        raise AssertionError("re-parsed a cached header")
    monkeypatch.setattr(spill, "_gguf_scan_moe", _boom)
    assert spill.gguf_moe_detail(moe_gguf) == first


# ═══════════ shape-derived detection (name-independent backstop) ════════════
# Empirically grounded 2026-07-24 against the real coder-next shards: an expert
# tensor is the STACKED one — n_dims>=3 with expert_count in the LAST dims slot
# (dims[-1]). These tensors are named unconventionally (NO _exps suffix) so ONLY
# the shape method can find them; their dims carry expert_count=512 in dims[-1].
_SHAPE_ONLY_TENSORS = (
    ("token_embd.weight", 512),
    ("blk.0.attn_q.weight", 1000),
    ("blk.0.router.weight", 96),                 # router — 2-D, not an expert
    ("blk.0.moe_up.weight", 4000),               # expert by SHAPE (nd=3), no _exps
    ("blk.0.moe_down.weight", 4000),             # expert by SHAPE
    ("blk.1.attn_q.weight", 1000),
    ("blk.1.moe_gate.weight", 8000),             # expert by SHAPE
    ("output.weight", 500),
)
_SHAPE_DIMS = {                                  # dims[-1]==512 == expert_count
    "blk.0.moe_up.weight": (2048, 512, 512),
    "blk.0.moe_down.weight": (512, 2048, 512),
    "blk.1.moe_gate.weight": (2048, 512, 512),
    "blk.0.router.weight": (2048, 512),          # 2-D: dims[-1]==512 but NOT nd>=3
}


def test_shape_detects_experts_when_naming_is_nonstandard(tmp_path, caplog):
    """expert_count set, tensors shaped [.,.,512] but NOT named *_exps -> the
    shape backstop finds them; a single WARNING names the nonconforming file."""
    p = _mk_gguf(tmp_path / "shape.gguf", tensors=_SHAPE_ONLY_TENSORS,
                 block_count=48, expert_count=512, expert_used=10,
                 dims_by_name=_SHAPE_DIMS)
    with caplog.at_level("WARNING"):
        d = spill.gguf_moe_detail(p)
    assert d["is_moe"] is True
    assert d["expert_count"] == 512 and d["expert_used_count"] == 10
    # experts = moe_up + moe_down + moe_gate = 4000+4000+8000; 2-D router excluded
    assert d["expert_bytes"] == 16000
    assert d["non_expert_bytes"] == 512 + 1000 + 96 + 1000 + 500
    assert d["expert_bytes_by_layer"] == {0: 8000, 1: 8000}
    assert any("nonstandard converter" in r.message for r in caplog.records)


def test_shape_2d_dim_collision_is_not_an_expert(tmp_path):
    """The nd>=3 guard: a 2-D tensor whose dims[-1] happens to equal
    expert_count is NOT an expert (only stacked/3-D weights are). The router
    here is 2-D (2048,512) -> excluded, so no false expert bytes."""
    p = _mk_gguf(tmp_path / "shape.gguf", tensors=_SHAPE_ONLY_TENSORS,
                 block_count=48, expert_count=512, expert_used=10,
                 dims_by_name=_SHAPE_DIMS)
    d = spill.gguf_moe_detail(p)
    assert d["expert_bytes"] == 16000            # router's 96 bytes stay non-expert


def test_count_but_no_experts_anywhere_falls_back_to_dense(tmp_path, caplog):
    """Header claims MoE (expert_count) but NEITHER name nor shape finds an
    expert tensor -> dense fallback (safe plain split) + one WARNING."""
    plain = tuple((n, s) for n, s in _MOE_TENSORS if "_exps" not in n)
    p = _mk_gguf(tmp_path / "liar.gguf", tensors=plain,
                 block_count=48, expert_count=512, expert_used=10)
    with caplog.at_level("WARNING"):
        d = spill.gguf_moe_detail(p)
    assert d["is_moe"] is False                  # safe fallback: no mispriced split
    assert d["expert_bytes"] == 0
    assert any("no expert tensors identifiable" in r.message
               for r in caplog.records)


def test_names_without_expert_count_are_dense(tmp_path, caplog):
    """REVERSE inconsistency: _exps-named tensors but NO expert_count -> names
    alone never activate the split; dense + one WARNING (metadata is the gate)."""
    p = _mk_gguf(tmp_path / "names.gguf", tensors=_MOE_TENSORS, block_count=48)
    with caplog.at_level("WARNING"):
        d = spill.gguf_moe_detail(p)
    assert d["is_moe"] is False
    assert d["expert_bytes"] == 0                # metadata gate: no split
    assert any("no positive expert_count" in r.message for r in caplog.records)


def test_conforming_file_logs_no_warning(moe_gguf, caplog):
    """A file where name and shape AGREE (or the standard _exps path) is silent —
    warnings are for DRIFT only, never the happy path."""
    with caplog.at_level("WARNING"):
        spill.gguf_moe_detail(moe_gguf)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ═══════════ per-layer-aware split pricing ══════════════════════════════════
def test_moe_split_need_all_and_sentinel(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    for n in (None, 999, spill.MOE_ALL_LAYERS, 2):
        got = spill.moe_split_need(d, n)
        assert got["cpu_bytes"] == 16000
        assert got["gpu_bytes"] == 3332


def test_moe_split_need_partial_is_per_layer_exact(moe_gguf):
    d = spill.gguf_moe_detail(moe_gguf)
    one = spill.moe_split_need(d, 1)             # first layer's experts only
    assert one == {"cpu_bytes": 8000, "gpu_bytes": 3332 + 8000,
                   "layers_on_cpu": 1}
    zero = spill.moe_split_need(d, 0)
    assert zero["cpu_bytes"] == 0
    assert zero["gpu_bytes"] == 16000 + 3332


def test_moe_split_need_dense_is_none():
    assert spill.moe_split_need({"is_moe": False}) is None
    assert spill.moe_split_need(None) is None


# ═══════════ ground truth: the real coder-next shards ═══════════════════════
_REAL_SHARD1 = ("/mnt/llm_storage/legacy/Qwen3-Coder-Next-GGUF/"
                "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf")


@pytest.mark.skipif(not Path(_REAL_SHARD1).is_file(),
                    reason="real coder-next shards not present on this box")
def test_real_coder_next_header_reconciliation():
    """Keeper-parsed ground truth (2026-07-24, reconciled against the live ae
    card within ~0.2 GiB): expert 43.59 GiB (96.7%), non-expert 1.49 GiB,
    expert_count 512, expert_used_count 10, 48 expert-bearing blocks. The
    non-expert share + KV is what the measured 3.2 GiB VRAM footprint is."""
    d = spill.gguf_moe_detail(_REAL_SHARD1)
    assert d["is_moe"] is True and d["files"] == 4
    assert d["expert_count"] == 512 and d["expert_used_count"] == 10
    assert d["expert_bytes"] / GIB == pytest.approx(43.59, abs=0.05)
    assert d["non_expert_bytes"] / GIB == pytest.approx(1.49, abs=0.05)
    assert len(d["expert_bytes_by_layer"]) == 48
    need = spill.moe_split_need(d)
    assert need["gpu_bytes"] == d["non_expert_bytes"]
    assert need["cpu_bytes"] == d["expert_bytes"]


@pytest.mark.skipif(not Path(_REAL_SHARD1).is_file(),
                    reason="real coder-next shards not present on this box")
def test_real_shard_name_and_shape_agree_exactly():
    """The strongest consistency proof: on the REAL file the NAME method and the
    SHAPE backstop must select the IDENTICAL expert set (same bytes, same
    per-layer map, same hit count). Verified 2026-07-24: 144 _exps tensors, all
    3-D with dims[-1]==expert_count=512; a conforming file logs NO warning."""
    shards = spill._gguf_shard_paths(_REAL_SHARD1)
    n_name = n_shape = 0
    b_name = b_shape = 0
    ec = None
    for sh in shards:
        # Thread the count forward exactly as gguf_moe_detail does (only shard 1
        # carries expert_count; the hint lets shape fire on later shards too).
        scan = spill._gguf_scan_moe(sh, expert_count_hint=ec)
        if ec is None and scan.get("expert_count") is not None:
            ec = int(scan["expert_count"])
        n_name += scan["name_expert_hits"]
        n_shape += scan["shape_expert_hits"]
        b_name += scan["expert_bytes"]
        b_shape += scan["expert_bytes_shape"]
    assert n_name == n_shape == 144              # identical tensor set, both ways
    assert b_name == b_shape                     # byte-identical split
    assert (b_name / GIB) == pytest.approx(43.59, abs=0.05)


# ═══════════ _build_cmd — THE argv choke point (loads AND relaunches) ═══════
@pytest.fixture
def cmd_rig(monkeypatch, moe_gguf, dense_gguf):
    """Route _build_cmd's collaborators: a fake native llama-server that exists
    (/bin/echo), --n-cpu-moe supported, autofit controllable."""
    serve = importlib.import_module("abstract_hugpy_dev.managers.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", "/bin/echo")
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: True)
    auto = {"value": 17}
    monkeypatch.setattr(spill, "autofit_gpu_layers",
                        lambda p, free_vram=None, extra_reserve_bytes=0: auto["value"])
    for env in ("HUGPY_ALLOC_MODE", "HUGPY_N_CPU_MOE", "HUGPY_N_GPU_LAYERS",
                "HUGPY_HOT_CACHE_ROOT", "HUGPY_MODEL_CACHE"):
        monkeypatch.delenv(env, raising=False)
    return type("Rig", (), {"auto": auto, "moe": moe_gguf, "dense": dense_gguf})()


def _argv_pairs(argv):
    return {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1)}


def test_auto_policy_moe_hybrid_becomes_expert_split(cmd_rig):
    cmd_rig.auto["value"] = 17                   # hybrid: partial layer split
    (argv, ngl, _c, _t, _cp, kind, total, ncm) = sa._build_cmd(
        "moe-model", path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert kind == "binary"
    assert ngl == -1 and pairs["--n-gpu-layers"] == "-1"
    assert ncm == spill.MOE_ALL_LAYERS and pairs["--n-cpu-moe"] == "999"
    assert total == 48


def test_auto_policy_moe_fits_whole_pins_experts_on_gpu(cmd_rig):
    cmd_rig.auto["value"] = -1                   # whole model fits
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    # fully-on-GPU as today, with an explicit 0 so a transition-era
    # LLAMA_ARG_N_CPU_MOE unit env can't silently displace the experts.
    assert ngl == -1 and ncm == 0 and pairs["--n-cpu-moe"] == "0"


def test_auto_policy_dense_is_byte_identical(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("dense-model", path=cmd_rig.dense)
    assert ngl == 17 and ncm is None
    assert "--n-cpu-moe" not in argv


def test_explicit_n_gpu_layers_wins_over_auto_split(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=20,
                                             path=cmd_rig.moe)
    assert ngl == 20 and ncm is None
    assert "--n-cpu-moe" not in argv


def test_explicit_n_cpu_moe_always_wins(cmd_rig):
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", n_gpu_layers=-1,
                                             n_cpu_moe=12, path=cmd_rig.moe)
    pairs = _argv_pairs(argv)
    assert ngl == -1 and ncm == 12 and pairs["--n-cpu-moe"] == "12"


def test_k37_mode_engine_disables_the_auto_split(cmd_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "max-ram")
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == 17 and ncm is None and "--n-cpu-moe" not in argv


def test_old_llama_server_degrades_to_layer_split(cmd_rig, monkeypatch):
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: False)
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, ncm) = sa._build_cmd("moe-model", path=cmd_rig.moe)
    assert ngl == 17 and ncm is None             # reverted, no unknown flag
    assert "--n-cpu-moe" not in argv


def test_python_child_degrades_to_layer_split(cmd_rig, monkeypatch):
    serve = importlib.import_module("abstract_hugpy_dev.managers.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", None)   # no native engine
    cmd_rig.auto["value"] = 17
    (argv, ngl, *_rest, kind, _total, ncm) = sa._build_cmd(
        "moe-model", path=cmd_rig.moe)
    assert kind == "python" and ngl == 17 and ncm is None
    assert "--n-cpu-moe" not in argv


# ═══════════ relaunch threads n_cpu_moe (k14 lever) ═════════════════════════
def test_slot_relaunch_accepts_and_forwards_n_cpu_moe():
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = "coder"
    s.ngl, s.ctx, s.threads, s.cpus, s.gpu = -1, 4096, 6, None, None
    s.profile_bin = None
    s._load_failures, s._load_backoff_until = {}, {}
    s.lock = threading.Lock()
    seen = {}

    def _fake_load(model_key, **kw):
        seen["model_key"] = model_key
        seen.update(kw)
        return {"model_key": model_key, "n_gpu_layers": -1, "n_cpu_moe": 999}
    s.load = _fake_load
    out = s.relaunch(n_gpu_layers=-1, n_cpu_moe=999)
    assert seen["n_cpu_moe"] == 999 and seen["force"] is True
    assert out["relaunched"] is True and out["requested_n_cpu_moe"] == 999


# ═══════════ slot admission threads the MoE verdict ═════════════════════════
def test_endpoint_for_threads_moe_verdict_into_load_opts(monkeypatch):
    posts = []

    def _fake_post(url, body, timeout):
        posts.append((url, dict(body)))
        return {"endpoint": "http://fake:1"}
    monkeypatch.setattr(SL, "_post", _fake_post)
    monkeypatch.setattr(
        SL.SlotPool, "statuses",
        lambda self: [{"_control": "http://fake:1", "healthy": True,
                       "model_key": None}])
    monkeypatch.setattr(SL, "_FIT_CHECK", lambda mk: False)     # over ceiling
    monkeypatch.setattr(SL, "_EVICTION_POLICY", None)
    monkeypatch.setattr(SL, "_MAKE_ROOM", lambda mk: {
        "action": "partial", "n_gpu_layers": -1, "n_cpu_moe": 999,
        "evicted": []})
    pool = SL.SlotPool(urls=["http://fake:1"])
    ep = pool.endpoint_for("moe-model")
    assert ep == "http://fake:1"
    _url, body = posts[-1]
    assert body["n_gpu_layers"] == -1 and body["n_cpu_moe"] == 999


# ═══════════ expert-aware need — the fit checks ═════════════════════════════
_EMPTY_CARD_DET = {
    "total": int(41.6 * GIB * 1.15),             # the opaque full-file need
    "moe_split": {"n_cpu_moe": 999, "gpu_total": int(2.9 * GIB),
                  "cpu_bytes": int(43.5 * GIB), "path": "/x.gguf"},
}


def test_slot_fit_check_passes_empty_card_moe_case(monkeypatch):
    """The live refusal this fixes: /probe (and the boot star) on an EMPTY
    23.6 GiB card said fit:false for the 41.6GB MoE. Under the split the GPU
    need is the non-expert share -> passes; experts checked against RAM."""
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_slot_fit_check("coder-next") is True


def test_slot_fit_check_without_split_still_refuses(monkeypatch):
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    det = {"total": _EMPTY_CARD_DET["total"]}    # no moe_split (dense/explicit)
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: det)
    assert A._worker_slot_fit_check("coder-next") is False


def test_slot_fit_check_moe_respects_ram_guard(monkeypatch):
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(10 * GIB))
    assert A._worker_slot_fit_check("coder-next") is False       # experts > RAM
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: None)
    assert A._worker_slot_fit_check("coder-next") is True        # unmeasurable: open


def test_contention_fit_check_uses_split_need(monkeypatch):
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23.6 * GIB))
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: int(60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(_EMPTY_CARD_DET))
    assert A._worker_fit_check("coder-next") is True
    det = {"total": _EMPTY_CARD_DET["total"]}
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: det)
    assert A._worker_fit_check("coder-next") is False


# ═══════════ _vram_evict_to_fit — the admission choke point ═════════════════
class _State:
    pass


@pytest.fixture
def evict_rig(monkeypatch):
    card = {"total": 24 * GIB, "free": 23 * GIB, "need": int(47.8 * GIB)}
    residents = {}
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: card.get("ram", 60 * GIB))
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: card["need"])
    monkeypatch.setattr(A, "_kv_need_bytes", lambda mk, cfg=None: (0, {}))
    monkeypatch.setattr(A, "_calib_correction", lambda mk: None)
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v,
                                    "host_mode": "subprocess", "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_queued_ahead_of", lambda mk: set())
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: {})
    plan = {"value": {"path": "/x.gguf", "n_cpu_moe": 999,
                      "gpu_weight_bytes": int(2.5 * GIB),
                      "cpu_bytes": int(43.5 * GIB),
                      "detail": {"expert_count": 512, "expert_used_count": 10,
                                 "sparsity": 10 / 512}}}
    monkeypatch.setattr(A, "_moe_plan_for", lambda mk: plan["value"])
    A._MOE_SPLIT.clear()
    A._PARTIAL_NGL.clear()
    yield type("Rig", (), {"card": card, "residents": residents, "plan": plan})()
    A._MOE_SPLIT.clear()
    A._PARTIAL_NGL.clear()


def test_admission_retargets_impossible_full_fit_to_moe_split(evict_rig):
    """47.8G full need can NEVER fit a 24G card — the admission prices the
    split (2.875G GPU) instead: admits with the MoE verdict, evicts NOBODY."""
    evict_rig.residents["innocent"] = 10 * GIB
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "partial"
    assert verdict["n_gpu_layers"] == -1 and verdict["n_cpu_moe"] == 999
    assert verdict["evicted"] == []              # no innocents evicted
    assert A._MOE_SPLIT["coder-next"]["n_cpu_moe"] == 999
    assert verdict["moe"]["cpu_bytes"] == int(43.5 * GIB)


def test_admission_moe_split_still_evicts_when_gpu_share_needs_room(evict_rig):
    # Split GPU need 2.875G; only 1G free; a 10G idle resident yields.
    evict_rig.card["free"] = 1 * GIB
    calls = []

    def _fake_evict(state, mk, force=False):
        calls.append(mk)
        freed = evict_rig.residents.pop(mk)
        evict_rig.card["free"] += freed
        return {"model_key": mk, "evicted": True, "vram_freed": freed,
                "host_mode": "subprocess"}
    A._evict_model, orig = _fake_evict, A._evict_model
    try:
        evict_rig.residents["idle"] = 10 * GIB
        verdict = A._vram_evict_to_fit(_State(), "coder-next")
    finally:
        A._evict_model = orig
    assert verdict["action"] == "partial" and verdict["n_cpu_moe"] == 999
    assert verdict["evicted"] == ["idle"]


def test_admission_moe_ram_guard_keeps_full_need(evict_rig):
    # Experts (43.5G) exceed budgetable RAM -> no re-target; the full-need
    # path stands and refuses honestly (no admit-then-thrash).
    evict_rig.card["ram"] = 20 * GIB
    verdict = A._vram_evict_to_fit(_State(), "coder-next")
    assert verdict["action"] == "refuse"
    assert "coder-next" not in A._MOE_SPLIT
    assert verdict["reason"]["moe_split"]["was_plan"] is False


def test_admission_fits_whole_after_evict_stays_full_gpu(evict_rig):
    # need 18G <= 24G - 2.4G reserve: full fit is possible -> today's
    # evict-to-fit-full, NOT the split (whole-model-fits keeps fully-on-GPU).
    evict_rig.card["need"] = 18 * GIB
    evict_rig.card["free"] = 1 * GIB
    calls = []

    def _fake_evict(state, mk, force=False):
        calls.append(mk)
        freed = evict_rig.residents.pop(mk)
        evict_rig.card["free"] += freed
        return {"model_key": mk, "evicted": True, "vram_freed": freed,
                "host_mode": "subprocess"}
    A._evict_model, orig = _fake_evict, A._evict_model
    try:
        evict_rig.residents["idle"] = 21 * GIB
        verdict = A._vram_evict_to_fit(_State(), "coder-next")
    finally:
        A._evict_model = orig
    assert verdict["action"] == "evicted"
    assert verdict.get("n_cpu_moe") is None
    assert "coder-next" not in A._MOE_SPLIT


def test_clear_partial_ngl_also_clears_moe_commit():
    A._MOE_SPLIT["m"] = {"path": "/x.gguf", "n_cpu_moe": 999}
    A._clear_partial_ngl("m")
    assert "m" not in A._MOE_SPLIT


# ═══════════ calibration honesty ════════════════════════════════════════════
def test_moe_split_residency_never_reads_as_a_full_load(monkeypatch):
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"base_total": 48 * GIB, "weights": 47 * GIB,
                                    "kv": 0, "ctx_pct": None})
    monkeypatch.setattr(A, "_model_framework", lambda mk: "gguf")
    A._MOE_SPLIT["m"] = {"path": "/x.gguf", "n_cpu_moe": 999}
    try:
        s = A._build_calibration_success(
            "m", {"device": "cuda", "n_gpu_layers": -1,
                  "vram_bytes": 3 * GIB, "rss_bytes": GIB})
    finally:
        A._MOE_SPLIT.clear()
    assert s["verdict"] == "partial"             # excluded from the full ratio


# ═══════════ central feasibility with MoE sizing ════════════════════════════
def test_feasibility_moe_split_makes_gpu_only_selectable():
    dense = AM.feasible_modes("gguf", int(45 * GIB), 24 * GIB, 128 * GIB)
    assert "gpu-only" not in dense               # 45G on a 24G card: eliminated
    moe = AM.feasible_modes("gguf", int(45 * GIB), 24 * GIB, 128 * GIB,
                            moe_split_gpu_bytes=int(3 * GIB))
    assert "gpu-only" in moe                     # the split makes it serveable
    assert "max-gpu" in moe


def test_feasibility_dense_paths_are_unchanged():
    base = AM.feasible_modes("gguf", 5 * GIB, 24 * GIB, 128 * GIB)
    assert base == AM.feasible_modes("gguf", 5 * GIB, 24 * GIB, 128 * GIB,
                                     moe_split_gpu_bytes=None)
    tf = AM.feasible_modes("transformers", int(68 * GIB), 24 * GIB, 124 * GIB)
    assert "gpu-only" not in tf and "max-gpu" not in tf and "ram-only" in tf


# ═══════════ the knob: overrides + spill env wire ═══════════════════════════
def test_override_field_is_first_class_int():
    assert "n_cpu_moe" in OV.ALLOWED_FIELDS
    assert OV._coerce("n_cpu_moe", "999") == 999
    assert OV._coerce("n_cpu_moe", 12) == 12
    assert OV._coerce("n_cpu_moe", "") is None   # clears


def test_apply_spill_maps_and_clears_n_cpu_moe(monkeypatch):
    monkeypatch.delenv("HUGPY_N_CPU_MOE", raising=False)
    A._apply_spill({"n_cpu_moe": 999})
    assert os.environ.get("HUGPY_N_CPU_MOE") == "999"
    assert spill.n_cpu_moe_env() == 999
    A._apply_spill({})                           # absent -> cleared (no leak)
    assert "HUGPY_N_CPU_MOE" not in os.environ
    assert spill.n_cpu_moe_env() is None


def test_n_cpu_moe_rides_the_version_gate():
    assert "n_cpu_moe" in AM.NEW_SPILL_KEYS
    gated, note = AM.gate_spill_for_worker({"n_cpu_moe": 999}, "0.1.150", "old")
    assert gated == {} and note                  # never a silent dead knob
    gated, note = AM.gate_spill_for_worker({"n_cpu_moe": 999}, "0.1.203", "new")
    assert gated == {"n_cpu_moe": 999} and note is None


# ═══════════ the governing-plan resolver ════════════════════════════════════
@pytest.fixture
def plan_rig(monkeypatch, moe_gguf):
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (moe_gguf, 48))
    for env in ("HUGPY_N_GPU_LAYERS", "HUGPY_N_CPU_MOE", "HUGPY_ALLOC_MODE"):
        monkeypatch.delenv(env, raising=False)
    return moe_gguf


def test_plan_auto_eligible_defaults_to_all_experts(plan_rig):
    plan = A._moe_plan_for("m")
    assert plan["n_cpu_moe"] == spill.MOE_ALL_LAYERS
    assert plan["cpu_bytes"] == 16000 and plan["gpu_weight_bytes"] == 3332


def test_plan_explicit_layer_designation_wins(plan_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "-1")        # gpu-only
    assert A._moe_plan_for("m") is None
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "17")        # explicit count
    assert A._moe_plan_for("m") is None
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "off")       # ram-only
    assert A._moe_plan_for("m") is None


def test_plan_mode_engine_wins(plan_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_ALLOC_MODE", "max-ram")
    assert A._moe_plan_for("m") is None


def test_plan_explicit_n_cpu_moe_wins_and_prices_per_layer(plan_rig, monkeypatch):
    monkeypatch.setenv("HUGPY_N_CPU_MOE", "1")
    plan = A._moe_plan_for("m")
    assert plan["n_cpu_moe"] == 1
    assert plan["cpu_bytes"] == 8000             # layer 0's experts only
    monkeypatch.setenv("HUGPY_N_CPU_MOE", "0")   # experts on GPU: no split
    assert A._moe_plan_for("m") is None


def test_plan_dense_is_none(monkeypatch, dense_gguf):
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (dense_gguf, 48))
    for env in ("HUGPY_N_GPU_LAYERS", "HUGPY_N_CPU_MOE", "HUGPY_ALLOC_MODE"):
        monkeypatch.delenv(env, raising=False)
    assert A._moe_plan_for("m") is None
