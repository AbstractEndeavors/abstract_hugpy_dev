"""GPU/CPU spill configuration — how much of a model lives on the GPU.

Two backends, two knobs:

  * llama.cpp (GGUF): ``n_gpu_layers`` passed to ``Llama(...)``.
        -1 = every layer on GPU, 0 = pure CPU, N = first N transformer layers
        on GPU (the rest spill to CPU/RAM). This is the ONLY thing that lights
        up the GPU for GGUF models — without it llama.cpp runs CPU-only.
  * transformers: ``max_memory`` passed to ``from_pretrained(device_map="auto")``,
        e.g. ``{0: "7GiB", "cpu": "32GiB"}`` — accelerate then shards layers to
        fit the per-device budget, spilling the overflow to CPU.

Config comes from environment variables so the worker agent (which owns the
process) can set it per model load without threading new fields through the
resolver/dispatch chain:

    HUGPY_N_GPU_LAYERS   "auto" | "off" | int   llama.cpp layers on GPU
    HUGPY_TENSOR_SPLIT   csv floats             multi-GPU split e.g. "0.7,0.3"
    HUGPY_MAIN_GPU       int                    primary GPU index (llama.cpp)
    HUGPY_GPU_MEM_GIB    float                  per-GPU budget (transformers)
    HUGPY_CPU_MEM_GIB    float                  CPU/RAM budget (transformers)
    HUGPY_N_GPU          int                    #GPUs to spread across

Everything is optional. Default mode is "auto": detect free VRAM, estimate the
model's size, and fit as many layers as will hold — falling back to CPU (0
layers) when no GPU is visible, so a CPU-only host behaves exactly as before.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Optional

logger = logging.getLogger("abstract_hugpy_dev.spill")

# Keep some VRAM headroom for the KV-cache + activations; never budget 100%.
_VRAM_SAFETY = 0.85
# When we can't read a GGUF's real layer count, assume a 7B-ish 32-layer model.
_ASSUMED_LAYERS = 32


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _env_float(name: str) -> Optional[float]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return None


def _env_int(name: str) -> Optional[int]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return None


# ---------------------------------------------------------------------------
# hardware probes (best-effort, never raise)
# ---------------------------------------------------------------------------
def vram_reserve_bytes() -> int:
    """VRAM kept out of every budget (HUGPY_VRAM_RESERVE_GIB, default 1.0).

    The box may have GPU consumers central knows nothing about (a desktop
    session, another app). Reserving a slice at the probe layer means autofit,
    preflights, and heartbeat-fed budgets all leave it alone, while the raw
    per-GPU numbers shown in the console stay truthful."""
    gib = _env_float("HUGPY_VRAM_RESERVE_GIB")
    return int((1.0 if gib is None else gib) * 2**30)


def ram_reserve_bytes() -> int:
    """RAM kept out of every budget (HUGPY_RAM_RESERVE_GIB, default 4.0).

    Same idea as the VRAM reserve: local processes central can't see need
    room, and a load that consumes MemAvailable to the floor gets the whole
    agent OOM-killed mid-request."""
    gib = _env_float("HUGPY_RAM_RESERVE_GIB")
    return int((4.0 if gib is None else gib) * 2**30)


def free_vram_bytes() -> Optional[int]:
    """Budgetable free VRAM on the primary GPU (raw minus the operator
    reserve), or None if no GPU / can't tell."""
    from .._platform.hardware import free_vram_bytes as _free_vram

    raw = _free_vram(_env_int("HUGPY_MAIN_GPU") or 0)
    if raw is None:
        return None
    return max(0, raw - vram_reserve_bytes())


def free_ram_bytes() -> Optional[int]:
    """Budgetable free RAM: raw minus the operator reserve, optionally hard-
    capped by HUGPY_RAM_MAX_GIB (an allocation CEILING for this box — "hugpy
    may use at most N GiB" — regardless of how much is actually free)."""
    from .._platform.hardware import free_ram_bytes as _free_ram

    raw = _free_ram()
    if raw is None:
        return None
    value = max(0, raw - ram_reserve_bytes())
    cap = _env_float("HUGPY_RAM_MAX_GIB")
    if cap is not None:
        value = min(value, int(cap * 2**30))
    return value


def cpu_resident_bytes(model_path: str, n_gpu_layers: int) -> Optional[int]:
    """Rough RAM footprint of a GGUF load: the weight share NOT offloaded."""
    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return None
    if n_gpu_layers == -1:
        return 0
    if n_gpu_layers <= 0:
        return file_bytes
    total = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    frac = min(1.0, max(0.0, 1.0 - (n_gpu_layers / max(total, 1))))
    return int(file_bytes * frac)


def _gguf_layer_count(model_path: str) -> Optional[int]:
    """Read ``*.block_count`` from a GGUF header. Best-effort; None on any issue."""
    try:
        import struct

        with open(model_path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", fh.read(4))[0]
            if version < 2:
                return None
            struct.unpack("<Q", fh.read(8))[0]              # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n).decode("utf-8", "replace")

            # Minimal GGUF value-type reader — enough to scan the KV table.
            def read_val(t: int):
                simple = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i",
                          5: "<I", 6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
                if t in simple:
                    fmt = simple[t]
                    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
                if t == 8:                                   # string
                    return read_str()
                if t == 9:                                   # array
                    et = struct.unpack("<I", fh.read(4))[0]
                    n = struct.unpack("<Q", fh.read(8))[0]
                    return [read_val(et) for _ in range(n)]
                raise ValueError(f"unknown gguf type {t}")

            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack("<I", fh.read(4))[0]
                val = read_val(vtype)
                if key.endswith(".block_count"):
                    return int(val)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# llama.cpp (GGUF)
# ---------------------------------------------------------------------------
def autofit_gpu_layers(model_path: str,
                       free_vram: Optional[int] = None) -> int:
    """How many GGUF layers fit in free VRAM. -1 (all) when the whole file fits."""
    if free_vram is None:
        free_vram = free_vram_bytes()
        # Operator VRAM budget (the console's "VRAM budget…" mode / spill
        # gpu_mem_gib) caps the autofit. GGUF loads ignored this before — the
        # knob only reached the transformers path, so a per-model budget on a
        # GGUF worker silently did nothing.
        gpu_gib = _env_float("HUGPY_GPU_MEM_GIB")
        if gpu_gib is not None:
            cap = int(gpu_gib * 2**30)
            free_vram = min(free_vram, cap) if free_vram else cap
    if not free_vram:                       # no GPU / unknown -> CPU only
        return 0

    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return 0
    if file_bytes <= 0:
        return 0

    # Weights are not the whole story: llama_context still needs VRAM AFTER
    # the weights land (KV cache scales with n_ctx, plus compute-graph
    # buffers). The flat safety factor leaves only ~3.5 GB on a 24 GB card —
    # a 70B uploads ~20 GB of weights and then dies with "Failed to create
    # llama_context". Reserve an explicit context allowance under the margin.
    ctx_reserve = int((_env_float("HUGPY_VRAM_CTX_RESERVE_GIB") or 2.5) * 2**30)
    budget = int(free_vram * _VRAM_SAFETY) - ctx_reserve
    if budget <= 0:
        return 0
    if budget >= file_bytes:
        return -1                           # everything fits on the GPU

    total_layers = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    # Weights dominate the file; approximate per-layer cost as an even split.
    per_layer = file_bytes / max(total_layers, 1)
    fit = int(budget // per_layer)
    fit = max(0, min(fit, total_layers))
    logger.info(
        "autofit gguf: file=%.1fGiB free=%.1fGiB -> %d/%d layers on GPU",
        file_bytes / 2**30, free_vram / 2**30, fit, total_layers,
    )
    return fit


def gguf_gpu_layers(model_path: str) -> int:
    """Resolve n_gpu_layers for a GGUF model from env (+autofit)."""
    raw = _env("HUGPY_N_GPU_LAYERS")
    if raw is None or raw.lower() == "auto":
        return autofit_gpu_layers(model_path)
    if raw.lower() in ("off", "cpu", "none"):
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("bad HUGPY_N_GPU_LAYERS=%r; using autofit", raw)
        return autofit_gpu_layers(model_path)


def tensor_split() -> Optional[list[float]]:
    raw = _env("HUGPY_TENSOR_SPLIT")
    if not raw:
        return None
    try:
        parts = [float(x) for x in raw.split(",") if x.strip()]
        return parts or None
    except ValueError:
        logger.warning("bad HUGPY_TENSOR_SPLIT=%r; ignoring", raw)
        return None


def main_gpu() -> Optional[int]:
    return _env_int("HUGPY_MAIN_GPU")


def rpc_servers() -> Optional[str]:
    """Comma-separated "host:port" of llama.cpp rpc-servers to shard onto, or None.

    Set (as a per-request spill override) by central's allocator when it decides
    to shard a model across multiple GPUs on different machines.
    """
    raw = _env("HUGPY_RPC_SERVERS")
    return raw.strip() if raw and raw.strip() else None


def _binding_supports_rpc() -> bool:
    """Whether this llama-cpp-python build accepts ``Llama(rpc_servers=…)``.

    The param existed in 0.2.78–0.2.90 and was dropped in the 0.3.x rewrite —
    passing it there raises TypeError. Shard leads on a >=0.3 binding must be
    served via ``llama-server --rpc …`` (the serve layer's extra_args) instead
    of the in-process runner.
    """
    try:
        import inspect
        from llama_cpp import Llama
        return "rpc_servers" in inspect.signature(Llama.__init__).parameters
    except Exception:
        return False


def llama_kwargs(model_path: str) -> dict[str, Any]:
    """Spill kwargs for ``Llama(...)``. Always includes n_gpu_layers.

    When ``HUGPY_RPC_SERVERS`` is set we're the LEAD of a cross-machine shard:
    pass ``rpc_servers`` and force ``n_gpu_layers=-1`` (offload ALL layers across
    the pooled GPUs — the whole point is to never touch CPU). ``tensor_split``
    (also supplied by the allocator) then weights layers across [local, *rpc].
    """
    rpc = rpc_servers()
    if rpc and not _binding_supports_rpc():
        logger.warning(
            "HUGPY_RPC_SERVERS=%s set but this llama-cpp-python has no "
            "rpc_servers param (dropped in 0.3.x) — ignoring the shard plan "
            "and loading locally. Serve shard leads via llama-server --rpc "
            "(cfg.extra['llama_extra_args']) instead.", rpc,
        )
        rpc = None
    if rpc:
        kwargs: dict[str, Any] = {"n_gpu_layers": -1, "rpc_servers": rpc}
    else:
        kwargs = {"n_gpu_layers": gguf_gpu_layers(model_path)}
    ts = tensor_split()
    if ts is not None:
        kwargs["tensor_split"] = ts
    mg = main_gpu()
    if mg is not None:
        kwargs["main_gpu"] = mg
    return kwargs


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------
def _gib(n: float) -> str:
    return f"{n:.2f}GiB"


def transformers_max_memory() -> Optional[dict]:
    """Build a ``max_memory`` map for device_map='auto', or None to skip.

    Explicit env budgets win; otherwise autofit from detected free VRAM/RAM.
    Returns None when no GPU is visible (let transformers stay on CPU).
    """
    gpu_gib = _env_float("HUGPY_GPU_MEM_GIB")
    cpu_gib = _env_float("HUGPY_CPU_MEM_GIB")
    n_gpu = _env_int("HUGPY_N_GPU") or 1

    if gpu_gib is None:
        fv = free_vram_bytes()
        if not fv:
            return None                     # no GPU -> no spill map
        gpu_gib = (fv * _VRAM_SAFETY) / 2**30

    if cpu_gib is None:
        fr = free_ram_bytes()
        cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0

    mm: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
    mm["cpu"] = _gib(cpu_gib)
    logger.info("transformers max_memory=%s", mm)
    return mm


def describe() -> dict[str, Any]:
    """Human-readable snapshot of the current spill config (for heartbeats/UI)."""
    return {
        "mode": (_env("HUGPY_N_GPU_LAYERS") or "auto"),
        "n_gpu_layers_env": _env("HUGPY_N_GPU_LAYERS"),
        "gpu_mem_gib": _env_float("HUGPY_GPU_MEM_GIB"),
        "cpu_mem_gib": _env_float("HUGPY_CPU_MEM_GIB"),
        "tensor_split": tensor_split(),
        "free_vram_bytes": free_vram_bytes(),
    }
