# deepcoder/config.py
import os
import os.path as osp
import threading
from dataclasses import dataclass
from typing import Any, Optional

from .imports import (
    ensure_model,
    get_model_config,
    resolve_model_source,
    get_logFile,
    require,
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TokenEvent,
    DEFAULT_LOCAL_FILES_ONLY
)
logger = get_logFile("deepcoder")
_SENTINEL = object()


@dataclass(frozen=True)
class DeepCoderConfig:
    model_dir: str
    device: str
    torch_dtype: Any
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY
    # SECURITY: loading a model with custom code (auto_map / modeling_*.py)
    # EXECUTES the author's Python on load — i.e. arbitrary RCE from an untrusted
    # HF repo. OFF by default; only ever True via an explicit opt-in (per call, or
    # the operator switch HUGPY_TRUST_REMOTE_CODE). Never defaults on.
    trust_remote_code: bool = False

    max_new_tokens_cap: int = 16000

    cpu_threads: Optional[int] = None
    cpu_interop_threads: Optional[int] = 1
    max_concurrent_generations: int = 1

    def cache_key(self) -> tuple:
        return (
            self.model_dir,
            self.device,
            str(self.torch_dtype),
            self.use_quantization,
            self.use_flash_attention,
            self.local_files_only,
            self.trust_remote_code,
            self.max_new_tokens_cap,
            self.cpu_threads,
            self.cpu_interop_threads,
            self.max_concurrent_generations,
        )


def pick_device_and_dtype(torch, device: Optional[str], dtype) -> tuple[str, Any]:
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is not None:
        return chosen, dtype

    if chosen == "cuda":
        return chosen, (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return chosen, torch.bfloat16

    return chosen, torch.float32


def build_deepcoder_runtime(
    *,
    model_key: str = "deepcoder",
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    use_flash_attention: bool = False,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    max_new_tokens_cap: int = 16000,
    max_concurrent_generations: int = 1,
    cpu_threads: Optional[int] = None,
    cpu_interop_threads: Optional[int] = 1,
    auto_download: bool = True,
) -> DeepCoderConfig:
    torch = require("torch", reason="DeepCoder requires PyTorch")
    get_model_config(model_key)

    if auto_download:
        model_dir = str(ensure_model(model_key))
    else:
        model_dir = resolve_model_source(model_key)

        if not osp.exists(model_dir):
            raise FileNotFoundError(
                f"Model {model_key!r} is not on disk and auto_download=False; "
                f"call ensure_model({model_key!r}) first."
            )

    chosen_device, chosen_dtype = pick_device_and_dtype(torch, device, torch_dtype)

    # SECURITY: trust_remote_code lets a model repo run arbitrary Python on load.
    # Default OFF; True only via an explicit opt-in here OR the operator switch
    # HUGPY_TRUST_REMOTE_CODE (1/true/yes/on). Never on by default.
    allow_remote_code = bool(trust_remote_code) or (
        os.environ.get("HUGPY_TRUST_REMOTE_CODE", "").strip().lower()
        in ("1", "true", "yes", "on")
    )

    return DeepCoderConfig(
        model_dir=model_dir,
        device=chosen_device,
        torch_dtype=chosen_dtype,
        use_quantization=use_quantization and chosen_device == "cuda",
        use_flash_attention=use_flash_attention and chosen_device == "cuda",
        local_files_only=local_files_only,
        trust_remote_code=allow_remote_code,
        max_new_tokens_cap=max_new_tokens_cap,
        max_concurrent_generations=max_concurrent_generations,
        cpu_threads=cpu_threads,
        cpu_interop_threads=cpu_interop_threads,
    )
build_deepcoder_config = build_deepcoder_runtime


class CancelStoppingCriteria:
    def __init__(self, cancel: threading.Event):
        self._cancel = cancel

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancel.is_set()
