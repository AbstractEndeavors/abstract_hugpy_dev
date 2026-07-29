"""review/criteria.py — what a model is being reviewed *for*.

A ReviewCriteria is a saved, re-runnable question: "find me text-generation
models that fit the 3090 at 16k context, are a real capability step over what
I already run, and actually generate at a usable speed." It drives both the
cheap metadata screen (screen.py) and, for survivors, the load test (smoke.py).

Deliberately plain dataclasses + dicts: this module is imported by the Flask
app, the CLI and a systemd timer, and must never drag in pydantic version
skew (see _compat_pydantic.py for why that hurts here).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

# One 24 GB card on `ae` today. Overridable so the same criteria file can be
# evaluated for a different box in the fleet without editing it.
DEFAULT_VRAM_BYTES = int(os.environ.get("REVIEW_VRAM_BYTES") or 24 * 1024**3)

# Leave the GPU room for the desktop/compositor, CUDA context and fragmentation.
# A model that "just fits" at 100% never actually loads.
VRAM_HEADROOM_FRACTION = 0.90


@dataclass
class ReviewCriteria:
    """The question. `name` identifies it in the store and on the timer."""

    name: str
    query: str = ""                          # HF search text
    task: str | None = "text-generation"     # pipeline_tag filter
    library: str | None = None               # HF `filter` (e.g. "gguf")
    author: str | None = None

    # ── fit / runtime cost ────────────────────────────────────────────────
    vram_bytes: int = DEFAULT_VRAM_BYTES
    target_context: int = 16384              # context to size the KV cache for
    min_context: int = 8192                  # reject models that can't reach it
    max_total_bytes: int | None = None       # hard disk cap for one download
    require_gguf: bool = True                # llama.cpp is the fleet's runtime
    allowed_quants: list[str] = field(default_factory=lambda: [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "IQ4_XS"])
    min_tokens_per_sec: float = 8.0          # smoke-test floor for "usable"

    # ── capability / quality ──────────────────────────────────────────────
    min_downloads: int = 500
    min_trust_tier: int = 0                  # 0 any, 1 community+, 2 first-party
    required_tags: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    max_age_days: int | None = None          # only recently-updated repos
    min_params: int | None = None            # reject toys
    max_params: int | None = None
    # Models already in the fleet. A candidate that is a fine-tune of one of
    # these is flagged as lineage — usually the interesting kind of candidate.
    incumbents: list[str] = field(default_factory=list)

    # ── pipeline behaviour ────────────────────────────────────────────────
    pool_limit: int = 60                     # candidates pulled from HF search
    max_downloads_per_run: int = 2           # disk/bandwidth guard for the timer
    smoke_test: bool = True
    judge: bool = True                       # ask a hugpy-agent for a read

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewCriteria":
        known = {f for f in cls.__dataclass_fields__}          # ignore extras
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    @property
    def usable_vram_bytes(self) -> int:
        return int(self.vram_bytes * VRAM_HEADROOM_FRACTION)


def criteria_dir() -> str:
    d = os.environ.get("REVIEW_CRITERIA_DIR") or os.path.expanduser(
        "~/.config/hugpy/review")
    os.makedirs(d, exist_ok=True)
    return d


def save_criteria(c: ReviewCriteria) -> str:
    path = os.path.join(criteria_dir(), f"{c.name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(c.to_dict(), fh, indent=2)
    os.replace(tmp, path)                       # atomic: the timer may be reading
    return path


def load_criteria(name: str) -> ReviewCriteria:
    path = os.path.join(criteria_dir(), f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return ReviewCriteria.from_dict(json.load(fh))


def list_criteria() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(criteria_dir())
                      if f.endswith(".json"))
    except OSError:
        return []
