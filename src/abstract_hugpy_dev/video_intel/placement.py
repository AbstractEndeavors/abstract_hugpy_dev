"""Placement projection — WHERE a media-bus job physically executes.

Today external execution (the ae worker, ComfyUI, the identity-render service on
:9750) is invisible to the console: a job shows a status but never WHERE it runs.
This module closes that gap. It is a read-only, central-only, FAIL-OPEN join of a
bus job to its physical execution locus, in priority order:

  1. ACTIVE GPU RESERVATION (the live truth for a heavy run) — from the central
     reservation registry, keyed by run_id (= the bus job_id). Carries the real
     worker_id + reserved bytes; overlaid with the template's device/process.
  2. RESERVATION TEMPLATE (the static hint for a reservable heavy task with no
     active claim yet — e.g. a queued/awaiting_capacity render): the representative
     (peak-driving) stage's host / gpu device / process.
  3. LIGHT-TASK LOCUS (crop/frame/audio extract + the legacy generate_* path):
     a small static map — these carry no reservation template (VIDEO-TASK-SEQUENCES
     §0/§3.9: CPU ffmpeg on P-central, or the central orchestrator).
  4. else None (unknown/unmapped task) — never fabricated.

Every value is omit-when-unset: a key is present only when we actually know it.
Any read error (reservation store hiccup, overlay problem) degrades to a lower
tier or None — placement is observability, it must never 500 a surface.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Light / orchestrator bus jobs that carry NO reservation template. Locus per
# VIDEO-TASK-SEQUENCES §0 (topology) + §3.9 (CPU-only stages). ffmpeg tasks run
# entirely on P-central's CPU; the legacy generate_* orchestrators run their fat
# loop on P-central (the per-frame GPU sub-calls are a separate dispatch, not this
# bus job) — so central is the honest locus for the bus job itself.
_LIGHT_LOCUS: Dict[str, Dict[str, Optional[str]]] = {
    "crop": {"host": "central", "gpu": None, "process": "ffmpeg"},
    "frame_extract": {"host": "central", "gpu": None, "process": "ffmpeg"},
    "audio_extract": {"host": "central", "gpu": None, "process": "ffmpeg"},
    "generate_image": {"host": "central", "gpu": None, "process": "P-central"},
    "generate_scene": {"host": "central", "gpu": None, "process": "P-central"},
}


def _prune(d: Dict[str, Any]) -> Dict[str, Any]:
    """Omit-when-unset: drop None values so a placement never carries a fabricated
    (null) field. `source` is always kept (it is never None here)."""
    return {k: v for k, v in d.items() if v is not None or k == "source"}


def _split_gpu(affinity: Any) -> Optional[str]:
    """A template stage's ``gpu_affinity`` -> the device label. "ae:cuda:0" ->
    "cuda:0"; a bare host label ("ae") or None -> None (no known device)."""
    if not affinity or not isinstance(affinity, str):
        return None
    if ":" in affinity:
        return affinity.split(":", 1)[1] or None
    return None


def _representative_stage(template: Any):
    """The stage that DRIVES the reservation peak — the one the run is really 'at'
    for placement purposes: the max-VRAM stage; else the first exclusive stage;
    else the first stage. None for an empty template."""
    stages = list(getattr(template, "stages", ()) or ())
    if not stages:
        return None
    numbered = [(s, s.vram_bytes()) for s in stages]
    numbered = [(s, v) for (s, v) in numbered if v is not None]
    if numbered:
        numbered.sort(key=lambda sv: -sv[1])
        return numbered[0][0]
    for s in stages:
        if getattr(s, "exclusive", False):
            return s
    return stages[0]


def _template_placement(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Tier 2/3: the static per-task hint (reservation template, else light-task
    locus map, else None). Fail-open to None on any import/overlay error."""
    if not name:
        return None
    try:
        from .reservation.templates import load_template, is_reservable
    except Exception:  # noqa: BLE001 — no engine present -> only the light map
        return _light_placement(name)
    try:
        if not is_reservable(name):
            return _light_placement(name)
        tmpl = load_template(name)
        if tmpl is None:
            return _light_placement(name)
        st = _representative_stage(tmpl)
        if st is None:
            host = getattr(tmpl, "gpu_affinity", None)
            return _prune({"source": "template", "host": host})
        return _prune({
            "source": "template",
            "host": getattr(st, "host", None),
            "gpu": _split_gpu(getattr(st, "gpu_affinity", None)),
            "process": getattr(st, "process", None),
        })
    except Exception:  # noqa: BLE001
        logger.debug("template placement failed for %s", name, exc_info=True)
        return _light_placement(name)


def _light_placement(name: Optional[str]) -> Optional[Dict[str, Any]]:
    locus = _LIGHT_LOCUS.get(name or "")
    if locus is None:
        return None
    return _prune({"source": "template", **locus})


def _reservation_placement(job_id: str,
                           name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Tier 1: the ACTIVE reservation claim for this run, overlaid with the
    template's device/process. None when there is no live claim (or on error)."""
    try:
        from .reservation.registry import reservation_registry as rr
    except Exception:  # noqa: BLE001
        return None
    try:
        row = rr.get(job_id)
    except Exception:  # noqa: BLE001 — a store hiccup is not a placement
        return None
    if not row or row.get("state") != "active":
        return None
    peak = row.get("peak_bytes")
    out: Dict[str, Any] = {
        "source": "reservation",
        "worker_id": row.get("worker_id") or None,
        # The registry's ``gpu`` is the task-level affinity LABEL — the box ("ae"),
        # which is the host. The device + process come from the template overlay.
        "host": row.get("gpu") or None,
        "reserved_bytes": (int(peak) if peak else None),
    }
    tmpl_pl = _template_placement(name or row.get("task"))
    if tmpl_pl:
        if out.get("host") is None and tmpl_pl.get("host") is not None:
            out["host"] = tmpl_pl["host"]
        for k in ("gpu", "process"):
            if tmpl_pl.get(k) is not None:
                out[k] = tmpl_pl[k]
    return _prune(out)


def job_placement(job_id: str,
                  name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The placement object for a bus job, or None.

    Shape (omit-when-unset): ``{source: "reservation"|"template", host?, worker_id?,
    gpu?, process?, reserved_bytes?}``. An active reservation wins over the static
    template hint. FAIL-OPEN: any error anywhere -> None (never raises)."""
    try:
        pl = _reservation_placement(job_id, name)
        if pl:
            return pl
        return _template_placement(name)
    except Exception:  # noqa: BLE001
        logger.debug("job_placement failed for %s", job_id, exc_info=True)
        return None
