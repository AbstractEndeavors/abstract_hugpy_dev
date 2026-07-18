"""Slot scheduler — assigns models to the root-free slot supervisors.

The pool is a fixed set of slot control URLs (``SLOT_COUNT`` slots starting at
``SLOT_PORT_BASE``). On demand it routes a model to a slot:

    1. a slot already serving that model      -> reuse it
    2. an idle slot                            -> load the model there (the slot
                                                  autofits GPU layers from the
                                                  VRAM still free, so slots fill
                                                  the card in order)
    3. all slots busy                          -> return None; the caller routes
                                                  the overflow through swap

Pure HTTP to the supervisors — no systemctl, no root.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("abstract_hugpy_dev.slots")


_LOGGED_COUNT = False

# Tiers-v2 eviction policy hook: a callable mk -> bool ("is this model
# on-demand, i.e. may it be bumped from its slot for a caller?"). Registered
# by the WORKER agent from its runtime settings; None (e.g. bare central)
# keeps the historical behavior: all-busy -> None -> swap fallback.
_EVICTION_POLICY = None

# Tiers-v3 residency lookup hook: a callable mk -> "serving"|"on-demand"|
# "static". Registered by the WORKER agent (its _residency). Lets the
# scheduler tell a merely-busy pool ("return None, swap handles it") from a
# STATIC-LOCKED pool, where every seat is immovable and the load must fail
# with a clear error instead. None (bare central) skips the check.
_RESIDENCY_LOOKUP = None

# Real-VRAM ceiling hook (Fix A, 2026-07-15): a callable mk -> bool answering
# "will loading THIS model keep the card at/under the ~90% ceiling given REAL
# current free VRAM (torch.cuda.mem_get_info — ComfyUI-visible, not managed-
# model bookkeeping)?". Registered by the WORKER agent (_worker_slot_fit_check).
# The gap this closes: an idle slot on a card 95%-full from a SEPARATE process
# (ComfyUI) would happily /load into the seat, then the child silently offloads
# fewer layers or OOMs — because slot routing keyed on slot-OCCUPANCY, never on
# real device pressure. When registered and it says NO (would breach ceiling),
# endpoint_for evicts the coldest on-demand occupant(s) via the SAME LRU
# mechanism the all-busy branch uses and re-checks, until the gate passes or
# nothing is evictable (then it proceeds anyway — honest-degrade, never HANG a
# legitimate request; the child's autofit does its best). None (bare central,
# no-GPU box, or a gate that can't measure) => byte-identical to today: the
# gate is skipped, occupancy-only routing stands.
_FIT_CHECK = None


def set_eviction_policy(fn) -> None:
    global _EVICTION_POLICY
    _EVICTION_POLICY = fn


def set_residency_lookup(fn) -> None:
    global _RESIDENCY_LOOKUP
    _RESIDENCY_LOOKUP = fn


def set_fit_check(fn) -> None:
    """Register the real-VRAM ceiling gate: ``fn(model_key) -> bool``, True when
    loading the model keeps the card at/under the ~90% ceiling given real current
    free VRAM. None disables the ceiling gate (bare central / no-GPU / can't
    measure) — occupancy-only routing, byte-identical to before."""
    global _FIT_CHECK
    _FIT_CHECK = fn


# CROSS-TIER make-room (slice 10): the slot ceiling loop above evicts only SLOT
# occupants — it is blind to an IN-PROCESS transformers resident squatting the
# card. This hook (registered by the worker) evicts ALL permissible residents
# (in-process included) from the pid-registry measured truth. Called once the
# slot-side eviction is exhausted, so a slot load can also reclaim VRAM held by a
# sibling in-process model. None -> the historical slot-only path.
_MAKE_ROOM = None


def set_make_room(fn) -> None:
    """Register the cross-tier VRAM make-room (slice 10): ``fn(model_key) -> dict``.
    None -> slot-only ceiling eviction, byte-identical to before."""
    global _MAKE_ROOM
    _MAKE_ROOM = fn


def _slot_count() -> int:
    global _LOGGED_COUNT
    raw = os.environ.get("SLOT_COUNT")
    try:
        n = max(0, int(raw)) if raw is not None else 2
    except ValueError:
        n = 2
    if not _LOGGED_COUNT:
        _LOGGED_COUNT = True
        # De-silence the env-layering ghost: a systemd drop-in can override the
        # unit's SLOT_COUNT and silently resurrect slots every restart (op's
        # limits.conf SLOT_COUNT=2). Record the EFFECTIVE value + raw env once
        # per process so the journal shows the truth.
        logger.info("SLOT_COUNT effective=%d (env=%r) — if this differs from the "
                    "unit file, a drop-in is overriding it", n, raw)
    return n


def slots_enabled() -> bool:
    # Per-box "never serve locally" policy: force the slot pool off regardless of
    # SLOT_COUNT, so serve routing/get_llama_runner never load a model into a
    # local slot (the path that spawned the OOM'ing llama-server on central).
    # Default off === today's behavior; workers never set the flag. See
    # .policy.no_local_serving.
    from .policy import no_local_serving
    if no_local_serving():
        return False
    return _slot_count() > 0


def _slot_host() -> str:
    return os.environ.get("SLOT_ADVERTISE") or os.environ.get("SLOT_HOST_ADDR") or "127.0.0.1"


def _slot_port_base() -> int:
    try:
        return int(os.environ.get("SLOT_PORT_BASE", "8101"))
    except ValueError:
        return 8101


def slot_urls() -> list[str]:
    # Under the no-local-serving policy the pool has no targets, so a directly
    # constructed SlotPool().endpoint_for() also returns None (defense-in-depth
    # alongside slots_enabled). Default off === today's behavior.
    from .policy import no_local_serving
    if no_local_serving():
        return []
    host, base = _slot_host(), _slot_port_base()
    return [f"http://{host}:{base + i}" for i in range(_slot_count())]


def _get(url: str, timeout: float = 3.0) -> dict:
    import httpx
    return httpx.get(url, timeout=timeout).json()


def _post(url: str, body: dict, timeout: float) -> dict:
    import httpx
    return httpx.post(url, json=body, timeout=timeout).json()


class SlotPool:
    def __init__(self, urls: list[str] | None = None):
        self.urls = urls if urls is not None else slot_urls()

    def statuses(self) -> list[dict]:
        out = []
        for url in self.urls:
            try:
                status = _get(url + "/status")
                status["_control"] = url
            except Exception as exc:  # a down slot shouldn't break scheduling
                status = {"_control": url, "healthy": False, "model_key": None,
                          "error": str(exc)}
            out.append(status)
        return out

    def _ceiling_ok(self, model_key: str) -> bool:
        """Whether loading ``model_key`` keeps the card at/under the real-VRAM
        ceiling. True (fits) when no ceiling gate is registered — bare central /
        no-GPU / unmeasurable all degrade to the historical occupancy-only path.
        A gate that raises is treated as "can't tell" => True (never block a load
        because the measurement broke)."""
        if _FIT_CHECK is None:
            return True
        try:
            return bool(_FIT_CHECK(model_key))
        except Exception:  # noqa: BLE001 — a broken gate must not crash serving
            return True

    def _evict_coldest_on_demand(self, statuses: list[dict],
                                 model_key: str) -> "dict | None":
        """Evict the single LRU idle on-demand slot occupant (the SAME candidate
        rule the all-busy promotion branch uses) and return the evicted status
        dict, or None when nothing is evictable. Reuses ``_EVICTION_POLICY`` (the
        worker answers True only for on-demand — never static, never a busy slot,
        never the incoming model). No-op (None) when no eviction policy is
        registered."""
        if _EVICTION_POLICY is None:
            return None
        candidates = [s for s in statuses
                      if s.get("model_key") and s.get("healthy")
                      and not s.get("busy")
                      and s.get("model_key") != model_key]
        try:
            candidates = [s for s in candidates
                          if _EVICTION_POLICY(s["model_key"])]
        except Exception:  # noqa: BLE001 — a broken policy must not crash serving
            return None
        candidates.sort(key=lambda s: s.get("last_used") or 0)
        for victim in candidates:
            try:
                self.unload(victim["_control"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("ceiling evict failed on %s: %s",
                               victim["_control"], exc)
                continue
            return victim
        return None

    def endpoint_for(self, model_key: str, *, load_timeout: float = 900.0,
                     opts: dict | None = None) -> str | None:
        """Resolve (and if needed load) the slot serving ``model_key``.

        ``opts`` is an optional per-load compute dict (n_gpu_layers, ctx,
        threads, cpus, gpu) applied when the model is loaded into a free slot.
        Returns the slot's inference endpoint, or None when every slot is busy
        with a different model (caller should fall back to swap).
        """
        # The opts actually applied to the /load. The cross-tier make-room hook
        # (below) may hand back a PARTIAL-offload plan for an oversize GGUF — the
        # honest layers-that-fit count — which we thread in here so the slot child
        # launches with --n-gpu-layers N (not the shard-blind autofit -1).
        eff_opts = dict(opts or {})
        statuses = self.statuses()

        # 1. already serving OR currently loading it — reuse, never load a 2nd
        #    copy. A slot mid-load has model_key set but healthy=False; if we
        #    returned that endpoint immediately the caller would proxy to a child
        #    that isn't up yet and get a 503. So WAIT for the loading slot to go
        #    healthy (coalescing concurrent requests onto the one load), just like
        #    loading into a fresh slot blocks. A down slot reports model_key=None.
        for s in statuses:
            if s.get("model_key") != model_key:
                continue
            ep = s.get("endpoint") or s["_control"]
            if s.get("healthy"):
                return ep
            deadline = time.time() + load_timeout
            while time.time() < deadline:
                time.sleep(2.0)
                try:
                    st = _get(s["_control"] + "/status")
                except Exception:
                    break                       # slot went away — reload below
                if st.get("healthy"):
                    return st.get("endpoint") or ep
                if not st.get("model_key"):
                    break                       # load aborted/failed — reload below
            break                               # not ready in time — fall through

        # 1b. Real-VRAM CEILING gate (Fix A, 2026-07-15). Before we seat this
        #     model — whether into an idle slot below or by promotion — check
        #     that loading it keeps the card at/under the ~90% ceiling given REAL
        #     current free VRAM (not slot-occupancy count). This catches the ae
        #     case: a card 95%-full from a SEPARATE ComfyUI process but with an
        #     idle slot would otherwise /load happily, then OOM/under-offload.
        #     When over ceiling, evict the coldest on-demand occupant(s) via the
        #     SAME LRU mechanism the all-busy branch uses (re-reading statuses so
        #     each re-check sees the freed seat) until the gate passes OR nothing
        #     is evictable. Nothing evictable + still over ceiling => proceed
        #     anyway (honest-degrade: the child's autofit does its best; we never
        #     HANG a legitimate request), with a clear warning. No-op when no
        #     ceiling gate is registered (bare central / no-GPU / can't measure).
        if not self._ceiling_ok(model_key):
            while not self._ceiling_ok(model_key):
                victim = self._evict_coldest_on_demand(statuses, model_key)
                if victim is None:
                    # Slot-side eviction exhausted. CROSS-TIER (slice 10): an
                    # IN-PROCESS transformers resident (invisible to the slot
                    # scheduler) may still be squatting the card — the make-room
                    # hook evicts ALL permissible residents from the pid-registry
                    # measured truth. If it evicts something, re-check the ceiling;
                    # if it REFUSES (nothing left to evict), honest-degrade below.
                    if _MAKE_ROOM is not None:
                        try:
                            verdict = _MAKE_ROOM(model_key)
                        except Exception:  # noqa: BLE001 — never hang a request
                            verdict = None
                        # PARTIAL-offload admission (autofit's hybrid contract): the
                        # full weights don't fit even after eviction, but the honest
                        # layers-that-fit plan admits. Launch the child with that
                        # exact n_gpu_layers and stop looping the (full-need) ceiling
                        # check — it can never pass, and re-looping would spin.
                        if (isinstance(verdict, dict)
                                and verdict.get("action") == "partial"
                                and verdict.get("n_gpu_layers") is not None):
                            eff_opts["n_gpu_layers"] = verdict["n_gpu_layers"]
                            logger.info(
                                "VRAM ceiling: %s admitted as a PARTIAL offload — "
                                "%s/%s layers on GPU (%s%%); launching child with "
                                "--n-gpu-layers %s", model_key,
                                verdict["n_gpu_layers"],
                                (verdict.get("partial") or {}).get("total_layers"),
                                verdict.get("gpu_pct"), verdict["n_gpu_layers"])
                            break
                        if isinstance(verdict, dict) and verdict.get("evicted"):
                            statuses = self.statuses()
                            continue         # re-check the ceiling with the freed room
                    logger.warning(
                        "VRAM ceiling: loading %s would exceed the real-VRAM "
                        "ceiling and nothing on-demand is evictable (slot or "
                        "in-process) — proceeding anyway (autofit will spill/"
                        "offload; not hanging the request)", model_key)
                    break
                logger.info(
                    "VRAM ceiling: evicted idle on-demand %s from %s to keep %s "
                    "under the real-VRAM ceiling", victim["model_key"],
                    victim["_control"], model_key)
                statuses = self.statuses()   # re-read: the seat is now free

        # 2. an idle slot (reachable, nothing loaded)
        for s in statuses:
            if "error" in s:
                continue
            if not s.get("model_key"):
                body = {"model_key": model_key, **eff_opts}
                resp = _post(s["_control"] + "/load", body, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(f"slot load failed: {resp['error']}")
                return resp.get("endpoint") or s["_control"]

        # 3. everything busy — tiers-v2 promotion: bump the LRU *idle*
        #    occupant whose model is itself on-demand (policy hook). Never a
        #    busy slot, never a serving/static/pinned model (the policy
        #    returns False for those), never for a model already handled
        #    above. STATIC occupants are immovable by construction: the
        #    worker's policy answers True only for on-demand.
        if _EVICTION_POLICY is not None:
            candidates = [s for s in statuses
                          if s.get("model_key") and s.get("healthy")
                          and not s.get("busy")
                          and s.get("model_key") != model_key]
            try:
                candidates = [s for s in candidates
                              if _EVICTION_POLICY(s["model_key"])]
            except Exception:  # noqa: BLE001 — a broken policy must not crash serving
                candidates = []
            candidates.sort(key=lambda s: s.get("last_used") or 0)
            for victim in candidates:
                logger.info(
                    "slot promotion: evicting idle on-demand %s from %s to "
                    "load %s", victim["model_key"], victim["_control"], model_key)
                try:
                    self.unload(victim["_control"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("promotion evict failed on %s: %s",
                                   victim["_control"], exc)
                    continue
                body = {"model_key": model_key, **eff_opts}
                resp = _post(victim["_control"] + "/load", body, load_timeout)
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(f"slot load failed: {resp['error']}")
                return resp.get("endpoint") or victim["_control"]

        # Static-lock check (tiers v3): when EVERY slot's occupant is static
        # (locked to its seat — never swapped out), returning None would lie
        # ("busy, try swap") about a pool that can never free a seat. Fail the
        # load with a clear, actionable error instead. A merely-busy pool
        # (any serving/on-demand/unknown occupant, or a down slot) keeps the
        # historical None -> swap/in-process fallback.
        if _RESIDENCY_LOOKUP is not None and statuses:
            occupants = [s.get("model_key") for s in statuses]
            if all(occupants):
                try:
                    all_static = all(_RESIDENCY_LOOKUP(mk) == "static"
                                     for mk in occupants)
                except Exception:  # noqa: BLE001 — a broken lookup must not crash serving
                    all_static = False
                if all_static:
                    raise RuntimeError(
                        f"all slots are static-locked — cannot seat {model_key}; "
                        "unlock a static model or raise the slot count")
        return None

    def load(self, model_key: str, control_url: str, *, timeout: float = 900.0) -> dict:
        return _post(control_url + "/load", {"model_key": model_key}, timeout)

    def unload(self, control_url: str) -> dict:
        return _post(control_url + "/unload", {}, 30.0)

    def overview(self) -> list[dict]:
        return self.statuses()


# --------------------------------------------------------------------------- #
# one-time install: N generic slot services (systemd template)                #
# --------------------------------------------------------------------------- #
def render_slot_unit(python_bin: str | None = None, user: str | None = None,
                     group: str | None = None, main_gpu: str | None = None) -> str:
    """A systemd TEMPLATE unit: ``abstract-hugpy-slot@.service``.

    The instance number is the slot id (``systemctl enable --now
    abstract-hugpy-slot@1``); the agent derives its port from SLOT_ID. Installed
    ONCE; thereafter the app drives slots over HTTP — no per-model units, no
    sudo at request time.
    """
    import sys
    python_bin = python_bin or sys.executable
    user = user or os.environ.get("LLAMA_SERVICE_USER", "solcatcher")
    group = group or os.environ.get("LLAMA_SERVICE_GROUP", "web")
    env_lines = [
        "Environment=SLOT_ID=%i",
        f"Environment=SLOT_PORT_BASE={_slot_port_base()}",
    ]
    if main_gpu is not None:
        env_lines.append(f"Environment=MAIN_GPU={main_gpu}")
    return "\n".join((
        "[Unit]",
        "Description=abstract_hugpy_dev model slot %i",
        "After=network.target",
        "StartLimitIntervalSec=120",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "Type=simple",
        f"User={user}",
        f"Group={group}",
        *env_lines,
        f"ExecStart={python_bin} -m abstract_hugpy_dev.managers.serve.slot_agent",
        "Restart=always",
        "RestartSec=5",
        "TimeoutStopSec=60",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ))


def slot_install_steps(unit_dir: str = "/etc/systemd/system",
                       main_gpu: str | None = None) -> list[tuple[str, str]]:
    """Return [(kind, payload)] describing the one-time install:

    ('write', unit_text) for the template, then ('cmd', shell) lines to enable
    each slot. The caller writes/executes (with sudo); this stays side-effect
    free so it can be shown as a dry run.
    """
    import os.path as osp
    unit_path = osp.join(unit_dir, "abstract-hugpy-slot@.service")
    steps = [("write:" + unit_path, render_slot_unit(main_gpu=main_gpu)),
             ("cmd", "systemctl daemon-reload")]
    for i in range(_slot_count()):
        steps.append(("cmd", f"systemctl enable --now abstract-hugpy-slot@{i + 1}"))
    return steps
