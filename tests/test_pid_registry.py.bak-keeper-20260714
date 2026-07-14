"""Precision model->PID registry — record/verify/forget, the recycled-PID guard,
reconcile attribution (subprocess vs in-process vs comfy + foreign squatter),
and degrade-to-empty on no-GPU inputs.

No real GPU or subprocess: a FAKE /proc probe injects process identity
(start-time + cmdline), so every path is exercised deterministically.

Runs like the other tests here:
    venv/bin/python tests/test_pid_registry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.worker_agent import pid_registry as PR  # noqa: E402

_MIB = 1024 * 1024

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


class FakeProc:
    """A settable process table: {pid: {"starttime", "cmdline", "name"}}.

    Deleting a pid = the process exited; re-adding the same pid with a DIFFERENT
    starttime = the OS recycled the number for a stranger.
    """

    def __init__(self):
        self.table = {}

    def probe(self, pid):
        return self.table.get(pid)


def new_registry(fake):
    r = PR.PidRegistry(proc_info=fake.probe)
    return r


# ── record / verify / forget ────────────────────────────────────────────────
def test_record_verify_forget():
    fake = FakeProc()
    fake.table[4242] = {"starttime": 100, "cmdline": "llama-server -m foo.gguf", "name": "llama-server"}
    r = new_registry(fake)

    rec = r.record_launch("foo/model", 4242, "subprocess")
    check("record returns the pid", rec["pid"] == 4242)
    check("record captured start_tick anchor", rec["start_tick"] == 100)
    check("verify returns live pid", r.verify("foo/model") == 4242)

    check("forget drops the record", r.forget("foo/model") is True)
    check("verify after forget -> None", r.verify("foo/model") is None)
    check("forget unknown -> False", r.forget("nope") is False)
    check("verify unknown -> None", r.verify("nope") is None)


# ── recycled-PID guard: the core "precision" ────────────────────────────────
def test_recycled_pid_guard():
    fake = FakeProc()
    fake.table[5001] = {"starttime": 555, "cmdline": "llama-server -m mA.gguf", "name": "llama-server"}
    r = new_registry(fake)
    r.record_launch("model/A", 5001, "subprocess")
    check("guard: fresh pid verifies", r.verify("model/A") == 5001)

    # Process exits -> pid vanishes from /proc.
    del fake.table[5001]
    check("guard: gone pid -> verify None", r.verify("model/A") is None)

    # OS reuses the SAME number 5001 for an unrelated process (different
    # starttime + different cmdline). The number is alive again, but it is NOT
    # our model's process.
    fake.table[5001] = {"starttime": 999, "cmdline": "python some_other_thing.py", "name": "python"}
    check("guard: recycled pid (new starttime) -> verify None", r.verify("model/A") is None)

    # sweep_dead removes the poisoned record.
    dropped = r.sweep_dead()
    check("guard: sweep_dead forgets the recycled record", "model/A" in dropped)
    check("guard: record gone after sweep", r.verify("model/A") is None)


def test_cmdline_fallback_when_no_starttime():
    # If starttime is unreadable at record time, identity falls back to cmdline.
    fake = FakeProc()
    fake.table[6001] = {"starttime": None, "cmdline": "llama-server -m secret.gguf", "name": "llama-server"}
    r = new_registry(fake)
    r.record_launch("model/B", 6001, "subprocess")
    check("cmdline-fallback: matching cmdline verifies", r.verify("model/B") == 6001)

    # Same pid, no starttime, but a DIFFERENT cmdline -> can't corroborate -> None.
    fake.table[6001] = {"starttime": None, "cmdline": "python stranger.py", "name": "python"}
    check("cmdline-fallback: changed cmdline -> verify None", r.verify("model/B") is None)


# ── reconcile: attribution across host modes + foreign squatter ─────────────
def test_reconcile_attribution():
    fake = FakeProc()
    worker_pid = 1000          # the worker python holding in-process torch models
    slot_pid = 2000            # a slot child (llama-server)
    comfy_pid = 3000           # external ComfyUI
    foreign_pid = 4000         # a ROGUE VRAM squatter the registry can't explain
    fake.table[worker_pid] = {"starttime": 10, "cmdline": "python -m ...agent", "name": "python"}
    fake.table[slot_pid] = {"starttime": 20, "cmdline": "llama-server -m g.gguf", "name": "llama-server"}
    fake.table[comfy_pid] = {"starttime": 30, "cmdline": "python main.py", "name": "ComfyUI"}
    fake.table[foreign_pid] = {"starttime": 40, "cmdline": "./miner", "name": "xmrig"}
    r = new_registry(fake)

    r.record_launch("slot/gguf", slot_pid, "subprocess")
    r.record_launch("inproc/vision", worker_pid, "in_process")
    r.record_launch("comfy/sdxl", comfy_pid, "comfy")

    # nvidia-smi ground truth (mib per pid).
    gpu_procs = {
        worker_pid: {"name": "python", "mib": 3600},   # in-process lump (torch splits it)
        slot_pid: {"name": "llama-server", "mib": 5120},
        comfy_pid: {"name": "ComfyUI", "mib": 8000},
        foreign_pid: {"name": "xmrig", "mib": 12000},  # squatter
    }
    inprocess_bytes = {"inproc/vision": {"vram_bytes": 2_000_000_000, "device": "cuda"}}
    comfy_bytes = 8000 * _MIB

    res = r.reconcile(gpu_procs, inprocess_bytes, comfy_bytes)
    att = res["attributed"]
    check("reconcile: subprocess model gets its pid's mib",
          att["slot/gguf"] == 5120 * _MIB)
    check("reconcile: in-process model gets torch-split bytes (not the lump)",
          att["inproc/vision"] == 2_000_000_000)
    check("reconcile: comfy model gets comfy_bytes",
          att["comfy/sdxl"] == 8000 * _MIB)

    unatt_pids = {u["pid"] for u in res["unattributed"]}
    check("reconcile: foreign squatter surfaced as unattributed",
          foreign_pid in unatt_pids)
    check("reconcile: worker-python lump NOT unattributed (explained by in-proc)",
          worker_pid not in unatt_pids)
    check("reconcile: slot pid NOT unattributed", slot_pid not in unatt_pids)
    check("reconcile: comfy pid NOT unattributed (name-matched)",
          comfy_pid not in unatt_pids)
    check("reconcile: exactly one unattributed", len(res["unattributed"]) == 1)

    # snapshot reflects the attribution + live guarded aliveness.
    snap = r.snapshot_for_heartbeat()
    by_key = {m["model_key"]: m for m in snap["models"]}
    check("snapshot: three model rows", len(snap["models"]) == 3)
    check("snapshot: slot row carries reconciled vram",
          by_key["slot/gguf"]["vram_bytes"] == 5120 * _MIB)
    check("snapshot: slot row host_mode", by_key["slot/gguf"]["host_mode"] == "subprocess")
    check("snapshot: all rows alive under guard",
          all(m["alive"] for m in snap["models"]))
    check("snapshot: unattributed squatter carried through",
          foreign_pid in {u["pid"] for u in snap["unattributed"]})

    # A dead model_key in the log reads alive=False after its process exits.
    del fake.table[slot_pid]
    snap2 = r.snapshot_for_heartbeat()
    by_key2 = {m["model_key"]: m for m in snap2["models"]}
    check("snapshot: exited slot child reads alive=False",
          by_key2["slot/gguf"]["alive"] is False)


# ── degrade-to-empty: no GPU / no models ────────────────────────────────────
def test_degrade_empty():
    fake = FakeProc()
    r = new_registry(fake)
    res = r.reconcile({}, {}, None)
    check("degrade: empty reconcile attributed {}", res["attributed"] == {})
    check("degrade: empty reconcile unattributed []", res["unattributed"] == [])
    snap = r.snapshot_for_heartbeat()
    check("degrade: empty snapshot models []", snap["models"] == [])
    check("degrade: empty snapshot unattributed []", snap["unattributed"] == [])

    # None inputs (no nvidia-smi at all) also degrade cleanly.
    res2 = r.reconcile(None, None, None)
    check("degrade: None inputs -> empty attributed", res2["attributed"] == {})
    check("degrade: None inputs -> empty unattributed", res2["unattributed"] == [])


# ── idempotent heartbeat-driven population ──────────────────────────────────
def test_record_idempotent_same_process():
    fake = FakeProc()
    fake.table[7000] = {"starttime": 77, "cmdline": "llama-server -m x.gguf", "name": "llama-server"}
    r = new_registry(fake)
    first = r.record_launch("m/x", 7000, "subprocess")
    launched_at = first["launched_at"]
    # Re-observe the SAME live process (heartbeat calls record each beat).
    again = r.record_launch("m/x", 7000, "subprocess")
    check("idempotent: launched_at preserved on same pid+starttime",
          again["launched_at"] == launched_at)
    # A genuine reload (new pid, new starttime) replaces the record.
    fake.table[7001] = {"starttime": 88, "cmdline": "llama-server -m x.gguf", "name": "llama-server"}
    reloaded = r.record_launch("m/x", 7001, "subprocess")
    check("idempotent: new launch replaces record", reloaded["pid"] == 7001)
    check("idempotent: verify tracks the new pid", r.verify("m/x") == 7001)


def main():
    test_record_verify_forget()
    test_recycled_pid_guard()
    test_cmdline_fallback_when_no_starttime()
    test_reconcile_attribution()
    test_degrade_empty()
    test_record_idempotent_same_process()
    print("\n%d ok, %d failed" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
