"""Central-side call-time attribution for relay-dispatched foreign GPU services
(identity-render) — the PURE correlation core.

Loads the module STANDALONE via importlib (not through the package) so the test
needs no flask app factory / comms / media_bus: the pure function
``attribute_foreign_relay_procs`` has zero package-level deps (comms/media_bus
imports are all lazy, inside the impure glue this test does not touch).

Runs like the other tests here:
    venv/bin/python tests/test_pid_attribution.py
"""
import importlib.util
import sys
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[1]
         / "src/abstract_hugpy_dev/flask_app/app/functions/imports/utils/pid_attribution.py")
_spec = importlib.util.spec_from_file_location("pid_attribution_under_test", _PATH)
PA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PA)

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


# ── identity-render pid + one active mesh job -> stamped ─────────────────────
def test_identity_render_attribution():
    pr = {"models": [], "unattributed": [
        {"pid": 5001, "name": "/srv/identity-render/venv/bin/python3", "mib": 5400},
        {"pid": 6001, "name": "/usr/bin/xmrig", "mib": 8000},  # genuine squatter
    ]}
    active = [{"kind": "identity_mesh_build", "id": "job-abc",
               "model": "identity_mesh_build", "slug": "luigi"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    by_pid = {u["pid"]: u for u in out["unattributed"]}
    check("identity: hy3dgen pid stamped host_mode",
          by_pid[5001].get("host_mode") == "identity-render")
    check("identity: hy3dgen pid stamped service",
          by_pid[5001].get("service") == "identity-render")
    check("identity: hy3dgen pid stamped job_id", by_pid[5001].get("job_id") == "job-abc")
    check("identity: hy3dgen pid stamped slug", by_pid[5001].get("slug") == "luigi")
    check("identity: model_key resolves to the slug", by_pid[5001].get("model_key") == "luigi")
    check("identity: attribution = relay-job", by_pid[5001].get("attribution") == "relay-job")
    check("identity: pid/name/mib preserved on the stamped row",
          by_pid[5001].get("pid") == 5001 and by_pid[5001].get("mib") == 5400)
    check("identity: genuine squatter left untouched (no host_mode)",
          by_pid[6001].get("host_mode") is None)


# ── hy3dgen marker also matches (process name variety) ───────────────────────
def test_hy3dgen_marker_and_slug_from_media_field():
    pr = {"unattributed": [{"pid": 5002,
                            "name": "/opt/hy3dgen/venv/bin/python", "mib": 5000}]}
    # No explicit slug on the job row -> falls back to model_key then model.
    active = [{"kind": "identity_mesh_build", "id": "j9", "model_key": "gio"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    e = out["unattributed"][0]
    check("hy3dgen: marker matches identity-render service",
          e.get("host_mode") == "identity-render")
    check("hy3dgen: model_key falls back to job.model_key when no slug",
          e.get("model_key") == "gio")


# ── recognized service, NO active job -> recognized-idle (ours, just idle) ───
def test_recognized_idle_no_job():
    pr = {"unattributed": [{"pid": 5001,
                            "name": "/srv/identity-render/venv/bin/python", "mib": 5400}]}
    out = PA.attribute_foreign_relay_procs(pr, [])
    e = out["unattributed"][0]
    check("idle: recognized service host_mode", e.get("host_mode") == "identity-render")
    check("idle: attribution = recognized-idle", e.get("attribution") == "recognized-idle")
    check("idle: no model_key", e.get("model_key") is None)


# ── multiple active mesh jobs -> honest ambiguity ───────────────────────────
def test_ambiguous_multiple_jobs():
    pr = {"unattributed": [{"pid": 5001,
                            "name": "/opt/hy3dgen/venv/bin/python", "mib": 5400}]}
    active = [{"kind": "identity_mesh_build", "id": "j1", "slug": "a"},
              {"kind": "identity_mesh_build", "id": "j2", "slug": "b"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    e = out["unattributed"][0]
    check("ambiguous: flagged relay-job-ambiguous",
          e.get("attribution") == "relay-job-ambiguous")
    check("ambiguous: job_id is the list of active job ids",
          isinstance(e.get("job_id"), list) and set(e.get("job_id")) == {"j1", "j2"})
    check("ambiguous: model_key not asserted", e.get("model_key") is None)


# ── a job of an UNRELATED kind never matches identity-render ─────────────────
def test_unrelated_job_kind_ignored():
    pr = {"unattributed": [{"pid": 5001,
                            "name": "/srv/identity-render/venv/bin/python", "mib": 5400}]}
    active = [{"kind": "crop", "id": "c1"}, {"kind": "studio_i2v", "id": "s1"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    e = out["unattributed"][0]
    check("unrelated-kind: recognized service but recognized-idle (no matching kind)",
          e.get("attribution") == "recognized-idle")


# ── degrade-safe: bad / empty inputs pass through unchanged ──────────────────
def test_degrade_safe():
    check("degrade: None pid_registry -> None",
          PA.attribute_foreign_relay_procs(None, []) is None)
    empty = {"unattributed": []}
    check("degrade: empty unattributed -> same object (no-op)",
          PA.attribute_foreign_relay_procs(
              empty, [{"kind": "identity_mesh_build", "id": "j"}]) is empty)
    no_key = {"models": []}
    check("degrade: missing unattributed key -> unchanged",
          PA.attribute_foreign_relay_procs(no_key, []) is no_key)
    only_squatters = {"unattributed": [{"pid": 9, "name": "/usr/bin/xmrig", "mib": 1}]}
    out = PA.attribute_foreign_relay_procs(
        only_squatters, [{"kind": "identity_mesh_build", "id": "j", "slug": "x"}])
    check("degrade: no recognized proc -> input returned unchanged",
          out is only_squatters)


def main():
    test_identity_render_attribution()
    test_hy3dgen_marker_and_slug_from_media_field()
    test_recognized_idle_no_job()
    test_ambiguous_multiple_jobs()
    test_unrelated_job_kind_ignored()
    test_degrade_safe()
    print("\n%d ok, %d failed" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
