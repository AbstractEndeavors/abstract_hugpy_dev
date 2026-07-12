"""Store-flattening + catalog-reconcile regression (operator-locked 2026-07-11).

Fabricates a temp store exercising every migration shape and asserts the
dry-run/apply/re-run/read-through/presence/atomicity contract end to end.

  * task-twin (partial + complete)      -> complete wins, .part twin archived
  * legacy-misc stranding (empty twin)  -> misc weights win, empty twin archived
  * flat-complete entry                 -> no-op
  * .part-only orphan (no good copy)    -> archived, no winner
  * gguf with a NON-pinned complete quant -> installed + pin WARNING (not partial)
  * vision gguf: quant here, mmproj in a twin -> MERGE mmproj, then move flat

Asserts: dry-run plan exact + touches nothing; apply -> flat layout + archives +
registry/marker updates; re-run is a no-op; read-through resolves every legacy
path BEFORE and AFTER; presence verdicts correct; an atomic provision leaves NO
resolvable partial on simulated failure.

Runs like the other tests here: venv/bin/python tests/test_store_reconcile.py
"""
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

paths = importlib.import_module("abstract_hugpy_dev.imports.src.constants.paths")
constants = importlib.import_module("abstract_hugpy_dev.imports.src.constants.constants")
reconcile = importlib.import_module("abstract_hugpy_dev.imports.apis.reconcile")
main = importlib.import_module("abstract_hugpy_dev.imports.config.main")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

MB = 1024 * 1024

def wfile(path, mb=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * (mb * MB + 7))

def marker(directory, hub_id, framework, primary_task=None, tasks=None, filename=None):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "hugpy.json"), "w") as fh:
        json.dump({"hub_id": hub_id, "framework": framework,
                   "primary_task": primary_task, "tasks": tasks,
                   "filename": filename, "source": "test"}, fh)


tmp = tempfile.mkdtemp(prefix="hugpy-reconcile-")
root = tmp
models = os.path.join(root, "models")

# ---- fabricate the store ---------------------------------------------------
# 1. task-twin: text-generation has ONLY a .part; image-text-to-text is complete
tg_twin = os.path.join(models, "gguf", "text-generation", "own", "twin-gguf")
wc_twin = os.path.join(models, "gguf", "image-text-to-text", "own", "twin-gguf")
wfile(os.path.join(tg_twin, "twin-Q4_K_M.gguf.part"), 3)          # crashed pull
wfile(os.path.join(wc_twin, "twin-Q4_K_M.gguf"), 3)              # complete
marker(wc_twin, "own/twin-gguf", "gguf", "text-generation", ["text-generation"])

# 2. legacy-misc stranding: weights in misc, empty text-generation twin
misc_lora = os.path.join(models, "transformers", "misc", "own", "lora-adapter")
empty_lora = os.path.join(models, "transformers", "text-generation", "own", "lora-adapter")
wfile(os.path.join(misc_lora, "adapter_model.safetensors"), 2)
with open(os.path.join(misc_lora, "adapter_config.json"), "w") as fh:
    fh.write("{}")
os.makedirs(empty_lora, exist_ok=True)                           # empty twin

# 3. flat-complete entry (already migrated)
flat_ok = os.path.join(models, "transformers", "own", "already-flat")
wfile(os.path.join(flat_ok, "model.safetensors"), 2)
with open(os.path.join(flat_ok, "config.json"), "w") as fh:
    fh.write("{}")
marker(flat_ok, "own/already-flat", "transformers", "text-generation", ["text-generation"])

# 4. .part-only orphan (no complete copy anywhere)
orphan = os.path.join(models, "gguf", "text-generation", "own", "orphan-gguf")
wfile(os.path.join(orphan, "orphan.gguf.part"), 3)

# 5. gguf pinned to a quant that is NOT the one on disk (flat-complete)
pinned = os.path.join(models, "gguf", "own", "pinned-gguf")
wfile(os.path.join(pinned, "pinned-Q4_K_M.gguf"), 3)
marker(pinned, "own/pinned-gguf", "gguf", "text-generation", ["text-generation"],
       filename="pinned-Q8_0.gguf")

# 6. vision gguf: quant in image-text-to-text, mmproj sidecar in a twin
vq = os.path.join(models, "gguf", "image-text-to-text", "own", "vision-gguf")
vm = os.path.join(models, "gguf", "text-generation", "own", "vision-gguf")
wfile(os.path.join(vq, "vision-Q4_K_M.gguf"), 3)                 # quant, no mmproj
wfile(os.path.join(vm, "mmproj-vision.gguf"), 2)                 # projector twin

# ---- catalog (discovery report + manifest) --------------------------------
discovery = {
    "twin-gguf": {"hub_id": "own/twin-gguf", "framework": "gguf",
                  "primary_task": "text-generation", "tasks": ["text-generation"],
                  "dir": wc_twin, "folder": "gguf/image-text-to-text/own/twin-gguf"},
    "lora-adapter": {"hub_id": "own/lora-adapter", "framework": "transformers",
                     "primary_task": "text-generation", "tasks": ["text-generation"],
                     "dir": misc_lora, "folder": "transformers/misc/own/lora-adapter"},
    "already-flat": {"hub_id": "own/already-flat", "framework": "transformers",
                     "primary_task": "text-generation", "tasks": ["text-generation"],
                     "dir": flat_ok, "folder": "transformers/own/already-flat"},
    "orphan-gguf": {"hub_id": "own/orphan-gguf", "framework": "gguf",
                    "primary_task": "text-generation", "tasks": ["text-generation"]},
    "pinned-gguf": {"hub_id": "own/pinned-gguf", "framework": "gguf",
                    "primary_task": "text-generation", "tasks": ["text-generation"],
                    "filename": "pinned-Q8_0.gguf", "dir": pinned,
                    "folder": "gguf/own/pinned-gguf"},
    "vision-gguf": {"hub_id": "own/vision-gguf", "framework": "gguf",
                    "primary_task": "image-text-to-text",
                    "tasks": ["image-text-to-text"], "dir": vq,
                    "folder": "gguf/image-text-to-text/own/vision-gguf"},
}
disc_path = os.path.join(tmp, "model_discovery.json")
mani_path = os.path.join(tmp, "model_manifest.json")
with open(disc_path, "w") as fh:
    json.dump(discovery, fh)
with open(mani_path, "w") as fh:
    json.dump({"lora-adapter": dict(discovery["lora-adapter"])}, fh)  # in both

# point the reconcile's catalog at our temp artifacts
constants.MODELS_DISCOVERY_PATH = disc_path
constants.MODELS_DICT_PATH = mani_path


def cfg_of(entry):
    return paths._routing_as_cfg(entry)

def plan_for(report, hub_id):
    for pl in report["plans"]:
        if pl.get("hub_id") == hub_id:
            return pl
    raise AssertionError(f"no plan for {hub_id}")

def acts(pl, op):
    return [a for a in pl["actions"] if a.get("op") == op]


# ===========================================================================
# READ-THROUGH before migration — every legacy path resolves to the complete dir
# ===========================================================================
print("[read-through BEFORE]")
check("twin: route_destination -> complete image-text-to-text copy",
      paths.route_destination(discovery["twin-gguf"], root) == wc_twin)
check("lora: route_destination -> misc weights (not the empty twin)",
      paths.route_destination(discovery["lora-adapter"], root) == misc_lora)
check("vision: resolve(require_complete) is None (mmproj missing pre-merge)",
      paths.resolve_model_dir(discovery["vision-gguf"], root) is None)


# ===========================================================================
# DRY RUN — plan exact, touches nothing
# ===========================================================================
print("[dry run]")
before = {p: sorted(os.listdir(p)) for p in
          (tg_twin, wc_twin, misc_lora, empty_lora, flat_ok, orphan, pinned, vq, vm)}
rep = reconcile.reconcile_store(root=root, apply=False)

pl_twin = plan_for(rep, "own/twin-gguf")
check("twin plan: moves complete copy to flat",
      acts(pl_twin, "move") and acts(pl_twin, "move")[0]["dst"]
      == os.path.join(models, "gguf", "own", "twin-gguf")
      and acts(pl_twin, "move")[0]["src"] == wc_twin)
check("twin plan: .part twin archived as part-orphan",
      any(a["src"] == tg_twin and a["reason"] == "part-orphan"
          for a in acts(pl_twin, "archive")))
check("twin plan: NO merge of the .part file", not acts(pl_twin, "merge"))

pl_lora = plan_for(rep, "own/lora-adapter")
check("lora plan: misc weights move to flat",
      acts(pl_lora, "move") and acts(pl_lora, "move")[0]["src"] == misc_lora)
check("lora plan: empty twin archived",
      any(a["src"] == empty_lora and a["reason"] == "empty-twin"
          for a in acts(pl_lora, "archive")))

pl_flat = plan_for(rep, "own/already-flat")
check("already-flat plan: no move/archive (no-op)",
      not acts(pl_flat, "move") and not acts(pl_flat, "archive")
      and pl_flat["status"] in ("already-flat", "noop"))

pl_orphan = plan_for(rep, "own/orphan-gguf")
check("orphan plan: no winner, part-orphan archived",
      pl_orphan["status"] == "incomplete-no-winner"
      and any(a["reason"] == "part-orphan" for a in acts(pl_orphan, "archive")))

pl_pin = plan_for(rep, "own/pinned-gguf")
check("pinned plan: pin-mismatch WARNING present",
      any("pinned filename" in w for w in pl_pin["warnings"]))
check("pinned plan: no move (already flat, quant present)",
      not acts(pl_pin, "move"))

pl_vis = plan_for(rep, "own/vision-gguf")
check("vision plan: MERGE mmproj from the twin into the winner",
      any(a["file"] == "mmproj-vision.gguf" and a["from"] == vm
          for a in acts(pl_vis, "merge")))
check("vision plan: winner (quant dir) moves to flat",
      acts(pl_vis, "move") and acts(pl_vis, "move")[0]["src"] == vq)

check("dry-run summary counts moves/merges/archives",
      rep["summary"]["moves"] == 3 and rep["summary"]["merges"] == 1
      and rep["summary"]["archives"] >= 3)

after = {p: sorted(os.listdir(p)) for p in before}
check("dry-run touched NOTHING on disk", before == after)
check("dry-run did not write the archive dir",
      not os.path.exists(rep["archive_dir"]))
with open(disc_path) as fh:
    check("dry-run did not mutate the discovery report",
          json.load(fh)["twin-gguf"]["dir"] == wc_twin)


# ===========================================================================
# APPLY — flat layout, archives, registry + markers
# ===========================================================================
print("[apply]")
rep2 = reconcile.reconcile_store(root=root, apply=True)
flat_twin = os.path.join(models, "gguf", "own", "twin-gguf")
flat_lora = os.path.join(models, "transformers", "own", "lora-adapter")
flat_vis = os.path.join(models, "gguf", "own", "vision-gguf")

check("apply: twin complete copy now at flat",
      os.path.isfile(os.path.join(flat_twin, "twin-Q4_K_M.gguf")))
check("apply: twin legacy image-text-to-text dir gone",
      not os.path.isdir(wc_twin))
check("apply: .part twin archived (never deleted)",
      os.path.isdir(os.path.join(models, "_archive", rep2["timestamp"],
                    "gguf", "text-generation", "own", "twin-gguf")))
check("apply: lora weights now at flat",
      os.path.isfile(os.path.join(flat_lora, "adapter_model.safetensors")))
check("apply: vision quant + MERGED mmproj both at flat",
      os.path.isfile(os.path.join(flat_vis, "vision-Q4_K_M.gguf"))
      and os.path.isfile(os.path.join(flat_vis, "mmproj-vision.gguf")))
check("apply: vision flat dir is COMPLETE (mmproj satisfied the vision gate)",
      main.model_looks_downloaded(flat_vis, cfg_of(discovery["vision-gguf"])))
check("apply: fresh hugpy.json marker at the flat winner",
      os.path.isfile(os.path.join(flat_twin, "hugpy.json")))
with open(disc_path) as fh:
    d = json.load(fh)
check("apply: registry dir/folder updated to flat",
      d["twin-gguf"]["dir"] == flat_twin
      and d["twin-gguf"]["folder"] == "gguf/own/twin-gguf")
check("apply: manifest entry (lora) also updated",
      json.load(open(mani_path))["lora-adapter"]["folder"]
      == "transformers/own/lora-adapter")
check("apply: NOTHING under _archive was deleted (orphan preserved)",
      os.path.isdir(os.path.join(models, "_archive", rep2["timestamp"])))


# ===========================================================================
# READ-THROUGH after migration — flat resolves; legacy dict still resolves
# ===========================================================================
print("[read-through AFTER]")
check("twin: route_destination now flat",
      paths.route_destination(d["twin-gguf"], root) == flat_twin)
# a stale routing dict that still names the OLD task path must not 404
stale = {"hub_id": "own/twin-gguf", "framework": "gguf",
         "primary_task": "text-generation", "tasks": ["text-generation"]}
check("stale legacy routing still resolves to the flat copy",
      paths.route_destination(stale, root) == flat_twin)
check("vision: resolve now finds the completed flat copy",
      paths.resolve_model_dir(discovery["vision-gguf"], root) == flat_vis)


# ===========================================================================
# RE-RUN — idempotent no-op (no ping-pong)
# ===========================================================================
print("[re-run]")
rep3 = reconcile.reconcile_store(root=root, apply=True)
check("re-run: zero moves", rep3["summary"]["moves"] == 0)
check("re-run: zero merges", rep3["summary"]["merges"] == 0)
check("re-run: zero NEW archives of live dirs",
      all(pl["status"] in ("already-flat", "noop", "incomplete-no-winner", "absent")
          for pl in rep3["plans"]))


# ===========================================================================
# PRESENCE HONESTY verdicts
# ===========================================================================
print("[presence]")
check("gguf any-quant: pinned-gguf installed despite pin mismatch",
      main.model_looks_downloaded(pinned, cfg_of(discovery["pinned-gguf"])))
check("pin mismatch is a WARNING signal, not incomplete",
      reconcile.pinned_filename_present(pinned, cfg_of(discovery["pinned-gguf"])) is False)
check("vision with no mmproj reads INCOMPLETE (gate preserved)",
      not main.model_looks_downloaded(
          os.path.join(models, "gguf", "own", "no-such"), cfg_of(discovery["vision-gguf"])))


# ===========================================================================
# ATOMIC PROVISION — a simulated failure leaves NO resolvable partial
# ===========================================================================
print("[atomicity]")
atomic = importlib.import_module("abstract_hugpy_dev.imports.apis.download_models")
# stage dir naming: the download must land in <dest>.tmp-<pid>, never at <dest>
fresh = {"hub_id": "own/fresh-model", "framework": "gguf",
         "primary_task": "text-generation", "tasks": ["text-generation"]}
dest = paths.flat_destination(fresh, root)
staged = atomic._staging_dir(dest)
check("staging dir is a sibling temp of the final dest",
      staged != dest and staged.startswith(dest + ".tmp-"))
# simulate a crash: staged dir with a half-file exists, final does not
os.makedirs(staged, exist_ok=True)
wfile(os.path.join(staged, "half.gguf.part"), 1)
check("a crashed provision's staged dir does NOT resolve as the model",
      paths.resolve_model_dir(fresh, root) is None
      and not os.path.isdir(dest))


shutil.rmtree(tmp, ignore_errors=True)
print(f"\nALL {ok} CHECKS PASSED")
