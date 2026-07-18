"""ID-locked STILL image generation via ComfyUI IP-Adapter (2026-07-12).

The STILL sibling of the studio VIDEO arm's id_lock (Wan-VACE reference-to-video):
give reference image(s) of a subject and generate NEW stills that HOLD the
identity, through the comfy delegation path with an IP-Adapter graph. Zero-train
reference-embedding guidance (video_intel.studio.enums.AttentionMethod.IP_ADAPTER).

Five halves, each an invariant of the slice:

  * REQUEST — ImageGenRequest gains reference_images (jailed abs paths) + id_strength
    (0..1) + the reference_images_b64 offload transport; the builder jail-resolves +
    image-classifies + count-checks the paths; ABSENT everything => byte-for-byte the
    legacy request.
  * BUILDER — a request with references composes the IPAdapter chain (loaders ->
    LoadImage(s) -> ImageBatch -> IPAdapterAdvanced) wired onto the EXISTING KSampler,
    with sampler/scheduler/cfg/seed all carried through and the id_strength on the
    apply node's weight. Family match (SD1.5 vs SDXL) picks the adapter weight.
  * TRANSPORT — a comfy worker (127.0.0.1) can't see central's paths, so
    remote._inline_reference_images base64s them into reference_images_b64 and DROPS
    the paths; the worker rebuilds the request + the comfy runner uploads the bytes.
  * DETECTION — the IPAdapter node pack is PROBED (object_info), never assumed: present
    -> id_lock True; absent -> False AND an id_lock request fails as data with the
    install pointer (NEVER a silent non-locked image).
  * ROUTING — an id_lock request only lands on a box whose comfy advertises id_lock
    (STRICT), skipping comfy-less / nodeless workers; a plain request is untouched.

Runs like the other tests here:
    venv/bin/python tests/test_comfy_id_lock.py
"""
import asyncio
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

builders = importlib.import_module(
    "abstract_hugpy_dev.managers.resolvers.categories.builders")
comfy_runner = importlib.import_module(
    "abstract_hugpy_dev.managers.comfy.comfy_runner")
remote = importlib.import_module("abstract_hugpy_dev.managers.resolvers.remote")
schemas = importlib.import_module("abstract_hugpy_dev.managers.imagegen.schemas")
W = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers")
from abstract_hugpy_dev.imports.src.constants.constants import UPLOADS_HOME  # noqa: E402
from worker_store_isolation import isolated_worker_store  # noqa: E402

ImageGenRequest = schemas.ImageGenRequest

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# A pair of real PNGs UNDER the storage jail (UPLOADS_HOME) for the path tests.
os.makedirs(UPLOADS_HOME, exist_ok=True)
_REF1 = os.path.join(UPLOADS_HOME, "_idlock_ut_ref1.png")
_REF2 = os.path.join(UPLOADS_HOME, "_idlock_ut_ref2.png")
Image.new("RGB", (8, 8), (200, 10, 10)).save(_REF1)
Image.new("RGB", (8, 8), (10, 10, 200)).save(_REF2)


print("REQUEST validation (jail / bounds / legacy)")
# jailed abs paths + id_strength accepted, image-classified, order preserved
req = builders._build_imagegen_request(
    {"prompt": "a hero", "reference_images": [_REF1, _REF2], "id_strength": 0.42},
    "comfy-model")
check("jailed reference paths accepted, order preserved",
      req.reference_images == [_REF1, _REF2])
check("id_strength forwarded", req.id_strength == 0.42)

# jail escape -> ValueError as data
try:
    builders._build_imagegen_request(
        {"prompt": "x", "reference_images": ["/etc/passwd"]}, "m")
    check("jail escape rejected", False)
except ValueError as e:
    check("jail escape rejected", "escapes the storage jail" in str(e))

# non-image reference -> ValueError as data
_txt = os.path.join(UPLOADS_HOME, "_idlock_ut_notimg.txt")
open(_txt, "w").write("not an image")
try:
    builders._build_imagegen_request(
        {"prompt": "x", "reference_images": [_txt]}, "m")
    check("non-image reference rejected", False)
except ValueError as e:
    check("non-image reference rejected", "is not an image" in str(e))

# over-count (>4) -> ValueError as data (builder count-check)
try:
    builders._build_imagegen_request(
        {"prompt": "x", "reference_images": [_REF1] * 5}, "m")
    check("over-count (>4) rejected", False)
except ValueError as e:
    check("over-count (>4) rejected", "at most" in str(e))

# id_strength out of [0,1] -> ValidationError (schema bound)
try:
    ImageGenRequest(request_id="r", model_key="m", prompt="p", id_strength=1.5)
    check("id_strength > 1 rejected", False)
except Exception as e:
    check("id_strength > 1 rejected", "ValidationError" in type(e).__name__)

# ABSENT everything == legacy request (no id fields set)
legacy = builders._build_imagegen_request({"prompt": "plain"}, "m")
check("absent references == legacy (reference_images None)",
      legacy.reference_images is None and legacy.id_strength is None
      and legacy.reference_images_b64 is None)


print("BUILDER graph structure (IPAdapter chain wired to the sampler)")
greq = ImageGenRequest(
    request_id="g", model_key="m", prompt="a knight", negative_prompt="blurry",
    sampler_name="dpmpp_2m", scheduler="karras", guidance_scale=6.5, seed=1234,
    id_strength=0.55)
base = comfy_runner._t2i_workflow("sd15-model.safetensors", greq, seed=1234)
wf = comfy_runner._ipadapter_workflow("sd15-model.safetensors", greq, base,
                                      ["ref0.png", "ref1.png"])
check("loader + clip-vision + 2 LoadImage + batch + apply nodes present",
      {"20", "21", "30", "31", "40", "50"} <= set(wf))
check("apply node is the IPAdapter apply class",
      wf["50"]["class_type"] == comfy_runner.IPADAPTER_APPLY_NODE)
check("loader node is the IPAdapter loader class",
      wf["20"]["class_type"] == comfy_runner.IPADAPTER_LOADER_NODE)
check("KSampler.model rewired onto the IPAdapter-patched model",
      wf["5"]["inputs"]["model"] == ["50", 0])
check("apply.model reads the checkpoint MODEL (node 1)",
      wf["50"]["inputs"]["model"] == ["1", 0])
check("apply.image reads the batched references (node 40)",
      wf["50"]["inputs"]["image"] == ["40", 0])
check("apply.ipadapter <- loader, apply.clip_vision <- clip-vision loader",
      wf["50"]["inputs"]["ipadapter"] == ["20", 0]
      and wf["50"]["inputs"]["clip_vision"] == ["21", 0])
check("ImageBatch folds the two references",
      wf["40"]["inputs"]["image1"] == ["30", 0]
      and wf["40"]["inputs"]["image2"] == ["31", 0])
check("id_strength -> apply weight", wf["50"]["inputs"]["weight"] == 0.55)
# sampler/scheduler/cfg/seed carried through UNTOUCHED
check("sampler/scheduler/cfg/seed preserved through id_lock wrap",
      wf["5"]["inputs"]["sampler_name"] == "dpmpp_2m"
      and wf["5"]["inputs"]["scheduler"] == "karras"
      and wf["5"]["inputs"]["cfg"] == 6.5
      and wf["5"]["inputs"]["seed"] == 1234)

# single reference: no batch node, apply reads the LoadImage directly
wf1 = comfy_runner._ipadapter_workflow(
    "sd15-model.safetensors", greq,
    comfy_runner._t2i_workflow("sd15-model.safetensors", greq, 1), ["only.png"])
check("single reference: no ImageBatch, apply reads LoadImage directly",
      "40" not in wf1 and wf1["50"]["inputs"]["image"] == ["30", 0])

# family match picks the adapter weight (SD1.5 vs SDXL)
check("SDXL checkpoint -> sdxl adapter weight",
      comfy_runner._ipadapter_workflow(
          "juggernautXL.safetensors", greq,
          comfy_runner._t2i_workflow("juggernautXL.safetensors", greq, 1),
          ["r.png"])["20"]["inputs"]["ipadapter_file"]
      == "ip-adapter_sdxl_vit-h.safetensors")
check("SD1.5 checkpoint -> sd15 adapter weight",
      comfy_runner._ipadapter_workflow(
          "sd15-pruned.safetensors", greq,
          comfy_runner._t2i_workflow("sd15-pruned.safetensors", greq, 1),
          ["r.png"])["20"]["inputs"]["ipadapter_file"]
      == "ip-adapter_sd15.safetensors")


print("TRANSPORT (central inline -> worker rebuild -> reference bytes)")
payload = builders._build_imagegen_request(
    {"prompt": "hero", "reference_images": [_REF1, _REF2], "id_strength": 0.5},
    "m").model_dump()
inlined = remote._inline_reference_images(payload)
check("inline succeeds and drops the unreachable paths",
      inlined and payload.get("reference_images") is None)
check("inline base64s each reference",
      len(payload.get("reference_images_b64") or []) == 2)
# worker rebuilds via the same builder (paths gone; b64 present)
wreq = builders._build_imagegen_request(payload, "m")
check("worker request carries b64, not paths",
      wreq.reference_images is None and len(wreq.reference_images_b64) == 2)

class _Cfg:
    model_key = "m"
    filename = "sd15.safetensors"

pays = comfy_runner.ComfyRunner(_Cfg())._reference_payloads(wreq)
check("reference bytes round-trip to the original files",
      [d for _, d in pays] == [open(_REF1, "rb").read(), open(_REF2, "rb").read()])


print("DETECTION (object_info probe -> capability + errors-as-data)")
class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
    def json(self):
        return self._p

class _FakeClient:
    """A comfy whose object_info advertises exactly ``present`` node classes."""
    def __init__(self, present):
        self.present = set(present)
    def get(self, url):
        cls = url.rsplit("/", 1)[-1]
        return _Resp(200, {cls: {"input": {}}} if cls in self.present else {})

both = comfy_runner.IPADAPTER_REQUIRED_NODES
check("object_info WITH the IPAdapter nodes -> capable",
      comfy_runner._probe_ipadapter("http://x", client=_FakeClient(both)) is True)
check("object_info missing the apply node -> NOT capable",
      comfy_runner._probe_ipadapter(
          "http://x", client=_FakeClient([comfy_runner.IPADAPTER_LOADER_NODE]))
      is False)
check("object_info with no IPAdapter nodes -> NOT capable",
      comfy_runner._probe_ipadapter("http://x", client=_FakeClient([])) is False)

# absent-nodes id_lock request -> clear errors-as-data envelope (never silent).
_orig_probe = comfy_runner.comfy_has_ipadapter
comfy_runner.comfy_has_ipadapter = lambda url, client=None: False
try:
    idreq = ImageGenRequest(request_id="e", model_key="m", prompt="hero",
                            reference_images_b64=[
                                "aGVsbG8="], id_strength=0.6)
    result = asyncio.run(comfy_runner.ComfyRunner(_Cfg()).run(idreq))
    check("absent-nodes id_lock request fails as data (ok=False)", result.ok is False)
    check("error names the missing IPAdapter nodes + the install pointer",
          "IPAdapter" in (result.error or "")
          and "WORKER-SETUP" in (result.error or ""))
finally:
    comfy_runner.comfy_has_ipadapter = _orig_probe


print("ROUTING (id_lock lands only on a comfy-with-nodes worker)")
def _fresh_store():
    # k3 isolation: isolated_worker_store() also redirects the assignment-
    # memory sidecar (settings.manifest_path) — see
    # tests/worker_store_isolation.py.
    store, _tmp = isolated_worker_store(prefix="hugpy-idlock-")
    return store

MODEL = "org/Comfy-Ckpt"
store = _fresh_store()
def _add(wid, comfy):
    store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid, models=[MODEL])
    store.set_admission(wid, "approved")
    if comfy is not None:
        store.heartbeat(wid, comfy=comfy)

_add("has_nodes", {"available": True, "id_lock": True})
_add("no_nodes", {"available": True, "id_lock": False})
_add("no_comfy", None)

ids_locked = {w["id"] for w in store.workers_for_model(
    MODEL, online_only=False, require_comfy_id_lock=True)}
check("id_lock routing keeps only the comfy-with-nodes worker",
      ids_locked == {"has_nodes"})
ids_plain = {w["id"] for w in store.workers_for_model(MODEL, online_only=False)}
check("plain routing (no id_lock) is untouched — every worker eligible",
      ids_plain == {"has_nodes", "no_nodes", "no_comfy"})
check("pick_for_model(require_comfy_id_lock) selects the capable box",
      (store.pick_for_model(MODEL, require_comfy_id_lock=True) or {}).get("id")
      == "has_nodes")

# the remote-side reroute filter agrees with the registry gate
check("remote _worker_comfy_id_lock_capable: nodes -> True",
      remote._worker_comfy_id_lock_capable(
          {"comfy": {"available": True, "id_lock": True}}) is True)
check("remote _worker_comfy_id_lock_capable: no nodes / no comfy -> False",
      remote._worker_comfy_id_lock_capable(
          {"comfy": {"available": True, "id_lock": False}}) is False
      and remote._worker_comfy_id_lock_capable({}) is False)


# cleanup the fixture files
for _f in (_REF1, _REF2, _txt):
    try:
        os.unlink(_f)
    except OSError:
        pass

print(f"\nall {ok} checks passed")
