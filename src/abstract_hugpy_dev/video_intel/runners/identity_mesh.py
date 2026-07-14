# video_intel/runners/identity_mesh.py
import os
import json
import time
import urllib.request
from typing import Any
from ...imports.src.constants.constants import DEFAULT_ROOT
from ..identity_reconstruction_schema import IdentityMeshSpec
from ..identity_profiles import get_profile

COMFY_API_URL = "http://127.0.0.1:8188"

def _resolve_view_uris(slug: str, recon_id: str, view_ids: tuple[str, ...]) -> dict[str, str]:
    """Extracts the physical disk paths for the selected approved angle views."""
    profile = get_profile(slug)
    recons = profile.get("reconstructions", [])
    target_recon = next((r for r in recons if r.get("recon_id") == recon_id), None)
    
    if not target_recon:
        raise RuntimeError(f"Reconstruction {recon_id} not found for {slug}")
        
    views = target_recon.get("views", [])
    uri_map = {}
    
    for vid in view_ids:
        view_data = next((v for v in views if v.get("viewId") == vid), None)
        if not view_data or not view_data.get("imageUri"):
            raise RuntimeError(f"Approved view {vid} missing or has no image data")
        uri_map[vid] = view_data["imageUri"]
        
    return uri_map

def build_hunyuan3d_payload(uri_map: dict[str, str]) -> dict:
    """
    Constructs the ComfyUI API JSON graph for Hunyuan3D-2mv.
    The graph explicitly isolates the Shape and Paint nodes to allow the ComfyUI
    execution engine to unload the Shape model from VRAM before loading the Texture model.
    """
    # Note: This is a structural skeleton of the ComfyUI API payload.
    # The actual node IDs and class types depend on your specific ComfyUI custom nodes
    # (e.g., 'Hunyuan3D_Multiview_Input', 'Hunyuan3D_ShapeGen', 'Hunyuan3D_PaintGen').
    
    prompt = {
        "1": {
            "class_type": "LoadImageList",
            "inputs": {
                # Map the 8 anchor paths (0, 45, 90, 135, 180, 225, 270, 315)
                "image_paths": list(uri_map.values()) 
            }
        },
        "2": {
            "class_type": "Hunyuan3DShapeGenerator",
            "inputs": {
                "images": ["1", 0],
                "force_vram_unload_after": True # Custom flag telling the node to release VRAM
            }
        },
        "3": {
            "class_type": "EmptyCUDACache", # Explicit barrier node
            "inputs": {
                "dependency": ["2", 0]
            }
        },
        "4": {
            "class_type": "Hunyuan3DPBRTextureGenerator",
            "inputs": {
                "mesh": ["2", 0],
                "primary_image": ["1", 0], # Usually the 0-degree front view
                "dependency": ["3", 0] # Ensure cache clears before paint loads
            }
        },
        "5": {
            "class_type": "SaveGLB",
            "inputs": {
                "mesh": ["4", 0],
                "filename_prefix": "identity_mesh"
            }
        }
    }
    return prompt

def run(spec: IdentityMeshSpec) -> dict[str, Any]:
    """
    Executes the 3D mesh building pipeline.
    """
    try:
        # 1. Gather the approved anchor images
        uri_map = _resolve_view_uris(spec.slug, spec.recon_id, spec.view_ids)
        
        # 2. Build the ComfyUI graph
        payload = {"prompt": build_hunyuan3d_payload(uri_map)}
        
        # 3. Submit to local ComfyUI
        req = urllib.request.Request(
            f"{COMFY_API_URL}/prompt", 
            data=json.dumps(payload).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read())
            prompt_id = res_data["prompt_id"]

        # 4. Poll ComfyUI /history endpoint until complete
        # (In a production runner, you would use a websocket for realtime progress)
        completed = False
        output_glb_name = None
        while not completed:
            time.sleep(2)
            hist_req = urllib.request.Request(f"{COMFY_API_URL}/history/{prompt_id}")
            with urllib.request.urlopen(hist_req) as hist_res:
                history = json.loads(hist_res.read())
                if prompt_id in history:
                    # Parse output node (Node 5) for the generated filename
                    outputs = history[prompt_id].get("outputs", {})
                    if "5" in outputs and "files" in outputs["5"]:
                        output_glb_name = outputs["5"]["files"][0]
                    completed = True
        
        # 5. Move the GLB from ComfyUI output dir to the durable identity profile dir
        comfy_output_path = os.path.join(DEFAULT_ROOT, "comfy_output", output_glb_name)
        durable_path = os.path.join(DEFAULT_ROOT, "identities", spec.slug, "mesh", f"{spec.recon_id}.glb")
        
        os.makedirs(os.path.dirname(durable_path), exist_ok=True)
        os.rename(comfy_output_path, durable_path)
        
        # 6. Update the identity_profiles.json mesh status to "completed" (done via bus result watcher usually)
        
        return {
            "ok": True,
            "outputs": [{
                "kind": "mesh",
                "uri": durable_path,
                "mime": "model/gltf-binary"
            }]
        }

    except Exception as e:
        return {
            "ok": False,
            "error": {"code": "MeshBuildFailed", "message": str(e)}
        }
