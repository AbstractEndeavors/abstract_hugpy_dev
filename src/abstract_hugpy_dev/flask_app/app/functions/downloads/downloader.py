from .imports import *
def model_status(model: dict) -> dict:
    destination = route_destination(model)              # was model_destination(...)
    marker = os.path.join(destination, HUGPY_MARKER)    # was install_marker(...)
    if os.path.exists(marker):
        status = "installed"
    elif os.path.exists(destination) and os.listdir(destination):
        status = "partial"
    else:
        status = "not_installed"
    return {"status": status, "destination": destination, "installed_marker": marker}

def write_install_marker(destination: str, model_key: str, model: dict[str, Any]) -> None:
    # The marker IS the authoritative hugpy.json declared-identity read back by
    # discovery (resolve_hugpy_marker / get_module) to reconstruct capability.
    # Discovery keys on the FULL `tasks` list + `primary_task`, so stamp both —
    # a singular-only marker silently drops secondary tasks (e.g. image-to-image)
    # on re-discovery. Mirror the live path (write_hugpy_marker / _stamp).
    primary = model.get("primary_task") or model.get("task")
    tasks = model.get("tasks") or ([primary] if primary else None)
    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]
    marker = os.path.join(destination, HUGPY_MARKER)
    payload = {
        "model_key": model_key,
        "hub_id": model.get("hub_id"),
        "framework": model.get("framework"),
        "task": primary,          # singular kept for back-compat; = primary_task
        "tasks": tasks,           # full capability list — what discovery reads
        "primary_task": primary or (tasks[0] if tasks else None),
        "filename": model.get("filename"),
        "include": model.get("include"),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(destination, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))



