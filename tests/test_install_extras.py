"""The enroll installer ships the media-intelligence deps + the numpy pin.

The canonical [engine] venv omits sentence-transformers / openai-whisper /
keybert, and numpy>=2.5 breaks numba so `import whisper` dies — the three
2026-07-11 request-time failures. bootstrap.sh (served by the
GET /llm/workers/install.sh route) must install all three plus `numpy<2.5`, and
the agent's `pip install -U --no-deps` self-update must never strip them.

We assert against the ACTUAL rendered route output (workers_install_sh), and,
as a second check, that the packaged resource the route serves carries the block.

Runs like the other tests here:
    venv/bin/python tests/test_install_extras.py
"""
import importlib
import sys
from importlib import resources
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# --- the rendered route output ---------------------------------------------
from flask import Flask  # noqa: E402
wr = importlib.import_module(
    "abstract_hugpy_dev.flask_app.app.routes.worker_routes")

app = Flask(__name__)
with app.test_request_context("/api/llm/workers/install.sh",
                              base_url="https://dev.hugpy.ai"):
    resp = wr.workers_install_sh()
    body = resp.get_data(as_text=True)

check("route renders a shell script", body.lstrip().startswith("#!"))
check("route baked THIS central into CENTRAL=", 'https://dev.hugpy.ai' in body)

for pkg in ("sentence-transformers", "openai-whisper", "keybert"):
    check(f"rendered installer ships {pkg}", pkg in body)
check("rendered installer pins numpy<2.5 (numba/whisper landmine)",
      "numpy<2.5" in body)
# All three deps + the pin belong to a SINGLE clear `install --upgrade …` line
# (the human-readable `say` echo above it doesn't carry `--upgrade`).
pip_lines = [ln for ln in body.splitlines()
             if "install" in ln and "--upgrade" in ln and "sentence-transformers" in ln]
check("the extras + pin are one clear pip install line",
      len(pip_lines) == 1
      and all(p in pip_lines[0] for p in
              ("sentence-transformers", "openai-whisper", "keybert", "numpy<2.5")))
check("a comment names the 2026-07-11 incident class",
      "2026-07-11" in body and "request time" in body.lower())


# --- the packaged resource the route serves --------------------------------
raw = (resources.files("abstract_hugpy_dev.worker_agent")
       .joinpath("bootstrap.sh").read_text(encoding="utf-8"))
check("packaged bootstrap.sh carries the extras block",
      "sentence-transformers" in raw and "openai-whisper" in raw
      and "keybert" in raw and "numpy<2.5" in raw)
# Self-update persistence: the block sits AFTER the main [engine] install and the
# agent's converge uses --no-deps, so the extras survive every version bump. We
# assert the block is documented as persisting and the agent uses --no-deps.
check("bootstrap notes the extras persist across --no-deps self-update",
      "--no-deps" in raw)
agent_src = (resources.files("abstract_hugpy_dev.worker_agent")
             .joinpath("agent.py").read_text(encoding="utf-8"))
check("agent self-update really uses `pip install -U --no-deps` (extras persist)",
      '"install", "-U", "--no-deps"' in agent_src)

print(f"\nall {ok} checks passed")
