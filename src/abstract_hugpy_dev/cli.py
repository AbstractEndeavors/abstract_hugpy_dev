"""hugpy command line.

    hugpy serve  [--host 0.0.0.0] [--port 7002] [--auth open|external] ...
    hugpy worker --central https://your-hugpy/ [worker_agent args...]
    hugpy bot    [--central http://127.0.0.1:7002] [--env PATH] [--guild ID]

`serve` runs the whole product from one process: the API, the built web
console (when a ui/dist exists — see flask_app._ui_dist_dir), model downloads,
chat, and the OpenAI-compatible /v1 surface. No nginx, no node.

`worker` joins this machine to a hugpy central as a GPU worker (or, with
--role rpc, lends its GPU to the cross-machine shard pool). All flags after
the subcommand go straight to the worker agent's own parser.

`keeper` opens the stationary keeper REPL: a model from a hugpy central keeps
this machine (or an LXD instance via --exec lxc:<name>) — chat plus a shell
action loop. See hugpy/keeper.py; it is stdlib-only and also runs straight
from the source tree without installing hugpy.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def _serve(args: argparse.Namespace) -> int:
    # Distribution default: single-operator instance, no login wall. The
    # /v1 API-key system still gates programmatic access. Deployments that
    # front a real auth service set --auth external (or HUGPY_AUTH_MODE).
    if args.auth:
        os.environ["HUGPY_AUTH_MODE"] = args.auth
    else:
        os.environ.setdefault("HUGPY_AUTH_MODE", "open")

    from abstract_hugpy_dev.flask_app import get_hugpy_flask

    origins = [o.strip() for o in (args.origins or "").split(",") if o.strip()] or None
    flask_app = get_hugpy_flask(name="hugpy", allowed_origins=origins, debug=args.debug)

    bind = f"{args.host}:{args.port}"
    try:
        from gunicorn.app.base import BaseApplication
    except ImportError:
        # gunicorn is POSIX-only. On Windows (or anywhere it's missing) prefer
        # waitress — a production-grade pure-Python WSGI server — and fall back
        # to the Flask dev server only if neither is available.
        try:
            from waitress import serve as _waitress_serve
        except ImportError:
            print(f"hugpy: gunicorn/waitress not installed; using the Flask dev server on {bind}",
                  file=sys.stderr)
            flask_app.run(host=args.host, port=args.port, debug=args.debug)
            return 0
        print(f"hugpy serving on http://{bind}  (console at /, API at /api/v1)  [waitress]")
        print(f"  first run? finish setup at  http://{bind}/welcome")
        _waitress_serve(flask_app, host=args.host, port=args.port, threads=args.threads)
        return 0

    class _App(BaseApplication):
        def load_config(self):
            self.cfg.set("bind", bind)
            self.cfg.set("workers", 1)          # singleton registries/job store
            self.cfg.set("threads", args.threads)
            self.cfg.set("timeout", 300)

        def load(self):
            return flask_app

    print(f"hugpy serving on http://{bind}  (console at /, API at /api/v1)")
    print(f"  first run? finish setup at  http://{bind}/welcome")
    _App().run()
    return 0


def _worker(_args: argparse.Namespace, passthrough: list[str]) -> int:
    from abstract_hugpy_dev.worker_agent.agent import main as worker_main
    return worker_main(passthrough)


def _bot(args: argparse.Namespace) -> int:
    """Run the discord bot arm — it drives a hugpy central over HTTP (by proxy
    of the console), so it can point at this machine or a remote central."""
    # config.py reads these at import time, so set them BEFORE importing the bot.
    if args.central:
        os.environ["HUGPY_BASE_URL"] = args.central
    if args.env:
        os.environ["HUGPY_BOT_ENV"] = args.env
    if args.guild:
        os.environ["GUILD_ID"] = str(args.guild)

    try:
        import discord  # noqa: F401
    except ImportError:
        print("hugpy bot: discord.py is not installed.\n"
              "  install it with:  pip install 'hugpy[bot]'   (or: pip install discord.py)",
              file=sys.stderr)
        return 1

    from abstract_hugpy_dev.bot.bot import HugpyBot
    from abstract_hugpy_dev.bot.config import get_discord_token
    try:
        token = get_discord_token()
    except RuntimeError as exc:
        print(f"hugpy bot: {exc}", file=sys.stderr)
        return 1

    # Importing the hugpy package pulls the root logger down to DEBUG, which makes
    # the bot's periodic outbox poll spew httpx/httpcore request logs into the
    # journal. Pin a sane level for the long-running bot; quiet the HTTP libs.
    logging.getLogger().setLevel(logging.INFO)
    for _noisy in ("httpx", "httpcore", "discord.http"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    HugpyBot().run(token)
    return 0


def _chat(args: argparse.Namespace) -> int:
    """DISC-01 — the terminal is just another transport on the same
    substrate: it streams /chat/stream with a request_id (so the run shows in
    the unified jobs view and the console can cancel it), authenticates with
    a real credential (principal token or API key — no implicit god-mode),
    and Ctrl-C cancels SERVER-SIDE via /llm/chat/cancel before exiting.
    Stdlib-only (urllib), like keeper.py, so it runs on thin installs."""
    import json as _json
    import signal
    import urllib.request
    import uuid as _uuid

    base = (args.central or os.environ.get("HUGPY_BASE_URL")
            or "http://127.0.0.1:7002").rstrip("/")
    token = (args.token or os.environ.get("HUGPY_TOKEN")
             or os.environ.get("HUGPY_API_KEY") or "").strip()

    def _post(path: str, payload: dict, stream: bool = False):
        req = urllib.request.Request(
            base + path, data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {token}"} if token else {})},
            method="POST")
        return urllib.request.urlopen(req, timeout=None if stream else 15)

    prompt = " ".join(args.prompt) if args.prompt else None
    if not prompt:
        try:
            prompt = input("hugpy> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
    if not prompt:
        return 0

    rid = _uuid.uuid4().hex
    body = {"prompt": prompt, "request_id": rid, "transport": "cli"}
    if args.model:
        body["model_key"] = args.model

    def _cancel(_sig=None, _frm=None):
        print("\n[cancelling server-side…]", file=sys.stderr)
        try:
            _post(f"/llm/chat/cancel/{rid}", {})
        except Exception:
            pass
        raise SystemExit(130)
    signal.signal(signal.SIGINT, _cancel)

    try:
        resp = _post("/chat/stream", body, stream=True)
    except Exception as exc:
        print(f"hugpy chat: cannot reach central at {base}: {exc}",
              file=sys.stderr)
        return 1
    finish = "stop"
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            event = _json.loads(line[5:].strip())
        except ValueError:
            continue
        etype = event.get("type")
        if etype == "token":
            print(event.get("text") or "", end="", flush=True)
        elif etype == "error":
            print(f"\nhugpy chat: {event.get('message')}", file=sys.stderr)
            return 1
        elif etype == "done":
            finish = event.get("finish_reason") or "stop"
            break
    print()
    if finish == "cancelled":
        print("[cancelled]", file=sys.stderr)
    return 0


def _install_engine(args: argparse.Namespace) -> int:
    """Provision the native llama.cpp binaries (llama-server / rpc-server).

    The in-process engine (llama-cpp-python) ships with `pip install hugpy`, so
    this is only needed for the always-on serve drivers and the GPU shard fleet.
    """
    from abstract_hugpy_dev.engine import build, fetch, resolve

    try:
        if args.build_from_source:
            info = build.build_from_source(cuda=args.cuda, tag=args.tag, jobs=args.jobs)
        else:
            try:
                info = fetch.install(cuda=args.cuda, tag=args.tag, force=args.force)
            except Exception as exc:
                print(f"hugpy: prebuilt fetch failed ({exc}); trying source build…",
                      file=sys.stderr)
                info = build.build_from_source(cuda=args.cuda, tag=args.tag, jobs=args.jobs)
    except Exception as exc:
        print(f"hugpy install-engine: failed: {exc}", file=sys.stderr)
        return 1

    print(f"hugpy engine ready — {info.get('note')}")
    print(f"  engine dir : {info.get('engine_dir')}")
    print(f"  llama-server: {info.get('server_bin') or resolve.server_bin()}")
    print(f"  rpc-server  : {info.get('rpc_bin') or '(not built)'}")
    return 0


def _detect_profile() -> tuple[str, str]:
    """Pick cpu-worker/gpu-worker by probing local hardware. Returns
    (profile, reason) — the reason is always printed, detection is never silent.
    Imports the worker agent's GPU probe lazily: it pulls the heavy managers/
    imports stack, which must stay optional for a bare `hugpy --help`."""
    try:
        from abstract_hugpy_dev.worker_agent.agent import detect_gpus
    except Exception as exc:
        return "cpu", f"GPU detection unavailable ({type(exc).__name__}: {exc}); defaulting to cpu-worker"

    try:
        gpus = detect_gpus()
    except Exception as exc:
        return "cpu", f"GPU detection failed ({type(exc).__name__}: {exc}); defaulting to cpu-worker"

    if not gpus:
        return "cpu", "no usable NVIDIA GPU detected; choosing cpu-worker"

    names = ", ".join(f"{g.get('name')} (index {g.get('index')})" for g in gpus)
    return "gpu", f"detected {len(gpus)} GPU(s): {names}; choosing gpu-worker"


def _install_deps(args: argparse.Namespace) -> int:
    """Install a worker box's dependency set based on what the box actually is,
    instead of a human guessing cpu-worker/gpu-worker by hand. See WORKER-SETUP.md
    §6 for the manual per-box recipe this complements (native/CUDA overrides
    still need that recipe — this only resolves the pip extras).

    Most fleet boxes have a GPU, so gpu-worker is the default profile — --cpu
    (or --profile cpu) is the explicit opt-out for CPU-only boxes. --profile auto
    runs hardware detection instead of trusting the operator's say-so."""
    if args.cpu and args.profile not in (None, "cpu"):
        print(f"hugpy install-deps: --cpu conflicts with --profile {args.profile}", file=sys.stderr)
        return 2
    requested = "cpu" if args.cpu else (args.profile or "gpu")

    if requested == "auto":
        profile, reason = _detect_profile()
        print(f"hugpy install-deps: {reason}")
    else:
        profile = requested
        print(f"hugpy install-deps: profile = {profile}-worker "
              f"({'--cpu' if args.cpu else 'default' if args.profile is None else f'--profile {profile}'})")
        if profile == "gpu":
            _, detect_reason = _detect_profile()
            if not detect_reason.startswith("detected"):
                why = "it is the default" if args.profile is None else "you asked for it"
                print(f"hugpy install-deps: WARNING — {detect_reason.split('; ')[0]}; "
                      f"proceeding with gpu-worker anyway ({why}). "
                      f"Pass --cpu if this box has no GPU.")

    extra = f"{profile}-worker"
    pkg = f"abstract_hugpy_dev[{extra}]"
    if args.version:
        pkg += f"=={args.version}"

    cmd = [sys.executable, "-m", "pip", "install", pkg]
    print("hugpy install-deps: would run:" if args.dry_run else "hugpy install-deps: will run:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0

    if not args.yes:
        try:
            reply = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nhugpy install-deps: aborted", file=sys.stderr)
            return 1
        if reply not in ("y", "yes"):
            print("hugpy install-deps: aborted", file=sys.stderr)
            return 1

    import subprocess
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"hugpy install-deps: pip install failed (exit {proc.returncode})", file=sys.stderr)
        return proc.returncode
    print(f"hugpy install-deps: installed {pkg}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="hugpy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the hugpy console + API in one process")
    s.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    s.add_argument("--port", type=int, default=7002, help="bind port (default: 7002)")
    s.add_argument("--threads", type=int, default=8, help="server threads (default: 8)")
    s.add_argument("--auth", choices=("open", "external"),
                   help="auth mode (default: open, or HUGPY_AUTH_MODE)")
    s.add_argument("--origins", help="comma-separated CORS origins (default: same-origin only)")
    s.add_argument("--debug", action="store_true")

    w = sub.add_parser("worker", help="join a hugpy central as a worker",
                       add_help=False)   # the agent owns its own --help

    b = sub.add_parser("bot", help="run the hugpy discord bot (drives a central over HTTP)")
    b.add_argument("--central",
                   help="hugpy central base URL the bot calls "
                        "(default: http://127.0.0.1:7002 or HUGPY_BASE_URL)")
    b.add_argument("--env", help="path to a .env with DISCORD_TOKEN/settings (sets HUGPY_BOT_ENV)")
    b.add_argument("--guild", type=int, help="restrict slash-command sync to one guild id")

    sub.add_parser("keeper", help="terminal keeper REPL — a model keeps this "
                   "machine (or an LXD instance) via chat + shell actions",
                   add_help=False)       # keeper.py owns its own --help

    c = sub.add_parser("chat", help="stream a chat from a hugpy central in the "
                       "terminal (tracked + cancellable like every transport)")
    c.add_argument("prompt", nargs="*", help="the prompt (omit for interactive)")
    c.add_argument("--central", help="central base URL (default: HUGPY_BASE_URL "
                   "or http://127.0.0.1:7002)")
    c.add_argument("--model", help="model_key (default: central decides)")
    c.add_argument("--token", help="principal token (hpp_…) or API key "
                   "(default: HUGPY_TOKEN / HUGPY_API_KEY)")

    e = sub.add_parser("install-engine",
                       help="download/build the native llama.cpp server binary")
    e.add_argument("--cuda", action="store_true", help="fetch/build a CUDA-enabled engine")
    e.add_argument("--build-from-source", action="store_true",
                   help="cmake build instead of a prebuilt release (needs git+cmake)")
    e.add_argument("--tag", help="llama.cpp release tag (default: latest)")
    e.add_argument("--jobs", type=int, help="parallel build jobs (source build only)")
    e.add_argument("--force", action="store_true", help="re-download even if already installed")

    i = sub.add_parser("install-deps",
                       help="install a worker box's pip extras (gpu-worker by default, "
                            "--cpu for CPU-only boxes)")
    i.add_argument("--profile", choices=("gpu", "cpu", "auto"),
                   help="gpu-worker/cpu-worker, or auto to detect (default: gpu)")
    i.add_argument("--cpu", action="store_true", help="shorthand for --profile cpu")
    i.add_argument("--version", help="pin abstract_hugpy_dev to this version (default: unpinned)")
    i.add_argument("--dry-run", action="store_true",
                   help="print the pip command and exit without running it")
    i.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    # Split: everything after `worker` belongs to the agent's parser.
    if argv and argv[0] == "worker":
        return _worker(w, argv[1:])
    # Same for `keeper` — and import it WITHOUT pulling the hugpy package
    # __init__ (heavy deps), so `hugpy keeper` works even on a thin install.
    if argv and argv[0] == "keeper":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hugpy_keeper", os.path.join(os.path.dirname(__file__), "keeper.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(argv[1:])
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "install-engine":
        return _install_engine(args)
    if args.cmd == "install-deps":
        return _install_deps(args)
    if args.cmd == "bot":
        return _bot(args)
    if args.cmd == "chat":
        return _chat(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
