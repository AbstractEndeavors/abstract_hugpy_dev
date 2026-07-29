"""``python -m abstract_hugpy_dev.downloader`` — same entry as the
``hugpy-downloader`` console script, so the systemd unit can run the daemon
straight out of the venv without depending on a re-install to place the script.
"""
from .daemon import main

if __name__ == "__main__":
    raise SystemExit(main())
