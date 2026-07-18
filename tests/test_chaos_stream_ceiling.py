"""chaos client: /chat/stream must honour a HARD wall-clock ceiling even when
the server streams 'awaiting-load' progress forever (the exact hang a broken
model — e.g. a smashed tiny-random artifact stuck at load progress 0.0 —
produced; the per-socket-read timeout never fires while data keeps flowing).

Run:  venv/bin/python tests/test_chaos_stream_ceiling.py
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract_hugpy_dev.chaos.client import CentralClient

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

cancels: list[str] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        if self.path.startswith("/llm/chat/cancel/"):
            cancels.append(self.path.rsplit("/", 1)[-1])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"cancelled": true}')
            return
        if self.path == "/chat/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # stream 'awaiting-load' progress FOREVER (never a token/done)
            try:
                for _ in range(2000):
                    ev = {"type": "status", "stage": "awaiting-load",
                          "worker_name": "faux", "progress": 0.0}
                    self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.1)
            except Exception:
                pass
            return
        self.send_response(404)
        self.end_headers()


srv = HTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

client = CentralClient(f"http://127.0.0.1:{port}")
t0 = time.time()
term = client.chat_stream("faux-model", "hi", "chaos-ceiling-test",
                          max_new_tokens=8, ceiling_s=1.5, read_timeout_s=1.0)
elapsed = time.time() - t0

check("chat_stream RETURNS (does not hang) on an endless load stream",
      term is not None)
check("returned within a small multiple of the ceiling (no hang)",
      elapsed < 6.0)
check("outcome is load-timeout (no token before the wall ceiling)",
      term["outcome"] == "load-timeout")
check("captured the loading worker from the status stream",
      term["served_worker"] == "faux")
check("recorded awaiting-load stages", any(
      s[0] == "awaiting-load" for s in term["stages"]))
check("error explains the wall-ceiling breach",
      term["error"] and "ceiling" in term["error"])
check("the timed-out job was CANCELLED (polite guest)",
      "chaos-ceiling-test" in cancels)

srv.shutdown()
print(f"\nALL {ok} stream-ceiling checks passed")
