#!/usr/bin/env python3
"""Deterministic, dependency-free fake TEI embedding server.

Mimics the subset of the HuggingFace Text Embeddings Inference (TEI) HTTP API
that MinSync's `tei:` embedder uses:

    POST /embed   body: {"inputs": [str, ...], "normalize": bool, ...}
                  resp: [[float, ...], ...]   (one L2-normalised vector per input)

The vector for a text is derived from sha256(text) so it is fully
deterministic and identical across processes/restarts, which makes two MinSync
binaries produce byte-identical indexes for identical workloads.

Extra endpoints for the benchmark harness:
    GET  /stats   -> {"requests": n, "inputs": m, "latency_ms": l, "dim": d}
    POST /reset   -> resets the counters
    GET  /health  -> 200 "ok"

No third-party packages; standard library only.
"""

import argparse
import hashlib
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_DIM = 16


def embed_text(text: str, dim: int) -> list:
    """Deterministic L2-normalised vector of length `dim` for `text`."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw = []
    block_index = 0
    while len(raw) < dim:
        block = hashlib.sha256(seed + block_index.to_bytes(4, "big")).digest()
        for byte in block:
            raw.append((byte / 255.0) * 2.0 - 1.0)
            if len(raw) >= dim:
                break
        block_index += 1
    norm = math.sqrt(sum(value * value for value in raw))
    if norm == 0.0:
        norm = 1.0
    return [value / norm for value in raw]


class _State:
    def __init__(self, dim: int, latency_ms: float):
        self.dim = dim
        self.latency_ms = latency_ms
        self.lock = threading.Lock()
        self.requests = 0
        self.inputs = 0

    def record(self, n_inputs: int) -> None:
        with self.lock:
            self.requests += 1
            self.inputs += n_inputs

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "requests": self.requests,
                "inputs": self.inputs,
                "latency_ms": self.latency_ms,
                "dim": self.dim,
            }

    def reset(self) -> None:
        with self.lock:
            self.requests = 0
            self.inputs = 0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _State = None  # set by main()

    def log_message(self, *_args):  # silence per-request logging
        pass

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/stats":
            self._send_json(200, self.state.snapshot())
        elif path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/reset":
            self.state.reset()
            self._send_json(200, {"ok": True})
            return
        if path != "/embed":
            self._send_json(404, {"error": "not found"})
            return

        raw = self._read_body()
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            self._send_json(400, {"error": f"bad json: {error}"})
            return

        inputs = payload.get("inputs", payload.get("input"))
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list):
            self._send_json(400, {"error": "inputs must be a string or list of strings"})
            return

        if self.state.latency_ms > 0:
            time.sleep(self.state.latency_ms / 1000.0)

        vector_dim = self.state.dim
        embeddings = [embed_text(str(text), vector_dim) for text in inputs]
        self.state.record(len(inputs))
        self._send_json(200, embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake TEI embedding server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="artificial per-request latency (default 0)",
    )
    args = parser.parse_args()

    _Handler.state = _State(args.dim, args.latency_ms)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True
    print(
        f"fake_tei listening on http://{args.host}:{args.port} "
        f"(dim={args.dim}, latency_ms={args.latency_ms})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
