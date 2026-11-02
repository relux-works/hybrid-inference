"""Prototype prefill-node protocol over a single source-pinned vLLM server."""

import argparse
from dataclasses import dataclass
import http.server
import json
import os
from pathlib import Path
import re
import secrets
import struct
import tempfile
import threading
import time
from typing import cast

from bridge import (MAX_BODY, Refusal, body_size, chunk_transfer, convert, digest, fingerprint,
                    require, token_hash, validate_layout)
from node_backend import VllmBackend
from fast_codec import qualify_node_codec


@dataclass
class Session:
    tokens: list
    last_used: int


class Adapter:
    def __init__(self, target, clients, backend, scratch, context=65536, codec="auto", public=False):
        validate_layout(target["model_layout"])
        digest(target["model_sha256"], "model hash")
        require(target["layout_fingerprint"] == fingerprint(target["model_layout"]), "layout fingerprint mismatch")
        require(type(context) is int and 256 <= context <= 1_048_576, "invalid context size")
        require(isinstance(clients, dict) and clients, "clients must be a nonempty id-to-token object")
        require(all(isinstance(k, str) and isinstance(v, str) and 16 <= len(v) <= 256 for k, v in clients.items())
                and len(set(clients.values())) == len(clients), "client tokens must be distinct and 16-256 bytes long")
        require(type(backend.block_size) is int and 16 <= backend.block_size <= 4096 and backend.block_size % 16 == 0, "wrong block size")
        self.target, self.backend, self.scratch = target, backend, Path(scratch)
        self.owners = {"Bearer " + token: owner for owner, token in clients.items()}
        self.codec, self.codec_check = qualify_node_codec(codec)
        self.context, self.public = context, public
        self.sessions, self.lock, self.extends = {}, threading.Lock(), 0
        self.max_request = 1024**2
        self.block = backend.block_size

    def info(self, session):
        return dict(n=len(session.tokens), prefix_hash=token_hash(session.tokens), tier="ram",
                    last_used=session.last_used, model_sha256=self.target["model_sha256"])

    def health(self, owner):
        value = dict(delta_version=2, status="ok" if self.backend.healthy() else "unavailable", **self.target,
                     max_ctx=self.context, n_ctx=self.context, ram_budget_bytes=128 * self.context * 4,
                     max_request_bytes=self.max_request, export_block_size=self.block,
                     backend="vllm", state_storage="prefix-cache-and-token-journal")
        if not self.public or owner is not None:
            value.update(sessions=len(self.sessions), ram_bytes=sum(len(s.tokens) * 4 for _, s in self.sessions.values()), evictions=0)
        return value

    def handle(self, method, path, body, authorization):
        owner = self.owners.get(authorization)
        if method == "GET" and path == "/v1/health":
            health = self.health(owner)
            return (200 if health["status"] == "ok" else 503), health, {}
        if owner is None:
            return 401, dict(error="unauthorized"), {"WWW-Authenticate": "Bearer"}
        if method == "GET" and path == "/v1/bench/link":
            return 200, b"x" * 1024**2, {}
        if method == "GET" and path == "/metrics":
            return 200, ("hybrid_extends_total %d\nhybrid_sessions %d\n" % (self.extends, len(self.sessions))).encode(), {}
        if not self.lock.acquire(blocking=False):
            return 429, dict(error="node busy"), {"Retry-After": "1"}
        try:
            try:
                return self.apply(method, path, body, owner)
            except (Refusal, KeyError, TypeError, ValueError):
                return 400, dict(error="request could not be applied"), {}
            except (OSError, TimeoutError):
                return 503, dict(error="vLLM backend unavailable"), {}
        finally:
            self.lock.release()

    def apply(self, method, path, body, owner):
        if path == "/v1/sessions":
            if method == "GET":
                return 200, [dict(session_id=sid, **self.info(s)) for sid, (client, s) in self.sessions.items() if client == owner], {}
            if method != "POST":
                return 404, dict(error="unknown endpoint"), {}
            if body["model_sha256"] != self.target["model_sha256"]:
                return 400, dict(error="model hash mismatch"), {}
            if len(self.sessions) >= 128:
                return 503, dict(error="session limit"), {}
            sid = secrets.token_hex(16)
            self.sessions[sid] = owner, Session([], int(time.time()))
            return 200, dict(session_id=sid), {}
        match = re.fullmatch(r"/v1/sessions/([a-f0-9]{32})(/extend|/snapshot)?", path)
        if not match:
            return 404, dict(error="unknown endpoint"), {}
        sid, action = match.groups()
        client, session = self.sessions.get(sid, (None, None))
        if client != owner:
            return 404, dict(error="session not found"), {}
        if action is None and method == "DELETE":
            del self.sessions[sid]
            return 204, None, {}
        if action is None and method == "GET":
            return 200, self.info(session), {}
        if action == "/snapshot" and method == "GET":
            if not session.tokens or len(session.tokens) % self.block:
                return 400, dict(error="unaligned historical endpoint"), {}
            return self.snapshot(session)
        if action != "/extend" or method != "POST":
            return 404, dict(error="unknown endpoint"), {}
        if type(body["n0"]) is not int or not isinstance(body["prefix_hash_n0"], str):
            return 400, dict(error="invalid append boundary"), {}
        n0 = len(session.tokens)
        if body["n0"] != n0 or body["prefix_hash_n0"] != token_hash(session.tokens):
            return 409, dict(n=n0, prefix_hash=token_hash(session.tokens)), {}
        tokens, return_delta = body["tokens"], body["return_delta"]
        if (not isinstance(tokens, list) or not tokens or len(tokens) > self.context - n0 or type(return_delta) is not bool):
            return 400, dict(error="invalid tokens, context limit or return_delta"), {}
        if not all(type(t) is int and 0 <= t < self.target["model_layout"]["vocab_size"] for t in tokens):
            return 400, dict(error="token out of vocabulary"), {}
        transfer_keys = {"transfer_id", "transfer_n0", "transfer_n1", "chunk_index", "chunk_count"}
        if set(body) - {"n0", "prefix_hash_n0", "tokens", "return_delta", "delta_version"} - transfer_keys:
            return 400, dict(error="unsupported execution geometry or historical endpoint"), {}
        all_tokens = session.tokens + tokens
        requested = len(all_tokens)
        n1 = requested // self.block * self.block if return_delta else requested
        if return_delta and (n1 <= n0 or body_size(n1 - n0) > MAX_BODY):
            return 400, dict(error="no aligned endpoint or delta exceeds body limit"), {}
        if "delta_version" in body:
            require(type(body["delta_version"]) is int and body["delta_version"] == 2, "unsupported delta version")
        transfer = {key: body[key] for key in transfer_keys if key in body}
        if transfer:
            require(return_delta and n1 == requested, "chunk endpoint must be exact")
            chunk_transfer(transfer, n0, n1, sid)
            require(transfer["transfer_n1"] <= self.context and transfer["transfer_n1"] % self.block == 0, "invalid transfer endpoint")
        accepted = all_tokens[:n1]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            capture = self.backend.prefill(accepted, n0, root, capture=return_delta)
            if return_delta:
                delta = root / "output.delta"
                header = convert(capture, self.target, delta, self.codec, sid, transfer or None)
                require(header["n0"] == n0 and header["n1"] == n1 and header["prefix_hash_n1"] == token_hash(accepted), "backend capture boundary mismatch")
                wire = delta.read_bytes()
            else:
                wire = None
        # A failed capture/conversion leaves the authoritative session journal unchanged.
        session.tokens, session.last_used = accepted, int(time.time())
        self.extends += 1
        headers = {"X-Hybrid-Prefill-Ms": str((time.perf_counter() - started) * 1000),
                   "X-Hybrid-Export-N1": str(n1), "X-Hybrid-Requested-N1": str(requested)}
        return (200, wire, headers) if return_delta else (202, dict(n=n1, prefix_hash=token_hash(accepted)), headers)

    def snapshot(self, session):
        if body_size(len(session.tokens)) > MAX_BODY:
            return 400, dict(error="snapshot exceeds body limit"), {}
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            capture = self.backend.prefill(session.tokens, 0, root)
            path = root / "snapshot.delta"
            convert(capture, self.target, path, self.codec)
            with path.open("rb") as source:
                size, = struct.unpack("<I", source.read(4))
                source.seek(4 + size)
                state = source.read(body_size(len(session.tokens)))
        # Native slot files have their own magic/version/token journal, then sequence state after its 8-byte header.
        wire = struct.pack("<III", 0x67677371, 4, len(session.tokens))
        wire += struct.pack("<" + "i" * len(session.tokens), *session.tokens) + state[8:]
        return 200, wire, {}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def serve(self):
        app = cast(Server, self.server).app
        self.connection.settimeout(30)
        try:
            authorization = self.headers.get("Authorization", "")
            if authorization not in app.owners and self.path != "/v1/health":
                status, value, headers = 401, dict(error="unauthorized"), {"WWW-Authenticate": "Bearer"}
            else:
                length = int(self.headers.get("Content-Length", "0"))
                require(0 <= length <= app.max_request and not self.headers.get("Transfer-Encoding"), "invalid request size")
                raw = self.rfile.read(length)
                require(len(raw) == length, "truncated request")
                body = json.loads(raw) if raw else {}
                require(isinstance(body, dict), "invalid JSON fields")
                status, value, headers = app.handle(self.command, self.path, body, authorization)
        except (ValueError, RecursionError):
            status, value, headers = 400, dict(error="invalid JSON fields or request size"), {}
        except OSError:
            status, value, headers = 408, dict(error="request timeout or disconnect"), {}
        binary = isinstance(value, bytes)
        data = value if binary else json.dumps(value).encode() if value is not None else b""
        self.send_response(status)
        content_type = "text/plain" if self.path == "/metrics" and binary else "application/octet-stream" if binary else "application/json"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_GET = do_POST = do_DELETE = serve


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    app: Adapter

    def __init__(self, address, handler=Handler):
        super().__init__(address, handler)
        self.workers = threading.BoundedSemaphore(4)

    def process_request(self, request, client_address):
        self.workers.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.workers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.workers.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "clients", "scratch", "control"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8093")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--codec", choices=("reference", "numpy", "torch", "auto"), default="auto")
    args = parser.parse_args()
    require(args.clients.stat().st_mode & 0o077 == 0, "private client token file required")
    target, clients = [json.loads(p.read_text()) for p in (args.target, args.clients)]
    args.scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    backend = VllmBackend(args.vllm_url, args.control, target, args.scratch)
    backend.probe()
    app = Adapter(target, clients, backend, args.scratch, args.context, args.codec,
                  public=args.host not in ("localhost", "127.0.0.1", "::1"))
    (args.scratch / "codec-check.json").write_text(json.dumps(dict(codec=app.codec, cuda_check=app.codec_check)) + "\n")
    server = Server((args.host, args.port))
    server.app = app
    server.serve_forever()


if __name__ == "__main__":
    main()
