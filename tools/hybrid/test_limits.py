"""CPU node limits, LRU replay and pinned response regressions; no model downloads."""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import pathlib
import secrets
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request

from test_prefill import prefix

MIB = 1024 * 1024


class Node:
    def __init__(self, args, name, flags=(), model=None, context="10000"):
        self.directory = pathlib.Path(args.artifacts) / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clients = {key: secrets.token_hex(24) for key in ("a", "b", "c")}
        credentials = self.directory / "clients.json"
        credentials.touch(mode=0o600)
        credentials.chmod(0o600)
        credentials.write_text(json.dumps(self.clients))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        command = [args.binary, model or args.model, str(credentials), str(self.port)]
        if context is not None:
            command.append(context)
        command.extend(flags)
        self.log = (self.directory / "node.log").open("wb")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
        (self.directory / "pid.json").write_text(json.dumps({"pid": self.process.pid, "command": command}))
        self.base = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        try:
            for _ in range(1200):
                if self.process.poll() is not None:
                    raise AssertionError("node startup failed; see private log")
                try:
                    self.health = self.request("GET", "/v1/health")
                    return self
                except urllib.error.URLError:
                    time.sleep(0.1)
            raise AssertionError("node startup timeout")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        (self.directory / "exit.json").write_text(json.dumps({"pid": self.process.pid, "exit_code": self.process.returncode}))

    def request(self, method, path, body=None, client="a", expected=200, raw=None):
        data = raw if raw is not None else None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Authorization": "Bearer " + self.clients[client], "Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            payload = response.read()
            assert response.code in (expected if isinstance(expected, tuple) else (expected,)), (method, path, response.code, expected, payload[:200])
            if "application/json" in response.headers.get("Content-Type", ""):
                return json.loads(payload)
            return payload

    def create(self, client="a"):
        return "/v1/sessions/" + self.request("POST", "/v1/sessions", {"model_sha256": self.health["model_sha256"]}, client)["session_id"]

    def extend(self, path, tokens, previous=(), client="a", delta=False, expected=None):
        return self.request("POST", path + "/extend", {"n0": len(previous), "prefix_hash_n0": prefix(previous),
                            "tokens": tokens, "return_delta": delta}, client, expected or (200 if delta else 202))

    def stall(self, path, client="a", body=None):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        sock.settimeout(3600)
        sock.connect(("127.0.0.1", self.port))
        data = b"" if body is None else json.dumps(body).encode()
        method = "GET" if body is None else "POST"
        headers = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {self.clients[client]}\r\n"
                   f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode()
        sock.sendall(headers + data)
        return sock


def response_headers(sock):
    result = b""
    while b"\r\n\r\n" not in result:
        piece = sock.recv(1)
        assert piece, "closed before response headers"
        result += piece
    return int(result.split(b" ", 2)[1])


def test_lru_replay_and_atomic_refusal(args):
    # Three sessions compete for two states: reads refresh LRU; replay restores the exact evicted bytes and tokens.
    with Node(args, "lru", ("--ram-budget-mib", "48")) as node:
        a, b, c = [node.create() for _ in range(3)]
        node.extend(a, [100, 101])
        node.extend(b, [102, 103])
        b_before = node.request("GET", b + "/snapshot")
        a_before = node.request("GET", a + "/snapshot")
        node.extend(c, [104, 105])
        node.request("GET", b, expected=404)
        assert node.request("GET", a + "/snapshot") == a_before
        health = node.request("GET", "/v1/health")
        assert health["evictions"] == 1 and health["sessions"] == 2 and 0 < health["ram_bytes"] <= 48 * MIB
        replay = node.create()
        node.extend(replay, [102, 103])
        assert node.request("GET", replay + "/snapshot") == b_before
        assert node.request("GET", replay)["prefix_hash"] == prefix([102, 103])
        node.request("GET", c, expected=404)
        print("PASS test_lru_replay: oldest idle evicted, read refreshes recency, 404 then replay byte-identical", flush=True)

        # A state larger than the entire budget rejects repeatedly, preserving the target and all neighbors.
        failed = node.create()
        paths = [a, replay, failed]
        snapshots = {path: node.request("GET", path + "/snapshot") for path in paths}
        before = node.request("GET", "/v1/health")
        for _ in range(2):
            refusal = node.extend(failed, [100] * 6000, expected=503)
            assert set(refusal) == {"error", "need", "budget"}
            assert refusal["error"] == "ram budget" and refusal["need"] > refusal["budget"] == 48 * MIB
            assert {path: node.request("GET", path + "/snapshot") for path in paths} == snapshots
            after = node.request("GET", "/v1/health")
            assert all(after[key] == before[key] for key in ("ram_bytes", "sessions", "evictions"))
        node.extend(failed, [106])
        assert node.request("GET", failed)["n"] == 1
        print(f"PASS test_atomic_budget_refusal: {refusal}; every session byte-identical; subsequent valid append accepted", flush=True)


def test_pinned_read_and_extend(args):
    # A stalled snapshot remains pinned while a concurrent extend evicts the other idle session.
    with Node(args, "pinned", ("--ram-budget-mib", "48")) as node:
        a, b, c = node.create(), node.create("b"), node.create("c")
        node.extend(a, [100])
        node.extend(b, [101], client="b")
        with contextlib.closing(node.stall(a + "/snapshot")) as reader:
            assert response_headers(reader) == 200
            node.request("GET", b, client="b")  # Make the pinned reader older than the idle neighbor.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(node.extend, c, [102], (), "c").result(timeout=60)
            assert node.request("GET", a)["n"] == 1
            node.request("GET", b, client="b", expected=404)
            assert node.request("GET", "/v1/health")["evictions"] == 1
        print("PASS test_pinned_snapshot: concurrent append evicts idle neighbor, streaming read survives", flush=True)

    # While an extend computes and streams its delta, a later competing append cannot evict its session.
    with Node(args, "extending", ("--ram-budget-mib", "48")) as node:
        a, b = node.create(), node.create("b")
        node.extend(b, [101], client="b")
        body = {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": [100] * 1000, "return_delta": True}
        with contextlib.closing(node.stall(a + "/extend", body=body)) as stream:
            assert response_headers(stream) == 200
            node.request("GET", b, client="b")  # The active response is now the oldest candidate.
            c = node.create("c")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(node.extend, c, [102], (), "c").result(timeout=180)
            assert node.request("GET", a)["n"] == 1000
            node.request("GET", b, client="b", expected=404)
        print("PASS test_pinned_extend: concurrent extend/eviction preserves computing and streaming session", flush=True)

    # If all reclaimable memory is pinned, reject with no eviction; disconnect makes the same append admissible.
    with Node(args, "all-pinned", ("--ram-budget-mib", "32")) as node:
        a, b = node.create(), node.create("b")
        node.extend(a, [100])
        before = node.request("GET", b + "/snapshot", client="b")
        with contextlib.closing(node.stall(a + "/snapshot")) as reader:
            assert response_headers(reader) == 200
            refusal = node.extend(b, [101], client="b", expected=503)
            assert refusal["error"] == "ram budget" and refusal["need"] > refusal["budget"] == 32 * MIB
            assert node.request("GET", b + "/snapshot", client="b") == before
            assert node.request("GET", a)["n"] == 1
            assert node.request("GET", "/v1/health")["evictions"] == 0
        for _ in range(100):
            # Poll the same admissible request until disconnection releases the response pin.
            result = node.extend(b, [101], client="b", expected=(202, 503))
            if "error" not in result:
                break
            assert result["error"] == "ram budget"
            time.sleep(0.05)
        else:
            raise AssertionError("reader pin leaked after disconnect")
        node.request("GET", a, expected=404)
        print("PASS test_all_pinned_refusal: exact 503, no mutation/eviction, disconnect releases pin", flush=True)


def test_request_caps_and_metrics(args):
    # Default limits and inclusive byte boundary remain; one byte over refuses before mutation. A larger flag is honored by both parsers.
    for limit in (1, 2):
        flags = () if limit == 1 else ("--max-request-mib", str(limit))
        with Node(args, f"request-{limit}", flags) as node:
            health = node.health
            assert health["max_ctx"] == 16384 and health["ram_budget_bytes"] == 512 * MIB
            assert health["max_request_bytes"] == limit * MIB and health["sessions"] == health["evictions"] == health["ram_bytes"] == 0
            assert len(health["model_sha256"]) == len(health["layout_fingerprint"]) == 64
            assert health["model_layout"]["vocab_size"] > 100000
            path = node.create()
            before = node.request("GET", path + "/snapshot")
            body = json.dumps({"n0": 0, "prefix_hash_n0": prefix([]), "tokens": [100], "return_delta": False}).encode()
            node.request("POST", path + "/extend", raw=body + b" " * (limit * MIB + 1 - len(body)), expected=413)
            assert node.request("GET", path + "/snapshot") == before
            node.request("POST", path + "/extend", raw=body + b" " * (limit * MIB - len(body)), expected=202)
            metrics = node.request("GET", "/metrics").decode()
            for key in ("max_ctx", "ram_budget_bytes", "ram_bytes", "sessions", "evictions", "max_request_bytes"):
                assert f"hybrid_{key} {node.request('GET', '/v1/health')[key]}\n" in metrics
            assert health["model_sha256"] in metrics and health["layout_fingerprint"] in metrics
        print(f"PASS test_request_boundary_{limit}mib: cap accepted; cap+1=413 and unchanged snapshot; metrics agree", flush=True)


def test_invalid_limits(args):
    # Invalid, zero, negative and overflowing CLI bounds refuse at startup before any listener or inference.
    for flag, value in (("--max-ctx", "255"), ("--max-ctx", "2147483648"),
                        ("--ram-budget-mib", "0"), ("--ram-budget-mib", "-1"),
                        ("--max-request-mib", "0"), ("--max-request-mib", "18446744073709551615"),
                        ("--max-ctx", "65536junk")):
        node = Node(args, "invalid-" + flag[2:] + "-" + value, (flag, value), context=None)
        try:
            assert node.process.wait(timeout=10) == 1
        finally:
            node.__exit__(None, None, None)
    print("PASS test_invalid_limits: seven invalid CLI bounds refused with exit 1", flush=True)



def test_model_identity(args):
    # Real re-quantization changes file SHA but preserves layout; no separate base model is downloaded.
    assert args.requantized, "--requantized required for a real two-quantization control"
    with Node(args, "identity-original", context="256") as node:
        original = node.health
    with Node(args, "identity-requantized", model=args.requantized, context="256") as node:
        requantized = node.health
    assert original["model_sha256"] != requantized["model_sha256"]
    assert original["layout_fingerprint"] == requantized["layout_fingerprint"]
    assert original["model_layout"] == requantized["model_layout"]
    pathlib.Path(args.artifacts, "identity.json").write_text(json.dumps({"original": original, "requantized": requantized}, indent=2))
    print(f"PASS test_requantized_identity: sha differs, layout={original['layout_fingerprint']} matches", flush=True)


def test_deep_context(args):
    # The configured 65536 context accepts a 39999-token delta for a 40000-token prompt and retains default wire bounds.
    with Node(args, "deep", ("--max-ctx", "65536"), context=None) as node:
        assert node.health["n_ctx"] == node.health["max_ctx"] == 65536
        path = node.create()
        tokens = [100, 101, 102, 103] * 10000
        wire = node.extend(path, tokens[:-1], delta=True)
        header_size = struct.unpack_from("<I", wire)[0]
        header = json.loads(wire[4:4 + header_size])
        assert header["n1"] == 39999 and header["tokens"] == tokens[:-1]
        assert header["prefix_hash_n1"] == prefix(tokens[:-1])
        assert header_size <= MIB and len(wire) - header_size - 52 <= 512 * MIB
        directory = pathlib.Path(args.artifacts)
        (directory / "deep.delta").write_bytes(wire)
        fixture = {"tokens": tokens, "model_sha256": node.health["model_sha256"], "n0": 0}
        (directory / "deep.tokens.json").write_text(json.dumps(fixture))
        health = node.request("GET", "/v1/health")
        (directory / "deep-metrics.json").write_text(json.dumps(health, indent=2))
        assert health["sessions"] == 1 and health["ram_bytes"] > 256 * MIB
        node.extend(path, [100] * (65536 - 39999 + 1), tokens[:-1], expected=400)
        assert node.request("GET", path)["n"] == 39999
        print(f"PASS test_deep_http: 39999-token delta={len(wire)} bytes, ram_bytes={health['ram_bytes']}; over-context=400", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--requantized")
    parser.add_argument("--case", choices=("limits", "identity", "deep", "pinned", "requests"), default="limits")
    args = parser.parse_args()
    if args.case == "identity":
        test_model_identity(args)
    elif args.case == "deep":
        test_deep_context(args)
    elif args.case == "pinned":
        test_pinned_read_and_extend(args)
    elif args.case == "requests":
        test_request_caps_and_metrics(args)
        test_invalid_limits(args)
    else:
        test_lru_replay_and_atomic_refusal(args)
        test_pinned_read_and_extend(args)
        test_request_caps_and_metrics(args)
        test_invalid_limits(args)


if __name__ == "__main__":
    main()
