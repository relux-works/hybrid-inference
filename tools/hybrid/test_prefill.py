"""Exercise HTTP append invariants and namespace isolation against the small CPU model."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import secrets
import select
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request


def prefix(tokens):
    return hashlib.sha256(b"".join(struct.pack("<i", token) for token in tokens)).hexdigest()


def test_slow_readers_do_not_starve_critical(request, clients, port, model_hash):
    # Four stalled X requests cannot keep Y from completing within 5 s; excess streams reject before mutation.
    def create(client):
        sid = request("POST", "/v1/sessions", client, {"model_sha256": model_hash})["session_id"]
        return "/v1/sessions/" + sid

    paths = [create("a") for _ in range(4)]
    other, third = create("b"), create("c")
    empty_snapshot = request("GET", paths[0] + "/snapshot")
    held = []

    def stall(path, client, method="POST"):
        data = b"" if method == "GET" else json.dumps({"n0": 0, "prefix_hash_n0": prefix([]), "tokens": [100], "return_delta": True}).encode()
        sock = socket.socket()
        held.append(sock)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", port))
        headers = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {clients[client]}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode()
        sock.sendall(headers + data)
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(1)
            assert chunk, "response closed before headers"
            response += chunk
        return int(response.split(b" ", 2)[1])

    def release():
        for sock in held:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        held.clear()

    try:
        statuses = [stall(path + "/extend", "a") for path in paths]
        append = {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": [101], "return_delta": True}
        start = time.monotonic()
        wire = request("POST", other + "/extend", "b", append, timeout=5)
        elapsed = time.monotonic() - start
        assert elapsed < 5 and wire[:4], "critical Y did not complete within 5 s"
        assert request("GET", other, "b", timeout=5)["n"] == 1
        assert statuses == [200, 200, 429, 429], statuses
        request("GET", paths[0], expected=429, timeout=5)
        # Three global streams still leave an HTTP worker for control requests and bounded rejection.
        other_snapshot = request("GET", other + "/snapshot", "b")
        assert stall(third + "/extend", "c") == 200
        blocked = dict(append, n0=1, prefix_hash_n0=prefix([101]))
        request("POST", other + "/extend", "b", blocked, expected=429, timeout=5)
        request("GET", other, "b", expected=429, timeout=5)
        assert request("GET", "/v1/health", timeout=5)["status"] == "ok"
        request("GET", paths[0] + "/snapshot", "b", expected=429, timeout=5)
        release()
        request("GET", paths[0] + "/snapshot", "b", expected=404, retry_overload=True)
        for path, status in zip(paths, statuses):
            assert request("GET", path, retry_overload=True)["n"] == (1 if status == 200 else 0)
        assert request("GET", other + "/snapshot", "b", retry_overload=True) == other_snapshot
        for path in paths[2:]:
            assert request("GET", path + "/snapshot", retry_overload=True) == empty_snapshot
        # Disconnection and normal completion both release permits; snapshots share the same budget.
        assert stall(paths[0] + "/snapshot", "a", "GET") == 200
        assert stall(paths[1] + "/snapshot", "a", "GET") == 200
        request("GET", paths[2] + "/snapshot", expected=429, timeout=5)
        request("POST", paths[2] + "/extend", body=append, expected=429, timeout=5)
        request("GET", paths[2], expected=429, timeout=5)
        request("POST", other + "/extend", "b", blocked, timeout=5)
        assert request("GET", other, "b")["n"] == 2
        release()
        assert request("GET", paths[2] + "/snapshot", retry_overload=True) == empty_snapshot
        assert request("GET", paths[2], retry_overload=True)["n"] == 0
        wire = request("POST", paths[2] + "/extend", body=append, retry_overload=True)
        assert wire[:4] and request("GET", paths[2])["n"] == 1
        for _ in range(3):
            assert request("GET", paths[3] + "/snapshot", retry_overload=True) == empty_snapshot
        print(f"PASS test_slow_readers_do_not_starve_critical: four X attempts {statuses}; Y completed within 5 s")
        print("PASS test_stream_budget_lifetime: global admission reserve, snapshot limits, 429 unchanged state, disconnect/completion release")
    finally:
        release()


def test_pending_admission(request, clients, port, model_hash):
    # Pending shadows and snapshots count against both quotas; controls/rejections finish before the active prefill.
    paths = {client: ["/v1/sessions/" + request("POST", "/v1/sessions", client, {"model_sha256": model_hash})["session_id"] for _ in range(3)] for client in clients}
    before = {client: request("GET", paths[client][1] + "/snapshot", client) for client in clients}
    held = []

    def start(client, path, body=None, method="POST", stalled=False):
        data = b"" if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
        sock = socket.socket()
        held.append(sock)
        if stalled:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        sock.settimeout(180)
        sock.connect(("127.0.0.1", port))
        sock.sendall((f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {clients[client]}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode() + data)
        return sock

    def close(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        held.remove(sock)

    def unreadable(sockets):
        assert not select.select(sockets, [], [], 0)[0], "prefill finished before the queued admission control"

    def overload(method, path, client, body=None):
        began = time.monotonic()
        request(method, path, client, body, expected=429, timeout=1)
        assert time.monotonic() - began < 1, "429 waited behind the model"

    def append(tokens, return_delta):
        return {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": tokens, "return_delta": return_delta}

    def drain(sock, expected):
        import http.client
        response = http.client.HTTPResponse(sock)
        response.begin()
        assert response.status == expected, response.status
        payload = response.read()
        response.close()
        close(sock)
        return payload

    try:
        running = start("a", paths["a"][0] + "/extend", append([100] * 8192, False))
        time.sleep(0.1)
        unreadable([running])
        queued = start("a", paths["a"][1] + "/extend", append([101], False))
        time.sleep(0.1)
        unreadable([running, queued])
        overload("POST", paths["a"][2] + "/extend", "a", append([102], True))
        # Snapshots also reserve before waiting, and list/info/create/delete use the same admission path.
        waiting_snapshot = start("b", paths["b"][1] + "/snapshot", method="GET")
        time.sleep(0.1)
        for method, path, body in (
            ("POST", paths["c"][1] + "/extend", append([103], True)),
            ("POST", paths["c"][1] + "/extend", append([103], False)),
            ("GET", paths["c"][1] + "/snapshot", None),
            ("GET", "/v1/sessions", None),
            ("GET", paths["c"][1], None),
            ("POST", "/v1/sessions", {"model_sha256": model_hash}),
            ("DELETE", paths["a"][0], None),
        ):
            overload(method, path, "a" if method == "DELETE" else "c", body)
        assert request("GET", "/v1/health", client=None, timeout=1)["status"] == "ok"
        assert b"hybrid_extends_total " in request("GET", "/metrics", client=None, timeout=1)
        unreadable([running, queued, waiting_snapshot])
        # A disconnected queued shadow frees its client/global permit before the active prefill completes.
        close(queued)
        time.sleep(0.2)
        replacement = start("a", paths["a"][2] + "/extend", append([102], True))
        time.sleep(0.1)
        unreadable([running, waiting_snapshot, replacement])
        overload("GET", paths["c"][0] + "/snapshot", "c")
        close(waiting_snapshot)
        close(replacement)
        time.sleep(0.2)
        # Recovery admits fresh work while the original request is still computing.
        replacement_snapshot = start("c", paths["c"][0] + "/snapshot", method="GET")
        time.sleep(0.1)
        unreadable([running, replacement_snapshot])
        assert drain(replacement_snapshot, 200) == before["c"]
        assert json.loads(drain(running, 202))["n"] == 8192
        for client in clients:
            assert request("GET", paths[client][1] + "/snapshot", client, retry_overload=True) == before[client]
        assert request("GET", paths["a"][2], "a")["n"] == 0
        assert len(request("GET", "/v1/sessions", "c")) == 3
        assert request("GET", paths["a"][0], "a")["n"] == 8192
        print("PASS test_pending_admission: queued shadow/snapshot quotas 2/client and 3/global, immediate 429, health/metrics during prefill, cancelled queued work unchanged")

        # 400, 409, missing fields and malformed JSON return both permits, with unchanged session state.
        path = paths["a"][1]
        valid = append([104], True)
        for _ in range(4):
            request("POST", path + "/extend", "a", dict(valid, tokens=[-1]), expected=400)
            assert request("POST", path + "/extend", "a", dict(valid, n0=1), expected=409) == {"n": 0, "prefix_hash": prefix([])}
            request("POST", path + "/extend", "a", {}, expected=400)
            assert b"invalid JSON fields" in drain(start("a", path + "/extend", [1]), 400)
            assert b"invalid JSON fields" in drain(start("a", path + "/extend", b"{"), 400)
        assert request("GET", path + "/snapshot", "a") == before["a"]
        request("POST", path + "/extend", "a", valid)
        saved = request("GET", path + "/snapshot", "a")
        # Populated snapshots stay streaming with small receive windows, proving both permits are reusable together.
        first = start("a", paths["a"][0] + "/snapshot", method="GET", stalled=True)
        second = start("a", path + "/snapshot", method="GET", stalled=True)
        for sock in (first, second):
            header = b""
            while not header.endswith(b"\r\n\r\n"):
                chunk = sock.recv(1)
                assert chunk, "snapshot closed before headers"
                header += chunk
            assert int(header.split(b" ", 2)[1]) == 200, header
        overload("GET", paths["a"][0], "a")
        close(first)
        close(second)
        assert request("GET", path + "/snapshot", "a", retry_overload=True) == saved
        assert request("GET", path, "a")["n"] == 1
        print("PASS test_pending_error_release: repeated 400/409/JSON exceptions preserve snapshots and return both client permits; valid append succeeds")
    finally:
        for sock in held[:]:
            close(sock)


def test_post_inference_error_release(request, model_hash):
    # Repeated post-inference RAM rejection releases permits and leaves the rejected session byte-identical.
    append = {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": [101], "return_delta": False}
    retained = []
    for _ in range(32):
        sid = request("POST", "/v1/sessions", body={"model_sha256": model_hash})["session_id"]
        path = "/v1/sessions/" + sid
        result = request("POST", path + "/extend", body=append, expected=(202, 503))
        if "error" in result:
            assert result == {"error": "RAM session budget exceeded"}
            break
        assert result == {"n": 1, "prefix_hash": prefix([101])}
        retained.append(path)
    else:
        raise AssertionError("RAM error control did not reach the retained-payload limit")
    before = request("GET", path)
    snapshot = request("GET", path + "/snapshot")
    assert before["n"] == 0
    critical = dict(append, return_delta=True)
    for _ in range(4):
        assert request("POST", path + "/extend", body=critical, expected=503) == {"error": "RAM session budget exceeded"}
        assert request("GET", path) == before
    assert request("GET", path + "/snapshot") == snapshot
    request("DELETE", retained.pop(), expected=204)
    wire = request("POST", path + "/extend", body=critical)
    assert wire[:4] and request("GET", path)["n"] == 1
    print("PASS test_post_inference_error_release: repeated 503 preserves empty state and returns permits; freeing RAM admits the same critical request")


def test_max_delta_header_16384(request, model_hash):
    # The largest supported HTTP range carries all 16384 six-digit token IDs; over-capacity rejects without writes.
    sid = request("POST", "/v1/sessions", body={"model_sha256": model_hash})["session_id"]
    path = "/v1/sessions/" + sid
    before = request("GET", path)
    snapshot = request("GET", path + "/snapshot")
    tokens = [100000] * 16384
    append = {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": tokens, "return_delta": True}
    request("POST", path + "/extend", body=dict(append, tokens=tokens + [100000]), expected=400)
    assert request("GET", path) == before and request("GET", path + "/snapshot") == snapshot
    wire = request("POST", path + "/extend", body=append, timeout=900)
    length = struct.unpack_from("<I", wire)[0]
    assert 65536 < length <= 4096 + 11 * 16384
    header = json.loads(wire[4:4 + length])
    assert header["n0"] == 0 and header["n1"] == 16384 and header["tokens"] == tokens
    assert header["prefix_hash_n1"] == prefix(tokens)
    native = wire[4 + length:-48]
    assert header["sections"] == [{"name": "llama_seq_state", "offset": 0, "bytes": len(native)}]
    assert wire[-32:] == hashlib.sha256(native).digest()
    saved = request("GET", path + "/snapshot")
    assert struct.unpack_from("<III", saved)[1:] == (4, 16384)
    assert list(struct.unpack_from("<16384i", saved, 12)) == tokens
    assert saved[12 + 4 * 16384:] == native[8:]
    assert request("GET", path)["n"] == 16384
    print(f"PASS test_max_delta_header_16384: {length}-byte header, exact tokens/hash/native snapshot; 16385 rejected with unchanged empty snapshot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--regression", choices=("slow-readers", "max-delta", "pending", "errors"))
    args = parser.parse_args()
    artifacts = pathlib.Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    clients = {name: secrets.token_hex(24) for name in ("a", "b", "c")}
    credentials = artifacts / "clients.json"
    credentials.touch(mode=0o600, exist_ok=True)
    credentials.chmod(0o600)
    credentials.write_text(json.dumps(clients))
    log = (artifacts / "prefill-node-01.log").open("wb")
    process = subprocess.Popen([args.binary, args.model, str(credentials), str(args.port), "16384" if args.regression == "max-delta" else "8192" if args.regression == "pending" else "512"], stdout=log, stderr=log)
    (artifacts / "prefill.pid").write_text(str(process.pid) + "\n")
    base = f"http://127.0.0.1:{args.port}"

    def request(method, path, client="a", body=None, expected=200, timeout=180, retry_overload=False):
        headers = {"Content-Type": "application/json"}
        if client is not None:
            headers["Authorization"] = "Bearer " + clients[client]
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        deadline = time.monotonic() + 5
        while True:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status, payload, response_headers = response.status, response.read(), response.headers
            except urllib.error.HTTPError as error:
                status, payload, response_headers = error.code, error.read(), error.headers
            if status != 429 or not retry_overload or time.monotonic() >= deadline:
                break
            assert response_headers.get("Retry-After") == "1"
            time.sleep(0.05)
        assert status in (expected if isinstance(expected, tuple) else (expected,)), (method, path, status, expected, payload[:120])
        if status == 429:
            assert response_headers.get("Retry-After") == "1"
            assert json.loads(payload) == {"error": "binary response limit"}
        if "application/json" in response_headers.get("Content-Type", ""):
            return json.loads(payload)
        return payload

    try:
        for _ in range(600):
            if process.poll() is not None:
                raise AssertionError("prefill node exited during startup; see artifact log")
            try:
                health = request("GET", "/v1/health")
                break
            except urllib.error.URLError:
                time.sleep(0.1)
        else:
            raise AssertionError("prefill startup timed out")
        model_hash = health["model_sha256"]
        if args.regression == "slow-readers":
            test_slow_readers_do_not_starve_critical(request, clients, args.port, model_hash)
            return
        if args.regression == "pending":
            test_pending_admission(request, clients, args.port, model_hash)
            return
        if args.regression == "errors":
            test_post_inference_error_release(request, model_hash)
            return
        if args.regression == "max-delta":
            test_max_delta_header_16384(request, model_hash)
            return
        request("GET", "/v1/sessions", client=None, expected=401)
        # Each rejection preserves the session list or existing session count, hash and snapshot bytes.
        request("POST", "/v1/sessions", body={"model_sha256": "0" * 64}, expected=400)
        assert request("GET", "/v1/sessions") == []
        sid = request("POST", "/v1/sessions", body={"model_sha256": model_hash})["session_id"]
        path = "/v1/sessions/" + sid
        assert request("GET", path)["n"] == 0
        assert request("GET", "/v1/sessions", "b") == []
        append = {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": [100, 101], "return_delta": True}
        wire = request("POST", path + "/extend", body=append)
        header_length = struct.unpack_from("<I", wire)[0]
        header = json.loads(wire[4:4 + header_length])
        native = wire[4 + header_length:-48]
        assert header["n0"] == 0 and header["n1"] == 2
        assert header["prefix_hash_n1"] == prefix([100, 101])
        assert wire[-32:] == hashlib.sha256(native).digest()
        before = request("GET", path)
        state_before = request("GET", path + "/snapshot")
        assert struct.unpack_from("<III", state_before)[1:] == (4, 2)
        assert struct.unpack_from("<ii", state_before, 12) == (100, 101)
        assert state_before[20:] == native[8:]
        for n0 in (0, 1, 3, -1):
            invalid = dict(append, n0=n0)
            response = request("POST", path + "/extend", body=invalid, expected=409)
            assert response == {"n": 2, "prefix_hash": prefix([100, 101])}
        request("POST", path + "/extend", body=dict(append, n0=2, prefix_hash_n0="0" * 64), expected=409)
        for tokens in ([], [-1], [999999999], [1.5], [True], [1] * 513):
            request("POST", path + "/extend", body=dict(append, n0=2, prefix_hash_n0=prefix([100, 101]), tokens=tokens), expected=400)
        for method, suffix, body in (("GET", "", None), ("GET", "/snapshot", None), ("POST", "/extend", append), ("DELETE", "", None)):
            request(method, path + suffix, "b", body, expected=404)
        assert request("GET", path) == before
        assert request("GET", path + "/snapshot") == state_before
        request("POST", path + "/extend", body={"n0": 2, "prefix_hash_n0": prefix([100, 101]), "tokens": [102], "return_delta": False}, expected=202)
        assert request("GET", path)["n"] == 3
        print("PASS wrong model, gap/overlap, prefix mismatch, token bounds, foreign read/extend/snapshot/delete; rejection leaves count/hash/snapshot unchanged")

        # Three clients each retain four independent sessions through interleaved offload and shadow appends.
        cases = [(client, index, request("POST", "/v1/sessions", client, {"model_sha256": model_hash})["session_id"])
                 for index in range(4) for client in clients]
        assert all(len(request("GET", "/v1/sessions", client)) >= 4 for client in clients)

        def exercise(case):
            client, index, other = case
            endpoint = "/v1/sessions/" + other
            tokens = [110 + index, 120 + index]
            request("POST", endpoint + "/extend", client, {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": tokens, "return_delta": True}, retry_overload=True)
            request("POST", endpoint + "/extend", client, {"n0": 2, "prefix_hash_n0": prefix(tokens), "tokens": [130 + index], "return_delta": False}, 202, retry_overload=True)
            final = request("GET", endpoint, client, retry_overload=True)
            assert final["n"] == 3 and final["prefix_hash"] == prefix(tokens + [130 + index])
            return other
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(exercise, cases))
        assert len(set(ids)) == 12
        for client, _, other in cases:
            request("DELETE", "/v1/sessions/" + other, client, expected=204)
        request("DELETE", path, expected=204)
        request("GET", path, expected=404)
        assert all(request("GET", "/v1/sessions", client) == [] for client in clients)
        assert b"hybrid_ram_bytes 0\n" in request("GET", "/metrics")
        print("PASS 3 clients x 4 sessions, interleaved offload/shadow, distinct IDs, expected prefix hashes, deletion")
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
