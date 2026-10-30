"""Exercise HTTP append invariants and namespace isolation against the small CPU model."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import secrets
import struct
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--port", type=int, default=18091)
    args = parser.parse_args()
    artifacts = pathlib.Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    clients = {name: secrets.token_hex(24) for name in ("a", "b", "c")}
    credentials = artifacts / "clients.json"
    credentials.touch(mode=0o600, exist_ok=True)
    credentials.chmod(0o600)
    credentials.write_text(json.dumps(clients))
    log = (artifacts / "prefill-node-01.log").open("wb")
    process = subprocess.Popen([args.binary, args.model, str(credentials), str(args.port), "512"], stdout=log, stderr=log)
    (artifacts / "prefill.pid").write_text(str(process.pid) + "\n")
    base = f"http://127.0.0.1:{args.port}"

    def request(method, path, client="a", body=None, expected=200):
        headers = {"Content-Type": "application/json"}
        if client is not None:
            headers["Authorization"] = "Bearer " + clients[client]
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                status, payload, response_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            status, payload, response_headers = error.code, error.read(), error.headers
        assert status == expected, (method, path, status, expected, payload[:120])
        if "application/json" in response_headers.get("Content-Type", ""):
            return json.loads(payload)
        return payload

    def prefix(tokens):
        return hashlib.sha256(b"".join(struct.pack("<i", token) for token in tokens)).hexdigest()

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
            request("POST", endpoint + "/extend", client, {"n0": 0, "prefix_hash_n0": prefix([]), "tokens": tokens, "return_delta": True})
            request("POST", endpoint + "/extend", client, {"n0": 2, "prefix_hash_n0": prefix(tokens), "tokens": [130 + index], "return_delta": False}, 202)
            final = request("GET", endpoint, client)
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
