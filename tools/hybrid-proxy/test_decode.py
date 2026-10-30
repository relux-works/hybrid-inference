"""Explicit opt-in native slot/CPU equivalence check; requires a built patched server."""
import argparse
import http.server
import os
import threading
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import time

from proxy import Handler, HTTPError, Node, Proxy
from store import prefix_hash
from stub_server import xxh128


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def unpack(wire):
    size, = struct.unpack("<I", wire[:4])
    return json.loads(wire[4:4 + size]), wire[4 + size:-48]


def envelope(header, state):
    header = dict(header, sections=[dict(name="llama_seq_state", offset=0, bytes=len(state))])
    text = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(text)) + text + state + xxh128(state) + hashlib.sha256(state).digest()


def run(node, golden, slots):
    metadata = json.loads(Path(str(golden) + ".tokens.json").read_text())
    tokens, n0, n1 = metadata["tokens"], metadata["n0"], metadata["n1"]
    wire = Path(str(golden) + ".delta").read_bytes()
    header, state = unpack(wire)
    session = header["session_id"]
    headers = {"Content-Type": "application/octet-stream", "X-Hybrid-Session": session}
    prefix = Path(str(golden) + ".prefix.state").read_bytes()
    first = dict(header, n0=0, n1=n0, tokens=tokens[:n0], prefix_hash_n1=prefix_hash(tokens[:n0]))
    node.request("POST", "/slots/0?action=import_delta", envelope(first, prefix), headers)
    # A's one-stream slot snapshot must restore into another slot of a two-stream decode node.
    slot_file = struct.pack("<III", 0x67677371, 4, n0)
    slot_file += b"".join(struct.pack("<i", token) for token in tokens[:n0]) + prefix[8:]
    (slots / "source-prefix.bin").write_bytes(slot_file)
    node.request("POST", "/slots/1?action=restore", {"filename": "source-prefix.bin"})
    restored = node.request("GET", "/slots/1/info")
    require(restored["tokens"] == tokens[:n0], "one-stream snapshot restore into parallel slot failed")
    print("PASS HTTP source snapshot restore into parallel slot", flush=True)
    before = node.request("GET", "/slots/0/info")
    require(before == dict(n=n0, prefix_hash=prefix_hash(tokens[:n0]), tokens=tokens[:n0]), "prefix import info mismatch")
    node.request("POST", "/slots/0?action=save", {"filename": "before.bin"})
    before_bytes = (slots / "before.bin").read_bytes()
    mutations = {
        "wrong-model": dict(header, model_sha256="f" * 64),
        "gap": dict(header, n0=n0 + 1), "overlap": dict(header, n0=n0 - 1),
        "prefix-hash": dict(header, prefix_hash_n1="f" * 64),
        "declared-layout": dict(header, kv_type="f16"),
        "session": dict(header, session_id="foreign-session"),
    }
    candidates = {name: envelope(value, state) for name, value in mutations.items()}
    candidates["truncated"] = wire[:-1]
    candidates["section-checksum"] = wire[:-48] + bytes([wire[-48] ^ 1]) + wire[-47:]
    candidates["body-checksum"] = wire[:-1] + bytes([wire[-1] ^ 1])
    # Recompute valid envelope checksums so this malformed native state reaches rollback.
    candidates["native-truncated"] = envelope(header, state[:-1])
    raw = bytearray(state)
    struct.pack_into("<i", raw, 16, n0 + 1)
    candidates["native-position"] = envelope(header, raw)
    count, = struct.unpack_from("<I", state, 12)
    # Qwen35's native KV metadata includes 12-byte position extensions (x, y, token).
    offset = 16 + count * 24
    raw = bytearray(state)
    layers, = struct.unpack_from("<I", raw, offset + 4)
    struct.pack_into("<I", raw, offset + 4, layers + 1)
    candidates["native-layer-layout"] = envelope(header, raw)
    # Each failed HTTP import preserves both the visible info and the saved native sequence bytes.
    for name, candidate in candidates.items():
        try:
            node.request("POST", "/slots/0?action=import_delta", candidate, headers)
        except HTTPError as error:
            require(error.status == 400, f"{name}: expected HTTP 400, got {error.status}")
            require(b"message" in error.body, f"{name}: missing rejection message")
        else:
            raise AssertionError(f"{name}: invalid delta accepted")
        require(node.request("GET", "/slots/0/info") == before, f"{name}: slot info changed")
        require(node.request("GET", "/slots/1/info") == restored, f"{name}: another slot changed")
        node.request("POST", "/slots/0?action=save", {"filename": "after.bin"})
        require((slots / "after.bin").read_bytes() == before_bytes, f"{name}: native slot bytes changed")
        print(f"PASS HTTP {name}: 400, slot unchanged", flush=True)
    node.request("POST", "/slots/0?action=import_delta", wire, headers)
    info = node.request("GET", "/slots/0/info")
    require(info["n"] == n1 and info["tokens"] == tokens[:n1], "valid delta info mismatch")
    require(info["prefix_hash"] == metadata["prefix_hash_n1"], "valid delta hash mismatch")
    # Slot 0 imports a real prefix+delta; slot 1 prefills locally on the same CPU backend.
    node.request("POST", "/slots/1?action=erase", {})
    request = dict(prompt=tokens, n_predict=64, temperature=0, seed=0, cache_prompt=True,
                   return_tokens=True, ignore_eos=True)
    imported = node.request("POST", "/completion", dict(request, id_slot=0))
    local = node.request("POST", "/completion", dict(request, id_slot=1))
    require(len(imported["tokens"]) == 64, "server did not return 64 greedy tokens")
    require(imported["tokens"] == local["tokens"], "HTTP import/plain-local greedy mismatch")
    require(imported.get("tokens_cached", 0) >= n1, "server discarded imported prefix and prefills locally")
    print(f"PASS HTTP incremental import + held token: 64/64 greedy tokens at depth {len(tokens)}", flush=True)


def run_proxy(decode, binary, model, golden, directory, slots):
    metadata = json.loads(Path(str(golden) + ".tokens.json").read_text())
    credentials = directory / "prefill-fixture.json"
    credentials.write_text(json.dumps({"a": "fixture-prefill-a-0001", "b": "fixture-prefill-b-0001"}))
    os.chmod(credentials, 0o600)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = open(directory / "prefill.log", "wb")
    command = [binary, model, str(credentials), str(port), "4096"]
    process = None
    app = None
    server = None
    thread = None

    def start_prefill():
        nonlocal process
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        with open(directory / "prefill-pids.txt", "a") as record:
            record.write(str(process.pid) + "\n")
        deadline = time.monotonic() + 120
        node = Node(f"http://127.0.0.1:{port}", timeout=120)
        while time.monotonic() < deadline:
            require(process.poll() is None, "prefill exited during startup")
            try:
                node.request("GET", "/v1/health")
                return
            except (OSError, HTTPError):
                time.sleep(0.1)
        raise AssertionError("prefill startup timed out")

    def stop_prefill():
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    try:
        start_prefill()
        app = Proxy(dict(model_sha256=metadata["model_sha256"], prefill_url=f"http://127.0.0.1:{port}",
                         decode_url=decode.url, prefill_timeout=120, slots=2, slot_path=str(slots),
                         clients={"fixture-a": dict(id="a", prefill_token="fixture-prefill-a-0001"),
                                  "fixture-b": dict(id="b", prefill_token="fixture-prefill-b-0001")},
                         policy=dict(always=True)), str(directory / "real-proxy.sqlite"))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.app = app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = Node(f"http://127.0.0.1:{server.server_port}", "fixture-a", timeout=120)
        system = "Follow the user request. Keep conversation state consistent. " * 80
        body = dict(model=Path(model).name, messages=[dict(role="system", content=system),
                    dict(role="user", content="Count the integers from one to ten.")],
                    max_tokens=64, temperature=0, seed=0, ignore_eos=True,
                    chat_template_kwargs=dict(enable_thinking=False))
        with client.open("POST", "/v1/chat/completions", body) as response:
            sid = response.headers["X-Hybrid-Session"]
            imported = json.loads(response.read())
        require(app.store.get(sid)["offload"], "real proxy did not import an offloaded delta")
        local = decode.request("POST", "/v1/chat/completions", dict(body, id_slot=1, cache_prompt=False))
        require(imported["choices"][0]["message"] == local["choices"][0]["message"], "real proxy/plain-local chat mismatch")
        require(imported["usage"]["completion_tokens"] == local["usage"]["completion_tokens"] == 64,
                "real proxy chat did not exercise 64 generated tokens")
        fork = dict(body, messages=[body["messages"][0], dict(role="user", content="Name the primary colors.")],
                    max_tokens=4, stream=True)
        with client.open("POST", "/v1/chat/completions", fork) as response:
            fork_id = response.headers["X-Hybrid-Session"]
            chunks = response.read()
        require(fork_id != sid and b"data: [DONE]" in chunks, "real proxy shared-prefix fork/stream failed")
        foreign = Node(client.url, "fixture-b", timeout=120)
        try:
            foreign.request("POST", "/v1/chat/completions", body, {"X-Hybrid-Session": sid})
        except HTTPError as error:
            require(error.status == 404, "real proxy foreign client must get 404")
        else:
            raise AssertionError("real proxy allowed a foreign session")
        stop_prefill()
        offline = dict(body, messages=[dict(role="user", content="Reply with a short greeting.")], max_tokens=4)
        with client.open("POST", "/v1/chat/completions", offline) as response:
            offline_id = response.headers["X-Hybrid-Session"]
            require(bool(json.loads(response.read())["choices"]), "offline real decode failed")
        require(not app.store.get(offline_id)["offload"], "offline request incorrectly marked offloaded")
        start_prefill()
        with app.lock:
            app.replay()
            app.replay()
            for session in app.store.pending():
                owner = next(c for c in app.clients.values() if c["id"] == session["client"])
                info = app.remote(owner).request("GET", "/v1/sessions/" + session["remote_id"])
                require(info["n"] == len(session["tokens"]) and info["prefix_hash"] == prefix_hash(session["tokens"]),
                        "real reconnect journal lost or duplicated tokens")
        print("PASS real proxy: imported chat/local equality for 64 tokens, streaming fork, foreign 404, offline and restart replay", flush=True)
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join()
        if app:
            app.store.close()
        stop_prefill()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-binary", help="Also run the real proxy against A's prefill node")
    parser.add_argument("--golden", required=True, help="Golden filename stem, e.g. .../prompt-512")
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    directory = Path(args.artifacts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    slots = directory / "slots"
    slots.mkdir(exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with open(directory / "server.log", "wb") as output:
        ready = subprocess.run([args.binary, "--version"], stdout=output, stderr=subprocess.STDOUT)
        require(ready.returncode == 0, "server readiness failed")
        command = [args.binary, "-m", args.model, "--host", "127.0.0.1", "--port", str(port),
                   "-ngl", "0", "-t", "2", "-tb", "2", "-c", "8192", "-b", "256", "-ub", "256",
                   "-np", "2", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "--slots",
                   "--no-context-shift", "--cache-ram", "0", "--no-cache-idle-slots",
                   "--ctx-checkpoints", "0", "--slot-save-path", str(slots) + "/"]
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        (directory / "pid.txt").write_text(str(process.pid) + "\n")
        try:
            node = Node(f"http://127.0.0.1:{port}", timeout=120)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                require(process.poll() is None, "server exited during startup")
                try:
                    node.request("GET", "/health")
                    break
                except (HTTPError, OSError):
                    time.sleep(0.1)
            else:
                raise AssertionError("server startup timed out")
            run(node, Path(args.golden), slots)
            if args.prefill_binary:
                run_proxy(node, args.prefill_binary, args.model, Path(args.golden), directory, slots)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
