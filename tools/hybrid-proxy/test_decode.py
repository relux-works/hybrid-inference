"""Explicit opt-in native slot/CPU equivalence check; requires a built patched server."""
import argparse
import http.server
import io
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
    state = bytes(state)
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
    node.request("POST", "/slots/1?action=save", {"filename": "neighbor-before.bin"})
    neighbor_bytes = (slots / "neighbor-before.bin").read_bytes()
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
    offset += 16
    for _ in range(2 * layers):
        row, = struct.unpack_from("<Q", state, offset + 4)
        offset += 12 + row * count
    require(struct.unpack_from("<I", state, offset)[0] == 1, "expected one recurrent cell")
    raw = state[:offset] + struct.pack("<I", 2) + state[offset + 4:offset + 12] * 2 + state[offset + 12:]
    candidates["native-recurrent-two-cells"] = envelope(header, raw)
    raw = bytearray(state)
    struct.pack_into("<i", raw, offset + 4, n1)
    candidates["native-recurrent-endpoint"] = envelope(header, raw)
    # Slot files include KV and recurrent data. Metadata equality alone cannot prove neighbor preservation.
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
        node.request("POST", "/slots/1?action=save", {"filename": "neighbor-after.bin"})
        require((slots / "neighbor-after.bin").read_bytes() == neighbor_bytes, f"{name}: neighbor native bytes changed")
        print(f"PASS HTTP {name}: 400, both slots byte-identical", flush=True)
    node.request("POST", "/slots/0?action=import_delta", wire, headers)
    info = node.request("GET", "/slots/0/info")
    require(info["n"] == n1 and info["tokens"] == tokens[:n1], "valid delta info mismatch")
    require(info["prefix_hash"] == metadata["prefix_hash_n1"], "valid delta hash mismatch")
    node.request("POST", "/slots/1?action=save", {"filename": "neighbor-after-valid.bin"})
    require((slots / "neighbor-after-valid.bin").read_bytes() == neighbor_bytes, "valid append changed neighbor native bytes")
    # Slot 0 imports a real prefix+delta; slot 1 prefills locally on the same CPU backend.
    node.request("POST", "/slots/1?action=erase", {})
    request = dict(prompt=tokens, n_predict=64, temperature=0, seed=0, cache_prompt=True,
                   return_tokens=True, ignore_eos=True)
    imported = node.request("POST", "/completion", dict(request, id_slot=0))
    local = node.request("POST", "/completion", dict(request, id_slot=1))
    require(len(imported["tokens"]) == 64, "server did not return 64 greedy tokens")
    require(imported["tokens"] == local["tokens"], "HTTP import/plain-local greedy mismatch")
    require(imported["timings"]["cache_n"] == n1 and imported["timings"]["prompt_n"] == 1,
            "server discarded imported prefix and prefills locally")
    print(f"PASS HTTP incremental import + held token: 64/64 greedy tokens at depth {len(tokens)}", flush=True)


def run_proxy(decode, binary, model, golden, directory, slots, checker):
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
        identity = dict(binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                        library_sha256=hashlib.sha256((Path(binary).parent / "libllama.so").read_bytes()).hexdigest())
        with open(directory / "prefill-identities.jsonl", "a") as record:
            record.write(json.dumps(identity) + "\n")
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
        prefill = Node(f"http://127.0.0.1:{port}", "fixture-prefill-a-0001", timeout=120)
        boundary_id = prefill.request("POST", "/v1/sessions", {"model_sha256": metadata["model_sha256"]})["session_id"]
        prefix = []
        decode.request("POST", "/slots/2?action=erase", {})
        for slot in (0, 1):
            decode.request("POST", f"/slots/{slot}?action=save", {"filename": f"boundary-before-{slot}.bin"})
        for token in metadata["tokens"][:2]:
            delta = prefill.request("POST", f"/v1/sessions/{boundary_id}/extend",
                dict(n0=len(prefix), prefix_hash_n0=prefix_hash(prefix), tokens=[token], return_delta=True), binary=True)
            decode.request("POST", "/slots/2?action=import_delta", delta,
                {"Content-Type": "application/octet-stream", "X-Hybrid-Session": boundary_id})
            prefix.append(token)
            info = decode.request("GET", "/slots/2/info")
            require(info == dict(n=len(prefix), tokens=prefix, prefix_hash=prefix_hash(prefix)), "one-token append endpoint mismatch")
            for slot in (0, 1):
                decode.request("POST", f"/slots/{slot}?action=save", {"filename": f"boundary-after-{slot}.bin"})
                require((slots / f"boundary-before-{slot}.bin").read_bytes() == (slots / f"boundary-after-{slot}.bin").read_bytes(),
                        "one-token append changed neighbor bytes")
        prefill.request("DELETE", f"/v1/sessions/{boundary_id}")
        print("PASS real prefill one-token boundaries: 0->1 and 1->2, exact endpoints and neighbor native bytes", flush=True)
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
        foreign = Node(client.url, "fixture-b", timeout=120)
        captured, imported_deltas = [], []
        decode_open = app.decode.open
        def capture_native_tokens(method, path, data=None, headers=None):
            if method == "POST" and "action=import_delta" in path:
                imported_deltas.append(bytes(data))
            response = decode_open(method, path, data, headers)
            if path == "/v1/chat/completions" and not data.get("stream"):
                with response:
                    raw = response.read()
                captured.append(json.loads(raw)["__verbose"])
                return io.BytesIO(raw)
            return response
        app.decode.open = capture_native_tokens
        system = "Follow the user request. Keep conversation state consistent. " * 40
        records = []
        # Two clients each fork the same long system prefix. A separate decoder slot prefills each local control.
        for owner, connection in (("a", client), ("b", foreign)):
            owner_sessions = []
            for index, question in enumerate(("Count the integers from one to ten.", "Name the primary colors.")):
                body = dict(model=Path(model).name, messages=[dict(role="system", content=system),
                            dict(role="user", content=question)],
                            max_tokens=64, temperature=0, seed=0, ignore_eos=True, return_tokens=True, verbose=True,
                            chat_template_kwargs=dict(enable_thinking=False))
                with connection.open("POST", "/v1/chat/completions", body) as response:
                    sid = response.headers["X-Hybrid-Session"]
                    imported = json.loads(response.read())
                require(app.store.get(sid)["offload"], "real proxy did not import an offloaded delta")
                native = captured[-1]
                decode.request("POST", "/slots/2?action=erase", {})
                local = decode.request("POST", "/v1/chat/completions", dict(body, id_slot=2, cache_prompt=False))
                (directory / f"comparison-{owner}-{index}.json").write_text(json.dumps(
                    dict(imported=native, local=local, session=app.store.get(sid)), indent=2) + "\n")
                actual_tokens, expected_tokens = native["tokens"], local["__verbose"]["tokens"]
                if actual_tokens != expected_tokens:
                    # Diagnose batch partition independently, without importing any state into the control slot.
                    prompt_ids = app.store.get(sid)["tokens"][:native["tokens_evaluated"]]
                    decode.request("POST", "/slots/2?action=erase", {})
                    decode.request("POST", "/completion", dict(prompt=prompt_ids[:-1], id_slot=2,
                                   n_predict=0, temperature=0, cache_prompt=False))
                    split_info = decode.request("GET", "/slots/2/info")
                    split = decode.request("POST", "/v1/chat/completions", dict(body, id_slot=2, cache_prompt=True))
                    diagnostic = dict(imported=actual_tokens, plain=expected_tokens,
                                      split=split["__verbose"]["tokens"], prefix=split_info,
                                      split_timings=split["__verbose"]["timings"])
                    (directory / f"partition-{owner}-{index}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
                require(len(actual_tokens) == len(expected_tokens) == 64, "chat did not return 64 token IDs")
                identical = actual_tokens == expected_tokens
                require(native["timings"]["cache_n"] > 256 and native["timings"]["prompt_n"] == 1,
                        "chat discarded imported prefix")
                require(local["__verbose"]["timings"]["cache_n"] == 0,
                        "local control reused a cached prefix")
                require(imported["usage"]["completion_tokens"] == local["usage"]["completion_tokens"] == 64,
                        "real proxy chat did not exercise 64 generated tokens")
                owner_sessions.append(sid)
                records.append(dict(client=owner, conversation=index, session_id=sid, greedy_64=actual_tokens,
                                    expected_64=expected_tokens, identical=identical,
                                    cached=native["timings"]["cache_n"]))
                prompt_ids = app.store.get(sid)["tokens"][:native["tokens_evaluated"]]
                fixture = dict(model_sha256=metadata["model_sha256"], tokens=prompt_ids,
                               greedy_64=actual_tokens, ignore_eos=True)
                (directory / f"conversation-{owner}-{index}.tokens.json").write_text(json.dumps(fixture) + "\n")
                require(len(imported_deltas) == len(records), "missing actual HTTP import record")
                (directory / f"conversation-{owner}-{index}.delta").write_bytes(imported_deltas[-1])
                match = sum(x == y for x, y in zip(actual_tokens, expected_tokens))
                first_difference = next((i for i, (x, y) in enumerate(zip(actual_tokens, expected_tokens)) if x != y), None)
                print(f"INFO whole-prompt free-running client={owner} session={index}: {match}/64 token IDs, first_divergence={first_difference}", flush=True)
            require(len(set(owner_sessions)) == 2, "shared-system conversations must fork")
        require(len({r["session_id"] for r in records}) == 4, "client/session mapping leaked")
        (directory / "greedy-equality.json").write_text(json.dumps(records, indent=2) + "\n")
        sid = records[0]["session_id"]
        try:
            foreign.request("POST", "/v1/chat/completions", body, {"X-Hybrid-Session": sid})
        except HTTPError as error:
            require(error.status == 404, "real proxy foreign client must get 404")
        else:
            raise AssertionError("real proxy allowed a foreign session")
        # Streaming uses another fork after the four bounded 64-token equality controls.
        fork = dict(body, messages=[body["messages"][0], dict(role="user", content="Say hello.")],
                    max_tokens=4, stream=True)
        with client.open("POST", "/v1/chat/completions", fork) as response:
            fork_id = response.headers["X-Hybrid-Session"]
            chunks = response.read()
        require(fork_id not in {r["session_id"] for r in records} and b"data: [DONE]" in chunks, "real proxy fork/stream failed")
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
        print("PASS real proxy: 2 clients x 2 sessions mapped, streaming fork, foreign 404, offline and restart replay", flush=True)
        # Spec v0.3.4 keeps whole-prompt free-running divergence informational; exactness and KL use independent local contexts.
        fixtures = sorted(str(path) for path in directory.glob("conversation-*.tokens.json"))
        require(len(fixtures) == 4, "missing teacher-forced conversation fixture")
        with open(directory / "distribution-check.log", "wb") as output:
            result = subprocess.run([checker, model, *fixtures], stdout=output, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f"same-partition/distribution checker exited {result.returncode}; see distribution-check.log")
        print("PASS real proxy: all four sessions same-partition greedy=64/64 and teacher-forced KL/top1 bounds", flush=True)
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
    parser.add_argument("--distribution-checker", help="Defaults to llama-hybrid-check-distributions beside --binary")
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
                   "-np", "3", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "--slots",
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
                checker = args.distribution_checker or str(Path(args.binary).resolve().with_name("llama-hybrid-check-distributions"))
                require(Path(checker).is_file(), "build llama-hybrid-check-distributions for the real proxy checks")
                run_proxy(node, args.prefill_binary, args.model, Path(args.golden), directory, slots, checker)
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
