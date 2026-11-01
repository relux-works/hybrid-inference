#!/usr/bin/env python3
"""Paired local-kept latency trace with existing native binaries. No builds or downloads."""
import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import http.server
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--prefill-binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--proxy-source", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--background-replay", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or args.gpu_layers < 0:
        parser.error("pairs must be positive and gpu-layers must be nonnegative")
    sys.path.insert(0, str(Path(args.proxy_source).resolve()))
    from journal import Renderer
    from proxy import Handler, HTTPError, Node, Proxy
    from store import Store, prefix_hash

    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(root).free > 28_000_000_000, "disk floor"
    model = Path(args.model).resolve()
    digest = sha256(model)
    identity = dict(command=sys.argv, harness_sha256=sha256(__file__), model_sha256=digest, sources={}, binaries={})
    for cls in (Proxy, Store, Renderer):
        source = Path(sys.modules[cls.__module__].__file__)
        identity["sources"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for binary in (args.binary, args.prefill_binary):
        directory = Path(binary).resolve().parent
        for file in [Path(binary).resolve(), *directory.glob("*.so*")]:
            identity["binaries"][str(file)] = sha256(file)
    (root / "identity.json").write_text(json.dumps(identity, indent=2))
    local = threading.local()
    trace_done = threading.Event()
    records = []
    decode_url = None

    def timed(cls, name, label):
        original = getattr(cls, name)

        @wraps(original)
        def call(self, *a, **kw):
            start = time.perf_counter()
            writes = getattr(local, "writes", 0)
            try:
                return original(self, *a, **kw)
            finally:
                if getattr(local, "events", None) is not None:
                    event = dict(stage=label, start_ms=(start - local.started) * 1000,
                                 ms=(time.perf_counter() - start) * 1000)
                    if cls is Store:
                        event["writes"] = local.writes - writes
                    if cls is Node:
                        role = "decoder" if self.url == decode_url else "node"
                        event.update(stage=role + ("_headers" if name == "open" else "_rpc"),
                                     method=a[0], path=a[1].rsplit("/", 1)[-1])
                    local.events.append(event)
        setattr(cls, name, call)

    store_init = Store.__init__

    def traced_store(self, *a, **kw):
        store_init(self, *a, **kw)

        def sql(statement):
            if getattr(local, "events", None) is not None and statement.split()[0] in ("UPDATE", "INSERT", "DELETE"):
                local.writes += 1
        self.db.set_trace_callback(sql)
    Store.__init__ = traced_store
    for name in ("save", "set_slot", "benchmark_mode", "select", "select_messages"):
        timed(Store, name, "db_" + name)
    for name in ("render", "encode", "template_identity", "user_boundaries", "layout", "append"):
        timed(Renderer, name, "template_" + name)
    timed(Node, "request", "rpc")
    timed(Node, "open", "headers")
    timed(Proxy, "prepare_session", "prepare_session")
    timed(Proxy, "finish", "journal_finish")
    prepare = Proxy.prepare

    @contextmanager
    def measured_prepare(self, *a, **kw):
        start = time.perf_counter()
        with prepare(self, *a, **kw) as result:
            local.events.append(dict(stage="proxy_prepare", start_ms=(start - local.started) * 1000,
                                     ms=(time.perf_counter() - start) * 1000))
            yield result
    Proxy.prepare = measured_prepare
    finish = Proxy.finish

    def measured_finish(self, session, prompt):
        finish(self, session, prompt)
        local.sampled = session["_capture"].native["tokens"]
        local.native_timings = session["_capture"].native["timings"]
        local.durable = list(session["tokens"])
    Proxy.finish = measured_finish

    class TraceHandler(Handler):
        def do_POST(self):
            local.started = time.perf_counter()
            local.events = []
            local.writes = 0
            local.sampled, local.durable = None, None
            local.native_timings = None
            try:
                super().do_POST()
            finally:
                records.append(dict(events=local.events, sampled_ids=local.sampled,
                                    durable_tokens=local.durable,
                                    native_timings=local.native_timings, sql_writes=local.writes,
                                    handler_ms=(time.perf_counter() - local.started) * 1000))
                local.events = None
                trace_done.set()

    @contextmanager
    def process(command, name, url, health):
        with (root / (name + ".log")).open("wb") as output:
            child = subprocess.Popen(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT)
            (root / (name + ".pid")).write_text(str(child.pid))
            try:
                deadline = time.monotonic() + 180
                node = Node(url, timeout=300)
                while True:
                    assert child.poll() is None, name + " exited during startup"
                    try:
                        node.request("GET", health)
                        break
                    except (OSError, HTTPError):
                        assert time.monotonic() < deadline, name + " readiness timed out"
                        time.sleep(0.1)
                yield node
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=15)
                (root / (name + "-cleanup.json")).write_text(json.dumps(dict(pid=child.pid, exit_code=child.returncode)))

    prefill_port, decode_port = port(), port()
    prefill_url, decode_url = f"http://127.0.0.1:{prefill_port}", f"http://127.0.0.1:{decode_port}"
    slots = root / "slots"
    slots.mkdir()
    # This is a public, synthetic loopback fixture credential, never an operator credential.
    (root / "clients.json").write_text(json.dumps(dict(bench="local-trace-fixture")))
    node_command = [args.prefill_binary, model, root / "clients.json", prefill_port, 2048,
                    "--host", "127.0.0.1", "--gpu-layers", args.gpu_layers, "--bench"]
    decode_command = [args.binary, "-m", model, "--host", "127.0.0.1", "--port", decode_port,
                      "-ngl", args.gpu_layers, "-t", 2, "-tb", 2, "-c", 2048, "-np", 1,
                      "-b", 256, "-ub", 256, "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
                      "--jinja", "--slots", "--no-context-shift", "--cache-ram", 0,
                      "--no-cache-idle-slots", "--ctx-checkpoints", 0, "--checkpoint-min-step", 0,
                      "--slot-save-path", str(slots) + "/"]
    with process(node_command, "prefill", prefill_url, "/v1/health") as node, process(
            decode_command, "decode", decode_url, "/health") as decode:
        config = dict(benchmark=True, model_sha256=digest, prefill_url=prefill_url, decode_url=decode_url,
                      slot_path=str(slots), slots=1, prefill_timeout=300, decode_timeout=300,
                      clients={"trace-fixture": dict(id="bench", prefill_token="local-trace-fixture")},
                      delta_geometry=dict(kv_bytes=6528, state_bytes=20201472))
        app = Proxy(config, str(root / "proxy.sqlite"))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TraceHandler)
        server.app = app
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        replay_worker = threading.Thread(target=app.replay_loop, daemon=True)
        if args.background_replay:
            replay_worker.start()
        connection = Node(f"http://127.0.0.1:{server.server_port}", "trace-fixture", timeout=300)
        pairs = []

        def chat(messages, mode, sid):
            body = dict(messages=messages, temperature=0, seed=42, max_tokens=8, stream=True,
                        chat_template_kwargs=dict(enable_thinking=False))
            headers = {"X-Hybrid-Bench-Mode": mode, "X-Hybrid-Session": sid}
            trace_done.clear()
            start, first, trace, done = time.perf_counter(), None, None, 0
            message = dict(role="assistant", content="")
            with connection.open("POST", "/v1/chat/completions", body, headers) as response:
                for line in response:
                    if line.strip() == b"data: [DONE]":
                        done += 1
                        continue
                    if not line.startswith(b"data: "):
                        continue
                    chunk = json.loads(line[6:])
                    if "hybrid_bench" in chunk:
                        trace = chunk["hybrid_bench"]
                    for choice in chunk.get("choices", []):
                        for key in ("content", "reasoning_content"):
                            value = choice.get("delta", {}).get(key)
                            if value:
                                if first is None:
                                    first = (time.perf_counter() - start) * 1000
                                message[key] = message.get(key, "") + value
            elapsed = (time.perf_counter() - start) * 1000
            with app.reserve(sid):
                pass
            assert trace_done.wait(10), "handler did not retire"
            assert done == 1 and records[-1]["sampled_ids"] is not None, "incomplete stream"
            assert trace is not None and trace["path"] == "local", "call was not policy-kept local"
            assert trace["remote_applied"] == 0 and trace["prefilled"] == trace["new_tokens"]
            return message, dict(ttft_ms=first, total_ms=elapsed, bench=trace, **records[-1])

        try:
            for pair in range(args.pairs):
                modes = ("local", "hybrid") if pair % 2 == 0 else ("hybrid", "local")
                runs = {}
                for mode in modes:
                    assert shutil.disk_usage(root).free > 28_000_000_000, "disk floor"
                    sid = f"trace-{pair}-{mode}"
                    history = [dict(role="system", content="Answer briefly."),
                               dict(role="user", content="blue " * 80 + "Count from one to five.")]
                    answer, seed = chat(history, mode, sid)
                    history += [answer, dict(role="user", content="blue " * 24 + "Now count from six to ten.")]
                    _, call = chat(history, mode, sid)
                    runs[mode] = dict(seed=seed, call=call)
                    # Replay runs only after the response, so this measures an immediate follow-up with a pending suffix.
                    app.replay()
                    app.replay()
                    current = app.store.get(sid)
                    if mode == "hybrid":
                        info = app.remote(config["clients"]["trace-fixture"]).request("GET", "/v1/sessions/" + current["remote_id"])
                        assert info["n"] == len(current["tokens"]) == current["remote_n"]
                        assert info["prefix_hash"] == prefix_hash(current["tokens"])
                        call["node_consistent_after_replay"] = True
                    connection.request("DELETE", "/v1/sessions/" + sid)
                    # A fresh cold slot for both members avoids cache and parking asymmetry.
                    decode.request("POST", "/slots/0?action=erase", {})
                    app.store.set_slot(sid, None)
                assert runs["local"]["call"]["sampled_ids"] == runs["hybrid"]["call"]["sampled_ids"], "sampled token mismatch"
                assert runs["local"]["call"]["durable_tokens"] == runs["hybrid"]["call"]["durable_tokens"], "durable token mismatch"
                pairs.append(runs)
                (root / "pairs.json").write_text(json.dumps(pairs, indent=2))
                print(json.dumps(dict(pair=pair, **{mode: {k: v for k, v in runs[mode]["call"].items() if k in ("ttft_ms", "total_ms")} for mode in modes})), flush=True)
        finally:
            app.stop.set()
            if args.background_replay:
                replay_worker.join()
            server.shutdown()
            worker.join()
            server.server_close()
            app.store.close()


if __name__ == "__main__":
    main()
