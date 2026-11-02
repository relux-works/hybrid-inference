"""Real adapter HTTP plus the proxy's existing Node, Proxy and decoder fixture."""

import copy
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid-proxy"))
from proxy import Handler as ProxyHandler, HTTPError, Node, Proxy
from store import prefix_hash
import stub_server
import test_proxy

import bridge
from fake_backend import FakeBackend
from node_adapter import Adapter, Server
from test_bridge import fixture


class ProxyServer(http.server.ThreadingHTTPServer):
    app: Proxy


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _, self.target = fixture()
        self.target["model_sha256"] = stub_server.MODEL
        self.backend = FakeBackend(self.target)
        self.app = Adapter(self.target, {"a": "remote-a-credential", "b": "remote-b-credential"}, self.backend, self.root, codec="numpy")
        self.servers, self.threads = [], []
        server = Server(("127.0.0.1", 0))
        server.app = self.app
        self.node = self.start(server, "remote-a-credential")

    def start(self, server, token=""):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append(server)
        self.threads.append(thread)
        return Node("http://127.0.0.1:" + str(server.server_port), token, timeout=120)

    def tearDown(self):
        for server in reversed(self.servers):
            server.shutdown()
            server.server_close()
            if isinstance(server.app, Proxy):
                server.app.store.close()
            elif hasattr(server.app, "close"):
                server.app.close()
        for thread in self.threads:
            thread.join()
        self.tmp.cleanup()

    def create(self):
        return "/v1/sessions/" + self.node.request("POST", "/v1/sessions", {"model_sha256": self.target["model_sha256"]})["session_id"]

    # The existing node-facing namespace test runs unchanged against this adapter after using configured credentials.
    def test_existing_proxy_namespace(self):
        case = test_proxy.HTTPTests("test_prefill_namespace")
        case.prefill = self.servers[0]
        case.config = dict(clients={"fixture-a": dict(prefill_token="remote-a-credential"), "fixture-b": dict(prefill_token="remote-b-credential")})
        case.test_prefill_namespace()

    def extend(self, path, before, suffix, delta=True):
        try:
            return self.node.request("POST", path + "/extend", dict(n0=len(before), prefix_hash_n0=prefix_hash(before),
                                     tokens=suffix, return_delta=delta), binary=delta)
        except HTTPError as error:
            self.fail("valid extend refused: " + str(error.status))

    # Fresh rounding, shadow catch-up and an unaligned append start match scalar full-envelope bytes and native import.
    def test_fresh_append_reference_bytes(self):
        path = self.create()
        first = self.extend(path, [], list(range(35)))
        header_size, = struct.unpack_from("<I", first)
        header = json.loads(first[4:4 + header_size])
        self.assertEqual((header["n0"], header["n1"], header["tokens"]), (0, 32, list(range(32))))
        self.assertEqual(self.node.request("GET", path)["n"], 32)
        self.extend(path, list(range(32)), [32, 33, 34], False)
        second = self.extend(path, list(range(35)), list(range(35, 52)))
        self.assertEqual(self.node.request("GET", path)["n"], 48)
        self.assertEqual(self.backend.calls[-1]["n0"], 35)
        self.assertEqual(self.backend.calls[-1]["reused"], 32)
        for n0, n1, wire in ((0, 32, first), (35, 48, second)):
            with tempfile.TemporaryDirectory(dir=self.root) as tmp:
                root = Path(tmp)
                capture = self.backend.prefill(list(range(n1)), n0, root)
                bridge.convert(capture, self.target, root / "reference.delta", "reference", path.rsplit("/", 1)[1])
                self.assertEqual(hashlib.sha256(wire).digest(), hashlib.sha256((root / "reference.delta").read_bytes()).digest())
        # An ordinary aligned append forms a real native import chain with the fresh delta.
        other = self.create()
        a = self.extend(other, [], list(range(32)))
        b = self.extend(other, list(range(32)), list(range(32, 64)))
        binary = os.environ.get("HYBRID_VLLM_NATIVE_CHECK")
        self.assertTrue(binary, "native helper required")
        assert binary is not None
        (self.root / "a.delta").write_bytes(a)
        (self.root / "b.delta").write_bytes(b)
        (self.root / "target.json").write_text(json.dumps(self.target))
        result = subprocess.run([binary, str(self.root / "a.delta"), str(self.root / "b.delta"), str(self.root / "target.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    # Borrowed V2 chunks preserve exact bytes and committed positions until the final fixed state commits the aligned endpoint.
    def test_chunked_final_fixed_state(self):
        path = self.create()
        wires = []
        for index in range(3):
            n0, n1 = index * 16, (index + 1) * 16
            wire = self.node.request("POST", path + "/extend", dict(n0=n0, prefix_hash_n0=prefix_hash(list(range(n0))),
                tokens=list(range(n0, n1)), return_delta=True, delta_version=2, transfer_id="chunked-adapter",
                transfer_n0=0, transfer_n1=48, chunk_index=index, chunk_count=3), binary=True)
            size, = struct.unpack_from("<I", wire)
            header = json.loads(wire[4:4 + size])
            self.assertEqual((header["version"], header["body_format"], header["fixed_state"]),
                             (2, "llama-native-chunks-v3", index == 2))
            self.assertEqual(header["sections"][0]["bytes"], (156_894_772 if index == 2 else 32 + 32 * 12) + 16 * (34_816 + 24))
            self.assertEqual(bridge.xxh128_file(io.BytesIO(wire[4 + size:-48])), wire[-48:])
            wires.append(wire)
        binary = os.environ["HYBRID_VLLM_NATIVE_CHECK"]
        paths = []
        for index, wire in enumerate(wires):
            file = self.root / (str(index) + ".delta")
            file.write_bytes(wire)
            paths.append(str(file))
        target = self.root / "target.json"
        target.write_text(json.dumps(self.target))
        result = subprocess.run([binary, *paths, str(target)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("borrowed native body views", result.stdout)
        result = subprocess.run([binary, *paths[:-1], str(target)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("missing final chunk", result.stderr)
        # Correctly checksummed extra/absent fixed bytes must fail native validation, not just envelope parsing.
        size, = struct.unpack_from("<I", wires[0])
        first_body = wires[0][4 + size:-48]
        last_size, = struct.unpack_from("<I", wires[-1])
        last_body = wires[-1][4 + last_size:-48]
        fixed = last_body[len(first_body):]
        for name, wire, body in (("early-fixed", wires[0], first_body + fixed), ("missing-fixed", wires[-1], last_body[:-len(fixed)])):
            size, = struct.unpack_from("<I", wire)
            header = json.loads(wire[4:4 + size])
            header["sections"][0]["bytes"] = len(body)
            encoded = json.dumps(header, separators=(",", ":")).encode()
            bad = self.root / (name + ".delta")
            bad.write_bytes(struct.pack("<I", len(encoded)) + encoded + body + bridge.xxh128_file(io.BytesIO(body)))
            chain = [str(bad), *paths[1:]] if name == "early-fixed" else [*paths[:-1], str(bad)]
            result = subprocess.run([binary, *chain, str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("invalid native chunk", result.stderr)

    # Invalid transfer metadata is rejected before backend work or journal changes, beside a valid standalone request.
    def test_chunk_metadata_refusals(self):
        path = self.create()
        good = dict(n0=0, prefix_hash_n0=prefix_hash([]), tokens=list(range(16)), return_delta=True,
                    delta_version=2, transfer_id="valid", transfer_n0=0, transfer_n1=32, chunk_index=0, chunk_count=2)
        before = self.node.request("GET", path)
        variants = [dict(good, **{key: value}) for key, value in
                    (("delta_version", 1), ("delta_version", True), ("transfer_id", ""), ("transfer_id", "x" * 129), ("transfer_id", "\u00e9" * 100),
                     ("chunk_index", True), ("chunk_index", 2), ("chunk_count", 0), ("chunk_count", 33),
                     ("transfer_n0", 1), ("transfer_n1", 16), ("transfer_n1", 33), ("return_delta", False))]
        for key in ("transfer_id", "transfer_n0", "transfer_n1", "chunk_index", "chunk_count"):
            bad = dict(good)
            del bad[key]
            variants.append(bad)
        for bad in variants:
            with self.subTest(body=bad), self.assertRaises(HTTPError) as error:
                self.node.request("POST", path + "/extend", bad, binary=True)
            self.assertEqual(error.exception.status, 400)
            self.assertEqual(self.node.request("GET", path), before)
            self.assertEqual(self.backend.calls, [])
        self.extend(path, [], list(range(16)))
        self.assertEqual(self.node.request("GET", path)["n"], 16)

    # Bad identity, stale prefixes, invalid tokens, unsupported slots and unaligned historical states leave the journal intact.
    def test_refusals_auth_release(self):
        path = self.create()
        for token, status in (("", 401), ("wrong", 401), ("remote-b-credential", 404)):
            with self.assertRaises(HTTPError) as error:
                Node(self.node.url, token).request("GET", path)
            self.assertEqual(error.exception.status, status)
        with self.assertRaises(HTTPError) as error:
            self.node.request("POST", "/v1/sessions", dict(model_sha256="f" * 64))
        self.assertEqual(error.exception.status, 400)
        self.extend(path, [], list(range(19)), False)
        before = self.node.request("GET", path)
        with self.assertRaises(HTTPError) as error:
            self.node.request("GET", path + "/snapshot", binary=True)
        self.assertEqual(error.exception.status, 400)
        self.assertEqual(json.loads(error.exception.body)["error"], "unaligned historical endpoint")
        good = dict(n0=19, prefix_hash_n0=prefix_hash(list(range(19))), tokens=list(range(19, 36)), return_delta=True)
        for key, value, status in (("n0", 0, 409), ("tokens", [-1], 400), ("tokens", [True], 400),
                                   ("return_delta", 1, 400), ("mtp", True, 400), ("speculative", {}, 400), ("endpoint", 18, 400)):
            with self.assertRaises(HTTPError) as error:
                self.node.request("POST", path + "/extend", dict(good, **{key: value}), binary=True)
            self.assertEqual(error.exception.status, status, key)
            self.assertEqual(self.node.request("GET", path), before)
        with mock.patch.object(self.backend, "prefill", side_effect=bridge.Refusal("speculative decoding is unsupported")):
            with self.assertRaises(HTTPError) as error:
                self.node.request("POST", path + "/extend", good)
            self.assertEqual(error.exception.status, 400)
        self.assertEqual(self.node.request("GET", path), before)
        self.assertIsNone(self.node.request("DELETE", path))
        with self.assertRaises(HTTPError) as error:
            self.node.request("GET", path)
        self.assertEqual(error.exception.status, 404)
        self.assertEqual(self.node.request("GET", "/v1/sessions"), [])
        bad = copy.deepcopy(self.target)
        bad["model_layout"]["layers"][3]["kv_heads"] = 2
        with self.assertRaisesRegex(bridge.Refusal, "head geometry"):
            Adapter(bad, {"a": "remote-a-credential"}, self.backend, self.root)

    # A checker consumer detects body corruption; public health hides tenant counters and backend failure is visible.
    def test_checksums_and_health(self):
        path = self.create()
        data = self.extend(path, [], list(range(16)))
        size, = struct.unpack_from("<I", data)
        body = data[4 + size:-48]
        self.assertEqual(bridge.xxh128_file(io.BytesIO(body)), data[-48:])
        changed = bytearray(body)
        changed[100] ^= 1
        self.assertNotEqual(bridge.xxh128_file(io.BytesIO(changed)), data[-48:])
        self.app.public = True
        health = Node(self.node.url).request("GET", "/v1/health")
        self.assertNotIn("sessions", health)
        self.assertEqual(health["export_block_size"], 16)
        self.backend.available = False
        with self.assertRaises(HTTPError) as error:
            self.node.request("GET", "/v1/health")
        self.assertEqual(error.exception.status, 503)

    # Reuse the proxy's HTTP machinery and decoder journal tests, replacing only its synthetic native-state reader.
    def test_proxy_fresh_append_tail_and_release(self):
        decode = self.start(stub_server.make_server("decode", slot_path=self.root), "decode-fixture")
        config = dict(benchmark=True, model_sha256=self.target["model_sha256"], prefill_url=self.node.url,
                      decode_url=decode.url, decode_token="decode-fixture", prefill_timeout=120,
                      slot_path=str(self.root / "slots"), slots=2, export_block_size=16, extend_chunk_tokens=35,
                      clients={"fixture-a": dict(id="a", prefill_token="remote-a-credential")}, policy=dict(always=True))
        app = Proxy(config, str(self.root / "proxy.sqlite"))
        server = ProxyServer(("127.0.0.1", 0), ProxyHandler)
        server.app = app
        client = self.start(server, "fixture-a")
        def native_decode(raw, model, before):
            size, = struct.unpack_from("<I", raw)
            header = json.loads(raw[4:4 + size])
            self.assertEqual(header["body_format"], "llama-native-chunks-v3")
            self.assertEqual(header["n0"], len(before))
            self.assertEqual(bridge.xxh128_file(io.BytesIO(raw[4 + size:-48])), raw[-48:])
            tokens = before + header["tokens"]
            self.assertEqual(header["prefix_hash_n1"], prefix_hash(tokens))
            return tokens
        with mock.patch.object(stub_server, "decode_delta", side_effect=native_decode):
            sid = "adapter-integration"
            for tokens, expected_remote in ((list(range(70)), 64), (list(range(70)) + [901, 902] + list(range(72, 110)), 24)):
                try:
                    result = client.request("POST", "/v1/chat/completions", dict(messages=[], _tokens=tokens),
                        {"X-Hybrid-Session": sid, "X-Hybrid-Bench-Mode": "offload"})
                except HTTPError as error:
                    self.fail("valid aligned proxy call refused: " + str(error.status))
                with app.reserve(sid):
                    pass
                trace = result["hybrid_bench"]
                self.assertIsNone(trace["fallback_reason"])
                self.assertEqual(sum(c["n1"] - c["n0"] for c in trace["chunks"]), expected_remote)
                self.assertEqual(app.decode.request("GET", "/slots/0/info")["tokens"], tokens + [901, 902])
            # Existing release path exercises authenticated DELETE on this adapter.
            self.assertEqual(client.request("DELETE", "/v1/sessions/" + sid), dict(status="released"))
            self.assertEqual(self.node.request("GET", "/v1/sessions"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
