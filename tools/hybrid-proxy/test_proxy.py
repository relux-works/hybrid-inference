"""Bounded synthetic HTTP contract tests. These do not prove engine equivalence."""
import hashlib
import io
from contextlib import nullcontext
from unittest import mock
import http.server
import http.client
import json
from pathlib import Path
import struct
import sqlite3
import tempfile
import threading
import unittest

from policy import Policy
from proxy import Handler, HTTPError, Node, Proxy, RollbackError
from store import Store, prefix_hash
from stub_server import MODEL, decode_delta, encode_delta, make_server


class PolicyTests(unittest.TestCase):
    # These controls check cost ordering, a hysteresis band, and invalid inputs.
    def test_cost_and_override(self):
        policy = Policy()
        self.assertFalse(policy.choose(1, 0)["offload"])
        self.assertTrue(policy.choose(8000, 0)["offload"])
        self.assertFalse(policy.choose(8000, 0, link_rate=10_000)["offload"])
        self.assertTrue(Policy(always=True).choose(1, 0)["offload"])
        self.assertFalse(Policy(always=True).choose(0, 0)["offload"])
        with self.assertRaises(ValueError):
            policy.choose(-1, 0)
        with self.assertRaises(ValueError):
            Policy(local_rates=((0, 0),))
        with self.assertRaises(ValueError):
            Policy(link_rate=float("nan"))
        with self.assertRaises(ValueError):
            policy.choose(1, 0, link_rate=float("inf"))

    def test_hysteresis_and_depth(self):
        policy = Policy(local_rates=((0, 100), (1000, 10)), server_rate=1000,
                        fixed=0.8, kv_bytes=0, state_bytes=0)
        self.assertFalse(policy.choose(100, 0, previous=False)["offload"])
        self.assertTrue(policy.choose(100, 0, previous=True)["offload"])
        self.assertTrue(policy.choose(100, 1000)["offload"])


class TransferTests(unittest.TestCase):
    # Server response headers arrive before the body timer starts; empty responses do not update rate.
    def test_body_transfer_measurement(self):
        node = Node("http://127.0.0.1")
        calls = []
        def opened(*args, **kwargs):
            calls.append("headers")
            return io.BytesIO(b"1234")
        def clock():
            self.assertIn("headers", calls)
            calls.append("clock")
            return 100 if calls.count("clock") == 1 else 102
        with mock.patch.object(node, "open", side_effect=opened), mock.patch("proxy.time.monotonic", side_effect=clock):
            self.assertEqual(node.request("GET", "/delta", binary=True), b"1234")
        self.assertEqual(node.last_transfer_rate, 2)
        with mock.patch.object(node, "open", return_value=io.BytesIO(b"")):
            self.assertEqual(node.request("GET", "/empty", binary=True), b"")
        self.assertIsNone(node.last_transfer_rate)

    # A truncated HTTP body is a reconnectable transport error, including during recovery GET.
    def test_truncated_http_body_is_transport_error(self):
        node = Node("http://127.0.0.1")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = http.client.IncompleteRead(b"", 10)
        with mock.patch.object(node, "open", return_value=response):
            with self.assertRaisesRegex(OSError, "connection interrupted"):
                node.request("GET", "/snapshot", binary=True)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "sessions.sqlite")
        self.store = Store(self.path, MODEL)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    # Exact tokens after the matching 256-token block prevent cross-fork reuse.
    def test_shared_system_prompt_forks_and_restart(self):
        system = list(range(600))
        first = self.store.select("a", system + [700])
        first["tokens"] = system + [700, 701]
        self.store.save(first)
        second = self.store.select("a", system + [800])
        self.assertNotEqual(first["id"], second["id"])
        second["tokens"] = system + [800, 801]
        self.store.save(second)
        self.assertEqual(self.store.select("a", first["tokens"] + [999])["id"], first["id"])
        self.store.close()
        self.store = Store(self.path, MODEL)
        self.assertEqual(self.store.select("a", second["tokens"] + [999])["id"], second["id"])
        self.assertEqual(len(self.store.get(first["id"])["hashes"]), 2)
        with self.assertRaises(KeyError):
            self.store.select("foreign", system, first["id"])
        with self.assertRaises(ValueError):
            self.store.select("a", system + [555], first["id"])

    # Explicit IDs have precedence, but cannot adopt another client's session.
    def test_explicit_and_short_prefix(self):
        first = self.store.select("a", [1, 2], "named-session")
        first["tokens"] = [1, 2]
        self.store.save(first)
        self.assertEqual(self.store.select("a", [1, 2, 3])["id"], first["id"])
        self.assertEqual(self.store.select("a", [1, 2, 4], "named-session")["id"], first["id"])
        self.assertNotEqual(self.store.select("b", [1, 2, 3])["id"], first["id"])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.servers, self.threads = [], []
        self.prefill = self.start(make_server("prefill", database=str(Path(self.temp.name) / "prefill.sqlite")))
        self.decode = self.start(make_server("decode", slot_path=self.temp.name))
        self.config = dict(model_sha256=MODEL, prefill_url=self.url(self.prefill), decode_url=self.url(self.decode),
                           decode_token="decode-fixture", slot_path=self.temp.name, slots=2, prefill_timeout=0.5,
                           clients={f"fixture-{c}": dict(id=c, prefill_token=f"remote-{c}") for c in "abc"},
                           policy=dict(always=True))
        self.app = Proxy(self.config, str(Path(self.temp.name) / "proxy.sqlite"))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.app = self.app
        self.proxy = self.start(server)
        self.client = Node(self.url(self.proxy), "fixture-a")

    def start(self, server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append(server)
        self.threads.append(thread)
        return server

    @staticmethod
    def url(server):
        return "http://127.0.0.1:" + str(server.server_port)

    def tearDown(self):
        for server in reversed(self.servers):
            server.shutdown()
            server.server_close()
            if hasattr(server, "app") and hasattr(server.app, "close"):
                server.app.close()
        for thread in self.threads:
            thread.join()
        self.app.store.close()
        self.temp.cleanup()

    def chat(self, tokens, client=None, sid=None, stream=False):
        client = client or self.client
        with client.open("POST", "/v1/chat/completions", dict(messages=[], _tokens=tokens, stream=stream),
                         {"X-Hybrid-Session": sid} if sid else {}) as response:
            sid, data = response.headers["X-Hybrid-Session"], response.read()
        # Content-Length can finish before the handler releases its lease; manual replay controls need that boundary.
        with self.app.reserve(sid):
            pass
        return sid, data

    # Streaming uses the actual HTTP route and imported prefix; forks and client IDs stay separate.
    def test_stream_forks_foreign_404_and_restart(self):
        base = list(range(600))
        sid, body = self.chat(base + [700], stream=True)
        self.assertIn(b"data: [DONE]", body)
        self.assertIn(b'"content": "s"', body)
        first = self.app.store.get(sid)
        self.assertEqual(first["tokens"], base + [700, 901, 902])
        sid2, _ = self.chat(base + [800])
        self.assertNotEqual(sid, sid2)
        with self.assertRaises(HTTPError) as error:
            self.chat(first["tokens"] + [4], Node(self.url(self.proxy), "fixture-b"), sid)
        self.assertEqual(error.exception.status, 404)
        self.app.replay()
        self.app.store.close()
        self.app.store = Store(str(Path(self.temp.name) / "proxy.sqlite"), MODEL)
        continued, _ = self.chat(first["tokens"] + [4])
        self.assertEqual(continued, sid)
        self.assertGreaterEqual(self.decode.app.imports, 2)

    # Retain A's C9 control: a critical HTTP 429 must complete locally and replay every token once.
    def test_429_critical_falls_back_locally_and_replays(self):
        original = self.prefill.app.handle
        calls = []
        def limited(method, path, query, body, client):
            if path.endswith('/extend') and body['return_delta']:
                calls.append(path)
                return 429, {'error': 'binary response limit'}
            return original(method, path, query, body, client)
        with mock.patch.object(self.prefill.app, 'handle', side_effect=limited):
            sid, _ = self.chat([71, 72, 73])
        self.assertEqual(len(calls), 1)
        local = self.app.store.get(sid)
        self.assertFalse(local['offload'])
        self.assertEqual(local['tokens'], [71, 72, 73, 901, 902])
        self.app.replay()
        self.app.replay()
        remote = self.app.remote(self.config['clients']['fixture-a']).request('GET', '/v1/sessions/' + local['remote_id'])
        self.assertEqual(remote['tokens'], local['tokens'])

    # Offline completions remain usable; lost acknowledgements and restart replay do not duplicate tokens.
    def test_offline_journal_and_lost_ack_replay(self):
        self.prefill.app.available = False
        sid, _ = self.chat([10, 11, 12])
        local = self.app.store.get(sid)
        self.assertEqual(local["tokens"], [10, 11, 12, 901, 902])
        self.assertIsNone(local["remote_id"])
        self.app.store.close()
        self.app.store = Store(str(Path(self.temp.name) / "proxy.sqlite"), MODEL)
        self.prefill.app.available = True
        self.prefill.app.lose_ack_once = True
        self.app.replay()
        self.app.replay()
        self.app.replay()
        local = self.app.store.get(sid)
        remote = self.app.remote(self.config["clients"]["fixture-a"]).request("GET", "/v1/sessions/" + local["remote_id"])
        self.assertEqual(remote["tokens"], local["tokens"])
        self.assertEqual(local["remote_n"], len(local["tokens"]))

    # 409 behind replays the durable suffix; ahead restores a verified server snapshot.
    def test_409_and_snapshot_resync(self):
        sid, _ = self.chat([1, 2, 3])
        session = self.app.store.get(sid)
        self.prefill.app.conflict_once = True
        continued, _ = self.chat(session["tokens"] + [4], sid=sid)
        self.assertEqual(continued, sid)
        session = self.app.store.get(sid)
        self.app.replay()
        ahead = session["tokens"] + [55]
        with self.prefill.app.db:
            self.prefill.app.db.execute("UPDATE sessions SET tokens=? WHERE id=?", (json.dumps(ahead), session["remote_id"]))
        continued, _ = self.chat(ahead + [66], sid=sid)
        self.assertEqual(continued, sid)
        self.assertEqual(self.app.store.get(sid)["tokens"], ahead + [66, 901, 902])

    # An incompatible remote prefix must be rejected before decoder save/restore or journal changes.
    def test_resync_incompatible_prefix_is_atomic(self):
        sid, _ = self.chat([1, 2, 3])
        self.app.replay()
        self.app.replay()
        session = self.app.store.get(sid)
        before = self.decode.app.slots[session["slot"]][:]
        with self.prefill.app.db:
            self.prefill.app.db.execute("UPDATE sessions SET tokens=? WHERE id=?",
                                        (json.dumps([8, 8, 8]), session["remote_id"]))
        with mock.patch.object(self.app.decode, "request", wraps=self.app.decode.request) as decoder:
            with self.assertRaises(HTTPError) as error:
                self.app.resync(session, self.config["clients"]["fixture-a"], before + [4])
            self.assertEqual(error.exception.status, 409)
            decoder.assert_not_called()
        self.assertEqual(self.decode.app.slots[session["slot"]], before)
        self.assertEqual(self.app.store.get(sid), session)

    # A foreign model is incompatible even when its token prefix happens to match the journal.
    def test_resync_incompatible_model_is_atomic(self):
        sid, _ = self.chat([21, 22, 23])
        self.app.replay()
        self.app.replay()
        session = self.app.store.get(sid)
        before = self.decode.app.slots[session["slot"]][:]
        with mock.patch.object(self.prefill.app, "model", "f" * 64), mock.patch.object(self.app.decode, "request", wraps=self.app.decode.request) as decoder:
            with self.assertRaises(HTTPError) as error:
                self.app.resync(session, self.config["clients"]["fixture-a"], before + [4])
            self.assertEqual(error.exception.status, 409)
            decoder.assert_not_called()
        self.assertEqual(self.decode.app.slots[session["slot"]], before)
        self.assertEqual(self.app.store.get(sid), session)

    # A changing snapshot or failed SQLite commit rolls the decoder back; the durable prefix stays old.
    def test_resync_snapshot_and_journal_failures_roll_back(self):
        for failure in ("snapshot", "journal"):
            with self.subTest(failure=failure):
                sid, _ = self.chat([31, 32, len(self.app.store.pending())])
                self.app.replay()
                self.app.replay()
                session = self.app.store.get(sid)
                before = self.decode.app.slots[session["slot"]][:]
                ahead = before + [55]
                with self.prefill.app.db:
                    self.prefill.app.db.execute("UPDATE sessions SET tokens=? WHERE id=?",
                                                (json.dumps(ahead), session["remote_id"]))
                original = self.prefill.app.handle
                def changed_snapshot(method, path, query, body, client):
                    if failure == "snapshot" and path.endswith("/snapshot"):
                        return 200, json.dumps(dict(model_sha256=MODEL, tokens=[8, 8, 8])).encode()
                    return original(method, path, query, body, client)
                if failure == "journal":
                    self.app.store.db.execute("CREATE TEMP TRIGGER fail_commit BEFORE UPDATE ON sessions BEGIN SELECT RAISE(ABORT, 'injected journal failure'); END")
                try:
                    with mock.patch.object(self.prefill.app, "handle", side_effect=changed_snapshot):
                        with self.assertRaises(HTTPError if failure == "snapshot" else sqlite3.IntegrityError) as error:
                            self.app.resync(session, self.config["clients"]["fixture-a"], ahead + [66])
                        if failure == "snapshot":
                            self.assertEqual(error.exception.status, 409)
                finally:
                    if failure == "journal":
                        self.app.store.db.execute("DROP TRIGGER fail_commit")
                self.assertEqual(self.decode.app.slots[session["slot"]], before)
                self.assertEqual(self.app.store.get(sid), session)

    # C12: rollback outages must escape recovery, retain the backup and force local prefill after restart.
    def test_recovery_rollback_failure_retains_backup_and_resets_slot(self):
        sid, _ = self.chat([31, 32, 33])
        self.app.replay()
        self.app.replay()
        session = self.app.store.get(sid)
        before = session["tokens"][:]
        ahead = before + [55]
        with self.prefill.app.db:
            self.prefill.app.db.execute("UPDATE sessions SET tokens=? WHERE id=?",
                                        (json.dumps(ahead), session["remote_id"]))
        original_remote, original_decode = self.prefill.app.handle, self.app.decode.request
        def inconsistent(method, path, query, body, client):
            if path.endswith("/snapshot"):
                return 200, json.dumps(dict(model_sha256=MODEL, tokens=[8, 8, 8])).encode()
            return original_remote(method, path, query, body, client)
        def rollback_outage(method, path, data=None, headers=None):
            if "action=restore" in path and data["filename"].endswith("-resync-backup.bin"):
                raise OSError("injected rollback connection loss")
            return original_decode(method, path, data, headers)
        for cleanup_outage in (False, True):
            self.decode.app.slots[session["slot"]] = before[:]
            self.app.store.verified.add(sid)
            cleanup = mock.patch.object(Path, "unlink", side_effect=OSError("injected cleanup failure")) if cleanup_outage else nullcontext()
            with self.subTest(cleanup_outage=cleanup_outage), cleanup, \
                    mock.patch.object(self.prefill.app, "handle", side_effect=inconsistent), \
                    mock.patch.object(self.app.decode, "request", side_effect=rollback_outage):
                with self.assertRaisesRegex(RollbackError, "rollback failed") as error:
                    self.app.recover(session, self.config["clients"]["fixture-a"], ahead + [66])
                self.assertEqual(error.exception.status, 503)
        backup = Path(self.temp.name) / (sid + "-resync-backup.bin")
        self.assertEqual(json.loads(backup.read_text())["tokens"], before)
        self.assertEqual(self.app.store.get(sid)["tokens"], before)
        self.assertNotIn(sid, self.app.store.verified)
        self.assertEqual(self.decode.app.slots[session["slot"]], [8, 8, 8])
        # A dirty slot may even look like a compatible extension. Restart must still erase it before reuse.
        self.decode.app.slots[session["slot"]] = ahead[:]
        self.app.store.close()
        self.app.store = Store(str(Path(self.temp.name) / "proxy.sqlite"), MODEL)
        self.prefill.app.available = False
        with mock.patch.object(self.app.decode, "request", wraps=original_decode) as decoder:
            self.chat(before + [66], sid=sid)
            self.assertTrue(any("action=erase" in call.args[1] for call in decoder.call_args_list))
        self.assertFalse(backup.exists())
        self.assertEqual(self.app.store.get(sid)["tokens"], before + [66, 901, 902])
        self.assertEqual(self.decode.app.slots[session["slot"]], before + [66, 901, 902])

    # Loss during either recovery branch must complete locally, then replay each token once on reconnect.
    def test_recovery_get_outage_completes_locally_and_replays(self):
        original = Node.open
        for branch in ("divergence", "409"):
            for outage in (OSError("recovery connection lost"), HTTPError(503), http.client.IncompleteRead(b"", 10)):
                with self.subTest(branch=branch, outage=type(outage).__name__):
                    sid, _ = self.chat([41, 42, len(self.app.store.pending())])
                    self.app.replay()
                    self.app.replay()
                    session = self.app.store.get(sid)
                    ahead = session["tokens"] + ([55] if branch == "divergence" else [])
                    with self.prefill.app.db:
                        self.prefill.app.db.execute("UPDATE sessions SET tokens=? WHERE id=?",
                                                    (json.dumps(ahead), session["remote_id"]))
                    self.prefill.app.conflict_once = branch == "409"
                    calls = []
                    def fail_recovery(node, method, path, data=None, headers=None):
                        if node.url == self.url(self.prefill) and method == "GET" and path.startswith("/v1/sessions/"):
                            calls.append(path)
                            if len(calls) == 2:
                                raise outage
                        return original(node, method, path, data, headers)
                    prompt = ahead + [66, 67]
                    with mock.patch.object(Node, "open", new=fail_recovery):
                        continued, _ = self.chat(prompt, sid=sid)
                    self.assertEqual(continued, sid)
                    self.assertEqual(len(calls), 2)
                    local = self.app.store.get(sid)
                    self.assertFalse(local["offload"])
                    self.assertEqual(local["tokens"], prompt + [901, 902])
                    self.app.replay()
                    self.app.replay()
                    remote = self.app.remote(self.config["clients"]["fixture-a"]).request("GET", "/v1/sessions/" + local["remote_id"])
                    self.assertEqual(remote["tokens"], local["tokens"])

    # Queued 202 shadows are not applied yet; FIFO drains before a critical extend and rejects stale duplicates.
    def test_delayed_202_shadows_critical_drain_without_token_loss(self):
        sid, _ = self.chat([10, 11, 12])
        queued, applied = [], []
        original = self.prefill.app.handle
        def delayed(method, path, query, body, client):
            if path.endswith("/extend"):
                if not body["return_delta"]:
                    queued.append((method, path, query, dict(body), client))
                    return 202, None
                for item in queued:
                    applied.append(original(*item)[0])
                queued.clear()
            return original(method, path, query, body, client)
        with mock.patch.object(self.prefill.app, "handle", side_effect=delayed):
            self.app.replay()
            self.app.replay()
            self.app.replay()
            self.assertEqual(len(queued), 3)
            self.assertEqual(queued[0][3]["n0"], queued[1][3]["n0"])
            old = self.app.store.get(sid)
            continued, _ = self.chat(old["tokens"] + [13, 14], sid=sid)
            self.assertEqual(continued, sid)
            self.assertTrue(self.app.store.get(sid)["offload"])
        self.assertEqual(applied[0], 202)
        self.assertTrue(all(status == 409 for status in applied[1:]))
        self.assertFalse(queued)
        self.app.replay()
        self.app.replay()
        local = self.app.store.get(sid)
        remote = self.app.remote(self.config["clients"]["fixture-a"]).request("GET", "/v1/sessions/" + local["remote_id"])
        self.assertEqual(local["tokens"], old["tokens"] + [13, 14, 901, 902])
        self.assertEqual(remote["tokens"], local["tokens"])
        self.assertEqual(self.app.store.get(sid)["remote_n"], len(local["tokens"]))

    # A RAM-only server restart may lose the session ID; the journal recreates its full state once.
    def test_server_session_loss_recreates_and_replays(self):
        sid, _ = self.chat([10, 20, 30])
        self.app.replay()
        self.app.replay()
        old = self.app.store.get(sid)
        with self.prefill.app.db:
            self.prefill.app.db.execute("DELETE FROM sessions")
        self.app.replay()
        self.app.replay()
        recovered = self.app.store.get(sid)
        self.assertNotEqual(old["remote_id"], recovered["remote_id"])
        node = self.app.remote(self.config["clients"]["fixture-a"])
        info = node.request("GET", "/v1/sessions/" + recovered["remote_id"])
        self.assertEqual(info["tokens"], old["tokens"])
        self.assertEqual(recovered["remote_n"], len(old["tokens"]))

    # Bad or missing authorization and unsafe IDs cannot allocate sessions or reach the decoder.
    def test_auth_and_session_id_validation(self):
        for client, sid, status in [(Node(self.url(self.proxy)), None, 401),
                                    (self.client, "../bad", 400)]:
            with self.assertRaises(HTTPError) as error:
                self.chat([1, 2], client, sid)
            self.assertEqual(error.exception.status, status)
        self.assertEqual(self.app.store.pending(), [])

    # All four server session operations hide foreign IDs and leave the victim unchanged.
    def test_prefill_namespace(self):
        a, b = Node(self.url(self.prefill), "a"), Node(self.url(self.prefill), "b")
        sid = a.request("POST", "/v1/sessions", {"model_sha256": MODEL})["session_id"]
        path = "/v1/sessions/" + sid
        before = a.request("GET", path)
        for method, suffix, data in [("GET", "", None), ("POST", "/extend", {}),
                                     ("GET", "/snapshot", None), ("DELETE", "", None)]:
            with self.subTest(method=method, suffix=suffix), self.assertRaises(HTTPError) as error:
                b.request(method, path + suffix, data)
            self.assertEqual(error.exception.status, 404)
        self.assertEqual(a.request("GET", path)["tokens"], before["tokens"])
        self.assertEqual(b.request("GET", "/v1/sessions"), [])

    # Three clients with four sessions each exercise parking and exact per-session journals.
    def test_three_clients_four_sessions_interleaved(self):
        records = []
        for c in "abc":
            client = Node(self.url(self.proxy), "fixture-" + c)
            for index in range(4):
                prompt = [ord(c), index, 77]
                sid, _ = self.chat(prompt, client)
                records.append((client, sid, prompt + [901, 902]))
        self.app.replay()
        self.app.policy = Policy(always=False, local_rates=((0, 1e9),))
        for client, sid, old in records:
            continued, _ = self.chat(old + [88], client, sid)
            self.assertEqual(continued, sid)
            self.assertEqual(self.app.store.get(sid)["tokens"], old + [88, 901, 902])
        self.app.replay()
        self.app.replay()
        for client, sid, _ in records:
            session = self.app.store.get(sid)
            config = self.app.client("Bearer " + client.token)
            remote = self.app.remote(config).request("GET", "/v1/sessions/" + session["remote_id"])
            self.assertEqual(remote["tokens"], session["tokens"])
        self.assertEqual(len({sid for _, sid, _ in records}), 12)

    # Synthetic corruption tests prove the fixture rejects each class atomically, not native KV imports.
    def test_synthetic_delta_negative_cases(self):
        node = Node(self.url(self.decode), "decode-fixture")
        node.request("POST", "/slots/0?action=import_delta", encode_delta(MODEL, "s", [], [1, 2]))
        before = node.request("GET", "/slots/0/info")
        good = encode_delta(MODEL, "s", [1, 2], [3])
        def header_change(key, value):
            size, = struct.unpack("<I", good[:4])
            header = json.loads(good[4:4 + size])
            header[key] = value
            raw = json.dumps(header).encode()
            return struct.pack("<I", len(raw)) + raw + good[4 + size:]
        cases = {
            "wrong_model": header_change("model_sha256", "f" * 64),
            "gap": header_change("n0", 3), "overlap": header_change("n0", 1),
            "truncated": good[:-1], "checksum": good[:-1] + bytes([good[-1] ^ 1]),
            "section_checksum": good[:-48] + bytes([good[-48] ^ 1]) + good[-47:],
            "prefix": header_change("prefix_hash_n1", "f" * 64),
            "layout": header_change("sections", []),
        }
        for name, delta in cases.items():
            with self.subTest(name=name), self.assertRaises(HTTPError) as error:
                node.request("POST", "/slots/0?action=import_delta", delta)
            self.assertEqual(error.exception.status, 400)
            after = node.request("GET", "/slots/0/info")
            self.assertEqual((after["tokens"], after["prefix_hash"]), (before["tokens"], before["prefix_hash"]))
        node.request("POST", "/slots/0?action=import_delta", good)
        self.assertEqual(node.request("GET", "/slots/0/info")["tokens"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
