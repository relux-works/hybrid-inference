"""Opt-in benchmark switches retain the production default."""
import json
import urllib.error
from unittest import mock
import unittest

import test_proxy
from proxy import HTTPError, Node, Proxy
from policy import extend_chunk_limit
from store import prefix_hash


class BenchmarkProxyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_proxy.HTTPTests('runTest')
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    # A direct proxy uses the 27B-safe bound independently of cost estimates; explicit geometry admits its maximum and rejects one more.
    def test_native_chunk_startup_bound(self):
        import tempfile
        from pathlib import Path
        config = dict(self.fixture.config)
        config['policy'] = dict(kv_bytes=1, state_bytes=0)
        for geometry in ({}, dict(kv_bytes=6528, state_bytes=20201472),
                         dict(kv_bytes=272, state_bytes=0, window_bytes=500 * 1024**2)):
            safe = extend_chunk_limit(**geometry)
            config['delta_geometry'] = geometry
            with tempfile.TemporaryDirectory() as directory:
                config.pop('extend_chunk_tokens', None)
                app = Proxy(config, str(Path(directory) / 'default.sqlite'))
                self.assertEqual(app.extend_chunk_tokens, safe)
                app.store.close()
                config['extend_chunk_tokens'] = safe
                app = Proxy(config, str(Path(directory) / 'maximum.sqlite'))
                self.assertEqual(app.extend_chunk_tokens, safe)
                app.store.close()
                config['extend_chunk_tokens'] = safe + 1
                with self.assertRaisesRegex(ValueError, 'delta body/header bound'):
                    Proxy(config, str(Path(directory) / 'over.sqlite'))

    # Benchmark headers have no authority unless the operator enables them; invalid modes also refuse.
    def test_switch_requires_opt_in(self):
        fixture = self.fixture
        body = dict(messages=[], _tokens=[1, 2, 3])
        for enabled, mode in ((False, 'local'), (True, 'invalid')):
            fixture.app.benchmark = enabled
            with self.assertRaises(HTTPError) as error:
                fixture.client.request('POST', '/v1/chat/completions', body, {'X-Hybrid-Bench-Mode': mode})
            self.assertEqual(error.exception.status, 400)
        fixture.chat([1, 2, 3])
        self.assertGreater(fixture.decode.app.imports, 0)

    # Local mode performs no remote request, while explicit offload imports on the same fixture.
    def test_local_disables_remote(self):
        fixture = self.fixture
        fixture.app.benchmark = True
        body = dict(messages=[], _tokens=[1, 2, 3])
        with mock.patch.object(fixture.app, 'sync', side_effect=AssertionError('local contacted prefill')):
            fixture.client.request('POST', '/v1/chat/completions', body, {'X-Hybrid-Bench-Mode': 'local'})
        self.assertEqual(fixture.decode.app.imports, 0)
        fixture.client.request('POST', '/v1/chat/completions', dict(body, _tokens=[5, 6, 7]),
                               {'X-Hybrid-Bench-Mode': 'offload'})
        self.assertEqual(fixture.decode.app.imports, 1)

class ChunkAndLifecycleTests(unittest.TestCase):
    setUp = BenchmarkProxyTests.setUp
    tearDown = BenchmarkProxyTests.tearDown

    # For several interrupted prefixes, fallback starts from the original slot and the staged host bytes are discarded.
    def test_interruption_preserves_slot_before_fallback(self):
        fixture = self.fixture
        sid, _ = fixture.chat([10, 11, 12, 13])
        fixture.app.extend_chunk_tokens = 2
        original = fixture.app.store.get(sid)
        owner = fixture.config['clients']['fixture-a']
        before = fixture.app.decode.request('GET', f"/slots/{original['slot']}/info")
        before.pop('last_used')
        dispatch = fixture.prefill.app.handle
        for k in (1, 2, 3):
            # Use a new remote after each interrupted attempt; its acknowledged prefix may be ahead locally.
            session = dict(original, remote_id=None, remote_n=0)
            calls = []
            def fail_after_k(method, path, query, body, client):
                if path.endswith('/extend') and body.get('return_delta'):
                    calls.append(body['n0'])
                    if len(calls) == k + 1:
                        return 503, {'error': 'ram budget'}
                return dispatch(method, path, query, body, client)
            with mock.patch.object(fixture.prefill.app, 'handle', side_effect=fail_after_k):
                fixture.app.prepare_session(owner, {}, session, original['tokens'] + list(range(20, 29)))
            actual = fixture.app.decode.request('GET', f"/slots/{session['slot']}/info")
            actual.pop('last_used')
            self.assertEqual(actual, before)
            self.assertEqual(session['_remote_applied'], 0)
            self.assertEqual(session['_bench']['fallback_reason'], 'ram_budget')
            self.assertNotIn(session['slot'], fixture.decode.app.pending)

    # A final import that commits but loses its acknowledgement is restored before fallback, with no partial application counted.
    def test_lost_final_ack_restores_original_slot(self):
        fixture = self.fixture
        sid, _ = fixture.chat([10, 11, 12, 13])
        session = fixture.app.store.get(sid)
        fixture.app.extend_chunk_tokens = 2
        before = fixture.app.decode.request('GET', f"/slots/{session['slot']}/info")
        before.pop('last_used')
        dispatch = fixture.app.decode.request
        seen = []
        def lose_final(method, path, data=None, headers=None, **kwargs):
            result = dispatch(method, path, data, headers, **kwargs)
            if 'action=import_delta' in path:
                import struct
                size, = struct.unpack('<I', data[:4])
                header = json.loads(data[4:4 + size])
                if header['fixed_state']:
                    seen.append(result)
                    raise OSError('lost final acknowledgement')
            return result
        with mock.patch.object(fixture.app.decode, 'request', side_effect=lose_final):
            fixture.app.prepare_session(fixture.config['clients']['fixture-a'], {}, session,
                                        session['tokens'] + list(range(20, 29)))
        self.assertEqual(len(seen), 1)
        self.assertGreater(seen[0]['n'], before['n'])
        actual = fixture.app.decode.request('GET', f"/slots/{session['slot']}/info")
        actual.pop('last_used')
        self.assertEqual(actual, before)
        self.assertEqual(session['_remote_applied'], 0)
        self.assertEqual(session['_bench']['fallback_reason'], 'transport')
        self.assertFalse(list(fixture.app.slot_path.glob('*-delta-backup.bin')))

    # Unconfirmed restore keeps the recovery file, detaches the slot and makes the existing reuse gate reject its native cache.
    def test_lost_final_ack_restore_failure_retains_recovery(self):
        fixture = self.fixture
        sid, _ = fixture.chat([10, 11, 12, 13])
        session = fixture.app.store.get(sid)
        owner = fixture.config['clients']['fixture-a']
        dispatch = fixture.app.decode.request
        def fail_restore(method, path, data=None, headers=None, **kwargs):
            if 'action=restore' in path:
                raise HTTPError(503)
            result = dispatch(method, path, data, headers, **kwargs)
            if 'action=import_delta' in path:
                raise OSError('lost final acknowledgement')
            return result
        with mock.patch.object(fixture.app.decode, 'request', side_effect=fail_restore):
            with self.assertRaises(HTTPError) as error:
                fixture.app.prepare_session(owner, {}, session, session['tokens'] + [20, 21, 22])
        self.assertEqual(error.exception.status, 503)
        self.assertIsNone(fixture.app.store.get(sid)['slot'])
        self.assertFalse(fixture.app.verified(sid))
        self.assertTrue((fixture.app.slot_path / (sid + '-delta-backup.bin')).is_file())
        fixture.app.acquire_slot(session)
        self.assertEqual(dispatch('GET', f"/slots/{session['slot']}/info")['tokens'], [])

    # An old node is refused before session creation or delta production; the original slot remains usable.
    def test_mixed_node_version_refuses_before_extend(self):
        fixture = self.fixture
        dispatch = fixture.prefill.app.handle
        def old_health(method, path, *args):
            status, value = dispatch(method, path, *args)
            if path == '/v1/health':
                value.pop('delta_version', None)
            return status, value
        fixture.app.extend_chunk_tokens = 2
        with mock.patch.object(fixture.prefill.app, 'handle', side_effect=old_health):
            with self.assertRaises(HTTPError) as error:
                fixture.chat([1, 2, 3, 4])
        self.assertEqual(error.exception.status, 412)
        self.assertIn(b'protocol 2', error.exception.body)
        self.assertEqual(fixture.decode.app.imports, 0)
        self.assertEqual(fixture.prefill.app.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 0)

    # A version-1 wire is refused even if node health advertises version 2; the valid wire remains accepted.
    def test_mixed_wire_version_refuses_before_import(self):
        import struct
        from stub_server import encode_delta, MODEL
        fixture = self.fixture
        valid = encode_delta(MODEL, 'version-control', [], [1])
        self.assertEqual(fixture.app.client_delta(valid), valid)
        size, = struct.unpack('<I', valid[:4])
        header = json.loads(valid[4:4 + size])
        header['version'] = 1
        encoded = json.dumps(header).encode()
        old = struct.pack('<I', len(encoded)) + encoded + valid[4 + size:]
        with self.assertRaises(HTTPError) as error:
            fixture.app.client_delta(old)
        self.assertEqual(error.exception.status, 412)
        self.assertIn('protocol 2', str(error.exception))
        self.assertEqual(fixture.decode.app.imports, 0)

    # Chunks commit only at the end; a middle rejection preserves the pre-transfer slot before local fallback.
    def test_chunks_and_middle_failure(self):
        fixture = self.fixture
        fixture.app.extend_chunk_tokens = 2
        sid, _ = fixture.chat(list(range(8)))
        session = fixture.app.store.get(sid)
        self.assertEqual(fixture.decode.app.imports, 4)
        self.assertEqual(fixture.app.decode.request('GET', f"/slots/{session['slot']}/info")['tokens'][:8], list(range(8)))
        fixture.app.benchmark = True
        before_imports = fixture.decode.app.imports
        dispatch = fixture.prefill.app.handle
        calls = []
        def fail_middle(method, path, *args, **kwargs):
            body = args[1] if args else kwargs.get('body')
            if path.endswith('/extend') and isinstance(body, dict) and body.get('return_delta'):
                calls.append(body['n0'])
                if len(calls) == 2:
                    return 503, {'error': 'ram budget', 'need': 100, 'budget': 50}
            return dispatch(method, path, *args, **kwargs)
        with mock.patch.object(fixture.prefill.app, 'handle', side_effect=fail_middle):
            result = fixture.client.request('POST', '/v1/chat/completions',
                dict(messages=[], _tokens=session['tokens'] + list(range(20, 28))),
                {'X-Hybrid-Session': session['id'], 'X-Hybrid-Bench-Mode': 'offload'})
            with fixture.app.reserve(session['id']):
                pass
            second = fixture.app.store.get(session['id'])
            self.assertEqual(result['hybrid_bench']['fallback_reason'], 'ram_budget')
            self.assertEqual([c['status'] for c in result['hybrid_bench']['chunks']], ['staged', 'requested'])
        self.assertEqual(len(calls), 2)
        self.assertGreater(fixture.decode.app.imports, before_imports)
        fixture.app.replay()
        current = fixture.app.store.get(second['id'])
        remote = fixture.app.remote(dict(id='a', prefill_token='remote-a')).request('GET', '/v1/sessions/' + current['remote_id'])
        self.assertEqual(remote['n'], len(current['tokens']))
        self.assertEqual(remote['prefix_hash'], prefix_hash(current['tokens']))

    # Default GET remains 501 and ordinary streaming emits exactly one DONE with no benchmark payload.
    def test_default_get_and_stream(self):
        fixture = self.fixture
        with self.assertRaises(HTTPError) as error:
            fixture.client.request('GET', '/health')
        self.assertEqual(error.exception.status, 501)
        with fixture.client.open('POST', '/v1/chat/completions', dict(messages=[], _tokens=[1, 2, 3], stream=True)) as response:
            wire = response.read()
        self.assertEqual(wire.count(b'data: [DONE]'), 1)
        self.assertNotIn(b'hybrid_bench', wire)

    # A local turn interrupted before finish stays local across replay; ending a mode frees only its owned remote session.
    def test_interrupted_local_replay_and_release(self):
        fixture = self.fixture
        fixture.app.benchmark = True
        client = dict(id='a', prefill_token='remote-a', benchmark_mode='local')
        with fixture.app.prepare(client, dict(messages=[], _tokens=[1, 2, 3]), 'interrupted'):
            pass
        with mock.patch.object(fixture.app, 'remote', side_effect=AssertionError('local replay contacted node')):
            fixture.app.replay()
        sid, _ = fixture.chat([4, 5, 6])
        session = fixture.app.store.get(sid)
        self.assertTrue(session['remote_id'])
        foreign = Node(fixture.client.url, 'fixture-b')
        with self.assertRaises(HTTPError) as error:
            foreign.request('DELETE', '/v1/sessions/' + sid)
        self.assertEqual(error.exception.status, 404)
        self.assertTrue(fixture.app.remote(client).request('GET', '/v1/sessions/' + session['remote_id']))
        fixture.client.request('DELETE', '/v1/sessions/' + session['id'])
        with self.assertRaises(HTTPError) as error:
            fixture.app.remote(client).request('GET', '/v1/sessions/' + session['remote_id'])
        self.assertEqual(error.exception.status, 404)
        # Durable local-mode meters retain their replay exclusion after the schema upgrade.
        legacy = fixture.app.store.get('interrupted')
        legacy['journal'] = dict(last_meter=dict(benchmark_mode='local'))
        fixture.app.store.save(legacy)
        with fixture.app.store.lock, fixture.app.store.db:
            fixture.app.store.db.execute("UPDATE sessions SET benchmark_mode=NULL WHERE id=?", ('interrupted',))
        with mock.patch.object(fixture.app, 'remote', side_effect=AssertionError('closed or legacy-local replay contacted node')):
            fixture.app.replay()

    # Node transport release retries are bounded; HTTP refusals are terminal, and successful recovery emits no failure body.
    def test_release_retry_and_leftovers(self):
        fixture = self.fixture
        fixture.app.benchmark = True
        for fault, expected_attempts in (('transient', 2), ('permanent', 3), ('unauthorized', 1), ('refused', 1)):
            with self.subTest(fault=fault):
                sid, _ = fixture.chat([1, 2, 3, len(fixture.app.store.pending())])
                session = fixture.app.store.get(sid)
                remote = fixture.app.remote(fixture.config['clients']['fixture-a'])
                original = remote.request
                calls = []
                def request(method, path, *args, **kwargs):
                    if method == 'DELETE':
                        calls.append(path)
                        if fault == 'permanent' or fault == 'transient' and len(calls) == 1:
                            raise urllib.error.URLError('disconnected remote-a fixture-a http://user:secret@node')
                        if fault in ('unauthorized', 'refused'):
                            raise HTTPError(401 if fault == 'unauthorized' else 503)
                    return original(method, path, *args, **kwargs)
                with mock.patch.object(fixture.app, 'remote', return_value=remote), \
                        mock.patch.object(remote, 'request', side_effect=request), mock.patch('release.time.sleep') as sleep:
                    if fault == 'transient':
                        self.assertEqual(fixture.client.request('DELETE', '/v1/sessions/' + sid), dict(status='released'))
                        self.assertEqual(sleep.call_args_list, [mock.call(.5)])
                    else:
                        with self.assertRaises(HTTPError) as raised:
                            fixture.client.request('DELETE', '/v1/sessions/' + sid)
                        body = json.loads(raised.exception.body)
                        self.assertEqual(raised.exception.status, 401 if fault == 'unauthorized' else 503)
                        self.assertEqual(body['node_session_ids'], [session['remote_id']])
                        self.assertEqual(body['exception_type'], 'URLError' if fault == 'permanent' else 'HTTPError')
                        for secret in ('remote-a', 'fixture-a', 'user:secret'):
                            self.assertNotIn(secret, body['message'])
                        self.assertEqual(sleep.call_count, 2 if fault == 'permanent' else 0)
                self.assertEqual(len(calls), expected_attempts)
                self.assertEqual(fixture.app.store.get(sid)['benchmark_mode'], 'closed')
                if fault != 'transient':
                    remote.request('DELETE', '/v1/sessions/' + session['remote_id'])

    # Different model bytes require explicit matching layout admission; mismatches never reach native import.
    def test_heterogeneous_identity_bridge(self):
        fixture = self.fixture
        config = dict(fixture.config, node_model_sha256='2' * 64)
        with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
            Proxy(config, ':memory:')
        config.update(hetero_quant=True, layout_fingerprint='1' * 64, node_layout_fingerprint='3' * 64)
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            Proxy(config, ':memory:')
        fixture.prefill.app.model = '2' * 64
        fixture.app.node_model = '2' * 64
        fixture.app.hetero_quant = True
        fixture.app.layout_fingerprint = '1' * 64
        sid, _ = fixture.chat([1, 2, 3, 4])
        self.assertEqual(fixture.decode.app.imports, 1)
        current = fixture.app.store.get(sid)
        self.assertEqual(current['tokens'][:4], [1, 2, 3, 4])
        before = list(fixture.decode.app.slots[current['slot']])
        fixture.prefill.app.layout_fingerprint = '3' * 64
        with self.assertRaises(HTTPError) as error:
            fixture.chat(current['tokens'] + [5, 6], sid=sid)
        self.assertEqual(error.exception.status, 412)
        self.assertEqual(fixture.decode.app.slots[current['slot']], before)


if __name__ == '__main__':
    unittest.main()
