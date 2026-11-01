"""Opt-in benchmark switches retain the production default."""
import json
from unittest import mock
import unittest

import test_proxy
from proxy import HTTPError, Node, Proxy
from store import prefix_hash


class BenchmarkProxyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_proxy.HTTPTests('runTest')
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

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

    # Three sequential imports preserve exact tokens and a middle rejection retains a usable verified prefix.
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
            self.assertEqual([c['status'] for c in result['hybrid_bench']['chunks']], ['imported', 'requested'])
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
