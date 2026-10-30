"""Synthetic ordering/lease fault controls; native overlap is a separate opt-in test."""
import concurrent.futures
import json
import socketserver
import sqlite3
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

import test_proxy as fixtures
from proxy import HTTPError, Node, SLOT_WAIT_SECONDS
from store import prefix_hash


class ConcurrencyTests(unittest.TestCase):
    start = fixtures.HTTPTests.start
    url = staticmethod(fixtures.HTTPTests.url)
    chat = fixtures.HTTPTests.chat

    def setUp(self):
        fixtures.HTTPTests.setUp(self)
        self.client.timeout = 60
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.blocks = {}
        self.active = set()
        self.busy = set()
        self.mutations = []
        self.drained = []
        self.guard = threading.Lock()
        opened = self.app.decode.open
        test = self

        class Stream:
            def __init__(self, response, data):
                self.response, self.data = response, data

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *args):
                return self.response.__exit__(*args)

            def __iter__(self):
                key = tuple(self.data['_tokens'])
                slot = self.data['id_slot']
                with test.guard:
                    test.assertNotIn(slot, test.active)
                    test.active.add(slot)
                try:
                    if key in test.blocks:
                        entered, release = test.blocks[key]
                        entered.set()
                        test.assertTrue(release.wait(60), 'stream control was not released')
                    yield from self.response
                    test.drained.append(key)
                finally:
                    with test.guard:
                        test.active.remove(slot)

        def capture(method, path, data=None, headers=None):
            if path.startswith('/slots/') and '?action=' in path:
                slot = int(path.split('/')[2].split('?')[0])
                with self.guard:
                    self.mutations.append((slot, path))
                    self.assertNotIn(slot, self.active, 'a pinned streaming slot was mutated')
                    self.assertNotIn(slot, self.busy, 'a processing slot was mutated')
            response = opened(method, path, data, headers)
            return Stream(response, data) if path == '/v1/chat/completions' and data.get('stream') else response
        self.app.decode.open = capture

    def tearDown(self):
        for _, release in self.blocks.values():
            release.set()
        self.pool.shutdown(wait=True)
        fixtures.HTTPTests.tearDown(self)

    def block(self, tokens):
        entered, release = threading.Event(), threading.Event()
        self.blocks[tuple(tokens)] = (entered, release)
        return entered, release

    def launch(self, tokens, sid):
        return self.pool.submit(self.chat, tokens, sid=sid, stream=True)

    def queued(self, sid, count):
        with self.app.lock:
            queue = self.app.queues[sid]
        with queue.condition:
            self.assertTrue(queue.condition.wait_for(lambda: len(queue.waiters) == count, timeout=5))

    # Independent live streams both progress; neither holds the shared store/slot lock across HTTP.
    def test_independent_streams_and_short_shared_lock(self):
        a, release_a = self.block([10])
        b, release_b = self.block([20])
        first = self.launch([10], 'a')
        self.assertTrue(a.wait(5))
        second = self.launch([20], 'b')
        self.assertTrue(b.wait(5), 'another session was serialized')
        def inspect():
            with self.app.lock:
                return len(self.app.store.pending()), len(self.app.pinned)
        self.assertEqual(self.pool.submit(inspect).result(timeout=2), (2, 2))
        release_a.set()
        release_b.set()
        first.result(timeout=5)
        second.result(timeout=5)

    # Three queued turns must run FIFO, with fresh post-predecessor validation and exact journal history.
    def test_same_session_fifo(self):
        tokens = [[10], [10, 901, 902, 11], [10, 901, 902, 11, 901, 902, 12]]
        gates = [self.block(t) for t in tokens]
        first = self.launch(tokens[0], 'fifo')
        self.assertTrue(gates[0][0].wait(5))
        second = self.launch(tokens[1], 'fifo')
        self.queued('fifo', 2)
        third = self.launch(tokens[2], 'fifo')
        self.queued('fifo', 3)
        self.assertFalse(gates[1][0].is_set())
        gates[0][1].set()
        self.assertTrue(gates[1][0].wait(5))
        self.assertFalse(gates[2][0].is_set())
        gates[1][1].set()
        self.assertTrue(gates[2][0].wait(5))
        gates[2][1].set()
        for future in [first, second, third]:
            future.result(timeout=5)
        self.assertEqual(self.app.store.get('fifo')['tokens'], tokens[-1] + [901, 902])

    # A stale second explicit history rejects after the first completes, without overwriting its journal.
    def test_same_session_stale_prompt_rejected(self):
        entered, release = self.block([10])
        first = self.launch([10], 'fifo')
        self.assertTrue(entered.wait(5))
        stale = self.launch([10, 11], 'fifo')
        self.queued('fifo', 2)
        release.set()
        first.result(timeout=5)
        with self.assertRaises(HTTPError) as error:
            stale.result(timeout=5)
        self.assertEqual(error.exception.status, 400)
        self.assertEqual(self.app.store.get('fifo')['tokens'], [10, 901, 902])

    # Inferred requests serialize once identified; they reload the predecessor's completed journal before continuing.
    def test_inferred_session_continues_after_queued_predecessor(self):
        sid, _ = self.chat([10])
        first_tokens = [10, 901, 902, 11]
        second_tokens = first_tokens + [901, 902, 12]
        first_gate, release_first = self.block(first_tokens)
        second_gate, release_second = self.block(second_tokens)
        first = self.launch(first_tokens, None)
        self.assertTrue(first_gate.wait(5))
        second = self.launch(second_tokens, None)
        self.queued(sid, 2)
        self.assertFalse(second_gate.is_set())
        release_first.set()
        self.assertTrue(second_gate.wait(5))
        release_second.set()
        self.assertEqual(first.result(timeout=5)[0], sid)
        self.assertEqual(second.result(timeout=5)[0], sid)
        self.assertEqual(self.app.store.get(sid)['tokens'], second_tokens + [901, 902])

    # A queued inferred history that diverges from the completed predecessor forks instead of overwriting it.
    def test_inferred_queued_fork_reselects_after_predecessor(self):
        sid, _ = self.chat([10])
        tokens = [10, 901, 902, 11]
        entered, release = self.block(tokens)
        first = self.launch(tokens, None)
        self.assertTrue(entered.wait(5))
        forked = self.launch(tokens + [99], None)
        self.queued(sid, 2)
        release.set()
        self.assertEqual(first.result(timeout=5)[0], sid)
        other = forked.result(timeout=5)[0]
        self.assertNotEqual(other, sid)
        self.assertEqual(self.app.store.get(sid)['tokens'], tokens + [901, 902])
        self.assertEqual(self.app.store.get(other)['tokens'], tokens + [99, 901, 902])

    # A faster later tokenizer cannot overtake an earlier turn of the inferred session; another client still runs.
    def test_inferred_fifo_with_out_of_order_tokenization(self):
        sid, _ = self.chat([10])
        first_tokens = [10, 901, 902, 11]
        second_tokens = first_tokens + [901, 902, 12]
        entered, release = threading.Event(), threading.Event()
        second_ready = threading.Event()
        tokenize = self.app.tokenize
        def controlled(body):
            if body['_tokens'] == first_tokens:
                entered.set()
                self.assertTrue(release.wait(10))
            result = tokenize(body)
            if body['_tokens'] == second_tokens:
                second_ready.set()
            return result
        first_gate, release_first = self.block(first_tokens)
        second_gate, release_second = self.block(second_tokens)
        self.app.tokenize = controlled
        first = self.launch(first_tokens, None)
        try:
            self.assertTrue(entered.wait(5))
            second = self.launch(second_tokens, None)
            self.assertTrue(second_ready.wait(5))
            with self.app.lock:
                queue = self.app.admissions['a']
            with queue.condition:
                self.assertEqual(len(queue.waiters), 2)
            self.assertFalse(second_gate.is_set())
            foreign = Node(self.client.url, 'fixture-b', timeout=60)
            self.chat([30], client=foreign, sid='other-client')
            release.set()
            self.assertTrue(first_gate.wait(5))
            self.assertFalse(second_gate.is_set())
            release_first.set()
            self.assertTrue(second_gate.wait(5))
            release_second.set()
            self.assertEqual(first.result(timeout=5)[0], sid)
            self.assertEqual(second.result(timeout=5)[0], sid)
            self.assertEqual(self.app.store.get(sid)['tokens'], second_tokens + [901, 902])
        finally:
            release.set()

    # Failed classification removes its admission ticket, so later valid work cannot remain blocked behind it.
    def test_failed_tokenizer_releases_admission(self):
        entered, release = threading.Event(), threading.Event()
        tokenize = self.app.tokenize
        def controlled(body):
            if body['_tokens'] == [10]:
                entered.set()
                self.assertTrue(release.wait(10))
                raise ValueError('injected invalid template')
            return tokenize(body)
        self.app.tokenize = controlled
        first = self.launch([10], None)
        try:
            self.assertTrue(entered.wait(5))
            second = self.launch([20], None)
            release.set()
            with self.assertRaises(HTTPError) as error:
                first.result(timeout=5)
            self.assertEqual(error.exception.status, 400)
            sid, _ = second.result(timeout=5)
            self.assertEqual(self.app.store.get(sid)['tokens'], [20, 901, 902])
        finally:
            release.set()

    # LRU eviction skips the pinned slot and preserves its tokens; an idle neighbor can be parked.
    def test_eviction_skips_pinned_slot(self):
        self.chat([20], sid='idle')
        entered, release = self.block([10])
        live = self.launch([10], 'live')
        self.assertTrue(entered.wait(5))
        pinned = self.app.store.get('live')['slot']
        before = self.app.decode.request('GET', f'/slots/{pinned}/info')
        self.chat([30], sid='replacement')
        self.assertIsNone(self.app.store.get('idle')['slot'])
        self.assertEqual(self.app.store.get('live')['slot'], pinned)
        after = self.app.decode.request('GET', f'/slots/{pinned}/info')
        for field in ('n', 'tokens', 'prefix_hash'):
            self.assertEqual(after[field], before[field])
        release.set()
        live.result(timeout=5)

    # All-pinned admission really waits 30 seconds then returns 503/Retry-After; freeing a slot admits a waiter.
    def test_all_pinned_timeout_and_release(self):
        self.assertEqual(SLOT_WAIT_SECONDS, 30)
        a, release_a = self.block([10])
        b, release_b = self.block([20])
        first, second = self.launch([10], 'a'), self.launch([20], 'b')
        self.assertTrue(a.wait(5) and b.wait(5))
        request = urllib.request.Request(self.client.url + '/v1/chat/completions',
            data=json.dumps(dict(messages=[], _tokens=[30])).encode(), method='POST',
            headers={'Authorization': 'Bearer fixture-a', 'Content-Type': 'application/json', 'X-Hybrid-Session': 'c'})
        began = time.monotonic()
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=40)
        self.assertEqual(error.exception.code, 503)
        self.assertEqual(error.exception.headers['Retry-After'], '1')
        self.assertIn(b'no decode slot available', error.exception.read())
        elapsed = time.monotonic() - began
        self.assertGreaterEqual(elapsed, 29.5)
        self.assertLess(elapsed, 40)
        print(f'OBSERVED all-pinned rejection elapsed_seconds={elapsed:.6f} status=503 retry_after=1', flush=True)
        waiting = self.launch([30], 'c')
        self.queued('c', 1)
        self.assertFalse(waiting.done())
        release_a.set()
        first.result(timeout=5)
        self.assertEqual(waiting.result(timeout=5)[0], 'c')
        self.assertFalse(second.done())
        release_b.set()
        second.result(timeout=5)

    # Six workers contend for two slots. Under the still-pinned finish hook, all native metadata agrees with the journal.
    def test_contention_bindings_and_exact_journals(self):
        checked = []
        finish = self.app.finish
        def audited(session, prompt):
            finish(session, prompt)
            info = self.app.decode.request('GET', f"/slots/{session['slot']}/info")
            self.assertEqual(info['tokens'], prompt + [901, 902])
            self.assertEqual(info['tokens'], self.app.store.get(session['id'])['tokens'])
            self.assertEqual(info['prefix_hash'], prefix_hash(info['tokens']))
            with self.app.lock:
                bindings = [s['slot'] for s in self.app.store.pending() if s['slot'] is not None]
                self.assertEqual(len(bindings), len(set(bindings)))
            checked.append(session['id'])
        self.app.finish = audited
        def conversation(index):
            tokens = [100 + index]
            for turn in range(4):
                tokens += [200 + turn]
                self.chat(tokens, sid=f'contend-{index}', stream=True)
                tokens += [901, 902]
            return f'contend-{index}', tokens
        futures = [self.pool.submit(conversation, i) for i in range(6)]
        expected = dict(f.result(timeout=60) for f in futures)
        self.assertEqual(len(checked), 24)
        self.app.replay()
        self.app.replay()
        for sid, tokens in expected.items():
            session = self.app.store.get(sid)
            self.assertEqual(session['tokens'], tokens)
            self.assertEqual(session['remote_n'], len(tokens))
            remote = self.app.remote(self.app.client('Bearer fixture-a')).request('GET', '/v1/sessions/' + session['remote_id'])
            self.assertEqual(remote['n'], len(tokens))
            self.assertEqual(remote['prefix_hash'], prefix_hash(tokens))

    # Replay's slow HTTP owns only its session; another live stream starts and busy sessions are skipped by replay.
    def test_replay_and_other_live_request_overlap(self):
        self.chat([10], sid='shadow')
        entered, release = threading.Event(), threading.Event()
        remote = self.app.remote
        def controlled(client):
            node = remote(client)
            request = node.request
            def blocked(*args, **kwargs):
                if threading.current_thread().name == 'controlled-replay':
                    entered.set()
                    self.assertTrue(release.wait(10))
                return request(*args, **kwargs)
            node.request = blocked
            return node
        self.app.remote = controlled
        errors = []
        def replay():
            try:
                self.app.replay()
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=replay, name='controlled-replay')
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            streaming, finish = self.block([20])
            live = self.launch([20], 'live')
            self.assertTrue(streaming.wait(5), 'replay blocked another live request')
            release.set()
            worker.join(timeout=5)

            self.assertFalse(worker.is_alive(), 'replay waited on the busy live session')
            self.assertFalse(errors)
            self.assertFalse(live.done())
            finish.set()
            live.result(timeout=5)
        finally:
            release.set()
            worker.join(timeout=5)

    # Eviction may park a replaying session, but a later journal save must not resurrect its old binding.
    def test_replay_does_not_resurrect_evicted_binding(self):
        self.app.slots = 1
        self.chat([10], sid='parked')
        entered, release = threading.Event(), threading.Event()
        remote = self.app.remote
        def controlled(client):
            node = remote(client)
            request = node.request
            def blocked(*args, **kwargs):
                if threading.current_thread().name == 'parked-replay':
                    entered.set()
                    self.assertTrue(release.wait(10))
                return request(*args, **kwargs)
            node.request = blocked
            return node
        self.app.remote = controlled
        errors = []
        def replay():
            try:
                self.app.replay()
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=replay, name='parked-replay')
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            self.chat([20], sid='replacement')
            self.assertIsNone(self.app.store.get('parked')['slot'])
            release.set()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors)
            self.assertIsNone(self.app.store.get('parked')['slot'])
            self.assertEqual(self.app.store.get('replacement')['slot'], 0)
            self.assertEqual(self.app.store.get('parked')['tokens'], [10, 901, 902])
        finally:
            release.set()
            worker.join(timeout=5)

    # Processing slots receive no erase/restore/import; a separate idle slot still completes a positive request.
    def test_native_busy_guard_does_not_block_other_slot(self):
        entered, release = threading.Event(), threading.Event()
        original = self.decode.app.handle
        with self.guard:
            self.busy.add(0)
        def states(method, path, query, body, client):
            if path == '/slots':
                entered.set()
                return 200, [dict(id=0, is_processing=not release.is_set()), dict(id=1, is_processing=False)]
            return original(method, path, query, body, client)
        with mock.patch.object(self.decode.app, 'handle', side_effect=states):
            waiting = self.launch([10], 'waiting')
            self.assertTrue(entered.wait(5))
            self.chat([20], sid='other')
            self.assertFalse(waiting.done())
            self.assertEqual(self.app.store.get('other')['slot'], 1)
            self.assertFalse(any(slot == 0 for slot, _ in self.mutations))
            with self.guard:
                self.busy.remove(0)
            release.set()
            waiting.result(timeout=5)
            self.assertEqual(self.app.store.get('waiting')['slot'], 0)

    # An unavailable idle probe leaves the old binding intact and quarantines the slot; a healthy probe admits reuse.
    def test_failed_idle_probe_quarantines_then_recovers(self):
        self.app.slots = 1
        self.chat([10], sid='old')
        before = self.app.store.get('old')
        mutations = len(self.mutations)
        original = self.decode.app.handle
        def unavailable(method, path, query, body, client):
            return (503, {}) if path == '/slots' else original(method, path, query, body, client)
        with mock.patch.object(self.decode.app, 'handle', side_effect=unavailable):
            with self.assertRaises(HTTPError) as error:
                self.chat([20], sid='rejected')
            self.assertEqual(error.exception.status, 503)
        self.assertEqual(self.app.quarantined, {0})
        self.assertEqual(self.app.store.get('old'), before)
        self.assertEqual(len(self.mutations), mutations)
        self.chat([30], sid='healthy')
        self.assertFalse(self.app.quarantined)
        self.assertIsNone(self.app.store.get('old')['slot'])
        self.assertEqual(self.app.store.get('healthy')['slot'], 0)

    # A downstream write failure still drains upstream and persists evaluated tokens before releasing the slot.
    def test_disconnected_reader_drains_and_commits_journal(self):
        write = socketserver._SocketWriter.write
        port = self.proxy.server_port
        failed = []
        def disconnected(writer, data):
            if writer._sock.getsockname()[1] == port and data.startswith(b'data: '):
                failed.append(True)
                raise BrokenPipeError('injected downstream disconnect')
            return write(writer, data)
        with mock.patch.object(socketserver._SocketWriter, 'write', new=disconnected):
            self.chat([10], sid='disconnected', stream=True)
        self.assertTrue(failed)
        self.assertIn((10,), self.drained)
        session = self.app.store.get('disconnected')
        info = self.app.decode.request('GET', f"/slots/{session['slot']}/info")
        self.assertEqual(session['tokens'], [10, 901, 902])
        self.assertEqual(session['tokens'], info['tokens'])
        self.assertEqual(info['prefix_hash'], prefix_hash(session['tokens']))
        self.assertFalse(self.app.pinned)
        self.assertFalse(self.app.quarantined)

    # SQLite rejects duplicate ownership independently of runtime leases, then permits an explicit transfer.
    def test_store_rejects_duplicate_slot_binding(self):
        first = self.app.store.select('a', [10], 'first')
        second = self.app.store.select('a', [20], 'second')
        self.app.store.set_slot(first['id'], 0)
        with self.assertRaises(sqlite3.IntegrityError):
            self.app.store.set_slot(second['id'], 0)
        self.assertEqual(self.app.store.occupant(0)['id'], 'first')
        self.assertIsNone(self.app.store.get('second')['slot'])
        self.app.store.set_slot(first['id'], None)
        self.app.store.set_slot(second['id'], 0)
        self.assertEqual(self.app.store.occupant(0)['id'], 'second')

    # Snapshot recovery keeps its slot pinned and its old journal until restore succeeds, while another session runs.
    def test_recovery_is_atomic_per_session_and_concurrent(self):
        self.chat([41, 42], sid='recovering')
        self.app.replay()
        self.app.replay()
        before = self.app.store.get('recovering')
        ahead = before['tokens'] + [55]
        with self.prefill.app.lock, self.prefill.app.db:
            self.prefill.app.db.execute('UPDATE sessions SET tokens=? WHERE id=?',
                                       (json.dumps(ahead), before['remote_id']))
        entered, release = threading.Event(), threading.Event()
        opened = self.app.decode.open
        def controlled(method, path, data=None, headers=None):
            if '?action=restore' in path and data['filename'] == 'recovering-resync.bin':
                entered.set()
                self.assertTrue(release.wait(10))
            return opened(method, path, data, headers)
        self.app.decode.open = controlled
        try:
            recovering = self.launch(ahead + [66], 'recovering')
            self.assertTrue(entered.wait(5))
            self.assertEqual(self.app.store.get('recovering')['tokens'], before['tokens'])
            slot = before['slot']
            self.assertEqual(self.app.decode.request('GET', f'/slots/{slot}/info')['tokens'], before['tokens'])
            self.chat([20], sid='other')
            self.assertFalse(recovering.done())
            self.assertEqual(self.app.store.get('recovering')['slot'], slot)
            release.set()
            recovering.result(timeout=5)
            self.assertEqual(self.app.store.get('recovering')['tokens'], ahead + [66, 901, 902])
            self.assertEqual(self.app.store.get('other')['tokens'], [20, 901, 902])
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
