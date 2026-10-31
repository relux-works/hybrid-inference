"""Opt-in benchmark switches retain the production default."""
import json
from unittest import mock
import unittest

import test_proxy
from proxy import HTTPError


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


if __name__ == '__main__':
    unittest.main()
