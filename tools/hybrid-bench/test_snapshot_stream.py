"""CPU disk/HTTP snapshot regression above the delta limit; no model or GPU."""
import hashlib
import http.server
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from growing_session import Events, MeasuredNode
from proxy import MAX_DELTA, Node


class SnapshotTests(unittest.TestCase):
    # A full snapshot above the old delta cap is captured byte-for-byte with bounded reads, not buffered as a delta.
    def test_snapshot_above_delta_limit(self):
        size = MAX_DELTA + 65536
        block = b'12345678' + (4).to_bytes(4, 'little') + b'x' * (65536 - 12)
        digest = hashlib.sha256()
        for offset in range(0, size, len(block)):
            digest.update(block[:min(len(block), size - offset)])
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                try:
                    for offset in range(0, size, len(block)):
                        payload = block[:min(len(block), size - offset)]
                        self.wfile.write(f'{len(payload):x}\r\n'.encode() + payload + b'\r\n')
                    self.wfile.write(b'0\r\n\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertGreater(shutil.disk_usage(root).free, 28e9 + size)
            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                events = Events(root)
                node = MeasuredNode(f'http://127.0.0.1:{server.server_port}', timeout=30,
                                    events=events, session='h', source='critical')
                def delta_response(*args, **kwargs):
                    node.last_bytes, node.last_transfer_seconds, node.last_transfer_rate = 5, 2, 2.5
                    return b'delta'
                with mock.patch.object(Node, 'request', side_effect=delta_response):
                    self.assertEqual(node.request('POST', '/v1/sessions/s/extend',
                        dict(n0=0, tokens=[1, 2, 3, 4], prefix_hash_n0='h', return_delta=True), binary=True), b'delta')
                self.assertEqual((node.last_bytes, node.last_transfer_seconds, node.last_transfer_rate), (5, 2, 2.5))
                capture = events.copy()[-1]['node_snapshot']
                snapshot = Path(capture['path'])
                self.assertEqual(snapshot.stat().st_size, size)
                self.assertEqual(capture['sha256'], digest.hexdigest())
                streamed = events.copy()[0]
                self.assertEqual(streamed['response_body_bytes'], size)
                self.assertEqual(streamed['status'], 'ok')
                self.assertGreater(size, MAX_DELTA)
                snapshot.unlink()
                Path(events.copy()[-1]['node_delta']['path']).unlink()
                # The unchanged binary delta reader still rejects the same body at its own cap.
                with mock.patch('proxy.MAX_DELTA', 65536):
                    with self.assertRaisesRegex(ValueError, 'upstream body too large'):
                        Node(node.url, timeout=30).request('GET', '/snapshot', binary=True)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
