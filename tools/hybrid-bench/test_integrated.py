"""Opt-in real-node budget classification and proxy LRU replay; no builds or downloads."""
import argparse
import contextlib
import copy
import http.server
import json
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/hybrid'), str(ROOT / 'tools/hybrid-proxy')]
from bench_common import context_plan, host_info, model_identity, private_json, sha256
from native_fixture import native_server
from proxy import Handler, Node, Proxy
from store import prefix_hash
from test_limits import Node as PrefillNode


@contextlib.contextmanager
def proxy_server(args, root, prefill, decode, slots):
    credential = secrets.token_hex(24)
    plan = context_plan(args.model, host_info(), 2048, True)
    config = dict(benchmark=True, model_sha256=sha256(args.model), prefill_url=prefill.base,
        decode_url=decode.url, decode_token="", prefill_timeout=120, decode_timeout=120, slots=1, slot_path=str(slots),
        client_credential=credential, clients={credential: dict(id='a', prefill_token=prefill.clients['a'])},
        policy=dict(kv_bytes=plan['kv_bytes_per_token'], state_bytes=plan['recurrent_state_bytes']))
    app = Proxy(config, str(root / 'proxy.sqlite'))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.app = app
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Node(f'http://127.0.0.1:{server.server_port}', credential, timeout=180)
    try:
        yield app, client, config
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        app.store.close()


def test_native_defaults(args):
    # Without optional flags or positional context, the native allocation and all three documented limits stay unchanged.
    with PrefillNode(args, 'default-node', context=None) as prefill:
        health = prefill.health
        assert health['n_ctx'] == 8448 and health['max_ctx'] == 16384
        assert health['ram_budget_bytes'] == 512 * 1024**2 and health['max_request_bytes'] == 1024**2
        assert health['sessions'] == health['ram_bytes'] == health['evictions'] == 0
        (Path(args.artifacts) / 'default-node/result.json').write_text(json.dumps(health, indent=2))
        print('PASS native defaults: allocation8448, max_ctx16384, retained512MiB, request1MiB, empty store', flush=True)


def test_eviction_replay(args, decode, slots):
    # A real global LRU eviction returns 404; proxy replay restores the exact sampled journal and leaves decoder tokens unchanged.
    with PrefillNode(args, 'eviction-node', ('--ram-budget-mib', '48'), context='2048') as prefill:
        root = Path(args.artifacts) / 'eviction-proxy'
        root.mkdir()
        with proxy_server(args, root, prefill, decode, slots) as (app, client, config):
            body = dict(messages=[dict(role='user', content='Count from 1 through 20.')], max_tokens=4,
                temperature=0, seed=0, ignore_eos=True, chat_template_kwargs=dict(enable_thinking=False))
            client.request('POST', '/v1/chat/completions', body,
                {'X-Hybrid-Session': 'lru-replay', 'X-Hybrid-Bench-Mode': 'offload'})
            with app.reserve('lru-replay'):
                pass
            app.replay()
            app.replay()
            with app.reserve('lru-replay'):
                before = copy.deepcopy(app.store.get('lru-replay'))
            assert before['tokens'] and before['remote_n'] == len(before['tokens'])
            old = '/v1/sessions/' + before['remote_id']
            before_decoder = decode.request('GET', f"/slots/{before['slot']}/info")
            for owner, token in (('b', 100), ('c', 101)):
                session = prefill.create(owner)
                prefill.extend(session, [token], client=owner)
            prefill.request('GET', old, expected=404)
            assert prefill.request('GET', '/v1/health')['evictions'] == 1
            app.replay()
            app.replay()
            with app.reserve('lru-replay'):
                after = app.store.get('lru-replay')
            assert after['remote_id'] != before['remote_id'] and after['tokens'] == before['tokens']
            assert after['remote_n'] == len(before['tokens'])
            path = '/v1/sessions/' + after['remote_id']
            info = prefill.request('GET', path)
            snapshot = prefill.request('GET', path + '/snapshot')
            count, = struct.unpack_from('<I', snapshot, 8)
            assert list(struct.unpack_from('<' + str(count) + 'i', snapshot, 12)) == before['tokens']
            assert info['prefix_hash'] == prefix_hash(before['tokens'])
            assert decode.request('GET', f"/slots/{before['slot']}/info") == before_decoder
            assert prefill.request('GET', '/v1/health')['evictions'] == 2
            (root / 'result.json').write_text(json.dumps(dict(tokens=count, prefix_hash=info['prefix_hash'],
                evictions=2, decoder_unchanged=True, remote_snapshot_tokens_equal=True), indent=2))
            client.request('DELETE', '/v1/sessions/lru-replay')
            print(f'PASS native proxy LRU replay: {count} exact snapshot/journal tokens, decoder unchanged, two evictions', flush=True)


def test_budget_benchmark(args, decode, slots):
    # The actual native 503 has the exact budget fields; the real benchmark reports that refusal, releases sessions and exits 2.
    with PrefillNode(args, 'budget-node', ('--ram-budget-mib', '1'), context='2048') as prefill:
        session = prefill.create()
        refusal = prefill.extend(session, [100], expected=503)
        assert set(refusal) == {'error', 'need', 'budget'}
        assert refusal['error'] == 'ram budget' and refusal['need'] > refusal['budget'] == 1024**2
        assert prefill.request('GET', session)['n'] == 0
        prefill.request('DELETE', session, expected=204)
        root = Path(args.artifacts) / 'budget-proxy'
        root.mkdir()
        with proxy_server(args, root, prefill, decode, slots) as (app, client, config):
            identity = model_identity(args.model)
            info = dict(model_sha256=config['model_sha256'], context=2048, backend='cpu', bench_link=False, **identity)
            private_json(root / 'proxy-config.json', config)
            private_json(root / 'client-info.json', dict(info, proxy_url=client.url))
            private_json(root / 'node-info.json', info)
            command = [sys.executable, str(ROOT / 'tools/hybrid-bench/hybrid_bench.py'), '--client-state', str(root),
                '--scratch', str(root), '--output', str(root), '--label', 'native-budget', '--smoke', '--modes', 'offload']
            with (root / 'benchmark.log').open('wb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                (root / 'benchmark.pid').write_text(str(process.pid))
                try:
                    code = process.wait(timeout=180)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
                    raise
            assert code == 2, code
            rows = [json.loads(line) for line in next(root.glob('hybrid-bench-*.jsonl')).read_text().splitlines()]
            assert any(row.get('stage') == 'error' and row.get('reason') == 'ram budget refusal (HTTP 503)' for row in rows)
            assert rows[-1]['status'] == 'aborted' and rows[-1]['exit_code'] == 2
            release = next(row for row in rows if row.get('stage') == 'release')
            assert release['status'] == 'released' and release['node_sessions'] == release['node_ram_bytes'] == 0
            assert prefill.request('GET', '/v1/health')['sessions'] == 0
            (root / 'result.json').write_text(json.dumps(dict(command=command, exit_code=code, refusal=refusal,
                reason='ram budget refusal (HTTP 503)', release=release), indent=2))
            print(f'PASS native budget refusal: {refusal}; benchmark aborted 2 with budget reason and zero sessions', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefill-binary', required=True)
    parser.add_argument('--server-binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--artifacts', required=True)
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve())
    args.binary = str(Path(args.prefill_binary).resolve())
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.artifacts = str(root)
    test_native_defaults(args)
    with native_server(args.server_binary, args.model, root / 'decoder') as (decode, slots):
        test_eviction_replay(args, decode, slots)
        test_budget_benchmark(args, decode, slots)


if __name__ == '__main__':
    main()
