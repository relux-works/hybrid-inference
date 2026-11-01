"""Opt-in real-node budget classification and proxy LRU replay; no builds or downloads."""
import argparse
import contextlib
import copy
import http.server
import hashlib
import json
import math
import os
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
from policy import extend_chunk_limit
from store import prefix_hash
from stub_server import xxh128
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


def test_prefill_timing(args, decode, slots):
    # Real forced-offload checks serialize finite noise metrics and CPU rules with and without the link endpoint, with no fallback or leaked session.
    for enabled in (False, True):
        name = 'timing-link' if enabled else 'timing-no-link'
        with PrefillNode(args, name, ('--bench',) if enabled else (), context='2048') as prefill:
            root = Path(args.artifacts) / (name + '-proxy')
            root.mkdir()
            with proxy_server(args, root, prefill, decode, slots) as (app, client, config):
                identity = model_identity(args.model)
                info = dict(model_sha256=config['model_sha256'], context=2048, backend='cpu', bench_link=enabled, **identity)
                private_json(root / 'proxy-config.json', config)
                private_json(root / 'client-info.json', dict(info, proxy_url=client.url, model_path=args.model,
                    build_dir=str(Path(args.checker_binary).resolve().parents[1])))
                private_json(root / 'node-info.json', info)
                command = [sys.executable, str(ROOT / 'tools/hybrid-bench/hybrid_bench.py'), '--client-state', str(root),
                    '--scratch', str(root), '--output', str(root), '--label', name, '--smoke', '--depths', '128', '--modes', 'offload', '--check', '--check-tokens', '8']
                with (root / 'benchmark.log').open('wb') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=log)
                    (root / 'benchmark.pid').write_text(str(process.pid))
                    try:
                        code = process.wait(timeout=180)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=15)
                rows = [json.loads(line) for line in next(root.glob('hybrid-bench-*.jsonl')).read_text().splitlines()]
                (root / 'result.json').write_text(json.dumps(dict(command=command, exit_code=code, rows=rows), indent=2))
                assert code == 0, (name, code, rows[-1])
                calls = [row for row in rows if row.get('stage') == 'call']
                assert len(calls) == 4 and all(row['path'] == 'offload' and row['server_prefill_ms'] is not None for row in calls)
                link = next(row for row in rows if row.get('stage') == 'link')
                assert link['status'] == ('passed' if enabled else 'skipped')
                assert rows[-1]['status'] == 'passed' and rows[-1]['aborted'] == rows[-1]['skipped'] == 0
                checks = [row for row in rows if row.get('stage') == 'equivalence']
                assert len(checks) == 2 and {row['check_mode'] for row in checks} == {'fresh', 'append_chain'}
                for row in checks:
                    assert row['same_partition_exact'] and row['status_rule'] == 'same_backend_same_partition_exact'
                    assert len(row['noise_partitions']) == 2 and math.isfinite(row['noise_kl_nats'])
                markdown = next(root.glob('hybrid-bench-*.md')).read_text()
                assert 'rule: same_backend_same_partition_exact' in markdown and '| Local chunks 128 / whole prompt |' in markdown
                release = next(row for row in rows if row.get('stage') == 'release')
                assert release['node_sessions'] == release['node_ram_bytes'] == 0
                print(f'PASS {name}: benchmark exit0, four timed offloads, link={link["status"]}, zero retained sessions', flush=True)


def test_native_chunk_slope(args):
    # Two actual delta bodies establish the 0.8B affine size; extrapolation admits the reserved maximum and exposes the old cap overflow.
    with PrefillNode(args, 'chunk-slope', context='2048') as prefill:
        session = prefill.create()
        sizes = []
        previous = []
        for count in (256, 1024):
            wire = prefill.extend(session, [100] * count, previous, delta=True)
            header_size, = struct.unpack_from('<I', wire)
            sizes.append(len(wire) - header_size - 52)
            previous += [100] * count
        slope = (sizes[1] - sizes[0]) // (1024 - 256)
        assert sizes[1] - sizes[0] == slope * (1024 - 256) and slope == 6552
        fixed = sizes[0] - slope * 256
        plan = context_plan(args.model, host_info(), 2048, True)
        state = plan['recurrent_state_bytes']
        chunk = extend_chunk_limit(plan['kv_bytes_per_token'], state)
        cap, reserve = 512 * 1024**2, 1024**2
        assert fixed - state < reserve and fixed + slope * chunk <= cap
        assert state + reserve + slope * chunk <= cap < state + reserve + slope * (chunk + 1)
        old = (cap - state - reserve) // plan['kv_bytes_per_token']
        assert fixed + slope * old > cap
        result = dict(sizes=sizes, slope=slope, fixed=fixed, planned_chunk=chunk, planned_body=fixed + slope * chunk,
                      old_chunk=old, old_body=fixed + slope * old, cap=cap, reserve=reserve,
                      limit='maximum extrapolated from real short deltas; no 79k-token prefill')
        (Path(args.artifacts) / 'chunk-slope/result.json').write_text(json.dumps(result, indent=2))
        print('PASS native chunk slope: ' + json.dumps(result), flush=True)


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


def test_equivalence_noise(args):
    # Real CPU deltas retain same-partition equality; distinct local partitions yield finite, nonzero noise at identical inputs/depth.
    root = Path(args.artifacts) / 'equivalence'
    root.mkdir()
    with PrefillNode(args, 'noise-node', context='2048') as prefill:
        tokens = [100 + i % 31 for i in range(132)]
        session = prefill.create()
        paths = []
        for n0, n1 in ((0, 48), (48, 96), (96, 131)):
            path = root / f'{n0}.delta'
            path.write_bytes(prefill.extend(session, tokens[n0:n1], tokens[:n0], delta=True))
            paths.append(str(path))
        prefill.request('DELETE', session, expected=204)
        assert prefill.request('GET', '/v1/health')['sessions'] == 0
        fixture = dict(tokens=tokens, model_sha256=sha256(args.model), k=8, deltas=paths, max_ctx=2048)
        token_path = root / 'tokens.json'
        private_json(token_path, fixture)

        def run(name, binary, data=fixture, gpu_layers=0):
            token_path.write_text(json.dumps(data))
            result_path = root / (name + '.json')
            command = [str(Path(binary).resolve()), args.model, str(token_path), paths[0], str(result_path), str(gpu_layers)]
            env = dict(os.environ, LD_LIBRARY_PATH=str(Path(args.checker_binary).resolve().parent))
            with (root / (name + '.log')).open('wb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
                (root / (name + '.pid')).write_text(str(process.pid))
                try:
                    code = process.wait(timeout=240)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=15)
            (root / (name + '.command.json')).write_text(json.dumps(dict(command=command, exit_code=code)))
            return code, json.loads(result_path.read_text()) if result_path.exists() else None

        code, result = run('noise', args.checker_binary)
        assert code == 0 and result['same_partition_exact'] and result['same_partition_agreed'] == 8
        assert result['status_rule'] == 'same_backend_same_partition_exact'
        assert result['status'] == 'passed'
        assert result['same_partition_kl_nats'] == result['same_partition_max_abs_logit_diff'] == 0
        assert result['same_partition_repeat_exact'] and result['same_partition_within_floor']
        assert result['same_partition_repeat_kl_nats'] == result['same_partition_repeat_max_abs_logit_diff'] == 0
        assert result['same_partition_repeat_seconds'] > 0 and result['noise_reference_seconds'] > 0
        assert result['noise_reference_variant'] == 'alternate_state_snapshot'
        assert all(row['exact'] and row['repeat_exact'] for row in result['same_partition_positions'])
        assert code == (0 if result['numerical_within_bounds'] else 3)
        assert len(result['noise_partitions']) == 2
        for row in result['noise_partitions']:
            assert math.isfinite(row['noise_kl_nats']) and row['noise_kl_nats'] >= 0
            assert math.isfinite(row['noise_top1_agreement']) and 0 <= row['noise_top1_agreement'] <= 1
            assert -1 <= row['noise_free_running_first_divergence'] < 8
        assert result['noise_kl_nats'] == max(r['noise_kl_nats'] for r in result['noise_partitions']) > 0
        assert math.isfinite(result['kl_to_noise_ratio'])
        # Both final prefill boundaries differ from hybrid holdback 131, including the reduced 255 request.
        assert [r['chunk_tokens'] for r in result['noise_partitions']] == [128, 130]
        assert all(r['last_prefill_boundary'] != result['hybrid_last_prefill_boundary'] for r in result['noise_partitions'])
        print('PASS CPU local noise: ' + json.dumps(result), flush=True)

        # The original CPU fields and exit code remain bit-for-bit identical to the unchanged M1 executable.
        if args.baseline_checker:
            baseline_code, baseline = run('baseline', args.baseline_checker)
            assert code == baseline_code
            assert all(result[key] == value for key, value in baseline.items()), (baseline, result)
            print('PASS CPU legacy fields/exit: identical to baseline executable', flush=True)

        # A byte flip in an imported body fails the intended checksum gate and emits no successful metrics.
        bad = root / 'corrupt.delta'
        wire = bytearray(Path(paths[0]).read_bytes())
        header_size, = struct.unpack_from('<I', wire)
        wire[4 + header_size + 1024] ^= 1
        bad.write_bytes(wire)
        bad_fixture = dict(fixture, deltas=[str(bad), *paths[1:]])
        bad_code, bad_result = run('corrupt', args.checker_binary, bad_fixture)
        assert bad_code == 2 and bad_result is None
        assert 'delta checksum mismatch' in (root / 'corrupt.log').read_text()
        print('PASS corrupt imported delta: exit 2, delta checksum mismatch, no metrics', flush=True)

        # Checksum-valid bodies reach numerical evaluation: severe corruption and argmax-preserving loss both fail the logit gate.
        original = Path(paths[0]).read_bytes()
        body_start = 4 + struct.unpack_from('<I', original)[0]
        original_body = original[body_start:-48]
        assert original[-48:] == xxh128(original_body) + hashlib.sha256(original_body).digest()
        for name, stride, mask in (('valid-corrupt', 1031, 0x40), ('valid-lossy', 4099, 0x01)):
            body = bytearray(original_body)
            for offset in range(4096, len(body), stride):
                body[offset] ^= mask
            body = bytes(body)
            path = root / (name + '.delta')
            path.write_bytes(original[:body_start] + body + xxh128(body) + hashlib.sha256(body).digest())
            bad_code, bad_result = run(name, args.checker_binary, dict(fixture, deltas=[str(path), *paths[1:]]))
            assert bad_code == 3 and bad_result['status'] == 'failed', (bad_code, bad_result)
            assert not bad_result['same_partition_exact'] and not bad_result['same_partition_within_floor']
            assert bad_result['same_partition_max_abs_logit_diff'] > bad_result['same_partition_repeat_max_abs_logit_diff']
            assert bad_result['same_partition_kl_nats'] > bad_result['same_partition_repeat_kl_nats']
            if name == 'valid-lossy':
                assert bad_result['same_partition_agreed'] == 8, bad_result
            print(f'PASS {name}: checksum-valid, exit 3 via same-partition logits; agreed={bad_result["same_partition_agreed"]}/8', flush=True)

        # A CPU-only build refuses requested GPU rather than selecting a weaker numerical rule; no GPU execution occurs.
        mismatch_code, mismatch_result = run('label-mismatch', args.checker_binary, dict(fixture, backend='cuda'))
        assert mismatch_code == 2 and mismatch_result is None
        assert 'backend label/device mismatch' in (root / 'label-mismatch.log').read_text()
        if not args.gpu_available:
            gpu_code, gpu_result = run('gpu-unavailable', args.checker_binary, gpu_layers=1)
            assert gpu_code == 2 and gpu_result is None
            assert 'no GPU offload device' in (root / 'gpu-unavailable.log').read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefill-binary', required=True)
    parser.add_argument('--server-binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--case', choices=('all', 'timing', 'chunk', 'noise'), default='all')
    parser.add_argument('--checker-binary', default=str(ROOT / 'build/bin/llama-hybrid-bench-check'))
    parser.add_argument('--gpu-available', action='store_true', help='skip CPU-only GPU-request refusal control')
    parser.add_argument('--baseline-checker', help='optional unchanged M1 executable for bit-for-bit CPU regression')
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve())
    args.binary = str(Path(args.prefill_binary).resolve())
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.artifacts = str(root)
    if args.case == 'noise':
        test_equivalence_noise(args)
        return
    if args.case == 'chunk':
        test_native_chunk_slope(args)
        return
    if args.case == 'all':
        test_equivalence_noise(args)
        test_native_defaults(args)
        test_native_chunk_slope(args)
    with native_server(args.server_binary, args.model, root / 'decoder') as (decode, slots):
        test_prefill_timing(args, decode, slots)
        if args.case == 'all':
            test_eviction_replay(args, decode, slots)
            test_budget_benchmark(args, decode, slots)


if __name__ == '__main__':
    main()
