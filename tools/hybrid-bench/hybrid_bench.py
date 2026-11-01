#!/usr/bin/env python3
"""Synthetic hybrid chat benchmark. Python 3.10+ standard library only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import urllib.error
import contextlib
import uuid

from bench_common import (Refusal, http, private_json, read_private, require, sha256, validate_identity, start_process)

# Measurement export v0.10, generated 2026-10-30, stats.json, agents.all.
# Source SHA-256 binds the copied aggregates; no original conversations are used or read at runtime.
PROVENANCE = dict(source='measurement export v0.10 / 2026-10-30 / stats.json / agents.all',
    sha256='8c5f1f2c44e344b78d32d0b4c216d0ebdf2ecdae3cde0f1a983cb329e488e2a7',
    probabilities=[.10, .25, .50, .75, .90, .99],
    tool_result_tokens_est=[17.5, 43.6, 165.8, 887.5, 3211.1, 11143.8],
    output_per_call=[54, 113, 229, 530, 1135, 4506],
    tool_sample_count=1275343, output_sample_count=1524671)


def quantile_sample(rng, values):
    p = rng.random()
    points = list(zip(PROVENANCE['probabilities'], values))
    if p <= points[0][0]:
        return max(1, round(points[0][1]))
    for (a, x), (b, y) in zip(points, points[1:]):
        if p <= b:
            return max(1, round(x + (y - x) * (p - a) / (b - a)))
    return max(1, round(points[-1][1]))


def schedule(seed, measured=False, smoke=False, depths=None):
    rng = random.Random(seed)
    require(depths is None or (depths and all(2 <= n <= 1048576 for n in depths) and
            all(a < b for a, b in zip(depths, depths[1:]))), 'depths must increase strictly within 2..1048576')
    ladder = depths if depths is not None else [128, 512] if smoke else [16384, 50000, 100000, 150000, 200000, 256000]
    for depth in ladder:
        yield dict(kind='depth', depth=depth, output=4 if smoke else 64)
        for fixed in ([16, 32, 64] if smoke else [512, 2048, 10000]):
            yield dict(kind='tool', depth=depth,
                tokens=quantile_sample(rng, PROVENANCE['tool_result_tokens_est']) if measured else fixed,
                output=min(8, quantile_sample(rng, PROVENANCE['output_per_call'])) if measured and smoke else
                       quantile_sample(rng, PROVENANCE['output_per_call']) if measured else 4 if smoke else 64)


def verify_meter(trace, previous, mode):
    require(trace['prefilled'] == trace['new_tokens'] and trace['replayed_prefix'] == 0 and trace['rollback_n'] is None,
            'full re-prefill detected: prompt work differs from new tokens')
    require(trace['depth_before'] == previous, 'full re-prefill detected: prior evaluated boundary was lost')
    require(trace['depth_after'] == trace['prompt_depth'] + trace['generated_tokens'], 'inconsistent generation boundary')
    require(trace['mode'] == mode, 'proxy ignored requested benchmark mode')
    require(mode != 'local' or trace['remote_applied'] == 0, 'local mode unexpectedly offloaded')
    if trace.get('fallback_reason') == 'ram_budget':
        raise Refusal('ram budget refusal (HTTP 503)')
    require(mode != 'offload' or trace['new_tokens'] <= 1 or
            (trace['path'] == 'offload' and not trace.get('fallback_reason')),
            'offload unavailable: request fell back locally; cause=' + str(trace.get('fallback_reason') or 'unknown'))
    if trace['path'] == 'offload':
        require(trace['server_prefill_ms'] is not None and trace['wire_bytes'] > 0, 'prefill trace missing on offloaded call')


def stream_chat(url, token, body, mode, sid, timeout=300):
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json', 'X-Hybrid-Bench-Mode': mode}
    if sid:
        headers['X-Hybrid-Session'] = sid
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(dict(body, stream=True)).encode(), headers)
    started = time.perf_counter()
    first = None
    message = dict(role='assistant', content='')
    trace = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        sid = response.headers['X-Hybrid-Session']
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                done = True
                continue
            chunk = json.loads(line[6:])
            if 'hybrid_bench' in chunk:
                trace = chunk['hybrid_bench']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                for key in ('content', 'reasoning_content'):
                    if delta.get(key):
                        if first is None:
                            first = time.perf_counter() - started
                        message[key] = message.get(key, '') + delta[key]
                require(not delta.get('tool_calls'), 'unexpected generated tool call in synthetic text fixture')
    total = time.perf_counter() - started
    require(done and trace is not None, 'stream aborted or missing committed benchmark trace')
    return sid, message, trace, first, total


def link_probe(config):
    rtts = []
    for _ in range(3):
        started = time.perf_counter()
        health = http(config['prefill_url'], '/v1/health', timeout=5)
        require(health['model_sha256'] == config.get('node_model_sha256', config['model_sha256']), 'link probe model mismatch')
        rtts.append(time.perf_counter() - started)
    started = time.perf_counter()
    token = next(iter(config['clients'].values()))['prefill_token']
    body = http(config['prefill_url'], '/v1/bench/link', token, timeout=10, binary=True)
    elapsed = time.perf_counter() - started
    require(body == b'x' * (1024 * 1024), 'link probe payload mismatch')
    return dict(stage='link', status='passed', rtt_ms=statistics.median(rtts) * 1000,
                bytes=len(body), seconds=elapsed, bytes_per_second=len(body) / elapsed,
                limitation='application-level download including RTT; not physical or upload bandwidth')


def run_checker(command, scratch):
    process = start_process(command, scratch, 'checker')
    try:
        process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        raise
    return process


def equivalence(config, client, tokens, args, scratch, boundaries=None, cross_backend=False):
    token = next(iter(config['clients'].values()))['prefill_token']
    base = config['prefill_url']
    remote = http(base, '/v1/sessions', token, dict(model_sha256=config.get('node_model_sha256', config['model_sha256'])))['session_id']
    try:
        # Replay the run's append endpoints, including intervening shadow tokens, into an isolated session.
        ends = sorted(set([n for n in (boundaries or []) if 0 < n < len(tokens)] + [len(tokens) - 1]))
        chunk = config.get('extend_chunk_tokens', max(1, min((1024**2 - 4096) // 11,
            int((512 * 1024**2 - client['sizing']['recurrent_state_bytes'] - 1024**2) // client['sizing']['kv_bytes_per_token']))))
        n0 = 0
        paths = []
        import struct
        for endpoint in ends:
            while n0 < endpoint:
                n1 = min(endpoint, n0 + chunk)
                prefix = hashlib.sha256(b''.join(struct.pack('<i', t) for t in tokens[:n0])).hexdigest()
                try:
                    data = http(base, '/v1/sessions/' + remote + '/extend', token,
                        dict(n0=n0, prefix_hash_n0=prefix, tokens=tokens[n0:n1], return_delta=True),
                        timeout=args.request_timeout or 120, binary=True)
                except urllib.error.HTTPError as error:
                    if error.code == 503:
                        try:
                            budget = json.loads(error.read(8192)).get('error') in ('ram budget', 'RAM session budget exceeded')
                        except (ValueError, AttributeError):
                            budget = False
                        raise Refusal('ram budget refusal (HTTP 503)' if budget else 'checker node refusal (HTTP 503)') from None
                    raise
                delta_path = scratch / ('check-' + str(n0) + '.delta')
                if config.get('hetero_quant'):
                    size, = struct.unpack('<I', data[:4])
                    require(0 < size <= 1024**2 and 4 + size + 48 <= len(data), 'invalid checker delta header')
                    header = json.loads(data[4:4 + size])
                    require(header['model_sha256'] == config['node_model_sha256'], 'checker delta source mismatch')
                    header['model_sha256'] = config['model_sha256']
                    encoded = json.dumps(header, separators=(',', ':')).encode()
                    data = struct.pack('<I', len(encoded)) + encoded + data[4 + size:]
                delta_path.write_bytes(data)
                paths.append(str(delta_path))
                n0 = n1
        stem = scratch / ('check-' + str(len(tokens)))
        token_path, result_path = [Path(str(stem) + suffix) for suffix in ('.json', '.result.json')]
        private_json(token_path, dict(tokens=tokens, model_sha256=config['model_sha256'], k=args.check_tokens,
                                     deltas=paths, max_ctx=client['context']))
        binary = Path(client['build_dir']) / 'bin/llama-hybrid-bench-check'
        process = run_checker([str(binary), client['model_path'], str(token_path), str(paths[0]), str(result_path),
            '0' if client['backend'] == 'cpu' else '999'], scratch)
        require(result_path.is_file(), 'equivalence checker aborted before producing metrics')
        result = json.loads(result_path.read_text())
        require(process.returncode in (0, 3), 'equivalence checker failed before valid metrics')
        status, code = classify_equivalence(result, cross_backend or config.get('hetero_quant', False))
        result.update(exit_code=code, checker_exit_code=process.returncode, status=status,
                      check_mode='append_chain' if boundaries is not None else 'fresh',
                      append_endpoints=ends, delta_count=len(paths), cross_backend=cross_backend,
                      cross_backend_thresholds_calibrated=False if cross_backend else None,
                      hetero_quant_thresholds_calibrated=False if config.get('hetero_quant') else None,
                      reference='same-partition and whole-prompt on client backend; full-vocabulary KL')
        return result
    finally:
        request = urllib.request.Request(base + '/v1/sessions/' + remote, method='DELETE',
                                         headers={'Authorization': 'Bearer ' + token})
        active_error = sys.exc_info()[1]
        try:
            with urllib.request.urlopen(request, timeout=10):
                pass
        except urllib.error.HTTPError as error:
            if error.code != 404:
                if isinstance(active_error, Refusal):
                    raise Refusal(str(active_error) + f'; checker release failed (HTTP {error.code})') from None
                raise


def classify_equivalence(result, cross_backend):
    if cross_backend:
        return ('uncalibrated' if result['numerical_within_bounds'] else 'uncalibrated_outside_reference', 5)
    return ('passed', 0) if result['same_partition_exact'] and result['numerical_within_bounds'] else ('failed', 3)


@contextlib.contextmanager
def mode_session(config, client, sid, rows, mode):
    try:
        yield
    finally:
        request = urllib.request.Request(client['proxy_url'] + '/v1/sessions/' + sid, method='DELETE',
            headers={'Authorization': 'Bearer ' + config['client_credential']})
        active_error = sys.exc_info()[0] is not None
        try:
            try:
                with urllib.request.urlopen(request, timeout=30):
                    pass
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
            health = http(config['prefill_url'], '/v1/health', timeout=5)
            rows.append(dict(stage='release', mode=mode, status='released',
                             node_sessions=health.get('sessions'), node_ram_bytes=health.get('ram_bytes'),
                             node_evictions=health.get('evictions')))
        except Exception as error:
            rows.append(dict(stage='release', mode=mode, status='failed', reason=type(error).__name__))
            if not active_error:
                raise


def sanitized_info(value):
    allowed = ('os', 'os_version', 'arch', 'chip', 'gpu', 'backend', 'memory_bytes', 'available_memory_bytes',
        'vram_bytes', 'available_vram_bytes', 'driver', 'cuda', 'commit', 'flags', 'binaries',
        'model_file', 'model_sha256', 'quantization_type', 'layout_fingerprint', 'hetero_quant',
        'node_model_sha256', 'node_quantization_type', 'node_limits', 'bench_link', 'threads', 'threads_batch', 'batch', 'ubatch', 'context', 'benchmark_context', 'port', 'sizing', 'policy')
    return {key: value[key] for key in allowed if key in value}


def bundle(root, name, rows, client, node):
    path = root / (name + '.jsonl')
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    summary = rows[-1]
    lines = ['# Hybrid benchmark ' + name, '', 'Status: ' + summary['status'] + '.', '',
             '| Mode | Calls | Mean TTFT (s) | Total (s) | Decode (tokens/s) |', '|---|---:|---:|---:|---:|']
    for mode in ('local', 'hybrid', 'offload'):
        calls = [r for r in rows if r.get('stage') == 'call' and r['mode'] == mode]
        if calls:
            first = [r['ttft_seconds'] for r in calls if r['ttft_seconds'] is not None]
            rates = [r['decode_tokens_per_second'] for r in calls if r['decode_tokens_per_second'] is not None]
            lines.append(f"| {mode} | {len(calls)} | {statistics.mean(first) if first else 0:.6f} | {sum(r['total_seconds'] for r in calls):.6f} | {statistics.mean(rates) if rates else 0:.3f} |")
    lines += ['', 'Skipped stages: ' + str(summary['skipped']) + '.', 'Aborted stages: ' + str(summary['aborted']) + '.',
              'Exit code: ' + str(summary['exit_code']) + '.',
              'Decode rate uses native generation time; TTFT is the first nonempty streamed text/reasoning delta.']
    for row in rows:
        if row.get('stage') == 'equivalence':
            lines += ['', 'Equivalence: `' + json.dumps(row, sort_keys=True) + '`.']
        if row.get('stage') == 'error':
            lines += ['', 'Aborted: ' + row['reason'] + '.']
    (root / (name + '.md')).write_text('\n'.join(lines) + '\n')
    # Bundle only these public, content-free artifacts, never the config/database/state/check inputs.
    public = {name + '.jsonl': path.read_bytes(), name + '.md': (root / (name + '.md')).read_bytes(),
              'client-info.json': (json.dumps(sanitized_info(client), indent=2) + '\n').encode(),
              'node-info.json': (json.dumps(sanitized_info(node), indent=2) + '\n').encode()}
    manifest = ''.join(hashlib.sha256(data).hexdigest() + '  ' + filename + '\n' for filename, data in sorted(public.items()))
    public['SHA256SUMS'] = manifest.encode()
    import io
    archive = root / (name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as target:
        for filename, data in public.items():
            member = tarfile.TarInfo(filename)
            member.size, member.mode = len(data), 0o600
            target.addfile(member, io.BytesIO(data))
    (root / (name + '.tar.gz.sha256')).write_text(sha256(archive) + '  ' + archive.name + '\n')
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-state', required=True)
    p.add_argument('--node-info', help='required when client was configured with --node-sha256 only')
    p.add_argument('--label', default='client')
    p.add_argument('--output', default='.')
    p.add_argument('--scratch', required=True, help='existing private task scratch directory; check inputs are hashed then removed')
    p.add_argument('--modes', nargs='+', choices=['local', 'hybrid', 'offload'], default=['local', 'hybrid', 'offload'])
    p.add_argument('--schedule', choices=['fixed', 'measured'], default='fixed')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--depths', type=int, nargs='+', help='override the depth ladder; smoke still controls tool/output sizes')
    p.add_argument('--request-timeout', type=int, help='HTTP response timeout in seconds for long CPU runs')
    p.add_argument('--check', action='store_true')
    p.add_argument('--check-tokens', type=int, default=64)
    p.add_argument('--check-depths', type=int, nargs='+', help='check first completed call at or above each depth')
    args = p.parse_args()
    require(args.request_timeout is None or 1 <= args.request_timeout <= 86400, '--request-timeout must be in 1..86400')
    require(re.fullmatch('[a-zA-Z0-9_-]{1,64}', args.label), 'label must contain 1-64 letters, digits, underscores or hyphens')
    require(1 <= args.check_tokens <= 256, '--check-tokens must be in 1..256')
    state, root, scratch = Path(args.client_state), Path(args.output), Path(args.scratch)
    require(root.is_dir() and scratch.is_dir(), 'output and scratch directories must already exist')
    config = read_private(state / 'proxy-config.json')
    client = read_private(state / 'client-info.json')
    node = json.loads(Path(args.node_info or state / 'node-info.json').read_text())
    name = 'hybrid-bench-' + args.label + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rows = [dict(stage='run', seed=args.seed, schedule=args.schedule, smoke=args.smoke, check_requested=args.check,
                 provenance=PROVENANCE, workload=list(schedule(args.seed, args.schedule == 'measured', args.smoke, args.depths)))]
    code, skipped, aborted = 0, 0, 0
    checks = 0
    interrupted = False
    try:
        require(client['model_sha256'] == config['model_sha256'], 'metadata model identity mismatch')
        validate_identity(client['model_sha256'], client, node['model_sha256'], node, config.get('hetero_quant', False))
        if node.get('bench_link', False):
            rows.append(link_probe(config))
        else:
            rows.append(dict(stage='link', status='skipped', reason='node launcher --bench-link not enabled'))
            skipped += 1
        limit = min(client['context'], node['context'])
        credential = config['client_credential']
        def decode(path, data=None):
            return http(config['decode_url'], path, config['decode_token'], data)
        def count_messages(messages, output):
            body = dict(messages=messages, chat_template_kwargs=dict(enable_thinking=False))
            text = decode('/apply-template', body)['prompt']
            return len(decode('/tokenize', dict(content=text, add_special=True))['tokens']) + output + 2
        def synthetic(count):
            text = ' x' * count
            require(len(decode('/tokenize', dict(content=text, add_special=False))['tokens']) == count,
                    'synthetic filler is not one token per repeat on this model')
            return text
        for mode in args.modes:
            history = [dict(role='system', content='Read synthetic tool results. Reply with decimal digits only.')]
            sid, previous = 'bench-' + uuid.uuid4().hex, 0
            with mode_session(config, client, sid, rows, mode):
                wanted = list(args.check_depths or ([128] if args.smoke else [4096, 32768, 131072]))
                for index, step in enumerate(rows[0]['workload']):
                    incoming = []
                    if step['kind'] == 'depth':
                        target = min(step['depth'], limit - step['output'] - args.check_tokens - 128)
                        if previous >= target:
                            rows.append(dict(stage='skip', mode=mode, reason='ladder depth already exceeded', requested_depth=step['depth']))
                            skipped += 1
                            continue
                        incoming = [dict(role='user', content=synthetic(max(1, target - previous - 24)))]
                    else:
                        incoming = [dict(role='tool', tool_call_id='synthetic-' + str(index), content=synthetic(step['tokens'])),
                                    dict(role='user', content='Continue with decimal digits.')]
                    messages = history + incoming
                    if count_messages(messages, step['output']) > limit:
                        rows.append(dict(stage='skip', mode=mode, reason='context cap', requested=step, context=limit))
                        skipped += 1
                        continue
                    # Check liveness for every mode: a killed node aborts the comparison instead of hiding fallback.
                    health = http(config['prefill_url'], '/v1/health', timeout=3)
                    require(health['model_sha256'] == config.get('node_model_sha256', config['model_sha256']), 'node identity changed during run')
                    body = dict(messages=messages, max_tokens=step['output'], temperature=0, seed=args.seed,
                                ignore_eos=True, chat_template_kwargs=dict(enable_thinking=False))
                    sid, message, trace, first, total = stream_chat(client['proxy_url'], credential, body, mode, sid, args.request_timeout or 300)
                    verify_meter(trace, previous, mode)
                    row = dict(stage='call', mode=mode, schedule_index=index, requested=step, ttft_seconds=first,
                               total_seconds=total, decode_tokens_per_second=(trace['generated_tokens'] * 1000 / trace['decode_ms']) if trace.get('decode_ms') else None,
                               **{k: v for k, v in trace.items() if k != 'mode'})
                    rows.append(row)
                    previous = trace['depth_after']
                    history = messages + [message]
                    if args.check and mode == 'offload' and wanted and trace['prompt_depth'] >= wanted[0]:
                        with tempfile.TemporaryDirectory(prefix='equivalence-', dir=scratch) as temporary:
                            check_root = Path(temporary)
                            info = decode(f"/slots/{trace['slot']}/info")
                            tokens = info['tokens'][:trace['prompt_depth']]
                            try:
                                for check_mode in ('fresh', 'append_chain'):
                                    check_dir = check_root / check_mode
                                    check_dir.mkdir()
                                    boundaries = None if check_mode == 'fresh' else [endpoint for r in rows
                                        if r.get('stage') == 'call' and r.get('mode') == 'offload' for c in r.get('chunks', [])
                                        if c['status'] == 'imported' for endpoint in (c['n0'], c['n1'])]
                                    result = equivalence(config, client, tokens, args, check_dir, boundaries,
                                                         node['backend'] != client['backend'])
                                    rows.append(result)
                                    checks += 1
                                    if result['exit_code']:
                                        code = result['exit_code']
                            finally:
                                with (scratch / 'state-dump-hashes.jsonl').open('a') as manifest:
                                    for artifact in check_root.rglob('*'):
                                        if not artifact.is_file():
                                            continue
                                        manifest.write(json.dumps(dict(file=artifact.name, bytes=artifact.stat().st_size, sha256=sha256(artifact))) + '\n')
                        wanted = [depth for depth in wanted if depth > trace['prompt_depth']]
                if args.check and mode == 'offload' and wanted:
                    rows.append(dict(stage='skip', mode=mode, reason='equivalence depths not reached', depths=wanted))
                    skipped += len(wanted)
        calls = [r for r in rows if r['stage'] == 'call']
        require(calls, 'no benchmark calls fit the available context')
        shapes = {}
        for call in calls:
            shape = (call['depth_before'], call['prompt_depth'], call['new_tokens'], call['generated_tokens'])
            key = call['schedule_index']
            if key in shapes and shapes[key] != shape:
                rows.append(dict(stage='skip', reason='mode token counts differ; comparison is partial', schedule_index=key))
                skipped += 1
            shapes[key] = shape
        if args.check and checks == 0:
            if 'offload' not in args.modes:
                rows.append(dict(stage='skip', reason='equivalence requires offload among selected modes'))
                skipped += 1
            code = 4
        if skipped and code == 0:
            code = 4
    except KeyboardInterrupt:
        interrupted, aborted, code = True, aborted + 1, 130
        rows.append(dict(stage='error', reason='KeyboardInterrupt'))
    except Exception as error:
        aborted += 1
        code = 2
        rows.append(dict(stage='error', reason=str(error) if isinstance(error, Refusal) else type(error).__name__))
    rows.append(dict(stage='summary', status='interrupted' if interrupted else 'aborted' if aborted else 'failed' if code == 3 else 'uncalibrated' if code == 5 else 'partial' if code else 'passed',
                     skipped=skipped, aborted=aborted, checks=checks, exit_code=code))
    for row in rows:
        row.update(client_model_sha256=client.get('model_sha256'), node_model_sha256=node.get('model_sha256'),
                   client_quantization_type=client.get('quantization_type'), node_quantization_type=node.get('quantization_type'),
                   hetero_quant=config.get('hetero_quant', False))
    archive = bundle(root, name, rows, client, node)
    print(str(archive))
    print('sha256 ' + sha256(archive))
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Refusal, OSError, ValueError, KeyError) as error:
        print('refused: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        sys.exit(2)
