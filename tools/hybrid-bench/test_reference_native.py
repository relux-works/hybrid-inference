"""Short opt-in CPU reference scoring, malformed-file and real hetero-node tests. No downloads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

from bench_common import ROOT, model_identity, sha256
from hybrid_bench import bundle, equivalence
from reference_logits import reference_header

sys.path.insert(0, str(ROOT / 'tools/hybrid'))
from test_limits import Node


def test_prefix_position_witness(result):
    # A separately prefilled prefix must predict the corresponding stored row within the measured encoding floor.
    assert result['local']['kl_nats']['max'] < 1e-4, 'wrong-position distribution exceeds prefix witness floor'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', default='build')
    parser.add_argument('--client', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--artifacts', required=True)
    args = parser.parse_args()
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    binary = Path(args.build_dir).resolve() / 'bin/llama-hybrid-score'
    client_hash, reference_hash = sha256(args.client), sha256(args.reference)
    tokens = [785, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    continuation = [200, 201, 202, 203, 204, 205, 206, 207, 208]
    base = dict(tokens=tokens, continuation=continuation, max_ctx=512, score_batch=3)
    reference_path = root / 'reference.bin'
    commands = []

    def run(name, model, fixture, expected=0, reason=None):
        source, output = root / (name + '.json'), root / (name + '.result.json')
        source.write_text(json.dumps(fixture) + '\n')
        command = [str(binary), model, str(source), str(output), '0']
        with (root / (name + '.log')).open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            (root / (name + '.pid')).write_text(str(process.pid) + '\n')
            try:
                code = process.wait(timeout=180)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
        commands.append(dict(name=name, command=command, exit_code=code, expected=expected))
        (root / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert code == expected, (name, code, (root / (name + '.log')).read_text()[-2000:])
        if reason:
            assert reason in (root / (name + '.log')).read_text(), name
            assert not output.exists() and not Path(str(output) + '.tmp').exists(), name
            assert not fixture.get('write_reference') or not Path(fixture['write_reference']).exists(), name
            print('PASS ' + name + ': intended exit2 ' + reason, flush=True)
            return None
        result = json.loads(output.read_text())
        assert len(result['positions']) == len(fixture['continuation'])
        assert [p['position'] for p in result['positions']] == list(range(len(fixture['tokens']), len(fixture['tokens']) + len(fixture['continuation'])))
        assert 'tokens' not in result['identity'] and 'continuation' not in result['identity']
        print('PASS ' + name, flush=True)
        return result

    # The native math control checks KL direction and independently specified spread values.
    with (root / 'self-test.log').open('wb') as log:
        control = subprocess.Popen([str(binary), '--self-test'], stdout=log, stderr=log)
        (root / 'self-test.pid').write_text(str(control.pid) + '\n')
        assert control.wait(timeout=15) == 0

    # Same-model repeats bound float16 error; Q4 local KL(D || Q4) is positive and repeatable on fixed positions.
    producer = run('reference-write', args.reference, dict(base, model_sha256=reference_hash, write_reference=str(reference_path)))
    compare = dict(base, reference_logits=str(reference_path), reference_model_sha256=reference_hash)
    same = run('reference-self', args.reference, dict(compare, model_sha256=reference_hash))
    same_repeat = run('reference-repeat', args.reference, dict(compare, model_sha256=reference_hash))
    local = run('client-local', args.client, dict(compare, model_sha256=client_hash))
    local_repeat = run('client-repeat', args.client, dict(compare, model_sha256=client_hash))
    assert same['positions'] == same_repeat['positions']
    assert local['positions'] == local_repeat['positions']
    assert same['local']['kl_nats']['max'] <= max(1e-12, 1.1 * producer['encoding']['kl_nats']['max'])
    assert local['local']['kl_nats']['mean'] > max(1e-6, 10 * same['local']['kl_nats']['max'])

    # The reverse-direction control also measures positive, stable KL(Q4 || Q8), independently of the study's D || arm endpoint.
    client_reference = root / 'q4-reference.bin'
    run('q4-reference-write', args.client, dict(base, model_sha256=client_hash, write_reference=str(client_reference)))
    reverse_fixture = dict(base, model_sha256=reference_hash, reference_model_sha256=client_hash, reference_logits=str(client_reference))
    reverse = run('q4-to-q8', args.reference, reverse_fixture)
    reverse_repeat = run('q4-to-q8-repeat', args.reference, reverse_fixture)
    assert reverse['positions'] == reverse_repeat['positions'] and reverse['local']['kl_nats']['mean'] > 1e-6

    # A schedule change requires its own reference; unbound continuation schedules must refuse before output.
    run('wrong-score-batch', args.reference, dict(compare, model_sha256=reference_hash, score_batch=1), 2, 'reference scoring schedule mismatch')
    scalar_path = root / 'scalar-reference.bin'
    run('scalar-reference', args.reference, dict(base, score_batch=1, model_sha256=reference_hash, write_reference=str(scalar_path)))
    scalar = run('scalar-batches', args.reference, dict(compare, reference_logits=str(scalar_path), model_sha256=reference_hash, score_batch=1))
    assert scalar['local']['kl_nats']['max'] < 1e-4
    assert all(p['local_top1_agreed'] for p in scalar['positions'])
    # Q4 fresh-prefix probes expose client batch-shape variation rather than borrowing the Q8 float16 floor.
    assert local['identity']['context_prefill'] == 'split-last-v1'
    assert local['schedule_floor']['kind'] == 'fresh-prefix-spots-v1'
    assert local['schedule_floor']['local']['max'] > 1e-5, 'Q4 schedule variation was not reported'

    # Each malformed identity/file must name its rejection and leave no result or partial output.
    run('wrong-model-file', args.client, dict(compare, model_sha256=client_hash, reference_logits=str(client_reference)), 2, 'reference model hash mismatch')
    run('wrong-model', args.client, dict(compare, model_sha256=client_hash, reference_model_sha256=client_hash), 2, 'reference model hash mismatch')
    run('wrong-token', args.client, dict(compare, model_sha256=client_hash, continuation=[209] + continuation[1:]), 2, 'reference token mismatch')
    run('wrong-context', args.client, dict(compare, model_sha256=client_hash, tokens=[786] + tokens[1:]), 2, 'reference token mismatch')
    run('wrong-n', args.client, dict(compare, model_sha256=client_hash, tokens=tokens[:-1]), 2, 'reference N/M mismatch')
    run('wrong-m', args.client, dict(compare, model_sha256=client_hash, continuation=continuation[:-1]), 2, 'reference N/M mismatch')
    original = reference_path.read_bytes()
    header_size, = struct.unpack_from('<I', original, 8)
    header = json.loads(original[12:12 + header_size])
    payload = original[12 + header_size:]
    stride = 44 + 2 * header['vocab']

    # Prefix witnesses independently exercise the last/first row of neighboring batches and the final row.
    def witness_fixture(index, row, name):
        prefix = tokens + continuation[:index]
        target = continuation[index:index + 1]
        token_hash = hashlib.sha256(b''.join(struct.pack('<i', token) for token in prefix + target)).hexdigest()
        subset = dict(header, tokens=prefix, continuation=target, n=len(prefix), m=1, token_sha256=token_hash)
        text = json.dumps(subset).encode()
        path = root / (name + '.bin')
        path.write_bytes(b'HYBLOG01' + struct.pack('<I', len(text)) + text + row)
        return dict(compare, tokens=prefix, continuation=target, model_sha256=reference_hash, reference_logits=str(path))

    for index in (3, 4, 8):
        name = 'prefix-witness-' + str(index)
        row = payload[index * stride:(index + 1) * stride]
        test_prefix_position_witness(run(name, args.reference, witness_fixture(index, row, name)))
    index = 3
    row = payload[index * stride:index * stride + 8] + payload[(index + 1) * stride + 8:(index + 2) * stride - 32]
    planted = row + hashlib.sha256(row).digest()
    planted_result = run('wrong-row-control', args.reference, witness_fixture(index, planted, 'wrong-row-control'))
    with (root / 'wrong-row-witness.log').open('wb') as log:
        command = [sys.executable, str(Path(__file__).resolve()), '--position-witness', str(root / 'wrong-row-control.result.json')]
        control = subprocess.Popen(command, stdout=log, stderr=log)
        (root / 'wrong-row-witness.pid').write_text(str(control.pid) + '\n')
        code = control.wait(timeout=15)
    assert code == 1 and 'wrong-position distribution exceeds prefix witness floor' in (root / 'wrong-row-witness.log').read_text()
    commands.append(dict(name='wrong-row-witness', command=command, exit_code=code, expected=1))
    (root / 'position-control.json').write_text(json.dumps(dict(test='test_prefix_position_witness',
        verdict='expected failure', witness_exit_code=code, scorer_exit_code=0,
        kl_nats=planted_result['local']['kl_nats']['max'])) + '\n')
    print('PASS wrong-row-control: test_prefix_position_witness fails as intended', flush=True)

    def malformed(name, data, reason):
        path = root / (name + '.bin')
        path.write_bytes(data)
        return run(name, args.client, dict(compare, model_sha256=client_hash, reference_logits=str(path)), 2, reason)

    malformed('truncated', original[:-1], 'truncated or trailing reference file')
    malformed('trailing', original + b'x', 'truncated or trailing reference file')
    changed_header = dict(header, vocab_sha256='0' * 64)
    text = json.dumps(changed_header).encode()
    malformed('wrong-vocabulary', b'HYBLOG01' + struct.pack('<I', len(text)) + text + payload, 'reference vocabulary mismatch')
    for name, half in [('nan', 0x7e00), ('positive-inf', 0x7c00), ('negative-inf', 0xfc00)]:
        row = bytearray(payload[:stride - 32])
        struct.pack_into('<H', row, 12, half)
        damaged = bytes(row) + hashlib.sha256(row).digest() + payload[stride:]
        malformed(name, original[:12 + header_size] + damaged, 'nonfinite reference log-probability')
    damaged = bytearray(payload)
    damaged[12] ^= 1
    malformed('checksum', original[:12 + header_size] + damaged, 'reference row checksum mismatch')

    changed_header = dict(header, vocab=header['vocab'] + 1)
    text = json.dumps(changed_header).encode()
    rows = []
    for offset in range(0, len(payload), stride):
        row = payload[offset:offset + stride - 32] + struct.pack('<e', -100)
        rows.append(row + hashlib.sha256(row).digest())
    malformed('wrong-vocabulary-size', b'HYBLOG01' + struct.pack('<I', len(text)) + text + b''.join(rows), 'reference vocabulary mismatch')
    assert reference_path.read_bytes() == original

    # A checksum-valid off-by-one row plant must fail on position identity, not on unrelated corruption.
    row = bytearray(payload[:stride - 32])
    struct.pack_into('<I', row, 0, len(tokens) + 1)
    shifted = original[:12 + header_size] + row + hashlib.sha256(row).digest() + payload[stride:]
    malformed('shifted-position-control', shifted, 'reference position mismatch')

    # Real Q8 node deltas retain source hashes; both hybrid and local Q4 arms survive orchestration and bundling.
    identity = model_identity(args.client)
    client = dict(model_sha256=client_hash, model_path=args.client, build_dir=str(Path(args.build_dir).resolve()),
                  backend='cpu', context=512, **identity)
    node_args = SimpleNamespace(artifacts=str(root), binary=str(Path(args.build_dir).resolve() / 'bin/llama-hybrid-prefill'), model=args.reference)
    records = []
    with Node(node_args, 'q8-node', context='512') as node:
        config = dict(model_sha256=client_hash, node_model_sha256=reference_hash, hetero_quant=True,
                      node_layout_fingerprint=identity['layout_fingerprint'], prefill_url=node.base,
                      clients={'test': dict(prefill_token=node.clients['a'])})
        check_args = SimpleNamespace(check_tokens=len(continuation), noise_chunks=[4, 7], request_timeout=120,
                                     reference_logits=str(reference_path), reference_model_sha256=reference_hash)
        def record(row):
            records.append(row)
            with (root / 'orchestration.jsonl').open('a') as out:
                out.write(json.dumps(row) + '\n')

        for name in ('hybrid-first', 'hybrid-repeat', 'hybrid-chain'):
            scratch = root / name
            scratch.mkdir(mode=0o700)
            result = equivalence(config, client, tokens, check_args, scratch, boundaries=[5, 10] if name == 'hybrid-chain' else None, record=record)
            assert result['status'] == 'uncalibrated' and result['checker_exit_code'] == 5
            assert node.request('GET', '/v1/sessions') == []
    scored = [r for r in records if r['stage'] == 'reference_scoring']
    assert len(scored) == 3
    assert scored[0]['positions'] == scored[1]['positions']
    assert scored[0]['local'] == local['local']
    assert scored[0]['hybrid'] is not None and scored[0]['local_to_hybrid'] is not None
    records.append(dict(stage='summary', status='uncalibrated', skipped=0, aborted=0, checks=3, exit_code=5))
    archive = bundle(root, 'reference-orchestration', records, client, dict(model_sha256=reference_hash))
    assert 'Reference scoring: KL(D || arm)' in (root / 'reference-orchestration.md').read_text()
    (root / 'measurements.json').write_text(json.dumps(dict(reference_sha256=sha256(reference_path), reference_bytes=len(original),
        producer=producer, same=same, local=local, reverse=reverse, scalar=scalar, hybrid=scored[0], archive_sha256=sha256(archive)), indent=2) + '\n')
    print('PASS real_hetero_reference_orchestration: native checker5, scorer0, two arms, stable repeat, no sessions', flush=True)
    print('ALL CPU REFERENCE TESTS PASSED', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--position-witness':
        test_prefix_position_witness(json.loads(Path(sys.argv[2]).read_text()))
    else:
        main()
