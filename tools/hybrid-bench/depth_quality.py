#!/usr/bin/env python3
"""Sequential depth-quality driver; Python standard library, one native context at a time."""
import argparse
from contextlib import contextmanager
import hashlib
import itertools
import os
from pathlib import Path
import signal
import struct
import sys
import time

import hetero_study as study

ARMS = ('offload', 'local', 'partition')


def settings(depth):
    return dict(yarn_factor=2 if depth <= 524288 else 4, rope_theta=10000000,
                original_context=262144, kv_type='q8_0', recurrent_type='f32',
                yarn_ext_factor=1, yarn_attn_factor=1, yarn_beta_fast=32, yarn_beta_slow=1)


def binding(fixture):
    def token_hash(tokens):
        return hashlib.sha256(struct.pack('<' + 'i' * len(tokens), *tokens)).hexdigest()
    return dict(model_sha256=fixture['model_sha256'], tokens_sha256=token_hash(fixture['tokens']),
                tail_sha256=token_hash(fixture['tail']), window=fixture.get('window', 'after'), settings=fixture['settings'], needles=fixture['needles'])


def exact_score(text, expected):
    # Only surrounding whitespace is ignored; substrings and explanatory text do not earn credit.
    return text.strip() == expected


def needle_entries(seed, count):
    return [dict(key=f'RECORD_{i:02d}', answer=hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:8].upper())
            for i in range(count)]


def validate_fixture(fixture):
    study.require(fixture.get('version') == 1, 'unsupported depth fixture')
    tokens, tail = fixture['tokens'], fixture['tail']
    study.require(tokens and tail and len(tokens) + len(tail) + 512 <= fixture['settings']['n_ctx'] <= 1048576, 'invalid fixture size')
    study.require(all(type(t) is int and 0 <= t <= 2147483647 for t in tokens + tail), 'invalid token')
    document = tokens + tail if fixture.get('window') == 'final' else tokens
    window = fixture.get('window', 'after')
    study.require(window in ('final', 'after') and fixture.get('context_depth', len(tokens)) == len(document), 'window depth mismatch')
    previous_end = 0
    for needle in fixture['needles']:
        pos, inserted = needle['position'], needle['inserted_tokens']
        study.require(previous_end <= pos and document[pos:pos+len(inserted)] == inserted, 'needle placement mismatch')
        previous_end = pos + len(inserted)
        study.require(needle['expected'] and needle['query'], 'empty needle')
    return binding(fixture)


def verify_arm(record, identity, logits):
    study.require(record['binding'] == identity, 'arm document or YaRN mismatch')
    study.require(record['rows'] > 0 and record['vocab'] > 0, 'invalid logit shape')
    study.require(logits.stat().st_size == record['rows'] * record['vocab'] * 4 and
                  study.sha256(logits) == record['logits_sha256'], 'arm logit checksum or size mismatch')
    expected = identity['needles']
    study.require(len(record['answers']) == len(expected), 'missing needle answers')
    for answer, needle in zip(record['answers'], expected):
        study.require((answer['key'], answer['expected']) == (needle['key'], needle['expected']) and
                      len(answer['tokens']) <= 32, 'needle identity or generation cap mismatch')


def invoke(binary, root, name, job, timeout):
    path = root / (name + '.job.json')
    study.save(path, job)
    started = time.monotonic()
    study.run([binary, path], root, name, timeout)
    study.save(root / (name + '.timing.json'), dict(seconds=time.monotonic()-started))


@contextmanager
def single_state(root):
    state = root / 'context.state'
    study.require(not state.exists() and not Path(str(state)+'.json').exists(), 'stale state exists; preserve and inspect before retrying')
    try:
        yield state
    finally:
        # This invocation reserved the absent path while holding its run lock.
        state.unlink(missing_ok=True)
        Path(str(state)+'.json').unlink(missing_ok=True)


def prepare(binary, model, text, source, root, depth, rows, seed, count, timeout, window='final'):
    fixture = root / 'fixture.json'
    invoke(binary, root, 'prepare', dict(mode='prepare', model=str(model), model_sha256=study.sha256(model),
        gpu_layers=0, text=str(text), source=source, depth=depth, rows=rows, seed=seed, window=window,
        settings=settings(depth), needles=needle_entries(seed, count), output=str(fixture)), timeout)
    validate_fixture(study.read(fixture))
    return fixture


def run_case(binary, model, fixture_path, root, gpu_layers=0, batch=256, partition_batch=127, timeout=14400, chunk_tokens=8192):
    root = study.private_dir(root)
    study.require(batch != partition_batch and min(batch, partition_batch) > 0, 'partition batches must differ')
    study.require(max(batch, partition_batch) <= 8192 and 256 <= chunk_tokens <= 8192 and
                  chunk_tokens % 256 == 0 and chunk_tokens % batch == 0, 'transfer chunks must align with local prefill batches')
    fixture_hash = study.sha256(fixture_path)
    fixture = study.read(fixture_path)
    identity = validate_fixture(fixture)
    model_hash = study.sha256(model)
    study.require(identity['model_sha256'] == model_hash, 'fixture/model mismatch')
    binary_hash = study.sha256(binary)
    study.save(root/'identity.json', dict(binding=identity, fixture_sha256=fixture_hash,
               binary_sha256=binary_hash, model_sha256=model_hash, gpu_layers=gpu_layers,
               batch=batch, partition_batch=partition_batch))
    records = {}
    for arm in ARMS:
        study.require(study.sha256(fixture_path) == fixture_hash and study.sha256(binary) == binary_hash, 'changed document or engine between arms')
        with single_state(root) as state:
            common = dict(model=str(model), model_sha256=model_hash, gpu_layers=gpu_layers,
                          fixture=str(fixture_path), binding=identity, state=str(state),
                          batch_size=partition_batch if arm == 'partition' else batch, chunk_tokens=chunk_tokens)
            if arm == 'offload':
                invoke(binary, root, 'offload-node', dict(common, mode='export'), timeout)
                # run() has reaped the node before any client is launched.
            result = root / (arm + '.json')
            logits = root / (arm + '.f32')
            invoke(binary, root, arm, dict(common, mode='evaluate', **{'import': arm == 'offload'},
                                          output=str(result), logits=str(logits)), timeout)
            record = study.read(result)
            study.require(record['rows'] == len(fixture['tail']), 'arm row count mismatch')
            study.require(record['snapshot_depth'] == fixture.get('context_depth', len(fixture['tokens'])), 'arm snapshot depth mismatch')
            verify_arm(record, identity, logits)
            records[arm] = record
    study.require(study.sha256(fixture_path) == fixture_hash, 'changed document between arms')
    for arm in ARMS:
        verify_arm(study.read(root/(arm+'.json')), identity, root/(arm+'.f32'))
    comparisons = {}
    for a, b in itertools.combinations(ARMS, 2):
        study.require((records[a]['rows'], records[a]['vocab']) == (records[b]['rows'], records[b]['vocab']), 'arm shape mismatch')
        path = root / (a + '-vs-' + b + '.json')
        invoke(binary, root, a + '-vs-' + b, dict(mode='compare', a=str(root/(a+'.f32')), b=str(root/(b+'.f32')),
            rows=records[a]['rows'], vocab=records[a]['vocab'], output=str(path)), timeout)
        comparisons[a+'-vs-'+b] = study.read(path)
    offload = comparisons['offload-vs-local']
    noise = comparisons['local-vs-partition']
    needles = []
    for i, needle in enumerate(fixture['needles']):
        answers = {a: records[a]['answers'][i]['text'] for a in ARMS}
        needles.append(dict(key=needle['key'], position=needle['position'], expected=needle['expected'], answers=answers,
            correct={a: exact_score(answers[a], needle['expected']) for a in ARMS},
            offload_agrees_local=answers['offload'] == answers['local'], partition_agrees_local=answers['partition'] == answers['local']))
    report = dict(context_depth=fixture.get('context_depth', len(fixture['tokens'])), window=fixture.get('window', 'after'), offload_boundary=len(fixture['tokens']), tail_positions=len(fixture['tail']),
        position_range=[len(fixture['tokens']), len(fixture['tokens'])+len(fixture['tail'])-1],
        binding=identity, comparisons=comparisons, needles=needles,
        offload_against_local=dict(kl_nats=offload['a_to_b_kl_nats'], top1_agreement=offload['top1_agreement'],
                                  max_abs_logit_difference=offload['max_abs_logit_difference']),
        partition_against_local=dict(kl_nats=noise['b_to_a_kl_nats'], top1_agreement=noise['top1_agreement'],
                                    max_abs_logit_difference=noise['max_abs_logit_difference']),
        limits='Descriptive same-device f32 delta-v2 encode/stage/apply handoff in one disk container. No HTTP/network measurement, equivalence threshold, confidence claim or backend-wide noise bound. Needle queries restore the full post-context snapshot; the selected window and offload boundary are explicit.')
    report['offload_minus_partition_noise'] = {key: report['offload_against_local'][key] - report['partition_against_local'][key]
        for key in ('kl_nats', 'top1_agreement', 'max_abs_logit_difference')}
    study.save(root/'report.json', report)
    lines = ['# Depth quality', '', f'Context depth: {report["context_depth"]}; teacher-forced positions: {report["tail_positions"]}.', '',
             '| Comparison | Mean KL (arm || local), nats | Top-1 agreement | Maximum absolute logit difference |',
             '|---|---:|---:|---:|']
    for label, key in [('Offload', 'offload_against_local'), ('Partition noise', 'partition_against_local')]:
        row = report[key]
        lines.append(f'| {label} | {row["kl_nats"]:.9g} | {row["top1_agreement"]:.6f} | {row["max_abs_logit_difference"]:.9g} |')
    lines += ['', '| Arm | Exact needles | Answers equal to local (raw) |', '|---|---:|---:|']
    for arm in ARMS:
        correct = sum(n['correct'][arm] for n in needles)
        equal = sum(n['answers'][arm] == n['answers']['local'] for n in needles)
        lines.append(f'| {arm} | {correct}/{len(needles)} | {equal}/{len(needles)} |')
    lines += ['', f'Window: {report["window"]}; positions after tokens {report["position_range"]}; offload boundary: {report["offload_boundary"]}.', report['limits'], '']
    study.atomic(root/'report.md', '\n'.join(lines).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--partition-batch', type=int, default=127)
    parser.add_argument('--chunk-tokens', type=int, default=8192)
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()
    os.umask(0o077)
    root = study.private_dir(args.output)
    with study.locked(root/'run.lock') as lock:
        study.ACTIVE_LOCK_FD = lock
        try:
            run_case(args.binary.resolve(), args.model.resolve(), args.fixture.resolve(), root,
                     args.gpu_layers, args.batch, args.partition_batch, args.timeout, args.chunk_tokens)
        finally:
            study.ACTIVE_LOCK_FD = None


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, study.interrupted)
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('failed depth-quality: ' + str(error), file=sys.stderr, flush=True)
        sys.exit(2)
    print('done depth-quality', flush=True)
