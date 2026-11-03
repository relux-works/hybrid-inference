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
import subprocess
import sys
import time

import hetero_study as study

ARMS = ('offload', 'local', 'partition')
HANDOFF_POSITIONS = 'forced inputs at depth..depth+63; logits predict depth+1..depth+64'


def settings(depth):
    return dict(yarn_factor=2 if depth <= 262144 else 4, rope_theta=10000000,
                original_context=262144, kv_type='q8_0', recurrent_type='f32',
                yarn_ext_factor=1, yarn_attn_factor=1, yarn_beta_fast=32, yarn_beta_slow=1)


def binding(fixture):
    def token_hash(tokens):
        return hashlib.sha256(struct.pack('<' + 'i' * len(tokens), *tokens)).hexdigest()
    return dict(model_sha256=fixture['model_sha256'], tokens_sha256=token_hash(fixture['tokens']),
                tail_sha256=token_hash(fixture['tail']), window=fixture.get('window', 'after'), needle_mode=fixture['needle_mode'], probe_tokens=fixture['probe_tokens'], source=fixture['source'], settings=fixture['settings'], needles=fixture['needles'])


def exact_score(text, expected):
    # Only surrounding whitespace is ignored; substrings and explanatory text do not earn credit.
    return text.strip() == expected


def needle_entries(seed, count):
    return [dict(key=f'RECORD_{i:02d}', answer=hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:8].upper())
            for i in range(count)]


def validate_fixture(fixture):
    study.require(fixture.get('version') == 2, 'unsupported depth fixture')
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
    study.require(fixture['needle_mode'] in ('sequential', 'independent') and len(fixture['probe_tokens']) == 64, 'missing needle protocol or 64-token probe')
    reserve = sum(max(len(n['query']), len(n['query_next'])) + 32 for n in fixture['needles']) if fixture['needle_mode'] == 'sequential' else 0
    study.require(len(document) + max(64, reserve) <= fixture['settings']['n_ctx'], 'needle chain or reference exceeds reserved context')
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


def prepare(binary, model, text, source, root, depth, rows, seed, count, timeout, window='final', needle_mode='sequential'):
    fixture = root / 'fixture.json'
    invoke(binary, root, 'prepare', dict(mode='prepare', model=str(model), model_sha256=study.sha256(model),
        gpu_layers=0, text=str(text), source=source, depth=depth, rows=rows, seed=seed, window=window, needle_mode=needle_mode,
        settings=settings(depth), needles=needle_entries(seed, count), output=str(fixture)), timeout)
    validate_fixture(study.read(fixture))
    return fixture


def run_case(binary, model, fixture_path, root, gpu_layers=0, batch=256, partition_batch=127, timeout=14400, chunk_tokens=8192, handoff_command=None, engine_ref=None):
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
               binary_sha256=binary_hash, model_sha256=model_hash, gpu_layers=gpu_layers, engine_ref=engine_ref,
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
                                          output=str(result), logits=str(logits), reference_logits=str(root/(arm+'-next64.f32'))), timeout)
            record = study.read(result)
            study.require(record['rows'] == len(fixture['tail']), 'arm row count mismatch')
            study.require(record['snapshot_depth'] == fixture.get('context_depth', len(fixture['tokens'])), 'arm snapshot depth mismatch')
            verify_arm(record, identity, logits)
            publish_handoff(root, arm, record, fixture_path, state, handoff_command, timeout)
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
        document=fixture['source'], observation_unit='document', needle_mode=fixture['needle_mode'],
        state_restores={a:records[a]['state_restores'] for a in ARMS},
        references={a:dict(file=a+'-next64.f32', sha256=records[a]['reference_logits_sha256'], rows=64,
                          positions=HANDOFF_POSITIONS) for a in ARMS},
        offload_against_local=dict(kl_nats=offload['a_to_b_kl_nats'], top1_agreement=offload['top1_agreement'],
                                  max_abs_logit_difference=offload['max_abs_logit_difference']),
        partition_against_local=dict(kl_nats=noise['b_to_a_kl_nats'], top1_agreement=noise['top1_agreement'],
                                    max_abs_logit_difference=noise['max_abs_logit_difference']),
        limits='Descriptive same-device f32 delta-v2 encode/stage/apply handoff in one disk container. No HTTP/network measurement, equivalence threshold, confidence claim or backend-wide noise bound. Needle queries restore the full post-context snapshot; the selected window and offload boundary are explicit.')
    report.update(needle_summary(needles))
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
    lines += ['', 'Local needle accuracy: ' + str(report['needle_accuracy']['local']['accuracy']), report['needle_conclusion'],
              'Observation unit: one document. Positions and needles are correlated. Needle mode: ' + report['needle_mode'] + '.',
              f'Window: {report["window"]}; positions after tokens {report["position_range"]}; offload boundary: {report["offload_boundary"]}.', report['limits'], '']
    study.atomic(root/'report.md', '\n'.join(lines).encode())
    return report


def needle_summary(needles):
    accuracy = {arm:dict(correct=sum(n['correct'][arm] for n in needles), total=len(needles)) for arm in ARMS}
    for row in accuracy.values():
        row['accuracy'] = row['correct']/row['total'] if row['total'] else None
    all_missed = bool(needles) and all(row['correct'] == 0 for row in accuracy.values())
    if all_missed:
        conclusion = 'Every arm missed every needle. Answer agreement supplies no retrieval-quality evidence here; the offload comparison rests on KL and the other logit measurements.'
    elif accuracy['local']['correct'] == 0:
        conclusion = 'The local arm retrieved no needles. Agreement alone cannot establish retrieval quality; inspect KL alongside all arm accuracies.'
    else:
        conclusion = 'Needle accuracy is reported separately from answer agreement. Later sequential questions condition on earlier answers; these are correlated checks within one document.'
    return dict(needle_accuracy=accuracy, local_needle_accuracy=accuracy['local']['accuracy'],
                all_arms_missed_all_needles=all_missed, needle_conclusion=conclusion)


def publish_handoff(root, arm, record, fixture, state, command, timeout):
    reference = root/(arm+'-next64.f32')
    study.require(record['reference_rows'] == 64 and record['probe_tokens'] == record['binding']['probe_tokens'], 'handoff probe identity mismatch')
    study.require(reference.stat().st_size == 64*record['vocab']*4 and study.sha256(reference) == record['reference_logits_sha256'], 'handoff reference hash/shape mismatch')
    producer = study.read(root/'identity.json')
    manifest = dict(schema='depth-slot-handoff-1', producer=producer, logits_dtype='float32-le', positions=HANDOFF_POSITIONS,
        depth=record['snapshot_depth'], rows=64, vocab=record['vocab'], continuation=record['probe_tokens'],
        binding=record['binding'], fixture=study.file_record(fixture), state=study.file_record(state),
        sidecar=study.file_record(Path(str(state)+'.json')), reference=study.file_record(reference),
        arm=arm, state_retention='ephemeral until callback returns; otherwise removed after this arm')
    path = root/(arm+'-handoff.json')
    study.save(path, manifest)
    if command is not None:
        study.run([command, path], root, arm+'-handoff', timeout)


def verify_handoff(args):
    root = study.private_dir(args.output)
    manifest = study.read(args.handoff)
    study.require(manifest.get('schema') == 'depth-slot-handoff-1' and manifest.get('logits_dtype') == 'float32-le' and
                  manifest.get('positions') == HANDOFF_POSITIONS and manifest.get('rows') == 64, 'handoff format or position convention mismatch')
    engine_ref = manifest.get('producer', {}).get('engine_ref')
    study.require(isinstance(engine_ref, str) and len(engine_ref) == 40 and all(c in '0123456789abcdef' for c in engine_ref),
                  'handoff requires a recorded engine source revision')
    current = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    study.require(current == engine_ref, 'handoff engine source revision mismatch')
    dirty = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'status', '--porcelain'], text=True).strip()
    study.require(not dirty, 'handoff engine source must be clean')
    fixture = study.read(args.fixture)
    identity = validate_fixture(fixture)
    study.require(manifest['binding'] == identity and manifest['continuation'] == fixture['probe_tokens'], 'handoff document, continuation or YaRN mismatch')
    study.require(manifest['depth'] == fixture['context_depth'], 'handoff depth mismatch')
    study.require(study.sha256(args.model) == identity['model_sha256'], 'handoff model mismatch')
    paths = dict(fixture=args.fixture, state=args.state, sidecar=Path(str(args.state)+'.json'), reference=args.reference)
    for key, path in paths.items():
        record = manifest[key]
        study.require(path.stat().st_size == record['size'] and study.sha256(path) == record['sha256'], 'handoff '+key+' hash/size mismatch')
    study.require(args.reference.stat().st_size == 64*manifest['vocab']*4, 'handoff reference shape mismatch')
    output, logits = root/'probe.json', root/'mac-next64.f32'
    invoke(args.binary, root, 'probe', dict(mode='probe', model=str(args.model), model_sha256=identity['model_sha256'],
        fixture=str(args.fixture), binding=identity, state=str(args.state), batch_size=32, gpu_layers=args.gpu_layers,
        output=str(output), logits=str(logits)), args.timeout)
    record = study.read(output)
    study.require(record['binding'] == identity and record['rows'] == 64 and record['vocab'] == manifest['vocab'] and
                  record['depth'] == manifest['depth'] and record['probe_tokens'] == manifest['continuation'] and
                  study.sha256(logits) == record['logits_sha256'], 'Mac probe identity mismatch')
    invoke(args.binary, root, 'compare', dict(mode='compare', a=str(args.reference), b=str(logits), rows=64,
        vocab=manifest['vocab'], output=str(root/'comparison.json')), args.timeout)
    study.save(root/'verification.json', dict(manifest_sha256=study.sha256(args.handoff), comparison=study.read(root/'comparison.json'),
        status='descriptive comparison; no cross-backend acceptance threshold', positions=HANDOFF_POSITIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--partition-batch', type=int, default=127)
    parser.add_argument('--engine-ref', help='full reviewed engine source revision recorded by the GPU stage')
    parser.add_argument('--handoff-command', type=Path, help='owned executable called with manifest while the sole state still exists')
    parser.add_argument('--handoff', type=Path, help='verify a transferred depth-slot-handoff-1 manifest')
    parser.add_argument('--state', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--chunk-tokens', type=int, default=8192)
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()
    os.umask(0o077)
    root = study.private_dir(args.output)
    with study.locked(root/'run.lock') as lock:
        study.ACTIVE_LOCK_FD = lock
        try:
            if args.handoff is not None:
                study.require(args.state is not None and args.reference is not None, 'handoff verification requires --state and --reference')
                verify_handoff(args)
                return
            run_case(args.binary.resolve(), args.model.resolve(), args.fixture.resolve(), root,
                     args.gpu_layers, args.batch, args.partition_batch, args.timeout, args.chunk_tokens, None if args.handoff_command is None else args.handoff_command.resolve(), args.engine_ref)
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
