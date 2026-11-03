#!/usr/bin/env python3
"""Fresh RTX PRO 6000 stage; execution/downloads require --execute on the GPU host."""
import argparse
import io
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile

import depth_quality as depth
import depth_corpus as corpus
import hetero_study as study
from gpu_stage import FLAGS, add_disk_floor_argument, clean_source, disk, exclusive, gpu_identity, interrupted, read, require, run, save, sha, stream_sha, terminal
from stage_kl_bf16 import download_models


def estimates(depth_tokens, rows=4096, count=16, hourly=1.57, documents=None, needle_mode='sequential'):
    documents = documents or (3 if depth_tokens <= 524288 else 2)
    state = depth_tokens * 34816 + 160_000_000
    result = []
    for label, prefill_seconds_at_million, checks_at_million in [('lower', 900, 900), ('upper', 1200, 1800)]:
        prefill_per_arm = prefill_seconds_at_million * depth_tokens/1040000
        checks = checks_at_million * max(.6, depth_tokens/1040000)
        extra_restores = 0 if needle_mode == 'sequential' else 3*(count-1)*state/2e9
        per_document = 3*prefill_per_arm + checks + extra_restores
        result.append(dict(scenario=label, gpu_hours=documents*per_document/3600,
            hours_per_document=per_document/3600, cost_usd=documents*per_document/3600*hourly,
            prefill_seconds=documents*3*prefill_per_arm, prefill_seconds_per_arm=prefill_per_arm,
            checks_restore_allowance_seconds=documents*checks, extra_independent_restore_seconds=documents*extra_restores))
    return dict(depth=depth_tokens, yarn_settings=depth.settings(depth_tokens), documents=documents, observation_unit='document', state_bytes=state,
        needle_mode=needle_mode, needle_restores_per_arm=1 if needle_mode == 'sequential' else count,
        hourly_usd_assumption=hourly, reference_positions_per_arm=64, scenarios=result,
        assumptions='At 1M: three 15-20 minute prefills plus 15-30 minutes for logits, reference probes, state I/O and needle questions; analyst planning envelope, not measured RTX timing.',
        excludes='setup, build, downloads, remote handoff transfer, archive and billing granularity; more than 4096 logits or 16 needles requires recalibration')


def build_depth(source, directory, ref, log, floor):
    clean_source(source, ref)
    require(not directory.exists(), 'build directory exists; preserve/inspect it')
    disk(directory.parent, 2*1024**3, floor=floor, log=log)
    compiler = subprocess.run(['pgrep', '-u', str(os.getuid()), 'cc1plus'], capture_output=True)
    require(compiler.returncode == 1, 'native compiler active or pgrep failed')
    run(['nice', '-n', '10', 'cmake', '-S', source, '-B', directory, *FLAGS, '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF'], log)
    run(['nice', '-n', '10', 'cmake', '--build', directory, '-j4', '--target', 'llama-hybrid-depth'], log)
    shutil.copyfile(directory/'CMakeCache.txt', directory.parent/'build-cache.txt')
    save(directory.parent/'build-identity.json', dict(source_ref=ref, flags=FLAGS,
        binaries={str(p.relative_to(directory)): sha(p) for p in (directory/'bin').iterdir() if p.is_file()}))


def archive_case(case):
    require(not (case/'context.state').exists(), 'state must be removed before archive')
    # Keep large f32 logits once on disk. The archive includes their hashes, metadata and derived results.
    paths = sorted(p for p in case.iterdir() if p.is_file() and p.suffix not in ('.f32', '.gz'))
    sums = ''.join(sha(p)+'  '+p.name+'\n' for p in paths).encode()
    archive = case/'results.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in paths:
            bundle.add(path, arcname=path.name, recursive=False)
        info = tarfile.TarInfo('SHA256SUMS')
        info.size = len(sums)
        bundle.addfile(info, io.BytesIO(sums))
    verify_bundle(archive)
    save(case/'archive.json', dict(archive_sha256=sha(archive), logits={p.name:sha(p) for p in case.glob('*.f32')}))


def verify_bundle(archive):
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        require(all(m.isfile() for m in members) and len({m.name for m in members}) == len(members), 'invalid archive members')
        sums = bundle.extractfile('SHA256SUMS')
        require(sums is not None, 'missing archive checksums')
        expected = dict(line.decode().rstrip('\n').split('  ',1)[::-1] for line in sums)
        require(set(expected) == {m.name for m in members} - {'SHA256SUMS'}, 'archive checksum member set differs')
        for name, digest in expected.items():
            stream = bundle.extractfile(name)
            require(stream is not None and stream_sha(stream) == digest, 'archive checksum differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--ref', help='full signed source commit SHA')
    parser.add_argument('--hourly-usd', type=float, default=1.57, help='planning assumption, not a live quote')
    parser.add_argument('--depths', nargs='+', type=int, default=[524288, 1040000])
    parser.add_argument('--window', choices=['final', 'after'], default='final', help='final N document positions, or held-out positions after the saved context')
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--needles', type=int, default=16)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--partition-batch', type=int, default=1024)
    parser.add_argument('--corpus-pins', type=Path, default=Path(__file__).with_name('depth_corpus_pins.json'))
    parser.add_argument('--million-documents', type=int, choices=[2, 3], default=2)
    parser.add_argument('--needle-mode', choices=['sequential', 'independent'], default='sequential')
    parser.add_argument('--handoff-command', type=Path, help='executable receiving each manifest while its sole state exists; transport must finish before return')
    add_disk_floor_argument(parser)
    args = parser.parse_args()
    require(args.depths and all(64 <= d and d+args.rows+767 <= 1048576 for d in args.depths) and
            0 < args.rows <= 65536 and 0 < args.needles <= 64 and args.hourly_usd > 0 and 0 < args.batch <= 8192 and 8192 % args.batch == 0 and
            0 < args.partition_batch <= 8192 and args.batch != args.partition_batch, 'invalid stage dimensions, batches or price')
    if args.handoff_command is not None:
        require(args.handoff_command.is_file() and os.access(args.handoff_command, os.X_OK), 'handoff command must be an existing executable')
    corpus_rows = corpus.documents(args.corpus_pins)
    planning = [estimates(d, args.rows, args.needles, args.hourly_usd, len(corpus.select(corpus_rows, d, args.million_documents)), args.needle_mode) for d in args.depths]
    print(json.dumps(planning, indent=2), flush=True)
    if not args.execute:
        return
    require(args.root is not None and args.ref is not None, '--execute requires --root and --ref')
    gpu = gpu_identity()
    require(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') >= 120_000_000_000, 'GPU stage needs at least 120 GB host RAM for transactional delta staging')
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    require(not occupied, 'GPU has active compute PIDs; leave them untouched and retry when idle')
    source = Path(__file__).resolve().parents[2]
    clean_source(source, args.ref)
    root = study.private_dir(args.root)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root/'tmp'))
    (root/'tmp').mkdir(exist_ok=True)
    # Root is exclusive and cannot contain prior output; failure artifacts are preserved for inspection.
    with exclusive(root/'stage.lock'), exclusive(root/'gpu.lock'):
        require(not (root/'manifest.json').exists(), 'stage output exists; use a fresh root')
        pin = read(source/'tools/hybrid-bench/state_bf16_model_pins.json')['models']['q4']
        cases = sum(p['documents'] for p in planning)
        logits_bytes = cases*3*(args.rows+64)*248320*4
        state_bytes = max(args.depths)*34816+160_000_000
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
        require(available >= state_bytes + pin['files'][0]['size'] + 16*1024**3, 'insufficient available host RAM for delta staging, weights and margin')
        disk(root, pin['files'][0]['size']+state_bytes+logits_bytes+8*1024**3, floor=args.disk_floor_bytes, log=root/'disk.log')
        versions = {name:subprocess.check_output([name, '--version'], text=True, stderr=subprocess.STDOUT).strip() for name in ('cmake', 'nvcc', 'c++', 'git', 'curl')}
        save(root/'manifest.json', dict(source_ref=args.ref, gpu=gpu, python=sys.version, platform=platform.platform(), tools=versions,
             environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')}, estimates=planning, args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, model_pin=pin))
        (root/'models').mkdir()
        models = download_models(dict(models={'q4': pin}), root/'models', args.disk_floor_bytes)
        pins = corpus.fetch(args.corpus_pins, root/'documents')
        build = root/'build'
        require(not build.exists(), 'build exists')
        try:
            with exclusive(root/'build.lock'):
                build_depth(source, build, args.ref, root/'build.log', args.disk_floor_bytes)
            binary = build/'bin/llama-hybrid-depth'
            for tokens in args.depths:
                for i, pin in enumerate(corpus.select(pins, tokens, args.million_documents)):
                    case = study.private_dir(root/f'depth-{tokens}-{pin["id"]}')
                    disk(root, state_bytes+3*(args.rows+64)*248320*4, floor=args.disk_floor_bytes, log=root/'disk.log')
                    text = Path(pin['path'])
                    require(sha(text) == pin['derived_sha256'], 'document changed before preparation')
                    fixture = depth.prepare(binary, models['q4'], text, pin, case, tokens, args.rows, 42+i, args.needles, 14400, args.window, args.needle_mode)
                    depth.run_case(binary, models['q4'], fixture, case, 999, args.batch, args.partition_batch, 14400,
                                   handoff_command=None if args.handoff_command is None else args.handoff_command.resolve(), engine_ref=args.ref)
                    archive_case(case)
                    print(f'done depth={tokens} document={pin["id"]} report={case/"report.md"}', flush=True)
            observations = [read(p) for p in root.glob('depth-*/report.json')]
            save(root/'document-observations.json', dict(observation_unit='document',
                note='Positions and needles within a document are correlated; no pooled-position confidence interval.',
                documents=[dict(depth=r['context_depth'], id=r['document']['id'], kind=r['document']['kind'],
                    local_needle_accuracy=r['local_needle_accuracy'], needle_accuracy=r['needle_accuracy'],
                    all_arms_missed_all_needles=r['all_arms_missed_all_needles'], logit_comparison=r['offload_against_local']) for r in observations]))
            metadata = [p for p in root.rglob('*') if p.is_file() and build not in p.parents and
                        root/'models' not in p.parents and p.suffix not in ('.f32', '.gz')]
            with tarfile.open(root/'stage-metadata.tar.gz', 'x:gz') as bundle:
                sums = ''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(metadata)).encode()
                for path in sorted(metadata):
                    bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
                info = tarfile.TarInfo('SHA256SUMS')
                info.size = len(sums)
                bundle.addfile(info, io.BytesIO(sums))
            verify_bundle(root/'stage-metadata.tar.gz')
            save(root/'stage-archive.json', dict(sha256=sha(root/'stage-metadata.tar.gz'),
                note='Collect this metadata archive, per-case results.tar.gz and the sole per-arm f32 files.'))
            save(root/'complete.json', dict(source_ref=args.ref, stage_archive_sha256=sha(root/'stage-metadata.tar.gz'),
                reports={str(p.relative_to(root)):sha(p) for p in root.glob('depth-*/report.json')}))
        finally:
            if build.exists():
                shutil.rmtree(build)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(terminal('depth-quality-stage', main))
