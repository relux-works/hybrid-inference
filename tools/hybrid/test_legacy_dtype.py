#!/usr/bin/env python3
"""CPU legacy-harness admission controls using an existing local model; no downloads."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess


def invoke(command, log, expected=0):
    with log.open('wb') as output:
        process = subprocess.Popen([str(v) for v in command], stdout=output, stderr=subprocess.STDOUT)
        record = dict(command=[str(v) for v in command], pid=process.pid)
        log.with_suffix('.process.json').write_text(json.dumps(record)+'\n')
        code = process.wait()
        log.with_suffix('.process.json').write_text(json.dumps(dict(record, exit_code=code))+'\n')
    if code != expected:
        raise RuntimeError(f'{log.name}: expected exit {expected}, got {code}, PID {process.pid}')
    return dict(command=[str(v) for v in command], pid=process.pid, exit_code=code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True, help='new private output directory')
    args = parser.parse_args()
    args.artifacts.mkdir(mode=0o700)
    binary = args.build_dir.resolve()/'bin'
    root = args.artifacts.resolve()
    evidence = [invoke([binary/'llama-hybrid-check', args.model, root/'golden', '512'], root/'generate.log')]
    shutil.copytree(root/'golden', root/'label-plant')
    # This header-only plant checks refusal before expansion. Valid bf16 payloads are covered by check-append --bf16.
    for path in (root/'label-plant').glob('*.delta'):
        wire = path.read_bytes()
        length, = struct.unpack('<I', wire[:4])
        header = json.loads(wire[4:4+length]); header['state_dtype'] = 'bf16'
        text = json.dumps(header, separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<I', len(text))+text+wire[4+length:])
    # Each actual legacy binary accepts the native f32 control and rejects a bf16 label with the intended reason.
    for role in ('client', 'distributions', 'guards', 'append'):
        for label in ('golden', 'label-plant'):
            stem = root/label/'prompt-512'
            if role == 'client':
                command = [binary/'llama-hybrid-check-client', args.model, '--identity', stem]
            elif role == 'distributions':
                command = [binary/'llama-hybrid-check-distributions', args.model, str(stem)+'.tokens.json']
            else:
                command = [binary/('llama-hybrid-check-'+role), args.model, stem]
            log = root/(role+'-'+label+'.log')
            row = invoke(command, log, 0 if label == 'golden' else 1)
            if label == 'label-plant' and 'FAIL state dtype mismatch' not in log.read_text():
                raise RuntimeError('wrong refusal: '+log.name)
            evidence.append(dict(row, claim='f32 positive control' if label == 'golden' else 'bf16 label refused before expansion'))
            print('PASS test_legacy_'+role+'_'+label, flush=True)
    (root/'results.json').write_text(json.dumps(evidence, indent=2)+'\n')


if __name__ == '__main__':
    main()
