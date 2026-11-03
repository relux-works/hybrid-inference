#!/usr/bin/env python3
"""Pinned Glimmer paired-RULER profile; no network or GPU operations."""
import argparse
import json

from hybrid_eval import require

NATIVE_REF = '18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d'
MODEL = dict(repo='meta-models/Muse-Glimmer-30B-GGUF',
             revision='b44f1072cf75e918a1b82a3a17b45c4389b076e7',
             file='Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf', bytes=16756683904,
             sha256='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e')
CONTEXT = 131072
WINDOW = 2048
DEPTHS = (8192, 32768, 65536, 122880)
ANSWER_RESERVE = 128 + 2


def saved_session_bytes(depth):
    require(depth in DEPTHS and depth + ANSWER_RESERVE <= CONTEXT, 'invalid Glimmer depth')
    # 10 KiB/position plus 64 MiB exceeds the kit's measured 7088*depth+43476268 bytes.
    return 10240 * (depth + ANSWER_RESERVE) + 64 * 1024**2


def validate_services(client, partition, node, depth, saved_bytes):
    require(depth in DEPTHS, 'invalid Glimmer depth')
    require(saved_bytes >= saved_session_bytes(depth), 'Glimmer saved-session bound too small')
    for info, batch in ((client, 64), (partition, 256), (node, 256)):
        require(info['model_sha256'] == MODEL['sha256'], 'Glimmer model pin differs')
        require(info.get('commit') == NATIVE_REF, 'Glimmer native pin differs')
        require(info['context'] == CONTEXT and depth + ANSWER_RESERVE <= info['context'], 'Glimmer context differs')
        require((info['batch'], info['ubatch']) == (batch, batch), 'Glimmer batch/ubatch differs')
        require(info['backend'] == 'cuda', 'Glimmer production profile requires CUDA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', type=int, choices=DEPTHS)
    args = parser.parse_args()
    if args.depth is not None:
        print(saved_session_bytes(args.depth))
    else:
        print(json.dumps(dict(model=MODEL, native_ref=NATIVE_REF, context=CONTEXT, window=WINDOW,
                              recurrent=False, depths=DEPTHS, seed=261102, samples_per_task=30,
                              slots={d: dict(saved_session_bytes=saved_session_bytes(d),
                                             disk_floor_bytes=2*saved_session_bytes(d)+1024**3) for d in DEPTHS}), indent=2))


if __name__ == '__main__':
    main()
