"""Real-header geometry fixtures; optional native checks use local headers and a bounded header stream."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

import bench_common as common

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/mirai-s-headers.json').read_text())['models']
FINGERPRINT = 'd4fb520322100b1e438d60aa641ee86bd0d22576198a6dc49890c04f181924a8'
BINARY = None
HEADERS = None
ARTIFACTS = None


def fixture(name='mirai-s'):
    meta = copy.deepcopy(FIXTURES[name]['metadata'])
    meta['_tensors'] = {key: (shape, kind) for key, (shape, kind) in meta['_tensors'].items()}
    return meta


def bad_cases():
    base = fixture()
    changes = [
        ('unknown-extra', 'unknown extra tensors', lambda m: m['_tensors'].update({'blk.64.nextn.surprise.weight': ([5120], 0)})),
        ('unknown-trunk-extra', 'unknown extra tensors', lambda m: m['_tensors'].update({'blk.0.surprise.weight': ([5120], 0)})),
        ('missing-first-trunk', 'missing trunk/MTP tensor', lambda m: m['_tensors'].pop('blk.0.attn_norm.weight')),
        ('missing-later-trunk', 'missing trunk/MTP tensor', lambda m: m['_tensors'].pop('blk.63.ffn_up.weight')),
        ('mtp-only', 'missing trunk/MTP tensor', lambda m: m.update(_tensors={n: t for n, t in m['_tensors'].items() if not n.startswith('blk.') or n.startswith('blk.64.')})),
        ('missing-mtp-projection', 'missing trunk/MTP tensor', lambda m: m['_tensors'].pop('blk.64.nextn.eh_proj.weight')),
        ('missing-mtp-query', 'missing trunk/MTP tensor', lambda m: m['_tensors'].pop('blk.64.attn_q.weight')),
        ('bad-mtp-shape', 'wrong tensor shape', lambda m: m['_tensors'].update({'blk.64.nextn.eh_proj.weight': ([5120, 5120], 8)})),
        ('bad-mtp-query-shape', 'wrong tensor shape', lambda m: m['_tensors'].update({'blk.64.attn_q.weight': ([5120, 6144], 8)})),
        ('bad-trunk-shape', 'wrong tensor shape', lambda m: m['_tensors'].update({'blk.3.attn_gate.weight': ([5120, 3072], 92)})),
        ('missing-row-scale', 'invalid Mirai row scale', lambda m: m['_tensors'].pop('blk.0.ffn_up.scale')),
        ('bad-row-scale-shape', 'invalid Mirai row scale', lambda m: m['_tensors'].update({'blk.0.ffn_up.scale': ([1], 0)})),
        ('bad-row-scale-type', 'invalid Mirai row scale', lambda m: m['_tensors'].update({'blk.0.ffn_up.scale': ([17408], 1)})),
        ('missing-rotation', 'invalid Mirai rotation', lambda m: m['_tensors'].pop('mirai.rot.5120')),
        ('bad-rotation-type', 'invalid Mirai rotation', lambda m: m['_tensors'].update({'mirai.rot.5120': ([5145], 1)})),
        ('bad-block-count', 'invalid layout layer count', lambda m: m.update({'qwen35.block_count': 513})),
        ('empty-block-count', 'invalid layout layer count', lambda m: m.update({'qwen35.block_count': 0})),
        ('bad-nextn-count', 'invalid nextn layer count', lambda m: m.update({'qwen35.nextn_predict_layers': 65})),
        ('negative-nextn-count', 'invalid nextn layer count', lambda m: m.update({'qwen35.nextn_predict_layers': -1})),
        ('bad-recurrent-length', 'invalid recurrent layout', lambda m: m.update({'qwen35.attention.recurrent_layers': [True] * 64})),
        ('recurrent-mtp', 'recurrent MTP', lambda m: m['qwen35.attention.recurrent_layers'].__setitem__(64, True)),
        ('bad-head-array', 'invalid raw layer array', lambda m: m.update({'qwen35.attention.head_count': [24] * 64})),
    ]
    for name, error, change in changes:
        meta = copy.deepcopy(base)
        change(meta)
        yield name, error, meta


class MiraiIdentityTests(unittest.TestCase):
    # These are derived from real tensor tables and hparams, with the real ordered-vocabulary hash.
    def test_real_header_layouts(self):
        for name in FIXTURES:
            with self.subTest(model=name):
                identity = common.model_identity_from_metadata(fixture(name))
                self.assertEqual(identity['model_layout']['layer_count'], 64)
                self.assertEqual(identity['layout_fingerprint'], FINGERPRINT)
                self.assertEqual(sum(v['recurrent'] for v in identity['model_layout']['layers']), 48)

    # Every malformed case has the valid combined file as a positive control.
    def test_refusals(self):
        common.model_identity_from_metadata(fixture())
        for name, error, meta in bad_cases():
            with self.subTest(case=name), self.assertRaisesRegex(common.Refusal, error):
                common.model_identity_from_metadata(meta)

    # No tail tensors is valid; a partial or misspelled tail is not treated as a trunk-only file.
    def test_trunk_only(self):
        meta = fixture()
        meta['_tensors'] = {n: t for n, t in meta['_tensors'].items() if not n.startswith('blk.64.')}
        self.assertEqual(common.model_identity_from_metadata(meta)['layout_fingerprint'], FINGERPRINT)
        meta['qwen35.attention.head_count'] = [24] * 65
        self.assertEqual(common.model_identity_from_metadata(meta)['layout_fingerprint'], FINGERPRINT)

    # MTP contributes neither cache rows nor recurrent memory; the raw metadata table still has 65 entries.
    def test_trunk_geometry(self):
        info = dict(available_memory_bytes=64 * common.GIB)
        plans = []
        with tempfile.NamedTemporaryFile() as model:
            for name in ('qwen3.8-27b-bf16', 'mirai-s'):
                with mock.patch.object(common, 'model_metadata', return_value=fixture(name)):
                    plans.append(common.context_plan(model.name, info, 1024))
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[0]['kv_bytes_per_token'], 34816)
        self.assertEqual(plans[0]['recurrent_state_bytes'], 156893184)

    # Matching layout enables only the explicit hetero mode; hash and mismatched-layout refusals remain active.
    def test_hetero_admission(self):
        stock = common.model_identity_from_metadata(fixture('qwen3.8-27b-bf16'))
        mirai = common.model_identity_from_metadata(fixture())
        with self.assertRaisesRegex(common.Refusal, 'sha256 mismatch'):
            common.validate_identity('a' * 64, mirai, 'b' * 64, stock)
        common.validate_identity('a' * 64, mirai, 'b' * 64, stock, True)
        meta = fixture('qwen3.8-27b-bf16')
        meta['qwen35.attention.head_count_kv'] = 8
        # A changed geometry cannot retain the old fingerprint.
        changed = common.model_identity_from_metadata(meta)
        with self.assertRaisesRegex(common.Refusal, 'fingerprint mismatch'):
            common.validate_identity('a' * 64, changed, 'b' * 64, stock, True)

    # This base supports only Qwen35 in Python admission; this change does not admit other architectures.
    def test_other_architectures_keep_refusal(self):
        for arch in ('gemma4', 'muse-glimmer', 'qwen35moe'):
            meta = fixture()
            meta['general.architecture'] = arch
            with self.subTest(arch=arch), self.assertRaisesRegex(common.Refusal, 'dense qwen35 only'):
                common.model_identity_from_metadata(meta)


def scalar(source, kind):
    return struct.unpack('<' + kind, source.read(struct.calcsize('<' + kind)))[0]


def string(source):
    return source.read(scalar(source, 'Q')).decode('utf-8')


def skip_value(source, kind):
    formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    if kind in formats:
        source.seek(struct.calcsize(formats[kind]), 1)
    elif kind == 8:
        source.seek(scalar(source, 'Q'), 1)
    elif kind == 9:
        subtype, count = scalar(source, 'I'), scalar(source, 'Q')
        for _ in range(count):
            skip_value(source, subtype)
    else:
        raise ValueError('invalid fixture metadata type')


def encoded_string(value):
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def header_fixture(source_path, destination, meta):
    """Retain exact tokenizer bytes; replace geometry for the native no_alloc header stream."""
    source = io.BytesIO(source_path.read_bytes())
    source.seek(16)
    count = scalar(source, 'Q')
    records = []
    for _ in range(count):
        start = source.tell()
        key, kind = string(source), scalar(source, 'I')
        skip_value(source, kind)
        end = source.tell()
        if key.startswith('qwen35.') and key in meta:
            value = meta[key]
            if isinstance(value, list):
                subtype = 7 if type(value[0]) is bool else 4
                data = struct.pack('<IQ', subtype, len(value))
                data += b''.join(struct.pack('<?' if subtype == 7 else '<I', v) for v in value)
                kind = 9
            else:
                # Signed negatives test the loader's typed key rejection, not unsigned wraparound.
                kind = 5 if type(value) is int and value < 0 else kind
                data = struct.pack({4: '<I', 5: '<i', 6: '<f', 7: '<?'}[kind], value)
            records.append(encoded_string(key) + struct.pack('<I', kind) + data)
        else:
            source.seek(start)
            records.append(source.read(end - start))
        source.seek(end)
    types = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 14: (256, 210),
             30: (1, 2), 90: (64, 17), 91: (64, 18), 92: (128, 50), 93: (128, 49)}
    tensors = []
    offset = 0
    for name, (dims, kind) in meta['_tensors'].items():
        tensors.append(encoded_string(name) + struct.pack('<I', len(dims)) +
                       struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, offset))
        block, size = types.get(kind, (1, 4))
        elements = 1
        for dim in dims:
            elements *= dim
        offset += (elements // block * size + 31) // 32 * 32
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(records)) + b''.join(records + tensors)
    data_offset = (len(header) + 31) // 32 * 32
    with destination.open('wb') as target:
        target.write(header)
        target.write(bytes(data_offset - len(header)))


@unittest.skipUnless(BINARY and HEADERS, 'supply --binary and --headers for native no_alloc checks')
class NativeHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, spec in FIXTURES.items():
            path = HEADERS / (name + '.header.gguf')
            if path.stat().st_size != spec['header_bytes'] or common.sha256(path) != spec['header_sha256']:
                raise ValueError('header source identity mismatch: ' + name)

    def native(self, meta, name='mirai-s', case='valid', expected=None):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.gguf'
            header_fixture(HEADERS / (name + '.header.gguf'), path, meta)
            result = subprocess.run([str(BINARY), '--header-only', str(path)], capture_output=True, text=True, timeout=45)
            if ARTIFACTS:
                (ARTIFACTS / (case + '.log')).write_text(result.stderr + result.stdout + f'\nexit={result.returncode}\n')
            self.assertEqual(result.returncode, 0 if expected is None else 1, result.stderr[-3000:])
            if expected is not None:
                self.assertRegex(result.stderr, expected)
                self.assertEqual(result.stdout, '')
                return None
            identity = json.loads(result.stdout)
            self.assertEqual(identity['model_layout']['layer_count'], 64)
            self.assertEqual(identity['layout_fingerprint'], FINGERPRINT)
            return identity

    # Exact, unmodified header bytes must pass the newer GGUF parser as well as the native model loader.
    def test_exact_header_python_equality(self):
        for name in FIXTURES:
            with self.subTest(model=name):
                path = HEADERS / (name + '.header.gguf')
                result = subprocess.run([str(BINARY), '--header-only', str(path)], capture_output=True, text=True, timeout=45)
                if ARTIFACTS:
                    (ARTIFACTS / ('exact-' + name + '.log')).write_text(result.stderr + result.stdout + f'\nexit={result.returncode}\n')
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
                native, python = json.loads(result.stdout), common.model_identity(path)
                self.assertEqual(native, {key: python[key] for key in native})
                self.assertEqual(native['layout_fingerprint'], FINGERPRINT)

    # Full native loaders validate every real name/shape/type without reading weights or enabling MTP.
    def test_native_python_equality(self):
        for name in FIXTURES:
            with self.subTest(model=name):
                meta = fixture(name)
                native = self.native(meta, name, name)
                python = common.model_identity_from_metadata(meta)
                self.assertEqual(native, {key: python[key] for key in native})

    # Every refusal must be a normal loader error with the intended diagnostic and no identity output.
    def test_native_refusals(self):
        self.native(fixture(), case='negative-control-valid')
        for name, _, meta in bad_cases():
            if name in ('bad-row-scale-type', 'bad-rotation-type'):
                error = 'invalid Mirai .*type|invalid Mirai row scale'
            elif name in ('bad-block-count', 'empty-block-count'):
                error = 'invalid block count'
            elif name.startswith('bad-nextn'):
                error = 'invalid nextn layer count'
            elif name == 'negative-nextn-count':
                error = 'wrong type|invalid nextn layer count'
            elif name == 'recurrent-mtp':
                error = 'recurrent MTP'
            elif name in ('bad-recurrent-length', 'bad-head-array'):
                error = 'wrong array length'
            elif name.startswith('unknown'):
                error = 'wrong number of tensors'
            elif name.startswith('missing') or name == 'mtp-only':
                error = 'not found'
            else:
                error = 'wrong shape'
            with self.subTest(case=name):
                self.native(meta, case=name, expected=error)

    # A trunk-only export may retain its nextn declaration, but every remaining tensor must still be accounted for.
    def test_native_trunk_only(self):
        meta = fixture()
        meta['_tensors'] = {n: t for n, t in meta['_tensors'].items() if not n.startswith('blk.64.')}
        self.native(meta, case='trunk-only')
        meta['qwen35.attention.head_count'] = [24] * 65
        self.native(meta, case='trunk-only-raw-head-array')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--headers', type=Path)
    parser.add_argument('--artifacts', type=Path)
    args, remaining = parser.parse_known_args()
    BINARY, HEADERS, ARTIFACTS = args.binary, args.headers, args.artifacts
    # unittest evaluates class decorators at import time.
    NativeHeaderTests.__unittest_skip__ = not (BINARY and HEADERS)
    if ARTIFACTS:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=['test_mirai_identity.py'] + remaining)
