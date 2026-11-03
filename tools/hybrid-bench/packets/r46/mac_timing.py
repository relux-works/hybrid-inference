#!/usr/bin/env python3
"""PRI-31 supervisor payload: admit a rate fixture, measure the link, then run two timing arms."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/hybrid-bench'))
sys.path.insert(0, str(Path(__file__).parent))
import growing_session as g
from growing_fixture import validate_fixture
from bench_common import Refusal, read_private, private_json, require, sha256
from journal import Renderer
from proxy import Node
from types import SimpleNamespace


def helper(packet, name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), packet / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IDENTITY_FILES = ['tools/hybrid-proxy/proxy.py', 'tools/hybrid-proxy/traffic.py', 'tools/hybrid-proxy/timing.py',
                  'src/llama-context.cpp', 'src/llama-kv-cache.cpp', 'src/llama-kv-cache-iswa.cpp',
                  'src/llama-io.cpp', 'src/llama-io.h', 'src/llama-mmap.cpp', 'tools/hybrid/hybrid-state.cpp',
                  'tools/server/server-context.cpp', 'tools/server/server-http.cpp', 'tools/server/server-task.h']


def import_identity(client):
    binaries = {}
    for name in client['binaries']:
        path = Path(client['build_dir']) / 'bin' / name
        require(path.is_file(), 'missing import binary: ' + name)
        binaries[name] = sha256(path)
    return dict(import_path_sha256={name: sha256(ROOT / name) for name in IDENTITY_FILES}, binaries=binaries,
                build_record=client.get('build_record'), model_sha256=client['model_sha256'])


def require_fit_identity(fit, client):
    current = import_identity(client)
    require(all(fit.get(key) == current[key] for key in current),
            'import fit source/binary identity mismatch; bounded calibration required')


def admission(client, node, geometry, fixture, fit, head):
    rates = validate_fixture(fixture, client['model_sha256'], client['model_layout']['architecture'])
    require(fit['source_head'] == head and fit['model_sha256'] == client['model_sha256'], 'import fit identity mismatch')
    require_fit_identity(fit, client)
    require(client['backend'] == 'metal' and node['backend'] == 'cuda', 'Mac packet requires Metal client and CUDA node')
    architecture = client['model_layout']['architecture']
    require(architecture in ('qwen35', 'muse-glimmer'), 'unsupported packet model')
    capacity = 131072 if architecture == 'muse-glimmer' else 196608
    require(client['context'] == node['context'] == capacity, 'packet context mismatch')
    if architecture == 'muse-glimmer':
        require(client['model_layout']['sliding_window'] == 2048 and geometry['state_bytes'] == 0,
                'Glimmer window/state mismatch')
    return rates, 76000 if architecture == 'muse-glimmer' else 100000


def budgets(client, geometry, plan):
    size = geometry['state_bytes'] + geometry.get('window_bytes', 0) + (geometry['kv_bytes'] + 32) * client['context'] + 1024**2
    model = Path(client['model_path']).stat().st_size
    # Capture-free runs retain bases and rollback slots only; reserve six capacity states and private logs/archive.
    return dict(disk_free_bytes=max(80_000_000_000 if geometry['state_bytes'] else 45_000_000_000,
                                   28_000_000_000 + 6 * size + 4_000_000_000),
                ram_required_bytes=2 * model + 5 * size + 8 * 1024**3,
                state_capacity_bytes=size, final_depth=plan['final_depth'],
                formula='disk=max(model-specific floor,28GB+6*S(context)+4GB); RAM=2*weights+5*S(context)+8GiB < 0.8*physical',
                limitation='Models/build already staged; 4GB covers logs plus selective archive, no state archive')


def command(state, output, policy, fixture, depth, link):
    return [sys.executable, str(ROOT / 'tools/hybrid-bench/growing_session.py'), '--client-state', str(state),
        '--output', str(output), '--depth', str(depth), '--turns', '40', '--seed', '1', '--start', 'pre-staged',
        '--checkpoint-turns', 'none', '--timing-spans', '--disconnect-turn', '2', '--disconnect-turns', '3',
        '--policy', str(policy), '--rate-fixture', str(fixture), '--link-label', link + '; M3 Max PRI31; CUDA to Metal; SSH',
        '--request-timeout', '3600']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-state', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--fixture', required=True, type=Path)
    p.add_argument('--import-fit', required=True, type=Path)
    p.add_argument('--r44-packet', required=True, type=Path)
    p.add_argument('--link', required=True, choices=['5G', 'Wi-Fi'])
    p.add_argument('--link-port', type=int, default=8093)
    a = p.parse_args(argv)
    a.client_state, a.output, a.fixture = (v.resolve() for v in (a.client_state, a.output, a.fixture))
    require(not a.output.exists() and a.output.parent.is_dir(), 'new output in existing parent required')
    client = read_private(a.client_state / 'client-info.json')
    config = read_private(a.client_state / 'proxy-config.json')
    node = json.loads((a.client_state / 'node-info.json').read_text())
    fixture = json.loads(a.fixture.read_text())
    fit = json.loads(a.import_fit.read_text())
    head = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    rates, depth = admission(client, node, config['delta_geometry'], fixture, fit, head)
    # Refuse missing/placeholder rates before link traffic, decoder shutdown, or measurement artifacts.
    priority = subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'pri=,nice='], text=True).split()
    require([int(v) for v in priority] == [31, 0], 'existing PRI-31 supervisor required')
    link_helper = helper(a.r44_packet, 'node-link')
    points = link_helper.measure('127.0.0.1', a.link_port)
    link = link_helper.summarize(points, node, '100k')
    policy = {k: link[k] for k in ('server_rate', 'link_rate', 'fixed', 'kv_bytes', 'state_bytes')}
    if fixture['architecture'] == 'muse-glimmer':
        policy['server_rate'] = 2823
    policy.update(local_rates=rates, import_fixed=fit['fixed_s'], import_per_token=fit['seconds_token'])
    policy_object = g.Policy(**policy)
    renderer = Renderer(Node(config['decode_url'], config['decode_token'], 120))
    framing = g.template_framing(SimpleNamespace(renderer=renderer))
    plan = g.depth_plan(depth, 40, 1, 10875, 4506, 340, framing, client['context'])
    budget = budgets(client, config['delta_geometry'], plan)
    physical = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    require(budget['ram_required_bytes'] < .8 * physical, 'RAM admission refused')
    for path in (a.output.parent, a.client_state, Path(client['build_dir'])):
        require(shutil.disk_usage(path).free > budget['disk_free_bytes'], 'disk admission refused: ' + str(path))
    admission_path = a.output.with_name(a.output.name + '-admission.json')
    private_json(admission_path, dict(utc=datetime.now(timezone.utc).isoformat(), source_head=head,
        fixture_sha256=sha256(a.fixture), import_fit_sha256=sha256(a.import_fit), physical_link=a.link,
        raw_link_points=points, policy=policy, budgets=budget, framing=framing,
        server_rate_source='FACTS 1a Glimmer118k Delta2000' if fixture['architecture'] == 'muse-glimmer' else 'FACTS 1a Qwen100k Delta2000',
        policy_limit='scalar node rate; depth-dependent import cost not represented'))
    # Production policy may choose all-local turns on a slow link; fidelity qualification still requires both routes.
    g.route_preflight(policy_object, plan, 2, 3, require_mix=False)
    policy_path = a.output.with_name(a.output.name + '-policy.json')
    private_json(policy_path, policy)
    # Stop only the launcher named by this metadata; the driver starts its own decoder.
    stop = subprocess.run([str(ROOT / 'tools/hybrid-bench/setup-client.sh'), '--state', str(a.client_state), '--stop'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(stop.returncode == 0, 'owned launcher stop failed')
    execute = helper(a.r44_packet, 'run-session').execute
    return execute(command(a.client_state, a.output, policy_path, a.fixture, depth, a.link), a.output)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Refusal, OSError, ValueError, KeyError) as error:
        print('refused: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        raise SystemExit(2)
