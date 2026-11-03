#!/usr/bin/env python3
"""One six-point import calibration with a pre-dispatch traffic budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/hybrid-bench'))
sys.path.insert(0, str(Path(__file__).parent))
import growing_session as g
import timing
from bench_common import Refusal, read_private, private_json, require, sha256, http
from proxy import TrafficBudgetExceeded
from mac_timing import helper, import_identity as identity, require_fit_identity

SIZES = [512, 1024, 2048, 4096, 6144, 2048]


def validate_plan(architecture, budget, sizes):
    cap = 3_000_000_000 if architecture == 'muse-glimmer' else 2_000_000_000
    require(type(budget) is int and 0 < budget <= cap, 'calibration budget exceeds model cap')
    require(len(sizes) == 6 and all(type(n) is int and 32 <= n <= 6144 for n in sizes) and
            max(sizes)-min(sizes) >= 1000, 'one six-point repetition with size span >=1000 required')
    return cap


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--r44-packet', type=Path, required=True)
    p.add_argument('--budget-bytes', type=int)
    p.add_argument('--depth', type=int)
    p.add_argument('--sizes', type=int, nargs=6, default=SIZES)
    a = p.parse_args(argv)
    client = read_private(a.client_state / 'client-info.json')
    config = read_private(a.client_state / 'proxy-config.json')
    architecture = client['model_layout']['architecture']
    budget = a.budget_bytes if a.budget_bytes is not None else (3_000_000_000 if architecture == 'muse-glimmer' else 2_000_000_000)
    validate_plan(architecture, budget, a.sizes)
    depth = a.depth if a.depth is not None else (76000 if architecture == 'muse-glimmer' else 100000)
    require(not a.output.exists() and a.output.parent.is_dir(), 'new calibration output required')
    require(sha256(client['model_path']) == client['model_sha256'], 'calibration model changed')
    binding = identity(client)
    require(all(binding['binaries'].get(name) == digest for name, digest in client['binaries'].items()
                if name in binding['binaries']), 'calibration binary changed')
    a.output.mkdir(mode=0o700)
    timing.start()
    events = g.Events(None)
    config = dict(config, benchmark=True, traffic_budget_bytes=budget, prefill_timeout=3600, decode_timeout=3600)
    app = g.MeasuredProxy(config, str(a.output / 'proxy.sqlite'), events, set())
    app.cleanup_prefill = config['prefill_url']
    result = dict(schema='hybrid-bounded-import-calibration-v1', status='running', repetitions=1, sizes=a.sizes,
                  depth=depth, budget_bytes=budget, points=[], calls=[], spans=[], **binding)
    failure = None
    try:
        with g.proxy_server(app) as url:
            framing = g.template_framing(app)
            require(depth + sum(a.sizes) + 6 + 2048 <= client['context'], 'calibration exceeds context margin')
            require(min(a.sizes) > framing, 'calibration delta must exceed template framing')
            messages = g.base_messages(app, depth)
            credential = config['client_credential']
            sid = None
            for index, size in enumerate([0] + a.sizes):
                if index:
                    messages += [dict(role='user', content=' x' * (size-framing))]
                with timing.span('calibration.call', call_id='calibration-' + str(index), origin='critical'):
                    sid, message, meter, ttft, total = g.stream_chat(url, credential,
                        dict(messages=messages, max_tokens=1, temperature=0, ignore_eos=True,
                             chat_template_kwargs=dict(enable_thinking=False)), 'offload' if index else 'local', sid, 3600)
                result['calls'].append(dict(index=index, meter=meter, ttft_seconds=ttft, total_seconds=total))
                require(index == 0 or (meter['path'] == 'offload' and not meter['fallback_reason']), 'calibration offload did not import')
                messages += [message]
                if index:
                    chunks = meter['chunks']
                    require([c['status'] for c in chunks] == ['staged']*(len(chunks)-1)+['imported'], 'incomplete calibration import')
                    result['points'].append(dict(tokens=sum(c['n1']-c['n0'] for c in chunks), seconds=meter['import_ms']/1000,
                        depth=meter['depth_before'], wire_bytes=meter['wire_bytes'],
                        stage_seconds=sum(c['import_ms'] for c in chunks if c['status']=='staged')/1000,
                        final_save_apply_seconds=chunks[-1]['import_ms']/1000))
            fit = helper(a.r44_packet, 'fit-import').fit(result['points'])
            fit.update(binding, source_head=subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip(),
                       calibration_trace='calibration.json', limitation='One bounded six-point pass; local warm prefix, depth grows between points; no independent conditioning-depth coefficient')
            result['status'] = 'passed'
            result['fit'] = fit
    except BaseException as error:
        result.update(status='refused', error_type=type(error).__name__, refusal=str(error) if isinstance(error,Refusal) else None)
        failure = error
    finally:
        try:
            remote = app.remote(next(iter(config['clients'].values())))
            result['native_node_checkpoint'] = remote.request('POST', '/v1/bench/trace-flush')
            result['native_client_checkpoint'] = http(config['decode_url'], '/slots/0/info?trace_flush=1', config['decode_token'])
            require(result['native_node_checkpoint'].get('complete') and result['native_client_checkpoint'].get('complete'),
                    'native calibration trace incomplete')
        except Exception as error:
            result.update(status='refused', trace_error_type=type(error).__name__)
            failure = failure or error
        result['events'] = events.copy()
        result['traffic'] = app.traffic_budget.status()
        result['spans'] = timing.rows()
        private_json(a.output / 'calibration.json', result)
    require(result['traffic']['charged_bytes'] <= budget and result['traffic']['outstanding_bytes'] == 0, 'calibration accounting exceeded cap')
    if failure is not None:
        raise Refusal('bounded calibration refused; inspect private calibration.json') from failure
    private_json(a.output / 'import-fit.json', result['fit'])
    print(json.dumps(dict(status=result['status'], budget=result['traffic'], fit=str(a.output / 'import-fit.json'))))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Refusal, OSError, ValueError, TrafficBudgetExceeded) as error:
        print('refused: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        raise SystemExit(2)
