#!/usr/bin/env python3
"""Bind PRI31 rates, measured HTTPS rate and selected-dtype import costs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import read_private,private_json,sha256,require
from growing_fixture import validate_fixture
from growing_session import bf16_policy
from policy import Policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('client-state','fixture','import-fit','probe','output'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--server-rate',required=True,type=float);p.add_argument('--server-rate-source',required=True)
    a=p.parse_args();client=read_private(a.client_state/'client-info.json');config=read_private(a.client_state/'proxy-config.json')
    fixture,fit,probe=[json.loads(v.read_text()) for v in (a.fixture,a.import_fit,a.probe)]
    require(fit['state_dtype'] in ('bf16','f32'),'explicit import profile required')
    local=validate_fixture(fixture,client['model_sha256'],client['model_layout']['architecture'])
    control=sorted(r['health_seconds'] for r in probe['samples'])[len(probe['samples'])//2]
    geometry=config['delta_geometry']
    baseline=dict(local_rates=local,server_rate=a.server_rate,link_rate=probe['bytes_per_second'],fixed=2*control,
        import_fixed=fit['fixed_s'],import_per_token=fit['seconds_token'],kv_bytes=geometry['kv_bytes'],state_bytes=geometry['state_bytes'])
    Policy(**baseline)
    live=bf16_policy(client['model_layout'],baseline) if fit['state_dtype']=='bf16' else baseline
    private_json(a.output,baseline);private_json(a.output.with_suffix('.live.json'),live)
    private_json(a.output.with_suffix('.sources.json'),dict(server_rate=a.server_rate,server_rate_source=a.server_rate_source,
        fixture_sha256=sha256(a.fixture),import_fit_sha256=sha256(a.import_fit),probe_sha256=sha256(a.probe),
        control_seconds=control,wire_profile=fit['state_dtype'],limitation='Server rate is an explicitly sourced prior unless separately calibrated; live node compute is recorded.'))


if __name__=='__main__':main()
