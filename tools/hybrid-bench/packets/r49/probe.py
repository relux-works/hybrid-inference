#!/usr/bin/env python3
"""Probe the real authenticated HTTPS path before a production session."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from session_measurement import probe
from bench_common import read_private,private_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--client-state',required=True,type=Path)
    p.add_argument('--ca-file',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--link-label',required=True);a=p.parse_args()
    config=read_private(a.client_state/'proxy-config.json')
    token=next(iter(config['clients'].values()))['prefill_token']
    result=probe(config['prefill_url'],token,str(a.ca_file.resolve()))
    result.update(ca_sha256=sha256(a.ca_file),link_label=a.link_label)
    private_json(a.output,result)
    print('payload MB/s: '+str(result['bytes_per_second']/1e6))


if __name__=='__main__':main()
