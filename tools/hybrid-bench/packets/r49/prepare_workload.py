#!/usr/bin/env python3
"""Select a hash-bound owner window, writing private token IDs and public counts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from session_runtime import select_workload
from bench_common import sha256,require,private_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner-export',required=True,type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--model-sha256',required=True);p.add_argument('--vocab-size',required=True,type=int)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require(a.owner_export.stat().st_size<=256*1024**2 and sha256(a.owner_export)==a.sha256,'owner export size/hash mismatch')
    row=select_workload(json.loads(a.owner_export.read_text()),a.model_sha256,a.vocab_size)
    private_json(a.output,row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('base_ids','turns','source_session')}))


if __name__=='__main__':main()
