#!/usr/bin/env python3
"""Freeze a complete population before deterministic preregistered selection."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,sha256,private_json
from session_protocol import prereg_binding,freeze_export,select_set,digest
from session_study import amendment_binding,freeze_receipt,verify_publication


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('freeze','select'))
    p.add_argument('--input',required=True,type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--prereg-commit',required=True);p.add_argument('--amendment-commit',required=True)
    p.add_argument('--study-id',required=True);p.add_argument('--published-event')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model-sha256');p.add_argument('--vocab-size',type=int)
    a=p.parse_args();require(not a.output.exists(),'frozen population/session set must not be overwritten')
    require(a.input.stat().st_size<=512*1024**2 and sha256(a.input)==a.sha256,'trusted input size/hash differs')
    prereg=prereg_binding(ROOT,a.prereg_commit);amendment=amendment_binding(ROOT,a.amendment_commit);source=json.loads(a.input.read_text())
    if a.phase=='freeze':
        require(a.model_sha256 is not None and a.vocab_size is not None,'model and vocabulary required')
        result=freeze_export(source,a.sha256,a.model_sha256,a.vocab_size,prereg)
        result['amendment']=amendment
        public={k:v for k,v in result.items() if k not in ('eligible',)}
    else:
        require(source['prereg']==prereg and source.get('amendment')==amendment,'frozen population registration differs')
        require(a.published_event,'published freeze event required before selection')
        publication=verify_publication(freeze_receipt(source,a.sha256,a.study_id),a.published_event)
        result=select_set(source);result.update(amendment=amendment,freeze_publication=publication)
        public=dict(prereg=prereg,frozen_population_digest=result['frozen_population_digest'],
            sets={name:{k:v for k,v in rows.items() if k!='sessions'} for name,rows in result['sets'].items()})
    private_json(a.output,result);public.update(amendment=amendment,study_id=a.study_id,artifact_sha256=sha256(a.output),canonical_digest=digest(result))
    private_json(a.output.with_suffix('.receipt.json'),public);print(json.dumps(public))


if __name__=='__main__':main()
