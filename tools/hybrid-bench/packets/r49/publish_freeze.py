#!/usr/bin/env python3
"""Publish the inventory receipt before deterministic selection; verify live readback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,sha256,private_json
from session_protocol import prereg_binding
from session_study import amendment_binding,freeze_receipt,verify_publication,read_events,SENDER,ROOM,TOPIC,RECIPIENT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--prereg-commit',required=True);p.add_argument('--amendment-commit',required=True)
    p.add_argument('--study-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'publication receipt cannot overwrite an earlier publication')
    require(sha256(a.population)==a.sha256,'inventory artifact changed')
    frozen=json.loads(a.population.read_text())
    require(frozen['prereg']==prereg_binding(ROOT,a.prereg_commit) and frozen['amendment']==amendment_binding(ROOT,a.amendment_commit),'inventory registration differs')
    receipt=freeze_receipt(frozen,a.sha256,a.study_id)
    for event in read_events():
        if event.get('sender')!=SENDER or event.get('type')!='message':continue
        try:previous=json.loads(event['body'])
        except (ValueError,KeyError):continue
        if previous.get('schema')==receipt['schema'] and previous.get('study_id')==a.study_id:
            require(previous==receipt,'study inventory already published with another hash; no reroll')
            private_json(a.output,verify_publication(receipt,event['event_id']));return
    body=a.output.with_suffix('.body.json')
    if body.exists():require(json.loads(body.read_text())==receipt,'pending publication body differs; no inventory reroll')
    else:private_json(body,receipt)
    event_id=subprocess.check_output(['a2a','--as',SENDER,'post',ROOM,'--topic',TOPIC,'--kind','notice','--to',RECIPIENT,'@'+str(body)],text=True).strip()
    require(event_id.startswith('$') and '\n' not in event_id,'a2a publication did not return a single event ID')
    private_json(a.output,verify_publication(receipt,event_id))
    print(event_id)


if __name__=='__main__':main()
