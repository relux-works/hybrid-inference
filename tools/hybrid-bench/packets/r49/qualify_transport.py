#!/usr/bin/env python3
"""Qualify each block by the unchanged tc grid or independent real-path T/R/T."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,read_private,private_json,sha256
from session_measurement import probe
from session_study import verify_qualification,replace_private_json,amendment_binding
from session_protocol import prereg_binding

PAYLOAD=384*1024**2


def physical(interface):
    result=subprocess.check_output(['netstat','-ib','-I',interface],text=True)
    require(interface in result,'physical interface counters missing')
    return result


def reference(url,ca,header,interface):
    require(urlsplit(url).scheme=='https','reference requires authenticated TLS')
    require(header.stat().st_mode & 0o077==0,'reference header file must be private')
    def fetch(index):
        n=PAYLOAD//8
        command=['curl','--silent','--show-error','--fail','--proto','=https','--max-time','1800',
            '--cacert',str(ca),'--header','@'+str(header),'--interface',interface,
            '--range',f'{index*n}-{(index+1)*n-1}','--output','/dev/null',
            '--write-out','%{http_code} %{size_download} %{time_total}',url]
        run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        require(run.returncode==0,'independent curl reference failed; credentials are not logged')
        status,count,seconds=run.stdout.split()
        require(status=='206' and int(count)==n,'independent reference truncated range or status differs')
        return dict(http_code=int(status),bytes=int(count),seconds=float(seconds))
    with ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(fetch,range(8)))


def real_trials(node_url,token,ca,reference_url,header,interface,progress=None):
    trials=[]
    for tool in ('T','R','T'):
        before=physical(interface);start=time.monotonic_ns()
        if tool=='T':
            raw=probe(node_url,token,str(ca),8,3)
            count=sum(r['bytes'] for r in raw['samples'])
        else:
            raw=reference(reference_url,ca,header,interface);count=sum(r['bytes'] for r in raw)
        end=time.monotonic_ns();after=physical(interface)
        trials.append(dict(tool=tool,payload_bytes=count,seconds=(end-start)/1e9,
            start_ns=start,end_ns=end,physical_before=before,physical_after=after,raw=raw))
        if progress:progress(trials)
    return trials


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prereg-commit',required=True);p.add_argument('--amendment-commit',required=True)
    p.add_argument('--method',choices=('tc','real-path'),required=True)
    for key in ('client-state','ca-file','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('series-id','block','side','link-series','link-label'):p.add_argument('--'+key,required=True)
    p.add_argument('--tc-results',type=Path);p.add_argument('--reference-url');p.add_argument('--reference-header',type=Path)
    p.add_argument('--interface');a=p.parse_args()
    require(not a.output.exists(),'qualification must not overwrite an earlier trial')
    config=read_private(a.client_state/'proxy-config.json');node_url=config['prefill_url']
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    result=dict(prereg=prereg_binding(ROOT,a.prereg_commit),amendment=amendment_binding(ROOT,a.amendment_commit),
        schema='hybrid-e2e-transport-qualification-v1',method=a.method,series_id=a.series_id,
        block=a.block,side=a.side,link_series=a.link_series,link_label=a.link_label,
        source_head=source,node_url=node_url,ca_sha256=sha256(a.ca_file),started_utc=datetime.now(timezone.utc).isoformat())
    if a.method=='tc':
        require(a.tc_results is not None,'full VM tc RESULTS.json required')
        result['results']=json.loads(a.tc_results.read_text())
        require(result['results'].get('source_head')==source,'tc qualification source differs')
        result['tc_results_sha256']=sha256(a.tc_results)
    else:
        require(sys.platform=='darwin' and a.interface and a.reference_url and a.reference_header,'actual Mac interface and independent reference required')
        node=urlsplit(node_url);ref=urlsplit(a.reference_url)
        require(node.hostname==ref.hostname and node.port!=ref.port,'reference must use same node with an independent server port')
        route=subprocess.check_output(['route','-n','get',node.hostname],text=True)
        require(('interface: '+a.interface) in route,'reference interface differs from transport route')
        result.update(reference_implementation='curl-ranges/nginx-static',reference_url=a.reference_url,
            interface=a.interface,route=route)
        def progress(trials):
            result.update(trials=trials,qualified=False);replace_private_json(a.output,result)
        try:
            result['trials']=real_trials(node_url,next(iter(config['clients'].values()))['prefill_token'],a.ca_file,a.reference_url,a.reference_header,a.interface,progress)
        except BaseException as error:
            result.update(qualified=False,error_type=type(error).__name__);replace_private_json(a.output,result);raise
    try:
        verify_qualification(result,a.method,a.link_series,a.link_label);result['qualified']=True
    except BaseException:
        result['qualified']=False;replace_private_json(a.output,result);raise
    result['ended_utc']=datetime.now(timezone.utc).isoformat();replace_private_json(a.output,result)
    print('transport qualified: '+a.method)


if __name__=='__main__':main()
