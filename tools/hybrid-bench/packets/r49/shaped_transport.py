#!/usr/bin/env python3
"""CPU HTTPS throughput qualification in an isolated tc-shaped Linux namespace."""
import argparse
import http.server
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from session_measurement import probe
from parallel_transport import Gateway


class Health(http.server.BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def do_GET(self):
        body=b'{}';self.send_response(200);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(body)


def command(argv):
    return subprocess.run(argv,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True).stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--inside',action='store_true');a=parser.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not a.inside:
        flags=['--net'] if os.geteuid()==0 else ['--user','--map-root-user','--net']
        argv=['unshare',*flags,sys.executable,str(Path(__file__).resolve()),'--inside','--output',str(a.output)]
        with (a.output/'namespace.log').open('wb') as log:
            result=subprocess.run(argv,stdout=log,stderr=log)
        if result.returncode:
            (a.output/('FAILED.json' if (a.output/'RESULTS.json').exists() else 'BLOCKED.json')).write_text(json.dumps(dict(status='unqualified',namespace_exit=result.returncode,
                reason='Qualification did not pass; inspect namespace.log and any RESULTS.json. No unshaped fallback is accepted.'))+'\n')
        return result.returncode
    if sys.platform!='linux' or os.readlink('/proc/self/ns/net')==os.readlink('/proc/1/ns/net'):
        raise RuntimeError('refusing to shape the host network namespace')
    command(['ip','link','set','lo','up'])
    cert,key=a.output/'cert.pem',a.output/'key.pem'
    command(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost',
        '-addext','subjectAltName=IP:127.0.0.1,DNS:localhost','-keyout',str(key),'-out',str(cert)])
    token=secrets.token_hex(32)
    upstream=http.server.ThreadingHTTPServer(('127.0.0.1',0),Health)
    gateway=Gateway(('127.0.0.1',0),'http://127.0.0.1:'+str(upstream.server_port),[token],cert,key)
    threads=[threading.Thread(target=s.serve_forever,daemon=True) for s in (upstream,gateway)]
    for t in threads:t.start()
    rows=[]
    try:
        for rate in (15,20):
            for rtt in (50,100,150):
                subprocess.run(['tc','qdisc','del','dev','lo','root'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                command(['tc','qdisc','add','dev','lo','root','handle','1:','netem','delay',str(rtt/2)+'ms','limit','20000'])
                command(['tc','qdisc','add','dev','lo','parent','1:1','handle','10:','tbf','rate',str(rate*8)+'mbit','burst','256kb','limit','8mb'])
                result=probe('https://127.0.0.1:'+str(gateway.server_port),token,str(cert),8,3)
                throughput=result['bytes_per_second']/1e6
                row=dict(rate_MB_s=rate,rtt_ms=rtt,payload_MB_s=throughput,capacity_fraction=throughput/rate,
                    passed=throughput>=.8*rate,probe=result,qdisc=json.loads(command(['tc','-s','-j','qdisc','show','dev','lo'])))
                rows.append(row)
                (a.output/'RESULTS.json').write_text(json.dumps(dict(source_head=command(['git','-C',str(ROOT),'rev-parse','HEAD']).strip(),rows=rows,all_passed=all(r['passed'] for r in rows)),indent=2)+'\n')
                print(json.dumps({k:row[k] for k in ('rate_MB_s','rtt_ms','payload_MB_s','capacity_fraction','passed')}),flush=True)
    finally:
        subprocess.run(['tc','qdisc','del','dev','lo','root'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for server in (gateway,upstream):server.shutdown();server.server_close()
        for thread in threads:thread.join()
        key.unlink(missing_ok=True)
    return 0 if len(rows)==6 and all(r['passed'] for r in rows) else 3


if __name__=='__main__':raise SystemExit(main())
