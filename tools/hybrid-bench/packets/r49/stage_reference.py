#!/usr/bin/env python3
"""Stage an independent unprivileged nginx static HTTPS range reference on the node."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,read_private,private_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','token-file','cert','key'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,default=9444);a=p.parse_args()
    require(shutil.which('nginx'),'nginx must be installed by the node operator')
    require(shutil.disk_usage(a.root.parent).free>25*1024**3+384*1024**2,'25 GiB node floor refused')
    require(not a.root.exists(),'reference root must be new')
    # A second run token can be supplied in the same private JSON shape as the node token file.
    tokens=read_private(a.token_file)
    require(isinstance(tokens,dict) and len(tokens)==1,'one node client token required for reference')
    token=next(iter(tokens.values()))
    require(re.fullmatch('[A-Za-z0-9_-]{32,256}',token),'reference token must be URL-safe random material')
    for path in (a.root,a.cert,a.key):require(re.fullmatch(r'[A-Za-z0-9_./-]+',str(path.resolve())),'nginx paths must contain no config syntax')
    a.root.mkdir(mode=0o700);payload=a.root/'payload.bin'
    with payload.open('wb') as stream:
        chunk=bytes(range(256))*4096
        for _ in range(384):stream.write(chunk)
    header=a.root/'auth.hdr';header.write_text('Authorization: Bearer '+token+'\n');header.chmod(0o600)
    config=a.root/'nginx.conf'
    config.write_text('''daemon off;
master_process off;
pid "'''+str(a.root.resolve()/ 'nginx.pid')+'''";
error_log /dev/null crit;
events { worker_connections 256; }
http {
 access_log off;
 server {
 listen '''+str(a.port)+''' ssl;
 ssl_certificate "'''+str(a.cert.resolve())+'''";
 ssl_certificate_key "'''+str(a.key.resolve())+'''";
 if ($http_authorization != "Bearer '''+token+'''") { return 403; }
 location = /payload.bin { root "'''+str(a.root.resolve())+'''"; }
 location / { return 404; }
 }
}
''');config.chmod(0o600)
    subprocess.run(['nginx','-t','-c',str(config.resolve()),'-p',str(a.root.resolve())],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    process=subprocess.Popen(['nginx','-c',str(config.resolve()),'-p',str(a.root.resolve())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    private_json(a.root/'receipt.json',dict(pid=process.pid,port=a.port,payload_bytes=payload.stat().st_size,payload_sha256=sha256(payload),server='nginx independent static TLS'))
    print('independent reference PID '+str(process.pid))


if __name__=='__main__':main()
