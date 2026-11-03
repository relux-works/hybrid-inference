#!/usr/bin/env python3
"""Compare f32 wire/import bytes against an independently built accepted parent."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

from bench_common import private_json, require, sha256
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'hybrid'),str(Path(__file__).resolve().parents[1]/'hybrid-proxy')]
from test_limits import Node
from native_fixture import native_server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--baseline',type=Path)
    a=p.parse_args();root=a.artifacts.resolve();root.mkdir(mode=0o700)
    args=SimpleNamespace(artifacts=str(root),binary=str(a.build_dir.resolve()/'bin/llama-hybrid-prefill'),model=str(a.model.resolve()))
    tokens=[100]*16
    with Node(args,'node',context='4096') as node:
        sid=node.create()
        wire=node.extend(sid,tokens,delta=True)
        n=struct.unpack('<I',wire[:4])[0];header=json.loads(wire[4:4+n])
        require(header['state_dtype']=='f32','default wire changed dtype')
        # Session identity is random; replace only it with an equally sized fixed ID, preserving every other byte.
        canonical=wire.replace(header['session_id'].encode(),b'0'*len(header['session_id']),1)
        # The default transfer ID derives from the session ID and must be normalized as well.
        transfer=header['transfer_id'].encode()
        canonical=canonical.replace(transfer,b'1'*len(transfer),1)
        (root/'wire.bin').write_bytes(canonical)
        with native_server(a.build_dir/'bin/llama-server',a.model,root/'decoder',checkpoints=0) as (decode,slots):
            decode.request('POST','/slots/0?action=import_delta',wire,
                           {'Content-Type':'application/octet-stream','X-Hybrid-Session':header['session_id']})
            info=decode.request('GET','/slots/0/info');require(info['tokens']==tokens,'f32 import journal changed')
            decode.request('POST','/slots/0?action=save',dict(filename='import.bin'))
            saved=slots/'import.bin';(root/'import.bin').write_bytes(saved.read_bytes())
        node.request('DELETE',sid,expected=204)
    result=dict(wire_sha256=sha256(root/'wire.bin'),body_sha256=hashlib.sha256(wire[4+n:]).hexdigest(),
                import_sha256=sha256(root/'import.bin'),model_sha256=sha256(a.model),
                binaries={p.name:sha256(p) for p in (a.build_dir/'bin').iterdir() if p.is_file()})
    if a.baseline:
        old=json.loads((a.baseline/'result.json').read_text())
        # Actual envelope/body/checksums and full imported native snapshot must match the accepted parent byte for byte.
        require(all(result[k]==old[k] for k in ('wire_sha256','body_sha256','import_sha256','model_sha256')),
                'f32 merge changed wire or imported snapshot')
        require((root/'wire.bin').read_bytes()==(a.baseline/'wire.bin').read_bytes() and
                (root/'import.bin').read_bytes()==(a.baseline/'import.bin').read_bytes(),'f32 merge byte regression')
        result['baseline_equal']=True
    private_json(root/'result.json',result)
    print('PASS f32 wire/body/checksums and imported snapshot'+(' match accepted parent' if a.baseline else ' baseline recorded'))


if __name__=='__main__':main()
