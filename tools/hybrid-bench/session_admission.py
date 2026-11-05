"""Reuse the pinned Mac kit's actual child supervisor and cooldown boundaries."""
import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import uuid
from bench_common import require,sha256


class KitAdmission:
    def __init__(self,root,commit,output):
        require(sys.platform=='darwin','Mac kit admission requires Darwin; no CPU fallback receipt')
        self.root,self.commit,self.output=Path(root).resolve(),commit,Path(output)
        require(len(commit)==40,'exact accepted kit commit required')
        subprocess.run(['git','-C',self.root,'verify-commit',commit],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.base=self.root/'bench/mac'
        files=['scripts/kitlib.py','tools/memory_guard.py','tools/slot_handoff.py','scripts/55-run-scaling.sh','VERSION']
        self.hashes={}
        for name in files:
            path=self.base/name
            recorded=subprocess.check_output(['git','-C',self.root,'show',commit+':bench/mac/'+name])
            require(path.read_bytes()==recorded,'kit source differs from accepted ref: '+name)
            self.hashes[name]=sha256(path)
        sys.path[:0]=[str(self.base/'tools'),str(self.base/'scripts')]
        self.priority=self.load('r49_kitlib',self.base/'scripts/kitlib.py')
        self.memory=self.load('r49_memory_guard',self.base/'tools/memory_guard.py')
        self.state=self.output/'r47-memory.json';self.key=uuid.uuid4().hex;self.client=None

    @staticmethod
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def command(self,decoder):
        return [sys.executable,self.base/'tools/memory_guard.py','serve','--state',self.state,'--key',self.key,
                '--owner',str(os.getpid()),'--',*decoder]

    def cooldown(self,process):
        state=self.memory.read(self.state)
        self.client=self.memory.Client(self.state,self.key,state['child_pid'],state['birth'])
        self.client.check();self.client.boundary('end','startup')
        self.priority.priority_check(self.output,server_pids=[state['child_pid']],step='before-cooldown',allow_background=False)
        self.client.boundary('begin','cooldown');start=time.monotonic_ns()
        for _ in range(60):
            time.sleep(5)
            require(process.poll() is None,'kit supervisor exited during cooldown')
        elapsed=(time.monotonic_ns()-start)/1e9
        self.client.boundary('end','cooldown')
        self.priority.priority_check(self.output,server_pids=[state['child_pid']],step='after-cooldown',allow_background=False)
        return dict(schema='hybrid-r47-admission-v1',qualified=True,kit_ref=self.commit,kit_files=self.hashes,
            cooldown_seconds=elapsed,cooldown_start_ns=start,cooldown_end_ns=time.monotonic_ns(),
            raw_memory_evidence=str(self.state),raw_priority_evidence=str(self.output/'OPERATOR-READINESS.json'),
            decoder_pid=state['child_pid'],decoder_birth=state['birth'])

    def stop(self,process):
        self.client.check();self.client.stop();code=process.wait(timeout=30)
        state=self.memory.read(self.state)
        require(code==0 and state.get('accepted') is True,'kit memory supervisor did not accept its owned child')
        return dict(memory_sha256=sha256(self.state),priority_sha256=sha256(self.output/'OPERATOR-READINESS.json'))

    def abort(self,process):
        if process.poll() is not None:return
        process.terminate()
        try:process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            state=self.memory.read(self.state)
            require(state.get('owner_pid')==os.getpid() and state.get('supervisor_pid')==process.pid and
                    state.get('launch_id')==self.key,'cannot prove ownership of kit child for cleanup')
            pid=state['child_pid']
            if self.memory.process_identity(pid)==state['birth']:
                os.killpg(pid,signal.SIGKILL)
            process.kill();process.wait(timeout=15)
