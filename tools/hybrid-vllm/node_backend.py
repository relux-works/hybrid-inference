"""Serialized vLLM HTTP requests and private post-forward capture control."""

import json
import os
from pathlib import Path
import urllib.request

from bridge import require


class VllmBackend:
    def __init__(self, url, control, target, scratch, timeout=600):
        self.url, self.control = url.rstrip("/"), Path(control)
        self.target, self.scratch, self.timeout = target, Path(scratch), timeout
        self.block_size = None
        self.failed = False

    def healthy(self):
        if self.failed:
            return False
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def request(self, config):
        if self.failed:
            raise OSError("vLLM request failed; restart the node before retrying")
        temporary = self.control.with_suffix(".tmp")
        temporary.write_text(json.dumps(config))
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.control)
        body = json.dumps(dict(model="hybrid-node", prompt=config["tokens"], max_tokens=1,
                               temperature=0, ignore_eos=True)).encode()
        request = urllib.request.Request(self.url + "/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read(1024**2))
            require(not result.get("error"), "vLLM prefill failed")
        except BaseException:
            self.failed = True
            raise
        finally:
            self.control.unlink(missing_ok=True)

    def probe(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            self.request(dict(node_mode=True, probe=True, target=self.target, tokens=[5], output=tmp))
            self.block_size = json.loads((Path(tmp) / "ready.json").read_text())["block_size"]
        return self.block_size

    def prefill(self, tokens, n0, output, capture=True):
        self.request(dict(node_mode=True, target=self.target, tokens=tokens, n0=n0,
                          capture=capture, output=str(output), rotations=[256, 64]))
        try:
            finished = json.loads((output / "finished.json").read_text())
            require(finished == dict(n=len(tokens), block_size=self.block_size), "incomplete vLLM prefill")
        except BaseException:
            self.failed = True
            raise
        return output / ("capture-%08d" % len(tokens)) / "snapshot.json" if capture else None
