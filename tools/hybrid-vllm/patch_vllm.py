"""Install the explicit post-forward dump hook into a vLLM 0.32.0 runner."""

import argparse
import hashlib
from pathlib import Path

from bridge import require

MARKER = '        with record_function_or_nullcontext("gpu_model_runner: postprocess"):\n'
HOOK = "        from exporter import after_prefill\n        after_prefill(self, scheduler_output)\n\n"
RUNNER_SHA256 = "4cf95efe4ea8caa3110dcd231ff95ceb97e8136f5056e6bdd388300a484c4a81"


def patch(path):
    path = Path(path)
    source = path.read_bytes().decode()
    original = source.replace(HOOK, "")
    require(hashlib.sha256(original.encode()).hexdigest() == RUNNER_SHA256, "runner differs from inspected v0.32.0 source")
    require(source.count(MARKER) == 1, "runner hook location changed")
    require("class GPUModelRunner(" in source and "def execute_model(" in source, "unsupported runner")
    if HOOK in source:
        require(source.count(HOOK) == source.count(HOOK + MARKER) == 1, "runner hook location changed")
        return
    path.write_text(source.replace(MARKER, HOOK + MARKER))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    patch(parser.parse_args().runner)
