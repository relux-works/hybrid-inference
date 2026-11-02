"""Start/stop the supervised vLLM node; local cached weights only."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid-bench"))
from bench_common import check_ports, model_identity, private_json, sha256
from bridge import fingerprint, require, validate_layout
from patch_vllm import HOOK, MARKER, patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def ticks(pid):
    try:
        return Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def healthy(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def wait_health(url, child, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(child.poll() is None, "child exited; inspect state logs")
        if healthy(url):
            return
        time.sleep(.2)
    raise RuntimeError("node startup timed out; inspect state logs")


def stop_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)
    # Workers can survive their launcher. The dedicated session group belongs to this launch.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def prepare(args):
    from gpu_run import cached_model
    state = args.state
    os.environ["HF_HOME"] = str(args.hf_cache)
    if args.hf_model:
        os.environ["HF_MODEL"] = str(args.hf_model)
    model = Path(cached_model())
    config = json.loads((model / "config.json").read_text())
    quantization = json.dumps(config.get("quantization_config", {})).lower()
    require("fp8" in quantization and "fp4" not in quantization, "cached source must have FP8 weights")
    identity = model_identity(args.layout_model)
    validate_layout(identity["model_layout"])
    shards = sorted(model.glob("*.safetensors"))
    require(bool(shards), "cached model has no safetensors weights")
    files = sorted(set(shards + [model / "config.json"] + list(model.glob("*.index.json"))))
    manifest = {p.name: sha256(p) for p in files}
    target = dict(identity, model_sha256=fingerprint(manifest), quantization_type="FP8")
    private_json(state / "source-files.json", manifest)
    private_json(state / "target.json", target)
    info = dict(target, backend="cuda", engine="vllm", quantization_type="FP8", context=args.context,
                model_file=model.name, bench_link=True, source_weights_identity="sha256 of canonical source-files.json mapping",
                layout_anchor_sha256=sha256(args.layout_model), layout_anchor=str(args.layout_model),
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    private_json(state / "node-info.json", info)
    (state / "hf-model.path").write_text(str(model) + "\n")
    return model


def serve(args):
    state = args.state
    children, runner, original, installed = [], None, None, None
    outputs = []
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        private_json(state / "supervisor.json", dict(pid=os.getpid(), start_ticks=ticks(os.getpid())))
        python = args.venv / "bin/python"
        probe = subprocess.run([str(python), "-c", "import importlib.metadata as m; import vllm, torch, numpy; print(vllm.__version__); print(m.distribution('vllm').locate_file('vllm/v1/worker/gpu_model_runner.py'))"],
                               capture_output=True, text=True, check=True)
        lines = probe.stdout.strip().splitlines()
        require(lines[-2] == "0.32.0", "unsupported vLLM version")
        runner = Path(lines[-1])
        original = runner.read_bytes()
        require(HOOK.encode() not in original, "runner already hooked; stop its owner first")
        (state / "gpu_model_runner.py.original").write_bytes(original)
        model = prepare(args)
        installed = original.decode().replace(MARKER, HOOK + MARKER).encode()
        patch(runner)
        require(runner.read_bytes() == installed, "runner installation mismatch")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(state / "tmp"),
                   HF_HOME=str(args.hf_cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HYBRID_VLLM_CONFIG=str(state / "control.json"), PYTHONPATH=str(HERE), VLLM_USE_V2_MODEL_RUNNER="0")
        env.pop("LD_PRELOAD", None)
        env.pop("LD_LIBRARY_PATH", None)
        def launch(name, command):
            output = (state / (name + "-01.log")).open("wb")
            outputs.append(output)
            child = subprocess.Popen([str(x) for x in command], env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            private_json(state / (name + ".json"), dict(pid=child.pid, pgid=child.pid, start_ticks=ticks(child.pid), command=[str(x) for x in command]))
            return child
        vllm = launch("vllm", [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
                    "--served-model-name", "hybrid-node", "--host", "127.0.0.1", "--port", args.vllm_port,
                    "--tensor-parallel-size", "1", "--max-num-seqs", "1", "--dtype", "bfloat16",
                    "--kv-cache-dtype", "auto", "--enable-prefix-caching", "--mamba-cache-mode", "align",
                    "--mamba-cache-dtype", "bfloat16", "--mamba-ssm-cache-dtype", "float32",
                    "--max-model-len", args.context, "--max-num-batched-tokens", "7840", "--enable-chunked-prefill",
                    "--gpu-memory-utilization", args.gpu_memory_utilization, "--no-async-scheduling", "--enforce-eager"])
        wait_health("http://127.0.0.1:%d/health" % args.vllm_port, vllm, args.startup_timeout)
        adapter = launch("adapter", [python, HERE / "node_adapter.py", "--target", state / "target.json",
                      "--clients", args.token_file, "--scratch", state / "scratch", "--control", state / "control.json",
                      "--vllm-url", "http://127.0.0.1:%d" % args.vllm_port, "--host", args.bind,
                      "--port", args.port, "--context", args.context, "--codec", args.codec])
        url = "http://127.0.0.1:%d/v1/health" % args.port
        wait_health(url, adapter, args.startup_timeout)
        with urllib.request.urlopen(url, timeout=3) as response:
            health = json.load(response)
        info_path = state / "node-info.json"
        info = json.loads(info_path.read_text())
        info.update(export_block_size=health["export_block_size"], node_limits={k: health[k] for k in ("max_ctx", "ram_budget_bytes", "max_request_bytes")})
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        (state / "ready").touch()
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise RuntimeError("node child exited; inspect logs")
    except KeyboardInterrupt:
        pass
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = []
        for child in reversed(children):
            try:
                stop_child(child)
            except OSError as error:
                cleanup_errors.append(type(error).__name__)
        if installed is not None and runner is not None and original is not None:
            if runner.read_bytes() == installed:
                runner.write_bytes(original)
            elif runner.read_bytes() != original:
                cleanup_errors.append("runner changed; original backup retained")
        for output in outputs:
            output.close()
        (state / "ready").unlink(missing_ok=True)
        (state / "stopped.json").write_text(json.dumps(dict(children=[dict(pid=c.pid, exit_code=c.returncode) for c in children], errors=cleanup_errors)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "serve", "stop", "status"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--venv", type=Path, default=Path("/root/kitb300/gpu/work/vllm-venv"))
    parser.add_argument("--hf-cache", type=Path, default=Path("/root/kitb300/gpu/work/hf"))
    parser.add_argument("--hf-model", type=Path)
    parser.add_argument("--layout-model", type=Path, default=Path("/root/models/Qwen3.8-27B-BF16.gguf"))
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--vllm-port", type=int, default=8093)
    parser.add_argument("--bind", default="127.0.0.1", choices=("127.0.0.1", "0.0.0.0"))
    parser.add_argument("--codec", choices=("reference", "numpy", "torch", "auto"), default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.45)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    args = parser.parse_args()
    args.state = args.state.resolve()
    if args.action in ("stop", "status"):
        record = json.loads((args.state / "supervisor.json").read_text())
        alive = ticks(record["pid"]) == record["start_ticks"]
        if args.action == "status":
            print(json.dumps(dict(alive=alive, ready=alive and (args.state / "ready").exists())))
        elif alive:
            os.kill(record["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 45
            while not (args.state / "stopped.json").exists() and time.monotonic() < deadline:
                time.sleep(.2)
            require((args.state / "stopped.json").exists(), "supervisor cleanup timed out")
            require(not json.loads((args.state / "stopped.json").read_text())["errors"], "node cleanup failed; inspect stopped.json")
        elif args.action == "stop":
            require((args.state / "stopped.json").exists(), "supervisor lost; inspect recorded child PIDs and runner backup")
        return
    require(args.token_file is not None and args.token_file.is_file(), "--token-file required")
    args.token_file = args.token_file.resolve()
    require(args.token_file.stat().st_mode & 0o077 == 0, "private token file required")
    require(0 < args.gpu_memory_utilization < 1, "invalid GPU memory utilization")
    if args.action == "serve":
        serve(args)
        return
    check_ports(args.port, args.vllm_port)
    args.state.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.state / "tmp").mkdir(mode=0o700)
    (args.state / "scratch").mkdir(mode=0o700)
    command = [sys.executable, str(Path(__file__).resolve()), "serve"] + sys.argv[2:]
    with (args.state / "supervisor-01.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not (args.state / "ready").is_file():
            require(child.poll() is None, "supervisor exited; inspect state logs")
            require(time.monotonic() < deadline, "node startup timed out; inspect state logs")
            time.sleep(.2)
        require(healthy("http://127.0.0.1:%d/v1/health" % args.port), "adapter unavailable after readiness")
        print(args.state / "node-info.json")
    except BaseException:
        stop_child(child)
        raise


if __name__ == "__main__":
    main()
