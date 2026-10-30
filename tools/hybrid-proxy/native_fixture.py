"""Small-model native fixtures shared by the opt-in append tests. No builds."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from proxy import HTTPError, Node


def require(value, message):
    if not value:
        raise AssertionError(message)


@contextmanager
def native_server(binary, model, root, template=None, checkpoints=32):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    slots = root / "slots"
    slots.mkdir(exist_ok=True)
    with socket.socket() as allocated:
        allocated.bind(("127.0.0.1", 0))
        port = allocated.getsockname()[1]
    command = [str(Path(binary).resolve()), "-m", str(Path(model).resolve()), "--host", "127.0.0.1",
               "--port", str(port), "-ngl", "0", "-t", "2", "-tb", "2", "-c", "12288", "-np", "3",
               "-b", "256", "-ub", "256", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "--jinja",
               "--slots", "--no-context-shift", "--cache-ram", "0", "--no-cache-idle-slots",
               "--ctx-checkpoints", str(checkpoints), "--checkpoint-min-step", "0",
               "--slot-save-path", str(slots) + "/", "--log-verbosity", "4"]
    if template:
        command += ["--chat-template-file", str(template)]
    (root / "identity.json").write_text(json.dumps(dict(command=command,
        binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(Path(template).read_bytes()).hexdigest() if template else None), indent=2))
    with (root / "server.log").open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        (root / "server.pid").write_text(str(process.pid) + "\n")
        try:
            node = Node(f"http://127.0.0.1:{port}", timeout=120)
            deadline = time.monotonic() + 120
            while True:
                require(process.poll() is None, "native fixture exited during startup")
                try:
                    node.request("GET", "/health")
                    break
                except (OSError, HTTPError):
                    require(time.monotonic() < deadline, "native fixture readiness timed out")
                    time.sleep(0.1)
            yield node, slots
        finally:
            resources = {}
            for name, pid in (("server", process.pid), ("test", os.getpid())):
                status = Path(f"/proc/{pid}/status")
                if status.exists():
                    resources[name] = [line for line in status.read_text().splitlines()
                                       if line.startswith(("VmHWM:", "VmRSS:"))]
            (root / "resources.json").write_text(json.dumps(resources))
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            (root / "cleanup.json").write_text(json.dumps(dict(pid=process.pid, exit_code=process.returncode)))


def rendered_tokens(node, body):
    text = node.request("POST", "/apply-template", body)["prompt"]
    return node.request("POST", "/tokenize", dict(content=text, add_special=True))["tokens"]


def streamed(node, body):
    chunks, message, final = [], dict(role="assistant", content=""), None
    with node.open("POST", "/v1/chat/completions", dict(body, stream=True)) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            if "__verbose" in chunk:
                final = chunk["__verbose"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                for key in ("content", "reasoning_content"):
                    if delta.get(key):
                        message[key] = message.get(key, "") + delta[key]
                for tool in delta.get("tool_calls", []):
                    calls = message.setdefault("tool_calls", [])
                    while len(calls) <= tool["index"]:
                        calls.append(dict(id="", type="function", function=dict(name="", arguments="")))
                    call = calls[tool["index"]]
                    if tool.get("id"):
                        call["id"] = tool["id"]
                    for key in ("name", "arguments"):
                        call["function"][key] += tool.get("function", {}).get(key, "")
    return message, final, chunks
