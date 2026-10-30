"""Opt-in native API controls for the M0.2 scope decision; never builds a server."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from proxy import HTTPError, Node
from store import prefix_hash


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--require-chat-token-input", action="store_true")
    args = parser.parse_args()
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=True)
    slots = root / "slots"
    slots.mkdir(exist_ok=True)
    with socket.socket() as allocated:
        allocated.bind(("127.0.0.1", 0))
        port = allocated.getsockname()[1]
    command = [str(Path(args.binary).resolve()), "-m", args.model, "--host", "127.0.0.1", "--port", str(port),
               "-ngl", "0", "-t", "2", "-tb", "2", "-c", "4096", "-b", "128", "-ub", "128", "-np", "2",
               "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "--slots", "--no-context-shift", "--cache-ram", "0",
               "--no-cache-idle-slots", "--ctx-checkpoints", "32", "--checkpoint-min-step", "0", "--jinja",
               "--log-verbosity", "4", "--slot-save-path", str(slots) + "/"]
    if args.template:
        command += ["--chat-template-file", args.template]
    identity = dict(command=command, binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
                    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.template:
        identity["template_sha256"] = hashlib.sha256(Path(args.template).read_bytes()).hexdigest()
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    result = {}
    with (root / "server.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (root / "server.pid").write_text(str(process.pid) + "\n")
        try:
            node = Node(f"http://127.0.0.1:{port}", timeout=120)
            deadline = time.monotonic() + 120
            while True:
                require(process.poll() is None, "server exited during startup")
                try:
                    node.request("GET", "/health")
                    break
                except (OSError, HTTPError):
                    require(time.monotonic() < deadline, "server readiness timed out")
                    time.sleep(0.1)

            def encode(text):
                return node.request("POST", "/tokenize", dict(content=text, add_special=True))["tokens"]

            body = dict(messages=[dict(role="user", content="Say hello.")], id_slot=0, n_predict=0,
                        temperature=0, return_tokens=True, verbose=True, cache_prompt=True,
                        chat_template_kwargs=dict(enable_thinking=False))
            rendered = node.request("POST", "/apply-template", body)["prompt"]
            expected = encode(rendered)
            authoritative = encode("An authoritative journal has a different token prefix.")
            require(authoritative != expected, "positive and negative prompt controls must differ")

            # Positive: raw completion evaluates the supplied IDs exactly, without rendering messages.
            node.request("POST", "/completion", dict(prompt=authoritative, id_slot=0, n_predict=0))
            info = node.request("GET", "/slots/0/info")
            require(info["tokens"] == authoritative and info["prefix_hash"] == prefix_hash(authoritative),
                    "raw completion did not honor exact token input")
            result["raw_token_input"] = True

            # Negative: chat ignores a conflicting prompt array; metadata and token hash show what was evaluated.
            response = node.request("POST", "/v1/chat/completions", dict(body, prompt=authoritative))
            info = node.request("GET", "/slots/0/info")
            require(info["tokens"] == expected and info["prefix_hash"] == prefix_hash(expected),
                    "chat prompt precedence changed; reassess the scope decision")
            result["chat_honored_token_input"] = info["tokens"] == authoritative
            result["chat_prompt_control"] = dict(supplied=authoritative, evaluated=info, response=response)

            # Positive: nonstream chat exposes all sampled IDs in __verbose when return_tokens is enabled.
            response = node.request("POST", "/v1/chat/completions", dict(body, n_predict=8, ignore_eos=True))
            sampled = response["__verbose"]["tokens"]
            require(len(sampled) == 8, "nonstream return_tokens control did not return eight IDs")
            info = node.request("GET", "/slots/0/info")
            require(info["tokens"] == expected + sampled[:-1], "slot sampled/evaluated boundary changed")
            result["nonstream_sampled_ids"] = sampled

            # Negative: verbose streaming omits sampled IDs; slot info excludes the final sampled token.
            chunks = []
            with node.open("POST", "/v1/chat/completions", dict(body, id_slot=1, stream=True,
                                                                  n_predict=8, ignore_eos=True)) as stream:
                for line in stream:
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        chunks.append(json.loads(line[6:]))
            verbose = [chunk["__verbose"] for chunk in chunks if "__verbose" in chunk]
            require(verbose and not any(item.get("tokens") for item in verbose),
                    "stream verbose now carries IDs; reassess the scope decision")
            info = node.request("GET", "/slots/1/info")
            require(len(info["tokens"]) == len(expected) + 7, "expected seven evaluated generated tokens")
            result["stream_chunks"] = chunks
            result["stream_evaluated_tokens"] = info

            # Negative: enabling logprobs to obtain stream IDs is rejected even for tool_choice=none.
            tool = dict(type="function", function=dict(name="lookup", description="Look up a value.",
                        parameters=dict(type="object", properties={})))
            try:
                node.request("POST", "/v1/chat/completions", dict(body, stream=True, n_predict=8,
                             tools=[tool], tool_choice="none", logprobs=True, top_logprobs=1))
            except HTTPError as error:
                require(error.status == 400 and b"logprobs is not supported with tools + stream" in error.body,
                        "tool/logprobs rejection was not the expected API constraint")
                result["tool_stream_logprobs"] = dict(status=error.status, body=error.body.decode())
            else:
                raise AssertionError("tool streaming now permits logprobs; reassess the scope decision")

            # Positive: native checkpoints preserve a nonzero prefix on a real edit of a later user turn.
            messages = [dict(role="system", content="Answer briefly. " * 30),
                        dict(role="user", content="First question: count to three."),
                        dict(role="assistant", content="1, 2, 3."),
                        dict(role="user", content="Second question: name a color.")]
            history = node.request("POST", "/apply-template", dict(body, messages=messages))["prompt"]
            marker = "<|im_start|>user\n"
            boundary = len(encode(history[:history.rindex(marker)]))
            node.request("POST", "/slots/0?action=erase", {})
            before = node.request("POST", "/v1/chat/completions", dict(body, messages=messages))
            messages[-1] = dict(role="user", content="Second question: name an animal.")
            after = node.request("POST", "/v1/chat/completions", dict(body, messages=messages))
            require(boundary > 0 and after["timings"]["cache_n"] == boundary,
                    "native rewrite did not restore the latest user-message boundary")
            edited = encode(node.request("POST", "/apply-template", dict(body, messages=messages))["prompt"])
            info = node.request("GET", "/slots/0/info")
            require(info["tokens"] == edited and info["prefix_hash"] == prefix_hash(edited),
                    "native rewrite ended with the wrong token prefix")
            result["checkpoint_rewrite"] = dict(boundary=boundary, before=before["timings"],
                                                  after=after["timings"], evaluated=info)
            (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(dict(raw_token_input=result["raw_token_input"],
                chat_honored_token_input=result["chat_honored_token_input"], nonstream_ids=len(sampled),
                streaming_verbose_ids=0, tool_stream_logprobs_status=400,
                checkpoint_cached_tokens=after["timings"]["cache_n"])), flush=True)
            if args.require_chat_token_input:
                require(result["chat_honored_token_input"],
                        "M0.2 prerequisite: chat must evaluate authoritative token IDs while retaining its chat session")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            (root / "cleanup.json").write_text(json.dumps(dict(pid=process.pid, exit_code=process.returncode)) + "\n")


if __name__ == "__main__":
    main()
