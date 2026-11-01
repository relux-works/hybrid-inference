"""Synthetic prefill/decode fixtures, not an engine or an equivalence oracle."""
import argparse
import ctypes
import ctypes.util
import hashlib
import http.server
import json
import sqlite3
import struct
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

from store import prefix_hash

MODEL = "0" * 64


def xxh128(body):
    class Digest(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]
    library = ctypes.CDLL(ctypes.util.find_library("xxhash") or "libxxhash.so.0")
    library.XXH3_128bits.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    library.XXH3_128bits.restype = Digest
    value = library.XXH3_128bits(body, len(body))
    return struct.pack(">QQ", value.high, value.low)


def encode_delta(model, sid, before, suffix):
    body = json.dumps(suffix).encode()
    header = dict(magic="HYBD", version=1, model_sha256=model, session_id=sid,
                  n0=len(before), n1=len(before) + len(suffix), tokens=suffix,
                  prefix_hash_n1=prefix_hash(before + suffix), kv_type="q8_0", state_dtype="f32",
                  body_format="synthetic-token-state-v1",
                  sections=[dict(name="synthetic_tokens", offset=0, bytes=len(body))])
    raw = json.dumps(header).encode()
    return struct.pack("<I", len(raw)) + raw + body + xxh128(body) + hashlib.sha256(body).digest()


def decode_delta(raw, model, before):
    if len(raw) < 4:
        raise ValueError("truncated header")
    size, = struct.unpack("<I", raw[:4])
    if size > 1024 * 1024 or 4 + size + 48 > len(raw):
        raise ValueError("truncated delta")
    header = json.loads(raw[4:4 + size])
    body = raw[4 + size:-48]
    if header["magic"] != "HYBD" or header["version"] != 1:
        raise ValueError("bad magic/version")
    if header["model_sha256"] != model:
        raise ValueError("wrong model hash")
    if header["n0"] != len(before):
        raise ValueError("n0 gap or overlap")
    if (header["body_format"] != "synthetic-token-state-v1"
            or header["sections"] != [dict(name="synthetic_tokens", offset=0, bytes=len(body))]):
        raise ValueError("layer layout mismatch")
    if raw[-48:-32] != xxh128(body) or raw[-32:] != hashlib.sha256(body).digest():
        raise ValueError("corrupt checksum")
    suffix = json.loads(body)
    if suffix != header["tokens"] or header["n1"] != len(before) + len(suffix):
        raise ValueError("invalid token range")
    if header["prefix_hash_n1"] != prefix_hash(before + suffix):
        raise ValueError("prefix mismatch")
    return before + suffix


class FixtureTemplate:
    """Text-only fixture renderers; token replay accepts only captured, aligned spans."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.props = dict(fixture["props"])
        self.label = fixture["label"]
        # Fixture properties advertise frame delimiters, not redistributed template source.
        self.props["chat_template"] = ("<|turn> <turn|>" if self.label.startswith("gemma") else
                                       "<|start|> <|eot|>" if self.label.startswith("muse-glimmer") else "ChatML")
        self.renders = [render for shape in fixture["shapes"].values()
                        for render in shape["renders"].values() if "text" in render]

    def tokenize(self, text, initial=False, with_pieces=False):
        for render in self.renders:
            if render["text"] == text:
                pieces = render["tokens_add_special" if initial else "tokens_plain"]
                break
        else:
            pieces = None
            for render in self.renders:
                captured = render["tokens_plain"]
                for start in range(len(captured)):
                    span = ""
                    for stop in range(start, len(captured)):
                        value = captured[stop]["piece"]
                        if not isinstance(value, str):
                            break
                        span += value
                        if span == text:
                            pieces = captured[start:stop + 1]
                            break
                        if not text.startswith(span):
                            break
                    if pieces is not None:
                        break
                if pieces is not None:
                    break
            if not text:
                pieces = []
            bos = self.props["bos_token"]
            added = self.renders[0]["tokens_add_special"]
            plain = self.renders[0]["tokens_plain"]
            bos_piece = added[:len(added) - len(plain)]
            if text == bos and bos_piece:
                pieces = bos_piece
            if pieces is None:
                raise ValueError("tokenization outside captured aligned spans")
            if initial:
                pieces = bos_piece + pieces
        return [dict(p) for p in pieces] if with_pieces else [p["id"] for p in pieces]

    @staticmethod
    def arguments(call):
        args = call["function"]["arguments"]
        return json.loads(args) if isinstance(args, str) else args

    def render(self, body):
        messages = body["messages"]
        if body.get("tools") or any(not isinstance(m.get("content") or "", str) for m in messages):
            raise ValueError("fixture renderer only covers text and calls without tool definitions")
        thinking = body.get("chat_template_kwargs", {}).get("enable_thinking", True)
        last_assistant = messages[-1]["role"] == "assistant"
        agp = body.get("add_generation_prompt", True) and not last_assistant
        if self.label.startswith("qwen"):
            opening, closing = "<|im_start|>", "<|im_end|>\n"
            instruction = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
            system = messages[0]["content"].strip() if messages[0]["role"] == "system" else ""
            system = ((instruction + ("\n\n" if system else "")) if thinking else "") + system
            text = opening + "system\n" + system + closing if system else ""
            for index, m in enumerate(messages):
                role, content = m["role"], (m.get("content") or "").strip()
                if role == "system":
                    continue
                if role == "tool":
                    if index == 0 or messages[index - 1]["role"] != "tool":
                        text += opening + "user"
                    text += "\n<tool_response>\n" + content + "\n</tool_response>"
                    if index + 1 == len(messages) or messages[index + 1]["role"] != "tool":
                        text += closing
                    continue
                text += opening + role + "\n"
                if role == "assistant":
                    text += "<think>\n" + m.get("reasoning_content", "").strip() + "\n</think>\n\n"
                text += content
                for call in m.get("tool_calls", []):
                    text += ("\n\n" if content else "") + "<tool_call>\n<function=" + call["function"]["name"] + ">\n"
                    for key, value in self.arguments(call).items():
                        text += "<parameter=" + key + ">\n" + (value if isinstance(value, str) else json.dumps(value)) + "\n</parameter>\n"
                    text += "</function>\n</tool_call>"
                if not (last_assistant and index == len(messages) - 1):
                    text += closing
            if agp:
                text += opening + "assistant\n<think>\n" + ("\n</think>\n\n" if not thinking else "")
            return text
        if self.label.startswith("gemma"):
            opening, closing = "<|turn>", "<turn|>\n"
            system = messages[0]["content"].strip() if messages[0]["role"] == "system" else ""
            text = opening + "system\n" + ("<|think|>\n" if thinking else "") + system + closing if system or thinking else ""
            last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
            previous_role = None
            for index, m in enumerate(messages):
                role, content = m["role"], (m.get("content") or "").strip()
                if role in ("system", "tool"):
                    continue
                if role != "assistant" or previous_role != "assistant":
                    text += opening + ("model" if role == "assistant" else role) + "\n"
                reasoning = m.get("reasoning") or m.get("reasoning_content")
                if role == "assistant" and index > last_user and (reasoning or last_assistant and index == len(messages) - 1):
                    text += "<|channel>thought\n" + (reasoning + "\n" if reasoning else "") + "<channel|>"
                for call in m.get("tool_calls", []):
                    args = ",".join(key + ":<|\"|>" + value + "<|\"|>" for key, value in self.arguments(call).items())
                    text += "<|tool_call>call:" + call["function"]["name"] + "{" + args + "}<tool_call|>"
                responded = False
                for follow in messages[index + 1:]:
                    if follow["role"] != "tool":
                        break
                    if not m.get("tool_calls"):
                        break
                    name = next((c["function"]["name"] for c in m["tool_calls"] if c.get("id") == follow.get("tool_call_id")), "unknown")
                    text += "<|tool_response>response:" + name + "{value:<|\"|>" + follow["content"] + "<|\"|>}<tool_response|>"
                    responded = True
                text += content
                next_role = next((follow["role"] for follow in messages[index + 1:] if follow["role"] != "tool"), None)
                continues = role == "assistant" and next_role == "assistant" and (not m.get("tool_calls") or responded)
                if m.get("tool_calls") and not responded:
                    text += "<|tool_response>"
                elif not continues and not (responded and not content and next_role is None) and not (last_assistant and index == len(messages) - 1):
                    text += closing
                previous_role = role
            if agp:
                if not messages[-1].get("role") == "tool" or not responded:
                    text += opening + "model\n" + ("<|channel>thought\n<channel|>" if not thinking else "")
                elif thinking:
                    text += "<|channel>thought\n"
            return text
        if not self.label.startswith("muse-glimmer"):
            raise ValueError("unknown fixture renderer")
        opening, closing = "<|start|>", "<|eot|>"
        system = messages[0]["content"] if messages[0]["role"] == "system" else "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-11-01."
        text = opening + "system<|message|>" + system + '\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".' + closing
        names = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls", [])}
        for index, m in enumerate(messages):
            role, content = m["role"], m.get("content") or ""
            if role == "system":
                continue
            if role == "tool":
                name = m.get("name") or names.get(m.get("tool_call_id"), m.get("tool_call_id") or "unknown")
                text += opening + "tool " + name + '<|message|><tool_output name="' + name + '">\n' + content + "\n</tool_output>" + closing
                continue
            if role == "assistant" and (m.get("reasoning_content") or last_assistant and index == len(messages) - 1):
                text += opening + "assistant to=self<|message|>" + m.get("reasoning_content", "") + "<|eom|>"
            calls = m.get("tool_calls", [])
            if calls:
                for call in calls:
                    name = call["function"]["name"]
                    text += opening + 'assistant to=' + name + '<|message|><atem:function_calls>\n<atem:invoke name="' + name + '">\n'
                    for key, value in self.arguments(call).items():
                        text += '<atem:parameter name="' + key + '">' + value + '</atem:parameter>\n'
                    text += '</atem:invoke>\n</atem:function_calls>' + closing
            else:
                text += opening + role + (" to=user" if role == "assistant" else "") + "<|message|>" + content
                if not (last_assistant and index == len(messages) - 1):
                    text += closing
        if agp:
            text += opening + "assistant"
        return text


class State:
    def __init__(self, role, database=":memory:", model=MODEL, slot_path=".", template_fixture=None):
        self.role, self.model, self.slot_path = role, model, Path(slot_path)
        self.template_fixture = template_fixture
        self.lock = threading.RLock()
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, client TEXT, tokens TEXT)")
        self.slots = {i: [] for i in range(2)}
        self.saved = {}
        self.available = True
        self.conflict_once = False
        self.lose_ack_once = False
        self.imports = 0
        self.layout_fingerprint = "1" * 64

    def close(self):
        self.db.close()

    def info(self, tokens):
        return dict(n=len(tokens), prefix_hash=prefix_hash(tokens), tokens=tokens,
                    tier="RAM", last_used=time.time(), model_sha256=self.model)

    def handle(self, method, path, query, body, client):
        if not self.available:
            return 503, {"error": "offline"}
        if self.role == "prefill" and path == "/v1/health":
            return 200, dict(model_sha256=self.model, layout_fingerprint=self.layout_fingerprint)
        if self.role == "prefill":
            if path in ("/v1/health", "/metrics"):
                return 200, {"ok": True}
            if path == "/v1/sessions":
                if method == "POST":
                    if body["model_sha256"] != self.model:
                        return 400, {"error": "model hash"}
                    sid = str(uuid.uuid4())
                    with self.db:
                        self.db.execute("INSERT INTO sessions VALUES(?,?,'[]')", (sid, client))
                    return 200, {"session_id": sid}
                return 200, [dict(session_id=sid, **self.info(json.loads(tokens)))
                             for sid, tokens in self.db.execute("SELECT id,tokens FROM sessions WHERE client=?", (client,))]
            parts = path.split("/")
            if len(parts) < 4 or parts[:3] != ["", "v1", "sessions"]:
                return 404, {}
            sid = parts[3]
            row = self.db.execute("SELECT tokens FROM sessions WHERE id=? AND client=?", (sid, client)).fetchone()
            if row is None:
                return 404, {}
            tokens = json.loads(row[0])
            if len(parts) == 4:
                if method == "DELETE":
                    with self.db:
                        self.db.execute("DELETE FROM sessions WHERE id=? AND client=?", (sid, client))
                    return 204, None
                return 200, self.info(tokens)
            if parts[4] == "snapshot":
                return 200, json.dumps(dict(model_sha256=self.model, tokens=tokens)).encode()
            if parts[4] != "extend":
                return 404, {}
            if self.conflict_once:
                self.conflict_once = False
                return 409, self.info(tokens)
            if body["n0"] != len(tokens) or body["prefix_hash_n0"] != prefix_hash(tokens):
                return 409, self.info(tokens)
            updated = tokens + body["tokens"]
            prefix_hash(updated)
            with self.db:
                self.db.execute("UPDATE sessions SET tokens=? WHERE id=?", (json.dumps(updated), sid))
            if self.lose_ack_once:
                self.lose_ack_once = False
                return 503, {"error": "ack lost after durable apply"}
            return (200, encode_delta(self.model, sid, tokens, body["tokens"])) if body["return_delta"] else (202, None)
        if self.template_fixture and path in ("/apply-template", "/props", "/tokenize"):
            if path == "/props":
                return 200, self.template_fixture.props
            if path == "/apply-template":
                return 200, dict(prompt=self.template_fixture.render(body))
            return 200, dict(tokens=self.template_fixture.tokenize(body["content"],
                body.get("add_special", False), body.get("with_pieces", False)))
        if path == "/apply-template":
            if "_tokens" in body:
                return 200, {"prompt": body["_tokens"]}
            text = "".join("<|im_start|>" + m["role"] + "\n" + (m.get("content") or "") + "<|im_end|>\n"
                           for m in body["messages"])
            if body.get("add_generation_prompt", True):
                text += "<|im_start|>assistant\n"
            return 200, {"prompt": text}
        if path == "/props":
            return 200, {"chat_template": "synthetic ChatML fixture; no recurrent checkpoints"}
        if path == "/tokenize":
            if isinstance(body["content"], list):
                return 200, {"tokens": body["content"]}
            text, pieces = body["content"], []
            while text:
                for marker, token in (("<|im_start|>", 2147480001), ("<|im_end|>", 2147480002)):
                    if text.startswith(marker):
                        piece = marker
                        break
                else:
                    piece, token = text[0], ord(text[0])
                pieces.append(dict(id=token, piece=piece))
                text = text[len(piece):]
            return 200, {"tokens": pieces if body.get("with_pieces") else [piece["id"] for piece in pieces]}
        if path == "/slots":
            return 200, [dict(id=slot, is_processing=False) for slot in self.slots]
        if path.startswith("/slots/"):
            parts = path.split("/")
            slot = int(parts[2])
            tokens = self.slots.setdefault(slot, [])
            if len(parts) == 4 and parts[3] == "info":
                return 200, self.info(tokens)
            action = query.get("action", [""])[0]
            if action == "erase":
                self.slots[slot] = []
            elif action == "save":
                self.saved[body["filename"]] = tokens[:]
                (self.slot_path / body["filename"]).write_text(json.dumps(dict(model_sha256=self.model, tokens=tokens)))
            elif action == "restore":
                if body["filename"] in self.saved:
                    self.slots[slot] = self.saved[body["filename"]][:]
                else:
                    try:
                        snapshot = json.loads((self.slot_path / body["filename"]).read_bytes())
                        if snapshot["model_sha256"] != self.model:
                            raise ValueError("snapshot model")
                        self.slots[slot] = snapshot["tokens"]
                    except FileNotFoundError:
                        return 400, {"error": "missing snapshot"}
            elif action == "import_delta":
                self.slots[slot] = decode_delta(body, self.model, tokens)
                self.imports += 1
            else:
                return 400, {"error": "action"}
            return 200, self.info(self.slots[slot])
        if path in ("/v1/chat/completions", "/completion"):
            if path == "/completion":
                prompt = body["prompt"]
            else:
                rendered = self.handle("POST", "/apply-template", {}, body, client)[1]["prompt"]
                prompt = body.get("prompt_tokens", self.handle("POST", "/tokenize", {}, {"content": rendered}, client)[1]["tokens"])
            if ("prompt_tokens" in body or path == "/completion") and (not isinstance(prompt, list) or not prompt or len(prompt) > 8192 or
                    any(type(token) is not int or not 0 <= token < 2**31 for token in prompt)):
                return 400, {"error": "prompt_tokens"}
            if path == "/completion" and body.get("n_predict") != 0:
                return 400, {"error": "synthetic raw completion supports prefill only"}
            previous = self.slots[body["id_slot"]]
            cached = len(previous) if prompt[:len(previous)] == previous else 0
            if body.get("n_predict", body.get("max_tokens")) == 0:
                self.slots[body["id_slot"]] = prompt
                return 200, {"tokens": [], "timings": {"cache_n": cached, "prompt_n": len(prompt) - cached}}
            # The last sampled token has not been evaluated, like llama-server.
            generated = [901, 902, 903]
            answer = "stub"
            if self.template_fixture:
                answer = "A1"
                end = "<turn|>" if self.template_fixture.label.startswith("gemma") else "<|eot|>" if self.template_fixture.label.startswith("muse") else "<|im_end|>"
                generated = self.template_fixture.tokenize(answer) + self.template_fixture.tokenize(end)
            self.slots[body["id_slot"]] = prompt + generated[:-1]
            response = dict(id="chatcmpl-stub", object="chat.completion", model="stub",
                            choices=[dict(index=0, message=dict(role="assistant", content=answer), finish_reason="stop")])
            native = dict(tokens=generated, timings=dict(cache_n=cached, prompt_n=len(prompt) - cached))
            if body.get("verbose") and body.get("return_tokens"):
                response["__verbose"] = native
            if body.get("stream"):
                chunks = [dict(id="chatcmpl-stub", object="chat.completion.chunk",
                               choices=[dict(index=0, delta=dict(content=part), finish_reason=None)]) for part in (("s", "tub") if answer == "stub" else ("A", "1"))]
                if body.get("verbose") and body.get("return_tokens"):
                    chunks.append(dict(choices=[], __verbose=native))
                return 200, b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
            return 200, response
        return 404, {}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def run_request(self):
        app = self.server.app
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self.send_error(401)
            return
        parsed = urllib.parse.urlsplit(self.path)
        size = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(size)
        body = json.loads(raw) if raw and self.headers.get("Content-Type", "").startswith("application/json") else raw
        try:
            with app.lock:
                status, data = app.handle(self.command, parsed.path, urllib.parse.parse_qs(parsed.query), body, auth[7:])
        except (ValueError, KeyError, TypeError) as error:
            status, data = 400, {"error": str(error)}
        data = data if isinstance(data, bytes) else json.dumps(data).encode() if data is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if parsed.path == "/v1/chat/completions" and isinstance(body, dict) and body.get("stream") else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_DELETE = run_request


def make_server(role, host="127.0.0.1", port=0, **kwargs):
    server = http.server.ThreadingHTTPServer((host, port), Handler)
    server.app = State(role, **kwargs)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--database", default=":memory:")
    parser.add_argument("--model-sha256", default=MODEL)
    parser.add_argument("--slot-path", default=".")
    args = parser.parse_args()
    server = make_server(args.role, port=args.port, database=args.database, model=args.model_sha256, slot_path=args.slot_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.close()


if __name__ == "__main__":
    main()
