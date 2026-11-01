"""Replay real decoder captures; aligned token spans are not a general tokenizer."""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from journal import JournalConflict, Renderer
from stub_server import FixtureTemplate, State
import test_proxy
from proxy import HTTPError

FIXTURES = Path(__file__).with_name("fixtures") / "templates"


def cases():
    for path in sorted(FIXTURES.glob("*.json")):
        fixture = json.loads(path.read_text())
        for name, shape in fixture["shapes"].items():
            for variant, render in shape["renders"].items():
                body = dict(messages=shape["messages"], add_generation_prompt=variant.startswith("agp/"))
                if variant.endswith("enable_thinking_false"):
                    body["chat_template_kwargs"] = dict(enable_thinking=False)
                yield fixture, name, variant, body, render


class ReplayNode:
    def __init__(self, fixture):
        self.template = FixtureTemplate(fixture)
        self.state = State("decode", template_fixture=self.template)

    def request(self, method, path, body=None):
        status, result = self.state.handle(method, path, {}, body, "fixture")
        if status != 200:
            raise AssertionError((status, result))
        return result


class TemplateTests(unittest.TestCase):
    # All 96 captured renders and both token modes replay exactly through the fixture-backed stub.
    def test_fixture_replay(self):
        count = 0
        for fixture, name, variant, body, captured in cases():
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                node = ReplayNode(fixture)
                self.addCleanup(node.state.close)
                self.assertEqual(node.request("POST", "/apply-template", body)["prompt"].encode(), captured["text"].encode())
                for initial, key in ((False, "tokens_plain"), (True, "tokens_add_special")):
                    result = node.request("POST", "/tokenize", dict(content=captured["text"], add_special=initial, with_pieces=True))["tokens"]
                    self.assertEqual(result, captured[key])
                    self.assertEqual(node.template.tokenize(captured["text"], initial), [p["id"] for p in result])
                count += 1
        self.assertEqual(count, 96)

    # Live probes give byte-exact starts, complete role coverage and decoder token positions, including zero-byte BOS.
    def test_layout_and_user_boundaries(self):
        for fixture, name, variant, body, captured in cases():
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                node = ReplayNode(fixture)
                self.addCleanup(node.state.close)
                renderer = Renderer(node)
                text = captured["text"]
                tokens = [p["id"] for p in captured["tokens_add_special"]]
                if fixture["label"].startswith("gemma") and name == "bench_tool_without_call":
                    with self.assertRaisesRegex(JournalConflict, "boundary probe"):
                        renderer.layout(body, text, tokens)
                    continue
                layout, end = renderer.layout(body, text, tokens)
                self.assertEqual([row["message_index"] for row in layout], list(range(len(body["messages"]))))
                raw = renderer.token_bytes(text, tokens)
                def byte(position):
                    return sum(map(len, raw[:position["n"]])) + position["byte_offset"]
                expected_end = len(text.encode()) if body["messages"][-1]["role"] == "assistant" else len(renderer.render(dict(body, add_generation_prompt=False)).encode())
                self.assertEqual(byte(end), expected_end)
                boundaries = renderer.user_boundaries(body, text, tokens)
                self.assertEqual([b["message_index"] for b in boundaries], [i for i, m in enumerate(body["messages"]) if m["role"] == "user"])
                search = 0
                for row in layout:
                    start, stop = byte(row["start"]), byte(row["end"])
                    self.assertLessEqual(start, stop)
                    role = row["role"]
                    opening_text = renderer.family(body)[0]
                    if role == "tool":
                        header = ("<|tool_response>" if opening_text == "<|turn>" else
                                  "<|im_start|>user\n<tool_response>\n" if opening_text == "<|im_start|>" else "<|start|>tool ")
                    else:
                        header = opening_text + ("model" if role == "assistant" and opening_text == "<|turn>" else role)
                    expected_start = text.encode().index(header.encode(), search)
                    self.assertEqual(start, expected_start)
                    search = expected_start + len(header)
                    segment = text.encode()[start:stop]
                    opening = renderer.family(body)[0].encode()
                    self.assertTrue(segment.startswith(opening) or row["role"] == "tool" and b"<|tool_response>" in segment)
                    if row["role"] == "user":
                        boundary = next(b for b in boundaries if b["message_index"] == row["message_index"])
                        self.assertEqual(row["start"], dict(n=boundary["n"], byte_offset=0))
                        self.assertEqual(tokens[boundary["n"]:boundary["n"] + len(renderer.user_header(body, text, row["message_index"]))], renderer.user_header(body, text, row["message_index"]))
                if body["messages"][0]["role"] == "system":
                    leading = len(tokens) - len(captured["tokens_plain"])
                    self.assertEqual(layout[0]["start"], dict(n=leading, byte_offset=0))

    # Exact append retains sampled IDs and overlaps one emitted EOT across normal, reasoning, tool and two-round histories.
    def test_append(self):
        for fixture, name, variant, body, captured in cases():
            if name not in ("system_user_assistant_user", "reasoning_assistant", "bench_tool_without_call", "tool_call_and_response", "two_rounds"):
                continue
            if fixture["label"].startswith("gemma") and name == "bench_tool_without_call":
                continue
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                node = ReplayNode(fixture)
                self.addCleanup(node.state.close)
                renderer = Renderer(node)
                tokens = [p["id"] for p in captured["tokens_add_special"]]
                past = 3
                if name == "two_rounds":
                    past = 5
                anchor = [71, 72]
                prompt, boundaries, closing, tail = renderer.append(body, captured["text"], tokens, past, anchor)
                self.assertEqual(prompt, anchor + tail)
                closing_text = "<|tool_response>" if fixture["label"].startswith("gemma") and name == "tool_call_and_response" else renderer.family(body)[1].rstrip("\n")
                end = renderer.encode(closing_text)
                self.assertEqual(len(end), 1)
                self.assertEqual(tail[0], end[0])
                self.assertEqual(renderer.append(body, captured["text"], tokens, past, anchor + end)[0], prompt)
                for boundary in boundaries:
                    expected = next(b for b in renderer.user_boundaries(body, captured["text"], tokens) if b["message_index"] == boundary["message_index"])
                    self.assertEqual(boundary["n"], len(anchor) + expected["n"] - (len(tokens) - len(tail)))
                self.assertGreaterEqual(closing, 1)
                layout, _ = renderer.layout(body, captured["text"], tokens, start=past,
                    token_base=len(tokens) - len(tail), anchor_base=len(anchor))
                self.assertTrue(all(row["start"]["n"] >= len(anchor) for row in layout))
                if name == "tool_call_and_response":
                    self.assertIn("T1", b"".join(renderer.token_bytes(captured["text"], tokens)[len(tokens) - len(tail):]).decode())

    # Byte offsets inside a token count UTF-8 bytes; a tokenizer BOS occupies one token and zero bytes.
    def test_utf8_offsets_inside_piece(self):
        from test_journal import TemplateNode
        node = TemplateNode()
        original = node.request
        def request(method, path, body=None):
            if path == "/props":
                return dict(chat_template="ChatML", bos_token="<bos>")
            if path == "/tokenize":
                text = body["content"]
                pieces = [dict(id=7, piece="<bos>")] if text == "<bos>" else [dict(id=400, piece=list(text.encode()))]
                if body.get("add_special"):
                    pieces.insert(0, dict(id=7, piece="<bos>"))
                return dict(tokens=pieces if body.get("with_pieces") else [p["id"] for p in pieces])
            return original(method, path, body)
        renderer = Renderer(mock.Mock(request=request))
        body = dict(messages=[dict(role="system", content="caf\u00e9"), dict(role="user", content="next")])
        text = renderer.render(body)
        layout, end = renderer.layout(body, text, [7, 400])
        offset = len("<|im_start|>system\ncaf\u00e9<|im_end|>\n".encode())
        self.assertEqual(layout[0]["start"], dict(n=1, byte_offset=0))
        self.assertEqual(layout[1]["start"], dict(n=1, byte_offset=offset))
        self.assertEqual(end, dict(n=1, byte_offset=len(renderer.render(dict(body, add_generation_prompt=False)).encode())))

    # Captures bind BOS/EOT spellings and IDs; the implementation's EOG table recognizes all three reply endings.
    def test_reply_endings(self):
        expected = {"qwen3.8-27b": ("<|im_end|>", 248046), "gemma4-31b": ("<turn|>", 106), "muse-glimmer-30b": ("<|eot|>", 200008)}
        source = (Path(__file__).resolve().parents[2] / "src/llama-vocab.cpp").read_text()
        for path in FIXTURES.glob("*.json"):
            fixture = json.loads(path.read_text())
            node = ReplayNode(fixture)
            self.addCleanup(node.state.close)
            token, token_id = expected[fixture["label"]]
            self.assertEqual(node.template.tokenize(token), [token_id])
            self.assertIn('t.first == "' + token + '"', source)


class TemplateHTTPTests(unittest.TestCase):
    setUp = test_proxy.HTTPTests.setUp
    tearDown = test_proxy.HTTPTests.tearDown
    start = test_proxy.HTTPTests.start
    url = staticmethod(test_proxy.HTTPTests.url)
    chat = test_proxy.HTTPTests.chat

    def configure(self, label="gemma4-31b"):
        fixture = json.loads((FIXTURES / (label + ".json")).read_text())
        self.decode.app.template_fixture = FixtureTemplate(fixture)
        return fixture

    def snapshot(self, sid):
        return (self.app.store.get(sid), copy.deepcopy(self.decode.app.slots), self.prefill.app.db.execute("SELECT * FROM sessions").fetchall())

    def reject(self, mutate, expected, append=True, synthetic=False):
        fixture = self.configure()
        if synthetic:
            self.decode.app.template_fixture = None
        shape = fixture["shapes"]["system_user"]
        body = dict(messages=copy.deepcopy(shape["messages"]), chat_template_kwargs=dict(enable_thinking=False))
        with self.client.open("POST", "/v1/chat/completions", body) as response:
            sid = response.headers["X-Hybrid-Session"]
            response.read()
        with self.app.reserve(sid):
            pass
        before = self.snapshot(sid)
        if append:
            body["messages"] = before[0]["journal"]["messages"] + [dict(role="user", content="U2")]
        original = self.decode.app.handle
        def handle(method, path, query, data, client):
            status, result = original(method, path, query, data, client)
            return status, mutate(path, data, result)
        with mock.patch.object(self.decode.app, "handle", side_effect=handle):
            with self.assertRaises(HTTPError) as error:
                self.client.request("POST", "/v1/chat/completions", body, {"X-Hybrid-Session": sid})
            self.assertEqual(error.exception.status, 409)
            self.assertIn(expected, error.exception.body.decode())
        self.assertEqual(self.snapshot(sid), before)

    # Real capture shapes traverse HTTP append with exact journal/slot positions for all three families.
    def test_http_append_all_families(self):
        self.app.benchmark = True
        for label, mode in ((label, mode) for label in ("qwen3.8-27b", "gemma4-31b", "muse-glimmer-30b") for mode in ("local", "offload")):
            with self.subTest(model=label, mode=mode):
                fixture = self.configure(label)
                body = dict(messages=copy.deepcopy(fixture["shapes"]["system_user"]["messages"]), chat_template_kwargs=dict(enable_thinking=False))
                with self.client.open("POST", "/v1/chat/completions", body, {"X-Hybrid-Bench-Mode": mode}) as response:
                    sid = response.headers["X-Hybrid-Session"]
                    response.read()
                with self.app.reserve(sid):
                    pass
                old = self.app.store.get(sid)
                follow = dict(body, messages=old["journal"]["messages"] + [dict(role="user", content="U2")])
                with self.client.open("POST", "/v1/chat/completions", follow, {"X-Hybrid-Session": sid, "X-Hybrid-Bench-Mode": mode}) as response:
                    self.assertEqual(response.headers["X-Hybrid-Session"], sid)
                    response.read()
                with self.app.reserve(sid):
                    pass
                new = self.app.store.get(sid)
                self.assertEqual(new["tokens"][:len(old["tokens"])], old["tokens"])
                info = self.decode.app.info(self.decode.app.slots[new["slot"]])
                self.assertEqual(info["tokens"], new["tokens"])
                self.assertEqual(new["journal"]["last_meter"]["prefilled"], new["journal"]["last_meter"]["new_tokens"])
                if mode == "local":
                    boundary = next(cp for cp in new["journal"]["checkpoints"] if cp["message_index"] == 3)["n"]
                else:
                    boundary = next(row for row in new["journal"]["message_boundaries"] if row["message_index"] == 3)["start"]["n"]
                rendered = self.app.renderer.render(follow)
                header = self.app.renderer.user_header(follow, rendered, 3)
                self.assertEqual(new["tokens"][boundary:boundary + len(header)], header)

    # Unknown frames reject at the intended family guard and preserve the journal, slots and remote prefix.
    def test_unknown_template_atomic(self):
        self.reject(lambda path, body, result: dict(prompt=result["prompt"].replace("<|turn>", "<unknown>")) if path == "/apply-template" and body["messages"][0]["role"] == "user" else result, "unknown template family")

    # A missing/altered plain piece rejects exact bytes without a journal write; a valid capture is the control.
    def test_altered_piece_atomic(self):
        for replacement in ("", "X"):
            def mutate(path, body, result):
                if path == "/tokenize" and body.get("with_pieces"):
                    result = copy.deepcopy(result)
                    next(p for p in result["tokens"] if p["piece"] == "SYS")["piece"] = replacement
                return result
            self.reject(mutate, "pieces do not preserve")

    # Only one declared BOS can precede plain IDs; a foreign or repeated special rejects without state changes.
    def test_foreign_special_atomic(self):
        def mutate(path, body, result):
            if path == "/tokenize" and body.get("add_special"):
                result = copy.deepcopy(result)
                result["tokens"][0] = dict(id=999, piece="<foreign>") if body.get("with_pieces") else 999
            return result
        self.reject(mutate, "unsupported leading tokenizer specials")

    # Two added BOS positions are forbidden even when both use the declared BOS.
    def test_double_bos_atomic(self):
        def mutate(path, body, result):
            if path == "/tokenize" and body.get("add_special"):
                result = copy.deepcopy(result)
                result["tokens"].insert(0, result["tokens"][0])
            return result
        self.reject(mutate, "unsupported leading tokenizer specials")

    # A BOS already in the rendered text cannot also become an invisible leading position.
    def test_bos_in_text_atomic(self):
        original = None
        bos = "<bos>"
        def tokenize(text, initial=False, with_pieces=False):
            prefix = text.startswith(bos)
            result = original(text[len(bos):] if prefix else text, initial, with_pieces)
            if prefix:
                result = ([dict(id=2, piece=bos)] if with_pieces else [2]) + result
            return result
        def mutate(path, body, result):
            nonlocal original
            if path == "/apply-template":
                if original is None:
                    original = self.decode.app.template_fixture.tokenize
                    self.decode.app.template_fixture.tokenize = tokenize
                return dict(prompt=bos + result["prompt"])
            return result
        self.reject(mutate, "unsupported leading tokenizer specials")

    # A marker probe may change its own message, never any preceding rendered byte.
    def test_probe_rewrites_context_atomic(self):
        def mutate(path, body, result):
            if path == "/apply-template" and len(body["messages"]) > 3 and str(body["messages"][-1].get("content", "")).startswith("hybrid"):
                return dict(prompt=result["prompt"].replace("SYS", "BAD"))
            return result
        self.reject(mutate, "preceding template context", synthetic=True)

    # The probed suffix must equal the actual text and token suffix, beyond merely starting with EOT.
    def test_changed_suffix_atomic(self):
        def mutate(path, body, result):
            if path == "/apply-template" and len(body["messages"]) > 3 and str(body["messages"][2].get("content", "")).startswith("hybrid"):
                return dict(prompt=result["prompt"] + "suffix-change")
            return result
        self.reject(mutate, "exact end-of-turn", synthetic=True)

    # A split EOT rejects at the single-token guard before append commits any journal or native writes.
    def test_split_eot_atomic(self):
        self.reject(lambda path, body, result: dict(tokens=[10, 11]) if path == "/tokenize" and body["content"] == "<turn|>" else result, "exact end-of-turn", append=True)

    # A generation prompt cannot alter an earlier message, even when its token pieces reconstruct exactly.
    def test_generation_rewrites_history_atomic(self):
        def mutate(path, body, result):
            if path == "/apply-template" and body.get("add_generation_prompt") is False and body["messages"][0]["role"] == "system":
                return dict(prompt=result["prompt"].replace("SYS", "BAD"))
            return result
        self.reject(mutate, "generation prompt changed previous messages")


if __name__ == "__main__":
    unittest.main()
