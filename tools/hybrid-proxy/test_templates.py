"""Replay real decoder captures; aligned token spans are not a general tokenizer."""
import copy
import json
import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from journal import JournalConflict, Renderer
from stub_server import FixtureTemplate, State
import test_proxy
from proxy import HTTPError
from template_oracle import TemplateOracle, source_fixture

FIXTURES = Path(__file__).with_name("fixtures") / "templates"


def cases(r20=False):
    for path in sorted(FIXTURES.glob("*.json")):
        if path.stem.endswith("-r20") != r20:
            continue
        fixture = json.loads(path.read_text())
        for name, shape in fixture["shapes"].items():
            for variant, render in shape["renders"].items():
                body = dict(messages=shape["messages"], add_generation_prompt=variant.startswith("agp/"))
                if shape.get("tools") is not None:
                    body["tools"] = shape["tools"]
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


class ProbeNode(ReplayNode):
    """Uncaptured frame tests use character pieces, not model tokenizations."""
    def request(self, method, path, body=None):
        if path != "/tokenize":
            return super().request(method, path, body)
        specials = ["<|turn>", "<turn|>", "<bos>", "<|tool_call>", "<tool_call|>",
                    "<|tool_response>", "<tool_response|>", '<|"|>', "<|channel>", "<channel|>", "<|think|>"]
        pieces = []
        for part in re.split("(" + "|".join(map(re.escape, specials)) + ")", body["content"]):
            if part in specials:
                pieces.append(dict(id=1000000 + specials.index(part), piece=part))
            else:
                pieces.extend(dict(id=1000 + ord(char), piece=char) for char in part)
        if body.get("add_special"):
            pieces.insert(0, dict(id=1000002, piece="<bos>"))
        return dict(tokens=pieces if body.get("with_pieces") else [p["id"] for p in pieces])


class TemplateGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((FIXTURES / "gemma4-31b.json").read_text())
        self.node = ProbeNode(self.fixture)
        self.addCleanup(self.node.state.close)
        self.renderer = Renderer(self.node)
        self.body = dict(messages=copy.deepcopy(self.fixture["shapes"]["system_user_assistant_user"]["messages"]),
                         chat_template_kwargs=dict(enable_thinking=False))

    def change(self, mutate):
        original = self.node.request
        return mock.patch.object(self.node, "request", side_effect=lambda method, path, body=None: mutate(path, body, original(method, path, body)))

    def rendered(self):
        text = self.renderer.render(self.body)
        return text, self.renderer.encode(text, initial=True)

    # Every advertised BOS/EOS/source property is required in addition to a valid live family render.
    def test_family_source_properties(self):
        self.assertEqual(self.renderer.family(self.body), ("<|turn>", "<turn|>\n"))
        for key, value in (("bos_token", "wrong"), ("eos_token", "wrong"),
                           ("chat_template", "<turn|>"), ("chat_template", "<|turn>")):
            with self.subTest(property=key, value=value), self.change(lambda path, body, result: dict(result, **{key: value}) if path == "/props" else result):
                with self.assertRaisesRegex(JournalConflict, "unknown template family"):
                    self.renderer.family(self.body)

    # Family selection rejects a wrong generation header even when the previous messages are identical.
    def test_family_generation_header(self):
        self.renderer.family(self.body)
        def mutate(path, body, result):
            if path == "/apply-template" and body.get("add_generation_prompt"):
                return dict(prompt=result["prompt"].removesuffix("<|turn>model\n<|channel>thought\n<channel|>") + "wrong-generation-header")
            return result
        with self.change(mutate), self.assertRaisesRegex(JournalConflict, "unknown template family"):
            self.renderer.family(self.body)

    # Each family-probe transition and the final closing are checked, beyond the initial user header.
    def test_family_closings(self):
        self.renderer.family(self.body)
        for transition in ("assistant", "user", "last"):
            def mutate(path, body, result):
                if path != "/apply-template":
                    return result
                text = result["prompt"]
                if transition == "last":
                    generation = "<|turn>model\n<|channel>thought\n<channel|>" if body.get("add_generation_prompt") else ""
                    return dict(prompt=(text.removesuffix(generation) if generation else text) + "extra" + generation)
                if transition == "assistant":
                    return dict(prompt=text.replace("<turn|>\n<|turn>model", "<turn|>\n<|turn>wrong", 1))
                return dict(prompt=text.replace("<turn|>\n<|turn>user", "<turn|>\n<|turn>wrong", 1))
            with self.subTest(transition=transition), self.change(mutate):
                with self.assertRaisesRegex(JournalConflict, "unknown template family"):
                    self.renderer.family(self.body)

    # Family generation must preserve the whole preceding render before checking its assistant header.
    def test_family_generation_prefix(self):
        self.renderer.family(self.body)
        with self.change(lambda path, body, result: dict(prompt=result["prompt"].replace("<|turn>user\n", "<|turn>USER\n", 1)) if path == "/apply-template" and body.get("add_generation_prompt") else result):
            with self.assertRaisesRegex(JournalConflict, "generation prompt changed previous messages"):
                self.renderer.family(self.body)

    # A user checkpoint requires the live header bytes, not just the preceding token prefix.
    def test_users_header_text(self):
        text, tokens = self.rendered()
        self.assertEqual([b["message_index"] for b in self.renderer.user_boundaries(self.body, text, tokens, start=3)], [3])
        def mutate(path, body, result):
            if path == "/apply-template" and len(body["messages"]) == 4 and body["messages"][-1]["content"].startswith("hybrid"):
                return dict(prompt=result["prompt"].replace("<|turn>user\n" + body["messages"][-1]["content"], "<|turn>user changed\n" + body["messages"][-1]["content"]))
            return result
        with self.change(mutate):
            self.assertEqual(self.renderer.user_boundaries(self.body, text, tokens, start=3), [])

    # An exact token/text suffix must start with the family's closing, not the next user header.
    def test_append_closing_prefix(self):
        text, tokens = self.rendered()
        self.renderer.append(self.body, text, tokens, 3, [71])
        def mutate(path, body, result):
            if path == "/apply-template" and len(body["messages"]) == 4:
                content = body["messages"][2]["content"]
                return dict(prompt=result["prompt"].replace(content + "<turn|>\n", content))
            return result
        with self.change(mutate):
            text, tokens = self.rendered()
            with self.assertRaisesRegex(JournalConflict, "exact end-of-turn"):
                self.renderer.append(self.body, text, tokens, 3, [71])

    def tool_probe(self, change, error):
        self.body["messages"] = copy.deepcopy(self.fixture["shapes"]["tool_call_and_response"]["messages"])
        text, tokens = self.rendered()
        self.renderer.append(self.body, text, tokens, 3, [71])
        def mutate(path, body, result):
            if path == "/apply-template" and len(body["messages"]) > 3:
                arguments = body["messages"][2].get("tool_calls", [{}])[0].get("function", {}).get("arguments", {})
                if isinstance(arguments, dict) and any(key.startswith("hybrid") for key in arguments):
                    return dict(prompt=change(result["prompt"], next(iter(arguments))))
            return result
        with self.change(mutate), self.assertRaisesRegex(JournalConflict, error):
            self.renderer.append(self.body, text, tokens, 3, [71])

    # A malformed argument probe cannot be accepted just because its trailing suffix still matches.
    def test_gemma_tool_probe_count(self):
        self.tool_probe(lambda text, marker: text.replace(marker, "key", 1), "did not preserve the tool boundary probe")

    # The call terminator itself is checked before slicing off its fixed-width bytes.
    def test_gemma_tool_probe_closing(self):
        self.tool_probe(lambda text, marker: text.replace('<|"|>}<tool_call|>', '<|"|>}<tool_cAll|>'), "tool call closing is not supported")

    # Equal starts and reversed starts cannot persist empty or overlapping message ranges.
    def test_layout_overlap(self):
        text, tokens = self.rendered()
        self.renderer.layout(self.body, text, tokens)
        for prefixes in (("", "", text[:10], text[:20]), ("", text[:20], text[:10], text[:30])):
            with self.subTest(prefixes=prefixes), mock.patch.object(self.renderer, "message_frame", side_effect=[(p, "") for p in prefixes]):
                with self.assertRaisesRegex(JournalConflict, "message boundaries overlap"):
                    self.renderer.layout(self.body, text, tokens)

    # A unique reasoning probe does not permit duplicated content probes.
    def test_duplicate_assistant_probe(self):
        self.body["messages"] = self.body["messages"][:3]
        text, _ = self.rendered()
        self.renderer.message_frame(self.body, text, 2, self.renderer.family(self.body))
        def mutate(path, body, result):
            if path == "/apply-template" and body["messages"][0]["role"] == "system" and body["messages"][-1]["content"].startswith("hybrid"):
                marker = body["messages"][-1]["content"]
                return dict(prompt=result["prompt"].replace(marker, marker + marker))
            return result
        with self.change(mutate), self.assertRaisesRegex(JournalConflict, "boundary probe"):
            self.renderer.message_frame(self.body, text, 2, ("<|turn>", "<turn|>\n"))


class ContinuedTurnTests(unittest.TestCase):
    setUp = TemplateGuardTests.setUp

    def ranges(self, body):
        text = self.renderer.render(body)
        tokens = self.renderer.encode(text, initial=True)
        try:
            layout, _ = self.renderer.layout(body, text, tokens)
        except JournalConflict as error:
            self.fail("supported Gemma continuation rejected: " + str(error))
        raw = self.renderer.token_bytes(text, tokens)
        def byte(position):
            return sum(map(len, raw[:position["n"]])) + position["byte_offset"]
        return [text.encode()[byte(row["start"]):byte(row["end"])].decode() for row in layout]

    # Consecutive assistants own their own content despite sharing one Gemma model turn.
    def test_consecutive_assistants(self):
        self.body["messages"].insert(3, dict(role="assistant", content="A2"))
        segments = self.ranges(self.body)
        self.assertEqual(segments[2], "<|turn>model\nA1")
        self.assertEqual(segments[3], "A2<turn|>\n")

    # Call -> tool -> answer -> user retains the response solely in the tool range, with no false overlap.
    def test_call_tool_answer(self):
        self.body["messages"] = copy.deepcopy(self.fixture["shapes"]["tool_call_and_response"]["messages"])
        self.body["messages"].insert(-1, dict(role="assistant", content="A2"))
        segments = self.ranges(self.body)
        self.assertIn("call:read", segments[2])
        self.assertNotIn("T1", segments[2])
        self.assertIn("T1", segments[3])
        self.assertNotIn("A2", segments[3])
        self.assertEqual(segments[4], "A2<turn|>\n")

    # A continued message's reasoning prefix belongs to that message, including the Gemma reasoning alias.
    def test_continued_reasoning(self):
        self.body["messages"] = self.body["messages"][:3] + [dict(role="assistant", content="A2", reasoning="R2")]
        segments = self.ranges(self.body)
        self.assertEqual(segments[2], "<|turn>model\nA1")
        self.assertEqual(segments[3], "<|channel>thought\nR2\n<channel|>A2")

    # An empty merged assistant has no byte range; interleaved call content is not a contiguous message range.
    def test_unrepresentable_messages(self):
        self.body["messages"].insert(3, dict(role="assistant", content=""))
        self.body["messages"].insert(4, dict(role="assistant", content="A2"))
        with self.assertRaisesRegex(JournalConflict, "message boundaries overlap"):
            text = self.renderer.render(self.body)
            self.renderer.layout(self.body, text, self.renderer.encode(text, initial=True))
        self.body["messages"] = copy.deepcopy(self.fixture["shapes"]["tool_call_and_response"]["messages"])
        self.body["messages"][2]["content"] = "after-tool"
        with self.assertRaisesRegex(JournalConflict, "interleaves assistant content"):
            text = self.renderer.render(self.body)
            self.renderer.layout(self.body, text, self.renderer.encode(text, initial=True))


class OracleTests(unittest.TestCase):
    # Source-free captures keep all derived outputs and bind external template bytes before executing Jinja.
    def test_sources_external(self):
        for path in FIXTURES.glob("*.json"):
            fixture = json.loads(path.read_text())
            for key in ("chat_template", "chat_template_tool_use"):
                self.assertNotIn(key, fixture["props"])
                self.assertIn(key + "_sha256", fixture["props"])
            loaded = source_fixture(fixture)
            self.assertEqual(hashlib.sha256(loaded["props"]["chat_template"].encode()).hexdigest(), fixture["props"]["chat_template_sha256"])

    # Absent sources give an explicit skip; a present but changed source is a hard digest failure.
    def test_source_missing_and_mismatch(self):
        fixture = json.loads((FIXTURES / "gemma4-31b.json").read_text())
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(unittest.SkipTest, "external template source absent"):
                source_fixture(fixture, root)
            (Path(root) / "gemma4-31b.json").write_text(json.dumps(dict(props=dict(chat_template="changed"))))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                source_fixture(fixture, root)

    # The reviewer oracle matches 88 captures; eight assistant-prefill renders need server assembly and are excluded.
    def test_oracle_captured_renders(self):
        oracles = {}
        count = 0
        for fixture, name, variant, body, captured in cases():
            if name == "assistant_last" and not fixture["label"].startswith("qwen"):
                continue
            if fixture["label"] not in oracles:
                oracles[fixture["label"]] = TemplateOracle(fixture)
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                self.assertEqual(oracles[fixture["label"]].render(body), captured["text"])
                count += 1
        self.assertEqual(count, 88)

    # Both new Gemma continued-turn shapes match the real template, independently of the frame code.
    def test_gemma_continued_renders(self):
        fixture = json.loads((FIXTURES / "gemma4-31b.json").read_text())
        oracle = TemplateOracle(fixture)
        for shape, tool in (("system_user_assistant_user", False), ("tool_call_and_response", True)):
            body = dict(messages=copy.deepcopy(fixture["shapes"][shape]["messages"]), chat_template_kwargs=dict(enable_thinking=False))
            body["messages"].insert(-1, dict(role="assistant", content="A2"))
            for agp in (False, True):
                body["add_generation_prompt"] = agp
                with self.subTest(tool=tool, agp=agp):
                    self.assertEqual(FixtureTemplate(fixture).render(body), oracle.render(body))

    # Frame, user and append probes execute the external oracle on the 84 supported captured variants.
    def test_oracle_probes(self):
        oracles = {}
        supported = 0
        for fixture, name, variant, body, captured in cases():
            if name == "assistant_last" and not fixture["label"].startswith("qwen"):
                continue
            if fixture["label"] not in oracles:
                oracles[fixture["label"]] = TemplateOracle(fixture)
            oracle = oracles[fixture["label"]]
            node = ReplayNode(fixture)
            self.addCleanup(node.state.close)
            node.template.props = oracle.fixture["props"]
            node.template.render = oracle.render
            renderer = Renderer(node)
            text = captured["text"]
            tokens = [p["id"] for p in captured["tokens_add_special"]]
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                if fixture["label"].startswith("gemma") and name == "bench_tool_without_call":
                    with self.assertRaisesRegex(JournalConflict, "boundary probe"):
                        renderer.layout(body, text, tokens)
                    continue
                expected = ReplayNode(fixture)
                self.addCleanup(expected.state.close)
                stub = Renderer(expected)
                self.assertEqual(renderer.layout(body, text, tokens), stub.layout(body, text, tokens))
                self.assertEqual(renderer.user_boundaries(body, text, tokens), stub.user_boundaries(body, text, tokens))
                if name in ("system_user_assistant_user", "reasoning_assistant", "bench_tool_without_call", "tool_call_and_response", "two_rounds"):
                    past = 5 if name == "two_rounds" else 3
                    self.assertEqual(renderer.append(body, text, tokens, past, [71]), stub.append(body, text, tokens, past, [71]))
                supported += 1
        self.assertEqual(supported, 84)

    # The continued-turn regression exercises real Jinja probes as well as the source-free stub.
    def test_oracle_continued_frames(self):
        fixture = json.loads((FIXTURES / "gemma4-31b.json").read_text())
        oracle = TemplateOracle(fixture)
        node = ProbeNode(fixture)
        self.addCleanup(node.state.close)
        node.template.render = oracle.render
        node.template.props = oracle.fixture["props"]
        self.renderer = Renderer(node)
        for shape in ("system_user_assistant_user", "tool_call_and_response"):
            body = dict(messages=copy.deepcopy(fixture["shapes"][shape]["messages"]), chat_template_kwargs=dict(enable_thinking=False))
            body["messages"].insert(-1, dict(role="assistant", content="A2"))
            segments = ContinuedTurnTests.ranges(self, body)
            self.assertEqual(segments[-2], "A2<turn|>\n")
            self.assertNotIn("A1", segments[-2])
            self.assertNotIn("T1", segments[-2])


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

    # Continued-turn ranges survive HTTP admission and persistence; IDs here are synthetic character pieces.
    def test_http_gemma_continued_ranges(self):
        fixture = self.configure()
        probe = ProbeNode(fixture)
        self.addCleanup(probe.state.close)
        def tokenize(text, initial=False, with_pieces=False):
            return probe.request("POST", "/tokenize", dict(content=text, add_special=initial, with_pieces=with_pieces))["tokens"]
        self.decode.app.template_fixture.tokenize = tokenize
        for shape in ("system_user_assistant_user", "tool_call_and_response"):
            body = dict(messages=copy.deepcopy(fixture["shapes"][shape]["messages"]), chat_template_kwargs=dict(enable_thinking=False))
            body["messages"].insert(-1, dict(role="assistant", content="A2"))
            with self.client.open("POST", "/v1/chat/completions", body) as response:
                sid = response.headers["X-Hybrid-Session"]
                response.read()
            with self.app.reserve(sid):
                pass
            session = self.app.store.get(sid)
            text = self.app.renderer.render(body)
            raw = self.app.renderer.token_bytes(text, self.app.renderer.encode(text, initial=True))
            def byte(position):
                return sum(map(len, raw[:position["n"]])) + position["byte_offset"]
            ranges = session["journal"]["message_boundaries"]
            index = len(body["messages"]) - 2
            row = next(row for row in ranges if row["message_index"] == index)
            self.assertEqual(text.encode()[byte(row["start"]):byte(row["end"])], b"A2<turn|>\n")
            self.assertEqual(self.decode.app.info(self.decode.app.slots[session["slot"]])["tokens"], session["tokens"])

    # Each r20 shape persists captured token IDs/ranges through HTTP with the benchmark's generation/thinking flags.
    def test_http_r20_shapes(self):
        for fixture, name, variant, body, captured in cases(r20=True):
            if variant != "agp/enable_thinking_false":
                continue
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                self.decode.app.template_fixture = FixtureTemplate(fixture)
                with self.client.open("POST", "/v1/chat/completions", body) as response:
                    sid = response.headers["X-Hybrid-Session"]
                    response.read()
                with self.app.reserve(sid):
                    pass
                session = self.app.store.get(sid)
                tokens = [p["id"] for p in captured["tokens_add_special"]]
                self.assertEqual(session["tokens"][:len(tokens)], tokens)
                layout, _ = self.app.renderer.layout(body, captured["text"], tokens)
                self.assertEqual(session["journal"]["message_boundaries"][:len(layout)], layout)
                self.assertEqual(self.decode.app.info(self.decode.app.slots[session["slot"]])["tokens"], session["tokens"])

    # Gemma's unsupported unmatched tool response is rejected before journal, slot or remote state changes.
    def test_http_gemma_unmatched_tool_atomic(self):
        fixture = self.configure()
        body = dict(messages=copy.deepcopy(fixture["shapes"]["system_user"]["messages"]), chat_template_kwargs=dict(enable_thinking=False))
        with self.client.open("POST", "/v1/chat/completions", body) as response:
            sid = response.headers["X-Hybrid-Session"]
            response.read()
        with self.app.reserve(sid):
            pass
        before = self.snapshot(sid)
        body["messages"] = before[0]["journal"]["messages"] + copy.deepcopy(fixture["shapes"]["bench_tool_without_call"]["messages"][-2:])
        with self.assertRaises(HTTPError) as error:
            self.client.request("POST", "/v1/chat/completions", body, {"X-Hybrid-Session": sid})
        self.assertEqual(error.exception.status, 409)
        self.assertIn("boundary probe", error.exception.body.decode())
        self.assertEqual(self.snapshot(sid), before)

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


class CapturedR20Tests(unittest.TestCase):
    # All 96 real r20 renders and both captured token modes match, including fixed tool definitions.
    def test_replay(self):
        count = 0
        for fixture, name, variant, body, captured in cases(r20=True):
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                template = FixtureTemplate(fixture)
                self.assertEqual(template.render(body), captured["text"])
                for initial, key in ((False, "tokens_plain"), (True, "tokens_add_special")):
                    self.assertEqual(template.tokenize(captured["text"], initial, True), captured[key])
                count += 1
        self.assertEqual(count, 96)

    def check_ranges(self, renderer, body, captured):
        text = captured["text"]
        tokens = [p["id"] for p in captured["tokens_add_special"]]
        layout, end = renderer.layout(body, text, tokens)
        raw = renderer.token_bytes(text, tokens)
        def byte(position):
            return sum(map(len, raw[:position["n"]])) + position["byte_offset"]
        opening = renderer.family(body)[0]
        search, previous = 0, None
        segments = []
        self.assertEqual([row["message_index"] for row in layout], list(range(len(body["messages"]))))
        for message, row in zip(body["messages"], layout):
            role = message["role"]
            if role == "tool":
                header = "<|tool_response>" if opening == "<|turn>" else "<|im_start|>user\n<tool_response>\n" if opening == "<|im_start|>" else "<|start|>tool "
            elif opening == "<|turn>" and role == "assistant" and previous == "assistant":
                header = "<|tool_call>" if message.get("tool_calls") else message["content"]
            else:
                header = opening + ("model" if role == "assistant" and opening == "<|turn>" else role)
            expected = text.encode().index(header.encode(), search)
            self.assertEqual(byte(row["start"]), expected)
            self.assertLess(byte(row["start"]), byte(row["end"]))
            segment = text.encode()[byte(row["start"]):byte(row["end"])].decode()
            segments.append(segment)
            search = expected + len(header.encode())
            if role != "tool":
                previous = role
        self.assertEqual(byte(end), len(renderer.render(dict(body, add_generation_prompt=False)).encode()))
        for index, message in enumerate(body["messages"]):
            if message["role"] == "tool":
                payload = message["content"].strip()
                self.assertIn(payload, segments[index])
                self.assertTrue(all(payload not in segment for i, segment in enumerate(segments) if i != index))
                if payload == "x x x x":
                    row = layout[index]
                    pieces = captured["tokens_add_special"][row["start"]["n"]:row["end"]["n"]]
                    filler = [p for p in pieces if p["piece"] in ("x", " x")]
                    self.assertEqual(len(filler), 4)
                    self.assertEqual("".join(p["piece"] for p in filler).strip(), payload)
        self.assertEqual([b["message_index"] for b in renderer.user_boundaries(body, text, tokens)],
                         [i for i, m in enumerate(body["messages"]) if m["role"] == "user"])
        return tokens

    # Starts are checked independently against real bytes; continued calls/answers and each response own disjoint ranges.
    def test_ranges(self):
        for fixture, name, variant, body, captured in cases(r20=True):
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                node = ReplayNode(fixture)
                self.addCleanup(node.state.close)
                self.check_ranges(Renderer(node), body, captured)

    # Real r20 suffixes retain tool payloads and anchor IDs; a merged Gemma A1 cannot supply an old closing anchor.
    def test_append(self):
        for fixture, name, variant, body, captured in cases(r20=True):
            if name == "system_user_tools":
                continue
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                node = ReplayNode(fixture)
                self.addCleanup(node.state.close)
                renderer = Renderer(node)
                text, tokens = captured["text"], [p["id"] for p in captured["tokens_add_special"]]
                past = 4 if name == "consecutive_assistants" or name.startswith("bench_tool") else 5 if name.startswith("call_tool_answer") else 3
                prompt, _, _, tail = renderer.append(body, text, tokens, past, [71, 72])
                self.assertEqual(prompt, [71, 72] + tail)
                layout, _ = renderer.layout(body, text, tokens, start=past, token_base=len(tokens) - len(tail), anchor_base=2)
                self.assertTrue(all(row["start"]["n"] >= 2 for row in layout))
                if name.startswith("bench_tool"):
                    self.assertIn("x x x x", b"".join(renderer.token_bytes(text, tokens)[len(tokens) - len(tail):]).decode())
                    if fixture["label"].startswith("gemma"):
                        with self.assertRaisesRegex(JournalConflict, "exact end-of-turn"):
                            renderer.append(body, text, tokens, 3, [71, 72])

    # An uncaptured tool schema cannot borrow the captured preamble or pretend to have token evidence.
    def test_uncaptured_tools(self):
        for fixture, _, _, body, _ in cases(r20=True):
            if body.get("tools"):
                template = FixtureTemplate(fixture)
                self.assertTrue(template.render(body))
                body = copy.deepcopy(body)
                body["tools"][0]["function"]["name"] = "unknown"
                with self.assertRaisesRegex(ValueError, "outside captured schema"):
                    template.render(body)


class R20OracleTests(unittest.TestCase):
    # The external digest-checked oracle validates all 96 captures and the stub's live layout/append probes.
    def test_captures_and_probes(self):
        oracles = {}
        for fixture, name, variant, body, captured in cases(r20=True):
            if fixture["label"] not in oracles:
                oracles[fixture["label"]] = TemplateOracle(fixture)
            with self.subTest(model=fixture["label"], shape=name, variant=variant):
                oracle = oracles[fixture["label"]]
                self.assertEqual(oracle.render(body), captured["text"])
                node = ReplayNode(fixture)
                self.addCleanup(node.state.close)
                node.template.props = oracle.fixture["props"]
                node.template.render = oracle.render
                renderer = Renderer(node)
                CapturedR20Tests.check_ranges(self, renderer, body, captured)
                if name != "system_user_tools":
                    past = 4 if name == "consecutive_assistants" or name.startswith("bench_tool") else 5 if name.startswith("call_tool_answer") else 3
                    tokens = [p["id"] for p in captured["tokens_add_special"]]
                    stub = ReplayNode(fixture)
                    self.addCleanup(stub.state.close)
                    self.assertEqual(renderer.append(body, captured["text"], tokens, past, [71]),
                                     Renderer(stub).append(body, captured["text"], tokens, past, [71]))


if __name__ == "__main__":
    unittest.main()
