"""Message identity, persistence and rendering rejection controls; native tests qualify real templates."""
import copy
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest import mock

from journal import Capture, JournalConflict, Renderer, matching_messages, settings_identity
from store import Store
import test_proxy as fixtures
from proxy import HTTPError, RollbackError


class TemplateNode:
    def __init__(self):
        self.mutate_suffix = False

    def request(self, method, path, data=None):
        if path == "/tokenize":
            parts = re.split(r"(<\|im_start\|>|<\|im_end\|>)", data["content"])
            tokens = []
            for part in parts:
                tokens += ([1] if part == "<|im_start|>" else [2] if part == "<|im_end|>" else
                           [100 + ord(c) for c in part])
            if data.get("with_pieces"):
                return dict(tokens=[dict(id=t, piece="<|im_start|>" if t == 1 else "<|im_end|>" if t == 2 else chr(t - 100))
                                    for t in tokens])
            return dict(tokens=tokens)
        if path == "/props":
            return dict(chat_template="fixture")
        messages = data["messages"]
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        text = ""
        for index, message in enumerate(messages):
            content = message.get("content") or ""
            text += "<|im_start|>" + message["role"] + "\n"
            if message["role"] == "assistant" and index > last_user:
                text += "<think>" + message.get("reasoning_content", "") + "</think>"
            text += content + "<|im_end|>\n"
            if self.mutate_suffix and content.startswith("hybrid"):
                text += "probe-only-change"
        return dict(prompt=text + ("<|im_start|>assistant\n<think>" if data.get("add_generation_prompt", True) else ""))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.messages = [dict(role="system", content="system"), dict(role="user", content="first"),
                         dict(role="assistant", content="answer", reasoning_content="private thought")]

    # Reasoning can be omitted, but supplied conflicting reasoning and ordinary message edits are real divergence.
    def test_missing_reasoning_only_is_ignored(self):
        incoming = copy.deepcopy(self.messages)
        incoming[-1].pop("reasoning_content")
        self.assertEqual(matching_messages(self.messages, incoming), 3)
        for value in ("", "edited"):
            incoming[-1]["reasoning_content"] = value
            self.assertEqual(matching_messages(self.messages, incoming), 2)
        incoming = copy.deepcopy(self.messages)
        incoming[1]["content"] = "changed user"
        self.assertEqual(matching_messages(self.messages, incoming), 1)

    def test_tool_output_and_call_identity_are_not_ignored(self):
        old = [dict(role="tool", content="result", tool_call_id="one")]
        for new in (dict(role="tool", content="edited", tool_call_id="one"),
                    dict(role="tool", content="result", tool_call_id="two")):
            self.assertEqual(matching_messages(old, [new]), 0)

    # Tool selection is per-turn grammar; changes to the template, tool schema or thinking settings change render identity.
    def test_render_and_generation_settings_are_distinct(self):
        body = dict(tools=[dict(name="lookup")], chat_template_kwargs=dict(enable_thinking=True))
        identity = settings_identity(body, "template")
        self.assertEqual(identity, settings_identity(dict(body, tool_choice="none"), "template"))
        self.assertNotEqual(identity, settings_identity(body, "another-template"))
        self.assertNotEqual(identity, settings_identity(dict(body, tools=[]), "template"))
        self.assertNotEqual(identity, settings_identity(dict(body, chat_template_kwargs={}), "template"))

    # An append retains authoritative old IDs even if the full template drops prior reasoning.
    def test_checked_suffix_keeps_old_ids_and_reports_user_boundary(self):
        renderer = Renderer(TemplateNode())
        body = dict(messages=self.messages + [dict(role="user", content="next")])
        text = renderer.render(body)
        full = renderer.encode(text, initial=True)
        anchor = [77, 88, 99]
        prompt, boundaries, closing, tail = renderer.append(body, text, full, 3, anchor)
        self.assertEqual(prompt, anchor + tail)
        self.assertEqual(boundaries, [dict(message_index=3, n=len(anchor) + closing)])
        self.assertNotEqual(prompt[:len(anchor)], full[:len(anchor)])
        prompt_eog, _, _, _ = renderer.append(body, text, full, 3, anchor + [2])
        self.assertEqual(prompt_eog, anchor + tail)

    # The template must prove the suffix and tokenizer boundary; a probe-dependent template cannot append.
    def test_probe_dependent_suffix_is_rejected(self):
        node = TemplateNode()
        renderer = Renderer(node)
        body = dict(messages=self.messages + [dict(role="user", content="next")])
        text = renderer.render(body)
        tokens = renderer.encode(text, initial=True)
        node.mutate_suffix = True
        with self.assertRaises(JournalConflict):
            renderer.append(body, text, tokens, 3, [77])

    def test_nonce_collision_never_finds_a_boundary_in_user_text(self):
        renderer = Renderer(TemplateNode())
        body = dict(messages=self.messages + [dict(role="user", content="hybridcollision")])
        text = renderer.render(body)
        with mock.patch("journal.uuid.uuid4", return_value=mock.Mock(hex="collision")):
            with self.assertRaises(JournalConflict):
                renderer.append(body, text, renderer.encode(text, initial=True), 3, [77])

    def test_tokenizer_boundary_mismatch_is_rejected(self):
        renderer = Renderer(TemplateNode())
        body = dict(messages=self.messages + [dict(role="user", content="next")])
        text = renderer.render(body)
        tokens = renderer.encode(text, initial=True)
        tokens[-1] += 1
        with self.assertRaises(JournalConflict):
            renderer.append(body, text, tokens, 3, [77])

    # Boundary records identify each input message and the generation prefix, independently of sampled text.
    def test_layout_records_message_and_generation_boundaries(self):
        renderer = Renderer(TemplateNode())
        body = dict(messages=self.messages[:2])
        text = renderer.render(body)
        tokens = renderer.encode(text, initial=True)
        layout, generated = renderer.layout(body, text, tokens)
        user = renderer.user_boundaries(body, text, tokens)[0]
        self.assertEqual(layout[0]["start"], dict(n=0, byte_offset=0))
        self.assertEqual(layout[1]["start"], dict(n=user["n"], byte_offset=0))
        self.assertEqual(layout[1]["end"], generated)
        self.assertLess(generated["n"], len(tokens))

    # The sampled ID vector and fragmented tool fields are captured independently of visible content.
    def test_stream_capture_preserves_ids_and_tool_fragments(self):
        capture = Capture()
        capture.feed(dict(choices=[dict(delta=dict(reasoning_content="thought", tool_calls=[
            dict(index=0, id="call", function=dict(name="look", arguments='{"x":'))]))]))
        capture.feed(dict(choices=[dict(delta=dict(tool_calls=[
            dict(index=0, function=dict(name="up", arguments="1}"))]))], __verbose=dict(tokens=[1, 2, 3])))
        self.assertEqual(capture.native["tokens"], [1, 2, 3])
        self.assertEqual(capture.message["tool_calls"][0]["function"], dict(name="lookup", arguments='{"x":1}'))
        with self.assertRaises(JournalConflict):
            capture.feed(dict(choices=[dict(delta=dict(tool_calls=[dict(index=1000000)]))]))

    # An existing SQLite database migrates without losing its journal, and remains client scoped after reopening.
    def test_store_migration_message_mapping_fork_and_foreign(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite"
            db = sqlite3.connect(path)
            db.executescript("CREATE TABLE meta(model TEXT NOT NULL); INSERT INTO meta VALUES('model');"
                "CREATE TABLE sessions(id TEXT PRIMARY KEY,client TEXT NOT NULL,tokens TEXT NOT NULL,"
                "hashes TEXT NOT NULL,remote_id TEXT,remote_n INTEGER NOT NULL DEFAULT 0,slot INTEGER,"
                "used REAL NOT NULL,offload INTEGER NOT NULL DEFAULT 0);")
            db.close()
            store = Store(str(path), "model")
            session = store.select("a", [1, 2])
            session["tokens"] = [1, 2, 3]
            session["journal"] = dict(identity="render", messages=self.messages,
                segments=[dict(kind="generated", start=2, end=3, sampled_ids=[3])],
                checkpoints=[dict(message_index=1, n=1)], evaluated_boundary=3, sampled_boundary=3)
            store.save(session)
            staged = copy.deepcopy(session)
            staged["_turn"] = {}
            staged["tokens"] += [4]
            staged["journal"]["messages"] += [dict(role="user", content="pending")]
            store.save(staged)
            self.assertEqual(store.get(session["id"])["tokens"], session["tokens"])
            self.assertEqual(store.get(session["id"])["journal"], session["journal"])
            incoming = copy.deepcopy(self.messages)
            incoming[-1].pop("reasoning_content")
            incoming += [dict(role="user", content="next")]
            self.assertEqual(store.select_messages("a", incoming, "render", [999])["id"], session["id"])
            changed = copy.deepcopy(self.messages)
            changed[1]["content"] = "another conversation"
            fork = store.select_messages("a", changed, "render", [1, 2, 3])
            self.assertTrue(fork["_fork"])
            self.assertEqual(fork["id"], session["id"])
            with self.assertRaises(KeyError):
                store.select_messages("foreign", incoming, "render", [999], session["id"])
            store.close()
            store = Store(str(path), "model")
            self.assertEqual(store.get(session["id"])["journal"], session["journal"])
            store.close()


class JournalHTTPTests(unittest.TestCase):
    setUp = fixtures.HTTPTests.setUp
    tearDown = fixtures.HTTPTests.tearDown
    start = fixtures.HTTPTests.start
    url = staticmethod(fixtures.HTTPTests.url)
    chat = fixtures.HTTPTests.chat

    # A failed journal commit cannot make unknown native generations part of a completed message history.
    def test_unknown_native_extension_is_never_adopted_into_message_journal(self):
        sid, _ = self.chat([10])
        session = self.app.store.get(sid)
        session["journal"] = dict(messages=[dict(role="user", content="old")])
        self.app.store.save(session)
        self.decode.app.slots[session["slot"]].append(999)
        self.app.verified(sid, False)
        with self.app.reserve(sid), self.app.slot_lease(session):
            self.assertEqual(self.app.decode.request("GET", f"/slots/{session['slot']}/info")["n"], 0)
            self.assertEqual(self.app.store.get(sid)["tokens"], [10, 901, 902])

    # A caller cannot replace the proxy's authoritative token input with a conflicting native API field.
    def test_external_prompt_tokens_is_rejected_without_slot_or_store_changes(self):
        sid, _ = self.chat([10])
        before = self.app.store.get(sid)
        info = self.app.decode.request("GET", f"/slots/{before['slot']}/info")
        with self.assertRaises(HTTPError) as error:
            self.client.request("POST", "/v1/chat/completions", dict(messages=[], _tokens=[10], prompt_tokens=[999]))
        self.assertEqual(error.exception.status, 400)
        after = self.app.decode.request("GET", f"/slots/{before['slot']}/info")
        self.assertEqual({key: after[key] for key in ("n", "tokens", "prefix_hash")},
                         {key: info[key] for key in ("n", "tokens", "prefix_hash")})
        self.assertEqual(self.app.store.get(sid), before)

    # Rollback and rename failure retain the backup and revoke the binding; a later lease clears the dirty slot.
    def test_rewrite_rollback_and_backup_rename_failure_invalidate_binding(self):
        sid, _ = self.chat([10])
        session = self.app.store.get(sid)
        session["tokens"] = [1, 2, 3]
        session["journal"] = dict(identity="render", messages=[dict(role="system", content="sys"),
            dict(role="user", content="old"), dict(role="assistant", content="answer")],
            checkpoints=[dict(message_index=1, n=1, prefix_hash=fixtures.prefix_hash([1]))])
        self.app.store.save(session)
        body = dict(messages=[dict(role="system", content="sys"), dict(role="user", content="edited")])
        backup = Path(self.temp.name) / (sid + "-rewrite-backup.bin")

        def failure(method, path, data=None, **kwargs):
            if "action=save" in path:
                backup.write_bytes(b"retained-native-backup")
                return {}
            raise OSError("injected decoder outage")
        with mock.patch.object(self.app.renderer, "render", return_value="rendered"), \
             mock.patch.object(self.app.renderer, "encode", side_effect=lambda text, **kw: [9] if text.startswith("<|") else [1, 9]), \
             mock.patch.object(self.app.renderer, "user_boundaries", return_value=[dict(message_index=1, n=1)]), \
             mock.patch.object(self.app.decode, "request", side_effect=failure), \
             mock.patch("proxy.os.replace", side_effect=OSError("injected rename failure")):
            with self.assertRaises(RollbackError):
                with self.app.prepare_messages(self.app.clients["fixture-a"], body, session, [1, 9], "render", "template"):
                    self.fail("failed checkpoint probe must not yield a request")
        self.assertTrue(backup.exists())
        self.assertFalse(self.app.verified(sid))
        fresh = self.app.store.get(sid)
        self.assertIsNone(fresh["slot"])
        self.assertEqual(fresh["tokens"], [1, 2, 3])
        with self.app.reserve(sid), self.app.slot_lease(fresh):
            self.assertEqual(self.app.decode.request("GET", f"/slots/{fresh['slot']}/info")["n"], 0)
        self.assertTrue(backup.exists())


if __name__ == "__main__":
    unittest.main()
