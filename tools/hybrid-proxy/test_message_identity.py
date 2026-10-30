"""Typed message identity regressions and an opt-in native rewrite witness; no builds."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest


TOOL_EDITS = (
    ("one-bool", '{"x":1}', '{"x":true}'),
    ("zero-bool", '{"x":0}', '{"x":false}'),
    ("int-float", '{"x":1}', '{"x":1.0}'),
    ("null-absent", '{"x":null}', '{}'),
    ("string-int", '{"x":"1"}', '{"x":1}'),
    ("list-bool", '{"x":[1]}', '{"x":[true]}'),
    ("object-bool", '{"x":{"y":0}}', '{"x":{"y":false}}'),
    ("nested-float", '{"x":[{"y":[1]}]}', '{"x":[{"y":[1.0]}]}'),
)
PRESENTATIONS = (
    ("identical", '{"x":1}', '{"x":1}'),
    ("key-order", '{"x":1,"y":2}', '{ "y": 2, "x": 1 }'),
    ("unicode", r'{"x":"\u00e9"}', '{"x":"\u00e9"}'),
)


def tool_message(arguments):
    return dict(role="assistant", content="", tool_calls=[dict(id="call-one", type="function",
                function=dict(name="lookup", arguments=arguments))])


class IdentityCases(unittest.TestCase):
    # Both edit directions must diverge for scalar types, absent keys and nested JSON argument values.
    def test_tool_argument_scalar_type_is_a_rewrite(self):
        from journal import matching_messages
        for name, left, right in TOOL_EDITS:
            for old, new in ((left, right), (right, left)):
                with self.subTest(case=name, old=old, new=new):
                    self.assertEqual(matching_messages([tool_message(old)], [tool_message(new)]), 0)

    # Whitespace, object key order and equivalent Unicode escapes preserve the same parsed JSON value.
    def test_same_json_value_control(self):
        from journal import matching_messages
        for name, old, new in PRESENTATIONS:
            with self.subTest(case=name):
                self.assertEqual(matching_messages([tool_message(old)], [tool_message(new)]), 1)

    # Message content parts and other fields use the same recursive scalar identity, not Python equality.
    def test_content_parts_and_message_fields_keep_json_types(self):
        from journal import matching_messages
        for name, left, right in TOOL_EDITS:
            for old, new in ((json.loads(left), json.loads(right)), (json.loads(right), json.loads(left))):
                for field in ("content", "name", "tool_call_id", "reasoning_content", "metadata"):
                    with self.subTest(case=name, field=field, old=old):
                        a = dict(role="assistant", content=[dict(type="text", text="same")])
                        b = copy.deepcopy(a)
                        a[field], b[field] = [dict(value=old)], [dict(value=new)]
                        self.assertEqual(matching_messages([a], [b]), 0)

    # Null, false, zero, empty text and a missing message field are distinct; omitted assistant reasoning is allowed.
    def test_null_and_absent_message_fields_are_not_defaults(self):
        from journal import matching_messages
        for field in ("content", "name", "tool_calls", "tool_call_id", "reasoning_content", "metadata"):
            for role in ("user", "assistant", "tool"):
                for value in (None, False, 0, 0.0, ""):
                    missing, present = dict(role=role), dict(role=role, **{field: value})
                    for old, new in ((missing, present), (present, missing)):
                        expected = int(role == "assistant" and field == "reasoning_content" and field not in new)
                        with self.subTest(field=field, role=role, value=value, supplied=field in new):
                            self.assertEqual(matching_messages([old], [new]), expected)
        for a, b in ((None, ""), (False, 0), (0, 0.0), ("1", 1)):
            self.assertEqual(matching_messages([dict(role="assistant", content=a)],
                                              [dict(role="assistant", content=b)]), 0)

    # Decoded HTTP content has the same Unicode characters regardless of transport escape spelling or object order.
    def test_content_presentation_and_identical_history_append_identity(self):
        from journal import matching_messages
        left = json.loads(r'{"role":"user","content":[{"type":"text","text":"caf\u00e9"}]}')
        right = dict(content=[dict(text="caf\u00e9", type="text")], role="user")
        self.assertEqual(matching_messages([left], [right]), 1)
        self.assertEqual(matching_messages([left, tool_message('{"x":1}')],
                                          [right, tool_message('{ "x": 1 }'), dict(role="user", content="next")]), 2)

    # Durable selection and pending admission must not reuse a history after a scalar-type edit.
    def test_store_and_pending_admission_use_typed_identity(self):
        import tempfile
        from proxy import Proxy
        with tempfile.TemporaryDirectory() as directory:
            app = Proxy(dict(model_sha256="fixture", prefill_url="http://127.0.0.1:1", slots=2,
                             decode_url="http://127.0.0.1:1", slot_path=directory,
                             clients={"fixture": dict(id="a", prefill_token="unused")}),
                        str(Path(directory) / "proxy.sqlite"))
            try:
                for name, left, right in TOOL_EDITS:
                    for old, new in ((left, right), (right, left)):
                        messages = [dict(role="user", content="start"), tool_message(old)]
                        stored = app.store.select("a", [])
                        stored["journal"] = dict(identity=name, messages=messages,
                                                 checkpoints=[dict(message_index=0, n=0)])
                        app.store.save(stored)
                        follow = [messages[0], tool_message(new), dict(role="assistant", content="answer"),
                                  dict(role="user", content="next")]
                        self.assertIsNone(app.store.select_messages("a", follow, name, [999], create=False))
                        app.inflight[stored["id"]] = dict(id=stored["id"], client="a", identity=name, messages=messages)
                        with app.lock:
                            selected = app.select_messages("a", follow, name, [999])
                        self.assertTrue(selected.get("_fork"), name)
                        app.inflight.clear()
                        # Repeated directions must not accidentally select a previous case's completed journal.
                        with app.store.db:
                            app.store.db.execute("DELETE FROM sessions")
            finally:
                app.store.close()


def native_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--proxy-source")
    parser.add_argument("--no-checkpoint-control", action="store_true")
    args = parser.parse_args()
    if args.proxy_source:
        sys.path.insert(0, str(Path(args.proxy_source).resolve()))
    from native_fixture import native_server, rendered_tokens, require
    from journal import JournalConflict
    from proxy import Proxy
    from store import prefix_hash
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "source-identity.json").write_text(json.dumps({name: hashlib.sha256(
        Path(__import__(name).__file__).read_bytes()).hexdigest() for name in ("journal", "proxy", "store")}, indent=2))
    cases = [(name + suffix, old, new, False) for name, left, right in TOOL_EDITS
             for suffix, old, new in (("-forward", left, right), ("-reverse", right, left))]
    cases += [(name, old, new, True) for name, old, new in PRESENTATIONS]
    results = []
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    with native_server(args.binary, args.model, root, checkpoints=0 if args.no_checkpoint_control else 32) as (decode, slots):
        for index, (name, old, new, equal) in enumerate(cases):
            app = Proxy(dict(model_sha256=model_hash,
                prefill_url="http://127.0.0.1:1", prefill_timeout=0.1, decode_url=decode.url, decode_timeout=120,
                slots=1, slot_path=str(slots), clients={"fixture": dict(id="a", prefill_token="unused")}),
                str(root / (str(index) + ".sqlite")))
            client = app.clients["fixture"]

            def complete(body):
                with app.prepare(client, body, "identity") as (session, prompt, request):
                    response = decode.request("POST", "/v1/chat/completions", request)
                    session["_capture"].feed(response)
                    app.finish(session, prompt)
                row = app.store.get("identity")
                info = decode.request("GET", "/slots/0/info")
                require(info["tokens"] == row["tokens"] and info["prefix_hash"] == prefix_hash(row["tokens"]),
                        name + ": native tokens/hash disagree with journal")
                return row

            try:
                body = dict(messages=[dict(role="system", content="Keep this prefix."),
                    dict(role="user", content="Look up x."), tool_message(old),
                    dict(role="tool", tool_call_id="call-one", content="Result one."),
                    dict(role="user", content="Answer briefly.")], temperature=0, seed=0, max_tokens=8,
                    ignore_eos=True, chat_template_kwargs=dict(enable_thinking=False))
                altered = copy.deepcopy(body)
                altered["messages"][2] = tool_message(new)
                before_render, after_render = rendered_tokens(decode, body), rendered_tokens(decode, altered)
                # Unicode escapes are equivalent because the native template parses arguments before rendering.
                if name == "unicode":
                    require(before_render == after_render, "Unicode presentation is not native-token equivalent")
                elif not equal and not name.startswith("string-int"):
                    require(before_render != after_render, name + ": native renderer must distinguish the edit")
                # This template renders string "1" and integer 1 alike; identity still conservatively keeps their types.
                before = complete(body)
                neighbor = decode.request("GET", "/slots/1/info")
                changed = copy.deepcopy(before["journal"]["messages"])
                changed[2] = tool_message(new)
                changed.append(dict(role="user", content="Continue briefly."))
                request = dict(body, messages=changed)
                if args.no_checkpoint_control and not equal:
                    # An unavailable native checkpoint must reject the typed edit without changing either journal or slot.
                    decode.request("POST", "/slots/0?action=save", dict(filename="before.bin"))
                    before_bytes = (slots / "before.bin").read_bytes()
                    try:
                        complete(request)
                    except JournalConflict as error:
                        require("checkpoint" in str(error), "rewrite did not reject an unavailable checkpoint")
                    else:
                        raise AssertionError(name + ": typed edit silently appended without a checkpoint")
                    rejected = app.store.get("identity")
                    require(rejected["tokens"] == before["tokens"] and rejected["journal"] == before["journal"],
                            "rejection changed durable journals")
                    decode.request("POST", "/slots/0?action=save", dict(filename="after.bin"))
                    require((slots / "after.bin").read_bytes() == before_bytes, "rejection changed native bytes")
                    result = dict(case=name, status=409, native_render_equal=before_render == after_render)
                else:
                    after = complete(request)
                    meter = after["journal"]["last_meter"]
                    result = dict(case=name, meter=meter, native_render_equal=before_render == after_render)
                    results.append(result)
                    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                    if equal:
                        # A truly identical or presentation-only history must append, retaining every original ID.
                        require(meter["rollback_n"] is None and meter["prefilled"] == meter["new_tokens"],
                                name + ": identical history did not append")
                        require(after["tokens"][:len(before["tokens"])] == before["tokens"], "append changed old IDs")
                    else:
                        # A typed edit must roll back, and the resulting prompt must contain the changed native tokens.
                        require(meter["rollback_n"] is not None, name + ": typed edit silently appended")
                        checkpoint = max(cp["n"] for cp in before["journal"]["checkpoints"] if cp["message_index"] <= 2)
                        require(meter["rollback_n"] == checkpoint, name + ": rewrite used the wrong user boundary")
                        expected = rendered_tokens(decode, request)
                        require(after["tokens"][:len(expected)] == expected, name + ": rewrite retained stale prompt IDs")
                        require(after["journal"]["messages"][2]["tool_calls"][0]["function"]["arguments"] == new,
                                "message journal lost the edited arguments")
                require(decode.request("GET", "/slots/1/info") == neighbor, "typed edit changed a neighbor slot")
                if args.no_checkpoint_control and not equal:
                    results.append(result)
                (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print("PASS native typed identity " + name + (" append" if equal else " conflict" if args.no_checkpoint_control else " rollback"), flush=True)
            finally:
                app.store.close()


if __name__ == "__main__":
    native_main()
