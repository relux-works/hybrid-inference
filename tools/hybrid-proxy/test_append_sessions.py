"""Opt-in three-user/tool-loop append, independent token oracle, rollback and M0.1 witness."""
import argparse
import copy
import hashlib
import http.server
import io
import json
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--omit-reasoning", action="store_true")
    parser.add_argument("--inferred", action="store_true")
    parser.add_argument("--proxy-source")
    parser.add_argument("--witness", action="store_true")
    parser.add_argument("--no-checkpoint-control", action="store_true")
    args = parser.parse_args()
    if args.proxy_source:
        sys.path.insert(0, str(Path(args.proxy_source).resolve()))
    from native_fixture import native_server, rendered_tokens, require
    from journal import Capture
    from proxy import Handler, HTTPError, Node, Proxy
    from store import prefix_hash

    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "proxy-identity.json").write_text(json.dumps(dict(command=sys.argv,
        sources={name: dict(path=str(Path(sys.modules[name].__file__).resolve()),
                           sha256=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest())
                 for name in ("proxy", "store", "policy", "concurrency", "journal")},
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    thinking = args.thinking == "on"
    records, violations, sessions = [], [], []
    with native_server(args.binary, args.model, root, args.template,
                       checkpoints=0 if args.no_checkpoint_control else 32) as (decode, slots):
        app = Proxy(dict(model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            prefill_url="http://127.0.0.1:1", prefill_timeout=0.1, decode_url=decode.url, slots=2,
            decode_timeout=120, slot_path=str(slots), clients={
                "fixture-a": dict(id="a", prefill_token="remote-a"),
                "fixture-b": dict(id="b", prefill_token="remote-b")}, policy=dict(link_rate=1)),
            str(root / "proxy.sqlite"))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.app = app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = Node(f"http://127.0.0.1:{server.server_port}", "fixture-a", timeout=120)
        foreign = Node(client.url, "fixture-b", timeout=120)
        native_records = []
        opened = app.decode.open
        phase = "initial"

        class Stream:
            def __init__(self, response, request):
                self.response, self.request = response, copy.deepcopy(request)
                self.native = None

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *exc):
                if self.native is not None:
                    native_records.append(dict(phase=phase, request=self.request, native=self.native))
                return self.response.__exit__(*exc)

            def __iter__(self):
                for line in self.response:
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        chunk = json.loads(line[6:])
                        if "__verbose" in chunk:
                            self.native = chunk["__verbose"]
                    yield line

        def capture(method, path, data=None, headers=None):
            response = opened(method, path, data, headers)
            if path != "/v1/chat/completions":
                return response
            if data.get("stream"):
                return Stream(response, data)
            with response:
                raw = response.read()
            value = json.loads(raw)
            if data.get("n_predict", data.get("max_tokens", 1)) != 0:
                native_records.append(dict(phase=phase, request=copy.deepcopy(data), native=value["__verbose"]))
            return io.BytesIO(raw)
        app.decode.open = capture

        def chat(connection, body, sid=None):
            headers = {"X-Hybrid-Session": sid} if sid else {}
            result = Capture()
            try:
                response = connection.open("POST", "/v1/chat/completions", body, headers)
            except HTTPError as error:
                (root / (phase + "-error.json")).write_text(json.dumps(dict(status=error.status,
                    body=error.body.decode(errors="replace")), indent=2))
                raise
            with response:
                selected = response.headers["X-Hybrid-Session"]
                if body.get("stream"):
                    for line in response:
                        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                            result.feed(json.loads(line[6:]))
                else:
                    result.feed(json.loads(response.read()))
            with app.reserve(selected):
                pass
            return selected, result.message

        tool = dict(type="function", function=dict(name="lookup", description="Return a color.",
                    parameters=dict(type="object", properties=dict(color=dict(type="string", enum=["blue"])),
                                    required=["color"], additionalProperties=False)))
        common = dict(model=Path(args.model).name, temperature=0, seed=0, return_tokens=True, verbose=True,
                      chat_template_kwargs=dict(enable_thinking=thinking), tools=[tool], parallel_tool_calls=False,
                      reasoning_budget_tokens=16)
        history = [dict(role="system", content="Use lookup when asked. After its result, count using decimal digits.")]
        anchor, sid = [], None
        known_reasoning = {}

        def independent_suffix(message, previous_ids):
            # Independent oracle: these literal frames are read from the two supplied Qwen templates, not Renderer.
            if message["role"] == "user":
                frame = "<|im_start|>user\n" + message["content"].strip() + "<|im_end|>\n"
            else:
                frame = "<|im_start|>user\n<tool_response>\n" + message["content"].strip() + "\n</tool_response><|im_end|>\n"
            ending = decode.request("POST", "/tokenize", dict(content="<|im_end|>"))["tokens"][0]
            close = "\n" if previous_ids and previous_ids[-1] == ending else "<|im_end|>\n"
            frame = close + frame + "<|im_start|>assistant\n"
            frame += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
            return decode.request("POST", "/tokenize", dict(content=frame))["tokens"]

        try:
            if args.no_checkpoint_control:
                # With checkpoints disabled an explicit rewrite is atomic 409; inferred history instead forks.
                body = dict(common, messages=[dict(role="system", content="Retain this nonzero system prefix."),
                            dict(role="user", content="Count decimal digits from 1.")],
                            tool_choice="none", max_tokens=64, ignore_eos=True)
                sid, message = chat(client, body)
                before = app.store.get(sid)
                slot = before["slot"]
                decode.request("POST", f"/slots/{slot}?action=save", dict(filename="no-checkpoint-before.bin"))
                before_bytes = (slots / "no-checkpoint-before.bin").read_bytes()
                changed = dict(body, messages=[body["messages"][0], dict(role="user", content="Count decimal digits from 100.")])
                try:
                    chat(client, changed, sid)
                except HTTPError as error:
                    require(error.status == 409, "missing checkpoint did not return 409")
                else:
                    raise AssertionError("explicit rewrite silently prefills without a checkpoint")
                require(app.store.get(sid)["tokens"] == before["tokens"], "rejected rewrite changed durable IDs")
                decode.request("POST", f"/slots/{slot}?action=save", dict(filename="no-checkpoint-after.bin"))
                require(before_bytes == (slots / "no-checkpoint-after.bin").read_bytes(), "rejected rewrite changed native bytes")
                fork, _ = chat(client, changed)
                require(fork != sid, "inferred rewrite without a checkpoint did not fork")
                forked = app.store.get(fork)
                info = decode.request("GET", f"/slots/{forked['slot']}/info")
                require(info["tokens"] == forked["tokens"] and info["prefix_hash"] == prefix_hash(forked["tokens"]),
                        "forked token prefix/hash differs from its journal")
                print("PASS absent checkpoint: explicit409, native bytes unchanged, inferred fork", flush=True)
                return

            # Three user turns each execute a real generated tool call, then a 64-token answer after the tool result.
            for turn in range(3):
                for kind in ("tool", "answer"):
                    phase = f"turn-{turn + 1}-{kind}"
                    if kind == "tool":
                        incoming = dict(role="user", content=f"Turn {turn + 1}: call lookup with color blue, then count decimal integers from 1 to 100.")
                        body = dict(common, tool_choice=dict(type="function", function=dict(name="lookup")),
                                    max_tokens=256, stream=True)
                    else:
                        require(history[-1].get("tool_calls"), "tool fixture did not produce a native tool call")
                        incoming = dict(role="tool", tool_call_id=history[-1]["tool_calls"][0]["id"],
                            content="blue. Count decimal integers from 1 to 100, separated by comma and space. Begin exactly: 1, 2, 3,")
                        body = dict(common, tool_choice="none", max_tokens=64, ignore_eos=True, stream=False)
                    history.append(incoming)
                    body["messages"] = copy.deepcopy(history)
                    expected = rendered_tokens(decode, body) if not anchor else anchor + independent_suffix(incoming, anchor)
                    expected_new = len(expected) - len(anchor)
                    explicit = sid if not args.inferred and not args.witness else None
                    sid, message = chat(client, body, explicit)
                    sessions.append(sid)
                    record = native_records[-1]
                    native = record["native"]
                    prompt = record["request"].get("prompt_tokens") or rendered_tokens(decode, body)
                    local = native["timings"]["prompt_n"]
                    actual = app.store.get(sid)
                    meter = (actual.get("journal") or {}).get("last_meter")
                    prefilled = meter["prefilled"] if meter else local
                    item = dict(phase=phase, session=sid, prompt=prompt, expected_prompt=expected,
                        expected_new=expected_new, prefilled=prefilled, meter=meter, native=native)
                    if prompt != expected or prefilled != expected_new:
                        violations.append(dict(phase=phase, prefilled=prefilled, expected_new=expected_new,
                                               prompt_matches=prompt == expected))
                    if not args.witness:
                        require(prompt == expected, phase + ": token input differs from independent template suffix")
                        require(prefilled == expected_new, phase + ": old tokens were prefilled again")
                        require(actual["tokens"] == prompt + native["tokens"], phase + ": exact sampled journal not reconciled")
                        require(actual["journal"]["evaluated_boundary"] == actual["journal"]["sampled_boundary"],
                                phase + ": sampled/evaluated boundaries differ")
                        for index, reasoning in known_reasoning.items():
                            require(actual["journal"]["messages"][index].get("reasoning_content") == reasoning,
                                    phase + ": omitted client reasoning was lost from the message journal")
                        if message.get("reasoning_content"):
                            known_reasoning[len(history)] = message["reasoning_content"]
                        # Independent solo context repeats the same per-turn partition and generation reconciliation.
                        solo = decode.request("POST", "/v1/chat/completions", dict(body, stream=False,
                            prompt_tokens=expected, id_slot=2, cache_prompt=True))
                        require(solo["__verbose"]["tokens"] == native["tokens"], phase + ": independent solo token mismatch")
                        require(solo["timings"]["cache_n"] == native["timings"]["cache_n"], phase + ": solo partition differs")
                        if kind == "answer":
                            require(len(native["tokens"]) == 64, phase + ": expected greedy64 tokens")
                        solo_ids = expected + solo["__verbose"]["tokens"]
                        decode.request("POST", "/completion", dict(prompt=solo_ids, id_slot=2,
                                       n_predict=0, cache_prompt=True))
                        require(decode.request("GET", "/slots/2/info")["tokens"] == solo_ids,
                                "solo generation reconciliation did not hold exact IDs")
                        item["solo_ids"] = solo["__verbose"]["tokens"]
                    records.append(item)
                    anchor = prompt + native["tokens"]
                    if args.omit_reasoning:
                        message.pop("reasoning_content", None)
                    history.append(message)
                    print(f"OBSERVED {phase} prefilled={prefilled} expected_new={expected_new} generated={len(native['tokens'])}", flush=True)
                    (root / "conversation.json").write_text(json.dumps(dict(records=records,
                        violations=violations, sessions=sessions, total_prefilled=sum(r["prefilled"] for r in records),
                        total_expected=sum(r["expected_new"] for r in records)), indent=2))
            if args.witness:
                require(violations and any(v["phase"].startswith("turn-2") for v in violations),
                        "M0.1 control did not detect later-user re-prefill")
                later_users = [r for r in records if r["phase"] in ("turn-2-tool", "turn-3-tool")]
                require(all(r["prefilled"] == len(r["prompt"]) and r["prefilled"] > r["expected_new"] for r in later_users),
                        "M0.1 control failed for a reason other than full prompt re-prefill")
                raise AssertionError("M0.1 witness: full re-prefill replaced append-only continuation; see conversation.json")
            require(len(set(sessions)) == 1, "append-only message history forked")

            # Another client's live journal and slot must remain isolated during checkpoint rewrites and rejection.
            phase = "neighbor"
            other, _ = chat(foreign, dict(common, messages=[dict(role="user", content="Count decimal integers from 100.")],
                                         tool_choice="none", max_tokens=64, ignore_eos=True))
            neighbor = app.store.get(other)
            neighbor_info = decode.request("GET", f"/slots/{neighbor['slot']}/info")
            try:
                chat(foreign, dict(body), sid)
            except HTTPError as error:
                require(error.status == 404, "foreign explicit session did not return404")
            else:
                raise AssertionError("foreign session was accessible")

            for mutation in ("drop-tool", "edit-user"):
                phase = mutation
                old = app.store.get(sid)
                changed = copy.deepcopy(old["journal"]["messages"])
                if mutation == "drop-tool":
                    divergence = [i for i, m in enumerate(changed) if m["role"] == "tool"][1]
                    changed.pop(divergence)
                else:
                    divergence = next(i for i, m in enumerate(changed) if m["role"] == "user")
                    changed[divergence]["content"] = "Edited earlier user: count decimal integers from 10."
                cp = max((cp for cp in old["journal"]["checkpoints"] if cp["message_index"] <= divergence),
                         key=lambda cp: cp["message_index"])
                changed.append(dict(role="user", content="Continue counting decimal integers from 1 to 100."))
                rewrite = dict(common, messages=changed, tool_choice="none", max_tokens=64, ignore_eos=True)
                reused, _ = chat(client, rewrite, sid)
                updated = app.store.get(sid)
                require(reused == sid and updated["journal"]["last_meter"]["rollback_n"] == cp["n"],
                        mutation + ": did not use the latest user boundary before divergence")
                require(updated["tokens"][:cp["n"]] == old["tokens"][:cp["n"]] and cp["n"] > 0,
                        mutation + ": checkpoint prefix changed or fell back to0")
                info = decode.request("GET", f"/slots/{updated['slot']}/info")
                require(info["tokens"] == updated["tokens"] and info["prefix_hash"] == prefix_hash(updated["tokens"]),
                        mutation + ": final token/hash mismatch")
                require(decode.request("GET", f"/slots/{neighbor['slot']}/info") == neighbor_info and
                        app.store.get(other)["journal"] == neighbor["journal"], mutation + ": leaked into neighbor")
                (root / (mutation + ".json")).write_text(json.dumps(dict(checkpoint=cp,
                    meter=updated["journal"]["last_meter"], info=info), indent=2))
                print(f"PASS {mutation}: rollback_n={cp['n']} prefix/hash and neighbor intact", flush=True)
            # An inferred compaction creates a child session using a copied native checkpoint; its parent stays intact.
            phase = "inferred-compaction"
            parent = app.store.get(sid)
            parent_info = decode.request("GET", f"/slots/{parent['slot']}/info")
            changed = copy.deepcopy(parent["journal"]["messages"])
            divergence = next(i for i, m in enumerate(changed) if m["role"] == "tool")
            changed.pop(divergence)
            expected_cp = max((cp for cp in parent["journal"]["checkpoints"] if cp["message_index"] <= divergence),
                              key=lambda cp: cp["message_index"])
            changed.append(dict(role="user", content="Continue counting decimal integers from 1."))
            child, _ = chat(client, dict(common, messages=changed, tool_choice="none", max_tokens=64, ignore_eos=True))
            forked = app.store.get(child)
            require(child != sid and forked["journal"]["last_meter"]["rollback_n"] == expected_cp["n"],
                    "inferred compaction did not fork from its latest native checkpoint")
            require(app.store.get(sid)["journal"] == parent["journal"] and
                    decode.request("GET", f"/slots/{parent['slot']}/info") == parent_info,
                    "inferred compaction changed the parent's journal or native state")
            print(f"PASS inferred compaction: new session, rollback_n={expected_cp['n']}, parent intact", flush=True)
            print("PASS three users, real tool loops, exact suffix metering and independent same-partition greedy64", flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
            app.store.close()


if __name__ == "__main__":
    main()
