"""Opt-in server prompt-token and final-ID controls; small model, existing binary only."""
import argparse
import copy
import json
from pathlib import Path

from native_fixture import native_server, rendered_tokens, require, streamed
from proxy import HTTPError


def parsed_choices(choices):
    result = copy.deepcopy(choices)
    for choice in result:
        for call in choice.get("message", {}).get("tool_calls", []):
            require(bool(call.get("id")), "native parser did not assign a tool-call ID")
            call["id"] = "generated-tool-id"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    root = Path(args.artifacts)
    records = []
    with native_server(args.binary, args.model, root) as (node, slots):
        base = dict(messages=[dict(role="system", content="Follow the user request."),
                              dict(role="user", content="Count in decimal digits: 1, 2, 3,")],
                    id_slot=0, temperature=0, seed=0, max_tokens=64, ignore_eos=True,
                    return_tokens=True, verbose=True, cache_prompt=False,
                    chat_template_kwargs=dict(enable_thinking=False))
        # Legacy control: the old binary and rebuilt binary produce identical native slot bytes.
        legacy = node.request("POST", "/v1/chat/completions", base)
        node.request("POST", "/slots/0?action=save", dict(filename="legacy.bin"))
        (root / "legacy-response.json").write_text(json.dumps(legacy, indent=2))
        (root / "legacy.bin").write_bytes((slots / "legacy.bin").read_bytes())
        if args.baseline:
            baseline = Path(args.baseline)
            require((root / "legacy.bin").read_bytes() == (baseline / "legacy.bin").read_bytes(),
                    "legacy slot bytes changed after the server patch")
            old = json.loads((baseline / "legacy-response.json").read_text())
            require(legacy["choices"] == old["choices"] and legacy["__verbose"]["tokens"] == old["__verbose"]["tokens"],
                    "legacy output changed after the server patch")
        print("PASS legacy request and native slot-byte control", flush=True)
        if args.legacy_only:
            return

        # Invalid input must return exactly 400 without changing either tokens/hash or native bytes.
        before = node.request("GET", "/slots/0/info")
        before_bytes = (slots / "legacy.bin").read_bytes()
        n_ctx = node.request("GET", "/props")["default_generation_settings"]["n_ctx"]
        n_vocab = node.request("GET", "/v1/models")["data"][0]["meta"]["n_vocab"]
        cases = {"string": "bad", "null": None, "empty": [], "bool": [True], "float": [1.0],
                 "negative": [-1], "minimum-signed": [-2**63], "out-of-range": [n_vocab],
                 "unsigned-overflow": [2**64 - 1],
                 "too-long": [1] * (n_ctx + 1)}
        for name, tokens in cases.items():
            try:
                node.request("POST", "/v1/chat/completions", dict(base, prompt_tokens=tokens))
            except HTTPError as error:
                require(error.status == 400, f"{name}: expected 400")
            else:
                raise AssertionError(f"{name}: invalid token input was accepted")
            require(node.request("GET", "/slots/0/info") == before, f"{name}: slot info changed")
            node.request("POST", "/slots/0?action=save", dict(filename="rejected.bin"))
            require((slots / "rejected.bin").read_bytes() == before_bytes, f"{name}: native slot bytes changed")
        print(f"PASS {len(cases)} malformed prompt_tokens controls, native bytes unchanged", flush=True)
        # A different valid array must override the rendered prompt, rather than merely validate an unused field.
        distinct = node.request("POST", "/tokenize", dict(content="Authoritative token journal."))["tokens"]
        node.request("POST", "/v1/chat/completions", dict(base, prompt_tokens=distinct, n_predict=0))
        require(node.request("GET", "/slots/0/info")["tokens"] == distinct, "valid prompt_tokens was ignored")
        # The upstream completion guard reserves a token even for n_predict=0; preserve that legacy limit.
        before_capacity = node.request("GET", "/slots/0/info")
        try:
            node.request("POST", "/v1/chat/completions", dict(base, prompt_tokens=[1] * n_ctx, n_predict=0))
        except HTTPError as error:
            require(error.status == 400, "upstream full-context guard returned an unexpected error")
        else:
            raise AssertionError("upstream full-context guard changed")
        require(node.request("GET", "/slots/0/info") == before_capacity, "full-context rejection changed the slot")
        # The highest valid vocabulary ID and the usable n_ctx-1 boundary have nearby positive controls.
        boundary = [n_vocab - 1] * (n_ctx - 1)
        node.request("POST", "/v1/chat/completions", dict(base, prompt_tokens=boundary, n_predict=0))
        require(node.request("GET", "/slots/0/info")["tokens"] == boundary, "valid context/vocabulary boundary rejected")

        tool = dict(type="function", function=dict(name="lookup", description="Look up a color.",
                    parameters=dict(type="object", properties=dict(color=dict(type="string", enum=["blue"])),
                                    required=["color"], additionalProperties=False)))
        fixtures = {
            "reasoning": dict(base, chat_template_kwargs=dict(enable_thinking=True)),
            "tool": dict(base, messages=[dict(role="user", content="Call lookup with color blue.")],
                         tools=[tool], tool_choice=dict(type="function", function=dict(name="lookup")),
                         max_tokens=256, ignore_eos=False),
            "stop": dict(base, messages=[dict(role="user", content="Output abcdefghijklmno.")],
                         grammar='root ::= "abcdefghijklmno"', stop=["ijk"], ignore_eos=False),
            "eog-buffered": dict(base, grammar='root ::= "\\U0001f642"'.replace('\\U0001f642', '\U0001f642'),
                                 ignore_eos=False),
        }
        # Equal rendered input IDs retain the native parser and grammar; stream IDs include hidden stop/EOG/buffered tokens.
        for name, body in fixtures.items():
            normal = node.request("POST", "/v1/chat/completions", body)
            token_input = node.request("POST", "/v1/chat/completions",
                                       dict(body, prompt_tokens=rendered_tokens(node, body)))
            (root / (name + ".json")).write_text(json.dumps(dict(normal=normal, token_input=token_input), indent=2))
            # Native-generated tool IDs are random transport identities; every other parsed field must match.
            require(parsed_choices(normal["choices"]) == parsed_choices(token_input["choices"]),
                    f"{name}: parsed chat output changed")
            require(normal["__verbose"]["tokens"] == token_input["__verbose"]["tokens"], f"{name}: sampled IDs changed")
            message, final, chunks = streamed(node, dict(body, prompt_tokens=rendered_tokens(node, body)))
            require(final and final["tokens"] == normal["__verbose"]["tokens"], f"{name}: final stream IDs lost tokens")
            expected = normal["choices"][0]["message"]
            require(message.get("content", "") == (expected.get("content") or "") and
                    message.get("reasoning_content", "") == (expected.get("reasoning_content") or ""),
                    f"{name}: streaming parsed content/reasoning differs")
            if name == "tool":
                require(expected.get("tool_calls") and message.get("tool_calls"), "tool control did not call a tool")
                require([c["function"] for c in message["tool_calls"]] ==
                        [c["function"] for c in expected["tool_calls"]], "streamed tool arguments differ")
            if name == "stop":
                require(final["stop_type"] == "word" and final["stopping_word"] == "ijk", "stop control did not stop on word")
                raw_text = node.request("POST", "/detokenize", dict(tokens=final["tokens"]))["content"]
                require("ijk" in raw_text and "ijk" not in message.get("content", ""), "stop IDs were not hidden from visible text")
            if name == "eog-buffered":
                require(final["stop_type"] == "eos", "EOG control did not reach an EOG token")
                require(len(final["tokens"]) == final["tokens_predicted"] and len(final["tokens"]) > 2,
                        "buffered/EOG control did not exercise multiple native token IDs")
            records.append(dict(name=name, normal=normal, token_input=token_input, streamed=chunks))
            print(f"PASS {name}: exact native IDs and parsed chat output, including final stream IDs", flush=True)
        (root / "results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
