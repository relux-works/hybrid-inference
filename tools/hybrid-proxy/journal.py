"""Message identity and checked incremental rendering for verified template families."""
import copy
import bisect
import hashlib
import json
import uuid


class JournalConflict(ValueError):
    pass


def message_identity(message, omit_reasoning=False):
    result = copy.deepcopy(message)
    if omit_reasoning:
        result.pop("reasoning_content", None)
    calls = result.get("tool_calls")
    for call in calls if isinstance(calls, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            try:
                function["arguments"] = json.loads(function["arguments"])
            except ValueError:
                pass
    # JSON encoding preserves scalar types and absent keys through nested lists/objects; Python equality does not.
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def matching_messages(recorded, incoming):
    for index, (old, new) in enumerate(zip(recorded, incoming)):
        omit_reasoning = new.get("role") == "assistant" and "reasoning_content" not in new
        if message_identity(old, omit_reasoning) != message_identity(new, omit_reasoning):
            return index
    return min(len(recorded), len(incoming))


def settings_identity(body, template):
    fields = ("tools", "chat_template_kwargs", "reasoning_format",
              "reasoning_effort", "continue_final_message", "add_generation_prompt")
    data = dict(template=template, settings={key: body[key] for key in fields if key in body})
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def segment_range(segments, n0, n1, replayed=False):
    result = []
    for segment in segments:
        start, end = max(n0, segment["start"]), min(n1, segment["end"])
        if start >= end:
            continue
        value = copy.deepcopy(segment)
        value.update(start=start, end=end)
        if "sampled_ids" in value:
            value["sampled_ids"] = value["sampled_ids"][start - segment["start"]:end - segment["start"]]
        if replayed:
            value["replayed"] = True
        result.append(value)
    return result


class Renderer:
    def __init__(self, node):
        self.node = node

    def render(self, body):
        return self.node.request("POST", "/apply-template", body)["prompt"]

    def encode(self, text, initial=False):
        return self.node.request("POST", "/tokenize", dict(content=text, add_special=initial))["tokens"]

    def template_identity(self):
        props = self.node.request("GET", "/props")
        data = {key: props.get(key) for key in ("chat_template", "chat_template_tool_use", "bos_token", "eos_token")}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def family(self, body):
        props = self.node.request("GET", "/props")
        marker, answer, next_user = ["hybrid" + uuid.uuid4().hex for _ in range(3)]
        probe = dict(body, messages=[dict(role="user", content=marker),
            dict(role="assistant", content=answer), dict(role="user", content=next_user)],
            add_generation_prompt=False)
        rendered = self.render(probe)
        if any(rendered.count(value) != 1 for value in (marker, answer, next_user)):
            raise JournalConflict("unknown template family")
        before, after = rendered.split(marker)
        families = (("<|im_start|>", "user\n", "<|im_end|>\n", None, None),
                    ("<|turn>", "user\n", "<turn|>\n", "<bos>", "<turn|>"),
                    ("<|start|>", "user<|message|>", "<|eot|>", "<|begin_of_text|>", "<|end_of_text|>"))
        for opening, user, closing, bos, eos in families:
            if before.endswith(opening + user) and after.startswith(closing):
                if bos and (props.get("bos_token") != bos or props.get("eos_token") != eos or
                            opening not in props.get("chat_template", "") or
                            closing.strip() not in props.get("chat_template", "")):
                    continue
                assistant = "model" if opening == "<|turn>" else "assistant"
                between = rendered.split(marker)[1].split(answer)[0]
                after_answer = rendered.split(answer)[1]
                if (not between.startswith(closing + opening + assistant) or
                        not after_answer.startswith(closing + opening + user) or
                        rendered.split(next_user)[1] != closing):
                    continue
                generated = self.render(dict(probe, add_generation_prompt=True))
                if not generated.startswith(rendered):
                    raise JournalConflict("generation prompt changed previous messages")
                if not generated[len(rendered):].startswith(opening + assistant):
                    continue
                return opening, closing
        raise JournalConflict("unknown template family")

    def token_bytes(self, text, tokens):
        def pieces(initial):
            return self.node.request("POST", "/tokenize",
                dict(content=text, add_special=initial, with_pieces=True))["tokens"]

        plain, special = pieces(False), pieces(True)
        raw = [p["piece"].encode() if isinstance(p["piece"], str) else bytes(p["piece"]) for p in plain]
        plain_ids = [p["id"] for p in plain]
        special_ids = [p["id"] for p in special]
        if b"".join(raw) != text.encode() or special_ids != tokens:
            raise JournalConflict("template token pieces do not preserve rendered bytes")
        if special_ids == plain_ids:
            if special != plain:
                raise JournalConflict("template token pieces do not preserve rendered bytes")
            return raw
        bos = self.node.request("GET", "/props").get("bos_token")
        bos_ids = self.encode(bos) if isinstance(bos, str) and bos else []
        if (len(bos_ids) != 1 or bos in text or special_ids != bos_ids + plain_ids or
                special[0]["piece"] != bos or special[1:] != plain):
            raise JournalConflict("template has unsupported leading tokenizer specials")
        return [b""] + raw

    def message_frame(self, body, text, index, family):
        opening, _ = family
        marker, reasoning = "hybrid" + uuid.uuid4().hex, "hybrid" + uuid.uuid4().hex
        probe = copy.deepcopy(body)
        message = probe["messages"][index]
        message["content"] = marker
        role = message["role"]
        if role == "assistant":
            message["reasoning_content"] = reasoning
        rendered = self.render(probe)
        markers = [rendered.index(value) for value in (marker, reasoning) if rendered.count(value) == 1]
        if not markers or (role != "assistant" and rendered.count(marker) != 1):
            raise JournalConflict("template did not preserve a message boundary probe")
        prefix = rendered[:min(markers)]
        if role == "tool" and opening == "<|turn>":
            begin = prefix.rfind("<|tool_response>")
        elif role == "tool" and opening == "<|im_start|>" and "\n<tool_response>\n" in prefix:
            begin = prefix.rfind("\n<tool_response>\n")
            user = opening + "user"
            if prefix[:begin].endswith(user):
                begin -= len(user)
        else:
            begin = prefix.rfind(opening)
        if begin < 0:
            raise JournalConflict("template message frame is not supported")
        preceding, header = prefix[:begin], prefix[begin:]
        if not text.startswith(preceding):
            raise JournalConflict("message probe changed its preceding template context")
        return preceding, header

    def user_header(self, body, text, index):
        _, header = self.message_frame(body, text, index, self.family(body))
        return self.encode(header)

    def user_boundaries(self, body, text, tokens, start=0):
        family = self.family(body)
        result = []
        for index, message in enumerate(body["messages"]):
            if index < start or message.get("role") != "user":
                continue
            prefix, header = self.message_frame(body, text, index, family)
            ids = self.encode(prefix, initial=True)
            # The live render and real token prefix locate the header, never a marker in user text.
            if text.startswith(prefix + header) and tokens[:len(ids)] == ids:
                result.append(dict(message_index=index, n=len(ids)))
        return result

    def layout(self, body, text, tokens, start=0, token_base=0, anchor_base=0):
        family = self.family(body)
        raw = self.token_bytes(text, tokens)
        offsets = [0]
        for piece in raw:
            offsets.append(offsets[-1] + len(piece))

        def position(byte):
            index = min(len(tokens), bisect.bisect_right(offsets, byte) - 1)
            return dict(n=anchor_base + index - token_base, byte_offset=byte - offsets[index])

        starts = []
        for index in range(start, len(body["messages"])):
            prefix, _ = self.message_frame(body, text, index, family)
            starts.append((index, body["messages"][index]["role"], len(prefix.encode())))
        end = len(text.encode())
        without_generation = self.render(dict(body, add_generation_prompt=False))
        if not text.startswith(without_generation):
            raise JournalConflict("generation prompt changed previous messages")
        if body["messages"][-1]["role"] != "assistant":
            end = len(without_generation.encode())
        result = []
        for item, (index, role, byte) in enumerate(starts):
            stop = starts[item + 1][2] if item + 1 < len(starts) else end
            if byte > stop:
                raise JournalConflict("template message boundaries overlap")
            result.append(dict(message_index=index, role=role, start=position(byte), end=position(stop)))
        return result, position(end)

    def append(self, body, full_text, full_tokens, past_count, anchor):
        if not past_count or body["messages"][past_count - 1].get("role") != "assistant":
            raise JournalConflict("no assistant boundary for incremental rendering")
        marker = "hybrid" + uuid.uuid4().hex
        probe = copy.deepcopy(body)
        assistant = probe["messages"][past_count - 1]
        opening, closing_text = self.family(body)
        calls = assistant.get("tool_calls")
        tool_boundary = opening == "<|turn>" and bool(calls)
        if tool_boundary:
            # Gemma emits responses inside the model turn, after the sampled response opener.
            function = calls[-1]["function"]
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise JournalConflict("tool boundary arguments are not a mapping")
            function["arguments"] = {marker: marker}
            closing_text = "<|tool_response>"
        else:
            assistant["content"] = marker
            calls = assistant.pop("tool_calls", None)
            # Glimmer resolves response names from calls that this content probe removes.
            if calls:
                names = {call.get("id"): call["function"]["name"] for call in calls}
                for message in probe["messages"][past_count:]:
                    if message.get("role") != "tool":
                        break
                    if message.get("tool_call_id") in names:
                        message.setdefault("name", names[message["tool_call_id"]])
        marked = self.render(probe)
        if tool_boundary:
            if marked.count(marker) != 2:
                raise JournalConflict("template did not preserve the tool boundary probe")
            after = marked.rsplit(marker, 1)[1]
            if not after.startswith('<|"|>}<tool_call|><|tool_response>'):
                raise JournalConflict("template tool call closing is not supported")
            suffix = after[len('<|"|>}<tool_call|>'):]
        else:
            if marked.count(marker) != 1:
                raise JournalConflict("template did not preserve the boundary probe")
            suffix = marked.split(marker, 1)[1]
        tail = self.encode(suffix)
        closing = self.encode(closing_text)
        end = self.encode(closing_text.rstrip("\n"))
        if (not tail or not closing or len(end) != 1 or tail[:len(closing)] != closing
                or not full_text.endswith(suffix) or full_tokens[-len(tail):] != tail):
            raise JournalConflict("template suffix is not an exact end-of-turn message boundary")
        overlap = int(bool(anchor) and anchor[-1] == end[0])
        base = len(full_tokens) - len(tail)
        boundaries = []
        for boundary in self.user_boundaries(body, full_text, full_tokens, start=past_count):
            if boundary["message_index"] >= past_count and boundary["n"] >= base:
                boundaries.append(dict(message_index=boundary["message_index"],
                    n=len(anchor) - overlap + boundary["n"] - base))
        return anchor + tail[overlap:], boundaries, len(closing), tail


class Capture:
    def __init__(self):
        self.message = dict(role="assistant", content="")
        self.native = None
        self.timings = None

    def feed(self, chunk):
        if "__verbose" in chunk:
            self.native = chunk["__verbose"]
        if "timings" in chunk:
            self.timings = chunk["timings"]
        for choice in chunk.get("choices", []):
            if "message" in choice:
                self.message = copy.deepcopy(choice["message"])
                continue
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content"):
                if delta.get(key):
                    self.message[key] = self.message.get(key, "") + delta[key]
            for tool in delta.get("tool_calls", []):
                index = tool.get("index", 0)
                if type(index) is not int or not 0 <= index < 128:
                    raise JournalConflict("invalid tool index in decoder response")
                calls = self.message.setdefault("tool_calls", [])
                while len(calls) <= index:
                    calls.append(dict(id="", type="function", function=dict(name="", arguments="")))
                call = calls[index]
                if tool.get("id"):
                    call["id"] = tool["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += tool.get("function", {}).get(key, "")
