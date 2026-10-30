"""Message identity and checked incremental rendering for ChatML-style templates."""
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

    def user_boundaries(self, body, text, tokens, start=0):
        result = []
        for index, message in enumerate(body["messages"]):
            if index < start or message.get("role") != "user":
                continue
            marker = "hybrid" + uuid.uuid4().hex
            probe = copy.deepcopy(body)
            probe["messages"][index]["content"] = marker
            rendered = self.render(probe)
            header = "<|im_start|>user\n"
            if rendered.count(marker) != 1:
                continue
            prefix = rendered[:rendered.index(marker)]
            if not prefix.endswith(header):
                continue
            prefix = prefix[:-len(header)]
            ids = self.encode(prefix, initial=True)
            # This checks the actual render and token boundary; a marker in user content is never searched.
            if text.startswith(prefix) and tokens[:len(ids)] == ids:
                result.append(dict(message_index=index, n=len(ids)))
        return result

    def layout(self, body, text, tokens, start=0, token_base=0, anchor_base=0):
        pieces = self.node.request("POST", "/tokenize", dict(content=text, add_special=True, with_pieces=True))["tokens"]
        raw = [piece["piece"].encode() if isinstance(piece["piece"], str) else bytes(piece["piece"]) for piece in pieces]
        if [piece["id"] for piece in pieces] != tokens or b"".join(raw) != text.encode():
            raise JournalConflict("template token pieces do not preserve rendered bytes")
        offsets = [0]
        for piece in raw:
            offsets.append(offsets[-1] + len(piece))

        def position(byte):
            index = min(len(tokens), bisect.bisect_right(offsets, byte) - 1)
            return dict(n=anchor_base + index - token_base, byte_offset=byte - offsets[index])

        starts = []
        for index in range(start, len(body["messages"])):
            marker, reasoning = "hybrid" + uuid.uuid4().hex, "hybrid" + uuid.uuid4().hex
            probe = copy.deepcopy(body)
            message = probe["messages"][index]
            message["content"] = marker
            role = message["role"]
            if role == "assistant":
                message["reasoning_content"] = reasoning
            rendered = self.render(probe)
            if rendered.count(marker) != 1:
                raise JournalConflict("template did not preserve a message boundary probe")
            prefix = rendered[:rendered.index(marker)]
            if role == "system" and index == 0 and text.startswith("<|im_start|>system\n"):
                starts.append((index, role, 0))
                continue
            frame = "<|im_start|>" + role + "\n"
            if role == "assistant" and reasoning in prefix:
                prefix = prefix[:prefix.index(reasoning)]
                frame += "<think>\n"
            elif role == "tool":
                frame = "\n<tool_response>\n"
            if not prefix.endswith(frame):
                raise JournalConflict("template message frame is not supported")
            prefix = prefix[:-len(frame)]
            if role == "tool" and prefix.endswith("<|im_start|>user"):
                prefix = prefix[:-len("<|im_start|>user")]
            if not text.startswith(prefix):
                raise JournalConflict("message probe changed its preceding template context")
            starts.append((index, role, len(prefix.encode())))
        end = len(text.encode())
        if body["messages"][-1]["role"] != "assistant":
            without_generation = self.render(dict(body, add_generation_prompt=False))
            if not text.startswith(without_generation):
                raise JournalConflict("generation prompt changed previous messages")
            end = len(without_generation.encode())
        result = []
        for item, (index, role, byte) in enumerate(starts):
            stop = starts[item + 1][2] if item + 1 < len(starts) else end
            result.append(dict(message_index=index, role=role, start=position(byte), end=position(stop)))
        return result, position(end)

    def append(self, body, full_text, full_tokens, past_count, anchor):
        if not past_count or body["messages"][past_count - 1].get("role") != "assistant":
            raise JournalConflict("no assistant boundary for incremental rendering")
        marker = "hybrid" + uuid.uuid4().hex
        probe = copy.deepcopy(body)
        assistant = probe["messages"][past_count - 1]
        assistant["content"] = marker
        assistant.pop("tool_calls", None)
        marked = self.render(probe)
        if marked.count(marker) != 1:
            raise JournalConflict("template did not preserve the boundary probe")
        suffix = marked.split(marker, 1)[1]
        tail = self.encode(suffix)
        closing = self.encode("<|im_end|>\n")
        end = self.encode("<|im_end|>")
        if (not tail or not closing or len(end) != 1 or tail[:len(closing)] != closing
                or not full_text.endswith(suffix) or full_tokens[-len(tail):] != tail):
            raise JournalConflict("template suffix is not an exact ChatML message boundary")
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
