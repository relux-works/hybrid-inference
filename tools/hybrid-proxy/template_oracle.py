"""Test-only Jinja2 oracle; full template sources stay outside the repository."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import unittest


def source_fixture(fixture, root=None):
    root = Path(root or os.environ.get("HYBRID_TEMPLATE_FIXTURES", "/home/agent/lanes/hybrid/.temp/template-fixtures/"))
    path = root / (fixture["label"] + ".json")
    if not path.is_file():
        raise unittest.SkipTest("external template source absent: " + str(path) + "; set HYBRID_TEMPLATE_FIXTURES")
    external = json.loads(path.read_text())
    result = copy.deepcopy(fixture)
    for key in ("chat_template", "chat_template_tool_use"):
        source = external["props"].get(key)
        digest = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
        if digest != fixture["props"][key + "_sha256"]:
            raise ValueError("external template SHA-256 mismatch: " + key)
        result["props"][key] = source
    return result


class TemplateOracle:
    """Reviewer r6o oracle, excluding server-side assistant-prefill assembly."""
    def __init__(self, fixture):
        try:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise unittest.SkipTest("Jinja2 unavailable; external template oracle requires Jinja2") from None
        self.fixture = source_fixture(fixture)
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        environment.filters["tojson"] = lambda value, indent=None, ensure_ascii=False, separators=None: json.dumps(value, indent=indent, ensure_ascii=ensure_ascii, separators=separators)
        def raise_exception(message):
            raise jinja2.TemplateError(message)
        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = lambda fmt: datetime.datetime(2026, 11, 1).strftime(fmt)
        self.template = environment.from_string(self.fixture["props"]["chat_template"])

    def render(self, body):
        messages = copy.deepcopy(body["messages"])
        for message in messages:
            for call in message.get("tool_calls", []) or []:
                if isinstance(call["function"].get("arguments"), str):
                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        props = self.fixture["props"]
        kwargs = dict(enable_thinking=True)
        kwargs.update(body.get("chat_template_kwargs", {}))
        last_assistant = messages[-1]["role"] == "assistant"
        text = self.template.render(messages=messages, add_generation_prompt=False if last_assistant else body.get("add_generation_prompt", True),
                                    bos_token=props.get("bos_token"), eos_token=props.get("eos_token"), tools=body.get("tools"), **kwargs)
        bos = props.get("bos_token") or ""
        text = text[len(bos):] if bos and text.startswith(bos) else text
        if last_assistant:
            content = messages[-1].get("content") or ""
            text = text[:text.rindex(content) + len(content)]
        return text
