"""Copied-source controls for template guards, with a surviving identity control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CONTROLS = [
    ("identity", "unchanged source under the same tests", "", []),
    ("plain-bytes", "IDs alone cannot prove byte preservation", "TemplateHTTPTests.test_altered_piece_atomic",
     [('b"".join(raw) != text.encode() or special_ids != tokens', 'special_ids != tokens')]),
    ("foreign-bos", "piece position alone does not authorize a foreign special", "TemplateHTTPTests.test_foreign_special_atomic",
     [('special_ids != bos_ids + plain_ids', 'False'), ('special[0]["piece"] != bos', 'False')]),
    ("double-bos", "BOS identity alone cannot limit its multiplicity", "TemplateHTTPTests.test_double_bos_atomic",
     [('special_ids != bos_ids + plain_ids', 'False'), ('special[1:] != plain', 'False')]),
    ("bos-in-text", "exact IDs alone do not permit a visible and invisible BOS", "TemplateHTTPTests.test_bos_in_text_atomic",
     [('bos in text or ', '')]),
    ("bos-bytes", "BOS spelling cannot advance a rendered-byte offset", "TemplateTests.test_layout_and_user_boundaries",
     [('return [b""] + raw', 'return [bos.encode()] + raw')]),
    ("unknown-family", "a failed live-family check cannot select a fallback", "TemplateHTTPTests.test_unknown_template_atomic",
     [('        raise JournalConflict("unknown template family")\n\n    def token_bytes',
       '        return "<|turn>", "<turn|>\\n"\n\n    def token_bytes')]),
    ("split-eot", "an exact suffix alone does not prove one ending token", "TemplateHTTPTests.test_split_eot_atomic",
     [(' or len(end) != 1', '')]),
    ("exact-suffix", "a recognizable ending alone cannot authorize a changed suffix", "TemplateHTTPTests.test_changed_suffix_atomic",
     [('or not full_text.endswith(suffix) or full_tokens[-len(tail):] != tail', 'or False')]),
    ("gemma-tool-stop", "the call terminator cannot replace the decoder response opener", "TemplateTests.test_append",
     [('closing_text = "<|tool_response>"', 'closing_text = "<tool_call|>"'),
      ("suffix = after[len('<|\"|>}<tool_call|>'):]", "suffix = after[len('<|\"|>}'):]")]),
    ("generation-history", "exact pieces alone do not protect history from generation prompts", "TemplateHTTPTests.test_generation_rewrites_history_atomic",
     [('if not text.startswith(without_generation):\n            raise JournalConflict("generation prompt changed previous messages")', 'if not text.startswith(without_generation):\n            pass')]),
    ("probe-context", "a recognizable header alone does not protect its preceding context", "TemplateHTTPTests.test_probe_rewrites_context_atomic",
     [('raise JournalConflict("message probe changed its preceding template context")', 'pass')]),
]

CONTROLS += [
    ("continued-turn-frame", "opening-frame search without continued-model boundaries", "ContinuedTurnTests.test_consecutive_assistants",
     [('continued = opening == "<|turn>"', 'continued = False and opening == "<|turn>"')]),
    ("continued-tool-frame", "opening-frame search after an embedded tool response", "ContinuedTurnTests.test_call_tool_answer",
     [('continued = opening == "<|turn>"', 'continued = False and opening == "<|turn>"')]),
    ("family-source-props", "live renders without BOS/EOS/source authority", "TemplateGuardTests.test_family_source_properties",
     [('if bos and (props.get("bos_token")', 'if False and (props.get("bos_token")')]),
    ("family-bos", "source framing without the declared BOS", "TemplateGuardTests.test_family_source_properties",
     [('props.get("bos_token") != bos', 'False')]),
    ("family-eos", "source framing without the declared EOS", "TemplateGuardTests.test_family_source_properties",
     [('props.get("eos_token") != eos', 'False')]),
    ("family-source-opening", "live frames without an opening in the template source", "TemplateGuardTests.test_family_source_properties",
     [('opening not in props.get("chat_template", "")', 'False')]),
    ("family-source-closing", "live frames without a closing in the template source", "TemplateGuardTests.test_family_source_properties",
     [('closing.strip() not in props.get("chat_template", "")', 'False')]),
    ("family-gen-header", "preserved history without a valid generation header", "TemplateGuardTests.test_family_generation_header",
     [('if not generated[len(rendered):].startswith(opening + assistant):', 'if False:')]),
    ("family-closing-after-last", "valid transitions without an exact final closing", "TemplateGuardTests.test_family_closings",
     [('rendered.split(next_user)[1] != closing', 'False')]),
    ("family-closing-before-assistant", "initial user framing without a valid assistant transition", "TemplateGuardTests.test_family_closings",
     [('not between.startswith(closing + opening + assistant)', 'False')]),
    ("family-closing-after-assistant", "initial user framing without a valid next-user transition", "TemplateGuardTests.test_family_closings",
     [('not after_answer.startswith(closing + opening + user)', 'False')]),
    ("family-generation-prefix", "valid generation header without preserved history", "TemplateGuardTests.test_family_generation_prefix",
     [('if not generated.startswith(rendered):\n                    raise JournalConflict("generation prompt changed previous messages")', 'if False:\n                    pass')]),
    ("append-closing-prefix", "exact suffix without its closing prefix", "TemplateGuardTests.test_append_closing_prefix",
     [(' or tail[:len(closing)] != closing', '')]),
    ("gemma-tool-count", "matching tool suffix without an exact probe count", "TemplateGuardTests.test_gemma_tool_probe_count",
     [('if marked.count(marker) != 2:', 'if False:')]),
    ("gemma-tool-closing", "matching response suffix without a valid call terminator", "TemplateGuardTests.test_gemma_tool_probe_closing",
     [("if not after.startswith('<|\"|>}<tool_call|><|tool_response>'):", 'if False:')]),
    ("layout-overlap", "valid frames without strict message range ordering", "TemplateGuardTests.test_layout_overlap",
     [('if byte >= stop:', 'if False:')]),
    ("layout-equal-starts", "ordered frames that still allow an empty message range", "TemplateGuardTests.test_layout_overlap",
     [('if byte >= stop:', 'if byte > stop:')]),
    ("users-header-text", "matching prefix tokens without matching live header bytes", "TemplateGuardTests.test_users_header_text",
     [('if text.startswith(prefix + header) and', 'if text.startswith(prefix) and')]),
    ("assistant-probe-count", "unique reasoning probe without unique content", "TemplateGuardTests.test_duplicate_assistant_probe",
     [('rendered.count(marker) > 1', 'False')]),
]

EXPECTED_ASSERTIONS = {
    "bos-bytes": "byte_offset",
    "gemma-tool-stop": "self.assertEqual(tail[0], end[0])",
    "continued-turn-frame": "AssertionError: supported Gemma continuation rejected:",
    "continued-tool-frame": "AssertionError: supported Gemma continuation rejected:",
    "users-header-text": "self.assertEqual(self.renderer.user_boundaries(self.body, text, tokens, start=3), [])",
}

EQUIVALENT_BOUNDS = {
    "frame-nonassistant-count": "non-assistant reasoning is absent; the unique-marker and duplicate-marker guards imply this count",
    "bos-len": "one extra piece plus exact bos_ids + plain_ids already implies one BOS ID",
    "plain-equal-branch": "when IDs agree, returned bytes come from the already checked plain pieces",
}
CONTROLS += [
    ("frame-nonassistant-count", EQUIVALENT_BOUNDS["frame-nonassistant-count"], "",
     [('(role != "assistant" and rendered.count(marker) != 1)', 'False')]),
    ("bos-len", EQUIVALENT_BOUNDS["bos-len"], "",
     [('len(bos_ids) != 1 or ', '')]),
    ("plain-equal-branch", EQUIVALENT_BOUNDS["plain-equal-branch"], "",
     [('            if special != plain:\n', '            if False:\n')]),
]


def worker(test, output):
    import journal
    import test_templates
    suite = (unittest.defaultTestLoader.loadTestsFromName("test_templates." + test) if test else
             unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName("test_templates." + name) for name in
                                ("TemplateTests", "TemplateHTTPTests", "TemplateGuardTests", "ContinuedTurnTests")))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(output).write_text(json.dumps(dict(module=str(Path(journal.__file__).resolve()),
        bound_namespace=test_templates.Renderer is journal.Renderer and journal.Renderer.layout.__globals__ is journal.__dict__,
        tests=result.testsRun, skipped=[str(t) for t in result.skipped],
        failures=[dict(test=case.id(), traceback=trace) for case, trace in result.failures],
        errors=[dict(test=case.id(), traceback=trace) for case, trace in result.errors]), indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    identity = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in list(source.glob("*.py")) + list((source / "fixtures/templates").glob("*.json"))}
    (args.output / "source-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    rows = []
    for name, bound, test, replacements in CONTROLS:
        row = dict(mutant=name, narrows_gate_to=bound, named_test=test or "test_templates", status="not-applied",
                   killed=False, survivor_bound=bound)
        with tempfile.TemporaryDirectory(prefix=name + "-", dir=args.output) as directory:
            copied = Path(directory) / "tools" / "hybrid-proxy"
            copied.mkdir(parents=True)
            for path in source.glob("*.py"):
                shutil.copyfile(path, copied / path.name)
            shutil.copytree(source / "fixtures", copied / "fixtures")
            vocab = Path(directory) / "src"
            vocab.mkdir()
            shutil.copyfile(source.parents[1] / "src/llama-vocab.cpp", vocab / "llama-vocab.cpp")
            path = copied / "journal.py"
            data = path.read_text()
            valid_plant = True
            for old, new in replacements:
                if data.count(old) != 1 or old == new:
                    row["reason"] = "plant must replace exactly one source occurrence"
                    valid_plant = False
                    break
                data = data.replace(old, new)
            if not valid_plant:
                rows.append(row)
                break
            path.write_text(data)
            result_path = (args.output / (name + ".json")).resolve()
            command = [sys.executable, "-B", str(copied / "template_controls.py"), "--worker", test, str(result_path)]
            row.update(command=command, module=str((copied / "journal.py").resolve()))
            with (args.output / (name + ".log")).open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                row["pid"] = process.pid
                (args.output / (name + "-process.json")).write_text(json.dumps(row, indent=2) + "\n")
                try:
                    row["exit_code"] = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    row.update(exit_code=process.returncode, reason="control exceeded 60 s")
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                valid = (result["module"] == row["module"] and result["bound_namespace"] and result["tests"] > 0
                         and not result["errors"] and not result["skipped"])
                row.update(failures=result["failures"], errors=result["errors"])
                if name == "identity" or name in EQUIVALENT_BOUNDS:
                    row["status"] = "survived" if valid and row["exit_code"] == 0 and not result["failures"] else "invalid"
                else:
                    row["killed"] = valid and row["exit_code"] == 1 and bool(result["failures"]) and all(
                        test in failure["test"] and EXPECTED_ASSERTIONS.get(name, "AssertionError: JournalConflict not raised" if test.startswith("TemplateGuardTests.") else "AssertionError: HTTPError not raised") in failure["traceback"] for failure in result["failures"])
                    row["status"] = "killed" if row["killed"] else "survived" if valid and row["exit_code"] == 0 else "invalid"
                    if row["killed"]:
                        row["survivor_bound"] = None
            else:
                row["status"] = "invalid"
            rows.append(row)
        if name == "identity" and row["status"] != "survived":
            break
    (args.output / "controls.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["mutant"] + ": " + row["status"])
    return 0 if len(rows) == len(CONTROLS) and rows[0]["status"] == "survived" and all(
        row["status"] == "survived" if row["mutant"] in EQUIVALENT_BOUNDS else row["killed"] for row in rows[1:]) else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2], sys.argv[3]))
    sys.exit(main())
