import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid-bench"))
import gpu_run
from bridge import Refusal


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.tools = self.root / "tools/hybrid-vllm"
        self.tools.mkdir(parents=True)
        self.script = self.tools / "gpu-run.sh"
        shutil.copyfile(Path(__file__).with_name("gpu-run.sh"), self.script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.run_dir = self.root / "absent/parents/run"
        self.calls = self.root / "calls"
        self.env = dict(os.environ, PATH=str(self.bin) + ":" + os.environ["PATH"],
                        PYTHON=str(self.bin / "python"), VLLM_VENV=str(self.root / "venv"),
                        EXPECTED_REF="a" * 40, RUN_DIR=str(self.run_dir),
                        TMPDIR=str(self.root / "disk-tmp"), CALLS=str(self.calls))
        self.command("python", 'printf "gather\\n" >> "$CALLS"\nexit "${GATHER_EXIT:-0}"\n')
        self.command("git", '''printf 'git %s\\n' "$*" >> "$CALLS"
case "$1" in
    rev-parse) printf '%s\\n' "${ACTUAL_REF:-$EXPECTED_REF}" ;;
    diff) exit "${DIRTY_EXIT:-0}" ;;
esac
''')
        self.command("cmake", '''printf 'cmake %s\\n' "$*" >> "$CALLS"
if [[ $1 == -S ]]; then
    mkdir -p "$BUILD_DIR/bin"
    printf '#!/bin/sh\\nexit "${CHECKER_EXIT:-0}"\\n' > "$BUILD_DIR/bin/llama-server"
    cp "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/llama-hybrid-bench-check"
    chmod +x "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/llama-hybrid-bench-check"
fi
exit "${CMAKE_EXIT:-0}"
''')

    def command(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o700)

    def run_step(self, step, *args, **env):
        return subprocess.run(["bash", str(self.script), step, *map(str, args)],
                              env=dict(self.env, **env), capture_output=True, text=True)

    # CPU command doubles prove fresh parents, gate order, target flags and binary hash recording.
    def test_build_and_paths(self):
        selected = self.root / "argument takes precedence/nested"
        result = self.run_step("build", selected)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls[0], "gather")
        self.assertIn("-DGGML_CUDA=ON", "\n".join(calls))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120a-real", "\n".join(calls))
        self.assertIn("--target llama-server llama-hybrid-bench-check", calls[-3])
        self.assertEqual((selected / "build.ref").read_text().strip(), "a" * 40)
        self.assertEqual(len((selected / "binaries.sha256").read_text().splitlines()), 2)
        self.assertFalse(self.run_dir.exists())
        self.assertTrue((self.root / "disk-tmp").is_dir())

    # A failed gather returns its real exit and blocks even the ref check and build.
    def test_gather_failure_stops_all(self):
        result = self.run_step("all", GATHER_EXIT="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(self.calls.read_text(), "gather\n")
        self.assertIn("FAIL exit=7", result.stderr)
        self.assertIn("gather_test.py", result.stderr)
        self.assertIn("EXIT 7", next(self.run_dir.glob("all-*.log")).read_text())

    # Ref mismatch, tracked modifications, invalid jobs and an old build directory refuse before cmake.
    def test_build_refusals(self):
        for env in ({"ACTUAL_REF": "b" * 40}, {"DIRTY_EXIT": "1"}, {"BUILD_JOBS": "5"}, {"EXPECTED_REF": "short"}):
            with self.subTest(env=env):
                result = self.run_step("build", **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("cmake", self.calls.read_text())
        (self.run_dir / "build").mkdir()
        result = self.run_step("build")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("cmake", self.calls.read_text())

    # A real child failure is visible in the terminal and log; the calling shell continues.
    def test_hidden_command_exit(self):
        result = subprocess.run(["bash", "-c", 'bash "$1" build; rc=$?; printf "CALLER_ALIVE exit=%s\\n" "$rc"',
                                 "probe", str(self.script)], env=dict(self.env, CMAKE_EXIT="9"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("CALLER_ALIVE exit=9", result.stdout)
        self.assertIn("FAIL exit=9", result.stderr)
        self.assertIn("nice -n 10 cmake", result.stderr)
        self.assertFalse((self.run_dir / "build.ref").exists())

    # A changed receiver binary cannot pass the stored exact-build gate or start a checker.
    def test_binary_tamper(self):
        result = self.run_step("build")
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.run_dir / "build/bin/llama-server").write_text("changed")
        result = self.run_step("check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stderr)
        self.assertFalse((self.run_dir / "check-bf16.exit").exists())

    # Exit 5 is retained as uncalibrated evidence; other checker errors stop with their real exit.
    def test_checker_exit_policy(self):
        self.command("nvidia-smi", "printf 'pid, process_name, used_memory\\n'\n")
        result = self.run_step("build")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_step("check", CHECKER_EXIT="5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.run_dir / "check-bf16.exit").read_text(), "5\n")
        self.assertEqual((self.run_dir / "check-q4_k_m.exit").read_text(), "5\n")
        self.assertIn("5 is uncalibrated, not passing", result.stderr)
        result = self.run_step("check", CHECKER_EXIT="2")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual((self.run_dir / "check-bf16.exit").read_text(), "2\n")

    # HTTP failure or TERM stops the owned dummy server, restores runner bytes and retains a failing exit.
    def test_capture_cleanup(self):
        result = self.run_step("build")
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = self.root / "runner.py"
        runner.write_text("original runner\n")
        self.command("python", '''case "$1" in
    */gpu_run.py) printf 'cached-model\\n' > "$RUN_DIR/hf-model.path" ;;
    -) cat > /dev/null; printf '%s\\n' "$FAKE_RUNNER" ;;
    */patch_vllm.py) printf 'patched\\n' > "$2" ;;
esac
''')
        venv = self.root / "venv/bin"
        venv.mkdir(parents=True)
        server = venv / "vllm"
        server.write_text("#!/bin/sh\nexec /bin/sleep 60\n")
        server.chmod(0o700)
        self.command("curl", '''if [[ ${!#} == */health ]]; then exit 0; fi
if [[ ${SIGNAL_CAPTURE:-0} == 1 ]]; then kill -TERM "$PPID"; exit 0; fi
exit 22
''')
        for signal, expected in (("0", 22), ("1", 143)):
            with self.subTest(signal=signal):
                result = self.run_step("capture", FAKE_RUNNER=str(runner), SIGNAL_CAPTURE=signal)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(runner.read_text(), "original runner\n")
                pid = int((self.run_dir / "vllm.pid").read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                (self.run_dir / "gpu_model_runner.py.original").unlink()

    # Sourcing refuses before changing shell options; help has no run-directory side effects.
    def test_source_and_help(self):
        result = subprocess.run(["bash", "-c", 'before=$-; source "$1"; rc=$?; [[ $before == "$-" ]] && printf "SAFE %s\\n" "$rc"',
                                 "probe", str(self.script)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("SAFE 2", result.stdout)
        result = self.run_step("--help")
        self.assertEqual(result.returncode, 0)
        self.assertFalse(self.run_dir.exists())


class DataTests(unittest.TestCase):
    # Offline auto-selection admits one matching snapshot and refuses absent or ambiguous matches.
    def test_cached_model_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(HF_HOME=directory)
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(Refusal, "set HF_MODEL"):
                    gpu_run.cached_model()
                config = dict(architectures=["Qwen3_5ForConditionalGeneration"],
                              text_config=dict(num_hidden_layers=64, hidden_size=5120),
                              quantization_config=dict(quant_method="fp8"))
                paths = [root / ("hub/models--test/snapshots/" + name) for name in ("first", "second")]
                paths[0].mkdir(parents=True)
                (paths[0] / "config.json").write_text(json.dumps(config))
                self.assertEqual(gpu_run.cached_model(), str(paths[0]))
                paths[1].mkdir()
                (paths[1] / "config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(Refusal, "set HF_MODEL"):
                    gpu_run.cached_model()
                with mock.patch.dict(os.environ, HF_MODEL=str(paths[0])):
                    self.assertEqual(gpu_run.cached_model(), str(paths[0]))

    # Default IDs are bounded; an absent external journal refuses before publishing a capture config.
    def test_prepare_token_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(TOKEN_FILE=str(root / "missing.json"), BF16_GGUF="fixture", EXPORT_N="16",
                       CONTEXT_SIZE="128", PROBE_TOKEN="1", EXPECTED_REF="a" * 40, HF_HOME="cache")
            target = dict(model_layout=dict(vocab_size=1000))
            with mock.patch.dict(os.environ, env), mock.patch.object(gpu_run, "model_identity", return_value=target), \
                    mock.patch.object(gpu_run, "sha256", return_value="b" * 64), \
                    mock.patch.object(gpu_run, "cached_model", return_value="cached-snapshot"), \
                    mock.patch.object(gpu_run.importlib.metadata, "version", side_effect=lambda p: "0.32.0" if p == "vllm" else "fixture"):
                with self.assertRaises(FileNotFoundError):
                    gpu_run.prepare(root)
                self.assertFalse((root / "bridge.json").exists())
                os.environ["TOKEN_FILE"] = str(root / "tokens.json")
                gpu_run.prepare(root)
                config = json.loads((root / "bridge.json").read_text())
                self.assertEqual(config["tokens"], [(37 * i + 5) % 1000 for i in range(16)])
                with self.assertRaisesRegex(Refusal, "already prepared"):
                    gpu_run.prepare(root)


if __name__ == "__main__":
    unittest.main()
