"""Hook control and owned-process cleanup without vLLM, a model or CUDA."""

import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import exporter
from node_backend import VllmBackend
import node_deploy
from node_deploy import stop_child, ticks
from test_bridge import fixture
import hybrid_bench


class RuntimeTests(unittest.TestCase):
    # Node identity follows source weight bytes, while the BF16 GGUF remains a separate geometric anchor.
    def test_deploy_source_identity(self):
        _, target = fixture()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ):
            root = Path(tmp)
            model = root / "hf"
            model.mkdir()
            (model / "config.json").write_text(json.dumps(dict(quantization_config=dict(quant_method="fp8"))))
            (model / "model.safetensors").write_bytes(b"first source weights")
            anchor = root / "anchor.gguf"
            anchor.write_bytes(b"separate layout anchor")
            identities = []
            for number, weights in enumerate((b"first source weights", b"different source weights")):
                (model / "model.safetensors").write_bytes(weights)
                state = root / str(number)
                state.mkdir()
                args = NS(state=state, hf_cache=model, hf_model=model, layout_model=anchor, context=512)
                with mock.patch.object(node_deploy, "model_identity", return_value=dict(model_layout=target["model_layout"], layout_fingerprint=target["layout_fingerprint"])):
                    self.assertEqual(node_deploy.prepare(args), model)
                info = json.loads((state / "node-info.json").read_text())
                identities.append(info["model_sha256"])
                self.assertEqual(info["quantization_type"], "FP8")
                self.assertEqual(info["layout_fingerprint"], target["layout_fingerprint"])
                self.assertNotEqual(info["model_sha256"], info["layout_anchor_sha256"])
            self.assertNotEqual(*identities)
            (model / "config.json").write_text(json.dumps(dict(quantization_config=dict(quant_algo="NVFP4", kv_cache_quant_algo="FP8"))))
            args.state = root / "invalid-source"
            args.state.mkdir()
            with mock.patch.object(node_deploy, "model_identity", return_value=dict(model_layout=target["model_layout"], layout_fingerprint=target["layout_fingerprint"])):
                with self.assertRaisesRegex(ValueError, "FP8 weights"):
                    node_deploy.prepare(args)
            self.assertEqual(list(args.state.iterdir()), [])

    # Checker exports end at aligned positions but the fixture retains every prompt token for local tail evaluation.
    def test_checker_aligned_tail(self):
        config = dict(prefill_url="http://fixture", model_sha256="0" * 64, clients={"a": dict(prefill_token="fixture")},
                      export_block_size=16, extend_chunk_tokens=35)
        client = dict(context=512, build_dir="build", model_path="fixture.gguf", backend="cpu")
        args = NS(check_tokens=2, noise_chunks=[128], request_timeout=60)
        calls = []
        def http(url, path, token, body, **kwargs):
            calls.append(body)
            return dict(session_id="fixture") if path == "/v1/sessions" else b"fixture"
        class Checked(Exception):
            pass
        def checker(command, scratch):
            value = json.loads(Path(command[2]).read_text())
            self.assertEqual(value["tokens"], list(range(50)))
            self.assertEqual(value["local_tail_start"], 48)
            self.assertEqual([c["n0"] for c in calls[1:]], [0, 32])
            self.assertEqual([len(c["tokens"]) for c in calls[1:]], [32, 16])
            raise Checked
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hybrid_bench, "http", side_effect=http), \
                mock.patch.object(hybrid_bench, "run_checker", side_effect=checker), mock.patch.object(hybrid_bench.urllib.request, "urlopen"):
            with self.assertRaises(Checked):
                hybrid_bench.equivalence(config, client, list(range(50)), args, Path(tmp))

    # A bounded local alignment tail is planned work; transport/contract fallback and a missed full block still refuse.
    def test_alignment_tail_meter(self):
        trace = dict(prefilled=4, new_tokens=4, replayed_prefix=0, rollback_n=None, depth_before=10,
                     depth_after=16, prompt_depth=14, generated_tokens=2, mode="offload", remote_applied=0,
                     path="local", fallback_reason=None, export_block_size=16)
        hybrid_bench.verify_meter(trace, 10, "offload")
        for changes in (dict(fallback_reason="http_400"), dict(prompt_depth=40, depth_after=42)):
            with self.assertRaisesRegex(hybrid_bench.Refusal, "offload unavailable"):
                hybrid_bench.verify_meter(dict(trace, **changes), 10, "offload")

    # Node mode skips intermediate scheduler chunks and captures only the exact current endpoint on every request.
    def test_node_hook_endpoints_and_probe(self):
        _, target = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control.json"
            req = NS(prompt_token_ids=list(range(48)), output_token_ids=[], num_computed_tokens=16)
            runner = NS(requests={"r": req}, kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=16))]))
            config = dict(node_mode=True, target=target, tokens=req.prompt_token_ids, n0=19, output=tmp)
            control.write_text(json.dumps(config))
            with mock.patch.dict(os.environ, HYBRID_VLLM_CONFIG=str(control)), mock.patch.object(exporter, "check_config"), \
                    mock.patch.object(exporter, "export_snapshot") as capture:
                exporter.after_prefill(runner, NS(num_scheduled_tokens={"r": 16}))
                capture.assert_not_called()
                self.assertFalse((root / "finished.json").exists())
                req.num_computed_tokens = 32
                exporter.after_prefill(runner, NS(num_scheduled_tokens={"r": 16}))
                self.assertEqual(capture.call_args.args[-1], 19)
                self.assertEqual(json.loads((root / "finished.json").read_text()), dict(n=48, block_size=16))
                # The next request gets a different range; no previous request's scalar cursor can suppress it.
                config.update(n0=35)
                control.write_text(json.dumps(config))
                exporter.after_prefill(runner, NS(num_scheduled_tokens={"r": 16}))
                self.assertEqual(capture.call_args.args[-1], 35)
                config.update(probe=True)
                control.write_text(json.dumps(config))
                exporter.after_prefill(runner, NS(num_scheduled_tokens={"r": 16}))
                self.assertEqual(json.loads((root / "ready.json").read_text()), dict(block_size=16))
                config.update(probe=False)
                control.write_text(json.dumps(config))
                req.output_token_ids = [3]
                with self.assertRaisesRegex(ValueError, "unexpected node request"):
                    exporter.after_prefill(runner, NS(num_scheduled_tokens={"r": 16}))
                control.unlink()
                exporter.after_prefill(runner, NS(num_scheduled_tokens={"r": 16}))

    # The actual HTTP backend atomically changes request control, verifies completion, and removes it on errors.
    def test_http_backend_control(self):
        _, target = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control.json"
            seen = []
            class FakeHandler(http.server.BaseHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass

                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    config = json.loads(control.read_text())
                    seen.append((body, config))
                    output = Path(config["output"])
                    if config.get("probe"):
                        (output / "ready.json").write_text(json.dumps(dict(block_size=16)))
                    elif body["prompt"] != [9]:
                        (output / "finished.json").write_text(json.dumps(dict(n=len(body["prompt"]), block_size=16)))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{}')
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                backend = VllmBackend("http://127.0.0.1:" + str(server.server_port), control, target, root)
                self.assertEqual(backend.probe(), 16)
                self.assertFalse(control.exists())
                path = backend.prefill(list(range(32)), 19, root)
                self.assertEqual(path, root / "capture-00000032/snapshot.json")
                self.assertEqual(seen[-1][1]["n0"], 19)
                self.assertEqual(seen[-1][0]["max_tokens"], 1)
                self.assertFalse(control.exists())
                (root / "finished.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    backend.prefill([9], 0, root)
                self.assertFalse(control.exists())
                self.assertFalse(backend.healthy())
                with self.assertRaisesRegex(OSError, "restart the node"):
                    backend.prefill(list(range(16)), 0, root)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    # Cleanup stops its recorded session group even when the launcher exits before a TERM-resistant worker.
    def test_stop_owned_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready"
            child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(120)"
            code = "import subprocess,sys,time; from pathlib import Path; c=subprocess.Popen([sys.executable,'-c',sys.argv[2]],stdout=subprocess.PIPE,text=True); c.stdout.readline(); Path(sys.argv[1]).write_text(str(c.pid)); print('ready',flush=True); time.sleep(120)"
            child = subprocess.Popen([sys.executable, "-B", "-c", code, str(path), child_code], stdout=subprocess.PIPE, text=True, start_new_session=True)
            assert child.stdout is not None
            worker = None
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                worker = int(path.read_text())
                self.assertIsNotNone(ticks(child.pid))
                stop_child(child)
                self.assertIsNotNone(child.returncode)
                # A killed orphan may briefly remain a zombie until its external parent reaps it.
                status = Path("/proc/%d/stat" % worker)
                if status.exists():
                    for _ in range(10000):
                        if not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                            break
                    self.assertTrue(not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                child.stdout.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
