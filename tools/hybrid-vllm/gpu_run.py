"""Data preparation for gpu-run.sh; no inference runtime imports."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import urllib.request

from bench_common import model_identity, sha256
from bridge import convert, require


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def cached_model():
    if os.environ.get("HF_MODEL"):
        model = os.environ["HF_MODEL"]
        if Path(model).is_dir():
            return str(Path(model).resolve())
        from huggingface_hub import snapshot_download
        return snapshot_download(model, local_files_only=True)
    home = Path(os.environ["HF_HOME"])
    hubs = [Path(os.environ["HF_HUB_CACHE"])] if os.environ.get("HF_HUB_CACHE") else [home / "hub", home]
    configs = sorted({p for hub in hubs for p in hub.glob("models--*/snapshots/*/config.json")})
    candidates = []
    for config in configs:
        raw = json.loads(config.read_text())
        text = raw.get("text_config", raw)
        if (raw.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
                and text.get("num_hidden_layers") == 64 and text.get("hidden_size") == 5120
                and "fp8" in json.dumps(raw.get("quantization_config", {})).lower()):
            candidates.append(str(config.parent.resolve()))
    require(len(candidates) == 1, "set HF_MODEL to one cached FP8 snapshot; candidates=" + repr(candidates))
    return candidates[0]


def prepare(root):
    require(not (root / "bridge.json").exists(), "capture attempt already prepared; use a fresh RUN_DIR")
    model = Path(os.environ["BF16_GGUF"])
    target = dict(model_identity(model), model_sha256=sha256(model))
    n = int(os.environ["EXPORT_N"])
    require(0 < n < int(os.environ["CONTEXT_SIZE"]) - 64, "EXPORT_N outside context budget")
    token_file = Path(os.environ["TOKEN_FILE"])
    if not token_file.exists() and token_file == root / "tokens.json":
        write_json(token_file, [(37 * i + 5) % target["model_layout"]["vocab_size"] for i in range(n)])
    tokens = json.loads(token_file.read_text())
    if isinstance(tokens, dict):
        tokens = tokens["tokens"]
    require(isinstance(tokens, list) and len(tokens) >= n, "insufficient token journal")
    tokens = tokens[:n]
    vocab = target["model_layout"]["vocab_size"]
    require(all(type(t) is int and 0 <= t < vocab for t in tokens), "invalid token journal")
    require(0 <= int(os.environ["PROBE_TOKEN"]) < vocab, "invalid probe token")
    hf_model = cached_model()
    versions = {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers", "flashinfer-python")}
    require(versions["vllm"] == "0.32.0", "unsupported vLLM version")
    write_json(root / "environment.json", dict(versions=versions, hf_model=hf_model,
               source_ref=os.environ["EXPECTED_REF"], hf_home=os.environ["HF_HOME"]))
    (root / "hf-model.path").write_text(hf_model + "\n")
    write_json(root / "bridge.json", dict(target=target, tokens=tokens, output=str(root / "captures"), rotations=[256, 64]))
    write_json(root / "request.json", dict(model="bridge", prompt=tokens, max_tokens=1, temperature=0, ignore_eos=True))


def convert_target(root, quant, model):
    require(quant in ("bf16", "q4_k_m"), "unsupported target leg")
    target = dict(model_identity(Path(model)), model_sha256=sha256(Path(model)))
    config = json.loads((root / "bridge.json").read_text())
    paths = sorted((root / "captures").glob("capture-*/snapshot.json"))
    require(bool(paths), "no captures")
    out = root / ("deltas-" + quant)
    out.mkdir(exist_ok=False)
    write_json(root / ("target-" + quant + ".json"), target)
    deltas, n = [], 0
    for snapshot in paths:
        raw = json.loads(snapshot.read_text())
        require(raw["n0"] == n, "discontinuous capture chain")
        dest = out / ("%08d.delta" % raw["n1"])
        header = convert(snapshot, target, dest)
        require(header["model_sha256"] == target["model_sha256"], "wrong target identity")
        deltas.append(str(dest))
        n = raw["n1"]
    require(n == len(config["tokens"]) == int(os.environ["EXPORT_N"]), "incomplete capture chain")
    fixture = dict(tokens=config["tokens"] + [int(os.environ["PROBE_TOKEN"])], k=64,
                   max_ctx=int(os.environ["CONTEXT_SIZE"]), model_sha256=target["model_sha256"],
                   deltas=deltas, backend="cuda", uncalibrated=True)
    write_json(root / ("check-" + quant + ".json"), fixture)
    print(quant, target["model_sha256"], deltas[0])


def import_target(root, quant):
    fixture = json.loads((root / ("check-" + quant + ".json")).read_text())
    url = "http://127.0.0.1:" + os.environ["LLAMA_PORT"]
    for path in fixture["deltas"]:
        request = urllib.request.Request(url + "/slots/0?action=import_delta", data=Path(path).read_bytes(), method="POST",
                    headers={"Content-Type": "application/octet-stream", "X-Hybrid-Session": "vllm-prototype"})
        with urllib.request.urlopen(request, timeout=int(os.environ["REQUEST_TIMEOUT"])) as response:
            print(response.read().decode())
    with urllib.request.urlopen(url + "/slots/0/info", timeout=30) as response:
        info = json.load(response)
    require(info["tokens"] == fixture["tokens"][:-1], "imported token journal mismatch")
    write_json(root / ("import-info-" + quant + ".json"), info)


if __name__ == "__main__":
    root = Path(os.environ["RUN_DIR"])
    if sys.argv[1] == "prepare":
        prepare(root)
    elif sys.argv[1] == "convert":
        convert_target(root, sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "import":
        import_target(root, sys.argv[2])
    else:
        raise SystemExit("unknown data command")
