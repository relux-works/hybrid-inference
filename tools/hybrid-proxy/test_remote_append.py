"""Opt-in real prefill-node append metering and checkpoint/replay transitions; no builds."""
import argparse
import copy
import hashlib
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from native_fixture import native_server, require
from policy import Policy
from proxy import Handler, HTTPError, Node, Proxy
from store import prefix_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--prefill-binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=True)
    credentials = root / "credentials.json"
    credentials.write_text(json.dumps({"a": "fixture-remote-a", "b": "fixture-remote-b"}))
    os.chmod(credentials, 0o600)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(Path(args.prefill_binary).resolve()), args.model, str(credentials), str(port), "4096"]
    (root / "prefill-identity.json").write_text(json.dumps(dict(command=command,
        sha256=hashlib.sha256(Path(args.prefill_binary).read_bytes()).hexdigest()), indent=2))
    app = server = thread = None
    with (root / "prefill.log").open("wb") as output:
        prefill = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        (root / "prefill.pid").write_text(str(prefill.pid) + "\n")
        try:
            remote = Node(f"http://127.0.0.1:{port}", "fixture-remote-a", timeout=120)
            deadline = time.monotonic() + 120
            while True:
                require(prefill.poll() is None, "prefill node exited during startup")
                try:
                    remote.request("GET", "/v1/health")
                    break
                except (OSError, HTTPError):
                    require(time.monotonic() < deadline, "prefill readiness timeout")
                    time.sleep(0.1)
            with native_server(args.binary, args.model, root / "decode") as (decode, slots):
                app = Proxy(dict(model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                    prefill_url=remote.url, prefill_timeout=120, decode_url=decode.url, decode_timeout=120,
                    slots=2, slot_path=str(slots), clients={"fixture-a": dict(id="a", prefill_token="fixture-remote-a")},
                    policy=dict(local_rates=((0, 1e12),), fixed=1e6)), str(root / "proxy.sqlite"))
                server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                server.app = app
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                client = Node(f"http://127.0.0.1:{server.server_port}", "fixture-a", timeout=120)

                def chat(body, sid=None):
                    with client.open("POST", "/v1/chat/completions", body,
                        {"X-Hybrid-Session": sid} if sid else {}) as response:
                        selected = response.headers["X-Hybrid-Session"]
                        json.loads(response.read())
                    with app.reserve(selected):
                        pass
                    return app.store.get(selected)

                def replayed(session):
                    app.replay()
                    app.replay()
                    current = app.store.get(session["id"])
                    info = remote.request("GET", "/v1/sessions/" + current["remote_id"])
                    require(info["n"] == len(current["tokens"]) and info["prefix_hash"] == prefix_hash(current["tokens"]),
                            "real shadow replay lost or duplicated IDs")
                    return current, info

                body = dict(messages=[dict(role="system", content="Retain this system prefix and use decimal digits."),
                            dict(role="user", content="Count from 1 through 100, separated by comma and space.")],
                    max_tokens=64, temperature=0, seed=0, ignore_eos=True,
                    chat_template_kwargs=dict(enable_thinking=False))
                # Local prefill creates a user checkpoint; real shadow replay catches up before a genuine rewrite.
                initial = chat(body)
                require(initial["journal"]["last_meter"]["remote_applied"] == 0 and initial["journal"]["checkpoints"],
                        "local positive control did not retain a user checkpoint")
                initial, _ = replayed(initial)
                original_remote = initial["remote_id"]
                changed = copy.deepcopy(body)
                changed["messages"][1]["content"] = "Count from 100 through 200, separated by comma and space."
                rewritten = chat(changed, initial["id"])
                require(rewritten["journal"]["last_meter"]["rollback_n"] > 0 and rewritten["remote_id"] != original_remote,
                        "checkpoint rewrite did not create a fresh remote session")
                rewritten, _ = replayed(rewritten)
                try:
                    remote.request("GET", "/v1/sessions/" + original_remote)
                except HTTPError as error:
                    require(error.status == 404, "retired remote session was not deleted safely")
                else:
                    raise AssertionError("retired remote state leaked after committed rewrite")
                print("PASS real checkpoint rewrite: nonzero local rollback, fresh remote replay, retired state removed", flush=True)

                # Offloaded append applies only the new suffix: remote N-1, local held token 1, exact token/hash journal.
                app.policy = Policy(always=True)
                history = copy.deepcopy(rewritten["journal"]["messages"])
                history.append(dict(role="user", content="Continue counting decimal integers from 200 to 300."))
                appended = chat(dict(body, messages=history), initial["id"])
                meter = appended["journal"]["last_meter"]
                require(meter["local_evaluated"] == 1 and meter["remote_applied"] == meter["new_tokens"] - 1 and
                        meter["prefilled"] == meter["new_tokens"], "remote append re-prefilled old IDs")
                info = decode.request("GET", f"/slots/{appended['slot']}/info")
                require(info["tokens"] == appended["tokens"] and info["prefix_hash"] == prefix_hash(appended["tokens"]),
                        "remote append journal differs from decoder IDs")
                require(not appended["journal"]["checkpoints"], "imported state falsely advertised native user checkpoints")
                appended, remote_before = replayed(appended)
                print(f"PASS real remote append: local=1 remote={meter['remote_applied']} new={meter['new_tokens']}", flush=True)

                # An imported slot has no user checkpoint: explicit rewrite is atomic409 and inferred rewrite forks remotely.
                invalid = copy.deepcopy(body)
                invalid["messages"][1]["content"] = "Edited user: count from 500 through 600."
                decode.request("POST", f"/slots/{appended['slot']}?action=save", dict(filename="before409.bin"))
                before_bytes = (slots / "before409.bin").read_bytes()
                try:
                    chat(invalid, appended["id"])
                except HTTPError as error:
                    require(error.status == 409, "explicit no-checkpoint transition did not return409")
                else:
                    raise AssertionError("explicit imported-session rewrite silently reset")
                decode.request("POST", f"/slots/{appended['slot']}?action=save", dict(filename="after409.bin"))
                require((slots / "after409.bin").read_bytes() == before_bytes, "explicit409 changed native bytes")
                child = chat(invalid)
                require(child["id"] != appended["id"] and child["remote_id"] != appended["remote_id"],
                        "no-checkpoint fork reused the parent's remote state")
                require(remote.request("GET", "/v1/sessions/" + appended["remote_id"]) == remote_before,
                        "fork changed the parent remote session")
                (root / "results.json").write_text(json.dumps(dict(initial=initial, rewritten=rewritten,
                    appended=appended, child=child), indent=2))
                print("PASS real fallback: explicit409/native bytes unchanged; inferred child uses fresh remote", flush=True)
        finally:
            if server:
                server.shutdown()
                server.server_close()
            if thread:
                thread.join(timeout=10)
            if app:
                app.store.close()
            if prefill.poll() is None:
                prefill.terminate()
                try:
                    prefill.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    prefill.kill()
                    prefill.wait(timeout=15)
            (root / "prefill-cleanup.json").write_text(json.dumps(dict(pid=prefill.pid, exit_code=prefill.returncode)))


if __name__ == "__main__":
    main()
