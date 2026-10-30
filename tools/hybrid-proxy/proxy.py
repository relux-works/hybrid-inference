"""Hybrid client proxy. Credentials are read from a JSON file, never CLI arguments."""
import argparse
import http.server
import http.client
import json
import logging
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from policy import Policy
from store import Store, prefix_hash

LOG = logging.getLogger("hybrid")
MAX_JSON = 8 * 1024 * 1024
MAX_DELTA = 512 * 1024 * 1024


class HTTPError(Exception):
    def __init__(self, status, body=b""):
        self.status, self.body = status, body
        super().__init__(f"upstream HTTP {status}")


class RollbackError(HTTPError):
    def __init__(self):
        super().__init__(503)
        self.args = ("decoder rollback failed; backup retained; slot requires local prefill",)


class Node:
    def __init__(self, url, token="", timeout=1.0):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self.last_transfer_rate = None

    def open(self, method, path, data=None, headers=None):
        headers = dict(headers or {})
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            with error:
                raise HTTPError(error.code, error.read(MAX_JSON)) from None

    def request(self, method, path, data=None, headers=None, binary=False):
        try:
            with self.open(method, path, data, headers) as response:
                limit = MAX_DELTA if binary else MAX_JSON
                started = time.monotonic()
                body = response.read(limit + 1)
                elapsed = max(time.monotonic() - started, 1e-6)
                self.last_transfer_rate = len(body) / elapsed if body else None
                if len(body) > limit:
                    raise ValueError("upstream body too large")
                return body if binary else (json.loads(body) if body else None)
        except http.client.HTTPException as error:
            raise OSError("upstream HTTP connection interrupted") from error


class Proxy:
    def __init__(self, config, database):
        self.model = config["model_sha256"]
        self.clients = config["clients"]  # incoming bearer -> {id, prefill_token}
        if (not self.clients or len({c["id"] for c in self.clients.values()}) != len(self.clients)
                or len({c["prefill_token"] for c in self.clients.values()}) != len(self.clients)):
            raise ValueError("configure distinct client IDs and prefill credentials")
        self.slot_path = Path(config["slot_path"]).resolve()
        self.slot_path.mkdir(parents=True, exist_ok=True)
        self.prefill = config["prefill_url"]
        self.timeout = config.get("prefill_timeout", 0.5)
        self.decode = Node(config["decode_url"], config.get("decode_token", ""), config.get("decode_timeout", 120))
        self.store = Store(database, self.model)
        self.policy = Policy(**config.get("policy", {}))
        self.slots = config.get("slots", 1)
        if self.slots < 1:
            raise ValueError("slots must be positive")
        self.lock = threading.RLock()
        self.link_rate = self.policy.link_rate
        self.stop = threading.Event()

    def client(self, authorization):
        value = self.clients.get(authorization.removeprefix("Bearer ")) if authorization.startswith("Bearer ") else None
        if value is None:
            raise HTTPError(401)
        return value

    def remote(self, client):
        return Node(self.prefill, client["prefill_token"], self.timeout)

    def sync(self, session, client, recreate=True):
        """202 is an acknowledgement of queuing, never proof that tokens were applied."""
        node = self.remote(client)
        if not session["remote_id"]:
            session["remote_id"] = node.request("POST", "/v1/sessions", {"model_sha256": self.model})["session_id"]
            session["remote_n"] = 0
            self.store.save(session)
        path = "/v1/sessions/" + urllib.parse.quote(session["remote_id"], safe="")
        try:
            info = node.request("GET", path)
        except HTTPError as error:
            if error.status != 404 or not recreate:
                raise
            # A RAM-only prefill node can lose sessions on restart. Recreate and replay all durable tokens.
            session["remote_id"] = None
            session["remote_n"] = 0
            self.store.save(session)
            return self.sync(session, client, recreate=False)
        n = info["n"]
        if (info["model_sha256"] != self.model or n < 0 or n > len(session["tokens"])
                or info["prefix_hash"] != prefix_hash(session["tokens"][:n])):
            LOG.error("divergence session=%s local_hash=%s remote_hash=%s", session["id"],
                      prefix_hash(session["tokens"]), info.get("prefix_hash"))
            raise ValueError("server prefix diverged; explicit snapshot recovery required")
        session["remote_n"] = n
        self.store.save(session)
        if n < len(session["tokens"]):
            node.request("POST", path + "/extend", {"n0": n, "prefix_hash_n0": prefix_hash(session["tokens"][:n]),
                         "tokens": session["tokens"][n:], "return_delta": False})
        return path

    def resync(self, session, client, prompt):
        node = self.remote(client)
        path = "/v1/sessions/" + urllib.parse.quote(session["remote_id"], safe="")
        info = node.request("GET", path)
        LOG.warning("resync session=%s local_hash=%s remote_hash=%s", session["id"],
                    prefix_hash(session["tokens"]), info.get("prefix_hash"))
        if info["model_sha256"] != self.model:
            raise HTTPError(409)
        if (0 <= info["n"] <= len(session["tokens"])
                and info["prefix_hash"] == prefix_hash(session["tokens"][:info["n"]])):
            self.sync(session, client)
            return
        n = info["n"]
        if not 0 <= n <= len(prompt) or info["prefix_hash"] != prefix_hash(prompt[:n]):
            raise HTTPError(409)
        snapshot = node.request("GET", path + "/snapshot", binary=True)
        filename = session["id"] + "-resync.bin"
        backup = session["id"] + "-resync-backup.bin"
        temporary = self.slot_path / (filename + ".tmp")
        slot = f"/slots/{session['slot']}"
        saved = False
        keep_backup = False
        try:
            with open(temporary, "wb") as output:
                output.write(snapshot)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.slot_path / filename)
            self.decode.request("POST", slot + "?action=save", {"filename": backup})
            saved = True
            self.decode.request("POST", slot + "?action=restore", {"filename": filename})
            actual = self.decode.request("GET", slot + "/info")
            if (actual["n"] != n or actual["prefix_hash"] != info["prefix_hash"]
                    or actual["tokens"] != prompt[:n]):
                raise HTTPError(409)
            updated = dict(session, tokens=actual["tokens"], remote_n=n)
            self.store.save(updated)
            session.update(updated)
            saved = False
        except Exception:
            if saved:
                try:
                    self.decode.request("POST", slot + "?action=restore", {"filename": backup})
                except Exception as error:
                    keep_backup = True
                    self.store.verified.discard(session["id"])
                    raise RollbackError() from error
            raise
        finally:
            cleanup = [temporary, self.slot_path / filename]
            if not keep_backup:
                cleanup.append(self.slot_path / backup)
            for path in cleanup:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    LOG.warning("resync cleanup failed session=%s", session["id"])

    def recover(self, session, client, prompt):
        try:
            self.resync(session, client, prompt)
        except RollbackError:
            raise
        except OSError:
            pass
        except HTTPError as error:
            if error.status not in (404, 408, 429, 500, 502, 503, 504):
                raise

    def replay(self):
        for session in self.store.pending():
            client = next((c for c in self.clients.values() if c["id"] == session["client"]), None)
            if client:
                try:
                    self.sync(session, client)
                except (OSError, HTTPError, ValueError):
                    pass

    def replay_loop(self):
        while not self.stop.wait(1):
            with self.lock:
                self.replay()

    def acquire_slot(self, session):
        # A retained rollback backup is a durable marker: never adopt this slot's unverified state.
        rollback_pending = (self.slot_path / (session["id"] + "-resync-backup.bin")).exists()
        if session["slot"] is not None:
            slot = session["slot"]
            if rollback_pending:
                self.decode.request("POST", f"/slots/{slot}?action=erase", {})
                self.store.verified.add(session["id"])
                return slot
            if session["id"] not in self.store.verified:
                info = self.decode.request("GET", f"/slots/{slot}/info")
                if (info["n"] >= len(session["tokens"])
                        and info.get("tokens", [])[:len(session["tokens"])] == session["tokens"]
                        and prefix_hash(info.get("tokens", [])) == info["prefix_hash"]):
                    session["tokens"] = info["tokens"]
                    self.store.save(session)
                if info["n"] != len(session["tokens"]) or info["prefix_hash"] != prefix_hash(session["tokens"]):
                    self.decode.request("POST", f"/slots/{slot}?action=erase", {})
                    # The durable tokens remain the source for a full local re-prefill.
                self.store.verified.add(session["id"])
            return slot
        occupants = [self.store.occupant(i) for i in range(self.slots)]
        free = next((i for i, item in enumerate(occupants) if item is None), None)
        if free is None:
            old = min(occupants, key=lambda row: row["used"])
            free = old["slot"]
            self.decode.request("POST", f"/slots/{free}?action=save", {"filename": old["id"] + ".bin"})
            old["slot"] = None
            self.store.save(old)
        self.decode.request("POST", f"/slots/{free}?action=erase", {})
        # Restore a parked sequence when there is a saved file. First-use sessions have none.
        if session["tokens"] and not rollback_pending:
            try:
                self.decode.request("POST", f"/slots/{free}?action=restore", {"filename": session["id"] + ".bin"})
            except HTTPError:
                pass  # Tokens permit a correct local re-prefill if a saved file was lost.
        session["slot"] = free
        self.store.save(session)
        self.store.verified.add(session["id"])
        return free

    def prepare(self, client, body, explicit):
        if body.get("n", 1) != 1:
            raise ValueError("only one completion per request is supported")
        prompt = self.decode.request("POST", "/apply-template", body)["prompt"]
        tokens = self.decode.request("POST", "/tokenize", {"content": prompt, "add_special": True})["tokens"]
        session = self.store.select(client["id"], tokens, explicit)
        slot = self.acquire_slot(session)
        info = self.decode.request("GET", f"/slots/{slot}/info")
        local_n = info["n"]
        if local_n > len(tokens) or info["prefix_hash"] != prefix_hash(tokens[:local_n]):
            self.decode.request("POST", f"/slots/{slot}?action=erase", {})
            local_n = 0
        # Imported memory has no logits. Decode the final prompt token locally.
        target = tokens[:-1]
        decision = self.policy.choose(max(0, len(target) - local_n), local_n,
                                      bool(session["offload"]), self.link_rate)
        try:
            path = self.sync(session, client)
            if decision["offload"] and len(target) > local_n:
                # Flush a queued shadow with a critical extend at the local boundary.
                if session["remote_n"] > local_n:
                    raise ValueError("server is ahead of local slot; local re-prefill required")
                if session["remote_n"] < local_n:
                    self.sync(session, client)
                remote = self.remote(client)
                delta = remote.request("POST", path + "/extend", {
                    "n0": local_n, "prefix_hash_n0": prefix_hash(tokens[:local_n]),
                    "tokens": target[local_n:], "return_delta": True}, binary=True)
                if remote.last_transfer_rate:
                    self.link_rate = 0.8 * self.link_rate + 0.2 * remote.last_transfer_rate
                session["tokens"] = target
                self.store.save(session)
                self.decode.request("POST", f"/slots/{slot}?action=import_delta", delta,
                                    {"Content-Type": "application/octet-stream",
                                     "X-Hybrid-Session": session["remote_id"]})
                actual = self.decode.request("GET", f"/slots/{slot}/info")
                if actual["n"] != len(target) or actual["prefix_hash"] != prefix_hash(target):
                    self.decode.request("POST", f"/slots/{slot}?action=erase", {})
                    raise ValueError("import prefix mismatch")
                session["offload"] = True
                session["tokens"] = target
                session["remote_n"] = len(target)
                self.store.save(session)
        except HTTPError as error:
            if error.status == 409:
                LOG.warning("409 resync session=%s", session["id"])
                # GET validates the acknowledged prefix; replay uses its exact boundary.
                self.recover(session, client, tokens)
            elif error.status not in (404, 408, 429, 500, 502, 503, 504):
                raise
            session["offload"] = False
        except ValueError:
            if session["remote_id"]:
                self.recover(session, client, tokens)
            session["offload"] = False
        except OSError:
            session["offload"] = False
        session["tokens"] = tokens
        self.store.save(session)
        request = dict(body, id_slot=slot, cache_prompt=True)
        return session, tokens, request

    def finish(self, session, prompt):
        info = self.decode.request("GET", f"/slots/{session['slot']}/info")
        evaluated = info["tokens"]
        if (len(evaluated) != info["n"] or evaluated[:len(prompt)] != prompt
                or prefix_hash(evaluated) != info["prefix_hash"]):
            raise ValueError("decode token journal does not match slot")
        session["tokens"] = evaluated
        self.store.save(session)
        (self.slot_path / (session["id"] + "-resync-backup.bin")).unlink(missing_ok=True)
        # Replay is asynchronous; the durable suffix beyond remote_n is the journal.


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "HybridProxy/0.1"

    def log_message(self, *_):
        pass  # Do not log credentials, request bodies, or token content.

    def json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        app = self.server.app
        started = False
        try:
            client = app.client(self.headers.get("Authorization", ""))
            if self.path != "/v1/chat/completions":
                raise HTTPError(404)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_JSON:
                raise ValueError("invalid request size")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("request must be a JSON object")
            explicit = self.headers.get("X-Hybrid-Session")
            if explicit and (len(explicit) > 128 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in explicit)):
                raise ValueError("invalid session id")
            with app.lock:
                session, prompt, request = app.prepare(client, body, explicit)
                with app.decode.open("POST", "/v1/chat/completions", request) as upstream:
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("X-Hybrid-Session", session["id"])
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        started = True
                        for line in upstream:
                            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                                chunk = json.loads(line[6:])
                                chunk.pop("__verbose", None)
                                line = b"data: " + json.dumps(chunk).encode() + b"\n"
                            self.wfile.write(line)
                            self.wfile.flush()
                    else:
                        data = json.loads(upstream.read(MAX_JSON))
                        data.pop("__verbose", None)
                        app.finish(session, prompt)
                        self.send_response(200)
                        output = json.dumps(data).encode()
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(output)))
                        self.send_header("X-Hybrid-Session", session["id"])
                        self.end_headers()
                        self.wfile.write(output)
                        return
                app.finish(session, prompt)
        except (HTTPError, KeyError, ValueError, OSError) as error:
            if not started:
                status = error.status if isinstance(error, HTTPError) else 404 if isinstance(error, KeyError) else 400
                self.json_response(status, {"error": {"message": str(error), "type": "hybrid_error"}})
            else:
                LOG.error("stream interrupted: %s", type(error).__name__)
                self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--database", default="hybrid.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.config) as source:
        app = Proxy(json.load(source), args.database)
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app
    thread = threading.Thread(target=app.replay_loop, daemon=True)
    thread.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        thread.join()
        server.server_close()
        app.store.close()


if __name__ == "__main__":
    main()
