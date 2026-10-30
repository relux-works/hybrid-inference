"""Hybrid client proxy. Credentials are read from a JSON file, never CLI arguments."""
import argparse
from contextlib import contextmanager
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
from concurrency import SessionQueue
from store import Store, prefix_hash

LOG = logging.getLogger("hybrid")
MAX_JSON = 8 * 1024 * 1024
MAX_DELTA = 512 * 1024 * 1024
SLOT_WAIT_SECONDS = 30


class HTTPError(Exception):
    def __init__(self, status, body=b""):
        self.status, self.body = status, body
        super().__init__(f"upstream HTTP {status}")


class RollbackError(HTTPError):
    def __init__(self):
        super().__init__(503)
        self.args = ("decoder rollback failed; backup retained; slot requires local prefill",)


class SlotUnavailable(HTTPError):
    def __init__(self):
        super().__init__(503)
        self.args = ("no decode slot available; retry later",)


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
        self.lock = self.store.lock
        self.slot_available = threading.Condition(self.lock)
        self.pinned = set()
        self.quarantined = set()
        self.queues = {}
        self.link_rate = self.policy.link_rate
        self.stop = threading.Event()

    def client(self, authorization):
        value = self.clients.get(authorization.removeprefix("Bearer ")) if authorization.startswith("Bearer ") else None
        if value is None:
            raise HTTPError(401)
        return value

    def remote(self, client):
        return Node(self.prefill, client["prefill_token"], self.timeout)

    def reserve(self, sid, skip_busy=False):
        with self.lock:
            queue = self.queues.setdefault(sid, SessionQueue())
            return queue.reserve(skip_busy)

    def verified(self, sid, value=None):
        with self.lock:
            if value is not None:
                if value:
                    self.store.verified.add(sid)
                else:
                    self.store.verified.discard(sid)
            return sid in self.store.verified

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
                    self.verified(session["id"], False)
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
        for pending in self.store.pending():
            client = next((c for c in self.clients.values() if c["id"] == pending["client"]), None)
            if client:
                ticket = self.reserve(pending["id"], skip_busy=True)
                if ticket is None:
                    continue
                try:
                    with ticket:
                        self.sync(self.store.get(pending["id"]), client)
                except (OSError, HTTPError, ValueError):
                    pass

    def replay_loop(self):
        while not self.stop.wait(1):
            self.replay()

    def wait_idle(self, slot):
        deadline = time.monotonic() + SLOT_WAIT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SlotUnavailable()
            node = Node(self.decode.url, self.decode.token, min(self.decode.timeout, remaining))
            states = node.request("GET", "/slots")
            if any(s["id"] == slot and not s["is_processing"] for s in states):
                return
            time.sleep(min(0.05, remaining))

    def refresh_quarantined(self, deadline):
        with self.lock:
            if not self.quarantined:
                return
        node = Node(self.decode.url, self.decode.token, min(1.0, max(0.001, deadline - time.monotonic())))
        try:
            states = node.request("GET", "/slots")
        except (OSError, HTTPError, ValueError):
            return
        with self.slot_available:
            for state in states:
                if not state["is_processing"]:
                    self.quarantined.discard(state["id"])
            self.slot_available.notify_all()

    @contextmanager
    def slot_lease(self, session):
        deadline = time.monotonic() + SLOT_WAIT_SECONDS
        while True:
            self.refresh_quarantined(deadline)
            with self.slot_available:
                # Another session can park this inactive binding while its owner is replaying or queued.
                session["slot"] = self.store.get(session["id"])["slot"]
                slot = session["slot"]
                occupants = [self.store.occupant(i) for i in range(self.slots)]
                candidates = [i for i in range(self.slots) if i not in self.pinned and i not in self.quarantined]
                if slot is not None:
                    candidates = [slot] if slot in candidates else []
                if candidates:
                    slot = min(candidates, key=lambda i: occupants[i]["used"] if occupants[i] else -1)
                    old = occupants[slot]
                    self.pinned.add(slot)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SlotUnavailable()
                self.slot_available.wait(min(remaining, 0.1))
        checked_idle = False
        try:
            self.wait_idle(slot)
            checked_idle = True
            if session["slot"] is None:
                if old:
                    self.decode.request("POST", f"/slots/{slot}?action=save", {"filename": old["id"] + ".bin"})
                    self.store.set_slot(old["id"], None)
                self.decode.request("POST", f"/slots/{slot}?action=erase", {})
                rollback = (self.slot_path / (session["id"] + "-resync-backup.bin")).exists()
                if session["tokens"] and not rollback:
                    try:
                        self.decode.request("POST", f"/slots/{slot}?action=restore", {"filename": session["id"] + ".bin"})
                    except HTTPError:
                        pass  # Durable tokens allow local prefill when a parked file is missing.
                session["slot"] = slot
                self.store.set_slot(session["id"], slot)
            self.acquire_slot(session)
            yield slot
        except BaseException:
            self.verified(session["id"], False)
            raise
        finally:
            idle = False
            try:
                if checked_idle:
                    self.wait_idle(slot)
                    idle = True
            except (HTTPError, OSError, ValueError):
                LOG.warning("slot idle status unavailable slot=%s; quarantined", slot)
            with self.slot_available:
                self.pinned.remove(slot)
                if not idle:
                    self.quarantined.add(slot)
                    self.verified(session["id"], False)
                self.slot_available.notify_all()

    def acquire_slot(self, session):
        # Called only while the slot lease is pinned and the native slot is idle.
        rollback_pending = (self.slot_path / (session["id"] + "-resync-backup.bin")).exists()
        slot = session["slot"]
        if rollback_pending:
            self.decode.request("POST", f"/slots/{slot}?action=erase", {})
            self.verified(session["id"], True)
            return slot
        if not self.verified(session["id"]):
            info = self.decode.request("GET", f"/slots/{slot}/info")
            if (info["n"] >= len(session["tokens"])
                    and info.get("tokens", [])[:len(session["tokens"])] == session["tokens"]
                    and prefix_hash(info.get("tokens", [])) == info["prefix_hash"]):
                session["tokens"] = info["tokens"]
                self.store.save(session)
            if info["n"] != len(session["tokens"]) or info["prefix_hash"] != prefix_hash(session["tokens"]):
                self.decode.request("POST", f"/slots/{slot}?action=erase", {})
            self.verified(session["id"], True)
        return slot

    def tokenize(self, body):
        prompt = self.decode.request("POST", "/apply-template", body)["prompt"]
        return self.decode.request("POST", "/tokenize", {"content": prompt, "add_special": True})["tokens"]

    @contextmanager
    def prepare(self, client, body, explicit):
        if body.get("n", 1) != 1:
            raise ValueError("only one completion per request is supported")
        if explicit:
            with self.lock:
                try:
                    existing = self.store.get(explicit)
                except KeyError:
                    existing = None
                if existing and existing["client"] != client["id"]:
                    raise KeyError("session not found")
                ticket = self.reserve(explicit)
            # Reserve explicit IDs before any HTTP, so a slow tokenizer cannot invert their arrival order.
            with ticket:
                tokens = self.tokenize(body)
                session = self.store.select(client["id"], tokens, explicit)
                with self.slot_lease(session):
                    yield self.prepare_session(client, body, session, tokens)
            return
        tokens = self.tokenize(body)
        while True:
            with self.lock:
                session = self.store.select(client["id"], tokens)
                ticket = self.reserve(session["id"])
            with ticket:
                session = self.store.get(session["id"])
                if tokens[:len(session["tokens"])] != session["tokens"]:
                    continue  # A predecessor completed while queued; resolve the fork against its fresh journal.
                with self.slot_lease(session):
                    yield self.prepare_session(client, body, session, tokens)
                return

    def prepare_session(self, client, body, session, tokens):
        slot = session["slot"]
        info = self.decode.request("GET", f"/slots/{slot}/info")
        local_n = info["n"]
        if local_n > len(tokens) or info["prefix_hash"] != prefix_hash(tokens[:local_n]):
            self.decode.request("POST", f"/slots/{slot}?action=erase", {})
            local_n = 0
        # Imported memory has no logits. Decode the final prompt token locally.
        target = tokens[:-1]
        with self.lock:
            link_rate = self.link_rate
        decision = self.policy.choose(max(0, len(target) - local_n), local_n,
                                      bool(session["offload"]), link_rate)
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
                    with self.lock:
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
        if status == 503:
            self.send_header("Retry-After", "1")
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
            with app.prepare(client, body, explicit) as (session, prompt, request):
                with app.decode.open("POST", "/v1/chat/completions", request) as upstream:
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("X-Hybrid-Session", session["id"])
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        started = True
                        disconnected = False
                        for line in upstream:
                            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                                chunk = json.loads(line[6:])
                                chunk.pop("__verbose", None)
                                line = b"data: " + json.dumps(chunk).encode() + b"\n"
                            if not disconnected:
                                try:
                                    self.wfile.write(line)
                                    self.wfile.flush()
                                except OSError:
                                    # Drain the upstream before releasing its slot; finish records all evaluated tokens.
                                    disconnected = True
                                    self.close_connection = True
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
