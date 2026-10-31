"""Hybrid client proxy. Credentials are read from a JSON file, never CLI arguments."""
import argparse
import copy
import hashlib
from contextlib import contextmanager
import http.server
import http.client
import json
import logging
import os
import struct
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from policy import Policy
from concurrency import SessionQueue
from journal import Capture, JournalConflict, Renderer, matching_messages, segment_range, settings_identity
from store import Store, prefix_hash

LOG = logging.getLogger("hybrid")
MAX_JSON = 8 * 1024 * 1024
MAX_DELTA = 512 * 1024 * 1024 + 1024 * 1024 + 52
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
        self.last_transfer_seconds = 0
        self.last_bytes = 0
        self.last_prefill_ms = None

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
                self.last_transfer_seconds = elapsed
                self.last_bytes = len(body)
                self.last_prefill_ms = getattr(response, "headers", {}).get("X-Hybrid-Prefill-Ms")
                if len(body) > limit:
                    raise ValueError("upstream body too large")
                return body if binary else (json.loads(body) if body else None)
        except http.client.HTTPException as error:
            raise OSError("upstream HTTP connection interrupted") from error


class Proxy:
    def __init__(self, config, database):
        self.model = config["model_sha256"]
        self.node_model = config.get("node_model_sha256", self.model)
        self.hetero_quant = config.get("hetero_quant", False)
        self.layout_fingerprint = config.get("layout_fingerprint")
        if self.hetero_quant:
            if (not isinstance(self.layout_fingerprint, str) or len(self.layout_fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in self.layout_fingerprint)
                    or self.layout_fingerprint != config.get("node_layout_fingerprint")):
                raise ValueError("heterogeneous layout fingerprint mismatch")
        elif self.node_model != self.model:
            raise ValueError("node/client model sha256 mismatch")
        self.benchmark = config.get("benchmark", False)
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
        body_tokens = int((512 * 1024**2 - self.policy.state_bytes - 1024**2) // max(1, self.policy.kv_bytes))
        safe_chunk = min((1024**2 - 4096) // 11, body_tokens)
        self.extend_chunk_tokens = config.get("extend_chunk_tokens", safe_chunk)
        if type(self.extend_chunk_tokens) is not int or not 1 <= self.extend_chunk_tokens <= safe_chunk:
            raise ValueError("extend_chunk_tokens exceeds delta body/header bound")
        self.slots = config.get("slots", 1)
        if self.slots < 1:
            raise ValueError("slots must be positive")
        self.lock = self.store.lock
        self.slot_available = threading.Condition(self.lock)
        self.pinned = set()
        self.quarantined = set()
        self.queues = {}
        self.admissions = {}
        self.link_rate = self.policy.link_rate
        self.stop = threading.Event()
        self.renderer = Renderer(self.decode)
        self.inflight = {}

    def client_delta(self, delta):
        if not self.hetero_quant:
            return delta
        if len(delta) < 4:
            raise ValueError("truncated heterogeneous envelope")
        size, = struct.unpack("<I", delta[:4])
        if not 0 < size <= 1024**2 or 4 + size + 48 > len(delta):
            raise ValueError("invalid heterogeneous envelope")
        header = json.loads(delta[4:4 + size])
        if header.get("model_sha256") != self.node_model:
            raise ValueError("heterogeneous delta source mismatch")
        # Rebind only the envelope identity after live layout admission. Native import verifies state and checksums.
        header["model_sha256"] = self.model
        encoded = json.dumps(header, separators=(",", ":")).encode()
        if len(encoded) > 1024**2:
            raise ValueError("heterogeneous header too large")
        return struct.pack("<I", len(encoded)) + encoded + delta[4 + size:]

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
        if self.hetero_quant:
            health = node.request("GET", "/v1/health")
            if health.get("model_sha256") != self.node_model or health.get("layout_fingerprint") != self.layout_fingerprint:
                raise HTTPError(412)
        if not session["remote_id"]:
            session["remote_id"] = node.request("POST", "/v1/sessions", {"model_sha256": self.node_model})["session_id"]
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
        if (info["model_sha256"] != self.node_model or n < 0 or n > len(session["tokens"])
                or info["prefix_hash"] != prefix_hash(session["tokens"][:n])):
            LOG.error("divergence session=%s local_hash=%s remote_hash=%s", session["id"],
                      prefix_hash(session["tokens"]), info.get("prefix_hash"))
            raise ValueError("server prefix diverged; explicit snapshot recovery required")
        acknowledged = max(0, n - session["remote_n"])
        if acknowledged:
            session["_remote_replayed"] = session.get("_remote_replayed", 0) + acknowledged
            LOG.info("remote_applied session=%s n0=%d n1=%d kind=shadow", session["id"], session["remote_n"], n)
        session["remote_n"] = n
        self.store.save(session)
        if n < len(session["tokens"]):
            node.request("POST", path + "/extend", {"n0": n, "prefix_hash_n0": prefix_hash(session["tokens"][:n]),
                         "tokens": session["tokens"][n:], "return_delta": False},
                         headers={"X-Hybrid-Bench": "1"} if client.get("benchmark_mode") else None)
            if client.get("benchmark_mode") and node.last_prefill_ms is not None:
                session["_bench"]["server_shadow_ms"] += float(node.last_prefill_ms)
        journal = session.get("journal")
        if journal:
            retired = journal.get("retired_remote_ids", [])
            for sid in retired[:1]:
                if sid == session["remote_id"]:
                    continue
                try:
                    node.request("DELETE", "/v1/sessions/" + urllib.parse.quote(sid, safe=""))
                except HTTPError as error:
                    if error.status != 404:
                        continue
                except OSError:
                    continue
                journal["retired_remote_ids"] = [value for value in retired if value != sid]
                self.store.save(session)
        return path

    def resync(self, session, client, prompt):
        node = self.remote(client)
        path = "/v1/sessions/" + urllib.parse.quote(session["remote_id"], safe="")
        info = node.request("GET", path)
        LOG.warning("resync session=%s local_hash=%s remote_hash=%s", session["id"],
                    prefix_hash(session["tokens"]), info.get("prefix_hash"))
        if info["model_sha256"] != self.node_model:
            raise HTTPError(409)
        if (0 <= info["n"] <= len(session["tokens"])
                and info["prefix_hash"] == prefix_hash(session["tokens"][:info["n"]])):
            self.sync(session, client)
            return
        n = info["n"]
        if not 0 <= n <= len(prompt) or info["prefix_hash"] != prefix_hash(prompt[:n]):
            raise HTTPError(409)
        if self.hetero_quant:
            # Whole snapshots carry the node identity; recover this ahead prefix by local continuation.
            raise HTTPError(503)
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
                        current = self.store.get(pending["id"])
                        if (current.get("benchmark_mode") or (current.get("journal") or {}).get("last_meter", {}).get("benchmark_mode")) not in ("local", "closed"):
                            self.sync(current, client)
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
        rollback_pending = any((self.slot_path / (session["id"] + suffix)).exists()
                               for suffix in ("-resync-backup.bin", "-rewrite-backup.bin"))
        slot = session["slot"]
        if rollback_pending:
            self.decode.request("POST", f"/slots/{slot}?action=erase", {})
            self.verified(session["id"], True)
            return slot
        if not self.verified(session["id"]):
            info = self.decode.request("GET", f"/slots/{slot}/info")
            if (not (session.get("_message_mode") or session.get("journal")) and info["n"] >= len(session["tokens"])
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

    def select_messages(self, client, messages, identity, tokens, explicit=None):
        # Called under the shared metadata lock. Pending history is visible for admission, never durable replay.
        if explicit:
            return self.store.select_messages(client, messages, identity, tokens, explicit)
        completed = self.store.select_messages(client, messages, identity, tokens, create=False)
        active = [value for value in self.inflight.values() if value["client"] == client and
                  value["identity"] == identity and len(messages) > len(value["messages"]) and
                  messages[len(value["messages"])].get("role") == "assistant" and
                  matching_messages(value["messages"], messages) == len(value["messages"])]
        pending = max(active, key=lambda value: len(value["messages"])) if active else None
        if completed and (not pending or len(completed["journal"]["messages"]) >= len(pending["messages"])):
            return completed
        if pending:
            return self.store.get(pending["id"], client)
        return self.store.select_messages(client, messages, identity, tokens)

    @contextmanager
    def prepare(self, client, body, explicit):
        if body.get("n", 1) != 1:
            raise ValueError("only one completion per request is supported")
        if "prompt_tokens" in body:
            raise ValueError("prompt_tokens is managed by the proxy")
        messages = body.get("messages")
        message_mode = isinstance(messages, list) and bool(messages)
        with self.lock:
            if explicit:
                try:
                    existing = self.store.get(explicit)
                except KeyError:
                    existing = None
                if existing and existing["client"] != client["id"]:
                    raise KeyError("session not found")
            admission = self.admissions.setdefault(client["id"], SessionQueue()).reserve()
        try:
            tokens = self.tokenize(body)
            template = self.renderer.template_identity() if message_mode else None
            identity = settings_identity(body, template) if message_mode else None
        except BaseException:
            admission.__exit__(None, None, None)
            raise
        # HTTP classification runs concurrently; retire its metadata in arrival order before session admission.
        with admission:
            with self.lock:
                if message_mode:
                    session = self.select_messages(client["id"], messages, identity, tokens, explicit)
                elif explicit and existing:
                    session = existing
                else:
                    session = self.store.select(client["id"], tokens, explicit)
                ticket = self.reserve(session["id"])
                fork_parent = session.get("_fork", False)
        forked = False
        while True:
            with ticket:
                if message_mode:
                    session = self.store.get(session["id"], client["id"])
                    journal = session.get("journal")
                    compatible = (not journal or journal["identity"] == identity and
                                  matching_messages(journal["messages"], messages) == len(journal["messages"]))
                    if fork_parent and not compatible:
                        with self.slot_lease(session):
                            child = self.store.select(client["id"], [])
                            self.decode.request("POST", f"/slots/{session['slot']}?action=save",
                                                {"filename": child["id"] + ".bin"})
                            child.update(tokens=list(session["tokens"]), journal=copy.deepcopy(journal))
                            with self.lock:
                                next_ticket = self.reserve(child["id"])
                                try:
                                    self.store.save(child)
                                except BaseException:
                                    next_ticket.__exit__(None, None, None)
                                    raise
                        next_session = child
                    elif explicit or compatible or forked:
                        session["_message_mode"] = True
                        with self.slot_lease(session):
                            try:
                                with self.prepare_messages(client, body, session, tokens, identity, template):
                                    yield self.prepare_session(client, body, session, session["_prompt"])
                            except JournalConflict:
                                if not forked or session.get("_prepared"):
                                    raise
                                # The copied native checkpoint was evicted or incompatible. Only the child may reset.
                                self.decode.request("POST", f"/slots/{session['slot']}?action=erase", {})
                                session.update(tokens=[], journal=None, remote_id=None, remote_n=0)
                                self.store.save(session)
                                with self.prepare_messages(client, body, session, tokens, identity, template):
                                    yield self.prepare_session(client, body, session, session["_prompt"])
                        return
                elif explicit:
                    session = self.store.select(client["id"], tokens, explicit)
                else:
                    session = self.store.get(session["id"])
                if not message_mode and (explicit or tokens[:len(session["tokens"])] == session["tokens"]):
                    with self.slot_lease(session):
                        yield self.prepare_session(client, body, session, tokens)
                    return
            # A predecessor completed while queued; resolve its fork against the fresh journal.
            if message_mode and fork_parent and not compatible:
                session, ticket = next_session, next_ticket
                fork_parent, forked = False, True
                continue
            with self.lock:
                session = (self.select_messages(client["id"], messages, identity, tokens) if message_mode else
                           self.store.select(client["id"], tokens))
                ticket = self.reserve(session["id"])
                fork_parent = session.get("_fork", False)

    @contextmanager
    def prepare_messages(self, client, body, session, full_tokens, identity, template):
        session["_prepared"] = False
        journal = session.get("journal")
        text = self.renderer.render(body)
        if self.renderer.encode(text, initial=True) != full_tokens:
            raise JournalConflict("template changed during request admission")
        old = copy.deepcopy(session)
        recorded_messages = copy.deepcopy(body["messages"])
        if journal:
            matched = matching_messages(journal["messages"], recorded_messages)
            for index in range(matched):
                if (recorded_messages[index].get("role") == "assistant" and
                        "reasoning_content" not in recorded_messages[index] and
                        "reasoning_content" in journal["messages"][index]):
                    recorded_messages[index]["reasoning_content"] = journal["messages"][index]["reasoning_content"]
        backup = None
        rollback_n = None
        probe_evaluated = 0
        try:
            if not journal:
                prompt = full_tokens
                boundaries = self.renderer.user_boundaries(body, text, full_tokens)
                checkpoints = []
                segment_start = 0
                replay_end = 0
                message_start, token_base, anchor_base = 0, 0, 0
            elif (journal["identity"] == identity and
                  matching_messages(journal["messages"], body["messages"]) == len(journal["messages"])):
                segment_start = len(session["tokens"])
                replay_end = segment_start
                prompt, boundaries, _, tail = self.renderer.append(body, text, full_tokens,
                    len(journal["messages"]), session["tokens"])
                checkpoints = copy.deepcopy(journal.get("checkpoints", []))
                message_start = len(journal["messages"])
                token_base, anchor_base = len(full_tokens) - len(tail), len(prompt) - len(tail)
            else:
                if journal["identity"] != identity:
                    raise JournalConflict("template or rendering settings changed")
                matched = matching_messages(journal["messages"], body["messages"])
                usable = [cp for cp in journal.get("checkpoints", []) if cp["message_index"] <= matched]
                if not usable:
                    raise JournalConflict("history rewrite has no user-boundary checkpoint")
                cp = max(usable, key=lambda item: item["message_index"])
                rollback_n = cp["n"]
                prefix = session["tokens"][:rollback_n]
                if prefix_hash(prefix) != cp["prefix_hash"]:
                    raise JournalConflict("checkpoint prefix does not match journal")
                index = cp["message_index"]
                message_start = index
                anchors = [segment for segment in journal.get("segments", []) if segment["kind"] == "generated"
                           and segment["end"] >= rollback_n and segment["message_indices"]
                           and max(segment["message_indices"]) < matched]
                replay_end = rollback_n
                if anchors:
                    anchor = max(anchors, key=lambda segment: segment["end"])
                    replay_end = anchor["end"]
                    message_start = max(anchor["message_indices"]) + 1
                    prompt, boundaries, _, tail = self.renderer.append(body, text, full_tokens,
                        message_start, session["tokens"][:replay_end])
                    token_base, anchor_base = len(full_tokens) - len(tail), len(prompt) - len(tail)
                elif index and body["messages"][index - 1].get("role") == "assistant":
                    _, mapped, closing, tail = self.renderer.append(body, text, full_tokens, index, [])
                    prompt = prefix + tail[closing:]
                    boundaries = [dict(message_index=b["message_index"], n=rollback_n + b["n"] - closing)
                                  for b in mapped]
                    token_base, anchor_base = len(full_tokens) - len(tail), rollback_n - closing
                else:
                    boundaries = self.renderer.user_boundaries(body, text, full_tokens)
                    found = next((b for b in boundaries if b["message_index"] == index), None)
                    if not found or full_tokens[:found["n"]] != prefix:
                        raise JournalConflict("rewritten prefix cannot retain the checkpoint")
                    prompt = full_tokens
                    token_base, anchor_base = 0, 0
                header = self.renderer.encode("<|im_start|>user\n")
                trial = prompt[:rollback_n + len(header)]
                if trial[rollback_n:] != header:
                    raise JournalConflict("checkpoint is not at a user boundary")
                backup = session["id"] + "-rewrite-backup.bin"
                self.decode.request("POST", f"/slots/{session['slot']}?action=save", {"filename": backup})
                result = self.decode.request("POST", "/v1/chat/completions", dict(body,
                    prompt_tokens=trial, id_slot=session["slot"], n_predict=0, stream=False,
                    cache_prompt=True, return_tokens=True, verbose=True))
                probe_evaluated = result["timings"]["prompt_n"]
                if result["timings"]["cache_n"] != rollback_n:
                    raise JournalConflict("native user-boundary checkpoint is no longer available")
                actual = self.decode.request("GET", f"/slots/{session['slot']}/info")
                if actual["tokens"] != trial or actual["prefix_hash"] != prefix_hash(trial):
                    raise JournalConflict("native checkpoint trial changed the token prefix")
                session["tokens"] = trial
                session["remote_id"], session["remote_n"] = None, 0
                checkpoints = [cp for cp in usable if cp["n"] <= rollback_n]
                segment_start = rollback_n
            layout, generation_start = self.renderer.layout(body, text, full_tokens, start=message_start,
                                                            token_base=token_base, anchor_base=anchor_base)
            session["_prompt"] = prompt
            session["_capture"] = Capture()
            session["_turn"] = dict(identity=identity, template=template, messages=recorded_messages,
                boundaries=boundaries, checkpoints=checkpoints, start=segment_start,
                rollback_n=rollback_n, probe_evaluated=probe_evaluated, message_start=message_start,
                replay_end=replay_end,
                layout=layout, generation_start=generation_start,
                settings=hashlib.sha256(json.dumps({key: value for key, value in body.items() if key != "messages"},
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                retired_remote=old["remote_id"] if rollback_n is not None else None)
            session["_prepared"] = True
            with self.lock:
                self.inflight[session["id"]] = dict(id=session["id"], client=client["id"], identity=identity,
                                                    messages=recorded_messages)
            yield
        except BaseException:
            if backup:
                try:
                    self.decode.request("POST", f"/slots/{session['slot']}?action=restore", {"filename": backup})
                    self.store.save(old)
                    self.verified(session["id"], True)
                except (OSError, HTTPError, ValueError):
                    self.verified(session["id"], False)
                    self.store.set_slot(session["id"], None)
                    # Both names mark the decoder untrusted; keep the original file if the rename fails.
                    try:
                        marker = self.slot_path / (session["id"] + "-resync-backup.bin")
                        os.replace(self.slot_path / backup, marker)
                    except OSError:
                        LOG.error("rewrite backup retained session=%s", session["id"])
                    backup = None
                    raise RollbackError() from None
            raise
        finally:
            with self.lock:
                self.inflight.pop(session["id"], None)
            if backup:
                (self.slot_path / backup).unlink(missing_ok=True)

    def prepare_session(self, client, body, session, tokens):
        slot = session["slot"]
        self.store.benchmark_mode(session["id"], client.get("benchmark_mode"))
        info = self.decode.request("GET", f"/slots/{slot}/info")
        local_n = info["n"]
        if local_n > len(tokens) or info["prefix_hash"] != prefix_hash(tokens[:local_n]):
            if "_turn" in session:
                raise JournalConflict("slot does not extend the message journal")
            self.decode.request("POST", f"/slots/{slot}?action=erase", {})
            local_n = 0
        # Imported memory has no logits. Decode the final prompt token locally.
        target = tokens[:-1]
        session["_remote_applied"] = 0
        session["_remote_replayed"] = 0
        with self.lock:
            link_rate = self.link_rate
        decision = self.policy.choose(max(0, len(target) - local_n), local_n,
                                      bool(session["offload"]), link_rate)
        mode = client.get("benchmark_mode")
        session["_bench"] = dict(mode=mode, path="local", depth_before=local_n,
            prompt_depth=len(tokens), server_prefill_ms=None, wire_bytes=0,
            server_shadow_ms=0, transfer_ms=0, import_ms=0, remote_prepare_ms=0, chunks=[], fallback_reason=None)
        if mode == "offload":
            decision["offload"] = len(target) > local_n
        elif mode == "local":
            decision["offload"] = False
        try:
            remote_started = time.monotonic()
            path = self.sync(session, client) if mode != "local" else None
            if decision["offload"] and len(target) > local_n:
                # Flush a queued shadow with a critical extend at the local boundary.
                if session["remote_n"] > local_n:
                    raise ValueError("server is ahead of local slot; local re-prefill required")
                if session["remote_n"] < local_n:
                    self.sync(session, client)
                remote = self.remote(client)
                for n0 in range(local_n, len(target), self.extend_chunk_tokens):
                    n1 = min(n0 + self.extend_chunk_tokens, len(target))
                    chunk = dict(n0=n0, n1=n1, status="requested", wire_bytes=0, import_ms=0)
                    session["_bench"]["chunks"].append(chunk)
                    delta = remote.request("POST", path + "/extend", {
                        "n0": n0, "prefix_hash_n0": prefix_hash(tokens[:n0]),
                        "tokens": target[n0:n1], "return_delta": True},
                        headers={"X-Hybrid-Bench": "1"} if mode else None, binary=True)
                    chunk.update(status="received", wire_bytes=remote.last_bytes,
                        transfer_ms=remote.last_transfer_seconds * 1000,
                        server_prefill_ms=float(remote.last_prefill_ms) if remote.last_prefill_ms is not None else None)
                    meter = session["_bench"]
                    meter["wire_bytes"] += chunk["wire_bytes"]
                    meter["transfer_ms"] += chunk["transfer_ms"]
                    if chunk["server_prefill_ms"] is not None:
                        meter["server_prefill_ms"] = (meter["server_prefill_ms"] or 0) + chunk["server_prefill_ms"]
                    if remote.last_transfer_rate:
                        with self.lock:
                            self.link_rate = 0.8 * self.link_rate + 0.2 * remote.last_transfer_rate
                    delta = self.client_delta(delta)
                    import_started = time.monotonic()
                    self.decode.request("POST", f"/slots/{slot}?action=import_delta", delta,
                                        {"Content-Type": "application/octet-stream",
                                         "X-Hybrid-Session": session["remote_id"]})
                    chunk["import_ms"] = (time.monotonic() - import_started) * 1000
                    meter["import_ms"] += chunk["import_ms"]
                    actual = self.decode.request("GET", f"/slots/{slot}/info")
                    if (actual["n"] != n1 or actual["prefix_hash"] != prefix_hash(tokens[:n1])
                            or actual["tokens"] != tokens[:n1]):
                        self.decode.request("POST", f"/slots/{slot}?action=erase", {})
                        session["_remote_applied"] = 0
                        raise ValueError("import prefix mismatch")
                    chunk["status"] = "imported"
                    session.update(offload=True, tokens=tokens[:n1], remote_n=n1)
                    session["_remote_applied"] = n1 - local_n
                    LOG.info("remote_applied session=%s n0=%d n1=%d kind=critical", session["id"], n0, n1)
                    if "_turn" in session:
                        session["_turn"]["checkpoints"] = []
                    self.store.save(session)
        except HTTPError as error:
            reason = f"http_{error.status}"
            if error.status == 503:
                try:
                    payload = json.loads(error.body)
                    if payload.get("error") in ("ram budget", "RAM session budget exceeded"):
                        reason = "ram_budget"
                except (ValueError, AttributeError):
                    pass
            session["_bench"]["fallback_reason"] = reason
            if error.status == 409:
                LOG.warning("409 resync session=%s", session["id"])
                # GET validates the acknowledged prefix; replay uses its exact boundary.
                self.recover(session, client, tokens)
            elif error.status not in (404, 408, 429, 500, 502, 503, 504):
                raise
            session["offload"] = False
        except ValueError:
            session["_bench"]["fallback_reason"] = "invalid_delta"
            if session["remote_id"]:
                self.recover(session, client, tokens)
            session["offload"] = False
        except OSError:
            session["_bench"]["fallback_reason"] = "transport"
            session["offload"] = False
        session["_bench"]["remote_prepare_ms"] = (time.monotonic() - remote_started) * 1000
        session["_bench"]["path"] = "offload" if session["_remote_applied"] else "local"
        session["tokens"] = tokens
        self.store.save(session)
        request = dict(body, id_slot=slot, cache_prompt=True)
        if "_turn" in session:
            request.update(prompt_tokens=tokens, return_tokens=True, verbose=True)
        return session, tokens, request

    def finish(self, session, prompt):
        info = self.decode.request("GET", f"/slots/{session['slot']}/info")
        evaluated = info["tokens"]
        if (len(evaluated) != info["n"] or evaluated[:len(prompt)] != prompt
                or prefix_hash(evaluated) != info["prefix_hash"]):
            raise ValueError("decode token journal does not match slot")
        if "_turn" in session:
            turn, capture = session["_turn"], session["_capture"]
            native = capture.native
            if not native or not isinstance(native.get("tokens"), list):
                raise JournalConflict("decoder did not return the complete sampled IDs")
            complete = prompt + native["tokens"]
            if evaluated != complete[:len(evaluated)] or len(complete) - len(evaluated) not in (0, 1):
                raise JournalConflict("sampled and evaluated token boundaries differ")
            generated_evaluated = 0
            if len(evaluated) < len(complete):
                # Evaluate the last sampled token as generation reconciliation, before the next prompt begins.
                tail = self.decode.request("POST", "/completion", dict(prompt=complete, n_predict=0,
                    id_slot=session["slot"], cache_prompt=True))
                generated_evaluated = tail["timings"]["prompt_n"]
                info = self.decode.request("GET", f"/slots/{session['slot']}/info")
                evaluated = info["tokens"]
                if evaluated != complete or info["prefix_hash"] != prefix_hash(complete):
                    raise JournalConflict("generated-token reconciliation changed the prefix")
            timings = native.get("timings") or capture.timings or {}
            local = timings["prompt_n"] + turn["probe_evaluated"]
            remote = session["_remote_applied"]
            checkpoints = turn["checkpoints"]
            if not remote:
                cached = turn["rollback_n"] if turn["rollback_n"] is not None else timings["cache_n"]
                for boundary in turn["boundaries"]:
                    if boundary["n"] >= cached and boundary["n"] > 0:
                        cp = dict(boundary, prefix_hash=prefix_hash(prompt[:boundary["n"]]))
                        if cp not in checkpoints:
                            checkpoints.append(cp)
            meter = dict(local_evaluated=local, remote_applied=remote,
                generated_evaluated=generated_evaluated, new_tokens=len(prompt) - turn["replay_end"],
                replayed_prefix=turn["replay_end"] - turn["start"],
                remote_replay_acknowledged=session.get("_remote_replayed", 0),
                rollback_n=turn["rollback_n"], prefilled=local + remote)
            if session.get("_bench", {}).get("mode"):
                meter["benchmark_mode"] = session["_bench"]["mode"]
                session["_bench"].update(meter, depth_after=len(complete), generated_tokens=len(native["tokens"]),
                    decode_ms=timings.get("predicted_ms"), slot=session["slot"])
            previous = session.get("journal") or {}
            retired = list(previous.get("retired_remote_ids", []))
            if turn["retired_remote"] and turn["retired_remote"] != session["remote_id"]:
                if turn["retired_remote"] not in retired:
                    retired.append(turn["retired_remote"])
            segments = segment_range(previous.get("segments", []), 0, turn["start"])
            if turn["replay_end"] > turn["start"]:
                segments += segment_range(previous.get("segments", []), turn["start"], turn["replay_end"], replayed=True)
            segments += [dict(kind="incoming", start=turn["replay_end"], end=len(prompt),
                             message_indices=list(range(turn["message_start"], len(turn["messages"])))),
                         dict(kind="generated", start=len(prompt), end=len(complete),
                             message_indices=[len(turn["messages"])], sampled_ids=native["tokens"])]
            layout = [row for row in previous.get("message_boundaries", []) if row["message_index"] < turn["message_start"]]
            if layout and turn["layout"]:
                layout[-1]["end"] = turn["layout"][0]["start"]
            layout += turn["layout"]
            messages = turn["messages"]
            if messages[-1]["role"] == "assistant":
                for key in ("content", "reasoning_content"):
                    messages[-1][key] = (messages[-1].get(key) or "") + (capture.message.get(key) or "")
                layout[-1]["end"] = dict(n=len(complete), byte_offset=0)
                segments[-1]["message_indices"] = [len(messages) - 1]
            else:
                layout.append(dict(message_index=len(messages), role="assistant", start=turn["generation_start"],
                                   end=dict(n=len(complete), byte_offset=0)))
                messages = messages + [capture.message]
            session["journal"] = dict(identity=turn["identity"], template=turn["template"],
                settings_sha256=turn["settings"], messages=messages, message_boundaries=layout,
                segments=segments, checkpoints=checkpoints, retired_remote_ids=retired,
                evaluated_boundary=len(evaluated), sampled_boundary=len(complete),
                last_meter=meter, meters=previous.get("meters", []) + [meter])
            LOG.info("prefill session=%s local=%d remote=%d generated=%d new=%d rollback=%s", session["id"],
                     local, remote, generated_evaluated, meter["new_tokens"], turn["rollback_n"])
        session["tokens"] = evaluated
        session["_commit"] = True
        self.store.save(session)
        (self.slot_path / (session["id"] + "-resync-backup.bin")).unlink(missing_ok=True)
        if "_turn" not in session or session["_turn"]["rollback_n"] is None:
            (self.slot_path / (session["id"] + "-rewrite-backup.bin")).unlink(missing_ok=True)
        # Replay is asynchronous; the durable suffix beyond remote_n is the journal.


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "HybridProxy/0.1"

    def log_message(self, *_):
        pass  # Do not log credentials, request bodies, or token content.

    def do_DELETE(self):
        app = self.server.app
        try:
            client = app.client(self.headers.get("Authorization", ""))
            if not app.benchmark or not self.path.startswith("/v1/sessions/"):
                raise HTTPError(404)
            sid = urllib.parse.unquote(self.path[len("/v1/sessions/"):])
            with app.reserve(sid):
                session = app.store.get(sid, client["id"])
                app.store.benchmark_mode(sid, "closed")
                ids = [session["remote_id"]] + (session.get("journal") or {}).get("retired_remote_ids", [])
                for remote_id in dict.fromkeys(ids):
                    if remote_id:
                        try:
                            app.remote(client).request("DELETE", "/v1/sessions/" + urllib.parse.quote(remote_id, safe=""))
                        except HTTPError as error:
                            if error.status != 404:
                                raise
                self.json_response(200, {"status": "released"})
        except KeyError:
            self.json_response(404, {"error": "session not found"})
        except (HTTPError, OSError) as error:
            self.json_response(error.status if isinstance(error, HTTPError) else 503, {"error": "release failed"})

    def do_GET(self):
        if not self.server.app.benchmark:
            self.send_error(501, "Unsupported method ('GET')")
            return
        try:
            self.server.app.client(self.headers.get("Authorization", ""))
            if self.path != "/health" or not self.server.app.benchmark:
                raise HTTPError(404)
            self.json_response(200, {"status": "ok"})
        except HTTPError as error:
            self.json_response(error.status, {"error": "unavailable"})

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
            mode = self.headers.get("X-Hybrid-Bench-Mode")
            if mode:
                if not app.benchmark or mode not in ("local", "hybrid", "offload"):
                    raise ValueError("benchmark mode requires benchmark=true and local, hybrid or offload")
                client = dict(client, benchmark_mode=mode)
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
                            if mode and line.strip() == b"data: [DONE]":
                                continue
                            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                                chunk = json.loads(line[6:])
                                if "_capture" in session:
                                    session["_capture"].feed(chunk)
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
                        if "_capture" in session:
                            session["_capture"].feed(data)
                        data.pop("__verbose", None)
                        app.finish(session, prompt)
                        if mode:
                            data["hybrid_bench"] = session["_bench"]
                        self.send_response(200)
                        output = json.dumps(data).encode()
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(output)))
                        self.send_header("X-Hybrid-Session", session["id"])
                        self.end_headers()
                        self.wfile.write(output)
                        return
                app.finish(session, prompt)
                if mode:
                    self.wfile.write(b"data: " + json.dumps({"hybrid_bench": session["_bench"]}).encode() + b"\n\ndata: [DONE]\n\n")
                    self.wfile.flush()
        except (HTTPError, KeyError, ValueError, OSError) as error:
            if not started:
                status = (error.status if isinstance(error, HTTPError) else 404 if isinstance(error, KeyError) else
                          409 if isinstance(error, JournalConflict) else 400)
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
