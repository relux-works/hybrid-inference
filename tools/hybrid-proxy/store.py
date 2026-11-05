"""Persistent token journal and client-scoped session mapping."""
import timing
import hashlib
from functools import wraps
import json
import sqlite3
import struct
import time
import threading
import uuid

from journal import JournalConflict, matching_messages


@timing.traced('prefix.sha256')
def prefix_hash(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        if type(token) is not int or not 0 <= token < 2**31:
            raise ValueError("token is not a nonnegative int32")
        digest.update(struct.pack("<i", token))
    return digest.hexdigest()


def block_hashes(tokens):
    digest = hashlib.sha256()
    result = []
    for i, token in enumerate(tokens, 1):
        digest.update(struct.pack("<i", token))
        if i % 256 == 0:
            result.append(digest.hexdigest())
    return result


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class Store:
    def __init__(self, path, model):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(model TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, client TEXT NOT NULL, tokens TEXT NOT NULL,
                hashes TEXT NOT NULL, remote_id TEXT, remote_n INTEGER NOT NULL DEFAULT 0,
                slot INTEGER, used REAL NOT NULL, offload INTEGER NOT NULL DEFAULT 0);
            CREATE UNIQUE INDEX IF NOT EXISTS unique_slot ON sessions(slot);
        """)
        if "journal" not in {row[1] for row in self.db.execute("PRAGMA table_info(sessions)")}:
            self.db.execute("ALTER TABLE sessions ADD COLUMN journal TEXT")
        if "benchmark_mode" not in {row[1] for row in self.db.execute("PRAGMA table_info(sessions)")}:
            self.db.execute("ALTER TABLE sessions ADD COLUMN benchmark_mode TEXT")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(sessions)")}
        if "state_dtype" not in columns:
            self.db.execute("ALTER TABLE sessions ADD COLUMN state_dtype TEXT NOT NULL DEFAULT 'f32'")
        if "state_imports" not in columns:
            self.db.execute("ALTER TABLE sessions ADD COLUMN state_imports TEXT NOT NULL DEFAULT '[]'")
        row = self.db.execute("SELECT model FROM meta").fetchone()
        if row and row[0] != model:
            self.db.close()
            raise ValueError("database belongs to another model")
        with self.db:
            if row is None:
                self.db.execute("INSERT INTO meta VALUES(?)", (model,))
        # Runtime slots must be revalidated or restored after every proxy restart.
        self.verified = set()

    @locked
    def close(self):
        self.db.close()

    @locked
    def get(self, sid, client=None):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row or (client is not None and row["client"] != client):
            raise KeyError("session not found")
        row = dict(row)
        row["tokens"] = json.loads(row["tokens"])
        row["state_imports"] = json.loads(row["state_imports"])
        row["hashes"] = json.loads(row["hashes"])
        row["journal"] = json.loads(row["journal"]) if row["journal"] else None
        return row

    @locked
    @timing.traced('journal.sqlite_save_commit')
    def save(self, session):
        if "_turn" in session and not session.get("_commit"):
            return  # Keep the last completed token/message pair durable until generation is reconciled.
        tokens = session["tokens"]
        prefix_hash(tokens)
        with self.db:
            self.db.execute("""UPDATE sessions SET tokens=?,hashes=?,remote_id=?,remote_n=?,
                used=?,offload=?,journal=?,state_dtype=?,state_imports=? WHERE id=?""", (
                json.dumps(tokens), json.dumps(block_hashes(tokens)), session["remote_id"],
                session["remote_n"], time.time(), int(session["offload"]),
                json.dumps(session.get("journal")) if session.get("journal") else None, session.get("state_dtype", "f32"),
                json.dumps(session.get("state_imports", [])), session["id"]))

    @locked
    def commit_evaluated(self, session):
        current = self.get(session['id'], session['client'])
        if session['tokens'][:len(current['tokens'])] != current['tokens']:
            raise ValueError('production commit does not extend the durable journal')
        if session['remote_id'] is None and current['remote_id'] is not None:
            session.update(remote_id=current['remote_id'],remote_n=current['remote_n'])
        elif session['remote_id'] == current['remote_id']:
            session['remote_n'] = max(session['remote_n'],current['remote_n'])
        if not 0 <= session['remote_n'] <= len(session['tokens']):
            raise ValueError('production remote acknowledgement exceeds committed journal')
        self.save(session)

    @locked
    def acknowledge_remote(self, sid, client, remote_id, n, digest):
        current = self.get(sid, client)
        if (type(n) is not int or not 0 <= n <= len(current['tokens'])
                or digest != prefix_hash(current['tokens'][:n])):
            raise ValueError('background node ahead of or divergent from committed journal')
        if current['remote_id'] not in (None, remote_id):
            raise ValueError('background acknowledgement belongs to a stale remote session')
        with self.db:
            self.db.execute('UPDATE sessions SET remote_id=?,remote_n=? WHERE id=?', (remote_id,n,sid))

    @locked
    def benchmark_mode(self, sid, mode):
        with self.db:
            self.db.execute("UPDATE sessions SET benchmark_mode=? WHERE id=?", (mode or "default", sid))

    @locked
    def set_slot(self, sid, slot):
        # Replay may save an older session copy while an inactive slot is parked; it must not restore that binding.
        with self.db:
            self.db.execute("UPDATE sessions SET slot=? WHERE id=?", (slot, sid))
        self.verified.discard(sid)

    @locked
    def select(self, client, tokens, explicit=None):
        prefix_hash(tokens)
        if explicit:
            row = self.db.execute("SELECT client FROM sessions WHERE id=?", (explicit,)).fetchone()
            if row:
                session = self.get(explicit, client)
                if tokens[:len(session["tokens"])] != session["tokens"]:
                    raise JournalConflict("explicit session history is not append-only")
                return session
        else:
            hashes = block_hashes(tokens)
            best = None
            for row in self.db.execute("SELECT id FROM sessions WHERE client=? ORDER BY used DESC", (client,)):
                session = self.get(row["id"])
                blocks = 0
                for a, b in zip(hashes, session["hashes"]):
                    if a != b:
                        break
                    blocks += 1
                matched = blocks * 256
                old = session["tokens"]
                while matched < min(len(tokens), len(old)) and tokens[matched] == old[matched]:
                    matched += 1
                if best is None or matched > best[0]:
                    best = (matched, session)
            if best and best[0] > 0 and best[0] == len(best[1]["tokens"]):
                return best[1]
        sid = explicit or str(uuid.uuid4())
        with self.db:
            self.db.execute("INSERT INTO sessions(id,client,tokens,hashes,used) VALUES(?,?,'[]','[]',?)",
                            (sid, client, time.time()))
        return self.get(sid)

    @locked
    def select_messages(self, client, messages, identity, tokens, explicit=None, create=True):
        if explicit:
            try:
                return self.get(explicit, client)
            except KeyError:
                if self.db.execute("SELECT 1 FROM sessions WHERE id=?", (explicit,)).fetchone():
                    raise
                return self.select(client, [], explicit)
        candidates = []
        forks = []
        for row in self.db.execute("SELECT id FROM sessions WHERE client=? ORDER BY used DESC", (client,)):
            session = self.get(row["id"])
            journal = session["journal"]
            if (journal and journal["identity"] == identity and
                    matching_messages(journal["messages"], messages) == len(journal["messages"])):
                candidates.append((len(journal["messages"]), session))
            elif journal and journal["identity"] == identity:
                matched = matching_messages(journal["messages"], messages)
                if matched and any(cp["message_index"] <= matched for cp in journal.get("checkpoints", [])):
                    session["_fork"] = True
                    forks.append((matched, session))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        if not create:
            return None
        if forks:
            return max(forks, key=lambda item: item[0])[1]
        # An inferred divergence is a fork. An explicit ID authorizes checkpoint rollback of that session.
        session = self.select(client, tokens)
        return self.select(client, []) if session.get("journal") else session

    @locked
    def occupant(self, slot):
        row = self.db.execute("SELECT id FROM sessions WHERE slot=?", (slot,)).fetchone()
        return self.get(row[0]) if row else None

    @locked
    def pending(self):
        return [self.get(row[0]) for row in self.db.execute("SELECT id FROM sessions")]
