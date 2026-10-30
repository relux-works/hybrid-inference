"""Persistent token journal and client-scoped session mapping."""
import hashlib
import json
import sqlite3
import struct
import time
import uuid


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


class Store:
    def __init__(self, path, model):
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
        """)
        row = self.db.execute("SELECT model FROM meta").fetchone()
        if row and row[0] != model:
            self.db.close()
            raise ValueError("database belongs to another model")
        with self.db:
            if row is None:
                self.db.execute("INSERT INTO meta VALUES(?)", (model,))
        # Runtime slots must be revalidated or restored after every proxy restart.
        self.verified = set()

    def close(self):
        self.db.close()

    def get(self, sid, client=None):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row or (client is not None and row["client"] != client):
            raise KeyError("session not found")
        row = dict(row)
        row["tokens"] = json.loads(row["tokens"])
        row["hashes"] = json.loads(row["hashes"])
        return row

    def save(self, session):
        tokens = session["tokens"]
        prefix_hash(tokens)
        with self.db:
            self.db.execute("""UPDATE sessions SET tokens=?,hashes=?,remote_id=?,remote_n=?,
                slot=?,used=?,offload=? WHERE id=?""", (
                json.dumps(tokens), json.dumps(block_hashes(tokens)), session["remote_id"],
                session["remote_n"], session["slot"], time.time(), int(session["offload"]), session["id"]))

    def select(self, client, tokens, explicit=None):
        prefix_hash(tokens)
        if explicit:
            row = self.db.execute("SELECT client FROM sessions WHERE id=?", (explicit,)).fetchone()
            if row:
                session = self.get(explicit, client)
                if tokens[:len(session["tokens"])] != session["tokens"]:
                    raise ValueError("explicit session history is not append-only")
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

    def occupant(self, slot):
        row = self.db.execute("SELECT id FROM sessions WHERE slot=?", (slot,)).fetchone()
        return self.get(row[0]) if row else None

    def pending(self):
        return [self.get(row[0]) for row in self.db.execute("SELECT id FROM sessions")]
