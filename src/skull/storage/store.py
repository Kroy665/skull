"""Local vector store for conversation history and persona/knowledge facts.

Embeddings are computed locally (sentence-transformers, all-MiniLM-L6-v2) so
this works fully offline with no dependency on the Qwen endpoint (which does
not expose an embeddings API).

Storage layout (memory/, at the project root - see skull.config.MEMORY_DIR):
    conversations.db  - every auto-logged chat turn
    persona.db        - curated facts about the user (identity, preferences,
                         behavior)

Each store is a single SQLite database with two tables: `entries` (id, text,
metadata, created_at) and a sqlite-vec vec0 virtual table `vec_items` keyed
by the same id, holding the L2-normalized embedding. Cosine similarity is
just the dot product since vectors are unit-normalized - sqlite-vec's `MATCH
... ORDER BY distance` does this natively via its L2/cosine ops.

This replaced an earlier jsonl+.npy design: that scheme rewrote the entire
jsonl file on every single delete() call (O(n) per delete, and a partial
write mid-rewrite could corrupt the whole file), and had no way to query
without loading everything into memory. SQLite gives real per-row deletes,
crash-safe transactions, and a path to filtering by metadata/time later
without another storage migration.
"""

import json
import os
import sqlite3
import threading

import numpy as np
import sqlite_vec

from skull.config import MEMORY_DIR

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2's fixed output size - the vec0 table needs this at creation time

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Once the model is cached locally, skip the Hugging Face Hub
                # metadata check entirely - it's a slow, noisy network round
                # trip on every process start for a model that never changes.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed(texts: list) -> np.ndarray:
    """Return L2-normalized embeddings, shape (len(texts), dim)."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def _connect(db_path) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("pragma journal_mode=WAL")
    db.execute(
        "create table if not exists entries ("
        "id integer primary key autoincrement, "
        "text text not null, "
        "metadata text not null, "
        "created_at text not null default (datetime('now'))"
        ")"
    )
    db.execute(
        f"create virtual table if not exists vec_items using vec0(embedding float[{EMBED_DIM}])"
    )
    db.commit()
    return db


class VectorStore:
    def __init__(self, name: str):
        self.name = name
        MEMORY_DIR.mkdir(exist_ok=True)
        self.db_path = MEMORY_DIR / f"{name}.db"
        self._db = _connect(self.db_path)

    def add(self, text: str, metadata: dict | None = None) -> dict:
        if not text or not text.strip():
            return {"error": "empty text, nothing stored"}

        vector = embed([text])[0]  # shape (dim,)
        with self._db:
            cur = self._db.execute(
                "insert into entries (text, metadata) values (?, ?)",
                (text, json.dumps(metadata or {})),
            )
            entry_id = cur.lastrowid
            self._db.execute(
                "insert into vec_items (rowid, embedding) values (?, ?)",
                (entry_id, vector.tobytes()),
            )

        count = self._db.execute("select count(*) from entries").fetchone()[0]
        return {"status": "stored", "count": count}

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list:
        count = self._db.execute("select count(*) from entries").fetchone()[0]
        if count == 0:
            return []

        query_vector = embed([query])[0]
        # sqlite-vec's `distance` for vec0 is squared L2. Vectors are unit
        # -normalized, so cosine similarity = 1 - (squared_L2 / 2) exactly.
        rows = self._db.execute(
            """
            select entries.text, entries.metadata, vec_items.distance
            from vec_items
            join entries on entries.id = vec_items.rowid
            where vec_items.embedding match ? and k = ?
            order by vec_items.distance
            """,
            (query_vector.tobytes(), min(k, count)),
        ).fetchall()

        results = []
        for text, metadata_json, distance in rows:
            score = 1.0 - (distance / 2.0)
            if score < min_score:
                continue
            results.append({"text": text, "metadata": json.loads(metadata_json), "score": score})
        return results

    def count(self) -> int:
        return self._db.execute("select count(*) from entries").fetchone()[0]

    def all(self) -> list:
        rows = self._db.execute("select text, metadata from entries order by id").fetchall()
        return [{"text": text, "metadata": json.loads(metadata_json)} for text, metadata_json in rows]

    def delete(self, text: str) -> dict:
        """Remove the entry whose text exactly matches `text`. Exact match
        (rather than an index) so callers can reference a fact by quoting it
        back, e.g. after finding it via search() - positions aren't stable
        identifiers once other entries are added or removed."""
        row = self._db.execute("select id from entries where text = ? limit 1", (text,)).fetchone()
        if row is None:
            return {"error": "no matching entry found", "deleted": False}

        entry_id = row[0]
        with self._db:
            self._db.execute("delete from entries where id = ?", (entry_id,))
            self._db.execute("delete from vec_items where rowid = ?", (entry_id,))

        remaining = self._db.execute("select count(*) from entries").fetchone()[0]
        return {"status": "deleted", "deleted": True, "remaining": remaining}


_stores = {}
_stores_lock = threading.Lock()


def get_store(name: str) -> VectorStore:
    with _stores_lock:
        if name not in _stores:
            _stores[name] = VectorStore(name)
        return _stores[name]


conversations = lambda: get_store("conversations")
persona = lambda: get_store("persona")
