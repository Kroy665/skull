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
# sentence-transformers resolves the short name above to this full HF repo
# id internally - the cache (and try_to_load_from_cache below) is keyed by
# the full id, not the short name, so this has to match exactly or the
# cache check always misses.
_EMBED_MODEL_HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2's fixed output size - the vec0 table needs this at creation time

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from huggingface_hub import try_to_load_from_cache
                from sentence_transformers import SentenceTransformer

                # Once the model is cached locally, skip the Hugging Face Hub
                # metadata check entirely - it's a slow, noisy network round
                # trip on every process start for a model that never changes.
                # Only go offline once the cache actually has it, though -
                # forcing this unconditionally broke the very first run on a
                # fresh install (no cache yet), turning "no internet to
                # config.json" into an uncaught crash instead of a real
                # download attempt.
                if try_to_load_from_cache(_EMBED_MODEL_HF_REPO, "config.json"):
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
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
        "created_at text not null default (datetime('now')), "
        "superseded_by integer references entries(id)"
        ")"
    )
    # Migration for databases created before superseded_by existed (CREATE
    # TABLE IF NOT EXISTS doesn't add columns to an already-existing table).
    existing_columns = {row[1] for row in db.execute("pragma table_info(entries)").fetchall()}
    if "superseded_by" not in existing_columns:
        db.execute("alter table entries add column superseded_by integer references entries(id)")
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
        count = self._db.execute("select count(*) from entries where superseded_by is null").fetchone()[0]
        if count == 0:
            return []

        query_vector = embed([query])[0]
        # sqlite-vec's `distance` for vec0 is squared L2. Vectors are unit
        # -normalized, so cosine similarity = 1 - (squared_L2 / 2) exactly.
        # Over-fetch past k since some hits may be superseded and filtered
        # out below - k=count guarantees enough candidates regardless of how
        # many of the top matches turn out to be stale.
        rows = self._db.execute(
            """
            select entries.text, entries.metadata, vec_items.distance
            from vec_items
            join entries on entries.id = vec_items.rowid
            where vec_items.embedding match ? and k = ? and entries.superseded_by is null
            order by vec_items.distance
            """,
            (query_vector.tobytes(), count),
        ).fetchall()

        results = []
        for text, metadata_json, distance in rows[:k]:
            score = 1.0 - (distance / 2.0)
            if score < min_score:
                continue
            results.append({"text": text, "metadata": json.loads(metadata_json), "score": score})
        return results

    def count(self) -> int:
        return self._db.execute("select count(*) from entries where superseded_by is null").fetchone()[0]

    def all(self) -> list:
        rows = self._db.execute(
            "select text, metadata from entries where superseded_by is null order by id"
        ).fetchall()
        return [{"text": text, "metadata": json.loads(metadata_json)} for text, metadata_json in rows]

    def history(self) -> list:
        """Every entry including superseded ones, oldest first - for
        auditing what got superseded and by what, or manually undoing a
        wrong supersede decision."""
        rows = self._db.execute(
            "select id, text, metadata, superseded_by from entries order by id"
        ).fetchall()
        return [
            {"id": id_, "text": text, "metadata": json.loads(metadata_json), "superseded_by": superseded_by}
            for id_, text, metadata_json, superseded_by in rows
        ]

    def _find_live_id(self, text: str) -> int | None:
        """Find the id of the LIVE (not already superseded) entry matching
        `text` exactly, preferring the most recently added if more than one
        live row somehow has the same text (e.g. the model re-storing an
        identical phrasing in two different sessions - text has no
        uniqueness constraint). Real bug this guards against: without the
        superseded_by filter and an explicit ORDER BY, an unordered `LIMIT 1`
        could match an already-dead duplicate instead of the current one,
        silently re-marking the wrong row while still reporting success."""
        row = self._db.execute(
            "select id from entries where text = ? and superseded_by is null order by id desc limit 1",
            (text,),
        ).fetchone()
        return row[0] if row else None

    def mark_superseded(self, old_text: str, new_text: str) -> dict:
        """Mark the entry matching `old_text` as superseded by the entry
        matching `new_text` - it stays in the database (visible via
        history()) but is excluded from search()/all()/count() from now on."""
        old_id = self._find_live_id(old_text)
        new_id = self._find_live_id(new_text)
        if old_id is None or new_id is None:
            return {"error": "one or both facts not found", "superseded": False}

        with self._db:
            self._db.execute("update entries set superseded_by = ? where id = ?", (new_id, old_id))
        return {"status": "superseded", "superseded": True}

    def delete(self, text: str) -> dict:
        """Remove the LIVE entry whose text exactly matches `text`. Exact
        match (rather than an index) so callers can reference a fact by
        quoting it back, e.g. after finding it via search() - positions
        aren't stable identifiers once other entries are added or removed.
        Only ever targets a live (non-superseded) row - an already-dead
        duplicate is not a valid target even if its text also matches."""
        entry_id = self._find_live_id(text)
        if entry_id is None:
            return {"error": "no matching entry found", "deleted": False}

        with self._db:
            self._db.execute("delete from entries where id = ?", (entry_id,))
            self._db.execute("delete from vec_items where rowid = ?", (entry_id,))

        remaining = self.count()
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
