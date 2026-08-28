"""Local vector store for conversation history and persona/knowledge facts.

Embeddings are computed locally (sentence-transformers, all-MiniLM-L6-v2) so
this works fully offline with no dependency on the Qwen endpoint (which does
not expose an embeddings API).

Storage layout (memory/):
    conversations.jsonl / conversations.npy  - every auto-logged chat turn
    persona.jsonl        / persona.npy       - curated facts about the user
                                                (identity, preferences, behavior)

Each store is a flat numpy matrix of normalized embeddings + a parallel
JSONL file of {text, metadata}. Cosine similarity via dot product since
vectors are unit-normalized. This is intentionally simple - fine for up to
tens of thousands of entries, which is far more than a personal assistant's
history will hit.
"""

import json
import os
import threading
from pathlib import Path

import numpy as np

MEMORY_DIR = Path(__file__).parent / "memory"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

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


class VectorStore:
    def __init__(self, name: str):
        self.name = name
        self.jsonl_path = MEMORY_DIR / f"{name}.jsonl"
        self.npy_path = MEMORY_DIR / f"{name}.npy"
        self._entries = []       # list of {"text": str, "metadata": dict}
        self._vectors = None     # np.ndarray, shape (n, dim), or None if empty
        self._load()

    def _load(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        if self.jsonl_path.exists():
            with self.jsonl_path.open() as f:
                self._entries = [json.loads(line) for line in f if line.strip()]
        if self.npy_path.exists() and self._entries:
            self._vectors = np.load(self.npy_path)
        elif self._entries:
            # jsonl exists but npy missing/stale - rebuild
            self._vectors = embed([e["text"] for e in self._entries])
            self._save_vectors()

    def _save_vectors(self):
        if self._vectors is not None:
            np.save(self.npy_path, self._vectors)

    def _append_jsonl(self, entry: dict):
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def add(self, text: str, metadata: dict | None = None) -> dict:
        if not text or not text.strip():
            return {"error": "empty text, nothing stored"}

        entry = {"text": text, "metadata": metadata or {}}
        vector = embed([text])  # shape (1, dim)

        self._entries.append(entry)
        if self._vectors is None:
            self._vectors = vector
        else:
            self._vectors = np.vstack([self._vectors, vector])

        self._append_jsonl(entry)
        self._save_vectors()
        return {"status": "stored", "count": len(self._entries)}

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list:
        if not self._entries or self._vectors is None:
            return []
        query_vec = embed([query])[0]  # shape (dim,)
        scores = self._vectors @ query_vec  # cosine similarity (both normalized)
        top_idx = np.argsort(-scores)[:k]
        results = []
        for i in top_idx:
            score = float(scores[i])
            if score < min_score:
                continue
            entry = self._entries[i]
            results.append({"text": entry["text"], "metadata": entry["metadata"], "score": score})
        return results

    def count(self) -> int:
        return len(self._entries)

    def all(self) -> list:
        return list(self._entries)


_stores = {}
_stores_lock = threading.Lock()


def get_store(name: str) -> VectorStore:
    with _stores_lock:
        if name not in _stores:
            _stores[name] = VectorStore(name)
        return _stores[name]


conversations = lambda: get_store("conversations")
persona = lambda: get_store("persona")
