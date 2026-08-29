"""Tests for storage/store.py - the local vector store (persistence,
search ranking, delete-by-exact-text, and reload-from-disk), using the
isolated_memory_dir fixture's fake deterministic embedder instead of a real
network-downloaded model."""

import numpy as np

from skull.storage import store as mem


def test_add_persists_entry_and_returns_count(isolated_memory_dir):
    store = mem.get_store("persona")
    result = store.add("likes tea", {"category": "preference"})
    assert result == {"status": "stored", "count": 1}
    assert store.count() == 1
    assert store.all() == [{"text": "likes tea", "metadata": {"category": "preference"}}]


def test_add_rejects_empty_text(isolated_memory_dir):
    store = mem.get_store("persona")
    result = store.add("   ")
    assert "error" in result
    assert store.count() == 0


def test_add_defaults_metadata_to_empty_dict(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("some fact")
    assert store.all()[0]["metadata"] == {}


def test_search_returns_exact_match_with_top_score(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("likes tea")
    store.add("likes coffee")
    store.add("works as an engineer")

    results = store.search("likes tea", k=5)
    assert results
    # The fake embedder is deterministic per exact string, so searching for
    # the exact stored text must rank itself first with a perfect score.
    assert results[0]["text"] == "likes tea"
    assert results[0]["score"] > 0.99


def test_search_respects_k_limit(isolated_memory_dir):
    store = mem.get_store("persona")
    for i in range(10):
        store.add(f"fact number {i}")
    results = store.search("fact number 0", k=3)
    assert len(results) == 3


def test_search_empty_store_returns_empty_list(isolated_memory_dir):
    store = mem.get_store("persona")
    assert store.search("anything") == []


def test_search_respects_min_score_threshold(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("likes tea")
    # An absurdly high min_score should filter out everything, even a
    # perfect self-match's near-1.0 score.
    results = store.search("likes tea", min_score=1.5)
    assert results == []


def test_delete_removes_exact_match(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("likes tea")
    store.add("likes coffee")

    result = store.delete("likes tea")
    assert result == {"status": "deleted", "deleted": True, "remaining": 1}
    assert store.count() == 1
    assert store.all() == [{"text": "likes coffee", "metadata": {}}]


def test_delete_missing_entry_returns_error(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("likes tea")
    result = store.delete("does not exist")
    assert result == {"error": "no matching entry found", "deleted": False}
    assert store.count() == 1


def test_delete_last_entry_clears_vectors(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("only entry")
    store.delete("only entry")
    assert store.count() == 0
    assert store._vectors is None
    # Searching an emptied store must not crash on a None vectors matrix.
    assert store.search("anything") == []


def test_store_reloads_from_disk_across_instances(isolated_memory_dir):
    """A fresh VectorStore instance pointed at the same directory must see
    previously persisted entries - this is what makes memory durable across
    process restarts."""
    store1 = mem.get_store("persona")
    store1.add("persisted fact")

    store2 = mem.VectorStore("persona")
    assert store2.count() == 1
    assert store2.all() == [{"text": "persisted fact", "metadata": {}}]


def test_store_rebuilds_vectors_when_npy_missing(isolated_memory_dir):
    """If the .jsonl exists but the .npy cache is missing/stale, vectors
    must be recomputed from the text rather than left as None."""
    store1 = mem.get_store("persona")
    store1.add("fact one")
    store1.add("fact two")

    store1.npy_path.unlink()

    store2 = mem.VectorStore("persona")
    assert store2.count() == 2
    assert store2._vectors is not None
    assert store2._vectors.shape[0] == 2


def test_conversations_and_persona_are_separate_stores(isolated_memory_dir):
    mem.persona().add("a persona fact")
    mem.conversations().add("a conversation turn")

    assert mem.persona().count() == 1
    assert mem.conversations().count() == 1
    assert mem.persona().all()[0]["text"] == "a persona fact"
    assert mem.conversations().all()[0]["text"] == "a conversation turn"
