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


def test_delete_last_entry_leaves_store_searchable(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("only entry")
    store.delete("only entry")
    assert store.count() == 0
    # Searching an emptied store must not crash (no rows left in either table).
    assert store.search("anything") == []


def test_delete_removes_matching_vec_row_not_just_entries_row(isolated_memory_dir):
    """A prior bug shape: deleting only from `entries` while leaving the
    vec_items row behind would silently corrupt future searches (a
    still-indexed vector for text that no longer exists). Deleting a middle
    entry and re-adding must not resurrect the deleted vector in search."""
    store = mem.get_store("persona")
    store.add("keep me")
    store.add("delete me")
    store.delete("delete me")

    results = store.search("delete me", k=5)
    assert all(r["text"] != "delete me" for r in results)


def test_store_reloads_from_disk_across_instances(isolated_memory_dir):
    """A fresh VectorStore instance pointed at the same directory must see
    previously persisted entries - this is what makes memory durable across
    process restarts."""
    store1 = mem.get_store("persona")
    store1.add("persisted fact")

    store2 = mem.VectorStore("persona")
    assert store2.count() == 1
    assert store2.all() == [{"text": "persisted fact", "metadata": {}}]


def test_store_persists_vectors_searchable_across_instances(isolated_memory_dir):
    """A fresh VectorStore pointed at the same db file must be able to
    search previously stored entries, not just list them - i.e. the
    embeddings themselves (not just the text) survive a reopen."""
    store1 = mem.get_store("persona")
    store1.add("fact one")
    store1.add("fact two")

    store2 = mem.VectorStore("persona")
    assert store2.count() == 2
    results = store2.search("fact one", k=5)
    assert results[0]["text"] == "fact one"
    assert results[0]["score"] > 0.99


def test_conversations_and_persona_are_separate_stores(isolated_memory_dir):
    mem.persona().add("a persona fact")
    mem.conversations().add("a conversation turn")

    assert mem.persona().count() == 1
    assert mem.conversations().count() == 1
    assert mem.persona().all()[0]["text"] == "a persona fact"
    assert mem.conversations().all()[0]["text"] == "a conversation turn"
