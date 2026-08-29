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


def test_connect_migrates_pre_existing_db_missing_superseded_by_column(isolated_memory_dir):
    """Real bug hit during development: an existing persona.db/conversations.db
    created before superseded_by existed has no such column - CREATE TABLE IF
    NOT EXISTS doesn't add columns to an already-existing table, so opening
    an old database without a migration crashes every call that references
    the column. Simulate an old-schema database and confirm _connect()
    upgrades it in place instead of failing."""
    import sqlite3

    old_db_path = mem.MEMORY_DIR / "legacy.db"
    mem.MEMORY_DIR.mkdir(exist_ok=True)
    raw = sqlite3.connect(old_db_path)
    raw.execute(
        "create table entries ("
        "id integer primary key autoincrement, "
        "text text not null, "
        "metadata text not null, "
        "created_at text not null default (datetime('now'))"
        ")"
    )
    raw.execute("insert into entries (text, metadata) values ('legacy fact', '{}')")
    raw.commit()
    raw.close()

    store = mem.VectorStore("legacy")
    assert store.all() == [{"text": "legacy fact", "metadata": {}}]
    columns = {row[1] for row in store._db.execute("pragma table_info(entries)").fetchall()}
    assert "superseded_by" in columns


def test_mark_superseded_excludes_old_fact_from_search_and_all(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("prefers terse answers")
    store.add("prefers detailed explanations")

    result = store.mark_superseded("prefers terse answers", "prefers detailed explanations")
    assert result == {"status": "superseded", "superseded": True}

    assert store.count() == 1
    assert store.all() == [{"text": "prefers detailed explanations", "metadata": {}}]
    results = store.search("prefers terse answers", k=5)
    assert all(r["text"] != "prefers terse answers" for r in results)


def test_mark_superseded_missing_fact_returns_error(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("only fact")
    result = store.mark_superseded("nonexistent", "only fact")
    assert result == {"error": "one or both facts not found", "superseded": False}


def test_history_includes_superseded_facts(isolated_memory_dir):
    store = mem.get_store("persona")
    store.add("old fact")
    store.add("new fact")
    store.mark_superseded("old fact", "new fact")

    history = store.history()
    assert len(history) == 2
    old_entry = next(e for e in history if e["text"] == "old fact")
    new_entry = next(e for e in history if e["text"] == "new fact")
    assert old_entry["superseded_by"] == new_entry["id"]
    assert new_entry["superseded_by"] is None


def test_conversations_and_persona_are_separate_stores(isolated_memory_dir):
    mem.persona().add("a persona fact")
    mem.conversations().add("a conversation turn")

    assert mem.persona().count() == 1
    assert mem.conversations().count() == 1
    assert mem.persona().all()[0]["text"] == "a persona fact"
    assert mem.conversations().all()[0]["text"] == "a conversation turn"
