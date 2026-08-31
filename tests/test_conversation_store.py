"""Tests for core/conversation_store.py - per-directory conversation
persistence, keyed by the resolved absolute cwd path."""

from skull.core import conversation_store as cs


def test_load_returns_none_when_nothing_saved(isolated_conversations_dir, tmp_path):
    assert cs.load(str(tmp_path)) is None


def test_save_then_load_round_trips_messages(isolated_conversations_dir, tmp_path):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    cs.save(str(tmp_path), messages)
    assert cs.load(str(tmp_path)) == messages


def test_different_directories_get_different_saves(isolated_conversations_dir, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    cs.save(str(dir_a), [{"role": "system", "content": "a"}])
    cs.save(str(dir_b), [{"role": "system", "content": "b"}])

    assert cs.load(str(dir_a)) == [{"role": "system", "content": "a"}]
    assert cs.load(str(dir_b)) == [{"role": "system", "content": "b"}]


def test_unresolved_relative_paths_resolve_to_the_same_key(isolated_conversations_dir, tmp_path, monkeypatch):
    """A relative path and its resolved absolute equivalent must hit the
    same saved conversation - real usage always calls os.getcwd() (already
    absolute), but this guards against any caller passing a relative path."""
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(tmp_path)

    cs.save("sub", [{"role": "system", "content": "x"}])
    assert cs.load(str(sub)) == [{"role": "system", "content": "x"}]


def test_clear_removes_the_saved_file(isolated_conversations_dir, tmp_path):
    cs.save(str(tmp_path), [{"role": "system", "content": "x"}])
    cs.clear(str(tmp_path))
    assert cs.load(str(tmp_path)) is None


def test_clear_on_nonexistent_save_does_not_raise(isolated_conversations_dir, tmp_path):
    cs.clear(str(tmp_path))  # must not raise even though nothing was ever saved


def test_load_returns_none_for_corrupt_file(isolated_conversations_dir, tmp_path):
    path = cs._path_for(str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json{{{")
    assert cs.load(str(tmp_path)) is None


def test_load_returns_none_when_messages_key_missing(isolated_conversations_dir, tmp_path):
    import json

    path = cs._path_for(str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cwd": str(tmp_path)}))
    assert cs.load(str(tmp_path)) is None


def test_save_creates_conversations_dir_if_missing(isolated_conversations_dir, tmp_path):
    assert not isolated_conversations_dir.exists()
    cs.save(str(tmp_path), [{"role": "system", "content": "x"}])
    assert isolated_conversations_dir.exists()
