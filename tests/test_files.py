"""Tests for tools/files.py - local filesystem access, particularly the
max_chars hard ceiling that exists because a single oversized tool result
can blow the context window in one shot (compaction can't undo that after
the result is already read and about to be sent)."""

from skull.tools import files


def test_read_file_returns_content(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hello world")
    result = files.read_file(str(p))
    assert result == {"path": str(p), "content": "hello world", "truncated": False}


def test_read_file_missing_returns_error(tmp_path):
    result = files.read_file(str(tmp_path / "nonexistent.txt"))
    assert "error" in result


def test_read_file_directory_returns_error(tmp_path):
    result = files.read_file(str(tmp_path))
    assert "error" in result
    assert "directory" in result["error"]


def test_read_file_truncates_at_requested_max_chars(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("x" * 1000)
    result = files.read_file(str(p), max_chars=500)
    assert len(result["content"]) == 500
    assert result["truncated"] is True


def test_read_file_max_chars_has_a_floor(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("x" * 1000)
    result = files.read_file(str(p), max_chars=1)
    assert len(result["content"]) == 200  # floor, not the requested 1


def test_read_file_ignores_max_chars_above_hard_ceiling(tmp_path):
    """The exact bug this guards against: a model requesting an enormous
    max_chars (e.g. 200000, to smuggle a whole binary file through as text)
    must not get anywhere close to that - the ceiling exists specifically so
    a single tool result can never approach the context window on its own."""
    p = tmp_path / "huge.txt"
    p.write_text("x" * (files.MAX_READ_CHARS_CEILING * 3))
    result = files.read_file(str(p), max_chars=200000)
    assert len(result["content"]) == files.MAX_READ_CHARS_CEILING
    assert result["truncated"] is True


def test_read_file_default_max_chars_unaffected_by_ceiling_change(tmp_path):
    p = tmp_path / "medium.txt"
    p.write_text("x" * (files.MAX_READ_CHARS + 5000))
    result = files.read_file(str(p))
    assert len(result["content"]) == files.MAX_READ_CHARS
    assert result["truncated"] is True


def test_read_file_auto_extracts_docx_content(tmp_path):
    import docx

    doc = docx.Document()
    doc.add_paragraph("Extracted paragraph text.")
    p = tmp_path / "doc.docx"
    doc.save(str(p))

    result = files.read_file(str(p))
    assert result["extracted"] is True
    assert "Extracted paragraph text." in result["content"]


def test_read_file_extraction_failure_returns_error(tmp_path):
    """A file with a document extension but corrupt/non-document content
    must return a clean error, not crash the tool call."""
    p = tmp_path / "fake.docx"
    p.write_bytes(b"not actually a docx")
    result = files.read_file(str(p))
    assert "error" in result


def test_list_directory_lists_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    result = files.list_directory(str(tmp_path))
    names = {e["name"]: e["type"] for e in result["entries"]}
    assert names == {"a.txt": "file", "subdir": "directory"}


def test_list_directory_missing_path_returns_error(tmp_path):
    result = files.list_directory(str(tmp_path / "nonexistent"))
    assert "error" in result


def test_list_directory_rejects_a_file_path(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("")
    result = files.list_directory(str(p))
    assert "error" in result


def test_write_file_creates_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "ask_permission", lambda *a, **k: True)
    p = tmp_path / "new.txt"
    result = files.write_file(str(p), "hello")
    assert result == {"status": "written", "path": str(p), "mode": "overwrite", "bytes": 5}
    assert p.read_text() == "hello"


def test_write_file_denied_by_user_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "ask_permission", lambda *a, **k: False)
    p = tmp_path / "new.txt"
    result = files.write_file(str(p), "hello")
    assert result == {"error": "denied by user", "denied": True}
    assert not p.exists()


def test_write_file_append_mode_adds_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "ask_permission", lambda *a, **k: True)
    p = tmp_path / "log.txt"
    p.write_text("line1\n")
    files.write_file(str(p), "line2\n", mode="append")
    assert p.read_text() == "line1\nline2\n"


def test_write_file_rejects_invalid_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "ask_permission", lambda *a, **k: True)
    result = files.write_file(str(tmp_path / "x.txt"), "content", mode="bogus")
    assert "error" in result


def test_write_file_rejects_writing_over_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "ask_permission", lambda *a, **k: True)
    result = files.write_file(str(tmp_path), "content")
    assert "error" in result
