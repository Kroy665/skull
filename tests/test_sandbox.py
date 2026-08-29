"""Tests for tools/sandbox.py - the E2B sandbox bridge, particularly
download_from_sandbox (which exists so binary files never need to be
smuggled through sandbox_read_file as base64 text - the exact failure mode
that once blew the context window in one shot) and the max_chars hard
ceiling on sandbox_read_file itself.

No real E2B or network calls: _get_sandbox and requests.get are always
monkeypatched to fakes.
"""

import pytest
import requests

from skull.tools import sandbox


class FakeSandbox:
    def __init__(self, files_content=None, download_urls=None):
        self._files_content = files_content or {}
        self._download_urls = download_urls or {}

    class files:
        pass

    def download_url(self, path):
        if path not in self._download_urls:
            raise RuntimeError(f"no such file: {path}")
        return self._download_urls[path]


def _fake_sandbox_with_files(files_content=None, download_urls=None):
    sbx = FakeSandbox(files_content, download_urls)

    class _Files:
        def read(self_inner, path, format="text"):
            content = files_content[path]
            if format == "bytes":
                return bytearray(content if isinstance(content, bytes) else content.encode())
            return content

    sbx.files = _Files()
    return sbx


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# sandbox_read_file - hard ceiling
# ---------------------------------------------------------------------------

def test_sandbox_read_file_ignores_max_chars_above_hard_ceiling(monkeypatch):
    """The exact bug this guards against: a model requesting max_chars=200000
    to read a base64-encoded binary file must not get anywhere close to
    that - a single tool result that large can blow the context window."""
    huge_text = "x" * (sandbox.MAX_FILE_READ_CHARS_CEILING * 3)
    sbx = _fake_sandbox_with_files(files_content={"/tmp/big.txt": huge_text})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)

    result = sandbox.sandbox_read_file("/tmp/big.txt", max_chars=200000)
    assert len(result["content"]) == sandbox.MAX_FILE_READ_CHARS_CEILING
    assert result["truncated"] is True


def test_sandbox_read_file_default_max_chars_unaffected(monkeypatch):
    text = "x" * (sandbox.MAX_FILE_READ_CHARS + 5000)
    sbx = _fake_sandbox_with_files(files_content={"/tmp/f.txt": text})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)

    result = sandbox.sandbox_read_file("/tmp/f.txt")
    assert len(result["content"]) == sandbox.MAX_FILE_READ_CHARS
    assert result["truncated"] is True


def test_sandbox_read_file_missing_path_returns_error():
    assert "error" in sandbox.sandbox_read_file("")


def test_sandbox_read_file_auto_extracts_docx_content(monkeypatch):
    import io

    import docx

    doc = docx.Document()
    doc.add_paragraph("Sandbox-generated paragraph.")
    buf = io.BytesIO()
    doc.save(buf)

    sbx = _fake_sandbox_with_files(files_content={"/tmp/out.docx": buf.getvalue()})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)

    result = sandbox.sandbox_read_file("/tmp/out.docx")
    assert result["extracted"] is True
    assert "Sandbox-generated paragraph." in result["content"]


def test_sandbox_read_file_extraction_failure_returns_error(monkeypatch):
    sbx = _fake_sandbox_with_files(files_content={"/tmp/fake.docx": b"not a real docx"})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)

    result = sandbox.sandbox_read_file("/tmp/fake.docx")
    assert "error" in result


# ---------------------------------------------------------------------------
# download_from_sandbox
# ---------------------------------------------------------------------------

def test_download_from_sandbox_denied_by_user_does_not_write(tmp_path, monkeypatch):
    sbx = FakeSandbox(download_urls={"/tmp/f.docx": "https://example.com/f.docx"})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)
    monkeypatch.setattr(sandbox, "ask_permission", lambda *a, **k: False)

    dest = tmp_path / "out.docx"
    result = sandbox.download_from_sandbox("/tmp/f.docx", str(dest))
    assert result == {"error": "denied by user", "denied": True}
    assert not dest.exists()


def test_download_from_sandbox_writes_raw_bytes(tmp_path, monkeypatch):
    binary_content = b"\x50\x4b\x03\x04not really a docx but binary-safe"
    sbx = FakeSandbox(download_urls={"/tmp/f.docx": "https://example.com/f.docx"})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)
    monkeypatch.setattr(sandbox, "ask_permission", lambda *a, **k: True)
    monkeypatch.setattr(sandbox.requests, "get", lambda url, timeout: FakeResponse(binary_content))

    dest = tmp_path / "out.docx"
    result = sandbox.download_from_sandbox("/tmp/f.docx", str(dest), reason="deliver generated cv")
    assert result == {"status": "downloaded", "local_path": str(dest), "bytes": len(binary_content)}
    assert dest.read_bytes() == binary_content


def test_download_from_sandbox_creates_parent_directories(tmp_path, monkeypatch):
    sbx = FakeSandbox(download_urls={"/tmp/f.txt": "https://example.com/f.txt"})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)
    monkeypatch.setattr(sandbox, "ask_permission", lambda *a, **k: True)
    monkeypatch.setattr(sandbox.requests, "get", lambda url, timeout: FakeResponse(b"data"))

    dest = tmp_path / "nested" / "dir" / "out.txt"
    result = sandbox.download_from_sandbox("/tmp/f.txt", str(dest))
    assert result["status"] == "downloaded"
    assert dest.read_bytes() == b"data"


def test_download_from_sandbox_rejects_local_path_that_is_a_directory(tmp_path, monkeypatch):
    sbx = FakeSandbox(download_urls={"/tmp/f.txt": "https://example.com/f.txt"})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)

    result = sandbox.download_from_sandbox("/tmp/f.txt", str(tmp_path))
    assert "error" in result


def test_download_from_sandbox_missing_sandbox_path_returns_error(tmp_path):
    result = sandbox.download_from_sandbox("", str(tmp_path / "out.txt"))
    assert "error" in result


def test_download_from_sandbox_missing_local_path_returns_error():
    result = sandbox.download_from_sandbox("/tmp/f.txt", "")
    assert "error" in result


def test_download_from_sandbox_propagates_download_url_failure(monkeypatch, tmp_path):
    sbx = FakeSandbox(download_urls={})  # download_url() will raise for any path
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)

    result = sandbox.download_from_sandbox("/tmp/nonexistent.txt", str(tmp_path / "out.txt"))
    assert "error" in result


def test_download_from_sandbox_propagates_http_failure(monkeypatch, tmp_path):
    sbx = FakeSandbox(download_urls={"/tmp/f.txt": "https://example.com/f.txt"})
    monkeypatch.setattr(sandbox, "_get_sandbox", lambda: sbx)
    monkeypatch.setattr(sandbox, "ask_permission", lambda *a, **k: True)

    def failing_get(url, timeout):
        return FakeResponse(b"", status_code=404)

    monkeypatch.setattr(sandbox.requests, "get", failing_get)

    result = sandbox.download_from_sandbox("/tmp/f.txt", str(tmp_path / "out.txt"))
    assert "error" in result
    assert not (tmp_path / "out.txt").exists()
