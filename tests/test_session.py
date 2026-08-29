"""Tests for core/session.py's handle_turn - specifically its error
recovery around stream_chat, since an uncaught exception there used to
propagate all the way out of the REPL loop (which doesn't wrap handle_turn
at all) and crash the whole process, losing the conversation.

stream_chat itself is mocked here (not the HTTP layer) so these tests
focus purely on handle_turn's own exception handling and message-rollback
behavior, independent of client.py's parsing logic (covered in
test_client.py)."""

import requests

from skull.core import session as session_module
from skull.core.client import StreamParseError
from skull.core.session import Session


def _make_session(monkeypatch):
    monkeypatch.setattr(session_module, "load_system_prompt", lambda: "system prompt")
    return Session()


def test_handle_turn_recovers_from_stream_parse_error_without_crashing(
    isolated_skills_dir, isolated_memory_dir, monkeypatch
):
    """The exact gap found via code review: a malformed streamed chunk
    (StreamParseError) must be caught by handle_turn and reported cleanly,
    not propagate and crash the process."""
    session = _make_session(monkeypatch)

    def fake_stream_chat(*args, **kwargs):
        raise StreamParseError("malformed chunk from the model endpoint mid-stream (KeyError: 'choices')")

    monkeypatch.setattr(session_module, "stream_chat", fake_stream_chat)

    ok = session.handle_turn("hello")
    assert ok is False


def test_handle_turn_rolls_back_messages_on_stream_parse_error(
    isolated_skills_dir, isolated_memory_dir, monkeypatch
):
    session = _make_session(monkeypatch)
    checkpoint_len = len(session.messages)

    monkeypatch.setattr(
        session_module,
        "stream_chat",
        lambda *a, **k: (_ for _ in ()).throw(StreamParseError("bad chunk")),
    )

    session.handle_turn("hello")
    # The failed turn's user message must be rolled back - a stream parse
    # failure must not poison every subsequent request with a half-added turn.
    assert len(session.messages) == checkpoint_len


def test_handle_turn_still_recovers_from_http_error(isolated_skills_dir, isolated_memory_dir, monkeypatch):
    """Confirms the pre-existing requests.HTTPError handling still works
    after adding the new except clause - it must not have been shadowed or
    broken by the new StreamParseError branch."""
    session = _make_session(monkeypatch)

    class FakeResponse:
        text = "server error body"

    def fake_stream_chat(*args, **kwargs):
        raise requests.HTTPError("500 Server Error", response=FakeResponse())

    monkeypatch.setattr(session_module, "stream_chat", fake_stream_chat)

    ok = session.handle_turn("hello")
    assert ok is False


def test_handle_turn_succeeds_on_well_formed_response(isolated_skills_dir, isolated_memory_dir, monkeypatch):
    session = _make_session(monkeypatch)

    monkeypatch.setattr(session_module, "stream_chat", lambda *a, **k: ("a reply", None, "stop"))

    ok = session.handle_turn("hello")
    assert ok is True
    assert session.messages[-1] == {"role": "assistant", "content": "a reply"}
