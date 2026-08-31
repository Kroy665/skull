"""Tests for core/client.py - the streaming chat-completion call, focused
on the real gap found via code review: a malformed/truncated JSON chunk (or
one missing the expected choices/delta shape) mid-stream had no error
handling at all, so it would propagate as a raw JSONDecodeError/KeyError
all the way out of handle_turn (whose except clauses only catch
requests.RequestException) and out of the REPL loop (which doesn't wrap
handle_turn at all), crashing the whole process and losing the
conversation."""

import pytest

from skull.core import client
from skull.core.client import StreamParseError, stream_chat


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8") if isinstance(line, str) else line


def _sse(payload: str) -> str:
    return f"data: {payload}"


def test_stream_chat_returns_content_and_tool_calls_on_well_formed_stream(monkeypatch):
    lines = [
        _sse('{"choices": [{"delta": {"content": "Hello"}}]}'),
        _sse('{"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}'),
        _sse("[DONE]"),
    ]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    content, tool_calls, finish_reason = stream_chat([{"role": "user", "content": "hi"}], tools=[])
    assert content == "Hello world"
    assert tool_calls is None
    assert finish_reason == "stop"


def test_stream_chat_raises_stream_parse_error_on_malformed_json(monkeypatch):
    lines = [_sse('{"choices": [{"delta": {"content": "partial'), _sse("[DONE]")]  # truncated JSON
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    with pytest.raises(StreamParseError, match="JSONDecodeError"):
        stream_chat([{"role": "user", "content": "hi"}], tools=[])


def test_stream_chat_raises_stream_parse_error_on_missing_choices_key(monkeypatch):
    lines = [_sse('{"unexpected": "shape"}'), _sse("[DONE]")]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    with pytest.raises(StreamParseError, match="KeyError"):
        stream_chat([{"role": "user", "content": "hi"}], tools=[])


def test_stream_chat_raises_stream_parse_error_on_empty_choices_list(monkeypatch):
    lines = [_sse('{"choices": []}'), _sse("[DONE]")]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    with pytest.raises(StreamParseError, match="IndexError"):
        stream_chat([{"role": "user", "content": "hi"}], tools=[])


def test_stream_chat_raises_stream_parse_error_on_missing_delta_key(monkeypatch):
    lines = [_sse('{"choices": [{"no_delta_here": true}]}'), _sse("[DONE]")]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    with pytest.raises(StreamParseError, match="KeyError"):
        stream_chat([{"role": "user", "content": "hi"}], tools=[])


def test_stream_chat_error_message_includes_raw_line_for_debugging(monkeypatch):
    lines = [_sse('{"broken json'), _sse("[DONE]")]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    with pytest.raises(StreamParseError, match=r"broken json"):
        stream_chat([{"role": "user", "content": "hi"}], tools=[])


def test_stream_chat_ignores_blank_lines_and_non_data_lines(monkeypatch):
    lines = [
        "",
        ": this is an SSE comment, not a data line",
        _sse('{"choices": [{"delta": {"content": "ok"}}]}'),
        _sse("[DONE]"),
    ]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    content, _, _ = stream_chat([{"role": "user", "content": "hi"}], tools=[])
    assert content == "ok"


def test_stream_chat_accumulates_tool_call_arguments_across_chunks(monkeypatch):
    lines = [
        _sse('{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "foo", "arguments": ""}}]}}]}'),
        _sse('{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"a\\": "}}]}}]}'),
        _sse('{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]}'),
        _sse('{"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}'),
        _sse("[DONE]"),
    ]
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(lines))

    content, tool_calls, finish_reason = stream_chat([{"role": "user", "content": "hi"}], tools=[])
    assert content == ""
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": '{"a": 1}'}}
    ]


def test_stream_chat_uses_a_live_updated_qwen_url_not_an_import_time_snapshot(monkeypatch):
    """Real bug found via a live install: the first-run setup wizard
    (config.run_first_time_setup) updates skull.config.LLM_URL after
    LLM_URL was found empty at startup, but client.py used to do
    `from skull.config import LLM_URL` - a frozen copy taken once at
    import time. The wizard's update never reached that frozen copy, so
    every request after "successful" first-run setup still went to
    '/v1/chat/completions' with no scheme or host at all. client.py must
    read config.LLM_URL live, at call time, not via a one-time import."""
    from skull import config

    monkeypatch.setattr(config, "LLM_URL", "https://set-after-client-py-was-imported.example.com")
    monkeypatch.setattr(config, "LLM_KEY", "a-key")
    monkeypatch.setattr(config, "LLM_MODEL", "a-model")

    requested_urls = []

    def fake_post(url, **kwargs):
        requested_urls.append(url)
        return FakeResponse([_sse("[DONE]")])

    monkeypatch.setattr(client.requests, "post", fake_post)

    stream_chat([{"role": "user", "content": "hi"}], tools=[])

    assert requested_urls == ["https://set-after-client-py-was-imported.example.com/v1/chat/completions"]


def test_stream_chat_uses_the_openai_preset_url_without_a_doubled_v1(monkeypatch):
    """Real bug found via live testing: PROVIDER_PRESETS["openai"]["base_url"]
    used to be "https://api.openai.com/v1" (already including /v1), and
    stream_chat's own hardcoded "/v1/chat/completions" suffix turned that
    into ".../v1/v1/chat/completions" - a real 404 from OpenAI (confirmed
    live via curl: 404 for the doubled path, 401 - reaching the real
    endpoint - for the correct single-/v1 path). The preset must be a bare
    host with no /v1, since stream_chat always appends the full
    "/v1/chat/completions" itself."""
    from skull import config

    monkeypatch.setattr(config, "LLM_URL", config.PROVIDER_PRESETS["openai"]["base_url"])
    monkeypatch.setattr(config, "LLM_KEY", "a-key")
    monkeypatch.setattr(config, "LLM_MODEL", "gpt-5")
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")

    requested_urls = []

    def fake_post(url, **kwargs):
        requested_urls.append(url)
        return FakeResponse([_sse("[DONE]")])

    monkeypatch.setattr(client.requests, "post", fake_post)

    stream_chat([{"role": "user", "content": "hi"}], tools=[])

    assert requested_urls == ["https://api.openai.com/v1/chat/completions"]


def test_stream_chat_omits_chat_template_kwargs_for_gemini(monkeypatch):
    """Real bug found via a live Gemini setup: chat_template_kwargs (a
    Qwen/vLLM-specific extension) used to be sent unconditionally, and
    Gemini's OpenAI-compat layer rejects it outright with a real 400
    ("Unknown name \"chat_template_kwargs\": Cannot find field") -
    breaking every single chat turn immediately after a successful
    first-run setup."""
    from skull import config

    monkeypatch.setattr(config, "LLM_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setattr(config, "LLM_KEY", "a-key")
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")

    sent_payloads = []

    def fake_post(url, json=None, **kwargs):
        sent_payloads.append(json)
        return FakeResponse([_sse("[DONE]")])

    monkeypatch.setattr(client.requests, "post", fake_post)

    stream_chat([{"role": "user", "content": "hi"}], tools=[])

    assert "chat_template_kwargs" not in sent_payloads[0]


def test_stream_chat_includes_chat_template_kwargs_for_custom_provider(monkeypatch):
    """The flip side: a self-hosted Qwen/vLLM endpoint (LLM_PROVIDER not
    in STRICT_OPENAI_COMPAT_PROVIDERS) still gets chat_template_kwargs -
    the fix must not have thrown out the feature for the case it was
    actually built for."""
    from skull import config

    monkeypatch.setattr(config, "LLM_URL", "https://my-qwen-endpoint.example.com")
    monkeypatch.setattr(config, "LLM_KEY", "a-key")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen3.8-27b")
    monkeypatch.setattr(config, "LLM_PROVIDER", "custom")

    sent_payloads = []

    def fake_post(url, json=None, **kwargs):
        sent_payloads.append(json)
        return FakeResponse([_sse("[DONE]")])

    monkeypatch.setattr(client.requests, "post", fake_post)

    stream_chat([{"role": "user", "content": "hi"}], tools=[])

    assert sent_payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
