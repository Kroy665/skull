"""Tests for core/compaction.py - the estimate/cut/summarize logic that had
several real bugs during development (blind spots in what counted toward
the token estimate, an internal re-check that silently nullified the
caller's more accurate decision, and the summarization call itself being
too large to send)."""

import json

import pytest

from skull.core import compaction as comp


def _mock_summarize_response(monkeypatch, summary_text: str = "a summary"):
    """Replace requests.post so _summarize() returns a canned response
    without any network call."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": summary_text}}]}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(comp.requests, "post", fake_post)


def _msg(role, content=None, tool_calls=None, tool_call_id=None):
    m = {"role": role, "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    if tool_call_id is not None:
        m["tool_call_id"] = tool_call_id
    return m


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_counts_message_content():
    messages = [_msg("user", "x" * 35)]  # 35 chars / 3.5 = 10 tokens
    assert comp.estimate_tokens(messages) == 10


def test_estimate_tokens_counts_tool_call_arguments():
    tool_calls = [{"function": {"name": "f", "arguments": json.dumps({"a": 1})}}]
    messages = [_msg("assistant", None, tool_calls=tool_calls)]
    expected_chars = len("f") + len(json.dumps({"a": 1}))
    assert comp.estimate_tokens(messages) == int(expected_chars / comp.CHARS_PER_TOKEN)


def test_estimate_tokens_includes_tools_schema():
    """This is the exact blind spot that let a real request slip past the
    old (messages-only) estimate and 400 with a genuine context-length
    error - tool schemas can be thousands of tokens on their own."""
    messages = [_msg("user", "hi")]
    tools = [{"type": "function", "function": {"name": "x" * 1000}}]

    without_tools = comp.estimate_tokens(messages)
    with_tools = comp.estimate_tokens(messages, tools=tools)

    assert with_tools > without_tools
    # Combined-then-divided vs. separately-divided-then-summed can differ by
    # a token or two from truncation alone - assert the difference is in the
    # right ballpark, not byte-exact against a naively recomputed formula.
    expected_diff = len(json.dumps(tools)) / comp.CHARS_PER_TOKEN
    assert abs((with_tools - without_tools) - expected_diff) <= 1


def test_estimate_tokens_includes_extra_chars():
    messages = [_msg("user", "hi")]
    baseline = comp.estimate_tokens(messages)
    with_extra = comp.estimate_tokens(messages, extra_chars=350)
    assert with_extra - baseline == int(350 / comp.CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# _find_cut_index - must never split a tool_calls message from its tool
# responses (would produce a structurally invalid messages list)
# ---------------------------------------------------------------------------

def test_find_cut_index_avoids_splitting_tool_call_pair():
    messages = [
        _msg("system", "sys"),
        _msg("user", "u1"),
        _msg("assistant", None, tool_calls=[{"id": "1", "function": {"name": "f"}}]),
        _msg("tool", "r1", tool_call_id="1"),
        _msg("assistant", "final"),
        _msg("user", "u2"),
    ]
    # Any target_end that would land inside the tool_calls/tool block should
    # back off to right before that block (index 2), not split it.
    for target_end in (3, 4):
        cut = comp._find_cut_index(messages, start=1, target_end=target_end)
        assert cut == 2, f"target_end={target_end} produced an unsafe cut at {cut}"


def test_find_cut_index_allows_clean_boundaries():
    messages = [
        _msg("system", "sys"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
    ]
    assert comp._find_cut_index(messages, start=1, target_end=2) == 2
    assert comp._find_cut_index(messages, start=1, target_end=3) == 3


def test_find_cut_index_returns_start_when_nothing_safe():
    messages = [
        _msg("system", "sys"),
        _msg("assistant", None, tool_calls=[{"id": "1", "function": {"name": "f"}}]),
        _msg("tool", "r1", tool_call_id="1"),
    ]
    # target_end lands inside the only tool_call block - no safe cut exists
    # before it other than `start` itself.
    assert comp._find_cut_index(messages, start=1, target_end=2) == 1


# ---------------------------------------------------------------------------
# _summarize - the transcript-size cap (this exact bug: summarizing an
# oversized chunk would itself 400, since it's sent as one request)
# ---------------------------------------------------------------------------

def test_summarize_caps_oversized_transcript(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers, json, timeout):
        captured["prompt"] = json["messages"][0]["content"]
        return FakeResponse()

    monkeypatch.setattr(comp.requests, "post", fake_post)

    huge_messages = [_msg("user", "x" * (comp.MAX_TRANSCRIPT_CHARS * 2))]
    comp._summarize(huge_messages)

    assert "omitted from the middle" in captured["prompt"]
    # The prompt's transcript portion must not reproduce the full oversized
    # input verbatim - it must actually have been trimmed.
    assert len(captured["prompt"]) < comp.MAX_TRANSCRIPT_CHARS * 2


def test_summarize_leaves_small_transcript_untouched(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers, json, timeout):
        captured["prompt"] = json["messages"][0]["content"]
        return FakeResponse()

    monkeypatch.setattr(comp.requests, "post", fake_post)

    small_messages = [_msg("user", "hello")]
    comp._summarize(small_messages)

    assert "omitted from the middle" not in captured["prompt"]
    assert "hello" in captured["prompt"]


# ---------------------------------------------------------------------------
# compact_if_needed - end-to-end behavior
# ---------------------------------------------------------------------------

def test_compact_if_needed_noop_below_threshold(monkeypatch):
    _mock_summarize_response(monkeypatch)
    messages = [_msg("system", "sys"), _msg("user", "hi")]
    result, compacted = comp.compact_if_needed(messages)
    assert compacted is False
    assert result == messages


def test_compact_if_needed_triggers_above_threshold(monkeypatch):
    _mock_summarize_response(monkeypatch, "SUMMARY_TEXT")
    messages = [_msg("system", "sys")]
    for i in range(200):
        messages.append(_msg("user", f"msg {i} " + "x" * 300))
        messages.append(_msg("assistant", f"reply {i} " + "y" * 300))

    result, compacted = comp.compact_if_needed(messages)
    assert compacted is True
    assert len(result) < len(messages)
    assert result[0] == messages[0]  # system prompt untouched
    assert result[-comp.KEEP_RECENT_MESSAGES:] == messages[-comp.KEEP_RECENT_MESSAGES:]
    assert any("SUMMARY_TEXT" in (m.get("content") or "") for m in result)


def test_compact_if_needed_summary_message_is_not_system_role(monkeypatch):
    """The Qwen endpoint rejects any system-role message that isn't first
    in the list - the summary must use a different role."""
    _mock_summarize_response(monkeypatch)
    messages = [_msg("system", "sys")]
    for i in range(200):
        messages.append(_msg("user", f"msg {i} " + "x" * 300))
        messages.append(_msg("assistant", f"reply {i} " + "y" * 300))

    result, compacted = comp.compact_if_needed(messages)
    assert compacted is True
    for m in result[1:]:
        assert m["role"] != "system"


def test_compact_if_needed_force_skips_internal_threshold_check(monkeypatch):
    """This is the exact bug found while wiring compaction into session.py:
    force=True must let a caller's more complete (tools-aware) decision
    take effect even when compact_if_needed's own messages-only estimate
    would say "not needed yet"."""
    _mock_summarize_response(monkeypatch)
    # Deliberately small - below the internal threshold on its own.
    messages = [_msg("system", "sys")]
    for i in range(20):
        messages.append(_msg("user", f"msg {i}"))
        messages.append(_msg("assistant", f"reply {i}"))

    assert comp.estimate_tokens(messages) < comp.COMPACT_TRIGGER_TOKENS

    _, compacted_without_force = comp.compact_if_needed(messages)
    assert compacted_without_force is False

    _, compacted_with_force = comp.compact_if_needed(messages, force=True)
    assert compacted_with_force is True


def test_compact_if_needed_preserves_tool_call_pairing(monkeypatch):
    """Verify a compaction over a conversation with real tool_calls/tool
    pairs never leaves an orphaned tool message or a tool_calls message
    missing its response."""
    _mock_summarize_response(monkeypatch)
    messages = [_msg("system", "sys")]
    for i in range(100):
        messages.append(_msg("user", f"do thing {i} " + "x" * 20))
        messages.append(
            _msg(
                "assistant",
                None,
                tool_calls=[{"id": f"call_{i}", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            )
        )
        messages.append(_msg("tool", "result " * 10, tool_call_id=f"call_{i}"))
        messages.append(_msg("assistant", f"done {i} " + "y" * 20))

    # force=True: this test is specifically about pairing-preservation
    # logic, not about re-verifying the trigger threshold (covered
    # elsewhere) - so it doesn't need to be sized to actually cross it.
    result, compacted = comp.compact_if_needed(messages, force=True)
    assert compacted is True

    for i, m in enumerate(result):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids_needed = {tc["id"] for tc in m["tool_calls"]}
            found_ids = set()
            j = i + 1
            while j < len(result) and result[j].get("role") == "tool":
                found_ids.add(result[j]["tool_call_id"])
                j += 1
            assert ids_needed <= found_ids, f"tool_calls at index {i} missing a response"

        if m.get("role") == "tool":
            # every tool message must be preceded (not necessarily
            # immediately) by an assistant tool_calls message containing
            # its id, before hitting a user message
            found = False
            for j in range(i - 1, -1, -1):
                if result[j].get("role") == "assistant" and result[j].get("tool_calls"):
                    found = any(tc["id"] == m["tool_call_id"] for tc in result[j]["tool_calls"])
                    break
                if result[j].get("role") == "user":
                    break
            assert found, f"orphaned tool message at index {i}"


def test_compact_if_needed_returns_unchanged_on_summarize_failure(monkeypatch):
    def failing_post(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(comp.requests, "post", failing_post)

    messages = [_msg("system", "sys")]
    for i in range(200):
        messages.append(_msg("user", f"msg {i} " + "x" * 300))
        messages.append(_msg("assistant", f"reply {i} " + "y" * 300))

    result, compacted = comp.compact_if_needed(messages, force=True)
    assert compacted is False
    assert result == messages  # nothing lost when summarization itself fails


def test_compact_if_needed_noop_with_insufficient_history(monkeypatch):
    _mock_summarize_response(monkeypatch)
    # Fewer messages than KEEP_RECENT_MESSAGES - nothing safe to compact.
    messages = [_msg("system", "sys"), _msg("user", "hi"), _msg("assistant", "hello")]
    result, compacted = comp.compact_if_needed(messages, force=True)
    assert compacted is False
    assert result == messages
