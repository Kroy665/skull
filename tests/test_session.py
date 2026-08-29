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


# ---------------------------------------------------------------------------
# _skill_ranking_query - real gap found via code review: skill relevance
# filtering (see registry.py's SKILL_FILTER_THRESHOLD) re-ranked against the
# ORIGINAL user_input every round-trip within a turn, frozen for the whole
# turn - a skill only discovered as relevant after seeing an earlier tool's
# result (e.g. low disk space -> now wants a notification skill never
# mentioned in the original phrasing) had no way to rank in. This builds the
# ranking query from the turn's accumulated context instead.
# ---------------------------------------------------------------------------

def test_skill_ranking_query_includes_original_user_input(isolated_skills_dir, isolated_memory_dir, monkeypatch):
    session = _make_session(monkeypatch)
    query = session._skill_ranking_query("check my disk space", [])
    assert "check my disk space" in query


def test_skill_ranking_query_includes_tool_call_names_from_this_turn(
    isolated_skills_dir, isolated_memory_dir, monkeypatch
):
    session = _make_session(monkeypatch)
    turn_messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "function": {"name": "disk_space", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": '{"free_gb": 2.1, "percent_used": 97}'},
    ]
    query = session._skill_ranking_query("check my disk space", turn_messages)
    assert "disk_space" in query
    assert "97" in query  # the tool result content must be folded in too


def test_skill_ranking_query_includes_assistant_reasoning_text(isolated_skills_dir, isolated_memory_dir, monkeypatch):
    session = _make_session(monkeypatch)
    turn_messages = [
        {"role": "assistant", "content": "Disk space is low, I should alert the user by email."},
    ]
    query = session._skill_ranking_query("check my disk space", turn_messages)
    assert "alert the user by email" in query


def test_skill_ranking_query_bounds_snippet_length(isolated_skills_dir, isolated_memory_dir, monkeypatch):
    session = _make_session(monkeypatch)
    huge_result = "x" * 100000
    turn_messages = [{"role": "tool", "tool_call_id": "1", "content": huge_result}]
    query = session._skill_ranking_query("hi", turn_messages)
    assert len(query) < 100000  # must not just concatenate the whole huge result unbounded


def test_skill_ranking_query_bounds_number_of_messages_considered(
    isolated_skills_dir, isolated_memory_dir, monkeypatch
):
    session = _make_session(monkeypatch)
    # An old tool result far enough back in a long turn shouldn't dominate
    # the ranking query forever - only the most recent messages count.
    old_messages = [{"role": "tool", "tool_call_id": str(i), "content": f"old-marker-{i}"} for i in range(20)]
    query = session._skill_ranking_query("hi", old_messages)
    assert "old-marker-0" not in query


def test_handle_turn_reranks_skills_using_accumulated_turn_context(
    isolated_skills_dir, isolated_memory_dir, monkeypatch
):
    """Integration-level confirmation: a skill whose name only appears in an
    intermediate tool result (not in the original user phrasing) must be
    present in the tools list built for the NEXT round-trip within the same
    turn - proving build_tools_and_impls is actually called with the
    enriched query, not just that the helper method computes one correctly
    in isolation.

    isolated_memory_dir's embedder is random noise (not semantically
    meaningful - documented in test_registry.py), so a controlled fake
    embedder is used here too: text mentioning "send_alert_email" gets a
    distinctive vector, everything else gets a near-orthogonal one - this
    isolates "does the enriched query reach the ranking step at all",
    not real-model semantic quality (covered by tests/scenarios.py against
    the live model)."""
    from skull.storage import store as mem
    from skull.tools import skills as sm

    def fake_embed(texts):
        import numpy as np

        vectors = []
        for t in texts:
            base = np.zeros(mem.EMBED_DIM, dtype=np.float32)
            if "send_alert_email" in t:
                base[0] = 1.0
            else:
                base[1] = 1.0
            vectors.append(base)
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(mem, "embed", fake_embed)

    for i in range(9):  # push skill count above SKILL_FILTER_THRESHOLD (8)
        sm.create_skill(
            f"filler_skill_{i}",
            f"does unrelated filler thing number {i}",
            {"type": "object", "properties": {}},
            "def run(**kwargs):\n    return {'ok': True}\n",
        )
    sm.create_skill(
        "send_alert_email",
        "sends an urgent alert email notification",
        {"type": "object", "properties": {}},
        "def run(**kwargs):\n    return {'sent': True}\n",
    )

    session = _make_session(monkeypatch)

    call_log = []

    def fake_stream_chat(request_messages, tools, spinner=None):
        call_log.append({t["function"]["name"] for t in tools})
        if len(call_log) == 1:
            return (
                None,
                [{"id": "1", "type": "function", "function": {"name": "disk_space", "arguments": "{}"}}],
                "tool_calls",
            )
        return ("done", None, "stop")

    monkeypatch.setattr(session_module, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(
        session_module,
        "run_tool_call",
        lambda tc, impls, verbose, spinner=None: (
            '{"free_gb": 2.1, "percent_used": 97, "note": "send_alert_email if critically low"}'
        ),
    )

    session.handle_turn("check my disk space")

    assert len(call_log) == 2
    # Second round-trip's tool list must include the skill discovered only
    # via the first round-trip's tool result content.
    assert "send_alert_email" in call_log[1]
