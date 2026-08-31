"""Tests for core/session.py's handle_turn - specifically its error
recovery around stream_chat, since an uncaught exception there used to
propagate all the way out of the REPL loop (which doesn't wrap handle_turn
at all) and crash the whole process, losing the conversation.

stream_chat itself is mocked here (not the HTTP layer) so these tests
focus purely on handle_turn's own exception handling and message-rollback
behavior, independent of client.py's parsing logic (covered in
test_client.py)."""

import pytest
import requests

from skull.core import session as session_module
from skull.core.client import StreamParseError
from skull.core.session import Session


def _make_session(monkeypatch, cwd=None):
    monkeypatch.setattr(session_module, "load_system_prompt", lambda: "system prompt")
    return Session(cwd=cwd)


def test_handle_turn_recovers_from_stream_parse_error_without_crashing(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
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
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
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


def test_handle_turn_still_recovers_from_http_error(isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch):
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


def test_handle_turn_succeeds_on_well_formed_response(isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch):
    session = _make_session(monkeypatch)

    monkeypatch.setattr(session_module, "stream_chat", lambda *a, **k: ("a reply", None, "stop"))

    ok = session.handle_turn("hello")
    assert ok is True
    assert session.messages[-1] == {"role": "assistant", "content": "a reply"}


# ---------------------------------------------------------------------------
# Real bug found via a live install with no network access to Hugging Face:
# the local embedding model failing to load (no internet, corrupted cache,
# disk full) used to be a raw uncaught exception - crashing the whole
# process even after the model had already produced a perfectly good
# answer the user could see. Both call sites that touch the embedder
# (memory retrieval before the model call, and conversation logging after)
# must degrade gracefully instead.
# ---------------------------------------------------------------------------

def test_handle_turn_survives_memory_context_failure(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
):
    session = _make_session(monkeypatch)
    monkeypatch.setattr(session_module, "stream_chat", lambda *a, **k: ("a reply", None, "stop"))

    def broken_memory_context(*a, **k):
        raise OSError("couldn't connect to huggingface.co")

    monkeypatch.setattr(session_module, "build_memory_context", broken_memory_context)

    ok = session.handle_turn("hello")
    assert ok is True
    assert session.messages[-1] == {"role": "assistant", "content": "a reply"}


def test_handle_turn_survives_conversation_memory_write_failure(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
):
    from skull.storage import store as mem

    session = _make_session(monkeypatch)
    monkeypatch.setattr(session_module, "stream_chat", lambda *a, **k: ("a reply", None, "stop"))

    class BrokenStore:
        def add(self, *a, **k):
            raise OSError("couldn't connect to huggingface.co")

    monkeypatch.setattr(mem, "conversations", lambda: BrokenStore())

    ok = session.handle_turn("hello")
    # The turn itself still succeeds - the user already has their answer -
    # even though saving it to long-term memory failed.
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

def test_skill_ranking_query_includes_original_user_input(isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch):
    session = _make_session(monkeypatch)
    query = session._skill_ranking_query("check my disk space", [])
    assert "check my disk space" in query


def test_skill_ranking_query_includes_tool_call_names_from_this_turn(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
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


def test_skill_ranking_query_includes_assistant_reasoning_text(isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch):
    session = _make_session(monkeypatch)
    turn_messages = [
        {"role": "assistant", "content": "Disk space is low, I should alert the user by email."},
    ]
    query = session._skill_ranking_query("check my disk space", turn_messages)
    assert "alert the user by email" in query


def test_skill_ranking_query_bounds_snippet_length(isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch):
    session = _make_session(monkeypatch)
    huge_result = "x" * 100000
    turn_messages = [{"role": "tool", "tool_call_id": "1", "content": huge_result}]
    query = session._skill_ranking_query("hi", turn_messages)
    assert len(query) < 100000  # must not just concatenate the whole huge result unbounded


def test_skill_ranking_query_bounds_number_of_messages_considered(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
):
    session = _make_session(monkeypatch)
    # An old tool result far enough back in a long turn shouldn't dominate
    # the ranking query forever - only the most recent messages count.
    old_messages = [{"role": "tool", "tool_call_id": str(i), "content": f"old-marker-{i}"} for i in range(20)]
    query = session._skill_ranking_query("hi", old_messages)
    assert "old-marker-0" not in query


def test_handle_turn_reranks_skills_using_accumulated_turn_context(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch
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


# ---------------------------------------------------------------------------
# run() - the module-level REPL entry point's required-config checks. Real
# requirement: QWEN_URL has no built-in default endpoint (removed along with
# every other reference to the original hardcoded self-hosted domain), so it
# must be checked and reported just as clearly as the pre-existing QWEN_KEY
# check, not left to fail later with a confusing malformed-URL request error.
# ---------------------------------------------------------------------------

def test_run_exits_with_clear_error_when_qwen_url_not_set(monkeypatch, capsys):
    monkeypatch.setattr(session_module, "QWEN_URL", "")
    monkeypatch.setattr(session_module, "QWEN_KEY", "some-key")

    with pytest.raises(SystemExit):
        session_module.run()

    err = capsys.readouterr().err
    assert "QWEN_URL" in err


def test_run_exits_with_clear_error_when_qwen_key_not_set(monkeypatch, capsys):
    monkeypatch.setattr(session_module, "QWEN_URL", "https://example.com")
    monkeypatch.setattr(session_module, "QWEN_KEY", "")

    with pytest.raises(SystemExit):
        session_module.run()

    err = capsys.readouterr().err
    assert "QWEN_KEY" in err


def test_run_checks_qwen_url_before_qwen_key(monkeypatch, capsys):
    """Both missing at once should report the QWEN_URL error first - it's
    checked first in run(), and a user missing both should see the more
    fundamental problem (no endpoint at all) rather than the key error.

    Both missing also means run() offers the interactive first-time setup
    wizard first (see the section below) - mocked here to simulate the
    user backing out, so this test stays focused on the fallback error
    path rather than the wizard itself."""
    monkeypatch.setattr(session_module, "QWEN_URL", "")
    monkeypatch.setattr(session_module, "QWEN_KEY", "")
    monkeypatch.setattr(session_module, "run_first_time_setup", lambda: None)

    with pytest.raises(SystemExit):
        session_module.run()

    err = capsys.readouterr().err
    assert "QWEN_URL" in err


# ---------------------------------------------------------------------------
# First-time setup wizard - real gap this fixes: previously, a user with no
# .env at all just got a stderr error pointing at a file path they had to
# go create themselves by hand. Now run() offers to collect QWEN_URL/
# QWEN_KEY/QWEN_MODEL directly in the terminal on first run.
# ---------------------------------------------------------------------------

def test_run_offers_setup_wizard_when_both_unset_and_uses_its_result(monkeypatch, capsys):
    monkeypatch.setattr(session_module, "QWEN_URL", "")
    monkeypatch.setattr(session_module, "QWEN_KEY", "")
    monkeypatch.setattr(
        session_module,
        "run_first_time_setup",
        lambda: {"QWEN_URL": "https://from-wizard.example.com", "QWEN_KEY": "wizard-key", "QWEN_MODEL": "wizard-model"},
    )

    # Setup succeeded, so run() should proceed past both checks into the
    # normal startup path - stop it right after by making Session() itself
    # blow up in a way we can detect, rather than running the full REPL.
    monkeypatch.setattr(session_module, "Session", lambda: (_ for _ in ()).throw(RuntimeError("reached Session()")))

    with pytest.raises(RuntimeError, match="reached Session"):
        session_module.run()

    assert session_module.QWEN_URL == "https://from-wizard.example.com"
    assert session_module.QWEN_KEY == "wizard-key"
    assert session_module.QWEN_MODEL == "wizard-model"


def test_run_does_not_offer_wizard_when_only_one_is_missing(monkeypatch, capsys):
    """A real partial misconfiguration (e.g. a real env var set for just
    one of the two) must get the direct, specific error - not the wizard,
    which could look like it's silently overriding an intentional env var
    setup for the other value."""
    monkeypatch.setattr(session_module, "QWEN_URL", "")
    monkeypatch.setattr(session_module, "QWEN_KEY", "some-key")
    wizard_called = []
    monkeypatch.setattr(session_module, "run_first_time_setup", lambda: wizard_called.append(True))

    with pytest.raises(SystemExit):
        session_module.run()

    assert wizard_called == []


def test_run_falls_back_to_error_when_wizard_is_cancelled(monkeypatch, capsys):
    monkeypatch.setattr(session_module, "QWEN_URL", "")
    monkeypatch.setattr(session_module, "QWEN_KEY", "")
    monkeypatch.setattr(session_module, "run_first_time_setup", lambda: None)

    with pytest.raises(SystemExit):
        session_module.run()

    err = capsys.readouterr().err
    assert "QWEN_URL" in err


# ---------------------------------------------------------------------------
# Per-directory conversation persistence (core/conversation_store.py) - a
# fresh Session for a given cwd picks up any conversation previously saved
# for that same cwd, and /reset (via reset_messages) clears the saved copy
# too, not just the in-memory one - otherwise the next launch from that
# directory would silently resurrect the "reset" conversation.
# ---------------------------------------------------------------------------

def test_session_starts_fresh_when_nothing_saved_for_cwd(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch, tmp_path
):
    session = _make_session(monkeypatch, cwd=str(tmp_path))
    assert session.messages == [{"role": "system", "content": "system prompt"}]
    assert session.resumed_message_count == 0


def test_session_resumes_saved_conversation_for_same_cwd(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch, tmp_path
):
    from skull.core import conversation_store as cs

    saved = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    cs.save(str(tmp_path), saved)

    session = _make_session(monkeypatch, cwd=str(tmp_path))
    assert session.messages == saved
    assert session.resumed_message_count == 2  # excludes the system message


def test_session_does_not_resume_conversation_saved_for_a_different_cwd(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch, tmp_path
):
    from skull.core import conversation_store as cs

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    cs.save(str(other_dir), [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])

    session = _make_session(monkeypatch, cwd=str(tmp_path))
    assert session.resumed_message_count == 0


def test_save_messages_persists_current_messages_for_session_cwd(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch, tmp_path
):
    from skull.core import conversation_store as cs

    session = _make_session(monkeypatch, cwd=str(tmp_path))
    session.messages.append({"role": "user", "content": "hello"})
    session.save_messages()

    assert cs.load(str(tmp_path)) == session.messages


def test_handle_turn_success_is_persisted_after_saving(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch, tmp_path
):
    from skull.core import conversation_store as cs

    session = _make_session(monkeypatch, cwd=str(tmp_path))
    monkeypatch.setattr(session_module, "stream_chat", lambda *a, **k: ("a reply", None, "stop"))

    session.handle_turn("hello")
    session.save_messages()

    assert cs.load(str(tmp_path)) == session.messages
    assert session.messages[-1] == {"role": "assistant", "content": "a reply"}


def test_reset_messages_clears_saved_conversation_for_cwd(
    isolated_skills_dir, isolated_memory_dir, isolated_conversations_dir, monkeypatch, tmp_path
):
    from skull.core import conversation_store as cs

    session = _make_session(monkeypatch, cwd=str(tmp_path))
    session.messages.append({"role": "user", "content": "hello"})
    session.save_messages()
    assert cs.load(str(tmp_path)) is not None

    session.reset_messages()

    assert session.messages == [{"role": "system", "content": "system prompt"}]
    assert cs.load(str(tmp_path)) is None
