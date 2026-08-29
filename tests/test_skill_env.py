"""Tests for tools/skill_env.py - the secrets store for skills, built so a
credential a skill needs (API key, password) never has to pass through the
model at all. Real problem this solves: a model asked a user to paste an
SMTP password directly into the chat conversation, landing it in plain
text in conversation history/memory - get_env()/request_skill_env() let a
skill's code read a secret directly from local storage instead."""

import pytest

from skull.tools import skill_env as scenv


def test_get_env_returns_none_when_not_set(isolated_skills_env):
    assert scenv.get_env("SMTP_PASSWORD") is None


def test_has_env_false_when_not_set(isolated_skills_env):
    assert scenv.has_env("SMTP_PASSWORD") is False


def test_request_skill_env_stores_value_via_getpass(isolated_skills_env, monkeypatch):
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "s3cr3t")

    result = scenv.request_skill_env("SMTP_PASSWORD", reason="send an email")
    assert result == {"status": "set", "name": "SMTP_PASSWORD"}
    assert scenv.get_env("SMTP_PASSWORD") == "s3cr3t"
    assert scenv.has_env("SMTP_PASSWORD") is True


def test_request_skill_env_result_never_contains_the_value(isolated_skills_env, monkeypatch):
    """The exact safety property this module exists for: the tool result
    the model sees must never include the actual secret value, under any
    circumstances, even on success."""
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "super-secret-value-12345")

    result = scenv.request_skill_env("API_KEY")
    assert "super-secret-value-12345" not in str(result)
    assert set(result.keys()) == {"status", "name"}


def test_request_skill_env_blank_input_cancels(isolated_skills_env, monkeypatch):
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "")

    result = scenv.request_skill_env("SMTP_PASSWORD")
    assert result == {"status": "cancelled", "name": "SMTP_PASSWORD"}
    assert scenv.has_env("SMTP_PASSWORD") is False


def test_request_skill_env_eof_cancels_gracefully(isolated_skills_env, monkeypatch):
    def raise_eof(prompt):
        raise EOFError()

    monkeypatch.setattr(scenv.getpass, "getpass", raise_eof)

    result = scenv.request_skill_env("SMTP_PASSWORD")
    assert result == {"status": "cancelled", "name": "SMTP_PASSWORD"}


def test_request_skill_env_rejects_invalid_name(isolated_skills_env):
    result = scenv.request_skill_env("not a valid name!")
    assert "error" in result


def test_list_missing_env_reports_unset_names(isolated_skills_env, monkeypatch):
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "value")
    scenv.request_skill_env("SMTP_USERNAME")

    missing = scenv.list_missing_env(["SMTP_USERNAME", "SMTP_PASSWORD"])
    assert missing == ["SMTP_PASSWORD"]


def test_list_missing_env_empty_when_all_set(isolated_skills_env, monkeypatch):
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "value")
    scenv.request_skill_env("SMTP_USERNAME")
    scenv.request_skill_env("SMTP_PASSWORD")

    assert scenv.list_missing_env(["SMTP_USERNAME", "SMTP_PASSWORD"]) == []


def test_clear_env_removes_a_set_value(isolated_skills_env, monkeypatch):
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "value")
    scenv.request_skill_env("API_KEY")

    result = scenv.clear_env("API_KEY")
    assert result == {"status": "cleared", "name": "API_KEY"}
    assert scenv.has_env("API_KEY") is False


def test_clear_env_missing_returns_error(isolated_skills_env):
    result = scenv.clear_env("NEVER_SET")
    assert "error" in result


def test_values_persist_across_separate_loads(isolated_skills_env, monkeypatch):
    """Confirms the file-backed storage actually round-trips - not just an
    in-memory cache that would reset between process runs."""
    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "persisted-value")
    scenv.request_skill_env("API_KEY")

    # _load() re-reads from disk every call rather than caching - simulate
    # a fresh read the way a new process would see it.
    assert scenv._load() == {"API_KEY": "persisted-value"}


def test_skills_env_file_is_permissioned_owner_only(isolated_skills_env, monkeypatch):
    import stat

    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "value")
    scenv.request_skill_env("API_KEY")

    mode = stat.S_IMODE(isolated_skills_env.stat().st_mode)
    assert mode == (stat.S_IRUSR | stat.S_IWUSR), f"expected 0600, got {oct(mode)}"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("SMTP_PASSWORD", True),
        ("API_KEY", True),
        ("A", True),
        ("smtp_password", False),  # lowercase
        ("123_KEY", False),  # doesn't start with a letter
        ("SMTP-PASSWORD", False),  # hyphen not allowed
        ("", False),
    ],
)
def test_is_valid_env_name(name, expected):
    assert scenv.is_valid_env_name(name) is expected
