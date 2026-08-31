"""Tests for config.py - specifically _user_config_dir() (the per-user
config directory a real installed `skull` command uses regardless of what
directory it's invoked from) and load_system_prompt()'s first-run copy of
the bundled default prompt.

config.py computes CONFIG_DIR and loads .env at IMPORT time (module-level
side effects), so these tests reload the module fresh under a controlled
HOME/XDG_CONFIG_HOME rather than testing the already-imported module's
already-computed CONFIG_DIR."""

import importlib
import sys

import pytest


def _reload_config(monkeypatch, **env_overrides):
    """Reload skull.config fresh with the given env vars set, restoring the
    original module state afterward so other test files' imports of
    skull.config aren't left pointing at a throwaway path."""
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    import skull.config as config_module

    importlib.reload(config_module)
    return config_module


@pytest.fixture(autouse=True)
def _restore_config_module_after_test():
    """Every test in this file reloads skull.config with monkeypatched env
    vars - reload it back to a normal state afterward (monkeypatch already
    restores the env vars themselves; this restores the module-level state
    that was computed from them) so later test files see a sane config."""
    yield
    import skull.config as config_module
    importlib.reload(config_module)


def test_user_config_dir_respects_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    xdg_dir = tmp_path / "custom_xdg"
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(xdg_dir))

    assert config.CONFIG_DIR == xdg_dir / "skull"
    assert config.CONFIG_DIR.exists()


def test_user_config_dir_created_if_missing(tmp_path, monkeypatch):
    xdg_dir = tmp_path / "does_not_exist_yet"
    assert not xdg_dir.exists()

    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(xdg_dir))
    assert config.CONFIG_DIR.exists()


def test_derived_paths_live_under_config_dir(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    assert config.SKILLS_DIR == config.CONFIG_DIR / "skills"
    assert config.MEMORY_DIR == config.CONFIG_DIR / "memory"
    assert config.PIPELINES_DIR == config.CONFIG_DIR / "pipelines"
    assert config.SKILLS_ENV_PATH == config.CONFIG_DIR / "skills.env"
    assert config.SYSTEM_PROMPT_PATH == config.CONFIG_DIR / "SYSTEM_PROMPT.md"


def test_load_dotenv_reads_from_config_dir_not_cwd(tmp_path, monkeypatch):
    """Real gap this is designed around: a genuinely installed command is
    run from anywhere, so LLM_KEY must be found via the config dir, not
    whatever the current working directory happens to be."""
    monkeypatch.delenv("LLM_KEY", raising=False)
    # CONFIG_DIR is <XDG_CONFIG_HOME>/skull, not XDG_CONFIG_HOME itself -
    # _user_config_dir() creates that "skull" subdirectory, so the .env
    # must be written there for load_dotenv to find it after reload.
    (tmp_path / "skull").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skull" / ".env").write_text("LLM_KEY=test-key-from-config-dir\n")

    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    assert config.LLM_KEY == "test-key-from-config-dir"


def test_load_system_prompt_copies_bundled_default_on_first_run(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    assert not config.SYSTEM_PROMPT_PATH.exists()
    prompt = config.load_system_prompt()

    assert config.SYSTEM_PROMPT_PATH.exists()
    assert len(prompt) > 0
    assert prompt == config.SYSTEM_PROMPT_PATH.read_text().strip()


def test_load_system_prompt_leaves_existing_user_copy_untouched(tmp_path, monkeypatch):
    """A user's own edits to their copy of SYSTEM_PROMPT.md must survive -
    load_system_prompt must never overwrite an existing user copy with the
    bundled default."""
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    config.SYSTEM_PROMPT_PATH.write_text("A custom user-edited prompt.")
    prompt = config.load_system_prompt()

    assert prompt == "A custom user-edited prompt."


def test_load_system_prompt_falls_back_when_bundled_default_missing(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "_BUNDLED_SYSTEM_PROMPT_PATH", tmp_path / "nonexistent.md")

    prompt = config.load_system_prompt()
    assert "helpful terminal assistant" in prompt


def test_llm_url_has_no_hardcoded_default(tmp_path, monkeypatch):
    """LLM_URL must not fall back to any built-in default endpoint - every
    user points this at their own chosen provider's endpoint explicitly."""
    monkeypatch.delenv("LLM_URL", raising=False)
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    assert config.LLM_URL == ""


def test_llm_url_reads_from_env_when_set(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path), LLM_URL="https://my-endpoint.example.com/")
    assert config.LLM_URL == "https://my-endpoint.example.com"  # trailing slash stripped


def test_llm_model_has_no_hardcoded_default(tmp_path, monkeypatch):
    """LLM_MODEL must not fall back to any built-in default model name -
    a stale hardcoded model for a multi-provider setup would silently
    break requests as soon as a different provider is picked."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    assert config.LLM_MODEL == ""


# ---------------------------------------------------------------------------
# run_first_time_setup - real gap this fixes: a user with no .env at all
# used to just get a stderr error naming a file path they had to go create
# by hand. Now it's collected directly, interactively, in the terminal -
# including an arrow-key provider picker (OpenAI/Gemini/custom, via
# questionary) and a web-ranked live model list fetched from that endpoint,
# since every provider offered here speaks the same OpenAI-compatible wire
# format (see PROVIDER_PRESETS).
#
# questionary.select/.text return a Question object whose .ask() is what
# actually drives the terminal - there's no input()/getpass to intercept
# for the select steps, so these tests replace questionary.select/.text
# themselves with a fake that returns a pre-set answer, one per call in
# sequence. LLM_KEY still goes through plain getpass.getpass (a real
# secret, kept as hidden free-text input rather than a select).
#
# Interaction sequence under test: provider select -> (custom URL text
# only if provider == "custom") -> key (getpass) -> model select (only if
# list_available_models returns something) or model text (if it returns
# []). list_available_models and _rank_models_by_web_search are mocked
# directly rather than mocking `requests`/web_search - their own
# request-shape/error-handling/ranking behavior is covered separately
# below.
# ---------------------------------------------------------------------------

def _fake_questionary(monkeypatch, config, answers):
    """Replace config.questionary.select/.text with fakes that return the
    next answer from `answers` (a list, consumed in call order) from
    .ask() - simulating a user's picks/typed text without touching the
    real terminal."""
    remaining = iter(answers)

    class FakeQuestion:
        def __init__(self, answer):
            self._answer = answer

        def ask(self):
            return self._answer

    def fake_select(*args, **kwargs):
        return FakeQuestion(next(remaining))

    def fake_text(*args, **kwargs):
        return FakeQuestion(next(remaining))

    monkeypatch.setattr(config.questionary, "select", fake_select)
    monkeypatch.setattr(config.questionary, "text", fake_text)


def test_run_first_time_setup_custom_url_with_model_list(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: ["model-b", "model-a"])

    # provider select -> "custom", url text, key (getpass), model select
    _fake_questionary(monkeypatch, config, ["custom", "https://my-qwen.example.com", "model-a"])
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    result = config.run_first_time_setup()

    assert result == {
        "LLM_URL": "https://my-qwen.example.com",
        "LLM_KEY": "my-secret-key",
        "LLM_MODEL": "model-a",
    }
    env_path = config.CONFIG_DIR / ".env"
    content = env_path.read_text()
    assert "LLM_URL=https://my-qwen.example.com" in content
    assert "LLM_KEY=my-secret-key" in content
    assert "LLM_MODEL=model-a" in content


def test_run_first_time_setup_openai_preset_prefills_url_and_ranks_models(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: ["gpt-4o", "gpt-5"])
    monkeypatch.setattr(config, "_rank_models_by_web_search", lambda models, query: ["gpt-5", "gpt-4o"])

    _fake_questionary(monkeypatch, config, ["openai", "gpt-5"])  # provider select, model select
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    result = config.run_first_time_setup()

    assert result["LLM_URL"] == config.PROVIDER_PRESETS["openai"]["base_url"]
    assert result["LLM_MODEL"] == "gpt-5"


def test_run_first_time_setup_gemini_preset_prefills_url(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: ["gemini-2.5-pro"])
    monkeypatch.setattr(config, "_rank_models_by_web_search", lambda models, query: models)

    _fake_questionary(monkeypatch, config, ["gemini", "gemini-2.5-pro"])
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    result = config.run_first_time_setup()

    assert result["LLM_URL"] == config.PROVIDER_PRESETS["gemini"]["base_url"]


def test_run_first_time_setup_falls_back_to_typed_model_when_list_fails(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: [])

    _fake_questionary(monkeypatch, config, ["custom", "https://my-qwen.example.com", "my-typed-model"])
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    result = config.run_first_time_setup()

    assert result["LLM_MODEL"] == "my-typed-model"


def test_run_first_time_setup_custom_provider_skips_web_ranking(tmp_path, monkeypatch):
    """A custom/self-hosted endpoint has no PROVIDER_PRESETS search_query -
    ranking must be skipped (not crash trying to search for "None")."""
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: ["model-b", "model-a"])

    def fail_if_called(*a, **k):
        raise AssertionError("_rank_models_by_web_search must not be called for a custom provider")

    monkeypatch.setattr(config, "_rank_models_by_web_search", fail_if_called)
    _fake_questionary(monkeypatch, config, ["custom", "https://my-qwen.example.com", "model-a"])
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    result = config.run_first_time_setup()

    assert result["LLM_MODEL"] == "model-a"


def test_run_first_time_setup_returns_none_when_provider_selection_cancelled(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    _fake_questionary(monkeypatch, config, [None])  # Ctrl-C/Ctrl-D inside questionary -> ask() returns None

    assert config.run_first_time_setup() is None
    assert not (config.CONFIG_DIR / ".env").exists()


def test_run_first_time_setup_returns_none_when_url_left_blank(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    _fake_questionary(monkeypatch, config, ["custom", ""])

    assert config.run_first_time_setup() is None
    assert not (config.CONFIG_DIR / ".env").exists()


def test_run_first_time_setup_returns_none_when_key_left_blank(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    _fake_questionary(monkeypatch, config, ["custom", "https://my-qwen.example.com"])
    monkeypatch.setattr("getpass.getpass", lambda *_: "")

    assert config.run_first_time_setup() is None
    assert not (config.CONFIG_DIR / ".env").exists()


def test_run_first_time_setup_returns_none_when_model_selection_cancelled(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: ["model-a"])
    monkeypatch.setattr(config, "_rank_models_by_web_search", lambda models, query: models)
    _fake_questionary(monkeypatch, config, ["openai", None])  # model select cancelled
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    assert config.run_first_time_setup() is None
    assert not (config.CONFIG_DIR / ".env").exists()


def test_run_first_time_setup_returns_none_on_keyboard_interrupt(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    def raise_interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(config.questionary, "select", raise_interrupt)

    assert config.run_first_time_setup() is None
    assert not (config.CONFIG_DIR / ".env").exists()


def test_run_first_time_setup_writes_file_with_owner_only_permissions(tmp_path, monkeypatch):
    import stat

    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr(config, "list_available_models", lambda url, key: [])

    _fake_questionary(monkeypatch, config, ["custom", "https://my-qwen.example.com", "a-model"])
    monkeypatch.setattr("getpass.getpass", lambda *_: "my-secret-key")

    config.run_first_time_setup()

    mode = (config.CONFIG_DIR / ".env").stat().st_mode
    assert stat.S_IMODE(mode) == stat.S_IRUSR | stat.S_IWUSR


# ---------------------------------------------------------------------------
# _rank_models_by_web_search - real-content-aware ranking on top of the raw
# /v1/models list, since that list has no "release date" or "is this
# current" signal. Never invents a model id that isn't in the real list -
# a web search snippet only decides ORDER, not membership.
# ---------------------------------------------------------------------------

def test_rank_models_by_web_search_surfaces_mentioned_models_first(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    def fake_web_search(query, count=8):
        return {
            "query": query,
            "results": [
                {"title": "OpenAI launches GPT-5.6", "snippet": "GPT-5.6 is the new flagship model."},
            ],
        }

    monkeypatch.setattr("skull.tools.web.web_search", fake_web_search)

    models = ["gpt-4o", "gpt-5.6", "gpt-3.5-turbo"]
    ranked = config._rank_models_by_web_search(models, "latest OpenAI GPT model")

    assert ranked[0] == "gpt-5.6"
    assert set(ranked) == set(models)  # never drops or invents a model id


def test_rank_models_by_web_search_falls_back_on_search_failure(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    monkeypatch.setattr("skull.tools.web.web_search", lambda query, count=8: {"error": "search failed"})

    models = ["gpt-4o", "gpt-5"]
    ranked = config._rank_models_by_web_search(models, "latest OpenAI GPT model")

    assert ranked == sorted(models, reverse=True)


def test_rank_models_by_web_search_falls_back_when_nothing_recognizable_mentioned(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    def fake_web_search(query, count=8):
        return {"query": query, "results": [{"title": "Unrelated article", "snippet": "no model names here"}]}

    monkeypatch.setattr("skull.tools.web.web_search", fake_web_search)

    models = ["gpt-4o", "gpt-5"]
    ranked = config._rank_models_by_web_search(models, "latest OpenAI GPT model")

    assert ranked == sorted(models, reverse=True)


def test_rank_models_by_web_search_never_crashes_on_exception(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    def raise_error(*a, **k):
        raise RuntimeError("network exploded")

    monkeypatch.setattr("skull.tools.web.web_search", raise_error)

    models = ["gpt-4o", "gpt-5"]
    ranked = config._rank_models_by_web_search(models, "latest OpenAI GPT model")

    assert ranked == sorted(models, reverse=True)


# ---------------------------------------------------------------------------
# list_available_models - the live /v1/models lookup used by the wizard's
# picker above. Must degrade to [] on any failure rather than raising, so
# a bad key or unreachable host never blocks setup - the wizard always has
# a "type it yourself" fallback.
# ---------------------------------------------------------------------------

def test_list_available_models_returns_sorted_ids(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "z-model"}, {"id": "a-model"}]}

    monkeypatch.setattr(config.requests, "get", lambda *a, **k: FakeResponse())

    assert config.list_available_models("https://example.com", "key") == ["a-model", "z-model"]


def test_list_available_models_returns_empty_list_on_http_error(tmp_path, monkeypatch):
    import requests

    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    def raise_error(*a, **k):
        raise requests.RequestException("connection failed")

    monkeypatch.setattr(config.requests, "get", raise_error)

    assert config.list_available_models("https://example.com", "bad-key") == []


def test_list_available_models_returns_empty_list_on_unexpected_response_shape(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"unexpected": "shape"}  # no "data" key

    monkeypatch.setattr(config.requests, "get", lambda *a, **k: FakeResponse())

    assert config.list_available_models("https://example.com", "key") == []


def test_list_available_models_sends_bearer_auth_header(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    def fake_get(url, headers=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(config.requests, "get", fake_get)

    config.list_available_models("https://example.com/v1/", "my-key")

    assert captured["url"] == "https://example.com/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer my-key"
