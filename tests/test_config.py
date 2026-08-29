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
    run from anywhere, so QWEN_KEY must be found via the config dir, not
    whatever the current working directory happens to be."""
    monkeypatch.delenv("QWEN_KEY", raising=False)
    # CONFIG_DIR is <XDG_CONFIG_HOME>/skull, not XDG_CONFIG_HOME itself -
    # _user_config_dir() creates that "skull" subdirectory, so the .env
    # must be written there for load_dotenv to find it after reload.
    (tmp_path / "skull").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skull" / ".env").write_text("QWEN_KEY=test-key-from-config-dir\n")

    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    assert config.QWEN_KEY == "test-key-from-config-dir"


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


def test_qwen_url_has_no_hardcoded_default(tmp_path, monkeypatch):
    """QWEN_URL must not fall back to any built-in default endpoint - every
    user points this at their own Qwen-compatible endpoint explicitly."""
    monkeypatch.delenv("QWEN_URL", raising=False)
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path))
    assert config.QWEN_URL == ""


def test_qwen_url_reads_from_env_when_set(tmp_path, monkeypatch):
    config = _reload_config(monkeypatch, XDG_CONFIG_HOME=str(tmp_path), QWEN_URL="https://my-endpoint.example.com/")
    assert config.QWEN_URL == "https://my-endpoint.example.com"  # trailing slash stripped
