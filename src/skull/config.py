"""Environment configuration, paths, and terminal color constants.

Runs as a real installed command (`skull`, from any directory), not tied
to a project checkout - so per-user data (skills/, memory/, pipelines/,
skills.env, the .env with QWEN_KEY, and an editable copy of
SYSTEM_PROMPT.md) lives in a standard per-user config directory, not next
to the source code. Everything here is found the same way regardless of
where the `skull` command is invoked from.
"""

import getpass
import os
import shutil
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv


def _user_config_dir() -> Path:
    """The conventional per-user config directory for a CLI tool: honors
    XDG_CONFIG_HOME on Linux (and anywhere else that sets it), falls back
    to the platform-standard location otherwise. Created if missing."""
    override = os.environ.get("XDG_CONFIG_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path.home() / ".config"
    config_dir = base / "skull"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


CONFIG_DIR = _user_config_dir()

# .env (QWEN_KEY, QWEN_URL, QWEN_MODEL, E2B_API_KEY) lives in the per-user
# config dir, not the current working directory - a real installed command
# is run from anywhere, so there's no project checkout to find a .env next
# to. load_dotenv() with a specific path (rather than its default upward
# search from cwd) is deliberate here for that reason.
load_dotenv(CONFIG_DIR / ".env")

# The package's own bundled copy - the source of the default prompt and of
# the first-run copy below. Not meant to be edited directly (it lives
# inside the installed package, which a normal install won't have write
# access to and which upgrades would overwrite anyway).
_BUNDLED_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "SYSTEM_PROMPT.md"

# The user's own editable copy - created from the bundled default on first
# run (see load_system_prompt) if it doesn't exist yet, and left alone on
# every run after that, so a user's edits survive both subsequent runs and
# package upgrades.
SYSTEM_PROMPT_PATH = CONFIG_DIR / "SYSTEM_PROMPT.md"

SKILLS_DIR = CONFIG_DIR / "skills"
MEMORY_DIR = CONFIG_DIR / "memory"
PIPELINES_DIR = CONFIG_DIR / "pipelines"
# Per-directory saved conversations (see core/conversation_store.py) - one
# file per working directory `skull` has been launched from, so a
# conversation resumes automatically when launched again from that same
# directory.
CONVERSATIONS_DIR = CONFIG_DIR / "conversations"
# Secrets for skills (API keys, passwords) - kept separate from the app's
# own .env so a skill's credentials are never something the model reads,
# sets, or sees a value from; only the user can set one, via a direct
# terminal prompt (see tools/skill_env.py).
SKILLS_ENV_PATH = CONFIG_DIR / "skills.env"

QWEN_URL = (os.environ.get("QWEN_URL") or "").rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
QWEN_KEY = os.environ.get("QWEN_KEY")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists() and _BUNDLED_SYSTEM_PROMPT_PATH.exists():
        shutil.copy2(_BUNDLED_SYSTEM_PROMPT_PATH, SYSTEM_PROMPT_PATH)
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text().strip()
    return "You are a helpful terminal assistant with access to tools."


def run_first_time_setup() -> dict | None:
    """Interactively prompt for QWEN_URL/QWEN_KEY directly in the terminal
    and write them to .env in CONFIG_DIR, instead of just erroring out and
    telling the user to go create the file themselves by hand.

    Only ever called when QWEN_URL/QWEN_KEY aren't already set (see
    session.run()) - never overwrites an existing .env. Returns the entered
    values as a dict ({"QWEN_URL": ..., "QWEN_KEY": ...}), or None if the
    user backed out (EOF/Ctrl-C, or left a required field blank) - the
    caller falls back to the normal "not set" error in that case.

    QWEN_KEY is read via getpass (hidden input, same trust model as
    tools/skill_env.py's credential prompts) since it's a real secret;
    QWEN_URL is not.

    Also updates this module's own QWEN_URL/QWEN_KEY/QWEN_MODEL globals -
    every other module that needs these reads them as `config.QWEN_URL`
    etc. at call time (never `from skull.config import QWEN_URL`, which
    freezes a copy at import time - the exact bug this fixes: client.py,
    compaction.py, and memory_supersede.py each had their own frozen,
    still-empty copy even after this function ran and the wizard reported
    success, sending every request to a URL-less endpoint).
    """
    global QWEN_URL, QWEN_KEY, QWEN_MODEL

    env_path = CONFIG_DIR / ".env"
    print(f"\n{BOLD}Welcome to skull!{RESET} No config found yet at {env_path}.")
    print("Let's set it up now - this only happens once.\n")

    try:
        url = input("Qwen-compatible chat-completions endpoint URL (QWEN_URL): ").strip()
        if not url:
            print("No URL entered - skipping setup.", file=sys.stderr)
            return None

        key = getpass.getpass("API key for that endpoint (QWEN_KEY, input hidden): ").strip()
        if not key:
            print("No key entered - skipping setup.", file=sys.stderr)
            return None

        model = input(f"Model name (QWEN_MODEL) [{QWEN_MODEL}]: ").strip() or QWEN_MODEL
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.", file=sys.stderr)
        return None

    lines = [f"QWEN_URL={url}", f"QWEN_KEY={key}"]
    if model != QWEN_MODEL:
        lines.append(f"QWEN_MODEL={model}")
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 - same as skills.env

    print(f"\nSaved to {env_path}. (Edit that file directly any time - e.g. to add E2B_API_KEY.)\n")

    QWEN_URL = url.rstrip("/")
    QWEN_KEY = key
    QWEN_MODEL = model
    return {"QWEN_URL": QWEN_URL, "QWEN_KEY": QWEN_KEY, "QWEN_MODEL": QWEN_MODEL}
