"""Environment configuration, paths, and terminal color constants.

Runs as a real installed command (`skull`, from any directory), not tied
to a project checkout - so per-user data (skills/, memory/, pipelines/,
skills.env, the .env with LLM_KEY, and an editable copy of
SYSTEM_PROMPT.md) lives in a standard per-user config directory, not next
to the source code. Everything here is found the same way regardless of
where the `skull` command is invoked from.
"""

import getpass
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import questionary
import requests
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

# .env (LLM_KEY, LLM_URL, LLM_MODEL, E2B_API_KEY) lives in the per-user
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

# Presets for run_first_time_setup()'s provider picker - a base URL to
# pre-fill (every provider here exposes an OpenAI-compatible
# /v1/chat/completions + /v1/models pair - the whole reason this is scoped
# to these providers instead of each one's own native SDK/wire format, see
# core/client.py) and a web search query used to figure out which of that
# endpoint's models are actually current (see _rank_models_by_web_search) -
# the /models list itself is unranked and mixes in non-chat models.
PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "search_query": "latest OpenAI GPT model",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "search_query": "latest Google Gemini model",
    },
}

# Matches model-id-shaped tokens in free-text web search results (titles/
# snippets) - e.g. "GPT-5.6", "gemini-2.5-pro" - so they can be checked
# against a provider's real /v1/models list. Deliberately generic (not
# hardcoded to today's specific model names) so it keeps working as
# providers release new versions.
_MODEL_TOKEN_RE = re.compile(r"\b(gpt|gemini|o)[-.]?[0-9][a-z0-9.\-]*", re.IGNORECASE)

LLM_URL = (os.environ.get("LLM_URL") or "").rstrip("/")
# No hardcoded default: this app is no longer tied to one specific
# self-hosted Qwen model, and a stale hardcoded model name for a
# multi-provider setup would silently break requests as soon as the user
# picked a different provider without also remembering to override this.
LLM_MODEL = os.environ.get("LLM_MODEL") or ""
LLM_KEY = os.environ.get("LLM_KEY")

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


def list_available_models(base_url: str, api_key: str) -> list:
    """Fetch the model id list from an OpenAI-compatible /v1/models (or
    equivalent) endpoint, for the setup wizard's model picker. Returns []
    on any failure (bad key, unreachable host, unexpected shape) - the
    wizard falls back to asking for a model name to type in by hand
    rather than blocking setup on this call succeeding.

    No filtering by name/capability: OpenAI's own list mixes chat models
    in with embeddings/whisper/tts/etc, but a hand-rolled allow/deny list
    would just as easily hide a legitimately new chat model under a stale
    heuristic - showing everything and letting the user pick (or type
    their own) is the more robust default.
    """
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted(m["id"] for m in data if "id" in m)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return []


def _rank_models_by_web_search(models: list, search_query: str) -> list:
    """Reorder `models` so entries that look current per a live web search
    come first, without ever inventing a model id that isn't in the real
    list. Search snippets are free text ("GPT-5.6 Sol"), never treated as
    literal API ids - only used to decide which of the REAL /v1/models
    entries to surface first.

    Falls back to returning `models` unchanged (just reverse-sorted, which
    tends to put newer dated/versioned ids first) on any search failure -
    this is a ranking nicety, not something setup should ever block on.
    """
    fallback = sorted(models, reverse=True)
    try:
        from skull.tools.web import web_search

        result = web_search(search_query, count=8)
    except Exception:
        return fallback
    if "error" in result:
        return fallback

    mentioned = set()
    for r in result.get("results", []):
        text = f"{r.get('title', '')} {r.get('snippet', '')}"
        for m in _MODEL_TOKEN_RE.finditer(text):
            mentioned.add(m.group(0).lower().replace(" ", "-"))

    if not mentioned:
        return fallback

    def looks_mentioned(model_id: str) -> bool:
        low = model_id.lower()
        return any(token in low or low in token for token in mentioned)

    ranked = [m for m in models if looks_mentioned(m)]
    ranked += [m for m in sorted(models, reverse=True) if m not in ranked]
    return ranked


def run_first_time_setup() -> dict | None:
    """Interactively prompt for an LLM provider/endpoint/key/model directly
    in the terminal and write them to .env in CONFIG_DIR, instead of just
    erroring out and telling the user to go create the file themselves by
    hand.

    Only ever called when LLM_URL/LLM_KEY aren't already set (see
    session.run()) - never overwrites an existing .env. Returns the entered
    values as a dict ({"LLM_URL": ..., "LLM_KEY": ...}), or None if the
    user backed out (EOF/Ctrl-C, or left a required field blank) - the
    caller falls back to the normal "not set" error in that case.

    Offers an arrow-key provider picker (OpenAI, Gemini, or a custom URL -
    see PROVIDER_PRESETS) that pre-fills the right base URL, since every
    provider offered here exposes an OpenAI-compatible
    /v1/chat/completions endpoint (this app speaks that one wire format
    only - see core/client.py - not each provider's own native SDK).

    After the key is entered, fetches the endpoint's real /v1/models list
    (list_available_models), ranks it using a live web search for what's
    actually current (_rank_models_by_web_search - the /models list itself
    has no reliable "release date" or "is this a chat model" signal), and
    presents it as an arrow-key select (questionary) - typing filters the
    list live. Falls back to typing a model name by hand if the /models
    call itself fails.

    LLM_KEY is read via getpass (hidden input, same trust model as
    tools/skill_env.py's credential prompts) since it's a real secret;
    LLM_URL is not.

    Also updates this module's own LLM_URL/LLM_KEY/LLM_MODEL globals -
    every other module that needs these reads them as `config.LLM_URL`
    etc. at call time (never `from skull.config import LLM_URL`, which
    freezes a copy at import time - the exact bug this fixes: client.py,
    compaction.py, and memory_supersede.py each had their own frozen,
    still-empty copy even after this function ran and the wizard reported
    success, sending every request to a URL-less endpoint).
    """
    global LLM_URL, LLM_KEY, LLM_MODEL

    env_path = CONFIG_DIR / ".env"
    print(f"\n{BOLD}Welcome to skull!{RESET} No config found yet at {env_path}.")
    print("Let's set it up now - this only happens once.\n")

    try:
        provider = questionary.select(
            "Which provider are you using?",
            choices=[
                questionary.Choice("OpenAI", value="openai"),
                questionary.Choice("Gemini", value="gemini"),
                questionary.Choice("Custom / self-hosted (any OpenAI-compatible endpoint)", value="custom"),
            ],
        ).ask()
        if provider is None:  # Ctrl-C/Ctrl-D inside questionary
            print("\nSetup cancelled.", file=sys.stderr)
            return None

        if provider == "custom":
            url = questionary.text("Chat-completions endpoint base URL (LLM_URL):").ask()
            search_query = None
        else:
            url = PROVIDER_PRESETS[provider]["base_url"]
            search_query = PROVIDER_PRESETS[provider]["search_query"]
        if not url:
            print("No URL entered - skipping setup.", file=sys.stderr)
            return None
        url = url.rstrip("/")

        key = getpass.getpass("API key for that endpoint (LLM_KEY, input hidden): ").strip()
        if not key:
            print("No key entered - skipping setup.", file=sys.stderr)
            return None

        model = None
        print("Looking up available models...")
        models = list_available_models(url, key)
        if models:
            if search_query:
                print("Checking the web for which of these are current...")
                models = _rank_models_by_web_search(models, search_query)
            else:
                models = sorted(models, reverse=True)
            model = questionary.select(
                f"Pick a model ({len(models)} available - type to filter):",
                choices=models,
                use_search_filter=True,
                use_jk_keys=False,
            ).ask()
            if model is None:
                print("\nSetup cancelled.", file=sys.stderr)
                return None
        else:
            print("Couldn't fetch a model list for that endpoint/key.")
            model = questionary.text("Model name (LLM_MODEL):").ask()
            if not model:
                print("No model entered - skipping setup.", file=sys.stderr)
                return None
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.", file=sys.stderr)
        return None

    lines = [f"LLM_URL={url}", f"LLM_KEY={key}", f"LLM_MODEL={model}"]
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 - same as skills.env

    print(f"\nSaved to {env_path}. (Edit that file directly any time - e.g. to add E2B_API_KEY.)\n")

    LLM_URL = url
    LLM_KEY = key
    LLM_MODEL = model
    return {"LLM_URL": LLM_URL, "LLM_KEY": LLM_KEY, "LLM_MODEL": LLM_MODEL}
