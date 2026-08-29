"""Secrets storage for skills (API keys, passwords, tokens), so a skill
needing a credential never has to get it by asking the model to relay it
through chat - which is exactly what happened in practice: the model asked
the user to paste an SMTP password directly into the conversation, landing
it in plain text in the terminal, conversation history, and memory/
(everything a skill's own kwargs go through the model, which sees, can log,
and could echo back).

Design: a skill's code reads a secret via `get_env(name)` at call time,
never through `kwargs` - the value never passes through the model at all.
The model can ask the user to SET a value (`request_skill_env`), but the
actual value is captured by a direct getpass() prompt in the terminal, seen
only by the user, and the model's tool result never includes it - just
whether it's now set. `has_env`/`list_missing_env` let the model check
what's configured without ever seeing a value.

Stored in SKILLS_ENV_PATH (skills.env at the project root, gitignored,
separate from the app's own .env) as plain KEY=VALUE lines, permissioned
0600 (owner read/write only) - the same trust model .env already uses for
QWEN_KEY/E2B_API_KEY, extended to per-skill secrets instead of just app
config.
"""

import getpass
import stat

from skull.config import SKILLS_ENV_PATH
from skull.ui.output import tprint

ENV_VAR_NAME_RE_MSG = "must be an UPPER_SNAKE_CASE name, e.g. SMTP_PASSWORD"


def is_valid_env_name(name: str) -> bool:
    return bool(name) and name.replace("_", "").isalnum() and name[0].isalpha() and name.upper() == name


def _load() -> dict:
    if not SKILLS_ENV_PATH.exists():
        return {}
    values = {}
    for line in SKILLS_ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value
    return values


def _save(values: dict) -> None:
    lines = [f"{key}={value}" for key, value in sorted(values.items())]
    SKILLS_ENV_PATH.write_text("\n".join(lines) + ("\n" if lines else ""))
    try:
        SKILLS_ENV_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: owner read/write only
    except OSError:
        pass  # best-effort on platforms without POSIX permission bits


def get_env(name: str) -> str | None:
    """For skill code to call at runtime: `from skull.tools.skill_env import
    get_env` then `get_env("SMTP_PASSWORD")`. Returns None if not set."""
    return _load().get(name)


def has_env(name: str) -> bool:
    return name in _load()


def list_missing_env(names: list) -> list:
    """Given a list of env var names a skill declares it needs, return the
    subset not currently set - for the model to check before attempting to
    call a skill that needs credentials, and to know which ones to ask the
    user to set."""
    current = _load()
    return [n for n in names if n not in current]


def request_skill_env(name: str, reason: str = "") -> dict:
    """Prompt the user directly (in the terminal, via getpass - never
    through the model) to set an env var value. The value is captured
    locally and stored in skills.env; the model's tool result never
    contains it, only whether it's now set. Returns {"status": "set"} or
    {"status": "cancelled"} (empty input) - never the value, under any
    circumstances, even on success."""
    if not is_valid_env_name(name):
        return {"error": f"invalid env var name {name!r} - {ENV_VAR_NAME_RE_MSG}"}

    tprint()
    tprint(f"A skill needs the environment variable \033[1m{name}\033[0m to be set.")
    if reason:
        tprint(f"  reason: {reason}")
    tprint("Input is hidden - nothing you type here is shown to the model or saved to conversation history.")

    try:
        value = getpass.getpass(f"Enter a value for {name} (leave blank to cancel): ")
    except (EOFError, KeyboardInterrupt):
        return {"status": "cancelled", "name": name}

    if not value:
        return {"status": "cancelled", "name": name}

    values = _load()
    values[name] = value
    _save(values)
    return {"status": "set", "name": name}


def clear_env(name: str) -> dict:
    """Remove a previously set env var - use when a credential is revoked
    or rotated out."""
    values = _load()
    if name not in values:
        return {"error": f"{name} is not currently set"}
    del values[name]
    _save(values)
    return {"status": "cleared", "name": name}
