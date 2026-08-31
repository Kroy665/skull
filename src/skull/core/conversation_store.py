"""Per-directory conversation persistence: save the active message list to
disk keyed by the working directory `skull` was launched from, and load it
back automatically the next time `skull` runs from that same directory.

This is distinct from storage/store.py's conversation memory (a fuzzy
fact-recall vector store used to seed context/suggestions across ANY
directory) - this module persists the literal, ordered message list so a
conversation thread can resume exactly where it left off, scoped to one
directory. The two coexist and are not merged.

Keyed by a hash of the resolved absolute path (not the path text itself) so
nested/deep directories don't produce unwieldy or filesystem-unsafe
filenames.
"""

import hashlib
import json
from pathlib import Path

from skull.config import CONVERSATIONS_DIR


def _key_for(cwd: str) -> str:
    resolved = str(Path(cwd).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:16]


def _path_for(cwd: str) -> Path:
    return CONVERSATIONS_DIR / f"{_key_for(cwd)}.json"


def load(cwd: str) -> list | None:
    """Return the saved message list for `cwd`, or None if there isn't one
    (or it's unreadable/corrupt - never crash startup over a bad save
    file)."""
    path = _path_for(cwd)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    return messages


def save(cwd: str, messages: list) -> None:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(cwd)
    path.write_text(json.dumps({"cwd": str(Path(cwd).resolve()), "messages": messages}))


def clear(cwd: str) -> None:
    path = _path_for(cwd)
    path.unlink(missing_ok=True)
