"""Local filesystem access, on the user's real machine (not the E2B sandbox
- see skull.tools.sandbox for that).

Reads are unguarded, same trust level as web_search: low risk, nothing to
approve. Writes require the same interactive y/n approval as run_command,
since overwriting or creating a file can destroy existing work - there is
no sandboxing here, paths can be anywhere on the machine, consistent with
run_command already having unrestricted shell access.
"""

from pathlib import Path

from skull.tools.permission import ask_permission

MAX_READ_CHARS = 20000
# Hard ceiling regardless of what max_chars the caller (model) requests. This
# exists because a single oversized tool result can blow the context window
# in one shot - the compaction system only manages accumulation of history
# over multiple turns, it can't shrink a result that's already been read and
# is about to be sent in the very next request. ~40000 chars is ~11,400
# tokens - large enough for a real file, small enough that no single call
# can approach the 32K window on its own.
MAX_READ_CHARS_CEILING = 40000
MAX_WRITE_PREVIEW_CHARS = 300


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


def read_file(path: str, max_chars: int = MAX_READ_CHARS) -> dict:
    if not path or not path.strip():
        return {"error": "no path provided"}
    max_chars = max(200, min(int(max_chars), MAX_READ_CHARS_CEILING))

    p = _resolve(path)
    if not p.exists():
        return {"error": f"no such file: {p}"}
    if p.is_dir():
        return {"error": f"{p} is a directory, not a file - use list_directory instead"}

    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": f"failed to read {p}: {e}"}

    truncated = len(text) > max_chars
    return {"path": str(p), "content": text[:max_chars], "truncated": truncated}


def list_directory(path: str = ".") -> dict:
    p = _resolve(path or ".")
    if not p.exists():
        return {"error": f"no such path: {p}"}
    if not p.is_dir():
        return {"error": f"{p} is not a directory"}

    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except Exception as e:
        return {"error": f"failed to list {p}: {e}"}

    return {
        "path": str(p),
        "entries": [
            {"name": e.name, "type": "directory" if e.is_dir() else "file"}
            for e in entries
        ],
    }


def write_file(path: str, content: str, mode: str = "overwrite", reason: str = "") -> dict:
    if not path or not path.strip():
        return {"error": "no path provided"}
    if mode not in ("overwrite", "append"):
        return {"error": "mode must be 'overwrite' or 'append'"}

    p = _resolve(path)
    existed = p.exists()
    if existed and p.is_dir():
        return {"error": f"{p} is a directory, cannot write to it as a file"}

    preview = content if len(content) <= MAX_WRITE_PREVIEW_CHARS else (
        content[:MAX_WRITE_PREVIEW_CHARS] + f"… ({len(content)} chars total)"
    )
    verb = "append to" if mode == "append" else ("overwrite" if existed else "create")
    action = f"{verb} this file on your machine"
    detail = [str(p), "---", preview]
    if not ask_permission(action, detail, reason):
        return {"error": "denied by user", "denied": True}

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with p.open("a") as f:
                f.write(content)
        else:
            p.write_text(content)
    except Exception as e:
        return {"error": f"failed to write {p}: {e}"}

    return {"status": "written", "path": str(p), "mode": mode, "bytes": len(content)}
