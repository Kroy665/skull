"""Throwaway code execution in an E2B cloud sandbox, for testing an approach
before it becomes a permanent skill (via create_skill).

Unlike skills_manager, nothing here is registered as a tool or persisted
across sessions. Execution happens in an isolated E2B sandbox - not on this
machine - so it has no access to local files, network state, or credentials
beyond what's explicitly passed in.

Requires E2B_API_KEY in the environment (see .env).
"""

import os
import threading
from pathlib import Path

import requests
from e2b_code_interpreter import Sandbox

from skull.tools.permission import ask_permission

RUN_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 4000
MAX_FILE_READ_CHARS = 20000
# Hard ceiling regardless of what max_chars the model requests - see the
# matching constant/comment in skull.tools.files for why this exists (a
# single oversized tool result can blow the context window in one shot,
# something compaction can't undo after the fact).
MAX_FILE_READ_CHARS_CEILING = 40000

_sandbox = None
_sandbox_lock = threading.Lock()


def _truncate(text: str) -> tuple:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return text[:MAX_OUTPUT_CHARS], True


def _get_sandbox() -> Sandbox:
    """Lazily create one sandbox per process and reuse it across calls, so
    state (variables, installed packages) persists within a session."""
    global _sandbox
    with _sandbox_lock:
        if _sandbox is None:
            api_key = os.environ.get("E2B_API_KEY")
            if not api_key:
                raise RuntimeError("E2B_API_KEY is not set (add it to .env)")
            _sandbox = Sandbox.create(api_key=api_key)
        return _sandbox


def clear_scratch() -> None:
    """Reset the sandbox at the start of a new session so state (variables,
    files) from a previous run doesn't leak in. No-op if none is running."""
    global _sandbox
    with _sandbox_lock:
        if _sandbox is not None:
            try:
                _sandbox.kill()
            except Exception:
                pass
            _sandbox = None


def run_python(code: str, timeout: int = RUN_TIMEOUT_SECONDS) -> dict:
    if not code or not code.strip():
        return {"error": "no code provided"}
    timeout = max(1, min(int(timeout), 60))

    try:
        sbx = _get_sandbox()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        execution = sbx.run_code(code, timeout=timeout)
    except Exception as e:
        return {"error": f"sandbox execution failed: {e}"}

    stdout = "".join(execution.logs.stdout)
    stderr = "".join(execution.logs.stderr)
    stdout, stdout_truncated = _truncate(stdout)
    stderr, stderr_truncated = _truncate(stderr)

    result = {
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "stderr": stderr,
        "stderr_truncated": stderr_truncated,
    }

    if execution.error:
        result["error_name"] = execution.error.name
        result["error_value"] = execution.error.value
        result["traceback"] = execution.error.traceback

    if execution.results:
        result["results"] = [str(r) for r in execution.results]

    return result


def sandbox_read_file(path: str, max_chars: int = MAX_FILE_READ_CHARS) -> dict:
    if not path or not path.strip():
        return {"error": "no path provided"}
    max_chars = max(200, min(int(max_chars), MAX_FILE_READ_CHARS_CEILING))

    try:
        sbx = _get_sandbox()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        text = sbx.files.read(path)
    except Exception as e:
        return {"error": f"failed to read {path} from sandbox: {e}"}

    truncated = len(text) > max_chars
    return {"path": path, "content": text[:max_chars], "truncated": truncated}


def sandbox_write_file(path: str, content: str) -> dict:
    if not path or not path.strip():
        return {"error": "no path provided"}

    try:
        sbx = _get_sandbox()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        sbx.files.write(path, content)
    except Exception as e:
        return {"error": f"failed to write {path} in sandbox: {e}"}

    return {"status": "written", "path": path, "bytes": len(content)}


def download_from_sandbox(sandbox_path: str, local_path: str, reason: str = "") -> dict:
    """Copy a file directly from the sandbox filesystem to the user's own
    machine, as raw bytes - no base64/text round-trip through the
    conversation. Requires interactive y/n approval like write_file, since
    this writes to the real local filesystem.

    This exists specifically so binary output (a generated .docx, image,
    zip, etc.) never needs to be smuggled through sandbox_read_file as text -
    that path has a hard size ceiling (see MAX_FILE_READ_CHARS_CEILING) and
    even under it would burn enormous context for no benefit, since none of
    that content is meant to be read by the model anyway.
    """
    if not sandbox_path or not sandbox_path.strip():
        return {"error": "no sandbox_path provided"}
    if not local_path or not local_path.strip():
        return {"error": "no local_path provided"}

    try:
        sbx = _get_sandbox()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        url = sbx.download_url(sandbox_path)
    except Exception as e:
        return {"error": f"failed to get download URL for {sandbox_path}: {e}"}

    dest = Path(local_path).expanduser()
    if dest.exists() and dest.is_dir():
        return {"error": f"{dest} is a directory, cannot write a file there"}

    action = "copy a file from the sandbox to this machine"
    detail = [f"sandbox:{sandbox_path}  ->  {dest}"]
    if not ask_permission(action, detail, reason):
        return {"error": "denied by user", "denied": True}

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"failed to download {sandbox_path}: {e}"}

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    except Exception as e:
        return {"error": f"failed to write {dest}: {e}"}

    return {"status": "downloaded", "local_path": str(dest), "bytes": len(resp.content)}


def sandbox_list_directory(path: str = "/") -> dict:
    try:
        sbx = _get_sandbox()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        entries = sbx.files.list(path or "/")
    except Exception as e:
        return {"error": f"failed to list {path} in sandbox: {e}"}

    return {
        "path": path,
        "entries": [
            {"name": e.name, "type": "directory" if e.type.value == "dir" else "file"}
            for e in entries
        ],
    }
