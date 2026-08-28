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

from e2b_code_interpreter import Sandbox

RUN_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 4000

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
