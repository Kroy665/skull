"""Shell command execution, gated by an interactive y/n confirmation.

Unlike run_python (isolated in an E2B sandbox) or self-created skills (plain
Python, no shell access), this runs a real command directly on the user's
machine with their full permissions - the largest blast radius of any tool
here. Every invocation must be explicitly approved before it executes.

Long-running processes (dev servers, watchers) can be started in the
background with background=True: the process is detached and its output
redirected to a log file, and run_command returns immediately instead of
blocking until it exits (which, for a server, is never - the original
foreground call just hung until timeout). list_background_commands() and
stop_background_command() manage what's running.
"""

import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from skull.tools.permission import ask_permission

RUN_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4000
LOG_TAIL_CHARS = 2000

_LOG_DIR = Path(tempfile.gettempdir()) / "skull-bg-logs"

# pid -> {"command": str, "log_file": Path, "process": subprocess.Popen}
_background_processes = {}
_background_lock = threading.Lock()


def _truncate(text: str) -> tuple:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return text[:MAX_OUTPUT_CHARS], True


def _start_background(command: str) -> dict:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / f"{int(time.time() * 1000)}.log"
    log_handle = open(log_file, "w")

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from this process's terminal/signals
        )
    except Exception as e:
        log_handle.close()
        return {"error": f"failed to start background command: {e}"}

    with _background_lock:
        _background_processes[proc.pid] = {
            "command": command,
            "log_file": log_file,
            "process": proc,
            "log_handle": log_handle,
        }

    return {
        "status": "started",
        "pid": proc.pid,
        "log_file": str(log_file),
        "note": (
            "The command is running in the background and this call returned "
            "immediately. Use list_background_commands to check its status, "
            "read its log file to see output, or stop_background_command to "
            "kill it."
        ),
    }


def run_command(
    command: str,
    reason: str = "",
    timeout: int = RUN_TIMEOUT_SECONDS,
    background: bool = False,
) -> dict:
    if not command or not command.strip():
        return {"error": "no command provided"}
    timeout = max(1, min(int(timeout), 120))

    action = "run a shell command in the background" if background else "run a shell command"
    if not ask_permission(action, [command], reason):
        return {"error": "denied by user", "denied": True}

    if background:
        return _start_background(command)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": (
                f"command timed out after {timeout}s - if this is meant to keep "
                "running (a dev server, watcher, etc.), call run_command again "
                "with background=true instead"
            )
        }
    except Exception as e:
        return {"error": f"failed to run command: {e}"}

    stdout, stdout_truncated = _truncate(proc.stdout)
    stderr, stderr_truncated = _truncate(proc.stderr)

    return {
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "stderr": stderr,
        "stderr_truncated": stderr_truncated,
    }


def _close_log_handle_if_open(info: dict) -> None:
    handle = info.get("log_handle")
    if handle is not None and not handle.closed:
        try:
            handle.close()
        except Exception:
            pass


def list_background_commands() -> dict:
    with _background_lock:
        entries = list(_background_processes.items())

    results = []
    for pid, info in entries:
        proc = info["process"]
        exit_code = proc.poll()
        if exit_code is not None:
            # The process exited on its own (crashed, or finished a
            # one-shot job) - nothing will ever write to this log file
            # again, so close the handle now rather than leaking it for
            # the rest of the process's lifetime. Real gap this closes:
            # previously only stop_background_command's (unreachable, see
            # its own fix) path and the never-called
            # clear_background_processes() ever closed a log handle, so a
            # background command that simply exited on its own leaked its
            # file descriptor for as long as this process kept running.
            _close_log_handle_if_open(info)
        results.append(
            {
                "pid": pid,
                "command": info["command"],
                "running": exit_code is None,
                "exit_code": exit_code,
                "log_file": str(info["log_file"]),
            }
        )
    return {"processes": results}


def read_background_log(pid: int, tail_chars: int = LOG_TAIL_CHARS) -> dict:
    with _background_lock:
        info = _background_processes.get(pid)
    if info is None:
        return {"error": f"no background process with pid {pid}"}

    try:
        text = info["log_file"].read_text(errors="replace")
    except FileNotFoundError:
        return {"error": "log file not found (may have been cleaned up)"}

    tail_chars = max(200, min(int(tail_chars), 20000))
    truncated = len(text) > tail_chars
    return {"pid": pid, "log": text[-tail_chars:], "truncated": truncated}


def stop_background_command(pid: int) -> dict:
    with _background_lock:
        info = _background_processes.get(pid)
    if info is None:
        return {"error": f"no background process with pid {pid}"}

    proc = info["process"]
    exit_code = proc.poll()
    if exit_code is not None:
        _close_log_handle_if_open(info)
        return {"status": "already exited", "pid": pid, "exit_code": exit_code}

    # The process was started with shell=True + start_new_session=True, so
    # `pid` is the shell wrapper, and it may have already exec'd or forked
    # into the real command (e.g. a dev server) as a distinct process in the
    # same process group. Terminating just the tracked Popen object can
    # leave that real process running and orphaned - signal the whole
    # group instead, since start_new_session made it the group leader.
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # group already gone
    except Exception as e:
        return {"error": f"failed to stop process group {pid}: {e}"}

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return {"error": f"process group {pid} did not exit even after SIGKILL"}

    _close_log_handle_if_open(info)
    return {"status": "stopped", "pid": pid}


def clear_background_processes() -> None:
    """Stop every still-running background process. Called at session start
    (and could be called at exit) so nothing from a previous session lingers
    or leaks indefinitely."""
    with _background_lock:
        entries = list(_background_processes.items())
    for pid, info in entries:
        proc = info["process"]
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            info["log_handle"].close()
        except Exception:
            pass
    with _background_lock:
        _background_processes.clear()
