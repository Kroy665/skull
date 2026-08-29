"""Tests for tools/shell.py - run_command's approval gate and background
process management, focused on the real gap found via code review: the log
file handle opened for a background process was only ever closed by
clear_background_processes(), which nothing in the codebase actually
calls - stop_background_command's own success path, its "already exited"
path, and list_background_commands' opportunistic detection of a process
that exited on its own all left the handle open indefinitely, leaking one
file descriptor per background command started for the rest of the
process's lifetime."""

import time

from skull.tools import shell


def _start_real_background_process(monkeypatch, command: str = "sleep 30"):
    monkeypatch.setattr(shell, "ask_permission", lambda *a, **k: True)
    result = shell.run_command(command, reason="test", background=True)
    assert result["status"] == "started"
    return result["pid"]


def test_run_command_requires_approval(monkeypatch):
    monkeypatch.setattr(shell, "ask_permission", lambda *a, **k: False)
    result = shell.run_command("echo hi", reason="test")
    assert result == {"error": "denied by user", "denied": True}


def test_run_command_foreground_returns_output(monkeypatch):
    monkeypatch.setattr(shell, "ask_permission", lambda *a, **k: True)
    result = shell.run_command("echo hello", reason="test")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


def test_run_command_rejects_empty_command(monkeypatch):
    result = shell.run_command("", reason="test")
    assert "error" in result


def test_background_command_starts_and_is_listed(monkeypatch):
    pid = _start_real_background_process(monkeypatch)
    try:
        result = shell.list_background_commands()
        pids = [p["pid"] for p in result["processes"]]
        assert pid in pids
        entry = next(p for p in result["processes"] if p["pid"] == pid)
        assert entry["running"] is True
    finally:
        shell.stop_background_command(pid)


def test_stop_background_command_closes_the_log_handle(monkeypatch):
    """The exact real gap: stop_background_command's success path never
    closed info['log_handle'] before this fix."""
    pid = _start_real_background_process(monkeypatch)
    info = shell._background_processes[pid]
    assert info["log_handle"].closed is False

    result = shell.stop_background_command(pid)
    assert result["status"] == "stopped"
    assert info["log_handle"].closed is True


def test_stop_background_command_on_already_exited_process_closes_handle(monkeypatch):
    """The "already exited" early-return path also never closed the handle
    before this fix - a process that finished on its own (e.g. a one-shot
    build script) between being started and being explicitly stopped."""
    pid = _start_real_background_process(monkeypatch, command="true")  # exits immediately
    info = shell._background_processes[pid]

    for _ in range(50):
        if info["process"].poll() is not None:
            break
        time.sleep(0.05)
    assert info["process"].poll() is not None, "test process did not exit in time"

    result = shell.stop_background_command(pid)
    assert result["status"] == "already exited"
    assert info["log_handle"].closed is True


def test_list_background_commands_closes_handle_for_self_exited_process(monkeypatch):
    """A background process that crashes/finishes on its own (never
    explicitly stopped) must still have its log handle closed the next
    time its status is checked - otherwise it leaks for the rest of this
    process's lifetime with no code path that would ever close it."""
    pid = _start_real_background_process(monkeypatch, command="true")
    info = shell._background_processes[pid]

    for _ in range(50):
        if info["process"].poll() is not None:
            break
        time.sleep(0.05)
    assert info["process"].poll() is not None, "test process did not exit in time"
    assert info["log_handle"].closed is False

    shell.list_background_commands()
    assert info["log_handle"].closed is True


def test_stop_background_command_missing_pid_returns_error():
    result = shell.stop_background_command(999999999)
    assert "error" in result


def test_read_background_log_missing_pid_returns_error():
    result = shell.read_background_log(999999999)
    assert "error" in result


def test_read_background_log_returns_output(monkeypatch):
    pid = _start_real_background_process(monkeypatch, command="echo background-output")
    try:
        for _ in range(50):
            if shell._background_processes[pid]["process"].poll() is not None:
                break
            time.sleep(0.05)
        result = shell.read_background_log(pid)
        assert "background-output" in result["log"]
    finally:
        shell.stop_background_command(pid)
