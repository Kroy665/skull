"""Raw-terminal line input with an inline greyed-out "ghost text" suggestion.

The suggestion is rendered dimmed right after the cursor. It vanishes the
instant the user presses any real key and is never part of what gets sent
unless explicitly accepted by pressing Right-arrow at an empty line.

The suggestion can arrive *after* the prompt is already showing (it's
computed by a background thread), so the read loop polls for it via
`select()` with a short timeout and live-redraws if it changes - not just a
one-time snapshot taken when the prompt starts.

POSIX only (termios/tty) - falls back to plain input() if unavailable
(e.g. on Windows, or when stdin isn't a real tty).
"""

import os
import re
import sys

try:
    import termios
    import tty
    import select
    RAW_MODE_AVAILABLE = True
except ImportError:
    RAW_MODE_AVAILABLE = False

DIM = "\033[2m"
RESET = "\033[0m"

BACKSPACE = {"\x7f", "\x08"}
CTRL_C = "\x03"
CTRL_D = "\x04"
TAB = "\t"
ENTER = {"\r", "\n"}

POLL_INTERVAL_SECONDS = 0.15

_ANSI_RE = re.compile(r"\033\[[0-9;]*[a-zA-Z]")


def _visible_len(text: str) -> int:
    """Length of `text` as it will actually render on screen, ignoring ANSI
    escape codes (color, dim, reset, etc.) which take up bytes but no
    columns - needed for correct cursor-position math."""
    return len(_ANSI_RE.sub("", text))


def _read_byte(fd) -> str:
    """Read exactly one byte directly from the fd (never through
    sys.stdin's buffered TextIOWrapper - mixing select() on the raw fd with
    stdin's own internal buffering causes select to be blind to bytes
    TextIOWrapper already over-read into its buffer, which drops or
    corrupts fast multi-byte sequences like arrow keys)."""
    return os.read(fd, 1).decode(errors="replace")


def _read_key_nonblocking(fd, timeout: float):
    """Wait up to `timeout` seconds for a keypress. Returns None on timeout
    (no key available yet), or the resolved key token/char otherwise.
    Arrow keys resolve to 'UP'/'DOWN'/'RIGHT'/'LEFT'."""
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None

    ch = _read_byte(fd)
    if ch != "\x1b":
        return ch

    # Escape sequence: expect '[' then a letter. These arrive as a fast,
    # atomic burst right after ESC, so a short bounded wait is enough to
    # distinguish a real arrow key from a lone ESC keypress.
    ready, _, _ = select.select([fd], [], [], 0.05)
    if not ready:
        return "ESC"
    seq = _read_byte(fd)
    if seq != "[":
        return ""  # unrecognized escape - ignore
    code = _read_byte(fd)
    return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(code, "")


def prompt_with_ghost(prompt_label: str, get_suggestion, history: list = None) -> str:
    """Show `prompt_label`, then the live suggestion (from calling
    `get_suggestion()`) dimmed as ghost text whenever the line is empty.
    `get_suggestion` is polled continuously so a suggestion that arrives
    after the prompt is already showing still appears, without waiting for
    a keystroke. Returns the final line the user submitted (never includes
    unaccepted ghost text).

    `get_suggestion` may also be a plain string for a one-shot fixed value.

    `history`, if given, is a list of previous inputs (oldest first, same
    list object the caller keeps appending to). Up/Down walk backward and
    forward through it, shell-style: Up from a fresh line jumps to the most
    recent entry; the in-progress line is preserved as a "draft" so pressing
    Down back past the oldest-visited entry returns exactly what was being
    typed before Up was first pressed, not an empty line.
    """
    get = get_suggestion if callable(get_suggestion) else (lambda: get_suggestion)
    history = history if history is not None else []

    if not RAW_MODE_AVAILABLE or not sys.stdin.isatty():
        # Fallback: plain input(), no ghost text, but still functional.
        try:
            return input(prompt_label)
        except EOFError:
            raise

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf = []       # list of chars the user has actually typed
    cursor = 0     # index into buf
    current_suggestion = get() or ""
    prompt_width = _visible_len(prompt_label)

    # History navigation state. history_index is None while on the "live"
    # (not-yet-submitted) line; otherwise it's an index into history, most
    # recent = len(history)-1. draft holds what was being typed before the
    # user first pressed Up, so Down can restore it past the newest entry.
    history_index = None
    draft = []

    def redraw():
        line = "".join(buf)
        ghost = current_suggestion if (not buf and current_suggestion) else ""
        sys.stdout.write("\r\033[K")
        sys.stdout.write(prompt_label + line)
        if ghost:
            sys.stdout.write(f"{DIM}{ghost}{RESET}")
        target_col = prompt_width + cursor
        sys.stdout.write(f"\r\033[{target_col}C" if target_col else "\r")
        sys.stdout.flush()

    def load_history_entry(index):
        nonlocal buf, cursor, history_index
        history_index = index
        buf = list(history[index])
        cursor = len(buf)

    try:
        tty.setraw(fd)
        redraw()
        while True:
            key = _read_key_nonblocking(fd, POLL_INTERVAL_SECONDS)

            if key is None:
                # Idle tick: check whether the suggestion changed (e.g. the
                # background prediction just finished) and redraw if so.
                if not buf:
                    fresh = get() or ""
                    if fresh != current_suggestion:
                        current_suggestion = fresh
                        redraw()
                continue

            if key in ENTER:
                break
            if key in (CTRL_C, CTRL_D):
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                raise (KeyboardInterrupt if key == CTRL_C else EOFError)
            if key in BACKSPACE:
                if cursor > 0:
                    buf.pop(cursor - 1)
                    cursor -= 1
                redraw()
                continue
            if key == TAB:
                if not buf and current_suggestion:
                    buf = list(current_suggestion)
                    cursor = len(buf)
                redraw()
                continue
            if key == "RIGHT":
                if not buf and current_suggestion:
                    # Accept the whole ghost suggestion as real input.
                    buf = list(current_suggestion)
                    cursor = len(buf)
                elif cursor < len(buf):
                    cursor += 1
                redraw()
                continue
            if key == "LEFT":
                if cursor > 0:
                    cursor -= 1
                redraw()
                continue
            if key == "UP":
                if history:
                    if history_index is None:
                        draft = list(buf)
                        load_history_entry(len(history) - 1)
                    elif history_index > 0:
                        load_history_entry(history_index - 1)
                    redraw()
                continue
            if key == "DOWN":
                if history_index is not None:
                    if history_index < len(history) - 1:
                        load_history_entry(history_index + 1)
                    else:
                        history_index = None
                        buf = list(draft)
                        cursor = len(buf)
                    redraw()
                continue
            if key in ("ESC", ""):
                continue  # unrecognized escape - silently ignored
            if len(key) == 1 and ord(key[0]) < 0x20:
                continue  # other control chars - ignore

            history_index = None  # editing exits history-browsing mode
            buf.insert(cursor, key)
            cursor += 1
            redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    sys.stdout.write("\r\n")
    sys.stdout.flush()
    return "".join(buf)
