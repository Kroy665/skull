"""Terminal output helper that doesn't depend on the terminal's own \\n -> \\r\\n
translation (the OPOST/ONLCR line discipline).

ghost_input.py puts the terminal into raw mode (which explicitly disables
OPOST) while reading a line, then restores the original settings before
returning. Some terminal emulators (observed with VS Code's integrated
terminal) don't reliably re-enable output post-processing in between raw
mode sessions, causing every bare '\\n' printed afterward to move the cursor
down without returning to column 0 - text drifts progressively further right
with every line, a "staircase" effect.

The fix: never rely on the terminal to translate '\\n'. Everything the app
prints - including model-streamed content, which contains embedded '\\n's we
don't control - goes through tprint()/twrite(), which always emit '\\r\\n'
explicitly.
"""

import sys


def twrite(text: str) -> None:
    """Write `text` to stdout, translating any '\\n' not already preceded by
    '\\r' into '\\r\\n'. No implicit trailing newline (unlike print())."""
    sys.stdout.write(text.replace("\r\n", "\n").replace("\n", "\r\n"))


def tprint(text: str = "", end: str = "\n", flush: bool = False) -> None:
    """Drop-in replacement for print() that is safe regardless of whether
    the terminal's OPOST/ONLCR line discipline is currently active."""
    twrite(text)
    twrite(end)
    if flush:
        sys.stdout.flush()
