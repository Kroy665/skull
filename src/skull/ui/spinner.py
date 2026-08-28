"""Distinct loading animation per action (thinking, calling a tool, waiting
on a tool's result), so it's visually clear what's happening."""

import itertools
import sys
import threading
import time

from skull.config import CYAN, MAGENTA, RESET, YELLOW


class Spinner:
    """A single-line terminal spinner that can be reconfigured mid-flight
    (different frames/label/color per phase) without leaving stray output."""

    FRAMES = {
        "thinking": (list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), CYAN),
        "tool_call": (["▸  ", " ▸ ", "  ▸", " ▸ "], MAGENTA),
        "tool_wait": (["◷", "◶", "◵", "◴"], YELLOW),
    }

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._label = ""
        self._style = "thinking"

    def _spin(self):
        frames, color = self.FRAMES.get(self._style, self.FRAMES["thinking"])
        for frame in itertools.cycle(frames):
            if self._stop.is_set():
                break
            with self._lock:
                label = self._label
            sys.stdout.write(f"\r{color}{frame} {label}{RESET}\033[K")
            sys.stdout.flush()
            time.sleep(0.08)

    def start(self, label: str, style: str = "thinking"):
        self._label = label
        self._style = style
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def update(self, label: str, style: str = None):
        with self._lock:
            self._label = label
            if style:
                self._style = style

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Extremely unlikely (the loop only sleeps 0.08s between
                # checks), but if it ever happens, don't silently proceed -
                # a still-running spinner thread writing to stdout at the
                # same time as the next print()/input() call corrupts the
                # terminal (interleaved writes, stray \r, misplaced cursor).
                raise RuntimeError("Spinner thread did not stop in time")
        # Clear the line AND move to a fresh one, so whatever prints next
        # (which may include long lines that wrap across multiple terminal
        # rows) always starts from a known, empty line rather than
        # overwriting/colliding with spinner remnants on the current row.
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
