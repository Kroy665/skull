"""Shared interactive y/n approval prompt for mutating tools that touch the
user's real machine (run_command, write_file). Reads/E2B-sandbox operations
don't use this - see each tool's own docstring for its trust level."""

from skull.config import BOLD, DIM, RESET, YELLOW
from skull.ui.output import tprint


def ask_permission(action_label: str, detail_lines: list, reason: str = "") -> bool:
    """Show `action_label` (e.g. "run a shell command"), each line in
    `detail_lines` indented under it, and an optional reason, then block on
    an interactive y/n. Any of y/yes approves; empty/n/no/anything else
    denies."""
    # A newline (not just \r) guarantees this starts on a fresh, empty line
    # regardless of anything printed just before (e.g. a spinner, or a long
    # line that wrapped across multiple terminal rows).
    tprint()
    tprint(f"{YELLOW}{BOLD}The model wants to {action_label}:{RESET}")
    for line in detail_lines:
        for subline in str(line).splitlines():
            tprint(f"{BOLD}  {subline}{RESET}")
    if reason:
        tprint(f"{DIM}  reason: {reason}{RESET}")
    tprint()
    while True:
        answer = input(f"{YELLOW}Allow this? [y/N] {RESET}").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        tprint("Please answer y or n.")
