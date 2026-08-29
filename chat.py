#!/usr/bin/env -S uv run
"""Local-development entry point - for a real installed command, use
`skull` instead (see [project.scripts] in pyproject.toml, and
src/skull/cli.py). This just delegates to the same code path, kept around
for the `uv run chat.py` workflow while developing this repo directly.

Usage:
    ./chat.py
    uv run chat.py

Env vars (also settable in ~/.config/skull/.env once installed for real -
see skull.config.CONFIG_DIR):
    QWEN_URL     - your Qwen-compatible chat-completions endpoint (required)
    QWEN_KEY     - bearer token (required)
    QWEN_MODEL   - model name, defaults to qwen3.8-27b
    E2B_API_KEY  - enables the run_python sandbox tool (optional)
"""

from skull.cli import main

if __name__ == "__main__":
    main()
