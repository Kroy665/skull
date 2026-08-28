#!/usr/bin/env -S uv run
"""Entry point for the skull terminal chat client.

Usage:
    ./chat.py
    uv run chat.py

Env vars:
    QWEN_KEY     - bearer token (required)
    QWEN_URL     - base URL, defaults to https://qwen.your-endpoint.example
    QWEN_MODEL   - model name, defaults to qwen3.8-27b
    E2B_API_KEY  - enables the run_python sandbox tool (optional)
"""

try:
    import readline  # noqa: F401  (importing wires up arrow keys/history for input())
except ImportError:
    pass  # not available on some platforms (e.g. plain Windows) - degrades gracefully

from skull.core.session import run

if __name__ == "__main__":
    run()
