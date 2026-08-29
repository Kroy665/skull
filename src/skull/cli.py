"""Console-script entry point (`skull`, installed via `[project.scripts]`
in pyproject.toml) - runnable from any directory once installed, unlike
the project-local `uv run chat.py` workflow. All per-user data (skills/,
memory/, pipelines/, the .env, skills.env) lives in a standard per-user
config directory (see skull.config.CONFIG_DIR), found the same way
regardless of where this command is invoked from.
"""

import sys


def main():
    try:
        import readline  # noqa: F401  (importing wires up arrow keys/history for input())
    except ImportError:
        pass  # not available on some platforms (e.g. plain Windows) - degrades gracefully

    from skull.core.session import run

    run()


if __name__ == "__main__":
    sys.exit(main())
