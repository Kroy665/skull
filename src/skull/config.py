"""Environment configuration, paths, and terminal color constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The project root is three levels up from this file:
# src/skull/config.py -> src/skull -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

QWEN_URL = os.environ.get("QWEN_URL", "https://qwen.your-endpoint.example").rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
QWEN_KEY = os.environ.get("QWEN_KEY")

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "SYSTEM_PROMPT.md"
SKILLS_DIR = PROJECT_ROOT / "skills"
MEMORY_DIR = PROJECT_ROOT / "memory"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text().strip()
    return "You are a helpful terminal assistant with access to tools."
