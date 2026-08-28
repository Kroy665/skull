# Qwen Terminal Chat

A terminal chat client for a Qwen model endpoint, with tool calling, a
self-extending skill system, long-term memory, and a plan/auto execution
mode.

## What it can do

- Chat with streaming responses in your terminal
- Search the web and scrape pages (`web_search`, `scrape_page`)
- Run throwaway Python in an isolated E2B cloud sandbox (`run_python`)
- **Write its own tools at runtime** (`create_skill`) — once created, a
  skill is saved to `skills/` and reused in every future session
- **Long-term memory** — remembers facts about you and past conversations
  across sessions, using local embeddings (no external API needed for this)
- **Plan mode** (`/plan`) — research-only mode where the model proposes a
  plan instead of taking any action; `/auto` returns to normal mode
- Inline "ghost text" suggestions for your next likely question, arrow-key
  input history, and per-action loading animations

## Requirements

- macOS or Linux (raw-terminal input handling is POSIX-only; on Windows it
  falls back to plain input with no ghost text/inline history, but should
  still run)
- [uv](https://docs.astral.sh/uv/) — Python package/project manager
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A Qwen-compatible chat-completions API endpoint and key
- Optional: an [E2B](https://e2b.dev) API key, to enable the `run_python`
  sandbox tool

## Setup

```bash
git clone <this-repo-url>
cd qwen_llm
cp .env.example .env
# edit .env and fill in QWEN_KEY (required), E2B_API_KEY (optional)
uv sync
```

`uv sync` creates a `.venv` and installs all dependencies — no manual pip
install needed.

## Run

```bash
uv run chat.py
```

On first run, a local sentence-embedding model (~90MB) downloads
automatically for the memory feature; this only happens once.

## In-session commands

| Command      | Effect                                                        |
|--------------|-----------------------------------------------------------------|
| `/plan`      | Enter plan mode — research only, no writes/actions             |
| `/auto`      | Return to normal mode                                           |
| `/verbose`   | Show full tool call output                                      |
| `/concise`   | Collapse tool output to a one-line summary (default)             |
| `/reset`     | Clear conversation history (keeps mode/skills/memory)            |
| `exit`/`quit`| Leave                                                            |

Arrow keys: Up/Down recall previous inputs, Left/Right move the cursor.
Tab or Right-arrow (on an empty line) accepts an inline suggestion.

## Project layout

```
chat.py              entry point / main loop
skills_manager.py     self-created skill storage (skills/<name>/)
memory_store.py       local vector store for persona facts + conversation history
web_tools.py           web_search / scrape_page implementations
scratch_runner.py      E2B sandbox execution for run_python
suggestion.py           background "next question" prediction
ghost_input.py          raw-terminal input with ghost text + history
SYSTEM_PROMPT.md        the model's system prompt (editable without touching code)
skills/                 self-created skills (starts empty)
memory/                 persona facts + conversation log (starts empty, gitignored)
```

`skills/` and `memory/` are gitignored — they're per-user state that
accumulates as you use the app, not something to commit or ship.
