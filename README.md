# skull

A terminal chat client for a Qwen model endpoint, with tool calling, a
self-extending skill system, long-term memory, and a plan/auto execution
mode.

## What it can do

- Chat with streaming responses in your terminal
- Search the web and scrape pages (`web_search`, `scrape_page`)
- Run throwaway Python in an isolated E2B cloud sandbox (`run_python`),
  including reading/writing files there (`sandbox_read_file`,
  `sandbox_write_file`, `sandbox_list_directory`)
- Read files on your own machine freely (`read_file`, `list_directory`); writing
  (`write_file`) and running shell commands (`run_command`) require your
  explicit y/n approval every time, since those touch your real filesystem
- **Background processes** — `run_command(background=true)` starts a
  long-running process (a dev server, a watcher) detached and returns
  immediately, instead of hanging until timeout; manage it with
  `list_background_commands`, `read_background_log`, `stop_background_command`
- **Write its own tools at runtime** (`create_skill`/`delete_skill`) — once
  created, a skill is saved to `skills/` and reused in every future session.
  Skills can call each other (`call_skill`) instead of duplicating logic
- **Chain skills into a saved DAG** (`create_pipeline`/`run_pipeline`/
  `list_pipelines`/`delete_pipeline`) — nodes are existing skills, edges wire
  one node's output field into another's parameter, with fan-out (one output
  feeding several next steps) and fan-in (one step needing several previous
  outputs) both supported. Validated at creation time (cycles, unbound
  parameters, unknown fields); if a node fails at run time, the whole run
  stops and reports exactly which node and why
- **Long-term memory** (`remember`/`forget`/`recall_memory`) — remembers
  facts about you and past conversations across sessions, using local
  embeddings (no external API needed for this)
- **Automatic context compaction** — when the conversation grows large, the
  oldest history is summarized into a compact note instead of the session
  erroring out or requiring a manual reset
- **Plan mode** (`/plan`) — research-only mode where the model proposes a
  plan instead of taking any action; `/auto` returns to normal mode
- Inline "ghost text" suggestions for your next likely question, arrow-key
  input history, and per-action loading animations

## Requirements

- macOS or Linux (raw-terminal input handling is POSIX-only; on Windows it
  falls back to plain input with no ghost text/inline history, but should
  still run)
- Python 3.12+ (needed for `sqlite3`'s extension-loading support, which
  `sqlite-vec` requires — some pyenv/system Python 3.10 builds lack this)
- [uv](https://docs.astral.sh/uv/) — Python package/project manager
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`); `uv sync` will fetch
  a matching Python automatically if you don't have one
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
chat.py                    thin entry point (uv run chat.py)
src/skull/
    config.py               env vars, paths, terminal colors
    core/
        client.py            streaming Qwen chat-completions calls
        session.py            turn-handling loop + interactive REPL
        memory_context.py      memory retrieval, plan-mode instructions
        compaction.py           summarizes old history to stay within context limits
    tools/
        registry.py            tool schemas + dispatch, plan-mode filtering
        web.py                  web_search / scrape_page
        sandbox.py               E2B sandbox execution for run_python + sandbox file I/O
        skills.py                self-created skill storage (skills/<name>/)
        skill_composition.py      call_skill() - lets a skill call another skill
        pipeline.py               skill DAGs: validation, topological execution (pipelines/<name>/)
        shell.py                 run_command (gated) + background process management
        files.py                 local read_file/write_file/list_directory
        permission.py            shared interactive y/n approval prompt
    storage/
        store.py                 local vector store (persona facts + conversation history), backed by SQLite + sqlite-vec
    ui/
        spinner.py               per-action loading animations
        ghost_input.py            raw-terminal input with ghost text + history
        suggestion.py             background "next question" prediction
        output.py                 terminal-safe print (explicit \r\n, avoids OPOST quirks)
SYSTEM_PROMPT.md            the model's system prompt (editable without touching code)
skills/                      self-created skills (starts empty)
pipelines/                    self-created skill pipelines/DAGs (starts empty)
memory/                      persona facts + conversation log (starts empty, gitignored)
```

`skills/`, `pipelines/`, and `memory/` live at the project root (not inside
`src/skull/`) since they're per-user runtime state, not part of the
installed package. All three are gitignored — they accumulate as you use
the app, not something to commit or ship.
