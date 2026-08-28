#!/usr/bin/env -S uv run
"""Terminal chat client for the Qwen endpoint, with tool calling.

Usage:
    ./chat.py
    uv run chat.py

Env vars:
    QWEN_KEY  - bearer token (required)
    QWEN_URL  - base URL, defaults to https://qwen.your-endpoint.example
    QWEN_MODEL - model name, defaults to qwen3.8-27b
"""

import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

try:
    import readline  # noqa: F401  (importing wires up arrow keys/history for input())
except ImportError:
    pass  # not available on some platforms (e.g. plain Windows) - degrades gracefully

import requests
from dotenv import load_dotenv

import ghost_input
import memory_store as mem
import scratch_runner as scratch
import skills_manager as sm
import web_tools as wt
from suggestion import SuggestionEngine

load_dotenv()

QWEN_URL = os.environ.get("QWEN_URL", "https://qwen.your-endpoint.example").rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
QWEN_KEY = os.environ.get("QWEN_KEY")

SYSTEM_PROMPT_PATH = Path(__file__).parent / "SYSTEM_PROMPT.md"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

VERBOSE_TOOLS = False  # toggled by /verbose and /concise; controls tool output detail


# ---------------------------------------------------------------------------
# Spinner: distinct loading animation per action (thinking, calling a tool,
# waiting on a tool's result), so it's visually clear what's happening.
# ---------------------------------------------------------------------------

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
            self._thread.join(timeout=0.5)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Tools: define the JSON schema the model sees, plus the local Python
# implementation that actually runs when the model calls it.
# ---------------------------------------------------------------------------

BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web (DuckDuckGo) and return a list of matching pages "
                "(title, url, snippet). Use this to find current information or "
                "URLs to look up in more detail with scrape_page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (1-15, default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": (
                "Fetch a web page by URL and return its readable text content "
                "(scripts, styles, and nav/header/footer chrome stripped out). "
                "Use this to read the actual content of a page found via web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full page URL to fetch"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters of text to return (default 5000)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run a throwaway Python snippet in an isolated E2B cloud sandbox to test "
                "an approach, debug logic, or inspect data - separate from create_skill, "
                "which is for saving something reusable. The sandbox is fully isolated from "
                "the local machine (no access to local files, network state, or credentials) "
                "and persists variables/state across calls within one session, but is reset "
                "at the start of every new session. Use print() to see output. Use this to "
                "try something out BEFORE turning it into a skill with create_skill."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to run as a script"},
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds to allow (default 15, max 60)",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

BUILTIN_IMPLS = {
    "web_search": lambda args: wt.web_search(args.get("query", ""), args.get("count", 5)),
    "scrape_page": lambda args: wt.scrape_page(args.get("url", ""), args.get("max_chars", 5000)),
    "run_python": lambda args: scratch.run_python(args.get("code", ""), args.get("timeout", 15)),
}

# --- Self-extension meta-tools -------------------------------------------
# These let the model write and register a brand new Python tool at runtime
# when none of the existing tools (built-in or previously self-created) can
# solve the task. Saved skills persist in skills/ and are auto-loaded as
# regular tools on every future run.

META_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": (
                "Create a brand new reusable Python tool ('skill') when no existing tool "
                "can solve the task. The code must define a top-level function "
                "`run(**kwargs)` that returns a JSON-serializable value. Standard library "
                "and already-installed third-party packages (requests, etc.) are available. "
                "The skill is saved to disk and immediately becomes callable, in this "
                "session and every future one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "lowercase_snake_case identifier, e.g. 'count_vowels'",
                    },
                    "description": {
                        "type": "string",
                        "description": "One sentence describing what the skill does, for future tool listings.",
                    },
                    "parameters": {
                        "type": "object",
                        "description": (
                            "JSON-schema object describing run()'s keyword arguments, e.g. "
                            '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}'
                        ),
                    },
                    "code": {
                        "type": "string",
                        "description": "Full Python source defining `def run(**kwargs): ...`",
                    },
                },
                "required": ["name", "description", "parameters", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all previously self-created skills available to call.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a durable fact about the user to long-term memory: their identity, "
                "role, preferences, behavior patterns, opinions, or anything distinctive "
                "about them worth recalling in future conversations. Use whenever the user "
                "reveals something about themselves, not just when explicitly asked to "
                "remember - this is how you build a persistent persona/knowledge base of "
                "the user over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, written as a clear standalone statement.",
                    },
                    "category": {
                        "type": "string",
                        "description": "One of: identity, preference, behavior, project, opinion, other",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search long-term memory (past conversations and saved persona facts) for "
                "anything relevant to a query. Relevant memory is already auto-injected each "
                "turn, but use this to dig deeper on a specific topic or when you suspect "
                "there's more history than what's already shown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search memory for"},
                    "k": {"type": "integer", "description": "Max results per store (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
]


def tool_list_skills() -> dict:
    return {"skills": sm.list_skills()}


def tool_remember(fact: str, category: str = "other") -> dict:
    return mem.persona().add(fact, {"category": category})


def tool_recall_memory(query: str, k: int = 5) -> dict:
    return {
        "persona": mem.persona().search(query, k=k),
        "conversations": mem.conversations().search(query, k=k),
    }


META_IMPLS = {
    "create_skill": lambda args: sm.create_skill(
        args.get("name", ""),
        args.get("description", ""),
        args.get("parameters", {}),
        args.get("code", ""),
    ),
    "list_skills": lambda args: tool_list_skills(),
    "remember": lambda args: tool_remember(args.get("fact", ""), args.get("category", "other")),
    "recall_memory": lambda args: tool_recall_memory(args.get("query", ""), args.get("k", 5)),
}


# Tools that mutate persistent state (files, memory). Excluded in plan mode,
# where the model may only research and propose - never act.
MUTATING_TOOL_NAMES = {"create_skill", "remember", "run_python"}


def build_tools_and_impls(plan_mode: bool = False):
    """Assemble the full tool list (builtin + meta + saved skills) fresh each turn,
    so a skill created mid-conversation is immediately callable.

    In plan mode, mutating built-in tools (create_skill, remember) and every
    self-created skill are excluded - a skill's side effects aren't tracked,
    so the safe default is to treat all of them as potentially mutating and
    only allow the known-read-only built-ins (web_search, scrape_page,
    list_skills, recall_memory) plus plain chat.
    """
    tools = list(BUILTIN_TOOLS) + list(META_TOOLS)
    impls = dict(BUILTIN_IMPLS)
    impls.update(META_IMPLS)

    if plan_mode:
        tools = [t for t in tools if t["function"]["name"] not in MUTATING_TOOL_NAMES]
        impls = {k: v for k, v in impls.items() if k not in MUTATING_TOOL_NAMES}
        return tools, impls

    for entry in sm.list_skills():
        name = entry["name"]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": entry["description"],
                    "parameters": entry.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
        impls[name] = (lambda args, _name=name: sm.run_skill(_name, args))

    return tools, impls


def _is_valid_json(text) -> bool:
    if not text:
        return True  # empty/None arguments are fine (treated as {})
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _format_args_preview(args: dict, max_len: int = 60) -> str:
    parts = []
    for k, v in args.items():
        v_str = json.dumps(v) if not isinstance(v, str) else v
        v_str = v_str.replace("\n", "\\n").replace("\r", "")
        if len(v_str) > 40:
            v_str = v_str[:40] + "…"
        parts.append(f"{k}={v_str}")
    preview = ", ".join(parts)
    return preview if len(preview) <= max_len else preview[:max_len] + "…"


def _summarize_result(result) -> str:
    """One-line human summary of a tool result, for the collapsed default view."""
    if isinstance(result, dict):
        if "error" in result:
            return f"error: {str(result['error'])[:80]}"
        if "results" in result and isinstance(result["results"], list):
            return f"{len(result['results'])} result(s)"
        if "skills" in result and isinstance(result["skills"], list):
            return f"{len(result['skills'])} skill(s)"
        if "status" in result:
            extra = f" ({result['name']})" if "name" in result else ""
            return f"{result['status']}{extra}"
        if "text" in result and isinstance(result["text"], str):
            n = len(result["text"])
            return f"fetched {n} chars" + (" (truncated)" if result.get("truncated") else "")
        if "result" in result:
            return f"-> {json.dumps(result['result'])[:80]}"
        if "stdout" in result:
            out = result["stdout"].strip().splitlines()
            first_line = out[0] if out else "(no output)"
            return first_line[:80]
    text = json.dumps(result)
    return text[:80] + ("…" if len(text) > 80 else "")


def run_tool_call(tool_call: dict, impls: dict, spinner: "Spinner" = None) -> str:
    name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}

    args_preview = _format_args_preview(args)

    if spinner:
        spinner.start(f"running {name}({args_preview})", style="tool_wait")

    impl = impls.get(name)
    if impl is None:
        result = {"error": f"unknown tool '{name}'"}
    else:
        try:
            result = impl(args)
        except Exception as e:
            result = {"error": str(e)}

    if spinner:
        spinner.stop()

    if VERBOSE_TOOLS:
        print(f"{MAGENTA}▸ {name}({raw_args}){RESET}")
        print(f"{DIM}{json.dumps(result, indent=2)}{RESET}")
    else:
        summary = _summarize_result(result)
        print(f"{MAGENTA}▸ {name}({args_preview}){RESET} {DIM}→ {summary}{RESET}")

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------

def stream_chat(messages, tools, spinner: "Spinner" = None):
    """Stream one chat completion turn, printing content tokens as they arrive.

    A "thinking" spinner runs until the first visible output (content token
    or a named tool call) arrives, then stops so streamed text/tool markers
    print cleanly on their own line.

    Returns (content: str, tool_calls: list[dict] | None, finish_reason: str).
    """
    if spinner:
        spinner.start("thinking", style="thinking")

    resp = requests.post(
        f"{QWEN_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {QWEN_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "tools": tools,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    full_content = []
    tool_calls = {}  # index -> accumulated tool call dict
    finish_reason = None
    spinner_stopped = False
    printed_assistant_label = False

    def _reveal():
        nonlocal spinner_stopped, printed_assistant_label
        if spinner and not spinner_stopped:
            spinner.stop()
            spinner_stopped = True
        if not printed_assistant_label:
            print(f"{CYAN}assistant>{RESET} ", end="", flush=True)
            printed_assistant_label = True

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        chunk = json.loads(data)
        choice = chunk["choices"][0]
        delta = choice["delta"]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        piece = delta.get("content")
        if piece:
            _reveal()
            print(piece, end="", flush=True)
            full_content.append(piece)

        for tc_delta in delta.get("tool_calls") or []:
            idx = tc_delta.get("index", 0)
            entry = tool_calls.setdefault(
                idx,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc_delta.get("id"):
                entry["id"] = tc_delta["id"]
            fn_delta = tc_delta.get("function") or {}
            if fn_delta.get("name"):
                entry["function"]["name"] += fn_delta["name"]
                if spinner and not spinner_stopped:
                    spinner.update(f"calling {entry['function']['name']}", style="tool_call")
            if fn_delta.get("arguments"):
                entry["function"]["arguments"] += fn_delta["arguments"]

    if spinner and not spinner_stopped:
        spinner.stop()
    if printed_assistant_label or full_content:
        print()
    ordered_tool_calls = [tool_calls[i] for i in sorted(tool_calls)] if tool_calls else None
    return "".join(full_content), ordered_tool_calls, finish_reason


MEMORY_TOP_K = 4
MEMORY_MIN_SCORE = 0.35  # cosine similarity floor - drop weak/irrelevant matches


def build_memory_context(query: str) -> str:
    """Retrieve relevant persona facts + past conversation turns for `query`,
    returned as a block of text to fold into the leading system message
    (this server rejects any system-role message that isn't first in the
    list, so memory can't be injected as its own separate message)."""
    persona_hits = mem.persona().search(query, k=MEMORY_TOP_K, min_score=MEMORY_MIN_SCORE)
    convo_hits = mem.conversations().search(query, k=MEMORY_TOP_K, min_score=MEMORY_MIN_SCORE)

    if not persona_hits and not convo_hits:
        return ""

    lines = ["\n\nRelevant long-term memory (may or may not be relevant - use your judgment):"]
    if persona_hits:
        lines.append("\nAbout the user:")
        lines.extend(f"- {h['text']}" for h in persona_hits)
    if convo_hits:
        lines.append("\nRelevant past conversation:")
        lines.extend(f"- {h['text']}" for h in convo_hits)

    return "\n".join(lines)


PLAN_MODE_ADDENDUM = (
    "\n\nPLAN MODE IS ACTIVE. You may research using read-only tools "
    "(web_search, scrape_page, list_skills, recall_memory) but create_skill, "
    "remember, run_python, and every self-created skill are temporarily "
    "hidden from your tool list - they still exist and will work normally "
    "once plan mode ends, they are just withheld right now so you can't take "
    "any mutating/action-taking step. Don't conclude a tool is missing or "
    "needs to be (re)created just because you can't see it in plan mode. "
    "Instead, investigate as needed and respond with a clear, concrete plan "
    "of what you would do. Wait for the user to leave plan mode (they'll "
    "type /auto) before anything in the plan gets executed."
)


def handle_turn(messages, user_input, plan_mode=False):
    """Append the user's message and run the assistant turn to completion,
    executing any tool calls in a loop.

    On any failure, rolls `messages` back to its state before this turn began
    (via truncation, in place) so a malformed tool call never poisons every
    subsequent request in the conversation.

    Retrieves relevant memory fresh each round-trip (folded into the leading
    system message for this request only, not persisted into `messages`) and
    auto-stores the final user/assistant exchange into long-term conversation
    memory on success. In plan mode, mutating tools are withheld and the
    model is instructed to propose a plan instead of acting.
    """
    checkpoint = len(messages)
    messages.append({"role": "user", "content": user_input})

    while True:
        # Rebuild tools/impls every round-trip: a skill created mid-turn (e.g.
        # create_skill just now) must be callable on the very next model call.
        tools, impls = build_tools_and_impls(plan_mode=plan_mode)

        memory_block = build_memory_context(user_input)
        extra = memory_block + (PLAN_MODE_ADDENDUM if plan_mode else "")
        if extra and messages and messages[0]["role"] == "system":
            request_messages = list(messages)
            request_messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + extra,
            }
        else:
            request_messages = messages

        spinner = Spinner()
        try:
            content, tool_calls, finish_reason = stream_chat(request_messages, tools, spinner=spinner)
        except requests.HTTPError as e:
            spinner.stop()
            print(f"{YELLOW}HTTP error: {e}{RESET}")
            print(f"{YELLOW}{e.response.text}{RESET}")
            del messages[checkpoint:]
            return False
        except requests.RequestException as e:
            spinner.stop()
            print(f"{YELLOW}Request failed: {e}{RESET}")
            del messages[checkpoint:]
            return False
        finally:
            spinner.stop()

        assistant_msg = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if content:
                mem.conversations().add(
                    f"User: {user_input}\nAssistant: {content}",
                    {"role": "exchange"},
                )
            return True

        malformed = [tc for tc in tool_calls if not _is_valid_json(tc["function"].get("arguments"))]
        if malformed:
            names = ", ".join(tc["function"]["name"] for tc in malformed)
            print(f"{YELLOW}Model produced malformed tool-call arguments for: {names}. Discarding this turn.{RESET}")
            del messages[checkpoint:]
            return False

        for tc in tool_calls:
            tool_spinner = Spinner()
            result = run_tool_call(tc, impls, spinner=tool_spinner)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )
        # loop again so the model can respond to the tool results


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text().strip()
    return "You are a helpful terminal assistant with access to tools."


SYSTEM_PROMPT = load_system_prompt()


def _last_session_context() -> list:
    """Reconstruct a small chat-style context from the most recent logged
    conversation exchange, for suggesting a first question in a fresh
    session with no history yet."""
    entries = mem.conversations().all()
    if not entries:
        return []
    last_text = entries[-1]["text"]  # "User: ...\nAssistant: ..."
    if "\nAssistant: " not in last_text:
        return []
    user_part, assistant_part = last_text.split("\nAssistant: ", 1)
    user_part = user_part.removeprefix("User: ")
    return [
        {"role": "user", "content": user_part},
        {"role": "assistant", "content": assistant_part},
    ]


def main():
    global VERBOSE_TOOLS

    if not QWEN_KEY:
        print("Error: set QWEN_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    scratch.clear_scratch()

    print(f"{BOLD}Qwen terminal chat{RESET} {DIM}({QWEN_MODEL} @ {QWEN_URL}){RESET}")
    print(f"{DIM}Built-in tools: {', '.join(BUILTIN_IMPLS)}{RESET}")
    skill_names = [e["name"] for e in sm.list_skills()]
    if skill_names:
        print(f"{DIM}Learned skills: {', '.join(skill_names)}{RESET}")
    print(
        f"{DIM}Type 'exit', 'quit', or Ctrl-D to leave. '/reset' clears history. "
        f"'/plan' enters plan mode (research-only, no writes). '/auto' returns to normal mode. "
        f"'/verbose' shows full tool output, '/concise' collapses it back to one line (default).{RESET}\n"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    plan_mode = False
    input_history = []

    suggestions = SuggestionEngine(QWEN_URL, QWEN_KEY, QWEN_MODEL)
    suggestions.refresh(_last_session_context())

    while True:
        prompt_label = f"{MAGENTA}plan{RESET} {GREEN}you>{RESET} " if plan_mode else f"{GREEN}you>{RESET} "
        try:
            user_input = ghost_input.prompt_with_ghost(
                prompt_label, suggestions.get, history=input_history
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if not input_history or input_history[-1] != user_input:
            input_history.append(user_input)
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            suggestions.clear()
            print(f"{DIM}history cleared{RESET}\n")
            continue
        if user_input == "/plan":
            plan_mode = True
            print(f"{DIM}Plan mode on: research-only, no create_skill/remember/skill calls until /auto.{RESET}\n")
            continue
        if user_input == "/auto":
            plan_mode = False
            print(f"{DIM}Auto mode on: full tool access restored.{RESET}\n")
            continue
        if user_input == "/verbose":
            VERBOSE_TOOLS = True
            print(f"{DIM}Verbose mode on: tool calls show full output.{RESET}\n")
            continue
        if user_input == "/concise":
            VERBOSE_TOOLS = False
            print(f"{DIM}Concise mode on: tool calls show a one-line summary.{RESET}\n")
            continue

        suggestions.clear()
        handle_turn(messages, user_input, plan_mode=plan_mode)
        suggestions.refresh(messages[1:])  # skip the (possibly huge) system message
        print()


if __name__ == "__main__":
    main()
