"""Tool schemas the model sees, dispatch to their Python implementations, and
assembly of the full tool list (built-in + meta + self-created skills) with
plan-mode filtering."""

import json

from skull.config import DIM, MAGENTA, RESET
from skull.core.memory_supersede import forget_fuzzy, remember_with_supersede
from skull.storage import store as mem
from skull.tools import files
from skull.tools import pipeline as pl
from skull.tools import shell
from skull.tools import skills as sm
from skull.tools import sandbox as scratch
from skull.tools import web as wt
from skull.ui.output import tprint

# ---------------------------------------------------------------------------
# Built-in tools: always available (unless withheld by plan mode).
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
                "an approach, debug, or inspect data - use before turning something into a "
                "skill with create_skill, which is for saving something reusable. Fully "
                "isolated from the local machine (no local files/network/credentials); "
                "state persists across calls within a session, reset at session start. "
                "Use print() to see output."
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
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command on the user's own machine with their full "
                "permissions - only for tasks genuinely requiring the local system "
                "(package installs, local file/process inspection, git, etc.) that "
                "run_python's isolated sandbox can't do. Requires interactive y/n "
                "approval every call and may be denied - handle a denial gracefully, "
                "don't retry the same command. Prefer run_python or a skill otherwise.\n\n"
                "IMPORTANT: pass background=true for anything that runs indefinitely (dev "
                "server, watcher, `npm run dev`) - without it the call blocks until exit "
                "or timeout, so a never-exiting process just hangs and times out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "reason": {
                        "type": "string",
                        "description": "One short sentence explaining why this command is needed, shown to the user in the approval prompt",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds to allow before considering it hung (default 30, max 120). Ignored when background=true.",
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "Set true for long-running/never-exiting commands (dev servers, "
                            "watchers). Starts the process detached and returns immediately "
                            "with a pid and log file instead of blocking. Use "
                            "list_background_commands / read_background_log / "
                            "stop_background_command to manage it afterward."
                        ),
                    },
                },
                "required": ["command", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_background_commands",
            "description": (
                "List every background command started this session via run_command "
                "(background=true), with its pid, whether it's still running, and its "
                "log file path."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_background_log",
            "description": (
                "Read the tail of a background command's output log, to check on "
                "progress or diagnose a problem (e.g. a dev server that failed to "
                "start) without stopping it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "The pid returned when the command was started"},
                    "tail_chars": {
                        "type": "integer",
                        "description": "How many characters of the end of the log to return (default 2000, max 20000)",
                    },
                },
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_background_command",
            "description": "Stop a background command started earlier via run_command(background=true).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "The pid returned when the command was started"},
                },
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the user's own machine, anywhere on the filesystem. "
                "PDF, DOCX, XLSX/XLSM, and PPTX are automatically extracted to readable "
                "text (result includes \"extracted\": true). Image files (PNG/JPG/etc.) "
                "are run through OCR to extract any TEXT that appears in the image - this "
                "is text recognition only, NOT visual understanding, so it will not "
                "describe a photo's content, only text visible within it (a screenshot, "
                "scanned document, sign, etc.); expect an empty/near-empty result for a "
                "photo with no text in it. Everything else is read as plain text. "
                "Read-only, no approval needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file (~ is expanded)"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 20000, max 40000)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory on the user's own machine. Read-only, no approval needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.', ~ is expanded)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create, overwrite, or append to a file on the user's own machine, anywhere "
                "on the filesystem. Requires the user's explicit interactive y/n approval "
                "every time, since this can destroy or corrupt an existing file. Shows the "
                "user the path and a preview of the content before asking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file (~ is expanded)"},
                    "content": {"type": "string", "description": "The content to write"},
                    "mode": {
                        "type": "string",
                        "description": "'overwrite' (default) replaces the whole file, 'append' adds to the end",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short sentence explaining why, shown in the approval prompt",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_read_file",
            "description": (
                "Read a file from the isolated E2B sandbox filesystem (not the user's "
                "machine - see read_file for that) - for content you want to inspect, "
                "not deliver. PDF, DOCX, XLSX/XLSM, and PPTX are automatically extracted "
                "to readable text (result includes \"extracted\": true); images are run "
                "through OCR to extract visible TEXT only (not visual understanding - it "
                "won't describe a photo, only text that appears in it); other text files "
                "are read as-is. For any OTHER binary output (zip, etc.) or anything "
                "meant to end up on the user's machine, use download_from_sandbox instead "
                "- do not base64-encode a binary file and read it through here, it wastes "
                "enormous context for no benefit. No approval needed, same trust level as "
                "run_python."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file inside the sandbox"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 20000, max 40000)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_write_file",
            "description": (
                "Write a file into the isolated E2B sandbox filesystem (not the user's "
                "machine - see write_file for that). No approval needed, same trust level "
                "as run_python. Creates parent directories as needed; overwrites if the "
                "file already exists. Files persist across run_python calls within the "
                "same session but are reset at session start."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file inside the sandbox"},
                    "content": {"type": "string", "description": "The content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_list_directory",
            "description": "List the contents of a directory inside the E2B sandbox filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path inside the sandbox (default '/')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_from_sandbox",
            "description": (
                "Copy a file directly from the sandbox to the user's own machine, as raw "
                "bytes - use this for ANY binary output (a generated .docx/.pdf/.png/.zip, "
                "etc.), never sandbox_read_file for that. sandbox_read_file is for reading "
                "text you want to see; this is for delivering a file to the user, and "
                "avoids burning huge amounts of context on a base64/text encoding of "
                "binary data the model doesn't need to read anyway. Requires the user's "
                "explicit y/n approval, same as write_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_path": {"type": "string", "description": "Path to the file inside the sandbox"},
                    "local_path": {"type": "string", "description": "Destination path on the user's machine (~ is expanded)"},
                    "reason": {
                        "type": "string",
                        "description": "One short sentence explaining why, shown in the approval prompt",
                    },
                },
                "required": ["sandbox_path", "local_path"],
            },
        },
    },
]

BUILTIN_IMPLS = {
    "web_search": lambda args: wt.web_search(args.get("query", ""), args.get("count", 5)),
    "scrape_page": lambda args: wt.scrape_page(args.get("url", ""), args.get("max_chars", 5000)),
    "run_python": lambda args: scratch.run_python(args.get("code", ""), args.get("timeout", 15)),
    "run_command": lambda args: shell.run_command(
        args.get("command", ""),
        args.get("reason", ""),
        args.get("timeout", 30),
        args.get("background", False),
    ),
    "list_background_commands": lambda args: shell.list_background_commands(),
    "read_background_log": lambda args: shell.read_background_log(
        args.get("pid"), args.get("tail_chars", 2000)
    ),
    "stop_background_command": lambda args: shell.stop_background_command(args.get("pid")),
    "read_file": lambda args: files.read_file(args.get("path", ""), args.get("max_chars", 20000)),
    "list_directory": lambda args: files.list_directory(args.get("path", ".")),
    "write_file": lambda args: files.write_file(
        args.get("path", ""),
        args.get("content", ""),
        args.get("mode", "overwrite"),
        args.get("reason", ""),
    ),
    "sandbox_read_file": lambda args: scratch.sandbox_read_file(
        args.get("path", ""), args.get("max_chars", 20000)
    ),
    "sandbox_write_file": lambda args: scratch.sandbox_write_file(
        args.get("path", ""), args.get("content", "")
    ),
    "sandbox_list_directory": lambda args: scratch.sandbox_list_directory(args.get("path", "/")),
    "download_from_sandbox": lambda args: scratch.download_from_sandbox(
        args.get("sandbox_path", ""), args.get("local_path", ""), args.get("reason", "")
    ),
}

# Tools that require blocking interactive stdin (a permission prompt) - the
# spinner must be stopped before these run so the prompt doesn't collide
# with spinner output on the same terminal line.
INTERACTIVE_TOOL_NAMES = {"run_command", "write_file", "download_from_sandbox"}

# ---------------------------------------------------------------------------
# Self-extension meta-tools: let the model write and register a brand new
# Python tool at runtime when none of the existing tools (built-in or
# previously self-created) can solve the task. Saved skills persist in
# skills/ and are auto-loaded as regular tools on every future run.
# ---------------------------------------------------------------------------

META_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": (
                "Create a brand new reusable Python tool ('skill') when no existing tool "
                "can solve the task. Code must define a top-level `run(**kwargs)` "
                "returning a JSON-serializable value. Stdlib and already-installed "
                "packages (requests, etc.) are available. Saved to disk, immediately "
                "callable this session and every future one.\n\n"
                "To reuse another skill's logic instead of duplicating it: "
                "`from skull.tools.skill_composition import call_skill` then "
                "`call_skill('other_skill_name', **kwargs)` - returns its result "
                "directly, raises SkillError on failure. Check list_skills first."
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
            "name": "delete_skill",
            "description": (
                "Permanently delete a previously self-created skill - use when a skill "
                "turned out broken, redundant with a better one you just made, or the "
                "user asks you to remove it. This cannot be undone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The exact skill name to delete"},
                },
                "required": ["name"],
            },
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
                "the user over time.\n\n"
                "IMPORTANT - when a fact CHANGES or CORRECTS an earlier one (a changed "
                "preference, a correction, \"actually, ...\"): just call remember with the "
                "new fact. Do NOT call forget first - remember automatically detects and "
                "supersedes the old fact for you (checked semantically, not by exact text "
                "match), and this is more reliable than forget, which only matches EXACT "
                "stored text and will silently fail on any paraphrase. forget is for when "
                "the user explicitly asks to delete/remove a fact outright, not for "
                "updates - manually forgetting-then-remembering an updated fact skips this "
                "safety check entirely."
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
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Permanently remove a previously saved persona fact - use ONLY when the "
                "user explicitly asks to delete/remove something, not for an updated or "
                "corrected fact (use remember for that - it supersedes the old fact "
                "automatically). Prefer quoting the exact stored text (e.g. from a prior "
                "recall_memory result), but a close paraphrase still works - falls back to "
                "a confirmed semantic match if there's no exact hit. ALWAYS check the "
                "result before telling the user it's done: {\"deleted\": false} means "
                "nothing was actually removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remove - the exact stored text if you have it, otherwise your best description of it",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_pipeline",
            "description": (
                "Create a saved DAG ('pipeline') chaining existing skills with explicit "
                "data flow - use instead of one big skill for a multi-step task, "
                "especially with fan-out/fan-in. All referenced skills must already exist "
                "(check list_skills, create_skill first if needed).\n\n"
                "`nodes`: {node_id: {\"type\":\"skill\",\"skill\":\"<name>\","
                "\"params\":{<literals, optional>}}}.\n"
                "`edges`: [{\"from\":\"<node>.<field>\",\"to\":\"<node>.<param>\"}] - wires an "
                "output field into another node's param. Call-time args are the reserved "
                "'input' pseudo-node, e.g. {\"from\":\"input.url\",\"to\":\"fetch.url\"}.\n\n"
                "Every node param needs EXACTLY ONE source: a literal in 'params' OR one "
                "incoming edge, never both/neither. Validated at creation time (cycles, "
                "unknown refs, unbound params reported by name)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "lowercase_snake_case identifier, e.g. 'scrape_and_summarize'",
                    },
                    "description": {
                        "type": "string",
                        "description": "One sentence describing what the pipeline does.",
                    },
                    "nodes": {
                        "type": "object",
                        "description": (
                            'Object mapping node id -> {"type":"skill","skill":"<name>",'
                            '"params":{...literal args...}}'
                        ),
                    },
                    "edges": {
                        "type": "array",
                        "description": 'List of {"from":"<node>.<field>","to":"<node>.<param>"}',
                    },
                },
                "required": ["name", "description", "nodes", "edges"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pipelines",
            "description": "List all previously created skill pipelines available to run.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": (
                "Run a previously created pipeline by name, passing the inputs its edges "
                "reference via the 'input' pseudo-node. Executes nodes in dependency "
                "order; if any node fails, the run stops immediately, reporting which "
                "node and why - nothing downstream runs. Returns terminal-node outputs "
                "plus a full per-node execution trace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The pipeline's name"},
                    "inputs": {
                        "type": "object",
                        "description": "Keyword arguments for the pipeline's 'input' pseudo-node",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_pipeline",
            "description": "Permanently delete a previously created pipeline. This cannot be undone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The exact pipeline name to delete"},
                },
                "required": ["name"],
            },
        },
    },
]


def _tool_list_skills() -> dict:
    return {"skills": sm.list_skills()}


def _tool_remember(fact: str, category: str = "other") -> dict:
    return remember_with_supersede(fact, {"category": category})


def _tool_recall_memory(query: str, k: int = 5) -> dict:
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
    "list_skills": lambda args: _tool_list_skills(),
    "delete_skill": lambda args: sm.delete_skill(args.get("name", "")),
    "remember": lambda args: _tool_remember(args.get("fact", ""), args.get("category", "other")),
    "recall_memory": lambda args: _tool_recall_memory(args.get("query", ""), args.get("k", 5)),
    "forget": lambda args: forget_fuzzy(args.get("fact", "")),
    "create_pipeline": lambda args: pl.create_pipeline(
        args.get("name", ""),
        args.get("description", ""),
        args.get("nodes", {}),
        args.get("edges", []),
    ),
    "list_pipelines": lambda args: {"pipelines": pl.list_pipelines()},
    "run_pipeline": lambda args: pl.run_pipeline(args.get("name", ""), **(args.get("inputs") or {})),
    "delete_pipeline": lambda args: pl.delete_pipeline(args.get("name", "")),
}

# Tools that mutate persistent state (files, memory) or execute arbitrary
# code. Excluded in plan mode, where the model may only research and
# propose - never act.
MUTATING_TOOL_NAMES = {
    "create_skill",
    "delete_skill",
    "remember",
    "forget",
    "run_python",
    "run_command",
    "stop_background_command",
    "write_file",
    "sandbox_write_file",
    "download_from_sandbox",
    "create_pipeline",
    "delete_pipeline",
    # run_pipeline's own effects are opaque (a node can be any skill, whose
    # side effects aren't tracked - same reasoning as self-created skills
    # being excluded wholesale in plan mode), so it's treated as mutating
    # even though it has no direct side effect of its own.
    "run_pipeline",
}


# Above this many saved skills, sending every skill's full schema on every
# turn regardless of relevance starts costing real context space for no
# benefit (most turns only need a handful) - switch to relevance filtering.
SKILL_FILTER_THRESHOLD = 8
SKILL_FILTER_TOP_K = 8
SKILL_FILTER_MIN_SCORE = 0.15


def _select_relevant_skills(query: str, always_include: set) -> list:
    """Return the subset of list_skills() entries worth sending this turn:
    the top-K most relevant to `query` by embedding similarity, plus any
    names in `always_include` (e.g. a skill already in use this
    conversation, so it doesn't disappear from under the model mid-task)."""
    entries = sm.list_skills()
    if len(entries) <= SKILL_FILTER_THRESHOLD:
        return entries

    from skull.storage.store import embed
    import numpy as np

    texts = [f"{e['name']}: {e['description']}" for e in entries]
    doc_vectors = embed(texts)
    query_vector = embed([query])[0]
    scores = doc_vectors @ query_vector
    top_idx = set(np.argsort(-scores)[:SKILL_FILTER_TOP_K].tolist())
    forced_idx = {i for i, e in enumerate(entries) if e["name"] in always_include}

    kept_idx = {i for i in top_idx if scores[i] >= SKILL_FILTER_MIN_SCORE} | forced_idx
    return [entries[i] for i in sorted(kept_idx)]


def build_tools_and_impls(plan_mode: bool = False, query: str = None, always_include_skills: set = None):
    """Assemble the full tool list (builtin + meta + saved skills) fresh each
    turn, so a skill created mid-conversation is immediately callable.

    In plan mode, mutating built-in tools (create_skill, delete_skill,
    remember, forget, run_python, run_command, stop_background_command,
    write_file, sandbox_write_file, create_pipeline, delete_pipeline,
    run_pipeline) and every self-created skill are excluded - a skill's
    (and by extension a pipeline node's) side effects aren't tracked, so the
    safe default is to treat all of them as potentially mutating and only
    allow the known-read-only built-ins (web_search, scrape_page,
    list_skills, recall_memory, read_file, list_directory,
    sandbox_read_file, sandbox_list_directory, list_pipelines) plus plain
    chat.

    Once the number of saved skills exceeds SKILL_FILTER_THRESHOLD, only the
    top-K skills most relevant to `query` (by embedding similarity against
    each skill's name+description) get their full schema sent - the rest
    stay reachable via list_skills/run via a future turn, keeping per-turn
    token cost roughly constant instead of growing with every skill ever
    created. Below the threshold, every skill is sent as before (filtering
    a handful of skills has no benefit and costs an embedding call).
    """
    tools = list(BUILTIN_TOOLS) + list(META_TOOLS)
    impls = dict(BUILTIN_IMPLS)
    impls.update(META_IMPLS)

    if plan_mode:
        tools = [t for t in tools if t["function"]["name"] not in MUTATING_TOOL_NAMES]
        impls = {k: v for k, v in impls.items() if k not in MUTATING_TOOL_NAMES}
        return tools, impls

    if query:
        skill_entries = _select_relevant_skills(query, always_include_skills or set())
    else:
        skill_entries = sm.list_skills()

    for entry in skill_entries:
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
        if "pipelines" in result and isinstance(result["pipelines"], list):
            return f"{len(result['pipelines'])} pipeline(s)"
        if "status" in result:
            extra = f" ({result['name']})" if "name" in result else ""
            return f"{result['status']}{extra}"
        if "text" in result and isinstance(result["text"], str):
            n = len(result["text"])
            return f"fetched {n} chars" + (" (truncated)" if result.get("truncated") else "")
        if "result" in result:
            return json.dumps(result["result"])[:80]
        if "stdout" in result:
            out = result["stdout"].strip().splitlines()
            first_line = out[0] if out else "(no output)"
            return first_line[:80]
    text = json.dumps(result)
    return text[:80] + ("…" if len(text) > 80 else "")


def run_tool_call(tool_call: dict, impls: dict, verbose: bool, spinner=None) -> str:
    name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}

    args_preview = _format_args_preview(args)
    needs_stdin = name in INTERACTIVE_TOOL_NAMES

    if spinner and not needs_stdin:
        spinner.start(f"running {name}({args_preview})", style="tool_wait")
    elif spinner:
        # This tool blocks on interactive input (e.g. a permission prompt) -
        # a running spinner would overwrite/collide with it on the same
        # terminal line, so stop it before the call instead of after.
        spinner.stop()

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

    if verbose:
        tprint(f"{MAGENTA}▸ {name}({raw_args}){RESET}")
        tprint(f"{DIM}{json.dumps(result, indent=2)}{RESET}")
    else:
        summary = _summarize_result(result)
        tprint(f"{MAGENTA}▸ {name}({args_preview}){RESET} {DIM}→ {summary}{RESET}")

    return json.dumps(result)
