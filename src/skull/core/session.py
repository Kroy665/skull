"""The turn-handling loop and the interactive REPL."""

import json
import sys

import requests

from skull.config import (
    BOLD,
    DIM,
    GREEN,
    MAGENTA,
    QWEN_KEY,
    QWEN_MODEL,
    QWEN_URL,
    RESET,
    YELLOW,
    load_system_prompt,
)
from skull.core.client import stream_chat
from skull.core.compaction import compact_if_needed
from skull.core.memory_context import (
    PLAN_MODE_ADDENDUM,
    build_memory_context,
    last_session_context,
)
from skull.storage import store as mem
from skull.tools import sandbox as scratch
from skull.tools import skills as sm
from skull.tools.registry import BUILTIN_IMPLS, _is_valid_json, build_tools_and_impls, run_tool_call
from skull.ui import ghost_input
from skull.ui.output import tprint
from skull.ui.spinner import Spinner
from skull.ui.suggestion import SuggestionEngine

MAX_MALFORMED_RETRIES = 2


class Session:
    """Holds the mutable state of one interactive chat session: message
    history, mode flags, and the background suggestion engine."""

    def __init__(self):
        self.system_prompt = load_system_prompt()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.plan_mode = False
        self.verbose_tools = False
        self.input_history = []
        self.suggestions = SuggestionEngine(QWEN_URL, QWEN_KEY, QWEN_MODEL)

    def reset_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def handle_turn(self, user_input: str) -> bool:
        """Append the user's message and run the assistant turn to completion,
        executing any tool calls in a loop.

        On any failure, rolls `self.messages` back to its state before this
        turn began (via truncation, in place) so a malformed tool call never
        poisons every subsequent request in the conversation.

        Retrieves relevant memory fresh each round-trip (folded into the
        leading system message for this request only, not persisted into
        `self.messages`) and auto-stores the final user/assistant exchange
        into long-term conversation memory on success. In plan mode,
        mutating tools are withheld and the model is instructed to propose
        a plan instead of acting.
        """
        # Compact BEFORE capturing the checkpoint - if this shrinks
        # self.messages, checkpoint must reflect the post-compaction length,
        # or the later `del messages[checkpoint:]` rollback would operate on
        # stale indices.
        self.messages, compacted = compact_if_needed(self.messages)
        if compacted:
            tprint(f"{DIM}(compacted older conversation history to stay within context limits){RESET}")

        messages = self.messages
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        malformed_retries = 0

        while True:
            # Rebuild tools/impls every round-trip: a skill created mid-turn
            # (e.g. create_skill just now) must be callable on the very next
            # model call.
            tools, impls = build_tools_and_impls(plan_mode=self.plan_mode)

            memory_block = build_memory_context(user_input)
            extra = memory_block + (PLAN_MODE_ADDENDUM if self.plan_mode else "")
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
                content, tool_calls, finish_reason = stream_chat(
                    request_messages, tools, spinner=spinner
                )
            except requests.HTTPError as e:
                spinner.stop()
                tprint(f"{YELLOW}HTTP error: {e}{RESET}")
                tprint(f"{YELLOW}{e.response.text}{RESET}")
                del messages[checkpoint:]
                return False
            except requests.RequestException as e:
                spinner.stop()
                tprint(f"{YELLOW}Request failed: {e}{RESET}")
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

            malformed = [
                tc for tc in tool_calls if not _is_valid_json(tc["function"].get("arguments"))
            ]
            if malformed:
                malformed_retries += 1
                if malformed_retries > MAX_MALFORMED_RETRIES:
                    names = ", ".join(tc["function"]["name"] for tc in malformed)
                    tprint(
                        f"{YELLOW}Model repeatedly produced malformed tool-call arguments "
                        f"for: {names}. Giving up on this turn.{RESET}"
                    )
                    del messages[checkpoint:]
                    return False

            # Execute every valid tool call and, for any malformed one, feed
            # back a normal tool-error message instead - every tool_calls
            # entry in assistant_msg needs a matching tool response either
            # way, and this lets the model retry (e.g. with shorter content)
            # instead of losing everything it already got right this turn.
            # Common cause of malformed args: the response got cut off
            # mid-argument by max_tokens (e.g. a large file-write payload),
            # leaving truncated/invalid JSON.
            for tc in tool_calls:
                if not _is_valid_json(tc["function"].get("arguments")):
                    tprint(
                        f"{YELLOW}Model produced malformed tool-call arguments for: "
                        f"{tc['function']['name']}. Asking it to retry.{RESET}"
                    )
                    result = json.dumps(
                        {
                            "error": (
                                "Your arguments for this tool call were not valid JSON "
                                "(the response may have been cut off before it finished, "
                                "often because the content was too long). Please retry - "
                                "consider shorter content, or splitting a large file write "
                                "into a create followed by append calls."
                            )
                        }
                    )
                else:
                    tool_spinner = Spinner()
                    result = run_tool_call(tc, impls, self.verbose_tools, spinner=tool_spinner)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            # loop again so the model can respond to the tool results


def _print_banner(session: Session):
    tprint(f"{BOLD}Qwen terminal chat{RESET} {DIM}({QWEN_MODEL} @ {QWEN_URL}){RESET}")
    tprint(f"{DIM}Built-in tools: {', '.join(BUILTIN_IMPLS)}{RESET}")
    skill_names = [e["name"] for e in sm.list_skills()]
    if skill_names:
        tprint(f"{DIM}Learned skills: {', '.join(skill_names)}{RESET}")
    tprint(
        f"{DIM}Type 'exit', 'quit', or Ctrl-D to leave. '/reset' clears history. "
        f"'/plan' enters plan mode (research-only, no writes). '/auto' returns to normal mode. "
        f"'/verbose' shows full tool output, '/concise' collapses it back to one line (default).{RESET}\n"
    )


def run():
    if not QWEN_KEY:
        print("Error: set QWEN_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    scratch.clear_scratch()

    session = Session()
    _print_banner(session)
    session.suggestions.refresh(last_session_context())

    while True:
        prompt_label = (
            f"{MAGENTA}plan{RESET} {GREEN}you>{RESET} " if session.plan_mode else f"{GREEN}you>{RESET} "
        )
        try:
            user_input = ghost_input.prompt_with_ghost(
                prompt_label, session.suggestions.get, history=session.input_history
            ).strip()
        except (EOFError, KeyboardInterrupt):
            tprint()
            break

        if not user_input:
            continue
        if not session.input_history or session.input_history[-1] != user_input:
            session.input_history.append(user_input)
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/reset":
            session.reset_messages()
            session.suggestions.clear()
            tprint(f"{DIM}history cleared{RESET}\n")
            continue
        if user_input == "/plan":
            session.plan_mode = True
            tprint(
                f"{DIM}Plan mode on: research-only, no create_skill/remember/skill "
                f"calls until /auto.{RESET}\n"
            )
            continue
        if user_input == "/auto":
            session.plan_mode = False
            tprint(f"{DIM}Auto mode on: full tool access restored.{RESET}\n")
            continue
        if user_input == "/verbose":
            session.verbose_tools = True
            tprint(f"{DIM}Verbose mode on: tool calls show full output.{RESET}\n")
            continue
        if user_input == "/concise":
            session.verbose_tools = False
            tprint(f"{DIM}Concise mode on: tool calls show a one-line summary.{RESET}\n")
            continue

        session.suggestions.clear()
        session.handle_turn(user_input)
        session.suggestions.refresh(session.messages[1:])  # skip the system message
        tprint()
