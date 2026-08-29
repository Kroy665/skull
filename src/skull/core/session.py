"""The turn-handling loop and the interactive REPL."""

import json
import sys

import requests

from skull.config import (
    BOLD,
    CONFIG_DIR,
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
from skull.core.client import StreamParseError, stream_chat
from skull.core.compaction import compact_if_needed, estimate_tokens, COMPACT_TRIGGER_TOKENS
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

    def _skill_ranking_query(self, user_input: str, turn_messages: list) -> str:
        """Build the text used to rank which saved skills are relevant
        enough to send this round-trip (see build_tools_and_impls' `query`).

        Real gap this fixes: using just the original user_input, frozen for
        the whole turn, meant a skill the model only realizes it needs AFTER
        seeing an earlier tool's result (e.g. "check disk space" -> low
        space found -> now wants to call a notification/email skill never
        mentioned in the original phrasing) could rank below the relevance
        threshold and simply not appear in the tool list on a later
        round-trip - not silently, since the model would get an "unknown
        tool" error if it guessed the name anyway, but with no way to
        actually complete the task in one turn. Folding in what's actually
        happened so far this turn (tool names called, a bounded snippet of
        their results, and the assistant's own reasoning) lets the ranking
        reflect what the model has since learned it needs, not just what
        the user originally typed.

        Bounded to the last few turn messages and a short snippet of each,
        so this doesn't let a single large tool result balloon the ranking
        query (and, with it, the per-round-trip embedding cost) without limit.
        """
        MAX_CONTEXT_MESSAGES = 6
        MAX_SNIPPET_CHARS = 200

        parts = [user_input]
        for m in turn_messages[-MAX_CONTEXT_MESSAGES:]:
            role = m.get("role")
            if role == "assistant":
                if m.get("content"):
                    parts.append(m["content"][:MAX_SNIPPET_CHARS])
                for tc in m.get("tool_calls") or []:
                    parts.append(tc["function"]["name"])
            elif role == "tool":
                parts.append((m.get("content") or "")[:MAX_SNIPPET_CHARS])
        return "\n".join(parts)

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
        # Cheap pre-check using just the prior turns' messages (tool schemas
        # aren't built yet at this point) - compact BEFORE capturing the
        # checkpoint, since a shrink here must be reflected in checkpoint or
        # the later `del messages[checkpoint:]` rollback would use a stale
        # index. The more precise, tools-aware check happens per round-trip
        # inside the loop below, since a single turn can itself accumulate
        # enough tool calls/results to cross the threshold.
        self.messages, compacted = compact_if_needed(self.messages)
        if compacted:
            tprint(f"{DIM}(compacted older conversation history to stay within context limits){RESET}")

        messages = self.messages
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        malformed_retries = 0
        skills_used_this_turn = set()

        while True:
            # Rebuild tools/impls every round-trip: a skill created mid-turn
            # (e.g. create_skill just now) must be callable on the very next
            # model call. `query` scopes which saved skills' schemas are
            # actually sent once the skill count is large (see
            # SKILL_FILTER_THRESHOLD) - built from the turn's accumulated
            # context (see _skill_ranking_query), not just the original
            # user_input, so a skill only discovered as relevant after an
            # earlier tool result still has a chance to rank in.
            # always_include_skills keeps a skill already called earlier
            # this turn from disappearing on the next round-trip just
            # because the relevance ranking shifted.
            ranking_query = self._skill_ranking_query(user_input, messages[checkpoint + 1:])
            tools, impls = build_tools_and_impls(
                plan_mode=self.plan_mode,
                query=ranking_query,
                always_include_skills=skills_used_this_turn,
            )

            memory_block = build_memory_context(user_input)
            extra = memory_block + (PLAN_MODE_ADDENDUM if self.plan_mode else "")

            # Re-check on every round-trip, not just once at turn start: a
            # single turn can accumulate several large tool calls/results
            # (e.g. a chain of create_skill/delete_skill retries embedding
            # generated code), growing well past the trigger threshold
            # entirely within one handle_turn call. The real request size
            # includes the tool schemas too - these can be sizable once
            # several self-created skills accumulate - and the memory/plan
            # text folded into the system message, not just `messages` alone.
            if estimate_tokens(messages, tools=tools, extra_chars=len(extra)) >= COMPACT_TRIGGER_TOKENS:
                pre_len = len(messages)
                compacted_messages, did_compact = compact_if_needed(messages, force=True)
                if did_compact:
                    removed = pre_len - len(compacted_messages)
                    checkpoint = max(0, checkpoint - removed)
                    messages[:] = compacted_messages  # in-place: self.messages is the same list
                    tprint(
                        f"{DIM}(compacted older conversation history to stay within "
                        f"context limits){RESET}"
                    )

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
            except StreamParseError as e:
                # A bad PAYLOAD on an otherwise-successful response (e.g.
                # the self-hosted endpoint restarting mid-stream) - not a
                # connection failure, so it wasn't covered by the
                # requests.RequestException catches above. Without this,
                # it propagated all the way out of the REPL loop (which
                # doesn't wrap handle_turn at all) and killed the whole
                # process, losing the conversation.
                spinner.stop()
                tprint(f"{YELLOW}Streaming response error: {e}{RESET}")
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
                    skills_used_this_turn.add(tc["function"]["name"])
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
    if not QWEN_URL:
        print(
            f"Error: QWEN_URL is not set. Add it to {CONFIG_DIR / '.env'} "
            "(e.g. QWEN_URL=https://your-qwen-endpoint), or set it as a real environment variable. "
            "This must point at your own Qwen-compatible chat-completions endpoint.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not QWEN_KEY:
        print(
            f"Error: QWEN_KEY is not set. Add it to {CONFIG_DIR / '.env'} "
            "(e.g. QWEN_KEY=your-key-here), or set it as a real environment variable.",
            file=sys.stderr,
        )
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
