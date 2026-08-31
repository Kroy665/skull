"""Streaming chat-completion calls against the Qwen endpoint."""

import json
import sys

import requests

from skull import config
from skull.config import CYAN, RESET
from skull.ui.output import tprint, twrite
from skull.ui.spinner import Spinner


class StreamParseError(Exception):
    """A server-sent-events line couldn't be parsed as the expected chunk
    shape (malformed/truncated JSON, or a chunk missing choices/delta).
    Distinct from requests.RequestException - this is a bad PAYLOAD on an
    otherwise-successful HTTP response, not a network/connection failure,
    and needs its own handling in the caller rather than crashing the
    whole process. Real gap this closes: json.loads()/dict-key access on
    each streamed line had no error handling at all, so a single malformed
    chunk (e.g. a self-hosted endpoint restarting mid-stream, or a chunk
    variance in the response shape) would propagate all the way out of
    handle_turn's try/except (which only catches requests exceptions) and
    out of the REPL loop (which doesn't wrap handle_turn at all), killing
    the entire session and losing the conversation."""


def stream_chat(messages: list, tools: list, spinner: Spinner = None):
    """Stream one chat completion turn, printing content tokens as they arrive.

    A "thinking" spinner runs until the first visible output (content token
    or a named tool call) arrives, then stops so streamed text/tool markers
    print cleanly on their own line.

    Returns (content: str, tool_calls: list[dict] | None, finish_reason: str).
    """
    if spinner:
        spinner.start("thinking", style="thinking")

    resp = requests.post(
        f"{config.LLM_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.LLM_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "max_tokens": 8192,
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
            twrite(f"{CYAN}assistant>{RESET} ")
            sys.stdout.flush()
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
        try:
            chunk = json.loads(data)
            choice = chunk["choices"][0]
            delta = choice["delta"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if spinner and not spinner_stopped:
                spinner.stop()
                spinner_stopped = True
            raise StreamParseError(
                f"malformed chunk from the model endpoint mid-stream ({type(e).__name__}: {e}) "
                f"- raw line: {data[:200]!r}"
            ) from e

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        piece = delta.get("content")
        if piece:
            _reveal()
            twrite(piece)
            sys.stdout.flush()
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
        tprint()
    ordered_tool_calls = [tool_calls[i] for i in sorted(tool_calls)] if tool_calls else None
    return "".join(full_content), ordered_tool_calls, finish_reason
