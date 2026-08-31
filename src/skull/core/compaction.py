"""Keep the conversation within the model's context window by summarizing
the oldest chunk of history into a compact note, instead of either erroring
out once the model rejects an over-length request or silently discarding
old turns outright.

There's no local tokenizer for this model, so size is estimated with a
simple chars-per-token heuristic - conservative enough to trigger
compaction comfortably before the real limit, not exactly at it.
"""

import json

import requests

from skull import config

# qwen3.8-27b's max_model_len is 32768 (see /v1/models). Reserve room for the
# reply (max_tokens) and injected memory context, and compact well before
# actually hitting the wall - both because the chars-per-token estimate is
# approximate and because waiting until the last moment risks a request
# that's already too large to even ask the model to summarize.
CONTEXT_LIMIT_TOKENS = 32768
RESERVED_FOR_REPLY_TOKENS = 8192 + 1000  # max_tokens + headroom for memory injection
COMPACT_TRIGGER_TOKENS = int((CONTEXT_LIMIT_TOKENS - RESERVED_FOR_REPLY_TOKENS) * 0.85)

CHARS_PER_TOKEN = 3.5  # conservative (English/code average is ~4); errs toward compacting sooner

KEEP_RECENT_MESSAGES = 8  # never summarize away the most recent exchanges
SUMMARY_MAX_TOKENS = 1024

# The chunk being summarized must itself fit in the model's context (it's
# sent as a single request), with plenty of room left for the summarization
# instructions and the reply - cap well under the ~32k limit.
MAX_TRANSCRIPT_CHARS = 24000


def estimate_tokens(messages: list, tools: list = None, extra_chars: int = 0) -> int:
    """Estimate the token size of an actual outgoing request: `messages`
    plus, if given, the `tools` schema list (this can be sizable once
    several self-created skills accumulate - each carries a description and
    a full JSON-schema parameters block) and any extra injected text (e.g.
    the memory-context block folded into the system message)."""
    total_chars = extra_chars
    for m in messages:
        content = m.get("content") or ""
        total_chars += len(content)
        for tc in m.get("tool_calls") or []:
            total_chars += len(tc.get("function", {}).get("arguments") or "")
            total_chars += len(tc.get("function", {}).get("name") or "")
    if tools:
        total_chars += len(json.dumps(tools))
    return int(total_chars / CHARS_PER_TOKEN)


def _find_cut_index(messages: list, start: int, target_end: int) -> int:
    """Find a safe index in (start, target_end] to cut at - i.e. one that
    doesn't split an assistant message with tool_calls from its matching
    tool response messages. Searches backward from target_end so the cut
    point is as close to it as possible."""
    for i in range(target_end, start, -1):
        prev = messages[i - 1]
        if prev.get("role") == "tool":
            continue  # cutting right after a tool message would orphan it
                       # from its assistant call if that call is also being
                       # kept - walk back further to be safe
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            continue  # this assistant message's tool results must stay together with it
        return i
    return start


def _summarize(messages_to_summarize: list) -> str:
    """Ask the model itself for a compact summary of a chunk of older
    conversation. Returns the summary text, or raises on failure (caller
    decides how to handle - see compact_if_needed)."""
    transcript_lines = []
    for m in messages_to_summarize:
        role = m.get("role")
        if role == "system":
            continue
        content = m.get("content") or ""
        if role == "tool":
            transcript_lines.append(f"[tool result]: {content[:500]}")
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
            transcript_lines.append(f"[assistant called tools: {names}]")
            if content:
                transcript_lines.append(f"assistant: {content}")
        else:
            transcript_lines.append(f"{role}: {content}")

    transcript = "\n".join(transcript_lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        # Too large to send to the model whole (it would just fail with its
        # own context-length error) - keep the earliest and latest portions,
        # which tend to carry the most useful context (how the chunk of
        # conversation started and where it left off), and note what was
        # dropped rather than silently going quiet about it.
        half = MAX_TRANSCRIPT_CHARS // 2
        dropped_chars = len(transcript) - MAX_TRANSCRIPT_CHARS
        transcript = (
            transcript[:half]
            + f"\n\n[... {dropped_chars} characters omitted from the middle ...]\n\n"
            + transcript[-half:]
        )

    prompt = (
        "Summarize the following conversation excerpt concisely, preserving "
        "concrete facts, decisions, file paths, and anything that would matter "
        "for continuing the conversation correctly (what was asked, what was "
        "done, what the results were). Write it as plain prose, third person, "
        "no preamble - this will be injected as context for the same "
        "conversation to continue.\n\n---\n\n" + transcript
    )

    resp = requests.post(
        f"{config.QWEN_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.QWEN_KEY}", "Content-Type": "application/json"},
        json={
            "model": config.QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": SUMMARY_MAX_TOKENS,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def compact_if_needed(messages: list, force: bool = False) -> tuple:
    """If `messages` is estimated to be approaching the context limit,
    summarize the oldest compactable chunk (everything after the system
    prompt, up to but not including the last KEEP_RECENT_MESSAGES) into a
    single system-role note, replacing that chunk in place.

    `force=True` skips this function's own (messages-only) threshold check -
    use when the caller has already decided compaction is needed based on a
    more complete estimate (e.g. including tool schemas, which this function
    has no visibility into). Without it, only `messages` itself is checked
    against COMPACT_TRIGGER_TOKENS.

    Returns (messages, compacted: bool). On any failure to summarize (e.g.
    the summarization call itself fails), returns the original messages
    unchanged with compacted=False - better to risk hitting the context
    limit than to lose history to a failed compaction.
    """
    if not force and estimate_tokens(messages) < COMPACT_TRIGGER_TOKENS:
        return messages, False

    if not messages or messages[0].get("role") != "system":
        return messages, False  # unexpected shape - don't touch it

    start = 1  # keep the system prompt
    target_end = max(start, len(messages) - KEEP_RECENT_MESSAGES)
    if target_end <= start:
        return messages, False  # not enough history to compact yet

    cut = _find_cut_index(messages, start, target_end)
    if cut <= start:
        return messages, False  # couldn't find a safe cut point

    chunk = messages[start:cut]
    if not chunk:
        return messages, False

    try:
        summary = _summarize(chunk)
    except Exception:
        return messages, False

    # role must NOT be "system" here - this server rejects any system-role
    # message that isn't first in the list (see memory_context.py, which
    # folds memory into the leading system message for the same reason).
    summary_msg = {
        "role": "user",
        "content": (
            "[Summary of earlier conversation, compacted to save context space - "
            "not something the user just said]\n" + summary
        ),
    }
    new_messages = messages[:start] + [summary_msg] + messages[cut:]
    return new_messages, True
