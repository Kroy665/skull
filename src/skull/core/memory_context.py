"""Retrieval of relevant long-term memory to fold into the system prompt,
and the plan-mode instruction addendum."""

from skull.storage import store as mem

MEMORY_TOP_K = 4
MEMORY_MIN_SCORE = 0.35  # cosine similarity floor - drop weak/irrelevant matches


def build_memory_context(query: str) -> str:
    """Retrieve relevant persona facts + past conversation turns for `query`,
    returned as a block of text to fold into the leading system message
    (the Qwen server rejects any system-role message that isn't first in the
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


def last_session_context() -> list:
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


PLAN_MODE_ADDENDUM = (
    "\n\nPLAN MODE IS ACTIVE. You may research using read-only tools "
    "(web_search, scrape_page, list_skills, recall_memory) and any "
    "self-created skill statically verified to have no side effects (pure "
    "computation, e.g. a unit converter) - those stay callable since they "
    "can't act on anything. IMPORTANT: if such a skill is in your tool "
    "list right now, actually CALL it to get the real answer - do not just "
    "describe what calling it would return, do not defer it to a plan, and "
    "do not tell the user to switch to /auto first just to run something "
    "that's already safe to run. Only genuinely mutating actions (writing "
    "files, running commands, creating/deleting skills, running code, "
    "remembering/forgetting facts) need to wait for /auto. "
    "create_skill, remember, run_python, and any skill that writes files, "
    "runs a subprocess, or hits the network are temporarily hidden from "
    "your tool list - they still exist and will work normally once plan "
    "mode ends, they are just withheld right now so you can't take any "
    "mutating/action-taking step. Don't conclude a tool is missing or "
    "needs to be (re)created just because you can't see it in plan mode. "
    "For anything that DOES require a hidden tool, investigate as needed "
    "and respond with a clear, concrete plan of what you would do, then "
    "wait for the user to leave plan mode (they'll type /auto) before it "
    "gets executed."
)
