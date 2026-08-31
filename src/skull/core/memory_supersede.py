"""Detects when a newly remembered persona fact contradicts/updates an
existing one, so old facts don't linger forever with equal standing to a
correction - e.g. "prefers terse answers" should stop being surfaced once
the user later says "prefers detailed explanations".

Two-stage check, both needed:
1. Embedding similarity pre-filter (cheap, local) - only facts on roughly
   the same topic are candidates at all. Calibrated against real examples
   with realistic (not hand-matched) phrasing: genuinely contradicting
   pairs ranged ~0.37-0.87 depending on how much vocabulary the two
   sentences happened to share, unrelated pairs scored ~0.02-0.1. The wide
   range on the contradicting side means this threshold is deliberately
   loose (SUPERSEDE_CANDIDATE_MIN_SCORE) - it exists only to skip obviously
   unrelated facts cheaply, not to make the actual contradiction call.
   That's stage 2's job.
2. LLM confirmation (one small non-streaming call, only for candidates that
   pass stage 1) - embedding similarity alone can't tell "prefers X" from
   "prefers not-X" apart from "these are both about preferences", so a
   candidate above threshold still needs the model to confirm it's an
   actual update/contradiction, not just a related-but-compatible fact.
   Because stage 1 is loose, stage 2 does the real filtering - a false
   positive from stage 1 just costs one cheap extra LLM call, whereas a
   false negative (threshold too high) would silently make the whole
   feature not fire on real-world phrasing, which is the failure mode that
   was actually hit during development (see git history / commit message).

Superseded facts are never deleted - VectorStore.mark_superseded() keeps
them in the database (visible via history()) but excludes them from
search()/all()/count() going forward. This only applies to the persona
store; conversations are an append-only log of what was actually said, not
a "current truth" table, so there's nothing to supersede there.

Also handles a related but distinct problem hit in practice: forget()
requires exact stored text, and a model that paraphrases instead of quoting
verbatim gets a silent "no matching entry found" - then narrates success
anyway without checking the tool result. forget_fuzzy() adds an embedding
pre-filter + LLM-confirmed fallback for when the exact match fails, but
this needed its own confirmation question ("is this the SAME fact, just
reworded") rather than reusing _confirm_supersedes' question ("does this
update/contradict") - real testing showed embedding similarity alone can't
tell those apart safely for a destructive delete: "User loves dark
chocolate" vs "User loves milk chocolate" scored a HIGHER similarity
(0.817) than several genuine paraphrases of unrelated facts (0.69-0.77),
so no single threshold is safe without a same-fact-or-different-fact
confirmation from the model first.
"""

import requests

from skull import config
from skull.storage import store as mem

SUPERSEDE_CANDIDATE_MIN_SCORE = 0.35
SUPERSEDE_CHECK_MAX_TOKENS = 5

FORGET_FUZZY_CANDIDATE_MIN_SCORE = 0.35
FORGET_FUZZY_TOP_K = 3


def _confirm_supersedes(old_fact: str, new_fact: str) -> bool:
    """Ask the model whether `new_fact` supersedes/contradicts `old_fact`
    (an update to the same specific thing) rather than merely being a
    related-but-compatible fact. Defaults to False (don't supersede) on any
    failure - a missed supersede just leaves an extra fact in memory, which
    is far less harmful than wrongly burying a still-valid one."""
    prompt = (
        "Two facts about the same user:\n"
        f'A (existing): "{old_fact}"\n'
        f'B (new): "{new_fact}"\n\n'
        "Does B supersede or contradict A - i.e. is B an update/correction to "
        "the same specific thing A states, making A no longer current? "
        "Answer with exactly one word: yes or no."
    )
    try:
        resp = requests.post(
            f"{config.LLM_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_KEY}", "Content-Type": "application/json"},
            json={
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": SUPERSEDE_CHECK_MAX_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def remember_with_supersede(fact: str, metadata: dict | None = None) -> dict:
    """Store `fact` in the persona store, first checking whether it
    supersedes an existing fact. Returns the same shape as VectorStore.add(),
    plus "superseded" (the old fact's text) when a supersede happened."""
    store = mem.persona()

    candidates = store.search(fact, k=3, min_score=SUPERSEDE_CANDIDATE_MIN_SCORE)
    result = store.add(fact, metadata)
    if "error" in result:
        return result

    for candidate in candidates:
        old_text = candidate["text"]
        if old_text == fact:
            continue
        if _confirm_supersedes(old_text, fact):
            store.mark_superseded(old_text, fact)
            result["superseded"] = old_text
            break  # one supersede per remember() call is enough for the common case

    return result


def _confirm_same_fact(candidate_fact: str, requested_fact: str) -> bool:
    """Ask the model whether `candidate_fact` (an actual stored fact) is the
    SAME underlying fact as `requested_fact` (what the caller asked to
    delete), just phrased differently - not merely related or on the same
    topic. Defaults to False on any failure or ambiguity, since a wrongly
    confirmed match here deletes the wrong fact."""
    prompt = (
        "A user asked to delete this fact from memory:\n"
        f'Requested: "{requested_fact}"\n\n'
        "The closest actual stored fact is:\n"
        f'Stored: "{candidate_fact}"\n\n'
        "Is \"Stored\" the SAME underlying fact as \"Requested\", just worded "
        "differently - not merely a related fact on a similar topic? Answer "
        "with exactly one word: yes or no."
    )
    try:
        resp = requests.post(
            f"{config.LLM_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_KEY}", "Content-Type": "application/json"},
            json={
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": SUPERSEDE_CHECK_MAX_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def forget_fuzzy(fact: str) -> dict:
    """Delete a persona fact by text, falling back to an embedding-filtered,
    LLM-confirmed match when no exact text match exists - covers a model
    paraphrasing instead of quoting a fact verbatim. Returns the same shape
    as VectorStore.delete(), plus "matched_text" when the fuzzy path found
    and deleted a differently-worded match."""
    store = mem.persona()

    exact_result = store.delete(fact)
    if exact_result.get("deleted"):
        return exact_result

    candidates = store.search(fact, k=FORGET_FUZZY_TOP_K, min_score=FORGET_FUZZY_CANDIDATE_MIN_SCORE)
    for candidate in candidates:
        candidate_text = candidate["text"]
        if _confirm_same_fact(candidate_text, fact):
            result = store.delete(candidate_text)
            if result.get("deleted"):
                result["matched_text"] = candidate_text
            return result

    return {"error": "no matching entry found", "deleted": False}
