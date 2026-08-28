"""Background next-question suggestion, shown as ghost text in the input prompt.

After each assistant reply, a short async call to the same Qwen model predicts
one likely follow-up the user might type next. Runs on a daemon thread so it
never blocks the input prompt - the suggestion just pops in whenever it's
ready (or stays empty if it isn't yet, or the call fails).
"""

import json
import threading

import requests

SUGGESTION_MAX_TOKENS = 40
SUGGESTION_TIMEOUT_SECONDS = 20
CONTEXT_TURNS = 6  # how many recent messages to include as context


class SuggestionEngine:
    def __init__(self, qwen_url: str, qwen_key: str, qwen_model: str):
        self.qwen_url = qwen_url
        self.qwen_key = qwen_key
        self.qwen_model = qwen_model
        self._lock = threading.Lock()
        self._suggestion = ""
        self._generation = 0  # bumped on each refresh() to invalidate stale threads

    def get(self) -> str:
        with self._lock:
            return self._suggestion

    def clear(self):
        with self._lock:
            self._suggestion = ""

    def refresh(self, context_messages: list):
        """Kick off a background prediction using the given chat-style
        messages as context (most recent conversation, or last session's
        history if the current one is empty). Non-blocking."""
        with self._lock:
            self._generation += 1
            my_generation = self._generation

        thread = threading.Thread(
            target=self._run, args=(context_messages, my_generation), daemon=True
        )
        thread.start()

    def _run(self, context_messages: list, my_generation: int):
        try:
            suggestion = self._predict(context_messages)
        except Exception:
            suggestion = ""

        with self._lock:
            if my_generation == self._generation:  # discard if superseded
                self._suggestion = suggestion

    def _predict(self, context_messages: list) -> str:
        if not context_messages:
            return ""

        recent = context_messages[-CONTEXT_TURNS:]
        instruction = {
            "role": "user",
            "content": (
                "Based on this conversation, predict the single most likely next "
                "question or task the user will type. Output ONLY that question/task "
                "as plain text, no quotes, no preamble, no explanation. Keep it under "
                "15 words. If nothing sensible comes to mind, output nothing."
            ),
        }
        messages = recent + [instruction]

        resp = requests.post(
            f"{self.qwen_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.qwen_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.qwen_model,
                "messages": messages,
                "max_tokens": SUGGESTION_MAX_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=SUGGESTION_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        return text.strip().strip('"').strip()
