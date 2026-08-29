"""Tests for core/memory_supersede.py - the two-stage (embedding pre-filter
+ LLM confirmation) check that marks an old persona fact as superseded when
a new one updates/contradicts it, instead of both lingering forever with
equal standing."""

from skull.core import memory_supersede as ms


def _mock_llm_response(monkeypatch, answer: str):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": answer}}]}

    monkeypatch.setattr(ms.requests, "post", lambda *a, **k: FakeResponse())


def test_remember_with_supersede_stores_first_fact_with_no_candidates(isolated_memory_dir):
    result = ms.remember_with_supersede("likes tea", {"category": "preference"})
    assert result["status"] == "stored"
    assert "superseded" not in result


def test_remember_with_supersede_marks_old_fact_when_llm_confirms(isolated_memory_dir, monkeypatch):
    """isolated_memory_dir's fake embedder is random noise, which never
    clears SUPERSEDE_CANDIDATE_MIN_SCORE on its own (same reasoning as the
    analogous fixture-vs-real-model gap in test_registry.py's skill-filter
    tests) - override it here so the two facts are guaranteed to look like
    the same topic, and the LLM confirmation step is what's under test."""
    from skull.storage import store as mem

    import numpy as np

    def fake_embed(texts):
        vectors = []
        for t in texts:
            v = np.zeros(mem.EMBED_DIM, dtype=np.float32)
            v[0] = 1.0
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(mem, "embed", fake_embed)

    ms.remember_with_supersede("prefers terse answers", {"category": "preference"})
    _mock_llm_response(monkeypatch, "yes")

    result = ms.remember_with_supersede("prefers detailed explanations", {"category": "preference"})
    assert result["status"] == "stored"
    assert result["superseded"] == "prefers terse answers"

    assert mem.persona().count() == 1
    assert mem.persona().all()[0]["text"] == "prefers detailed explanations"


def test_remember_with_supersede_keeps_both_when_llm_denies(isolated_memory_dir, monkeypatch):
    ms.remember_with_supersede("likes tea", {"category": "preference"})
    _mock_llm_response(monkeypatch, "no")

    result = ms.remember_with_supersede("likes coffee too", {"category": "preference"})
    assert "superseded" not in result

    from skull.storage import store as mem
    assert mem.persona().count() == 2


def test_remember_with_supersede_defaults_to_not_superseding_on_llm_failure(isolated_memory_dir, monkeypatch):
    def failing_post(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(ms.requests, "post", failing_post)
    ms.remember_with_supersede("likes tea", {"category": "preference"})

    result = ms.remember_with_supersede("likes coffee", {"category": "preference"})
    assert "superseded" not in result

    from skull.storage import store as mem
    assert mem.persona().count() == 2


def test_remember_with_supersede_propagates_add_error(isolated_memory_dir):
    result = ms.remember_with_supersede("   ")
    assert "error" in result


def test_confirm_supersedes_parses_yes_no(monkeypatch):
    _mock_llm_response(monkeypatch, "Yes.")
    assert ms._confirm_supersedes("old fact", "new fact") is True

    _mock_llm_response(monkeypatch, "No, these are unrelated.")
    assert ms._confirm_supersedes("old fact", "new fact") is False


# ---------------------------------------------------------------------------
# forget_fuzzy - real bug hit in practice: the model paraphrased a fact
# instead of quoting it exactly, forget() silently found nothing, and the
# model told the user it was done anyway. forget_fuzzy adds an LLM-confirmed
# fallback for exactly this case.
# ---------------------------------------------------------------------------

def test_forget_fuzzy_deletes_on_exact_match_without_any_llm_call(isolated_memory_dir, monkeypatch):
    from skull.storage import store as mem

    mem.persona().add("likes tea")

    def fail_if_called(*a, **k):
        raise AssertionError("should not call the LLM when an exact match exists")

    monkeypatch.setattr(ms.requests, "post", fail_if_called)

    result = ms.forget_fuzzy("likes tea")
    assert result["deleted"] is True
    assert "matched_text" not in result
    assert mem.persona().count() == 0


def test_forget_fuzzy_falls_back_to_confirmed_paraphrase_match(isolated_memory_dir, monkeypatch):
    """The exact real-world case that failed: model said "Prefers terse
    answers with no fluff" but the stored fact was "Koushik prefers terse
    answers with no fluff" - no exact match, but a clear paraphrase."""
    from skull.storage import store as mem

    import numpy as np

    def fake_embed(texts):
        vectors = []
        for t in texts:
            v = np.zeros(mem.EMBED_DIM, dtype=np.float32)
            v[0] = 1.0
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(mem, "embed", fake_embed)
    mem.persona().add("Koushik prefers terse answers with no fluff")
    _mock_llm_response(monkeypatch, "yes")

    result = ms.forget_fuzzy("Prefers terse answers with no fluff")
    assert result["deleted"] is True
    assert result["matched_text"] == "Koushik prefers terse answers with no fluff"
    assert mem.persona().count() == 0


def test_forget_fuzzy_does_not_delete_a_merely_related_fact(isolated_memory_dir, monkeypatch):
    """The exact false-positive risk found during calibration: "loves dark
    chocolate" vs "loves milk chocolate" scored HIGH embedding similarity
    despite being different facts - the LLM confirmation must catch this."""
    from skull.storage import store as mem

    import numpy as np

    def fake_embed(texts):
        vectors = []
        for t in texts:
            v = np.zeros(mem.EMBED_DIM, dtype=np.float32)
            v[0] = 1.0
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(mem, "embed", fake_embed)
    mem.persona().add("User loves dark chocolate")
    _mock_llm_response(monkeypatch, "no")  # LLM correctly says these are different facts

    result = ms.forget_fuzzy("User loves milk chocolate")
    assert result == {"error": "no matching entry found", "deleted": False}
    assert mem.persona().count() == 1


def test_forget_fuzzy_returns_error_with_no_candidates_at_all(isolated_memory_dir):
    result = ms.forget_fuzzy("nonexistent fact")
    assert result == {"error": "no matching entry found", "deleted": False}


def test_confirm_same_fact_parses_yes_no(monkeypatch):
    _mock_llm_response(monkeypatch, "yes")
    assert ms._confirm_same_fact("stored", "requested") is True

    _mock_llm_response(monkeypatch, "no")
    assert ms._confirm_same_fact("stored", "requested") is False
