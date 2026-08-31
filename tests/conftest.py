"""Shared pytest fixtures.

Tests must never touch the real skills/, pipelines/, or memory/ directories
at the project root - those hold real accumulated user data. Every fixture
here redirects the relevant module's directory constant to an isolated
tmp_path for the duration of the test.

Modules import these as plain names (e.g. `from skull.config import
SKILLS_DIR`), not looked up dynamically - so patching skull.config.SKILLS_DIR
alone would not affect skull.tools.skills, which already bound its own
module-level SKILLS_DIR at import time. Each fixture patches the constant on
every module that holds its own reference.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def _no_live_context_limit_lookup(monkeypatch):
    """core/compaction.py's compact_trigger_tokens() (called from
    compact_if_needed(), in turn called from session.py's handle_turn())
    makes a real HTTP call via config.detect_model_context_limit() the
    first time it's needed, then caches the result for the rest of the
    process's life. Autoused globally and forced to return None (the
    "endpoint didn't report a context limit" case) for every test in the
    whole suite - otherwise any test that exercises handle_turn/
    compact_if_needed would depend on network availability and whatever
    LLM_URL happens to be set to on the machine running the suite, not
    just on what it mocks directly. Also resets the module-level cache
    each test so one test's result can't leak into the next.
    """
    from skull.core import compaction as comp

    monkeypatch.setattr(comp, "_cached_context_limit_tokens", None)
    monkeypatch.setattr(comp.config, "detect_model_context_limit", lambda *a, **k: None)
    monkeypatch.setattr(comp.config, "LLM_PROVIDER", "custom")  # -> FALLBACK_CONTEXT_LIMIT_TOKENS
    yield


@pytest.fixture
def isolated_skills_dir(tmp_path, monkeypatch):
    """Redirect skill storage to an empty temp directory."""
    from skull.tools import skills as sm

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr(sm, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(sm, "INDEX_PATH", skills_dir / "index.json")
    return skills_dir


@pytest.fixture
def isolated_skills_env(tmp_path, monkeypatch):
    """Redirect skill secrets storage to an empty temp file."""
    from skull.tools import skill_env as scenv

    monkeypatch.setattr(scenv, "SKILLS_ENV_PATH", tmp_path / "skills.env")
    return tmp_path / "skills.env"


@pytest.fixture
def isolated_pipelines_dir(tmp_path, monkeypatch, isolated_skills_dir):
    """Redirect pipeline storage to an empty temp directory. Depends on
    isolated_skills_dir since pipeline validation looks up skills."""
    from skull.tools import pipeline as pl

    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir()
    monkeypatch.setattr(pl, "PIPELINES_DIR", pipelines_dir)
    return pipelines_dir


@pytest.fixture
def isolated_memory_dir(tmp_path, monkeypatch):
    """Redirect the vector store to an empty temp directory, and stub out
    the embedding model so tests don't need network access or a real
    sentence-transformers download."""
    from skull.storage import store as mem

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr(mem, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(mem, "_stores", {})  # clear any cached VectorStore instances

    import numpy as np

    def fake_embed(texts):
        # Deterministic, cheap "embedding": each text maps to a fixed-size
        # vector derived from its hash, normalized. Good enough to test
        # storage/search plumbing without a real model. Dimension must match
        # mem.EMBED_DIM - the vec0 virtual table's column width is fixed at
        # table-creation time.
        vectors = []
        for t in texts:
            h = abs(hash(t))
            rng = np.random.default_rng(h % (2**32))
            v = rng.normal(size=mem.EMBED_DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(mem, "embed", fake_embed)
    return memory_dir


@pytest.fixture
def isolated_conversations_dir(tmp_path, monkeypatch):
    """Redirect per-directory saved conversations to an empty temp directory."""
    from skull.core import conversation_store as cs

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(cs, "CONVERSATIONS_DIR", conversations_dir)
    return conversations_dir
