"""Tests for tools/skill_composition.py - call_skill(), which lets one
skill's code call another skill directly, raising SkillError instead of
returning the {"error": ...} dict shape run_skill() uses for model-facing
results."""

import pytest

from skull.tools import skills as sm
from skull.tools.skill_composition import SkillError, call_skill

SIMPLE_PARAMS = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

DOUBLE_CODE = """
def run(**kwargs):
    return {"doubled": kwargs["n"] * 2}
"""

BROKEN_CODE = """
def run(**kwargs):
    raise ValueError("boom")
"""

CALLS_DOUBLER_CODE = """
from skull.tools.skill_composition import call_skill

def run(**kwargs):
    result = call_skill("doubler", n=kwargs["n"])
    return {"quadrupled": result["doubled"] * 2}
"""


def test_call_skill_returns_unwrapped_result(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    result = call_skill("doubler", n=10)
    assert result == {"doubled": 20}


def test_call_skill_raises_on_missing_skill(isolated_skills_dir):
    with pytest.raises(SkillError):
        call_skill("nonexistent", n=1)


def test_call_skill_raises_on_internal_exception(isolated_skills_dir):
    sm.create_skill("broken", "raises", SIMPLE_PARAMS, BROKEN_CODE)
    with pytest.raises(SkillError, match="boom"):
        call_skill("broken", n=1)


def test_skill_can_compose_another_skill_end_to_end(isolated_skills_dir):
    """The actual composition scenario: one skill's run.py imports and calls
    call_skill() to invoke another skill, exercised through run_skill() just
    like the model would trigger it."""
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("quadrupler", "quadruples via doubler", SIMPLE_PARAMS, CALLS_DOUBLER_CODE)

    outcome = sm.run_skill("quadrupler", {"n": 5})
    assert outcome == {"result": {"quadrupled": 20}}


def test_call_skill_raises_skill_error_not_key_error_on_truncated_result(isolated_skills_dir):
    """Real bug found via code review: run_skill's truncation wrapper (see
    MAX_RESULT_CHARS) returns a dict with neither "error" nor "result" -
    call_skill only checked for "error" before blindly doing
    outcome["result"], so a composed skill whose inner call returned an
    oversized result raised a bare KeyError instead of the documented
    SkillError. A KeyError can't be caught by a composing skill's own
    `except SkillError:` handler and carries no context about which skill
    or why - confirmed this was the actual failure shape before the fix."""
    huge_code = "def run(**kwargs):\n    return {'data': 'x' * 100000}\n"
    sm.create_skill("huge", "returns an oversized result", SIMPLE_PARAMS, huge_code)

    with pytest.raises(SkillError, match="too large"):
        call_skill("huge", n=1)


def test_composing_skill_gets_skill_error_not_key_error_end_to_end(isolated_skills_dir):
    """Same as above, but exercised through run_skill() the way the model
    would actually trigger it (a composing skill's own code hits the
    truncated inner call) - confirms the composing skill's run() sees a
    catchable SkillError, not an uncaught-by-name KeyError."""
    huge_code = "def run(**kwargs):\n    return {'data': 'x' * 100000}\n"
    catches_it_code = """
from skull.tools.skill_composition import call_skill, SkillError

def run(**kwargs):
    try:
        call_skill("huge", n=1)
        return {"caught": False}
    except SkillError as e:
        return {"caught": True, "message": str(e)}
"""
    sm.create_skill("huge", "returns an oversized result", SIMPLE_PARAMS, huge_code)
    sm.create_skill("catcher", "calls huge and catches SkillError", SIMPLE_PARAMS, catches_it_code)

    outcome = sm.run_skill("catcher", {"n": 1})
    assert outcome["result"]["caught"] is True
    assert "too large" in outcome["result"]["message"]
