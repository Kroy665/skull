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
