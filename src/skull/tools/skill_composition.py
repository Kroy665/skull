"""Lets a self-created skill call another skill by name, so skills can be
composed instead of duplicating logic.

Exposed to skill code as `from skull.tools.skill_composition import
call_skill` (skull is importable from anywhere in the process since it's
installed in editable mode - see pyproject.toml).
"""

from skull.tools import skills as sm


class SkillError(Exception):
    """Raised by call_skill() when the target skill doesn't exist or raises
    internally - lets calling skill code use a normal try/except instead of
    checking a {"error": ...} dict shape, which run_skill() returns (that
    shape is for the model-facing tool result, not for skill-to-skill calls)."""


def call_skill(name: str, **kwargs):
    """Call another skill by name with keyword arguments, returning its
    result directly (not wrapped in {"result": ...}). Raises SkillError if
    the skill doesn't exist or its run() raises.

    Example, inside a skill's run.py:
        from skull.tools.skill_composition import call_skill
        def run(**kwargs):
            celsius = call_skill("fahrenheit_to_celsius", fahrenheit=98.6)["celsius"]
            ...
    """
    outcome = sm.run_skill(name, kwargs)
    if "error" in outcome:
        raise SkillError(f"call_skill({name!r}) failed: {outcome['error']}")
    return outcome["result"]
