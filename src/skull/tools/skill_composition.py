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
    the skill doesn't exist, its run() raises, or its result was too large
    and got truncated (see run_skill's MAX_RESULT_CHARS).

    Example, inside a skill's run.py:
        from skull.tools.skill_composition import call_skill
        def run(**kwargs):
            celsius = call_skill("fahrenheit_to_celsius", fahrenheit=98.6)["celsius"]
            ...
    """
    outcome = sm.run_skill(name, kwargs)
    if "error" in outcome:
        raise SkillError(f"call_skill({name!r}) failed: {outcome['error']}")
    if outcome.get("result_truncated"):
        # run_skill's result-size ceiling (see MAX_RESULT_CHARS) kicked in -
        # outcome has no "result" key in this shape, just a raw (possibly
        # invalid-JSON-at-the-cut-point) truncated string, which composing
        # code almost never wants to consume programmatically. Without this
        # check, `outcome["result"]` below raised a bare KeyError instead of
        # SkillError - the wrong exception type for a calling skill's own
        # `except SkillError:` to catch, and one that carries no context
        # about which skill or why.
        raise SkillError(
            f"call_skill({name!r}) result was too large ({outcome['original_length']} chars) and got "
            "truncated - not usable as a structured result. Have the called skill limit/paginate its "
            "own output instead of composing with its truncated form."
        )
    return outcome["result"]
