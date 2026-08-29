"""Tests for tools/skills.py - the self-extending skill store (create/list/
get/run/delete round-trip, and the guardrails around writing untrusted,
model-authored code to disk)."""

from skull.tools import skills as sm

SIMPLE_PARAMS = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

DOUBLE_CODE = """
def run(**kwargs):
    return {"doubled": kwargs["n"] * 2}
"""

BROKEN_CODE = """
def run(**kwargs):
    raise ValueError("boom")
"""

NOT_IMPORTABLE_CODE = """
import this_module_does_not_exist

def run(**kwargs):
    return 1
"""

NO_RUN_FUNCTION_CODE = """
def helper(**kwargs):
    return 1
"""


def test_create_skill_writes_files_and_registers(isolated_skills_dir):
    result = sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    assert result == {"status": "created", "name": "doubler"}

    skill_dir = isolated_skills_dir / "doubler"
    assert (skill_dir / "run.py").exists()
    assert (skill_dir / "SKILL.md").exists()
    assert sm.get_skill("doubler")["description"] == "doubles a number"


def test_create_skill_rejects_invalid_name(isolated_skills_dir):
    result = sm.create_skill("NotSnakeCase", "desc", SIMPLE_PARAMS, DOUBLE_CODE)
    assert "error" in result
    assert sm.get_skill("NotSnakeCase") is None


def test_create_skill_rejects_missing_run_function(isolated_skills_dir):
    result = sm.create_skill("no_run", "desc", SIMPLE_PARAMS, NO_RUN_FUNCTION_CODE)
    assert "error" in result
    assert sm.get_skill("no_run") is None


def test_create_skill_rolls_back_on_import_failure(isolated_skills_dir):
    """A skill whose code doesn't even import must not be left on disk or in
    the registry - a broken skill silently sitting in skills/ would look
    exactly like a working one in list_skills()."""
    result = sm.create_skill("broken_import", "desc", SIMPLE_PARAMS, NOT_IMPORTABLE_CODE)
    assert "error" in result
    assert sm.get_skill("broken_import") is None
    assert not (isolated_skills_dir / "broken_import").exists()


def test_create_skill_does_not_rollback_existing_skill_on_reimport_failure(isolated_skills_dir):
    """Overwriting an existing, working skill with broken code should report
    the error, but must not delete the skill directory entirely (is_new is
    False in that branch) - only brand-new skill dirs get rolled back."""
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    result = sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, NOT_IMPORTABLE_CODE)
    assert "error" in result
    # Directory still exists (not rolled back) - the previous broken write
    # happened before we could restore the old code, so this is a known trade-off,
    # not a bug: verify the actual documented behavior.
    assert (isolated_skills_dir / "doubler").exists()


def test_create_skill_replaces_existing_entry(isolated_skills_dir):
    sm.create_skill("doubler", "v1", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "v2", SIMPLE_PARAMS, DOUBLE_CODE)
    matches = [e for e in sm.list_skills() if e["name"] == "doubler"]
    assert len(matches) == 1
    assert matches[0]["description"] == "v2"


def test_run_skill_returns_result(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    outcome = sm.run_skill("doubler", {"n": 21})
    assert outcome == {"result": {"doubled": 42}}


def test_run_skill_missing_skill_returns_error(isolated_skills_dir):
    outcome = sm.run_skill("nonexistent", {})
    assert "error" in outcome


def test_run_skill_captures_internal_exception(isolated_skills_dir):
    sm.create_skill("broken", "raises", SIMPLE_PARAMS, BROKEN_CODE)
    outcome = sm.run_skill("broken", {"n": 1})
    assert "error" in outcome
    assert "boom" in outcome["error"]
    assert "traceback" in outcome


def test_delete_skill_removes_files_and_registry_entry(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    result = sm.delete_skill("doubler")
    assert result == {"status": "deleted", "name": "doubler"}
    assert sm.get_skill("doubler") is None
    assert not (isolated_skills_dir / "doubler").exists()


def test_delete_skill_missing_returns_error(isolated_skills_dir):
    result = sm.delete_skill("nonexistent")
    assert "error" in result


def test_list_skills_empty_by_default(isolated_skills_dir):
    assert sm.list_skills() == []
