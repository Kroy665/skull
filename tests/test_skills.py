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


def test_create_skill_restores_previous_working_version_on_reimport_failure(isolated_skills_dir):
    """Overwriting an existing, working skill with code that fails the
    import sanity-check must report the error AND leave the previously
    working version live - not a half-broken skill under the same name,
    and not deleted entirely (only brand-new skill dirs get fully rolled
    back, since there's nothing previous to restore for those)."""
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    result = sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, NOT_IMPORTABLE_CODE)
    assert "error" in result

    assert (isolated_skills_dir / "doubler").exists()
    # The skill must still actually work - the previous version was restored.
    outcome = sm.run_skill("doubler", {"n": 5})
    assert outcome == {"result": {"doubled": 10}}


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


TRIPLE_CODE = """
def run(**kwargs):
    return {"tripled": kwargs["n"] * 3}
"""


# ---------------------------------------------------------------------------
# Skill versioning / rollback
# ---------------------------------------------------------------------------

def test_create_skill_does_not_archive_a_brand_new_skill(isolated_skills_dir):
    sm.create_skill("doubler", "v1", SIMPLE_PARAMS, DOUBLE_CODE)
    result = sm.list_skill_versions("doubler")
    assert result == {"name": "doubler", "versions": []}


def test_create_skill_archives_previous_version_on_overwrite(isolated_skills_dir):
    sm.create_skill("doubler", "doubles", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "triples now", SIMPLE_PARAMS, TRIPLE_CODE)

    result = sm.list_skill_versions("doubler")
    assert result["name"] == "doubler"
    assert len(result["versions"]) == 1
    assert result["versions"][0]["description"] == "doubles"

    # Live version is the new one.
    outcome = sm.run_skill("doubler", {"n": 5})
    assert outcome == {"result": {"tripled": 15}}


def test_list_skill_versions_orders_most_recent_first(isolated_skills_dir):
    sm.create_skill("doubler", "v1", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "v2", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "v3", SIMPLE_PARAMS, DOUBLE_CODE)

    result = sm.list_skill_versions("doubler")
    descriptions = [v["description"] for v in result["versions"]]
    assert descriptions == ["v2", "v1"]


def test_list_skill_versions_missing_skill_returns_error(isolated_skills_dir):
    result = sm.list_skill_versions("nonexistent")
    assert "error" in result


def test_versions_are_bounded_to_max_kept(isolated_skills_dir):
    sm.create_skill("doubler", "v0", SIMPLE_PARAMS, DOUBLE_CODE)
    for i in range(1, sm.MAX_VERSIONS_KEPT + 3):
        sm.create_skill("doubler", f"v{i}", SIMPLE_PARAMS, DOUBLE_CODE)

    result = sm.list_skill_versions("doubler")
    assert len(result["versions"]) == sm.MAX_VERSIONS_KEPT
    # The oldest versions must have been pruned, keeping only the most recent.
    descriptions = [v["description"] for v in result["versions"]]
    assert "v0" not in descriptions
    assert "v1" not in descriptions


def test_rollback_skill_restores_most_recent_version_by_default(isolated_skills_dir):
    sm.create_skill("doubler", "doubles", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "triples now", SIMPLE_PARAMS, TRIPLE_CODE)

    result = sm.rollback_skill("doubler")
    assert result["status"] == "rolled_back"
    assert result["name"] == "doubler"

    assert sm.get_skill("doubler")["description"] == "doubles"
    outcome = sm.run_skill("doubler", {"n": 5})
    assert outcome == {"result": {"doubled": 10}}


def test_rollback_skill_restores_specific_version(isolated_skills_dir):
    sm.create_skill("doubler", "v1", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "v2", SIMPLE_PARAMS, TRIPLE_CODE)
    sm.create_skill("doubler", "v3", SIMPLE_PARAMS, DOUBLE_CODE)

    result = sm.rollback_skill("doubler", version=1)
    assert result["status"] == "rolled_back"
    assert result["restored_version"] == 1
    assert sm.get_skill("doubler")["description"] == "v1"


def test_rollback_skill_is_itself_undoable(isolated_skills_dir):
    """A rollback archives whatever was live before restoring the older
    version, so rolling back twice in a row returns to where you started."""
    sm.create_skill("doubler", "doubles", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "triples now", SIMPLE_PARAMS, TRIPLE_CODE)

    sm.rollback_skill("doubler")  # back to "doubles"
    assert sm.get_skill("doubler")["description"] == "doubles"

    sm.rollback_skill("doubler")  # undo the rollback - back to "triples now"
    assert sm.get_skill("doubler")["description"] == "triples now"


def test_rollback_skill_missing_skill_returns_error(isolated_skills_dir):
    result = sm.rollback_skill("nonexistent")
    assert "error" in result


def test_rollback_skill_no_versions_available_returns_error(isolated_skills_dir):
    sm.create_skill("doubler", "doubles", SIMPLE_PARAMS, DOUBLE_CODE)
    result = sm.rollback_skill("doubler")
    assert "error" in result


def test_rollback_skill_nonexistent_version_number_returns_error(isolated_skills_dir):
    sm.create_skill("doubler", "v1", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "v2", SIMPLE_PARAMS, DOUBLE_CODE)
    result = sm.rollback_skill("doubler", version=99)
    assert "error" in result


def test_run_skill_never_serves_stale_bytecode_after_overwrite(isolated_skills_dir):
    """Real bug found via testing (not the versioning feature's fault, but
    surfaced by it): _import_skill used the same module name for every
    import of a given skill, and Python's default SourceFileLoader writes
    a __pycache__/*.pyc keyed loosely enough that re-importing the same
    module name after the underlying run.py changed could serve stale
    compiled bytecode from the previous version - confirmed directly with
    a minimal repro outside this codebase. Overwrite a skill many times in
    a row with materially different logic each time and check every one
    actually takes effect, not just the first overwrite."""
    for i in range(5):
        code = f"def run(**kwargs):\n    return {{'value': kwargs['n'] + {i}}}\n"
        sm.create_skill("counter", f"adds {i}", SIMPLE_PARAMS, code)
        outcome = sm.run_skill("counter", {"n": 100})
        assert outcome == {"result": {"value": 100 + i}}, f"stale result after overwrite #{i}"


def test_delete_skill_also_removes_version_history(isolated_skills_dir):
    sm.create_skill("doubler", "v1", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.create_skill("doubler", "v2", SIMPLE_PARAMS, DOUBLE_CODE)
    sm.delete_skill("doubler")
    assert not (isolated_skills_dir / "doubler").exists()
