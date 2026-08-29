"""Tests for tools/skills.py - the self-extending skill store (create/list/
get/run/delete round-trip, and the guardrails around writing untrusted,
model-authored code to disk)."""

import json

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
    assert result == {"status": "created", "name": "doubler", "side_effects": "read_only"}

    skill_dir = isolated_skills_dir / "doubler"
    assert (skill_dir / "run.py").exists()
    assert (skill_dir / "SKILL.md").exists()
    assert sm.get_skill("doubler")["description"] == "doubles a number"


def test_reclassify_all_skills_backfills_missing_side_effects(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    # Simulate a pre-existing skill with no side_effects field.
    index = sm._load_index()
    for entry in index:
        del entry["side_effects"]
    sm._save_index(index)
    assert "side_effects" not in sm.get_skill("doubler")

    result = sm.reclassify_all_skills()
    assert result["status"] == "reclassified"
    assert result["count"] == 1
    assert sm.get_skill("doubler")["side_effects"] == "read_only"


def test_reclassify_all_skills_reports_what_changed(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    index = sm._load_index()
    for entry in index:
        entry["side_effects"] = "mutating"  # wrong on purpose
    sm._save_index(index)

    result = sm.reclassify_all_skills()
    assert result["changed"] == [{"name": "doubler", "from": "mutating", "to": "read_only"}]


def test_create_skill_classifies_pure_code_as_read_only(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    assert sm.get_skill("doubler")["side_effects"] == "read_only"


def test_create_skill_classifies_file_writing_code_as_mutating(isolated_skills_dir):
    code = "def run(**kwargs):\n    with open(kwargs['path'], 'w') as f:\n        f.write('x')\n    return {'ok': True}\n"
    sm.create_skill("writer", "writes a file", SIMPLE_PARAMS, code)
    assert sm.get_skill("writer")["side_effects"] == "mutating"


# ---------------------------------------------------------------------------
# create_skill(required_env=...) - credentials never as a plain kwarg. Real
# problem this solves: a model asked the user to paste an SMTP password
# directly into chat, landing it in plain text in conversation history.
# ---------------------------------------------------------------------------

SEND_CODE_USING_ENV = """
from skull.tools.skill_env import get_env

def run(**kwargs):
    password = get_env("FAKE_API_KEY")
    return {"had_key": password is not None}
"""


def test_create_skill_stores_required_env(isolated_skills_dir, isolated_skills_env):
    result = sm.create_skill(
        "notify", "sends a notification", SIMPLE_PARAMS, SEND_CODE_USING_ENV, required_env=["FAKE_API_KEY"]
    )
    assert result["status"] == "created"
    assert sm.get_skill("notify")["required_env"] == ["FAKE_API_KEY"]


def test_create_skill_reports_missing_env_in_result(isolated_skills_dir, isolated_skills_env):
    result = sm.create_skill(
        "notify", "sends a notification", SIMPLE_PARAMS, SEND_CODE_USING_ENV, required_env=["FAKE_API_KEY"]
    )
    assert result["missing_env"] == ["FAKE_API_KEY"]
    assert "note" in result


def test_create_skill_omits_missing_env_when_none_required(isolated_skills_dir, isolated_skills_env):
    result = sm.create_skill("doubler", "doubles", SIMPLE_PARAMS, DOUBLE_CODE)
    assert "missing_env" not in result


def test_create_skill_no_missing_env_once_all_set(isolated_skills_dir, isolated_skills_env, monkeypatch):
    from skull.tools import skill_env as scenv

    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "a-real-value")
    scenv.request_skill_env("FAKE_API_KEY")

    result = sm.create_skill(
        "notify", "sends a notification", SIMPLE_PARAMS, SEND_CODE_USING_ENV, required_env=["FAKE_API_KEY"]
    )
    assert "missing_env" not in result


def test_create_skill_rejects_invalid_required_env_name(isolated_skills_dir):
    result = sm.create_skill(
        "notify", "sends a notification", SIMPLE_PARAMS, DOUBLE_CODE, required_env=["not valid!"]
    )
    assert "error" in result
    assert sm.get_skill("notify") is None


def test_skill_can_read_its_own_env_via_get_env(isolated_skills_dir, isolated_skills_env, monkeypatch):
    """End-to-end: the skill's own code reads the secret via get_env() at
    call time, never through kwargs - confirms the actual runtime path
    works, not just that create_skill records the declaration."""
    from skull.tools import skill_env as scenv

    sm.create_skill(
        "notify", "sends a notification", SIMPLE_PARAMS, SEND_CODE_USING_ENV, required_env=["FAKE_API_KEY"]
    )

    result_before = sm.run_skill("notify", {"n": 1})
    assert result_before == {"result": {"had_key": False}}

    monkeypatch.setattr(scenv.getpass, "getpass", lambda prompt: "a-real-value")
    scenv.request_skill_env("FAKE_API_KEY")

    result_after = sm.run_skill("notify", {"n": 1})
    assert result_after == {"result": {"had_key": True}}


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


# ---------------------------------------------------------------------------
# run_skill output truncation - real gap found via code review: unlike
# every other content-returning tool (run_python: 4000 chars,
# read_file/sandbox_read_file: 40000-char ceiling, scrape_page: 20000),
# run_skill had NO output size limit at all. A real skill (unsplash_search)
# already returns an uncapped API response; any skill fetching external
# data has the same exposure to blowing the context window in one call.
# ---------------------------------------------------------------------------

HUGE_RESULT_CODE = """
def run(**kwargs):
    return {"data": "x" * 100000}
"""

NON_JSON_SERIALIZABLE_CODE = """
def run(**kwargs):
    return {"bad": object()}
"""


def test_run_skill_passes_through_small_results_unchanged(isolated_skills_dir):
    sm.create_skill("doubler", "doubles a number", SIMPLE_PARAMS, DOUBLE_CODE)
    outcome = sm.run_skill("doubler", {"n": 5})
    assert outcome == {"result": {"doubled": 10}}
    assert "result_truncated" not in outcome


def test_run_skill_truncates_oversized_result(isolated_skills_dir):
    sm.create_skill("huge", "returns a huge result", SIMPLE_PARAMS, HUGE_RESULT_CODE)
    outcome = sm.run_skill("huge", {"n": 1})

    assert outcome["result_truncated"] is True
    assert outcome["original_length"] > sm.MAX_RESULT_CHARS
    assert len(outcome["truncated_json_string"]) == sm.MAX_RESULT_CHARS
    assert "note" in outcome
    assert "result" not in outcome  # the raw (huge) result must not also be present


def test_run_skill_truncation_marker_is_always_valid_json(isolated_skills_dir):
    """The truncated payload itself (the wrapper dict) must always be valid
    JSON the model can parse, even though truncated_json_string's CONTENT
    may be cut mid-structure - truncating the raw result naively (instead
    of via this wrapper) would risk returning an unparseable blob."""
    sm.create_skill("huge", "returns a huge result", SIMPLE_PARAMS, HUGE_RESULT_CODE)
    outcome = sm.run_skill("huge", {"n": 1})
    reserialized = json.dumps(outcome)  # must not raise
    assert json.loads(reserialized) == outcome


def test_run_skill_rejects_non_json_serializable_result(isolated_skills_dir):
    sm.create_skill("bad_return", "returns something unserializable", SIMPLE_PARAMS, NON_JSON_SERIALIZABLE_CODE)
    outcome = sm.run_skill("bad_return", {"n": 1})
    assert "error" in outcome


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


def test_rollback_skill_preserves_required_env(isolated_skills_dir, isolated_skills_env):
    """Real bug found via code review: _archive_current_version's meta.json
    only ever recorded description/parameters, never required_env, so
    rolling back a credentialed skill silently and permanently wiped its
    credential requirement - the model would lose the "call
    request_skill_env first" signal and just see get_env() return None at
    call time, with no explanation, and no later rollback could recover the
    lost required_env since it was never archived in the first place."""
    sm.create_skill(
        "notify", "sends a notification using an API key", SIMPLE_PARAMS, SEND_CODE_USING_ENV,
        required_env=["FAKE_API_KEY"],
    )
    sm.create_skill("notify", "sends a notification, v2", SIMPLE_PARAMS, SEND_CODE_USING_ENV)

    result = sm.rollback_skill("notify")
    assert result["status"] == "rolled_back"
    assert sm.get_skill("notify")["required_env"] == ["FAKE_API_KEY"]


def test_rollback_skill_preserves_required_env_across_multiple_rollbacks(isolated_skills_dir, isolated_skills_env):
    sm.create_skill(
        "notify", "v1", SIMPLE_PARAMS, SEND_CODE_USING_ENV, required_env=["FAKE_API_KEY"],
    )
    sm.create_skill("notify", "v2", SIMPLE_PARAMS, SEND_CODE_USING_ENV)
    sm.create_skill("notify", "v3", SIMPLE_PARAMS, SEND_CODE_USING_ENV)

    sm.rollback_skill("notify")  # -> v2 (never had required_env)
    assert sm.get_skill("notify")["required_env"] == []

    sm.rollback_skill("notify", version=1)  # -> v1
    assert sm.get_skill("notify")["required_env"] == ["FAKE_API_KEY"]


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
