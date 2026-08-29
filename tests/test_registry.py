"""Tests for tools/registry.py - plan-mode tool filtering (the safety
boundary between "research only" and "can act") and run_tool_call's
malformed-JSON handling and dispatch/error wrapping."""

import json

from skull.tools import registry


# ---------------------------------------------------------------------------
# build_tools_and_impls - plan-mode filtering
# ---------------------------------------------------------------------------

def test_normal_mode_includes_mutating_tools(isolated_skills_dir):
    tools, impls = registry.build_tools_and_impls(plan_mode=False)
    names = {t["function"]["name"] for t in tools}
    assert "write_file" in names
    assert "run_command" in names
    assert "create_skill" in names
    assert "write_file" in impls
    assert "create_skill" in impls


def test_plan_mode_excludes_all_mutating_tools(isolated_skills_dir):
    tools, impls = registry.build_tools_and_impls(plan_mode=True)
    names = {t["function"]["name"] for t in tools}
    for mutating_name in registry.MUTATING_TOOL_NAMES:
        assert mutating_name not in names, f"{mutating_name} leaked into plan mode tools"
        assert mutating_name not in impls, f"{mutating_name} leaked into plan mode impls"


def test_plan_mode_keeps_read_only_builtins(isolated_skills_dir):
    tools, impls = registry.build_tools_and_impls(plan_mode=True)
    names = {t["function"]["name"] for t in tools}
    for read_only_name in ("web_search", "scrape_page", "read_file", "list_directory", "list_skills", "recall_memory"):
        assert read_only_name in names
        assert read_only_name in impls


def test_plan_mode_excludes_self_created_skills(isolated_skills_dir):
    """Skill side effects aren't tracked, so every self-created skill is
    treated as potentially mutating and excluded wholesale in plan mode -
    even one that's actually harmless."""
    from skull.tools import skills as sm

    sm.create_skill(
        "harmless_skill",
        "does nothing dangerous",
        {"type": "object", "properties": {}},
        "def run(**kwargs):\n    return {'ok': True}\n",
    )

    plan_tools, plan_impls = registry.build_tools_and_impls(plan_mode=True)
    plan_names = {t["function"]["name"] for t in plan_tools}
    assert "harmless_skill" not in plan_names
    assert "harmless_skill" not in plan_impls

    normal_tools, normal_impls = registry.build_tools_and_impls(plan_mode=False)
    normal_names = {t["function"]["name"] for t in normal_tools}
    assert "harmless_skill" in normal_names
    assert "harmless_skill" in normal_impls


def test_build_tools_includes_freshly_created_skill_without_restart(isolated_skills_dir):
    """A skill created mid-conversation must be immediately callable - the
    tool list is reassembled fresh each turn, not cached at startup."""
    from skull.tools import skills as sm

    before_tools, _ = registry.build_tools_and_impls(plan_mode=False)
    before_names = {t["function"]["name"] for t in before_tools}
    assert "brand_new_skill" not in before_names

    sm.create_skill(
        "brand_new_skill",
        "desc",
        {"type": "object", "properties": {}},
        "def run(**kwargs):\n    return {'ok': True}\n",
    )

    after_tools, after_impls = registry.build_tools_and_impls(plan_mode=False)
    after_names = {t["function"]["name"] for t in after_tools}
    assert "brand_new_skill" in after_names
    assert after_impls["brand_new_skill"]({}) == {"result": {"ok": True}}


# ---------------------------------------------------------------------------
# _is_valid_json
# ---------------------------------------------------------------------------

def test_is_valid_json_accepts_empty_and_none():
    assert registry._is_valid_json("") is True
    assert registry._is_valid_json(None) is True


def test_is_valid_json_accepts_valid_json():
    assert registry._is_valid_json('{"a": 1}') is True


def test_is_valid_json_rejects_malformed_json():
    assert registry._is_valid_json('{"a": ') is False


# ---------------------------------------------------------------------------
# run_tool_call - dispatch, malformed-args recovery, error wrapping
# ---------------------------------------------------------------------------

def test_run_tool_call_dispatches_to_impl_and_returns_json():
    tool_call = {"function": {"name": "echo", "arguments": '{"x": 1}'}}
    impls = {"echo": lambda args: {"got": args}}
    result_json = registry.run_tool_call(tool_call, impls, verbose=False)
    assert json.loads(result_json) == {"got": {"x": 1}}


def test_run_tool_call_unknown_tool_returns_error():
    tool_call = {"function": {"name": "nonexistent", "arguments": "{}"}}
    result_json = registry.run_tool_call(tool_call, {}, verbose=False)
    result = json.loads(result_json)
    assert "error" in result
    assert "unknown tool" in result["error"]


def test_run_tool_call_malformed_arguments_falls_back_to_empty_dict():
    """A model that emits invalid JSON for tool-call arguments must not crash
    the whole turn - the call proceeds with args={} instead."""
    tool_call = {"function": {"name": "echo", "arguments": "{not valid json"}}
    impls = {"echo": lambda args: {"got": args}}
    result_json = registry.run_tool_call(tool_call, impls, verbose=False)
    assert json.loads(result_json) == {"got": {}}


def test_run_tool_call_missing_arguments_key_defaults_to_empty_dict():
    tool_call = {"function": {"name": "echo"}}
    impls = {"echo": lambda args: {"got": args}}
    result_json = registry.run_tool_call(tool_call, impls, verbose=False)
    assert json.loads(result_json) == {"got": {}}


def test_run_tool_call_impl_exception_is_captured_not_raised():
    tool_call = {"function": {"name": "boom", "arguments": "{}"}}

    def raiser(args):
        raise RuntimeError("kaboom")

    impls = {"boom": raiser}
    result_json = registry.run_tool_call(tool_call, impls, verbose=False)
    result = json.loads(result_json)
    assert result == {"error": "kaboom"}
