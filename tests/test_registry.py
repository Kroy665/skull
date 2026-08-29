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


def test_download_from_sandbox_is_interactive_and_mutating():
    """download_from_sandbox writes to the real local filesystem, so it must
    require interactive approval (like write_file) and be withheld in plan
    mode (like every other mutating tool) - this is the tool added so binary
    sandbox output no longer needs to be smuggled through sandbox_read_file
    as base64 text."""
    assert "download_from_sandbox" in registry.INTERACTIVE_TOOL_NAMES
    assert "download_from_sandbox" in registry.MUTATING_TOOL_NAMES


def test_plan_mode_excludes_all_mutating_tools(isolated_skills_dir):
    tools, impls = registry.build_tools_and_impls(plan_mode=True)
    names = {t["function"]["name"] for t in tools}
    for mutating_name in registry.MUTATING_TOOL_NAMES:
        assert mutating_name not in names, f"{mutating_name} leaked into plan mode tools"
        assert mutating_name not in impls, f"{mutating_name} leaked into plan mode impls"


def test_skill_versioning_tools_are_registered(isolated_skills_dir):
    tools, impls = registry.build_tools_and_impls(plan_mode=False)
    names = {t["function"]["name"] for t in tools}
    assert "list_skill_versions" in names
    assert "rollback_skill" in names
    assert "list_skill_versions" in impls
    assert "rollback_skill" in impls


def test_rollback_skill_is_mutating_list_skill_versions_is_not():
    assert "rollback_skill" in registry.MUTATING_TOOL_NAMES
    assert "list_skill_versions" not in registry.MUTATING_TOOL_NAMES


def test_rollback_skill_impl_round_trips_through_registry(isolated_skills_dir):
    from skull.tools import skills as sm

    params = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    sm.create_skill("doubler", "v1", params, "def run(**kwargs):\n    return kwargs['n'] * 2\n")
    sm.create_skill("doubler", "v2", params, "def run(**kwargs):\n    return kwargs['n'] * 3\n")

    _, impls = registry.build_tools_and_impls(plan_mode=False)
    result = impls["rollback_skill"]({"name": "doubler"})
    assert result["status"] == "rolled_back"
    assert sm.get_skill("doubler")["description"] == "v1"


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


# ---------------------------------------------------------------------------
# build_tools_and_impls - skill relevance filtering (SKILL_FILTER_THRESHOLD)
# ---------------------------------------------------------------------------

SKILL_CODE = "def run(**kwargs):\n    return {'ok': True}\n"


def _make_n_skills(n: int):
    from skull.tools import skills as sm

    for i in range(n):
        sm.create_skill(f"skill_{i}", f"does thing number {i}", {"type": "object", "properties": {}}, SKILL_CODE)


def test_below_threshold_sends_every_skill_without_query(isolated_skills_dir, isolated_memory_dir):
    _make_n_skills(registry.SKILL_FILTER_THRESHOLD)  # exactly at threshold, not over it
    tools, impls = registry.build_tools_and_impls(plan_mode=False, query="anything")
    names = {t["function"]["name"] for t in tools}
    for i in range(registry.SKILL_FILTER_THRESHOLD):
        assert f"skill_{i}" in names


def test_no_query_sends_every_skill_regardless_of_count(isolated_skills_dir, isolated_memory_dir):
    """Without a query (query=None/empty), filtering never kicks in - callers
    that don't have a natural query text (if any existed) get the full list
    rather than an arbitrarily filtered one."""
    _make_n_skills(registry.SKILL_FILTER_THRESHOLD + 5)
    tools, impls = registry.build_tools_and_impls(plan_mode=False, query=None)
    names = {t["function"]["name"] for t in tools}
    for i in range(registry.SKILL_FILTER_THRESHOLD + 5):
        assert f"skill_{i}" in names


def test_above_threshold_filters_to_top_k_relevant_skills(isolated_skills_dir, isolated_memory_dir, monkeypatch):
    """The isolated_memory_dir fixture's embedder is random noise (not
    semantically meaningful) - true unrelated-text pairs from it cluster
    near-zero cosine similarity, which can fall below SKILL_FILTER_MIN_SCORE
    entirely and make top-K selection untestable through it. Use a
    controlled fake embedder here instead, one that guarantees a clear
    top-K winner well above the score floor, so this test isolates top-K
    selection specifically rather than also depending on the noise
    fixture clearing the min-score threshold by chance."""
    from skull.storage import store as mem

    def fake_embed(texts):
        import numpy as np
        # Text containing "number 3" gets a distinctive vector; everything
        # else gets a near-orthogonal one.
        vectors = []
        for t in texts:
            base = np.zeros(mem.EMBED_DIM, dtype=np.float32)
            if "number 3" in t:
                base[0] = 1.0
            else:
                base[1] = 1.0
            vectors.append(base)
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(mem, "embed", fake_embed)

    _make_n_skills(registry.SKILL_FILTER_THRESHOLD + 5)
    tools, impls = registry.build_tools_and_impls(plan_mode=False, query="does thing number 3")
    skill_names = {t["function"]["name"] for t in tools if t["function"]["name"].startswith("skill_")}
    assert "skill_3" in skill_names  # the one clearly-relevant match must survive
    assert len(skill_names) <= registry.SKILL_FILTER_TOP_K
    assert len(skill_names) < registry.SKILL_FILTER_THRESHOLD + 5


def test_always_include_skills_survive_filtering(isolated_skills_dir, isolated_memory_dir):
    """A skill already used earlier this turn must not disappear from the
    tool list just because the relevance ranking didn't favor it."""
    _make_n_skills(registry.SKILL_FILTER_THRESHOLD + 5)
    tools, impls = registry.build_tools_and_impls(
        plan_mode=False, query="does thing number 3", always_include_skills={"skill_0", "skill_1"}
    )
    names = {t["function"]["name"] for t in tools}
    assert "skill_0" in names
    assert "skill_1" in names


def test_filtered_skills_remain_individually_callable(isolated_skills_dir, isolated_memory_dir):
    """Filtering only affects which schemas are *sent* to the model - every
    skill's impl must still work if called (e.g. because it was included via
    always_include, or the model recalls its name from an earlier listing)."""
    _make_n_skills(registry.SKILL_FILTER_THRESHOLD + 5)
    tools, impls = registry.build_tools_and_impls(
        plan_mode=False, query="does thing number 3", always_include_skills={"skill_0"}
    )
    assert impls["skill_0"]({}) == {"result": {"ok": True}}
