"""Tests for tools/pipeline.py - skill DAGs (validation at creation time,
topological execution, fan-out/fan-in, and stop-on-first-failure at run
time)."""

from skull.tools import pipeline as pl
from skull.tools import skills as sm

NUM_PARAMS = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
TWO_NUM_PARAMS = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}

DOUBLE_CODE = """
def run(**kwargs):
    return {"doubled": kwargs["n"] * 2}
"""

ADD_CODE = """
def run(**kwargs):
    return {"sum": kwargs["a"] + kwargs["b"]}
"""

FAILS_CODE = """
def run(**kwargs):
    raise RuntimeError("skill blew up")
"""

RETURNS_NON_DICT_CODE = """
def run(**kwargs):
    return 42
"""


def _make_doubler(isolated_skills_dir):
    sm.create_skill("doubler", "doubles", NUM_PARAMS, DOUBLE_CODE)


def _make_adder(isolated_skills_dir):
    sm.create_skill("adder", "adds", TWO_NUM_PARAMS, ADD_CODE)


def _make_failer(isolated_skills_dir):
    sm.create_skill("failer", "fails", NUM_PARAMS, FAILS_CODE)


def _make_non_dict_returner(isolated_skills_dir):
    sm.create_skill("non_dict_returner", "returns a scalar", NUM_PARAMS, RETURNS_NON_DICT_CODE)


# ---------------------------------------------------------------------------
# validate_graph - the 6+ fail-fast error paths
# ---------------------------------------------------------------------------

def test_validate_graph_rejects_empty_nodes(isolated_pipelines_dir):
    assert "error" in pl.validate_graph({}, [])


def test_validate_graph_rejects_reserved_input_node_id(isolated_pipelines_dir):
    nodes = {"input": {"type": "skill", "skill": "doubler"}}
    result = pl.validate_graph(nodes, [])
    assert "error" in result
    assert "reserved" in result["error"]


def test_validate_graph_rejects_unknown_skill(isolated_pipelines_dir):
    nodes = {"a": {"type": "skill", "skill": "no_such_skill", "params": {"n": 1}}}
    result = pl.validate_graph(nodes, [])
    assert "error" in result
    assert "no such skill" in result["error"]


def test_validate_graph_rejects_edge_to_unknown_node(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler", "params": {"n": 1}}}
    edges = [{"from": "input.n", "to": "b.n"}]
    result = pl.validate_graph(nodes, edges)
    assert "error" in result
    assert "unknown target node" in result["error"]


def test_validate_graph_rejects_self_loop(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [{"from": "a.doubled", "to": "a.n"}]
    result = pl.validate_graph(nodes, edges)
    assert "error" in result
    assert "cannot feed itself" in result["error"]


def test_validate_graph_rejects_cycle(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    sm.create_skill("doubler2", "doubles", NUM_PARAMS, DOUBLE_CODE)
    nodes = {
        "a": {"type": "skill", "skill": "doubler"},
        "b": {"type": "skill", "skill": "doubler2"},
    }
    edges = [
        {"from": "a.doubled", "to": "b.n"},
        {"from": "b.doubled", "to": "a.n"},
    ]
    result = pl.validate_graph(nodes, edges)
    assert "error" in result
    assert "cycle" in result["error"]


def test_validate_graph_rejects_param_bound_by_two_edges(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [
        {"from": "input.x", "to": "a.n"},
        {"from": "input.y", "to": "a.n"},
    ]
    result = pl.validate_graph(nodes, edges)
    assert "error" in result
    assert "more than one edge" in result["error"]


def test_validate_graph_rejects_param_bound_by_literal_and_edge(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler", "params": {"n": 5}}}
    edges = [{"from": "input.n", "to": "a.n"}]
    result = pl.validate_graph(nodes, edges)
    assert "error" in result
    assert "literal" in result["error"]


def test_validate_graph_rejects_missing_required_param(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}  # no params, no edges for "n"
    result = pl.validate_graph(nodes, [])
    assert "error" in result
    assert "missing required" in result["error"]


def test_validate_graph_accepts_valid_single_node_graph(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [{"from": "input.n", "to": "a.n"}]
    assert pl.validate_graph(nodes, edges) == {}


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def test_create_pipeline_persists_and_registers(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [{"from": "input.n", "to": "a.n"}]
    result = pl.create_pipeline("double_it", "doubles the input", nodes, edges)
    assert result == {"status": "created", "name": "double_it"}
    assert pl.get_pipeline("double_it")["description"] == "doubles the input"
    assert (isolated_pipelines_dir / "double_it" / "graph.json").exists()


def test_create_pipeline_rejects_invalid_name(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [{"from": "input.n", "to": "a.n"}]
    result = pl.create_pipeline("Not Valid", "desc", nodes, edges)
    assert "error" in result


def test_create_pipeline_propagates_validation_error(isolated_pipelines_dir, isolated_skills_dir):
    nodes = {"a": {"type": "skill", "skill": "no_such_skill"}}
    result = pl.create_pipeline("bad_pipeline", "desc", nodes, [])
    assert "error" in result
    assert pl.get_pipeline("bad_pipeline") is None


def test_delete_pipeline_removes_files_and_entry(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [{"from": "input.n", "to": "a.n"}]
    pl.create_pipeline("double_it", "desc", nodes, edges)
    result = pl.delete_pipeline("double_it")
    assert result == {"status": "deleted", "name": "double_it"}
    assert pl.get_pipeline("double_it") is None
    assert not (isolated_pipelines_dir / "double_it").exists()


def test_delete_pipeline_missing_returns_error(isolated_pipelines_dir):
    result = pl.delete_pipeline("nonexistent")
    assert "error" in result


def test_list_pipelines_empty_by_default(isolated_pipelines_dir):
    assert pl.list_pipelines() == []


# ---------------------------------------------------------------------------
# run_pipeline - execution semantics
# ---------------------------------------------------------------------------

def test_run_pipeline_missing_pipeline_returns_error(isolated_pipelines_dir):
    result = pl.run_pipeline("nonexistent")
    assert "error" in result


def test_run_pipeline_single_node_linear(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "doubler"}}
    edges = [{"from": "input.n", "to": "a.n"}]
    pl.create_pipeline("double_it", "desc", nodes, edges)

    result = pl.run_pipeline("double_it", n=21)
    assert result["status"] == "completed"
    assert result["outputs"] == {"a": {"doubled": 42}}
    assert len(result["trace"]) == 1


def test_run_pipeline_fan_out(isolated_pipelines_dir, isolated_skills_dir):
    """One input feeds two independent downstream nodes."""
    _make_doubler(isolated_skills_dir)
    sm.create_skill("doubler2", "doubles", NUM_PARAMS, DOUBLE_CODE)
    nodes = {
        "a": {"type": "skill", "skill": "doubler"},
        "b": {"type": "skill", "skill": "doubler2"},
    }
    edges = [
        {"from": "input.n", "to": "a.n"},
        {"from": "input.n", "to": "b.n"},
    ]
    pl.create_pipeline("fan_out", "desc", nodes, edges)

    result = pl.run_pipeline("fan_out", n=5)
    assert result["status"] == "completed"
    assert result["outputs"] == {"a": {"doubled": 10}, "b": {"doubled": 10}}


def test_run_pipeline_fan_in(isolated_pipelines_dir, isolated_skills_dir):
    """One node needs outputs from two upstream nodes."""
    _make_doubler(isolated_skills_dir)
    sm.create_skill("doubler2", "doubles", NUM_PARAMS, DOUBLE_CODE)
    _make_adder(isolated_skills_dir)
    nodes = {
        "a": {"type": "skill", "skill": "doubler"},
        "b": {"type": "skill", "skill": "doubler2"},
        "c": {"type": "skill", "skill": "adder"},
    }
    edges = [
        {"from": "input.n", "to": "a.n"},
        {"from": "input.n", "to": "b.n"},
        {"from": "a.doubled", "to": "c.a"},
        {"from": "b.doubled", "to": "c.b"},
    ]
    pl.create_pipeline("fan_in", "desc", nodes, edges)

    result = pl.run_pipeline("fan_in", n=5)
    assert result["status"] == "completed"
    # a and b are non-terminal (feed c), only c's output is reported
    assert result["outputs"] == {"c": {"sum": 20}}
    assert len(result["trace"]) == 3


def test_run_pipeline_stops_on_node_failure(isolated_pipelines_dir, isolated_skills_dir):
    """A downstream node must never run after an upstream one fails, and the
    failure must name which node and why."""
    _make_doubler(isolated_skills_dir)
    _make_failer(isolated_skills_dir)
    nodes = {
        "a": {"type": "skill", "skill": "failer"},
        "b": {"type": "skill", "skill": "doubler"},
    }
    edges = [
        {"from": "input.n", "to": "a.n"},
        {"from": "a.doubled", "to": "b.n"},
    ]
    pl.create_pipeline("will_fail", "desc", nodes, edges)

    result = pl.run_pipeline("will_fail", n=5)
    assert "error" in result
    assert "node 'a'" in result["error"]
    assert "skill blew up" in result["error"]
    # b must never have run - it's not in the trace
    assert all(t["node"] != "b" for t in result["trace"])


def test_run_pipeline_reports_missing_field_from_non_dict_output(isolated_pipelines_dir, isolated_skills_dir):
    _make_non_dict_returner(isolated_skills_dir)
    _make_doubler(isolated_skills_dir)
    nodes = {
        "a": {"type": "skill", "skill": "non_dict_returner"},
        "b": {"type": "skill", "skill": "doubler"},
    }
    edges = [
        {"from": "input.n", "to": "a.n"},
        {"from": "a.doubled", "to": "b.n"},
    ]
    pl.create_pipeline("bad_output", "desc", nodes, edges)

    result = pl.run_pipeline("bad_output", n=5)
    assert "error" in result
    assert "not an object" in result["error"]


def test_run_pipeline_reports_missing_field_name(isolated_pipelines_dir, isolated_skills_dir):
    _make_doubler(isolated_skills_dir)
    sm.create_skill("doubler2", "doubles", NUM_PARAMS, DOUBLE_CODE)
    nodes = {
        "a": {"type": "skill", "skill": "doubler"},
        "b": {"type": "skill", "skill": "doubler2"},
    }
    edges = [
        {"from": "input.n", "to": "a.n"},
        {"from": "a.nonexistent_field", "to": "b.n"},
    ]
    pl.create_pipeline("bad_field", "desc", nodes, edges)

    result = pl.run_pipeline("bad_field", n=5)
    assert "error" in result
    assert "not found" in result["error"]


def test_run_pipeline_uses_literal_params_alongside_edges(isolated_pipelines_dir, isolated_skills_dir):
    _make_adder(isolated_skills_dir)
    nodes = {"a": {"type": "skill", "skill": "adder", "params": {"b": 100}}}
    edges = [{"from": "input.n", "to": "a.a"}]
    pl.create_pipeline("literal_and_edge", "desc", nodes, edges)

    result = pl.run_pipeline("literal_and_edge", n=1)
    assert result["status"] == "completed"
    assert result["outputs"] == {"a": {"sum": 101}}
