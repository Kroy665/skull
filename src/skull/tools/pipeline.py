"""Skill pipelines: a saved DAG where nodes are skill calls and edges are
field-level data mappings between them.

A pipeline is a distinct concept from a skill - a skill is one function; a
pipeline is a graph of skill calls - so it's stored separately in
pipelines/<name>/graph.json rather than mixed into skills/.

Graph shape:
    {
      "name": "...",
      "description": "...",
      "nodes": {
        "input": {"type": "input"},
        "some_id": {"type": "skill", "skill": "skill_name", "params": {...literal args...}},
        ...
      },
      "edges": [
        {"from": "input.some_field", "to": "some_id.param_name"},
        {"from": "some_id.output_field", "to": "other_id.param_name"},
        ...
      ]
    }

"input" is a pseudo-node: its "output" is whatever kwargs run_pipeline() is
called with - there is no literal input node in the file, it's implicit.
Every real node's parameters must be satisfied by exactly one source:
either a literal in its own "params", or exactly one incoming edge - never
both, never neither. This is validated at creation time (fail fast), not
discovered mid-run.

Execution order is derived from the edges via topological sort (Kahn's
algorithm) - cycles are rejected at creation time. On any node's failure,
the whole run stops immediately (no partial-branch continuation) and
reports which node failed and why.
"""

import json
import shutil
from pathlib import Path

from skull.config import PIPELINES_DIR
from skull.tools import skills as sm
from skull.tools.skill_composition import SkillError, call_skill

NAME_RE = sm.NAME_RE  # same lowercase_snake_case rule as skill names
INPUT_NODE_ID = "input"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _pipeline_dir(name: str) -> Path:
    return PIPELINES_DIR / name


def _graph_path(name: str) -> Path:
    return _pipeline_dir(name) / "graph.json"


def _index_path() -> Path:
    return PIPELINES_DIR / "index.json"


def _load_index() -> list:
    path = _index_path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save_index(index: list) -> None:
    PIPELINES_DIR.mkdir(exist_ok=True)
    _index_path().write_text(json.dumps(index, indent=2) + "\n")


def list_pipelines() -> list:
    """Return the registry: [{name, description}, ...]"""
    return _load_index()


def get_pipeline(name: str) -> dict | None:
    for entry in _load_index():
        if entry["name"] == name:
            return entry
    return None


def _load_graph(name: str) -> dict:
    return json.loads(_graph_path(name).read_text())


# ---------------------------------------------------------------------------
# Validation (all at creation time - fail fast, never discovered mid-run)
# ---------------------------------------------------------------------------

def _parse_ref(ref: str) -> tuple:
    """Split "node_id.field_name" into (node_id, field_name)."""
    if "." not in ref:
        raise ValueError(f"'{ref}' is not a valid node.field reference")
    node_id, field = ref.split(".", 1)
    return node_id, field


def _topological_order(node_ids: list, edges: list) -> list:
    """Kahn's algorithm. Raises ValueError on a cycle."""
    depends_on = {n: set() for n in node_ids}
    for edge in edges:
        from_node, _ = _parse_ref(edge["from"])
        to_node, _ = _parse_ref(edge["to"])
        if from_node != INPUT_NODE_ID:
            depends_on[to_node].add(from_node)

    order = []
    remaining = set(node_ids)
    while remaining:
        ready = [n for n in remaining if not (depends_on[n] & remaining)]
        if not ready:
            cyclic = ", ".join(sorted(remaining))
            raise ValueError(f"cycle detected among nodes: {cyclic}")
        ready.sort()  # deterministic order when multiple nodes are ready at once
        for n in ready:
            order.append(n)
            remaining.discard(n)
    return order


def validate_graph(nodes: dict, edges: list) -> dict:
    """Validate a graph definition. Returns {"error": ...} on the first
    problem found, or {} if the graph is valid."""
    if not isinstance(nodes, dict) or not nodes:
        return {"error": "nodes must be a non-empty object"}
    if INPUT_NODE_ID in nodes:
        return {"error": f"'{INPUT_NODE_ID}' is a reserved pseudo-node id and must not appear in nodes"}

    real_node_ids = set(nodes.keys())
    all_node_ids = real_node_ids | {INPUT_NODE_ID}

    for node_id, node in nodes.items():
        if node.get("type") != "skill":
            return {"error": f"node '{node_id}': type must be 'skill'"}
        skill_name = node.get("skill")
        if not skill_name:
            return {"error": f"node '{node_id}': missing 'skill'"}
        if sm.get_skill(skill_name) is None:
            return {"error": f"node '{node_id}': no such skill '{skill_name}'"}

    if not isinstance(edges, list):
        return {"error": "edges must be a list"}

    incoming_params = {node_id: set() for node_id in real_node_ids}
    for i, edge in enumerate(edges):
        if "from" not in edge or "to" not in edge:
            return {"error": f"edge {i}: must have 'from' and 'to'"}
        try:
            from_node, from_field = _parse_ref(edge["from"])
            to_node, to_field = _parse_ref(edge["to"])
        except ValueError as e:
            return {"error": f"edge {i}: {e}"}

        if from_node not in all_node_ids:
            return {"error": f"edge {i}: unknown source node '{from_node}'"}
        if to_node not in real_node_ids:
            return {"error": f"edge {i}: unknown target node '{to_node}' (or it's the reserved 'input' node)"}
        if to_node == from_node:
            return {"error": f"edge {i}: a node cannot feed itself ('{to_node}')"}

        if to_field in incoming_params[to_node]:
            return {
                "error": (
                    f"node '{to_node}' param '{to_field}' is targeted by more than one edge "
                    "- each param must have exactly one source"
                )
            }
        literal_params = nodes[to_node].get("params") or {}
        if to_field in literal_params:
            return {
                "error": (
                    f"node '{to_node}' param '{to_field}' is set both by a literal in "
                    "'params' and by an incoming edge - each param must have exactly one source"
                )
            }
        incoming_params[to_node].add(to_field)

    # Every required param of every node's target skill must be bound by
    # either a literal or an edge - otherwise the run will fail on a
    # missing-argument TypeError instead of failing fast at creation time.
    for node_id, node in nodes.items():
        skill_entry = sm.get_skill(node["skill"])
        required = (skill_entry.get("parameters") or {}).get("required") or []
        literal_params = node.get("params") or {}
        bound = set(literal_params.keys()) | incoming_params[node_id]
        missing = [p for p in required if p not in bound]
        if missing:
            return {
                "error": (
                    f"node '{node_id}' (skill '{node['skill']}') is missing required "
                    f"param(s) {missing} - bind via a literal in 'params' or an incoming edge"
                )
            }

    try:
        _topological_order(list(real_node_ids), edges)
    except ValueError as e:
        return {"error": str(e)}

    return {}


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_pipeline(name: str, description: str, nodes: dict, edges: list) -> dict:
    if not NAME_RE.match(name):
        return {"error": "name must be lowercase snake_case, 2-64 chars, start with a letter"}

    problem = validate_graph(nodes, edges)
    if problem:
        return problem

    pipeline_dir = _pipeline_dir(name)
    is_new = not pipeline_dir.exists()
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    graph = {"name": name, "description": description, "nodes": nodes, "edges": edges}
    _graph_path(name).write_text(json.dumps(graph, indent=2) + "\n")

    index = [e for e in _load_index() if e["name"] != name]
    index.append({"name": name, "description": description})
    _save_index(index)

    return {"status": "created", "name": name}


def delete_pipeline(name: str) -> dict:
    if get_pipeline(name) is None:
        return {"error": f"no such pipeline '{name}'"}

    index = [e for e in _load_index() if e["name"] != name]
    _save_index(index)
    shutil.rmtree(_pipeline_dir(name), ignore_errors=True)
    return {"status": "deleted", "name": name}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_pipeline(name: str, **inputs) -> dict:
    entry = get_pipeline(name)
    if entry is None:
        return {"error": f"no such pipeline '{name}'"}

    graph = _load_graph(name)
    nodes = graph["nodes"]
    edges = graph["edges"]

    try:
        order = _topological_order(list(nodes.keys()), edges)
    except ValueError as e:
        return {"error": f"invalid graph: {e}"}

    # node_id -> its output (the raw value returned by call_skill)
    outputs = {INPUT_NODE_ID: inputs}
    trace = []

    # Precompute, per target node, the list of (field, from_node, from_field)
    # bindings so each node's call args can be assembled in one pass.
    bindings_by_node = {node_id: [] for node_id in nodes}
    for edge in edges:
        from_node, from_field = _parse_ref(edge["from"])
        to_node, to_field = _parse_ref(edge["to"])
        bindings_by_node[to_node].append((to_field, from_node, from_field))

    for node_id in order:
        node = nodes[node_id]
        skill_name = node["skill"]
        call_args = dict(node.get("params") or {})

        for to_field, from_node, from_field in bindings_by_node[node_id]:
            source_output = outputs[from_node]
            if not isinstance(source_output, dict):
                return {
                    "error": (
                        f"node '{node_id}': cannot read field '{from_field}' from "
                        f"'{from_node}' - its output is not an object (got "
                        f"{type(source_output).__name__})"
                    ),
                    "trace": trace,
                }
            if from_field not in source_output:
                return {
                    "error": (
                        f"node '{node_id}': field '{from_field}' not found in "
                        f"'{from_node}' output (available: {list(source_output.keys())})"
                    ),
                    "trace": trace,
                }
            call_args[to_field] = source_output[from_field]

        try:
            result = call_skill(skill_name, **call_args)
        except SkillError as e:
            return {
                "error": f"node '{node_id}' (skill '{skill_name}') failed: {e}",
                "trace": trace,
            }
        except Exception as e:
            return {
                "error": f"node '{node_id}' (skill '{skill_name}') raised {type(e).__name__}: {e}",
                "trace": trace,
            }

        outputs[node_id] = result
        trace.append({"node": node_id, "skill": skill_name, "args": call_args, "output": result})

    # Terminal nodes: those with no outgoing edges, i.e. nothing downstream
    # consumes their output. Their outputs are the pipeline's overall result.
    has_downstream = {edge["from"].split(".", 1)[0] for edge in edges}
    terminal_ids = [n for n in nodes if n not in has_downstream]

    return {
        "status": "completed",
        "outputs": {n: outputs[n] for n in terminal_ids},
        "trace": trace,
    }
