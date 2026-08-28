"""Persistent, self-extending skill store.

Each skill lives in its own directory:

    skills/<name>/run.py     - defines a top-level `run(**kwargs)` function
    skills/<name>/SKILL.md   - human-readable description, parameters, usage

skills/index.json is a lightweight registry (name, description, parameters)
kept in sync with each skill's SKILL.md, used for fast tool-list assembly
without re-reading every file on disk.

skills/ lives at the project root (user data, not part of the installed
package) - see skull.config.SKILLS_DIR.
"""

import importlib.util
import json
import re
import shutil
import sys
import traceback
from pathlib import Path

from skull.config import SKILLS_DIR

INDEX_PATH = SKILLS_DIR / "index.json"

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _load_index() -> list:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text())


def _save_index(index: list) -> None:
    SKILLS_DIR.mkdir(exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n")


def _skill_dir(name: str) -> Path:
    return SKILLS_DIR / name


def _render_skill_md(name: str, description: str, parameters: dict) -> str:
    params_block = json.dumps(parameters, indent=2)
    return (
        f"# {name}\n\n"
        f"{description}\n\n"
        f"## Parameters\n\n"
        f"JSON-schema for `run(**kwargs)`:\n\n"
        f"```json\n{params_block}\n```\n"
    )


def list_skills() -> list:
    """Return the registry: [{name, description, parameters}, ...]"""
    return _load_index()


def get_skill(name: str) -> dict | None:
    for entry in _load_index():
        if entry["name"] == name:
            return entry
    return None


def create_skill(name: str, description: str, parameters: dict, code: str) -> dict:
    """Write a new skill to disk and register it.

    `code` must define a top-level function `run(**kwargs)` that returns a
    JSON-serializable result (or raises on failure).
    `parameters` is a JSON-schema `properties`-style object, e.g.
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

    Writes skills/<name>/run.py and skills/<name>/SKILL.md.
    """
    if not NAME_RE.match(name):
        return {"error": "name must be lowercase snake_case, 2-64 chars, start with a letter"}
    if "def run(" not in code:
        return {"error": "code must define a top-level function `run(**kwargs)`"}

    skill_dir = _skill_dir(name)
    is_new = not skill_dir.exists()
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "run.py").write_text(code)
    (skill_dir / "SKILL.md").write_text(_render_skill_md(name, description, parameters))

    # Sanity-check: does it even import?
    try:
        _import_skill(name)
    except Exception as e:
        if is_new:
            shutil.rmtree(skill_dir, ignore_errors=True)
        return {"error": f"skill failed to import after write: {e}"}

    index = _load_index()
    index = [e for e in index if e["name"] != name]  # replace if exists
    index.append({"name": name, "description": description, "parameters": parameters})
    _save_index(index)

    return {"status": "created", "name": name}


def delete_skill(name: str) -> dict:
    index = [e for e in _load_index() if e["name"] != name]
    _save_index(index)
    shutil.rmtree(_skill_dir(name), ignore_errors=True)
    sys.modules.pop(f"skills.{name}.run", None)
    return {"status": "deleted", "name": name}


def _import_skill(name: str):
    path = _skill_dir(name) / "run.py"
    spec = importlib.util.spec_from_file_location(f"skills.{name}.run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_skill(name: str, args: dict) -> dict:
    entry = get_skill(name)
    if entry is None:
        return {"error": f"no such skill '{name}'"}
    try:
        module = _import_skill(name)
        result = module.run(**args)
        return {"result": result}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
