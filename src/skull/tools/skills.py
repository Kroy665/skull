"""Persistent, self-extending skill store.

Each skill lives in its own directory:

    skills/<name>/run.py           - defines a top-level `run(**kwargs)` function
    skills/<name>/SKILL.md         - human-readable description, parameters, usage
    skills/<name>/versions/<n>/    - archived previous versions (run.py, SKILL.md, meta.json)

skills/index.json is a lightweight registry (name, description, parameters)
kept in sync with each skill's SKILL.md, used for fast tool-list assembly
without re-reading every file on disk.

Overwriting an existing skill (create_skill called again with the same
name) has no undo by default - a bad rewrite would silently destroy the
only working copy. Before writing new code over an existing skill, the
current version is archived to versions/<n>/, bounded to the most recent
MAX_VERSIONS_KEPT. rollback_skill() restores an archived version as the
live one (itself archiving whatever was live first, so a rollback is
undoable too).

skills/ lives at the project root (user data, not part of the installed
package) - see skull.config.SKILLS_DIR.
"""

import importlib.machinery
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

MAX_VERSIONS_KEPT = 5


def _load_index() -> list:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text())


def _save_index(index: list) -> None:
    SKILLS_DIR.mkdir(exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n")


def _skill_dir(name: str) -> Path:
    return SKILLS_DIR / name


def _versions_dir(name: str) -> Path:
    return _skill_dir(name) / "versions"


def _next_version_number(name: str) -> int:
    versions_dir = _versions_dir(name)
    if not versions_dir.exists():
        return 1
    existing = [int(p.name) for p in versions_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return (max(existing) if existing else 0) + 1


def _archive_current_version(name: str, entry: dict) -> None:
    """Copy the currently-live run.py/SKILL.md into versions/<n>/ before
    they get overwritten, then prune anything past MAX_VERSIONS_KEPT (the
    oldest first). No-op if the skill has no live files yet (a brand new
    skill has nothing to archive)."""
    skill_dir = _skill_dir(name)
    run_py = skill_dir / "run.py"
    skill_md = skill_dir / "SKILL.md"
    if not run_py.exists():
        return

    version_num = _next_version_number(name)
    version_dir = _versions_dir(name) / str(version_num)
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_py, version_dir / "run.py")
    if skill_md.exists():
        shutil.copy2(skill_md, version_dir / "SKILL.md")
    (version_dir / "meta.json").write_text(
        json.dumps({"description": entry["description"], "parameters": entry.get("parameters") or {}}, indent=2)
        + "\n"
    )

    versions_dir = _versions_dir(name)
    all_versions = sorted((int(p.name) for p in versions_dir.iterdir() if p.is_dir() and p.name.isdigit()))
    for old_version in all_versions[:-MAX_VERSIONS_KEPT]:
        shutil.rmtree(versions_dir / str(old_version), ignore_errors=True)


def list_skill_versions(name: str) -> dict:
    """List archived previous versions of a skill, most recent first."""
    if get_skill(name) is None:
        return {"error": f"no such skill '{name}'"}

    versions_dir = _versions_dir(name)
    if not versions_dir.exists():
        return {"name": name, "versions": []}

    versions = []
    for version_dir in sorted(versions_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1, reverse=True):
        if not version_dir.is_dir() or not version_dir.name.isdigit():
            continue
        meta_path = version_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        versions.append({"version": int(version_dir.name), "description": meta.get("description", "")})

    return {"name": name, "versions": versions}


def rollback_skill(name: str, version: int = None) -> dict:
    """Restore an archived version as the live one. Defaults to the most
    recently archived version (i.e. undo the last create_skill overwrite).
    The version being replaced is itself archived first, so a rollback can
    be undone with another rollback_skill call."""
    entry = get_skill(name)
    if entry is None:
        return {"error": f"no such skill '{name}'"}

    versions_dir = _versions_dir(name)
    available = sorted(
        (int(p.name) for p in versions_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        reverse=True,
    ) if versions_dir.exists() else []
    if not available:
        return {"error": f"no archived versions for '{name}' to roll back to"}

    target_version = version if version is not None else available[0]
    version_dir = versions_dir / str(target_version)
    if not version_dir.exists():
        return {"error": f"no such version {target_version} for '{name}' (available: {available})"}

    meta_path = version_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    code = (version_dir / "run.py").read_text()

    result = create_skill(
        name,
        meta.get("description", entry["description"]),
        meta.get("parameters", entry.get("parameters") or {}),
        code,
    )
    if "error" in result:
        return result
    return {"status": "rolled_back", "name": name, "restored_version": target_version}


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

    Writes skills/<name>/run.py and skills/<name>/SKILL.md. If a skill
    with this name already exists, its current version is archived first
    (see rollback_skill/list_skill_versions) - overwriting used to have no
    undo, so a bad rewrite would silently destroy the only working copy.
    """
    if not NAME_RE.match(name):
        return {"error": "name must be lowercase snake_case, 2-64 chars, start with a letter"}
    if "def run(" not in code:
        return {"error": "code must define a top-level function `run(**kwargs)`"}

    skill_dir = _skill_dir(name)
    is_new = not skill_dir.exists()
    existing_entry = get_skill(name)
    if not is_new and existing_entry is not None:
        _archive_current_version(name, existing_entry)

    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "run.py").write_text(code)
    (skill_dir / "SKILL.md").write_text(_render_skill_md(name, description, parameters))

    # Sanity-check: does it even import?
    try:
        _import_skill(name)
    except Exception as e:
        if is_new:
            shutil.rmtree(skill_dir, ignore_errors=True)
        elif existing_entry is not None:
            # This was an overwrite of a previously-working skill - restore
            # the version just archived above instead of leaving broken
            # code live under the skill's name.
            archived_versions = _versions_dir(name)
            latest = max(
                (int(p.name) for p in archived_versions.iterdir() if p.is_dir() and p.name.isdigit()),
                default=None,
            )
            if latest is not None:
                version_dir = archived_versions / str(latest)
                shutil.copy2(version_dir / "run.py", skill_dir / "run.py")
                if (version_dir / "SKILL.md").exists():
                    shutil.copy2(version_dir / "SKILL.md", skill_dir / "SKILL.md")
                shutil.rmtree(version_dir, ignore_errors=True)
        return {"error": f"skill failed to import after write: {e}"}

    index = _load_index()
    index = [e for e in index if e["name"] != name]  # replace if exists
    index.append({"name": name, "description": description, "parameters": parameters})
    _save_index(index)

    return {"status": "created", "name": name}


def delete_skill(name: str) -> dict:
    if get_skill(name) is None:
        return {"error": f"no such skill '{name}'"}

    index = [e for e in _load_index() if e["name"] != name]
    _save_index(index)
    shutil.rmtree(_skill_dir(name), ignore_errors=True)
    sys.modules.pop(f"skills.{name}.run", None)
    return {"status": "deleted", "name": name}


def _import_skill(name: str):
    """Import skills/<name>/run.py fresh every time - no bytecode caching.

    Real bug found via testing: the default SourceFileLoader writes a
    __pycache__/*.pyc next to run.py and, because every skill is always
    imported under the exact same module name (f"skills.{name}.run"), an
    overwritten skill could serve STALE compiled bytecode from a previous
    version instead of the new code - confirmed directly: after
    overwriting a skill's run.py with materially different logic, run_skill
    kept returning the OLD result. Setting dont_write_bytecode on the spec
    (and skipping the cache entirely via SourceFileLoader) avoids ever
    writing or trusting a .pyc for skill code, which changes far too often
    (every create_skill call) for that cache to be safe.
    """
    path = _skill_dir(name) / "run.py"
    loader = importlib.machinery.SourceFileLoader(f"skills.{name}.run", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.set_data = lambda *a, **k: None  # never write a .pyc for skill code
    loader.exec_module(module)
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
