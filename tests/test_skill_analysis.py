"""Tests for tools/skill_analysis.py - the static AST classifier deciding
whether a skill can be safely shown in plan mode.

This is a whitelist-based safety classifier: a skill is "mutating" by
default, and only "read_only" if every call in its code resolves to an
explicit safe pattern. Every test here either confirms a genuinely pure
skill gets classified read_only, or confirms something that CAN mutate (or
that the analyzer can't fully account for) stays mutating - never the
other way around, since a false "read_only" would be a real safety hole."""

from skull.tools.skill_analysis import classify_skill_code


def test_classifies_pure_arithmetic_as_read_only():
    code = """
def run(**kwargs):
    c = kwargs["celsius"]
    return {"fahrenheit": round(c * 9 / 5 + 32, 2)}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_string_methods_as_read_only():
    code = """
def run(**kwargs):
    text = kwargs.get("text", "")
    return {"result": text.upper().strip()}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_str_constructor_then_method_as_read_only():
    """The exact real-world shape that initially broke the analyzer:
    str(x).upper() - a method call on the RESULT of a call, not a plain
    Name/Attribute chain."""
    code = """
def run(text, **kwargs):
    return {"result": str(text).upper()}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_local_helper_function_calls_as_read_only():
    code = """
def _gb(n):
    return round(n / (1024 ** 3), 2)

def run(**kwargs):
    return {"size": _gb(kwargs["bytes"])}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_helper_defined_after_use_as_read_only():
    """Python allows forward references to functions defined later in the
    same module (the body isn't executed until called) - the analyzer must
    handle this via its two-pass collection of local function names."""
    code = """
def run(**kwargs):
    return {"size": _gb(kwargs["bytes"])}

def _gb(n):
    return round(n / (1024 ** 3), 2)
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_math_module_as_read_only():
    code = """
import math

def run(**kwargs):
    return {"root": math.sqrt(kwargs["n"])}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_shutil_disk_usage_as_read_only():
    """The real skill this was calibrated against: disk_space imports
    shutil (a module that ALSO has shutil.rmtree, very much mutating) but
    only calls the read-only disk_usage - must not get flagged just for
    the import."""
    code = """
import os
import shutil

def run(**kwargs):
    path = kwargs.get("path") or os.getcwd()
    total, used, free = shutil.disk_usage(path)
    return {"total": total, "used": used, "free": free, "path": os.path.abspath(path)}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_datetime_now_and_formatting_as_read_only():
    code = """
from datetime import datetime

def run(**kwargs):
    now = datetime.now()
    return {"iso": now.isoformat(), "day_name": now.strftime("%A")}
"""
    assert classify_skill_code(code) == "read_only"


def test_classifies_file_write_as_mutating():
    code = """
def run(**kwargs):
    with open(kwargs["path"], "w") as f:
        f.write(kwargs["content"])
    return {"status": "written"}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_subprocess_as_mutating_unconditionally():
    """The exact real-world case that ruled out any blacklist-only
    design: build_docx's actual file write happens inside a string
    executed via subprocess in a completely separate process, invisible to
    AST analysis of this file - subprocess itself must be an unconditional
    red flag, regardless of what's passed to it."""
    code = """
import subprocess

def run(**kwargs):
    subprocess.run(["echo", "hello"], capture_output=True)
    return {"status": "ok"}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_os_remove_as_mutating():
    code = """
import os

def run(**kwargs):
    os.remove(kwargs["path"])
    return {"status": "deleted"}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_network_request_as_mutating():
    code = """
import requests

def run(**kwargs):
    resp = requests.get(kwargs["url"])
    return {"text": resp.text}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_eval_as_mutating():
    code = """
def run(**kwargs):
    return {"result": eval(kwargs["expr"])}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_exec_as_mutating():
    code = """
def run(**kwargs):
    exec(kwargs["code"])
    return {"status": "ok"}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_dynamic_import_as_mutating():
    code = """
def run(**kwargs):
    mod = __import__(kwargs["module_name"])
    return {"status": "ok"}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_call_on_unresolvable_expression_as_mutating():
    """Calling through a subscript/comprehension result - can't statically
    prove anything about it, must default to mutating."""
    code = """
def run(**kwargs):
    handlers = kwargs["handlers"]
    return {"result": handlers[0]()}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_syntax_error_as_mutating():
    code = "def run(**kwargs)\n    return {'oops': True}"  # missing colon
    assert classify_skill_code(code) == "mutating"


def test_classifies_shutil_rmtree_as_mutating_despite_disk_usage_whitelist():
    """Confirms the whitelist is per-function, not per-module - shutil
    itself is never blanket-safe just because disk_usage is."""
    code = """
import shutil

def run(**kwargs):
    shutil.rmtree(kwargs["path"])
    return {"status": "deleted"}
"""
    assert classify_skill_code(code) == "mutating"


def test_classifies_os_path_functions_as_read_only():
    code = """
import os

def run(**kwargs):
    return {"abs": os.path.abspath(kwargs["path"]), "exists": os.path.exists(kwargs["path"])}
"""
    assert classify_skill_code(code) == "read_only"
