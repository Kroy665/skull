"""Static classification of whether a skill's code can mutate anything
outside its own local variables, so plan mode can allow a genuinely
read-only skill (celsius_to_fahrenheit) while still excluding one that
writes files, runs a subprocess, or hits the network (build_docx,
unsplash_search) - instead of excluding every self-created skill wholesale
just because most of them *could* be mutating.

Whitelist-based, not blacklist-based, and this direction matters: a skill
defaults to MUTATING unless its whole AST provably contains nothing but
calls to an explicit small set of known-safe stdlib functions/modules. A
blacklist ("flag subprocess/os.remove/etc.") was considered and rejected -
real skills already in this repo prove it's unsafe. build_docx's actual
file write happens inside a string that gets exec'd in a *subprocess* via
`subprocess.run([py, "-c", code])` - the write itself is invisible to AST
analysis of build_docx/run.py, since it's not real code in that file's
AST, just string content. The only safe stance once a skill uses
subprocess, eval, exec, compile, importlib, or any dynamically-constructed
code is to call it mutating unconditionally - there's no way to prove
what a subprocess or an exec'd string actually does by reading the
outer file's AST.

Conversely, a naive import-based blacklist is also wrong: disk_space (a
real skill in this repo) imports shutil, but only for the read-only
shutil.disk_usage() - "imports shutil" alone is not a safe signal either
direction, which is why this classifier resolves down to specific
allowed (module, function) pairs, not whole modules.
"""

import ast

# (module, name) pairs that are safe to call regardless of arguments - pure
# computation, or reads of already-safe process/OS state. Deliberately
# narrow: this whitelist grows only when a specific real need shows up,
# never speculatively.
SAFE_CALLS = {
    ("math", None),  # any math.* function - all pure computation
    ("datetime", None),  # datetime.now() etc. reads the clock, doesn't mutate
    ("json", "dumps"),
    ("json", "loads"),
    ("re", None),  # re.* - pure string matching
    ("shutil", "disk_usage"),  # read-only despite living in a "mutating-sounding" module
    ("os", "path"),  # the os.path submodule is all pure string/stat operations
}

# Modules that are unconditionally safe to import and call anything from -
# genuinely can't mutate external state no matter what function is called.
SAFE_MODULES = {"math", "re", "string", "collections", "itertools", "functools", "textwrap", "unicodedata"}

# os/shutil functions that only read process/filesystem state, never
# create/modify/delete anything - distinct from SAFE_CALLS' (module, func)
# pairs because these live under `os`/`shutil`, which are NOT blanket-safe
# modules (os.remove, shutil.rmtree, etc. very much mutate).
SAFE_OS_READS = {"os.getcwd", "os.name", "os.path", "shutil.disk_usage"}

# Builtin container methods safe to call on ANY object regardless of type -
# these never mutate anything outside the object's own value (a dict/list
# a skill already owns, or a string, which is immutable in Python anyway).
# Only str/dict/list read-only methods are included; anything mutating
# (list.append, dict.update, etc.) is deliberately excluded even though
# those only mutate a LOCAL value, since telling "kwargs.get(...)" apart
# from "some_shared_object.append(...)" isn't reliably possible without
# full type inference - excluding all of them is the conservative choice.
SAFE_INSTANCE_METHODS = {
    "get", "upper", "lower", "strip", "lstrip", "rstrip", "split", "rsplit",
    "join", "replace", "startswith", "endswith", "format", "title", "count",
    "find", "rfind", "index", "isdigit", "isalpha", "isalnum", "isspace",
    "encode", "decode", "keys", "values", "items", "capitalize", "swapcase",
    "zfill", "ljust", "rjust", "center", "partition", "rpartition", "expandtabs",
    # datetime/date/time formatting - reads an already-constructed value,
    # never mutates anything (datetime objects are immutable anyway).
    "isoformat", "strftime", "timestamp", "date", "time", "weekday", "isoweekday",
}

# Any of these appearing anywhere in the code (import or call) makes the
# skill mutating unconditionally, full stop - no further analysis needed,
# because they can execute code this analyzer can't see into.
DYNAMIC_EXECUTION_NAMES = {"eval", "exec", "compile", "__import__"}
DYNAMIC_EXECUTION_MODULES = {"subprocess", "os.system", "importlib", "ctypes", "multiprocessing"}


class _MutationFinder(ast.NodeVisitor):
    def __init__(self):
        self.is_mutating = False
        self._imported_names = {}  # local name -> real module path, e.g. "np" -> "numpy"
        self._local_functions = set()  # names defined with `def` inside this skill's code

    def _mark_mutating(self):
        self.is_mutating = True

    def visit_FunctionDef(self, node):
        self._local_functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._local_functions.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name
            local_name = alias.asname or alias.name.split(".")[0]
            self._imported_names[local_name] = module
            if module.split(".")[0] in DYNAMIC_EXECUTION_MODULES or module in DYNAMIC_EXECUTION_MODULES:
                self._mark_mutating()
            elif module.split(".")[0] not in SAFE_MODULES and module not in ("os",):
                # An import of anything outside the safe set is only a
                # problem once it's actually CALLED unsafely - handled in
                # visit_Call - but os itself is allowed to be imported
                # since os.path (safe) lives under it; other os.* calls are
                # caught at call time below.
                pass
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        if module.split(".")[0] in DYNAMIC_EXECUTION_MODULES or module in DYNAMIC_EXECUTION_MODULES:
            self._mark_mutating()
        for alias in node.names:
            local_name = alias.asname or alias.name
            self._imported_names[local_name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name, is_bare_method = self._resolve_call_name(node.func)
        if func_name is None:
            # Couldn't statically resolve what's being called at all (e.g.
            # calling a subscript result, a lambda, an arbitrary
            # expression) - can't prove it's safe, so treat conservatively.
            self._mark_mutating()
        elif func_name in DYNAMIC_EXECUTION_NAMES:
            self._mark_mutating()
        elif is_bare_method:
            # A method call on something that isn't a simple imported-module
            # path (e.g. `text.upper()`, `str(x).lower()`, `kwargs.get(...)`)
            # - only safe if the bare method name itself is on the
            # instance-method whitelist, since the receiver's actual type
            # can't be inferred statically.
            if func_name not in SAFE_INSTANCE_METHODS:
                self._mark_mutating()
        elif not self._is_whitelisted(func_name):
            self._mark_mutating()
        self.generic_visit(node)

    def _resolve_call_name(self, func_node) -> tuple:
        """Return (dotted_name, is_bare_method) for a call target.

        dotted_name is e.g. "os.path.join" or "round" when the chain
        resolves all the way back to a plain Name (a module or builtin);
        is_bare_method is True and dotted_name is just the trailing
        attribute (e.g. "upper") when the chain bottoms out at something
        else - a Call, subscript, etc. - meaning this is a method call on
        some value, not a module-qualified function, and must be judged
        purely by method name via SAFE_INSTANCE_METHODS.

        Returns (None, False) if nothing can be resolved at all (e.g. the
        call target is itself an arbitrary non-attribute expression)."""
        if isinstance(func_node, ast.Name):
            return func_node.id, False

        if not isinstance(func_node, ast.Attribute):
            return None, False

        parts = [func_node.attr]
        node = func_node.value
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            # A Name only counts as the root of a module-qualified path if
            # it's actually a known import (or a local function, handled
            # separately) - `kwargs.get(...)`, `word.lower()`, `text.split()`
            # all bottom out at a Name too, but `kwargs`/`word`/`text` are
            # local variables, not imports, so these are method calls on a
            # value and must be judged by bare method name only.
            if node.id in self._imported_names or node.id in self._local_functions:
                parts.append(node.id)
                parts.reverse()
                return ".".join(parts), False
            return func_node.attr, True

        # Receiver is something other than a plain dotted name (a Call,
        # subscript, etc.) - this is a method call on a value, judged by
        # bare method name only.
        return func_node.attr, True

    def _is_whitelisted(self, dotted_name: str) -> bool:
        root = dotted_name.split(".")[0]

        # A call to a function defined earlier/later in this same skill's
        # code - safe as a call site, since the helper's own body is
        # visited independently and any unsafe call inside IT still marks
        # the whole skill mutating.
        if root in self._local_functions and root not in self._imported_names:
            return True

        # Builtins with no side effects: pure computation/formatting only.
        # This list is intentionally short - anything not here (open, print
        # is fine but excluded for simplicity, input, etc.) falls through
        # to "not whitelisted" and marks the skill mutating.
        safe_builtins = {
            "len", "str", "int", "float", "round", "abs", "min", "max", "sum",
            "sorted", "reversed", "list", "dict", "set", "tuple", "enumerate",
            "zip", "range", "map", "filter", "any", "all", "isinstance", "type",
            "bool", "repr", "format", "ord", "chr", "print",
        }
        if root not in self._imported_names and root in safe_builtins:
            return True

        resolved_root = self._imported_names.get(root, root)
        resolved_root_top = resolved_root.split(".")[0]

        if resolved_root_top in SAFE_MODULES:
            return True

        # Specific read-only os/shutil functions - these modules are NOT
        # blanket-safe (os.remove, shutil.rmtree mutate), so only an exact
        # dotted-path match here counts.
        full_resolved = f"{resolved_root}.{dotted_name.split('.', 1)[1]}" if "." in dotted_name else resolved_root
        if full_resolved in SAFE_OS_READS or resolved_root in SAFE_OS_READS:
            return True
        # os.path.<anything> is safe (os.path itself is in SAFE_OS_READS as
        # a prefix) - e.g. os.path.abspath, os.path.join, os.path.exists.
        if resolved_root_top == "os" and dotted_name.startswith("os.path."):
            return True

        # Check exact (module, function) whitelist entries, e.g.
        # ("shutil", "disk_usage") or ("json", "dumps").
        rest = dotted_name.split(".", 1)[1] if "." in dotted_name else None
        for safe_module, safe_func in SAFE_CALLS:
            if resolved_root_top != safe_module:
                continue
            if safe_func is None:
                return True
            if rest == safe_func or dotted_name.endswith(f".{safe_func}"):
                return True
            if resolved_root == f"{safe_module}.{safe_func}":
                return True

        return False


def classify_skill_code(code: str) -> str:
    """Return "read_only" if `code` can be statically proven to never
    mutate anything outside its own local variables, else "mutating".
    Any code this analyzer can't fully account for - a syntax error, a
    call it can't resolve, anything using subprocess/eval/exec/dynamic
    imports - is classified "mutating". This is a safety classifier: a
    false "mutating" just means a skill stays hidden in plan mode
    unnecessarily; a false "read_only" would be a real safety hole, so
    every uncertain case must resolve to "mutating"."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "mutating"

    # Two passes: first collect every locally-defined function name so a
    # call to a helper defined LATER in the file (legal in Python - a
    # function body isn't executed until called, so forward references
    # within the same module are fine) is still recognized as safe. The
    # helper's own body is still visited normally in the second pass, so
    # unsafe code inside a helper is still caught.
    finder = _MutationFinder()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            finder._local_functions.add(node.name)

    try:
        finder.visit(tree)
    except Exception:
        return "mutating"

    return "mutating" if finder.is_mutating else "read_only"
