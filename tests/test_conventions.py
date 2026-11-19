"""tests/test_conventions.py — Engineer A, milestone A4 (integration guard).

AST-scans every nanogpt/*.py file EXCEPT config.py, and fails if any of
them:
  1. contains a float literal equal to one of the Frozen-Conventions values
     that must instead be imported from nanogpt.config (reinventing them
     locally is exactly the bug this guard exists to catch);
  2. references float32 / float16 (as an attribute like `np.float32`, a
     bare name, or a string such as `"float32"`);
  3. imports torch, jax, tensorflow or autograd;
  4. contains a `TODO(SPRINT<n>-...)` marker for any sprint (a temporary
     fallback per SPRINT_1_PLAN.md §10 / SPRINT_2_PLAN.md §9 that must be
     removed before that sprint ships). Widened from `TODO(SPRINT1-` to
     `TODO(SPRINT` at Sprint 2 integration (SPRINT_2_PLAN.md §10 item 1).

This is a textual/structural guard over the other engineers' source, not a
semantic one: it enforces the *shape* of the Frozen Conventions contract
(single source of truth, no stray framework imports, no leftover
scaffolding markers), so B's and C's modules cannot ship with a locally
reinvented epsilon or a temporary config shim still in place.
"""

import ast
from pathlib import Path

import pytest

NANOGPT_DIR = Path(__file__).resolve().parent.parent / "nanogpt"

FORBIDDEN_FLOATS = {1e-5, 1e-8, 1e-6, 1e-10, 0.02, 0.044715}
ALLOWED_FLOATS = {0.0, 0.5, 1.0, 2.0, 3.0}  # documented as always fine (GELU/LN algebra, eps=0)
FORBIDDEN_DTYPE_NAMES = {"float32", "float16"}
FORBIDDEN_IMPORT_ROOTS = {"torch", "jax", "tensorflow", "autograd"}
TODO_MARKER = "TODO(SPRINT"


def _scanned_files():
    return sorted(p for p in NANOGPT_DIR.glob("*.py") if p.name != "config.py")


def _check_file(path: Path):
    """Returns a list of human-readable violation strings for `path`."""
    source = path.read_text(encoding="utf-8")
    violations = []

    if TODO_MARKER in source:
        line_no = next(
            (i + 1 for i, line in enumerate(source.splitlines()) if TODO_MARKER in line),
            None,
        )
        violations.append(f"{path.name}:{line_no}: contains a {TODO_MARKER}...) marker")

    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        # 1. Forbidden float literals.
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if any(node.value == forbidden for forbidden in FORBIDDEN_FLOATS):
                violations.append(
                    f"{path.name}:{getattr(node, 'lineno', '?')}: forbidden float literal "
                    f"{node.value!r} (must be imported from nanogpt.config)"
                )

        # 2. float32 / float16 as attribute, bare name, or string constant.
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DTYPE_NAMES:
            violations.append(
                f"{path.name}:{getattr(node, 'lineno', '?')}: forbidden dtype attribute "
                f"'.{node.attr}'"
            )
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_DTYPE_NAMES:
            violations.append(
                f"{path.name}:{getattr(node, 'lineno', '?')}: forbidden dtype name '{node.id}'"
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN_DTYPE_NAMES:
                violations.append(
                    f"{path.name}:{getattr(node, 'lineno', '?')}: forbidden dtype string "
                    f"{node.value!r}"
                )

        # 3. Forbidden framework imports.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{path.name}:{getattr(node, 'lineno', '?')}: forbidden import "
                        f"'{alias.name}'"
                    )
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{path.name}:{getattr(node, 'lineno', '?')}: forbidden import "
                        f"from '{node.module}'"
                    )

    return violations


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: p.name)
def test_module_has_no_convention_violations(path):
    violations = _check_file(path)
    assert not violations, "\n" + "\n".join(violations)


def test_at_least_one_module_is_scanned():
    # Sanity: if this ever returns [], the guard is silently vacuous.
    assert len(_scanned_files()) >= 1


def test_config_py_itself_is_excluded_from_the_scan():
    assert all(p.name != "config.py" for p in _scanned_files())


# ---------------------------------------------------------------------------
# Test-the-test: the guard machinery must actually catch what it claims to.
# ---------------------------------------------------------------------------
def test_guard_catches_forbidden_float_literal(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("EPS = 1e-5\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("1e-05" in v or "forbidden float literal" in v for v in violations)


def test_guard_allows_whitelisted_floats(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("x = 0.0\ny = 0.5\nz = 1.0\nw = 2.0\nv = 3.0\n", encoding="utf-8")
    assert _check_file(ok) == []


def test_guard_catches_float32_attribute(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import numpy as np\nx = np.float32\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("float32" in v for v in violations)


def test_guard_catches_float32_string(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("dtype_name = 'float32'\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("float32" in v for v in violations)


def test_guard_catches_torch_import(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("torch" in v for v in violations)


def test_guard_catches_jax_import_from(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from jax import numpy as jnp\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("jax" in v for v in violations)


def test_guard_catches_todo_marker(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("# TODO(SPRINT1-CONFIG): delete this block\nEPS = 0.0\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("TODO(SPRINT" in v for v in violations)


@pytest.mark.parametrize("marker", ["TODO(SPRINT2-ATTN)", "TODO(SPRINT3-MODEL)"])
def test_guard_catches_later_sprint_todo_markers(tmp_path, marker):
    # Sprint 2 widening: a shim marker from any sprint, not just Sprint 1,
    # must make the guard fail.
    bad = tmp_path / "bad.py"
    bad.write_text(f"# {marker}: delete this shim\nx = 0.0\n", encoding="utf-8")
    violations = _check_file(bad)
    assert any("bad.py:1:" in v and "TODO(SPRINT" in v for v in violations), violations


def test_guard_clean_file_has_no_violations(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text(
        "import numpy as np\n"
        "from nanogpt.config import LN_EPS\n"
        "def f(x):\n"
        "    return 0.5 * x + 1.0\n",
        encoding="utf-8",
    )
    assert _check_file(clean) == []
