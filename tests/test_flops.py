"""tests/test_flops.py -- Engineer B, Sprint 4 (SPRINT_4_PLAN.md TL-27..TL-30).

Milestone [B1] (this commit): the AST chokepoint guard over the TL-27
`ops.matmul` refactor (nanogpt/ops.py, nanogpt/attention.py, nanogpt/model.py),
its own test-the-test on synthetic sources, `kind` validation, and
bit-identity of `ops.matmul` vs raw `@`.

Milestones [B2] (FlopCounter + exact inventory) and [B3] (C9 band +
discriminators) are appended to this same file in later commits -- never
rewritten.

Owner: Engineer B. Files: nanogpt/flops.py, tests/test_flops.py (+ the TL-27
chokepoint refactor of nanogpt/ops.py, nanogpt/attention.py, nanogpt/model.py).
"""
from __future__ import annotations

import ast
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from nanogpt import ops

NANOGPT_DIR = Path(__file__).resolve().parent.parent / "nanogpt"

# attention_reference.py is the Phase-2 naive oracle: not production code, its
# AST-guarded style (TL-27, SPRINT_2_PLAN.md A3) must not change, and it never
# runs inside a FlopCounter. It is excluded from the chokepoint scan.
CHOKEPOINT_EXCLUDED_FILES = {"attention_reference.py"}

# np.<attr>(...) forms that bypass the single chokepoint. "dot" is handled
# separately below: it is forbidden both as np.dot(...) AND as the generic
# `.dot(` method-call syntax on any object (e.g. x.dot(y)), so it is not
# gated behind the np-root check the other names use.
NUMPY_FORBIDDEN_ATTRS = {"matmul", "einsum", "tensordot", "inner", "vdot"}


def _chokepoint_scanned_files(base_dir: Path = NANOGPT_DIR):
    return sorted(p for p in base_dir.glob("*.py") if p.name not in CHOKEPOINT_EXCLUDED_FILES)


def _is_np_name(value) -> bool:
    return isinstance(value, ast.Name) and value.id == "np"


class _ChokepointVisitor(ast.NodeVisitor):
    """Walks one file's AST, tracking function nesting so the single
    permitted `a @ b` (inside a function literally named `matmul` in a file
    literally named `ops.py`) can be exempted and every other site flagged.
    """

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.func_stack: list = []
        self.violations: list = []

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            allowed = (
                self.file_name == "ops.py"
                and bool(self.func_stack)
                and self.func_stack[-1] == "matmul"
            )
            if not allowed:
                self.violations.append(
                    f"{self.file_name}:{node.lineno}: bare '@' MatMult outside ops.matmul"
                )
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "dot":
                self.violations.append(
                    f"{self.file_name}:{node.lineno}: forbidden '.dot(' call"
                )
            elif func.attr in NUMPY_FORBIDDEN_ATTRS and _is_np_name(func.value):
                self.violations.append(
                    f"{self.file_name}:{node.lineno}: forbidden call 'np.{func.attr}('"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "nanogpt.ops":
            for alias in node.names:
                if alias.name == "matmul":
                    self.violations.append(
                        f"{self.file_name}:{node.lineno}: forbidden "
                        "'from nanogpt.ops import matmul'"
                    )
        self.generic_visit(node)


def _check_chokepoint(path: Path):
    """Returns a list of human-readable violation strings for `path`."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _ChokepointVisitor(path.name)
    visitor.visit(tree)
    return visitor.violations


def _count_allowed_matmul_sites():
    """Counts BinOp MatMult nodes that qualify for the single exemption
    (inside a function named `matmul` in a file named `ops.py`), across every
    scanned file. Used to assert there is exactly one such site."""
    total = 0
    for path in _chokepoint_scanned_files():
        if path.name != "ops.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "matmul":
                for inner in ast.walk(node):
                    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.MatMult):
                        total += 1
    return total


# ---------------------------------------------------------------------------
# Guard over the real, production tree.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", _chokepoint_scanned_files(), ids=lambda p: p.name)
def test_chokepoint_guard_on_production_tree(path):
    violations = _check_chokepoint(path)
    assert not violations, "\n" + "\n".join(violations)


def test_chokepoint_guard_counts_exactly_one_allowed_site():
    assert _count_allowed_matmul_sites() == 1


def test_attention_reference_is_excluded_from_the_scan():
    assert all(p.name != "attention_reference.py" for p in _chokepoint_scanned_files())


def test_at_least_one_module_is_scanned():
    # Sanity: if this ever returns [], the guard is silently vacuous.
    assert len(_chokepoint_scanned_files()) >= 1


# ---------------------------------------------------------------------------
# Test-the-test: the guard machinery must actually catch what it claims to
# (on synthetic tmp sources, never on files literally named "ops.py" unless
# that is the point of the test).
# ---------------------------------------------------------------------------
def test_chokepoint_guard_catches_bare_matmul_operator(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text("def f(x, y):\n    return x @ y\n", encoding="utf-8")
    violations = _check_chokepoint(bad)
    assert any("bad_module.py:2:" in v and "MatMult" in v for v in violations), violations


def test_chokepoint_guard_catches_np_dot(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text(
        "import numpy as np\ndef f(x, y):\n    return np.dot(x, y)\n", encoding="utf-8"
    )
    violations = _check_chokepoint(bad)
    # np.dot(...) is caught by the generic '.dot(' rule (attr == "dot"
    # regardless of receiver), which subsumes the np-rooted case.
    assert any(".dot(" in v for v in violations), violations


def test_chokepoint_guard_catches_np_matmul(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text(
        "import numpy as np\ndef f(x, y):\n    return np.matmul(x, y)\n", encoding="utf-8"
    )
    violations = _check_chokepoint(bad)
    assert any("np.matmul(" in v for v in violations), violations


def test_chokepoint_guard_catches_generic_dot_method(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text("def f(x, y):\n    return x.dot(y)\n", encoding="utf-8")
    violations = _check_chokepoint(bad)
    assert any(".dot(" in v for v in violations), violations


def test_chokepoint_guard_catches_einsum_tensordot_inner_vdot(tmp_path):
    for attr in ("einsum", "tensordot", "inner", "vdot"):
        bad = tmp_path / f"bad_{attr}.py"
        bad.write_text(
            f"import numpy as np\ndef f(x, y):\n    return np.{attr}(x, y)\n",
            encoding="utf-8",
        )
        violations = _check_chokepoint(bad)
        assert any(f"np.{attr}(" in v for v in violations), (attr, violations)


def test_chokepoint_guard_catches_from_import_matmul(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text("from nanogpt.ops import matmul\n", encoding="utf-8")
    violations = _check_chokepoint(bad)
    assert any("from nanogpt.ops import matmul" in v for v in violations), violations


def test_chokepoint_guard_allows_ops_matmul_attribute_call(tmp_path):
    ok = tmp_path / "ok_module.py"
    ok.write_text(
        "from nanogpt import ops\n"
        "def f(x, y):\n"
        "    return ops.matmul(x, y, kind='dense')\n",
        encoding="utf-8",
    )
    assert _check_chokepoint(ok) == []


def test_chokepoint_guard_allows_the_one_site_inside_ops_matmul():
    # A standalone file named ops.py whose only MatMult is inside `matmul`
    # itself must be clean -- this is the production shape being asserted.
    source = textwrap.dedent(
        """
        def matmul(a, b, *, kind="dense"):
            if kind not in ("dense", "attention"):
                raise ValueError("bad kind")
            return a @ b
        """
    ).strip() + "\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ops.py"
        p.write_text(source, encoding="utf-8")
        assert _check_chokepoint(p) == []


def test_chokepoint_guard_flags_a_second_matmul_site_in_ops_py():
    # Even inside a file named ops.py, a SECOND bare '@' outside the matmul
    # function must still be flagged -- the exemption is for the one site,
    # not a blanket pass for the whole file.
    source = textwrap.dedent(
        """
        def matmul(a, b, *, kind="dense"):
            return a @ b

        def sneaky(a, b):
            return a @ b
        """
    ).strip() + "\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ops.py"
        p.write_text(source, encoding="utf-8")
        violations = _check_chokepoint(p)
        # Exactly one violation: the matmul()-body '@' is exempt, the
        # sneaky() one is not.
        assert len(violations) == 1, violations
        assert "MatMult" in violations[0]


def test_chokepoint_guard_flags_matmul_named_function_in_wrong_file(tmp_path):
    # The exemption is keyed on BOTH the file name and the function name: a
    # function named `matmul` living anywhere other than ops.py is not exempt.
    bad = tmp_path / "not_ops.py"
    bad.write_text(
        "def matmul(a, b, *, kind='dense'):\n    return a @ b\n", encoding="utf-8"
    )
    violations = _check_chokepoint(bad)
    assert any("MatMult" in v for v in violations), violations


def test_chokepoint_guard_clean_file_has_no_violations(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text(
        "from nanogpt import ops\n"
        "def f(x, y):\n"
        "    return ops.matmul(x, y)\n",
        encoding="utf-8",
    )
    assert _check_chokepoint(clean) == []


# ---------------------------------------------------------------------------
# kind validation.
# ---------------------------------------------------------------------------
def test_matmul_kind_default_is_dense():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(ops.matmul(a, b), a @ b)


def test_matmul_kind_dense_explicit():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(ops.matmul(a, b, kind="dense"), a @ b)


def test_matmul_kind_attention_explicit():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(ops.matmul(a, b, kind="attention"), a @ b)


@pytest.mark.parametrize("bad_kind", ["sparse", "DENSE", "", "attn", None, 1])
def test_matmul_invalid_kind_raises(bad_kind):
    a = np.eye(2)
    with pytest.raises(ValueError):
        ops.matmul(a, a, kind=bad_kind)


# ---------------------------------------------------------------------------
# Bit-identity of ops.matmul vs raw @, on 2-D, batched and broadcast shapes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((3, 4), (4, 5)),  # plain 2-D
        ((2, 3, 4), (4, 5)),  # batched vs unbatched (broadcast)
        ((2, 3, 4), (2, 4, 5)),  # batched, matching batch
        ((2, 1, 3, 4), (1, 6, 4, 5)),  # 4-D broadcast batch
        ((5, 6), (6,)),  # matrix @ vector (NumPy 1-D promotion)
        ((6,), (6, 5)),  # vector @ matrix
        ((7,), (7,)),  # vector @ vector (dot product)
        ((2, 5, 6), (2, 6, 7)),  # batched 3-D, equal batch
    ],
    ids=lambda s: str(s),
)
def test_matmul_bit_identical_to_at_operator(a_shape, b_shape):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(a_shape)
    b = rng.standard_normal(b_shape)
    assert np.array_equal(ops.matmul(a, b), a @ b)
    assert np.array_equal(ops.matmul(a, b, kind="attention"), a @ b)
