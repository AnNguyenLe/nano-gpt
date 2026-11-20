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


# ===========================================================================
# [B2] FlopCounter + exact matmul inventory (TL-28 / TL-29)
# ===========================================================================
import functools  # noqa: E402

from nanogpt import flops  # noqa: E402
from nanogpt.config import DEFAULT_SEED, ModelConfig, make_rng, tiny_config  # noqa: E402
from nanogpt.model import GPT  # noqa: E402

_B_SEED = DEFAULT_SEED + 502  # per-file fixture offset (never DEFAULT_SEED itself)
_V = 55

# TL-30 config table. "in_range" marks the spec ranges (L 2-4, d 64-128, T 64-128).
C9_CONFIGS = {
    "tiny": ModelConfig(V=_V, T=4, d=8, L=2, H=2),
    "default": ModelConfig(V=_V, T=64, d=64, L=2, H=4),
    "long": ModelConfig(V=_V, T=128, d=64, L=4, H=4),
    "max": ModelConfig(V=_V, T=128, d=128, L=4, H=4),
    "discriminator": ModelConfig(V=_V, T=128, d=16, L=2, H=2),
}
IN_RANGE_C9 = ("default", "long", "max")


@functools.lru_cache(maxsize=None)
def _measure(name):
    cfg = C9_CONFIGS[name]
    model = GPT(cfg)
    params = model.init_params(make_rng(_B_SEED))
    rng = make_rng(_B_SEED + 1)
    idx = rng.integers(0, cfg.V, size=(1, cfg.T))
    tgt = rng.integers(0, cfg.V, size=(1, cfg.T))
    return cfg, flops.measure_train_flops(model, params, idx, tgt)


def test_matmul_flops_unit_cases():
    assert flops.matmul_flops((3, 4), (4, 5)) == 120
    assert flops.matmul_flops((2, 3, 4), (4, 5)) == 240
    assert flops.matmul_flops((2, 1, 3, 4), (1, 6, 4, 5)) == 2 * 12 * 3 * 4 * 5
    assert flops.matmul_flops((4,), (4, 5)) == 2 * 1 * 4 * 5   # 1-D a: m = 1
    assert flops.matmul_flops((3, 4), (4,)) == 2 * 3 * 4 * 1   # 1-D b: n = 1
    assert flops.matmul_flops((4,), (4,)) == 2 * 4


def test_matmul_flops_mismatched_k_raises():
    with pytest.raises(ValueError):
        flops.matmul_flops((3, 4), (5, 6))


def test_matmul_flops_unbroadcastable_batch_raises():
    with pytest.raises(ValueError):
        flops.matmul_flops((2, 3, 4), (5, 4, 6))


def test_counter_restores_original_matmul_on_exit_and_on_error():
    orig = ops.matmul
    with flops.FlopCounter():
        assert ops.matmul is not orig
    assert ops.matmul is orig

    with pytest.raises(RuntimeError, match="boom"):
        with flops.FlopCounter():
            raise RuntimeError("boom")
    assert ops.matmul is orig
    # and a fresh counter can start again (the active slot was released)
    with flops.FlopCounter():
        pass
    assert ops.matmul is orig


def test_calls_outside_a_counter_are_not_counted():
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    ops.matmul(a, b)  # before
    with flops.FlopCounter() as fc:
        ops.matmul(a, b)
    ops.matmul(a, b)  # after
    assert len(fc.calls) == 1
    assert fc.total() == 2 * 2 * 3 * 4


def test_nested_counter_raises_runtime_error_and_outer_survives():
    orig = ops.matmul
    with flops.FlopCounter() as outer:
        with pytest.raises(RuntimeError):
            with flops.FlopCounter():
                pass
        ops.matmul(np.ones((2, 2)), np.ones((2, 2)))
        assert outer.total() == 2 * 2 * 2 * 2
    assert ops.matmul is orig


def test_phase_attribution_and_unphased_default():
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    with flops.FlopCounter() as fc:
        ops.matmul(a, b)
        with fc.phase("forward"):
            ops.matmul(a, b)
        with fc.phase("backward"):
            ops.matmul(a, b, kind="attention")
    assert fc.counts[("unphased", "dense")] == 48
    assert fc.counts[("forward", "dense")] == 48
    assert fc.counts[("backward", "attention")] == 48
    assert fc.total("forward") == 48
    assert fc.total(kind="dense") == 96
    assert fc.total() == 144
    with pytest.raises(ValueError):
        with flops.FlopCounter() as fc2:
            with fc2.phase("sideways"):
                pass


def test_counter_returns_result_unchanged_and_validates_kind():
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    with flops.FlopCounter() as fc:
        out = ops.matmul(a, b)
        with pytest.raises(ValueError):
            ops.matmul(a, b, kind="bogus")
    assert np.array_equal(out, a @ b)
    assert len(fc.calls) == 1  # the rejected call is not counted


@pytest.mark.parametrize("cfg_name", ["tiny", "default"])
def test_counter_does_not_change_loss_or_gradients(cfg_name):
    cfg = C9_CONFIGS[cfg_name]
    model = GPT(cfg)
    params = model.init_params(make_rng(_B_SEED))
    rng = make_rng(_B_SEED + 2)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T))
    tgt = rng.integers(0, cfg.V, size=(2, cfg.T))

    loss0, c0 = model.forward_loss(idx, tgt, params)
    _, g0 = model.backward_loss(1.0, c0)
    with flops.FlopCounter():
        loss1, c1 = model.forward_loss(idx, tgt, params)
        _, g1 = model.backward_loss(1.0, c1)
    assert loss0 == loss1
    assert set(g0) == set(g1)
    for k in g0:
        assert np.array_equal(g0[k], g1[k]), k


def test_kind_tags_agree_with_operand_shapes():
    # "attention" calls are the 4-D (B, H, T, .) products; "dense" never are.
    cfg = C9_CONFIGS["tiny"]
    model = GPT(cfg)
    params = model.init_params(make_rng(_B_SEED))
    rng = make_rng(_B_SEED + 3)
    idx = rng.integers(0, cfg.V, size=(2, cfg.T))
    tgt = rng.integers(0, cfg.V, size=(2, cfg.T))
    with flops.FlopCounter() as fc:
        with fc.phase("forward"):
            _, cache = model.forward_loss(idx, tgt, params)
        with fc.phase("backward"):
            model.backward_loss(1.0, cache)
    kinds = {k for _, k, _, _ in fc.calls}
    assert kinds == {"dense", "attention"}
    for _, kind, a_shape, b_shape in fc.calls:
        if kind == "attention":
            assert len(a_shape) >= 4 and len(b_shape) >= 4, (a_shape, b_shape)
        else:
            assert not (len(a_shape) >= 4 and len(b_shape) >= 4), (a_shape, b_shape)


@pytest.mark.parametrize("name", list(C9_CONFIGS))
def test_exact_inventory_matches_counter_to_the_integer(name):
    cfg, rep = _measure(name)
    inv = flops.exact_forward_matmul_per_token(cfg)
    assert rep.tokens == cfg.T  # B = 1, sequence length = cfg.T
    assert rep.fwd_dense == inv["dense"] * rep.tokens
    assert rep.fwd_attn == inv["attention"] * rep.tokens
    # backward does exactly 2x the matmul work of forward, per kind
    assert rep.bwd_dense == 2 * rep.fwd_dense
    assert rep.bwd_attn == 2 * rep.fwd_attn


def test_exact_inventory_closed_forms_at_known_configs():
    # Hand-derived literals (TL-30 table): fwd/token = dense + attention.
    expected = {"tiny": 4208, "default": 236416, "long": 531328,
                "max": 1849088, "discriminator": 30432}
    for name, want in expected.items():
        cfg, rep = _measure(name)
        assert rep.fwd_per_token == want, (name, rep.fwd_per_token)
        assert rep.train_per_token == 3 * want


# ===========================================================================
# [B3] C9: corrected FLOPs formula band + discriminators (TL-30)
# ===========================================================================
# Ratios from the TL-30 prototype table, to 4 dp (measured / corrected formula).
TL30_FWD_RATIO = {"tiny": 0.8915, "default": 0.9526, "long": 0.9580,
                  "max": 0.9754, "discriminator": 0.8591}
TL30_NAIVE_RATIO = {"tiny": 0.9427, "default": 1.0974, "long": 1.2545,
                    "max": 1.1320, "discriminator": 1.5983}
TL30_ATTN_SHARE = {"tiny": 0.054, "default": 0.132, "long": 0.236,
                   "max": 0.138, "discriminator": 0.463}
C9_BAND_CONFIGS = ("tiny", "default", "long", "max")  # in-band by spec/TL-30


def test_formula_values_match_hand_derivation():
    cfg = C9_CONFIGS["default"]  # N = 107,712 (C1)
    assert flops.formula_forward_per_token(cfg) == 2 * 107_712 + 4 * 2 * 64 * 64
    assert flops.formula_train_per_token(cfg) == 6 * 107_712 + 12 * 2 * 64 * 64
    assert flops.naive_forward_per_token(cfg) == 2 * 107_712
    assert flops.naive_train_per_token(cfg) == 6 * 107_712


def test_within_band_edges():
    assert flops.within_band(100, 100)
    assert flops.within_band(125, 100) and flops.within_band(75, 100)
    assert not flops.within_band(126, 100) and not flops.within_band(74, 100)
    assert flops.within_band(120, 100, band=0.2) and not flops.within_band(121, 100, band=0.2)


@pytest.mark.parametrize("name", C9_BAND_CONFIGS)
def test_c9_measured_flops_within_band_of_corrected_formula(name, capsys):
    cfg, rep = _measure(name)
    fwd_pred = flops.formula_forward_per_token(cfg)
    train_pred = flops.formula_train_per_token(cfg)
    fwd_ratio = rep.fwd_per_token / fwd_pred
    train_ratio = rep.train_per_token / train_pred
    print(f"C9 {name}: fwd/tok={rep.fwd_per_token:.0f} corrected={fwd_pred} "
          f"ratio={fwd_ratio:.4f} train/tok={rep.train_per_token:.0f} "
          f"ratio={train_ratio:.4f} attn_share(measured)={rep.attn_share_fwd:.3f}")
    assert flops.within_band(rep.fwd_per_token, fwd_pred)
    assert flops.within_band(rep.train_per_token, train_pred)
    # training is exactly 3x forward in measurement AND in the formula
    assert train_ratio == pytest.approx(fwd_ratio, rel=0, abs=1e-12)
    assert round(fwd_ratio, 4) == TL30_FWD_RATIO[name]


def test_c9_discriminator_config_also_measured_and_reported():
    cfg, rep = _measure("discriminator")
    ratio = rep.fwd_per_token / flops.formula_forward_per_token(cfg)
    assert round(ratio, 4) == TL30_FWD_RATIO["discriminator"]


@pytest.mark.parametrize("name", list(C9_CONFIGS))
def test_attention_share_interpretation(name):
    cfg, rep = _measure(name)
    inv = flops.exact_forward_matmul_per_token(cfg)
    assert rep.attn_share_fwd == pytest.approx(
        inv["attention"] / (inv["dense"] + inv["attention"]))
    # share of the corrected total, per TL-30: 4LTd / (2N + 4LTd)
    formula_share = 4 * cfg.L * cfg.T * cfg.d / flops.formula_forward_per_token(cfg)
    assert round(formula_share, 3) == TL30_ATTN_SHARE[name]
    if name in ("default", "max"):
        assert 0.13 <= formula_share <= 0.20  # spec's 13-20% attention term


def test_naive_formula_fails_band_at_long_and_discriminator():
    for name in ("long", "discriminator"):
        cfg, rep = _measure(name)
        assert round(rep.fwd_per_token / flops.naive_forward_per_token(cfg), 4) == TL30_NAIVE_RATIO[name]
        assert not flops.within_band(rep.fwd_per_token, flops.naive_forward_per_token(cfg))
        assert not flops.within_band(rep.train_per_token, flops.naive_train_per_token(cfg))
        # ...while the corrected formula does pass on the in-range long config
    cfg, rep = _measure("long")
    assert flops.within_band(rep.fwd_per_token, flops.formula_forward_per_token(cfg))


def test_naive_formula_passes_band_at_default_so_band_alone_does_not_discriminate():
    # Documented on purpose: nobody may claim the +-25% band by itself proves
    # the attention term. C9 discriminates only via (a) the integer-exact
    # inventory and (b) the attention-heavy configs above.
    cfg, rep = _measure("default")
    assert flops.within_band(rep.fwd_per_token, flops.naive_forward_per_token(cfg))
    assert round(rep.fwd_per_token / flops.naive_forward_per_token(cfg), 4) == TL30_NAIVE_RATIO["default"]


def _measure_attention_blind(cfg):
    """Test-local counter variant that IGNORES kind='attention' calls."""
    model = GPT(cfg)
    params = model.init_params(make_rng(_B_SEED))
    rng = make_rng(_B_SEED + 1)
    idx = rng.integers(0, cfg.V, size=(1, cfg.T))
    tgt = rng.integers(0, cfg.V, size=(1, cfg.T))
    orig = ops.matmul
    tally = {"fwd": 0}

    def blind(a, b, *, kind="dense"):
        if kind == "dense":
            tally["fwd"] += flops.matmul_flops(np.shape(a), np.shape(b))
        return orig(a, b, kind=kind)

    ops.matmul = blind
    try:
        model.forward_loss(idx, tgt, params)
    finally:
        ops.matmul = orig
    return tally["fwd"] / idx.size


def test_attention_blind_counter_fails_exact_inventory_and_discriminator_band():
    cfg = C9_CONFIGS["discriminator"]
    blind = _measure_attention_blind(cfg)
    inv = flops.exact_forward_matmul_per_token(cfg)
    assert blind != inv["dense"] + inv["attention"]
    assert not flops.within_band(blind, flops.formula_forward_per_token(cfg))


def test_backward_outside_counter_fails_the_training_band():
    for name in ("default", "long", "max"):
        cfg, rep = _measure(name)
        fwd_only = flops.FlopReport(tokens=rep.tokens, fwd_dense=rep.fwd_dense,
                                    fwd_attn=rep.fwd_attn, bwd_dense=0, bwd_attn=0)
        assert fwd_only.train_per_token == fwd_only.fwd_per_token
        assert not flops.within_band(fwd_only.train_per_token,
                                     flops.formula_train_per_token(cfg))
        ratio = fwd_only.train_per_token / flops.formula_train_per_token(cfg)
        assert 0.30 < ratio < 0.35  # about 0.32, i.e. 1/3 of the full training work


def test_subclass_counter_approach_is_not_reimplemented():
    # TL-27: the ndarray-subclass counter is unsound (np.asarray strips it).
    # The module may mention it in prose, but must not define it in code.
    tree = ast.parse((NANOGPT_DIR / "flops.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                assert not (isinstance(base, ast.Attribute) and base.attr == "ndarray"), node.name
        if isinstance(node, ast.FunctionDef):
            assert node.name != "__array_ufunc__"