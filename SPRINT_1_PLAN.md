# Sprint 1 Plan — Phase 0 + Phase 1 (Tech Lead breakdown)

**Scope:** Execution Plan Phase 0 (Frozen Conventions, Scaffolding & Data) + Phase 1 (Core
Differentiable Primitives). Nothing else.
**Sources reconciled:** `../nanoGPT_build_requirements.md` (spec, incl. §0 Frozen Conventions),
`../nanoGPT_execution_plan.md` (plan), PM sprint backlog (P0-01..P1-11), PO Sprint 1 acceptance
priorities. Where they disagree, the order of authority is: **spec > plan > PO > PM**. The Tech
Lead (TL) rulings in §2 settle what the documents leave open.
**Team:** 3 engineers. Each owns a set of files, and no file has two owners (see §4).

---

## 1. Sprint 1 Definition of Done (what "green" means)

1. The corpus is resolved without network access. `data/corpus.txt` is used if present.
   Otherwise the seeded synthetic generator `synthetic-v1` is used. The generator, seed, char
   count, SHA-256 and V are recorded in `PROGRESS.md`. The tokenizer round-trips losslessly, and
   V is identical on every run.
2. `nanogpt/config.py` is the single source of truth for every §0 row. A guard test
   (`tests/test_conventions.py`) proves that no `nanogpt/` module redefines those values and
   that none imports torch, jax, tensorflow or autograd.
3. `nanogpt/gradcheck.py` is a generic central-difference harness. Before any real layer is
   checked, it passes on `y = xW` and flags deliberately broken backwards.
4. linear, embedding (`np.add.at`, duplicate indices tested), GELU-tanh, stable softmax,
   stable cross-entropy and affine LayerNorm (ddof=0, ε=1e-5) all follow the module contract
   (§3) and pass gradcheck with `rel_err < 1e-6` in float64 on every input and parameter tensor.
5. **C3** passes. The ε=0 no-affine instance has rank d−2, J·1=0 and J·x=0. The production
   ε=1e-5 instance has rank d−1, which is recorded as expected. The test must not pass
   vacuously.
6. **C4** passes. All eigenvalues of diag(p)−ppᵀ are ≥ −1e-10, its rows sum to 0, and
   shift-invariance holds. The Jacobian is extracted **from the implementation**, not only
   evaluated from the formula.
7. `python -m pytest tests/ -q` is one command, all tests are green, the whole suite takes
   < 60 s (target < 20 s), and no single test file takes > 10 s (the course's checker budget).
8. `PROGRESS.md` has a Phase 0 entry and a Phase 1 entry, each with what passed, deviations
   and the next phase pointer (Phase 2: Causal Multi-Head Self-Attention).

Of C1–C9, **only C3 and C4** are Sprint 1 acceptance criteria. The others are deferred (§9).

---

## 2. Reconciliation log and Tech Lead rulings

### 2.1 PM backlog vs. PO priorities

| # | Item | PM backlog | PO priorities | **Ruling** |
|---|---|---|---|---|
| R1 | Skeleton | P0-01: stub all 16 modules + all test stubs | Skeleton only for files Sprint 1 touches; do not build past the primitives | **PO.** Create only the Sprint 1 files listed in §4. No `attention.py`/`mlp.py`/`param_count.py`/`flops.py`… stubs. Empty placeholders invite premature work, and the guard test would scan them. Phase 2+ creates its own files. |
| R2 | config contents | P0-02 omits GELU constants, γ/β/bias init values, d%H validation | Lists them all | **PO, plus** the spec tolerances (C3 SVD 1e-6, C4 PSD 1e-10, gradcheck 1e-6) and seeds. Full API in §5. |
| R3 | Corpus generator location | P0-03 separate task, location unspecified | Inside `tokenizer.py` | **PO/plan:** `tokenizer.py` (plan §3 layout). |
| R4 | CE normalizer ("batch_size") | Left ambiguous | Number of target positions averaged by the forward loss | **PO.** N = `targets.size` (= B·T for a (B,T) batch). Forward and backward use the same N. Record in PROGRESS.md. |
| R5 | Default H | Not pinned | Default H dividing d, unused until Phase 2 | **H = 4** for the default config (d=64 ⇒ head_dim=16). The tiny config (d=8, L=2, H=2, T=4) uses H=2. `ModelConfig` validates d % H == 0. |
| R6 | Where per-module gradchecks live | P1-08: one combined task, implicitly one test file (`test_gradients.py`) | Not specified | **Split by owner** to avoid a shared-file collision: `tests/test_ops.py` (B), `tests/test_layernorm.py` + `tests/test_loss.py` (C). `tests/test_gradients.py` is **reserved for Phase 3 (C6)** and is not created in Sprint 1. Record as a deviation from the plan's file list. |
| R7 | C3 coverage | P1-09: rank only | Rank + J·1=0 + J·x=0 + d−1 production case | **PO, plus** the σ² precondition from TL-3 below. |
| R8 | C4 coverage | P1-10: properties of the formula | Properties incl. shift-invariance | **PO, plus** the Jacobian must be extracted from `softmax_backward` (basis vectors) and checked against both the closed form and an FD Jacobian. Otherwise C4 tests the textbook, not our code. |
| R9 | Test-the-test, determinism, guard test, LSE smoke test | Absent | P1/P2 | **Included** as acceptance items for the relevant owner (§6–§8). |
| R10 | Loss dependency on softmax | P1-05 "reuses" P1-04 | — | `loss.py` imports `ops.logsumexp`. B ships `logsumexp` + `softmax_forward` as its **first** commit (milestone B1). Fallback protocol in §10. |
| R11 | Phase 1 sign-off vs. corpus gate | P1-11 cannot close before P0-03/04 | Same | **Agreed.** Primitive coding may start before the corpus lands. Phase 1 sign-off may not. |

### 2.2 Technical rulings (TL, from my own reading of the math)

**TL-1. Gating `rel_err` is applied tensor-wise with the Frobenius norm.**
`config.rel_err(a, b) = ‖a−b‖ / max(‖a‖+‖b‖, 1e-8)`. For scalars this is *exactly* the §0
formula. Why: with h=1e-5 in float64, the central-difference absolute error is about
1e-11·|f| (roundoff) + 1.7e-11·|f'''| (truncation), so roughly 1e-10 for our tiny fixtures. An
element-wise ratio with only a 1e-8 floor therefore fails spuriously on any gradient element
of about 1e-4 or smaller. Seeded fixtures would then fail deterministically, and an engineer
would chase a non-bug. The harness **also reports** the element-wise max (`max_elem_rel_err`)
as a diagnostic, but it is not gated. The 1e-8 floor still makes exact-zero gradients
(T=1 Q/K in Phase 2) report 0, not 0/0. Record this in PROGRESS.md as an interpretation of the spec.

**TL-2. The production LayerNorm backward uses the same formula as the plan's no-affine
ε=0 backward, with s = √(σ²+ε) in place of σ.** This is exact, not an approximation.
Derivation: ∂s/∂x_j = (x_j−μ)/(d·s) ⇒ ∂ŷ_i/∂x_j = (1/s)(δ_ij − 1/d) − (1/s)(1/d)ŷ_iŷ_j with
ŷ = (x−μ)/s. So J = (1/s)(I − 11ᵀ/d − ŷŷᵀ/d) for **any** ε ≥ 0, and J is symmetric. The production
backward is therefore:

```
dβ = Σ_lead d_out           dγ = Σ_lead d_out ⊙ ŷ
g  = d_out ⊙ γ
dx = (1/s) · (g − mean(g) − ŷ ⊙ mean(g ⊙ ŷ))      # means over the last axis, keepdims
```
The PO's concern "must use 1/√(σ²+ε)" is satisfied by *how ŷ and s are computed*. Plugging the
ε=0 σ into the production path is the bug to avoid.

**TL-3. The C3 production "rank d−1" result depends on σ² and needs a precondition.**
In the ŷ direction, J's eigenvalue is (1/s)(1 − ‖ŷ‖²/d) = (1/s)·ε/(σ²+ε). All other eigenvalues
on the complement of {1, ŷ} are 1/s = σ_max. So the smallest nonzero singular value relative to
σ_max is **ε/(σ²+ε)**. With the spec threshold 1e-6·σ_max and ε=1e-5, the production
instance reads rank d−1 only if σ² ≲ 9. **Fixture rule:** standardize the production C3 input
to population variance exactly 1 (ratio ≈ 1e-5, which is 10× above threshold). If σ² ≳ 10, the
production instance numerically reads d−2. That is also not a bug. Do **not** change ε or the
SVD threshold to force either answer. Engineer C adds a test that documents this boundary
(TL-3 test in §8).

**TL-4. Softmax backward notation.** The plan's "dS = P⊙(dP − Σ(dP⊙P))" means: given the
upstream gradient `dp = ∂L/∂p`, the gradient w.r.t. the scores is
`dz = p ⊙ (dp − sum(dp ⊙ p, axis=-1, keepdims=True))`.

**TL-5. Softmax must tolerate −∞ entries.** Phase 2 masks with −∞, so a row containing −∞ but
at least one finite entry must give exact 0 probabilities there and no NaN. An all-−∞ row is a
precondition violation (the causal mask always keeps the diagonal). This is documented, not
handled.

**TL-6. Gradcheck LayerNorm/linear with non-init parameters.** At γ=1, a backward that forgets
`g = d_out⊙γ` is invisible. Gradcheck fixtures use γ = 1 + 0.5·N(0,1), β ~ N(0,1) and b ~ N(0,1).
C3 is the only place γ=1, β=0 is required, because C3 is about the normalization map.

**TL-7. Dtype strictness.** Float inputs and parameters must already be float64. Modules raise
`TypeError` on any other float dtype instead of silently casting. Float32 drift would otherwise
show up as a mysterious gradcheck failure. Integer index arrays must be an integer dtype.

**TL-8. ε=0 LayerNorm and constant rows.** The ε=0 instance raises `ValueError` if any row has
σ == 0 (spec precondition "non-constant x"). The production ε>0 path handles constant rows
(ŷ=0).

---

## 3. Shared module contract (all three engineers code to this)

```python
forward(x, params)    -> (out, cache)
backward(d_out, cache) -> (d_x, d_params)
```

- **`params`** is a `dict[str, np.ndarray]` keyed by **global** parameter name. A module reads
  only its own keys and **never mutates** `params` or `x`.
- **`d_params`** is a new `dict[str, np.ndarray]` with **exactly** the module's own keys (same
  shapes, float64). Stateless ops (GELU, softmax, loss) return `{}`.
- **Cross-module accumulation** goes through `ops.accumulate_grads(grads, d_params)`: on the
  first insert it stores a *copy*, and after that it does `grads[k] += v`. It never rebinds or
  overwrites. This makes C7 (tied `wte`/`lm_head`) hold by construction in Phase 3.
- **Integer-index input** (embedding): `backward` returns `d_x = None`. `d_W` is built with
  `np.add.at(dW, idx, d_out)`, **never** `dW[idx] += d_out`.
- **Loss:** `cross_entropy_forward(logits, targets) -> (loss, cache)`, where `loss` is a Python
  float. `cross_entropy_backward(d_loss, cache) -> (d_logits, {})`. The gradcheck harness takes it
  through a closure: `lambda x, p: cross_entropy_forward(x, targets)`.
- **`cache`** is opaque. `backward` depends only on `(d_out, cache)`, with no globals and no
  reading of `params` again, except through references stored in the cache. `backward` must not
  mutate `d_out`.
- **Shapes:** every op works on arbitrary leading dims `(..., feature)`. Normalization, softmax
  and loss reductions act on the **last axis**. Test with a 3-D `(B, T, d)` shape.
- **Parameterized ops are small classes** that hold names and hyper-parameters, with `forward`,
  `backward` and `init_params(rng)` methods. Stateless ops are module-level functions.
- **Parameter naming:** `Linear(prefix, …)` → `f"{prefix}.W"`, `f"{prefix}.b"`.
  `Embedding(prefix, …)` → `f"{prefix}.W"`. `LayerNorm(prefix, …)` → `f"{prefix}.gamma"`,
  `f"{prefix}.beta"`. (Phase 3 will tie `lm_head` to `"wte.W"`.)
- **Init** (§0): weights `rng.normal(0, INIT_STD, shape)`, biases 0, γ = 1, β = 0, all float64,
  with an explicit `np.random.Generator` passed in. Nothing uses global RNG state.

---

## 4. Repository layout and file ownership

Root: `c:/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy/` (directories created by TL).

| Path | Owner | Notes |
|---|---|---|
| `README.md` | A | Stub: env setup + `python -m pytest tests/ -q` only |
| `PROGRESS.md` | A | Phase 0 entry; Phase 1 entry written at integration from B/C result blocks |
| `requirements.txt` | A | `numpy`, `scipy`, `pytest` only |
| `conftest.py` (repo root, empty or near-empty) | A | Puts the repo root on `sys.path` so `import nanogpt` works under `pytest tests/` |
| `data/` (+ `data/.gitkeep`) | A | `corpus.txt` is optional and absent today, so synthetic is used |
| `nanogpt/__init__.py` | A | Docstring + `__version__` only. **No eager submodule imports.** |
| `nanogpt/config.py` | A | Frozen Conventions |
| `nanogpt/tokenizer.py` | A | Corpus resolution + synthetic generator + char tokenizer |
| `nanogpt/gradcheck.py` | A | Generic FD harness + `numerical_jacobian` |
| `tests/test_config.py`, `tests/test_tokenizer.py`, `tests/test_gradcheck_harness.py`, `tests/test_conventions.py` | A | |
| `nanogpt/ops.py` | B | linear, embedding, GELU, softmax, logsumexp, accumulate_grads |
| `tests/test_ops.py`, `tests/test_softmax_properties.py` | B | gradchecks + C4 |
| `nanogpt/layernorm.py`, `nanogpt/loss.py` | C | |
| `tests/test_layernorm.py`, `tests/test_layernorm_rank.py`, `tests/test_loss.py` | C | gradchecks + C3 |

There is no `tests/__init__.py`, and test basenames are unique. Do not create any file not
listed here without TL sign-off.

**Environment (A, first hour):** this machine's default `python` is a fresh 3.15.2 with no
NumPy or pytest. Create `nanogpt_numpy/.venv` and run `pip install -r requirements.txt`. If
NumPy/SciPy wheels are not available for 3.15, use `py install 3.13` (or 3.12) and build the
venv from that interpreter instead. Record the interpreter and package versions in PROGRESS.md.
B and C use the same venv.

---

## 5. `nanogpt/config.py` — frozen interface (B and C code against these exact names)

```python
import numpy as np, math
DTYPE          = np.float64
INT_DTYPE      = np.int64
INIT_STD       = 0.02
BIAS_INIT      = 0.0
LN_GAMMA_INIT  = 1.0
LN_BETA_INIT   = 0.0
LN_EPS         = 1e-5
MLP_HIDDEN_MULT = 4
LINEAR_HAS_BIAS  = True     # Q/K/V/O + MLP linears
LM_HEAD_HAS_BIAS = False    # tied head, no bias
N_HEAD         = 4          # default H; must divide d
GELU_C         = math.sqrt(2.0 / math.pi)
GELU_A         = 0.044715
FD_STEP        = 1e-5
REL_ERR_FLOOR  = 1e-8
GRADCHECK_TOL  = 1e-6
C3_SVD_REL_TOL = 1e-6       # singular values < C3_SVD_REL_TOL * s_max are zero
C4_PSD_TOL     = 1e-10      # eigenvalues >= -C4_PSD_TOL
ROUNDOFF_ATOL  = 1e-12      # TL: for "exact identity" checks (row sums, J·1, closed-form vs impl)
INIT_LOSS_TOL  = 0.1        # C2 (Phase 3)  — defined now, used later
KV_CACHE_ABS_TOL = 1e-10    # C8 (Phase 5)
FLOPS_REL_BAND = 0.25       # C9 (Phase 4)
DEFAULT_SEED   = 1337
CORPUS_SEED    = 1337

def rel_err(a, b) -> float:            # TL-1: Frobenius-norm form; equals §0 formula for scalars
def rel_err_elementwise(a, b) -> np.ndarray   # diagnostic only, same floor
def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator
def check_float64(*arrays, names=None) -> None  # raises TypeError on non-float64 (TL-7)

@dataclass(frozen=True)
class ModelConfig:              # validates d % H == 0, all fields > 0
    V: int; T: int = 64; d: int = 64; L: int = 2; H: int = N_HEAD
    head_dim (property) = d // H
def tiny_config(V: int) -> ModelConfig   # d=8, L=2, H=2, T=4
```

---

## 6. Engineer A: Phase 0 (config, corpus/tokenizer, gradcheck harness, scaffolding)

### 6.1 Milestones (in this order, and announce each one)
- **A1 (≤ 2 h):** venv + scaffolding + `config.py` exactly as in §5 + `tests/test_config.py`.
  *This unblocks B and C.* Announce "config frozen". Any later change to a name requires TL
  approval.
- **A2 (≤ ½ day):** `gradcheck.py` + `tests/test_gradcheck_harness.py` (the y=xW sanity proof).
  Announce "harness validated".
- **A3:** `tokenizer.py` (corpus resolution + synthetic generator + tokenizer) + tests, then
  the PROGRESS.md Phase 0 entry. This is the **Phase 0 gate**.
- **A4:** `tests/test_conventions.py` guard + README stub. At integration, run the full suite
  and write the PROGRESS.md Phase 1 entry.

### 6.2 `gradcheck.py` spec
```python
@dataclass
class TensorReport: name: str; shape: tuple; rel_err: float; max_elem_rel_err: float; max_abs_err: float
@dataclass
class GradcheckReport:
    tensors: dict[str, TensorReport]; tol: float
    passed: bool          # all(t.rel_err < tol)
    def worst(self) -> TensorReport
    def __str__(self)     # one line per tensor; used in assertion messages

def gradcheck(forward, backward, params, x, *, wrt_input=True, h=FD_STEP,
              tol=GRADCHECK_TOL, seed=DEFAULT_SEED) -> GradcheckReport
def numerical_jacobian(f, x, h=FD_STEP) -> np.ndarray   # central diff; f: R^n -> R^m (any shapes, flattened); J shape (m, n)
```
Algorithm:
1. `out, cache = forward(x, params)`. Scalarize: if `out` is 0-d, use `R = 1.0`. Otherwise
   `R = make_rng(seed).standard_normal(out.shape)`, and the objective is
   `L(x, params) = Σ out ⊙ R`.
2. Analytic: `d_x, d_params = backward(R, cache)`.
3. Validate structure **before** any numbers. `set(d_params) == set(params)` must hold, and each
   `d_params[k]` must have the shape of `params[k]` and be float64. Otherwise raise `ValueError`
   naming the key. If `wrt_input` is True and x is a float array, `d_x` must be non-None with
   x's shape. Integer `x` automatically means input check skipped.
4. Numeric: for every parameter element (`np.ndindex`), save the value, set `+h`, evaluate L, set
   `−h`, evaluate L, **restore the exact saved value**, and record `(L₊ − L₋)/(2h)`. Do the same
   on a **copy** of x for the input, under the report key `"x"`.
5. Report per tensor: `rel_err = config.rel_err(analytic, numeric)` (gated),
   `max_elem_rel_err` and `max_abs_err` (diagnostic only).
6. After the run, `params` must be bit-identical to the input. Assert this inside the harness.
- Use float64 only: call `check_float64` on x (if float) and on params. **Do not** build a tape,
  graph or autograd-like mechanism. The harness only perturbs and compares (PO scope risk).

### 6.3 `tokenizer.py` spec
```python
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GENERATOR_ID = "synthetic-v1"
SYNTHETIC_N_CHARS = 1_000_000
@dataclass(frozen=True)
class CorpusInfo: text: str; source: str ("bundled"|"synthetic"); generator: str|None; seed: int|None; n_chars: int; sha256: str
def generate_synthetic_corpus(seed=CORPUS_SEED, n_chars=SYNTHETIC_N_CHARS) -> str
def resolve_corpus(data_dir=DATA_DIR) -> CorpusInfo
class CharTokenizer:
    @classmethod from_text(cls, text) ; vocab_size: int ; stoi: dict ; itos: list
    encode(s: str) -> np.ndarray[int64]      # raises ValueError on unknown char
    decode(ids) -> str
def load_corpus_and_tokenizer(data_dir=DATA_DIR) -> tuple[CorpusInfo, CharTokenizer]
```
- **Resolution:** if `data_dir/corpus.txt` exists, read it as UTF-8, require it to be non-empty
  and < 5 MB, and return `source="bundled"`. Otherwise generate it. **The generator never writes
  `data/corpus.txt`**, so provenance can't silently flip on the next run. Never use any network
  call.
- **Generator `synthetic-v1`:** use a local `random.Random(seed)` (stdlib, stable across Python
  versions) and no global RNG. Use a fixed in-file word list (about 150–250 common lowercase
  English words). Sentences are 4–12 words, with the first word capitalized and an occasional
  ", " between words. Each sentence ends in ". ", "! " or "? ". Paragraphs of 3–7 sentences are
  joined with "\n\n". Generate until n_chars, then truncate to exactly n_chars. Keep it
  simple. It is a learnable, structured source, not a text-synthesis project.
- `vocab = sorted(set(text))`, so V is deterministic because the text is.

### 6.4 Acceptance criteria for A
- `test_config.py`: every §5 name exists with the exact value. `rel_err(0,0) == 0`.
  `rel_err(1.0, 1.0+1e-9) ≈ 5e-10`. The scalar form equals `|a−b|/max(|a|+|b|,1e-8)`.
  `ModelConfig(V=10, d=64, H=5)` raises. `tiny_config(V)` has d=8, L=2, H=2, T=4 and
  head_dim=4. `check_float64` raises on float32.
- `test_gradcheck_harness.py` (**Phase 0 exit criterion, before any real layer is checked**):
  - Correct `y = x @ W` (x (2,3,4), W (4,5)) gives `passed` and every `rel_err < 1e-8`.
  - The harness **fails** each of these: (i) dW·1.01, (ii) dW computed from the wrong
    operand order on square W (transposed), (iii) dx wrong while dW is correct, which shows
    the input is checked too.
  - A missing key or a wrong-shape key in `d_params` raises `ValueError`.
  - An unused bias param whose true gradient is exactly 0 reports `rel_err == 0.0`. This shows
    the floor works (it is the stand-in for the T=1 case).
  - With integer x, `d_x=None` is accepted and the input is skipped.
  - `params` is bit-identical after the run.
  - `numerical_jacobian` of `x ↦ A x` equals A (rel_err < 1e-9).
- `test_tokenizer.py`: two independent `generate_synthetic_corpus(CORPUS_SEED)` calls are
  identical, and a different seed differs. `decode(encode(text)) == text` on the **full**
  corpus. V is identical across two fresh `load_corpus_and_tokenizer()` calls. Use `tmp_path`
  to test both branches: bundled (write a small `corpus.txt`) and synthetic (empty dir). An
  unknown char raises. `encode` returns int64. A test monkeypatches `socket.socket` to raise
  and confirms `resolve_corpus` still works (no network).
- `test_conventions.py` (P2 guard): AST-scan `nanogpt/*.py` **except `config.py`**. Fail on
  any float literal in {1e-5, 1e-8, 1e-6, 1e-10, 0.02, 0.044715}, any `np.float32`/`float32`/
  `float16` attribute or string, and any import of `torch`, `jax`, `tensorflow` or `autograd`.
  Allowed: `0.0`, `0.5`, `1.0`, `2.0`, `3.0` (needed by GELU/LN algebra, and ε=0 for the C3
  instance). **It also fails on any `TODO(SPRINT1-CONFIG)` / `TODO(SPRINT1-OPS)` marker left in
  `nanogpt/`**, which is how the fallbacks in §10 are forced back out.
- PROGRESS.md Phase 0 entry (template §12) includes: interpreter and package versions, corpus
  source = synthetic-v1, seed 1337, n_chars, SHA-256, V, and the TL-1/R4/R5/R6 interpretations.
- Each of A's test files runs in < 10 s.

---

## 7. Engineer B: `nanogpt/ops.py` (linear, embedding, GELU, softmax) + C4

### 7.1 API
```python
def logsumexp(z, axis=-1, keepdims=False) -> np.ndarray      # m + log Σ exp(z − m), m = max (B1)
def softmax_forward(z, params=None) -> (p, cache)             # (B1)
def softmax_backward(dp, cache) -> (dz, {})
def gelu_forward(x, params=None) -> (y, cache)
def gelu_backward(dy, cache) -> (dx, {})
class Linear:     __init__(self, prefix, d_in, d_out, bias=LINEAR_HAS_BIAS); init_params(rng); forward; backward
class Embedding:  __init__(self, prefix, num_embeddings, d);                init_params(rng); forward; backward
def accumulate_grads(grads: dict, d_params: dict) -> dict     # copy on first insert, += after; returns grads
```

### 7.2 Formulas (do not re-derive. Restated here.)
- **Linear:** W (d_in, d_out), b (d_out,). `out = x @ W + b` for x (..., d_in).
  Backward, with x2 = x.reshape(−1, d_in) and g2 = d_out.reshape(−1, d_out):
  `dx = d_out @ W.T`, `dW = x2.T @ g2`, `db = g2.sum(0)`. With `bias=False` there is no `.b`
  key in params or in d_params. Keep `Linear` as the single matmul site, but **do not add FLOP
  instrumentation** (Phase 4).
- **Embedding:** W (num, d). `out = W[idx]` → (..., d). Validate that idx is an integer dtype
  and that `0 ≤ idx < num` (raise otherwise). Backward: `dW = np.zeros_like(W);
  np.add.at(dW, idx, d_out)`. Return `(None, {f"{prefix}.W": dW})`. Positional embeddings use
  the same class with `idx = np.arange(T)`.
- **GELU (tanh):** with c = `GELU_C` = √(2/π), a = `GELU_A` = 0.044715,
  u = c(x + a x³), t = tanh(u), `y = 0.5·x·(1 + t)`, and
  `dy/dx = 0.5(1 + t) + 0.5·x·(1 − t²)·c·(1 + 3a x²)`. Cache x and t.
- **Softmax** (last axis): `m = max(z, -1, keepdims)` and `e = exp(z − m)`, giving `p = e / Σe`.
  Rows with −∞ entries (at least one finite) give exact 0s and no NaN (TL-5). Cache p.
  Backward (TL-4): `dz = p ⊙ (dp − Σ(dp ⊙ p, -1, keepdims))`.
- **Init:** W ~ N(0, INIT_STD²), b = BIAS_INIT, from the passed `rng`, as float64.
- All constants are imported from `nanogpt.config`. None are defined locally (the guard test
  enforces this).

### 7.3 Acceptance criteria for B
`tests/test_ops.py` (gradcheck with `gradcheck.gradcheck`, `rel_err < GRADCHECK_TOL` on every
tensor, tiny shapes, explicit seeds, and non-init params per TL-6):
- **Linear:** x (2,3,4) → d_out 5. Check x, W and b. Also `bias=False`: no `.b` key, and the
  check passes.
- **Embedding:** num=5, d=3, `idx = [[0,2,2],[1,0,2]]` (**duplicates required**), check W only.
  *Test-the-test:* a local variant using `dW[idx] += d_out` **fails** the gradcheck on this
  duplicate idx and **passes** on an all-unique idx, which shows why duplicates are mandatory.
  Out-of-range and float idx raise.
- **GELU:** x includes 0, ±1e-3, ±1, ±10, plus random (2,3,4). Forward matches the closed form
  at x=0 → 0, and x=10 → ≈10 within 1e-12.
- **Softmax:** gradcheck on z (2,3,6).
- **Dtype:** a float32 input raises `TypeError`.
- **accumulate_grads:** the same key accumulated twice gives the sum. The first-inserted array
  is a copy (mutating the source afterwards does not change grads). The accumulator array's
  identity is preserved (`+=`, not rebinding).
- **LSE smoke (P1):** z = 1e3·N(0,1) of shape (4,7). p is finite, rows sum to 1 within
  ROUNDOFF_ATOL, and `logsumexp` is finite. Inside `np.errstate(over="ignore", invalid="ignore")`,
  a naive `exp(z)/Σexp(z)` gives non-finite values (test-the-test). Formal sign-off is in Phase 6.
- **Masked softmax (TL-5):** a row with −∞ entries gives exact zeros there, no NaN, and a
  backward that is finite and 0 at masked positions.

`tests/test_softmax_properties.py`, **C4**, d = 6 and 4 seeds:
- **J_impl:** apply `softmax_backward` to each basis vector to build J. It must match
  `diag(p) − ppᵀ` within `ROUNDOFF_ATOL` and match `numerical_jacobian(softmax)` with
  `rel_err < GRADCHECK_TOL`.
- **PSD:** `eigvalsh((J+Jᵀ)/2).min() ≥ −C4_PSD_TOL`, with `‖J − Jᵀ‖_max < ROUNDOFF_ATOL`.
- **Rows sum to 0:** `‖J_impl @ 1‖_∞ < ROUNDOFF_ATOL`.
- **Shift-invariance:** `softmax(z + c·1) == softmax(z)` within ROUNDOFF_ATOL for
  c ∈ {−5.0, 3.7, 1e3}. At least one nonzero c is required, and 1e3 also exercises stability.
- **Rank note (spec C4):** with k finite entries (others −∞), `rank(J) = k − 1` (SVD,
  threshold `C3_SVD_REL_TOL·s_max`). For dense p the rank is d−1.
- **Test-the-test:** (i) a broken backward `dz = p ⊙ dp` (dropped term) gives a J_impl whose
  rows do **not** sum to 0 and whose gradcheck **fails**. (ii) A non-shift-invariant
  "softmax" `e^z/(Σe^z + 1)` **fails** the shift check.

Every B test file runs in < 10 s. B returns a results block (§12) for PROGRESS.md.

---

## 8. Engineer C: `nanogpt/layernorm.py` + `nanogpt/loss.py` + C3

### 8.1 API
```python
class LayerNorm:
    __init__(self, prefix, d, eps=LN_EPS, affine=True); init_params(rng)  # γ=1, β=0 (no rng use needed but accept it)
    forward(x, params) -> (out, cache); backward(d_out, cache) -> (dx, d_params)
def c3_test_instance(d) -> LayerNorm          # LayerNorm("ln_c3", d, eps=0.0, affine=False). TEST CONSTRUCTION ONLY.
def cross_entropy_forward(logits, targets) -> (loss: float, cache)        # in loss.py
def cross_entropy_backward(d_loss, cache) -> (d_logits, {})
```
With `affine=False`, `params` is ignored (`{}` or `None` is fine), `out = ŷ`, and `d_params = {}`.

### 8.2 Formulas (restated: production path and C3 path share one backward, see TL-2)
- **Forward** (last axis, population variance ddof=0):
  `μ = mean(x)`, `σ² = mean((x−μ)²)`, `s = √(σ² + ε)`, `ŷ = (x−μ)/s`, `out = γ⊙ŷ + β`.
  Do not use `np.var(..., ddof=1)`. With ε=0, raise `ValueError` if any σ == 0 (TL-8).
  Cache ŷ, `rstd = 1/s`, γ.
- **Backward:** `dβ = Σ_lead d_out`, `dγ = Σ_lead d_out⊙ŷ`, `g = d_out⊙γ` (or `g = d_out` if
  no affine), `dx = rstd·(g − mean(g) − ŷ⊙mean(g⊙ŷ))`.
- **Closed-form Jacobian** (any ε ≥ 0, single row): `J = (1/s)(I − 11ᵀ/d − ŷŷᵀ/d)`, which is
  symmetric. At ε=0, s=σ, giving J·1 = 0 and J·x = 0, so rank d−2. At ε>0 the ŷ-direction
  eigenvalue is `(1/s)·ε/(σ²+ε)`, so rank d−1 (TL-3).
- **Cross-entropy** (logits (..., V), integer targets (...) with matching leading shape,
  validated `0 ≤ t < V`): `N = targets.size` (R4).
  `lse = ops.logsumexp(z)`, `loss = mean(lse − take_along_axis(z, t))`.
  Backward: `p = exp(z − lse[...,None])`, `d_logits = d_loss·(p − onehot(t)) / N`. Build onehot
  by subtracting 1 at the target index on a copy of p. Do not materialize a dense one-hot from
  `np.eye`.

### 8.3 Acceptance criteria for C
`tests/test_layernorm.py` (gradcheck, `rel_err < GRADCHECK_TOL`):
- Production ε=LN_EPS, affine, x (2,3,6), **γ = 1+0.5·N(0,1), β ~ N(0,1)** (TL-6). Check x,
  γ and β.
- The ε=0 no-affine instance on non-constant x (2,3,6): check x.
- Population variance: on a hand-computable row, e.g. x=[1,2,3,4] with ε=0, ŷ equals
  `(x−2.5)/√1.25` within ROUNDOFF_ATOL. This catches ddof=1.
- ε=0 with a constant row raises `ValueError`. The production path with a constant row gives
  ŷ = 0 and no NaN.
- **Test-the-test:** a backward missing the `ŷ⊙mean(g⊙ŷ)` term **fails** gradcheck. So does a
  backward missing `⊙γ`. The second one only fails because γ ≠ 1, which is why TL-6 exists.

`tests/test_layernorm_rank.py`, **C3**, d = 8, seeded:
- **ε=0 instance:** x non-constant (`3.0 + N(0,1)`). Build J_impl from `backward` on each basis
  vector. It must match the closed form within ROUNDOFF_ATOL·scale and match
  `numerical_jacobian(forward)` with `rel_err < GRADCHECK_TOL`. `‖J·1‖_∞` and `‖J·x‖_∞` are
  < ROUNDOFF_ATOL (scaled by ‖x‖). The SVD rank with threshold `C3_SVD_REL_TOL·s_max` is
  exactly **d−2**.
- **Production instance** (ε=LN_EPS, γ=1, β=0, the normalization map per the PO): x standardized
  to **population variance exactly 1** (TL-3 precondition). The rank is exactly **d−1**. J·1≈0
  still holds, and J·x ≠ 0 (it is about ε/s³·(x−μ)). The test docstring says this is expected
  per spec §4, not a bug.
- **Non-vacuity:** assert the two ranks differ. Also repeat the ε=0 check for d ∈ {3, 5, 8},
  since d=3 is the minimum.
- **TL-3 boundary test:** the same production instance with x scaled so σ² = 1e3 reads rank
  d−2 at the same threshold. The ratio is `ε/(σ²+ε)` ≈ 1e-8, below 1e-6. The test asserts the
  measured `s[-2]/s[0]` (second-smallest singular value over σ_max, s sorted descending) equals
  `ε/(σ²+ε)` within 1e-3 relative, at both σ²=1 and σ²=1e3. This turns the
  "don't touch ε or the threshold" rule into an executable explanation.
- **Test-the-test:** with the broken backward (dropped ŷ term), J_impl on the ε=0 instance has
  rank d−1 ≠ d−2, and its closed-form comparison fails.

`tests/test_loss.py`:
- Gradcheck on logits (2,3,7) with targets that include repeats.
- **Closed-form anchors:** all-zero logits give loss = ln V exactly (within ROUNDOFF_ATOL) for
  V=7. **Integration anchor:** the same holds with V from `load_corpus_and_tokenizer()`, which
  ties the loss to the resolved corpus and is a cheap precursor of C2. The target logit 50
  above all others gives loss < 1e-20.
- Each row of `d_logits·N/d_loss` sums to 0 within ROUNDOFF_ATOL. N is `targets.size`: a (2,3)
  batch gives the same per-position gradient as the flattened (6,) batch (R4 consistency).
- **LSE smoke (P1):** logits = 1e3·N(0,1) give finite loss and finite gradient.
- Out-of-range targets and float logits of the wrong dtype raise.

Every C test file runs in < 10 s. C returns a results block (§12) for PROGRESS.md.

---

## 9. Explicitly out of scope for Sprint 1 (do not build)

`attention.py`, `mlp.py`, `block.py`, `model.py`, `param_count.py` (C1), `flops.py` / any FLOP
hooks (C9), `optim.py` (AdamW anchor), `kv_cache.py` / `sampling.py` / `checkpoint.py` (C8),
`train.py`, `generate.py`, `tests/test_gradients.py` (C6), C2/C5/C7 tests, the T=1 and
shifted-mask tests (Phase 2), the formal `test_numerical_stability.py` (Phase 6), GPT-2 weights
(Phase 7), the full README (Phase 8). Sprint 1 only fixes the contract, constants and
`accumulate_grads` they will rely on.

---

## 10. Dependency protocol (what B and C do if A or B isn't ready yet)

```
A1 config.py ──┬──> B (all), C (all)
               └──> A2 gradcheck.py ──> B/C gradcheck tests
A3 corpus+tokenizer ──> C's integration ln V anchor ──> Phase 1 sign-off
B1 logsumexp + softmax_forward ──> C loss.py
```
- **config.py not landed yet:** at the top of `ops.py` / `layernorm.py` / `loss.py`, add one
  clearly fenced block:
  ```python
  # TODO(SPRINT1-CONFIG): delete this block; replace with
  #   from nanogpt.config import DTYPE, INIT_STD, LN_EPS, GELU_C, GELU_A, ...
  DTYPE = np.float64; INIT_STD = 0.02; ...   # names IDENTICAL to config.py §5
  ```
  Use the **exact names** from §5 so the swap is a one-line import. The guard test fails while
  any `TODO(SPRINT1-CONFIG)` marker or forbidden literal remains, so this cannot ship. **Never
  create or edit `nanogpt/config.py` yourself.** Ask A to add anything that's missing.
- **gradcheck.py not landed yet:** write the forward/backward and the non-gradcheck tests first
  (closed forms, C3/C4 closed-form vs. J_impl, invariants). Write the gradcheck tests against
  the §6.2 API. Any throwaway FD loop stays in your scratchpad and is never committed.
- **ops.logsumexp not landed yet (C only):** use a private `_logsumexp` in `loss.py` under
  `# TODO(SPRINT1-OPS): replace with ops.logsumexp`. The guard test removes it at integration.
- Interface changes to §3/§5/§6.2/§7.1/§8.1 require TL approval. Changes to the internals do not.

---

## 11. Timeline (3–4 engineering days)

| Day | A | B | C |
|---|---|---|---|
| 1 AM | venv, scaffold, **A1 config** | **B1** logsumexp + softmax fwd/bwd | LayerNorm fwd/bwd (production + ε=0) |
| 1 PM | **A2 gradcheck** + y=xW validation | GELU, Linear | closed-form J, C3 rank tests (no harness needed) |
| 2 | **A3 corpus + tokenizer**, Phase 0 PROGRESS entry (**gate**) | Embedding (`np.add.at`), accumulate_grads, gradchecks | loss.py, gradchecks, LN gradchecks |
| 3 | guard test, README stub | C4 suite + test-the-test | C3 boundary + test-the-test, loss anchors |
| 3–4 | **Integration:** remove fallbacks, full `pytest tests/`, Phase 1 PROGRESS entry | fix-ups | fix-ups |

TL reviews: A1 (interface freeze), A2 (harness proof), and each module before it merges.

---

## 12. Integration and sign-off checklist (TL runs this)

1. `grep -rn "TODO(SPRINT1-" nanogpt/` finds nothing. `tests/test_conventions.py` is green.
2. `python -m pytest tests/ -q --durations=10` is all green, takes < 60 s total, and no
   file takes > 10 s.
3. Two consecutive runs give identical results (determinism). The corpus SHA-256 matches
   PROGRESS.md.
4. Phase 0 exit: config is the single source, the harness is proven on y=xW, and the corpus is
   resolved and round-trips. **The Phase 0 entry is dated before Phase 1 sign-off.**
5. Phase 1 exit: every primitive passes gradcheck < 1e-6 (the table below is pasted into
   PROGRESS.md). C3 gives d−2 / d−1 plus non-vacuity. C4 passes on all three parts.
   Test-the-test is demonstrated for embedding duplicates, the LN dropped term, the LN missing
   γ, the softmax dropped term, and shift-invariance.
6. There is no Phase 2+ file in the repo.

**Results block each engineer hands to A for PROGRESS.md:**
```
Module: <file>   Owner: <A|B|C>
Gradcheck (norm-wise rel_err, gated < 1e-6):  <tensor>: <value>  ...
Diagnostics (max elementwise rel_err):        <tensor>: <value>  ...
Invariant checks: <C3/C4 items with measured values, e.g. "rank eps=0: 6 (d=8)">
Test-the-test: <variant> -> FAILED as expected (rel_err=<value>)
Deviations from SPRINT_1_PLAN.md: <none | what + why>
```

**PROGRESS.md entry template (A):**
```
## Phase <n> — <name>  [DONE <date>]
Passed: <tests / checks, with key numbers>
Corpus: <bundled|synthetic-v1>, seed=<>, n_chars=<>, sha256=<>, V=<>
Env: python <>, numpy <>, scipy <>, pytest <>
Interpretations/deviations: TL-1 norm-wise rel_err; R4 CE normalizer N=targets.size;
  R5 H=4 default (tiny H=2); R6 per-owner gradcheck test files, test_gradients.py reserved for C6; ...
Next: Phase <n+1> — <name>
```
