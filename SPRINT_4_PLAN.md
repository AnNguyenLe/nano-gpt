# Sprint 4 — Technical Plan: Phase 4 (Optimizer, FLOP Accounting & Training Loop)

**Author:** Tech Lead (Opus 5.5). **Inputs:** PM backlog T0..T9, PO priorities (P0/P1/P2),
`nanoGPT_build_requirements.md` (the spec: §1.2, C9, §3), `nanoGPT_execution_plan.md` §Phase 4 + §8
(AdamW update equations, corrected FLOPs formula), and `PROGRESS.md` (Phase 0–3 state, HEAD
`dfbd373`, 363 tests green in ~23 s).

**Scope: Phase 4 only.** `nanogpt/optim.py` (AdamW), `nanogpt/flops.py` (instrumented FLOP counter,
C9), `train.py` (training loop), and their tests. KV-cache (C8), sampling, checkpoint save/load,
generation, the large-logit stability pass and GPT-2 weights are out of scope (§13).

Every number quoted as "measured" below comes from a TL throwaway prototype run against the
committed Sprint 1–3 modules (scratchpad only, not committed, not a deliverable). Engineers must
reproduce them in their own tests; nobody copies them in as expected values unless this plan says
the value is a hand-derived literal (the §3.2 anchor table).

**The non-negotiable framing for this sprint.** A decreasing training loss is **never** evidence of
correctness — not for the model and not for the optimizer. AdamW is accepted only on the §5
closed-form anchors; the model is already accepted on C1/C2/C5/C6/C7. `train.py` is a budget and
integration check. No test, results block or sign-off line may assert or cite "loss went down".

---

## 1. Sprint 4 Definition of Done

1. **TL-24 decay-group ruling** is recorded here (§2.2) and in PROGRESS.md, and `optim.py`
   implements it as an explicit, exhaustive name allowlist — not inferred from `ndim`.
2. **AdamW** (`nanogpt/optim.py`) implements exactly:
   m_t = β₁m_{t−1} + (1−β₁)g_t; v_t = β₂v_{t−1} + (1−β₂)g_t²; m̂_t = m_t/(1−β₁ᵗ); v̂_t = v_t/(1−β₂ᵗ),
   t starting at 1; θ_t = θ_{t−1} − η(m̂_t/(√v̂_t + ε) + λθ_{t−1}), with λ applied only to the decay
   group and **never added to the gradient**.
3. **Single optimizer state for the tied parameter:** exactly one (m, v) pair for `wte.W`, even when
   the optimizer is built from `named_parameters()` (which also exposes `lm_head.W` as the same
   object). Proven by a dedicated test with a discriminating adversary.
4. **Closed-form AdamW anchor** (`tests/test_adamw.py`): the first step matches
   Δθ = −η(g/(|g|+ε) + λθ) (≈ −η·sign(g) − ηλθ) at `ROUNDOFF_ATOL`; a 3-step quadratic trajectory
   matches the §3.2 literal table (θ, m, v, m̂, v̂ at every step) at `ROUNDOFF_ATOL`; four buggy
   variants (off-by-one t, no bias correction, coupled L2 decay, decay on an excluded tensor) each
   fail.
5. **C9** (`tests/test_flops.py`): an instrumented counter at the single matmul chokepoint
   (TL-27) gives measured forward and training FLOPs/token; they (a) equal the exact matmul
   inventory (TL-29) to the integer, and (b) lie within ±25% (`FLOPS_REL_BAND`) of
   fwd ≈ 2N + 4LTd and train ≈ 6N + 12LTd on tiny, default, an in-range long-context config and
   max. The naive 2N / 6N formula is shown to **fail** the band at an attention-heavy config.
6. **`train.py`** runs end to end on the real corpus (`python train.py`), steps AdamW, logs loss;
   a tiny-config pytest (`tests/test_train.py`) exercises the real loop in < 5 s.
7. **Real run (TL, outside pytest):** one full run at the default config completes in ≤ 30 min CPU;
   profile taken right after the first working loop; wall-clock, step count, s/step and loss log
   recorded in PROGRESS.md (loss shown for the record only).
8. **Commit discipline (P0 hard gate):** every Sprint 4 commit is green when its own tree is
   checked out alone; the TL verifies this per commit (§11 item 6), not by self-report.
9. Suite green, < 60 s total, every file < 10 s (new files reported explicitly), deterministic over
   two runs; PROGRESS.md Phase 4 entry appended with "Next: Phase 5".

---

## 2. Reconciliation log and Tech Lead rulings

### 2.1 PM backlog vs. PO priorities vs. spec

| # | Topic | Conflict / ambiguity | Ruling |
|---|---|---|---|
| R22 | Checkpointing | Execution plan Phase 4 prose: train.py "checkpoints via nanogpt/checkpoint.py". PM T5 and PO scope: checkpointing is Phase 5. | **Out of scope.** No `checkpoint.py`, no save hook (not even a no-op) in `train.py`. Phase 5 adds save/load together with its round-trip test, where C8 needs it. AdamW state is kept in plain dicts of float64 arrays so it is serializable later. |
| R23 | Test file name | Computed task suggests `tests/test_optim.py`; execution-plan layout and PM T2 say `tests/test_adamw.py`. | **`tests/test_adamw.py`** (repo layout §3 of the execution plan wins). |
| R24 | `train.py` location | Computed task says `nanogpt/train.py`; execution-plan layout puts `train.py` at the repo root next to `generate.py`. | **Repo root `train.py`.** The root `conftest.py` already puts the repo root on `sys.path`, so `import train` works in tests. It is not scanned by the conventions guard (that scans `nanogpt/*.py`), but it still imports all numeric constants from `nanogpt.config`. |
| R25 | C9 configs | Computed task: "tiny and default". PM T4: a config where the attention term is non-trivial and the naive formula fails. | **Both, plus more** (TL-30): tiny, default, in-range long-context (d=64, T=128, L=4), max (d=128, T=128, L=4), and one out-of-range attention-heavy discriminator (d=16, T=128, L=2). |
| R26 | Engineer split | Suggested split A = optim, B = flops, C = train. | **Kept.** B also owns the TL-27 matmul-chokepoint refactor of `ops.py` / `attention.py` / `model.py`, which B authored in Sprints 1–3. |
| R27 | P2 items | README section, loss averaging, LR schedule, val split. | **Not in this sprint's DoD.** Constant LR, no schedule, no val split, no README edit (Phase 8). The PROGRESS.md entry documents the reproduce commands (that covers PO P2 docs at no cost). |

### 2.2 Technical rulings (TL-24 .. TL-31)

**TL-24. AdamW decay groups (PM T0; for PO ratification).** Membership is decided by an explicit,
exhaustive name allowlist over `GPT.param_keys()`, never by `ndim` (LN γ is a 1-D weight; `wpe.W`
is a 2-D weight matrix that we do not decay):

| Group | Keys (regex over canonical names) | Count (L blocks) | Elements, default cfg |
|---|---|---|---|
| **decay** (λ applied) | `^h\.\d+\.attn\.(q\|k\|v\|o)\.W$`, `^h\.\d+\.mlp\.(fc\|proj)\.W$` | 6L | L·12d² = 98,304 |
| **no decay** | `^(wte\|wpe)\.W$`, `\.b$` (all Q/K/V/O/fc/proj biases), `\.(gamma\|beta)$` (ln1, ln2, ln_f) | 4 + 10L | Vd + Td + 13Ld + 2d = 9,408 |

The two counts sum to 4 + 16L tensors and N = 107,712 elements at the default config (C1). Any
other name raises `ValueError` (an unknown key is a bug, not a default). The tied `wte.W` is
no-decay, as PM T0 and PO P0 both require. `wpe.W` is also no-decay: the spec says "weight
matrices", and we read that as the linear maps, not the lookup tables. This is the minGPT
convention. Karpathy's nanoGPT decays every ≥2-D tensor, including embeddings, which is why this
needs **PO ratification**. The alias `lm_head.W` is never classified on its own. It is removed by
canonicalization (TL-25) first.

**TL-25. One state per unique array; canonical names; in-place updates.**
- `AdamW(params)` accepts `params` from `init_params()` (only `wte.W`) **or** from
  `named_parameters()` (`wte.W` plus `lm_head.W` as the same object). It canonicalizes by **object
  identity**. Keys that are the same object collapse to one entry, named by the first key that is not
  `lm_head.W`. Two keys whose arrays overlap without being identical (`np.shares_memory`, for
  example a `.T` view) raise `ValueError`, the same contract as `param_count.live_param_count`.
  Non-float64 or non-C-contiguous params also raise.
- `opt.state` has exactly one `{"m": ..., "v": ...}` per canonical name. So
  `len(opt.state) == len(model.param_keys())`, and there is no `"lm_head.W"` key anywhere in the
  optimizer.
- `opt.step(grads)` requires `set(grads) == set(opt.params)`. A missing key, an extra key, or the
  alias `lm_head.W` raises `ValueError`. This matches `model.backward`, which emits exactly
  `param_keys()`, with C7's accumulated tied gradient under `wte.W`.
- θ, m and v are updated **in place**, so the array identity seen by the model and by
  `named_parameters()` (TL-17) never changes and C-contiguity is kept. `grads` is never mutated.

**TL-26. Step counter.** It is a single per-optimizer integer `opt.t`, starting at 0 and
incremented at the **start** of `step()`, so the first update uses t = 1. Bias corrections are
`1 − β₁**t` and `1 − β₂**t` in float64. The decay term uses θ_{t−1}, the value before this
step's update.

**TL-27. Single matmul chokepoint: `ops.matmul` (non-invasive instrumentation).**
- The counter must see every matmul.
- Prototype finding: the "ndarray subclass + `__array_ufunc__`" trick is **unsound** here.
  - `LayerNorm.forward` returns `np.asarray(out)`, and `cross_entropy_forward` calls
    `np.asarray(logits)`. Both strip the subclass.
  - So `g_out = d_logits.T @ y` (plain @ plain) is silently uncounted.
  - Measured: 7,040 FLOPs/token missing at the default config, so backward/forward = 1.970
    instead of 2. The same miss would hide any future plain-on-plain matmul.
- So we use one explicit dispatch, following the `ops.accumulate_grads` precedent (C7 monkeypatches
  a module attribute):
  ```python
  # nanogpt/ops.py
  def matmul(a, b, *, kind="dense"):
      """Single chokepoint for every production matmul. kind ∈ {"dense", "attention"}."""
      return a @ b
  ```
- Every production `@` is rewritten to `ops.matmul(...)`, called as a **module attribute** (never
  `from nanogpt.ops import matmul`). There are 12 sites today:
  - `ops.py`: 3, `Linear` forward and backward.
  - `attention.py`: 6, all with `kind="attention"`: S = QKᵀ, Y = PV, dP, dV, dQ, dK.
  - `model.py`: 3, the tied head forward, `g_out` and `dy`.
- The q/k/v/o projections go through `ops.Linear`, so they count as `dense`.
- Results are bit-identical (`a @ b` and `np.matmul` are the same ufunc), and the existing 363
  tests re-run unchanged as the regression proof.
- **No signature of any verified module changes.**
- Docstrings that show `@` in formulas stay as they are.
- `attention_reference.py` (the oracle) is **excluded**: it is not production code, its
  AST-guarded style must not change, and it never runs inside the counter.

**TL-28. FLOPs of one call.** For `a` with shape (..., m, k) and `b` with shape (..., k, n), the
count is 2·prod(broadcast(batch_a, batch_b))·m·k·n. A 1-D operand follows NumPy's promotion rules
(m = 1 or n = 1). Mismatched k raises. Additions of bias, softmax, LN, GELU, the optimizer and
embedding gathers are **not** counted. C9 counts matmuls only (2·m·n·k), as the spec says.

**TL-29. Exact matmul inventory (the counter's own closed-form anchor).**
- With sequence length = cfg.T and the full T×T score matrix (masking happens after the matmul,
  so nothing is skipped), the exact per-token counts are:
  - forward dense = 24Ld² + 2Vd
  - forward attention = 4LTd
  - backward = 2 × forward for each kind
  - training = 3 × forward, exactly
- The difference from 2N + 4LTd is 2Td + 26Ld + 4d. That is wpe (a lookup, not a matmul) plus
  biases and LN (not matmuls). This is why the spec's band is ±25% and not an exact match.
- The test asserts **integer equality** of the measured count with this inventory.
- That equality catches an uncounted site, such as the TL-27 subclass miss, which a ±25% band would
  hide.

**TL-30. C9 configs and expected numbers (TL prototype, exact chokepoint counts).**

| Config (V=55) | measured fwd/tok | 2N+4LTd | ratio | naive 2N ratio | attention share (formula) |
|---|---|---|---|---|---|
| tiny d=8, T=4, L=2, H=2 | 4,208 | 4,720 | 0.8915 | 0.9427 | 5.4% |
| default d=64, T=64, L=2, H=4 | 236,416 | 248,192 | 0.9526 | 1.0974 | 13.2% |
| long d=64, T=128, L=4, H=4 (in range) | 531,328 | 554,624 | 0.9580 | **1.2545 (fails band)** | 23.6% |
| max d=128, T=128, L=4, H=4 | 1,849,088 | 1,895,680 | 0.9754 | 1.1320 | 13.8% |
| discriminator d=16, T=128, L=2, H=2 (out of range) | 30,432 | 35,424 | 0.8591 | **1.5983 (fails band)** | 46.3% |

- Training ratios equal the forward ratios exactly, because training is 3× forward exactly (TL-29).
- Interpretation for PO P1: at default and max, the attention term is 13.2% and 13.8% of the
  corrected total, inside the spec's 13–20%. At the in-range long-context config it reaches 23.6%.
- The naive formula **passes** the band at tiny, default and max. So C9 is discriminating only
  because of (a) the integer-exact inventory and (b) the two attention-heavy configs.
- The in-range failure of the naive formula (1.2545 > 1.25) is narrow but fully deterministic,
  since it is integer arithmetic. The out-of-range discriminator gives the robust margin.
- Both are asserted.

**TL-31. `train.py` contract and budget.**
- Defaults: the project default `ModelConfig(V, T=64, d=64, L=2, H=4)`, batch 16, constant
  lr = 1e-3, β = `ADAMW_BETAS`, λ = `ADAMW_WEIGHT_DECAY`, 4000 steps, logging every 100 steps,
  seed `DEFAULT_SEED`.
- Prototype timing: one forward+backward at the default config and B=16 takes 0.21 s (12-core
  CPU, numpy 2.5.5). So 4000 steps take about 14–15 min, about half the budget.
- For comparison: d=128, T=128, L=4 at B=16 takes 4.2 s/step, which allows only about 350 steps in
  25 min. So the default config is the right real-run choice, and the trimming lever (PM T7) is
  batch size or steps, not exotic optimizations.
- `main()` refuses configs outside the spec ranges (L 2–4, d 64–128, T 64–128) unless
  `--allow-out-of-range` is passed.
- The step-0 loss is printed next to ln V. This is a consistency print, not a test.

---

## 3. Frozen interfaces and formulas (all engineers code to this)

### 3.1 Conventions (import from `nanogpt.config`, never redefine)

Existing: `DTYPE`, `INT_DTYPE`, `DEFAULT_SEED`, `make_rng`, `check_float64`, `ROUNDOFF_ATOL = 1e-12`,
`FLOPS_REL_BAND = 0.25`, `ModelConfig`, `tiny_config`. **New (Engineer A, milestone A1, additive
only):**
```python
ADAMW_BETAS = (0.9, 0.99)      # nanoGPT char-level default
ADAMW_EPS = 1e-8               # NOTE: 1e-8 is a FORBIDDEN_FLOATS literal outside config.py
ADAMW_WEIGHT_DECAY = 0.1
```
`optim.py` therefore **must** import `ADAMW_EPS` (a local `1e-8` fails `test_conventions.py`). No
existing constant changes value. `train.py` keeps its run defaults (lr, steps, batch) as
module-level constants in `train.py` itself, since they are not §0 conventions.

### 3.2 `nanogpt/optim.py` (Engineer A)

```python
DECAY_PATTERNS: tuple[re.Pattern, ...]      # TL-24 decay allowlist
NO_DECAY_PATTERNS: tuple[re.Pattern, ...]   # TL-24 no-decay allowlist
def is_decay_param(name: str) -> bool       # True/False per TL-24; unknown name (incl. "lm_head.W") -> ValueError
def canonicalize_params(params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]   # TL-25
class AdamW:
    def __init__(self, params, *, lr: float, betas=ADAMW_BETAS, eps=ADAMW_EPS,
                 weight_decay=ADAMW_WEIGHT_DECAY): ...
    params: dict[str, np.ndarray]           # canonical name -> the SAME array objects passed in
    decay: dict[str, bool]                  # canonical name -> is_decay_param(name)
    state: dict[str, dict[str, np.ndarray]] # canonical name -> {"m": zeros_like, "v": zeros_like}
    t: int                                  # 0 before the first step (TL-26)
    lr: float                               # plain attribute (a later schedule may set it)
    def step(self, grads: Mapping[str, np.ndarray]) -> None
```
`step`, per canonical name with g = grads[name] (validated: key set, shape, float64):
```
t += 1 (once per step, before the loop)
m = β₁·m + (1−β₁)·g            (in place)
v = β₂·v + (1−β₂)·g²           (in place)
m̂ = m / (1−β₁ᵗ);  v̂ = v / (1−β₂ᵗ)
u = m̂ / (√v̂ + ε)
if decay[name]: u = u + λ·θ     (θ = θ_{t−1}; decoupled: λθ never enters m or v)
θ -= η·u                        (in place)
```
Constructor validation: lr > 0, 0 ≤ β₁, β₂ < 1, eps > 0, weight_decay ≥ 0, else ValueError.
No `zero_grad` (grads are fresh dicts each backward). No `state_dict`/`load_state_dict` (R22).

**Hand-computed anchor table (literal expected values; 60-digit Decimal, TL).** Objective
f(θ) = ½ Σᵢ aᵢθᵢ², so gᵢ = aᵢθᵢ (the test computes g from the current θ). θ₀ = [1.0, −0.5, 2.0],
a = [2.0, 4.0, 0.5], η = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 1e-8. Decay case λ = 0.01 (θ is named as a
decay-group key, e.g. `"h.0.mlp.fc.W"`):

| t | i | g | m | v | m̂ | v̂ | θ_t |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2 | 0.2 | 0.004 | 2 | 4 | 0.89900000049999995 |
| 1 | 1 | −2 | −0.2 | 0.004 | −2 | 4 | −0.39950000050000001 |
| 1 | 2 | 1 | 0.1 | 0.001 | 1 | 1 | 1.898000001 |
| 2 | 0 | 1.7980000009999999 | 0.35980000010000002 | 0.0072288040035960004 | 1.8936842110526315 | 3.6162101068514256 | 0.79851902716852097 |
| 2 | 1 | −1.598000002 | −0.33980000020000001 | 0.0065496040063920003 | −1.7884210536842104 | 3.2764402233076537 | −0.30029783822405853 |
| 2 | 2 | 0.9490000005 | 0.18490000005000001 | 0.0018996010009489999 | 0.97315789500000005 | 0.95027563829364681 | 1.7962725901605623 |
| 3 | 0 | 1.5970380543370419 | 0.48352380552370416 | 0.0097721057465930475 | 1.7842206845893143 | 3.2606281234450867 | 0.69891118315823153 |
| 3 | 1 | −1.2011913528962341 | −0.42593913546962342 | 0.007985915068658294 | −1.5717311271941823 | 2.6646354367777301 | −0.20371238138538447 |
| 3 | 2 | 0.89813629508028114 | 0.25622362955302813 | 0.0027043502044885848 | 0.94547464779715173 | 0.90235211949832006 | 1.6949445166806532 |

No-decay case (same θ₀, a, η, β, ε; λ irrelevant because θ is named as a no-decay key, e.g.
`"h.0.mlp.fc.b"`, while the optimizer is constructed with λ = 0.01). θ_t only (m/v/m̂/v̂ at t=1 are
as above):

| t | θ_t[0] | θ_t[1] | θ_t[2] |
|---|---|---|---|
| 1 | 0.90000000049999995 | −0.40000000050000001 | 1.900000001 |
| 2 | 0.80041222869179218 | −0.30118742062342935 | 1.8001664876318755 |
| 3 | 0.7015862729460296 | −0.20487125097072312 | 1.7006233943434106 |

`m̂` and `v̂` are read back as `state["m"]/(1−β₁ᵗ)` and `state["v"]/(1−β₂ᵗ)`. The optimizer does
not have to store them.

### 3.3 `nanogpt/ops.py`, `attention.py`, `model.py` — TL-27 chokepoint (Engineer B, milestone B1)

`ops.matmul(a, b, *, kind="dense")` exactly as in TL-27. Each of the 12 `@` sites is replaced by
`ops.matmul(...)`, keeping operand order, transposes, `swapaxes` and scale placement **unchanged**.
For example, `S = ops.matmul(qh, kh.swapaxes(-1, -2), kind="attention") * self.scale` and
`dqh = self.scale * ops.matmul(dS, kh, kind="attention")`. `kind` must be `"dense"` or
`"attention"`, else ValueError. After B1, the AST of every `nanogpt/*.py` except
`attention_reference.py` contains:
- no `ast.MatMult` BinOp, except the single `a @ b` inside `ops.matmul`;
- no call to `np.matmul`, `np.dot`, `np.einsum`, `np.tensordot`, `np.inner`, `np.vdot` or
  `.dot(`;
- no `from nanogpt.ops import matmul` (it is always called as `ops.matmul`).

### 3.4 `nanogpt/flops.py` (Engineer B)

```python
class FlopCounter:                  # context manager; TL-27 monkeypatch of ops.matmul
    def __enter__(self) -> "FlopCounter"     # swaps ops.matmul for a counting wrapper; a nested/second
                                              # active counter -> RuntimeError
    def __exit__(self, *exc) -> None          # ALWAYS restores the original ops.matmul object (also on error)
    def phase(self, name: str)                # context manager: "forward" | "backward"; calls outside a
                                              # phase are attributed to "unphased"
    counts: dict[tuple[str, str], int]        # (phase, kind) -> FLOPs, kind in {"dense","attention"}
    calls: list[tuple[str, str, tuple, tuple]]  # (phase, kind, a.shape, b.shape), for diagnostics/tests
    def total(self, phase=None, kind=None) -> int
def matmul_flops(a_shape, b_shape) -> int     # TL-28
@dataclass(frozen=True)
class FlopReport:  tokens: int; fwd_dense: int; fwd_attn: int; bwd_dense: int; bwd_attn: int
    # properties: fwd_per_token, train_per_token (= (fwd+bwd)/tokens), attn_share_fwd
def measure_train_flops(model, params, idx, targets) -> FlopReport
    # forward_loss under phase("forward"), backward_loss(1.0) under phase("backward")
def formula_forward_per_token(cfg) -> int     # 2N + 4·L·T·d, N = param_count.formula_from_config(cfg)
def formula_train_per_token(cfg) -> int       # 6N + 12·L·T·d
def naive_forward_per_token(cfg) -> int       # 2N      (test-the-test only)
def naive_train_per_token(cfg) -> int         # 6N
def exact_forward_matmul_per_token(cfg) -> dict  # {"dense": 24Ld²+2Vd, "attention": 4LTd}   (TL-29)
def within_band(measured, predicted, band=FLOPS_REL_BAND) -> bool   # |measured/predicted − 1| ≤ band
```
The wrapper calls the original `ops.matmul` and returns its result unchanged. It never copies or
converts operands.

### 3.5 `train.py` at the repo root (Engineer C)

```python
DEFAULT_STEPS = 4000; DEFAULT_BATCH = 16; DEFAULT_LR = 1e-3; DEFAULT_LOG_EVERY = 100
def get_batch(data: np.ndarray, batch_size: int, T: int, rng) -> tuple[np.ndarray, np.ndarray]
    # starts i ~ rng.integers(0, len(data) − T, size=B)  (so i+T ≤ len(data)−1);
    # x[b] = data[i:i+T], y[b] = data[i+1:i+T+1]; both int64 (B, T); len(data) ≤ T -> ValueError
@dataclass
class TrainResult: params: dict; optimizer: AdamW; model: GPT; losses: list[float];
                   step_times: list[float]; wall_time_s: float; init_loss: float
def train(cfg: ModelConfig, data: np.ndarray, *, steps: int, batch_size: int, lr: float,
          weight_decay: float = ADAMW_WEIGHT_DECAY, betas=ADAMW_BETAS, seed: int = DEFAULT_SEED,
          log_every: int = DEFAULT_LOG_EVERY, log_fn=print) -> TrainResult
    # model = GPT(cfg); params = model.init_params(make_rng(seed)); batch rng = make_rng(seed + 1)
    # opt = AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    # each step: x, y = get_batch(...); loss, c = model.forward_loss(x, y, params);
    #            _, grads = model.backward_loss(1.0, c); opt.step(grads); losses.append(loss)
    # non-finite loss -> raise FloatingPointError (fail loudly, never skip a step)
def main(argv=None) -> int
    # argparse: --steps --batch-size --lr --weight-decay --seed --log-every --L --d --T --H
    #           --allow-out-of-range --profile N (cProfile the first N steps, print top 20 by cumtime)
    # resolves corpus via tokenizer.load_corpus_and_tokenizer(); V = tokenizer vocab size;
    # prints corpus source/sha256[:8], V, N (param count), config, step-0 loss next to ln V,
    # then "step k | loss | ms/step | elapsed", and a final summary line with wall time and steps.
if __name__ == "__main__": raise SystemExit(main())
```
`train.py` contains no checkpoint, sampling, generation or KV-cache code (R22). It does not touch
`flops.py`.

### 3.6 Facts every test relies on (TL prototype)

- V = 55, corpus SHA `7afc055a…2854`, 1,000,000 tokens. N: tiny 2,232, default 107,712, long
  211,776, max 816,768, discriminator 9,520.
- The default-config init loss on real windows (B=16) is about 4.03 (ln 55 = 4.0073), consistent
  with C2.
- Instrumenting with a monkeypatched chokepoint leaves loss and every gradient bit-identical. The
  prototype confirmed this for the subclass variant too, and the chokepoint is strictly simpler.
- `attn.k.b` gets a structurally zero gradient (TL-9; ~1e-16). Under AdamW with ε = 1e-8, its
  per-step update is about η·|g|/ε ≲ 1e-11. It stays essentially fixed and is output-invariant
  anyway, so any "every parameter changed" check must exempt it explicitly.

---

## 4. File ownership (no file has two owners)

| Owner | Creates | Modifies |
|---|---|---|
| Engineer A | `nanogpt/optim.py`, `tests/test_adamw.py` | `nanogpt/config.py` (3 additive constants), `tests/test_config.py` (append one test) |
| Engineer B | `nanogpt/flops.py`, `tests/test_flops.py` | `nanogpt/ops.py`, `nanogpt/attention.py`, `nanogpt/model.py` (TL-27 only: `@` → `ops.matmul`) |
| Engineer C | `train.py`, `tests/test_train.py` | none |
| Tech Lead | `SPRINT_4_PLAN.md` | `PROGRESS.md` (sign-off), `TL fixup:` commits if needed |

Nobody else edits these files. No other Sprint 1–3 test file is edited. If one fails after your
change, your change is wrong.

---

## 5. Engineer A: decay groups, AdamW, closed-form anchor

### 5.1 Milestones (one commit each, in this order)

- **[A1] Constants + decay-group policy.** `config.py` gets ADAMW_* (§3.1).
  `tests/test_config.py` appends `test_adamw_constants`. `optim.py` has `is_decay_param`,
  `canonicalize_params` and the patterns. `tests/test_adamw.py` holds the group/canonicalization
  tests. No dependency on any teammate.
- **[A2] AdamW + closed-form anchors.** The `AdamW` class; the first-step anchor; the 3-step
  trajectory against the §3.2 tables (decay and no-decay); decoupled-decay checks.
- **[A3] Tied single state + test-the-tests.** The tied-parameter tests on a real `GPT`; the four
  buggy-variant adversaries; validation/error tests.

### 5.2 Spec — §2.2 TL-24/25/26, §3.1, §3.2 exactly.

### 5.3 Acceptance criteria (`tests/test_adamw.py` < 2 s)

- **Groups (A1):**
  - On `GPT(tiny)` and `GPT(default)`, the decay set is exactly the 6L `attn.{q,k,v,o}.W` /
    `mlp.{fc,proj}.W` keys, and the no-decay set is exactly the other 4 + 10L keys.
  - Element counts are 98,304 / 9,408 at the default config and sum to `formula_from_config`.
  - Every `.gamma` is no-decay, even though it is weight-like. So are `wte.W` and `wpe.W`.
  - An unknown key, `"lm_head.W"` passed to `is_decay_param`, and a near-miss such as
    `"h.0.attn.q.Wx"` all raise ValueError.
- **Canonicalization (A1):**
  - `canonicalize_params(model.named_parameters(p))` has the same keys as `p`, and its `wte.W` is
    `p["wte.W"]` (identity).
  - A dict holding `X` and `X.T` under two names raises. So do a float32 param and a
    Fortran-order param.
- **First step (A2):** a 3-element param with θ = [0.7, −1.2, 0.5] and g = [3.0, −2e-3, 0.0],
  η = 0.1, β = (0.9, 0.999), ε = `ADAMW_EPS`, λ = 0.01, decay key.
  - `θ₁ − θ₀` equals −η(g/(|g|+ε) + λθ₀) within `ROUNDOFF_ATOL` (closed form, since m̂₁ = g and
    v̂₁ = g² exactly).
  - It is also within η·ε/|g| of −η·sign(g) − ηλθ₀ for g ≠ 0. For g = 0 the update is exactly
    −ηλθ₀.
  - `state["m"] == (1−β₁)g` and `state["v"] == (1−β₂)g²`, both at `ROUNDOFF_ATOL`.
  - `opt.t == 1`.
  - With a no-decay key, the same step gives exactly −η·g/(|g|+ε).
- **Trajectory (A2):** each of the 3 steps matches the §3.2 table (θ, m, v, m̂, v̂ for the decay
  case; θ for the no-decay case) at `ROUNDOFF_ATOL`, checked at **every** step and not only the
  end state. The literals are typed into the test from the table, not recomputed by the
  implementation.
- **Decoupled decay (A2):** with g ≡ 0 for 3 steps, m and v stay exactly 0, and θ_t = θ₀(1−ηλ)ᵗ.
  This proves λθ never enters the moments.
- **Tied single state (A3, the PO P0 dedicated test):**
  - Build `AdamW(model.named_parameters(p))` on tiny.
  - `len(opt.state) == len(model.param_keys())`, and `"lm_head.W" not in opt.state`.
  - `opt.params["wte.W"] is p["wte.W"]`.
  - After one step with the real `model.backward_loss` gradients:
    - `state["wte.W"]["m"] == (1−β₁)·grads["wte.W"]` at `ROUNDOFF_ATOL` (one accumulation, not two);
    - the `wte.W` update matches the single-step closed form;
    - `named_parameters(p)["lm_head.W"] is p["wte.W"]` still holds;
    - `p["wte.W"]` is still C-contiguous.
  - **Adversary:** a test-local naive loop that keeps one (m, v) per `named_parameters()` key
    (two states, two updates of the same array) moves `wte.W` by about 2× the closed form. The test
    asserts it differs from the real optimizer by > 1e-3·η.
  - `opt.step(grads | {"lm_head.W": g})` raises.
- **Test-the-test (A3):** each test-local buggy single-step/3-step function must **mismatch** the
  §3.2 table by more than 1e-6. The real optimizer must match it. The variants:
  1. off-by-one t (first update uses t = 2);
  2. no bias correction;
  3. coupled L2 (g ← g + λθ fed into m and v);
  4. decay applied to a no-decay key (the no-decay table is then missed).

  t starting at 0 is also shown to give a non-finite θ₁ (division by 1 − β⁰ = 0). This is recorded
  as caught, so it cannot silently pass.
- **Validation (A3):** a missing, extra or wrong-shape grad, a float32 grad, lr ≤ 0, β ≥ 1 and
  eps ≤ 0 all raise. `grads` is not mutated. Two identical runs are bit-identical.
- **No test may assert that a loss decreases.**

### 5.4 Process guidance (verbatim, binding)

GIT RULES (the repo has Sprint 1-3 history on it already):
- Identify yourself ONLY via environment variables on each git command, never via `git config`: prefix every git add/commit with GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local".
- Only `git add` the exact files you own — never `git add -A`, `git add .`, or any path outside your assignment.
- End every commit message with a blank line then exactly: Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
- If a git command fails with an index.lock error, a teammate is committing at that exact moment — wait ~2 seconds and retry, up to 3 times.
- Never run `git push`, `git reset --hard`, `git rebase`, or touch any git config file.
- COMMIT DISCIPLINE (Sprint 2's D1 failure, fixed in Sprint 3, must stay fixed): split your own work into small, dependency-ordered, individually-green commits — never batch multiple distinct milestones into one commit. Before depending on a teammate's file, confirm it's actually committed (`git log --oneline -- <their path>`), not just present on disk. Each of your commits should pass the suite given what's actually committed so far plus your own staged change.

Your identity: `Engineer A` / `engineer-a@nanogpt-numpy.local`. You depend on no Sprint 4 teammate.
B1 edits `model.py`/`attention.py` concurrently, but your tests only call their public API, which does
not change. Run the §8.2 check before **each** of A1, A2, A3. Engineer C is waiting on **[A2]**, so
commit it as soon as it is green.

---

## 6. Engineer B: matmul chokepoint, FLOP counter, C9

### 6.1 Milestones (one commit each; order is binding)

- **[B1] Chokepoint refactor (pure, bit-identical).** Add `ops.matmul`, rewrite the 12 sites
  (§3.3), and create `tests/test_flops.py` with: the AST chokepoint guard plus its test-the-test
  (a tmp source containing `x @ y`, `np.dot` or `from nanogpt.ops import matmul` is flagged), the
  `kind` validation, and a bit-identity test (`ops.matmul(a, b)` equals `a @ b` exactly, using
  `np.array_equal`, on 2-D, batched 3-D/4-D and broadcast shapes). The **full** existing suite
  (363) must stay green unchanged. That is the regression proof for code that is already
  gradchecked.
- **[B2] `flops.py` counter + exact inventory (TL-28/29).** Includes: `matmul_flops` unit cases;
  leak-proofing; the exact inventory on all five §TL-30 configs; the per-kind backward = 2 × forward
  structural check.
- **[B3] C9 band + interpretation + test-the-tests.** Formula functions, `within_band`, the band
  assertions, the naive-formula discriminator and the attention-share report.

### 6.2 Spec — §2.2 TL-27..TL-30, §3.3, §3.4 exactly.

### 6.3 Acceptance criteria (`tests/test_flops.py` < 5 s; use B = 1 or 2, seq len = cfg.T)

- **Guard (B1):** the §3.3 AST rules hold for every `nanogpt/*.py` except `attention_reference.py`.
  The counted `@` sites are exactly 1 (inside `ops.matmul`).
- **Bit-identity (B1):** with and without an active `FlopCounter` (from B2 on), `forward_loss` and
  `backward_loss` on tiny and default give an identical loss (`==`) and `np.array_equal` for every
  gradient key.
- **`matmul_flops` (B2):** (3,4)@(4,5) gives 120; (2,3,4)@(4,5) gives 240; (2,1,3,4)@(1,6,4,5)
  gives 2·12·3·4·5; mismatched k raises.
- **Leak-proofing (B2):**
  - Once the counter exits, `ops.matmul` is the original object again, including when the body
    raises.
  - Calls outside any counter are not counted.
  - A nested counter raises RuntimeError.
  - Every `kind="attention"` call has two operands with ndim ≥ 4, and no `dense` call does
    (consistency between the explicit tag and the shapes).
- **Exact inventory (B2):**
  - On tiny, default, long, max and the discriminator: `fwd_dense/tokens == 24Ld² + 2Vd` and
    `fwd_attn/tokens == 4LTd`, both as **integer** equality.
  - `bwd_dense == 2·fwd_dense` and `bwd_attn == 2·fwd_attn`, also exact.
  - Expected fwd/token values are in the TL-30 table.
- **C9 band (B3):** for all four in-range configs plus tiny, `within_band(fwd/tok,
  formula_forward_per_token)` and `within_band(train/tok, formula_train_per_token)` are True. The
  test prints the ratios and the attention share. They should match TL-30 to 4 dp: default 0.9526,
  long 0.9580, max 0.9754, tiny 0.8915.
- **Test-the-test (B3):**
  1. Naive 2N/6N is **outside** the band at long (ratio 1.2545) and at the discriminator (1.5983).
  2. Naive 2N/6N is inside at default. This is asserted and documented so that nobody claims the
     band alone discriminates.
  3. A counter variant that ignores `kind="attention"` calls (a test-local monkeypatch) fails the
     exact inventory **and** misses the corrected band at the discriminator.
  4. A run with the backward outside the counter gives train/tok ≈ fwd/tok and fails the training
     band (ratio about 0.32).
  5. The TL-27 subclass approach is *not* re-implemented. A comment cites the prototype finding.
- **No change to any public signature** in ops/attention/model. `git diff` of B1 shows only
  `@` → `ops.matmul(...)` substitutions plus the new function.

### 6.4 Process guidance (verbatim, binding)

GIT RULES (the repo has Sprint 1-3 history on it already):
- Identify yourself ONLY via environment variables on each git command, never via `git config`: prefix every git add/commit with GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local".
- Only `git add` the exact files you own — never `git add -A`, `git add .`, or any path outside your assignment.
- End every commit message with a blank line then exactly: Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
- If a git command fails with an index.lock error, a teammate is committing at that exact moment — wait ~2 seconds and retry, up to 3 times.
- Never run `git push`, `git reset --hard`, `git rebase`, or touch any git config file.
- COMMIT DISCIPLINE (Sprint 2's D1 failure, fixed in Sprint 3, must stay fixed): split your own work into small, dependency-ordered, individually-green commits — never batch multiple distinct milestones into one commit. Before depending on a teammate's file, confirm it's actually committed (`git log --oneline -- <their path>`), not just present on disk. Each of your commits should pass the suite given what's actually committed so far plus your own staged change.

Your identity: `Engineer B` / `engineer-b@nanogpt-numpy.local`. You depend on no Sprint 4 teammate
(`param_count.py` is already committed). B1 touches three already-verified modules. Commit it
**alone**, and only after the full 363-test suite passes in the §8.2 scratch tree. Do not fold any
`flops.py` code into B1.

---

## 7. Engineer C: training loop

### 7.1 Milestones (one commit each, in this order)

- **[C1] Batching.** `train.py` with `get_batch` and the module constants only, plus
  `tests/test_train.py` batching tests. No dependency on any Sprint 4 teammate, so it can land on
  day 1.
- **[C2] Training loop + CLI.** `train()`, `TrainResult`, `main()` (with `--profile`), and the
  tiny-config smoke tests. **Requires [A2] committed.** Check with
  `git log --oneline -- nanogpt/optim.py`, which must show `[A2]`.
- **[C3] Loop invariants.** Determinism, the tie preserved through training, a single optimizer
  state, the TL-9 `k.b` exemption, the AdamW-is-really-used checks, and the "no loss-decrease
  assertion" guard.

### 7.2 Spec — §2.2 R22/R24/TL-31 and §3.5 exactly. Imports: `nanogpt.config`, `nanogpt.model`, `nanogpt.optim`, `nanogpt.tokenizer`, `nanogpt.param_count` (for the printed N). Nothing else from Sprint 4.

### 7.3 Acceptance criteria (`tests/test_train.py` < 5 s; tiny config, ≤ 10 steps, B ≤ 4)

- **Batching (C1):**
  - On a synthetic `np.arange(100)` "corpus", `y == x + 1` elementwise, the shapes are (B, T), the
    dtype is int64, and every window stays in range.
  - Hitting the extreme start index `len(data) − T − 1` is valid. Use a fixed rng that hits it, or
    assert the bound analytically.
  - The same seed gives identical batches. `len(data) ≤ T` raises.
  - On the real corpus ids (from `load_corpus_and_tokenizer`), a batch decodes to a contiguous
    slice of the corpus text.
- **Smoke (C2):**
  - `train(tiny_config(V), real_ids, steps=5, batch_size=4, lr=1e-2)` completes.
  - `len(losses) == 5` and all are finite.
  - `init_loss` is within `INIT_LOSS_TOL` of ln V. This is a consistency check of the C2 anchor
    through the real loop, not optimizer evidence.
  - Every parameter array changed (`not np.array_equal` against a pre-training copy), **except**
    `h.{l}.attn.k.b`. For those, assert max|Δ| ≤ 1e-6 (TL-9, §3.6).
  - `main([...tiny flags..., "--allow-out-of-range", "--steps", "3"])` returns 0, and the captured
    stdout contains a step-0 line and the summary line.
  - An out-of-range config without the flag exits non-zero, or raises SystemExit.
- **Invariants (C3):**
  - Two `train` calls with the same seed give bit-identical `losses` and params.
  - After training, `result.params["wte.W"]` is the same object as at init, is still C-contiguous,
    and `model.named_parameters(params)["lm_head.W"] is params["wte.W"]`.
  - `len(result.optimizer.state) == len(model.param_keys())`, `opt.t == steps`, and
    `"lm_head.W" not in opt.state`.
  - Monkeypatch `nanogpt.optim.AdamW.step` with a counting wrapper: it is called exactly `steps`
    times.
  - A non-finite loss (monkeypatch `forward_loss` to return NaN) raises FloatingPointError.
  - An AST/text guard on `train.py` finds:
    - no `checkpoint`, `kv_cache`, `sampling` or `generate` import;
    - no `np.save` or `pickle`;
  - On `tests/test_train.py` itself, the guard finds no comparison of the form `losses[...] <
    losses[...]`. The spirit of this is the non-functional rule: no assertion that loss decreased.
- `python train.py --steps 20 --profile 20` (manual, not pytest) runs and prints a profile. Paste
  its top 10 lines into your results block.

### 7.4 Process guidance (verbatim, binding)

GIT RULES (the repo has Sprint 1-3 history on it already):
- Identify yourself ONLY via environment variables on each git command, never via `git config`: prefix every git add/commit with GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local".
- Only `git add` the exact files you own — never `git add -A`, `git add .`, or any path outside your assignment.
- End every commit message with a blank line then exactly: Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
- If a git command fails with an index.lock error, a teammate is committing at that exact moment — wait ~2 seconds and retry, up to 3 times.
- Never run `git push`, `git reset --hard`, `git rebase`, or touch any git config file.
- COMMIT DISCIPLINE (Sprint 2's D1 failure, fixed in Sprint 3, must stay fixed): split your own work into small, dependency-ordered, individually-green commits — never batch multiple distinct milestones into one commit. Before depending on a teammate's file, confirm it's actually committed (`git log --oneline -- <their path>`), not just present on disk. Each of your commits should pass the suite given what's actually committed so far plus your own staged change.

Your identity: `Engineer C` / `engineer-c@nanogpt-numpy.local`. C1 has no Sprint 4 dependency.
C2 and C3 need **[A2]** (`nanogpt/optim.py`) committed. Until then you may draft against a local
scratchpad AdamW stand-in that implements the §3.2 interface. Never put it in the repo. Re-validate
against the real committed `optim.py` in the §8.2 scratch tree before committing, and say so in
your results block.

---

## 8. Git protocol

### 8.1 Rules (shared; identical to the verbatim blocks in §5.4/§6.4/§7.4)

- **Identities:** `Engineer A` / `engineer-a@nanogpt-numpy.local`, `Engineer B` /
  `engineer-b@nanogpt-numpy.local`, `Engineer C` / `engineer-c@nanogpt-numpy.local`, `Tech Lead` /
  `tech-lead@nanogpt-numpy.local`.
- **Shell:** run git from **Git Bash**. The `VAR=value git …` prefix does not work in PowerShell.
- **Pathspec:** commit with an explicit pathspec (`git commit -m … -m … -- <your files>`). This
  avoids the Sprint 3 D3 shared-index race.
- **Subjects:** `Phase 4 [<milestone>]: <what>`.
- **TL fixups:** separate commits titled `TL fixup: <file>: <what>`.
- **Amend:** never `--amend` anything except your own last commit.
- **Ignored paths:** `.venv/`, `__pycache__/` and `.pytest_cache/` are gitignored. Never force-add
  them.

### 8.2 Mandatory pre-commit check (committed tree + only your staged files)

```bash
cd /c/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy
git log --oneline -- <each teammate file you import>       # must show the milestone you need
GIT_AUTHOR_NAME="Engineer X" GIT_AUTHOR_EMAIL="engineer-x@nanogpt-numpy.local" \
GIT_COMMITTER_NAME="Engineer X" GIT_COMMITTER_EMAIL="engineer-x@nanogpt-numpy.local" \
  git add <your files for THIS milestone only>
git diff --cached --name-only                              # only your files; if a teammate's file
                                                           # is listed, commit with `-- <pathspec>`
S=$(mktemp -d)
git archive HEAD | tar -x -C "$S"                          # the COMMITTED tree
git checkout-index -f --prefix="$S/" -- <your files>       # + exactly your STAGED versions
(cd "$S" && /c/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy/.venv/Scripts/python.exe \
   -m pytest tests/ -q -p no:cacheprovider)                # must be all green (no skips of gating tests)
rm -rf "$S"
GIT_AUTHOR_NAME="Engineer X" GIT_AUTHOR_EMAIL="engineer-x@nanogpt-numpy.local" \
GIT_COMMITTER_NAME="Engineer X" GIT_COMMITTER_EMAIL="engineer-x@nanogpt-numpy.local" \
  git commit -m "Phase 4 [X1]: <what>" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" \
  -- <your files>
```
If the scratch run fails only because a teammate's module is missing, you depended on uncommitted
work. Unstage, wait for their commit, and re-run. Paste the scratch run's last line ("N passed")
into your results block for every commit.

---

## 9. Dependency graph and commit ordering

```
A1 groups ──> A2 AdamW + anchors ──┬─> A3 tied state + test-the-tests
                                   ├─> C2 train loop + CLI ──> C3 loop invariants
C1 batching (independent) ─────────┘   (C2 also needs C1)
B1 ops.matmul chokepoint ──> B2 flops.py + exact inventory ──> B3 C9 band + discriminators
```
Required order in `git log` (checked at sign-off): `[A1] < [A2] < [A3]`, `[A2] < [C2]`,
`[C1] < [C2] < [C3]`, `[B1] < [B2] < [B3]`. A and B are independent of each other; C1 is independent
of everyone. No commit contains two milestones. No stand-in or `TODO(SPRINT4-…)` marker is ever
committed (the `TODO(SPRINT` guard fails on it). Interface changes to §3 need TL approval; internals
do not.

---

## 10. Timeline (≈ 3 engineering days)

| Day | A | B | C |
|---|---|---|---|
| 1 AM | **A1** constants + groups (commit) | **B1** chokepoint refactor, full suite in scratch tree (commit) | **C1** batching (commit) |
| 1 PM | **A2** AdamW + anchors (commit ASAP; C waits) | B2 counter + inventory | draft C2 against a scratchpad AdamW stand-in |
| 2 AM | **A3** tied state + test-the-tests (commit) | **B2** (commit) | **C2** after `[A2]` (commit) |
| 2 PM | results block | **B3** C9 (commit) | **C3** (commit); `--profile 20` run |
| 3 | | results block | results block |
| 3–4 | **TL integration**: per-commit isolation, real 30-min run + profile, sign-off | | |

TL duties during the sprint: review each milestone as it lands (fixes as separate `TL fixup:`
commits), and verify each new commit green in isolation the same day (§11 item 6) so a D1 repeat is
caught immediately, not at sign-off.

---

## 11. Integration and sign-off checklist (TL; re-measured, not copied)

1. `pytest tests/ -q --durations=0`: all green, < 60 s, each file < 10 s. Report test_adamw,
   test_flops and test_train timings explicitly. Two consecutive runs must be identical.
2. TL throwaway re-measurement:
   - AdamW first step and 3-step table, from the TL's own scalar loop;
   - decay/no-decay key sets and element counts;
   - the tied single-state check on the **default** config;
   - C9 measured vs. corrected vs. naive at all five TL-30 configs, plus attention share;
   - **one real `train.py` step under `FlopCounter`**, which must equal the exact inventory (AdamW
     adds no matmul).
3. **Profile right after the first working loop (PM T7):** run `python train.py --steps 50
   --profile 50` as soon as [C2] lands. Record the top functions by cumulative time and the
   measured s/step. If the projected default run is over about 25 min, trim batch size or steps,
   then L/d/T within range (in that order) before any code optimization.
4. **Real run (outside pytest):** `python train.py` at the defaults. Record:
   - wall-clock time, steps and mean s/step;
   - step-0 loss vs. ln 55;
   - the logged loss every 100 steps, **for the record only, not as evidence**.
   It must be ≤ 30 min.
5. All test-the-tests are confirmed by passing assertions:
   - AdamW: off-by-one t, no bias correction, coupled L2, decay on an excluded key, t = 0 →
     non-finite, and the two-state tied adversary;
   - C9: naive outside the band at long and at the discriminator, attention-blind counter, backward
     not counted;
   - chokepoint guard on synthetic sources.
6. **Commit-discipline gate (P0):** for every commit c in `<kickoff>..HEAD`, run
   `S=$(mktemp -d); git archive $c | tar -x -C $S; (cd $S && python -m pytest tests/ -q
   -p no:cacheprovider)`. It must be green.
   - Each engineer commit touches only that engineer's files (`git show --stat`) and holds one
     milestone.
   - The §9 order holds.
   - Any red interior commit is a rejection-level deviation, recorded and not rewritten.
7. Static checks:
   - `grep -rn "TODO(SPRINT" nanogpt/` is empty and the conventions guard is green;
   - no torch, jax, tf or autograd;
   - no `lm_head` key in `param_keys()` or in `opt.state`;
   - the chokepoint guard is green;
   - no bare `1e-8` in `optim.py`.
8. **No Phase 5+ files:** `kv_cache.py`, `sampling.py`, `checkpoint.py`, `generate.py`,
   `tests/test_kv_cache.py` and `tests/test_numerical_stability.py` do not exist.
   `model.py`/`attention.py` have no KV-cache argument, `generate()` or sampling code. `train.py`
   has no save/load.
9. The corpus SHA-256 is still `7afc055a…2854` and V = 55. Sprint 1–3 hashes are unchanged
   (`dfbd373` is an ancestor of HEAD). `git status` is clean.
10. Append the PROGRESS.md Phase 4 entry (§12) and commit `Sprint 4 integration sign-off`.

---

## 12. Results block and PROGRESS template

**Results block each engineer hands the TL:**
```
Files: <>   Owner: <A|B|C>   Commits: <hash + subject, one per milestone>
Per commit: dependency check (git log -- <path>) output + scratch-tree run "<N passed>"
A: decay/no-decay key lists + element counts; first-step residuals; per-step max |impl − table|
   for θ/m/v/m̂/v̂; tied-state numbers (len(state), m residual, adversary Δ ratio)
B: B1 suite count before/after; fwd/tok, train/tok, ratio vs corrected and naive, attention share,
   for all five configs; inventory equality per config
C: smoke run losses (finite), init_loss − ln V, k.b max|Δ|, opt.t, profile top-10 (20 steps)
Test-the-test: <variant> -> FAILED as expected (<metric>=<value>)
Local stand-ins used before a dependency was committed: <none | what, when removed>
Timing: <file> <s>   Deviations from SPRINT_4_PLAN.md: <none | what + why>
```

**PROGRESS.md Phase 4 entry (TL; append-only):**
- owners and commits, with the per-commit isolation table (item 6);
- suite status and per-file timings;
- the TL-24 decay-group ruling, **for PO ratification**: `wpe.W` is no-decay;
- the AdamW anchor numbers and tied single-state evidence;
- the C9 table (TL-30 format) with the attention-share interpretation;
- the profile and the real-run timing, plus the loss log, marked "record only, not evidence";
- rulings R22–R27 and TL-24..TL-31; deviations;
- confirmation that no Phase 5+ files exist;
- "Next: Phase 5": `kv_cache.py`, `sampling.py`, `checkpoint.py`, C8.

---

## 13. Explicitly out of scope for Sprint 4 (do not build)

- C8 / `kv_cache.py` or any past-K/V argument.
- `sampling.py`, temperature/top-k, `generate.py`, and any `generate()` on the model.
- `checkpoint.py` and any save/load or optimizer `state_dict` (R22).
- The Phase 6 large-logit stability pass and GPT-2 weight loading.
- LR schedules, gradient clipping, loss averaging, validation split, README edits (R27).
- Performance work beyond the TL-31 trimming levers.
- Any edit to Sprint 1–3 test files, and any change to a public signature in
  ops/attention/model/block/mlp/layernorm/loss.
