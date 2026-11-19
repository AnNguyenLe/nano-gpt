# Sprint 2 Plan — Phase 2: Causal Multi-Head Self-Attention (Tech Lead breakdown)

**Scope:** Execution Plan Phase 2 only. Nothing from Phase 3 or later.
**Sources reconciled:** `../nanoGPT_build_requirements.md` (spec, incl. §0 Frozen Conventions and §4
edge cases), `../nanoGPT_execution_plan.md` (Phase 2 section and risk table), `PROGRESS.md` (Sprint 1
sign-off state and watch items), PM Sprint 2 backlog (S2-00..S2-11), PO Sprint 2 acceptance
priorities. Order of authority, unchanged from Sprint 1: **spec > plan > PO > PM**. The TL rulings in
§2 settle what the documents leave open. Sprint 1 rulings TL-1..TL-8 and R1..R11 remain in force.
**Team:** 3 engineers. Each owns a disjoint set of files (§4), so they can commit concurrently without
collisions.
**Process change (S2-00, approved by the user):** the team commits as it works, under per-role
identities. See §8. Sprint 1 history (`283c20f`..`74044c4`) is never rewritten.

**Prototype evidence.** Before writing this plan the TL built a throwaway scratch prototype (outside
the repo, never committed) of exactly the design frozen in §3. It was run against the real
`nanogpt.ops` and `nanogpt.gradcheck`. The measured values quoted below ("measured: …") come from that
prototype. They are the **expected** values that your independent implementations should reproduce.
They are not something to copy. If your implementation disagrees with one of them, investigate before
you change a test.

---

## 1. Sprint 2 Definition of Done

1. `nanogpt/attention.py` provides the vectorized causal multi-head self-attention, with a
   hand-derived backward that follows the module contract. It reuses `ops.Linear` and
   `ops.softmax_forward` / `ops.softmax_backward`, is float64-strict, and takes every constant from
   `nanogpt/config.py`.
2. A naive loop-over-heads oracle (`nanogpt/attention_reference.py`) exists, uses no reshape or
   transpose, and has its own closed-form anchors. The vectorized forward matches it to float64
   precision (max abs ≤ `ROUNDOFF_ATOL`) on the tiny config (H=2), the default H=4 / d=64, H ∈
   {1,2,4,8} at d=8, B>1 and odd T. **This match is the hard gate (S2-07) before the backward is
   trusted.**
3. Module-level gradcheck on the tiny config (d=8, H=2, T=4, B=2) covers all 8 Q/K/V/O W and b plus
   x. Norm-wise `rel_err < GRADCHECK_TOL`, with `max_elem_rel_err` reported beside it for **every**
   tensor. `attn.k.b` is gated under TL-9 (§2.2), because its true gradient is identically zero.
4. **C5 (isolated):** a numerical Jacobian shows ∂out_t/∂x_s = 0 for every s > t, on both the
   vectorized module and the oracle. Every s ≤ t block is non-zero (non-vacuity). **Shifted-mask
   test-the-test:** `j ≤ i+1` is actually substituted into the module and C5 FAILS. The leak sits
   exactly at s = t+1, with zero at s ≥ t+2.
5. **T=1:** forward and backward are finite. The analytic dW_q, db_q, dW_k, db_k are exactly 0
   (`np.all(g == 0)`). Gradcheck reports `rel_err == 0.0` for those four tensors and passes. In the
   paired T=4 run, W_q, b_q and W_k are non-zero (b_k stays zero per TL-9).
6. P1 items: the oracle catches a wrong head split (test-the-test), the vectorized backward matches
   FD through the oracle, the Sprint 1 watch items (seed collision, contiguity) are enforced, and
   nothing produces NaN/Inf.
7. `python -m pytest tests/ -q` is all green and deterministic over two runs. The suite takes
   < 60 s and no file takes > 10 s.
8. The git history shows incremental per-role commits in dependency order (§9), with TL fixups as
   separate commits, and ends in a TL sign-off commit on a green tree.
9. `PROGRESS.md` gets an appended Phase 2 entry with TL-re-measured numbers (template in §12),
   including the open item **"C5 full-model re-run owed in Phase 3"**.

Of C1–C9, only **C5 (isolated half)** is a Sprint 2 acceptance criterion. C5 is **not retired** by
this sprint.

---

## 2. Reconciliation log and Tech Lead rulings

### 2.1 PM backlog vs. PO priorities vs. spec/plan

| # | Item | Source says | **Ruling** |
|---|---|---|---|
| R12 | Where the oracle lives | Plan §3 layout: `attention.py` holds "vectorized multi-head + a naive loop-over-heads reference oracle". PO repeats it. | **Deviation, recorded.** The oracle goes in its own file `nanogpt/attention_reference.py`, owned by a **different engineer** from the vectorized module. Two reasons. (1) One file with two owners would make concurrent commits collide. (2) An oracle written by the same person, in the same file, as the code it checks is not independent: correlated misreadings of the head convention go undetected. The masks get a tiny neutral module `nanogpt/attention_mask.py`, so production code never imports from a "reference" module. Record in PROGRESS as R12. |
| R13 | Test files | Plan: `tests/test_causal_mask.py` (C5 isolated, T=1, shifted mask). PO: that file plus per-owner attention test files (R6). | `tests/test_causal_mask.py` (C), `tests/test_attention.py` (B), `tests/test_attention_reference.py` (A). `tests/test_gradients.py` stays **reserved for C6** (Phase 3). |
| R14 | T=1 pairing | PO: "a paired T>1 test shows the same Q/K gradients are non-zero" (W_q, b_q, W_k **and b_k**). | **PO corrected on b_k** (TL-9). At T>1, W_q, b_q and W_k are non-zero. **b_k is zero at every T**, by softmax shift-invariance. The paired test asserts non-zero for W_q/b_q/W_k and, separately, the TL-9 structural zero for b_k. |
| R15 | Mask application | PM S2-03: "−∞ added to scores". | Equivalent, but implemented as `np.where(allowed, S, -np.inf)` with a **boolean** `allowed` matrix (TL-12). No arithmetic is ever done on −∞, and the backward never touches the masked scores. |
| R16 | Scaffolding task | PM S2-01 is a separate "skeleton only" task. | Folded into B's first milestone B1 (constructor + forward). A skeleton-only commit has no test that can be green, and Sprint 1 R1 forbids empty placeholders. |
| R17 | Seed collision scope | PM S2-08: Q, K, V share shape (B,T,H,hd), so it is a risk. PO: x has the output's shape (B,T,d). | The PO identifies the real collision: gradcheck's R has `out.shape == x.shape == (B,T,d)`. The rule is broader than "same shape", though. `rng.standard_normal` produces the **same stream regardless of shape**, so the first draw of *any* fixture tensor from `make_rng(DEFAULT_SEED)` is a prefix of R. **Ruling (TL-13): no Sprint 2 fixture tensor is ever drawn from `make_rng(DEFAULT_SEED)`.** Q/K/V are intermediates, not fixture draws, so the PM's concern becomes TL-13's "one rng per fixture, drawn sequentially, assert W_q ≠ W_k". |
| R18 | S2-00 documentation | PM: document the git convention "at the top of PROGRESS.md's Phase 2 entry once the sprint starts". | This committed plan (§8) is the sprint-start record. PROGRESS.md is append-only, with one entry per phase written at sign-off. The Phase 2 entry **opens** with the git convention and any deviations (§12 template). No mid-sprint stub entry. |
| R19 | Fused vs. separate QKV | PO scope risk: it must be a deliberate, recorded choice (it feeds Phase 3's C1). | **Separate** Q, K, V, O `ops.Linear` instances (TL-10). Recorded now. |
| R20 | C5 sequence length | PO: T ≥ 3. Tiny config: T = 4. | C5 runs at T = 4 (tiny d=8, H=2) on both implementations, with a spot check at T = 6 for d=64, H=4. |
| R21 | KV-cache / perf | PO scope risks. | Out of scope. `forward(x, params)` keeps its simple signature, with no past-K/V arguments, no einsum tricks and no profiling. |

### 2.2 Technical rulings (TL-9 .. TL-15)

**TL-9. The key bias b_k has an identically zero true gradient at every T. It is gated by a
structural-zero protocol, not by the rel_err ratio. This needs PO ratification.**
Derivation: b_k adds the same vector to every key, so S[b,h,i,j] gains `scale · q_i^(h) · b_k^(h)`,
which **does not depend on j**. Every allowed score in row i shifts by the same constant, and softmax
is shift-invariant (C4), so P, and with it out, is unchanged. Hence ∂out/∂b_k ≡ 0 for every T, every
input and every parameter value. The analytic backward agrees: `db_k = Σ_j dk_j = scale · Σ_i q_i ·
(Σ_j dS_ij) = 0`, because every row of the softmax backward sums to 0. The FD estimate is pure
roundoff (≈ ε_mach·|L|/h). **Measured** on the tiny fixture: analytic ‖db_k‖∞ = 2.8e-16, FD max ≈
6.7e-11, so the gated `rel_err = 7.1e-3` and `max_elem_rel_err = 6.7e-3`. The 1e-8 floor only rescues
a gradient when **both** sides are exactly zero (T=1), not when FD returns noise. A literal gate would
fail a correct implementation. Even the spec's literal elementwise formula fails here.
**Protocol for `attn.k.b`** (in addition to reporting rel_err and max_elem_rel_err like every
tensor):
- (a) analytic `np.max(np.abs(db_k)) ≤ ROUNDOFF_ATOL`;
- (b) the FD side is below the frozen zero floor: `report.tensors["attn.k.b"].max_abs_err ≤
  REL_ERR_FLOOR` (measured 6.7e-11, a 150× margin);
- (c) a closed-form invariance anchor: `out(b_k + δ) == out(b_k)` within `ROUNDOFF_ATOL` for δ =
  10·N(0,1) (measured 5.6e-16);
- (d) test-the-test: substituting the dropped-term softmax backward `dz = p⊙dp` gives
  ‖db_k‖∞ ≫ `REL_ERR_FLOOR` (measured 1.6), so the protocol is discriminating.

b_k stays a parameter: the spec's Frozen Conventions require a bias on every Q/K/V/O linear, and C1's
L(12d²+13d) counts it. This is the "materially large max_elem_rel_err on a tensor that is not
well-conditioned, investigated rather than waved through" case the PO's TL-1 condition anticipated.
It is reported as **TL-9** in the Phase 2 entry for PO ratification. Every **other** tensor is gated
on the literal `rel_err < GRADCHECK_TOL`. Their measured max_elem_rel_err are all ≤ 7e-9.

**TL-10. Parameter layout: four separate `ops.Linear` instances, keys `{prefix}.q.W`, `{prefix}.q.b`,
`{prefix}.k.W`, `{prefix}.k.b`, `{prefix}.v.W`, `{prefix}.v.b`, `{prefix}.o.W`, `{prefix}.o.b`.**
The default prefix is `"attn"` (Phase 3 will use e.g. `"h0.attn"`). Each W is (d, d) in `ops.Linear`'s
(d_in, d_out) convention, and each b is (d,). There is no fused QKV weight. Every parameter is
therefore its own C-contiguous array, the hardened gradcheck contiguity check is satisfied by
construction (watch item b), and C1 counts 4d² + 4d per layer for attention. **Do not relax the
gradcheck contiguity check.**

**TL-11. Head-column convention (frozen; A and B both code to this).** Head h ∈ {0..H−1} owns
feature columns `[h·hd, (h+1)·hd)` of q, k and v, and writes the same columns of the pre-projection
concat Y. With `hd = d // H`: `split(a)[b,h,t,c] = a[b,t,h·hd + c]` and `merge(y)[b,t,h·hd + c] =
y[b,h,t,c]`. This is the GPT-2 convention. In NumPy: `a.reshape(B,T,H,hd).transpose(0,2,1,3)` and
its inverse `y.transpose(0,2,1,3).reshape(B,T,d)`.

**TL-12. Mask representation and injection.** A mask is a pure function `mask_fn(T) -> np.ndarray`
of dtype `bool` and shape (T,T), where `allowed[i, j]` is True iff query i may attend to key j.
- **Correct** (`causal_mask`): `allowed[i,j] = (j <= i)`, i.e. exclusion strictly above the diagonal.
- **Adversary** (`shifted_causal_mask`, tests only): `allowed[i,j] = (j <= i + 1)`. Exactly one
  extra future token is visible.

Both `CausalSelfAttention` and the oracle take `mask_fn` as an argument, default `causal_mask`.
That argument is the **only** way the shifted variant is substituted, so "substitute the bug into
attention" means `CausalSelfAttention(..., mask_fn=shifted_causal_mask)`, with no monkeypatching. The
causal mask always keeps the diagonal, so no row is all-masked (TL-5 precondition). The "diagonal
masked" variant `j < i` is a **different** bug. It makes row 0 all −∞ and therefore NaN (measured).
It must never be used as the C5 adversary.

**TL-13. Fixture rules (Sprint 1 watch item a, generalized).**
- Each test file defines one fixture seed: `_FIXTURE_SEED = DEFAULT_SEED + 201` (A), `+ 202` (B),
  `+ 203` (C). **No fixture tensor is ever drawn from `make_rng(DEFAULT_SEED)`** (R17).
- Create one `Generator` per fixture and draw its tensors **sequentially** from it. Never re-create
  the rng per tensor, which would make W_q == W_k. Each fixture asserts
  `not np.array_equal(W_q, W_k)`.
- Gradcheck/C5 fixtures use **non-init** parameters (TL-6 analogue): `W = rng.standard_normal((d,d))
  / sqrt(d)` and `b = 0.1 * rng.standard_normal(d)`, with `x = rng.standard_normal((B,T,d))`. At the
  §0 init (std 0.02, biases 0) the scores are ~1e-3, attention is near-uniform, and Q/K gradients are
  ~1e-12. The checks would still pass, but they would discriminate far less. The init-scale case is
  covered only by B's `init_params` shape/dtype tests.

**TL-14. C5 numerics.** Masked probabilities are exactly 0.0 (`exp(-inf - m) = 0`, m finite), and
GEMM rows are independent, so perturbing x_s for s > t leaves out_t **bitwise** unchanged. The future
blocks of the FD Jacobian are therefore exactly 0.0 (measured 0.0 on both implementations and at
d=64). Tests assert `≤ ROUNDOFF_ATOL`, which is robust to an exotic BLAS, and the results block
reports the measured value. The expected value is 0.0. Non-vacuity: every s ≤ t block has Frobenius
norm `> NONVACUITY_MIN = 1e-6`, a test-local constant (measured min 0.13). Shifted mask: the (t, t+1)
blocks are non-zero (measured max 0.63), and the s ≥ t+2 blocks are still exactly 0, because one
attention layer only sees x_{≤t+1}.

**TL-15. Backward never touches −∞.** The forward caches `P`, `allowed`, `qh`, `kh`, `vh` and the
four Linear caches, and **not** Z. The backward computes `dS = np.where(allowed, dZ, 0.0)`. Since P=0
at masked entries, dZ is already exactly 0 there. The `where` makes it explicit and guarantees that
nothing of the form −∞·0 is ever evaluated.

---

## 3. Frozen interfaces and formulas (all three engineers code to this)

### 3.1 Conventions restated (spec §0; import, never redefine)
- `DTYPE = float64`. Call `check_float64` on every float input, parameter and upstream gradient, and
  raise `TypeError` otherwise (TL-7).
- H heads with `d % H == 0` and `head_dim = hd = d // H`. Scores are scaled by `scale = 1.0 /
  math.sqrt(hd)`, computed from hd. The literal 1.0 is allowed by the guard. Never hard-code 0.5 or
  0.3535….
- Defaults: `N_HEAD = 4` (d=64 ⇒ hd=16). `tiny_config(V)`: d=8, L=2, H=2, T=4 (hd=4). The attention
  module does not use V. Tests pass any positive V, e.g. `tiny_config(V=11)`.
- `LINEAR_HAS_BIAS = True` for Q/K/V/O. `FD_STEP = 1e-5`. `GRADCHECK_TOL = 1e-6`.
- Gating `rel_err(a,b) = ‖a−b‖_F / max(‖a‖_F + ‖b‖_F, REL_ERR_FLOOR)` with `REL_ERR_FLOOR = 1e-8`
  (TL-1; equals the spec formula for scalars). `rel_err_elementwise` is a diagnostic. Exact-identity
  checks use `ROUNDOFF_ATOL = 1e-12`. SVD rank uses `C3_SVD_REL_TOL·s_max`.
- No new constants are added to `config.py` this sprint. None are needed, and nobody edits
  `config.py`.

### 3.2 `nanogpt/attention_mask.py` (Engineer A)
```python
def causal_mask(T: int) -> np.ndarray          # bool (T,T): allowed[i,j] = (j <= i)
def shifted_causal_mask(T: int) -> np.ndarray  # ADVERSARY, tests only: allowed[i,j] = (j <= i+1)
```
- Implement them literally from the definition, e.g. `np.arange(T)[None, :] <= np.arange(T)[:, None]`
  (and `+ 1` for the shifted one), so that the code reads like the spec.
- Raise `ValueError` unless T is a positive int (bool rejected). Return a **fresh** array on every
  call, with no cached global that a test could mutate.
- The module docstring states that `shifted_causal_mask` exists only for the C5 test-the-test and
  must never be a default anywhere (C adds a guard test, §7).

### 3.3 `nanogpt/attention_reference.py` (Engineer A): the oracle
```python
def reference_attention_forward(x, params, H, prefix="attn", mask_fn=causal_mask) -> np.ndarray
```
- **Forward only.** It returns `out` (B,T,d) with no cache and no backward.
- Validate: `x.ndim == 3`, `d = x.shape[-1]`, `d % H == 0` (else `ValueError`), and float64 on x and
  the 8 params.
- Algorithm (plain Python loops; **no** `reshape`, `transpose`, `swapaxes`, `moveaxis`, `einsum`, `.T`,
  or `inf` anywhere in the function):
  ```
  q = x @ Wq + bq ; k = x @ Wk + bk ; v = x @ Wv + bv          # (B,T,d), direct matmul
  allowed = mask_fn(T) ; y = zeros((B,T,d))
  for b in range(B): for h in range(H):
      cols = slice(h*hd, (h+1)*hd)                              # TL-11
      for i in range(T):
          js = [j for j in range(T) if allowed[i, j]]           # allowed keys only, no -inf
          s  = array([scale * dot(q[b,i,cols], k[b,j,cols]) for j in js])
          p, _ = ops.softmax_forward(s)                         # 1-D softmax over allowed keys
          for jj, j in enumerate(js): y[b,i,cols] += p[jj] * v[b,j,cols]
  out = y @ Wo + bo
  ```
- It softmaxes over the **allowed subset** instead of over −∞-padded rows. The oracle therefore also
  independently checks the vectorized module's −∞ masking mechanism. It never mutates x or params.

### 3.4 `nanogpt/attention.py` (Engineer B): the vectorized module
```python
def _split_heads(a, H) -> np.ndarray   # (B,T,d) -> (B,H,T,hd), TL-11
def _merge_heads(y) -> np.ndarray      # (B,H,T,hd) -> (B,T,d), exact inverse of _split_heads

class CausalSelfAttention:
    def __init__(self, prefix: str, d: int, H: int = N_HEAD, mask_fn=causal_mask)
        # ValueError unless d, H positive ints and d % H == 0 (do not rely on ModelConfig having checked)
        # self.head_dim = d // H ; self.scale = 1.0 / math.sqrt(self.head_dim)
        # self.q, self.k, self.v, self.o = ops.Linear(f"{prefix}.q", d, d, bias=LINEAR_HAS_BIAS), ...
    @classmethod
    def from_config(cls, cfg: ModelConfig, prefix="attn", mask_fn=causal_mask)
    def param_keys(self) -> list[str]      # [q.W, q.b, k.W, k.b, v.W, v.b, o.W, o.b] with prefix
    def init_params(self, rng) -> dict     # q, k, v, o Linear.init_params from the SAME rng, in that order
    def forward(self, x, params) -> (out, cache)
    def backward(self, d_out, cache) -> (d_x, d_params)
```
**Forward** (x is (B,T,d) float64 with T ≥ 1; `ValueError` if `x.ndim != 3` or `x.shape[-1] != d`):
```
q, cq = self.q.forward(x, params) ; k, ck = ... ; v, cv = ...          # (B,T,d)
qh, kh, vh = _split_heads(q,H), _split_heads(k,H), _split_heads(v,H)    # (B,H,T,hd)
S  = (qh @ kh.swapaxes(-1,-2)) * self.scale                            # (B,H,T,T)
allowed = self.mask_fn(T)                                              # (T,T) bool, broadcasts
Z  = np.where(allowed, S, -np.inf)
P, _ = ops.softmax_forward(Z)                                          # masked P == 0.0 exactly
Y  = _merge_heads(P @ vh)                                              # (B,T,d)
out, co = self.o.forward(Y, params)
cache = {"cq","ck","cv","co","qh","kh","vh","P","allowed"}             # NOT Z (TL-15)
```
`cache["P"]` (B,H,T,T) and `cache["allowed"]` are **frozen, documented keys**. C's rank test reads
them read-only. The other cache keys are private.

**Backward** (hand-derived; restated so no one re-derives it differently):
```
dY, g_o   = self.o.backward(d_out, co)          # dW_o = Σ Yᵀd_out, db_o = Σ d_out, dY = d_out W_oᵀ
dYh       = _split_heads(dY, H)                 # adjoint of merge = its inverse (a permutation)
dP        = dYh @ vh.swapaxes(-1,-2)            # (B,H,T,T)
dvh       = P.swapaxes(-1,-2) @ dYh             # (B,H,T,hd)
dZ, _     = ops.softmax_backward(dP, {"p": P})  # P ⊙ (dP − Σ_j dP⊙P)   (TL-4, reused, not re-implemented)
dS        = np.where(allowed, dZ, 0.0)          # TL-15
dqh       = self.scale * (dS @ kh)              # S = scale·Q Kᵀ  ⇒  dQ = scale·dS K
dkh       = self.scale * (dS.swapaxes(-1,-2) @ qh)   #                 dK = scale·dSᵀ Q
dx_q, g_q = self.q.backward(_merge_heads(dqh), cq)   # adjoint of split = merge
dx_k, g_k = self.k.backward(_merge_heads(dkh), ck)
dx_v, g_v = self.v.backward(_merge_heads(dvh), cv)
d_x       = dx_q + dx_k + dx_v                  # x fans out to 3 consumers: SUM, never overwrite
d_params  = {**g_q, **g_k, **g_v, **g_o}        # exactly the 8 keys
```
- The forward and backward must go through the **two module-level helpers** `_split_heads` /
  `_merge_heads`. B's test-the-test monkeypatches them (§6).
- `backward` depends only on `(d_out, cache)`. It mutates nothing and calls `check_float64(d_out)`.
- There is no KV-cache argument, no dropout, no FLOP hooks and no einsum (R21).

### 3.5 Facts every test relies on (derived by the TL, confirmed by the prototype)
- **F1 (masking is exact):** masked P == 0.0 and masked dZ == 0.0, so causality holds bitwise (TL-14).
- **F2 (T=1):** the softmax runs over one key, giving P ≡ 1.0 exactly (`exp(z−z) = 1`), so `dZ = 1·(dP
  − dP) = 0` exactly. Then dq = dk = 0, and **dW_q, db_q, dW_k, db_k are exactly 0**. The FD estimate
  is also exactly 0, because P is bitwise 1 under any perturbation, so `rel_err = 0/1e-8 = 0.0`
  (measured). Closed-form anchor: `out = (x W_v + b_v) W_o + b_o` exactly (measured diff 0.0).
- **F3 (TL-9):** ∂out/∂b_k ≡ 0 for every T.
- **F4 (C5):** ∂out_t/∂x_s = 0 for s > t. The shifted mask leaks exactly at s = t+1.
- **F5 (C4 note):** the softmax Jacobian of causal row i, diag(p) − ppᵀ, has rank (i+1) − 1 = i
  (measured: rows 0..3 give ranks 0..3).
- **F6:** a uniform-attention anchor. If W_q = 0 and b_q = 0, then S ≡ 0 and
  `out_t = (mean_{j≤t} v_j) W_o + b_o` (cumulative mean). This holds for every head, so it checks
  masking and the weighted sum, independent of the head split.
- **F7 (why the oracle is mandatory):** an *interleaved* head split (`reshape(B,T,hd,H).transpose(0,3,1,2)`
  with its matching inverse) **passes gradcheck** on every non-b_k tensor (worst 5.5e-11) **and
  passes C5**, yet differs from the oracle by 0.61. A no-transpose split
  (`reshape(B,H,T,hd)`) differs by 1.6. Only the oracle catches the first one.

---

## 4. File ownership (no file has two owners)

Root: `c:/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy/`

| Path | Owner | Notes |
|---|---|---|
| `nanogpt/attention_mask.py` | **A** | `causal_mask`, `shifted_causal_mask` |
| `nanogpt/attention_reference.py` | **A** | `reference_attention_forward` (oracle) |
| `tests/test_attention_reference.py` | **A** | mask unit tests, oracle anchors, oracle source guard |
| `nanogpt/attention.py` | **B** | `CausalSelfAttention`, `_split_heads`, `_merge_heads` |
| `tests/test_attention.py` | **B** | oracle match (gate), gradcheck table, TL-9, test-the-tests, robustness |
| `tests/test_causal_mask.py` | **C** | C5 (both impls) + shifted test-the-test + T=1 + backward causality + rank |
| `SPRINT_2_PLAN.md`, `PROGRESS.md` (Phase 2 entry), `tests/test_conventions.py` (one-line marker extension, §10) | **TL** | TL edits to engineer files happen **only** as separate `TL fixup` commits |

No engineer edits `config.py`, `ops.py`, `gradcheck.py`, `layernorm.py`, `loss.py`, `tokenizer.py`,
`conftest.py` or any Sprint 1 test. If you need a change there, ask the TL. There is no
`tests/__init__.py` and no shared test helpers (no `tests/conftest.py`). Each test file builds its own
fixtures, and that duplication is deliberate. Do not create any file outside your rows without TL
sign-off.

Environment: reuse the Sprint 1 venv. Run `.venv/Scripts/python.exe -m pytest tests/<your file> -q`
from the repo root.

---

## 5. Engineer A: masks + naive oracle (the foundation; on the critical path)

### 5.1 Milestones (commit each one, §8)
- **A1 (≤ 1 h; unblocks B and C):** `nanogpt/attention_mask.py` + its mask tests in
  `tests/test_attention_reference.py`. Commit: `Phase 2 [A1]: causal and shifted-adversary mask functions`.
- **A2 (≤ ½ day; unblocks B2 = the hard gate, and C's oracle-side C5):** `nanogpt/attention_reference.py`
  + its anchors. Commit: `Phase 2 [A2]: naive loop-over-heads reference oracle + closed-form anchors`.
- **A3:** the oracle source guard + remaining anchors + the results block (§12) handed to the TL.

### 5.2 Spec
§3.2 and §3.3 exactly. The oracle is the ground truth, so its own correctness rests on **closed-form
anchors**, not on agreement with B.

### 5.3 Acceptance criteria (`tests/test_attention_reference.py`, `_FIXTURE_SEED = DEFAULT_SEED + 201`, < 10 s)
Masks:
- `causal_mask(T)` equals an explicit double loop `[[j <= i for j in range(T)] for i in range(T)]`
  for T = 1..6. Its dtype is bool and its shape (T,T). The diagonal is all True. Row i has exactly i+1
  True entries.
- `shifted_causal_mask(T)` equals the loop `j <= i + 1`. Row i has `min(i+2, T)` True entries.
  `shifted & ~causal` is True **exactly** on the superdiagonal (i, i+1) and nowhere else, which
  proves it is the precise one-extra-token adversary.
- T ∈ {0, −1, 2.0, True} raises `ValueError`. Mutating a returned mask does not affect the next call.

Oracle anchors (all to `ROUNDOFF_ATOL`; non-init fixture per TL-13 unless stated):
- **T=1 (F2):** `out == (x W_v + b_v) W_o + b_o` at B=2.
- **Uniform attention (F6):** with W_q = 0 and b_q = 0, out equals the cumulative-mean formula
  computed independently with `np.cumsum(v, axis=1) / np.arange(1, T+1)[:, None]`, at B=2, T=5, H=2.
- **Two-token hand computation:** d=4, H=2, T=2, W_v = W_o = I, b_v = b_o = 0. Compute row 1 of each
  head by hand with `math.exp` from the two scaled scores. Row 0 must equal v_0 per head.
- **Head independence (TL-11):** W_o = I and b_o = 0, so out == Y. Changing the head-1 column block
  of W_q/W_k leaves `out[..., 0:hd]` bitwise unchanged and changes `out[..., hd:2hd]`. Zeroing the
  head-1 columns of W_v/b_v makes `out[..., hd:2hd] == 0`.
- **H=1:** equals a matrix-form single-head textbook reference written inline in the test,
  `softmax(QKᵀ/√d masked) V W_o + b_o`.
- **Mask injection:** with `mask_fn=shifted_causal_mask`, out_t changes for t < T−1, and
  `out_{T−1}` is identical (the last row already sees every key).
- **Source guard:** AST-scan the `reference_attention_forward` function body. Fail on any attribute
  or name `reshape`, `transpose`, `swapaxes`, `moveaxis`, `einsum`, `T` (`.T`), `inf`. This keeps the
  oracle independent of the bookkeeping it exists to check. Include a test-the-test: the scanner flags
  a snippet that uses `.transpose(`.
- Validation: `d % H != 0` raises `ValueError`, float32 x raises `TypeError`, and x and params are
  bit-identical after a call.

---

## 6. Engineer B: vectorized `attention.py` forward + backward + gradcheck

### 6.1 Milestones (commit each one; order is binding, §9)
- **B1 (needs A1):** constructor, `_split_heads`/`_merge_heads`, `init_params`, forward + constructor/
  init/forward-shape tests. Commit: `Phase 2 [B1]: CausalSelfAttention constructor and vectorized forward`.
- **B2 (needs A2), the HARD GATE:** vectorized-vs-oracle forward match tests. Commit: `Phase 2 [B2]:
  vectorized forward matches loop-over-heads oracle`.
- **B3:** backward + tiny-config gradcheck + the TL-9 protocol. **Do not commit B3 before B2 is
  committed**, even if the backward is ready earlier. Keep it local. Commit: `Phase 2 [B3]:
  hand-derived attention backward + module gradcheck`.
- **B4:** test-the-tests, the backward-vs-oracle check, robustness, and extra coverage. Commit:
  `Phase 2 [B4]: oracle/gradcheck test-the-tests, robustness, extra coverage`. Then hand the TL the
  results block (§12) with the full gradcheck table.

### 6.2 Spec
§3.4 exactly, plus TL-9, TL-10, TL-11, TL-12, TL-13 and TL-15.

### 6.3 Acceptance criteria (`tests/test_attention.py`, `_FIXTURE_SEED = DEFAULT_SEED + 202`, < 10 s)
Construction:
- `CausalSelfAttention("attn", 8, 3)` raises `ValueError`. `head_dim == d // H`, and
  `scale == 1.0 / math.sqrt(head_dim)` exactly. `param_keys()` is exactly the 8 TL-10 keys.
  `from_config(tiny_config(11))` gives H=2, hd=4. The default `mask_fn is causal_mask`.
- `init_params(make_rng(_FIXTURE_SEED))` gives W of shape (d,d) and b of shape (d,), all float64,
  all C-contiguous. Biases are 0. `W_q != W_k`. Sample std of the W entries is within a loose band of
  `INIT_STD` at d=64.

**Oracle forward match (S2-07 hard gate):** `max|out_vec − reference_attention_forward(...)| ≤
ROUNDOFF_ATOL` (measured 2.2e-16 tiny, 8.9e-16 at d=64) on all of:
- tiny d=8, H=2, B=2, T=4;
- default d=64, H=4, B=2, T=7;
- d=8, T=5, H ∈ {1, 2, 4, 8} (hd = 8, 4, 2, 1);
- T=1;
- B=3.

The tiny case is run with **both** `causal_mask` and `shifted_causal_mask`, which shows that the
`mask_fn` plumbing is identical in the two implementations.

**Module gradcheck (tiny: d=8, H=2, B=2, T=4; non-init fixture; gradcheck default seed):**
- `rel_err < GRADCHECK_TOL` on q.W, q.b, k.W, v.W, v.b, o.W, o.b and x (prototype ≤ 6e-11).
- `attn.k.b` uses the TL-9 protocol (a)–(d).
- A helper prints the full 9-row table (`name, shape, rel_err, max_elem_rel_err, max_abs_err`) and
  the test emits it, so `pytest -s` shows it. The table goes into the results block. **Omitting
  max_elem_rel_err for any tensor voids sign-off (PO condition).**
- A second gradcheck at d=16, H=4, B=1, T=3 catches head-count-dependent bugs.

**Backward vs. oracle (P1):** run gradcheck with an adapter forward that returns the **oracle's**
output plus the vectorized cache: `def fwd(x, p): _, c = attn.forward(x, p); return
reference_attention_forward(x, p, H), c`. The FD side then differentiates the oracle, while the
analytic side is B's backward. Gate as above, including TL-9 for k.b (prototype ≤ 1.4e-10 on the
rest, 0.2 s).

**Test-the-tests** (local broken variants in the test file; monkeypatch `attention._split_heads` /
`_merge_heads` with pytest's `monkeypatch`):
- (i) **Interleaved heads** (F7): split `a.reshape(B,T,hd,H).transpose(0,3,1,2)` with merge
  `y.transpose(0,2,3,1).reshape(B,T,d)`. **Assert that gradcheck passes** on all non-k.b tensors,
  documenting that gradcheck is blind to this bug class, **and** assert that the oracle match fails
  (`max abs > 1e-3`; measured 0.61).
- (ii) **No-transpose split**: `a.reshape(B,H,T,hd)` with merge `y.reshape(B,T,d)`. The oracle match
  fails (measured 1.6).
- (iii) **Dropped fan-out**: d_x = dx_v only. The x gradcheck fails.
- (iv) **Missing scale in backward**: dS not multiplied by scale. Gradcheck fails on q.W and k.W.
- (v) The TL-9 dropped-term softmax backward, given in (d) above.

Robustness:
- forward and backward are finite at T ∈ {1, 2, 4, 7}, including row 0 of the mask;
- `cache["P"]` is exactly 0.0 above the diagonal, and each row sums to 1 within `ROUNDOFF_ATOL`;
- params, x and d_out are bit-identical after forward and backward;
- float32 x, params or d_out raise `TypeError`;
- `x.ndim != 3` or the wrong d raises `ValueError`.

---

## 7. Engineer C: `tests/test_causal_mask.py` (C5 isolated, shifted test-the-test, T=1)

C is the **independent verifier**: C writes no production code and checks causality from outside
both implementations. C codes against the frozen interfaces in §3 from day one.

### 7.1 Milestones (commit each one)
- **C1 (needs A1 + A2):** the C5 helper; C5 + the shifted test-the-test on the **oracle**; the
  strict-mask documentation test. Commit: `Phase 2 [C1]: C5 numerical-Jacobian check + shifted-mask
  test-the-test on oracle`.
- **C2 (needs B1):** C5 + shifted on the **vectorized** module, regression guards, and the d=64 spot
  check. Commit: `Phase 2 [C2]: C5 + shifted-mask test-the-test on vectorized attention`.
- **C3 (needs B3):** the T=1 suite, backward causality, and the masked-softmax rank. Commit:
  `Phase 2 [C3]: T=1 degenerate case, backward causality, masked-softmax rank`. Then hand the TL the
  results block.

### 7.2 Spec
```python
NONVACUITY_MIN = 1e-6                  # test-local; s<=t blocks must exceed it
def _c5_blocks(f, x_td) -> J4          # J = gradcheck.numerical_jacobian(f, x_td) with x_td (T,d);
                                       # returns J.reshape(T, d, T, d), J4[t,:,s,:] = d out_t / d x_s
def _assert_c5(J4)                     # raises AssertionError (with the offending (t,s) and value) if
                                       #   max_{s>t} |J4[t,:,s,:]| > ROUNDOFF_ATOL, or
                                       #   min_{s<=t} ||J4[t,:,s,:]||_F <= NONVACUITY_MIN
```
Use B=1 and wrap as `f = lambda z: attn.forward(z[None], params)[0][0]` (or the oracle). Build the
Jacobian with the **existing** `nanogpt.gradcheck.numerical_jacobian`, not a private FD loop. Use the
non-init fixture per TL-13 with `_FIXTURE_SEED = DEFAULT_SEED + 203`.

### 7.3 Acceptance criteria (< 10 s)
**C5, correct mask (T=4, d=8, H=2):**
- `_assert_c5` passes on the vectorized module **and** on the oracle.
- Record the measured future max (expected 0.0) and past min (prototype 0.13).

**Shifted test-the-test (the required adversary):**
- `CausalSelfAttention(..., mask_fn=shifted_causal_mask)`, and separately the oracle with that
  `mask_fn`, makes `_assert_c5` **raise** (`pytest.raises(AssertionError)`). The bug is actually
  reintroduced, not inspected.
- The leak structure is asserted: `‖J4[t,:,t+1,:]‖_F > NONVACUITY_MIN` for every t < T−1 (prototype
  max 0.63), and every block with s ≥ t+2 is `≤ ROUNDOFF_ATOL`. This proves the check catches exactly
  the one-extra-token leak.

**Wrong adversary, kept separate (P0 per PO):** a test-local strict mask `j < i` yields NaN in row 0
of the forward (under `np.errstate(invalid="ignore")`). The docstring explains that this is the
diagonal-masking interpretation the plan rejects, and that it does **not** count as the C5
test-the-test.

**Backward causality (P1):**
- Build the analytic Jacobian from `backward` with one-hot upstream gradients e_{t,i}. For every
  t, `d_x[s]` is `≤ ROUNDOFF_ATOL` for all s > t.
- The analytic Jacobian matches the numerical J with `rel_err < GRADCHECK_TOL`.

**Batch independence (P2):** with B=2, ∂out[b]/∂x[b'] is exactly 0 for b ≠ b'.

**d=64, H=4, T=6 spot check (P2):** `_assert_c5` passes on the vectorized module (prototype 0.04 s).

**T=1 (B=2, T=1, non-init fixture):**
- Forward and backward are finite. The anchor `out == (xW_v+b_v)W_o+b_o` holds within ROUNDOFF_ATOL.
- `np.all(g == 0)` for the analytic dW_q, db_q, dW_k, db_k. Exact zeros are required, not merely
  small values.
- Gradcheck reports `rel_err == 0.0` **exactly** for those four tensors, and `report.passed` is True
  over all 9 tensors.
- **Paired T=4** (same weights, new x): ‖dW_q‖, ‖db_q‖, ‖dW_k‖ > NONVACUITY_MIN. b_k stays
  `≤ ROUNDOFF_ATOL`, per TL-9 / R14, with a comment explaining why.
- **Disconnection adversary:** use a test-local forward that ignores `params["attn.q.W"]` and
  `params["attn.q.b"]`, calling `attn.forward` with zeros substituted for them. At T=4, the FD
  gradient of W_q through that forward (via `numerical_jacobian` on a scalarized objective) is 0, so
  the paired non-zero helper **rejects** it. This shows the pairing distinguishes "zero because
  softmax over one key is constant" from "zero because Q is disconnected".

**Masked-softmax rank (P2, F5):**
- For each causal row i of `cache["P"][0, h]`, build that row's softmax Jacobian from
  `ops.softmax_backward` applied to basis vectors, as in C4.
- Its SVD rank, at threshold `C3_SVD_REL_TOL·s_max`, equals i. Row 0 has J ≡ 0, so its rank is 0.

**Regression guards:**
- `CausalSelfAttention("attn", 8, 2).mask_fn is causal_mask`.
- A text scan of `nanogpt/*.py` finds the name `shifted_causal_mask` **only** in
  `attention_mask.py`. A future regression that silently makes the adversary a default is caught
  automatically.

---

## 8. Git protocol (S2-00; every engineer follows this exactly)

The repo is already initialized on branch `master`, with Sprint 1 history `283c20f`..`74044c4`.
There is no global and no local `user.*` config, and none is ever written.

**GIT RULES** (follow exactly; the repo already has Sprint 1 history on it):
- Identify yourself **only** through environment variables on each git command, never through
  `git config`. Concurrent config writes from teammates would corrupt the shared `.git/config`.
  Prefix every `git add` and `git commit` with
  `GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local"`.
  Roles and emails are exactly: `Engineer A` / `engineer-a@nanogpt-numpy.local`, `Engineer B` /
  `engineer-b@nanogpt-numpy.local`, `Engineer C` / `engineer-c@nanogpt-numpy.local`, `Tech Lead` /
  `tech-lead@nanogpt-numpy.local`. These match the Sprint 1 history.
- Only `git add` the exact files you own. Never use `git add -A`, `git add .`, or any path outside
  your assignment. Teammates may be committing concurrently, and a broad add can scoop up their
  unstaged work.
- End every commit message with a blank line, then exactly:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
- If `git add` or `git commit` fails with an "Unable to create '.git/index.lock'" error, a teammate
  is committing at that moment. Wait about 2 seconds and retry, up to 3 times. This is expected, not
  a real failure.
- Never run `git push`, `git reset --hard` or `git rebase`, and never touch any git config file.
  Commit directly on the current branch.

**TL additions (consistent with the rules above):**
- Run git from the **Bash tool (Git Bash)**. The `VAR=value git ...` prefix syntax does not work in
  PowerShell.
- **Commit with an explicit pathspec** so that a teammate's staged-but-not-yet-committed files can
  never end up in your commit: `git add <your files>` followed by `git commit -m "..." -m "..." --
  <your files>`. The pathspec form commits only those paths.
- Before you commit, check that your own test file is green, and run `git diff --cached --name-only
  -- <your files>` to see what you are committing. Never commit a red or skipped gating test. If a
  dependency hasn't landed, don't commit the test that needs it yet.
- Use subject lines of the form `Phase 2 [<milestone>]: <what>` (§5–§7). Each milestone gets one
  commit or a few, in small steps.
- Template:
  ```bash
  cd /c/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy
  GIT_AUTHOR_NAME="Engineer B" GIT_AUTHOR_EMAIL="engineer-b@nanogpt-numpy.local" \
  GIT_COMMITTER_NAME="Engineer B" GIT_COMMITTER_EMAIL="engineer-b@nanogpt-numpy.local" \
    git add nanogpt/attention.py tests/test_attention.py
  GIT_AUTHOR_NAME="Engineer B" GIT_AUTHOR_EMAIL="engineer-b@nanogpt-numpy.local" \
  GIT_COMMITTER_NAME="Engineer B" GIT_COMMITTER_EMAIL="engineer-b@nanogpt-numpy.local" \
    git commit -m "Phase 2 [B2]: vectorized forward matches loop-over-heads oracle" \
               -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" \
               -- nanogpt/attention.py tests/test_attention.py
  ```
- `.venv/`, `__pycache__/`, `*.pyc` and `.pytest_cache/` are already gitignored. Never force-add them.
- **TL fixups** to an engineer-owned file are their own commits, authored as `Tech Lead`, with the
  subject `TL fixup: <file>: <what>`. They are never folded into an engineer's commit, and an
  engineer never amends a TL fixup. Nobody uses `--amend` on a commit that is not the very last
  commit and their own.

---

## 9. Dependency protocol and commit ordering

```
A1 masks ──┬─> B1 forward ──> C2 (C5 on vectorized)
           └─> A2 oracle ──┬─> B2 ORACLE MATCH (hard gate) ──> B3 backward+gradcheck ──> B4 ──> C3 (T=1, bwd causality)
                           └─> C1 (C5 + shifted on oracle)
```
- **B3 is never committed before B2.** This is the PM's top sequencing risk: a reshape bug with a
  self-consistent backward gets a false green gradcheck (F7 proves it can). The git history must show
  `[A2] < [B2] < [B3]`, and the TL checks this at sign-off.
- **The shifted adversary exists from A1**, before any C5 check is committed, so C5 is never
  committed without its test-the-test. C1 and C2 each commit the C5 check and its shifted failure
  together.
- **If A2 is late:** B keeps building the backward locally and commits only B1. C writes the
  vectorized-side C5 (C2) first.
- **If B1 is late:** C completes C1 on the oracle.
- No fallback shims are ever committed. If you need a mask before A1 lands, inline it **locally,
  uncommitted**. If a `TODO(SPRINT2-…)` marker is ever needed, the guard extension in §10 makes it
  un-shippable.
- Interface changes to §3 need TL approval. Internals do not.

---

## 10. Timeline (≈ 3 engineering days) and TL integration duties

| Day | A | B | C |
|---|---|---|---|
| 1 AM | **A1** masks (commit) | B1 constructor, split/merge, forward | C5 helper, strict-mask doc test |
| 1 PM | **A2** oracle + anchors (commit) | B1 commit, then **B2** oracle gate (commit) | **C1** on oracle (commit) |
| 2 | A3 source guard, remaining anchors; review B2 numbers | **B3** backward + gradcheck + TL-9 (commit) | **C2** on vectorized (commit) |
| 3 | results block | **B4** test-the-tests, adapter check (commit); results block | **C3** T=1, backward causality, rank (commit); results block |
| 3–4 | | | **TL integration and sign-off** |

TL integration (each item is its own commit, as `Tech Lead`):
1. **TL fixup** `tests/test_conventions.py`: widen the marker check from `TODO(SPRINT1-` to
   `TODO(SPRINT`, with its existing test-the-test updated, so that any Sprint 2 shim is caught.
2. Review each engineer's milestone as it lands. Fixes to engineer files go in separate `TL fixup:`
   commits.
3. Run the sign-off checklist (§11). Append the PROGRESS.md Phase 2 entry, then commit `Sprint 2
   integration sign-off` on a fully green tree.

---

## 11. Integration and sign-off checklist (TL runs this; numbers are re-measured, not copied)

1. `.venv/Scripts/python.exe -m pytest tests/ -q --durations=0` is all green, < 60 s in total, and
   every file is < 10 s. Two consecutive runs give identical results. The corpus SHA-256 still matches
   Phase 0.
2. The TL re-measures with its own throwaway script (not committed):
   - the tiny-config gradcheck table, giving **rel_err and max_elem_rel_err (and max_abs_err) for all
     9 tensors**;
   - the same table for the backward-vs-oracle adapter;
   - the TL-9 numbers (a)–(d) for b_k;
   - the oracle-match max abs at tiny and default H=4;
   - the C5 future max and past min (correct mask), and the shifted leak at s = t+1 versus s ≥ t+2;
   - the T=1 values (four exact zeros, rel_err == 0.0) and the paired T=4 norms;
   - the F7 interleaved-split numbers.
3. All test-the-tests are confirmed by their passing assertions: shifted mask (both implementations),
   interleaved and no-transpose split, dropped fan-out, missing scale, the TL-9 dropped term, and
   Q disconnection.
4. `grep -rn "TODO(SPRINT" nanogpt/` finds nothing, and the guard is green. `nanogpt/attention*.py`
   import every constant from `config.py` and nothing from torch/jax/tf/autograd. `shifted_causal_mask`
   is referenced only in `attention_mask.py` and in tests.
5. **No Phase 3+ files exist:** `mlp.py`, `block.py`, `model.py`, `param_count.py`, `flops.py`,
   `optim.py`, `kv_cache.py`, `sampling.py`, `checkpoint.py`, `train.py`, `generate.py`,
   `tests/test_gradients.py`, nor any KV-cache argument in `attention.py`.
6. Git checks:
   - `git log --format='%h %an <%ae> | %s'` shows each engineer's milestones under their own role,
     in the §9 order (`[A2]` < `[B2]` < `[B3]`).
   - TL fixups are separate commits.
   - `283c20f`..`74044c4` still have their original hashes (ancestry unchanged).
   - `git status` is clean, and no `.venv`, `__pycache__` or `.pytest_cache` path is tracked.
7. The PROGRESS.md Phase 2 entry is appended per §12, with TL-9 flagged for PO ratification and C5
   listed as **open (full-model re-run owed in Phase 3)**. Then commit the sign-off as `Tech Lead`.

---

## 12. Results block and PROGRESS template

**Results block each engineer hands to the TL:**
```
Module/tests: <files>   Owner: <A|B|C>   Commits: <short hashes + subjects>
Gradcheck (gated norm-wise rel_err < 1e-6) | max_elem_rel_err | max_abs_err : one row per tensor, ALL tensors
TL-9 b_k: analytic max=<>, FD max_abs_err=<>, invariance max=<>, dropped-term max=<>
Oracle match max abs: tiny=<> default(H=4,d=64)=<> H-sweep=<>
C5: future max=<> past min=<> | shifted: leak(t,t+1) min/max=<> , s>=t+2 max=<>
T=1: exact zeros {q.W,q.b,k.W,k.b}=<bool> rel_err=<> | paired T=4 norms=<>
Test-the-test: <variant> -> FAILED as expected (<metric>=<value>)
Timing: <file> <seconds>
Deviations from SPRINT_2_PLAN.md: <none | what + why>
```

**PROGRESS.md Phase 2 entry (TL; append-only):**
```
## Phase 2 — Causal Multi-Head Self-Attention  [DONE <date>]
Process (S2-00): commit-as-you-go per role via env-var identities; TL fixups separate; deviations: <>
Owners/commits: A <hashes>, B <hashes>, C <hashes>, TL fixups <hashes>
Passed: suite <n> passed in <s>; per-file timings; determinism x2
Gradcheck table (rel_err | max_elem_rel_err | max_abs_err), all 9 tensors + adapter table
Oracle match; C5 isolated numbers; shifted-mask evidence; T=1 values; rank check
Interpretations/deviations: R12 oracle/mask in own files; R13 test files; R14 PO T>1 b_k correction;
  TL-9 b_k structural-zero gate (FOR PO RATIFICATION); TL-10 separate Q/K/V/O; carried: TL-1, R4, R5, R6
Open items: C5 full-model re-run owed in Phase 3 (NOT retired); weight-tying groundwork (gradcheck
  contiguous wte.W, never a transposed view); TL-9 also applies to every block's b_k in C6
Next: Phase 3 — Transformer Block & Full Model Assembly
```

---

## 13. Explicitly out of scope for Sprint 2 (do not build)

`mlp.py`, `block.py`, `model.py` (embeddings, L blocks, final LN, tied lm_head), `param_count.py`
(C1), C2, `tests/test_gradients.py` (C6), C7 weight tying, `kv_cache.py` (C8) or any past-K/V
argument, `flops.py` / FLOP hooks (C9), `optim.py` (AdamW anchor), training, sampling, checkpoints,
the full-model stability pass (Phase 6), GPT-2 weights (Phase 7), and performance work. **C5 on the
full assembled model is a Phase 3 obligation.** The isolated pass in this sprint does not retire it.
**Phase 3 note (from TL-9):** C6's full-model gradcheck will hit the same structural zero for every
block's `attn.k.b`, and the same protocol applies there.
