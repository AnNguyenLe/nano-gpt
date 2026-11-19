# Sprint 3 — Technical Plan: Phase 3 (Transformer Block & Full Model Assembly)

**Author:** Tech Lead (Opus 5.5). **Inputs:** PM backlog S3-01..S3-10, PO priorities (P0/P1/P2),
`nanoGPT_build_requirements.md` (the spec), `nanoGPT_execution_plan.md` §Phase 3 + §8, and
`PROGRESS.md` (Phase 0–2 state, HEAD `65ca41e`, 256 tests green).

**Scope: Phase 3 only.** `mlp.py`, `block.py`, `model.py`, `param_count.py`, and the checks C1, C2,
C5 (full-model re-run), C6, C7. Everything in Phase 4+ (C8, C9, AdamW, training, sampling,
generation, checkpoints, GPT-2 weights) is out of scope (§13).

Every number quoted as "measured" below comes from a TL throwaway prototype run against the
committed Sprint 1+2 modules (not committed, not a deliverable). Engineers must reproduce them in
their own tests; nobody copies them.

---

## 1. Sprint 3 Definition of Done

1. `nanogpt/mlp.py`, `nanogpt/block.py`, `nanogpt/model.py`, `nanogpt/param_count.py` exist and
   follow the §3 interfaces exactly.
2. **C1:** formula N = Vd + Td + L(12d² + 13d) + 2d equals the live unique-array count on tiny,
   default and max configs; GPT-2 small formula = 124,439,808.
3. **C2:** |loss_init − ln V| < 0.1 (V = 55, real tokenizer, real corpus tokens) on the default
   config, with a failing wrong-init adversary.
4. **C6:** full-model gradcheck on tiny (d=8, L=2, H=2, T=4), all 36 parameter tensors + the
   embedding-stream input, norm-wise rel_err < 1e-6, **except** every `h.{l}.attn.k.b`, which is
   gated by the TL-9 (a)–(d) protocol **for every layer l** (PO ratification condition).
5. **C7:** tied gradient = input-side + output-side; each partial independently confirmed by FD;
   accumulation via `+=` into one array; gradcheck only on the contiguous `wte.W`.
6. **C5 (full model):** no future leak on the assembled model; the shifted `j ≤ i+1` mask substituted
   into **all** blocks and into **each block individually** is caught.
7. **Commit discipline (P0 hard gate):** every Sprint 3 commit is green when its own tree is
   checked out alone. The TL verifies this per commit at sign-off (§11 item 6), not by self-report.
8. Suite green, < 60 s total, every file < 10 s, deterministic over two runs; PROGRESS.md Phase 3
   entry appended.

---

## 2. Reconciliation log and Tech Lead rulings

### 2.1 PM backlog vs. PO priorities vs. spec

| ID | Topic | Inputs said | Ruling |
|---|---|---|---|
| R15 | MLP key names | PM: "W1/b1/W2/b2" | Keys are `{prefix}.fc.W/.b` (d→4d) and `{prefix}.proj.W/.b` (4d→d). Same tensors, GPT-2 naming. |
| R16 | GPT-2 small check (PO P1) | "verified to match a known case (GPT-2 small → 124,439,808)" | **Formula-only.** A live GPT-2-small float64 model is ~1 GB; we do not instantiate it. Live-vs-formula equality is checked on tiny/default/max configs. |
| R17 | C6 "plus x" (PM S3-06) | "every parameter tensor … plus x" | The model input is **integer token ids** (not differentiable; `d_x = None`). The float input is checked one level down: `forward_from_embedding(h0, ·)` is gradchecked w.r.t. `h0` (TL-18). Block-level `x` is gradchecked in `test_block.py`. |
| R18 | P2 extras | robustness sweeps; param_count CLI | **Not in this sprint's DoD.** A single d=64 MLP spot check (PM S3-01) is kept because it is cheap. No CLI. |
| R19 | Ownership split | Suggested: B owns block+model+param_count+C2 | **Rebalanced (TL-23):** C1 and C2 move to Engineer A. B owns only the critical path (block.py, model.py). |

### 2.2 Technical rulings (TL-16 .. TL-23)

**TL-16. Model API (frozen, §3.4).** `GPT` exposes three layers of entry point so that C5, C6 and C7
can each probe the exact sub-map they need without monkeypatching: `embed` (ids → h0),
`forward_from_embedding` (h0 → logits), `forward` (ids → logits), plus `forward_loss`. Each has its
backward counterpart.

**TL-17. Tied head representation.** The params dict holds **one** array, `"wte.W"` (V, d),
C-contiguous, owning its data. The head computes `logits = y @ params["wte.W"].T` — the `.T` is a
transient operand of one matmul and is **never stored** in `params`, in the cache as a parameter, or
handed to gradcheck. There is no `lm_head.W` key and no head bias key in `params`.
`GPT.named_parameters(params)` returns a *per-role* listing that includes `"lm_head.W"` mapped to
**the same object** `params["wte.W"]` (identity, not a copy, not a transpose); this is what C1's live
counter walks, so the tie is real and must be de-duplicated by identity.

**TL-18. C6 float-input coverage.** Since ids are integers, C6 covers the continuous input by
gradchecking `forward_from_embedding` w.r.t. `h0` (all params except `wpe.W`, which that sub-map does
not read). Fixture `h0` must not be drawn from `make_rng(DEFAULT_SEED)` (Sprint 1 seed-collision
watch item: R would equal the fixture draw).

**TL-19. TL-9 for every layer.** Softmax shift-invariance makes `h.{l}.attn.k.b` an exact structural
zero for every l (b_k adds a row-constant q_i·b_k to the scores of row i). The literal rel_err gate
is never applied to these tensors. The gated-exempt set must be **exactly**
`{f"h.{l}.attn.k.b" for l in range(L)}`; a test asserts this so no other tensor is quietly exempted.
Measured on tiny/non-init fixture: (a) analytic max|db_k| = 1.7e-15 (l=0), 2.6e-16 (l=1);
(b) FD max_abs_err = 4.4e-11 / 5.4e-18 (loss objective), 8.0e-10 / 2.7e-10 (logits objective);
(c) invariance max|logits(b_k + 10·N) − logits| = 2.6e-15 / 1.4e-15;
(d) dropped-term softmax backward: max|db_k| = 16.3 / 6.0.

**TL-20. C7 decomposition is verified independently, not tautologically.** `GPT.backward` takes two
keyword-only instrumentation flags, `include_input_side=True, include_output_side=True`. With both
True it is the production backward. The partials from the flags are checked against
**finite differences of independent sub-maps**: the output-side partial against FD of
`forward_from_embedding` w.r.t. `wte.W` with `h0 = embed(ids)` held fixed; the full gradient against
FD of `forward`; the input-side partial against (FD_full − FD_out). The partial sum identity alone is
not sufficient evidence (it is true by construction).

**TL-21. C5 on the full model.** `GPT(cfg, mask_fn=...)` accepts either one callable (used by every
block) or a sequence of exactly L callables (one per block). The adversary is substituted only
through this constructor argument, never by monkeypatching. Measured leak structure (T=4, L=2): with
all blocks shifted the leak reaches exactly `s = t + L` (|J| block at (0,2) = 0.55, (0,3) = 0.0); with
only block l shifted it reaches exactly `s = t + 1`. Causal: every s > t block is exactly 0.0, and
token substitution at s leaves logits[:, t < s] bit-identical (measured 0.0).

**TL-22. Commit-green verification (the S3-09 / PO P0 gate).** Before each commit, an engineer runs
the full suite on **HEAD's committed tree plus only their own staged files**, materialised via
`git archive HEAD` + `git checkout-index` (recipe in §8.2; TL-verified on this machine: 256 passed
from a clean archive). Running against the shared working tree does not count.

**TL-23. Ownership.** B's critical path (block → model forward → model backward) gates everyone. Moving
C1/C2 to A (who finishes mlp.py on day 1) shortens B's queue and lets C1/C2 run in parallel with C6/C7.

---

## 3. Frozen interfaces and formulas (all engineers code to this)

### 3.1 Conventions restated (import from `nanogpt.config`, never redefine)

`DTYPE=float64`, `INIT_STD=0.02`, `BIAS_INIT=0.0`, `LN_GAMMA_INIT=1.0`, `LN_BETA_INIT=0.0`,
`LN_EPS=1e-5`, `MLP_HIDDEN_MULT=4`, `LINEAR_HAS_BIAS=True`, `LM_HEAD_HAS_BIAS=False`, `FD_STEP=1e-5`,
`REL_ERR_FLOOR=1e-8`, `GRADCHECK_TOL=1e-6`, `ROUNDOFF_ATOL=1e-12`, `INIT_LOSS_TOL=0.1`,
`DEFAULT_SEED=1337`, `rel_err` (norm-wise, TL-1), `make_rng`, `check_float64`, `ModelConfig`,
`tiny_config(V)` = (T=4, d=8, L=2, H=2). `tests/test_conventions.py` fails on any literal `0.02`,
`1e-5`, `1e-6`, `1e-8`, `1e-10` in `nanogpt/*.py`, so new modules must import these.
`shifted_causal_mask` must not be referenced anywhere in `nanogpt/` except `attention_mask.py`
(existing guard in `tests/test_causal_mask.py`).

Module contract (unchanged): `forward(x, params) -> (out, cache)`, `backward(d_out, cache) ->
(d_x, d_params)`; `d_params` is a new dict keyed by global names; nothing mutates `params`/`x`/`d_out`.

### 3.2 `nanogpt/mlp.py` (Engineer A)

```python
class MLP:
    def __init__(self, prefix: str, d: int)      # positive int d, else ValueError
    # self.fc   = ops.Linear(f"{prefix}.fc",   d, MLP_HIDDEN_MULT*d, bias=LINEAR_HAS_BIAS)
    # self.proj = ops.Linear(f"{prefix}.proj", MLP_HIDDEN_MULT*d, d, bias=LINEAR_HAS_BIAS)
    def param_keys(self) -> list   # [fc.W, fc.b, proj.W, proj.b]
    def init_params(self, rng) -> dict   # fc then proj, same rng, in that order
    def forward(self, x, params)  -> (out, cache)
    def backward(self, d_out, cache) -> (d_x, d_params)   # exactly the 4 keys
```
Forward: `h = x W_fc + b_fc` (…, 4d); `a = GELU_tanh(h)` (`ops.gelu_forward`); `out = a W_proj + b_proj`.
Backward: `da, g_proj = proj.backward(d_out)`; `dh = ops.gelu_backward(da)` (GELU′ =
0.5(1+t) + 0.5·h·(1−t²)·c·(1+3a·h²), t = tanh(c(h + a h³))); `dx, g_fc = fc.backward(dh)`.
No GELU/Linear math re-implemented locally.

### 3.3 `nanogpt/block.py` (Engineer B)

```python
class Block:
    def __init__(self, prefix: str, d: int, H: int, mask_fn=causal_mask)
    # self.ln1  = LayerNorm(f"{prefix}.ln1", d)
    # self.attn = CausalSelfAttention(f"{prefix}.attn", d, H, mask_fn=mask_fn)
    # self.ln2  = LayerNorm(f"{prefix}.ln2", d)
    # self.mlp  = MLP(f"{prefix}.mlp", d)
    @classmethod from_config(cls, cfg, prefix, mask_fn=causal_mask)
    def param_keys(self) -> list   # ln1(γ,β), attn(8, TL-10 order), ln2(γ,β), mlp(4)  = 16 keys
    def init_params(self, rng) -> dict   # ln1, attn, ln2, mlp, in that order, same rng
    def forward(self, x, params)  -> (out, cache)
    def backward(self, d_out, cache) -> (d_x, d_params)   # exactly the 16 keys
```
Pre-LN residual block:
```
x1  = x  + attn(LN1(x))
out = x1 + mlp(LN2(x1))
```
Hand-derived backward (restated so nobody re-derives it differently):
```
dz2, g_mlp  = mlp.backward(d_out)        ;  du2, g_ln2  = ln2.backward(dz2)
dx1         = d_out + du2                # residual: identity path + branch path, SUM
dz1, g_attn = attn.backward(dx1)         ;  du1, g_ln1  = ln1.backward(dz1)
dx          = dx1 + du1                  # residual again, SUM
d_params    = merge of g_ln1, g_attn, g_ln2, g_mlp via ops.accumulate_grads into a fresh dict
```
`mask_fn` is passed straight into `CausalSelfAttention`; `block.attn.mask_fn is mask_fn` must hold.
`block.py` contains no LayerNorm/softmax/GELU arithmetic of its own (no `np.mean`, `np.sqrt`,
`np.exp`, `np.tanh`); it only composes the four modules.

### 3.4 `nanogpt/model.py` (Engineer B)

```python
class GPT:
    def __init__(self, cfg: ModelConfig, mask_fn=causal_mask)
    # mask_fn: a callable (used for all L blocks) OR a list/tuple of exactly cfg.L callables
    #          (block l uses mask_fn[l]); anything else -> ValueError.
    # self.wte = ops.Embedding("wte", cfg.V, cfg.d); self.wpe = ops.Embedding("wpe", cfg.T, cfg.d)
    # self.blocks = [Block(f"h.{l}", cfg.d, cfg.H, mask_fns[l]) for l in range(cfg.L)]
    # self.ln_f = LayerNorm("ln_f", cfg.d)
    def param_keys(self) -> list   # wte.W, wpe.W, h.0.* (16), ..., h.{L-1}.* (16), ln_f.gamma, ln_f.beta
    def init_params(self, rng) -> dict   # same order as param_keys, one rng; 4 + 16L tensors
    def named_parameters(self, params) -> dict
        # every role, INCLUDING "lm_head.W": params["wte.W"]  (the SAME object; TL-17)
    def embed(self, idx, params) -> (h0, cache)                 # ids (B,T) int -> (B,T,d)
    def embed_backward(self, d_h0, cache) -> dict               # {"wte.W": input-side, "wpe.W": ...}
    def forward_from_embedding(self, h0, params) -> (logits, cache)
    def backward_from_embedding(self, d_logits, cache) -> (d_h0, d_params)
        # d_params: every key EXCEPT "wpe.W"; "wte.W" holds the OUTPUT-side contribution only
    def forward(self, idx, params) -> (logits, cache)           # = embed then forward_from_embedding
    def backward(self, d_logits, cache, *, include_input_side=True, include_output_side=True)
        -> (None, d_params)                                     # all 4+16L keys, always
    def forward_loss(self, idx, targets, params) -> (loss, cache)   # via loss.cross_entropy_forward
    def backward_loss(self, d_loss, cache) -> (None, d_params)
```
Forward:
```
h0     = wte.W[idx] + wpe.W[arange(T)]            # ops.Embedding for both; wpe broadcast over B
h_l    = Block_l(h_{l-1}),  l = 0..L-1
y      = LN_f(h_L)
logits = y @ params["wte.W"].T                    # tied head, NO bias (LM_HEAD_HAS_BIAS=False)
```
Backward:
```
g_out  = d_logits.reshape(-1,V).T @ y.reshape(-1,d)          # (V,d) output-side contribution
dy     = d_logits @ params["wte.W"]
dh, g  = ln_f.backward(dy); blocks backward in reverse order  # accumulate each into grads
g_pos  = wpe.backward(dh.sum(axis=0))                        # np.add.at inside Embedding
g_in   = wte.backward(dh)                                    # input-side contribution (np.add.at)
grads  : ops.accumulate_grads(grads, {"wte.W": g_out}) then ops.accumulate_grads(grads, g_in)
         -> the second call does grads["wte.W"] += g_in on the SAME array (C7 by construction)
```
Rules: every merge goes through `ops.accumulate_grads`, called as `ops.accumulate_grads(...)`
(module attribute, so C7 can monkeypatch it). No statement of the form `<dict>["wte.W"] = ...` in
`model.py`. When a flag excludes a side, that side is simply not accumulated; `"wte.W"` is always
present (zeros if both sides are excluded). Validation: `idx` is a 2-D integer array with
1 ≤ T ≤ cfg.T (else ValueError); ids out of range raise via `Embedding`. Init uses only the classes'
`init_params` (std = INIT_STD, biases/β = 0, γ = 1). No KV-cache argument, no sampling, no training.

### 3.5 `nanogpt/param_count.py` (Engineer A)

```python
def formula_param_count(V: int, T: int, d: int, L: int) -> int   # N = Vd + Td + L(12d² + 13d) + 2d
def formula_from_config(cfg: ModelConfig) -> int
def live_param_count(named_params: dict) -> int
```
Formula derivation (restate in the docstring): per block, attention 4(d² + d), MLP d·4d + 4d + 4d·d + d
= 8d² + 5d, two LNs 4d → **12d² + 13d**; plus wte Vd (counted once, it is also the head), wpe Td,
final LN 2d. **Preconditions** (asserted at import or call time from `config`, raising if violated):
`MLP_HIDDEN_MULT == 4`, `LINEAR_HAS_BIAS is True`, `LM_HEAD_HAS_BIAS is False`, LN affine. Integer
arithmetic only (Python int, no float).
`live_param_count`: sums `.size` over **unique arrays by identity** (`id`). If two *distinct* objects
share memory (`np.shares_memory`), raise `ValueError` — that is a transposed-view or sliced tie,
which TL-17 forbids.

Expected values: tiny (V=55, T=4, d=8, L=2) **2,232**; default (V=55, T=64, d=64, L=2) **107,712**;
max (V=55, T=128, d=128, L=4) **816,768**; GPT-2 small (V=50257, T=1024, d=768, L=12)
**124,439,808**.

### 3.6 Facts every test relies on (measured by the TL prototype)

| Fact | Value |
|---|---|
| Tiny model tensor count | 4 + 16L = **36** |
| Tiny full-model gradcheck time (logits obj.) / (loss obj.) | 1.45 s / 1.58 s |
| Tiny worst non-k.b rel_err (logits / loss objective), non-init fixture | 9.9e-10 / 4.6e-9 |
| C2 loss − ln 55, real corpus, 3 seeds: default / max / tiny | ≤ 0.042 / ≤ 0.032 / ≤ 0.019 |
| C2 adversary (all 2-D weights std = 1.0): default | +12.2 … +17.1 (fails by > 100×) |
| C7 ‖g_in + g_out − g_full‖_max; ‖g_in‖, ‖g_out‖ | 0.0; 117, 86 |
| TL-9 per layer | see TL-19 |
| C5 leak reach | causal: 0.0 for all s>t; all-shifted: t+L; single-block-shifted: t+1 |

"Non-init fixture" = init params, then W ← W + 0.3·N, b/β ← 0.3·N, γ ← 1 + 0.3·N, all
`np.ascontiguousarray`; at pure init many gradients are degenerate (β=0, γ=1, biases 0) and a
gradcheck there is weaker.

---

## 4. File ownership (no file has two owners)

| File | Owner | Purpose |
|---|---|---|
| `nanogpt/mlp.py` | **A** | MLP module |
| `tests/test_mlp.py` | **A** | MLP gradchecks + test-the-tests |
| `nanogpt/param_count.py` | **A** | C1 formula + live counter |
| `tests/test_param_count.py` | **A** | C1 |
| `tests/test_loss_at_init.py` | **A** | C2 |
| `nanogpt/block.py` | **B** | Transformer block |
| `tests/test_block.py` | **B** | Block gradcheck, TL-9 at block level, residual/mask test-the-tests |
| `nanogpt/model.py` | **B** | GPT assembly |
| `tests/test_model.py` | **B** | Shapes, init, tie structure, mask_fn plumbing, dev gradcheck at L=1 |
| `tests/test_gradients.py` | **C** | **C6** (reserved since Sprint 1, R6/R13) |
| `tests/test_weight_tying.py` | **C** | **C7** |
| `tests/test_causal_mask.py` | **C** (owner since Sprint 2) | **append** a full-model C5 section; existing tests untouched |
| `SPRINT_3_PLAN.md`, `PROGRESS.md`, TL fixups | **TL** | |

Per-file fixture seeds (never `DEFAULT_SEED` itself for float fixtures): test_mlp `+301`,
test_param_count `+302`, test_loss_at_init `+303`, test_block `+311`, test_model `+312`,
test_gradients `+321`, test_weight_tying `+322`, test_causal_mask full-model section `+323`.

---

## 5. Engineer A: `mlp.py`, then C1 (`param_count.py`), then C2

### 5.1 Milestones (one commit each, in this order)

- **A1** `nanogpt/mlp.py` + `tests/test_mlp.py`. No dependencies; start immediately.
- **A2** `nanogpt/param_count.py` + `tests/test_param_count.py`. **Starts only after B2 is committed**
  (`git log --oneline -- nanogpt/model.py` shows `[B2]`): C1's live count needs the final structure.
  You may write `formula_param_count` and its GPT-2 test before that, but commit A2 only with the live
  checks included and green against committed `model.py`.
- **A3** `tests/test_loss_at_init.py` (C2). After B2 is committed (needs only the forward).

### 5.2 Spec — §3.2 and §3.5 exactly.

### 5.3 Acceptance criteria (each file < 10 s)

`tests/test_mlp.py`:
- Gradcheck (harness) tiny: d=8, B=2, T=4, non-init params: fc.W, fc.b, proj.W, proj.b, x all
  rel_err < 1e-6. Spot check d=64, B=1, T=2: same five tensors < 1e-6 (keep under ~3 s).
- Shapes: hidden activation is (…, 4d); `d_params` has exactly the 4 keys; float32 x raises.
- Closed-form anchor: with W_fc = I-padded (d→4d, first d columns identity), b=0, W_proj selecting
  those columns back, `out == GELU_tanh(x)` to ≤ ROUNDOFF_ATOL (computed from the §0 formula in the test).
- **Test-the-test:** (i) a backward that skips `gelu_backward` (passes `da` straight through) fails the
  x and fc.* gradchecks; (ii) a backward that returns `dx` from `fc` with W_proj ignored fails.
  Use `monkeypatch` on the module's `ops` attribute; never commit a broken variant in `nanogpt/`.

`tests/test_param_count.py` (C1):
- `formula_param_count(50257, 1024, 768, 12) == 124_439_808` (formula only, R16).
- For tiny, default and max configs: `live_param_count(GPT(cfg).named_parameters(params)) ==
  formula_from_config(cfg)` exactly (2,232 / 107,712 / 816,768), and `"lm_head.W" in named_parameters`
  with `named["lm_head.W"] is params["wte.W"]`.
- Per-block breakdown: sum of sizes of one block's 16 tensors == 12d² + 13d.
- **Test-the-test:** naive `sum(v.size for v in named.values())` == formula + V·d (double count,
  proves the tie is really present in the listing); a dict with `"lm_head.W": params["wte.W"].T`
  (shares memory, different object) makes `live_param_count` raise `ValueError`; a model with a
  hypothetical head bias (add a `(V,)` array to the listing) makes live ≠ formula.
- Preconditions: monkeypatching `param_count`'s view of `MLP_HIDDEN_MULT` to 3 (or
  `LM_HEAD_HAS_BIAS` to True) makes the formula raise rather than return a wrong number.

`tests/test_loss_at_init.py` (C2):
- V from `load_corpus_and_tokenizer()` (assert V == 55, the Phase 0 value). Default config
  (V=55, T=64, d=64, L=2, H=4); B=4 windows of real corpus ids (inputs and next-char targets);
  params from `GPT.init_params(make_rng(DEFAULT_SEED + s))` for s in {0,1,2}:
  **|loss − ln V| < INIT_LOSS_TOL (0.1)** for every seed. Also tiny and max configs.
- Init statistics: every bias and β exactly 0.0, every γ exactly 1.0, every 2-D weight with ≥ 512
  entries has sample std within ±10% of INIT_STD and |mean| < 0.1·INIT_STD.
- **Test-the-test:** scaling every 2-D weight to std 1.0 makes |loss − ln V| > 1 on the default config
  (measured +12 to +17). Do **not** loosen the tolerance if the real check is tight; fix the init (spec C2).

### 5.4 Process guidance (verbatim, binding)

GIT RULES (the repo has Sprint 1+2 history on it already):
- Identify yourself ONLY via environment variables on each git command, never via `git config`: prefix every git add/commit with GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local".
- Only `git add` the exact files you own — never `git add -A`, `git add .`, or any path outside your assignment.
- End every commit message with a blank line then exactly: Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
- If a git command fails with an index.lock error, a teammate is committing at that exact moment — wait ~2 seconds and retry, up to 3 times.
- Never run `git push`, `git reset --hard`, `git rebase`, or touch any git config file.

SPRINT 2 PROCESS FAILURE TO NOT REPEAT (the PO called a repeat of this "rejection-level"): last sprint, one engineer batched 3 distinct milestones into a single commit built against teammates' not-yet-committed files, leaving several intermediate commits broken if checked out alone (HEAD was still green, but the interior of the history wasn't). This sprint:
- Split your own work into the smallest sensible commits, strictly in dependency order — never batch multiple distinct milestones (e.g. "implement X" and "verify X" and "verify Y") into one commit.
- Before relying on a teammate's file in your own work, confirm it is actually committed (`git log --oneline -- <their path>`), not just present on disk uncommitted. If it isn't committed yet, either wait briefly and retry, or proceed using the documented local-stand-in fallback and say so explicitly in your notes — never silently assume an on-disk file is safely there to depend on.
- Each of your own commits should pass the test suite on its own at the moment you make it, given what's actually committed so far (by you and others) plus your own staged change.

Your identity: `Engineer A` / `engineer-a@nanogpt-numpy.local`. Run the §8.2 pre-commit check before
**each** of A1, A2, A3. A2 and A3 depend on `nanogpt/model.py` `[B2]` being **committed**.

---

## 6. Engineer B: `block.py`, then `model.py` (the critical path)

### 6.1 Milestones (one commit each; order is binding)

- **B1** `nanogpt/block.py` + `tests/test_block.py`. Depends on A1 (`nanogpt/mlp.py`) being
  **committed**. Until then you may develop against a local, uncommitted MLP stand-in in your own
  scratch copy (not in the repo working tree under `nanogpt/mlp.py`, which A owns); say so in your notes.
- **B2** `nanogpt/model.py` (constructor, `param_keys`, `init_params`, `named_parameters`, `embed`,
  `forward_from_embedding`, `forward`, `forward_loss`) + `tests/test_model.py` forward/structure tests.
  Subject must end with `(structure final)`: after B2 the param structure is frozen; any later change
  to keys/shapes needs TL approval and invalidates C1 (notify A).
- **B3** `model.py` backward (`embed_backward`, `backward_from_embedding`, `backward` with the two
  C7 flags, `backward_loss`) + the dev gradcheck in `tests/test_model.py`.

### 6.2 Spec — §3.3 and §3.4 exactly.

### 6.3 Acceptance criteria (each file < 10 s)

`tests/test_block.py` (B1):
- Gradcheck tiny (d=8, H=2, B=2, T=4, non-init params, x from seed `+311`): all 16 tensors + x;
  every tensor except `blk.attn.k.b` rel_err < 1e-6; `blk.attn.k.b` gated by TL-9 (a)–(d) at block
  level (same thresholds as §7.3).
- Residual anchor: with attn.o.W, attn.o.b, mlp.proj.W, mlp.proj.b all zero, `out == x` exactly.
- `Block(..., mask_fn=f).attn.mask_fn is f`; block output with `shifted_causal_mask` differs from
  causal by > 1e-3 (plumbing live). Import the adversary only in the test file.
- **Test-the-test** (monkeypatch, nothing broken committed): (i) dropping the identity term
  `dx1 = du2` fails the x gradcheck; (ii) dropping the second residual `dx = du1` fails x;
  (iii) swapping LN1/LN2 caches fails the ln gradchecks.

`tests/test_model.py` (B2/B3):
- B2: `param_keys()` equals the §3.4 order, 4 + 16L entries; no key contains `lm_head` and no head
  bias key exists; `params["wte.W"]` is C-contiguous with `.base is None`; `named_parameters` maps
  `"lm_head.W"` to the same object. Logits shape (B,T,V), float64; T < cfg.T works; T > cfg.T,
  1-D idx, float idx raise. Same seed → bit-identical params; init stats (biases/β = 0, γ = 1).
  `forward(idx) == forward_from_embedding(embed(idx))` exactly. Tied-head anchor:
  `logits == LN_f(h_L) @ wte.W.T` recomputed from `cache` to ≤ ROUNDOFF_ATOL. mask_fn: a single
  callable is applied to every block (`blocks[l].attn.mask_fn is f`); a list of L callables is applied
  per block; a list of wrong length raises.
- B3: dev gradcheck on L=1, d=8, H=2, T=3, V=11 (cheap), logits objective, all tensors except
  `h.0.attn.k.b` < 1e-6 (the formal C6 is C's). `backward(..., include_input_side=False,
  include_output_side=False)["wte.W"]` is all zeros and every other key is unchanged.

### 6.4 Process guidance (verbatim, binding)

GIT RULES (the repo has Sprint 1+2 history on it already):
- Identify yourself ONLY via environment variables on each git command, never via `git config`: prefix every git add/commit with GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local".
- Only `git add` the exact files you own — never `git add -A`, `git add .`, or any path outside your assignment.
- End every commit message with a blank line then exactly: Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
- If a git command fails with an index.lock error, a teammate is committing at that exact moment — wait ~2 seconds and retry, up to 3 times.
- Never run `git push`, `git reset --hard`, `git rebase`, or touch any git config file.

SPRINT 2 PROCESS FAILURE TO NOT REPEAT (the PO called a repeat of this "rejection-level"): last sprint, one engineer batched 3 distinct milestones into a single commit built against teammates' not-yet-committed files, leaving several intermediate commits broken if checked out alone (HEAD was still green, but the interior of the history wasn't). This sprint:
- Split your own work into the smallest sensible commits, strictly in dependency order — never batch multiple distinct milestones (e.g. "implement X" and "verify X" and "verify Y") into one commit.
- Before relying on a teammate's file in your own work, confirm it is actually committed (`git log --oneline -- <their path>`), not just present on disk uncommitted. If it isn't committed yet, either wait briefly and retry, or proceed using the documented local-stand-in fallback and say so explicitly in your notes — never silently assume an on-disk file is safely there to depend on.
- Each of your own commits should pass the test suite on its own at the moment you make it, given what's actually committed so far (by you and others) plus your own staged change.

Your identity: `Engineer B` / `engineer-b@nanogpt-numpy.local`. Run the §8.2 pre-commit check before
**each** of B1, B2, B3. B1 requires `[A1]` committed. Commit B2 as soon as it is green: A and C are
waiting on it.

---

## 7. Engineer C: full-model verification (C5 re-run, C6, C7)

### 7.1 Milestones (one commit each — three commits, never one)

- **C1** append the full-model C5 section to `tests/test_causal_mask.py`. Needs `[B2]` committed
  (forward only).
- **C2** `tests/test_gradients.py` (C6). Needs `[B3]` committed.
- **C3** `tests/test_weight_tying.py` (C7). Needs `[B3]` committed.

Before `[B2]`/`[B3]` land, you may draft against a local scratch stand-in of the §3.4 interface (e.g.
built in your scratchpad from the TL's formulas), but you commit **only** after `git log --oneline --
nanogpt/model.py` shows the milestone you need, and only after the §8.2 check passes against the
committed `model.py`. Note the stand-in use in your results block.

### 7.2 Spec

Tiny config `tiny_config(55)` = d=8, L=2, H=2, T=4 (V = 55 to match the real vocab), B=2, non-init
fixture (§3.6) from seed `+321`/`+322`/`+323`. Use `nanogpt.gradcheck.gradcheck` and
`numerical_jacobian` only; no bespoke FD loops. Share expensive gradcheck reports through
module-scoped pytest fixtures so each gradcheck runs once per file.

**TL-9 protocol, applied to every layer l in range(L)** (restated; thresholds frozen):
- (a) analytic: max|d_params[f"h.{l}.attn.k.b"]| ≤ ROUNDOFF_ATOL (1e-12) for a random upstream R.
- (b) FD-noise bound: the gradcheck report's `max_abs_err` for that tensor ≤ REL_ERR_FLOOR (1e-8),
  on both the logits-objective and loss-objective gradchecks.
- (c) invariance: max|logits(params with b_k ← b_k + 10·N(0,1)) − logits(params)| ≤ ROUNDOFF_ATOL.
- (d) dropped-term adversary: with `attention.ops.softmax_backward` monkeypatched to `p * dp` (no
  row-sum correction), max|d h.{l}.attn.k.b| > 1e-3 (measured 16.3 / 6.0).
- Report (not gate) rel_err and max_elem_rel_err for each k.b.
- Assert the exempt set is **exactly** `{f"h.{l}.attn.k.b" for l in range(L)}`.

### 7.3 Acceptance criteria (each file < 10 s)

C5, full model (appended to `tests/test_causal_mask.py`, C1):
- Jacobian J of `forward_from_embedding` logits (B=1, T=4, V) w.r.t. h0 (1, 4, d) via
  `numerical_jacobian`; block norms N[t,s] = ‖J[t,:,s,:]‖_F.
- Causal model: max over s > t of N[t,s] ≤ ROUNDOFF_ATOL (measured exactly 0.0); min over s ≤ t of
  N[t,s] > 0.1 (non-vacuity; measured ≥ 1.5).
- Token substitution: changing `idx[0, s]` leaves `logits[0, t]` for all t < s identical to
  ≤ ROUNDOFF_ATOL (measured 0.0) and changes `logits[0, s]`.
- **Shifted adversary through every block:** `GPT(cfg, mask_fn=shifted_causal_mask)` — C5 fails
  (future max > 0.1); N[t, t+1] > 0.1 for all t < T−1; leak reaches exactly s = t + L
  (N[t, t+L] > 0.1 where defined, N[t, s] ≤ ROUNDOFF_ATOL for s ≥ t + L + 1).
- **Shifted adversary in each single block:** for each l, `mask_fn` list with only block l shifted —
  C5 fails at s = t+1 (> 0.1) and N[t, s] ≤ ROUNDOFF_ATOL for s ≥ t+2. This proves block l's mask is
  really threaded (if it were ignored this case would look causal and the test fails).
- Token-substitution also detects the all-shifted model (changing idx[t+1] changes logits[t]).
- Substitution only via the constructor; no monkeypatching of masks. Existing Sprint 2 tests and
  guards in the file are not modified.

C6 (`tests/test_gradients.py`, C2):
- G1: `gradcheck(model.forward, model.backward, params, idx)` — logits objective, integer idx.
- G2: gradcheck of `forward_loss`/`backward_loss` (wrapped to the harness signature with fixed
  targets) — scalar loss objective.
- G3: gradcheck of `forward_from_embedding`/`backward_from_embedding` on `params` minus `wpe.W`,
  w.r.t. h0 (float, seed `+321`, never DEFAULT_SEED) — covers the continuous input (TL-18).
- Each report covers **exactly** the expected key set (G1/G2: all 36; G3: 35 + "x"). Every
  non-exempt tensor rel_err < GRADCHECK_TOL (1e-6). TL-9 (a)–(d) for every layer (§7.2).
- Print the full table (rel_err | max_elem_rel_err | max_abs_err) for the TL.
- **Test-the-test:** with `block.py`'s residual identity term dropped via monkeypatch (or the
  ln_f backward replaced by identity), G1 fails on at least one tensor — shows the full-model gate
  can fail.

C7 (`tests/test_weight_tying.py`, C3):
- Structure: `params` has `wte.W` and no `lm_head`/head-bias key; `named_parameters()["lm_head.W"] is
  params["wte.W"]`; `params["wte.W"].flags.c_contiguous` and `.base is None`.
- Partials: g_full, g_in (`include_output_side=False`), g_out (`include_input_side=False`) for the
  same upstream R: ‖g_in + g_out − g_full‖_max ≤ ROUNDOFF_ATOL; ‖g_in‖_F > 1e-3 and ‖g_out‖_F > 1e-3.
- **Independent FD (TL-20):** FD_out = `numerical_jacobian` of W ↦ Σ(R ⊙ forward_from_embedding(h0,
  params[wte.W ← W])) with h0 = embed(idx) frozen; FD_full = same for W ↦ Σ(R ⊙ forward(idx,
  params[wte.W ← W])). Then rel_err(g_out, FD_out) < 1e-6, rel_err(g_full, FD_full) < 1e-6,
  rel_err(g_in, FD_full − FD_out) < 1e-6.
- Accumulation (structural): AST scan of `nanogpt/model.py` finds no assignment whose target is a
  subscript with the constant `"wte.W"`. **Test-the-test:** monkeypatch `model.ops.accumulate_grads`
  with an overwrite variant (`grads[k] = v.copy()` always) → g_full ≠ g_in + g_out and the wte.W
  gradcheck fails.
- Contiguity trap: gradcheck on a params copy whose `wte.W` is `np.asfortranarray`/transposed-view
  data raises `ValueError` (harness refusal); the production array passes.

### 7.4 Process guidance (verbatim, binding)

GIT RULES (the repo has Sprint 1+2 history on it already):
- Identify yourself ONLY via environment variables on each git command, never via `git config`: prefix every git add/commit with GIT_AUTHOR_NAME="<your role>" GIT_AUTHOR_EMAIL="<your-role>@nanogpt-numpy.local" GIT_COMMITTER_NAME="<your role>" GIT_COMMITTER_EMAIL="<your-role>@nanogpt-numpy.local".
- Only `git add` the exact files you own — never `git add -A`, `git add .`, or any path outside your assignment.
- End every commit message with a blank line then exactly: Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
- If a git command fails with an index.lock error, a teammate is committing at that exact moment — wait ~2 seconds and retry, up to 3 times.
- Never run `git push`, `git reset --hard`, `git rebase`, or touch any git config file.

SPRINT 2 PROCESS FAILURE TO NOT REPEAT (the PO called a repeat of this "rejection-level"): last sprint, one engineer batched 3 distinct milestones into a single commit built against teammates' not-yet-committed files, leaving several intermediate commits broken if checked out alone (HEAD was still green, but the interior of the history wasn't). This sprint:
- Split your own work into the smallest sensible commits, strictly in dependency order — never batch multiple distinct milestones (e.g. "implement X" and "verify X" and "verify Y") into one commit.
- Before relying on a teammate's file in your own work, confirm it is actually committed (`git log --oneline -- <their path>`), not just present on disk uncommitted. If it isn't committed yet, either wait briefly and retry, or proceed using the documented local-stand-in fallback and say so explicitly in your notes — never silently assume an on-disk file is safely there to depend on.
- Each of your own commits should pass the test suite on its own at the moment you make it, given what's actually committed so far (by you and others) plus your own staged change.

Your identity: `Engineer C` / `engineer-c@nanogpt-numpy.local`. You were the D1 engineer last sprint:
three milestones, three commits, each after its own §8.2 check. C1 needs `[B2]`; C2 and C3 need `[B3]`.

---

## 8. Git protocol

### 8.1 Rules (shared; identical to the verbatim blocks in §5.4/§6.4/§7.4)

Identities: `Engineer A` / `engineer-a@nanogpt-numpy.local`, `Engineer B` /
`engineer-b@nanogpt-numpy.local`, `Engineer C` / `engineer-c@nanogpt-numpy.local`, `Tech Lead` /
`tech-lead@nanogpt-numpy.local`. Run git from **Git Bash** (the `VAR=value git …` prefix does not work
in PowerShell). Commit with an explicit pathspec (`git commit -m … -m … -- <your files>`). Subjects:
`Phase 3 [<milestone>]: <what>`. TL fixups are separate commits titled `TL fixup: <file>: <what>`.
Never `--amend` anything but your own last commit. `.venv/`, `__pycache__/`, `.pytest_cache/` are
gitignored; never force-add them.

### 8.2 Mandatory pre-commit check (committed tree + only your staged files)

```bash
cd /c/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy
git log --oneline -- <each teammate file you import>       # must show the milestone you need
GIT_AUTHOR_NAME="Engineer X" GIT_AUTHOR_EMAIL="engineer-x@nanogpt-numpy.local" \
GIT_COMMITTER_NAME="Engineer X" GIT_COMMITTER_EMAIL="engineer-x@nanogpt-numpy.local" \
  git add <your files for THIS milestone only>
git diff --cached --name-only                              # only your files; if a teammate's file
                                                           # is listed, do not proceed; tell the TL
S=$(mktemp -d)
git archive HEAD | tar -x -C "$S"                          # the COMMITTED tree
git checkout-index -f --prefix="$S/" -- <your files>       # + exactly your STAGED versions
(cd "$S" && /c/Users/AnNLT/Downloads/MathematicsForComputing/nanogpt_numpy/.venv/Scripts/python.exe \
   -m pytest tests/ -q -p no:cacheprovider)                # must be all green (no skips of gating tests)
rm -rf "$S"
GIT_AUTHOR_NAME="Engineer X" GIT_AUTHOR_EMAIL="engineer-x@nanogpt-numpy.local" \
GIT_COMMITTER_NAME="Engineer X" GIT_COMMITTER_EMAIL="engineer-x@nanogpt-numpy.local" \
  git commit -m "Phase 3 [X1]: <what>" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" \
  -- <your files>
```
If the scratch run fails only because a teammate's module is missing, you depended on uncommitted
work: unstage, wait for their commit, and re-run. Paste the scratch run's last line ("N passed") into
your results block for every commit.

---

## 9. Dependency graph and commit ordering

```
A1 mlp ──> B1 block ──> B2 model fwd (structure final) ──┬─> A2 C1 param_count
                                                          ├─> A3 C2 loss@init
                                                          ├─> C1 C5 full-model
                                                          └─> B3 model bwd ──┬─> C2 C6 test_gradients
                                                                             └─> C3 C7 test_weight_tying
```
Required order in `git log` (checked at sign-off): `[A1] < [B1] < [B2] < {[A2], [A3], [C1], [B3]}`,
`[B3] < [C2]`, `[B3] < [C3]`. No commit contains two milestones. No fallback shim or
`TODO(SPRINT3-…)` marker is ever committed (the existing `TODO(SPRINT` guard fails on it).
Interface changes to §3 need TL approval; internals do not.

---

## 10. Timeline (≈ 3 engineering days)

| Day | A | B | C |
|---|---|---|---|
| 1 AM | **A1** mlp + tests (commit) | B1 block against local MLP stand-in | draft C5/C6/C7 against a scratch stand-in |
| 1 PM | draft C1 formula + GPT-2 test | **B1** commit (after `[A1]`), start B2 | keep drafting; nothing committed yet |
| 2 AM | | **B2** model forward (commit, "structure final") | **C1** C5 full model (commit) |
| 2 PM | **A2** C1 (commit), **A3** C2 (commit) | **B3** backward + dev gradcheck (commit) | |
| 3 | results block | results block | **C2** C6 (commit), **C3** C7 (commit), results block |
| 3–4 | | | **TL integration and sign-off** |

TL duties during the sprint: review each milestone as it lands (fixes as separate `TL fixup:`
commits), and check after every push of a milestone that it is green in isolation (§11 item 6) so a
D1 repeat is caught the same day, not at sign-off.

---

## 11. Integration and sign-off checklist (TL; re-measured, not copied)

1. `pytest tests/ -q --durations=0`: all green, < 60 s, each file < 10 s; two consecutive runs identical.
2. TL throwaway re-measurement: MLP/block/model gradcheck tables (all tensors, rel_err |
   max_elem_rel_err | max_abs_err); TL-9 (a)–(d) per layer; C1 values (2,232 / 107,712 / 816,768 /
   124,439,808); C2 deltas for each config and seed + the wrong-init delta; C7 partial norms, sum
   residual, FD agreements; C5 N[t,s] tables for causal, all-shifted, each single-block-shifted.
3. All test-the-tests confirmed by passing assertions (MLP gelu-skip, block residual drops, C1
   double-count/view/head-bias, C2 wrong init, C6 broken residual, C7 overwrite, TL-9 dropped term per
   layer, C5 shifted all/per-block).
4. `grep -rn "TODO(SPRINT" nanogpt/` empty; conventions guard green; no torch/jax/tf/autograd;
   `shifted_causal_mask` not referenced in `nanogpt/` outside `attention_mask.py`; no `lm_head` key in
   `param_keys()`; no `wte.W` subscript assignment in `model.py`.
5. Corpus SHA-256 still `7afc055a…2854`, V = 55.
6. **Commit-discipline gate (P0):** for every commit c in `<kickoff>..HEAD`:
   `S=$(mktemp -d); git archive $c | tar -x -C $S; (cd $S && python -m pytest tests/ -q
   -p no:cacheprovider)` is green. Each engineer commit touches only that engineer's files
   (`git show --stat`), holds one milestone, and the §9 order holds. Any red interior commit is
   recorded as a deviation and reported to the PO as a rejection-level finding (not rewritten).
7. **No Phase 4+ files:** `optim.py`, `kv_cache.py`, `sampling.py`, `flops.py`, `checkpoint.py`,
   `train.py`, `generate.py`, `tests/test_kv_cache.py`, `tests/test_flops.py`, `tests/test_adamw.py`;
   no KV-cache or sampling argument in `attention.py`/`model.py`.
8. Sprint 1+2 hashes unchanged (`283c20f`..`65ca41e` ancestry); `git status` clean.
9. Append the PROGRESS.md Phase 3 entry (§12) and commit `Sprint 3 integration sign-off`.

---

## 12. Results block and PROGRESS template

**Results block each engineer hands the TL:**
```
Files: <>   Owner: <A|B|C>   Commits: <hash + subject, one per milestone>
Per commit: dependency check (git log -- <path>) output + scratch-tree run "<N passed>"
Gradcheck tables (rel_err | max_elem_rel_err | max_abs_err), ALL tensors
TL-9 per layer: (a) <> (b) <> (c) <> (d) <>
C1: <values>  C2: <deltas per config/seed; wrong-init delta>  C7: <norms, residual, FD rel_errs>
C5: N[t,s] causal / all-shifted / per-block-shifted
Test-the-test: <variant> -> FAILED as expected (<metric>=<value>)
Local stand-ins used before a dependency was committed: <none | what, when removed>
Timing: <file> <s>   Deviations from SPRINT_3_PLAN.md: <none | what + why>
```

**PROGRESS.md Phase 3 entry (TL; append-only):** owners/commits; the per-commit isolation results
(item 6); suite status and timings; C1/C2/C5/C6/C7 tables; TL-9-for-every-layer outcome; rulings
R15–R19, TL-16..TL-23; deviations for PO ratification; Next: Phase 4 (AdamW + training loop + C9).

---

## 13. Explicitly out of scope for Sprint 3 (do not build)

C8 / `kv_cache.py` or any past-K/V argument; C9 / `flops.py` or FLOP hooks; `optim.py` (AdamW);
`train.py` and any training loop; `generate.py`, `sampling.py`, temperature/top-k; `checkpoint.py`;
GPT-2 weight loading; the Phase 6 large-logit stability pass; performance work; P2 extras (extra
L/H/d sweeps, param_count CLI). `model.py` must not grow a `generate()` or `train_step()`.
