# nanoGPT in NumPy — Progress Log

This file is the resumption mechanism for a long-running, multi-engineer
build. Each phase gets one entry, appended in order, written by Engineer A
(Tech Lead integration role for Sprint 1). Entries are never rewritten
after sign-off; a correction is a new entry.

---

## Phase 0 — Frozen Conventions, Scaffolding & Data  [DONE 2026-11-20]

**Owner:** Engineer A (Sonnet 5)

**Environment:**
- Interpreter: Python 3.15.2 (`py -3.15`, Windows), venv at `nanogpt_numpy/.venv`
- numpy 2.5.5
- scipy 1.18.1
- pytest 9.2.0
- Wheels for numpy/scipy/pytest were available for 3.15 directly (verified
  with a dry-run install before building the real venv), so the
  "fall back to `py install 3.13`" contingency in SPRINT_1_PLAN.md §4/§6.1
  was **not needed**.

**Corpus:**
- source: `synthetic` (no `data/corpus.txt` present; `data/` only holds
  `.gitkeep`)
- generator: `synthetic-v1`
- seed: `1337` (= `CORPUS_SEED`)
- n_chars: `1000000`
- sha256: `7afc055acf69c9217f38edc2b562e001fb6251e5ecdede83fbc40c0425ca2854`
- V (vocab size): `55`
- The generator never writes `data/corpus.txt`; `resolve_corpus` was
  verified to touch no network (socket.socket monkeypatched to raise, in
  both the bundled and synthetic branches).

**Passed:**
- `tests/test_config.py` — 24 passed. Every §5 constant name/value,
  `rel_err` (gated, Frobenius-norm form) incl. `rel_err(0,0)==0.0` and the
  scalar-equivalence check, `rel_err_elementwise` as a diagnostic,
  `make_rng`, `check_float64` (raises on float32/float16, ignores int64),
  `ModelConfig` validation (`d % H == 0`, positivity, frozen), `tiny_config`
  (d=8, L=2, H=2, T=4, head_dim=4).
- `tests/test_gradcheck_harness.py` — 13 passed (**Phase 0 exit
  criterion**). Correct `y = x@W` passes with every `rel_err < 1e-8`.
  Harness FAILS on: dW scaled by 1.01, transposed dW on square W, and a
  wrong-but-right-shaped dx with correct dW (proves the input is actually
  checked). Missing/wrong-shape `d_params` keys raise `ValueError`. An
  unused bias with an exactly-zero true gradient reports `rel_err == 0.0`
  (floor works; stands in for the Phase-2 T=1 case). Integer `x` with
  `d_x=None` is accepted even with `wrt_input=True`. Params are
  bit-identical after the run, including after a failing check.
  `numerical_jacobian` of `x -> A x` matches `A` with `rel_err < 1e-9`,
  on both a vector and a non-trivial-shaped input.
- `tests/test_tokenizer.py` — 19 passed. Same seed -> identical corpus,
  different seed -> different corpus. Exact-length truncation at several
  `n_chars`. `decode(encode(text)) == text` on a 50k-char slice of the
  full generator output (full corpus round-trip is additionally exercised
  implicitly by `load_corpus_and_tokenizer` + `test_loss.py`'s integration
  anchor, which C owns). `resolve_corpus` bundled branch (tmp_path
  `corpus.txt`) and synthetic branch (empty dir) both verified, including
  that the generator never writes `corpus.txt` to disk, and that
  `socket.socket` can be monkeypatched to raise in either branch with no
  effect. V is identical across two fresh `load_corpus_and_tokenizer()`
  calls. Unknown characters raise `ValueError`; `encode` returns int64.

**Interpretations / deviations (recorded per SPRINT_1_PLAN.md §6.4):**
- **TL-1:** `rel_err` is gated tensor-wise (whole-tensor Frobenius norm),
  not elementwise. `rel_err_elementwise` exists only as an unglated
  diagnostic. This is why an exactly-zero true gradient (unused bias; T=1
  Q/K in Phase 2) reports a clean `0.0` instead of a degenerate ratio.
- **R4:** the cross-entropy normalizer is `N = targets.size` (= B·T for a
  (B,T) batch). Both forward and backward will use the same N (owned by
  Engineer C in `loss.py`, Phase 1).
- **R5:** default `H = 4` (d=64 default -> head_dim=16); the tiny config
  uses `H = 2` (d=8 -> head_dim=4). `ModelConfig` validates `d % H == 0`.
- **R6:** gradcheck tests live in per-owner files
  (`tests/test_gradcheck_harness.py` here, `tests/test_ops.py` +
  `tests/test_softmax_properties.py` for B, `tests/test_layernorm.py` +
  `tests/test_layernorm_rank.py` + `tests/test_loss.py` for C).
  `tests/test_gradients.py` is **reserved** for Phase 3 (C6) and was not
  created in Sprint 1.
- No other deviations from SPRINT_1_PLAN.md in Phase 0.

**Next:** Phase 1 — Core Differentiable Primitives (ops.py, layernorm.py,
loss.py; C3/C4 acceptance). *(Phase 1 entry below, written at integration
after B/C hand off their result blocks.)*

---

## Phase 1 — Core Differentiable Primitives  [DONE 2026-11-20]

**Owners:** Engineer B (`ops.py`), Engineer C (`layernorm.py`, `loss.py`).
**Integration and sign-off:** Tech Lead (Opus 5.5). Every number below was
re-measured by the TL on the integrated tree, not copied from a
self-reported results block. Test-the-test outcomes are confirmed by
their passing assertions in the integrated suite.

### What Sprint 1 delivered (Phase 0 + Phase 1)

| Area | Files | Status |
|---|---|---|
| Frozen Conventions (single source of truth) | `nanogpt/config.py` | done, guarded by `tests/test_conventions.py` |
| Corpus + char tokenizer (offline, seeded) | `nanogpt/tokenizer.py` | done (synthetic-v1, V=55) |
| Generic FD gradcheck harness + `numerical_jacobian` | `nanogpt/gradcheck.py` | done, proven on y=xW |
| Linear, Embedding (`np.add.at`), GELU-tanh, stable softmax/logsumexp, `accumulate_grads` | `nanogpt/ops.py` | done, C4 passes |
| Affine LayerNorm (ddof=0, ε=1e-5) + ε=0 C3 instance | `nanogpt/layernorm.py` | done, C3 passes |
| Stable cross-entropy (N = targets.size) | `nanogpt/loss.py` | done |

### Real test status (TL run, integrated tree)

`.venv\Scripts\python.exe -m pytest tests/ -q` gives **155 passed in about 1.2 s**.
Two consecutive runs gave identical results. Per file (all far under the
10 s budget):

| File | Tests | Time |
|---|---|---|
| test_config.py | 24 | 0.30 s |
| test_conventions.py | 16 | 0.09 s |
| test_gradcheck_harness.py | 14 | 0.34 s |
| test_tokenizer.py | 19 | 0.53 s |
| test_ops.py | 22 | 0.19 s |
| test_softmax_properties.py | 36 | 0.23 s |
| test_layernorm.py | 8 | 0.21 s |
| test_layernorm_rank.py | 8 | 0.19 s |
| test_loss.py | 8 | 0.24 s |

Sign-off checklist (SPRINT_1_PLAN.md §12): (1) `TODO(SPRINT1-` in
`nanogpt/` finds nothing and the guard test is green. (2) All green, under
60 s, every file under 10 s. (3) The run is deterministic, and the corpus
SHA-256 re-computed by the TL matches the Phase 0 entry
(`7afc055a…2854`, V=55). (4) The Phase 0 entry above is dated before this
sign-off. (5) See the tables below. (6) No Phase 2+ file is in the repo.

### Gradcheck table (gated: norm-wise rel_err < 1e-6; TL re-measurement)

| Module / tensor | rel_err | max elementwise rel_err (diagnostic) |
|---|---|---|
| Linear(bias) x(2,3,4)→5: W / b / x | 2.0e-11 / 1.7e-11 / 1.6e-11 | 3.0e-10 / 3.4e-11 / 3.4e-10 |
| Linear(bias=False): W / x (no `.b` key) | 1.6e-11 / 8.6e-12 | 3.3e-10 / 7.1e-11 |
| Embedding num=5,d=3, idx=[[0,2,2],[1,0,2]]: W | 1.1e-11 | 2.8e-10 |
| GELU-tanh, random + probes 0, ±1e-3, ±1, ±10: x | 4.8e-11 | 7.4e-09 |
| Softmax z(2,3,6): x | 1.9e-11 | 3.5e-09 |
| LayerNorm production (ε=1e-5, γ=1+0.5N, β~N): x / γ / β | 5.5e-11 / 4.2e-11 / 1.8e-11 | 1.2e-07 / 1.8e-10 / 2.8e-10 |
| LayerNorm ε=0 no-affine x(2,3,6): x | 3.4e-11 | 2.6e-10 |
| Cross-entropy logits(2,3,7), repeated targets: logits | 1.2e-10 | 4.8e-09 |

### Invariant checks

- **C3 (d=8):** the ε=0 instance has rank **6 = d−2** (also d=3→1 and d=5→3),
  and J·1 and J·x vanish to roundoff. The production instance at σ²=1 has
  rank **7 = d−1**, which is expected per spec §4 and not a bug.
  s[−2]/s[0] = 9.9999e-06 vs the predicted ε/(σ²+ε) = 9.9999e-06. TL-3
  boundary at σ²=1e3: rank 6, ratio 1.0000e-08 vs 1.0000e-08 predicted. The
  non-vacuity test confirms the two ranks differ.
- **C4 (d=6, seeds 0–3, J extracted from `softmax_backward`):**
  max|J − (diag p − ppᵀ)| = 2.8e-17. min eig((J+Jᵀ)/2) = −8.1e-17 (≥ −1e-10).
  max‖J·1‖∞ = 5.6e-17. Shift-invariance holds for c ∈ {−5, 3.7, 1e3}. With k
  finite entries rank(J) = k−1, for k = 2..6. J also matches
  `numerical_jacobian` within 1e-6.
- **Loss anchors:** zero logits give ln 7, and ln V with V=55 from the
  real tokenizer. A +50 target margin gives loss < 1e-20. The rows of
  d_logits·N/d_loss sum to 0. A (2,3) batch matches the flattened (6,)
  batch (R4). Logits of 1e3·N(0,1) give a finite loss and gradient.

### Test-the-test (each deliberately broken variant is detected)

- **Embedding:** the naive `dW[idx] += d_out` backward **fails gradcheck** on
  duplicate idx and **passes** on unique idx. It also disagrees with a
  python-loop reference (rel_err > 1e-3).
- **LayerNorm:** the backward missing ŷ⊙mean(g⊙ŷ) fails gradcheck
  (rel_err ≥ 1e-6) and gives rank d−1 on the ε=0 instance. The backward
  missing ⊙γ fails at γ≠1 and passes at γ=1 exactly (TL-6).
- **Softmax:** the dropped-term backward `dz = p⊙dp` breaks the row-sum and
  closed-form checks **and fails gradcheck**. The non-shift-invariant
  e^z/(Σe^z+1) fails the shift check.
- **GELU (TL, ad hoc, not committed):** a backward that drops the tanh′
  term fails gradcheck (rel_err 1.4e-1).
- **LSE smoke:** the naive exp(z)/Σexp(z) overflows at z = 1e3·N(0,1). The
  stable path stays finite.

### Integration fixes made by the TL at sign-off

1. **GELU had no harness gradcheck.** Only a single-direction FD spot check
   existed, so DoD item 4 was not met. Added `test_gelu_gradcheck`
   (random values plus the §7.3 probe points).
2. **Embedding and softmax test-the-test now use the harness itself**, as
   §7.3 / §12 item 5 require. Previously they were checked only against a
   reference loop or closed form. Added
   `test_embedding_test_the_test_naive_backward_fails_gradcheck_on_duplicates_only`,
   and extended the softmax dropped-term test with a gradcheck-fails assertion.
3. **TL-7 gap in `LayerNorm.backward`.** It silently cast `d_out` to
   float64 instead of raising. It now calls `check_float64`, and a float32
   `d_out` test was added to `test_dtype_strictness_float32_raises`.
4. **`gradcheck` hardening.**
   - A non-C-contiguous float parameter now raises `ValueError`. Before,
     `reshape(-1)` would silently perturb a copy and report a zero numeric
     gradient, a latent trap for Phase 2/3 transposed or tied weights.
     Covered by a new harness test.
   - Documented C's seed-collision pitfall in the `gradcheck` docstring:
     if a fixture x and gradcheck's internal R come from the same
     DEFAULT_SEED, R can land in the Jacobian null space.
5. **Removed the leftover §10 fallback scaffolding** (`HAS_GRADCHECK` /
   `needs_gradcheck` skip-guards) from `test_ops.py` and
   `test_softmax_properties.py`, so no gating test can silently skip.
   Removed the unused `REL_ERR_FLOOR` import.
6. **Verified, nothing to do:**
   - All three modules import only from `nanogpt/config.py`. There are no
     local constant copies and no `TODO(SPRINT1-` markers left.
   - `tests/_sprint1_gradcheck_shim.py`, flagged by A and B, had already
     been deleted by C.
   - A's interim report of two failing LayerNorm tests was superseded: the
     cause was C's fixture seed colliding with gradcheck's R seed, not a
     backward bug. C fixed it with `_FIXTURE_SEED`, and the TL confirmed
     the backward formula against TL-2 by reading the code.

### Deviations from SPRINT_1_PLAN.md (accepted by TL)

- **D1:** `tests/test_ops.py` keeps B's three `*_directional_fd_check` tests,
  and `tests/test_softmax_properties.py` keeps `_fd_jacobian_local`. §10
  said throwaway FD loops are not committed. They are kept because they
  are small, clearly labelled, independent second opinions, and every
  gating claim is also checked by `nanogpt.gradcheck`. They can be deleted
  without losing coverage.
- **D2:** the TL edited files owned by A, B and C during integration (fixes
  1–5 above). This is within the §11 day-3–4 integration remit. No
  interface in §3/§5/§6.2/§7.1/§8.1 changed. `gradcheck` only gained a
  stricter precondition.
- Carried over from Phase 0: TL-1 norm-wise rel_err, R4 N=targets.size,
  R5 H=4 by default (tiny config H=2), and R6 per-owner gradcheck files
  (`tests/test_gradients.py` stays reserved for C6).

### Remaining issues / watch items for Sprint 2

- No blocking issues.
- Fixture seeds: Phase 2+ gradcheck fixtures must not draw x from
  `make_rng(DEFAULT_SEED)` when x has the output's shape (see the
  `gradcheck` docstring).
- Weight tying (Phase 3): if `lm_head` is expressed as a transposed view of
  `wte.W`, gradcheck the underlying contiguous `wte.W`. The harness now
  refuses non-contiguous params.

**Next: Phase 2 — Causal Multi-Head Self-Attention (Sprint 2).** Scope per
the execution plan:
- `nanogpt/attention.py`: Q/K/V/O linears with bias, H=4 by default and
  H=2 in the tiny config, scaled scores QKᵀ/√head_dim, causal mask with
  −∞ (softmax already tolerates −∞, TL-5), and the softmax backward
  `dS = P⊙(dP − Σ(dP⊙P))` reused from `ops`.
- Gradcheck of every Q/K/V/O W and b and of x on the tiny config.
- A naive loop-over-heads reference oracle that the vectorized
  implementation must match.
- The T=1 test: the true Q/K gradients are exactly 0, and the norm-wise
  rel_err reports 0.
- The shifted-mask adversarial test: a mask `j ≤ i+1` must be caught by a
  causality / future-leak check.

---

## Phase 2 — Causal Multi-Head Self-Attention  [DONE 2026-11-20]

**Process (S2-00, approved by the user: "fold that into Sprint 2"):** this was the
first sprint where the team committed as it worked, under per-role identities set
only by `GIT_AUTHOR_*` / `GIT_COMMITTER_*` environment variables (no `git config`
writes), with explicit pathspecs. TL fixups are separate commits. Sprint 1 history
(`283c20f`..`74044c4`) still has its original hashes (ancestry checked).

**Owners / commits (each verified with `git show --stat` to touch only the owner's files):**

| Role | Commits | Files |
|---|---|---|
| Engineer A | `9e1849f` [A1] masks, `eb185c0` [A2] oracle + anchors, `d36dd45` [A3] oracle source guard | `nanogpt/attention_mask.py`, `nanogpt/attention_reference.py`, `tests/test_attention_reference.py` |
| Engineer B | `1276ee6` [B1] forward, `53105eb` [B2] oracle match (hard gate), `c579179` [B3] backward + gradcheck, `4f92d66` [B4] test-the-tests | `nanogpt/attention.py`, `tests/test_attention.py` |
| Engineer C | `13e0519` [C1-C3] in one commit (see D1) | `tests/test_causal_mask.py` |
| Tech Lead | `31016a0` kickoff plan; `267028f`, `dfffd66` TL fixups; Sprint 2 sign-off commit | `SPRINT_2_PLAN.md`, `tests/test_conventions.py`, `tests/test_causal_mask.py` (fixup), `PROGRESS.md` |

The §9 ordering holds: `[A1] 9e1849f < [B1] 1276ee6`, and
`[A2] eb185c0 < [B2] 53105eb < [B3] c579179 < [B4] 4f92d66`.

### What Sprint 2 delivered

| Area | Files | Status |
|---|---|---|
| Causal + shifted-adversary masks (bool `allowed`, fresh array per call) | `nanogpt/attention_mask.py` | done |
| Naive loop-over-heads oracle (no reshape/transpose/einsum/`.T`/`inf`, enforced by AST guard) | `nanogpt/attention_reference.py` | done, closed-form anchors pass |
| Vectorized `CausalSelfAttention` (separate Q/K/V/O `ops.Linear`, TL-11 head split, `np.where` −∞ mask, hand-derived backward reusing `ops.softmax_backward`) | `nanogpt/attention.py` | done, matches oracle, gradcheck passes |
| C5 isolated, shifted-mask test-the-test, T=1, backward causality, masked-softmax rank | `tests/test_causal_mask.py` | done |

### Real test status (TL run on the integrated tree)

`.venv/Scripts/python.exe -m pytest tests/ -q` gives **256 passed in about 1.7 s**,
identical over two consecutive runs. (254 from the engineers plus 2 tests added by
TL fixup `267028f`.) Every file is far under the 10 s budget:

| File | Tests | Time |
|---|---|---|
| test_attention.py (B) | 36 | 0.67 s |
| test_attention_reference.py (A) | 46 | 0.19 s |
| test_causal_mask.py (C) | 14 | 0.30 s |
| test_conventions.py | 21 | 0.10 s |
| Sprint 1 files (config, gradcheck_harness, tokenizer, ops, softmax_properties, layernorm, layernorm_rank, loss) | 139 | each ≤ 0.5 s |

The corpus SHA-256 re-computed by the TL is still `7afc055a…2854`, with V=55 (matches
Phase 0). `TODO(SPRINT` finds nothing in `nanogpt/`. No torch/jax/tf/autograd imports.
No Phase 3+ file exists and `attention.py` has no KV-cache argument.

### Gradcheck tables (TL re-measurement, own fixture seed `DEFAULT_SEED+900`; gated: norm-wise rel_err < 1e-6)

Tiny config d=8, H=2, B=2, T=4 (all 9 tensors):

| Tensor | rel_err | max_elem_rel_err | max_abs_err |
|---|---|---|---|
| attn.q.W | 3.2e-11 | 1.6e-09 | 1.5e-10 |
| attn.q.b | 1.2e-11 | 1.3e-09 | 4.1e-11 |
| attn.k.W | 3.4e-11 | 2.1e-09 | 1.6e-10 |
| **attn.k.b (TL-9)** | 5.9e-03 (not gated) | 2.2e-03 | **2.2e-11 ≤ REL_ERR_FLOOR** |
| attn.v.W | 1.2e-11 | 8.4e-10 | 1.1e-10 |
| attn.v.b | 7.4e-12 | 2.3e-09 | 6.2e-11 |
| attn.o.W | 5.1e-12 | 5.1e-10 | 5.6e-11 |
| attn.o.b | 2.2e-12 | 5.3e-12 | 1.7e-11 |
| x | 3.4e-11 | 2.6e-09 | 8.6e-11 |

Backward-vs-oracle adapter (FD differentiates the **oracle**, analytic side is B's
backward): every non-k.b tensor has rel_err in [2.2e-12, 3.5e-11], and k.b has
max_abs_err 4.4e-11 ≤ REL_ERR_FLOOR. B's second gradcheck at d=16, H=4, B=1, T=3 is
green in the suite.

**TL-9 (b_k structural zero), protocol (a)–(d):** analytic max|db_k| = 3.6e-16
(≤ 1e-12). FD max_abs_err = 2.2e-11 (≤ 1e-8). Invariance max|out(b_k+10N) − out| =
7.5e-16. The dropped-term softmax backward gives max|db_k| = 4.25, so the protocol
discriminates.

### Oracle match (S2-07 hard gate; all ≤ ROUNDOFF_ATOL = 1e-12)

Tiny (d=8, H=2, B=2, T=4) 2.2e-16. Default (d=64, H=4, B=2, T=7) 6.7e-16. H-sweep at
d=8, T=5: H=1 2.2e-16, H=2 3.3e-16, H=4 3.3e-16, H=8 1.7e-16. T=1 0.0. B=3 2.2e-16.
The tiny case also matches with `shifted_causal_mask` in both implementations
(B's test).

### C5 isolated (T=4, d=8, H=2; numerical Jacobian via `gradcheck.numerical_jacobian`)

| Mask | Implementation | future max (s>t) | past min ‖block‖_F (s≤t) | leak ‖J[t,t+1]‖_F min / max | s≥t+2 max |
|---|---|---|---|---|---|
| causal | vectorized | **0.0** | 0.57 | n/a | 0.0 |
| causal | oracle | **0.0** | 0.57 | n/a | 0.0 |
| shifted `j ≤ i+1` | vectorized | 0.61, **C5 fails** | 0.57 | 0.52 / 1.68 | **0.0** |
| shifted `j ≤ i+1` | oracle | 0.61, **C5 fails** | 0.57 | 0.52 / 1.68 | **0.0** |

The shifted adversary is substituted for real via `mask_fn` (not inspected). It leaks
exactly at s = t+1 and nowhere at s ≥ t+2, so the check catches the one-extra-token
leak precisely. C's suite additionally covers: backward causality (the analytic
Jacobian from one-hot upstream gradients is ≤ 1e-12 for s > t and matches FD), batch
independence, a d=64/H=4/T=6 spot check, and the strict-mask `j < i` NaN case, kept
separate as *not* the C5 adversary.

### T=1 degenerate case

At B=2, T=1, dW_q, db_q, dW_k and db_k are **exactly 0** (`np.all(g == 0)`). Gradcheck
reports rel_err **== 0.0** for all four and `passed=True` over all 9 tensors. The
anchor `out == (xW_v+b_v)W_o+b_o` holds with diff 0.0. Paired T=4 run (same weights,
new x): ‖dW_q‖ = 7.8, ‖db_q‖ = 2.3, ‖dW_k‖ = 5.8, and max|db_k| = 2.5e-16 (TL-9: zero
at every T). C's disconnection adversary (Q ignored) is rejected by the paired helper.

### Masked-softmax rank (F5)

Causal row i of `cache["P"]` has a softmax Jacobian of rank exactly i: ranks
[0,1,2,3] for both heads.

### Test-the-test (each broken variant is detected)

- **Interleaved head split (F7):** gradcheck **passes** (worst non-k.b rel_err 5.9e-11,
  so gradcheck is blind to this bug class), while the oracle differs by **1.36**. This
  is why the oracle is mandatory.
- **No-transpose split:** the oracle differs by 3.35.
- **Dropped fan-out** (d_x = dx_v only): the x gradcheck fails (B's test).
- **Missing scale in backward:** the q.W and k.W gradchecks fail (B's test).
- **TL-9 dropped term:** max|db_k| = 4.25.
- **Shifted mask:** C5 fails on both implementations.
- **Q disconnection:** the paired non-zero helper rejects it.
- **Oracle source guard:** flags `.transpose(`, `.T` and `np.inf`, and correctly
  ignores a local variable named `T`.

### Integration fixes made by the TL (separate commits)

1. **`267028f` TL fixup: `tests/test_conventions.py`.** Widened the shim-marker guard
   from `TODO(SPRINT1-` to `TODO(SPRINT` (planned §10 item 1, not yet done), plus a
   parametrized test-the-test for SPRINT2/SPRINT3 markers.
2. **`dfffd66` TL fixup: `tests/test_causal_mask.py`.** C's AST-based adversary guard
   matched only `Name` / `alias` / `def` nodes, so
   `attention_mask.shifted_causal_mask` (an `ast.Attribute`) slipped through. Added the
   Attribute case, and also enforced the literal §7.3 **text** scan. That scan is now
   satisfiable because A rephrased the one docstring that named the adversary, and it
   also catches `getattr(m, "shifted_causal_mask")`. Both adversaries were verified
   ad hoc (in a scratch copy) to make the guard fail.

### Deviations from SPRINT_2_PLAN.md

- **D1 (process; not fixable without rewriting history):** Engineer C committed C1+C2+C3
  as one commit (`13e0519`) **before** its dependencies A2 and B1/B3 were committed. C
  ran its tests against teammates' *uncommitted* working-tree files. As a result, the
  committed trees at `13e0519`, `eb185c0` and `d36dd45` fail test collection
  (`attention.py` not yet committed), and `1276ee6` and `53105eb` have 3 failing C tests
  (backward not yet committed). The tree is green from `c579179` onward and at HEAD.
  This is recorded rather than rewritten (no rebase, per the git rules). **For
  bisecting, treat `9e1849f` and `c579179` onward as the known-green Sprint 2 points.**
- **D2:** Engineer B developed against local, uncommitted stand-ins for A's mask and
  oracle files, as §9 allows. A overwrote them with the spec-exact versions before
  committing, and B's B2–B4 numbers were measured against A's committed files. No drift.
- **D3:** A's `attention_reference.py` docstring was rephrased so it no longer names
  the adversary. The TL ruling is that **both** the AST guard (C) and the literal text
  scan (§7.3) are enforced.
- Interpretations carried as planned: **R12** (oracle and masks in their own files,
  different owner), **R13** (per-owner test files; `tests/test_gradients.py` still
  reserved for C6), **R14** (PO's T>1 b_k expectation corrected), **TL-9** (b_k
  structural-zero gate, **FOR PO RATIFICATION**), **TL-10** (separate Q/K/V/O, no
  fused QKV). Carried from Sprint 1: TL-1, R4, R5, R6.

### Open items / watch items for Sprint 3

- **C5 is NOT retired.** The full-model C5 re-run is owed in Phase 3, on the assembled
  model (embeddings + L blocks + final LN + lm_head). It includes the shifted-mask
  test-the-test, substituted through every block's `mask_fn`.
- **TL-9 applies to every block's `attn.k.b` in C6** (full-model gradcheck). Use the
  same (a)–(d) protocol, not the literal rel_err gate.
- **Weight tying (C7):** gradcheck the contiguous `wte.W`, never a transposed view (the
  harness refuses non-contiguous params).
- **Process lesson from D1:** before committing, each engineer runs their test file
  against the **committed** tree plus their own staged files (e.g. `git archive HEAD |
  tar -x -C <scratch>`), never against teammates' uncommitted files. A test is
  committed only once every module it imports is committed.

**Next: Phase 3 — Transformer Block & Full Model Assembly (Sprint 3).** Scope per the
execution plan: `mlp.py` (Linear–GELU–Linear), `block.py` (pre-LN residual block:
x + attn(LN(x)), x + mlp(LN(x))) and `model.py` (token + positional embeddings, L
blocks, final LN, tied lm_head). Also `param_count.py` (C1), the full-model gradcheck
in `tests/test_gradients.py` (C6, with TL-9 for every b_k), C7 weight tying, and the
**C5 re-run on the full model**.

---

## Phase 3 — Transformer Block & Full Model Assembly  [DONE 2026-11-20]

**Process:** same rules as Sprint 2 (per-role env-var identities, explicit pathspecs,
no history rewrites), plus the new P0 commit-discipline gate (SPRINT_3_PLAN.md §8.2 /
TL-22): every commit must be green against the *committed* tree plus the author's own
staged files. Sprint 1+2 history is unchanged (`65ca41e` is an ancestor of HEAD).

**Owners / commits (each checked with `git show --stat`; each touches only its owner's files):**

| Role | Commits | Files |
|---|---|---|
| Engineer A | `88fc79b` [A1] MLP + gradcheck + closed-form anchor; `b0fa874` [A2] C1 param count; `03f8ab7` [A3] C2 loss at init | `nanogpt/mlp.py`, `tests/test_mlp.py`, `nanogpt/param_count.py`, `tests/test_param_count.py`, `tests/test_loss_at_init.py` |
| Engineer B | `f6d8fe2` [B1] pre-LN block; `83b05d1` [B2] GPT forward + tied head (structure final); `74a22b8` [B3] GPT backward + C7 flags + dev gradcheck | `nanogpt/block.py`, `tests/test_block.py`, `nanogpt/model.py`, `tests/test_model.py` |
| Engineer C | `93f9577` [C1] full-model C5 re-run; `180749c` [C2] C6 full-model gradcheck + TL-9 per layer; `f036139` [C3] C7 weight tying | `tests/test_causal_mask.py` (append-only; the Sprint 2 portion is byte-identical), `tests/test_gradients.py`, `tests/test_weight_tying.py` |
| Tech Lead | `f5d6464` kickoff plan; Sprint 3 sign-off commit (this entry) | `SPRINT_3_PLAN.md`, `PROGRESS.md` |

Commit order: `A1 < B1 < B2 < {A2, C1} < B3 < {A3, C2, C3}`. This satisfies §9:
`[A1]<[B1]<[B2]<{[A2],[A3],[C1],[B3]}`, `[B3]<[C2]` and `[B3]<[C3]`. Each commit is one
milestone. No commit batches milestones together.

### Per-commit isolation check (P0 gate; TL re-run, `git archive <c>` into a fresh scratch dir, full suite)

| Commit | Result |
|---|---|
| `88fc79b` [A1] | 273 passed |
| `f6d8fe2` [B1] | 288 passed |
| `83b05d1` [B2] | 306 passed |
| `b0fa874` [A2] | 325 passed |
| `93f9577` [C1] | 332 passed |
| `74a22b8` [B3] | 335 passed |
| `03f8ab7` [A3] | 342 passed |
| `180749c` [C2] | 354 passed |
| `f036139` [C3] | 363 passed |

**Every interior commit is green.** Sprint 2's D1 (batched milestones committed against
uncommitted dependencies) was **not repeated**. The counts match the numbers each
engineer reported from their own scratch-tree runs. **Every Sprint 3 commit is a valid
bisect point.**

Local stand-ins (allowed under §6, reported by the engineers): B validated `block.py`
against a scratchpad MLP stand-in before [A1] landed, then re-ran against the real
`mlp.py` before committing. C drafted all three test files against scratchpad
MLP/Block/GPT stand-ins, then re-validated them against the real committed B2/B3
before committing. No stand-in was ever added to the repo. A used none: it waited for
[B2] and committed only model-independent work before that.

### Real test status (TL run on HEAD `f036139`)

`.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` gives **363 passed
in about 23 s**, within the 60 s budget. Every file is under the 10 s per-file budget:

| File | Tests | Time |
|---|---|---|
| test_mlp.py (A) | 16 | 4.6 s (the d=64 spot check accounts for most of it) |
| test_param_count.py (A) | 18 | 0.2 s |
| test_loss_at_init.py (A) | 7 | 1.2 s |
| test_block.py (B) | 14 | 1.5 s |
| test_model.py (B) | 20 | 0.6 s |
| test_causal_mask.py (C, Sprint 2 + appended C5 full model) | 21 | 0.5 s |
| test_gradients.py (C) | 12 | 6.0 s |
| test_weight_tying.py (C) | 9 | 3.6 s |
| Sprint 1+2 files (unchanged) | 246 | each ≤ 0.8 s |

### Sign-off checklist (§11)

- `grep -rn "TODO(SPRINT" nanogpt/` is empty, and the conventions guard is green.
- No torch, jax, tf or autograd.
- `shifted_causal_mask` is referenced in `nanogpt/` only by `attention_mask.py`.
- There is no `lm_head` key in `param_keys()`. `lm_head.W` appears only as an alias in
  `named_parameters()`, where it is the same object as `wte.W`.
- `model.py` contains no `["wte.W"] = ...` subscript assignment (C's AST scan also
  enforces this).
- The corpus SHA-256 is still `7afc055a…2854` and V = 55.
- No Phase 4+ files exist (`optim.py`, `flops.py`, `train.py`, `kv_cache.py`,
  `sampling.py`, `checkpoint.py`, `generate.py`). There is no KV-cache, sampling,
  `generate()` or `train_step()` in `model.py` or `attention.py`.
- `git status` is clean.

### C1 — parameter count

- `formula_param_count(50257, 1024, 768, 12) == 124,439,808`. This is GPT-2 small,
  checked by formula only (R16).
- Live unique-array count equals the formula exactly: tiny **2,232**, default
  **107,712**, max **816,768**. The TL re-measured tiny and default independently.
- One block has 16 tensors totalling 12d² + 13d.
- Test-the-tests:
  - A naive sum double-counts the tie (formula + V·d).
  - A `.T` alias entry is rejected (ValueError).
  - A phantom head bias breaks the equality.
  - All 4 precondition monkeypatches raise.

### C2 — loss at init (real corpus windows, B=4, |loss − ln 55|; INIT_LOSS_TOL = 0.1)

| Config | seed +0 | +1 | +2 |
|---|---|---|---|
| default (V=55, T=64, d=64, L=2, H=4) | 0.0064 | 0.0331 | 0.0155 |
| tiny | 0.0038 | 0.0136 | 0.0051 |
| max | 0.0043 | 0.0103 | 0.0456 |

- Init statistics: biases and β are exactly 0, γ is exactly 1, and every 2-D weight with
  ≥512 entries has std within ±10% of INIT_STD.
- Wrong-init test-the-test (std = 1.0): loss 19.75, delta **15.75**, above the >1.0
  gate. The tolerance was not loosened.

### C6 — full-model gradcheck (tiny d=8/L=2/H=2/T=4, non-init fixture; gated norm-wise rel_err < 1e-6)

- **G1/G2/G3** (logits objective, loss objective, and w.r.t. the float input `h0` per
  TL-18): every tensor except the exempt set passes. The exempt set is asserted to be
  exactly `{h.0.attn.k.b, h.1.attn.k.b}`.
- **B dev gradcheck** (L=1, d=8, V=11): all 20 non-k.b tensors pass. The worst
  max_elem_rel_err is 4.66e-7 (`h.0.mlp.proj.W`).
- **Block level** (B): all 16 non-k.b block tensors plus `x` pass, worst 1.19e-10. The
  residual anchor (zeroed output projections) gives `out == x` exactly.
- **TL independent re-measurement** (own seed `DEFAULT_SEED+950`, params perturbed by
  0.3·N(0,1), loss objective, own central-difference loop, 36 tensors): no non-k.b
  failures. Worst non-k.b rel_err is 1.8e-9 (`h.1.attn.k.W`). For k.b, analytic
  max|g| ≤ 2.1e-17 and FD max_abs_err ≤ 4.4e-11.

### TL-9 reapplied to every block's `attn.k.b` (PO's ratification condition, TL-19)

| Check | Block level (B) | Full model l=0 (C) | Full model l=1 (C) | Threshold |
|---|---|---|---|---|
| (a) analytic max abs db_k | 2.29e-16 | 1.75e-15 | 4.44e-16 | ≤ 1e-12 |
| (b) FD max_abs_err | 1.78e-10 | 7.1e-10 (logits) / 4.4e-11 (loss) | 7.1e-10 / 1.65e-17 | ≤ 1e-8 |
| (c) invariance max abs Δout under b_k + 10·N | 1.33e-15 | 5.1e-15 | 2.3e-15 | ≤ 1e-12 |
| (d) dropped-term adversary max abs db_k | 4.37 | 23.75 | 6.10 | > 1e-3 (discriminates) |

The protocol is applied unchanged at every layer. It is never replaced by the literal
rel_err gate.

### C7 — weight tying

- Partial-sum identity: max abs (g_in + g_out − g_full) is **0.0**, with ‖g_in‖ = 54.6
  and ‖g_out‖ = 51.8 (both non-trivial).
- Independent FD (TL-20; never built from the production split):
  - rel_err(g_out, FD_out) = 2.85e-11
  - rel_err(g_full, FD_full) = 2.43e-10
  - rel_err(g_in, FD_full − FD_out) = 3.10e-10
- When both flags are False, `wte.W` is exactly zero and every other key is
  bit-identical.
- The tied-head anchor `logits == y @ wte.W.T` holds with max_abs 0.0.
- Test-the-tests:
  - An overwriting `accumulate_grads` breaks the identity and fails the gradcheck.
  - A Fortran-order or transposed-view `wte.W` is refused by the contiguity guard.
  - The production array passes.

### C5 re-run on the full model (embeddings + L blocks + ln_f + tied head; Jacobian of logits w.r.t. h0)

- **Correct mask:** future_max = 0.0 and past_min > 0.1 (non-vacuous). Substituting a
  token at position s leaves logits[t<s] bit-identical and changes logits[s].
- **All blocks shifted:** leakage reaches exactly s = t+L (> 0.1) and is zero for
  s ≥ t+L+1. Token substitution independently confirms the leak.
- **One block shifted** (each l): leakage reaches exactly s = t+1 and is zero for
  s ≥ t+2. Identity checks confirm the mask is threaded to each block. A wrong-length
  `mask_fn` list raises.
- **C5 is now retired** at both the isolated (Sprint 2) and full-model (Sprint 3)
  levels.

### Rulings carried

R15 (fc/proj key names), R16 (GPT-2 small formula-only), R17/TL-18 (C6 float input =
`h0`), R18 (no P2 extras), R19/TL-23 (C1/C2 moved to A), TL-16 (three-layer model API),
TL-17 (single `wte.W` array, tied head with no bias), TL-19 (TL-9 per layer), TL-20 (C7
independent FD), TL-21 (`mask_fn` callable-or-list), TL-22 (commit-green gate). All
earlier rulings (TL-1, TL-9, TL-10, R4–R6, R12–R14) still hold.

### Integration fixes made by the TL

**None needed.** HEAD and every interior commit were green, ownership and order were
correct, and all §11 checklist items passed. There are no TL fixup commits this sprint.

### Deviations from SPRINT_3_PLAN.md (for PO ratification)

- **None at the rejection level.** The P0 commit-discipline gate passes for all 9
  engineer commits.
- **D1 (minor, informational):** A's C2 windows (offset DEFAULT_SEED+303) differ from
  the TL prototype's windows, because the plan says only "real corpus". The deltas are
  the same order of magnitude as the TL prototype's.
- **D2 (minor, informational):** C needed a seed-swept fixture draw so the all-shifted
  leak clears the 0.1 bar robustly. This is noted inline in the test, and the
  threshold was not changed.
- **D3 (process, benign):** A and C both hit a shared-index race (another engineer's
  staged files were in the index). Both used `git commit -- <own pathspec>` and checked
  with `git show --stat`. No teammate file was captured.

### Open items / watch items for Sprint 4

- `tests/test_mlp.py` (4.6 s) and `tests/test_gradients.py` (6.0 s) take most of the
  suite's runtime. Keep the full suite under 60 s as Phase 4 tests are added.
- Keep the P0 per-commit isolation gate and the per-commit scratch-tree evidence in
  engineer reports.
- TL-9 applies wherever attention gradients are checked again (for example, any
  optimizer-step gradcheck).

**Next: Phase 4 — Optimizer & Training Loop (Sprint 4).** Scope per the execution
plan:
- `optim.py`: AdamW with decoupled weight decay and correct bias correction. Decide
  explicitly which tensors are excluded from decay (biases, LN γ/β, and the tied
  `wte.W` applied once).
- `flops.py`: the **corrected FLOPs accounting and the C9 check**.
- `train.py`: the training loop on the real corpus, with the loss going down from the
  C2 anchor of about ln 55.

KV-cache (C8), sampling, generation and checkpointing stay out of scope until their
phases.


---

## Phase 4 — Optimizer, FLOP Accounting & Training Loop  [DONE 2026-11-20]

**Process:** same rules as Sprints 2–3 (per-role env-var identities, explicit pathspecs, no history
rewrites, P0 per-commit isolation gate). Sprint 3 sign-off `dfbd373` is an ancestor of HEAD. See
**Deviations** below: this sprint's orchestration died mid-Build and was finished by a single agent.

**Owners / commits (each checked with `git show --name-only`; each touches only its owner's files):**

| Role | Commits | Files |
|---|---|---|
| Engineer A | `515c709` [A1] constants + TL-24 groups; `c1352cd` [A2] AdamW + closed-form anchors; `18e8349` [A3] tied single state + test-the-tests | `nanogpt/optim.py`, `tests/test_adamw.py`; A1 also `nanogpt/config.py` (3 additive constants), `tests/test_config.py` (1 appended test) |
| Engineer B | `bbf5863` [B1] `ops.matmul` chokepoint; `c8205d0` [B2] `FlopCounter` + exact inventory; `a4e6bf2` [B3] C9 band + discriminators | `nanogpt/flops.py`, `tests/test_flops.py`; B1 also `ops.py`, `attention.py`, `model.py` (`@` -> `ops.matmul` only) |
| Engineer C | `e4babc8` [C1] batching; `04cec1b` [C2] loop + CLI; `0e351c9` [C3] loop invariants | `train.py`, `tests/test_train.py` |
| Tech Lead | `7d16bb6` kickoff plan; Sprint 4 sign-off commit (this entry) | `SPRINT_4_PLAN.md`, `PROGRESS.md` |

§9 order holds: `A1<A2<A3`, `A2<C2`, `C1<C2<C3`, `B1<B2<B3`. One milestone per commit.

### Per-commit isolation (P0 gate; TL re-run, `git archive -o` + `tar -xf` into a fresh dir, full suite)

| Commit | Result | | Commit | Result |
|---|---|---|---|---|
| `7d16bb6` kickoff | 363 passed | | `18e8349` [A3] | 459 passed |
| `515c709` [A1] | 378 passed | | `c8205d0` [B2] | 478 passed |
| `e4babc8` [C1] | 387 passed | | `a4e6bf2` [B3] | 495 passed |
| `c1352cd` [A2] | 392 passed | | `04cec1b` [C2] | 502 passed |
| `bbf5863` [B1] | 437 passed | | `0e351c9` [C3] | 516 passed |

**Every interior commit is green.** Sprint 2's D1 was not repeated. Every Sprint 4 commit is a valid
bisect point.

### Real test status (TL run, HEAD `0e351c9`)

`pytest tests/ -q -p no:cacheprovider`: **516 passed**, 25.6 s and 20.4 s on two consecutive runs
(same count and results). The first full-suite run during development took longer (up to 44 s) while
a 14-minute training run was competing for the CPU. That was contention, not a test regression.

| File | Tests | Time |
|---|---|---|
| test_adamw.py (A) | 40 | ~1.5 s |
| test_flops.py (B) | 80 | ~0.8 s |
| test_train.py (C) | 30 | ~0.6 s |
| slowest existing | test_mlp d=64 spot check 3.8-4.5 s; test_weight_tying ~2.5 s; test_gradients fixtures ~1.5 s each | all files < 10 s |

### AdamW (TL-24/25/26)

- **TL independent check** (own 60-digit `Decimal` scalar loop, not the plan's table): max
  |impl - reference| over the 3-step quadratic trajectory is **7.8e-16** for both the decay and the
  no-decay case.
- **Groups (TL-24):** decay = 6L keys (`attn.{q,k,v,o}.W`, `mlp.{fc,proj}.W`) = **98,304** elements
  at the default config; no-decay = 4 + 10L keys = **9,408**. They sum to `formula_from_config`
  = 107,712. Allowlist is explicit, never `ndim`-based; unknown key and `lm_head.W` raise.
- **Tied single state:** `len(opt.state) == len(param_keys()) == 36` at the default config,
  `lm_head.W` absent, `opt.params["wte.W"] is p["wte.W"]`. After one step `m == (1-b1)*g` (one
  accumulation, not two). Adversary with two states moves `wte.W` **~2x** as far (asserted 1.9-2.1).
- **Test-the-tests (all confirmed by passing assertions):** off-by-one t, no bias correction,
  coupled L2, decay on an excluded key each miss the section 3.2 table by > 1e-6; t from 0 gives a
  non-finite first update; two-state tied adversary caught.
- **No test asserts that a loss decreased**, and `test_train.py` enforces this with an AST guard
  (itself test-the-tested on synthetic sources).

### C9 — FLOPs per token (instrumented counter at the single `ops.matmul` chokepoint, TL-27)

TL re-measured on its own seed (B=2), five configs. Counts equal the exact matmul inventory
(24Ld^2 + 2Vd dense, 4LTd attention) **to the integer**; backward = 2x forward exactly; train/fwd
= 3.0000.

| Config (V=55) | fwd/tok | corrected 2N+4LTd | ratio | naive 2N ratio | attn share |
|---|---|---|---|---|---|
| tiny d=8 T=4 L=2 | 4,208 | 4,720 | 0.8915 | 0.9427 | 6.1% |
| default d=64 T=64 L=2 | 236,416 | 248,192 | 0.9526 | 1.0974 | 13.9% |
| long d=64 T=128 L=4 (in range) | 531,328 | 554,624 | 0.9580 | **1.2545 (fails band)** | 24.7% |
| max d=128 T=128 L=4 | 1,849,088 | 1,895,680 | 0.9754 | 1.1320 | 14.2% |
| discriminator d=16 T=128 L=2 (out of range) | 30,432 | 35,424 | 0.8591 | **1.5983 (fails band)** | 53.8% |

(Attention share here is of the *measured* matmul total. The plan's TL-30 column used the share of
the *corrected formula* total, which gives 13.2% / 13.8% at default / max. Both lie in the spec's
13-20% at default and max.)

The naive 2N/6N formula **passes** the band at tiny, default and max. It fails only at the two
attention-heavy configs. So C9 is discriminating because of the integer-exact inventory plus those
configs, not because of the band alone; the tests assert this on purpose. Test-the-tests: an
attention-blind counter fails the exact inventory and the discriminator band; a backward pass
outside the counter gives train/tok ~ fwd/tok (ratio ~0.32) and fails the training band.
**One real training step under `FlopCounter`** (B=16, default config) equals the exact inventory,
and `AdamW.step` adds **0** matmuls. The TL-27 subclass counter was not re-implemented (AST-checked).

### Chokepoint refactor (B1) — regression proof

12 production `@` sites became `ops.matmul(...)`. Bit-identical results: the 363 pre-existing tests
passed unmodified at `bbf5863` (437 total with the new B1 tests). A new AST guard fails on any bare
`@` outside `ops.matmul`, on `np.dot/matmul/einsum/tensordot/inner/vdot`, on any `.dot(`, and on
`from nanogpt.ops import matmul`; it is itself tested on synthetic sources. `attention_reference.py`
is excluded.

### Real training run (TL, outside pytest) — C-budget check

`python -u train.py` at the defaults (`ModelConfig(V=55, T=64, d=64, L=2, H=4)`, N = 107,712,
batch 16, lr 1e-3 constant, wd 0.1, betas (0.9, 0.99), 4000 steps, seed 1337, synthetic corpus
SHA `7afc055a`):

- **841.1 s = 14.0 min, 0.210 s/step.** Budget is 30 min. Headroom ~2.1x.
- Step-0 loss **4.0333** vs ln 55 = 4.0073 (consistency print only; C2 already proves this).
- Loss log every 100 steps, **record only, not evidence of correctness**: 2.524 (100), 1.897 (500),
  1.197 (1000), 1.142 (2000), 1.098 (3000), 1.138 (3999). The loss plateaus near 1.1 because the
  synthetic corpus has irreducible entropy, not because of a defect. This is not used for anything.
- Profile (20 steps, `--profile 20`): ~72% of time in backward, ~28% forward; `ops.matmul` itself is
  ~4.2 of 6.9 s under cProfile; `gelu_forward` + `gelu_backward` are the next largest (~20%).
  No trimming of batch/steps/L/d/T was needed.

### Rulings (TL-24..TL-31, R22..R27) and ratification

R22 no checkpointing/save hook this sprint; R23 `tests/test_adamw.py`; R24 `train.py` at repo root;
R25 C9 configs = tiny, default, long, max + out-of-range discriminator; R26 split A=optim, B=flops,
C=train; R27 no LR schedule / val split / README edit. TL-24 decay groups, TL-25 canonical names +
in-place updates, TL-26 `t` starts at 1, TL-27 chokepoint, TL-28/29 FLOP counting + exact
inventory, TL-30 C9 configs, TL-31 `train.py` contract and budget.

**FOR PO RATIFICATION (unratified): TL-24 — `wpe.W` is excluded from weight decay.** The spec says
"weight matrices". We read that as the linear maps (minGPT convention). Karpathy's nanoGPT decays
every >= 2-D tensor including embeddings, so this is a genuine reading choice. The decision is a
one-line change to `NO_DECAY_PATTERNS`, covered by `test_decay_groups_match_param_keys_exactly`.

### Static checks (TL)

`grep "TODO(SPRINT" nanogpt/` empty; conventions guard green; no torch/jax/tf/autograd; no bare
`1e-8` in `optim.py` (it imports `ADAMW_EPS`); no `lm_head` key in `param_keys()` or `opt.state`;
chokepoint guard green. **No Phase 5+ files exist** (`kv_cache.py`, `sampling.py`, `checkpoint.py`,
`generate.py`, `tests/test_kv_cache.py`, `tests/test_numerical_stability.py`); `train.py` has no
save/load, sampling or generation. Corpus SHA-256 still `7afc055a...2854`, V = 55.

### Deviations from SPRINT_4_PLAN.md

- **D-S4-1 (process, material).** The Workflow orchestration did not finish. Planning ran (TL plan
  committed as `7d16bb6`); Build produced `[A1]` and `[C1]` as commits and left `[A2]` + `[B1]`
  staged but uncommitted. `[A2]`/`[B1]` were split into their proper commits, and `[A3]`, `[B2]`,
  `[B3]`, `[C2]`, `[C3]` were written by a single agent directly from the plan, committed under the
  `Engineer A/B/C` identities the plan assigns to those files. Authorship on those five commits is
  role-accurate, not agent-accurate. Each commit passed the section 8.2 pre-commit check
  (committed tree + only that milestone's files) before landing, and the isolation table above was
  re-run afterwards.
- **D-S4-2 (process, material). No PO pass was run for Sprint 4**: no independent fresh-seed
  re-verification, mutant injection or accept/reject verdict by a PO. The TL re-measurements above
  are the only independent verification. TL-24 is therefore unratified. Both items are carried into
  Sprint 5's acceptance phase unless the user asks for a PO pass first.
- **D-S4-3 (minor).** `git archive HEAD | tar -x` corrupts the stream under PowerShell; the gate was
  run via `git archive -o <file>.tar` then `tar -xf`.
- **D-S4-4 (minor).** `python train.py` buffers stdout when piped; use `python -u` for a background run.
- **D-S4-5 (minor).** `train.py` gained an additive `profile_steps` keyword on `train()` (section 3.5
  lists no such argument) so that `--profile N` can cProfile the first N steps of the real loop.
  It defaults to 0 and changes nothing otherwise.

### Watch items for Sprint 5

- **Phase 5 changes the forward path structurally.** The plan (`SPRINT_5_PLAN.md`, TL-32) makes the
  cached path purely additive (`forward_step`/`prefill`) so that C5/C6/C7/C9 stay signed off by the
  unmodified Sprint 1-4 suite. TL-9 is to be reconfirmed on HEAD at sign-off regardless.
- Every new matmul must go through `ops.matmul` (`kind="attention"` for the per-head products) or
  the chokepoint guard fails.
- The conventions guard scans only `nanogpt/*.py`; root-level `train.py` and the coming `generate.py`
  are covered by discipline and by their own AST guards, not by that scan.
- Suite runtime: 516 tests ~20-25 s. Budget 60 s.

**Next: Phase 5 — Generation (Sprint 5).** `kv_cache.py`, `sampling.py`, `checkpoint.py`,
`generate.py`, and C8 (teacher-forced, length <= T, 1e-10 absolute).


---

## Phase 5 — Generation: KV-Cache, Sampling, Checkpointing  [DONE 2026-11-20]

**Process:** as Sprints 3-4 (per-role env-var identities, explicit pathspecs, no history rewrites,
P0 per-commit isolation gate). This sprint was run by a single agent directly from
`SPRINT_5_PLAN.md` (no Workflow orchestration, no PO pass); see Deviations. Sprint 4 sign-off
`8eb0b17` is an ancestor of HEAD.

| Role | Commits | Files |
|---|---|---|
| Engineer A | `5fc9310` [A1] KVCache; `b0c1c0a` [A2] cached forward; `3ffcd09` [A3] C8 + adversaries + FLOP closed form | `nanogpt/kv_cache.py`, `tests/test_kv_cache.py`; A2 also `attention.py`, `block.py`, `model.py` (**additions only**) |
| Engineer B | `4f9e37e` [B1] temperature + greedy; `13bb2ab` [B2] top-k + sample + 4 bug variants | `nanogpt/sampling.py`, `tests/test_sampling.py` |
| Engineer C | `862bc27` [C1] checkpoint; `fa248b9` [C2] generate.py | `nanogpt/checkpoint.py`, `generate.py`, `tests/test_checkpoint.py`, `tests/test_generate.py` |
| Tech Lead | `838b9c5` kickoff plan; `33fccd6` TL fixup (TL-40); sign-off commit (this entry) | `SPRINT_5_PLAN.md`, `train.py`, `tests/test_train.py`, `PROGRESS.md` |

Order holds: `A1<A2<A3`, `B1<B2`, `C1<C2`, `{A2,B2}<C2`. One milestone per commit.

### Per-commit isolation (P0 gate; TL re-run via `git archive -o` + `tar -xf`, full suite)

`5fc9310` 537 | `b0c1c0a` 546 | `3ffcd09` 563 | `4f9e37e` 589 | `13bb2ab` 624 | `862bc27` 641 |
`fa248b9` 664 | `33fccd6` 665 — **all passed.** Every Sprint 5 commit is a valid bisect point, and
each touches only its owner's files (checked with `git show --name-only`).

### Real test status (TL run, HEAD `33fccd6`)

`pytest tests/ -q`: **665 passed**, 27.7 s and 22.5 s on two consecutive runs. New files: test_kv_cache
(45 tests, ~0.6 s), test_sampling (59, ~0.3 s), test_checkpoint (15, ~1.1 s),
test_generate (23, ~1.2 s). The Sprint 4 chokepoint guard in `test_flops.py` (now 83 tests) parametrizes over
`nanogpt/*.py`, so it scanned the three new modules automatically and found no bare `@`.
All files < 10 s; whole suite well under 60 s.

### C8 — KV-cache equivalence (TL independent re-measurement, own seed, own full-forward reference)

Teacher-forced on a fixed sequence of length exactly `cfg.T` (the boundary of the "length <= T"
precondition), batch 2, perturbed (non-trivial) weights, three prefill/decode splits each. Gate is
**absolute**, 1e-10 (`KV_CACHE_ABS_TOL`):

| Config | worst max abs diff | logit scale |
|---|---|---|
| tiny d=8 T=4 | 8.9e-16 | 2.6 |
| default d=64 T=64 | 3.3e-14 | 8.6 |
| max d=128 T=128 L=4 | 2.6e-13 | 13.6 |

A cached-vs-prefix check also confirms logits at position i equal the full forward of just the
prefix `[0..i]`: later tokens cannot influence them.

**Test-the-tests (each caught):** shifted mask => cached and full **disagree** (3.08 max abs; the
full path leaks one future token that the cache structurally cannot see — C8's 1e-10 claim is made
for `causal_mask` only, TL-34); stale cache (advance never bumps `length`) => diverges; wrong
position embedding (`pos+1`) => diverges; feeding a different teacher-forced token changes every
later logit. **Honest record:** a cached path that *ignores* its mask row is bit-identical under
`causal_mask`, so C8 **cannot** detect it today. TL-33 (honour `mask_fn`) is therefore a robustness
ruling, not a discriminated one; a test shows the converse (a restrictive mask row does change
logits) so the hook is not dead code.

### Cache cost (instrumented, Sprint 4's `FlopCounter`)

Per-step cached matmul FLOPs equal the hand-derived closed form `24Ld^2 + 2Vd + 4Ld(pos+1)` **to the
integer** at every position, both configs. Generating 32 tokens after a 32-token prompt at the
default config: **7.3M FLOPs cached vs 356.0M recomputing the full forward each step = 48.7x fewer.**

### Training path untouched (TL-32) — regression proof

`git diff --numstat 838b9c5..HEAD`: `attention.py` +49/-0, `block.py` +10/-0, `model.py` +42/-0.
Only new methods (`forward_step`, `prefill`) were added; `forward`, `backward`, `embed`,
`forward_from_embedding`, `forward_loss`, `backward_loss` are byte-identical. No Sprint 1-3 test
file changed. The only edited pre-existing test file is `tests/test_train.py` (see TL-40 below).
Consequently **C5, C6, C7 and C9 stay signed off by the unmodified suite**, which passes.
**TL-9 reconfirmed on HEAD** for each block's `attn.k.b` (tiny, perturbed params, own seed):
(a) analytic max|g| 3.0e-18 / 6.0e-18 (<= 1e-12); (b) FD max|g| 0.0 (<= 1e-8); (c) invariance
max|dlogits| 7.8e-16 / 6.7e-16 (<= 1e-12); (d) the dropped-term adversary is the unmodified
`tests/test_gradients.py` case, which passes.

### Sampling, checkpointing, generate

- **Sampling:** closed-form anchors (temperature identity at 1.0, `log(p_a/p_b) = (a-b)/t`, monotone
  convergence to uniform, `top_k=V` no-op, exactly k finite per row, `top_k=1` == greedy at any
  temperature, masked probabilities exactly 0, rows sum to 1, deterministic tie-break by index, CDF
  edge cases `u=0` and `u=1-eps`). A clearly-labelled statistical check (20k draws, 4 SE) is *not* a
  correctness anchor. **Four buggy variants are each caught** by a shared anchor battery that the real
  implementation passes: top-k keeps the smallest; top-k off by one; sampling from un-renormalised
  probabilities; temperature applied as multiply. `temperature=0` raises (no silent argmax).
- **Checkpoint:** params round-trip bit-identically; loaded keys are exactly `param_keys()`; `wte.W`
  is one contiguous array and `named_parameters()["lm_head.W"] is params["wte.W"]`; AdamW `t`, `m`,
  `v` and hyperparameters round-trip exactly and `restore_optimizer` shares the loaded arrays by
  identity. **Resume equivalence** (train 3, save, load, step once == train 4 uninterrupted) is
  **bit-identical**, TL-reproduced independently; a fresh-optimizer "resume" is shown to differ.
  Rejected: `param/lm_head.W`, wrong/missing format, missing/extra param keys, incomplete or
  misshapen optimizer state, and pickled object arrays (`allow_pickle=False`, AST-checked).
- **generate.py:** deterministic per seed; cached greedy generation equals a naive loop that re-runs
  the full non-cached forward each step; fills the window exactly and refuses `len(prompt)+tokens > T`;
  rejects out-of-vocab prompts, bad ids/dtypes/counts, a non-empty or batch>1 cache, and a vocab
  mismatch with a checkpoint. **No test asserts anything about the content of generated text.**
- **End to end (TL, record only):** `train.py --steps 300 --save <ckpt>` then
  `generate.py --checkpoint <ckpt> --prompt "the " --tokens 40 --top-k 8` runs and prints text. This
  demonstrates the pipeline works; it is not evidence of correctness.

### Rulings (TL-32..TL-40)

TL-32 cached path additive (new methods, no signature changes, no backward/gradcheck); TL-33 cached
attention uses the last row of `mask_fn(pos+1)`; TL-34 shifted mask is C8's discriminator and
cached != full there is correct; TL-35 pre-allocated cache, strict layer order, hard capacity `T`,
no sliding window; TL-36 absolute-position embedding read before layers append, `prefill` loops
`forward_step`; TL-37 sampling contract; TL-38 checkpoint format (`np.savez`, `allow_pickle=False`);
TL-39 `generate.py` contract. **TL-40 (new, recorded deviation from the plan):** `train.py` gains an
optional `--save PATH` (calls `checkpoint.save`) so train -> generate works from the CLI. This needed
one edit to a Sprint 4 test file (`tests/test_train.py`): the R22 scope guard now permits importing
`checkpoint` only (kv_cache/sampling/generate/pickle/np.save remain forbidden), the matching synthetic
case was removed, and two tests were added. Shipped as `TL fixup` `33fccd6`.

### Deviations from SPRINT_5_PLAN.md

- **D-S5-1 (TL-40, above).** One Sprint 4 test file edited, deliberately and disclosed, to make
  the pipeline usable end to end. All other pre-existing tests are untouched.
- **D-S5-2 (process, material, carried).** No PO pass for Sprint 4 *or* Sprint 5; both sprints were
  signed off by TL re-measurement only, and Sprint 5 was implemented by one agent (commits carry role
  identities). **TL-24 (`wpe.W` no-decay) remains unratified.** Carried to the Phase 8 final gate,
  where the plan already reserves an independent high-effort PO pass.
- **D-S5-3 (minor).** Two weak tests were written first in `test_generate.py` (one vacuous assertion,
  one unused variable) and fixed before commit; no vacuous assertion was committed.

### Watch items for Sprint 6

- Phase 6 is the large-logit pass at **full-model** level (logits ~1e3) plus re-confirmation that T=1,
  the shifted mask and the cached path stay green: `tests/test_numerical_stability.py` does not exist
  yet. Stable softmax/CE already survive 1e3 logits at unit level (`test_loss.py`).
- `softmax_forward` has no all-`-inf` row protection; the cached path never produces one (the
  diagonal is always allowed), but a future mask change could.
- Suite ~22-28 s of the 60 s budget.

**Next: Phase 6 — Numerical stability (large-logit, full model), then Phase 8 wrap-up.**
