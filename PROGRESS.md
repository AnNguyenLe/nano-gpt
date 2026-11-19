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
