# Sprint 5 — Technical Plan: Phase 5 (Generation: KV-Cache, Sampling, Checkpointing)

**Author:** Tech Lead. **Inputs:** `nanoGPT_build_requirements.md` (§1.3, C8), `nanoGPT_execution_plan.md`
§Phase 5, `PROGRESS.md` (Phases 0–4), `SPRINT_4_PLAN.md` (TL-24..TL-31, still binding), and
`HANDOFF.md` §5/§5a/§6.1 (accumulated watch-items and the split this plan implements).

**Scope: Phase 5 only.** `nanogpt/kv_cache.py`, `nanogpt/sampling.py`, `nanogpt/checkpoint.py`,
root `generate.py`, the inference-only `forward_step` path through `attention.py`/`block.py`/
`model.py`, and their tests. The Phase-6 large-logit stability pass, GPT-2 weight loading, the
README rewrite and any LR-schedule/perf work stay out of scope (§13).

**The non-negotiable framing for this sprint.** Sprint 3 signed off C5/C6/C7 and Sprint 4 signed
off C9 against the *existing* `forward`/`backward`. Nothing in this sprint may invalidate that.
The cached path is **additive** (TL-32): new methods, no changed signatures, no edits to any
Sprint 1–4 test file. A decreasing loss remains inadmissible as evidence anywhere, and generated
text that "looks like English" is **never** evidence that the KV-cache is correct — only C8's
numeric equivalence counts.

---

## 1. Sprint 5 Definition of Done

1. **C8** (`tests/test_kv_cache.py`): token-by-token cached logits equal a full-forward pass over
   the same sequence to within `KV_CACHE_ABS_TOL = 1e-10` **absolute**, under the spec's
   preconditions — teacher-forced on a fixed token sequence, length ≤ `cfg.T`. Verified on tiny
   and default configs and at the exact boundary length `T`.
2. **C8 is proven discriminating** (TL-34): with `shifted_causal_mask` substituted the cached and
   full paths **disagree**, and that disagreement is asserted. Plus a stale-cache adversary, a
   wrong-position-embedding adversary, and a "cached path ignores its mask row" adversary.
3. **The training path is byte-identical.** `forward`, `backward`, `forward_loss`,
   `backward_loss` and every public signature in `ops`/`attention`/`block`/`mlp`/`layernorm`/
   `loss`/`model` are unchanged. All Sprint 1–4 tests pass **unmodified**, which is the
   regression proof for C5/C6/C7/C9.
4. **TL-9 reconfirmed** on HEAD for every block's `attn.k.b`, using the same (a)–(d) protocol.
   The backward is untouched, so this is a confirmation, not a re-derivation — but it is recorded,
   per the standing rule that TL-9 is re-verified at every structural change.
5. **Sampling** (`nanogpt/sampling.py`): temperature scaling and top-k, each with closed-form
   anchors (§3.3), deterministic under a seeded rng, and four buggy variants proven to fail.
6. **Checkpointing** (`nanogpt/checkpoint.py`): save/load round-trips model params **and** AdamW
   state (`t`, `m`, `v`, hyperparameters) bit-identically, restores the tie as **one** array, and
   uses `allow_pickle=False`. A resumed optimizer takes the same next step as one that never
   stopped.
7. **`generate.py`** at the repo root: loads a checkpoint (or `--random-init`), generates with the
   KV cache and the sampling module, is deterministic for a fixed seed, and runs in well under a
   second for a short sample at the default config.
8. **FLOP evidence for the cache** (A3): measured with Sprint 4's `FlopCounter`, generating `n`
   tokens incrementally costs strictly fewer matmul FLOPs than `n` full forward passes over the
   growing prefix, and the per-step cached count matches a hand-derived closed form.
9. **Commit discipline (P0 hard gate, unchanged):** every commit green when its own tree is
   checked out alone, verified per commit by the TL, not by self-report. One milestone per commit.
10. Suite green, < 60 s total, every file < 10 s (new files reported explicitly), deterministic
    over two runs; `PROGRESS.md` Phase 5 entry appended with "Next: Phase 6".

---

## 2. Tech Lead rulings (TL-32 .. TL-39)

**TL-32. The cached path is additive. This is the single most important ruling of the sprint.**
Do **not** add a `past=` / `kv=` argument to `CausalSelfAttention.forward`, `Block.forward`,
`GPT.forward`, `GPT.embed` or `GPT.forward_from_embedding`. Add new methods instead:

| Module | New method |
|---|---|
| `CausalSelfAttention` | `forward_step(x_step, params, kv=None, layer=0)` |
| `Block` | `forward_step(x_step, params, kv=None, layer=0)` |
| `GPT` | `forward_step(idx_step, params, kv)` and `prefill(idx, params, kv)` |

Rationale: the existing `forward`/`backward` pair carries C5, C6, C7 and the TL-9 protocol. A
signature change re-opens every one of those sign-offs. Additive methods leave the training path
bit-identical, so the Sprint 1–4 suite is itself the regression proof. **`forward_step` has no
backward and is never gradchecked** — it is inference-only, and `tests/test_gradients.py` must not
grow a cached-path case. If the cached forward is wrong, C8 catches it; that is the whole point of
C8 existing.

**TL-33. The cached path must honour `mask_fn`, not assume causality.**
At absolute position `pos`, with `pos+1` entries cached (the new token included), the allowed keys
are `self.mask_fn(pos + 1)[pos]` — the **last row** of the mask for a length-`(pos+1)` sequence,
shape `(pos+1,)`. Apply it exactly as the full forward does: `Z = np.where(allowed_row, S, -inf)`
then `ops.softmax_forward(Z)`. For `causal_mask` that row is all-True, so no `-inf` ever appears
and the softmax is over `pos+1` finite scores. Do **not** shortcut this to "attend to everything
in the cache": the mask row is what makes TL-34 meaningful and what keeps the cached path honest
if a future sprint changes the mask.

**TL-34. The shifted mask is C8's discriminator, and `cached ≠ full` is the correct result there.**
Under `shifted_causal_mask` (`j ≤ i+1`) the full forward leaks one future token; the cached path
*cannot*, because at the time position `i` is decoded, position `i+1` has not been computed. So the
two paths legitimately disagree. Assert that disagreement (max abs difference > 1e-3 on a
non-degenerate fixture). **C8's 1e-10 claim is made for `causal_mask` only**, and the sign-off
table must say so. This also means the cached path is *not* a second implementation of C5: C5 stays
retired on the `forward` path where it was proven.

**TL-35. `KVCache` layout and write discipline.**
- Pre-allocated `float64` buffers of shape `(L, B, H, T, head_dim)` for K and V, built from a
  `ModelConfig` plus a batch size. No `np.concatenate` per step, no reallocation.
- `length: int` is the number of fully-written positions, starting at 0.
- `append(layer, kh_step, vh_step) -> (kh_all, vh_all)` writes the step at index `length` and
  returns views covering `[0, length]` inclusive, shape `(B,H,length+1,head_dim)`.
- **Strict ordering**, enforced with `ValueError`: within one step, layers must be appended exactly
  once each, in order `0..L-1`; then `advance()` increments `length`. Appending out of order,
  twice, or calling `advance()` before every layer has been appended is an error. This is what
  catches the "stale cache" and "layer mix-up" bug classes by construction.
- `length == cfg.T` is the hard ceiling (C8's precondition). `append` beyond it raises. **There is
  no sliding window** — a window would shift absolute positions and invalidate C8, exactly as the
  spec warns.
- `reset()` zeroes `length` (buffers need not be cleared, since only `[0, length)` is ever read).

**TL-36. Position indexing.**
`GPT.forward_step` embeds the new token with `wpe` row `pos = kv.length` (read **before** the
layers append). That absolute index is what makes the cached logits equal the full forward's logits
at that position. `prefill(idx, params, kv)` runs positions `0..T_prefix-1` by calling
`forward_step` in a loop — simple, obviously correct, and it keeps one code path instead of two.
A batched prefill is a performance optimisation and is **out of scope**.

**TL-37. Sampling contract.**
- `sampling.py` works on a logits array of shape `(..., V)`, never on the model.
- `temperature=0` **raises** rather than being special-cased to argmax: `logits/0` overflows, and
  silently branching to argmax hides the division. Greedy decoding is the separate, explicit
  `greedy(logits)`.
- `apply_temperature(logits, t)` returns `logits / t` for `t > 0`.
- `apply_top_k(logits, k)` sets all but the k largest entries of each row to `-inf`, requires
  `1 <= k <= V`, and is an exact no-op at `k == V`. Ties are broken by index (`np.argpartition`
  order is not relied on: sort the kth threshold explicitly), so the result is deterministic.
- `sample(logits, rng, *, temperature=1.0, top_k=None)` composes the above, converts to
  probabilities with `ops.softmax_forward` (so masked entries become exactly 0 and the
  renormalisation is automatic and stable), and draws with `rng.random()` against a cumulative sum.
  Returns `INT_DTYPE` ids.
- Every function is pure: it never mutates its input.

**TL-38. Checkpoint format.**
- `np.savez` (uncompressed) with a flat key scheme and **`allow_pickle=False` on load**:
  `format` (version string), `cfg_V/cfg_T/cfg_d/cfg_L/cfg_H`, `param/<key>` for every
  `param_keys()` entry, `opt_t`, `opt_lr`, `opt_beta1`, `opt_beta2`, `opt_eps`, `opt_weight_decay`,
  and `opt_m/<key>`, `opt_v/<key>`.
- Saving the optimizer is optional (`save(path, model, params, opt=None)`), loading detects it.
- `load(path)` returns `(cfg, params, opt_state_or_None)`; `restore_optimizer(params, opt_state)`
  rebuilds an `AdamW` whose `params` are the **same array objects** as the loaded `params` and
  whose `t`/`m`/`v` match what was saved.
- The tie is restored **by construction**: `params` has exactly `param_keys()` (so no
  `lm_head.W` entry), `wte.W` is one contiguous array, and `named_parameters` re-aliases it. A
  checkpoint containing a `param/lm_head.W` key is rejected.
- No pickle, no custom objects, no `state_dict` on `AdamW` itself (the optimizer stays a plain
  object; serialisation lives in `checkpoint.py`).

**TL-39. `generate.py` contract.**
Root-level CLI, same shape as `train.py`. `--checkpoint PATH` or `--random-init`,
`--prompt TEXT`, `--tokens N`, `--temperature`, `--top-k`, `--seed`, `--greedy`. Resolves the
tokenizer from the corpus (so the vocab matches), refuses a prompt containing characters outside
the vocab with a clear error, refuses `len(prompt) + tokens > cfg.T` (TL-35: no sliding window),
and prints the completion. Deterministic for a fixed seed.

---

## 3. Frozen interfaces (all engineers code to these)

### 3.1 `nanogpt/kv_cache.py` (Engineer A)

```python
class KVCache:
    def __init__(self, cfg: ModelConfig, B: int): ...
    cfg: ModelConfig ; B: int ; length: int
    def append(self, layer: int, kh_step, vh_step) -> tuple   # (kh_all, vh_all), TL-35
    def advance(self) -> None
    def reset(self) -> None
    @property
    def capacity(self) -> int        # cfg.T
```
`kh_step`/`vh_step` are `(B, H, 1, head_dim)` float64. Shape, dtype, layer-order and capacity
violations all raise `ValueError`.

### 3.2 Cached forward (Engineer A; additive, TL-32)

```python
# CausalSelfAttention
def forward_step(self, x_step, params, kv=None, layer=0):
    # x_step: (B,1,d). With kv=None this is the pos=0 case (no history).
    # Returns out_step: (B,1,d). Appends this step's K/V into kv.
# Block
def forward_step(self, x_step, params, kv=None, layer=0): ...   # pre-LN, same residual structure
# GPT
def prefill(self, idx, params, kv):        # (B,T0) -> logits (B,T0,V); loops forward_step
def forward_step(self, idx_step, params, kv):   # (B,1) -> logits (B,1,V)
```
Every new matmul goes through `ops.matmul`, `kind="attention"` for the two per-head products
(`q_step @ kh_all^T` and `P @ vh_all`) and `kind="dense"` for everything else. The Sprint 4 AST
guard will fail the build otherwise.

### 3.3 `nanogpt/sampling.py` (Engineer B)

```python
def apply_temperature(logits, temperature: float): ...     # t > 0, else ValueError
def apply_top_k(logits, k: int): ...                        # 1 <= k <= V, no-op at k == V
def greedy(logits): ...                                     # argmax over the last axis -> INT_DTYPE
def sample(logits, rng, *, temperature: float = 1.0, top_k=None): ...
```

**Closed-form anchors Engineer B must hit:**
- `apply_temperature(l, 1.0)` is `l` exactly (`np.array_equal`).
- Softmax after temperature `t` equals `softmax(l/t)`; as `t` grows the distribution converges to
  uniform (max prob → 1/V monotonically over `t = 1, 10, 100, 1000`).
- For two logits `a > b`, `p_a/p_b = exp((a-b)/t)` exactly (to `ROUNDOFF_ATOL` in log space).
- `apply_top_k(l, 1)` leaves exactly one finite entry, at `argmax(l)`; `greedy` picks it.
- `apply_top_k(l, V)` is a no-op; `apply_top_k(l, k)` leaves exactly `k` finite entries per row.
- Probabilities after top-k sum to 1 and are exactly 0 on the masked entries.
- A fixed `rng` gives a reproducible id sequence; two rngs with the same seed agree.
- With `top_k=1`, `sample` equals `greedy` for every draw, at any temperature.
- Over 20k draws from a 3-way logit vector, the empirical frequencies match the softmax
  probabilities within 4 standard errors (a statistical sanity check, clearly labelled as such —
  not a correctness anchor).

**Test-the-tests (each must fail):** (1) top-k that keeps the k *smallest*; (2) top-k off by one
(keeps `k-1`); (3) sampling from un-renormalised probabilities after masking; (4) temperature
applied as `logits * t` instead of `logits / t`.

### 3.4 `nanogpt/checkpoint.py` (Engineer C)

```python
FORMAT = "nanogpt-numpy-ckpt-v1"
def save(path, model, params, opt=None) -> None
def load(path) -> tuple          # (cfg, params, opt_state | None)
def restore_optimizer(params, opt_state) -> AdamW
```

### 3.5 `generate.py` at the repo root (Engineer C)

```python
DEFAULT_TOKENS = 200
def generate(model, params, kv, prompt_ids, n_tokens, rng, *, temperature=1.0, top_k=None,
             greedy_decode=False) -> np.ndarray      # (n_tokens,) ids
def main(argv=None) -> int
```

---

## 4. File ownership (no file has two owners)

| Owner | Creates | Modifies |
|---|---|---|
| Engineer A | `nanogpt/kv_cache.py`, `tests/test_kv_cache.py` | `nanogpt/attention.py`, `nanogpt/block.py`, `nanogpt/model.py` (**additive only**: new `forward_step`/`prefill`) |
| Engineer B | `nanogpt/sampling.py`, `tests/test_sampling.py` | none |
| Engineer C | `nanogpt/checkpoint.py`, `generate.py`, `tests/test_checkpoint.py`, `tests/test_generate.py` | none |
| Tech Lead | `SPRINT_5_PLAN.md` | `PROGRESS.md`, `TL fixup:` commits if needed |

**No Sprint 1–4 test file may be edited.** If one fails after your change, your change is wrong —
that is the whole regression argument for C5/C6/C7/C9 this sprint.

---

## 5. Milestones and commit order

```
A1 kv_cache.py ──> A2 forward_step (attn/block/model) ──> A3 C8 + test-the-tests + FLOP evidence
B1 sampling: temperature + greedy ──> B2 top-k + sample + test-the-tests
C1 checkpoint.py + round-trip ───────────────────────────> C2 generate.py  (needs A2 + B2)
```
Required order in `git log`: `[A1] < [A2] < [A3]`, `[B1] < [B2]`, `[C1] < [C2]`, and
`{[A2], [B2]} < [C2]`. One milestone per commit. No stand-in or `TODO(SPRINT5-…)` marker is ever
committed.

---

## 6. Acceptance criteria per engineer

### 6.1 Engineer A — `tests/test_kv_cache.py` (< 6 s)

- **A1 cache mechanics:** shapes/dtype validated; `length` grows only on `advance()`; views have
  the right length; out-of-order, duplicate and post-`advance()`-less appends raise; capacity
  `cfg.T` enforced; `reset()` works; a second step does not disturb the first step's stored K/V.
- **A2 single-step:** with an empty cache, `forward_step` on a 1-token input equals `forward` on
  the same 1-token input, to `ROUNDOFF_ATOL`. (This is the degenerate `T=1` case, which Sprint 2
  already characterised.)
- **A3 C8 (the gate):** for tiny and default configs, prefill a `T0`-token prefix then decode the
  remaining tokens one at a time, teacher-forced from a fixed sequence; every cached logit vector
  matches the corresponding row of a single full forward over the whole sequence to
  `KV_CACHE_ABS_TOL = 1e-10` absolute. Report the **max absolute** difference, not a relative one
  (the spec states C8 in absolute terms). Include the boundary case `total length == cfg.T`.
- **A3 test-the-tests:** (1) shifted mask ⇒ cached and full disagree (TL-34); (2) stale cache — do
  not `advance()` and the logits diverge; (3) wrong position embedding (`pos+1`) ⇒ diverges;
  (4) a cached path that ignores its mask row is *indistinguishable* under `causal_mask` and
  **must be shown to be so**, with a comment recording that TL-33 is therefore a
  future-proofing/robustness ruling rather than something C8 can detect today — do not claim a
  discrimination that does not exist.
- **A3 FLOP evidence:** under `FlopCounter`, generating the last `n` tokens incrementally costs
  strictly fewer FLOPs than `n` full forwards over the growing prefix, and the per-step cached
  count equals a hand-derived closed form (`24Ld² + 2Vd` dense plus `4L·(pos+1)·d` attention for
  one token).
- Sprint 1–4 suite passes unmodified.

### 6.2 Engineer B — `tests/test_sampling.py` (< 3 s)

Everything in §3.3, including the four failing variants and the clearly-labelled statistical check.

### 6.3 Engineer C — `tests/test_checkpoint.py`, `tests/test_generate.py` (< 5 s combined)

- **C1:** save/load round-trips every param bit-identically (`np.array_equal`); the loaded
  `params` keys are exactly `param_keys()`; `wte.W` is one contiguous array and
  `named_parameters()["lm_head.W"] is params["wte.W"]`; the optimizer's `t`, `m`, `v` and
  hyperparameters round-trip exactly; `restore_optimizer` produces an optimizer whose `params` are
  the loaded arrays by identity. **Resume equivalence:** train 3 steps, save, load, step once more
  → bit-identical to training 4 steps without stopping. `load` on a file with
  `param/lm_head.W`, a wrong `format` string, or a missing param key raises. `allow_pickle=False`
  is asserted by loading a deliberately pickled object and confirming it is refused.
- **C2:** `generate` is deterministic for a fixed seed and differs for a different seed; output
  length and dtype are right; `greedy_decode=True` matches `top_k=1`; a prompt with an
  out-of-vocab character raises; `len(prompt) + tokens > cfg.T` raises; `main([... --random-init
  ...])` returns 0 and prints a completion of the requested length. No test asserts anything about
  the *content* of generated text.

---

## 7. Git protocol

Identical to SPRINT_4_PLAN.md §8, with identities `Engineer A|B|C` / `engineer-x@nanogpt-numpy.local`
and `Tech Lead` / `tech-lead@nanogpt-numpy.local`. Subjects: `Phase 5 [<milestone>]: <what>`.
Commit with an explicit pathspec. Never `git add -A`.

**Windows note (D-S4-3):** `git archive HEAD | tar -x` corrupts the stream in PowerShell. Use
`git archive -o <tmp>.tar HEAD` then `tar -xf <tmp>.tar -C <scratch>`, overlay only your own staged
files, and run the suite there. A helper script with that shape lives at `%TEMP%\kilo\verify.ps1`.

---

## 8. Integration and sign-off checklist (TL; re-measured, not copied)

1. Full suite twice: green, < 60 s, every file < 10 s, identical across runs. Report
   `test_kv_cache`, `test_sampling`, `test_checkpoint`, `test_generate` timings explicitly.
2. **Regression proof:** confirm no Sprint 1–4 test file changed (`git diff --stat
   <sprint5-base>..HEAD -- tests/` must list only the four new files), and that
   `attention.py`/`block.py`/`model.py` diffs are **purely additive** (no edits inside `forward`,
   `backward`, `embed`, `forward_from_embedding`, `forward_loss`, `backward_loss`).
3. TL independent re-measurement: C8 max abs difference on tiny and default with the TL's own
   seed and its own full-forward reference; the shifted-mask divergence; the cached-vs-uncached
   FLOP counts; one checkpoint resume-equivalence check.
4. **TL-9 reconfirmation** on HEAD, protocol (a)–(d), every block.
5. Per-commit isolation gate for every Sprint 5 commit.
6. Static checks: no `TODO(SPRINT`; conventions guard green; chokepoint guard green; no torch/jax/
   tf/autograd; no `lm_head` key in `param_keys()` or in any checkpoint; `allow_pickle=False`
   everywhere a `.npz` is read; no sliding-window code.
7. No Phase 6+ files (`tests/test_numerical_stability.py` does not exist yet).
8. Corpus SHA-256 still `7afc055a…2854`, V = 55. `git status` clean.
9. Append the PROGRESS.md Phase 5 entry and commit `Sprint 5 integration sign-off`.

---

## 9. Explicitly out of scope for Sprint 5 (do not build)

- The Phase 6 large-logit stability pass and `tests/test_numerical_stability.py`.
- GPT-2 weight loading (Phase 7; recommended for descoping at the Phase 8 gate).
- The README rewrite and the reproducible test log (Phase 8).
- A sliding-window / rolling-position cache (TL-35 forbids it: it invalidates C8).
- A backward pass for `forward_step`, or any gradcheck of the cached path (TL-32).
- Batched prefill, perf work, LR schedules, loss averaging, validation splits.
- Any edit to a Sprint 1–4 test file, and any change to an existing public signature.
