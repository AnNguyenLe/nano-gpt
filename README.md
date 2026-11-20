# nanoGPT in NumPy

A character-level GPT built end to end in **raw NumPy** (plus SciPy for a few dense linear-algebra
checks): every forward and backward pass is derived by hand and verified, with AdamW, a KV-cache,
sampling and checkpointing. No PyTorch, JAX, TensorFlow or autograd anywhere.

The correctness story is the point. A decreasing loss is **never** used as evidence that anything
works: every component is checked against a closed-form value, an exact invariant, a reference
oracle, or a central-difference gradient check (float64, relative error < 1e-6), and each check has
a deliberately broken variant proving it can actually fail.

- `PROGRESS.md` is the build log (what passed, deviations, what is next). Read it first if resuming.
- `nanoGPT_build_requirements.md` and `nanoGPT_execution_plan.md` (one directory up) are the spec
  and the plan; `SPRINT_*_PLAN.md` are the per-sprint technical plans.

## Setup

```
py -3.15 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` lists only `numpy`, `scipy` and `pytest`. If wheels are not available for your
Python's exact minor version, build the venv from an older interpreter (3.12 or 3.13 work).
No network access is needed at any point: the corpus is either `data/corpus.txt` (if you add one,
under 5 MB) or a seeded synthetic corpus generated in memory (`synthetic-v1`, seed 1337, 1,000,000
characters, vocabulary V = 55, SHA-256 prefix `7afc055a`). Nothing is ever downloaded.

A root `conftest.py` puts the repo root on `sys.path`, so `import nanogpt`, `import train` and
`import generate` work without installing the package.

## Verify (the one command)

```
.venv\Scripts\python.exe -m pytest tests/ -q
```

This runs every correctness check below (about 680 tests, roughly 25 seconds). To capture a
reproducible log with the environment recorded:

```
.venv\Scripts\python.exe run_checks.py
```

which writes `test_log.txt` (Python/NumPy/SciPy versions, corpus SHA-256, the full `pytest` output
and the exit status).

## Train

```
.venv\Scripts\python.exe -u train.py --save model.ckpt
```

Defaults: `L=2, d=64, T=64, H=4` (107,712 parameters), batch 16, constant LR 1e-3, AdamW with
weight decay 0.1, 4000 steps, seed 1337. **Measured: 841 s (14.0 min, 0.21 s/step) on one CPU,
against the 30-minute budget.** Use `-u` so progress is not buffered when the output is piped.

Useful flags: `--steps N`, `--batch-size N`, `--lr X`, `--L/--d/--T/--H` (the spec ranges are
L 2-4, d 64-128, T 64-128; anything else needs `--allow-out-of-range`), `--profile N` (cProfile the
first N steps), `--save PATH` (write params + AdamW state). The step-0 loss is printed next to
ln V as a consistency check only.

## Chat UI (browser)

```
.venv\Scripts\python.exe -u train.py --save model.ckpt     # once, ~14 min
.venv\Scripts\python.exe serve.py                          # then open http://127.0.0.1:8000
```

`serve.py` is a small local web server (standard library only, no new dependencies, bound to
127.0.0.1) with a ChatGPT-style page: streaming output, sliders for max new characters,
temperature and top-k, a greedy switch, and a seed box (the seed used is shown under each reply so a
run can be reproduced). It loads `model.ckpt` if it exists; otherwise it serves untrained weights and
shows a warning banner. Options: `--checkpoint PATH --port N`.

Set expectations: this is **not an assistant**. It is a tiny character-level model that *continues
your text*; it will not answer questions or follow instructions. Its context is 64 characters, so
your prompt is cut to the last `64 - max_new` characters (the UI tells you when that happens), and it
only knows the 55 characters of its training corpus (unknown characters are ignored and listed).

## Generate (command line)

```
.venv\Scripts\python.exe generate.py --checkpoint model.ckpt --prompt "the " --tokens 100 --top-k 8 --temperature 0.8
.venv\Scripts\python.exe generate.py --random-init --tokens 20      # untrained weights
```

Generation uses the KV-cache and the sampling module (temperature + top-k, or `--greedy`), is
deterministic for a fixed `--seed`, and refuses `len(prompt) + tokens > T`: there is deliberately
no sliding window, because it would shift absolute positions and break the C8 equivalence.
Generated text is for reading; **no test treats it as evidence of correctness.**

## Which test proves which requirement

| # | Requirement | Where it is proved |
|---|---|---|
| C1 | Parameter count `N = Vd + Td + L(12d^2 + 13d) + 2d`, tied matrix counted once; GPT-2-small gives 124,439,808 | `tests/test_param_count.py` |
| C2 | Loss at init within 0.1 of ln V | `tests/test_loss_at_init.py` |
| C3 | LayerNorm Jacobian rank d-2 at eps=0, d-1 at eps>0 (SVD) | `tests/test_layernorm_rank.py` |
| C4 | Softmax Jacobian PSD, rows sum to 0, shift invariance | `tests/test_softmax_properties.py` |
| C5 | Causal mask: no leakage, on the isolated attention module and on the full model; shifted-mask adversary must fail | `tests/test_causal_mask.py`, `tests/test_attention.py`, `tests/test_model.py` |
| C6 | Full-model gradient check, tiny config, every tensor, rel. error < 1e-6 | `tests/test_gradients.py` |
| C7 | Tied-embedding gradient = input side + output side, accumulated with `+=` into one array | `tests/test_weight_tying.py` |
| C8 | KV-cache logits equal full-forward logits to 1e-10 absolute, teacher-forced, length <= T | `tests/test_kv_cache.py` |
| C9 | FLOPs/token: instrumented matmul counter vs `2N + 4LTd` within +-25%, and equal to an exact integer inventory | `tests/test_flops.py` |

Other guarantees and where they live:

| Topic | File |
|---|---|
| AdamW against a closed-form first step and a hand-computed 3-step trajectory, decay groups, single tied optimizer state | `tests/test_adamw.py` |
| Large logits (1e3 to 1e5) through the full model, loss, gradients, sampling and cached decoding | `tests/test_numerical_stability.py` |
| T=1 degenerate attention, vectorized vs naive loop-over-heads oracle | `tests/test_attention.py`, `tests/test_attention_reference.py` |
| Frozen conventions (dtype, eps, init std, FD step, rel-err) defined once in `nanogpt/config.py`; no stray literals, no float32, no framework imports | `tests/test_conventions.py` |
| Single matmul chokepoint (`ops.matmul`) so the FLOP counter cannot miss a product | `tests/test_flops.py` |
| Sampling, checkpoint round trip (bit-identical resume), generation | `tests/test_sampling.py`, `tests/test_checkpoint.py`, `tests/test_generate.py` |
| Chat UI server: request validation, streaming equals `generate()`, real HTTP round trip | `tests/test_serve.py` |
| Training loop determinism, tie preservation, no loss-decrease assertion anywhere | `tests/test_train.py` |

## Layout

```
nanogpt/   config  tokenizer  gradcheck  ops  layernorm  attention  attention_reference  attention_mask
           mlp  block  model  loss  optim  flops  param_count  kv_cache  sampling  checkpoint
train.py   generate.py   serve.py   run_checks.py   conftest.py   web/   data/   tests/
```

Every differentiable module follows `forward(x, params) -> (out, cache)` and
`backward(d_out, cache) -> (d_x, d_params)`, with gradients keyed by global parameter name and
accumulated with `+=`, which is what makes the weight-tying check hold by construction.

## Honest caveats

- `attn.k.b` (the key bias) has an identically zero gradient at every T: it shifts every score in a
  softmax row by a constant, and softmax is shift-invariant. This is expected, not a bug, and is
  pinned by a four-part protocol in the tests.
- The +-25% FLOP band alone does not discriminate the naive `2N` formula at the default config; the
  integer-exact matmul inventory and the attention-heavy configs do. The tests say so explicitly.
- A KV-cache that ignores its mask row is indistinguishable from a correct one under the causal
  mask, so C8 cannot detect that bug today.
- The decision to exclude `wpe.W` from weight decay (matching minGPT, unlike nanoGPT) is a recorded
  reading of "weight matrices" and has not been independently ratified.
- The optional GPT-2-small weight-loading bonus (Phase 7) is **not implemented**: it is the only
  networked phase and GPT-2 small (T=1024, d=768, L=12) is far outside this project's own size
  ranges. C1 does verify the parameter formula against GPT-2 small analytically.
