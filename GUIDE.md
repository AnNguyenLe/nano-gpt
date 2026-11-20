# Beginner's Guide: Run and Talk to the NumPy nanoGPT

This guide assumes **you have never run a Python project before**. Follow it top to bottom and you
will end up with a small language model, trained by you, that you can talk to in your browser.

Everything here was checked on **Windows 11 + PowerShell** with **Python 3.15.2, NumPy 2.5.5, SciPy
1.18.1, pytest 9.2.0**. macOS/Linux instructions are in [section 12](#12-macos-and-linux) and are
**not tested** (the code is plain Python, so they should work, but I have not run them there).

---

## Table of contents

1. [What is this project?](#1-what-is-this-project)
2. [What to expect (read this once)](#2-what-to-expect-read-this-once)
3. [What you need](#3-what-you-need)
4. [Get the code and open a terminal](#4-get-the-code-and-open-a-terminal)
5. [One-time setup](#5-one-time-setup)
6. [Step 1: prove everything works (the tests)](#6-step-1-prove-everything-works-the-tests)
7. [Step 2: train your model](#7-step-2-train-your-model)
8. [Step 3: chat with it in your browser](#8-step-3-chat-with-it-in-your-browser)
9. [Other ways to use the model](#9-other-ways-to-use-the-model)
10. [Experiments to try](#10-experiments-to-try)
11. [Troubleshooting](#11-troubleshooting)
12. [macOS and Linux](#12-macos-and-linux)
13. [Tour of the source code](#13-tour-of-the-source-code)
14. [Glossary](#14-glossary)
15. [Quick reference card](#15-quick-reference-card)

---

## 1. What is this project?

It is a tiny version of the technology behind ChatGPT, built **from scratch** using only NumPy (a
Python library for fast number crunching). It is called a *GPT*: a neural network that reads some
text and predicts the next character, over and over.

What makes it unusual is that **nothing is hidden inside a framework**. PyTorch, TensorFlow and JAX
are not used. Every calculation, including the "backward pass" that lets the network learn, was
derived by hand and written out in plain code you can read. Then ~700 automated tests check the
maths against exact answers.

| | This project | ChatGPT |
|---|---|---|
| Parameters (the numbers it learns) | **107,712** | hundreds of billions |
| What it predicts | the next **character** | the next word-piece |
| Memory of the conversation | **64 characters** | tens of thousands of words |
| Trained on | a computer-generated corpus of simple sentences | a large part of the internet |
| Training time | about **14 minutes** on a laptop CPU | months on thousands of GPUs |
| Can answer questions | **No** | Yes |

## 2. What to expect (read this once)

This sets honest expectations so you are not disappointed.

- **It is not an assistant.** It does not answer questions or follow instructions. You give it the
  *start* of some text and it *continues* it. Think "autocomplete", not "chat".
- **It produces real-looking words, not meaningful sentences.** A real output from the trained model:

  ```
  prompt:  The rain
  output:  cloud! 

           Picture not, one from farmer wa
  ```

  The words come from its training sentences. The grammar and meaning do not hold together. That is
  normal for a model this small, and it is a deliberate trade-off: it is small enough to read,
  understand and train on a laptop.
- **It only knows 55 characters:** the letters `A-W`, `Y`, `a-y` (no `X`, `Z`, `z`), a space,
  newline, and `! , . ?`. There are **no digits**. Unknown characters are ignored (the page tells you).
- **It can only see 64 characters at a time**, prompt and reply together. If you type a long prompt,
  only its last part is used.
- **Everything runs on your own computer.** No internet is needed after installation, no account, no
  data leaves your machine.

## 3. What you need

- A computer running **Windows 10/11** (other systems: see [section 12](#12-macos-and-linux)).
- **About 300 MB** of free disk space (the project is 3 MB; the installed libraries are about 200 MB) and **a few GB of RAM** (training uses little).
- **Python 3.12 or newer** (this was built and tested on 3.15.2).
- A web browser (any modern one).
- **No GPU is needed.**

### Installing Python (if you do not have it)

1. Go to <https://www.python.org/downloads/> and download the latest Python 3 for Windows.
2. Run the installer. **Tick "Add python.exe to PATH"** on the first screen, then click *Install Now*.
3. Check it worked. Open PowerShell (see next section) and run:

   ```powershell
   py --version
   ```

   You should see something like `Python 3.15.2`. If `py` is not found, close and reopen PowerShell
   and try again; if it still fails, reinstall and make sure you ticked the PATH box.

## 4. Get the code and open a terminal

### The code

The project is the folder **`nanogpt_numpy`**. In this setup it lives at:

```
C:\Users\AnNLT\Downloads\MathematicsForComputing\nanogpt_numpy
```

To move it to another computer, copy the whole folder **except `.venv`** (the `.venv` folder is
tied to the computer it was made on; you will recreate it in step 5). A zip of the folder is
perfect. If you do not have Git, you do not need it; nothing below requires Git.

### Opening a terminal in that folder

A *terminal* is a window where you type commands. On Windows:

1. Open File Explorer and go into the `nanogpt_numpy` folder.
2. Click the address bar at the top, type `powershell`, and press **Enter**.
3. A blue PowerShell window opens already "inside" the folder. Check with:

   ```powershell
   dir
   ```

   You should see `README.md`, `train.py`, `serve.py`, a `nanogpt` folder and a `tests` folder.

> **Every command in this guide is run from inside `nanogpt_numpy`.** If a command says a file
> "cannot be found", you are almost certainly in the wrong folder.

## 5. One-time setup

You do this once per computer. It creates a private Python environment for the project (a "virtual
environment", the `.venv` folder) and installs the three libraries it needs.

```powershell
# 1. Create the private environment
py -3 -m venv .venv

# 2. Install the three libraries (numpy, scipy, pytest) into it
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The second command downloads and installs about 200 MB of libraries and takes a minute or two. It is the **only** step that
needs the internet.

Check the install:

```powershell
.venv\Scripts\python.exe -c "import numpy, scipy, pytest; print('all good', numpy.__version__)"
```

You should see `all good` and a version number.

### A note about "activating" (you can skip it)

Many tutorials tell you to "activate" the environment with `.venv\Scripts\Activate.ps1`. **This guide
never does**, because on many Windows PCs PowerShell blocks that script with an error about
"running scripts is disabled". Instead we always write the full path
`.venv\Scripts\python.exe`, which always works. It is just a longer way to say "use the Python
that has our libraries installed".

## 6. Step 1: prove everything works (the tests)

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

This takes **about 25 seconds** and ends with something like:

```
715 passed in 25.0s
```

What just happened? `pytest` ran hundreds of small checks. For example: "does the hand-written gradient
match a numerical estimate of the gradient to 6 decimal places?", "does the parameter count equal the
exact formula?", "does generating with the cache give the same numbers as the slow way to 10
decimal places?". If all pass, the maths in the project is behaving. You do not need to understand
them yet; it is a **smoke alarm that is currently silent**.

Want a saved, shareable record? Run:

```powershell
.venv\Scripts\python.exe run_checks.py
```

It writes **`test_log.txt`** containing the Python/NumPy versions, the corpus fingerprint and the
full result.

## 7. Step 2: train your model

Right now the model has random numbers inside and produces noise. *Training* means showing it lots of
text so it gradually adjusts its 107,712 numbers to predict the next character better.

```powershell
.venv\Scripts\python.exe -u train.py --save model.ckpt
```

(The `-u` makes the progress lines appear live instead of all at the end.)

### What you will see

```
corpus: source=synthetic sha256=7afc055a chars=1000000 V=55
config: ModelConfig(V=55, T=64, d=64, L=2, H=4) N=107712 | batch=16 lr=0.001 wd=0.1 steps=4000 seed=1337
step 0 | loss 4.0333 | 159.7 ms/step | elapsed 0.2s
step 100 | loss 2.5241 | 282.6 ms/step | elapsed 19.4s
step 500 | loss 1.8901 | ...
step 1000 | loss 1.1971 | ...
...
step 3999 | loss 1.1376 | 224.1 ms/step | elapsed 841.1s
saved checkpoint to model.ckpt
```

How to read it:

- **`corpus`**: the training text. If you have no `data/corpus.txt`, the program *generates* one
  (1,000,000 characters of simple English-like sentences) from a fixed random seed, so everyone gets
  the same text. `V=55` is the number of different characters in it.
- **`N=107712`**: how many numbers the model learns.
- **`loss`**: how wrong the model's predictions are. **Lower is better.** At step 0 it is about
  `4.0`, which is exactly "guessing randomly among 55 characters" (`ln 55 = 4.007`). Then it drops
  quickly and levels off around `1.1`. It levels off because the training text itself has
  randomness the model can never predict, not because training broke.
- **`ms/step`**: how long one learning step takes.

### How long does it take?

About **14 minutes** on the machine this was built on (841 seconds; yours will differ with your CPU).
It is fine to leave it running. It prints a line every 100 steps.

> **In a hurry?** Do a quick, lower-quality version first (about 1 minute):
> ```powershell
> .venv\Scripts\python.exe -u train.py --steps 300 --save model.ckpt
> ```
> The output will be mostly gibberish with a few real word fragments. Run the full command later to
> overwrite it.

When it finishes you will have a file **`model.ckpt`** (about 2.6 MB). That file *is* your trained
model: all 107,712 numbers plus the optimizer's memory. Training is **repeatable**: with the same
settings, you get exactly the same final loss (`1.1376`) every time.

## 8. Step 3: chat with it in your browser

```powershell
.venv\Scripts\python.exe serve.py
```

You will see:

```
model: model.ckpt | L=2 d=64 H=4 T=64 V=55
open http://127.0.0.1:8000   (Ctrl+C to stop)
```

Now **open <http://127.0.0.1:8000> in your browser**. (`127.0.0.1` means "this computer". Nobody else
on your network can reach it.)

### Using the page

1. Click the box at the bottom and type **the start of a sentence**, for example `The rain `.
   (Keep the trailing space; it gives the model a natural place to continue.)
2. Press **Enter** (or click **Continue**). The reply streams in, one character at a time.
   Your text appears in grey, the model's continuation in white.
3. Click **Stop** to interrupt it. Use **Shift+Enter** for a new line in your message.

Things worth trying as a first message: `The rain `, `Remember `, `Good, from `, `Water animal `.

### The settings on the left

| Setting | What it does | Try |
|---|---|---|
| **Max new characters** | How many characters to generate. Your prompt is shortened so that prompt + this never exceeds 64. | 20 to 60 |
| **Temperature** | Randomness. **Low (0.1-0.5)** = safe, repetitive. **1.0** = as the model really thinks. **High (1.5+)** = chaotic. | 0.3, 0.8, 1.5 |
| **Top-k** | Only choose among the *k* most likely characters. **off** = no limit. Small values (3-8) cut out silly choices. | off, 8, 3 |
| **Greedy** | Always take the single most likely character. Deterministic, but tends to loop (`their their then their...`). | on, then off |
| **Seed** | The dice. Same seed + same settings + same prompt = **identical output**. Blank = a random one. | 7, then 7 again |

Under every reply you see the **seed it used**. Paste that seed into the Seed box to reproduce that
exact reply.

### What the grey notes under a reply mean

- `your prompt was cut by N leading character(s)...` : your prompt was too long to fit in the
  64-character window with the reply, so the **beginning** was dropped.
- `ignored unknown characters: "123"` : you typed characters the model has never seen (digits,
  `z`, etc.). They were removed before the model saw your text.

### Stopping the server

Click in the PowerShell window and press **Ctrl+C**. To start it again later, run
`.venv\Scripts\python.exe serve.py` again. You do **not** need to retrain; `model.ckpt` is saved.

> If you start the server before training, it still runs but shows a **yellow warning banner** and
> the output is random noise, because there are no learned numbers yet. Train first.

## 9. Other ways to use the model

### From the command line (no browser)

```powershell
.venv\Scripts\python.exe generate.py --checkpoint model.ckpt --prompt "The rain " --tokens 40 --top-k 8 --temperature 0.8
```

| Option | Meaning |
|---|---|
| `--checkpoint PATH` | which trained model to load |
| `--random-init` | use an untrained model instead (instead of `--checkpoint`) |
| `--prompt "text"` | the text to continue (default: the corpus's first character) |
| `--tokens N` | how many characters to generate (prompt + N must be at most 64) |
| `--temperature X` | randomness (default 1.0) |
| `--top-k N` | only sample from the N most likely characters |
| `--greedy` | always pick the most likely character |
| `--seed N` | repeatability |

### Training options

```powershell
.venv\Scripts\python.exe -u train.py --steps 1000 --batch-size 16 --lr 0.001 --save short.ckpt
```

| Option | Meaning |
|---|---|
| `--steps N` | how many learning steps (default 4000) |
| `--batch-size N` | how many text snippets per step (default 16) |
| `--lr X` | learning rate: how big each adjustment is (default 0.001) |
| `--L N` / `--d N` / `--T N` / `--H N` | model size: layers, width, context length, attention heads |
| `--log-every N` | print a progress line every N steps |
| `--save PATH` | write the trained model to this file |
| `--profile N` | measure where time goes during the first N steps |
| `--allow-out-of-range` | allow sizes outside L 2-4, d 64-128, T 64-128 (the project's spec) |

To serve a differently named checkpoint or port:

```powershell
.venv\Scripts\python.exe serve.py --checkpoint short.ckpt --port 8001
```

### Use it from your own Python code

```python
import numpy as np
from nanogpt import checkpoint
from nanogpt.model import GPT
from nanogpt.kv_cache import KVCache
from nanogpt.tokenizer import load_corpus_and_tokenizer
from nanogpt.config import make_rng
import generate

_, tok = load_corpus_and_tokenizer()
cfg, params, _ = checkpoint.load("model.ckpt")
model = GPT(cfg)

ids = generate.generate(model, params, KVCache(cfg, 1), tok.encode("The rain "), 40, make_rng(7),
                        temperature=0.8, top_k=8)
print("The rain " + tok.decode(ids))
```

Save that as `try_it.py` in the `nanogpt_numpy` folder and run
`.venv\Scripts\python.exe try_it.py`.

## 10. Experiments to try

These are the best way to *feel* what the settings do.

1. **Same seed, same output.** Send `The rain ` with seed `7` twice. Identical replies.
2. **Temperature sweep.** Same prompt and seed at temperature 0.1, 0.8, 2.0. Watch it go from
   repetitive to normal to chaotic.
3. **Greedy loops.** Turn on Greedy. The model gets stuck repeating a likely word. This is a famous
   weakness of "always pick the best".
4. **Watch it learn.** Train for `--steps 50`, `--steps 300`, then `--steps 4000`, saving to
   different files, and generate from each with the same prompt and seed. Garbage, then word
   fragments, then real words.
5. **Bigger model.** `train.py --d 128 --L 4 --steps 1000 --save big.ckpt`. It is several times
   slower per step. Does it reach a lower loss? (Compare at the same number of steps.)
6. **Train on your own text.** Create a plain-text file `data\corpus.txt` (under 5 MB, UTF-8) and run
   `train.py --save mine.ckpt`. The program automatically uses it, and the vocabulary becomes
   whatever characters your file contains. Then run `serve.py --checkpoint mine.ckpt`.

   Three things to know about experiment 6, which I tested:
   - **Old checkpoints stop loading.** A model trained on the default corpus has 55 characters; yours
     will have a different number. `serve.py` stops with
     `checkpoint vocab 55 does not match the corpus tokenizer 30`. You must retrain.
   - **About 17 of the tests will fail**, because they deliberately pin the default corpus. That is
     expected and does not mean your model is broken. **Delete `data\corpus.txt` to go back** to the
     default corpus and passing tests.
   - Small text files train to a low loss quickly because the model can memorise them. That is not
     the same as learning.

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `'py' is not recognized` | Python is not on PATH. Reinstall Python and tick **Add python.exe to PATH**, then reopen PowerShell. |
| `can't open file ... train.py` / `No such file` | Your terminal is in the wrong folder. `cd` into `nanogpt_numpy`, or reopen PowerShell from that folder's address bar. |
| `.venv\Scripts\python.exe` is not recognised | You skipped step 5, or you copied the folder from another PC. Run the two commands in [step 5](#5-one-time-setup). |
| `ModuleNotFoundError: No module named 'numpy'` | You ran plain `python` instead of `.venv\Scripts\python.exe`. Use the full path. |
| `running scripts is disabled on this system` | You tried `Activate.ps1`. You do not need it. Use the full-path commands. |
| `pip install` fails with "no matching distribution" | Your Python is very new or very old for the available NumPy/SciPy builds. Install Python 3.12 or 3.13 from python.org and rebuild `.venv` (delete the `.venv` folder, redo step 5 with `py -3.13 -m venv .venv`). |
| Browser says "This site can't be reached" | `serve.py` is not running, or you are on a different port. Check the PowerShell window says `open http://127.0.0.1:8000`. |
| `OSError: [WinError 10048]` / address already in use | Something is on port 8000 (maybe a previous `serve.py`). Use `serve.py --port 8001` and open `:8001`. |
| Yellow banner: "No trained checkpoint found" | `model.ckpt` is missing. Run step 7, then restart `serve.py`. |
| The replies are nonsense | Check the model is trained (no banner) and that you did not stop training very early. Also remember the limits in [section 2](#2-what-to-expect-read-this-once). |
| `checkpoint vocab 55 does not match the corpus tokenizer N` | You added `data\corpus.txt` after training. Retrain, or delete that file. |
| `error: len(prompt)=... + --tokens=... exceeds context length` | Prompt plus requested characters is more than 64. Shorten one of them. |
| Training seems frozen | Progress prints every 100 steps (roughly every 20 seconds). Be patient, and use `-u`. |
| Tests fail after you edited something | Undo your edit. The tests are strict on purpose. |
| Everything is slow | Close other heavy programs. The model uses your CPU only; it is not optimised for speed. |

## 12. macOS and Linux

> **Untested.** The code is pure Python and uses portable file paths, so this should work, but I
> only ran it on Windows. If a step fails, the Windows steps above show the intent.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -u train.py --save model.ckpt
.venv/bin/python serve.py          # then open http://127.0.0.1:8000
```

Everywhere this guide writes `.venv\Scripts\python.exe`, use `.venv/bin/python`. Use `/` instead of
`\` in paths (`data/corpus.txt`).

## 13. Tour of the source code

Total: about **3,100 lines** of Python in the library and scripts, plus about **7,200** in the tests. Open any file
in a text editor (Notepad works, [VS Code](https://code.visualstudio.com/) is nicer). Every file
starts with a comment explaining what it does and why.

### The big picture

```
nanogpt_numpy/
├── README.md            short project summary
├── GUIDE.md             this guide
├── PROGRESS.md          the build log: what was built, what passed, honest caveats
├── requirements.txt     the 3 libraries: numpy, scipy, pytest
│
├── train.py             run this to TRAIN the model          (220 lines)
├── generate.py          run this to generate text (terminal) (113 lines)
├── serve.py             run this for the browser chat UI     (238 lines)
├── run_checks.py        run all tests and write test_log.txt  (41 lines)
├── web/index.html       the chat page itself (HTML+CSS+JS)   (189 lines)
├── data/                put your own corpus.txt here (optional)
│
├── nanogpt/             THE LIBRARY: the model and all its maths
│   ├── config.py          the fixed rules (sizes, epsilons, seeds) defined ONCE
│   ├── tokenizer.py       turns text into numbers and back; makes the training text
│   ├── ops.py             building blocks: linear layer, GELU, softmax, matmul
│   ├── layernorm.py       LayerNorm layer
│   ├── attention.py       multi-head self-attention (the key idea of a transformer)
│   ├── attention_mask.py  the "no peeking at the future" rule
│   ├── attention_reference.py  a slow, obvious version used to double-check attention
│   ├── mlp.py             the feed-forward layer
│   ├── block.py           one transformer block = LayerNorm + attention + LayerNorm + MLP
│   ├── model.py           the full GPT: stacks blocks, embeddings, output layer
│   ├── loss.py            cross-entropy: "how wrong was the prediction?"
│   ├── optim.py           AdamW: "how should the numbers change to be less wrong?"
│   ├── kv_cache.py        memory that makes generation fast
│   ├── sampling.py        temperature / top-k / greedy: picking the next character
│   ├── checkpoint.py      save and load a model
│   ├── param_count.py     counts the model's numbers (and checks a formula)
│   ├── flops.py           counts how much arithmetic the model does
│   └── gradcheck.py       the tool that checks hand-written gradients
│
└── tests/               ~700 automated checks (one file per topic)
```

### How a GPT works, in five lines

1. **Text becomes numbers.** Each character gets an ID (`tokenizer.py`).
2. **Each ID becomes a list of 64 numbers** (an *embedding*), plus a list that says which position it
   is in (`model.py`, `ops.Embedding`).
3. **Blocks mix the information.** Each *block* lets every position look at the earlier positions
   (*attention*) and then think about what it gathered (*MLP*) (`block.py`).
4. **The output layer scores every possible next character** (55 scores per position).
5. **Softmax turns scores into probabilities** and one character is picked (`sampling.py`).

### The code that matters, with real excerpts

**`nanogpt/ops.py`: every multiplication in the model goes through one function**

```python
def matmul(a, b, *, kind="dense"):
    """Single chokepoint for every production matmul. kind in {"dense", "attention"}.
    Returns exactly ``a @ b`` ... ``kind`` is purely a diagnostic tag for FlopCounter."""
    if kind not in ("dense", "attention"):
        raise ValueError(...)
    return a @ b
```

Why bother? So `flops.py` can wrap this one function and count *all* the arithmetic, with nothing
able to sneak around it. A test enforces that nothing else multiplies matrices.

**`nanogpt/attention.py`: the heart of the transformer** (the forward pass, slightly shortened)

```python
q, cq = self.q.forward(x, params)      # "what am I looking for?"
k, ck = self.k.forward(x, params)      # "what do I contain?"
v, cv = self.v.forward(x, params)      # "what do I hand over if chosen?"
qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)   # 4 heads, 16 numbers each

S = ops.matmul(qh, kh.swapaxes(-1, -2), kind="attention") * self.scale   # similarity scores
allowed = self.mask_fn(T)                      # position i may only look at j <= i
Z = np.where(allowed, S, -np.inf)              # forbid looking at the future
P, _ = ops.softmax_forward(Z)                  # scores -> probabilities that sum to 1
Y = merge_heads(ops.matmul(P, vh, kind="attention"))   # weighted mix of the values
out, co = self.o.forward(Y, params)
```

**`nanogpt/kv_cache.py`: why generation is fast.** Without it, writing each new character means
re-reading the entire text from scratch. With it, the model remembers what it already worked out:

```python
def append(self, layer, kh_step, vh_step):
    ...
    pos = self.length
    self.K[layer, :, :, pos : pos + 1, :] = kh_step   # store this character's "key"
    self.V[layer, :, :, pos : pos + 1, :] = vh_step   # and "value"
    self._next_layer += 1
    return self.K[layer, :, :, : pos + 1, :], self.V[layer, :, :, : pos + 1, :]
```

Measured result: generating 32 characters is **48.7x less arithmetic** with the cache, and the
output matches the slow way to within 0.0000000001 (it measured closer than that: 0.0000000000003 at worst).

**`nanogpt/sampling.py`: choosing the next character**

```python
def apply_top_k(logits, k):
    order = np.argsort(-logits, axis=-1, kind="stable")   # best first
    keep = np.zeros(logits.shape, dtype=bool)
    np.put_along_axis(keep, order[..., :k], True, axis=-1)
    return np.where(keep, logits, -np.inf)                # everything else is forbidden
```

**`generate.py`: the generation loop** (the whole idea in a few lines)

```python
logits = model.prefill(prompt, params, kv)[:, -1, :]      # read the prompt once
for i in range(n_tokens):
    nxt = sampling.sample(logits, rng, temperature=temperature, top_k=top_k)   # pick a character
    out[i] = nxt[0]
    logits = model.forward_step(nxt[:, None], params, kv)[:, -1, :]            # feed it back in
```

**`nanogpt/optim.py`: how it learns.** AdamW nudges every number a little in the direction that
reduces the loss. The rule it implements (from the project spec):

```
m = b1*m + (1-b1)*g          # smoothed gradient
v = b2*v + (1-b2)*g*g        # smoothed squared gradient
m_hat = m / (1 - b1^t)       # correct the early-steps bias
v_hat = v / (1 - b2^t)
theta = theta - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * theta )
```

It is tested by comparing against a separate hand-calculation to **15 decimal places**.

### The training loop in one picture

```
repeat 4000 times:
    pick 16 random 64-character snippets of the corpus          (train.get_batch)
    predict the next character at every position                (model.forward_loss)
    measure how wrong that was                                  (loss.cross_entropy)
    work out how each of the 107,712 numbers affected the error (model.backward_loss)  <- hand-derived
    nudge every number slightly                                 (optim.AdamW.step)
```

### Why so many tests?

The famous trap in machine learning is a model whose loss goes down while the maths inside is
**subtly wrong**. So this project never accepts "the loss went down" as proof. Instead every piece is
checked against something exact:

- **Gradients** are compared with a numerical estimate (central differences), to 1 part in a million.
- **The parameter count** must match a formula exactly (and it reproduces GPT-2-small's 124,439,808).
- **The "no peeking at the future" rule** is tested by changing a later character and confirming
  earlier outputs do not change at all.
- **The cache** must give the same numbers as the slow way, to 10 decimal places.
- **Each test is itself tested**: a deliberately broken version of the code must make the test fail.

Test files are named after what they check (`test_kv_cache.py`, `test_adamw.py`, ...). The mapping
from each requirement (C1-C9) to its test file is in `README.md`.

## 14. Glossary

| Term | Plain meaning |
|---|---|
| **Language model** | A program that predicts what comes next in a piece of text. |
| **GPT** | "Generative Pre-trained Transformer": the design used here (and by ChatGPT). |
| **Character-level** | It reads and writes one character at a time, not whole words. |
| **Parameters / weights** | The numbers the model learns. This model has 107,712. |
| **Training** | Repeatedly adjusting the parameters so predictions improve. |
| **Loss** | A single number for "how wrong". Lower is better. Starts near 4.0, ends near 1.1 here. |
| **Gradient** | For each parameter: which direction and how much to change it to reduce the loss. |
| **Backward pass / backpropagation** | The calculation of all the gradients. Hand-written in this project. |
| **AdamW** | The rule for turning gradients into parameter changes. |
| **Learning rate** | How big each nudge is. |
| **Embedding** | Replacing a character ID by a list of numbers the model can work with. |
| **Attention** | The mechanism by which each position gathers information from earlier ones. |
| **Context window** | How much text the model can see at once (64 characters here). |
| **Logits** | The raw scores for each possible next character, before they become probabilities. |
| **Softmax** | Turns scores into probabilities that add up to 1. |
| **Temperature** | Divides the scores before softmax: low = confident, high = random. |
| **Top-k** | Only allow the k highest-scoring characters. |
| **Greedy** | Always choose the single highest-scoring character. |
| **Seed** | The starting point of the random number generator; same seed, same "random" choices. |
| **Checkpoint (`.ckpt`)** | A saved model: all parameters (and the optimizer's memory). |
| **KV-cache** | Stored intermediate results that avoid recomputing the past while generating. |
| **Virtual environment (`.venv`)** | A private folder of Python libraries for one project. |
| **Terminal / PowerShell** | The text window where you type commands. |
| **pytest** | The tool that runs the automated checks. |
| **FLOPs** | A count of arithmetic operations: a measure of how much work something is. |

## 15. Quick reference card

Run everything from inside `nanogpt_numpy`.

```powershell
# --- ONE TIME ---
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# --- CHECK IT WORKS (~25 s) ---
.venv\Scripts\python.exe -m pytest tests/ -q

# --- TRAIN (~14 min; or add --steps 300 for a fast rough one) ---
.venv\Scripts\python.exe -u train.py --save model.ckpt

# --- CHAT IN THE BROWSER ---
.venv\Scripts\python.exe serve.py            # then open http://127.0.0.1:8000   (Ctrl+C stops it)

# --- OR GENERATE IN THE TERMINAL ---
.venv\Scripts\python.exe generate.py --checkpoint model.ckpt --prompt "The rain " --tokens 40 --top-k 8
```

Good first prompts: `The rain `, `Remember `, `Good, from `, `Water animal `.

**If something goes wrong:** read the error's last line, find it in
[Troubleshooting](#11-troubleshooting), and check you are in the `nanogpt_numpy` folder.

For the complete engineering record (what was verified, every deviation, and what is still open),
see `PROGRESS.md`.
