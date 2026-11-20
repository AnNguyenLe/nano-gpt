"""nanogpt.optim -- AdamW optimizer (Phase 4, Engineer A).

See SPRINT_4_PLAN.md section 2.2 (TL-24/25/26) and section 3.2 for the
derivation this module implements exactly.

TL-24 (decay-group ruling, additive, for PO ratification): group membership
is decided by an EXHAUSTIVE, EXPLICIT regex allowlist over `GPT.param_keys()`
/ `named_parameters()` names -- never inferred from `ndim` (LayerNorm gamma
is a 1-D "weight"; `wpe.W` is a 2-D lookup table that is nonetheless
no-decay). Any name outside both lists is a bug, not a default, and raises
ValueError. The alias "lm_head.W" is never classified by `is_decay_param` on
its own: it is removed first by `canonicalize_params` (TL-25), which maps it
onto the same canonical name as "wte.W".

TL-25 (one state per unique array, canonical names): `AdamW(params)` accepts
`params` built from `GPT.init_params()` (only "wte.W") or from
`GPT.named_parameters()` ("wte.W" plus "lm_head.W" aliasing the SAME object).
`canonicalize_params` dedupes by object identity (`id()`), not by value
equality or shape, so the first key that reaches a given array object (in
dict-iteration order) becomes its canonical name and every later alias of
that SAME object collapses into it silently. Two DISTINCT objects that
nonetheless `np.shares_memory` without being identical (e.g. one is `.T`, or
a view/slice, of the other) are a forbidden partial-overlap tie and raise
ValueError -- the same contract as `nanogpt.param_count.live_param_count`.
Non-float64 and non-C-contiguous arrays also raise.

TL-26 (step counter): a single per-optimizer integer `self.t`, starting at 0
and incremented ONCE at the very start of `step()` (after validation, before
the per-tensor loop), so the first real update uses t = 1. Bias corrections
use `1 - beta**t` in float64. The decay term always uses theta_{t-1} (the
value before this step's own update), which is automatic since we compute
`u` from the pre-update `theta` and only then subtract `lr * u` from it.

No `zero_grad` (every backward call starts from a fresh grads dict). No
`state_dict` / `load_state_dict` (R22: checkpointing is out of scope for
Sprint 4).

Owner: Engineer A. Files: nanogpt/optim.py, tests/test_adamw.py.
"""
from __future__ import annotations

import re
from typing import Mapping

import numpy as np

from nanogpt.config import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_WEIGHT_DECAY,
    DTYPE,
    check_float64,
)

# ---------------------------------------------------------------------------
# TL-24: decay-group allowlist (exhaustive; membership is never ndim-based)
# ---------------------------------------------------------------------------
DECAY_PATTERNS: tuple = (
    re.compile(r"^h\.\d+\.attn\.(q|k|v|o)\.W$"),
    re.compile(r"^h\.\d+\.mlp\.(fc|proj)\.W$"),
)
NO_DECAY_PATTERNS: tuple = (
    re.compile(r"^(wte|wpe)\.W$"),
    re.compile(r"\.b$"),  # every Q/K/V/O/fc/proj bias
    re.compile(r"\.(gamma|beta)$"),  # every LayerNorm (ln1, ln2, ln_f)
)


def is_decay_param(name: str) -> bool:
    """True if `name` is in the TL-24 decay group, False if no-decay.

    Raises ValueError for any name matching neither allowlist, including
    the tied-head alias "lm_head.W" (never classified on its own --
    `canonicalize_params` removes it first) and any near-miss such as
    "h.0.attn.q.Wx".
    """
    if any(p.search(name) for p in DECAY_PATTERNS):
        return True
    if any(p.search(name) for p in NO_DECAY_PATTERNS):
        return False
    raise ValueError(
        f"is_decay_param: {name!r} matches neither the decay nor the "
        "no-decay allowlist (TL-24) -- this is a bug, not a default"
    )


def canonicalize_params(params: Mapping[str, np.ndarray]) -> dict:
    """Dedupe `params` by object identity (TL-25).

    Two dict entries whose VALUES are the exact same ndarray object collapse
    into one entry, named after whichever key reaches that object first in
    `params`' iteration order (so `named_parameters()`'s "wte.W" wins over
    its later "lm_head.W" alias, since that dict lists "wte.W" first).

    Raises ValueError if:
      - a value is not C-contiguous;
      - a value, while NOT identical (by `id`) to an already-seen value,
        nonetheless `np.shares_memory` with it (a partial-overlap view such
        as `.T` or a slice -- forbidden; tie only by object identity).
    Raises TypeError (via `check_float64`) if a value is not float64.
    """
    seen_id_to_name: dict = {}  # id(arr) -> canonical name
    seen_arrays: list = []  # [(name, arr), ...] insertion order, for shares_memory scan
    out: dict = {}

    for key, arr in params.items():
        arr = np.asarray(arr)
        check_float64(arr, names=[key])
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"canonicalize_params: params[{key!r}] is not C-contiguous"
            )

        aid = id(arr)
        if aid in seen_id_to_name:
            continue  # same object, already canonicalized under an earlier key

        for other_name, other_arr in seen_arrays:
            if np.shares_memory(arr, other_arr):
                raise ValueError(
                    f"canonicalize_params: params[{key!r}] shares memory "
                    f"with params[{other_name!r}] without being the SAME "
                    "object (a partial-overlap view, e.g. .T or a slice, "
                    "is forbidden; tie parameters by identity only)"
                )

        seen_id_to_name[aid] = key
        seen_arrays.append((key, arr))
        out[key] = arr

    return out
