"""nanogpt.checkpoint -- save / load model parameters and AdamW state.

SPRINT_5_PLAN.md TL-38 and section 3.4.

Format: one uncompressed ``.npz`` (``np.savez``) with a flat key scheme and
``allow_pickle=False`` on load, so a checkpoint can never execute code:

    format                      "nanogpt-numpy-ckpt-v1"
    cfg_V cfg_T cfg_d cfg_L cfg_H
    param/<key>                 every ``GPT.param_keys()`` entry, and ONLY those
    opt_t opt_lr opt_beta1 opt_beta2 opt_eps opt_weight_decay     (optional)
    opt_m/<key> opt_v/<key>                                       (optional)

The weight tie is restored by construction: ``params`` has exactly
``param_keys()`` (no ``lm_head.W``), ``wte.W`` is one contiguous array, and
``GPT.named_parameters`` re-aliases it. A checkpoint that contains a
``param/lm_head.W`` key is rejected. The optimizer is serialised here, not on
``AdamW`` (it stays a plain object; no ``state_dict``).
"""
from __future__ import annotations

import numpy as np

from nanogpt.config import DTYPE, ModelConfig
from nanogpt.model import GPT
from nanogpt.optim import AdamW

FORMAT = "nanogpt-numpy-ckpt-v1"
_CFG_FIELDS = ("V", "T", "d", "L", "H")
_OPT_SCALARS = ("t", "lr", "beta1", "beta2", "eps", "weight_decay")


def save(path, model: GPT, params: dict, opt: AdamW = None) -> None:
    """Write ``params`` (and ``opt`` if given) to ``path`` exactly as named."""
    keys = list(model.param_keys())
    if set(params) != set(keys):
        raise ValueError(
            f"checkpoint.save: params keys must equal model.param_keys(); "
            f"missing={sorted(set(keys) - set(params))}, extra={sorted(set(params) - set(keys))}"
        )
    data = {"format": np.array(FORMAT)}
    for f in _CFG_FIELDS:
        data[f"cfg_{f}"] = np.array(getattr(model.cfg, f), dtype=np.int64)
    for k in keys:
        a = np.asarray(params[k])
        if a.dtype != DTYPE or not a.flags["C_CONTIGUOUS"]:
            raise ValueError(f"checkpoint.save: params[{k!r}] must be C-contiguous float64")
        data[f"param/{k}"] = a

    if opt is not None:
        if set(opt.params) != set(keys):
            raise ValueError("checkpoint.save: optimizer params do not match model.param_keys()")
        for k in keys:
            if opt.params[k] is not params[k]:
                raise ValueError(
                    f"checkpoint.save: optimizer param {k!r} is not the same array object as params[{k!r}]"
                )
        data["opt_t"] = np.array(opt.t, dtype=np.int64)
        data["opt_lr"] = np.array(opt.lr, dtype=DTYPE)
        data["opt_beta1"] = np.array(opt.betas[0], dtype=DTYPE)
        data["opt_beta2"] = np.array(opt.betas[1], dtype=DTYPE)
        data["opt_eps"] = np.array(opt.eps, dtype=DTYPE)
        data["opt_weight_decay"] = np.array(opt.weight_decay, dtype=DTYPE)
        for k in keys:
            data[f"opt_m/{k}"] = opt.state[k]["m"]
            data[f"opt_v/{k}"] = opt.state[k]["v"]

    with open(path, "wb") as f:  # file object: np.savez must not append ".npz" behind our back
        np.savez(f, **data)


def load(path):
    """Return ``(cfg, params, opt_state_or_None)``. Never unpickles anything."""
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "format" not in files or str(z["format"]) != FORMAT:
            got = str(z["format"]) if "format" in files else None
            raise ValueError(f"checkpoint.load: unsupported format {got!r}, expected {FORMAT!r}")
        missing_cfg = [f"cfg_{f}" for f in _CFG_FIELDS if f"cfg_{f}" not in files]
        if missing_cfg:
            raise ValueError(f"checkpoint.load: missing config fields {missing_cfg}")
        cfg = ModelConfig(**{f: int(z[f"cfg_{f}"]) for f in _CFG_FIELDS})

        if "param/lm_head.W" in files:
            raise ValueError("checkpoint.load: 'param/lm_head.W' present; the tie must be stored once as wte.W")
        want = list(GPT(cfg).param_keys())
        have = {f[len("param/") :] for f in files if f.startswith("param/")}
        if have != set(want):
            raise ValueError(
                f"checkpoint.load: param keys mismatch; missing={sorted(set(want) - have)}, "
                f"extra={sorted(have - set(want))}"
            )
        params = {}
        for k in want:
            a = np.array(z[f"param/{k}"], dtype=DTYPE, order="C", copy=True)
            params[k] = a

        opt_state = None
        if any(f.startswith("opt_") for f in files):
            absent = [f"opt_{s}" for s in _OPT_SCALARS if f"opt_{s}" not in files]
            if absent:
                raise ValueError(f"checkpoint.load: incomplete optimizer state, missing {absent}")
            m, v = {}, {}
            for k in want:
                for tag, dst in (("m", m), ("v", v)):
                    name = f"opt_{tag}/{k}"
                    if name not in files:
                        raise ValueError(f"checkpoint.load: missing {name}")
                    arr = np.array(z[name], dtype=DTYPE, order="C", copy=True)
                    if arr.shape != params[k].shape:
                        raise ValueError(f"checkpoint.load: {name} has shape {arr.shape}, expected {params[k].shape}")
                    dst[k] = arr
            opt_state = {
                "t": int(z["opt_t"]),
                "lr": float(z["opt_lr"]),
                "betas": (float(z["opt_beta1"]), float(z["opt_beta2"])),
                "eps": float(z["opt_eps"]),
                "weight_decay": float(z["opt_weight_decay"]),
                "m": m,
                "v": v,
            }
    return cfg, params, opt_state


def restore_optimizer(params: dict, opt_state: dict) -> AdamW:
    """Rebuild an ``AdamW`` whose ``params`` are the SAME array objects as ``params``
    and whose ``t`` / ``m`` / ``v`` equal the saved ones."""
    opt = AdamW(
        params,
        lr=opt_state["lr"],
        betas=opt_state["betas"],
        eps=opt_state["eps"],
        weight_decay=opt_state["weight_decay"],
    )
    if set(opt.state) != set(opt_state["m"]) or set(opt.state) != set(opt_state["v"]):
        raise ValueError("restore_optimizer: optimizer state keys do not match params")
    opt.t = int(opt_state["t"])
    for k in opt.state:
        opt.state[k]["m"][...] = opt_state["m"][k]
        opt.state[k]["v"][...] = opt_state["v"][k]
    return opt
