"""nanoGPT in NumPy — a from-scratch, dependency-free (beyond NumPy/SciPy)
reimplementation of nanoGPT's forward/backward math, built module by module
with central-difference gradient checks as the correctness oracle.

This package intentionally performs NO eager submodule imports here. Each
submodule (config, tokenizer, gradcheck, and later ops/layernorm/loss/...)
is imported explicitly by the code that needs it. This keeps `import nanogpt`
cheap and keeps Phase 0 from accidentally depending on modules that don't
exist yet.
"""

__version__ = "0.1.0"
