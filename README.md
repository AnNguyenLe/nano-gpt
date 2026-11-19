# nanoGPT in NumPy

A from-scratch reimplementation of nanoGPT's forward/backward math in NumPy
(plus SciPy for a few dense linear-algebra checks), with central-difference
gradient checks as the correctness oracle at every layer. See
`SPRINT_1_PLAN.md` for the Sprint 1 plan and `PROGRESS.md` for the running
build log (read `PROGRESS.md` first if you're resuming this project).

## Setup

```
py -3.15 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(`requirements.txt` lists only `numpy`, `scipy` and `pytest`. If wheels
aren't available for your Python's exact minor version, use `py install 3.13`
or `py install 3.12` and build the venv from that interpreter instead.)

## Running the tests

```
.venv\Scripts\python.exe -m pytest tests/ -q
```

or, with the venv's `Scripts` directory on `PATH` / activated:

```
python -m pytest tests/ -q
```

A root `conftest.py` puts the repo root on `sys.path`, so `import nanogpt`
resolves without installing the package.
