# Root conftest.py (intentionally near-empty).
#
# Its only job is to exist: pytest inserts the directory of any conftest.py
# that has no __init__.py beside it onto sys.path during collection. Since
# this repo root has no __init__.py, having this file here puts the repo
# root on sys.path, so `import nanogpt` resolves under `pytest tests/`
# without an installed package or a PYTHONPATH hack.
