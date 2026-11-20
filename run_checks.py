"""run_checks.py -- one command that runs the whole correctness suite and writes a log.

    python run_checks.py            # writes test_log.txt next to this file, exits with pytest's status

The log records the environment (Python / NumPy / SciPy versions), the corpus identity (source and
SHA-256, resolved with no network access) and the full `pytest` output, so a result can be
reproduced and compared from a clean checkout.
"""
from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "test_log.txt"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy

    from nanogpt.tokenizer import load_corpus_and_tokenizer

    info, tok = load_corpus_and_tokenizer()
    header = [
        f"nanogpt_numpy test log, {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"python {platform.python_version()} on {platform.platform()}",
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"corpus: source={info.source} generator={info.generator} seed={info.seed} "
        f"chars={info.n_chars} sha256={info.sha256}",
        f"vocab size V={tok.vocab_size}",
        "command: python -m pytest tests/ -q -p no:cacheprovider",
        "",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    body = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr.strip() else "")
    LOG.write_text("\n".join(header) + body + f"\nexit status: {proc.returncode}\n", encoding="utf-8")
    print(f"wrote {LOG.name}: exit status {proc.returncode}")
    print(body.strip().splitlines()[-1] if body.strip() else "(no output)")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
