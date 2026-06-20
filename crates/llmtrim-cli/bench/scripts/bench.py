#!/usr/bin/env python3
"""Single entry point for the benchkit benchmark: `bench.py <competitor> [flags]`.

Thin wrapper: put this scripts/ dir on sys.path and hand off to benchkit.cli:main. Run it
through the Makefile (`make -C crates/llmtrim-cli/bench help`) where possible.

    python3 scripts/bench.py headroom --check --limit 5
    python3 scripts/bench.py headroom --limit 40
    OPENROUTER_API_KEY=... python3 scripts/bench.py headroom --live --budget 0.90
    python3 scripts/bench.py caveman          # self-contained system-prompt A/B
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchkit.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
