"""Generic benchmark configuration: corpora, scorers, llmtrim presets, and paths.

Nothing here is competitor-specific. A competitor brings its own config grid (see
`benchkit.competitors.base.Competitor.config_grid`); the corpora, the deterministic
scorers, the llmtrim tiers, and the iso-compression match points are shared by every
comparison.
"""
from pathlib import Path

# Package layout: this file is scripts/benchkit/config.py, so parents[3] is crates/llmtrim-cli.
CRATE_ROOT = Path(__file__).resolve().parents[3]  # crates/llmtrim-cli (bench/ lives here)
DATA_DIR = CRATE_ROOT / "bench" / "data"
PRICING = CRATE_ROOT / "bench" / "pricing.json"
RESULTS_DIR = CRATE_ROOT / "bench" / "snapshots" / "vs-headroom"
BASELINE = CRATE_ROOT / "bench" / "baseline.json"

# Content corpora for the headline. The self-authored synthetic tool-output corpus is
# deliberately excluded (a vendor-written corpus discredits the numbers next to it).
QA_CORPORA = ["gsm8k", "hotpotqa", "squad2", "truthfulqa", "cnn"]
LONGBENCH = ["lb_qasper", "lb_multifieldqa", "lb_2wikimqa", "lb_gov_report", "lb_multinews"]
CORPORA = QA_CORPORA + LONGBENCH

# Scorers score_v2() can compute deterministically. 'choice' (truthfulqa MC1) and 'rouge'
# (summarization) are added on top of lib.score(); 'tool'/'judge'/'pass@1' need an LLM judge
# or harness, so those rows are dropped from the live leg (but still counted on the token axis).
DETERMINISTIC_SCORERS = {"numeric", "f1", "contains", "choice", "rouge"}

# Summarization corpora must be scored with ROUGE, not bag-of-words token-F1 (P0-3). The
# data files predate this, so the scorer is forced here by corpus name.
SUMMARIZATION = {"cnn", "lb_gov_report", "lb_multinews"}

# The three user-facing tiers (decision): safe = lossless input only, auto = the smart
# default (shape-routes to agent/code/rag/aggressive per request), aggressive = squeeze
# everything, accept lossy. The internal routing targets (agent/code/rag/cache) are not
# swept directly - auto exercises them per request shape.
LLMTRIM_PRESETS = ["safe", "auto", "aggressive"]

# Live points: each names a llmtrim preset. The competitor arm is chosen PER RUN as the grid
# config whose achieved reduction is closest to that preset's (P0-2: never pair against a
# no-op, and robust to run-to-run ML variance - match within the same run, not a stale label).
MATCHED = {"iso-moderate": "auto", "iso-aggressive": "aggressive"}
