#!/usr/bin/env python3
"""Refresh the embedded LMArena capability snapshot.

The agent-loop frugality directive is model-gated (see `llmtrim-core/src/capability.rs`): it
only injects for models capable enough to act on the steer. The signal is a static snapshot of
the LMArena text leaderboard (overall Elo), embedded in the core crate. Run this on release to
refresh it, the same way `bench/pricing.json` is refreshed.

Source: the official LMArena leaderboard dataset on Hugging Face (full board, not the top-10
free API). Reads the `text/latest` parquet, keeps the `overall` category, writes
`crates/llmtrim-core/data/lmarena_text.json` as `{leaderboard_publish_date, models:{id:elo}}`.

Usage:  python3 tools/refresh_lmarena.py
Deps:   pip install pyarrow  (only needed to run this script, not to build the crate)
"""

import json
import sys
import urllib.request
from pathlib import Path

PARQUET_URL = (
    "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/"
    "resolve/main/text/latest-00000-of-00001.parquet"
)
OUT = Path(__file__).resolve().parent.parent / "crates/llmtrim-core/data/lmarena_text.json"


def main() -> int:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        sys.exit("pyarrow is required: pip install pyarrow")

    tmp = OUT.with_suffix(".parquet.tmp")
    urllib.request.urlretrieve(PARQUET_URL, tmp)  # noqa: S310 (trusted HF url)
    rows = pq.read_table(tmp).to_pylist()
    tmp.unlink()

    overall = sorted(
        (r for r in rows if r["category"] == "overall"),
        key=lambda r: -r["rating"],
    )
    models: dict[str, int] = {}
    for r in overall:  # first (highest-rated) wins per model id
        models.setdefault(r["model_name"], round(r["rating"]))

    date = overall[0]["leaderboard_publish_date"] if overall else None
    OUT.write_text(
        json.dumps(
            {"leaderboard_publish_date": date, "models": models},
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {len(models)} models (board {date}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
