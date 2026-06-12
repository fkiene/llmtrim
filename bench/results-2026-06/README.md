# Benchmark results snapshot — June 2026

Raw per-case A/B results backing the tables in [`bench/README.md`](../README.md).
Committed as measurement evidence: reruns hit a live model and won't reproduce
byte-for-byte.

- **Produced from:** llmtrim commit `af1f9f3` (v0.1.6-dev), 2026-06-13
- **Model:** `qwen/qwen3-next-80b-a3b-instruct` · judge (open-ended shapes only): `gpt-4o-mini`
- **Contents:** one JSON per corpus×preset run (`<corpus>__<preset>.json`; bare
  `<corpus>.json` = the preset cited in the README table). Each file: run config,
  aggregate savings/quality, and per-case metrics (token counts, costs, quality
  orig vs compressed). No dataset text — corpora are license-bound and rebuilt
  via `bench/scripts/download.py`.
- **Rerun:** `bench/scripts/run_all.sh` (needs an API key; see `bench/README.md`).

`bench/results/` (gitignored) stays the scratch directory for new runs; future
snapshots get their own dated directory.
