# Named-benchmark accuracy snapshot (June 2026)

Raw per-case A/B results backing the named-benchmark accuracy table in
[`bench/README.md`](../../README.md). These are the standard academic suites a reader can
compare against published numbers: GSM8K (already in the main frontier), TruthfulQA,
SQuAD v2, and BFCL. Committed as measurement evidence: a rerun hits a live model and will
not reproduce byte-for-byte.

- **Produced from:** llmtrim commit `7b1747c` (v0.1.11-dev), 2026-06-14.
- **Model:** `qwen/qwen3-next-80b-a3b-instruct` via OpenRouter, route unset (let OpenRouter
  pick). Same headline model as the main frontier, so the rows are comparable.
- **Files:** `<corpus>__<preset>.json`. The headline rows use the conservative
  shape-matched preset (`truthfulqa__safe`, `squad2__rag`, `bfcl__agent`); the `__auto`
  files are the same corpora under the default `auto` preset, kept to show that the picture
  holds out of the box. `truthfulqa__ngram.json` is the discarded experiment that turned
  ngram on (see the TruthfulQA note below).
- **n:** 20 cases per run.
- **Scorers:** TruthfulQA = `choice` (MC1, the selected option letter); SQuAD v2 = `f1`
  for answerable rows and `contains` against the `unanswerable` sentinel for the no-answer
  rows (a correct refusal scores as a hit); BFCL = `tool` (the called function name).

## What the numbers show

- **BFCL live_multiple (agent):** 32.6% input saved, quality 95% to 95% (retention +0.0pp,
  paired 95% CI ±15.2). The multi-tool slice gives 2 to 37 candidate functions per call, so
  tool selection drops the schemas the query doesn't need without dropping the gold tool.
  The single-tool `simple` slice has nothing to select, so it is not used here.
- **SQuAD v2 (rag):** 11.1% input saved, quality 84.2% to 84.2% (retention -0.0pp,
  paired 95% CI ±15.2). Clean compression with no quality loss, unanswerable rows included.
- **TruthfulQA MC1 (safe):** ~0% input saved, quality 75% to 75%. A ~75-token prompt that
  is almost all answer text, so the safe preset finds nothing to cut and factual accuracy
  holds exactly. A second `safe` run scored 85% to 80%; the requests are near-identical at
  0% compression, so the spread is model sampling noise, not a compression effect. The one
  lever that touches it (`ngram` on the repeated answer stems) buys ~7% input but moves
  quality 80% to 75% (-5.0±18.4pp at n=20), inside the noise band but not a clean win, so
  the conservative row is the honest one.

No dataset text is committed: the corpora are license-bound and rebuilt deterministically
via `bench/scripts/download.py 40 truthfulqa,squad2,bfcl` (pins live in
`bench/data/manifest.json`).

## Rerun

```bash
python3 bench/scripts/download.py 40 truthfulqa,squad2,bfcl
cargo run -q --features live -- bench quality \
  --corpus bench/data/squad2.jsonl --preset rag \
  --model qwen/qwen3-next-80b-a3b-instruct --route "" --n 20
```
