# Benchmark spec - llmtrim vs Headroom

What `scripts/bench.py headroom` actually does and why. The artifact it produces is
`snapshots/vs-headroom/{results.json,README.md}`.

## Why this benchmark exists

Reporting token reduction and answer quality as two separate axes lets a lossy compressor
"win" the reduction column by deleting the answer. This benchmark scores the buyer's real
objective instead: **cost per unit of answer quality** - a tool that compresses more but is
wrong more, or that makes the model ramble, costs more.

## Two legs

- **Deterministic ($0, default run).** Sweep llmtrim presets and a Headroom config grid over
  public corpora; record token reduction (token-weighted, with a bootstrap CI on that same
  statistic), compress overhead (p95 + median), and cold start. No API calls. This is the
  Pareto x-axis and the citable core.
- **Live (`--live`, budget-capped).** At two iso-compression points, generate
  original/llmtrim/Headroom answers across seeds, score each with the corpus's own scorer,
  and compute CPCA. A hard $ guard estimates each call from `pricing.json` and stops before
  overrun, writing partial results.

## Headline metric: CPCA

```
CPCA(arm) = total token cost / sum of per-case scores
```

Cost is `price_in*input + price_out*output` from `pricing.json`. Scores are **fractional**
(ROUGE-L / token-F1 / numeric / choice in [0,1]), so dividing by their sum gives "cost per
unit of answer quality" - one flipped answer can't swing an integer denominator. We do not
threshold to a binary "correct".

Significance on the quality gap is a **paired bootstrap** on per-sample
(llmtrim − Headroom) score differences; a 95% CI excluding 0 means a real difference. (No
McNemar - the scorers are continuous.)

## Corpora

Public, fetched by `download.py` (sha-pinned in `data/manifest.json`), **not vendored** - the
CI gate and any reproduction download them fresh. The self-authored synthetic tool-output
corpus is excluded (a vendor-written corpus discredits the numbers beside it).

| corpus | source | scorer |
|---|---|---|
| gsm8k | openai/gsm8k | numeric |
| hotpotqa | hotpotqa/hotpot_qa | token-F1 |
| squad2 | rajpurkar/squad_v2 | token-F1 / contains (no-answer) |
| truthfulqa | truthfulqa/truthful_qa (MC1) | choice (letter) |
| cnn | abisee/cnn_dailymail | ROUGE-L |
| lb_qasper / lb_multifieldqa / lb_2wikimqa | THUDM/LongBench (bzantium mirror) | token-F1 |
| lb_gov_report / lb_multinews | THUDM/LongBench | ROUGE-L |

Tool-calling corpora (bfcl, glaive) are **deferred**: they need tool-schema plumbing through
both libraries and a call-arg scorer. They are llmtrim's expected home turf, so their absence
is conservative against llmtrim, not for it.

## Arms

- **llmtrim presets:** `safe` (lossless input), `auto` (the shape-routing default), `aggressive`
  (squeeze, accept lossy). No lossy tier is added - llmtrim stays preservation-first.
- **Headroom grid:** `hr-default` (its library defaults, a near-no-op on prose) → `hr-0.6` →
  `hr-0.4` → `hr-max`, spanning no-op to max aggression. Run in its **best** mode
  (`headroom-ai[all]`, ML on).
- **Headroom no-ML:** `hr-max` with the ML model disabled, to show what the torch+ModernBERT
  dependency buys (on prose: nothing - its deterministic routers no-op).

## Other competitors

The harness runs more than Headroom. Each competitor is one file under
`scripts/benchkit/competitors/` and takes one of two shapes:

- **On-grid (the `Competitor` interface):** the adapter exposes a config grid and a
  `compress(messages, cfg, repeats)` that the generic engine sweeps over the same prose
  corpora and o200k span as llmtrim. These ride the deterministic sweep ($0) and the optional
  live CPCA leg.
  - `leanctx` - LLMLingua-2 (Microsoft) via the `leanctx` package, run in its local Lingua
    mode. ML compression on the local machine, no network, $0.
  - `entroly` - the `entroly` package's `compress_messages`, a deterministic local path (no
    LLM call). Its budget/preserve/distill knobs drive the sweep; high reduction is lossy.
- **Self-contained (`run(argv)` + `SELF_CONTAINED`):** the comparison does not fit the
  corpora x grid engine, so the adapter owns its own `run()` and the CLI dispatches to it
  before the generic engine parses flags. These compress tool OUTPUT (shell-command text), not
  message arrays, and run on their own real-output fixtures.
  - `rtk` - the RTK CLI's native per-tool filters, scored on `rtk_fixtures/*.txt`. Lossy.
  - `snip` - the snip CLI's per-tool filters, scored on `snip_fixtures/*.txt`. Lossy.
  - `caveman` - measures OUTPUT tokens from a paid live call against a system-prompt strategy,
    a different axis again (not token-for-token parity with rtk/snip).

## Iso-compression matching

Headroom's ML reduction varies run-to-run and caps on prose, so the live Headroom arm is
chosen **per run** as the grid config whose achieved reduction is nearest the llmtrim
preset's (never the no-op `hr-default`; ties break toward the more-aggressive config). When an
exact match isn't possible (Headroom's cap), the point is labelled "near-iso, N pp apart" and
the gap is shown - so a few-pp difference is disclosed, not hidden.

## Latency

Python wall-clock around each `compress()`. Not a like-for-like CPU measurement (llmtrim is
Rust via FFI; Headroom is in-process Python+torch). Reported as p95 (lead) + median + a
one-time cold start. Per-Headroom-arm latency is confounded by Headroom caching embeddings
across arms, so only the first ML arm reflects true inference cost.

## Pinning & reproducibility

- `headroom-ai==0.26.0`, `tiktoken==0.13.0`, `rouge-score==0.1.2` (requirements file). llmtrim
  is built locally; its version + the Headroom version + platform + data-manifest date are
  recorded in `results.json["meta"]["provenance"]`.
- `make setup` (wheel + deps + corpora), then `make bench` / `bench LIVE=1` / `bench NOML=1`
  / `check` / `baseline` (see Makefile).
- **CI gate** (`.github/workflows/bench-gate.yml`, advisory): downloads the gated corpora,
  builds the wheel, runs `--check` - asserts lossless `safe`, `auto>0`, `aggressive≥auto`,
  data-manifest integrity, and llmtrim reduction within ±3pp of `baseline.json`. Headroom is
  not gated (non-deterministic ML). Moving the numbers requires committing a new
  `baseline.json` - a reviewed diff line.

## Model

`openai/gpt-oss-20b` via the pinned `wandb/fp4` route. A second tier (e.g. `gpt-4o-mini`, for
a non-zero cache-read term) is future work.

## Honest limitations

- Live sample is small and uses few seeds, so quality differences are typically **not
  statistically significant** - read them as directional.
- Single model; the output-inflation cost argument is ratio-dependent and shown on one ratio.
- Headroom's reduction is non-deterministic run-to-run; the deterministic table reports a
  single run, the live arm is matched within its own run.
- Headroom no-ML is 0% only because these corpora are prose; on JSON/code/log inputs its
  deterministic routers would compress (out of scope here).
- Headroom's bundled RTK shell-output rewriter is active only in its `wrap`/proxy mode, not in
  `headroom.compress`; out of scope for this library-vs-library comparison.
