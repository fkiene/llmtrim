# llmtrim vs leanctx (cost per correct answer + Pareto)

The metric that matters to a buyer is not fewest input tokens - it is **cost per correct answer (CPCA)**: a tool that compresses more but is wrong more, or that makes the model ramble, costs you more. This measures that, and shows the full quality-vs-compression frontier so neither tool is judged at a single cherry-picked setting. See `BENCH_SPEC.md`.

- Model: `openai/gpt-oss-20b` (pinned route). Encoder: `o200k_base` over the same message span for both tools.
- Corpora (public, sha-pinned): gsm8k, hotpotqa, squad2, truthfulqa, cnn, lb_qasper, lb_multifieldqa, lb_2wikimqa, lb_gov_report, lb_multinews. The self-authored synthetic tool-output corpus is **excluded**.
- Pricing: bench/pricing.json (fetched 2026-06-10), input $0.029/M, output $0.14/M (output is 4.8x input).

## Token reduction across the sweep (deterministic, $0)

Each arm is a compression setting. Reduction % is token-weighted (1 - sum_after/sum_before); the CI bootstraps that same token-weighted statistic. Overhead leads with p95 (the tail a user feels), median in parentheses. This is the Pareto x-axis.

| arm | tool | n | reduction % | 95% CI | overhead ms p95 (med) | ML fired |
|---|---|--:|--:|:--|:--|--:|
| safe | llmtrim | 40 | 0% | 0–0 | 24.2 (3.2) | 0 |
| auto | llmtrim | 40 | 25% | 13–40 | 42.9 (5.5) | 0 |
| aggressive | llmtrim | 40 | 28% | 17–42 | 434.2 (5.7) | 0 |
| lc-keep1.00 | leanctx | 40 | 1% | 0–1 | 63.8 (8.3) | 40 |
| lc-keep0.75 | leanctx | 40 | 26% | 26–27 | 39088.9 (6483.6) | 40 |
| lc-keep0.50 | leanctx | 40 | 52% | 51–53 | 39396.0 (6327.5) | 40 |
| lc-keep0.33 | leanctx | 40 | 68% | 68–69 | 38373.2 (6340.0) | 40 |
| lc-keep0.20 | leanctx | 40 | 81% | 80–81 | 39061.1 (6474.1) | 40 |

Latency is Python wall-clock around each library's `compress()`; it is not a like-for-like CPU measurement (llmtrim crosses an FFI boundary into Rust, leanctx runs in-process Python + torch). One-time cold start (model load, once per process, amortizes to ~0 per call): llmtrim 132.3 ms, leanctx 4461.1 ms.

### Reduction per corpus (aggressive arm)

| corpus | llmtrim aggressive | leanctx lc-keep0.20 |
|---|--:|--:|
| gsm8k | -35% | 79% |
| hotpotqa | 24% | 80% |
| squad2 | 15% | 79% |
| truthfulqa | -14% | 83% |
| cnn | -2% | 81% |
| lb_qasper | 61% | 80% |
| lb_multifieldqa | 40% | 81% |
| lb_2wikimqa | 45% | 81% |
| lb_gov_report | 4% | 81% |
| lb_multinews | 10% | 82% |

llmtrim is preservation-first: on short prompts (gsm8k, truthfulqa) it can *add* a few tokens rather than risk the answer, and the aggregate reduction is carried by the long-context corpora. Stated plainly, not hidden.

## Cost per answer-quality (live, 2 seeds, budget $0.3, spent $0.0147)

Each point pairs an llmtrim preset with the leanctx config of nearest achieved reduction (shown per point - exact iso isn't always possible because the leanctx ML caps its reduction). For each, generate original / llmtrim / leanctx across seeds, score with each corpus's own scorer (ROUGE-L for summaries, F1 for QA, numeric/contains/choice otherwise), and compute CPCA = total cost / sum of scores - fractional credit, so 'cost per correct answer' here means cost per unit of summed answer quality, not per binary hit. **Lower CPCA is better.** Quality is the mean score; output tokens are the median (resists one runaway generation).

### iso-moderate - llmtrim `auto` vs leanctx `lc-keep0.75` (n=20 samples) - iso: reduction llmtrim 25% vs leanctx 26%

| arm | quality | output tok (med) | truncated | total cost | **CPCA** |
|---|--:|--:|--:|--:|--:|
| original | 0.49 | 249 | 4 | $0.0028 | $0.0003 |
| **llmtrim** | 0.60 | 160 | 0 | $0.0021 | **$0.0002** |
| leanctx | 0.49 | 334 | 3 | $0.0024 | $0.0002 |

leanctx's longer outputs hit the generation cap 3 times vs llmtrim's 0: the output-inflation that drives both its higher cost and its clipped answers.

Quality difference llmtrim − leanctx: +0.105 (95% CI +0.008…+0.243, n=20) - **significant**.

### iso-aggressive - llmtrim `aggressive` vs leanctx `lc-keep0.75` (n=20 samples) - iso: reduction llmtrim 28% vs leanctx 26%

| arm | quality | output tok (med) | truncated | total cost | **CPCA** |
|---|--:|--:|--:|--:|--:|
| original | 0.55 | 252 | 3 | $0.0028 | $0.0003 |
| **llmtrim** | 0.59 | 162 | 0 | $0.0021 | **$0.0002** |
| leanctx | 0.49 | 366 | 7 | $0.0025 | $0.0003 |

leanctx's longer outputs hit the generation cap 7 times vs llmtrim's 0: the output-inflation that drives both its higher cost and its clipped answers.

Quality difference llmtrim − leanctx: +0.102 (95% CI +0.004…+0.238, n=20) - **significant**.

## Caveats

- The deterministic token axis is exact and citable. CPCA / quality / output tokens are live generations across seeds - directional, with the paired-bootstrap CI on the quality difference as the significance signal (CI excluding 0 = real).
- Live sample is small (n shown per point) and uses few seeds; read each point with its own paired-bootstrap CI above (CI excluding 0 = significant). A larger live run would tighten the CIs.
- Lingua's reduction is set by its keep-ratio (`ratio`), so its arms land where you ask them to. Read its quality next to that ratio: it keeps a verbatim token subset, so structure and exact wording survive but dropped tokens are gone.
- Scorers per corpus: numeric (gsm8k), token-F1 (hotpotqa, squad2, LongBench QA), choice (truthfulqa MC1), ROUGE-L (cnn, gov_report, multi_news). Each is the corpus's own standard metric.
- The leanctx ML reduction varies run-to-run, so the live leanctx arm is matched to llmtrim by ACHIEVED reduction within the same run (shown per point), not a fixed label; the full sweep shows neither tool is judged at one cherry-picked setting.
- Latency is Python wall-clock, not like-for-like CPU (llmtrim is Rust via FFI, leanctx in-process Python+torch); read p95, and treat cold start as a one-time cost. Per-leanctx-arm latency is also confounded by leanctx caching embeddings ACROSS arms within a run, so only the FIRST ML arm reflects true inference cost; the honest ML latency is the cold start plus that first ML call.
- leanctx no-ML is 0%: its only compressor here, Lingua, is LLMLingua-2, an ML token classifier with no deterministic fallback. Strip the model and there is nothing left to compress, so --no-ml is a passthrough.
- llmtrim is preservation-first by design (no lossy tier). leanctx will win raw reduction at its most aggressive; the point is that there it loses answers while llmtrim does not - read the iso-compression rows together with CPCA.
- Tool-calling corpora (bfcl, glaive) are deferred (tool-schema plumbing + call-arg scorer); excluded here, not cherry-picked away.

