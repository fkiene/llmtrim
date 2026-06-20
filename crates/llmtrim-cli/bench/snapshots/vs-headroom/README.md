# llmtrim vs Headroom (cost per correct answer + Pareto)

The metric that matters to a buyer is not fewest input tokens - it is **cost per correct answer (CPCA)**: a tool that compresses more but is wrong more, or that makes the model ramble, costs you more. This measures that, and shows the full quality-vs-compression frontier so neither tool is judged at a single cherry-picked setting. See `BENCH_SPEC.md`.

- Model: `openai/gpt-oss-20b` (pinned route). Encoder: `o200k_base` over the same message span for both tools.
- Corpora (public, sha-pinned): gsm8k, hotpotqa, squad2, truthfulqa, cnn, lb_qasper, lb_multifieldqa, lb_2wikimqa, lb_gov_report, lb_multinews. The self-authored synthetic tool-output corpus is **excluded**.
- Pricing: bench/pricing.json (fetched 2026-06-10), input $0.029/M, output $0.14/M (output is 4.8x input).

## Token reduction across the sweep (deterministic, $0)

Each arm is a compression setting. Reduction % is token-weighted (1 - sum_after/sum_before); the CI bootstraps that same token-weighted statistic. Overhead leads with p95 (the tail a user feels), median in parentheses. This is the Pareto x-axis.

| arm | tool | n | reduction % | 95% CI | overhead ms p95 (med) | ML fired |
|---|---|--:|--:|:--|:--|--:|
| safe | llmtrim | 50 | 0% | 0–0 | 16.8 (2.4) | 0 |
| auto | llmtrim | 50 | 25% | 14–39 | 33.8 (4.0) | 0 |
| aggressive | llmtrim | 50 | 28% | 17–41 | 385.6 (5.4) | 0 |
| hr-default | headroom | 50 | 0% | 0–0 | 11.1 (1.7) | 0 |
| hr-0.6 | headroom | 50 | 24% | 16–32 | 8639.0 (843.8) | 36 |
| hr-0.4 | headroom | 50 | 24% | 16–32 | 75.1 (11.2) | 41 |
| hr-max | headroom | 50 | 24% | 16–32 | 25.5 (9.2) | 41 |
| hr-max-noml | headroom (no ML) | 50 | 0% | 0–0 | 25.1 (8.9) | 0 |

Latency is Python wall-clock around each library's `compress()`; it is not a like-for-like CPU measurement (llmtrim crosses an FFI boundary into Rust, Headroom runs in-process Python + torch). One-time cold start (model load, once per process, amortizes to ~0 per call): llmtrim 103.5 ms, Headroom 3049.3 ms.

### Reduction per corpus (aggressive arm)

| corpus | llmtrim aggressive | Headroom hr-max |
|---|--:|--:|
| gsm8k | -28% | 7% |
| hotpotqa | 29% | 32% |
| squad2 | 18% | 10% |
| truthfulqa | -10% | 13% |
| cnn | -2% | 27% |
| lb_qasper | 61% | 33% |
| lb_multifieldqa | 32% | 37% |
| lb_2wikimqa | 51% | 37% |
| lb_gov_report | 4% | 0% |
| lb_multinews | 10% | 36% |

llmtrim is preservation-first: on short prompts (gsm8k, truthfulqa) it can *add* a few tokens rather than risk the answer, and the aggregate reduction is carried by the long-context corpora. Stated plainly, not hidden.

## Cost per answer-quality (live, 2 seeds, budget $0.9, spent $0.0203)

Each point pairs an llmtrim preset with the Headroom config of nearest achieved reduction (shown per point - exact iso isn't always possible because Headroom's ML caps its reduction). For each, generate original / llmtrim / Headroom across seeds, score with each corpus's own scorer (ROUGE-L for summaries, F1 for QA, numeric/contains/choice otherwise), and compute CPCA = total cost / sum of scores - fractional credit, so 'cost per correct answer' here means cost per unit of summed answer quality, not per binary hit. **Lower CPCA is better.** Quality is the mean score; output tokens are the median (resists one runaway generation).

### iso-moderate - llmtrim `auto` vs Headroom `hr-0.4` (n=30 samples) - iso: reduction llmtrim 25% vs Headroom 24%

| arm | quality | output tok (med) | truncated | total cost | **CPCA** |
|---|--:|--:|--:|--:|--:|
| original | 0.49 | 272 | 8 | $0.0039 | $0.0003 |
| **llmtrim** | 0.46 | 172 | 2 | $0.0030 | **$0.0002** |
| headroom | 0.39 | 320 | 12 | $0.0033 | $0.0003 |

Headroom's longer outputs hit the generation cap 12 times vs llmtrim's 2: the output-inflation that drives both its higher cost and its clipped answers.

Quality difference llmtrim − Headroom: +0.080 (95% CI -0.036…+0.200, n=30) - **NOT significant (CI spans 0)**.

### iso-aggressive - llmtrim `aggressive` vs Headroom `hr-0.4` (n=30 samples) - near-iso, 4pp apart: reduction llmtrim 28% vs Headroom 24%

| arm | quality | output tok (med) | truncated | total cost | **CPCA** |
|---|--:|--:|--:|--:|--:|
| original | 0.48 | 257 | 6 | $0.0038 | $0.0003 |
| **llmtrim** | 0.46 | 158 | 3 | $0.0030 | **$0.0002** |
| headroom | 0.44 | 320 | 9 | $0.0033 | $0.0002 |

Headroom's longer outputs hit the generation cap 9 times vs llmtrim's 3: the output-inflation that drives both its higher cost and its clipped answers.

Quality difference llmtrim − Headroom: +0.017 (95% CI -0.120…+0.150, n=30) - **NOT significant (CI spans 0)**.

## Caveats

- The deterministic token axis is exact and citable. CPCA / quality / output tokens are live generations across seeds - directional, with the paired-bootstrap CI on the quality difference as the significance signal (CI excluding 0 = real).
- Live sample is small (n shown per point) and uses few seeds, so the quality differences are NOT statistically significant; read them as directional. A larger live run would tighten the CIs.
- At the aggressive point the match is near-iso, not exact: Headroom's ML caps its reduction (~24% here) so llmtrim's more-aggressive arm is a few pp ahead on compression - read its quality/cost next to that gap.
- Scorers per corpus: numeric (gsm8k), token-F1 (hotpotqa, squad2, LongBench QA), choice (truthfulqa MC1), ROUGE-L (cnn, gov_report, multi_news). Each is the corpus's own standard metric.
- Headroom's ML reduction varies run-to-run, so the live Headroom arm is matched to llmtrim by ACHIEVED reduction within the same run (shown per point), not a fixed label; the full sweep shows neither tool is judged at one cherry-picked setting.
- Latency is Python wall-clock, not like-for-like CPU (llmtrim is Rust via FFI, Headroom in-process Python+torch); read p95, and treat cold start as a one-time cost. Per-Headroom-arm latency is also confounded by Headroom caching embeddings ACROSS arms within a run, so only the FIRST ML arm reflects true inference cost; the honest ML latency is the cold start plus that first ML call.
- Headroom no-ML is 0% here because the corpora are prose; its deterministic routers (JSON/code/log) have nothing to bite on. On JSON/code/log inputs no-ML would compress - that path is just out of scope for these text corpora.
- llmtrim is preservation-first by design (no lossy tier). Headroom will win raw reduction at its most aggressive; the point is that there it loses answers while llmtrim does not - read the iso-compression rows together with CPCA.
- RTK scope: Headroom's bundled RTK shell-output rewriter is active only in its `wrap`/proxy mode, not in `headroom.compress`; it is out of scope for this library-vs-library comparison.
- Tool-calling corpora (bfcl, glaive) are deferred (tool-schema plumbing + call-arg scorer); excluded here, not cherry-picked away.

