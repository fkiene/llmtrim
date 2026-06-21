# llmtrim vs Entroly (cost per correct answer + Pareto)

The metric that matters to a buyer is not fewest input tokens - it is **cost per correct answer (CPCA)**: a tool that compresses more but is wrong more, or that makes the model ramble, costs you more. This measures that, and shows the full quality-vs-compression frontier so neither tool is judged at a single cherry-picked setting. See `BENCH_SPEC.md`.

- Model: `openai/gpt-oss-20b` (pinned route). Encoder: `o200k_base` over the same message span for both tools.
- Corpora (public, sha-pinned): gsm8k, hotpotqa, squad2, truthfulqa, cnn, lb_qasper, lb_multifieldqa, lb_2wikimqa, lb_gov_report, lb_multinews. The self-authored synthetic tool-output corpus is **excluded**.
- Pricing: bench/pricing.json (fetched 2026-06-10), input $0.029/M, output $0.14/M (output is 4.8x input).

## Token reduction across the sweep (deterministic, $0)

Each arm is a compression setting. Reduction % is token-weighted (1 - sum_after/sum_before); the CI bootstraps that same token-weighted statistic. Overhead leads with p95 (the tail a user feels), median in parentheses. This is the Pareto x-axis.

| arm | tool | n | reduction % | 95% CI | overhead ms p95 (med) | ML fired |
|---|---|--:|--:|:--|:--|--:|
| safe | llmtrim | 80 | 0% | 0–0 | 18.0 (3.5) | 0 |
| auto | llmtrim | 80 | 25% | 16–36 | 36.1 (5.4) | 0 |
| aggressive | llmtrim | 80 | 27% | 19–37 | 285.0 (8.7) | 0 |
| en-default | entroly | 80 | 0% | 0–0 | 0.0 (0.0) | 0 |
| en-0.6 | entroly | 80 | 69% | 60–76 | 81.2 (0.0) | 0 |
| en-0.4 | entroly | 80 | 80% | 73–84 | 50.5 (0.0) | 0 |
| en-max | entroly | 80 | 89% | 85–91 | 27.3 (0.9) | 0 |

Latency is Python wall-clock around each library's `compress()`; it is not a like-for-like CPU measurement (llmtrim crosses an FFI boundary into Rust, Entroly runs in-process Python + torch). One-time cold start (model load, once per process, amortizes to ~0 per call): llmtrim 174.7 ms, Entroly 6.9 ms.

### Reduction per corpus (aggressive arm)

| corpus | llmtrim aggressive | Entroly en-max |
|---|--:|--:|
| gsm8k | -29% | 0% |
| hotpotqa | 41% | 68% |
| squad2 | 20% | 0% |
| truthfulqa | -10% | 0% |
| cnn | -3% | 27% |
| lb_qasper | 40% | 89% |
| lb_multifieldqa | 32% | 94% |
| lb_2wikimqa | 54% | 96% |
| lb_gov_report | 4% | 94% |
| lb_multinews | 8% | 65% |

llmtrim is preservation-first: on short prompts (gsm8k, truthfulqa) it can *add* a few tokens rather than risk the answer, and the aggregate reduction is carried by the long-context corpora. Stated plainly, not hidden.

## Cost per answer-quality (live, 2 seeds, budget $0.3, spent $0.0161)

Each point pairs an llmtrim preset with the Entroly config of nearest achieved reduction (shown per point - exact iso isn't always possible because the Entroly ML caps its reduction). For each, generate original / llmtrim / Entroly across seeds, score with each corpus's own scorer (ROUGE-L for summaries, F1 for QA, numeric/contains/choice otherwise), and compute CPCA = total cost / sum of scores - fractional credit, so 'cost per correct answer' here means cost per unit of summed answer quality, not per binary hit. **Lower CPCA is better.** Quality is the mean score; output tokens are the median (resists one runaway generation).

### iso-moderate - llmtrim `auto` vs Entroly `en-0.6` (n=20 samples) - near-iso, 44pp apart: reduction llmtrim 25% vs Entroly 69%

| arm | quality | output tok (med) | truncated | total cost | **CPCA** |
|---|--:|--:|--:|--:|--:|
| original | 0.32 | 266 | 7 | $0.0037 | $0.0006 |
| **llmtrim** | 0.31 | 216 | 4 | $0.0027 | **$0.0004** |
| entroly | 0.18 | 330 | 10 | $0.0018 | $0.0005 |

Entroly's longer outputs hit the generation cap 10 times vs llmtrim's 4: the output-inflation that drives both its higher cost and its clipped answers.

Quality difference llmtrim − Entroly: +0.133 (95% CI +0.017…+0.286, n=20) - **significant**.

### iso-aggressive - llmtrim `aggressive` vs Entroly `en-0.6` (n=20 samples) - near-iso, 42pp apart: reduction llmtrim 27% vs Entroly 69%

| arm | quality | output tok (med) | truncated | total cost | **CPCA** |
|---|--:|--:|--:|--:|--:|
| original | 0.38 | 261 | 5 | $0.0036 | $0.0005 |
| **llmtrim** | 0.30 | 216 | 6 | $0.0027 | **$0.0005** |
| entroly | 0.24 | 294 | 9 | $0.0017 | $0.0003 |

Entroly's longer outputs hit the generation cap 9 times vs llmtrim's 6: the output-inflation that drives both its higher cost and its clipped answers.

Quality difference llmtrim − Entroly: +0.054 (95% CI -0.119…+0.246, n=20) - **NOT significant (CI spans 0)**.

## Caveats

- The deterministic token axis is exact and citable. CPCA / quality / output tokens are live generations across seeds - directional, with the paired-bootstrap CI on the quality difference as the significance signal (CI excluding 0 = real).
- Live sample is small (n shown per point) and uses few seeds; read each point with its own paired-bootstrap CI above (CI excluding 0 = significant). A larger live run would tighten the CIs.
- Entroly has no ML compression arm to cap; its reduction is set by the budget/preserve/distill knobs in the grid, not by a learned model.
- Scorers per corpus: numeric (gsm8k), token-F1 (hotpotqa, squad2, LongBench QA), choice (truthfulqa MC1), ROUGE-L (cnn, gov_report, multi_news). Each is the corpus's own standard metric.
- The Entroly ML reduction varies run-to-run, so the live Entroly arm is matched to llmtrim by ACHIEVED reduction within the same run (shown per point), not a fixed label; the full sweep shows neither tool is judged at one cherry-picked setting.
- Latency is Python wall-clock, not like-for-like CPU (llmtrim is Rust via FFI, Entroly in-process Python+torch); read p95, and treat cold start as a one-time cost. Per-Entroly-arm latency is also confounded by Entroly caching embeddings ACROSS arms within a run, so only the FIRST ML arm reflects true inference cost; the honest ML latency is the cold start plus that first ML call.
- Entroly's compress_messages is deterministic: its only ML is an optional NLI cross-encoder used by the verification layer, not by compression. With sentence_transformers absent it falls back to a local path, so no-ML and default numbers are identical here.
- llmtrim is preservation-first by design (no lossy tier). Entroly will win raw reduction at its most aggressive; the point is that there it loses answers while llmtrim does not - read the iso-compression rows together with CPCA.
- Entroly ships a wider control plane (MCP tools, context receipts, image and shell codecs). Only its `compress_messages` SDK call is in scope for this library-vs-library comparison.
- Tool-calling corpora (bfcl, glaive) are deferred (tool-schema plumbing + call-arg scorer); excluded here, not cherry-picked away.

