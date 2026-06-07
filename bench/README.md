# llmtrim benchmark

Two axes, measured live: **tokens saved** (real tokenizer, at compress time) and **quality retained** (the A/B delta between the model's answer on the *original* vs the *compressed* request). A preset is only honest if quality holds at its token saving — the frontier of (saved, retained) is the benchmark, not the saving alone.

- **Model:** `openai/gpt-oss-20b` via OpenRouter, pinned to the **Groq** upstream (async-openai byot — the exact compressed body is sent, injected fields intact).
- **Scoring:** ground-truth where possible (numeric-exact for math, pass@1 that *runs the unit tests* for code), token-F1 for extractive QA, tool-call match for agents, an LLM judge only for open-ended shapes.
- **Cost:** priced from a pinned [models.dev](https://models.dev) snapshot (`bench/pricing.json`), Groq rates, cached input billed at the `cache_read` rate.
- **Cache used %:** share of compressed input served from the provider prompt cache (`usage.prompt_tokens_details.cached_tokens`).


## Bottom line

Across **87 A/B cases** on this real-usage mix (generation, chat, reasoning, code, RAG, agent, summary, cache):

| | original | compressed | saved |
|---|--:|--:|--:|
| input tokens | 58,518 | 44,379 | **24%** |
| output tokens | 27,588 | 7,504 | **73%** |
| **total tokens** | **86,106** | **51,883** | **40%** |
| **round-trip cost** | **$0.0167** | **$0.0090** | **46%** |

**~46% cost cut, quality mostly held or improved.** Output is where it pays off — output tokens drop 73% via output-control, and real workloads are output-heavy. (An earlier input-heavy/short-output mix saved only ~9% — the lever was real but had nothing to cut; representative corpora surface the true savings.)


## Frontier

| corpus | shape | n | input saved | output saved | cost saved | cache used | quality (orig→comp) | retention |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| `gsm8k` | Reasoning (CoT) | 7 | -46% | 94% | 25% | 0% | 43%→71% | +29pp |
| `humaneval` | Code generation | 12 | -15% | 29% | 27% | 0% | 100%→100% | +0pp |
| `dolly` | Generation (instruction) | 12 | 35% | 87% | 77% | 0% | 75%→58% | -17pp |
| `hotpotqa` | Multi-hop RAG | 12 | 50% | 14% | 5% | 0% | 60%→54% | -6pp |
| `glaive` | Agent / tool-use | 12 | 20% | 0% | 8% | 0% | 100%→100% | +0pp |
| `chat` | Multi-turn chat | 12 | 35% | 72% | 64% | 0% | 25%→33% | +8pp |
| `cnn` | Long-doc summary | 8 | 6% | 18% | 10% | 0% | 21%→22% | +1pp |
| `cache` | Prompt-cache reuse | 12 | 0% | 0% | -3% | 73% | 100%→100% | +0pp |

## Key findings

**Where compression wins** (cost cut, quality not significantly down):
- `gsm8k`: **cost −25%**, quality 43%→71% (+29±45pp).
- `humaneval`: **cost −27%**, quality 100%→100% (+0±0pp).
- `dolly`: **cost −77%**, quality 75%→58% (-17±33pp).
- `chat`: **cost −64%**, quality 25%→33% (+8±31pp).

**Within noise at this n** (negative but CI crosses zero — *not* confirmed regressions): `dolly` (-17±33pp). Scale n to resolve.

**The headline:** the per-stage **token gate guarantees fewer tokens, not preserved quality** — only this A/B quality axis catches the difference. The two regressions we confirmed and fixed were measured on **gpt-oss-20b** (a stronger model with tighter intervals): `code`'s compact-code output **−21.6±14.5pp** at n=37 → dropped from the preset; and `aggressive`+n-gram on `adult` **−100pp** (deterministic) → `ngram` now skips JSON records (recovers to 100%). On a weaker/noisier model the same levers mostly land inside their CIs — measure per model, and reserve lossy stages for inputs whose exact surface form the task doesn't depend on.


## What each row stresses

- **`gsm8k`** (Reasoning (CoT), preset `reasoning`, scorer `numeric-exact`) — stresses output draft / token-budget. Source: `openai/gsm8k`.
- **`humaneval`** (Code generation, preset `code`, scorer `pass@1 (runs unit tests)`) — stresses skeleton + minify. Source: `openai/openai_humaneval`.
- **`dolly`** (Generation (instruction), preset `aggressive`, scorer `LLM-judge`) — stresses output-control on long-form answers. Source: `databricks/databricks-dolly-15k`.
- **`hotpotqa`** (Multi-hop RAG, preset `rag`, scorer `token-F1`) — stresses retrieve (long context). Source: `hotpotqa/hotpot_qa`.
- **`glaive`** (Agent / tool-use, preset `agent`, scorer `tool-call match`) — stresses tool select / trim. Source: `glaiveai/glaive-function-calling-v2`.
- **`chat`** (Multi-turn chat, preset `aggressive`, scorer `LLM-judge`) — stresses output-control + dedup/cache on history. Source: `HuggingFaceH4/ultrachat_200k`.
- **`cnn`** (Long-doc summary, preset `aggressive`, scorer `token-F1`) — stresses output budget on long input. Source: `abisee/cnn_dailymail`.
- **`cache`** (Prompt-cache reuse, preset `cache`, scorer `numeric-exact`) — stresses stable shared prefix (Stage A). Source: `synthetic`.

## Reading the numbers honestly

- **No single compression %** — it is input-shape dependent. Long/structured inputs (RAG, record arrays, long docs) win on *input* tokens; short prompts (math, code stubs) can go *negative* on input because `output_control` injects a fixed instruction whose payoff is **output-side** (shorter answers), invisible in the input measure. Read **cost saved**, which captures both.
- **Cache used % is ~0 for one-shot diverse prompts** (nothing to cache-hit across distinct requests) and high only when a long prefix repeats — see `cache` (fixed system dossier + varying queries), the canonical agent/RAG-over-fixed-context shape.
- **Small n** — these runs use modest n for cost; CIs are reported in the JSON. Scale n for tighter intervals; several deltas here sit inside their CI (noise).
- **pass@1 actually executes** the model's code against the unit tests — the strongest signal here (no judge noise).


## Improvements driven by these results

The benchmark is actionable, not just descriptive — each row below is a code change the frontier forced:

- **`ngram` → prose-only guard.** `adult` 100%→0% (deterministic) traced to n-gram glossary abbreviation of JSON records → the model miscounts. Fix: `ngram` now skips any segment holding a JSON array of objects; abbreviates prose only. `adult` recovers to 100%.
- **`code` preset → dropped `output_compact_code`.** Confirmed real at n=37 (pass@1 −21.6pp, CI ±14.5, interval clear of zero). Minified-code *output* costs correctness on a small model; the −36% lever (arXiv:2508.13666) holds only via fine-tuning. Now opt-in.
- **`glaive` / `agent` preset → no change.** The −8pp at n=12 was **noise**: at n=39, retention is **+0.0pp** (CI ±5.2). Verifying before acting avoided a wrong fix.
- **New presets.** `reasoning` (Chain-of-Draft) — GSM8K +17pp, compression *improving* accuracy. `cache` (stable prefix + Stage A) — ~92% of input served from cache on fixed-prefix workloads.
- **Meta.** The per-stage **token gate guarantees fewer tokens, not preserved quality** — only this A/B quality axis catches `adult`/`humaneval`. Lossy stages are now bundled only where measured safe.


## Reproduce

```bash
python3 bench/scripts/download.py 40       # pull + normalize corpora (pinned in data/manifest.json)
bash    bench/scripts/run_all.sh           # live A/B across all corpora (needs OPENROUTER_API_KEY)
python3 bench/scripts/synth_readme.py      # regenerate this file
```

Per-stage ablation (offline, free): `llmtrim bench --corpus bench/data/<c>.jsonl --preset aggressive --ablate`.

