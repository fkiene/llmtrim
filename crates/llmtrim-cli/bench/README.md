# llmtrim benchmark

Two axes, measured live: **tokens saved** (real tokenizer, at compress time) and **quality retained** (the A/B delta between the model's answer on the *original* vs the *compressed* request). A preset is only honest if quality holds at its token saving: the frontier of (saved, retained) is the benchmark, not the saving alone.

- **Model:** `qwen/qwen3-next-80b-a3b-instruct` via OpenRouter (async-openai byot: the exact compressed body is sent, injected fields intact). A cheap, strong, **non-reasoning** instruct model: its visible output is the whole billed output, so prompt-side output shaping (terse / Chain-of-Draft) translates to real cost. Reasoning models bill hidden chain-of-thought as output that no prompt-side lever can cut, so the cost win shrinks there.
- **Scoring:** ground-truth where possible (numeric-exact for math, pass@1 that *runs the unit tests* for code), token-F1 for extractive QA, tool-call match for agents, an LLM judge only for open-ended shapes.
- **Cost:** priced from a pinned [models.dev](https://models.dev) snapshot (`bench/pricing.json`), cached input billed at the `cache_read` rate.
- **Cache used %:** share of compressed input served from the provider prompt cache (`usage.prompt_tokens_details.cached_tokens`).


## Bottom line

Across **112 A/B cases** on this real-usage mix (generation, chat, reasoning, code, RAG, agent, summary, cache):

| | original | compressed | saved |
|---|--:|--:|--:|
| input tokens | 71,031 | 49,062 | **31%** |
| output tokens | 25,843 | 6,628 | **74%** |
| **total tokens** | **96,874** | **55,690** | **43%** |
| **round-trip cost** | **$0.0365** | **$0.0126** | **66%** |
| **answer quality** | **78.9%** | **82.2%** | **+3.3pp** |

**~66% cost cut, quality up +3.3pp.** Output is where it pays off: output tokens drop 74% via output-control, and real workloads are output-heavy. The cost % rides on the model's output:input price ratio (≈12:1 here); the underlying token cuts are model-independent (−31% input, −74% output), projecting to −57% / −59% at GPT-4o / Claude Sonnet rates and −44% on the input-heavy gpt-oss-20b. (An earlier input-heavy/short-output mix saved only ~9%; the lever was real but had nothing to cut, and representative corpora surface the true savings.)


## Frontier

| corpus | shape | n | input saved | output saved | cost saved | cache used | quality (orig→comp) | retention |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| `gsm8k` | Reasoning (CoT) | 12 | -47% | 77% | 71% | 0% | 100%→92% | -8pp |
| `humaneval` | Code generation | 12 | -13% | 22% | 18% | 0% | 100%→100% | +0pp |
| `dolly` | Generation (instruction) | 12 | 35% | 91% | 89% | 0% | 75%→83% | +8pp |
| `hotpotqa` | Multi-hop RAG | 12 | 39% | 63% | 42% | 0% | 55%→76% | +21pp |
| `glaive` | Agent / tool-use | 12 | 19% | 0% | 5% | 0% | 100%→100% | +0pp |
| `chat` | Multi-turn chat | 12 | 38% | 75% | 71% | 0% | 25%→33% | +8pp |
| `cnn` | Long-doc summary | 8 | -3% | 12% | 7% | 0% | 22%→24% | +2pp |
| `cache` | Prompt-cache reuse | 12 | 0% | 0% | 6% | 17% | 100%→100% | +0pp |
| `toolout` | Tool output (mixed) | 10 | 84% | 93% | 88% | 0% | 100%→100% | +0pp |
| `tl_ours` | Tool output (logs) | 5 | 88% | 93% | 89% | 0% | 100%→100% | +0pp |
| `tl_rtk` | Tool output (grep) | 5 | -29% | 87% | 66% | 0% | 100%→100% | +0pp |

## Key findings

**Where compression wins big** (cost cut, quality held or up):
- Tool output (`toolout`/`tl_ours`/`tl_rtk`): **cost −66% to −89%**, quality 100%→100%, the cleanest win (logs/diffs/grep are mostly droppable noise).
- `dolly`: **cost −89%**, quality 75%→83% (+8pp).
- `chat`: **cost −71%**, quality 25%→33% (+8pp).
- `hotpotqa`: **cost −42%**, quality 55%→76% (+21pp); retrieval drops distractors the model was tripping on.

**Where it can't help** (nothing to cut): `glaive` (short tool-call output, cost −5%), `cache` (prefix already cached). `gsm8k` is the one quality dip (−8pp at n=12): the reasoning preset's Chain-of-Draft scaffold trades a small accuracy risk for −71% cost, so measure per workload before enabling.

**The headline:** the per-stage **token gate guarantees fewer tokens, not preserved quality**, and only this A/B quality axis catches the difference. The two regressions we confirmed and fixed were measured on **gpt-oss-20b** (a stronger model with tighter intervals): `code`'s compact-code output **−21.6±14.5pp** at n=37 → dropped from the preset; and `aggressive`+n-gram on `adult` **−100pp** (deterministic) → `ngram` now skips JSON records (recovers to 100%). On a weaker/noisier model the same levers mostly land inside their CIs, so measure per model, and reserve lossy stages for inputs whose exact surface form the task doesn't depend on.


## What each row stresses

- **`gsm8k`** (Reasoning (CoT), preset `reasoning`, scorer `numeric-exact`): stresses output draft / token-budget. Source: `openai/gsm8k`.
- **`humaneval`** (Code generation, preset `code`, scorer `pass@1 (runs unit tests)`): stresses skeleton + minify. Source: `openai/openai_humaneval`.
- **`dolly`** (Generation (instruction), preset `aggressive`, scorer `LLM-judge`): stresses output-control on long-form answers. Source: `databricks/databricks-dolly-15k`.
- **`hotpotqa`** (Multi-hop RAG, preset `rag`, scorer `token-F1`): stresses retrieve (long context). Source: `hotpotqa/hotpot_qa`.
- **`glaive`** (Agent / tool-use, preset `agent`, scorer `tool-call match`): stresses tool select / trim. Source: `glaiveai/glaive-function-calling-v2`.
- **`chat`** (Multi-turn chat, preset `aggressive`, scorer `LLM-judge`): stresses output-control + dedup/cache on history. Source: `HuggingFaceH4/ultrachat_200k`.
- **`cnn`** (Long-doc summary, preset `aggressive`, scorer `token-F1`): stresses output budget on long input. Source: `abisee/cnn_dailymail`.
- **`cache`** (Prompt-cache reuse, preset `cache`, scorer `numeric-exact`): stresses stable shared prefix (Stage A). Source: `synthetic`.

## Reading the numbers honestly

- **No single compression %:** it is input-shape dependent. Long/structured inputs (RAG, record arrays, long docs) win on *input* tokens; short prompts (math, code stubs) can go *negative* on input because `output_control` injects a fixed instruction whose payoff is **output-side** (shorter answers), invisible in the input measure. Read **cost saved**, which captures both.
- **Cache used % is ~0 for one-shot diverse prompts** (nothing to cache-hit across distinct requests) and high only when a long prefix repeats; see `cache` (fixed system dossier + varying queries), the canonical agent/RAG-over-fixed-context shape.
- **Small n:** these runs use modest n for cost; CIs are reported in the JSON. Scale n for tighter intervals; several deltas here sit inside their CI (noise).
- **pass@1 actually executes** the model's code against the unit tests, the strongest signal here (no judge noise).


## Improvements driven by these results

The benchmark is actionable, not just descriptive; each row below is a code change the frontier forced:

- **`ngram` → prose-only guard.** `adult` 100%→0% (deterministic) traced to n-gram glossary abbreviation of JSON records → the model miscounts. Fix: `ngram` now skips any segment holding a JSON array of objects; abbreviates prose only. `adult` recovers to 100%.
- **`code` preset → dropped `output_compact_code`.** Confirmed real at n=37 (pass@1 −21.6pp, CI ±14.5, interval clear of zero). Minified-code *output* costs correctness on a small model; the −36% lever (arXiv:2508.13666) holds only via fine-tuning. Now opt-in.
- **`glaive` / `agent` preset → no change.** The −8pp at n=12 was **noise**: at n=39, retention is **+0.0pp** (CI ±5.2). Verifying before acting avoided a wrong fix.
- **New presets.** `reasoning` (Chain-of-Draft): GSM8K +17pp, compression *improving* accuracy. `cache` (stable prefix + Stage A): ~92% of input served from cache on fixed-prefix workloads.
- **Meta.** The per-stage **token gate guarantees fewer tokens, not preserved quality**, and only this A/B quality axis catches `adult`/`humaneval`. Lossy stages are now bundled only where measured safe.


## Reproduce

```bash
python3 bench/scripts/download.py 40       # pull + normalize corpora (pinned in data/manifest.json)
cargo run -q --features live -- bench suite   # live A/B across all corpora (needs OPENROUTER_API_KEY)
python3 bench/scripts/synth_readme.py      # regenerate this file
```

`bench suite` runs the full corpus matrix in one process and writes one enveloped result
JSON per corpus to `bench/results/` (gitignored scratch). To keep a run as evidence, copy it
into `bench/snapshots/` (by date for the corpus frontier, `vs-<tool>` for head-to-heads). The
other axes share the same dispatcher:

```bash
llmtrim bench quality --corpus bench/data/<c>.jsonl --preset aggressive --ablate  # per-stage ablation (offline, free)
llmtrim bench agent                        # per-iteration token economics (dry-run; --live for real)
llmtrim bench latency                      # warm compress-path latency + attribution (offline)
llmtrim bench compare headroom             # head-to-head vs Headroom (dispatches the Python comparator)
```


## Head-to-head: Headroom

Both libraries run through their Python APIs on the same inputs and the same `o200k_base` denominator. Full tables (input saved, per-stage attribution, latency, and the live gpt-oss-20b output A/B) are in [snapshots/vs-headroom/README.md](snapshots/vs-headroom/README.md).

```bash
crates/llmtrim-uniffi/scripts/build-wheel.sh --release   # build the llmtrim wheel
pip install --user target/wheels/llmtrim-*.whl
pip install --user -r bench/scripts/requirements-vs-headroom.txt
llmtrim bench compare headroom                           # deterministic axes (offline)
llmtrim bench compare headroom --live -- --live-n 12     # output A/B (needs OPENROUTER_API_KEY)
```

`bench compare` is a thin dispatcher over the Python comparator (`vs_headroom.py` /
`caveman_ab.py`); it measures the real SDK path, so it still needs the wheel and the deps
above. Arguments after `--` pass straight through to the script.

