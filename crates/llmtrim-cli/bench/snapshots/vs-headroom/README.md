# llmtrim vs Headroom (matched-config, fair)

Both libraries are driven through their Python APIs (`llmtrim.compress`, `headroom.compress`). Before/after token counts use the **same** `o200k_base` encoder over the **same** message-content span. Latency is the median compress time over 5 runs (one-time model load excluded by a warm-up).

**This comparison is matched, not rigged.** Each tool is run at TWO points, and at each point llmtrim's preset is paired with a Headroom config of the same aggressiveness, not against Headroom's worst-case setting:

| point | llmtrim preset | Headroom config |
|---|---|---|
| moderate | `agent` | **defaults** (`compress_user_messages=False`, `protect_recent=4`, `target_ratio=None`, `min_tokens_to_compress=250`) |
| aggressive | `aggressive` | **max** (`compress_user_messages=True`, `protect_recent=0`, `target_ratio=0.2`, `min_tokens_to_compress=50`) |

Two corpora: **general** is the real golden corpora (gsm8k, hotpotqa, squad2) with their own ground-truth answers, the neutral quality signal. **tool-output** is llmtrim's own synthetic corpus (`synth_toolout.py`); its golds sit on lines llmtrim is built to keep, so it is llmtrim-favouring on quality and is reported separately.

## Point: moderate, llmtrim `agent` vs Headroom defaults

### general (n=24, real golden corpora, neutral signal)

| tool | tokens before→after | saved | per-case spread % | median ms |
|---|--:|--:|:--|--:|
| **llmtrim** | 14,407 → 14,407 | **0%** | 0 / 0 / 0 (σ 0) | 0.4 |
| Headroom | 14,407 → 14,407 | 0% | 0 / 0 / 0 (σ 0) | 0.3 |

### tool-output (n=15, llmtrim-authored synthetic, llmtrim-favouring)

| tool | tokens before→after | saved | per-case spread % | median ms |
|---|--:|--:|:--|--:|
| **llmtrim** | 11,748 → 8,349 | **29%** | 0 / 21 / 93 (σ 33) | 1.4 |
| Headroom | 11,748 → 6,123 | 48% | 0 / 45 / 96 (σ 41) | 6.4 |

Headroom's ML Kompress (ModernBERT) path fired on 0/15 tool-output cases at this point.

### Live quality A/B at point `moderate` (gpt-oss-20b)

Each case sent to `openai/gpt-oss-20b` three ways (original / llmtrim / Headroom); the answer is scored with the **corpus's own scorer** (numeric / token-F1 / contains). Faithful and adversarial cases are separated so neither tool's mean is distorted.

**faithful cases (n=10)**

| arm | answer accuracy | output tokens |
|---|--:|--:|
| original (uncompressed) | 90% | 1,936 |
| **llmtrim** | **90%** | 1,666 |
| Headroom | 80% | 2,859 |

**adversarial cases (n=3)**

| arm | answer accuracy | output tokens |
|---|--:|--:|
| original (uncompressed) | 100% | 188 |
| **llmtrim** | **100%** | 187 |
| Headroom | 33% | 268 |

<details><summary>Per-case quality (adv flagged)</summary>

| case | group | adv | scorer | original | llmtrim | Headroom |
|---|---|:-:|---|:-:|:-:|:-:|
| gsm8k-0 | general |  | numeric | OK | OK | OK |
| gsm8k-1 | general |  | numeric | OK | OK | OK |
| gsm8k-2 | general |  | numeric | OK | OK | OK |
| gsm8k-3 | general |  | numeric | OK | OK | OK |
| log-build-undef | tool-output |  | contains | OK | OK | OK |
| log-pytest-fail | tool-output |  | contains | OK | OK | OK |
| log-db-timeout | tool-output |  | contains | miss | OK | OK |
| log-rate-limit | tool-output |  | contains | OK | miss | miss |
| log-panic-file | tool-output |  | contains | OK | OK | miss |
| diff-signature | tool-output |  | contains | OK | OK | OK |
| adv-info-rowcount | tool-output | Y | contains | OK | OK | miss |
| adv-info-config | tool-output | Y | contains | OK | OK | miss |
| adv-diff-context-const | tool-output | Y | contains | OK | OK | OK |

</details>

## Point: aggressive, llmtrim `aggressive` vs Headroom max-aggression

### general (n=24, real golden corpora, neutral signal)

| tool | tokens before→after | saved | per-case spread % | median ms |
|---|--:|--:|:--|--:|
| **llmtrim** | 14,407 → 9,275 | **36%** | -51 / 3 / 58 (σ 34) | 1.6 |
| Headroom | 14,407 → 4,285 | 70% | 0 / 52 / 77 (σ 23) | 4.2 |

### tool-output (n=15, llmtrim-authored synthetic, llmtrim-favouring)

| tool | tokens before→after | saved | per-case spread % | median ms |
|---|--:|--:|:--|--:|
| **llmtrim** | 11,748 → 4,884 | **58%** | -1 / 75 / 91 (σ 35) | 3.1 |
| Headroom | 11,748 → 3,045 | 74% | 0 / 81 / 96 (σ 36) | 7.0 |

Headroom's ML Kompress (ModernBERT) path fired on 3/15 tool-output cases at this point.

### Live quality A/B at point `aggressive` (gpt-oss-20b)

Each case sent to `openai/gpt-oss-20b` three ways (original / llmtrim / Headroom); the answer is scored with the **corpus's own scorer** (numeric / token-F1 / contains). Faithful and adversarial cases are separated so neither tool's mean is distorted.

**faithful cases (n=10)**

| arm | answer accuracy | output tokens |
|---|--:|--:|
| original (uncompressed) | 100% | 1,612 |
| **llmtrim** | **100%** | 944 |
| Headroom | 50% | 3,981 |

**adversarial cases (n=3)**

| arm | answer accuracy | output tokens |
|---|--:|--:|
| original (uncompressed) | 100% | 189 |
| **llmtrim** | **100%** | 191 |
| Headroom | 33% | 315 |

<details><summary>Per-case quality (adv flagged)</summary>

| case | group | adv | scorer | original | llmtrim | Headroom |
|---|---|:-:|---|:-:|:-:|:-:|
| gsm8k-0 | general |  | numeric | OK | OK | miss |
| gsm8k-1 | general |  | numeric | OK | OK | OK |
| gsm8k-2 | general |  | numeric | OK | OK | miss |
| gsm8k-3 | general |  | numeric | OK | OK | OK |
| log-build-undef | tool-output |  | contains | OK | OK | OK |
| log-pytest-fail | tool-output |  | contains | OK | OK | miss |
| log-db-timeout | tool-output |  | contains | OK | OK | OK |
| log-rate-limit | tool-output |  | contains | OK | OK | miss |
| log-panic-file | tool-output |  | contains | OK | OK | miss |
| diff-signature | tool-output |  | contains | OK | OK | OK |
| adv-info-rowcount | tool-output | Y | contains | OK | OK | miss |
| adv-info-config | tool-output | Y | contains | OK | OK | miss |
| adv-diff-context-const | tool-output | Y | contains | OK | OK | OK |

</details>

## Verdict: who wins each axis

- **moderate / general tokens:** llmtrim 0% vs Headroom 0% → **tie**.
- **moderate / tool-output tokens:** llmtrim 29% vs Headroom 48% → **Headroom**.
- **moderate / faithful quality (n=10):** original 90%, llmtrim 90%, Headroom 80% → **llmtrim**.
- **moderate / adversarial quality (n=3):** original 100%, llmtrim 100%, Headroom 33% → **llmtrim**.
- **aggressive / general tokens:** llmtrim 36% vs Headroom 70% → **Headroom**.
- **aggressive / tool-output tokens:** llmtrim 58% vs Headroom 74% → **Headroom**.
- **aggressive / faithful quality (n=10):** original 100%, llmtrim 100%, Headroom 50% → **llmtrim**.
- **aggressive / adversarial quality (n=3):** original 100%, llmtrim 100%, Headroom 33% → **llmtrim**.

## Caveats (read these)

- **Matched configs, stated plainly.** At `moderate`, llmtrim runs its `agent` preset and Headroom runs its library DEFAULTS (which protect user messages and recent turns, so Headroom no-ops on many cases by design). At `aggressive`, llmtrim runs `aggressive` and Headroom runs its max config. Neither tool is pitted against the other's worst-case setting.
- **Corpus bias.** The `tool-output` group is llmtrim's own synthetic corpus; its golds sit on lines llmtrim keeps, so it flatters llmtrim on quality. Treat the `general` group (real golden corpora) as the less-biased quality signal.
- **Scorer.** Quality uses each corpus's own deterministic scorer (numeric / token-F1 / contains). We deliberately skip `judge` and `tool` cases (they need an LLM judge / call-arg parsing) so every number is a scorer this script actually computes. Token-F1 with a 0.5 OK threshold is lenient; read it as 'kept enough of the answer', not exact match.
- **Small n.** The live A/B is a budget sweep (a dozen-ish scored cases per point). Numbers are directional, not a significance test. Transient API errors (429/timeout) skip that case rather than abort the run.
- **Headroom's ML path.** Headroom runs with its `[ml]` extra enabled (ModernBERT Kompress + deterministic JSON/log/diff routers); no generative LLM call, model load excluded from latency.
- **Reproducibility.** Only the token-count axis (before/after/saved) is deterministic and citable. The `median ms` latency is machine-specific and the live `output tokens` are single-run, non-deterministic generations; read both as directional, not point estimates.
- Model is `openai/gpt-oss-20b` via the pinned `wandb/fp4` route (CLAUDE.md).

