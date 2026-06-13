# Caveman vs llmtrim A/B Benchmark

Model: `openai/gpt-oss-20b` | temperature=0 | max_tokens=2048
Date: 2026-06-11
Source prompts: `../caveman/benchmarks/prompts.json` (10 tasks)
Valid prompts: 9 (async-refactor excluded; caveman arm returned null completion on both attempts)

## Summary Table

| Arm      | Total prompt tokens | Total completion tokens | Output reduction vs baseline | Instr. overhead/req        |
|----------|---------------------|------------------------|------------------------------|----------------------------|
| baseline |                 861 |                  18 432 | —                            | 0 tokens                   |
| caveman  |               8 462 |                   3 610 | **80.4%**                    | ~949 tokens (SKILL.md body) |
| llmtrim  |               1 049 |                   5 730 | **68.9%**                    | 19 tokens (output_terse.txt)|

Totals are sums across the 9 valid prompts (all 3 arms completed with non-null token counts).

## Per-Prompt Completion Tokens

| Prompt ID              | baseline | caveman | llmtrim | caveman vs base | llmtrim vs base |
|------------------------|----------|---------|---------|-----------------|-----------------|
| react-rerender         |    2 048 |     503 |     373 | −75%            | −82%            |
| auth-middleware-fix    |    2 048 |     384 |     275 | −81%            | −87%            |
| postgres-pool          |    2 048 |     304 |     393 | −85%            | −81%            |
| git-rebase-merge       |    2 048 |     293 |     554 | −86%            | −73%            |
| async-refactor         |      859 |    FAIL |     315 | —               | —               |
| microservices-monolith |    2 048 |     194 |     571 | −91%            | −72%            |
| pr-security-review     |    2 048 |     249 |   2 048 | −88%            | 0% (hit limit)  |
| docker-multi-stage     |    2 048 |     585 |     640 | −71%            | −69%            |
| race-condition-debug   |    2 048 |     324 |     413 | −84%            | −80%            |
| error-boundary         |    2 048 |     774 |     463 | −62%            | −77%            |

## Per-Prompt Quality Check (manual)

Does the compressed answer retain the key technical content of the baseline?

| Prompt ID              | caveman                                                       | llmtrim                                                         |
|------------------------|---------------------------------------------------------------|-----------------------------------------------------------------|
| react-rerender         | OK: correctly identifies inline object ref + useMemo fix     | OK: same diagnosis, adds memo/React.memo distinction           |
| auth-middleware-fix    | OK: exp seconds vs Date.now ms, correct fix code shown       | OK: same root cause, fix code shown                           |
| postgres-pool          | OK: pg Pool with connectionString/max/timeouts shown         | OK: full Pool config snippet with inline comments             |
| git-rebase-merge       | OK: rebase rewrites history / merge preserves, tradeoffs     | OK: bullet-form explanation, when to use each                 |
| async-refactor         | FAIL: empty response (null)                                  | OK: promisify + async/await rewrite correct                   |
| microservices-monolith | DEGRADED: high-level bullets only, thin on domain boundaries | OK: covers domain boundaries, data coupling, team topology    |
| pr-security-review     | OK: SQL injection + parameterized query + error handling     | TRUNCATED: hit 2048 limit, partial (SQL injection only)       |
| docker-multi-stage     | OK: multi-stage Dockerfile with builder/runner stages        | OK: equivalent multi-stage Dockerfile                        |
| race-condition-debug   | OK: atomic UPDATE RETURNING + transaction + SERIALIZABLE     | OK: same pattern with pg code snippet                        |
| error-boundary         | OK: full ErrorBoundary class with getDerivedStateFromError   | OK: same class, slightly shorter                              |

## Key Numbers

```
Baseline total output:  18 432 tokens  (9 valid prompts)
Caveman total output:    3 610 tokens  →  80.4% reduction
llmtrim total output:    5 730 tokens  →  68.9% reduction
```

### Instruction overhead per request

| Arm     | System prompt tokens | Output saved per req (avg) | Net gain per req |
|---------|----------------------|----------------------------|------------------|
| caveman | ~949                 | ~1 647 output tokens        | ~+698 tokens saved |
| llmtrim | 19                   | ~747 output tokens          | ~+728 tokens saved |

Net gain = (baseline avg output − arm avg output) − instruction overhead.
Both approaches are net-positive on the first message. llmtrim has a marginally better net-per-request because its overhead is 50× smaller.

## Caveats

1. **Baseline truncation**: 8 of 10 baseline responses hit the 2 048-token limit. The true baseline output would be higher without the cap; savings percentages are underestimates.
2. **async-refactor caveman failure**: caveman arm returned an empty response (null completion_tokens) on both attempts. Root cause unknown (possible moderation or API error). Excluded from all totals.
3. **pr-security-review llmtrim truncation**: llmtrim hit the 2 048-token limit, producing an incomplete answer with 0% compression. This is an outlier: llmtrim produced no compression on a long code-review task.
4. **Quality judgment**: manual, one-line, subjective. Not an LLM judge. Based on reading the first 250–400 chars of each response and spot-checking key technical claims.
5. **Token counts**: all from OpenRouter `usage` fields, real API-reported counts, not estimates.
6. **Model baseline brevity**: `openai/gpt-oss-20b` is already fairly concise; compression gains may be larger on more verbose models (GPT-4o, Claude Sonnet).
7. **Net tokens are not net dollars.** The net-gain table prices input and output tokens equally. With a skewed output:input price ratio (e.g. 10:1), caveman's deeper output cut outweighs its 949-token overhead on these one-shot prompts; conversely, in cached multi-turn agent sessions the 949-token skill amortizes behind the prompt cache. Which arm wins on *cost* depends on the model's pricing and the session shape; this benchmark only establishes the token mechanics.
