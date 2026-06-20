# RTK vs llmtrim (tool-output mode)

Tool: rtk 0.42.4 | encoder: `o200k_base` | mode: BEST (native filter per tool-output type)

RTK rewrites shell-command OUTPUT, not message arrays, so it is measured on real command output (not the prose corpora). Each native filter runs on the content type it was built for, scored on the same `o200k_base` encoder.

## Per-filter reduction

| Filter | In (tok) | Out (tok) | Reduction |
|--------|---------:|----------:|----------:|
| pytest | 181 | 141 | 22.1% |
| grep | 640 | 159 | 75.2% |
| git-log | 185 | 73 | 60.5% |
| git-diff | 831 | 852 | -2.5% |
| **total** | **1837** | **1225** | **33.3%** |

## Caveats

- RTK is lossy by design: the pytest filter keeps the failures and drops the per-test pass detail, grep groups and truncates matches. Reduction here is the token win, not a quality-matched comparison.
- A small diff can grow rather than shrink: the git-diff filter adds a header, so on a two-line diff the output is larger than the input. That negative is real, not hidden.
- The total is input-token-weighted, so the largest fixture dominates it. `--limit` keeps the FILTERS order, so a capped run reports the first N filters, not the largest.
- Scope is tool output only. RTK has no filter for prose, file contents, or web-search results, so the prose corpora are out of scope rather than counted as 0%.
- All numbers are local and deterministic (`rtk pipe --filter`, no network).
