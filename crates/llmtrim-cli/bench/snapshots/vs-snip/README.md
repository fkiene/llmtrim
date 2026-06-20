# snip vs llmtrim (tool-output mode)

Tool: snip vdev | encoder: `o200k_base` | mode: BEST (native filter per tool-output type)

snip filters shell-command OUTPUT, not message arrays, so it is measured on real command output (not the prose corpora). Each native filter runs on the content type it was built for, scored on the same `o200k_base` encoder.

## Per-filter reduction

| Filter | In (tok) | Out (tok) | Reduction |
|--------|---------:|----------:|----------:|
| cargo-test | 973 | 15 | 98.5% |
| rspec | 173 | 78 | 54.9% |
| rails-routes | 434 | 15 | 96.5% |
| rails-migrate | 174 | 11 | 93.7% |
| bundle-install | 295 | 10 | 96.6% |
| ls | 770 | 424 | 44.9% |
| **total** | **2819** | **553** | **80.4%** |

## Caveats

- snip is lossy by design: cargo-test keeps the pass/fail tallies and drops the per-test detail, rails routes reports a count instead of the table. Reduction here is the token win, not a quality-matched comparison.
- Only non-injecting filters are measured. snip's injecting filters (git-log, git-diff, git-status, go-test) re-run the command with extra flags, so the pipeline sees a different stream than a static raw fixture; measuring them on raw output would distort the result, so they are out of scope rather than faked.
- The total is input-token-weighted, so the largest fixture dominates it. --limit keeps the CASES order, so a capped run reports the first N filters, not the largest.
- Scope is tool output only. snip has no filter for prose, file contents, or web-search results, so the prose corpora are out of scope rather than counted as 0%.
- All numbers are local and deterministic (snip run --, no network).
