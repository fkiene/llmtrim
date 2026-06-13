#!/bin/bash
# Live A/B benchmark across all 8 corpora, each with a shape-matched preset.
# Writes per-corpus JSON to bench/results/ for README synthesis.
# Credentials: OPENROUTER_API_KEY (env or .env). Model+route: defaults (gpt-oss-20b / groq).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p bench/results

# The A/B compares llmtrim's IN-PROCESS compression against the true original request.
# If the llmtrim daemon proxy is in the environment (`llmtrim setup` exports HTTPS_PROXY),
# both arms get re-compressed in flight and the baseline arm is no longer original —
# contaminating every number. Always bypass any proxy for the live calls.
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy

run() { # corpus preset n
  echo "=== $1 ($2, n=$3) ==="
  # Keep the summary panel short but never swallow per-case diagnostics: a `tail -8`
  # alone hid the `FAIL <case> …: <error>` and judge-failure lines that explain a
  # failed/zeroed corpus.
  out=$(cargo run -q --features live -- bench --corpus "bench/data/$1.jsonl" --preset "$2" --n "$3" \
    --json-out "bench/results/$1.json" 2>&1) || echo "FAILED: $1"
  printf '%s\n' "$out" | grep -E '^  FAIL|judge call failed' || true
  printf '%s\n' "$out" | tail -8
  echo
}

run gsm8k     reasoning  12   # reasoning   → Chain-of-Draft (bench: +17pp)
run humaneval code       12   # code gen    → skeleton/minify (compact-code dropped)
run dolly     aggressive 12   # generation  → output-control cuts long-form answers (judge)
run hotpotqa  rag        12   # multi-hop   → retrieve (long ctx)
run glaive    agent      12   # tool        → tool select/trim
run chat      aggressive 12   # multi-turn  → output-control + dedup/cache on history (judge)
run cnn       aggressive 8    # long doc    → output budget
run cache     cache      12   # shared prefix → cache-first preset (Stage A)
echo "ALL DONE"
