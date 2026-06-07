#!/bin/bash
# Live A/B benchmark across all 7 corpora, each with a shape-matched preset.
# Writes per-corpus JSON to bench/results/ for README synthesis.
# Credentials: OPENROUTER_API_KEY (env or .env). Model+route: defaults (gpt-oss-20b / groq).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p bench/results

run() { # corpus preset n
  echo "=== $1 ($2, n=$3) ==="
  cargo run -q --features live -- bench --corpus "bench/data/$1.jsonl" --preset "$2" --n "$3" \
    --json-out "bench/results/$1.json" 2>&1 | tail -8 || echo "FAILED: $1"
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
