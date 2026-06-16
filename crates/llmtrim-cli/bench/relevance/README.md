# Relevance evaluation — lexical baseline vs a hypothetical semantic scorer

llmtrim's retrieval stage is **lexical only**: BM25+ against the live query, TextRank
centrality when there is no query, with RM3 pseudo-relevance feedback to widen sparse
queries (`crates/llmtrim-core/src/stages/retrieve.rs`). No model, no embeddings. That is
a spec scope rule, not an oversight: the proxy has a no-async, fast-startup budget, and an
embedding scorer means bundling an ONNX runtime plus a model file, which costs startup
latency and binary size on every request whether or not retrieval even fires.

The competitor Headroom ships ML/embedding semantic relevance routing. Before we spend the
budget to match it, we want a number: **how much better, if at all, would a semantic scorer
actually rank our retrieval candidates than the lexical stage we already have?** This
directory is the methodology and the golden set to answer that. It does not implement a
semantic scorer. The point is to decide whether implementing one is worth it.

## What we measure

A retrieval *case* is `(context, chunks, gold)`:

- `context` — the live conversational signal the stage ranks against. In production this is
  the final user turn plus short segments (the query anchor built in `retrieve.rs`); here it
  is a short string standing in for that need.
- `chunks` — the candidate segments competing to be kept (tool results, log sections, doc
  passages, prior turns). These are what the stage scores and prunes.
- `gold` — the indices of the chunks a correct retrieval must keep to answer the context.
  Curated by hand, not by either scorer, so neither approach is graded against its own bias.

The golden set lives in `golden.jsonl`, one case per line.

### Metrics

For a ranking that keeps the top `k` chunks:

- **precision@k** — of the `k` kept chunks, the fraction that are gold. Penalizes keeping
  noise (wasted budget).
- **recall@k** — of the gold chunks, the fraction kept. Penalizes dropping the answer,
  the failure mode that breaks the downstream task.
- **nDCG** — discounted cumulative gain over the full ranking, normalized to the ideal
  ranking. Rewards putting gold chunks *high*, which matters because the stage keeps a
  budgeted top slice and reorders into a head/tail U-shape.

Report each metric per case and as a mean over the set, with `k` set to the gold count of
each case (`k = |gold|`) so precision and recall are comparable. recall@k is the headline:
in this stage a dropped gold chunk is unrecoverable (chunks are elided by position, never a
hash), so recall failures are the real cost.

## The two scorers

### Lexical baseline (already implemented)

The exact production ranking from `retrieve.rs`: `bm25_rank` (BM25+ with the Lv & Zhai δ
floor, then one RM3 round on sparse or flat queries) when `context` is non-empty, else
`textrank_rank`. This is the number to beat. It is free at runtime — pure token statistics,
no model, no allocation beyond the chunks.

### Semantic scorer (hypothetical)

A stand-in for what we would build to match Headroom: embed `context` and each chunk with a
sentence-embedding model, rank chunks by cosine similarity to the context. For the
evaluation it can be a one-off offline script (any embedding model, run outside the proxy)
since we are measuring ranking quality, not runtime cost. We are **not** wiring it into the
crate; that is the build this evaluation decides for or against.

## How to run

This is a methodology and data scaffold. There is intentionally no Rust harness here —
wiring one in would touch shared modules (the retrieve stage, the bench crate) and conflict
with concurrent work. To evaluate:

1. Score each case in `golden.jsonl` with the lexical baseline. The production ranking is
   `bm25_rank(chunks, lex_words(context))` from `retrieve.rs`; an offline driver can call it
   directly, or reproduce BM25+/RM3 with the same parameters.
2. Score the same cases with a chosen embedding model (cosine of context vs each chunk).
3. Compute precision@k, recall@k, and nDCG per case (`k = |gold|`) and the means.
4. Report both rankings side by side per case so disagreements are inspectable, not just the
   aggregate.

## Decision criterion

Build the semantic scorer **only if** it beats the lexical baseline by a meaningful margin
on this set — concretely, a mean recall@k gain of at least 10 points that holds across the
case mix (not driven by one or two paraphrase-heavy cases), with no precision@k regression.

That bar is deliberately high because the win has to pay for its cost. A semantic scorer
adds an ONNX runtime and a model file to a tool whose whole pitch is no-async, fast startup,
and small binary, and retrieval is an opt-in stage that often does not fire. A few points of
recall do not justify that on every request. If the lexical baseline lands within the margin
— the expected outcome on agent-context retrieval, where the query and the relevant tool
output share concrete tokens (file paths, identifiers, error strings, command names) rather
than needing paraphrase matching — then lexical stays, and this directory is the evidence
for the no-embeddings decision rather than a step toward reversing it.

Where semantic ranking is known to help is paraphrase and synonym gaps — the context asks
about "the database connection" and the gold chunk says "Postgres pool timeout" with no
shared term. The golden set includes such cases on purpose, so the evaluation surfaces the
real ceiling of the lexical approach instead of a set rigged for it to win.
