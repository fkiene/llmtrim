# Design spike: ML/embedding semantic relevance routing

Status: spike, no decision committed.
Question: should llmtrim add machine-learned semantic relevance scoring (text
embeddings) to its retrieval stage to close the quality gap with Headroom, or stay
lexical?

## Summary

llmtrim's relevance routing is lexical-only today: BM25+ with RM3 query expansion when
there is a query, TextRank centrality when there isn't, feeding a submodular
budget-aware selector. Headroom adds a learned semantic layer (small text embeddings via
an ONNX runtime in its Rust port, ModernBERT in its Python package) and falls back to
BM25 when that model is not loaded. The pull toward parity is real, but an embedding
model collides head-on with two of this repo's hard constraints: no async and a fast,
small binary. The recommendation is to not bundle an ONNX runtime now. Spend the next
increment on stronger lexical retrieval (already partly built), and treat an
optional, feature-gated semantic scorer as a later step that has to earn its place on a
fixed relevance eval before any code lands.

## Current state: lexical-only, by design

The retrieval stage is `crates/llmtrim-core/src/stages/retrieve.rs`. Its module header
states the constraint outright: "Pure lexical: no model, no embeddings (spec scope rule
1)." Two more comments in the file repeat "no model, no embeddings." This is a
deliberate scope boundary, not an oversight.

What the stage actually does, verified against the source:

- Role-aware segmentation. It only prunes bulk *context*, never the system instruction
  or the final user turn (the live question), and never the inside of harness directive
  blocks such as `<system-reminder>`. Roles come from the provider seam (`role_at`,
  `turn_index`), so this works across `/messages`, Google `/contents`, and OpenAI
  Responses `/input`, not just one wire shape.
- Chunking. Paragraphs by blank line; for unstructured prose, TextTiling lexical-cohesion
  boundaries (Hearst 1997, depth scoring after Eisenstein and Barzilay 2008); otherwise
  lines or a single blob. Optional sentence granularity (DSLR, arXiv:2407.03627).
- Ranking. With a query: BM25+ with the Lv and Zhai delta lower bound (`BM25_PLUS_DELTA
  = 1.0`), plus one optional RM3 pseudo-relevance-feedback round when the query is sparse
  or the ranking is flat (Lavrenko and Croft 2001). Without a query: TextRank centrality
  over a lexical-overlap graph, with an O(n) head-plus-tail fallback above
  `TEXTRANK_MAX_CHUNKS = 2000` so a huge log can't blow up the dense matrix.
- Language awareness. BM25 stemmer and stopwords follow the whatlang-detected language;
  tokenization is Unicode-segmented (UAX#29) with bm25 normalization disabled so
  non-Latin scripts are not transliterated. Not English-only.
- Failure protection. Chunks carrying failure signals (the toolout STRONG set: `error`,
  `panic`, TAP `not ok`, and their indented continuations) survive regardless of query
  overlap. A test failure rarely shares words with "run the tests," so pure relevance
  ranking would elide exactly the lines the agent needs.

Selection is `crates/llmtrim-core/src/select.rs` (note: at the crate root, not under
`stages/`). It is a token-budgeted monotone-submodular selector: modular relevance
blended with Lin and Bilmes saturating bigram coverage (ACL 2011), maximized by a lazy
CELF cost-ratio greedy under a knapsack constraint. Coverage is a `bigram ->
covered-weight` map, never an n-by-n similarity matrix, so memory stays linear.
Deterministic: ties break by original index. The ranker hands it relevance scores; it
decides what fits the budget without re-covering near-duplicates.

So "relevance routing" already exists. It is lexical end to end: term overlap, corpus
statistics, and bigram coverage. There is no semantic component, no model file, no vector
index.

Quality is checked offline by recall@k unit tests (for example
`bm25_recall_at_k_finds_the_answer_chunk` in retrieve.rs), not yet by a live quality
gate. The token gate reverts the stage whenever it fails to cut tokens.

## Competitor: Headroom's semantic layer

Headroom scores relevance with a learned model rather than term overlap:

- Rust port: small text embeddings (a bge-small-class model in the fastembed family) plus
  magika for content typing, both run through the `ort` ONNX Runtime bindings. Relevance
  becomes cosine similarity in embedding space, which catches paraphrase and synonymy
  that BM25 misses ("auth failure" vs "login error").
- Python package: ModernBERT embeddings for the same role.
- Fallback: when the model is not loaded, it drops back to BM25, the same lexical floor
  llmtrim already sits on.

The plain-language gap: a semantic scorer keeps a chunk that answers the question in
*different words*. Lexical retrieval (even BM25+ with RM3 expansion) only bridges that gap
when expansion happens to surface the right term from the corpus. On paraphrase-heavy
context this is a genuine quality difference, and it is the case for adding embeddings.

## The hard constraint

This repo's rules push directly against bundling an embedding model. From
`.claude/rules/rust-patterns.md`, rule 1: "No async. Zero tokio, async-std, futures.
Single-threaded by design. Async adds 5-10ms startup." CLAUDE.md frames llmtrim as a
local MITM proxy on the request hot path, and the testing rules target sub-10ms startup,
under-5MB memory, and an under-5MB binary.

An ONNX-runtime path conflicts with all of that:

- Binary and dependency size. Linking `ort` pulls in ONNX Runtime (native library, tens
  of MB) and a tokenizer and model-loading stack. That alone can dwarf an under-5MB
  binary target, and it widens the dependency and license surface (the repo is
  AGPL-3.0-only and tracks dependency compatibility).
- Model weights. bge-small is on the order of 100MB+; some embedding models are several
  hundred MB. Either it ships in the artifact (huge) or it downloads on first use
  (network dependency, cold-start latency, a failure mode on the proxy hot path, and a
  cache directory to manage).
- Startup and latency. Loading a model and running inference is the opposite of a sub-10ms
  cold start. Even a warm model adds per-request inference latency to a proxy that sits
  inline.
- Async pressure. ONNX inference and model download want a thread pool or async I/O. The
  no-async rule means any such work has to be carefully boxed onto blocking calls, fighting
  the grain of the runtimes those libraries assume.

This is not "embeddings are bad." It is that embeddings as a *bundled, always-on* path
break the product's stated shape (fast, small, inline, deterministic). Headroom made a
different shape choice; copying its scorer without copying its size and latency budget
would regress llmtrim's reason to exist.

## Options

### Option 1: do nothing, stay lexical

Keep BM25+/RM3/TextRank plus submodular selection. Zero size, latency, or async cost.
Stays deterministic and language-universal.

Tradeoff: gives up the paraphrase and synonymy recall that semantic similarity buys. On
context where the answer is worded unlike the question, lexical retrieval can drop the
answer-bearing chunk. We do not currently know how often that happens on real traffic,
because there is no relevance eval measuring it (see Open questions).

### Option 2: optional, feature-gated ONNX semantic scorer with BM25 fallback

A `semantic` (or `embeddings`) Cargo feature, off by default. When enabled and a model is
present, rank by embedding cosine similarity; otherwise fall back to the existing BM25+
path, mirroring Headroom's fallback. The default build stays lexical, small, and fast; only
opted-in users pay the size and latency cost.

Tradeoffs:

- Honest cost even when gated: a real second code path to test, a new heavy optional
  dependency, model acquisition and caching, and the async or blocking-inference question
  to solve cleanly under the no-async rule.
- "Optional" erodes over time. If the semantic path becomes the recommended one, the
  default build's small-and-fast promise quietly stops being what users run.
- Determinism. Embedding inference can vary across hardware and ONNX versions; the rest of
  the pipeline is bit-deterministic and the selector relies on stable ranking. This needs
  pinning and testing.
- The win is unquantified until the eval in Option 3's gate exists.

### Option 3: lighter middle ground, stronger lexical (recommended near-term)

Close part of the gap without a model:

- RM3 is already implemented; tune when it fires and how aggressively it expands, measured
  on the eval rather than by intuition.
- Query expansion beyond RM3: lightweight morphological or co-occurrence expansion to reach
  near-synonyms that share a stem or commonly co-occur.
- Bigram and phrase coverage is already in the selector (Lin-Bilmes). Feeding richer
  features (selective trigrams, proximity terms) into ranking, not just selection, is cheap
  and stays lexical.
- Learned-sparse retrieval (SPLADE-style term weighting) sits between BM25 and dense
  embeddings: it captures some semantic expansion while still producing a sparse,
  inspectable, deterministic term vector and a much smaller model than dense embeddings.
  It is a middle option worth evaluating *if* the eval shows lexical is leaving real recall
  on the table; it still carries a model and the same size and async questions in lighter
  form, so it is not free.

Tradeoff: less ceiling than true embeddings. It will not match dense semantic similarity on
heavy paraphrase. But it keeps every property the product depends on (size, speed,
determinism, no async, language-universal) and almost certainly recovers a meaningful
fraction of the gap for a fraction of the cost.

## Recommendation

Do Option 3 now, hold Option 2 behind evidence, do not do a bundled always-on embedding
path.

Concretely:

1. Build the relevance eval first. The recall@k unit tests are a start but they are
   hand-built fixtures. Stand up a fixed golden relevance set (query plus context plus
   known answer-bearing chunks) and a recall and answer-retention metric, in the spirit of
   the existing agent golden tasks under `crates/llmtrim-cli/bench/agent/` and the
   evidence-over-anecdote rule in CLAUDE.md. This eval is the gate for everything below.
2. Measure today's lexical stage on it. Establish the baseline recall and answer-retention.
3. Tune and extend the lexical path (Option 3) and re-measure. Keep what moves the metric.
4. Only if a real gap remains after that, prototype Option 2 behind a default-off feature
   and require it to beat the tuned lexical baseline on the same eval by a margin that
   justifies the size, latency, determinism, and async cost. If it does not clear that bar,
   it does not ship.

The recommendation depends entirely on that eval. Without it, "embeddings would help" is an
assumption, exactly the kind CLAUDE.md tells us not to act on from a single anecdote.

## Open questions and decision criteria

- How often does lexical retrieval actually drop the answer on real traffic? Unknown until
  the eval exists. This is the single most important number; it decides whether any
  semantic work is worth starting.
- What is the per-request latency budget for the proxy on the hot path? An embedding
  forward pass has to fit inside it, warm, or it is a non-starter regardless of recall.
- What binary-size and dependency ceiling is acceptable for an *opt-in* build? Even gated,
  `ort` plus a model is a large surface; AGPL and dependency-compat tracking apply.
- Can embedding inference be made deterministic enough (pinned model, pinned ONNX, fixed
  precision) that the downstream selector stays reproducible?
- How is the model acquired and cached without violating no-async and without a network
  dependency on the request path?
- Does learned-sparse (Option 3's SPLADE variant) capture most of the semantic win at a
  fraction of the size? If yes, it may dominate dense embeddings for this product's shape.

Decision rule: ship a semantic path only when the relevance eval shows it beats the tuned
lexical baseline by a margin that pays for its size, latency, and complexity, and only as a
default-off option. Until then, stay lexical and make the lexical path as strong as the
evidence rewards.
