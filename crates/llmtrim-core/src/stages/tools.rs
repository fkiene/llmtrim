//! Stage G — tool layer: static tool selection + schema-description trimming. Opt-in.
//!
//! Tool/function schemas are resent every request and are often the largest hidden
//! input cost in agent loops. Static tool selection keeps only the tools whose
//! name/description lexically overlaps the conversation (keyword match, no model),
//! dropping the rest; description trimming caps verbose descriptions. Lossy — a
//! dropped or trimmed tool may be the one the model needed — so off by default and
//! InputTokens-gated. (Tool-output hygiene — collapsing repeated log lines in tool
//! results — is handled by Stage E dedup.)
//!
//! Stopwords (which prevent spurious overlap like "the" matching a SQL tool) come
//! from the `stop-words` crate, for the language `whatlang` detects in the request —
//! not a hardcoded English list.

use std::collections::HashSet;

use anyhow::Result;
use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct ToolStage {
    pub select: bool,
    pub trim_desc: bool,
    pub minify_schema: bool,
    pub max_desc_chars: usize,
}

impl Transform for ToolStage {
    fn name(&self) -> &str {
        "tools"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn scope(&self) -> crate::gate::Scope {
        crate::gate::Scope::Tools // selects/trims tool schemas; content text untouched
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        if self.select {
            select_tools(req, provider);
        }
        // Minify each surviving tool's parameter schema before the description trim, so the
        // trim's char cap also lands on any per-property descriptions the minifier kept.
        if self.minify_schema {
            minify_tool_schemas(req, self.max_desc_chars);
        }
        if self.trim_desc {
            provider.truncate_tool_descriptions(req, self.max_desc_chars);
        }
        Ok(())
    }
}

/// Keys a tool's parameter JSON Schema sits under directly (after descending into an OpenAI
/// Chat `function` wrapper, handled in [`minify_tool_schemas`]): Anthropic `input_schema`,
/// OpenAI Chat/Responses `parameters`. Tried in order; both are checked since a malformed tool
/// could carry either.
const SCHEMA_KEYS: [&str; 2] = ["input_schema", "parameters"];

/// Apply the API-safe schema minifier ([`tool_schema::minify_schema`]) to every tool's
/// parameter schema, in place. `tools` is a top-level array in all wire shapes; per element the
/// schema is under `input_schema` / `parameters` (Anthropic, OpenAI Responses), nested under
/// `function.parameters` (OpenAI Chat), or — for Gemini — one level deeper under each
/// `functionDeclarations[].parameters`. Tools without an object schema are skipped. Kept
/// provider-agnostic so no new trait method is needed (matches `tools_used_in_history`).
fn minify_tool_schemas(req: &mut Request, max_desc_chars: usize) {
    let Some(Value::Array(tools)) = req.raw_mut().get_mut("tools") else {
        return;
    };
    for tool in tools.iter_mut() {
        // Gemini groups callables under `functionDeclarations`; each carries its own schema.
        if let Some(decls) = tool
            .get_mut("functionDeclarations")
            .and_then(Value::as_array_mut)
        {
            for d in decls.iter_mut() {
                minify_schema_at(d, max_desc_chars);
            }
            continue;
        }
        // OpenAI Chat nests the callable under `function`; Responses/Anthropic are flat.
        let scope = match tool.get_mut("function").filter(|f| f.is_object()) {
            Some(f) => f,
            None => tool,
        };
        minify_schema_at(scope, max_desc_chars);
    }
}

/// Minify whichever parameter-schema field (`input_schema` / `parameters`) is present on a
/// single tool/declaration object.
fn minify_schema_at(scope: &mut Value, max_desc_chars: usize) {
    let Some(obj) = scope.as_object_mut() else {
        return;
    };
    for key in SCHEMA_KEYS {
        if let Some(schema) = obj.get_mut(key) {
            crate::stages::tool_schema::minify_schema(schema, max_desc_chars);
        }
    }
}

/// The reliably-detected language of `sample` (whatlang), or `None` when detection
/// is unreliable or absent. The single language-detection seam in the crate — Stage B
/// retrieval (BM25 + pruning stopwords) and tool selection all route through it, so
/// "what language is this" is decided in exactly one place.
pub(crate) fn detect_lang(sample: &str) -> Option<whatlang::Lang> {
    whatlang::detect(sample)
        .filter(|info| info.is_reliable())
        .map(|info| info.lang())
}

/// Stopwords for the language detected in `sample` (NLTK/ISO lists via the
/// `stop-words` crate), falling back to English when detection is unreliable or the
/// language isn't in our supported map. The map is enum→enum glue; the word lists
/// themselves come from the crate. Shared with Stage B sentence pruning.
pub(crate) fn stopword_set(sample: &str) -> HashSet<&'static str> {
    use stop_words::LANGUAGE as L;
    use whatlang::Lang;
    // Detect on a leading slice of a large segment (see LANG_DETECT_MAX_BYTES): matches
    // whole-text detection for monolingual inputs while avoiding a multi-KB rescan.
    // Char-boundary-safe.
    let head = if sample.len() > LANG_DETECT_MAX_BYTES {
        let mut end = LANG_DETECT_MAX_BYTES;
        while !sample.is_char_boundary(end) {
            end -= 1;
        }
        &sample[..end]
    } else {
        sample
    };
    let language = match detect_lang(head) {
        Some(Lang::Fra) => L::French,
        Some(Lang::Spa) => L::Spanish,
        Some(Lang::Deu) => L::German,
        Some(Lang::Ita) => L::Italian,
        Some(Lang::Por) => L::Portuguese,
        Some(Lang::Nld) => L::Dutch,
        Some(Lang::Rus) => L::Russian,
        Some(Lang::Jpn) => L::Japanese,
        Some(Lang::Kor) => L::Korean,
        Some(Lang::Cmn) => L::Chinese,
        Some(Lang::Ara) => L::Arabic,
        Some(Lang::Tur) => L::Turkish,
        Some(Lang::Pol) => L::Polish,
        Some(Lang::Swe) => L::Swedish,
        Some(Lang::Dan) => L::Danish,
        Some(Lang::Fin) => L::Finnish,
        Some(Lang::Ell) => L::Greek,
        Some(Lang::Hun) => L::Hungarian,
        Some(Lang::Ron) => L::Romanian,
        Some(Lang::Ces) => L::Czech,
        Some(Lang::Ukr) => L::Ukrainian,
        Some(Lang::Vie) => L::Vietnamese,
        Some(Lang::Ind) => L::Indonesian,
        Some(Lang::Hin) => L::Hindi,
        // Any other (or undetected) language falls back to English (graceful, never panics).
        _ => L::English,
    };
    stop_words::get(language).iter().copied().collect()
}

/// Lowercased lexical tokens via the Unicode word segmenter (UAX#29) — works across
/// scripts (CJK, Cyrillic, …) rather than an ASCII `is_alphanumeric` split, which
/// would collapse a space-less script into one token. The shared tokenizer for Stage
/// B retrieval ranking, Stage E SimHash dedup, and tool selection.
pub(crate) fn lex_words(s: &str) -> Vec<String> {
    s.unicode_words().map(str::to_lowercase).collect()
}

/// 64-bit FNV-1a hash. Unlike `DefaultHasher`, output is **stable across Rust versions
/// and platforms** — safe to compare across restarts (cache fingerprints, bigram sets).
/// Non-cryptographic; used only for equality checks. Takes any byte iterator so hot
/// callers (per-pair bigram hashing in sizing) can chain without an intermediate `Vec`.
pub(crate) fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Content words of `lower` (already lowercased) as a set of **borrowed** slices —
/// Unicode-segmented (universal), snake_case split (`run_sql` → `run`, `sql`), stopwords +
/// single chars dropped. Borrows from `lower`, so no per-word allocation.
fn content_words<'a>(lower: &'a str, stop: &HashSet<&str>) -> HashSet<&'a str> {
    lower
        .unicode_words()
        .flat_map(|w| w.split('_'))
        .filter(|w| w.len() >= 2 && !stop.contains(w))
        .collect()
}

/// Enough leading content to detect the language (whatlang needs a sample, not the whole
/// prompt) — bounded so we never join tens of KB of context just to pick a stopword list.
const LANG_SAMPLE_BYTES: usize = 2048;

/// Cap for language detection on a large segment. `whatlang` is O(input) and its verdict
/// stabilizes within a few KB, so above this we detect on a leading slice rather than
/// rescanning tens of KB (Stage B sentence pruning on big RAG contexts — ~5ms on a 200KB
/// request). 8 KB is generous and representative, so the detected language — hence the
/// stopword set — matches whole-text detection for any monolingual input.
const LANG_DETECT_MAX_BYTES: usize = 8 * 1024;

/// Cap on the (recent) content scanned to build the tool-selection query word-set. Lowercasing
/// and word-segmenting the whole resent prompt every call dominated this stage (~15ms on a
/// 120K request); tool relevance tracks the current task, so a bounded slice of the newest
/// content suffices (already-invoked tools are protected separately).
const TOOL_QUERY_MAX_BYTES: usize = 16 * 1024;

/// BM25F field weights (ToolRegistry, arXiv:2507.10593, 2025): a tool's NAME is the strongest
/// relevance signal — a query naming the operation it wants (`weather`, `sql`) should match the
/// tool called that far above one that merely mentions it in prose — so the name field is boosted
/// well over description. Parameter property names are load-bearing too (`city`, `query`), between
/// name and description. Description text is the weak field (advisory, often boilerplate), weight
/// 1. Name ×4 sits mid-range of ToolRegistry's 3–5× name boost.
const FIELD_W_NAME: f64 = 4.0;
const FIELD_W_PARAMS: f64 = 2.0;
const FIELD_W_DESC: f64 = 1.0;

/// BM25 saturation `k1` and length-normalization `b` — the standard defaults (Robertson &
/// Zaragoza, "The Probabilistic Relevance Framework", 2009). Tool documents are short and
/// uniform, so the exact constants matter little; these are the conventional, well-understood
/// values.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// Keep only tools that BM25F-rank as relevant to the conversation, plus any already invoked.
/// Safety: if nothing scores, keep all tools (never strip the whole toolset on a weak query).
///
/// Upgrades the former lexical keyword-overlap test to fielded BM25 (BM25F, ToolRegistry
/// arXiv:2507.10593): each tool is a 3-field document — name (boosted), parameter property
/// names, description — scored against the same conversation-derived query as before. Fielding
/// lets a query that names a tool outrank one that only mentions the term in a long description,
/// which flat overlap (and flat BM25) cannot distinguish.
fn select_tools(req: &mut Request, provider: &dyn Provider) {
    // Cache-safety gate (issue #9): prune only on the FIRST turn of a conversation. The kept
    // tool subset depends on the conversation, so it changes as an agent loop grows — and
    // providers fold the `tools[]` block into the cached prompt prefix, so a changing block
    // busts the prefix every turn (cache reads collapse, the prefix is rebilled as fresh input,
    // costing more than the few schema tokens pruning saved). On the first turn there is no
    // prior prefix to bust, so pruning is a free saving; from the second turn on the block must
    // stay byte-stable, so we leave it intact and rely on the deterministic trim/minify stages
    // to shrink it without churning it. (This also keeps `aggressive`, which likewise enables
    // the cache stage, from busting its own prefix.)
    if !is_first_turn(req) {
        return;
    }

    let descriptors = provider.tool_descriptors(req);
    if descriptors.len() < 2 {
        return; // nothing meaningful to prune
    }

    let pointers = provider.content_text_pointers(req);
    // Build the query text from the most-recent content only, bounded by `TOOL_QUERY_MAX_BYTES`:
    // scanning newest-first and stopping at the cap keeps this O(cap) instead of O(whole resent
    // prompt), the stage's former dominant cost. Already-invoked tools are kept regardless
    // (below), so bounding the scan can't dangle a `tool_use`.
    let mut lower = String::new();
    for p in pointers.iter().rev() {
        if let Some(s) = req.get_str(p) {
            lower.push_str(&s.to_lowercase());
            lower.push(' ');
            if lower.len() >= TOOL_QUERY_MAX_BYTES {
                break;
            }
        }
    }
    let sample_end = lower.len().min(LANG_SAMPLE_BYTES);
    let stop = stopword_set(lower.get(..sample_end).unwrap_or(&lower));
    let query = content_words(&lower, &stop);
    if query.is_empty() {
        return;
    }

    // Per-tool fielded documents (aligned to `descriptors` / the tools array order). Property
    // names come from the raw schema; provider-agnostic, like `tools_used_in_history`.
    let param_fields = tool_param_words(req, &stop);
    let docs: Vec<ToolDoc> = descriptors
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| ToolDoc {
            name: bag(&content_words(&name.to_lowercase(), &stop)),
            params: param_fields.get(i).cloned().unwrap_or_default(),
            desc: bag(&content_words(&desc.to_lowercase(), &stop)),
        })
        .collect();
    let scores = bm25f_scores(&docs, &query);

    // Explicit-mention rail: a tool whose exact name appears as a standalone token anywhere in
    // the (first-turn) content text is kept regardless of score. This is a case-SENSITIVE
    // exact-name match (`contains_standalone`), so it catches a tool the lowercased BM25F query
    // tokenizes away — e.g. an underscore-joined `mcp__server__thing`, or a name referenced only
    // by an instruction ("use ToolSearch before calling deferred tools") whose split doesn't
    // score the tool. A false keep costs a few schema tokens; a false drop breaks a tool call,
    // so bias to keep. (The former "keep already-invoked tools" rail is gone: selection now only
    // runs on the first turn — see the `is_first_turn` gate — where no tool has been invoked yet.)
    let mut mentioned: HashSet<&str> = HashSet::new();
    for p in pointers.iter() {
        if mentioned.len() == descriptors.len() {
            break;
        }
        if let Some(s) = req.get_str(p) {
            for (name, _) in descriptors.iter() {
                if !mentioned.contains(name.as_str()) && contains_standalone(s, name) {
                    mentioned.insert(name.as_str());
                }
            }
        }
    }

    let keep: Vec<bool> = descriptors
        .iter()
        .zip(&scores)
        .map(|((name, _), &s)| mentioned.contains(name.as_str()) || s > 0.0)
        .collect();
    if keep.iter().any(|&k| k) {
        provider.retain_tools(req, &keep);
    }
}

/// True if `name` occurs in `text` as a standalone token: an exact, **case-sensitive** match
/// whose neighbors are not identifier characters (Unicode-aware `char::is_alphanumeric`, plus
/// `_`, since tool names like `mcp__server__thing` are underscore-joined identifiers). Tool
/// names are literal identifiers (`ToolSearch`), so case-sensitivity is deliberate — it keeps a
/// generic prose word ("run", "search") from accidentally pinning a same-named tool, at the
/// accepted cost of missing a miscased mention. Single forward scan over `find` hits; no regex.
fn contains_standalone(text: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(i) = text[from..].find(name) {
        let start = from + i;
        let end = start + name.len();
        let before_ok = text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_ident(c));
        let after_ok = text[end..].chars().next().is_none_or(|c| !is_ident(c));
        if before_ok && after_ok {
            return true;
        }
        // Advance past the first char of this hit (char-boundary-safe) to find later hits.
        from = start + name.chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// A tool as a 3-field bag-of-content-words document for BM25F: tokenized name, parameter
/// property names, and description (each a `(term, term-frequency)` list). Built with the same
/// `content_words` tokenization as the query (Unicode-segmented, snake_case-split, stopwords
/// dropped) so terms live in one space.
#[derive(Default)]
struct ToolDoc {
    name: Vec<(String, u32)>,
    params: Vec<(String, u32)>,
    desc: Vec<(String, u32)>,
}

/// Collapse a content-word set into a `(term, count=1)` bag. The name/description fields are
/// drawn from a `HashSet`, so each term appears once; BM25F still weights them by field.
fn bag(words: &HashSet<&str>) -> Vec<(String, u32)> {
    words.iter().map(|w| (w.to_string(), 1)).collect()
}

/// Parameter property names per tool (aligned to the tools array order), each as a `(term,
/// frequency)` bag built with `content_words`. Property names (`city`, `start_date`) are
/// load-bearing relevance signal, so BM25F scores them as their own field. Reads the raw
/// `tools[]` directly across wire shapes: schema under `function.parameters` (OpenAI Chat),
/// `parameters` (Responses / Gemini declaration), or `input_schema` (Anthropic). Empty for a
/// tool with no `properties`.
fn tool_param_words(req: &Request, stop: &HashSet<&str>) -> Vec<Vec<(String, u32)>> {
    let Some(tools) = req.raw().get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        // Gemini nests declarations one level deeper; each is its own tool entry downstream.
        if let Some(decls) = tool.get("functionDeclarations").and_then(Value::as_array) {
            for d in decls {
                out.push(prop_name_bag(d, stop));
            }
            continue;
        }
        let scope = tool
            .get("function")
            .filter(|f| f.is_object())
            .unwrap_or(tool);
        out.push(prop_name_bag(scope, stop));
    }
    out
}

/// Content-word bag of the `properties` keys of whichever parameter schema a tool/declaration
/// object carries (`input_schema` / `parameters`). Property *keys* only — values are the nested
/// schemas, not relevance text.
fn prop_name_bag(scope: &Value, stop: &HashSet<&str>) -> Vec<(String, u32)> {
    let props = SCHEMA_KEYS
        .iter()
        .find_map(|k| scope.pointer(&format!("/{k}/properties")))
        .and_then(Value::as_object);
    let Some(props) = props else {
        return Vec::new();
    };
    let joined = props
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    bag(&content_words(&joined, stop))
}

/// BM25F score of every tool document against the query term-set (index-aligned to `docs`).
///
/// Canonical BM25F (Robertson, Zaragoza & Taylor, "Simple BM25 Extension to Multiple Weighted
/// Fields", CIKM 2004): combine per-field term frequencies into one pseudo-TF with **per-field**
/// length normalization, *then* apply a single BM25 saturation — not per-field BM25 summed
/// (which double-saturates). `tf̃(t) = Σ_f W_f · tf_f(t) / (1 − b + b · len_f / avglen_f)`, and
/// `score = Σ_t IDF(t) · tf̃(t) / (k1 + tf̃(t))`. IDF is the standard BM25 form over the tool
/// corpus. Deterministic: pure arithmetic over the fixed term/field order.
fn bm25f_scores(docs: &[ToolDoc], query: &HashSet<&str>) -> Vec<f64> {
    let n = docs.len();
    if n == 0 {
        return Vec::new();
    }
    // Per-field average length (in content-word tokens) across the corpus, for length norm.
    let field_len = |d: &ToolDoc, f: usize| -> f64 {
        let v = [&d.name, &d.params, &d.desc][f];
        v.iter().map(|(_, c)| *c as u64).sum::<u64>() as f64
    };
    let weights = [FIELD_W_NAME, FIELD_W_PARAMS, FIELD_W_DESC];
    let mut avg = [0.0f64; 3];
    for (f, a) in avg.iter_mut().enumerate() {
        let total: f64 = docs.iter().map(|d| field_len(d, f)).sum();
        *a = (total / n as f64).max(1.0); // avoid div-by-zero on an all-empty field
    }

    // Document frequency of each query term: a tool "contains" the term if any field does.
    let term_df = |term: &str| -> usize {
        docs.iter()
            .filter(|d| {
                [&d.name, &d.params, &d.desc]
                    .iter()
                    .any(|fld| fld.iter().any(|(w, _)| w == term))
            })
            .count()
    };
    // Standard BM25 IDF, floored at 0 so a term in >half the tools can't push scores negative.
    let idf = |term: &str| -> f64 {
        let df = term_df(term) as f64;
        (((n as f64 - df + 0.5) / (df + 0.5)) + 1.0).ln().max(0.0)
    };
    let idfs: Vec<(&str, f64)> = query.iter().map(|t| (*t, idf(t))).collect();

    docs.iter()
        .map(|d| {
            let fields = [&d.name, &d.params, &d.desc];
            let lens = [field_len(d, 0), field_len(d, 1), field_len(d, 2)];
            idfs.iter()
                .map(|(term, w_idf)| {
                    // Combined, per-field-length-normalized pseudo-TF.
                    let mut tf = 0.0f64;
                    for f in 0..3 {
                        let raw = fields[f]
                            .iter()
                            .find(|(w, _)| w == term)
                            .map_or(0u32, |(_, c)| *c) as f64;
                        if raw > 0.0 {
                            let norm = 1.0 - BM25_B + BM25_B * lens[f] / avg[f];
                            tf += weights[f] * raw / norm;
                        }
                    }
                    if tf > 0.0 {
                        w_idf * tf / (BM25_K1 + tf)
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect()
}

/// True when the request is the first turn of a conversation — at most one non-system message
/// and no tool invoked yet. Tool selection prunes only here (see `select_tools`): later turns
/// must keep the `tools[]` block byte-stable so the provider prompt-cache prefix stays warm.
/// Cross-wire-shape: counts non-system turns in `messages` / `input` / `contents` (Anthropic
/// keeps `system` out of the array, so its first turn is one `user` message; OpenAI's is
/// `system` + `user`), and treats any already-invoked tool as proof of a live loop.
fn is_first_turn(req: &Request) -> bool {
    if !tools_used_in_history(req).is_empty() {
        return false;
    }
    let raw = req.raw();
    let non_system_turns = ["messages", "input", "contents"]
        .iter()
        .filter_map(|k| raw.get(*k).and_then(Value::as_array))
        .map(|a| {
            a.iter()
                .filter(|m| m.get("role").and_then(Value::as_str) != Some("system"))
                .count()
        })
        .max()
        .unwrap_or(0);
    non_system_turns <= 1
}

/// Names of tools already invoked in the conversation — OpenAI Chat `tool_calls[].function.name`,
/// OpenAI Responses `input[]` items of `{type: function_call, name}`, Anthropic `{type: tool_use,
/// name}` content blocks, and Google `parts[].functionCall.name`.
fn tools_used_in_history(req: &Request) -> HashSet<String> {
    let mut used = HashSet::new();
    let raw = req.raw();
    // OpenAI Chat / Anthropic use "messages"; Google uses "contents"; OpenAI Responses uses "input".
    let turns = raw
        .get("messages")
        .or_else(|| raw.get("contents"))
        .or_else(|| raw.get("input"))
        .and_then(Value::as_array);
    let Some(turns) = turns else {
        return used;
    };
    for m in turns {
        // OpenAI Responses: a tool call is an input item `{type: function_call, name, ...}`.
        if m.get("type").and_then(Value::as_str) == Some("function_call")
            && let Some(n) = m.get("name").and_then(Value::as_str)
        {
            used.insert(n.to_string());
        }
        // OpenAI: tool_calls[].function.name
        if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
            for c in calls {
                if let Some(n) = c.pointer("/function/name").and_then(Value::as_str) {
                    used.insert(n.to_string());
                }
            }
        }
        // Anthropic: content[].{type:tool_use, name}
        if let Some(blocks) = m.get("content").and_then(Value::as_array) {
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("tool_use")
                    && let Some(n) = b.get("name").and_then(Value::as_str)
                {
                    used.insert(n.to_string());
                }
            }
        }
        // Google: parts[].functionCall.name
        if let Some(parts) = m.get("parts").and_then(Value::as_array) {
            for p in parts {
                if let Some(n) = p.pointer("/functionCall/name").and_then(Value::as_str) {
                    used.insert(n.to_string());
                }
            }
        }
    }
    used
}

/// True if a text segment is **structured / positional data** — JSON, CSV/TSV, a table, a
/// key-value/config block, or symbol-dense code/markup — rather than natural-language prose.
///
/// Lossy prose transforms (n-gram abbreviation, near-duplicate line collapse) are safe on
/// prose but corrupt structured data, where token *position* and *count* are load-bearing:
/// the model aligns columns, counts records, or parses syntax, so even a byte-reversible
/// change makes it misread the data (a record array abbreviated by n-gram miscounts rows —
/// `adult` −100pp in the bench). Such segments are left verbatim.
///
/// Format- and language-universal: structure is detected by *shape*, not keywords, and
/// scripts are classified with Unicode `char` categories (not ASCII), so prose in any
/// script — Latin, CJK, Arabic, Cyrillic, Indic … — reads as prose.
pub(crate) fn is_structured_segment(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }

    // 1. JSON — a whole value, or adjacent objects forming a record array.
    if t.starts_with(['{', '[']) && serde_json::from_str::<Value>(t).is_ok() {
        return true;
    }
    if t.contains("},{") || t.contains("}, {") {
        return true;
    }

    let lines: Vec<&str> = t.lines().map(str::trim).filter(|l| !l.is_empty()).collect();

    // 2. Tabular — one column delimiter recurring with the same count across most lines
    //    (CSV / TSV / Markdown table). Fields are position-indexed, so collapsing or
    //    abbreviating rows corrupts the alignment.
    if lines.len() >= 3 {
        for delim in [',', '\t', '|', ';'] {
            let counts: Vec<usize> = lines.iter().map(|l| l.matches(delim).count()).collect();
            let cols = counts.iter().copied().max().unwrap_or(0);
            if cols >= 1 && counts.iter().filter(|&&c| c == cols).count() * 4 >= lines.len() * 3 {
                return true; // ≥75% of lines share the same ≥2-column shape
            }
        }
    }

    // 3. Key-value / config — most lines are `key: value` / `key = value` (YAML, TOML, ini,
    //    env, headers). Keys are positional; abbreviating them breaks lookups.
    if lines.len() >= 3 && lines.iter().filter(|l| is_kv_line(l)).count() * 4 >= lines.len() * 3 {
        return true;
    }

    // 4. Symbol density — code / markup / dense structure. Prose in *any* script is mostly
    //    letters; punctuation + symbols stay well under ~15%. Above ~22% means structure.
    let mut symbols = 0usize;
    let mut nonspace = 0usize;
    for c in t.chars() {
        if c.is_whitespace() {
            continue;
        }
        nonspace += 1;
        if !c.is_alphanumeric() {
            symbols += 1;
        }
    }
    nonspace >= 40 && symbols * 100 >= nonspace * 22
}

/// A `key: value` / `key = value` line with a short, single-clause key — the shape of a
/// config / header line, not a prose sentence that merely contains a colon.
fn is_kv_line(line: &str) -> bool {
    match line.find([':', '=']) {
        Some(i) if i > 0 && i + 1 < line.len() => {
            let key = line[..i].trim();
            !key.is_empty() && key.chars().count() <= 40 && !key.contains(['.', '!', '?'])
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::{AnthropicProvider, OpenAiProvider};
    use crate::tokenizer::counter_for;
    use serde_json::{Value, json};

    #[test]
    fn structured_detects_json_and_record_arrays() {
        assert!(is_structured_segment("[{\"a\":1},{\"a\":2}]"));
        assert!(is_structured_segment("{\"k\": \"v\"}"));
        // JSON records followed by a question (whole thing doesn't parse) → adjacency signal.
        assert!(is_structured_segment(
            "[{\"occupation\":\"Sales\"},{\"occupation\":\"Tech\"}] then a question"
        ));
    }

    #[test]
    fn structured_detects_csv_tsv_and_markdown_tables() {
        assert!(is_structured_segment(
            "name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,40,SF"
        ));
        assert!(is_structured_segment(
            "| col | val |\n|-----|-----|\n| a | 1 |\n| b | 2 |"
        ));
    }

    #[test]
    fn structured_detects_key_value_config() {
        assert!(is_structured_segment(
            "host: localhost\nport: 8080\ndebug: true\nname: app"
        ));
        assert!(is_structured_segment("KEY=val\nFOO=bar\nBAZ=qux"));
    }

    #[test]
    fn structured_detects_code_by_symbol_density() {
        assert!(is_structured_segment(
            "for (let i = 0; i < n; i++) { out[i] = (a[i] + b[i]) * w - bias / 2; }"
        ));
    }

    #[test]
    fn prose_is_not_structured_in_any_script() {
        assert!(!is_structured_segment(
            "The quick brown fox jumps over the lazy dog. It was a calm, bright morning, \
             and nothing at all seemed out of the ordinary on that particular day."
        ));
        // CJK prose: enough characters to pass the length floor, but few symbols, and the
        // ideographs are alphabetic → must read as prose, not a table.
        assert!(!is_structured_segment(
            "这是一段用于测试的中文散文文本，它包含足够多的汉字以超过长度阈值，\
             但是标点符号很少，因此不应该被误判成结构化数据或者表格。"
        ));
        // A single prose line with a colon is not key-value.
        assert!(!is_structured_segment(
            "Note: this is an ordinary sentence that merely happens to contain a colon."
        ));
    }

    fn openai_tools() -> Value {
        json!([
            {"type":"function","function":{"name":"get_weather","description":"Get the weather forecast for a city","parameters":{}}},
            {"type":"function","function":{"name":"send_email","description":"Send an email to a recipient","parameters":{}}},
            {"type":"function","function":{"name":"run_sql","description":"Execute a SQL query against the database","parameters":{}}}
        ])
    }

    fn select_stage() -> Box<dyn Transform> {
        Box::new(ToolStage {
            select: true,
            trim_desc: false,
            minify_schema: false,
            max_desc_chars: 200,
        })
    }

    #[test]
    fn openai_selection_keeps_relevant_tool() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"what is the weather forecast in Paris today?"}],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let out = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        assert!(
            out.stages[0].applied,
            "dropping irrelevant tools reduces tokens"
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["get_weather"], "only the weather tool is kept");
    }

    #[test]
    fn selection_skipped_once_a_tool_was_invoked() {
        // `run_sql` was called earlier, so the conversation is a live agent loop. Selection is
        // skipped (issue #9): pruning the `tools[]` block now would change the cached prompt
        // prefix every turn. The full toolset ships unchanged — which also can't dangle the
        // earlier `tool_use`.
        let body = json!({
            "model":"gpt-4o",
            "messages":[
                {"role":"assistant","tool_calls":[{"function":{"name":"run_sql"}}]},
                {"role":"user","content":"now what is the weather forecast in Paris?"}
            ],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let out = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        assert!(!out.stages[0].applied, "selection skipped mid-loop");
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(
            names,
            vec!["get_weather", "send_email", "run_sql"],
            "the full toolset ships unchanged mid-loop: {names:?}"
        );
    }

    #[test]
    fn keeps_all_when_nothing_matches() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"hello there friend"}],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        assert_eq!(
            req.raw()
                .get("tools")
                .and_then(Value::as_array)
                .unwrap()
                .len(),
            3,
            "weak query keeps the whole toolset (safety)"
        );
    }

    #[test]
    fn french_query_uses_french_stopwords() {
        // "des", "la", "pour" are French stopwords; without French detection they
        // would survive and create spurious overlap. The relevant tool still wins.
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"quelle est la météo pour la ville de Paris aujourd'hui"}],
            "tools":[
                {"type":"function","function":{"name":"meteo","description":"Obtenir les prévisions météo pour une ville","parameters":{}}},
                {"type":"function","function":{"name":"envoyer_email","description":"Envoyer un courriel à un destinataire","parameters":{}}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["meteo"], "only the weather tool kept (French)");
    }

    #[test]
    fn anthropic_selection_and_trim() {
        let long_desc = "x".repeat(400);
        let body = json!({
            "max_tokens":100,
            "messages":[{"role":"user","content":"run a sql query on the orders table please"}],
            "tools":[
                {"name":"run_sql","description": long_desc,"input_schema":{}},
                {"name":"get_weather","description":"weather forecast","input_schema":{}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::Anthropic, body);
        let counter = counter_for(ProviderKind::Anthropic, None).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(ToolStage {
            select: true,
            trim_desc: true,
            minify_schema: false,
            max_desc_chars: 50,
        })];
        pipeline::run(&mut req, &AnthropicProvider, counter.as_ref(), &stages);
        let tools = req.raw().get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools.len(), 1, "only run_sql kept");
        let desc = tools[0].get("description").and_then(Value::as_str).unwrap();
        assert!(
            desc.chars().count() <= 51,
            "description trimmed to max+ellipsis"
        );
    }

    #[test]
    fn google_function_call_not_dropped() {
        // tools_used_in_history must find Google functionCall parts so select_tools
        // never orphans a tool already invoked in a Gemini conversation.
        let body = json!({
            "contents": [
                {"role": "user", "parts": [{"text": "what is the weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp": "15°C"}}}]}
            ],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "weather forecast"},
                {"name": "run_sql", "description": "database query"}
            ]}]
        });
        // Ensure get_weather is recognised as already-used; a full pipeline run would
        // keep it even if the keyword selector otherwise wouldn't.
        let req = Request::from_value(ProviderKind::Google, body);
        let used = tools_used_in_history(&req);
        assert!(
            used.contains("get_weather"),
            "Google functionCall must be tracked"
        );
    }

    // ── BM25F fielded tool selection ─────────────────────────────────────────────────────

    /// Tool whose NAME matches the query term must outrank a tool that only carries the term in
    /// its description — the whole point of fielding (flat overlap can't tell them apart).
    #[test]
    fn bm25f_name_match_outranks_description_match() {
        let mk = |words: &[&str]| -> Vec<(String, u32)> {
            words.iter().map(|w| (w.to_string(), 1)).collect()
        };
        let docs = vec![
            // Tool A: "weather" is the NAME.
            ToolDoc {
                name: mk(&["weather"]),
                params: mk(&["city"]),
                desc: mk(&["forecast", "data"]),
            },
            // Tool B: "weather" appears only deep in the DESCRIPTION.
            ToolDoc {
                name: mk(&["search"]),
                params: Vec::new(),
                desc: mk(&["look", "up", "the", "weather", "and", "more"]),
            },
        ];
        let query: HashSet<&str> = ["weather"].into_iter().collect();
        let scores = bm25f_scores(&docs, &query);
        assert!(
            scores[0] > scores[1],
            "name-field match must score above description-only: {scores:?}"
        );
    }

    /// End-to-end through the pipeline: a query naming `run_sql` keeps it and drops the tool
    /// that only mentions "sql" in prose-y description — and the kept set is non-empty.
    #[test]
    fn bm25f_selection_prefers_name_field_end_to_end() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"please run_sql on the users table"}],
            "tools":[
                {"type":"function","function":{"name":"run_sql","description":"Execute a query","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}},
                {"type":"function","function":{"name":"notes","description":"Keep notes about sql and other topics you discuss","parameters":{}}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(
            names.contains(&"run_sql"),
            "name-matched tool kept: {names:?}"
        );
    }

    /// An already-invoked tool survives selection even when it scores at the very bottom
    /// (zero BM25F relevance to the current turn) — the multi-turn breaker is independent of
    /// the score.
    #[test]
    fn bm25f_already_invoked_survives_at_rank_bottom() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[
                {"role":"assistant","tool_calls":[{"function":{"name":"send_email"}}]},
                {"role":"user","content":"now run a sql query on the orders table"}
            ],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"run_sql"), "relevant tool kept");
        assert!(
            names.contains(&"send_email"),
            "already-invoked tool kept despite zero relevance now: {names:?}"
        );
    }

    /// BM25F scoring is deterministic — identical inputs yield byte-identical score vectors,
    /// run to run (no hash-ordering or float-accumulation drift across the fixed term order).
    #[test]
    fn bm25f_is_deterministic() {
        let mk = |words: &[&str]| -> Vec<(String, u32)> {
            words.iter().map(|w| (w.to_string(), 1)).collect()
        };
        let docs = vec![
            ToolDoc {
                name: mk(&["weather", "city"]),
                params: mk(&["city"]),
                desc: mk(&["forecast"]),
            },
            ToolDoc {
                name: mk(&["sql", "run"]),
                params: mk(&["query"]),
                desc: mk(&["database"]),
            },
            ToolDoc {
                name: mk(&["email"]),
                params: mk(&["to"]),
                desc: mk(&["send", "message"]),
            },
        ];
        let query: HashSet<&str> = ["weather", "city", "run"].into_iter().collect();
        let a = bm25f_scores(&docs, &query);
        let b = bm25f_scores(&docs, &query);
        assert_eq!(a, b, "BM25F is deterministic");
    }

    /// Parameter property names feed the BM25F params field: a query term that appears only as a
    /// tool's parameter name still selects it.
    #[test]
    fn bm25f_param_property_names_are_scored() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"set the timezone please"}],
            "tools":[
                {"type":"function","function":{"name":"configure","description":"adjust settings","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}}},
                {"type":"function","function":{"name":"unrelated","description":"does other things","parameters":{}}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(
            names.contains(&"configure"),
            "tool selected via its parameter name: {names:?}"
        );
    }

    // ── Mid-conversation: selection is skipped to keep the cached prefix stable ──────────

    /// Regression (live capture 1781204413588398-27117d) + issue #9: a deferred tool referenced
    /// only by an early instruction must not be dropped. Selection is first-turn-only, so on this
    /// multi-turn conversation it is skipped entirely — the whole toolset (including the
    /// early-mentioned `ToolSearch`) ships unchanged, both keeping the tool call valid and leaving
    /// the cached `tools[]` prefix byte-stable.
    #[test]
    fn is_first_turn_across_wire_shapes() {
        use ProviderKind::{Anthropic, Google, OpenAi};
        let ft = |kind, body: Value| is_first_turn(&Request::from_value(kind, body));

        // OpenAI Chat: leading system + one user message is the first turn.
        assert!(ft(
            OpenAi,
            json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]})
        ));
        assert!(!ft(
            OpenAi,
            json!({"messages":[
                {"role":"system","content":"s"},{"role":"user","content":"hi"},
                {"role":"assistant","tool_calls":[{"function":{"name":"f"}}]},
                {"role":"tool","tool_call_id":"1","content":"x"},
                {"role":"user","content":"again"}]})
        ));

        // Anthropic: `system` is top-level, so the first turn is a single `user` message.
        assert!(ft(
            Anthropic,
            json!({"system":"s","messages":[{"role":"user","content":"hi"}]})
        ));
        assert!(!ft(
            Anthropic,
            json!({"system":"s","messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[{"type":"tool_use","name":"f","input":{}}]}]})
        ));

        // Gemini: `contents` with user/model roles; tool calls are `parts[].functionCall`.
        assert!(ft(
            Google,
            json!({"contents":[{"role":"user","parts":[{"text":"hi"}]}]})
        ));
        assert!(!ft(
            Google,
            json!({"contents":[
                {"role":"user","parts":[{"text":"hi"}]},
                {"role":"model","parts":[{"functionCall":{"name":"f"}}]}]})
        ));

        // OpenAI Responses: `instructions` is system, turns live in `input`.
        assert!(ft(
            OpenAi,
            json!({"instructions":"s","input":[{"role":"user","content":"hi"}]})
        ));
        // A lone `function_call` item counts as one turn, but the loop guard
        // (`tools_used_in_history` scanning `input`) still marks it mid-loop.
        assert!(!ft(
            OpenAi,
            json!({"instructions":"s","input":[
                {"type":"function_call","name":"f","arguments":"{}","call_id":"1"}]})
        ));
    }

    #[test]
    fn multi_turn_conversation_is_not_pruned() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[
                {"role":"user","content":"Use ToolSearch with query select:<name> to load tool schemas before calling them."},
                {"role":"user","content":"what is the weather forecast in Paris today?"}
            ],
            "tools":[
                {"type":"function","function":{"name":"get_weather","description":"Get the weather forecast for a city","parameters":{}}},
                {"type":"function","function":{"name":"ToolSearch","description":"Load deferred tool schemas","parameters":{}}},
                {"type":"function","function":{"name":"send_email","description":"Send an email to a recipient","parameters":{}}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let out = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        assert!(
            !out.stages[0].applied,
            "selection is skipped past the first turn (no churn of the cached tool block)"
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(
            names,
            vec!["get_weather", "ToolSearch", "send_email"],
            "the full toolset ships unchanged mid-conversation: {names:?}"
        );
    }

    /// Word-boundary + case-sensitivity rules of the mention scan.
    #[test]
    fn contains_standalone_word_boundaries() {
        assert!(contains_standalone("use ToolSearch now", "ToolSearch"));
        assert!(contains_standalone(
            "query select:ToolSearch.",
            "ToolSearch"
        )); // punctuation neighbors
        assert!(contains_standalone("ToolSearch", "ToolSearch")); // text edges
        assert!(!contains_standalone("MyToolSearcher", "ToolSearch")); // embedded in identifier
        assert!(!contains_standalone("ToolSearch_v2", "ToolSearch")); // underscore joins identifiers
        assert!(!contains_standalone("use toolsearch now", "ToolSearch")); // case-sensitive
        assert!(contains_standalone(
            "call mcp__server__thing here",
            "mcp__server__thing"
        ));
        assert!(!contains_standalone(
            "mcp__server__thing2",
            "mcp__server__thing"
        ));
        // Unicode neighbor counts as identifier char: no false boundary on non-ASCII letters.
        assert!(!contains_standalone("préToolSearch", "ToolSearch"));
    }

    // ── Schema minification through the stage ────────────────────────────────────────────

    /// The stage applies the schema minifier in place: `$schema`/`title` drop, single-type
    /// arrays collapse, strict-mode `additionalProperties:false` + `required` survive — across
    /// both the OpenAI `function.parameters` and Anthropic `input_schema` field shapes.
    #[test]
    fn stage_minifies_openai_and_anthropic_schemas() {
        let verbose = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Args",
            "type": "object",
            "additionalProperties": false,
            "properties": {"q": {"type": ["string"], "title": "Q", "description": "text"}},
            "required": ["q"]
        });
        // OpenAI Chat: schema under function.parameters.
        let oa = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"search","description":"d","parameters": verbose.clone()}}]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, oa);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stage: Vec<Box<dyn Transform>> = vec![Box::new(ToolStage {
            select: false,
            trim_desc: false,
            minify_schema: true,
            max_desc_chars: 300,
        })];
        pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stage);
        let schema = req.raw().pointer("/tools/0/function/parameters").unwrap();
        assert!(schema.get("$schema").is_none(), "$schema dropped");
        assert!(schema.get("title").is_none(), "root title dropped");
        assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
        assert_eq!(schema.get("required"), Some(&json!(["q"])));
        assert_eq!(
            schema.pointer("/properties/q/type").and_then(Value::as_str),
            Some("string"),
            "single-type array collapsed"
        );

        // Anthropic: schema under input_schema.
        let an = json!({
            "max_tokens": 100,
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"name":"search","description":"d","input_schema": verbose}]
        });
        let mut req = Request::from_value(ProviderKind::Anthropic, an);
        let counter = counter_for(ProviderKind::Anthropic, None).unwrap();
        let stage: Vec<Box<dyn Transform>> = vec![Box::new(ToolStage {
            select: false,
            trim_desc: false,
            minify_schema: true,
            max_desc_chars: 300,
        })];
        pipeline::run(&mut req, &AnthropicProvider, counter.as_ref(), &stage);
        let schema = req.raw().pointer("/tools/0/input_schema").unwrap();
        assert!(schema.get("$schema").is_none(), "anthropic $schema dropped");
        assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
        assert_eq!(
            schema.pointer("/properties/q/type").and_then(Value::as_str),
            Some("string")
        );
    }
}
