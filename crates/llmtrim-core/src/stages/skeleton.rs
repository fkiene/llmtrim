//! Stage C — tree-sitter skeletonization of non-focus code. LOSSY / opt-in.
//!
//! Drops function bodies in fenced code blocks to bare interface skeletons —
//! signatures, imports, and type/struct definitions stay; bodies become
//! a per-language stub (the *RepoCoder* result: AST skeletons beat raw source for
//! cross-file context). Static AST work, no model. Supports the top languages — Rust,
//! JS, TS(X), Python, Go, Java, C, C++, C#, Kotlin, Swift, Zig, Ruby, PHP — each with
//! its own function-node kinds, body location, and stub (brace `{ /* … */ }`, Zig `{}`,
//! Python `...`, Ruby `# …`). Unknown languages pass through untouched.
//!
//! **Relevance-graded pruning** (*Hierarchical Context Pruning*, arXiv:2406.18294, 2024).
//! HCP's ablations: cross-file function bodies are mostly noise — pruning them costs ≈0
//! accuracy (validating the uniform stage above) — *but* keeping full bodies for the top-k
//! functions most relevant to the completion point, signatures for the next tier, and
//! dropping the rest beats uniform treatment (+5.75pp EM at 83–85% token reduction); and
//! direct-dependency (import-depth-1) code deserves richer treatment than depth-2+. HCP
//! ranked relevance with embeddings; spec rule 1 forbids those, so we replace it with
//! **lexical identifier overlap** (the same family as the `bm25` crate's weighting) between
//! each function's identifier set — name, params, called identifiers, extracted via
//! tree-sitter, snake/camel-split, Unicode-aware — and the conversation query (recent user
//! prose + short/focus segments, the same anchor Stage B retrieval uses). Three tiers,
//! decided per function node across the whole request: (1) **keep-full** — top-k by overlap
//! keep their bodies verbatim (k small, `skeleton_keep_full_top_k`, default 5 across the
//! request); (2) **skeletonize** — everyone else (the default uniform behavior); (3) **drop**
//! — signature removed too; OFF by default (`skeleton_drop_unmatched`), only for zero-overlap
//! functions whose body exceeds `skeleton_drop_min_body_lines`.
//! When multiple blocks share `import`/`use` module names with a query-overlapping (focus)
//! block, that block's functions get a small rank bonus (HCP's depth-1 finding, scoped to
//! what a proxy sees); unresolved relationships degrade gracefully to pure lexical ranking.
//! With no conversational query the stage falls back to uniform skeletonization — exactly
//! today's semantics (regression-safe).
//!
//! Lossy (bodies are gone), so off by default and InputTokens-gated. The token gate
//! reverts it if a block doesn't actually shrink.

use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;
use tree_sitter::{Language, Node, Parser};

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::stages::tools::{lex_words, stopword_set};

/// Fenced code block: ```lang[ info-string]\r?\n…code…``` (DOTALL, non-greedy). The info
/// line may carry a CRLF and a *whitespace-delimited* info string (```` ```rust title=x ````);
/// group 1 is the language tag (first token), group 2 the rest of the info line including
/// its leading space (kept on reconstruction), group 3 the code. The info string must start
/// at whitespace, so a `c#`-style tag isn't silently split into `c`. Without `\r?` and the
/// info-string allowance, CRLF / titled fences never matched and passed through uncompressed.
static FENCE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```([A-Za-z0-9_+-]*)([ \t][^\r\n]*)?\r?\n(.*?)```").unwrap());

/// Brace-language body stub (valid wherever `/* … */` block comments are).
const BRACE: &str = "{ /* … */ }";

/// A supported language for fenced-code transforms: its tree-sitter grammar, the node
/// kinds that wrap a function/method, how to find the body block, and the stub that
/// replaces it.
struct LangCfg {
    language: fn() -> Language,
    fn_kinds: &'static [&'static str],
    /// Field name of the body block; `None` → locate the body by kind (Kotlin's
    /// `function_declaration` exposes no `body` field).
    body_field: Option<&'static str>,
    /// Node kinds accepted as an elidable body. Guards JS/TS arrow *expression* bodies
    /// (left intact) and drives the by-kind fallback when `body_field` is `None`.
    body_kinds: &'static [&'static str],
    /// Replacement for the body node's span: [`BRACE`], or a language-specific stub
    /// where that's invalid (Zig has no block comments; Python/Ruby aren't braced).
    placeholder: &'static str,
    /// Whitespace outside strings is insignificant → safe for `minify_code` to strip
    /// indentation. `false` for off-side-rule languages (Python) where indent = structure.
    minifiable: bool,
}

/// Map a fenced-block language tag to its config. Node kinds + body fields verified
/// empirically against tree-sitter 0.26 (see each grammar's parse). Unknown → `None`.
fn lang_for(tag: &str) -> Option<LangCfg> {
    let cfg = match tag.to_ascii_lowercase().as_str() {
        "rust" | "rs" => LangCfg {
            language: || Language::new(tree_sitter_rust::LANGUAGE),
            fn_kinds: &["function_item"],
            body_field: Some("body"),
            body_kinds: &["block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "js" | "javascript" | "jsx" | "mjs" | "cjs" => LangCfg {
            language: || Language::new(tree_sitter_javascript::LANGUAGE),
            fn_kinds: &[
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
                "generator_function",
            ],
            body_field: Some("body"),
            body_kinds: &["statement_block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "ts" | "typescript" | "mts" | "cts" => LangCfg {
            language: || Language::new(tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
            fn_kinds: &[
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
            ],
            body_field: Some("body"),
            body_kinds: &["statement_block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "tsx" => LangCfg {
            language: || Language::new(tree_sitter_typescript::LANGUAGE_TSX),
            fn_kinds: &[
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
            ],
            body_field: Some("body"),
            body_kinds: &["statement_block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "python" | "py" => LangCfg {
            language: || Language::new(tree_sitter_python::LANGUAGE),
            fn_kinds: &["function_definition"],
            body_field: Some("body"),
            body_kinds: &["block"],
            placeholder: "...",
            minifiable: false, // off-side rule: indentation is structure
        },
        "go" | "golang" => LangCfg {
            language: || Language::new(tree_sitter_go::LANGUAGE),
            fn_kinds: &["function_declaration", "method_declaration"],
            body_field: Some("body"),
            body_kinds: &["block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "java" => LangCfg {
            language: || Language::new(tree_sitter_java::LANGUAGE),
            fn_kinds: &["method_declaration", "constructor_declaration"],
            body_field: Some("body"),
            body_kinds: &["block", "constructor_body"],
            placeholder: BRACE,
            minifiable: true,
        },
        "c" | "h" => LangCfg {
            language: || Language::new(tree_sitter_c::LANGUAGE),
            fn_kinds: &["function_definition"],
            body_field: Some("body"),
            body_kinds: &["compound_statement"],
            placeholder: BRACE,
            minifiable: true,
        },
        "cpp" | "c++" | "cc" | "cxx" | "hpp" | "hxx" => LangCfg {
            language: || Language::new(tree_sitter_cpp::LANGUAGE),
            fn_kinds: &["function_definition"],
            body_field: Some("body"),
            body_kinds: &["compound_statement"],
            placeholder: BRACE,
            minifiable: true,
        },
        "csharp" | "cs" | "c#" => LangCfg {
            language: || Language::new(tree_sitter_c_sharp::LANGUAGE),
            fn_kinds: &["method_declaration", "constructor_declaration"],
            body_field: Some("body"),
            body_kinds: &["block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "kotlin" | "kt" | "kts" => LangCfg {
            language: || Language::new(tree_sitter_kotlin_ng::LANGUAGE),
            fn_kinds: &["function_declaration", "secondary_constructor"],
            body_field: None, // no `body` field — locate by kind
            body_kinds: &["function_body", "block"],
            placeholder: BRACE,
            minifiable: true,
        },
        "swift" => LangCfg {
            language: || Language::new(tree_sitter_swift::LANGUAGE),
            fn_kinds: &["function_declaration", "init_declaration"],
            body_field: Some("body"),
            body_kinds: &["function_body"],
            placeholder: BRACE,
            minifiable: true,
        },
        "zig" => LangCfg {
            language: || Language::new(tree_sitter_zig::LANGUAGE),
            fn_kinds: &["function_declaration"],
            body_field: Some("body"),
            body_kinds: &["block"],
            placeholder: "{}", // Zig has no `/* */` block comments
            minifiable: true,
        },
        "ruby" | "rb" => LangCfg {
            language: || Language::new(tree_sitter_ruby::LANGUAGE),
            fn_kinds: &["method", "singleton_method"],
            body_field: Some("body"),
            body_kinds: &["body_statement"],
            placeholder: "# …",
            minifiable: true,
        },
        "php" => LangCfg {
            language: || Language::new(tree_sitter_php::LANGUAGE_PHP),
            fn_kinds: &["function_definition", "method_declaration"],
            body_field: Some("body"),
            body_kinds: &["compound_statement"],
            placeholder: BRACE,
            minifiable: true,
        },
        _ => return None,
    };
    Some(cfg)
}

/// Locate the elidable body of a function node: by field name when the grammar labels
/// it, else the first child of an accepted body kind (Kotlin). The kind check also skips
/// JS/TS arrow *expression* bodies — those must not be replaced with a brace stub.
fn body_node<'a>(node: &tree_sitter::Node<'a>, cfg: &LangCfg) -> Option<tree_sitter::Node<'a>> {
    let candidate = match cfg.body_field {
        Some(field) => node.child_by_field_name(field)?,
        None => {
            let mut c = node.walk();
            node.children(&mut c)
                .find(|ch| cfg.body_kinds.contains(&ch.kind()))?
        }
    };
    cfg.body_kinds
        .contains(&candidate.kind())
        .then_some(candidate)
}

/// Replace each outermost function body in `code` with the language stub — the uniform
/// (every body) skeletonization, used as the no-query fallback and the per-function
/// "skeletonize" tier. Returns the input unchanged if the grammar can't be loaded or
/// parsing fails (never block).
fn skeletonize_code(code: &str, cfg: &LangCfg) -> String {
    skeletonize_graded(code, cfg, None)
}

/// What to do with one function node's span (HCP three-tier decision).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tier {
    /// Keep the whole function verbatim (body included): a query-relevant top-k node.
    KeepFull,
    /// Replace the body with the language stub (signature/declaration stays).
    Skeletonize,
    /// Remove the entire function — signature too (zero-overlap, large body, opt-in).
    Drop,
}

/// One outermost function in a block: its byte span (whole node + body), its identifier
/// set (lowercased subtokens), and its body line count. Spans are relative to the block's
/// code string. Collected once, then ranked across the request.
struct FnRecord {
    node_start: usize,
    node_end: usize,
    body_start: usize,
    body_end: usize,
    idents: HashSet<String>,
    body_lines: usize,
}

/// Walk a parsed block and collect its outermost functions. Descent stops at a function
/// (nested bodies are subsumed by the outer node), so the records mirror what
/// [`skeletonize_code`] elides. Deterministic order: byte-start ascending.
fn collect_fns(code: &str, cfg: &LangCfg) -> Vec<FnRecord> {
    let mut parser = Parser::new();
    if parser.set_language(&(cfg.language)()).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(code.as_bytes(), None) else {
        return Vec::new();
    };
    let bytes = code.as_bytes();
    let mut fns: Vec<FnRecord> = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if cfg.fn_kinds.contains(&node.kind())
            && let Some(body) = body_node(&node, cfg)
        {
            let mut idents = HashSet::new();
            collect_identifiers(node, bytes, &mut idents);
            let slice = code.get(node.start_byte()..node.end_byte()).unwrap_or("");
            fns.push(FnRecord {
                node_start: node.start_byte(),
                node_end: node.end_byte(),
                body_start: body.start_byte(),
                body_end: body.end_byte(),
                idents,
                body_lines: slice.lines().count(),
            });
            continue;
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
    fns.sort_by_key(|f| f.node_start);
    fns
}

/// Apply per-function tier decisions to a block's `code` (decisions indexed by the order
/// [`collect_fns`] returns). `None` ⇒ uniform skeletonization (every body → stub), the
/// no-query fallback. Splices back-to-front so earlier byte offsets stay valid.
fn skeletonize_graded(code: &str, cfg: &LangCfg, tiers: Option<&[Tier]>) -> String {
    let fns = collect_fns(code, cfg);
    if fns.is_empty() {
        return code.to_string();
    }
    // (span, replacement) edits, applied back-to-front.
    let mut edits: Vec<(usize, usize, &str)> = Vec::with_capacity(fns.len());
    for (i, f) in fns.iter().enumerate() {
        let tier = tiers.map_or(Tier::Skeletonize, |t| {
            t.get(i).copied().unwrap_or(Tier::Skeletonize)
        });
        match tier {
            Tier::KeepFull => {}
            Tier::Skeletonize => edits.push((f.body_start, f.body_end, cfg.placeholder)),
            // Remove the whole node; the surrounding fence/markers stay. Leaving a bare
            // language-comment marker would need a per-language comment syntax, so we drop
            // the span outright — the gate still measures the net token win.
            Tier::Drop => edits.push((f.node_start, f.node_end, "")),
        }
    }
    edits.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
    let mut out = code.to_string();
    for (start, end, repl) in edits {
        if start < end && end <= out.len() {
            out.replace_range(start..end, repl);
        }
    }
    out
}

/// Recursively collect identifier subtokens under `node` into `out`. Language-agnostic:
/// every tree-sitter grammar exposes identifier-like leaf kinds (`identifier`,
/// `type_identifier`, `field_identifier`, `property_identifier`, `shorthand_property_…`,
/// `simple_identifier`, …), so we match any node whose kind *contains* "identifier"
/// rather than hand-writing per-language regexes (spec rule 2). Each identifier's text is
/// snake/camel-split into Unicode-aware subtokens for overlap matching.
fn collect_identifiers(node: Node, bytes: &[u8], out: &mut HashSet<String>) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        if n.kind().contains("identifier")
            && let Ok(text) = n.utf8_text(bytes)
        {
            split_subtokens(text, out);
        }
        let mut c = n.walk();
        for ch in n.children(&mut c) {
            stack.push(ch);
        }
    }
}

/// Split an identifier into lowercased subtokens, Unicode-aware (spec rule 2 + 5):
/// first segment on non-alphanumeric separators (snake_case, kebab, `::`, `.`), then split
/// each camelCase/PascalCase run on lower→upper boundaries. The whole identifier (lowercased)
/// is always kept too, so a script without case or word separators (e.g. CJK identifiers)
/// still matches by whole token. Subtokens shorter than 2 chars are dropped as noise.
fn split_subtokens(ident: &str, out: &mut HashSet<String>) {
    let whole: String = ident.chars().flat_map(char::to_lowercase).collect();
    if whole.chars().count() >= 2 {
        out.insert(whole);
    }
    for seg in ident.split(|c: char| !c.is_alphanumeric()) {
        if seg.is_empty() {
            continue;
        }
        let mut cur = String::new();
        let mut prev_lower = false;
        for ch in seg.chars() {
            if ch.is_uppercase() && prev_lower && !cur.is_empty() {
                flush_subtoken(&cur, out);
                cur.clear();
            }
            cur.extend(ch.to_lowercase());
            prev_lower = ch.is_lowercase();
        }
        flush_subtoken(&cur, out);
    }
}

/// Insert a subtoken if it carries signal (≥2 chars).
fn flush_subtoken(tok: &str, out: &mut HashSet<String>) {
    if tok.chars().count() >= 2 {
        out.insert(tok.to_string());
    }
}

/// Module/name tokens of a block's `import`/`use`/`include`/`require` statements, for the
/// HCP depth-1 import bonus. Language-agnostic: collect identifier subtokens from every
/// node whose kind names an import (`use_declaration`, `import_statement`, `import_from_…`,
/// `import_declaration`, `preproc_include`, …). Empty when the grammar exposes none or the
/// block has no imports — the bonus then degrades to nothing (pure lexical ranking).
fn import_tokens(code: &str, cfg: &LangCfg) -> HashSet<String> {
    let mut parser = Parser::new();
    let mut out = HashSet::new();
    if parser.set_language(&(cfg.language)()).is_err() {
        return out;
    }
    let Some(tree) = parser.parse(code.as_bytes(), None) else {
        return out;
    };
    let bytes = code.as_bytes();
    let mut stack = vec![tree.root_node()];
    while let Some(n) = stack.pop() {
        let k = n.kind();
        if (k.contains("import") || k.contains("use_") || k.contains("include") || k == "require")
            && !k.contains("identifier")
        {
            collect_identifiers(n, bytes, &mut out);
            continue; // whole import subtree captured
        }
        let mut c = n.walk();
        for ch in n.children(&mut c) {
            stack.push(ch);
        }
    }
    out
}

/// Lexical overlap score between a function's identifier set and the query term set:
/// the count of distinct query terms the function mentions (a weighted-Jaccard numerator —
/// the denominator is constant across functions for a fixed query, so it doesn't change the
/// ranking). Returns 0 when disjoint. Embedding-free (spec rule 1).
fn overlap_score(idents: &HashSet<String>, query: &HashSet<String>) -> usize {
    if query.len() <= idents.len() {
        query.iter().filter(|q| idents.contains(*q)).count()
    } else {
        idents.iter().filter(|i| query.contains(*i)).count()
    }
}

/// A function candidate gathered across the whole request, with everything needed to rank
/// it globally and then rewrite its block.
struct Candidate {
    ptr_idx: usize,
    block_idx: usize,
    fn_idx: usize,
    score: usize,
    has_import_bonus: bool,
    body_lines: usize,
    overlaps: bool,
}

pub struct SkeletonStage {
    /// Functions (by query overlap, across the whole request) that keep their full body.
    pub keep_full_top_k: usize,
    /// Also remove the signature of zero-overlap functions with large bodies.
    pub drop_unmatched: bool,
    /// Minimum body line count for the drop tier to apply.
    pub drop_min_body_lines: usize,
}

impl Transform for SkeletonStage {
    fn name(&self) -> &str {
        "skeleton"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> anyhow::Result<()> {
        let pointers = provider.content_text_pointers(req);
        let query = build_query(req, &pointers);

        // No conversational query ⇒ no relevance signal: fall back to uniform
        // skeletonization (today's exact behavior — the regression-safe path).
        if query.is_empty() {
            rewrite_fenced_code(req, provider, skeletonize_code);
            return Ok(());
        }

        // Mutation targets: only cache-compressible segments (the invariant cached
        // prefix is never rewritten), matching `rewrite_fenced_code`'s pointer source.
        let targets = crate::cache_zone::compressible_pointers(req, provider);

        // Phase 1 — collect every function across every fenced block, plus per-block
        // import tokens and which block(s) are "focus" (carry query-overlapping imports
        // or are in a short/query segment). One immutable pass.
        let scan = scan_request(req, &targets, &query);
        if scan.candidates.is_empty() {
            // Query present but no rankable functions (e.g. unsupported langs only): still
            // apply uniform skeletonization so supported blocks aren't left uncompressed.
            rewrite_fenced_code(req, provider, skeletonize_code);
            return Ok(());
        }

        // Phase 2 — rank globally and assign tiers, then rewrite each block.
        let tiers = self.assign_tiers(&scan);
        apply_tiers(req, &targets, &tiers);
        Ok(())
    }
}

impl SkeletonStage {
    /// Rank all candidates and assign a [`Tier`] to each, keyed by `(ptr, block, fn)`.
    /// Top-k by (overlap score + import bonus) keep their bodies; zero-overlap large
    /// bodies drop entirely when enabled; everything else is skeletonized.
    fn assign_tiers(&self, scan: &Scan) -> std::collections::HashMap<(usize, usize, usize), Tier> {
        // Rank: higher score first, import-bonus breaks ties, then earliest position
        // (ptr, block, fn) for determinism. Only candidates that actually overlap the
        // query (score > 0) are eligible for keep-full — never promote a noise function.
        let mut ranked: Vec<&Candidate> = scan.candidates.iter().filter(|c| c.overlaps).collect();
        ranked.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then(b.has_import_bonus.cmp(&a.has_import_bonus))
                .then(a.ptr_idx.cmp(&b.ptr_idx))
                .then(a.block_idx.cmp(&b.block_idx))
                .then(a.fn_idx.cmp(&b.fn_idx))
        });
        let keep: HashSet<(usize, usize, usize)> = ranked
            .iter()
            .take(self.keep_full_top_k)
            .map(|c| (c.ptr_idx, c.block_idx, c.fn_idx))
            .collect();

        scan.candidates
            .iter()
            .map(|c| {
                let key = (c.ptr_idx, c.block_idx, c.fn_idx);
                let tier = if keep.contains(&key) {
                    Tier::KeepFull
                } else if self.drop_unmatched
                    && !c.overlaps
                    && c.body_lines >= self.drop_min_body_lines
                {
                    Tier::Drop
                } else {
                    Tier::Skeletonize
                };
                (key, tier)
            })
            .collect()
    }
}

/// The result of the immutable phase-1 scan: every function candidate (with its global
/// score) across the request.
struct Scan {
    candidates: Vec<Candidate>,
}

/// The `/messages/{i}` index a content pointer addresses, if any. Local copy of Stage B's
/// `msg_index` (its is private and retrieve.rs is out of scope to touch).
fn msg_index(pointer: &str) -> Option<usize> {
    let rest = pointer.strip_prefix("/messages/")?;
    let end = rest.find('/').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Build the conversation query term set — the relevance anchor. Mirrors Stage B
/// retrieval's anchor: the final user turn plus any short segments (the question + focused
/// snippets), excluding large bulk context that would make every function "overlap".
/// Lowercased, snake/camel-split, stopwords dropped, so the terms match function-identifier
/// subtokens. Empty ⇒ no anchor (the caller then falls back to uniform skeletonization).
fn build_query(req: &Request, pointers: &[String]) -> HashSet<String> {
    // Short-segment threshold: the same spirit as retrieve's `min_segment_chars` default —
    // segments below it are treated as query/focus, not bulk context.
    const SHORT_SEGMENT_CHARS: usize = 600;

    let role_of = |i: usize| -> Option<String> {
        req.raw()
            .pointer(&format!("/messages/{i}/role"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    let last_user = pointers
        .iter()
        .filter_map(|p| msg_index(p))
        .filter(|&i| role_of(i).as_deref() == Some("user"))
        .max();

    let mut text = String::new();
    for p in pointers {
        let Some(s) = req.get_str(p) else { continue };
        let idx = msg_index(p);
        let is_last_user = idx.is_some() && idx == last_user;
        let is_short = s.chars().count() < SHORT_SEGMENT_CHARS;
        if is_last_user || is_short {
            // The query is the natural-language intent, NOT the pasted code: strip fenced
            // blocks so a function never ranks itself relevant. A lone code block (the whole
            // user turn is one fence) therefore yields an empty query → uniform fallback,
            // exactly today's behavior (the regression guard).
            let prose = FENCE.replace_all(s, " ");
            text.push_str(&prose);
            text.push(' ');
        }
    }
    if text.trim().is_empty() {
        return HashSet::new();
    }
    let stops = stopword_set(&text);
    lex_words(&text)
        .into_iter()
        .flat_map(|w| {
            // Split query terms the same way function identifiers are, so `loadUser`
            // in the prompt matches `load_user` in the code (and vice versa).
            let mut subs = HashSet::new();
            split_subtokens(&w, &mut subs);
            subs.insert(w);
            subs.into_iter()
        })
        .filter(|w| w.chars().count() >= 2 && !stops.contains(w.as_str()))
        .collect()
}

/// Phase-1 scan: for every fenced block in every content pointer, collect its functions
/// (with query-overlap scores) and apply the import-tier bonus. Read-only — no mutation,
/// so tier assignment sees the whole request before anything changes.
fn scan_request(req: &Request, pointers: &[String], query: &HashSet<String>) -> Scan {
    // First pass over blocks: gather each block's (ptr_idx, block_idx, lang, code, fns,
    // import_tokens). Then decide focus blocks, then score.
    struct BlockInfo {
        ptr_idx: usize,
        block_idx: usize,
        imports: HashSet<String>,
        fns: Vec<FnRecord>,
    }
    let mut blocks: Vec<BlockInfo> = Vec::new();
    for (ptr_idx, ptr) in pointers.iter().enumerate() {
        let Some(s) = req.get_str(ptr) else { continue };
        for (block_idx, caps) in FENCE.captures_iter(s).enumerate() {
            let tag = &caps[1];
            let code = &caps[3];
            let Some(cfg) = lang_for(tag) else { continue };
            let fns = collect_fns(code, &cfg);
            if fns.is_empty() {
                continue;
            }
            blocks.push(BlockInfo {
                ptr_idx,
                block_idx,
                imports: import_tokens(code, &cfg),
                fns,
            });
        }
    }

    // Import-tier bonus (HCP depth-1), only meaningful with MULTIPLE blocks: a block whose
    // imports overlap the query is a "focus" block; OTHER blocks sharing import module names
    // with a focus block get the bonus. Degrades to no bonus when relationships don't resolve.
    let multi = blocks.len() > 1;
    let focus_imports: HashSet<String> = if multi {
        blocks
            .iter()
            .filter(|b| !b.imports.is_disjoint(query))
            .flat_map(|b| b.imports.iter().cloned())
            .collect()
    } else {
        HashSet::new()
    };

    let mut candidates = Vec::new();
    for b in &blocks {
        let import_bonus =
            multi && !focus_imports.is_empty() && !b.imports.is_disjoint(&focus_imports);
        for (fn_idx, f) in b.fns.iter().enumerate() {
            let base = overlap_score(&f.idents, query);
            // The bonus nudges ranking without manufacturing overlap from nothing: a
            // dependency-linked block's functions sort above an unrelated block's at the
            // same lexical score, but a zero-overlap function stays zero (never keep-full).
            let score = base + if import_bonus && base > 0 { 1 } else { 0 };
            candidates.push(Candidate {
                ptr_idx: b.ptr_idx,
                block_idx: b.block_idx,
                fn_idx,
                score,
                has_import_bonus: import_bonus,
                body_lines: f.body_lines,
                overlaps: base > 0,
            });
        }
    }
    Scan { candidates }
}

/// Phase-2 rewrite: replay `FENCE` over each pointer and apply the assigned tiers to each
/// supported block's functions (indexed by the same `(ptr, block, fn)` keys phase 1 used).
fn apply_tiers(
    req: &mut Request,
    pointers: &[String],
    tiers: &std::collections::HashMap<(usize, usize, usize), Tier>,
) {
    for (ptr_idx, ptr) in pointers.iter().enumerate() {
        let Some(s) = req.get_str(ptr).map(str::to_string) else {
            continue;
        };
        let mut block_idx = 0usize;
        let rewritten = FENCE.replace_all(&s, |caps: &regex::Captures| {
            // Index EVERY fenced block (matching phase 1's `captures_iter().enumerate()`),
            // so the `(ptr, block, fn)` keys line up whether or not the lang is supported.
            let bi = block_idx;
            block_idx += 1;
            let tag = &caps[1];
            // Rest of the info line (e.g. " title=x"); optional, preserved on reconstruction.
            let info = caps.get(2).map_or("", |m| m.as_str());
            let code = &caps[3];
            match lang_for(tag) {
                Some(cfg) => {
                    // Tier per function in collect_fns order; default Skeletonize when a
                    // function wasn't scored (it would only be missing if the block has no
                    // candidate, in which case the per-fn lookup falls back to Skeletonize).
                    let fns = collect_fns(code, &cfg);
                    let per_fn: Vec<Tier> = (0..fns.len())
                        .map(|fi| {
                            tiers
                                .get(&(ptr_idx, bi, fi))
                                .copied()
                                .unwrap_or(Tier::Skeletonize)
                        })
                        .collect();
                    format!(
                        "```{tag}{info}\n{}```",
                        skeletonize_graded(code, &cfg, Some(&per_fn))
                    )
                }
                None => caps[0].to_string(),
            }
        });
        if rewritten != s {
            req.set(ptr, serde_json::Value::String(rewritten.into_owned()));
        }
    }
}

/// Rewrite every fenced code block across the request's content text segments,
/// applying `f` to the code of supported-language blocks and leaving other
/// fences untouched. The shared body of [`SkeletonStage`] and [`MinifyCodeStage`] —
/// only the per-block transform `f` differs.
fn rewrite_fenced_code(
    req: &mut Request,
    provider: &dyn Provider,
    f: impl Fn(&str, &LangCfg) -> String,
) {
    for ptr in crate::cache_zone::compressible_pointers(req, provider) {
        let Some(s) = req.get_str(&ptr).map(str::to_string) else {
            continue;
        };
        let rewritten = FENCE.replace_all(&s, |caps: &regex::Captures| {
            let tag = &caps[1];
            // Rest of the info line (e.g. " title=x"); optional, preserved on reconstruction.
            let info = caps.get(2).map_or("", |m| m.as_str());
            let code = &caps[3];
            match lang_for(tag) {
                Some(cfg) => format!("```{tag}{info}\n{}```", f(code, &cfg)),
                None => caps[0].to_string(),
            }
        });
        if rewritten != s {
            req.set(&ptr, serde_json::Value::String(rewritten.into_owned()));
        }
    }
}

/// Strip insignificant whitespace from brace-language code — indentation + blank
/// lines — while protecting string literals (the format-removal lever,
/// arXiv:2508.13666: ~24.5% input tokens at near-parity). Whitespace is
/// insignificant in brace languages, so this is semantically lossless; Python and
/// other whitespace-significant languages aren't in `lang_for`, so never touched.
fn minify_code(code: &str, cfg: &LangCfg) -> String {
    if !cfg.minifiable {
        return code.to_string(); // off-side-rule language: indentation is significant
    }
    let mut parser = Parser::new();
    if parser.set_language(&(cfg.language)()).is_err() {
        return code.to_string();
    }
    let Some(tree) = parser.parse(code.as_bytes(), None) else {
        return code.to_string();
    };
    // Byte ranges of string literals — lines overlapping these are kept verbatim.
    let mut protected: Vec<(usize, usize)> = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(n) = stack.pop() {
        if n.kind().contains("string") || n.kind().contains("heredoc") {
            protected.push((n.start_byte(), n.end_byte()));
        }
        let mut c = n.walk();
        for ch in n.children(&mut c) {
            stack.push(ch);
        }
    }
    let mut out = String::with_capacity(code.len());
    let mut pos = 0usize;
    for line in code.split_inclusive('\n') {
        let (start, end) = (pos, pos + line.len());
        pos = end;
        let overlaps_string = protected.iter().any(|&(s, e)| s < end && e > start);
        if overlaps_string {
            out.push_str(line); // protect string content (incl. its indentation)
        } else {
            let trimmed = line.trim_start();
            if !trimmed.trim().is_empty() {
                out.push_str(trimmed); // strip indentation; drop blank lines
            }
        }
    }
    out
}

pub struct MinifyCodeStage;

impl Transform for MinifyCodeStage {
    fn name(&self) -> &str {
        "minify-code"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> anyhow::Result<()> {
        rewrite_fenced_code(req, provider, minify_code);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default-configured stage (keep-full top-5, drop tier off) — the shipped `code`/`auto`
    /// behavior. Tests that exercise the drop tier build the struct inline.
    fn skeleton_stage() -> SkeletonStage {
        SkeletonStage {
            keep_full_top_k: 5,
            drop_unmatched: false,
            drop_min_body_lines: 8,
        }
    }

    /// Approximate token count (whitespace words) for savings-threshold assertions, matching
    /// the project's CLI-testing convention.
    fn count_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }

    #[test]
    fn auto_skeletonizes_kotlin_end_to_end() {
        // Full path: auto → route(fenced code) → `code` preset → skeleton stage.
        use serde_json::json;
        let code = "fun handle(req: Request): Response {\n    val auth = req.header(\"authorization\")\n    val user = lookupUser(auth)\n    val perms = loadPermissions(user.id)\n    val payload = req.parseBody()\n    val saved = repository.persist(payload, user)\n    audit.log(user, saved.id)\n    return Response.ok(saved)\n}";
        let input = json!({"model":"gpt-4o","messages":[{"role":"user","content":format!("Review:\n```kotlin\n{code}\n```")}]}).to_string();
        let cfg = crate::config::DenseConfig::auto();
        let res = crate::compress_with_config(&input, Some(crate::ir::ProviderKind::OpenAi), &cfg)
            .unwrap();
        assert!(
            !res.request_json.contains("loadPermissions")
                && res.request_json.contains("fun handle"),
            "auto path should skeletonize the Kotlin body; got:\n{}",
            res.request_json
        );
    }

    #[test]
    fn minify_strips_indentation_protects_strings() {
        let code = "fn main() {\n    let x = 1;\n    let s = \"  keep  spaces  \";\n}\n";
        let out = minify_code(code, &lang_for("rust").unwrap());
        assert!(out.contains("let x = 1;"));
        assert!(!out.contains("    let x"), "indentation stripped");
        assert!(
            out.contains("\"  keep  spaces  \""),
            "whitespace inside string literals protected"
        );
    }

    #[test]
    fn rust_bodies_elided_signatures_kept() {
        let code =
            "use std::fmt;\n\nfn add(a: i32, b: i32) -> i32 {\n    let s = a + b;\n    s\n}\n";
        let cfg = lang_for("rust").unwrap();
        let out = skeletonize_code(code, &cfg);
        assert!(out.contains("use std::fmt;"), "imports kept");
        assert!(
            out.contains("fn add(a: i32, b: i32) -> i32"),
            "signature kept"
        );
        assert!(out.contains("{ /* … */ }"), "body elided");
        assert!(!out.contains("let s = a + b"), "body content gone");
    }

    #[test]
    fn rust_impl_methods_skeletonized_struct_kept() {
        let code =
            "struct P { x: i32 }\nimpl P {\n    fn get(&self) -> i32 {\n        self.x\n    }\n}\n";
        let out = skeletonize_code(code, &lang_for("rust").unwrap());
        assert!(out.contains("struct P { x: i32 }"), "struct kept");
        assert!(
            out.contains("fn get(&self) -> i32"),
            "method signature kept"
        );
        assert!(out.contains("{ /* … */ }"));
        assert!(!out.contains("self.x"), "method body gone");
    }

    #[test]
    fn javascript_function_body_elided() {
        let code = "function greet(name) {\n  return `hi ${name}`;\n}\n";
        let out = skeletonize_code(code, &lang_for("js").unwrap());
        assert!(out.contains("function greet(name)"));
        assert!(out.contains("{ /* … */ }"));
        assert!(!out.contains("return"), "body gone");
    }

    #[test]
    fn skeletonizes_top_languages() {
        // (tag, code, signature-kept, body-content-gone, stub-present)
        let cases: &[(&str, &str, &str, &str, &str)] = &[
            (
                "ts",
                "function add(a: number): number {\n  const s = a + 1;\n  return s;\n}\n",
                "function add(a: number): number",
                "const s = a + 1",
                BRACE,
            ),
            (
                "python",
                "def add(a):\n    s = a + 1\n    return s\n",
                "def add(a):",
                "s = a + 1",
                "...",
            ),
            (
                "go",
                "func add(a int) int {\n\ts := a + 1\n\treturn s\n}\n",
                "func add(a int) int",
                "s := a + 1",
                BRACE,
            ),
            (
                "java",
                "class A {\n  int add(int a) {\n    int s = a + 1;\n    return s;\n  }\n}\n",
                "int add(int a)",
                "int s = a + 1",
                BRACE,
            ),
            (
                "c",
                "int add(int a) {\n  int s = a + 1;\n  return s;\n}\n",
                "int add(int a)",
                "int s = a + 1",
                BRACE,
            ),
            (
                "cpp",
                "int add(int a) {\n  int s = a + 1;\n  return s;\n}\n",
                "int add(int a)",
                "int s = a + 1",
                BRACE,
            ),
            (
                "csharp",
                "class A {\n  int Add(int a) {\n    int s = a + 1;\n    return s;\n  }\n}\n",
                "int Add(int a)",
                "int s = a + 1",
                BRACE,
            ),
            (
                "kotlin",
                "fun add(a: Int): Int {\n  val s = a + 1\n  return s\n}\n",
                "fun add(a: Int): Int",
                "val s = a + 1",
                BRACE,
            ),
            (
                "swift",
                "func add(a: Int) -> Int {\n  let s = a + 1\n  return s\n}\n",
                "func add(a: Int) -> Int",
                "let s = a + 1",
                BRACE,
            ),
            (
                "zig",
                "fn add(a: i32) i32 {\n  const s = a + 1;\n  return s;\n}\n",
                "fn add(a: i32) i32",
                "const s = a + 1",
                "{}",
            ),
            (
                "ruby",
                "def add(a)\n  s = a + 1\n  s\nend\n",
                "def add(a)",
                "s = a + 1",
                "# …",
            ),
            (
                "php",
                "<?php\nfunction add($a) {\n  $s = $a + 1;\n  return $s;\n}\n",
                "function add($a)",
                "$s = $a + 1",
                BRACE,
            ),
        ];
        for (tag, code, sig, gone, stub) in cases {
            let cfg = lang_for(tag).unwrap_or_else(|| panic!("no lang_for({tag})"));
            let out = skeletonize_code(code, &cfg);
            assert!(
                out.contains(sig),
                "[{tag}] signature kept: {sig:?} not in {out:?}"
            );
            assert!(
                !out.contains(gone),
                "[{tag}] body elided: {gone:?} still in {out:?}"
            );
            assert!(out.contains(stub), "[{tag}] stub: {stub:?} not in {out:?}");
        }
    }

    #[test]
    fn js_arrow_expression_body_kept_block_body_elided() {
        // Concise arrow (expression body) must NOT be brace-stubbed — would be invalid.
        let out = skeletonize_code("const f = (x) => x + 1;\n", &lang_for("js").unwrap());
        assert!(out.contains("x + 1"), "arrow expression body kept");
        // Block-bodied arrow IS elided.
        let out2 = skeletonize_code(
            "const g = (x) => {\n  const y = x + 1;\n  return y;\n};\n",
            &lang_for("js").unwrap(),
        );
        assert!(out2.contains(BRACE), "arrow block body elided");
        assert!(!out2.contains("const y = x + 1"), "block body content gone");
    }

    #[test]
    fn python_minify_is_noop_indentation_significant() {
        let code = "def f():\n    if x:\n        return 1\n";
        let out = minify_code(code, &lang_for("python").unwrap());
        assert_eq!(
            out, code,
            "Python indentation is structure — minify must not touch it"
        );
    }

    #[test]
    fn stage_skeletonizes_fenced_block_in_content() {
        use crate::ir::ProviderKind;
        use crate::pipeline;
        use crate::provider::OpenAiProvider;
        use crate::tokenizer::counter_for;
        use serde_json::json;

        let big_fn = format!(
            "```rust\nfn process() {{\n{}\n}}\n```",
            "    println!(\"step\");\n".repeat(20)
        );
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":big_fn}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(skeleton_stage())];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "skeletonizing a big body reduces tokens"
        );
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(now.contains("fn process()") && now.contains("/* … */"));
        assert!(!now.contains("step"), "repeated body lines gone");
    }

    #[test]
    fn crlf_and_info_string_fences_are_skeletonized() {
        use crate::ir::ProviderKind;
        use crate::pipeline;
        use crate::provider::OpenAiProvider;
        use crate::tokenizer::counter_for;
        use serde_json::json;

        let body_lines = "    let s = a + b;\n    s\n";
        // CRLF fence with an info string after the language tag — previously never matched.
        let content = format!(
            "Look:\r\n```rust title=add.rs\r\nfn add(a: i32, b: i32) -> i32 {{\r\n{body_lines}}}\r\n```"
        );
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(skeleton_stage())];
        let _ = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(
            now.contains("fn add(a: i32, b: i32) -> i32"),
            "signature kept"
        );
        assert!(
            now.contains("/* … */"),
            "CRLF + info-string fence was skeletonized"
        );
        assert!(now.contains("title=add.rs"), "info string preserved");
        assert!(!now.contains("let s = a + b"), "body elided");
    }

    #[test]
    fn fence_with_unmatched_info_tag_is_left_alone() {
        // `c#` is not a whitespace-delimited info string after `c`, so the fence must NOT be
        // misread as plain C — it passes through (lang tag `c#` isn't followed by `\n`).
        let out = FENCE
            .replace_all("```c#\nint x;\n```", |c: &regex::Captures| {
                format!("HIT:{}", &c[1])
            })
            .into_owned();
        assert_eq!(out, "```c#\nint x;\n```", "c# fence not split into c");
    }

    #[test]
    fn unsupported_language_fence_untouched() {
        // Newly-supported languages resolve…
        assert!(lang_for("python").is_some(), "python now supported");
        assert!(lang_for("kotlin").is_some() && lang_for("zig").is_some());
        // …while unknown tags and untagged fences pass through untouched.
        assert!(lang_for("haskell").is_none(), "unsupported → passthrough");
        assert!(lang_for("text").is_none());
        assert!(lang_for("").is_none(), "untagged fence → passthrough");
    }

    #[test]
    fn minify_protects_heredoc_body() {
        // Ruby heredoc — indentation inside the body is part of the string value.
        let ruby = "def greet(name)\n  msg = <<~HEREDOC\n    Hello,   #{name}!\n    Welcome.\n  HEREDOC\n  puts msg\nend\n";
        let cfg = lang_for("ruby").unwrap();
        let out = minify_code(ruby, &cfg);
        assert!(
            out.contains("Hello,   "),
            "heredoc body whitespace must be preserved; got:\n{out}"
        );
        assert!(
            out.contains("Welcome."),
            "heredoc body content must be preserved; got:\n{out}"
        );

        // PHP heredoc — same requirement.
        let php = "<?php\nfunction greet($name) {\n  $msg = <<<EOT\n    Hello,   $name!\n    Welcome.\nEOT;\n  echo $msg;\n}\n";
        let cfg_php = lang_for("php").unwrap();
        let out_php = minify_code(php, &cfg_php);
        assert!(
            out_php.contains("Hello,   "),
            "PHP heredoc body whitespace must be preserved; got:\n{out_php}"
        );
        assert!(
            out_php.contains("Welcome."),
            "PHP heredoc body content must be preserved; got:\n{out_php}"
        );
    }

    // ── Relevance-graded pruning (HCP) ─────────────────────────────────────────────

    /// Run a `SkeletonStage` over a two-message OpenAI request (a prose question + the code
    /// block), returning the rewritten code content. The two-turn shape gives a real query
    /// distinct from the code (the lone-block case is covered separately).
    fn run_skeleton(stage: SkeletonStage, question: &str, code_block: &str) -> String {
        use crate::ir::ProviderKind;
        use crate::pipeline;
        use crate::provider::OpenAiProvider;
        use crate::tokenizer::counter_for;
        use serde_json::json;
        let body = json!({"model":"gpt-4o","messages":[
            {"role":"user","content": code_block},
            {"role":"user","content": question}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(stage)];
        let _ = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        req.get_str("/messages/0/content").unwrap().to_string()
    }

    #[test]
    fn import_tokens_extracts_module_names_rust_python() {
        // The depth-1 import bonus only fires when module names actually resolve; assert the
        // extraction works (rather than silently degrading) for two representative grammars.
        let r = import_tokens(
            "use payments::charge;\nfn f(){}",
            &lang_for("rust").unwrap(),
        );
        assert!(
            r.contains("payments"),
            "[rust] use-declaration module name extracted: {r:?}"
        );
        let p = import_tokens(
            "import payments\nfrom payments import charge\ndef f():\n    pass\n",
            &lang_for("python").unwrap(),
        );
        assert!(
            p.contains("payments"),
            "[py] import module name extracted: {p:?}"
        );
    }

    #[test]
    fn split_subtokens_handles_snake_camel_and_cjk() {
        // snake_case → parts + whole
        let mut s = HashSet::new();
        split_subtokens("load_user_id", &mut s);
        assert!(s.contains("load") && s.contains("user") && s.contains("id"));
        assert!(s.contains("load_user_id"), "whole identifier kept too");
        // camelCase / PascalCase → split on lower→upper
        let mut c = HashSet::new();
        split_subtokens("loadUserId", &mut c);
        assert!(c.contains("load") && c.contains("user") && c.contains("id"));
        let mut p = HashSet::new();
        split_subtokens("HttpServerConfig", &mut p);
        assert!(p.contains("http") && p.contains("server") && p.contains("config"));
        // CJK identifier (no case, no separators) → whole-token only (Unicode-aware,
        // spec rule 5: never assume ASCII).
        let mut j = HashSet::new();
        split_subtokens("ユーザー検索", &mut j);
        assert!(
            j.contains("ユーザー検索"),
            "CJK identifier kept as a whole token: {j:?}"
        );
    }

    #[test]
    fn keep_full_keeps_queried_function_skeletonizes_others() {
        // Query mentions `foo`; the block has three functions. `foo` keeps its body; the
        // other two are skeletonized. The core HCP upgrade.
        let code = "```rust\n\
            fn foo(a: i32) -> i32 {\n    let secret = a * 2;\n    secret + 1\n}\n\n\
            fn bar(b: i32) -> i32 {\n    let other = b - 3;\n    other\n}\n\n\
            fn baz(c: i32) -> i32 {\n    let third = c + 9;\n    third\n}\n```";
        let out = run_skeleton(
            SkeletonStage {
                keep_full_top_k: 1,
                drop_unmatched: false,
                drop_min_body_lines: 8,
            },
            "Can you explain what foo does here?",
            code,
        );
        assert!(
            out.contains("let secret = a * 2"),
            "foo body kept (keep-full)"
        );
        assert!(out.contains("fn bar"), "bar signature kept");
        assert!(out.contains("fn baz"), "baz signature kept");
        assert!(!out.contains("let other = b - 3"), "bar body skeletonized");
        assert!(!out.contains("let third = c + 9"), "baz body skeletonized");
        assert!(
            out.contains("{ /* … */ }"),
            "stub present for the dropped tiers"
        );
    }

    #[test]
    fn no_query_falls_back_to_uniform_skeletonization() {
        // A lone code block (the whole user turn is one fence) → no prose query → uniform
        // skeletonization, exactly today's behavior. Regression guard for the upgrade.
        use crate::ir::ProviderKind;
        use crate::pipeline;
        use crate::provider::OpenAiProvider;
        use crate::tokenizer::counter_for;
        use serde_json::json;
        let code = "```rust\n\
            fn foo(a: i32) -> i32 {\n    let secret = a * 2;\n    secret + 1\n}\n\n\
            fn bar(b: i32) -> i32 {\n    let other = b - 3;\n    other\n}\n```";
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":code}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(skeleton_stage())];
        let _ = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let out = req.get_str("/messages/0/content").unwrap();
        assert!(
            !out.contains("let secret = a * 2") && !out.contains("let other = b - 3"),
            "no query → every body skeletonized (uniform): {out}"
        );
        assert!(
            out.contains("fn foo") && out.contains("fn bar"),
            "signatures kept"
        );
    }

    #[test]
    fn ranking_is_deterministic_across_runs() {
        // Same input, ten runs → byte-identical output. Tie-breaks (equal scores) resolve by
        // position, so there is no run-to-run nondeterminism from HashMap/HashSet iteration.
        let code = "```rust\n\
            fn alpha(x: i32) -> i32 {\n    let a = x + 1;\n    a\n}\n\n\
            fn beta(x: i32) -> i32 {\n    let b = x + 2;\n    b\n}\n\n\
            fn gamma(x: i32) -> i32 {\n    let g = x + 3;\n    g\n}\n```";
        let q = "compare alpha and beta and gamma please";
        let first = run_skeleton(skeleton_stage(), q, code);
        for _ in 0..9 {
            assert_eq!(
                run_skeleton(skeleton_stage(), q, code),
                first,
                "ranking + rewrite must be deterministic"
            );
        }
    }

    #[test]
    fn keep_full_extracts_identifiers_rust_python_ts() {
        // Language coverage for keep-full: the queried function keeps its body in each of
        // Rust, Python, and TypeScript — identifier extraction is grammar-driven, not
        // per-language regex (spec rule 2).
        let rust = "```rust\n\
            fn parse_config(s: &str) -> i32 {\n    let v = s.len();\n    v as i32\n}\n\n\
            fn other_thing(s: &str) -> i32 {\n    let z = s.len() + 7;\n    z as i32\n}\n```";
        let out = run_skeleton(skeleton_stage(), "how does parse_config work", rust);
        assert!(
            out.contains("let v = s.len()"),
            "[rust] parse_config body kept"
        );
        assert!(
            !out.contains("let z = s.len() + 7"),
            "[rust] other body skeletonized"
        );

        let py = "```python\n\
            def parse_config(s):\n    v = len(s)\n    return v\n\n\
            def other_thing(s):\n    z = len(s) + 7\n    return z\n```";
        let out = run_skeleton(skeleton_stage(), "explain parse_config", py);
        assert!(out.contains("v = len(s)"), "[py] parse_config body kept");
        assert!(
            !out.contains("z = len(s) + 7"),
            "[py] other body skeletonized"
        );
        assert!(out.contains("..."), "[py] stub present");

        let ts = "```ts\n\
            function parseConfig(s: string): number {\n  const v = s.length;\n  return v;\n}\n\n\
            function otherThing(s: string): number {\n  const z = s.length + 7;\n  return z;\n}\n```";
        // Query uses snake_case `parse_config`; identifier is camelCase `parseConfig` — the
        // shared subtoken split must bridge them (parse, config).
        let out = run_skeleton(skeleton_stage(), "what does parse_config return", ts);
        assert!(
            out.contains("const v = s.length"),
            "[ts] parseConfig body kept (camel↔snake match)"
        );
        assert!(
            !out.contains("const z = s.length + 7"),
            "[ts] other body skeletonized"
        );
    }

    #[test]
    fn import_bonus_lifts_dependency_block() {
        // Two blocks. Block A imports `payments`; the query is about payments. Block B's
        // function shares no query term lexically but imports `payments` too — the depth-1
        // import bonus must NOT promote it (zero base overlap stays zero), while a tie among
        // overlapping functions across the two dependency-linked blocks is bonus-broken.
        // Here we assert the graceful, conservative property: a zero-overlap function never
        // becomes keep-full from the bonus alone.
        let code = "```rust\nuse payments::charge;\n\
            fn run_charge(a: i32) -> i32 {\n    let amt = a * 2;\n    charge(amt)\n}\n```\n\n\
            ```rust\nuse payments::refund;\n\
            fn unrelated_helper(a: i32) -> i32 {\n    let q = a + 99;\n    q\n}\n```";
        let out = run_skeleton(
            SkeletonStage {
                keep_full_top_k: 1,
                drop_unmatched: false,
                drop_min_body_lines: 8,
            },
            "how is the payments charge computed",
            code,
        );
        // The overlapping function (run_charge: shares "charge") is kept full…
        assert!(
            out.contains("let amt = a * 2"),
            "queried function kept full"
        );
        // …and the zero-overlap helper is skeletonized despite sharing an import module name.
        assert!(
            !out.contains("let q = a + 99"),
            "zero-overlap function not promoted by the import bonus alone"
        );
    }

    #[test]
    fn drop_tier_removes_signature_when_enabled() {
        // With `drop_unmatched`, a zero-overlap function whose body exceeds the line floor is
        // removed entirely (signature too); the queried function survives full. OFF by default
        // is covered by every other test (signatures always survive there).
        let big_unrelated = (0..10)
            .map(|i| format!("    let x{i} = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!(
            "```rust\n\
            fn foo(a: i32) -> i32 {{\n    let keep = a + 1;\n    keep\n}}\n\n\
            fn legacy_unused(z: i32) -> i32 {{\n{big_unrelated}\n    z\n}}\n```"
        );
        let out = run_skeleton(
            SkeletonStage {
                keep_full_top_k: 1,
                drop_unmatched: true,
                drop_min_body_lines: 5,
            },
            "explain foo",
            &code,
        );
        assert!(out.contains("let keep = a + 1"), "queried foo kept full");
        assert!(
            !out.contains("fn legacy_unused"),
            "zero-overlap large function dropped signature and all: {out}"
        );
    }

    #[test]
    fn drop_tier_off_by_default_keeps_signature() {
        // Same input as above but default config (drop off): the unrelated function is
        // skeletonized, never dropped — its signature survives (conservative default).
        let big_unrelated = (0..10)
            .map(|i| format!("    let x{i} = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!(
            "```rust\n\
            fn foo(a: i32) -> i32 {{\n    let keep = a + 1;\n    keep\n}}\n\n\
            fn legacy_unused(z: i32) -> i32 {{\n{big_unrelated}\n    z\n}}\n```"
        );
        let out = run_skeleton(skeleton_stage(), "explain foo", &code);
        assert!(
            out.contains("fn legacy_unused"),
            "default keeps the signature tier (no drop): {out}"
        );
        assert!(!out.contains("let x9 = 9"), "but its body is skeletonized");
    }

    #[test]
    fn graded_skeletonization_still_meets_savings_threshold() {
        // The relevance-graded path must still hit the stage's savings bar on a realistic
        // many-function block where only one function is relevant (most bodies elided).
        let mut fns = String::from("fn target(a: i32) -> i32 {\n    let r = a + 1;\n    r\n}\n\n");
        for i in 0..12 {
            // Realistic multi-line helper bodies — skeletonization's win scales with body
            // size, so a one-liner body is an unrepresentative fixture (see cli-testing.md).
            fns.push_str(&format!("fn helper_{i}(a: i32) -> i32 {{\n"));
            for j in 0..8 {
                fns.push_str(&format!(
                    "    let v{i}_{j} = a * {i} + {j} * 7 - 3;\n    println!(\"step {i} {j} = {{}}\", v{i}_{j});\n"
                ));
            }
            fns.push_str(&format!("    v{i}_0\n}}\n\n"));
        }
        let code = format!("```rust\n{fns}```");
        let out = run_skeleton(skeleton_stage(), "describe the target function", &code);
        let saved = 100.0 - (count_tokens(&out) as f64 / count_tokens(&code) as f64 * 100.0);
        assert!(
            saved >= 60.0,
            "graded skeletonization should keep ≥60% savings, got {saved:.1}%"
        );
        assert!(
            out.contains("let r = a + 1"),
            "the one relevant body survives"
        );
    }
}
