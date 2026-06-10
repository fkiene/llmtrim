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
//! Lossy (bodies are gone), so off by default and InputTokens-gated. The token gate
//! reverts it if a block doesn't actually shrink.

use once_cell::sync::Lazy;
use regex::Regex;
use tree_sitter::{Language, Parser};

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

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

/// Replace each outermost function body in `code` with `{ /* … */ }`. Returns the
/// input unchanged if the grammar can't be loaded or parsing fails (never block).
fn skeletonize_code(code: &str, cfg: &LangCfg) -> String {
    let mut parser = Parser::new();
    if parser.set_language(&(cfg.language)()).is_err() {
        return code.to_string();
    }
    let Some(tree) = parser.parse(code.as_bytes(), None) else {
        return code.to_string();
    };

    // Collect outermost function-body byte ranges: stop descending at a function so
    // nested bodies are subsumed by the outer replacement.
    let mut bodies: Vec<(usize, usize)> = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if cfg.fn_kinds.contains(&node.kind())
            && let Some(body) = body_node(&node, cfg)
        {
            bodies.push((body.start_byte(), body.end_byte()));
            continue;
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }

    // Splice from the back so earlier byte offsets stay valid.
    bodies.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
    let mut out = code.to_string();
    for (start, end) in bodies {
        if start < end && end <= out.len() {
            out.replace_range(start..end, cfg.placeholder);
        }
    }
    out
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

pub struct SkeletonStage;

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
        rewrite_fenced_code(req, provider, skeletonize_code);
        Ok(())
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
        if n.kind().contains("string") {
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
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SkeletonStage)];
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
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SkeletonStage)];
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
}
