//! `llmtrim discover` — scan the before/after capture corpus and report WHERE
//! compressible tokens are still escaping compression, so the next compression target
//! is chosen from real traffic instead of guesswork.
//!
//! Read-only over the corpus (never mutates or deletes a capture). For each capture it
//! re-buckets the request's token surface — the exact segments behind the proxy's
//! `input_*` counts (`provider::content_text_pointers` + the `tools` schema) — by block
//! kind (system / user / assistant / tool_result / tool_call_args / document /
//! tool_schema) and, for tool results, by tool name. It buckets BOTH the `before` and
//! `after` bodies, so each row shows the residual still in the compressed request AND how
//! much compression already bit out of that bucket (the before→after delta). The ranked
//! table then surfaces the buckets holding the most uncompressed residual.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use llmtrim_core::ir::{ProviderKind, Request};
use llmtrim_core::provider::{self, Provider, Role};
use llmtrim_core::tokenizer::{self, TokenCounter};
use serde::Serialize;
use serde_json::Value;

use crate::ui;

/// One residual bucket: a block kind, optionally narrowed to a tool name (tool results).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct BucketKey {
    kind: String,
    /// Tool name for `tool_result` rows; `None` for every other kind.
    tool: Option<String>,
}

/// Accumulated counts for a bucket, summed across the whole corpus.
#[derive(Default, Clone, Serialize)]
struct BucketStat {
    /// Tokens surviving in the compressed (`after`) requests — the residual.
    residual_tokens: u64,
    /// Bytes (UTF-8) surviving in the compressed requests.
    residual_bytes: u64,
    /// Tokens in the original (`before`) requests, same bucket.
    before_tokens: u64,
    /// Number of captures that contributed at least one segment to this bucket.
    captures: u64,
}

/// The full report: corpus totals (from the stored proxy counts) plus the ranked buckets.
#[derive(Serialize)]
struct Report {
    captures: u64,
    skipped: u64,
    /// Input tokens before compression, summed from the captures' stored `input_before`.
    input_before_tokens: u64,
    /// Input tokens after compression, summed from the captures' stored `input_after`.
    input_after_tokens: u64,
    /// Realized savings already happening across the corpus, percent.
    realized_savings_pct: f64,
    /// Total residual tokens across every bucket (the denominator for each row's share).
    total_residual_tokens: u64,
    rows: Vec<ReportRow>,
}

#[derive(Serialize)]
struct ReportRow {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool: Option<String>,
    residual_tokens: u64,
    residual_bytes: u64,
    /// Share of the corpus-wide residual this bucket holds, percent.
    residual_share_pct: f64,
    /// Before→after token reduction already applied to this bucket, percent (negative
    /// would mean a stage grew it, e.g. an injected legend).
    compressed_pct: f64,
    captures: u64,
}

/// Run the discover scan. `dir` defaults to the capture corpus (`$LLMTRIM_CAPTURE_DIR`
/// or `~/.llmtrim/capture`). `by_tool` expands `tool_result` into per-tool rows.
pub fn run(dir: Option<PathBuf>, json: bool, by_tool: bool, limit: Option<usize>) -> Result<()> {
    let dir = resolve_dir(dir)?;
    let report = scan(&dir, by_tool, limit)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("serialize discover report")?
        );
    } else {
        print!("{}", render(&report, by_tool, ui::color_stdout()));
    }
    Ok(())
}

/// Resolve the corpus directory: explicit `--dir`, else `$LLMTRIM_CAPTURE_DIR`, else
/// `~/.llmtrim/capture`.
fn resolve_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(d) = dir {
        return Ok(d);
    }
    if let Ok(d) = std::env::var("LLMTRIM_CAPTURE_DIR")
        && !d.is_empty()
    {
        return Ok(PathBuf::from(d));
    }
    Ok(crate::daemon::home_dir()?.join("capture"))
}

/// Walk every `*.json` capture in `dir`, accumulate buckets, and build the report.
/// Malformed, empty, or unparseable captures are counted as skipped, never fatal.
fn scan(dir: &Path, by_tool: bool, limit: Option<usize>) -> Result<Report> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("read capture dir {}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort(); // chronological (timestamp-prefixed names) and deterministic
    if let Some(n) = limit {
        paths.truncate(n);
    }

    let mut buckets: BTreeMap<BucketKey, BucketStat> = BTreeMap::new();
    let mut captures = 0u64;
    let mut skipped = 0u64;
    let mut input_before = 0u64;
    let mut input_after = 0u64;

    for path in &paths {
        match accumulate(path, by_tool, &mut buckets) {
            Ok(Some((before, after))) => {
                captures += 1;
                input_before += before;
                input_after += after;
            }
            Ok(None) => skipped += 1,
            // A read/parse error on one capture (partial write, I/O fault) is never fatal,
            // but surface it so a stale mount or unreadable corpus isn't a silent empty report.
            Err(e) => {
                eprintln!("llmtrim discover: skipping {}: {e:#}", path.display());
                skipped += 1;
            }
        }
    }

    let total_residual: u64 = buckets.values().map(|s| s.residual_tokens).sum();
    let mut rows: Vec<ReportRow> = buckets
        .into_iter()
        .map(|(k, s)| ReportRow {
            kind: k.kind,
            tool: k.tool,
            residual_tokens: s.residual_tokens,
            residual_bytes: s.residual_bytes,
            residual_share_pct: pct(s.residual_tokens, total_residual),
            compressed_pct: ui::saved_pct(s.before_tokens as f64, s.residual_tokens as f64),
            captures: s.captures,
        })
        .collect();
    // Ranked "where the next gains are": most residual first.
    rows.sort_by(|a, b| {
        b.residual_tokens
            .cmp(&a.residual_tokens)
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.tool.cmp(&b.tool))
    });

    Ok(Report {
        captures,
        skipped,
        input_before_tokens: input_before,
        input_after_tokens: input_after,
        realized_savings_pct: ui::saved_pct(input_before as f64, input_after as f64),
        total_residual_tokens: total_residual,
        rows,
    })
}

/// Read one capture, bucket its `before`/`after` bodies into `buckets`, and return the
/// stored `(input_before, input_after)` token counts. `Ok(None)` for a capture that can't
/// be read as the expected envelope (empty/partial file, missing fields).
fn accumulate(
    path: &Path,
    by_tool: bool,
    buckets: &mut BTreeMap<BucketKey, BucketStat>,
) -> Result<Option<(u64, u64)>> {
    let raw = std::fs::read_to_string(path)?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let env: Value = serde_json::from_str(&raw)?;
    let (Some(before_str), Some(after_str)) = (
        env.get("before").and_then(Value::as_str),
        env.get("after").and_then(Value::as_str),
    ) else {
        return Ok(None);
    };

    let before: Value = serde_json::from_str(before_str)?;
    let after: Value = serde_json::from_str(after_str)?;

    let kind = provider_kind(&env, &after);
    let counter = tokenizer::counter_for(kind, env.get("model").and_then(Value::as_str))
        .context("build token counter")?;

    // Bucket the after body (residual) and the before body (delta context), then merge.
    let after_b = bucketize(kind, &after, counter.as_ref(), by_tool);
    let before_b = bucketize(kind, &before, counter.as_ref(), by_tool);

    for (key, (tokens, bytes)) in &after_b {
        let stat = buckets.entry(key.clone()).or_default();
        stat.residual_tokens += tokens;
        stat.residual_bytes += bytes;
        stat.captures += 1;
    }
    for (key, (tokens, _bytes)) in &before_b {
        buckets.entry(key.clone()).or_default().before_tokens += tokens;
    }

    let input_before = env.get("input_before").and_then(Value::as_u64).unwrap_or(0);
    let input_after = env.get("input_after").and_then(Value::as_u64).unwrap_or(0);
    Ok(Some((input_before, input_after)))
}

/// Provider kind for a capture: trust the stored `provider` label, else sniff the body.
fn provider_kind(env: &Value, body: &Value) -> ProviderKind {
    env.get("provider")
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<ProviderKind>().ok())
        .or_else(|| provider::detect(body))
        .unwrap_or(ProviderKind::OpenAi)
}

/// Bucket every compressible text segment of one request body by block kind (+ tool name
/// for tool results). Keyed identically to the corpus accumulator. Reuses the core's
/// `content_text_pointers` surface so the buckets reconstruct the proxy's content-token
/// count; the `tools` schema (also counted by the proxy) is added as its own bucket.
fn bucketize(
    kind: ProviderKind,
    body: &Value,
    counter: &dyn TokenCounter,
    by_tool: bool,
) -> BTreeMap<BucketKey, (u64, u64)> {
    let req = Request::from_value(kind, body.clone());
    let prov = provider::for_kind(kind);
    let id2name = tool_id_names(body);
    let mut out: BTreeMap<BucketKey, (u64, u64)> = BTreeMap::new();

    for ptr in prov.content_text_pointers(&req) {
        let Some(text) = req.get_str(&ptr) else {
            continue;
        };
        let (kind_label, tool) = classify(&req, prov.as_ref(), &ptr, &id2name);
        let key = bucket_key(kind_label, tool, by_tool);
        let slot = out.entry(key).or_default();
        slot.0 += counter.count(text) as u64;
        slot.1 += text.len() as u64;
    }

    // Tool schemas: the `tools` array is resent every call and counted by the proxy.
    if let Some(tools) = body.get("tools").filter(|t| !t.is_null()) {
        let s = tools.to_string();
        let slot = out
            .entry(BucketKey {
                kind: "tool_schema".to_string(),
                tool: None,
            })
            .or_default();
        slot.0 += counter.count(&s) as u64;
        slot.1 += s.len() as u64;
    }
    out
}

/// Build the bucket key, collapsing a tool name into the plain `tool_result` bucket unless
/// `by_tool` is set (so the default table stays compact).
fn bucket_key(kind: &str, tool: Option<String>, by_tool: bool) -> BucketKey {
    let tool = if by_tool && kind == "tool_result" {
        tool.or_else(|| Some("unknown".to_string()))
    } else {
        None
    };
    BucketKey {
        kind: kind.to_string(),
        tool,
    }
}

/// Map a content pointer to its block kind and (for tool results / tool calls) tool name.
fn classify(
    req: &Request,
    prov: &dyn Provider,
    ptr: &str,
    id2name: &BTreeMap<String, String>,
) -> (&'static str, Option<String>) {
    let raw = req.raw();
    let segs: Vec<&str> = ptr.split('/').collect();

    // `/messages/{i}/...` — the OpenAI Chat + Anthropic shared shape, where block kind and
    // tool identity live in the message structure.
    if segs.get(1) == Some(&"messages")
        && let Some(i) = segs.get(2)
    {
        // OpenAI assistant tool-call arguments: /messages/{i}/tool_calls/{j}/function/arguments
        if segs.get(3) == Some(&"tool_calls") {
            let name = segs.get(4).and_then(|j| {
                raw.pointer(&format!("/messages/{i}/tool_calls/{j}/function/name"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
            return ("tool_call_args", name);
        }
        if segs.get(3) == Some(&"content") {
            // String content (`/messages/{i}/content`): a tool message is a tool_result.
            if segs.len() == 4 {
                if role_label(prov, req, ptr) == "tool" {
                    let name = raw
                        .pointer(&format!("/messages/{i}/tool_call_id"))
                        .and_then(Value::as_str)
                        .and_then(|id| id2name.get(id))
                        .cloned();
                    return ("tool_result", name);
                }
            } else if let Some(j) = segs.get(4) {
                // Array content: classify by the block's own type.
                let block = raw.pointer(&format!("/messages/{i}/content/{j}"));
                match block.and_then(|b| b.get("type")).and_then(Value::as_str) {
                    Some("tool_result") => {
                        let name = block
                            .and_then(|b| b.get("tool_use_id"))
                            .and_then(Value::as_str)
                            .and_then(|id| id2name.get(id))
                            .cloned();
                        return ("tool_result", name);
                    }
                    Some("tool_use") => {
                        let name = block
                            .and_then(|b| b.get("name"))
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        return ("tool_call_args", name);
                    }
                    Some("document") => return ("document", None),
                    _ => {}
                }
            }
        }
    }
    (role_label(prov, req, ptr), None)
}

/// Normalized role label for a content pointer (top-level system text → `system`).
fn role_label(prov: &dyn Provider, req: &Request, ptr: &str) -> &'static str {
    match prov.role_at(req, ptr) {
        None | Some(Role::System) => "system",
        Some(Role::User) => "user",
        Some(Role::Assistant) => "assistant",
        Some(Role::Tool) => "tool_result",
    }
}

/// Collect `tool_use_id`/`tool_call_id` → tool-name mappings from one request body, so a
/// tool result can be attributed to the tool that produced it. Covers Anthropic `tool_use`
/// blocks and OpenAI `tool_calls`.
fn tool_id_names(body: &Value) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return map;
    };
    for msg in messages {
        if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("tool_use")
                    && let (Some(id), Some(name)) = (
                        b.get("id").and_then(Value::as_str),
                        b.get("name").and_then(Value::as_str),
                    )
                {
                    map.insert(id.to_string(), name.to_string());
                }
            }
        }
        if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for c in calls {
                if let (Some(id), Some(name)) = (
                    c.get("id").and_then(Value::as_str),
                    c.pointer("/function/name").and_then(Value::as_str),
                ) {
                    map.insert(id.to_string(), name.to_string());
                }
            }
        }
    }
    map
}

/// `part / whole` as a percentage; 0 when `whole` is 0.
fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

/// Human report: a header line of corpus totals, then the ranked residual table.
fn render(report: &Report, by_tool: bool, color: bool) -> String {
    let mut out = String::new();
    out.push_str(&ui::panel(
        color,
        "discover",
        &[format!(
            "{} captures ({} skipped)   input {} → {} tokens   realized savings {:.1}%   residual {} tokens",
            ui::commas(report.captures as i64),
            ui::commas(report.skipped as i64),
            ui::commas(report.input_before_tokens as i64),
            ui::commas(report.input_after_tokens as i64),
            report.realized_savings_pct,
            ui::commas(report.total_residual_tokens as i64),
        )],
    ));
    out.push('\n');

    let label = if by_tool { "bucket" } else { "kind" };
    let mut t = ui::table(
        color,
        &[
            label,
            "residual tok",
            "share%",
            "compressed%",
            "bytes",
            "captures",
        ],
    );
    for r in &report.rows {
        let name = match &r.tool {
            Some(tool) => format!("{}:{tool}", r.kind),
            None => r.kind.clone(),
        };
        t.add_row(vec![
            name,
            ui::commas(r.residual_tokens as i64),
            format!("{:.1}", r.residual_share_pct),
            format!("{:.1}", r.compressed_pct),
            ui::human(r.residual_bytes as i64),
            ui::commas(r.captures as i64),
        ]);
    }
    out.push_str(&t.to_string());
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/discover")
    }

    #[test]
    fn skips_empty_and_malformed_captures() {
        // The corpus has four files: two valid captures, one empty, one broken JSON.
        let report = scan(&fixtures(), false, None).unwrap();
        assert_eq!(report.captures, 2);
        assert_eq!(report.skipped, 2);
    }

    #[test]
    fn buckets_reconstruct_stored_input_after() {
        // The bucketed residual must equal the proxy's summed `input_after` — the buckets
        // cover exactly the token surface behind that count. Allow a small tolerance for
        // the tools-array stringify vs. the proxy's own counting path.
        let report = scan(&fixtures(), false, None).unwrap();
        let after = report.input_after_tokens as i64;
        let residual = report.total_residual_tokens as i64;
        assert!(
            (after - residual).abs() <= after / 10,
            "residual {residual} should track input_after {after}"
        );
    }

    #[test]
    fn attributes_tool_results_to_their_tool() {
        let report = scan(&fixtures(), true, None).unwrap();
        // Both captures call Bash and feed its output back as a tool_result.
        let bash = report
            .rows
            .iter()
            .find(|r| r.kind == "tool_result" && r.tool.as_deref() == Some("Bash"));
        assert!(bash.is_some(), "expected a tool_result:Bash bucket");
        assert!(bash.unwrap().residual_tokens > 0);
    }

    #[test]
    fn compressed_pct_reflects_before_to_after() {
        // The system prompt was shortened in both captures, so the system bucket shows a
        // positive before→after reduction; the untouched tool_result bucket shows ~0.
        let report = scan(&fixtures(), false, None).unwrap();
        let system = report.rows.iter().find(|r| r.kind == "system").unwrap();
        assert!(
            system.compressed_pct > 0.0,
            "system should show compression, got {}",
            system.compressed_pct
        );
    }

    #[test]
    fn renders_ranked_human_table() {
        let report = scan(&fixtures(), true, None).unwrap();
        let out = render(&report, true, false); // no color for a stable assertion
        // Header line carries the corpus totals.
        assert!(out.contains("2 captures"), "header totals:\n{out}");
        assert!(out.contains("realized savings"), "header savings:\n{out}");
        // The ranked table names the per-tool bucket and the column header.
        assert!(out.contains("tool_result:Bash"), "by-tool row:\n{out}");
        assert!(out.contains("residual tok"), "table header:\n{out}");
        // Rows are ranked by residual descending — the first data row holds the most.
        let first = report.rows.first().unwrap();
        assert!(
            report
                .rows
                .iter()
                .all(|r| r.residual_tokens <= first.residual_tokens),
            "rows must be sorted by residual descending"
        );
    }
}
