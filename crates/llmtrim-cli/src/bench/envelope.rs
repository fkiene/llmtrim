//! Shared result envelope for every benchmark axis.
//!
//! Each axis (quality, suite, agent, latency, compare) writes a different result body,
//! but they all share one outer shape so any reader can identify the schema, the code
//! that produced it, and when. The body is carried verbatim under `result`; readers that
//! predate the envelope can fall back to treating a bare body as the result.

use serde_json::{Value, json};

/// Wrap a result body in the standard envelope.
///
/// `schema` is the axis tag (e.g. `"quality-v1"`); it is namespaced under
/// `llmtrim-bench/`. `meta` carries axis-specific run parameters; `result` is the body.
pub fn wrap(schema: &str, meta: Value, result: Value) -> Value {
    json!({
        "schema": format!("llmtrim-bench/{schema}"),
        "produced_at": now_rfc3339(),
        "commit": git_commit(),
        "llmtrim_version": env!("CARGO_PKG_VERSION"),
        "meta": meta,
        "result": result,
    })
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Short commit of the working tree, or `"unknown"` outside a git checkout.
fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}
