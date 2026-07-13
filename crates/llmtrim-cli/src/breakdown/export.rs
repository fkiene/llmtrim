//! Machine-readable views of the per-source breakdown, for the `status` export modes
//! (`--json` / `--csv` / period reports). These surface the same data the TUI shows —
//! every session and the corpus-wide per-source cost down to each MCP server — so a
//! script can consume what the interactive view displays.
//!
//! Each public entry point opens the ledger once and delegates to a pure formatter
//! (`*_value` / `*_csv_rows` / `*_block`) that takes already-loaded rows, so the shaping
//! is unit-tested without touching a database.

use anyhow::Result;
use serde_json::{Value, json};

use super::db::{BreakdownDb, CostRow, SessionRow};
use crate::ui::{self, Tone};

/// The breakdown block merged into `status --json`: every session plus the corpus-wide
/// per-source cost. Returns `None` when there's no breakdown data yet (nothing recorded),
/// so the export simply omits the section rather than showing empty arrays.
pub fn breakdown_json() -> Option<Value> {
    let db = BreakdownDb::open().ok()?;
    breakdown_value(&db.sessions().ok()?, &db.all_sources().ok()?)
}

/// Pure shaper: `None` when there are no sessions (so the export omits the section).
fn breakdown_value(sessions: &[SessionRow], sources: &[CostRow]) -> Option<Value> {
    if sessions.is_empty() {
        return None;
    }
    Some(json!({
        "sessions": sessions.iter().map(session_json).collect::<Vec<_>>(),
        "sources": sources.iter().map(source_json).collect::<Vec<_>>(),
    }))
}

fn session_json(s: &SessionRow) -> Value {
    json!({
        "session_id": s.session_id,
        "agent": s.agent,
        "project": s.project,
        "name": s.session_name,
        "turns": s.turns,
        "tokens": s.tokens,
        "cache_hit_pct": (s.cache_hit * 100.0).round(),
        "bill_usd": s.bill_usd(),
    })
}

fn source_json(c: &CostRow) -> Value {
    json!({
        "group": c.group_label,
        "label": c.label,
        "mcp_server": c.mcp_server,
        "tool": c.tool_name,
        "usd": c.usd,
        "read_usd": c.read_usd,
        "write_usd": c.write_usd,
        "new_usd": c.new_usd,
    })
}

/// Per-session breakdown as CSV (one row per session) — the detail the `--csv` export adds
/// over the plain time series. `None` when no data is available.
pub fn sessions_csv() -> Option<String> {
    let db = BreakdownDb::open().ok()?;
    let sessions = db.sessions().ok()?;
    if sessions.is_empty() {
        return None;
    }
    Some(sessions_csv_rows(&sessions))
}

/// Pure shaper: render the per-session CSV (header + one row each).
fn sessions_csv_rows(sessions: &[SessionRow]) -> String {
    let mut o = String::from("agent,project,session,turns,tokens,cache_hit_pct,bill_usd\n");
    for s in sessions {
        let name = s.session_name.as_deref().unwrap_or(&s.session_id);
        o.push_str(&format!(
            "{},{},{},{},{},{:.0},{:.4}\n",
            csv_field(&s.agent),
            csv_field(s.project.as_deref().unwrap_or("")),
            csv_field(name),
            s.turns,
            s.tokens,
            s.cache_hit * 100.0,
            s.bill_usd(),
        ));
    }
    o
}

/// Escape a CSV field: quote and double internal quotes when it contains a comma, quote,
/// or a line break (RFC 4180), so a project path with a comma or CR/LF can't shift columns.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// A compact "top sources by cost" block appended to the text period reports, mirroring
/// the TUI's cost pane. `color` follows the caller's TTY/`NO_COLOR` decision.
pub fn top_sources_report(color: bool, max: usize) -> Result<String> {
    let db = BreakdownDb::open()?;
    Ok(top_sources_block(color, &db.all_sources()?, max))
}

/// Pure shaper: the top-N sources by cost as a text block (empty when there are none).
fn top_sources_block(color: bool, sources: &[CostRow], max: usize) -> String {
    if sources.is_empty() {
        return String::new();
    }
    let mut out = format!(
        "\n{}\n",
        ui::paint(color, Tone::Bold, "TOP SOURCES BY COST")
    );
    for c in sources.iter().take(max) {
        let label = match (&c.mcp_server, &c.tool_name) {
            (Some(srv), Some(tool)) => format!("{} · {srv} · {tool}", c.label),
            (Some(srv), None) => format!("{} · {srv}", c.label),
            (None, Some(tool)) => format!("{} · {tool}", c.label),
            (None, None) => c.label.clone(),
        };
        out.push_str(&format!(
            "  {:<40} {}\n",
            label,
            ui::paint(color, Tone::Accent, &format!("${:.2}", c.usd))
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(name: Option<&str>, project: Option<&str>) -> SessionRow {
        SessionRow {
            session_id: "abc123def456".to_string(),
            cc_session_id: None,
            agent: "claude-code".to_string(),
            project: project.map(str::to_string),
            session_name: name.map(str::to_string),
            turns: 4,
            tokens: 12_345,
            cache_hit: 0.5,
            bill_micros: 1_230_000,
            input_before: 1000,
            input_after: 600,
            last_ts: "2026-06-19T00:00:00+00:00".to_string(),
            last_sub_provider: None,
            last_model: None,
            last_cache_hit: None,
            last_input_tokens: 0,
        }
    }

    fn source(label: &str, mcp: Option<&str>, tool: Option<&str>, usd: f64) -> CostRow {
        CostRow {
            group_label: "Static".to_string(),
            label: label.to_string(),
            mcp_server: mcp.map(str::to_string),
            tool_name: tool.map(str::to_string),
            usd,
            read_usd: usd,
            write_usd: 0.0,
            new_usd: 0.0,
        }
    }

    #[test]
    fn breakdown_value_is_none_without_sessions() {
        assert!(breakdown_value(&[], &[]).is_none());
    }

    #[test]
    fn breakdown_value_shapes_sessions_and_sources() {
        let v = breakdown_value(
            &[session(Some("my work"), Some("/proj"))],
            &[source(
                "MCP tools",
                Some("github"),
                Some("create_issue"),
                0.10,
            )],
        )
        .unwrap();
        assert_eq!(v["sessions"][0]["agent"], "claude-code");
        assert_eq!(v["sessions"][0]["turns"], 4);
        assert_eq!(v["sessions"][0]["cache_hit_pct"], 50.0);
        assert_eq!(v["sources"][0]["mcp_server"], "github");
        assert_eq!(v["sources"][0]["tool"], "create_issue");
    }

    #[test]
    fn csv_quotes_fields_with_separators() {
        // A project path with a comma and a name with a quote must be quoted/escaped.
        let csv = sessions_csv_rows(&[session(Some("a\"b"), Some("/x,y"))]);
        let line = csv.lines().nth(1).unwrap();
        assert!(line.contains("\"/x,y\""));
        assert!(line.contains("\"a\"\"b\""));
        // Numeric columns render with the documented precision.
        assert!(line.ends_with("1.2300"));
    }

    #[test]
    fn csv_falls_back_to_session_id_when_unnamed() {
        let csv = sessions_csv_rows(&[session(None, None)]);
        assert!(csv.lines().nth(1).unwrap().contains("abc123def456"));
    }

    #[test]
    fn top_sources_block_labels_and_truncates() {
        let sources = vec![
            source("MCP tools", Some("github"), Some("create_issue"), 0.50),
            source("System prompt", None, None, 0.30),
            source("user text", None, None, 0.10),
        ];
        let block = top_sources_block(false, &sources, 2);
        assert!(block.contains("TOP SOURCES BY COST"));
        assert!(block.contains("MCP tools · github · create_issue"));
        assert!(block.contains("$0.50"));
        // Truncated to 2 → the third source is absent.
        assert!(!block.contains("user text"));
    }

    #[test]
    fn top_sources_block_empty_when_no_sources() {
        assert!(top_sources_block(false, &[], 5).is_empty());
    }
}
