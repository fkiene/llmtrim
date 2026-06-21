//! Read-only query layer for the cost-breakdown TUI.
//!
//! Opens the same `tracking.db` the proxy writes to (WAL mode lets a reader and the
//! daemon's writer proceed together) and serves the aggregates the two screens need:
//! the session list for the Sessions tab, and the per-source occupancy + cost rows for
//! the Detail drill-down. All dollar figures are computed in SQL from each turn's frozen
//! rates, so a historical session always prices at what it actually cost.

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

/// One session aggregate for the Sessions tab tree (grouped agent → project → session
/// in the UI; this row is the leaf).
#[derive(Debug, Clone)]
pub struct SessionRow {
    pub session_id: String,
    pub agent: String,
    pub project: Option<String>,
    pub session_name: Option<String>,
    pub turns: i64,
    /// All tokens seen this session (fresh + cache read + cache write + output).
    pub tokens: i64,
    /// Cache-hit fraction: cache-read over total input (0.0 when no input billed).
    pub cache_hit: f64,
    /// Total bill in micro-USD (frozen per turn, summed).
    pub bill_micros: i64,
    /// Input tokens before / after compression, summed — the session's savings.
    pub input_before: i64,
    pub input_after: i64,
    /// rfc3339 timestamp of the most recent turn, for sorting newest-first.
    pub last_ts: String,
}

impl SessionRow {
    pub fn bill_usd(&self) -> f64 {
        self.bill_micros as f64 / 1_000_000.0
    }

    /// Percentage of input tokens llmtrim trimmed this session (0 when nothing measured).
    pub fn saved_pct(&self) -> f64 {
        if self.input_before > 0 {
            (self.input_before - self.input_after).max(0) as f64 / self.input_before as f64 * 100.0
        } else {
            0.0
        }
    }
}

/// One per-source occupancy row of the latest turn (Detail, top pane).
#[derive(Debug, Clone)]
pub struct OccupancyRow {
    pub group_label: String,
    pub label: String,
    pub mcp_server: Option<String>,
    pub tool_name: Option<String>,
    pub tokens: i64,
}

/// One per-source cost row aggregated over a session (Detail, bottom pane). Dollars are
/// already priced from the turns' frozen rates.
#[derive(Debug, Clone)]
pub struct CostRow {
    pub group_label: String,
    pub label: String,
    pub mcp_server: Option<String>,
    pub tool_name: Option<String>,
    /// Total cost of this source ($).
    pub usd: f64,
    /// Cache-read share ($).
    pub read_usd: f64,
    /// Cache-write share ($).
    pub write_usd: f64,
    /// Fresh-input + output share ($) — the "new" spend.
    pub new_usd: f64,
}

pub struct BreakdownDb {
    conn: Connection,
}

impl BreakdownDb {
    /// Open the ledger for reading. Goes through [`Tracker`], which creates the file if
    /// absent (fresh install, before any proxy run — the TUI then just shows no sessions),
    /// runs the schema migration, and sets WAL + a busy timeout. WAL lets this reader run
    /// alongside the daemon's writer without blocking it; opening read-write (we only ever
    /// `SELECT`) sidesteps the read-only-on-WAL pitfalls of a bare `SQLITE_OPEN_READ_ONLY`.
    pub fn open() -> Result<Self> {
        Ok(Self {
            conn: crate::tracking::Tracker::open_reader()?.into_connection(),
        })
    }

    /// Test/embedding seam: wrap an already-open connection (e.g. an in-memory ledger).
    pub fn from_connection(conn: Connection) -> Self {
        Self { conn }
    }

    /// All sessions with at least one recorded turn, newest activity first.
    pub fn sessions(&self) -> Result<Vec<SessionRow>> {
        let mut stmt = self
            .conn
            .prepare(
                // One pass: rank each turn within its session (a named row first, then newest)
                // so the latest `session_name` is `rn = 1`, then aggregate — instead of a
                // correlated subquery that re-scanned the table once per session group.
                "WITH ranked AS (
                     SELECT session_id, agent, project, session_name, ts, id,
                            fresh_input, cache_read, cache_write, output_tok,
                            bill_micros, input_before, input_after,
                            ROW_NUMBER() OVER (
                                PARTITION BY session_id, agent, project
                                ORDER BY (session_name IS NOT NULL) DESC, id DESC
                            ) AS rn
                     FROM breakdown_turns
                 )
                 SELECT session_id, agent, project,
                        MAX(CASE WHEN rn = 1 THEN session_name END),
                        COUNT(*),
                        COALESCE(SUM(fresh_input + cache_read + cache_write + output_tok), 0),
                        COALESCE(SUM(cache_read), 0),
                        COALESCE(SUM(fresh_input + cache_read + cache_write), 0),
                        COALESCE(SUM(bill_micros), 0),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        MAX(ts)
                 FROM ranked
                 GROUP BY session_id, agent, project
                 ORDER BY MAX(ts) DESC",
            )
            .context("failed to prepare sessions query")?;
        let rows = stmt
            .query_map([], |r| {
                let cache_read: i64 = r.get(6)?;
                let total_in: i64 = r.get(7)?;
                Ok(SessionRow {
                    session_id: r.get(0)?,
                    agent: r.get(1)?,
                    project: r.get(2)?,
                    session_name: r.get(3)?,
                    turns: r.get(4)?,
                    tokens: r.get(5)?,
                    cache_hit: if total_in > 0 {
                        cache_read as f64 / total_in as f64
                    } else {
                        0.0
                    },
                    bill_micros: r.get(8)?,
                    input_before: r.get(9)?,
                    input_after: r.get(10)?,
                    last_ts: r.get(11)?,
                })
            })
            .context("failed to query sessions")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read sessions")
    }

    /// The id and context window of a session's most recent turn (for occupancy %).
    pub fn latest_turn(&self, session_id: &str) -> Result<Option<(i64, i64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, window FROM breakdown_turns
                 WHERE session_id = ?1 ORDER BY id DESC LIMIT 1",
            )
            .context("failed to prepare latest-turn query")?;
        let row = stmt
            .query_row(params![session_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .ok();
        Ok(row)
    }

    /// Per-source input-token occupancy of one turn, largest first.
    pub fn occupancy(&self, turn_id: i64) -> Result<Vec<OccupancyRow>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT group_label, label, mcp_server, tool_name, SUM(raw_tokens)
                 FROM breakdown_blocks
                 WHERE turn_id = ?1 AND zone = 'input'
                 GROUP BY group_label, label, mcp_server, tool_name
                 ORDER BY SUM(raw_tokens) DESC",
            )
            .context("failed to prepare occupancy query")?;
        let rows = stmt
            .query_map(params![turn_id], |r| {
                Ok(OccupancyRow {
                    group_label: r.get(0)?,
                    label: r.get(1)?,
                    mcp_server: r.get(2)?,
                    tool_name: r.get(3)?,
                    tokens: r.get(4)?,
                })
            })
            .context("failed to query occupancy")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read occupancy")
    }

    /// Per-source cumulative cost across a whole session, priced from the turns' frozen
    /// rates, largest first.
    pub fn cost(&self, session_id: &str) -> Result<Vec<CostRow>> {
        self.cost_grouped(Some(session_id))
    }

    /// Per-source cost aggregated across *all* sessions, largest first — the corpus-wide
    /// "where did the spend go" view used by the `--json`/`--csv` exports.
    pub fn all_sources(&self) -> Result<Vec<CostRow>> {
        self.cost_grouped(None)
    }

    /// Shared per-source cost aggregation; `session` filters to one session, `None` spans all.
    fn cost_grouped(&self, session: Option<&str>) -> Result<Vec<CostRow>> {
        let where_clause = if session.is_some() {
            "WHERE t.session_id = ?1"
        } else {
            ""
        };
        let sql = format!(
            "SELECT b.group_label, b.label, b.mcp_server, b.tool_name,
                    SUM(b.cache_read_tok * t.cache_read_rate) / 1e6 AS read_usd,
                    SUM(b.cache_write_tok * t.cache_write_rate) / 1e6 AS write_usd,
                    SUM(b.fresh_tok * t.input_rate + b.output_tok * t.output_rate) / 1e6 AS new_usd
             FROM breakdown_blocks b JOIN breakdown_turns t ON b.turn_id = t.id
             {where_clause}
             GROUP BY b.group_label, b.label, b.mcp_server, b.tool_name
             ORDER BY (read_usd + write_usd + new_usd) DESC"
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("failed to prepare cost query")?;
        let map = |r: &rusqlite::Row| {
            let read_usd: f64 = r.get(4)?;
            let write_usd: f64 = r.get(5)?;
            let new_usd: f64 = r.get(6)?;
            Ok(CostRow {
                group_label: r.get(0)?,
                label: r.get(1)?,
                mcp_server: r.get(2)?,
                tool_name: r.get(3)?,
                usd: read_usd + write_usd + new_usd,
                read_usd,
                write_usd,
                new_usd,
            })
        };
        let rows = match session {
            Some(id) => stmt.query_map(params![id], map),
            None => stmt.query_map([], map),
        }
        .context("failed to query cost")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read cost")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::{BreakdownBlock, BreakdownTurn, Tracker};

    fn turn(session: &str) -> BreakdownTurn {
        BreakdownTurn {
            session_id: session.to_string(),
            agent: "claude-code".to_string(),
            project: Some("/proj".to_string()),
            session_name: Some("my session".to_string()),
            provider: "anthropic".to_string(),
            model: Some("claude-sonnet-4".to_string()),
            window: 200_000,
            fresh_input: 50,
            cache_read: 120,
            cache_write: 30,
            output_tok: 40,
            input_rate: 3.0,
            output_rate: 15.0,
            cache_read_rate: 0.3,
            cache_write_rate: 3.75,
            bill_micros: 1_000,
            input_before: 300,
            input_after: 200,
        }
    }

    fn block(group: &str, label: &str, mcp: Option<&str>, fresh: f64, read: f64) -> BreakdownBlock {
        BreakdownBlock {
            zone: "input".to_string(),
            section: "static".to_string(),
            bucket: "schema".to_string(),
            group_label: group.to_string(),
            label: label.to_string(),
            mcp_server: mcp.map(str::to_string),
            tool_name: None,
            role: None,
            msg_index: None,
            raw_tokens: (fresh + read) as i64,
            fresh_tok: fresh,
            cache_read_tok: read,
            cache_write_tok: 0.0,
            output_tok: 0.0,
        }
    }

    /// Build a populated in-memory ledger, then re-wrap its connection for read queries.
    /// (One connection both writes and reads — fine for the in-memory test.)
    fn seeded_db() -> BreakdownDb {
        let tracker = Tracker::open_in_memory().unwrap();
        tracker
            .record_breakdown(
                &turn("sess-a"),
                &[
                    block("Static", "System prompt", None, 50.0, 100.0),
                    block("Static", "MCP tools", Some("github"), 0.0, 20.0),
                ],
            )
            .unwrap();
        tracker
            .record_breakdown(
                &turn("sess-a"),
                &[block("Static", "System prompt", None, 50.0, 100.0)],
            )
            .unwrap();
        BreakdownDb::from_connection(tracker.into_connection())
    }

    #[test]
    fn sessions_aggregate_turns_and_cache_hit() {
        let db = seeded_db();
        let rows = db.sessions().unwrap();
        assert_eq!(rows.len(), 1);
        let s = &rows[0];
        assert_eq!(s.session_id, "sess-a");
        assert_eq!(s.turns, 2);
        assert_eq!(s.session_name.as_deref(), Some("my session"));
        // cache_read 120*2 over total_in (50+120+30)*2 = 240/400 = 0.6.
        assert!((s.cache_hit - 0.6).abs() < 1e-6);
    }

    #[test]
    fn occupancy_uses_latest_turn_only() {
        let db = seeded_db();
        let (turn_id, window) = db.latest_turn("sess-a").unwrap().expect("a turn");
        assert_eq!(window, 200_000);
        let occ = db.occupancy(turn_id).unwrap();
        // Latest turn had only the System prompt block.
        assert_eq!(occ.len(), 1);
        assert_eq!(occ[0].label, "System prompt");
        assert_eq!(occ[0].tokens, 150);
    }

    #[test]
    fn all_sources_aggregates_across_sessions() {
        let db = seeded_db();
        let all = db.all_sources().unwrap();
        // System prompt appears in both turns; its tokens/cost aggregate across the session.
        let sys = all.iter().find(|r| r.label == "System prompt").unwrap();
        assert!(sys.usd > 0.0);
        // The github MCP source from the first turn is present corpus-wide.
        assert!(
            all.iter()
                .any(|r| r.mcp_server.as_deref() == Some("github"))
        );
    }

    #[test]
    fn cost_prices_from_frozen_rates_and_splits_mcp() {
        let db = seeded_db();
        let rows = db.cost("sess-a").unwrap();
        let mcp = rows
            .iter()
            .find(|r| r.mcp_server.as_deref() == Some("github"))
            .unwrap();
        // 20 cache-read tokens * 0.3/1e6 = 6e-6.
        assert!((mcp.read_usd - 20.0 * 0.3 / 1e6).abs() < 1e-12);
        assert!(mcp.new_usd.abs() < 1e-12);
    }
}
