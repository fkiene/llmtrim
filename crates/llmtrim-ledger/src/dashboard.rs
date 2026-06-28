//! Per-agent dashboard data layer and pure contract builder for the tray app.
//!
//! `BreakdownDb` extensions: `agent_aggregates`, `agent_trend`, `open_readonly`.
//! Pure builder: `build_dashboard` — no DB, no Tauri; fully covered by unit tests.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{OpenFlags, params};
use serde::Serialize;

use crate::breakdown_db::BreakdownDb;
use crate::tracking::Period;

// ---------------------------------------------------------------------------
// Query result types
// ---------------------------------------------------------------------------

/// Per-agent rollup produced by `BreakdownDb::agent_aggregates`.
///
/// SECURITY: this struct must NOT have `project` or `session_name` fields.
/// Those columns are absolute filesystem paths and human session names.
/// The `GROUP BY agent` aggregation structurally drops them; the type
/// enforces it at the boundary so they can never reach the webview.
#[derive(Debug, Clone)]
pub struct AgentAggregate {
    pub agent: String,
    pub input_before: i64,
    pub input_after: i64,
    pub bill_micros: i64,
    pub cache_read: i64,
    pub last_event_ts: Option<String>,
    /// True iff at least one turn recorded non-NULL `input_before` (i.e. the
    /// compression meter was active). False for entirely pre-meter agents.
    pub has_savings_data: bool,
}

/// One time bucket of gross per-agent savings, for the trend sparkline.
#[derive(Debug, Clone)]
pub struct PeriodSaved {
    pub bucket: String,
    /// Gross saved pct for this bucket: max(0, before-after)/before*100; 0 when before==0.
    pub saved_pct: f64,
}

// ---------------------------------------------------------------------------
// Serializable contract types (§2 of BUILD-PLAN.md)
// ---------------------------------------------------------------------------

/// Per-agent card sent to the tray frontend.
#[derive(Debug, Clone, Serialize)]
pub struct AgentCard {
    /// Raw agent id from the ledger (e.g. "claude-code").
    pub agent: String,
    /// Human-readable name; known ids have a fixed map, unknown ids get Unicode title-case.
    pub display_name: String,
    pub input_before: i64,
    pub input_after: i64,
    /// Gross input savings ratio: max(0, before-after)/before*100; 0.0 when before==0.
    pub saved_pct: f64,
    /// False when all turns predate the compression-meter columns; the UI shows "—" instead
    /// of "0% saved" to avoid implying we measured and found nothing.
    pub has_savings_data: bool,
    /// Total bill in micro-USD; frontend divides by 1_000_000 for display.
    pub bill_micros: i64,
    /// Provider-reported cache-read tokens summed over all turns.
    pub cache_read_tokens: i64,
    /// Raw saved_pct per period bucket, chronological; frontend scales for the sparkline.
    pub trend: Vec<f64>,
    pub last_event_ts: Option<String>,
}

/// Workspace totals across all agents.
#[derive(Debug, Clone, Serialize)]
pub struct DashboardTotals {
    pub input_before: i64,
    pub input_after: i64,
    pub saved_pct: f64,
    pub bill_micros: i64,
}

/// Top-level payload returned by the Tauri `get_dashboard` command.
#[derive(Debug, Clone, Serialize)]
pub struct Dashboard {
    pub cards: Vec<AgentCard>,
    pub totals: DashboardTotals,
    /// RFC-3339 timestamp when this snapshot was built.
    pub generated_at: String,
    /// Seconds until the next poll; drives the "Next update in Ns" footer.
    pub next_update_secs: u64,
}

// ---------------------------------------------------------------------------
// BreakdownDb extensions
// ---------------------------------------------------------------------------

impl BreakdownDb {
    /// Open the ledger at `path` read-only, **skipping `migrate()`**.
    ///
    /// The proxy is the sole writer and owns DDL; the tray must never take an
    /// exclusive lock or run schema migrations. Returns a clear error if the
    /// `breakdown_turns` table is absent (proxy not yet started).
    pub fn open_readonly(path: &Path) -> Result<Self> {
        let conn = rusqlite::Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open ledger read-only at {}", path.display()))?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='breakdown_turns'",
                [],
                |r| r.get(0),
            )
            .context("failed to check ledger schema")?;
        if count == 0 {
            anyhow::bail!(
                "ledger at {} has no breakdown_turns table — run the proxy first to initialise the schema",
                path.display()
            );
        }
        Ok(Self::from_connection(conn))
    }

    /// Per-agent aggregates from `breakdown_turns`, newest activity first.
    ///
    /// Groups by `agent` only — `project` and `session_name` are deliberately
    /// excluded (security gate: absolute paths / human names must not surface in
    /// the webview). `AgentAggregate` is structurally incapable of carrying them.
    pub fn agent_aggregates(&self) -> Result<Vec<AgentAggregate>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT agent,
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        COALESCE(SUM(bill_micros), 0),
                        COALESCE(SUM(cache_read), 0),
                        MAX(ts),
                        COUNT(input_before)
                 FROM breakdown_turns
                 GROUP BY agent
                 ORDER BY MAX(ts) DESC",
            )
            .context("failed to prepare agent_aggregates query")?;
        let rows = stmt
            .query_map([], |r| {
                let has_meter: i64 = r.get(6)?;
                Ok(AgentAggregate {
                    agent: r.get(0)?,
                    input_before: r.get(1)?,
                    input_after: r.get(2)?,
                    bill_micros: r.get(3)?,
                    cache_read: r.get(4)?,
                    last_event_ts: r.get(5)?,
                    has_savings_data: has_meter > 0,
                })
            })
            .context("failed to query agent_aggregates")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to collect agent_aggregates")
    }

    /// Per-bucket gross savings trend for `agent`, at most `buckets` entries in
    /// chronological order.
    ///
    /// Uses `breakdown_turns` — NOT `Tracker::by_period`, which reads `compressions`
    /// and has no `agent` column. The `idx_breakdown_turns_agent` index keeps this fast.
    pub fn agent_trend(
        &self,
        agent: &str,
        period: Period,
        buckets: usize,
    ) -> Result<Vec<PeriodSaved>> {
        let bucket_expr = period.sql_bucket();
        let sql = format!(
            "SELECT {bucket_expr} AS bucket,
                    COALESCE(SUM(input_before), 0),
                    COALESCE(SUM(input_after), 0)
             FROM breakdown_turns
             WHERE agent = ?1
             GROUP BY bucket
             ORDER BY bucket DESC
             LIMIT ?2"
        );
        // Clamp into i64 — a raw `buckets as i64` would turn `usize::MAX` into -1, and
        // SQLite reads `LIMIT -1` as "unlimited", a silent no-limit backdoor.
        let limit = i64::try_from(buckets).unwrap_or(i64::MAX);
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("failed to prepare agent_trend query")?;
        let mut rows: Vec<PeriodSaved> = stmt
            .query_map(params![agent, limit], |r| {
                let before: i64 = r.get(1)?;
                let after: i64 = r.get(2)?;
                Ok((r.get::<_, String>(0)?, before, after))
            })
            .context("failed to query agent_trend")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to collect agent_trend")?
            .into_iter()
            .map(|(bucket, before, after)| PeriodSaved {
                bucket,
                saved_pct: gross_saved_pct(before, after),
            })
            .collect();
        // Query returned DESC (newest first); reverse to chronological order.
        rows.reverse();
        Ok(rows)
    }
}

// ---------------------------------------------------------------------------
// Pure dashboard builder
// ---------------------------------------------------------------------------

/// Known agent id → display name. Unknown ids fall back to `title_case`.
fn known_display_name(agent: &str) -> Option<&'static str> {
    match agent {
        "claude-code" => Some("Claude Code"),
        "codex" => Some("Codex"),
        "gemini" => Some("Gemini"),
        _ => None,
    }
}

/// Unicode-aware title-case: split on `-`, `_`, or space; capitalise the first `char`
/// of each segment via `char::to_uppercase()`, which correctly uppercases non-ASCII
/// code points (e.g. `ñ` → `Ñ`, `ß` → `SS`). Does not require extra crates.
fn title_case(s: &str) -> String {
    s.split(['-', '_', ' '])
        .filter(|w| !w.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Gross savings percentage: max(0, before-after)/before*100; 0.0 when before <= 0.
fn gross_saved_pct(before: i64, after: i64) -> f64 {
    if before <= 0 {
        0.0
    } else {
        (before - after).max(0) as f64 / before as f64 * 100.0
    }
}

/// Pure transform: aggregates + pre-fetched trends → `Dashboard`. No DB, no Tauri.
///
/// The Tauri `get_dashboard` command is a one-line wrapper over this function; this
/// function itself is fully covered by unit tests (see below).
///
/// `trends`: agent → chronological vec of per-bucket `saved_pct` floats (from
/// `BreakdownDb::agent_trend`). Missing keys produce an empty `trend` vec.
pub fn build_dashboard(
    aggregates: Vec<AgentAggregate>,
    trends: HashMap<String, Vec<f64>>,
    generated_at: String,
    next_update_secs: u64,
) -> Dashboard {
    let cards: Vec<AgentCard> = aggregates
        .into_iter()
        .map(|a| {
            let display_name = known_display_name(&a.agent)
                .map(str::to_string)
                .unwrap_or_else(|| title_case(&a.agent));
            let saved_pct = if a.has_savings_data {
                gross_saved_pct(a.input_before, a.input_after)
            } else {
                0.0
            };
            let trend = trends.get(&a.agent).cloned().unwrap_or_default();
            AgentCard {
                agent: a.agent,
                display_name,
                input_before: a.input_before,
                input_after: a.input_after,
                saved_pct,
                has_savings_data: a.has_savings_data,
                bill_micros: a.bill_micros,
                cache_read_tokens: a.cache_read,
                trend,
                last_event_ts: a.last_event_ts,
            }
        })
        .collect();

    let total_before: i64 = cards.iter().map(|c| c.input_before).sum();
    let total_after: i64 = cards.iter().map(|c| c.input_after).sum();
    let total_bill: i64 = cards.iter().map(|c| c.bill_micros).sum();

    Dashboard {
        totals: DashboardTotals {
            input_before: total_before,
            input_after: total_after,
            saved_pct: gross_saved_pct(total_before, total_after),
            bill_micros: total_bill,
        },
        cards,
        generated_at,
        next_update_secs,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::{BreakdownTurn, Tracker};

    /// Minimal turn fixture for seeding. `input_before` and `input_after` are non-NULL
    /// (the compression meter was active).
    fn metered_turn(agent: &str, before: i64, after: i64, bill: i64) -> BreakdownTurn {
        BreakdownTurn {
            session_id: format!("sess-{agent}"),
            agent: agent.to_string(),
            project: Some("/some/project".to_string()),
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
            bill_micros: bill,
            input_before: before,
            input_after: after,
        }
    }

    fn seeded(agent: &str, before: i64, after: i64, bill: i64) -> BreakdownDb {
        let tracker = Tracker::open_in_memory().expect("in-memory tracker");
        tracker
            .record_breakdown(&metered_turn(agent, before, after, bill), &[])
            .expect("record");
        BreakdownDb::from_connection(tracker.into_connection())
    }

    // --- agent_aggregates happy path ---

    #[test]
    fn agent_aggregates_gross_formula() {
        // before=1000, after=600 → saved_pct = 40.0
        let db = seeded("claude-code", 1_000, 600, 5_000);
        let rows = db.agent_aggregates().expect("aggregates");
        assert_eq!(rows.len(), 1);
        let a = &rows[0];
        assert_eq!(a.agent, "claude-code");
        assert_eq!(a.input_before, 1_000);
        assert_eq!(a.input_after, 600);
        assert_eq!(a.bill_micros, 5_000);
        assert!(a.has_savings_data);
        // Gross formula check via build_dashboard
        let dash = build_dashboard(rows, HashMap::new(), "2024-01-01T00:00:00Z".into(), 30);
        let card = &dash.cards[0];
        assert!(
            (card.saved_pct - 40.0).abs() < 1e-9,
            "saved_pct={}",
            card.saved_pct
        );
    }

    // --- empty ledger ---

    #[test]
    fn empty_ledger_aggregates_is_empty() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        let db = BreakdownDb::from_connection(tracker.into_connection());
        let rows = db.agent_aggregates().expect("aggregates");
        assert!(rows.is_empty());
    }

    #[test]
    fn empty_ledger_dashboard_zero_totals() {
        let dash = build_dashboard(vec![], HashMap::new(), "2024-01-01T00:00:00Z".into(), 30);
        assert!(dash.cards.is_empty());
        assert_eq!(dash.totals.input_before, 0);
        assert_eq!(dash.totals.input_after, 0);
        assert_eq!(dash.totals.bill_micros, 0);
        assert_eq!(dash.totals.saved_pct, 0.0);
    }

    // --- pre-meter NULL input_before rows ---

    #[test]
    fn premeter_null_has_savings_data_false_and_pct_zero() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        tracker
            .record_breakdown_premeter("codex", 9_000)
            .expect("premeter row");
        let db = BreakdownDb::from_connection(tracker.into_connection());
        let rows = db.agent_aggregates().expect("aggregates");
        assert_eq!(rows.len(), 1);
        let a = &rows[0];
        assert_eq!(a.agent, "codex");
        assert!(
            !a.has_savings_data,
            "no meter data → has_savings_data must be false"
        );
        assert_eq!(a.input_before, 0, "COALESCE(NULL,0)=0");
        assert_eq!(a.input_after, 0, "COALESCE(NULL,0)=0");
        // build_dashboard must not compute saved_pct when has_savings_data is false
        let dash = build_dashboard(rows, HashMap::new(), "2024-01-01T00:00:00Z".into(), 30);
        assert_eq!(dash.cards[0].saved_pct, 0.0);
        assert!(!dash.cards[0].has_savings_data);
    }

    // --- input_before == 0 guard: no divide-by-zero ---

    #[test]
    fn input_before_zero_no_divide_by_zero() {
        let db = seeded("gemini", 0, 0, 1_000);
        let rows = db.agent_aggregates().expect("aggregates");
        let dash = build_dashboard(rows, HashMap::new(), "2024-01-01T00:00:00Z".into(), 30);
        let card = &dash.cards[0];
        // has_savings_data=true (input_before col is non-NULL, just happens to be 0)
        assert!(card.has_savings_data);
        assert_eq!(card.saved_pct, 0.0, "before==0 must yield 0.0, not NaN/inf");
        assert!(card.saved_pct.is_finite());
    }

    // --- multi-agent: no cross-contamination ---

    #[test]
    fn multi_agent_no_cross_contamination() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        tracker
            .record_breakdown(&metered_turn("claude-code", 1_000, 800, 3_000), &[])
            .expect("record claude-code");
        tracker
            .record_breakdown(&metered_turn("codex", 2_000, 1_200, 7_000), &[])
            .expect("record codex");
        // Second claude-code turn
        tracker
            .record_breakdown(&metered_turn("claude-code", 500, 400, 1_500), &[])
            .expect("record claude-code 2");

        let db = BreakdownDb::from_connection(tracker.into_connection());
        let rows = db.agent_aggregates().expect("aggregates");
        assert_eq!(rows.len(), 2);

        let cc = rows
            .iter()
            .find(|r| r.agent == "claude-code")
            .expect("claude-code");
        let cx = rows.iter().find(|r| r.agent == "codex").expect("codex");

        // bill_micros must not bleed across agents
        assert_eq!(cc.bill_micros, 3_000 + 1_500, "claude-code bill sum");
        assert_eq!(cx.bill_micros, 7_000, "codex bill sum");
        // input sums are per-agent
        assert_eq!(cc.input_before, 1_500);
        assert_eq!(cc.input_after, 1_200);
        assert_eq!(cx.input_before, 2_000);
        assert_eq!(cx.input_after, 1_200);
    }

    // --- agent_trend: Day bucket boundaries ---

    #[test]
    fn agent_trend_day_buckets() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        let base = metered_turn("claude-code", 1_000, 600, 1_000);
        tracker
            .record_breakdown_with_ts(&base, "2024-01-01T12:00:00+00:00")
            .expect("day 1");
        tracker
            .record_breakdown_with_ts(&base, "2024-01-02T12:00:00+00:00")
            .expect("day 2");
        // Different agent — must not appear in claude-code trend
        let other = metered_turn("codex", 500, 500, 500);
        tracker
            .record_breakdown_with_ts(&other, "2024-01-01T12:00:00+00:00")
            .expect("codex day 1");

        let db = BreakdownDb::from_connection(tracker.into_connection());
        let trend = db
            .agent_trend("claude-code", Period::Day, 10)
            .expect("trend");
        assert_eq!(trend.len(), 2, "two distinct day buckets");
        assert_eq!(trend[0].bucket, "2024-01-01", "chronological order");
        assert_eq!(trend[1].bucket, "2024-01-02");
        assert!((trend[0].saved_pct - 40.0).abs() < 1e-9);
    }

    // --- agent_trend: Week bucket — Monday edge ---
    //
    // SQLite strftime('%W') counts weeks with Monday as first day.
    // 2024-01-07 = Sunday → week 01 (same week as Monday 2024-01-01).
    // 2024-01-08 = Monday → week 02 (new week).
    // The two timestamps must fall in different week buckets.

    #[test]
    fn agent_trend_week_monday_edge() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        let base = metered_turn("claude-code", 1_000, 500, 1_000);
        tracker
            .record_breakdown_with_ts(&base, "2024-01-07T23:59:59+00:00")
            .expect("Sunday");
        tracker
            .record_breakdown_with_ts(&base, "2024-01-08T00:00:01+00:00")
            .expect("Monday");

        let db = BreakdownDb::from_connection(tracker.into_connection());
        let trend = db
            .agent_trend("claude-code", Period::Week, 10)
            .expect("trend");
        assert_eq!(
            trend.len(),
            2,
            "Sunday and Monday must be in different week buckets"
        );
        // Pin the exact bucket strings so a change to sql_bucket()'s format is caught,
        // not just relative ordering.
        assert_eq!(trend[0].bucket, "2024-W01", "Sunday → week 01");
        assert_eq!(trend[1].bucket, "2024-W02", "Monday → week 02");
    }

    // --- agent_trend: edge cases ---

    #[test]
    fn agent_trend_unknown_agent_returns_empty() {
        let db = seeded("claude-code", 100, 50, 100);
        let trend = db
            .agent_trend("nonexistent", Period::Day, 10)
            .expect("trend");
        assert!(trend.is_empty(), "no rows for an agent that never ran");
    }

    #[test]
    fn agent_trend_zero_buckets_returns_empty() {
        let db = seeded("claude-code", 100, 50, 100);
        // LIMIT 0 must yield an empty vec, not error and not "unlimited".
        let trend = db
            .agent_trend("claude-code", Period::Day, 0)
            .expect("trend");
        assert!(trend.is_empty());
    }

    #[test]
    fn mixed_meter_agent_has_savings_data_and_sums_metered_only() {
        // One pre-meter turn (NULL input_before/after) + one metered turn for the
        // same agent. COUNT(input_before) > 0 → has_savings_data; SUM skips the NULL
        // row's tokens but still sums its bill.
        let tracker = Tracker::open_in_memory().expect("tracker");
        tracker
            .record_breakdown_premeter("claude-code", 1_000)
            .expect("premeter");
        tracker
            .record_breakdown(&metered_turn("claude-code", 800, 400, 2_000), &[])
            .expect("metered");
        let db = BreakdownDb::from_connection(tracker.into_connection());
        let rows = db.agent_aggregates().expect("aggregates");
        let a = rows
            .iter()
            .find(|r| r.agent == "claude-code")
            .expect("claude-code present");
        assert!(a.has_savings_data, "at least one metered row → true");
        assert_eq!(
            a.input_before, 800,
            "NULL row contributes nothing to the sum"
        );
        assert_eq!(a.input_after, 400);
        assert_eq!(a.bill_micros, 3_000, "both turns' bills are summed");
    }

    // --- agent_trend: Month bucket ---

    #[test]
    fn agent_trend_month_buckets() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        let base = metered_turn("claude-code", 900, 300, 1_000);
        tracker
            .record_breakdown_with_ts(&base, "2024-01-15T00:00:00+00:00")
            .expect("jan");
        tracker
            .record_breakdown_with_ts(&base, "2024-02-15T00:00:00+00:00")
            .expect("feb");

        let db = BreakdownDb::from_connection(tracker.into_connection());
        let trend = db
            .agent_trend("claude-code", Period::Month, 10)
            .expect("trend");
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0].bucket, "2024-01");
        assert_eq!(trend[1].bucket, "2024-02");
        // saved_pct = (900-300)/900*100 ≈ 66.67% for both buckets
        for t in &trend {
            assert!(
                (t.saved_pct - 200.0 / 3.0).abs() < 1e-6,
                "saved_pct={}",
                t.saved_pct
            );
        }
    }

    // --- agent_trend: bucket limit ---

    #[test]
    fn agent_trend_respects_bucket_limit() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        let base = metered_turn("claude-code", 100, 50, 100);
        for day in 1..=10 {
            tracker
                .record_breakdown_with_ts(&base, &format!("2024-01-{day:02}T00:00:00+00:00"))
                .expect("day");
        }
        let db = BreakdownDb::from_connection(tracker.into_connection());
        let trend = db
            .agent_trend("claude-code", Period::Day, 5)
            .expect("trend");
        // LIMIT 5 → 5 most recent buckets; reversed to chronological → days 6-10
        assert_eq!(trend.len(), 5);
        assert_eq!(trend[0].bucket, "2024-01-06");
        assert_eq!(trend[4].bucket, "2024-01-10");
    }

    // --- i64 bill_micros sum: headroom documented ---
    //
    // i64::MAX ≈ 9.2 × 10^18 micro-USD = $9.2 × 10^12.
    // Even at $1 M/day (an absurd ceiling for any real user), a ledger would need
    // 9.2 × 10^6 days (~25,000 years) to overflow. A near-overflow fixture would
    // require seeding ~10^13 rows; instead we verify correctness at a large but
    // realistic scale.

    #[test]
    fn bill_micros_large_sum_no_overflow() {
        let tracker = Tracker::open_in_memory().expect("tracker");
        // $1,000 each (1_000_000_000 µ-USD), 10 turns = $10,000 total — well within i64.
        let large = metered_turn("claude-code", 1_000, 500, 1_000_000_000);
        for _ in 0..10 {
            tracker.record_breakdown(&large, &[]).expect("record");
        }
        let db = BreakdownDb::from_connection(tracker.into_connection());
        let rows = db.agent_aggregates().expect("aggregates");
        assert_eq!(rows[0].bill_micros, 10_000_000_000i64);
    }

    // --- Unicode unknown-agent title-case ---

    #[test]
    fn unicode_agent_title_case_in_dashboard() {
        // "ñoño-bot" is unknown → title_case → "Ñoño Bot" (non-ASCII uppercase)
        let agg = AgentAggregate {
            agent: "ñoño-bot".to_string(),
            input_before: 100,
            input_after: 50,
            bill_micros: 1_000,
            cache_read: 0,
            last_event_ts: None,
            has_savings_data: true,
        };
        let dash = build_dashboard(vec![agg], HashMap::new(), "2024-01-01T00:00:00Z".into(), 30);
        assert_eq!(dash.cards[0].display_name, "Ñoño Bot");
    }

    #[test]
    fn known_agents_get_fixed_display_names() {
        for (id, expected) in [
            ("claude-code", "Claude Code"),
            ("codex", "Codex"),
            ("gemini", "Gemini"),
        ] {
            let agg = AgentAggregate {
                agent: id.to_string(),
                input_before: 0,
                input_after: 0,
                bill_micros: 0,
                cache_read: 0,
                last_event_ts: None,
                has_savings_data: false,
            };
            let dash = build_dashboard(vec![agg], HashMap::new(), "t".into(), 30);
            assert_eq!(dash.cards[0].display_name, expected, "agent={id}");
        }
    }

    // --- insta snapshot over build_dashboard JSON ---

    #[test]
    fn snapshot_dashboard_json() {
        let aggregates = vec![
            AgentAggregate {
                agent: "claude-code".to_string(),
                input_before: 10_000,
                input_after: 4_000,
                bill_micros: 500_000,
                cache_read: 3_000,
                last_event_ts: Some("2024-06-01T10:00:00+00:00".to_string()),
                has_savings_data: true,
            },
            AgentAggregate {
                agent: "codex".to_string(),
                input_before: 0,
                input_after: 0,
                bill_micros: 100_000,
                cache_read: 0,
                last_event_ts: Some("2024-06-01T09:00:00+00:00".to_string()),
                has_savings_data: false,
            },
        ];
        let mut trends = HashMap::new();
        trends.insert("claude-code".to_string(), vec![55.0, 60.0, 58.0]);
        let dash = build_dashboard(
            aggregates,
            trends,
            "2024-06-01T10:05:00+00:00".to_string(),
            30,
        );
        let json = serde_json::to_string_pretty(&dash).expect("serialize");
        insta::assert_snapshot!(json);
    }
}
