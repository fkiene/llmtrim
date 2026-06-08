//! SQLite savings ledger.
//!
//! Mirrors RTK's tracking approach (rusqlite, `~/.local/share/<tool>/tracking.db`)
//! but records **real-tokenizer** counts (spec §6) and carries nullable
//! output-token columns for the round-trip cost model once the proxy phase can
//! measure them. Recording is best-effort at the CLI layer: a ledger failure must
//! never block the user's compressed output.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

/// One compression event.
#[derive(Debug, Clone)]
pub struct Record {
    pub provider: String,
    pub model: Option<String>,
    pub tokenizer: String,
    pub exact: bool,
    pub input_before: i64,
    pub input_after: i64,
    pub output_before: Option<i64>,
    pub output_after: Option<i64>,
    /// Microseconds spent compressing this request (proxy overhead); `None` for CLI paths.
    pub compress_micros: Option<i64>,
    /// Provider-reported cached input tokens reused on this request (Anthropic
    /// `cache_read_input_tokens`), billed at ~10% of input price — a real bill discount the
    /// token % can't show. `None` when the provider reports none / on CLI paths.
    pub cache_read_tokens: Option<i64>,
}

/// Per-provider aggregate row.
#[derive(Debug, Clone)]
pub struct ProviderRow {
    pub provider: String,
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub exact: bool,
    pub output_before: i64,
    pub output_after: i64,
    pub output_events: i64,
}

/// Per-(provider, model) aggregate row — used to price savings with a per-model rate.
#[derive(Debug, Clone)]
pub struct ModelRow {
    pub provider: String,
    pub model: Option<String>,
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub output_after: i64,
}

/// Time-series bucket granularity for `by_period`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    Day,
    Week,
    Month,
}

impl Period {
    /// SQL expression that buckets the rfc3339 `ts` column (string-sliced, no date parse
    /// for day/month; `strftime` only for ISO week).
    fn sql_bucket(self) -> &'static str {
        match self {
            Period::Day => "substr(ts, 1, 10)",       // YYYY-MM-DD
            Period::Month => "substr(ts, 1, 7)",      // YYYY-MM
            Period::Week => "strftime('%Y-W%W', ts)", // YYYY-Www
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Period::Day => "daily",
            Period::Week => "weekly",
            Period::Month => "monthly",
        }
    }
}

/// One time bucket of savings, for `--daily/--weekly/--monthly` reports.
#[derive(Debug, Clone)]
pub struct PeriodRow {
    pub bucket: String,
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub output_before: i64,
    pub output_after: i64,
}

/// Aggregate savings for the dashboard.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub any_approximate: bool,
    pub by_provider: Vec<ProviderRow>,
    pub output_before: i64,
    pub output_after: i64,
    pub output_events: i64,
    /// Mean compression overhead (µs) across recorded requests; `None` if none recorded it.
    pub avg_compress_micros: Option<f64>,
    /// Total cached input tokens reused (Anthropic prompt-cache hits) — the discounted prefix.
    pub cache_read_tokens: i64,
}

impl Summary {
    pub fn saved(&self) -> i64 {
        self.input_before - self.input_after
    }

    /// Percentage of input tokens saved (0.0 when no data).
    pub fn saved_pct(&self) -> f64 {
        if self.input_before <= 0 {
            0.0
        } else {
            (self.saved() as f64 / self.input_before as f64) * 100.0
        }
    }

    pub fn output_saved(&self) -> i64 {
        self.output_before - self.output_after
    }

    /// Percentage of output tokens saved (0.0 when no counterfactual data).
    pub fn output_saved_pct(&self) -> f64 {
        if self.output_before <= 0 {
            0.0
        } else {
            (self.output_saved() as f64 / self.output_before as f64) * 100.0
        }
    }
}

/// Default ledger row cap — the most-recent N compression events are retained. Each row is
/// metadata only (~100 bytes), so this bounds the file to roughly 10-15 MB.
pub const DEFAULT_MAX_ROWS: i64 = 100_000;

pub struct Tracker {
    conn: Connection,
}

impl Tracker {
    /// Open (creating if needed) the ledger at the default path, or the path in
    /// `LLMTRIM_DB_PATH` when set.
    pub fn open() -> Result<Self> {
        let path = default_db_path()?;
        Self::open_at(&path)
    }

    pub fn open_at(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open ledger at {}", path.display()))?;
        let tracker = Self { conn };
        tracker.migrate()?;
        // Bound the ledger on open: row cap + (if configured) age retention. The daemon
        // opens once and re-prunes periodically (see serve.rs); CLI paths prune per call.
        let _ = tracker.prune_default();
        Ok(tracker)
    }

    /// In-memory ledger (tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory ledger")?;
        let tracker = Self { conn };
        tracker.migrate()?;
        Ok(tracker)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS compressions (
                    id            INTEGER PRIMARY KEY,
                    ts            TEXT NOT NULL,
                    provider      TEXT NOT NULL,
                    model         TEXT,
                    tokenizer     TEXT NOT NULL,
                    exact         INTEGER NOT NULL,
                    input_before  INTEGER NOT NULL,
                    input_after   INTEGER NOT NULL,
                    output_before INTEGER,
                    output_after  INTEGER,
                    compress_micros INTEGER,
                    cache_read_tokens INTEGER
                );",
            )
            .context("failed to migrate ledger schema")?;
        // Additive columns for ledgers created before these fields existed — each ALTER errors
        // with "duplicate column" once it exists (and on fresh DBs the CREATE already has it),
        // which we ignore.
        for col in ["compress_micros", "cache_read_tokens"] {
            let _ = self.conn.execute(
                &format!("ALTER TABLE compressions ADD COLUMN {col} INTEGER"),
                [],
            );
        }
        Ok(())
    }

    /// Apply retention to the ledger: drop rows older than `max_age_days` (when set), then
    /// trim to the most recent `max_rows`. Returns the number of rows deleted. The ledger
    /// holds only metadata (no prompt/response text), but it must still stay bounded for the
    /// always-on daemon — analytics only need recent history.
    pub fn prune(&self, max_rows: i64, max_age_days: Option<i64>) -> Result<u64> {
        let mut deleted: u64 = 0;
        // Age-based: `ts` is rfc3339 UTC (always `+00:00`), so a lexical `<` compare against
        // the cutoff is a correct chronological compare — no date parsing needed.
        if let Some(days) = max_age_days.filter(|d| *d > 0) {
            let delta = chrono::TimeDelta::try_days(days).unwrap_or_else(chrono::TimeDelta::zero);
            let cutoff = (chrono::Utc::now() - delta).to_rfc3339();
            deleted += self
                .conn
                .execute("DELETE FROM compressions WHERE ts < ?1", params![cutoff])
                .context("failed to age-prune ledger")? as u64;
        }
        // Row cap: keep only the most recent `max_rows` rows by id.
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM compressions", [], |row| row.get(0))
            .context("failed to count ledger rows")?;
        if n > max_rows {
            deleted += self
                .conn
                .execute(
                    "DELETE FROM compressions WHERE id <= (SELECT MAX(id) - ?1 FROM compressions)",
                    params![max_rows],
                )
                .context("failed to cap-prune ledger")? as u64;
        }
        Ok(deleted)
    }

    /// Prune with the default policy: the built-in row cap ([`DEFAULT_MAX_ROWS`]) plus the
    /// configured age retention (`LLMTRIM_RETENTION_DAYS` env or `retention_days` in the
    /// config file; `None` = age retention disabled, row cap only).
    pub fn prune_default(&self) -> Result<u64> {
        self.prune(DEFAULT_MAX_ROWS, crate::config::retention_days())
    }

    pub fn record(&self, r: &Record) -> Result<()> {
        let ts = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO compressions
                    (ts, provider, model, tokenizer, exact, input_before, input_after,
                     output_before, output_after, compress_micros, cache_read_tokens)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    ts,
                    r.provider,
                    r.model,
                    r.tokenizer,
                    i64::from(r.exact),
                    r.input_before,
                    r.input_after,
                    r.output_before,
                    r.output_after,
                    r.compress_micros,
                    r.cache_read_tokens,
                ],
            )
            .context("failed to record compression")?;
        Ok(())
    }

    /// Test-only: insert a record stamped with an explicit `ts`, to exercise age retention
    /// without waiting real time (`record` always stamps `now`).
    #[cfg(test)]
    fn record_with_ts(&self, r: &Record, ts: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO compressions
                    (ts, provider, model, tokenizer, exact, input_before, input_after,
                     output_before, output_after, compress_micros, cache_read_tokens)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    ts,
                    r.provider,
                    r.model,
                    r.tokenizer,
                    i64::from(r.exact),
                    r.input_before,
                    r.input_after,
                    r.output_before,
                    r.output_after,
                    r.compress_micros,
                    r.cache_read_tokens,
                ],
            )
            .context("failed to record compression (test)")?;
        Ok(())
    }

    pub fn summary(&self) -> Result<Summary> {
        let (events, input_before, input_after, approx, output_before, output_after, output_events): (
            i64, i64, i64, i64, i64, i64, i64,
        ) = self
            .conn
            .query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        COALESCE(SUM(CASE WHEN exact = 0 THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(output_before), 0),
                        COALESCE(SUM(output_after), 0),
                        COALESCE(SUM(CASE WHEN output_after IS NOT NULL THEN 1 ELSE 0 END), 0)
                 FROM compressions",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .context("failed to summarize ledger")?;

        let mut stmt = self
            .conn
            .prepare(
                "SELECT provider, COUNT(*),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        MIN(exact),
                        COALESCE(SUM(output_before), 0),
                        COALESCE(SUM(output_after), 0),
                        COALESCE(SUM(CASE WHEN output_after IS NOT NULL THEN 1 ELSE 0 END), 0)
                 FROM compressions GROUP BY provider ORDER BY provider",
            )
            .context("failed to prepare provider summary")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ProviderRow {
                    provider: row.get(0)?,
                    events: row.get(1)?,
                    input_before: row.get(2)?,
                    input_after: row.get(3)?,
                    exact: row.get::<_, i64>(4)? != 0,
                    output_before: row.get(5)?,
                    output_after: row.get(6)?,
                    output_events: row.get(7)?,
                })
            })
            .context("failed to query provider summary")?;
        let by_provider = rows
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read provider summary")?;

        // Mean compression overhead + total cached-prefix tokens reused. AVG/SUM ignore NULL
        // (CLI rows / pre-feature ledgers); AVG returns NULL → None when nothing has it.
        let (avg_compress_micros, cache_read_tokens): (Option<f64>, i64) = self
            .conn
            .query_row(
                "SELECT AVG(compress_micros), COALESCE(SUM(cache_read_tokens), 0) FROM compressions",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .context("failed to summarize latency + cache")?;

        Ok(Summary {
            events,
            input_before,
            input_after,
            any_approximate: approx > 0,
            by_provider,
            output_before,
            output_after,
            output_events,
            avg_compress_micros,
            cache_read_tokens,
        })
    }

    /// Per-(provider, model) aggregates, for pricing each model's savings at its own rate.
    pub fn by_model(&self) -> Result<Vec<ModelRow>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT provider, model, COUNT(*),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        COALESCE(SUM(output_after), 0)
                 FROM compressions GROUP BY provider, model ORDER BY provider, model",
            )
            .context("failed to prepare model summary")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ModelRow {
                    provider: row.get(0)?,
                    model: row.get(1)?,
                    events: row.get(2)?,
                    input_before: row.get(3)?,
                    input_after: row.get(4)?,
                    output_after: row.get(5)?,
                })
            })
            .context("failed to query model summary")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read model summary")
    }

    /// Savings grouped into time buckets (day/week/month), oldest first.
    pub fn by_period(&self, period: Period) -> Result<Vec<PeriodRow>> {
        let sql = format!(
            "SELECT {} AS bucket, COUNT(*),
                    COALESCE(SUM(input_before), 0),
                    COALESCE(SUM(input_after), 0),
                    COALESCE(SUM(output_before), 0),
                    COALESCE(SUM(output_after), 0)
             FROM compressions GROUP BY bucket ORDER BY bucket",
            period.sql_bucket()
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("failed to prepare period summary")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(PeriodRow {
                    bucket: row.get(0)?,
                    events: row.get(1)?,
                    input_before: row.get(2)?,
                    input_after: row.get(3)?,
                    output_before: row.get(4)?,
                    output_after: row.get(5)?,
                })
            })
            .context("failed to query period summary")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read period summary")
    }
}

/// The ledger file path (respects `LLMTRIM_DB_PATH` / `XDG_DATA_HOME`). Exposed so
/// `uninstall --purge` can remove it.
pub fn db_path() -> Result<PathBuf> {
    default_db_path()
}

fn default_db_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LLMTRIM_DB_PATH") {
        return Ok(PathBuf::from(p));
    }
    let base = if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg)
    } else {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("set HOME (or USERPROFILE), or LLMTRIM_DB_PATH")?;
        PathBuf::from(home).join(".local/share")
    };
    Ok(base.join("llmtrim").join("tracking.db"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(provider: &str, exact: bool, before: i64, after: i64) -> Record {
        Record {
            provider: provider.to_string(),
            model: Some("m".to_string()),
            tokenizer: "t".to_string(),
            exact,
            input_before: before,
            input_after: after,
            output_before: None,
            output_after: None,
            compress_micros: None,
            cache_read_tokens: None,
        }
    }

    #[test]
    fn record_and_summarize() {
        let t = Tracker::open_in_memory().unwrap();
        t.record(&rec("openai", true, 100, 60)).unwrap();
        t.record(&rec("openai", true, 50, 40)).unwrap();
        t.record(&rec("anthropic", false, 200, 150)).unwrap();

        let s = t.summary().unwrap();
        assert_eq!(s.events, 3);
        assert_eq!(s.input_before, 350);
        assert_eq!(s.input_after, 250);
        assert_eq!(s.saved(), 100);
        assert!((s.saved_pct() - 28.57).abs() < 0.1);
        assert!(s.any_approximate, "an anthropic (approx) row exists");
        assert_eq!(s.by_provider.len(), 2);

        let oa = s
            .by_provider
            .iter()
            .find(|r| r.provider == "openai")
            .unwrap();
        assert_eq!(oa.events, 2);
        assert!(oa.exact);
        let an = s
            .by_provider
            .iter()
            .find(|r| r.provider == "anthropic")
            .unwrap();
        assert!(!an.exact);
    }

    #[test]
    fn empty_ledger_summary_is_zero() {
        let t = Tracker::open_in_memory().unwrap();
        let s = t.summary().unwrap();
        assert_eq!(s.events, 0);
        assert_eq!(s.saved(), 0);
        assert_eq!(s.saved_pct(), 0.0);
        assert!(!s.any_approximate);
        assert_eq!(s.output_before, 0);
        assert_eq!(s.output_after, 0);
        assert_eq!(s.output_events, 0);
    }

    #[test]
    fn output_tokens_round_trip_and_aggregate() {
        let t = Tracker::open_in_memory().unwrap();

        // Row 1: has measured output tokens.
        t.record(&Record {
            provider: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            tokenizer: "tiktoken".to_string(),
            exact: true,
            input_before: 100,
            input_after: 60,
            output_before: None,
            output_after: Some(42),
            compress_micros: Some(300),
            cache_read_tokens: Some(50),
        })
        .unwrap();

        // Row 2: also has measured output tokens.
        t.record(&Record {
            provider: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            tokenizer: "tiktoken".to_string(),
            exact: true,
            input_before: 80,
            input_after: 50,
            output_before: None,
            output_after: Some(17),
            compress_micros: Some(500),
            cache_read_tokens: Some(70),
        })
        .unwrap();

        // Row 3: network-free (no output measurement).
        t.record(&rec("openai", true, 50, 30)).unwrap();

        let s = t.summary().unwrap();

        // Three total events.
        assert_eq!(s.events, 3);

        // Only two rows had output_after set.
        assert_eq!(s.output_events, 2);

        // Sum of the two measured output_after values.
        assert_eq!(s.output_after, 59);

        // output_before stays NULL → sums to 0.
        assert_eq!(s.output_before, 0);

        // Mean compression overhead over the two timed rows (the rec() row is NULL → ignored).
        assert_eq!(s.avg_compress_micros, Some(400.0));
        // Cached-prefix tokens summed over the rows that reported them.
        assert_eq!(s.cache_read_tokens, 120);

        // Per-provider reflects the same aggregation.
        let oa = s
            .by_provider
            .iter()
            .find(|r| r.provider == "openai")
            .unwrap();
        assert_eq!(oa.output_events, 2);
        assert_eq!(oa.output_after, 59);
        assert_eq!(oa.output_before, 0);
    }

    #[test]
    fn prune_caps_to_most_recent_rows() {
        let t = Tracker::open_in_memory().unwrap();
        for _ in 0..10 {
            t.record(&rec("openai", true, 10, 5)).unwrap();
        }
        let deleted = t.prune(4, None).unwrap();
        assert_eq!(deleted, 6, "10 rows capped to 4 → 6 deleted");
        assert_eq!(t.summary().unwrap().events, 4);
    }

    #[test]
    fn prune_drops_rows_older_than_max_age() {
        let t = Tracker::open_in_memory().unwrap();
        // Three ancient rows (explicit old ts), two fresh.
        for _ in 0..3 {
            t.record_with_ts(&rec("openai", true, 10, 5), "2000-01-01T00:00:00+00:00")
                .unwrap();
        }
        t.record(&rec("openai", true, 10, 5)).unwrap();
        t.record(&rec("openai", true, 10, 5)).unwrap();

        let deleted = t.prune(DEFAULT_MAX_ROWS, Some(30)).unwrap();
        assert_eq!(deleted, 3, "only the three >30d-old rows are dropped");
        assert_eq!(t.summary().unwrap().events, 2);
    }

    #[test]
    fn prune_without_age_keeps_old_rows_within_cap() {
        let t = Tracker::open_in_memory().unwrap();
        t.record_with_ts(&rec("openai", true, 10, 5), "2000-01-01T00:00:00+00:00")
            .unwrap();
        // No age policy and under the cap → the ancient row survives.
        let deleted = t.prune(DEFAULT_MAX_ROWS, None).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(t.summary().unwrap().events, 1);
    }
}
