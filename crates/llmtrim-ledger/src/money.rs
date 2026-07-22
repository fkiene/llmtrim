//! Canonical money totals from `breakdown_turns` frozen rates.
//!
//! **Paid** is always `SUM(bill_micros)` — the same number Sessions/Detail/Tray use.
//! **Saved** is the input-side counterfactual (blend, per turn, frozen rates), so
//! `would_have = paid + saved` holds by construction. Output is in paid, not saved.
//!
//! Compressions remain the source for token volume / latency / frozen-zone gauges;
//! this module is the sole path for dollars.

use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use serde::Serialize;

use crate::breakdown_db::BreakdownDb;

/// Maximum compression fraction applied in the counterfactual (matches `cost_estimate`).
pub const PCT_CLAMP: f64 = 0.95;

/// Inclusive lower bound on `breakdown_turns.ts` (rfc3339 UTC), or all time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoneyWindow {
    #[default]
    All,
    /// Rows with `ts >= start` (lexical == chronological for rfc3339 UTC).
    SinceTs,
}

/// Aggregated money for a window on `breakdown_turns`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MoneyTotals {
    /// Actual bill: `SUM(bill_micros)`.
    pub paid_micros: i64,
    /// Input-side counterfactual savings (blend, per-turn, frozen rates).
    pub saved_micros: i64,
    /// `paid_micros + saved_micros` (output unchanged in the counterfactual).
    pub would_have_micros: i64,
    /// Turns in the money window.
    pub turns: i64,
    /// Turns with usage tokens but `bill_micros == 0` (unpriced model / zero rates).
    pub turns_unpriced: i64,
    /// Distinct models contributing to paid/saved.
    pub models_priced: i64,
}

impl MoneyTotals {
    pub fn paid_usd(&self) -> f64 {
        self.paid_micros as f64 / 1_000_000.0
    }
    pub fn saved_usd(&self) -> f64 {
        self.saved_micros as f64 / 1_000_000.0
    }
    pub fn would_have_usd(&self) -> f64 {
        self.would_have_micros as f64 / 1_000_000.0
    }

    /// True when at least one turn contributed a non-zero bill or savings estimate.
    pub fn has_money(&self) -> bool {
        self.turns > 0 && (self.paid_micros > 0 || self.saved_micros > 0 || self.turns_unpriced > 0)
    }

    /// True when there are turns but every turn is unpriced (all bills zero with usage).
    pub fn all_unpriced(&self) -> bool {
        self.turns > 0 && self.paid_micros == 0 && self.turns_unpriced == self.turns
    }
}

/// Coverage of money attribution vs the compressions request log.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MoneyCoverage {
    pub compressions_events: i64,
    pub breakdown_turns: i64,
    /// `breakdown_turns / compressions_events` when compressions > 0; else 1.0 if both 0.
    pub coverage_ratio: f64,
}

impl MoneyCoverage {
    pub fn compute(compressions_events: i64, breakdown_turns: i64) -> Self {
        // No compressions → ratio 1.0 (empty or breakdown-only is fine). With compressions,
        // ratio is turns/events clamped to [0, 1].
        let coverage_ratio = if compressions_events > 0 {
            (breakdown_turns as f64 / compressions_events as f64).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Self {
            compressions_events,
            breakdown_turns,
            coverage_ratio,
        }
    }

    /// Compressions exist but no billed turns — Overview must not show $0 as truth.
    pub fn empty_money_with_traffic(&self) -> bool {
        self.compressions_events > 0 && self.breakdown_turns == 0
    }

    pub fn partial(&self) -> bool {
        self.compressions_events > 0
            && self.breakdown_turns > 0
            && self.breakdown_turns < self.compressions_events
    }
}

/// One calendar day of money (UTC date key `YYYY-MM-DD`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoneyByDay {
    pub day: String,
    pub paid_micros: i64,
    pub saved_micros: i64,
}

/// Per-model money rollup.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoneyByModel {
    pub model: String,
    pub paid_micros: i64,
    pub saved_micros: i64,
    pub turns: i64,
}

// ── pure per-turn formula (unit-tested; SQL mirrors this) ───────────────────────

/// Compression fraction for one turn, clamped to `[0, PCT_CLAMP]`.
///
/// Returns 0 when meters are missing or non-positive before.
pub fn turn_pct(input_before: Option<i64>, input_after: Option<i64>) -> f64 {
    let Some(before) = input_before.filter(|&b| b > 0) else {
        return 0.0;
    };
    let after = input_after.unwrap_or(before).max(0);
    let cut = (before - after).max(0) as f64;
    (cut / before as f64).clamp(0.0, PCT_CLAMP)
}

/// Frozen input-side rates for one turn (USD per 1M tokens), plus usage counts for pricing.
#[derive(Debug, Clone, Copy)]
pub struct TurnInputBill {
    pub fresh_input: i64,
    pub cache_read: i64,
    pub cache_write: i64,
    pub input_rate: f64,
    pub cache_read_rate: f64,
    pub cache_write_rate: f64,
}

/// Input-side bill in micro-USD from frozen rates (excludes output).
pub fn input_bill_micros(b: TurnInputBill) -> i64 {
    (b.fresh_input as f64 * b.input_rate
        + b.cache_read as f64 * b.cache_read_rate
        + b.cache_write as f64 * b.cache_write_rate)
        .round() as i64
}

/// Blend counterfactual savings for one turn (micro-USD).
///
/// `saved = input_bill × pct / (1 − pct)`. Same shape as `cost_estimate::net_saved`,
/// but rates are the turn's frozen rates. **Blend** understates pure live-zone savings
/// on cache-heavy traffic; that is intentional and labeled in the UI.
pub fn turn_saved_micros(
    input_before: Option<i64>,
    input_after: Option<i64>,
    bill: TurnInputBill,
) -> i64 {
    let pct = turn_pct(input_before, input_after);
    if pct <= 0.0 {
        return 0;
    }
    let input_bill = input_bill_micros(bill) as f64;
    if input_bill <= 0.0 {
        return 0;
    }
    // pct is clamped to PCT_CLAMP so (1-pct) >= 0.05.
    (input_bill * pct / (1.0 - pct)).round() as i64
}

/// SQL expression for per-turn saved micros (mirrors [`turn_saved_micros`]).
///
/// `alias` is the table/CTE qualifier including a trailing `.` when non-empty
/// (e.g. `"t."` or `""` for bare column names in a CTE select list).
/// Used by [`money_totals`] **and** [`BreakdownDb::sessions`] so Overview and
/// Sessions cannot drift.
pub fn saved_micros_sql(alias: &str) -> String {
    // pct = min(0.95, max(0, (before-after)/before)) when before > 0
    // input_bill = fresh*input_rate + read*cache_read_rate + write*cache_write_rate
    // saved = input_bill * pct / (1-pct)
    let a = alias;
    format!(
        "CASE
            WHEN {a}input_before IS NULL OR {a}input_before <= 0 THEN 0
            ELSE CAST(ROUND(
                MAX(0.0,
                    {a}fresh_input * {a}input_rate
                    + {a}cache_read * {a}cache_read_rate
                    + {a}cache_write * {a}cache_write_rate
                )
                * MIN({clamp}, MAX(0.0,
                    ({a}input_before - COALESCE({a}input_after, {a}input_before)) * 1.0
                    / {a}input_before
                  ))
                / (1.0 - MIN({clamp}, MAX(0.0,
                    ({a}input_before - COALESCE({a}input_after, {a}input_before)) * 1.0
                    / {a}input_before
                  )))
            ) AS INTEGER)
         END",
        clamp = PCT_CLAMP
    )
}

/// Money totals over all time (or since `since_ts` when set).
///
/// `since_ts` is an inclusive lower bound on `ts` (rfc3339 UTC). Does **not** scan sessions.
pub fn money_totals(conn: &Connection, since_ts: Option<&str>) -> Result<MoneyTotals> {
    let saved = saved_micros_sql("t.");
    let (sql, use_since) = if since_ts.is_some() {
        (
            format!(
                "SELECT
                    COALESCE(SUM(t.bill_micros), 0),
                    COALESCE(SUM({saved}), 0),
                    COUNT(*),
                    COALESCE(SUM(CASE
                        WHEN t.bill_micros = 0
                         AND (t.fresh_input + t.cache_read + t.cache_write + t.output_tok) > 0
                        THEN 1 ELSE 0 END), 0),
                    COUNT(DISTINCT CASE
                        WHEN t.bill_micros > 0 OR ({saved}) > 0
                        THEN COALESCE(t.model, '') END)
                 FROM breakdown_turns t
                 WHERE t.ts >= ?1"
            ),
            true,
        )
    } else {
        (
            format!(
                "SELECT
                    COALESCE(SUM(t.bill_micros), 0),
                    COALESCE(SUM({saved}), 0),
                    COUNT(*),
                    COALESCE(SUM(CASE
                        WHEN t.bill_micros = 0
                         AND (t.fresh_input + t.cache_read + t.cache_write + t.output_tok) > 0
                        THEN 1 ELSE 0 END), 0),
                    COUNT(DISTINCT CASE
                        WHEN t.bill_micros > 0 OR ({saved}) > 0
                        THEN COALESCE(t.model, '') END)
                 FROM breakdown_turns t"
            ),
            false,
        )
    };
    let map = |row: &rusqlite::Row<'_>| -> rusqlite::Result<MoneyTotals> {
        let paid: i64 = row.get(0)?;
        let saved_m: i64 = row.get(1)?;
        Ok(MoneyTotals {
            paid_micros: paid,
            saved_micros: saved_m,
            would_have_micros: paid.saturating_add(saved_m),
            turns: row.get(2)?,
            turns_unpriced: row.get(3)?,
            models_priced: row.get(4)?,
        })
    };
    if use_since {
        conn.query_row(&sql, params![since_ts.unwrap()], map)
            .context("failed to query money_totals (since)")
    } else {
        conn.query_row(&sql, [], map)
            .context("failed to query money_totals")
    }
}

/// Last `n_days` calendar days (UTC) with activity, newest-first then reversed to oldest→newest.
pub fn money_by_day(conn: &Connection, n_days: usize) -> Result<Vec<MoneyByDay>> {
    if n_days == 0 {
        return Ok(Vec::new());
    }
    let saved = saved_micros_sql("t.");
    let sql = format!(
        "SELECT substr(t.ts, 1, 10) AS day,
                COALESCE(SUM(t.bill_micros), 0),
                COALESCE(SUM({saved}), 0)
         FROM breakdown_turns t
         GROUP BY day
         ORDER BY day DESC
         LIMIT ?1"
    );
    let mut stmt = conn
        .prepare(&sql)
        .context("failed to prepare money_by_day")?;
    let rows = stmt
        .query_map(params![n_days as i64], |r| {
            Ok(MoneyByDay {
                day: r.get(0)?,
                paid_micros: r.get(1)?,
                saved_micros: r.get(2)?,
            })
        })
        .context("failed to query money_by_day")?;
    let mut out: Vec<MoneyByDay> = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("failed to read money_by_day")?;
    out.reverse();
    Ok(out)
}

/// Top models by saved micros (then paid), limit `n`.
pub fn money_by_model(conn: &Connection, n: usize) -> Result<Vec<MoneyByModel>> {
    let saved = saved_micros_sql("t.");
    let sql = format!(
        "SELECT COALESCE(t.model, '(unknown)'),
                COALESCE(SUM(t.bill_micros), 0),
                COALESCE(SUM({saved}), 0),
                COUNT(*)
         FROM breakdown_turns t
         GROUP BY t.model
         ORDER BY 3 DESC, 2 DESC
         LIMIT ?1"
    );
    let mut stmt = conn
        .prepare(&sql)
        .context("failed to prepare money_by_model")?;
    let rows = stmt
        .query_map(params![n as i64], |r| {
            Ok(MoneyByModel {
                model: r.get(0)?,
                paid_micros: r.get(1)?,
                saved_micros: r.get(2)?,
                turns: r.get(3)?,
            })
        })
        .context("failed to query money_by_model")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .context("failed to read money_by_model")
}

/// Paid micros for one session (canonical Detail footer).
pub fn session_bill_micros(conn: &Connection, session_id: &str) -> Result<i64> {
    conn.query_row(
        "SELECT COALESCE(SUM(bill_micros), 0) FROM breakdown_turns WHERE session_id = ?1",
        params![session_id],
        |r| r.get(0),
    )
    .context("failed to query session_bill_micros")
}

/// Coverage: compressions events vs breakdown turns (same DB file).
pub fn money_coverage(conn: &Connection) -> Result<MoneyCoverage> {
    let compressions: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='compressions'",
            [],
            |r| r.get(0),
        )
        .context("failed to probe compressions table")?;
    let compressions_events = if compressions > 0 {
        conn.query_row("SELECT COUNT(*) FROM compressions", [], |r| r.get(0))
            .context("failed to count compressions")?
    } else {
        0
    };
    let breakdown_turns: i64 = conn
        .query_row("SELECT COUNT(*) FROM breakdown_turns", [], |r| r.get(0))
        .context("failed to count breakdown_turns")?;
    Ok(MoneyCoverage::compute(compressions_events, breakdown_turns))
}

impl BreakdownDb {
    /// Ensure the ts index used by money range queries exists (idempotent).
    pub fn ensure_money_indexes(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_breakdown_turns_ts ON breakdown_turns(ts);",
            )
            .context("failed to create breakdown_turns ts index")?;
        Ok(())
    }

    pub fn money_totals(&self, since_ts: Option<&str>) -> Result<MoneyTotals> {
        money_totals(&self.conn, since_ts)
    }

    pub fn money_by_day(&self, n_days: usize) -> Result<Vec<MoneyByDay>> {
        money_by_day(&self.conn, n_days)
    }

    pub fn money_by_model(&self, n: usize) -> Result<Vec<MoneyByModel>> {
        money_by_model(&self.conn, n)
    }

    pub fn session_bill_micros(&self, session_id: &str) -> Result<i64> {
        session_bill_micros(&self.conn, session_id)
    }

    pub fn money_coverage(&self) -> Result<MoneyCoverage> {
        money_coverage(&self.conn)
    }
}

/// Start of the current **UTC** calendar day as an rfc3339 lower bound.
/// Documented UTC (not local) so "today" matches `by_model_today` / day buckets.
pub fn today_start_utc() -> String {
    let d = chrono::Utc::now().date_naive();
    format!("{d}T00:00:00+00:00")
}

/// Pad `money_by_day` results to exactly `n` trailing UTC days (zeros for quiet days).
pub fn pad_daily_saved(days: &[MoneyByDay], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let today = chrono::Utc::now().date_naive();
    let mut map: std::collections::HashMap<&str, i64> = std::collections::HashMap::new();
    for d in days {
        map.insert(d.day.as_str(), d.saved_micros);
    }
    (0..n)
        .map(|i| {
            let day = today - chrono::Duration::days((n - 1 - i) as i64);
            let key = day.format("%Y-%m-%d").to_string();
            map.get(key.as_str()).copied().unwrap_or(0) as f64 / 1_000_000.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::{BreakdownBlock, BreakdownTurn, Tracker};

    struct SeedTurn {
        session: &'static str,
        model: &'static str,
        bill: i64,
        before: i64,
        after: i64,
        fresh: i64,
        cache_read: i64,
        out: i64,
        rates: (f64, f64, f64, f64),
    }

    fn turn(s: SeedTurn) -> BreakdownTurn {
        BreakdownTurn {
            session_id: s.session.into(),
            cc_session_id: None,
            agent: "claude-code".into(),
            project: Some("/proj".into()),
            session_name: Some("s".into()),
            provider: "anthropic".into(),
            model: Some(s.model.into()),
            sub_provider: None,
            window: 200_000,
            fresh_input: s.fresh,
            cache_read: s.cache_read,
            cache_write: 0,
            output_tok: s.out,
            input_rate: s.rates.0,
            output_rate: s.rates.1,
            cache_read_rate: s.rates.2,
            cache_write_rate: s.rates.3,
            bill_micros: s.bill,
            input_before: s.before,
            input_after: s.after,
        }
    }

    fn bill(
        fresh: i64,
        cache_read: i64,
        cache_write: i64,
        input_rate: f64,
        cache_read_rate: f64,
        cache_write_rate: f64,
    ) -> TurnInputBill {
        TurnInputBill {
            fresh_input: fresh,
            cache_read,
            cache_write,
            input_rate,
            cache_read_rate,
            cache_write_rate,
        }
    }

    fn seed(t: &Tracker, turns: &[BreakdownTurn]) {
        for tr in turns {
            t.record_breakdown(tr, &[]).unwrap();
        }
    }

    #[test]
    fn turn_pct_clamps_and_null_safe() {
        assert_eq!(turn_pct(None, Some(10)), 0.0);
        assert_eq!(turn_pct(Some(0), Some(0)), 0.0);
        assert!((turn_pct(Some(100), Some(50)) - 0.5).abs() < 1e-9);
        assert!((turn_pct(Some(100), Some(0)) - PCT_CLAMP).abs() < 1e-9);
        assert_eq!(turn_pct(Some(100), Some(100)), 0.0);
        assert_eq!(turn_pct(Some(100), Some(150)), 0.0); // grew
    }

    #[test]
    fn per_turn_saved_not_equal_to_aggregate_formula() {
        // Asymmetric cache mixes make sum(per-turn) diverge from formula-on-sums.
        // Turn A: heavy cache, small cut. Turn B: all-fresh, large cut.
        let r = (3.0, 15.0, 0.3, 3.75);
        let s1 = turn_saved_micros(
            Some(100_000),
            Some(95_000),
            bill(5_000, 90_000, 0, r.0, r.2, r.3),
        );
        let s2 = turn_saved_micros(
            Some(100_000),
            Some(40_000),
            bill(40_000, 0, 0, r.0, r.2, r.3),
        );
        let sum = s1 + s2;
        // Wrong aggregate: sum tokens then one pct.
        let before = 200_000.0;
        let after = 135_000.0;
        let agg_bill = input_bill_micros(bill(45_000, 90_000, 0, r.0, r.2, r.3)) as f64;
        let agg_pct = (before - after) / before;
        let agg = (agg_bill * agg_pct / (1.0 - agg_pct)).round() as i64;
        assert_ne!(
            sum, agg,
            "Jensen: per-turn sum must differ from aggregate ({sum} vs {agg})"
        );
        assert!(sum > 0 && s2 > s1);
    }

    #[test]
    fn cache_blend_understates_live_cut_documented() {
        // 90k cache-read @ 0.1×, 10k fresh @ 1×, cut 10k live → before=110k after=100k
        // rates: input=3, cache_read=0.3 ($/1M micro convention)
        let input_rate = 3.0;
        let cache_read_rate = 0.3;
        let saved = turn_saved_micros(
            Some(110_000),
            Some(100_000),
            bill(10_000, 90_000, 0, input_rate, cache_read_rate, 0.0),
        );
        let live_true = (10_000.0 * input_rate).round() as i64; // cut at full rate
        assert!(
            saved < live_true / 2,
            "blend must understate live cut: saved={saved} live={live_true}"
        );
        // Lock expected so the understatement is intentional, not accidental drift.
        // input_bill = 10k*3 + 90k*0.3 = 30000+27000 = 57000
        // pct = 10000/110000 ≈ 0.090909
        // saved = 57000 * pct/(1-pct) ≈ 5700
        assert!(
            (saved - 5700).abs() < 5,
            "expected ~5700 blend micros, got {saved}"
        );
    }

    #[test]
    fn money_totals_matches_sum_of_bills() {
        let t = Tracker::open_in_memory().unwrap();
        let rates = (3.0, 15.0, 0.3, 3.75);
        // bill = fresh*3 + out*15
        let bill1 = (50_000.0_f64 * 3.0 + 100.0 * 15.0).round() as i64;
        let bill2 = (80_000.0_f64 * 3.0 + 200.0 * 15.0).round() as i64;
        seed(
            &t,
            &[
                turn(SeedTurn {
                    session: "s1",
                    model: "m1",
                    bill: bill1,
                    before: 100_000,
                    after: 50_000,
                    fresh: 50_000,
                    cache_read: 0,
                    out: 100,
                    rates,
                }),
                turn(SeedTurn {
                    session: "s2",
                    model: "m1",
                    bill: bill2,
                    before: 100_000,
                    after: 80_000,
                    fresh: 80_000,
                    cache_read: 0,
                    out: 200,
                    rates,
                }),
            ],
        );
        let db = BreakdownDb::from_connection(t.into_connection());
        let m = db.money_totals(None).unwrap();
        assert_eq!(m.paid_micros, bill1 + bill2);
        assert_eq!(m.turns, 2);
        assert_eq!(m.would_have_micros, m.paid_micros + m.saved_micros);
        assert!(m.saved_micros > 0);

        // Pure formula must match SQL for each turn summed.
        let s1 = turn_saved_micros(
            Some(100_000),
            Some(50_000),
            bill(50_000, 0, 0, rates.0, rates.2, rates.3),
        );
        let s2 = turn_saved_micros(
            Some(100_000),
            Some(80_000),
            bill(80_000, 0, 0, rates.0, rates.2, rates.3),
        );
        assert_eq!(m.saved_micros, s1 + s2);
    }

    #[test]
    fn unpriced_and_premeter() {
        let t = Tracker::open_in_memory().unwrap();
        // Unpriced: usage but bill 0, rates 0
        let unpriced = turn(SeedTurn {
            session: "s1",
            model: "mystery",
            bill: 0,
            before: 1000,
            after: 500,
            fresh: 500,
            cache_read: 0,
            out: 10,
            rates: (0.0, 0.0, 0.0, 0.0),
        });
        // Premeter-like: no compression meters (0 before → pct 0), but has bill
        let mut premeter = turn(SeedTurn {
            session: "s2",
            model: "m1",
            bill: 1000,
            before: 0,
            after: 0,
            fresh: 100,
            cache_read: 0,
            out: 0,
            rates: (3.0, 15.0, 0.3, 3.75),
        });
        premeter.input_before = 0;
        premeter.input_after = 0;
        seed(&t, &[unpriced, premeter]);
        let db = BreakdownDb::from_connection(t.into_connection());
        let m = db.money_totals(None).unwrap();
        assert_eq!(m.turns, 2);
        assert_eq!(m.turns_unpriced, 1);
        assert_eq!(m.paid_micros, 1000);
        assert_eq!(m.saved_micros, 0);
    }

    #[test]
    fn coverage_empty_money_with_traffic() {
        let c = MoneyCoverage::compute(10, 0);
        assert!(c.empty_money_with_traffic());
        assert!(!c.partial());
        let p = MoneyCoverage::compute(10, 4);
        assert!(p.partial());
        assert!((p.coverage_ratio - 0.4).abs() < 1e-9);
    }

    #[test]
    fn session_bill_is_canonical_paid() {
        let t = Tracker::open_in_memory().unwrap();
        let rates = (3.0, 15.0, 0.3, 3.75);
        let bill = 12_345;
        seed(
            &t,
            &[turn(SeedTurn {
                session: "sess-a",
                model: "m1",
                bill,
                before: 1000,
                after: 800,
                fresh: 800,
                cache_read: 0,
                out: 0,
                rates,
            })],
        );
        // Also write a block that would reprice differently if Detail used block sum only
        let tr = turn(SeedTurn {
            session: "sess-a",
            model: "m1",
            bill,
            before: 1000,
            after: 800,
            fresh: 800,
            cache_read: 0,
            out: 0,
            rates,
        });
        // re-open path: second turn same session
        let t2 = Tracker::open_in_memory().unwrap();
        let block = BreakdownBlock {
            zone: "messages".into(),
            section: "user".into(),
            bucket: "text".into(),
            group_label: "Messages".into(),
            label: "user".into(),
            raw_tokens: 800,
            fresh_tok: 800.0,
            ..Default::default()
        };
        t2.record_breakdown(&tr, &[block]).unwrap();
        let db = BreakdownDb::from_connection(t2.into_connection());
        assert_eq!(db.session_bill_micros("sess-a").unwrap(), bill);
        let m = db.money_totals(None).unwrap();
        assert_eq!(m.paid_micros, bill);
    }

    #[test]
    fn money_by_model_orders_by_saved() {
        let t = Tracker::open_in_memory().unwrap();
        let rates = (3.0, 15.0, 0.3, 3.75);
        let b = (50_000.0_f64 * 3.0).round() as i64;
        seed(
            &t,
            &[
                turn(SeedTurn {
                    session: "s1",
                    model: "cheap",
                    bill: b,
                    before: 100_000,
                    after: 90_000,
                    fresh: 50_000,
                    cache_read: 0,
                    out: 0,
                    rates,
                }),
                turn(SeedTurn {
                    session: "s2",
                    model: "saver",
                    bill: b,
                    before: 100_000,
                    after: 40_000,
                    fresh: 50_000,
                    cache_read: 0,
                    out: 0,
                    rates,
                }),
            ],
        );
        let db = BreakdownDb::from_connection(t.into_connection());
        let models = db.money_by_model(8).unwrap();
        assert_eq!(models[0].model, "saver");
        assert!(models[0].saved_micros > models[1].saved_micros);
    }

    #[test]
    fn pad_daily_saved_length() {
        let days = vec![MoneyByDay {
            day: chrono::Utc::now()
                .date_naive()
                .format("%Y-%m-%d")
                .to_string(),
            paid_micros: 0,
            saved_micros: 1_000_000,
        }];
        let v = pad_daily_saved(&days, 7);
        assert_eq!(v.len(), 7);
        assert!((v[6] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sessions_sum_matches_money_totals() {
        let t = Tracker::open_in_memory().unwrap();
        let rates = (3.0, 15.0, 0.3, 3.75);
        let bill1 = (50_000.0_f64 * 3.0 + 100.0 * 15.0).round() as i64;
        let bill2 = (20_000.0_f64 * 3.0).round() as i64;
        seed(
            &t,
            &[
                turn(SeedTurn {
                    session: "s1",
                    model: "m1",
                    bill: bill1,
                    before: 100_000,
                    after: 50_000,
                    fresh: 50_000,
                    cache_read: 0,
                    out: 100,
                    rates,
                }),
                turn(SeedTurn {
                    session: "s2",
                    model: "m1",
                    bill: bill2,
                    before: 40_000,
                    after: 20_000,
                    fresh: 20_000,
                    cache_read: 0,
                    out: 0,
                    rates,
                }),
            ],
        );
        let conn = t.into_connection();
        let money = money_totals(&conn, None).unwrap();
        let db = BreakdownDb::from_connection(conn);
        let sessions = db.sessions().unwrap();
        let bill_sum: i64 = sessions.iter().map(|s| s.bill_micros).sum();
        let saved_sum: i64 = sessions.iter().map(|s| s.saved_micros).sum();
        assert_eq!(bill_sum, money.paid_micros);
        assert_eq!(saved_sum, money.saved_micros);
        assert_eq!(
            money.would_have_micros,
            money.paid_micros + money.saved_micros
        );
    }

    #[test]
    fn compressions_only_coverage_is_empty_money() {
        use crate::tracking::Record;
        let t = Tracker::open_in_memory().unwrap();
        t.record(&Record {
            provider: "openai".into(),
            model: Some("gpt-4o".into()),
            tokenizer: "tiktoken".into(),
            exact: true,
            input_before: 1000,
            input_after: 600,
            output_before: None,
            output_after: None,
            compress_micros: None,
            cache_read_tokens: None,
            fresh_input_tokens: None,
            cache_write_tokens: None,
            output_shaped: Some(false),
            frozen_input_tokens: Some(0),
            outcome: None,
        })
        .unwrap();
        let cov = money_coverage(t.connection()).unwrap();
        assert!(cov.empty_money_with_traffic());
        assert_eq!(cov.breakdown_turns, 0);
        let m = money_totals(t.connection(), None).unwrap();
        assert_eq!(m.turns, 0);
        assert_eq!(m.paid_micros, 0);
    }
}
