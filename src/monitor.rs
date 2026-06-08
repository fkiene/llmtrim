//! Terminal rendering for the `monitor` command — savings snapshot, live dashboard,
//! time-series reports, and machine-readable export.
//!
//! Pure formatting: `main.rs` gathers the ledger + daemon state + pricing and feeds it
//! here, so this module stays decoupled from the interceptor feature and I/O. Hand-rolled
//! ANSI (no TUI dependency, on-brand with the zero-bloat ethos); colour is passed in by
//! the caller, which disables it for non-TTY stdout or when `NO_COLOR` is set.

use serde_json::json;

use crate::tracking::{PeriodRow, Summary};

// ── view models (built by main from the ledger/daemon/pricing) ──────────────────

/// Interceptor daemon state for the header line.
pub struct DaemonView {
    pub running: bool,
    pub pid: u32,
    pub port: u16,
    pub uptime: String,
    pub ca_present: bool,
}

/// Benchmark output-token reduction from llmtrim's A/B suite (see bench/README). The live
/// proxy can't measure a per-request output baseline (it never sees the un-instructed reply),
/// so the dashboard *projects* the output side from this measured benchmark factor and labels
/// it as such.
const BENCH_OUTPUT_REDUCTION: f64 = 0.73;

/// Projected $ saved: measured input saving + the benchmark-estimated output saving. Compressed
/// output is ~(1−r) of the un-instructed baseline, so the output saving is `out_spend · r/(1−r)`.
fn projected_saved_usd(saved: f64, out_spend: f64) -> f64 {
    saved + out_spend * BENCH_OUTPUT_REDUCTION / (1.0 - BENCH_OUTPUT_REDUCTION)
}

/// Projected round-trip %: projected saving over the projected un-compressed bill.
fn projected_round_trip_pct(saved: f64, spend: f64, out_spend: f64) -> f64 {
    let projected = projected_saved_usd(saved, out_spend);
    let baseline = spend + projected;
    if baseline > 0.0 {
        projected / baseline * 100.0
    } else {
        0.0
    }
}

/// USD cost saved + the compressed spend, priced via the provider registry. `saved`/`spend`
/// are *measured*; the `projected_*` helpers add the benchmark-estimated output saving.
#[derive(Clone, Copy)]
pub struct Cost {
    pub saved: f64,
    pub spend: f64,
    /// The output-token portion of `spend` ($) — what we project the output saving against.
    pub out_spend: f64,
}

impl Cost {
    /// Measured round-trip cost saved as a percentage of the bill — input-side only, since
    /// output savings isn't measurable live (small; understates the real win).
    fn pct(&self) -> f64 {
        let total = self.saved + self.spend;
        if total > 0.0 {
            self.saved / total * 100.0
        } else {
            0.0
        }
    }

    fn projected_saved(&self) -> f64 {
        projected_saved_usd(self.saved, self.out_spend)
    }

    fn projected_pct(&self) -> f64 {
        projected_round_trip_pct(self.saved, self.spend, self.out_spend)
    }
}

/// One per-model row in the breakdown table. `cost_saved`/`spend`/`out_spend` are the measured
/// USD figures (present when the registry prices the model), used to project the per-model
/// round-trip the same way the hero does.
pub struct ModelView {
    pub name: String,
    pub events: i64,
    pub saved_pct: f64,
    pub cost_saved: Option<f64>,
    pub spend: Option<f64>,
    pub out_spend: Option<f64>,
}

// ── ANSI helpers ────────────────────────────────────────────────────────────────

const GREEN: &str = "1;32";
const DIM: &str = "2";
const BOLD: &str = "1";
const YELLOW: &str = "33";

fn paint(color: bool, code: &str, s: &str) -> String {
    if color {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

/// A `width`-cell depletion bar for a `saved` percentage (0–100, clamped): the kept portion
/// is filled + dim (what you still pay), the saved tail is dotted + green (what was cut), so
/// the green dots line up with the green savings label. ≥100 → all dots, ≤0 → all filled.
fn bar(color: bool, saved: f64, width: usize) -> String {
    let cut = ((saved.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    let kept = width.saturating_sub(cut);
    format!(
        "{}{}",
        paint(color, DIM, &"█".repeat(kept)),
        paint(color, GREEN, &"░".repeat(cut)),
    )
}

/// Compact human token count: 1_234_567 → "1.2M", 12_345 → "12.3K".
fn human(n: i64) -> String {
    let a = n.unsigned_abs();
    let sign = if n < 0 { "-" } else { "" };
    if a >= 1_000_000 {
        format!("{sign}{:.1}M", a as f64 / 1_000_000.0)
    } else if a >= 1_000 {
        format!("{sign}{:.1}K", a as f64 / 1_000.0)
    } else {
        format!("{sign}{a}")
    }
}

/// Group digits with commas: 1234567 → "1,234,567".
fn commas(n: i64) -> String {
    let s = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

/// A `saved → label` line with a bar and a signed percentage (green when saving, yellow
/// when it grew). `before`/`after` are token counts.
fn axis(color: bool, name: &str, before: i64, after: i64) -> String {
    let pct = if before > 0 {
        (before - after) as f64 / before as f64 * 100.0
    } else {
        0.0
    };
    let pct_str = format!("{:+.0}%", -pct); // show as a signed delta (-41% = saved 41%)
    let pct_col = if pct >= 0.0 { GREEN } else { YELLOW };
    format!(
        "  {:<7} {} {:>6}   {} → {}",
        name,
        bar(color, pct, 22),
        paint(color, pct_col, &pct_str),
        human(before),
        human(after),
    )
}

// ── snapshot ────────────────────────────────────────────────────────────────────

/// The savings dashboard: daemon header, a hero panel (cost / round-trip / requests),
/// per-axis bars, and a per-model table. Returned as a string so the watch loop can
/// repaint it atomically.
pub fn snapshot(
    color: bool,
    daemon: Option<&DaemonView>,
    s: &Summary,
    models: &[ModelView],
    cost: Option<&Cost>,
) -> String {
    let mut o = String::new();

    // header — daemon state
    if let Some(d) = daemon {
        if d.running {
            o.push_str(&format!(
                " {} {}  {}\n",
                paint(color, GREEN, "llmtrim ●"),
                paint(color, DIM, "running"),
                paint(
                    color,
                    DIM,
                    &format!("pid {} · :{} · up {}", d.pid, d.port, d.uptime)
                ),
            ));
        } else {
            o.push_str(&format!(
                " {} {}\n",
                paint(color, DIM, "llmtrim ○"),
                paint(color, DIM, "stopped — start: llmtrim setup"),
            ));
        }
        if !d.ca_present {
            o.push_str(&paint(color, YELLOW, "  ca missing — run: llmtrim ca\n"));
        }
    }

    if s.events == 0 {
        o.push_str(&paint(
            color,
            DIM,
            "\n no activity yet — run `llmtrim setup`, then use your tools as normal.\n",
        ));
        return o;
    }

    // hero panel
    let hero = match cost {
        Some(c) => format!(
            "{}   {} round-trip   {} requests",
            paint(color, GREEN, &format!("~${:.2} saved", c.projected_saved())),
            paint(color, BOLD, &format!("~-{:.0}%", c.projected_pct())),
            commas(s.events),
        ),
        None => format!(
            "{}   {} requests",
            paint(
                color,
                GREEN,
                &format!("-{:.0}% input tokens", s.saved_pct())
            ),
            commas(s.events),
        ),
    };
    o.push('\n');
    let title = if cost.is_some() {
        "saved (projected · A/B)"
    } else {
        "saved (all time)"
    };
    o.push_str(&panel(color, title, &hero));
    o.push('\n');

    // axes
    o.push_str(&axis(color, "input", s.input_before, s.input_after));
    o.push('\n');
    if s.output_events > 0 {
        // No live output baseline → show the benchmark bar + the real billed volume, tagged.
        o.push_str(&format!(
            "  {:<7} {} {:>6}   {} billed   {}",
            "output",
            bar(color, BENCH_OUTPUT_REDUCTION * 100.0, 22),
            paint(
                color,
                GREEN,
                &format!("~{:+.0}%", -BENCH_OUTPUT_REDUCTION * 100.0)
            ),
            human(s.output_after),
            paint(color, DIM, "(projected)"),
        ));
        o.push('\n');
    }

    // by-model table
    if !models.is_empty() {
        o.push_str(&paint(color, DIM, "\n by model\n"));
        for m in models {
            // Projected round-trip per model when the registry prices it (matches the hero);
            // otherwise the measured input %. The `~` marks the projection.
            let (pct, mark, dollars) = match (m.cost_saved, m.spend, m.out_spend) {
                (Some(saved), Some(spend), Some(out_spend)) => (
                    projected_round_trip_pct(saved, spend, out_spend),
                    "~",
                    Some(projected_saved_usd(saved, out_spend)),
                ),
                _ => (m.saved_pct, "", m.cost_saved),
            };
            let cost_col = dollars
                .map(|c| format!("  {}", paint(color, GREEN, &format!("${c:.2}"))))
                .unwrap_or_default();
            let pct_col = if pct >= 0.0 { GREEN } else { YELLOW };
            o.push_str(&format!(
                "   {:<22} {:>6}  {}{}\n",
                truncate(&m.name, 22),
                commas(m.events),
                paint(color, pct_col, &format!("{mark}{:+.0}%", -pct)),
                cost_col,
            ));
        }
    }

    if s.cache_read_tokens > 0 {
        o.push_str(&paint(
            color,
            GREEN,
            &format!(
                " cache · {} prefix tokens reused, billed at ~10% (the resent context, discounted)\n",
                human(s.cache_read_tokens)
            ),
        ));
    }
    if let Some(us) = s.avg_compress_micros {
        o.push_str(&paint(
            color,
            DIM,
            &format!(
                " added latency ~{:.2} ms/req · llmtrim compression overhead\n",
                us / 1000.0
            ),
        ));
    }
    if s.any_approximate {
        o.push_str(&paint(
            color,
            DIM,
            " * some counts approximate (provider lacks a public tokenizer)\n",
        ));
    }
    if cost.is_some() {
        o.push_str(&paint(
            color,
            DIM,
            " ~ projected: output −73% (A/B bench); input exact. Proxy never sees uncompressed reply — output projected, not measured.\n",
        ));
    }
    o
}

/// A single-line boxed hero panel with a dim title above it.
fn panel(color: bool, title: &str, content: &str) -> String {
    // width by visible content length (strip ANSI for measuring)
    let inner = visible_len(content).max(visible_len(title) + 2) + 2;
    let top = format!(
        "╭─ {} {}╮",
        title,
        "─".repeat(inner.saturating_sub(title.len() + 3))
    );
    let mid = format!(
        "│ {}{} │",
        content,
        " ".repeat(inner.saturating_sub(visible_len(content) + 2))
    );
    let bot = format!("╰{}╯", "─".repeat(inner));
    format!(
        " {}\n {}\n {}\n",
        paint(color, DIM, &top),
        mid,
        paint(color, DIM, &bot)
    )
}

/// Visible length, ignoring ANSI escape sequences.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else if c == '\x1b' {
            in_esc = true;
        } else {
            n += 1;
        }
    }
    n
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max - 1).collect::<String>())
    }
}

// ── time-series report ──────────────────────────────────────────────────────────

/// A `--daily/--weekly/--monthly` table: one row per bucket with input/output savings.
pub fn period_report(color: bool, label: &str, rows: &[PeriodRow]) -> String {
    let mut o = format!(
        "{}\n",
        paint(color, BOLD, &format!("llmtrim — {label} savings"))
    );
    if rows.is_empty() {
        o.push_str(&paint(color, DIM, " no activity recorded yet\n"));
        return o;
    }
    o.push_str(&paint(
        color,
        DIM,
        &format!(
            "  {:<12} {:>6}  {:>16}  {:>16}\n",
            "period", "reqs", "input", "output"
        ),
    ));
    for r in rows {
        let in_pct = pct(r.input_before, r.input_after);
        let out = if r.output_before > 0 {
            format!(
                "{} ({:+.0}%)",
                human(r.output_after),
                -pct(r.output_before, r.output_after)
            )
        } else if r.output_after > 0 {
            human(r.output_after)
        } else {
            "—".to_string()
        };
        o.push_str(&format!(
            "  {:<12} {:>6}  {:>9} {}  {:>16}\n",
            r.bucket,
            commas(r.events),
            format!("{}→{}", human(r.input_before), human(r.input_after)),
            paint(color, GREEN, &format!("{:+.0}%", -in_pct)),
            out,
        ));
    }
    o
}

fn pct(before: i64, after: i64) -> f64 {
    if before > 0 {
        (before - after) as f64 / before as f64 * 100.0
    } else {
        0.0
    }
}

// ── machine-readable export ─────────────────────────────────────────────────────

/// Full snapshot as JSON (for Grafana/Prometheus/scripts).
pub fn export_json(
    s: &Summary,
    models: &[ModelView],
    cost: Option<&Cost>,
    periods: &[PeriodRow],
) -> String {
    let v = json!({
        "requests": s.events,
        "input": { "before": s.input_before, "after": s.input_after, "saved_pct": s.saved_pct() },
        "output": { "before": s.output_before, "after": s.output_after,
                    "events": s.output_events, "saved_pct": s.output_saved_pct() },
        "cost": cost.map(|c| json!({ "saved_usd": c.saved, "spend_usd": c.spend, "round_trip_pct": c.pct(),
                                     "projected_saved_usd": c.projected_saved(), "projected_round_trip_pct": c.projected_pct() })),
        "added_latency_ms": s.avg_compress_micros.map(|us| us / 1000.0),
        "cache_read_tokens": s.cache_read_tokens,
        "approximate": s.any_approximate,
        "by_model": models.iter().map(|m| json!({
            "model": m.name, "requests": m.events, "saved_pct": m.saved_pct, "cost_saved_usd": m.cost_saved,
        })).collect::<Vec<_>>(),
        "by_period": periods.iter().map(|p| json!({
            "period": p.bucket, "requests": p.events,
            "input_before": p.input_before, "input_after": p.input_after,
            "output_before": p.output_before, "output_after": p.output_after,
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".to_string())
}

/// Time-series as CSV (spreadsheet/Prometheus-friendly).
pub fn export_csv(periods: &[PeriodRow]) -> String {
    let mut o =
        String::from("period,requests,input_before,input_after,output_before,output_after\n");
    for p in periods {
        o.push_str(&format!(
            "{},{},{},{},{},{}\n",
            p.bucket, p.events, p.input_before, p.input_after, p.output_before, p.output_after
        ));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summ() -> Summary {
        Summary {
            events: 1204,
            input_before: 2_100_000,
            input_after: 1_240_000,
            any_approximate: false,
            by_provider: vec![],
            output_before: 880_000,
            output_after: 229_000,
            output_events: 1204,
            avg_compress_micros: Some(310.0),
            cache_read_tokens: 1_200_000,
        }
    }

    #[test]
    fn human_and_commas() {
        assert_eq!(human(1_234_567), "1.2M");
        assert_eq!(human(12_345), "12.3K");
        assert_eq!(human(512), "512");
        assert_eq!(commas(1_234_567), "1,234,567");
    }

    #[test]
    fn bar_clamps_and_fills() {
        assert_eq!(bar(false, 50.0, 10), "█████░░░░░"); // 50% saved: 5 kept, 5 cut
        assert_eq!(bar(false, 0.0, 4), "████"); // nothing saved → all filled
        assert_eq!(bar(false, 100.0, 4), "░░░░"); // all saved → all dotted
        assert_eq!(bar(false, 150.0, 4), "░░░░"); // clamp high → all dotted
        assert_eq!(bar(false, -20.0, 4), "████"); // clamp low → all filled
    }

    #[test]
    fn snapshot_plain_has_hero_and_axes() {
        let cost = Cost {
            saved: 12.47,
            spend: 9.0,
            out_spend: 0.0,
        };
        let models = vec![ModelView {
            name: "gpt-4o".into(),
            events: 420,
            saved_pct: 61.0,
            cost_saved: Some(4.10),
            spend: Some(6.0),
            out_spend: Some(0.0),
        }];
        let out = snapshot(false, None, &summ(), &models, Some(&cost));
        assert!(out.contains("$12.47 saved"), "hero cost");
        assert!(out.contains("1,204 requests"), "request count");
        assert!(
            out.contains("input") && out.contains("2.1M → 1.2M"),
            "input axis"
        );
        assert!(out.contains("gpt-4o") && out.contains("$4.10"), "model row");
        assert!(out.contains("projected"), "projected label present");
        assert!(out.contains("ms/req"), "added-latency line");
        assert!(!out.contains('\x1b'), "no ANSI when color=false");
    }

    #[test]
    fn projects_output_saving_from_benchmark() {
        // out baseline = 0.27 / (1 − 0.73) = 1.0, so projected output saved = 1.0 − 0.27 = 0.73.
        let c = Cost {
            saved: 1.0,
            spend: 1.0,
            out_spend: 0.27,
        };
        assert!((c.projected_saved() - 1.73).abs() < 1e-9);
        assert!((c.projected_pct() - 1.73 / 2.73 * 100.0).abs() < 1e-9);
    }

    #[test]
    fn snapshot_empty_ledger_guides_user() {
        let out = snapshot(false, None, &Summary::default(), &[], None);
        assert!(out.contains("no activity yet"));
    }

    #[test]
    fn export_json_roundtrips() {
        let out = export_json(&summ(), &[], None, &[]);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["requests"], 1204);
        assert_eq!(v["cost"], serde_json::Value::Null);
    }

    #[test]
    fn export_csv_has_header_and_rows() {
        let rows = vec![PeriodRow {
            bucket: "2026-06-07".into(),
            events: 10,
            input_before: 100,
            input_after: 60,
            output_before: 0,
            output_after: 20,
        }];
        let out = export_csv(&rows);
        assert!(out.starts_with("period,requests,"));
        assert!(out.contains("2026-06-07,10,100,60,0,20"));
    }
}
