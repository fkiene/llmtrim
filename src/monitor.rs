//! Terminal rendering for the `monitor` command — savings snapshot, live dashboard,
//! time-series reports, and machine-readable export.
//!
//! Pure formatting: `main.rs` gathers the ledger + daemon state + pricing and feeds it
//! here, so this module stays decoupled from the interceptor feature and I/O. All
//! styling goes through `crate::ui`; colour is passed in by the caller, which disables
//! it for non-TTY stdout or when `NO_COLOR` is set.

use serde_json::json;

use crate::tracking::{PeriodRow, Summary};
use crate::ui::{self, Tone};

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

// ── rendering helpers ───────────────────────────────────────────────────────────

/// A `width`-cell depletion bar for a `saved` percentage (0–100, clamped): the kept portion
/// is filled + dim (what you still pay), the saved tail is dotted + the accent (what was cut),
/// so the accent dots line up with the accent savings label. ≥100 → all dots, ≤0 → all filled.
fn bar(color: bool, saved: f64, width: usize) -> String {
    let cut = ((saved.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    let kept = width.saturating_sub(cut);
    format!(
        "{}{}",
        ui::paint(color, Tone::Dim, &"█".repeat(kept)),
        ui::paint(color, Tone::Accent, &"░".repeat(cut)),
    )
}

/// A `saved → label` line with a bar and a signed percentage (accent when saving, warn
/// when it grew). `before`/`after` are token counts.
fn axis(color: bool, name: &str, before: i64, after: i64) -> String {
    let pct = ui::saved_pct(before as f64, after as f64);
    let pct_str = format!("{:+.0}%", -pct); // show as a signed delta (-41% = saved 41%)
    let pct_tone = if pct >= 0.0 { Tone::Accent } else { Tone::Warn };
    format!(
        "  {:<7} {} {:>6}   {} → {}",
        name,
        bar(color, pct, 22),
        ui::paint(color, pct_tone, &pct_str),
        ui::human(before),
        ui::human(after),
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
                ui::paint(color, Tone::Accent, "llmtrim ●"),
                ui::paint(color, Tone::Dim, "running"),
                ui::paint(
                    color,
                    Tone::Dim,
                    &format!("pid {} · :{} · up {}", d.pid, d.port, d.uptime)
                ),
            ));
        } else {
            o.push_str(&format!(
                " {} {}\n",
                ui::paint(color, Tone::Dim, "llmtrim ○"),
                ui::paint(color, Tone::Dim, "stopped — start: llmtrim setup"),
            ));
        }
        if !d.ca_present {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                "  ca missing — run: llmtrim ca\n",
            ));
        }
    }

    if s.events == 0 {
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            "\n no activity yet — run `llmtrim setup`, then use your tools as normal.\n",
        ));
        return o;
    }

    // Hero panel — the headline is the MEASURED input-side saving (real, per-row): every
    // request's input is compressed and re-tokenized, so this is honest for all traffic.
    // The output side is NOT in the headline: the proxy never sees the un-instructed reply,
    // so any output saving is a benchmark projection that only holds when output is actually
    // shaped — projecting it onto agent traffic (output left unshaped by design) would
    // overstate the number ~2.7×. We surface it separately, clearly labeled, below.
    let hero = match cost {
        Some(c) => format!(
            "{}   {} round-trip   {} requests",
            ui::paint(color, Tone::Accent, &format!("${:.2} saved", c.saved)),
            ui::paint(color, Tone::Bold, &format!("-{:.0}%", c.pct())),
            ui::commas(s.events),
        ),
        None => format!(
            "{}   {} requests",
            ui::paint(
                color,
                Tone::Accent,
                &format!("-{:.0}% input tokens", s.saved_pct())
            ),
            ui::commas(s.events),
        ),
    };
    o.push('\n');
    let title = if cost.is_some() {
        "saved (input · measured)"
    } else {
        "saved (all time)"
    };
    o.push_str(&ui::panel(color, title, &[hero]));
    o.push('\n');

    // axes
    o.push_str(&axis(color, "input", s.input_before, s.input_after));
    o.push('\n');
    if s.output_events > 0 {
        // No live output baseline (the proxy never sees the un-instructed reply), so this is
        // the A/B benchmark factor, not a measurement — and it only holds where output is
        // actually shaped. Show the real billed volume + the benchmark bar, tagged as an
        // estimate so it's never read as measured per-row data.
        o.push_str(&format!(
            "  {:<7} {} {:>6}   {} billed   {}",
            "output",
            bar(color, BENCH_OUTPUT_REDUCTION * 100.0, 22),
            ui::paint(
                color,
                Tone::Accent,
                &format!("~{:+.0}%", -BENCH_OUTPUT_REDUCTION * 100.0)
            ),
            ui::human(s.output_after),
            ui::paint(color, Tone::Dim, "(est · if output shaped)"),
        ));
        o.push('\n');
    }
    if s.cache_read_tokens > 0 {
        // Prompt-cache reads bill at ~10% of input price → a flat 90% discount on the reused
        // prefix. The bar shows that discount; the count is the volume served from cache.
        o.push_str(&format!(
            "  {:<7} {} {:>6}   {} reused at ~10%\n",
            "cache",
            bar(color, 90.0, 22),
            ui::paint(color, Tone::Accent, "~-90%"),
            ui::human(s.cache_read_tokens),
        ));
    }

    // by-model table — MEASURED input-side saving per model (matches the honest headline):
    // input % saved and the input-side $ saved where the registry prices the model. No output
    // projection here, so a model that serves agent traffic isn't credited an unshaped-output win.
    if !models.is_empty() {
        o.push_str(&ui::paint(color, Tone::Dim, "\n by model\n"));
        let mut t = ui::table(color, &["model", "requests", "saved", "$ saved"]);
        for m in models {
            let pct = m.saved_pct;
            let pct_tone = if pct >= 0.0 { Tone::Accent } else { Tone::Warn };
            t.add_row(vec![
                comfy_table::Cell::new(ui::truncate(&m.name, 28)),
                ui::right(ui::commas(m.events)),
                ui::right(ui::paint(color, pct_tone, &format!("{:+.0}%", -pct))),
                ui::right(
                    m.cost_saved
                        .map(|c| ui::paint(color, Tone::Accent, &format!("${c:.2}")))
                        .unwrap_or_default(),
                ),
            ]);
        }
        for line in t.to_string().lines() {
            o.push_str(&format!(" {line}\n"));
        }
    }

    if let Some(us) = s.avg_compress_micros {
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            &format!(
                " added latency ~{:.2} ms/req · llmtrim compression overhead\n",
                us / 1000.0
            ),
        ));
    }
    if s.any_approximate {
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            " * some counts approximate (provider lacks a public tokenizer)\n",
        ));
    }
    if let Some(c) = cost {
        // Surface the output-side projection separately and clearly as an estimate that holds
        // ONLY when output is shaped — never folded into the measured headline above.
        let extra = c.projected_saved() - c.saved;
        if extra > 0.0 {
            o.push_str(&ui::paint(
                color,
                Tone::Dim,
                &format!(
                    " ~ + est. ${extra:.2} more if output is shaped (A/B bench −73%); not measured live, excluded from the headline.\n"
                ),
            ));
        }
    }
    o
}

// ── time-series report ──────────────────────────────────────────────────────────

/// A `--daily/--weekly/--monthly` table: one row per bucket with input/output savings.
pub fn period_report(color: bool, label: &str, rows: &[PeriodRow]) -> String {
    let mut o = format!(
        "{}\n",
        ui::paint(color, Tone::Bold, &format!("llmtrim — {label} savings"))
    );
    if rows.is_empty() {
        o.push_str(&ui::paint(color, Tone::Dim, " no activity recorded yet\n"));
        return o;
    }
    let mut t = ui::table(color, &["period", "requests", "input", "saved", "output"]);
    for r in rows {
        let in_pct = ui::saved_pct(r.input_before as f64, r.input_after as f64);
        let out = if r.output_before > 0 {
            format!(
                "{} ({:+.0}%)",
                ui::human(r.output_after),
                -ui::saved_pct(r.output_before as f64, r.output_after as f64)
            )
        } else if r.output_after > 0 {
            ui::human(r.output_after)
        } else {
            "—".to_string()
        };
        t.add_row(vec![
            comfy_table::Cell::new(&r.bucket),
            ui::right(ui::commas(r.events)),
            ui::right(format!(
                "{}→{}",
                ui::human(r.input_before),
                ui::human(r.input_after)
            )),
            ui::right(ui::paint(color, Tone::Accent, &format!("{:+.0}%", -in_pct))),
            ui::right(out),
        ]);
    }
    for line in t.to_string().lines() {
        o.push_str(&format!(" {line}\n"));
    }
    o
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
            out_spend: 3.0, // non-zero → an output estimate exists to surface separately
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
        // Headline shows the MEASURED input-side saving, not a projection that assumes shaping.
        assert!(
            out.contains("$12.47 saved"),
            "hero shows measured input saving"
        );
        assert!(out.contains("1,204 requests"), "request count");
        assert!(
            out.contains("input") && out.contains("2.1M → 1.2M"),
            "input axis"
        );
        assert!(out.contains("gpt-4o") && out.contains("$4.10"), "model row");
        // The output projection is surfaced separately and labeled as an estimate, never
        // baked into the headline dollar number.
        assert!(
            out.contains("if output is shaped") || out.contains("if output shaped"),
            "output projection clearly labeled as estimate"
        );
        assert!(out.contains("ms/req"), "added-latency line");
        assert!(!out.contains('\x1b'), "no ANSI when color=false");
    }

    #[test]
    fn headline_excludes_output_projection() {
        // The measured headline ($ saved + round-trip %) must equal the input-side cost,
        // NOT the larger projected figure — otherwise unshaped (agent) traffic is overstated.
        let cost = Cost {
            saved: 10.0,
            spend: 10.0,
            out_spend: 5.0,
        };
        let out = snapshot(false, None, &summ(), &[], Some(&cost));
        assert!(
            out.contains("$10.00 saved"),
            "headline = measured input saving"
        );
        assert!(
            !out.contains(&format!("${:.2} saved", cost.projected_saved())),
            "projected total ({:.2}) is not the headline",
            cost.projected_saved()
        );
        // Round-trip % is the measured input-side pct (50%), not the projected one.
        assert!(out.contains("-50%"), "measured round-trip pct in headline");
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
