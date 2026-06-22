//! Terminal rendering for the `monitor` command — savings snapshot, live dashboard,
//! time-series reports, and machine-readable export.
//!
//! Pure formatting: `main.rs` gathers the ledger + daemon state + pricing and feeds it
//! here, so this module stays decoupled from the interceptor feature and I/O. All
//! styling goes through `crate::ui`; colour is passed in by the caller, which disables
//! it for non-TTY stdout or when `NO_COLOR` is set.

use anyhow::Result;
use serde_json::json;

use crate::tracking::{ModelRow, Period, PeriodRow, Summary, Tracker};
use crate::ui::{self, Tone};

// ── view models (built by main from the ledger/daemon/pricing) ──────────────────

/// Interceptor daemon + wiring state for the header's health chain
/// (daemon alive → port accepting → env wired → traffic flowing).
pub struct DaemonView {
    pub running: bool,
    pub pid: u32,
    pub port: u16,
    pub uptime: String,
    pub uptime_secs: i64,
    pub ca_present: bool,
    /// TCP probe of the daemon's port — a live pidfile only proves the supervisor exists,
    /// not that the proxy is accepting. Also set when a proxy is found on the wired port
    /// with no pidfile to read (a `running: true`, `pid: 0` view).
    pub port_accepting: bool,
    /// Interceptor port wired into the persistent env (shell profile / registry), if any.
    pub env_port: Option<u16>,
    /// Run-at-login enabled.
    pub autostart: bool,
    /// Supervisor crash-restarts since the daemon started.
    pub restarts: u32,
    /// Version recorded in the daemon's pidfile (`None` = pre-field daemon).
    pub version: Option<String>,
    /// Version of the binary rendering this dashboard.
    pub binary_version: String,
    /// Daemon log path, shown when a check fails.
    pub log_path: Option<String>,
    /// Humanized age of the most recent ledger row ("4s ago"); `None` on an empty ledger.
    pub last_request: Option<String>,
}

/// Overall install health, derived from [`DaemonView`] — also the `status` exit code,
/// so scripts can gate on it (`llmtrim status -q && …`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// Running, port accepting, env wired to the daemon's port, CA present.
    Healthy,
    /// Something needs attention: running with a broken link (port dead, env missing or
    /// pointing elsewhere, CA missing), or stopped with the env still wired — the latter
    /// actively breaks LLM HTTPS on this machine.
    Degraded,
    /// Not running and not wired — a clean off state.
    Stopped,
}

impl Health {
    /// `systemctl is-active` convention: 0 healthy, non-zero otherwise; degraded gets its
    /// own code so scripts can tell "off" from "broken".
    pub fn exit_code(self) -> i32 {
        match self {
            Health::Healthy => 0,
            Health::Stopped => 1,
            Health::Degraded => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Health::Healthy => "healthy",
            Health::Stopped => "stopped",
            Health::Degraded => "degraded",
        }
    }
}

/// Derive overall health from the view (see [`Health`] for the rules).
pub fn health(d: &DaemonView) -> Health {
    if d.running {
        // `pid == 0` ⇒ detected only by a live port probe, no pidfile. The proxy appears
        // to be serving, but we can't confirm it's *ours* (any listener on the wired port
        // passes the probe), so this is "needs a look", not Healthy.
        if d.pid == 0 {
            Health::Degraded
        } else if d.port_accepting && d.env_port == Some(d.port) && d.ca_present {
            Health::Healthy
        } else {
            Health::Degraded
        }
    } else if d.env_port.is_some() {
        Health::Degraded
    } else {
        Health::Stopped
    }
}

/// Benchmark output-token reduction from llmtrim's A/B suite (see bench/README). The live
/// proxy can't measure a per-request output baseline (it never sees the un-instructed reply),
/// so the dashboard *projects* the output side from this measured benchmark factor — and only
/// onto traffic that actually carried the shaping instruction (`out_spend_shaped`), labeled
/// as an estimate. Projecting it onto unshaped (agent) traffic would overstate ~3.7×.
const BENCH_OUTPUT_REDUCTION: f64 = 0.73;

/// Projected $ saved: measured input saving + the benchmark-estimated output saving on the
/// *shaped* spend. Shaped output is ~(1−r) of the un-instructed baseline, so the saving on it
/// is `out_spend_shaped · r/(1−r)`. Unshaped spend gets no projection — its baseline IS the
/// billed amount.
fn projected_saved_usd(saved: f64, out_spend_shaped: f64) -> f64 {
    saved + out_spend_shaped * BENCH_OUTPUT_REDUCTION / (1.0 - BENCH_OUTPUT_REDUCTION)
}

/// Projected round-trip %: projected saving over the projected un-compressed bill.
fn projected_round_trip_pct(saved: f64, spend: f64, out_spend_shaped: f64) -> f64 {
    let projected = projected_saved_usd(saved, out_spend_shaped);
    let baseline = spend + projected;
    if baseline > 0.0 {
        projected / baseline * 100.0
    } else {
        0.0
    }
}

/// USD cost saved + the compressed spend, priced via the provider registry. `saved`/`spend`
/// value measured tokens at list rates; `net_saved` re-prices the saving against the real
/// (cache-discounted) bill; the `projected_*` helpers add the benchmark-estimated output
/// saving on the shaped share only.
#[derive(Clone, Copy)]
pub struct Cost {
    /// Input tokens cut × list input rate — the list-rate value of the cut. Used for the
    /// JSON/accounting outputs; the dashboard hero shows the measured `net_saved`, not this.
    pub saved: f64,
    /// Compressed bill at list rates: input_after + measured output.
    pub spend: f64,
    /// The REAL compressed bill actually paid — the cache-discounted input bill (fresh 1× +
    /// cache-write 1.25× + cache-read ~0.1×, per provider) plus measured output. Unlike
    /// `spend` (list), this is what the user really paid, so the receipt reconciles:
    /// `net_spend + net_saved` == the would-have-paid counterfactual.
    pub net_spend: f64,
    /// The output-token portion of `spend` ($).
    pub out_spend: f64,
    /// The same measured saving priced against the provider-reported usage split (cache
    /// reads ~10%, writes 125%, fresh 100%) — what actually came off the bill. Equals
    /// `saved` when the traffic uses no prompt cache.
    pub net_saved: f64,
    /// Output spend from requests that carried the shaping instruction — the only spend
    /// the A/B benchmark factor may be projected onto.
    pub out_spend_shaped: f64,
    /// The measured saving priced at the live-zone rate: cut tokens live in the compressible
    /// zone, which bills as fresh (1×) / cache-write (1.25× on Anthropic) — never at the ~10%
    /// cache-read rate the `net_saved` blend assumes, so this runs `≥ net_saved` (and can top
    /// `saved` when cache writes are involved). Kept for accounting; the dashboard no longer
    /// renders it (the hero is the conservative `net_saved`).
    pub live_saved: f64,
}

impl Cost {
    /// Measured round-trip cost saved as a percentage of the bill — input-side only, since
    /// output savings isn't measurable live (small; understates the real win). List-rate
    /// numerator over list-rate denominator: a consistent basis.
    fn pct(&self) -> f64 {
        let total = self.saved + self.spend;
        if total > 0.0 {
            self.saved / total * 100.0
        } else {
            0.0
        }
    }

    fn projected_saved(&self) -> f64 {
        projected_saved_usd(self.saved, self.out_spend_shaped)
    }

    fn projected_pct(&self) -> f64 {
        projected_round_trip_pct(self.saved, self.spend, self.out_spend_shaped)
    }
}

/// One per-model row in the breakdown table. `cost_saved`/`spend`/`out_spend` are the measured
/// USD figures (present when the registry prices the model), used to project the per-model
/// round-trip the same way the hero does. `cached` groups the table: a model whose traffic
/// used the provider prompt cache reads its `$ saved` as list value (the real bill is
/// cache-discounted), while un-cached traffic's `$ saved` comes straight off the bill.
pub struct ModelView {
    pub name: String,
    pub events: i64,
    pub saved_pct: f64,
    pub cost_saved: Option<f64>,
    pub spend: Option<f64>,
    pub out_spend: Option<f64>,
    pub cached: bool,
    /// Measured saving over this model's compressible (new-content) surface — the
    /// frozen-zone meter's per-model %. `None` until the model has metered rows; the
    /// table then falls back to `saved_pct` (the all-input figure).
    pub new_pct: Option<f64>,
}

/// One render-ready BY MODEL row for the breakdown TUI's native table. `is_header` rows are
/// the cached / un-cached group separators (their value columns are blank).
pub struct ModelLine {
    pub label: String,
    pub requests: String,
    pub saved: String,
    pub cost: String,
    pub is_header: bool,
}

/// Build the grouped BY MODEL rows for native rendering — same grouping as the text table:
/// "fresh context" models first, then "reused context" (cache-served), with the group-header
/// rows only when both kinds are present.
pub fn model_lines(models: &[ModelView]) -> Vec<ModelLine> {
    let grouped = models.iter().any(|m| m.cached) && models.iter().any(|m| !m.cached);
    let mut out = Vec::new();
    for (cached, header) in [(false, "fresh context"), (true, "reused context")] {
        let rows: Vec<&ModelView> = models.iter().filter(|m| m.cached == cached).collect();
        if rows.is_empty() {
            continue;
        }
        if grouped {
            out.push(ModelLine {
                label: header.to_string(),
                requests: String::new(),
                saved: String::new(),
                cost: String::new(),
                is_header: true,
            });
        }
        for m in rows {
            out.push(ModelLine {
                label: m.name.clone(),
                requests: ui::commas(m.events),
                saved: format!("{:.0}%", m.saved_pct),
                cost: m.cost_saved.map(|c| format!("${c:.2}")).unwrap_or_default(),
                is_header: false,
            });
        }
    }
    out
}

// ── rendering helpers ───────────────────────────────────────────────────────────

/// A `width`-cell gauge for a `saved` percentage (0–100, clamped): the saved portion is a
/// solid accent (blue) block — the win, growing left-to-right with the accent `-pct` label —
/// and the kept portion trails as dim dots (what you still pay). ≥100 → all filled, ≤0 → all dots.
fn bar(color: bool, saved: f64, width: usize) -> String {
    let cut = ((saved.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    let kept = width.saturating_sub(cut);
    format!(
        "{}{}",
        ui::paint(color, Tone::Accent, &"█".repeat(cut)),
        ui::paint(color, Tone::Dim, &"░".repeat(kept)),
    )
}

/// Cells reserved for the savings field (`before ─✂─▶ after`, or `N billed`) so the bars
/// line up across axes regardless of how wide each number prints. Sized for the widest
/// realistic value (`999.9M ─✂─▶ 999.9M`).
const SHEAR_FIELD_W: usize = 20;

/// Pad an already-styled cell to [`SHEAR_FIELD_W`] display cells (ANSI-aware).
fn pad_field(styled: &str) -> String {
    let pad = " ".repeat(SHEAR_FIELD_W.saturating_sub(ui::visible_width(styled)));
    format!("{styled}{pad}")
}

/// `before ─✂─▶ after`, padded to [`SHEAR_FIELD_W`] — the shear metaphor over a measured
/// before/after.
fn shear_field(color: bool, before: i64, after: i64) -> String {
    pad_field(&format!(
        "{} {} {}",
        ui::paint(color, Tone::Dim, &ui::human(before)),
        ui::paint(color, Tone::Dim, "─✂─▶"),
        ui::paint(color, Tone::Bold, &ui::human(after)),
    ))
}

/// A `name  before ─✂─▶ after  [bar]  ±pct` line — the shear metaphor over a measured
/// before/after. Percent is accent when saving, warn when it grew. Counts are token totals.
fn axis(color: bool, name: &str, before: i64, after: i64) -> String {
    let pct = ui::saved_pct(before as f64, after as f64);
    let pct_str = format!("{:>5}", format!("{:+.0}%", -pct)); // signed delta, -41% = saved 41%
    let pct_tone = if pct >= 0.0 { Tone::Accent } else { Tone::Warn };
    format!(
        "  {:<7} {}   {}   {}",
        name,
        shear_field(color, before, after),
        bar(color, pct, 18),
        ui::paint(color, pct_tone, &pct_str),
    )
}

// ── snapshot ────────────────────────────────────────────────────────────────────

/// The daemon header + health chain. Healthy links collapse to one calm dim line; every
/// broken link gets its own warn line naming the fix, because `status`'s first job is
/// answering "why is it broken?".
fn render_header(color: bool, d: &DaemonView) -> String {
    let mut o = String::new();
    if d.running {
        // `pid == 0` ⇒ detected by a live port probe with no pidfile to read from
        // (never recorded, or lost to a full disk). The proxy serves fine; it's just
        // unmanaged by our bookkeeping, so we can't show pid/uptime/version.
        let unmanaged = d.pid == 0;
        let mut meta = if unmanaged {
            format!(":{}", d.port)
        } else {
            format!(":{} · up {}", d.port, d.uptime)
        };
        if let Some(v) = &d.version {
            meta.push_str(&format!(" · v{v}"));
        }
        if d.autostart {
            meta.push_str(" · autostart on");
        }
        // One calm strip: wordmark, the live dot + meta, and the overall health word. The
        // per-check detail collapses into that word; broken links still get their own warn
        // lines below (and `llmtrim doctor` carries the full chain).
        let badge = match health(d) {
            Health::Healthy => ui::paint(color, Tone::Accent, "✓ healthy"),
            _ => ui::paint(color, Tone::Warn, "⚠ degraded"),
        };
        o.push_str(&format!(
            " {} {} {} {}   {}\n",
            ui::wordmark(color),
            ui::paint(color, Tone::Accent, "●"),
            ui::paint(color, Tone::Dim, "running"),
            ui::paint(color, Tone::Dim, &format!("· {meta}")),
            badge,
        ));

        // One warn line per broken link, each naming its fix.
        let log = d.log_path.as_deref().unwrap_or("~/.llmtrim/serve.log");
        if !d.port_accepting {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!(
                    "  {} port :{} not accepting connections — check log: {log}\n",
                    ui::WARN,
                    d.port
                ),
            ));
        }
        match d.env_port {
            Some(p) if p != d.port => o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!(
                    "  {} env points at :{p} but the daemon listens on :{} — run: llmtrim setup\n",
                    ui::WARN,
                    d.port
                ),
            )),
            None => o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!(
                    "  {} env not wired — traffic bypasses llmtrim; run: llmtrim setup\n",
                    ui::WARN
                ),
            )),
            _ => {}
        }
        if !d.ca_present {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!("  {} ca missing — run: llmtrim ca\n", ui::WARN),
            ));
        }
        if let Some(v) = &d.version
            && *v != d.binary_version
        {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!(
                    "  {} daemon is v{v}, binary is v{} — restart to update: llmtrim stop && llmtrim start\n",
                    ui::WARN, d.binary_version
                ),
            ));
        }
        if d.restarts > 0 {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!(
                    "  {} crashed and restarted {}× since start — check log: {log}\n",
                    ui::WARN,
                    d.restarts
                ),
            ));
        }
        if unmanaged {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!(
                    "  {} no pidfile — a proxy is answering on :{}, but llmtrim can't confirm it owns it (pidfile lost to a full disk, or a foreign listener); re-run: llmtrim setup\n",
                    ui::WARN,
                    d.port
                ),
            ));
        }
    } else if let Some(p) = d.env_port {
        // Stopped with HTTPS_PROXY still wired: every LLM HTTPS call on this machine fails
        // right now. Deliberately the loudest state in the dashboard.
        o.push_str(&format!(
            " {} {}\n",
            ui::paint(color, Tone::Warn, "llmtrim ○ stopped"),
            ui::paint(
                color,
                Tone::Warn,
                &format!("— env still points at :{p}; LLM calls will fail until it runs")
            ),
        ));
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            "  fix: llmtrim start   (or llmtrim uninstall to unwire the env)\n",
        ));
    } else {
        o.push_str(&format!(
            " {} {}\n",
            ui::paint(color, Tone::Dim, "llmtrim ○"),
            ui::paint(color, Tone::Dim, "stopped — start: llmtrim setup"),
        ));
        if !d.ca_present {
            o.push_str(&ui::paint(
                color,
                Tone::Warn,
                &format!("  {} ca missing — run: llmtrim ca\n", ui::WARN),
            ));
        }
    }
    o
}

/// The savings dashboard: daemon header + health chain, a hero panel (cost / round-trip /
/// requests), per-axis bars, and a per-model table. Returned as a string so the watch
/// loop can repaint it atomically. `today_saved_usd` is the priced saving for today (UTC),
/// shown next to the all-time hero when it is non-trivial.
#[allow(clippy::too_many_arguments)]
pub fn snapshot(
    color: bool,
    daemon: Option<&DaemonView>,
    s: &Summary,
    models: &[ModelView],
    cost: Option<&Cost>,
    today_saved_usd: Option<f64>,
    trend: &[i64],
    // When true, the input SAVINGS axis is measured over ALL input (the frozen cached prefix
    // included in the denominator) — the cache-diluted view. When false (default), it's
    // measured over the compressible surface only (cache excluded), the honest "what we can
    // touch" figure. The breakdown TUI flips this with the `c` key.
    cache_included: bool,
    // When true, a SOH (`\u{1}`) sentinel is emitted just before the BY MODEL section so the
    // breakdown TUI can split the dashboard into a left column (health/hero/savings/trend)
    // and a right column (the model table). Always false for the plain/pipe output.
    split_marker: bool,
) -> String {
    let mut o = String::new();

    if let Some(d) = daemon {
        o.push_str(&render_header(color, d));
    }

    if s.events == 0 {
        // Diagnose, don't shrug: the empty ledger looks identical for "never set up",
        // "set up but not wired", and "wired, just no traffic yet" — say which one it is.
        let hint = match daemon {
            Some(d) if d.running && d.env_port != Some(d.port) => {
                "\n proxy is up but nothing routes through it — see the env warning above; \
                 run `llmtrim setup`, then open a new terminal.\n"
            }
            Some(d) if d.running => {
                "\n wired and waiting for the first request — use your tools as normal.\n"
            }
            _ => "\n no activity yet — run `llmtrim setup`, then use your tools as normal.\n",
        };
        o.push_str(&ui::paint(color, Tone::Dim, hint));
        return o;
    }

    // Hero box — one dominant figure, supporting facts demoted to a second line. The
    // headline is the MEASURED input-side saving (real, per-row): every request's input is
    // compressed and re-tokenized, so it is honest for all traffic. The output side is NOT
    // in the headline — the proxy never sees the un-instructed reply, so any output saving
    // is a benchmark projection that only holds when output is actually shaped (projecting
    // it onto agent traffic would overstate ~2.7×); it is surfaced separately, below.
    o.push('\n');
    let (line1, line2) = match cost {
        Some(c) => {
            // Anchor the all-time number in the present: "what did it do for me today?"
            // Hidden when today has no priced saving — a perpetual $0.00 reads like fake data.
            let today = today_saved_usd
                .filter(|t| *t >= 0.005)
                .map(|t| {
                    format!(
                        " {} {}",
                        ui::paint(color, Tone::Dim, "·"),
                        // "saved today", not a bare "↑ $X" — an up-arrow next to a dollar on a
                        // billing screen reads as spend going up, the opposite of the truth.
                        ui::paint(color, Tone::Accent, &format!("${t:.2} saved today")),
                    )
                })
                .unwrap_or_default();
            // The headline is the MEASURED figure that came off the real, cache-discounted
            // bill — the conservative truth, so it owns "off your real bill". The list-rate
            // value and the higher live-zone estimate are upside, shown dim below as an
            // ascending ladder, never as the hero (a headline above its own list ceiling
            // reads as cooked).
            (
                format!(
                    "{} {}{}",
                    ui::hero(color, &format!("${:.2}", c.net_saved)),
                    ui::paint(color, Tone::Dim, "off your real bill"),
                    today,
                ),
                // No input % here: the SAVINGS axis below carries the honest figure (over
                // the compressible surface). Repeating the diluted all-input % would
                // contradict it.
                format!(
                    "{} tokens trimmed {} {} requests",
                    ui::human(s.saved()),
                    ui::paint(color, Tone::Dim, "·"),
                    ui::commas(s.events),
                ),
            )
        }
        None => (
            format!(
                "{} {}",
                ui::hero(color, &format!("-{:.0}%", s.saved_pct())),
                ui::paint(color, Tone::Dim, "input tokens trimmed"),
            ),
            format!("{} requests", ui::commas(s.events)),
        ),
    };
    o.push_str(&ui::boxed(
        color,
        &[line1, ui::paint(color, Tone::Dim, &line2)],
    ));
    // The logo's promise, free of brackets, right under the box.
    o.push_str(&format!("   {}\n", ui::ok(color, ui::TAGLINE)));
    // (No list-rate / live-zone "upside" line: the hero is the real, cache-discounted dollars
    // that came off the bill — the number the user actually cares about. Pricing the same cut
    // at list or cache-write rates is internal accounting, not a saving they can spend.)

    // savings axes, one gauge language for the whole section
    o.push_str(&ui::paint(color, Tone::Dim, "\n SAVINGS\n"));
    // Input axis — measured over the compressible surface only (input minus the frozen
    // cached prefix the stages skip by design), the honest % of what we're allowed to
    // touch. Metered rows only, so the figure is never diluted by legacy rows; ledgers
    // with no metered traffic yet fall back to the all-input axis.
    let new_before = s.metered_input_before - s.frozen_input_tokens;
    let new_after = s.metered_input_after - s.frozen_input_tokens;
    if cache_included {
        // Cache-included: trim measured against the whole prompt, frozen cached prefix and
        // all — the diluted denominator. Tagged so it can't be confused with the default.
        o.push_str(&axis(color, "input", s.input_before, s.input_after));
        o.push_str(&ui::paint(color, Tone::Dim, "   (all input · cache)"));
        o.push('\n');
    } else if s.frozen_input_tokens > 0 && new_before > 0 {
        o.push_str(&axis(color, "input", new_before, new_after));
        o.push_str(&ui::paint(color, Tone::Dim, "   (cache excluded)"));
        o.push('\n');
    } else {
        o.push_str(&axis(color, "input", s.input_before, s.input_after));
        o.push('\n');
    }
    if s.output_events > 0 {
        // No live output baseline (the proxy never sees the un-instructed reply), so we show
        // the real billed volume and the A/B-benchmark reduction as a clearly-tagged estimate
        // — never a fabricated before→after that would read as a measured trim. The `~%` and
        // "(est · if output shaped)" tag mark it; the bar aligns with the input axis.
        let billed = pad_field(&format!(
            "{} {}",
            ui::paint(color, Tone::Bold, &ui::human(s.output_after)),
            ui::paint(color, Tone::Dim, "billed"),
        ));
        o.push_str(&format!(
            "  {:<7} {}   {}   {}   {}\n",
            "output",
            billed,
            bar(color, BENCH_OUTPUT_REDUCTION * 100.0, 18),
            ui::paint(
                color,
                Tone::Accent,
                &format!(
                    "{:>5}",
                    format!("~{:+.0}%", -BENCH_OUTPUT_REDUCTION * 100.0)
                ),
            ),
            ui::paint(color, Tone::Dim, "(estimate)"),
        ));
    }
    // (No cache line: cache-safety is a property, not a per-run number. A token count here
    // credited llmtrim for the provider's own cache discount, and a static "we don't bust
    // your cache" reassurance is dashboard clutter — that belongs in the docs, not status.)

    // 7-day trend — a sparkline of daily tokens saved, oldest→newest. Sells the recurring
    // win the all-time hero can't. Hidden when there's no day-over-day data to plot.
    if trend.len() >= 2 {
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            "\n 7-DAY TREND  tokens saved / day\n",
        ));
        let peak = trend.iter().copied().max().unwrap_or(0);
        let last = *trend.last().unwrap_or(&0);
        // Direction compares the first and last bucket; the arrow points the way it moved and
        // is coloured so the signal reads at a glance (accent up, warn down).
        let (arrow, word, tone) = match (trend.first(), trend.last()) {
            (Some(a), Some(b)) if b > a => ("▲", "up", Tone::Accent),
            (Some(a), Some(b)) if b < a => ("▼", "down", Tone::Warn),
            _ => ("→", "flat", Tone::Dim),
        };
        o.push_str(&format!(
            "  {}   {} {} {}\n",
            ui::paint(color, Tone::Accent, &ui::sparkline(trend)),
            ui::paint(
                color,
                Tone::Dim,
                &format!("peak {} · last {} ·", ui::human(peak), ui::human(last)),
            ),
            ui::paint(color, tone, arrow),
            ui::paint(color, Tone::Dim, word),
        ));
    }

    // by-model table — MEASURED input-side saving per model (matches the honest headline):
    // input % saved and the input-side $ saved where the registry prices the model. No output
    // projection here, so a model that serves agent traffic isn't credited an unshaped-output win.
    // Grouped by cache usage when both kinds exist: un-cached $ comes straight off the bill,
    // cached $ is list value (the real bill is already cache-discounted) — the honesty split
    // is structural instead of an asterisk per row.
    if !models.is_empty() {
        if split_marker {
            o.push('\u{1}'); // split point: everything after goes to the TUI's right column
        }
        o.push_str(&ui::paint(color, Tone::Dim, "\n BY MODEL\n"));
        let mut t = ui::table(color, &["model", "requests", "saved", "$ saved"]);
        let grouped = models.iter().any(|m| m.cached) && models.iter().any(|m| !m.cached);
        let add_rows = |t: &mut comfy_table::Table, cached: bool| {
            if grouped {
                let label = if cached {
                    "reused context"
                } else {
                    "fresh context"
                };
                t.add_row(vec![comfy_table::Cell::new(ui::paint(
                    color,
                    Tone::Dim,
                    label,
                ))]);
            }
            // Biggest $ saved first within each group (unpriced rows last).
            let mut group: Vec<&ModelView> = models.iter().filter(|m| m.cached == cached).collect();
            group.sort_by(|a, b| {
                b.cost_saved
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&a.cost_saved.unwrap_or(f64::NEG_INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for m in group {
                // Metered models show the new-content % (the compressible surface, the
                // honest big number); pre-meter models fall back to the all-input figure.
                let pct = m.new_pct.unwrap_or(m.saved_pct);
                let pct_tone = if pct >= 0.0 { Tone::Accent } else { Tone::Warn };
                let name = ui::truncate(&m.name, 28);
                // `$0.00` reads as "did nothing"; a real but sub-cent saving shows `<$0.01`,
                // and an unpriced model shows a dim placeholder.
                let dollars = match m.cost_saved {
                    Some(c) if c >= 0.005 => ui::paint(color, Tone::Accent, &format!("${c:.2}")),
                    Some(_) => ui::paint(color, Tone::Dim, "<$0.01"),
                    None => ui::paint(color, Tone::Dim, "—"),
                };
                t.add_row(vec![
                    comfy_table::Cell::new(if grouped { format!("  {name}") } else { name }),
                    ui::right(ui::commas(m.events)),
                    ui::right(ui::paint(color, pct_tone, &format!("{:+.0}%", -pct))),
                    ui::right(dollars),
                ]);
            }
        };
        add_rows(&mut t, false);
        add_rows(&mut t, true);
        // Indent the boxed table one space, lining its left border up with the hero box, and
        // lighten the header rule (heavy ╞═══╡ → light ├───┤) so this secondary table doesn't
        // out-weight the hero. (BORDERS_ONLY uses ═ only for that rule, so the swap is safe.)
        for line in t.to_string().lines() {
            let light = line.replace('╞', "├").replace('╡', "┤").replace('═', "─");
            o.push_str(&format!(" {light}\n"));
        }
        if split_marker {
            o.push('\u{1}'); // close the model block; the TUI rejoins the text around it
        }
    }

    if let Some(us) = s.avg_compress_micros {
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            &format!(" + ~{:.0} ms/req · compression overhead\n", us / 1000.0),
        ));
    }
    if let Some(c) = cost {
        // Surface the output-side projection separately and clearly as an estimate — never
        // folded into the measured headline above. Projected ONLY onto the spend that
        // actually carried the shaping instruction; unshaped (agent) output is billed at
        // its own baseline, so there is nothing to project there.
        // Only when the projection is real money — a perpetual "$0.01 more" footnote costs
        // a line for no information (the output axis already carries the "est · if shaped"
        // caveat, covering the shaping-off case too).
        let extra = c.projected_saved() - c.saved;
        if extra >= 1.0 {
            o.push_str(&ui::paint(
                color,
                Tone::Dim,
                &format!(
                    " ~ + est. ${extra:.2} more saved by output shaping (A/B bench −73%); estimated, excluded from the headline.\n"
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
        "{}  {}\n",
        ui::wordmark(color),
        ui::paint(color, Tone::Dim, &format!("{label} savings")),
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
    daemon: Option<&DaemonView>,
) -> String {
    let v = json!({
        "daemon": daemon.map(|d| json!({
            "running": d.running,
            "health": health(d).label(),
            "pid": (d.running && d.pid != 0).then_some(d.pid),
            "port": d.running.then_some(d.port),
            "uptime_secs": (d.running && d.pid != 0).then_some(d.uptime_secs),
            "port_accepting": d.port_accepting,
            "env_port": d.env_port,
            "autostart": d.autostart,
            "restarts": d.restarts,
            "version": d.version,
            "binary_version": d.binary_version,
        })),
        "last_request_ts": s.last_ts,
        "requests": s.events,
        "input": { "before": s.input_before, "after": s.input_after, "saved_pct": s.saved_pct() },
        "output": { "before": s.output_before, "after": s.output_after,
                    "events": s.output_events, "saved_pct": s.output_saved_pct() },
        "cost": cost.map(|c| json!({ "saved_usd": c.saved, "spend_usd": c.spend, "round_trip_pct": c.pct(),
                                     "net_saved_usd": c.net_saved, "out_spend_usd": c.out_spend,
                                     "out_spend_shaped_usd": c.out_spend_shaped,
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

// ── ledger → view models + pricing (the single source of truth shared by the `status`
//    dashboard and the MCP `llmtrim_stats` tool) ─────────────────────────────────────

/// Full machine-readable stats snapshot from the ledger: the same JSON `status --json`
/// emits. The MCP server returns this verbatim so a tool call and the dashboard never
/// disagree. `daemon` is `None` for callers that don't inspect the proxy (the MCP tool),
/// which renders as `"daemon": null`.
pub fn stats_json(tracker: &Tracker, daemon: Option<&DaemonView>) -> Result<String> {
    let summary = tracker.summary()?;
    let models = model_views(tracker)?;
    let cost = monitor_cost(tracker);
    let periods = tracker.by_period(Period::Day)?;
    Ok(export_json(
        &summary,
        &models,
        cost.as_ref(),
        &periods,
        daemon,
    ))
}

pub fn monitor_cost(tracker: &Tracker) -> Option<Cost> {
    let models = tracker.by_model().ok()?;
    cost_estimate(&models)
}

/// Per-model rows for the breakdown, top 8 by request volume, priced where the registry
/// knows the model.
pub fn model_views(tracker: &Tracker) -> Result<Vec<ModelView>> {
    Ok(model_views_from(&tracker.by_model()?))
}

/// Same as [`model_views`] but from already-fetched rows, so a caller that also needs the
/// cost estimate can run `by_model()` once and feed both (the query is the heaviest per refresh).
pub fn model_views_from(rows: &[ModelRow]) -> Vec<ModelView> {
    let mut models: Vec<ModelView> = rows
        .iter()
        .filter(|m| m.events > 0)
        .map(|m| {
            // Per-model USD figures where the registry prices the model — used to project the
            // round-trip the same way the hero does.
            let priced = m.model.as_deref().and_then(llm_prices);
            // Frozen-zone meter, per model: the saving over the compressible surface
            // (metered rows only). `None` until traffic recorded the meter.
            let new_before = m.metered_input_before - m.frozen_input_tokens;
            let new_pct = (m.frozen_input_tokens > 0 && new_before > 0).then(|| {
                ui::saved_pct(
                    new_before as f64,
                    (m.metered_input_after - m.frozen_input_tokens) as f64,
                )
            });
            let cost_saved = priced
                .map(|(inp, _)| (m.input_before - m.input_after).max(0) as f64 / 1_000_000.0 * inp);
            let out_spend = priced.map(|(_, outp)| m.output_after as f64 / 1_000_000.0 * outp);
            let spend = priced.map(|(inp, outp)| {
                m.input_after as f64 / 1_000_000.0 * inp
                    + m.output_after as f64 / 1_000_000.0 * outp
            });
            ModelView {
                name: m
                    .model
                    .clone()
                    .unwrap_or_else(|| format!("{} · unknown model", m.provider)),
                events: m.events,
                saved_pct: ui::saved_pct(m.input_before as f64, m.input_after as f64),
                cost_saved,
                spend,
                out_spend,
                cached: m.cache_read > 0,
                new_pct,
            }
        })
        .collect();
    models.sort_unstable_by_key(|b| std::cmp::Reverse(b.events));
    models.truncate(8);
    models
}

/// Today's priced saving (UTC), for the hero's recency anchor. `None` when nothing
/// priced ran today — the dashboard hides the figure rather than showing $0.00.
pub fn today_saved_usd(tracker: &Tracker) -> Option<f64> {
    let models = tracker.by_model_today().ok()?;
    cost_estimate(&models).map(|c| c.saved)
}

/// Per-1M-token rates for one turn, frozen into the breakdown ledger so a historical session
/// always prices at what it actually cost. `cache_read`/`cache_write` apply the provider's
/// cache multipliers to the input rate (matching [`cost_estimate`]'s net-bill math).
/// Only the interceptor records turns, so this is gated with its sole caller (`serve`).
#[cfg(feature = "intercept")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BreakdownRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Resolve the frozen rates for a (provider, model) pair. Unknown models price at 0
/// (the TUI then shows a blank cost cell rather than a misleading $0.00).
#[cfg(feature = "intercept")]
pub(crate) fn rates_for(provider: &str, model: Option<&str>) -> BreakdownRates {
    let (input, output) = model.and_then(llm_prices).unwrap_or((0.0, 0.0));
    let (read_mult, write_mult) = cache_multipliers(provider);
    BreakdownRates {
        input,
        output,
        cache_read: input * read_mult,
        cache_write: input * write_mult,
    }
}

fn cache_multipliers(provider: &str) -> (f64, f64) {
    match provider {
        "anthropic" => (0.10, 1.25),
        "openai" => (0.50, 0.0),
        "google" => (0.25, 0.0),
        _ => (1.0, 0.0),
    }
}

/// USD cost figures priced per model via `llm_prices`, `None` when no recorded model
/// is priced.
///
/// - `saved`/`spend` value tokens at **list input rates** (tokens cut × input price; the
///   compressed input_after + measured output). Consistent numerator/denominator — the
///   headline basis.
/// - `net_saved` re-prices the same measured saving against the **real bill**: the
///   provider-reported usage split (fresh × 1.0 + cache writes × 1.25 + cache reads ×
///   0.1, per provider) gives the actual input bill, and the counterfactual bill scales
///   it by the measured compression ratio (same cache mix, prompt larger by `pct`):
///   `net_saved = net_bill × pct / (1 − pct)`. On traffic with no cache usage this
///   degrades exactly to the list-rate figure.
/// - `out_spend_shaped` is the output spend from requests that actually carried the
///   output-shaping instruction — the only spend the benchmark factor may be projected on.
pub fn cost_estimate(models: &[ModelRow]) -> Option<Cost> {
    let mut cost = Cost {
        saved: 0.0,
        spend: 0.0,
        net_spend: 0.0,
        out_spend: 0.0,
        net_saved: 0.0,
        out_spend_shaped: 0.0,
        live_saved: 0.0,
    };
    let mut matched = false;
    for m in models {
        let Some(model_id) = m.model.as_deref() else {
            continue;
        };
        if let Some((input_price, output_price)) = llm_prices(model_id) {
            let delta = (m.input_before - m.input_after).max(0) as f64;
            cost.saved += delta / 1_000_000.0 * input_price;
            let out = m.output_after as f64 / 1_000_000.0 * output_price;
            cost.spend += m.input_after as f64 / 1_000_000.0 * input_price + out;
            cost.out_spend += out;
            cost.out_spend_shaped += m.output_after_shaped as f64 / 1_000_000.0 * output_price;

            let (read_mult, write_mult) = cache_multipliers(&m.provider);
            let net_bill = (m.fresh_input_est as f64
                + m.cache_write as f64 * write_mult
                + m.cache_read as f64 * read_mult)
                / 1_000_000.0
                * input_price;
            // What was really paid for this model: cache-discounted input + measured output.
            cost.net_spend += net_bill + out;
            // The .min(0.95) clamp is load-bearing: it bounds the `1 - pct` denominator below
            // away from zero (>= 0.05), so a 100%-compressed model can't produce Infinity here.
            let pct = if m.input_before > 0 {
                (delta / m.input_before as f64).min(0.95)
            } else {
                0.0
            };
            cost.net_saved += net_bill * pct / (1.0 - pct);
            // Live-zone attribution: the cut happens in the compressible zone, billed as
            // fresh (1×) + cache writes (1.25× on Anthropic) — never at the ~10% read rate
            // the net blend assumes — so price the cut at that mix. No usage split recorded
            // → rate 1.0, degrading to the list figure.
            let live_used = m.fresh_input_est + m.cache_write;
            let live_rate = if live_used > 0 {
                (m.fresh_input_est as f64 + m.cache_write as f64 * write_mult.max(1.0))
                    / live_used as f64
            } else {
                1.0
            };
            cost.live_saved += delta / 1_000_000.0 * input_price * live_rate;
            matched = true;
        }
    }
    matched.then_some(cost)
}

/// Assemble the breakdown TUI's Overview view-model from the ledger. The `status` closure builds
/// the health line from the single summary scan plus whether there's traffic, so the binary can
/// supply the live daemon-derived status and the SVG exporter a forced-healthy one — without
/// either re-deriving the (subtle) numeric fields. The ledger is scanned once for the summary
/// and once for `by_model`. `sessions` is filled later by the caller's single `sessions()` scan.
#[cfg(feature = "breakdown")]
pub fn overview_data(
    tracker: &Tracker,
    status: impl FnOnce(&crate::tracking::Summary, bool) -> crate::breakdown::app::StatusLine,
) -> crate::breakdown::app::OverviewData {
    use crate::breakdown::app::OverviewData;
    use crate::tracking::Period;

    let summary = tracker.summary().unwrap_or_default();
    let has_traffic = summary.events > 0;
    let model_rows = tracker.by_model().unwrap_or_default();
    let models = model_views_from(&model_rows);
    let cost = cost_estimate(&model_rows);
    let status = status(&summary, has_traffic);

    // One honest basis: cache-discounted bill and the real saving off it, so
    // would_have − paid == saved and the savers reconcile to it.
    let paid_usd = cost.as_ref().map(|c| c.net_spend);
    let saved_usd = cost.as_ref().map(|c| c.net_saved);
    let would_have_usd = match (paid_usd, saved_usd) {
        (Some(p), Some(s)) => Some(p + s),
        _ => None,
    };

    // Savings fraction over the compressible (non-frozen) surface — the honest basis.
    let new_before = summary.metered_input_before - summary.frozen_input_tokens;
    let new_after = summary.metered_input_after - summary.frozen_input_tokens;
    let pct_less = if summary.frozen_input_tokens > 0 && new_before > 0 {
        (new_before - new_after).max(0) as f64 / new_before as f64
    } else if summary.input_before > 0 {
        summary.saved() as f64 / summary.input_before as f64
    } else {
        0.0
    };

    // Daily $ trend: scale the daily token-savings series by the blended $/token rate.
    let blend = match cost.as_ref() {
        Some(c) if summary.saved() > 0 => c.net_saved / summary.saved() as f64,
        _ => 0.0,
    };
    let trend_daily_usd: Vec<f64> = tracker
        .by_period(Period::Day)
        .unwrap_or_default()
        .iter()
        .rev()
        .take(7)
        .rev()
        .map(|r| (r.input_before - r.input_after).max(0) as f64 * blend)
        .collect();

    // Biggest savers: priced models by $ saved, scaled to the net-of-cache basis.
    let net_ratio = cost
        .as_ref()
        .map(|c| {
            if c.saved > 0.0 {
                c.net_saved / c.saved
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);
    let mut savers: Vec<(String, f64)> = models
        .iter()
        .filter_map(|m| m.cost_saved.map(|s| (m.name.clone(), s * net_ratio)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    savers.sort_by(|a, b| b.1.total_cmp(&a.1));
    savers.truncate(8);

    OverviewData {
        status,
        paid_usd,
        would_have_usd,
        saved_usd,
        saved_today_usd: today_saved_usd(tracker).filter(|t| *t >= 0.005),
        pct_less,
        added_ms: summary.avg_compress_micros.map(|us| us / 1000.0),
        requests: summary.events,
        trend_daily_usd,
        savers,
        input_before: summary.input_before,
        input_after: summary.input_after,
        output_billed: summary.output_after,
        // The A/B-benchmark output-reduction factor; shown only in the expert strip, labeled "(est)".
        output_est_pct: BENCH_OUTPUT_REDUCTION * 100.0,
        has_traffic,
        approximate: summary.any_approximate,
        // Filled by App::reload from the single sessions() scan it already runs.
        sessions: 0,
        update_available: crate::update::check(false),
    }
}

/// Per-1M-token `(input, output)` price for a model: the `llm_providers` registry first,
/// matched across every provider (the ledger records the wire-shape provider, not the
/// upstream brand), then the embedded models.dev snapshot for models the registry hasn't
/// shipped yet (e.g. day-one releases like claude-fable-5 on 0.14.3).
fn llm_prices(model_id: &str) -> Option<(f64, f64)> {
    #[cfg(feature = "intercept")]
    for &provider_id in llm_providers::get_providers_data().keys() {
        if let Some(model) = llm_providers::get_model_ref(provider_id, model_id) {
            return Some((model.input_price, model.output_price));
        }
    }
    snapshot_prices(model_id)
}

/// Fallback table: the pinned models.dev snapshot the bench prices from, embedded at
/// compile time (the package re-includes bench/pricing.json for this). Exact id first,
/// then with the `provider/` prefix stripped, mirroring [`crate::bench::resolve_pricing`]
/// — but zero-priced rows (free tiers, parse gaps) return `None` so an unknown model
/// shows a blank cost cell rather than a misleading $0.00.
fn snapshot_prices(model_id: &str) -> Option<(f64, f64)> {
    use crate::bench;
    static TABLE: once_cell::sync::Lazy<bench::PriceTable> =
        once_cell::sync::Lazy::new(|| bench::load_pricing(include_str!("../bench/pricing.json")));
    let p = TABLE.get(model_id).or_else(|| {
        let (_, bare) = model_id.split_once('/')?;
        TABLE.get(bare)
    })?;
    let (input, output) = (p.input_per_1k * 1000.0, p.output_per_1k * 1000.0);
    (input > 0.0 || output > 0.0).then_some((input, output))
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
            last_ts: Some("2026-06-10T12:00:00+00:00".into()),
            frozen_input_tokens: 0,
            metered_input_before: 0,
            metered_input_after: 0,
        }
    }

    fn priced() -> Cost {
        Cost {
            saved: 10.0,
            spend: 10.0,
            net_spend: 0.0,
            out_spend: 0.0,
            net_saved: 10.0,
            live_saved: 10.0,
            out_spend_shaped: 0.0,
        }
    }

    #[test]
    fn model_lines_group_cached_and_uncached() {
        let mv = |name: &str, cached: bool| ModelView {
            name: name.into(),
            events: 10,
            saved_pct: 50.0,
            cost_saved: Some(1.0),
            spend: None,
            out_spend: None,
            cached,
            new_pct: None,
        };
        // Both kinds present → a header row per group, un-cached first.
        let lines = model_lines(&[mv("a", false), mv("b", true)]);
        assert!(
            lines
                .iter()
                .any(|l| l.is_header && l.label == "fresh context")
        );
        assert!(
            lines
                .iter()
                .any(|l| l.is_header && l.label == "reused context")
        );
        assert_eq!(lines.iter().filter(|l| !l.is_header).count(), 2);
        // One kind only → no group headers.
        assert!(model_lines(&[mv("a", false)]).iter().all(|l| !l.is_header));
    }

    #[test]
    fn cache_toggle_switches_input_axis_basis() {
        // 1M input, 900k of it a frozen cached prefix; 67k trimmed off the 100k compressible
        // surface. Cache-excluded reads 67% (of what we can touch); cache-included dilutes it
        // to 7% (of the whole prompt).
        let mut s = summ();
        s.input_before = 1_000_000;
        s.input_after = 933_000;
        s.frozen_input_tokens = 900_000;
        s.metered_input_before = 1_000_000;
        s.metered_input_after = 933_000;
        let c = priced();
        let excluded = snapshot(false, None, &s, &[], Some(&c), None, &[], false, false);
        assert!(excluded.contains("(cache excluded)"));
        assert!(excluded.contains("-67%"), "cache-excluded axis: {excluded}");
        let included = snapshot(false, None, &s, &[], Some(&c), None, &[], true, false);
        assert!(included.contains("(all input · cache)"));
        assert!(included.contains("-7%"), "cache-included axis: {included}");
    }

    #[test]
    fn trend_section_renders_sparkline_and_direction() {
        let c = priced();
        // Rising series → ▲ up, sparkline scaled to the series max (0 → ▁, max → █).
        let up = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&c),
            None,
            &[0, 20, 80],
            false,
            false,
        );
        assert!(up.contains("7-DAY TREND"));
        assert!(
            up.contains('▁') && up.contains('█'),
            "sparkline scaled: {up}"
        );
        assert!(up.contains("peak 80") && up.contains("last 80"));
        assert!(up.contains('▲') && up.contains("up"), "rising → up: {up}");
        // Falling series → ▼ down.
        let down = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&c),
            None,
            &[80, 10],
            false,
            false,
        );
        assert!(
            down.contains('▼') && down.contains("down"),
            "falling → down: {down}"
        );
        // Fewer than two buckets → the whole section is hidden.
        let one = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&c),
            None,
            &[5],
            false,
            false,
        );
        assert!(
            !one.contains("7-DAY TREND"),
            "single bucket hides trend: {one}"
        );
    }

    #[test]
    fn by_model_unpriced_row_shows_dash_placeholder() {
        let models = vec![ModelView {
            name: "mystery-model".into(),
            events: 7,
            saved_pct: 20.0,
            cost_saved: None, // registry doesn't price it → dim — placeholder, not blank
            spend: None,
            out_spend: None,
            cached: false,
            new_pct: None,
        }];
        let out = snapshot(
            false,
            None,
            &summ(),
            &models,
            Some(&priced()),
            None,
            &[],
            false,
            false,
        );
        assert!(
            out.contains("mystery-model") && out.contains('—'),
            "unpriced $ saved shows a dash: {out}"
        );
    }

    #[test]
    fn bar_clamps_and_fills() {
        assert_eq!(bar(false, 50.0, 10), "█████░░░░░"); // 50% saved: 5 filled (accent), 5 dots
        assert_eq!(bar(false, 0.0, 4), "░░░░"); // nothing saved → all dots
        assert_eq!(bar(false, 100.0, 4), "████"); // all saved → all filled (accent)
        assert_eq!(bar(false, 150.0, 4), "████"); // clamp high → all filled
        assert_eq!(bar(false, -20.0, 4), "░░░░"); // clamp low → all dots
    }

    #[test]
    fn snapshot_plain_has_hero_and_axes() {
        let cost = Cost {
            saved: 12.47,
            spend: 9.0,
            net_spend: 0.0,
            out_spend: 3.0,
            net_saved: 12.47, // no cache discount → net line hidden
            live_saved: 12.47,
            out_spend_shaped: 3.0, // shaped → an output estimate exists to surface separately
        };
        let models = vec![ModelView {
            name: "gpt-4o".into(),
            events: 420,
            saved_pct: 61.0,
            cost_saved: Some(4.10),
            spend: Some(6.0),
            out_spend: Some(0.0),
            cached: false,
            new_pct: None,
        }];
        let out = snapshot(
            false,
            None,
            &summ(),
            &models,
            Some(&cost),
            None,
            &[],
            false,
            false,
        );
        // Headline shows the MEASURED figures (tokens trimmed + real-bill $), not a
        // projection that assumes shaping.
        assert!(
            out.contains("tokens trimmed") && out.contains("$12.47 off your real bill"),
            "hero shows measured trimmed tokens + real-bill saving"
        );
        assert!(out.contains("1,204 requests"), "request count");
        assert!(
            out.contains("✓ same answers, smaller bill"),
            "logo value promise stamped under the hero"
        );
        assert!(
            out.contains("input") && out.contains("2.1M ─✂─▶ 1.2M"),
            "input axis"
        );
        assert!(out.contains("gpt-4o") && out.contains("$4.10"), "model row");
        // The output projection is surfaced separately and labeled as an estimate, never
        // baked into the headline dollar number.
        assert!(
            out.contains("(estimate)"),
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
            net_spend: 0.0,
            out_spend: 5.0,
            net_saved: 10.0,
            live_saved: 10.0,
            out_spend_shaped: 5.0,
        };
        let out = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&cost),
            None,
            &[],
            false,
            false,
        );
        assert!(
            out.contains("$10.00 off your real bill"),
            "headline = measured net saving"
        );
        assert!(
            !out.contains(&format!(
                "${:.2} off your real bill",
                cost.projected_saved()
            )),
            "projected total ({:.2}) is not the headline",
            cost.projected_saved()
        );
    }

    #[test]
    fn projects_output_saving_from_benchmark() {
        // out baseline = 0.27 / (1 − 0.73) = 1.0, so projected output saved = 1.0 − 0.27 = 0.73.
        let c = Cost {
            saved: 1.0,
            spend: 1.0,
            net_spend: 0.0,
            out_spend: 0.27,
            net_saved: 1.0,
            live_saved: 1.0,
            out_spend_shaped: 0.27, // all of it carried the instruction
        };
        assert!((c.projected_saved() - 1.73).abs() < 1e-9);
        assert!((c.projected_pct() - 1.73 / 2.73 * 100.0).abs() < 1e-9);
    }

    #[test]
    fn unshaped_output_gets_no_projection() {
        // Agent traffic: output billed without the shaping instruction. Its baseline IS the
        // billed amount, so projecting the bench factor onto it would overstate ~3.7× —
        // the projection must be zero and the footnote must say shaping is off, not "+ $".
        let c = Cost {
            saved: 10.0,
            spend: 10.0,
            net_spend: 0.0,
            out_spend: 5.0,
            net_saved: 10.0,
            live_saved: 10.0,
            out_spend_shaped: 0.0,
        };
        assert!((c.projected_saved() - c.saved).abs() < 1e-9);
        let out = snapshot(false, None, &summ(), &[], Some(&c), None, &[], false, false);
        // No footnote either way: the output axis' "(est · if output shaped)" tag carries
        // the caveat; a dedicated line only appears when the projection is ≥ $1.
        assert!(!out.contains("more saved by output shaping"));
    }

    #[test]
    fn hero_is_the_measured_real_bill_only() {
        // Cache-heavy traffic: cut tokens are worth $100 at list rates but $25 came off the
        // real (cache-discounted) bill. The hero is the measured $25 — the dollars the user
        // can actually spend — and the list/live-zone figures are deliberately NOT shown.
        let c = Cost {
            saved: 100.0,
            spend: 50.0,
            net_spend: 0.0,
            out_spend: 5.0,
            net_saved: 25.0,
            live_saved: 130.0,
            out_spend_shaped: 0.0,
        };
        let out = snapshot(false, None, &summ(), &[], Some(&c), None, &[], false, false);
        assert!(
            out.contains("$25.00 off your real bill"),
            "hero is the measured real-bill figure: {out}"
        );
        assert!(
            !out.contains("at list") && !out.contains("$100.00") && !out.contains("$130.00"),
            "no list-rate / live-zone upside line: {out}"
        );
    }

    #[test]
    fn new_content_axis_and_measured_hero() {
        // Metered traffic: 1.0M of the 1.5M metered prompt is frozen prefix, so the
        // compressible surface went 500K → 100K (−80%) — the axis the meter unlocks.
        let mut s = summ();
        s.frozen_input_tokens = 1_000_000;
        s.metered_input_before = 1_500_000;
        s.metered_input_after = 1_100_000;
        let c = Cost {
            saved: 100.0,
            spend: 50.0,
            net_spend: 0.0,
            out_spend: 5.0,
            net_saved: 25.0,
            out_spend_shaped: 0.0,
            live_saved: 130.0,
        };
        let out = snapshot(false, None, &s, &[], Some(&c), None, &[], false, false);
        assert!(
            out.contains("$25.00 off your real bill"),
            "hero is the measured real-bill figure: {out}"
        );
        assert!(
            !out.contains("at list") && !out.contains("$130.00"),
            "no list-rate / live-zone upside line: {out}"
        );
        assert!(
            out.contains("500.0K ─✂─▶ 100.0K") && out.contains("(cache excluded)"),
            "input axis over the compressible surface: {out}"
        );
        assert!(
            !out.contains("2.1M ─✂─▶ 1.2M"),
            "diluted all-input axis replaced, not shown alongside: {out}"
        );

        // No metered rows → fall back to the all-input axis (pre-meter ledgers unchanged).
        let out = snapshot(false, None, &summ(), &[], Some(&c), None, &[], false, false);
        assert!(!out.contains("cache excluded"));
        assert!(out.contains("2.1M ─✂─▶ 1.2M"), "fallback axis: {out}");
    }

    #[test]
    fn snapshot_empty_ledger_guides_user() {
        let out = snapshot(
            false,
            None,
            &Summary::default(),
            &[],
            None,
            None,
            &[],
            false,
            false,
        );
        assert!(out.contains("no activity yet"));
    }

    #[test]
    fn export_json_roundtrips() {
        let out = export_json(&summ(), &[], None, &[], None);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["requests"], 1204);
        assert_eq!(v["cost"], serde_json::Value::Null);
    }

    // stats_json is the single source the `status --json` dashboard and the MCP llmtrim_stats
    // tool both read; assert it assembles real ledger rows into the export shape.
    #[test]
    fn stats_json_reads_the_ledger() {
        use crate::tracking::Record;
        let tracker = Tracker::open_in_memory().unwrap();
        for _ in 0..2 {
            tracker
                .record(&Record {
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
                })
                .unwrap();
        }
        let out = stats_json(&tracker, None).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["requests"], 2);
        assert_eq!(v["daemon"], serde_json::Value::Null);
        assert!(v["by_model"].as_array().is_some_and(|m| !m.is_empty()));
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

    /// A fully healthy view; tests break one link at a time.
    fn dv() -> DaemonView {
        DaemonView {
            running: true,
            pid: 4242,
            port: 8788,
            uptime: "33m45s".into(),
            uptime_secs: 2025,
            ca_present: true,
            port_accepting: true,
            env_port: Some(8788),
            autostart: true,
            restarts: 0,
            version: Some("0.1.0".into()),
            binary_version: "0.1.0".into(),
            log_path: Some("/home/u/.llmtrim/serve.log".into()),
            last_request: Some("4s ago".into()),
        }
    }

    #[test]
    fn health_matrix() {
        assert_eq!(health(&dv()), Health::Healthy);
        // Running with any broken link → degraded.
        for broken in [
            DaemonView {
                port_accepting: false,
                ..dv()
            },
            DaemonView {
                env_port: None,
                ..dv()
            },
            DaemonView {
                env_port: Some(8787),
                ..dv()
            },
            DaemonView {
                ca_present: false,
                ..dv()
            },
        ] {
            assert_eq!(health(&broken), Health::Degraded);
        }
        // Stopped + env still wired = active breakage, not a clean off state.
        let stopped_wired = DaemonView {
            running: false,
            ..dv()
        };
        assert_eq!(health(&stopped_wired), Health::Degraded);
        let stopped_clean = DaemonView {
            running: false,
            env_port: None,
            ..dv()
        };
        assert_eq!(health(&stopped_clean), Health::Stopped);
        // Exit codes follow the systemctl convention.
        assert_eq!(Health::Healthy.exit_code(), 0);
        assert_eq!(Health::Stopped.exit_code(), 1);
        assert_eq!(Health::Degraded.exit_code(), 2);
    }

    /// A live proxy on the wired port with no pidfile (`pid: 0`) must read as running (not
    /// the false "stopped — LLM calls will fail" a missing pidfile used to trigger), but
    /// only Degraded — we can't confirm the listener is ours, so it's not a clean Healthy.
    #[test]
    fn unmanaged_live_proxy_is_running_but_degraded() {
        let unmanaged = DaemonView {
            pid: 0,
            uptime: String::new(),
            uptime_secs: 0,
            version: None,
            ..dv()
        };
        assert_eq!(health(&unmanaged), Health::Degraded);

        let out = render_header(false, &unmanaged);
        assert!(out.contains("running"), "should report running: {out}");
        assert!(!out.contains("stopped"), "must not say stopped: {out}");
        assert!(
            !out.contains("pid 0"),
            "must not print the bogus pid 0: {out}"
        );
        assert!(out.contains("no pidfile"), "should explain the gap: {out}");
        assert!(
            out.contains("degraded"),
            "unmanaged proxy reads as degraded, not healthy: {out}"
        );
    }

    #[test]
    fn header_healthy_collapses_to_one_calm_line() {
        let out = render_header(false, &dv());
        assert!(out.contains("running") && out.contains("✓ healthy"));
        // The per-check chain collapses into the health word — no detail line, no warnings.
        assert!(!out.contains("✓ port") && !out.contains("last request"));
        assert!(out.contains(":8788") && out.contains("autostart on") && out.contains("v0.1.0"));
        assert!(!out.contains('⚠'), "healthy header has no warnings");
    }

    #[test]
    fn header_warns_per_broken_link() {
        let out = render_header(
            false,
            &DaemonView {
                port_accepting: false,
                ..dv()
            },
        );
        assert!(out.contains("not accepting connections"));
        assert!(out.contains("serve.log"), "points at the log");

        let out = render_header(
            false,
            &DaemonView {
                env_port: None,
                ..dv()
            },
        );
        assert!(out.contains("env not wired"));
        assert!(out.contains("llmtrim setup"));

        let out = render_header(
            false,
            &DaemonView {
                env_port: Some(8787),
                ..dv()
            },
        );
        assert!(
            out.contains(":8787") && out.contains(":8788"),
            "names both ports"
        );
    }

    #[test]
    fn header_stopped_but_wired_is_loud() {
        let out = render_header(
            false,
            &DaemonView {
                running: false,
                ..dv()
            },
        );
        assert!(out.contains("LLM calls will fail"));
        assert!(out.contains("llmtrim start"));
        // Clean stop stays calm.
        let out = render_header(
            false,
            &DaemonView {
                running: false,
                env_port: None,
                ..dv()
            },
        );
        assert!(out.contains("stopped — start: llmtrim setup"));
        assert!(!out.contains("will fail"));
    }

    #[test]
    fn header_flags_version_skew_and_restarts() {
        let out = render_header(
            false,
            &DaemonView {
                version: Some("0.0.9".into()),
                ..dv()
            },
        );
        assert!(out.contains("daemon is v0.0.9, binary is v0.1.0"));
        assert!(out.contains("llmtrim stop && llmtrim start"));

        let out = render_header(
            false,
            &DaemonView {
                restarts: 3,
                ..dv()
            },
        );
        assert!(out.contains("restarted 3×"));

        // A pre-version pidfile must not be flagged as skew (nothing to compare).
        let out = render_header(
            false,
            &DaemonView {
                version: None,
                ..dv()
            },
        );
        assert!(!out.contains("restart to update"));
    }

    #[test]
    fn empty_ledger_diagnoses_why() {
        // Running but unwired → say traffic bypasses, not a generic "run setup".
        let unwired = DaemonView {
            env_port: None,
            ..dv()
        };
        let out = snapshot(
            false,
            Some(&unwired),
            &Summary::default(),
            &[],
            None,
            None,
            &[],
            false,
            false,
        );
        assert!(out.contains("nothing routes through it"));

        // Running and wired → it's just waiting; don't tell the user to re-setup.
        let out = snapshot(
            false,
            Some(&dv()),
            &Summary::default(),
            &[],
            None,
            None,
            &[],
            false,
            false,
        );
        assert!(out.contains("waiting for the first request"));

        // Not installed at all → the original guidance.
        let off = DaemonView {
            running: false,
            env_port: None,
            ..dv()
        };
        let out = snapshot(
            false,
            Some(&off),
            &Summary::default(),
            &[],
            None,
            None,
            &[],
            false,
            false,
        );
        assert!(out.contains("no activity yet"));
    }

    #[test]
    fn hero_shows_today_when_priced() {
        let cost = Cost {
            saved: 100.0,
            spend: 50.0,
            net_spend: 0.0,
            out_spend: 0.0,
            net_saved: 100.0,
            live_saved: 100.0,
            out_spend_shaped: 0.0,
        };
        let out = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&cost),
            Some(1.84),
            &[],
            false,
            false,
        );
        assert!(out.contains("$1.84 saved today"));
        // A ~zero today figure is hidden — an idle proxy must not print a "today" delta.
        let out = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&cost),
            Some(0.0),
            &[],
            false,
            false,
        );
        assert!(!out.contains("saved today"));
        let out = snapshot(
            false,
            None,
            &summ(),
            &[],
            Some(&cost),
            None,
            &[],
            false,
            false,
        );
        assert!(!out.contains("saved today"));
    }

    #[test]
    fn export_json_carries_daemon_health() {
        let out = export_json(&summ(), &[], None, &[], Some(&dv()));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["daemon"]["running"], true);
        assert_eq!(v["daemon"]["health"], "healthy");
        assert_eq!(v["daemon"]["port"], 8788);
        assert_eq!(v["daemon"]["env_port"], 8788);
        assert_eq!(v["daemon"]["autostart"], true);
        assert_eq!(v["daemon"]["restarts"], 0);
        assert_eq!(v["last_request_ts"], "2026-06-10T12:00:00+00:00");

        // Stopped: pid/port/uptime are null, health says so.
        let stopped = DaemonView {
            running: false,
            env_port: None,
            ..dv()
        };
        let out = export_json(&summ(), &[], None, &[], Some(&stopped));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["daemon"]["running"], false);
        assert_eq!(v["daemon"]["health"], "stopped");
        assert_eq!(v["daemon"]["pid"], serde_json::Value::Null);
    }
}
