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
    /// TCP probe of the daemon's port (always `false` when not running) — a live pidfile
    /// only proves the supervisor exists, not that the proxy is accepting.
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
        if d.port_accepting && d.env_port == Some(d.port) && d.ca_present {
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
    /// Input tokens cut × list input rate — the headline (what those tokens cost at list).
    pub saved: f64,
    /// Compressed bill at list rates: input_after + measured output.
    pub spend: f64,
    /// The output-token portion of `spend` ($).
    pub out_spend: f64,
    /// The same measured saving priced against the provider-reported usage split (cache
    /// reads ~10%, writes 125%, fresh 100%) — what actually came off the bill. Equals
    /// `saved` when the traffic uses no prompt cache.
    pub net_saved: f64,
    /// Output spend from requests that carried the shaping instruction — the only spend
    /// the A/B benchmark factor may be projected onto.
    pub out_spend_shaped: f64,
    /// The measured saving priced at the live-zone rate: cut tokens live in the
    /// compressible zone, which bills as fresh (1×) / cache-write (1.25×) — never at the
    /// ~10% cache-read rate the `net_saved` blend assumes. The honest mid of the
    /// `net_saved → saved` range; equals `saved` when the traffic reports no usage split.
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

/// The daemon header + health chain. Healthy links collapse to one calm dim line; every
/// broken link gets its own warn line naming the fix, because `status`'s first job is
/// answering "why is it broken?".
fn render_header(color: bool, d: &DaemonView) -> String {
    let mut o = String::new();
    if d.running {
        let mut meta = format!("pid {} · :{} · up {}", d.pid, d.port, d.uptime);
        if let Some(v) = &d.version {
            meta.push_str(&format!(" · v{v}"));
        }
        if d.autostart {
            meta.push_str(" · autostart on");
        }
        o.push_str(&format!(
            " {} {}  {}\n",
            ui::paint(color, Tone::Accent, "llmtrim ●"),
            ui::paint(color, Tone::Dim, "running"),
            ui::paint(color, Tone::Dim, &meta),
        ));

        // The chain, healthy links first…
        let env_ok = d.env_port == Some(d.port);
        let mut ok_bits: Vec<String> = Vec::new();
        if d.port_accepting {
            ok_bits.push(format!("{} port", ui::OK));
        }
        if env_ok {
            ok_bits.push(format!("{} env :{}", ui::OK, d.port));
        }
        if d.ca_present {
            ok_bits.push(format!("{} ca", ui::OK));
        }
        ok_bits.push(match &d.last_request {
            Some(age) => format!("last request {age}"),
            None => "no requests yet".to_string(),
        });
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            &format!("  {}\n", ok_bits.join("   ")),
        ));

        // …then one warn line per broken link, each naming its fix.
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
pub fn snapshot(
    color: bool,
    daemon: Option<&DaemonView>,
    s: &Summary,
    models: &[ModelView],
    cost: Option<&Cost>,
    today_saved_usd: Option<f64>,
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

    // Hero panel — the headline is the MEASURED input-side saving (real, per-row): every
    // request's input is compressed and re-tokenized, so this is honest for all traffic.
    // The output side is NOT in the headline: the proxy never sees the un-instructed reply,
    // so any output saving is a benchmark projection that only holds when output is actually
    // shaped — projecting it onto agent traffic (output left unshaped by design) would
    // overstate the number ~2.7×. We surface it separately, clearly labeled, below.
    // Hero — the biggest TRUE numbers first: tokens trimmed (measured, absolute) and the
    // dollars that actually came off the bill (`net_saved`, cache-discounted). The gross
    // list-rate figure moves to a dim support line below: it's the ceiling, not the claim.
    let hero = match cost {
        Some(c) => {
            // Anchor the all-time number in the present: "what did it do for me today?"
            // Hidden when today has no priced saving — a perpetual $0.00 reads like fake data.
            let today = today_saved_usd
                .filter(|t| *t >= 0.005)
                .map(|t| ui::paint(color, Tone::Dim, &format!(" · today ${t:.2}")))
                .unwrap_or_default();
            // The $ headline is the live-zone estimate when usage data supports it (cut
            // tokens bill at the fresh/write rate, not the cache-read blend) — marked `~`;
            // the measured floor moves to the ladder line below. Without a usage split the
            // two coincide and the figure prints unmarked.
            let real = if c.live_saved > c.net_saved + 0.005 {
                format!("~${:.2} off your real bill", c.live_saved)
            } else {
                format!("${:.2} off your real bill", c.net_saved)
            };
            format!(
                "{} trimmed{}   {}   {} requests",
                ui::paint(
                    color,
                    Tone::Accent,
                    &format!("{} tokens", ui::human(s.saved()))
                ),
                today,
                ui::paint(color, Tone::Bold, &real),
                ui::commas(s.events),
            )
        }
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
        "llmtrim"
    } else {
        "saved (all time)"
    };
    o.push_str(&ui::panel(color, title, &[hero]));
    o.push('\n');
    // The honesty ladder, one quiet line: the same cut is worth more at list rates — show
    // the ceiling so the hero floor never reads as the whole story. Hidden when the traffic
    // uses no prompt cache (the two figures coincide).
    if let Some(c) = cost
        && c.net_saved + 0.005 < c.saved
    {
        o.push_str(&ui::paint(
            color,
            Tone::Dim,
            &format!(
                "  floor ${:.2} measured off the bill · ${:.2} at list — the estimate prices the cut where it bills (cache writes run 1.25×)\n",
                c.net_saved, c.saved
            ),
        ));
    }

    // savings axes, one gauge language for the whole section
    o.push_str(&ui::paint(color, Tone::Dim, "\n savings\n"));
    // Input axis — measured over the compressible surface only (input minus the frozen
    // cached prefix the stages skip by design), the honest % of what we're allowed to
    // touch. Metered rows only, so the figure is never diluted by legacy rows; ledgers
    // with no metered traffic yet fall back to the all-input axis.
    let new_before = s.metered_input_before - s.frozen_input_tokens;
    let new_after = s.metered_input_after - s.frozen_input_tokens;
    if s.frozen_input_tokens > 0 && new_before > 0 {
        o.push_str(&axis(color, "input", new_before, new_after));
        o.push_str(&ui::paint(color, Tone::Dim, "   (cache excluded)"));
        o.push('\n');
    } else {
        o.push_str(&axis(color, "input", s.input_before, s.input_after));
        o.push('\n');
    }
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
    // (No cache line: cache-safety is a property, not a per-run number. A token count here
    // credited llmtrim for the provider's own cache discount, and a static "we don't bust
    // your cache" reassurance is dashboard clutter — that belongs in the docs, not status.)

    // by-model table — MEASURED input-side saving per model (matches the honest headline):
    // input % saved and the input-side $ saved where the registry prices the model. No output
    // projection here, so a model that serves agent traffic isn't credited an unshaped-output win.
    // Grouped by cache usage when both kinds exist: un-cached $ comes straight off the bill,
    // cached $ is list value (the real bill is already cache-discounted) — the honesty split
    // is structural instead of an asterisk per row.
    if !models.is_empty() {
        o.push_str(&ui::paint(color, Tone::Dim, "\n by model\n"));
        let mut t = ui::table(color, &["model", "requests", "saved", "$ saved"]);
        let grouped = models.iter().any(|m| m.cached) && models.iter().any(|m| !m.cached);
        let add_rows = |t: &mut comfy_table::Table, cached: bool| {
            if grouped {
                let label = if cached {
                    "cached · $ at list value"
                } else {
                    "un-cached · $ off the bill"
                };
                t.add_row(vec![comfy_table::Cell::new(ui::paint(
                    color,
                    Tone::Dim,
                    label,
                ))]);
            }
            for m in models.iter().filter(|m| m.cached == cached) {
                // Metered models show the new-content % (the compressible surface, the
                // honest big number); pre-meter models fall back to the all-input figure.
                let pct = m.new_pct.unwrap_or(m.saved_pct);
                let pct_tone = if pct >= 0.0 { Tone::Accent } else { Tone::Warn };
                let name = ui::truncate(&m.name, 28);
                t.add_row(vec![
                    comfy_table::Cell::new(if grouped { format!("  {name}") } else { name }),
                    ui::right(ui::commas(m.events)),
                    ui::right(ui::paint(color, pct_tone, &format!("{:+.0}%", -pct))),
                    ui::right(
                        m.cost_saved
                            .map(|c| ui::paint(color, Tone::Accent, &format!("${c:.2}")))
                            .unwrap_or_default(),
                    ),
                ]);
            }
        };
        add_rows(&mut t, false);
        add_rows(&mut t, true);
        for line in t.to_string().lines() {
            o.push_str(&format!(" {line}\n"));
        }
        if models.iter().any(|m| m.new_pct.is_some()) {
            o.push_str(&ui::paint(
                color,
                Tone::Dim,
                " saved % measured without the cached prefix where metered\n",
            ));
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
    daemon: Option<&DaemonView>,
) -> String {
    let v = json!({
        "daemon": daemon.map(|d| json!({
            "running": d.running,
            "health": health(d).label(),
            "pid": d.running.then_some(d.pid),
            "port": d.running.then_some(d.port),
            "uptime_secs": d.running.then_some(d.uptime_secs),
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
    let mut models: Vec<ModelView> = tracker
        .by_model()?
        .into_iter()
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
    Ok(models)
}

/// Today's priced saving (UTC), for the hero's recency anchor. `None` when nothing
/// priced ran today — the dashboard hides the figure rather than showing $0.00.
pub fn today_saved_usd(tracker: &Tracker) -> Option<f64> {
    let models = tracker.by_model_today().ok()?;
    cost_estimate(&models).map(|c| c.saved)
}

fn cache_multipliers(provider: &str) -> (f64, f64) {
    match provider {
        "anthropic" => (0.10, 1.25),
        "openai" => (0.50, 0.0),
        "google" => (0.25, 0.0),
        _ => (1.0, 0.0),
    }
}

/// USD cost figures priced per model via [`llm_prices`], `None` when no recorded model
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
fn cost_estimate(models: &[ModelRow]) -> Option<Cost> {
    let mut cost = Cost {
        saved: 0.0,
        spend: 0.0,
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
        let out = snapshot(false, None, &summ(), &models, Some(&cost), None);
        // Headline shows the MEASURED figures (tokens trimmed + real-bill $), not a
        // projection that assumes shaping.
        assert!(
            out.contains("tokens trimmed") && out.contains("$12.47 off your real bill"),
            "hero shows measured trimmed tokens + real-bill saving"
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
            net_saved: 10.0,
            live_saved: 10.0,
            out_spend_shaped: 5.0,
        };
        let out = snapshot(false, None, &summ(), &[], Some(&cost), None);
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
            out_spend: 5.0,
            net_saved: 10.0,
            live_saved: 10.0,
            out_spend_shaped: 0.0,
        };
        assert!((c.projected_saved() - c.saved).abs() < 1e-9);
        let out = snapshot(false, None, &summ(), &[], Some(&c), None);
        // No footnote either way: the output axis' "(est · if output shaped)" tag carries
        // the caveat; a dedicated line only appears when the projection is ≥ $1.
        assert!(!out.contains("more saved by output shaping"));
    }

    #[test]
    fn net_line_surfaces_cache_discounted_saving() {
        // Cache-heavy traffic: tokens cut are worth $100 at list rates but $25 came off
        // the real (cache-discounted) bill. Both must be visible — the hero leads with the
        // real-bill figure, the dim line right under it carries the list-rate ceiling.
        let c = Cost {
            saved: 100.0,
            spend: 50.0,
            out_spend: 5.0,
            net_saved: 25.0,
            live_saved: 25.0,
            out_spend_shaped: 0.0,
        };
        let out = snapshot(false, None, &summ(), &[], Some(&c), None);
        assert!(
            out.contains("$25.00 off your real bill"),
            "hero leads with the real-bill figure"
        );
        assert!(
            out.contains("$100.00 at list"),
            "list-rate ceiling printed under the hero"
        );

        // No prompt cache → the figures coincide → no redundant line.
        let same = Cost {
            net_saved: 100.0,
            live_saved: 100.0,
            ..c
        };
        let out = snapshot(false, None, &summ(), &[], Some(&same), None);
        assert!(!out.contains("at list"));
    }

    #[test]
    fn new_content_axis_and_live_hero() {
        // Metered traffic: 1.0M of the 1.5M metered prompt is frozen prefix, so the
        // compressible surface went 500K → 100K (−80%) — the axis the meter unlocks.
        let mut s = summ();
        s.frozen_input_tokens = 1_000_000;
        s.metered_input_before = 1_500_000;
        s.metered_input_after = 1_100_000;
        let c = Cost {
            saved: 100.0,
            spend: 50.0,
            out_spend: 5.0,
            net_saved: 25.0,
            out_spend_shaped: 0.0,
            live_saved: 80.0,
        };
        let out = snapshot(false, None, &s, &[], Some(&c), None);
        assert!(
            out.contains("~$80.00 off your real bill"),
            "hero = live-zone estimate, ~-marked: {out}"
        );
        assert!(
            out.contains("floor $25.00 measured"),
            "measured floor on the ladder line: {out}"
        );
        assert!(
            out.contains("500.0K → 100.0K") && out.contains("(cache excluded)"),
            "input axis over the compressible surface: {out}"
        );
        assert!(
            !out.contains("2.1M → 1.2M"),
            "diluted all-input axis replaced, not shown alongside: {out}"
        );

        // No metered rows → fall back to the all-input axis (pre-meter ledgers unchanged).
        let out = snapshot(false, None, &summ(), &[], Some(&c), None);
        assert!(!out.contains("cache excluded"));
        assert!(out.contains("2.1M → 1.2M"), "fallback axis: {out}");
    }

    #[test]
    fn snapshot_empty_ledger_guides_user() {
        let out = snapshot(false, None, &Summary::default(), &[], None, None);
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

    #[test]
    fn header_healthy_collapses_to_one_calm_line() {
        let out = render_header(false, &dv());
        assert!(out.contains("running"));
        assert!(out.contains("✓ port") && out.contains("✓ env :8788") && out.contains("✓ ca"));
        assert!(out.contains("last request 4s ago"));
        assert!(out.contains("autostart on"));
        assert!(out.contains("v0.1.0"));
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
        let out = snapshot(false, Some(&unwired), &Summary::default(), &[], None, None);
        assert!(out.contains("nothing routes through it"));

        // Running and wired → it's just waiting; don't tell the user to re-setup.
        let out = snapshot(false, Some(&dv()), &Summary::default(), &[], None, None);
        assert!(out.contains("waiting for the first request"));

        // Not installed at all → the original guidance.
        let off = DaemonView {
            running: false,
            env_port: None,
            ..dv()
        };
        let out = snapshot(false, Some(&off), &Summary::default(), &[], None, None);
        assert!(out.contains("no activity yet"));
    }

    #[test]
    fn hero_shows_today_when_priced() {
        let cost = Cost {
            saved: 100.0,
            spend: 50.0,
            out_spend: 0.0,
            net_saved: 100.0,
            live_saved: 100.0,
            out_spend_shaped: 0.0,
        };
        let out = snapshot(false, None, &summ(), &[], Some(&cost), Some(1.84));
        assert!(out.contains("today $1.84"));
        // A ~zero today figure is hidden — an idle proxy must not print "$0.00 today".
        let out = snapshot(false, None, &summ(), &[], Some(&cost), Some(0.0));
        assert!(!out.contains("today $"));
        let out = snapshot(false, None, &summ(), &[], Some(&cost), None);
        assert!(!out.contains("today $"));
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
