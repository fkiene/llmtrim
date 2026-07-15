//! `statusline` — one elegant line for Claude Code's custom status line.
//!
//! Claude Code pipes a JSON session blob on stdin and renders whatever the command
//! prints (see <https://code.claude.com/docs/en/statusline>). This module reads that
//! blob, folds in llmtrim's own live signals from the ledger + config (compression
//! saved, interceptor health, the active `sub` reroute), and prints a single
//! width-adaptive line:
//!
//! ```text
//! ◆ Opus→gpt-5.6-terra   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 3h·24% · 4d·12%   ♻ 63% cached
//! ```
//!
//! The three left segments (model→backend, context, ✂ trim) are core and never
//! truncate; the extras (5h/7d quota, then this turn's prompt-cache reuse) shed right-to-left
//! as the terminal narrows (`COLUMNS`). The context gauge fills and colours against the *real*
//! window of the model serving the turn — the rerouted backend's window under `sub`, not
//! Claude's — green below 40%, orange 40–65%, red above; and red whenever the prompt cache has
//! gone cold, where the cache segment becomes `♻ cache cold`. Segments whose data is
//! absent — no reroute, an API-key user with no rate limits — simply don't render.
//!
//! Under `sub` the arrow shows the concrete model that served the last turn (e.g.
//! `→gpt-5.6-terra`) for Codex reroutes; Kimi shows the provider shortname (`→kimi`), since all
//! its tiers collapse to one internal wire id. The arrow is read from the ledger — what actually
//! answered — not from config, because in `fallback` mode config predicts nothing: Anthropic
//! serves the turn and the chain fires only if it fails. So there the arrow appears only on turns
//! a backend really served, and the gauge keeps Claude's window until one does.
//!
//! `install` wires it into `~/.claude/settings.json`; rendering itself never touches the
//! network or API tokens.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::monitor::{self, DaemonView, Health};
use crate::tracking::Tracker;
use crate::ui;

/// Default context window (tokens) when neither the Claude Code blob nor the model registry
/// reports one — a conservative floor so the gauge still renders.
const CTX_WINDOW_DEFAULT: i64 = 200_000;
const CTX_BAR_WIDTH: usize = 8;
/// Fill fraction of the *real* window at which the gauge leaves green (< 40%) and enters red
/// (>= 65%); the band between is amber. Percent of the window, per the user's spec — a capacity
/// reading, not the fixed quality budget.
const CTX_GREEN_PCT: i64 = 40;
const CTX_AMBER_PCT: i64 = 65;
/// Prompt-cache TTL. Claude Code sends `cache_control: {ttl: "1h"}` on ~92% of breakpoints
/// (verified from the capture corpus: 3333×1h vs 288×5m), and the cache stays warm up to the
/// TTL since the last intercepted request. Past this idle gap the cache is cold, so a stale
/// "cached %" would lie about the next turn paying a cold write — show it cold instead.
/// Shared with [`crate::guard`], so the hook and the status line agree on when a cache is cold.
pub(crate) const CACHE_TTL_SECS: i64 = 3600;
/// How often Claude Code re-runs the status line while a session sits idle. Without it Claude
/// Code only re-renders on conversation events, so an abandoned session keeps the line drawn at
/// its last turn — green and warm — straight through the cache expiring, and `♻ cache cold`
/// only appears *after* the turn that pays for it. Rendering is local (no network, no tokens),
/// so a refresh costs one short process; 5 minutes is far finer than the 1h TTL it watches, and
/// 25× cheaper than polling every 10s would be.
const REFRESH_INTERVAL_SECS: i64 = 300;

// ── ANSI palette ────────────────────────────────────────────────────────────────
// The status line is captured by Claude Code (never a TTY), but Claude Code renders ANSI,
// so colour is emitted unconditionally — gated only by NO_COLOR, per the docs' examples.

const BRAND: &str = "38;2;153;204;255"; // llmtrim accent blue
const CYAN: &str = "36"; // codex
const VIOLET: &str = "38;2;181;137;255"; // kimi
const GREEN: &str = "32";
const AMBER: &str = "33";
const ORANGE: &str = "38;2;255;140;0"; // true orange, distinct from amber quota tiers
const RED: &str = "31";
const DIM: &str = "2";
const BOLD: &str = "1";

fn paint(color: bool, code: &str, s: &str) -> String {
    if color {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

// ── the Claude Code stdin blob (only the fields we render) ────────────────────────

struct CcInput {
    model: String,
    /// Claude Code's `effort.level`. Parsed but not currently rendered — the field doesn't track
    /// the `/model` effort override, so showing it is misleading (see [`model_segment`]). Kept so
    /// re-enabling is a one-liner once Claude Code fixes it.
    #[allow(dead_code)]
    effort: Option<String>,
    /// Claude Code's own session id (the `x-claude-code-session-id` it also tags on every
    /// intercepted request), used to scope trim to *this* session's ledger rows. `None` if
    /// absent — trim then falls back to the lifetime figure.
    session_id: Option<String>,
    /// Total input tokens currently in the context window (fresh + cache), from the last
    /// API response. `0` before the first response.
    ctx_tokens: i64,
    /// Claude Code's model id (`model.id`, e.g. `claude-opus-4-8[1m]`), used to resolve the real
    /// window under a `sub` reroute (where CC still reports its own Claude window, not the
    /// backend's).
    model_id: String,
    /// The model's context window in tokens, as Claude Code reports it (`context_window_size`).
    /// `None` when absent — the gauge then falls back to the registry or a default.
    window: Option<i64>,
    /// 5-hour and 7-day rate-limit usage %, Claude.ai subscribers only.
    five_hour_pct: Option<f64>,
    seven_day_pct: Option<f64>,
    /// Rate-limit window reset times (Unix epoch seconds, as Claude Code sends them), used to show
    /// the remaining duration beside each quota.
    five_hour_resets_at: Option<i64>,
    seven_day_resets_at: Option<i64>,
    /// Share of this turn's input served from the prompt cache, % — computed from the last
    /// API call's `current_usage`. `None` before the first response or right after `/compact`.
    cache_pct: Option<f64>,
}

fn parse_cc(input: &str) -> CcInput {
    let v: Value = serde_json::from_str(input).unwrap_or(Value::Null);
    let model = v
        .pointer("/model/display_name")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    let model_id = v
        .pointer("/model/id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let effort = v
        .pointer("/effort/level")
        .and_then(Value::as_str)
        .map(str::to_string);
    let session_id = v
        .pointer("/session_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let ctx_tokens = v
        .pointer("/context_window/total_input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let window = v
        .pointer("/context_window/context_window_size")
        .and_then(Value::as_i64)
        .filter(|&w| w > 0);
    let five_hour_pct = v
        .pointer("/rate_limits/five_hour/used_percentage")
        .and_then(Value::as_f64);
    let seven_day_pct = v
        .pointer("/rate_limits/seven_day/used_percentage")
        .and_then(Value::as_f64);
    let five_hour_resets_at = v
        .pointer("/rate_limits/five_hour/resets_at")
        .and_then(Value::as_i64);
    let seven_day_resets_at = v
        .pointer("/rate_limits/seven_day/resets_at")
        .and_then(Value::as_i64);
    let cache_pct = {
        let cu = v.pointer("/context_window/current_usage");
        let field = |k: &str| {
            cu.and_then(|c| c.get(k))
                .and_then(Value::as_i64)
                .unwrap_or(0)
        };
        let read = field("cache_read_input_tokens");
        let total = read + field("input_tokens") + field("cache_creation_input_tokens");
        (total > 0).then(|| read as f64 / total as f64 * 100.0)
    };
    CcInput {
        model,
        effort,
        session_id,
        model_id,
        ctx_tokens,
        window,
        five_hour_pct,
        seven_day_pct,
        five_hour_resets_at,
        seven_day_resets_at,
        cache_pct,
    }
}

// ── llmtrim's own signals, read from the ledger + config ──────────────────────────

struct Led {
    health: Health,
    /// Input compression saved, % of input tokens — scoped to this Claude Code session when
    /// its id is known, else lifetime. `None` when there are no rows to measure yet (a fresh
    /// session), which renders an idle `✂ –` rather than a misleading `✂ 0.0%`.
    trim_pct: Option<f64>,
    /// Active reroute provider (`codex`/`kimi`), if `sub` is on.
    reroute: Option<String>,
    /// The real window (tokens) of the model actually serving the turn, when a `sub` reroute is
    /// active — looked up from the model registry, since Claude Code reports its own Claude
    /// window on the wire, not the backend's. `None` when not rerouted (use the blob's window).
    reroute_window: Option<i64>,
    /// The concrete upstream model id resolved for a `sub` reroute (e.g. "gpt-5.6-terra" for a
    /// Codex Opus tier), rendered after the arrow so the status line shows the *real* model
    /// serving the turn. `None` when not rerouted, or on Kimi (all tiers collapse to one wire
    /// id) — the arrow then falls back to the provider shortname.
    resolved_model: Option<String>,
    /// The prompt cache has gone cold: the session has been idle past the TTL, so the next turn
    /// pays a cold write. Renders the cache segment red with a `/compact` nudge.
    cache_cold: bool,
    /// Cache-hit rate of this session's last *completed* turn, %, as the proxy measured it on the
    /// wire. Stands in for the blob's `current_usage` while a turn is in flight — see
    /// [`cache_pct_for`].
    last_cache_pct: Option<f64>,
    /// Input tokens of that same turn — stands in for the blob's `total_input_tokens` while a
    /// turn is in flight, so the gauge doesn't empty. See [`ctx_tokens_for`].
    last_ctx_tokens: i64,
}

/// Minimal [`DaemonView`] for the health check — mirrors `main::daemon_view` but fills only
/// the fields [`monitor::health`] reads (running/pid/port/port_accepting/env_port/ca), since
/// the status line needs the health verdict, not the full dashboard header.
fn proxy_health() -> Health {
    use crate::daemon;
    let ca_present = matches!(crate::serve::ca_cert_path(), Ok(p) if p.exists());
    let env_port = crate::setup::configured_port();
    let view = |running: bool, pid: u32, port: u16, accepting: bool| DaemonView {
        running,
        pid,
        port,
        uptime: String::new(),
        uptime_secs: 0,
        ca_present,
        port_accepting: accepting,
        env_port,
        autostart: false,
        restarts: 0,
        version: None,
        binary_version: String::new(),
        log_path: None,
        last_request: None,
    };
    let dv = match daemon::running() {
        Some(s) => view(true, s.pid, s.port, daemon::probe_port(s.port)),
        // No pidfile: trust a live probe on the wired port before declaring stopped.
        None => match env_port.filter(|&p| daemon::probe_port(p)) {
            Some(p) => view(true, 0, p, true),
            None => view(false, 0, 0, false),
        },
    };
    monitor::health(&dv)
}

fn ledger_snapshot(cc: &CcInput) -> Led {
    let cfg = llmtrim_core::config::RuntimeConfig::get();
    let global = cfg.sub.clone().filter(|s| !s.is_empty() && s != "off");
    let window_intent = crate::window_sub::lookup(cc.session_id.as_deref());
    let configured = match &window_intent {
        Some(crate::window_sub::Intent::Enabled { provider }) => Some(provider.clone()),
        Some(crate::window_sub::Intent::Disabled) => None,
        None => global,
    };
    let health = proxy_health();

    // One session-row read serves both trim and cache-cold. Match on `cc_session_id` — the real
    // Claude Code session id the proxy now records — not the ledger's `session_id`, which is a
    // hash of the system prompt and never equals Claude Code's UUID. The per-session ledger view
    // lives behind the `breakdown` feature; without it we scope nothing and fall back to lifetime.
    let row = cc.session_id.as_deref().and_then(session_row);
    // Lifetime figure, read only when we have no session to scope to.
    let lifetime = || {
        Tracker::open()
            .and_then(|t| t.summary())
            .ok()
            .filter(|s| s.input_before > 0)
            .map(|s| ui::saved_pct(s.input_before as f64, s.input_after as f64))
    };
    let session_savings = row.as_ref().map(|r| (r.input_before, r.input_after));
    let trim_pct = trim_for(cc.session_id.as_deref(), session_savings, lifetime);
    let ledger_cold = row.as_ref().is_some_and(|r| cache_cold(&r.last_ts));
    let cache_cold = effective_cache_cold(cc, ledger_cold);

    let arrow = if matches!(window_intent, Some(crate::window_sub::Intent::Disabled)) {
        ArrowSource::None
    } else {
        arrow_source(configured.as_deref(), cfg.sub_fallback, row.as_ref())
    };
    let (reroute, reroute_window, resolved_model) = match arrow {
        // Truth: the backend that answered the last turn, and the model recorded on it.
        ArrowSource::Served { provider, model } => {
            let window = model.as_deref().and_then(upstream_window);
            // Kimi collapses every tier to one internal wire id, which tells the user nothing
            // the provider shortname doesn't — so the arrow keeps the shortname there.
            let shown = model.filter(|_| provider != "kimi");
            (Some(provider), window, shown)
        }
        // Prediction (always mode, no turn yet): resolve from the configured tier mapping.
        ArrowSource::Predicted(provider) => (
            Some(provider.clone()),
            reroute_real_window(&provider, &cc.model_id, &cfg.sub_tiers),
            reroute_resolved_model(&provider, &cc.model_id, &cfg.sub_tiers),
        ),
        ArrowSource::None => (None, None, None),
    };

    Led {
        health,
        trim_pct,
        reroute,
        reroute_window,
        resolved_model,
        cache_cold,
        last_cache_pct: row
            .as_ref()
            .and_then(|r| r.last_cache_hit)
            .map(|f| f * 100.0),
        last_ctx_tokens: row.as_ref().map_or(0, |r| r.last_input_tokens),
    }
}

/// Resolve the context occupancy the gauge fills against. Claude Code reports
/// `total_input_tokens` from the *last response*, so like `current_usage` it reads 0 on every
/// re-render while a turn is in flight — emptying the bar mid-request and refilling it after.
/// The proxy counted the same input on the wire, so the last completed turn holds the gauge
/// steady. A genuinely empty context (fresh session, no turn yet) has no ledger row either, so
/// it still renders empty rather than inventing a fill.
fn ctx_tokens_for(blob: i64, ledger_last: i64) -> i64 {
    if blob > 0 { blob } else { ledger_last }
}

/// Resolve the cache figure the segment shows. Claude Code fills `current_usage` only once a
/// response's usage has landed, so it is absent from every re-render *during* a turn — trusting
/// it alone drops the segment mid-request and pops it back afterwards. The proxy measured the
/// same quantity on the wire for the last completed turn, so it holds the number steady across
/// the gap instead of blanking it.
fn cache_pct_for(blob: Option<f64>, ledger_last: Option<f64>) -> Option<f64> {
    blob.or(ledger_last)
}

/// Where the reroute arrow's content comes from.
enum ArrowSource {
    /// The ledger recorded this backend serving the session's last turn. Ground truth.
    Served {
        provider: String,
        model: Option<String>,
    },
    /// No turn recorded yet, but `always` mode reroutes every turn — so config predicts it.
    Predicted(String),
    /// Anthropic is serving (or nothing is configured): no arrow.
    None,
}

/// Decide what the reroute arrow may claim.
///
/// The ledger wins whenever it has a turn for this session, because only it knows who actually
/// answered. Config is consulted solely to bridge the gap before the first turn lands, and only in
/// `always` mode — where every turn reroutes, so the prediction is sound and the arrow doesn't have
/// to blink in late on a fresh session.
///
/// In `fallback` mode config predicts nothing: the chain fires only when Anthropic actually fails.
/// Claiming a reroute there would be a lie on every healthy turn (and would rescale the context
/// gauge to the wrong backend's window), so the arrow stays off until the ledger shows a turn a
/// backend really served.
fn arrow_source(
    configured: Option<&str>,
    fallback_mode: bool,
    row: Option<&SessionLedgerRow>,
) -> ArrowSource {
    match row {
        Some(r) => match &r.last_sub_provider {
            Some(provider) => ArrowSource::Served {
                provider: provider.clone(),
                model: r.last_model.clone(),
            },
            // A recorded turn that no backend served: Anthropic answered it.
            None => ArrowSource::None,
        },
        None => match configured {
            Some(p) if !fallback_mode => ArrowSource::Predicted(p.to_string()),
            _ => ArrowSource::None,
        },
    }
}

/// The ledger fields the status line needs from a session's row: its summed savings and the
/// timestamp of its last turn (for the cold-cache check).
struct SessionLedgerRow {
    input_before: i64,
    input_after: i64,
    last_ts: String,
    /// The `sub` backend that served this session's most recent turn, and that turn's model id.
    /// The ground truth behind the reroute arrow: in fallback mode a reroute happens only when
    /// Anthropic actually failed, which config cannot predict.
    last_sub_provider: Option<String>,
    last_model: Option<String>,
    /// Cache-hit fraction of this session's most recent *completed* turn. Claude Code omits
    /// `current_usage` from the blob while a turn is in flight, so this is what keeps the cache
    /// segment from blanking out mid-request.
    last_cache_hit: Option<f64>,
    /// Input tokens of that same turn — what the context gauge falls back to when the blob
    /// reports no occupancy mid-request.
    last_input_tokens: i64,
}

/// This Claude Code session's aggregated ledger row, matched on the real `cc_session_id`. Behind
/// the `breakdown` feature (the per-session query lives there); `None` without it, so trim falls
/// back to lifetime and cold-cache is never flagged.
#[cfg(feature = "breakdown")]
fn session_row(sid: &str) -> Option<SessionLedgerRow> {
    crate::breakdown::db::BreakdownDb::open()
        .ok()
        .and_then(|db| db.sessions().ok())
        .and_then(|rows| {
            rows.into_iter()
                .find(|r| r.cc_session_id.as_deref() == Some(sid))
        })
        .map(|r| SessionLedgerRow {
            input_before: r.input_before,
            input_after: r.input_after,
            last_ts: r.last_ts,
            last_sub_provider: r.last_sub_provider,
            last_model: r.last_model,
            last_cache_hit: r.last_cache_hit,
            last_input_tokens: r.last_input_tokens,
        })
}

#[cfg(not(feature = "breakdown"))]
fn session_row(_sid: &str) -> Option<SessionLedgerRow> {
    None
}

/// The concrete model a `sub` backend served this session's last turn with, if any — the only
/// thing [`crate::guard`] takes from the ledger, and only to price a rerouted turn correctly.
/// `None` (no row, no reroute, or no `breakdown` feature) never changes the guard's decision.
pub(crate) fn session_sub_model(session_id: &str) -> Option<String> {
    let row = session_row(session_id)?;
    row.last_sub_provider.as_ref()?;
    row.last_model
}

/// Decide the trim figure: this session's own savings when we have its row; idle (`None`) for a
/// known Claude Code session with no recorded turn yet — *not* the lifetime figure, which would
/// flash a misleading number for a beat before the first turn lands; and only the lifetime figure
/// (via `lifetime`) when there is no session id to scope to at all.
fn trim_for(
    session_id: Option<&str>,
    session_savings: Option<(i64, i64)>,
    lifetime: impl FnOnce() -> Option<f64>,
) -> Option<f64> {
    match (session_savings, session_id) {
        (Some((before, after)), _) => {
            (before > 0).then(|| ui::saved_pct(before as f64, after as f64))
        }
        (None, Some(_)) => None,
        (None, None) => lifetime(),
    }
}

/// The real context window of the model a `sub` reroute actually serves this turn, from the
/// model registry. Claude Code reports its own Claude window on the wire, so under reroute we
/// must resolve the backend model id and look *it* up. `None` if unknown (gauge keeps the blob
/// window). Kimi's internal id isn't a registry key, so it's mapped to its public model id.
///
/// Reroute resolution lives behind the `intercept` feature; without it a `sub` reroute can't be
/// active anyway, so the gauge just uses the blob's window.
#[cfg(feature = "intercept")]
fn reroute_real_window(
    provider: &str,
    incoming_model_id: &str,
    tiers: &std::collections::BTreeMap<String, String>,
) -> Option<i64> {
    use crate::reroute::SubProvider;
    let sp = match provider {
        "codex" => SubProvider::Codex,
        "kimi" => SubProvider::Kimi,
        _ => return None,
    };
    upstream_window(&crate::reroute::resolve_model(sp, incoming_model_id, tiers))
}

/// Registry window of a concrete upstream model id. `kimi-for-coding` is an internal routing id,
/// not a models.dev key — map it to the public one.
#[cfg(feature = "intercept")]
fn upstream_window(model: &str) -> Option<i64> {
    let lookup = if model == crate::reroute::KIMI_MODEL {
        "moonshotai/kimi-k2"
    } else {
        model
    };
    llmtrim_core::context_window(lookup).map(|w| w as i64)
}

#[cfg(not(feature = "intercept"))]
fn upstream_window(_model: &str) -> Option<i64> {
    None
}

#[cfg(not(feature = "intercept"))]
fn reroute_real_window(
    _provider: &str,
    _incoming_model_id: &str,
    _tiers: &std::collections::BTreeMap<String, String>,
) -> Option<i64> {
    None
}

/// Parallel to [`reroute_real_window`]: returns the concrete upstream model id (e.g.
/// "gpt-5.6-terra") chosen by tier mapping for the status line. `None` for Kimi, whose tiers all
/// collapse to one internal wire id: the shortname is what a reader wants there.
#[cfg(feature = "intercept")]
fn reroute_resolved_model(
    provider: &str,
    incoming_model_id: &str,
    tiers: &std::collections::BTreeMap<String, String>,
) -> Option<String> {
    use crate::reroute::SubProvider;
    if provider != "codex" {
        return None;
    }
    Some(crate::reroute::resolve_model(
        SubProvider::Codex,
        incoming_model_id,
        tiers,
    ))
}

#[cfg(not(feature = "intercept"))]
fn reroute_resolved_model(
    _provider: &str,
    _incoming_model_id: &str,
    _tiers: &std::collections::BTreeMap<String, String>,
) -> Option<String> {
    None
}

/// Whether the prompt cache has gone cold: `last_ts` (rfc3339, the most recent intercepted
/// request) is older than the TTL. An unparseable timestamp is treated as not-cold (don't warn
/// on a data glitch).
fn cache_cold(last_ts: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(last_ts)
        .map(|t| chrono::Utc::now().signed_duration_since(t).num_seconds() >= CACHE_TTL_SECS)
        .unwrap_or(false)
}

/// Fold ledger idle time with Claude Code's live context. `/compact` and other internal turns
/// can refresh the prompt cache without updating the interceptor's `last_ts`, so a stale ledger
/// alone would keep the gauge red after the user already compacted.
fn effective_cache_cold(cc: &CcInput, ledger_cold: bool) -> bool {
    if !ledger_cold {
        return false;
    }
    // Post-compact reset: empty window, no `current_usage` yet. Do not also key off
    // `cache_creation_tokens` from the last response — that field is stale once the TTL
    // expires and would hide a legitimate cold warning on an idle session.
    cc.ctx_tokens != 0 || cc.cache_pct.is_some()
}

// ── rendering ─────────────────────────────────────────────────────────────────────

/// Colour a rate-limit % with an extra orange step before red: green < 70, amber 70–80,
/// orange 80–90, red ≥ 90 — so a filling quota escalates in visible stages, not one jump to red.
fn quota_color(pct: f64) -> &'static str {
    if pct >= 90.0 {
        RED
    } else if pct >= 80.0 {
        ORANGE
    } else if pct >= 70.0 {
        AMBER
    } else {
        GREEN
    }
}

/// The largest useful unit left before a quota window resets. `resets_at` is Unix epoch seconds,
/// the shape Claude Code actually sends (`"resets_at": 1738425600`).
fn remaining_time_label(resets_at: Option<i64>) -> Option<String> {
    let minutes = (resets_at? - chrono::Utc::now().timestamp()).max(0) / 60;
    Some(if minutes >= 24 * 60 {
        format!("{}d", minutes / (24 * 60))
    } else if minutes >= 60 {
        format!("{}h", minutes / 60)
    } else {
        format!("{minutes}m")
    })
}

/// `◆ Opus→gpt-5.6-terra` — health-brand glyph, model (Claude tier name), reroute target.
/// The target is the resolved upstream model for Codex reroutes (so you see the real model
/// serving the turn) or the provider shortname for Kimi. The arrow is suppressed when the
/// proxy isn't healthy (traffic isn't being intercepted, so it isn't actually rerouting).
///
/// Effort is intentionally not shown: Claude Code's `effort.level` field doesn't track the
/// `/model` effort override (it reports a stale/base level), so rendering it is misleading. The
/// field is still parsed — re-append `·{effort}` here once Claude Code fixes the field.
fn model_segment(cc: &CcInput, led: &Led, color: bool) -> String {
    let mut s = format!(
        "{} {}",
        paint(color, BRAND, "◆"),
        paint(color, BOLD, &cc.model)
    );
    if let (Health::Healthy, Some(p)) = (led.health, &led.reroute) {
        let code = match p.as_str() {
            "kimi" => VIOLET,
            _ => CYAN,
        };
        let tail = led.resolved_model.clone().unwrap_or_else(|| p.clone());
        s.push_str(&paint(color, code, &format!("→{tail}")));
    }
    s
}

/// The real context window in play: the model actually serving the turn. Under a `sub` reroute
/// that's the backend model's window (from the registry); otherwise Claude Code's reported
/// `context_window_size`; falling back to a default when neither is known.
fn effective_window(cc: &CcInput, led: &Led) -> i64 {
    led.reroute_window
        .or(cc.window)
        .filter(|&w| w > 0)
        .unwrap_or(CTX_WINDOW_DEFAULT)
}

/// `▓▓▓▓▓░░░ 142k` — gauge filled and coloured against the *real* window: green below 40% of the
/// window, orange 40–65%, red at/above 65% — and red unconditionally when the prompt cache has
/// gone cold (idle past the TTL), since the next turn then pays a cold write regardless of fill.
/// Label is absolute k.
fn context_segment(ctx_tokens: i64, window: i64, cache_cold: bool, color: bool) -> String {
    let tokens = ctx_tokens.max(0);
    let window = window.max(1);
    // Clamp before multiplying so a pathological token count can't overflow (bar pins full anyway).
    let filled = (tokens.min(window) * CTX_BAR_WIDTH as i64 / window) as usize;
    let bar: String = "▓".repeat(filled) + &"░".repeat(CTX_BAR_WIDTH - filled);
    let k = (tokens as f64 / 1000.0).round() as i64;
    // Ratio in f64 so a pathologically large token count can't overflow the multiply.
    let pct = (tokens as f64 / window as f64 * 100.0) as i64;
    let code = if cache_cold {
        RED
    } else if tokens == 0 {
        DIM
    } else if pct >= CTX_AMBER_PCT {
        RED
    } else if pct >= CTX_GREEN_PCT {
        ORANGE
    } else {
        GREEN
    };
    paint(color, code, &format!("{bar} {k}k"))
}

/// The third core segment: `✂ 6.8%` when healthy and this session has saved something, a dim
/// `✂ –` when healthy but idle (nothing trimmed yet — avoids a misleading `✂ 0.0%` while still
/// signalling "llmtrim is on"), `⚠ llmtrim degraded` when broken, and nothing at all when
/// cleanly stopped (llmtrim is simply off — not an error to flag).
fn trim_or_health_segment(led: &Led, color: bool) -> Option<String> {
    match led.health {
        // Trim is a savings figure, not a tier — no "bad" value to warn on — so it stays dim;
        // colour is reserved for state signals (health, quota, context, cold cache).
        Health::Healthy => Some(match led.trim_pct {
            Some(pct) => paint(color, DIM, &format!("✂ {pct:.1}%")),
            None => paint(color, DIM, "✂ –"),
        }),
        Health::Degraded => Some(paint(color, RED, "⚠ llmtrim degraded")),
        Health::Stopped => None,
    }
}

/// Build the ordered extra segments (quota, then this session's cache); later ones drop first on
/// a narrow terminal.
fn extra_segments(cc: &CcInput, led: &Led, color: bool) -> Vec<String> {
    let mut out = Vec::new();
    // One quota segment carrying both rolling windows, labelled by the time remaining until
    // each reset: `◔ 3h·15% · 4d·12%`. Only the *percentage* is coloured on its own value
    // (a maxed 5h doesn't paint a comfortable 7d); the time labels stay dim.
    let quota = |label: &str, p: f64| {
        format!(
            "{}{}",
            paint(color, DIM, &format!("{label}·")),
            paint(color, quota_color(p), &format!("{}%", p.floor() as i64))
        )
    };
    let glyph = paint(color, DIM, "◔");
    let sep = paint(color, DIM, "·");
    let five_label =
        remaining_time_label(cc.five_hour_resets_at).unwrap_or_else(|| "5h".to_string());
    let seven_label =
        remaining_time_label(cc.seven_day_resets_at).unwrap_or_else(|| "7d".to_string());
    match (cc.five_hour_pct, cc.seven_day_pct) {
        (Some(h), Some(d)) => {
            out.push(format!(
                "{glyph} {} {sep} {}",
                quota(&five_label, h),
                quota(&seven_label, d)
            ));
        }
        (Some(h), None) => out.push(format!("{glyph} {}", quota(&five_label, h))),
        (None, Some(d)) => out.push(format!("{glyph} {}", quota(&seven_label, d))),
        (None, None) => {}
    }
    if led.cache_cold {
        // Cache expired: the next turn pays a cold write. State only, no `/compact` nudge —
        // compacting re-reads the same cold context to summarise it, so it pays that charge too
        // rather than avoiding it (see `crate::guard::message`).
        out.push(paint(color, RED, "♻ cache cold"));
    } else if let Some(c) = cache_pct_for(cc.cache_pct, led.last_cache_pct) {
        // Floor, not round: only a genuine 100% cache shows `100%` (99.9 stays `99%`).
        out.push(paint(
            color,
            DIM,
            &format!("♻ {}% cached", c.floor() as i64),
        ));
    }
    out
}

const SEP: &str = "   ";

/// Assemble the line: core segments always in, extras appended left-to-right only while they
/// fit `cols` (0 = unknown width ⇒ no truncation). Once one extra overflows, stop — keeping
/// the higher-priority leftmost extras.
fn render_line(cc: &CcInput, led: &Led, cols: usize, color: bool) -> String {
    let mut core = vec![
        model_segment(cc, led, color),
        context_segment(
            ctx_tokens_for(cc.ctx_tokens, led.last_ctx_tokens),
            effective_window(cc, led),
            led.cache_cold,
            color,
        ),
    ];
    if let Some(seg) = trim_or_health_segment(led, color) {
        core.push(seg);
    }
    let mut line = core.join(SEP);

    for extra in extra_segments(cc, led, color) {
        let candidate = format!("{line}{SEP}{extra}");
        if cols == 0 || ui::visible_width(&candidate) <= cols {
            line = candidate;
        } else {
            break;
        }
    }
    line
}

/// Render the status line from a Claude Code JSON blob (stdin). Pure apart from the ledger
/// read, so tests drive `render_line` directly.
pub fn run() -> Result<()> {
    use std::io::Read;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).ok();

    let cc = parse_cc(&input);
    let led = ledger_snapshot(&cc);
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let color = std::env::var_os("NO_COLOR").is_none();

    println!("{}", render_line(&cc, &led, cols, color));
    Ok(())
}

// ── install / uninstall (wire ~/.claude/settings.json) ────────────────────────────

/// Whether Claude Code appears to be installed (its `~/.claude` config dir exists). Used by
/// `setup` to hint at the status line for Claude Code users only, not users of other agents —
/// setup itself is client-agnostic and never writes this file.
pub fn claude_code_present() -> bool {
    claude_settings_path()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::is_dir))
        .unwrap_or(false)
}

/// Claude Code settings.json — honors `CLAUDE_CONFIG_DIR`, else `~/.claude`.
pub(crate) fn claude_settings_path() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return Ok(PathBuf::from(dir).join("settings.json"));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("neither HOME nor USERPROFILE is set")?;
    Ok(PathBuf::from(home).join(".claude").join("settings.json"))
}

/// Shell-safe single-quoted path for hook/statusline command strings (POSIX; Windows uses doubles).
pub(crate) fn shell_quote_path(path: &str) -> String {
    #[cfg(windows)]
    {
        format!("\"{}\"", path.replace('"', "\\\""))
    }
    #[cfg(not(windows))]
    {
        // POSIX: single-quote and break/reopen for embedded quotes.
        format!("'{}'", path.replace('\'', "'\"'\"'"))
    }
}

/// Absolute or stable PATH path to this llmtrim binary (no subcommand).
pub fn stable_exe_string() -> String {
    std::env::current_exe()
        .ok()
        .filter(|p| p.exists())
        .map(|p| stable_executable_path(&p, std::env::var_os("PATH").as_deref()))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "llmtrim".to_string())
}

/// Atomically write pretty JSON (temp in same dir + rename).
pub(crate) fn atomic_write_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename onto {}", path.display()))?;
    Ok(())
}

/// Return a stable PATH alias for `current_exe` when one resolves to the same binary.
///
/// Package managers such as Homebrew launch versioned binaries from a `Cellar` path while
/// exposing a stable symlink in `PATH`. Persisting the versioned path makes Claude Code's
/// statusline break after an upgrade. We keep the absolute-command behavior, but prefer that
/// stable alias when it is provably the same executable. If no alias is available, the caller
/// retains the existing absolute-path fallback.
fn stable_executable_path(current_exe: &Path, path: Option<&OsStr>) -> PathBuf {
    let Some(current_real) = std::fs::canonicalize(current_exe).ok() else {
        return current_exe.to_path_buf();
    };
    let Some(name) = current_exe.file_name() else {
        return current_exe.to_path_buf();
    };
    let Some(path) = path else {
        return current_exe.to_path_buf();
    };

    for dir in std::env::split_paths(path) {
        if !dir.is_absolute() {
            continue;
        }
        let candidate = dir.join(name);
        if candidate == current_exe {
            continue;
        }
        if std::fs::canonicalize(&candidate).ok().as_ref() == Some(&current_real) {
            return candidate;
        }
    }

    current_exe.to_path_buf()
}

/// The command string Claude Code should run for one of our subcommands: an absolute path or a
/// stable PATH alias, so it survives a package manager replacing a versioned executable. Shared
/// with [`crate::guard`], which writes a hook command the same way.
pub(crate) fn exe_command(subcommand: &str) -> String {
    format!("{} {subcommand}", shell_quote_path(&stable_exe_string()))
}

/// The `statusLine` object we write.
fn statusline_config() -> Value {
    serde_json::json!({
        "type": "command",
        "command": exe_command("statusline"),
        "padding": 0,
        "refreshInterval": REFRESH_INTERVAL_SECS,
    })
}

/// Whether a Claude statusline command is one that `llmtrim statusline install` created.
fn is_llmtrim_statusline_command(command: &str) -> bool {
    is_llmtrim_command(command, "statusline")
}

/// Whether `command` is the llmtrim binary invoked with exactly `subcommand`. Recognize both
/// POSIX and Windows separators because settings can be moved between systems.
pub(crate) fn is_llmtrim_command(command: &str, subcommand: &str) -> bool {
    let Some(executable) = command.strip_suffix(&format!(" {subcommand}")) else {
        return false;
    };
    let executable = unquote_path_token(executable.trim());
    let name = executable.rsplit(['/', '\\']).next().unwrap_or(executable);
    name.strip_suffix(".exe").unwrap_or(name) == "llmtrim"
}

/// Strip one layer of shell quotes from a path token (double or single).
pub(crate) fn unquote_path_token(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        return &s[1..s.len() - 1];
    }
    if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        return &s[1..s.len() - 1];
    }
    s
}

/// Set our `statusLine` key on a parsed settings object, preserving every other key. Pure
/// transform (no I/O) so [`install`]'s merge is unit-testable.
fn set_statusline(settings: &mut Value, path: &std::path::Path) -> Result<()> {
    let obj = settings
        .as_object_mut()
        .with_context(|| format!("{} is not a JSON object", path.display()))?;
    obj.insert("statusLine".to_string(), statusline_config());
    Ok(())
}

/// Remove our `statusLine` key, returning whether one was present. Pure transform.
fn clear_statusline(settings: &mut Value, path: &std::path::Path) -> Result<bool> {
    let obj = settings
        .as_object_mut()
        .with_context(|| format!("{} is not a JSON object", path.display()))?;
    Ok(obj.remove("statusLine").is_some())
}

/// Wire the status line into `~/.claude/settings.json` (merging, not clobbering). `print`
/// just emits the settings snippet instead of editing the file.
pub fn install(print: bool) -> Result<()> {
    if print {
        let snippet = serde_json::json!({ "statusLine": statusline_config() });
        println!("{}", serde_json::to_string_pretty(&snippet)?);
        return Ok(());
    }

    let path = claude_settings_path()?;
    let mut settings: Value = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s)
            .with_context(|| format!("{} is not valid JSON", path.display()))?,
        Err(_) => Value::Object(Default::default()),
    };
    set_statusline(&mut settings, &path)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    write_settings(&path, &settings)?;

    println!(
        "Wired the llmtrim status line into {}. Restart Claude Code to see it.",
        path.display()
    );
    Ok(())
}

/// Ownership of the Claude Code `statusLine` key relative to this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnedStatus {
    /// No `statusLine` key (or missing settings file).
    Missing,
    /// Ours, but command path or refreshInterval differs from what we would write now.
    Stale,
    /// Ours and matches the current payload.
    Current,
    /// Present but not an llmtrim statusline — leave it alone.
    Foreign,
}

/// Whether ensure should install / refresh the status line.
pub fn owned_status() -> OwnedStatus {
    let Ok(path) = claude_settings_path() else {
        return OwnedStatus::Missing;
    };
    let Ok(s) = std::fs::read_to_string(&path) else {
        return OwnedStatus::Missing;
    };
    let Ok(settings) = serde_json::from_str::<Value>(&s) else {
        return OwnedStatus::Missing;
    };
    owned_status_of(&settings)
}

fn owned_status_of(settings: &Value) -> OwnedStatus {
    let Some(status_line) = settings.get("statusLine").and_then(Value::as_object) else {
        return OwnedStatus::Missing;
    };
    let Some(command) = status_line.get("command").and_then(Value::as_str) else {
        return OwnedStatus::Foreign;
    };
    if !is_llmtrim_statusline_command(command) {
        return OwnedStatus::Foreign;
    }
    let desired = statusline_config();
    let desired_cmd = desired.get("command").and_then(Value::as_str).unwrap_or("");
    let desired_refresh = desired
        .get("refreshInterval")
        .and_then(Value::as_i64)
        .unwrap_or(REFRESH_INTERVAL_SECS);
    let refresh = status_line.get("refreshInterval").and_then(Value::as_i64);
    if command == desired_cmd && refresh == Some(desired_refresh) {
        OwnedStatus::Current
    } else {
        OwnedStatus::Stale
    }
}

/// Outcome of [`sync_owned`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOutcome {
    Installed,
    Refreshed,
    AlreadyCurrent,
    SkippedForeign,
}

/// Install or refresh our status line. Never overwrites a foreign custom status line.
pub fn sync_owned() -> Result<SyncOutcome> {
    let path = claude_settings_path()?;
    let mut settings: Value = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s)
            .with_context(|| format!("{} is not valid JSON", path.display()))?,
        Err(_) => Value::Object(Default::default()),
    };
    match owned_status_of(&settings) {
        OwnedStatus::Foreign => Ok(SyncOutcome::SkippedForeign),
        OwnedStatus::Current => Ok(SyncOutcome::AlreadyCurrent),
        OwnedStatus::Missing => {
            set_statusline(&mut settings, &path)?;
            write_settings(&path, &settings)?;
            Ok(SyncOutcome::Installed)
        }
        OwnedStatus::Stale => {
            // Replace only our owned fields (command + refreshInterval); keep padding etc.
            refresh_statusline_config(&mut settings)?;
            write_settings(&path, &settings)?;
            Ok(SyncOutcome::Refreshed)
        }
    }
}

fn write_settings(path: &Path, settings: &Value) -> Result<()> {
    atomic_write_json(path, settings)
}

/// Re-point an existing llmtrim-owned Claude statusline at the current binary and ensure
/// `refreshInterval` is set. A custom command is left alone. Returns whether anything changed.
fn refresh_statusline_config(settings: &mut Value) -> Result<bool> {
    let Some(status_line) = settings
        .get_mut("statusLine")
        .and_then(Value::as_object_mut)
    else {
        return Ok(false);
    };
    let Some(command) = status_line.get("command").and_then(Value::as_str) else {
        return Ok(false);
    };
    if !is_llmtrim_statusline_command(command) {
        return Ok(false);
    }

    let desired = statusline_config();
    let desired_cmd = desired
        .get("command")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_default();
    let desired_refresh = desired
        .get("refreshInterval")
        .cloned()
        .unwrap_or(Value::from(REFRESH_INTERVAL_SECS));

    let mut changed = false;
    if status_line.get("command").and_then(Value::as_str) != Some(desired_cmd.as_str()) {
        status_line.insert("command".to_string(), Value::String(desired_cmd));
        changed = true;
    }
    if status_line.get("refreshInterval") != Some(&desired_refresh) {
        status_line.insert("refreshInterval".to_string(), desired_refresh);
        changed = true;
    }
    Ok(changed)
}

/// Refresh an existing llmtrim-owned Claude statusline without touching custom commands.
/// Returns whether the settings file was rewritten. This runs during `llmtrim update` before
/// package managers can remove a versioned executable path. Does **not** install when missing.
pub fn refresh_if_installed() -> Result<bool> {
    match owned_status() {
        OwnedStatus::Missing | OwnedStatus::Foreign | OwnedStatus::Current => Ok(false),
        OwnedStatus::Stale => {
            let path = claude_settings_path()?;
            let Ok(s) = std::fs::read_to_string(&path) else {
                return Ok(false);
            };
            let mut settings: Value = serde_json::from_str(&s)
                .with_context(|| format!("{} is not valid JSON", path.display()))?;
            if !refresh_statusline_config(&mut settings)? {
                return Ok(false);
            }
            write_settings(&path, &settings)?;
            Ok(true)
        }
    }
}

/// Remove the `statusLine` key we wrote (leaves the rest of `settings.json` untouched).
pub fn uninstall() -> Result<()> {
    let path = claude_settings_path()?;
    let Ok(s) = std::fs::read_to_string(&path) else {
        println!("No {} to edit — nothing to remove.", path.display());
        return Ok(());
    };
    let mut settings: Value = serde_json::from_str(&s)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    if clear_statusline(&mut settings, &path)? {
        std::fs::write(&path, serde_json::to_string_pretty(&settings)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("Removed the llmtrim status line from {}.", path.display());
    } else {
        println!("No llmtrim status line found in {}.", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn led(health: Health) -> Led {
        Led {
            health,
            trim_pct: Some(6.8),
            reroute: Some("codex".to_string()),
            reroute_window: None,
            resolved_model: Some("gpt-5.6-terra".to_string()),
            cache_cold: false,
            last_cache_pct: None,
            last_ctx_tokens: 0,
        }
    }

    fn cc(ctx: i64) -> CcInput {
        CcInput {
            model: "Opus".to_string(),
            effort: Some("high".to_string()),
            session_id: None,
            model_id: "claude-opus-4-8".to_string(),
            ctx_tokens: ctx,
            window: Some(200_000),
            five_hour_pct: Some(24.0),
            seven_day_pct: Some(12.0),
            five_hour_resets_at: None,
            seven_day_resets_at: None,
            cache_pct: Some(63.0),
        }
    }

    #[test]
    fn full_line_has_every_segment_when_wide() {
        // 142k of a 200k window = 71% ⇒ red gauge; both rolling quota windows shown.
        let out = render_line(&cc(142_000), &led(Health::Healthy), 0, false);
        assert_eq!(
            out,
            "◆ Opus→gpt-5.6-terra   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24% · 7d·12%   ♻ 63% cached"
        );
    }

    #[test]
    fn context_gauge_colors_by_percent_of_real_window() {
        // Bands are % of the model's real window. On a 1M window, 142k is only 14% ⇒ green.
        let mut c = cc(142_000);
        c.window = Some(1_000_000);
        let out = context_segment(c.ctx_tokens, 1_000_000, false, true);
        assert!(out.contains(GREEN), "14% of 1M is green: {out}");
        // Same 142k on a 200k window is 71% ⇒ red.
        let out = context_segment(142_000, 200_000, false, true);
        assert!(out.contains(RED), "71% of 200k is red: {out}");
        // 50% lands in the orange band (40–65%).
        let out = context_segment(100_000, 200_000, false, true);
        assert!(out.contains(ORANGE), "50% is orange: {out}");
    }

    #[test]
    fn context_gauge_pins_full_over_window() {
        let out = render_line(&cc(210_000), &led(Health::Healthy), 0, false);
        assert!(
            out.contains("▓▓▓▓▓▓▓▓ 210k"),
            "over window pins full: {out}"
        );
    }

    #[test]
    fn cold_cache_forces_red_gauge_regardless_of_fill() {
        // A near-empty context still reddens the gauge when the cache is cold.
        let mut l = led(Health::Healthy);
        l.cache_cold = true;
        let out = context_segment(1_000, 1_000_000, true, true);
        assert!(out.contains(RED), "cold cache reddens even at 0.1%: {out}");
    }

    #[test]
    fn reroute_arrow_hidden_when_not_healthy() {
        let out = render_line(&cc(142_000), &led(Health::Degraded), 0, false);
        assert!(!out.contains("→codex"), "no arrow when degraded: {out}");
        assert!(
            out.contains("⚠ llmtrim degraded"),
            "warns instead of ✂: {out}"
        );
    }

    #[test]
    fn stopped_omits_trim_and_arrow_without_warning() {
        let mut l = led(Health::Stopped);
        l.reroute = None;
        l.resolved_model = None;
        let out = render_line(&cc(48_000), &l, 0, false);
        assert!(!out.contains('✂'), "no trim when off: {out}");
        assert!(!out.contains('⚠'), "clean off is not an error: {out}");
        assert!(out.starts_with("◆ Opus"), "model still shown: {out}");
    }

    #[test]
    fn narrow_terminal_sheds_extras_right_to_left() {
        // Wide enough for core + quota, but not the cache extra.
        let full = render_line(&cc(142_000), &led(Health::Healthy), 0, false);
        let width =
            ui::visible_width("◆ Opus→gpt-5.6-terra   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24% · 7d·12%");
        let out = render_line(&cc(142_000), &led(Health::Healthy), width, false);
        assert!(out.ends_with("7d·12%"), "keeps quota, sheds cache: {out}");
        assert!(!out.contains("cached"), "cache dropped first: {out}");
        assert!(full.len() > out.len());
    }

    #[test]
    fn absent_data_segments_do_not_render() {
        let mut c = cc(48_000);
        c.effort = None; // non-reasoning model
        c.five_hour_pct = None; // API-key user
        c.seven_day_pct = None;
        c.cache_pct = None; // before first API response
        let mut l = led(Health::Healthy);
        l.reroute = None; // no reroute
        l.resolved_model = None;
        let out = render_line(&c, &l, 0, false);
        assert_eq!(out, "◆ Opus   ▓░░░░░░░ 48k   ✂ 6.8%");
    }

    #[test]
    fn cold_cache_shows_compact_hint_not_stale_pct() {
        let mut l = led(Health::Healthy);
        l.cache_cold = true;
        let out = render_line(&cc(48_000), &l, 0, false);
        assert!(out.contains("♻ cache cold"), "cold hint shown: {out}");
        assert!(!out.contains("cached"), "no stale % when cold: {out}");
    }

    #[test]
    fn post_compact_reset_clears_cold_even_when_ledger_stale() {
        let mut c = cc(0);
        c.cache_pct = None;
        assert!(
            !effective_cache_cold(&c, true),
            "empty context after compact is not cold"
        );
        let mut l = led(Health::Healthy);
        l.cache_cold = effective_cache_cold(&c, true);
        let out = render_line(&c, &l, 0, false);
        assert!(!out.contains("cold"), "no cold nudge after compact: {out}");
        let gauge = context_segment(c.ctx_tokens, 200_000, l.cache_cold, true);
        assert!(
            gauge.contains(DIM),
            "gauge dim (not red) on fresh post-compact context: {gauge}"
        );
        assert!(
            !gauge.contains(RED),
            "gauge not red on fresh post-compact context: {gauge}"
        );
    }

    #[test]
    fn idle_cold_stays_when_context_still_populated() {
        let mut c = cc(142_000);
        c.cache_pct = Some(5.0);
        assert!(
            effective_cache_cold(&c, true),
            "populated context with stale ledger still shows cold"
        );
    }

    #[test]
    fn quota_labels_show_time_remaining_until_reset() {
        let mut c = cc(48_000);
        c.five_hour_resets_at = Some(
            (chrono::Utc::now() + chrono::Duration::hours(3) + chrono::Duration::minutes(30))
                .timestamp(),
        );
        c.seven_day_resets_at = Some(
            (chrono::Utc::now() + chrono::Duration::days(4) + chrono::Duration::hours(12))
                .timestamp(),
        );
        let out = render_line(&c, &led(Health::Healthy), 0, false);
        assert!(out.contains("◔ 3h·24% · 4d·12%"), "remaining labels: {out}");
    }

    #[test]
    fn quota_color_escalates_green_amber_orange_red() {
        assert_eq!(quota_color(50.0), GREEN);
        assert_eq!(quota_color(75.0), AMBER);
        assert_eq!(quota_color(85.0), ORANGE); // the added 80%+ step
        assert_eq!(quota_color(95.0), RED);
    }

    #[test]
    fn quota_windows_colour_independently() {
        // A maxed 5h next to a comfortable 7d: 5h red, 7d green — not both painted by the worse.
        let mut c = cc(48_000);
        c.five_hour_pct = Some(95.0);
        c.seven_day_pct = Some(12.0);
        let segs = extra_segments(&c, &led(Health::Healthy), true);
        let quota = segs
            .iter()
            .find(|s| s.contains("5h"))
            .expect("quota segment");
        // Only the percentage is coloured; the `5h`/`7d` labels stay dim.
        assert!(
            quota.contains(&format!("\x1b[{DIM}m5h·")),
            "5h label dim: {quota}"
        );
        assert!(
            quota.contains(&format!("\x1b[{RED}m95%")),
            "5h pct red: {quota}"
        );
        assert!(
            quota.contains(&format!("\x1b[{DIM}m7d·")),
            "7d label dim: {quota}"
        );
        assert!(
            quota.contains(&format!("\x1b[{GREEN}m12%")),
            "7d pct green: {quota}"
        );
    }

    #[test]
    fn trim_is_not_coloured_by_tier() {
        // The savings figure is dim, not green — colour is for state signals only.
        let seg = trim_or_health_segment(&led(Health::Healthy), true).unwrap();
        assert!(
            seg.contains(&format!("\x1b[{DIM}m✂ 6.8%")),
            "trim dim: {seg}"
        );
        assert!(
            !seg.contains(&format!("\x1b[{GREEN}m✂")),
            "not green: {seg}"
        );
    }

    #[test]
    fn quota_shows_worse_window_and_folds_to_one_segment() {
        // Both windows in one `◔` segment; only one present ⇒ just that one.
        let out = render_line(&cc(48_000), &led(Health::Healthy), 0, false);
        assert!(out.contains("◔ 5h·24% · 7d·12%"), "both windows: {out}");
        let mut c = cc(48_000);
        c.seven_day_pct = None;
        let out = render_line(&c, &led(Health::Healthy), 0, false);
        assert!(
            out.contains("◔ 5h·24%") && !out.contains("7d"),
            "5h only: {out}"
        );
    }

    #[test]
    fn kimi_reroute_when_healthy() {
        let tiers = std::collections::BTreeMap::new();
        let mut l = led(Health::Healthy);
        l.reroute = Some("kimi".to_string());
        l.resolved_model = reroute_resolved_model("kimi", "claude-opus-4-6", &tiers);
        assert!(
            l.resolved_model.is_none(),
            "kimi collapses to one wire id; the arrow shows the shortname"
        );
        let out = render_line(&cc(72_000), &l, 0, true);
        assert!(out.contains("→kimi"), "kimi arrow present: {out}");
        assert!(
            !out.contains("kimi-for-coding"),
            "no internal wire id on the line: {out}"
        );
    }

    fn ledger_row(sub: Option<&str>, model: Option<&str>) -> SessionLedgerRow {
        SessionLedgerRow {
            input_before: 1000,
            input_after: 940,
            last_ts: chrono::Utc::now().to_rfc3339(),
            last_sub_provider: sub.map(str::to_string),
            last_model: model.map(str::to_string),
            last_cache_hit: None,
            last_input_tokens: 0,
        }
    }

    #[test]
    fn fallback_mode_shows_no_arrow_until_a_backend_actually_serves() {
        // Anthropic served the turn: the chain is armed but never fired, so claiming a reroute
        // would be a lie — and would rescale the gauge to Codex's window.
        let row = ledger_row(None, Some("claude-opus-4-8"));
        assert!(matches!(
            arrow_source(Some("codex"), true, Some(&row)),
            ArrowSource::None
        ));
    }

    #[test]
    fn fallback_mode_shows_the_backend_once_the_chain_fires() {
        let row = ledger_row(Some("codex"), Some("gpt-5.6-terra"));
        let ArrowSource::Served { provider, model } = arrow_source(Some("codex"), true, Some(&row))
        else {
            panic!("a fired fallback must surface the backend that served it");
        };
        assert_eq!(provider, "codex");
        assert_eq!(model.as_deref(), Some("gpt-5.6-terra"));
    }

    #[test]
    fn fallback_mode_never_predicts_from_config_on_a_fresh_session() {
        // No turn yet. `always` mode could predict; fallback mode cannot — Anthropic serves the
        // first turn and the chain only fires if it fails.
        assert!(matches!(
            arrow_source(Some("codex"), true, None),
            ArrowSource::None
        ));
        assert!(matches!(
            arrow_source(Some("codex"), false, None),
            ArrowSource::Predicted(p) if p == "codex"
        ));
    }

    #[test]
    fn ledger_truth_overrides_a_stale_config_prediction() {
        // `sub` says kimi, but the last turn was actually served by codex — show what happened.
        let row = ledger_row(Some("codex"), Some("gpt-5.6-terra"));
        let ArrowSource::Served { provider, .. } = arrow_source(Some("kimi"), false, Some(&row))
        else {
            panic!("the ledger outranks config");
        };
        assert_eq!(provider, "codex");
    }

    #[test]
    fn trim_for_does_not_flash_lifetime_on_a_fresh_session() {
        let lifetime = || Some(6.8);
        // A known session with no recorded turn yet ⇒ idle (None), NOT the lifetime figure —
        // the "6.8% then flips to the real number" flash we're fixing.
        assert_eq!(trim_for(Some("sess-uuid"), None, lifetime), None);
        // Its own rows ⇒ the per-session figure (300→200 = 33.3%).
        let own = trim_for(Some("sess-uuid"), Some((300, 200)), lifetime).unwrap();
        assert!((own - 33.333).abs() < 0.01, "per-session figure: {own}");
        // No session id at all (non-Claude-Code client) ⇒ lifetime is the only honest figure.
        assert_eq!(trim_for(None, None, lifetime), Some(6.8));
    }

    #[test]
    fn context_gauge_holds_its_fill_while_a_turn_is_in_flight() {
        // `total_input_tokens` comes from the last *response*, so mid-request the blob reports 0
        // and the bar empties — the flicker the cache segment had. The last completed turn's
        // input, measured on the wire, holds it.
        let mut mid_turn = cc(48_000);
        mid_turn.ctx_tokens = 0;
        let mut l = led(Health::Healthy);
        l.last_ctx_tokens = 48_000;
        assert_eq!(
            render_line(&mid_turn, &l, 0, false),
            render_line(&cc(48_000), &l, 0, false),
            "mid-turn gauge matches the completed-turn gauge"
        );
        // The blob wins whenever it has a figure; a truly fresh session (no turn, no row) still
        // renders empty rather than inventing a fill.
        assert_eq!(ctx_tokens_for(142_000, 48_000), 142_000);
        assert_eq!(ctx_tokens_for(0, 0), 0);
    }

    #[test]
    fn cache_segment_holds_its_value_while_a_turn_is_in_flight() {
        // Claude Code drops `current_usage` from the blob mid-request, so `cc.cache_pct` is None
        // on every re-render during a turn. Without a fallback the segment vanishes and pops back
        // when the turn lands — a flicker that reads like the cache collapsing.
        let mut mid_turn = cc(48_000);
        mid_turn.cache_pct = None;
        let mut l = led(Health::Healthy);
        l.last_cache_pct = Some(98.0);
        let out = render_line(&mid_turn, &l, 0, false);
        assert!(out.contains("♻ 98% cached"), "last turn stands in: {out}");

        // The blob still wins the moment it has a figure — the ledger is one turn stale.
        let out = render_line(&cc(48_000), &l, 0, false);
        assert!(
            out.contains("♻ 63% cached"),
            "blob wins when present: {out}"
        );

        // Nothing anywhere (fresh session, no turn yet) ⇒ no segment, not a fake 0%.
        assert_eq!(cache_pct_for(None, None), None);
    }

    #[test]
    fn healthy_but_idle_shows_dim_marker_not_zero() {
        // Nothing trimmed yet in this session: `✂ –`, never a misleading `✂ 0.0%`.
        let mut l = led(Health::Healthy);
        l.trim_pct = None;
        let out = render_line(&cc(48_000), &l, 0, false);
        assert!(out.contains("✂ –"), "idle marker shown: {out}");
        assert!(!out.contains("0.0%"), "no fake zero: {out}");
    }

    #[test]
    fn quota_and_cache_floor_not_round() {
        // 99.9% cache is not 100%; 89.9% quota is not 90%.
        let mut c = cc(48_000);
        c.five_hour_pct = Some(89.9);
        c.cache_pct = Some(99.9);
        let out = render_line(&c, &led(Health::Healthy), 0, false);
        assert!(out.contains("5h·89%"), "quota floored: {out}");
        assert!(out.contains("♻ 99% cached"), "cache floored: {out}");
    }

    #[test]
    fn parse_cc_reads_session_id() {
        let cc = parse_cc(r#"{"model":{"display_name":"Opus"},"session_id":"abc-123"}"#);
        assert_eq!(cc.session_id.as_deref(), Some("abc-123"));
        // Empty id is treated as absent (falls back to lifetime trim).
        let cc = parse_cc(r#"{"model":{"display_name":"Opus"},"session_id":""}"#);
        assert!(cc.session_id.is_none());
    }

    #[test]
    fn parse_cc_reads_nested_fields() {
        let blob = r#"{"model":{"display_name":"Sonnet","id":"claude-sonnet-5"},"effort":{"level":"medium"},
            "context_window":{"total_input_tokens":123456,"context_window_size":1000000,
              "current_usage":{"input_tokens":10,"cache_creation_input_tokens":10,"cache_read_input_tokens":80}},
            "rate_limits":{"five_hour":{"used_percentage":41.2,"resets_at":1738425600},
              "seven_day":{"used_percentage":9.0,"resets_at":1738857600}}}"#;
        let cc = parse_cc(blob);
        assert_eq!(cc.model, "Sonnet");
        assert_eq!(cc.model_id, "claude-sonnet-5");
        assert_eq!(cc.effort.as_deref(), Some("medium"));
        assert_eq!(cc.ctx_tokens, 123456);
        assert_eq!(cc.window, Some(1_000_000));
        assert_eq!(cc.five_hour_pct, Some(41.2));
        assert_eq!(cc.seven_day_pct, Some(9.0));
        // Claude Code sends `resets_at` as Unix epoch seconds (an integer), not an RFC3339 string.
        assert_eq!(cc.five_hour_resets_at, Some(1_738_425_600));
        assert_eq!(cc.seven_day_resets_at, Some(1_738_857_600));
        // 80 cache reads of 100 total input = 80%.
        assert_eq!(cc.cache_pct, Some(80.0));
    }

    #[test]
    fn install_merge_preserves_unrelated_keys() {
        let p = std::path::Path::new("settings.json");
        let mut settings = serde_json::json!({
            "theme": "dark",
            "permissions": { "allow": ["Bash"] },
        });
        set_statusline(&mut settings, p).unwrap();
        // Our key is present with a command...
        assert_eq!(settings["statusLine"]["type"], "command");
        // ...and the pre-existing keys are untouched.
        assert_eq!(settings["theme"], "dark");
        assert_eq!(settings["permissions"]["allow"][0], "Bash");
    }

    #[cfg(unix)]
    #[test]
    fn statusline_prefers_stable_path_alias_for_active_binary() {
        let root = std::env::temp_dir().join(format!(
            "llmtrim-statusline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let versioned_dir = root.join("Cellar/llmtrim/0.9.5/bin");
        let stable_dir = root.join("bin");
        std::fs::create_dir_all(&versioned_dir).unwrap();
        std::fs::create_dir_all(&stable_dir).unwrap();

        let name = "llmtrim";
        let current = versioned_dir.join(name);
        let stable = stable_dir.join(name);
        std::fs::write(&current, b"binary").unwrap();
        std::os::unix::fs::symlink(&current, &stable).unwrap();
        let path = std::env::join_paths([&stable_dir]).unwrap();

        assert_eq!(
            stable_executable_path(&current, Some(path.as_os_str())),
            stable
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recognizes_only_llmtrim_statusline_commands_for_refresh() {
        assert!(is_llmtrim_statusline_command(
            "/opt/homebrew/Cellar/llmtrim/0.9.4/bin/llmtrim statusline"
        ));
        assert!(is_llmtrim_statusline_command(
            r#""C:\Program Files\llmtrim\llmtrim.exe" statusline"#
        ));
        assert!(!is_llmtrim_statusline_command("my-statusline-command"));
        assert!(!is_llmtrim_statusline_command(
            "llmtrim statusline --custom"
        ));
    }

    #[test]
    fn refreshes_owned_statusline_without_touching_other_settings() {
        let mut settings = serde_json::json!({
            "theme": "dark",
            "statusLine": {
                "type": "command",
                "command": "/opt/homebrew/Cellar/llmtrim/0.9.4/bin/llmtrim statusline",
                "padding": 2
            }
        });

        assert!(refresh_statusline_config(&mut settings).unwrap());
        assert_eq!(settings["theme"], "dark");
        assert_ne!(
            settings["statusLine"]["command"],
            "/opt/homebrew/Cellar/llmtrim/0.9.4/bin/llmtrim statusline"
        );
        // Only `command` moves: a padding the user chose survives the update.
        assert_eq!(settings["statusLine"]["padding"], 2);
        assert_eq!(settings["statusLine"]["type"], "command");

        let mut custom = serde_json::json!({
            "statusLine": { "type": "command", "command": "my-statusline-command" }
        });
        assert!(!refresh_statusline_config(&mut custom).unwrap());
        assert_eq!(custom["statusLine"]["command"], "my-statusline-command");
    }

    #[test]
    fn refresh_is_a_no_op_when_the_command_already_points_at_this_binary() {
        let mut settings = serde_json::json!({ "statusLine": statusline_config() });
        assert!(!refresh_statusline_config(&mut settings).unwrap());
    }

    #[test]
    fn statusline_config_asks_claude_code_to_refresh_while_idle() {
        // Claude Code otherwise re-renders only on conversation events, so the cold-cache
        // warning would land after the expensive turn instead of before it.
        assert_eq!(
            statusline_config()["refreshInterval"],
            serde_json::json!(300)
        );
    }

    #[test]
    fn uninstall_removes_only_our_key_and_reports_presence() {
        let p = std::path::Path::new("settings.json");
        let mut settings = serde_json::json!({ "theme": "dark", "statusLine": { "x": 1 } });
        assert!(
            clear_statusline(&mut settings, p).unwrap(),
            "reports it was present"
        );
        assert!(settings.get("statusLine").is_none(), "our key gone");
        assert_eq!(settings["theme"], "dark", "other keys kept");
        // Second removal reports absence and is a no-op.
        assert!(!clear_statusline(&mut settings, p).unwrap());
    }

    #[test]
    fn merge_rejects_a_non_object_settings_file() {
        let p = std::path::Path::new("settings.json");
        let mut settings = serde_json::json!([1, 2, 3]);
        assert!(set_statusline(&mut settings, p).is_err());
        assert!(clear_statusline(&mut settings, p).is_err());
    }

    #[test]
    fn cache_pct_absent_without_current_usage() {
        let cc = parse_cc(
            r#"{"model":{"display_name":"Opus"},"context_window":{"total_input_tokens":100}}"#,
        );
        assert!(
            cc.cache_pct.is_none(),
            "no current_usage ⇒ no cache segment"
        );
    }

    #[test]
    fn parse_cc_tolerates_garbage_and_missing_fields() {
        let cc = parse_cc("not json");
        assert_eq!(cc.model, "?");
        assert_eq!(cc.ctx_tokens, 0);
        assert!(cc.effort.is_none());
    }

    #[test]
    fn context_gauge_handles_extreme_token_counts() {
        // Negative clamps to empty/0k; a pathologically huge count pins full without
        // overflowing i64 (regression: `tokens * width` used to overflow before clamping).
        assert!(context_segment(-5000, 200_000, false, false).starts_with("░░░░░░░░ 0k"));
        assert!(context_segment(i64::MAX / 4, 200_000, false, false).starts_with("▓▓▓▓▓▓▓▓"));
    }

    #[test]
    fn cache_cold_parses_timestamps() {
        assert!(!cache_cold("garbage"), "unparseable ⇒ not cold");
        let fresh = chrono::Utc::now().to_rfc3339();
        assert!(!cache_cold(&fresh), "just now ⇒ warm");
        let stale =
            (chrono::Utc::now() - chrono::Duration::seconds(CACHE_TTL_SECS + 60)).to_rfc3339();
        assert!(cache_cold(&stale), "past the TTL ⇒ cold");
    }

    #[test]
    fn effective_window_prefers_reroute_then_blob() {
        let mut l = led(Health::Healthy);
        // Reroute window (backend's real window) wins over the blob's Claude window.
        l.reroute_window = Some(262_144);
        assert_eq!(effective_window(&cc(0), &l), 262_144);
        // No reroute ⇒ the blob's context_window_size.
        l.reroute_window = None;
        assert_eq!(effective_window(&cc(0), &l), 200_000);
        // Neither ⇒ the default floor.
        let mut c = cc(0);
        c.window = None;
        assert_eq!(effective_window(&c, &l), CTX_WINDOW_DEFAULT);
    }
}
