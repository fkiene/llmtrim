//! `statusline` — one elegant line for Claude Code's custom status line.
//!
//! Claude Code pipes a JSON session blob on stdin and renders whatever the command
//! prints (see <https://code.claude.com/docs/en/statusline>). This module reads that
//! blob, folds in llmtrim's own live signals from the ledger + config (compression
//! saved, interceptor health, the active `sub` reroute), and prints a single
//! width-adaptive line:
//!
//! ```text
//! ◆ Opus·high→codex   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24% · 7d·12%   ♻ 63% cached
//! ```
//!
//! The three left segments (model·effort→backend, context, ✂ trim) are core and never
//! truncate; the extras (5h/7d quota, then this turn's prompt-cache reuse) shed right-to-left
//! as the terminal narrows (`COLUMNS`). The context gauge fills and colours against the *real*
//! window of the model serving the turn — the rerouted backend's window under `sub`, not
//! Claude's — green below 40%, orange 40–65%, red above; and red whenever the prompt cache has
//! gone cold, where the cache segment becomes `♻ cold · /compact`. Segments whose data is
//! absent — no reroute, an API-key user with no rate limits, a non-reasoning model with no
//! effort — simply don't render.
//!
//! `install` wires it into `~/.claude/settings.json`; rendering itself never touches the
//! network or API tokens.

use std::path::PathBuf;

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
const CACHE_TTL_SECS: i64 = 3600;

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
    /// The prompt cache has gone cold: the session has been idle past the TTL, so the next turn
    /// pays a cold write. Renders the cache segment red with a `/compact` nudge.
    cache_cold: bool,
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
    let reroute = cfg
        .sub
        .clone()
        .filter(|s| !s.is_empty() && s != "off");
    let reroute_window = reroute
        .as_deref()
        .and_then(|p| reroute_real_window(p, &cc.model_id, &cfg.sub_tiers));
    let health = proxy_health();

    // One session-row read serves both trim and cache-cold.
    let row = cc.session_id.as_deref().and_then(|sid| {
        crate::breakdown::db::BreakdownDb::open()
            .ok()
            .and_then(|db| db.sessions().ok())
            .and_then(|rows| rows.into_iter().find(|r| r.session_id == sid))
    });
    let trim_pct = match &row {
        Some(r) => (r.input_before > 0)
            .then(|| ui::saved_pct(r.input_before as f64, r.input_after as f64)),
        None => Tracker::open()
            .and_then(|t| t.summary())
            .ok()
            .filter(|s| s.input_before > 0)
            .map(|s| ui::saved_pct(s.input_before as f64, s.input_after as f64)),
    };
    let cache_cold = row.as_ref().is_some_and(|r| cache_cold(&r.last_ts));

    Led {
        health,
        trim_pct,
        reroute,
        reroute_window,
        cache_cold,
    }
}

/// The real context window of the model a `sub` reroute actually serves this turn, from the
/// model registry. Claude Code reports its own Claude window on the wire, so under reroute we
/// must resolve the backend model id and look *it* up. `None` if unknown (gauge keeps the blob
/// window). Kimi's internal id isn't a registry key, so it's mapped to its public model id.
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
    let resolved = crate::reroute::resolve_model(sp, incoming_model_id, tiers);
    // `kimi-for-coding` is an internal routing id, not a models.dev key — map to the public one.
    let lookup = if resolved == crate::reroute::KIMI_MODEL {
        "moonshotai/kimi-k2"
    } else {
        resolved.as_str()
    };
    llmtrim_core::context_window(lookup).map(|w| w as i64)
}

/// Whether the prompt cache has gone cold: `last_ts` (rfc3339, the most recent intercepted
/// request) is older than the TTL. An unparseable timestamp is treated as not-cold (don't warn
/// on a data glitch).
fn cache_cold(last_ts: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(last_ts)
        .map(|t| chrono::Utc::now().signed_duration_since(t).num_seconds() >= CACHE_TTL_SECS)
        .unwrap_or(false)
}

// ── rendering ─────────────────────────────────────────────────────────────────────

/// Colour a value by threshold (`< warn` first colour, `< bad` second, else third).
fn tier_color(v: f64, warn: f64, bad: f64) -> &'static str {
    if v >= bad {
        RED
    } else if v >= warn {
        AMBER
    } else {
        GREEN
    }
}

/// `◆ Opus·high→codex` — health-brand glyph, model, glued effort, reroute arrow. The arrow is
/// suppressed when the proxy isn't healthy (traffic isn't being intercepted, so it isn't
/// actually rerouting).
fn model_segment(cc: &CcInput, led: &Led, color: bool) -> String {
    let mut s = format!(
        "{} {}",
        paint(color, BRAND, "◆"),
        paint(color, BOLD, &cc.model)
    );
    if let Some(effort) = &cc.effort {
        s.push_str(&paint(color, DIM, &format!("·{effort}")));
    }
    if let (Health::Healthy, Some(p)) = (led.health, &led.reroute) {
        let code = match p.as_str() {
            "kimi" => VIOLET,
            _ => CYAN,
        };
        s.push_str(&paint(color, code, &format!("→{p}")));
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
    // One quota segment carrying both rolling windows: `◔ 5h·15% · 7d·12%`. `◔` = a window
    // filling up; `·` keeps `5h`/`7d` from reading as durations. Each window is coloured on its
    // own value, so a maxed 5h doesn't paint a comfortable 7d red (and vice-versa).
    let quota = |label: &str, p: f64| {
        paint(color, tier_color(p, 70.0, 90.0), &format!("{label}·{}%", p.floor() as i64))
    };
    let glyph = paint(color, DIM, "◔");
    let sep = paint(color, DIM, "·");
    match (cc.five_hour_pct, cc.seven_day_pct) {
        (Some(h), Some(d)) => {
            out.push(format!("{glyph} {} {sep} {}", quota("5h", h), quota("7d", d)));
        }
        (Some(h), None) => out.push(format!("{glyph} {}", quota("5h", h))),
        (None, Some(d)) => out.push(format!("{glyph} {}", quota("7d", d))),
        (None, None) => {}
    }
    if led.cache_cold {
        // Cache expired: the next turn pays a cold write, so `/compact` (re-baselines the prompt)
        // pays off here. `cold` communicates the state faster than a stale `0% cached`.
        out.push(paint(color, RED, "♻ cold · /compact"));
    } else if let Some(c) = cc.cache_pct {
        // Floor, not round: only a genuine 100% cache shows `100%` (99.9 stays `99%`).
        out.push(paint(color, DIM, &format!("♻ {}% cached", c.floor() as i64)));
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
        context_segment(cc.ctx_tokens, effective_window(cc, led), led.cache_cold, color),
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

fn claude_settings_path() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("neither HOME nor USERPROFILE is set")?;
    Ok(PathBuf::from(home).join(".claude").join("settings.json"))
}

/// The `statusLine` object we write. `command` is this binary's absolute path plus the
/// subcommand, so it works regardless of PATH.
fn statusline_config() -> Value {
    let exe = std::env::current_exe()
        .ok()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "llmtrim".to_string());
    let command = if exe.contains(' ') {
        format!("\"{exe}\" statusline")
    } else {
        format!("{exe} statusline")
    };
    serde_json::json!({ "type": "command", "command": command, "padding": 0 })
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
    std::fs::write(&path, serde_json::to_string_pretty(&settings)?)
        .with_context(|| format!("failed to write {}", path.display()))?;

    println!(
        "Wired the llmtrim status line into {}. Restart Claude Code to see it.",
        path.display()
    );
    Ok(())
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
            cache_cold: false,
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
            cache_pct: Some(63.0),
        }
    }

    #[test]
    fn full_line_has_every_segment_when_wide() {
        // 142k of a 200k window = 71% ⇒ red gauge; both rolling quota windows shown.
        let out = render_line(&cc(142_000), &led(Health::Healthy), 0, false);
        assert_eq!(
            out,
            "◆ Opus·high→codex   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24% · 7d·12%   ♻ 63% cached"
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
        assert!(out.contains("▓▓▓▓▓▓▓▓ 210k"), "over window pins full: {out}");
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
        let out = render_line(&cc(48_000), &l, 0, false);
        assert!(!out.contains('✂'), "no trim when off: {out}");
        assert!(!out.contains('⚠'), "clean off is not an error: {out}");
        assert!(out.starts_with("◆ Opus·high"), "model still shown: {out}");
    }

    #[test]
    fn narrow_terminal_sheds_extras_right_to_left() {
        // Wide enough for core + quota, but not the cache extra.
        let full = render_line(&cc(142_000), &led(Health::Healthy), 0, false);
        let width =
            ui::visible_width("◆ Opus·high→codex   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24% · 7d·12%");
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
        let out = render_line(&c, &l, 0, false);
        assert_eq!(out, "◆ Opus   ▓░░░░░░░ 48k   ✂ 6.8%");
    }

    #[test]
    fn cold_cache_shows_compact_hint_not_stale_pct() {
        let mut l = led(Health::Healthy);
        l.cache_cold = true;
        let out = render_line(&cc(48_000), &l, 0, false);
        assert!(out.contains("♻ cold · /compact"), "cold hint shown: {out}");
        assert!(!out.contains("cached"), "no stale % when cold: {out}");
    }

    #[test]
    fn quota_windows_colour_independently() {
        // A maxed 5h next to a comfortable 7d: 5h red, 7d green — not both painted by the worse.
        let mut c = cc(48_000);
        c.five_hour_pct = Some(95.0);
        c.seven_day_pct = Some(12.0);
        let segs = extra_segments(&c, &led(Health::Healthy), true);
        let quota = segs.iter().find(|s| s.contains("5h")).expect("quota segment");
        assert!(quota.contains(&format!("\x1b[{RED}m5h·95%")), "5h red: {quota}");
        assert!(quota.contains(&format!("\x1b[{GREEN}m7d·12%")), "7d green: {quota}");
    }

    #[test]
    fn trim_is_not_coloured_by_tier() {
        // The savings figure is dim, not green — colour is for state signals only.
        let seg = trim_or_health_segment(&led(Health::Healthy), true).unwrap();
        assert!(seg.contains(&format!("\x1b[{DIM}m✂ 6.8%")), "trim dim: {seg}");
        assert!(!seg.contains(&format!("\x1b[{GREEN}m✂")), "not green: {seg}");
    }

    #[test]
    fn quota_shows_worse_window_and_folds_to_one_segment() {
        // Both windows in one `◔` segment; only one present ⇒ just that one.
        let out = render_line(&cc(48_000), &led(Health::Healthy), 0, false);
        assert!(out.contains("◔ 5h·24% · 7d·12%"), "both windows: {out}");
        let mut c = cc(48_000);
        c.seven_day_pct = None;
        let out = render_line(&c, &led(Health::Healthy), 0, false);
        assert!(out.contains("◔ 5h·24%") && !out.contains("7d"), "5h only: {out}");
    }

    #[test]
    fn kimi_reroute_when_healthy() {
        let mut l = led(Health::Healthy);
        l.reroute = Some("kimi".to_string());
        let out = render_line(&cc(72_000), &l, 0, true);
        assert!(out.contains("→kimi"), "kimi arrow present: {out}");
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
            "rate_limits":{"five_hour":{"used_percentage":41.2},"seven_day":{"used_percentage":9.0}}}"#;
        let cc = parse_cc(blob);
        assert_eq!(cc.model, "Sonnet");
        assert_eq!(cc.model_id, "claude-sonnet-5");
        assert_eq!(cc.effort.as_deref(), Some("medium"));
        assert_eq!(cc.ctx_tokens, 123456);
        assert_eq!(cc.window, Some(1_000_000));
        assert_eq!(cc.five_hour_pct, Some(41.2));
        assert_eq!(cc.seven_day_pct, Some(9.0));
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
        let stale = (chrono::Utc::now() - chrono::Duration::seconds(CACHE_TTL_SECS + 60)).to_rfc3339();
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
