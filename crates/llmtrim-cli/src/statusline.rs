//! `statusline` — one elegant line for Claude Code's custom status line.
//!
//! Claude Code pipes a JSON session blob on stdin and renders whatever the command
//! prints (see <https://code.claude.com/docs/en/statusline>). This module reads that
//! blob, folds in llmtrim's own live signals from the ledger + config (compression
//! saved, interceptor health, the active `sub` reroute), and prints a single
//! width-adaptive line:
//!
//! ```text
//! ◆ Opus·high→codex   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24%   ♻ 63% cached
//! ```
//!
//! The three left segments (model·effort→backend, context, ✂ trim) are core and never
//! truncate; the extras (quota, then this turn's prompt-cache reuse) shed right-to-left as
//! the terminal narrows (`COLUMNS`). Segments whose data is absent — no reroute, an API-key
//! user with no rate limits, a non-reasoning model with no effort — simply don't render.
//!
//! `install` wires it into `~/.claude/settings.json`; rendering itself never touches the
//! network or API tokens.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::monitor::{self, DaemonView, Health};
use crate::tracking::Tracker;
use crate::ui;

/// Context-health budget: the token count at which the gauge reads full/red. Anchored to a
/// fixed "this is already a lot" ceiling rather than the model's technical window, so 200k
/// reads as heavy whether the window is 200k or 1M (a raw window-% would whisper "20%").
const CTX_BUDGET_TOKENS: i64 = 200_000;
const CTX_BAR_WIDTH: usize = 8;

// ── ANSI palette ────────────────────────────────────────────────────────────────
// The status line is captured by Claude Code (never a TTY), but Claude Code renders ANSI,
// so colour is emitted unconditionally — gated only by NO_COLOR, per the docs' examples.

const BRAND: &str = "38;2;153;204;255"; // llmtrim accent blue
const CYAN: &str = "36"; // codex
const VIOLET: &str = "38;2;181;137;255"; // kimi
const GREEN: &str = "32";
const AMBER: &str = "33";
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
    /// 5-hour rate-limit usage %, Claude.ai subscribers only.
    five_hour_pct: Option<f64>,
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
    let five_hour_pct = v
        .pointer("/rate_limits/five_hour/used_percentage")
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
        ctx_tokens,
        five_hour_pct,
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

fn ledger_snapshot(session_id: Option<&str>) -> Led {
    let reroute = llmtrim_core::config::RuntimeConfig::get()
        .sub
        .clone()
        .filter(|s| !s.is_empty() && s != "off");
    let health = proxy_health();
    let trim_pct = session_trim_pct(session_id);
    Led {
        health,
        trim_pct,
        reroute,
    }
}

/// Compression saved for `session_id` (its ledger rows), falling back to the lifetime figure
/// when the id is unknown. `None` — no measurable input yet — renders the idle marker.
fn session_trim_pct(session_id: Option<&str>) -> Option<f64> {
    let (before, after) = match session_id {
        Some(sid) => crate::breakdown::db::BreakdownDb::open()
            .ok()
            .and_then(|db| db.sessions().ok())
            .and_then(|rows| rows.into_iter().find(|r| r.session_id == sid))
            .map(|r| (r.input_before, r.input_after))?,
        None => {
            let s = Tracker::open().and_then(|t| t.summary()).ok()?;
            (s.input_before, s.input_after)
        }
    };
    (before > 0).then(|| ui::saved_pct(before as f64, after as f64))
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

/// `▓▓▓▓▓░░░ 142k` — gauge filled against the 200k health budget, coloured by absolute
/// tokens (green < 80k, amber 80–160k, red ≥ 160k), label in absolute k.
fn context_segment(ctx_tokens: i64, color: bool) -> String {
    let tokens = ctx_tokens.max(0);
    // Clamp to the budget *before* multiplying so a pathologically large token count can't
    // overflow i64 (the bar pins full past the budget anyway).
    let clamped = tokens.min(CTX_BUDGET_TOKENS);
    let filled = (clamped * CTX_BAR_WIDTH as i64 / CTX_BUDGET_TOKENS) as usize;
    let bar: String = "▓".repeat(filled) + &"░".repeat(CTX_BAR_WIDTH - filled);
    let k = (tokens as f64 / 1000.0).round() as i64;
    let code = if tokens == 0 {
        DIM
    } else {
        tier_color(tokens as f64, 80_000.0, 160_000.0)
    };
    paint(color, code, &format!("{bar} {k}k"))
}

/// The third core segment: `✂ 6.8%` when healthy and this session has saved something, a dim
/// `✂ –` when healthy but idle (nothing trimmed yet — avoids a misleading `✂ 0.0%` while still
/// signalling "llmtrim is on"), `⚠ llmtrim degraded` when broken, and nothing at all when
/// cleanly stopped (llmtrim is simply off — not an error to flag).
fn trim_or_health_segment(led: &Led, color: bool) -> Option<String> {
    match led.health {
        Health::Healthy => Some(match led.trim_pct {
            Some(pct) => paint(color, GREEN, &format!("✂ {pct:.1}%")),
            None => paint(color, DIM, "✂ –"),
        }),
        Health::Degraded => Some(paint(color, RED, "⚠ llmtrim degraded")),
        Health::Stopped => None,
    }
}

/// Build the ordered extra segments (quota, then per-session cache); later ones drop first.
fn extra_segments(cc: &CcInput, color: bool) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(p) = cc.five_hour_pct {
        let code = tier_color(p, 70.0, 90.0);
        // `◔` = the 5-hour window filling up; `·` keeps `5h` from reading as a duration.
        out.push(paint(color, code, &format!("◔ 5h·{}%", p.floor() as i64)));
    }
    if let Some(c) = cc.cache_pct {
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
        context_segment(cc.ctx_tokens, color),
    ];
    if let Some(seg) = trim_or_health_segment(led, color) {
        core.push(seg);
    }
    let mut line = core.join(SEP);

    for extra in extra_segments(cc, color) {
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
    let led = ledger_snapshot(cc.session_id.as_deref());
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
        }
    }

    fn cc(ctx: i64) -> CcInput {
        CcInput {
            model: "Opus".to_string(),
            effort: Some("high".to_string()),
            session_id: None,
            ctx_tokens: ctx,
            five_hour_pct: Some(24.0),
            cache_pct: Some(63.0),
        }
    }

    #[test]
    fn full_line_has_every_segment_when_wide() {
        let out = render_line(&cc(142_000), &led(Health::Healthy), 0, false);
        assert_eq!(
            out,
            "◆ Opus·high→codex   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24%   ♻ 63% cached"
        );
    }

    #[test]
    fn context_gauge_anchors_to_budget_not_window() {
        // 200k reads full regardless of the model's window: absolute label, pinned bar.
        let out = render_line(&cc(210_000), &led(Health::Healthy), 0, false);
        assert!(
            out.contains("▓▓▓▓▓▓▓▓ 210k"),
            "over budget pins full: {out}"
        );
        // A comfortable 48k is not full.
        let out = render_line(&cc(48_000), &led(Health::Healthy), 0, false);
        assert!(
            out.contains("▓░░░░░░░ 48k"),
            "48k is 1/8 blocks (floor): {out}"
        );
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
        let width = ui::visible_width("◆ Opus·high→codex   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 5h·24%");
        let out = render_line(&cc(142_000), &led(Health::Healthy), width, false);
        assert!(out.ends_with("5h·24%"), "keeps quota, sheds cache: {out}");
        assert!(!out.contains("cached"), "cache dropped first: {out}");
        assert!(full.len() > out.len());
    }

    #[test]
    fn absent_data_segments_do_not_render() {
        let mut c = cc(48_000);
        c.effort = None; // non-reasoning model
        c.five_hour_pct = None; // API-key user
        c.cache_pct = None; // before first API response
        let mut l = led(Health::Healthy);
        l.reroute = None; // no reroute
        let out = render_line(&c, &l, 0, false);
        assert_eq!(out, "◆ Opus   ▓░░░░░░░ 48k   ✂ 6.8%");
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
        let blob = r#"{"model":{"display_name":"Sonnet"},"effort":{"level":"medium"},
            "context_window":{"total_input_tokens":123456,
              "current_usage":{"input_tokens":10,"cache_creation_input_tokens":10,"cache_read_input_tokens":80}},
            "rate_limits":{"five_hour":{"used_percentage":41.2}}}"#;
        let cc = parse_cc(blob);
        assert_eq!(cc.model, "Sonnet");
        assert_eq!(cc.effort.as_deref(), Some("medium"));
        assert_eq!(cc.ctx_tokens, 123456);
        assert_eq!(cc.five_hour_pct, Some(41.2));
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
        // overflowing i64 (regression: `tokens * 8` used to overflow before clamping).
        assert!(context_segment(-5000, false).starts_with("░░░░░░░░ 0k"));
        assert!(context_segment(i64::MAX / 4, false).starts_with("▓▓▓▓▓▓▓▓"));
    }
}
