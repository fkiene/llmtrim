//! Subscription rate-limit windows for the Claude Code status line under `/sub`.
//!
//! Claude Code's stdin blob only carries *Anthropic* 5h/7d quotas. When traffic is rerouted to a
//! ChatGPT/Codex plan those numbers are the wrong account (and often absent). This module polls the
//! provider's usage endpoint, caches the result under `~/.llmtrim/sub-rate-limits.json`, and lets
//! the status line render the *active* plan's windows instead.
//!
//! **Codex (v1).** `GET https://chatgpt.com/backend-api/wham/usage` with the ChatGPT OAuth bearer
//! token + `ChatGPT-Account-Id` — the same endpoint headroom and CodeBurn use. The response's
//! `rate_limit.primary_window` / `secondary_window` each carry `used_percent`,
//! `limit_window_seconds`, and `reset_at`. Window size is dynamic per plan (Plus is often
//! weekly-only; Pro-style plans add a ~5h primary next to a weekly secondary), so labels are
//! derived from `limit_window_seconds`, not from primary/secondary position.
//!
//! **Kimi / Grok.** No public usage endpoint is wired yet; the cache schema is provider-keyed so
//! they can fill in later without touching the status line.
//!
//! Polling is throttled (default 60s) and never raises: a failed or unauthenticated fetch leaves
//! the previous cache (if any) alone. The status line only *reads* the cache — network work
//! happens on the proxy path when a Codex sub turn is about to fire.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// ChatGPT Codex usage endpoint. Overridable for tests via `LLMTRIM_CODEX_USAGE_URL`.
fn codex_usage_url() -> String {
    std::env::var("LLMTRIM_CODEX_USAGE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "https://chatgpt.com/backend-api/wham/usage".to_string())
}

/// Minimum seconds between live usage polls (process-local + honored via `fetched_at` on disk).
const POLL_MIN_INTERVAL_SECS: u64 = 60;
/// Bound the usage GET so a slow upstream never wedges a request path.
const POLL_TIMEOUT: Duration = Duration::from_secs(8);
/// Status line still renders a cache this old; older snapshots are treated as absent.
const CACHE_MAX_AGE_SECS: i64 = 30 * 60;

static POLL_GATE: Lazy<Mutex<PollGate>> = Lazy::new(|| Mutex::new(PollGate::default()));

#[derive(Default)]
struct PollGate {
    last_attempt: Option<Instant>,
    inflight: bool,
}

/// One rolling rate-limit window from a provider usage response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RateWindow {
    pub used_percent: f64,
    /// Unix epoch seconds when the window resets, when the provider sends one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resets_at: Option<i64>,
    /// Nominal window length in seconds (`18000` ≈ 5h, `604800` ≈ 7d). Used only to label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_window_seconds: Option<u64>,
}

/// Cached snapshot of the active subscription's rate-limit windows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubQuotaSnapshot {
    /// Provider shortname (`codex` today; `kimi`/`grok` reserved).
    pub provider: String,
    /// Unix epoch seconds when this snapshot was written.
    pub fetched_at: i64,
    /// Windows in provider order (primary then secondary for Codex). Empty means "no data".
    #[serde(default)]
    pub windows: Vec<RateWindow>,
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn cache_path() -> Option<PathBuf> {
    crate::daemon::home_dir()
        .ok()
        .map(|h| h.join("sub-rate-limits.json"))
}

/// Read the on-disk snapshot if it is present, parseable, and fresh enough for the status line.
pub fn load_fresh() -> Option<SubQuotaSnapshot> {
    load_raw().filter(|s| now_secs().saturating_sub(s.fetched_at) <= CACHE_MAX_AGE_SECS)
}

/// Read whatever is on disk, ignoring age. Used by tests and the poller (to preserve last-good).
pub fn load_raw() -> Option<SubQuotaSnapshot> {
    let path = cache_path()?;
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn store(snapshot: &SubQuotaSnapshot) {
    let Some(path) = cache_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let Ok(json) = serde_json::to_string_pretty(snapshot) else {
        return;
    };
    // Same-dir temp + rename so a concurrent statusline read never sees a half-write.
    // Pid + monotonic nanos keeps concurrent writers (restart race, two daemons) from
    // clobbering each other's temp file; `with_extension` would also mangle a multi-dot
    // basename, so build the sibling name explicitly.
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_file_name(format!(
        "sub-rate-limits.{}.{}.tmp",
        std::process::id(),
        nonce
    ));
    if fs::write(&tmp, json.as_bytes()).is_ok() {
        if fs::rename(&tmp, &path).is_err() {
            let _ = fs::remove_file(&tmp);
        }
    } else {
        let _ = fs::remove_file(&tmp);
    }
}

/// Parse a `GET /wham/usage` JSON body into a snapshot. `None` when no usable windows are present.
pub fn parse_codex_usage_payload(payload: &Value) -> Option<SubQuotaSnapshot> {
    let rate_limit = payload.get("rate_limit")?;
    let mut windows = Vec::new();
    for key in ["primary_window", "secondary_window"] {
        // Missing *or* JSON-null secondary must not abort parse of a good primary.
        if let Some(w) = rate_limit.get(key).and_then(window_from_usage_json) {
            windows.push(w);
        }
    }
    // Some plans put extra per-model buckets under `additional_rate_limits`; the status line only
    // has room for the two main windows, so they are intentionally ignored here.
    if windows.is_empty() {
        return None;
    }
    Some(SubQuotaSnapshot {
        provider: "codex".to_string(),
        fetched_at: now_secs(),
        windows,
    })
}

fn window_from_usage_json(win: &Value) -> Option<RateWindow> {
    let used = win.get("used_percent")?.as_f64().or_else(|| {
        // chatgpt.com has been observed sending integers; accept both.
        win.get("used_percent")?.as_i64().map(|i| i as f64)
    })?;
    if !used.is_finite() {
        return None;
    }
    let limit_window_seconds = win
        .get("limit_window_seconds")
        .and_then(Value::as_u64)
        .or_else(|| {
            win.get("limit_window_seconds")
                .and_then(Value::as_i64)
                .filter(|&s| s > 0)
                .map(|s| s as u64)
        });
    let resets_at = win
        .get("reset_at")
        .and_then(Value::as_i64)
        .or_else(|| win.get("reset_at").and_then(Value::as_u64).map(|u| u as i64));
    Some(RateWindow {
        used_percent: used,
        resets_at,
        limit_window_seconds,
    })
}

/// Classify a window by its nominal length for status-line labelling.
///
/// Codex does not name windows "5h"/"7d" on the wire — Plus often exposes only a weekly primary,
/// while Pro-style plans add a ~5h primary next to a weekly secondary. Thresholds are wide so a
/// 4h or 6h short window still maps to the short bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    /// Rolling short window (≈ 5 hours).
    Short,
    /// Rolling weekly (or multi-day) window.
    Weekly,
    /// Anything else — still renderable, labelled from size / reset time.
    Other,
}

pub fn classify_window(limit_window_seconds: Option<u64>) -> WindowKind {
    match limit_window_seconds {
        Some(s) if s > 0 && s < 12 * 3600 => WindowKind::Short,
        Some(s) if s >= 3 * 24 * 3600 => WindowKind::Weekly,
        _ => WindowKind::Other,
    }
}

/// Fallback label when `resets_at` is missing: `5h` / `7d` / `Nh` / `Nd` from window size.
pub fn size_label(limit_window_seconds: Option<u64>) -> String {
    match limit_window_seconds {
        Some(s) if s > 0 && s < 3600 => format!("{}m", (s + 59) / 60),
        Some(s) if s < 12 * 3600 => {
            let h = (s + 1800) / 3600;
            if h <= 1 {
                "1h".to_string()
            } else {
                format!("{h}h")
            }
        }
        Some(s) if s >= 24 * 3600 => {
            let d = (s + 12 * 3600) / (24 * 3600);
            if d <= 1 {
                "1d".to_string()
            } else {
                format!("{d}d")
            }
        }
        Some(s) if s > 0 => format!("{}h", (s + 1800) / 3600),
        _ => "?".to_string(),
    }
}

/// Split a snapshot into (short/5h, weekly/7d) slots for the status-line segment.
///
/// When only one window is present it goes into the matching slot; a lone weekly does **not**
/// invent a 5h row. If both windows share a kind (schema drift), the first keeps the slot and the
/// second is dropped rather than double-rendering.
pub fn split_short_weekly(
    snapshot: &SubQuotaSnapshot,
) -> (Option<&RateWindow>, Option<&RateWindow>) {
    let mut short = None;
    let mut weekly = None;
    for w in &snapshot.windows {
        match classify_window(w.limit_window_seconds) {
            WindowKind::Short if short.is_none() => short = Some(w),
            WindowKind::Weekly if weekly.is_none() => weekly = Some(w),
            WindowKind::Other => {
                // Prefer to surface an unclassified window as weekly (longer-horizon) when that
                // slot is free; otherwise as short. Keeps a single unknown window visible.
                if weekly.is_none() && short.is_some() {
                    weekly = Some(w);
                } else if short.is_none() {
                    short = Some(w);
                } else if weekly.is_none() {
                    weekly = Some(w);
                }
            }
            _ => {}
        }
    }
    // Single weekly-classified window with empty short is the Plus shape (primary = 7d).
    if short.is_none() && weekly.is_none() {
        // Defensive: classification filtered everything — surface first window as weekly.
        if let Some(w) = snapshot.windows.first() {
            return (None, Some(w));
        }
    }
    (short, weekly)
}

/// Blocking GET of the Codex usage endpoint. Returns a stored snapshot on success.
pub fn fetch_codex_usage(access_token: &str, account_id: Option<&str>) -> Option<SubQuotaSnapshot> {
    let account_id = account_id.filter(|s| !s.is_empty())?;
    if access_token.is_empty() {
        return None;
    }
    let url = codex_usage_url();
    let mut resp = ureq::get(&url)
        .config()
        .proxy(None)
        .http_status_as_error(false)
        .timeout_global(Some(POLL_TIMEOUT))
        .build()
        .header("Authorization", &format!("Bearer {access_token}"))
        .header("ChatGPT-Account-Id", account_id)
        .header("Accept", "application/json")
        .header("User-Agent", "llmtrim")
        .call()
        .ok()?;
    let status = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().ok()?;
    if !(200..300).contains(&status) {
        return None;
    }
    let payload: Value = serde_json::from_str(&body).ok()?;
    let snap = parse_codex_usage_payload(&payload)?;
    store(&snap);
    Some(snap)
}

/// Fire-and-forget a throttled Codex usage poll on a detached thread.
///
/// Safe to call on every Codex sub turn: process-local gate + on-disk `fetched_at` keep it to
/// roughly one live GET per [`POLL_MIN_INTERVAL_SECS`]. Never blocks the caller for network I/O.
pub fn maybe_schedule_codex_poll(access_token: String, account_id: Option<String>) {
    if access_token.is_empty() || account_id.as_deref().unwrap_or("").is_empty() {
        return;
    }
    // Disk freshness: skip even scheduling if a recent snapshot already exists.
    if load_raw().is_some_and(|s| {
        s.provider == "codex" && now_secs().saturating_sub(s.fetched_at) < POLL_MIN_INTERVAL_SECS as i64
    }) {
        return;
    }
    {
        let mut gate = POLL_GATE.lock().unwrap_or_else(|p| p.into_inner());
        if gate.inflight {
            return;
        }
        if gate
            .last_attempt
            .is_some_and(|t| t.elapsed() < Duration::from_secs(POLL_MIN_INTERVAL_SECS))
        {
            return;
        }
        gate.inflight = true;
        gate.last_attempt = Some(Instant::now());
    }
    // If spawn itself fails, clear inflight immediately — otherwise the flag stays true
    // for the life of the process and every later Codex turn skips the poll forever.
    let spawn = std::thread::Builder::new()
        .name("llmtrim-codex-quota".into())
        .spawn(move || {
            let _ = fetch_codex_usage(&access_token, account_id.as_deref());
            if let Ok(mut gate) = POLL_GATE.lock() {
                gate.inflight = false;
            }
        });
    if spawn.is_err() {
        if let Ok(mut gate) = POLL_GATE.lock() {
            gate.inflight = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_plus_weekly_only_primary() {
        let payload = json!({
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "primary_window": {
                    "used_percent": 1,
                    "limit_window_seconds": 604800,
                    "reset_after_seconds": 498478,
                    "reset_at": 1784791984
                },
                "secondary_window": null
            }
        });
        let snap = parse_codex_usage_payload(&payload).expect("snapshot");
        assert_eq!(snap.provider, "codex");
        assert_eq!(snap.windows.len(), 1);
        assert_eq!(snap.windows[0].used_percent, 1.0);
        assert_eq!(snap.windows[0].limit_window_seconds, Some(604_800));
        assert_eq!(snap.windows[0].resets_at, Some(1_784_791_984));
        let (short, weekly) = split_short_weekly(&snap);
        assert!(short.is_none(), "Plus weekly-only must not invent a 5h row");
        assert_eq!(weekly.unwrap().used_percent, 1.0);
    }

    #[test]
    fn parses_pro_style_primary_5h_secondary_weekly() {
        let payload = json!({
            "rate_limit": {
                "primary_window": {
                    "used_percent": 23.5,
                    "limit_window_seconds": 18000,
                    "reset_at": 1700000000
                },
                "secondary_window": {
                    "used_percent": 6,
                    "limit_window_seconds": 604800,
                    "reset_at": 1700500000
                }
            }
        });
        let snap = parse_codex_usage_payload(&payload).unwrap();
        let (short, weekly) = split_short_weekly(&snap);
        assert_eq!(short.unwrap().used_percent, 23.5);
        assert_eq!(weekly.unwrap().used_percent, 6.0);
        assert_eq!(classify_window(Some(18_000)), WindowKind::Short);
        assert_eq!(classify_window(Some(604_800)), WindowKind::Weekly);
    }

    #[test]
    fn empty_rate_limit_returns_none() {
        assert!(parse_codex_usage_payload(&json!({"rate_limit": {}})).is_none());
        assert!(parse_codex_usage_payload(&json!({})).is_none());
        assert!(
            parse_codex_usage_payload(&json!({
                "rate_limit": {"primary_window": {"limit_window_seconds": 60}}
            }))
            .is_none(),
            "missing used_percent skips the window"
        );
    }

    #[test]
    fn size_label_rounds_common_windows() {
        assert_eq!(size_label(Some(18_000)), "5h");
        assert_eq!(size_label(Some(604_800)), "7d");
        assert_eq!(size_label(Some(3600)), "1h");
        assert_eq!(size_label(None), "?");
    }

    #[test]
    fn snapshot_json_round_trip() {
        let snap = SubQuotaSnapshot {
            provider: "codex".into(),
            fetched_at: 1_700_000_000,
            windows: vec![RateWindow {
                used_percent: 42.0,
                resets_at: Some(1_700_000_000),
                limit_window_seconds: Some(18_000),
            }],
        };
        let json = serde_json::to_string_pretty(&snap).unwrap();
        let loaded: SubQuotaSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, snap);
    }
}
