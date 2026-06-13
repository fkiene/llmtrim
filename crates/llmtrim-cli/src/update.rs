//! End-user updates: a channel-aware `update` command + an occasional, cached, opt-out
//! "new release available" check.
//!
//! No heavy self-update machinery. The binary channel re-runs the canonical installer
//! (which downloads the latest release and restarts the daemon via `setup`); cargo /
//! Homebrew installs are told to use their package manager. The release check hits the
//! GitHub API, cached ≤ once/day (so the unauthenticated rate limit is irrelevant), and is
//! skipped offline or when `LLMTRIM_NO_UPDATE_CHECK` is set.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

// `Context` is only used by the non-Windows installer arm in `run()`.
#[cfg_attr(windows, allow(unused_imports))]
use anyhow::{Context, Result};

const CURRENT: &str = env!("CARGO_PKG_VERSION");

/// `owner/name` parsed from the crate's repository URL.
fn repo() -> &'static str {
    env!("CARGO_PKG_REPOSITORY")
        .trim_end_matches('/')
        .trim_start_matches("https://github.com/")
}

#[derive(PartialEq, Eq)]
pub(crate) enum Channel {
    Binary,
    Cargo,
    Homebrew,
    Npm,
}

/// Where this binary was installed from — determines how to update it.
pub(crate) fn channel() -> Channel {
    channel_of(
        &std::env::current_exe()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default(),
    )
}

fn channel_of(p: &str) -> Channel {
    if p.contains("node_modules") || p.contains("/_npx/") || p.contains("\\_npx\\") {
        // npm global install or an npx cache — npm owns this binary, never the installer.
        Channel::Npm
    } else if p.contains("/.cargo/") || p.contains("\\.cargo\\") {
        Channel::Cargo
    } else if p.contains("/Cellar/") || p.contains("/homebrew/") || p.contains("/linuxbrew/") {
        Channel::Homebrew
    } else {
        Channel::Binary
    }
}

/// (major, minor, patch) for a loose comparison; non-numeric / pre-release suffixes ignored.
fn semver(s: &str) -> (u64, u64, u64) {
    let mut it = s.trim_start_matches('v').split(['.', '-', '+']);
    let n = |x: Option<&str>| x.and_then(|v| v.parse().ok()).unwrap_or(0);
    (n(it.next()), n(it.next()), n(it.next()))
}

/// Latest released version (without a leading `v`), or `None` on any failure (offline,
/// rate-limit, parse error) — callers stay silent on `None`.
fn fetch_latest() -> Option<String> {
    let url = format!("https://api.github.com/repos/{}/releases/latest", repo());
    let mut req = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(3)))
        .http_status_as_error(false)
        .build();
    req = req.header("User-Agent", "llmtrim-update-check"); // GitHub API requires a UA
    let body = req.call().ok()?.body_mut().read_to_string().ok()?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    Some(
        v.get("tag_name")?
            .as_str()?
            .trim_start_matches('v')
            .to_string(),
    )
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn cache_path() -> Option<std::path::PathBuf> {
    crate::daemon::home_dir()
        .ok()
        .map(|h| h.join("update-check.json"))
}

fn write_cache(latest: &str) {
    if let Some(p) = cache_path() {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(
            &p,
            serde_json::json!({ "checked_at": now_secs(), "latest": latest }).to_string(),
        );
    }
}

/// A newer-version string if a release beyond the running version is known. Cached ≤ 24h;
/// opt out with `LLMTRIM_NO_UPDATE_CHECK`; silent on any failure. Used for the passive
/// `monitor` banner — `force` bypasses the cache (used by the `update` command).
pub fn check(force: bool) -> Option<String> {
    if std::env::var_os("LLMTRIM_NO_UPDATE_CHECK").is_some() {
        return None;
    }
    if !force
        && let Some(txt) = cache_path().and_then(|p| std::fs::read_to_string(p).ok())
        && let Ok(c) = serde_json::from_str::<serde_json::Value>(&txt)
    {
        let at = c.get("checked_at").and_then(|x| x.as_u64()).unwrap_or(0);
        if now_secs().saturating_sub(at) < 86_400 {
            let latest = c.get("latest").and_then(|x| x.as_str()).unwrap_or("");
            return newer(latest);
        }
    }
    // Cache the result either way — including "" on failure — so an offline box backs off
    // for 24h instead of re-hitting the network on every `monitor`.
    let latest = fetch_latest().unwrap_or_default();
    write_cache(&latest);
    newer(&latest)
}

fn newer(latest: &str) -> Option<String> {
    (!latest.is_empty() && semver(latest) > semver(CURRENT)).then(|| latest.to_string())
}

/// The `llmtrim update` command — channel-aware.
pub fn run() -> Result<()> {
    let color = crate::ui::color_stdout();
    println!(
        "{}",
        crate::ui::paint(
            color,
            crate::ui::Tone::Dim,
            &format!("llmtrim v{CURRENT} — checking the latest release…")
        )
    );
    let latest = fetch_latest();
    match &latest {
        Some(v) if semver(v) <= semver(CURRENT) => {
            println!(
                "{}",
                crate::ui::ok(color, &format!("Already up to date (v{CURRENT})."))
            );
            return Ok(());
        }
        Some(v) => println!(
            "{} v{v} available {}",
            crate::ui::paint(color, crate::ui::Tone::Accent, "→"),
            crate::ui::paint(
                color,
                crate::ui::Tone::Dim,
                &format!("(you have v{CURRENT})")
            )
        ),
        None => println!(
            "{}",
            crate::ui::note(
                color,
                "Couldn't reach GitHub to confirm the version — proceeding anyway."
            )
        ),
    }

    // Package-manager channels get their commands in a panel; the binary channel on
    // Unix actually runs the installer, so its output stays plain.
    let instructions = |title: &str, cmds: &[&str]| {
        let lines: Vec<String> = cmds.iter().map(|c| c.to_string()).collect();
        print!("\n{}", crate::ui::panel(color, title, &lines));
    };
    match channel() {
        Channel::Cargo => instructions(
            "update via cargo",
            &[
                "cargo install --locked llmtrim --force",
                "llmtrim setup    # restart the daemon on the new binary",
            ],
        ),
        Channel::Homebrew => instructions(
            "update via Homebrew",
            &[
                "brew upgrade llmtrim",
                "llmtrim setup    # restart the daemon on the new binary",
            ],
        ),
        Channel::Npm => instructions(
            "update via npm",
            &[
                "npm update -g @llmtrim/cli",
                "llmtrim setup    # restart the daemon on the new binary",
            ],
        ),
        Channel::Binary => {
            let tag = latest
                .as_deref()
                .map(|v| format!("v{v}"))
                .unwrap_or_else(|| "main".to_string());
            #[cfg(windows)]
            instructions(
                "update via the installer",
                &[
                    &format!(
                        "iwr -useb https://raw.githubusercontent.com/{}/{tag}/install.ps1 | iex",
                        repo()
                    ),
                    "llmtrim setup    # restart the daemon on the new binary",
                ],
            );
            #[cfg(not(windows))]
            {
                let url = format!(
                    "https://raw.githubusercontent.com/{}/{tag}/install.sh",
                    repo()
                );
                println!(
                    "Updating via the installer (downloads the latest release; `setup` restarts the daemon)…"
                );
                let mut cmd = std::process::Command::new("sh");
                cmd.args(["-c", &format!("curl -fsSL {url} | sh")]);
                // Pin the *installed* version to the same resolved tag as the script, so the
                // whole update is deterministic (no second "latest" lookup inside install.sh).
                if tag != "main" {
                    cmd.env("LLMTRIM_VERSION", &tag);
                }
                let status = cmd
                    .status()
                    .context("failed to launch the installer (curl + sh required)")?;
                if !status.success() {
                    anyhow::bail!("installer exited non-zero");
                }
                if let Some(v) = latest {
                    write_cache(&v); // clear the `monitor` banner
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_detection_by_path() {
        assert!(matches!(
            channel_of("/home/u/.cargo/bin/llmtrim"),
            Channel::Cargo
        ));
        assert!(matches!(
            channel_of("C:\\Users\\u\\.cargo\\bin\\llmtrim.exe"),
            Channel::Cargo
        ));
        assert!(matches!(
            channel_of("/opt/homebrew/Cellar/llmtrim/0.1.1/bin/llmtrim"),
            Channel::Homebrew
        ));
        assert!(matches!(
            channel_of("/home/linuxbrew/.linuxbrew/bin/llmtrim"),
            Channel::Homebrew
        ));
        assert!(matches!(
            channel_of("/usr/lib/node_modules/llmtrim-linux-x64/bin/llmtrim"),
            Channel::Npm
        ));
        assert!(matches!(
            channel_of("/home/u/.npm/_npx/abc123/node_modules/.bin/llmtrim"),
            Channel::Npm
        ));
        assert!(matches!(
            channel_of("C:\\Users\\u\\AppData\\npm-cache\\_npx\\x\\llmtrim.exe"),
            Channel::Npm
        ));
        assert!(matches!(
            channel_of("/home/u/.local/bin/llmtrim"),
            Channel::Binary
        ));
        assert!(matches!(
            channel_of("C:\\Users\\u\\AppData\\Local\\llmtrim\\bin\\llmtrim.exe"),
            Channel::Binary
        ));
        assert!(matches!(channel_of(""), Channel::Binary));
    }

    #[test]
    fn semver_compares() {
        assert!(semver("0.2.0") > semver("0.1.9"));
        assert!(semver("v1.0.0") > semver("0.9.9"));
        assert_eq!(semver("0.1.0"), semver("0.1.0-rc1")); // pre-release suffix ignored
    }

    #[test]
    fn newer_only_when_ahead() {
        assert!(newer("999.0.0").is_some());
        assert!(newer(CURRENT).is_none());
        assert!(newer("0.0.1").is_none());
        assert!(newer("").is_none());
    }

    #[test]
    fn repo_is_owner_slash_name() {
        assert!(!repo().starts_with("http"));
        assert_eq!(repo().matches('/').count(), 1, "owner/name");
    }
}
