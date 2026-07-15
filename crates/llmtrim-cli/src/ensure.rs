//! Bring this install to the recommended current state — one verb for setup, update, and repair.
//!
//! Happy path: `llmtrim setup` / `llmtrim update` / `llmtrim doctor --fix` / status `f` all
//! land here. Power-user install/uninstall commands remain as escape hatches; humans should not
//! need them after a release. Owned Claude Code hooks and the status line are rewritten in place
//! when the binary path or feature payload changes so changelogs never say "re-run install".

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ui;

const CURRENT: &str = env!("CARGO_PKG_VERSION");
const STATE_FILE: &str = "integrations.json";

/// User opt-outs, remembered forever so ensure never re-prompts after an explicit no.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct OptOut {
    pub statusline: bool,
    pub guard: bool,
    pub window_sub: bool,
    pub compact: bool,
    pub tray_autostart: bool,
    /// User declined the optional Linux tray download.
    pub tray_download: bool,
    /// User dismissed the one-time subscription onboarding nudge.
    pub sub_nudge: bool,
}

/// Persistent integration state under `~/.llmtrim/integrations.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct State {
    pub last_ensured_version: Option<String>,
    pub opt_out: OptOut,
    /// Subscription onboarding already shown in the status TUI.
    pub sub_nudge_shown: bool,
}

impl State {
    /// Load state; missing file → defaults. Corrupt file → error (never reset opt-outs silently).
    pub fn load() -> Result<Self> {
        let path = state_path()?;
        load_at(&path)
    }

    pub fn save(&self) -> Result<()> {
        let path = state_path()?;
        save_at(&path, self)
    }
}

fn state_path() -> Result<PathBuf> {
    Ok(crate::daemon::home_dir()?.join(STATE_FILE))
}

fn load_at(path: &Path) -> Result<State> {
    if !path.exists() {
        return Ok(State::default());
    }
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| {
        format!(
            "{} is corrupt or invalid JSON — fix or delete it before running ensure              (deleting resets opt-outs)",
            path.display()
        )
    })
}

fn save_at(path: &Path, state: &State) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(state)?)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename onto {}", path.display()))?;
    Ok(())
}

/// How ensure was invoked — controls prompting and how loud the report is.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// May prompt on a TTY (first-run compact/tray choices). Non-interactive defaults to yes
    /// for recommended items, no for optional network downloads.
    pub interactive: bool,
    /// Skip the success panel (daemon start / quiet migrations).
    pub quiet: bool,
    /// Restart the daemon when binary/daemon versions disagree.
    pub restart_daemon: bool,
    /// Allow downloading the Linux tray binary when missing (interactive confirm, or forced).
    pub download_tray: bool,
    /// When false (quiet auto-heal), only refresh *owned* stale integrations and daemon skew —
    /// never first-install statusline/guard//sub/compact or enable tray autostart.
    pub install_missing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            interactive: std::io::IsTerminal::is_terminal(&std::io::stdin()),
            quiet: false,
            restart_daemon: true,
            download_tray: false,
            install_missing: true,
        }
    }
}

/// One gap between the desired recommended state and reality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub id: &'static str,
    pub label: String,
    pub detail: String,
}

/// Outcome of [`apply`].
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub rows: Vec<(&'static str, String, String)>,
    /// Integration ids that were changed.
    pub applied: Vec<&'static str>,
    pub gaps_before: usize,
}

impl Report {
    pub fn changed(&self) -> bool {
        !self.applied.is_empty()
    }
}

/// Probe without writing. Pure enough for the status TUI and doctor rows.
pub fn probe() -> Vec<Gap> {
    match State::load() {
        Ok(s) => probe_with(&s),
        Err(_) => Vec::new(), // corrupt state: do not invent gaps; ensure/apply will error
    }
}

fn probe_with(state: &State) -> Vec<Gap> {
    let mut gaps = Vec::new();
    let claude = crate::statusline::claude_code_present();

    if claude && !state.opt_out.statusline {
        match crate::statusline::owned_status() {
            crate::statusline::OwnedStatus::Missing => gaps.push(Gap {
                id: "statusline",
                label: "Statusline".into(),
                detail: "not installed — recommended for Claude Code".into(),
            }),
            crate::statusline::OwnedStatus::Stale => gaps.push(Gap {
                id: "statusline",
                label: "Statusline".into(),
                detail: "stale (binary path or refresh settings)".into(),
            }),
            crate::statusline::OwnedStatus::Current | crate::statusline::OwnedStatus::Foreign => {}
        }
    }

    if claude && !state.opt_out.guard {
        match crate::guard::owned_status() {
            crate::guard::OwnedStatus::Missing => gaps.push(Gap {
                id: "guard",
                label: "Guard".into(),
                detail: "not installed — warns before cold-cache resumes".into(),
            }),
            crate::guard::OwnedStatus::Stale => gaps.push(Gap {
                id: "guard",
                label: "Guard".into(),
                detail: "stale (binary path)".into(),
            }),
            crate::guard::OwnedStatus::Current => {}
        }
    }

    if claude && !state.opt_out.window_sub {
        match crate::window_sub::owned_status() {
            crate::window_sub::OwnedStatus::Missing => gaps.push(Gap {
                id: "window_sub",
                label: "/sub".into(),
                detail: "window-local subscription controls not installed".into(),
            }),
            crate::window_sub::OwnedStatus::Stale => gaps.push(Gap {
                id: "window_sub",
                label: "/sub".into(),
                detail: "stale (binary path)".into(),
            }),
            crate::window_sub::OwnedStatus::Current => {}
        }
    }

    if claude && !state.opt_out.compact && !llmtrim_core::config::compact_models_configured() {
        gaps.push(Gap {
            id: "compact",
            label: "Compact".into(),
            detail: "cheaper /compact models not configured".into(),
        });
    }

    // Version skew: daemon older than this binary.
    if let Some(d) = crate::daemon::running()
        && let Some(v) = d.version.as_deref()
        && v != CURRENT
    {
        gaps.push(Gap {
            id: "daemon",
            label: "Daemon".into(),
            detail: format!("running v{v}, binary is v{CURRENT}"),
        });
    }

    // Tray binary appeared but login entry never enabled (and user did not opt out).
    if crate::tray::tray_binary().is_some()
        && !state.opt_out.tray_autostart
        && !crate::autostart::is_tray_enabled()
    {
        gaps.push(Gap {
            id: "tray_autostart",
            label: "Tray".into(),
            detail: "installed but not set to open at login".into(),
        });
    }

    gaps
}

/// Whether the status TUI / doctor should offer a one-key fix.
pub fn needs_attention() -> bool {
    !probe().is_empty()
}

/// Short banner line for the status dashboard (None when clean).
pub fn attention_summary() -> Option<String> {
    let gaps = probe();
    if gaps.is_empty() {
        return None;
    }
    let labels: Vec<&str> = gaps.iter().map(|g| g.label.as_str()).collect();
    Some(format!(
        "{} need{} attention — press f to fix",
        labels.join(", "),
        if gaps.len() == 1 { "s" } else { "" }
    ))
}

/// Apply recommended state. Idempotent; honors opt-outs.
pub fn apply(opts: Options) -> Result<Report> {
    let mut state = State::load()?;
    let gaps_before = probe_with(&state).len();
    let mut report = Report {
        gaps_before,
        ..Report::default()
    };
    let claude = crate::statusline::claude_code_present();

    if claude && !state.opt_out.statusline {
        let status = crate::statusline::owned_status();
        let skip_missing =
            !opts.install_missing && matches!(status, crate::statusline::OwnedStatus::Missing);
        if !skip_missing {
            match crate::statusline::sync_owned() {
                Ok(crate::statusline::SyncOutcome::Installed) => {
                    report.applied.push("statusline");
                    report.rows.push((
                        ui::OK,
                        "Statusline".into(),
                        "installed in ~/.claude/settings.json".into(),
                    ));
                }
                Ok(crate::statusline::SyncOutcome::Refreshed) => {
                    report.applied.push("statusline");
                    report.rows.push((
                        ui::OK,
                        "Statusline".into(),
                        "refreshed for this binary".into(),
                    ));
                }
                Ok(crate::statusline::SyncOutcome::AlreadyCurrent) => {
                    report
                        .rows
                        .push((ui::OK, "Statusline".into(), "current".into()));
                }
                Ok(crate::statusline::SyncOutcome::SkippedForeign) => {
                    report.rows.push((
                        ui::NOTE,
                        "Statusline".into(),
                        "custom status line left alone".into(),
                    ));
                }
                Err(e) => {
                    report
                        .rows
                        .push((ui::WARN, "Statusline".into(), format!("not wired: {e:#}")))
                }
            }
        }
    } else if claude && state.opt_out.statusline {
        report
            .rows
            .push((ui::NOTE, "Statusline".into(), "opted out".into()));
    }

    if claude && !state.opt_out.guard {
        let gstatus = crate::guard::owned_status();
        let skip_missing =
            !opts.install_missing && matches!(gstatus, crate::guard::OwnedStatus::Missing);
        if !skip_missing {
            match crate::guard::sync_owned() {
                Ok(true) => {
                    report.applied.push("guard");
                    report.rows.push((
                        ui::OK,
                        "Guard".into(),
                        "warns before a cold resumed turn".into(),
                    ));
                }
                Ok(false) => {
                    report.rows.push((ui::OK, "Guard".into(), "current".into()));
                }
                Err(e) => report
                    .rows
                    .push((ui::WARN, "Guard".into(), format!("not wired: {e:#}"))),
            }
        }
    } else if claude && state.opt_out.guard {
        report
            .rows
            .push((ui::NOTE, "Guard".into(), "opted out".into()));
    }

    if claude && !state.opt_out.window_sub {
        let wstatus = crate::window_sub::owned_status();
        let skip_missing =
            !opts.install_missing && matches!(wstatus, crate::window_sub::OwnedStatus::Missing);
        if !skip_missing {
            let was_installed = matches!(
                wstatus,
                crate::window_sub::OwnedStatus::Current | crate::window_sub::OwnedStatus::Stale
            );
            match crate::window_sub::install(&crate::statusline::stable_exe_string()) {
                Ok(()) => {
                    if was_installed {
                        report.rows.push((
                            ui::OK,
                            "/sub".into(),
                            "window-local controls current".into(),
                        ));
                    } else {
                        report.applied.push("window_sub");
                        report.rows.push((
                            ui::OK,
                            "/sub".into(),
                            "window-local on|off|status installed".into(),
                        ));
                    }
                }
                Err(e) => {
                    report
                        .rows
                        .push((ui::WARN, "/sub".into(), format!("not installed: {e:#}")))
                }
            }
        }
    } else if claude && state.opt_out.window_sub {
        report
            .rows
            .push((ui::NOTE, "/sub".into(), "opted out".into()));
    }

    if claude
        && opts.install_missing
        && !state.opt_out.compact
        && !llmtrim_core::config::compact_models_configured()
    {
        let want = if opts.interactive {
            confirm_default_yes("Use cheaper models for Claude Code /compact (Haiku → Sonnet)?")
        } else {
            true
        };
        if want {
            match llmtrim_core::config::write_compact_models(&["haiku".into(), "sonnet".into()]) {
                Ok(()) => {
                    report.applied.push("compact");
                    report.rows.push((
                        ui::OK,
                        "Compact".into(),
                        "Haiku → Sonnet → original model".into(),
                    ));
                }
                Err(e) => {
                    report
                        .rows
                        .push((ui::WARN, "Compact".into(), format!("not configured: {e:#}")))
                }
            }
        } else {
            // Remember opt-out as empty models list + flag.
            let _ = llmtrim_core::config::write_compact_models(&[]);
            state.opt_out.compact = true;
            report.rows.push((
                ui::OK,
                "Compact".into(),
                "original model only (remembered)".into(),
            ));
        }
    } else if claude && llmtrim_core::config::compact_models_configured() {
        report
            .rows
            .push((ui::OK, "Compact".into(), "configured".into()));
    }

    // Tray autostart when binary is present (explicit ensure/setup only).
    if opts.install_missing && crate::tray::tray_binary().is_some() && !state.opt_out.tray_autostart
    {
        if !crate::autostart::is_tray_enabled() {
            let want = if opts.interactive {
                confirm_default_yes("Enable the llmtrim desktop tray (opens at login)?")
            } else {
                true
            };
            if want {
                match crate::autostart::configure_tray(true) {
                    Ok(()) => {
                        report.applied.push("tray_autostart");
                        report.rows.push((
                            ui::OK,
                            "Tray".into(),
                            "opens at login · run `llmtrim tray` to open now".into(),
                        ));
                    }
                    Err(e) => report.rows.push((
                        ui::WARN,
                        "Tray".into(),
                        format!("autostart failed: {e:#}"),
                    )),
                }
            } else {
                state.opt_out.tray_autostart = true;
                report.rows.push((
                    ui::NOTE,
                    "Tray".into(),
                    "left off · enable later with `llmtrim autostart --tray`".into(),
                ));
            }
        } else {
            report
                .rows
                .push((ui::OK, "Tray".into(), "autostart on".into()));
        }
    } else if opts.install_missing
        && crate::tray::tray_binary().is_none()
        && !state.opt_out.tray_download
        && cfg!(target_os = "linux")
        && std::env::var_os("DISPLAY")
            .or_else(|| std::env::var_os("WAYLAND_DISPLAY"))
            .is_some()
    {
        // Optional one-shot tray download on Linux desktops.
        let want = opts.download_tray
            || (opts.interactive
                && confirm_default_no(
                    "Desktop tray not installed. Download llmtrim-tray from GitHub releases?",
                ));
        if want {
            match download_linux_tray() {
                Ok(path) => {
                    report.applied.push("tray_download");
                    report.rows.push((
                        ui::OK,
                        "Tray".into(),
                        format!("installed {}", path.display()),
                    ));
                    if !state.opt_out.tray_autostart {
                        let _ = crate::autostart::configure_tray(true);
                    }
                }
                Err(e) => {
                    report
                        .rows
                        .push((ui::WARN, "Tray".into(), format!("download failed: {e:#}")))
                }
            }
        } else if opts.interactive {
            state.opt_out.tray_download = true;
            report.rows.push((
                ui::NOTE,
                "Tray".into(),
                "skipped — download later from the GitHub release".into(),
            ));
        }
    }

    // Daemon version skew.
    if opts.restart_daemon
        && let Some(d) = crate::daemon::running()
        && d.version.as_deref().is_some_and(|v| v != CURRENT)
    {
        match crate::update::restart_daemon_silent() {
            Ok(()) => {
                report.applied.push("daemon");
                report
                    .rows
                    .push((ui::OK, "Daemon".into(), format!("restarted on v{CURRENT}")));
            }
            Err(e) => report.rows.push((
                ui::WARN,
                "Daemon".into(),
                format!("restart failed: {e:#} — try `llmtrim start --force`"),
            )),
        }
    }

    state.last_ensured_version = Some(CURRENT.to_string());
    if let Err(e) = state.save() {
        report.rows.push((
            ui::WARN,
            "State".into(),
            format!("could not save integrations.json: {e:#}"),
        ));
    }

    Ok(report)
}

/// Auto-heal after a version bump (daemon start / first status). Quiet, non-interactive,
/// never downloads. Returns whether anything changed.
pub fn maybe_auto() -> Result<bool> {
    let state = match State::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("llmtrim: skip auto-heal ({e:#})");
            return Ok(false);
        }
    };
    let gaps = probe_with(&state);
    let version_mismatch = state.last_ensured_version.as_deref() != Some(CURRENT);
    let needs_refresh = gaps
        .iter()
        .any(|g| g.id == "daemon" || g.detail.contains("stale"));
    // Quiet path never first-installs missing integrations — only refresh owned stale
    // pieces / daemon skew, or stamp the version after a binary bump.
    if !version_mismatch && !needs_refresh {
        return Ok(false);
    }
    let report = apply(Options {
        interactive: false,
        quiet: true,
        restart_daemon: true,
        download_tray: false,
        install_missing: false,
    })?;
    Ok(report.changed())
}

/// CLI entry for `llmtrim ensure` and `doctor --fix`.
pub fn run_cli(quiet: bool) -> Result<()> {
    let color = ui::color_stdout();
    let report = apply(Options {
        interactive: std::io::IsTerminal::is_terminal(&std::io::stdin()),
        quiet,
        restart_daemon: true,
        download_tray: false,
        install_missing: true,
    })?;
    if !quiet {
        print!(
            "{}",
            ui::panel(color, "llmtrim ensure", &ui::kv_rows(color, &report.rows))
        );
        if report.applied.is_empty() {
            println!("{}", ui::ok(color, "already at the recommended state."));
        } else {
            println!(
                "{}",
                ui::ok(
                    color,
                    &format!(
                        "applied {} change{} · you're on v{CURRENT}",
                        report.applied.len(),
                        if report.applied.len() == 1 { "" } else { "s" }
                    )
                )
            );
        }
    }
    Ok(())
}

/// Record an opt-out (e.g. after `guard uninstall`).
pub fn set_opt_out(id: &str, value: bool) -> Result<()> {
    let mut state = State::load()?;
    match id {
        "statusline" => state.opt_out.statusline = value,
        "guard" => state.opt_out.guard = value,
        "window_sub" | "window-sub" | "sub" => state.opt_out.window_sub = value,
        "compact" => state.opt_out.compact = value,
        "tray_autostart" | "tray" => state.opt_out.tray_autostart = value,
        "tray_download" => state.opt_out.tray_download = value,
        "sub_nudge" => state.opt_out.sub_nudge = value,
        _ => anyhow::bail!("unknown integration id: {id}"),
    }
    state.save()
}

/// Whether the status TUI should show one-time sub onboarding.
pub fn should_show_sub_nudge() -> bool {
    let Ok(state) = State::load() else {
        return false;
    };
    if state.opt_out.sub_nudge || state.sub_nudge_shown {
        return false;
    }
    if !crate::statusline::claude_code_present() {
        return false;
    }
    // Only nudge when sub is not already configured.
    let cfg = llmtrim_core::config::RuntimeConfig::get();
    cfg.sub
        .as_deref()
        .is_none_or(|s| s == "off" || s.is_empty())
}

/// Mark the sub onboarding nudge as shown / dismissed.
pub fn mark_sub_nudge_shown() -> Result<()> {
    let mut state = State::load()?;
    state.sub_nudge_shown = true;
    state.save()
}

pub fn dismiss_sub_nudge() -> Result<()> {
    let mut state = State::load()?;
    state.sub_nudge_shown = true;
    state.opt_out.sub_nudge = true;
    state.save()
}

fn confirm_default_yes(prompt: &str) -> bool {
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return true;
    }
    eprint!("{prompt} [Y/n] ");
    let _ = std::io::Write::flush(&mut std::io::stderr());
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false; // do not enable optional bits the user never confirmed
    }
    let t = line.trim().to_ascii_lowercase();
    t.is_empty() || t == "y" || t == "yes"
}

fn confirm_default_no(prompt: &str) -> bool {
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return false;
    }
    eprint!("{prompt} [y/N] ");
    let _ = std::io::Write::flush(&mut std::io::stderr());
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    let t = line.trim().to_ascii_lowercase();
    t == "y" || t == "yes"
}

/// Download the Linux gnu tray binary next to the CLI.
fn download_linux_tray() -> Result<PathBuf> {
    #[cfg(not(target_os = "linux"))]
    {
        anyhow::bail!("tray download is only implemented for Linux");
    }
    #[cfg(target_os = "linux")]
    {
        let exe = std::env::current_exe()
            .ok()
            .filter(|p| p.exists())
            .context("current_exe")?;
        let dir = exe.parent().context("exe has no parent")?;
        let dest = dir.join("llmtrim-tray");
        let triple = linux_tray_target_triple()?;
        let tag = format!("v{CURRENT}");
        let url = format!(
            "https://github.com/fkiene/llmtrim/releases/download/{tag}/llmtrim-tray-{triple}.tar.gz"
        );
        let tmp_dir = tempfile_dir()?;
        let tarball = tmp_dir.join("tray.tar.gz");
        download_file(&url, &tarball)?;
        let extracted = extract_tray_tarball_safe(&tarball, &tmp_dir)?;
        if let Err(e) = std::fs::rename(&extracted, &dest) {
            std::fs::copy(&extracted, &dest).with_context(|| {
                format!("install tray to {} (rename failed: {e})", dest.display())
            })?;
            let _ = std::fs::remove_file(&extracted);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&dest)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&dest, perms)?;
        }
        let _ = std::fs::remove_dir_all(&tmp_dir);
        Ok(dest)
    }
}

/// Release asset triple for the prebuilt Linux tray, or an error if unsupported.
#[cfg(target_os = "linux")]
fn linux_tray_target_triple() -> Result<&'static str> {
    match std::env::consts::ARCH {
        "x86_64" => Ok("x86_64-unknown-linux-gnu"),
        "aarch64" => Ok("aarch64-unknown-linux-gnu"),
        other => anyhow::bail!("no prebuilt llmtrim-tray for arch {other}"),
    }
}

#[cfg(target_os = "linux")]
fn tempfile_dir() -> Result<PathBuf> {
    let base = std::env::temp_dir().join(format!("llmtrim-tray-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

#[cfg(target_os = "linux")]
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let mut resp = ureq::get(url)
        .config()
        .timeout_global(Some(std::time::Duration::from_secs(60)))
        .http_status_as_error(false)
        .build()
        .call()
        .with_context(|| format!("GET {url}"))?;
    let code = resp.status().as_u16();
    if !(200..300).contains(&code) {
        anyhow::bail!("download returned HTTP {code}");
    }
    use std::io::Read;
    let mut reader = resp.body_mut().as_reader();
    let mut file =
        std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    // Bound size (~64 MiB) so a runaway response cannot fill the disk.
    let mut limited = Read::take(&mut reader, 64 * 1024 * 1024);
    let n = std::io::copy(&mut limited, &mut file).context("write download")?;
    if n >= 64 * 1024 * 1024 {
        anyhow::bail!("download exceeded 64 MiB size limit");
    }
    Ok(())
}

/// List archive members, reject path traversal, extract into `tmp_dir`, return path to llmtrim-tray.
#[cfg(target_os = "linux")]
fn extract_tray_tarball_safe(tarball: &Path, tmp_dir: &Path) -> Result<PathBuf> {
    let list = std::process::Command::new("tar")
        .args(["-tzf"])
        .arg(tarball)
        .output()
        .context("run tar -tzf")?;
    if !list.status.success() {
        anyhow::bail!("tar -tzf failed listing tray archive");
    }
    let listing = String::from_utf8_lossy(&list.stdout);
    let mut members = Vec::new();
    for line in listing.lines() {
        let name = line.trim().trim_start_matches("./");
        if name.is_empty() {
            continue;
        }
        if name.starts_with('/') || name.contains("..") {
            anyhow::bail!("tray archive has unsafe path: {name}");
        }
        members.push(name.to_string());
    }
    let has_tray = members
        .iter()
        .any(|m| Path::new(m).file_name().and_then(|f| f.to_str()) == Some("llmtrim-tray"));
    if !has_tray {
        anyhow::bail!("tray archive does not contain llmtrim-tray");
    }
    let status = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(tarball)
        .arg("-C")
        .arg(tmp_dir)
        .status()
        .context("run tar extract")?;
    if !status.success() {
        anyhow::bail!("tar extract failed");
    }
    // Find extracted binary (top-level or one directory deep).
    let candidates = [
        tmp_dir.join("llmtrim-tray"),
        tmp_dir.join("llmtrim-tray").join("llmtrim-tray"),
    ];
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
    }
    for ent in std::fs::read_dir(tmp_dir)? {
        let ent = ent?;
        let p = ent.path();
        if p.file_name().and_then(|n| n.to_str()) == Some("llmtrim-tray") && p.is_file() {
            return Ok(p);
        }
        if p.is_dir() {
            let nested = p.join("llmtrim-tray");
            if nested.is_file() {
                return Ok(nested);
            }
        }
    }
    anyhow::bail!("could not locate extracted llmtrim-tray");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_roundtrip() {
        let dir = std::env::temp_dir().join(format!("llmtrim-ensure-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(STATE_FILE);
        let mut s = State::default();
        s.opt_out.guard = true;
        s.last_ensured_version = Some("0.10.2".into());
        save_at(&path, &s).unwrap();
        let loaded = load_at(&path).unwrap();
        assert_eq!(loaded, s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opt_out_defaults_are_false() {
        let s = State::default();
        assert!(!s.opt_out.statusline);
        assert!(!s.opt_out.guard);
        assert!(!s.opt_out.window_sub);
    }
}
