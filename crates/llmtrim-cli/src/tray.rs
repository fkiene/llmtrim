//! Launch the desktop tray app.
//!
//! The tray is a separate GUI binary (`llmtrim-tray`) shipped alongside the CLI
//! by the desktop bundles (npm / Homebrew / `cargo install` on a desktop OS).
//! This module locates that sibling binary and starts it; it is also the seam
//! the tray autostart entry ([`crate::autostart::configure_tray`]) registers.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Filename of the tray GUI binary next to the CLI (`.exe` on Windows).
const TRAY_BIN: &str = if cfg!(windows) {
    "llmtrim-tray.exe"
} else {
    "llmtrim-tray"
};

/// Resolve the tray GUI binary, or `None` if it isn't installed.
///
/// The bundles install `llmtrim-tray` in the same directory as `llmtrim`, so we
/// look next to the running executable. A plain `cargo install llmtrim` on Linux
/// (no GUI feature) never builds it, hence the `Option`.
pub fn tray_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    resolve_tray_binary(exe.parent()?)
}

/// Inner seam for [`tray_binary`]: look for the tray binary in `dir`. Tested
/// directly with a temp dir so the lookup needs no real install.
fn resolve_tray_binary(dir: &Path) -> Option<PathBuf> {
    let candidate = dir.join(TRAY_BIN);
    candidate.is_file().then_some(candidate)
}

/// `llmtrim tray`: start the tray app, leaving it running after the CLI exits.
pub fn run() -> Result<()> {
    let Some(bin) = tray_binary() else {
        anyhow::bail!(
            "the llmtrim tray app isn't installed next to this binary.\n\
             It ships with the desktop bundles — install with `npm i -g @llmtrim/cli` \
             or `brew install fkiene/tap/llmtrim` and try again."
        );
    };
    // Inherit stderr so the tray's own startup errors (e.g. missing WebKit
    // libraries on a minimal Linux host) reach the terminal instead of vanishing
    // — `spawn` returns Ok the moment the child forks, even if it exits at once.
    std::process::Command::new(&bin)
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to launch the tray app at {}", bin.display()))?;
    Ok(())
}

/// Launch the tray as a detached background process, discarding its stdio. Meant
/// for launching from inside the status dashboard: the terminal is in raw-mode
/// alt-screen there, so any inherited child output would corrupt the display.
/// A no-op (still `Ok`) when the tray isn't installed.
pub fn launch_detached() -> Result<()> {
    launch_detached_from(tray_binary())
}

/// Inner seam for [`launch_detached`], tested with an explicit path: given the resolved
/// tray binary (or `None`), launch it detached or no-op.
fn launch_detached_from(bin: Option<PathBuf>) -> Result<()> {
    let Some(bin) = bin else {
        return Ok(());
    };
    std::process::Command::new(&bin)
        // Fully detach from the dashboard's terminal: a background GUI has no use for the
        // parent's stdin, and inheriting it would leave a child holding the raw-mode TTY.
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("failed to launch the tray app at {}", bin.display()))?;
    Ok(())
}

/// Marker recording that the one-time "desktop tray is available" hint has been
/// shown, so it appears at most once per install.
const NUDGE_MARKER: &str = "tray-nudged";

/// One-time discoverability hint for the desktop tray, meant to print after the
/// interceptor starts. Returns the message to show, or `None` when there's
/// nothing to say: the tray binary isn't installed next to the CLI (a
/// cargo-only install that can't run it stays quiet), or the hint was already
/// shown once. Best-effort — a failure to record the marker is never surfaced;
/// at worst the hint repeats.
pub fn nudge_once() -> Option<String> {
    let dir = crate::daemon::home_dir().ok()?;
    nudge_in(&dir, tray_binary().is_some())
}

/// Inner seam for [`nudge_once`], tested with a temp dir: given the state
/// directory and whether the tray is installed, decide whether to show the hint
/// and record that it was shown.
fn nudge_in(state_dir: &Path, tray_present: bool) -> Option<String> {
    if !tray_present {
        return None;
    }
    let marker = state_dir.join(NUDGE_MARKER);
    if marker.exists() {
        return None;
    }
    let _ = std::fs::create_dir_all(state_dir);
    let _ = std::fs::write(&marker, b"1\n");
    Some("Desktop tray available: run `llmtrim tray` to watch savings in your menu bar.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(suffix: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("llmtrim-tray-test-{}-{suffix}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolve_returns_none_when_absent() {
        let dir = TempDir::new("absent");
        assert!(resolve_tray_binary(dir.path()).is_none());
    }

    #[test]
    fn resolve_finds_sibling_binary() {
        let dir = TempDir::new("present");
        let bin = dir.path().join(TRAY_BIN);
        std::fs::write(&bin, b"#!/bin/sh\n").expect("write fake binary");
        assert_eq!(resolve_tray_binary(dir.path()), Some(bin));
    }

    #[test]
    fn resolve_ignores_a_directory_named_like_the_binary() {
        // A directory matching the binary name must not be treated as the app.
        let dir = TempDir::new("dir-named-bin");
        std::fs::create_dir_all(dir.path().join(TRAY_BIN)).expect("create dir");
        assert!(resolve_tray_binary(dir.path()).is_none());
    }

    #[test]
    fn launch_detached_is_a_no_op_when_not_installed() {
        // No resolved binary: returns Ok without spawning anything.
        assert!(launch_detached_from(None).is_ok());
    }

    #[test]
    fn nudge_stays_quiet_when_tray_not_installed() {
        let dir = TempDir::new("nudge-absent");
        assert!(nudge_in(dir.path(), false).is_none());
        // Nothing recorded, so a cargo-only user never accretes a stray marker.
        assert!(!dir.path().join(NUDGE_MARKER).exists());
    }

    #[test]
    fn nudge_shows_once_then_goes_silent() {
        let dir = TempDir::new("nudge-once");
        let first = nudge_in(dir.path(), true);
        assert!(first.is_some_and(|m| m.contains("llmtrim tray")));
        assert!(dir.path().join(NUDGE_MARKER).exists());
        // Second call sees the marker and says nothing.
        assert!(nudge_in(dir.path(), true).is_none());
    }
}
