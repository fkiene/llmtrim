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
}
