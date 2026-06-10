//! Run the interceptor at login.
//!
//! **Windows**: one canonical entry under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`
//! (via `winreg`). Enabling also purges the artifacts the *old* build left — it used the
//! `auto-launch` crate, whose default "Dynamic" mode writes to `HKLM\...\Run` (machine-wide,
//! when it can elevate) plus a `StartupApproved\Run` enable-toggle. Without that cleanup an
//! upgraded user gets the HKLM entry *and* our HKCU entry firing at login — two daemons racing
//! for the port. So we collapse everything to the single HKCU entry.
//!
//! **macOS / Linux**: the `auto-launch` crate (launchd agent / XDG `.desktop` entry).

use anyhow::{Context, Result};

/// Enable (or disable) running `llmtrim serve --port <port>` at login. Silent on
/// success — callers (the `autostart` command, `setup`, `uninstall`) own the
/// messaging so each flow keeps its own voice.
pub fn configure(enable: bool, port: u16) -> Result<()> {
    #[cfg(windows)]
    {
        configure_windows(enable, port)
    }
    #[cfg(not(windows))]
    {
        configure_auto_launch(enable, port)
    }
}

// ── Windows: HKCU\Software\Microsoft\Windows\CurrentVersion\Run ─────────────────

#[cfg(windows)]
const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
#[cfg(windows)]
const STARTUP_APPROVED_KEY: &str =
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run";
#[cfg(windows)]
const VALUE_NAME: &str = "llmtrim";

#[cfg(windows)]
fn configure_windows(enable: bool, port: u16) -> Result<()> {
    use winreg::RegKey;
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};

    if enable {
        let exe = std::env::current_exe().context("could not find the llmtrim executable")?;
        let (key, _) = RegKey::predef(HKEY_CURRENT_USER)
            .create_subkey_with_flags(RUN_KEY, KEY_READ | KEY_WRITE)
            .with_context(|| format!("failed to open HKCU\\{RUN_KEY}"))?;
        let cmd = format!("\"{}\" serve --port {} --supervised", exe.display(), port);
        key.set_value(VALUE_NAME, &cmd)
            .context("failed to set llmtrim autostart in the registry Run key")?;
        // Collapse to this single entry: remove any legacy auto-launch leftovers (the HKLM
        // Run entry it wrote under elevation, and the StartupApproved toggles in either hive)
        // so login starts exactly one daemon, not two.
        purge_legacy_autostart();
    } else {
        // Disable: clear our entry *and* every legacy location, so uninstall leaves nothing
        // that revives the daemon at next login.
        remove_autostart_everywhere();
    }
    Ok(())
}

/// Delete the `llmtrim` value under `subkey` in `hive`, best-effort. Opening for write can fail
/// without admin (HKLM) — that's fine, we just couldn't find/clean it. A missing value is fine
/// too (idempotent). Never errors: cleanup must never block enable/disable.
#[cfg(windows)]
fn best_effort_delete(hive: winreg::HKEY, subkey: &str) {
    use winreg::RegKey;
    use winreg::enums::KEY_WRITE;
    if let Ok(key) = RegKey::predef(hive).open_subkey_with_flags(subkey, KEY_WRITE) {
        let _ = key.delete_value(VALUE_NAME);
    }
}

/// Remove every autostart artifact *except* our canonical HKCU Run entry — i.e. the old
/// `auto-launch` build's HKLM Run entry and its StartupApproved toggles in both hives.
#[cfg(windows)]
fn purge_legacy_autostart() {
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    best_effort_delete(HKEY_LOCAL_MACHINE, RUN_KEY);
    best_effort_delete(HKEY_CURRENT_USER, STARTUP_APPROVED_KEY);
    best_effort_delete(HKEY_LOCAL_MACHINE, STARTUP_APPROVED_KEY);
}

/// Remove the autostart entry from every location we (or the old build) could have written:
/// the Run key and the StartupApproved toggle, in both HKCU and HKLM.
#[cfg(windows)]
fn remove_autostart_everywhere() {
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    best_effort_delete(HKEY_CURRENT_USER, RUN_KEY);
    best_effort_delete(HKEY_LOCAL_MACHINE, RUN_KEY);
    best_effort_delete(HKEY_CURRENT_USER, STARTUP_APPROVED_KEY);
    best_effort_delete(HKEY_LOCAL_MACHINE, STARTUP_APPROVED_KEY);
}

// ── macOS / Linux: auto-launch crate ────────────────────────────────────────────

/// The XDG autostart directory relative to a home base (Linux only).
#[cfg(target_os = "linux")]
fn xdg_autostart_dir(base: &std::path::Path) -> std::path::PathBuf {
    base.join(".config").join("autostart")
}

/// The content of the `.desktop` entry for a given `exe` path and `port`.
#[cfg(target_os = "linux")]
fn desktop_entry(exe: &std::path::Path, port: u16) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Version=1.0\n\
         Name=llmtrim\n\
         Comment=llmtrim startup script\n\
         Exec={} serve --port {} --supervised\n\
         StartupNotify=false\n\
         Terminal=false",
        exe.display(),
        port
    )
}

/// Inner seam: enable or disable the XDG autostart entry using `base` as the home directory
/// instead of the real `$HOME`. Tests pass a temp dir; production passes the actual home.
/// On Linux only — macOS goes through `configure_auto_launch` directly without a base seam.
#[cfg(target_os = "linux")]
fn configure_in(enable: bool, port: u16, base: &std::path::Path) -> Result<()> {
    let dir = xdg_autostart_dir(base);
    let file = dir.join("llmtrim.desktop");

    if enable {
        let exe = std::env::current_exe().context("could not find the llmtrim executable")?;
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        std::fs::write(&file, desktop_entry(&exe, port))
            .with_context(|| format!("failed to write {}", file.display()))?;
    } else if file.exists() {
        std::fs::remove_file(&file)
            .with_context(|| format!("failed to remove {}", file.display()))?;
        // Already absent → no-op (idempotent disable).
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn configure_auto_launch(enable: bool, port: u16) -> Result<()> {
    let home = std::env::var("HOME")
        .context("HOME is not set — cannot determine XDG autostart directory")?;
    configure_in(enable, port, std::path::Path::new(&home))
}

#[cfg(all(not(windows), not(target_os = "linux")))]
fn configure_auto_launch(enable: bool, port: u16) -> Result<()> {
    use auto_launch::AutoLaunchBuilder;

    let exe = std::env::current_exe().context("could not find the llmtrim executable")?;
    let path = exe.to_string_lossy();
    let port_arg = port.to_string();

    let auto = AutoLaunchBuilder::new()
        .set_app_name("llmtrim")
        .set_app_path(path.as_ref())
        .set_args(&["serve", "--port", port_arg.as_str(), "--supervised"])
        .build()
        .map_err(|e| anyhow::anyhow!("failed to configure autostart: {e}"))?;

    if enable {
        auto.enable()
            .map_err(|e| anyhow::anyhow!("failed to enable autostart: {e}"))?;
    } else {
        auto.disable()
            .map_err(|e| anyhow::anyhow!("failed to disable autostart: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Minimal RAII temp-dir guard — same approach as setup.rs tests.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(suffix: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "llmtrim-autostart-test-{}-{}",
                std::process::id(),
                suffix
            ));
            std::fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // These tests exercise configure_in directly, which is only compiled on Linux.
    // On Windows the cfg(target_os = "linux") gate keeps them out; the Windows
    // registry arm has its own scratch-key test in setup.rs.
    #[cfg(target_os = "linux")]
    #[test]
    fn configure_in_install_creates_desktop_file_with_expected_content() {
        let dir = TempDir::new("install");
        let base = dir.path();

        configure_in(true, 8787, base).expect("configure_in enable");

        let file = base
            .join(".config")
            .join("autostart")
            .join("llmtrim.desktop");
        assert!(file.exists(), ".desktop file not created");

        let content = std::fs::read_to_string(&file).expect("read .desktop");
        assert!(
            content.contains("[Desktop Entry]"),
            "missing [Desktop Entry]"
        );
        assert!(content.contains("Name=llmtrim"), "missing Name=");
        assert!(
            content.contains("serve --port 8787 --supervised"),
            "missing serve invocation with port"
        );
        assert!(content.contains("Terminal=false"), "missing Terminal=false");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn configure_in_install_twice_is_idempotent() {
        let dir = TempDir::new("idempotent");
        let base = dir.path();

        configure_in(true, 8787, base).expect("first enable");
        configure_in(true, 8787, base).expect("second enable");

        // Exactly one .desktop file should exist (no duplicate, no error).
        let autostart_dir = base.join(".config").join("autostart");
        let count = std::fs::read_dir(&autostart_dir)
            .expect("read autostart dir")
            .filter_map(|e| e.ok())
            .count();
        assert_eq!(count, 1, "expected exactly 1 .desktop file, found {count}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn configure_in_remove_deletes_desktop_file() {
        let dir = TempDir::new("remove");
        let base = dir.path();

        configure_in(true, 8787, base).expect("enable");
        let file = base
            .join(".config")
            .join("autostart")
            .join("llmtrim.desktop");
        assert!(file.exists(), "file should exist before disable");

        configure_in(false, 8787, base).expect("disable");
        assert!(!file.exists(), ".desktop file still present after disable");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn configure_in_remove_when_absent_is_ok_noop() {
        let dir = TempDir::new("remove-absent");
        let base = dir.path();

        // Disable without ever enabling — must return Ok without error.
        configure_in(false, 8787, base).expect("disable when absent must be Ok");

        // The autostart dir itself may or may not exist; either way, no .desktop file.
        let file = base
            .join(".config")
            .join("autostart")
            .join("llmtrim.desktop");
        assert!(!file.exists(), ".desktop file should not exist");
    }
}
