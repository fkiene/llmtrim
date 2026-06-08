//! `llmtrim setup` — the one-command bootstrap. llmtrim is *only* a MITM proxy, so
//! integration is purely at the environment level: it ensures the local CA, writes a managed
//! block to your shell profile (`HTTPS_PROXY` + `NODE_EXTRA_CA_CERTS`) so every
//! shell-launched tool routes through the interceptor and trusts the CA — **no IDE settings
//! touched, no sudo** — enables run-at-login, and starts the daemon.
//!
//! Best-effort and idempotent: a step that fails warns and the rest proceeds.

use std::path::PathBuf;

use anyhow::{Context, Result};

const BEGIN: &str = "# >>> llmtrim >>>";
const END: &str = "# <<< llmtrim <<<";

pub fn run(port: u16) -> Result<()> {
    // 1. Local CA (generated on first run, name-constrained to LLM domains).
    crate::serve::ensure_ca()?;
    let ca = crate::serve::ca_cert_path()?.to_string_lossy().to_string();
    let proxy = format!("http://127.0.0.1:{port}");
    println!("✓ Local CA: {ca}");

    // 2. Route + trust at the environment level (shell profile managed block).
    match write_profile_block(&proxy, &ca)? {
        Some(path) => {
            println!("✓ Updated {} with the interceptor env:", path.display());
            println!("    HTTPS_PROXY={proxy}");
            println!("    NODE_EXTRA_CA_CERTS={ca}");
        }
        None => {
            #[cfg(windows)]
            {
                println!("• No PowerShell profile found — set these yourself:");
                println!("    $env:HTTPS_PROXY = \"{proxy}\"");
                println!("    $env:NODE_EXTRA_CA_CERTS = \"{ca}\"");
            }
            #[cfg(not(windows))]
            {
                println!("• No shell profile found — export these yourself:");
                println!("    export HTTPS_PROXY={proxy}");
                println!("    export NODE_EXTRA_CA_CERTS={ca}");
            }
        }
    }

    // 3. Run at login (systemd / launchd / Windows, via auto-launch).
    if let Err(e) = crate::autostart::configure(true, port) {
        eprintln!("• Autostart not enabled: {e}");
    }

    // 4. (Re)start the interceptor. Stop any existing daemon first so re-running `setup`
    //    after an update actually goes live — otherwise the old process keeps serving the
    //    old binary until a manual restart (the silent-stale-update trap).
    let _ = crate::daemon::stop();
    match crate::daemon::spawn_detached(port) {
        Ok(pid) => println!("✓ Interceptor running (pid {pid}, port {port})."),
        Err(e) => eprintln!("• Daemon not started: {e}"),
    }

    // The managed block only lands in *future* shells — already-running tools (editors,
    // Claude Code, open terminals) keep their old environment until relaunched. Spell that
    // out: it's the #1 "why don't I see any traffic?" confusion.
    let check = if cfg!(windows) {
        "echo $env:HTTPS_PROXY"
    } else {
        "echo $HTTPS_PROXY"
    };
    println!("\nDone. The interceptor is running.");
    println!(
        "Note: only programs started from a new shell pick up the proxy env. Already-running\n\
         tools (your editor, Claude Code, open terminals) keep their old environment until\n\
         relaunched. To route one through llmtrim:"
    );
    println!("  1. open a new terminal (or re-source your shell profile)");
    println!("  2. verify it took:  {check}  →  {proxy}");
    println!("  3. launch your tool from that shell");
    println!("Watch savings: llmtrim status");
    #[cfg(windows)]
    println!(
        "(For non-PowerShell / GUI apps, trust the CA system-wide: \
         certutil -addstore -user Root \"{ca}\" — or see `llmtrim ca`.)"
    );
    #[cfg(not(windows))]
    println!(
        "(GUI apps that ignore the shell env need the CA trusted system-wide — see `llmtrim ca`.)"
    );
    Ok(())
}

/// `llmtrim uninstall` — the transparent inverse of `setup`: stop the daemon, disable
/// autostart, strip the shell-profile block, and remove the CA + state (and, unless told
/// otherwise, the binary itself). Each action is printed; nothing is silent.
pub fn uninstall(purge: bool, keep_binary: bool) -> Result<()> {
    // 1. Stop the running daemon.
    match crate::daemon::stop() {
        Ok(Some(pid)) => println!("✓ Stopped interceptor (pid {pid})."),
        Ok(None) => println!("• No daemon was running."),
        Err(e) => eprintln!("• Could not stop daemon: {e}"),
    }

    // 2. Disable run-at-login (matched by app name, so the port is irrelevant here).
    match crate::autostart::configure(false, 8787) {
        Ok(()) => println!("✓ Autostart disabled."),
        Err(e) => eprintln!("• Autostart not changed: {e}"),
    }

    // 3. Remove the managed env block from the shell profile.
    match remove_profile_block()? {
        Some(path) => println!("✓ Removed the env block from {}.", path.display()),
        None => println!("• No shell-profile block to remove."),
    }

    // 4. Remove the CA + daemon state (~/.llmtrim).
    let home = crate::daemon::home_dir()?;
    if home.exists() {
        std::fs::remove_dir_all(&home)
            .with_context(|| format!("failed to remove {}", home.display()))?;
        println!("✓ Removed {} (CA, key, daemon state).", home.display());
    } else {
        println!("• No state directory to remove.");
    }

    // 5. The savings ledger — kept by default (it's your history), removed with --purge.
    match crate::tracking::db_path() {
        Ok(db) if db.exists() && purge => {
            std::fs::remove_file(&db).ok();
            println!("✓ Removed the savings ledger {}.", db.display());
        }
        Ok(db) if db.exists() => {
            println!(
                "• Kept the savings ledger {} (use --purge to remove).",
                db.display()
            );
        }
        _ => {}
    }

    // 6. The binary itself (Unix can unlink a running executable; Windows can't).
    if keep_binary {
        println!("• Kept the binary.");
    } else if let Ok(exe) = std::env::current_exe() {
        #[cfg(unix)]
        {
            std::fs::remove_file(&exe).ok();
            println!("✓ Removed the binary {}.", exe.display());
        }
        #[cfg(not(unix))]
        {
            println!("• Remove the binary manually: {}", exe.display());
        }
    }

    println!("\nDone. Open a new shell so the environment changes take effect.");
    println!("(If you trusted the CA system-wide manually, remove it from your OS trust store.)");
    Ok(())
}

/// Strip the llmtrim managed block from the shell profile, if present.
fn remove_profile_block() -> Result<Option<PathBuf>> {
    let Some((path, _)) = profile_target() else {
        return Ok(None);
    };
    let Ok(existing) = std::fs::read_to_string(&path) else {
        return Ok(None);
    };
    if !existing.contains(BEGIN) {
        return Ok(None);
    }
    std::fs::write(&path, strip_block(&existing))
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(Some(path))
}

/// Is the llmtrim env block present in the shell profile? Used to warn that stopping
/// the daemon while `HTTPS_PROXY` still points at it will break the client's HTTPS.
pub fn profile_has_block() -> bool {
    profile_target()
        .and_then(|(p, _)| std::fs::read_to_string(p).ok())
        .map(|t| t.contains(BEGIN))
        .unwrap_or(false)
}

/// Which shell dialect the profile uses, so the managed block is written in its native syntax.
/// Each variant is constructed on only one platform (`Posix` off-Windows, `PowerShell` on
/// Windows), yet both arms of `env_block` are compiled and unit-tested everywhere so the
/// formatting is verifiable on either OS — hence the unconditional `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Syntax {
    Posix,
    PowerShell,
}

/// The profile file to write the managed env block into, and the syntax it uses. Unix: the
/// `$SHELL` rc file (`export`). Windows: the current-user PowerShell profile (`$env:`).
fn profile_target() -> Option<(PathBuf, Syntax)> {
    #[cfg(not(windows))]
    {
        let home = std::env::var("HOME").ok()?;
        let shell = std::env::var("SHELL").unwrap_or_default();
        let file = if shell.ends_with("zsh") {
            ".zshrc"
        } else if shell.ends_with("bash") {
            ".bashrc"
        } else {
            ".profile"
        };
        Some((PathBuf::from(home).join(file), Syntax::Posix))
    }
    #[cfg(windows)]
    {
        powershell_profile().map(|p| (p, Syntax::PowerShell))
    }
}

/// Resolve `$PROFILE.CurrentUserAllHosts` (handles PowerShell 5 vs 7 and a redirected/OneDrive
/// `Documents`), falling back to the conventional location if PowerShell can't be queried.
#[cfg(windows)]
fn powershell_profile() -> Option<PathBuf> {
    if let Ok(out) = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", "$PROFILE.CurrentUserAllHosts"])
        .output()
    {
        let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    let up = std::env::var("USERPROFILE").ok()?;
    Some(
        PathBuf::from(up)
            .join("Documents")
            .join("PowerShell")
            .join("profile.ps1"),
    )
}

/// The managed env block, in the profile's native syntax. Both variants are unit-tested.
fn env_block(proxy: &str, ca: &str, syntax: Syntax) -> String {
    match syntax {
        Syntax::Posix => format!(
            "{BEGIN}\n\
             export HTTPS_PROXY=\"{proxy}\"\n\
             export HTTP_PROXY=\"{proxy}\"\n\
             export NODE_EXTRA_CA_CERTS=\"{ca}\"\n\
             {END}\n"
        ),
        Syntax::PowerShell => format!(
            "{BEGIN}\n\
             $env:HTTPS_PROXY = \"{proxy}\"\n\
             $env:HTTP_PROXY = \"{proxy}\"\n\
             $env:NODE_EXTRA_CA_CERTS = \"{ca}\"\n\
             {END}\n"
        ),
    }
}

/// Replace (or append) the llmtrim managed block in the shell profile. Idempotent — a
/// re-run updates the existing block rather than stacking duplicates.
fn write_profile_block(proxy: &str, ca: &str) -> Result<Option<PathBuf>> {
    let Some((path, syntax)) = profile_target() else {
        return Ok(None);
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent); // the PowerShell profile dir may not exist yet
    }
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut base = strip_block(&existing);
    if !base.is_empty() && !base.ends_with('\n') {
        base.push('\n');
    }
    let block = env_block(proxy, ca, syntax);
    std::fs::write(&path, format!("{base}{block}"))
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(Some(path))
}

/// Remove any existing llmtrim managed block (between the markers, inclusive).
fn strip_block(s: &str) -> String {
    let mut out = String::new();
    let mut skip = false;
    for line in s.lines() {
        match line.trim() {
            BEGIN => skip = true,
            END => skip = false,
            _ if !skip => {
                out.push_str(line);
                out.push('\n');
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_block_removes_managed_section_only() {
        let input = format!("keep1\n{BEGIN}\nexport X=1\n{END}\nkeep2\n");
        let out = strip_block(&input);
        assert_eq!(out, "keep1\nkeep2\n");
    }

    #[test]
    fn strip_block_is_noop_without_markers() {
        assert_eq!(strip_block("a\nb\n"), "a\nb\n");
    }

    #[test]
    fn env_block_posix_uses_export() {
        let b = env_block("http://127.0.0.1:8787", "/home/u/ca.pem", Syntax::Posix);
        assert!(b.contains("export HTTPS_PROXY=\"http://127.0.0.1:8787\""));
        assert!(b.contains("export NODE_EXTRA_CA_CERTS=\"/home/u/ca.pem\""));
        assert!(b.starts_with(BEGIN) && b.trim_end().ends_with(END));
    }

    #[test]
    fn env_block_powershell_uses_env_assignment() {
        let b = env_block(
            "http://127.0.0.1:8787",
            "C:\\Users\\u\\ca.pem",
            Syntax::PowerShell,
        );
        assert!(b.contains("$env:HTTPS_PROXY = \"http://127.0.0.1:8787\""));
        assert!(b.contains("$env:NODE_EXTRA_CA_CERTS = \"C:\\Users\\u\\ca.pem\""));
        assert!(!b.contains("export ")); // no posix syntax leaked in
    }

    #[test]
    fn strip_block_reverses_powershell_block() {
        let withblock = format!("keep\n{}", env_block("p", "c", Syntax::PowerShell));
        assert_eq!(strip_block(&withblock), "keep\n");
    }
}
