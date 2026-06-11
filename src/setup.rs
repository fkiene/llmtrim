//! `llmtrim setup` — the one-command bootstrap. llmtrim is *only* a MITM proxy, so
//! integration is purely at the environment level: it ensures the local CA, then sets
//! `HTTPS_PROXY` + `NODE_EXTRA_CA_CERTS` for the user (POSIX: a managed shell-profile
//! block; Windows: `HKCU\Environment`) so every newly-launched tool routes through the
//! interceptor and trusts the CA — **no IDE settings touched, no sudo** — enables
//! run-at-login, and starts the daemon.
//!
//! Best-effort and idempotent: a step that fails warns and the rest proceeds.

use std::net::{Ipv4Addr, TcpListener};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::ui::{self, Tone};

const BEGIN: &str = "# >>> llmtrim >>>";
const END: &str = "# <<< llmtrim <<<";

/// Default interceptor port; the scan for a free port starts here. Chosen to be unassigned
/// by IANA and below the OS ephemeral range (so it isn't grabbed as a transient client port),
/// avoiding clashes with common dev servers. Single source of truth — `main.rs` references it.
pub const DEFAULT_PORT: u16 = 43117;

/// First loopback port that actually binds, scanning `start..=start+span`. A successful bind
/// (immediately dropped) proves the port is usable *right now*; because we accept only `Ok`,
/// this also skips Windows reserved/excluded ranges, which fail the bind with `PermissionDenied`
/// rather than `AddrInUse`. Probes `127.0.0.1` to match exactly what `serve` binds. `None` if the
/// whole window is unusable.
fn first_free_port(start: u16, span: u16) -> Option<u16> {
    (start..=start.saturating_add(span))
        .find(|&p| TcpListener::bind((Ipv4Addr::LOCALHOST, p)).is_ok())
}

/// Outcome of resolving which port to wire: a definite port to use, or a starting point to
/// scan from for a free one. Split out so the precedence is pure and unit-testable.
#[derive(Debug, PartialEq, Eq)]
enum PortChoice {
    /// Use exactly this port (caller does not scan).
    Use(u16),
    /// No port is pinned — scan upward from here for the first free one.
    ScanFrom(u16),
}

/// Decide the interceptor port, in precedence order — *without* scanning, so it's pure:
///
/// 1. an explicit `--port` (honor the user verbatim),
/// 2. the port a live llmtrim daemon is already serving (reuse it — never migrate a running
///    proxy off the port its clients point at),
/// 3. the port already wired into the environment (`HTTPS_PROXY`), so re-running converges
///    on what existing shells expect,
/// 4. otherwise scan from the default.
///
/// Steps 2–3 are why re-running `setup` is now idempotent: the old code scanned from 8787
/// every time, and since the running daemon *held* 8787 the scan skipped to 8788 — each
/// re-run drifted the port upward and rewrote the env/autostart to match, breaking every
/// already-launched client. Reusing the live/recorded port stops that.
fn choose_port(explicit: Option<u16>, running: Option<u16>, configured: Option<u16>) -> PortChoice {
    if let Some(p) = explicit.or(running).or(configured) {
        PortChoice::Use(p)
    } else {
        PortChoice::ScanFrom(DEFAULT_PORT)
    }
}

/// Resolve the port to wire, scanning for a free one only when nothing is pinned (the
/// first-install case). `running` is the live daemon's port, if any. Used by `setup` and the
/// `start` command so both agree on the same port without drifting.
pub fn resolve_port(explicit: Option<u16>, running: Option<u16>) -> Result<u16> {
    match choose_port(explicit, running, configured_port()) {
        PortChoice::Use(p) => Ok(p),
        PortChoice::ScanFrom(start) => first_free_port(start, 64)
            .with_context(|| format!("no free port in {start}..={}", start.saturating_add(64))),
    }
}

/// Extract the port from a local proxy URL embedded anywhere in `text` — i.e. the number
/// right after `127.0.0.1:`. Lets us read back the port we previously wired into the env
/// (the shell-profile block on POSIX, `HKCU\Environment\HTTPS_PROXY` on Windows). Pure.
pub(crate) fn parse_proxy_port(text: &str) -> Option<u16> {
    let after = text.split("127.0.0.1:").nth(1)?;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The interceptor port currently wired into the environment, if any — read from the live env
/// source for this platform (POSIX: the shell-profile block; Windows: `HKCU\Environment`).
/// Public so `status`/`doctor` can compare the wired port against the daemon's.
pub fn configured_port() -> Option<u16> {
    #[cfg(windows)]
    {
        user_env_key()
            .ok()
            .and_then(|env| env.get_value::<String, _>("HTTPS_PROXY").ok())
            .and_then(|v| parse_proxy_port(&v))
    }
    #[cfg(not(windows))]
    {
        profile_target()
            .and_then(|(p, _)| std::fs::read_to_string(p).ok())
            .and_then(|t| parse_proxy_port(&t))
    }
}

pub fn run(requested: Option<u16>) -> Result<()> {
    let color = ui::color_stdout();

    // 0. Resolve the port *once*, here, before anything is wired. The port is a contract
    //    between three parties that must agree: the profile's HTTPS_PROXY (clients), the
    //    autostart entry (`serve --port N` at login), and the daemon that binds it. We reuse
    //    the port a live daemon already serves (or one already wired into the env) instead of
    //    scanning blindly — otherwise the running daemon holds 8787, the scan drifts to 8788,
    //    and every re-run rewrites the env/autostart to a new port, breaking running clients.
    let running = crate::daemon::running();
    let running_port = running.as_ref().map(|s| s.port);
    let configured = configured_port();
    let pinned = requested.or(running_port).or(configured);
    let port = match choose_port(requested, running_port, configured) {
        PortChoice::Use(p) => p,
        PortChoice::ScanFrom(start) => first_free_port(start, 64)
            .with_context(|| format!("no free port in {start}..={}", start.saturating_add(64)))?,
    };
    // Only chatter about the port when we had to pick one nobody asked for (first install,
    // default busy). When we're reusing a pinned port, silence is correct.
    if pinned.is_none() && port != DEFAULT_PORT {
        println!(
            "{}",
            ui::note(color, &format!("Port {DEFAULT_PORT} busy — using {port}."))
        );
    }

    // Steps are collected as checklist rows and rendered as one summary panel at the
    // end; soft failures become `⚠` rows instead of stderr asides, so the user sees
    // one coherent report.
    let mut rows: Vec<(&str, String, String)> = Vec::new();

    // 1. Local CA (generated on first run, name-constrained to LLM domains).
    crate::serve::ensure_ca()?;
    let ca = crate::serve::ca_cert_path()?.to_string_lossy().to_string();
    let proxy = format!("http://127.0.0.1:{port}");
    rows.push((ui::OK, "Local CA".into(), ca.clone()));

    // 2. Route + trust at the environment level.
    //
    // POSIX: a managed block in the shell rc file (`export …`).
    // Windows: the *user environment* in `HKCU\Environment`, NOT a shell profile — a profile
    //   only helps PowerShell, and ExecutionPolicy can stop it loading entirely (the silent
    //   "no traffic" trap). The registry is read by every process at launch (PS5, pwsh7, Git
    //   Bash, cmd, GUI apps alike), independent of any profile running.
    #[cfg(windows)]
    {
        set_user_env(&proxy, &ca)?;
        rows.push((
            ui::OK,
            "Environment".into(),
            "HKCU\\Environment — HTTPS_PROXY + CA trust".into(),
        ));
        // Upgrade path: drop any legacy managed block a previous version wrote to the
        // PowerShell profile, so a dead (possibly ExecutionPolicy-blocked) block isn't
        // left behind.
        if let Ok(paths) = remove_profile_block() {
            for path in paths {
                rows.push((
                    ui::OK,
                    "Profile".into(),
                    format!("legacy env block removed from {}", path.display()),
                ));
            }
        }
        // Tell Explorer to re-read the environment so freshly-launched terminals/editors
        // inherit it without a logout (a raw registry write alone is invisible to running
        // processes).
        broadcast_env_change();
    }
    #[cfg(not(windows))]
    let manual_env = match write_profile_block(&proxy, &ca)? {
        Some(path) => {
            rows.push((
                ui::OK,
                "Profile".into(),
                format!("{} — HTTPS_PROXY + CA trust", path.display()),
            ));
            false
        }
        None => {
            rows.push((
                ui::NOTE,
                "Profile".into(),
                "no shell profile found — set the env yourself (below)".into(),
            ));
            true
        }
    };

    // 3. Run at login (systemd / launchd / Windows, via auto-launch).
    match crate::autostart::configure(true, port) {
        Ok(()) => rows.push((ui::OK, "Autostart".into(), "runs at login".into())),
        Err(e) => rows.push((ui::WARN, "Autostart".into(), format!("not enabled: {e}"))),
    }

    // 4. Reconcile the interceptor. If a healthy daemon is already serving the resolved port,
    //    leave it running — re-running `setup` must not drop in-flight requests (the old code
    //    stopped + respawned unconditionally on every run). Restart only when the port is
    //    changing (explicit `--port`, or self-healing a drifted state) or the daemon is gone —
    //    that also picks up a new binary after an update (the silent-stale-update trap).
    let daemon_ok = match &running {
        Some(state) if state.port == port => {
            rows.push((
                ui::OK,
                "Interceptor".into(),
                format!("already running · pid {} · port {port}", state.pid),
            ));
            true
        }
        _ => {
            let _ = crate::daemon::stop(); // clear a dead/old-port daemon + its pidfile first
            match crate::daemon::spawn_detached(port) {
                Ok(pid) => {
                    rows.push((
                        ui::OK,
                        "Interceptor".into(),
                        format!("running · pid {pid} · port {port}"),
                    ));
                    true
                }
                Err(e) => {
                    rows.push((ui::WARN, "Interceptor".into(), format!("not started: {e}")));
                    false
                }
            }
        }
    };

    print!(
        "{}",
        ui::panel(color, "llmtrim setup", &ui::kv_rows(color, &rows))
    );

    // On Windows the env is written to the registry above, never manually.
    #[cfg(not(windows))]
    if manual_env {
        println!();
        println!("Export these in your shell yourself:");
        println!("    export HTTPS_PROXY={proxy}");
        println!("    export NODE_EXTRA_CA_CERTS={ca}");
    }

    // The env only reaches *future* processes — already-running tools (editors, Claude
    // Code, open terminals) keep their old environment until relaunched. Spell that
    // out: it's the #1 "why don't I see any traffic?" confusion.
    let check = if cfg!(windows) {
        "echo $env:HTTPS_PROXY"
    } else {
        "echo $HTTPS_PROXY"
    };
    println!();
    if daemon_ok {
        println!(
            "{}",
            ui::paint(color, Tone::Bold, "Done — the interceptor is running.")
        );
    } else {
        println!(
            "{}",
            ui::warn(
                color,
                "Setup finished, but the interceptor is not running — see above."
            )
        );
    }
    println!(
        "Only programs started after this pick up the proxy env; already-running\n\
         tools (your editor, Claude Code, open terminals) keep their old environment\n\
         until relaunched. To route one through llmtrim:"
    );
    println!();
    let new_shell = if cfg!(windows) {
        "open a new terminal (any shell — the env is set for your whole user)"
    } else {
        "open a new terminal (or re-source your shell profile)"
    };
    println!("  1. {new_shell}");
    println!("  2. verify it took:  {check}  →  {proxy}");
    println!("  3. launch your tool from that shell");
    println!();
    println!(
        "  {}  llmtrim status",
        ui::paint(color, Tone::Dim, "watch savings")
    );
    #[cfg(windows)]
    println!(
        "{}",
        ui::note(
            color,
            &format!(
                "For GUI apps that pin their own trust store, trust the CA system-wide: \
                 certutil -addstore -user Root \"{ca}\" — or see llmtrim ca."
            )
        )
    );
    #[cfg(not(windows))]
    println!(
        "{}",
        ui::note(
            color,
            "GUI apps that ignore the shell env need the CA trusted system-wide — see llmtrim ca."
        )
    );
    Ok(())
}

/// `llmtrim uninstall` — the transparent inverse of `setup`: stop the daemon, disable
/// autostart, strip the shell-profile block, and remove the CA + state (and, unless told
/// otherwise, the binary itself). Best-effort: a failed step becomes a `⚠` row and the
/// rest proceeds; every action lands in the summary panel, nothing is silent.
pub fn uninstall(purge: bool, keep_binary: bool) -> Result<()> {
    let color = ui::color_stdout();
    let mut rows: Vec<(&str, String, String)> = Vec::new();

    // 1. Stop the running daemon.
    match crate::daemon::stop() {
        Ok(Some(pid)) => rows.push((ui::OK, "Interceptor".into(), format!("stopped (pid {pid})"))),
        Ok(None) => rows.push((
            ui::NOTE,
            "Interceptor".into(),
            "no daemon was running".into(),
        )),
        Err(e) => rows.push((
            ui::WARN,
            "Interceptor".into(),
            format!("could not stop: {e}"),
        )),
    }

    // 2. Disable run-at-login (matched by app name, so the port is irrelevant here).
    match crate::autostart::configure(false, DEFAULT_PORT) {
        Ok(()) => rows.push((ui::OK, "Autostart".into(), "disabled".into())),
        Err(e) => rows.push((ui::WARN, "Autostart".into(), format!("not changed: {e}"))),
    }

    // 3. Remove the interceptor env. Windows: the `HKCU\Environment` values (plus any legacy
    //    profile block a prior version left). POSIX: the managed block in the shell rc file.
    #[cfg(windows)]
    {
        match clear_user_env() {
            Ok(true) => rows.push((
                ui::OK,
                "Environment".into(),
                "interceptor env removed from HKCU\\Environment".into(),
            )),
            Ok(false) => rows.push((
                ui::NOTE,
                "Environment".into(),
                "no interceptor env to remove".into(),
            )),
            Err(e) => rows.push((ui::WARN, "Environment".into(), format!("not cleaned: {e}"))),
        }
        if let Ok(paths) = remove_profile_block() {
            for path in paths {
                rows.push((
                    ui::OK,
                    "Profile".into(),
                    format!("legacy env block removed from {}", path.display()),
                ));
            }
        }
        // Refresh Explorer's environment so new processes stop seeing the removed values.
        broadcast_env_change();
    }
    #[cfg(not(windows))]
    match remove_profile_block() {
        Ok(paths) if paths.is_empty() => {
            rows.push((ui::NOTE, "Profile".into(), "no env block to remove".into()))
        }
        Ok(paths) => {
            for path in paths {
                rows.push((
                    ui::OK,
                    "Profile".into(),
                    format!("env block removed from {}", path.display()),
                ));
            }
        }
        Err(e) => rows.push((ui::WARN, "Profile".into(), format!("not cleaned: {e}"))),
    }

    // 4. Remove the CA + daemon state (~/.llmtrim).
    let home = crate::daemon::home_dir()?;
    if home.exists() {
        match std::fs::remove_dir_all(&home) {
            Ok(()) => rows.push((
                ui::OK,
                "State".into(),
                format!("removed {} (CA, key, daemon state)", home.display()),
            )),
            Err(e) => rows.push((
                ui::WARN,
                "State".into(),
                format!("could not remove {}: {e}", home.display()),
            )),
        }
    } else {
        rows.push((
            ui::NOTE,
            "State".into(),
            "no state directory to remove".into(),
        ));
    }

    // 4b. Untrust the CA from the OS trust store. Deleting ca.pem (step 4) leaves any
    //     system-wide trust (`llmtrim ca` → certutil -addstore) dangling — a keyless root CA
    //     still trusted. Pull it out so uninstall actually reverses `llmtrim ca`.
    #[cfg(windows)]
    match untrust_ca_root() {
        Ok(true) => rows.push((
            ui::OK,
            "Trust".into(),
            "removed the CA from your user Root store".into(),
        )),
        Ok(false) => rows.push((
            ui::NOTE,
            "Trust".into(),
            "CA was not trusted system-wide".into(),
        )),
        Err(e) => rows.push((ui::WARN, "Trust".into(), format!("not untrusted: {e}"))),
    }
    #[cfg(target_os = "macos")]
    match untrust_ca_keychain() {
        Ok(true) => rows.push((
            ui::OK,
            "Trust".into(),
            "removed the CA from your login keychain".into(),
        )),
        Ok(false) => rows.push((
            ui::NOTE,
            "Trust".into(),
            "CA was not in the login keychain".into(),
        )),
        Err(e) => rows.push((ui::WARN, "Trust".into(), format!("not untrusted: {e}"))),
    }

    // 5. The savings ledger — kept by default (it's your history), removed with --purge.
    match crate::tracking::db_path() {
        Ok(db) if db.exists() && purge => {
            std::fs::remove_file(&db).ok();
            rows.push((ui::OK, "Ledger".into(), format!("removed {}", db.display())));
        }
        Ok(db) if db.exists() => {
            rows.push((
                ui::NOTE,
                "Ledger".into(),
                format!("kept {} (use --purge to remove)", db.display()),
            ));
        }
        _ => {}
    }

    // 6. The binary itself (Unix can unlink a running executable; Windows can't).
    if keep_binary {
        rows.push((ui::NOTE, "Binary".into(), "kept".into()));
    } else if let Ok(exe) = std::env::current_exe() {
        #[cfg(unix)]
        {
            std::fs::remove_file(&exe).ok();
            rows.push((
                ui::OK,
                "Binary".into(),
                format!("removed {}", exe.display()),
            ));
        }
        // Windows can't unlink a running .exe. But we CAN stop `llmtrim` resolving as a
        // command — drop the installer's bin dir from the user PATH — and schedule the
        // install dir's removal after we exit. Only for installer builds (exe under
        // %LOCALAPPDATA%\llmtrim); a cargo/dev binary elsewhere is left untouched.
        #[cfg(windows)]
        {
            match remove_bin_dir_from_path() {
                Ok(true) => rows.push((
                    ui::OK,
                    "PATH".into(),
                    "removed the llmtrim bin dir from your user PATH".into(),
                )),
                Ok(false) => {}
                Err(e) => rows.push((ui::WARN, "PATH".into(), format!("not cleaned: {e}"))),
            }
            if let Some(dir) = installer_dir_of(&exe) {
                schedule_dir_removal(&dir);
                rows.push((
                    ui::OK,
                    "Binary".into(),
                    format!("scheduled removal of {} after exit", dir.display()),
                ));
            } else {
                // Not an installer build (e.g. `cargo install` → ~/.cargo/bin). A running
                // .exe can't unlink itself, so schedule the lone file's deletion after exit.
                schedule_file_removal(&exe);
                rows.push((
                    ui::OK,
                    "Binary".into(),
                    format!("scheduled removal of {} after exit", exe.display()),
                ));
            }
            broadcast_env_change(); // re-broadcast so the dropped PATH entry takes effect
        }
        #[cfg(all(not(unix), not(windows)))]
        {
            rows.push((
                ui::NOTE,
                "Binary".into(),
                format!("remove manually: {}", exe.display()),
            ));
        }
    }

    print!(
        "{}",
        ui::panel(color, "llmtrim uninstall", &ui::kv_rows(color, &rows))
    );
    println!();
    println!(
        "{}",
        ui::paint(
            color,
            Tone::Bold,
            "Done. Open a new shell so the environment changes take effect."
        )
    );
    // The env is gone from disk, but processes that were already running inherited it at
    // launch and keep the now-dead `127.0.0.1:<port>` proxy until they restart — on Windows
    // that includes GUI apps and services (browsers, media servers like Plex), which fail
    // every outbound request with "couldn't connect to 127.0.0.1". We can't reach into a
    // running process's environment, so the only honest fix is: restart them, or reboot.
    #[cfg(windows)]
    println!(
        "{}",
        ui::note(
            color,
            "Apps already running (browsers, Plex/media servers, anything networked) keep the \
             old proxy until you restart them — reboot to clear them all at once."
        )
    );
    #[cfg(not(windows))]
    println!(
        "{}",
        ui::note(
            color,
            "If you trusted the CA system-wide manually, remove it from your OS trust store."
        )
    );
    Ok(())
}

/// All POSIX shell profile files that llmtrim may have written a managed block into.
/// A user can install under bash (block goes to `.bashrc`), later switch to zsh, and the
/// stale block in `.bashrc` keeps `HTTPS_PROXY` pointed at a dead proxy unless we sweep
/// all three candidates on every removal. `base` is the home directory; tests pass a
/// temp dir so no real `$HOME` is ever mutated.
#[cfg(not(windows))]
fn candidate_profiles(base: &std::path::Path) -> Vec<PathBuf> {
    [".bashrc", ".zshrc", ".profile"]
        .iter()
        .map(|f| base.join(f))
        .collect()
}

/// Strip the llmtrim managed block from **every** POSIX shell profile that contains it,
/// using `base` as the home directory. Returns the paths that were actually cleaned.
/// A file that does not exist or cannot be read is silently skipped; a write failure is
/// returned as an error so the caller can report it. Windows: always returns `Ok(vec![])`.
#[cfg_attr(windows, allow(dead_code))]
fn remove_profile_block_in(base: &std::path::Path) -> Result<Vec<PathBuf>> {
    #[cfg(windows)]
    {
        let _ = base;
        return Ok(vec![]);
    }
    #[cfg(not(windows))]
    {
        let mut cleaned = Vec::new();
        for path in candidate_profiles(base) {
            let existing = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue, // absent or unreadable — skip
            };
            if !existing.contains(BEGIN) {
                continue;
            }
            std::fs::write(&path, strip_block(&existing))
                .with_context(|| format!("failed to write {}", path.display()))?;
            cleaned.push(path);
        }
        Ok(cleaned)
    }
}

/// Strip the llmtrim managed block from all candidate shell profiles under `$HOME`.
/// Thin `$HOME`-reading wrapper around [`remove_profile_block_in`].
/// On Windows only deals with any legacy PowerShell profile block (registry is the live path).
fn remove_profile_block() -> Result<Vec<PathBuf>> {
    #[cfg(windows)]
    {
        // Windows live env is the registry; this only handles a legacy profile block that a
        // prior POSIX-style version may have written to the PowerShell profile.
        let Some((path, _)) = profile_target() else {
            return Ok(vec![]);
        };
        let existing = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return Ok(vec![]),
        };
        if !existing.contains(BEGIN) {
            return Ok(vec![]);
        }
        std::fs::write(&path, strip_block(&existing))
            .with_context(|| format!("failed to write {}", path.display()))?;
        return Ok(vec![path]);
    }
    #[cfg(not(windows))]
    {
        let Ok(home) = std::env::var("HOME") else {
            return Ok(vec![]);
        };
        remove_profile_block_in(std::path::Path::new(&home))
    }
}

/// Is the interceptor env still wired up? Used to warn that stopping the daemon while
/// `HTTPS_PROXY` still points at it will break the client's HTTPS. Windows reads the
/// `HKCU\Environment` value; POSIX checks **all three candidate shell profiles** —
/// a stale block in any of them means the env may still be wired for that shell.
pub fn profile_has_block() -> bool {
    #[cfg(windows)]
    {
        user_env_has_proxy()
    }
    #[cfg(not(windows))]
    {
        let Ok(home) = std::env::var("HOME") else {
            return false;
        };
        profile_has_block_in(std::path::Path::new(&home))
    }
}

/// Inner check used by [`profile_has_block`] and tests. Scans all candidate profiles under
/// `base`; returns `true` if the BEGIN marker is found in any of them.
#[cfg(not(windows))]
fn profile_has_block_in(base: &std::path::Path) -> bool {
    candidate_profiles(base).into_iter().any(|path| {
        std::fs::read_to_string(&path)
            .map(|t| t.contains(BEGIN))
            .unwrap_or(false)
    })
}

// ── Windows user environment (`HKCU\Environment`) ───────────────────────────────
// On Windows the proxy env lives in the registry, not a shell profile: it's inherited by
// every process at launch (PS5, pwsh7, Git Bash, cmd, GUI apps) and survives an
// ExecutionPolicy that would block a profile from running. Only processes started after
// the write see it — that's why setup still says "open a new terminal".

/// The three values llmtrim manages in the user environment.
#[cfg(windows)]
const ENV_KEYS: [&str; 3] = ["HTTPS_PROXY", "HTTP_PROXY", "NODE_EXTRA_CA_CERTS"];

/// Open `HKCU\Environment` for read+write (created if somehow absent).
#[cfg(windows)]
fn user_env_key() -> Result<winreg::RegKey> {
    use winreg::RegKey;
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (env, _) = hkcu
        .create_subkey_with_flags("Environment", KEY_READ | KEY_WRITE)
        .context("failed to open HKCU\\Environment")?;
    Ok(env)
}

/// Set `HTTPS_PROXY`/`HTTP_PROXY`/`NODE_EXTRA_CA_CERTS` in the user environment.
#[cfg(windows)]
fn set_user_env(proxy: &str, ca: &str) -> Result<()> {
    set_env_in(&user_env_key()?, proxy, ca)
}

/// Delete the managed values from the user environment. Returns true if anything was
/// removed. Missing values are not an error (idempotent uninstall).
#[cfg(windows)]
fn clear_user_env() -> Result<bool> {
    clear_env_in(&user_env_key()?)
}

/// Does the user environment's `HTTPS_PROXY` point at a local llmtrim interceptor?
#[cfg(windows)]
fn user_env_has_proxy() -> bool {
    user_env_key().is_ok_and(|env| has_proxy_in(&env))
}

/// Broadcast `WM_SETTINGCHANGE("Environment")` so Explorer (and through it, newly-launched
/// terminals, editors, and GUI apps) re-reads `HKCU\Environment` without a logout — a raw
/// registry write alone is invisible until then (`setx` sends the same message). The call
/// needs `SendMessageTimeout`, which is `unsafe` FFI this crate forbids
/// (`unsafe_code = "forbid"`), so shell out to PowerShell with a one-shot P/Invoke.
/// Best-effort: a failure just means "open a new shell" still applies; never breaks setup.
#[cfg(windows)]
fn broadcast_env_change() {
    // HWND_BROADCAST = 0xffff, WM_SETTINGCHANGE = 0x1A, SMTO_ABORTIFHUNG = 0x2, 5 s timeout.
    // (Keep this comment outside the PS string: the string is one line, so an inline `#`
    // would comment out the rest of it and silently no-op the broadcast.)
    const PS: &str = "\
        $sig = '[DllImport(\"user32.dll\", SetLastError=true, CharSet=CharSet.Auto)]\
        public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, \
        string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);';\
        $t = Add-Type -MemberDefinition $sig -Name NativeMethods -Namespace Win32 -PassThru;\
        $r = [UIntPtr]::Zero;\
        [void]$t::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 0x2, 5000, [ref]$r)";
    let _ = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", PS])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

// The registry mechanics, taking the key as a seam so tests can exercise them against a
// throwaway subkey instead of the real `HKCU\Environment`.

#[cfg(windows)]
fn set_env_in(env: &winreg::RegKey, proxy: &str, ca: &str) -> Result<()> {
    env.set_value("HTTPS_PROXY", &proxy)
        .context("failed to set HTTPS_PROXY")?;
    env.set_value("HTTP_PROXY", &proxy)
        .context("failed to set HTTP_PROXY")?;
    env.set_value("NODE_EXTRA_CA_CERTS", &ca)
        .context("failed to set NODE_EXTRA_CA_CERTS")?;
    Ok(())
}

#[cfg(windows)]
fn clear_env_in(env: &winreg::RegKey) -> Result<bool> {
    let mut removed = false;
    for key in ENV_KEYS {
        match env.delete_value(key) {
            Ok(()) => removed = true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("failed to delete {key}")),
        }
    }
    Ok(removed)
}

#[cfg(windows)]
fn has_proxy_in(env: &winreg::RegKey) -> bool {
    env.get_value::<String, _>("HTTPS_PROXY")
        .is_ok_and(|v| v.contains("127.0.0.1"))
}

// ── Windows OS trust-store cleanup ───────────────────────────────────────────────
// `llmtrim ca` tells GUI/non-Node apps to trust the CA system-wide via
// `certutil -addstore -user Root`. Removing ~/.llmtrim/ca.pem does NOT untrust that copy —
// a stale, now-keyless root CA would linger in the store. Uninstall must pull it back out.

/// The CA's subject CommonName, matched verbatim with the value set at generation in
/// `serve.rs` (`dn.push(DnType::CommonName, "llmtrim local CA")`). Keep the two in sync.
#[cfg(any(windows, target_os = "macos"))]
const CA_SUBJECT_CN: &str = "llmtrim local CA";

/// Remove the llmtrim CA from the current-user **Root** trust store if it was trusted
/// system-wide (via `llmtrim ca` / `certutil -addstore`). Matched by subject CN.
///
/// `Remove-Item Cert:\CurrentUser\Root\..` refuses to run non-interactively ("the operation
/// occurred in the user's main store and the UI is not allowed"), so we shell out to
/// `certutil -delstore`, which deletes from the user store without a prompt and without
/// elevation. Best-effort and idempotent: a non-zero exit is almost always "no certificate
/// matched" — i.e. it was never trusted system-wide — and is reported as `Ok(false)`.
#[cfg(windows)]
fn untrust_ca_root() -> Result<bool> {
    let out = std::process::Command::new("certutil")
        .args(["-user", "-delstore", "Root", CA_SUBJECT_CN])
        .output()
        .context("failed to run certutil -delstore")?;
    Ok(out.status.success())
}

/// macOS: remove the llmtrim CA from the **login** keychain if it was trusted there (via
/// `llmtrim ca` → `security add-trusted-cert`). Matched by common name; no sudo, since the
/// login keychain is the user's own. Best-effort and idempotent: a non-zero exit is "no such
/// certificate" — i.e. it was never trusted — and is reported as `Ok(false)`. A *system*
/// keychain trust (sudo-installed) is left to the printed manual note, as on Linux.
#[cfg(target_os = "macos")]
fn untrust_ca_keychain() -> Result<bool> {
    let out = std::process::Command::new("security")
        .args(["delete-certificate", "-c", CA_SUBJECT_CN])
        .output()
        .context("failed to run security delete-certificate")?;
    Ok(out.status.success())
}

// ── Windows binary + PATH cleanup (the installer's footprint) ────────────────────
// install.ps1 drops llmtrim.exe in %LOCALAPPDATA%\llmtrim\bin and adds that dir to the user
// PATH. Uninstall has to reverse both, or `llmtrim` keeps resolving as a command afterwards.

/// The installer's bin dir, `%LOCALAPPDATA%\llmtrim\bin` (the entry it adds to the user PATH).
#[cfg(windows)]
fn installer_bin_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join("llmtrim").join("bin"))
}

/// `%LOCALAPPDATA%\llmtrim` when `exe` lives under it — i.e. this is an installer build, safe
/// to schedule for deletion. A cargo/dev binary elsewhere returns `None` (never self-deleted).
#[cfg(windows)]
fn installer_dir_of(exe: &std::path::Path) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("llmtrim");
    exe.starts_with(&root).then_some(root)
}

/// UTF-16LE bytes with a trailing NUL — the on-disk form of a `REG_SZ`/`REG_EXPAND_SZ` value,
/// so we can write PATH back in whatever string type it already used.
#[cfg(windows)]
fn encode_utf16_nul(s: &str) -> Vec<u8> {
    s.encode_utf16()
        .chain(std::iter::once(0))
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// Drop the installer's bin dir from the user PATH (`HKCU\Environment\Path`). Returns true if
/// it was present and removed. Preserves the value's registry type (`REG_EXPAND_SZ` stays
/// expandable — rewriting it as plain `REG_SZ` would break any `%VAR%` still in the PATH).
#[cfg(windows)]
fn remove_bin_dir_from_path() -> Result<bool> {
    use winreg::enums::RegType;
    use winreg::types::FromRegValue;
    let Some(bin) = installer_bin_dir() else {
        return Ok(false);
    };
    let env = user_env_key()?;
    let Ok(raw) = env.get_raw_value("Path") else {
        return Ok(false); // no user PATH set → nothing of ours to remove
    };
    if raw.vtype != RegType::REG_SZ && raw.vtype != RegType::REG_EXPAND_SZ {
        return Ok(false); // leave an unexpected type untouched
    }
    let current = String::from_reg_value(&raw).unwrap_or_default();
    let stripped = strip_path_entry(&current, &bin.to_string_lossy());
    if stripped == current {
        return Ok(false);
    }
    let new_raw = winreg::RegValue {
        bytes: encode_utf16_nul(&stripped),
        vtype: raw.vtype,
    };
    env.set_raw_value("Path", &new_raw)
        .context("failed to update the user PATH")?;
    Ok(true)
}

/// Schedule deletion of the install dir once we've exited. A running `.exe` can't be unlinked
/// on Windows, so spawn a detached `cmd` that waits (~2 s via `ping`, the reliable console-less
/// delay) then `rmdir`s the tree. Best-effort: uninstall never fails on this.
#[cfg(windows)]
fn schedule_dir_removal(dir: &std::path::Path) {
    use std::os::windows::process::CommandExt;
    use std::process::Stdio;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let script = format!(
        "ping 127.0.0.1 -n 3 >nul & rmdir /s /q \"{}\"",
        dir.display()
    );
    let _ = std::process::Command::new("cmd")
        .args(["/c", &script])
        .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Schedule deletion of a single file once we've exited — the running-`.exe` self-delete for
/// builds that live outside `%LOCALAPPDATA%\llmtrim` (e.g. a `cargo install` binary under
/// `~/.cargo/bin`). Same detached-`cmd` trick as [`schedule_dir_removal`], `del` not `rmdir`.
#[cfg(windows)]
fn schedule_file_removal(file: &std::path::Path) {
    use std::os::windows::process::CommandExt;
    use std::process::Stdio;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let script = format!("ping 127.0.0.1 -n 3 >nul & del /f /q \"{}\"", file.display());
    let _ = std::process::Command::new("cmd")
        .args(["/c", &script])
        .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Remove every occurrence of `dir` from a `;`-separated PATH string, preserving the other
/// entries and their order. Ignores case and a trailing slash (Windows path semantics). Pure,
/// so it's unit-tested on every platform even though it's only called on Windows.
#[cfg_attr(not(windows), allow(dead_code))]
fn strip_path_entry(path: &str, dir: &str) -> String {
    let norm = |s: &str| s.trim().trim_end_matches(['\\', '/']).to_ascii_lowercase();
    let target = norm(dir);
    // Drop only the matching segment(s); other entries (and any pre-existing empties) keep
    // their original text and order — we touch the PATH as little as possible.
    path.split(';')
        .filter(|seg| norm(seg) != target)
        .collect::<Vec<_>>()
        .join(";")
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

/// The rc file for a `$SHELL` value (its basename decides; unknown shells get `.profile`).
/// Single source for the shell→file mapping — used by both [`profile_target`] and
/// [`write_profile_block_in`].
#[cfg(not(windows))]
fn shell_profile_file(shell: &str) -> &'static str {
    if shell.ends_with("zsh") {
        ".zshrc"
    } else if shell.ends_with("bash") {
        ".bashrc"
    } else {
        ".profile"
    }
}

/// The profile file to write the managed env block into, and the syntax it uses. Unix: the
/// `$SHELL` rc file (`export`). Windows: the current-user PowerShell profile (`$env:`).
fn profile_target() -> Option<(PathBuf, Syntax)> {
    #[cfg(not(windows))]
    {
        let home = std::env::var("HOME").ok()?;
        let shell = std::env::var("SHELL").unwrap_or_default();
        let file = shell_profile_file(&shell);
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

/// The managed env block, in the profile's native syntax. Both variants are unit-tested on
/// every platform; on Windows the live env path is the registry, so this is test-only there.
#[allow(dead_code)]
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

/// Inner seam for [`write_profile_block`]: write into a profile file named by `shell` (the
/// basename of `$SHELL`, e.g. `"bash"` → `.bashrc`, `"zsh"` → `.zshrc`) under `base` as the
/// home directory. Sweeps stale blocks from all other candidates under `base` first, then
/// writes the new block. Returns the path written. `base` is never `$HOME` itself; it is
/// always a caller-supplied directory, which tests supply as a temp dir.
#[cfg(not(windows))]
fn write_profile_block_in(
    base: &std::path::Path,
    shell: &str,
    proxy: &str,
    ca: &str,
) -> Result<PathBuf> {
    let path = base.join(shell_profile_file(shell));
    // Sweep stale blocks from ALL candidate profiles under base (including the target).
    // Best-effort: a failure to strip a stale file must not block writing the new block.
    let _ = remove_profile_block_in(base);
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    // strip_block is a safety net in case remove_profile_block_in skipped a file with a
    // broken BEGIN-without-END that it left intact.
    let mut base_content = strip_block(&existing);
    if !base_content.is_empty() && !base_content.ends_with('\n') {
        base_content.push('\n');
    }
    let block = env_block(proxy, ca, Syntax::Posix);
    std::fs::write(&path, format!("{base_content}{block}"))
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

/// Replace (or append) the llmtrim managed block in the shell profile. Idempotent — a
/// re-run updates the existing block rather than stacking duplicates. Also sweeps stale
/// blocks from all other candidate profile files (e.g. `.bashrc` when now running zsh),
/// so re-setup under a different shell does not leave a dead proxy block behind.
/// POSIX-only: on Windows the env lives in the registry, so `run()` never calls this there.
#[allow(dead_code)]
fn write_profile_block(proxy: &str, ca: &str) -> Result<Option<PathBuf>> {
    #[cfg(not(windows))]
    {
        // Delegate to the seam so production and tests run the same code path.
        let Ok(home) = std::env::var("HOME") else {
            return Ok(None);
        };
        let shell = std::env::var("SHELL").unwrap_or_default();
        write_profile_block_in(std::path::Path::new(&home), &shell, proxy, ca).map(Some)
    }
    #[cfg(windows)]
    {
        // Legacy PowerShell-profile arm: the live Windows env is the registry; this is
        // only reachable for a profile-style install a prior version may have used.
        let Some((path, syntax)) = profile_target() else {
            return Ok(None);
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent); // the PowerShell profile dir may not exist yet
        }
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let mut base_content = strip_block(&existing);
        if !base_content.is_empty() && !base_content.ends_with('\n') {
            base_content.push('\n');
        }
        let block = env_block(proxy, ca, syntax);
        std::fs::write(&path, format!("{base_content}{block}"))
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(Some(path))
    }
}

/// Remove any existing llmtrim managed block (between the markers, inclusive).
fn strip_block(s: &str) -> String {
    // If BEGIN exists but END is missing (e.g. user deleted it), return original
    // unchanged rather than silently erasing everything from BEGIN to EOF.
    let has_begin = s.lines().any(|l| l.trim() == BEGIN);
    let has_end = s.lines().any(|l| l.trim() == END);
    if has_begin && !has_end {
        return s.to_string();
    }
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
    fn choose_port_precedence() {
        // Explicit `--port` always wins, even over a running daemon and a configured env.
        assert_eq!(
            choose_port(Some(9000), Some(8800), Some(8700)),
            PortChoice::Use(9000)
        );
        // No explicit → reuse the running daemon's port (don't migrate a live proxy).
        assert_eq!(
            choose_port(None, Some(8800), Some(8700)),
            PortChoice::Use(8800)
        );
        // No daemon → reuse what the env already points at, so re-running converges.
        assert_eq!(choose_port(None, None, Some(8700)), PortChoice::Use(8700));
        // Nothing pinned → scan from the default (the only case that probes for a free port).
        assert_eq!(
            choose_port(None, None, None),
            PortChoice::ScanFrom(DEFAULT_PORT)
        );
    }

    #[test]
    fn parse_proxy_port_reads_the_wired_port() {
        assert_eq!(parse_proxy_port("http://127.0.0.1:8787"), Some(8787));
        // Embedded in a real profile/registry line, with trailing content after the digits.
        assert_eq!(
            parse_proxy_port("export HTTPS_PROXY=\"http://127.0.0.1:9001\"\nexport X=1\n"),
            Some(9001)
        );
        assert_eq!(parse_proxy_port("no proxy here"), None);
        assert_eq!(parse_proxy_port("127.0.0.1:"), None); // present but portless
    }

    #[test]
    fn strip_path_entry_removes_only_the_target() {
        let path = r"C:\Windows;C:\Users\u\AppData\Local\llmtrim\bin;C:\tools";
        let dir = r"C:\Users\u\AppData\Local\llmtrim\bin";
        assert_eq!(strip_path_entry(path, dir), r"C:\Windows;C:\tools");

        // Case- and trailing-slash-insensitive (Windows path semantics), order preserved.
        let messy = r"C:\a;c:\users\u\appdata\local\LLMTRIM\BIN\;C:\b";
        assert_eq!(strip_path_entry(messy, dir), r"C:\a;C:\b");

        // Absent → unchanged. Other entries (incl. pre-existing empties) are left as-is.
        assert_eq!(strip_path_entry(r"C:\a;C:\b", dir), r"C:\a;C:\b");
        // A leading-semicolon PATH (installer appended to an empty user PATH) collapses cleanly.
        assert_eq!(strip_path_entry(&format!(";{dir}"), dir), "");
    }

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
    fn strip_block_missing_end_marker_returns_original() {
        let input = "before\n# >>> llmtrim >>>\nexport HTTPS_PROXY=http://127.0.0.1:8787\n";
        // BEGIN present but END absent → return unchanged (don't erase everything after BEGIN)
        assert_eq!(strip_block(input), input);
    }

    #[test]
    fn strip_block_normal_removes_block() {
        let input = "before\n# >>> llmtrim >>>\nexport HTTPS_PROXY=http://127.0.0.1:8787\n# <<< llmtrim <<<\nafter\n";
        let result = strip_block(input);
        assert!(result.contains("before"));
        assert!(result.contains("after"));
        assert!(!result.contains("HTTPS_PROXY"));
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

    #[test]
    fn write_then_strip_is_idempotent() {
        // Writing a block then stripping it returns to the original content.
        let original = "export FOO=bar\n";
        let proxy = "http://127.0.0.1:8787";
        let ca = "/home/user/.llmtrim/ca.crt";
        let block = env_block(proxy, ca, Syntax::Posix);
        let written = format!("{original}{block}");
        let stripped = strip_block(&written);
        assert_eq!(
            stripped.trim_end(),
            original.trim_end(),
            "strip after write returns original"
        );
    }

    #[test]
    fn double_write_does_not_duplicate_block() {
        // Calling the write-then-strip flow twice should not duplicate the block.
        let proxy = "http://127.0.0.1:8787";
        let ca = "/home/user/.llmtrim/ca.crt";
        let original = "export FOO=bar\n";
        let block = env_block(proxy, ca, Syntax::Posix);
        let after_first = format!("{original}{block}");
        // Simulate a second write: strip then re-add (the real setup flow)
        let after_second = format!("{}{}", strip_block(&after_first), block);
        // The block should appear exactly once
        let begin_count = after_second.matches(">>> llmtrim >>>").count();
        assert_eq!(
            begin_count, 1,
            "block must appear exactly once after double write"
        );
    }

    // Exercise the registry set/has/clear cycle against a throwaway subkey under HKCU so
    // the real `HKCU\Environment` is never touched. The process's own PID keys the scratch
    // path so concurrent test runs don't collide.
    #[cfg(windows)]
    #[test]
    fn registry_env_set_has_clear_roundtrip() {
        use winreg::RegKey;
        use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};

        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let scratch = format!("Software\\llmtrim-test-{}", std::process::id());
        let (env, _) = hkcu
            .create_subkey_with_flags(&scratch, KEY_READ | KEY_WRITE)
            .expect("create scratch key");

        assert!(!has_proxy_in(&env), "fresh key has no proxy");
        assert!(
            !clear_env_in(&env).expect("clear on empty key"),
            "nothing to clear yet"
        );

        set_env_in(&env, "http://127.0.0.1:18784", "C:\\Users\\u\\ca.pem").expect("set env");
        assert!(has_proxy_in(&env), "proxy set");
        assert_eq!(
            env.get_value::<String, _>("NODE_EXTRA_CA_CERTS")
                .expect("read CA value"),
            "C:\\Users\\u\\ca.pem"
        );

        assert!(
            clear_env_in(&env).expect("clear set values"),
            "values removed"
        );
        assert!(!has_proxy_in(&env), "proxy gone after clear");

        // Tidy up the scratch key.
        hkcu.delete_subkey_all(&scratch).ok();
    }

    #[test]
    fn first_free_port_rejects_occupied_accepts_free() {
        // Hold a real port open → occupied. Scanning just that port (span 0) finds nothing,
        // proving a bound port is rejected (this is the bug we hit: 8787 held by VS Code).
        let held = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind ephemeral");
        let taken = held.local_addr().expect("local_addr").port();
        assert_eq!(
            first_free_port(taken, 0),
            None,
            "occupied port not rejected"
        );

        // Release it; the same port is now bindable and the probe returns it.
        drop(held);
        assert_eq!(
            first_free_port(taken, 0),
            Some(taken),
            "free port not accepted"
        );
    }

    // ── Multi-profile sweep tests (POSIX only) ─────────────────────────────────────
    // All tests use a temp dir as the synthetic $HOME so real profile files are never
    // touched and tests are hermetic under parallel `cargo test` runs.
    //
    // `TempDir` is a drop guard: the directory (and its contents) is deleted when it goes
    // out of scope, with no external crate required.

    #[cfg(not(windows))]
    struct TempDir(PathBuf);

    #[cfg(not(windows))]
    impl TempDir {
        fn new(suffix: &str) -> Self {
            // Use PID + a suffix so concurrent test threads don't collide.
            let dir = std::env::temp_dir().join(format!(
                "llmtrim-test-{}-{}",
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

    #[cfg(not(windows))]
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Helper: write a managed block into a candidate file under `base`.
    #[cfg(not(windows))]
    fn write_block_to(base: &std::path::Path, name: &str) {
        let block = format!("{BEGIN}\nexport HTTPS_PROXY=\"http://127.0.0.1:8787\"\n{END}\n");
        std::fs::write(base.join(name), block).expect("write test block");
    }

    /// Block present in two files → both cleaned; function returns both paths.
    #[cfg(not(windows))]
    #[test]
    fn remove_profile_block_in_cleans_all_files_that_contain_block() {
        let dir = TempDir::new("sweep-two");
        let base = dir.path();
        write_block_to(base, ".bashrc");
        write_block_to(base, ".zshrc");
        // .profile intentionally absent

        let cleaned = remove_profile_block_in(base).expect("remove_profile_block_in");
        let mut names: Vec<String> = cleaned
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec![".bashrc", ".zshrc"]);

        // Both files now contain no managed block.
        for name in [".bashrc", ".zshrc"] {
            let content = std::fs::read_to_string(base.join(name)).expect("read back");
            assert!(
                !content.contains(BEGIN),
                "{name} still contains BEGIN marker after sweep"
            );
        }
    }

    /// No files contain the block → Ok(empty vec), no error.
    #[cfg(not(windows))]
    #[test]
    fn remove_profile_block_in_noop_when_no_blocks() {
        let dir = TempDir::new("noop");
        let base = dir.path();
        // Write files with ordinary content, no managed block.
        std::fs::write(base.join(".bashrc"), "export FOO=bar\n").expect("write .bashrc");

        let cleaned = remove_profile_block_in(base).expect("remove_profile_block_in");
        assert!(cleaned.is_empty(), "expected no files cleaned");
        // Content unchanged.
        let content = std::fs::read_to_string(base.join(".bashrc")).expect("read");
        assert_eq!(content, "export FOO=bar\n");
    }

    /// One file is absent/unreadable, others are cleaned without error.
    #[cfg(not(windows))]
    #[test]
    fn remove_profile_block_in_skips_absent_files_cleans_rest() {
        let dir = TempDir::new("skip-absent");
        let base = dir.path();
        // Only .zshrc exists with a block; .bashrc and .profile are absent.
        write_block_to(base, ".zshrc");

        let cleaned = remove_profile_block_in(base).expect("remove_profile_block_in");
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].file_name().unwrap().to_string_lossy(), ".zshrc");
    }

    /// profile_has_block_in returns true when ANY candidate file contains the block.
    #[cfg(not(windows))]
    #[test]
    fn profile_has_block_in_detects_block_in_any_candidate() {
        let dir = TempDir::new("has-block");
        let base = dir.path();

        // Nothing present → false.
        assert!(!profile_has_block_in(base));

        // Block only in .bashrc → still detected.
        write_block_to(base, ".bashrc");
        assert!(profile_has_block_in(base));

        // After sweep → false again.
        remove_profile_block_in(base).expect("sweep");
        assert!(!profile_has_block_in(base));

        // Block only in .zshrc → detected.
        write_block_to(base, ".zshrc");
        assert!(profile_has_block_in(base));
    }

    // ── write_profile_block_in tests ────────────────────────────────────────────────

    /// Round-trip: write then read back the block, confirm it contains valid export lines and
    /// BEGIN/END markers, and that unrelated content above it is preserved.
    #[cfg(not(windows))]
    #[test]
    fn write_profile_block_in_roundtrip_preserves_existing_content() {
        let dir = TempDir::new("wpb-roundtrip");
        let base = dir.path();
        let proxy = "http://127.0.0.1:8787";
        let ca = "/tmp/ca.crt";

        // Pre-populate .bashrc with unrelated content.
        std::fs::write(base.join(".bashrc"), "export FOO=bar\nexport BAZ=qux\n")
            .expect("write pre-existing .bashrc");

        let path = write_profile_block_in(base, "bash", proxy, ca).expect("write_profile_block_in");
        assert_eq!(path, base.join(".bashrc"));

        let content = std::fs::read_to_string(&path).expect("read back");
        // Unrelated content must be preserved above the block.
        assert!(
            content.contains("export FOO=bar"),
            "pre-existing content lost"
        );
        // Block delimiters must be present.
        assert!(content.contains(BEGIN), "BEGIN marker missing");
        assert!(content.contains(END), "END marker missing");
        // POSIX export syntax.
        assert!(
            content.contains(&format!("export HTTPS_PROXY=\"{proxy}\"")),
            "proxy export missing"
        );
        assert!(
            content.contains(&format!("export NODE_EXTRA_CA_CERTS=\"{ca}\"")),
            "CA export missing"
        );
    }

    /// Idempotency: calling write_profile_block_in twice on the same shell must leave exactly
    /// ONE managed block (BEGIN appears exactly once) and preserve unrelated content.
    #[cfg(not(windows))]
    #[test]
    fn write_profile_block_in_idempotent_second_call_does_not_duplicate_block() {
        let dir = TempDir::new("wpb-idempotent");
        let base = dir.path();
        let proxy = "http://127.0.0.1:8787";
        let ca = "/tmp/ca.crt";

        std::fs::write(base.join(".bashrc"), "# user config\n").expect("write .bashrc");

        write_profile_block_in(base, "bash", proxy, ca).expect("first write");
        write_profile_block_in(base, "bash", proxy, ca).expect("second write");

        let content = std::fs::read_to_string(base.join(".bashrc")).expect("read back");
        let begin_count = content.matches(BEGIN).count();
        assert_eq!(
            begin_count, 1,
            "BEGIN marker appears {begin_count} times after two writes — expected 1"
        );
        assert!(
            content.contains("# user config"),
            "pre-existing content lost after double write"
        );
    }

    /// Shell-switch sweep: block written for bash, then write_profile_block_in called with zsh.
    /// The stale .bashrc block must be removed; .zshrc gets the new block.
    #[cfg(not(windows))]
    #[test]
    fn write_profile_block_in_sweeps_stale_shell_on_switch() {
        let dir = TempDir::new("wpb-switch");
        let base = dir.path();
        let proxy = "http://127.0.0.1:8787";
        let ca = "/tmp/ca.crt";

        // Simulate a prior bash setup: .bashrc has a managed block.
        write_block_to(base, ".bashrc");

        // Now "switch" to zsh.
        let path =
            write_profile_block_in(base, "zsh", proxy, ca).expect("write_profile_block_in for zsh");
        assert_eq!(path, base.join(".zshrc"), "should write to .zshrc");

        // .bashrc must have its stale block swept.
        let bashrc = std::fs::read_to_string(base.join(".bashrc")).expect("read .bashrc");
        assert!(
            !bashrc.contains(BEGIN),
            ".bashrc still contains stale block after shell switch"
        );

        // .zshrc must have exactly one block.
        let zshrc = std::fs::read_to_string(base.join(".zshrc")).expect("read .zshrc");
        assert_eq!(
            zshrc.matches(BEGIN).count(),
            1,
            ".zshrc should have exactly one block"
        );
    }

    // ── strip_block adversarial cases ────────────────────────────────────────────────

    /// Block at the very start of the file (no content before BEGIN).
    #[test]
    fn strip_block_at_file_start() {
        let input = format!("{BEGIN}\nexport X=1\n{END}\nafter\n");
        let out = strip_block(&input);
        assert_eq!(out, "after\n");
    }

    /// Block at the very end of the file (no content after END).
    #[test]
    fn strip_block_at_file_end() {
        let input = format!("before\n{BEGIN}\nexport X=1\n{END}\n");
        let out = strip_block(&input);
        assert_eq!(out, "before\n");
    }

    /// Block in the middle of the file (content before and after).
    #[test]
    fn strip_block_in_the_middle() {
        let input = format!("top\n{BEGIN}\nexport X=1\n{END}\nbottom\n");
        let out = strip_block(&input);
        assert_eq!(out, "top\nbottom\n");
    }

    /// Two stacked (adjacent) managed blocks — both must be stripped.
    #[test]
    fn strip_block_two_stacked_blocks() {
        let input =
            format!("before\n{BEGIN}\nexport X=1\n{END}\n{BEGIN}\nexport Y=2\n{END}\nafter\n");
        let out = strip_block(&input);
        assert_eq!(out, "before\nafter\n");
    }

    /// File with no trailing newline — strip_block must not panic and must return
    /// a sane result (either the content before the block, or the original if no block).
    #[test]
    fn strip_block_no_trailing_newline() {
        // No block, no trailing newline — must return unchanged.
        let no_block = "just some text";
        let out = strip_block(no_block);
        assert_eq!(out, "just some text\n"); // strip_block always appends \n per line

        // With block and no trailing newline after END.
        let with_block = format!("before\n{BEGIN}\nexport X=1\n{END}");
        let out2 = strip_block(&with_block);
        assert_eq!(out2, "before\n");
    }

    // ── env_block syntax verification ───────────────────────────────────────────────

    /// POSIX block: every line between the markers is a valid `export KEY="value"` line.
    #[test]
    fn env_block_posix_all_lines_are_valid_exports() {
        let proxy = "http://127.0.0.1:8787";
        let ca = "/home/user/.llmtrim/ca.crt";
        let block = env_block(proxy, ca, Syntax::Posix);

        // Collect non-marker, non-empty lines inside the block.
        let inner: Vec<&str> = block.lines().filter(|l| *l != BEGIN && *l != END).collect();
        assert!(!inner.is_empty(), "no inner lines in POSIX block");
        for line in &inner {
            assert!(
                line.starts_with("export ") && line.contains("=\""),
                "line is not a valid POSIX export: {line:?}"
            );
        }
    }

    /// PowerShell block: every line between the markers uses `$env:` assignment syntax, never
    /// POSIX `export`.
    #[test]
    fn env_block_powershell_all_lines_use_dollar_env() {
        let proxy = "http://127.0.0.1:8787";
        let ca = "C:\\Users\\u\\ca.pem";
        let block = env_block(proxy, ca, Syntax::PowerShell);

        let inner: Vec<&str> = block.lines().filter(|l| *l != BEGIN && *l != END).collect();
        assert!(!inner.is_empty(), "no inner lines in PowerShell block");
        for line in &inner {
            assert!(
                line.starts_with("$env:") && line.contains(" = \""),
                "line is not a valid PowerShell env assignment: {line:?}"
            );
            assert!(
                !line.starts_with("export "),
                "POSIX `export` leaked into PowerShell block: {line:?}"
            );
        }
    }
}
