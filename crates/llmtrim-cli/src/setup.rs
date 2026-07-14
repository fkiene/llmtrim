//! `llmtrim setup` — the one-command bootstrap. llmtrim is *only* a MITM proxy, so
//! integration is purely at the environment level: it ensures the local CA, then sets
//! `HTTPS_PROXY` + `NODE_EXTRA_CA_CERTS` for the user (POSIX: a managed shell-profile
//! block; Windows: `HKCU\Environment`) so newly-launched tools route through the
//! interceptor and trust the CA — **no IDE settings touched, no sudo** — enables
//! run-at-login, and starts the daemon. On POSIX the block wires the vars only while the
//! daemon is up (gated on `llmtrim _alive`), so a shell opened after `stop` routes directly
//! instead of at a dead proxy; Windows syncs the registry on daemon start/stop instead.
//!
//! Best-effort and idempotent: a step that fails warns and the rest proceeds.

use std::net::{Ipv4Addr, TcpListener};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::ui::{self, Tone};

const BEGIN: &str = "# >>> llmtrim >>>";
const END: &str = "# <<< llmtrim <<<";

/// The shell command that clears the interceptor env vars from the *current* shell. A running
/// process can't rewrite its parent shell's environment, so `stop`/`uninstall` print this for
/// the user to run (or they open a new shell, which self-heals via the daemon-gated block).
/// Single source of truth for the five managed vars, in both `NO_PROXY` casings.
pub const UNSET_HINT: &str = "unset HTTPS_PROXY HTTP_PROXY NO_PROXY no_proxy NODE_EXTRA_CA_CERTS SSL_CERT_FILE CURL_CA_BUNDLE";

/// Hosts/ranges that must bypass the interceptor: loopback, link-local, and the private LAN
/// ranges (RFC-1918 + IPv6 ULA). llmtrim only MITMs a fixed set of public LLM API hosts (the
/// CA's domain set — see `serve::should_intercept`), so routing local or LAN traffic through the
/// proxy can only break it: localhost dev servers, intranet, and local device discovery
/// (Plex/DLNA, printers, NAS). Without `NO_PROXY`, *every* proxy-aware program on the machine
/// (not just LLM tools) funnels its local and LAN calls at `127.0.0.1:<port>` and they fail with
/// "couldn't connect" whenever the interceptor is down or on a stale port. Set alongside
/// `HTTPS_PROXY` so the bypass travels with the proxy.
///
/// Portability note: the literal hosts (`localhost`, `127.0.0.1`, `::1`) and the `*.local` suffix
/// are honored by virtually every client. The **CIDR** ranges are not universal — curl and
/// Node/undici match only by exact host or domain suffix, not CIDR (Go and some others honor
/// CIDR). There is no portable way to express "all of 192.168.0.0/16" in `NO_PROXY`, so LAN
/// bypass *by raw IP* is best-effort on non-CIDR clients; loopback/`localhost`/`*.local` is the
/// portable, high-value core. The CIDR entries are harmless where ignored.
const NO_PROXY: &str = "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,fd00::/8,*.local";

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

/// (pid, process name) of whatever LISTENs on `port`, via the platform's native tools —
/// a cold setup-time path, so shelling out beats adding a process-inspection dependency.
/// Best-effort: any parse failure is `None` and the caller falls back to the plain note.
fn port_holder(port: u16) -> Option<(u32, String)> {
    #[cfg(windows)]
    {
        let out = std::process::Command::new("netstat")
            .args(["-ano"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let needle = format!(":{port}");
        let pid: u32 = text
            .lines()
            .filter(|l| l.contains("LISTENING") && l.contains(&needle))
            .filter_map(|l| l.split_whitespace().last()?.parse().ok())
            .next()?;
        let out = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .ok()?;
        let line = String::from_utf8_lossy(&out.stdout);
        let name = line
            .trim()
            .trim_start_matches('"')
            .split('"')
            .next()?
            .to_string();
        (!name.is_empty()).then_some((pid, name))
    }
    #[cfg(unix)]
    {
        // `ss -ltnp` prints `users:(("name",pid=123,fd=7))`; fall back to lsof.
        if let Ok(out) = std::process::Command::new("ss")
            .args(["-ltnp", &format!("sport = :{port}")])
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(users) = text.split("users:((\"").nth(1) {
                let name = users.split('"').next().unwrap_or_default().to_string();
                if let Some(pid) = users.split("pid=").nth(1).and_then(|s| {
                    s.chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>()
                        .parse()
                        .ok()
                }) {
                    return Some((pid, name));
                }
            }
        }
        let out = std::process::Command::new("lsof")
            .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-Fpc"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let pid = text.lines().find(|l| l.starts_with('p'))?[1..]
            .parse()
            .ok()?;
        let name = text.lines().find(|l| l.starts_with('c'))?[1..].to_string();
        Some((pid, name))
    }
}

/// Force-kill a process we identified as an orphaned llmtrim. Best-effort by design.
fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .output();
    }
    #[cfg(unix)]
    {
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }
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

/// Self-heal an existing managed env block that predates the `NO_PROXY` bypass, *in place* and
/// without re-running `setup`. Called once when the daemon comes up at login: an install wired
/// before the `NO_PROXY` fix has `HTTPS_PROXY` set but no bypass, so every proxy-aware app funnels
/// its LAN/local traffic at the dead-for-LAN interceptor. We rewrite only blocks that already
/// exist and lack the bypass — never create a block (that's `setup`'s job), never touch an install
/// that's already current. Best-effort by design: a heal failure must never stop the proxy serving.
///
/// Note the OS limit this can't beat: a process inherits its env at launch, so already-running
/// apps (Plex, open shells) keep the old, bypass-less env until they restart. This fixes what
/// starts *after* the heal; the one-time restart of running apps is unavoidable.
pub fn heal_managed_env() -> Result<Vec<PathBuf>> {
    #[cfg(windows)]
    {
        let env = user_env_key()?;
        // Only heal a wired install that predates the bypass: HTTPS_PROXY present, NO_PROXY absent.
        if !has_proxy_in(&env) || env.get_value::<String, _>("NO_PROXY").is_ok() {
            return Ok(vec![]);
        }
        let proxy: String = env.get_value("HTTPS_PROXY").context("read HTTPS_PROXY")?;
        let ca: String = env.get_value("NODE_EXTRA_CA_CERTS").unwrap_or_default();
        set_env_in(&env, &proxy, &ca)?; // re-set adds NO_PROXY alongside the existing values
        broadcast_env_change();
        Ok(vec![PathBuf::from("HKCU\\Environment")])
    }
    #[cfg(not(windows))]
    {
        let Ok(home) = std::env::var("HOME") else {
            return Ok(vec![]);
        };
        // Reconstruct the env the block should now carry from what's already wired.
        let Some(port) = configured_port() else {
            return Ok(vec![]); // nothing wired → nothing to heal
        };
        let proxy = format!("http://127.0.0.1:{port}");
        let ca_path = crate::serve::ca_cert_path()?;
        let ca = ca_path.to_string_lossy().into_owned();
        // Best-effort: a bundle build failure just means the heal keeps the Node-only trust.
        let bundle = ensure_ca_bundle(&ca_path).ok().flatten();
        let bundle_str = bundle.as_ref().map(|p| p.to_string_lossy().into_owned());
        heal_profiles_in(
            std::path::Path::new(&home),
            &proxy,
            &ca,
            bundle_str.as_deref(),
        )
    }
}

/// Inner seam for [`heal_managed_env`] (POSIX), against `base` as the home dir so tests stay
/// hermetic. Rewrites the managed block only in profiles that already contain a healable one (see
/// [`managed_block_needs_heal`]); returns the paths actually rewritten.
/// Does `s` contain a *well-formed* managed block (`BEGIN`…`END`) that predates the current block
/// shape and should be rewritten? A block is healable when it lacks either the `NO_PROXY` bypass
/// or the daemon-liveness gate (`llmtrim _alive`) — both are silent rewrites into the current
/// form, and the second is what migrates every pre-gate install so `stop` unwires new shells
/// without the user re-running `setup`. Only a well-formed block qualifies: a malformed
/// BEGIN-without-END is skipped (rewriting it would stack a duplicate), an already-current block
/// (both markers present inside it) is skipped, and a `NO_PROXY`/guard the user placed *outside*
/// the markers does not count — it must be inside the block we manage. Pure/testable.
#[cfg(not(windows))]
fn managed_block_needs_heal(s: &str) -> bool {
    let (mut in_block, mut saw_begin, mut saw_end, mut bypass_in_block, mut gate_in_block) =
        (false, false, false, false, false);
    for line in s.lines() {
        match line.trim() {
            BEGIN => {
                in_block = true;
                saw_begin = true;
            }
            END => {
                if in_block {
                    saw_end = true;
                }
                in_block = false;
            }
            l if in_block && l.contains("llmtrim _alive") => gate_in_block = true,
            l if in_block && l.contains("NO_PROXY") => bypass_in_block = true,
            _ => {}
        }
    }
    saw_begin && saw_end && (!bypass_in_block || !gate_in_block)
}

#[cfg(not(windows))]
fn heal_profiles_in(
    base: &std::path::Path,
    proxy: &str,
    ca: &str,
    bundle: Option<&str>,
) -> Result<Vec<PathBuf>> {
    let block = env_block(proxy, ca, bundle, Syntax::Posix);
    let mut healed = Vec::new();
    for path in candidate_profiles(base) {
        let existing = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue, // absent or unreadable — skip
        };
        // Heal only a well-formed old block (markers present, bypass missing *inside the block*).
        // Scoping the check to the block — not the whole file — means a user's own `NO_PROXY`
        // elsewhere can't mask a stale managed block; refusing a BEGIN-without-END block avoids
        // strip_block returning the file unchanged and us stacking a second block onto it.
        if !managed_block_needs_heal(&existing) {
            continue;
        }
        let mut base_content = strip_block(&existing);
        if !base_content.is_empty() && !base_content.ends_with('\n') {
            base_content.push('\n');
        }
        std::fs::write(&path, format!("{base_content}{block}"))
            .with_context(|| format!("failed to write {}", path.display()))?;
        healed.push(path);
    }
    Ok(healed)
}

pub fn run(requested: Option<u16>, force: bool) -> Result<()> {
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
    let mut port = match choose_port(requested, running_port, configured) {
        PortChoice::Use(p) => p,
        PortChoice::ScanFrom(start) => first_free_port(start, 64)
            .with_context(|| format!("no free port in {start}..={}", start.saturating_add(64)))?,
    };
    // Only chatter about the port when we had to pick one nobody asked for (first install,
    // default busy). When we're reusing a pinned port, silence is correct.
    if pinned.is_none() && port != DEFAULT_PORT {
        // Who holds the default? An orphaned llmtrim (binary replaced/removed while the
        // old daemon kept running — e.g. `npm uninstall` can't stop it) is reclaimed
        // instead of silently drifting ports and stranding the zombie.
        match port_holder(DEFAULT_PORT) {
            Some((pid, name)) if name.to_lowercase().contains("llmtrim") => {
                kill_pid(pid);
                // Give the OS a moment to release the socket, then take the default back.
                for _ in 0..20 {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    if std::net::TcpListener::bind(("127.0.0.1", DEFAULT_PORT)).is_ok() {
                        port = DEFAULT_PORT;
                        break;
                    }
                }
                if port == DEFAULT_PORT {
                    println!(
                        "{}",
                        ui::note(
                            color,
                            &format!(
                                "Stopped an orphaned llmtrim daemon (pid {pid}) holding port {DEFAULT_PORT}."
                            )
                        )
                    );
                } else {
                    println!(
                        "{}",
                        ui::note(
                            color,
                            &format!(
                                "Port {DEFAULT_PORT} held by an old llmtrim (pid {pid}) that wouldn't die — using {port}."
                            )
                        )
                    );
                }
            }
            Some((pid, name)) => {
                println!(
                    "{}",
                    ui::note(
                        color,
                        &format!("Port {DEFAULT_PORT} busy ({name}, pid {pid}) — using {port}.")
                    )
                );
            }
            None => {
                println!(
                    "{}",
                    ui::note(color, &format!("Port {DEFAULT_PORT} busy — using {port}."))
                );
            }
        }
    }

    // Steps are collected as checklist rows and rendered as one summary panel at the
    // end; soft failures become `⚠` rows instead of stderr asides, so the user sees
    // one coherent report.
    let mut rows: Vec<(&str, String, String)> = Vec::new();

    // 1. Local CA (generated on first run, name-constrained to LLM domains).
    crate::serve::ensure_ca()?;
    let ca_path = crate::serve::ca_cert_path()?;
    let ca = ca_path.to_string_lossy().to_string();
    let proxy = format!("http://127.0.0.1:{port}");
    rows.push((ui::OK, "Local CA".into(), ca.clone()));

    // 1b. Combined bundle (OS roots + CA) for native TLS clients that ignore NODE_EXTRA_CA_CERTS
    //     (curl, git, Python, rustls tools like OpenAI Codex). POSIX only; Windows uses the OS
    //     cert store. Best-effort: a missing OS root bundle just skips SSL_CERT_FILE.
    #[cfg(not(windows))]
    let bundle = match ensure_ca_bundle(&ca_path) {
        Ok(Some(p)) => {
            rows.push((ui::OK, "CA bundle".into(), p.to_string_lossy().into_owned()));
            Some(p.to_string_lossy().into_owned())
        }
        Ok(None) => {
            rows.push((
                ui::NOTE,
                "CA bundle".into(),
                "no OS root bundle found — native TLS clients use the OS trust store".into(),
            ));
            None
        }
        Err(e) => {
            rows.push((ui::WARN, "CA bundle".into(), format!("not built: {e}")));
            None
        }
    };

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
    let manual_env = {
        let paths = write_profile_block(&proxy, &ca, bundle.as_deref())?;
        if paths.is_empty() {
            rows.push((
                ui::NOTE,
                "Profile".into(),
                "no shell profile found — set the env yourself (below)".into(),
            ));
            true
        } else {
            let names = paths
                .iter()
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .collect::<Vec<_>>()
                .join(", ");
            rows.push((
                ui::OK,
                "Profile".into(),
                format!("{names} — HTTPS_PROXY + CA trust"),
            ));
            false
        }
    };

    // 3. Run at login (systemd / launchd / Windows, via auto-launch).
    match crate::autostart::configure(true, port) {
        Ok(()) => rows.push((ui::OK, "Autostart".into(), "runs at login".into())),
        Err(e) => rows.push((ui::WARN, "Autostart".into(), format!("not enabled: {e}"))),
    }

    // 3b. Desktop tray. Only offered when the GUI binary is installed next to the
    //     CLI (the desktop bundles ship it; a headless `cargo install` doesn't),
    //     which is also how "only on a desktop OS" is enforced. Default is on;
    //     `--force` / a non-interactive setup take that default without asking.
    if crate::tray::tray_binary().is_some() {
        let want = force || confirm_tray_default_yes();
        match crate::autostart::configure_tray(want) {
            Ok(()) if want => rows.push((
                ui::OK,
                "Tray".into(),
                "opens at login · run `llmtrim tray` to open now".into(),
            )),
            Ok(()) => rows.push((
                ui::OK,
                "Tray".into(),
                "left disabled · enable later with `llmtrim autostart --tray`".into(),
            )),
            Err(e) => rows.push((ui::WARN, "Tray".into(), format!("not enabled: {e}"))),
        }
    }

    // 3c. Claude Code compaction routing. Ask only on first configuration; re-running setup must
    // never overwrite a customized order or a remembered opt-out. Scripted setup takes the
    // recommended default, matching the tray's non-interactive default behavior.
    let mut compact_changed = false;
    if crate::statusline::claude_code_present()
        && !llmtrim_core::config::compact_models_configured()
    {
        let want = force || confirm_compact_default_yes();
        let models = if want {
            vec!["haiku".to_string(), "sonnet".to_string()]
        } else {
            Vec::new()
        };
        match llmtrim_core::config::write_compact_models(&models) {
            Ok(()) => {
                compact_changed = true;
                rows.push((
                    ui::OK,
                    "Compact".into(),
                    if want {
                        "Haiku → Sonnet → original model".into()
                    } else {
                        "original model only".into()
                    },
                ));
            }
            Err(e) => rows.push((ui::WARN, "Compact".into(), format!("not configured: {e}"))),
        }
    }

    // 4. Reconcile the interceptor. If a healthy daemon is already serving the resolved port,
    //    leave it running — re-running `setup` must not drop in-flight requests (the old code
    //    stopped + respawned unconditionally on every run). Restart only when the port is
    //    changing (explicit `--port`, or self-healing a drifted state) or the daemon is gone —
    //    that also picks up a new binary after an update (the silent-stale-update trap).
    let daemon_ok = match &running {
        // `--force` falls through to the restart arm even on a matching port (e.g. to pick up a
        // freshly installed binary); without it a healthy same-port daemon is left untouched.
        Some(state) if state.port == port && !force && !compact_changed => {
            rows.push((
                ui::OK,
                "Interceptor".into(),
                format!("already running · pid {} · port {port}", state.pid),
            ));
            true
        }
        _ => {
            // Clear a dead/old-port daemon (or the --force target) and wait for the port to free
            // before respawning, so the new daemon doesn't lose the bind race. A timeout here is
            // surfaced (not silently swallowed) so a confusing spawn failure isn't a mystery.
            if let Ok(false) = crate::daemon::stop_and_wait_free(port) {
                rows.push((
                    ui::WARN,
                    "Interceptor".into(),
                    format!("port {port} still held 5s after stop; starting anyway"),
                ));
            }
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
        println!("    export NO_PROXY={NO_PROXY}");
        println!("    export NODE_EXTRA_CA_CERTS={ca}");
        if let Some(b) = bundle.as_deref() {
            println!("    export SSL_CERT_FILE={b}");
            println!("    export CURL_CA_BUNDLE={b}");
        }
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
    // Claude Code users only: hint at the status line without writing their config (setup is
    // client-agnostic and never touches IDE settings). No-op for users of other agents.
    if crate::statusline::claude_code_present() {
        println!(
            "  {}  llmtrim statusline install",
            ui::paint(color, Tone::Dim, "status line  ")
        );
    }
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

    // caveman (the output-compression skill) shapes model output on every request. llmtrim's
    // `auto` already shapes output where it pays (code / long context / plain prose), and
    // deliberately skips it on agent traffic — tool-call replies are already short,
    // so terse shaping there saves no tokens (bench/README glaive: cost ~-5%, quality +0pp).
    // So caveman is redundant where output shaping helps and a no-op win where it doesn't.
    // Warn and print caveman's *own* uninstall commands — never run them ourselves: removing
    // another tool is the user's call, not setup's.
    if caveman_installed() {
        println!();
        println!(
            "{}",
            ui::warn(
                color,
                "caveman detected. It shapes model output on every request. llmtrim's auto \
                 mode already shapes output where it pays (code, long context, plain prose) \
                 and skips it on tool-calling agent traffic on purpose: tool-call replies are \
                 already short, so terse shaping there saves no tokens (our bench measured \
                 quality neutral). So caveman adds nothing llmtrim does not already do where \
                 it helps. To remove it:"
            )
        );
        #[cfg(not(windows))]
        println!(
            "    bash <(curl -s https://raw.githubusercontent.com/JuliusBrussee/caveman/main/src/hooks/uninstall.sh)"
        );
        #[cfg(windows)]
        println!(
            "    irm https://raw.githubusercontent.com/JuliusBrussee/caveman/main/src/hooks/uninstall.ps1 | iex"
        );
        println!("    claude plugin disable caveman        # if installed as a Claude Code plugin");
        println!("    npx skills remove caveman            # Cursor, Windsurf, Cline, Copilot, …");
        println!("    gemini extensions uninstall caveman  # Gemini CLI");
    }
    Ok(())
}

/// Ask whether to route Claude Code compaction through cheaper models, defaulting to yes. A
/// scripted setup takes the default without blocking on stdin.
fn confirm_compact_default_yes() -> bool {
    use std::io::{IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return true;
    }
    print!("Use cheaper models for Claude Code /compact (Haiku -> Sonnet -> original)? [Y/n] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    !matches!(line.trim().to_ascii_lowercase().as_str(), "n" | "no")
}

/// Ask whether to enable the desktop tray, defaulting to yes. Returns `true`
/// (the default) without asking when stdin isn't a terminal, so a scripted
/// `setup` stays non-interactive and still enables the tray.
fn confirm_tray_default_yes() -> bool {
    use std::io::{IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return true;
    }
    print!("Enable the llmtrim desktop tray (opens at login)? [Y/n] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        // The prompt couldn't be completed (broken pipe, closed stdin). Decline
        // rather than enable an optional component the user never confirmed.
        return false;
    }
    !matches!(line.trim().to_ascii_lowercase().as_str(), "n" | "no")
}

/// Best-effort caveman detection: the flag file its session hook writes, its standalone
/// hook files, or a Claude Code plugin-cache entry. Read-only probes; any I/O failure
/// reads as "not installed" — setup must never fail because of someone else's tool.
fn caveman_installed() -> bool {
    let claude_dir = std::env::var("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .ok()
                .map(|h| PathBuf::from(h).join(".claude"))
        });
    let Some(dir) = claude_dir else {
        return false;
    };
    caveman_installed_in(&dir)
}

/// The detection itself, on an explicit Claude config dir (testable without env games).
fn caveman_installed_in(dir: &std::path::Path) -> bool {
    if dir.join(".caveman-active").is_file()
        || dir.join("hooks").join("caveman-activate.js").is_file()
        || dir.join("hooks").join("caveman-config.js").is_file()
    {
        return true;
    }
    // Plugin install: a "caveman" entry somewhere shallow in the plugin cache.
    fn walk(p: &std::path::Path, depth: u8) -> bool {
        if depth == 0 {
            return false;
        }
        let Ok(rd) = std::fs::read_dir(p) else {
            return false;
        };
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().contains("caveman") {
                return true;
            }
            let path = e.path();
            if path.is_dir() && walk(&path, depth - 1) {
                return true;
            }
        }
        false
    }
    walk(&dir.join("plugins"), 3)
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

    // 2b. Disable the tray's separate login entry too, so uninstall leaves nothing
    //     that revives a GUI at next login.
    match crate::autostart::configure_tray(false) {
        Ok(()) => rows.push((ui::OK, "Tray autostart".into(), "disabled".into())),
        Err(e) => rows.push((
            ui::WARN,
            "Tray autostart".into(),
            format!("not changed: {e}"),
        )),
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
    // Package-manager-owned binaries are NOT deleted: removing the file out from under
    // npm/cargo/brew leaves their bookkeeping broken — print their command instead.
    let manager_cmd = match crate::update::channel() {
        crate::update::Channel::Npm => Some("npm uninstall -g @llmtrim/cli"),
        crate::update::Channel::Cargo => Some("cargo uninstall llmtrim"),
        crate::update::Channel::Homebrew => Some("brew uninstall llmtrim"),
        crate::update::Channel::Binary => None,
    };
    if keep_binary {
        rows.push((ui::NOTE, "Binary".into(), "kept".into()));
    } else if let Some(cmd) = manager_cmd {
        rows.push((
            ui::NOTE,
            "Binary".into(),
            format!("owned by your package manager — finish with: {cmd}"),
        ));
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
            &format!(
                "Done. Your current shell still has HTTPS_PROXY, HTTP_PROXY, NO_PROXY, and \
                 NODE_EXTRA_CA_CERTS exported. Open a new shell to clear them, or run: {UNSET_HINT}"
            )
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

/// Every POSIX shell rc file llmtrim may write its managed block into. Setup writes the
/// block to all of these that exist (so whichever shell the terminal launches picks up the
/// env, independent of `$SHELL`); uninstall sweeps the block from all of them. Covers the
/// interactive AND login rc of the two common shells, plus the universal `.profile`:
/// zsh interactive `.zshrc` / login `.zprofile`; bash interactive `.bashrc` / login
/// `.bash_profile`; and `.profile` (sh, and bash-login when no `.bash_profile`). `base` is
/// the home directory; tests pass a temp dir so no real `$HOME` is ever mutated.
#[cfg(not(windows))]
fn candidate_profiles(base: &std::path::Path) -> Vec<PathBuf> {
    [
        ".zshrc",
        ".zprofile",
        ".bashrc",
        ".bash_profile",
        ".profile",
    ]
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
        Ok(vec![])
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
        Ok(vec![path])
    }
    #[cfg(not(windows))]
    {
        let Ok(home) = std::env::var("HOME") else {
            return Ok(vec![]);
        };
        remove_profile_block_in(std::path::Path::new(&home))
    }
}

/// Has `setup` wired the interceptor env for this user? Windows reads the `HKCU\Environment`
/// value; POSIX reports whether the managed block exists in any candidate shell profile. Note
/// that on POSIX block *presence* no longer implies the env is actively exported: the block is
/// gated on `llmtrim _alive`, so it can be present yet inert while the daemon is down. Callers
/// (`start`, `wrap`) use this only as a "has setup run" signal, not a liveness check.
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

/// The values llmtrim manages in the user environment.
#[cfg(windows)]
const ENV_KEYS: [&str; 4] = [
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
];

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

/// Windows: wire the interceptor env into `HKCU\Environment` (and broadcast the change) when the
/// daemon comes up. The registry is read at process launch, so unlike POSIX there is no per-shell
/// liveness gate — the env is set on daemon start and cleared on `stop`, keeping newly-launched
/// processes in sync with whether the interceptor is actually running. Called from the supervised
/// `serve` path, which is the single process `start` and login autostart both bring the daemon up
/// through. Best-effort; ensures the CA exists so `NODE_EXTRA_CA_CERTS` points at a real file.
#[cfg(windows)]
pub fn wire_env_windows(port: u16) -> Result<()> {
    crate::serve::ensure_ca()?;
    let ca = crate::serve::ca_cert_path()?.to_string_lossy().to_string();
    let proxy = format!("http://127.0.0.1:{port}");
    set_user_env(&proxy, &ca)?;
    broadcast_env_change();
    Ok(())
}

/// Windows: clear the interceptor env from `HKCU\Environment` (and broadcast) when the daemon
/// stops, so a terminal or app launched afterwards doesn't route at the now-dead proxy. Returns
/// true if anything was set. Idempotent: clearing an already-clear env is not an error.
#[cfg(windows)]
pub fn unwire_env_windows() -> Result<bool> {
    let cleared = clear_user_env()?;
    if cleared {
        broadcast_env_change();
    }
    Ok(cleared)
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
    env.set_value("NO_PROXY", &NO_PROXY)
        .context("failed to set NO_PROXY")?;
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
        bytes: encode_utf16_nul(&stripped).into(),
        vtype: raw.vtype,
    };
    env.set_raw_value("Path", &new_raw)
        .context("failed to update the user PATH")?;
    Ok(true)
}

/// Schedule deletion of the install dir once we've exited. A running `.exe` can't be unlinked
/// on Windows, so spawn a detached `cmd` that retries the delete for ~60 s — one shot after a
/// fixed delay loses the race when the shell or Defender still holds the freshly-exited exe.
/// Best-effort: uninstall never fails on this.
#[cfg(windows)]
fn schedule_dir_removal(dir: &std::path::Path) {
    schedule_removal_script(&format!(
        "rmdir /s /q \"{0}\" >nul 2>&1 & if not exist \"{0}\" exit",
        dir.display()
    ));
}

/// Schedule deletion of a single file once we've exited — the running-`.exe` self-delete for
/// builds that live outside `%LOCALAPPDATA%\llmtrim` (e.g. a `cargo install` binary under
/// `~/.cargo/bin`). Same detached-`cmd` trick as [`schedule_dir_removal`], `del` not `rmdir`.
#[cfg(windows)]
fn schedule_file_removal(file: &std::path::Path) {
    schedule_removal_script(&format!(
        "del /f /q \"{0}\" >nul 2>&1 & if not exist \"{0}\" exit",
        file.display()
    ));
}

/// Run `attempt` (which must `exit` on success) every ~2 s for ~60 s in a detached,
/// windowless `cmd`. `ping` is the reliable console-less delay; the retry loop covers the
/// window where the just-exited exe is still locked (shell handle inheritance, Defender
/// scanning fresh executables).
#[cfg(windows)]
fn schedule_removal_script(attempt: &str) {
    use std::os::windows::process::CommandExt;
    use std::process::Stdio;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let script = format!("for /l %i in (1,1,30) do (ping 127.0.0.1 -n 3 >nul & {attempt})");
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

/// Candidate paths to the OS's PEM bundle of trusted root CAs, in probe order — the same list
/// OpenSSL, Go's `crypto/x509`, and `openssl-probe` use. The first that exists is the platform's
/// trust anchor set. POSIX-only: Windows keeps its roots in the schannel cert store, not a file.
#[cfg(not(windows))]
const SYSTEM_CA_CANDIDATES: [&str; 6] = [
    "/etc/ssl/certs/ca-certificates.crt", // Debian, Ubuntu, Arch, Gentoo
    "/etc/pki/tls/certs/ca-bundle.crt",   // Fedora, RHEL, CentOS
    "/etc/ssl/ca-bundle.pem",             // openSUSE
    "/etc/pki/tls/cacert.pem",            // OpenELEC
    "/etc/ssl/cert.pem",                  // Alpine, macOS (Homebrew OpenSSL), *BSD
    "/usr/local/etc/openssl/cert.pem",    // macOS Homebrew openssl@1.1
];

/// First existing OS root bundle, or `None` when none of the well-known paths exist (e.g. a
/// stock macOS with no OpenSSL PEM on disk — trust lives only in the keychain there).
#[cfg(not(windows))]
fn system_ca_bundle() -> Option<PathBuf> {
    SYSTEM_CA_CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

/// Build (or refresh) `~/.llmtrim/ca-bundle.pem` = the OS root bundle **plus** the llmtrim CA,
/// returning its path. Native TLS clients (curl, git, Python, and rustls tools like OpenAI's
/// Codex) don't read `NODE_EXTRA_CA_CERTS` — that's Node-only — they take a full bundle via
/// `SSL_CERT_FILE`/`CURL_CA_BUNDLE`. Because llmtrim MITMs only a fixed host set and
/// blind-tunnels everything else, the bundle MUST carry the real OS roots too, or every
/// non-intercepted HTTPS call would fail to verify — hence a concatenation, not the CA alone.
/// Returns `None` (caller then omits the native vars, since a CA-only file would break tunneled
/// hosts) when no OS bundle can be located.
#[cfg(not(windows))]
fn ensure_ca_bundle(ca_path: &std::path::Path) -> Result<Option<PathBuf>> {
    let Some(system) = system_ca_bundle() else {
        return Ok(None);
    };
    let system_pem = std::fs::read_to_string(&system)
        .with_context(|| format!("failed to read {}", system.display()))?;
    let ca_pem = std::fs::read_to_string(ca_path)
        .with_context(|| format!("failed to read {}", ca_path.display()))?;
    let mut combined = system_pem;
    if !combined.ends_with('\n') {
        combined.push('\n');
    }
    combined.push_str(&ca_pem);
    let bundle_path = ca_path.with_file_name("ca-bundle.pem");
    std::fs::write(&bundle_path, combined)
        .with_context(|| format!("failed to write {}", bundle_path.display()))?;
    Ok(Some(bundle_path))
}

/// The managed env block, in the profile's native syntax. Both variants are unit-tested on
/// every platform; on Windows the live env path is the registry, so this is test-only there.
/// `bundle` (POSIX only) is the combined OS-roots + CA bundle for native TLS clients; when
/// `Some`, it also exports `SSL_CERT_FILE`/`CURL_CA_BUNDLE`.
#[allow(dead_code)]
fn env_block(proxy: &str, ca: &str, bundle: Option<&str>, syntax: Syntax) -> String {
    match syntax {
        // NO_PROXY is set in both cases (lowercase too on POSIX: curl/libcurl, Go, and others
        // only honor `no_proxy`). Windows env vars are case-insensitive, so one suffices there.
        // Wire the env only while the daemon is actually up. A new shell opened after
        // `llmtrim stop` must not route at a now-dead proxy — so the block gates every export
        // behind `llmtrim _alive`, a fast pidfile check (no network, no TCP connect). Daemon
        // down → the vars stay unset → tools talk to the LLM host directly and just work.
        // The current shell can't be fixed this way (a child can't rewrite its parent's env);
        // that's what `stop`'s printed `unset` line and `uninstall` handle.
        Syntax::Posix => {
            // Node trusts the CA via NODE_EXTRA_CA_CERTS; native OpenSSL/rustls clients don't
            // read it, so point their bundle vars at the full combined bundle when we have one.
            // Indented to sit inside the liveness guard alongside the other exports.
            let native = bundle
                .map(|b| {
                    format!(
                        "\x20   export SSL_CERT_FILE=\"{b}\"\n\
                         \x20   export CURL_CA_BUNDLE=\"{b}\"\n"
                    )
                })
                .unwrap_or_default();
            format!(
                "{BEGIN}\n\
                 if command -v llmtrim >/dev/null 2>&1 && llmtrim _alive 2>/dev/null; then\n\
                 \x20   export HTTPS_PROXY=\"{proxy}\"\n\
                 \x20   export HTTP_PROXY=\"{proxy}\"\n\
                 \x20   export NO_PROXY=\"{NO_PROXY}\"\n\
                 \x20   export no_proxy=\"{NO_PROXY}\"\n\
                 \x20   export NODE_EXTRA_CA_CERTS=\"{ca}\"\n\
                 {native}fi\n\
                 {END}\n"
            )
        }
        Syntax::PowerShell => format!(
            "{BEGIN}\n\
             $env:HTTPS_PROXY = \"{proxy}\"\n\
             $env:HTTP_PROXY = \"{proxy}\"\n\
             $env:NO_PROXY = \"{NO_PROXY}\"\n\
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
    bundle: Option<&str>,
) -> Result<Vec<PathBuf>> {
    // Refresh, never accumulate: strip any prior managed block from EVERY candidate first,
    // so a re-run (or a proxy/port change) leaves no stale block behind. Best-effort - a
    // failure to strip one file must not block writing the fresh block.
    let _ = remove_profile_block_in(base);

    // Write the block into every candidate rc file that already EXISTS, plus the file the
    // user's `$SHELL` sources (created if absent). Writing to all existing files is what
    // fixes the macOS trap: setup keyed the target off `$SHELL`, but `$SHELL` is the login
    // shell and can disagree with the shell the terminal actually launches (iTerm running
    // zsh while `$SHELL=/bin/bash`), so the block landed in a file the running shell never
    // sourced. Now whichever shell starts, its rc already carries the env. The managed
    // BEGIN/END markers make a block in several files idempotent and cleanly removable.
    let mut targets: Vec<PathBuf> = candidate_profiles(base)
        .into_iter()
        .filter(|p| p.exists())
        .collect();
    let shell_default = base.join(shell_profile_file(shell));
    if !targets.contains(&shell_default) {
        targets.push(shell_default); // guarantee the running shell's rc gets it, even if absent
    }

    let block = env_block(proxy, ca, bundle, Syntax::Posix);
    let mut written = Vec::with_capacity(targets.len());
    for path in targets {
        // strip_block is a safety net in case remove_profile_block_in skipped a file with a
        // broken BEGIN-without-END that it left intact.
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let mut base_content = strip_block(&existing);
        if !base_content.is_empty() && !base_content.ends_with('\n') {
            base_content.push('\n');
        }
        std::fs::write(&path, format!("{base_content}{block}"))
            .with_context(|| format!("failed to write {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// Replace (or append) the llmtrim managed block in the shell profile. Idempotent — a
/// re-run updates the existing block rather than stacking duplicates. Also sweeps stale
/// blocks from all other candidate profile files (e.g. `.bashrc` when now running zsh),
/// so re-setup under a different shell does not leave a dead proxy block behind.
/// POSIX-only: on Windows the env lives in the registry, so `run()` never calls this there.
#[allow(dead_code)]
fn write_profile_block(proxy: &str, ca: &str, bundle: Option<&str>) -> Result<Vec<PathBuf>> {
    #[cfg(not(windows))]
    {
        // Delegate to the seam so production and tests run the same code path.
        let Ok(home) = std::env::var("HOME") else {
            return Ok(Vec::new());
        };
        let shell = std::env::var("SHELL").unwrap_or_default();
        write_profile_block_in(std::path::Path::new(&home), &shell, proxy, ca, bundle)
    }
    #[cfg(windows)]
    {
        // Legacy PowerShell-profile arm: the live Windows env is the registry; this is
        // only reachable for a profile-style install a prior version may have used.
        let Some((path, syntax)) = profile_target() else {
            return Ok(Vec::new());
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent); // the PowerShell profile dir may not exist yet
        }
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let mut base_content = strip_block(&existing);
        if !base_content.is_empty() && !base_content.ends_with('\n') {
            base_content.push('\n');
        }
        let block = env_block(proxy, ca, bundle, syntax);
        std::fs::write(&path, format!("{base_content}{block}"))
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(vec![path])
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
    fn caveman_detection_covers_flag_hooks_and_plugin() {
        let tmp = std::env::temp_dir().join(format!("llmtrim-caveman-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);

        // Empty / missing dir → not installed.
        assert!(!caveman_installed_in(&tmp));
        std::fs::create_dir_all(&tmp).expect("create test dir");
        assert!(!caveman_installed_in(&tmp));

        // Flag file written by caveman's session hook.
        std::fs::write(tmp.join(".caveman-active"), "1").expect("write flag");
        assert!(caveman_installed_in(&tmp));
        std::fs::remove_file(tmp.join(".caveman-active")).expect("rm flag");

        // Standalone hook file.
        std::fs::create_dir_all(tmp.join("hooks")).expect("mkdir hooks");
        std::fs::write(tmp.join("hooks/caveman-activate.js"), "//").expect("write hook");
        assert!(caveman_installed_in(&tmp));
        std::fs::remove_file(tmp.join("hooks/caveman-activate.js")).expect("rm hook");
        assert!(!caveman_installed_in(&tmp));

        // Plugin-cache entry, nested one level (within the depth-3 walk).
        std::fs::create_dir_all(tmp.join("plugins/cache/caveman")).expect("mkdir plugin");
        assert!(caveman_installed_in(&tmp));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(not(windows))]
    #[test]
    fn heal_rewrites_old_block_adds_no_proxy_preserving_port_and_surroundings() {
        let tmp = std::env::temp_dir().join(format!("llmtrim-heal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");

        // An old-schema block (no NO_PROXY) wired at port 43117, with user content around it.
        let old = format!(
            "export FOO=bar\n{BEGIN}\n\
             export HTTPS_PROXY=\"http://127.0.0.1:43117\"\n\
             export HTTP_PROXY=\"http://127.0.0.1:43117\"\n\
             export NODE_EXTRA_CA_CERTS=\"/home/u/ca.pem\"\n{END}\nexport BAZ=qux\n"
        );
        let rc = tmp.join(".bashrc");
        std::fs::write(&rc, &old).expect("write rc");

        let healed = heal_profiles_in(&tmp, "http://127.0.0.1:43117", "/home/u/ca.pem", None)
            .expect("heal runs");
        assert_eq!(healed, vec![rc.clone()], "the old block is healed");

        let after = std::fs::read_to_string(&rc).expect("read back");
        assert!(after.contains("export NO_PROXY="), "bypass added");
        assert!(after.contains("127.0.0.1:43117"), "port preserved");
        assert!(after.contains("export FOO=bar") && after.contains("export BAZ=qux"));
        assert_eq!(after.matches(BEGIN).count(), 1, "block not duplicated");

        // Idempotent: a second pass sees the bypass and does nothing.
        let again = heal_profiles_in(&tmp, "http://127.0.0.1:43117", "/home/u/ca.pem", None)
            .expect("second heal");
        assert!(again.is_empty(), "already-current block is left alone");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(not(windows))]
    #[test]
    fn managed_block_needs_heal_distinguishes_old_current_malformed_and_outside() {
        let old = format!("{BEGIN}\nexport HTTPS_PROXY=\"x\"\n{END}\n");
        assert!(managed_block_needs_heal(&old), "old block needs heal");

        // Pre-gate block: has the NO_PROXY bypass but not the `llmtrim _alive` liveness gate. It
        // must still heal so existing installs migrate to the daemon-gated form (the fix that lets
        // `stop` unwire new shells) without the user re-running `setup`.
        let pre_gate = format!("{BEGIN}\nexport NO_PROXY=\"y\"\n{END}\n");
        assert!(
            managed_block_needs_heal(&pre_gate),
            "block missing the _alive gate needs heal"
        );

        // The actually-current block carries both the bypass and the gate, so it is skipped.
        let current = env_block(
            "http://127.0.0.1:8787",
            "/home/u/ca.pem",
            None,
            Syntax::Posix,
        );
        assert!(!managed_block_needs_heal(&current), "current block skipped");

        let malformed = format!("{BEGIN}\nexport HTTPS_PROXY=\"x\"\n"); // no END
        assert!(
            !managed_block_needs_heal(&malformed),
            "malformed block skipped"
        );

        // A user's own NO_PROXY *outside* the markers must not mask a stale managed block.
        let outside = format!("export NO_PROXY=mine\n{BEGIN}\nexport HTTPS_PROXY=\"x\"\n{END}\n");
        assert!(
            managed_block_needs_heal(&outside),
            "NO_PROXY outside the block does not count as healed"
        );

        assert!(!managed_block_needs_heal("export FOO=bar\n"), "no block");
    }

    #[cfg(not(windows))]
    #[test]
    fn heal_skips_malformed_block_without_corrupting_it() {
        let tmp = std::env::temp_dir().join(format!("llmtrim-heal-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");
        // BEGIN with no END (user deleted it). strip_block would no-op, so a naive heal would
        // stack a second block; managed_block_needs_heal must reject it.
        let bad = format!("keep\n{BEGIN}\nexport HTTPS_PROXY=\"http://127.0.0.1:43117\"\n");
        std::fs::write(tmp.join(".bashrc"), &bad).expect("write");

        let healed =
            heal_profiles_in(&tmp, "http://127.0.0.1:43117", "/home/u/ca.pem", None).expect("heal");
        assert!(healed.is_empty(), "malformed block is not healed");
        assert_eq!(
            std::fs::read_to_string(tmp.join(".bashrc")).expect("read"),
            bad,
            "malformed file left byte-for-byte untouched"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(not(windows))]
    #[test]
    fn heal_touches_only_the_stale_profile_in_a_multi_shell_setup() {
        let tmp = std::env::temp_dir().join(format!("llmtrim-heal-multi-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");
        // .bashrc carries an old block; .zshrc already has the current (NO_PROXY) block.
        let old = format!("{BEGIN}\nexport HTTPS_PROXY=\"http://127.0.0.1:43117\"\n{END}\n");
        let current = env_block(
            "http://127.0.0.1:43117",
            "/home/u/ca.pem",
            None,
            Syntax::Posix,
        );
        std::fs::write(tmp.join(".bashrc"), &old).expect("write bashrc");
        std::fs::write(tmp.join(".zshrc"), &current).expect("write zshrc");

        let healed =
            heal_profiles_in(&tmp, "http://127.0.0.1:43117", "/home/u/ca.pem", None).expect("heal");
        assert_eq!(
            healed,
            vec![tmp.join(".bashrc")],
            "only the stale profile healed"
        );
        assert!(
            std::fs::read_to_string(tmp.join(".bashrc"))
                .expect("read bashrc")
                .contains("NO_PROXY"),
            "stale profile now carries the bypass"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.join(".zshrc")).expect("read zshrc"),
            current,
            "already-current profile left untouched"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(not(windows))]
    #[test]
    fn heal_ignores_files_without_a_managed_block() {
        let tmp = std::env::temp_dir().join(format!("llmtrim-heal-none-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");
        std::fs::write(tmp.join(".zshrc"), "export FOO=bar\n").expect("write");

        let healed =
            heal_profiles_in(&tmp, "http://127.0.0.1:43117", "/home/u/ca.pem", None).expect("heal");
        assert!(
            healed.is_empty(),
            "a file with no block is never created/touched"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.join(".zshrc")).expect("read"),
            "export FOO=bar\n",
            "content untouched"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

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

    /// The public boundary `llmtrim autostart` resolves through. A running daemon's port must
    /// win over any default — this is the regression `fix/autostart-port-resolve` locks in
    /// (bare `autostart` used to hardcode the default). Deterministic: `running` short-circuits
    /// `choose_port` before `configured_port()` reads any real env.
    #[test]
    fn resolve_port_prefers_running_over_default() {
        assert_eq!(resolve_port(None, Some(9001)).expect("running port"), 9001);
        assert_eq!(
            resolve_port(Some(7000), Some(9001)).expect("explicit"),
            7000
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
        let b = env_block(
            "http://127.0.0.1:8787",
            "/home/u/ca.pem",
            None,
            Syntax::Posix,
        );
        assert!(b.contains("export HTTPS_PROXY=\"http://127.0.0.1:8787\""));
        assert!(b.contains("export NODE_EXTRA_CA_CERTS=\"/home/u/ca.pem\""));
        // LAN/local bypass travels with the proxy, in both casings tools read.
        assert!(b.contains(&format!("export NO_PROXY=\"{NO_PROXY}\"")));
        assert!(b.contains(&format!("export no_proxy=\"{NO_PROXY}\"")));
        assert!(b.starts_with(BEGIN) && b.trim_end().ends_with(END));
        // No bundle → native TLS vars are omitted (a CA-only file would break tunneled hosts).
        assert!(!b.contains("SSL_CERT_FILE"));
        assert!(!b.contains("CURL_CA_BUNDLE"));
    }

    #[test]
    fn env_block_posix_with_bundle_exports_native_tls_vars() {
        let b = env_block(
            "http://127.0.0.1:8787",
            "/home/u/.llmtrim/ca.pem",
            Some("/home/u/.llmtrim/ca-bundle.pem"),
            Syntax::Posix,
        );
        // Node keeps NODE_EXTRA_CA_CERTS; native clients (curl, rustls) get the full bundle.
        assert!(b.contains("export NODE_EXTRA_CA_CERTS=\"/home/u/.llmtrim/ca.pem\""));
        assert!(b.contains("export SSL_CERT_FILE=\"/home/u/.llmtrim/ca-bundle.pem\""));
        assert!(b.contains("export CURL_CA_BUNDLE=\"/home/u/.llmtrim/ca-bundle.pem\""));
        assert!(b.trim_end().ends_with(END));
    }

    #[cfg(not(windows))] // system_ca_bundle/ensure_ca_bundle are POSIX-only (Windows uses the cert store)
    #[test]
    fn ensure_ca_bundle_concats_roots_and_ca() {
        // Only meaningful where the OS ships a PEM root bundle (Linux CI, some macOS). Where it
        // doesn't (stock macOS), the builder correctly returns None and there is nothing to check.
        let Some(system) = system_ca_bundle() else {
            return;
        };
        let tmp = std::env::temp_dir().join(format!("llmtrim-catest-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let ca_path = tmp.join("ca.pem");
        std::fs::write(
            &ca_path,
            "-----BEGIN CERTIFICATE-----\nLLMTRIMFAKE\n-----END CERTIFICATE-----\n",
        )
        .expect("write ca");

        let bundle = ensure_ca_bundle(&ca_path)
            .expect("build")
            .expect("some bundle");
        assert_eq!(bundle, tmp.join("ca-bundle.pem"));
        let contents = std::fs::read_to_string(&bundle).expect("read bundle");
        // Carries both the OS roots (so tunneled hosts still verify) and our CA (so MITM'd ones do).
        let roots = std::fs::read_to_string(&system).expect("read system");
        assert!(contents.contains(roots.trim()));
        assert!(contents.contains("LLMTRIMFAKE"));

        std::fs::remove_dir_all(&tmp).ok();
    }

    // The block is sourced into every user's .bashrc/.zshrc, so a shell-syntax error breaks their
    // terminal startup — worse than a wrong var. `sh -n`/`bash -n` parse without executing, so
    // this catches an unbalanced if/fi or a bad quote that string-offset checks can't.
    #[cfg(unix)]
    #[test]
    fn env_block_posix_is_syntactically_valid_shell() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let block = env_block(
            "http://127.0.0.1:8787",
            "/home/u/ca.pem",
            None,
            Syntax::Posix,
        );
        for shell in ["sh", "bash"] {
            let mut child = match Command::new(shell)
                .arg("-n")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(c) => c,
                Err(_) => continue, // shell not on this runner — skip
            };
            child
                .stdin
                .take()
                .unwrap()
                .write_all(block.as_bytes())
                .unwrap();
            let status = child.wait().unwrap();
            assert!(status.success(), "{shell} -n rejected the block:\n{block}");
        }
    }

    #[test]
    fn env_block_posix_gates_exports_on_daemon_liveness() {
        // The exports must be guarded by the `_alive` probe so a new shell opened after
        // `stop` doesn't wire itself at a dead proxy. Both the guard and a `fi` must be present,
        // and every export must sit inside the conditional (i.e. after the `if` line).
        let b = env_block(
            "http://127.0.0.1:8787",
            "/home/u/ca.pem",
            None,
            Syntax::Posix,
        );
        assert!(
            b.contains("llmtrim _alive"),
            "block must probe daemon liveness"
        );
        let if_at = b.find("if command -v llmtrim").expect("guard present");
        let fi_at = b.find("\nfi\n").expect("guard closed with fi");
        let export_at = b.find("export HTTPS_PROXY").expect("export present");
        assert!(
            if_at < export_at && export_at < fi_at,
            "exports must live inside the liveness guard"
        );

        // The native-TLS bundle exports (added for curl/git/rustls clients) must be gated too,
        // so a new shell after `stop` doesn't keep trusting the interceptor CA at a dead proxy.
        let with_bundle = env_block(
            "http://127.0.0.1:8787",
            "/home/u/ca.pem",
            Some("/home/u/.llmtrim/ca-bundle.pem"),
            Syntax::Posix,
        );
        let fi_at = with_bundle.find("\nfi\n").expect("guard closed with fi");
        let ssl_at = with_bundle
            .find("export SSL_CERT_FILE")
            .expect("bundle exports present");
        assert!(
            ssl_at < fi_at,
            "native-TLS exports must sit inside the guard"
        );
    }

    #[test]
    fn env_block_powershell_uses_env_assignment() {
        let b = env_block(
            "http://127.0.0.1:8787",
            "C:\\Users\\u\\ca.pem",
            None,
            Syntax::PowerShell,
        );
        assert!(b.contains("$env:HTTPS_PROXY = \"http://127.0.0.1:8787\""));
        assert!(b.contains("$env:NODE_EXTRA_CA_CERTS = \"C:\\Users\\u\\ca.pem\""));
        assert!(b.contains(&format!("$env:NO_PROXY = \"{NO_PROXY}\"")));
        assert!(!b.contains("export ")); // no posix syntax leaked in
    }

    #[test]
    fn strip_block_reverses_powershell_block() {
        let withblock = format!("keep\n{}", env_block("p", "c", None, Syntax::PowerShell));
        assert_eq!(strip_block(&withblock), "keep\n");
    }

    #[test]
    fn write_then_strip_is_idempotent() {
        // Writing a block then stripping it returns to the original content.
        let original = "export FOO=bar\n";
        let proxy = "http://127.0.0.1:8787";
        let ca = "/home/user/.llmtrim/ca.crt";
        let block = env_block(proxy, ca, None, Syntax::Posix);
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
        let block = env_block(proxy, ca, None, Syntax::Posix);
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

        // With the port still held, a wider scan skips the occupied port and returns a free
        // one further along. (Re-probing the *same* port after drop is racy: under parallel
        // tests another thread can grab the freed ephemeral port before we rebind it.)
        let found = first_free_port(taken, 64).expect("a free port exists in the span");
        assert_ne!(found, taken, "scan returned the still-occupied port");
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

        let paths =
            write_profile_block_in(base, "bash", proxy, ca, None).expect("write_profile_block_in");
        // Only `.bashrc` exists and it is the bash default -> that's the sole target.
        assert_eq!(paths, vec![base.join(".bashrc")]);

        let content = std::fs::read_to_string(base.join(".bashrc")).expect("read back");
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

        write_profile_block_in(base, "bash", proxy, ca, None).expect("first write");
        write_profile_block_in(base, "bash", proxy, ca, None).expect("second write");

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

    /// Re-setup writes to every existing rc file AND the `$SHELL` default, refreshing (not
    /// orphaning) a prior block. Prior bash setup left a block (old port) in `.bashrc`;
    /// re-running under zsh with a new port must update `.bashrc` in place *and* create
    /// `.zshrc`, with the old port gone from both.
    #[cfg(not(windows))]
    #[test]
    fn write_profile_block_in_refreshes_all_existing_and_creates_shell_default() {
        let dir = TempDir::new("wpb-switch");
        let base = dir.path();
        let ca = "/tmp/ca.crt";

        // Prior bash setup: `.bashrc` carries a managed block pinned to the OLD port (8787).
        write_block_to(base, ".bashrc");

        // Re-setup under zsh, NEW port.
        let new_proxy = "http://127.0.0.1:9999";
        let paths =
            write_profile_block_in(base, "zsh", new_proxy, ca, None).expect("write for zsh");
        // Both the existing `.bashrc` and the freshly-created zsh default are written.
        assert!(
            paths.contains(&base.join(".bashrc")),
            "existing .bashrc written: {paths:?}"
        );
        assert!(
            paths.contains(&base.join(".zshrc")),
            "zsh default created: {paths:?}"
        );

        for name in [".bashrc", ".zshrc"] {
            let body = std::fs::read_to_string(base.join(name)).expect("read rc");
            assert_eq!(body.matches(BEGIN).count(), 1, "{name}: exactly one block");
            assert!(body.contains("9999"), "{name}: refreshed to the new port");
            assert!(
                !body.contains("8787"),
                "{name}: old port must be gone (no stale block)"
            );
        }
    }

    /// `$SHELL` default is created even when NO rc file exists yet, and ONLY that file
    /// (no littering `.bashrc`/`.profile` on a fresh zsh-only home).
    #[cfg(not(windows))]
    #[test]
    fn write_profile_block_in_creates_only_the_shell_default_when_none_exist() {
        let dir = TempDir::new("wpb-none");
        let base = dir.path();
        let paths =
            write_profile_block_in(base, "/bin/zsh", "http://127.0.0.1:8788", "/tmp/ca", None)
                .expect("write");
        assert_eq!(
            paths,
            vec![base.join(".zshrc")],
            "only the zsh default is created"
        );
        assert!(base.join(".zshrc").exists());
        for other in [".zprofile", ".bashrc", ".bash_profile", ".profile"] {
            assert!(!base.join(other).exists(), "{other} must not be created");
        }
    }

    /// Multiple shells installed (several rc files present): the block lands in every one,
    /// so whichever shell the terminal launches is covered.
    #[cfg(not(windows))]
    #[test]
    fn write_profile_block_in_writes_to_all_existing_rc_files() {
        let dir = TempDir::new("wpb-all");
        let base = dir.path();
        for f in [".zshrc", ".bashrc", ".profile"] {
            std::fs::write(base.join(f), "# pre-existing\n").expect("seed rc");
        }
        let paths =
            write_profile_block_in(base, "/bin/bash", "http://127.0.0.1:8788", "/tmp/ca", None)
                .expect("write");
        for f in [".zshrc", ".bashrc", ".profile"] {
            assert!(paths.contains(&base.join(f)), "{f} written: {paths:?}");
            let body = std::fs::read_to_string(base.join(f)).expect("read");
            assert_eq!(body.matches(BEGIN).count(), 1, "{f}: one block");
            assert!(body.contains("# pre-existing"), "{f}: kept user content");
        }
        // The two login rc files that did NOT exist stay absent (only existing + default).
        assert!(!base.join(".zprofile").exists());
        assert!(!base.join(".bash_profile").exists());
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

    /// POSIX block: the env lines are valid `export KEY="value"` lines, wrapped in the daemon
    /// liveness guard (`if … ; then` / `fi`). Every non-marker line is either the guard, the
    /// closing `fi`, or an export.
    #[test]
    fn env_block_posix_all_lines_are_valid_exports() {
        let proxy = "http://127.0.0.1:8787";
        let ca = "/home/user/.llmtrim/ca.crt";
        let block = env_block(proxy, ca, None, Syntax::Posix);

        // Collect non-marker, non-empty lines inside the block.
        let inner: Vec<&str> = block.lines().filter(|l| *l != BEGIN && *l != END).collect();
        assert!(!inner.is_empty(), "no inner lines in POSIX block");
        for line in &inner {
            let trimmed = line.trim_start();
            let is_guard = trimmed.starts_with("if ") || trimmed == "fi";
            let is_export = trimmed.starts_with("export ") && trimmed.contains("=\"");
            assert!(
                is_guard || is_export,
                "line is neither the liveness guard nor a valid export: {line:?}"
            );
        }
    }

    /// PowerShell block: every line between the markers uses `$env:` assignment syntax, never
    /// POSIX `export`.
    #[test]
    fn env_block_powershell_all_lines_use_dollar_env() {
        let proxy = "http://127.0.0.1:8787";
        let ca = "C:\\Users\\u\\ca.pem";
        let block = env_block(proxy, ca, None, Syntax::PowerShell);

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
