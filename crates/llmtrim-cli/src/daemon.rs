//! Background-daemon control for the interceptor: a pidfile under `~/.llmtrim`, plus
//! detached-spawn / liveness / stop. Pure std (no async, no GUI) — the rich CLI face of
//! the always-on proxy. `status` reads this plus the savings ledger.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Recorded state of a running interceptor daemon (the pidfile contents).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonState {
    pub pid: u32,
    pub port: u16,
    /// Unix seconds when the daemon started (for uptime).
    pub started_at: i64,
    /// Version of the binary that spawned the daemon (`None` in pidfiles written
    /// before this field existed) — lets `status` flag a stale daemon after `update`.
    #[serde(default)]
    pub version: Option<String>,
    /// Crash-restarts the supervisor performed since start (0 = clean run).
    #[serde(default)]
    pub restarts: u32,
}

/// Base directory for llmtrim state (`$LLMTRIM_HOME` or `~/.llmtrim`). Falls back to
/// `%USERPROFILE%` on Windows, where `HOME` is usually unset.
pub fn home_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LLMTRIM_HOME") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("neither HOME nor USERPROFILE is set")?;
    Ok(PathBuf::from(home).join(".llmtrim"))
}

fn pidfile() -> Result<PathBuf> {
    Ok(home_dir()?.join("serve.pid"))
}

pub fn logfile() -> Result<PathBuf> {
    Ok(home_dir()?.join("serve.log"))
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Write the pidfile for a just-started daemon.
pub fn write_state(pid: u32, port: u16) -> Result<()> {
    let dir = home_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let state = DaemonState {
        pid,
        port,
        started_at: now_secs(),
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        restarts: 0,
    };
    std::fs::write(pidfile()?, serde_json::to_string(&state)?)?;
    Ok(())
}

/// Bump the supervisor's crash-restart counter in the pidfile (best-effort: the counter
/// is diagnostics for `status`, never worth failing a restart over).
pub fn bump_restarts() {
    if let Some(mut state) = read_state() {
        state.restarts = state.restarts.saturating_add(1);
        if let Ok(path) = pidfile()
            && let Ok(json) = serde_json::to_string(&state)
        {
            let _ = std::fs::write(path, json);
        }
    }
}

/// The recorded daemon state, if the pidfile exists and parses.
pub fn read_state() -> Option<DaemonState> {
    let text = std::fs::read_to_string(pidfile().ok()?).ok()?;
    serde_json::from_str(&text).ok()
}

/// True if `tasklist` CSV output reports a process with this pid. `tasklist` prints
/// `INFO: No tasks ...` to stdout when nothing matches; a match is a CSV row whose second
/// field is the pid. Pure + unit-tested so the Windows liveness logic is verifiable off-Windows.
#[cfg(any(windows, test))]
fn tasklist_reports_pid(stdout: &str, pid: u32) -> bool {
    stdout.lines().any(|line| {
        line.split(',')
            .nth(1)
            .map(|field| field.trim().trim_matches('"') == pid.to_string())
            .unwrap_or(false)
    })
}

/// True if the process at `pid` is the llmtrim binary (guards against killing a foreign process
/// that recycled a stale pidfile entry). Permissive on I/O errors and unknown platforms — it
/// is better to attempt a kill on an uncertain match than to leave a real daemon running.
fn pid_is_llmtrim(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
        comm.trim().starts_with("llmtrim")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        true
    }
}

/// Is a process with this pid alive? `kill -0` on Unix, `tasklist` on Windows — both report
/// whether the process exists without touching it.
pub fn is_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .map(|o| tasklist_reports_pid(&String::from_utf8_lossy(&o.stdout), pid))
            .unwrap_or(false)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// The running daemon, if the pidfile points at a live process. Clears a stale pidfile.
pub fn running() -> Option<DaemonState> {
    let state = read_state()?;
    if is_alive(state.pid) {
        Some(state)
    } else {
        let _ = std::fs::remove_file(pidfile().ok()?);
        None
    }
}

/// Uptime (seconds) for a daemon started at `started_at`.
pub fn uptime_secs(started_at: i64) -> i64 {
    (now_secs() - started_at).max(0)
}

/// Is anything accepting TCP on `127.0.0.1:port`? A live pidfile only proves the
/// *supervisor* process exists — the proxy inside it can be crash-looping on a bind
/// failure while `kill -0` stays green. One connect (~1ms, 300ms cap) closes that gap.
pub fn probe_port(port: u16) -> bool {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300))
        .map(|s| {
            let _ = s.shutdown(std::net::Shutdown::Both);
            true
        })
        .unwrap_or(false)
}

/// Spawn the interceptor as a detached background process, redirecting its output to the
/// logfile and recording the pidfile. Returns the child pid.
pub fn spawn_detached(port: u16) -> Result<u32> {
    if let Some(state) = running() {
        anyhow::bail!(
            "interceptor already running (pid {}, port {}) — `llmtrim stop` first",
            state.pid,
            state.port
        );
    }
    let exe = std::env::current_exe().context("could not find the llmtrim executable")?;
    let dir = home_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    if let Ok(log_path) = logfile()
        && std::fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0) > 10 * 1024 * 1024
    {
        let rotated = log_path.with_extension("log.1");
        let _ = std::fs::rename(&log_path, rotated);
    }
    let log = std::fs::File::create(logfile()?)?;
    let log_err = log.try_clone()?;

    let mut cmd = std::process::Command::new(exe);
    // `--supervised`: the detached process restarts the proxy on crash, so a dead daemon
    // (which would break the client's HTTPS_PROXY entirely) self-heals.
    cmd.args(["serve", "--port", &port.to_string(), "--supervised"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err));
    // Detach so the daemon survives our exit.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // leave the controlling terminal's process group
    }
    #[cfg(windows)]
    {
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: no inherited
        // console, own process group, survives the launching shell.
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    let child = cmd.spawn().context("failed to spawn the interceptor")?;
    let pid = child.id();
    write_state(pid, port)?;
    // Readiness: the proxy warms its tokenizer tables before binding (~2-3s), so returning
    // at fork time makes an immediate `status` read "degraded" and an immediate request
    // fail. Poll until the port accepts (10s cap) so success means "serving", not "forked".
    for _ in 0..100 {
        if probe_port(port) {
            return Ok(pid);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    eprintln!(
        "llmtrim: interceptor spawned (pid {pid}) but :{port} is not accepting after 10s — \
         check `llmtrim status` and the log"
    );
    Ok(pid)
}

/// Stop the running daemon (SIGTERM) and clear the pidfile. Returns the stopped pid.
/// Blocks until the process actually exits (5s cap): returning at signal time lets an
/// immediate `start` race the dying daemon for the port — the new proxy hits
/// EADDRINUSE, crash-restarts once, and `status` reports a crash we caused ourselves.
pub fn stop() -> Result<Option<u32>> {
    let Some(state) = read_state() else {
        return Ok(None);
    };
    if is_alive(state.pid) {
        if !pid_is_llmtrim(state.pid) {
            eprintln!(
                "llmtrim: pidfile points to foreign process (pid {}), skipping kill",
                state.pid
            );
        } else {
            #[cfg(unix)]
            {
                let _ = std::process::Command::new("kill")
                    .arg(state.pid.to_string())
                    .status();
            }
            #[cfg(windows)]
            {
                // /T kills the child tree, /F forces termination.
                let _ = std::process::Command::new("taskkill")
                    .args(["/PID", &state.pid.to_string(), "/T", "/F"])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            }
            for _ in 0..50 {
                if !is_alive(state.pid) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            if is_alive(state.pid) {
                eprintln!(
                    "llmtrim: pid {} still running 5s after the stop signal",
                    state.pid
                );
            }
        }
    }
    let _ = std::fs::remove_file(pidfile()?);
    Ok(Some(state.pid))
}

/// Format a duration in seconds as `3h12m` / `5m` / `42s`.
pub fn human_uptime(secs: i64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

/// Humanized age of an rfc3339 timestamp ("4s ago", "3h12m ago"); `None` if it doesn't
/// parse. Clock skew putting it in the future clamps to "0s ago" rather than lying.
pub fn human_age(rfc3339: &str) -> Option<String> {
    let t = chrono::DateTime::parse_from_rfc3339(rfc3339).ok()?;
    let secs = (chrono::Utc::now().timestamp() - t.timestamp()).max(0);
    Some(format!("{} ago", human_uptime(secs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_uptime_formats() {
        assert_eq!(human_uptime(42), "42s");
        assert_eq!(human_uptime(305), "5m05s");
        assert_eq!(human_uptime(3 * 3600 + 12 * 60), "3h12m");
    }

    #[test]
    fn human_age_parses_and_clamps() {
        let past = (chrono::Utc::now() - chrono::Duration::seconds(40)).to_rfc3339();
        let s = human_age(&past).expect("parses");
        assert!(s.ends_with(" ago"), "{s}");
        // Future timestamps (clock skew) clamp to zero instead of going negative.
        let future = (chrono::Utc::now() + chrono::Duration::seconds(120)).to_rfc3339();
        assert_eq!(human_age(&future).expect("parses"), "0s ago");
        assert_eq!(human_age("not a timestamp"), None);
    }

    #[test]
    fn tasklist_parse_detects_pid() {
        // A real `tasklist /FO CSV /NH` row, and the "no match" message.
        let row = "\"llmtrim.exe\",\"4242\",\"Console\",\"1\",\"12,345 K\"";
        assert!(tasklist_reports_pid(row, 4242));
        assert!(!tasklist_reports_pid(row, 99)); // 99 must not match "12,345 K" etc.
        assert!(!tasklist_reports_pid(
            "INFO: No tasks are running which match the specified criteria.",
            4242
        ));
    }

    #[test]
    fn state_round_trips() {
        let s = DaemonState {
            pid: 123,
            port: 8787,
            started_at: 1000,
            version: Some("0.1.0".into()),
            restarts: 2,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: DaemonState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pid, 123);
        assert_eq!(back.port, 8787);
        assert_eq!(back.version.as_deref(), Some("0.1.0"));
        assert_eq!(back.restarts, 2);
    }

    #[test]
    fn state_parses_pre_version_pidfile() {
        // Pidfiles written before version/restarts existed must still parse (upgrade path).
        let back: DaemonState =
            serde_json::from_str(r#"{"pid":9,"port":8787,"started_at":1000}"#).unwrap();
        assert_eq!(back.version, None);
        assert_eq!(back.restarts, 0);
    }

    #[test]
    fn probe_port_detects_listener_and_absence() {
        // Bind an ephemeral port → probe sees it; drop it → probe fails.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let port = listener.local_addr().expect("local addr").port();
        assert!(probe_port(port));
        drop(listener);
        assert!(!probe_port(port));
    }
}
