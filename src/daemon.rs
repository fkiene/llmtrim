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
    };
    std::fs::write(pidfile()?, serde_json::to_string(&state)?)?;
    Ok(())
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
    Ok(pid)
}

/// Stop the running daemon (SIGTERM) and clear the pidfile. Returns the stopped pid.
pub fn stop() -> Result<Option<u32>> {
    let Some(state) = read_state() else {
        return Ok(None);
    };
    if is_alive(state.pid) {
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
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: DaemonState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pid, 123);
        assert_eq!(back.port, 8787);
    }
}
