//! `llmtrim doctor` — end-to-end install diagnosis (optionally repairs with `--fix`).
//!
//! One pass/fail row per link in the chain (binary → daemon → port → env → CA →
//! autostart → ledger → integrations), each failing row naming its fix. `status` shows
//! the same chain compressed to a header; doctor is the long form for "it doesn't work,
//! why?". Pass `--fix` to run [`crate::ensure`] first. Gathering is separated from
//! rendering so the row logic is unit-testable.

use crate::ui;

/// Everything doctor probes, gathered once (see [`gather`]) and turned into rows by the
/// pure [`build`].
pub struct State {
    pub exe: String,
    pub binary_version: String,
    pub running: bool,
    pub pid: u32,
    pub port: u16,
    pub uptime: String,
    pub daemon_version: Option<String>,
    pub restarts: u32,
    pub port_accepting: bool,
    /// Port wired into the persistent env (shell profile / registry).
    pub env_port: Option<u16>,
    /// Port in `HTTPS_PROXY` as inherited by *this* process — catches "setup ran but
    /// this terminal predates it".
    pub shell_proxy_port: Option<u16>,
    /// CA file path when it exists on disk.
    pub ca_path: Option<String>,
    /// `NODE_EXTRA_CA_CERTS` set in this process.
    pub shell_ca_set: bool,
    pub autostart: bool,
    /// Recorded requests, `None` when the ledger failed to open.
    pub ledger_rows: Option<i64>,
    pub ledger_error: Option<String>,
    /// Humanized age of the last recorded request.
    pub last_request: Option<String>,
    pub log_path: Option<String>,
    /// Newer released version, if the (cached) update check knows one.
    pub update_available: Option<String>,
}

/// The finished diagnosis: checklist rows + how many are real problems (`⚠` rows;
/// `•` notes don't fail the run).
pub struct Report {
    pub rows: Vec<(&'static str, String, String)>,
    pub problems: usize,
}

/// Probe the live system. The only impure step — every probe is read-only and
/// individually best-effort, so doctor itself can never make things worse.
pub fn gather() -> Report {
    let (ledger_rows, ledger_error, last_request) = match crate::tracking::Tracker::open() {
        Ok(t) => match t.summary() {
            Ok(s) => (
                Some(s.events),
                None,
                s.last_ts.as_deref().and_then(crate::daemon::human_age),
            ),
            Err(e) => (None, Some(format!("{e:#}")), None),
        },
        Err(e) => (None, Some(format!("{e:#}")), None),
    };
    let daemon = crate::daemon::running();
    let state = State {
        exe: std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".into()),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        running: daemon.is_some(),
        pid: daemon.as_ref().map(|d| d.pid).unwrap_or(0),
        port: daemon.as_ref().map(|d| d.port).unwrap_or(0),
        uptime: daemon
            .as_ref()
            .map(|d| crate::daemon::human_uptime(crate::daemon::uptime_secs(d.started_at)))
            .unwrap_or_default(),
        daemon_version: daemon.as_ref().and_then(|d| d.version.clone()),
        restarts: daemon.as_ref().map(|d| d.restarts).unwrap_or(0),
        port_accepting: daemon
            .as_ref()
            .map(|d| crate::daemon::probe_port(d.port))
            .unwrap_or(false),
        env_port: crate::setup::configured_port(),
        shell_proxy_port: std::env::var("HTTPS_PROXY")
            .ok()
            .as_deref()
            .and_then(crate::setup::parse_proxy_port),
        ca_path: crate::serve::ca_cert_path()
            .ok()
            .filter(|p| p.exists())
            .map(|p| p.display().to_string()),
        shell_ca_set: std::env::var("NODE_EXTRA_CA_CERTS").is_ok(),
        autostart: crate::autostart::is_enabled(),
        ledger_rows,
        ledger_error,
        last_request,
        log_path: crate::daemon::logfile()
            .ok()
            .map(|p| p.display().to_string()),
        update_available: crate::update::check(false),
    };
    let mut report = build(&state);
    // Gaps are not part of the public `State` shape (semver); append after pure `build`.
    for gap in crate::ensure::probe() {
        report.rows.push((
            ui::WARN,
            gap.label.to_ascii_lowercase(),
            format!("{} — fix: llmtrim ensure", gap.detail),
        ));
        report.problems += 1;
    }
    // Money coverage: only the empty case is a real problem (CLI/MCP-only or attribution never
    // ran). Partial ratio is a weak heuristic (different tables + retention caps), so we do
    // not WARN on partial pure-proxy installs.
    if let Ok(t) = crate::tracking::Tracker::open_reader()
        && let Ok(cov) = llmtrim_ledger::money::money_coverage(t.connection())
        && cov.empty_money_with_traffic()
    {
        report.rows.push((
            ui::WARN,
            "money coverage".into(),
            format!(
                "{} compressions, 0 proxy turns — dollars need proxy attribution \
                 (CLI/MCP-only has no session bill)",
                cov.compressions_events
            ),
        ));
        report.problems += 1;
    }
    report
}

/// Turn probed state into checklist rows. Pure — the testable core of doctor.
pub fn build(s: &State) -> Report {
    let mut rows: Vec<(&'static str, String, String)> = Vec::new();
    let log = s.log_path.as_deref().unwrap_or("~/.llmtrim/serve.log");

    // binary — informational, always present.
    rows.push((
        ui::OK,
        "binary".into(),
        format!("v{} · {}", s.binary_version, s.exe),
    ));

    // daemon + port — a live pidfile is necessary, an accepting port is sufficient.
    if s.running {
        rows.push((
            ui::OK,
            "daemon".into(),
            format!("running · pid {} · :{} · up {}", s.pid, s.port, s.uptime),
        ));
        if s.port_accepting {
            rows.push((ui::OK, "port".into(), format!(":{} accepting", s.port)));
        } else {
            rows.push((
                ui::WARN,
                "port".into(),
                format!(":{} not accepting — check log: {log}", s.port),
            ));
        }
        if s.restarts > 0 {
            rows.push((
                ui::WARN,
                "stability".into(),
                format!(
                    "crashed and restarted {}× since start — check log: {log}",
                    s.restarts
                ),
            ));
        }
    } else {
        rows.push((
            ui::WARN,
            "daemon".into(),
            "not running — start: llmtrim start".into(),
        ));
    }

    // env (persisted) — must exist and agree with the daemon's port.
    match (s.env_port, s.running) {
        (Some(p), true) if p == s.port => {
            rows.push((ui::OK, "env".into(), format!("wired to :{p}")));
        }
        (Some(p), true) => rows.push((
            ui::WARN,
            "env".into(),
            format!(
                "points at :{p} but the daemon listens on :{} — run: llmtrim setup",
                s.port
            ),
        )),
        // POSIX self-gates (the shell block only wires while the daemon is up), so a stopped
        // daemon means new shells route directly — informational, not a failure. Windows env
        // lives in the registry and is only cleared by an explicit `stop`; if the daemon died
        // without one, new terminals still hit the dead proxy and fail — a real problem.
        #[cfg(not(windows))]
        (Some(p), false) => rows.push((
            ui::NOTE,
            "env".into(),
            format!(
                "persisted (:{p}) but the daemon is stopped — new shells route directly \
                 (uncompressed, untracked); run: llmtrim start to resume interception"
            ),
        )),
        #[cfg(windows)]
        (Some(p), false) => rows.push((
            ui::WARN,
            "env".into(),
            format!(
                "wired to :{p} but the daemon is stopped — new terminals fail until you run \
                 llmtrim start (or llmtrim stop to clear the env)"
            ),
        )),
        (None, _) => rows.push((
            ui::WARN,
            "env".into(),
            "not wired — traffic bypasses llmtrim; run: llmtrim setup".into(),
        )),
    }

    // env (this shell) — the env can be persisted yet absent from this terminal, either because
    // the terminal predates setup or because the daemon is stopped (the managed block only wires
    // HTTPS_PROXY while the daemon is up, so a stopped daemon correctly leaves new shells clear).
    match (s.shell_proxy_port, s.env_port) {
        (Some(p), _) => rows.push((ui::OK, "this shell".into(), format!("HTTPS_PROXY=:{p}"))),
        (None, Some(_)) if !s.running => rows.push((
            ui::NOTE,
            "this shell".into(),
            "HTTPS_PROXY not set here — the daemon is stopped, so shells route directly".into(),
        )),
        (None, Some(_)) => rows.push((
            ui::NOTE,
            "this shell".into(),
            "HTTPS_PROXY not set here — this terminal predates setup; open a new one".into(),
        )),
        (None, None) => rows.push((
            ui::NOTE,
            "this shell".into(),
            "HTTPS_PROXY not set (env not wired)".into(),
        )),
    }

    // CA — the file, plus whether this shell trusts it.
    match &s.ca_path {
        Some(path) => {
            let trust = if s.shell_ca_set {
                String::new()
            } else {
                " · NODE_EXTRA_CA_CERTS not set in this shell".to_string()
            };
            rows.push((ui::OK, "ca".into(), format!("{path}{trust}")));
        }
        None => rows.push((ui::WARN, "ca".into(), "missing — run: llmtrim ca".into())),
    }

    // autostart — off is a legitimate choice, so a note, not a problem.
    if s.autostart {
        rows.push((ui::OK, "autostart".into(), "runs at login".into()));
    } else {
        rows.push((
            ui::NOTE,
            "autostart".into(),
            "off — enable: llmtrim autostart".into(),
        ));
    }

    // ledger — recording is the product's proof of value.
    match (s.ledger_rows, &s.ledger_error) {
        (Some(0), _) => rows.push((ui::NOTE, "ledger".into(), "empty — no requests yet".into())),
        (Some(n), _) => {
            let last = s
                .last_request
                .as_deref()
                .map(|a| format!(" · last {a}"))
                .unwrap_or_default();
            rows.push((
                ui::OK,
                "ledger".into(),
                format!("{} requests recorded{last}", ui::commas(n)),
            ));
        }
        (None, err) => rows.push((
            ui::WARN,
            "ledger".into(),
            format!("unreadable — {}", err.as_deref().unwrap_or("unknown error")),
        )),
    }

    // version skew — a daemon older than the binary keeps serving old code after update.
    if s.running
        && let Some(v) = &s.daemon_version
        && *v != s.binary_version
    {
        rows.push((
            ui::WARN,
            "version".into(),
            format!(
                "daemon is v{v}, binary is v{} — restart to update: llmtrim start --force",
                s.binary_version
            ),
        ));
    }
    if let Some(v) = &s.update_available {
        rows.push((
            ui::NOTE,
            "update".into(),
            format!("v{v} available — run: llmtrim update"),
        ));
    }

    let problems = rows.iter().filter(|(g, _, _)| *g == ui::WARN).count();
    Report { rows, problems }
}

/// Render the report as the standard checklist panel + a one-line verdict.
pub fn render(color: bool, report: &Report) -> String {
    let lines = ui::kv_rows(color, &report.rows);
    let mut o = ui::panel(color, "llmtrim doctor", &lines);
    if report.problems == 0 {
        o.push_str(&format!("{}\n", ui::ok(color, "all checks passed")));
    } else {
        o.push_str(&format!(
            "{}\n",
            ui::warn(
                color,
                &format!(
                    "{} problem{} found — fixes above",
                    report.problems,
                    if report.problems == 1 { "" } else { "s" }
                )
            )
        ));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fully healthy install; tests break one probe at a time.
    fn healthy() -> State {
        State {
            exe: "/usr/local/bin/llmtrim".into(),
            binary_version: "0.1.0".into(),
            running: true,
            pid: 4242,
            port: 8788,
            uptime: "33m45s".into(),
            daemon_version: Some("0.1.0".into()),
            restarts: 0,
            port_accepting: true,
            env_port: Some(8788),
            shell_proxy_port: Some(8788),
            ca_path: Some("/home/u/.llmtrim/ca.pem".into()),
            shell_ca_set: true,
            autostart: true,
            ledger_rows: Some(3348),
            ledger_error: None,
            last_request: Some("4s ago".into()),
            log_path: Some("/home/u/.llmtrim/serve.log".into()),
            update_available: None,
        }
    }

    #[test]
    fn healthy_state_has_no_problems() {
        let r = build(&healthy());
        assert_eq!(r.problems, 0, "rows: {:?}", r.rows);
        assert!(r.rows.iter().all(|(g, _, _)| *g != ui::WARN));
        let out = render(false, &r);
        assert!(out.contains("all checks passed"));
        assert!(out.contains("3,348 requests recorded · last 4s ago"));
        assert!(!out.contains('\x1b'), "no ANSI when color=false");
    }

    #[test]
    fn stopped_but_wired_flags_the_daemon_and_explains_direct_routing() {
        let r = build(&State {
            running: false,
            port_accepting: false,
            ..healthy()
        });
        // The stopped daemon is always the actionable problem. The persisted-env row differs by
        // platform: POSIX self-gates (new shells route directly — a note, not a second problem),
        // while Windows env lives in the registry and stays wired at a dead proxy until an
        // explicit `stop`, so it's a real warning.
        assert!(r.problems >= 1, "daemon down is a problem");
        let out = render(false, &r);
        assert!(out.contains("not running — start: llmtrim start"));
        #[cfg(not(windows))]
        {
            assert!(out.contains("route directly"));
            assert!(out.contains("resume interception"));
        }
        #[cfg(windows)]
        assert!(out.contains("new terminals fail"));
    }

    #[test]
    fn env_mismatch_names_both_ports() {
        let r = build(&State {
            env_port: Some(8787),
            ..healthy()
        });
        assert_eq!(r.problems, 1);
        let detail = &r.rows.iter().find(|(_, l, _)| l == "env").unwrap().2;
        assert!(detail.contains(":8787") && detail.contains(":8788"));
    }

    #[test]
    fn stale_terminal_is_a_note_not_a_problem() {
        let r = build(&State {
            shell_proxy_port: None,
            ..healthy()
        });
        assert_eq!(r.problems, 0, "a stale terminal must not fail doctor");
        let out = render(false, &r);
        assert!(out.contains("open a new one"));
    }

    #[test]
    fn dead_port_and_crash_loop_are_problems() {
        let r = build(&State {
            port_accepting: false,
            restarts: 4,
            ..healthy()
        });
        assert_eq!(r.problems, 2);
        let out = render(false, &r);
        assert!(out.contains("not accepting"));
        assert!(out.contains("restarted 4×"));
    }

    #[test]
    fn version_skew_flagged_only_when_known() {
        let r = build(&State {
            daemon_version: Some("0.0.9".into()),
            ..healthy()
        });
        assert_eq!(r.problems, 1);
        // Pre-version pidfile: nothing to compare, no false alarm.
        let r = build(&State {
            daemon_version: None,
            ..healthy()
        });
        assert_eq!(r.problems, 0);
    }

    #[test]
    fn unreadable_ledger_is_a_problem() {
        let r = build(&State {
            ledger_rows: None,
            ledger_error: Some("disk I/O error".into()),
            ..healthy()
        });
        assert_eq!(r.problems, 1);
        let out = render(false, &r);
        assert!(out.contains("disk I/O error"));
    }
}
