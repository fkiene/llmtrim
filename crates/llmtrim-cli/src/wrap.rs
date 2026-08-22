//! `llmtrim wrap <agent> [-- <args>]` — thin convenience launcher.
//!
//! This is *sugar*, not a provider system. It does two things:
//!
//!   1. Confirm the interceptor is wired (the same `HTTPS_PROXY` mechanism `setup`
//!      installs and `start` checks), so the agent's HTTPS to LLM hosts routes through
//!      llmtrim — there is **no** per-agent quirk handling, no base-URL writing, no
//!      allow-list of "supported" agents. Any binary on PATH works. One exception:
//!      on Windows, the DeepSeek Harness CLI (`dsh`) is an npm `.cmd`/`.ps1` shim that
//!      `Command` cannot exec directly, so `wrap` resolves it to
//!      `node <shimDir>\node_modules\@deepseek-ai\dsh\lib\bin.js` (see `resolve_launch`).
//!   2. Exec the named binary as a subprocess that inherits the current environment
//!      (which, post-`setup` + a fresh shell, already carries `HTTPS_PROXY` and the CA
//!      trust vars), forwarding the passthrough args and propagating its exit code.
//!
//! Setup-check behaviour (deliberate, least-surprising): if the env isn't wired we do
//! **not** silently mutate the user's shell profile or env — that's `setup`'s job and
//! doing it from a launcher would be a surprising side effect. We print a clear pointer
//! to `llmtrim setup` and refuse, so the user never gets a wrapped agent that quietly
//! bypasses compression. We *do* start a stopped daemon only when the env is already
//! wired (trivially safe: the contract — port + CA — is already in place, same as `start`).

use anyhow::{Context, Result};

use crate::ui::{self, Tone};

/// A parsed `wrap` invocation: the agent binary to launch and the args to forward to it.
#[derive(Debug, PartialEq, Eq)]
pub struct WrapInvocation {
    /// The agent binary name (or path) to run — free-form, resolved on PATH at launch.
    pub agent: String,
    /// Arguments forwarded verbatim to the agent (everything after `<agent>`/`--`).
    pub args: Vec<String>,
}

/// A few well-known agent names, used *only* to enrich the "not found" hint. This is NOT
/// an allow-list: any binary on PATH is accepted. Kept tiny and advisory on purpose.
const KNOWN_AGENTS: &[&str] = &[
    "claude", "codex", "cursor", "aider", "copilot", "gemini", "dsh",
];

/// Split the raw `wrap` arguments into the agent and its passthrough args. The first token
/// is the agent; everything after it is forwarded as-is. A leading `--` separator (clap
/// convention) is dropped if present. Pure, so it's unit-tested without launching anything.
fn parse_invocation(raw: &[String]) -> Result<WrapInvocation> {
    let mut it = raw.iter();
    let agent = it
        .next()
        .context("`wrap` needs an agent to run, e.g. `llmtrim wrap claude`")?
        .clone();
    let mut args: Vec<String> = it.cloned().collect();
    // Drop a single leading `--` (the conventional end-of-options marker) so
    // `llmtrim wrap claude -- --foo` forwards `--foo`, not `-- --foo`.
    if args.first().map(String::as_str) == Some("--") {
        args.remove(0);
    }
    Ok(WrapInvocation { agent, args })
}

/// Is the interceptor usable for a freshly-launched child? It needs both halves of the
/// contract: a live daemon (so requests have somewhere to go) and the env wired (so the
/// child inherits `HTTPS_PROXY` + CA trust). Returns which half, if any, is missing.
#[derive(Debug, PartialEq, Eq)]
enum Readiness {
    Ready,
    /// Env wired but no daemon listening — trivially fixable by starting it.
    DaemonDown,
    /// Env not wired — needs `setup` (we won't mutate the profile from a launcher).
    EnvUnwired,
}

/// Decide readiness from the two facts `start`/`setup` already expose. Pure seam so the
/// precedence is unit-testable without touching the real daemon or shell profile.
fn readiness(daemon_running: bool, env_wired: bool) -> Readiness {
    match (env_wired, daemon_running) {
        (true, true) => Readiness::Ready,
        (true, false) => Readiness::DaemonDown,
        (false, _) => Readiness::EnvUnwired,
    }
}

/// Does *this* process actually carry an `HTTPS_PROXY` pointing at the local interceptor?
/// This is what matters: the child inherits our live environment, not the shell profile on
/// disk. Checking `profile_has_block()` would pass when `setup` has run but the current
/// shell predates it, launching the agent with no proxy and silently skipping compression.
pub fn https_proxy_is_local() -> bool {
    std::env::var("HTTPS_PROXY")
        .or_else(|_| std::env::var("https_proxy"))
        .map(|v| v.contains("127.0.0.1") || v.contains("localhost"))
        .unwrap_or(false)
}

pub fn run(raw: Vec<String>) -> Result<()> {
    let inv = parse_invocation(&raw)?;
    let color = ui::color_stdout();

    // Resolve the real launch command up front (Windows `dsh` → node + bin.js), so
    // readiness hints and the "not found" error talk about the agent the user typed.
    let (program, final_args) = resolve_launch(&inv.agent, &inv.args, None)?;

    // Reuse the exact helpers `start`/`setup` use — do not reimplement the checks.
    let daemon_running = crate::daemon::running().is_some();
    let env_wired = https_proxy_is_local();

    match readiness(daemon_running, env_wired) {
        Readiness::Ready => {}
        Readiness::DaemonDown => {
            // Env already wired, so the port + CA contract is in place: starting the
            // daemon here is trivially safe and consistent with `llmtrim start`.
            let port = crate::setup::resolve_port(None, None)?;
            let pid = crate::daemon::spawn_detached(port)
                .context("interceptor is down and could not be started")?;
            eprintln!(
                "{}",
                ui::note(
                    ui::color_stderr(),
                    &format!("Started the interceptor (pid {pid} · port {port}).")
                )
            );
        }
        Readiness::EnvUnwired => {
            // Don't silently edit the user's environment from a launcher — point at setup.
            // If setup already ran, the profile has the block but this shell predates it, so
            // tailor the hint instead of telling the user to re-run setup pointlessly.
            let hint = if crate::setup::profile_has_block() {
                "You've run `llmtrim setup`, but this shell started before it. Open a new \
                 shell (or re-source your profile) and try again."
            } else {
                "Run `llmtrim setup` once (then open a new shell), and try again."
            };
            anyhow::bail!(
                "HTTPS_PROXY isn't pointing at llmtrim in this shell, so `{}` wouldn't route \
                 through it.\n{hint}",
                inv.agent
            );
        }
    }

    // The child inherits our environment as-is: post-setup that already contains
    // HTTPS_PROXY + the CA trust vars, which is the entire interception mechanism.
    // When global sub is always-on, also inject a dummy Anthropic auth token so Claude
    // Code skips OAuth (same idea as claude-code-proxy's ANTHROPIC_AUTH_TOKEN=unused).
    eprintln!(
        "{}",
        ui::paint(color, Tone::Dim, &format!("llmtrim wrap → {}", inv.agent))
    );

    exec_agent(&program, &final_args, &inv.agent)
}

/// True when the agent binary looks like Claude Code (not Codex/Gemini/etc.).
fn agent_is_claude(agent: &str) -> bool {
    let base = agent
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(agent)
        .to_ascii_lowercase();
    base == "claude" || base.starts_with("claude-") || base == "claude.exe"
}

/// True when the agent binary is the DeepSeek Harness CLI. On Windows `dsh` is an
/// npm `.cmd`/`.ps1` shim that `Command` cannot exec directly (see `resolve_launch`).
fn agent_is_dsh(agent: &str) -> bool {
    let base = agent
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(agent)
        .to_ascii_lowercase();
    let base = base.as_str();
    matches!(base, "dsh" | "dsh.exe" | "dsh.cmd" | "dsh.ps1")
}

/// First directory on `PATH` that contains a shim named `bin` (bare, `.exe`, `.cmd`,
/// or `.ps1`). `path_env` is a seam for tests; `None` reads the live environment.
fn shim_dir_on_path(bin: &str, path_env: Option<&str>) -> Option<std::path::PathBuf> {
    let path = match path_env {
        Some(p) => p.to_string(),
        None => std::env::var_os("PATH")?.to_string_lossy().into_owned(),
    };
    let names = [
        bin.to_string(),
        format!("{bin}.exe"),
        format!("{bin}.cmd"),
        format!("{bin}.ps1"),
    ];
    std::env::split_paths(&path).find(|dir| names.iter().any(|n| dir.join(n).is_file()))
}

/// Derive the npm-installed entry script next to a `dsh` shim, if present.
fn dsh_node_entry(path_env: Option<&str>) -> Option<std::path::PathBuf> {
    let dir = shim_dir_on_path("dsh", path_env)?;
    let entry = dir
        .join("node_modules")
        .join("@deepseek-ai")
        .join("dsh")
        .join("lib")
        .join("bin.js");
    entry.is_file().then_some(entry)
}

/// Rewrite a `wrap` invocation into the real (program, args) to launch.
///
/// Only Windows `dsh` is special: npm ships `dsh` as `.cmd`/`.ps1` shims that Rust's
/// `Command` cannot exec directly, so we resolve the shim's directory on PATH and
/// launch `node <shimDir>\node_modules\@deepseek-ai\dsh\lib\bin.js`. Every other
/// agent resolves to itself (today's generic behavior). `path_env` is a seam for
/// tests; `None` reads the live environment.
fn resolve_launch(
    agent: &str,
    args: &[String],
    path_env: Option<&str>,
) -> Result<(String, Vec<String>)> {
    if cfg!(windows) && agent_is_dsh(agent) {
        if let Some(entry) = dsh_node_entry(path_env) {
            let mut final_args = Vec::with_capacity(args.len() + 1);
            final_args.push(entry.to_string_lossy().into_owned());
            final_args.extend(args.iter().cloned());
            return Ok(("node".to_string(), final_args));
        }
        // dsh shim present but entry missing → actionable hint.
        if shim_dir_on_path("dsh", path_env).is_some() {
            anyhow::bail!(
                "found the `dsh` shim but no `node_modules\\@deepseek-ai\\dsh\\lib\\bin.js` \
                 beside it — run `npm install -g @deepseek-ai/dsh` and try again"
            );
        }
    }
    Ok((agent.to_string(), args.to_vec()))
}

/// Launch the resolved program and propagate its exit code. This is the real-IO
/// entrypoint (it spawns a subprocess), so it is left uncovered by unit tests — the
/// testable logic lives in `parse_invocation` / `readiness` / `resolve_launch`.
fn exec_agent(program: &str, args: &[String], display: &str) -> Result<()> {
    let mut cmd = std::process::Command::new(program);
    cmd.args(args);
    // Always-sub skip-login: Claude Code must not require a live Anthropic OAuth session.
    // Only inject for Claude-ish binaries — never pollute codex/gemini/etc.
    // Prefer an already-set user value; only inject when missing so a real key still wins.
    if llmtrim_core::config::sub_skip_anthropic_login()
        && agent_is_claude(display)
        && std::env::var_os("ANTHROPIC_AUTH_TOKEN").is_none()
    {
        cmd.env(
            "ANTHROPIC_AUTH_TOKEN",
            crate::statusline::SUB_AUTH_TOKEN_VALUE,
        );
    }
    let status = cmd.status().with_context(|| {
        if KNOWN_AGENTS.contains(&display) {
            format!("failed to launch `{display}`: is it installed and on your PATH?")
        } else {
            format!(
                "failed to launch `{display}`: not found on PATH (pass an installed binary, \
                 e.g. one of: {})",
                KNOWN_AGENTS.join(", ")
            )
        }
    })?;

    // Per the repo's exit-code rule: mirror the child's status so CI/scripts see the truth.
    std::process::exit(status.code().unwrap_or(1));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_agent_with_no_args() {
        let inv = parse_invocation(&s(&["claude"])).expect("agent only");
        assert_eq!(inv.agent, "claude");
        assert!(inv.args.is_empty());
    }

    #[test]
    fn forwards_trailing_args_verbatim() {
        let inv = parse_invocation(&s(&["claude", "chat", "--model", "x"])).expect("with args");
        assert_eq!(inv.agent, "claude");
        assert_eq!(inv.args, s(&["chat", "--model", "x"]));
    }

    #[test]
    fn drops_single_leading_double_dash() {
        let inv = parse_invocation(&s(&["aider", "--", "--foo", "bar"])).expect("dash sep");
        assert_eq!(inv.agent, "aider");
        assert_eq!(inv.args, s(&["--foo", "bar"]));
    }

    #[test]
    fn only_first_double_dash_is_dropped() {
        let inv = parse_invocation(&s(&["x", "--", "--", "y"])).expect("two dashes");
        assert_eq!(inv.args, s(&["--", "y"]));
    }

    #[test]
    fn empty_invocation_is_an_error() {
        assert!(parse_invocation(&[]).is_err());
    }

    #[test]
    fn accepts_any_binary_name_not_just_known_ones() {
        let inv = parse_invocation(&s(&["some-random-tool"])).expect("free-form");
        assert_eq!(inv.agent, "some-random-tool");
    }

    #[test]
    fn readiness_ready_when_both_present() {
        assert_eq!(readiness(true, true), Readiness::Ready);
    }

    #[test]
    fn readiness_daemon_down_when_env_wired_only() {
        assert_eq!(readiness(false, true), Readiness::DaemonDown);
    }

    #[test]
    fn readiness_env_unwired_takes_precedence() {
        assert_eq!(readiness(false, false), Readiness::EnvUnwired);
        assert_eq!(readiness(true, false), Readiness::EnvUnwired);
    }

    #[test]
    fn agent_is_claude_matches_claude_binaries_only() {
        assert!(agent_is_claude("claude"));
        assert!(agent_is_claude("/usr/bin/claude"));
        assert!(agent_is_claude("claude-2"));
        assert!(agent_is_claude(r"C:\Tools\claude.exe"));
        assert!(!agent_is_claude("codex"));
        assert!(!agent_is_claude("gemini"));
        assert!(!agent_is_claude("/usr/bin/aider"));
    }

    #[test]
    fn agent_is_dsh_matches_dsh_binaries_only() {
        assert!(agent_is_dsh("dsh"));
        assert!(agent_is_dsh("dsh.exe"));
        assert!(agent_is_dsh("dsh.cmd"));
        assert!(agent_is_dsh("dsh.ps1"));
        assert!(agent_is_dsh(r"C:\Tools\dsh.exe"));
        assert!(agent_is_dsh("/usr/bin/dsh"));
        assert!(!agent_is_dsh("claude"));
        assert!(!agent_is_dsh("codex"));
        assert!(!agent_is_dsh("dsh-custom"));
    }

    #[test]
    fn non_dsh_agents_resolve_to_themselves() {
        let out = resolve_launch("claude", &s(&["--x"]), Some("C:\\bin")).expect("generic");
        assert_eq!(out, ("claude".to_string(), s(&["--x"])));
        let out = resolve_launch("codex", &[], Some("C:\\bin")).expect("generic");
        assert_eq!(out, ("codex".to_string(), Vec::<String>::new()));
    }

    #[cfg(windows)]
    #[test]
    fn dsh_resolves_to_node_entry_when_shim_and_entry_present() {
        let dir = std::env::temp_dir().join(format!("llmtrim-wrap-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(
            dir.join("node_modules")
                .join("@deepseek-ai")
                .join("dsh")
                .join("lib"),
        )
        .expect("mkdir");
        std::fs::write(dir.join("dsh.cmd"), "@echo off\r\nexit /b 0\r\n").expect("shim");
        std::fs::write(
            dir.join("node_modules")
                .join("@deepseek-ai")
                .join("dsh")
                .join("lib")
                .join("bin.js"),
            "#!/usr/bin/env node\r\n",
        )
        .expect("entry");

        let path = dir.to_string_lossy().into_owned();
        let out = resolve_launch("dsh", &s(&["web"]), Some(path.as_str())).expect("resolve");
        assert_eq!(out.0, "node");
        assert_eq!(out.1.len(), 2);
        assert!(
            out.1[0].ends_with("node_modules\\@deepseek-ai\\dsh\\lib\\bin.js")
                || out.1[0].ends_with("node_modules/@deepseek-ai/dsh/lib/bin.js")
        );
        assert_eq!(out.1[1], "web");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn dsh_with_shim_but_no_entry_returns_npm_install_hint() {
        let dir = std::env::temp_dir().join(format!("llmtrim-wrap-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("dsh.cmd"), "@echo off\r\nexit /b 0\r\n").expect("shim");

        let path = dir.to_string_lossy().into_owned();
        let err = resolve_launch("dsh", &[], Some(path.as_str())).expect_err("should error");
        assert!(err.to_string().contains("npm install -g @deepseek-ai/dsh"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dsh_not_on_path_resolves_to_itself() {
        // No dsh shim anywhere on PATH → generic behavior (the later exec error
        // handles the not-found case, matching today's semantics for any agent).
        let out = resolve_launch("dsh", &[], Some("C:\\other")).expect("generic");
        assert_eq!(out, ("dsh".to_string(), Vec::<String>::new()));
    }
}
