//! `llmtrim wrap <agent> [-- <args>]` — thin convenience launcher.
//!
//! This is *sugar*, not a provider system. It does two things:
//!
//!   1. Confirm the interceptor is wired (the same `HTTPS_PROXY` mechanism `setup`
//!      installs and `start` checks), so the agent's HTTPS to LLM hosts routes through
//!      llmtrim — there is **no** per-agent quirk handling, no base-URL writing, no
//!      allow-list of "supported" agents. Any binary on PATH works.
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
const KNOWN_AGENTS: &[&str] = &["claude", "codex", "cursor", "aider", "copilot", "gemini"];

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
fn https_proxy_is_local() -> bool {
    std::env::var("HTTPS_PROXY")
        .or_else(|_| std::env::var("https_proxy"))
        .map(|v| v.contains("127.0.0.1") || v.contains("localhost"))
        .unwrap_or(false)
}

pub fn run(raw: Vec<String>) -> Result<()> {
    let inv = parse_invocation(&raw)?;
    let color = ui::color_stdout();

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
    // HTTPS_PROXY + the CA trust vars, which is the entire interception mechanism. We add
    // nothing agent-specific.
    eprintln!(
        "{}",
        ui::paint(color, Tone::Dim, &format!("llmtrim wrap → {}", inv.agent))
    );

    exec_agent(&inv)
}

/// Launch the agent and propagate its exit code. This is the real-IO entrypoint (it spawns
/// a subprocess), so it is left uncovered by unit tests — the testable logic lives in
/// `parse_invocation` / `readiness`, which are tested below.
fn exec_agent(inv: &WrapInvocation) -> Result<()> {
    let status = std::process::Command::new(&inv.agent)
        .args(&inv.args)
        .status()
        .with_context(|| {
            if KNOWN_AGENTS.contains(&inv.agent.as_str()) {
                format!(
                    "failed to launch `{}`: is it installed and on your PATH?",
                    inv.agent
                )
            } else {
                format!(
                    "failed to launch `{}`: not found on PATH (pass an installed binary, \
                     e.g. one of: {})",
                    inv.agent,
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
}
