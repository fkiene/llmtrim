//! llmtrim CLI.
//!
//! Network-free surface: `compress` reads a provider request body on stdin and writes the
//! compressed body to stdout; `send` adds the network round-trip; `status` shows
//! savings from the SQLite ledger; `serve`/`setup` run the MITM interceptor. The pure
//! transform core lives in `lib.rs`.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use llmtrim::bench::{self, BenchCase};
use llmtrim::monitor;
use llmtrim::tracking::{Period, Record, Tracker};
use llmtrim::transport::Endpoint;
use llmtrim::ui::{self, Tone};
use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;

/// Coloured `--help` in the dashboard's accent family. clap gates these on
/// TTY/NO_COLOR itself via anstream, so piped help stays plain.
const HELP_STYLES: clap::builder::Styles = clap::builder::Styles::styled()
    .header(
        clap::builder::styling::AnsiColor::BrightBlue
            .on_default()
            .bold(),
    )
    .usage(
        clap::builder::styling::AnsiColor::BrightBlue
            .on_default()
            .bold(),
    )
    .literal(
        clap::builder::styling::AnsiColor::BrightCyan
            .on_default()
            .bold(),
    )
    .placeholder(clap::builder::styling::AnsiColor::Cyan.on_default());

/// Hand-written top-level help: clap can't group subcommands into sections, so the
/// command list is maintained here. Keep it in sync with the `Commands` enum.
const HELP_TEMPLATE: &str = "\
{about-with-newline}
{usage-heading} {usage}

Get started:
  setup      Set everything up and start saving (CA, env, autostart, integrations, daemon)
  status     Show the savings dashboard + interceptor health  [aliases: monitor, gain]
  update     Update to the latest release and refresh integrations
  ensure     Bring this machine to the recommended current state
  wrap       Launch an agent (claude, codex, …) routed through the interceptor
  sub        Reroute Claude Code to another subscription's backend (codex|kimi|grok)
  tray       Open the desktop tray app (savings menu-bar / system-tray)

Daemon:
  start      Start the background interceptor (no-op if already running)
  stop       Stop the background interceptor
  autostart  Run the interceptor at login (--off to disable)

When something's wrong:
  doctor     Check the install end-to-end; `--fix` applies repairs
  uninstall  Undo everything `setup` did

Pipes & one-shots:
  compress   Compress a request from stdin to stdout
  send       Compress a request, send it to the provider, print the response
  serve      Run the HTTPS interceptor in the foreground
  ca         Print the local CA certificate path and how to trust it

Measurement (dev):
  eval       Measure retrieval recall + token savings on a corpus
  bench      A/B benchmark: tokens saved vs quality retained, on a real model
  discover   Scan the capture corpus for where tokens still escape compression

Options:
{options}
Run `llmtrim help <command>` for details on any command.
";

#[derive(Parser)]
#[command(
    name = "llmtrim",
    version,
    about = "Static, deterministic LLM prompt/payload compressor",
    styles = HELP_STYLES,
    help_template = HELP_TEMPLATE
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a request from stdin to stdout
    ///
    /// Reads a provider-shaped request body on stdin and writes the compressed
    /// JSON to stdout — the pipe-friendly core. Savings are recorded to the ledger.
    Compress {
        /// Target provider: openai|anthropic|google|gemini. Omit to auto-detect from the request shape.
        #[arg(long)]
        provider: Option<String>,
    },
    /// Compress a request, send it to the provider, print the response
    ///
    /// Like `compress`, plus the network round-trip. Needs the provider API key in
    /// the environment (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY).
    Send {
        /// Target provider: openai|anthropic|google|gemini. Omit to auto-detect from the request shape.
        #[arg(long)]
        provider: Option<String>,
    },
    /// Subscription reroute: use another subscription's backend for Claude Code
    ///
    /// With `sub` enabled, intercepted Anthropic traffic is translated and sent to your
    /// ChatGPT (codex) or Kimi subscription instead of Anthropic. Authenticate first with
    /// `llmtrim sub auth codex login` / `llmtrim sub auth kimi login` / `llmtrim sub auth grok login`.
    Sub {
        #[command(subcommand)]
        action: SubCmd,
    },
    /// Configure cheaper models for Claude Code `/compact`
    ///
    /// Configured alternatives are tried in order when they fit the request. The model Claude Code
    /// originally requested is always the implicit final fallback. Prefer `llmtrim ensure` /
    /// `setup` for the recommended defaults — this is the power-user escape hatch.
    #[command(hide = true)]
    Compact {
        #[command(subcommand)]
        action: CompactCmd,
    },
    /// Run the HTTPS interceptor in the foreground
    ///
    /// A local MITM proxy covering every tool and provider: set HTTPS_PROXY to it
    /// and trust the CA (`llmtrim ca`). No API key needed — the client's own auth
    /// passes through untouched. To run it in the background instead, use `start`.
    Serve {
        /// Port to listen on (127.0.0.1).
        #[arg(long, default_value_t = llmtrim::setup::DEFAULT_PORT)]
        port: u16,
        /// Replace an llmtrim daemon already holding the port (stops it first) instead of
        /// refusing to start.
        #[arg(long)]
        force: bool,
        /// Internal: run with crash-restart supervision (used by `start`/autostart).
        #[arg(long, hide = true)]
        supervised: bool,
        /// Internal: hide the console window at startup (used by the Windows autostart
        /// Run-key entry, which Explorer would otherwise launch with a visible console).
        #[arg(long, hide = true)]
        hide_console: bool,
    },
    /// Set everything up and start saving (CA, environment, autostart, daemon)
    ///
    /// The fastest path from install to compressing: ensures the local CA, sets
    /// HTTPS_PROXY + CA trust in your environment (shell profile on POSIX,
    /// HKCU\Environment on Windows), enables run-at-login, wires Claude Code
    /// integrations (statusline, guard, /sub, compact), and starts the interceptor.
    /// Idempotent — re-running reuses the same port and won't restart a healthy daemon.
    /// No sudo.
    Setup {
        /// Interceptor port. Omit to auto-select a free port starting at 43117.
        #[arg(long)]
        port: Option<u16>,
        /// Restart the daemon even if a healthy one is already on the chosen port (e.g. to pick
        /// up a new binary). By default setup leaves a healthy same-port daemon running.
        #[arg(long)]
        force: bool,
    },
    /// Bring this machine to the recommended current state
    ///
    /// Idempotent: rewrites owned Claude Code hooks/statusline when stale, installs
    /// recommended integrations you have not opted out of, and restarts a version-skewed
    /// daemon. Same engine `setup` and `update` use — the one verb after a release.
    Ensure {
        /// Suppress the summary panel (for scripts / postinstall hooks).
        #[arg(long, short = 'q')]
        quiet: bool,
    },
    /// Undo everything `setup` did
    ///
    /// Stops the daemon, disables autostart, removes the interceptor env, and
    /// removes the CA + state (and the binary). Every step is printed.
    Uninstall {
        /// Also delete the savings ledger (kept by default).
        #[arg(long)]
        purge: bool,
        /// Leave the binary in place (default removes it on Unix).
        #[arg(long)]
        keep_binary: bool,
    },
    /// Start the background interceptor daemon (no-op if already running)
    ///
    /// The lightweight partner to `stop`: starts the daemon without re-running the full
    /// `setup`. Reuses the port already wired into your environment (or `--port`), so it
    /// matches what your tools point at. First time? Run `setup` instead — it wires the
    /// environment too. To run in the foreground, use `serve`.
    Start {
        /// Port to listen on. Omit to reuse the configured port (or 43117).
        #[arg(long)]
        port: Option<u16>,
        /// Restart a running daemon: by default `start` is a no-op when one is already up;
        /// `--force` stops it first and starts a fresh one.
        #[arg(long)]
        force: bool,
    },
    /// Run an agent, guaranteeing its traffic routes through llmtrim
    ///
    /// After `setup`, tools route through llmtrim automatically, so you rarely need this. Use
    /// it to be certain a session is compressed: `wrap` refuses to launch when HTTPS_PROXY
    /// isn't pointing at llmtrim in this shell (a shell opened before `setup`, say), instead of
    /// letting the agent run uncompressed without you noticing. When the environment is wired
    /// but the daemon is down, it starts the daemon for you. `<agent>` is any binary on PATH
    /// (claude, codex, cursor, aider, …); args after it (or after `--`) are forwarded verbatim,
    /// and the agent's exit code is propagated.
    Wrap {
        /// The agent binary to launch, followed by its arguments (use `--` before flags).
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Stop the background interceptor daemon
    Stop,
    /// Fast liveness probe for the shell-profile block (hidden): exit 0 if the daemon is up,
    /// 1 if not. Reads the pidfile + `kill -0` only (no network), so it's cheap enough to run
    /// on every new shell — the managed env block calls it to wire HTTPS_PROXY only while the
    /// interceptor is actually running.
    #[command(name = "_alive", hide = true)]
    Alive,
    /// Update llmtrim to the latest release and refresh integrations
    ///
    /// Channel-aware: a binary install self-updates via the installer, restarts the
    /// daemon, and runs `ensure`. Cargo / Homebrew / npm print their package command;
    /// run `llmtrim ensure` after that finishes (or open status and press f).
    Update,
    /// Show the savings dashboard + interceptor health
    ///
    /// Savings from the ledger, plus the health chain (daemon → port → env → CA →
    /// traffic). Default: a snapshot, exiting 0 healthy / 1 stopped / 2 degraded;
    /// `-q` for the health word only; on a TTY opens the live cost-breakdown TUI;
    /// `--daily/--weekly/--monthly` for time-series; `--json/--csv` to export.
    #[command(name = "status", visible_aliases = ["monitor", "gain"])]
    Monitor {
        /// Deprecated no-op kept for backward compatibility: `status` opens the live
        /// dashboard on a TTY by default, so `--watch` is no longer needed. Hidden from
        /// help; still accepted so existing scripts and aliases don't break.
        #[arg(long, hide = true)]
        watch: bool,
        /// Refresh interval for the live dashboard, in seconds.
        #[arg(long, default_value_t = 2)]
        interval: u64,
        /// Daily time-series report.
        #[arg(long)]
        daily: bool,
        /// Weekly time-series report.
        #[arg(long)]
        weekly: bool,
        /// Monthly time-series report.
        #[arg(long)]
        monthly: bool,
        /// Emit JSON instead of the dashboard (snapshot + time-series).
        #[arg(long)]
        json: bool,
        /// Include the heavy per-source/per-session `breakdown` object in `--json`.
        /// Off by default: it aggregates the entire ledger history (a full
        /// `breakdown_blocks` scan that is expensive on a large DB), and the common
        /// consumers — the health badge and savings percentages — never read it.
        #[arg(long, requires = "json")]
        breakdown: bool,
        /// Emit CSV time-series instead of the dashboard.
        #[arg(long)]
        csv: bool,
        /// Health only: print healthy|degraded|stopped and exit 0/2/1 (script-friendly).
        #[arg(long, short, conflicts_with_all = ["daily", "weekly", "monthly", "json", "csv"])]
        quiet: bool,
    },
    /// Run an MCP server over stdio (or `mcp install` to register it with a client)
    ///
    /// Exposes llmtrim's compression and savings stats as Model Context Protocol
    /// tools, so any MCP client (Claude Code, Cursor, custom agents) can compress a
    /// request and read the ledger directly. Speaks JSON-RPC over stdin/stdout — point
    /// a client at `command: llmtrim, args: ["mcp"]`. Honors your ~/.llmtrim config like
    /// the proxy and CLI do. Run `llmtrim mcp install` to register it for you.
    Mcp {
        #[command(subcommand)]
        action: Option<McpAction>,
    },
    /// Render Claude Code's custom status line (hook entrypoint; prefer `llmtrim ensure`)
    ///
    /// With no subcommand, reads Claude Code's JSON session blob on stdin and prints one
    /// elegant line. `install` / `uninstall` are power-user escape hatches — `setup`,
    /// `update`, and `ensure` keep the status line current automatically.
    #[command(hide = true)]
    Statusline {
        #[command(subcommand)]
        action: Option<StatuslineCmd>,
    },
    /// Cold-cache resume warning hook (prefer `llmtrim ensure`)
    ///
    /// With no subcommand, acts as Claude Code's `UserPromptSubmit` hook. `install` /
    /// `uninstall` are escape hatches — `setup` / `update` / `ensure` wire this for you.
    #[command(hide = true)]
    Guard {
        #[command(subcommand)]
        action: Option<GuardCmd>,
    },
    /// Check the install end-to-end and explain anything broken
    ///
    /// Diagnosis: binary, daemon, port, env wiring (persisted + this shell), CA, autostart,
    /// ledger, version skew, Claude Code integrations. Exits non-zero if any check fails.
    /// Pass `--fix` to apply repairs (same as `llmtrim ensure`).
    Doctor {
        /// Apply recommended fixes instead of only naming them.
        #[arg(long)]
        fix: bool,
    },
    /// Run the interceptor at login (`--off` to disable)
    ///
    /// systemd (Linux) / launchd (macOS) / registry run-key (Windows).
    Autostart {
        /// Disable autostart instead of enabling it.
        #[arg(long)]
        off: bool,
        /// Port the autostarted interceptor listens on. Omit to reuse the port already
        /// wired into your environment / the running daemon (only a first install with
        /// nothing pinned falls back to the default port).
        #[arg(long)]
        port: Option<u16>,
        /// Configure the desktop tray app's autostart instead of the daemon's.
        /// `--port` is ignored with `--tray`.
        #[arg(long)]
        tray: bool,
        /// Print whether autostart is currently enabled (`enabled`/`disabled`)
        /// and exit without changing anything. Combine with `--tray` for the
        /// tray entry. Scriptable: used by the tray's settings panel.
        #[arg(long)]
        status: bool,
    },
    /// Open the desktop tray app (compression savings menu-bar / system-tray)
    ///
    /// Launches the bundled `llmtrim-tray` GUI. Installed by the desktop bundles
    /// (npm / Homebrew); not built by a plain headless `cargo install`.
    Tray,
    /// Print the local CA certificate path and how to trust it
    ///
    /// Generates the CA on first run. Required once before `serve` can intercept
    /// HTTPS; the CA is name-constrained to LLM API domains only.
    Ca {
        /// Print the certificate PEM to stdout (for piping out of containers:
        /// `docker run --rm -v llmtrim-state:/data ghcr.io/fkiene/llmtrim ca --pem > ca.pem`)
        #[arg(long)]
        pem: bool,
    },
    /// Measure retrieval recall + token savings on a corpus
    ///
    /// Runs the lexical-retrieval stage over a held-out corpus JSONL and reports
    /// per-case recall against the gold answers.
    Eval {
        /// Corpus JSONL: lines with {context|input, question|query, answers|answer}.
        #[arg(long)]
        corpus: PathBuf,
        /// Provider for the cases: openai|anthropic|google|gemini.
        #[arg(long, default_value = "openai")]
        provider: String,
        /// Retrieval keep_ratio to evaluate (clamped to [0,1]).
        #[arg(long, default_value_t = 0.5)]
        keep_ratio: f64,
    },
    /// Scan the capture corpus for where compressible tokens still escape compression
    ///
    /// Read-only over the before/after captures (written when `LLMTRIM_CAPTURE_DIR` is set).
    /// Re-buckets each request's token surface by block kind (system/user/assistant/
    /// tool_result/tool_call_args/document/tool_schema) and, with `--by-tool`, by the tool
    /// behind each tool_result. Each row shows the residual still in the compressed request,
    /// how much of it is in the LIVE (uncached) zone that compression can still reach, and how
    /// much compression already removed (before→after) — so the next target is picked from
    /// real traffic. `--json` for the machine-readable report.
    Discover {
        /// Corpus directory. Omit to use `$LLMTRIM_CAPTURE_DIR` or `~/.llmtrim/capture`.
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Split tool_result into per-tool rows (default collapses to one tool_result row).
        #[arg(long)]
        by_tool: bool,
        /// Scan at most N captures (oldest first). Omit to scan the whole corpus.
        #[arg(long)]
        limit: Option<usize>,
        /// Emit the report as JSON instead of the human table.
        #[arg(long)]
        json: bool,
    },
    /// Benchmark suite: quality A/B, agent economics, latency, head-to-head comparisons
    ///
    /// One dispatcher over every measurement axis. `quality` is the single-corpus A/B;
    /// `suite` runs the full corpus matrix in-process; `agent` measures per-iteration token
    /// economics; `latency` times the warm compress path; `compare` drives the Python
    /// head-to-head comparators (Headroom, caveman).
    #[command(subcommand)]
    Bench(BenchCmd),
    /// Internal Claude Code window-local subscription integration.
    #[command(hide = true)]
    WindowSub {
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
}

/// Benchmark suite — a single dispatcher over every measurement axis.
///
/// `quality` is the single-corpus A/B primitive; `suite` runs the full corpus matrix
/// in-process; `agent` measures per-iteration token economics; `latency` times the warm
/// compress path; `compare` drives the head-to-head Python comparators (Headroom, caveman).
#[derive(clap::Subcommand)]
enum BenchCmd {
    /// A/B on one corpus: tokens saved vs quality retained, on a real model.
    ///
    /// Sends ORIGINAL and COMPRESSED requests, scores both, and prices the
    /// round-trip. Credentials come from the env or a local `.env` (OpenRouter).
    /// `--offline`/`--ablate` measure tokens without any network calls.
    Quality(Box<BenchArgs>),
    /// Full corpus matrix in one process — replaces the old run_all.sh.
    ///
    /// Runs every (corpus, preset) pair from the built-in table back to back, reusing the
    /// loaded tokenizer/regexes, and writes one enveloped JSON per corpus into `--out`.
    Suite(Box<SuiteArgs>),
    /// Agent-loop benchmark (issue #14): per-iteration token economics over a golden task set
    ///
    /// Drives a tool-calling loop with deterministic tool stubs and records input/cached/output
    /// tokens + tool-call count per iteration, per condition (baseline vs presets). Default is a
    /// synthetic `--dry-run` (zero API cost); `--live` calls the model (needs `--features live`).
    Agent(Box<BenchAgentArgs>),
    /// Warm compress-path latency + per-stage attribution (offline, no network).
    Latency(LatencyArgs),
    /// Head-to-head vs a third-party compressor (Python comparators).
    Compare(CompareArgs),
}

/// Subscription reroute: send Claude Code traffic to another subscription's backend.
#[derive(Subcommand)]
enum SubCmd {
    /// Open the interactive tier→model mapping editor for a provider (codex|kimi|grok).
    Setup {
        /// Provider to edit: codex|kimi|grok.
        provider: String,
    },
    /// Enable reroute to a provider (writes `sub = <provider>` to the config).
    ///
    /// Omit the provider to re-enable the last provider you used. `sub use` and `sub start`
    /// are accepted aliases.
    #[command(visible_alias = "use", alias = "start")]
    On {
        /// Provider to route to: codex|kimi|grok. Omit to re-enable the last provider used.
        provider: Option<String>,
        /// Don't restart a running interceptor to apply the change (just print the hint).
        #[arg(long)]
        no_restart: bool,
    },
    /// Disable reroute (sets `sub = off`); traffic goes back to Anthropic with compression only.
    ///
    /// `sub stop` is an accepted alias.
    #[command(alias = "stop")]
    Off {
        /// Don't restart a running interceptor to apply the change (just print the hint).
        #[arg(long)]
        no_restart: bool,
    },
    /// Set the reroute mode: `always` (reroute every turn) or `fallback` (use the ordered
    /// subscription chain only when Anthropic cannot serve the turn).
    Mode {
        /// `always` or `fallback`.
        mode: String,
        /// Don't restart a running interceptor to apply the change (just print the hint).
        #[arg(long)]
        no_restart: bool,
    },
    /// Set the ordered providers used by `sub mode fallback` (for example `codex,kimi,grok`).
    Chain {
        /// Comma-separated providers in try order: codex,kimi,grok.
        providers: String,
        /// Don't restart a running interceptor to apply the change.
        #[arg(long)]
        no_restart: bool,
    },
    /// Override the Codex reasoning effort on every rerouted request. By default the reroute honors
    /// the effort Claude Code asks for per turn; this forces one level instead (Kimi ignores it).
    Effort {
        /// `none` | `low` | `medium` | `high` | `xhigh` (`max` = `xhigh`).
        level: String,
        /// Don't restart a running interceptor to apply the change (just print the hint).
        #[arg(long)]
        no_restart: bool,
    },
    /// Map one incoming model (a Claude tier `opus|sonnet|haiku|fable`, or an exact model id) to a
    /// provider model. Non-interactive form of `setup`, for scripts and the tray.
    Map {
        /// Provider whose mapping to edit: codex|kimi|grok.
        provider: String,
        /// Incoming model or tier name to map from.
        from: String,
        /// Provider model to route it to.
        to: String,
        /// Don't restart a running interceptor to apply the change (just print the hint).
        #[arg(long)]
        no_restart: bool,
    },
    /// Remove one mapping entry (falls back to the preset default for that model).
    Unmap {
        /// Provider whose mapping to edit: codex|kimi|grok.
        provider: String,
        /// Incoming model or tier name to unmap.
        from: String,
        /// Don't restart a running interceptor to apply the change (just print the hint).
        #[arg(long)]
        no_restart: bool,
    },
    /// List the provider's candidate models (for autocompletion). `--json` for machine output.
    Models {
        /// Provider: codex|kimi|grok.
        provider: String,
        /// Emit a JSON array of model ids.
        #[arg(long)]
        json: bool,
    },
    /// Print the active reroute selection and tier→model mapping. `--json` for machine output.
    Status {
        /// Emit the full reroute state (provider, mode, mapping, auth) as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Sign in / out of a subscription (`llmtrim sub auth codex login`).
    Auth {
        /// Provider: codex|kimi|grok.
        provider: String,
        #[command(subcommand)]
        action: AuthAction,
    },
}

#[derive(Subcommand)]
enum CompactCmd {
    /// Replace the ordered compact-model alternatives
    Models {
        /// Models or portable Claude tiers, in attempt order (for example: haiku sonnet).
        #[arg(required = true)]
        models: Vec<String>,
        /// Save without restarting a running interceptor.
        #[arg(long)]
        no_restart: bool,
    },
    /// Disable compact model substitution (the original model still handles `/compact`)
    Off {
        /// Save without restarting a running interceptor.
        #[arg(long)]
        no_restart: bool,
    },
    /// Show the configured order and effective backend mappings
    Status {
        /// Emit JSON.
        #[arg(long)]
        json: bool,
    },
}

/// OAuth management for a subscription provider (codex / kimi).
#[derive(Subcommand)]
enum AuthAction {
    /// Sign in (browser OAuth for codex; device-code for kimi).
    Login,
    /// Device-code sign-in for headless machines (codex only).
    Device,
    /// Show stored credentials + token expiry. `--json` for machine output.
    Status {
        /// Emit `{ logged_in, expires_at }` as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Delete stored credentials.
    Logout,
}

#[derive(Subcommand)]
enum McpAction {
    /// Register the llmtrim MCP server with your MCP client
    ///
    /// Installs it into Claude Code via its `claude mcp add` CLI (idempotent — re-running
    /// is a no-op). For any other client, `--print` emits the config block to paste.
    Install {
        /// Print the client config JSON instead of installing it.
        #[arg(long)]
        print: bool,
        /// Overwrite an existing `llmtrim` server entry that differs.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum StatuslineCmd {
    /// Wire the status line into `~/.claude/settings.json`
    Install {
        /// Print the settings snippet instead of editing the file.
        #[arg(long)]
        print: bool,
    },
    /// Remove the status line from `~/.claude/settings.json`
    Uninstall,
}

#[derive(Subcommand)]
enum GuardCmd {
    /// Wire the guard hook into `~/.claude/settings.json`
    Install {
        /// Print the settings snippet instead of editing the file.
        #[arg(long)]
        print: bool,
    },
    /// Remove the guard hook from `~/.claude/settings.json`
    Uninstall,
}

#[derive(clap::Args)]
struct BenchAgentArgs {
    /// Golden task JSON files or directories.
    #[arg(long = "tasks", default_value = "bench/agent")]
    tasks: Vec<PathBuf>,
    /// Comma-separated conditions: `baseline` plus preset names (e.g. `baseline,agent,cache`).
    #[arg(long, default_value = "baseline,agent")]
    conditions: String,
    /// Repeats per (task, condition) — average over noise on live runs.
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    /// Override every task's model id (e.g. `openai/gpt-4o-mini`).
    #[arg(long)]
    model: Option<String>,
    /// Pinned pricing snapshot (models.dev export).
    #[arg(long, default_value = "bench/pricing.json")]
    pricing: PathBuf,
    /// Call the real model. Off by default (synthetic dry-run, no API spend); needs `--features live`.
    #[arg(long)]
    live: bool,
    /// Dry-run only: tool-call rounds before the synthetic model answers.
    #[arg(long, default_value_t = 2)]
    tool_turns: usize,
    /// Write per-run results as JSON here.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Normalized corpus JSONL (friendly {context,question,gold,scorer} or explicit {request,…}).
    #[arg(long)]
    corpus: PathBuf,
    /// Request shape to compress: openai|anthropic|google|gemini. The live A/B path talks to
    /// OpenRouter (OpenAI-shaped), so only `openai` runs live; the rest work `--offline`/`--ablate`.
    #[arg(long, default_value = "openai")]
    provider: String,
    /// Preset to evaluate: auto|safe|lossless|rag|agent|code|aggressive|cache|reasoning.
    #[arg(long, default_value = "auto")]
    preset: String,
    /// Model id to send (OpenRouter style, e.g. openai/gpt-oss-20b).
    #[arg(long, default_value = "openai/gpt-oss-20b")]
    model: String,
    /// LLM-judge model (open-ended scorers only). Defaults to a DIFFERENT model than `--model`
    /// so the judge doesn't grade its own answers; recorded in `--json-out` for reproducibility.
    #[arg(long, default_value = "openai/gpt-4o-mini")]
    judge_model: String,
    /// Pin OpenRouter to one upstream as `provider` or `provider/quant`
    /// (e.g. `groq`). Empty = let OpenRouter choose.
    #[arg(long, default_value = "groq")]
    route: String,
    /// Request a reasoning pass at this effort (`low`/`medium`/`high`), sent as OpenRouter's
    /// `reasoning.effort`. Empty = no reasoning field (default). Needed to exercise
    /// reasoning-gated levers (e.g. the anti-overthink directive), which key off this field
    /// being present on the wire, not the model id.
    #[arg(long, default_value = "")]
    reasoning_effort: String,
    /// Limit the number of cases run.
    #[arg(long)]
    n: Option<usize>,
    /// Override `--preset` with an explicit config TOML (isolates one flag for measurement).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Pinned pricing snapshot (models.dev export).
    #[arg(long, default_value = "bench/pricing.json")]
    pricing: PathBuf,
    /// Skip live calls: compress + measure input-token savings only.
    #[arg(long)]
    offline: bool,
    /// Per-stage input-token ablation (offline): each stage's token contribution.
    #[arg(long)]
    ablate: bool,
    /// Write per-case + frontier results as JSON here.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(clap::Args)]
struct SuiteArgs {
    /// Directory of normalized corpus JSONL files (one per corpus in the matrix).
    #[arg(long, default_value = "bench/data")]
    data_dir: PathBuf,
    /// Directory to write one enveloped result JSON per corpus.
    #[arg(long, default_value = "bench/results")]
    out: PathBuf,
    /// Model id to send (OpenRouter style).
    #[arg(long, default_value = "openai/gpt-oss-20b")]
    model: String,
    /// LLM-judge model for open-ended scorers.
    #[arg(long, default_value = "openai/gpt-4o-mini")]
    judge_model: String,
    /// Pin OpenRouter to one upstream (e.g. `groq`). Empty = let OpenRouter choose.
    #[arg(long, default_value = "groq")]
    route: String,
    /// Pinned pricing snapshot (models.dev export).
    #[arg(long, default_value = "bench/pricing.json")]
    pricing: PathBuf,
    /// Override the per-corpus case count for every corpus (smoke runs).
    #[arg(long)]
    n: Option<usize>,
    /// Skip live calls: compress + measure input-token savings only.
    #[arg(long)]
    offline: bool,
}

#[derive(clap::Args)]
struct LatencyArgs {
    /// Request JSON to compress. Omit for the built-in representative coding turn.
    #[arg(long)]
    request: Option<PathBuf>,
    /// Request shape: openai|anthropic|google|gemini. Inferred when omitted.
    #[arg(long)]
    provider: Option<String>,
    /// Timed iterations on the warm path.
    #[arg(long, default_value_t = 100)]
    iterations: usize,
}

#[derive(clap::Args)]
struct CompareArgs {
    /// Comparator to run: `headroom`, `caveman`, `leanctx`, `entroly`, `rtk`,
    /// or `snip` (drives the matching Python script).
    tool: String,
    /// Call the real model (passed through to the comparator). Off = token-only/dry.
    #[arg(long)]
    live: bool,
    /// Extra arguments forwarded verbatim to the comparator script (after `--`).
    #[arg(last = true)]
    extra: Vec<String>,
}

fn read_stdin() -> Result<String> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("failed to read stdin")?;
    Ok(buf)
}

/// Resolve which subscription backend a window `/sub on [provider]` should use, and require it
/// to be signed in. With an explicit name (`/sub on grok`), only that provider is considered.
/// Without one (`/sub on`), fall back to the global/last-enabled `sub` config.
#[cfg(feature = "intercept")]
fn window_sub_provider(requested: Option<&str>) -> Result<String> {
    use llmtrim::reroute::SubProvider;
    let configured = match requested {
        Some(raw) => {
            let p = SubProvider::parse(raw).ok_or_else(|| {
                anyhow::anyhow!("unknown provider '{raw}' (codex|kimi|grok)")
            })?;
            p.as_str().to_string()
        }
        None => llmtrim_core::config::RuntimeConfig::get()
            .sub
            .clone()
            .or_else(llmtrim_core::config::sub_reenable_provider)
            .context(
                "no provider to re-enable — pass one (`/sub on grok`) or run `llmtrim sub on codex|kimi|grok` first",
            )?,
    };
    let provider =
        SubProvider::parse(&configured).context("configured subscription provider is invalid")?;
    let logged_in = llmtrim::reroute::auth::auth_status_json(provider)
        .get("logged_in")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if !logged_in {
        bail!(
            "{} is not signed in — run `llmtrim sub auth {} login` first",
            provider.as_str(),
            provider.as_str()
        );
    }
    Ok(provider.as_str().to_string())
}

#[cfg(not(feature = "intercept"))]
fn window_sub_provider(_requested: Option<&str>) -> Result<String> {
    bail!("window-scoped rerouting needs the `intercept` feature")
}

fn run_window_sub(args: Vec<String>) -> Result<()> {
    let action = args.first().map(String::as_str).unwrap_or("status");
    let session_from_env = || {
        std::env::var("CLAUDE_CODE_SESSION_ID")
            .ok()
            .or_else(|| std::env::var("CLAUDE_SESSION_ID").ok())
    };
    match action {
        "install" => {
            llmtrim::window_sub::install(&llmtrim::statusline::stable_exe_string())?;
            if let Err(e) = llmtrim::ensure::set_opt_out("window_sub", false) {
                eprintln!("llmtrim: could not clear /sub opt-out: {e:#}");
            }
            println!("Claude Code /sub integration installed.");
        }
        "uninstall" => {
            llmtrim::window_sub::uninstall()?;
            if let Err(e) = llmtrim::ensure::set_opt_out("window_sub", true) {
                eprintln!("llmtrim: could not record /sub opt-out: {e:#}");
            }
            println!("Claude Code /sub integration removed.");
        }
        "hook-start" => {
            let mut raw = String::new();
            std::io::stdin().read_to_string(&mut raw)?;
            let value: serde_json::Value = serde_json::from_str(&raw)?;
            let session = value
                .get("session_id")
                .and_then(serde_json::Value::as_str)
                .context("SessionStart missing session_id")?;
            let source = value
                .get("source")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("startup");
            let existing = std::env::var("LLMTRIM_CLAUDE_WINDOW_TOKEN").ok();
            let token = llmtrim::window_sub::session_start(session, source, existing.as_deref())?;
            if let Ok(file) = std::env::var("CLAUDE_ENV_FILE") {
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file)?;
                writeln!(f, "export LLMTRIM_CLAUDE_WINDOW_TOKEN={token}")?;
            }
        }
        "hook-end" => {
            let mut raw = String::new();
            std::io::stdin().read_to_string(&mut raw)?;
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(s) = v.get("session_id").and_then(serde_json::Value::as_str) {
                let reason = v
                    .get("reason")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let _ = llmtrim::window_sub::session_end(s, reason);
            }
        }
        "slash" => {
            // `$ARGUMENTS` arrives as a single string: "on", "on grok", "off", "status", …
            let raw = args.get(1).map(String::as_str).unwrap_or("status");
            let parts: Vec<&str> = raw.split_whitespace().collect();
            let session = args
                .get(2)
                .filter(|x| !x.is_empty())
                .cloned()
                .or_else(session_from_env)
                .context("Claude Code session unavailable")?;
            match parts.as_slice() {
                ["on"] => {
                    // Re-enable *this window's* last provider, or the global `sub` if none.
                    // Do not silently fall back to another backend when preferred auth fails —
                    // that used to flip grok → codex and clobber `last_provider`.
                    let preferred = llmtrim::window_sub::last_provider(&session);
                    let provider = match preferred.as_deref() {
                        Some(p) => window_sub_provider(Some(p))?,
                        None => window_sub_provider(None)?,
                    };
                    llmtrim::window_sub::set(&session, true, Some(&provider))?;
                    println!("Subscription rerouting enabled for this window ({provider}).");
                }
                ["on", provider_name] => {
                    let provider = window_sub_provider(Some(provider_name))?;
                    llmtrim::window_sub::set(&session, true, Some(&provider))?;
                    println!("Subscription rerouting enabled for this window ({provider}).");
                }
                ["off"] => {
                    llmtrim::window_sub::set(&session, false, None)?;
                    println!(
                        "Subscription rerouting disabled for this window; Anthropic compression remains active."
                    );
                }
                ["status"] | [] => match llmtrim::window_sub::status(&session)? {
                    Some(llmtrim::window_sub::Intent::Enabled { provider }) => {
                        println!("This window: subscription rerouting enabled ({provider}).")
                    }
                    Some(llmtrim::window_sub::Intent::Disabled) => {
                        println!("This window: subscription rerouting disabled.")
                    }
                    None => println!("This window follows the global subscription policy."),
                },
                _ => bail!("usage: /sub on [codex|kimi|grok] | off | status"),
            }
        }
        _ => bail!("unknown window-sub action"),
    };
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprint!("{}", ui::render_error(ui::color_stderr(), &e));
        std::process::exit(1);
    }
}

/// Handle `llmtrim <codex|kimi|grok> auth <action>`.
#[cfg(feature = "intercept")]
fn run_auth(provider: &str, action: AuthAction) -> Result<()> {
    use llmtrim::reroute::{SubProvider, auth};
    // `auth status --json` is provider-agnostic (reads the stored credential file).
    if let AuthAction::Status { json: true } = action {
        let p = SubProvider::parse(provider).ok_or_else(|| anyhow::anyhow!("unknown provider"))?;
        println!("{}", auth::auth_status_json(p));
        return Ok(());
    }
    match (provider, action) {
        ("codex", AuthAction::Login) => auth::codex_login(),
        ("codex", AuthAction::Device) => auth::codex_device(),
        ("codex", AuthAction::Status { .. }) => auth::codex_status(),
        ("codex", AuthAction::Logout) => auth::codex_logout(),
        ("kimi", AuthAction::Login) => auth::kimi_login(),
        ("kimi", AuthAction::Device) => {
            anyhow::bail!("kimi sign-in is device-code already — use `llmtrim sub auth kimi login`")
        }
        ("kimi", AuthAction::Status { .. }) => auth::kimi_status(),
        ("kimi", AuthAction::Logout) => auth::kimi_logout(),
        ("grok", AuthAction::Login) => auth::grok_login(),
        ("grok", AuthAction::Device) => auth::grok_device(),
        ("grok", AuthAction::Status { .. }) => auth::grok_status(),
        ("grok", AuthAction::Logout) => auth::grok_logout(),
        (other, _) => anyhow::bail!("unknown provider '{other}' (codex|kimi|grok)"),
    }
}

// No non-intercept `run_auth` stub: auth is only dispatched from `run_sub`'s `Auth` arm, which is
// itself `#[cfg(feature = "intercept")]`, so a non-intercept build never references it.

/// Report that reroute is now on, nudging to sign in only when there's no stored token. Returns
/// whether a token is present: when it isn't, the caller skips the daemon restart (routing can't
/// work until sign-in, so applying it now is pointless).
#[cfg(feature = "intercept")]
fn print_reroute_enabled(p: llmtrim::reroute::SubProvider) -> bool {
    let logged_in =
        llmtrim::reroute::auth::auth_status_json(p)["logged_in"].as_bool() == Some(true);
    if logged_in {
        println!("Reroute enabled: {}.", p.as_str());
    } else {
        println!(
            "Reroute enabled: {}. Sign in with `llmtrim sub auth {} login`, then \
             `llmtrim start --force`.",
            p.as_str(),
            p.as_str()
        );
    }
    logged_in
}

/// Apply a `sub` config change to the interceptor. The daemon reads its config once at boot
/// (`RuntimeConfig` is a `OnceLock`), so a running one keeps the old routing until restarted.
/// Restarts it in place unless `no_restart` is set or nothing is running (the next `start` picks
/// up the new config on its own).
#[cfg(feature = "intercept")]
fn apply_sub_change(no_restart: bool) {
    let Some(state) = llmtrim::daemon::running() else {
        return;
    };
    if no_restart {
        println!("Restart to apply: `llmtrim start --force`.");
        return;
    }
    let port = state.port;
    let restarted = matches!(llmtrim::daemon::stop_and_wait_free(port), Ok(true))
        .then(|| llmtrim::daemon::spawn_detached(port).ok())
        .flatten();
    match restarted {
        Some(pid) => println!("↻ Restarted interceptor (pid {pid}) to apply."),
        None => eprintln!(
            "llmtrim: change saved, but the restart failed — run `llmtrim start --force`."
        ),
    }
}

/// Apply a per-provider mapping edit (`map`/`unmap`) for `edited`. Only restarts when `edited` is
/// the provider reroute is currently pointed at — a change to some other provider's mapping is
/// inert until you switch to it, so there's no reason to disturb a live session.
#[cfg(feature = "intercept")]
fn apply_sub_map_change(edited: llmtrim::reroute::SubProvider, no_restart: bool) {
    let active = llmtrim_core::config::RuntimeConfig::get()
        .sub
        .as_deref()
        .and_then(llmtrim::reroute::SubProvider::parse);
    if active == Some(edited) {
        apply_sub_change(no_restart);
    }
}

#[cfg(feature = "intercept")]
fn run_compact(action: CompactCmd) -> Result<()> {
    match action {
        CompactCmd::Models { models, no_restart } => {
            let mut normalized = Vec::new();
            for model in models
                .iter()
                .flat_map(|model| model.split(','))
                .map(str::trim)
                .filter(|model| !model.is_empty())
            {
                if !normalized.iter().any(|existing| existing == model) {
                    normalized.push(model.to_string());
                }
            }
            if normalized.is_empty() {
                anyhow::bail!("compact model list is empty; use `llmtrim compact off` to disable");
            }
            llmtrim_core::config::write_compact_models(&normalized)?;
            if let Err(e) = llmtrim::ensure::set_opt_out("compact", false) {
                eprintln!("llmtrim: could not clear compact opt-out: {e:#}");
            }
            println!(
                "Compact models: {} -> original model (implicit).",
                normalized.join(" -> ")
            );
            apply_sub_change(no_restart);
            Ok(())
        }
        CompactCmd::Off { no_restart } => {
            llmtrim_core::config::write_compact_models(&[])?;
            if let Err(e) = llmtrim::ensure::set_opt_out("compact", true) {
                eprintln!("llmtrim: could not record compact opt-out: {e:#}");
            }
            println!("Compact model substitution disabled; `/compact` uses the original model.");
            apply_sub_change(no_restart);
            Ok(())
        }
        CompactCmd::Status { json } => {
            let cfg = llmtrim_core::config::RuntimeConfig::get();
            let sub = cfg
                .sub
                .as_deref()
                .and_then(llmtrim::reroute::SubProvider::parse);
            let original = "<original model>";
            let plan = llmtrim::compact::plan(&cfg.compact_models, original, sub, &cfg.sub_tiers);
            if json {
                let out = serde_json::json!({
                    "models": cfg.compact_models,
                    "original_fallback": true,
                    "resolved": plan.iter().map(|candidate| serde_json::json!({
                        "logical": candidate.logical_model,
                        "upstream": candidate.upstream_model,
                        "context_window": llmtrim_core::context_window(&candidate.upstream_model),
                        "original": candidate.is_original,
                    })).collect::<Vec<_>>(),
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else if cfg.compact_models.is_empty() {
                println!("Compact models: off (original model only).");
            } else {
                println!("Compact models: {}", cfg.compact_models.join(" -> "));
                for candidate in plan.iter().filter(|candidate| !candidate.is_original) {
                    let window = llmtrim_core::context_window(&candidate.upstream_model)
                        .map(|tokens| format!("{tokens} tokens"))
                        .unwrap_or_else(|| "unknown context".to_string());
                    println!(
                        "  {} -> {} ({window})",
                        candidate.logical_model, candidate.upstream_model
                    );
                }
                println!("  then the original requested model (implicit)");
            }
            Ok(())
        }
    }
}

/// Handle `llmtrim sub <setup|on|off|status>`.
#[cfg(feature = "intercept")]
fn run_sub(action: SubCmd) -> Result<()> {
    use llmtrim::reroute::{SubProvider, Tier, default_codex_tier_model};
    let parse = |p: &str| {
        SubProvider::parse(p)
            .ok_or_else(|| anyhow::anyhow!("unknown provider '{p}' (codex|kimi|grok)"))
    };
    match action {
        SubCmd::Setup { provider } => {
            let p = parse(&provider)?;
            #[cfg(feature = "breakdown")]
            {
                llmtrim::reroute::tui::run(p)
            }
            #[cfg(not(feature = "breakdown"))]
            {
                let _ = p;
                anyhow::bail!("the mapping editor needs the `breakdown` feature")
            }
        }
        SubCmd::On {
            provider,
            no_restart,
        } => {
            let logged_in = match provider {
                // Explicit provider: (re)write the default preset, as `use` always has.
                Some(provider) => {
                    let p = parse(&provider)?;
                    let mut map = std::collections::BTreeMap::new();
                    match p {
                        SubProvider::Codex => {
                            for t in Tier::ALL {
                                map.insert(
                                    t.as_str().to_string(),
                                    default_codex_tier_model(t).to_string(),
                                );
                            }
                        }
                        SubProvider::Grok => {
                            for t in Tier::ALL {
                                map.insert(
                                    t.as_str().to_string(),
                                    llmtrim::reroute::default_grok_tier_model(t).to_string(),
                                );
                            }
                        }
                        SubProvider::Kimi => {}
                        // SubProvider is non_exhaustive for semver; new backends need a map here.
                        _ => {}
                    }
                    llmtrim_core::config::write_sub_mapping(p.as_str(), &map)?;
                    print_reroute_enabled(p)
                }
                // Bare `sub on`: restore the last provider, keeping its saved mapping.
                None => {
                    let provider = llmtrim_core::config::sub_reenable_provider().ok_or_else(|| {
                        anyhow::anyhow!(
                            "no provider to re-enable — run `llmtrim sub on codex` (or kimi|grok) first"
                        )
                    })?;
                    let p = parse(&provider)?;
                    llmtrim_core::config::enable_sub(p.as_str())?;
                    print_reroute_enabled(p)
                }
            };
            // Skip the restart when signed out: routing can't work until sign-in anyway.
            if logged_in {
                apply_sub_change(no_restart);
            }
            Ok(())
        }
        SubCmd::Off { no_restart } => {
            llmtrim_core::config::disable_sub()?;
            println!("Reroute disabled — traffic goes to Anthropic (compression only).");
            apply_sub_change(no_restart);
            Ok(())
        }
        SubCmd::Mode { mode, no_restart } => {
            let fallback = match mode.trim().to_ascii_lowercase().as_str() {
                // The old names for the same mode: accepted so an existing muscle-memory command
                // (or script) keeps working, but persisted under the new spelling.
                "fallback" | "on-error" | "on_error" | "onerror" => true,
                "always" => false,
                other => anyhow::bail!("unknown mode '{other}' (always|fallback)"),
            };
            llmtrim_core::config::write_sub_mode(fallback)?;
            println!(
                "Reroute mode: {}.",
                if fallback { "fallback" } else { "always" }
            );
            apply_sub_change(no_restart);
            Ok(())
        }
        SubCmd::Chain {
            providers,
            no_restart,
        } => {
            let mut chain = Vec::new();
            for raw in providers.split(',') {
                let raw = raw.trim();
                if raw.is_empty() {
                    continue;
                }
                let p = parse(raw)?;
                if !chain.contains(&p.as_str().to_string()) {
                    chain.push(p.as_str().to_string());
                }
            }
            if chain.is_empty() {
                anyhow::bail!("fallback chain is empty (expected codex,kimi)");
            }
            llmtrim_core::config::write_sub_chain(&chain)?;
            println!("Fallback chain: {}.", chain.join(" -> "));
            apply_sub_change(no_restart);
            Ok(())
        }
        SubCmd::Effort { level, no_restart } => {
            let level = match level.trim().to_ascii_lowercase().as_str() {
                "max" => "xhigh".to_string(),
                l @ ("none" | "low" | "medium" | "high" | "xhigh") => l.to_string(),
                other => anyhow::bail!("unknown effort '{other}' (none|low|medium|high|xhigh)"),
            };
            llmtrim_core::config::write_sub_effort(SubProvider::Codex.as_str(), &level)?;
            println!("Codex reasoning effort: {level}.");
            // Effort is Codex-only; if the active provider isn't Codex it's inert until you switch.
            let active = llmtrim_core::config::RuntimeConfig::get()
                .sub
                .as_deref()
                .and_then(SubProvider::parse);
            if active == Some(SubProvider::Codex) {
                apply_sub_change(no_restart);
            } else {
                eprintln!(
                    "note: reroute is not on codex right now, so this effort has no effect until \
                     `llmtrim sub on codex`."
                );
            }
            Ok(())
        }
        SubCmd::Map {
            provider,
            from,
            to,
            no_restart,
        } => {
            let p = parse(&provider)?;
            llmtrim_core::config::write_sub_map_entry(p.as_str(), &from, &to)?;
            println!("Mapped {from} -> {to} for {}.", p.as_str());
            apply_sub_map_change(p, no_restart);
            Ok(())
        }
        SubCmd::Unmap {
            provider,
            from,
            no_restart,
        } => {
            let p = parse(&provider)?;
            llmtrim_core::config::remove_sub_map_entry(p.as_str(), &from)?;
            println!(
                "Unmapped {from} for {} (back to the preset default).",
                p.as_str()
            );
            apply_sub_map_change(p, no_restart);
            Ok(())
        }
        SubCmd::Models { provider, json } => {
            let p = parse(&provider)?;
            let ids: Vec<String> = llmtrim::reroute::catalog::models_for(p)
                .into_iter()
                .map(|e| e.id)
                .collect();
            if json {
                println!("{}", serde_json::to_string(&ids)?);
            } else {
                for id in ids {
                    println!("{id}");
                }
            }
            Ok(())
        }
        SubCmd::Status { json } => {
            let cfg = llmtrim_core::config::RuntimeConfig::get();
            let active = cfg.sub.as_deref().and_then(SubProvider::parse);
            if json {
                // One effective map: the four resolved Codex tiers, then every configured entry
                // laid over them (a configured tier wins, a free-form `model-id -> model` adds a
                // row). One key, not a resolved/raw pair the caller has to reconcile.
                let mut mapping: std::collections::BTreeMap<String, String> = match active {
                    Some(SubProvider::Codex) => Tier::ALL
                        .iter()
                        .map(|t| {
                            (
                                t.as_str().to_string(),
                                default_codex_tier_model(*t).to_string(),
                            )
                        })
                        .collect(),
                    Some(SubProvider::Grok) => Tier::ALL
                        .iter()
                        .map(|t| {
                            (
                                t.as_str().to_string(),
                                llmtrim::reroute::default_grok_tier_model(*t).to_string(),
                            )
                        })
                        .collect(),
                    _ => std::collections::BTreeMap::new(),
                };
                for (k, v) in &cfg.sub_tiers {
                    mapping.insert(k.clone(), v.clone());
                }
                let out = serde_json::json!({
                    "provider": active.map(|p| p.as_str()),
                    "mode": if cfg.sub_fallback { "fallback" } else { "always" },
                    "chain": cfg.sub_chain,
                    "effort": cfg.sub_effort,
                    "mapping": mapping,
                    "auth": {
                        "codex": llmtrim::reroute::auth::auth_status_json(SubProvider::Codex),
                        "kimi": llmtrim::reroute::auth::auth_status_json(SubProvider::Kimi),
                        "grok": llmtrim::reroute::auth::auth_status_json(SubProvider::Grok),
                    },
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }
            match active {
                None => println!("Reroute: off"),
                Some(p) => {
                    println!(
                        "Reroute: {} ({})",
                        p.as_str(),
                        if cfg.sub_fallback {
                            "fallback"
                        } else {
                            "always"
                        }
                    );
                    if cfg.sub_fallback {
                        println!(
                            "  chain   -> {}",
                            if cfg.sub_chain.is_empty() {
                                p.as_str().to_string()
                            } else {
                                cfg.sub_chain.join(" -> ")
                            }
                        );
                    }
                    match p {
                        SubProvider::Codex => {
                            println!(
                                "  effort  -> {}",
                                cfg.sub_effort.as_deref().unwrap_or("none")
                            );
                            for t in Tier::ALL {
                                let model = cfg
                                    .sub_tiers
                                    .get(t.as_str())
                                    .cloned()
                                    .unwrap_or_else(|| default_codex_tier_model(t).to_string());
                                println!("  {:<7} -> {model}", t.as_str());
                            }
                            // Free-form model→model overrides (keys that aren't one of the four tiers).
                            let tier_keys: [&str; 4] = Tier::ALL.map(|t| t.as_str());
                            for (from, to) in &cfg.sub_tiers {
                                if !tier_keys.contains(&from.as_str()) {
                                    println!("  {from} -> {to}");
                                }
                            }
                        }
                        SubProvider::Grok => {
                            for t in Tier::ALL {
                                let model =
                                    cfg.sub_tiers.get(t.as_str()).cloned().unwrap_or_else(|| {
                                        llmtrim::reroute::default_grok_tier_model(t).to_string()
                                    });
                                println!("  {:<7} -> {model}", t.as_str());
                            }
                            let tier_keys: [&str; 4] = Tier::ALL.map(|t| t.as_str());
                            for (from, to) in &cfg.sub_tiers {
                                if !tier_keys.contains(&from.as_str()) {
                                    println!("  {from} -> {to}");
                                }
                            }
                        }
                        SubProvider::Kimi => {
                            println!("  all tiers -> {}", llmtrim::reroute::KIMI_MODEL);
                        }
                        _ => println!("  (unknown provider)"),
                    }
                    // Auth is what makes reroute actually work — surface it here, not just in JSON.
                    let auth = llmtrim::reroute::auth::auth_status_json(p);
                    if auth["logged_in"].as_bool().unwrap_or(false) {
                        println!("  auth: logged in");
                    } else {
                        println!(
                            "  auth: NOT logged in — run `llmtrim sub auth {} login`",
                            p.as_str()
                        );
                    }
                }
            }
            Ok(())
        }
        SubCmd::Auth { provider, action } => run_auth(&provider, action),
    }
}

#[cfg(not(feature = "intercept"))]
fn run_sub(_action: SubCmd) -> Result<()> {
    anyhow::bail!("this build has no interceptor; rebuild with `--features intercept`")
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compress { provider } => {
            let input = read_stdin()?;
            let kind = provider
                .as_deref()
                .map(ProviderKind::from_str)
                .transpose()?;
            let result = llmtrim_core::compress(&input, kind)?;

            // Record to the savings ledger (best-effort: a ledger failure must never
            // block the user's compressed output).
            if let Ok(tracker) = Tracker::open() {
                let _ = tracker.record(&Record {
                    provider: result.provider.as_str().to_string(),
                    model: result.model.clone(),
                    tokenizer: result.tokenizer_label.clone(),
                    exact: result.tokenizer_exact,
                    input_before: result.input_tokens_before.0 as i64,
                    input_after: result.input_tokens_after.0 as i64,
                    output_before: None,
                    output_after: None,
                    compress_micros: None,
                    cache_read_tokens: None,
                    fresh_input_tokens: None,
                    cache_write_tokens: None,
                    output_shaped: Some(result.output_shaped),
                    frozen_input_tokens: Some(result.frozen_input_tokens.0 as i64),
                });
            }

            let mut stdout = std::io::stdout().lock();
            stdout
                .write_all(result.request_json.as_bytes())
                .context("failed to write stdout")?;
            let _ = stdout.write_all(b"\n");
        }
        Commands::Send { provider } => {
            let input = read_stdin()?;
            let kind = provider
                .as_deref()
                .map(ProviderKind::from_str)
                .transpose()?;
            let result = llmtrim_core::compress(&input, kind)?;
            let endpoint = Endpoint::from_env(result.provider)?;
            let proxy_url = llmtrim::transport::upstream_proxy_url(None)?;
            let response = endpoint.send(&result.request_json, proxy_url.as_deref())?;

            // Record to the savings ledger (best-effort: never block the user's output).
            if let Ok(counter) =
                llmtrim_core::tokenizer::counter_for(result.provider, result.model.as_deref())
            {
                let output_after = serde_json::from_str::<serde_json::Value>(&response)
                    .ok()
                    .and_then(|v| llmtrim_core::provider::for_kind(result.provider).answer_text(&v))
                    .map(|text| counter.count(&text) as i64);
                if let Ok(tracker) = Tracker::open() {
                    let _ = tracker.record(&Record {
                        provider: result.provider.as_str().to_string(),
                        model: result.model.clone(),
                        tokenizer: counter.label().to_string(),
                        exact: counter.is_exact(),
                        input_before: result.input_tokens_before.0 as i64,
                        input_after: result.input_tokens_after.0 as i64,
                        output_before: None,
                        output_after,
                        compress_micros: None,
                        cache_read_tokens: None,
                        fresh_input_tokens: None,
                        cache_write_tokens: None,
                        output_shaped: Some(result.output_shaped),
                        frozen_input_tokens: Some(result.frozen_input_tokens.0 as i64),
                    });
                }
            }

            println!("{response}");
        }
        Commands::Serve {
            port,
            force,
            supervised,
            hide_console,
        } => {
            if hide_console {
                llmtrim::autostart::hide_console_window();
            }
            if supervised {
                // Self-heal an env block wired before the NO_PROXY bypass existed, so installs
                // predating it stop funneling LAN/local traffic at the interceptor — without the
                // user re-running `setup`. Gated to the supervised daemon (the at-login path):
                // a foreground `llmtrim serve` must not rewrite the user's shell profiles.
                // Best-effort: never let it stop the proxy coming up.
                let _ = llmtrim::setup::heal_managed_env();
                // Windows: the daemon is up, so wire the interceptor env into the registry now.
                // POSIX self-gates via the shell block's `_alive` probe, but Windows env is read
                // at process launch with no per-shell hook, so we set it here (the one process
                // both `start` and login autostart bring the daemon up through) and clear it in
                // `stop`. Best-effort: never block the proxy coming up.
                #[cfg(windows)]
                let _ = llmtrim::setup::wire_env_windows(port);
                llmtrim::serve::run_supervised(port, force)?;
            } else {
                llmtrim::serve::run(port, force)?;
            }
        }
        Commands::Setup { port, force } => llmtrim::setup::run(port, force)?,
        Commands::Ensure { quiet } => llmtrim::ensure::run_cli(quiet)?,
        Commands::Uninstall { purge, keep_binary } => {
            llmtrim::setup::uninstall(purge, keep_binary)?
        }
        Commands::Start { port, force } => {
            let color = ui::color_stdout();
            if force && let Some(state) = llmtrim::daemon::running() {
                println!(
                    "{}",
                    ui::ok(
                        color,
                        &format!("Stopping existing daemon · pid {} (--force)", state.pid)
                    )
                );
                // Wait for the port to actually free before spawning, so the new daemon doesn't
                // lose the bind race and crash on EADDRINUSE.
                if !llmtrim::daemon::stop_and_wait_free(state.port)? {
                    anyhow::bail!(
                        "--force stopped pid {} but port {} was still held after 5s",
                        state.pid,
                        state.port
                    );
                }
            }
            if let Some(state) = llmtrim::daemon::running() {
                println!(
                    "{}",
                    ui::ok(
                        color,
                        &format!(
                            "Interceptor already running · pid {} · port {}",
                            state.pid, state.port
                        )
                    )
                );
                // Quiet migration after version bumps — no prompts, no tray download.
                let _ = llmtrim::ensure::maybe_auto();
            } else {
                // No live daemon → resolve the port the same way `setup` does (explicit, else
                // the one already wired into the env, else DEFAULT_PORT) so we match clients.
                let port = llmtrim::setup::resolve_port(port, None)?;
                let pid = llmtrim::daemon::spawn_detached(port)?;
                println!(
                    "{}",
                    ui::ok(
                        color,
                        &format!("Interceptor running · pid {pid} · port {port}")
                    )
                );
                let _ = llmtrim::ensure::maybe_auto();
                for (label, value) in [
                    ("watch", "llmtrim status".to_string()),
                    ("stop", "llmtrim stop".to_string()),
                ] {
                    println!(
                        "    {}  {value}",
                        ui::paint(color, Tone::Dim, &format!("{label:<5}"))
                    );
                }
                if !llmtrim::setup::profile_has_block() {
                    eprintln!(
                        "{}",
                        ui::note(
                            ui::color_stderr(),
                            "HTTPS_PROXY isn't set for your user yet — run `llmtrim setup` so \
                             tools actually route through the interceptor."
                        )
                    );
                }
            }
            // Once-only pointer to the desktop tray, for users who updated into a
            // bundle that now ships it (silent on cargo-only installs).
            if let Some(msg) = llmtrim::tray::nudge_once() {
                println!("{}", ui::paint(color, Tone::Dim, &msg));
            }
        }
        Commands::Wrap { args } => llmtrim::wrap::run(args)?,
        Commands::Alive => {
            // The shell-profile block runs this on every new shell to decide whether to wire
            // HTTPS_PROXY. Keep it silent and pidfile-only (no TCP probe) so shell startup
            // stays cheap. Exit 0 = daemon up, 1 = down.
            if llmtrim::daemon::running().is_none() {
                std::process::exit(1);
            }
        }
        Commands::Stop => match llmtrim::daemon::stop()? {
            Some(pid) => {
                println!(
                    "{}",
                    ui::ok(
                        ui::color_stdout(),
                        &format!("Stopped interceptor (pid {pid}).")
                    )
                );
                // New shells self-heal: the managed block only wires HTTPS_PROXY while the
                // daemon is up, so a fresh terminal now talks to LLM hosts directly. But this
                // shell already exported the vars at launch and a child can't unset them in its
                // parent — so hand the user the one-liner. Only when HTTPS_PROXY actually points at
                // the local interceptor, so we never tell someone to strip an unrelated corporate
                // proxy they happen to have set.
                #[cfg(not(windows))]
                if llmtrim::wrap::https_proxy_is_local() {
                    eprintln!(
                        "{}",
                        ui::warn(
                            ui::color_stderr(),
                            &format!(
                                "This shell still points at the stopped interceptor. New \
                                 terminals are already clear; to fix this one, run:\n    {}",
                                llmtrim::setup::UNSET_HINT
                            )
                        )
                    );
                }
                // Windows has no per-shell gate (env lives in HKCU\Environment, read at process
                // launch), so clear the registry values now: newly-launched terminals and apps
                // come up clear. `start` re-wires them when the daemon is back up.
                #[cfg(windows)]
                match llmtrim::setup::unwire_env_windows() {
                    Ok(true) => eprintln!(
                        "{}",
                        ui::note(
                            ui::color_stderr(),
                            "Cleared HTTPS_PROXY from your user environment — new terminals and \
                             apps are clear. Apps already running keep the old proxy until you \
                             restart them."
                        )
                    ),
                    Ok(false) => {}
                    Err(e) => eprintln!(
                        "{}",
                        ui::warn(
                            ui::color_stderr(),
                            &format!("could not clear the interceptor env: {e}")
                        )
                    ),
                }
            }
            None => println!(
                "{}",
                ui::note(ui::color_stdout(), "No interceptor daemon was running.")
            ),
        },
        Commands::Sub { action } => run_sub(action)?,
        Commands::Compact { action } => run_compact(action)?,
        Commands::WindowSub { args } => {
            let hook = args.first().is_some_and(|arg| arg.starts_with("hook-"));
            if let Err(error) = run_window_sub(args) {
                if hook {
                    eprintln!("llmtrim: window /sub hook skipped: {error:#}");
                } else {
                    return Err(error);
                }
            }
        }
        Commands::Update => llmtrim::update::run()?,
        Commands::Mcp { action } => match action {
            None => llmtrim::mcp::run()?,
            Some(McpAction::Install { print, force }) => llmtrim::mcp::install(print, force)?,
        },
        Commands::Statusline { action } => match action {
            None => llmtrim::statusline::run()?,
            Some(StatuslineCmd::Install { print }) => {
                llmtrim::statusline::install(print)?;
                if !print && let Err(e) = llmtrim::ensure::set_opt_out("statusline", false) {
                    eprintln!("llmtrim: could not clear statusline opt-out: {e:#}");
                }
            }
            Some(StatuslineCmd::Uninstall) => {
                llmtrim::statusline::uninstall()?;
                if let Err(e) = llmtrim::ensure::set_opt_out("statusline", true) {
                    eprintln!("llmtrim: could not record statusline opt-out: {e:#}");
                }
            }
        },
        Commands::Guard { action } => match action {
            // The hook path: Claude Code reads the exit code, so return it rather than a Result —
            // 2 blocks the prompt, anything else lets it through.
            None => std::process::exit(llmtrim::guard::run()),
            Some(GuardCmd::Install { print }) => {
                llmtrim::guard::install(print)?;
                if !print && let Err(e) = llmtrim::ensure::set_opt_out("guard", false) {
                    eprintln!("llmtrim: could not clear guard opt-out: {e:#}");
                }
            }
            Some(GuardCmd::Uninstall) => {
                llmtrim::guard::uninstall()?;
                if let Err(e) = llmtrim::ensure::set_opt_out("guard", true) {
                    eprintln!("llmtrim: could not record guard opt-out: {e:#}");
                }
            }
        },
        Commands::Monitor {
            // Deprecated no-op (see the field docs): accepted, then ignored.
            watch: _,
            interval,
            daily,
            weekly,
            monthly,
            json,
            breakdown,
            csv,
            quiet,
        } => run_monitor(
            interval, daily, weekly, monthly, json, breakdown, csv, quiet,
        )?,
        Commands::Doctor { fix } => {
            if fix {
                llmtrim::ensure::run_cli(false)?;
            }
            let report = llmtrim::doctor::gather();
            print!("{}", llmtrim::doctor::render(ui::color_stdout(), &report));
            if report.problems > 0 {
                std::process::exit(1);
            }
        }
        Commands::Tray => llmtrim::tray::run()?,
        Commands::Autostart {
            off,
            port,
            tray,
            status,
        } => {
            let color = ui::color_stdout();
            if status {
                let enabled = if tray {
                    llmtrim::autostart::is_tray_enabled()
                } else {
                    llmtrim::autostart::is_enabled()
                };
                println!("{}", if enabled { "enabled" } else { "disabled" });
                return Ok(());
            }
            if tray {
                llmtrim::autostart::configure_tray(!off)?;
                let msg = if off {
                    "Tray autostart disabled."
                } else {
                    "Tray autostart enabled — the llmtrim tray opens at login."
                };
                println!("{}", ui::ok(color, msg));
                return Ok(());
            }
            if off {
                // Port is unused when disabling; pass the default to match `uninstall`.
                llmtrim::autostart::configure(false, llmtrim::setup::DEFAULT_PORT)?;
                println!("{}", ui::ok(color, "Autostart disabled."));
            } else {
                // Match the daemon/env so login doesn't come up on a port the env isn't
                // wired to. `resolve_port` prefers an explicit `--port`, then the running
                // daemon, then the configured env, and only scans from the default when
                // nothing is pinned (a first install).
                let running = llmtrim::daemon::running().map(|s| s.port);
                let port = llmtrim::setup::resolve_port(port, running)
                    .context("llmtrim autostart: could not pick a port — pass --port explicitly")?;
                llmtrim::autostart::configure(true, port)?;
                println!(
                    "{}",
                    ui::ok(
                        color,
                        &format!("Autostart enabled — llmtrim serve --port {port} runs at login.")
                    )
                );
                println!(
                    "    {}  llmtrim autostart --off",
                    ui::paint(color, Tone::Dim, "undo ")
                );
            }
        }
        Commands::Ca { pem } => {
            let path = llmtrim::serve::ca_cert_path()?;
            llmtrim::serve::ensure_ca()?; // generate on first run
            if pem {
                // Bare PEM for piping (e.g. out of a container into NODE_EXTRA_CA_CERTS).
                print!("{}", std::fs::read_to_string(&path)?);
                return Ok(());
            }
            let color = ui::color_stdout();
            print!(
                "{}",
                ui::panel(color, "llmtrim local CA", &[path.display().to_string()])
            );
            println!();
            println!("Trust it for your tool, then route its traffic through llmtrim:");
            #[cfg(windows)]
            {
                println!("  $env:NODE_EXTRA_CA_CERTS = \"{}\"", path.display());
                println!(
                    "  $env:HTTPS_PROXY = \"http://127.0.0.1:{}\"",
                    llmtrim::setup::DEFAULT_PORT
                );
                println!("  llmtrim serve");
                println!();
                println!("System-wide trust (non-PowerShell / GUI apps):");
                println!("  certutil -addstore -user Root \"{}\"", path.display());
            }
            #[cfg(not(windows))]
            {
                println!("  export NODE_EXTRA_CA_CERTS={}", path.display());
                println!(
                    "  export HTTPS_PROXY=http://127.0.0.1:{}",
                    llmtrim::setup::DEFAULT_PORT
                );
                println!("  llmtrim serve");
            }
            println!();
            println!(
                "{}",
                ui::note(
                    ui::color_stdout(),
                    "The CA is name-constrained to LLM API domains only."
                )
            );
        }
        Commands::Eval {
            corpus,
            provider,
            keep_ratio,
        } => {
            let kind = ProviderKind::from_str(&provider)?;
            let jsonl = std::fs::read_to_string(&corpus)
                .with_context(|| format!("failed to read corpus {}", corpus.display()))?;
            let cases = llmtrim::quality::load_corpus(&jsonl, kind)?;
            // Clamp the user-supplied keep ratio to its valid [0,1] domain at the boundary.
            let keep_ratio = keep_ratio.clamp(0.0, 1.0);
            let config = llmtrim_core::config::DenseConfig {
                retrieve: true,
                retrieve_keep_ratio: keep_ratio,
                retrieve_min_segment_chars: 120,
                output_control: false,
                ..Default::default()
            };
            let results = llmtrim::quality::run_recall(&cases, &config)?;
            let color = ui::color_stdout();
            // "-36.7%" for a real saving, plain dim "0.0%" when nothing was cut —
            // a literal "-" prefix would render the odd-looking "-0.0%".
            let saved_cell = |pct: f64| {
                if pct > 0.0 {
                    ui::paint(color, Tone::Accent, &format!("-{pct:.1}%"))
                } else {
                    ui::paint(color, Tone::Dim, &format!("{:.1}%", pct.abs()))
                }
            };
            let mut t = ui::table(color, &["case", "recall", "tokens", "saved"]);
            for r in &results {
                t.add_row(vec![
                    comfy_table::Cell::new(&r.name),
                    ui::right(format!("{:.2}", r.recall)),
                    ui::right(format!("{} → {}", r.tokens_before, r.tokens_after)),
                    ui::right(saved_cell(r.savings_pct())),
                ]);
            }
            for line in t.to_string().lines() {
                println!(" {line}");
            }
            let recall = llmtrim::quality::mean_recall(&results);
            let savings = if results.is_empty() {
                0.0
            } else {
                results.iter().map(|r| r.savings_pct()).sum::<f64>() / results.len() as f64
            };
            print!(
                "\n{}",
                ui::panel(
                    color,
                    "eval",
                    &[format!(
                        "{} cases   mean recall {}   mean savings {}   keep_ratio {keep_ratio}",
                        results.len(),
                        ui::paint(color, Tone::Bold, &format!("{recall:.2}")),
                        saved_cell(savings),
                    )]
                )
            );
        }
        Commands::Discover {
            dir,
            by_tool,
            limit,
            json,
        } => llmtrim::discover::run(dir, json, by_tool, limit)?,
        Commands::Bench(cmd) => match cmd {
            BenchCmd::Quality(args) => run_bench(*args)?,
            BenchCmd::Suite(args) => run_bench_suite(*args)?,
            BenchCmd::Agent(args) => run_bench_agent(*args)?,
            BenchCmd::Latency(args) => run_bench_latency(args)?,
            BenchCmd::Compare(args) => run_bench_compare(args)?,
        },
    }
    Ok(())
}

/// Read a key from the process env, falling back to a local `.env` parsed by `dotenvy`
/// (parse-only — not loaded into the global environment, which edition 2024 makes
/// `unsafe`; this keeps the compressor core free of env mutation).
#[cfg(feature = "live")]
fn dotenv_get(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key) {
        return Some(v);
    }
    dotenvy::from_path_iter(".env")
        .ok()?
        .flatten()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
}

/// Pin the request to the chosen model and (optionally) a single OpenRouter upstream
/// provider and/or a reasoning pass. All three fields are top-level passthrough, so
/// compression preserves them and the original and compressed sends route identically.
fn prepare_request(request_json: &str, model: &str, route: &str, reasoning_effort: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(request_json) {
        Ok(mut v) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("model".to_string(), serde_json::json!(model));
                if !route.is_empty() {
                    // route is "provider" or "provider/quant" (e.g. "parasail/fp4") —
                    // OpenRouter pins the upstream via `order` + `quantizations`.
                    let (provider, quant) = match route.split_once('/') {
                        Some((p, q)) => (p, Some(q)),
                        None => (route, None),
                    };
                    let mut routing =
                        serde_json::json!({"order": [provider], "allow_fallbacks": false});
                    if let Some(q) = quant {
                        routing["quantizations"] = serde_json::json!([q]);
                    }
                    obj.insert("provider".to_string(), routing);
                }
                if !reasoning_effort.is_empty() {
                    obj.insert(
                        "reasoning".to_string(),
                        serde_json::json!({"effort": reasoning_effort}),
                    );
                }
                // Ask OpenRouter for the detailed usage breakdown (incl. cached_tokens).
                obj.insert("usage".to_string(), serde_json::json!({"include": true}));
            }
            v.to_string()
        }
        Err(_) => request_json.to_string(),
    }
}

fn run_bench(args: BenchArgs) -> Result<()> {
    let kind = ProviderKind::from_str(&args.provider)?;
    let jsonl = std::fs::read_to_string(&args.corpus)
        .with_context(|| format!("failed to read corpus {}", args.corpus.display()))?;
    let mut cases = bench::load_bench_corpus(&jsonl, kind, &args.model)?;
    if let Some(limit) = args.n {
        cases.truncate(limit);
    }
    // Pin every case to the chosen model + upstream provider so the orig/compressed
    // sends are comparable and route identically.
    for c in &mut cases {
        c.request = prepare_request(&c.request, &args.model, &args.route, &args.reasoning_effort);
    }
    // `--config FILE` overrides the preset with an explicit config (isolates a single flag
    // for measurement); otherwise resolve the named preset.
    let mut config = match &args.config {
        Some(path) => {
            let t = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read config {}", path.display()))?;
            toml::from_str::<DenseConfig>(&t)
                .with_context(|| format!("failed to parse config {}", path.display()))?
        }
        None => DenseConfig::preset(&args.preset).with_context(|| {
            format!(
                "unknown preset '{}' (auto|safe|lossless|rag|agent|code|aggressive|cache|reasoning)",
                args.preset
            )
        })?,
    };
    // Clamp ratio knobs to their valid [0,1] domain at the boundary, so an out-of-range
    // config/preset value can't drive a nonsensical keep fraction downstream.
    config.retrieve_keep_ratio = config.retrieve_keep_ratio.clamp(0.0, 1.0);
    config.retrieve_mmr_lambda = config.retrieve_mmr_lambda.clamp(0.0, 1.0);

    if args.ablate {
        run_ablation(&cases, &config, &args.preset)
    } else if args.offline {
        run_offline(&cases, kind, &config, &args)
    } else {
        run_live(&cases, kind, &config, &args)
    }
}

/// The corpus matrix: each corpus paired with the shape-matched preset and case count.
/// Ported from the old run_all.sh so the suite is the single source of truth.
const SUITE_MATRIX: &[(&str, &str, usize)] = &[
    ("gsm8k", "reasoning", 12),  // reasoning  → Chain-of-Draft
    ("humaneval", "code", 12),   // code gen   → skeleton/minify
    ("dolly", "aggressive", 12), // generation → output-control (judge)
    ("hotpotqa", "rag", 12),     // multi-hop  → retrieve (long ctx)
    ("glaive", "agent", 12),     // tool use   → tool select/trim
    ("chat", "aggressive", 12),  // multi-turn → output-control + dedup (judge)
    ("cnn", "aggressive", 8),    // long doc   → output budget
    ("cache", "cache", 12),      // shared prefix → cache-first (Stage A)
    // Named academic benchmarks, run at a conservative shape-matched preset (the honest
    // headline for an accuracy-preservation claim).
    ("truthfulqa", "safe", 20), // factual MC1 → safe (no lossy output cuts)
    ("squad2", "rag", 20),      // extractive QA → retrieve (long ctx)
    ("bfcl", "agent", 20),      // function call (multi-tool) → tool select/trim
];

fn run_bench_suite(args: SuiteArgs) -> Result<()> {
    // The A/B compares in-process compression against the true original request. If the
    // llmtrim proxy is in the environment, both arms get re-compressed in flight and the
    // baseline is no longer original, contaminating every number. Refuse to run live
    // through the proxy rather than silently produce bad data.
    if !args.offline {
        for k in [
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
        ] {
            if std::env::var_os(k).is_some() {
                anyhow::bail!(
                    "{k} is set; the llmtrim proxy would re-compress the baseline arm and \
                     contaminate the A/B. Unset all *_PROXY vars before a live suite run."
                );
            }
        }
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("failed to create {}", args.out.display()))?;

    for (corpus, preset, default_n) in SUITE_MATRIX {
        let n = args.n.or(Some(*default_n));
        eprintln!(
            "{}",
            ui::note(
                ui::color_stderr(),
                &format!("=== {corpus} ({preset}, n={}) ===", n.unwrap_or(0))
            )
        );
        let corpus_args = BenchArgs {
            corpus: args.data_dir.join(format!("{corpus}.jsonl")),
            provider: "openai".to_string(),
            preset: (*preset).to_string(),
            model: args.model.clone(),
            judge_model: args.judge_model.clone(),
            route: args.route.clone(),
            reasoning_effort: String::new(),
            n,
            config: None,
            pricing: args.pricing.clone(),
            offline: args.offline,
            ablate: false,
            json_out: Some(args.out.join(format!("{corpus}.json"))),
        };
        // One failing corpus must not abort the matrix; report and continue.
        if let Err(e) = run_bench(corpus_args) {
            eprintln!(
                "{}",
                ui::render_error(ui::color_stderr(), &e.context(format!("corpus {corpus}")))
            );
        }
    }
    eprintln!("{}", ui::note(ui::color_stderr(), "suite complete"));
    Ok(())
}

fn run_bench_latency(args: LatencyArgs) -> Result<()> {
    use std::time::Instant;
    // Representative coding turn: system + a user message with a fenced code block + prose,
    // exercising hygiene, skeletonization, and tokenization.
    const FIXTURE: &str = r#"{"model":"gpt-4o","messages":[
{"role":"system","content":"You are a meticulous coding assistant. Answer precisely."},
{"role":"user","content":"Review this function for bugs:\n```rust\nfn process(data: &[i32]) -> i32 {\n    let mut total = 0;\n    for x in data {\n        if *x > 0 {\n            total += x * 2;\n        }\n    }\n    total\n}\n```\nDoes it handle the spec's edge cases?"}
]}"#;
    let input = match &args.request {
        Some(p) => {
            std::fs::read_to_string(p).with_context(|| format!("failed to read {}", p.display()))?
        }
        None => FIXTURE.to_string(),
    };
    let kind = args
        .provider
        .as_deref()
        .map(ProviderKind::from_str)
        .transpose()?;
    let config = DenseConfig::auto();

    // Warm: tokenizer vocab + lazy regexes (the daemon pays this once at startup, not per
    // request), so steady-state numbers reflect the warm path.
    for _ in 0..5 {
        let _ = llmtrim_core::compress_with_config(&input, kind, &config)?;
    }
    let n = args.iterations.max(1);
    // Run once up front so `r` is unconditionally populated (no Option/expect), then time n.
    let mut r = llmtrim_core::compress_with_config(&input, kind, &config)?;
    let t = Instant::now();
    for _ in 0..n {
        r = llmtrim_core::compress_with_config(&input, kind, &config)?;
    }
    let per_req_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

    let counter = llmtrim_core::tokenizer::counter_for(r.provider, r.model.as_deref())?;
    let value: serde_json::Value = serde_json::from_str(&input).context("parse request JSON")?;
    let tools_str = value
        .get("tools")
        .map(|t| t.to_string())
        .unwrap_or_default();
    let bench = |label: &str, f: &dyn Fn()| {
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        println!(
            "  {label}: {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0 / n as f64
        );
    };

    println!("request: {} bytes, provider={:?}", input.len(), r.provider);
    println!(
        "input tokens: {} -> {} ({:.1}% saved)",
        r.input_tokens_before,
        r.input_tokens_after,
        100.0 * (1.0 - r.input_tokens_after.0 as f64 / r.input_tokens_before.0.max(1) as f64)
    );
    println!("compress latency: {per_req_ms:.2} ms/req (warm, avg of {n})");
    println!("attribution (1x each):");
    bench("full tokenize (content+tools)", &|| {
        let _ = counter.count(&input);
    });
    bench("tools tokenize only", &|| {
        let _ = counter.count(&tools_str);
    });
    bench("Value clone (per-stage snapshot)", &|| {
        let _ = value.clone();
    });
    bench("JSON parse", &|| {
        let _ = serde_json::from_str::<serde_json::Value>(&input);
    });
    bench("JSON serialize", &|| {
        let _ = value.to_string();
    });
    Ok(())
}

fn run_bench_compare(args: CompareArgs) -> Result<()> {
    match args.tool.as_str() {
        "headroom" | "caveman" | "leanctx" | "entroly" | "rtk" | "snip" => {}
        other => anyhow::bail!(
            "unknown comparator '{other}' (expected: \
             headroom|caveman|leanctx|entroly|rtk|snip)"
        ),
    }
    // One entry point for every competitor: `bench.py <competitor>`. Adding a comparator is a
    // new file under bench/scripts/benchkit/competitors/, no change here beyond the match above.
    let script = "bench/scripts/bench.py";
    if !std::path::Path::new(script).exists() {
        anyhow::bail!(
            "{script} not found - run `llmtrim bench compare` from the repo root (it dispatches \
             the Python comparator, which needs its own deps; see bench/README.md)"
        );
    }
    let mut cmd = std::process::Command::new("python3");
    cmd.arg(script).arg(&args.tool);
    if args.live {
        cmd.arg("--live");
    }
    cmd.args(&args.extra);
    let status = cmd
        .status()
        .with_context(|| format!("failed to run python3 {script}"))?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

fn run_bench_agent(args: BenchAgentArgs) -> Result<()> {
    use llmtrim::bench::agent::Condition;
    let mut tasks = load_agent_tasks(&args.tasks)?;
    if tasks.is_empty() {
        anyhow::bail!("no agent tasks found under {:?}", args.tasks);
    }
    if let Some(m) = &args.model {
        for t in &mut tasks {
            t.model = m.clone();
        }
    }
    let conditions: Vec<Condition> = args
        .conditions
        .split(',')
        .map(|s| Condition::parse(s.trim()))
        .collect();
    let table = std::fs::read_to_string(&args.pricing)
        .map(|s| bench::load_pricing(&s))
        .unwrap_or_default();

    if args.live {
        run_agent_live(&tasks, &conditions, &args, &table)
    } else {
        run_agent_dry(&tasks, &conditions, &args, &table)
    }
}

/// Collect agent task files: each path is a `.json` file, or a directory scanned for `*.json`.
fn load_agent_tasks(paths: &[PathBuf]) -> Result<Vec<llmtrim::bench::agent::AgentTask>> {
    use llmtrim::bench::agent::AgentTask;
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            for entry in
                std::fs::read_dir(p).with_context(|| format!("read dir {}", p.display()))?
            {
                let path = entry?.path();
                // `_`-prefixed files (e.g. _tools.json) are shared fragments, not tasks.
                let is_underscore = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('_'));
                if !is_underscore && path.extension().and_then(|e| e.to_str()) == Some("json") {
                    files.push(path);
                }
            }
        } else {
            files.push(p.clone());
        }
    }
    files.sort();
    // Tasks that omit `tools` share the tool catalog in `_tools.json` alongside them, so the
    // ten identical definitions live in one place. Cache per directory.
    let mut shared: std::collections::HashMap<PathBuf, serde_json::Value> = Default::default();
    let mut tasks = Vec::new();
    for f in files {
        let s = std::fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))?;
        let mut task =
            AgentTask::from_json(&s).with_context(|| format!("parse {}", f.display()))?;
        if task.tools.is_null() || task.tools.as_array().is_some_and(|a| a.is_empty()) {
            let dir = f.parent().unwrap_or_else(|| std::path::Path::new("."));
            let tools = match shared.get(dir) {
                Some(v) => v.clone(),
                None => {
                    let path = dir.join("_tools.json");
                    let v = std::fs::read_to_string(&path)
                        .ok()
                        .and_then(|t| serde_json::from_str(&t).ok())
                        .unwrap_or(serde_json::Value::Null);
                    shared.insert(dir.to_path_buf(), v.clone());
                    v
                }
            };
            task.tools = tools;
        }
        tasks.push(task);
    }
    Ok(tasks)
}

/// Synthetic, no-API run: drives each task/condition through the dry-run transport.
fn run_agent_dry(
    tasks: &[llmtrim::bench::agent::AgentTask],
    conditions: &[llmtrim::bench::agent::Condition],
    args: &BenchAgentArgs,
    table: &bench::PriceTable,
) -> Result<()> {
    use llmtrim::bench::agent::{OpenAiAgent, dry_run_transport, run_agent_loop};
    let provider = OpenAiAgent;
    let mut results = Vec::new();
    for task in tasks {
        let price = bench::resolve_pricing(table, &task.model);
        for cond in conditions {
            for _ in 0..args.repeats.max(1) {
                let mut send = dry_run_transport(args.tool_turns);
                results.push(run_agent_loop(&provider, task, cond, &price, &mut send)?);
            }
        }
    }
    print_agent_results(&results, true);
    write_agent_json(&results, &args.json_out)
}

/// Live run: same loop, but `send` calls the real model. Behind `--features live`.
#[cfg(feature = "live")]
fn run_agent_live(
    tasks: &[llmtrim::bench::agent::AgentTask],
    conditions: &[llmtrim::bench::agent::Condition],
    args: &BenchAgentArgs,
    table: &bench::PriceTable,
) -> Result<()> {
    use llmtrim::bench::agent::{OpenAiAgent, run_agent_loop};
    let api_key =
        dotenv_get("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set (env or a .env)")?;
    let model = llmtrim::quality::OpenRouterModel::new(api_key, ProviderKind::OpenAi)?;
    let provider = OpenAiAgent;
    let mut results = Vec::new();
    for task in tasks {
        let price = bench::resolve_pricing(table, &task.model);
        for cond in conditions {
            for _ in 0..args.repeats.max(1) {
                let mut send = |req: &str| model.send_raw(req);
                results.push(run_agent_loop(&provider, task, cond, &price, &mut send)?);
            }
        }
    }
    print_agent_results(&results, false);
    write_agent_json(&results, &args.json_out)
}

#[cfg(not(feature = "live"))]
fn run_agent_live(
    _tasks: &[llmtrim::bench::agent::AgentTask],
    _conditions: &[llmtrim::bench::agent::Condition],
    _args: &BenchAgentArgs,
    _table: &bench::PriceTable,
) -> Result<()> {
    anyhow::bail!(
        "--live needs the `live` feature: rebuild with `cargo run --features live -- bench agent --live …`"
    )
}

fn print_agent_results(results: &[llmtrim::bench::agent::AgentRunResult], dry: bool) {
    let color = ui::color_stdout();
    let mut t = ui::table(
        color,
        &[
            "task",
            "condition",
            "iters",
            "input",
            "cached",
            "output",
            "cost$",
            "done",
        ],
    );
    for r in results {
        t.add_row(vec![
            r.task_id.clone(),
            r.condition.clone(),
            r.iterations.to_string(),
            r.input_tokens.to_string(),
            r.cached_tokens.to_string(),
            r.output_tokens.to_string(),
            format!("{:.4}", r.cost_usd),
            if r.completed { "yes" } else { "no" }.to_string(),
        ]);
    }
    println!("{t}");
    if dry {
        println!(
            "(dry-run: synthetic usage, cached estimated by prefix overlap — no API calls; use --live for real numbers)"
        );
    }
}

fn write_agent_json(
    results: &[llmtrim::bench::agent::AgentRunResult],
    out: &Option<PathBuf>,
) -> Result<()> {
    if let Some(path) = out {
        let doc = bench::envelope::wrap(
            "agent-v1",
            serde_json::json!({ "runs": results.len() }),
            serde_json::to_value(results).context("serialize agent results")?,
        );
        let json = serde_json::to_string_pretty(&doc).context("serialize agent results")?;
        std::fs::write(path, json).with_context(|| format!("write {}", path.display()))?;
    }
    Ok(())
}

/// Offline per-stage input-token ablation: full vs each stage removed. A stage's
/// contribution = (after with it removed) − (after of full).
fn run_ablation(cases: &[BenchCase], config: &DenseConfig, preset: &str) -> Result<()> {
    let configs = bench::ablation_configs(config);
    let rows = bench::run_token_ablation(cases, &configs)?;
    let full_after = rows.first().map(|(_, _, a)| *a).unwrap_or(0) as f64;
    let color = ui::color_stdout();
    println!(
        "{}",
        ui::paint(
            color,
            Tone::Bold,
            &format!(
                "ablation — input tokens, offline, preset={preset}, {} cases",
                cases.len()
            )
        )
    );
    let mut t = ui::table(
        color,
        &["config", "before", "after", "saved", "stage saves"],
    );
    for (label, before, after) in &rows {
        let saved = ui::saved_pct(*before as f64, *after as f64);
        let contribution = if label == "full" {
            String::new()
        } else {
            format!("{:+.0} tok", *after as f64 - full_after)
        };
        t.add_row(vec![
            comfy_table::Cell::new(label),
            ui::right(before.to_string()),
            ui::right(after.to_string()),
            ui::right(ui::paint(color, Tone::Accent, &format!("{saved:.1}%"))),
            ui::right(contribution),
        ]);
    }
    for line in t.to_string().lines() {
        println!(" {line}");
    }
    Ok(())
}

/// Token-only smoke path: compress + measure input savings, no model calls.
/// Run metadata shared by the live and offline result envelopes. Records both the config
/// source (the `--config` path when it overrode the preset, otherwise the preset name) and
/// the resolved settings, so two runs with the same metadata are genuinely the same run (#15).
fn quality_meta(args: &BenchArgs, config: &DenseConfig) -> serde_json::Value {
    serde_json::json!({
        "preset": args.preset,
        "config_path": args.config.as_ref().map(|p| p.display().to_string()),
        "config_resolved": config,
        "model": args.model,
        "judge_model": args.judge_model,
        "route": args.route,
        "reasoning_effort": args.reasoning_effort,
        "corpus": args.corpus.display().to_string(),
    })
}

fn run_offline(
    cases: &[BenchCase],
    kind: ProviderKind,
    config: &DenseConfig,
    args: &BenchArgs,
) -> Result<()> {
    let (mut before, mut after) = (0usize, 0usize);
    let mut rows = Vec::with_capacity(cases.len());
    for c in cases {
        let r = llmtrim_core::compress_with_config(&c.request, Some(kind), config)?;
        before += r.input_tokens_before.0;
        after += r.input_tokens_after.0;
        rows.push(serde_json::json!({
            "name": c.name,
            "tokens_in_before": r.input_tokens_before.0,
            "tokens_in_after": r.input_tokens_after.0,
        }));
    }
    // Human-facing summary line. The benchkit harness measures via the Python binding, not
    // by parsing this stdout, so it is no longer a machine contract; keep it readable.
    println!(
        "offline: {} cases  input {before} -> {after} tok ({:.1}% saved)  preset={}",
        cases.len(),
        ui::saved_pct(before as f64, after as f64),
        args.preset,
    );
    // Unlike the live A/B, offline has no quality/cost axis — only input-token savings. It
    // gets its own schema so a reader never mistakes it for a full quality run.
    if let Some(path) = &args.json_out {
        let result = serde_json::json!({
            "n": cases.len(),
            "tokens_in_before": before,
            "tokens_in_after": after,
            "tokens_in_saved_pct": ui::saved_pct(before as f64, after as f64),
            "cases": rows,
        });
        let doc = bench::envelope::wrap("quality-offline-v1", quality_meta(args, config), result);
        let json = serde_json::to_string_pretty(&doc).context("serialize offline results")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
        eprintln!(
            "{}",
            ui::note(ui::color_stderr(), &format!("wrote {}", path.display()))
        );
    }
    Ok(())
}

/// Live A/B: send ORIGINAL and COMPRESSED requests, score both, price the round-trip,
/// print the frontier, and optionally dump per-case results as JSON.
#[cfg(feature = "live")]
fn run_live(
    cases: &[BenchCase],
    kind: ProviderKind,
    config: &DenseConfig,
    args: &BenchArgs,
) -> Result<()> {
    // The live A/B talks to OpenRouter's OpenAI-compatible endpoint, and answers are extracted
    // with the adapter for `kind`. For a non-OpenAI shape the request would post OpenAI-style
    // but be parsed with the wrong adapter → every case errors → an empty frontier with no
    // signal. Fail loudly instead, pointing at the paths that DO work for other shapes.
    if kind != ProviderKind::OpenAi {
        anyhow::bail!(
            "live A/B bench only supports --provider openai (the OpenRouter endpoint is \
             OpenAI-shaped; a {} request would be sent OpenAI-style and mis-parsed, scoring \
             nothing). Use --offline or --ablate to measure {}-shaped input-token savings, \
             or re-run with --provider openai.",
            args.provider,
            args.provider
        );
    }
    // Guard the obvious self-judging footgun: an LLM-judge corpus where the judge IS the model
    // under test grades its own answers (optimistic bias). The default judge differs by design.
    if args.judge_model == args.model
        && cases
            .iter()
            .any(|c| matches!(c.scorer, bench::Scorer::LlmJudge))
    {
        eprintln!(
            "{}",
            ui::warn(
                ui::color_stderr(),
                &format!(
                    "--judge-model == --model ({}) — the model is grading its own answers; \
                     pass a different --judge-model for an unbiased judge.",
                    args.model
                )
            )
        );
    }

    let table = std::fs::read_to_string(&args.pricing)
        .map(|s| bench::load_pricing(&s))
        .unwrap_or_default();
    let price = bench::resolve_pricing(&table, &args.model);
    let counter = llmtrim_core::tokenizer::counter_for(kind, Some(&args.model))?;
    let api_key = dotenv_get("OPENROUTER_API_KEY")
        .context("OPENROUTER_API_KEY not set (in env or a local .env)")?;
    let llm = llmtrim::quality::OpenRouterModel::new(api_key, kind)?;
    // BenchScorer covers pass@1 (runs tests), tool-call match, and the LLM judge
    // (reusing this endpoint), plus resource-free text scoring. The judge uses a fixed
    // DIFFERENT model from the one under test so it doesn't grade its own answers.
    let scorer = bench::BenchScorer {
        exec_timeout: 10,
        judge: Some(&llm),
        judge_model: args.judge_model.clone(),
    };

    let run = bench::run_ab(cases, config, &llm, counter.as_ref(), &scorer, price)?;
    let color = ui::color_stdout();
    let mut t = ui::table(color, &["case", "quality", "input", "output"]);
    for o in &run.outcomes {
        let q_tone = if o.quality_comp < o.quality_orig {
            Tone::Warn
        } else {
            Tone::Accent
        };
        t.add_row(vec![
            comfy_table::Cell::new(ui::truncate(&o.name, 24)),
            ui::right(ui::paint(
                color,
                q_tone,
                &format!("{:.2} → {:.2}", o.quality_orig, o.quality_comp),
            )),
            ui::right(format!("{} → {}", o.tokens_in_before, o.tokens_in_after)),
            ui::right(format!("{} → {}", o.tokens_out_orig, o.tokens_out_comp)),
        ]);
    }
    for line in t.to_string().lines() {
        println!(" {line}");
    }
    let f = bench::summarize(&run);
    let cache_note = if f.cache_busted {
        "cache busted (per-arm nonce; cache stage off)"
    } else {
        "cache stage under test"
    };
    let metric = |v: f64| ui::paint(color, Tone::Accent, &format!("{v:.1}%"));
    let lines = vec![
        ui::paint(
            color,
            Tone::Dim,
            &format!(
                "{} · model {} · judge {}",
                args.corpus.display(),
                args.model,
                args.judge_model
            ),
        ),
        String::new(),
        format!("input saved   {}", metric(f.tokens_in_saved_pct)),
        format!("output saved  {}", metric(f.tokens_out_saved_pct)),
        format!("cost saved    {}", metric(f.cost_saved_pct)),
        format!("cache used    {:.1}%  ({cache_note})", f.cache_used_pct),
        format!(
            "quality       {} → {}  (retention {:+.1}pp, paired 95%CI ±{:.1})",
            ui::paint(
                color,
                Tone::Bold,
                &format!("{:.1}%", f.quality_orig.mean * 100.0)
            ),
            ui::paint(
                color,
                Tone::Bold,
                &format!("{:.1}%", f.quality_comp.mean * 100.0)
            ),
            f.retention_pp,
            f.retention_ci95_pp,
        ),
        format!(
            "cases         {} scored, {} failed (compressed 4xx → 0), {} skipped (transient)",
            f.n, f.failed, f.skipped
        ),
    ];
    print!(
        "\n{}",
        ui::panel(
            color,
            &format!("bench · {} · n={}", args.preset, f.n),
            &lines
        )
    );
    if let Some(path) = &args.json_out {
        let rows: Vec<_> = run
            .outcomes
            .iter()
            .map(|o| {
                serde_json::json!({
                    "name": o.name,
                    "quality_orig": o.quality_orig, "quality_comp": o.quality_comp,
                    "tokens_in_before": o.tokens_in_before, "tokens_in_after": o.tokens_in_after,
                    "tokens_out_orig": o.tokens_out_orig, "tokens_out_comp": o.tokens_out_comp,
                    "cached_in_orig": o.cached_in_orig, "cached_in_comp": o.cached_in_comp,
                    "cost_orig": o.cost_orig, "cost_comp": o.cost_comp,
                })
            })
            .collect();
        let meta = quality_meta(args, config);
        let result = serde_json::json!({
            "n": f.n,
            "failed": f.failed,
            "skipped": f.skipped,
            "cache_busted": f.cache_busted,
            "tokens_in_saved_pct": f.tokens_in_saved_pct,
            "tokens_out_saved_pct": f.tokens_out_saved_pct,
            "cost_saved_pct": f.cost_saved_pct,
            "cache_used_pct": f.cache_used_pct,
            "quality_orig": f.quality_orig.mean, "quality_comp": f.quality_comp.mean,
            "retention_pp": f.retention_pp,
            "retention_ci95_pp": f.retention_ci95_pp,
            "cases": rows,
        });
        let doc = bench::envelope::wrap("quality-v1", meta, result);
        let json = serde_json::to_string_pretty(&doc).context("serialize quality results")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
        eprintln!(
            "{}",
            ui::note(ui::color_stderr(), &format!("wrote {}", path.display()))
        );
    }
    Ok(())
}

/// Live A/B needs the `live` feature (async-openai + tokio). Built without it, the
/// command explains how to enable it; `--offline`/`--ablate` still measure tokens.
#[cfg(not(feature = "live"))]
fn run_live(
    _cases: &[BenchCase],
    _kind: ProviderKind,
    _config: &DenseConfig,
    _args: &BenchArgs,
) -> Result<()> {
    anyhow::bail!(
        "live A/B bench requires the `live` feature: rebuild with \
         `cargo run --features live -- bench …`. Use `--offline` (or `--ablate`) for \
         token-only measurement, which needs no extra feature."
    )
}

fn period_flag(daily: bool, weekly: bool, monthly: bool) -> Option<Period> {
    if daily {
        Some(Period::Day)
    } else if weekly {
        Some(Period::Week)
    } else if monthly {
        Some(Period::Month)
    } else {
        None
    }
}

/// Daemon + wiring state for the dashboard's health chain: pidfile liveness, a real TCP
/// probe of the port, the env-wired port, autostart, version skew, and the age of the
/// last recorded request (from `summary`) — everything `status` needs to say *why*
/// something is broken, not just that it is.
fn daemon_view(summary: &llmtrim::tracking::Summary) -> monitor::DaemonView {
    use llmtrim::daemon;
    let ca_present = matches!(llmtrim::serve::ca_cert_path(), Ok(p) if p.exists());
    let env_port = llmtrim::setup::configured_port();
    let autostart = llmtrim::autostart::is_enabled();
    let log_path = daemon::logfile().ok().map(|p| p.display().to_string());
    let last_request = summary.last_ts.as_deref().and_then(daemon::human_age);
    let binary_version = env!("CARGO_PKG_VERSION").to_string();
    match daemon::running() {
        Some(s) => monitor::DaemonView {
            running: true,
            pid: s.pid,
            port: s.port,
            uptime: daemon::human_uptime(daemon::uptime_secs(s.started_at)),
            uptime_secs: daemon::uptime_secs(s.started_at),
            ca_present,
            port_accepting: daemon::probe_port(s.port),
            env_port,
            autostart,
            restarts: s.restarts,
            version: s.version,
            binary_version,
            log_path,
            last_request,
        },
        // No pidfile (never recorded, or lost to a transient I/O failure like a full disk).
        // Don't trust the absent file alone — a proxy may still be live on the wired port.
        // Probe it so `status` stays truthful instead of crying "stopped" while LLM calls
        // keep flowing. A live probe with no pidfile is marked by `pid: 0` (no recorded pid),
        // which the header renders as "running (unmanaged)".
        None => match env_port.filter(|&p| daemon::probe_port(p)) {
            Some(p) => monitor::DaemonView {
                running: true,
                pid: 0,
                port: p,
                uptime: String::new(),
                uptime_secs: 0,
                ca_present,
                port_accepting: true,
                env_port,
                autostart,
                restarts: 0,
                version: None,
                binary_version,
                log_path,
                last_request,
            },
            None => monitor::DaemonView {
                running: false,
                pid: 0,
                port: 0,
                uptime: String::new(),
                uptime_secs: 0,
                ca_present,
                port_accepting: false,
                env_port,
                autostart,
                restarts: 0,
                version: None,
                binary_version,
                log_path,
                last_request,
            },
        },
    }
}

fn render_snapshot(
    tracker: &Tracker,
    color: bool,
    cache_included: bool,
    split_marker: bool,
) -> Result<(String, monitor::Health)> {
    let summary = tracker.summary()?;
    let models = monitor::model_views(tracker)?;
    let daemon = daemon_view(&summary);
    let health = monitor::health(&daemon);
    // Last (up to) 7 days of input tokens saved, oldest→newest, for the 7-DAY TREND sparkline.
    let trend: Vec<i64> = tracker
        .by_period(Period::Day)
        .unwrap_or_default()
        .iter()
        .rev()
        .take(7)
        .rev()
        .map(|r| (r.input_before - r.input_after).max(0))
        .collect();
    // Frozen breakdown money for hero — same source as Overview (never live cost_estimate).
    let (hero_saved, today_saved) = monitor::hero_money(tracker);
    let mut out = monitor::snapshot(
        color,
        Some(&daemon),
        &summary,
        &models,
        hero_saved,
        today_saved,
        &trend,
        cache_included,
        split_marker,
    );
    // Passive, cached (≤24h), opt-out update notice (LLMTRIM_NO_UPDATE_CHECK to disable).
    if let Some(v) = llmtrim::update::check(false) {
        out.push_str(&format!(
            "\n  {} llmtrim v{v} available — run llmtrim update\n",
            ui::paint(color, Tone::Accent, "↑")
        ));
    }
    Ok((out, health))
}

/// Build the structured data the breakdown TUI's native Overview renders. Re-uses the same
/// ledger view-models as `render_snapshot`, mapped into plain-language fields.
#[cfg(feature = "breakdown")]
fn overview_data(tracker: &Tracker) -> llmtrim::breakdown::app::OverviewData {
    use llmtrim::breakdown::app::{StatusKind, StatusLine};
    // The numeric derivation lives in monitor::overview_data (shared with the SVG exporter so the
    // two can't drift); here we only supply the live, daemon-derived health line.
    monitor::overview_data(tracker, |summary, has_traffic| {
        let daemon = daemon_view(summary);
        // Spoken health: only the dangerous / actionable states reveal a fix command.
        let version_stale = daemon
            .version
            .as_deref()
            .is_some_and(|v| v != daemon.binary_version);
        if daemon.running && version_stale {
            StatusLine {
                kind: StatusKind::Stale,
                text: "llmtrim is on, but running an older version — restart to apply the update"
                    .into(),
                // `llmtrim start --force` restarts the daemon onto the new binary (there is no
                // `restart` subcommand). The `u` key runs this for you.
                fix: Some("llmtrim start --force".into()),
                uninstall: None,
            }
        } else if daemon.running {
            let text = match daemon.last_request {
                Some(ago) => format!("llmtrim is on and working · last request {ago}"),
                None => "llmtrim is on and ready · waiting for your first request".into(),
            };
            StatusLine {
                kind: if has_traffic {
                    StatusKind::Working
                } else {
                    StatusKind::Ready
                },
                text,
                fix: None,
                uninstall: None,
            }
        } else if daemon.env_port.is_some() {
            StatusLine {
                kind: StatusKind::Off,
                text: "llmtrim is OFF — your AI tools can't reach the API right now".into(),
                fix: Some("llmtrim start".into()),
                uninstall: Some("llmtrim uninstall".into()),
            }
        } else {
            // Not running and not wired: never set up (or cleanly removed) — `start` alone won't
            // route traffic without the env, so point at `setup`.
            StatusLine {
                kind: StatusKind::Degraded,
                text: "llmtrim is not set up — your AI traffic isn't routed through it".into(),
                fix: Some("llmtrim setup".into()),
                uninstall: None,
            }
        }
    })
}

/// Merge the per-source breakdown (every session + corpus-wide per-source cost) into a
/// `status --json` string. No-op unless `--breakdown` is given (the section aggregates
/// the full ledger history, which is expensive on a large DB) and the breakdown feature
/// is built in with data present; the `breakdown` key is purely additive when emitted.
fn merge_breakdown_json(s: String, include: bool) -> String {
    if !include {
        return s;
    }
    #[cfg(feature = "breakdown")]
    if let Some(bd) = llmtrim::breakdown::export::breakdown_json() {
        // A parse failure here means a regression upstream in the status JSON — surface it
        // rather than silently dropping the breakdown section.
        let mut v: serde_json::Value = match serde_json::from_str(&s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("llmtrim: could not parse status json for breakdown merge: {e}");
                return s;
            }
        };
        if let Some(obj) = v.as_object_mut() {
            obj.insert("breakdown".to_string(), bd);
            match serde_json::to_string_pretty(&v) {
                Ok(out) => return out,
                Err(e) => eprintln!("llmtrim: could not merge breakdown into json: {e}"),
            }
        } else {
            eprintln!("llmtrim: status json is not an object; skipped breakdown merge");
        }
    }
    s
}

/// Append the per-session breakdown table after the time-series CSV (feature-gated).
fn print_sessions_csv() {
    #[cfg(feature = "breakdown")]
    if let Some(sc) = llmtrim::breakdown::export::sessions_csv() {
        println!();
        print!("{sc}");
    }
}

/// Append the "top sources by cost" block after a text period report (feature-gated).
fn print_top_sources() {
    #[cfg(feature = "breakdown")]
    if let Ok(block) = llmtrim::breakdown::export::top_sources_report(ui::color_stdout(), 10) {
        print!("{block}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_monitor(
    interval: u64,
    daily: bool,
    weekly: bool,
    monthly: bool,
    json: bool,
    breakdown: bool,
    csv: bool,
    quiet: bool,
) -> Result<()> {
    // `interval` only drives the breakdown TUI's refresh; without that feature there is no
    // live view to refresh, so it is unused.
    #[cfg(not(feature = "breakdown"))]
    let _ = interval;
    let tracker = Tracker::open().context("failed to open savings ledger")?;

    // Health-only mode: one word + the health exit code, for scripts and prompts
    // (`llmtrim status -q && …`).
    if quiet {
        let summary = tracker.summary()?;
        let health = monitor::health(&daemon_view(&summary));
        println!("{}", health.label());
        std::process::exit(health.exit_code());
    }

    // Time-series report / export. Exports always exit 0: they are data queries, not
    // health checks — scripts read `daemon.health` from the JSON instead.
    if let Some(period) = period_flag(daily, weekly, monthly) {
        let rows = tracker.by_period(period)?;
        if csv {
            print!("{}", monitor::export_csv(&rows));
            print_sessions_csv();
        } else if json {
            let s = tracker.summary()?;
            let models = monitor::model_views(&tracker)?;
            let daemon = daemon_view(&s);
            let out = monitor::export_json(
                &s,
                &models,
                monitor::monitor_cost(&tracker).as_ref(),
                &rows,
                Some(&daemon),
                monitor::money_json(&tracker),
            );
            println!("{}", merge_breakdown_json(out, breakdown));
        } else {
            print!(
                "{}",
                monitor::period_report(ui::color_stdout(), period.label(), &rows)
            );
            print_top_sources();
        }
        return Ok(());
    }

    // Whole-snapshot export (no period selected).
    if json || csv {
        if csv {
            let rows = tracker.by_period(Period::Day)?;
            print!("{}", monitor::export_csv(&rows));
            print_sessions_csv();
        } else {
            let daemon = daemon_view(&tracker.summary()?);
            println!(
                "{}",
                merge_breakdown_json(monitor::stats_json(&tracker, Some(&daemon))?, breakdown)
            );
        }
        return Ok(());
    }

    // On a TTY, the interactive view opens the cost-breakdown TUI:
    // a tabbed Overview / Sessions / Detail explorer. Piped output and the export modes
    // above keep the plain snapshot, so scripts and `| less` are unaffected.
    #[cfg(feature = "breakdown")]
    {
        use std::io::IsTerminal as _;
        if std::io::stdout().is_terminal() {
            // The TUI uses its own read-only connections (below); release the outer read-write
            // handle so we don't hold an idle writer open for the whole interactive session.
            drop(tracker);
            // Open the ledger ONCE and reuse it every refresh (WAL lets these long-lived
            // readers see the daemon's writes). The snapshot closure runs on a background
            // thread, so it owns its own connections and the UI never touches SQLite on a timer.
            let tracker = llmtrim::tracking::db_path()
                .ok()
                .as_deref()
                .and_then(|p| Tracker::open_reader_at(p).ok());
            let sessions_db = llmtrim::breakdown::db::BreakdownDb::open().ok();
            let snapshot = move || {
                let ov = tracker.as_ref().map(overview_data).unwrap_or_default();
                let rows = sessions_db
                    .as_ref()
                    .and_then(|d| d.sessions().ok())
                    .unwrap_or_default();
                (ov, rows)
            };
            // The TUI may queue a follow-up command (d = doctor/repair, u = update); run it on
            // the normal screen after the alt-screen tears down.
            use llmtrim::breakdown::app::PostAction;
            // Heal integrations from a version bump before the TUI opens (quiet).
            let _ = llmtrim::ensure::maybe_auto();
            match llmtrim::breakdown::app::run(interval.max(2), snapshot)? {
                PostAction::Doctor => {
                    let report = llmtrim::doctor::gather();
                    print!("{}", llmtrim::doctor::render(ui::color_stdout(), &report));
                }
                PostAction::Update => llmtrim::update::run()?,
                PostAction::Restart => {
                    // Stale daemon: restart it onto the freshly-installed binary via the shared
                    // helper (`start --force`, the documented "pick up a new binary" path).
                    // Surface a failed restart (port held, permission, ...) instead of exiting 0.
                    if let Err(e) = llmtrim::update::restart_daemon(ui::color_stdout()) {
                        eprintln!("{e:#}");
                        std::process::exit(1);
                    }
                }
                PostAction::None => {}
            }
            return Ok(());
        }
    }

    let (out, health) = render_snapshot(&tracker, ui::color_stdout(), false, false)?;
    print!("{out}");
    // Propagate health as the exit code (0 healthy / 1 stopped / 2 degraded) so
    // `llmtrim status && …` means "llmtrim is actually working".
    if health != monitor::Health::Healthy {
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
        std::process::exit(health.exit_code());
    }
    Ok(())
}

/// Per-provider `(cache_read, cache_write)` price multipliers vs the list input rate:
/// Anthropic bills cache reads at 10% and cache writes at 125%; OpenAI bills cached
/// prompt tokens at 50% (no write surcharge); Gemini implicit caching discounts cached
/// tokens 75%. Unknown providers assume no discount, which collapses the net figure to
/// the list-rate one instead of inventing a discount.
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    // `--watch` is a deprecated no-op, but it must still parse: removing it outright would
    // break existing `llmtrim status --watch` scripts and aliases.
    #[test]
    fn status_still_accepts_deprecated_watch_flag() {
        let cli = Cli::try_parse_from(["llmtrim", "status", "--watch"])
            .expect("status --watch must still parse for backward compatibility");
        match cli.command {
            Commands::Monitor { watch, .. } => assert!(watch),
            _ => panic!("expected status to parse as the Monitor command"),
        }
    }

    // The heavy per-source `breakdown` object is opt-in: without `--breakdown`, `status --json`
    // must not run the full-history aggregation, so the merge is a no-op that returns the input
    // untouched (the common consumers — health badge, savings % — never read that section).
    #[test]
    fn breakdown_section_is_omitted_unless_requested() {
        let base = r#"{"daemon":{"running":true}}"#.to_string();
        assert_eq!(merge_breakdown_json(base.clone(), false), base);
    }

    // `_alive` is baked verbatim into every user's shell profile (setup.rs `env_block`), so its
    // name is a forever-contract: a rename silently breaks proxy wiring for every new shell. Pin
    // it here so CI catches an accidental rename, and confirm it stays hidden from help.
    #[test]
    fn alive_subcommand_name_is_stable_and_hidden() {
        let cli = Cli::try_parse_from(["llmtrim", "_alive"]).expect("`_alive` must parse");
        assert!(matches!(cli.command, Commands::Alive));
        let help = {
            use clap::CommandFactory;
            let mut out = Vec::new();
            Cli::command().write_long_help(&mut out).unwrap();
            String::from_utf8(out).unwrap()
        };
        assert!(
            !help.contains("_alive"),
            "`_alive` must stay hidden from help"
        );
    }

    // The flag only makes sense with `--json`; clap must reject it on its own.
    #[test]
    fn breakdown_flag_requires_json() {
        assert!(
            Cli::try_parse_from(["llmtrim", "status", "--breakdown"]).is_err(),
            "--breakdown without --json must be rejected"
        );
        Cli::try_parse_from(["llmtrim", "status", "--json", "--breakdown"])
            .expect("status --json --breakdown must parse");
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("llmtrim_bench_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn load_agent_tasks_injects_shared_tools_and_skips_underscore_files() {
        let dir = scratch_dir("dedup");
        std::fs::write(
            dir.join("_tools.json"),
            r#"[{"type":"function","function":{"name":"grep"}}]"#,
        )
        .unwrap();
        // No `tools` key: should inherit the shared catalog.
        std::fs::write(
            dir.join("a.json"),
            r#"{"id":"a","model":"m","system":"s","user":"u"}"#,
        )
        .unwrap();
        // Own `tools`: must be preserved, not overwritten.
        std::fs::write(
            dir.join("b.json"),
            r#"{"id":"b","model":"m","system":"s","user":"u","tools":[{"type":"function","function":{"name":"own"}}]}"#,
        )
        .unwrap();

        let tasks = load_agent_tasks(std::slice::from_ref(&dir)).unwrap();
        assert_eq!(tasks.len(), 2, "_tools.json is a fragment, not a task");
        let a = tasks.iter().find(|t| t.id == "a").unwrap();
        let b = tasks.iter().find(|t| t.id == "b").unwrap();
        assert_eq!(
            a.tools.pointer("/0/function/name").unwrap(),
            "grep",
            "shared catalog injected when tools omitted"
        );
        assert_eq!(
            b.tools.pointer("/0/function/name").unwrap(),
            "own",
            "task's own tools left intact"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn suite_matrix_covers_the_documented_corpora() {
        assert_eq!(
            SUITE_MATRIX.len(),
            11,
            "eight workload corpora + three named benchmarks"
        );
        let cnn = SUITE_MATRIX.iter().find(|(c, _, _)| *c == "cnn").unwrap();
        assert_eq!(cnn.2, 8, "cnn runs fewer cases (long docs)");
        // Every preset name must resolve, or a live suite run dies mid-matrix.
        for (corpus, preset, n) in SUITE_MATRIX {
            assert!(*n > 0, "{corpus}: case count must be positive");
            assert!(
                DenseConfig::preset(preset).is_some(),
                "{corpus}: unknown preset '{preset}'"
            );
        }
    }

    #[test]
    fn prepare_request_pins_model_route_and_reasoning() {
        let base = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let out = prepare_request(base, "openai/gpt-oss-20b", "wandb/fp4", "medium");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["model"], "openai/gpt-oss-20b");
        assert_eq!(v["provider"]["order"], serde_json::json!(["wandb"]));
        assert_eq!(v["provider"]["quantizations"], serde_json::json!(["fp4"]));
        assert_eq!(v["reasoning"]["effort"], "medium");
    }

    #[test]
    fn prepare_request_omits_reasoning_when_effort_is_empty() {
        let base = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let out = prepare_request(base, "openai/gpt-4o-mini", "", "");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("reasoning").is_none(), "no reasoning field: {v}");
        assert!(v.get("provider").is_none(), "no provider field: {v}");
    }

    const CORPUS_LINE: &str = r#"{"context":"The cat sat on the red mat.","question":"What color was the mat?","gold":"red","scorer":"token_f1"}"#;

    fn offline_args(corpus: PathBuf, json_out: Option<PathBuf>) -> BenchArgs {
        BenchArgs {
            corpus,
            provider: "openai".to_string(),
            preset: "rag".to_string(),
            model: "openai/gpt-4o-mini".to_string(),
            judge_model: "openai/gpt-4o-mini".to_string(),
            route: String::new(),
            reasoning_effort: String::new(),
            n: Some(1),
            config: None,
            pricing: PathBuf::new(), // unused: offline never reads pricing
            offline: true,
            ablate: false,
            json_out,
        }
    }

    #[test]
    fn offline_quality_writes_a_valid_enveloped_file() {
        let dir = scratch_dir("offline");
        let corpus = dir.join("c.jsonl");
        std::fs::write(&corpus, format!("{CORPUS_LINE}\n")).unwrap();
        let out = dir.join("out.json");

        run_bench(offline_args(corpus, Some(out.clone()))).unwrap();

        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(doc["schema"], "llmtrim-bench/quality-offline-v1");
        assert_eq!(doc["meta"]["preset"], "rag");
        assert_eq!(doc["result"]["n"], 1);
        assert_eq!(doc["result"]["cases"].as_array().unwrap().len(), 1);
        // The Python readers flatten {meta, result}; the keys they read must be present.
        assert!(doc["result"]["tokens_in_before"].is_number());
        assert!(doc["meta"]["model"].is_string());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn suite_runs_offline_and_skips_missing_corpora() {
        let data = scratch_dir("suite_data");
        let out = scratch_dir("suite_out");
        // Provide just one of the eight matrix corpora; the rest must be skipped, not fatal.
        std::fs::write(data.join("gsm8k.jsonl"), format!("{CORPUS_LINE}\n")).unwrap();

        let args = SuiteArgs {
            data_dir: data.clone(),
            out: out.clone(),
            model: "openai/gpt-4o-mini".to_string(),
            judge_model: "openai/gpt-4o-mini".to_string(),
            route: String::new(),
            pricing: PathBuf::new(), // unused: offline never reads pricing
            n: Some(1),
            offline: true,
        };
        run_bench_suite(args).expect("missing corpora must not abort the matrix");

        assert!(
            out.join("gsm8k.json").is_file(),
            "present corpus produced a result"
        );
        assert!(
            !out.join("cnn.json").exists(),
            "absent corpus left no result"
        );
        std::fs::remove_dir_all(&data).unwrap();
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn latency_runs_on_the_builtin_fixture() {
        let r = run_bench_latency(LatencyArgs {
            request: None,
            provider: None,
            iterations: 3,
        });
        assert!(
            r.is_ok(),
            "latency on the built-in fixture should not error: {r:?}"
        );
    }

    #[test]
    fn compare_rejects_an_unknown_tool() {
        let err = run_bench_compare(CompareArgs {
            tool: "nope".to_string(),
            live: false,
            extra: vec![],
        })
        .unwrap_err();
        assert!(err.to_string().contains("unknown comparator"), "{err}");
    }
}
