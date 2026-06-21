//! llmtrim CLI.
//!
//! Network-free surface: `compress` reads a provider request body on stdin and writes the
//! compressed body to stdout; `send` adds the network round-trip; `status` shows
//! savings from the SQLite ledger; `serve`/`setup` run the MITM interceptor. The pure
//! transform core lives in `lib.rs`.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
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
  setup      Set everything up and start saving (CA, env, autostart, daemon)
  status     Show the savings dashboard + interceptor health  [aliases: monitor, gain]
  wrap       Launch an agent (claude, codex, …) routed through the interceptor

Daemon:
  start      Start the background interceptor (no-op if already running)
  stop       Stop the background interceptor
  autostart  Run the interceptor at login (--off to disable)

When something's wrong:
  doctor     Check the install end-to-end and explain anything broken
  update     Update llmtrim to the latest release
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
    /// HKCU\Environment on Windows), enables run-at-login, and starts the
    /// interceptor. Idempotent — re-running reuses the same port and won't restart a
    /// healthy daemon. No IDE settings are touched, no sudo.
    Setup {
        /// Interceptor port. Omit to auto-select a free port starting at 43117.
        #[arg(long)]
        port: Option<u16>,
        /// Restart the daemon even if a healthy one is already on the chosen port (e.g. to pick
        /// up a new binary). By default setup leaves a healthy same-port daemon running.
        #[arg(long)]
        force: bool,
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
    /// Update llmtrim to the latest release
    ///
    /// Channel-aware: a binary install self-updates via the installer; cargo and
    /// Homebrew installs print their package manager's command.
    Update,
    /// Show the savings dashboard + interceptor health
    ///
    /// Savings from the ledger, plus the health chain (daemon → port → env → CA →
    /// traffic). Default: a snapshot, exiting 0 healthy / 1 stopped / 2 degraded;
    /// `-q` for the health word only; `--watch` for a live view;
    /// `--daily/--weekly/--monthly` for time-series; `--json/--csv` to export.
    #[command(name = "status", visible_aliases = ["monitor", "gain"])]
    Monitor {
        /// Live refreshing dashboard (Ctrl-C to exit).
        #[arg(long)]
        watch: bool,
        /// Refresh interval for `--watch`, in seconds.
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
        /// Emit CSV time-series instead of the dashboard.
        #[arg(long)]
        csv: bool,
        /// Health only: print healthy|degraded|stopped and exit 0/2/1 (script-friendly).
        #[arg(long, short, conflicts_with_all = ["watch", "daily", "weekly", "monthly", "json", "csv"])]
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
    /// Check the install end-to-end and explain anything broken
    ///
    /// Read-only diagnosis: binary, daemon, port, env wiring (persisted + this shell),
    /// CA, autostart, ledger, version skew. Exits non-zero if any check fails.
    Doctor,
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
    },
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

fn main() {
    if let Err(e) = run() {
        eprint!("{}", ui::render_error(ui::color_stderr(), &e));
        std::process::exit(1);
    }
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
                llmtrim::serve::run_supervised(port, force)?;
            } else {
                llmtrim::serve::run(port, force)?;
            }
        }
        Commands::Setup { port, force } => llmtrim::setup::run(port, force)?,
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
        }
        Commands::Wrap { args } => llmtrim::wrap::run(args)?,
        Commands::Stop => match llmtrim::daemon::stop()? {
            Some(pid) => {
                println!(
                    "{}",
                    ui::ok(
                        ui::color_stdout(),
                        &format!("Stopped interceptor (pid {pid}).")
                    )
                );
                if llmtrim::setup::profile_has_block() {
                    eprintln!(
                        "{}",
                        ui::warn(
                            ui::color_stderr(),
                            "HTTPS_PROXY still points at llmtrim in your environment — \
                             new HTTPS to LLM hosts will fail until you start it again \
                             (llmtrim start) or run llmtrim uninstall."
                        )
                    );
                }
            }
            None => println!(
                "{}",
                ui::note(ui::color_stdout(), "No interceptor daemon was running.")
            ),
        },
        Commands::Update => llmtrim::update::run()?,
        Commands::Mcp { action } => match action {
            None => llmtrim::mcp::run()?,
            Some(McpAction::Install { print, force }) => llmtrim::mcp::install(print, force)?,
        },
        Commands::Monitor {
            watch,
            interval,
            daily,
            weekly,
            monthly,
            json,
            csv,
            quiet,
        } => run_monitor(watch, interval, daily, weekly, monthly, json, csv, quiet)?,
        Commands::Doctor => {
            let report = llmtrim::doctor::gather();
            print!("{}", llmtrim::doctor::render(ui::color_stdout(), &report));
            if report.problems > 0 {
                std::process::exit(1);
            }
        }
        Commands::Autostart { off, port } => {
            let color = ui::color_stdout();
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
/// provider. Both fields are top-level passthrough, so compression preserves them and
/// the original and compressed sends route identically.
fn prepare_request(request_json: &str, model: &str, route: &str) -> String {
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
        c.request = prepare_request(&c.request, &args.model, &args.route);
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
    let cost = monitor::monitor_cost(tracker);
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
    let mut out = monitor::snapshot(
        color,
        Some(&daemon),
        &summary,
        &models,
        cost.as_ref(),
        monitor::today_saved_usd(tracker),
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

/// Restores the normal screen buffer when the watch loop unwinds (error/broken pipe).
/// The Ctrl-C path can't rely on Drop (`process::exit` skips destructors), so the
/// signal handler in [`run_watch`] writes the same escape itself.
struct AltScreenGuard;

impl Drop for AltScreenGuard {
    fn drop(&mut self) {
        let mut out = std::io::stdout();
        let _ = out.write_all(b"\x1b[?1049l");
        let _ = out.flush();
    }
}

/// Live dashboard. On a TTY it runs in the alternate screen buffer (the user's
/// scrollback survives, like htop/less), repaints in place — home + per-line
/// clear-to-EOL instead of a full-screen wipe, wrapped in synchronized-output marks
/// (DEC 2026) so capable terminals commit each frame atomically: no flicker, no
/// infinite scroll. Piped/redirected output keeps plain appended frames. Exits on
/// Ctrl-C, restoring the normal screen.
fn run_watch(tracker: &Tracker, interval: u64) -> Result<()> {
    let color = ui::color_stdout();
    // Screen-control escapes are for interactive terminals only — piped/redirected
    // watch output gets appended frames instead of raw escapes.
    let tty = ui::stdout_is_tty();
    let _alt = if tty {
        let mut out = std::io::stdout();
        let _ = out.write_all(b"\x1b[?1049h\x1b[H");
        let _ = out.flush();
        // Default SIGINT would kill the process inside the alternate screen, leaving
        // the terminal stuck in it — restore first, then exit.
        let _ = ctrlc::set_handler(|| {
            let mut out = std::io::stdout();
            let _ = out.write_all(b"\x1b[?1049l");
            let _ = out.flush();
            std::process::exit(0);
        });
        Some(AltScreenGuard)
    } else {
        None
    };
    let mut prev: Option<i64> = None;
    let mut frame_n: usize = 0;
    loop {
        let summary = tracker.summary()?;
        let mut body = render_snapshot(tracker, color, false, false)?.0;
        // Live throughput this interval, folded into the status bar below — only when traffic
        // actually flowed (a perpetual "+0/s" on an idle proxy reads like fake data), and
        // humanised to match the rest of the dashboard.
        let rate_seg = match prev {
            Some(p) if (summary.saved() - p) as f64 / interval as f64 >= 0.5 => {
                let rate = (summary.saved() - p) as f64 / interval as f64;
                format!(" · +{} tok/s saved", ui::human(rate as i64))
            }
            _ => String::new(),
        };
        prev = Some(summary.saved());

        // Live status-bar footer. On a TTY: a spinner that advances every refresh (proves the
        // view is live even when the numbers don't move), a ticking clock, the live rate when
        // traffic flowed, and a right-aligned quit keycap. Piped output keeps a plain one-liner.
        let cols = if tty {
            terminal_size::terminal_size()
                .map(|(w, _)| w.0 as usize)
                .unwrap_or(80)
        } else {
            0
        };
        if tty {
            const SPIN: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
            // Workload liveness — "is traffic flowing?" — is more useful here than a wall
            // clock (the spinner already proves the UI is live). Idle shows how long ago the
            // last request was, so a stalled workload is obvious.
            let traffic = summary
                .last_ts
                .as_deref()
                .and_then(llmtrim::daemon::human_age)
                .map(|age| format!("last request {age}"))
                .unwrap_or_else(|| "no requests yet".to_string());
            let left = format!(
                "{}  {}",
                ui::paint(color, Tone::Accent, SPIN[frame_n % SPIN.len()]),
                ui::paint(color, Tone::Dim, &format!("{traffic}{rate_seg}")),
            );
            let right = format!(
                "{} {}",
                ui::paint(color, Tone::Bold, "Ctrl-C"),
                ui::paint(color, Tone::Dim, "exit"),
            );
            let used = 1 + ui::visible_width(&left) + ui::visible_width(&right);
            let pad = " ".repeat(cols.saturating_sub(used).max(1));
            body.push_str(&format!(" {left}{pad}{right}\n"));
        } else {
            body.push_str(&format!(
                "  refreshing every {interval}s{rate_seg} · Ctrl-C to exit\n"
            ));
        }
        frame_n += 1;

        let frame = if tty {
            // Each logical line must occupy exactly one screen row, or the home + per-line
            // `\x1b[K` repaint drifts when a line soft-wraps. Truncate to the terminal width
            // (ANSI-aware), then sync-begin + home, clear each line's tail (`\x1b[K`), clear
            // below the frame (`\x1b[0J`), sync-end. The full untruncated text stays in the
            // one-shot `status` (no repaint there, so wrapping is harmless).
            let painted: String = body
                .lines()
                .map(|l| format!("{}\x1b[K\n", ui::truncate_visible(l, cols)))
                .collect();
            format!("\x1b[?2026h\x1b[H{painted}\x1b[0J\x1b[?2026l")
        } else {
            body
        };
        // Write, don't print!: a piped reader that exits (e.g. `| head`) closes the
        // pipe, and print! would panic on the broken pipe — exit cleanly instead.
        let mut stdout = std::io::stdout();
        if stdout.write_all(frame.as_bytes()).is_err() {
            return Ok(());
        }
        stdout.flush().ok();
        std::thread::sleep(std::time::Duration::from_secs(interval));
    }
}

/// Merge the per-source breakdown (every session + corpus-wide per-source cost) into a
/// `status --json` string. No-op when the breakdown feature is off or no data exists yet,
/// so the JSON shape is a superset of the prior one (additive `breakdown` key).
fn merge_breakdown_json(s: String) -> String {
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
    watch: bool,
    interval: u64,
    daily: bool,
    weekly: bool,
    monthly: bool,
    json: bool,
    csv: bool,
    quiet: bool,
) -> Result<()> {
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
            );
            println!("{}", merge_breakdown_json(out));
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
                merge_breakdown_json(monitor::stats_json(&tracker, Some(&daemon))?)
            );
        }
        return Ok(());
    }

    // On a TTY, the interactive view (watch or default) opens the cost-breakdown TUI:
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
            match llmtrim::breakdown::app::run(interval.max(2), snapshot)? {
                PostAction::Doctor => {
                    let report = llmtrim::doctor::gather();
                    print!("{}", llmtrim::doctor::render(ui::color_stdout(), &report));
                }
                PostAction::Update => llmtrim::update::run()?,
                PostAction::Restart => {
                    // Stale daemon: restart it onto the freshly-installed binary. `start --force`
                    // is the documented "pick up a new binary after an update" path.
                    let exe = std::env::current_exe().context("locate llmtrim binary")?;
                    std::process::Command::new(exe)
                        .args(["start", "--force"])
                        .status()
                        .context("run llmtrim start --force")?;
                }
                PostAction::None => {}
            }
            return Ok(());
        }
    }

    if watch {
        run_watch(&tracker, interval.max(1))
    } else {
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

    const CORPUS_LINE: &str = r#"{"context":"The cat sat on the red mat.","question":"What color was the mat?","gold":"red","scorer":"token_f1"}"#;

    fn offline_args(corpus: PathBuf, json_out: Option<PathBuf>) -> BenchArgs {
        BenchArgs {
            corpus,
            provider: "openai".to_string(),
            preset: "rag".to_string(),
            model: "openai/gpt-4o-mini".to_string(),
            judge_model: "openai/gpt-4o-mini".to_string(),
            route: String::new(),
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
