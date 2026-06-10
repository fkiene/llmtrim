//! llmtrim CLI.
//!
//! Network-free surface: `compress` reads a provider request body on stdin and writes the
//! compressed body to stdout; `send` adds the network round-trip; `monitor` shows
//! savings from the SQLite ledger; `serve`/`setup` run the MITM interceptor. The pure
//! transform core lives in `lib.rs`.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llmtrim::bench::{self, BenchCase};
use llmtrim::config::DenseConfig;
use llmtrim::ir::ProviderKind;
use llmtrim::monitor;
use llmtrim::tracking::{Period, Record, Tracker};
use llmtrim::transport::Endpoint;
use llmtrim::ui::{self, Tone};

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

#[derive(Parser)]
#[command(
    name = "llmtrim",
    version,
    about = "Static, deterministic LLM prompt/payload compressor",
    styles = HELP_STYLES
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
    /// Run the HTTPS interceptor that compresses LLM traffic in flight
    ///
    /// A local MITM proxy covering every tool and provider: set HTTPS_PROXY to it
    /// and trust the CA (`llmtrim ca`). No API key needed — the client's own auth
    /// passes through untouched.
    Serve {
        /// Port to listen on (127.0.0.1).
        #[arg(long, default_value_t = 8787)]
        port: u16,
        /// Run detached in the background; manage with `status` / `stop`.
        #[arg(long)]
        daemon: bool,
        /// Internal: run with crash-restart supervision (used by `--daemon`/autostart).
        #[arg(long, hide = true)]
        supervised: bool,
    },
    /// Set everything up and start saving (CA, environment, autostart, daemon)
    ///
    /// The fastest path from install to compressing: ensures the local CA, sets
    /// HTTPS_PROXY + CA trust in your environment (shell profile on POSIX,
    /// HKCU\Environment on Windows), enables run-at-login, and starts the
    /// interceptor. Idempotent — re-running reuses the same port and won't restart a
    /// healthy daemon. No IDE settings are touched, no sudo.
    Setup {
        /// Interceptor port. Omit to auto-select a free port starting at 8787.
        #[arg(long)]
        port: Option<u16>,
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
    /// matches what your tools point at. Run `setup` first if you haven't.
    Start {
        /// Port to listen on. Omit to reuse the configured port (or 8787).
        #[arg(long)]
        port: Option<u16>,
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
    #[command(visible_aliases = ["status", "gain"])]
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
        /// Port the autostarted interceptor listens on.
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
    /// Print the local CA certificate path and how to trust it
    ///
    /// Generates the CA on first run. Required once before `serve` can intercept
    /// HTTPS; the CA is name-constrained to LLM API domains only.
    Ca,
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
    /// A/B benchmark: tokens saved vs quality retained, on a real model
    ///
    /// Sends ORIGINAL and COMPRESSED requests, scores both, and prices the
    /// round-trip. Credentials come from the env or a local `.env` (OpenRouter).
    /// `--offline`/`--ablate` measure tokens without any network calls.
    Bench(BenchArgs),
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
            let result = llmtrim::compress(&input, kind)?;

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
            let result = llmtrim::compress(&input, kind)?;
            let endpoint = Endpoint::from_env(result.provider)?;
            let response = endpoint.send(&result.request_json)?;

            // Record to the savings ledger (best-effort: never block the user's output).
            if let Ok(counter) =
                llmtrim::tokenizer::counter_for(result.provider, result.model.as_deref())
            {
                let output_after = serde_json::from_str::<serde_json::Value>(&response)
                    .ok()
                    .and_then(|v| llmtrim::provider::for_kind(result.provider).answer_text(&v))
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
                    });
                }
            }

            println!("{response}");
        }
        Commands::Serve {
            port,
            daemon,
            supervised,
        } => {
            if daemon {
                let pid = llmtrim::daemon::spawn_detached(port)?;
                let color = ui::color_stdout();
                println!(
                    "{}",
                    ui::ok(
                        color,
                        &format!("Interceptor running in background · pid {pid} · port {port}")
                    )
                );
                for (label, value) in [
                    ("logs", llmtrim::daemon::logfile()?.display().to_string()),
                    ("watch", "llmtrim status".to_string()),
                    ("stop", "llmtrim stop".to_string()),
                ] {
                    println!(
                        "    {}  {value}",
                        ui::paint(color, Tone::Dim, &format!("{label:<5}"))
                    );
                }
            } else if supervised {
                llmtrim::serve::run_supervised(port)?;
            } else {
                llmtrim::serve::run(port)?;
            }
        }
        Commands::Setup { port } => llmtrim::setup::run(port)?,
        Commands::Uninstall { purge, keep_binary } => {
            llmtrim::setup::uninstall(purge, keep_binary)?
        }
        Commands::Start { port } => {
            let color = ui::color_stdout();
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
                // the one already wired into the env, else 8787) so we match existing clients.
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
            llmtrim::autostart::configure(!off, port)?;
            let color = ui::color_stdout();
            if off {
                println!("{}", ui::ok(color, "Autostart disabled."));
            } else {
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
        Commands::Ca => {
            let path = llmtrim::serve::ca_cert_path()?;
            llmtrim::serve::ensure_ca()?; // generate on first run
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
                println!("  $env:HTTPS_PROXY = \"http://127.0.0.1:8787\"");
                println!("  llmtrim serve");
                println!();
                println!("System-wide trust (non-PowerShell / GUI apps):");
                println!("  certutil -addstore -user Root \"{}\"", path.display());
            }
            #[cfg(not(windows))]
            {
                println!("  export NODE_EXTRA_CA_CERTS={}", path.display());
                println!("  export HTTPS_PROXY=http://127.0.0.1:8787");
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
            let config = llmtrim::config::DenseConfig {
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
        Commands::Bench(args) => run_bench(args)?,
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
        run_offline(&cases, kind, &config, &args.preset)
    } else {
        run_live(&cases, kind, &config, &args)
    }
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
fn run_offline(
    cases: &[BenchCase],
    kind: ProviderKind,
    config: &DenseConfig,
    preset: &str,
) -> Result<()> {
    let (mut before, mut after) = (0usize, 0usize);
    for c in cases {
        let r = llmtrim::compress_with_config(&c.request, Some(kind), config)?;
        before += r.input_tokens_before.0;
        after += r.input_tokens_after.0;
    }
    // FROZEN wording: bench/scripts/vs_headroom.py regex-parses this exact line
    // (`input (\d+) -> (\d+) tok \(([\d.]+)% saved\)`). Restyle only in lockstep with it.
    println!(
        "offline: {} cases  input {before} -> {after} tok ({:.1}% saved)  preset={preset}",
        cases.len(),
        ui::saved_pct(before as f64, after as f64),
    );
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
    let counter = llmtrim::tokenizer::counter_for(kind, Some(&args.model))?;
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
        let doc = serde_json::json!({
            // Reproducibility: record the config SOURCE (the path when --config overrode the
            // preset; else the preset name) AND the actually-resolved settings, so two runs
            // with the same metadata are genuinely the same run (#15).
            "preset": args.preset,
            "config_path": args.config.as_ref().map(|p| p.display().to_string()),
            "config_resolved": config,
            "model": args.model,
            "judge_model": args.judge_model,
            "route": args.route,
            "corpus": args.corpus.display().to_string(),
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
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)
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
    }
}

fn monitor_cost(tracker: &Tracker) -> Option<monitor::Cost> {
    let models = tracker.by_model().ok()?;
    cost_estimate(&models)
}

/// Per-model rows for the breakdown, top 8 by request volume, priced where the registry
/// knows the model.
fn model_views(tracker: &Tracker) -> Result<Vec<monitor::ModelView>> {
    let mut models: Vec<monitor::ModelView> = tracker
        .by_model()?
        .into_iter()
        .filter(|m| m.events > 0)
        .map(|m| {
            // Per-model USD figures where the registry prices the model — used to project the
            // round-trip the same way the hero does.
            let priced = m.model.as_deref().and_then(llm_prices);
            let cost_saved = priced
                .map(|(inp, _)| (m.input_before - m.input_after).max(0) as f64 / 1_000_000.0 * inp);
            let out_spend = priced.map(|(_, outp)| m.output_after as f64 / 1_000_000.0 * outp);
            let spend = priced.map(|(inp, outp)| {
                m.input_after as f64 / 1_000_000.0 * inp
                    + m.output_after as f64 / 1_000_000.0 * outp
            });
            monitor::ModelView {
                name: m
                    .model
                    .unwrap_or_else(|| format!("{} · unknown model", m.provider)),
                events: m.events,
                saved_pct: ui::saved_pct(m.input_before as f64, m.input_after as f64),
                cost_saved,
                spend,
                out_spend,
            }
        })
        .collect();
    models.sort_unstable_by_key(|b| std::cmp::Reverse(b.events));
    models.truncate(8);
    Ok(models)
}

/// Today's priced saving (UTC), for the hero's recency anchor. `None` when nothing
/// priced ran today — the dashboard hides the figure rather than showing $0.00.
fn today_saved_usd(tracker: &Tracker) -> Option<f64> {
    let models = tracker.by_model_today().ok()?;
    cost_estimate(&models).map(|c| c.saved)
}

fn render_snapshot(tracker: &Tracker, color: bool) -> Result<(String, monitor::Health)> {
    let summary = tracker.summary()?;
    let models = model_views(tracker)?;
    let cost = monitor_cost(tracker);
    let daemon = daemon_view(&summary);
    let health = monitor::health(&daemon);
    let mut out = monitor::snapshot(
        color,
        Some(&daemon),
        &summary,
        &models,
        cost.as_ref(),
        today_saved_usd(tracker),
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

/// Live dashboard: clear + repaint each tick, with an input-token save-rate once we have
/// two samples. Exits on Ctrl-C (default SIGINT).
fn run_watch(tracker: &Tracker, interval: u64) -> Result<()> {
    let color = ui::color_stdout();
    // Screen-control escapes (clear + home) are for interactive terminals only —
    // piped/redirected watch output gets appended frames instead of raw escapes.
    let tty = ui::stdout_is_tty();
    let mut prev: Option<i64> = None;
    loop {
        let summary = tracker.summary()?;
        let mut frame = if tty {
            String::from("\x1b[2J\x1b[H") // clear screen + cursor home
        } else {
            String::new()
        };
        frame.push_str(&render_snapshot(tracker, color)?.0);
        if let Some(p) = prev {
            let rate = (summary.saved() - p) as f64 / interval as f64;
            // Only show the rate when traffic actually flowed this interval — a perpetual
            // "+0/s" on an idle proxy reads like fake data.
            if rate.abs() >= 0.5 {
                frame.push_str(&format!("\n  {rate:+.0} input tokens/s saved\n"));
            }
        }
        prev = Some(summary.saved());
        frame.push_str(&ui::paint(
            color,
            Tone::Dim,
            &format!("  refreshing every {interval}s · Ctrl-C to exit\n"),
        ));
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
        } else if json {
            let s = tracker.summary()?;
            let models = model_views(&tracker)?;
            let daemon = daemon_view(&s);
            println!(
                "{}",
                monitor::export_json(
                    &s,
                    &models,
                    monitor_cost(&tracker).as_ref(),
                    &rows,
                    Some(&daemon)
                )
            );
        } else {
            print!(
                "{}",
                monitor::period_report(ui::color_stdout(), period.label(), &rows)
            );
        }
        return Ok(());
    }

    // Whole-snapshot export (no period selected).
    if json || csv {
        let rows = tracker.by_period(Period::Day)?;
        if csv {
            print!("{}", monitor::export_csv(&rows));
        } else {
            let s = tracker.summary()?;
            let models = model_views(&tracker)?;
            let daemon = daemon_view(&s);
            println!(
                "{}",
                monitor::export_json(
                    &s,
                    &models,
                    monitor_cost(&tracker).as_ref(),
                    &rows,
                    Some(&daemon)
                )
            );
        }
        return Ok(());
    }

    if watch {
        run_watch(&tracker, interval.max(1))
    } else {
        let (out, health) = render_snapshot(&tracker, ui::color_stdout())?;
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
fn cache_multipliers(provider: &str) -> (f64, f64) {
    match provider {
        "anthropic" => (0.10, 1.25),
        "openai" => (0.50, 0.0),
        "google" => (0.25, 0.0),
        _ => (1.0, 0.0),
    }
}

/// USD cost figures priced per model via [`llm_prices`], `None` when no recorded model
/// is priced.
///
/// - `saved`/`spend` value tokens at **list input rates** (tokens cut × input price; the
///   compressed input_after + measured output). Consistent numerator/denominator — the
///   headline basis.
/// - `net_saved` re-prices the same measured saving against the **real bill**: the
///   provider-reported usage split (fresh × 1.0 + cache writes × 1.25 + cache reads ×
///   0.1, per provider) gives the actual input bill, and the counterfactual bill scales
///   it by the measured compression ratio (same cache mix, prompt larger by `pct`):
///   `net_saved = net_bill × pct / (1 − pct)`. On traffic with no cache usage this
///   degrades exactly to the list-rate figure.
/// - `out_spend_shaped` is the output spend from requests that actually carried the
///   output-shaping instruction — the only spend the benchmark factor may be projected on.
fn cost_estimate(models: &[llmtrim::tracking::ModelRow]) -> Option<monitor::Cost> {
    let mut cost = monitor::Cost {
        saved: 0.0,
        spend: 0.0,
        out_spend: 0.0,
        net_saved: 0.0,
        out_spend_shaped: 0.0,
    };
    let mut matched = false;
    for m in models {
        let Some(model_id) = m.model.as_deref() else {
            continue;
        };
        if let Some((input_price, output_price)) = llm_prices(model_id) {
            let delta = (m.input_before - m.input_after).max(0) as f64;
            cost.saved += delta / 1_000_000.0 * input_price;
            let out = m.output_after as f64 / 1_000_000.0 * output_price;
            cost.spend += m.input_after as f64 / 1_000_000.0 * input_price + out;
            cost.out_spend += out;
            cost.out_spend_shaped += m.output_after_shaped as f64 / 1_000_000.0 * output_price;

            let (read_mult, write_mult) = cache_multipliers(&m.provider);
            let net_bill = (m.fresh_input_est as f64
                + m.cache_write as f64 * write_mult
                + m.cache_read as f64 * read_mult)
                / 1_000_000.0
                * input_price;
            let pct = if m.input_before > 0 {
                (delta / m.input_before as f64).min(0.95)
            } else {
                0.0
            };
            cost.net_saved += net_bill * pct / (1.0 - pct);
            matched = true;
        }
    }
    matched.then_some(cost)
}

/// Per-1M-token `(input, output)` price for a model: the `llm_providers` registry first,
/// matched across every provider (the ledger records the wire-shape provider, not the
/// upstream brand), then the embedded models.dev snapshot for models the registry hasn't
/// shipped yet (e.g. day-one releases like claude-fable-5 on 0.14.3).
fn llm_prices(model_id: &str) -> Option<(f64, f64)> {
    #[cfg(feature = "intercept")]
    for &provider_id in llm_providers::get_providers_data().keys() {
        if let Some(model) = llm_providers::get_model_ref(provider_id, model_id) {
            return Some((model.input_price, model.output_price));
        }
    }
    snapshot_prices(model_id)
}

/// Fallback table: the pinned models.dev snapshot the bench prices from, embedded at
/// compile time (the package re-includes bench/pricing.json for this). Exact id first,
/// then with the `provider/` prefix stripped, mirroring [`bench::resolve_pricing`] — but
/// zero-priced rows (free tiers, parse gaps) return `None` so an unknown model shows a
/// blank cost cell rather than a misleading $0.00.
fn snapshot_prices(model_id: &str) -> Option<(f64, f64)> {
    static TABLE: once_cell::sync::Lazy<bench::PriceTable> =
        once_cell::sync::Lazy::new(|| bench::load_pricing(include_str!("../bench/pricing.json")));
    let p = TABLE.get(model_id).or_else(|| {
        let (_, bare) = model_id.split_once('/')?;
        TABLE.get(bare)
    })?;
    let (input, output) = (p.input_per_1k * 1000.0, p.output_per_1k * 1000.0);
    (input > 0.0 || output > 0.0).then_some((input, output))
}
