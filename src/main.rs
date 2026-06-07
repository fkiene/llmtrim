//! llmtrim CLI.
//!
//! Network-free surface: `compress` reads a provider request body on stdin and writes the
//! compressed body to stdout; `send`/`batch` add the network round-trip; `monitor` shows
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

#[derive(Parser)]
#[command(
    name = "llmtrim",
    version,
    about = "Static, deterministic LLM prompt/payload compressor"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a provider request body read from stdin; write compressed JSON to stdout.
    Compress {
        /// Target provider: openai|anthropic. Omit to auto-detect from the request shape.
        #[arg(long)]
        provider: Option<String>,
    },
    /// Compress a request from stdin, send it to the provider, print the response.
    /// Requires the provider API key in the environment (see `transport`).
    Send {
        /// Target provider: openai|anthropic. Omit to auto-detect from the request shape.
        #[arg(long)]
        provider: Option<String>,
    },
    /// Run the MITM interceptor: an HTTPS proxy that compresses LLM API requests in
    /// flight, across every tool and provider. Set HTTPS_PROXY to it and trust the CA
    /// (`llmtrim ca`). No API key needed — the client's own auth is passed through.
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
    /// One command: ensure the CA, set HTTPS_PROXY + trust the CA in your shell profile,
    /// enable run-at-login, and start the interceptor. The fastest path from install to
    /// compressing. No IDE settings are touched — llmtrim is purely a MITM proxy.
    Setup {
        /// Interceptor port.
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
    /// Undo everything `setup` did: stop the daemon, disable autostart, strip the
    /// shell-profile env block, and remove the CA + state (and the binary). Transparent —
    /// prints each step.
    Uninstall {
        /// Also delete the savings ledger (kept by default).
        #[arg(long)]
        purge: bool,
        /// Leave the binary in place (default removes it on Unix).
        #[arg(long)]
        keep_binary: bool,
    },
    /// Stop the background interceptor daemon.
    Stop,
    /// Update to the latest release (channel-aware: binary self-updates via the installer;
    /// cargo/Homebrew print their command) and restart the daemon onto the new binary.
    Update,
    /// Savings dashboard from the ledger + interceptor state. Default: a snapshot;
    /// `--watch` for a live view; `--daily/--weekly/--monthly` for time-series;
    /// `--json/--csv` to export. Aliased as `status` and `gain`.
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
    },
    /// Run the interceptor at login (`--off` to disable). systemd (Linux) / launchd (macOS).
    Autostart {
        /// Disable autostart instead of enabling it.
        #[arg(long)]
        off: bool,
        /// Port the autostarted interceptor listens on.
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
    /// Print the local CA certificate path (generating it on first run) and how to trust
    /// it. Required once before `serve` can intercept HTTPS.
    Ca,
    /// Compress every request in a batch JSONL (stdin → stdout) for the Batch API.
    Batch {
        /// Provider of the batch requests: openai|anthropic.
        #[arg(long, default_value = "openai")]
        provider: String,
    },
    /// Evaluate Stage B retrieval recall + savings on a held-out corpus JSONL (§6).
    Eval {
        /// Corpus JSONL: lines with {context|input, question|query, answers|answer}.
        #[arg(long)]
        corpus: PathBuf,
        /// Provider for the cases: openai|anthropic.
        #[arg(long, default_value = "openai")]
        provider: String,
        /// Stage B keep_ratio to evaluate.
        #[arg(long, default_value_t = 0.5)]
        keep_ratio: f64,
    },
    /// A/B benchmark (§6 quality axis): tokens saved vs quality retained, on a real
    /// corpus + model. Sends ORIGINAL and COMPRESSED requests, scores both, prices the
    /// round-trip. Credentials come from the env or a local `.env` (OpenRouter preferred).
    Bench(BenchArgs),
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Normalized corpus JSONL (friendly {context,question,gold,scorer} or explicit {request,…}).
    #[arg(long)]
    corpus: PathBuf,
    /// Provider: openai|anthropic.
    #[arg(long, default_value = "openai")]
    provider: String,
    /// Preset to evaluate: auto|safe|rag|agent|code|aggressive|cache|reasoning.
    #[arg(long, default_value = "auto")]
    preset: String,
    /// Model id to send (OpenRouter style, e.g. openai/gpt-oss-20b).
    #[arg(long, default_value = "openai/gpt-oss-20b")]
    model: String,
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

fn main() -> Result<()> {
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
                println!("llmtrim: interceptor running in background (pid {pid}, port {port})");
                println!("  logs:   {}", llmtrim::daemon::logfile()?.display());
                println!("  status: llmtrim status     stop: llmtrim stop");
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
        Commands::Stop => match llmtrim::daemon::stop()? {
            Some(pid) => {
                println!("Stopped interceptor (pid {pid}).");
                if llmtrim::setup::profile_has_block() {
                    eprintln!(
                        "⚠ HTTPS_PROXY still points at llmtrim in your shell profile — \
                         new HTTPS to LLM hosts will fail until you start it again \
                         (`llmtrim serve --daemon`) or run `llmtrim uninstall`."
                    );
                }
            }
            None => println!("No interceptor daemon was running."),
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
        } => run_monitor(watch, interval, daily, weekly, monthly, json, csv)?,
        Commands::Autostart { off, port } => llmtrim::autostart::configure(!off, port)?,
        Commands::Ca => {
            let path = llmtrim::serve::ca_cert_path()?;
            llmtrim::serve::ensure_ca()?; // generate on first run
            println!("llmtrim local CA: {}", path.display());
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
            println!("The CA is name-constrained to LLM API domains only.");
        }
        Commands::Batch { provider } => {
            let kind = ProviderKind::from_str(&provider)?;
            let config = llmtrim::config::DenseConfig::load().unwrap_or_else(|e| {
                eprintln!("llmtrim: {e}; using defaults");
                llmtrim::config::DenseConfig::default()
            });
            let input = read_stdin()?;
            let (mut before, mut after) = (0usize, 0usize);
            let mut out = String::new();
            for line in input.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match llmtrim::batch::compress_line(line, kind, &config) {
                    Ok(b) => {
                        before += b.tokens_before;
                        after += b.tokens_after;
                        out.push_str(&b.line);
                        out.push('\n');
                    }
                    Err(e) => {
                        eprintln!("llmtrim batch: passing a line through unchanged ({e})");
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
            print!("{out}");
            eprintln!(
                "llmtrim batch: {before} -> {after} input tokens; submit the output JSONL to the provider Batch API for ~50% off"
            );
            // Record the batch aggregate to the savings ledger (best-effort; mirrors
            // `compress` — a ledger failure must never fail the batch output).
            if before > 0
                && let (Ok(tracker), Ok(counter)) =
                    (Tracker::open(), llmtrim::tokenizer::counter_for(kind, None))
            {
                let _ = tracker.record(&Record {
                    provider: kind.as_str().to_string(),
                    model: None,
                    tokenizer: counter.label().to_string(),
                    exact: counter.is_exact(),
                    input_before: before as i64,
                    input_after: after as i64,
                    output_before: None,
                    output_after: None,
                });
            }
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
            let config = llmtrim::config::DenseConfig {
                retrieve: true,
                retrieve_keep_ratio: keep_ratio,
                retrieve_min_segment_chars: 120,
                output_control: false,
                ..Default::default()
            };
            let results = llmtrim::quality::run_recall(&cases, &config)?;
            for r in &results {
                println!(
                    "  {:<12} recall={:.2}  {} -> {} tok ({:.1}%)",
                    r.name,
                    r.recall,
                    r.tokens_before,
                    r.tokens_after,
                    r.savings_pct()
                );
            }
            let recall = llmtrim::quality::mean_recall(&results);
            let savings = if results.is_empty() {
                0.0
            } else {
                results.iter().map(|r| r.savings_pct()).sum::<f64>() / results.len() as f64
            };
            println!(
                "corpus: {} cases  mean recall={recall:.2}  mean savings={savings:.1}%  (keep_ratio={keep_ratio})",
                results.len()
            );
        }
        Commands::Bench(args) => run_bench(args)?,
    }
    Ok(())
}

/// Percent reduction from `before` to `after` (0 when `before` is 0). The savings
/// formula shared by the bench/offline/gain reporting paths.
fn saved_pct(before: f64, after: f64) -> f64 {
    if before > 0.0 {
        (before - after) / before * 100.0
    } else {
        0.0
    }
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
    let config = match &args.config {
        Some(path) => {
            let t = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read config {}", path.display()))?;
            toml::from_str::<DenseConfig>(&t)
                .with_context(|| format!("failed to parse config {}", path.display()))?
        }
        None => DenseConfig::preset(&args.preset).with_context(|| {
            format!(
                "unknown preset '{}' (safe|rag|agent|code|aggressive)",
                args.preset
            )
        })?,
    };

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
    println!(
        "ablation — input tokens, offline, preset={preset}, {} cases",
        cases.len()
    );
    println!(
        "  {:<18} {:>7} {:>7} {:>8} {:>12}",
        "config", "before", "after", "saved%", "stage saves"
    );
    for (label, before, after) in &rows {
        let saved = saved_pct(*before as f64, *after as f64);
        let contribution = if label == "full" {
            String::new()
        } else {
            format!("{:+.0} tok", *after as f64 - full_after)
        };
        println!("  {label:<18} {before:>7} {after:>7} {saved:>7.1}% {contribution:>12}");
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
    println!(
        "offline: {} cases  input {before} -> {after} tok ({:.1}% saved)  preset={preset}",
        cases.len(),
        saved_pct(before as f64, after as f64),
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
    let table = std::fs::read_to_string(&args.pricing)
        .map(|s| bench::load_pricing(&s))
        .unwrap_or_default();
    let price = bench::resolve_pricing(&table, &args.model);
    let counter = llmtrim::tokenizer::counter_for(kind, Some(&args.model))?;
    let api_key = dotenv_get("OPENROUTER_API_KEY")
        .context("OPENROUTER_API_KEY not set (in env or a local .env)")?;
    let llm = llmtrim::quality::OpenRouterModel::new(api_key, kind)?;
    // BenchScorer covers pass@1 (runs tests), tool-call match, and the LLM judge
    // (reusing this endpoint), plus resource-free text scoring.
    let scorer = bench::BenchScorer {
        exec_timeout: 10,
        judge: Some(&llm),
        judge_model: args.model.clone(),
        route: args.route.clone(),
    };

    let outcomes = bench::run_ab(cases, config, &llm, counter.as_ref(), &scorer, price)?;
    for o in &outcomes {
        println!(
            "  {:<16} q {:.2}->{:.2}  in {}->{}  out {}->{}",
            o.name,
            o.quality_orig,
            o.quality_comp,
            o.tokens_in_before,
            o.tokens_in_after,
            o.tokens_out_orig,
            o.tokens_out_comp,
        );
    }
    let f = bench::summarize(&outcomes);
    println!(
        "\n{} on {} (model={}, n={})\n  input saved   {:.1}%\n  output saved  {:.1}%\n  cost saved    {:.1}%\n  cache used    {:.1}%  (compressed input served from prompt cache)\n  quality       {:.1}% -> {:.1}%  (retention {:+.1}pp, 95%CI ±{:.1})",
        args.preset,
        args.corpus.display(),
        args.model,
        f.n,
        f.tokens_in_saved_pct,
        f.tokens_out_saved_pct,
        f.cost_saved_pct,
        f.cache_used_pct,
        f.quality_orig.mean * 100.0,
        f.quality_comp.mean * 100.0,
        f.retention_pp,
        f.quality_comp.ci95 * 100.0,
    );
    if let Some(path) = &args.json_out {
        let rows: Vec<_> = outcomes
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
            "preset": args.preset, "model": args.model, "corpus": args.corpus.display().to_string(),
            "n": f.n,
            "tokens_in_saved_pct": f.tokens_in_saved_pct,
            "tokens_out_saved_pct": f.tokens_out_saved_pct,
            "cost_saved_pct": f.cost_saved_pct,
            "cache_used_pct": f.cache_used_pct,
            "quality_orig": f.quality_orig.mean, "quality_comp": f.quality_comp.mean,
            "retention_pp": f.retention_pp, "ci95": f.quality_comp.ci95,
            "cases": rows,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
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

/// Rich `status`: daemon state, CA presence, and savings (tokens + input-side cost).
/// Colour the dashboard only for an interactive terminal that hasn't opted out.
fn should_color() -> bool {
    use std::io::IsTerminal;
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
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

/// Interceptor daemon + CA state for the dashboard header.
fn daemon_view() -> monitor::DaemonView {
    use llmtrim::daemon;
    let ca_present = matches!(llmtrim::serve::ca_cert_path(), Ok(p) if p.exists());
    match daemon::running() {
        Some(s) => monitor::DaemonView {
            running: true,
            pid: s.pid,
            port: s.port,
            uptime: daemon::human_uptime(daemon::uptime_secs(s.started_at)),
            ca_present,
        },
        None => monitor::DaemonView {
            running: false,
            pid: 0,
            port: 0,
            uptime: String::new(),
            ca_present,
        },
    }
}

fn monitor_cost(tracker: &Tracker) -> Option<monitor::Cost> {
    let models = tracker.by_model().ok()?;
    cost_estimate(&models).map(|(saved, spend)| monitor::Cost { saved, spend })
}

/// Per-model rows for the breakdown, top 8 by request volume, priced where the registry
/// knows the model.
fn model_views(tracker: &Tracker) -> Result<Vec<monitor::ModelView>> {
    let mut models: Vec<monitor::ModelView> = tracker
        .by_model()?
        .into_iter()
        .filter(|m| m.events > 0)
        .map(|m| {
            let cost_saved =
                m.model.as_deref().and_then(llm_prices).map(|(inp, _)| {
                    (m.input_before - m.input_after).max(0) as f64 / 1_000_000.0 * inp
                });
            monitor::ModelView {
                name: m
                    .model
                    .unwrap_or_else(|| format!("{} · unknown model", m.provider)),
                events: m.events,
                saved_pct: saved_pct(m.input_before as f64, m.input_after as f64),
                cost_saved,
            }
        })
        .collect();
    models.sort_unstable_by_key(|b| std::cmp::Reverse(b.events));
    models.truncate(8);
    Ok(models)
}

fn render_snapshot(tracker: &Tracker, color: bool) -> Result<String> {
    let summary = tracker.summary()?;
    let models = model_views(tracker)?;
    let cost = monitor_cost(tracker);
    let daemon = daemon_view();
    let mut out = monitor::snapshot(color, Some(&daemon), &summary, &models, cost.as_ref());
    // Passive, cached (≤24h), opt-out update notice (LLMTRIM_NO_UPDATE_CHECK to disable).
    if let Some(v) = llmtrim::update::check(false) {
        out.push_str(&format!(
            "\n  ↑ llmtrim v{v} available — run `llmtrim update`\n"
        ));
    }
    Ok(out)
}

/// Live dashboard: clear + repaint each tick, with an input-token save-rate once we have
/// two samples. Exits on Ctrl-C (default SIGINT).
fn run_watch(tracker: &Tracker, interval: u64) -> Result<()> {
    let color = should_color();
    let mut prev: Option<i64> = None;
    loop {
        let summary = tracker.summary()?;
        let mut frame = String::from("\x1b[2J\x1b[H"); // clear screen + cursor home
        frame.push_str(&render_snapshot(tracker, color)?);
        if let Some(p) = prev {
            let rate = (summary.saved() - p) as f64 / interval as f64;
            frame.push_str(&format!("\n  {rate:+.0} input tokens/s saved\n"));
        }
        prev = Some(summary.saved());
        frame.push_str(&format!(
            "  refreshing every {interval}s · Ctrl-C to exit\n"
        ));
        print!("{frame}");
        std::io::stdout().flush().ok();
        std::thread::sleep(std::time::Duration::from_secs(interval));
    }
}

fn run_monitor(
    watch: bool,
    interval: u64,
    daily: bool,
    weekly: bool,
    monthly: bool,
    json: bool,
    csv: bool,
) -> Result<()> {
    let tracker = Tracker::open().context("failed to open savings ledger")?;

    // Time-series report / export.
    if let Some(period) = period_flag(daily, weekly, monthly) {
        let rows = tracker.by_period(period)?;
        if csv {
            print!("{}", monitor::export_csv(&rows));
        } else if json {
            let s = tracker.summary()?;
            let models = model_views(&tracker)?;
            println!(
                "{}",
                monitor::export_json(&s, &models, monitor_cost(&tracker).as_ref(), &rows)
            );
        } else {
            print!(
                "{}",
                monitor::period_report(should_color(), period.label(), &rows)
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
            println!(
                "{}",
                monitor::export_json(&s, &models, monitor_cost(&tracker).as_ref(), &rows)
            );
        }
        return Ok(());
    }

    if watch {
        run_watch(&tracker, interval.max(1))
    } else {
        print!("{}", render_snapshot(&tracker, should_color())?);
        Ok(())
    }
}

/// `(input savings, total spend)` in USD, priced per model via the `llm_providers`
/// registry. Savings = input tokens we cut × input price; total spend = the actual
/// input_after + measured output, at registry rates. `None` when no recorded model matches
/// the registry (or this build has no interceptor feature).
#[cfg(feature = "intercept")]
fn cost_estimate(models: &[llmtrim::tracking::ModelRow]) -> Option<(f64, f64)> {
    let mut saved = 0.0;
    let mut spend = 0.0;
    let mut matched = false;
    for m in models {
        let Some(model_id) = m.model.as_deref() else {
            continue;
        };
        if let Some((input_price, output_price)) = llm_prices(model_id) {
            saved += (m.input_before - m.input_after).max(0) as f64 / 1_000_000.0 * input_price;
            spend += m.input_after as f64 / 1_000_000.0 * input_price
                + m.output_after as f64 / 1_000_000.0 * output_price;
            matched = true;
        }
    }
    matched.then_some((saved, spend))
}

/// Per-1M-token `(input, output)` price for a model, matched across every provider in the
/// registry (the ledger records the wire-shape provider, not the upstream brand).
#[cfg(feature = "intercept")]
fn llm_prices(model_id: &str) -> Option<(f64, f64)> {
    for &provider_id in llm_providers::get_providers_data().keys() {
        if let Some(model) = llm_providers::get_model_ref(provider_id, model_id) {
            return Some((model.input_price, model.output_price));
        }
    }
    None
}

#[cfg(not(feature = "intercept"))]
fn cost_estimate(_models: &[llmtrim::tracking::ModelRow]) -> Option<(f64, f64)> {
    None
}

#[cfg(not(feature = "intercept"))]
fn llm_prices(_model_id: &str) -> Option<(f64, f64)> {
    None
}
