//! `mcp` — Model Context Protocol server over stdio.
//!
//! A fourth way to reach the engine, next to the proxy, the CLI, and the language
//! bindings: any MCP client (Claude Code, Cursor, custom agents) spawns `llmtrim mcp`
//! and calls llmtrim's compression and savings stats as MCP tools. The transport is
//! JSON-RPC 2.0 over stdin/stdout — the form clients spawn by default.
//!
//! Like [`crate::serve`], the real implementation is feature-gated (`mcp`); a build
//! without it keeps the `mcp` subcommand but bails with a clear rebuild hint.

#[cfg(not(feature = "mcp"))]
pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("this build has no MCP server; rebuild with `--features mcp`")
}

#[cfg(not(feature = "mcp"))]
pub fn install(_print: bool, _force: bool) -> anyhow::Result<()> {
    anyhow::bail!("this build has no MCP server; rebuild with `--features mcp`")
}

#[cfg(feature = "mcp")]
pub use imp::{install, run};

/// The MCP handler the `mcp` command serves, for protocol-level tests that drive it over
/// an in-memory transport instead of stdio. `db` is an isolated ledger path so the test
/// never writes to the user's real savings DB, and `config` is a fixed compression config so
/// the test never reads the developer's `~/.llmtrim` (a malformed one used to fail it). Not a
/// stable API: it exists only for the `tests/mcp_protocol.rs` integration test (which can't
/// reach the private `mod imp`).
#[doc(hidden)]
#[cfg(feature = "mcp")]
pub fn test_server(
    db: std::path::PathBuf,
    config: llmtrim_core::config::DenseConfig,
) -> impl rmcp::ServerHandler + Clone {
    imp::server_at(db, config)
}

#[cfg(feature = "mcp")]
mod imp {
    use std::path::PathBuf;
    use std::str::FromStr;

    use anyhow::{Context, Result};
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::{CallToolResult, ContentBlock};
    use rmcp::{
        ErrorData as McpError, ServerHandler, ServiceExt, schemars, tool, tool_handler,
        tool_router, transport::stdio,
    };
    use serde::Deserialize;
    use serde_json::{Value, json};

    use crate::tracking::{Record, Tracker};
    use llmtrim_core::CompressResult;
    use llmtrim_core::config::DenseConfig;
    use llmtrim_core::ir::ProviderKind;
    use llmtrim_core::tokenizer;

    /// The model `llmtrim_compress_text` wraps a blob under. Arbitrary (the request is
    /// synthetic and never sent); it only selects the tokenizer for the reported counts.
    const TEXT_WRAP_MODEL: &str = "gpt-4o";

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct CompressArgs {
        /// The provider request body (OpenAI, Anthropic, or Google shape). Accepts either the
        /// JSON object itself or a JSON string of it; the whole body is compressed and
        /// returned in the same shape.
        request: RequestArg,
        /// Provider hint: `openai`, `anthropic`, or `google`. Leave unset to detect it
        /// from the request shape.
        #[serde(default)]
        provider: Option<String>,
    }

    /// A request body passed as either a JSON object (the natural form an agent emits) or a
    /// JSON string of one. Both reduce to the string the engine takes.
    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    #[serde(untagged)]
    enum RequestArg {
        Text(String),
        Json(serde_json::Map<String, Value>),
    }

    impl RequestArg {
        fn into_body(self) -> String {
            match self {
                RequestArg::Text(s) => s,
                RequestArg::Json(map) => Value::Object(map).to_string(),
            }
        }
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct CompressTextArgs {
        /// A single text blob to shrink (a tool output, a document, a message). It is
        /// wrapped in a one-message request, compressed, and the shrunk text is returned.
        text: String,
    }

    // `llmtrim_stats` takes no parameters, but the `#[tool]` macro still wants a typed args
    // struct, so this is an empty one (the advertised input schema is `{}`).
    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct StatsArgs {}

    /// The MCP handler. `db` selects the ledger: `None` is the real one (`Tracker::open`,
    /// honoring `LLMTRIM_DB_PATH`/XDG like every other front-end); `Some(path)` is an
    /// isolated ledger for tests, so the protocol test never writes to the user's real DB.
    #[derive(Clone)]
    pub(super) struct LlmtrimServer {
        db: Option<PathBuf>,
        /// The compression config: `None` is the real one (`DenseConfig::load`, honoring
        /// `~/.llmtrim` like every other front-end); `Some(c)` is a fixed config for tests,
        /// so the protocol test never depends on the developer's config file.
        config: Option<DenseConfig>,
    }

    pub(super) fn server() -> LlmtrimServer {
        LlmtrimServer {
            db: None,
            config: None,
        }
    }

    pub(super) fn server_at(db: PathBuf, config: DenseConfig) -> LlmtrimServer {
        LlmtrimServer {
            db: Some(db),
            config: Some(config),
        }
    }

    impl LlmtrimServer {
        /// The config to compress under: the injected one, else the on-disk one.
        fn config(&self) -> Result<DenseConfig, McpError> {
            match &self.config {
                Some(c) => Ok(c.clone()),
                None => DenseConfig::load().map_err(internal),
            }
        }

        fn tracker(&self) -> Result<Tracker> {
            match &self.db {
                Some(p) => Tracker::open_at(p),
                None => Tracker::open(),
            }
        }

        /// Record a savings row best-effort: a ledger failure must never fail the tool call.
        fn record(&self, r: &Record) {
            if let Ok(tracker) = self.tracker() {
                let _ = tracker.record(r);
            }
        }
    }

    #[tool_router]
    impl LlmtrimServer {
        #[tool(
            description = "Compress an LLM request body and report the token savings. Pass the raw request JSON; get back the compressed request in the same shape plus before/after token counts and the per-stage breakdown."
        )]
        fn llmtrim_compress(
            &self,
            Parameters(args): Parameters<CompressArgs>,
        ) -> Result<CallToolResult, McpError> {
            let result = compress_with(
                &args.request.into_body(),
                args.provider.as_deref(),
                &self.config()?,
            )?;
            self.record(&ledger_record(&result));
            ok_json(&compress_payload(&result))
        }

        #[tool(
            description = "Compress a single text blob and report the token savings. Use this to shrink one chunk (a tool output, a document) rather than a whole request. The text is wrapped in a minimal request, compressed, and the shrunk text is returned."
        )]
        fn llmtrim_compress_text(
            &self,
            Parameters(args): Parameters<CompressTextArgs>,
        ) -> Result<CallToolResult, McpError> {
            let (payload, record) = compress_text(&args.text)?;
            self.record(&record);
            ok_json(&payload)
        }

        #[tool(
            description = "Report recent savings from the local ledger: tokens trimmed and dollars saved. The same headline figures the `llmtrim status --json` dashboard shows."
        )]
        fn llmtrim_stats(
            &self,
            Parameters(_args): Parameters<StatsArgs>,
        ) -> Result<CallToolResult, McpError> {
            let tracker = self.tracker().map_err(internal)?;
            let stats = crate::monitor::stats_json(&tracker, None).map_err(internal)?;
            Ok(CallToolResult::success(vec![ContentBlock::text(stats)]))
        }
    }

    #[tool_handler(
        name = "llmtrim",
        instructions = "llmtrim compresses LLM request payloads with no extra model calls. Use llmtrim_compress for a full request body, llmtrim_compress_text for a single text blob, and llmtrim_stats to read the savings ledger."
    )]
    impl ServerHandler for LlmtrimServer {}

    /// Run the engine once against an explicit config. The seam the config-independent
    /// callers use: no file is read, so a caller that already has a config (or a test that
    /// wants a fixed one) never depends on the machine's `~/.llmtrim`. Pure: the caller
    /// records the savings. A bad provider hint or malformed request comes back as a
    /// JSON-RPC error, never a panic.
    fn compress_with(
        request: &str,
        provider: Option<&str>,
        config: &DenseConfig,
    ) -> Result<CompressResult, McpError> {
        let kind = provider
            .map(ProviderKind::from_str)
            .transpose()
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        llmtrim_core::compress_with_config(request, kind, config)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))
    }

    /// Ledger row for a `compress_text` blob: the content saving with no model attribution,
    /// since no model call happened. Matches the proxy/CLI schema (the unknown fields are
    /// `None`, exactly as the one-shot `compress` path leaves them).
    fn text_ledger_record(tokenizer: &str, exact: bool, before: usize, after: usize) -> Record {
        Record {
            provider: ProviderKind::OpenAi.as_str().to_string(),
            model: None,
            tokenizer: tokenizer.to_string(),
            exact,
            input_before: before as i64,
            input_after: after as i64,
            output_before: None,
            output_after: None,
            compress_micros: None,
            cache_read_tokens: None,
            fresh_input_tokens: None,
            cache_write_tokens: None,
            output_shaped: Some(false),
            frozen_input_tokens: Some(0),
        }
    }

    fn compress_payload(result: &CompressResult) -> Value {
        let stages: Vec<Value> = result
            .stages
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "applied": s.applied,
                    "tokens_before": s.tokens_before.0,
                    "tokens_after": s.tokens_after.0,
                    "note": s.note,
                })
            })
            .collect();
        json!({
            "request_json": result.request_json,
            "provider": result.provider.as_str(),
            "model": result.model,
            "tokenizer_label": result.tokenizer_label,
            "tokenizer_exact": result.tokenizer_exact,
            "input_tokens_before": result.input_tokens_before.0,
            "input_tokens_after": result.input_tokens_after.0,
            // Signed: `output_control` (Stage F) can add a terse-output instruction that grows
            // the input to buy a larger output saving, so this goes negative on small requests.
            // `output_shaped` below says when that tradeoff is in play.
            "tokens_saved": result.input_tokens_before.0 as i64 - result.input_tokens_after.0 as i64,
            "frozen_input_tokens": result.frozen_input_tokens.0,
            "output_shaped": result.output_shaped,
            "stages": stages,
        })
    }

    /// Shrink a single text blob. The blob is wrapped in a one-message OpenAI request only so
    /// the engine has something to operate on; we then run a **content-only** config (the
    /// lossless `safe` preset) so the request-envelope stages never fire: `output_control`
    /// would inject an instruction meant for a model answering a prompt, and `cache` a
    /// `prompt_cache_key` for an API call — neither applies to a bare blob, and the caller
    /// only gets the text back. The reported token counts are of the text itself (in vs out),
    /// not the synthetic wrapper, so the numbers describe exactly what's returned. Pure: the
    /// caller records the returned `Record`.
    fn compress_text(text: &str) -> Result<(Value, Record), McpError> {
        let body = json!({
            "model": TEXT_WRAP_MODEL,
            "messages": [{ "role": "user", "content": text }],
        })
        .to_string();
        let config = DenseConfig::preset("safe").expect("built-in preset");
        let result = llmtrim_core::compress_with_config(&body, Some(ProviderKind::OpenAi), &config)
            .map_err(internal)?;
        let out = user_content(&result.request_json);

        let counter = tokenizer::counter_for(ProviderKind::OpenAi, Some(TEXT_WRAP_MODEL))
            .map_err(internal)?;
        let before = counter.count(text);
        let after = counter.count(&out);

        let record = text_ledger_record(counter.label(), counter.is_exact(), before, after);
        let payload = json!({
            "text": out,
            "input_tokens_before": before,
            "input_tokens_after": after,
            "tokens_saved": before as i64 - after as i64,
        });
        Ok((payload, record))
    }

    /// Mirror the ledger row the CLI `compress` path writes, so MCP-driven traffic shows
    /// up in `llmtrim status`/monitor identically — output/cache/timing fields are unknown
    /// here (no upstream round-trip), exactly as in the one-shot CLI.
    fn ledger_record(result: &CompressResult) -> Record {
        Record {
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
        }
    }

    /// Pull the first user message's text back out of a compressed request, for
    /// `llmtrim_compress_text`. Content may be a plain string or an array of typed blocks
    /// (any provider, any language); concatenate the text parts. Falls back to the whole
    /// compressed JSON if the shape is unexpected.
    fn user_content(request_json: &str) -> String {
        let parsed: Value = match serde_json::from_str(request_json) {
            Ok(v) => v,
            Err(_) => return request_json.to_string(),
        };
        let Some(msg) = parsed
            .get("messages")
            .and_then(Value::as_array)
            .and_then(|m| {
                m.iter()
                    .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            })
        else {
            return request_json.to_string();
        };
        match msg.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(blocks)) => {
                let text: Vec<&str> = blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(Value::as_str))
                    .collect();
                if text.is_empty() {
                    request_json.to_string()
                } else {
                    text.join("")
                }
            }
            _ => request_json.to_string(),
        }
    }

    fn ok_json(payload: &Value) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            payload.to_string(),
        )]))
    }

    fn internal(e: anyhow::Error) -> McpError {
        McpError::internal_error(e.to_string(), None)
    }

    /// Serve the MCP protocol over stdio until the client disconnects. The async runtime
    /// stays confined here so the command dispatch in `main.rs` stays synchronous, like the
    /// proxy's `serve`. A single stdio client needs no parallelism, so this uses a
    /// current-thread runtime (the proxy, fielding many connections, uses multi-thread).
    pub fn run() -> Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to start the MCP runtime")?;
        rt.block_on(async {
            let service = server()
                .serve(stdio())
                .await
                .context("failed to start the MCP server")?;
            service.waiting().await.context("MCP server error")?;
            Ok(())
        })
    }

    // ── `llmtrim mcp install`: register the server with a client ────────────────────────

    /// The MCP-client config block for the llmtrim server, for clients configured by hand
    /// (Cursor, custom agents). The server is launched as `llmtrim mcp`.
    fn client_config_json() -> String {
        serde_json::to_string_pretty(&json!({
            "mcpServers": { "llmtrim": { "command": "llmtrim", "args": ["mcp"] } }
        }))
        .expect("static JSON serializes")
    }

    /// The `claude` CLI argv that registers the server at user scope. Kept separate so it can
    /// be asserted in tests without spawning the real CLI.
    fn claude_add_args() -> Vec<&'static str> {
        vec![
            "mcp", "add", "llmtrim", "-s", "user", "--", "llmtrim", "mcp",
        ]
    }

    /// Run a `claude` subcommand. `Ok(None)` means the CLI isn't installed (spawn failed with
    /// not-found); `Ok(Some(status))` carries its exit status.
    /// Run a `claude` subcommand. `Ok(None)` means the CLI isn't installed (spawn failed with
    /// not-found); `Ok(Some(success))` is whether it exited zero. This is the only real-IO
    /// part of install; the orchestration in [`install_with`] takes it as a parameter so it
    /// can be tested without spawning a process or touching the user's Claude config.
    fn run_claude(args: &[&str]) -> Result<Option<bool>> {
        use std::process::Command;
        match Command::new("claude").args(args).output() {
            Ok(out) => Ok(Some(out.status.success())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).context("failed to run the `claude` CLI"),
        }
    }

    /// Register the llmtrim MCP server with the user's client. Today that means Claude Code
    /// via its own `claude mcp add` CLI (which owns the config file, so we don't hand-edit
    /// it); idempotent, with `--force` to reinstall a stale entry. Any other client is served
    /// the config block to paste. `--print` skips all writes and just emits that block.
    pub fn install(print: bool, force: bool) -> Result<()> {
        install_with(print, force, run_claude)
    }

    /// `install` with the `claude` runner injected (see [`run_claude`]).
    fn install_with(
        print: bool,
        force: bool,
        run: impl Fn(&[&str]) -> Result<Option<bool>>,
    ) -> Result<()> {
        if print {
            println!("{}", client_config_json());
            return Ok(());
        }

        // `claude mcp get` exits non-zero when the server is absent; `None` means no CLI on
        // PATH, so fall back to the paste-this-config path.
        let present = match run(&["mcp", "get", "llmtrim"])? {
            None => {
                eprintln!(
                    "No `claude` CLI found on PATH. Paste this into your MCP client's config:\n"
                );
                println!("{}", client_config_json());
                return Ok(());
            }
            Some(found) => found,
        };

        if present && !force {
            println!(
                "llmtrim is already registered with Claude Code (`llmtrim mcp install --force` to reinstall)."
            );
            return Ok(());
        }
        if present && force {
            let _ = run(&["mcp", "remove", "llmtrim", "-s", "user"])?;
        }

        match run(&claude_add_args())? {
            Some(true) => {
                println!(
                    "Registered llmtrim with Claude Code (user scope). Restart the client to pick it up."
                );
                Ok(())
            }
            Some(false) => anyhow::bail!("`claude mcp add` failed"),
            None => anyhow::bail!("the `claude` CLI vanished between checks"),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn req() -> String {
            json!({
                "model": "gpt-4o",
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant.   " },
                    { "role": "user", "content": "Hello   world\n\n\nthis  has   redundant     whitespace." }
                ]
            })
            .to_string()
        }

        #[test]
        fn compress_maps_result_to_payload() {
            let result = llmtrim_core::compress_with_config(&req(), None, &DenseConfig::lossless())
                .expect("compress should succeed on a valid request");
            let payload = compress_payload(&result);

            // Every documented field is present and correctly typed.
            assert!(payload["request_json"].is_string());
            assert_eq!(payload["provider"], "openai");
            assert_eq!(payload["model"], "gpt-4o");
            assert!(payload["tokenizer_label"].is_string());
            assert!(payload["tokenizer_exact"].as_bool().unwrap());
            assert!(payload["frozen_input_tokens"].as_u64().is_some());
            assert_eq!(payload["output_shaped"], false); // lossless config shapes nothing
            let before = payload["input_tokens_before"].as_u64().unwrap();
            let after = payload["input_tokens_after"].as_u64().unwrap();
            assert!(before > 0);
            assert!(after <= before);
            assert_eq!(
                payload["tokens_saved"].as_i64().unwrap(),
                before as i64 - after as i64
            );
            // Per-stage report carries name + before/after for each stage.
            let stages = payload["stages"].as_array().expect("stages array");
            assert!(!stages.is_empty());
            assert!(stages.iter().all(|s| s["name"].is_string()
                && s["tokens_before"].as_u64().is_some()
                && s["tokens_after"].as_u64().is_some()));
        }

        #[test]
        fn output_shaped_request_reports_signed_negative_savings() {
            // A tiny request with output shaping on: Stage F injects a terse-output
            // instruction that grows the input to buy an output saving, so tokens_saved goes
            // negative and output_shaped flags the tradeoff.
            let tiny =
                json!({ "model": "gpt-4o", "messages": [{ "role": "user", "content": "hi" }] })
                    .to_string();
            let shaped = DenseConfig {
                output_control: true,
                ..DenseConfig::lossless()
            };
            let result =
                llmtrim_core::compress_with_config(&tiny, Some(ProviderKind::OpenAi), &shaped)
                    .expect("compress should succeed");
            let payload = compress_payload(&result);

            assert_eq!(payload["output_shaped"], true);
            assert!(
                payload["tokens_saved"].as_i64().unwrap() < 0,
                "shaping a tiny request grows the input; tokens_saved must be honest about it"
            );
        }

        #[test]
        fn ledger_records_match_the_proxy_schema() {
            // Full-request record carries the model and the result's token counts.
            let result = llmtrim_core::compress_with_config(&req(), None, &DenseConfig::lossless())
                .expect("compress should succeed");
            let rec = ledger_record(&result);
            assert_eq!(rec.provider, "openai");
            assert_eq!(rec.model.as_deref(), Some("gpt-4o"));
            assert_eq!(rec.input_before, result.input_tokens_before.0 as i64);
            assert_eq!(rec.input_after, result.input_tokens_after.0 as i64);
            assert!(rec.output_after.is_none() && rec.compress_micros.is_none());

            // Blob record has no model attribution (no model call happened).
            let blob = text_ledger_record("tiktoken", true, 100, 60);
            assert_eq!(blob.provider, "openai");
            assert_eq!(blob.model, None);
            assert_eq!(blob.input_before, 100);
            assert_eq!(blob.input_after, 60);
            assert_eq!(blob.output_shaped, Some(false));
        }

        #[test]
        fn bad_provider_is_invalid_params_not_panic() {
            let config = DenseConfig::preset("auto").expect("built-in preset");
            let err = compress_with(&req(), Some("not-a-provider"), &config).unwrap_err();
            assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }

        #[test]
        fn malformed_request_is_an_error() {
            // A fixed config, not `DenseConfig::load()`: this test is about malformed JSON,
            // not config loading, so it must not read (or fail on) the machine's config file.
            let config = DenseConfig::preset("auto").expect("built-in preset");
            let err = compress_with("{ not json", None, &config).unwrap_err();
            assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }

        #[test]
        fn compress_text_reports_blob_level_deltas_and_shrinks() {
            // An exact duplicate line: the lossless `safe` config collapses it via dedup.
            let blob =
                "the quick brown fox jumps\nthe quick brown fox jumps\nfoo bar baz qux quux corge";
            let (payload, record) = compress_text(blob).expect("compress_text should succeed");

            let before = payload["input_tokens_before"].as_u64().unwrap();
            let after = payload["input_tokens_after"].as_u64().unwrap();
            assert!(before > 0);
            assert!(
                after < before,
                "safe dedup should shrink a blob with a repeated line"
            );
            assert_eq!(
                payload["tokens_saved"].as_i64().unwrap(),
                before as i64 - after as i64
            );

            // Content-only: no output-shaping instruction leaks into the returned text, and
            // the numbers describe the blob (after is far below a wrapped request's token count).
            let text = payload["text"].as_str().unwrap();
            assert!(!text.to_lowercase().contains("be concise"));
            assert!(
                after < 40,
                "reported tokens are the blob's, not the wrapper's"
            );

            // The ledger row mirrors the blob-level numbers, no model attribution.
            assert_eq!(record.model, None);
            assert_eq!(record.input_before, before as i64);
            assert_eq!(record.input_after, after as i64);
        }

        #[test]
        fn user_content_handles_string_and_blocks() {
            let s = json!({ "messages": [{ "role": "user", "content": "café ☕ 日本語" }] })
                .to_string();
            assert_eq!(user_content(&s), "café ☕ 日本語");

            let blocks = json!({
                "messages": [{ "role": "user", "content": [
                    { "type": "text", "text": "part one " },
                    { "type": "text", "text": "part two" }
                ] }]
            })
            .to_string();
            assert_eq!(user_content(&blocks), "part one part two");
        }

        #[test]
        fn request_arg_accepts_both_object_and_string() {
            // An agent (and the MCP Inspector) passes `request` as a JSON object; a stricter
            // client may stringify it. Both must reduce to the same engine input.
            let body =
                json!({ "model": "gpt-4o", "messages": [{ "role": "user", "content": "hi" }] });

            let from_obj: CompressArgs =
                serde_json::from_value(json!({ "request": body })).expect("object form");
            let from_str: CompressArgs =
                serde_json::from_value(json!({ "request": body.to_string() }))
                    .expect("string form");

            let obj_body = from_obj.request.into_body();
            assert_eq!(obj_body, from_str.request.into_body());
            // And it round-trips to a request the engine accepts.
            assert!(
                serde_json::from_str::<serde_json::Value>(&obj_body)
                    .unwrap()
                    .get("messages")
                    .is_some()
            );
        }

        #[test]
        fn install_config_and_argv_launch_the_server() {
            // The paste-this-config block and the claude argv must both launch `llmtrim mcp`,
            // matching the command MCP clients spawn.
            let cfg: serde_json::Value =
                serde_json::from_str(&client_config_json()).expect("valid JSON");
            assert_eq!(cfg["mcpServers"]["llmtrim"]["command"], "llmtrim");
            assert_eq!(cfg["mcpServers"]["llmtrim"]["args"][0], "mcp");

            let argv = claude_add_args();
            assert_eq!(&argv[..4], &["mcp", "add", "llmtrim", "-s"]);
            // Everything after `--` is the launch command, and it is `llmtrim mcp`.
            let sep = argv.iter().position(|a| *a == "--").expect("-- separator");
            assert_eq!(&argv[sep + 1..], &["llmtrim", "mcp"]);
        }

        use std::cell::RefCell;
        use std::rc::Rc;

        type Calls = Rc<RefCell<Vec<String>>>;

        // A fake `claude` runner: appends each subcommand it's asked to run to `log` and
        // replies from a queue of canned results, so install's branches are tested without
        // spawning anything.
        fn fake_runner(
            log: Calls,
            replies: Vec<Result<Option<bool>>>,
        ) -> impl Fn(&[&str]) -> Result<Option<bool>> {
            let replies = RefCell::new(replies.into_iter());
            move |args: &[&str]| {
                log.borrow_mut().push(args.join(" "));
                replies.borrow_mut().next().expect("a canned reply")
            }
        }

        #[test]
        fn install_print_writes_nothing() {
            // print mode must never invoke the runner.
            let run = |_: &[&str]| -> Result<Option<bool>> { panic!("runner must not be called") };
            install_with(true, false, run).expect("print mode succeeds");
        }

        #[test]
        fn install_without_claude_cli_falls_back() {
            let calls: Calls = Rc::default();
            install_with(false, false, fake_runner(calls.clone(), vec![Ok(None)]))
                .expect("fallback succeeds");
            assert_eq!(*calls.borrow(), vec!["mcp get llmtrim"]); // probed, then gave up
        }

        #[test]
        fn install_is_idempotent_when_already_present() {
            let calls: Calls = Rc::default();
            install_with(
                false,
                false,
                fake_runner(calls.clone(), vec![Ok(Some(true))]),
            )
            .expect("already-present is a no-op success");
            assert_eq!(*calls.borrow(), vec!["mcp get llmtrim"]); // no add attempted
        }

        #[test]
        fn install_adds_when_absent() {
            let calls: Calls = Rc::default();
            install_with(
                false,
                false,
                fake_runner(calls.clone(), vec![Ok(Some(false)), Ok(Some(true))]),
            )
            .expect("registers");
            let calls = calls.borrow();
            assert_eq!(calls[0], "mcp get llmtrim");
            assert_eq!(calls[1], "mcp add llmtrim -s user -- llmtrim mcp");
        }

        #[test]
        fn install_force_reinstalls_present_entry() {
            let calls: Calls = Rc::default();
            install_with(
                false,
                true,
                fake_runner(
                    calls.clone(),
                    vec![Ok(Some(true)), Ok(Some(true)), Ok(Some(true))],
                ),
            )
            .expect("force reinstalls");
            let calls = calls.borrow();
            assert_eq!(calls[1], "mcp remove llmtrim -s user");
            assert_eq!(calls[2], "mcp add llmtrim -s user -- llmtrim mcp");
        }

        #[test]
        fn install_errors_when_add_fails() {
            let calls: Calls = Rc::default();
            let run = fake_runner(calls, vec![Ok(Some(false)), Ok(Some(false))]); // absent, add fails
            assert!(install_with(false, false, run).is_err());
        }

        #[test]
        fn user_content_falls_back_to_the_whole_json_on_odd_shapes() {
            // Each defensive branch returns the input unchanged rather than losing data.
            let malformed = "{ not json";
            assert_eq!(user_content(malformed), malformed);

            let no_user = json!({ "messages": [{ "role": "system", "content": "x" }] }).to_string();
            assert_eq!(user_content(&no_user), no_user);

            let empty_blocks =
                json!({ "messages": [{ "role": "user", "content": [{ "type": "image" }] }] })
                    .to_string();
            assert_eq!(user_content(&empty_blocks), empty_blocks);

            let odd_content =
                json!({ "messages": [{ "role": "user", "content": 42 }] }).to_string();
            assert_eq!(user_content(&odd_content), odd_content);
        }
    }
}
