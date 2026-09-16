//! `mcp` — Model Context Protocol server over stdio.
//!
//! A fourth way to reach the engine, next to the proxy, the CLI, and the language
//! bindings: any MCP client (Claude Code, Cursor, custom agents) spawns `llmtrim mcp`
//! and calls llmtrim's compression and savings stats as MCP tools. The transport is
//! JSON-RPC 2.0 over stdin/stdout — the form clients spawn by default.
//!
//! Like [`crate::serve`], the real implementation is feature-gated (`mcp`); a build
//! without it keeps the `mcp` subcommand but bails with a clear rebuild hint.

/// Which MCP client `mcp install` registers with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum McpClient {
    /// Claude Code, through its own `claude mcp add` CLI.
    Claude,
    /// DeepSeek Harness, through its user patch layer (`$DSH_HOME/cordis.patch.yml`).
    Dsh,
    /// Every client llmtrim knows how to register with.
    All,
}

#[cfg(not(feature = "mcp"))]
pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("this build has no MCP server; rebuild with `--features mcp`")
}

#[cfg(not(feature = "mcp"))]
pub fn install(_print: bool, _force: bool) -> anyhow::Result<()> {
    anyhow::bail!("this build has no MCP server; rebuild with `--features mcp`")
}

#[cfg(not(feature = "mcp"))]
pub fn install_for_client(_print: bool, _force: bool, _client: McpClient) -> anyhow::Result<()> {
    anyhow::bail!("this build has no MCP server; rebuild with `--features mcp`")
}

#[cfg(feature = "mcp")]
pub use imp::{install, install_for_client, run};

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

    use super::McpClient;
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
            outcome: None,
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
            outcome: None,
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

    /// Register the llmtrim MCP server with Claude Code via its own `claude mcp add` CLI (which
    /// owns the config file, so we don't hand-edit it); idempotent, with `--force` to reinstall a
    /// stale entry. With no `claude` CLI on PATH it falls back to printing the config block to
    /// paste. `--print` skips all writes and just emits that block.
    fn install_claude(print: bool, force: bool) -> Result<()> {
        install_with(print, force, run_claude)
    }

    /// Register the llmtrim MCP server with Claude Code. Kept at its original two-argument
    /// signature: this crate is published as a library, so widening a public function's
    /// parameters is a breaking API change the SemVer gate rejects. Other clients go through
    /// [`install_for_client`].
    pub fn install(print: bool, force: bool) -> Result<()> {
        install_for_client(print, force, McpClient::Claude)
    }

    /// Register with the requested client. Claude Code is driven through its own `claude mcp add`
    /// CLI (which owns the config file, so we don't hand-edit it); DeepSeek Harness has no such
    /// CLI, so its user patch layer is written directly. Idempotent either way, with `--force` to
    /// reinstall a stale entry. `--print` skips every write.
    pub fn install_for_client(print: bool, force: bool, client: McpClient) -> Result<()> {
        install_for_client_with(print, force, client, install_claude, install_dsh)
    }

    /// [`install_for_client`] with both client halves injected, so the `All` branch's ordering and
    /// reporting can be tested without touching a real client's config.
    fn install_for_client_with(
        print: bool,
        force: bool,
        client: McpClient,
        claude: impl Fn(bool, bool) -> Result<()>,
        dsh: impl Fn(bool, bool, bool) -> Result<()>,
    ) -> Result<()> {
        match client {
            McpClient::Claude => claude(print, force),
            McpClient::Dsh => dsh(print, force, true),
            McpClient::All => {
                if print {
                    println!("# Claude Code");
                    claude(true, force)?;
                    println!("\n# DeepSeek Harness");
                    return dsh(true, force, false);
                }
                // Bind both before combining: `Result::and` takes its argument eagerly, so the DSH
                // half runs even when the Claude half fails. Written longhand because a refactor to
                // `and_then` would silently skip it.
                let claude_result = claude(false, force);
                let dsh_result = dsh(false, force, false);
                if claude_result.is_err() && dsh_result.is_ok() {
                    eprintln!(
                        "llmtrim was registered with DeepSeek Harness; the Claude Code half failed."
                    );
                }
                claude_result.and(dsh_result)
            }
        }
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

    // ── `--client dsh`: the DeepSeek Harness user patch layer ──────────────────────────

    /// Marker comments around our `- insert:` block. They are the convention documented for
    /// hand-written DSH entries, which is also what lets a re-run recognize a block the user
    /// wrote by hand instead of appending a twin.
    const DSH_BEGIN: &str = "# --- llmtrim MCP server ---";
    const DSH_END: &str = "# --- end llmtrim MCP server ---";

    /// The row id we register under. The block text spells it literally; this is for the finder and
    /// the duplicate guard, which must recognize every spelling of the id a hand edit can produce.
    const ROW_ID: &str = "mcp-llmtrim";

    /// The patch block that registers llmtrim with DSH: a loader row mounting the bundled
    /// `@deepseek-ai/dsh-mcp-client` bridge, which re-exposes the tools as
    /// `mcp__llmtrim__llmtrim_compress`, `…_compress_text` and `…_stats`. `command: llmtrim` is
    /// the same launch command the Claude entry and the paste-this block use, and the bridge
    /// spawns it through the MCP SDK's cross-spawn, which resolves an npm `.cmd` shim on Windows.
    const DSH_BLOCK: &str = "\
# --- llmtrim MCP server ---
- insert:
  - id: mcp-llmtrim
    name: '@deepseek-ai/dsh-mcp-client'
    config:
      serverName: llmtrim
      transport: stdio
      command: llmtrim
      args:
        - mcp
# --- end llmtrim MCP server ---
";

    /// What a patch file already says about the llmtrim row.
    #[derive(Debug, PartialEq, Eq)]
    enum DshEntry {
        /// No `- insert:` block carries `id: mcp-llmtrim`.
        Absent,
        /// One does, and it launches `llmtrim mcp`.
        Current,
        /// One does, but it launches something else (a stale absolute path, another binary).
        Stale,
        /// Our row is in a shape this build must not rewrite: an `- insert:` list holding more
        /// than one entry, or more than one block carrying our id. The message names what to fix.
        /// Both are refused with or without `--force`: rewriting either would drop a sibling
        /// registration or leave the duplicate that fails DSH's boot.
        Refuse(String),
    }

    /// Split a patch file into lines without their terminators, and report whether it used
    /// CRLF so the write can restore the user's line ending. A trailing newline does not produce
    /// a trailing empty entry. Pure.
    fn split_patch(content: &str) -> (Vec<String>, bool) {
        let crlf = content.contains("\r\n");
        let mut lines: Vec<String> = content
            .split('\n')
            .map(|l| l.strip_suffix('\r').unwrap_or(l).to_string())
            .collect();
        if lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        (lines, crlf)
    }

    /// The row id on a line, tolerating the spellings a hand edit or another writer produces: an
    /// optional list marker, a quoted or spaced key (`id`, `"id"`, `id :`), quotes around the
    /// value, and a flow mapping (`{id: mcp-llmtrim, …}`). Returns `None` for any other key, so
    /// `identifier:` or a `config:` line cannot be mistaken for a row id.
    fn row_id(line: &str) -> Option<String> {
        let rest = line
            .trim()
            .trim_start_matches("- ")
            .trim_start_matches('{')
            .trim();
        let (key, value) = rest.split_once(':')?;
        if key.trim().trim_matches(['\'', '"']) != "id" {
            return None;
        }
        let value = value.split(',').next().unwrap_or(value).trim();
        let value = value.trim_matches(['\'', '"']).trim();
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    }

    /// Does this line mention the given row id, in a block row or inside a flow mapping?
    fn line_mentions_id(line: &str, want: &str) -> bool {
        if row_id(line).as_deref() == Some(want) {
            return true;
        }
        line.match_indices('{')
            .any(|(i, _)| row_id(&line[i + 1..]).as_deref() == Some(want))
    }

    /// Is this line a list item (`- …`)? An `id:` key nested inside another item's mapping is not,
    /// so a foreign config that happens to contain `id: mcp-llmtrim` — under an `env:` mapping,
    /// say — is never mistaken for our row.
    fn is_list_item(line: &str) -> bool {
        let t = line.trim_start();
        t == "-" || t.starts_with("- ")
    }

    /// This line's indentation.
    fn indent_of(line: &str) -> usize {
        line.len() - line.trim_start().len()
    }

    /// The item level of an `insert:` list: the shallowest indentation at which a list item
    /// appears. Items deeper than that — an `args:` entry, a nested list inside an item's config —
    /// are not entries of the list.
    fn item_indent(block: &[String]) -> Option<usize> {
        block
            .iter()
            .map(String::as_str)
            .filter(|l| is_list_item(l))
            .map(indent_of)
            .min()
    }

    /// Find our insert block: a column-0 `- insert:` whose list holds a row with our id, running to
    /// the closing marker or the first non-indented line after it. A column-0 `- id` row on its own
    /// is a patch override — the row a settings page or a hand edit uses to enable or disable an
    /// entry — not a registration. Returns the block's line range (BEGIN marker included) so the
    /// caller rewrites exactly what it found, or [`DshEntry::Refuse`] when rewriting is unsafe.
    /// Pure.
    fn dsh_entry(lines: &[String]) -> (DshEntry, Option<std::ops::Range<usize>>) {
        let mut i = 0;
        let mut hits: Vec<(usize, usize, usize)> = Vec::new();
        while i < lines.len() {
            if lines[i] != "- insert:" {
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < lines.len() {
                let line = &lines[end];
                if line.trim() == DSH_END {
                    end += 1;
                    break;
                }
                // Empty and indented lines are ours to examine; the first non-indented non-empty
                // line that is not our END marker — another row, or a comment — starts what comes
                // next, so the range stops before it and a rewrite can never swallow it.
                if line.len() == line.trim_start().len() && !line.trim().is_empty() {
                    break;
                }
                end += 1;
            }
            let block = &lines[i..end];
            // The list body, not the `- insert:` header: the header is itself a list item (at indent
            // 0), so including it would make the item level 0 and no nested row could ever match.
            let Some(level) = item_indent(&lines[i + 1..end]) else {
                i = end;
                continue;
            };
            // Ours only if a list item *at the list's item level* carries our id. A deeper `id:` key
            // is configuration data in someone else's entry, not a row.
            let ours = block.iter().map(String::as_str).any(|l| {
                is_list_item(l) && indent_of(l) == level && row_id(l).as_deref() == Some(ROW_ID)
            });
            if !ours {
                i = end;
                continue;
            }
            // More than one item at the list's level means we do not solely own the list: rewriting
            // it would delete a sibling registration. Counting at the list's item level — not at our
            // own row's indent — is what catches a sibling written at another indent.
            let entries = block
                .iter()
                .map(String::as_str)
                .filter(|l| is_list_item(l) && indent_of(l) == level)
                .count();
            if hits.is_empty() && entries > 1 {
                return (
                    DshEntry::Refuse(format!(
                        "the `- insert:` list at line {} holds more than one entry, so llmtrim does not \
                         solely own it; rewriting it would drop the other registrations. Move our row \
                         into its own `- insert:` list by hand and re-run — `--force` does not rewrite \
                         this.",
                        i + 1
                    )),
                    None,
                );
            }
            let start = if i > 0 && lines[i - 1].trim() == DSH_BEGIN {
                i - 1
            } else {
                i
            };
            hits.push((i, start, end));
            i = end;
        }

        if hits.is_empty() {
            return (DshEntry::Absent, None);
        }
        if hits.len() > 1 {
            let numbered = hits
                .iter()
                .map(|(insert, _, _)| (*insert + 1).to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return (
                DshEntry::Refuse(format!(
                    "{} `- insert:` blocks carry `{ROW_ID}` (lines {numbered}); delete one by hand \
                     and re-run — `--force` does not rewrite this, because two rows for one id fail \
                     at DSH boot.",
                    hits.len()
                )),
                None,
            );
        }
        let (insert, start, end) = hits[0];
        let state = if launches_llmtrim(&lines[insert..end]) {
            DshEntry::Current
        } else {
            DshEntry::Stale
        };
        (state, Some(start..end))
    }

    /// Does the file mention our row id in a shape [`dsh_entry`] does not claim? It reads only
    /// column-0 `- insert:` blocks, so a group-nested `insert:` or a flow-style `- insert: [{…}]`
    /// would look absent and get a duplicate appended — and a second row for one id fails at DSH
    /// boot. A top-level override row (`- id: …` or `- {id: …, disabled: …}`), which a settings page
    /// or a hand edit writes to enable or disable an entry, is not a registration and is
    /// deliberately not matched: appending beside it is correct. Commented lines are not part of
    /// the file.
    fn has_unparsed_llmtrim_row(lines: &[String]) -> bool {
        lines.iter().any(|l| {
            if l.trim().starts_with('#') || !is_list_item(l) || !line_mentions_id(l, ROW_ID) {
                return false;
            }
            let indented = l.len() != l.trim_start().len();
            let override_row = !indented && row_id(l).is_some();
            !override_row
        })
    }

    /// Does an insert block register llmtrim the way we would? The file is user-editable, so the
    /// same registration is spelled several ways — DSH's own profile layer writes quoted scalars
    /// and flow-style arg lists. A command that names a path is deliberately NOT this
    /// registration: we write the bare name so an upgrade cannot leave a stale absolute path, and
    /// `--force` is what replaces one. A foreign `serverName` is not ours either, since the tools
    /// would appear under a different namespace.
    fn launches_llmtrim(block: &[String]) -> bool {
        let value = |key: &str| -> Option<String> {
            block
                .iter()
                .find_map(|l| l.trim().strip_prefix(key))
                .map(|v| v.trim().trim_matches(['\'', '"']).trim().to_string())
        };
        let bare = |v: Option<String>| v.is_some_and(|s| s.eq_ignore_ascii_case("llmtrim"));
        bare(value("command:")) && bare(value("serverName:")) && has_mcp_arg(block)
    }

    /// Is `mcp` in the block's argument list, in either the block form we write or the flow form
    /// a hand-edited file may use (`args: [mcp]`, `args: ["mcp"]`)?
    fn has_mcp_arg(block: &[String]) -> bool {
        if block.iter().any(|l| l.trim() == "- mcp") {
            return true;
        }
        block.iter().any(|l| {
            l.trim()
                .strip_prefix("args:")
                .map(|v| v.trim().trim_matches(['[', ']']))
                .is_some_and(|v| {
                    v.trim()
                        .trim_matches(['\'', '"'])
                        .eq_ignore_ascii_case("mcp")
                })
        })
    }

    /// What [`install_dsh_at`] did, so the caller can report it.
    #[derive(Debug, PartialEq, Eq)]
    enum DshOutcome {
        /// The file already registers us with the command we would write.
        AlreadyRegistered,
        /// The block was appended to the patch file.
        Written,
        /// An existing block was rewritten in place.
        Rewritten,
        /// An existing block differs and `--force` was not given.
        Stale,
    }

    /// Append or refresh the llmtrim row in a DSH user patch layer. `path` is injected so tests
    /// point at a temp file. A current entry is never written back — the file stays
    /// byte-identical — because a second `- insert:` row for the same id mounts a second
    /// mcp-client fiber on a `serverName` that is reserved while one is live. Only the block we
    /// found is replaced; every other byte of the user's file is preserved. A shape we cannot
    /// safely rewrite is refused before anything is written.
    fn install_dsh_at(path: &std::path::Path, force: bool) -> Result<DshOutcome> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(e).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        let (mut lines, crlf) = split_patch(&content);
        let (state, range) = dsh_entry(&lines);
        let block: Vec<String> = DSH_BLOCK.lines().map(str::to_string).collect();
        let outcome = match state {
            DshEntry::Current => DshOutcome::AlreadyRegistered,
            DshEntry::Stale if !force => DshOutcome::Stale,
            DshEntry::Stale => match range {
                Some(range) => {
                    lines.splice(range, block);
                    DshOutcome::Rewritten
                }
                // Unreachable: a `Stale` verdict always comes with the block it found. Appending
                // is the safe reading of "we could not locate it".
                None => {
                    append_block(&mut lines, block);
                    DshOutcome::Written
                }
            },
            DshEntry::Absent => {
                if has_unparsed_llmtrim_row(&lines) {
                    anyhow::bail!(
                        "{} already mentions `mcp-llmtrim` in a shape this build does not parse; \
                         refusing to add a second row for one server (duplicate rows fail at DSH \
                         boot). Remove that row by hand, then re-run.",
                        path.display()
                    );
                }
                append_block(&mut lines, block);
                DshOutcome::Written
            }
            DshEntry::Refuse(reason) => anyhow::bail!("{}: {reason}", path.display()),
        };

        if matches!(outcome, DshOutcome::AlreadyRegistered | DshOutcome::Stale) {
            return Ok(outcome);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let eol = if crlf { "\r\n" } else { "\n" };
        let mut text = lines.join(eol);
        text.push_str(eol);
        // Stage beside the real file: `fs::rename` over a symlinked patch layer would replace the
        // link itself and detach a dotfile-managed file from its manager. The messages keep naming
        // the user-facing path.
        let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let file_name = target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("cordis.patch.yml");
        let tmp = target.with_file_name(format!("{file_name}.llmtrim-tmp"));
        let staged = || -> Result<()> {
            #[cfg(unix)]
            {
                use std::io::Write as _;
                use std::os::unix::fs::OpenOptionsExt as _;
                // `mode` applies only when the file is *created*, and a stale temp from a killed
                // run could be wider, so drop it first and let 0600 take effect.
                let _ = std::fs::remove_file(&tmp);
                std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(&tmp)
                    .and_then(|mut f| f.write_all(text.as_bytes()))
                    .with_context(|| format!("failed to write {}", tmp.display()))?;
            }
            #[cfg(not(unix))]
            std::fs::write(&tmp, &text)
                .with_context(|| format!("failed to write {}", tmp.display()))?;
            // Keep the original's permissions: a patch layer can carry tokens, and the temp file
            // would otherwise widen them. An absent file (first install) has none to copy.
            if let Ok(meta) = std::fs::metadata(&target) {
                std::fs::set_permissions(&tmp, meta.permissions())
                    .with_context(|| format!("failed to set permissions on {}", tmp.display()))?;
            }
            // The old content survives until this rename succeeds, so a crash leaves the user's
            // patch file intact.
            std::fs::rename(&tmp, &target)
                .with_context(|| format!("failed to replace {}", path.display()))?;
            Ok(())
        };
        if let Err(e) = staged() {
            // Never leave a `.llmtrim-tmp` beside the user's patch layer.
            #[cfg(windows)]
            clear_read_only(&tmp);
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(outcome)
    }

    /// Add the block to the end of a patch file, keeping one blank line between it and whatever
    /// was there.
    fn append_block(lines: &mut Vec<String>, block: Vec<String>) {
        if !lines.is_empty() && !lines.last().is_some_and(|l| l.trim().is_empty()) {
            lines.push(String::new());
        }
        lines.extend(block);
    }

    /// Clear the Windows read-only attribute so a stale temp file can be unlinked. Windows-only:
    /// here `set_readonly(false)` clears `FILE_ATTRIBUTE_READONLY`, which is exactly what makes a
    /// read-only temp removable, and the lint's Unix concern ("world writable") cannot arise.
    #[cfg(windows)]
    #[allow(clippy::permissions_set_readonly_false)]
    fn clear_read_only(path: &std::path::Path) {
        if let Ok(meta) = std::fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_readonly(false);
            let _ = std::fs::set_permissions(path, perms);
        }
    }

    /// The DSH user patch layer, plus whether a DSH install looks present. `$DSH_HOME` counts as
    /// present on its own — the user pointed us at it; the default home must show `profiles/`, so
    /// we never create `~/.dsh` for an app that is not installed. The home is resolved the way DSH
    /// itself resolves it: `USERPROFILE` first on Windows, where a Git Bash `HOME=/c/Users/…` would
    /// send us to `C:\c\Users\…`, and `HOME` first elsewhere. Note `crate::daemon::home_dir()` is
    /// *not* this — it is llmtrim's own state dir (`$LLMTRIM_HOME`/`~/.llmtrim`).
    fn dsh_patch_path() -> Result<(PathBuf, bool)> {
        if let Some(dir) = std::env::var_os("DSH_HOME").filter(|v| !v.is_empty()) {
            return Ok(dsh_patch_path_from_dsh_home(&dir));
        }
        let home = if cfg!(windows) {
            std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))
        } else {
            std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))
        }
        .context("neither HOME nor USERPROFILE is set")?;
        Ok(dsh_patch_path_from_home(std::path::Path::new(&home)))
    }

    /// `$DSH_HOME/cordis.patch.yml`. An explicit home counts as installed on its own — the user
    /// pointed us at it.
    fn dsh_patch_path_from_dsh_home(dir: &std::ffi::OsStr) -> (PathBuf, bool) {
        (PathBuf::from(dir).join("cordis.patch.yml"), true)
    }

    /// `~/.dsh/cordis.patch.yml`, present only when the install looks real (`profiles/` exists),
    /// so we never create `~/.dsh` for an app that is not installed. Pure, so the heuristic is
    /// testable without touching the process environment.
    fn dsh_patch_path_from_home(home: &std::path::Path) -> (PathBuf, bool) {
        let dir = home.join(".dsh");
        let installed = dir.join("profiles").is_dir();
        (dir.join("cordis.patch.yml"), installed)
    }

    /// Report that there is no DSH here: an error when the user named DSH (`--client dsh`), a
    /// note when `--client all` merely tried it. Split out so both branches are testable without
    /// reading the machine's real home.
    fn dsh_absence(required: bool, path: &std::path::Path) -> Result<()> {
        let msg = format!(
            "No DeepSeek Harness install found at {} — nothing written (set DSH_HOME to point at one).",
            path.display()
        );
        if required {
            anyhow::bail!("{msg}");
        }
        println!("{msg}");
        Ok(())
    }

    /// Register with DeepSeek Harness by writing its user patch layer. Idempotent; `--force`
    /// rewrites a stale block. DSH hot-reloads both patch layers on the web profile, so the tools
    /// appear without a DSH restart.
    fn install_dsh(print: bool, force: bool, required: bool) -> Result<()> {
        if print {
            print!("{DSH_BLOCK}");
            return Ok(());
        }
        let (path, installed) = dsh_patch_path()?;
        if !installed {
            return dsh_absence(required, &path);
        }
        match install_dsh_at(&path, force)? {
            DshOutcome::AlreadyRegistered => println!(
                "llmtrim is already registered with DeepSeek Harness ({}).",
                path.display()
            ),
            DshOutcome::Written => println!(
                "Registered llmtrim with DeepSeek Harness ({}). DSH hot-reloads the patch layer, so no restart is needed.",
                path.display()
            ),
            DshOutcome::Rewritten => {
                println!(
                    "Rewrote the llmtrim entry in {} (`--force`).",
                    path.display()
                )
            }
            DshOutcome::Stale => anyhow::bail!(
                "{} already has an llmtrim entry that differs from the canonical `serverName: llmtrim` + `command: llmtrim` + `args: [mcp]`; re-run with `--force` to rewrite it.",
                path.display()
            ),
        }
        Ok(())
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
        fn dsh_block_registers_the_bridge_over_stdio() {
            // The markers are what lets a re-run recognize a hand-written block, so they are part
            // of the contract, not decoration.
            assert!(DSH_BLOCK.starts_with(DSH_BEGIN));
            assert!(DSH_BLOCK.trim_end().ends_with(DSH_END));
            assert!(DSH_BLOCK.contains("  - id: mcp-llmtrim\n"));
            assert!(DSH_BLOCK.contains("    name: '@deepseek-ai/dsh-mcp-client'\n"));
            assert!(DSH_BLOCK.contains("      serverName: llmtrim\n"));
            assert!(DSH_BLOCK.contains("      transport: stdio\n"));
            assert!(DSH_BLOCK.contains("      command: llmtrim\n"));
            assert!(DSH_BLOCK.contains("        - mcp\n"));
        }

        fn state_of(content: &str) -> DshEntry {
            dsh_entry(&split_patch(content).0).0
        }

        #[test]
        fn dsh_entry_finds_only_insert_blocks() {
            assert_eq!(state_of(""), DshEntry::Absent);
            // A column-0 row is a top-level patch override (enable/disable), not a registration.
            assert_eq!(
                state_of("- id: mcp-llmtrim\n  disabled: true\n"),
                DshEntry::Absent
            );
            assert_eq!(state_of(DSH_BLOCK), DshEntry::Current);
            let stale = DSH_BLOCK.replace("command: llmtrim", "command: C:\\old\\llmtrim.exe");
            assert_eq!(state_of(&stale), DshEntry::Stale);
            // A file written with Windows line endings scans identically.
            assert_eq!(
                state_of(&DSH_BLOCK.replace('\n', "\r\n")),
                DshEntry::Current
            );
        }

        #[test]
        fn dsh_entry_ignores_other_servers() {
            let other = "\
# --- playwright MCP server ---
- insert:
  - id: mcp-playwright
    name: '@deepseek-ai/dsh-mcp-client'
    config:
      serverName: playwright
      transport: stdio
      command: npx
      args:
        - -y
        - '@playwright/mcp'
# --- end playwright MCP server ---
";
            assert_eq!(state_of(other), DshEntry::Absent);
            // Ours after someone else's is still found, and the range points at ours.
            let both = format!("{other}\n{DSH_BLOCK}");
            let lines = split_patch(&both).0;
            let (state, range) = dsh_entry(&lines);
            assert_eq!(state, DshEntry::Current);
            // `lines[range]` would be a slice, not the element: take the range's start.
            assert_eq!(lines[range.expect("found").start].trim(), DSH_BEGIN);
        }

        #[test]
        fn dsh_entry_accepts_a_quoted_command_and_flow_args() {
            // DSH's own profile layer writes quoted scalars and flow arg lists, so the same
            // registration must not read as stale just because it is spelled that way.
            let quoted = DSH_BLOCK
                .replace("command: llmtrim", "command: 'llmtrim'")
                .replace("      args:\n        - mcp\n", "      args: [mcp]\n");
            // Pin the fixture: if that second replace ever misses, the block-form `- mcp` line keeps
            // the assertion below green and the flow form goes untested.
            assert!(quoted.contains("args: [mcp]"));
            assert!(!quoted.contains("- mcp"));
            assert_eq!(state_of(&quoted), DshEntry::Current);
        }

        #[test]
        fn dsh_entry_rejects_a_foreign_server_name() {
            // Same command, different namespace: the tools would appear as `mcp__other__…`, so this
            // is not our registration and has to be rewriteable with --force.
            let foreign = DSH_BLOCK.replace("serverName: llmtrim", "serverName: other");
            assert_eq!(state_of(&foreign), DshEntry::Stale);
        }

        #[test]
        fn dsh_guard_classifies_every_row_spelling() {
            // A guard false positive blocks an append that is correct; a false negative appends a
            // second row for one id, which fails at DSH boot. Both sides are pinned here.
            let guard = |content: &str| has_unparsed_llmtrim_row(&split_patch(content).0);
            // Not registrations: the override rows a settings page or a hand edit writes, and a
            // commented-out row.
            assert!(!guard("- id: mcp-llmtrim\n  disabled: true\n"));
            assert!(!guard("- {id: mcp-llmtrim, disabled: true}\n"));
            assert!(!guard("# - insert: [{id: mcp-llmtrim, disabled: false}]\n"));
            assert!(!guard("- id: mcp-playwright\n  disabled: true\n"));
            // `identifier:` is not the `id` key.
            assert!(!guard("identifier: mcp-llmtrim\n"));
            // Registrations `dsh_entry` does not claim, in every spelling that parses to a row:
            // refusing to append is the only safe answer.
            assert!(guard("- insert:\n    - id: mcp-llmtrim\n"));
            assert!(guard(
                "- insert:\n    - {id: mcp-llmtrim, config: {serverName: llmtrim}, disabled: false}\n"
            ));
            assert!(guard("- insert:\n    - id: \"mcp-llmtrim\"\n"));
            assert!(guard("- insert:\n    - id : mcp-llmtrim\n"));
            assert!(guard(
                "- insert: [{id: mcp-llmtrim, config: {serverName: llmtrim}}]\n"
            ));
        }

        #[test]
        fn dsh_entry_accepts_a_quoted_or_spaced_row_id() {
            // A quoted value or a space before the colon is the same row id, so a canonical block
            // spelled that way must read as already registered rather than as absent.
            for spelling in ["    - id: \"mcp-llmtrim\"\n", "    - id : mcp-llmtrim\n"] {
                let content = format!(
                    "- insert:\n{spelling}      name: '@deepseek-ai/dsh-mcp-client'\n      config:\n        serverName: llmtrim\n        command: llmtrim\n        args:\n          - mcp\n"
                );
                assert_eq!(state_of(&content), DshEntry::Current, "{content}");
                let path = temp_patch("quoted-id");
                std::fs::write(&path, &content).expect("seed");
                assert_eq!(
                    install_dsh_at(&path, false).expect("no-op"),
                    DshOutcome::AlreadyRegistered
                );
                assert_eq!(std::fs::read_to_string(&path).expect("read"), content);
            }
        }

        /// A fresh temp directory per test; the tag keeps the parallel test runner from sharing files.
        fn temp_dir_for(tag: &str) -> std::path::PathBuf {
            let dir =
                std::env::temp_dir().join(format!("llmtrim-mcp-dsh-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("temp dir");
            dir
        }

        /// A fresh patch path per test.
        fn temp_patch(tag: &str) -> std::path::PathBuf {
            temp_dir_for(tag).join("cordis.patch.yml")
        }

        #[test]
        fn dsh_install_appends_once_and_reruns_are_byte_identical() {
            let path = temp_patch("idempotent");
            let _ = std::fs::remove_file(&path);

            assert_eq!(
                install_dsh_at(&path, false).expect("first write"),
                DshOutcome::Written
            );
            let first = std::fs::read(&path).expect("file exists");

            // A second `- insert:` row for the same id would mount a second mcp-client fiber on a
            // serverName that is reserved while one is live, so a re-run must not write at all.
            assert_eq!(
                install_dsh_at(&path, false).expect("re-run"),
                DshOutcome::AlreadyRegistered
            );
            assert_eq!(std::fs::read(&path).expect("file exists"), first);
            assert_eq!(
                String::from_utf8(first)
                    .unwrap()
                    .matches("- id: mcp-llmtrim")
                    .count(),
                1
            );
        }

        #[test]
        fn dsh_install_preserves_a_hand_written_entry() {
            let path = temp_patch("handwritten");
            let before = "\
# --- llmtrim MCP server ---
- insert:
  - id: mcp-llmtrim
    name: '@deepseek-ai/dsh-mcp-client'
    config:
      serverName: llmtrim
      transport: stdio
      command: llmtrim
      args:
        - mcp
# --- end llmtrim MCP server ---

- id: mcp-playwright
  disabled: true
";
            std::fs::write(&path, before).expect("seed");

            assert_eq!(
                install_dsh_at(&path, false).expect("no-op"),
                DshOutcome::AlreadyRegistered
            );
            assert_eq!(std::fs::read_to_string(&path).expect("read"), before);
        }

        #[test]
        fn dsh_install_refuses_a_stale_entry_without_force() {
            let path = temp_patch("stale");
            let stale = DSH_BLOCK.replace("command: llmtrim", "command: C:\\old\\llmtrim.exe");
            std::fs::write(&path, &stale).expect("seed");

            assert_eq!(
                install_dsh_at(&path, false).expect("refused"),
                DshOutcome::Stale
            );
            assert_eq!(std::fs::read_to_string(&path).expect("read"), stale);
        }

        #[test]
        fn dsh_install_force_rewrites_in_place() {
            let path = temp_patch("force");
            let stale = DSH_BLOCK.replace("command: llmtrim", "command: C:\\old\\llmtrim.exe");
            std::fs::write(
                &path,
                format!("- id: keep-me\n  disabled: false\n\n{stale}"),
            )
            .expect("seed");

            assert_eq!(
                install_dsh_at(&path, true).expect("rewrite"),
                DshOutcome::Rewritten
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert_eq!(after.matches("- id: mcp-llmtrim").count(), 1);
            assert!(after.contains("      command: llmtrim\n"));
            assert!(!after.contains("C:\\old\\llmtrim.exe"));
            assert!(
                after.contains("- id: keep-me\n  disabled: false\n"),
                "the user's other rows survive"
            );
        }

        #[test]
        fn dsh_install_keeps_crlf() {
            let path = temp_patch("crlf");
            std::fs::write(&path, "- id: keep-me\r\n  disabled: false\r\n").expect("seed");

            assert_eq!(
                install_dsh_at(&path, false).expect("write"),
                DshOutcome::Written
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert!(after.contains("- id: keep-me\r\n"));
            assert!(after.contains("      command: llmtrim\r\n"));
            assert!(
                !after.replace("\r\n", "").contains('\n'),
                "no LF-only line slipped into a CRLF file"
            );
        }

        #[test]
        fn dsh_install_refuses_a_row_it_cannot_parse_instead_of_duplicating_it() {
            let path = temp_patch("unparsed");
            let grouped = "\
- id: some-group
  insert:
    - id: mcp-llmtrim
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: llmtrim
        command: llmtrim
";
            std::fs::write(&path, grouped).expect("seed");

            // Two rows for one id mount two mcp-client fibers on a reserved serverName, which fails
            // at DSH boot, so refusing is the only safe answer.
            assert!(install_dsh_at(&path, false).is_err());
            assert!(install_dsh_at(&path, true).is_err());
            assert_eq!(std::fs::read_to_string(&path).expect("read"), grouped);

            // Flow style, carrying `disabled: false`: still a registration (that value mounts it), so
            // it must be refused like the nested form above.
            let flow = "\
- insert: [{id: mcp-llmtrim, name: '@deepseek-ai/dsh-mcp-client', config: {serverName: llmtrim, command: llmtrim, args: [mcp]}, disabled: false}]
";
            let flow_path = temp_patch("unparsed-flow");
            std::fs::write(&flow_path, flow).expect("seed");
            assert!(install_dsh_at(&flow_path, false).is_err());
            assert_eq!(std::fs::read_to_string(&flow_path).expect("read"), flow);
        }

        #[test]
        fn dsh_install_appends_beside_a_toggle_row() {
            // The `disabled:` row a user's settings page writes is not a registration, so ours is
            // still appended next to it — the guard's deliberate carve-out.
            let path = temp_patch("toggle-row");
            std::fs::write(&path, "- id: mcp-llmtrim\n  disabled: true\n").expect("seed");

            assert_eq!(
                install_dsh_at(&path, false).expect("write"),
                DshOutcome::Written
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert!(after.starts_with("- id: mcp-llmtrim\n  disabled: true\n"));
            assert_eq!(after.matches("- id: mcp-llmtrim").count(), 2);
            assert!(
                after.contains("  - id: mcp-llmtrim\n"),
                "the block's nested row was appended"
            );

            // A flow toggle override without an `insert:` is not a registration either, so the
            // carve-out still applies and the append goes ahead.
            let flow_path = temp_patch("toggle-row-flow");
            std::fs::write(&flow_path, "- {id: mcp-llmtrim, disabled: true}\n").expect("seed");
            assert_eq!(
                install_dsh_at(&flow_path, false).expect("write"),
                DshOutcome::Written
            );
            let flow_after = std::fs::read_to_string(&flow_path).expect("read");
            assert!(flow_after.starts_with("- {id: mcp-llmtrim, disabled: true}\n"));
            // The id now appears twice: the user's flow toggle, and our appended nested row. The
            // substring is `id: …`, not `- id: …`, because a flow row is spelled `- {id: …}`.
            assert_eq!(flow_after.matches("id: mcp-llmtrim").count(), 2);
        }

        #[test]
        fn dsh_install_refuses_an_insert_list_it_does_not_solely_own() {
            // One `- insert:` list holding two entries: `--force` used to replace the whole list,
            // deleting the sibling registration. Order must not matter, and neither force mode may
            // touch the file.
            let ours = "  - id: mcp-llmtrim\n    name: '@deepseek-ai/dsh-mcp-client'\n    config:\n      serverName: llmtrim\n      command: llmtrim\n      args:\n        - mcp\n";
            let theirs = "  - id: mcp-playwright\n    name: '@deepseek-ai/dsh-mcp-client'\n    config:\n      serverName: playwright\n      command: npx\n";
            // Pin the fixture: a Rust `"\` continuation strips the next line's leading whitespace,
            // which would move these rows to column 0, empty the insert list and turn this test
            // into a no-op.
            assert!(ours.starts_with("  - id: mcp-llmtrim"));
            assert!(theirs.starts_with("  - id: mcp-playwright"));
            for order in ["playwright-first", "llmtrim-first"] {
                let path = temp_patch(order);
                let list = if order == "playwright-first" {
                    format!("- insert:\n{theirs}{ours}")
                } else {
                    format!("- insert:\n{ours}{theirs}")
                };
                std::fs::write(&path, &list).expect("seed");

                for force in [false, true] {
                    let err = install_dsh_at(&path, force).expect_err("must refuse");
                    assert!(
                        err.to_string().contains("more than one entry"),
                        "reason names the list: {err}"
                    );
                    assert_eq!(
                        std::fs::read_to_string(&path).expect("read"),
                        list,
                        "a refused rewrite must leave the file byte-identical"
                    );
                }
            }
        }

        #[test]
        fn dsh_install_refuses_two_blocks_carrying_our_id() {
            // The state this tool exists to prevent must not be reported as success: name both.
            let path = temp_patch("two-blocks");
            let stale = DSH_BLOCK.replace("command: llmtrim", "command: C:\\old\\llmtrim.exe");
            let content = format!("{stale}\n{DSH_BLOCK}");
            std::fs::write(&path, &content).expect("seed");

            for force in [false, true] {
                let err = install_dsh_at(&path, force).expect_err("must refuse");
                let message = err.to_string();
                assert!(message.contains("2 `- insert:` blocks"), "{message}");
                assert!(
                    message.contains("lines 2, 14"),
                    "names both blocks: {message}"
                );
                assert_eq!(std::fs::read_to_string(&path).expect("read"), content);
            }
        }

        #[test]
        fn dsh_install_rewrites_only_our_row_and_keeps_a_trailing_comment() {
            // A marker-less registration used to take everything up to the next top-level row into
            // its range, so `--force` deleted the user's trailing notes.
            let path = temp_patch("markerless");
            let content = "\
- insert:
  - id: mcp-llmtrim
    name: '@deepseek-ai/dsh-mcp-client'
    config:
      serverName: llmtrim
      command: C:\\old\\llmtrim.exe
      args:
        - mcp
# my own notes - do not delete
";
            std::fs::write(&path, content).expect("seed");

            assert_eq!(
                install_dsh_at(&path, true).expect("rewrite"),
                DshOutcome::Rewritten
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert!(after.contains("# my own notes - do not delete"), "{after}");
            assert!(after.contains("      command: llmtrim\n"), "{after}");
            assert_eq!(after.matches("- id: mcp-llmtrim").count(), 1);
        }

        #[test]
        fn dsh_install_claims_but_does_not_append_beside_a_flow_row() {
            // A row spelled as an indented flow mapping is ours: a plain install reports it as
            // stale rather than duplicating it, and `--force` rewrites just that row.
            let path = temp_patch("flow-row");
            let content = "\
- insert:
  - {id: mcp-llmtrim, name: '@deepseek-ai/dsh-mcp-client', config: {serverName: llmtrim, command: C:\\old\\llmtrim.exe, args: [mcp]}}
";
            std::fs::write(&path, content).expect("seed");

            assert_eq!(
                install_dsh_at(&path, false).expect("stale"),
                DshOutcome::Stale
            );
            assert_eq!(std::fs::read_to_string(&path).expect("read"), content);
            assert_eq!(
                install_dsh_at(&path, true).expect("rewrite"),
                DshOutcome::Rewritten
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert_eq!(after.matches("mcp-llmtrim").count(), 1);
            assert!(after.contains("      command: llmtrim\n"), "{after}");
        }

        #[test]
        fn dsh_install_never_claims_a_foreign_block_by_a_nested_id() {
            // A foreign entry whose config happens to contain `id: mcp-llmtrim` is not our row: that
            // id is a mapping key inside the other entry, not a list item of the insert list.
            // Claiming it made `--force` splice the whole block and delete the foreign registration.
            let content = "- insert:\n  - id: mcp-playwright\n    name: '@deepseek-ai/dsh-mcp-client'\n    config:\n      serverName: playwright\n      command: npx\n      args:\n        - '@playwright/mcp'\n      env:\n        a: b\n        id: mcp-llmtrim\n";
            assert!(
                content.contains("\n        id: mcp-llmtrim\n"),
                "the foreign id stays nested under `env:`"
            );

            for force in [false, true] {
                let path = temp_patch(if force {
                    "nested-id-force"
                } else {
                    "nested-id"
                });
                std::fs::write(&path, content).expect("seed");

                assert_eq!(
                    install_dsh_at(&path, force).expect("append"),
                    DshOutcome::Written
                );
                let after = std::fs::read_to_string(&path).expect("read");
                assert!(
                    after.starts_with(content),
                    "the foreign block is untouched: {after}"
                );
                assert!(
                    after.contains("  - id: mcp-playwright\n"),
                    "the foreign entry survives: {after}"
                );
                assert!(after.contains("      command: llmtrim\n"), "{after}");
            }
        }

        #[test]
        fn dsh_install_refuses_a_row_below_the_list_item_level() {
            // Our row is a list item, but the list's item level is the foreign entry's indent, so we
            // do not solely own this list: rewriting it would delete playwright.
            let content = "- insert:\n  - id: mcp-playwright\n    name: '@deepseek-ai/dsh-mcp-client'\n    config:\n      serverName: playwright\n      command: npx\n    - id: mcp-llmtrim\n";
            assert!(
                content.contains("\n    - id: mcp-llmtrim\n"),
                "our row sits below the item level"
            );

            for force in [false, true] {
                let path = temp_patch(if force { "deep-row-force" } else { "deep-row" });
                std::fs::write(&path, content).expect("seed");

                let err = install_dsh_at(&path, force).expect_err("must refuse");
                assert!(err.to_string().contains("mcp-llmtrim"), "{err}");
                assert_eq!(std::fs::read_to_string(&path).expect("read"), content);
            }
        }

        #[test]
        fn dsh_install_force_stops_at_the_next_top_level_row() {
            // The most common real terminator: the next column-0 row. `--force` must rewrite only
            // our row and leave that row alone.
            let content = "- insert:\n  - id: mcp-llmtrim\n    name: '@deepseek-ai/dsh-mcp-client'\n    config:\n      serverName: llmtrim\n      command: C:\\old\\llmtrim.exe\n      args:\n        - mcp\n- id: keep-me\n  disabled: true\n";
            assert!(
                content.contains("\n- id: keep-me\n"),
                "the terminator row is at column 0"
            );
            let path = temp_patch("next-top-level-row");
            std::fs::write(&path, content).expect("seed");

            assert_eq!(
                install_dsh_at(&path, true).expect("rewrite"),
                DshOutcome::Rewritten
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert!(
                after.contains("- id: keep-me\n  disabled: true\n"),
                "the next top-level row survives: {after}"
            );
            assert!(after.contains("      command: llmtrim\n"), "{after}");
            assert_eq!(after.matches("- insert:").count(), 1);
            assert_eq!(after.matches("- id: mcp-llmtrim").count(), 1);
        }

        #[test]
        fn dsh_install_appends_to_a_file_without_a_trailing_newline() {
            let path = temp_patch("no-newline");
            std::fs::write(&path, "- id: keep-me\n  disabled: false").expect("seed");

            assert_eq!(
                install_dsh_at(&path, false).expect("write"),
                DshOutcome::Written
            );
            let after = std::fs::read_to_string(&path).expect("read");
            assert!(after.contains("- id: keep-me\n  disabled: false\n\n"));
            assert!(after.ends_with(&format!("{DSH_END}\n")));
            assert_eq!(after.matches("- id: mcp-llmtrim").count(), 1);
        }

        #[test]
        fn dsh_install_leaves_no_temp_file_behind() {
            let path = temp_patch("atomic");
            let _ = std::fs::remove_file(&path);
            assert_eq!(
                install_dsh_at(&path, false).expect("write"),
                DshOutcome::Written
            );
            assert!(
                !path.with_file_name("cordis.patch.yml.llmtrim-tmp").exists(),
                "the atomic write must clean up after itself"
            );
        }

        #[test]
        fn dsh_patch_path_resolution_is_pure_and_conservative() {
            // No `profiles/` directory: not a DSH install, so `--client dsh` refuses rather than
            // creating `~/.dsh` for an app that is not there.
            let home = temp_dir_for("path");
            // A previous run in a reused pid could have left `profiles/` behind, which would make the
            // "not installed" assertion below depend on run order.
            let _ = std::fs::remove_dir_all(home.join(".dsh"));
            let (path, installed) = dsh_patch_path_from_home(&home);
            assert_eq!(path, home.join(".dsh").join("cordis.patch.yml"));
            assert!(!installed);

            std::fs::create_dir_all(home.join(".dsh").join("profiles")).expect("profiles dir");
            let (path, installed) = dsh_patch_path_from_home(&home);
            assert_eq!(path, home.join(".dsh").join("cordis.patch.yml"));
            assert!(installed);

            // An explicit `$DSH_HOME` is trusted as-is.
            let (path, installed) =
                dsh_patch_path_from_dsh_home(std::ffi::OsStr::new("D:\\explicit-dsh"));
            assert_eq!(
                path,
                std::path::PathBuf::from("D:\\explicit-dsh").join("cordis.patch.yml")
            );
            assert!(installed);
        }

        #[test]
        fn dsh_absence_is_fatal_only_when_the_client_was_named() {
            // `--client dsh` asked for it by name, so a missing install must not look like success;
            // `--client all` merely tried it, so it is a note. The message has to name the path, or
            // the user cannot tell which home we looked in.
            let nowhere = std::path::Path::new("C:\\no\\such\\dsh\\cordis.patch.yml");
            let err = dsh_absence(true, nowhere).expect_err("a named client must be fatal");
            let message = err.to_string();
            assert!(
                message.contains("No DeepSeek Harness install found"),
                "{message}"
            );
            assert!(
                message.contains("C:\\no\\such\\dsh\\cordis.patch.yml"),
                "{message}"
            );
            assert!(dsh_absence(false, nowhere).is_ok());
        }

        #[test]
        fn dsh_print_mode_short_circuits_before_the_absence_check() {
            // `--print` returns before any path is resolved, so it is usable on a machine with no
            // DSH at all — and `--client dsh --print` on this host must not touch the real home.
            // Asking with `required = true` is what makes this assert something: an absence branch
            // that were reached would have to error, so `Ok` proves the print branch returned first.
            assert!(install_dsh(true, false, true).is_ok());
        }

        #[test]
        fn install_for_client_all_runs_both_halves() {
            let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let claude_calls = std::rc::Rc::clone(&calls);
            let dsh_calls = std::rc::Rc::clone(&calls);
            let claude = move |_print: bool, _force: bool| -> Result<()> {
                claude_calls.borrow_mut().push("claude".to_string());
                Ok(())
            };
            let dsh = move |_print: bool, _force: bool, required: bool| -> Result<()> {
                dsh_calls
                    .borrow_mut()
                    .push(format!("dsh required={required}"));
                Ok(())
            };

            install_for_client_with(false, false, McpClient::All, claude, dsh)
                .expect("both halves");
            assert_eq!(
                *calls.borrow(),
                vec!["claude".to_string(), "dsh required=false".to_string()]
            );
        }

        #[test]
        fn install_for_client_all_print_forwards_print_to_both_halves() {
            let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let claude_seen = std::rc::Rc::clone(&seen);
            let dsh_seen = std::rc::Rc::clone(&seen);
            let claude = move |print: bool, _force: bool| -> Result<()> {
                claude_seen
                    .borrow_mut()
                    .push(format!("claude print={print}"));
                Ok(())
            };
            let dsh = move |print: bool, _force: bool, required: bool| -> Result<()> {
                dsh_seen
                    .borrow_mut()
                    .push(format!("dsh print={print} required={required}"));
                Ok(())
            };

            install_for_client_with(true, false, McpClient::All, claude, dsh).expect("print");
            assert_eq!(
                *seen.borrow(),
                vec![
                    "claude print=true".to_string(),
                    "dsh print=true required=false".to_string()
                ]
            );
        }

        #[test]
        fn install_for_client_dsh_requires_an_install() {
            // The DSH-only path must demand an install and must not run the Claude half at all.
            let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let dsh_seen = std::rc::Rc::clone(&seen);
            let claude = |_print: bool, _force: bool| -> Result<()> {
                panic!("the Claude half must not run for --client dsh")
            };
            let dsh = move |print: bool, force: bool, required: bool| -> Result<()> {
                dsh_seen
                    .borrow_mut()
                    .push(format!("print={print} force={force} required={required}"));
                Ok(())
            };

            install_for_client_with(false, true, McpClient::Dsh, claude, dsh).expect("dsh half");
            assert_eq!(
                *seen.borrow(),
                vec!["print=false force=true required=true".to_string()]
            );
        }

        #[test]
        fn install_for_client_all_returns_the_first_error_and_still_runs_dsh() {
            let dsh_ran = std::rc::Rc::new(std::cell::Cell::new(false));
            let flag = std::rc::Rc::clone(&dsh_ran);
            let claude =
                |_print: bool, _force: bool| -> Result<()> { anyhow::bail!("claude broke") };
            let dsh = move |_print: bool, _force: bool, _required: bool| -> Result<()> {
                flag.set(true);
                Ok(())
            };

            let err = install_for_client_with(false, false, McpClient::All, claude, dsh)
                .expect_err("the Claude error is the one reported");
            assert_eq!(err.to_string(), "claude broke");
            assert!(
                dsh_ran.get(),
                "a failing Claude half must not skip the DSH half"
            );
        }

        #[cfg(unix)]
        #[test]
        fn dsh_install_keeps_the_patch_files_permissions() {
            use std::os::unix::fs::PermissionsExt as _;
            let path = temp_patch("perms");
            let _ = std::fs::remove_file(&path);
            std::fs::write(&path, "- id: keep-me\n  disabled: false\n").expect("seed");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");

            assert_eq!(
                install_dsh_at(&path, false).expect("write"),
                DshOutcome::Written
            );
            let mode = std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "the write must not widen the patch layer");
        }

        #[test]
        fn dsh_install_reports_a_failed_write_without_clobbering_anything() {
            // Occupy the temp path with a directory: no platform will open a directory as the
            // staged file, and the cleanup must not delete what it did not create.
            let path = temp_patch("failed-write");
            let _ = std::fs::remove_file(&path);
            let tmp = path.with_file_name("cordis.patch.yml.llmtrim-tmp");
            std::fs::create_dir_all(&tmp).expect("occupy the temp path");

            let err = install_dsh_at(&path, false).expect_err("the write must fail");
            assert!(err.to_string().contains("failed to write"), "{err}");
            assert!(
                tmp.is_dir(),
                "the cleanup must not remove what it did not create"
            );
            assert!(!path.exists(), "nothing may be written at the patch layer");
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
