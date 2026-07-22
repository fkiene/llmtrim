//! SQLite savings ledger.
//!
//! A rusqlite ledger at `~/.local/share/<tool>/tracking.db` that records
//! **real-tokenizer** counts and carries nullable
//! output-token columns for the round-trip cost model once the proxy phase can
//! measure them. Recording is best-effort at the CLI layer: a ledger failure must
//! never block the user's compressed output.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

/// One compression event.
#[derive(Debug, Clone)]
pub struct Record {
    pub provider: String,
    pub model: Option<String>,
    pub tokenizer: String,
    pub exact: bool,
    pub input_before: i64,
    pub input_after: i64,
    pub output_before: Option<i64>,
    pub output_after: Option<i64>,
    /// Microseconds spent compressing this request (proxy overhead); `None` for CLI paths.
    pub compress_micros: Option<i64>,
    /// Provider-reported cached input tokens reused on this request (Anthropic
    /// `cache_read_input_tokens`), billed at ~10% of input price — a real bill discount the
    /// token % can't show. `None` when the provider reports none / on CLI paths.
    pub cache_read_tokens: Option<i64>,
    /// Provider-reported *uncached* input tokens billed at the full rate (Anthropic
    /// `input_tokens`; OpenAI `prompt_tokens − cached_tokens`). With `cache_read_tokens`
    /// and `cache_write_tokens` this reconstructs the real input bill. `None` when the
    /// response carried no usage / on CLI paths.
    pub fresh_input_tokens: Option<i64>,
    /// Provider-reported cache-write tokens (Anthropic `cache_creation_input_tokens`,
    /// billed at 1.25×). `None` for providers without a write surcharge / no usage.
    pub cache_write_tokens: Option<i64>,
    /// Whether the request that was actually forwarded carried the output-shaping
    /// instruction (Stage F ran and the compressed body was kept). `None` on rows recorded
    /// before this column existed.
    pub output_shaped: Option<bool>,
    /// Tokens in the frozen (cache-controlled) prefix the stages skipped by cache-zone
    /// discipline. `input_before − frozen` is the compressible surface — the honest
    /// denominator for the "saved on new content" figure. `None` on pre-meter rows.
    pub frozen_input_tokens: Option<i64>,
    /// Explicit non-success outcome for a turn that never produced measurable output, e.g.
    /// `transport_reset` when the upstream keep-alive died before any response. `None` on
    /// ordinary rows (including pre-column ledgers) so old readers stay valid.
    pub outcome: Option<String>,
}

/// One attributed proxy turn for the breakdown view: identity + provider-reported usage +
/// the pricing snapshot frozen at the time of the request. Per-source dollars are
/// derived at render from the blocks' token splits times these frozen rates, so a
/// historical session always shows what it actually cost even after price tables move.
#[derive(Debug, Clone, Default)]
pub struct BreakdownTurn {
    pub session_id: String,
    /// Claude Code's own session id (`x-claude-code-session-id` header), when present.
    /// Distinct from `session_id` (a hash of the system prompt) — this lets the status
    /// line match a session exactly. `None` for non-Claude-Code traffic or older rows.
    pub cc_session_id: Option<String>,
    pub agent: String,
    pub project: Option<String>,
    pub session_name: Option<String>,
    pub provider: String,
    pub model: Option<String>,
    /// The `sub` backend that actually served this turn (`codex`/`kimi`), when one did. `None`
    /// for a turn Anthropic served — including a fallback-mode turn where Anthropic answered and
    /// the chain never fired. `provider` above stays the *wire* provider (`anthropic`, the shape
    /// we parse and reply in), so this is the only record of who really answered; `model` already
    /// carries the upstream model id on a rerouted turn.
    pub sub_provider: Option<String>,
    /// Context window of the model, for occupancy %.
    pub window: i64,
    pub fresh_input: i64,
    pub cache_read: i64,
    pub cache_write: i64,
    pub output_tok: i64,
    /// Frozen rates, USD per 1M tokens.
    pub input_rate: f64,
    pub output_rate: f64,
    pub cache_read_rate: f64,
    pub cache_write_rate: f64,
    /// Frozen total bill of this turn in micro-USD (integer, drift-free for sums).
    pub bill_micros: i64,
    /// Input tokens before / after llmtrim's compression — the per-turn savings, summed per
    /// session to show "how much we trimmed" alongside the spend.
    pub input_before: i64,
    pub input_after: i64,
}

/// One attributed source block of a turn, with the provider usage reconciled onto it.
#[derive(Debug, Clone, Default)]
pub struct BreakdownBlock {
    pub zone: String,
    pub section: String,
    pub bucket: String,
    /// Source group: `Static` / `Messages` / `Output`.
    pub group_label: String,
    /// Source category label within the group (e.g. "System prompt", "MCP tools").
    pub label: String,
    pub mcp_server: Option<String>,
    pub tool_name: Option<String>,
    pub role: Option<String>,
    pub msg_index: Option<i64>,
    /// Attributed block token count (pre-reconcile measure).
    pub raw_tokens: i64,
    /// Provider usage reconciled onto this block (sum across blocks == turn usage).
    pub fresh_tok: f64,
    pub cache_read_tok: f64,
    pub cache_write_tok: f64,
    pub output_tok: f64,
}

/// Per-provider aggregate row.
#[derive(Debug, Clone)]
pub struct ProviderRow {
    pub provider: String,
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub exact: bool,
    pub output_before: i64,
    pub output_after: i64,
    pub output_events: i64,
}

/// Per-(provider, model) aggregate row — used to price savings with a per-model rate.
#[derive(Debug, Clone)]
pub struct ModelRow {
    pub provider: String,
    pub model: Option<String>,
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub output_after: i64,
    /// Provider-reported cache-read tokens (discounted prefix), summed.
    pub cache_read: i64,
    /// Provider-reported cache-write tokens (surcharged), summed.
    pub cache_write: i64,
    /// Full-rate input tokens, summed. Rows without usage fall back to
    /// `max(input_after − cache_read, 0)` so pre-usage ledgers still price sanely.
    pub fresh_input_est: i64,
    /// Output tokens from requests where the output-shaping instruction was actually
    /// forwarded — the only output the A/B bench factor may be projected onto.
    pub output_after_shaped: i64,
    /// Frozen-zone meter sums over this model's metered rows (frozen recorded): total
    /// frozen-prefix tokens and the input before/after of those same rows, so
    /// `(metered_before − frozen) → (metered_after − frozen)` is the measured saving on
    /// the model's compressible (new-content) surface. All zero on pre-meter rows.
    pub frozen_input_tokens: i64,
    pub metered_input_before: i64,
    pub metered_input_after: i64,
}

/// Time-series bucket granularity for `by_period`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    Day,
    Week,
    Month,
}

impl Period {
    /// SQL expression that buckets the rfc3339 `ts` column (string-sliced, no date parse
    /// for day/month; `strftime` only for ISO week).
    pub(crate) fn sql_bucket(self) -> &'static str {
        match self {
            Period::Day => "substr(ts, 1, 10)",       // YYYY-MM-DD
            Period::Month => "substr(ts, 1, 7)",      // YYYY-MM
            Period::Week => "strftime('%Y-W%W', ts)", // YYYY-Www
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Period::Day => "daily",
            Period::Week => "weekly",
            Period::Month => "monthly",
        }
    }
}

/// One time bucket of savings, for `--daily/--weekly/--monthly` reports.
#[derive(Debug, Clone)]
pub struct PeriodRow {
    pub bucket: String,
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub output_before: i64,
    pub output_after: i64,
}

/// Aggregate savings for the dashboard.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub events: i64,
    pub input_before: i64,
    pub input_after: i64,
    pub any_approximate: bool,
    pub by_provider: Vec<ProviderRow>,
    pub output_before: i64,
    pub output_after: i64,
    pub output_events: i64,
    /// Mean compression overhead (µs) across recorded requests; `None` if none recorded it.
    pub avg_compress_micros: Option<f64>,
    /// Total cached input tokens reused (Anthropic prompt-cache hits) — the discounted prefix.
    pub cache_read_tokens: i64,
    /// rfc3339 UTC timestamp of the most recent recorded request (`None` on an empty
    /// ledger) — the end-to-end "traffic actually flows" signal for `status`.
    pub last_ts: Option<String>,
    /// Frozen-zone meter sums, restricted to rows that recorded the meter (post-feature):
    /// total frozen-prefix tokens, and the input before/after of those same rows — so
    /// `(metered_before − frozen) → (metered_after − frozen)` is the measured saving on
    /// the compressible (new-content) surface. All zero on pre-meter ledgers.
    pub frozen_input_tokens: i64,
    pub metered_input_before: i64,
    pub metered_input_after: i64,
}

impl Summary {
    pub fn saved(&self) -> i64 {
        self.input_before - self.input_after
    }

    /// Percentage of input tokens saved (0.0 when no data).
    pub fn saved_pct(&self) -> f64 {
        if self.input_before <= 0 {
            0.0
        } else {
            (self.saved() as f64 / self.input_before as f64) * 100.0
        }
    }

    pub fn output_saved(&self) -> i64 {
        self.output_before - self.output_after
    }

    /// Percentage of output tokens saved (0.0 when no counterfactual data).
    pub fn output_saved_pct(&self) -> f64 {
        if self.output_before <= 0 {
            0.0
        } else {
            (self.output_saved() as f64 / self.output_before as f64) * 100.0
        }
    }
}

/// Default ledger row cap — the most-recent N compression events are retained. Each row is
/// metadata only (~100 bytes), so this bounds the file to roughly 10-15 MB.
pub const DEFAULT_MAX_ROWS: i64 = 100_000;

/// Retention cap for the per-source breakdown, in *turns*. A turn fans out into one
/// `breakdown_blocks` row per source (hundreds in agent traffic), so the block table grows
/// far faster than `compressions` and needs its own cap — reusing `DEFAULT_MAX_ROWS` here let
/// the blocks table reach millions of rows and made `status --json` aggregate a
/// multi-hundred-MB join on every call.
///
/// The expensive aggregate is now opt-in (`status --json --breakdown`), so the cap no longer
/// governs the hot path — it only bounds disk. We keep it generous to preserve a long history
/// for the breakdown TUI; the file scales with usage (heavy agent traffic at this cap can reach
/// a few hundred MB to ~1 GB).
pub const DEFAULT_MAX_BREAKDOWN_TURNS: i64 = 50_000;

pub struct Tracker {
    conn: Connection,
}

impl Tracker {
    /// Open (creating if needed) the ledger at the default path, or the path in
    /// `LLMTRIM_DB_PATH` when set.
    pub fn open() -> Result<Self> {
        let path = default_db_path()?;
        Self::open_at(&path)
    }

    pub fn open_at(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open ledger at {}", path.display()))?;
        let tracker = Self { conn };
        tracker.migrate()?;
        // Bound the ledger on open: row cap + (if configured) age retention. The daemon
        // opens once and re-prunes periodically (see serve.rs); CLI paths prune per call.
        let _ = tracker.prune_default();
        // Also bound the per-source breakdown here, not just in the daemon's write loop, so a
        // restart immediately reclaims a table that grew unbounded under an older build (it
        // checkpoints the WAL when it deletes). A no-op COUNT once the table is within cap.
        let _ = tracker.prune_breakdown(Self::breakdown_turns_cap());
        Ok(tracker)
    }

    /// Open the ledger for a **read-only** consumer (the breakdown TUI): migrate the schema if
    /// needed, but do NOT prune (the daemon owns retention — a reader writing a `DELETE` on the
    /// shared WAL is pure waste), and tune the connection for repeated SELECT-only aggregate
    /// scans. `query_only` is set after `migrate()` (which may itself write schema on a fresh or
    /// stale DB), so it guarantees no *query-time* writes — not that the open is write-free.
    pub fn open_reader() -> Result<Self> {
        let path = default_db_path()?;
        Self::open_reader_at(&path)
    }

    /// `open_reader` at an explicit path.
    pub fn open_reader_at(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open ledger at {}", path.display()))?;
        let tracker = Self { conn };
        tracker.migrate()?;
        // Read-side tuning; all best-effort (a missing pragma must not break the TUI).
        let _ = tracker.conn.pragma_update(None, "temp_store", "MEMORY");
        let _ = tracker.conn.pragma_update(None, "cache_size", -8000_i64);
        let _ = tracker
            .conn
            .pragma_update(None, "mmap_size", 268_435_456_i64);
        // Set last: from here the connection refuses query-time writes (the daemon keeps pruning).
        let _ = tracker.conn.pragma_update(None, "query_only", "ON");
        Ok(tracker)
    }

    /// Consume the tracker and return its raw connection. Used by the breakdown query
    /// layer's tests to read back an in-memory ledger through one connection.
    pub fn into_connection(self) -> Connection {
        self.conn
    }

    /// Shared connection for read-side money / breakdown queries that live outside `Tracker`
    /// (e.g. [`crate::money::money_totals`]). Prefer this over re-opening the file so in-memory
    /// test ledgers and the daemon's live path stay on one SQLite handle.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// In-memory ledger (tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory ledger")?;
        let tracker = Self { conn };
        tracker.migrate()?;
        Ok(tracker)
    }

    fn migrate(&self) -> Result<()> {
        // The ledger is written by the daemon thread AND by every CLI compress/send while
        // the `status` TUI reads it every ~2s. With rusqlite's defaults (rollback journal,
        // busy_timeout 0) a concurrent writer/reader hits SQLITE_BUSY immediately, dropping
        // rows or failing reads. WAL lets a reader and a writer proceed together; a 2s busy
        // timeout absorbs the brief writer-vs-writer overlap. Best-effort: an in-memory DB
        // (tests) can't use WAL, and a missing pragma must not break recording.
        let _ = self.conn.pragma_update(None, "journal_mode", "WAL");
        let _ = self.conn.pragma_update(None, "busy_timeout", 2000);
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS compressions (
                    id            INTEGER PRIMARY KEY,
                    ts            TEXT NOT NULL,
                    provider      TEXT NOT NULL,
                    model         TEXT,
                    tokenizer     TEXT NOT NULL,
                    exact         INTEGER NOT NULL,
                    input_before  INTEGER NOT NULL,
                    input_after   INTEGER NOT NULL,
                    output_before INTEGER,
                    output_after  INTEGER,
                    compress_micros INTEGER,
                    cache_read_tokens INTEGER,
                    fresh_input_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    output_shaped INTEGER,
                    frozen_input_tokens INTEGER,
                    outcome TEXT
                );
                CREATE TABLE IF NOT EXISTS breakdown_turns (
                    id            INTEGER PRIMARY KEY,
                    ts            TEXT NOT NULL,
                    session_id    TEXT NOT NULL,
                    cc_session_id TEXT,
                    agent         TEXT NOT NULL,
                    project       TEXT,
                    session_name  TEXT,
                    provider      TEXT NOT NULL,
                    model         TEXT,
                    sub_provider  TEXT,
                    window        INTEGER NOT NULL,
                    fresh_input   INTEGER NOT NULL,
                    cache_read    INTEGER NOT NULL,
                    cache_write   INTEGER NOT NULL,
                    output_tok    INTEGER NOT NULL,
                    input_rate       REAL NOT NULL,
                    output_rate      REAL NOT NULL,
                    cache_read_rate  REAL NOT NULL,
                    cache_write_rate REAL NOT NULL,
                    bill_micros   INTEGER NOT NULL,
                    input_before  INTEGER,
                    input_after   INTEGER
                );
                CREATE TABLE IF NOT EXISTS breakdown_blocks (
                    id            INTEGER PRIMARY KEY,
                    turn_id       INTEGER NOT NULL REFERENCES breakdown_turns(id),
                    zone          TEXT NOT NULL,
                    section       TEXT NOT NULL,
                    bucket        TEXT NOT NULL,
                    group_label   TEXT NOT NULL,
                    label         TEXT NOT NULL,
                    mcp_server    TEXT,
                    tool_name     TEXT,
                    role          TEXT,
                    msg_index     INTEGER,
                    raw_tokens    INTEGER NOT NULL,
                    fresh_tok        REAL NOT NULL,
                    cache_read_tok   REAL NOT NULL,
                    cache_write_tok  REAL NOT NULL,
                    output_tok       REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_breakdown_turns_session ON breakdown_turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_breakdown_turns_agent ON breakdown_turns(agent);
                CREATE INDEX IF NOT EXISTS idx_breakdown_blocks_turn ON breakdown_blocks(turn_id);
                -- Serves sessions()'s GROUP BY + per-session ROW_NUMBER ordering and latest_turn().
                CREATE INDEX IF NOT EXISTS idx_breakdown_turns_group ON breakdown_turns(session_id, agent, project, id);
                -- Money range queries (today / week) and prune-friendly scans.
                CREATE INDEX IF NOT EXISTS idx_breakdown_turns_ts ON breakdown_turns(ts);
                -- Lets the day-scoped aggregate (by_model_today) range-seek instead of scanning.
                CREATE INDEX IF NOT EXISTS idx_compressions_ts ON compressions(ts);",
            )
            .context("failed to migrate ledger schema")?;
        // Additive columns for ledgers created before these fields existed — each ALTER errors
        // with "duplicate column" once it exists (and on fresh DBs the CREATE already has it),
        // which we ignore. A *different* failure (read-only / corrupt DB) must surface here, not
        // hide until a confusing later INSERT error, so we only swallow the duplicate-column case.
        for col in [
            "compress_micros",
            "cache_read_tokens",
            "fresh_input_tokens",
            "cache_write_tokens",
            "output_shaped",
            "frozen_input_tokens",
        ] {
            if let Err(e) = self.conn.execute(
                &format!("ALTER TABLE compressions ADD COLUMN {col} INTEGER"),
                [],
            ) && !is_duplicate_column(&e)
            {
                return Err(e).with_context(|| format!("failed to add ledger column {col}"));
            }
        }
        // Additive columns for breakdown_turns ledgers created before these fields existed.
        for col in ["input_before", "input_after"] {
            if let Err(e) = self.conn.execute(
                &format!("ALTER TABLE breakdown_turns ADD COLUMN {col} INTEGER"),
                [],
            ) && !is_duplicate_column(&e)
            {
                return Err(e).with_context(|| format!("failed to add breakdown column {col}"));
            }
        }
        // TEXT columns, not INTEGER — additive ALTERs kept separate from the loop above.
        for col in ["cc_session_id", "sub_provider"] {
            if let Err(e) = self.conn.execute(
                &format!("ALTER TABLE breakdown_turns ADD COLUMN {col} TEXT"),
                [],
            ) && !is_duplicate_column(&e)
            {
                return Err(e).with_context(|| format!("failed to add breakdown column {col}"));
            }
        }
        for col in ["outcome"] {
            if let Err(e) = self.conn.execute(
                &format!("ALTER TABLE compressions ADD COLUMN {col} TEXT"),
                [],
            ) && !is_duplicate_column(&e)
            {
                return Err(e).with_context(|| format!("failed to add ledger column {col}"));
            }
        }
        Ok(())
    }

    /// Apply retention to the ledger: drop rows older than `max_age_days` (when set), then
    /// trim to the most recent `max_rows`. Returns the number of rows deleted. The ledger
    /// holds only metadata (no prompt/response text), but it must still stay bounded for the
    /// always-on daemon — analytics only need recent history.
    pub fn prune(&self, max_rows: i64, max_age_days: Option<i64>) -> Result<u64> {
        let mut deleted: u64 = 0;
        // Age-based: `ts` is rfc3339 UTC (always `+00:00`), so a lexical `<` compare against
        // the cutoff is a correct chronological compare — no date parsing needed.
        if let Some(days) = max_age_days.filter(|d| *d > 0) {
            let delta = chrono::TimeDelta::try_days(days).unwrap_or_else(chrono::TimeDelta::zero);
            let cutoff = (chrono::Utc::now() - delta).to_rfc3339();
            deleted += self
                .conn
                .execute("DELETE FROM compressions WHERE ts < ?1", params![cutoff])
                .context("failed to age-prune ledger")? as u64;
        }
        // Row cap: keep only the most recent `max_rows` rows by id.
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM compressions", [], |row| row.get(0))
            .context("failed to count ledger rows")?;
        if n > max_rows {
            deleted += self
                .conn
                .execute(
                    "DELETE FROM compressions WHERE id <= (SELECT MAX(id) - ?1 FROM compressions)",
                    params![max_rows],
                )
                .context("failed to cap-prune ledger")? as u64;
        }
        Ok(deleted)
    }

    /// Prune with the default policy: the configured row cap (`LLMTRIM_MAX_ROWS` env or
    /// `max_rows` in the config file, default [`DEFAULT_MAX_ROWS`]) plus the configured age
    /// retention (`LLMTRIM_RETENTION_DAYS` env or `retention_days` in the config file;
    /// `None` = age retention disabled, row cap only).
    pub fn prune_default(&self) -> Result<u64> {
        self.prune(
            llmtrim_core::config::max_rows().unwrap_or(DEFAULT_MAX_ROWS),
            llmtrim_core::config::retention_days(),
        )
    }

    /// The configured per-source breakdown cap in turns (`LLMTRIM_MAX_BREAKDOWN_TURNS`
    /// env or `max_breakdown_turns` in the config file) or [`DEFAULT_MAX_BREAKDOWN_TURNS`].
    pub fn breakdown_turns_cap() -> i64 {
        llmtrim_core::config::max_breakdown_turns().unwrap_or(DEFAULT_MAX_BREAKDOWN_TURNS)
    }

    pub fn record(&self, r: &Record) -> Result<()> {
        let ts = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO compressions
                    (ts, provider, model, tokenizer, exact, input_before, input_after,
                     output_before, output_after, compress_micros, cache_read_tokens,
                     fresh_input_tokens, cache_write_tokens, output_shaped, frozen_input_tokens,
                     outcome)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![
                    ts,
                    r.provider,
                    r.model,
                    r.tokenizer,
                    i64::from(r.exact),
                    r.input_before,
                    r.input_after,
                    r.output_before,
                    r.output_after,
                    r.compress_micros,
                    r.cache_read_tokens,
                    r.fresh_input_tokens,
                    r.cache_write_tokens,
                    r.output_shaped.map(i64::from),
                    r.frozen_input_tokens,
                    r.outcome,
                ],
            )
            .context("failed to record compression")?;
        Ok(())
    }

    /// Insert one breakdown turn and its attributed blocks in a single transaction. Returns the
    /// turn's rowid. Best-effort at the call site: the proxy swallows any error so a failed
    /// attribution write never blocks proxying or the main `compressions` ledger.
    pub fn record_breakdown(&self, turn: &BreakdownTurn, blocks: &[BreakdownBlock]) -> Result<i64> {
        let ts = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO breakdown_turns
                    (ts, session_id, cc_session_id, agent, project, session_name, provider, model,
                     sub_provider, window, fresh_input, cache_read, cache_write, output_tok,
                     input_rate, output_rate, cache_read_rate, cache_write_rate, bill_micros,
                     input_before, input_after)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)",
                params![
                    ts,
                    turn.session_id,
                    turn.cc_session_id,
                    turn.agent,
                    turn.project,
                    turn.session_name,
                    turn.provider,
                    turn.model,
                    turn.sub_provider,
                    turn.window,
                    turn.fresh_input,
                    turn.cache_read,
                    turn.cache_write,
                    turn.output_tok,
                    turn.input_rate,
                    turn.output_rate,
                    turn.cache_read_rate,
                    turn.cache_write_rate,
                    turn.bill_micros,
                    turn.input_before,
                    turn.input_after,
                ],
            )
            .context("failed to record breakdown turn")?;
        let turn_id = self.conn.last_insert_rowid();
        let mut stmt = self
            .conn
            .prepare_cached(
                "INSERT INTO breakdown_blocks
                    (turn_id, zone, section, bucket, group_label, label, mcp_server, tool_name,
                     role, msg_index, raw_tokens, fresh_tok, cache_read_tok, cache_write_tok, output_tok)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            )
            .context("failed to prepare breakdown block insert")?;
        for b in blocks {
            stmt.execute(params![
                turn_id,
                b.zone,
                b.section,
                b.bucket,
                b.group_label,
                b.label,
                b.mcp_server,
                b.tool_name,
                b.role,
                b.msg_index,
                b.raw_tokens,
                b.fresh_tok,
                b.cache_read_tok,
                b.cache_write_tok,
                b.output_tok,
            ])
            .context("failed to record breakdown block")?;
        }
        Ok(turn_id)
    }

    /// Cap the breakdown tables to the most recent `max_turns` turns, deleting the blocks of any
    /// dropped turn. Mirrors the `compressions` row cap so the always-on daemon stays bounded.
    pub fn prune_breakdown(&self, max_turns: i64) -> Result<u64> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM breakdown_turns", [], |row| row.get(0))
            .context("failed to count breakdown turns")?;
        if n <= max_turns {
            return Ok(0);
        }
        self.conn
            .execute(
                "DELETE FROM breakdown_blocks WHERE turn_id <= \
                 (SELECT MAX(id) - ?1 FROM breakdown_turns)",
                params![max_turns],
            )
            .context("failed to cap-prune breakdown blocks")?;
        let deleted = self
            .conn
            .execute(
                "DELETE FROM breakdown_turns WHERE id <= (SELECT MAX(id) - ?1 FROM breakdown_turns)",
                params![max_turns],
            )
            .context("failed to cap-prune breakdown turns")? as u64;
        Ok(deleted)
    }

    /// Test-only: insert a breakdown turn with NULL `input_before`/`input_after` to simulate
    /// pre-meter rows (rows recorded before the compression-meter columns were added).
    #[cfg(test)]
    pub fn record_breakdown_premeter(&self, agent: &str, bill_micros: i64) -> Result<i64> {
        let ts = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO breakdown_turns
                    (ts, session_id, agent, provider, window,
                     fresh_input, cache_read, cache_write, output_tok,
                     input_rate, output_rate, cache_read_rate, cache_write_rate,
                     bill_micros, input_before, input_after)
                 VALUES (?1, ?2, ?3, 'anthropic', 200000,
                         0, 0, 0, 0, 3.0, 15.0, 0.3, 3.75,
                         ?4, NULL, NULL)",
                params![ts, format!("premeter-{agent}"), agent, bill_micros],
            )
            .context("failed to record pre-meter turn")?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Test-only: insert a breakdown turn with an explicit `ts` (for bucket-boundary tests).
    #[cfg(test)]
    pub fn record_breakdown_with_ts(&self, turn: &BreakdownTurn, ts: &str) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO breakdown_turns
                    (ts, session_id, cc_session_id, agent, project, session_name, provider, model,
                     sub_provider, window, fresh_input, cache_read, cache_write, output_tok,
                     input_rate, output_rate, cache_read_rate, cache_write_rate, bill_micros,
                     input_before, input_after)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)",
                params![
                    ts,
                    turn.session_id,
                    turn.cc_session_id,
                    turn.agent,
                    turn.project,
                    turn.session_name,
                    turn.provider,
                    turn.model,
                    turn.sub_provider,
                    turn.window,
                    turn.fresh_input,
                    turn.cache_read,
                    turn.cache_write,
                    turn.output_tok,
                    turn.input_rate,
                    turn.output_rate,
                    turn.cache_read_rate,
                    turn.cache_write_rate,
                    turn.bill_micros,
                    turn.input_before,
                    turn.input_after,
                ],
            )
            .context("failed to record breakdown turn with ts")?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Test-only: insert a record stamped with an explicit `ts`, to exercise age retention
    /// without waiting real time (`record` always stamps `now`).
    #[cfg(test)]
    fn record_with_ts(&self, r: &Record, ts: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO compressions
                    (ts, provider, model, tokenizer, exact, input_before, input_after,
                     output_before, output_after, compress_micros, cache_read_tokens,
                     fresh_input_tokens, cache_write_tokens, output_shaped, frozen_input_tokens,
                     outcome)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![
                    ts,
                    r.provider,
                    r.model,
                    r.tokenizer,
                    i64::from(r.exact),
                    r.input_before,
                    r.input_after,
                    r.output_before,
                    r.output_after,
                    r.compress_micros,
                    r.cache_read_tokens,
                    r.fresh_input_tokens,
                    r.cache_write_tokens,
                    r.output_shaped.map(i64::from),
                    r.frozen_input_tokens,
                    r.outcome,
                ],
            )
            .context("failed to record compression (test)")?;
        Ok(())
    }

    pub fn summary(&self) -> Result<Summary> {
        let (events, input_before, input_after, approx, output_before, output_after, output_events): (
            i64, i64, i64, i64, i64, i64, i64,
        ) = self
            .conn
            .query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        COALESCE(SUM(CASE WHEN exact = 0 THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(output_before), 0),
                        COALESCE(SUM(output_after), 0),
                        COALESCE(SUM(CASE WHEN output_after IS NOT NULL THEN 1 ELSE 0 END), 0)
                 FROM compressions",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .context("failed to summarize ledger")?;

        let mut stmt = self
            .conn
            .prepare(
                "SELECT provider, COUNT(*),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        MIN(exact),
                        COALESCE(SUM(output_before), 0),
                        COALESCE(SUM(output_after), 0),
                        COALESCE(SUM(CASE WHEN output_after IS NOT NULL THEN 1 ELSE 0 END), 0)
                 FROM compressions GROUP BY provider ORDER BY provider",
            )
            .context("failed to prepare provider summary")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ProviderRow {
                    provider: row.get(0)?,
                    events: row.get(1)?,
                    input_before: row.get(2)?,
                    input_after: row.get(3)?,
                    exact: row.get::<_, i64>(4)? != 0,
                    output_before: row.get(5)?,
                    output_after: row.get(6)?,
                    output_events: row.get(7)?,
                })
            })
            .context("failed to query provider summary")?;
        let by_provider = rows
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read provider summary")?;

        // Mean compression overhead + total cached-prefix tokens reused + most recent ts.
        // AVG/SUM ignore NULL (CLI rows / pre-feature ledgers); AVG returns NULL → None when
        // nothing has it; MAX(ts) is a lexical max, correct because ts is rfc3339 UTC.
        let (avg_compress_micros, cache_read_tokens, last_ts): (Option<f64>, i64, Option<String>) =
            self.conn
                .query_row(
                    "SELECT AVG(compress_micros), COALESCE(SUM(cache_read_tokens), 0), MAX(ts)
                     FROM compressions",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .context("failed to summarize latency + cache")?;

        // Frozen-zone meter aggregates over the rows that have it (post-meter), so the
        // new-content % is measured on a consistent population, never diluted by legacy rows.
        let (frozen_input_tokens, metered_input_before, metered_input_after): (i64, i64, i64) =
            self.conn
                .query_row(
                    "SELECT COALESCE(SUM(frozen_input_tokens), 0),
                            COALESCE(SUM(CASE WHEN frozen_input_tokens IS NOT NULL
                                THEN input_before ELSE 0 END), 0),
                            COALESCE(SUM(CASE WHEN frozen_input_tokens IS NOT NULL
                                THEN input_after ELSE 0 END), 0)
                     FROM compressions",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .context("failed to summarize frozen-zone meter")?;

        Ok(Summary {
            events,
            input_before,
            input_after,
            any_approximate: approx > 0,
            by_provider,
            output_before,
            output_after,
            output_events,
            avg_compress_micros,
            cache_read_tokens,
            last_ts,
            frozen_input_tokens,
            metered_input_before,
            metered_input_after,
        })
    }

    /// Per-(provider, model) aggregates, for pricing each model's savings at its own rate.
    /// `fresh_input_est` falls back to `max(input_after − cache_read, 0)` on rows recorded
    /// before usage capture existed, so legacy ledgers still get a sane (slightly
    /// conservative) bill estimate instead of a full-rate one.
    pub fn by_model(&self) -> Result<Vec<ModelRow>> {
        self.by_model_where("")
    }

    /// [`Self::by_model`] restricted to rows recorded today (UTC) — prices the dashboard's
    /// "today" figure. Same day bucketing as `by_period(Day)`: `ts` is rfc3339 UTC, so its
    /// first 10 chars are the UTC date.
    pub fn by_model_today(&self) -> Result<Vec<ModelRow>> {
        // Sargable range (ts is rfc3339 UTC with a `+00:00` offset, written by `record()`, so a
        // lexical compare == chronological) lets the idx_compressions_ts index seek today's slice
        // rather than scan. Recent SQLite (>= 3.38) uses the index here; older versions scan, but
        // correctness holds either way.
        self.by_model_where("WHERE ts >= date('now') || 'T00:00:00+00:00'")
    }

    /// Shared query for [`Self::by_model`]/[`by_model_today`]. `where_clause` is a static SQL
    /// fragment chosen by the callers above, never user input.
    fn by_model_where(&self, where_clause: &str) -> Result<Vec<ModelRow>> {
        let mut stmt = self
            .conn
            .prepare(&format!(
                "SELECT provider, model, COUNT(*),
                        COALESCE(SUM(input_before), 0),
                        COALESCE(SUM(input_after), 0),
                        COALESCE(SUM(output_after), 0),
                        COALESCE(SUM(cache_read_tokens), 0),
                        COALESCE(SUM(cache_write_tokens), 0),
                        COALESCE(SUM(COALESCE(fresh_input_tokens,
                            MAX(input_after - COALESCE(cache_read_tokens, 0), 0))), 0),
                        COALESCE(SUM(CASE WHEN output_shaped = 1
                            THEN COALESCE(output_after, 0) ELSE 0 END), 0),
                        COALESCE(SUM(frozen_input_tokens), 0),
                        COALESCE(SUM(CASE WHEN frozen_input_tokens IS NOT NULL
                            THEN input_before ELSE 0 END), 0),
                        COALESCE(SUM(CASE WHEN frozen_input_tokens IS NOT NULL
                            THEN input_after ELSE 0 END), 0)
                 FROM compressions {where_clause} GROUP BY provider, model ORDER BY provider, model",
            ))
            .context("failed to prepare model summary")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ModelRow {
                    provider: row.get(0)?,
                    model: row.get(1)?,
                    events: row.get(2)?,
                    input_before: row.get(3)?,
                    input_after: row.get(4)?,
                    output_after: row.get(5)?,
                    cache_read: row.get(6)?,
                    cache_write: row.get(7)?,
                    fresh_input_est: row.get(8)?,
                    output_after_shaped: row.get(9)?,
                    frozen_input_tokens: row.get(10)?,
                    metered_input_before: row.get(11)?,
                    metered_input_after: row.get(12)?,
                })
            })
            .context("failed to query model summary")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read model summary")
    }

    /// Savings grouped into time buckets (day/week/month), oldest first.
    pub fn by_period(&self, period: Period) -> Result<Vec<PeriodRow>> {
        let sql = format!(
            "SELECT {} AS bucket, COUNT(*),
                    COALESCE(SUM(input_before), 0),
                    COALESCE(SUM(input_after), 0),
                    COALESCE(SUM(output_before), 0),
                    COALESCE(SUM(output_after), 0)
             FROM compressions GROUP BY bucket ORDER BY bucket",
            period.sql_bucket()
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("failed to prepare period summary")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(PeriodRow {
                    bucket: row.get(0)?,
                    events: row.get(1)?,
                    input_before: row.get(2)?,
                    input_after: row.get(3)?,
                    output_before: row.get(4)?,
                    output_after: row.get(5)?,
                })
            })
            .context("failed to query period summary")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("failed to read period summary")
    }
}

/// True if `e` is SQLite's "duplicate column name" error — the expected outcome of
/// re-running an additive `ALTER TABLE … ADD COLUMN` on a ledger that already has it.
/// Any other ALTER failure (read-only / corrupt) is a real error to surface.
fn is_duplicate_column(e: &rusqlite::Error) -> bool {
    e.to_string().to_lowercase().contains("duplicate column")
}

/// The ledger file path (respects `LLMTRIM_DB_PATH` / `XDG_DATA_HOME`). Exposed so
/// `uninstall --purge` can remove it.
pub fn db_path() -> Result<PathBuf> {
    default_db_path()
}

fn default_db_path() -> Result<PathBuf> {
    if let Some(p) = llmtrim_core::config::RuntimeConfig::get().db_path.clone() {
        return Ok(p);
    }
    let base = if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg)
    } else {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("set HOME (or USERPROFILE), or LLMTRIM_DB_PATH")?;
        PathBuf::from(home).join(".local/share")
    };
    Ok(base.join("llmtrim").join("tracking.db"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(provider: &str, exact: bool, before: i64, after: i64) -> Record {
        Record {
            provider: provider.to_string(),
            model: Some("m".to_string()),
            tokenizer: "t".to_string(),
            exact,
            input_before: before,
            input_after: after,
            output_before: None,
            output_after: None,
            compress_micros: None,
            cache_read_tokens: None,
            fresh_input_tokens: None,
            cache_write_tokens: None,
            output_shaped: None,
            frozen_input_tokens: None,
            outcome: None,
        }
    }

    fn breakdown_turn(session: &str) -> BreakdownTurn {
        BreakdownTurn {
            session_id: session.to_string(),
            cc_session_id: None,
            agent: "claude-code".to_string(),
            project: Some("/proj".to_string()),
            session_name: None,
            provider: "anthropic".to_string(),
            model: Some("claude-sonnet-4".to_string()),
            sub_provider: None,
            window: 200_000,
            fresh_input: 1000,
            cache_read: 500,
            cache_write: 100,
            output_tok: 200,
            input_rate: 3.0,
            output_rate: 15.0,
            cache_read_rate: 0.3,
            cache_write_rate: 3.75,
            bill_micros: 6_750,
            input_before: 2000,
            input_after: 1600,
        }
    }

    fn breakdown_block(label: &str, raw: i64) -> BreakdownBlock {
        BreakdownBlock {
            zone: "input".to_string(),
            section: "static".to_string(),
            bucket: "system".to_string(),
            group_label: "Static".to_string(),
            label: label.to_string(),
            mcp_server: None,
            tool_name: None,
            role: None,
            msg_index: None,
            raw_tokens: raw,
            fresh_tok: raw as f64,
            cache_read_tok: 0.0,
            cache_write_tok: 0.0,
            output_tok: 0.0,
        }
    }

    #[test]
    fn record_breakdown_round_trips() {
        let t = Tracker::open_in_memory().unwrap();
        let id = t
            .record_breakdown(
                &breakdown_turn("sess-a"),
                &[
                    breakdown_block("System prompt", 800),
                    breakdown_block("MCP tools", 200),
                ],
            )
            .expect("record breakdown");
        assert!(id > 0);
        let turns: i64 = t
            .conn
            .query_row("SELECT COUNT(*) FROM breakdown_turns", [], |r| r.get(0))
            .unwrap();
        let blocks: i64 = t
            .conn
            .query_row(
                "SELECT COUNT(*) FROM breakdown_blocks WHERE turn_id = ?1",
                [id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!((turns, blocks), (1, 2));
    }

    #[test]
    fn prune_breakdown_caps_turns_and_orphan_blocks() {
        let t = Tracker::open_in_memory().unwrap();
        for i in 0..5 {
            t.record_breakdown(
                &breakdown_turn(&format!("s{i}")),
                &[breakdown_block("System prompt", 10)],
            )
            .unwrap();
        }
        let deleted = t.prune_breakdown(2).unwrap();
        assert_eq!(deleted, 3);
        let turns: i64 = t
            .conn
            .query_row("SELECT COUNT(*) FROM breakdown_turns", [], |r| r.get(0))
            .unwrap();
        let blocks: i64 = t
            .conn
            .query_row("SELECT COUNT(*) FROM breakdown_blocks", [], |r| r.get(0))
            .unwrap();
        assert_eq!((turns, blocks), (2, 2));
    }

    #[test]
    fn record_and_summarize() {
        let t = Tracker::open_in_memory().unwrap();
        t.record(&rec("openai", true, 100, 60)).unwrap();
        t.record(&rec("openai", true, 50, 40)).unwrap();
        t.record(&rec("anthropic", false, 200, 150)).unwrap();

        let s = t.summary().unwrap();
        assert_eq!(s.events, 3);
        assert_eq!(s.input_before, 350);
        assert_eq!(s.input_after, 250);
        assert_eq!(s.saved(), 100);
        assert!((s.saved_pct() - 28.57).abs() < 0.1);
        assert!(s.any_approximate, "an anthropic (approx) row exists");
        assert_eq!(s.by_provider.len(), 2);

        let oa = s
            .by_provider
            .iter()
            .find(|r| r.provider == "openai")
            .unwrap();
        assert_eq!(oa.events, 2);
        assert!(oa.exact);
        let an = s
            .by_provider
            .iter()
            .find(|r| r.provider == "anthropic")
            .unwrap();
        assert!(!an.exact);
    }

    #[test]
    fn empty_ledger_summary_is_zero() {
        let t = Tracker::open_in_memory().unwrap();
        let s = t.summary().unwrap();
        assert_eq!(s.events, 0);
        assert_eq!(s.saved(), 0);
        assert_eq!(s.saved_pct(), 0.0);
        assert!(!s.any_approximate);
        assert_eq!(s.output_before, 0);
        assert_eq!(s.output_after, 0);
        assert_eq!(s.output_events, 0);
    }

    #[test]
    fn frozen_meter_sums_metered_rows_only() {
        let t = Tracker::open_in_memory().unwrap();
        // Pre-meter row (frozen NULL) — must not dilute the metered population.
        t.record(&rec("anthropic", true, 1000, 900)).unwrap();
        // Metered row: 600 of 1000 frozen → compressible surface 400 → 300.
        let mut r = rec("anthropic", true, 1000, 900);
        r.frozen_input_tokens = Some(600);
        t.record(&r).unwrap();

        let s = t.summary().unwrap();
        assert_eq!(s.frozen_input_tokens, 600);
        assert_eq!(s.metered_input_before, 1000, "metered rows only");
        assert_eq!(s.metered_input_after, 900);
        // Global sums still cover everything.
        assert_eq!(s.input_before, 2000);

        // Per-model meter: same restriction to metered rows.
        let models = t.by_model().unwrap();
        let m = models
            .iter()
            .find(|m| m.provider == "anthropic")
            .expect("anthropic row");
        assert_eq!(m.frozen_input_tokens, 600);
        assert_eq!(m.metered_input_before, 1000, "metered rows only");
        assert_eq!(m.metered_input_after, 900);
    }

    #[test]
    fn output_tokens_round_trip_and_aggregate() {
        let t = Tracker::open_in_memory().unwrap();

        // Row 1: has measured output tokens.
        t.record(&Record {
            provider: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            tokenizer: "tiktoken".to_string(),
            exact: true,
            input_before: 100,
            input_after: 60,
            output_before: None,
            output_after: Some(42),
            compress_micros: Some(300),
            cache_read_tokens: Some(50),
            fresh_input_tokens: Some(80),
            cache_write_tokens: Some(12),
            output_shaped: Some(true),
            frozen_input_tokens: None,
            outcome: None,
        })
        .unwrap();

        // Row 2: also has measured output tokens.
        t.record(&Record {
            provider: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            tokenizer: "tiktoken".to_string(),
            exact: true,
            input_before: 80,
            input_after: 50,
            output_before: None,
            output_after: Some(17),
            compress_micros: Some(500),
            cache_read_tokens: Some(70),
            fresh_input_tokens: None,
            cache_write_tokens: None,
            output_shaped: Some(false),
            frozen_input_tokens: None,
            outcome: None,
        })
        .unwrap();

        // Row 3: network-free (no output measurement).
        t.record(&rec("openai", true, 50, 30)).unwrap();

        let s = t.summary().unwrap();

        // Three total events.
        assert_eq!(s.events, 3);

        // Only two rows had output_after set.
        assert_eq!(s.output_events, 2);

        // Sum of the two measured output_after values.
        assert_eq!(s.output_after, 59);

        // output_before stays NULL → sums to 0.
        assert_eq!(s.output_before, 0);

        // Mean compression overhead over the two timed rows (the rec() row is NULL → ignored).
        assert_eq!(s.avg_compress_micros, Some(400.0));
        // Cached-prefix tokens summed over the rows that reported them.
        assert_eq!(s.cache_read_tokens, 120);

        // Per-provider reflects the same aggregation.
        let oa = s
            .by_provider
            .iter()
            .find(|r| r.provider == "openai")
            .unwrap();
        assert_eq!(oa.output_events, 2);
        assert_eq!(oa.output_after, 59);
        assert_eq!(oa.output_before, 0);

        // Per-model billing aggregates: usage sums, the legacy fresh-input fallback
        // (row 2 has no fresh_input → max(50 − 70, 0) = 0), and the shaped-output split
        // (only row 1 was shaped → 42 of the 59 output tokens).
        let models = t.by_model().unwrap();
        let gpt = models
            .iter()
            .find(|m| m.model.as_deref() == Some("gpt-4o"))
            .unwrap();
        assert_eq!(gpt.cache_read, 120);
        assert_eq!(gpt.cache_write, 12);
        assert_eq!(gpt.fresh_input_est, 80, "row1 usage + row2 fallback of 0");
        assert_eq!(gpt.output_after_shaped, 42);

        // Frozen-zone meter: row 1 metered (frozen NULL → both rows here are pre-meter),
        // so the metered sums stay zero — see `frozen_meter_sums_metered_rows_only`.
        assert_eq!(s.frozen_input_tokens, 0);
        assert_eq!(s.metered_input_before, 0);
        let m = models
            .iter()
            .find(|m| m.model.as_deref() == Some("m"))
            .unwrap();
        assert_eq!(
            m.fresh_input_est, 30,
            "no usage, no cache → fallback to input_after"
        );
        assert_eq!(m.output_after_shaped, 0, "shaped unknown → not credited");
    }

    #[test]
    fn prune_caps_to_most_recent_rows() {
        let t = Tracker::open_in_memory().unwrap();
        for _ in 0..10 {
            t.record(&rec("openai", true, 10, 5)).unwrap();
        }
        let deleted = t.prune(4, None).unwrap();
        assert_eq!(deleted, 6, "10 rows capped to 4 → 6 deleted");
        assert_eq!(t.summary().unwrap().events, 4);
    }

    #[test]
    fn prune_drops_rows_older_than_max_age() {
        let t = Tracker::open_in_memory().unwrap();
        // Three ancient rows (explicit old ts), two fresh.
        for _ in 0..3 {
            t.record_with_ts(&rec("openai", true, 10, 5), "2000-01-01T00:00:00+00:00")
                .unwrap();
        }
        t.record(&rec("openai", true, 10, 5)).unwrap();
        t.record(&rec("openai", true, 10, 5)).unwrap();

        let deleted = t.prune(DEFAULT_MAX_ROWS, Some(30)).unwrap();
        assert_eq!(deleted, 3, "only the three >30d-old rows are dropped");
        assert_eq!(t.summary().unwrap().events, 2);
    }

    #[test]
    fn prune_without_age_keeps_old_rows_within_cap() {
        let t = Tracker::open_in_memory().unwrap();
        t.record_with_ts(&rec("openai", true, 10, 5), "2000-01-01T00:00:00+00:00")
            .unwrap();
        // No age policy and under the cap → the ancient row survives.
        let deleted = t.prune(DEFAULT_MAX_ROWS, None).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(t.summary().unwrap().events, 1);
    }

    #[test]
    fn file_ledger_enables_wal_and_busy_timeout() {
        // WAL + a non-zero busy timeout protect the always-on daemon writer against the
        // the `status` TUI reader. (In-memory DBs can't run WAL, so test a file path.)
        let dir = std::env::temp_dir().join(format!("llmtrim_wal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.db");
        let t = Tracker::open_at(&path).expect("open file ledger");
        let mode: String = t
            .conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal", "WAL journal mode is set");
        let timeout: i64 = t
            .conn
            .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
            .unwrap();
        assert!(timeout >= 2000, "busy_timeout set (got {timeout})");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_is_idempotent_on_existing_columns() {
        // Re-running migrate() must not error on the additive columns it already added
        // (the duplicate-column ALTER is the expected, swallowed case).
        let t = Tracker::open_in_memory().unwrap();
        t.migrate()
            .expect("second migrate swallows duplicate-column ALTERs");
        t.record(&rec("openai", true, 10, 5)).unwrap();
        assert_eq!(t.summary().unwrap().events, 1);
    }

    #[test]
    fn transport_reset_outcome_round_trips() {
        // Transport failures used to look like anonymous out=0 rows. The outcome column
        // lets status/monitor distinguish a reset from a real zero-output completion.
        let t = Tracker::open_in_memory().unwrap();
        let mut r = rec("anthropic", true, 100, 80);
        r.outcome = Some("transport_reset".into());
        t.record(&r).unwrap();
        let outcome: Option<String> = t
            .conn
            .query_row(
                "SELECT outcome FROM compressions WHERE provider = 'anthropic'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(outcome.as_deref(), Some("transport_reset"));
        // Re-migrate must keep the column and not reject a second insert.
        t.migrate().expect("outcome ALTER is idempotent");
        t.record(&rec("openai", true, 10, 5)).unwrap();
        assert_eq!(t.summary().unwrap().events, 2);
    }

    #[test]
    fn duplicate_column_classifier_matches_only_that_error() {
        // The classifier underpinning #2: only the duplicate-column ALTER is swallowed; a
        // genuine failure (here, a syntax error) is reported as distinct.
        let t = Tracker::open_in_memory().unwrap();
        let dup = t
            .conn
            .execute("ALTER TABLE compressions ADD COLUMN model TEXT", [])
            .expect_err("model already exists");
        assert!(is_duplicate_column(&dup), "duplicate column recognized");
        let other = t
            .conn
            .execute("ALTER TABLE compressions ADD COLUMN", [])
            .expect_err("malformed ALTER");
        assert!(
            !is_duplicate_column(&other),
            "non-duplicate error not swallowed"
        );
    }
}
