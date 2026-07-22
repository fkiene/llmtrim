//! `guard` — a Claude Code `UserPromptSubmit` hook that stops the first turn of a resumed,
//! cold-cache, large-context session so the user sees what it costs *before* paying it.
//!
//! Past the prompt-cache TTL (the same 1h Claude Code asks for and the status line uses)
//! a resumed session re-writes its whole context on the next request, billed at the cache-write
//! rate. Nothing at the prompt says so. So on the first submit after a long idle gap on a big
//! context, emit a JSON block decision and exit 2 — Claude Code blocks the prompt with no API
//! call, and a resend goes straight through.
//!
//! The block response sets `suppressOriginalPrompt` so Claude Code does not append the blocked
//! text as `Original prompt:` (that echo nested the warning inside itself on every resend). The
//! blocked text is saved to disk instead; on the *next* submit that is allowed through, if the
//! user typed something different, the saved text is reinjected via `additionalContext` so it
//! is not lost when Claude Code cleared the input box.
//!
//! Data comes from Claude Code's own transcript, not llmtrim's ledger: the ledger's `session_id`
//! is a hash of the system prompt, so a *subagent* turn (different system prompt, same Claude Code
//! session) would mask a stale main conversation — and a session that never went through the proxy
//! has no ledger row at all. The transcript carries `isSidechain`, which excludes subagent turns
//! exactly. The ledger is consulted only to price a `sub`-rerouted backend, never to decide.
//!
//! Every fallible step maps to exit 0. A bug in here must never block a prompt.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::statusline::CACHE_TTL_SECS;

/// Context size (tokens) below which a cold turn isn't worth interrupting anyone for.
const MIN_TOKENS: i64 = 100_000;
/// A 1h-TTL cache write is billed at 2x the base input rate. The cold turn pays exactly that,
/// which is why the prototype's plain input rate understated it ~2x.
const CACHE_WRITE_MULTIPLIER: f64 = 2.0;

// ── the hook payload (only the fields we need) ───────────────────────────────────

/// Claude Code's `UserPromptSubmit` JSON on stdin. It carries no token/context fields — those
/// live in the *statusline* blob, a different payload — so the numbers come from the transcript.
struct HookInput {
    session_id: String,
    transcript_path: PathBuf,
    /// What the user typed. Blocking erases it from the input box, so it is saved to disk.
    prompt: String,
}

fn parse_hook(input: &str) -> Option<HookInput> {
    let v: Value = serde_json::from_str(input).ok()?;
    let transcript = v.get("transcript_path").and_then(Value::as_str)?;
    if transcript.is_empty() {
        return None;
    }
    Some(HookInput {
        session_id: v
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        transcript_path: PathBuf::from(transcript),
        prompt: v
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

// ── the transcript ───────────────────────────────────────────────────────────────

/// What the transcript says about *this* conversation (subagent turns excluded).
struct Scan {
    /// Newest timestamp among non-sidechain entries — the point we are resuming from.
    last_ts: DateTime<Utc>,
    /// Input the next request re-sends: everything the newest non-sidechain assistant turn was
    /// billed for on input (fresh + cache write + cache read).
    tokens: i64,
    model: String,
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

/// Scan the transcript JSONL. Unparseable lines are skipped, not fatal — Claude Code appends
/// while we read, so a torn last line is normal.
fn scan(reader: impl BufRead) -> Option<Scan> {
    let mut last_ts: Option<DateTime<Utc>> = None;
    let mut tokens = 0_i64;
    let mut model = String::new();

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let Ok(entry) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        // Subagent turns run on their own context; they are not the one we would resume.
        if entry.get("isSidechain").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(ts) = entry
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_ts)
            && last_ts.is_none_or(|prev| ts > prev)
        {
            last_ts = Some(ts);
        }
        if let Some(usage) = entry.pointer("/message/usage") {
            let field = |k: &str| usage.get(k).and_then(Value::as_i64).unwrap_or(0);
            tokens = field("input_tokens")
                + field("cache_creation_input_tokens")
                + field("cache_read_input_tokens");
            if let Some(m) = entry
                .pointer("/message/model")
                .and_then(Value::as_str)
                .filter(|m| !m.is_empty())
            {
                model = m.to_string();
            }
        }
    }

    Some(Scan {
        last_ts: last_ts?,
        tokens,
        model,
    })
}

/// The rule, shared with the status line's cold-cache signal: idle past the cache TTL, on a
/// context big enough that re-writing it costs real money.
fn should_warn(idle_secs: i64, tokens: i64) -> bool {
    idle_secs >= CACHE_TTL_SECS && tokens >= MIN_TOKENS
}

// ── the once-per-gap marker ──────────────────────────────────────────────────────

/// Markers live beside the ledger, under llmtrim's existing state dir.
fn guard_dir() -> Result<PathBuf> {
    let db = crate::tracking::db_path()?;
    Ok(db
        .parent()
        .context("ledger path has no parent directory")?
        .join("guard"))
}

/// The marker for one session, under `dir`. Taking the directory as an argument keeps
/// [`decide_in`] testable without reaching for the process-global `LLMTRIM_DB_PATH`.
fn marker_path(dir: &Path, session_id: &str) -> PathBuf {
    // A session id from the payload is a UUID, but never let it walk out of the state dir.
    let safe: String = session_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    dir.join(format!("{safe}.acked"))
}

/// Whether this exact idle gap was already warned about. The gap is keyed by the timestamp we
/// are resuming from, so a *later* gap in the same session re-arms the warning.
fn already_acked(path: &Path, gap_id: &str) -> bool {
    std::fs::read_to_string(path).is_ok_and(|s| s.trim() == gap_id)
}

/// Path of the stashed blocked prompt for one session.
fn prompt_path(dir: &Path, session_id: &str) -> PathBuf {
    marker_path(dir, session_id).with_extension("prompt")
}

/// Stash the blocked prompt beside its marker. Claude Code does not restore it to the input box,
/// so this is the only copy; failing to write it must not stop the warning.
fn save_prompt(dir: &Path, session_id: &str, prompt: &str) -> Option<PathBuf> {
    if prompt.is_empty() {
        return None;
    }
    let path = prompt_path(dir, session_id);
    std::fs::create_dir_all(dir).ok()?;
    std::fs::write(&path, prompt).ok()?;
    Some(path)
}

/// Take the previously blocked prompt, if any. Returns `None` when there is nothing to reinject
/// (no file, empty file, or the user retyped the same text). Always removes the file so a later
/// gap cannot re-surface a stale draft.
fn take_saved_prompt(dir: &Path, session_id: &str, current: &str) -> Option<String> {
    let path = prompt_path(dir, session_id);
    let saved = std::fs::read_to_string(&path).ok();
    let _ = std::fs::remove_file(&path);
    let saved = saved?.trim_end().to_string();
    if saved.is_empty() || saved == current.trim_end() {
        return None;
    }
    Some(saved)
}

fn ack(path: &Path, gap_id: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, gap_id).with_context(|| format!("failed to write {}", path.display()))
}

// ── cost + message ───────────────────────────────────────────────────────────────

/// What the cold turn costs, in dollars, at the cache-write rate. `None` when the model isn't in
/// the pricing snapshot — the warning then simply carries no figure.
fn cold_turn_cost(tokens: i64, model: &str) -> Option<f64> {
    // Claude Code tags some ids with a context suffix (`claude-opus-4-8[1m]`), which no registry
    // knows; the base id prices the same tokens.
    let base = model.split('[').next().unwrap_or(model);
    let (input_per_1m, _) = crate::monitor::llm_prices(base)?;
    Some(tokens as f64 / 1_000_000.0 * input_per_1m * CACHE_WRITE_MULTIPLIER)
}

/// The model to price the cold turn at: the `sub` backend that actually served this session's last
/// turn, when there is one, else the model the transcript recorded. Ledger absence never changes
/// the decision — only the number.
fn pricing_model(session_id: &str, transcript_model: &str) -> String {
    crate::statusline::session_sub_model(session_id).unwrap_or_else(|| transcript_model.to_string())
}

fn human_idle(secs: i64) -> String {
    let (h, m) = (secs / 3600, (secs % 3600) / 60);
    if h > 0 {
        format!("{h}h {m}m")
    } else {
        format!("{m}m")
    }
}

/// The warning. It must not claim `/compact` saves money: `/compact` reads the same cold context
/// to summarise it (paying the charge), then the next turn writes a fresh cache for the summary
/// (paying again) — measured on real captures at 128,758 then 45,131 tokens. Resending pays once.
///
/// The user's prompt is not echoed here. Claude Code used to append it as `Original prompt:`,
/// which nested the warning inside itself on every resend; the block response now sets
/// `suppressOriginalPrompt` to stop that. The text is *saved* instead — a blocked prompt is not
/// restored to the input box (observed: the resend came back as `promptSource: typed`), so without
/// this the text would be lost. On the next allowed submit it is reinjected if the user typed
/// something else.
fn message(idle_secs: i64, tokens: i64, cost: Option<f64>) -> String {
    let cost = match cost {
        Some(c) => format!(" — about ${c:.2}"),
        None => String::new(),
    };
    format!(
        "Idle {}, {:.0}k tokens of context. The prompt cache has expired, so the next turn \
         re-writes the whole context{} before any work happens.\n\
         \n\
         You pay that whichever way you go: /compact pays it too (it reads the full context to \
         summarise it), and only comes out ahead if you are staying for several more turns. To \
         avoid the charge entirely, ask in a fresh session.\n\
         \n\
         Not sent. Your message is saved and will be reinjected on the next turn if you type \
         something else. Resend to continue as-is.",
        human_idle(idle_secs),
        tokens as f64 / 1000.0,
        cost,
    )
}

/// JSON body Claude Code reads on stdout for a block decision. `suppressOriginalPrompt` keeps
/// the blocked text out of the warning so a resend cannot nest the previous block message.
fn block_stdout(reason: &str) -> String {
    serde_json::json!({
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "suppressOriginalPrompt": true,
        },
    })
    .to_string()
}

/// JSON body that reinjects a previously blocked prompt as model context. Used on the pass that
/// follows a block, when the user typed something different from what was saved.
fn reinject_stdout(saved: &str) -> String {
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": format!(
                "Previous prompt that was blocked by the cold-cache guard (not restored to the \
                 input box by Claude Code):\n\n{saved}"
            ),
        },
    })
    .to_string()
}

// ── the hook entrypoint ──────────────────────────────────────────────────────────

/// Exit code for "block this prompt". Claude Code reads the JSON reason from stdout and makes no
/// API call. Exit 2 with plain stderr still works, but JSON is what unlocks `suppressOriginalPrompt`.
const BLOCK: i32 = 2;
const PASS: i32 = 0;

/// What the hook should emit. Separated from the exit code so tests can assert the JSON body
/// without capturing process stdout.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    Pass,
    /// Allow the prompt, but attach previously blocked text as model context.
    PassWith(String),
    Block(String),
}

impl Verdict {
    fn code(&self) -> i32 {
        match self {
            Verdict::Pass | Verdict::PassWith(_) => PASS,
            Verdict::Block(_) => BLOCK,
        }
    }

    fn emit(self) {
        match self {
            Verdict::Pass => {}
            Verdict::PassWith(body) | Verdict::Block(body) => println!("{body}"),
        }
    }
}

/// Decide against an explicit marker directory, so tests drive the real path without the
/// process-global state dir. [`run`] maps any panic to `PASS`.
fn decide_in(input: &str, now: DateTime<Utc>, dir: &Path) -> Verdict {
    let Some(hook) = parse_hook(input) else {
        return Verdict::Pass;
    };
    let Ok(file) = std::fs::File::open(&hook.transcript_path) else {
        return Verdict::Pass;
    };
    let Some(s) = scan(std::io::BufReader::new(file)) else {
        return Verdict::Pass;
    };

    let idle = now.signed_duration_since(s.last_ts).num_seconds();
    if !should_warn(idle, s.tokens) {
        // Not a cold-cache block, but a previous block in this session may have stashed a draft.
        return match take_saved_prompt(dir, &hook.session_id, &hook.prompt) {
            Some(saved) => Verdict::PassWith(reinject_stdout(&saved)),
            None => Verdict::Pass,
        };
    }

    // Warn once per idle gap; a resend then goes straight through (and may reinject).
    let gap_id = s.last_ts.to_rfc3339();
    let marker = marker_path(dir, &hook.session_id);
    if already_acked(&marker, &gap_id) {
        return match take_saved_prompt(dir, &hook.session_id, &hook.prompt) {
            Some(saved) => Verdict::PassWith(reinject_stdout(&saved)),
            None => Verdict::Pass,
        };
    }
    // Ack before printing: if the marker cannot be written we fail open rather than block
    // without recording that we did.
    if ack(&marker, &gap_id).is_err() {
        return Verdict::Pass;
    }

    // Best effort: losing the saved copy is not a reason to let the expensive turn through.
    let _ = save_prompt(dir, &hook.session_id, &hook.prompt);

    let model = pricing_model(&hook.session_id, &s.model);
    let reason = message(idle, s.tokens, cold_turn_cost(s.tokens, &model));
    Verdict::Block(block_stdout(&reason))
}

/// Run the hook: read stdin, print JSON to stdout, return the exit code for `main` to exit with.
///
/// Fail open, unconditionally: an error, a panic, a missing transcript — anything but a genuine
/// stale-and-large session — returns 0 and the prompt goes through.
pub fn run() -> i32 {
    fail_open(|| {
        use std::io::Read;
        let mut input = String::new();
        if std::io::stdin().read_to_string(&mut input).is_err() {
            return PASS;
        }
        let verdict = match guard_dir() {
            Ok(dir) => decide_in(&input, Utc::now(), &dir),
            Err(_) => Verdict::Pass,
        };
        let code = verdict.code();
        verdict.emit();
        code
    })
}

/// The outer boundary: a panic in the handler must not block a prompt either.
fn fail_open(f: impl FnOnce() -> i32 + std::panic::UnwindSafe) -> i32 {
    std::panic::catch_unwind(f).unwrap_or(PASS)
}

// ── install / uninstall (wire ~/.claude/settings.json) ───────────────────────────
//
// Unlike `statusLine` (a singleton key), `hooks.UserPromptSubmit` is an ARRAY that may already
// hold the user's own hooks. So we find-or-create the array and replace *our* entry in place —
// never clobber the object.

/// One `UserPromptSubmit` matcher group holding just our command.
fn guard_hook_entry() -> Value {
    serde_json::json!({
        "hooks": [{
            "type": "command",
            "command": crate::statusline::exe_command("guard"),
            "timeout": 10,
        }]
    })
}

/// Whether a hook command is one `llmtrim guard install` wrote: the llmtrim binary invoked with
/// `guard`, or — for an install renamed or vendored under another file name — the exact command
/// this binary writes, so uninstall/refresh still recognise their own entry.
fn is_llmtrim_guard_command(command: &str) -> bool {
    crate::statusline::is_llmtrim_command(command, "guard")
        || command == crate::statusline::exe_command("guard")
}

/// Whether a `UserPromptSubmit` group is ours (any of its commands is our guard command).
fn is_ours(group: &Value) -> bool {
    group
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|hooks| {
            hooks.iter().any(|h| {
                h.get("command")
                    .and_then(Value::as_str)
                    .is_some_and(is_llmtrim_guard_command)
            })
        })
}

/// Add (or refresh) our hook in a parsed settings object, preserving every other hook. Pure
/// transform, so the merge is unit-testable.
fn set_guard_hook(settings: &mut Value, path: &Path) -> Result<()> {
    let obj = settings
        .as_object_mut()
        .with_context(|| format!("{} is not a JSON object", path.display()))?;
    let hooks = obj
        .entry("hooks")
        .or_insert_with(|| Value::Object(Default::default()))
        .as_object_mut()
        .with_context(|| format!("`hooks` in {} is not a JSON object", path.display()))?;
    let list = hooks
        .entry("UserPromptSubmit")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .with_context(|| {
            format!(
                "`hooks.UserPromptSubmit` in {} is not a JSON array",
                path.display()
            )
        })?;
    match list.iter().position(is_ours) {
        Some(i) => list[i] = guard_hook_entry(),
        None => list.push(guard_hook_entry()),
    }
    Ok(())
}

/// Remove only our hook, pruning the array/object if we were the last one in it. Returns whether
/// anything was removed. Pure transform.
fn clear_guard_hook(settings: &mut Value, path: &Path) -> Result<bool> {
    let obj = settings
        .as_object_mut()
        .with_context(|| format!("{} is not a JSON object", path.display()))?;
    let Some(hooks) = obj.get_mut("hooks").and_then(Value::as_object_mut) else {
        return Ok(false);
    };
    let Some(list) = hooks
        .get_mut("UserPromptSubmit")
        .and_then(Value::as_array_mut)
    else {
        return Ok(false);
    };
    let before = list.len();
    list.retain(|g| !is_ours(g));
    let removed = list.len() != before;
    if list.is_empty() {
        hooks.remove("UserPromptSubmit");
    }
    if hooks.is_empty() {
        obj.remove("hooks");
    }
    Ok(removed)
}

fn claude_settings_path() -> Result<PathBuf> {
    crate::statusline::claude_settings_path()
}

/// Ownership of the guard hook relative to this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnedStatus {
    Missing,
    Stale,
    Current,
}

/// Whether ensure should wire / refresh the cold-cache guard.
pub fn owned_status() -> OwnedStatus {
    let Ok(path) = claude_settings_path() else {
        return OwnedStatus::Missing;
    };
    let Ok(s) = std::fs::read_to_string(&path) else {
        return OwnedStatus::Missing;
    };
    let Ok(settings) = serde_json::from_str::<Value>(&s) else {
        return OwnedStatus::Missing;
    };
    owned_status_of(&settings)
}

fn owned_status_of(settings: &Value) -> OwnedStatus {
    let Some(list) = settings
        .get("hooks")
        .and_then(|h| h.get("UserPromptSubmit"))
        .and_then(Value::as_array)
    else {
        return OwnedStatus::Missing;
    };
    let Some(group) = list.iter().find(|g| is_ours(g)) else {
        return OwnedStatus::Missing;
    };
    let desired = crate::statusline::exe_command("guard");
    let current = group
        .get("hooks")
        .and_then(Value::as_array)
        .and_then(|hooks| {
            hooks.iter().find_map(|h| {
                h.get("command")
                    .and_then(Value::as_str)
                    .filter(|c| is_llmtrim_guard_command(c))
            })
        });
    match current {
        Some(c) if c == desired => OwnedStatus::Current,
        Some(_) => OwnedStatus::Stale,
        None => OwnedStatus::Missing,
    }
}

/// Install or refresh our guard hook. Returns `true` when the file changed.
pub fn sync_owned() -> Result<bool> {
    match owned_status() {
        OwnedStatus::Current => Ok(false),
        OwnedStatus::Missing | OwnedStatus::Stale => {
            wire()?;
            Ok(true)
        }
    }
}

/// Wire the hook into `~/.claude/settings.json` (merging, never clobbering). Returns the file it
/// wrote, so `setup` can report it without printing its own line.
pub fn wire() -> Result<PathBuf> {
    wire_at(claude_settings_path()?)
}

/// [`wire`] against an explicit settings file, so tests exercise the read-modify-write round
/// trip instead of only the pure merge.
fn wire_at(path: PathBuf) -> Result<PathBuf> {
    let mut settings: Value = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s)
            .with_context(|| format!("{} is not valid JSON", path.display()))?,
        Err(_) => Value::Object(Default::default()),
    };
    set_guard_hook(&mut settings, &path)?;
    crate::statusline::atomic_write_json(&path, &settings)?;
    Ok(path)
}

/// `guard install` — wire the hook, or just print the settings snippet with `--print`.
pub fn install(print: bool) -> Result<()> {
    if print {
        let snippet = serde_json::json!({ "hooks": { "UserPromptSubmit": [guard_hook_entry()] } });
        println!("{}", serde_json::to_string_pretty(&snippet)?);
        return Ok(());
    }
    let path = wire()?;
    println!(
        "Wired the llmtrim guard into {}. Restart Claude Code to arm it.",
        path.display()
    );
    Ok(())
}

/// Remove our hook from `~/.claude/settings.json`, leaving the user's other hooks alone.
/// Returns whether one was present.
pub fn unwire() -> Result<bool> {
    unwire_at(claude_settings_path()?)
}

/// [`unwire`] against an explicit settings file. See [`wire_at`].
fn unwire_at(path: PathBuf) -> Result<bool> {
    let Ok(s) = std::fs::read_to_string(&path) else {
        return Ok(false);
    };
    let mut settings: Value = serde_json::from_str(&s)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    if !clear_guard_hook(&mut settings, &path)? {
        return Ok(false);
    }
    crate::statusline::atomic_write_json(&path, &settings)?;
    Ok(true)
}

/// `guard uninstall`.
pub fn uninstall() -> Result<()> {
    let path = claude_settings_path()?;
    if unwire()? {
        println!("Removed the llmtrim guard from {}.", path.display());
    } else {
        println!("No llmtrim guard found in {}.", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A transcript entry, as Claude Code writes them.
    fn entry(ts: &str, tokens: i64, sidechain: bool) -> String {
        serde_json::json!({
            "timestamp": ts,
            "isSidechain": sidechain,
            "message": {
                "model": "claude-opus-4-8",
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": tokens - 10,
                    "cache_read_input_tokens": 0,
                },
            },
        })
        .to_string()
    }

    /// Write a transcript + hook payload into a fresh temp dir; returns (payload, dir).
    fn fixture(lines: &[String]) -> (String, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "llmtrim-guard-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let transcript = dir.join("transcript.jsonl");
        std::fs::write(&transcript, lines.join("\n")).unwrap();
        let payload = serde_json::json!({
            "session_id": format!("sess-{}", dir.file_name().unwrap().to_string_lossy()),
            "transcript_path": transcript.display().to_string(),
            "prompt": "carry on",
        })
        .to_string();
        (payload, dir)
    }

    fn ago(secs: i64) -> String {
        (Utc::now() - chrono::Duration::seconds(secs)).to_rfc3339()
    }

    /// The real [`decide_in`], with markers kept inside the test's own dir. Drives the whole
    /// path — scan, trigger, marker, message — not a re-implementation of it.
    fn verdict(payload: &str, dir: &Path) -> i32 {
        decide_in(payload, Utc::now(), dir).code()
    }

    fn decide_verdict(payload: &str, dir: &Path) -> Verdict {
        decide_in(payload, Utc::now(), dir)
    }

    #[test]
    fn cold_and_big_blocks() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, false)]);
        assert_eq!(verdict(&payload, &dir), BLOCK);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn warm_session_passes() {
        let (payload, dir) = fixture(&[entry(&ago(60), 150_000, false)]);
        assert_eq!(verdict(&payload, &dir), PASS);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cold_but_small_passes() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 20_000, false)]);
        assert_eq!(verdict(&payload, &dir), PASS);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_newer_subagent_turn_does_not_hide_a_stale_conversation() {
        // The regression the ledger-based design would have shipped: the main conversation is
        // stale, but a subagent answered a minute ago. Only `isSidechain` separates them.
        let (payload, dir) = fixture(&[
            entry(&ago(7200), 150_000, false),
            entry(&ago(30), 150_000, true),
        ]);
        assert_eq!(verdict(&payload, &dir), BLOCK);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sidechain_only_transcript_never_fires() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, true)]);
        assert_eq!(verdict(&payload, &dir), PASS);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn once_per_gap_then_again_on_the_next_gap() {
        let dir = std::env::temp_dir().join(format!("llmtrim-guard-marker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("sess.acked");
        let gap = ago(7200);

        assert!(!already_acked(&marker, &gap), "first submit warns");
        ack(&marker, &gap).unwrap();
        assert!(already_acked(&marker, &gap), "the resend goes through");

        // A later idle gap in the same session re-arms the warning.
        let next_gap = ago(3700);
        assert!(!already_acked(&marker, &next_gap));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fails_open_on_bad_input() {
        let dir = std::env::temp_dir();
        assert_eq!(verdict("not json", &dir), PASS);
        assert_eq!(verdict("{}", &dir), PASS);
        // Missing transcript / malformed stdin: decide_in returns Pass without a marker dir.
        assert_eq!(
            decide_in(
                r#"{"transcript_path":"/nonexistent/x.jsonl"}"#,
                Utc::now(),
                &dir
            )
            .code(),
            PASS,
            "missing transcript fails open"
        );
        assert_eq!(
            decide_in("garbage", Utc::now(), &dir).code(),
            PASS,
            "malformed stdin fails open"
        );
    }

    #[test]
    fn a_transcript_of_junk_lines_fails_open() {
        let (payload, dir) = fixture(&["not json".into(), "{".into()]);
        assert_eq!(verdict(&payload, &dir), PASS);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_panic_in_the_handler_never_blocks() {
        // The outer boundary is what guarantees "a bug here can't cost the user a turn".
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let code = fail_open(|| panic!("bug"));
        std::panic::set_hook(prev);
        assert_eq!(code, PASS);
    }

    #[test]
    fn scan_sums_the_whole_resent_context() {
        let line = serde_json::json!({
            "timestamp": "2026-07-14T10:00:00Z",
            "message": {
                "model": "claude-opus-4-8",
                "usage": {
                    "input_tokens": 5,
                    "cache_creation_input_tokens": 1000,
                    "cache_read_input_tokens": 120_000,
                },
            },
        })
        .to_string();
        let s = scan(line.as_bytes()).unwrap();
        assert_eq!(s.tokens, 121_005);
        assert_eq!(s.model, "claude-opus-4-8");
    }

    #[test]
    fn message_does_not_sell_compact_as_a_saving() {
        let m = message(7200, 150_000, Some(1.5));
        assert!(m.contains("2h 0m") && m.contains("150k"), "{m}");
        assert!(m.contains("$1.50"), "cost at the cache-write rate: {m}");
        assert!(
            m.contains("/compact pays it too"),
            "compact is not a money-saver: {m}"
        );
        assert!(
            !m.contains("Run /compact"),
            "no compact recommendation: {m}"
        );
        assert!(!m.contains("carry on"), "never echo the prompt: {m}");
        // No price for an unknown model ⇒ no figure, but still a warning.
        assert!(!message(7200, 150_000, None).contains('$'));
    }

    #[test]
    fn block_stdout_suppresses_the_original_prompt_echo() {
        let body = block_stdout("idle warning");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["decision"], "block");
        assert_eq!(v["reason"], "idle warning");
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit");
        assert_eq!(
            v["hookSpecificOutput"]["suppressOriginalPrompt"], true,
            "Claude Code must not append 'Original prompt:' — that nests the warning"
        );
        // The reason itself never carries the user's text either.
        assert!(!body.contains("carry on"));
    }

    #[test]
    fn a_block_emits_suppress_json_not_plain_stderr() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, false)]);
        match decide_verdict(&payload, &dir) {
            Verdict::Block(body) => {
                let v: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(v["decision"], "block");
                assert_eq!(v["hookSpecificOutput"]["suppressOriginalPrompt"], true);
                assert!(
                    v["reason"].as_str().unwrap().contains("prompt cache"),
                    "reason carries the warning: {v}"
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn next_submit_reinjects_a_different_blocked_prompt() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, false)]);
        assert_eq!(verdict(&payload, &dir), BLOCK, "first submit is stopped");

        // User types something new after the block cleared the input box.
        let mut v: Value = serde_json::from_str(&payload).unwrap();
        v["prompt"] = Value::String("something else".into());
        // Still cold, but the gap is already acked → pass, with reinjection.
        match decide_verdict(&v.to_string(), &dir) {
            Verdict::PassWith(body) => {
                let out: Value = serde_json::from_str(&body).unwrap();
                let ctx = out["hookSpecificOutput"]["additionalContext"]
                    .as_str()
                    .unwrap();
                assert!(ctx.contains("carry on"), "saved prompt reinjected: {ctx}");
                assert!(
                    !prompt_path(&dir, v["session_id"].as_str().unwrap()).exists(),
                    "stash is consumed so it cannot reappear later"
                );
            }
            other => panic!("expected PassWith reinjection, got {other:?}"),
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resending_the_same_text_does_not_reinject() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, false)]);
        assert_eq!(verdict(&payload, &dir), BLOCK);
        // Exact resend: Claude Code may still hold the text, or the user retyped it.
        assert_eq!(
            decide_verdict(&payload, &dir),
            Verdict::Pass,
            "identical text is not double-injected"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cold_turn_is_priced_at_the_cache_write_rate() {
        // Whatever the snapshot says, the cold write is 2x the base input rate.
        let Some((input_per_1m, _)) = crate::monitor::llm_prices("claude-opus-4-8") else {
            return; // model not in the pricing snapshot on this build — nothing to assert
        };
        let cost = cold_turn_cost(1_000_000, "claude-opus-4-8[1m]").unwrap();
        assert!(
            (cost - input_per_1m * 2.0).abs() < 1e-9,
            "1M cold tokens = 2x input rate: {cost}"
        );
    }

    #[test]
    fn a_hostile_session_id_cannot_walk_the_marker_out_of_its_directory() {
        let dir = Path::new("/tmp/llmtrim-guard-test");
        let p = marker_path(dir, "../../etc/passwd");
        assert_eq!(p.parent().unwrap(), dir, "stays put");
        assert_eq!(
            p.file_name().unwrap().to_string_lossy(),
            "------etc-passwd.acked"
        );
    }

    #[test]
    fn markers_live_beside_the_ledger() {
        // The directory itself is only resolved in production; `decide_in` takes it as an
        // argument so the tests above never touch the real state dir.
        assert_eq!(guard_dir().unwrap().file_name().unwrap(), "guard");
    }

    // ── settings merge ──────────────────────────────────────────────────────────

    fn p() -> &'static Path {
        Path::new("settings.json")
    }

    fn commands(settings: &Value) -> Vec<String> {
        settings["hooks"]["UserPromptSubmit"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|g| g["hooks"].as_array().unwrap())
            .map(|h| h["command"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn install_into_empty_settings_adds_the_hook() {
        let mut settings = serde_json::json!({});
        set_guard_hook(&mut settings, p()).unwrap();
        assert_eq!(commands(&settings).len(), 1);
        assert!(is_llmtrim_guard_command(&commands(&settings)[0]));
    }

    #[test]
    fn install_preserves_unrelated_user_hooks() {
        let mut settings = serde_json::json!({
            "hooks": {
                "UserPromptSubmit": [
                    { "hooks": [{ "type": "command", "command": "~/.claude/hooks/mine" }] }
                ],
                "Stop": [{ "hooks": [{ "type": "command", "command": "notify" }] }],
            }
        });
        set_guard_hook(&mut settings, p()).unwrap();
        let cmds = commands(&settings);
        assert_eq!(cmds.len(), 2, "ours is appended, not substituted: {cmds:?}");
        assert_eq!(cmds[0], "~/.claude/hooks/mine");
        assert!(is_llmtrim_guard_command(&cmds[1]));
        assert_eq!(
            settings["hooks"]["Stop"][0]["hooks"][0]["command"],
            "notify"
        );
    }

    #[test]
    fn install_refreshes_our_entry_in_place_instead_of_duplicating_it() {
        let mut settings = serde_json::json!({
            "hooks": { "UserPromptSubmit": [
                { "hooks": [{ "type": "command",
                              "command": "/opt/homebrew/Cellar/llmtrim/0.9.4/bin/llmtrim guard" }] }
            ]}
        });
        set_guard_hook(&mut settings, p()).unwrap();
        let cmds = commands(&settings);
        assert_eq!(cmds.len(), 1, "refreshed in place: {cmds:?}");
        assert_ne!(
            cmds[0],
            "/opt/homebrew/Cellar/llmtrim/0.9.4/bin/llmtrim guard"
        );
        assert!(is_llmtrim_guard_command(&cmds[0]));
    }

    #[test]
    fn uninstall_removes_only_ours_and_prunes_empties() {
        let mut settings = serde_json::json!({ "theme": "dark" });
        set_guard_hook(&mut settings, p()).unwrap();
        assert!(clear_guard_hook(&mut settings, p()).unwrap());
        assert!(
            settings.get("hooks").is_none(),
            "an emptied hooks object is pruned: {settings}"
        );
        assert_eq!(settings["theme"], "dark");
        // A second removal is a no-op.
        assert!(!clear_guard_hook(&mut settings, p()).unwrap());

        // With a user hook alongside, only ours goes.
        let mut settings = serde_json::json!({
            "hooks": { "UserPromptSubmit": [
                { "hooks": [{ "type": "command", "command": "~/.claude/hooks/mine" }] }
            ]}
        });
        set_guard_hook(&mut settings, p()).unwrap();
        assert!(clear_guard_hook(&mut settings, p()).unwrap());
        assert_eq!(
            commands(&settings),
            vec!["~/.claude/hooks/mine".to_string()]
        );
    }

    #[test]
    fn recognizes_only_llmtrim_guard_commands() {
        assert!(is_llmtrim_guard_command(
            "/opt/homebrew/Cellar/llmtrim/0.9.4/bin/llmtrim guard"
        ));
        assert!(is_llmtrim_guard_command(
            r#""C:\Program Files\llmtrim\llmtrim.exe" guard"#
        ));
        assert!(!is_llmtrim_guard_command("llmtrim statusline"));
        assert!(!is_llmtrim_guard_command("~/.claude/hooks/guard"));
    }

    #[test]
    fn merge_rejects_a_non_object_settings_file() {
        let mut settings = serde_json::json!([1, 2, 3]);
        assert!(set_guard_hook(&mut settings, p()).is_err());
        assert!(clear_guard_hook(&mut settings, p()).is_err());
    }

    // ── the real file round trip ────────────────────────────────────────────────

    #[test]
    fn wire_creates_then_unwire_removes_a_settings_file_on_disk() {
        let (_, dir) = fixture(&[]);
        let settings = dir.join("settings.json");

        // A settings file that does not exist yet is created, not an error.
        wire_at(settings.clone()).unwrap();
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
        assert!(is_ours(&written["hooks"]["UserPromptSubmit"][0]));

        // Wiring twice must not duplicate the entry.
        wire_at(settings.clone()).unwrap();
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
        assert_eq!(
            written["hooks"]["UserPromptSubmit"]
                .as_array()
                .unwrap()
                .len(),
            1
        );

        assert!(unwire_at(settings.clone()).unwrap(), "ours was removed");
        assert!(
            !unwire_at(settings.clone()).unwrap(),
            "nothing left to remove"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unwire_on_a_missing_settings_file_is_not_an_error() {
        assert!(!unwire_at(PathBuf::from("/nonexistent/settings.json")).unwrap());
    }

    #[test]
    fn a_blocked_prompt_is_saved_because_claude_code_does_not_restore_it() {
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, false)]);
        assert_eq!(verdict(&payload, &dir), BLOCK);

        let saved: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "prompt"))
            .collect();
        assert_eq!(saved.len(), 1, "the blocked prompt is stashed");
        assert_eq!(
            std::fs::read_to_string(saved[0].path()).unwrap(),
            "carry on"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn blocking_then_resending_the_same_gap_goes_through() {
        // The whole production path, not a re-implementation of its trigger rule.
        let (payload, dir) = fixture(&[entry(&ago(7200), 150_000, false)]);
        assert_eq!(verdict(&payload, &dir), BLOCK, "first submit is stopped");
        assert_eq!(verdict(&payload, &dir), PASS, "the resend goes through");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
