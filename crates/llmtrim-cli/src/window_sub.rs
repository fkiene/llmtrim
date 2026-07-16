//! Private, per-Claude-Code-window subscription overrides.
//!
//! This deliberately lives outside the global llmtrim config: a slash command must not
//! restart the proxy or alter another Claude Code window.
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const TTL: Duration = Duration::from_secs(30 * 60);
const TOUCH: Duration = Duration::from_secs(60);
const COMMAND_NAME: &str = "sub";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Registry {
    #[serde(default)]
    windows: BTreeMap<String, Window>,
    #[serde(default)]
    sessions: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Window {
    /// `None` follows the global policy; a value is this window's explicit override.
    intent: Option<Intent>,
    touched: u64,
    #[serde(default)]
    last_provider: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    Enabled { provider: String },
    Disabled,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 160
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !value.contains("..")
}
fn valid_provider(value: &str) -> bool {
    matches!(value, "codex" | "kimi" | "grok")
}

pub fn registry_path() -> Result<PathBuf> {
    Ok(crate::daemon::home_dir()?.join("claude-window-sub.json"))
}
struct RegistryLock(PathBuf);

impl Drop for RegistryLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn lock_registry(path: &Path) -> Result<RegistryLock> {
    let lock = path.with_extension("lock");
    if let Some(parent) = lock.parent() {
        fs::create_dir_all(parent)?;
    }
    for _ in 0..200 {
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(_) => return Ok(RegistryLock(lock)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let stale = fs::metadata(&lock)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|age| age > Duration::from_secs(30));
                if stale {
                    let _ = fs::remove_file(&lock);
                } else {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            Err(e) => return Err(e).context("creating window registry lock"),
        }
    }
    bail!("timed out waiting for window registry lock")
}

fn load_at(path: &Path) -> Result<Registry> {
    if !path.exists() {
        return Ok(Registry::default());
    }
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
        bail!("refusing symlinked window registry");
    }
    let mut raw = String::new();
    fs::File::open(path)?.read_to_string(&mut raw)?;
    serde_json::from_str(&raw).context("corrupt window registry")
}
fn replace_file(tmp: &Path, destination: &Path) -> Result<()> {
    #[cfg(windows)]
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(tmp, destination)?;
    Ok(())
}

fn save_at(path: &Path, registry: &Registry) -> Result<()> {
    let parent = path.parent().context("registry has no parent")?;
    fs::create_dir_all(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    }
    let tmp = path.with_extension("json.tmp");
    if tmp.exists() && fs::symlink_metadata(&tmp)?.file_type().is_symlink() {
        bail!("refusing symlinked registry temp file");
    }
    let bytes = serde_json::to_vec_pretty(registry)?;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(&bytes)?;
    file.sync_all()?;
    replace_file(&tmp, path)?;
    Ok(())
}
fn prune(registry: &mut Registry, current: u64) {
    registry
        .windows
        .retain(|_, w| current.saturating_sub(w.touched) <= TTL.as_secs());
    registry
        .sessions
        .retain(|_, t| registry.windows.contains_key(t));
}
fn token() -> String {
    let mut b = [0u8; 24];
    if let Ok(mut f) = fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut b);
    }
    if b.iter().all(|x| *x == 0) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seed = nanos
            ^ ((std::process::id() as u128) << 64)
            ^ COUNTER.fetch_add(1, Ordering::Relaxed) as u128;
        for (i, x) in b.iter_mut().enumerate() {
            *x = seed.rotate_left((i * 7 % 128) as u32) as u8 ^ (i as u8).wrapping_mul(29);
        }
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}
/// Register a fresh startup/resume window, or retain its token across clear/compact.
pub fn session_start(session: &str, source: &str, existing: Option<&str>) -> Result<String> {
    if !valid(session) {
        bail!("invalid Claude session id");
    }
    let path = registry_path()?;
    let _lock = lock_registry(&path)?;
    let mut r = load_at(&path)?;
    let t = now();
    prune(&mut r, t);
    let token = if matches!(source, "clear" | "compact") {
        existing
            .filter(|x| valid(x) && r.windows.contains_key(*x))
            .map(str::to_owned)
            .unwrap_or_else(token)
    } else {
        token()
    };
    r.windows
        .entry(token.clone())
        .or_insert(Window {
            intent: None,
            touched: t,
            last_provider: None,
        })
        .touched = t;
    r.sessions.insert(session.to_owned(), token.clone());
    save_at(&path, &r)?;
    Ok(token)
}
pub fn session_end(session: &str, reason: &str) -> Result<()> {
    if !valid(session) {
        return Ok(());
    }
    let path = registry_path()?;
    let _lock = lock_registry(&path)?;
    let mut r = load_at(&path)?;
    if let Some(token) = r.sessions.remove(session)
        && reason != "clear"
    {
        r.windows.remove(&token);
        r.sessions.retain(|_, mapped| mapped != &token);
    }
    prune(&mut r, now());
    save_at(&path, &r)
}
pub fn set(session: &str, enabled: bool, provider: Option<&str>) -> Result<()> {
    if !valid(session) {
        bail!("invalid Claude session id");
    }
    let path = registry_path()?;
    let _lock = lock_registry(&path)?;
    let mut r = load_at(&path)?;
    prune(&mut r, now());
    let t = r
        .sessions
        .get(session)
        .cloned()
        .context("this Claude Code window is not registered; restart or resume it first")?;
    let w = r
        .windows
        .get_mut(&t)
        .context("window expired; restart or resume it first")?;
    w.touched = now();
    if enabled {
        let p = provider
            .map(str::to_owned)
            .or_else(|| w.last_provider.clone())
            .context(
                "no configured subscription provider; run llmtrim sub on codex, kimi, or grok first",
            )?;
        if !valid_provider(&p) {
            bail!("unsupported subscription provider");
        }
        w.last_provider = Some(p.clone());
        w.intent = Some(Intent::Enabled { provider: p });
    } else {
        w.intent = Some(Intent::Disabled);
    }
    save_at(&path, &r)
}
/// Read intent by inbound Claude session ID. Corruption deliberately falls back to global policy.
pub fn lookup(session: Option<&str>) -> Option<Intent> {
    let session = session.filter(|s| valid(s))?;
    let path = registry_path().ok()?;
    let _lock = lock_registry(&path).ok()?;
    let mut r = load_at(&path).ok()?;
    let token = r.sessions.get(session)?.clone();
    let window = r.windows.get_mut(&token)?;
    let current = now();
    if current.saturating_sub(window.touched) > TTL.as_secs() {
        r.windows.remove(&token);
        r.sessions.retain(|_, mapped| mapped != &token);
        let _ = save_at(&path, &r);
        return None;
    }
    let intent = window.intent.clone();
    if current.saturating_sub(window.touched) >= TOUCH.as_secs() {
        window.touched = current;
        let _ = save_at(&path, &r);
    }
    intent
}
pub fn status(session: &str) -> Result<Option<Intent>> {
    Ok(lookup(Some(session)))
}

/// Last provider this window successfully enabled, if any (used by bare `/sub on`).
pub fn last_provider(session: &str) -> Option<String> {
    if !valid(session) {
        return None;
    }
    let path = registry_path().ok()?;
    let _lock = lock_registry(&path).ok()?;
    let r = load_at(&path).ok()?;
    let token = r.sessions.get(session)?;
    let window = r.windows.get(token)?;
    let current = now();
    if current.saturating_sub(window.touched) > TTL.as_secs() {
        return None;
    }
    window.last_provider.clone()
}

#[cfg(unix)]
fn quoted_exe(exe: &str) -> String {
    format!("'{}'", exe.replace('\'', "'\"'\"'"))
}

#[cfg(windows)]
fn quoted_exe(exe: &str) -> String {
    format!("\"{}\"", exe.replace('"', "\"\""))
}

#[cfg(not(any(unix, windows)))]
fn quoted_exe(exe: &str) -> String {
    exe.to_string()
}

/// Bash permission rule that auto-allows the skill's inline `!`…`` shell line.
///
/// Claude Code skill expansion requires `behavior: "allow"` and cannot prompt;
/// without this, `/sub` fails with "This command requires approval".
pub fn slash_bash_allow_rule(exe: &str) -> String {
    // Wildcard form (not legacy `:*`) — matches after `$ARGUMENTS` is substituted.
    format!("Bash({} window-sub slash *)", quoted_exe(exe))
}

pub fn command_markdown(exe: &str) -> String {
    let exe_q = quoted_exe(exe);
    let allow = slash_bash_allow_rule(exe);
    format!(
        "---\n\
         description: Toggle llmtrim subscription rerouting for this Claude Code window only.\n\
         disable-model-invocation: true\n\
         argument-hint: \"on [codex|kimi|grok]|off|status\"\n\
         allowed-tools: {allow}\n\
         ---\n\
         \n\
         <!-- llmtrim-owned-window-sub -->\n\
         Window-local subscription override (does not change other windows or the global\n\
         `llmtrim sub` setting).\n\
         \n\
         - `/sub on` — enable using the last provider for this window, or the global `sub`\n\
         - `/sub on codex` / `/sub on kimi` / `/sub on grok` — enable a specific provider\n\
         - `/sub off` — force Anthropic (with compression) for this window\n\
         - `/sub status` — show this window's override\n\
         \n\
         !`{exe_q} window-sub slash \"$ARGUMENTS\"`\n"
    )
}

/// True for Bash allow rules llmtrim itself wrote for `/sub`.
///
/// Shape: `Bash(<llmtrim-path> window-sub slash *)` (or legacy `:*`). The path
/// token must basename to `llmtrim` / `llmtrim.exe` so a hand-written
/// `Bash(echo window-sub slash *)` is never treated as owned.
fn is_owned_slash_allow_rule(rule: &str) -> bool {
    let r = rule.trim();
    let Some(inner) = r.strip_prefix("Bash(").and_then(|s| s.strip_suffix(')')) else {
        return false;
    };
    // Path may be bare, single-quoted, or double-quoted. On Windows the path can
    // contain backslashes (`C:\bin\llmtrim.exe`); treat `\` as a path char, not an
    // escape of the closing quote.
    let (path, tail) = if let Some(rest) = inner.strip_prefix('\'') {
        let Some(end) = rest.find('\'') else {
            return false;
        };
        (&rest[..end], &rest[end + 1..])
    } else if let Some(rest) = inner.strip_prefix('"') {
        let Some(end) = rest.find('"') else {
            return false;
        };
        (&rest[..end], &rest[end + 1..])
    } else {
        let Some(sp) = inner.find(' ') else {
            return false;
        };
        (&inner[..sp], &inner[sp..])
    };
    // `Path::file_name` is OS-separator-sensitive; rules may use the other OS's
    // separators when inspected cross-platform, so split on both.
    let basename = path.rsplit(['/', '\\']).next().unwrap_or(path);
    if basename != "llmtrim" && !basename.eq_ignore_ascii_case("llmtrim.exe") {
        return false;
    }
    matches!(tail, " window-sub slash *" | " window-sub slash:*")
}

fn slash_allow_rule_present(root: &serde_json::Value, exe: &str) -> bool {
    let rule = slash_bash_allow_rule(exe);
    root.pointer("/permissions/allow")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|allow| allow.iter().any(|v| v.as_str() == Some(rule.as_str())))
}

fn ensure_slash_allow_rule(root: &mut serde_json::Value, exe: &str) -> Result<()> {
    let obj = root
        .as_object_mut()
        .context("Claude settings root must be an object")?;
    let permissions = obj
        .entry("permissions")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Claude permissions must be an object")?;
    let allow = permissions
        .entry("allow")
        .or_insert_with(|| serde_json::json!([]))
        .as_array_mut()
        .context("Claude permissions.allow must be an array")?;
    allow.retain(|v| {
        v.as_str()
            .map(|s| !is_owned_slash_allow_rule(s))
            .unwrap_or(true)
    });
    let rule = slash_bash_allow_rule(exe);
    if !allow.iter().any(|v| v.as_str() == Some(rule.as_str())) {
        allow.push(serde_json::Value::String(rule));
    }
    Ok(())
}

fn remove_slash_allow_rule(root: &mut serde_json::Value) {
    if let Some(allow) = root
        .pointer_mut("/permissions/allow")
        .and_then(serde_json::Value::as_array_mut)
    {
        allow.retain(|v| {
            v.as_str()
                .map(|s| !is_owned_slash_allow_rule(s))
                .unwrap_or(true)
        });
    }
}
fn settings_path() -> Result<PathBuf> {
    let h = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|x| PathBuf::from(x).join(".claude"))
        })
        .context("could not determine Claude config directory")?;
    Ok(h.join("settings.json"))
}
const HOOK_MARKER: &str = "# llmtrim-owned-window-sub-hook";

fn owned(v: &serde_json::Value) -> bool {
    v.get("command")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|command| command.trim_end().ends_with(HOOK_MARKER))
}

fn remove_owned_hooks(groups: &mut Vec<serde_json::Value>) {
    for group in groups.iter_mut() {
        if let Some(hooks) = group
            .get_mut("hooks")
            .and_then(serde_json::Value::as_array_mut)
        {
            hooks.retain(|hook| !owned(hook));
        }
    }
    groups.retain(|group| {
        group
            .get("hooks")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|hooks| !hooks.is_empty())
    });
}
/// Ownership of the `/sub` skill + hooks relative to this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnedStatus {
    Missing,
    Stale,
    Current,
}

/// Whether the Claude Code `/sub` skill + owned hooks are present.
pub fn is_installed() -> bool {
    !matches!(owned_status(), OwnedStatus::Missing)
}

/// Missing / present-but-stale-path / current for ensure probe + apply.
pub fn owned_status() -> OwnedStatus {
    let Ok(p) = settings_path() else {
        return OwnedStatus::Missing;
    };
    let Some(skill) = p
        .parent()
        .map(|dir| dir.join("skills").join(COMMAND_NAME).join("SKILL.md"))
    else {
        return OwnedStatus::Missing;
    };
    let skill_text = fs::read_to_string(&skill).unwrap_or_default();
    if !skill_text.contains("llmtrim-owned-window-sub") {
        return OwnedStatus::Missing;
    }
    let Some(root) = fs::read_to_string(&p)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    else {
        return OwnedStatus::Missing;
    };
    let Some(hooks) = root.get("hooks").and_then(serde_json::Value::as_object) else {
        return OwnedStatus::Missing;
    };
    let desired = crate::statusline::stable_exe_string();
    let desired_quoted = quoted_exe(&desired);
    let desired_allow = slash_bash_allow_rule(&desired);
    let mut saw = false;
    let mut current = true;
    for event in ["SessionStart", "SessionEnd"] {
        let Some(groups) = hooks.get(event).and_then(|v| v.as_array()) else {
            return OwnedStatus::Missing;
        };
        let mut event_ok = false;
        for group in groups {
            let Some(hs) = group.get("hooks").and_then(|v| v.as_array()) else {
                continue;
            };
            for h in hs {
                if !owned(h) {
                    continue;
                }
                saw = true;
                event_ok = true;
                let cmd = h.get("command").and_then(|c| c.as_str()).unwrap_or("");
                // Hook is current if it embeds this binary (quoted or raw).
                if !(cmd.contains(&desired_quoted) || cmd.contains(&desired)) {
                    current = false;
                }
            }
        }
        if !event_ok {
            return OwnedStatus::Missing;
        }
    }
    if !saw {
        return OwnedStatus::Missing;
    }
    // Pre-approval for skill bash is required; missing/stale → ensure reinstalls.
    if !slash_allow_rule_present(&root, &desired) {
        current = false;
    }
    // Skill body must match the current template (no shell-expanded session id;
    // allowed-tools frontmatter present for this binary).
    if skill_text.contains("CLAUDE_CODE_SESSION_ID") || !skill_text.contains(&desired_allow) {
        current = false;
    }
    if current {
        OwnedStatus::Current
    } else {
        OwnedStatus::Stale
    }
}

pub fn install(exe: &str) -> Result<()> {
    let hook_exe = quoted_exe(exe);
    let p = settings_path()?;
    let skill = p
        .parent()
        .context("settings has no parent")?
        .join("skills")
        .join(COMMAND_NAME);
    let skill_file = skill.join("SKILL.md");
    if skill_file.exists()
        && !fs::read_to_string(&skill_file)
            .unwrap_or_default()
            .contains("llmtrim-owned-window-sub")
    {
        bail!(
            "{} already exists and is not owned by llmtrim",
            skill_file.display()
        );
    }
    let mut root = if p.exists() {
        serde_json::from_str(&fs::read_to_string(&p)?)?
    } else {
        serde_json::json!({})
    };
    let obj = root
        .as_object_mut()
        .context("Claude settings root must be an object")?;
    let hooks = obj
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Claude hooks must be an object")?;
    for (event, command) in [
        (
            "SessionStart",
            format!("{hook_exe} window-sub hook-start {HOOK_MARKER}"),
        ),
        (
            "SessionEnd",
            format!("{hook_exe} window-sub hook-end {HOOK_MARKER}"),
        ),
    ] {
        let arr = hooks
            .entry(event)
            .or_insert_with(|| serde_json::json!([]))
            .as_array_mut()
            .context("Claude hook event must be an array")?;
        remove_owned_hooks(arr);
        arr.push(serde_json::json!({"hooks":[{"type":"command","command":command,"timeout":10}]}));
    }
    // Skill inline `!`…`` cannot prompt — pre-approve the slash command's Bash pattern.
    ensure_slash_allow_rule(&mut root, exe)?;
    let dir = p.parent().context("settings has no parent")?;
    fs::create_dir_all(dir)?;
    let tmp = p.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(&root)?)?;
    replace_file(&tmp, &p)?;
    fs::create_dir_all(&skill)?;
    fs::write(skill_file, command_markdown(exe))?;
    Ok(())
}
pub fn uninstall() -> Result<()> {
    let p = settings_path()?;
    if p.exists() {
        let mut r: serde_json::Value = serde_json::from_str(&fs::read_to_string(&p)?)?;
        if let Some(h) = r
            .get_mut("hooks")
            .and_then(serde_json::Value::as_object_mut)
        {
            for e in ["SessionStart", "SessionEnd"] {
                if let Some(groups) = h.get_mut(e).and_then(serde_json::Value::as_array_mut) {
                    remove_owned_hooks(groups);
                }
            }
        }
        remove_slash_allow_rule(&mut r);
        fs::write(&p, serde_json::to_vec_pretty(&r)?)?;
    }
    let skill = settings_path()?
        .parent()
        .unwrap()
        .join("skills")
        .join(COMMAND_NAME);
    let skill_file = skill.join("SKILL.md");
    if fs::read_to_string(&skill_file)
        .unwrap_or_default()
        .contains("llmtrim-owned-window-sub")
    {
        fs::remove_file(skill_file)?;
        if fs::read_dir(&skill)?.next().is_none() {
            fs::remove_dir(skill)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal_and_accepts_opaque_ids() {
        assert!(valid("01abc_DEF-9"));
        assert!(!valid("../escape"));
        assert!(!valid("bad/slash"));
    }

    #[test]
    fn registry_prunes_expired_window_and_session() {
        let mut r = Registry::default();
        r.windows.insert(
            "token".into(),
            Window {
                intent: None,
                touched: 1,
                last_provider: None,
            },
        );
        r.sessions.insert("session".into(), "token".into());
        prune(&mut r, TTL.as_secs() + 2);
        assert!(r.windows.is_empty());
        assert!(r.sessions.is_empty());
    }

    #[test]
    fn skill_uses_session_environment_without_exposing_it() {
        let text = command_markdown("llmtrim");
        // Session id must come from the process env (window-sub slash falls back to
        // CLAUDE_CODE_SESSION_ID). Putting `$CLAUDE_CODE_SESSION_ID` on the skill
        // command line is rejected by Claude Code's skill-bash permission check as
        // `Contains simple_expansion` — skills cannot prompt for approval.
        assert!(
            !text.contains("CLAUDE_CODE_SESSION_ID"),
            "skill command must not shell-expand session id: {text}"
        );
        assert!(
            text.contains("window-sub slash \"$ARGUMENTS\""),
            "skill command should only pass $ARGUMENTS: {text}"
        );
        assert!(
            text.contains(&format!(
                "allowed-tools: {}",
                slash_bash_allow_rule("llmtrim")
            )),
            "skill must pre-approve its Bash pattern (skills cannot prompt): {text}"
        );
        assert!(!text.contains("LLMTRIM_CLAUDE_WINDOW_TOKEN"));
        assert!(
            text.contains("on [codex|kimi|grok]"),
            "argument-hint should list providers: {text}"
        );
        assert!(
            text.contains("/sub on grok"),
            "skill body should document explicit provider: {text}"
        );
    }

    #[test]
    fn slash_allow_rule_is_scoped_to_window_sub_slash() {
        let rule = slash_bash_allow_rule("/opt/llmtrim");
        assert_eq!(rule, "Bash('/opt/llmtrim' window-sub slash *)");
        assert!(is_owned_slash_allow_rule(&rule));
        assert!(is_owned_slash_allow_rule(
            "Bash('/old/path/llmtrim' window-sub slash *)"
        ));
        assert!(is_owned_slash_allow_rule(
            "Bash(llmtrim window-sub slash:*)"
        ));
        assert!(is_owned_slash_allow_rule(
            r#"Bash("C:\bin\llmtrim.exe" window-sub slash *)"#
        ));
        // Not ours: wrong basename, wrong subcommand, or bare tool allow.
        assert!(!is_owned_slash_allow_rule("Bash(echo window-sub slash *)"));
        assert!(!is_owned_slash_allow_rule(
            "Bash('/opt/other' window-sub slash *)"
        ));
        assert!(!is_owned_slash_allow_rule("Bash(llmtrim *)"));
        assert!(!is_owned_slash_allow_rule("Bash(git:*)"));
        assert!(!is_owned_slash_allow_rule(
            "Bash('/opt/llmtrim' window-sub hook-start *)"
        ));
    }

    #[test]
    fn ensure_slash_allow_rule_replaces_stale_owned_entry() {
        let mut root = serde_json::json!({
            "permissions": {
                "allow": [
                    "Bash(git:*)",
                    "Bash(echo window-sub slash *)",
                    "Bash('/old/llmtrim' window-sub slash *)"
                ]
            }
        });
        ensure_slash_allow_rule(&mut root, "/new/llmtrim").unwrap();
        let allow = root["permissions"]["allow"].as_array().unwrap();
        let rules: Vec<&str> = allow.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(
            rules,
            vec![
                "Bash(git:*)",
                "Bash(echo window-sub slash *)",
                "Bash('/new/llmtrim' window-sub slash *)"
            ]
        );
        assert!(slash_allow_rule_present(&root, "/new/llmtrim"));
        assert!(!slash_allow_rule_present(&root, "/old/llmtrim"));
        remove_slash_allow_rule(&mut root);
        let rules: Vec<&str> = root["permissions"]["allow"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(rules, vec!["Bash(git:*)", "Bash(echo window-sub slash *)"]);
    }

    #[test]
    fn slash_allow_rule_present_requires_exact_current_rule() {
        let root = serde_json::json!({
            "permissions": {
                "allow": ["Bash('/opt/llmtrim' window-sub slash *)"]
            }
        });
        assert!(slash_allow_rule_present(&root, "/opt/llmtrim"));
        assert!(!slash_allow_rule_present(&root, "/other/llmtrim"));
        assert!(!slash_allow_rule_present(
            &serde_json::json!({}),
            "/opt/llmtrim"
        ));
    }

    #[test]
    fn accepts_all_subscription_providers() {
        assert!(valid_provider("codex"));
        assert!(valid_provider("kimi"));
        assert!(valid_provider("grok"));
        assert!(!valid_provider("anthropic"));
        assert!(!valid_provider("openai"));
    }

    #[test]
    fn removing_owned_hook_preserves_neighbors_in_the_same_group() {
        let mut groups = vec![serde_json::json!({
            "hooks": [
                {"type": "command", "command": "user-hook"},
                {"type": "command", "command": format!("llmtrim window-sub hook-start {HOOK_MARKER}")}
            ]
        })];
        remove_owned_hooks(&mut groups);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0]["hooks"].as_array().unwrap().len(), 1);
        assert_eq!(groups[0]["hooks"][0]["command"], "user-hook");
    }

    #[cfg(unix)]
    #[test]
    fn executable_path_is_single_quote_escaped() {
        let command = command_markdown("/tmp/a'$(touch nope)/llmtrim");
        assert!(command.contains("'/tmp/a'\"'\"'$(touch nope)/llmtrim'"));
    }
}
