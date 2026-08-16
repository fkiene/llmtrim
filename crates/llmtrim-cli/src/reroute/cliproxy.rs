//! Managed [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) sidecar.
//!
//! `sub on` installs and starts this process. The interceptor then rewrites Anthropic
//! `/v1/messages` to the sidecar (which already speaks the Claude API) instead of doing
//! first-party Codex/Kimi/Grok protocol translation.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::SubProvider;
use super::UpstreamRewrite;

pub const REPO: &str = "router-for-me/CLIProxyAPI";
pub const DEFAULT_PORT: u16 = 18317;
const DEFAULT_HOST: &str = "127.0.0.1";

/// Override the sidecar base URL (`http://127.0.0.1:8317`). When set, llmtrim does not
/// manage a private binary — it just redirects to this instance.
pub const URL_ENV: &str = "LLMTRIM_CLIPROXY_URL";
/// API key for [`URL_ENV`] (or for the managed sidecar when you want to pin one).
pub const KEY_ENV: &str = "LLMTRIM_CLIPROXY_KEY";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub id: String,
    pub owned_by: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub enabled: bool,
    pub installed: bool,
    pub running: bool,
    pub managed: bool,
    pub version: Option<String>,
    pub base_url: String,
}

/// One CLIProxyAPI login / model family. Names match the sidecar's OAuth CLIs so
/// `sub on gemini` / `/sub on antigravity` / tab 4 can address the same backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backend {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub owned_by: &'static [&'static str],
    pub model_prefixes: &'static [&'static str],
}

pub const BACKENDS: &[Backend] = &[
    Backend {
        id: "codex",
        aliases: &["codex", "chatgpt", "openai"],
        owned_by: &["openai", "codex"],
        model_prefixes: &["gpt-"],
    },
    Backend {
        id: "claude",
        aliases: &["claude", "anthropic"],
        owned_by: &["anthropic", "claude"],
        model_prefixes: &["claude-"],
    },
    Backend {
        id: "gemini",
        aliases: &["gemini", "antigravity", "aistudio"],
        owned_by: &["google", "gemini", "antigravity"],
        model_prefixes: &["gemini-"],
    },
    Backend {
        id: "grok",
        aliases: &["grok", "xai", "x-ai"],
        owned_by: &["xai", "x-ai", "grok"],
        model_prefixes: &["grok-"],
    },
    Backend {
        id: "kimi",
        aliases: &["kimi", "moonshot"],
        owned_by: &["kimi", "moonshot", "moonshotai"],
        model_prefixes: &["kimi"],
    },
    Backend {
        id: "vertex",
        aliases: &["vertex"],
        owned_by: &["vertex"],
        model_prefixes: &[],
    },
    Backend {
        id: "qwen",
        aliases: &["qwen"],
        owned_by: &["qwen"],
        model_prefixes: &["qwen"],
    },
    Backend {
        id: "copilot",
        aliases: &["copilot", "github"],
        owned_by: &["github", "copilot"],
        model_prefixes: &[],
    },
];

/// Enable sidecar with no model pin (`sub on` / `/sub on`).
const PASSTHROUGH: &[&str] = &[
    "on",
    "cliproxy",
    "cli-proxy",
    "cli-proxy-api",
    "cliproxyapi",
];

/// A fallback-chain hop: real Anthropic, or a CLIProxyAPI CLI family.
pub fn parse_hop(raw: &str) -> Option<String> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() || s == "off" {
        return None;
    }
    if s == "anthropic" || s == "direct" {
        return Some("anthropic".into());
    }
    if is_passthrough_label(&s) {
        return Some("on".into());
    }
    backend_by_alias(&s).map(|b| b.id.to_string())
}

pub fn is_anthropic_hop(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "anthropic" | "direct"
    )
}

pub fn is_passthrough_label(raw: &str) -> bool {
    let s = raw.trim().to_ascii_lowercase();
    PASSTHROUGH.contains(&s.as_str())
}

pub fn backend_by_alias(raw: &str) -> Option<&'static Backend> {
    let s = raw.trim().to_ascii_lowercase();
    BACKENDS
        .iter()
        .find(|b| b.id == s || b.aliases.contains(&s.as_str()))
}

impl Backend {
    pub fn matches(&self, model: &Model) -> bool {
        let owner = model.owned_by.to_ascii_lowercase();
        if self.owned_by.iter().any(|o| owner.contains(o)) {
            return true;
        }
        let id = model.id.to_ascii_lowercase();
        self.model_prefixes
            .iter()
            .any(|p| !p.is_empty() && id.starts_with(p))
    }
}

/// What `sub on X` / `/sub on X` means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinRequest {
    /// Just turn the sidecar on; leave Claude model ids alone.
    Enable,
    /// Force this CLIProxyAPI model id (or a backend alias to expand later).
    Pin(String),
}

pub fn parse_pin_request(raw: &str) -> Option<PinRequest> {
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("off") {
        return None;
    }
    if is_passthrough_label(raw) {
        return Some(PinRequest::Enable);
    }
    if backend_by_alias(raw).is_some() {
        return Some(PinRequest::Pin(raw.trim().to_ascii_lowercase()));
    }
    let ok_chars = raw
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/'));
    if raw.len() <= 160
        && ok_chars
        && !raw.contains("..")
        && (raw.contains('-') || raw.contains('/') || raw.contains('.'))
    {
        return Some(PinRequest::Pin(raw.to_string()));
    }
    None
}

/// Turn a stored pin (model id or backend alias) into a wire model id when the sidecar
/// has published models. Backend aliases with no matching model stay unresolved so the
/// Claude id is left alone.
pub fn expand_pin(pin: &str, models: &[Model]) -> Option<String> {
    if let Some(exact) = models.iter().find(|m| m.id.eq_ignore_ascii_case(pin)) {
        return Some(exact.id.clone());
    }
    let backend = backend_by_alias(pin)?;
    models
        .iter()
        .find(|m| backend.matches(m))
        .map(|m| m.id.clone())
}

pub fn resolve_pin_live(pin: &str) -> Option<String> {
    let models = list_models().unwrap_or_default();
    expand_pin(pin, &models).or_else(|| {
        if backend_by_alias(pin).is_some() {
            None
        } else {
            Some(pin.to_string())
        }
    })
}

const OFFICIAL_MODELS_URL: &str = "https://models.router-for.me/models.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfficialModel {
    pub id: String,
    pub owned_by: String,
    pub display_name: String,
    pub family: String,
}

impl OfficialModel {
    pub fn as_live(&self) -> Model {
        Model {
            id: self.id.clone(),
            owned_by: self.owned_by.clone(),
        }
    }
}

fn official_cache_path() -> Result<PathBuf> {
    Ok(dir()?.join("official-models.json"))
}

pub fn parse_official_models(raw: &Value) -> Vec<OfficialModel> {
    let Some(obj) = raw.as_object() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (family, arr) in obj {
        let Some(list) = arr.as_array() else {
            continue;
        };
        for item in list {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            if id.is_empty() || !seen.insert(id.to_string()) {
                continue;
            }
            out.push(OfficialModel {
                id: id.to_string(),
                owned_by: item
                    .get("owned_by")
                    .and_then(Value::as_str)
                    .unwrap_or(family)
                    .to_string(),
                display_name: item
                    .get("display_name")
                    .and_then(Value::as_str)
                    .unwrap_or(id)
                    .to_string(),
                family: family.clone(),
            });
        }
    }
    out
}

pub fn search_official(catalog: &[OfficialModel], query: &str) -> Vec<OfficialModel> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return catalog.to_vec();
    }
    catalog
        .iter()
        .filter(|m| {
            m.id.to_ascii_lowercase().contains(&q)
                || m.display_name.to_ascii_lowercase().contains(&q)
                || m.family.to_ascii_lowercase().contains(&q)
                || m.owned_by.to_ascii_lowercase().contains(&q)
        })
        .cloned()
        .collect()
}

fn skip_aux_model(id: &str) -> bool {
    let i = id.to_ascii_lowercase();
    i.contains("imagine")
        || i.contains("video")
        || i.contains("image")
        || i.contains("tts")
        || i.contains("embed")
}

pub fn default_tier_map(
    backend: Option<&Backend>,
    catalog: &[OfficialModel],
) -> std::collections::BTreeMap<String, String> {
    let pool: Vec<&OfficialModel> = catalog
        .iter()
        .filter(|m| !skip_aux_model(&m.id))
        .filter(|m| backend.is_none_or(|b| b.matches(&m.as_live())))
        .collect();
    if pool.is_empty() {
        return std::collections::BTreeMap::new();
    }
    let is_small = |m: &OfficialModel| {
        let i = m.id.to_ascii_lowercase();
        i.contains("fast")
            || i.contains("mini")
            || i.contains("flash")
            || i.contains("lite")
            || i.contains("haiku")
            || i.contains("composer")
    };
    let is_flagship = |m: &OfficialModel| {
        let i = m.id.to_ascii_lowercase();
        i.contains("opus") || i.contains("pro") || i.contains("terra") || i.contains("4.6")
    };
    let haiku = pool
        .iter()
        .copied()
        .find(|m| is_small(m))
        .unwrap_or(pool[pool.len() - 1]);
    let opus = pool
        .iter()
        .copied()
        .find(|m| is_flagship(m) && !is_small(m))
        .or_else(|| pool.iter().copied().find(|m| !is_small(m)))
        .unwrap_or(pool[0]);
    let sonnet = pool
        .iter()
        .copied()
        .find(|m| !is_small(m) && m.id != opus.id && !m.id.to_ascii_lowercase().contains("opus"))
        .unwrap_or(opus);
    let mut map = std::collections::BTreeMap::new();
    map.insert("opus".into(), opus.id.clone());
    map.insert("fable".into(), opus.id.clone());
    map.insert("sonnet".into(), sonnet.id.clone());
    map.insert("haiku".into(), haiku.id.clone());
    map
}

pub fn official_models() -> Vec<OfficialModel> {
    if let Some(cached) = read_official_cache() {
        return cached;
    }
    match fetch_official_models() {
        Ok(list) if !list.is_empty() => list,
        _ => {
            let live = list_models()
                .unwrap_or_default()
                .into_iter()
                .map(|m| OfficialModel {
                    id: m.id,
                    owned_by: m.owned_by,
                    display_name: String::new(),
                    family: String::new(),
                })
                .collect::<Vec<_>>();
            if live.is_empty() {
                fallback_official_models()
            } else {
                live
            }
        }
    }
}

/// Offline / failed-catalog picker so tab 4 is not empty after a 0.13 upgrade.
fn fallback_official_models() -> Vec<OfficialModel> {
    const ROWS: &[(&str, &str, &str, &str)] = &[
        ("grok-4.6", "xai", "Grok 4.6", "grok"),
        (
            "grok-composer-2.5-fast",
            "xai",
            "Grok Composer 2.5 Fast",
            "grok",
        ),
        ("gpt-5.6-terra", "openai", "GPT-5.6 Terra", "codex"),
        ("gpt-5.6-luna", "openai", "GPT-5.6 Luna", "codex"),
        ("gpt-5.4-mini", "openai", "GPT-5.4 Mini", "codex"),
        ("kimi-k2.5", "moonshot", "Kimi K2.5", "kimi"),
    ];
    ROWS.iter()
        .map(|(id, owned_by, display_name, family)| OfficialModel {
            id: (*id).into(),
            owned_by: (*owned_by).into(),
            display_name: (*display_name).into(),
            family: (*family).into(),
        })
        .collect()
}

fn read_official_cache() -> Option<Vec<OfficialModel>> {
    let path = official_cache_path().ok()?;
    let raw = fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&raw).ok()?;
    let at = v.get("fetched_at").and_then(Value::as_u64).unwrap_or(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if now.saturating_sub(at) > 86_400 {
        return None;
    }
    let models = parse_official_models(v.get("models").unwrap_or(&v));
    (!models.is_empty()).then_some(models)
}

pub fn fetch_official_models() -> Result<Vec<OfficialModel>> {
    let mut req = ureq::get(OFFICIAL_MODELS_URL)
        .config()
        .timeout_global(Some(Duration::from_secs(8)))
        .http_status_as_error(true)
        .build();
    req = req.header("User-Agent", "llmtrim-cliproxy");
    let body = req
        .call()
        .context("official CLIProxyAPI models")?
        .body_mut()
        .read_to_string()
        .context("read official models")?;
    let v: Value = serde_json::from_str(&body).context("parse official models")?;
    let list = parse_official_models(&v);
    if let Ok(path) = official_cache_path() {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = fs::write(
            path,
            serde_json::json!({ "fetched_at": now, "models": v }).to_string(),
        );
    }
    Ok(list)
}

pub fn dir() -> Result<PathBuf> {
    Ok(crate::daemon::home_dir()?.join("cliproxy"))
}

pub fn bin_path() -> Result<PathBuf> {
    let name = if cfg!(windows) {
        "CLIProxyAPI.exe"
    } else {
        "CLIProxyAPI"
    };
    Ok(dir()?.join(name))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(dir()?.join("config.yaml"))
}

pub fn version_path() -> Result<PathBuf> {
    Ok(dir()?.join("version"))
}

fn pidfile() -> Result<PathBuf> {
    Ok(crate::daemon::home_dir()?.join("cliproxy.pid"))
}

fn logfile() -> Result<PathBuf> {
    Ok(crate::daemon::home_dir()?.join("cliproxy.log"))
}

fn key_path() -> Result<PathBuf> {
    Ok(dir()?.join("api-key"))
}

pub fn is_installed() -> bool {
    bin_path().ok().is_some_and(|p| p.is_file())
}

pub fn is_managed_user() -> bool {
    is_installed() || is_enabled()
}

pub fn is_enabled() -> bool {
    llmtrim_core::config::sub_always_on()
}

pub fn installed_version() -> Option<String> {
    fs::read_to_string(version_path().ok()?)
        .ok()
        .map(|s| s.trim().trim_start_matches('v').trim().to_string())
}

/// Release asset name for this OS/arch (`None` if we do not ship that target).
pub fn release_asset(version: &str) -> Option<String> {
    release_asset_for(version, std::env::consts::OS, std::env::consts::ARCH)
}

pub fn release_asset_for(version: &str, os: &str, arch: &str) -> Option<String> {
    let ver = version.trim_start_matches('v');
    let (os, arch, ext) = match (os, arch) {
        ("linux", "x86_64") => ("linux", "amd64", "tar.gz"),
        ("linux", "aarch64") => ("linux", "aarch64", "tar.gz"),
        ("macos", "x86_64") => ("darwin", "amd64", "tar.gz"),
        ("macos", "aarch64") => ("darwin", "aarch64", "tar.gz"),
        ("windows", "x86_64") => ("windows", "amd64", "zip"),
        ("windows", "aarch64") => ("windows", "aarch64", "zip"),
        _ => return None,
    };
    Some(format!("CLIProxyAPI_{ver}_{os}_{arch}.{ext}"))
}

pub fn default_base_url() -> String {
    format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}")
}

pub fn base_url() -> String {
    std::env::var(URL_ENV)
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(default_base_url)
}

pub fn is_externally_configured() -> bool {
    std::env::var(URL_ENV)
        .ok()
        .is_some_and(|s| !s.trim().is_empty())
}

pub fn api_key() -> Result<String> {
    if let Ok(k) = std::env::var(KEY_ENV) {
        let k = k.trim().to_string();
        if !k.is_empty() {
            return Ok(k);
        }
    }
    let path = key_path()?;
    if let Ok(existing) = fs::read_to_string(&path) {
        let k = existing.trim().to_string();
        if !k.is_empty() {
            return Ok(k);
        }
    }
    let key = format!("llmtrim-{}", uuid::Uuid::new_v4().simple());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, &key)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    }
    Ok(key)
}

pub fn auth_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let shared = PathBuf::from(home).join(".cli-proxy-api");
        if shared.is_dir() {
            return shared;
        }
    }
    dir()
        .map(|d| d.join("auth"))
        .unwrap_or_else(|_| PathBuf::from("auth"))
}

pub fn config_yaml(port: u16, key: &str, auth: &Path) -> String {
    format!(
        "host: \"{DEFAULT_HOST}\"\n\
         port: {port}\n\
         auth-dir: \"{}\"\n\
         api-keys:\n\
           - \"{key}\"\n\
         remote-management:\n\
           allow-remote: false\n\
           secret-key: \"\"\n\
           disable-control-panel: true\n\
         debug: false\n",
        auth.display()
    )
}

pub fn ensure_config() -> Result<()> {
    if is_externally_configured() {
        return Ok(());
    }
    let dir = dir()?;
    fs::create_dir_all(dir.join("auth")).with_context(|| format!("create {}", dir.display()))?;
    let path = config_path()?;
    if path.is_file() {
        return Ok(());
    }
    let yaml = config_yaml(DEFAULT_PORT, &api_key()?, &auth_dir());
    fs::write(&path, yaml).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn pid_running() -> Option<u32> {
    let raw = fs::read_to_string(pidfile().ok()?).ok()?;
    let pid: u32 = raw.trim().parse().ok()?;
    if process_alive(pid) {
        Some(pid)
    } else {
        let _ = fs::remove_file(pidfile().ok()?);
        None
    }
}

fn process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .ok()
            .is_some_and(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

pub fn is_healthy() -> bool {
    probe_models().is_ok()
}

pub fn is_running() -> bool {
    is_healthy() || pid_running().is_some()
}

fn ureq_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .http_status_as_error(false)
        .build()
        .into()
}

fn models_url() -> String {
    format!("{}/v1/models", base_url())
}

fn probe_models() -> Result<Value> {
    let key = api_key().unwrap_or_default();
    let mut req = ureq_agent().get(models_url());
    if !key.is_empty() {
        req = req
            .header("Authorization", format!("Bearer {key}"))
            .header("x-api-key", &key);
    }
    let mut res = req.call().context("CLIProxyAPI /v1/models")?;
    let status = res.status();
    let body = res.body_mut().read_to_string().unwrap_or_default();
    if !status.is_success() {
        bail!("CLIProxyAPI /v1/models returned {status}: {body}");
    }
    serde_json::from_str(&body).context("parse CLIProxyAPI /v1/models")
}

pub fn parse_models(value: &Value) -> Vec<Model> {
    let Some(arr) = value.get("data").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        if id.is_empty() {
            continue;
        }
        out.push(Model {
            id: id.to_string(),
            owned_by: item
                .get("owned_by")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out.dedup_by(|a, b| a.id == b.id);
    out
}

pub fn list_models() -> Result<Vec<Model>> {
    Ok(parse_models(&probe_models()?))
}

pub fn status() -> Status {
    Status {
        enabled: is_enabled(),
        installed: is_installed(),
        running: is_running(),
        managed: !is_externally_configured(),
        version: installed_version(),
        base_url: base_url(),
    }
}

/// Build the MITM rewrite that sends the (already Anthropic-shaped) body to CLIProxyAPI.
pub fn rewrite(anthropic_body: &Value) -> Result<UpstreamRewrite> {
    if !is_running() {
        bail!("CLIProxyAPI is not running — run `llmtrim sub on`");
    }
    let model = anthropic_body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let mut body = anthropic_body.clone();
    strip_tiny_images(&mut body);
    let key = api_key()?;
    let url = base_url();
    let (host, path) = split_base_url(&url)?;
    Ok(UpstreamRewrite {
        host,
        path: format!("{path}/v1/messages"),
        headers: vec![
            ("authorization".into(), format!("Bearer {key}")),
            ("x-api-key".into(), key),
            ("content-type".into(), "application/json".into()),
        ],
        body: serde_json::to_vec(&body)?,
        model,
        provider: SubProvider::CliProxy,
    })
}

/// Grok (and some others) reject images under 8×8. Drop those blocks so a 2×2
/// placeholder from Claude Code does not 400 the whole turn.
const MIN_IMAGE_EDGE: u32 = 8;

pub(crate) fn strip_tiny_images(body: &mut Value) -> usize {
    let mut n = 0;
    if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for msg in msgs {
            n += strip_tiny_in_content(msg.get_mut("content"));
        }
    }
    n
}

fn strip_tiny_in_content(content: Option<&mut Value>) -> usize {
    let Some(content) = content else {
        return 0;
    };
    match content {
        Value::Array(blocks) => {
            let mut n = 0;
            let mut i = 0;
            while i < blocks.len() {
                n += strip_tiny_in_content(blocks[i].get_mut("content"));
                if is_tiny_image(&blocks[i]) {
                    blocks[i] = serde_json::json!({
                        "type": "text",
                        "text": "[image omitted: smaller than 8×8]"
                    });
                    n += 1;
                }
                i += 1;
            }
            n
        }
        _ => 0,
    }
}

fn is_tiny_image(block: &Value) -> bool {
    if block.get("type").and_then(Value::as_str) != Some("image") {
        return false;
    }
    let Some(source) = block.get("source") else {
        return false;
    };
    if source.get("type").and_then(Value::as_str) != Some("base64") {
        return false;
    }
    let Some(data) = source.get("data").and_then(Value::as_str) else {
        return false;
    };
    image_edges(data).is_some_and(|(w, h)| w < MIN_IMAGE_EDGE || h < MIN_IMAGE_EDGE)
}

fn image_edges(b64: &str) -> Option<(u32, u32)> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(b64.trim().replace('\n', "")))
        .ok()?;
    png_edges(&raw)
        .or_else(|| jpeg_edges(&raw))
        .or_else(|| gif_edges(&raw))
}

fn png_edges(raw: &[u8]) -> Option<(u32, u32)> {
    if raw.len() < 24 || &raw[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    if &raw[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(raw[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(raw[20..24].try_into().ok()?);
    Some((w, h))
}

fn jpeg_edges(raw: &[u8]) -> Option<(u32, u32)> {
    if raw.len() < 4 || raw[0] != 0xFF || raw[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 9 < raw.len() {
        if raw[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = raw[i + 1];
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        if i + 4 > raw.len() {
            break;
        }
        let len = u16::from_be_bytes([raw[i + 2], raw[i + 3]]) as usize;
        if (0xC0..=0xC2).contains(&marker) && i + 9 < raw.len() {
            let h = u16::from_be_bytes([raw[i + 5], raw[i + 6]]) as u32;
            let w = u16::from_be_bytes([raw[i + 7], raw[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

fn gif_edges(raw: &[u8]) -> Option<(u32, u32)> {
    if raw.len() < 10 || (&raw[0..6] != b"GIF87a" && &raw[0..6] != b"GIF89a") {
        return None;
    }
    let w = u16::from_le_bytes([raw[6], raw[7]]) as u32;
    let h = u16::from_le_bytes([raw[8], raw[9]]) as u32;
    Some((w, h))
}

pub fn split_base_url(url: &str) -> Result<(String, String)> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .context("CLIProxyAPI URL must be http(s)://host[:port]")?;
    let (host, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    if host.is_empty() {
        bail!("CLIProxyAPI URL has no host");
    }
    let prefix = prefix.trim_end_matches('/');
    let prefix = if prefix.is_empty() {
        String::new()
    } else {
        format!("/{prefix}")
    };
    Ok((host.to_string(), prefix))
}

pub fn fetch_latest_tag() -> Result<String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let mut req = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(8)))
        .http_status_as_error(false)
        .build();
    req = req.header("User-Agent", "llmtrim-cliproxy");
    let body = req
        .call()
        .context("CLIProxyAPI releases")?
        .body_mut()
        .read_to_string()
        .context("read CLIProxyAPI release")?;
    let v: Value = serde_json::from_str(&body).context("parse CLIProxyAPI release")?;
    let tag = v
        .get("tag_name")
        .and_then(Value::as_str)
        .context("CLIProxyAPI release has no tag_name")?;
    Ok(tag.trim_start_matches('v').to_string())
}

pub fn ensure_installed() -> Result<String> {
    if is_externally_configured() {
        return Ok("external".into());
    }
    if is_installed() {
        return Ok(installed_version().unwrap_or_else(|| "unknown".into()));
    }
    install_latest()
}

pub fn install_latest() -> Result<String> {
    if is_externally_configured() {
        bail!("LLMTRIM_CLIPROXY_URL is set — I will not replace an external CLIProxyAPI");
    }
    let tag = fetch_latest_tag()?;
    install_tag(&tag)?;
    Ok(tag)
}

pub fn install_tag(tag: &str) -> Result<()> {
    let asset = release_asset(tag).with_context(|| {
        format!(
            "no CLIProxyAPI build for {}/{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let url = format!("https://github.com/{REPO}/releases/download/v{tag}/{asset}");
    let dest = dir()?;
    fs::create_dir_all(&dest)?;
    let archive = dest.join(&asset);
    download(&url, &archive)?;
    extract(&archive, &dest)?;
    let _ = fs::remove_file(&archive);
    locate_binary(&dest)?;
    fs::write(version_path()?, tag)?;
    ensure_config()?;
    Ok(())
}

/// Bound so a runaway response cannot fill the disk. Must be above current
/// CLIProxyAPI archives (~20 MiB); ureq `read_to_vec` defaults to 10 MiB and
/// rejects them (`the response body is larger than request limit: 10485760`).
const MAX_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024;

fn download(url: &str, dest: &Path) -> Result<()> {
    let mut req = ureq::get(url)
        .config()
        .timeout_global(Some(Duration::from_secs(120)))
        .http_status_as_error(true)
        .build();
    req = req.header("User-Agent", "llmtrim-cliproxy");
    let mut res = req.call().with_context(|| format!("download {url}"))?;
    use std::io::Read;
    // `as_reader()` is unlimited by default; `read_to_vec()` is not (10 MiB).
    let mut reader = res.body_mut().as_reader();
    let mut file = fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    let mut limited = Read::take(&mut reader, MAX_ARCHIVE_BYTES);
    let n = std::io::copy(&mut limited, &mut file).context("read CLIProxyAPI archive")?;
    if n >= MAX_ARCHIVE_BYTES {
        bail!("CLIProxyAPI archive exceeded 64 MiB size limit");
    }
    Ok(())
}

fn extract(archive: &Path, dest: &Path) -> Result<()> {
    let name = archive.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name.ends_with(".zip") {
        let status = if cfg!(windows) {
            std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                        archive.display(),
                        dest.display()
                    ),
                ])
                .status()
        } else {
            std::process::Command::new("unzip")
                .args([
                    "-o",
                    &archive.to_string_lossy(),
                    "-d",
                    &dest.to_string_lossy(),
                ])
                .status()
        }
        .context("extract CLIProxyAPI zip")?;
        if !status.success() {
            bail!("failed to extract {}", archive.display());
        }
        return Ok(());
    }
    let status = std::process::Command::new("tar")
        .args([
            "-xzf",
            &archive.to_string_lossy(),
            "-C",
            &dest.to_string_lossy(),
        ])
        .status()
        .context("extract CLIProxyAPI tar.gz (tar required)")?;
    if !status.success() {
        bail!("failed to extract {}", archive.display());
    }
    Ok(())
}

fn locate_binary(dest: &Path) -> Result<PathBuf> {
    let expected = bin_path()?;
    if expected.is_file() {
        chmod_exec(&expected);
        return Ok(expected);
    }
    for entry in walkdir_bins(dest) {
        let Some(name) = entry.file_name() else {
            continue;
        };
        let name = name.to_string_lossy();
        if name == "CLIProxyAPI" || name == "CLIProxyAPI.exe" || name == "cli-proxy-api" {
            if entry != expected {
                let _ = fs::copy(&entry, &expected);
            }
            chmod_exec(&expected);
            return Ok(expected);
        }
    }
    bail!("CLIProxyAPI binary not found in {}", dest.display());
}

fn walkdir_bins(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(root) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walkdir_bins(&path));
        } else {
            out.push(path);
        }
    }
    out
}

fn chmod_exec(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut mode = meta.permissions().mode();
            mode |= 0o755;
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
        }
    }
    let _ = path;
}

pub fn ensure_running() -> Result<()> {
    if is_externally_configured() {
        if is_healthy() {
            return Ok(());
        }
        bail!(
            "CLIProxyAPI at {} is not reachable — start it, or unset {URL_ENV}",
            base_url()
        );
    }
    ensure_installed()?;
    ensure_config()?;
    if is_healthy() {
        return Ok(());
    }
    if pid_running().is_some() {
        // Process up but not healthy yet — give it a moment.
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(150));
            if is_healthy() {
                return Ok(());
            }
        }
    }
    start()?;
    for _ in 0..40 {
        if is_healthy() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    bail!(
        "CLIProxyAPI started but {}/v1/models is not answering — see {}",
        base_url(),
        logfile()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "cliproxy.log".into())
    );
}

pub fn start() -> Result<u32> {
    if is_externally_configured() {
        bail!("LLMTRIM_CLIPROXY_URL is set — start that instance yourself");
    }
    if let Some(pid) = pid_running() {
        return Ok(pid);
    }
    ensure_installed()?;
    ensure_config()?;
    let bin = bin_path()?;
    let cfg = config_path()?;
    let log = fs::File::create(logfile()?)?;
    let err = log.try_clone()?;
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(["--config", &cfg.to_string_lossy()])
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(err);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }
    let child = cmd
        .spawn()
        .with_context(|| format!("spawn {}", bin.display()))?;
    let pid = child.id();
    fs::write(pidfile()?, pid.to_string())?;
    Ok(pid)
}

pub fn stop() -> Result<Option<u32>> {
    if is_externally_configured() {
        return Ok(None);
    }
    let Some(pid) = pid_running() else {
        return Ok(None);
    };
    #[cfg(unix)]
    {
        let _ = std::process::Command::new("kill")
            .args([pid.to_string()])
            .status();
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
    }
    for _ in 0..30 {
        if !process_alive(pid) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = fs::remove_file(pidfile()?);
    Ok(Some(pid))
}

/// Launch the CLIProxyAPI TUI so the user can sign in to providers.
pub fn auth_tui() -> Result<()> {
    ensure_installed()?;
    ensure_config()?;
    let bin = bin_path()?;
    let cfg = config_path()?;
    let status = std::process::Command::new(&bin)
        .args(["--tui", "--config", &cfg.to_string_lossy()])
        .status()
        .with_context(|| format!("run {} --tui", bin.display()))?;
    if !status.success() {
        bail!("CLIProxyAPI TUI exited non-zero");
    }
    Ok(())
}

pub fn update_if_used() -> Result<Option<String>> {
    if !is_managed_user() || is_externally_configured() {
        return Ok(None);
    }
    if !is_installed() {
        if !is_enabled() {
            return Ok(None);
        }
        return ensure_for_existing_user().map(Some);
    }
    let latest = fetch_latest_tag()?;
    if installed_version().as_deref() == Some(latest.as_str()) {
        let imported = migrate_legacy_tokens()?;
        if is_enabled() {
            let _ = ensure_running();
        }
        if imported.is_empty() {
            return Ok(Some(format!("CLIProxyAPI already {latest}")));
        }
        return Ok(Some(format!(
            "CLIProxyAPI already {latest}; imported {}",
            imported.join(", ")
        )));
    }
    let was_running = pid_running().is_some() || is_enabled();
    if pid_running().is_some() {
        let _ = stop();
    }
    install_tag(&latest)?;
    let imported = migrate_legacy_tokens()?;
    if was_running {
        let _ = ensure_running();
    }
    let mut msg = format!("CLIProxyAPI updated to {latest}");
    if !imported.is_empty() {
        msg.push_str(&format!("; imported {}", imported.join(", ")));
    }
    Ok(Some(msg))
}

/// Install, import existing `sub` tokens, and start the sidecar. Used by `ensure` / `update`
/// so a user who already had `sub = codex|kimi|grok` needs no extra command.
pub fn ensure_for_existing_user() -> Result<String> {
    if is_externally_configured() {
        if is_healthy() {
            return Ok(format!("CLIProxyAPI at {} reachable", base_url()));
        }
        bail!("CLIProxyAPI at {} is not reachable", base_url());
    }
    ensure_installed()?;
    ensure_config()?;
    let imported = migrate_legacy_tokens()?;
    ensure_running()?;
    if imported.is_empty() {
        Ok(format!("CLIProxyAPI ready at {}", base_url()))
    } else {
        Ok(format!(
            "CLIProxyAPI ready at {}; imported {}",
            base_url(),
            imported.join(", ")
        ))
    }
}

/// Copy first-party `~/.llmtrim/{{codex,kimi,grok}}/auth.json` into the CLIProxyAPI auth dir
/// once. Existing sidecar files are left alone.
pub fn migrate_legacy_tokens() -> Result<Vec<String>> {
    let dest = auth_dir();
    fs::create_dir_all(&dest)?;
    let home = crate::daemon::home_dir()?;
    let mut imported = Vec::new();
    if import_one(
        &home.join("codex").join("auth.json"),
        &dest.join("codex-llmtrim.json"),
        convert_codex_auth,
    )? {
        imported.push("codex".into());
    }
    if import_one(
        &home.join("kimi").join("auth.json"),
        &dest.join("kimi-llmtrim.json"),
        convert_kimi_auth,
    )? {
        imported.push("kimi".into());
    }
    if import_one(
        &home.join("grok").join("auth.json"),
        &dest.join("xai-llmtrim.json"),
        convert_grok_auth,
    )? {
        imported.push("grok".into());
    }
    Ok(imported)
}

fn import_one(src: &Path, dest: &Path, convert: fn(&Value) -> Option<Value>) -> Result<bool> {
    if dest.is_file() || !src.is_file() {
        return Ok(false);
    }
    let raw = fs::read_to_string(src).with_context(|| format!("read {}", src.display()))?;
    let src_val: Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", src.display()))?;
    let Some(out) = convert(&src_val) else {
        return Ok(false);
    };
    let bytes = serde_json::to_vec_pretty(&out)?;
    fs::write(dest, bytes).with_context(|| format!("write {}", dest.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dest, fs::Permissions::from_mode(0o600));
    }
    Ok(true)
}

fn epoch_ms_to_rfc3339(ms: u64) -> String {
    use chrono::{TimeZone, Utc};
    Utc.timestamp_millis_opt(ms as i64)
        .single()
        .unwrap_or_else(Utc::now)
        .to_rfc3339()
}

fn json_str(v: &Value, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(s) = v.get(*k).and_then(Value::as_str).filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    None
}

fn json_expires_ms(v: &Value) -> Option<u64> {
    v.get("expires")
        .and_then(Value::as_u64)
        .or_else(|| v.get("expires").and_then(Value::as_i64).map(|n| n as u64))
}

pub(crate) fn convert_codex_auth(src: &Value) -> Option<Value> {
    let access = json_str(src, &["access", "access_token"])?;
    let refresh = json_str(src, &["refresh", "refresh_token"])?;
    let account = json_str(src, &["accountId", "account_id"]).unwrap_or_default();
    let expired = json_expires_ms(src)
        .map(epoch_ms_to_rfc3339)
        .unwrap_or_default();
    Some(serde_json::json!({
        "type": "codex",
        "access_token": access,
        "refresh_token": refresh,
        "id_token": json_str(src, &["id_token"]).unwrap_or_default(),
        "account_id": account,
        "email": "llmtrim-migrated",
        "last_refresh": chrono::Utc::now().to_rfc3339(),
        "expired": expired,
    }))
}

pub(crate) fn convert_kimi_auth(src: &Value) -> Option<Value> {
    let access = json_str(src, &["access", "access_token"])?;
    let refresh = json_str(src, &["refresh", "refresh_token"])?;
    let expired = json_expires_ms(src)
        .map(epoch_ms_to_rfc3339)
        .unwrap_or_default();
    Some(serde_json::json!({
        "type": "kimi",
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "scope": json_str(src, &["scope"]).unwrap_or_default(),
        "device_id": json_str(src, &["device_id", "deviceId", "userId"]).unwrap_or_default(),
        "expired": expired,
    }))
}

pub(crate) fn convert_grok_auth(src: &Value) -> Option<Value> {
    let access = json_str(src, &["access", "access_token"])?;
    let refresh = json_str(src, &["refresh", "refresh_token"])?;
    let expired = json_expires_ms(src)
        .map(epoch_ms_to_rfc3339)
        .unwrap_or_default();
    Some(serde_json::json!({
        "type": "xai",
        "auth_kind": "oauth",
        "access_token": access,
        "refresh_token": refresh,
        "id_token": json_str(src, &["id_token"]).unwrap_or_default(),
        "token_type": "Bearer",
        "expired": expired,
        "last_refresh": chrono::Utc::now().to_rfc3339(),
        "email": "llmtrim-migrated",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn asset_name_linux_amd64() {
        assert_eq!(
            release_asset_for("7.2.130", "linux", "x86_64").as_deref(),
            Some("CLIProxyAPI_7.2.130_linux_amd64.tar.gz")
        );
    }

    #[test]
    fn asset_name_darwin_arm() {
        assert_eq!(
            release_asset_for("v7.2.130", "macos", "aarch64").as_deref(),
            Some("CLIProxyAPI_7.2.130_darwin_aarch64.tar.gz")
        );
    }

    #[test]
    fn asset_name_windows_zip() {
        assert_eq!(
            release_asset_for("7.2.130", "windows", "x86_64").as_deref(),
            Some("CLIProxyAPI_7.2.130_windows_amd64.zip")
        );
    }

    #[test]
    fn asset_name_unknown_none() {
        assert_eq!(release_asset_for("7.2.130", "linux", "riscv64"), None);
    }

    #[test]
    fn split_url_strips_scheme_and_prefix() {
        assert_eq!(
            split_base_url("http://127.0.0.1:18317").unwrap(),
            ("127.0.0.1:18317".into(), String::new())
        );
        assert_eq!(
            split_base_url("http://127.0.0.1:8317/proxy").unwrap(),
            ("127.0.0.1:8317".into(), "/proxy".into())
        );
    }

    #[test]
    fn config_yaml_is_localhost_only() {
        let yaml = config_yaml(18317, "llmtrim-test", Path::new("/tmp/auth"));
        assert!(yaml.contains("host: \"127.0.0.1\""));
        assert!(yaml.contains("port: 18317"));
        assert!(yaml.contains("llmtrim-test"));
        assert!(yaml.contains("allow-remote: false"));
    }

    #[test]
    fn parse_models_reads_openai_list() {
        let v = json!({
            "data": [
                {"id": "gpt-5.4", "owned_by": "openai"},
                {"id": "gemini-3-flash", "owned_by": "google"},
                {"id": "gpt-5.4", "owned_by": "dup"},
                {"owned_by": "skip-me"}
            ]
        });
        let models = parse_models(&v);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gemini-3-flash");
        assert_eq!(models[1].id, "gpt-5.4");
        assert_eq!(models[1].owned_by, "openai");
    }

    #[test]
    fn convert_codex_maps_llmtrim_auth_json() {
        let src = json!({
            "access": "at-1",
            "refresh": "rt-1",
            "expires": 1_700_000_000_000u64,
            "accountId": "acct-9"
        });
        let out = convert_codex_auth(&src).unwrap();
        assert_eq!(out["type"], "codex");
        assert_eq!(out["access_token"], "at-1");
        assert_eq!(out["refresh_token"], "rt-1");
        assert_eq!(out["account_id"], "acct-9");
        assert!(out["expired"].as_str().unwrap().contains("2023"));
    }

    #[test]
    fn convert_kimi_and_grok_require_refresh() {
        assert!(convert_kimi_auth(&json!({"access": "a"})).is_none());
        let kimi = convert_kimi_auth(&json!({
            "access": "a",
            "refresh": "r",
            "expires": 1_700_000_000_000u64,
            "userId": "u1"
        }))
        .unwrap();
        assert_eq!(kimi["type"], "kimi");
        assert_eq!(kimi["device_id"], "u1");
        let grok = convert_grok_auth(&json!({
            "access": "a",
            "refresh": "r",
            "expires": 1_700_000_000_000u64
        }))
        .unwrap();
        assert_eq!(grok["type"], "xai");
        assert_eq!(grok["auth_kind"], "oauth");
    }

    #[test]
    fn parse_pin_accepts_every_cliproxy_backend() {
        assert_eq!(parse_pin_request("on"), Some(PinRequest::Enable));
        assert_eq!(
            parse_pin_request("gemini"),
            Some(PinRequest::Pin("gemini".into()))
        );
        assert_eq!(
            parse_pin_request("antigravity"),
            Some(PinRequest::Pin("antigravity".into()))
        );
        assert_eq!(
            parse_pin_request("claude"),
            Some(PinRequest::Pin("claude".into()))
        );
        assert_eq!(
            parse_pin_request("vertex"),
            Some(PinRequest::Pin("vertex".into()))
        );
        assert_eq!(
            parse_pin_request("qwen"),
            Some(PinRequest::Pin("qwen".into()))
        );
        assert_eq!(
            parse_pin_request("copilot"),
            Some(PinRequest::Pin("copilot".into()))
        );
        assert_eq!(
            parse_pin_request("gpt-5.4"),
            Some(PinRequest::Pin("gpt-5.4".into()))
        );
        assert!(parse_pin_request("off").is_none());
        assert!(parse_pin_request("no spaces allowed here!").is_none());
        assert_eq!(parse_hop("anthropic").as_deref(), Some("anthropic"));
        assert_eq!(parse_hop("codex").as_deref(), Some("codex"));
        assert_eq!(parse_hop("gemini").as_deref(), Some("gemini"));
        assert_eq!(parse_hop("on").as_deref(), Some("on"));
        assert!(parse_hop("nope").is_none());
    }

    #[test]
    fn expand_pin_uses_owned_by_then_prefix() {
        let models = vec![
            Model {
                id: "gemini-3-flash".into(),
                owned_by: "google".into(),
            },
            Model {
                id: "gpt-5.4".into(),
                owned_by: "openai".into(),
            },
        ];
        assert_eq!(
            expand_pin("gemini", &models).as_deref(),
            Some("gemini-3-flash")
        );
        assert_eq!(expand_pin("codex", &models).as_deref(), Some("gpt-5.4"));
        assert_eq!(expand_pin("gpt-5.4", &models).as_deref(), Some("gpt-5.4"));
        assert_eq!(expand_pin("kimi", &models), None);
    }

    #[test]
    fn official_catalog_search_and_tier_defaults() {
        let raw = json!({
            "xai": [
                {"id": "grok-4.6", "owned_by": "xai", "display_name": "Grok 4.6"},
                {"id": "grok-composer-2.5-fast", "owned_by": "xai", "display_name": "Composer Fast"}
            ],
            "claude": [
                {"id": "claude-opus-5", "owned_by": "anthropic", "display_name": "Opus 5"}
            ]
        });
        let cat = parse_official_models(&raw);
        assert_eq!(cat.len(), 3);
        let hits = search_official(&cat, "composer");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "grok-composer-2.5-fast");
        let grok = backend_by_alias("grok").unwrap();
        let map = default_tier_map(Some(grok), &cat);
        assert_eq!(map.get("opus").map(String::as_str), Some("grok-4.6"));
        assert_eq!(
            map.get("haiku").map(String::as_str),
            Some("grok-composer-2.5-fast")
        );
        assert!(!map.values().any(|v| v == "claude-opus-5"));
    }

    #[test]
    fn fallback_catalog_lists_grok_46() {
        let cat = fallback_official_models();
        assert!(cat.iter().any(|m| m.id == "grok-4.6"));
        assert!(cat.iter().any(|m| m.id == "grok-composer-2.5-fast"));
    }

    const PNG_2X2: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

    #[test]
    fn png_2x2_is_tiny() {
        assert_eq!(image_edges(PNG_2X2), Some((2, 2)));
        assert!(image_edges(PNG_2X2).is_some_and(|(w, h)| w < 8 || h < 8));
    }

    #[test]
    fn strip_tiny_images_replaces_2x2_keeps_text() {
        let mut body = json!({
            "model": "grok-4.6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": PNG_2X2
                    }}
                ]
            }]
        });
        assert_eq!(strip_tiny_images(&mut body), 1);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["text"], "see");
        assert_eq!(content[1]["type"], "text");
        assert!(content[1]["text"].as_str().unwrap().contains("8"));
    }
}
