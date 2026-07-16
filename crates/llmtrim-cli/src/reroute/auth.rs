//! OAuth + runtime token supply for the subscription reroute providers (Codex/ChatGPT, Kimi, and
//! Grok). Owns the interactive login flows (PKCE browser + device-code) and [`get_token`], the
//! hot-path token accessor that refreshes on the fly.
//!
//! **Blocking by design.** Everything here uses blocking `ureq` (no async): the CLI login
//! commands run on the main thread, and [`get_token`] is called from the serve layer via
//! `tokio::task::spawn_blocking`. A blocking mutex is therefore fine and correct.
//!
//! **Single-flight refresh.** Subscription refresh tokens ROTATE: two concurrent refreshes with
//! the same old token race, the loser 401s, and a 401 wipes the stored creds. [`get_token`]
//! guards each provider with a process-wide `Mutex` and re-checks expiry after taking the lock,
//! so only one caller ever fires a refresh.
//!
//! **Terms of service.** Driving a ChatGPT/Kimi/Grok subscription through a non-official client
//! violates the provider ToS and can get the account restricted. The login flows print this
//! warning before starting; use at your own risk.
//!
//! Storage: `~/.llmtrim/{codex,kimi,grok}/auth.json` (dir `0700`, file `0600` on unix), written
//! atomically (temp file + rename).

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use once_cell::sync::Lazy;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::daemon::home_dir;
use crate::reroute::SubProvider;

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// A valid access token for a provider, plus the ChatGPT account id (Codex) when known.
pub struct TokenSet {
    pub access: String,
    pub account_id: Option<String>,
}

/// Return a valid access token for `provider`, refreshing it if it is within the 5-minute
/// expiry margin. Process-wide single-flight: the per-provider lock plus a post-lock expiry
/// re-check guarantees concurrent callers don't each fire a (token-rotating) refresh. Blocking.
pub fn get_token(provider: SubProvider) -> Result<TokenSet> {
    let _guard = provider_lock(provider)
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    match provider {
        SubProvider::Codex => {
            let stored = read_codex().context(
                "no Codex credentials — run `llmtrim auth codex login` (or `... device`) first",
            )?;
            let now = now_ms();
            let stored = if needs_refresh(stored.expires, now) {
                codex_refresh(&stored)?
            } else {
                stored
            };
            Ok(TokenSet {
                access: stored.access,
                account_id: stored.account_id,
            })
        }
        SubProvider::Kimi => {
            let stored =
                read_kimi().context("no Kimi credentials — run `llmtrim auth kimi login` first")?;
            let now = now_ms();
            let stored = if needs_refresh(stored.expires, now) {
                kimi_refresh(&stored)?
            } else {
                stored
            };
            Ok(TokenSet {
                access: stored.access,
                account_id: None,
            })
        }
        SubProvider::Grok => {
            let stored = read_grok()
                .context("no Grok credentials — run `llmtrim sub auth grok login` first")?;
            let now = now_ms();
            let stored = if needs_refresh(stored.expires, now) {
                grok_refresh(&stored)?
            } else {
                stored
            };
            Ok(TokenSet {
                access: stored.access,
                account_id: None,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CODEX_ISSUER: &str = "https://auth.openai.com";
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_PORT: u16 = 1455;
const CODEX_REDIRECT: &str = "http://localhost:1455/auth/callback";
const CODEX_DEVICE_REDIRECT: &str = "https://auth.openai.com/deviceauth/callback";

const KIMI_HOST: &str = "https://auth.kimi.com";
const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const KIMI_CLI_VERSION: &str = "1.37.0";

/// Grok CLI public OAuth client (same as Grok Build / claude-code-proxy).
const GROK_ISSUER: &str = "https://auth.x.ai";
const GROK_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const GROK_SCOPES: &str = "openid profile email offline_access grok-cli:access api:access conversations:read conversations:write";

/// Refresh anything within 5 minutes of expiry.
const REFRESH_MARGIN_MS: u64 = 5 * 60 * 1000;

/// Ceiling on any single OAuth HTTP round-trip.
const HTTP_TIMEOUT: Duration = Duration::from_secs(60);

/// Overall wall-clock budget for the interactive flows (browser callback / device poll).
const FLOW_TIMEOUT: Duration = Duration::from_secs(300);

const TOS_WARNING: &str = "\
WARNING: Using a ChatGPT/Kimi/Grok subscription through a non-official client violates the
provider's Terms of Service and can get your account restricted or banned. This is
opt-in and unsupported. Proceed at your own risk.\n";

// ---------------------------------------------------------------------------
// Stored credential shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodexStored {
    access: String,
    refresh: String,
    /// Epoch milliseconds.
    expires: u64,
    /// New key `accountId`; also read the legacy `account_id`.
    #[serde(
        rename = "accountId",
        alias = "account_id",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    account_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KimiStored {
    access: String,
    refresh: String,
    /// Epoch milliseconds.
    expires: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<String>,
    #[serde(rename = "userId", default, skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GrokStored {
    access: String,
    refresh: String,
    /// Epoch milliseconds.
    expires: u64,
    #[serde(default)]
    issuer: String,
    #[serde(rename = "clientId", default)]
    client_id: String,
}

// ---------------------------------------------------------------------------
// Wire response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TokenResponse {
    #[serde(default)]
    id_token: Option<String>,
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default, deserialize_with = "de_opt_u64")]
    expires_in: Option<u64>,
    #[serde(default)]
    scope: Option<String>,
}

// ---------------------------------------------------------------------------
// Single-flight locks
// ---------------------------------------------------------------------------

static CODEX_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static KIMI_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static GROK_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

fn provider_lock(provider: SubProvider) -> &'static Mutex<()> {
    match provider {
        SubProvider::Codex => &CODEX_LOCK,
        SubProvider::Kimi => &KIMI_LOCK,
        SubProvider::Grok => &GROK_LOCK,
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested)
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// True when a token expiring at `expires_ms` (epoch ms) is at or within the 5-minute margin of
/// `now_ms` and should be refreshed.
fn needs_refresh(expires_ms: u64, now_ms: u64) -> bool {
    expires_ms <= now_ms.saturating_add(REFRESH_MARGIN_MS)
}

fn b64url_encode(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

fn b64url_decode(s: &str) -> Result<Vec<u8>> {
    URL_SAFE_NO_PAD
        .decode(s.as_bytes())
        .context("base64url decode failed")
}

/// PKCE challenge = base64url-nopad(SHA256(verifier)).
fn pkce_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    b64url_encode(&digest)
}

/// 32 random bytes, base64url-nopad — used for the PKCE verifier and the OAuth `state`.
fn random_b64url_32() -> String {
    use rand::RngCore;
    let mut buf = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut buf);
    b64url_encode(&buf)
}

/// A fresh uuid v4 with dashes stripped (32 hex chars). Kept separate from [`kimi_device_id`] so
/// the format is unit-testable without touching the filesystem.
fn new_device_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Extract the ChatGPT account id from a JWT: decode the middle (payload) segment as base64url
/// and read, in precedence order, `chatgpt_account_id`,
/// `["https://api.openai.com/auth"].chatgpt_account_id`, then `organizations[0].id`.
fn account_id_from_jwt(jwt: &str) -> Option<String> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = b64url_decode(payload).ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&bytes).ok()?;

    if let Some(id) = claims.get("chatgpt_account_id").and_then(|v| v.as_str()) {
        return Some(id.to_string());
    }
    if let Some(id) = claims
        .get("https://api.openai.com/auth")
        .and_then(|a| a.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
    {
        return Some(id.to_string());
    }
    claims
        .get("organizations")
        .and_then(|o| o.get(0))
        .and_then(|o| o.get("id"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Read a single JWT claim string from the payload segment (best-effort). Used for Kimi's
/// `user_id`.
fn jwt_claim_str(jwt: &str, claim: &str) -> Option<String> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = b64url_decode(payload).ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    claims
        .get(claim)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// `expires_in` (seconds, defaulted) → absolute epoch-ms expiry.
fn expiry_ms(expires_in: Option<u64>, default_secs: u64) -> u64 {
    now_ms().saturating_add(expires_in.unwrap_or(default_secs).saturating_mul(1000))
}

/// Deserialize an optional `u64` that a provider may send as either a JSON number or a JSON string
/// (`"5"`). OAuth device/token endpoints are inconsistent about `interval`/`expires_in`; a strict
/// `Option<u64>` rejects the string form and aborts the whole flow. An unparseable string reads as
/// `None` (the caller's default applies) rather than erroring.
fn de_opt_u64<'de, D>(d: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StrOrNum {
        S(String),
        N(u64),
    }
    Ok(match Option::<StrOrNum>::deserialize(d)? {
        None => None,
        Some(StrOrNum::N(n)) => Some(n),
        Some(StrOrNum::S(s)) => s.trim().parse().ok(),
    })
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

fn codex_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join("codex"))
}

fn kimi_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join("kimi"))
}

fn grok_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join("grok"))
}

fn codex_auth_path() -> Result<PathBuf> {
    Ok(codex_dir()?.join("auth.json"))
}

fn kimi_auth_path() -> Result<PathBuf> {
    Ok(kimi_dir()?.join("auth.json"))
}

fn grok_auth_path() -> Result<PathBuf> {
    Ok(grok_dir()?.join("auth.json"))
}

/// Create `dir` (recursively) and, on unix, tighten it to `0700`.
fn ensure_dir(dir: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create dir {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to chmod 0700 {}", dir.display()))?;
    }
    Ok(())
}

/// Atomic write (temp file in the same dir + rename), `0600` on unix.
fn write_atomic(path: &PathBuf, contents: &str) -> Result<()> {
    let dir = path
        .parent()
        .map(PathBuf::from)
        .context("auth path has no parent directory")?;
    ensure_dir(&dir)?;
    let tmp = dir.join(format!(".auth.{}.tmp", std::process::id()));
    {
        // Create the temp file `0600` at open time (not chmod-after), so the token bytes are never
        // group/other-readable even for the instant between write and chmod.
        #[cfg(unix)]
        let mut f = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)
                .with_context(|| format!("failed to create temp file {}", tmp.display()))?
        };
        #[cfg(not(unix))]
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("failed to create temp file {}", tmp.display()))?;
        f.write_all(contents.as_bytes())
            .with_context(|| format!("failed to write {}", tmp.display()))?;
        f.flush().ok();
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

fn read_codex() -> Result<CodexStored> {
    let path = codex_auth_path()?;
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_codex(auth: &CodexStored) -> Result<()> {
    let path = codex_auth_path()?;
    let json = serde_json::to_string_pretty(auth).context("failed to serialize Codex auth")?;
    write_atomic(&path, &json)
}

fn read_kimi() -> Result<KimiStored> {
    let path = kimi_auth_path()?;
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_kimi(auth: &KimiStored) -> Result<()> {
    let path = kimi_auth_path()?;
    let json = serde_json::to_string_pretty(auth).context("failed to serialize Kimi auth")?;
    write_atomic(&path, &json)
}

fn read_grok() -> Result<GrokStored> {
    let path = grok_auth_path()?;
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_grok(auth: &GrokStored) -> Result<()> {
    let path = grok_auth_path()?;
    let json = serde_json::to_string_pretty(auth).context("failed to serialize Grok auth")?;
    write_atomic(&path, &json)
}

fn delete_if_exists(path: &PathBuf) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("failed to delete {}", path.display())),
    }
}

// ---------------------------------------------------------------------------
// Kimi persistent device id
// ---------------------------------------------------------------------------

/// The persistent Kimi device id (uuid v4, dashes stripped): generated and stored at
/// `~/.llmtrim/kimi/device_id` on first use, then reused forever. Falls back to a freshly minted
/// (non-persisted) id if the file can't be read or written, so a callable value is always
/// returned.
pub fn kimi_device_id() -> String {
    let Ok(dir) = kimi_dir() else {
        return new_device_id();
    };
    let path = dir.join("device_id");
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let id = new_device_id();
    if ensure_dir(&dir).is_ok() {
        let _ = write_atomic(&path, &id);
    }
    id
}

// ---------------------------------------------------------------------------
// HTTP plumbing (factored out so the pure logic above stays testable offline)
// ---------------------------------------------------------------------------

/// POST a form-urlencoded body. Returns `(status, body_text)`. `.proxy(None)` suppresses any
/// ambient `HTTPS_PROXY` (which points at the llmtrim daemon) so auth never loops back through us.
fn post_form(
    url: &str,
    headers: &[(&str, &str)],
    form: &[(&str, String)],
) -> Result<(u16, String)> {
    let mut req = ureq::post(url)
        .config()
        .proxy(None)
        .http_status_as_error(false)
        .timeout_global(Some(HTTP_TIMEOUT))
        .build();
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let mut resp = req
        .send_form(form.iter().map(|(k, v)| (*k, v.as_str())))
        .map_err(|e| anyhow!("POST {url} failed: {e}"))?;
    let status = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().unwrap_or_default();
    Ok((status, body))
}

/// POST a JSON body. Returns `(status, body_text)`.
fn post_json(
    url: &str,
    headers: &[(&str, &str)],
    body: &serde_json::Value,
) -> Result<(u16, String)> {
    let mut req = ureq::post(url)
        .config()
        .proxy(None)
        .http_status_as_error(false)
        .timeout_global(Some(HTTP_TIMEOUT))
        .build();
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let payload = serde_json::to_string(body).unwrap_or_else(|_| "{}".to_string());
    req = req.header("content-type", "application/json");
    let mut resp = req
        .send(&payload)
        .map_err(|e| anyhow!("POST {url} failed: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string().unwrap_or_default();
    Ok((status, text))
}

/// The `X-Msh-*` + user-agent header set Kimi's CLI sends. `device_id` must outlive the call.
fn kimi_headers(device_id: &str) -> Vec<(&'static str, String)> {
    vec![
        ("X-Msh-Platform", "kimi_cli".to_string()),
        ("X-Msh-Version", KIMI_CLI_VERSION.to_string()),
        ("X-Msh-Device-Id", device_id.to_string()),
        ("User-Agent", format!("KimiCLI/{KIMI_CLI_VERSION}")),
    ]
}

fn as_ref_pairs<'a>(headers: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
    headers.iter().map(|(k, v)| (*k, v.as_str())).collect()
}

// ---------------------------------------------------------------------------
// Codex token exchange + account id
// ---------------------------------------------------------------------------

/// Turn an OAuth token response into stored Codex creds, deriving the account id from the JWT
/// (id_token first, else access_token). `prev_account_id` is kept if the fresh tokens omit one.
fn codex_stored_from(tr: TokenResponse, prev_account_id: Option<String>) -> CodexStored {
    let account_id = tr
        .id_token
        .as_deref()
        .and_then(account_id_from_jwt)
        .or_else(|| account_id_from_jwt(&tr.access_token))
        .or(prev_account_id);
    CodexStored {
        access: tr.access_token,
        refresh: tr.refresh_token.unwrap_or_default(),
        expires: expiry_ms(tr.expires_in, 3600),
        account_id,
    }
}

fn codex_exchange_code(
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
) -> Result<TokenResponse> {
    let url = format!("{CODEX_ISSUER}/oauth/token");
    let form = [
        ("grant_type", "authorization_code".to_string()),
        ("code", code.to_string()),
        ("redirect_uri", redirect_uri.to_string()),
        ("client_id", CODEX_CLIENT_ID.to_string()),
        ("code_verifier", code_verifier.to_string()),
    ];
    let (status, body) = post_form(&url, &[], &form)?;
    if !(200..300).contains(&status) {
        bail!("Codex token exchange failed (HTTP {status}): {body}");
    }
    serde_json::from_str(&body).context("failed to parse Codex token response")
}

fn codex_refresh(stored: &CodexStored) -> Result<CodexStored> {
    let url = format!("{CODEX_ISSUER}/oauth/token");
    let form = [
        ("client_id", CODEX_CLIENT_ID.to_string()),
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", stored.refresh.clone()),
    ];
    let (status, body) = post_form(&url, &[], &form)?;
    if status == 401 || status == 403 {
        let _ = delete_if_exists(&codex_auth_path()?);
        bail!(
            "Codex refresh rejected (HTTP {status}); credentials cleared — run \
             `llmtrim auth codex login` again"
        );
    }
    if !(200..300).contains(&status) {
        bail!("Codex refresh failed (HTTP {status}): {body}");
    }
    let tr: TokenResponse =
        serde_json::from_str(&body).context("failed to parse Codex refresh response")?;
    let mut fresh = codex_stored_from(tr, stored.account_id.clone());
    // A refresh response may omit refresh_token; keep the old one so the chain survives.
    if fresh.refresh.is_empty() {
        fresh.refresh = stored.refresh.clone();
    }
    write_codex(&fresh)?;
    Ok(fresh)
}

// ---------------------------------------------------------------------------
// Codex — PKCE browser login (127.0.0.1:1455 callback)
// ---------------------------------------------------------------------------

/// PKCE browser flow: prints an authorize URL, runs a one-shot loopback listener on
/// `127.0.0.1:1455`, exchanges the returned code, stores the creds.
pub fn codex_login() -> Result<()> {
    print!("{TOS_WARNING}");

    let verifier = random_b64url_32();
    let challenge = pkce_challenge(&verifier);
    let state = random_b64url_32();

    let authorize = format!(
        "{CODEX_ISSUER}/oauth/authorize?response_type=code&client_id={cid}\
         &redirect_uri={redir}&scope=openid%20profile%20email%20offline_access\
         &code_challenge={chal}&code_challenge_method=S256&id_token_add_organizations=true\
         &codex_cli_simplified_flow=true&state={state}&originator=llmtrim",
        cid = CODEX_CLIENT_ID,
        redir = urlencode(CODEX_REDIRECT),
        chal = challenge,
        state = state,
    );

    // Bind before printing so a failure to grab the port surfaces before the user clicks.
    let listener = std::net::TcpListener::bind(("127.0.0.1", CODEX_PORT))
        .with_context(|| format!("failed to bind 127.0.0.1:{CODEX_PORT} for the OAuth callback"))?;

    println!("Open this URL in your browser to authorize llmtrim:\n\n  {authorize}\n");
    try_open_browser(&authorize);
    println!("Waiting for the authorization callback (up to 5 minutes)...");

    let code = wait_for_callback(listener, &state)?;
    let tr = codex_exchange_code(&code, CODEX_REDIRECT, &verifier)?;
    let stored = codex_stored_from(tr, None);
    write_codex(&stored)?;

    println!(
        "Codex login complete{}.",
        match &stored.account_id {
            Some(id) => format!(" (account {id})"),
            None => String::new(),
        }
    );
    Ok(())
}

/// Block on the one-shot callback listener until it receives `GET /auth/callback?code=&state=`
/// with a matching `state`, or the 5-minute budget elapses. Returns the authorization code.
fn wait_for_callback(listener: std::net::TcpListener, expected_state: &str) -> Result<String> {
    listener
        .set_nonblocking(true)
        .context("failed to set callback listener non-blocking")?;
    let deadline = std::time::Instant::now() + FLOW_TIMEOUT;

    loop {
        if std::time::Instant::now() >= deadline {
            bail!("timed out waiting for the OAuth callback after 5 minutes");
        }
        match listener.accept() {
            Ok((mut stream, _addr)) => {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let target = first_request_target(&req);

                match target.and_then(parse_callback) {
                    Some((code, state)) => {
                        if state != expected_state {
                            respond(&mut stream, "state mismatch — please retry the login");
                            bail!("OAuth state mismatch (possible CSRF); aborting");
                        }
                        respond(
                            &mut stream,
                            "llmtrim authorization received. You can close this tab.",
                        );
                        return Ok(code);
                    }
                    None => {
                        // Favicon / probe / unrelated path — answer politely and keep waiting.
                        respond(&mut stream, "waiting for the authorization callback...");
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(e).context("callback listener accept failed"),
        }
    }
}

/// Extract the request target (path?query) from the first line of an HTTP request.
fn first_request_target(req: &str) -> Option<&str> {
    let line = req.lines().next()?;
    let mut parts = line.split_whitespace();
    let _method = parts.next()?;
    parts.next()
}

/// Parse `/auth/callback?code=X&state=Y` into `(code, state)`. Returns `None` for any other path
/// or when `code` is absent.
fn parse_callback(target: &str) -> Option<(String, String)> {
    let (path, query) = target.split_once('?')?;
    if path != "/auth/callback" {
        return None;
    }
    let mut code = None;
    let mut state = None;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=')?;
        match k {
            "code" => code = Some(urldecode(v)),
            "state" => state = Some(urldecode(v)),
            _ => {}
        }
    }
    Some((code?, state.unwrap_or_default()))
}

fn respond(stream: &mut std::net::TcpStream, message: &str) {
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>llmtrim</title></head>\
         <body style=\"font-family:system-ui;padding:3rem;text-align:center\">\
         <h2>{message}</h2></body></html>"
    );
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        html.len(),
        html
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

// ---------------------------------------------------------------------------
// Codex — device-code login (headless)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CodexDeviceStart {
    device_auth_id: String,
    user_code: String,
    #[serde(default, deserialize_with = "de_opt_u64")]
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CodexDevicePoll {
    authorization_code: String,
    code_verifier: String,
}

/// Device-code flow for headless machines: prints a code to enter at the verification page, polls
/// for approval, then exchanges the returned code.
pub fn codex_device() -> Result<()> {
    print!("{TOS_WARNING}");

    let start_url = format!("{CODEX_ISSUER}/api/accounts/deviceauth/usercode");
    let (status, body) = post_json(
        &start_url,
        &[],
        &serde_json::json!({ "client_id": CODEX_CLIENT_ID }),
    )?;
    if !(200..300).contains(&status) {
        bail!("Codex device start failed (HTTP {status}): {body}");
    }
    let start: CodexDeviceStart =
        serde_json::from_str(&body).context("failed to parse Codex device-start response")?;

    println!(
        "To authorize llmtrim, visit:\n\n  {CODEX_ISSUER}/codex/device\n\nand enter the code: {}\n",
        start.user_code
    );

    let mut interval = start.interval.unwrap_or(5);
    let poll_url = format!("{CODEX_ISSUER}/api/accounts/deviceauth/token");
    let deadline = std::time::Instant::now() + FLOW_TIMEOUT;

    let poll: CodexDevicePoll = loop {
        if std::time::Instant::now() >= deadline {
            bail!("timed out waiting for device authorization after 5 minutes");
        }
        std::thread::sleep(Duration::from_secs(interval));
        let (status, body) = post_json(
            &poll_url,
            &[],
            &serde_json::json!({
                "device_auth_id": start.device_auth_id,
                "user_code": start.user_code,
            }),
        )?;
        match status {
            s if (200..300).contains(&s) => {
                break serde_json::from_str(&body)
                    .context("failed to parse Codex device-poll response")?;
            }
            403 | 404 => {
                // Not authorized yet — back off and keep polling.
                interval = interval.saturating_add(3);
            }
            s => bail!("Codex device poll failed (HTTP {s}): {body}"),
        }
    };

    let tr = codex_exchange_code(
        &poll.authorization_code,
        CODEX_DEVICE_REDIRECT,
        &poll.code_verifier,
    )?;
    let stored = codex_stored_from(tr, None);
    write_codex(&stored)?;
    println!("Codex device login complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Codex — status / logout
// ---------------------------------------------------------------------------

pub fn codex_status() -> Result<()> {
    let stored = read_codex().context("no Codex credentials stored")?;
    let account = stored.account_id.as_deref().unwrap_or("(unknown)");
    println!("Codex: logged in");
    println!("  account id: {account}");
    println!("  {}", expiry_line(stored.expires));
    Ok(())
}

/// Machine-readable login state for `provider`: `{ logged_in, expires_ms, account_id }`. Used by
/// `auth status --json` and the tray so the UI can show connect/disconnect state without parsing
/// the human status text. Never errors — a missing/invalid credential file reads as logged out.
pub fn auth_status_json(provider: super::SubProvider) -> serde_json::Value {
    use super::SubProvider;
    let stored = match provider {
        SubProvider::Codex => read_codex().ok().map(|s| (s.expires, s.account_id)),
        SubProvider::Kimi => read_kimi().ok().map(|s| (s.expires, None)),
        SubProvider::Grok => read_grok().ok().map(|s| (s.expires, None)),
    };
    match stored {
        Some((expires, account_id)) => serde_json::json!({
            "logged_in": true,
            "expires_ms": expires,
            "account_id": account_id,
        }),
        None => serde_json::json!({ "logged_in": false }),
    }
}

pub fn codex_logout() -> Result<()> {
    delete_if_exists(&codex_auth_path()?)?;
    println!("Codex credentials deleted.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Kimi — device-code login (RFC 8628)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct KimiDeviceStart {
    user_code: String,
    device_code: String,
    verification_uri_complete: String,
    #[serde(default, deserialize_with = "de_opt_u64")]
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct KimiPollError {
    error: String,
}

pub fn kimi_login() -> Result<()> {
    print!("{TOS_WARNING}");

    let device_id = kimi_device_id();
    let headers = kimi_headers(&device_id);
    let hdr_refs = as_ref_pairs(&headers);

    let start_url = format!("{KIMI_HOST}/api/oauth/device_authorization");
    let (status, body) = post_form(
        &start_url,
        &hdr_refs,
        &[("client_id", KIMI_CLIENT_ID.to_string())],
    )?;
    if !(200..300).contains(&status) {
        bail!("Kimi device authorization failed (HTTP {status}): {body}");
    }
    let start: KimiDeviceStart = serde_json::from_str(&body)
        .context("failed to parse Kimi device-authorization response")?;

    println!(
        "To authorize llmtrim, open:\n\n  {}\n\nand confirm the code: {}\n",
        start.verification_uri_complete, start.user_code
    );

    let mut interval = start.interval.unwrap_or(5);
    let poll_url = format!("{KIMI_HOST}/api/oauth/token");
    let deadline = std::time::Instant::now() + FLOW_TIMEOUT;

    let tr: TokenResponse = loop {
        if std::time::Instant::now() >= deadline {
            bail!("timed out waiting for Kimi device authorization after 5 minutes");
        }
        std::thread::sleep(Duration::from_secs(interval) + Duration::from_millis(500));
        let (status, body) = post_form(
            &poll_url,
            &hdr_refs,
            &[
                ("client_id", KIMI_CLIENT_ID.to_string()),
                ("device_code", start.device_code.clone()),
                (
                    "grant_type",
                    "urn:ietf:params:oauth:grant-type:device_code".to_string(),
                ),
            ],
        )?;
        if (200..300).contains(&status) {
            break serde_json::from_str(&body).context("failed to parse Kimi token response")?;
        }
        // Non-2xx: inspect the OAuth error code to decide keep-polling vs abort.
        match serde_json::from_str::<KimiPollError>(&body) {
            Ok(e) => match e.error.as_str() {
                "authorization_pending" => {}
                "slow_down" => interval = interval.saturating_add(1),
                "expired_token" => bail!("Kimi device code expired; run login again"),
                other => bail!("Kimi authorization failed: {other}"),
            },
            Err(_) => bail!("Kimi device poll failed (HTTP {status}): {body}"),
        }
    };

    let stored = kimi_stored_from(tr, &device_id, None);
    write_kimi(&stored)?;
    println!("Kimi login complete.");
    Ok(())
}

/// Build stored Kimi creds from a token response. `user_id` comes from the JWT `user_id` claim
/// (best-effort), falling back to `prev_user_id`.
fn kimi_stored_from(
    tr: TokenResponse,
    _device_id: &str,
    prev_user_id: Option<String>,
) -> KimiStored {
    let user_id = jwt_claim_str(&tr.access_token, "user_id").or(prev_user_id);
    KimiStored {
        access: tr.access_token,
        refresh: tr.refresh_token.unwrap_or_default(),
        expires: expiry_ms(tr.expires_in, 900),
        scope: tr.scope,
        user_id,
    }
}

fn kimi_refresh(stored: &KimiStored) -> Result<KimiStored> {
    let device_id = kimi_device_id();
    let headers = kimi_headers(&device_id);
    let hdr_refs = as_ref_pairs(&headers);

    let url = format!("{KIMI_HOST}/api/oauth/token");
    let (status, body) = post_form(
        &url,
        &hdr_refs,
        &[
            ("client_id", KIMI_CLIENT_ID.to_string()),
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", stored.refresh.clone()),
        ],
    )?;
    if status == 401 || status == 403 {
        let _ = delete_if_exists(&kimi_auth_path()?);
        bail!(
            "Kimi refresh rejected (HTTP {status}); credentials cleared — run \
             `llmtrim auth kimi login` again"
        );
    }
    if !(200..300).contains(&status) {
        bail!("Kimi refresh failed (HTTP {status}): {body}");
    }
    let tr: TokenResponse =
        serde_json::from_str(&body).context("failed to parse Kimi refresh response")?;
    let mut fresh = kimi_stored_from(tr, &device_id, stored.user_id.clone());
    if fresh.refresh.is_empty() {
        fresh.refresh = stored.refresh.clone();
    }
    if fresh.scope.is_none() {
        fresh.scope = stored.scope.clone();
    }
    write_kimi(&fresh)?;
    Ok(fresh)
}

pub fn kimi_status() -> Result<()> {
    let stored = read_kimi().context("no Kimi credentials stored")?;
    println!("Kimi: logged in");
    if let Some(id) = &stored.user_id {
        println!("  user id: {id}");
    }
    println!("  {}", expiry_line(stored.expires));
    Ok(())
}

pub fn kimi_logout() -> Result<()> {
    delete_if_exists(&kimi_auth_path()?)?;
    println!("Kimi credentials deleted.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Grok — OIDC discovery + PKCE browser login + device-code / paste fallback
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GrokDiscovery {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
    /// Present when the issuer supports RFC 8628 device authorization.
    #[serde(default)]
    device_authorization_endpoint: Option<String>,
}

fn grok_discover() -> Result<GrokDiscovery> {
    let url = format!("{GROK_ISSUER}/.well-known/openid-configuration");
    let mut resp = ureq::get(&url)
        .config()
        .proxy(None)
        .http_status_as_error(false)
        .timeout_global(Some(HTTP_TIMEOUT))
        .build()
        .call()
        .map_err(|e| anyhow!("Grok OIDC discovery failed: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string().unwrap_or_default();
    if !(200..300).contains(&status) {
        bail!("Grok OIDC discovery failed (HTTP {status}): {text}");
    }
    let discovery: GrokDiscovery =
        serde_json::from_str(&text).context("failed to parse Grok OIDC discovery")?;
    if discovery.issuer != GROK_ISSUER {
        bail!("Grok OIDC discovery issuer mismatch");
    }
    // Endpoints must stay on auth.x.ai.
    let mut endpoints = vec![
        discovery.authorization_endpoint.as_str(),
        discovery.token_endpoint.as_str(),
    ];
    if let Some(ref d) = discovery.device_authorization_endpoint {
        endpoints.push(d.as_str());
    }
    for endpoint in endpoints {
        if !endpoint.starts_with(GROK_ISSUER) {
            bail!("Grok OIDC endpoint is outside the canonical issuer: {endpoint}");
        }
    }
    Ok(discovery)
}

/// Build stored creds from a token response.
///
/// `require_refresh`: login/authorization-code must grant an offline session. Refresh responses
/// often omit `refresh_token` (RFC 6749); pass `false` and supply `prev_refresh` so the chain
/// survives (same pattern as Codex/Kimi).
fn grok_stored_from(
    tr: TokenResponse,
    require_refresh: bool,
    prev_refresh: Option<&str>,
) -> Result<GrokStored> {
    if tr.access_token.is_empty() {
        bail!("Grok token response missing access_token");
    }
    let refresh = tr
        .refresh_token
        .filter(|r| !r.is_empty())
        .or_else(|| prev_refresh.map(str::to_string))
        .filter(|r| !r.is_empty());
    let refresh = match refresh {
        Some(r) => r,
        None if require_refresh => {
            bail!("Grok login did not grant an offline session (no refresh_token)")
        }
        None => bail!("Grok token response missing refresh_token and no prior session to keep"),
    };
    Ok(GrokStored {
        access: tr.access_token,
        refresh,
        expires: expiry_ms(tr.expires_in, 3600),
        issuer: GROK_ISSUER.to_string(),
        client_id: GROK_CLIENT_ID.to_string(),
    })
}

fn grok_exchange_code(
    token_endpoint: &str,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
) -> Result<TokenResponse> {
    let form = [
        ("grant_type", "authorization_code".to_string()),
        ("code", code.to_string()),
        ("redirect_uri", redirect_uri.to_string()),
        ("client_id", GROK_CLIENT_ID.to_string()),
        ("code_verifier", code_verifier.to_string()),
    ];
    let (status, body) = post_form(token_endpoint, &[], &form)?;
    if !(200..300).contains(&status) {
        bail!("Grok token exchange failed (HTTP {status}): {body}");
    }
    serde_json::from_str(&body).context("failed to parse Grok token response")
}

fn grok_refresh(stored: &GrokStored) -> Result<GrokStored> {
    if stored.issuer != GROK_ISSUER || stored.client_id != GROK_CLIENT_ID {
        bail!("unsupported Grok OAuth session — run `llmtrim sub auth grok login` again");
    }
    let discovery = grok_discover()?;
    let form = [
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", stored.refresh.clone()),
        ("client_id", GROK_CLIENT_ID.to_string()),
    ];
    let (status, body) = post_form(&discovery.token_endpoint, &[], &form)?;
    if status == 401 || status == 403 {
        let _ = delete_if_exists(&grok_auth_path()?);
        bail!(
            "Grok refresh rejected (HTTP {status}); credentials cleared — run \
             `llmtrim sub auth grok login` again"
        );
    }
    if !(200..300).contains(&status) {
        bail!("Grok refresh failed (HTTP {status}): {body}");
    }
    let tr: TokenResponse =
        serde_json::from_str(&body).context("failed to parse Grok refresh response")?;
    // Refresh responses may omit refresh_token; keep the prior one (Codex/Kimi do the same).
    let fresh = grok_stored_from(tr, false, Some(&stored.refresh))?;
    write_grok(&fresh)?;
    Ok(fresh)
}

/// PKCE browser flow against `auth.x.ai` with an ephemeral loopback callback.
///
/// When the browser cannot reach this machine (SSH, container, remote desktop),
/// xAI may show a one-time code (or a `http://127.0.0.1.../callback?...` URL).
/// Paste that into the terminal — same pattern as pi-grok-cli / Hermes paste
/// fallback. For fully headless machines prefer [`grok_device`].
pub fn grok_login() -> Result<()> {
    print!("{TOS_WARNING}");

    let discovery = grok_discover()?;
    let verifier = random_b64url_32();
    let challenge = pkce_challenge(&verifier);
    let state = random_b64url_32();

    // Ephemeral port (unlike Codex's fixed 1455) so concurrent tools don't collide.
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .context("failed to bind loopback for the Grok OAuth callback")?;
    let port = listener
        .local_addr()
        .context("failed to read Grok callback port")?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let authorize = format!(
        "{auth}?response_type=code&client_id={cid}&redirect_uri={redir}&scope={scope}\
         &code_challenge={chal}&code_challenge_method=S256&state={state}",
        auth = discovery.authorization_endpoint,
        cid = urlencode(GROK_CLIENT_ID),
        redir = urlencode(&redirect_uri),
        scope = urlencode(GROK_SCOPES),
        chal = challenge,
        state = state,
    );

    println!("Open this URL in your browser to authorize llmtrim:\n\n  {authorize}\n");
    try_open_browser(&authorize);
    println!(
        "Waiting for the authorization callback (up to 5 minutes)...\n\
         If the browser cannot reach this machine, paste the full callback URL\n\
         or the one-time code xAI shows (\"copy into Grok Build\") and press Enter.\n\
         Headless alternative: `llmtrim sub auth grok device`.\n"
    );

    let code = wait_for_grok_callback(listener, &state)?;
    let tr = grok_exchange_code(&discovery.token_endpoint, &code, &redirect_uri, &verifier)?;
    let stored = grok_stored_from(tr, true, None)?;
    write_grok(&stored)?;
    println!("Grok login complete.");
    Ok(())
}

/// Same shape as Codex's callback waiter, but the path is `/callback` (Grok OIDC convention)
/// rather than `/auth/callback`. Also accepts a pasted callback URL or bare authorization
/// code from stdin so remote users can finish login on their own device.
fn wait_for_grok_callback(listener: std::net::TcpListener, expected_state: &str) -> Result<String> {
    listener
        .set_nonblocking(true)
        .context("failed to set Grok callback listener non-blocking")?;
    let deadline = std::time::Instant::now() + FLOW_TIMEOUT;

    // Background stdin reader: paste races the loopback callback; first valid wins.
    let paste_rx = spawn_stdin_line_reader();

    loop {
        if std::time::Instant::now() >= deadline {
            bail!("timed out waiting for the Grok OAuth callback after 5 minutes");
        }

        // Manual paste (callback URL, query string, or bare code) from the other device.
        match paste_rx.try_recv() {
            Ok(line) => match parse_grok_manual_paste(line.trim(), expected_state) {
                Ok(code) => return Ok(code),
                Err(err) => {
                    eprintln!("Ignored pasted input: {err}");
                    eprintln!(
                        "Paste the complete callback URL (http://127.0.0.1.../callback?code=...)\n\
                         or the one-time code xAI showed, then press Enter."
                    );
                }
            },
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                // Stdin closed (EOF); loopback callback can still finish the flow.
            }
        }

        match listener.accept() {
            Ok((mut stream, _addr)) => {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let target = first_request_target(&req);

                match target.and_then(parse_grok_callback) {
                    Some((code, state)) => {
                        if state != expected_state {
                            respond(&mut stream, "state mismatch — please retry the login");
                            bail!("OAuth state mismatch (possible CSRF); aborting");
                        }
                        respond(
                            &mut stream,
                            "llmtrim authorization received. You can close this tab.",
                        );
                        return Ok(code);
                    }
                    None => {
                        respond(&mut stream, "waiting for the authorization callback...");
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(e).context("Grok callback listener accept failed"),
        }
    }
}

/// Spawn a thread that forwards non-empty stdin lines until EOF or the receiver drops.
fn spawn_stdin_line_reader() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut stdin = stdin.lock();
        loop {
            let mut line = String::new();
            match stdin.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) if line.trim().is_empty() => continue,
                Ok(_) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    rx
}

/// Parse `/callback?code=X&state=Y` into `(code, state)`.
fn parse_grok_callback(target: &str) -> Option<(String, String)> {
    let (path, query) = target.split_once('?')?;
    if path != "/callback" {
        return None;
    }
    parse_grok_callback_query(query)
}

fn parse_grok_callback_query(query: &str) -> Option<(String, String)> {
    let mut code = None;
    let mut state = None;
    for pair in query.split('&') {
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        match k {
            "code" => code = Some(urldecode(v)),
            "state" => state = Some(urldecode(v)),
            "error" => return None,
            _ => {}
        }
    }
    Some((code?, state.unwrap_or_default()))
}

/// Accept a pasted callback URL, `?code=&state=` query, or bare authorization code.
///
/// Matches pi-grok-cli / Hermes remote paste: when xAI cannot redirect to loopback it
/// shows a one-time code (French UI: "copiez le code dans Grok Build") for the CLI.
fn parse_grok_manual_paste(input: &str, expected_state: &str) -> Result<String> {
    let value = input.trim();
    if value.is_empty() {
        bail!("pasted input was empty");
    }

    // Full URL: http://127.0.0.1:PORT/callback?code=...&state=...
    if value.starts_with("http://") || value.starts_with("https://") {
        let after_scheme = value
            .split_once("://")
            .map(|(_, rest)| rest)
            .unwrap_or(value);
        let path_and_query = after_scheme
            .find('/')
            .map(|i| &after_scheme[i..])
            .unwrap_or("/");
        let (code, state) = parse_grok_callback(path_and_query)
            .ok_or_else(|| anyhow!("pasted URL is not a Grok /callback with code="))?;
        if !state.is_empty() && state != expected_state {
            bail!("OAuth state mismatch (possible CSRF); aborting");
        }
        return Ok(code);
    }

    // Query fragment: code=...&state=...  (optional leading ?)
    let as_query = value.strip_prefix('?').unwrap_or(value);
    if as_query.contains("code=") {
        let (code, state) = parse_grok_callback_query(as_query)
            .ok_or_else(|| anyhow!("pasted query did not contain a valid code="))?;
        if !state.is_empty() && state != expected_state {
            bail!("OAuth state mismatch (possible CSRF); aborting");
        }
        return Ok(code);
    }

    // Bare one-time authorization code (xAI "copy into Grok Build" page).
    if is_plausible_oauth_code(value) {
        return Ok(value.to_string());
    }

    bail!("unrecognized paste — expected a callback URL, code=... query, or one-time auth code")
}

/// Heuristic for a bare OAuth authorization code (not a short device user_code like ABCD-1234).
fn is_plausible_oauth_code(s: &str) -> bool {
    let len = s.len();
    (32..=2048).contains(&len)
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '~'))
}

/// True for `https://x.ai/...` or `https://*.x.ai/...` (with optional port). Used to reject
/// open redirects from a compromised discovery/device response.
fn is_trusted_xai_https_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let host_port = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host_port
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(host_port);
    // Drop port; IPv6 literals are not expected for xAI hosts.
    let host = host.split(':').next().unwrap_or(host).to_ascii_lowercase();
    host == "x.ai" || host.ends_with(".x.ai")
}

#[derive(Debug, Deserialize)]
struct GrokDeviceStart {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    #[serde(default, deserialize_with = "de_opt_u64")]
    expires_in: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64")]
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct GrokPollError {
    error: String,
}

/// RFC 8628 device-code login for headless / remote machines.
///
/// Prints a verification URL + user code; the user completes sign-in on any
/// browser (laptop/phone). llmtrim polls `auth.x.ai` until approved — no
/// localhost callback required. Same public client id as Grok Build / OpenClaw /
/// pi-grok-cli (consent UI may still say "Grok Build").
pub fn grok_device() -> Result<()> {
    print!("{TOS_WARNING}");

    let discovery = grok_discover()?;
    let device_endpoint = discovery
        .device_authorization_endpoint
        .as_deref()
        .ok_or_else(|| {
            anyhow!(
                "Grok OIDC discovery has no device_authorization_endpoint — \
                 use `llmtrim sub auth grok login` instead"
            )
        })?;

    let (status, body) = post_form(
        device_endpoint,
        &[],
        &[
            ("client_id", GROK_CLIENT_ID.to_string()),
            ("scope", GROK_SCOPES.to_string()),
        ],
    )?;
    if !(200..300).contains(&status) {
        bail!("Grok device authorization failed (HTTP {status}): {body}");
    }
    let start: GrokDeviceStart = serde_json::from_str(&body)
        .context("failed to parse Grok device-authorization response")?;

    let verify_url = start
        .verification_uri_complete
        .as_deref()
        .filter(|u| !u.is_empty())
        .unwrap_or(start.verification_uri.as_str());
    if !is_trusted_xai_https_url(verify_url) {
        bail!("Grok device verification URI is outside a trusted xAI host: {verify_url}");
    }
    // Base verification_uri is also user-visible if complete URI is used only for open.
    if !is_trusted_xai_https_url(&start.verification_uri) {
        bail!(
            "Grok device verification_uri is outside a trusted xAI host: {}",
            start.verification_uri
        );
    }

    println!(
        "To authorize llmtrim on any device, open:\n\n  {verify_url}\n\n\
         and confirm the code: {}\n\n\
         Waiting for approval (up to 5 minutes)...\n",
        start.user_code
    );
    try_open_browser(verify_url);

    let mut interval = start.interval.unwrap_or(5).max(1);
    let deadline = std::time::Instant::now()
        + start
            .expires_in
            .map(|s| Duration::from_secs(s.min(FLOW_TIMEOUT.as_secs())))
            .unwrap_or(FLOW_TIMEOUT);

    // Poll immediately, then back off (same shape as OpenClaw / RFC 8628 clients).
    let tr: TokenResponse = loop {
        if std::time::Instant::now() >= deadline {
            bail!("timed out waiting for Grok device authorization");
        }
        let (status, body) = post_form(
            &discovery.token_endpoint,
            &[],
            &[
                ("client_id", GROK_CLIENT_ID.to_string()),
                ("device_code", start.device_code.clone()),
                (
                    "grant_type",
                    "urn:ietf:params:oauth:grant-type:device_code".to_string(),
                ),
            ],
        )?;
        if (200..300).contains(&status) {
            break serde_json::from_str(&body).context("failed to parse Grok token response")?;
        }
        match serde_json::from_str::<GrokPollError>(&body) {
            Ok(e) => match e.error.as_str() {
                "authorization_pending" => {}
                "slow_down" => interval = interval.saturating_add(5),
                "expired_token" => bail!("Grok device code expired; run login again"),
                "access_denied" | "authorization_denied" => {
                    bail!("Grok device authorization was denied")
                }
                other => bail!("Grok authorization failed: {other}"),
            },
            Err(_) => bail!("Grok device poll failed (HTTP {status}): {body}"),
        }
        std::thread::sleep(
            Duration::from_secs(interval).saturating_add(Duration::from_millis(500)),
        );
    };

    let stored = grok_stored_from(tr, true, None)?;
    write_grok(&stored)?;
    println!("Grok login complete.");
    Ok(())
}

pub fn grok_status() -> Result<()> {
    let stored = read_grok().context("no Grok credentials stored")?;
    println!("Grok: logged in");
    println!("  {}", expiry_line(stored.expires));
    Ok(())
}

pub fn grok_logout() -> Result<()> {
    delete_if_exists(&grok_auth_path()?)?;
    println!("Grok credentials deleted.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Small shared bits
// ---------------------------------------------------------------------------

fn expiry_line(expires_ms: u64) -> String {
    let now = now_ms();
    if expires_ms <= now {
        "token: expired (will refresh on next use)".to_string()
    } else {
        let secs = (expires_ms - now) / 1000;
        format!("token: valid for ~{}m {}s", secs / 60, secs % 60)
    }
}

/// Best-effort: try to launch the platform browser opener. Silent on failure — the URL is always
/// printed, so the flow works headless too. No new dependency: shells out to the OS opener.
fn try_open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let cmd = ("open", vec![url]);
    #[cfg(target_os = "windows")]
    let cmd = ("cmd", vec!["/C", "start", "", url]);
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = ("xdg-open", vec![url]);

    let _ = std::process::Command::new(cmd.0)
        .args(cmd.1)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Minimal percent-encoding for the pieces we place in a URL (the redirect_uri). Encodes
/// everything outside the RFC 3986 unreserved set.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Decode percent-encoded query values (and `+` as space) from the callback.
fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// Tests (pure, offline only — no network)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64url_roundtrip_no_pad() {
        let data = b"\x00\x01\x02hello world \xff\xfe";
        let enc = b64url_encode(data);
        assert!(!enc.contains('='), "must be no-pad: {enc}");
        assert!(
            !enc.contains('+') && !enc.contains('/'),
            "must be url-safe: {enc}"
        );
        assert_eq!(b64url_decode(&enc).unwrap(), data);
    }

    #[test]
    fn pkce_challenge_is_sha256_b64url() {
        // Known vector: verifier "abc" -> base64url-nopad(SHA256("abc")).
        let verifier = "abc";
        let expected = {
            let d = Sha256::digest(verifier.as_bytes());
            URL_SAFE_NO_PAD.encode(d)
        };
        assert_eq!(pkce_challenge(verifier), expected);
        // And it decodes back to a 32-byte digest.
        assert_eq!(b64url_decode(&pkce_challenge(verifier)).unwrap().len(), 32);
    }

    fn unsigned_jwt(payload: serde_json::Value) -> String {
        let header = b64url_encode(br#"{"alg":"none"}"#);
        let payload = b64url_encode(payload.to_string().as_bytes());
        format!("{header}.{payload}.sig")
    }

    #[test]
    fn account_id_from_top_level_claim() {
        let jwt = unsigned_jwt(serde_json::json!({ "chatgpt_account_id": "acct_123" }));
        assert_eq!(account_id_from_jwt(&jwt), Some("acct_123".to_string()));
    }

    #[test]
    fn account_id_from_namespaced_claim() {
        let jwt = unsigned_jwt(serde_json::json!({
            "https://api.openai.com/auth": { "chatgpt_account_id": "acct_ns" }
        }));
        assert_eq!(account_id_from_jwt(&jwt), Some("acct_ns".to_string()));
    }

    #[test]
    fn account_id_from_organizations_fallback() {
        let jwt = unsigned_jwt(serde_json::json!({
            "organizations": [{ "id": "org_first" }, { "id": "org_second" }]
        }));
        assert_eq!(account_id_from_jwt(&jwt), Some("org_first".to_string()));
    }

    #[test]
    fn account_id_precedence_top_level_wins() {
        let jwt = unsigned_jwt(serde_json::json!({
            "chatgpt_account_id": "top",
            "https://api.openai.com/auth": { "chatgpt_account_id": "ns" },
            "organizations": [{ "id": "org" }],
        }));
        assert_eq!(account_id_from_jwt(&jwt), Some("top".to_string()));
    }

    #[test]
    fn account_id_none_when_absent() {
        let jwt = unsigned_jwt(serde_json::json!({ "sub": "user" }));
        assert_eq!(account_id_from_jwt(&jwt), None);
        assert_eq!(account_id_from_jwt("garbage"), None);
    }

    #[test]
    fn kimi_user_id_claim() {
        let jwt = unsigned_jwt(serde_json::json!({ "user_id": "u-42" }));
        assert_eq!(jwt_claim_str(&jwt, "user_id"), Some("u-42".to_string()));
        assert_eq!(jwt_claim_str(&jwt, "missing"), None);
    }

    #[test]
    fn codex_stored_serde_uses_camelcase_key() {
        let a = CodexStored {
            access: "a".into(),
            refresh: "r".into(),
            expires: 123,
            account_id: Some("acct".into()),
        };
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("\"accountId\":\"acct\""), "got: {json}");
        assert!(!json.contains("account_id"));
        let back: CodexStored = serde_json::from_str(&json).unwrap();
        assert_eq!(back.account_id.as_deref(), Some("acct"));
    }

    #[test]
    fn codex_stored_reads_legacy_snake_case_key() {
        let legacy = r#"{"access":"a","refresh":"r","expires":9,"account_id":"legacy"}"#;
        let parsed: CodexStored = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.account_id.as_deref(), Some("legacy"));
    }

    #[test]
    fn codex_stored_account_id_optional() {
        let json = r#"{"access":"a","refresh":"r","expires":9}"#;
        let parsed: CodexStored = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.account_id, None);
    }

    #[test]
    fn kimi_stored_serde_roundtrip() {
        let a = KimiStored {
            access: "a".into(),
            refresh: "r".into(),
            expires: 456,
            scope: Some("coding".into()),
            user_id: Some("u1".into()),
        };
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("\"userId\":\"u1\""), "got: {json}");
        let back: KimiStored = serde_json::from_str(&json).unwrap();
        assert_eq!(back.expires, 456);
        assert_eq!(back.scope.as_deref(), Some("coding"));
        assert_eq!(back.user_id.as_deref(), Some("u1"));
    }

    #[test]
    fn kimi_stored_optional_fields() {
        let json = r#"{"access":"a","refresh":"r","expires":1}"#;
        let parsed: KimiStored = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.scope, None);
        assert_eq!(parsed.user_id, None);
    }

    #[test]
    fn needs_refresh_margin() {
        let now = 1_000_000_000_000u64;
        // Expires exactly at the 5-minute margin -> refresh.
        assert!(needs_refresh(now + REFRESH_MARGIN_MS, now));
        // Expires just inside the margin -> refresh.
        assert!(needs_refresh(now + REFRESH_MARGIN_MS - 1, now));
        // Already expired -> refresh.
        assert!(needs_refresh(now - 1, now));
        // Comfortably beyond the margin -> no refresh.
        assert!(!needs_refresh(now + REFRESH_MARGIN_MS + 1, now));
    }

    #[test]
    fn device_start_accepts_string_or_number_interval() {
        // Some OAuth device endpoints send `interval` as a JSON string, which a strict
        // `Option<u64>` rejected — aborting the whole device flow.
        let as_string = r#"{"device_auth_id":"d","user_code":"AB-12","interval":"5"}"#;
        let s: CodexDeviceStart = serde_json::from_str(as_string).unwrap();
        assert_eq!(s.interval, Some(5));
        let as_number = r#"{"device_auth_id":"d","user_code":"AB-12","interval":5}"#;
        let n: CodexDeviceStart = serde_json::from_str(as_number).unwrap();
        assert_eq!(n.interval, Some(5));
        let missing = r#"{"device_auth_id":"d","user_code":"AB-12"}"#;
        let m: CodexDeviceStart = serde_json::from_str(missing).unwrap();
        assert_eq!(m.interval, None);
        // A non-numeric string degrades to None (the caller's default applies), not an error.
        let junk = r#"{"device_auth_id":"d","user_code":"AB-12","interval":"soon"}"#;
        let j: CodexDeviceStart = serde_json::from_str(junk).unwrap();
        assert_eq!(j.interval, None);
    }

    #[test]
    fn token_response_accepts_string_expires_in() {
        let r = r#"{"access_token":"a","expires_in":"3600"}"#;
        let t: TokenResponse = serde_json::from_str(r).unwrap();
        assert_eq!(t.expires_in, Some(3600));
    }

    #[test]
    fn device_id_is_32_hex_no_dashes() {
        let id = new_device_id();
        assert_eq!(id.len(), 32, "uuid v4 simple form is 32 chars: {id}");
        assert!(!id.contains('-'));
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "must be hex: {id}"
        );
        // Distinct calls yield distinct ids.
        assert_ne!(new_device_id(), new_device_id());
    }

    #[test]
    fn parse_callback_extracts_code_and_state() {
        let (code, state) =
            parse_callback("/auth/callback?code=abc123&state=xyz").expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz");
    }

    #[test]
    fn parse_callback_rejects_other_paths_and_missing_code() {
        assert_eq!(parse_callback("/favicon.ico?x=1"), None);
        assert_eq!(parse_callback("/auth/callback?state=only"), None);
    }

    #[test]
    fn parse_callback_urldecodes_values() {
        let (code, _) = parse_callback("/auth/callback?code=a%2Bb%3Dc&state=s").unwrap();
        assert_eq!(code, "a+b=c");
    }

    #[test]
    fn urlencode_encodes_redirect_uri() {
        assert_eq!(
            urlencode("http://localhost:1455/auth/callback"),
            "http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback"
        );
    }

    #[test]
    fn first_request_target_parses_get_line() {
        let req = "GET /auth/callback?code=1 HTTP/1.1\r\nHost: x\r\n\r\n";
        assert_eq!(first_request_target(req), Some("/auth/callback?code=1"));
    }

    #[test]
    fn grok_callback_path_parses() {
        let (code, state) =
            parse_grok_callback("/callback?code=tok&state=st").expect("should parse");
        assert_eq!(code, "tok");
        assert_eq!(state, "st");
        assert!(parse_grok_callback("/auth/callback?code=tok&state=st").is_none());
    }

    #[test]
    fn grok_manual_paste_accepts_url_query_and_bare_code() {
        let state = "expected-state";
        let code = "a".repeat(40);

        let url = format!("http://127.0.0.1:4242/callback?code={code}&state={state}");
        assert_eq!(parse_grok_manual_paste(&url, state).unwrap(), code);

        let query = format!("code={code}&state={state}");
        assert_eq!(parse_grok_manual_paste(&query, state).unwrap(), code);

        assert_eq!(parse_grok_manual_paste(&code, state).unwrap(), code);

        // Short device-style codes are not authorization codes.
        assert!(parse_grok_manual_paste("ABCD-12", state).is_err());

        // State mismatch on URL paste is rejected.
        let bad = format!("http://127.0.0.1:1/callback?code={code}&state=other");
        assert!(parse_grok_manual_paste(&bad, state).is_err());
    }

    #[test]
    fn grok_device_start_accepts_string_or_number_fields() {
        let raw = r#"{
            "device_code": "dc",
            "user_code": "WDJB-MJHT",
            "verification_uri": "https://auth.x.ai/device",
            "verification_uri_complete": "https://auth.x.ai/device?user_code=WDJB-MJHT",
            "expires_in": "600",
            "interval": "5"
        }"#;
        let s: GrokDeviceStart = serde_json::from_str(raw).unwrap();
        assert_eq!(s.user_code, "WDJB-MJHT");
        assert_eq!(s.expires_in, Some(600));
        assert_eq!(s.interval, Some(5));
        assert!(s.verification_uri_complete.unwrap().contains("user_code"));
    }

    #[test]
    fn trusted_xai_https_url_allows_issuer_hosts_only() {
        assert!(is_trusted_xai_https_url("https://auth.x.ai/oauth2/device"));
        assert!(is_trusted_xai_https_url(
            "https://accounts.x.ai/device?user_code=X"
        ));
        assert!(is_trusted_xai_https_url("https://x.ai/device"));
        assert!(!is_trusted_xai_https_url("http://auth.x.ai/device"));
        assert!(!is_trusted_xai_https_url("https://evil.com/"));
        assert!(!is_trusted_xai_https_url("https://evil.x.ai.attacker.com/"));
        assert!(!is_trusted_xai_https_url("https://notx.ai/"));
    }
}
