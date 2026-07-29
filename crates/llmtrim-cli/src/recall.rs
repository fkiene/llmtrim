//! Ephemeral recovery storage for future lossy first-arrival tool-output shaping.
//!
//! Entries live only in the proxy process. The local control endpoint uses a bounded,
//! length-delimited protocol and is deliberately separate from HTTPS proxy traffic.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "intercept")]
use anyhow::Context;
use anyhow::{Result, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;

pub const DEFAULT_TTL: Duration = Duration::from_secs(18_000);
pub const DEFAULT_MAX_ENTRIES: usize = 256;
pub const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;
/// Hard upper bound for a single control-protocol response, even if configuration is corrupt.
pub const MAX_PROTOCOL_RESPONSE_BYTES: usize = DEFAULT_MAX_ENTRY_BYTES;
#[cfg(feature = "intercept")]
const MAX_HANDLE_BYTES: usize = 45;
#[cfg(feature = "intercept")]
const MAX_HEADER_BYTES: usize = 64;
#[cfg(feature = "intercept")]
const MAX_ERROR_BYTES: usize = 1024;
#[cfg(feature = "intercept")]
const IO_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub ttl: Duration,
    pub max_entries: usize,
    pub max_bytes: usize,
    pub max_entry_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            ttl: DEFAULT_TTL,
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
            max_entry_bytes: DEFAULT_MAX_ENTRY_BYTES,
        }
    }
}

#[derive(Debug)]
struct Entry {
    bytes: Vec<u8>,
    expires_at: Instant,
    sequence: u64,
    /// Present only for transactional first-arrival admissions. Ordinary IPC/store inserts have
    /// no rerun fingerprint.
    fingerprint: Option<[u8; 32]>,
}
#[derive(Debug)]
struct Inner {
    entries: HashMap<String, Entry>,
    bytes: usize,
    next_sequence: u64,
    /// Hashed `(session, tool identity, content)` keys. These intentionally outlive raw entries
    /// for one TTL so an expired/restarted recovery cannot create a compression loop.
    fingerprints: HashMap<[u8; 32], Instant>,
    fingerprint_secret: [u8; 32],
}

/// Admission distinguishes a retrievable new entry from a known rerun that must pass through.
pub enum Admission {
    Stored(String),
    Seen,
}

/// Bounded, process-local raw-byte store. Handles are opaque capabilities rather than paths.
#[derive(Debug)]
pub struct RecallStore {
    limits: Limits,
    inner: Mutex<Inner>,
}

impl RecallStore {
    pub fn new(limits: Limits) -> Self {
        let mut fingerprint_secret = [0_u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut fingerprint_secret);
        Self {
            limits,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                bytes: 0,
                next_sequence: 0,
                fingerprints: HashMap::new(),
                fingerprint_secret,
            }),
        }
    }

    /// Transactionally admits bytes for one session-scoped tool result. Only a hash is retained
    /// after the raw entry expires; the bounded map is capped at `max_entries`. The fingerprint
    /// deliberately excludes the JSON pointer/tool-use id: a recall arrives in a new turn at a new
    /// pointer, but must still be recognized as the same raw result and pass through in full.
    pub fn admit(&self, session: &str, bytes: Vec<u8>) -> Result<Admission> {
        if bytes.len() > self.limits.max_entry_bytes || bytes.len() > self.limits.max_bytes {
            bail!("recall entry exceeds the configured size limit")
        }
        let now = Instant::now();
        let mut inner = self.inner.lock().expect("recall store lock poisoned");
        purge_expired(&mut inner, now);
        let key = fingerprint(&inner.fingerprint_secret, session, &bytes);
        if inner.fingerprints.contains_key(&key) {
            return Ok(Admission::Seen);
        }
        while inner.entries.len() >= self.limits.max_entries
            || inner.bytes.saturating_add(bytes.len()) > self.limits.max_bytes
        {
            let Some(oldest) = oldest_handle(&inner) else {
                break;
            };
            let _ = remove(&mut inner, &oldest);
        }
        if inner.entries.len() >= self.limits.max_entries
            || inner.bytes.saturating_add(bytes.len()) > self.limits.max_bytes
        {
            bail!("recall store has no capacity for entry")
        }
        if inner.fingerprints.len() >= self.limits.max_entries
            && let Some(oldest) = inner
                .fingerprints
                .iter()
                .min_by_key(|(_, when)| *when)
                .map(|(k, _)| *k)
        {
            inner.fingerprints.remove(&oldest);
        }
        let handle = new_handle(&inner);
        let sequence = inner.next_sequence;
        inner.next_sequence = inner.next_sequence.wrapping_add(1);
        inner.bytes += bytes.len();
        inner.entries.insert(
            handle.clone(),
            Entry {
                bytes,
                expires_at: now + self.limits.ttl,
                sequence,
                fingerprint: Some(key),
            },
        );
        inner
            .fingerprints
            .insert(key, now + self.limits.ttl.saturating_mul(2));
        Ok(Admission::Stored(handle))
    }

    /// Stores exact bytes and returns an unguessable, URL-safe opaque handle.
    pub fn insert(&self, bytes: Vec<u8>) -> Result<String> {
        if bytes.len() > self.limits.max_entry_bytes || bytes.len() > self.limits.max_bytes {
            bail!("recall entry exceeds the configured size limit")
        }
        let now = Instant::now();
        let mut inner = self.inner.lock().expect("recall store lock poisoned");
        purge_expired(&mut inner, now);
        while inner.entries.len() >= self.limits.max_entries
            || inner.bytes.saturating_add(bytes.len()) > self.limits.max_bytes
        {
            let Some(oldest) = oldest_handle(&inner) else {
                break;
            };
            let _ = remove(&mut inner, &oldest);
        }
        if inner.entries.len() >= self.limits.max_entries
            || inner.bytes.saturating_add(bytes.len()) > self.limits.max_bytes
        {
            bail!("recall store has no capacity for entry")
        }
        let handle = new_handle(&inner);
        let sequence = inner.next_sequence;
        inner.next_sequence = inner.next_sequence.wrapping_add(1);
        inner.bytes += bytes.len();
        inner.entries.insert(
            handle.clone(),
            Entry {
                bytes,
                expires_at: now + self.limits.ttl,
                sequence,
                fingerprint: None,
            },
        );
        Ok(handle)
    }

    /// Rolls back an uncommitted admission after a later gate rejects its marker. The rerun
    /// fingerprint must go too: the agent never saw this handle, so a later identical result is
    /// still a legitimate first arrival.
    pub fn rollback(&self, handle: &str) {
        let mut inner = self.inner.lock().expect("recall store lock poisoned");
        if let Some(entry) = remove(&mut inner, handle)
            && let Some(fingerprint) = entry.fingerprint
        {
            inner.fingerprints.remove(&fingerprint);
        }
    }

    /// Retrieves a copy. Entries are multi-use until their fixed expiry.
    pub fn get(&self, handle: &str) -> Result<Vec<u8>> {
        if !valid_handle(handle) {
            bail!("malformed recall handle")
        }
        let mut inner = self.inner.lock().expect("recall store lock poisoned");
        purge_expired(&mut inner, Instant::now());
        inner
            .entries
            .get(handle)
            .map(|entry| entry.bytes.clone())
            .ok_or_else(|| anyhow::anyhow!("unknown or expired recall handle"))
    }
}

fn new_handle(inner: &Inner) -> String {
    loop {
        let mut raw = [0_u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut raw);
        let handle = format!("r_{}", URL_SAFE_NO_PAD.encode(raw));
        if !inner.entries.contains_key(&handle) {
            return handle;
        }
    }
}
fn valid_handle(handle: &str) -> bool {
    let Some(encoded) = handle.strip_prefix("r_") else {
        return false;
    };
    encoded.len() == 43
        && encoded
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
fn fingerprint(secret: &[u8; 32], session: &str, bytes: &[u8]) -> [u8; 32] {
    // Domain separation and length prefixes prevent tuple ambiguity; the random process secret
    // keeps session-derived keys unlinkable outside this daemon lifetime.
    let mut h = Sha256::new();
    h.update(secret);
    for part in [session.as_bytes(), bytes] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().into()
}
fn purge_expired(inner: &mut Inner, now: Instant) {
    inner.fingerprints.retain(|_, expires| *expires > now);
    let expired: Vec<String> = inner
        .entries
        .iter()
        .filter(|(_, e)| e.expires_at <= now)
        .map(|(h, _)| h.clone())
        .collect();
    for handle in expired {
        let _ = remove(inner, &handle);
    }
}
fn oldest_handle(inner: &Inner) -> Option<String> {
    inner
        .entries
        .iter()
        .min_by_key(|(_, e)| e.sequence)
        .map(|(h, _)| h.clone())
}
fn remove(inner: &mut Inner, handle: &str) -> Option<Entry> {
    let entry = inner.entries.remove(handle)?;
    inner.bytes -= entry.bytes.len();
    Some(entry)
}

pub type SharedStore = Arc<RecallStore>;
pub fn from_runtime(config: &llmtrim_core::config::RuntimeConfig) -> SharedStore {
    Arc::new(RecallStore::new(Limits {
        ttl: Duration::from_secs(
            config
                .first_arrival_recall_ttl_secs
                .unwrap_or(DEFAULT_TTL.as_secs()),
        ),
        max_entries: config
            .first_arrival_recall_max_entries
            .unwrap_or(DEFAULT_MAX_ENTRIES),
        max_bytes: config
            .first_arrival_recall_max_bytes
            .unwrap_or(DEFAULT_MAX_BYTES),
        max_entry_bytes: config
            .first_arrival_recall_max_entry_bytes
            .unwrap_or(DEFAULT_MAX_ENTRY_BYTES)
            .min(MAX_PROTOCOL_RESPONSE_BYTES),
    }))
}

/// `GET <length>\n<handle>` requests and `OK|ERR <length>\n<payload>` replies are binary-safe.
/// The declared length is always checked before allocation or reading payload bytes.
#[cfg(feature = "intercept")]
mod wire {
    use super::*;
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    pub async fn read_header<R: AsyncRead + Unpin>(stream: &mut R) -> Result<(String, usize)> {
        let mut header = Vec::new();
        loop {
            let mut byte = [0_u8; 1];
            if stream.read_exact(&mut byte).await.is_err() {
                bail!("truncated recall frame")
            }
            if byte[0] == b'\n' {
                break;
            }
            if header.len() == MAX_HEADER_BYTES {
                bail!("overlong recall frame header")
            }
            header.push(byte[0]);
        }
        let header = std::str::from_utf8(&header).context("non-UTF-8 recall frame header")?;
        let Some((status, len)) = header.split_once(' ') else {
            bail!("malformed recall frame header")
        };
        if status.is_empty() || len.is_empty() || len.contains(' ') {
            bail!("malformed recall frame header")
        }
        let len = len
            .parse::<usize>()
            .context("invalid recall frame length")?;
        Ok((status.to_owned(), len))
    }

    pub async fn read_exact_bounded<R: AsyncRead + Unpin>(
        stream: &mut R,
        len: usize,
        max: usize,
    ) -> Result<Vec<u8>> {
        if len > max {
            bail!("recall frame exceeds size limit")
        }
        let mut body = vec![0; len];
        stream
            .read_exact(&mut body)
            .await
            .context("truncated recall frame payload")?;
        Ok(body)
    }

    pub async fn write_frame<W: AsyncWrite + Unpin>(
        stream: &mut W,
        status: &str,
        body: &[u8],
        max: usize,
    ) -> Result<()> {
        if body.len() > max {
            bail!("recall frame exceeds size limit")
        }
        stream
            .write_all(format!("{status} {}\n", body.len()).as_bytes())
            .await?;
        stream.write_all(body).await?;
        stream.flush().await?;
        Ok(())
    }

    pub async fn serve_connection<S: AsyncRead + AsyncWrite + Unpin>(
        mut stream: S,
        store: SharedStore,
        max_response: usize,
    ) {
        let result = tokio::time::timeout(IO_TIMEOUT, async {
            let (verb, len) = read_header(&mut stream).await?;
            if verb != "GET" {
                bail!("malformed recall request")
            }
            let request = read_exact_bounded(&mut stream, len, MAX_HANDLE_BYTES).await?;
            let handle = std::str::from_utf8(&request).context("malformed recall handle")?;
            store.get(handle)
        })
        .await;
        match result {
            Ok(Ok(bytes)) => {
                let _ = tokio::time::timeout(
                    IO_TIMEOUT,
                    write_frame(&mut stream, "OK", &bytes, max_response),
                )
                .await;
            }
            Ok(Err(error)) => {
                let message = error.to_string();
                let message = &message.as_bytes()[..message.len().min(MAX_ERROR_BYTES)];
                let _ = tokio::time::timeout(
                    IO_TIMEOUT,
                    write_frame(&mut stream, "ERR", message, MAX_ERROR_BYTES),
                )
                .await;
            }
            Err(_) => {
                let _ = tokio::time::timeout(
                    IO_TIMEOUT,
                    write_frame(
                        &mut stream,
                        "ERR",
                        b"recall request timed out",
                        MAX_ERROR_BYTES,
                    ),
                )
                .await;
            }
        }
    }

    pub async fn request_over<S: AsyncRead + AsyncWrite + Unpin>(
        mut stream: S,
        handle: &str,
        max_response: usize,
    ) -> Result<Vec<u8>> {
        tokio::time::timeout(IO_TIMEOUT, async {
            write_frame(&mut stream, "GET", handle.as_bytes(), MAX_HANDLE_BYTES).await?;
            let (status, len) = read_header(&mut stream).await?;
            let max = if status == "ERR" {
                MAX_ERROR_BYTES
            } else {
                max_response
            };
            let body = read_exact_bounded(&mut stream, len, max).await?;
            match status.as_str() {
                "OK" => Ok(body),
                "ERR" if len <= MAX_ERROR_BYTES => bail!("{}", String::from_utf8_lossy(&body)),
                _ => bail!("malformed recall response"),
            }
        })
        .await
        .context("recall request timed out")?
    }
}

#[cfg(all(unix, feature = "intercept"))]
fn socket_path() -> Result<std::path::PathBuf> {
    Ok(crate::daemon::home_dir()?.join("recall.sock"))
}

#[cfg(feature = "intercept")]
fn client_max_response() -> usize {
    llmtrim_core::config::RuntimeConfig::get()
        .first_arrival_recall_max_entry_bytes
        .unwrap_or(DEFAULT_MAX_ENTRY_BYTES)
        .min(MAX_PROTOCOL_RESPONSE_BYTES)
}

#[cfg(all(unix, feature = "intercept"))]
pub fn request(handle: &str) -> Result<Vec<u8>> {
    if !valid_handle(handle) {
        bail!("malformed recall handle")
    }
    let path = socket_path()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()?;
    runtime.block_on(async {
        let stream = tokio::time::timeout(IO_TIMEOUT, tokio::net::UnixStream::connect(&path))
            .await
            .context("recall connect timed out")?
            .with_context(|| format!("recall daemon unavailable at {}", path.display()))?;
        wire::request_over(stream, handle, client_max_response()).await
    })
}

#[cfg(all(unix, not(feature = "intercept")))]
pub fn request(_handle: &str) -> Result<Vec<u8>> {
    bail!("recall requires the interceptor build feature")
}

#[cfg(windows)]
fn pipe_name() -> Result<String> {
    let user = std::env::var("USERNAME").or_else(|_| std::env::var("USER"))?;
    Ok(format!(r"\\.\pipe\llmtrim-recall-{user}"))
}

#[cfg(all(windows, feature = "intercept"))]
pub fn request(handle: &str) -> Result<Vec<u8>> {
    use tokio::net::windows::named_pipe::ClientOptions;
    if !valid_handle(handle) {
        bail!("malformed recall handle")
    }
    let name = pipe_name()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()?;
    runtime.block_on(async {
        let stream = tokio::time::timeout(IO_TIMEOUT, async {
            loop {
                match ClientOptions::new().open(&name) {
                    Ok(pipe) => return Ok(pipe),
                    Err(error)
                        if error.kind() == std::io::ErrorKind::NotFound
                            || error.raw_os_error()
                                == Some(windows_sys::Win32::Foundation::ERROR_PIPE_BUSY as i32) =>
                    {
                        tokio::time::sleep(Duration::from_millis(20)).await
                    }
                    Err(error) => return Err(error),
                }
            }
        })
        .await
        .context("recall connect timed out")??;
        wire::request_over(stream, handle, client_max_response()).await
    })
}

#[cfg(all(windows, not(feature = "intercept")))]
pub fn request(_handle: &str) -> Result<Vec<u8>> {
    bail!("recall requires the interceptor build feature")
}

#[cfg(not(any(unix, windows)))]
pub fn request(_handle: &str) -> Result<Vec<u8>> {
    bail!("recall is unsupported on this platform")
}

#[cfg(all(unix, feature = "intercept"))]
struct SocketOwner(std::path::PathBuf);
#[cfg(all(unix, feature = "intercept"))]
impl Drop for SocketOwner {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(all(unix, feature = "intercept"))]
fn ensure_private_socket_parent(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let parent = path
        .parent()
        .context("recall socket has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let metadata = std::fs::symlink_metadata(parent)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != nix::unistd::geteuid().as_raw()
    {
        bail!("recall socket parent is not a private directory owned by this user")
    }
    if metadata.mode() & 0o077 != 0 {
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
    }
    if std::fs::symlink_metadata(parent)?.mode() & 0o077 != 0 {
        bail!("recall socket parent is accessible by other users")
    }
    Ok(())
}

#[cfg(all(unix, feature = "intercept"))]
pub async fn serve(store: SharedStore) -> Result<()> {
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};
    let path = socket_path()?;
    ensure_private_socket_parent(&path)?;
    if let Ok(metadata) = std::fs::symlink_metadata(&path) {
        if !metadata.file_type().is_socket() {
            bail!("refusing to replace non-socket recall endpoint")
        }
        match tokio::time::timeout(IO_TIMEOUT, tokio::net::UnixStream::connect(&path)).await {
            Ok(Ok(_)) | Err(_) => bail!("recall endpoint is already owned by a running daemon"),
            Ok(Err(error)) if error.kind() == std::io::ErrorKind::ConnectionRefused => {
                std::fs::remove_file(&path)
                    .with_context(|| format!("failed to remove stale {}", path.display()))?
            }
            Ok(Err(error)) => return Err(error).context("failed to probe recall endpoint"),
        }
    }
    let listener = tokio::net::UnixListener::bind(&path)
        .with_context(|| format!("failed to bind {}", path.display()))?;
    let owner = SocketOwner(path.clone());
    if let Err(error) = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)) {
        drop(owner);
        return Err(error.into());
    }
    let max_response = store.limits.max_entry_bytes;
    tokio::spawn(async move {
        let _owner = owner;
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(wire::serve_connection(stream, store.clone(), max_response));
        }
    });
    Ok(())
}

#[cfg(all(windows, feature = "intercept"))]
pub async fn serve(store: SharedStore) -> Result<()> {
    use tokio::net::windows::named_pipe::ServerOptions;
    let name = pipe_name()?;
    let max_response = store.limits.max_entry_bytes;
    tokio::spawn(async move {
        // Tokio's default security descriptor uses the creating user's default DACL; reject
        // remote clients and FIRST_PIPE_INSTANCE prevent network peers and pipe-name takeover.
        let mut first = true;
        loop {
            let pipe = match ServerOptions::new()
                .first_pipe_instance(first)
                .reject_remote_clients(true)
                .create(&name)
            {
                Ok(pipe) => pipe,
                Err(_) => break,
            };
            first = false;
            if pipe.connect().await.is_err() {
                continue;
            }
            tokio::spawn(wire::serve_connection(pipe, store.clone(), max_response));
        }
    });
    Ok(())
}

#[cfg(not(all(any(unix, windows), feature = "intercept")))]
pub async fn serve(_store: SharedStore) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "intercept")]
    use tokio::io::AsyncWriteExt;
    #[test]
    fn exact_bytes_and_multi_use() {
        let s = RecallStore::new(Limits::default());
        let bytes = b"nul\0unicode: \xF0\x9F\xA6\x80".to_vec();
        let h = s.insert(bytes.clone()).unwrap();
        assert!(valid_handle(&h));
        assert_eq!(s.get(&h).unwrap(), bytes);
        assert_eq!(s.get(&h).unwrap(), bytes);
    }
    #[test]
    fn limits_evict_oldest_and_reject_oversized() {
        let s = RecallStore::new(Limits {
            ttl: Duration::from_secs(60),
            max_entries: 2,
            max_bytes: 4,
            max_entry_bytes: 3,
        });
        let a = s.insert(vec![1]).unwrap();
        let b = s.insert(vec![2]).unwrap();
        let _ = s.insert(vec![3, 4, 5]).unwrap();
        assert!(s.get(&a).is_err());
        assert!(s.get(&b).is_ok());
        assert!(s.insert(vec![0; 4]).is_err());
    }
    #[test]
    fn seen_session_scoped_rerun_is_not_readmitted() {
        let s = RecallStore::new(Limits::default());
        assert!(matches!(
            s.admit("s1", b"raw".to_vec()).unwrap(),
            Admission::Stored(_)
        ));
        assert!(matches!(
            s.admit("s1", b"raw".to_vec()).unwrap(),
            Admission::Seen
        ));
        assert!(matches!(
            s.admit("s2", b"raw".to_vec()).unwrap(),
            Admission::Stored(_)
        ));
    }
    #[test]
    fn rolled_back_admission_can_be_admitted_again() {
        let s = RecallStore::new(Limits::default());
        let Admission::Stored(handle) = s.admit("s1", b"raw".to_vec()).unwrap() else {
            panic!("first result must be stored");
        };
        s.rollback(&handle);
        assert!(s.get(&handle).is_err());
        assert!(matches!(
            s.admit("s1", b"raw".to_vec()).unwrap(),
            Admission::Stored(_)
        ));
    }

    #[test]
    fn malformed_and_unknown_fail() {
        let s = RecallStore::new(Limits::default());
        assert!(s.get("bad").is_err());
        assert!(s.get(&format!("r_{}", "A".repeat(43))).is_err());
    }
    #[test]
    fn immediate_expiry() {
        let s = RecallStore::new(Limits {
            ttl: Duration::ZERO,
            ..Limits::default()
        });
        let h = s.insert(vec![1]).unwrap();
        assert!(s.get(&h).is_err());
    }

    #[cfg(feature = "intercept")]
    #[tokio::test]
    async fn framed_response_preserves_err_prefix_nul_and_unicode() {
        let payload = b"ERR \0unicode: \xF0\x9F\xA6\x80".to_vec();
        let (mut client, mut server) = tokio::io::duplex(256);
        let handle = format!("r_{}", "A".repeat(43));
        let expected = payload.clone();
        let peer = tokio::spawn(async move {
            let (verb, len) = wire::read_header(&mut server).await.unwrap();
            assert_eq!(verb, "GET");
            let _ = wire::read_exact_bounded(&mut server, len, MAX_HANDLE_BYTES)
                .await
                .unwrap();
            wire::write_frame(&mut server, "OK", &expected, 128)
                .await
                .unwrap();
        });
        assert_eq!(
            wire::request_over(&mut client, &handle, 128).await.unwrap(),
            payload
        );
        peer.await.unwrap();
    }

    #[cfg(feature = "intercept")]
    #[tokio::test]
    async fn protocol_rejects_overlong_headers_and_payloads() {
        let (mut writer, mut reader) = tokio::io::duplex(256);
        let peer = tokio::spawn(async move {
            writer
                .write_all(&[b'x'; MAX_HEADER_BYTES + 1])
                .await
                .unwrap();
            writer.write_all(b"\n").await.unwrap();
        });
        assert!(wire::read_header(&mut reader).await.is_err());
        peer.await.unwrap();
        let (mut writer, _reader) = tokio::io::duplex(256);
        assert!(
            wire::write_frame(&mut writer, "OK", &[0; 129], 128)
                .await
                .is_err()
        );
    }

    #[cfg(all(unix, feature = "intercept"))]
    #[test]
    fn private_parent_is_created_and_restricted() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("llmtrim-recall-test-{}", std::process::id()));
        let path = dir.join("recall.sock");
        ensure_private_socket_parent(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&dir).unwrap().permissions().mode() & 0o077,
            0
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
