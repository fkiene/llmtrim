//! The loopback HTTP server Codex actually talks to.
//!
//! The parent module decides *what* the gateway does with a request; this file is the part
//! that speaks HTTP. It frames nothing by hand — hyper owns the local side of the wire and
//! ureq owns the upstream side — and it never buffers a reply: bytes reach Codex as the
//! upstream produces them, through a queue whose depth is fixed.
//!
//! The server is generic over the upstream seam, so the whole exchange — a real socket, a
//! real HTTP client, the product's real compression — is exercised without a network.

use std::collections::BTreeMap;
use std::io::Read;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use http_body_util::{BodyExt, LengthLimitError, Limited};
use hyper::body::{Body, Bytes, Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio::task::JoinSet;

use super::listener::{BindError, bind_address};
use super::upstream::{StreamingUpstream, UpstreamStream};
use super::{GatewayError, GatewayRequest, UpstreamRequest, prepare};

/// The port the gateway listens on unless the caller names another one. Picked high and
/// unassigned; nothing else in the product uses it.
pub const DEFAULT_PORT: u16 = 43117;

/// Ceiling on one local request body.
///
/// The gateway has to hold a whole body to compress it, so this is what stops a local client
/// from turning the loopback port into an out-of-memory switch. 64 MiB is not a new number:
/// it is the ceiling the rest of the product already uses wherever it must materialise a
/// payload (`ensure`, `recall`, the CLIProxyAPI download).
pub const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Connections served at once. Codex opens one; the bound exists so nothing can make the
/// gateway spawn tasks without limit.
pub const MAX_LOCAL_CLIENTS: usize = 32;

/// Response chunks that may sit between the upstream reader and the local writer.
///
/// This is the backpressure. The reader is a blocking thread using `blocking_send`, so when
/// Codex reads slower than the upstream sends, the thread parks and the upstream connection
/// stops being drained. Memory tracks this depth, never the length of the generation.
pub const RESPONSE_QUEUE_DEPTH: usize = 8;

/// Bytes taken from the upstream per relay step.
pub const RESPONSE_READ_SIZE: usize = 16 * 1024;

/// How long a local client may take to send its request headers before hyper drops it.
pub const LOCAL_HEADER_TIMEOUT: Duration = Duration::from_secs(60);

/// What the server bounds. Not a configuration surface: the command line cannot reach it,
/// [`ServerLimits::default`] is what production uses, and the only reason it is a value at all
/// is so a test can shrink a ceiling instead of shipping 64 MiB over a socket to prove it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerLimits {
    pub max_request_body_bytes: usize,
    pub max_connections: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_request_body_bytes: MAX_REQUEST_BODY_BYTES,
            max_connections: MAX_LOCAL_CLIENTS,
        }
    }
}

/// Status and relayed headers, handed back the moment the upstream answers — before its body
/// has finished arriving, which is what makes the relay incremental.
type UpstreamHead = Result<(u16, BTreeMap<String, String>), GatewayError>;

/// One piece of the upstream body, in the order it came off the wire.
type ResponseChunk = Result<Bytes, std::io::Error>;

/// Why the gateway could not start listening.
#[derive(Debug)]
pub enum ServeError {
    /// The requested host was not IPv4 loopback.
    Bind(BindError),
    /// The operating system refused the bind — almost always a port already in use.
    Listen(std::io::Error),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind(error) => write!(formatter, "{error}"),
            Self::Listen(error) => write!(formatter, "could not listen on 127.0.0.1: {error}"),
        }
    }
}

impl std::error::Error for ServeError {}

/// The body the gateway hands hyper: either nothing at all, or the bounded queue the upstream
/// reader feeds. There is no third case — the gateway never composes a body of its own.
pub struct GatewayBody {
    inner: BodyInner,
}

enum BodyInner {
    Empty,
    Streaming(mpsc::Receiver<ResponseChunk>),
}

impl GatewayBody {
    fn empty() -> Self {
        Self {
            inner: BodyInner::Empty,
        }
    }

    fn streaming(chunks: mpsc::Receiver<ResponseChunk>) -> Self {
        Self {
            inner: BodyInner::Streaming(chunks),
        }
    }
}

impl Body for GatewayBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match &mut self.get_mut().inner {
            BodyInner::Empty => Poll::Ready(None),
            BodyInner::Streaming(chunks) => match chunks.poll_recv(context) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(Frame::data(chunk)))),
                Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            },
        }
    }
}

/// A running gateway. Dropping this leaves the server running; [`GatewayServer::shutdown`] is
/// how it stops, and it does not return until the port is free.
pub struct GatewayServer {
    local_addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl GatewayServer {
    /// The address actually bound. With port 0 this is how the caller learns the real port.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Stop accepting, drop the listener, end the connections still open, release the port.
    pub async fn shutdown(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
    }
}

/// Start the gateway on loopback. Returns once the port is bound, so the caller can read the
/// real port before anything connects.
///
/// # Errors
///
/// [`ServeError::Bind`] for any host that is not IPv4 loopback, [`ServeError::Listen`] if the
/// port cannot be opened.
pub async fn serve<U: StreamingUpstream>(
    requested_host: Option<IpAddr>,
    port: u16,
    limits: ServerLimits,
    upstream: Arc<U>,
) -> Result<GatewayServer, ServeError> {
    let address = bind_address(requested_host, port).map_err(ServeError::Bind)?;
    let bound = TcpListener::bind(address).await;
    let listener = bound.map_err(ServeError::Listen)?;
    let local_addr = listener.local_addr().map_err(ServeError::Listen)?;
    let (stop, stopped) = oneshot::channel();
    let task = tokio::spawn(accept_loop(listener, limits, upstream, stopped));
    Ok(GatewayServer {
        local_addr,
        stop: Some(stop),
        task,
    })
}

async fn accept_loop<U: StreamingUpstream>(
    listener: TcpListener,
    limits: ServerLimits,
    upstream: Arc<U>,
    mut stopped: oneshot::Receiver<()>,
) {
    let permits = Arc::new(Semaphore::new(limits.max_connections));
    let mut connections = JoinSet::new();
    loop {
        // Taking the permit before accepting is what bounds the work: with every permit out,
        // the loop waits here instead of queuing connections it cannot serve.
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            break;
        };
        let accepted = tokio::select! {
            _ = &mut stopped => break,
            accepted = listener.accept() => accepted,
        };
        let (stream, _) = match accepted {
            Ok(accepted) => accepted,
            Err(_) => continue,
        };
        let upstream = Arc::clone(&upstream);
        connections.spawn(async move {
            let service = service_fn(move |request: Request<Incoming>| {
                serve_request(request, limits, Arc::clone(&upstream))
            });
            let io = TokioIo::new(stream);
            let mut builder = hyper::server::conn::http1::Builder::new();
            builder.timer(TokioTimer::new());
            builder.header_read_timeout(LOCAL_HEADER_TIMEOUT);
            let _ = builder.serve_connection(io, service).await;
            drop(permit);
        });
    }
    // Dropping the listener is what frees the port; the connection tasks are then ended so a
    // shutdown cannot outlive the handle that asked for it.
    drop(listener);
    connections.shutdown().await;
}

async fn serve_request<U: StreamingUpstream>(
    request: Request<Incoming>,
    limits: ServerLimits,
    upstream: Arc<U>,
) -> Result<Response<GatewayBody>, std::convert::Infallible> {
    let response = match relay(request, limits, upstream).await {
        Ok(response) => response,
        Err(error) => refuse(&error),
    };
    Ok(response)
}

/// Which refusal a failed body read is. Only an over-length body is a 413; anything else is a
/// request the gateway could not read at all, and neither answer says anything about content.
fn body_error(error: Box<dyn std::error::Error + Send + Sync>) -> GatewayError {
    if error.downcast_ref::<LengthLimitError>().is_some() {
        GatewayError::RequestTooLarge
    } else {
        GatewayError::MalformedRequest
    }
}

/// A refusal carries a status and nothing else: no body, no echoed header, no reason text.
/// An error path is exactly where a credential or a prompt fragment would leak if it could.
fn refuse(error: &GatewayError) -> Response<GatewayBody> {
    Response::builder()
        .status(error.status())
        .body(GatewayBody::empty())
        .expect("a status-only response is always well formed")
}

async fn relay<U: StreamingUpstream>(
    request: Request<Incoming>,
    limits: ServerLimits,
    upstream: Arc<U>,
) -> Result<Response<GatewayBody>, GatewayError> {
    let (parts, body) = request.into_parts();

    // A target in absolute or authority form carries a destination. This gateway has exactly
    // one, so anything but an origin-form path is simply not a route it serves.
    if parts.uri.authority().is_some() || parts.uri.scheme().is_some() {
        return Err(GatewayError::RouteNotFound);
    }

    let method = parts.method.as_str().to_string();
    let path = parts.uri.path().to_string();
    let mut headers = BTreeMap::new();
    for (name, value) in &parts.headers {
        if let Ok(value) = value.to_str() {
            headers
                .entry(name.as_str().to_ascii_lowercase())
                .or_insert_with(|| value.to_string());
        }
    }

    // The cheap gates first: a request that will be refused must not cost a body read, and
    // must never reach the upstream leg.
    super::validate_route(&method, &path)?;
    super::auth_gate(&headers)?;

    let collected = Limited::new(body, limits.max_request_body_bytes)
        .collect()
        .await
        .map_err(body_error)?
        .to_bytes();

    let local = GatewayRequest {
        method,
        path,
        headers,
        body: collected.to_vec(),
    };
    // Both entry points run the same policy. `prepare` re-checks the two gates above; that is
    // two comparisons, and one guarantee that the socket path cannot drift away from it.
    let (outgoing, metrics) = prepare(&local)?;

    let (head_sender, head_receiver) = oneshot::channel();
    let (chunk_sender, chunk_receiver) = mpsc::channel(RESPONSE_QUEUE_DEPTH);
    tokio::task::spawn_blocking(move || {
        pump(upstream.as_ref(), outgoing, head_sender, chunk_sender);
    });

    let head = head_receiver
        .await
        .map_err(|_| GatewayError::UpstreamFailed)?;
    let (status, upstream_headers) = head?;

    // Metadata only, and only once the exchange is under way: byte counts and a status. No
    // header, no body, nothing written down.
    println!(
        "codex-gateway: {} -> {} bytes upstream ({:.2}% smaller), status {status}",
        metrics.local_request_bytes,
        metrics.upstream_request_bytes,
        metrics.reduction_basis_points() as f64 / 100.0
    );

    let mut response = Response::builder().status(status);
    for (name, value) in &upstream_headers {
        response = response.header(name.as_str(), value.as_str());
    }
    response
        .body(GatewayBody::streaming(chunk_receiver))
        .map_err(|_| GatewayError::UpstreamFailed)
}

/// The upstream leg, start to finish, on one blocking thread.
///
/// Sending, reading the head and pumping the body all happen here, so the upstream reader
/// never crosses a thread boundary and `blocking_send` can park it when the local client is
/// the slower of the two.
fn pump<U: StreamingUpstream>(
    upstream: &U,
    outgoing: UpstreamRequest,
    head: oneshot::Sender<UpstreamHead>,
    chunks: mpsc::Sender<ResponseChunk>,
) {
    let stream = match upstream.send(outgoing) {
        Ok(stream) => stream,
        Err(error) => {
            let _ = head.send(Err(error));
            return;
        }
    };
    let UpstreamStream {
        status,
        headers,
        mut body,
    } = stream;
    if head.send(Ok((status, headers))).is_err() {
        return;
    }
    let mut buffer = vec![0_u8; RESPONSE_READ_SIZE];
    loop {
        match body.read(&mut buffer) {
            Ok(0) => return,
            Ok(read) => {
                let chunk = Bytes::copy_from_slice(&buffer[..read]);
                if chunks.blocking_send(Ok(chunk)).is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = chunks.blocking_send(Err(error));
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::io::Read;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::{
        DEFAULT_PORT, GatewayServer, MAX_LOCAL_CLIENTS, MAX_REQUEST_BODY_BYTES,
        RESPONSE_QUEUE_DEPTH, ServeError, ServerLimits, serve,
    };

    use crate::codex_gateway::listener::BindError;
    use crate::codex_gateway::upstream::{StreamingUpstream, UpstreamStream};
    use crate::codex_gateway::{GATEWAY_ROUTE, GatewayError, UpstreamRequest, upstream_url};

    /// Fictitious sentinels. A real credential here would be a bug in the test, not a secret.
    const FAKE_BEARER: &str = "Bearer SUPER_SECRET_DO_NOT_LEAK";
    const FAKE_ACCOUNT: &str = "FAKE_ACCOUNT_DO_NOT_LEAK";
    const MARKER: &str = "REQ-000057";

    // ----- the fake upstream ------------------------------------------------------------

    /// Stands in for the fixed origin: records what the gateway sent and replies with a
    /// scripted stream, one chunk per `read`.
    struct FakeUpstream {
        seen: Mutex<Vec<UpstreamRequest>>,
        status: u16,
        chunks: Vec<Vec<u8>>,
        gate: Mutex<Option<mpsc::Receiver<()>>>,
        fail: bool,
    }

    impl FakeUpstream {
        fn replying(chunks: Vec<Vec<u8>>) -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                status: 200,
                chunks,
                gate: Mutex::new(None),
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                fail: true,
                ..Self::replying(Vec::new())
            }
        }

        fn gated(chunks: Vec<Vec<u8>>, gate: mpsc::Receiver<()>) -> Self {
            Self {
                gate: Mutex::new(Some(gate)),
                ..Self::replying(chunks)
            }
        }

        fn seen(&self) -> Vec<UpstreamRequest> {
            self.seen.lock().expect("upstream log").clone()
        }
    }

    impl StreamingUpstream for FakeUpstream {
        fn send(&self, request: UpstreamRequest) -> Result<UpstreamStream, GatewayError> {
            self.seen.lock().expect("upstream log").push(request);
            if self.fail {
                return Err(GatewayError::UpstreamFailed);
            }
            let gate = self.gate.lock().expect("gate").take();
            Ok(UpstreamStream {
                status: self.status,
                headers: BTreeMap::from([(
                    "content-type".to_string(),
                    "text/event-stream".to_string(),
                )]),
                body: Box::new(ScriptedBody {
                    chunks: self.chunks.clone().into(),
                    gate,
                    delivered: 0,
                }),
            })
        }
    }

    /// Hands back one scripted chunk per `read`. With a gate, the second chunk waits until the
    /// test confirms the first one already reached the client — which is how the suite proves
    /// the relay is incremental instead of buffered.
    struct ScriptedBody {
        chunks: VecDeque<Vec<u8>>,
        gate: Option<mpsc::Receiver<()>>,
        delivered: usize,
    }

    impl Read for ScriptedBody {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let gate = if self.delivered == 1 {
                self.gate.take()
            } else {
                None
            };
            if let Some(gate) = gate {
                gate.recv().expect("the test releases the gate");
            }
            let Some(chunk) = self.chunks.pop_front() else {
                return Ok(0);
            };
            assert!(chunk.len() <= buffer.len(), "test chunks fit one read");
            buffer[..chunk.len()].copy_from_slice(&chunk);
            self.delivered += 1;
            Ok(chunk.len())
        }
    }

    // ----- the fake Codex ---------------------------------------------------------------

    struct Reply {
        status: u16,
        body: Vec<u8>,
    }

    /// A real HTTP client on the loopback port. Blocking, so it runs on the blocking pool
    /// while the runtime keeps driving the server.
    fn client() -> ureq::Agent {
        ureq::Agent::config_builder()
            .proxy(None)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into()
    }

    fn blocking_post(
        address: SocketAddr,
        path: &str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> Result<Reply, String> {
        let url = format!("http://{address}{path}");
        let mut request = client().post(&url);
        for (name, value) in &headers {
            request = request.header(name.as_str(), value.as_str());
        }
        let mut response = request.send(&body[..]).map_err(|error| error.to_string())?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .read_to_vec()
            .map_err(|error| error.to_string())?;
        Ok(Reply { status, body })
    }

    async fn try_post(
        address: SocketAddr,
        path: &str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> Result<Reply, String> {
        let path = path.to_string();
        tokio::task::spawn_blocking(move || blocking_post(address, &path, headers, body))
            .await
            .expect("the client task runs")
    }

    async fn post(
        address: SocketAddr,
        path: &str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> Reply {
        try_post(address, path, headers, body)
            .await
            .expect("the gateway answers")
    }

    async fn get(address: SocketAddr, path: &str) -> Reply {
        let path = path.to_string();
        tokio::task::spawn_blocking(move || {
            let url = format!("http://{address}{path}");
            let mut response = client().get(&url).call().expect("the gateway answers");
            let status = response.status().as_u16();
            let body = response.body_mut().read_to_vec().expect("body");
            Reply { status, body }
        })
        .await
        .expect("the client task runs")
    }

    fn auth_headers() -> Vec<(String, String)> {
        vec![
            ("authorization".to_string(), FAKE_BEARER.to_string()),
            ("chatgpt-account-id".to_string(), FAKE_ACCOUNT.to_string()),
            ("content-type".to_string(), "application/json".to_string()),
        ]
    }

    fn sse() -> Vec<Vec<u8>> {
        vec![
            b"event: response.created\ndata: {\"a\":1}\n\n".to_vec(),
            b"event: response.output_text.delta\ndata: {\"d\":\"x\"}\n\n".to_vec(),
            b"event: response.completed\ndata: {\"done\":true}\n\n".to_vec(),
        ]
    }

    /// A Responses-shaped body carrying repetitive tool output — the shape llmtrim shrinks.
    /// Pretty-printed on purpose: a pre-minified fixture would leave the lossless `safe`
    /// preset nothing to do and prove nothing.
    fn codex_body() -> Vec<u8> {
        let records: Vec<serde_json::Value> = (0..80)
            .map(|i| {
                serde_json::json!({
                    "requestId": format!("REQ-{i:06}"),
                    "service": "billing",
                    "level": "INFO",
                    "latencyMs": 12 + i,
                    "route": "/api/v1/billing/items",
                    "region": "sa-east-1",
                    "message": "request completed",
                })
            })
            .collect();
        let tool_output = serde_json::to_string_pretty(&records).expect("pretty tool output");
        let content = format!(
            "PRIVATE_PROMPT_DO_NOT_LEAK Read the log tool output and reply with the requestId \
             of {MARKER}.\n\n{tool_output}"
        );
        serde_json::to_vec(&serde_json::json!({
            "model": "gpt-5.6-sol",
            "messages": [{ "role": "user", "content": content }],
        }))
        .expect("serialize body")
    }

    async fn start(upstream: &Arc<FakeUpstream>) -> GatewayServer {
        serve(None, 0, ServerLimits::default(), Arc::clone(upstream))
            .await
            .expect("the gateway binds loopback")
    }

    // ----- bind policy ------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_binds_ipv4_loopback_only() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        assert_eq!(address.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert!(address.ip().is_loopback());
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_port_zero_returns_actual_loopback_port() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        assert_ne!(address.port(), 0, "the caller must learn the real port");
        TcpStream::connect(address).expect("the port is open");
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_rejects_non_loopback_configuration() {
        for host in [
            Ipv4Addr::UNSPECIFIED,
            Ipv4Addr::new(192, 168, 1, 10),
            Ipv4Addr::new(8, 8, 8, 8),
        ] {
            let upstream = Arc::new(FakeUpstream::replying(sse()));
            let host = IpAddr::V4(host);
            let outcome = serve(Some(host), 0, ServerLimits::default(), upstream).await;
            let error = outcome.err().expect("a non-loopback host is refused");
            assert!(
                matches!(error, ServeError::Bind(BindError::NotLoopback)),
                "{host}"
            );
        }
    }

    #[test]
    fn the_default_port_is_the_documented_one() {
        assert_eq!(DEFAULT_PORT, 43117);
    }

    // ----- the served exchange ----------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_post_valid_route_reaches_pipeline() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), codex_body()).await;
        assert_eq!(reply.status, 200);
        let seen = upstream.seen();
        assert_eq!(seen.len(), 1, "one request, no retry, no second provider");
        assert_eq!(seen[0].url, upstream_url());
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_wrong_route_returns_404() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        for path in ["/v1/responses", "/", "/backend-api/codex/other"] {
            let reply = post(address, path, auth_headers(), codex_body()).await;
            assert_eq!(reply.status, 404, "{path}");
        }
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_wrong_method_returns_405() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let reply = get(server.local_addr(), GATEWAY_ROUTE).await;
        assert_eq!(reply.status, 405);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_missing_auth_returns_401() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let headers = vec![("content-type".to_string(), "application/json".to_string())];
        let reply = post(address, GATEWAY_ROUTE, headers, codex_body()).await;
        assert_eq!(reply.status, 401);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_malformed_auth_returns_401() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        for value in ["Token abc", "Bearer ", "SUPER_SECRET_DO_NOT_LEAK"] {
            let headers = vec![("authorization".to_string(), value.to_string())];
            let reply = post(address, GATEWAY_ROUTE, headers, codex_body()).await;
            assert_eq!(reply.status, 401, "{value}");
        }
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_compression_failure_returns_500() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        // Not UTF-8, so the product's real transform cannot run. The gateway fails closed
        // rather than forwarding a body llmtrim never touched.
        let broken = vec![0x80, 0x81, 0xfe, 0xff];
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), broken).await;
        assert_eq!(reply.status, 500);
        assert!(upstream.seen().is_empty(), "nothing uncompressed goes out");
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_upstream_failure_returns_502() {
        let upstream = Arc::new(FakeUpstream::failing());
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), codex_body()).await;
        assert_eq!(reply.status, 502);
        assert_eq!(upstream.seen().len(), 1, "one attempt, no retry");
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_does_not_forward_on_auth_failure() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(address, GATEWAY_ROUTE, Vec::new(), codex_body()).await;
        assert_eq!(reply.status, 401);
        assert!(
            upstream.seen().is_empty(),
            "the loopback port is not an anonymous relay"
        );
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_does_not_forward_on_route_failure() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(
            address,
            "/v1/chat/completions",
            auth_headers(),
            codex_body(),
        )
        .await;
        assert_eq!(reply.status, 404);
        assert!(upstream.seen().is_empty(), "a refused route costs nothing");
        server.shutdown().await;
    }

    // ----- compression ------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_real_compression_reduces_body() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let body = codex_body();
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), body.clone()).await;
        assert_eq!(reply.status, 200);
        let sent = upstream.seen();
        assert!(
            sent[0].body.len() < body.len(),
            "{} is not smaller than {}",
            sent[0].body.len(),
            body.len()
        );
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_semantic_marker_survives() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), codex_body()).await;
        assert_eq!(reply.status, 200);
        let sent = upstream.seen();
        let text = String::from_utf8(sent[0].body.clone()).expect("utf-8");
        assert!(text.contains(MARKER), "the task's identifier must survive");
        server.shutdown().await;
    }

    // ----- streaming --------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_sse_order_preserved() {
        let chunks = sse();
        let upstream = Arc::new(FakeUpstream::replying(chunks.clone()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), codex_body()).await;
        assert_eq!(reply.status, 200);
        assert_eq!(reply.body, chunks.concat(), "bytes relayed in order");
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_relays_incrementally_without_buffering_the_response() {
        let chunks = sse();
        let (release, gate) = mpsc::channel();
        let upstream = Arc::new(FakeUpstream::gated(chunks.clone(), gate));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let first = chunks[0].len();
        let expected = chunks.concat();

        // The upstream refuses to produce its second chunk until the client says the first one
        // already arrived. A gateway that collected the whole response before answering would
        // deadlock here, which the timeout turns into a failure rather than a hang.
        let exchange = tokio::task::spawn_blocking(move || {
            let url = format!("http://{address}{GATEWAY_ROUTE}");
            let mut request = client().post(&url);
            for (name, value) in auth_headers() {
                request = request.header(name.as_str(), value.as_str());
            }
            let body = codex_body();
            let mut response = request.send(&body[..]).expect("the gateway answers");
            let mut reader = response.body_mut().as_reader();
            let mut head = vec![0_u8; first];
            reader.read_exact(&mut head).expect("the first chunk");
            release.send(()).expect("release the upstream");
            let mut rest = Vec::new();
            reader
                .read_to_end(&mut rest)
                .expect("the rest of the stream");
            head.extend_from_slice(&rest);
            head
        });

        let relayed = tokio::time::timeout(Duration::from_secs(30), exchange)
            .await
            .expect("the relay starts before the upstream finishes")
            .expect("the client task runs");
        assert_eq!(relayed, expected);
        server.shutdown().await;
    }

    #[test]
    fn the_response_queue_is_bounded() {
        // Bounded on purpose: the blocking reader parks on a full queue, so memory tracks the
        // queue depth rather than the length of the generation.
        assert!(RESPONSE_QUEUE_DEPTH > 0);
        assert!(RESPONSE_QUEUE_DEPTH <= 64);
        assert!(MAX_LOCAL_CLIENTS > 0);
        assert!(MAX_LOCAL_CLIENTS <= 1024);
    }

    // ----- limits -----------------------------------------------------------------------

    #[test]
    fn the_default_request_ceiling_is_the_products_own() {
        assert_eq!(MAX_REQUEST_BODY_BYTES, 64 * 1024 * 1024);
        assert_eq!(
            ServerLimits::default().max_request_body_bytes,
            MAX_REQUEST_BODY_BYTES
        );
        assert_eq!(ServerLimits::default().max_connections, MAX_LOCAL_CLIENTS);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_body_over_the_ceiling_is_refused_before_the_upstream_leg() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let limits = ServerLimits {
            max_request_body_bytes: 4096,
            ..ServerLimits::default()
        };
        let server = serve(None, 0, limits, Arc::clone(&upstream))
            .await
            .expect("the gateway binds loopback");
        let address = server.local_addr();
        let oversized = vec![b'x'; 8192];
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), oversized).await;
        assert_eq!(reply.status, 413);
        assert!(upstream.seen().is_empty(), "no partial body is forwarded");
        server.shutdown().await;
    }

    // ----- the client cannot pick a destination -----------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_local_header_can_redirect_the_gateway() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let mut headers = auth_headers();
        for name in ["x-forwarded-host", "forwarded", "x-upstream", "origin"] {
            headers.push((name.to_string(), "evil.example".to_string()));
        }
        let reply = post(address, GATEWAY_ROUTE, headers, codex_body()).await;
        assert_eq!(reply.status, 200);
        let sent = upstream.seen();
        assert_eq!(sent[0].url, upstream_url());
        for name in [
            "x-forwarded-host",
            "forwarded",
            "x-upstream",
            "origin",
            "host",
            "content-length",
        ] {
            assert!(!sent[0].headers.contains_key(name), "{name} was relayed");
        }
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_callers_credentials_reach_the_upstream_byte_for_byte() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let reply = post(address, GATEWAY_ROUTE, auth_headers(), codex_body()).await;
        assert_eq!(reply.status, 200);
        let sent = upstream.seen();
        assert_eq!(
            sent[0].headers.get("authorization").map(String::as_str),
            Some(FAKE_BEARER)
        );
        assert_eq!(
            sent[0]
                .headers
                .get("chatgpt-account-id")
                .map(String::as_str),
            Some(FAKE_ACCOUNT)
        );
        server.shutdown().await;
    }

    // ----- concurrency and lifecycle ----------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn several_connections_are_served_without_unbounded_work() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let mut calls = Vec::new();
        for _ in 0..3 {
            calls.push(post(address, GATEWAY_ROUTE, auth_headers(), codex_body()));
        }
        for call in calls {
            assert_eq!(call.await.status, 200);
        }
        assert_eq!(upstream.seen().len(), 3);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_shutdown_releases_port() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        server.shutdown().await;
        let rebound = TcpListener::bind(address).expect("the port is free again");
        drop(rebound);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_shutdown_leaves_no_listener() {
        let upstream = Arc::new(FakeUpstream::replying(sse()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        server.shutdown().await;
        assert!(
            TcpStream::connect(address).is_err(),
            "nothing is still listening on {address}"
        );
    }

    // ----- nothing leaks ----------------------------------------------------------------

    #[test]
    fn the_socket_layer_writes_nothing_and_carries_no_secret() {
        let whole = include_str!("socket.rs");
        let marker = concat!("#[cfg(", "test)]");
        let source = whole
            .split(marker)
            .next()
            .expect("production half")
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for forbidden in [
            "fs::write",
            "File::create",
            "OpenOptions",
            "capture_dir",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CODEX_API_KEY",
            "env_key",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NODE_EXTRA_CA_CERTS",
            "SSL_CERT_FILE",
            "CURL_CA_BUNDLE",
            "rcgen",
            "hudsucker",
            "ca.key",
        ] {
            assert!(!source.contains(forbidden), "{forbidden} must not appear");
        }
    }

    #[test]
    fn a_start_up_error_renders_no_credential() {
        let errors = [
            ServeError::Bind(BindError::NotLoopback),
            ServeError::Listen(std::io::Error::other("address in use")),
        ];
        for error in errors {
            let rendered = format!("{error} {error:?}");
            for sentinel in [
                "SUPER_SECRET_DO_NOT_LEAK",
                "FAKE_ACCOUNT_DO_NOT_LEAK",
                "PRIVATE_PROMPT_DO_NOT_LEAK",
                "Bearer ",
            ] {
                assert!(!rendered.contains(sentinel), "{error:?} leaked {sentinel}");
            }
        }
    }

    // ----- socket end-to-end ------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn socket_end_to_end_fake_codex_to_real_socket_to_fake_upstream() {
        let chunks = sse();
        let upstream = Arc::new(FakeUpstream::replying(chunks.clone()));
        let server = start(&upstream).await;
        let address = server.local_addr();
        let body = codex_body();

        let reply = post(address, GATEWAY_ROUTE, auth_headers(), body.clone()).await;

        // A: a real client spoke HTTP to a real loopback socket.
        assert!(address.ip().is_loopback());
        assert_eq!(reply.status, 200);
        // B: exactly one upstream request, to the one destination that exists.
        let sent = upstream.seen();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].url, upstream_url());
        // C: the product's real compression ran and the payload shrank.
        assert!(sent[0].body.len() < body.len());
        // D: the semantic marker survived it.
        let text = String::from_utf8(sent[0].body.clone()).expect("utf-8");
        assert!(text.contains(MARKER));
        // E: the caller's own credentials went upstream untouched.
        assert_eq!(
            sent[0].headers.get("authorization").map(String::as_str),
            Some(FAKE_BEARER)
        );
        // F: the stream came back byte for byte, in order.
        assert_eq!(reply.body, chunks.concat());

        let saved = body.len() - sent[0].body.len();
        println!("SOCKET_E2E=PASS");
        println!("SOCKET_LOCAL_REQUEST_BYTES={}", body.len());
        println!("SOCKET_UPSTREAM_REQUEST_BYTES={}", sent[0].body.len());
        println!(
            "SOCKET_LOCAL_PAYLOAD_REDUCTION_PERCENT={:.2}",
            saved as f64 * 100.0 / body.len() as f64
        );
        println!("SOCKET_RESPONSE_BYTES={}", reply.body.len());

        server.shutdown().await;
    }
}
