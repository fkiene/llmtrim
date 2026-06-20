//! Proxy-Wasm plugin: compress the LLM API request body with llmtrim on its way upstream.
//!
//! One wasm module serves both Kong and Higress (both proxy-wasm 0.2 ABI hosts). The host
//! buffers the request body, hands it to [`llmtrim_gateway::compress_body`], and forwards what
//! comes back. All decisions (fail-open, size guard, config parsing) live in `llmtrim-gateway`,
//! which is unit-tested on the host; this crate is the thin lifecycle glue that only builds for
//! wasm, so it carries no logic worth testing here.

use llmtrim_gateway::{Config, compress_body};
use proxy_wasm::traits::{Context, HttpContext, RootContext};
use proxy_wasm::types::{Action, ContextType, LogLevel};

proxy_wasm::main! {{
    proxy_wasm::set_log_level(LogLevel::Warn);
    proxy_wasm::set_root_context(|_| -> Box<dyn RootContext> {
        Box::new(Root { config: Config::default() })
    });
}}

/// Holds the parsed plugin config and stamps it onto each request's filter.
struct Root {
    config: Config,
}

impl Context for Root {}

impl RootContext for Root {
    fn on_configure(&mut self, _config_size: usize) -> bool {
        // Fail-open: a missing or malformed config yields safe defaults (auto routing,
        // default body-size guard), so the plugin still runs rather than rejecting traffic.
        let bytes = self.get_plugin_configuration().unwrap_or_default();
        self.config = Config::from_json_bytes(&bytes);
        true
    }

    fn create_http_context(&self, _context_id: u32) -> Option<Box<dyn HttpContext>> {
        Some(Box::new(Filter {
            config: self.config.clone(),
        }))
    }

    fn get_type(&self) -> Option<ContextType> {
        Some(ContextType::HttpContext)
    }
}

/// Per-request filter: compress the buffered request body in place.
struct Filter {
    config: Config,
}

impl Context for Filter {}

impl HttpContext for Filter {
    fn on_http_request_body(&mut self, body_size: usize, end_of_stream: bool) -> Action {
        if !end_of_stream {
            // Wait until the host has buffered the whole body; then `body_size` is the total.
            return Action::Pause;
        }
        if body_size == 0 {
            return Action::Continue;
        }
        if let Some(body) = self.get_http_request_body(0, body_size) {
            let outcome = compress_body(&body, &self.config);
            if !outcome.is_passthrough() {
                self.set_http_request_body(0, body_size, &outcome.body);
            }
            // Set Content-Length to the final body length so HTTP/1.1 upstreams (the OpenAI /
            // Anthropic endpoints) get a well-framed request. The request is still buffered
            // here (we paused the body), so its headers have not gone upstream yet and remain
            // mutable. `outcome.body` is the original bytes on passthrough, the compressed
            // bytes otherwise, so this length is correct either way.
            self.set_http_request_header("content-length", Some(&outcome.body.len().to_string()));
        }
        Action::Continue
    }
}
