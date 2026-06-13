//! Per-request compression latency — the overhead llmtrim adds on top of the network
//! round-trip (which dwarfs it). Warms first (the daemon pays the tokenizer-vocab + lazy-
//! regex load once at startup, not per request), then times the warm path and attributes
//! the cost across tokenize / clone / parse / serialize.
//!
//! Zero-config: `cargo bench --bench latency` runs on a bundled representative request.
//! Override with your own: `cargo bench --bench latency -- path/to/request.json [provider]`.

use std::str::FromStr;
use std::time::Instant;

/// Representative request: a coding-assistant turn (system + a user message with a fenced
/// code block + prose) — exercises hygiene, skeletonization, and tokenization.
const FIXTURE: &str = r#"{"model":"gpt-4o","messages":[
{"role":"system","content":"You are a meticulous coding assistant. Answer precisely."},
{"role":"user","content":"Review this function for bugs:\n```rust\nfn process(data: &[i32]) -> i32 {\n    let mut total = 0;\n    for x in data {\n        if *x > 0 {\n            total += x * 2;\n        }\n    }\n    total\n}\n```\nDoes it handle the spec's edge cases?"}
]}"#;

fn main() {
    // First non-flag arg = request path (skips cargo's injected `--bench`); next = provider.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let input = match args.first() {
        Some(p) => std::fs::read_to_string(p).expect("failed to read request file"),
        None => FIXTURE.to_string(),
    };
    let kind = args
        .get(1)
        .map(|p| llmtrim_core::ir::ProviderKind::from_str(p).expect("unknown provider"));
    let config = llmtrim_core::config::DenseConfig::auto();

    // Warm: tokenizer vocab + lazy regexes (daemon pays this once, not per request).
    for _ in 0..5 {
        let _ = llmtrim_core::compress_with_config(&input, kind, &config).unwrap();
    }

    let n = 100;
    let t = Instant::now();
    let mut last = None;
    for _ in 0..n {
        last = Some(llmtrim_core::compress_with_config(&input, kind, &config).unwrap());
    }
    let per_req_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
    let r = last.unwrap();

    let counter = llmtrim_core::tokenizer::counter_for(r.provider, r.model.as_deref()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&input).unwrap();
    let tools_str = value
        .get("tools")
        .map(|t| t.to_string())
        .unwrap_or_default();
    let bench = |label: &str, f: &dyn Fn()| {
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        println!(
            "  {label}: {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0 / f64::from(n)
        );
    };

    println!("request: {} bytes, provider={:?}", input.len(), r.provider);
    println!(
        "input tokens: {} -> {} ({:.1}% saved)",
        r.input_tokens_before,
        r.input_tokens_after,
        100.0 * (1.0 - r.input_tokens_after.0 as f64 / r.input_tokens_before.0.max(1) as f64)
    );
    println!("compress latency: {per_req_ms:.2} ms/req (warm, avg of {n})");
    println!("attribution (1x each):");
    bench("full tokenize (content+tools)", &|| {
        let _ = counter.count(&input);
    });
    bench("tools tokenize only", &|| {
        let _ = counter.count(&tools_str);
    });
    bench("Value clone (per-stage snapshot)", &|| {
        let _ = value.clone();
    });
    bench("JSON parse", &|| {
        let _: serde_json::Value = serde_json::from_str(&input).unwrap();
    });
    bench("JSON serialize", &|| {
        let _ = value.to_string();
    });
}
