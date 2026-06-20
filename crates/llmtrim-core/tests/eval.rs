//! Evaluation harness — input-side and output-side token savings, kept separate.
//!
//! Input and output evals run separately: the input eval runs
//! INPUT-side stages only (hygiene + serialize); the output-shaping terse
//! instruction is excluded so it doesn't pollute the input measurement. All
//! measurement uses the real tokenizer.
//!
//! Run `cargo test --test eval -- --nocapture` to see the savings report.

use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;
use serde_json::json;

mod common;
use common::{input_only, user_chat};

const USERS: &str = include_str!("fixtures/openai_users.json");
const EVENTS: &str = include_str!("fixtures/anthropic_events.json");
const PROSE: &str = include_str!("fixtures/openai_prose.json");
const PRETTY: &str = include_str!("fixtures/openai_pretty.json");

fn pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        (before as f64 - after as f64) / before as f64 * 100.0
    }
}

fn eval_input(name: &str, body: &str, provider: ProviderKind) -> f64 {
    let r = llmtrim_core::compress_with_config(body, Some(provider), &input_only())
        .unwrap_or_else(|e| panic!("{name}: compress failed: {e}"));
    let p = pct(r.input_tokens_before.0, r.input_tokens_after.0);
    println!(
        "  {name:<22} {:>5} -> {:>5} tokens  ({p:>5.1}%)  [{}]",
        r.input_tokens_before, r.input_tokens_after, r.tokenizer_label
    );
    p
}

#[test]
fn input_side_savings_report() {
    println!("\nllmtrim input-side eval (Stage D only):");
    let users = eval_input("openai_users(array)", USERS, ProviderKind::OpenAi);
    let events = eval_input("anthropic_events(arr)", EVENTS, ProviderKind::Anthropic);
    let prose = eval_input("openai_prose", PROSE, ProviderKind::OpenAi);
    let pretty = eval_input("openai_pretty(json)", PRETTY, ProviderKind::OpenAi);

    assert!(
        users >= 25.0,
        "uniform array (OpenAI) should save >=25%, got {users:.1}%"
    );
    assert!(
        events >= 20.0,
        "uniform array (Anthropic) should save >=20%, got {events:.1}%"
    );
    assert!(
        prose.abs() < 0.01,
        "prose must not change (no input stage applies), got {prose:.1}%"
    );
    assert!(
        pretty > 0.0,
        "pretty JSON should minify to a net win, got {pretty:.1}%"
    );
}

#[test]
fn robustness_never_panics_on_edge_inputs() {
    let cfg = DenseConfig::lossless();
    for body in [
        r#"{}"#,
        r#"{"messages":[]}"#,
        r#"{"messages":[{"role":"user","content":"héllo 日本語 🚀 commas, and \"quotes\""}]}"#,
        r#"{"messages":[{"role":"user","content":"[]"}]}"#,
    ] {
        let _ = llmtrim_core::compress_with_config(body, Some(ProviderKind::OpenAi), &cfg);
    }
    // Malformed JSON must error, not panic.
    assert!(
        llmtrim_core::compress_with_config("{not json", Some(ProviderKind::OpenAi), &cfg).is_err()
    );
}

/// Stage D guardrail: non-uniform / non-flat / too-small arrays must pass
/// through unchanged as JSON — never mis-encoded to TOON.
#[test]
fn guardrails_skip_unsafe_array_shapes() {
    let cases = [
        ("ragged keys", json!([{"a":1,"b":2},{"a":3}])),
        ("nested value", json!([{"a":1},{"a":{"x":1}}])),
        ("single row", json!([{"a":1,"b":2}])),
    ];
    for (label, arr) in cases {
        let content = serde_json::to_string(&arr).unwrap();
        let body = user_chat("gpt-4o", &[content.as_str()]);
        let r =
            llmtrim_core::compress_with_config(&body, Some(ProviderKind::OpenAi), &input_only())
                .unwrap();
        let out: serde_json::Value = serde_json::from_str(&r.request_json).unwrap();
        let c = out
            .pointer("/messages/0/content")
            .and_then(|v| v.as_str())
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(c)
            .unwrap_or_else(|_| panic!("{label}: content is not JSON => was wrongly TOON-encoded"));
        assert_eq!(parsed, arr, "{label}: array must pass through unchanged");
    }
}

/// Stage D base64 stripping (opt-in): saves tokens when enabled, untouched by default.
#[test]
fn base64_strip_eval() {
    let blob = "Zm9vYmFy".repeat(40); // 320 base64 chars
    let content = format!("log dump: {blob} (end)");
    let body = user_chat("gpt-4o", &[content.as_str()]);

    let on = DenseConfig {
        strip_base64: true,
        ..input_only()
    };
    let r = llmtrim_core::compress_with_config(&body, Some(ProviderKind::OpenAi), &on).unwrap();
    println!(
        "\nbase64 strip eval: {} -> {} tokens",
        r.input_tokens_before, r.input_tokens_after
    );
    assert!(
        r.input_tokens_after < r.input_tokens_before,
        "stripping base64 saves tokens"
    );
    assert!(r.request_json.contains("elided"));

    let r2 = llmtrim_core::compress_with_config(&body, Some(ProviderKind::OpenAi), &input_only())
        .unwrap();
    assert!(
        r2.request_json.contains(&blob),
        "default keeps base64 (opt-in only)"
    );
}

/// Stage B retrieval (opt-in): prune a large multi-topic doc to the query-relevant
/// chunks; the answer chunk must survive (recall) and tokens must drop.
#[test]
fn retrieval_eval() {
    let topics = [
        "The cafeteria serves lunch from noon until two in the afternoon.",
        "Parking is available in the north lot for all visitors and staff.",
        "Quarterly revenue for the logistics division was 4.2 million dollars.",
        "Recycling bins are located on every floor near the elevators.",
        "Office hours run from nine to five on weekdays only.",
        "The fire assembly point is the south courtyard by the flagpole.",
    ];
    let big = topics.join("\n\n");
    let body = user_chat(
        "gpt-4o",
        &[big.as_str(), "what was the quarterly logistics revenue?"],
    );
    let cfg = DenseConfig {
        retrieve: true,
        retrieve_keep_ratio: 0.34,
        retrieve_min_segment_chars: 120,
        ..input_only()
    };
    let r = llmtrim_core::compress_with_config(&body, Some(ProviderKind::OpenAi), &cfg).unwrap();
    println!(
        "\nStage B retrieval eval: {} -> {} tokens  ({:.1}%)",
        r.input_tokens_before,
        r.input_tokens_after,
        pct(r.input_tokens_before.0, r.input_tokens_after.0)
    );
    assert!(
        r.input_tokens_after < r.input_tokens_before,
        "retrieval prunes tokens"
    );
    let out: serde_json::Value = serde_json::from_str(&r.request_json).unwrap();
    let kept = out
        .pointer("/messages/0/content")
        .and_then(|v| v.as_str())
        .unwrap();
    assert!(
        kept.contains("revenue") && kept.contains("logistics"),
        "answer chunk retained (recall)"
    );
}
