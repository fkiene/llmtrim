//! Output-side eval — the complement to the input-side `eval.rs`.
//!
//! Real output-token savings (terse → ~73% fewer response tokens) are a property of
//! the model's *response*, so they're measured live/out-of-band. Offline
//! and network-free like its sibling evals, this gate proves the rest end-to-end
//! through the public `compress_with_config` API:
//!   1. provider-agnostic injection — the shaping instruction reaches the request for
//!      OpenAI (a `system` message) and Anthropic (top-level `system`) alike;
//!   2. bounded input cost — the instruction's *input*-token overhead stays small: the
//!      fixed cost that buys the output saving must not balloon (the bloat regression
//!      `stages/output.rs` warns against);
//!   3. config -> request wiring — draft / soft-budget / compact-code / hard cap each
//!      flow from `DenseConfig` to the emitted request, and `output_control:false`
//!      ships none of them.
//!
//! Run `cargo test --test output -- --nocapture` to see the overhead report.

use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;
use llmtrim_core::stages::output::{
    COMPACT_CODE_INSTRUCTION, DRAFT_INSTRUCTION, TERSE_INSTRUCTION,
};
use serde_json::Value;

mod common;
use common::user_chat;

const PROSE: &str = include_str!("fixtures/openai_prose.json");

/// Output shaping on, input stages at their defaults (so only the terse instruction
/// distinguishes it from `DenseConfig::default()`, which ships output shaping off).
fn output_on() -> DenseConfig {
    DenseConfig {
        output_control: true,
        ..DenseConfig::default()
    }
}

fn request_json(body: &str, provider: ProviderKind, cfg: &DenseConfig) -> String {
    llmtrim_core::compress_with_config(body, Some(provider), cfg)
        .expect("compress failed")
        .request_json
}

/// Every instruction the shaper can inject, gathered wherever the provider puts it:
/// OpenAI/Google in `messages`, Anthropic in the top-level `system` field, Google in
/// `systemInstruction`. Lets one assertion cover all wire shapes.
fn injected_text(request_json: &str) -> String {
    let v: Value = serde_json::from_str(request_json).expect("request is valid JSON");
    let mut out = String::new();
    let push_blocks = |blocks: &[Value], out: &mut String| {
        for b in blocks {
            if let Some(t) = b.get("text").and_then(Value::as_str) {
                out.push_str(t);
                out.push('\n');
            }
        }
    };
    if let Some(msgs) = v.pointer("/messages").and_then(Value::as_array) {
        for m in msgs {
            match m.get("content") {
                Some(Value::String(s)) => {
                    out.push_str(s);
                    out.push('\n');
                }
                Some(Value::Array(blocks)) => push_blocks(blocks, &mut out),
                _ => {}
            }
        }
    }
    match v.get("system") {
        Some(Value::String(s)) => out.push_str(s),
        Some(Value::Array(blocks)) => push_blocks(blocks, &mut out),
        _ => {}
    }
    if let Some(parts) = v
        .pointer("/systemInstruction/parts")
        .and_then(Value::as_array)
    {
        push_blocks(parts, &mut out);
    }
    out
}

/// The output-token cap, read field-agnostically (OpenAI writes `max_completion_tokens`
/// when absent, updates `max_tokens` when the caller already set it).
fn cap(request_json: &str) -> Option<u64> {
    let v: Value = serde_json::from_str(request_json).unwrap();
    v.pointer("/max_tokens")
        .or_else(|| v.pointer("/max_completion_tokens"))
        .and_then(Value::as_u64)
}

/// The eval number: real output savings are response-side (live-only), so offline we
/// gate the *input* cost of buying them — the terse instruction must reach the request
/// (so it costs input) yet stay cheap (output costs ~3-5x input; a few input tokens buy
/// a large output cut, but the instruction itself must not bloat the prompt).
#[test]
fn output_instruction_input_overhead_is_small() {
    let off = llmtrim_core::compress_with_config(
        PROSE,
        Some(ProviderKind::OpenAi),
        &DenseConfig::default(),
    )
    .unwrap();
    let on = llmtrim_core::compress_with_config(PROSE, Some(ProviderKind::OpenAi), &output_on())
        .unwrap();
    let overhead = on.input_tokens_after.0 as i64 - off.input_tokens_after.0 as i64;

    println!("\nllmtrim output-side eval (terse instruction input cost):");
    println!(
        "  off {:>4} -> on {:>4} tokens  (+{overhead} input buys the output saving)  [{}]",
        off.input_tokens_after, on.input_tokens_after, on.tokenizer_label
    );

    assert!(
        overhead > 0,
        "terse instruction must reach the request (it costs input tokens)"
    );
    assert!(
        overhead <= 40,
        "terse instruction input overhead must stay small, was +{overhead}"
    );
    assert!(
        injected_text(&on.request_json).contains(TERSE_INSTRUCTION),
        "terse instruction present when output_control on"
    );
    assert!(
        !injected_text(&off.request_json).contains(TERSE_INSTRUCTION),
        "no instruction when output_control off"
    );
}

/// The instruction reaches the request through the public API for every provider's wire
/// shape (OpenAI system message vs Anthropic top-level `system`), and only when enabled.
#[test]
fn terse_injected_provider_agnostic() {
    for (provider, model) in [
        (ProviderKind::OpenAi, "gpt-4o"),
        (ProviderKind::Anthropic, "claude-sonnet-4"),
    ] {
        let body = user_chat(model, &["What caused the outage?"]);

        let on = request_json(&body, provider, &output_on());
        assert!(
            injected_text(&on).contains(TERSE_INSTRUCTION),
            "{provider:?}: terse instruction injected end-to-end"
        );

        let off = request_json(&body, provider, &DenseConfig::default());
        assert!(
            !injected_text(&off).contains(TERSE_INSTRUCTION),
            "{provider:?}: no instruction when output shaping off"
        );
    }
}

/// Each output knob in `DenseConfig` reaches the emitted request — proving the
/// config -> stage wiring in `lib.rs`, not just the stage in isolation.
#[test]
fn config_knobs_flow_through_to_request() {
    let body = user_chat("gpt-4o", &["Refactor this function."]);

    let draft = request_json(
        &body,
        ProviderKind::OpenAi,
        &DenseConfig {
            output_control: true,
            output_level: "draft".to_string(),
            ..DenseConfig::default()
        },
    );
    assert!(
        injected_text(&draft).contains(DRAFT_INSTRUCTION),
        "Chain-of-Draft instruction flows through"
    );

    let budget = request_json(
        &body,
        ProviderKind::OpenAi,
        &DenseConfig {
            output_control: true,
            output_token_budget: Some(120),
            ..DenseConfig::default()
        },
    );
    assert!(
        injected_text(&budget).contains("120 tokens"),
        "soft token budget flows through"
    );

    // Compact-code is gated on its own flag — it ships even with `output_control` off.
    let compact = request_json(
        &body,
        ProviderKind::OpenAi,
        &DenseConfig {
            output_compact_code: true,
            ..DenseConfig::default()
        },
    );
    assert!(
        injected_text(&compact).contains(COMPACT_CODE_INSTRUCTION),
        "compact-code instruction flows through with output_control off"
    );
}

/// The hard output cap is imposed end-to-end when the caller set none, and a
/// caller-set cap is never overwritten.
#[test]
fn hard_cap_imposed_only_when_absent() {
    let capped = DenseConfig {
        output_control: true,
        output_max_tokens: Some(256),
        ..DenseConfig::default()
    };

    let uncapped = user_chat("gpt-4o", &["hi"]);
    let r = request_json(&uncapped, ProviderKind::OpenAi, &capped);
    assert_eq!(cap(&r), Some(256), "our cap imposed when caller set none");

    let caller_capped =
        r#"{"model":"gpt-4o","max_tokens":99,"messages":[{"role":"user","content":"hi"}]}"#;
    let r2 = request_json(caller_capped, ProviderKind::OpenAi, &capped);
    assert_eq!(cap(&r2), Some(99), "caller-set cap must not be overwritten");
}
