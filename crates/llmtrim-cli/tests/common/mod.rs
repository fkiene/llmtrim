//! Shared helpers for the `eval`, `quality`, and `output` integration crates (per
//! `.claude/rules/cli-testing.md`). Each `tests/*.rs` is its own crate, so these
//! live in a `common/` subdir module each `mod common;`-includes — not a top-level
//! `tests/common.rs`, which cargo would compile as its own test binary.
//!
//! `allow(dead_code)`: every crate that includes this module uses only a subset of
//! the helpers, so the unused ones would warn per-crate (standard `tests/common`
//! idiom).
#![allow(dead_code)]

use llmtrim_core::config::DenseConfig;
use serde_json::{Value, json};

/// Input-side config: the terse output-shaping instruction is off so input savings
/// measure cleanly (input and output evals run separately).
pub fn input_only() -> DenseConfig {
    DenseConfig {
        output_control: false,
        ..DenseConfig::default()
    }
}

/// A chat request with one user message per `contents` entry, in order — the
/// `{"model","messages":[…]}` body shape every eval/quality case builds.
pub fn user_chat(model: &str, contents: &[&str]) -> String {
    let messages: Vec<Value> = contents
        .iter()
        .map(|c| json!({"role": "user", "content": c}))
        .collect();
    json!({"model": model, "messages": messages}).to_string()
}
