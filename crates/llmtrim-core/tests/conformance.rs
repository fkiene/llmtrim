//! Cross-binding conformance suite for the adapter contract.
//!
//! Every integration adapter (Kong, Higress, Vercel AI SDK, OpenCode, Continue, Genkit,
//! LiteLLM, LangChain) calls [`llmtrim_core::rewrite_request`] through its binding. These
//! fixtures pin that entrypoint's behavior so a core change cannot silently alter what an
//! adapter forwards. The JS (vitest) and Python (pytest) harnesses load the SAME JSON files
//! and assert the same invariants, so all three surfaces stay in lockstep.
//!
//! Each fixture is `tests/conformance/<name>.json`:
//!
//! ```json
//! {
//!   "input":   { ...raw provider request body... },
//!   "provider": "anthropic",     // or null to auto-detect
//!   "preset":   "agent",         // or null (the adapter default `auto` is applied)
//!   "expect": {
//!     "provider":       "anthropic",   // required: detected/declared provider
//!     "output_shaped":  false,         // required: tokenizer-independent
//!     "request_json":   { ... }        // optional golden; deep-equal on parsed JSON
//!   }
//! }
//! ```
//!
//! Goldens are engine-generated, never hand-written: run `LLMTRIM_BLESS=1 cargo test -p
//! llmtrim-core --test conformance` to (re)write `expect.request_json` from the actual
//! output, review the diff, and commit. Token counts are deliberately NOT asserted — the
//! WASM build has no tiktoken, so counts differ by build while the rewritten body does not.

use std::fs;
use std::path::Path;

use llmtrim_core::ir::ProviderKind;
use serde_json::Value;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance")
}

fn provider_kind(s: &str) -> ProviderKind {
    s.parse()
        .unwrap_or_else(|_| panic!("fixture has unknown provider {s:?}"))
}

#[test]
fn conformance_fixtures_hold_across_the_adapter_contract() {
    let bless = std::env::var_os("LLMTRIM_BLESS").is_some();
    let dir = fixtures_dir();
    let mut checked = 0;
    // Collect every fixture's failures and report them together: one failing fixture must
    // not mask regressions in the others (the JS/Python harnesses run the same set).
    let mut failures: Vec<String> = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort();
    let total = entries.len();

    for path in entries {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let raw = fs::read_to_string(&path).expect("read fixture");
        let mut fixture: Value = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("{name}: fixture is not valid JSON: {e}"));

        let input = serde_json::to_string(&fixture["input"])
            .unwrap_or_else(|_| panic!("{name}: missing `input`"));
        let provider = fixture["provider"].as_str().map(provider_kind);
        let preset = fixture["preset"].as_str().map(str::to_string);

        let result = llmtrim_core::rewrite_request(&input, provider, preset.as_deref())
            .unwrap_or_else(|e| panic!("{name}: rewrite_request failed: {e:#}"));

        // Deep-equal on PARSED JSON (never string-equal: serde, JS, and Python serializers
        // order keys and format numbers differently).
        let actual_body: Value =
            serde_json::from_str(&result.request_json).expect("rewritten body is valid JSON");

        if bless {
            // Seed every expectation from the engine, then review the written values.
            fixture["expect"] = serde_json::json!({
                "provider": result.provider.as_str(),
                "output_shaped": result.output_shaped,
                "request_json": actual_body,
            });
            let pretty = serde_json::to_string_pretty(&fixture).unwrap();
            fs::write(&path, pretty + "\n").expect("write blessed fixture");
            checked += 1;
            continue;
        }

        let expect = &fixture["expect"];

        // Tokenizer-independent invariants, identical on every binding.
        let want_provider = expect["provider"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: expect.provider missing"));
        if result.provider.as_str() != want_provider {
            failures.push(format!(
                "{name}: provider mismatch: got {}, want {want_provider}",
                result.provider.as_str()
            ));
        }
        let want_shaped = expect["output_shaped"]
            .as_bool()
            .unwrap_or_else(|| panic!("{name}: expect.output_shaped missing"));
        if result.output_shaped != want_shaped {
            failures.push(format!(
                "{name}: output_shaped mismatch: got {}, want {want_shaped}",
                result.output_shaped
            ));
        }

        let golden = expect["request_json"].clone();
        assert!(
            !golden.is_null(),
            "{name}: no golden request_json (run with LLMTRIM_BLESS=1 to seed it)"
        );
        // The load-bearing invariant for every adapter: any frozen (cache_control) block must
        // survive byte-for-byte, or we churn the provider's cached prefix. Check it explicitly
        // so a careless rebless can't quietly relax it under the full-body deep-equal.
        assert_cache_control_stable(&name, &actual_body, &golden, &mut failures);
        if actual_body != golden {
            failures.push(format!(
                "{name}: rewritten body diverged from golden (review and rebless if intended)"
            ));
        }
        checked += 1;
    }

    assert_eq!(
        checked, total,
        "only {checked}/{total} fixtures ran; some were skipped"
    );
    assert!(
        total > 0,
        "no conformance fixtures found in {}",
        dir.display()
    );
    assert!(
        failures.is_empty(),
        "conformance failures:\n{}",
        failures.join("\n")
    );
}

/// Assert every array entry carrying a `cache_control` marker in the golden is reproduced
/// byte-for-byte (same position, same surrounding fields) in the actual output. Scans the
/// top-level arrays providers use for the cacheable prefix (`system`, `messages`, `tools`).
fn assert_cache_control_stable(
    name: &str,
    actual: &Value,
    golden: &Value,
    failures: &mut Vec<String>,
) {
    for key in ["system", "messages", "tools"] {
        let (Some(g_arr), a_arr) = (
            golden.get(key).and_then(Value::as_array),
            actual.get(key).and_then(Value::as_array),
        ) else {
            continue;
        };
        for (i, g) in g_arr.iter().enumerate() {
            if g.get("cache_control").is_none() {
                continue;
            }
            let a = a_arr.and_then(|a| a.get(i));
            if a != Some(g) {
                failures.push(format!(
                    "{name}: frozen {key}[{i}] (cache_control) not byte-stable"
                ));
            }
        }
    }
}
