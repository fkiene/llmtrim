//! Build script — validate and track embedded assets at build time.
//!
//! Mirrors RTK's `build.rs` discipline: fail the build early if a required prompt
//! asset (the Stage D/F prompt fragments under `prompts/`) is missing or empty, and
//! trigger a rebuild when one changes. The assets themselves are embedded in the
//! binary via `include_str!` from `src/config.rs` and the stage modules.

use std::path::Path;

const REQUIRED_ASSETS: &[&str] = &[
    "prompts/toon_legend.txt",
    "prompts/csv_legend.txt",
    "prompts/output_terse.txt",
    "prompts/output_draft.txt",
    "prompts/output_compact_code.txt",
    "prompts/output_token_budget.txt",
    "prompts/ngram_glossary.txt",
];

fn main() {
    for asset in REQUIRED_ASSETS {
        println!("cargo:rerun-if-changed={asset}");
        let content = std::fs::read_to_string(Path::new(asset))
            .unwrap_or_else(|e| panic!("required asset `{asset}` is unreadable: {e}"));
        assert!(
            !content.trim().is_empty(),
            "required asset `{asset}` must not be empty",
        );
    }
}
