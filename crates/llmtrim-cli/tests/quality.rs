//! Quality gate: recall-vs-savings sweep for lossy Stage B retrieval.
//!
//! The actionable output is the lowest `keep_ratio` at which answer-bearing content
//! still survives — the gate for turning retrieval default-on. Network-free (recall
//! is a static proxy for task success on retrieval).
//!
//! Run `cargo test --test quality -- --nocapture` to see the curve.

use llmtrim::quality::{RecallCase, mean_recall, run_recall};
use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;

mod common;
use common::{input_only, user_chat};

fn case(name: &str, topics: &[&str], query: &str, must_keep: &str) -> RecallCase {
    let doc = topics.join("\n\n");
    let request = user_chat("gpt-4o", &[doc.as_str(), query]);
    RecallCase {
        name: name.to_string(),
        request,
        provider: ProviderKind::OpenAi,
        must_keep: vec![must_keep.to_string()],
    }
}

fn corpus() -> Vec<RecallCase> {
    vec![
        case(
            "revenue",
            &[
                "The cafeteria serves lunch from noon until two.",
                "Parking is in the north lot for visitors and staff.",
                "Quarterly revenue for the logistics division was 4.2 million dollars.",
                "Recycling bins are on every floor near the elevators.",
                "Office hours run nine to five on weekdays.",
            ],
            "what was the quarterly logistics revenue?",
            "4.2 million",
        ),
        case(
            "parking-permit",
            &[
                "The annual gala is scheduled for the third Friday in October.",
                "Visitor parking permits are issued at the front desk for ten dollars.",
                "The library closes at eight on weeknights.",
                "Coffee is restocked every morning by eight thirty.",
                "The gym requires a separate membership badge.",
            ],
            "how much is a visitor parking permit?",
            "ten dollars",
        ),
        case(
            "wifi",
            &[
                "Conference room B seats twelve and has a projector.",
                "The guest wifi password is rotated weekly and is SUNFLOWER42.",
                "Lunch options include vegetarian and halal meals.",
                "Badge access is required after seven in the evening.",
                "The mail room is on the ground floor.",
            ],
            "what is the guest wifi password?",
            "SUNFLOWER42",
        ),
    ]
}

#[test]
fn recall_vs_savings_sweep() {
    let cases = corpus();
    println!("\nStage B recall-vs-savings sweep (keep_ratio):");
    let mut rows = Vec::new();
    for &ratio in &[0.2_f64, 0.4, 0.6, 0.8] {
        let cfg = DenseConfig {
            retrieve: true,
            retrieve_keep_ratio: ratio,
            retrieve_min_segment_chars: 120,
            // This sweep characterizes *retrieve's own* recall-vs-savings curve (to choose
            // a safe keep_ratio), so the quality gate is disabled here: with it on, the
            // tightest ratio (0.2) drops the answer chunk and the gate reverts the prune —
            // honest product behavior, but it masks the raw recall drop this diagnostic
            // exists to expose, and flattens savings to 0% at 0.2.
            quality_gate: false,
            ..input_only()
        };
        let results = run_recall(&cases, &cfg).unwrap();
        let recall = mean_recall(&results);
        let savings = results.iter().map(|r| r.savings_pct()).sum::<f64>() / results.len() as f64;
        rows.push((ratio, recall, savings));
        println!("  keep={ratio:.1}  recall={recall:.2}  savings={savings:.1}%");
    }

    // BM25 ranks each answer chunk top, so answers survive even under aggressive
    // pruning: recall stays perfect across the sweep on this corpus.
    let at_06 = rows.iter().find(|(r, ..)| (*r - 0.6).abs() < 1e-9).unwrap();
    assert!(
        (at_06.1 - 1.0).abs() < 1e-9,
        "recall must be 1.0 at keep_ratio 0.6"
    );

    // Lower keep_ratio prunes more, so it saves at least as much as a higher ratio.
    let low = rows.first().unwrap();
    let high = rows.last().unwrap();
    assert!(
        low.2 >= high.2,
        "lower keep_ratio should save at least as much"
    );
    assert!(low.2 > 0.0, "retrieval should produce positive savings");
}
