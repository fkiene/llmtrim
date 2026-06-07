//! Sequential gated stage driver — the static fan-in pipeline.
//!
//! Runs each [`Transform`] in order, gating it per [`GateKind`], and accumulates
//! the rehydration plan plus a per-stage report. Token measurement uses the real
//! [`TokenCounter`] over the provider's content text segments.

use crate::gate::{GateKind, PlanEntry, Scope, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::tokenizer::{TokenCounter, Tokens};

/// What one stage did, for reporting and the savings ledger.
#[derive(Debug, Clone)]
pub struct StageReport {
    pub name: String,
    pub applied: bool,
    pub tokens_before: Tokens,
    pub tokens_after: Tokens,
    pub note: Option<String>,
}

/// The result of running the pipeline over a request.
#[derive(Debug, Clone, Default)]
pub struct PipelineOutcome {
    pub stages: Vec<StageReport>,
    pub plan: Vec<PlanEntry>,
    pub input_tokens_before: Tokens,
    pub input_tokens_after: Tokens,
}

/// Tokens over the content text segments the model reads (system + messages).
fn count_content(req: &Request, provider: &dyn Provider, counter: &dyn TokenCounter) -> usize {
    provider
        .content_text_pointers(req)
        .iter()
        .filter_map(|p| req.get_str(p))
        .map(|s| counter.count(s))
        .sum()
}

/// Tokens over the tool/function schemas (resent every call — Stage G prunes them).
fn count_tools(req: &Request, counter: &dyn TokenCounter) -> usize {
    req.raw()
        .get("tools")
        .map_or(0, |tools| counter.count(&tools.to_string()))
}

/// Sum of token counts over every content text segment + tool schemas. The input-token
/// measure the gate uses (the text the model actually reads), not the raw JSON envelope.
pub fn content_tokens(req: &Request, provider: &dyn Provider, counter: &dyn TokenCounter) -> usize {
    count_content(req, provider, counter) + count_tools(req, counter)
}

/// Run `stages` over `req`, gating each one. The request is mutated in place to
/// its final compressed form.
pub fn run(
    req: &mut Request,
    provider: &dyn Provider,
    counter: &dyn TokenCounter,
    stages: &[Box<dyn Transform>],
) -> PipelineOutcome {
    let mut plan: Vec<PlanEntry> = Vec::new();
    let mut reports = Vec::with_capacity(stages.len());
    // Opt-in per-stage wall-clock attribution (apply + re-count), to stderr. Set
    // `LLMTRIM_PROFILE=1` when benchmarking; zero cost otherwise (one env read per run).
    let profile = std::env::var_os("LLMTRIM_PROFILE").is_some();
    // Track content and tools token counts separately, carried between stages: a stage's
    // `before` is the prior stage's result, and after a stage we re-count ONLY the part it
    // can change (its `scope`). Most agent stages touch only `tools`, so the big content
    // text isn't re-tokenized each time. Same gating decisions, far fewer tokenizations —
    // the per-request hot path.
    let mut content = count_content(req, provider, counter);
    let mut tools = count_tools(req, counter);
    let input_tokens_before = content + tools;

    for stage in stages {
        let before = content + tools;
        let snapshot = req.clone();
        let plan_mark = plan.len();
        let timer = profile.then(std::time::Instant::now);

        let (applied, after, note) = match stage.apply(req, provider, &mut plan) {
            Err(e) => {
                *req = snapshot;
                plan.truncate(plan_mark);
                (false, before, Some(format!("error: {e}")))
            }
            Ok(()) => {
                // Re-count only the part this stage can change; keep the rest cached.
                // A stage that left the request byte-identical can't have changed any
                // count, so skip the (BPE-expensive) re-tokenization — a structural
                // `Value` compare is ~an order of magnitude cheaper than tokenizing, and
                // most stages no-op on any given request shape (the `aggressive` preset
                // runs every stage; the non-matching ones revert).
                let (new_content, new_tools) = if req.raw() == snapshot.raw() {
                    (content, tools)
                } else {
                    match stage.scope() {
                        Scope::Content => (count_content(req, provider, counter), tools),
                        Scope::Tools => (content, count_tools(req, counter)),
                        Scope::Both => (
                            count_content(req, provider, counter),
                            count_tools(req, counter),
                        ),
                    }
                };
                let after = new_content + new_tools;
                if stage.gate_kind() == GateKind::InputTokens && after >= before {
                    // No net token win: revert (never block the user); counts unchanged.
                    *req = snapshot;
                    plan.truncate(plan_mark);
                    (false, before, Some("no token reduction".to_string()))
                } else {
                    content = new_content;
                    tools = new_tools;
                    (true, after, None)
                }
            }
        };

        if let Some(timer) = timer {
            eprintln!(
                "llmtrim profile: {:>14} {:>7.2} ms  {:>6} -> {:<6} tok{}",
                stage.name(),
                timer.elapsed().as_secs_f64() * 1000.0,
                before,
                after,
                if applied { "" } else { "  (reverted)" },
            );
        }

        reports.push(StageReport {
            name: stage.name().to_string(),
            applied,
            tokens_before: Tokens(before),
            tokens_after: Tokens(after),
            note,
        });
    }

    let input_tokens_after = content + tools;
    PipelineOutcome {
        stages: reports,
        plan,
        input_tokens_before: Tokens(input_tokens_before),
        input_tokens_after: Tokens(input_tokens_after),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateKind;
    use crate::ir::ProviderKind;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::Value;

    /// Test transform: overwrite the first user message content (InputTokens gate).
    struct SetContent {
        name: String,
        text: String,
    }
    impl Transform for SetContent {
        fn name(&self) -> &str {
            &self.name
        }
        fn gate_kind(&self) -> GateKind {
            GateKind::InputTokens
        }
        fn apply(
            &self,
            req: &mut Request,
            _provider: &dyn Provider,
            _plan: &mut Vec<PlanEntry>,
        ) -> anyhow::Result<()> {
            req.set("/messages/0/content", Value::String(self.text.clone()));
            Ok(())
        }
    }

    /// Test transform: add a system instruction (OutputShaping gate).
    struct AddSystem;
    impl Transform for AddSystem {
        fn name(&self) -> &str {
            "add-system"
        }
        fn gate_kind(&self) -> GateKind {
            GateKind::OutputShaping
        }
        fn apply(
            &self,
            req: &mut Request,
            provider: &dyn Provider,
            _plan: &mut Vec<PlanEntry>,
        ) -> anyhow::Result<()> {
            provider.add_system_instruction(req, "no preamble, no restating the question");
            Ok(())
        }
    }

    /// Test transform: always errors (must be reverted, never block).
    struct Boom;
    impl Transform for Boom {
        fn name(&self) -> &str {
            "boom"
        }
        fn gate_kind(&self) -> GateKind {
            GateKind::InputTokens
        }
        fn apply(
            &self,
            req: &mut Request,
            _provider: &dyn Provider,
            _plan: &mut Vec<PlanEntry>,
        ) -> anyhow::Result<()> {
            req.set("/messages/0/content", Value::String("damaged".to_string()));
            anyhow::bail!("intentional failure");
        }
    }

    fn fresh() -> (Request, Box<dyn TokenCounter>) {
        let req = Request::parse(
            ProviderKind::OpenAi,
            r#"{"messages":[{"role":"user","content":"this is a fairly long original message about widgets and gadgets"}]}"#,
        )
        .unwrap();
        (
            req,
            counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap(),
        )
    }

    #[test]
    fn input_stage_applies_when_it_shrinks() {
        let (mut req, counter) = fresh();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SetContent {
            name: "shrink".into(),
            text: "hi".into(),
        })];
        let out = run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied);
        assert_eq!(req.get_str("/messages/0/content"), Some("hi"));
        assert!(out.input_tokens_after < out.input_tokens_before);
    }

    #[test]
    fn input_stage_reverts_when_it_bloats() {
        let (mut req, counter) = fresh();
        let original = req.get_str("/messages/0/content").unwrap().to_string();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SetContent {
            name: "bloat".into(),
            text: "word ".repeat(80),
        })];
        let out = run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(!out.stages[0].applied);
        assert_eq!(out.stages[0].note.as_deref(), Some("no token reduction"));
        assert_eq!(req.get_str("/messages/0/content"), Some(original.as_str()));
    }

    #[test]
    fn output_shaping_stage_is_never_reverted_on_tokens() {
        let (mut req, counter) = fresh();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(AddSystem)];
        let out = run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "output-shaping must apply despite adding tokens"
        );
        assert_eq!(
            req.raw()
                .pointer("/messages/0/role")
                .and_then(Value::as_str),
            Some("system")
        );
    }

    #[test]
    fn erroring_stage_is_reverted_and_does_not_block() {
        let (mut req, counter) = fresh();
        let original = req.get_str("/messages/0/content").unwrap().to_string();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(Boom)];
        let out = run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(!out.stages[0].applied);
        assert!(out.stages[0].note.as_deref().unwrap().contains("error"));
        assert_eq!(req.get_str("/messages/0/content"), Some(original.as_str()));
    }
}
