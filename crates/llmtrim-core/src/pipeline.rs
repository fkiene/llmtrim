//! Sequential gated stage driver — the static fan-in pipeline.
//!
//! Runs each [`Transform`] in order, gating it per [`GateKind`], and accumulates
//! the rehydration plan plus a per-stage report. Token measurement uses the real
//! [`TokenCounter`] over the provider's content text segments.

use std::collections::HashMap;

use crate::gate::{GateKind, PlanEntry, Scope, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::quality_gate::{self, COVERAGE_THRESHOLD};
use crate::tokenizer::{TokenCounter, Tokens};

/// True if the stage left the `/tools` subtree byte-identical to `snapshot`, so the cached
/// tools count is still valid and the array need not be re-serialized + re-tokenized (P1).
/// Compares the actual subtree (not the stage's declared [`Scope`], which only promises
/// "content text unchanged", not "only `/tools` mutated").
fn tools_unchanged(req: &Request, snapshot: &Request) -> bool {
    req.raw().get("tools") == snapshot.raw().get("tools")
}

/// Stage names that drop/window query-relevant *content* and are therefore subject to
/// the quality gate as well as the token gate. The token gate can't tell these apart
/// from a beneficial cut — both reduce tokens — so coverage decides if the cut hurt.
///
/// Kept here (not on the stage types) because adding `Transform::quality_gated`
/// overrides to the stage modules is out of this change's scope; a stage may still
/// opt in by overriding [`Transform::quality_gated`] (OR'd with this list).
fn quality_gated_by_name(name: &str) -> bool {
    matches!(name, "retrieve" | "toolout")
}

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
    /// Tokens inside the frozen (cache-controlled) prefix — content the stages never touch
    /// by cache-zone discipline. `input − frozen` is the compressible surface, the honest
    /// denominator for "how much of what we CAN compress did we compress".
    pub frozen_input_tokens: Tokens,
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

/// Like [`count_content`] but memoizes per-segment counts across the pipeline run: a content
/// segment whose text is unchanged since a prior stage is summed from `cache` (a hash lookup)
/// instead of re-tokenized (a BPE pass). The win is multi-segment prompts — when a stage drops
/// or reorders whole segments (retrieve, dedup) the kept ones are reused, not re-tokenized. A
/// string is only cloned into the cache on a miss, i.e. when we have to tokenize it anyway.
fn count_content_cached(
    req: &Request,
    provider: &dyn Provider,
    counter: &dyn TokenCounter,
    cache: &mut HashMap<String, usize>,
) -> usize {
    provider
        .content_text_pointers(req)
        .iter()
        .filter_map(|p| req.get_str(p))
        .map(|s| match cache.get(s) {
            Some(&c) => c,
            None => {
                let c = counter.count(s);
                cache.insert(s.to_string(), c);
                c
            }
        })
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

/// Run `stages` over `req`, gating each one (token gate + quality gate on). The
/// request is mutated in place to its final compressed form.
///
/// The quality gate is **on** here — the safe default for the product promise: it only
/// ever *reverts* an over-aggressive compression, never breaks output, so leaving it on
/// can only protect the response. Use [`run_gated`] to disable it (the `quality_gate`
/// config knob routes through there).
pub fn run(
    req: &mut Request,
    provider: &dyn Provider,
    counter: &dyn TokenCounter,
    stages: &[Box<dyn Transform>],
) -> PipelineOutcome {
    run_gated(req, provider, counter, stages, true)
}

/// Like [`run`], but with the quality gate explicitly toggled. `quality_gate = false`
/// runs the token gate only (the pre-quality-gate behavior). Separate entry point so
/// [`run`]'s signature stays source-compatible for the many existing call sites.
pub fn run_gated(
    req: &mut Request,
    provider: &dyn Provider,
    counter: &dyn TokenCounter,
    stages: &[Box<dyn Transform>],
    quality_gate: bool,
) -> PipelineOutcome {
    let mut plan: Vec<PlanEntry> = Vec::new();
    let mut reports = Vec::with_capacity(stages.len());
    // Query terms for the quality gate (the question the compression must keep
    // answerable). Computed once from the incoming request — the live question is stable
    // across the content stages, and a stage that edits a short query segment is
    // structurally prevented from pruning it. Skipped entirely when the gate is off.
    let query = if quality_gate {
        quality_gate::query_terms(req, provider)
    } else {
        Vec::new()
    };
    // Opt-in per-stage wall-clock attribution (apply + re-count), to stderr. Set
    // `LLMTRIM_PROFILE=1` when benchmarking; zero cost otherwise (one env read per run).
    let profile = std::env::var_os("LLMTRIM_PROFILE").is_some();
    // Track content and tools token counts separately, carried between stages: a stage's
    // `before` is the prior stage's result, and after a stage we re-count ONLY the part it
    // can change (its `scope`). Most agent stages touch only `tools`, so the big content
    // text isn't re-tokenized each time. Same gating decisions, far fewer tokenizations —
    // the per-request hot path.
    // Per-segment token-count memo, reused across stages: a content segment whose text is
    // unchanged since a prior stage is summed from the cache, not re-tokenized.
    let mut seg_cache: HashMap<String, usize> = HashMap::new();
    let mut content = count_content_cached(req, provider, counter, &mut seg_cache);
    let mut tools = count_tools(req, counter);
    let input_tokens_before = content + tools;
    // Frozen-zone meter: size of the cache-controlled prefix the stages will skip. Counted
    // once up front (the zone is immutable by discipline); `seg_cache` is already warm from
    // the full content count, so this is hash lookups, not a second BPE pass.
    let frozen_input_tokens: usize = crate::cache_zone::frozen_pointers(req, provider)
        .iter()
        .filter_map(|p| req.get_str(p))
        .map(|s| {
            seg_cache
                .get(s)
                .copied()
                .unwrap_or_else(|| counter.count(s))
        })
        .sum();

    for stage in stages {
        let scope = stage.scope();
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
                    let new_content = match scope {
                        Scope::Tools => content,
                        Scope::Content | Scope::Both => {
                            count_content_cached(req, provider, counter, &mut seg_cache)
                        }
                    };
                    // Tools count is re-serialized + re-tokenized only when the `/tools`
                    // subtree actually changed (P1): a stage that rewrites only content keeps
                    // the cached tools count instead of paying a full tools BPE pass. Verified
                    // against the actual subtree regardless of the stage's declared scope —
                    // `Scope` only promises "content text unchanged", not "only /tools moved"
                    // (CacheStage declares `Tools` yet writes into `/system` and `/messages`).
                    let new_tools = if tools_unchanged(req, &snapshot) {
                        tools
                    } else {
                        count_tools(req, counter)
                    };
                    (new_content, new_tools)
                };
                let after = new_content + new_tools;
                if stage.gate_kind() == GateKind::InputTokens && after >= before {
                    // No net token win: revert (never block the user); counts unchanged.
                    *req = snapshot;
                    plan.truncate(plan_mark);
                    (false, before, Some("no token reduction".to_string()))
                } else if quality_gate
                    && !query.is_empty()
                    && stage.gate_kind() == GateKind::InputTokens
                    && scope != Scope::Tools
                    && (stage.quality_gated() || quality_gated_by_name(stage.name()))
                    && {
                        // Coverage of query-relevant source content surviving this cut. The
                        // token gate already accepted (it saved tokens); the question now is
                        // whether it saved them by deleting the answer. Source = pre-stage
                        // *context*, compressed = post-stage *context* (the question itself is
                        // excluded — it is pinned and always survives, so it must not satisfy
                        // its own coverage). Only computed for the handful of lossy content
                        // stages, and only when there is a distinct question to protect
                        // (`!query.is_empty()` — a monolithic prompt or a log dump has none,
                        // and blanket coverage would wrongly revert the pruning such a stage
                        // exists to do).
                        let source = quality_gate::context_text(&snapshot, provider);
                        let compressed = quality_gate::context_text(req, provider);
                        quality_gate::coverage(&source, &compressed, &query) < COVERAGE_THRESHOLD
                    }
                {
                    // The cut saved tokens but dropped too much query-relevant content:
                    // revert exactly like a token-gate failure (counts unchanged).
                    *req = snapshot;
                    plan.truncate(plan_mark);
                    (
                        false,
                        before,
                        Some("quality-gate reverted: coverage below threshold".to_string()),
                    )
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
        frozen_input_tokens: Tokens(frozen_input_tokens),
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

    /// Test transform (`Scope::Tools`): replace the whole `/tools` array with a smaller one.
    struct ShrinkTools;
    impl Transform for ShrinkTools {
        fn name(&self) -> &str {
            "shrink-tools"
        }
        fn gate_kind(&self) -> GateKind {
            GateKind::InputTokens
        }
        fn scope(&self) -> Scope {
            Scope::Tools
        }
        fn apply(
            &self,
            req: &mut Request,
            _provider: &dyn Provider,
            _plan: &mut Vec<PlanEntry>,
        ) -> anyhow::Result<()> {
            req.set("/tools", serde_json::json!([{"name": "f"}]));
            Ok(())
        }
    }

    fn fresh_with_tools() -> (Request, Box<dyn TokenCounter>) {
        let req = Request::parse(
            ProviderKind::OpenAi,
            r#"{"messages":[{"role":"user","content":"this is a fairly long original message about widgets and gadgets"}],"tools":[{"type":"function","function":{"name":"search_documents","description":"search a large corpus for relevant passages","parameters":{"q":"string"}}}]}"#,
        )
        .unwrap();
        (
            req,
            counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap(),
        )
    }

    /// P1: a content-only (Both-scope) stage keeps the cached tools count, and the reported
    /// total must still equal a full independent recount that includes the tools tokens.
    #[test]
    fn content_stage_preserves_tools_token_count() {
        let (mut req, counter) = fresh_with_tools();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SetContent {
            name: "shrink".into(),
            text: "hi".into(),
        })];
        let out = run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied);
        // The cached-tools path (P1) must produce the same number as a fresh full count.
        let independent = content_tokens(&req, &OpenAiProvider, counter.as_ref());
        assert_eq!(out.input_tokens_after.0, independent);
        // And the tools tokens are genuinely part of that total (non-trivial tools block).
        assert!(out.input_tokens_after.0 > count_content(&req, &OpenAiProvider, counter.as_ref()));
    }

    /// P1: a stage that actually mutates `/tools` triggers the recount, and the new total
    /// reflects the smaller tools array.
    #[test]
    fn tools_stage_recounts_when_tools_change() {
        let (mut req, counter) = fresh_with_tools();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(ShrinkTools)];
        let out = run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied);
        assert!(out.input_tokens_after < out.input_tokens_before);
        assert_eq!(
            out.input_tokens_after.0,
            content_tokens(&req, &OpenAiProvider, counter.as_ref())
        );
    }
}
