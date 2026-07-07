//! Stage F — output control (request-shaping).
//!
//! These transforms change request fields/instructions whose payoff is on the
//! *response* side (fewer/cheaper output tokens), so they use the OutputShaping
//! gate (always applied; never reverted on input tokens). Their output savings are
//! validated out-of-band — recorded fixtures or the proxy phase — since input
//! and output compression are evaluated separately.
//!
//! The `terse` instruction — concise, full sentences; clean, low garble risk; ~73%
//! output cut in a live test. (`draft` below is a separate reasoning-scaffold tier.)

use anyhow::Result;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::{Provider, Role};
use crate::stages::tools::{detect_lang, is_first_turn};

/// Cap on the user-prose sample handed to `detect_lang`: a reliable detection needs only a
/// sentence or two, and agent requests can carry megabytes of pasted context. Enforced before
/// each copy so one huge block can't blow the budget (char-boundary-safe, like `stopword_set`).
const PROSE_SAMPLE_MAX_BYTES: usize = 4096;

/// Concatenate the request's user-turn text — the language signal for the reply-language
/// decision — up to [`PROSE_SAMPLE_MAX_BYTES`].
fn user_prose(req: &Request, provider: &dyn Provider) -> String {
    let mut prose = String::new();
    for ptr in provider.content_text_pointers(req) {
        if provider.role_at(req, &ptr) == Some(Role::User)
            && let Some(text) = req.get_str(&ptr)
        {
            let mut take = PROSE_SAMPLE_MAX_BYTES
                .saturating_sub(prose.len())
                .min(text.len());
            while take > 0 && !text.is_char_boundary(take) {
                take -= 1;
            }
            prose.push_str(&text[..take]);
            prose.push(' ');
            if prose.len() >= PROSE_SAMPLE_MAX_BYTES {
                break;
            }
        }
    }
    prose
}

/// `terse` tier: a small, fixed input cost for a real output-token reduction
/// (output tokens cost ~3–5× input).
// Instructions stay verbose on purpose: the bench showed a shorter instruction cuts a
// few input tokens but is LESS forceful → the model rambles → far more output tokens.
// Output costs ~3–5× input, so the instruction's small input cost buys a much larger
// output saving. Don't trade it away to flatter the input %.
pub const TERSE_INSTRUCTION: &str = include_str!("../../prompts/output_terse.txt");

/// Language-preservation clause appended to the primary shaping instruction. The injected
/// instructions are in English and land last, biasing the model to answer in English even
/// when the user wrote in another language; this one universal clause (no per-language
/// detection) corrects that for a handful of input tokens. Only ships when shaping is on.
pub const REPLY_LANGUAGE_CLAUSE: &str = " Reply in the user's language.";

/// `draft` tier: Chain-of-Draft — collapse the reasoning scaffold, not the prose
/// (arXiv:2502.18600). Targets reasoning-model output tokens, which concentrate in
/// the chain-of-thought.
pub const DRAFT_INSTRUCTION: &str = include_str!("../../prompts/output_draft.txt");

/// Compact-code output instruction: emit minified code (arXiv:2508.13666 reports
/// up to −36% output tokens with no correctness loss on capable models). Model-
/// gated — weak models can emit syntactically broken compact code.
pub const COMPACT_CODE_INSTRUCTION: &str = include_str!("../../prompts/output_compact_code.txt");

/// Soft prompt-side token budget (TALE zero-shot, arXiv:2412.18547). `{budget}` is
/// replaced with the cap; complements the hard `max_tokens` cap.
pub const TOKEN_BUDGET_TMPL: &str = include_str!("../../prompts/output_token_budget.txt");

/// Agent-loop frugality directive. Prose shaping is skipped on tool-call-shaped requests
/// (it can't shrink call arguments), but the *trajectory* — how many tool calls the agent
/// makes and how much it reads — is the real token sink in an agent loop. This steers
/// toward the fewest tool-use turns (batch independent calls into one turn, don't repeat a
/// call), the anti-pattern being a swarm of one-per-turn round-trips that each re-send the
/// cached prefix. Domain-neutral: it talks about tool calls, not files. Model-gated: only
/// harnesses that obey system-level steering respond. Its effect on the trajectory can only
/// be validated by a full-task agent bench (tokens AND task success), never a single turn.
pub const TOOLS_FRUGAL_INSTRUCTION: &str = include_str!("../../prompts/tools_frugal.txt");

/// Stable substring of [`TOOLS_FRUGAL_INSTRUCTION`] used to detect the directive already
/// sitting in the system block, so a repeated inject is a no-op (idempotent guard).
const TOOLS_FRUGAL_MARKER: &str = "fewest tool-use turns";

/// Anti-overthinking directive for quantized reasoning models (arXiv:2606.00206): post-training
/// quantization amplifies a tendency to reach a correct answer mid-reasoning and then restart
/// ("Wait", "But", "Actually…") instead of committing to it — up to 52% of quantized-model
/// failures are this "overthinking" pattern, not a capability loss. Live bench, this lever in
/// isolation (`llmtrim bench quality --corpus bench/data/gsm8k.jsonl --config <anti_overthink-
/// only.toml> --model openai/gpt-oss-20b --route wandb/fp4 --reasoning-effort medium`, n=40):
/// output tokens cut 62.0%, quality retention +0.0pp (unchanged, paired 95% CI ±10.2).
pub const ANTI_OVERTHINK_INSTRUCTION: &str =
    include_str!("../../prompts/output_anti_overthink.txt");

/// Stable substring of [`ANTI_OVERTHINK_INSTRUCTION`] for the idempotent guard. Deliberately NOT
/// one of the restart words the instruction names ("Wait"/"But"/"Actually") — those are common
/// enough to appear verbatim in an unrelated system prompt, which would falsely match and skip
/// injection. "commit to it immediately" is specific to this directive.
const ANTI_OVERTHINK_MARKER: &str = "commit to it immediately";

/// Output-control intensity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputLevel {
    Terse,
    Draft,
}

impl OutputLevel {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "draft" | "cod" => OutputLevel::Draft,
            _ => OutputLevel::Terse,
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            OutputLevel::Terse => TERSE_INSTRUCTION,
            OutputLevel::Draft => DRAFT_INSTRUCTION,
        }
    }
}

pub struct OutputControlStage {
    /// Inject the `level` prose-shaping instruction (terse / Chain-of-Draft) on non-tool
    /// requests. Off when the stage runs only for a sibling lever (`compact_code`,
    /// `frugal_tools`), so e.g. preset `frugal` steers the agent loop WITHOUT also forcing
    /// terse prose on every non-tool request — which would confound the directive's bench.
    pub output_control: bool,
    pub level: OutputLevel,
    /// If set and the request has no cap, impose this output-token cap.
    pub max_tokens: Option<u64>,
    /// If set, inject a *soft* token budget into the prompt ("answer within N
    /// tokens") — the prompt-side complement of the hard `max_tokens` cap
    /// (TALE zero-shot, arXiv:2412.18547).
    pub token_budget: Option<u64>,
    /// Instruct the model to emit minified code (arXiv:2508.13666). Model-gated.
    pub compact_code: bool,
    /// Inject the agent-loop frugality directive on tool-call-shaped requests — the one
    /// request shape prose shaping skips. Opt-in, model-gated; ship only behind a full
    /// agent-bench that confirms total tokens fall AND task success holds.
    pub frugal_tools: bool,
    /// Inject the anti-overthinking directive (see [`ANTI_OVERTHINK_INSTRUCTION`]) on prose
    /// requests that explicitly declare BOTH a quantized serving tier and a reasoning pass.
    /// Gated on wire fields, not a model-name list — same policy as `reasoning_model_request`.
    pub anti_overthink: bool,
}

impl Transform for OutputControlStage {
    fn name(&self) -> &str {
        "output-control"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::OutputShaping
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        // Tool-call-shaped request: the expected answer is a function-call payload, not
        // prose. Prose-shaping instructions can't shrink call arguments — the live A/B's
        // agent corpus saves 0.0% output tokens with them — so on the most-resent request
        // shape (agent loops) they are pure input cost. Skip them; the hard `max_tokens`
        // cap below stays (it costs nothing). `tool_choice: "none"` means the model is
        // told NOT to call, so the answer is prose again and shaping applies.
        if !tool_call_shaped(req) {
            // Only inject the prose-shaping instruction when output control is actually on. The
            // stage also runs for the sibling levers (`compact_code`, `frugal_tools`), and those
            // must NOT drag terse/draft prose in with them — preset `frugal` isolates the
            // agent-loop directive, so forcing terse on non-tool requests would confound it.
            if self.output_control {
                // The clause only earns its tokens when the user wrote in a non-English language;
                // an English (or too-short-to-detect) prompt already answers in English, so skip
                // it there and add nothing. `detect_lang` returns `Some` only on a reliable
                // detection, so ambiguous/short prose keeps the clause (cheap, safe).
                let non_english =
                    detect_lang(&user_prose(req, provider)) != Some(whatlang::Lang::Eng);
                let instruction = if non_english {
                    format!("{}{}", self.level.instruction(), REPLY_LANGUAGE_CLAUSE)
                } else {
                    self.level.instruction().to_string()
                };
                provider.add_system_instruction(req, &instruction);
            }
            // Soft numeric token budgets ("answer within N tokens") FAIL on reasoning
            // models: the batch-prompting overthinking study (arXiv:2511.04108, 2025)
            // found explicit thinking-budget instructions are ignored on DeepSeek-R1 /
            // OpenAI o1, and when followed they cut accuracy. They stay valid only for
            // NON-reasoning models (TALE, arXiv:2412.18547). The terse/draft style
            // instruction above and compact_code below are prose/scaffold shaping, not a
            // numeric cap — Chain-of-Draft (arXiv:2502.18600) validated terse/draft on
            // gpt-oss, a reasoning model — so they stay for all models. Skip ONLY the
            // soft budget here; the hard `max_tokens` cap below is server-enforced, not an
            // instruction the model can ignore, so it stays too.
            if let Some(budget) = self.token_budget
                && !reasoning_model_request(req)
            {
                provider.add_system_instruction(
                    req,
                    &TOKEN_BUDGET_TMPL.replace("{budget}", &budget.to_string()),
                );
            }
            if self.compact_code {
                provider.add_system_instruction(req, COMPACT_CODE_INSTRUCTION);
            }
        }
        // Independent of the prose branch above (its own gate on `tool_call_shaped`), so it and the
        // anti-overthink block below both get a clean shot at a tool-shaped request.
        if self.frugal_tools
            && tool_call_shaped(req)
            && is_first_turn(req)
            && crate::capability::model_honors_steering(req.model_id().unwrap_or(""))
            && !frugal_directive_present(req, provider)
        {
            // Tool-call-shaped (agent loop): prose shaping is skipped above because it can't
            // shrink call arguments, but the loop's real token cost is the trajectory — how
            // many tool-use turns the agent takes. Steer that toward the fewest turns (batch
            // independent calls into one turn, don't repeat a call).
            //
            // FIRST TURN ONLY, and idempotent: re-injecting the directive on every iteration
            // was pure recurring input cost (the bench's inert +250..+1100 tok tasks) with no
            // trajectory change, and it churned the provider cache prefix on later turns. The
            // wasteful exploration decisions this targets happen early, so a single turn-1
            // inject captures the steer at a fraction of the cost. `is_first_turn` treats any
            // already-invoked tool as a live loop; the presence check stops a double-inject if
            // the client's history already carries the directive.
            //
            // Model-gated (`model_honors_steering`): only capable models act on the steer; cheap
            // models ignore it and just pay the directive's input cost, so they are skipped. See
            // `crate::capability`. Opt-out: an unknown model id still injects.
            provider.add_system_instruction(req, TOOLS_FRUGAL_INSTRUCTION);
        }
        // Anti-overthinking directive (arXiv:2606.00206): a reasoning pass restarting mid-CoT
        // ("Wait", "Actually…") after it already had the answer. Runs on BOTH request shapes,
        // outside the prose/tool branch above: a thinking model runs a CoT before it emits tool
        // calls too, so an agent turn overthinks the same way a prose turn does — and Claude Code
        // turns are always tool-shaped, so a prose-only gate would never reach them.
        //
        // The trigger is reasoning *capability*, not a wire field: Claude Code sends the bare id
        // (`claude-sonnet-5`) with no `thinking`/`reasoning` on the wire, so `reasoning_model_request`
        // alone can't see it. `model_is_reasoning_capable` reads the models.dev `reasoning` flag
        // from the embedded snapshot — opt-in, so a non-reasoning model gets nothing. Either
        // signal qualifies: an explicit wire reasoning field OR a known reasoning-mode model.
        //
        // First-turn-only ON THE AGENT PATH, like the frugality directive above: there the two
        // share the system block, so injecting anti-overthink on later turns (when frugal is
        // already gone) would diverge the cached prefix turn-to-turn and churn the provider cache.
        // On the prose path there is no frugal to stay in lockstep with and no agent-loop cache to
        // protect, so let every reasoning turn (incl. a multi-turn chat) get the steer.
        //
        // Accepted risk, pending a quality bench: benched only on a quantized reasoning model
        // (gpt-oss-20b); on full-precision flagships the "commit, don't reconsider" steer is
        // unproven and reconsideration can be productive. Kept armed by choice.
        if self.anti_overthink
            && (!tool_call_shaped(req) || is_first_turn(req))
            && (reasoning_model_request(req)
                || crate::capability::model_is_reasoning_capable(req.model_id().unwrap_or("")))
            && !anti_overthink_present(req, provider)
        {
            provider.add_system_instruction(req, ANTI_OVERTHINK_INSTRUCTION);
        }
        if let Some(cap) = self.max_tokens
            && provider.max_tokens(req).is_none()
        {
            provider.set_max_tokens(req, cap);
        }
        Ok(())
    }
}

/// True when the request carries a non-empty `tools` array and tool calling isn't
/// disabled (`tool_choice: "none"`) — i.e. the answer is expected to be a tool call.
/// Shared shape across OpenAI and Anthropic bodies (both use `tools`/`tool_choice`).
fn tool_call_shaped(req: &Request) -> bool {
    let raw = req.raw();
    raw.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty())
        && raw.get("tool_choice").and_then(Value::as_str) != Some("none")
}

/// True when the frugality directive is already present in the request's system prose — the
/// idempotent guard against a second inject when the client's resent history carries it forward.
///
/// The system prose lives in a `system`-role message (OpenAI Chat) OR in a top-level system
/// field (Anthropic `/system`, Google `/systemInstruction`, OpenAI Responses `/instructions`).
/// `role_at` returns `None` for those top-level fields by contract (no enclosing turn), so the
/// guard must accept BOTH `Some(System)` and `None` — matching only `Some(System)` silently
/// missed every provider that keeps its system prompt top-level, defeating the guard there.
fn frugal_directive_present(req: &Request, provider: &dyn Provider) -> bool {
    provider.content_text_pointers(req).iter().any(|ptr| {
        matches!(provider.role_at(req, ptr), Some(Role::System) | None)
            && req
                .get_str(ptr)
                .is_some_and(|t| t.contains(TOOLS_FRUGAL_MARKER))
    })
}

/// True when the anti-overthink directive is already present in the request's system prose —
/// mirrors [`frugal_directive_present`]'s idempotent guard and its both-shapes rationale.
fn anti_overthink_present(req: &Request, provider: &dyn Provider) -> bool {
    provider.content_text_pointers(req).iter().any(|ptr| {
        matches!(provider.role_at(req, ptr), Some(Role::System) | None)
            && req
                .get_str(ptr)
                .is_some_and(|t| t.contains(ANTI_OVERTHINK_MARKER))
    })
}

/// True when the request has opted into a reasoning pass — detected ONLY from explicit,
/// provider-set request fields, never from model-id lists (model names are not universal:
/// any hardcoded family table is wrong for the next provider and rots as models ship):
///
/// - `reasoning`         — OpenRouter / OpenAI Responses reasoning config object.
/// - `reasoning_effort`  — OpenAI effort knob.
/// - `thinking`          — Anthropic extended-thinking block.
///
/// Soft numeric token budgets are counter-productive on reasoning passes
/// (arXiv:2511.04108: ignored, or accuracy drops when followed); the caller skips that
/// one lever when this returns true. Known limitation, accepted deliberately: a
/// reasoning-by-default model invoked WITHOUT any of these fields is not detected — the
/// soft budget then ships exactly as it does today (status quo; per the paper it is
/// most likely ignored). That trade keeps detection universal and maintenance-free.
fn reasoning_model_request(req: &Request) -> bool {
    let raw = req.raw();
    raw.get("reasoning").is_some()
        || raw.get("reasoning_effort").is_some()
        || raw.get("thinking").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::json;

    fn run_one(body: Value, stage: OutputControlStage) -> Request {
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(stage)];
        let _ = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        req
    }

    /// Run the stage against an explicit provider/kind (for the non-OpenAI-Chat wire shapes:
    /// Anthropic top-level `system`, OpenAI Responses `instructions`).
    fn run_with(
        kind: ProviderKind,
        provider: &dyn Provider,
        body: Value,
        stage: OutputControlStage,
    ) -> Request {
        let mut req = Request::from_value(kind, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(stage)];
        let _ = pipeline::run(&mut req, provider, counter.as_ref(), &stages);
        req
    }

    fn frugal_stage() -> OutputControlStage {
        OutputControlStage {
            output_control: false,
            level: OutputLevel::Terse,
            max_tokens: None,
            token_budget: None,
            compact_code: false,
            frugal_tools: true,
            anti_overthink: false,
        }
    }

    #[test]
    fn level_parses() {
        assert_eq!(OutputLevel::parse("draft"), OutputLevel::Draft);
        assert_eq!(OutputLevel::parse("terse"), OutputLevel::Terse);
        assert_eq!(OutputLevel::parse("ultra"), OutputLevel::Terse);
        assert_eq!(OutputLevel::parse("whatever"), OutputLevel::Terse);
    }

    #[test]
    fn draft_injects_chain_of_draft() {
        let req = run_one(
            json!({"messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Draft,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let sys = req.get_str("/messages/0/content").unwrap();
        assert!(sys.contains("draft") && sys.contains("step"));
    }

    #[test]
    fn token_budget_injects_soft_limit() {
        let req = run_one(
            json!({"messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: Some(120),
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let joined: String = req
            .raw()
            .pointer("/messages")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect();
        assert!(joined.contains("120 tokens"), "soft budget injected");
    }

    #[test]
    fn tool_call_request_skips_prose_shaping_but_keeps_cap() {
        // tools present + tool_choice auto ⇒ the answer is a function call: no terse/budget
        // instruction (pure input cost, 0% output saving on the agent corpus), but the free
        // hard cap still applies.
        let req = run_one(
            json!({"messages":[{"role":"user","content":"book a flight"}],
                   "tools":[{"type":"function","function":{"name":"book","parameters":{}}}],
                   "tool_choice":"auto"}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: Some(900),
                token_budget: Some(120),
                compact_code: true,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let joined: String = req
            .raw()
            .pointer("/messages")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect();
        assert!(
            !joined.contains("concise") && !joined.contains("120 tokens"),
            "no prose-shaping instructions on a tool-call request: {joined}"
        );
        assert_eq!(
            req.raw()
                .get("max_completion_tokens")
                .and_then(Value::as_u64),
            Some(900),
            "hard cap still set (free)"
        );
    }

    #[test]
    fn tool_choice_none_restores_prose_shaping() {
        // tools present but calling disabled ⇒ the answer is prose; shaping applies again.
        let req = run_one(
            json!({"messages":[{"role":"user","content":"book a flight"}],
                   "tools":[{"type":"function","function":{"name":"book","parameters":{}}}],
                   "tool_choice":"none"}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let sys = req.get_str("/messages/0/content").unwrap();
        assert!(sys.contains("concise"), "prose shaping applies: {sys}");
    }

    #[test]
    fn frugal_tools_injects_on_tool_call_request_only() {
        // On a tool-call-shaped request, prose shaping is skipped but the frugality directive
        // fires — it targets the agent trajectory, not response prose.
        let req = run_one(
            json!({"messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: true,
                anti_overthink: false,
            },
        );
        let joined = joined_content(&req);
        assert!(
            joined.contains("fewest tool-use turns") && !joined.contains("concise"),
            "frugal directive fires on tool-call shape, prose shaping stays skipped: {joined}"
        );

        // On a plain prose request, frugal_tools stays silent (prose shaping owns that shape).
        let prose = run_one(
            json!({"messages":[{"role":"user","content":"explain this"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: true,
                anti_overthink: false,
            },
        );
        let pj = joined_content(&prose);
        assert!(
            pj.contains("concise") && !pj.contains("fewest tool-use turns"),
            "no frugal directive on a prose request: {pj}"
        );
    }

    #[test]
    fn frugal_tools_gated_by_model_capability() {
        // Same first-turn tool-call request, two models: a weak model (below the LMArena bar)
        // must NOT get the directive — it ignores the steer and would only pay the input cost —
        // while a capable model does. Wires `crate::capability` into the stage.
        let body = |model: &str| {
            json!({"model": model,
                   "messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"})
        };
        let weak = run_one(body("gpt-4o-mini"), frugal_stage());
        assert!(
            !joined_content(&weak).contains("fewest tool-use turns"),
            "weak model is gated out of the frugal directive: {}",
            joined_content(&weak)
        );
        let capable = run_one(body("claude-opus-4-8"), frugal_stage());
        assert!(
            joined_content(&capable).contains("fewest tool-use turns"),
            "capable model still gets the directive: {}",
            joined_content(&capable)
        );
    }

    #[test]
    fn frugal_gate_reads_gemini_model_from_url_hint() {
        // Gemini carries the model in the URL, not the body, so the gate reads it from the
        // out-of-band hint the proxy sets. A weak Gemini tier must be gated out; a capable one
        // still gets the directive. Guards the provider whose body-only lookup would otherwise
        // return "" and wrongly inject for every model.
        use crate::provider::GoogleProvider;
        let run_gemini = |model: &str| -> Request {
            let mut req = Request::from_value(
                ProviderKind::Google,
                json!({"contents":[{"role":"user","parts":[{"text":"find the bug"}]}],
                       "tools":[{"functionDeclarations":[{"name":"grep"}]}]}),
            );
            req.set_model_hint(Some(model));
            let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
            let stages: Vec<Box<dyn Transform>> = vec![Box::new(frugal_stage())];
            let _ = pipeline::run(&mut req, &GoogleProvider, counter.as_ref(), &stages);
            req
        };
        let has_directive = |req: &Request| {
            GoogleProvider.content_text_pointers(req).iter().any(|p| {
                req.get_str(p)
                    .is_some_and(|t| t.contains(TOOLS_FRUGAL_MARKER))
            })
        };
        assert!(
            !has_directive(&run_gemini("gemini-2.0-flash")),
            "weak Gemini tier (URL model, below the bar) is gated out"
        );
        assert!(
            has_directive(&run_gemini("gemini-3-pro")),
            "capable Gemini tier still gets the directive via the URL-model hint"
        );
    }

    #[test]
    fn frugal_tools_skips_past_first_turn() {
        // Later turns of an agent loop (a tool already invoked in history) must NOT re-inject:
        // the directive is first-turn-only to avoid recurring cost and cache churn.
        let req = run_one(
            json!({"messages":[
                {"role":"user","content":"find the bug"},
                {"role":"assistant","content":null,
                 "tool_calls":[{"id":"c1","type":"function",
                   "function":{"name":"grep","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"match at line 4"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: true,
                anti_overthink: false,
            },
        );
        assert!(
            !joined_content(&req).contains("fewest tool-use turns"),
            "no re-inject once the loop is live: {}",
            joined_content(&req)
        );
    }

    #[test]
    fn frugal_tools_idempotent_when_already_present() {
        // If the directive already sits in a system turn (client resent history carrying it),
        // don't add a second copy.
        let req = run_one(
            json!({"messages":[
                {"role":"system","content":TOOLS_FRUGAL_INSTRUCTION},
                {"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: true,
                anti_overthink: false,
            },
        );
        let hits = joined_content(&req)
            .matches("fewest tool-use turns")
            .count();
        assert_eq!(hits, 1, "directive present exactly once, not duplicated");
    }

    #[test]
    fn frugal_marker_is_substring_of_the_prompt() {
        // The idempotent guard matches TOOLS_FRUGAL_MARKER against the injected prompt. If the
        // prompt text is reworded and the marker isn't, the guard silently no-ops and the
        // directive double-injects. Pin the invariant here so any drift fails a fast unit test.
        assert!(
            TOOLS_FRUGAL_INSTRUCTION.contains(TOOLS_FRUGAL_MARKER),
            "marker {TOOLS_FRUGAL_MARKER:?} must be a substring of the prompt {TOOLS_FRUGAL_INSTRUCTION:?}"
        );
    }

    #[test]
    fn frugal_alone_does_not_leak_terse_on_prose() {
        // Preset `frugal` runs the stage with output_control OFF to isolate the agent-loop
        // directive. A plain prose request must then get NEITHER the frugal directive (wrong
        // shape) NOR the terse prose instruction — otherwise the directive's bench is confounded.
        let req = run_one(
            json!({"messages":[{"role":"user","content":"explain this"}]}),
            frugal_stage(),
        );
        let joined = joined_content(&req);
        assert!(
            !joined.contains("concise") && !joined.contains("fewest tool-use turns"),
            "frugal-only stage stays silent on a prose request: {joined}"
        );
    }

    #[test]
    fn frugal_idempotent_on_anthropic_top_level_system() {
        // Anthropic keeps the system prompt in a top-level `system` field, where `role_at`
        // returns None. The guard must still recognize the directive there and not double-inject.
        let req = run_with(
            ProviderKind::Anthropic,
            &crate::provider::AnthropicProvider,
            json!({"system": TOOLS_FRUGAL_INSTRUCTION,
                   "messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"name":"grep","input_schema":{"type":"object"}}]}),
            frugal_stage(),
        );
        let system = req
            .raw()
            .get("system")
            .and_then(Value::as_str)
            .unwrap_or("");
        assert_eq!(
            system.matches("fewest tool-use turns").count(),
            1,
            "directive present exactly once in Anthropic system, not duplicated: {system}"
        );
    }

    #[test]
    fn frugal_injects_on_anthropic_first_turn() {
        // First-turn Anthropic tool-call request with no directive yet: it must be injected into
        // the top-level `system` field (the arm that the broken guard would otherwise re-fire).
        let req = run_with(
            ProviderKind::Anthropic,
            &crate::provider::AnthropicProvider,
            json!({"system":"You are a helpful assistant.",
                   "messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"name":"grep","input_schema":{"type":"object"}}]}),
            frugal_stage(),
        );
        let system = req
            .raw()
            .get("system")
            .and_then(Value::as_str)
            .unwrap_or("");
        assert!(
            system.contains("fewest tool-use turns"),
            "directive injected into Anthropic system on first turn: {system}"
        );
    }

    #[test]
    fn frugal_idempotent_on_responses_instructions() {
        // OpenAI Responses carries the system prompt in top-level `instructions` (role_at None).
        let req = run_with(
            ProviderKind::OpenAi,
            &OpenAiProvider,
            json!({"instructions": TOOLS_FRUGAL_INSTRUCTION,
                   "input":[{"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","name":"grep","parameters":{}}]}),
            frugal_stage(),
        );
        let instr = req
            .raw()
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        assert_eq!(
            instr.matches("fewest tool-use turns").count(),
            1,
            "directive present exactly once in Responses instructions, not duplicated: {instr}"
        );
    }

    #[test]
    fn terse_injects_concise() {
        let req = run_one(
            json!({"messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let sys = req.get_str("/messages/0/content").unwrap();
        assert!(sys.contains("concise"));
        assert!(
            sys.contains("Reply in the user's language."),
            "language-preservation clause rides the shaping instruction: {sys}"
        );
        assert_eq!(
            req.raw()
                .pointer("/messages/0/role")
                .and_then(Value::as_str),
            Some("system")
        );
    }

    #[test]
    fn non_english_prompt_gets_language_clause() {
        let req = run_one(
            json!({"messages":[{"role":"user",
                "content":"Peux-tu m'expliquer comment fonctionne ce module de compression ?"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let sys = req.get_str("/messages/0/content").unwrap();
        assert!(
            sys.contains("Reply in the user's language."),
            "non-English prompt keeps the language clause: {sys}"
        );
    }

    #[test]
    fn english_prompt_skips_language_clause() {
        let req = run_one(
            json!({"messages":[{"role":"user",
                "content":"Can you explain how this compression module works under the hood?"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let sys = req.get_str("/messages/0/content").unwrap();
        assert!(sys.contains("concise"), "shaping still applies: {sys}");
        assert!(
            !sys.contains("Reply in the user's language."),
            "English prompt pays no clause tokens: {sys}"
        );
    }

    #[test]
    fn user_prose_caps_a_huge_block_without_panicking() {
        // A single multi-megabyte user turn must not be copied whole, and slicing a
        // multibyte-char block must stay on a char boundary.
        let huge = "é".repeat(1_000_000); // 2 MB, 2-byte chars
        let req = Request::from_value(
            ProviderKind::OpenAi,
            json!({"messages": [{"role": "user", "content": huge}]}),
        );
        let sample = user_prose(&req, &OpenAiProvider);
        assert!(
            sample.len() <= PROSE_SAMPLE_MAX_BYTES + 1,
            "sample bounded: {}",
            sample.len()
        );
    }

    #[test]
    fn sets_max_tokens_only_when_absent() {
        let req = run_one(
            json!({"messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: Some(256),
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        assert_eq!(OpenAiProvider.max_tokens(&req), Some(256));

        let req2 = run_one(
            json!({"max_tokens":99,"messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: Some(256),
                token_budget: None,
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        assert_eq!(
            OpenAiProvider.max_tokens(&req2),
            Some(99),
            "must not overwrite a caller-set cap"
        );
    }

    fn joined_content(req: &Request) -> String {
        req.raw()
            .pointer("/messages")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect()
    }

    #[test]
    fn reasoning_request_skips_soft_budget_keeps_terse_and_cap() {
        // arXiv:2511.04108: soft numeric budgets are ignored / hurt on reasoning passes.
        // Skip ONLY the soft budget; terse stays (Chain-of-Draft validates it on reasoning
        // models) and the server-enforced hard cap stays. Detection is by the explicit
        // `reasoning` request field — never by model-id lists (not universal).
        let req = run_one(
            json!({"model":"deepseek/deepseek-r1","reasoning":{"effort":"high"},
                   "messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: Some(256),
                token_budget: Some(120),
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let joined = joined_content(&req);
        assert!(
            !joined.contains("120 tokens"),
            "soft budget must be skipped on a reasoning model: {joined}"
        );
        assert!(
            joined.contains("concise"),
            "terse instruction still injected: {joined}"
        );
        assert_eq!(OpenAiProvider.max_tokens(&req), Some(256), "hard cap stays");
    }

    #[test]
    fn reasoning_field_skips_soft_budget() {
        // The `reasoning` request field alone marks a reasoning request, regardless of model.
        let req = run_one(
            json!({"model":"some-model","reasoning":{"effort":"low"},
                   "messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: Some(120),
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        assert!(!joined_content(&req).contains("120 tokens"));
    }

    #[test]
    fn non_reasoning_model_still_gets_soft_budget() {
        // Regression guard: a plain chat model must keep the TALE soft budget.
        let req = run_one(
            json!({"model":"gpt-4o-mini",
                   "messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: Some(120),
                compact_code: false,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        assert!(
            joined_content(&req).contains("120 tokens"),
            "soft budget must still be injected on a non-reasoning model"
        );
    }

    fn req_with(body: Value) -> Request {
        Request::from_value(ProviderKind::OpenAi, body)
    }

    #[test]
    fn detects_reasoning_request_fields() {
        assert!(reasoning_model_request(&req_with(
            json!({"model":"x","reasoning":{"effort":"low"}})
        )));
        assert!(reasoning_model_request(&req_with(
            json!({"model":"x","reasoning_effort":"high"})
        )));
        // Anthropic extended-thinking block.
        assert!(reasoning_model_request(&req_with(
            json!({"model":"claude-3-7-sonnet","thinking":{"type":"enabled","budget_tokens":1024}})
        )));
    }

    #[test]
    fn model_id_alone_never_marks_reasoning() {
        // Detection is fields-only by design: model names are not universal, so NO id —
        // however reasoning-flavored it looks — flips the guard without an explicit field.
        for id in [
            "deepseek/deepseek-r1",
            "o1-mini",
            "openai/gpt-5",
            "qwen/qwq-32b",
            "gpt-4o",
            "phi-4",
            "solar-pro",
        ] {
            assert!(
                !reasoning_model_request(&req_with(json!({"model": id}))),
                "{id}: id-based detection must never fire (fields-only)"
            );
        }
        // And with no model at all.
        assert!(!reasoning_model_request(&req_with(
            json!({"messages":[{"role":"user","content":"hi"}]})
        )));
    }

    #[test]
    fn compact_code_injects_instruction() {
        let req = run_one(
            json!({"messages":[{"role":"user","content":"hi"}]}),
            OutputControlStage {
                output_control: true,
                level: OutputLevel::Terse,
                max_tokens: None,
                token_budget: None,
                compact_code: true,
                frugal_tools: false,
                anti_overthink: false,
            },
        );
        let joined: String = req
            .raw()
            .pointer("/messages")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect();
        assert!(
            joined.contains("minified"),
            "compact-code instruction injected"
        );
    }

    fn anti_overthink_stage() -> OutputControlStage {
        OutputControlStage {
            output_control: false,
            level: OutputLevel::Terse,
            max_tokens: None,
            token_budget: None,
            compact_code: false,
            frugal_tools: false,
            anti_overthink: true,
        }
    }

    #[test]
    fn anti_overthink_marker_is_substring_of_the_prompt() {
        // Same invariant as `frugal_marker_is_substring_of_the_prompt`: if the prompt is
        // reworded and the marker isn't, the idempotent guard silently no-ops and double-injects.
        assert!(
            ANTI_OVERTHINK_INSTRUCTION.contains(ANTI_OVERTHINK_MARKER),
            "marker {ANTI_OVERTHINK_MARKER:?} must be a substring of the prompt {ANTI_OVERTHINK_INSTRUCTION:?}"
        );
    }

    #[test]
    fn anti_overthink_injects_on_any_reasoning_pass() {
        // Reasoning declared (with or without a quantized tier) → the directive fires. The gate
        // is the reasoning signal alone, so full-precision reasoning passes (e.g. sonnet with
        // `thinking`) get it too.
        for req in [
            json!({"provider":{"quantizations":["fp4"]},
                   "reasoning":{"effort":"medium"},
                   "messages":[{"role":"user","content":"what is 2+2?"}]}),
            json!({"reasoning":{"effort":"medium"},
                   "messages":[{"role":"user","content":"what is 2+2?"}]}),
            json!({"thinking":{"type":"enabled","budget_tokens":1024},
                   "messages":[{"role":"user","content":"what is 2+2?"}]}),
        ] {
            let out = run_one(req, anti_overthink_stage());
            assert!(
                joined_content(&out).contains(ANTI_OVERTHINK_MARKER),
                "reasoning pass gets the directive: {}",
                joined_content(&out)
            );
        }

        // No reasoning field: the model has no CoT to restart, so the directive must NOT fire.
        let no_reasoning = json!({"provider":{"quantizations":["fp4"]},
               "messages":[{"role":"user","content":"what is 2+2?"}]});
        let req = run_one(no_reasoning, anti_overthink_stage());
        assert!(
            !joined_content(&req).contains(ANTI_OVERTHINK_MARKER),
            "non-reasoning request stays silent: {}",
            joined_content(&req)
        );
    }

    #[test]
    fn anti_overthink_fires_on_tool_call_shaped_reasoning_requests() {
        // A thinking model runs a CoT before it emits tool calls, so an agent turn overthinks the
        // same way a prose turn does — and Claude Code turns are always tool-shaped. The directive
        // must ride the tool-call branch too (it did NOT before this change).
        let req = run_one(
            json!({"reasoning":{"effort":"medium"},
                   "messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            anti_overthink_stage(),
        );
        assert!(
            joined_content(&req).contains(ANTI_OVERTHINK_MARKER),
            "anti-overthink directive rides a tool-call-shaped reasoning request: {}",
            joined_content(&req)
        );
    }

    #[test]
    fn anti_overthink_detects_reasoning_model_by_id_without_a_wire_field() {
        // The Claude Code case: bare model id, no `thinking`/`reasoning` on the wire. The directive
        // must still fire (model-capability signal), on a tool-shaped agent turn.
        let req = run_one(
            json!({"model":"claude-sonnet-5",
                   "messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            anti_overthink_stage(),
        );
        assert!(
            joined_content(&req).contains(ANTI_OVERTHINK_MARKER),
            "anti-overthink fires on a known reasoning model with no wire field: {}",
            joined_content(&req)
        );

        // A non-reasoning model with no wire field must stay silent (opt-in, no misapplication).
        let req = run_one(
            json!({"model":"gpt-4o-mini",
                   "messages":[{"role":"user","content":"find the bug"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            anti_overthink_stage(),
        );
        assert!(
            !joined_content(&req).contains(ANTI_OVERTHINK_MARKER),
            "non-reasoning model stays silent: {}",
            joined_content(&req)
        );
    }

    #[test]
    fn anti_overthink_first_turn_gate_applies_only_to_the_agent_path() {
        // Multi-turn PROSE chat on a reasoning model (no tools, so not first-turn): still injects —
        // no agent-loop cache to protect, every reasoning turn gets the steer.
        let chat = run_one(
            json!({"model":"claude-sonnet-5",
                   "messages":[
                       {"role":"user","content":"hi"},
                       {"role":"assistant","content":"hello"},
                       {"role":"user","content":"what is 17*23?"}]}),
            anti_overthink_stage(),
        );
        assert!(
            joined_content(&chat).contains(ANTI_OVERTHINK_MARKER),
            "multi-turn prose chat keeps the directive: {}",
            joined_content(&chat)
        );

        // Tool-shaped LATER turn (a tool already invoked): suppressed, to keep the agent loop's
        // cached prefix byte-stable in lockstep with the first-turn-only frugality directive.
        let later_agent_turn = run_one(
            json!({"model":"claude-sonnet-5",
                   "messages":[
                       {"role":"user","content":"find the bug"},
                       {"role":"assistant","tool_calls":[
                           {"id":"c1","type":"function","function":{"name":"grep","arguments":"{}"}}]},
                       {"role":"tool","tool_call_id":"c1","content":"no match"}],
                   "tools":[{"type":"function","function":{"name":"grep","parameters":{}}}],
                   "tool_choice":"auto"}),
            anti_overthink_stage(),
        );
        assert!(
            !joined_content(&later_agent_turn).contains(ANTI_OVERTHINK_MARKER),
            "later agent turn stays silent (first-turn-only on the tool path): {}",
            joined_content(&later_agent_turn)
        );
    }

    #[test]
    fn anti_overthink_idempotent_when_already_present() {
        let req = run_one(
            json!({"provider":{"quantizations":["fp4"]},
                   "reasoning":{"effort":"medium"},
                   "messages":[
                       {"role":"system","content":ANTI_OVERTHINK_INSTRUCTION},
                       {"role":"user","content":"what is 2+2?"}]}),
            anti_overthink_stage(),
        );
        let hits = joined_content(&req).matches(ANTI_OVERTHINK_MARKER).count();
        assert_eq!(hits, 1, "directive present exactly once, not duplicated");
    }

    #[test]
    fn agent_rag_code_aggressive_presets_enable_anti_overthink() {
        for p in ["agent", "rag", "code", "aggressive"] {
            assert!(
                crate::config::DenseConfig::preset(p)
                    .unwrap()
                    .output_anti_overthink,
                "{p} should enable the anti-overthink lever"
            );
        }
        for p in ["safe", "lossless", "frugal"] {
            assert!(
                !crate::config::DenseConfig::preset(p)
                    .unwrap()
                    .output_anti_overthink,
                "{p} must stay silent — no behavioral directive"
            );
        }
    }
}
