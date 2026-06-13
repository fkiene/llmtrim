//! Per-stage gating: a transform only sticks if it earns its keep.
//!
//! The rule, applied per stage: a transform that fails returns the raw input —
//! never block the user. An [`GateKind::InputTokens`] transform that errors *or* fails
//! to reduce net input tokens is reverted, leaving the request exactly as it was.
//! [`GateKind::OutputShaping`] transforms (Stage F) change request fields whose
//! payoff is on the response side, so they are not gated on input tokens — their
//! token win is validated out-of-band (input and output evals run
//! separately).

use anyhow::Result;

use crate::ir::Request;
use crate::provider::Provider;

/// A rehydration-plan entry a transform records so the response can be reversed.
/// Opaque JSON for now; stages define concrete entry shapes as needed.
pub type PlanEntry = serde_json::Value;

/// How a transform's effect is validated before it is allowed to stick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
    /// Must reduce net input tokens (measured on content text). Reverted otherwise.
    InputTokens,
    /// Shapes the request for output-side savings. Always applied (validated
    /// out-of-band); never reverted on input-token grounds.
    OutputShaping,
    /// Lossless structural change whose payoff is amortized/latent (e.g. cache
    /// breakpoints — the provider discount lands on a later call). Always applied;
    /// not gated on per-call input tokens.
    Structural,
}

/// Which part of the prompt a stage can change. Lets the gate re-count only the affected
/// part instead of re-tokenizing the whole prompt after every stage (the hot path).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// System + message text (`content_text_pointers`).
    Content,
    /// Tool / function schemas.
    Tools,
    /// Both — the conservative default (always re-counts everything).
    Both,
}

/// One deterministic transform stage.
pub trait Transform {
    /// Stable, short stage name (for reports and the savings ledger).
    fn name(&self) -> &str;

    /// How the pipeline decides whether to keep this stage's output.
    fn gate_kind(&self) -> GateKind;

    /// Which part of the prompt this stage can change (default: conservatively both). A
    /// narrower scope lets the gate skip re-tokenizing the unchanged part.
    fn scope(&self) -> Scope {
        Scope::Both
    }

    /// Whether this stage's output is subject to the **quality gate** (in addition to
    /// the token gate): after the token gate accepts a lossy *content* stage, the
    /// pipeline re-checks that query-relevant source content survived
    /// ([`crate::quality_gate::coverage`] ≥ [`crate::quality_gate::COVERAGE_THRESHOLD`])
    /// and reverts if it didn't — the token gate alone can't tell a cut saved tokens by
    /// deleting the answer.
    ///
    /// Default `false` (opt-in) keeps the trait source-compatible: a stage that doesn't
    /// override this is never quality-gated, so other modules' transforms compile and
    /// behave unchanged. The two lossy content-dropping stages this protects today
    /// (retrieve, toolout) are recognized by the pipeline by name (their modules are out
    /// of this change's scope), but any future stage may opt in by overriding this — the
    /// pipeline OR's the two. Only consulted for [`GateKind::InputTokens`] stages whose
    /// [`Scope`] includes content.
    fn quality_gated(&self) -> bool {
        false
    }

    /// Apply the transform in place. `provider` gives the stage structural access
    /// (content segments, output-control fields). A stage may push rehydration
    /// entries onto `plan`. On `Err`, the pipeline restores the pre-transform
    /// request and discards any plan entries this stage added (never block).
    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        plan: &mut Vec<PlanEntry>,
    ) -> Result<()>;
}
