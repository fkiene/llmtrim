//! Re-exports the ledger data layer from `llmtrim-ledger`.
//!
//! All types and functions live in `crates/llmtrim-ledger`; this module keeps existing
//! `crate::tracking::*` call sites working without modification.
pub use llmtrim_ledger::tracking::*;
