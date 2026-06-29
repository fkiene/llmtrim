//! Re-exports the breakdown query layer from `llmtrim-ledger`.
//!
//! All types and methods live in `crates/llmtrim-ledger`; this module keeps existing
//! `super::db::*` and `crate::breakdown::db::*` call sites working without modification.
pub use llmtrim_ledger::breakdown_db::*;
