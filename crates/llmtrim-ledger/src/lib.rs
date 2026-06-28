//! SQLite data layer shared by `llmtrim-cli` and `llmtrim-tray`.
//!
//! `tracking` — the core compression ledger (Tracker, Record, BreakdownTurn, …).
//! `breakdown_db` — read-only query layer for the cost-breakdown view (BreakdownDb, SessionRow, …).

pub mod breakdown_db;
pub mod dashboard;
pub mod tracking;
