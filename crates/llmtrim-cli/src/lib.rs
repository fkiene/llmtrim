//! llmtrim CLI/proxy — the user-facing surface over the [`llmtrim_core`] engine.
//!
//! This crate owns everything the embeddable core deliberately excludes: the
//! command-line interface (`main.rs`), the MITM interceptor (`serve`), the local
//! daemon/autostart/setup machinery, the SQLite token ledger (`tracking`), the live
//! A/B benchmark + quality harness, and the terminal UI. The compression itself lives
//! in `llmtrim_core` and is called through its public `compress*` API.

pub mod autostart;
pub mod bench;
#[cfg(feature = "breakdown")]
pub mod breakdown;
#[cfg(feature = "intercept")]
pub mod codex_gateway;
pub mod compact;
pub mod daemon;
pub mod discover;
pub mod doctor;
pub mod ensure;
pub mod guard;
pub mod mcp;
pub mod monitor;
pub mod quality;
pub mod recall;
#[cfg(feature = "intercept")]
pub mod reroute;
#[cfg(feature = "intercept")]
pub mod route_agents;
pub mod serve;
pub mod setup;
pub mod statusline;
pub mod tracking;
pub mod transport;
pub mod tray;
pub mod ui;
pub mod update;
pub mod window_sub;
pub mod wrap;
