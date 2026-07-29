//! Interactive cost-breakdown TUI.
//!
//! A tabbed terminal app over the proxy's token ledger: an **Overview** tab (the savings
//! dashboard), a **Sessions** tab (every session grouped agent → project → session with
//! token and dollar totals), a **Detail** drill-down (context-window occupancy and
//! per-source cost down to each MCP server), and a **Sub** tab for subscription routing
//! presets + tier maps. Launched by `status`/`monitor` on a TTY; piped or `--json`/`--csv`
//! invocations keep the plain snapshot.

pub mod app;
pub mod db;
pub mod export;
pub mod palette;
#[cfg(feature = "intercept")]
pub mod sub_panel;
pub mod tree;
