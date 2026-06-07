//! Run the interceptor at login, via the cross-platform `auto-launch` crate: Windows
//! registry run-key, macOS launchd agent, Linux XDG `.desktop` autostart entry.

use anyhow::{Context, Result};
use auto_launch::AutoLaunchBuilder;

/// Enable (or disable) running `llmtrim serve --port <port>` at login.
pub fn configure(enable: bool, port: u16) -> Result<()> {
    let exe = std::env::current_exe().context("could not find the llmtrim executable")?;
    let path = exe.to_string_lossy();
    let port_arg = port.to_string();

    let auto = AutoLaunchBuilder::new()
        .set_app_name("llmtrim")
        .set_app_path(path.as_ref())
        .set_args(&["serve", "--port", port_arg.as_str(), "--supervised"])
        .build()
        .map_err(|e| anyhow::anyhow!("failed to configure autostart: {e}"))?;

    if enable {
        auto.enable()
            .map_err(|e| anyhow::anyhow!("failed to enable autostart: {e}"))?;
        println!("✓ Autostart enabled — `llmtrim serve --port {port}` runs at login.");
        println!("  disable: llmtrim autostart --off");
    } else {
        auto.disable()
            .map_err(|e| anyhow::anyhow!("failed to disable autostart: {e}"))?;
        println!("Autostart disabled.");
    }
    Ok(())
}
