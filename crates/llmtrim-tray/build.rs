fn main() {
    // Restrict the webview to exactly the tray's own commands and let tauri-build
    // autogenerate their `allow-*`/`deny-*` permissions, which capabilities/main.json
    // references as `llmtrim-tray:allow-<command>`. Without this list the build emits no
    // app-command permissions and the capability fails to resolve them.
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "get_dashboard",
            "get_agent_trend",
            "set_poll_interval",
            "start_proxy",
            "stop_proxy",
            "get_tray_autostart",
            "set_tray_autostart",
            "quit",
        ]),
    ))
    .expect("failed to run tauri-build");
}
