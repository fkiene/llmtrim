// Prevents a Windows console window from opening alongside the tray icon.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use llmtrim_ledger::breakdown_db::BreakdownDb;
use llmtrim_ledger::dashboard::{
    ChildCard, Dashboard, build_child_cards, build_dashboard, parse_period, sanitise_error,
};
use llmtrim_ledger::tracking::{Period, db_path};
use tauri::image::Image;
use tauri::menu::{MenuBuilder, MenuItem, MenuItemBuilder, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_positioner::{Position, WindowExt};

/// Menu-bar glyph bytes. macOS gets the black template image (the system tints
/// it for light/dark bars); every other platform gets the green glyph so it
/// stays visible on a dark taskbar. See `tools/gen_icons.py`.
#[cfg(target_os = "macos")]
const TRAY_ICON_PNG: &[u8] = include_bytes!("../icons/tray-mono.png");
#[cfg(not(target_os = "macos"))]
const TRAY_ICON_PNG: &[u8] = include_bytes!("../icons/tray-color.png");

/// Window is re-shown only if the last blur-dismiss is older than this. A tray
/// click that dismisses a focused popover delivers `Focused(false)` (hide) just
/// before the click event, so without this guard the click would re-open it.
const DISMISS_DEBOUNCE: Duration = Duration::from_millis(250);

/// Granularity of the poll sleep, so a quit is observed within ~1s rather than
/// after a full (possibly 30s) interval.
const POLL_TICK: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// Shared mutable state managed by Tauri.
struct TrayState {
    /// Current poll interval in seconds. Updated by `set_poll_interval`.
    poll_interval_secs: u64,
    /// Instant of the most recent blur-driven hide, used to debounce the
    /// tray-click that caused the blur.
    last_dismiss: Option<Instant>,
}

impl Default for TrayState {
    fn default() -> Self {
        Self {
            poll_interval_secs: 30,
            last_dismiss: None,
        }
    }
}

/// Lock the state recovering from a poisoned mutex instead of panicking — a
/// panic in one path must not take down every later IPC call (no `.unwrap()`
/// in production, per project rules). Recovering the inner value is safe here:
/// the only state is `poll_interval_secs` and `last_dismiss`, so the worst case
/// after a poisoning panic is a stale interval or dismiss timestamp, never a
/// corrupt invariant.
fn lock_state(state: &Mutex<TrayState>) -> MutexGuard<'_, TrayState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---------------------------------------------------------------------------
// Tauri commands — thin wrappers over llmtrim-ledger pure functions.
//
// Real-IO entry-points: legitimately uncovered by unit tests (the logic lives
// in llmtrim-ledger and is fully tested there).
// ---------------------------------------------------------------------------

/// Return the full dashboard snapshot. The ledger path is resolved the same way the
/// proxy/CLI does — via `llmtrim_ledger::tracking::db_path()` which respects the
/// `LLMTRIM_DB_PATH` env var and `XDG_DATA_HOME`.
///
/// SECURITY: filesystem paths in errors are stripped before reaching JS. The full
/// error detail is written to stderr so it appears in `llmtrim serve` logs.
#[tauri::command]
fn get_dashboard(state: State<'_, Arc<Mutex<TrayState>>>) -> Result<Dashboard, String> {
    let poll_secs = lock_state(&state).poll_interval_secs;
    let dashboard = load_dashboard(poll_secs).map_err(|e| {
        // Strip filesystem paths: log full detail, surface only the class of failure.
        eprintln!("llmtrim-tray: get_dashboard failed: {e:#}");
        sanitise_error(&e)
    })?;
    Ok(dashboard)
}

/// Return raw saved_pct trend buckets for one agent over a given period.
/// `period` accepts "day", "week", or "month" (case-insensitive).
#[tauri::command]
fn get_agent_trend(agent: String, period: String) -> Result<Vec<f64>, String> {
    let p = parse_period(&period).map_err(|e| format!("unknown period {period:?}: {e}"))?;
    let path = db_path().map_err(|e| {
        eprintln!("llmtrim-tray: db_path failed: {e:#}");
        "could not resolve ledger path".to_string()
    })?;
    let db = BreakdownDb::open_readonly(&path).map_err(|e| {
        eprintln!("llmtrim-tray: open_readonly failed: {e:#}");
        sanitise_error(&e)
    })?;
    let trend = db.agent_trend(&agent, p, 30).map_err(|e| {
        eprintln!("llmtrim-tray: agent_trend failed: {e:#}");
        "failed to query trend data".to_string()
    })?;
    Ok(trend.into_iter().map(|b| b.saved_pct).collect())
}

/// Drill-down level 2: the projects under one agent, lazy-fetched when its card
/// is expanded. Opens the ledger read-only; a not-yet-initialised ledger returns
/// an empty list rather than an error (same empty-state rule as the dashboard).
#[tauri::command]
fn get_agent_projects(agent: String) -> Result<Vec<ChildCard>, String> {
    child_cards(|db| db.project_aggregates(&agent))
}

/// Drill-down level 3 (leaf): the sessions under one agent/project. `project` is
/// the opaque `key` from a project `ChildCard` (empty string == the no-project bucket).
#[tauri::command]
fn get_project_sessions(agent: String, project: String) -> Result<Vec<ChildCard>, String> {
    child_cards(|db| db.session_aggregates(&agent, &project))
}

/// Shared body for the two drill-down commands: resolve the ledger path, open it
/// read-only (empty list when not initialised), run `query`, build the cards.
/// Every failure is logged in full and mapped to a path-free string for the webview.
fn child_cards(
    query: impl FnOnce(&BreakdownDb) -> anyhow::Result<Vec<llmtrim_ledger::dashboard::ChildAggregate>>,
) -> Result<Vec<ChildCard>, String> {
    let path = db_path().map_err(|e| {
        eprintln!("llmtrim-tray: db_path failed: {e:#}");
        "could not resolve ledger path".to_string()
    })?;
    let Some(db) = BreakdownDb::open_readonly_if_ready(&path).map_err(|e| {
        eprintln!("llmtrim-tray: open_readonly_if_ready failed: {e:#}");
        sanitise_error(&e)
    })?
    else {
        return Ok(Vec::new());
    };
    let aggregates = query(&db).map_err(|e| {
        eprintln!("llmtrim-tray: drill-down query failed: {e:#}");
        "failed to load breakdown data".to_string()
    })?;
    Ok(build_child_cards(aggregates))
}

/// Update the poll interval for the background refresh loop.
#[tauri::command]
fn set_poll_interval(secs: u64, state: State<'_, Arc<Mutex<TrayState>>>) {
    // Floor at 1s: `secs = 0` would make the poll loop spin without sleeping,
    // pinning a core on continuous SQLite reads and webview events.
    lock_state(&state).poll_interval_secs = secs.max(1);
}

/// Quit the application cleanly.
#[tauri::command]
fn quit(app: AppHandle) {
    app.exit(0);
}

// ---------------------------------------------------------------------------
// Proxy / autostart control — shells out to the sibling `llmtrim` CLI.
//
// Actions run in Rust (not the JS shell plugin), spawning only the resolved
// `llmtrim` binary with fixed arguments. No user input reaches the command line,
// so the JS shell capability stays disabled and the CSP keeps `connect-src
// 'none'`. These are real-IO entry-points, legitimately uncovered by unit tests.
// ---------------------------------------------------------------------------

/// Resolve the `llmtrim` CLI: prefer the binary installed next to the tray app
/// (how every bundle ships it), else fall back to a bare name resolved on PATH.
///
/// The PATH fallback is a deliberate, bounded trade-off: every caller passes a
/// fixed argument literal (`start` / `stop` / `autostart …`), never user or JS
/// input, so there is no argument-injection surface. A maintainer adding a new
/// caller must keep that invariant — do not pass dynamic data on this command
/// line, or the PATH fallback becomes a hijack vector.
fn llmtrim_binary() -> std::path::PathBuf {
    let name = if cfg!(windows) {
        "llmtrim.exe"
    } else {
        "llmtrim"
    };
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return sibling;
        }
    }
    std::path::PathBuf::from(name)
}

/// Build a `Command` for the `llmtrim` CLI with `args`.
///
/// On Windows the tray is a GUI (`windows_subsystem = "windows"`) process, so
/// spawning a console subprocess flashes a console window each call. Since the
/// poll loop shells out to `status --json` every interval, that window would pop
/// up repeatedly. `CREATE_NO_WINDOW` (0x0800_0000) suppresses it. No-op elsewhere.
fn llmtrim_command(args: &[&str]) -> std::process::Command {
    let mut cmd = std::process::Command::new(llmtrim_binary());
    cmd.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Run `llmtrim <args>` to completion, mapping any failure to a sanitised string
/// (full detail logged to stderr, never surfaced to JS).
fn run_llmtrim(args: &[&str]) -> Result<(), String> {
    let output = llmtrim_command(args).output().map_err(|e| {
        eprintln!("llmtrim-tray: failed to run llmtrim {args:?}: {e}");
        "could not run the llmtrim CLI — is it installed?".to_string()
    })?;
    if output.status.success() {
        Ok(())
    } else {
        eprintln!(
            "llmtrim-tray: llmtrim {args:?} exited {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        Err("the llmtrim command failed — see the proxy logs".to_string())
    }
}

/// Start the background interceptor (no-op if already running).
#[tauri::command]
fn start_proxy() -> Result<(), String> {
    run_llmtrim(&["start"])
}

/// Stop the background interceptor.
#[tauri::command]
fn stop_proxy() -> Result<(), String> {
    run_llmtrim(&["stop"])
}

/// Whether the interceptor daemon is currently running.
///
/// Running state is not in the ledger DB — it's the live pidfile + port probe the CLI
/// owns (`daemon::running`). We read it back through the CLI's authoritative
/// `status --json` `daemon.running` boolean rather than re-implementing cross-platform
/// process liveness here. (`status --quiet`/health is the wrong signal: it reports
/// `degraded` for a stopped-but-still-wired proxy, so it can't tell running from stopped.)
/// Any failure reads as "not running" so the menu offers Start, the safe default.
fn proxy_running() -> bool {
    let Ok(output) = llmtrim_command(&["status", "--json"]).output() else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&output.stdout)
        .ok()
        .and_then(|v| v["daemon"]["running"].as_bool())
        .unwrap_or(false)
}

/// The single proxy menu item, retained so the poll loop and the menu handler can flip
/// its label between "Start proxy" and "Stop proxy". Only one line ever shows; the click
/// re-checks the live state and runs the matching action.
struct ProxyMenu {
    item: MenuItem<tauri::Wry>,
}

/// Update the proxy item's label to match whether the proxy is running.
fn refresh_proxy_menu(app: &AppHandle) {
    let label = if proxy_running() {
        "Stop proxy"
    } else {
        "Start proxy"
    };
    let _ = app.state::<ProxyMenu>().item.set_text(label);
}

/// Whether the tray is set to open at login. Reads the CLI's scriptable status;
/// any failure reads as "off" so the toggle defaults to a safe state.
#[tauri::command]
fn get_tray_autostart() -> bool {
    llmtrim_command(&["autostart", "--tray", "--status"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "enabled")
        .unwrap_or(false)
}

/// Enable or disable opening the tray at login.
#[tauri::command]
fn set_tray_autostart(enable: bool) -> Result<(), String> {
    if enable {
        run_llmtrim(&["autostart", "--tray"])
    } else {
        run_llmtrim(&["autostart", "--tray", "--off"])
    }
}

/// Whether the proxy is set to run at login. Separate login item from the tray's
/// (`autostart` vs `autostart --tray`); any failure reads as "off".
#[tauri::command]
fn get_proxy_autostart() -> bool {
    llmtrim_command(&["autostart", "--status"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "enabled")
        .unwrap_or(false)
}

/// Enable or disable running the proxy at login. Omitting `--port` reuses the
/// port the interceptor already uses.
#[tauri::command]
fn set_proxy_autostart(enable: bool) -> Result<(), String> {
    if enable {
        run_llmtrim(&["autostart"])
    } else {
        run_llmtrim(&["autostart", "--off"])
    }
}

/// Whether the proxy is currently running, so Settings can show a single
/// Start/Stop button matching the live state. Reads the same authoritative
/// `status --json` `daemon.running` flag the tray menu uses.
#[tauri::command]
fn get_proxy_running() -> bool {
    proxy_running()
}

// ---------------------------------------------------------------------------
// Application setup
// ---------------------------------------------------------------------------

fn main() {
    tauri::Builder::default()
        // Must be registered first (Tauri guidance). A second launch — `y` in the
        // status dashboard, `llmtrim tray`, or autostart racing a manual start —
        // surfaces the existing popover instead of adding a duplicate tray icon.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_popover(app);
        }))
        .plugin(tauri_plugin_positioner::init())
        .manage(Arc::new(Mutex::new(TrayState::default())))
        .manage(Arc::new(AtomicBool::new(false)))
        .setup(|app| {
            // ----------------------------------------------------------------
            // macOS: run as a menubar app — hide the Dock icon.
            // `ActivationPolicy::Accessory` is the AppKit equivalent of
            // `setActivationPolicy(.accessory)` in Swift; it removes the Dock
            // entry and the default menu bar without touching our tray icon.
            // ----------------------------------------------------------------
            #[cfg(target_os = "macos")]
            {
                app.set_activation_policy(tauri::ActivationPolicy::Accessory);
            }

            // ----------------------------------------------------------------
            // Popover window — hidden at start; toggled by tray click.
            // ----------------------------------------------------------------
            let popover = app
                .get_webview_window("popover")
                .expect("popover window not found — check tauri.conf.json");

            // Debug builds: open DevTools and show the window on start so the
            // webview console is visible for diagnosing render failures. Compiled
            // out of release builds.
            #[cfg(debug_assertions)]
            {
                popover.open_devtools();
                let _ = popover.show();
            }

            // Auto-hide on blur (works on both macOS and Windows). Record the
            // dismiss time so the tray click that caused this blur doesn't
            // immediately re-open the window (see `toggle_popover`).
            let popover_blur = popover.clone();
            let blur_handle = app.handle().clone();
            popover.on_window_event(move |event| {
                if let tauri::WindowEvent::Focused(false) = event {
                    let _ = popover_blur.hide();
                    let state = blur_handle.state::<Arc<Mutex<TrayState>>>();
                    lock_state(&state).last_dismiss = Some(Instant::now());
                }
            });

            // ----------------------------------------------------------------
            // Tray icon with macOS title and tooltip.
            // ----------------------------------------------------------------
            let tray_app = app.handle().clone();
            // tauri::Error is a std::error::Error, so `?` converts into the
            // setup closure's Box<dyn Error> directly (anyhow's Context would not).
            let tray_icon = Image::from_bytes(TRAY_ICON_PNG)?;

            // Right-click context menu. On Linux this is the primary interaction:
            // left-click toggle events are not reliably delivered by the
            // StatusNotifier/AppIndicator host, so the menu's "Open" item is how
            // Linux users reach the popover. On macOS/Windows the menu is the
            // right-click companion to the left-click toggle.
            let open_item = MenuItemBuilder::with_id("open", "Open llmtrim").build(app)?;
            // One toggling item, not separate Start/Stop lines: the label tracks the proxy's
            // state and the click runs the matching action (see the "proxy" menu event). The
            // initial label is corrected by `refresh_proxy_menu` once the state is known.
            let proxy_item = MenuItemBuilder::with_id("proxy", "Start proxy").build(app)?;
            let settings_item = MenuItemBuilder::with_id("settings", "Settings…").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let menu = MenuBuilder::new(app)
                .items(&[
                    &open_item,
                    &PredefinedMenuItem::separator(app)?,
                    &proxy_item,
                    &PredefinedMenuItem::separator(app)?,
                    &settings_item,
                    &PredefinedMenuItem::separator(app)?,
                    &quit_item,
                ])
                .build()?;

            // Retain the item so the poll loop and menu handler can relabel it, and set the
            // initial label off the main thread so startup stays snappy.
            app.manage(ProxyMenu {
                item: proxy_item.clone(),
            });
            let init_app = app.handle().clone();
            std::thread::spawn(move || refresh_proxy_menu(&init_app));

            let menu_app = app.handle().clone();
            TrayIconBuilder::with_id("main")
                .icon(tray_icon)
                // macOS template: the black glyph is auto-tinted per menu-bar theme.
                .icon_as_template(cfg!(target_os = "macos"))
                .tooltip("llmtrim — compression savings")
                .menu(&menu)
                // Left-click toggles the popover via `on_tray_icon_event`; keep the
                // menu on right-click only. (No-op on Linux, where this is unsupported.)
                .show_menu_on_left_click(false)
                .on_menu_event(move |_app, event| match event.id().as_ref() {
                    "open" => show_popover(&menu_app),
                    // Run on a worker thread: `run_llmtrim` blocks until the CLI
                    // exits (first-run CA generation can take a moment) and menu
                    // events fire on the main event loop, so calling inline would
                    // freeze the icon. Errors are logged to stderr by `run_llmtrim`;
                    // the Settings panel is the interactive surface for failures.
                    // Re-check the live state at click time (the label is only a hint),
                    // run the opposite action, then relabel the item.
                    "proxy" => {
                        let handle = menu_app.clone();
                        std::thread::spawn(move || {
                            if proxy_running() {
                                let _ = stop_proxy();
                            } else {
                                let _ = start_proxy();
                            }
                            refresh_proxy_menu(&handle);
                        });
                    }
                    "settings" => {
                        // Open the popover and ask the UI to show its settings view.
                        show_popover(&menu_app);
                        let _ = menu_app.emit("show-settings", ());
                    }
                    "quit" => menu_app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(move |tray, event| {
                    // Forward to positioner so TrayCenter positioning works.
                    tauri_plugin_positioner::on_tray_event(tray.app_handle(), &event);

                    // Match only the left-button release. `Click` fires on both
                    // press and release; the wildcard would toggle twice per click.
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        toggle_popover(&tray_app);
                    }
                })
                .build(app)?;

            // ----------------------------------------------------------------
            // Update macOS menubar title with aggregate % saved on first load.
            // ----------------------------------------------------------------
            #[cfg(target_os = "macos")]
            {
                if let Ok(dash) = load_dashboard(30) {
                    let pct = dash.totals.saved_pct;
                    if let Some(tray) = app.tray_by_id("main") {
                        let _ = tray.set_title(Some(&format!("{pct:.0}% saved")));
                    }
                }
            }

            // ----------------------------------------------------------------
            // Background poll loop: emits a `dashboard` event every N seconds.
            // The stop flag lets the loop exit promptly on app shutdown.
            // ----------------------------------------------------------------
            let poll_app = app.handle().clone();
            let stop = app.state::<Arc<AtomicBool>>().inner().clone();
            std::thread::spawn(move || {
                poll_loop(poll_app, stop);
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_dashboard,
            get_agent_trend,
            get_agent_projects,
            get_project_sessions,
            set_poll_interval,
            start_proxy,
            stop_proxy,
            get_tray_autostart,
            set_tray_autostart,
            get_proxy_autostart,
            set_proxy_autostart,
            get_proxy_running,
            quit,
        ])
        .build(tauri::generate_context!())
        .expect("error building llmtrim tray")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // Signal the poll thread to stop before the process tears down.
                app.state::<Arc<AtomicBool>>()
                    .store(true, Ordering::Relaxed);
            }
        });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a fresh dashboard snapshot from the ledger.
fn load_dashboard(poll_secs: u64) -> anyhow::Result<Dashboard> {
    let path = db_path().context("could not resolve ledger path")?;
    let now = chrono::Utc::now().to_rfc3339();
    // No ledger yet (proxy never ran) is the empty state, not an error.
    let Some(db) = BreakdownDb::open_readonly_if_ready(&path).context("could not open ledger")?
    else {
        return Ok(build_dashboard(Vec::new(), HashMap::new(), now, poll_secs));
    };
    let aggregates = db
        .agent_aggregates()
        .context("agent_aggregates query failed")?;
    let mut trends: HashMap<String, Vec<f64>> = HashMap::new();
    for agg in &aggregates {
        let trend = db
            .agent_trend(&agg.agent, Period::Day, 30)
            .context("agent_trend query failed")?;
        trends.insert(
            agg.agent.clone(),
            trend.into_iter().map(|b| b.saved_pct).collect(),
        );
    }
    Ok(build_dashboard(aggregates, trends, now, poll_secs))
}

/// Toggle the popover window: show (positioned at TrayCenter) or hide.
/// Includes a short debounce so a tray click while the window is closing
/// doesn't re-open it immediately.
fn toggle_popover(app: &AppHandle) {
    let Some(popover) = app.get_webview_window("popover") else {
        return;
    };
    let visible = popover.is_visible().unwrap_or(false);
    if visible {
        let _ = popover.hide();
    } else {
        // If the window was just dismissed by blur (this very click moved focus
        // away first), don't re-open it — that would make a dismissing click a
        // no-op flicker.
        let state = app.state::<Arc<Mutex<TrayState>>>();
        if let Some(t) = lock_state(&state).last_dismiss
            && t.elapsed() < DISMISS_DEBOUNCE
        {
            return;
        }
        show_popover(app);
    }
}

/// Show the popover positioned next to the tray icon, unconditionally. Used by
/// the menu "Open" item (which, unlike a tray click, has no blur to debounce).
fn show_popover(app: &AppHandle) {
    let Some(popover) = app.get_webview_window("popover") else {
        return;
    };
    let _ = popover.move_window(Position::TrayCenter);
    let _ = popover.show();
    let _ = popover.set_focus();
}

/// Background poll loop: sleeps `poll_interval_secs`, then emits a `dashboard`
/// event on the app so the frontend can refresh without polling from JS.
fn poll_loop(app: AppHandle, stop: Arc<AtomicBool>) {
    loop {
        // Read current interval from state.
        let secs = lock_state(&app.state::<Arc<Mutex<TrayState>>>()).poll_interval_secs;

        // Sleep in short ticks so a quit is observed within ~POLL_TICK rather
        // than after a full (possibly 30s) interval. A mid-sleep change to the
        // interval also takes effect on the next cycle.
        let mut elapsed = Duration::ZERO;
        let target = Duration::from_secs(secs);
        while elapsed < target {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(POLL_TICK);
            elapsed += POLL_TICK;
        }
        if stop.load(Ordering::Relaxed) {
            return;
        }

        // Keep the proxy item's label in sync with state changed outside the tray
        // (e.g. `llmtrim start`/`stop` from a shell, or a crash).
        refresh_proxy_menu(&app);

        match load_dashboard(secs) {
            Ok(dash) => {
                // Update macOS menubar title.
                #[cfg(target_os = "macos")]
                {
                    let pct = dash.totals.saved_pct;
                    if let Some(tray) = app.tray_by_id("main") {
                        let _ = tray.set_title(Some(&format!("{pct:.0}% saved")));
                    }
                }
                // Emit to all webview windows (front-end listens for "dashboard").
                let _ = app.emit("dashboard", &dash);
            }
            Err(e) => {
                eprintln!("llmtrim-tray: poll failed: {e:#}");
            }
        }
    }
}
