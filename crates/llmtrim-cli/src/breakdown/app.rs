//! The tabbed cost-breakdown TUI: event loop, terminal lifecycle, and the three views.
//!
//! Tabs: **Overview** (the savings dashboard, supplied by the caller as plain text),
//! **Sessions** (every session grouped agent → project → session), and **Detail** (drill
//! into a session: context-window occupancy on top, per-source cost down to each MCP
//! server below). Keyboard driven; live-refreshes the ledger on a timer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Bar, BarChart, BarGroup, Block, BorderType, Borders, Cell, Clear, Gauge, Padding, Paragraph,
    Row, Table, Wrap,
};
use ratatui::{Frame, Terminal};

use super::db::{BreakdownDb, CostRow, OccupancyRow, SessionRow};
use super::palette;
use super::tree::{Activate, Column, TreeNode, TreeTable};

/// Spoken health of the proxy — plain words, no port/pid/CA jargon. `Off` is the dangerous
/// state (wired but not running); `fix`/`uninstall` are the literal commands to type.
#[derive(Clone, Default)]
pub struct StatusLine {
    pub kind: StatusKind,
    pub text: String,
    pub fix: Option<String>,
    pub uninstall: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum StatusKind {
    /// On, healthy, with recent traffic.
    Working,
    /// On, healthy, no traffic yet.
    #[default]
    Ready,
    /// On and serving, but the running daemon is an older binary than the one installed — a
    /// restart applies the update. Not a fault, so it shows the normal dashboard with a `u`
    /// (Update) nudge, not the Repair alert.
    Stale,
    /// On but needs attention / broken — the Repair alert state.
    Degraded,
    /// Not catching traffic (wired but down) — the alarm state.
    Off,
}

/// Everything the Overview tab renders — built by the caller from the ledger each refresh,
/// so the TUI draws it natively (no ANSI-string parsing).
#[derive(Clone, Default)]
pub struct OverviewData {
    pub status: StatusLine,
    /// Money figures from frozen `breakdown_turns.bill_micros` (`None` when no billed turns
    /// yet, or compressions exist but attribution has not landed — see `money_coverage`).
    pub paid_usd: Option<f64>,
    pub would_have_usd: Option<f64>,
    pub saved_usd: Option<f64>,
    pub saved_today_usd: Option<f64>,
    /// Input-savings fraction (0..1) over the compressible surface → spoken as "about a third
    /// less". This is the headline basis ("% of new content").
    pub pct_less: f64,
    /// The same saving over the whole prompt (the cached prefix included) — the diluted basis
    /// the `c` toggle reveals. Shares `pct_less`'s numerator with a wider denominator, so it is
    /// always `<= pct_less` (see `overview_data`).
    pub pct_less_whole: f64,
    pub added_ms: Option<f64>,
    pub requests: i64,
    /// Net $ saved per day, oldest→newest, last 7 days (for the sparkline).
    pub trend_daily_usd: Vec<f64>,
    /// Top models by $ saved (frozen-rate blend). Footer Total is all-time, not sum of rows.
    pub savers: Vec<(String, f64)>,
    /// Expert strip (behind `m`): raw input before/after, output billed, output est %.
    pub input_before: i64,
    pub input_after: i64,
    pub output_billed: i64,
    pub output_est_pct: f64,
    pub has_traffic: bool,
    /// Some figures are estimated (the provider didn't report exact token counts) — surfaces
    /// the honest "≈ estimated" flag on the trust strip.
    pub approximate: bool,
    /// Distinct sessions the ledger has recorded — the coverage metric tile.
    pub sessions: i64,
    /// A newer release version if the (cached, opt-out) update check found one — shown in the
    /// meta bar; `u` then runs the updater.
    pub update_available: Option<String>,
    /// Compressions events vs breakdown turns — drives partial/empty money UX.
    /// `pub(crate)` so adding these is not a SemVer break for external struct literals.
    pub(crate) coverage_compressions: i64,
    pub(crate) coverage_turns: i64,
    /// Turns with usage but zero bill (unpriced model).
    pub(crate) turns_unpriced: i64,
    /// True when compressions exist but no breakdown bills — do not show $0 as truth.
    pub(crate) money_unavailable: bool,
}

impl OverviewData {
    /// Size-reduction fraction for the gauge: over the whole prompt (cached prefix included)
    /// when `whole_prompt`, else over the compressible surface — the default headline. Dollars
    /// are always the real net bill; only this percentage has two honest framings.
    fn pct_smaller(&self, whole_prompt: bool) -> f64 {
        if whole_prompt {
            self.pct_less_whole
        } else {
            self.pct_less
        }
    }
}

/// Which tab is active.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Sessions,
    Detail,
}

/// Sessions-tree payload: only session leaves are drillable.
#[derive(Clone)]
enum Node {
    Group,
    Session(String),
}

/// Which Detail pane has focus.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pane {
    Occupancy,
    Cost,
}

struct Detail {
    session_id: String,
    title: String,
    occupancy: TreeTable<()>,
    cost: TreeTable<()>,
    focus: Pane,
    /// True until the detail worker's queries return for this session — the panes show a
    /// "Loading…" placeholder meanwhile (SQLite never runs on the UI thread).
    loading: bool,
}

/// A Detail drill-down computed off the UI thread, tagged with the session it was built for so
/// the UI can drop a stale result when the user drilled a different session in the meantime.
struct DetailSnapshot {
    session_id: String,
    data: DetailData,
}

/// Restores the terminal (raw mode + main screen) on drop, even on panic/error.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
    }
}

/// A ledger snapshot computed off the UI thread: the Overview data + the raw session rows.
pub type Snapshot = (OverviewData, Vec<SessionRow>);

/// Run the TUI. `snapshot()` re-queries the ledger and is run on a **background thread** so the
/// UI never blocks on SQLite; the latest result is published through a mutex and the UI applies
/// it whenever one is ready. `interval` is the data-refresh period (seconds).
pub fn run(
    interval: u64,
    mut snapshot: impl FnMut() -> Snapshot + Send + 'static,
) -> Result<PostAction> {
    // Pick the initial Catppuccin flavor: env-first, then the config file `theme` key, else
    // the default (Mocha). The `t` key cycles it live and persists the choice thereafter.
    if let Some(f) = llmtrim_core::config::RuntimeConfig::get()
        .theme
        .as_deref()
        .and_then(palette::from_name)
    {
        palette::set(f);
    }
    // The Detail panes' SQLite connection. It is moved into the detail worker thread below, so
    // the UI thread never runs a query for Detail.
    let detail_db = BreakdownDb::open().context("failed to open breakdown ledger")?;
    enable_raw_mode().context("failed to enter raw mode")?;
    let guard = TerminalGuard;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    // Buffer the backend so each frame is one write/flush instead of many small syscalls.
    let backend = CrosstermBackend::new(std::io::BufWriter::new(std::io::stdout()));
    let mut terminal = Terminal::new(backend).context("failed to build terminal")?;

    let refresh = Duration::from_secs(interval.max(1));

    // Background refresh: compute the snapshot off-thread and publish the latest into a mutex.
    // The UI thread only ever does cheap tree-building when it picks one up, so heavy ledger
    // scans never freeze input or the redraw.
    let latest: Arc<Mutex<Option<Snapshot>>> = Arc::new(Mutex::new(None));
    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let latest = Arc::clone(&latest);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let snap = snapshot();
                if let Ok(mut g) = latest.lock() {
                    *g = Some(snap);
                }
                // Sleep in small slices so quitting is responsive.
                let mut slept = Duration::ZERO;
                while slept < refresh && !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(100));
                    slept += Duration::from_millis(100);
                }
            }
        })
    };

    // Detail worker: owns the Detail SQLite connection and waits for drill requests on a
    // channel. For each session id it runs the three queries off-thread and publishes a
    // `DetailSnapshot`. It blocks on the channel with a timeout so the `stop` flag stays
    // responsive (the channel-disconnect arm also exits on teardown when the sender is dropped).
    let detail_latest: Arc<Mutex<Option<DetailSnapshot>>> = Arc::new(Mutex::new(None));
    let (detail_tx, detail_rx) = std::sync::mpsc::channel::<String>();
    let detail_worker = {
        let detail_latest = Arc::clone(&detail_latest);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            use std::sync::mpsc::RecvTimeoutError;
            while !stop.load(Ordering::Relaxed) {
                match detail_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(mut session_id) => {
                        // Coalesce rapid drills: if more requests queued while we waited, only
                        // the most recent session matters — skip the superseded queries.
                        while let Ok(newer) = detail_rx.try_recv() {
                            session_id = newer;
                        }
                        let data = compute_detail(&detail_db, &session_id);
                        if let Ok(mut g) = detail_latest.lock() {
                            *g = Some(DetailSnapshot { session_id, data });
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        })
    };

    let mut app = App::new(None, refresh);
    app.detail_req = Some(detail_tx);

    // Redraw only when something changed — a key, a fresh snapshot, or the once-a-second
    // animation tick (clock/mascot/quip). Idle repaints ~1×/s instead of the ~4×/s the 250ms
    // poll would force. Drawing sits at the loop top, so a key repaints immediately.
    let mut dirty = true;
    let mut last_sec = u64::MAX;
    loop {
        // If either worker thread died, don't sit on a frozen, silently-stale view — quit so the
        // join after the loop surfaces the panic. Query errors don't reach here (both workers
        // degrade to empty data, never panic), so this only fires on an unexpected panic.
        if worker.is_finished() || detail_worker.is_finished() {
            break;
        }
        // Pick up the latest background snapshot, if any (non-blocking). Recover a poisoned lock
        // (a worker panic while publishing) rather than dropping every future result.
        let snapshot = {
            let mut g = latest.lock().unwrap_or_else(|p| p.into_inner());
            g.take()
        };
        if let Some(snap) = snapshot {
            app.apply(snap.0, snap.1);
            dirty = true;
        }
        // Pick up a ready Detail result (non-blocking); install it only if it still matches the
        // drilled session (guards against a stale result after the user drilled elsewhere).
        let detail_snap = {
            let mut g = detail_latest.lock().unwrap_or_else(|p| p.into_inner());
            g.take()
        };
        if let Some(snap) = detail_snap
            && app.apply_detail(snap)
        {
            dirty = true;
        }
        if dirty {
            terminal.draw(|f| app.render(f))?;
            dirty = false;
        }
        // The poll timeout caps the loop at ~4Hz (it blocks, no spin). A poll error means the
        // terminal is gone (fd closed / backgrounded); break rather than spin forever.
        match event::poll(Duration::from_millis(250)) {
            Ok(true) => {
                if let Ok(Event::Key(key)) = event::read()
                    && key.kind == KeyEventKind::Press
                {
                    let ctrl_c = key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c');
                    if ctrl_c || app.handle_key(key.code) {
                        break;
                    }
                    dirty = true;
                }
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("llmtrim: terminal input error: {e}");
                break;
            }
        }
        // Animation tick: clock/mascot/quip change at most once per wall-clock second.
        let sec = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if sec != last_sec {
            last_sec = sec;
            dirty = true;
        }
    }
    stop.store(true, Ordering::Relaxed);
    // Drop the sender so the detail worker's channel disconnects and it exits promptly.
    app.detail_req = None;
    if let Err(e) = worker.join() {
        eprintln!("llmtrim: breakdown refresh thread panicked: {e:?}");
    }
    if let Err(e) = detail_worker.join() {
        eprintln!("llmtrim: breakdown detail thread panicked: {e:?}");
    }
    let action = app.action;
    let pending_ensure = app.pending_ensure;
    let pending_sub_login = app.pending_sub_login;
    // Leave alt-screen before any normal stdout work (ensure / sub hints).
    drop(terminal);
    drop(guard);
    if pending_ensure {
        crate::ensure::run_cli(false)?;
    }
    if pending_sub_login {
        let _ = crate::ensure::mark_sub_nudge_shown();
        println!("Subscription setup: pick a provider, then log in.");
        println!("  llmtrim sub auth codex login   # ChatGPT / Codex plan");
        println!("  llmtrim sub auth kimi login    # Kimi coding plan");
        println!("  llmtrim sub on codex           # enable after login");
    }
    Ok(action)
}

/// What the caller should do after the TUI exits — set by a key, run on the normal screen.
///
/// Keep the public variant set stable (semver): new TUI actions that only need work inside
/// this crate are handled before returning (see `f` / sub-nudge keys).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum PostAction {
    #[default]
    None,
    /// `d` — run `llmtrim doctor` (repair check).
    Doctor,
    /// `u` — run `llmtrim update` (a newer release is available).
    Update,
    /// `u` on a stale daemon — restart it (`llmtrim start --force`) to load the new binary.
    Restart,
}

struct App {
    /// Only the SVG export test builds Detail panes synchronously from this connection. The
    /// live TUI moves its Detail `BreakdownDb` into the detail worker, so the UI thread holds
    /// no connection and runs zero SQLite for Detail — this stays `None` there.
    #[cfg_attr(not(test), allow(dead_code))]
    db: Option<BreakdownDb>,
    /// The data-refresh period — shown in the meta bar (`↻ Ns`); the actual refresh runs on a
    /// background thread now, not on this timer.
    refresh: Duration,
    tab: Tab,
    /// `c` flips the size gauge between "% of new content" (`false`, the default headline) and
    /// "% of the whole prompt" (`true`, diluted by the cached prefix). Overview-only. Dollars
    /// never change — they are always the real net-of-cache bill.
    whole_prompt: bool,
    overview: Option<OverviewData>,
    /// Full-screen keymap overlay (`?`); dismissed by any key.
    show_help: bool,
    sessions: TreeTable<Node>,
    detail: Option<Detail>,
    /// Drill requests to the detail worker: the selected session id is sent here, the worker
    /// runs the SQLite queries off-thread and publishes a `DetailSnapshot`. `None` in tests,
    /// which build the Detail synchronously instead.
    detail_req: Option<std::sync::mpsc::Sender<String>>,
    /// Set by `d`/`u`/`y` to a command to run after the TUI tears down (so it runs on the normal
    /// screen, not inside the alt-screen).
    action: PostAction,
    /// Whether the tray GUI is installed next to the CLI — gates the `y tray` action so the
    /// hint never offers something that can't run. Resolved once at construction.
    tray_available: bool,
    /// Integrations need attention (statusline/guard/daemon skew…) — gates `f` fix.
    needs_ensure: bool,
    /// One-time subscription onboarding on the Overview tab.
    show_sub_nudge: bool,
    /// Run `ensure` after the TUI tears down (alt-screen left). Keeps `PostAction` stable.
    pending_ensure: bool,
    /// Print sub login steps after the TUI tears down.
    pending_sub_login: bool,
}

impl App {
    fn new(db: Option<BreakdownDb>, refresh: Duration) -> Self {
        App {
            db,
            refresh,
            tab: Tab::Overview,
            whole_prompt: false,
            overview: None,
            show_help: false,
            sessions: TreeTable::new(
                "agent · project · session",
                session_columns(),
                palette::frame(),
            )
            .empty_hint("no sessions yet — run llmtrim and use your agent"),
            detail: None,
            detail_req: None,
            action: PostAction::None,
            tray_available: crate::tray::tray_binary().is_some(),
            needs_ensure: crate::ensure::needs_attention(),
            show_sub_nudge: crate::ensure::should_show_sub_nudge(),
            pending_ensure: false,
            pending_sub_login: false,
        }
    }

    /// Apply a snapshot computed on the background thread: the Overview data plus the session
    /// rows (one `sessions()` scan feeds both the tree and the Overview's session count). No
    /// SQLite runs on the UI thread here — just cheap tree building.
    fn apply(&mut self, mut ov: OverviewData, rows: Vec<SessionRow>) {
        ov.sessions = rows.len() as i64;
        self.overview = Some(ov);
        // Grand-total footer: total spend, overall savings %, and total messages.
        let bill: i64 = rows.iter().map(|r| r.bill_micros).sum();
        let turns: i64 = rows.iter().map(|r| r.turns).sum();
        let before: i64 = rows.iter().map(|r| r.input_before).sum();
        let after: i64 = rows.iter().map(|r| r.input_after).sum();
        let saved = if before > 0 {
            (before - after).max(0) as f64 / before as f64 * 100.0
        } else {
            0.0
        };
        let saved_m: i64 = rows.iter().map(|r| r.saved_micros).sum();
        self.sessions.footer = Some(vec![
            "total".into(),
            format!("${:.2}", bill as f64 / 1_000_000.0),
            format!("${:.2}", saved_m as f64 / 1_000_000.0),
            format!("{saved:.0}%"),
            String::new(),
            turns.to_string(),
            String::new(),
        ]);
        self.sessions.set_roots(build_session_tree(&rows));
    }

    /// Handle a key; returns true to quit.
    fn handle_key(&mut self, code: KeyCode) -> bool {
        // A help overlay swallows the next keypress to dismiss itself.
        if self.show_help {
            self.show_help = false;
            return false;
        }
        match code {
            KeyCode::Char('q') => return true,
            KeyCode::Char('?') => self.show_help = true,
            // `t` cycles the Catppuccin flavor (Mocha → Macchiato → Frappé → Latte); the next
            // frame repaints since every element reads its color live. Persist the choice to
            // the config file, best-effort — a write failure must never disrupt the TUI.
            KeyCode::Char('t') => {
                palette::cycle();
                let _ = llmtrim_core::config::save_theme(palette::ident());
            }
            // `d`/`f`/`u` queue a command and quit, so it runs on the normal screen (not inside
            // the alt-screen): `d` = doctor, `f` = ensure/fix integrations, `u` = update.
            KeyCode::Char('d') => {
                self.action = PostAction::Doctor;
                return true;
            }
            KeyCode::Char('f') if self.needs_ensure => {
                // Run ensure after alt-screen exit so `PostAction` stays semver-stable.
                self.pending_ensure = true;
                return true;
            }
            KeyCode::Char('u') => {
                // A newer release wins: download it first (so a restart lands on the newest
                // binary, not one that's itself outdated). Only when there's no newer release
                // but the running daemon is stale does `u` just restart onto the installed binary.
                let ov = self.overview.as_ref();
                self.action = if ov.is_some_and(|o| o.update_available.is_some()) {
                    PostAction::Update
                } else if ov.is_some_and(|o| o.status.kind == StatusKind::Stale) {
                    PostAction::Restart
                } else {
                    PostAction::Update
                };
                return true;
            }
            // One-time sub onboarding: `s` = show login steps (after exit), `n` = never again.
            KeyCode::Char('s') if self.show_sub_nudge && self.tab == Tab::Overview => {
                self.show_sub_nudge = false;
                self.pending_sub_login = true;
                return true;
            }
            KeyCode::Char('n') if self.show_sub_nudge && self.tab == Tab::Overview => {
                self.show_sub_nudge = false;
                let _ = crate::ensure::dismiss_sub_nudge();
                return false;
            }
            // `c` flips the Overview gauge between "% of new content" and "% of the whole prompt"
            // — the one figure with two honest framings. It's a no-op on the other tabs, which
            // have no size figure to reframe. Dollars are always the real net bill, so they
            // never move.
            KeyCode::Char('c') if self.tab == Tab::Overview => {
                self.whole_prompt = !self.whole_prompt;
            }
            // `y` launches the desktop tray, only when it's installed — otherwise the key is
            // inert and the hint is hidden. The tray is its own window, so we launch it in the
            // background and stay on the dashboard (best-effort: a failed spawn is swallowed
            // rather than corrupting the alt-screen). Unlike `d`/`u`, this does not quit.
            KeyCode::Char('y') if self.tray_available => {
                let _ = crate::tray::launch_detached();
            }
            KeyCode::Char('1') => self.tab = Tab::Overview,
            KeyCode::Char('2') => self.tab = Tab::Sessions,
            KeyCode::Char('3') => self.tab = Tab::Detail,
            // Tab cycles the top-level tabs everywhere (consistent). Shift-Tab steps the
            // Detail panes when there (else previous tab), so Tab never changes meaning.
            KeyCode::Tab => self.cycle_tab(),
            KeyCode::BackTab if self.tab == Tab::Detail && self.detail.is_some() => {
                if let Some(d) = &mut self.detail {
                    d.focus = match d.focus {
                        Pane::Occupancy => Pane::Cost,
                        Pane::Cost => Pane::Occupancy,
                    };
                }
            }
            KeyCode::BackTab => self.cycle_tab_back(),
            _ => match self.tab {
                Tab::Overview => self.overview_key(code),
                Tab::Sessions => self.sessions_key(code),
                Tab::Detail => self.detail_key(code),
            },
        }
        false
    }

    fn cycle_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Overview => Tab::Sessions,
            Tab::Sessions => Tab::Detail,
            Tab::Detail => Tab::Overview,
        };
    }

    fn cycle_tab_back(&mut self) {
        self.tab = match self.tab {
            Tab::Overview => Tab::Detail,
            Tab::Sessions => Tab::Overview,
            Tab::Detail => Tab::Sessions,
        };
    }

    fn overview_key(&mut self, _code: KeyCode) {
        // The native Overview is a single fixed screen — nothing to scroll.
    }

    fn sessions_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.sessions.move_up(),
            KeyCode::Down | KeyCode::Char('j') => self.sessions.move_down(),
            KeyCode::PageUp => self.sessions.move_page(-10),
            KeyCode::PageDown => self.sessions.move_page(10),
            KeyCode::Home | KeyCode::Char('g') => self.sessions.jump_top(),
            KeyCode::End | KeyCode::Char('G') => self.sessions.jump_bottom(),
            KeyCode::Left | KeyCode::Char('h') => {
                self.sessions.collapse();
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.sessions.expand();
            }
            KeyCode::Enter => self.activate_session(),
            _ => {}
        }
    }

    fn activate_session(&mut self) {
        let target = match self.sessions.toggle() {
            Activate::Leaf => self.sessions.selected().and_then(|n| match &n.data {
                Node::Session(id) => Some((id.clone(), n.label.clone())),
                Node::Group => None,
            }),
            _ => None,
        };
        if let Some((session_id, title)) = target {
            // Ask the detail worker to query this session off-thread; show a loading pane until
            // it answers. No SQLite runs on the UI thread here. If the send fails the worker has
            // exited (the loop's is_finished guard will quit) — don't show a pane that can never
            // resolve.
            if let Some(tx) = &self.detail_req
                && tx.send(session_id.clone()).is_err()
            {
                return;
            }
            let d = Detail {
                session_id,
                title,
                occupancy: TreeTable::new(
                    "context · this turn",
                    occupancy_columns(),
                    palette::frame(),
                ),
                cost: TreeTable::new("cost · cumulative", cost_columns(), palette::frame()),
                focus: Pane::Occupancy,
                loading: true,
            };
            self.detail = Some(d);
            self.tab = Tab::Detail;
        }
    }

    /// Install a Detail result from the worker, but only if it matches the session the user is
    /// currently drilled into — a stale result (the user drilled elsewhere before this query
    /// returned) is dropped. Returns true if it was installed (the frame needs a repaint).
    fn apply_detail(&mut self, snap: DetailSnapshot) -> bool {
        match &mut self.detail {
            Some(d) if d.session_id == snap.session_id => {
                install_detail(d, snap.data);
                true
            }
            _ => false,
        }
    }

    fn detail_key(&mut self, code: KeyCode) {
        let Some(d) = &mut self.detail else { return };
        // Esc returns to Sessions but keeps the drilled session loaded, so the Detail tab
        // stays populated and re-selectable. (Pane switch is Shift-Tab, in handle_key.)
        if matches!(code, KeyCode::Esc | KeyCode::Backspace) {
            self.tab = Tab::Sessions;
            return;
        }
        let pane = match d.focus {
            Pane::Occupancy => &mut d.occupancy,
            Pane::Cost => &mut d.cost,
        };
        match code {
            KeyCode::Up | KeyCode::Char('k') => pane.move_up(),
            KeyCode::Down | KeyCode::Char('j') => pane.move_down(),
            KeyCode::PageUp => pane.move_page(-10),
            KeyCode::PageDown => pane.move_page(10),
            KeyCode::Home | KeyCode::Char('g') => pane.jump_top(),
            KeyCode::End | KeyCode::Char('G') => pane.jump_bottom(),
            KeyCode::Left | KeyCode::Char('h') => {
                pane.collapse();
            }
            KeyCode::Right | KeyCode::Char('l') => {
                pane.expand();
            }
            KeyCode::Enter => {
                pane.toggle();
            }
            _ => {}
        }
    }

    fn render(&mut self, f: &mut Frame) {
        // Paint the base tone behind everything so panels sit on an intentional surface
        // instead of the terminal default — the whole UI reads as one designed object.
        f.render_widget(
            Block::default().style(Style::default().bg(palette::bg())),
            f.area(),
        );
        let chunks = Layout::vertical([
            Constraint::Length(1), // tab bar
            Constraint::Min(0),    // body
            Constraint::Length(1), // help line
        ])
        .split(f.area());

        self.render_tabs(f, chunks[0]);
        match self.tab {
            Tab::Overview => self.render_overview(f, chunks[1]),
            Tab::Sessions => self.sessions.render(f, chunks[1], true),
            Tab::Detail => self.render_detail(f, chunks[1]),
        }
        self.render_help(f, chunks[2]);
        if self.show_help {
            render_help_overlay(f, self.tray_available);
        }
    }

    fn render_tabs(&self, f: &mut Frame, area: Rect) {
        // Manual tab strip: the active tab is an accent-fill swatch (mauve bg, base fg, bold)
        // so focus reads instantly as navigation; inactive tabs recede to dim muted text and
        // the Detail tab is dimmer still until a session is drilled. The accent owns focus
        // everywhere, which frees blue to mean only money.
        let tabs = [
            (" 1 Overview ", Tab::Overview),
            (" 2 Sessions ", Tab::Sessions),
            (" 3 Detail ", Tab::Detail),
        ];
        let mut spans = Vec::new();
        for (i, (label, tab)) in tabs.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw(" "));
            }
            let style = if *tab == self.tab {
                Style::default()
                    .bg(palette::accent())
                    .fg(palette::bg())
                    .add_modifier(Modifier::BOLD)
            } else if *tab == Tab::Detail && self.detail.is_none() {
                Style::default()
                    .fg(palette::frame())
                    .add_modifier(Modifier::DIM)
            } else {
                Style::default()
                    .fg(palette::muted_gray())
                    .add_modifier(Modifier::DIM)
            };
            spans.push(Span::styled(*label, style));
        }

        // Right-aligned meta cluster: version · clock · refresh interval. Render into its own
        // sub-rect so Paragraph's area-wide style reset can't wipe the tab colors.
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let (h24, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
        let (ampm, h24m) = if h24 < 12 {
            ("AM", h24)
        } else {
            ("PM", h24 - 12)
        };
        let h12 = if h24m == 0 { 12 } else { h24m };
        let muted = Style::default().fg(palette::muted_gray());
        let mut meta = vec![Span::styled(
            format!("llmtrim v{}", env!("CARGO_PKG_VERSION")),
            muted,
        )];
        // A newer release? Show it next to the version in the caution hue (`u` updates).
        if let Some(v) = self
            .overview
            .as_ref()
            .and_then(|o| o.update_available.as_deref())
        {
            meta.push(Span::styled(
                format!("  ↑ v{v}"),
                Style::default()
                    .fg(palette::warn())
                    .add_modifier(Modifier::BOLD),
            ));
        }
        // Integrations stale / missing — `f` fixes without reading the changelog.
        if self.needs_ensure {
            meta.push(Span::styled(
                "  ⚠ fix",
                Style::default()
                    .fg(palette::warn())
                    .add_modifier(Modifier::BOLD),
            ));
        }
        meta.push(Span::styled(
            format!(
                " │ {h12}:{m:02}:{s:02} {ampm} │ ↻ {}s",
                self.refresh.as_secs().max(1)
            ),
            muted,
        ));
        let meta_w = (meta
            .iter()
            .map(|s| s.content.chars().count())
            .sum::<usize>() as u16
            + 1)
        .min(area.width);
        let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(meta_w)]).split(area);
        f.render_widget(Paragraph::new(Line::from(spans)), cols[0]);
        f.render_widget(
            Paragraph::new(Line::from(meta)).alignment(Alignment::Right),
            cols[1],
        );
    }

    fn render_overview(&mut self, f: &mut Frame, area: Rect) {
        let frame = Style::default().fg(palette::frame());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(frame)
            .title(" overview ")
            .title_style(frame.add_modifier(Modifier::BOLD));
        let full = block.inner(area);
        f.render_widget(block, area);
        // Cap content width on ultrawide terminals; center the capped region.
        const MAX_W: u16 = 120;
        let inner = if full.width > MAX_W {
            let x = full.x + (full.width - MAX_W) / 2;
            Rect::new(x, full.y, MAX_W, full.height)
        } else {
            full
        };
        // Borrow, don't clone: the sub-renderers take `&OverviewData`, so cloning the whole
        // struct (its Vecs + Strings) every frame was pure waste.
        let whole_prompt = self.whole_prompt;
        let Some(ov) = self.overview.as_ref() else {
            return;
        };
        match ov.status.kind {
            // When the proxy is down, the alarm is the hero and savings demote to an aside.
            StatusKind::Off | StatusKind::Degraded => render_overview_alert(f, inner, ov),
            // A stale daemon always shows the dashboard (and its `u  Update` nudge), even before
            // the first request — otherwise the empty-state card would hide the restart prompt.
            StatusKind::Stale => render_overview_main(f, inner, ov, whole_prompt),
            _ if !ov.has_traffic => render_overview_empty(f, inner, ov),
            _ => render_overview_main(f, inner, ov, whole_prompt),
        }
    }

    fn render_detail(&mut self, f: &mut Frame, area: Rect) {
        let Some(d) = &mut self.detail else {
            // No session drilled yet — guide the user instead of a blank pane.
            let p = Paragraph::new(
                Line::from("drill into a session on the Sessions tab (Enter) to see its breakdown")
                    .centered(),
            )
            .style(Style::default().add_modifier(Modifier::DIM))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(
                        Style::default()
                            .fg(palette::frame())
                            .add_modifier(Modifier::DIM),
                    )
                    .title(" detail "),
            );
            f.render_widget(p, area);
            return;
        };
        // The worker's queries haven't returned yet — show a placeholder instead of empty panes.
        if d.loading {
            let p = Paragraph::new(Line::from(format!("Loading {}…", d.title)).centered())
                .style(Style::default().add_modifier(Modifier::DIM))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(palette::frame()))
                        .title(" detail "),
                );
            f.render_widget(p, area);
            return;
        }
        // Too short for two stacked bordered panes — show the occupancy pane alone.
        if area.height < 10 {
            d.occupancy.title = format!("context · {}", d.title);
            d.occupancy.render(f, area, true);
            return;
        }
        let halves =
            Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);
        d.occupancy.title = format!("context · {} · latest turn", d.title);
        d.occupancy.render(f, halves[0], d.focus == Pane::Occupancy);
        d.cost.render(f, halves[1], d.focus == Pane::Cost);
    }

    fn render_help(&self, f: &mut Frame, area: Rect) {
        // `d repair` only when a real problem is detected (a stale daemon isn't a fault, it just
        // needs a restart); `u update` when a newer release exists or the daemon is stale — so
        // the bar isn't cluttered with actions that do nothing useful.
        let stale = self
            .overview
            .as_ref()
            .is_some_and(|o| o.status.kind == StatusKind::Stale);
        let problem = self
            .overview
            .as_ref()
            .is_some_and(|o| o.status.fix.is_some())
            && !stale;
        let update = stale
            || self
                .overview
                .as_ref()
                .is_some_and(|o| o.update_available.is_some());
        // `c` flips the saved-% basis on the Overview gauge only — that's the one screen with a
        // size figure to reframe — so the hint lives on that tab alone.
        let mut keys = match self.tab {
            Tab::Overview => String::from(" Tab tabs · c %"),
            Tab::Sessions => String::from(" Tab tabs · ↑↓ move · →/← expand · ⏎ drill"),
            Tab::Detail => String::from(" Tab tabs · ⇧Tab pane · ↑↓ move · →/← expand"),
        };
        if problem {
            keys.push_str(" · d doctor");
        }
        if self.needs_ensure {
            keys.push_str(" · f fix");
        }
        if update {
            keys.push_str(" · u update");
        }
        if self.tray_available {
            keys.push_str(" · y tray");
        }
        if self.show_sub_nudge && self.tab == Tab::Overview {
            keys.push_str(" · s sub · n dismiss");
        }
        keys.push_str(match self.tab {
            Tab::Overview => " · t theme · ? help · q quit",
            _ => " · t theme · q",
        });
        // A persistent bottom status bar: filled surface tone with muted text so it reads as
        // a bar rather than floating text on the base. The active flavor sits on the right.
        let bar = Style::default()
            .bg(palette::surface())
            .fg(palette::muted_gray());
        f.render_widget(Paragraph::new(Line::from(keys)).style(bar), area);
        let flavor = format!(" {} ", palette::name());
        let fw = flavor.chars().count() as u16;
        if area.width > fw {
            let right = Rect::new(area.x + area.width - fw, area.y, fw, 1);
            f.render_widget(
                Paragraph::new(Line::from(flavor)).style(
                    Style::default()
                        .bg(palette::surface())
                        .fg(palette::accent()),
                ),
                right,
            );
        }
    }
}

/// Centered full-screen keymap overlay, dismissed by any key. `tray` adds the
/// tray launch line only when the desktop app is installed.
fn render_help_overlay(f: &mut Frame, tray: bool) {
    let area = f.area();
    let w = 56.min(area.width);
    let h = (if tray { 21 } else { 20 }).min(area.height);
    let rect = Rect::new(
        area.x + (area.width.saturating_sub(w)) / 2,
        area.y + (area.height.saturating_sub(h)) / 2,
        w,
        h,
    );
    let mut lines = vec![
        Line::from(""),
        Line::from("  Tab / Shift-Tab    next / previous tab"),
        Line::from("  1 / 2 / 3          jump to Overview / Sessions / Detail"),
        Line::from("  ↑ ↓  or  j k       move cursor / scroll"),
        Line::from("  g / G              top / bottom"),
        Line::from("  → ← or l h         expand / collapse a row"),
        Line::from("  Enter              drill into a session"),
        Line::from("  Shift-Tab          switch pane (in Detail)"),
        Line::from("  Esc                back to Sessions"),
        Line::from("  t                  cycle theme (Mocha/Macchiato/Frappé/Latte)"),
        Line::from("  c                  size %: new content / whole prompt (Overview)"),
    ];
    if tray {
        lines.push(Line::from("  y                  open the desktop tray"));
    }
    lines.extend([
        Line::from("  f                  fix integrations (ensure)"),
        Line::from("  d                  doctor (diagnose)"),
        Line::from("  u                  update / restart stale daemon"),
        Line::from("  s / n              sub setup / dismiss (when shown)"),
        Line::from(""),
        Line::from("  \"you paid\" = what the proxy billed (same as Sessions)."),
        Line::from("  \"saved$\" = input $ off at rates locked per turn."),
        Line::from("  \"trim\" = token size % (not dollars)."),
        Line::from("  ? quit this help   q quit the app"),
    ]);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(
            Style::default()
                .fg(palette::accent())
                .add_modifier(Modifier::BOLD),
        )
        .title(" keys ")
        .title_style(
            Style::default()
                .fg(palette::accent())
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .style(Style::default().bg(palette::surface()).fg(palette::text())),
        rect,
    );
}

// ── native Overview rendering ─────────────────────────────────────────────────────

/// Format a USD amount with thousands grouping: `1234.5 → "$1,234.50"`. The bill is genuinely
/// in USD (provider rates are USD), so `$` is the real billing currency, not a locale guess.
fn money(v: f64) -> String {
    let cents = (v.abs() * 100.0).round() as u64;
    let (whole, frac) = (cents / 100, cents % 100);
    let digits = whole.to_string();
    let n = digits.len();
    let mut grouped = String::with_capacity(n + n / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (n - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(c);
    }
    format!("{}${grouped}.{frac:02}", if v < 0.0 { "-" } else { "" })
}

/// The savings percentage off the would-have bill (0 when nothing priced).
fn saved_pct(ov: &OverviewData) -> f64 {
    match (ov.saved_usd, ov.would_have_usd) {
        (Some(s), Some(w)) if w > 0.0 => (s / w * 100.0).clamp(0.0, 100.0),
        _ => 0.0,
    }
}

/// USD rounded to whole dollars for the trend-bar value labels; keeps thousands grouping.
fn money_round(v: f64) -> String {
    money((v.abs()).round()).trim_end_matches(".00").to_string()
}

/// A standard card: rounded border, plain Title-Case title, 1-col interior padding. `focal`
/// gives the hero its bold accent (green) border so it dominates the grid.
fn card(title: &str, focal: bool) -> Block<'_> {
    let border = if focal {
        Style::default()
            .fg(palette::green())
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette::frame())
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border)
        .title(format!(" {title} "))
        .title_style(Style::default().fg(palette::muted_gray()))
        .padding(Padding::new(1, 1, 0, 0))
}

/// The spoken health line: a calm dot + plain words (no port/pid jargon).
fn status_line(ov: &OverviewData) -> Paragraph<'static> {
    // Healthy + working = the win hue; on but idle stays a quiet neutral so blue is only money.
    let dot = if matches!(ov.status.kind, StatusKind::Working) {
        palette::green()
    } else {
        palette::muted_gray()
    };
    Paragraph::new(Line::from(vec![
        Span::styled("● ", Style::default().fg(dot)),
        Span::styled(
            ov.status.text.clone(),
            Style::default().fg(palette::muted_gray()),
        ),
    ]))
}

/// Group an integer with thousands separators: `33142 → "33,142"`.
fn group(n: i64) -> String {
    let s = n.abs().to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

/// Weekday abbreviations for the last `n` days (oldest→newest, matching `trend_daily_usd`).
/// 1970-01-01 was a Thursday, so `days_since_epoch % 7 == 0` is Thursday.
fn weekday_labels(n: usize) -> Vec<&'static str> {
    const NAMES: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs() / 86_400) as i64)
        .unwrap_or(0);
    (0..n as i64)
        .map(|i| {
            let d = days - (n as i64 - 1) + i;
            NAMES[(((d % 7) + 7) % 7) as usize]
        })
        .collect()
}

/// Band B — the status banner: a ✓ health icon + the spoken health line and a quip, with an
/// optional right-aligned `u  Update` button (a newer release, or a stale daemon to restart).
/// Full width, rounded border.
fn render_status_banner(f: &mut Frame, area: Rect, ov: &OverviewData) {
    // Stale shares Degraded's warn dot; harmless because they never both reach this banner
    // (Degraded routes to the Repair alert instead).
    let dot = match ov.status.kind {
        StatusKind::Working => palette::green(),
        StatusKind::Ready => palette::blue(),
        StatusKind::Stale => palette::warn(),
        StatusKind::Degraded => palette::warn(),
        StatusKind::Off => palette::alarm(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::frame()))
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dim = Style::default().fg(palette::muted_gray());
    let lines = vec![
        Line::from(vec![
            Span::styled("✓  ", Style::default().fg(dot)),
            Span::styled(
                ov.status.text.clone(),
                Style::default()
                    .fg(palette::text())
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(sheep_quip(ov, secs), dim)),
    ];
    // The healthy banner has no "problem", so no Repair button here (it lives on the alert
    // screen). Show a high-visibility Update button when a newer release exists, or when the
    // running daemon is stale (a restart applies the installed update — also driven by `u`).
    if ov.update_available.is_some() || ov.status.kind == StatusKind::Stale {
        let btn = " u  Update ";
        let w = (btn.chars().count() as u16).min(inner.width);
        let cols = Layout::horizontal([Constraint::Min(10), Constraint::Length(w)]).split(inner);
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), cols[0]);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                btn,
                Style::default()
                    .bg(palette::warn())
                    .fg(palette::bg())
                    .add_modifier(Modifier::BOLD),
            ))),
            cols[1],
        );
    } else {
        // No update to nudge → the right slot hosts a tiny state-driven sheep: it snoozes when
        // idle and looks around when traffic is flowing. (Off/Degraded never reach here — that
        // screen keeps the serious ⚠/Repair tone instead.)
        let face = sheep_mascot(ov.status.kind, secs);
        let w = 9u16.min(inner.width);
        let cols = Layout::horizontal([Constraint::Min(10), Constraint::Length(w)]).split(inner);
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), cols[0]);
        f.render_widget(
            Paragraph::new(vec![Line::from(""), Line::from(Span::styled(face, dim))])
                .alignment(Alignment::Right),
            cols[1],
        );
    }
}

/// A tiny clock-driven sheep face for the status banner's right slot — `Working` looks around
/// (watching traffic), `Ready` snoozes with Z's. Frames within a set are fixed-width so the
/// right-aligned face doesn't jitter.
fn sheep_mascot(kind: StatusKind, secs: u64) -> &'static str {
    // Step a frame every 2s (calm, not a flicker) and only swap the whole animation every 24s.
    let f = (secs / 2 % 4) as usize;
    // Frames within each animation are equal-width so the right-aligned face never jitters.
    let blink = ["(o.o)", "(o.o)", "(-.-)", "(o.o)"]; // mostly open, a quick close
    let look = ["(o.o)", "(o.-)", "(-.o)", "(o.o)"]; // eyes dart around — watching traffic
    let perk = ["(o.o)", "(O.O)", "(O.o)", "(O.O)"]; // wide-eyed, alert
    let happy = ["(o.o)", "(^.^)", "(o.o)", "(^.^)"]; // content beam
    let chew = ["(o.o)", "(o_o)", "(o.o)", "(o_o)"]; // calm rumination
    let snooze = ["(-.-)    ", "(-.-) z  ", "(-.-) zZ ", "(-.-) zZz"]; // padded to 9 cells
    let pick = (secs / 24) as usize;
    match kind {
        // Working = traffic flowing: alert + cheerful faces.
        StatusKind::Working => {
            let sets = [look, perk, happy, blink];
            sets[pick % sets.len()][f]
        }
        // Idle: calm faces — snooze, ruminate, blink.
        _ => {
            let sets = [snooze, chew, blink];
            sets[pick % sets.len()][f]
        }
    }
}

/// A woozy "needs a restart" sheep for the Degraded alert (single row, 5 cells).
fn sheep_dizzy(secs: u64) -> &'static str {
    ["(o.O)", "(*.*)", "(O.o)", "(*.*)"][(secs % 4) as usize]
}

/// An on-brand sheep-shearing quip for the status banner's second line — "wool/fluff" stands
/// in for the trimmed tokens (also our jargon-free word for them). Most carry a real figure so
/// the joke never lies; rotates by the clock so the banner feels alive without any state.
fn sheep_quip(ov: &OverviewData, secs: u64) -> String {
    // Pick the line first, then format only that one (was building all 21 strings per frame).
    let pct = ov.pct_less * 100.0;
    // When billing is unavailable, never say "$0.00 saved" — stay on token/fluff quips only.
    if ov.money_unavailable || ov.saved_usd.is_none() {
        const N: u64 = 8;
        return match (secs / 12) % N {
            0 => format!("Shorn {pct:.0}% of the fluff off your prompts"),
            1 => format!(
                "Trimmed the dead weight off {} requests",
                group(ov.requests)
            ),
            2 => format!("Fleece gone, answers intact — {pct:.0}% lighter prompts"),
            3 => format!(
                "{} prompts through the shears, not one bleat",
                group(ov.requests)
            ),
            4 => "Wool off, wisdom on.".to_string(),
            5 => {
                format!("Sheared {pct:.0}% off your prompts — wool's on the floor, answers aren't")
            }
            6 => "Less wool in, same wisdom out".to_string(),
            _ => format!("Shears down. {pct:.0}% gone. Nothing important bleated."),
        };
    }
    const N: u64 = 21;
    // One line every 12s — fast enough to feel alive, slow enough to actually read.
    match (secs / 12) % N {
        0 => format!("Shorn {pct:.0}% of the fluff off your prompts"),
        1 => format!(
            "Trimmed the dead weight off {} requests",
            group(ov.requests)
        ),
        2 => format!(
            "{} of wool left on the floor — same flock of answers",
            money(ov.saved_usd.unwrap_or(0.0))
        ),
        3 => format!("Fleece gone, answers intact — {pct:.0}% lighter prompts"),
        4 => format!(
            "{} prompts through the shears, not one bleat",
            group(ov.requests)
        ),
        5 => "Wool off, wisdom on.".to_string(),
        6 => format!(
            "Ewe pocketed {} today — same flock of answers",
            money(ov.saved_today_usd.unwrap_or(0.0))
        ),
        7 => format!("Sheared {pct:.0}% off your prompts — wool's on the floor, answers aren't"),
        8 => format!("{} requests shorn, $0 of meaning lost", group(ov.requests)),
        9 => format!(
            "{} saved all-time, one fleece at a time",
            money(ov.saved_usd.unwrap_or(0.0))
        ),
        10 => format!("Trimmed {pct:.0}% of fluff. The model didn't notice. Ewe will."),
        11 => format!(
            "{} lighter today — pure wool, no muscle",
            money(ov.saved_today_usd.unwrap_or(0.0))
        ),
        12 => format!("Flock of {} prompts, freshly shorn", group(ov.requests)),
        13 => format!(
            "Bald prompts, full wallets: {} and counting",
            money(ov.saved_usd.unwrap_or(0.0))
        ),
        14 => format!(
            "{} trimmed this week. The fluff doesn't grow back.",
            money(ov.trend_daily_usd.iter().sum())
        ),
        15 => "Less wool in, same wisdom out".to_string(),
        16 => format!("Shears down. {pct:.0}% gone. Nothing important bleated."),
        17 => format!(
            "{} sessions, {} prompts, not a gram of wool wasted",
            ov.sessions,
            group(ov.requests)
        ),
        18 => "We took the coat, left the sheep".to_string(),
        19 => format!(
            "{} that never hit your bill",
            money(ov.saved_usd.unwrap_or(0.0))
        ),
        _ => format!(
            "Today's haul: {} of trimmed wool",
            money(ov.saved_today_usd.unwrap_or(0.0))
        ),
    }
}

/// Band C — five KPI tiles. At full width all five show (mockup order); narrower widths drop
/// the least critical (would-have, then paid) and keep saved/today/week.
fn render_kpis(f: &mut Frame, area: Rect, ov: &OverviewData) {
    let dim = Style::default().fg(palette::muted_gray());
    let green_b = Style::default()
        .fg(palette::green())
        .add_modifier(Modifier::BOLD);
    let blue_b = Style::default()
        .fg(palette::blue())
        .add_modifier(Modifier::BOLD);
    let white_b = Style::default()
        .fg(palette::text())
        .add_modifier(Modifier::BOLD);
    let warn_b = Style::default()
        .fg(palette::warn())
        .add_modifier(Modifier::BOLD);

    // Compressions without breakdown bills / all unpriced: never paint "$0.00" as truth.
    if ov.money_unavailable {
        let block = card("BILLING", false);
        let cell = block.inner(area);
        f.render_widget(block, area);
        let chip = if ov.coverage_turns == 0 && ov.coverage_compressions > 0 {
            format!(
                "0 proxy turns billed · {} compression log events",
                ov.coverage_compressions
            )
        } else if ov.turns_unpriced > 0 {
            format!("{} turns unpriced (unknown model rates)", ov.turns_unpriced)
        } else {
            "no proxy-attributed turns yet".into()
        };
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "No session bill yet — token savings still count",
                    warn_b,
                )),
                Line::from(Span::styled(
                    "Dollars need traffic through the proxy (not CLI/MCP alone)",
                    dim,
                )),
                Line::from(Span::styled(chip, dim)),
            ]),
            cell,
        );
        return;
    }

    let saved = ov.saved_usd.unwrap_or(0.0);
    let paid = ov.paid_usd.unwrap_or(0.0);
    let would = ov.would_have_usd.unwrap_or(0.0);
    let today = ov.saved_today_usd.unwrap_or(0.0);
    let n = ov.trend_daily_usd.len();
    let yesterday = if n >= 2 {
        ov.trend_daily_usd[n - 2]
    } else {
        0.0
    };
    let delta = today - yesterday;
    let week: f64 = ov.trend_daily_usd.iter().sum();
    let coverage_note = if ov.coverage_compressions > 0
        && ov.coverage_turns > 0
        && ov.coverage_turns < ov.coverage_compressions
    {
        format!(
            " · {}/{} attributed",
            ov.coverage_turns, ov.coverage_compressions
        )
    } else if ov.turns_unpriced > 0 {
        format!(" · {} unpriced", ov.turns_unpriced)
    } else {
        String::new()
    };

    // Plain-language subtitles. Money % = saved / would-have (would-have includes output).
    let tiles: [(String, String, Style, String); 5] = [
        (
            "TOTAL SAVED".into(),
            money(saved),
            green_b,
            format!(
                "{:.0}% less than without trim{coverage_note}",
                saved_pct(ov)
            ),
        ),
        (
            "YOU PAID".into(),
            money(paid),
            blue_b,
            "what the proxy billed".into(),
        ),
        (
            "WOULD HAVE COST".into(),
            money(would),
            white_b,
            "without input trim".into(),
        ),
        (
            "SAVED TODAY".into(),
            money(today),
            green_b,
            format!(
                "{}{} vs yest. (UTC)",
                if delta < 0.0 { "-" } else { "+" },
                money(delta.abs())
            ),
        ),
        (
            "SAVED THIS WEEK".into(),
            money(week),
            green_b,
            "7 days (UTC)".into(),
        ),
    ];
    // Priority when we can't fit all five (keep saved/today/week first).
    const PRIORITY: [usize; 5] = [0, 3, 4, 1, 2];
    let fit = (area.width / 18).clamp(1, 5) as usize;
    let mut idxs: Vec<usize> = if fit >= 5 {
        (0..5).collect()
    } else {
        let mut v: Vec<usize> = PRIORITY[..fit].to_vec();
        v.sort_unstable();
        v
    };
    idxs.truncate(fit);

    let cols = Layout::horizontal(vec![Constraint::Ratio(1, idxs.len() as u32); idxs.len()])
        .spacing(1)
        .split(area);
    for (slot, &i) in idxs.iter().enumerate() {
        let (label, value, vstyle, sub) = &tiles[i];
        let block = card(label, false);
        let cell = block.inner(cols[slot]);
        f.render_widget(block, cols[slot]);
        // Truncate to the cell width (with an ellipsis) so a long subtitle clips cleanly at a
        // boundary instead of being chopped mid-word by the terminal.
        let w = cell.width as usize;
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(super::tree::truncate_w(value, w), *vstyle)),
                Line::from(Span::styled(super::tree::truncate_w(sub, w), dim)),
            ]),
            cell,
        );
    }
}

/// Band D-left — savings-trend vertical bar chart with $-labeled bars + weekday labels; the
/// most recent day is brightened.
fn render_trend(f: &mut Frame, area: Rect, ov: &OverviewData) {
    let block = card("SAVINGS TREND · $/DAY", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if ov.money_unavailable || ov.saved_usd.is_none() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "— needs proxy bills",
                Style::default().fg(palette::muted_gray()),
            )))
            .alignment(Alignment::Center),
            inner,
        );
        return;
    }
    if ov.trend_daily_usd.is_empty() || inner.width == 0 {
        return;
    }
    // The SVG exporter replaces this card's block-bars with a real vector bar chart; record
    // where it landed. Compiled out of the real binary.
    #[cfg(test)]
    tests::capture_trend_rect(inner);
    let labels = weekday_labels(ov.trend_daily_usd.len());
    let last = ov.trend_daily_usd.len() - 1;
    let green = Style::default().fg(palette::green());
    let green_b = green.add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(palette::muted_gray());
    let dim_b = dim.add_modifier(Modifier::BOLD);
    // Reserve a row for the $ value and a row for the weekday, both BELOW the bars, so the
    // amount sits between the bar and the day instead of being painted over the bar (ratatui's
    // BarChart can only overlay `text_value` on the bar base).
    let rows = Layout::vertical([
        Constraint::Min(1),    // bars
        Constraint::Length(1), // $ value
        Constraint::Length(1), // weekday
    ])
    .split(inner);
    let bars: Vec<Bar> = ov
        .trend_daily_usd
        .iter()
        .enumerate()
        .map(|(i, _)| {
            Bar::default()
                .value((ov.trend_daily_usd[i] * 100.0).round().max(0.0) as u64)
                .text_value(String::new())
                .style(if i == last { green_b } else { green })
        })
        .collect();
    // Keep bars slim (≤3 cells) so a 3-char value like "$24" fills exactly instead of being
    // padded with block glyphs; widen the gap to use the leftover width.
    let n = bars.len() as u16;
    let bw = (inner.width / n).saturating_sub(1).clamp(1, 3);
    let gap = if bw <= 2 { 1 } else { 2 };
    let chart = BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width(bw)
        .bar_gap(gap)
        .value_style(Style::default().fg(palette::muted_gray()))
        .label_style(Style::default().fg(palette::muted_gray()));
    f.render_widget(chart, rows[0]);

    // Center each label under its bar (bars start flush-left at inner.x, pitch = bw + gap).
    let pitch = (bw + gap) as usize;
    let bw = bw as usize;
    let centered_row = |cells: Vec<(String, Style)>| -> Line<'static> {
        let mut spans: Vec<Span> = Vec::new();
        let mut col = 0usize;
        for (i, (text, style)) in cells.into_iter().enumerate() {
            let len = text.chars().count();
            let start = i * pitch + bw.saturating_sub(len) / 2;
            if start > col {
                spans.push(Span::raw(" ".repeat(start - col)));
                col = start;
            }
            col += len;
            spans.push(Span::styled(text, style));
        }
        Line::from(spans)
    };
    let dollars = centered_row(
        ov.trend_daily_usd
            .iter()
            .enumerate()
            .map(|(i, v)| (money_round(*v), if i == last { dim_b } else { dim }))
            .collect(),
    );
    let days = centered_row(labels.iter().map(|l| (l.to_string(), dim)).collect());
    f.render_widget(Paragraph::new(dollars), rows[1]);
    f.render_widget(Paragraph::new(days), rows[2]);
}

/// Band D-mid — the "you send smaller requests" gauge: the percentage, a bar meter, and the
/// 0–100 scale. (ratatui has no arc widget; a horizontal Gauge is the honest built-in.)
fn render_gauge(f: &mut Frame, area: Rect, ov: &OverviewData, whole_prompt: bool) {
    let block = card("SMALLER REQUESTS", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let pct = ov.pct_smaller(whole_prompt).clamp(0.0, 1.0);
    let dim = Style::default().fg(palette::muted_gray());
    let rows = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(1), // "38% smaller"
        Constraint::Length(1), // gauge
        Constraint::Length(1), // 0% .. 100%
        Constraint::Length(2), // caption
        Constraint::Min(0),
    ])
    .split(inner);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{:.0}% smaller", pct * 100.0),
            Style::default()
                .fg(palette::green())
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center),
        rows[1],
    );
    // The SVG exporter replaces this Gauge with a rounded vector progress bar.
    #[cfg(test)]
    tests::capture_gauge_rect(rows[2]);
    f.render_widget(
        Gauge::default()
            .ratio(pct)
            .label("")
            .gauge_style(Style::default().fg(palette::green()).bg(palette::frame())),
        rows[2],
    );
    let ends = Layout::horizontal([Constraint::Min(0), Constraint::Length(4)]).split(rows[3]);
    f.render_widget(Paragraph::new(Line::from(Span::styled("0%", dim))), ends[0]);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled("100%", dim))).alignment(Alignment::Right),
        ends[1],
    );
    // Second caption line names the active basis, so the gauge is never ambiguous (the `c` key
    // that flips it is documented in the footer and help overlay, not crammed in here).
    let basis = if whole_prompt {
        "of the whole prompt"
    } else {
        "of new content"
    };
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("Average size reduction", dim)),
            Line::from(Span::styled(basis, dim)),
        ])
        .alignment(Alignment::Center),
        rows[4],
    );
}

/// Band D-right — TOP MODELS by $ saved: ranked rows (rank · name · inline bar · $) with a
/// bold Total row that reconciles to all-time saved.
fn render_savers(f: &mut Frame, area: Rect, ov: &OverviewData) {
    let block = card("TOP MODELS", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if ov.money_unavailable || ov.saved_usd.is_none() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "— needs proxy bills",
                Style::default().fg(palette::muted_gray()),
            )))
            .alignment(Alignment::Center),
            inner,
        );
        return;
    }
    let split = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner);

    let max = ov.savers.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
    let amt_w = 9u16;
    let name_w = 18u16.min(split[0].width.saturating_sub(amt_w + 6));
    let bar_w = split[0].width.saturating_sub(name_w + amt_w + 6);
    let green = Style::default().fg(palette::green());
    let dim = Style::default().fg(palette::muted_gray());
    let rows = ov.savers.iter().enumerate().map(|(i, (name, usd))| {
        let filled = if max > 0.0 {
            ((*usd / max) * bar_w as f64).round() as usize
        } else {
            0
        };
        Row::new(vec![
            Cell::from(Span::styled(format!("{}", i + 1), dim)),
            Cell::from(super::tree::truncate_w(name, name_w as usize)),
            Cell::from(Span::styled("█".repeat(filled.min(bar_w as usize)), green)),
            Cell::from(
                Line::from(Span::styled(
                    money(*usd),
                    green.add_modifier(Modifier::BOLD),
                ))
                .right_aligned(),
            ),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Length(name_w),
            Constraint::Min(1),
            Constraint::Length(amt_w),
        ],
    );
    f.render_widget(table, split[0]);

    // "All models" (not sum of visible top-N rows) — all-time saved from MoneyTotals.
    let total = ov.saved_usd.unwrap_or(0.0);
    let tcols = Layout::horizontal([Constraint::Min(1), Constraint::Length(amt_w)]).split(split[1]);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "All models",
            Style::default()
                .fg(palette::muted_gray())
                .add_modifier(Modifier::BOLD),
        ))),
        tcols[0],
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            money(total),
            green.add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Right),
        tcols[1],
    );
}

/// Band E — three honest operational metric tiles: requests handled, the average added delay
/// (the real tradeoff, in the caution hue), and how many sessions were covered.
fn render_metrics(f: &mut Frame, area: Rect, ov: &OverviewData) {
    let dim = Style::default().fg(palette::muted_gray());
    let white_b = Style::default()
        .fg(palette::text())
        .add_modifier(Modifier::BOLD);
    let warn_b = Style::default()
        .fg(palette::warn())
        .add_modifier(Modifier::BOLD);
    let overhead = match ov.added_ms {
        Some(ms) if ms >= 1.0 => format!("~{ms:.0}ms"),
        Some(_) => "<1ms".into(),
        None => "—".into(),
    };
    // (title, value, value-style, caption)
    let tiles: [(&str, String, Style, &str); 3] = [
        ("REQUESTS HANDLED", group(ov.requests), white_b, "total"),
        (
            "ADDED DELAY",
            overhead,
            warn_b,
            "per request, you won't notice",
        ),
        (
            "SESSIONS COVERED",
            group(ov.sessions),
            white_b,
            "since you turned it on",
        ),
    ];
    let cols = Layout::horizontal([Constraint::Ratio(1, 3); 3])
        .spacing(1)
        .split(area);
    for (slot, (title, value, vstyle, caption)) in tiles.iter().enumerate() {
        let block = card(title, false);
        let cell = block.inner(cols[slot]);
        f.render_widget(block, cols[slot]);
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(value.clone(), *vstyle)),
                Line::from(Span::styled(*caption, dim)),
            ]),
            cell,
        );
    }
}

/// Healthy Overview — the edge-to-edge dashboard from the design mockup: status banner ·
/// 5 KPI tiles · charts row (trend | gauge | top models) · 5 metric tiles · footer banner.
/// Bands drop under height pressure; tiny terminals fall back to the spoken status line.
fn render_overview_main(f: &mut Frame, inner: Rect, ov: &OverviewData, whole_prompt: bool) {
    // Tiny terminals: never hand widgets zero-height rects.
    if inner.height < 6 {
        f.render_widget(status_line(ov), inner);
        return;
    }

    // Band heights. The charts row is CAPPED (not a greedy sponge): tall terminals would make
    // the three cards absurdly tall, so cap them and let the leftover height become equal empty
    // space above and below — the vertical equivalent of the MAX_W width centering. Drop the
    // metric tiles, then the KPI row, as height shrinks. (No footer band — the status banner and
    // meta bar already cover reassurance + freshness.)
    const CHART_H: u16 = 12;
    let with_metrics = inner.height >= 16;
    let with_kpis = inner.height >= 12;
    let fixed = 4 + u16::from(with_kpis) * 4 + u16::from(with_metrics) * 4;
    // Center vertically only when there's more room than the content wants; otherwise let the
    // charts band flex down so nothing is clipped on a short terminal.
    let centered = inner.height > fixed + CHART_H;

    let mut bands: Vec<Constraint> = Vec::new();
    if centered {
        bands.push(Constraint::Min(0)); // top spacer (empty)
    }
    bands.push(Constraint::Length(4)); // B: status banner
    if with_kpis {
        bands.push(Constraint::Length(4)); // C: KPI tiles
    }
    bands.push(if centered {
        Constraint::Length(CHART_H) // D: charts row (capped)
    } else {
        Constraint::Min(8) // D: charts row (flex when cramped)
    });
    if with_metrics {
        bands.push(Constraint::Length(4)); // E: metric tiles
    }
    if centered {
        bands.push(Constraint::Min(0)); // bottom spacer (empty)
    }
    let rows = Layout::vertical(bands).spacing(0).split(inner);

    let mut i = usize::from(centered); // skip the top spacer when centered
    render_status_banner(f, rows[i], ov);
    i += 1;
    if with_kpis {
        render_kpis(f, rows[i], ov);
        i += 1;
    }

    // Charts row reflows by width: 3 cols (trend | gauge | top models) → 2 (trend | top
    // models) → 1 (top models, the most useful single panel).
    let charts = rows[i];
    i += 1;
    if charts.width >= 96 {
        let c = Layout::horizontal([
            Constraint::Percentage(40),
            Constraint::Percentage(24),
            Constraint::Percentage(36),
        ])
        .spacing(1)
        .split(charts);
        render_trend(f, c[0], ov);
        render_gauge(f, c[1], ov, whole_prompt);
        render_savers(f, c[2], ov);
    } else if charts.width >= 60 {
        let c = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .spacing(1)
            .split(charts);
        render_trend(f, c[0], ov);
        render_savers(f, c[1], ov);
    } else {
        render_savers(f, charts, ov);
    }

    if with_metrics {
        render_metrics(f, rows[i], ov);
    }
}

/// Off/Degraded Overview: stat tiles of stale data would mislead, so a single full-width
/// message panel replaces the grid — the symptom, the literal fix, and the frozen all-time
/// saving so the user remembers what's at stake.
fn render_overview_alert(f: &mut Frame, inner: Rect, ov: &OverviewData) {
    let off = ov.status.kind == StatusKind::Off;
    let accent = if off {
        palette::alarm()
    } else {
        palette::warn()
    };
    let dim = Style::default().fg(palette::muted_gray());
    // Degraded is a real fault (e.g. not set up) — a woozy sheep keeps it light. Off is the
    // dangerous "can't reach the API" case, mascot-free and serious. (Stale isn't a fault and
    // never reaches this alert; it shows the normal dashboard with a `u  Update` nudge.)
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut headline = vec![Span::styled(
        format!("⚠  {}", ov.status.text),
        Style::default().fg(accent).add_modifier(Modifier::BOLD),
    )];
    if !off {
        headline.push(Span::styled(format!("   {}", sheep_dizzy(secs)), dim));
    }
    let mut lines: Vec<Line> = vec![Line::from(headline), Line::from("")];
    if off {
        lines.push(Line::from(Span::styled(
            "Your AI tools are routed through llmtrim, so while it's off",
            Style::default().fg(palette::text()),
        )));
        lines.push(Line::from(Span::styled(
            "their requests may fail. Turn it back on:",
            Style::default().fg(palette::text()),
        )));
        lines.push(Line::from(""));
    }
    if let Some(fix) = &ov.status.fix {
        lines.push(Line::from(vec![
            Span::styled("  run:  ", dim),
            Span::styled(fix.clone(), Style::default().fg(accent)),
        ]));
    }
    if let Some(un) = &ov.status.uninstall {
        lines.push(Line::from(vec![
            Span::styled("  or remove it entirely:  ", dim),
            Span::styled(un.clone(), dim),
        ]));
    }
    if let Some(saved) = ov.saved_usd.filter(|s| *s >= 0.005) {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Saved while it was on:  ", dim),
            Span::styled(money(saved), Style::default().fg(palette::green())),
        ]));
    }
    let block = card("llmtrim", false).border_style(Style::default().fg(accent));
    let binner = block.inner(inner);
    f.render_widget(block, inner);
    // A problem is showing, so surface a high-visibility Repair button (runs doctor on exit).
    let btn = " d  Repair ";
    let w = (btn.chars().count() as u16).min(binner.width);
    let cols = Layout::horizontal([Constraint::Min(10), Constraint::Length(w)]).split(binner);
    f.render_widget(Paragraph::new(lines), cols[0]);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            btn,
            Style::default()
                .bg(accent)
                .fg(palette::bg())
                .add_modifier(Modifier::BOLD),
        ))),
        cols[1],
    );
}

/// Ready-but-idle Overview: a calm full-width panel, never a fake $0.00.
fn render_overview_empty(f: &mut Frame, inner: Rect, ov: &OverviewData) {
    let dot = match ov.status.kind {
        StatusKind::Working => palette::green(),
        _ => palette::blue(),
    };
    let lines = vec![
        Line::from(vec![
            Span::styled("● ", Style::default().fg(dot)),
            Span::styled(ov.status.text.clone(), Style::default().fg(palette::text())),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Nothing saved yet.",
            Style::default()
                .fg(palette::green())
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "llmtrim is on and watching. The next time your AI tools make a",
            Style::default().fg(palette::text()),
        )),
        Line::from(Span::styled(
            "request, it starts trimming the bill — and the savings show up here.",
            Style::default().fg(palette::text()),
        )),
    ];
    let block = card("llmtrim", false);
    f.render_widget(Paragraph::new(lines).block(block), inner);
}

// ── column definitions ──────────────────────────────────────────────────────────

fn session_columns() -> Vec<Column> {
    // Decision metric (cost) leads; `saved$` is money (frozen blend); `trim` is token size %
    // (never called "saved" — that word is dollars elsewhere). `reuse` is cache hit on leaves.
    vec![
        Column::left("agent · project · session", 32),
        Column::right("cost", 9).colored(palette::blue()),
        Column::right("saved$", 8).colored(palette::green()),
        Column::right("trim", 5),
        Column::right("when", 6),
        Column::right("msgs", 5),
        Column::right("reuse", 6),
    ]
}

fn occupancy_columns() -> Vec<Column> {
    // "used%" = share of the context window this source takes (the decision metric here, so
    // it leads and carries the focus hue, blue); the bar draws the same ratio in the same hue.
    // `tokens` is a supporting count and rides neutral.
    vec![
        Column::left("source", 28),
        Column::right("used%", 6).colored(palette::blue()),
        Column::left("", 16).colored(palette::blue()), // the █░░░ occupancy bar
        Column::right("tokens", 8),
    ]
}

fn cost_columns() -> Vec<Column> {
    // On Detail you're diagnosing spend, not celebrating: total$ is the decision metric, so it
    // leads in blue (the bill hue). The cache split (read/write/new) and %bill are diagnostic
    // detail and ride neutral — coloring them would rainbow the pane.
    vec![
        Column::left("source", 24),
        Column::right("total$", 9).colored(palette::blue()),
        Column::right("read$", 8),
        Column::right("write$", 8),
        Column::right("new$", 8),
        Column::right("%bill", 6),
    ]
}

// ── tree builders ───────────────────────────────────────────────────────────────

/// Group session rows into an agent → project → session tree with rolled-up columns.
fn build_session_tree(rows: &[SessionRow]) -> Vec<TreeNode<Node>> {
    use std::collections::BTreeMap;
    // Preserve newest-first order from the query while grouping.
    let mut agents: Vec<String> = Vec::new();
    let mut by_agent: BTreeMap<String, Vec<&SessionRow>> = BTreeMap::new();
    for r in rows {
        if !by_agent.contains_key(&r.agent) {
            agents.push(r.agent.clone());
        }
        by_agent.entry(r.agent.clone()).or_default().push(r);
    }

    let mut out = Vec::new();
    for agent in &agents {
        let sessions = &by_agent[agent];
        let mut by_proj: BTreeMap<String, Vec<&SessionRow>> = BTreeMap::new();
        let mut proj_order: Vec<String> = Vec::new();
        for s in sessions {
            let key = s
                .project
                .clone()
                .unwrap_or_else(|| "(no project)".to_string());
            if !by_proj.contains_key(&key) {
                proj_order.push(key.clone());
            }
            by_proj.entry(key).or_default().push(s);
        }
        let mut proj_nodes = Vec::new();
        for proj in &proj_order {
            let ss = &by_proj[proj];
            let session_nodes: Vec<TreeNode<Node>> = ss
                .iter()
                .map(|s| {
                    let label = s
                        .session_name
                        .clone()
                        .unwrap_or_else(|| short_id(&s.session_id));
                    TreeNode::leaf(label, session_cols(s), Node::Session(s.session_id.clone()))
                })
                .collect();
            // Labels stay stable (no volatile session count) so expansion state survives
            // refreshes — the session count is implicit in the visible child rows.
            let base = proj.rsplit('/').next().unwrap_or(proj).to_string();
            proj_nodes.push(TreeNode::branch(
                base,
                agg_cols(ss),
                Node::Group,
                session_nodes,
            ));
        }
        out.push(
            TreeNode::branch(agent.clone(), agg_cols(sessions), Node::Group, proj_nodes)
                .bold()
                .expanded(),
        );
    }
    out
}

fn session_cols(s: &SessionRow) -> Vec<String> {
    vec![
        format!("${:.2}", s.bill_usd()),
        format!("${:.2}", s.saved_usd()),
        format!("{:.0}%", s.trim_pct()),
        rel_time(&s.last_ts),
        s.turns.to_string(),
        format!("{:.0}%", s.cache_hit * 100.0),
    ]
}

fn agg_cols(rows: &[&SessionRow]) -> Vec<String> {
    let turns: i64 = rows.iter().map(|r| r.turns).sum();
    let bill: i64 = rows.iter().map(|r| r.bill_micros).sum();
    let saved_m: i64 = rows.iter().map(|r| r.saved_micros).sum();
    let before: i64 = rows.iter().map(|r| r.input_before).sum();
    let after: i64 = rows.iter().map(|r| r.input_after).sum();
    let trim = if before > 0 {
        (before - after).max(0) as f64 / before as f64 * 100.0
    } else {
        0.0
    };
    // Most recent activity in the group, for the `when` column.
    let latest = rows.iter().map(|r| r.last_ts.as_str()).max().unwrap_or("");
    vec![
        format!("${:.2}", bill as f64 / 1_000_000.0),
        format!("${:.2}", saved_m as f64 / 1_000_000.0),
        format!("{trim:.0}%"),
        rel_time(latest),
        turns.to_string(),
        // Cache reuse is only meaningful per session, so it's left blank at the rollup level.
        String::new(),
    ]
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Compact relative age of an rfc3339 timestamp ("4s", "12m", "3h", "5d"); "" if unparseable.
fn rel_time(ts: &str) -> String {
    let Ok(t) = chrono::DateTime::parse_from_rfc3339(ts) else {
        return String::new();
    };
    let secs = (chrono::Utc::now() - t.with_timezone(&chrono::Utc))
        .num_seconds()
        .max(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

/// The built Detail panes (tree roots + footers), computed off the UI thread. Holds only
/// plain owned data (`Send`), so the detail worker can build it from its own SQLite
/// connection and ship it to the UI through a channel.
struct DetailData {
    occupancy_roots: Vec<TreeNode<()>>,
    occupancy_footer: Option<Vec<String>>,
    cost_roots: Vec<TreeNode<()>>,
    cost_footer: Option<Vec<String>>,
}

/// Query the ledger and build a session's two Detail panes. Runs the SQLite work, so it must
/// stay off the UI thread (the detail worker calls it); the UI only installs the result.
fn compute_detail(db: &BreakdownDb, session_id: &str) -> DetailData {
    let mut data = DetailData {
        occupancy_roots: Vec::new(),
        occupancy_footer: None,
        cost_roots: Vec::new(),
        cost_footer: None,
    };
    if let Ok(Some((turn_id, window))) = db.latest_turn(session_id)
        && let Ok(rows) = db.occupancy(turn_id)
    {
        data.occupancy_roots = build_occupancy_tree(&rows, window);
        let total: i64 = rows.iter().map(|r| r.tokens).sum();
        data.occupancy_footer = Some(vec![
            "total".into(),
            pct(total as f64, window as f64),
            bar(total, window, 14),
            crate::ui::human(total),
        ]);
    }
    if let Ok(rows) = db.cost(session_id) {
        // Canonical paid = SUM(bill_micros); block rows allocate that bill (may residual).
        let bill = db
            .session_bill_micros(session_id)
            .map(|m| m as f64 / 1_000_000.0)
            .unwrap_or_else(|_| rows.iter().map(|r| r.usd).sum());
        let read: f64 = rows.iter().map(|r| r.read_usd).sum();
        let write: f64 = rows.iter().map(|r| r.write_usd).sum();
        let new: f64 = rows.iter().map(|r| r.new_usd).sum();
        let blocks = read + write + new;
        data.cost_roots = build_cost_tree(&rows, bill);
        // When block realloc ≠ bill (e.g. empty blocks), keep bill as ground truth and
        // mark the component sum so %bill does not silently over-claim.
        let bill_label = if (blocks - bill).abs() > 0.005 && bill > 0.0 {
            format!("${bill:.2}*")
        } else {
            format!("${bill:.2}")
        };
        data.cost_footer = Some(vec![
            "bill".into(),
            bill_label,
            format!("${read:.2}"),
            format!("${write:.2}"),
            format!("${new:.2}"),
            "—".into(),
        ]);
    }
    data
}

/// Install already-built pane data into a Detail and clear its loading flag. Pure in-memory
/// work — safe on the UI thread.
fn install_detail(d: &mut Detail, data: DetailData) {
    d.occupancy.set_roots(data.occupancy_roots);
    d.occupancy.footer = data.occupancy_footer;
    d.cost.set_roots(data.cost_roots);
    d.cost.footer = data.cost_footer;
    d.loading = false;
}

/// Re-query and rebuild a Detail's two panes from the ledger (synchronous; used by the SVG
/// export test, which has no worker thread).
#[cfg(test)]
fn rebuild_detail(db: &BreakdownDb, d: &mut Detail) {
    let data = compute_detail(db, &d.session_id);
    install_detail(d, data);
}

/// The grouping key shared by both Detail panes: (group, category, MCP server, tool).
type GroupKey = (String, String, Option<String>, Option<String>);

/// Build the occupancy tree (group → category → MCP server → tool) of input tokens.
fn build_occupancy_tree(rows: &[OccupancyRow], window: i64) -> Vec<TreeNode<()>> {
    group_tree(
        rows,
        |r| {
            (
                r.group_label.clone(),
                r.label.clone(),
                r.mcp_server.clone(),
                r.tool_name.clone(),
            )
        },
        |r| r.tokens,
        |a, b| a + b,
        |t| {
            vec![
                pct(*t as f64, window as f64),
                bar(*t, window, 14),
                crate::ui::human(*t),
            ]
        },
    )
}

/// Four cost figures summed up the tree: total, cache-read, cache-write, and "new" (fresh
/// input + output) — the columns the cost pane shows.
#[derive(Default, Clone, Copy)]
struct Money {
    usd: f64,
    read: f64,
    write: f64,
    new: f64,
}

/// Build the cost tree (group → category → MCP server → tool) priced from frozen rates.
fn build_cost_tree(rows: &[CostRow], bill: f64) -> Vec<TreeNode<()>> {
    group_tree(
        rows,
        |r| {
            (
                r.group_label.clone(),
                r.label.clone(),
                r.mcp_server.clone(),
                r.tool_name.clone(),
            )
        },
        |r| Money {
            usd: r.usd,
            read: r.read_usd,
            write: r.write_usd,
            new: r.new_usd,
        },
        |a, b| Money {
            usd: a.usd + b.usd,
            read: a.read + b.read,
            write: a.write + b.write,
            new: a.new + b.new,
        },
        move |m| {
            vec![
                format!("${:.2}", m.usd),
                format!("${:.2}", m.read),
                format!("${:.2}", m.write),
                format!("${:.2}", m.new),
                if bill > 0.0 {
                    format!("{:.0}%", m.usd / bill * 100.0)
                } else {
                    "—".into()
                },
            ]
        },
    )
}

/// Generic four-level builder: rows → Static/Messages/Output → category → MCP server → tool.
/// `value` extracts a per-row accumulator `A`, `add` folds children into their parent, and
/// `cols` formats each node's rolled-up value. Both Detail panes are this function with a
/// different `A` (token count vs the four cost figures). Group order is fixed; category /
/// server / tool order follows the rows' input order (the DB returns them largest-first).
fn group_tree<R, A: Default + Clone>(
    rows: &[R],
    key: impl Fn(&R) -> GroupKey,
    value: impl Fn(&R) -> A,
    add: impl Fn(&A, &A) -> A,
    cols: impl Fn(&A) -> Vec<String>,
) -> Vec<TreeNode<()>> {
    use std::collections::BTreeMap;
    // Decorate each row once with its key and accumulator value.
    let decorated: Vec<(GroupKey, A)> = rows.iter().map(|r| (key(r), value(r))).collect();

    // These group names are the fixed internal vocabulary emitted by
    // `attribution::BlockAttribution::category` (and stored verbatim in the ledger), not
    // user- or locale-derived text, so the match is exhaustive by construction.
    let mut out = Vec::new();
    for g in ["Static", "Messages", "Output"] {
        let in_group: Vec<&(GroupKey, A)> = decorated.iter().filter(|(k, _)| k.0 == g).collect();
        if in_group.is_empty() {
            continue;
        }
        // category → (server-or-direct) → tool, preserving first-seen order at each level.
        let mut cat_order: Vec<String> = Vec::new();
        let mut cats: BTreeMap<String, Vec<&(GroupKey, A)>> = BTreeMap::new();
        for row in &in_group {
            let label = &row.0.1;
            if !cats.contains_key(label) {
                cat_order.push(label.clone());
            }
            cats.entry(label.clone()).or_default().push(row);
        }

        let mut gacc = A::default();
        let mut cat_nodes = Vec::new();
        for cat in &cat_order {
            let crows = &cats[cat];
            let mut srv_order: Vec<String> = Vec::new();
            let mut servers: BTreeMap<String, Vec<&(GroupKey, A)>> = BTreeMap::new();
            let mut direct: Vec<&(GroupKey, A)> = Vec::new();
            let mut cacc = A::default();
            for row in crows {
                cacc = add(&cacc, &row.1);
                match &row.0.2 {
                    Some(s) => {
                        if !servers.contains_key(s) {
                            srv_order.push(s.clone());
                        }
                        servers.entry(s.clone()).or_default().push(row);
                    }
                    None => direct.push(row),
                }
            }
            let mut sub = Vec::new();
            for s in &srv_order {
                let trows = &servers[s];
                let mut sacc = A::default();
                let mut tools = Vec::new();
                for row in trows {
                    sacc = add(&sacc, &row.1);
                    if let Some(t) = &row.0.3 {
                        tools.push(TreeNode::leaf(t.clone(), cols(&row.1), ()));
                    }
                }
                sub.push(TreeNode::branch(s.clone(), cols(&sacc), (), tools));
            }
            // A category that also has named (server/tool) children must show its un-named
            // direct rows too, else the children won't sum to the category total. A category
            // that is *only* un-named direct rows needs no child — its own branch total says it.
            let has_named = !servers.is_empty() || direct.iter().any(|r| r.0.3.is_some());
            for row in &direct {
                let label = match &row.0.3 {
                    Some(t) => Some(t.clone()),
                    None if has_named => Some(cat.clone()),
                    None => None,
                };
                if let Some(label) = label {
                    sub.push(TreeNode::leaf(label, cols(&row.1), ()));
                }
            }
            gacc = add(&gacc, &cacc);
            cat_nodes.push(TreeNode::branch(cat.clone(), cols(&cacc), (), sub));
        }
        out.push(
            TreeNode::branch(g.to_string(), cols(&gacc), (), cat_nodes)
                .bold()
                .expanded(),
        );
    }
    out
}

/// A proportional block bar `████░░░░` of the given width. Uses the block-element shade
/// `░` for the unfilled portion (matching the Overview gauges and keeping a consistent
/// visual weight), both glyphs being reliably single-width.
fn bar(value: i64, total: i64, width: usize) -> String {
    let frac = if total > 0 {
        (value as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = (frac * width as f64).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn pct(value: f64, total: f64) -> String {
    if total > 0.0 {
        format!("{:.0}%", value / total * 100.0)
    } else {
        "—".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a draw closure to a headless TestBackend and flatten the buffer to a string
    /// (no newlines) so we can assert on-screen content — the closest thing to a live check.
    fn render_to_string(w: u16, h: u16, draw: impl FnOnce(&mut Frame)) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(draw).unwrap();
        term.backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    fn sample_overview() -> OverviewData {
        OverviewData {
            status: StatusLine {
                kind: StatusKind::Working,
                text: "llmtrim is on and working · last request 4s ago".into(),
                fix: None,
                uninstall: None,
            },
            paid_usd: Some(84.24),
            would_have_usd: Some(226.82),
            saved_usd: Some(142.58),
            saved_today_usd: Some(4.10),
            pct_less: 0.33,
            pct_less_whole: 0.12,
            added_ms: Some(32.0),
            requests: 1204,
            trend_daily_usd: vec![7.0, 9.0, 11.0, 10.0, 13.0, 14.0, 16.0],
            savers: vec![("gpt-4o".into(), 84.24), ("claude-sonnet".into(), 41.10)],
            input_before: 2_100_000,
            input_after: 1_240_000,
            output_billed: 229_000,
            output_est_pct: 73.0,
            has_traffic: true,
            approximate: false,
            sessions: 42,
            update_available: None,
            coverage_compressions: 1204,
            coverage_turns: 1204,
            turns_unpriced: 0,
            money_unavailable: false,
        }
    }

    #[test]
    fn money_groups_thousands() {
        assert_eq!(money(1284.5), "$1,284.50");
        assert_eq!(money(7.1), "$7.10");
        assert_eq!(money_round(1284.0), "$1,284");
        assert_eq!(money_round(1284.6), "$1,285");
    }

    #[test]
    fn overview_cards_show_kpis_charts_and_metrics() {
        let ov = sample_overview();
        let s = render_to_string(140, 30, |f| render_overview_main(f, f.area(), &ov, false));
        // Band C — KPI tiles (values are plain text now, no block-glyph art).
        assert!(s.contains("TOTAL SAVED"), "{s}");
        assert!(s.contains("WOULD HAVE COST") && s.contains("226.82"));
        assert!(s.contains("YOU PAID") && s.contains("84.24"));
        // Band D — charts row.
        assert!(s.contains("SAVINGS TREND"));
        assert!(s.contains("SMALLER REQUESTS"));
        assert!(s.contains("TOP MODELS") && s.contains("gpt-4o"));
        // Band E — the honest operational metric tiles.
        assert!(s.contains("REQUESTS HANDLED") && s.contains("ADDED DELAY"));
        assert!(s.contains("SESSIONS COVERED"));
    }

    #[test]
    fn size_gauge_toggles_basis_but_dollars_never_change() {
        let ov = sample_overview();
        // Default (new content): the compressible-surface 33%, dollars are the net bill.
        let new = render_to_string(140, 30, |f| render_overview_main(f, f.area(), &ov, false));
        assert!(new.contains("33% smaller"), "{new}");
        assert!(new.contains("of new content"), "{new}");
        // Whole prompt: the diluted 12%. The dollar KPIs are byte-for-byte the same — only the
        // percentage moved, so the toggle can never overstate the saving in money terms.
        let whole = render_to_string(140, 30, |f| render_overview_main(f, f.area(), &ov, true));
        assert!(whole.contains("12% smaller"), "{whole}");
        assert!(whole.contains("of the whole prompt"), "{whole}");
        for dollars in ["226.82", "84.24", "142.58"] {
            assert_eq!(
                new.contains(dollars),
                whole.contains(dollars),
                "dollar figure {dollars} must not depend on the size-basis toggle"
            );
        }
    }

    #[test]
    fn overview_keeps_jargon_off_the_default_screen() {
        let ov = sample_overview();
        let s = render_to_string(140, 30, |f| render_overview_main(f, f.area(), &ov, false));
        assert!(s.contains("REQUESTS HANDLED"));
        // The old expert "more numbers" strip (and its jargon) is gone entirely.
        assert!(!s.contains("net of cache"));
    }

    #[test]
    fn overview_off_is_an_alarm_not_a_celebration() {
        let mut ov = sample_overview();
        ov.status = StatusLine {
            kind: StatusKind::Off,
            text: "llmtrim is OFF — your AI tools can't reach the API right now".into(),
            fix: Some("llmtrim start".into()),
            uninstall: Some("llmtrim uninstall".into()),
        };
        let s = render_to_string(140, 20, |f| render_overview_alert(f, f.area(), &ov));
        assert!(s.contains("OFF"));
        assert!(s.contains("llmtrim start"));
        // savings demoted to a frozen aside, not the hero
        assert!(s.contains("Saved while it was on"));
    }

    #[test]
    fn stale_daemon_shows_update_not_repair() {
        let mut ov = sample_overview();
        ov.status = StatusLine {
            kind: StatusKind::Stale,
            text: "llmtrim is on, but running an older version — restart to apply the update"
                .into(),
            fix: Some("llmtrim start --force".into()),
            uninstall: None,
        };
        ov.update_available = None;
        // A stale daemon is not broken: it renders the normal dashboard with a `u Update` nudge,
        // never the Repair alert.
        let s = render_to_string(140, 30, |f| render_overview_main(f, f.area(), &ov, false));
        assert!(s.contains("older version"), "{s}");
        assert!(s.contains("Update"), "u Update affordance shown: {s}");
        assert!(
            !s.contains("Repair"),
            "no Repair on a stale (not broken) daemon: {s}"
        );
    }

    #[test]
    fn c_key_flips_the_size_basis_on_overview_only() {
        let mut app = App::new(None, Duration::from_secs(2));
        app.overview = Some(sample_overview());

        // On Overview, `c` toggles the gauge basis.
        assert!(!app.whole_prompt);
        app.handle_key(KeyCode::Char('c'));
        assert!(app.whole_prompt);
        app.handle_key(KeyCode::Char('c'));
        assert!(!app.whole_prompt);

        // On the other tabs there is no size figure to reframe, so `c` is a no-op.
        for tab in [Tab::Sessions, Tab::Detail] {
            app.tab = tab;
            app.handle_key(KeyCode::Char('c'));
            assert!(
                !app.whole_prompt,
                "c must not toggle the basis off Overview"
            );
        }
    }

    #[test]
    fn y_key_opens_the_tray_without_quitting_when_installed() {
        // Not installed: `y` is inert — no quit, no action.
        let mut app = App::new(None, Duration::from_secs(2));
        app.tray_available = false;
        assert!(!app.handle_key(KeyCode::Char('y')));
        assert_eq!(app.action, PostAction::None);

        // Installed: `y` launches the tray in the background and stays on the dashboard
        // (returns false = no quit) without queuing a post-teardown action. The launch is
        // best-effort — no sibling binary exists in tests, so it no-ops.
        app.tray_available = true;
        assert!(!app.handle_key(KeyCode::Char('y')));
        assert_eq!(app.action, PostAction::None);
    }

    #[test]
    fn u_key_restarts_a_stale_daemon_else_updates() {
        let mut app = App::new(None, Duration::from_secs(2));
        let mut ov = sample_overview();
        ov.status.kind = StatusKind::Stale;
        app.overview = Some(ov);
        app.handle_key(KeyCode::Char('u'));
        assert_eq!(app.action, PostAction::Restart);

        let mut app = App::new(None, Duration::from_secs(2));
        let mut ov = sample_overview();
        ov.status.kind = StatusKind::Working;
        app.overview = Some(ov);
        app.handle_key(KeyCode::Char('u'));
        assert_eq!(app.action, PostAction::Update);
    }

    #[test]
    fn u_key_downloads_when_stale_and_a_release_is_available() {
        // Both: daemon is stale AND a newer release exists. Download wins (restarting onto an
        // already-outdated binary would be pointless); the next `u` then restarts.
        let mut app = App::new(None, Duration::from_secs(2));
        let mut ov = sample_overview();
        ov.status.kind = StatusKind::Stale;
        ov.update_available = Some("0.3.0".into());
        app.overview = Some(ov);
        app.handle_key(KeyCode::Char('u'));
        assert_eq!(app.action, PostAction::Update);
    }

    #[test]
    fn overview_empty_reassures_without_fake_zero() {
        let ov = OverviewData {
            status: StatusLine {
                kind: StatusKind::Ready,
                text: "llmtrim is on and ready · waiting for your first request".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let s = render_to_string(80, 20, |f| render_overview_empty(f, f.area(), &ov));
        assert!(s.contains("Nothing saved yet"));
        assert!(!s.contains("$0.00"));
    }

    fn srow(agent: &str, project: Option<&str>, id: &str, turns: i64, bill: i64) -> SessionRow {
        SessionRow {
            session_id: id.to_string(),
            cc_session_id: None,
            agent: agent.to_string(),
            project: project.map(str::to_string),
            session_name: None,
            turns,
            tokens: 1000,
            cache_hit: 0.5,
            bill_micros: bill,
            saved_micros: bill / 4,
            input_before: 1000,
            input_after: 600,
            last_ts: "2026-06-19T00:00:00+00:00".to_string(),
            last_sub_provider: None,
            last_model: None,
            last_cache_hit: None,
            last_input_tokens: 0,
        }
    }

    #[test]
    fn session_tree_groups_agent_project_session() {
        let rows = vec![
            srow("claude-code", Some("/a/proj1"), "s1", 3, 1000),
            srow("claude-code", Some("/a/proj1"), "s2", 2, 500),
            srow("codex", None, "s3", 1, 200),
        ];
        let tree = build_session_tree(&rows);
        assert_eq!(tree.len(), 2); // two agents
        let claude = tree
            .iter()
            .find(|n| n.label.starts_with("claude-code"))
            .unwrap();
        // one project under claude-code, two sessions under it
        assert_eq!(claude.children.len(), 1);
        assert_eq!(claude.children[0].children.len(), 2);
        // columns: [cost, saved$, trim, when, msgs, reuse]; rolled-up turns = 5 (msgs, index 4).
        assert_eq!(claude.cols[4], "5");
    }

    #[test]
    fn cost_tree_nests_mcp_server_under_category() {
        let rows = vec![
            CostRow {
                group_label: "Static".into(),
                label: "MCP tools".into(),
                mcp_server: Some("github".into()),
                tool_name: Some("create_issue".into()),
                usd: 0.10,
                read_usd: 0.02,
                write_usd: 0.0,
                new_usd: 0.08,
            },
            CostRow {
                group_label: "Static".into(),
                label: "System prompt".into(),
                mcp_server: None,
                tool_name: None,
                usd: 0.40,
                read_usd: 0.40,
                write_usd: 0.0,
                new_usd: 0.0,
            },
        ];
        let tree = build_cost_tree(&rows, 0.50);
        let static_g = tree.iter().find(|n| n.label == "Static").unwrap();
        let mcp = static_g
            .children
            .iter()
            .find(|n| n.label == "MCP tools")
            .unwrap();
        // server level then tool leaf
        assert_eq!(mcp.children.len(), 1);
        assert_eq!(mcp.children[0].label, "github");
        assert_eq!(mcp.children[0].children[0].label, "create_issue");
    }

    #[test]
    fn bar_is_proportional() {
        assert_eq!(bar(50, 100, 10), "█████░░░░░");
        assert_eq!(bar(0, 100, 4), "░░░░");
        assert_eq!(bar(200, 100, 4), "████"); // clamps
    }

    // ── README hero export: render the REAL TUI to SVG (the only way the asset matches) ──
    // Run with: LLMTRIM_EXPORT_SVG=1 cargo test -p llmtrim --features intercept,mcp,breakdown
    //           export_status_watch_svgs -- --ignored --nocapture

    // CW must match the monospace advance at FS (≈0.6·FS) so glyphs render at natural size and
    // block-element bars tile seamlessly — too small and lengthAdjust squeezes every glyph.
    const CW: u32 = 9; // cell width  (px)
    const CH: u32 = 18; // cell height (px)
    const FS: u32 = 15; // font size  (px)
    const PAD: u32 = 14;
    const TITLE: u32 = 30;

    fn hexp(c: ratatui::style::Color) -> Option<String> {
        match c {
            ratatui::style::Color::Rgb(r, g, b) => Some(format!("#{r:02x}{g:02x}{b:02x}")),
            _ => None,
        }
    }

    fn xesc(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    /// (fg, bg) for a cell, applying reverse-video.
    fn swapped(c: &ratatui::buffer::Cell) -> (ratatui::style::Color, ratatui::style::Color) {
        if c.modifier.contains(Modifier::REVERSED) {
            (c.bg, c.fg)
        } else {
            (c.fg, c.bg)
        }
    }

    thread_local! {
        // Cell rect (x,y,w,h) where render_trend drew its bar chart, captured during export so
        // the SVG can replace the block-bars with a real vector chart.
        static TREND_RECT: std::cell::Cell<Option<(u16, u16, u16, u16)>> =
            const { std::cell::Cell::new(None) };
    }
    pub(super) fn capture_trend_rect(r: Rect) {
        TREND_RECT.with(|c| c.set(Some((r.x, r.y, r.width, r.height))));
    }

    thread_local! {
        // Cell rect of render_gauge's 1-row Gauge, captured so the SVG can draw a rounded bar.
        static GAUGE_RECT: std::cell::Cell<Option<(u16, u16, u16, u16)>> =
            const { std::cell::Cell::new(None) };
    }
    pub(super) fn capture_gauge_rect(r: Rect) {
        GAUGE_RECT.with(|c| c.set(Some((r.x, r.y, r.width, r.height))));
    }

    /// A horizontal bar segment as a path, with left and/or right ends rounded (radius r). Lets
    /// the gauge fill and its track abut with a clean straight seam (no overlapping rounded caps).
    fn pill(x: u32, y: u32, w: u32, h: u32, r: u32, lr: bool, rr: bool) -> String {
        let r = r.min(h / 2).min(w);
        let (xl, xr, yt, yb) = (x, x + w, y, y + h);
        let mut d = format!("M{} {yt} ", xl + if lr { r } else { 0 });
        d += &format!("H{} ", xr - if rr { r } else { 0 });
        if rr {
            d += &format!(
                "A{r} {r} 0 0 1 {xr} {} V{} A{r} {r} 0 0 1 {} {yb} ",
                yt + r,
                yb - r,
                xr - r
            );
        } else {
            d += &format!("V{yb} ");
        }
        d += &format!("H{} ", xl + if lr { r } else { 0 });
        if lr {
            d += &format!(
                "A{r} {r} 0 0 1 {xl} {} V{} A{r} {r} 0 0 1 {} {yt} ",
                yb - r,
                yt + r,
                xl + r
            );
        } else {
            d += &format!("V{yt} ");
        }
        d + "Z"
    }

    /// A rounded vector progress bar for the "smaller requests" gauge: green fill and grey track
    /// meet with a clean straight edge (fill rounds only its left, track only its right).
    fn gauge_svg(
        rect: (u16, u16, u16, u16),
        pct: f64,
        gx0: u32,
        gy0: u32,
        fill: &str,
        track: &str,
    ) -> String {
        let (cx, cy, cw, ch) = rect;
        let x = gx0 + cx as u32 * CW;
        let w = cw as u32 * CW;
        let h = ch as u32 * CH;
        // Full cell-height bar, like every other bar on the dashboard.
        let bar_h = h;
        let by = gy0 + cy as u32 * CH;
        let r = 6u32;
        let fw = ((w as f64) * pct.clamp(0.0, 1.0)).round() as u32;
        let mut out = String::new();
        if fw < w {
            // unfilled remainder: square where it meets the fill, rounded at the far right
            out.push_str(&format!(
                "<path d=\"{}\" fill=\"{track}\"/>",
                pill(x + fw, by, w - fw, bar_h, r, fw == 0, true)
            ));
        }
        if fw > 0 {
            out.push_str(&format!(
                "<path d=\"{}\" fill=\"{fill}\"/>",
                pill(x, by, fw, bar_h, r, true, fw >= w)
            ));
        }
        out
    }

    /// A clean vector bar chart for the savings trend (value labels above, weekday labels below),
    /// drawn into the captured card-inner cell rect.
    fn trend_svg(
        vals: &[f64],
        rect: (u16, u16, u16, u16),
        gx0: u32,
        gy0: u32,
        green: &str,
        muted: &str,
    ) -> String {
        let (cx, cy, cw, ch) = rect;
        let x = gx0 + cx as u32 * CW;
        let y = gy0 + cy as u32 * CH;
        let w = cw as u32 * CW;
        let h = ch as u32 * CH;
        let n = vals.len().max(1) as u32;
        let maxv = vals.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let pad_top = FS + 6; // room for the value label
        let day_h = FS + 4; // room for the weekday label
        let base_y = y + h - day_h;
        let bar_top = y + pad_top;
        let bar_max = base_y.saturating_sub(bar_top).max(1);
        let col = (w / n).max(1);
        let bw = ((col as f64) * 0.6).round().max(2.0) as u32;
        let names = weekday_labels(vals.len());
        let last = vals.len().saturating_sub(1);
        let mut out = String::new();
        for (i, v) in vals.iter().enumerate() {
            let bh = ((*v / maxv) * bar_max as f64).round().max(1.0) as u32;
            let colx = x + i as u32 * col;
            let bx = colx + (col - bw) / 2;
            let by = base_y - bh;
            let op = if i == last { "1" } else { "0.78" };
            out.push_str(&format!(
                "<rect x=\"{bx}\" y=\"{by}\" width=\"{bw}\" height=\"{bh}\" rx=\"2\" fill=\"{green}\" opacity=\"{op}\"/>"
            ));
            let bold = if i == last {
                " font-weight=\"bold\""
            } else {
                ""
            };
            out.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" fill=\"{muted}\" font-size=\"{}\" text-anchor=\"middle\"{bold}>{}</text>",
                bx + bw / 2,
                by - 4,
                FS - 3,
                xesc(&money_round(*v))
            ));
            out.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" fill=\"{muted}\" font-size=\"{}\" text-anchor=\"middle\">{}</text>",
                colx + col / 2,
                base_y + FS,
                FS - 2,
                names.get(i).copied().unwrap_or_default()
            ));
        }
        out
    }

    /// What a cell's glyph is, for SVG output: block elements become rects (so bars are crisp),
    /// everything else is text.
    #[derive(Clone, Copy)]
    enum Glyph {
        Space,
        Solid,      // █  — full cell rect
        Shade(f32), // ░▒▓ — full cell rect at opacity
        VFrac(f32), // ▁▂▃▄▅▆▇ — bottom-anchored partial-height rect
        HFrac(f32), // ▏▎▍▌▋▊▉ — left-anchored partial-width rect
        Text,
    }
    fn classify(s: &str) -> Glyph {
        match s {
            " " | "" => Glyph::Space,
            "█" => Glyph::Solid,
            "░" => Glyph::Shade(0.30),
            "▒" => Glyph::Shade(0.50),
            "▓" => Glyph::Shade(0.75),
            "▁" => Glyph::VFrac(0.125),
            "▂" => Glyph::VFrac(0.25),
            "▃" => Glyph::VFrac(0.375),
            "▄" => Glyph::VFrac(0.5),
            "▅" => Glyph::VFrac(0.625),
            "▆" => Glyph::VFrac(0.75),
            "▇" => Glyph::VFrac(0.875),
            "▏" => Glyph::HFrac(0.125),
            "▎" => Glyph::HFrac(0.25),
            "▍" => Glyph::HFrac(0.375),
            "▌" => Glyph::HFrac(0.5),
            "▋" => Glyph::HFrac(0.625),
            "▊" => Glyph::HFrac(0.75),
            "▉" => Glyph::HFrac(0.875),
            _ => Glyph::Text,
        }
    }

    /// Box-drawing glyphs (borders) rendered as crisp line rects instead of text — text glyphs
    /// don't tile into continuous lines. Returns `None` for non-box symbols. Rounded corners are
    /// approximated by the two half-segments meeting (the radius is sub-pixel at this scale).
    fn box_rects(sym: &str, cx: u32, cy: u32, fg: &str) -> Option<String> {
        let t = 2u32;
        let vx = cx + CW / 2; // vertical-line center x
        let vy = cy + CH / 2; // horizontal-line center y
        let h = |x0: u32, x1: u32| {
            format!(
                "<rect x=\"{x0}\" y=\"{}\" width=\"{}\" height=\"{t}\" fill=\"{fg}\"/>",
                vy - t / 2,
                x1 - x0
            )
        };
        let v = |y0: u32, y1: u32| {
            format!(
                "<rect x=\"{}\" y=\"{y0}\" width=\"{t}\" height=\"{}\" fill=\"{fg}\"/>",
                vx - t / 2,
                y1 - y0
            )
        };
        let r = match sym {
            "─" => h(cx, cx + CW),
            "│" => v(cy, cy + CH),
            "╭" => format!("{}{}", h(vx, cx + CW), v(vy, cy + CH)),
            "╮" => format!("{}{}", h(cx, vx + t), v(vy, cy + CH)),
            "╰" => format!("{}{}", h(vx, cx + CW), v(cy, vy + t)),
            "╯" => format!("{}{}", h(cx, vx + t), v(cy, vy + t)),
            "├" => format!("{}{}", v(cy, cy + CH), h(vx, cx + CW)),
            "┤" => format!("{}{}", v(cy, cy + CH), h(cx, vx + t)),
            "┬" => format!("{}{}", h(cx, cx + CW), v(vy, cy + CH)),
            "┴" => format!("{}{}", h(cx, cx + CW), v(cy, vy + t)),
            "┼" => format!("{}{}", h(cx, cx + CW), v(cy, cy + CH)),
            _ => return None,
        };
        Some(r)
    }

    /// Serialize one rendered buffer to SVG, faithfully. Block-element glyphs become `<rect>`s
    /// (crisp bars/gauges), text becomes one `<text>` per run with an explicit per-glyph x-list
    /// (each char pinned to its column) and NO `textLength` — so glyphs render at natural size,
    /// never stretched. `base` is the screen background, already drawn, so we skip it.
    fn frame_svg(buf: &ratatui::buffer::Buffer, x0: u32, y0: u32, base: &str) -> String {
        let text_default = hexp(palette::text()).unwrap();
        let w = buf.area.width as u32;
        let h = buf.area.height as u32;
        let cells = buf.content();
        let mut out = String::new();
        for y in 0..h {
            let ry = y0 + y * CH;
            // ── background run rects (coalesced) ──
            let mut col = 0u32;
            while col < w {
                let bg0 = swapped(&cells[(y * w + col) as usize]).1;
                let start = col;
                while col < w && swapped(&cells[(y * w + col) as usize]).1 == bg0 {
                    col += 1;
                }
                if let Some(bh) = hexp(bg0)
                    && bh != base
                {
                    out.push_str(&format!(
                        "<rect x=\"{}\" y=\"{ry}\" width=\"{}\" height=\"{CH}\" fill=\"{bh}\"/>",
                        x0 + start * CW,
                        (col - start) * CW
                    ));
                }
            }
            // ── glyphs: blocks → rects, text → one run with per-glyph x-list ──
            let mut tx: Vec<u32> = Vec::new();
            let mut trun = String::new();
            let mut tstyle: Option<(ratatui::style::Color, Modifier)> = None;
            let flush = |tx: &mut Vec<u32>,
                         trun: &mut String,
                         st: Option<(ratatui::style::Color, Modifier)>,
                         out: &mut String| {
                if trun.is_empty() {
                    return;
                }
                let (fg, m) = st.unwrap();
                let fh = hexp(fg).unwrap_or_else(|| text_default.clone());
                let mut attrs = String::new();
                if m.contains(Modifier::BOLD) {
                    attrs.push_str(" font-weight=\"bold\"");
                }
                if m.contains(Modifier::DIM) {
                    attrs.push_str(" opacity=\"0.6\"");
                }
                if m.contains(Modifier::CROSSED_OUT) {
                    attrs.push_str(" text-decoration=\"line-through\"");
                }
                let xs = tx
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push_str(&format!(
                    "<text x=\"{xs}\" y=\"{}\" fill=\"{fh}\"{attrs}>{}</text>",
                    ry + FS - 3,
                    xesc(trun)
                ));
                tx.clear();
                trun.clear();
            };
            let mut x = 0u32;
            while x < w {
                let cell = &cells[(y * w + x) as usize];
                let (fg, _) = swapped(cell);
                let cx = x0 + x * CW;
                let fh = hexp(fg).unwrap_or_else(|| text_default.clone());
                // Box-drawing borders → crisp line rects (before the text path).
                if let Some(rects) = box_rects(cell.symbol(), cx, ry, &fh) {
                    flush(&mut tx, &mut trun, tstyle, &mut out);
                    tstyle = None;
                    out.push_str(&rects);
                    x += 1;
                    continue;
                }
                let g = classify(cell.symbol());
                match g {
                    Glyph::Space => {
                        flush(&mut tx, &mut trun, tstyle, &mut out);
                        tstyle = None;
                        x += 1;
                    }
                    Glyph::Text => {
                        let st = (fg, cell.modifier);
                        if tstyle != Some(st) {
                            flush(&mut tx, &mut trun, tstyle, &mut out);
                            tstyle = Some(st);
                        }
                        tx.push(cx);
                        trun.push_str(cell.symbol());
                        x += 1;
                    }
                    // Full / shade runs → ONE coalesced rect (no per-cell seams), rounded for the
                    // filled bars. Same fg + modifier + category extends the run.
                    Glyph::Solid | Glyph::Shade(_) => {
                        flush(&mut tx, &mut trun, tstyle, &mut out);
                        tstyle = None;
                        let m = cell.modifier;
                        let start = x;
                        x += 1;
                        while x < w {
                            let c2 = &cells[(y * w + x) as usize];
                            let (fg2, _) = swapped(c2);
                            let same = fg2 == fg
                                && c2.modifier == m
                                && matches!(
                                    (classify(c2.symbol()), g),
                                    (Glyph::Solid, Glyph::Solid)
                                        | (Glyph::Shade(_), Glyph::Shade(_))
                                );
                            if !same {
                                break;
                            }
                            x += 1;
                        }
                        let bx = x0 + start * CW;
                        let bw = (x - start) * CW;
                        let (op, rx) = match g {
                            // Track/shade: keep square; filled bars: round the ends.
                            Glyph::Shade(o) => (o, 0),
                            _ => (1.0, (CH / 2).min(bw / 2).min(4)),
                        };
                        let opacity = if op < 1.0 {
                            format!(" opacity=\"{op}\"")
                        } else {
                            String::new()
                        };
                        let radius = if rx > 0 {
                            format!(" rx=\"{rx}\"")
                        } else {
                            String::new()
                        };
                        out.push_str(&format!(
                            "<rect x=\"{bx}\" y=\"{ry}\" width=\"{bw}\" height=\"{CH}\" fill=\"{fh}\"{opacity}{radius}/>"
                        ));
                    }
                    Glyph::VFrac(fr) => {
                        flush(&mut tx, &mut trun, tstyle, &mut out);
                        tstyle = None;
                        let fh_px = (fr * CH as f32).round() as u32;
                        out.push_str(&format!(
                            "<rect x=\"{cx}\" y=\"{}\" width=\"{CW}\" height=\"{fh_px}\" fill=\"{fh}\"/>",
                            ry + CH - fh_px
                        ));
                        x += 1;
                    }
                    Glyph::HFrac(fr) => {
                        flush(&mut tx, &mut trun, tstyle, &mut out);
                        tstyle = None;
                        let fw_px = (fr * CW as f32).round() as u32;
                        out.push_str(&format!(
                            "<rect x=\"{cx}\" y=\"{ry}\" width=\"{fw_px}\" height=\"{CH}\" fill=\"{fh}\"/>"
                        ));
                        x += 1;
                    }
                }
            }
            flush(&mut tx, &mut trun, tstyle, &mut out);
        }
        out
    }

    /// Seed an in-memory ledger with a couple of sessions + source blocks so every tab has
    /// representative content.
    fn seed_export_db() -> BreakdownDb {
        use crate::tracking::{BreakdownBlock, BreakdownTurn, Tracker};
        let t = Tracker::open_in_memory().unwrap();
        let turn = |sid: &str, agent: &str, proj: &str, name: &str| BreakdownTurn {
            session_id: sid.into(),
            cc_session_id: None,
            agent: agent.into(),
            project: Some(proj.into()),
            session_name: Some(name.into()),
            provider: "anthropic".into(),
            model: Some("claude-opus-4-8".into()),
            sub_provider: None,
            window: 200_000,
            fresh_input: 6_000,
            cache_read: 22_000,
            cache_write: 1_200,
            output_tok: 1_500,
            input_rate: 15.0,
            output_rate: 75.0,
            cache_read_rate: 1.5,
            cache_write_rate: 18.75,
            bill_micros: 1_400_000,
            input_before: 31_000,
            input_after: 11_200,
        };
        let blk = |g: &str,
                   l: &str,
                   mcp: Option<&str>,
                   tool: Option<&str>,
                   raw: i64,
                   fresh: f64,
                   read: f64| {
            BreakdownBlock {
                zone: "input".into(),
                section: "static".into(),
                bucket: "x".into(),
                group_label: g.into(),
                label: l.into(),
                mcp_server: mcp.map(str::to_string),
                tool_name: tool.map(str::to_string),
                role: None,
                msg_index: None,
                raw_tokens: raw,
                fresh_tok: fresh,
                cache_read_tok: read,
                cache_write_tok: 0.0,
                output_tok: 0.0,
            }
        };
        let detail_blocks = [
            blk("Static", "System prompt", None, None, 1_820, 520.0, 1_300.0),
            blk(
                "Static",
                "Tool schemas",
                None,
                None,
                9_640,
                3_000.0,
                6_640.0,
            ),
            blk(
                "Messages",
                "Tool results",
                None,
                None,
                25_310,
                9_000.0,
                16_310.0,
            ),
            blk(
                "Messages",
                "MCP tools",
                Some("chrome-devtools"),
                Some("take_snapshot"),
                4_790,
                1_700.0,
                3_090.0,
            ),
        ];
        t.record_breakdown(
            &turn("s1", "claude-code", "/home/me/my-app", "refactor proxy"),
            &detail_blocks,
        )
        .unwrap();
        t.record_breakdown(
            &turn("s2", "claude-code", "/home/me/my-app", "add cache layer"),
            &[blk(
                "Static",
                "System prompt",
                None,
                None,
                1_800,
                500.0,
                1_300.0,
            )],
        )
        .unwrap();
        t.record_breakdown(
            &turn("s3", "codex-cli", "/home/me/infra", "ci pipeline"),
            &[blk(
                "Static",
                "System prompt",
                None,
                None,
                1_700,
                400.0,
                1_300.0,
            )],
        )
        .unwrap();
        BreakdownDb::from_connection(t.into_connection())
    }

    /// Build a real `OverviewData` from this machine's ledger via the shared
    /// `monitor::overview_data` (no duplicated derivation) with a forced healthy status, so the
    /// asset shows genuine magnitudes. `None` when the ledger has no traffic (caller falls back).
    fn real_overview() -> Option<OverviewData> {
        let tr = crate::tracking::Tracker::open_reader().ok()?;
        let ov = crate::monitor::overview_data(&tr, |_summary, _has_traffic| StatusLine {
            // Force the healthy "on and working" status — the asset always shows the dashboard,
            // never the alert screen, regardless of whether the daemon is up right now.
            kind: StatusKind::Working,
            text: "llmtrim is on and working · last request 4s ago".into(),
            fix: None,
            uninstall: None,
        });
        if !ov.has_traffic {
            return None;
        }
        // Drop the update chip for the asset (the live updater check is irrelevant to a static SVG).
        Some(OverviewData {
            update_available: None,
            ..ov
        })
    }

    /// Scrub identifying bits from real session rows for the asset: project paths and session
    /// names become generic labels (`session_id` left for drilling but not shown). Agent names
    /// are tool identifiers, not personal, so they stay.
    ///
    /// DELIBERATE: financial magnitudes (dollars, token counts, request/session totals) are NOT
    /// scrubbed — showing real, defensible numbers is the whole point of the asset (the dashboard
    /// "vitrine"). Whoever regenerates this is committing their own ledger's magnitudes into a
    /// versioned, possibly-public SVG; that is the intended trade, not an oversight. If that is
    /// ever not acceptable, scale or round the figures here.
    fn sanitize_sessions(rows: &mut [SessionRow]) {
        use std::collections::HashMap;
        let mut projects: HashMap<String, String> = HashMap::new();
        for (i, r) in rows.iter_mut().enumerate() {
            if let Some(p) = &r.project {
                let n = projects.len();
                let label = projects
                    .entry(p.clone())
                    .or_insert_with(|| format!("project-{}", (b'a' + n as u8) as char))
                    .clone();
                r.project = Some(label);
            }
            r.session_name = Some(format!("session {}", i + 1));
        }
    }

    #[test]
    #[ignore = "asset generator; run explicitly with LLMTRIM_EXPORT_SVG=1"]
    fn export_status_watch_svgs() {
        if std::env::var("LLMTRIM_EXPORT_SVG").is_err() {
            return;
        }
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        const W: u16 = 116;
        // Just tall enough for the full Overview (banner + KPIs + charts + metrics + chrome) so
        // the frame isn't padded with empty rows.
        const H: u16 = 29;

        // Prefer this machine's real ledger (sanitized); fall back to the synthetic seed when
        // there's no traffic so the asset still renders.
        let (db, ov, rows) = match real_overview() {
            Some(ov) => {
                let db = BreakdownDb::open().expect("open ledger");
                let mut rows = db.sessions().unwrap_or_default();
                sanitize_sessions(&mut rows);
                (db, ov, rows)
            }
            None => {
                let db = seed_export_db();
                let rows = db.sessions().unwrap_or_default();
                (db, sample_overview(), rows)
            }
        };
        // Detail drills the most recent session: real session_id (for the query), sanitized title.
        let (detail_sid, detail_title) = rows
            .first()
            .map(|r| {
                (
                    r.session_id.clone(),
                    r.session_name.clone().unwrap_or_else(|| "session".into()),
                )
            })
            .unwrap_or_else(|| ("s1".into(), "session".into()));

        let mut app = App::new(Some(db), Duration::from_secs(12));
        app.apply(ov, rows);
        let mut d = Detail {
            session_id: detail_sid,
            title: detail_title,
            occupancy: TreeTable::new("context · this turn", occupancy_columns(), palette::frame()),
            cost: TreeTable::new("cost · cumulative", cost_columns(), palette::frame()),
            focus: Pane::Occupancy,
            // The export builds Detail synchronously below, so it's never in the loading state.
            loading: false,
        };
        if let Some(db) = &app.db {
            rebuild_detail(db, &mut d);
        }
        app.detail = Some(d);

        let gx0 = PAD;
        let gy0 = PAD + TITLE;
        let cw_px = W as u32 * CW;
        let ch_px = H as u32 * CH;
        let canvas_w = cw_px + 2 * PAD;
        let canvas_h = ch_px + TITLE + 2 * PAD;

        // Opacity keyframes for a 3-tab crossfade loop.
        let anim = [
            ("1;1;0;0;0;0;1", "0;0.30;0.34;0.64;0.68;0.97;1"),
            ("0;0;1;1;0;0", "0;0.30;0.34;0.64;0.68;1"),
            ("0;0;1;1;0", "0;0.64;0.68;0.97;1"),
        ];
        let tabs = [Tab::Overview, Tab::Sessions, Tab::Detail];

        for (flavor, file) in [
            (palette::Flavor::Mocha, "status-watch-dark.svg"),
            (palette::Flavor::Latte, "status-watch-light.svg"),
        ] {
            palette::set(flavor);
            let base = hexp(palette::bg()).unwrap();
            let frame_col = hexp(palette::frame()).unwrap();
            let muted = hexp(palette::muted_gray()).unwrap();

            let mut svg = format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{canvas_w}\" height=\"{canvas_h}\" \
                 viewBox=\"0 0 {canvas_w} {canvas_h}\" \
                 font-family=\"ui-monospace,SFMono-Regular,Menlo,Consolas,monospace\" font-size=\"{FS}\">"
            );
            // window chrome
            svg.push_str(&format!(
                "<rect width=\"{canvas_w}\" height=\"{canvas_h}\" rx=\"12\" fill=\"{base}\" stroke=\"{frame_col}\"/>"
            ));
            svg.push_str(&format!("<circle cx=\"{}\" cy=\"15\" r=\"6\" fill=\"#f38ba8\"/><circle cx=\"{}\" cy=\"15\" r=\"6\" fill=\"#fab387\"/><circle cx=\"{}\" cy=\"15\" r=\"6\" fill=\"#a6e3a1\"/>", PAD + 8, PAD + 28, PAD + 48));
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"19\" fill=\"{muted}\" text-anchor=\"middle\" font-size=\"12\">llmtrim status</text>",
                canvas_w / 2
            ));
            // screen background
            svg.push_str(&format!(
                "<rect x=\"{gx0}\" y=\"{gy0}\" width=\"{cw_px}\" height=\"{ch_px}\" fill=\"{base}\"/>"
            ));

            let green = hexp(palette::green()).unwrap();
            let trend = app
                .overview
                .as_ref()
                .map(|o| o.trend_daily_usd.clone())
                .unwrap_or_default();
            let pct = app.overview.as_ref().map(|o| o.pct_less).unwrap_or(0.0);
            let track = hexp(palette::frame()).unwrap();
            let erase = |rect: (u16, u16, u16, u16)| {
                let (cx, cy, cw, ch) = rect;
                format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{base}\"/>",
                    gx0 + cx as u32 * CW,
                    gy0 + cy as u32 * CH,
                    cw as u32 * CW,
                    ch as u32 * CH
                )
            };
            for (i, tab) in tabs.iter().enumerate() {
                app.tab = *tab;
                let mut term = Terminal::new(TestBackend::new(W, H)).unwrap();
                term.draw(|f| app.render(f)).unwrap();
                let mut body = frame_svg(term.backend().buffer(), gx0, gy0, &base);
                // On the Overview, swap the transcribed block-bars/gauge for real vector charts.
                if *tab == Tab::Overview {
                    if !trend.is_empty()
                        && let Some(rect) = TREND_RECT.with(|c| c.get())
                    {
                        body.push_str(&erase(rect));
                        body.push_str(&trend_svg(&trend, rect, gx0, gy0, &green, &muted));
                    }
                    if let Some(rect) = GAUGE_RECT.with(|c| c.get()) {
                        body.push_str(&erase(rect));
                        body.push_str(&gauge_svg(rect, pct, gx0, gy0, &green, &track));
                    }
                }
                let init = if i == 0 { "1" } else { "0" };
                svg.push_str(&format!("<g opacity=\"{init}\"><animate attributeName=\"opacity\" dur=\"12s\" repeatCount=\"indefinite\" values=\"{}\" keyTimes=\"{}\"/>{body}</g>", anim[i].0, anim[i].1));
            }
            svg.push_str("</svg>\n");

            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets")
                .join(file);
            std::fs::write(&path, svg).unwrap();
            eprintln!("wrote {}", path.display());
        }
    }
}
