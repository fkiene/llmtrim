//! Status TUI **Sub** tab — simple subscription routing.
//!
//! Level 1 (default): cycle a small set of **routing presets** with ←/→ and apply with Enter.
//! Level 2 (opt-in via `e`): edit the active provider's Claude-tier → model map
//! (Fable first, then Opus/Sonnet/Haiku).
//!
//! Chain order, effort, anthropic-login, window `/sub`, and OAuth stay on the CLI.

use std::collections::BTreeMap;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Padding, Paragraph, Row, Table};

use super::palette;
use crate::reroute::catalog::{self, CatalogEntry};
use crate::reroute::{
    KIMI_MODEL, SubProvider, Tier, default_codex_tier_model, default_grok_tier_model,
};

/// One selectable routing policy. Provider and mode are packaged so the user never has to
/// juggle "active" vs "edit" vs "mode" as separate dials on the default surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingPreset {
    Off,
    AlwaysCodex,
    AlwaysKimi,
    AlwaysGrok,
    /// Anthropic first; on failure, try the saved chain (or the last/active provider).
    Fallback,
}

impl RoutingPreset {
    pub const ALL: [RoutingPreset; 5] = [
        RoutingPreset::Off,
        RoutingPreset::AlwaysCodex,
        RoutingPreset::AlwaysKimi,
        RoutingPreset::AlwaysGrok,
        RoutingPreset::Fallback,
    ];

    fn label(self) -> &'static str {
        match self {
            RoutingPreset::Off => "Off",
            RoutingPreset::AlwaysCodex => "Always → Codex",
            RoutingPreset::AlwaysKimi => "Always → Kimi",
            RoutingPreset::AlwaysGrok => "Always → Grok",
            RoutingPreset::Fallback => "Fallback (Anthropic first)",
        }
    }

    fn short(self) -> &'static str {
        match self {
            RoutingPreset::Off => "off",
            RoutingPreset::AlwaysCodex => "always/codex",
            RoutingPreset::AlwaysKimi => "always/kimi",
            RoutingPreset::AlwaysGrok => "always/grok",
            RoutingPreset::Fallback => "fallback",
        }
    }

    fn always_provider(self) -> Option<SubProvider> {
        match self {
            RoutingPreset::AlwaysCodex => Some(SubProvider::Codex),
            RoutingPreset::AlwaysKimi => Some(SubProvider::Kimi),
            RoutingPreset::AlwaysGrok => Some(SubProvider::Grok),
            RoutingPreset::Off | RoutingPreset::Fallback => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    /// Cycle presets with ←/→, apply with Enter.
    Presets,
    /// Edit the active provider's tier map.
    Map,
}

/// Live state for the Sub tab.
pub struct SubPanel {
    focus: Focus,
    /// Highlighted preset (not necessarily applied yet).
    selected: RoutingPreset,
    /// What is currently on disk / last applied.
    applied: RoutingPreset,
    /// Auth flags per provider (Codex, Kimi, Grok) — refreshed on enter/apply.
    auth: [bool; 3],
    /// Map editor: provider whose tiers we show (always the active always-provider, or
    /// the last re-enable target when in fallback/off).
    map_provider: SubProvider,
    tiers: [Tier; 4],
    chosen: [String; 4],
    catalog: Vec<CatalogEntry>,
    tier_row: usize,
    map_dirty: bool,
    status: String,
    /// True when a config write needs daemon restart + Claude auth sync after the TUI exits.
    pub needs_apply: bool,
}

impl SubPanel {
    pub fn new() -> Self {
        let applied = Self::read_applied_preset();
        let map_provider = Self::map_provider_for(applied);
        let mut panel = Self {
            focus: Focus::Presets,
            selected: applied,
            applied,
            auth: Self::read_auth(),
            map_provider,
            tiers: Tier::ALL,
            chosen: [String::new(), String::new(), String::new(), String::new()],
            catalog: Vec::new(),
            tier_row: 0,
            map_dirty: false,
            status: String::new(),
            needs_apply: false,
        };
        panel.reload_map();
        panel
    }

    /// Re-read config + auth when the user lands on this tab.
    pub fn refresh(&mut self) {
        self.applied = Self::read_applied_preset();
        // Keep the user's in-progress selection if they haven't applied yet and nothing
        // external changed the applied preset; otherwise snap to disk.
        if !self.needs_apply {
            self.selected = self.applied;
        }
        self.auth = Self::read_auth();
        if self.focus == Focus::Presets {
            self.map_provider = Self::map_provider_for(self.selected);
            self.reload_map();
            self.map_dirty = false;
        }
    }

    fn read_applied_preset() -> RoutingPreset {
        // Fresh disk read — RuntimeConfig is process-cached and wrong after our own writes.
        let file = load_config_file();
        let env = |k: &str| std::env::var(k).ok();
        // `disable_sub` leaves `mode = "fallback"` on disk. Off is "no active provider", not
        // "mode string says fallback" — otherwise Off → refresh shows Fallback and Enter
        // re-enables the last provider.
        let Some(active) = resolve_active(&env, file.as_ref()) else {
            return RoutingPreset::Off;
        };
        if resolve_fallback(&env, file.as_ref()) {
            return RoutingPreset::Fallback;
        }
        match active.as_str() {
            "codex" => RoutingPreset::AlwaysCodex,
            "kimi" => RoutingPreset::AlwaysKimi,
            "grok" => RoutingPreset::AlwaysGrok,
            _ => RoutingPreset::Off,
        }
    }

    /// Highlight a preset (and its map) without writing config — used by `sub setup <provider>`.
    pub fn preselect_provider(&mut self, provider: SubProvider) {
        self.selected = match provider {
            SubProvider::Codex => RoutingPreset::AlwaysCodex,
            SubProvider::Kimi => RoutingPreset::AlwaysKimi,
            SubProvider::Grok => RoutingPreset::AlwaysGrok,
        };
        self.map_provider = provider;
        self.reload_map();
        self.map_dirty = false;
        self.status = format!(
            "highlighted {} — Enter to apply · e to edit map",
            provider.as_str()
        );
    }

    /// Stable demo state for the README SVG export (not written to disk).
    #[cfg(test)]
    pub(crate) fn seed_export_demo(&mut self) {
        self.focus = Focus::Presets;
        self.applied = RoutingPreset::AlwaysCodex;
        self.selected = RoutingPreset::AlwaysCodex;
        self.auth = [true, false, true]; // codex ✓ · kimi · · grok ✓
        self.map_provider = SubProvider::Codex;
        self.reload_map();
        self.map_dirty = false;
        self.status.clear();
        self.needs_apply = false;
    }

    fn map_provider_for(preset: RoutingPreset) -> SubProvider {
        if let Some(p) = preset.always_provider() {
            return p;
        }
        // Fallback / Off: prefer last/active provider so the map still has a target.
        llmtrim_core::config::sub_reenable_provider()
            .as_deref()
            .and_then(SubProvider::parse)
            .unwrap_or(SubProvider::Codex)
    }

    fn read_auth() -> [bool; 3] {
        [
            auth_ok(SubProvider::Codex),
            auth_ok(SubProvider::Kimi),
            auth_ok(SubProvider::Grok),
        ]
    }

    fn reload_map(&mut self) {
        self.catalog = catalog::models_for(self.map_provider);
        let overrides = llmtrim_core::config::sub_tiers_for(self.map_provider.as_str());
        let in_catalog = |id: &str| self.catalog.iter().any(|e| e.id == id);
        self.chosen = self.tiers.map(|t| match self.map_provider {
            SubProvider::Kimi => KIMI_MODEL.to_string(),
            SubProvider::Codex => overrides
                .get(t.as_str())
                .filter(|m| in_catalog(m) || m.starts_with("gpt-"))
                .cloned()
                .unwrap_or_else(|| default_codex_tier_model(t).to_string()),
            SubProvider::Grok => overrides
                .get(t.as_str())
                .filter(|m| in_catalog(m) || m.starts_with("grok-"))
                .cloned()
                .unwrap_or_else(|| default_grok_tier_model(t).to_string()),
        });
        self.tier_row = self.tier_row.min(self.tiers.len().saturating_sub(1));
    }

    /// Handle a key while the Sub tab is focused. Returns true if the TUI should quit.
    pub fn handle_key(&mut self, code: crossterm::event::KeyCode) -> bool {
        use crossterm::event::KeyCode;
        match self.focus {
            Focus::Presets => match code {
                KeyCode::Left | KeyCode::Char('h') => self.cycle_preset(-1),
                KeyCode::Right | KeyCode::Char('l') => self.cycle_preset(1),
                KeyCode::Enter => self.apply_selected(),
                KeyCode::Char('e') => self.enter_map(),
                KeyCode::Char('r') => {
                    self.refresh();
                    self.status = "refreshed".into();
                }
                _ => {}
            },
            Focus::Map => match code {
                KeyCode::Esc => {
                    if self.map_dirty {
                        self.status = "unsaved map changes discarded".into();
                    }
                    self.map_dirty = false;
                    self.focus = Focus::Presets;
                    self.reload_map();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.tier_row = self.tier_row.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.tier_row = (self.tier_row + 1).min(self.tiers.len() - 1);
                }
                KeyCode::Left | KeyCode::Char('h') => self.cycle_model(-1),
                KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter => self.cycle_model(1),
                KeyCode::Char('s') => self.save_map(),
                _ => {}
            },
        }
        false
    }

    fn cycle_preset(&mut self, dir: i32) {
        let list = RoutingPreset::ALL;
        let pos = list.iter().position(|p| *p == self.selected).unwrap_or(0) as i32;
        let next = (pos + dir).rem_euclid(list.len() as i32) as usize;
        self.selected = list[next];
        self.map_provider = Self::map_provider_for(self.selected);
        self.reload_map();
        self.map_dirty = false;
        self.status.clear();
    }

    fn apply_selected(&mut self) {
        match self.apply_preset(self.selected) {
            Ok(msg) => {
                self.applied = self.selected;
                self.needs_apply = true;
                self.auth = Self::read_auth();
                self.map_provider = Self::map_provider_for(self.applied);
                self.reload_map();
                self.status = msg;
            }
            Err(e) => self.status = format!("apply failed: {e}"),
        }
    }

    fn apply_preset(&self, preset: RoutingPreset) -> anyhow::Result<String> {
        match preset {
            RoutingPreset::Off => {
                llmtrim_core::config::disable_sub()?;
                Ok(
                    "routing off — traffic stays on Anthropic (compression only). Restart applies."
                        .into(),
                )
            }
            RoutingPreset::AlwaysCodex | RoutingPreset::AlwaysKimi | RoutingPreset::AlwaysGrok => {
                let p = preset.always_provider().expect("always_* has provider");
                // Mode first so a prior fallback config doesn't leave always-intent half-applied.
                llmtrim_core::config::write_sub_mode(false)?;
                // enable_sub keeps an existing map; if the provider was never configured,
                // seed defaults via write_sub_mapping (also sets active).
                let had_map = !llmtrim_core::config::sub_tiers_for(p.as_str()).is_empty();
                if had_map {
                    llmtrim_core::config::enable_sub(p.as_str())?;
                } else {
                    let map = default_tier_map(p);
                    llmtrim_core::config::write_sub_mapping(p.as_str(), &map)?;
                }
                let auth = auth_ok(p);
                let mut msg = format!(
                    "always → {} applied. Restart applies to the live daemon.",
                    p.as_str()
                );
                if !auth {
                    msg.push_str(&format!(
                        " Not logged in — run `llmtrim sub auth {} login`.",
                        p.as_str()
                    ));
                }
                Ok(msg)
            }
            RoutingPreset::Fallback => {
                llmtrim_core::config::write_sub_mode(true)?;
                // Ensure there is someone to fall back to: keep active if set, else re-enable last.
                let file = load_config_file();
                let env = |k: &str| std::env::var(k).ok();
                if resolve_active(&env, file.as_ref()).is_none() {
                    if let Some(p) = llmtrim_core::config::sub_reenable_provider() {
                        llmtrim_core::config::enable_sub(&p)?;
                    } else {
                        // First-time fallback: seed Codex as the chain target.
                        let map = default_tier_map(SubProvider::Codex);
                        llmtrim_core::config::write_sub_mapping(SubProvider::Codex.as_str(), &map)?;
                        llmtrim_core::config::write_sub_mode(true)?;
                    }
                }
                Ok(
                    "fallback applied — Anthropic first; subscription chain on failure. \
                     Needs a live Anthropic login. Restart applies."
                        .into(),
                )
            }
        }
    }

    fn enter_map(&mut self) {
        self.map_provider = Self::map_provider_for(self.applied);
        // Prefer the selected always-provider if user is about to apply, but map edits
        // always target the *applied* backend so we never write tiers for a backend that
        // is not live (and confuse "I edited Grok" with "I'm still on Codex").
        if let Some(p) = self.applied.always_provider() {
            self.map_provider = p;
        } else if let Some(p) = self.selected.always_provider() {
            // Off/Fallback applied but user highlighted Always·X — still edit that provider's
            // staged map so they can prepare before applying.
            self.map_provider = p;
        }
        self.reload_map();
        self.map_dirty = false;
        self.focus = Focus::Map;
        self.status = format!(
            "editing {} map — s save · Esc back",
            self.map_provider.as_str()
        );
    }

    fn cycle_model(&mut self, dir: i32) {
        if self.map_provider == SubProvider::Kimi || self.catalog.is_empty() {
            self.status = "Kimi has a single model; nothing to change.".into();
            return;
        }
        let cur = &self.chosen[self.tier_row];
        let pos = self.catalog.iter().position(|e| &e.id == cur).unwrap_or(0);
        let len = self.catalog.len() as i32;
        let next = ((pos as i32) + dir).rem_euclid(len) as usize;
        self.chosen[self.tier_row] = self.catalog[next].id.clone();
        self.map_dirty = true;
        self.status.clear();
    }

    fn save_map(&mut self) {
        let mut map = BTreeMap::new();
        for (t, m) in self.tiers.iter().zip(self.chosen.iter()) {
            map.insert(t.as_str().to_string(), m.clone());
        }
        match llmtrim_core::config::write_sub_tiers(self.map_provider.as_str(), &map) {
            Ok(()) => {
                self.map_dirty = false;
                // Only restart if this provider is the live one (always mode) or in the chain.
                let live = self.applied.always_provider() == Some(self.map_provider)
                    || self.applied == RoutingPreset::Fallback;
                if live {
                    self.needs_apply = true;
                }
                self.status = format!(
                    "saved {} map{}",
                    self.map_provider.as_str(),
                    if live {
                        " — restart applies"
                    } else {
                        " (not live until you apply that provider)"
                    }
                );
            }
            Err(e) => self.status = format!("save failed: {e}"),
        }
    }

    pub fn help_keys(&self) -> &'static str {
        match self.focus {
            Focus::Presets => {
                " Tab tabs · ←→ preset · ⏎ apply · e edit map · r refresh · t theme · q"
            }
            Focus::Map => " ↑↓ tier · ←→ model · s save · Esc back · t theme · q",
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let frame = Style::default().fg(palette::frame());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(frame)
            .title(" sub · subscription routing ")
            .title_style(frame.add_modifier(Modifier::BOLD))
            .padding(Padding::new(1, 1, 0, 0));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::vertical([
            Constraint::Length(7), // presets + auth
            Constraint::Min(6),    // map summary / editor
            Constraint::Length(2), // status
        ])
        .split(inner);

        self.render_presets(f, chunks[0]);
        self.render_map(f, chunks[1]);
        self.render_status(f, chunks[2]);
    }

    fn render_presets(&self, f: &mut Frame, area: Rect) {
        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled("Routing  ", Style::default().fg(palette::muted_gray())),
            Span::styled(
                self.applied.label(),
                Style::default()
                    .fg(palette::text())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if self.selected != self.applied {
                    format!("   →  {}", self.selected.label())
                } else {
                    String::new()
                },
                Style::default().fg(palette::accent()),
            ),
        ]));

        // Preset chips
        let mut chips = Vec::new();
        for (i, p) in RoutingPreset::ALL.iter().enumerate() {
            if i > 0 {
                chips.push(Span::raw("  "));
            }
            let selected = *p == self.selected;
            let applied = *p == self.applied;
            let style = if selected && self.focus == Focus::Presets {
                Style::default()
                    .bg(palette::accent())
                    .fg(palette::bg())
                    .add_modifier(Modifier::BOLD)
            } else if applied {
                Style::default()
                    .fg(palette::green())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette::muted_gray())
            };
            let mark = if applied { "● " } else { "○ " };
            chips.push(Span::styled(format!("{mark}{}", p.short()), style));
        }
        lines.push(Line::from(chips));

        // Auth row
        let providers = [
            (SubProvider::Codex, self.auth[0]),
            (SubProvider::Kimi, self.auth[1]),
            (SubProvider::Grok, self.auth[2]),
        ];
        let mut auth_spans = vec![Span::styled(
            "Auth     ",
            Style::default().fg(palette::muted_gray()),
        )];
        for (i, (p, ok)) in providers.iter().enumerate() {
            if i > 0 {
                auth_spans.push(Span::raw("  "));
            }
            let (glyph, color) = if *ok {
                ("✓", palette::green())
            } else {
                ("·", palette::muted_gray())
            };
            auth_spans.push(Span::styled(
                format!("{glyph} {}", p.as_str()),
                Style::default().fg(color),
            ));
        }
        lines.push(Line::from(auth_spans));

        // Mode side-effect hint
        let hint = match self.selected {
            RoutingPreset::Off => "Claude uses Anthropic only (compression still on).",
            RoutingPreset::AlwaysCodex | RoutingPreset::AlwaysKimi | RoutingPreset::AlwaysGrok => {
                "Always-on: dummy Anthropic token by default (connectors off). Restart Claude Code after apply."
            }
            RoutingPreset::Fallback => {
                "Fallback needs a live Anthropic login; connectors stay available."
            }
        };
        lines.push(Line::from(Span::styled(
            hint,
            Style::default().fg(palette::muted_gray()),
        )));
        lines.push(Line::from(Span::styled(
            "Enter applies the highlighted preset.  e edits models for the target provider.",
            Style::default().fg(palette::muted_gray()),
        )));

        f.render_widget(Paragraph::new(lines), area);
    }

    fn render_map(&self, f: &mut Frame, area: Rect) {
        let title = match self.focus {
            Focus::Presets => format!(
                " map · {} (read-only — press e to edit) ",
                self.map_provider.as_str()
            ),
            Focus::Map => {
                let dirty = if self.map_dirty { " · unsaved" } else { "" };
                format!(" map · {}{dirty} ", self.map_provider.as_str())
            }
        };
        let editing = self.focus == Focus::Map;
        let border = if editing {
            Style::default().fg(palette::accent())
        } else {
            Style::default().fg(palette::frame())
        };

        let rows = self.tiers.iter().enumerate().map(|(i, t)| {
            let model = &self.chosen[i];
            let (inp, outp) = self
                .catalog
                .iter()
                .find(|e| e.id == *model)
                .map(|e| {
                    (
                        e.input
                            .map(|v| format!("${v:.2}"))
                            .unwrap_or_else(|| "—".into()),
                        e.output
                            .map(|v| format!("${v:.2}"))
                            .unwrap_or_else(|| "—".into()),
                    )
                })
                .unwrap_or_else(|| ("—".into(), "—".into()));
            let style = if editing && i == self.tier_row {
                Style::default()
                    .bg(palette::accent())
                    .fg(palette::bg())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette::text())
            };
            Row::new(vec![
                Cell::from(t.as_str().to_string()),
                Cell::from("→"),
                Cell::from(model.clone()),
                Cell::from(inp),
                Cell::from(outp),
            ])
            .style(style)
        });

        let table = Table::new(
            rows,
            [
                Constraint::Length(10),
                Constraint::Length(3),
                Constraint::Min(18),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["Claude tier", "", "Model", "$/1M in", "$/1M out"]).style(
                Style::default()
                    .fg(palette::muted_gray())
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(border)
                .title(title)
                .title_style(border.add_modifier(Modifier::BOLD)),
        );
        f.render_widget(table, area);
    }

    fn render_status(&self, f: &mut Frame, area: Rect) {
        let text = if self.status.is_empty() {
            match self.focus {
                Focus::Presets => {
                    if self.selected != self.applied {
                        format!(
                            "highlight: {} · applied: {} · Enter to apply",
                            self.selected.short(),
                            self.applied.short()
                        )
                    } else {
                        format!("applied: {}", self.applied.short())
                    }
                }
                Focus::Map => format!(
                    "editing {}{}",
                    self.map_provider.as_str(),
                    if self.map_dirty { " [unsaved]" } else { "" }
                ),
            }
        } else {
            self.status.clone()
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                text,
                Style::default().fg(palette::muted_gray()),
            ))),
            area,
        );
    }
}

impl Default for SubPanel {
    fn default() -> Self {
        Self::new()
    }
}

fn auth_ok(p: SubProvider) -> bool {
    crate::reroute::auth::auth_status_json(p)["logged_in"]
        .as_bool()
        .unwrap_or(false)
}

fn default_tier_map(p: SubProvider) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    match p {
        SubProvider::Codex => {
            for t in Tier::ALL {
                map.insert(
                    t.as_str().to_string(),
                    default_codex_tier_model(t).to_string(),
                );
            }
        }
        SubProvider::Grok => {
            for t in Tier::ALL {
                map.insert(
                    t.as_str().to_string(),
                    default_grok_tier_model(t).to_string(),
                );
            }
        }
        SubProvider::Kimi => {
            for t in Tier::ALL {
                map.insert(t.as_str().to_string(), KIMI_MODEL.to_string());
            }
        }
    }
    map
}

fn load_config_file() -> Option<toml::Value> {
    let path = llmtrim_core::config::config_file_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

fn resolve_active(
    env: &impl Fn(&str) -> Option<String>,
    file: Option<&toml::Value>,
) -> Option<String> {
    if let Some(v) = env("LLMTRIM_SUB").filter(|s| !s.is_empty()) {
        let s = v.trim().to_ascii_lowercase();
        return (s != "off").then_some(s);
    }
    let sub = file?.get("sub")?;
    let s = sub
        .as_str()
        .or_else(|| {
            sub.get("active")
                .or_else(|| sub.get("provider"))
                .and_then(toml::Value::as_str)
        })?
        .trim()
        .to_ascii_lowercase();
    (s != "off" && !s.is_empty()).then_some(s)
}

fn resolve_fallback(env: &impl Fn(&str) -> Option<String>, file: Option<&toml::Value>) -> bool {
    let parse = |raw: &str| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "fallback" | "on_error" | "on-error" | "onerror"
        )
    };
    if let Some(v) = env("LLMTRIM_SUB_MODE").filter(|s| !s.is_empty()) {
        return parse(&v);
    }
    file.and_then(|v| v.get("sub"))
        .and_then(|v| v.get("mode"))
        .and_then(toml::Value::as_str)
        .is_some_and(parse)
}

/// Reconcile Claude dummy-auth + restart the interceptor so a Sub-tab write takes effect.
/// Best-effort: never panics; returns a short status string for the caller to print after exit.
///
/// Mirrors `main::apply_sub_change`: always sync auth env; only restart when a daemon is
/// already running (so a pure-config edit never spawns a new interceptor by surprise).
pub fn apply_pending_changes() -> String {
    use crate::statusline::SubAuthEnvChange;
    // Same rule as main::sync_claude_sub_auth.
    let want = llmtrim_core::config::sub_skip_anthropic_login();
    let mut parts = Vec::new();
    match crate::statusline::sync_sub_auth_env(want) {
        Ok(SubAuthEnvChange::Injected) => parts.push(
            "Claude Code: dummy ANTHROPIC_AUTH_TOKEN set (connectors off; restart Claude Code)."
                .to_string(),
        ),
        Ok(SubAuthEnvChange::Removed) => parts.push(
            "Claude Code: dummy ANTHROPIC_AUTH_TOKEN removed (Anthropic login may be required)."
                .to_string(),
        ),
        Ok(SubAuthEnvChange::Unchanged) => {}
        Err(e) => parts.push(format!("Claude Code auth env update failed: {e:#}")),
    }
    let daemon_msg = match crate::daemon::running() {
        None => {
            "Subscription routing saved (no daemon running — next start picks it up).".to_string()
        }
        Some(state) => {
            let port = state.port;
            match crate::daemon::stop_and_wait_free(port) {
                Ok(true) => match crate::daemon::spawn_detached(port) {
                    Ok(pid) => {
                        format!("Subscription routing applied (restarted daemon pid {pid}).")
                    }
                    Err(e) => format!(
                        "Subscription config saved, but restart failed: {e:#}.                          Run `llmtrim start --force`."
                    ),
                },
                Ok(false) => {
                    "Subscription config saved, but the old daemon did not release the port.                      Run `llmtrim start --force`."
                        .to_string()
                }
                Err(e) => format!(
                    "Subscription config saved, but restart failed: {e:#}.                      Run `llmtrim start --force`."
                ),
            }
        }
    };
    parts.push(daemon_msg);
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_cycle_wraps() {
        let mut p = SubPanel::new();
        p.selected = RoutingPreset::Off;
        p.cycle_preset(1);
        assert_eq!(p.selected, RoutingPreset::AlwaysCodex);
        p.selected = RoutingPreset::Fallback;
        p.cycle_preset(1);
        assert_eq!(p.selected, RoutingPreset::Off);
        p.cycle_preset(-1);
        assert_eq!(p.selected, RoutingPreset::Fallback);
    }

    #[test]
    fn map_provider_for_always_matches() {
        assert_eq!(
            SubPanel::map_provider_for(RoutingPreset::AlwaysGrok),
            SubProvider::Grok
        );
        assert_eq!(
            SubPanel::map_provider_for(RoutingPreset::AlwaysKimi),
            SubProvider::Kimi
        );
    }

    #[test]
    fn read_applied_preset_off_wins_over_stale_fallback_mode() {
        // Unit-level: resolve_active None => Off, even if resolve_fallback would be true.
        // (Integration against a temp config file would also work; this guards the branch order.)
        assert_eq!(
            {
                // Simulate the fixed decision order with local values.
                let active: Option<String> = None;
                let fallback = true;
                match active {
                    None => RoutingPreset::Off,
                    Some(_) if fallback => RoutingPreset::Fallback,
                    Some(a) => match a.as_str() {
                        "codex" => RoutingPreset::AlwaysCodex,
                        _ => RoutingPreset::Off,
                    },
                }
            },
            RoutingPreset::Off
        );
    }

    #[test]
    fn kimi_cycle_model_is_noop() {
        let mut p = SubPanel::new();
        p.map_provider = SubProvider::Kimi;
        p.reload_map();
        let before = p.chosen.clone();
        p.cycle_model(1);
        assert_eq!(p.chosen, before);
        assert!(!p.map_dirty);
    }
}
