//! `llmtrim sub setup` — a small ratatui editor for the Claude-tier → provider-model mapping.
//!
//! Rows are the four Claude tiers Claude Code selects between (Opus/Sonnet/Haiku/Fable). Each row's
//! target is chosen from the provider catalog ([`super::catalog`]) with live pricing shown. Saving
//! writes `sub = <provider>` plus `[sub.<provider>.tiers]` to the config file via
//! [`llmtrim_core::config::write_sub_mapping`]. Kimi exposes a single model, so its mapping is
//! read-only (every tier collapses to `kimi-for-coding`).

use std::collections::BTreeMap;
use std::io::Stdout;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::{Frame, Terminal};

use super::catalog::{self, CatalogEntry};
use super::{SubProvider, Tier, default_codex_tier_model};

/// Entry point for `llmtrim sub setup`. Opens the editor for `provider` (Codex is editable; Kimi is
/// shown read-only). Blocks until the user quits; saving persists to the config file.
pub fn run(provider: SubProvider) -> Result<()> {
    let mut app = App::new(provider);
    let mut term = enter()?;
    let res = app.event_loop(&mut term);
    leave(&mut term);
    res
}

struct App {
    provider: SubProvider,
    tiers: [Tier; 4],
    /// The chosen model per tier (index parallel to `tiers`).
    chosen: [String; 4],
    /// Candidate models to cycle through.
    catalog: Vec<CatalogEntry>,
    selected: usize,
    dirty: bool,
    status: String,
    /// Current reroute mode, shown read-only (set it with `llmtrim sub mode`). The editor edits
    /// the model map only, so it flags where the other half of the setting lives.
    fallback: bool,
}

impl App {
    fn new(provider: SubProvider) -> Self {
        let tiers = Tier::ALL;
        let catalog = catalog::models_for(provider);
        // Seed from *this* provider's table — not RuntimeConfig::sub_tiers (active provider only).
        // Global `sub = codex` must not seed `llmtrim sub setup grok` with gpt-5.6-* ids.
        let rc = llmtrim_core::config::RuntimeConfig::get();
        let overrides = llmtrim_core::config::sub_tiers_for(provider.as_str());
        let in_catalog = |id: &str| catalog.iter().any(|e| e.id == id);
        let chosen = tiers.map(|t| match provider {
            SubProvider::Kimi => super::KIMI_MODEL.to_string(),
            SubProvider::Codex => overrides
                .get(t.as_str())
                .filter(|m| in_catalog(m) || m.starts_with("gpt-"))
                .cloned()
                .unwrap_or_else(|| default_codex_tier_model(t).to_string()),
            SubProvider::Grok => overrides
                .get(t.as_str())
                .filter(|m| in_catalog(m) || m.starts_with("grok-"))
                .cloned()
                .unwrap_or_else(|| super::default_grok_tier_model(t).to_string()),
        });
        Self {
            provider,
            tiers,
            chosen,
            catalog,
            selected: 0,
            dirty: false,
            status: String::new(),
            fallback: rc.sub_fallback,
        }
    }

    fn event_loop(&mut self, term: &mut Term) -> Result<()> {
        loop {
            term.draw(|f| self.draw(f)).context("draw failed")?;
            let Event::Key(key) = event::read().context("read event failed")? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Up | KeyCode::Char('k') => {
                    self.selected = self.selected.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.selected = (self.selected + 1).min(self.tiers.len() - 1);
                }
                KeyCode::Left | KeyCode::Char('h') => self.cycle(-1),
                KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter => self.cycle(1),
                KeyCode::Char('s') => self.save(),
                _ => {}
            }
        }
    }

    /// Cycle the selected tier's model through the catalog (Codex only; Kimi is fixed).
    fn cycle(&mut self, dir: i32) {
        if self.provider == SubProvider::Kimi || self.catalog.is_empty() {
            self.status = "Kimi exposes a single model; nothing to change.".into();
            return;
        }
        let cur = &self.chosen[self.selected];
        let pos = self.catalog.iter().position(|e| &e.id == cur).unwrap_or(0);
        let len = self.catalog.len() as i32;
        let next = (((pos as i32) + dir).rem_euclid(len)) as usize;
        self.chosen[self.selected] = self.catalog[next].id.clone();
        self.dirty = true;
        self.status.clear();
    }

    fn save(&mut self) {
        let mut map = BTreeMap::new();
        for (t, m) in self.tiers.iter().zip(self.chosen.iter()) {
            map.insert(t.as_str().to_string(), m.clone());
        }
        match llmtrim_core::config::write_sub_mapping(self.provider.as_str(), &map) {
            Ok(()) => {
                self.dirty = false;
                let path = llmtrim_core::config::config_file_path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                self.status = format!("saved to {path}");
            }
            Err(e) => self.status = format!("save failed: {e}"),
        }
    }

    fn price(&self, id: &str) -> Option<&CatalogEntry> {
        self.catalog.iter().find(|e| e.id == id)
    }

    fn draw(&self, f: &mut Frame) {
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(3),
        ])
        .split(f.area());

        let header = Paragraph::new(Line::from(vec![
            Span::raw("Reroute mapping — provider "),
            Span::styled(
                self.provider.as_str(),
                Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(
                "   mode: {} (set with `llmtrim sub mode`)",
                if self.fallback { "fallback" } else { "always" }
            )),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" llmtrim sub setup "),
        );
        f.render_widget(header, chunks[0]);

        let rows = self.tiers.iter().enumerate().map(|(i, t)| {
            let model = &self.chosen[i];
            let (inp, outp) = self
                .price(model)
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
            let style = if i == self.selected {
                Style::new().add_modifier(Modifier::REVERSED)
            } else {
                Style::new()
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
                Constraint::Min(20),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["Claude tier", "", "Model", "$/1M in", "$/1M out"])
                .style(Style::new().bold()),
        )
        .block(Block::default().borders(Borders::ALL).title(" tiers "));
        f.render_widget(table, chunks[1]);

        let hint = if self.dirty { "  [unsaved]" } else { "" };
        let footer = Paragraph::new(Line::from(format!(
            "↑↓ select · ←→/Enter change model · s save · q quit{hint}   {}",
            self.status
        )))
        .block(Block::default().borders(Borders::ALL));
        f.render_widget(footer, chunks[2]);
    }
}

type Term = Terminal<CrosstermBackend<Stdout>>;

fn enter() -> Result<Term> {
    enable_raw_mode().context("failed to enter raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(std::io::stdout());
    Terminal::new(backend).context("failed to build terminal")
}

fn leave(term: &mut Term) {
    let _ = disable_raw_mode();
    let _ = execute!(term.backend_mut(), LeaveAlternateScreen);
    let _ = term.show_cursor();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codex_app(catalog: Vec<CatalogEntry>) -> App {
        let start = catalog.first().map(|e| e.id.clone()).unwrap_or_default();
        App {
            provider: SubProvider::Codex,
            tiers: Tier::ALL,
            chosen: [start.clone(), start.clone(), start.clone(), start],
            catalog,
            selected: 0,
            dirty: false,
            status: String::new(),
            fallback: false,
        }
    }

    fn entry(id: &str) -> CatalogEntry {
        CatalogEntry {
            id: id.to_string(),
            input: None,
            output: None,
            cache_read: None,
        }
    }

    #[test]
    fn cycle_wraps_forward_and_backward_and_marks_dirty() {
        let mut app = codex_app(vec![entry("a"), entry("b"), entry("c")]);
        assert_eq!(app.chosen[0], "a");
        app.cycle(1);
        assert_eq!(app.chosen[0], "b");
        assert!(app.dirty);
        // Forward past the end wraps to the first entry.
        app.cycle(1);
        app.cycle(1);
        assert_eq!(app.chosen[0], "a");
        // Backward from the first entry wraps to the last.
        app.cycle(-1);
        assert_eq!(app.chosen[0], "c");
        // Only the selected tier moves.
        assert_eq!(app.chosen[1], "a");
    }

    #[test]
    fn cycle_is_a_noop_for_kimi() {
        let mut app = codex_app(vec![entry("a"), entry("b")]);
        app.provider = SubProvider::Kimi;
        app.cycle(1);
        assert_eq!(app.chosen[0], "a");
        assert!(!app.dirty);
        assert!(app.status.contains("single model"));
    }

    #[test]
    fn cycle_is_a_noop_with_empty_catalog() {
        let mut app = codex_app(Vec::new());
        app.cycle(1);
        assert!(!app.dirty);
    }
}
