//! Shared terminal UI — one visual voice for every human-facing command.
//!
//! The design language: rounded panels (`╭─╮│╰╯`) for multi-row summaries, glyph
//! status lines (`✓`/`•`/`⚠`) for single events, the `#99ccff` savings accent,
//! dim chrome, and cargo-style `error:` rendering. Everything returns a `String`
//! and takes an explicit `color: bool` computed once at the edge (`color_stdout`/
//! `color_stderr`: `NO_COLOR` + TTY), so piped output is byte-clean.
//!
//! Out of scope, by contract: machine outputs (`compress`/`send` stdout JSON,
//! `monitor --json/--csv`, the offline bench line, `bench --json-out`) and
//! `serve`'s stderr lines, which double as the daemon log file and stay plain.

use owo_colors::{OwoColorize, Style};
use unicode_width::UnicodeWidthChar;

// ── colour ──────────────────────────────────────────────────────────────────────

/// The palette. One accent, restrained everywhere else.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// Savings / success accent (#99ccff).
    Accent,
    Dim,
    Bold,
    /// Warnings and regressions.
    Warn,
    /// Fatal errors.
    Err,
}

fn style(tone: Tone) -> Style {
    match tone {
        Tone::Accent => Style::new().truecolor(153, 204, 255),
        Tone::Dim => Style::new().dimmed(),
        Tone::Bold => Style::new().bold(),
        Tone::Warn => Style::new().yellow(),
        Tone::Err => Style::new().red().bold(),
    }
}

pub fn paint(color: bool, tone: Tone, s: &str) -> String {
    if color {
        s.style(style(tone)).to_string()
    } else {
        s.to_string()
    }
}

/// Colour stdout output? `NO_COLOR` unset and stdout is an interactive terminal.
pub fn color_stdout() -> bool {
    use std::io::IsTerminal;
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

/// Colour stderr output (error/warning rendering)?
pub fn color_stderr() -> bool {
    use std::io::IsTerminal;
    std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal()
}

/// Is stdout an interactive terminal? Gates screen-control escapes (watch-mode
/// clear), which must never reach a pipe regardless of `NO_COLOR`.
pub fn stdout_is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

// ── glyphs & status lines ───────────────────────────────────────────────────────

pub const OK: &str = "✓";
pub const NOTE: &str = "•";
pub const WARN: &str = "⚠";

/// `✓ message` — a completed step / success event.
pub fn ok(color: bool, msg: &str) -> String {
    format!("{} {msg}", paint(color, Tone::Accent, OK))
}

/// `• message` — a neutral note (skipped step, kept file, FYI).
pub fn note(color: bool, msg: &str) -> String {
    format!("{} {msg}", paint(color, Tone::Dim, NOTE))
}

/// `⚠ message` — a non-fatal warning.
pub fn warn(color: bool, msg: &str) -> String {
    format!("{} {msg}", paint(color, Tone::Warn, WARN))
}

// ── measurement ─────────────────────────────────────────────────────────────────

/// Terminal display width, ignoring ANSI escape sequences (multibyte- and
/// wide-character-correct, unlike byte or char counts).
pub fn visible_width(s: &str) -> usize {
    let mut w = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else if c == '\x1b' {
            in_esc = true;
        } else {
            w += c.width().unwrap_or(0);
        }
    }
    w
}

/// Truncate to `max` display cells, appending `…` when cut.
pub fn truncate(s: &str, max: usize) -> String {
    if visible_width(s) <= max {
        return s.to_string();
    }
    let budget = max.saturating_sub(1);
    let mut out = String::new();
    let mut w = 0;
    for c in s.chars() {
        let cw = c.width().unwrap_or(0);
        if w + cw > budget {
            break;
        }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out
}

// ── panels ──────────────────────────────────────────────────────────────────────

/// A rounded panel with a title in the top border:
///
/// ```text
///  ╭─ title ──────────────╮
///  │ line                 │
///  ╰──────────────────────╯
/// ```
///
/// Width follows the widest line (display cells, ANSI-stripped). Borders dim,
/// title bold; the glyphs render in plain output too — only ANSI is gated.
pub fn panel(color: bool, title: &str, lines: &[String]) -> String {
    let content_w = lines
        .iter()
        .map(|l| visible_width(l))
        .max()
        .unwrap_or(0)
        .max(visible_width(title) + 2);
    let inner = content_w + 2; // one space of padding each side
    let border = |s: &str| paint(color, Tone::Dim, s);

    let fill = inner.saturating_sub(visible_width(title) + 3);
    let mut o = format!(
        " {}{}{}\n",
        border("╭─ "),
        paint(color, Tone::Bold, title),
        border(&format!(" {}╮", "─".repeat(fill))),
    );
    for l in lines {
        let pad = " ".repeat(content_w.saturating_sub(visible_width(l)));
        o.push_str(&format!(" {} {l}{pad} {}\n", border("│"), border("│")));
    }
    o.push_str(&format!(
        " {}\n",
        border(&format!("╰{}╯", "─".repeat(inner)))
    ));
    o
}

/// Aligned `glyph label  detail` rows for a checklist panel — labels pad to the
/// widest so the details form a column. Glyph tone: `✓` accent, `⚠` warn, else dim.
pub fn kv_rows(color: bool, rows: &[(&str, String, String)]) -> Vec<String> {
    let w = rows
        .iter()
        .map(|(_, l, _)| visible_width(l))
        .max()
        .unwrap_or(0);
    rows.iter()
        .map(|(g, label, detail)| {
            let tone = match *g {
                OK => Tone::Accent,
                WARN => Tone::Warn,
                _ => Tone::Dim,
            };
            let pad = " ".repeat(w.saturating_sub(visible_width(label)));
            format!("{} {label}{pad}  {detail}", paint(color, tone, g))
        })
        .collect()
}

// ── tables ──────────────────────────────────────────────────────────────────────

/// A rounded-corner table matching the panel chrome: dim headers, condensed rows.
/// Cells may carry ANSI when `color` is true — the `custom_styling` feature keeps
/// the width math correct.
pub fn table(color: bool, headers: &[&str]) -> comfy_table::Table {
    use comfy_table::{Cell, ContentArrangement, Table, modifiers, presets};
    let mut t = Table::new();
    // Outer border + header rule only — column separators would fight the panel chrome.
    t.load_preset(presets::UTF8_BORDERS_ONLY);
    t.apply_modifier(modifiers::UTF8_ROUND_CORNERS);
    t.set_content_arrangement(ContentArrangement::Disabled);
    t.set_header(
        headers
            .iter()
            .map(|h| Cell::new(paint(color, Tone::Dim, h))),
    );
    t
}

/// A right-aligned cell (numeric columns).
pub fn right(s: impl Into<String>) -> comfy_table::Cell {
    comfy_table::Cell::new(s.into()).set_alignment(comfy_table::CellAlignment::Right)
}

// ── numbers ─────────────────────────────────────────────────────────────────────

/// Compact human token count: 1_234_567 → "1.2M", 12_345 → "12.3K".
pub fn human(n: i64) -> String {
    let a = n.unsigned_abs();
    let sign = if n < 0 { "-" } else { "" };
    if a >= 1_000_000 {
        format!("{sign}{:.1}M", a as f64 / 1_000_000.0)
    } else if a >= 1_000 {
        format!("{sign}{:.1}K", a as f64 / 1_000.0)
    } else {
        format!("{sign}{a}")
    }
}

/// Group digits with commas: 1234567 → "1,234,567".
pub fn commas(n: i64) -> String {
    let s = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

/// Percent reduction from `before` to `after` (0 when `before` is 0) — the one
/// savings formula shared by monitor, bench, and eval reporting.
pub fn saved_pct(before: f64, after: f64) -> f64 {
    if before > 0.0 {
        (before - after) / before * 100.0
    } else {
        0.0
    }
}

// ── errors ──────────────────────────────────────────────────────────────────────

/// Cargo-style fatal error: bold red `error:`, the message, then the dim
/// `caused by:` chain from the anyhow context stack.
pub fn render_error(color: bool, err: &anyhow::Error) -> String {
    let mut o = format!("{} {err}\n", paint(color, Tone::Err, "error:"));
    for cause in err.chain().skip(1) {
        o.push_str(&paint(color, Tone::Dim, &format!("  caused by: {cause}\n")));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_width_ignores_ansi_and_counts_cells() {
        assert_eq!(visible_width("\x1b[1mab\x1b[0m"), 2);
        assert_eq!(visible_width("a·b"), 3); // multibyte, single cell
        assert_eq!(visible_width("日本"), 4); // wide chars are 2 cells
    }

    #[test]
    fn truncate_is_width_aware() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 4), "hel…");
        assert_eq!(truncate("日本語", 5), "日本…");
    }

    #[test]
    fn panel_borders_align_with_multibyte_title() {
        // `·` is multibyte: the old byte-length math drew a short top border.
        let out = panel(false, "saved (projected · A/B)", &["x".repeat(40)]);
        let widths: Vec<usize> = out.lines().map(visible_width).collect();
        assert!(
            widths.windows(2).all(|w| w[0] == w[1]),
            "ragged panel: {out}"
        );
    }

    #[test]
    fn panel_plain_has_no_ansi() {
        let out = panel(false, "t", &["line".to_string()]);
        assert!(!out.contains('\x1b'));
        assert!(out.contains("╭─ t"));
    }

    #[test]
    fn kv_rows_align_details() {
        let rows = kv_rows(
            false,
            &[
                (OK, "Local CA".into(), "/tmp/ca.pem".into()),
                (NOTE, "Profile".into(), "kept".into()),
            ],
        );
        assert_eq!(
            rows[0].find("/tmp/ca.pem").unwrap(),
            rows[1].find("kept").unwrap(),
            "details misaligned: {rows:?}"
        );
    }

    #[test]
    fn human_and_commas() {
        assert_eq!(human(1_234_567), "1.2M");
        assert_eq!(human(12_345), "12.3K");
        assert_eq!(human(512), "512");
        assert_eq!(commas(1_234_567), "1,234,567");
    }

    #[test]
    fn render_error_plain_and_chained() {
        let err = anyhow::anyhow!("io fail").context("failed to read corpus");
        let out = render_error(false, &err);
        assert!(out.starts_with("error: failed to read corpus"));
        assert!(out.contains("caused by: io fail"));
        assert!(!out.contains('\x1b'));
    }

    #[test]
    fn paint_gates_ansi_on_color() {
        assert_eq!(paint(false, Tone::Accent, "x"), "x");
        assert!(paint(true, Tone::Accent, "x").contains('\x1b'));
    }
}
