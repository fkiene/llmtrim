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
            // A CSI sequence ends at its final byte (an ASCII letter: `m` for SGR colour,
            // `H`/`J`/`K`/`h` for cursor/screen control); params/intermediates like `[`, `;`,
            // `?`, digits are not letters, so they don't end it.
            if c.is_ascii_alphabetic() {
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

/// Truncate to `max` display cells like [`truncate`], but ANSI-aware: escape sequences pass
/// through with zero width (never counted or cut mid-sequence), and a reset is appended after
/// the `…` so a cut inside a styled run doesn't bleed colour onto the rest of the screen. For
/// the watch repaint, which needs each line to be exactly one screen row.
pub fn truncate_visible(s: &str, max: usize) -> String {
    if visible_width(s) <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let budget = max.saturating_sub(1);
    let mut out = String::new();
    let mut w = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            out.push(c);
            if c.is_ascii_alphabetic() {
                in_esc = false;
            }
        } else if c == '\x1b' {
            in_esc = true;
            out.push(c);
        } else {
            let cw = c.width().unwrap_or(0);
            if w + cw > budget {
                break;
            }
            out.push(c);
            w += cw;
        }
    }
    out.push('…');
    out.push_str("\x1b[0m");
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

/// A rounded box with no title, sized to the widest line (display cells, ANSI-stripped).
/// Like [`panel`] but for a free-form hero block rather than a labelled summary.
///
/// ```text
///  ╭────────────────────────╮
///  │ $746 off your real bill │
///  ╰────────────────────────╯
/// ```
pub fn boxed(color: bool, lines: &[String]) -> String {
    let content_w = lines.iter().map(|l| visible_width(l)).max().unwrap_or(0);
    let inner = content_w + 2; // one space of padding each side
    let border = |s: &str| paint(color, Tone::Dim, s);
    let mut o = format!(" {}\n", border(&format!("╭{}╮", "─".repeat(inner))));
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

// ── logo motifs ───────────────────────────────────────────────────────────────────
//
// The visual narrative from the logo: a fluffy request is sheared down to just what
// matters, and the promise — "same answers, smaller bill" — is stamped beneath it.

/// The logo tagline, also the default `badge` message.
pub const TAGLINE: &str = "same answers, smaller bill";

/// The wordmark banner — the logo's hand-drawn `llmtrim` with its motion ticks:
///
/// ```text
/// ‹‹ llmtrim ››
/// ```
pub fn wordmark(color: bool) -> String {
    format!(
        "{} {} {}",
        paint(color, Tone::Dim, "‹‹"),
        paint(color, Tone::Bold, "llmtrim"),
        paint(color, Tone::Dim, "››"),
    )
}

/// The hero-number style: the one place the accent is combined with bold weight, for the
/// single dominant figure on the dashboard (`$746`).
pub fn hero(color: bool, s: &str) -> String {
    if color {
        s.style(style(Tone::Accent).bold()).to_string()
    } else {
        s.to_string()
    }
}

/// A compact Unicode block sparkline over `vals` (e.g. tokens saved per day), scaled to the
/// series max. Empty input → empty string; negatives floor at zero. Caller paints it.
pub fn sparkline(vals: &[i64]) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let max = vals.iter().copied().max().unwrap_or(0).max(1) as f64;
    let top = BLOCKS.len() - 1; // scale each value into 0..=top of the block ramp
    vals.iter()
        .map(|&v| BLOCKS[(((v.max(0) as f64 / max) * top as f64).round() as usize).min(top)])
        .collect()
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
    fn boxed_borders_align_and_plain_has_no_ansi() {
        let out = boxed(false, &["short".into(), "a longer line here".into()]);
        assert!(!out.contains('\x1b'));
        let widths: Vec<usize> = out.lines().map(visible_width).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "ragged box: {out}");
    }

    #[test]
    fn sparkline_scales_to_series_max() {
        assert_eq!(sparkline(&[]), "");
        assert_eq!(sparkline(&[0, 10]), "▁█"); // min floor → top block at the max
        let s = sparkline(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(s.chars().count(), 8);
        assert_eq!(s.chars().last().unwrap(), '█');
        assert_eq!(sparkline(&[-5, 10]).chars().next().unwrap(), '▁'); // negatives floor
    }

    #[test]
    fn truncate_visible_keeps_escapes_and_resets() {
        // Short string: unchanged.
        assert_eq!(truncate_visible("hello", 10), "hello");
        // Plain cut counts cells, not bytes.
        assert_eq!(truncate_visible("hello world", 4), "hel…\x1b[0m");
        // ANSI: the escape doesn't count toward width and isn't cut mid-sequence; result
        // ends with a reset so colour can't bleed past the cut.
        let colored = paint(true, Tone::Accent, "abcdef");
        let out = truncate_visible(&colored, 3);
        assert_eq!(visible_width(&out), 3); // 2 chars + the … = 3 cells
        assert!(out.starts_with('\x1b') && out.ends_with("\x1b[0m"));
    }

    #[test]
    fn hero_is_plain_without_color() {
        assert_eq!(hero(false, "$746"), "$746");
        assert!(hero(true, "$746").contains('\x1b'));
    }

    #[test]
    fn wordmark_plain_and_colored() {
        assert_eq!(wordmark(false), "‹‹ llmtrim ››");
        assert!(wordmark(true).contains('\x1b'));
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
