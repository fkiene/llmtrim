//! The breakdown TUI's color palette — one place to tune the whole look.
//!
//! Built on **Catppuccin** (a public MIT palette) with four selectable flavors: the three
//! dark variants `Mocha` / `Macchiato` / `Frappé` and the light `Latte`. Whichever is
//! active, the UI lives on a tight tonal ramp — a base tone, a couple of barely-distinct
//! surface tiers, soft text — so chrome physically recedes. The eye is directed by one
//! **navigation accent** (mauve, reserved for focus: active tab, focused pane, selection)
//! plus **data hues** that only appear when they carry meaning: BLUE = a controllable money
//! fact, GREEN = the win / all-clear, ALARM = breakage, WARN = a soft caution. On a healthy
//! data screen only blue + green show; the accent is chrome.
//!
//! The active flavor is process-global (the TUI is single-threaded): set it once from
//! `LLMTRIM_THEME` at startup ([`set`]/[`from_name`]) and cycle it live with the `t` key
//! ([`cycle`]). Every element reads its color through the accessor fns ([`accent`], [`bg`],
//! …) so a toggle repaints instantly.

use std::sync::atomic::{AtomicU8, Ordering};

use ratatui::style::Color;

/// One flavor's full set of semantic colors. Field names are roles, not Catppuccin labels;
/// the mapping (per flavor) is base→`bg`, surface0→`surface`, surface1→`select_bg`,
/// surface2→`frame`, overlay1→`muted_gray`, text→`text`/`selection_fg`, mauve→`accent`,
/// blue/green/red/peach→`blue`/`green`/`alarm`/`warn`.
pub struct Palette {
    pub bg: Color,
    pub surface: Color,
    pub select_bg: Color,
    pub frame: Color,
    pub muted_gray: Color,
    pub text: Color,
    pub accent: Color,
    pub blue: Color,
    pub green: Color,
    pub alarm: Color,
    pub warn: Color,
    pub selection_fg: Color,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::Rgb(r, g, b)
}

pub const MOCHA: Palette = Palette {
    bg: rgb(0x1E, 0x1E, 0x2E),
    surface: rgb(0x31, 0x32, 0x44),
    select_bg: rgb(0x45, 0x47, 0x5A),
    frame: rgb(0x58, 0x5B, 0x70),
    muted_gray: rgb(0x7F, 0x84, 0x9C),
    text: rgb(0xCD, 0xD6, 0xF4),
    accent: rgb(0xCB, 0xA6, 0xF7),
    blue: rgb(0x89, 0xB4, 0xFA),
    green: rgb(0xA6, 0xE3, 0xA1),
    alarm: rgb(0xF3, 0x8B, 0xA8),
    warn: rgb(0xFA, 0xB3, 0x87),
    selection_fg: rgb(0xCD, 0xD6, 0xF4),
};

pub const MACCHIATO: Palette = Palette {
    bg: rgb(0x24, 0x27, 0x3A),
    surface: rgb(0x36, 0x3A, 0x4F),
    select_bg: rgb(0x49, 0x4D, 0x64),
    frame: rgb(0x5B, 0x60, 0x78),
    muted_gray: rgb(0x80, 0x87, 0xA2),
    text: rgb(0xCA, 0xD3, 0xF5),
    accent: rgb(0xC6, 0xA0, 0xF6),
    blue: rgb(0x8A, 0xAD, 0xF4),
    green: rgb(0xA6, 0xDA, 0x95),
    alarm: rgb(0xED, 0x87, 0x96),
    warn: rgb(0xF5, 0xA9, 0x7F),
    selection_fg: rgb(0xCA, 0xD3, 0xF5),
};

pub const FRAPPE: Palette = Palette {
    bg: rgb(0x30, 0x34, 0x46),
    surface: rgb(0x41, 0x45, 0x59),
    select_bg: rgb(0x51, 0x57, 0x6D),
    frame: rgb(0x62, 0x68, 0x80),
    muted_gray: rgb(0x83, 0x8B, 0xA7),
    text: rgb(0xC6, 0xD0, 0xF5),
    accent: rgb(0xCA, 0x9E, 0xE6),
    blue: rgb(0x8C, 0xAA, 0xEE),
    green: rgb(0xA6, 0xD1, 0x89),
    alarm: rgb(0xE7, 0x82, 0x84),
    warn: rgb(0xEF, 0x9F, 0x76),
    selection_fg: rgb(0xC6, 0xD0, 0xF5),
};

pub const LATTE: Palette = Palette {
    bg: rgb(0xEF, 0xF1, 0xF5),
    surface: rgb(0xCC, 0xD0, 0xDA),
    select_bg: rgb(0xBC, 0xC0, 0xCC),
    frame: rgb(0xAC, 0xB0, 0xBE),
    muted_gray: rgb(0x8C, 0x8F, 0xA1),
    text: rgb(0x4C, 0x4F, 0x69),
    accent: rgb(0x88, 0x39, 0xEF),
    blue: rgb(0x1E, 0x66, 0xF5),
    green: rgb(0x40, 0xA0, 0x2B),
    alarm: rgb(0xD2, 0x0F, 0x39),
    warn: rgb(0xFE, 0x64, 0x0B),
    selection_fg: rgb(0x4C, 0x4F, 0x69),
};

/// The selectable flavors, in cycle order (dark first, light last).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Mocha,
    Macchiato,
    Frappe,
    Latte,
}

const FLAVORS: [Flavor; 4] = [
    Flavor::Mocha,
    Flavor::Macchiato,
    Flavor::Frappe,
    Flavor::Latte,
];

/// Process-global active flavor index (the TUI is single-threaded; `Relaxed` is enough).
static CURRENT: AtomicU8 = AtomicU8::new(0);

/// The active flavor.
pub fn flavor() -> Flavor {
    FLAVORS[(CURRENT.load(Ordering::Relaxed) as usize) % FLAVORS.len()]
}

/// Set the active flavor (e.g. from `LLMTRIM_THEME` at startup).
pub fn set(f: Flavor) {
    let i = FLAVORS.iter().position(|x| *x == f).unwrap_or(0);
    CURRENT.store(i as u8, Ordering::Relaxed);
}

/// Advance to the next flavor (the `t` key); returns the new one.
pub fn cycle() -> Flavor {
    let n = (CURRENT.load(Ordering::Relaxed) as usize + 1) % FLAVORS.len();
    CURRENT.store(n as u8, Ordering::Relaxed);
    FLAVORS[n]
}

/// Display name of the active flavor.
pub fn name() -> &'static str {
    match flavor() {
        Flavor::Mocha => "Mocha",
        Flavor::Macchiato => "Macchiato",
        Flavor::Frappe => "Frappé",
        Flavor::Latte => "Latte",
    }
}

/// The config-file slug of the active flavor (ASCII, lowercase) — the form [`from_name`]
/// parses and `save_theme` persists.
pub fn ident() -> &'static str {
    match flavor() {
        Flavor::Mocha => "mocha",
        Flavor::Macchiato => "macchiato",
        Flavor::Frappe => "frappe",
        Flavor::Latte => "latte",
    }
}

/// Parse a flavor name (case-insensitive; accepts `frappe`/`frappé`). `None` if unknown.
pub fn from_name(s: &str) -> Option<Flavor> {
    match s.trim().to_lowercase().as_str() {
        "mocha" => Some(Flavor::Mocha),
        "macchiato" => Some(Flavor::Macchiato),
        "frappe" | "frappé" => Some(Flavor::Frappe),
        "latte" => Some(Flavor::Latte),
        _ => None,
    }
}

/// The active flavor's full palette.
pub fn active() -> &'static Palette {
    match flavor() {
        Flavor::Mocha => &MOCHA,
        Flavor::Macchiato => &MACCHIATO,
        Flavor::Frappe => &FRAPPE,
        Flavor::Latte => &LATTE,
    }
}

// Per-role accessors — every render site reads color through these so a flavor toggle
// repaints the whole UI on the next frame.
pub fn bg() -> Color {
    active().bg
}
pub fn surface() -> Color {
    active().surface
}
pub fn select_bg() -> Color {
    active().select_bg
}
pub fn frame() -> Color {
    active().frame
}
pub fn muted_gray() -> Color {
    active().muted_gray
}
pub fn text() -> Color {
    active().text
}
pub fn accent() -> Color {
    active().accent
}
pub fn blue() -> Color {
    active().blue
}
pub fn green() -> Color {
    active().green
}
pub fn alarm() -> Color {
    active().alarm
}
pub fn warn() -> Color {
    active().warn
}
pub fn selection_fg() -> Color {
    active().selection_fg
}
