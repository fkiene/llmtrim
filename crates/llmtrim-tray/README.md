# llmtrim-tray

A menu-bar / system-tray app that shows llmtrim's compression savings per agent
(Claude Code, Codex, Gemini, OpenCode, ...). Click the tray icon for a popover
with the aggregate savings, a per-agent breakdown, and a savings trend.

![llmtrim tray popover](docs/popover.svg)

Ships on macOS, Windows, and Linux (X11/Wayland via WebKitGTK and an AppIndicator
tray). On Linux it needs `libwebkit2gtk-4.1` and `libayatana-appindicator3` at
runtime.

## How it works

The app is a thin [Tauri v2](https://v2.tauri.app) shell. All of the data logic
lives in `llmtrim-ledger` and is tested there:

- `dashboard::build_dashboard` shapes the popover view model.
- `breakdown_db::BreakdownDb::agent_aggregates` / `agent_trend` read the ledger.

`src/main.rs` only wires those functions to the tray icon, the popover window,
and a background poll loop. The frontend (`src-ui/`) is Vite + TypeScript and
renders the snapshot it receives over Tauri IPC.

The crate is excluded from the workspace `default-members`, so the main CI and a
plain `cargo build` on Linux never try to compile Tauri. Its dedicated gate is
`.github/workflows/tray.yml`, which builds it on macOS, Windows, and Linux.

## How it ships

`llmtrim-tray` is `publish = false`: it never goes to crates.io, and
`cargo install llmtrim` does not build it (the published CLI keeps
`unsafe_code = "forbid"`, which Tauri's macros can't satisfy). It rides the
prebuilt channels instead. The release workflow builds it for five targets. On
the four desktop targets (macOS x64/arm64, Windows x64/arm64) it goes next to the
CLI in the same archive, so Homebrew, Scoop, and the npm platform packages all
carry the tray. The fifth target, Linux x86_64, builds in its own job and ships
as a standalone `llmtrim-tray-x86_64-unknown-linux-gnu.tar.gz` on the GitHub
Release rather than bundled (musl can't link Tauri; WebKitGTK is a glibc runtime
dependency).

## Develop

```bash
cd crates/llmtrim-tray
npm install
npm run tauri dev      # hot-reload the UI and the Rust shell
```

The popover reads the same ledger the proxy writes, resolved via
`LLMTRIM_DB_PATH` / `XDG_DATA_HOME`. Run the proxy at least once so the ledger
exists, otherwise the popover shows a "start the llmtrim proxy first" notice.

## Build a release binary

```bash
cd crates/llmtrim-tray
npm install
npm run build          # emits dist/, which tauri-build embeds at compile time
cargo build -p llmtrim-tray --release
```

### Bundle for distribution

```bash
cd crates/llmtrim-tray
npm run tauri build    # .app/.dmg on macOS, .msi/.exe on Windows
```

The raw `cargo build` above gives you a runnable binary; the installer bundle
needs `npm run tauri build`.

## Security posture

- The Tauri IPC contract carries only aggregate numbers. It never exposes
  `project`, `session_name`, `mcp_server`, or `tool_name`. This guarantee is the
  shape of the `Dashboard` / `AgentCard` structs in `llmtrim-ledger`
  (`dashboard.rs`); update this claim if you change those types.
- Errors crossing into the webview are stripped of filesystem paths; the full
  detail is logged to stderr.
- The CSP keeps `connect-src 'none'`; the capability allowlist excludes shell,
  fs, http, updater, and clipboard. The ledger is opened read-only.

## Icons

`tools/gen_icons.py` regenerates every icon from the two master SVGs
(`icons/tray.svg`, `icons/app-icon.svg`). The PNG/ICO/ICNS outputs are committed
so a build never needs Python:

```bash
cd crates/llmtrim-tray
python3 tools/gen_icons.py    # needs cairosvg + Pillow
```

The macOS tray glyph is a black template image (the system tints it for the
menu-bar theme); other platforms get the green glyph. The generator asserts the
template image is black-and-alpha only.
