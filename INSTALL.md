# Installing llmtrim

## Quick install (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh
```

Downloads the latest release binary for your platform into `~/.local/bin`. Override with:

```bash
LLMTRIM_INSTALL_DIR=/usr/local/bin LLMTRIM_VERSION=v0.1.0 \
  curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh
```

If `~/.local/bin` isn't on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc or ~/.zshrc
```

## Quick install (Windows)

```powershell
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Downloads the latest release binary into `%LOCALAPPDATA%\llmtrim\bin` and adds it to your
user `PATH`. Override with:

```powershell
$env:LLMTRIM_VERSION = "v0.1.0"    # pin a release
$env:LLMTRIM_NO_SETUP = "1"        # install the binary only, skip `setup`
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Open a new PowerShell window afterward so the `PATH` and profile env apply. Prebuilt binaries
ship for both x64 and ARM64. WSL users: use the Linux line above.

## Homebrew (macOS / Linux)

```bash
brew install fkiene/tap/llmtrim
# or, from this repo's formula:
brew install --build-from-source ./Formula/llmtrim.rb
```

## With Cargo

```bash
cargo install --git https://github.com/fkiene/llmtrim
```

## From source

```bash
git clone https://github.com/fkiene/llmtrim
cd llmtrim
cargo build --release
# binary at target/release/llmtrim
cargo install --path .
```

Requires Rust 1.88+ (edition 2024). `rusqlite` is bundled (no system SQLite needed) and pinned at 0.39: 0.40+ pulls `libsqlite3-sys` 0.38, whose build script needs the still-unstable `cfg_select` ([rust#115585](https://github.com/rust-lang/rust/issues/115585)) and won't build on stable.

## Verify

```bash
llmtrim --version
llmtrim --help
```

## Next: bootstrap the interceptor

The `curl | sh` installer runs this for you. If you built from source or skipped it
(`LLMTRIM_NO_SETUP=1`), run it yourself:

```bash
llmtrim setup     # CA + HTTPS_PROXY/NODE_EXTRA_CA_CERTS in your shell profile + autostart + start
llmtrim status    # savings dashboard (add --watch for a live view)
```

llmtrim is purely a MITM proxy - it configures your **environment** (no IDE settings).
See [the README](README.md#how-it-reaches-your-tools) for how it reaches your tools.

## Update

One command, channel-aware - it detects how llmtrim was installed and **restarts the daemon
onto the new binary** (a binary swap alone leaves the old version running):

```bash
llmtrim update
```

- **Binary** (`curl | sh`): re-runs the installer to fetch the latest release, then restarts.
- **Cargo / Homebrew**: prints the right command (`cargo install --git … --force` /
  `brew upgrade llmtrim`), then run `llmtrim setup` to restart the daemon on it.

`status` shows a one-line notice when a newer release exists (checked at most once a day,
cached; set `LLMTRIM_NO_UPDATE_CHECK=1` to disable, and it's skipped offline). Pin a version
in production and update promptly - security fixes land on the latest release (see SECURITY.md).

## Uninstall

One command, fully transparent - the exact inverse of `setup`:

```bash
llmtrim uninstall            # stop daemon, disable autostart, strip env block, remove CA + state + binary
llmtrim uninstall --purge    # also delete the savings ledger
llmtrim uninstall --keep-binary
```
