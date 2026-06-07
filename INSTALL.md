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

Requires Rust 1.85+ (edition 2024). `rusqlite` is bundled (no system SQLite needed).

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
llmtrim monitor   # savings dashboard (add --watch for a live view)
```

llmtrim is purely a MITM proxy — it configures your **environment** (no IDE settings).
See [the README](README.md#how-it-reaches-your-tools) for how it reaches your tools.

## Uninstall

One command, fully transparent — the exact inverse of `setup`:

```bash
llmtrim uninstall            # stop daemon, disable autostart, strip env block, remove CA + state + binary
llmtrim uninstall --purge    # also delete the savings ledger
llmtrim uninstall --keep-binary
```
