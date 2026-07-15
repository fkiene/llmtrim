# Installing llmtrim

Three verbs cover the whole lifecycle:

| You want… | Run |
|---|---|
| **First install** | install the binary, then `llmtrim setup` |
| **New version** | `llmtrim update` (then `llmtrim ensure` on package-manager channels) |
| **Something feels off** | `llmtrim ensure` · or `llmtrim doctor --fix` |

Most people only need **Get the binary** + **Bootstrap** below.

---

## Get the binary

### npm (recommended)

```bash
npm install -g @llmtrim/cli@latest && llmtrim setup
```

Prebuilt binary for your platform — no Rust. Prefer the global install over `npx` for
`setup`: the daemon and autostart need a path that survives `npm cache clean`.

### curl (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh
```

Installs into `~/.local/bin` and runs `setup` for you. Overrides:

```bash
LLMTRIM_INSTALL_DIR=/usr/local/bin LLMTRIM_VERSION=v0.10.2 \
  curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh
```

If `~/.local/bin` is not on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc or ~/.zshrc
```

### PowerShell (Windows)

```powershell
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Binary lands in `%LOCALAPPDATA%\llmtrim\bin` (user `PATH` updated). Overrides:

```powershell
$env:LLMTRIM_VERSION = "v0.10.2"   # pin a release
$env:LLMTRIM_NO_SETUP = "1"        # binary only; run setup yourself later
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Open a new PowerShell window so `PATH` and env apply. x64 and ARM64 ship; WSL uses the Linux line.

### Other channels

```bash
brew install fkiene/tap/llmtrim
cargo binstall llmtrim                 # or: cargo install --locked llmtrim
scoop bucket add llmtrim https://github.com/fkiene/scoop-bucket && scoop install llmtrim
```

Homebrew from this repo: `brew install --build-from-source ./Formula/llmtrim.rb`.

### Docker

```bash
docker run -d --name llmtrim -p 43117:43117 -v llmtrim-state:/data ghcr.io/fkiene/llmtrim
docker run --rm -v llmtrim-state:/data ghcr.io/fkiene/llmtrim ca --pem > ~/.llmtrim-ca.pem
export HTTPS_PROXY=http://localhost:43117 NODE_EXTRA_CA_CERTS=~/.llmtrim-ca.pem
```

Image binds `0.0.0.0` inside the container (`LLMTRIM_BIND` to change). Put the two
`export`s in the job or shell that should route through the proxy.

### From source

```bash
git clone https://github.com/fkiene/llmtrim
cd llmtrim
cargo build --release                  # target/release/llmtrim
cargo install --path . --locked
```

Rust 1.88+ (edition 2024). `rusqlite` is bundled (no system SQLite) and pinned at 0.39:
0.40+ pulls `libsqlite3-sys` 0.38, which needs unstable `cfg_select`
([rust#115585](https://github.com/rust-lang/rust/issues/115585)) and will not build on stable.

---

## Bootstrap (after the binary is on PATH)

The curl / npm installers run this for you. After Cargo, Homebrew, Scoop, source, or
`LLMTRIM_NO_SETUP=1`, run it yourself:

```bash
llmtrim setup      # CA + shell env + autostart + Claude Code integrations + daemon
llmtrim status     # live savings dashboard
```

Open a **new** terminal so tools inherit `HTTPS_PROXY`. Re-running `setup` is safe
(idempotent).

When Claude Code is present (`~/.claude`), `setup` also enables:

- status line  
- cold-cache guard  
- window-local `/sub`  
- cheaper `/compact` model chain  

No separate install checklist. Later upgrades refresh these via `update` / `ensure`.

llmtrim is a local MITM proxy (plus optional Claude Code hooks).  
`llmtrim uninstall` reverses it. How traffic reaches tools: [README](README.md#what-it-does).

### Verify

```bash
llmtrim --version
llmtrim --help
llmtrim doctor          # end-to-end; doctor --fix applies repairs
```

---

## Update

One command when possible — new binary, daemon restart, integration refresh. No
per-feature reinstall notes.

```bash
llmtrim update
```

| Channel | What happens |
|---|---|
| **Binary** (`curl \| sh`) | Re-runs the installer, restarts the daemon, runs `ensure` |
| **npm / Homebrew / Cargo** | Prints the package command; then run `llmtrim ensure` (or press **`f`** in `status`) |

`status` shows a one-line notice when a newer release exists (cached ≤ once/day;
`LLMTRIM_NO_UPDATE_CHECK=1` to disable; skipped offline). Pin versions in production;
security fixes land on the latest release ([SECURITY.md](SECURITY.md)).

Incomplete after an upgrade?

```bash
llmtrim ensure
llmtrim doctor --fix
```

---

## Uninstall

Exact inverse of `setup`:

```bash
llmtrim uninstall              # stop daemon, disable autostart, strip env, remove CA + state + binary
llmtrim uninstall --purge      # also delete the savings ledger
llmtrim uninstall --keep-binary
```

**Package managers:** run `llmtrim uninstall` **first**. Removing only the package leaves
`HTTPS_PROXY` pointing at a dead proxy. `uninstall` detects the channel and prints the
follow-up (`npm uninstall -g @llmtrim/cli` / `cargo uninstall llmtrim` / `brew uninstall llmtrim`).
