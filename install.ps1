# llmtrim — Windows installer (PowerShell 5.1+)
#
#   irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
#
# Builds from source with cargo, then wires the interceptor with `llmtrim setup`
# (CA into the Windows trust flow, HTTPS_PROXY/NODE_EXTRA_CA_CERTS into your PowerShell
# profile, autostart, and the background daemon). WSL users: use install.sh instead.
#
# (A prebuilt-binary path lands with tagged releases; until then this builds from source,
#  which needs the Rust toolchain.)

$ErrorActionPreference = "Stop"

Write-Host "Installing llmtrim..."

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "cargo not found. Install Rust from https://rustup.rs, reopen PowerShell, then re-run."
    exit 1
}

cargo install --git https://github.com/fkiene/llmtrim
if ($LASTEXITCODE -ne 0) { Write-Error "cargo install failed."; exit 1 }

# `cargo install` drops binaries in %USERPROFILE%\.cargo\bin (normally already on PATH).
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
$llmtrim = Join-Path $cargoBin "llmtrim.exe"
if ($env:PATH -notlike "*$cargoBin*") {
    Write-Host "Note: add $cargoBin to your PATH (cargo's bin directory)."
}

Write-Host "Running setup (CA + HTTPS_PROXY in your PowerShell profile + autostart + start)..."
& $llmtrim setup

Write-Host ""
Write-Host "Done. Open a new PowerShell window so the profile env applies."
Write-Host "Watch savings:  llmtrim status"
Write-Host "Back out:       llmtrim uninstall"
