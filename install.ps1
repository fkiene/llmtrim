# llmtrim — Windows installer (PowerShell 5.1+)
#
#   irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
#
# Downloads the prebuilt binary for the latest release and wires the interceptor with
# `llmtrim setup` (CA into the Windows trust flow, HTTPS_PROXY/NODE_EXTRA_CA_CERTS into
# your PowerShell profile, autostart, and the background daemon). Falls back to a
# from-source `cargo install` when no prebuilt binary exists for your architecture
# (e.g. ARM64). WSL users: use install.sh instead.
#
# Override:
#   $env:LLMTRIM_VERSION  = "v0.1.0"   pin a specific release
#   $env:LLMTRIM_NO_SETUP = "1"        install the binary only, skip setup

$ErrorActionPreference = "Stop"
$Repo = "fkiene/llmtrim"
$BinDir = Join-Path $env:LOCALAPPDATA "llmtrim\bin"

function Info($m) { Write-Host "[INFO] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }

# Map the OS architecture to a release target triple. PROCESSOR_ARCHITEW6432 holds the
# real machine arch when a 32-bit PowerShell runs under WOW64; prefer it when set.
# Unknown arches (e.g. 32-bit x86) return $null and take the cargo fallback.
function Get-Target {
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    switch ($arch) {
        "AMD64" { return "x86_64-pc-windows-msvc" }
        "ARM64" { return "aarch64-pc-windows-msvc" }
        default { return $null }
    }
}

# Latest version via the releases/latest redirect, with the GitHub API as a fallback.
# Reads the resolved URI cross-edition (PS 5.1: ResponseUri; PS 7+: RequestMessage).
function Get-LatestVersion {
    try {
        $resp = Invoke-WebRequest "https://github.com/$Repo/releases/latest" -UseBasicParsing
        $uri = if ($resp.BaseResponse.ResponseUri) { $resp.BaseResponse.ResponseUri.AbsoluteUri }
               elseif ($resp.BaseResponse.RequestMessage) { $resp.BaseResponse.RequestMessage.RequestUri.AbsoluteUri }
               else { "" }
        if ($uri -match '/tag/([^/]+)$') { return $Matches[1] }
    } catch { }
    Warn "Redirect lookup failed, falling back to GitHub API..."
    return (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
}

# Build from source when no prebuilt binary matches the architecture.
function Install-FromSource {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Error "No prebuilt binary for this architecture and cargo not found. Install Rust from https://rustup.rs, reopen PowerShell, then re-run."
        exit 1
    }
    Info "Building from source with cargo..."
    cargo install --git "https://github.com/$Repo"
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo install failed."; exit 1 }
    return (Join-Path $env:USERPROFILE ".cargo\bin\llmtrim.exe")
}

# Download + extract the prebuilt zip, placing llmtrim.exe in $BinDir on the user PATH.
# Returns $null (so the caller falls back to source) when no asset exists for the tag.
function Install-Prebuilt($target) {
    $version = if ($env:LLMTRIM_VERSION) { $env:LLMTRIM_VERSION } else { Get-LatestVersion }
    if (-not $version) { Write-Error "Failed to resolve latest version (set LLMTRIM_VERSION=vX.Y.Z to pin)."; exit 1 }

    $asset = "llmtrim-$target.zip"
    $url = "https://github.com/$Repo/releases/download/$version/$asset"
    Info "Detected: windows $target, version $version"
    Info "Downloading $url"

    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("llmtrim-" + [System.IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    $zip = Join-Path $tmp $asset
    try {
        Invoke-WebRequest $url -OutFile $zip -UseBasicParsing
    } catch {
        Warn "No prebuilt asset at $url"
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
        return $null
    }

    # Verify SHA-256 checksum against the .sha256 sidecar uploaded by CI.
    $sha256Url = "$url.sha256"
    try {
        $expectedLine = (Invoke-WebRequest $sha256Url -UseBasicParsing).Content.Trim()
        $expectedHash = ($expectedLine -split '\s+')[0].ToUpper()
        $actualHash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToUpper()
        if ($actualHash -ne $expectedHash) {
            Write-Error "Checksum mismatch! Expected $expectedHash, got $actualHash. Aborting."
            Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
            exit 1
        }
        Info "Checksum verified."
    } catch {
        Write-Error "Failed to fetch or verify checksum from $sha256Url"
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
        exit 1
    }

    # Reject absolute or parent-traversal paths before extracting (CWE-22).
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
    try {
        foreach ($entry in $archive.Entries) {
            if ($entry.FullName -match '(^|[\\/])\.\.([\\/]|$)' -or [System.IO.Path]::IsPathRooted($entry.FullName)) {
                Write-Error "Archive contains unsafe paths — refusing to extract."
                exit 1
            }
        }
    } finally {
        $archive.Dispose()
    }

    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $exe = Get-ChildItem -Path $tmp -Recurse -Filter "llmtrim.exe" | Select-Object -First 1
    if (-not $exe) { Write-Error "llmtrim.exe not found in archive."; exit 1 }

    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    Copy-Item $exe.FullName (Join-Path $BinDir "llmtrim.exe") -Force
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    Info "Installed llmtrim to $BinDir\llmtrim.exe"

    # Persist $BinDir on the user PATH (and this session) if it isn't already there.
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$BinDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$BinDir", "User")
        $env:PATH = "$env:PATH;$BinDir"
        Info "Added $BinDir to your user PATH."
    }
    return (Join-Path $BinDir "llmtrim.exe")
}

Write-Host "Installing llmtrim..."

$target = Get-Target
$llmtrim = $null
if ($target) { $llmtrim = Install-Prebuilt $target }
if (-not $llmtrim) { $llmtrim = Install-FromSource }

if ($env:LLMTRIM_NO_SETUP -eq "1") {
    Write-Host ""
    Write-Host "Binary installed. Skipped setup (LLMTRIM_NO_SETUP=1)."
    Write-Host "Wire it later with:  llmtrim setup"
    exit 0
}

Write-Host "Running setup (CA + HTTPS_PROXY in your PowerShell profile + autostart + start)..."
& $llmtrim setup

Write-Host ""
Write-Host "Done. Open a new PowerShell window so the profile env applies."
Write-Host "Watch savings:  llmtrim status"
Write-Host "Back out:       llmtrim uninstall"
