#!/usr/bin/env sh
# llmtrim installer
# Usage: curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh
#
# Override:
#   LLMTRIM_INSTALL_DIR=/usr/local/bin   install location (default: ~/.local/bin)
#   LLMTRIM_VERSION=v0.1.0               pin a specific release

set -e

REPO="fkiene/llmtrim"
BINARY_NAME="llmtrim"
INSTALL_DIR="${LLMTRIM_INSTALL_DIR:-$HOME/.local/bin}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; exit 1; }

detect_os() {
    case "$(uname -s)" in
        Linux*)  OS="linux";;
        Darwin*) OS="darwin";;
        *)       error "Unsupported operating system: $(uname -s)";;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  ARCH="x86_64";;
        arm64|aarch64) ARCH="aarch64";;
        *)             error "Unsupported architecture: $(uname -m)";;
    esac
}

# Latest version via the releases/latest redirect (no API call, no rate limit).
get_latest_version() {
    VERSION=$(curl -sI "https://github.com/${REPO}/releases/latest" \
        | grep -i '^location:' \
        | sed -E 's|.*/tag/([^[:space:]]+).*|\1|' \
        | tr -d '\r')
    if [ -z "$VERSION" ]; then
        warn "Redirect lookup failed, falling back to GitHub API..."
        VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
    fi
    [ -n "$VERSION" ] || error "Failed to resolve latest version (set LLMTRIM_VERSION=vX.Y.Z to pin)"
}

get_target() {
    case "$OS" in
        linux)
            case "$ARCH" in
                x86_64)  TARGET="x86_64-unknown-linux-musl";;
                aarch64) TARGET="aarch64-unknown-linux-gnu";;
            esac;;
        darwin) TARGET="${ARCH}-apple-darwin";;
    esac
}

install() {
    info "Detected: $OS $ARCH ($TARGET), version $VERSION"
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY_NAME}-${TARGET}.tar.gz"
    TEMP_DIR=$(mktemp -d)
    ARCHIVE="${TEMP_DIR}/${BINARY_NAME}.tar.gz"

    info "Downloading $DOWNLOAD_URL"
    curl -fsSL "$DOWNLOAD_URL" -o "$ARCHIVE" || error "Download failed"

    # Reject absolute paths or '..' components before extracting (CWE-22).
    if tar -tzf "$ARCHIVE" | grep -qE '^/|(^|/)\.\.(/|$)'; then
        error "Archive contains unsafe paths — refusing to extract"
    fi

    tar -xzf "$ARCHIVE" -C "$TEMP_DIR"
    mkdir -p "$INSTALL_DIR"
    mv "${TEMP_DIR}/${BINARY_NAME}" "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
    rm -rf "$TEMP_DIR"
    info "Installed ${BINARY_NAME} to ${INSTALL_DIR}/${BINARY_NAME}"
}

verify() {
    if command -v "$BINARY_NAME" >/dev/null 2>&1; then
        info "Verification: $($BINARY_NAME --version)"
    else
        warn "Binary installed but not on PATH. Add to your shell profile:"
        warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
}

main() {
    info "Installing $BINARY_NAME..."
    detect_os
    detect_arch
    get_target
    if [ -n "$LLMTRIM_VERSION" ]; then
        VERSION="$LLMTRIM_VERSION"
        info "Using pinned version: $VERSION"
    else
        get_latest_version
    fi
    install
    verify
    echo ""
    # One-liner: bootstrap the interceptor (CA + shell-profile env + autostart + start).
    # Skip with LLMTRIM_NO_SETUP=1 to install the binary only.
    if [ -z "$LLMTRIM_NO_SETUP" ]; then
        info "Running setup (CA + HTTPS_PROXY in your shell profile + autostart + start)..."
        "${INSTALL_DIR}/${BINARY_NAME}" setup || warn "setup did not complete; run '${BINARY_NAME} setup' manually"
    else
        info "Binary installed. Run '${BINARY_NAME} setup' to bootstrap the interceptor."
    fi
}

main
