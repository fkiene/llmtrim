#!/bin/sh
# Build the npm packages for one release: a meta package `@llmtrim/cli` (bin shim +
# optionalDependencies) and one `@llmtrim/<os>-<arch>` package per platform carrying the
# prebuilt binary (esbuild pattern — npm installs only the matching platform package).
# The unscoped name `llmtrim` belongs to an unrelated 2025 package; the org scope is ours.
#
# Usage: build_npm_packages.sh vX.Y.Z <assets-dir> <out-dir>
#   assets-dir: extracted release archives, one subdir per target containing llmtrim[.exe]
# Publishing (CI): for d in out/*; do npm publish "$d" --access public; done
set -eu

VERSION="${1#v}"
ASSETS="$2"
OUT="$3"
REPO="https://github.com/fkiene/llmtrim"

# target triple : npm os : npm cpu
TARGETS="
x86_64-unknown-linux-musl:linux:x64
aarch64-unknown-linux-gnu:linux:arm64
x86_64-apple-darwin:darwin:x64
aarch64-apple-darwin:darwin:arm64
x86_64-pc-windows-msvc:win32:x64
aarch64-pc-windows-msvc:win32:arm64
"

mkdir -p "$OUT"
OPTIONAL_DEPS=""

for entry in $TARGETS; do
    target="${entry%%:*}"; rest="${entry#*:}"
    os="${rest%%:*}"; cpu="${rest#*:}"
    pkg="@llmtrim/${os}-${cpu}"
    bin="llmtrim"; [ "$os" = "win32" ] && bin="llmtrim.exe"
    src="$ASSETS/$target/$bin"
    [ -f "$src" ] || { echo "missing binary: $src" >&2; exit 1; }

    dir="$OUT/${os}-${cpu}"
    mkdir -p "$dir/bin"
    cp "$src" "$dir/bin/$bin"
    chmod +x "$dir/bin/$bin"
    printf '# %s\n\nPrebuilt llmtrim binary for %s. Install [@llmtrim/cli](https://www.npmjs.com/package/@llmtrim/cli) instead of this package directly.\n' "$pkg" "$os-$cpu" > "$dir/README.md"
    cat > "$dir/package.json" <<EOF
{
  "name": "$pkg",
  "version": "$VERSION",
  "description": "llmtrim prebuilt binary for $os-$cpu",
  "repository": "$REPO",
  "license": "MPL-2.0",
  "os": ["$os"],
  "cpu": ["$cpu"],
  "files": ["bin"],
  "readme": "README.md"
}
EOF
    OPTIONAL_DEPS="$OPTIONAL_DEPS    \"$pkg\": \"$VERSION\",\n"
done

# Meta package: a tiny Node shim that exec's the platform package's binary.
META="$OUT/cli"
mkdir -p "$META/bin"
cat > "$META/bin/llmtrim.js" <<'EOF'
#!/usr/bin/env node
// Resolve the platform package's prebuilt binary and exec it with our args.
const { spawnSync } = require("child_process");
const path = require("path");
const pkg = `@llmtrim/${process.platform}-${process.arch}`;
const bin = process.platform === "win32" ? "llmtrim.exe" : "llmtrim";
let exe;
try {
  exe = path.join(path.dirname(require.resolve(`${pkg}/package.json`)), "bin", bin);
} catch {
  console.error(`llmtrim: no prebuilt binary for ${process.platform}-${process.arch}.`);
  console.error("Install alternatives: https://github.com/fkiene/llmtrim#install");
  process.exit(1);
}
const r = spawnSync(exe, process.argv.slice(2), { stdio: "inherit" });
process.exit(r.status === null ? 1 : r.status);
EOF
cp "$(dirname "$0")/README.md" "$META/README.md"

DEPS=$(printf "$OPTIONAL_DEPS" | sed '$s/,$//')
cat > "$META/package.json" <<EOF
{
  "name": "@llmtrim/cli",
  "version": "$VERSION",
  "description": "Cut your LLM bill: drop-in proxy that compresses input, output, and cache. Any provider, answers unchanged.",
  "mcpName": "io.github.fkiene/llmtrim",
  "repository": "$REPO",
  "homepage": "$REPO#readme",
  "license": "MPL-2.0",
  "keywords": ["llm", "tokens", "compression", "proxy", "mcp", "openai", "anthropic", "claude"],
  "bin": { "llmtrim": "bin/llmtrim.js" },
  "files": ["bin", "README.md"],
  "optionalDependencies": {
$DEPS
  }
}
EOF

echo "built $(ls "$OUT" | wc -l) packages in $OUT"
