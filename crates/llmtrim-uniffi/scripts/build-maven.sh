#!/usr/bin/env bash
# Assemble + build the publishable Kotlin/JVM artifact for llmtrim.
#
#   1. generate the UniFFI Kotlin glue (from the unstripped build) into the Gradle
#      project's src/main/kotlin/;
#   2. build the optimized cdylib and place it under src/main/resources/<os-arch>/ so JNA
#      finds it on the classpath (the jar is then self-contained for this platform);
#   3. run `gradle <task>` (default: build) in packaging/kotlin/.
#
# A release jar bundles every platform's library by running step 2 per target before the
# final `gradle jar`. Usage: build-maven.sh [gradle-task...]   (default: build)
# Env: LLMTRIM_VERSION, GRADLE (path to a gradle binary; else expects `gradle` on PATH).
set -euo pipefail

pkg_dir="$(cd "$(dirname "$0")/../packaging/kotlin" && pwd)"
workspace_root="$(cd "$(dirname "$0")/../../.." && pwd)"
gradle_bin="${GRADLE:-gradle}"
cd "$workspace_root"

echo "==> generating UniFFI Kotlin glue (from the unstripped debug build)"
cargo build -p llmtrim-uniffi
dbg=""
for ext in so dylib dll; do [ -f "target/debug/libllmtrim_ffi.$ext" ] && { dbg="target/debug/libllmtrim_ffi.$ext"; break; }; done
[ -n "$dbg" ] || { echo "error: no unstripped cdylib in target/debug/" >&2; exit 1; }
rm -rf "$pkg_dir/src/main/kotlin"
cargo run -q --bin uniffi-bindgen -p llmtrim-uniffi -- generate --library "$dbg" --language kotlin --out-dir "$pkg_dir/src/main/kotlin"

echo "==> bundling the optimized cdylib at the JNA resource prefix"
cargo build --release -p llmtrim-uniffi
rel=""
for ext in so dylib dll; do [ -f "target/release/libllmtrim_ffi.$ext" ] && { rel="target/release/libllmtrim_ffi.$ext"; break; }; done
[ -n "$rel" ] || { echo "error: no release cdylib in target/release/" >&2; exit 1; }
# JNA's classpath resource prefix (e.g. linux-x86-64, darwin-aarch64).
os="$(uname -s)"; case "$os" in Linux) os=linux;; Darwin) os=darwin;; esac
arch="$(uname -m)"; case "$arch" in x86_64) arch=x86-64;; arm64|aarch64) arch=aarch64;; esac
prefix="$os-$arch"
mkdir -p "$pkg_dir/src/main/resources/$prefix"
cp "$rel" "$pkg_dir/src/main/resources/$prefix/libllmtrim_ffi.${rel##*.}"

echo "==> gradle ${*:-build}"
cd "$pkg_dir"
"$gradle_bin" --no-daemon "${@:-build}"
echo "==> done: $(ls -t "$pkg_dir"/build/libs/*.jar 2>/dev/null | head -1 || echo '(no jar — task may not produce one)')"
