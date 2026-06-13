#!/usr/bin/env bash
# Generate UniFFI foreign-language bindings for llmtrim from one Rust definition.
#
# Emits glue for Python, Ruby, Swift and Kotlin into <out-dir>/<lang>/. The generated
# files are build artifacts (their internal checksums are pinned to the compiled library's
# ABI), so they are NOT committed — regenerate them per release alongside the native lib.
#
# IMPORTANT: library-mode bindgen reads UniFFI metadata symbols from the cdylib, so it must
# run against an UNSTRIPPED build. The workspace release profile sets `strip = true`, which
# removes those symbols — hence this script generates from the (unstripped) debug build. The
# native library you actually ship can be a stripped `--release` cdylib of the same crate;
# the generated glue loads it by name (`libllmtrim_ffi`) regardless of optimization level.
#
# Usage: crates/llmtrim-uniffi/scripts/generate-bindings.sh [out-dir]   (default: ./bindings)
set -euo pipefail

workspace_root="$(cd "$(dirname "$0")/../../.." && pwd)"
out_dir="${1:-$(pwd)/bindings}"
cd "$workspace_root"

echo "==> building unstripped cdylib (for metadata)"
cargo build -p llmtrim-uniffi
lib="$workspace_root/target/debug/libllmtrim_ffi.so"
[ -f "$lib" ] || lib="$(ls "$workspace_root"/target/debug/libllmtrim_ffi.* | head -1)"

for lang in python ruby swift kotlin; do
    echo "==> generating $lang -> $out_dir/$lang"
    cargo run -q --bin uniffi-bindgen -p llmtrim-uniffi -- \
        generate --library "$lib" --language "$lang" --out-dir "$out_dir/$lang"
done

echo "==> done. Native lib for distribution: cargo build --release -p llmtrim-uniffi"
