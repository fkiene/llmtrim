#!/usr/bin/env bash
# Build a self-contained Python wheel for llmtrim from the UniFFI bindings.
#
# Why this script instead of plain `maturin build`: maturin's `bindings = "uniffi"`
# auto-packaging is sensitive to the maturin↔uniffi version pair. With maturin 1.14 +
# uniffi 0.31 it builds the cdylib into the wheel but leaves the generated Python glue
# out (an empty package `__init__.py`). This script runs maturin for the native build,
# then injects the freshly generated bindings as the package's `__init__.py` and repacks
# the wheel so its RECORD hashes stay valid. Drop it once the auto path packages cleanly.
#
# Usage: crates/llmtrim-uniffi/scripts/build-wheel.sh [--release]
set -euo pipefail

crate_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_root="$(cd "$crate_dir/../.." && pwd)"
profile_flag="${1:-}"            # pass --release for an optimized build
dist_dir="$workspace_root/target/wheels"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

cd "$crate_dir"

echo "==> maturin build (native cdylib in wheel)"
maturin build $profile_flag -o "$dist_dir"

wheel="$(ls -t "$dist_dir"/llmtrim-*.whl | head -1)"
echo "==> base wheel: $wheel"

echo "==> generating UniFFI Python bindings (from the unstripped debug build)"
# Generate from a debug cdylib, NOT maturin's release lib: the release profile sets
# `strip = true`, which removes the UniFFI metadata symbols library-mode bindgen reads, so
# generating from it silently emits nothing. The shipped wheel still carries maturin's
# optimized library; only the glue is generated here.
cargo build -p llmtrim-uniffi
lib=""
for ext in so dylib dll; do
    [ -f "$workspace_root/target/debug/libllmtrim_ffi.$ext" ] && { lib="$workspace_root/target/debug/libllmtrim_ffi.$ext"; break; }
done
[ -n "$lib" ] || { echo "error: no unstripped cdylib in target/debug/" >&2; exit 1; }
cargo run -q --bin uniffi-bindgen -p llmtrim-uniffi -- \
    generate --library "$lib" --language python --out-dir "$work_dir/glue"

echo "==> injecting glue and repacking wheel"
python3 -m wheel unpack "$wheel" -d "$work_dir/unpacked"
pkg_dir="$(ls -d "$work_dir"/unpacked/*/llmtrim_ffi 2>/dev/null | head -1 || true)"
[ -n "$pkg_dir" ] || { echo "error: maturin wheel has no 'llmtrim_ffi' package dir — its layout changed; update this script" >&2; exit 1; }
cp "$work_dir/glue/llmtrim_ffi.py" "$pkg_dir/__init__.py"
# Friendly top-level package so `pip install llmtrim` is imported as `import llmtrim`
# (the UniFFI module itself is `llmtrim_ffi`).
root="$(dirname "$pkg_dir")"
mkdir -p "$root/llmtrim"
printf 'from llmtrim_ffi import compress, Provider, CompressOutput, LlmtrimError  # noqa: F401\n' > "$root/llmtrim/__init__.py"
python3 -m wheel pack "$root" -d "$dist_dir"

echo "==> done: $(ls -t "$dist_dir"/llmtrim-*.whl | head -1)"
