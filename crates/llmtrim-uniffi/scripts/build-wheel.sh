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

echo "==> generating UniFFI Python bindings"
# Locate the cdylib maturin built (.so Linux / .dylib macOS / .dll Windows); never the .a.
# maturin 1.14 stages it under target/maturin/; cover the plain and per-target
# (cross-compile / explicit --target) profile dirs too.
lib=""
for cand in "$workspace_root"/target/maturin/libllmtrim_ffi.{so,dylib,dll} \
            "$workspace_root"/target/{release,debug}/libllmtrim_ffi.{so,dylib,dll} \
            "$workspace_root"/target/*/{release,debug}/libllmtrim_ffi.{so,dylib,dll}; do
    [ -f "$cand" ] && { lib="$cand"; break; }
done
[ -n "$lib" ] || { echo "error: could not locate the built libllmtrim_ffi cdylib" >&2; exit 1; }
cargo run -q --bin uniffi-bindgen -p llmtrim-uniffi -- \
    generate --library "$lib" --language python --out-dir "$work_dir/glue"

echo "==> injecting glue and repacking wheel"
python3 -m wheel unpack "$wheel" -d "$work_dir/unpacked"
pkg_dir="$(ls -d "$work_dir"/unpacked/*/llmtrim_ffi 2>/dev/null | head -1 || true)"
[ -n "$pkg_dir" ] || { echo "error: maturin wheel has no 'llmtrim_ffi' package dir — its layout changed; update this script" >&2; exit 1; }
cp "$work_dir/glue/llmtrim_ffi.py" "$pkg_dir/__init__.py"
python3 -m wheel pack "$(dirname "$pkg_dir")" -d "$dist_dir"

echo "==> done: $(ls -t "$dist_dir"/llmtrim-*.whl | head -1)"
