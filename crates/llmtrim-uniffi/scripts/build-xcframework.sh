#!/usr/bin/env bash
# Build the llmtrim Swift package's XCFramework. macOS + Xcode only.
#
# Steps:
#   1. generate the Swift API + FFI header/modulemap from the (unstripped) debug cdylib —
#      uniffi-bindgen reads metadata from a loadable library, not a static .a;
#   2. build a release static lib (libllmtrim_ffi.a) for each Apple target;
#   3. lipo the same-platform arch slices (macOS universal; iOS-sim universal);
#   4. assemble llmtrimFFI.xcframework from the slices + a Headers dir whose modulemap is
#      named `module.modulemap` (what XCFrameworks expect);
#   5. drop the generated Swift into Sources/Llmtrim/.
#
# Then `swift build` (or add the remote binaryTarget for release). Usage: build-xcframework.sh
set -euo pipefail

pkg_dir="$(cd "$(dirname "$0")/../packaging/swift" && pwd)"
workspace_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$workspace_root"

[ "$(uname -s)" = "Darwin" ] || { echo "error: build-xcframework.sh requires macOS + Xcode" >&2; exit 1; }

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
hdrs="$work/Headers"; mkdir -p "$hdrs"

echo "==> generating Swift API + FFI header/modulemap"
cargo build -p llmtrim-uniffi
dylib="target/debug/libllmtrim_ffi.dylib"
[ -f "$dylib" ] || { echo "error: no debug cdylib at $dylib" >&2; exit 1; }
gen="$work/gen"; mkdir -p "$gen"
cargo run -q --bin uniffi-bindgen -p llmtrim-uniffi -- generate --library "$dylib" --language swift --out-dir "$gen"
cp "$gen/llmtrim_ffiFFI.h" "$hdrs/"
# XCFrameworks expect the umbrella modulemap to be named module.modulemap.
cp "$gen/llmtrim_ffiFFI.modulemap" "$hdrs/module.modulemap"
mkdir -p "$pkg_dir/Sources/Llmtrim"
cp "$gen/llmtrim_ffi.swift" "$pkg_dir/Sources/Llmtrim/llmtrim_ffi.swift"

echo "==> building static libs per Apple target"
# Note: x86_64 iOS is only ever the simulator, so its target is `x86_64-apple-ios` (no
# `-sim` suffix — that exists only for aarch64, to split arm64 device vs arm64 sim).
#
# XCFRAMEWORK_FAST=1 builds only the slices a host `swift test` actually links against
# (Apple-silicon macOS + iOS-sim arm64), skipping the x86_64 and iOS-device cross-compiles.
# CI PR runs set it to cut wall-clock; the release build leaves it unset for the full
# universal/device xcframework.
if [ "${XCFRAMEWORK_FAST:-0}" = "1" ]; then
    # The host `swift test` runs on Apple-silicon macOS and links only the macOS arm64
    # slice, so build just that one target — the iOS/sim slices are never exercised here.
    echo "    (fast mode: Apple-silicon macOS only)"
    targets="aarch64-apple-darwin"
else
    targets="aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios"
fi
for t in $targets; do
    rustup target add "$t" >/dev/null 2>&1 || true
    cargo build --release -p llmtrim-uniffi --target "$t"
done

lib() { echo "target/$1/release/libllmtrim_ffi.a"; }
mkdir -p "$work/macos"
xcargs=()
if [ "${XCFRAMEWORK_FAST:-0}" = "1" ]; then
    cp "$(lib aarch64-apple-darwin)" "$work/macos/libllmtrim_ffi.a"
    xcargs+=(-library "$work/macos/libllmtrim_ffi.a" -headers "$hdrs")
else
    mkdir -p "$work/ios" "$work/iossim"
    lipo -create "$(lib aarch64-apple-darwin)" "$(lib x86_64-apple-darwin)" -output "$work/macos/libllmtrim_ffi.a"
    cp "$(lib aarch64-apple-ios)" "$work/ios/libllmtrim_ffi.a"
    lipo -create "$(lib aarch64-apple-ios-sim)" "$(lib x86_64-apple-ios)" -output "$work/iossim/libllmtrim_ffi.a"
    xcargs+=(-library "$work/macos/libllmtrim_ffi.a" -headers "$hdrs")
    xcargs+=(-library "$work/ios/libllmtrim_ffi.a" -headers "$hdrs")
    xcargs+=(-library "$work/iossim/libllmtrim_ffi.a" -headers "$hdrs")
fi

echo "==> assembling llmtrimFFI.xcframework"
rm -rf "$pkg_dir/llmtrimFFI.xcframework"
xcodebuild -create-xcframework "${xcargs[@]}" -output "$pkg_dir/llmtrimFFI.xcframework"
echo "==> done: $pkg_dir/llmtrimFFI.xcframework"
