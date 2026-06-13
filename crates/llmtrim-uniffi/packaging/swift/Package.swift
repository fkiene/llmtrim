// swift-tools-version:5.9
import PackageDescription

// SwiftPM package for the llmtrim Swift bindings.
//
// The native engine ships as a binary XCFramework (built by scripts/build-xcframework.sh
// on macOS: per-Apple-target static libs + the UniFFI-generated FFI header/modulemap), and
// the generated Swift API lives in Sources/Llmtrim/. Both the xcframework and the generated
// Swift source are build artifacts produced by the script (git-ignored).
//
// For a tagged release, swap the local binaryTarget for the remote form so consumers don't
// build anything:
//   .binaryTarget(
//       name: "llmtrimFFI",
//       url: "https://github.com/fkiene/llmtrim/releases/download/vX.Y.Z/llmtrimFFI.xcframework.zip",
//       checksum: "<swift package compute-checksum llmtrimFFI.xcframework.zip>")
let package = Package(
    name: "Llmtrim",
    platforms: [.macOS(.v11), .iOS(.v13)],
    products: [
        .library(name: "Llmtrim", targets: ["Llmtrim"]),
    ],
    targets: [
        .binaryTarget(name: "llmtrimFFI", path: "llmtrimFFI.xcframework"),
        .target(
            name: "Llmtrim",
            dependencies: ["llmtrimFFI"],
            path: "Sources/Llmtrim"
        ),
    ]
)
