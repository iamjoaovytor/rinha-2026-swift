// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "api",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "api", targets: ["api"]),
        .executable(name: "preprocess", targets: ["preprocess"]),
        .executable(name: "evaluator", targets: ["evaluator"]),
        .library(name: "Domain", targets: ["Domain"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.0"),
        .package(url: "https://github.com/iamjoaovytor/swift-nio-handoff.git", exact: "2.86.2-handoff.1")
    ],
    targets: [
        .target(
            name: "Domain",
            dependencies: ["CSearch"],
            path: "Sources/Domain"
        ),
        .target(
            name: "CSearch",
            path: "Sources/CSearch",
            sources: ["CSearch.c"],
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "api",
            dependencies: [
                "Domain",
                .product(name: "Logging", package: "swift-log"),
                .product(name: "NIOConcurrencyHelpers", package: "swift-nio-handoff"),
                .product(name: "NIOCore", package: "swift-nio-handoff"),
                .product(name: "NIOPosix", package: "swift-nio-handoff"),
                .product(name: "NIOHTTP1", package: "swift-nio-handoff")
            ],
            path: "Sources/api"
        ),
        .executableTarget(
            name: "preprocess",
            dependencies: ["Domain"],
            path: "Sources/preprocess"
        ),
        .executableTarget(
            name: "evaluator",
            dependencies: ["Domain"],
            path: "Sources/evaluator"
        ),
        .testTarget(
            name: "DomainTests",
            dependencies: ["Domain"],
            path: "Tests/DomainTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
