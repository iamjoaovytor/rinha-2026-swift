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
        .library(name: "Domain", targets: ["Domain"])
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.22.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.0"),
        .package(url: "https://github.com/apple/swift-nio.git", "2.65.0"..<"2.87.0")
    ],
    targets: [
        .target(
            name: "Domain",
            dependencies: ["CSearch"]
        ),
        .target(
            name: "CSearch",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "api",
            dependencies: [
                "Domain",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "NIOConcurrencyHelpers", package: "swift-nio"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio")
            ]
        ),
        .executableTarget(
            name: "preprocess",
            dependencies: ["Domain"]
        ),
        .testTarget(
            name: "DomainTests",
            dependencies: ["Domain"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
