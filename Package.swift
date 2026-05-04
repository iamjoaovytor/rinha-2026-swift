// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "api",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "api", targets: ["api"])
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.22.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.0"),
        .package(url: "https://github.com/apple/swift-nio.git", "2.65.0"..<"2.87.0")
    ],
    targets: [
        .executableTarget(
            name: "api",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .testTarget(
            name: "apiTests",
            dependencies: ["api"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
