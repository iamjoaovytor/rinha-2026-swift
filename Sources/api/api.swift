import Domain
import Foundation
import Hummingbird
import Logging
import NIOCore
import NIOPosix

actor LoaderState {
    private var index: ReferencesIndex?
    private var failure: String?

    var isReady: Bool { index != nil }
    var lastError: String? { failure }

    func install(_ index: ReferencesIndex) {
        self.index = index
    }

    func recordFailure(_ message: String) {
        self.failure = message
    }
}

@main
struct RinhaAPI {
    static let referencesPathDefault = "/app/resources/references.bin"

    static func main() async throws {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = .critical
            return handler
        }

        let eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let state = LoaderState()
        let referencesPath = ProcessInfo.processInfo.environment["REFERENCES_BIN"]
            ?? referencesPathDefault

        Task.detached {
            do {
                let index = try ReferencesIndex.load(path: referencesPath)
                index.warm()
                await state.install(index)
                FileHandle.standardError.write(Data(
                    "loader: ready (count=\(index.header.count), scale=\(index.header.scale))\n".utf8
                ))
            } catch {
                let message = "\(error)"
                await state.recordFailure(message)
                FileHandle.standardError.write(Data("loader: \(message)\n".utf8))
            }
        }

        let router = Router()
        router.get("/ready") { _, _ -> HTTPResponse.Status in
            await state.isReady ? .ok : .serviceUnavailable
        }

        router.post("/fraud-score") { _, _ -> Response in
            // Temporary safe fallback until vectorization and KNN are implemented.
            jsonResponse(body: #"{"approved":false,"fraud_score":1.0}"#)
        }

        var logger = Logger(label: "rinha-api")
        logger.logLevel = .critical

        let configuration = ApplicationConfiguration(
            address: bindAddress(),
            serverName: nil,
            backlog: 4096
        )

        let app = Application(
            router: router,
            configuration: configuration,
            eventLoopGroupProvider: .shared(eventLoopGroup),
            logger: logger
        )

        do {
            try await app.runService()
            try await eventLoopGroup.shutdownGracefully()
        } catch {
            try? await eventLoopGroup.shutdownGracefully()
            throw error
        }
    }

    private static func bindAddress() -> BindAddress {
        let environment = ProcessInfo.processInfo.environment
        if let socketPath = environment["SOCKET_PATH"], !socketPath.isEmpty {
            try? FileManager.default.removeItem(atPath: socketPath)
            return .unixDomainSocket(path: socketPath)
        }

        let port = environment["PORT"].flatMap(Int.init) ?? 9999
        return .hostname("0.0.0.0", port: port)
    }

    private static func jsonResponse(body: String) -> Response {
        var buffer = ByteBufferAllocator().buffer(capacity: body.utf8.count)
        buffer.writeString(body)

        return Response(
            status: .ok,
            headers: [
                .contentType: "application/json",
                .contentLength: "\(buffer.readableBytes)"
            ],
            body: .init(byteBuffer: buffer)
        )
    }
}
