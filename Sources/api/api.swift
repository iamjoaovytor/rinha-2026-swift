import Foundation
import Hummingbird
import Logging
import NIOCore
import NIOPosix

@main
struct RinhaAPI {
    static func main() async throws {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = .critical
            return handler
        }

        let eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)

        let router = Router()
        router.get("/ready") { _, _ -> HTTPResponse.Status in
            .ok
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
