import Domain
import Foundation
import Hummingbird
import Logging
import NIOConcurrencyHelpers
import NIOCore
import NIOPosix

struct LoadedState: Sendable {
    let index: ReferencesIndex
    let vectorizer: Vectorizer
}

final class LoaderState: @unchecked Sendable {
    private struct State {
        var loaded: LoadedState?
        var failure: String?
    }

    private let state = NIOLockedValueBox(State())

    var current: LoadedState? {
        self.state.withLockedValue { $0.loaded }
    }

    var isReady: Bool {
        self.state.withLockedValue { $0.loaded != nil }
    }

    var lastError: String? {
        self.state.withLockedValue { $0.failure }
    }

    func install(_ loaded: LoadedState) {
        self.state.withLockedValue { state in
            state.loaded = loaded
            state.failure = nil
        }
    }

    func recordFailure(_ message: String) {
        self.state.withLockedValue { state in
            state.failure = message
            state.loaded = nil
        }
    }
}

@main
struct RinhaAPI {
    static let referencesPathDefault = "/app/resources/references.bin"
    static let mccRiskPathDefault = "/app/resources/mcc_risk.json"
    static let fallbackBody = #"{"approved":false,"fraud_score":1.0}"#
    static let decoder = JSONDecoder()

    static func main() async throws {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = .critical
            return handler
        }

        let eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let state = LoaderState()
        let env = ProcessInfo.processInfo.environment
        let referencesPath = env["REFERENCES_BIN"] ?? referencesPathDefault
        let mccRiskPath = env["MCC_RISK_JSON"] ?? mccRiskPathDefault

        Task.detached {
            do {
                let mccRisk = try MccRiskTable.load(path: mccRiskPath)
                let index = try ReferencesIndex.load(path: referencesPath)
                index.warm()
                let loaded = LoadedState(
                    index: index,
                    vectorizer: Vectorizer(mccRisk: mccRisk)
                )
                state.install(loaded)
                FileHandle.standardError.write(Data(
                    "loader: ready (count=\(index.header.count), scale=\(index.header.scale))\n".utf8
                ))
            } catch {
                let message = "\(error)"
                state.recordFailure(message)
                FileHandle.standardError.write(Data("loader: \(message)\n".utf8))
            }
        }

        let router = Router()
        router.get("/ready") { _, _ -> HTTPResponse.Status in
            state.isReady ? .ok : .serviceUnavailable
        }

        router.post("/fraud-score") { request, context -> Response in
            guard let loaded = state.current else {
                return jsonResponse(body: fallbackBody)
            }
            do {
                let bodyBuffer = try await request.body.collect(upTo: 64 * 1024)
                let quantized: [Int16]
                do {
                    quantized = try bodyBuffer.withUnsafeReadableBytes { rawBuffer in
                        try FastRequestParser.quantizedQuery(
                            from: rawBuffer,
                            vectorizer: loaded.vectorizer
                        )
                    }
                } catch {
                    let body = Data(bodyBuffer.readableBytesView)
                    let fraudRequest = try decoder.decode(FraudRequest.self, from: body)
                    let raw = try loaded.vectorizer.vectorize(fraudRequest)
                    quantized = loaded.vectorizer.quantize(raw)
                }
                let fraudVotes = KNN.fraudVoteCount(query: quantized, in: loaded.index, k: 5)
                return jsonResponse(body: FraudScoring.responseBody(fraudVoteCount: fraudVotes))
            } catch {
                context.logger.debug("fraud-score: \(error)")
                return jsonResponse(body: fallbackBody)
            }
        }

        var logger = Logger(label: "rinha-api")
        logger.logLevel = .critical

        let configuration = ApplicationConfiguration(
            address: bindAddress(),
            serverName: nil,
            backlog: 16384
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
        return makeResponse(buffer: buffer)
    }

    private static func jsonResponse(body: Data) -> Response {
        var buffer = ByteBufferAllocator().buffer(capacity: body.count)
        buffer.writeBytes(body)
        return makeResponse(buffer: buffer)
    }

    private static func makeResponse(buffer: ByteBuffer) -> Response {
        Response(
            status: .ok,
            headers: [
                .contentType: "application/json",
                .contentLength: "\(buffer.readableBytes)"
            ],
            body: .init(byteBuffer: buffer)
        )
    }
}
