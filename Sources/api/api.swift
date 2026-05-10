import Domain
import Foundation
import Hummingbird
import Logging
import NIOConcurrencyHelpers
import NIOCore
import NIOPosix

struct LoadedState: Sendable {
    let index: ReferencesIndex
    let ivf: IVFIndex?
    let pq: IVFPQIndex?
    let searchConfig: SearchConfig
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
        let debugStats = DebugStatsCollector(
            enabled: ProcessInfo.processInfo.environment["DEBUG_STATS"] == "1"
        )
        let env = ProcessInfo.processInfo.environment
        let referencesPath = env["REFERENCES_BIN"] ?? referencesPathDefault
        let mccRiskPath = env["MCC_RISK_JSON"] ?? mccRiskPathDefault
        let ivfPath = env["IVF_BIN"] ?? IVFIndex.defaultPath(for: referencesPath)
        let ivfpqPath = env["IVFPQ_BIN"] ?? IVFPQIndex.defaultPath(for: referencesPath)
        let nprobe = env["IVF_NPROBE"].flatMap(Int.init) ?? 4
        let initialNprobe = env["IVF_INITIAL_NPROBE"].flatMap(Int.init)
        let adaptiveMinFraudVotes = env["IVF_ADAPTIVE_MIN_VOTES"].flatMap(Int.init) ?? 2
        let adaptiveMaxFraudVotes = env["IVF_ADAPTIVE_MAX_VOTES"].flatMap(Int.init) ?? 3
        let ivfpqRerankCandidates = env["IVFPQ_RERANK_CANDIDATES"].flatMap(Int.init)
        let useBoundingBoxes = env["IVF_USE_BBOX"] == "1"

        Task.detached {
            do {
                let mccRisk = try MccRiskTable.load(path: mccRiskPath)
                let index = try ReferencesIndex.load(path: referencesPath)
                let ivf: IVFIndex?
                if FileManager.default.fileExists(atPath: ivfPath) {
                    ivf = try IVFIndex.load(path: ivfPath)
                    ivf?.warm()
                } else {
                    ivf = nil
                }
                let pq: IVFPQIndex?
                if FileManager.default.fileExists(atPath: ivfpqPath) {
                    pq = try IVFPQIndex.load(path: ivfpqPath)
                    pq?.warm()
                } else {
                    pq = nil
                }
                index.warm()
                let loaded = LoadedState(
                    index: index,
                    ivf: ivf,
                    pq: pq,
                    searchConfig: SearchConfig(
                        nprobe: nprobe,
                        initialNprobe: initialNprobe,
                        adaptiveMinFraudVotes: adaptiveMinFraudVotes,
                        adaptiveMaxFraudVotes: adaptiveMaxFraudVotes,
                        ivfpqRerankCandidates: ivfpqRerankCandidates,
                        useBoundingBoxes: useBoundingBoxes
                    ),
                    vectorizer: Vectorizer(mccRisk: mccRisk)
                )
                Self.synthesizeAndWarmKNN(loaded: loaded)
                state.install(loaded)
                let adaptiveDetails: String
                if loaded.searchConfig.adaptiveEnabled {
                    adaptiveDetails = ", initial_nprobe=\(loaded.searchConfig.initialNprobe), ambiguous_votes=\(loaded.searchConfig.adaptiveMinFraudVotes)...\(loaded.searchConfig.adaptiveMaxFraudVotes)"
                } else {
                    adaptiveDetails = ""
                }
                let pqDetails: String
                if loaded.searchConfig.ivfpqEnabled {
                    pqDetails = ", pq=on, pq_rerank=\(loaded.searchConfig.ivfpqRerankCandidates ?? 0)"
                } else {
                    pqDetails = ", pq=\(pq != nil ? "loaded" : "off")"
                }
                let bboxDetails = loaded.searchConfig.useBoundingBoxes ? ", bbox=on" : ", bbox=off"
                FileHandle.standardError.write(Data(
                    "loader: ready (count=\(index.header.count), scale=\(index.header.scale), ivf=\(ivf != nil ? "on" : "off")\(pqDetails), nprobe=\(loaded.searchConfig.nprobe)\(adaptiveDetails)\(bboxDetails))\n".utf8
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

        router.get("/debug/stats") { _, _ -> Response in
            do {
                return jsonResponse(body: try debugStats.jsonData())
            } catch {
                return jsonResponse(body: fallbackBody)
            }
        }

        router.post("/debug/stats/reset") { _, _ -> Response in
            debugStats.reset()
            return jsonResponse(body: Data(#"{"ok":true}"#.utf8))
        }

        router.post("/fraud-score") { request, context -> Response in
            guard let loaded = state.current else {
                return jsonResponse(body: fallbackBody)
            }
            var metrics = RequestPhaseMetrics()
            do {
                let bodyCollectStarted = DispatchTime.now().uptimeNanoseconds
                let bodyBuffer = try await request.body.collect(upTo: 64 * 1024)
                metrics.bodyCollectNs = DispatchTime.now().uptimeNanoseconds - bodyCollectStarted
                let quantized: [Int16]
                do {
                    let parseStarted = DispatchTime.now().uptimeNanoseconds
                    let parsed = try bodyBuffer.withUnsafeReadableBytes { rawBuffer in
                        try FastRequestParser.parsedQuery(from: rawBuffer)
                    }
                    metrics.parseNs = DispatchTime.now().uptimeNanoseconds - parseStarted
                    let vectorizeStarted = DispatchTime.now().uptimeNanoseconds
                    quantized = loaded.vectorizer.quantize(
                        transactionAmount: parsed.transactionAmount,
                        installments: parsed.installments,
                        requestedAt: parsed.requestedAt,
                        customerAvgAmount: parsed.customerAvgAmount,
                        customerTxCount24h: parsed.customerTxCount24h,
                        knownMerchant: parsed.knownMerchant,
                        merchantAvgAmount: parsed.merchantAvgAmount,
                        terminalIsOnline: parsed.terminalIsOnline,
                        terminalCardPresent: parsed.terminalCardPresent,
                        terminalKmFromHome: parsed.terminalKmFromHome,
                        merchantMccCode: parsed.merchantMccCode,
                        lastTransaction: parsed.lastTransaction
                    )
                    metrics.vectorizeNs = DispatchTime.now().uptimeNanoseconds - vectorizeStarted
                    metrics.fastPath = true
                } catch {
                    let body = Data(bodyBuffer.readableBytesView)
                    let parseStarted = DispatchTime.now().uptimeNanoseconds
                    let fraudRequest = try decoder.decode(FraudRequest.self, from: body)
                    metrics.parseNs = DispatchTime.now().uptimeNanoseconds - parseStarted
                    let vectorizeStarted = DispatchTime.now().uptimeNanoseconds
                    let raw = try loaded.vectorizer.vectorize(fraudRequest)
                    quantized = loaded.vectorizer.quantize(raw)
                    metrics.vectorizeNs = DispatchTime.now().uptimeNanoseconds - vectorizeStarted
                    metrics.fallbackPath = true
                }
                let searchStarted = DispatchTime.now().uptimeNanoseconds
                let rawFraudVotes: Int
                if debugStats.isEnabled {
                    var searchMetrics = SearchMetrics()
                    rawFraudVotes = KNN.fraudVoteCount(
                        query: quantized,
                        in: loaded.index,
                        ivf: loaded.ivf,
                        pq: loaded.pq,
                        config: loaded.searchConfig,
                        metrics: &searchMetrics,
                        k: 5
                    )
                    metrics.searchCentroidNs = searchMetrics.centroidSearchNs
                    metrics.searchShortlistNs = searchMetrics.shortlistNs
                    metrics.searchExactFallbackCount = searchMetrics.exactFallbackCount
                    metrics.searchAdaptiveExpandCount = searchMetrics.adaptiveExpandCount
                } else {
                    rawFraudVotes = KNN.fraudVoteCount(
                        query: quantized,
                        in: loaded.index,
                        ivf: loaded.ivf,
                        pq: loaded.pq,
                        config: loaded.searchConfig,
                        k: 5
                    )
                }
                metrics.searchNs = DispatchTime.now().uptimeNanoseconds - searchStarted
                let responseStarted = DispatchTime.now().uptimeNanoseconds
                let response = jsonResponse(body: FraudScoring.responseBody(fraudVoteCount: rawFraudVotes))
                metrics.responseNs = DispatchTime.now().uptimeNanoseconds - responseStarted
                debugStats.record(metrics)
                return response
            } catch {
                context.logger.debug("fraud-score: \(error)")
                metrics.failed = true
                let responseStarted = DispatchTime.now().uptimeNanoseconds
                let response = jsonResponse(body: fallbackBody)
                metrics.responseNs = DispatchTime.now().uptimeNanoseconds - responseStarted
                debugStats.record(metrics)
                return response
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

    private static func synthesizeAndWarmKNN(loaded: LoadedState) {
        let warmupCount = ProcessInfo.processInfo.environment["WARMUP_COUNT"].flatMap(Int.init) ?? 5_000
        guard warmupCount > 0 else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        let count = loaded.index.header.count
        let stride = loaded.index.header.stride
        let basePtr = loaded.index.vectors.baseAddress!
        var rng = SplitMix64(seed: 0xCAFE_BABE_DEAD_BEEF)
        var sink = 0
        for _ in 0..<warmupCount {
            let recIdx = Int(rng.next() % UInt64(count))
            var query = [Int16](repeating: 0, count: 16)
            for lane in 0..<stride {
                query[lane] = basePtr[recIdx * stride + lane]
            }
            // Perturb a couple lanes so query isn't an exact ref hit (forces full scan path).
            query[0] = query[0] &+ 17
            query[7] = query[7] &+ 23
            sink &+= KNN.fraudVoteCount(
                query: query,
                in: loaded.index,
                ivf: loaded.ivf,
                pq: loaded.pq,
                config: loaded.searchConfig,
                k: 5
            )
        }
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        FileHandle.standardError.write(Data("warmup: \(warmupCount) queries in \(String(format: "%.1f", elapsedMs)) ms (sink=\(sink))\n".utf8))
    }

    private struct SplitMix64 {
        private var state: UInt64
        init(seed: UInt64) { self.state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
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
