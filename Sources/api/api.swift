import Domain
import Foundation
import NIOConcurrencyHelpers
import NIOCore
import NIOPosix
import NIOHTTP1

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
    static let okBody = #"{"ok":true}"#
    static let decoder = JSONDecoder()

    static func main() async throws {
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
                FileHandle.standardError.write(Data(
                    "loader: ready (count=\(index.header.count), scale=\(index.header.scale), nprobe=\(loaded.searchConfig.nprobe))\n".utf8
                ))
            } catch {
                state.recordFailure("\(error)")
                FileHandle.standardError.write(Data("loader: \(error)\n".utf8))
            }
        }

        let serverChannel: Channel
        if let socketPath = env["SOCKET_PATH"], !socketPath.isEmpty {
            try? FileManager.default.removeItem(atPath: socketPath)
            let bootstrap = ServerBootstrap(group: eventLoopGroup)
                .serverChannelOption(ChannelOptions.backlog, value: 16384)
                .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
                .childChannelInitializer { channel in
                    channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                        channel.pipeline.addHandler(FraudHandler(state: state, debugStats: debugStats))
                    }
                }
            serverChannel = try await bootstrap.bind(unixDomainSocketPath: socketPath).get()
        } else {
            let port = env["PORT"].flatMap(Int.init) ?? 9999
            let bootstrap = ServerBootstrap(group: eventLoopGroup)
                .serverChannelOption(ChannelOptions.backlog, value: 16384)
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelOption(ChannelOptions.socketOption(.tcp_nodelay), value: 1)
                .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
                .childChannelInitializer { channel in
                    channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                        channel.pipeline.addHandler(FraudHandler(state: state, debugStats: debugStats))
                    }
                }
            serverChannel = try await bootstrap.bind(host: "0.0.0.0", port: port).get()
        }
        try await serverChannel.closeFuture.get()
        try await eventLoopGroup.shutdownGracefully()
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
}

private final class FraudHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let state: LoaderState
    private let debugStats: DebugStatsCollector
    private var head: HTTPRequestHead?
    private var body: ByteBuffer?

    init(state: LoaderState, debugStats: DebugStatsCollector) {
        self.state = state
        self.debugStats = debugStats
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)
        switch part {
        case .head(let head):
            self.head = head
            self.body = nil
        case .body(var buffer):
            if self.body == nil {
                self.body = buffer
            } else {
                self.body!.writeBuffer(&buffer)
            }
        case .end:
            guard let head = self.head else { return }
            handle(context: context, head: head, body: self.body)
            self.head = nil
            self.body = nil
        }
    }

    private func handle(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer?) {
        let keepAlive = head.isKeepAlive
        switch (head.method, head.uri) {
        case (.POST, "/fraud-score"):
            handleFraud(context: context, body: body, keepAlive: keepAlive)
        case (.GET, "/ready"):
            let status: HTTPResponseStatus = state.isReady ? .ok : .serviceUnavailable
            writeEmpty(context: context, status: status, keepAlive: keepAlive)
        case (.GET, "/debug/stats"):
            do {
                let data = try debugStats.jsonData()
                var buf = context.channel.allocator.buffer(capacity: data.count)
                buf.writeBytes(data)
                writeJSON(context: context, status: .ok, buffer: buf, keepAlive: keepAlive)
            } catch {
                writeJSONString(context: context, status: .internalServerError, body: RinhaAPI.fallbackBody, keepAlive: keepAlive)
            }
        case (.POST, "/debug/stats/reset"):
            debugStats.reset()
            writeJSONString(context: context, status: .ok, body: RinhaAPI.okBody, keepAlive: keepAlive)
        default:
            writeEmpty(context: context, status: .notFound, keepAlive: keepAlive)
        }
    }

    private func handleFraud(context: ChannelHandlerContext, body: ByteBuffer?, keepAlive: Bool) {
        guard let loaded = state.current, let body = body else {
            writeJSONString(context: context, status: .ok, body: RinhaAPI.fallbackBody, keepAlive: keepAlive)
            return
        }
        var metrics = RequestPhaseMetrics()
        do {
            let quantized: [Int16]
            do {
                let parseStarted = DispatchTime.now().uptimeNanoseconds
                let parsed = try body.withUnsafeReadableBytes { rawBuffer in
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
                let bodyData = Data(body.readableBytesView)
                let parseStarted = DispatchTime.now().uptimeNanoseconds
                let fraudRequest = try RinhaAPI.decoder.decode(FraudRequest.self, from: bodyData)
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
            writeJSONString(context: context, status: .ok, body: FraudScoring.responseBody(fraudVoteCount: rawFraudVotes), keepAlive: keepAlive)
            metrics.responseNs = DispatchTime.now().uptimeNanoseconds - responseStarted
            debugStats.record(metrics)
        } catch {
            metrics.failed = true
            writeJSONString(context: context, status: .ok, body: RinhaAPI.fallbackBody, keepAlive: keepAlive)
            debugStats.record(metrics)
        }
    }

    private func writeJSONString(context: ChannelHandlerContext, status: HTTPResponseStatus, body: String, keepAlive: Bool) {
        var buf = context.channel.allocator.buffer(capacity: body.utf8.count)
        buf.writeString(body)
        writeJSON(context: context, status: status, buffer: buf, keepAlive: keepAlive)
    }

    private func writeJSON(context: ChannelHandlerContext, status: HTTPResponseStatus, buffer: ByteBuffer, keepAlive: Bool) {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "application/json")
        headers.add(name: "content-length", value: "\(buffer.readableBytes)")
        if !keepAlive { headers.add(name: "connection", value: "close") }
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        context.write(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        if keepAlive {
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
        } else {
            let promise = context.eventLoop.makePromise(of: Void.self)
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: promise)
            promise.futureResult.whenComplete { _ in
                context.close(promise: nil)
            }
        }
    }

    private func writeEmpty(context: ChannelHandlerContext, status: HTTPResponseStatus, keepAlive: Bool) {
        var headers = HTTPHeaders()
        headers.add(name: "content-length", value: "0")
        if !keepAlive { headers.add(name: "connection", value: "close") }
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        if keepAlive {
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
        } else {
            let promise = context.eventLoop.makePromise(of: Void.self)
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: promise)
            promise.futureResult.whenComplete { _ in
                context.close(promise: nil)
            }
        }
    }
}
