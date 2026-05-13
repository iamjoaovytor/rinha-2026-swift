import Domain
import Foundation
import NIOCore
import NIOHTTP1

final class FraudHandler: ChannelInboundHandler, @unchecked Sendable {
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
