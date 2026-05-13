import Domain
import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1

extension RinhaAPI {
    static func makeChannelSetup(
        state: LoaderState,
        debugStats: DebugStatsCollector
    ) -> @Sendable (Channel) -> EventLoopFuture<Void> {
        { channel in
            channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                channel.pipeline.addHandler(FraudHandler(state: state, debugStats: debugStats))
            }
        }
    }

    static func loadState(config: APIConfig, into state: LoaderState) async {
        do {
            let mccRisk = try MccRiskTable.load(path: config.mccRiskPath)
            let index = try ReferencesIndex.load(path: config.referencesPath)
            let ivf: IVFIndex?
            if FileManager.default.fileExists(atPath: config.ivfPath) {
                ivf = try IVFIndex.load(path: config.ivfPath)
                ivf?.warm()
            } else {
                ivf = nil
            }
            let pq: IVFPQIndex?
            if FileManager.default.fileExists(atPath: config.ivfpqPath) {
                pq = try IVFPQIndex.load(path: config.ivfpqPath)
                pq?.warm()
            } else {
                pq = nil
            }
            index.warm()
            let loaded = LoadedState(
                index: index,
                ivf: ivf,
                pq: pq,
                searchConfig: config.searchConfig,
                vectorizer: Vectorizer(mccRisk: mccRisk)
            )
            SearchWarmup.run(loaded: loaded)
            state.install(loaded)
            FileHandle.standardError.write(Data(
                "loader: ready (count=\(index.header.count), scale=\(index.header.scale), nprobe=\(loaded.searchConfig.nprobe))\n".utf8
            ))
        } catch {
            state.recordFailure("\(error)")
            FileHandle.standardError.write(Data("loader: \(error)\n".utf8))
        }
    }

    static func runServer(
        config: APIConfig,
        eventLoopGroup: MultiThreadedEventLoopGroup,
        channelSetup: @escaping @Sendable (Channel) -> EventLoopFuture<Void>
    ) async throws {
        if let socketPath = config.socketPath {
            if config.useSocketHandoff {
                try await runSocketHandoffServer(
                    socketPath: socketPath,
                    eventLoopGroup: eventLoopGroup,
                    channelSetup: channelSetup
                )
            } else {
                try await runUnixServer(
                    socketPath: socketPath,
                    eventLoopGroup: eventLoopGroup,
                    channelSetup: channelSetup
                )
            }
        } else {
            try await runTCPServer(
                port: config.port,
                eventLoopGroup: eventLoopGroup,
                channelSetup: channelSetup
            )
        }
    }

    private static func runSocketHandoffServer(
        socketPath: String,
        eventLoopGroup: MultiThreadedEventLoopGroup,
        channelSetup: @escaping @Sendable (Channel) -> EventLoopFuture<Void>
    ) async throws {
        let ctrlPath = "\(socketPath).ctrl"
        let bootstrap = ClientBootstrap(group: eventLoopGroup)
            .channelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .channelInitializer(channelSetup)
        let handoffAcceptor = SocketHandoffAcceptor(
            ctrlPath: ctrlPath,
            bootstrap: bootstrap,
            logger: { message in
                FileHandle.standardError.write(Data("\(message)\n".utf8))
            }
        )
        try handoffAcceptor.start()
        FileHandle.standardError.write(Data("handoff: listening on \(ctrlPath)\n".utf8))
        while true {
            try await Task.sleep(nanoseconds: 86_400_000_000_000)
        }
    }

    private static func runUnixServer(
        socketPath: String,
        eventLoopGroup: MultiThreadedEventLoopGroup,
        channelSetup: @escaping @Sendable (Channel) -> EventLoopFuture<Void>
    ) async throws {
        try? FileManager.default.removeItem(atPath: socketPath)
        let bootstrap = ServerBootstrap(group: eventLoopGroup)
            .serverChannelOption(ChannelOptions.backlog, value: 16384)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .childChannelInitializer(channelSetup)
        let serverChannel = try await bootstrap.bind(unixDomainSocketPath: socketPath).get()
        try await serverChannel.closeFuture.get()
        try await eventLoopGroup.shutdownGracefully()
    }

    private static func runTCPServer(
        port: Int,
        eventLoopGroup: MultiThreadedEventLoopGroup,
        channelSetup: @escaping @Sendable (Channel) -> EventLoopFuture<Void>
    ) async throws {
        let bootstrap = ServerBootstrap(group: eventLoopGroup)
            .serverChannelOption(ChannelOptions.backlog, value: 16384)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.socketOption(.tcp_nodelay), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .childChannelInitializer(channelSetup)
        let serverChannel = try await bootstrap.bind(host: "0.0.0.0", port: port).get()
        try await serverChannel.closeFuture.get()
        try await eventLoopGroup.shutdownGracefully()
    }
}
