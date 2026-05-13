import Domain
import Foundation
import NIOPosix

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
        let config = APIConfig(environment: ProcessInfo.processInfo.environment)
        let channelSetup = makeChannelSetup(state: state, debugStats: debugStats)

        Task.detached {
            await loadState(config: config, into: state)
        }

        try await runServer(config: config, eventLoopGroup: eventLoopGroup, channelSetup: channelSetup)
    }
}
