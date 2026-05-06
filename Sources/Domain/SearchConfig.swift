import Foundation

public struct SearchConfig: Sendable {
    public let nprobe: Int
    public let initialNprobe: Int
    public let adaptiveMinFraudVotes: Int
    public let adaptiveMaxFraudVotes: Int

    public var adaptiveEnabled: Bool {
        initialNprobe < nprobe && adaptiveMinFraudVotes <= adaptiveMaxFraudVotes
    }

    public init(
        nprobe: Int = 4,
        initialNprobe: Int? = nil,
        adaptiveMinFraudVotes: Int = 2,
        adaptiveMaxFraudVotes: Int = 3
    ) {
        let clampedNProbe = max(1, nprobe)
        let clampedInitial = min(max(1, initialNprobe ?? clampedNProbe), clampedNProbe)
        self.nprobe = clampedNProbe
        self.initialNprobe = clampedInitial
        self.adaptiveMinFraudVotes = adaptiveMinFraudVotes
        self.adaptiveMaxFraudVotes = adaptiveMaxFraudVotes
    }

    public func shouldExpand(after fraudVotes: Int) -> Bool {
        adaptiveEnabled &&
        fraudVotes >= adaptiveMinFraudVotes &&
        fraudVotes <= adaptiveMaxFraudVotes
    }
}
