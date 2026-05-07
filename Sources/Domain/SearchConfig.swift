import Foundation

public struct SearchConfig: Sendable {
    public let nprobe: Int
    public let initialNprobe: Int
    public let adaptiveMinFraudVotes: Int
    public let adaptiveMaxFraudVotes: Int
    public let ivfpqRerankCandidates: Int?
    public let useBoundingBoxes: Bool

    public var adaptiveEnabled: Bool {
        initialNprobe < nprobe && adaptiveMinFraudVotes <= adaptiveMaxFraudVotes
    }

    public init(
        nprobe: Int = 8,
        initialNprobe: Int? = nil,
        adaptiveMinFraudVotes: Int = 2,
        adaptiveMaxFraudVotes: Int = 3,
        ivfpqRerankCandidates: Int? = nil,
        useBoundingBoxes: Bool = false
    ) {
        let clampedNProbe = max(1, nprobe)
        let clampedInitial = min(max(1, initialNprobe ?? clampedNProbe), clampedNProbe)
        let clampedRerank = ivfpqRerankCandidates.map { max(1, $0) }
        self.nprobe = clampedNProbe
        self.initialNprobe = clampedInitial
        self.adaptiveMinFraudVotes = adaptiveMinFraudVotes
        self.adaptiveMaxFraudVotes = adaptiveMaxFraudVotes
        self.ivfpqRerankCandidates = clampedRerank
        self.useBoundingBoxes = useBoundingBoxes
    }

    public func shouldExpand(after fraudVotes: Int) -> Bool {
        adaptiveEnabled &&
        fraudVotes >= adaptiveMinFraudVotes &&
        fraudVotes <= adaptiveMaxFraudVotes
    }

    public var ivfpqEnabled: Bool {
        if let ivfpqRerankCandidates {
            return ivfpqRerankCandidates > 5
        }
        return false
    }
}
