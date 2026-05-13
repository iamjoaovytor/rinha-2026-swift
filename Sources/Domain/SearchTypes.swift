/// Result of a single nearest-neighbor probe against `references.bin`.
public struct Neighbor: Sendable, Equatable {
    public let recordIndex: Int
    public let distanceSquared: Int64

    public init(recordIndex: Int, distanceSquared: Int64) {
        self.recordIndex = recordIndex
        self.distanceSquared = distanceSquared
    }
}

public struct ScoreResult: Sendable, Equatable {
    public let approved: Bool
    public let fraudScore: Double
    public let topNeighbors: [Neighbor]

    public init(approved: Bool, fraudScore: Double, topNeighbors: [Neighbor]) {
        self.approved = approved
        self.fraudScore = fraudScore
        self.topNeighbors = topNeighbors
    }
}

public struct SearchMetrics: Sendable, Equatable {
    public var centroidSearchNs: UInt64 = 0
    public var shortlistNs: UInt64 = 0
    public var exactFallbackCount: UInt64 = 0
    public var adaptiveExpandCount: UInt64 = 0

    public init() {}
}
