import Foundation

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

public enum KNN {
    /// Pure Swift exact k-NN against the `references.bin` mapping. Slow at
    /// 3M × 14 dims (oracle baseline; replace with C+SIMD before going to
    /// production). Distances are computed in Int64 over `dim` lanes; the
    /// padding lanes 14 and 15 are zero on both sides so they would not
    /// shift the result, but we skip them to save eight multiplies.
    public static func topK(
        query: [Int16],
        in index: ReferencesIndex,
        k: Int = 5
    ) -> [Neighbor] {
        precondition(query.count == index.header.stride,
                     "query lanes must match index stride")
        precondition(k > 0, "k must be positive")

        let count = index.header.count
        let stride = index.header.stride
        let dim = index.header.dim
        let vectors = index.vectors

        var top = [Neighbor]()
        top.reserveCapacity(k)

        query.withUnsafeBufferPointer { queryBuffer in
            let queryPointer = queryBuffer.baseAddress!
            let basePointer = vectors.baseAddress!

            for record in 0..<count {
                let recordPointer = basePointer.advanced(by: record * stride)
                var sum: Int64 = 0
                for lane in 0..<dim {
                    let q = Int32(queryPointer[lane])
                    let r = Int32(recordPointer[lane])
                    let diff = q - r
                    sum &+= Int64(diff &* diff)
                }
                insert(
                    Neighbor(recordIndex: record, distanceSquared: sum),
                    into: &top, capacity: k
                )
            }
        }

        return top
    }

    /// Inserts `candidate` into a max-distance heap-like array kept in
    /// ascending order. Cheap because `k` is tiny (5).
    @inline(__always)
    private static func insert(
        _ candidate: Neighbor,
        into top: inout [Neighbor],
        capacity: Int
    ) {
        if top.count < capacity {
            // Insertion sort entry; small k keeps cost negligible.
            var i = top.count
            top.append(candidate)
            while i > 0, top[i - 1].distanceSquared > candidate.distanceSquared {
                top[i] = top[i - 1]
                i -= 1
            }
            top[i] = candidate
            return
        }
        guard candidate.distanceSquared < top[capacity - 1].distanceSquared else {
            return
        }
        var i = capacity - 1
        while i > 0, top[i - 1].distanceSquared > candidate.distanceSquared {
            top[i] = top[i - 1]
            i -= 1
        }
        top[i] = candidate
    }
}

public enum FraudScoring {
    public static let approvalThreshold: Double = 0.5

    /// Reduces `topNeighbors` to a fraud score using majority vote among
    /// labels and approves below `approvalThreshold`. Spec uses k=5, so the
    /// score is one of `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`.
    public static func score(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> ScoreResult {
        precondition(!neighbors.isEmpty, "need at least one neighbor")
        let labels = index.labels
        var fraudVotes = 0
        for neighbor in neighbors where labels[neighbor.recordIndex] == 1 {
            fraudVotes += 1
        }
        let fraudScore = Double(fraudVotes) / Double(neighbors.count)
        let approved = fraudScore < approvalThreshold
        return ScoreResult(
            approved: approved,
            fraudScore: fraudScore,
            topNeighbors: neighbors
        )
    }
}
