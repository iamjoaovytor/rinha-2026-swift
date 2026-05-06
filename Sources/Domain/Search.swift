import CSearch
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
    /// Exact k-NN against the `references.bin` mapping using a native C
    /// kernel. On x86_64 the kernel switches to AVX2 when supported at
    /// runtime; other architectures stay on the scalar C path.
    public static func topK(
        query: [Int16],
        in index: ReferencesIndex,
        k: Int = 5
    ) -> [Neighbor] {
        withTopKRaw(query: query, in: index, k: k) { rawNeighbors in
            rawNeighbors.map {
                Neighbor(
                    recordIndex: Int($0.record_index),
                    distanceSquared: $0.distance_squared
                )
            }
        }
    }

    public static func fraudVoteCount(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex? = nil,
        config: SearchConfig = SearchConfig(),
        k: Int = 5
    ) -> Int {
        if let ivf {
            return fraudVoteCountIVF(
                query: query,
                in: index,
                ivf: ivf,
                config: config,
                k: k
            )
        }
        return withTopKRaw(query: query, in: index, k: k) { rawNeighbors in
            let labels = index.labels
            var fraudVotes = 0
            for neighbor in rawNeighbors where labels[Int(neighbor.record_index)] == 1 {
                fraudVotes += 1
            }
            return fraudVotes
        }
    }

    private static func fraudVoteCountIVF(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        config: SearchConfig,
        k: Int
    ) -> Int {
        let initialVotes = fraudVoteCountIVF(
            query: query,
            in: index,
            ivf: ivf,
            nprobe: config.initialNprobe,
            k: k
        )
        guard config.shouldExpand(after: initialVotes) else {
            return initialVotes
        }
        return fraudVoteCountIVF(
            query: query,
            in: index,
            ivf: ivf,
            nprobe: config.nprobe,
            k: k
        )
    }

    private static func fraudVoteCountIVF(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        nprobe: Int,
        k: Int
    ) -> Int {
        let clusterCount = ivf.header.clusterCount
        let nprobe = min(max(1, nprobe), clusterCount)
        precondition(k > 0, "k must be positive")

        let centroidNeighbors = query.withUnsafeBufferPointer { queryBuffer in
            withUnsafeTemporaryAllocation(of: rinha_neighbor_t.self, capacity: nprobe) { rawNeighbors in
                rinha_topk_exact_i16(
                    queryBuffer.baseAddress,
                    ivf.centroids.baseAddress,
                    numericCast(clusterCount),
                    numericCast(index.header.dim),
                    numericCast(ivf.header.stride),
                    numericCast(nprobe),
                    rawNeighbors.baseAddress
                )
                return Array(UnsafeBufferPointer(start: rawNeighbors.baseAddress, count: nprobe))
            }
        }

        let offsets = ivf.clusterOffsets
        var candidateCount = 0
        for centroid in centroidNeighbors {
            let cluster = Int(centroid.record_index)
            let start = Int(offsets[cluster])
            let end = Int(offsets[cluster + 1])
            candidateCount += end - start
        }

        if candidateCount < k {
            return fraudVoteCount(query: query, in: index, k: k)
        }

        return withUnsafeTemporaryAllocation(of: UInt32.self, capacity: candidateCount) { candidates in
            var writeIndex = 0
            let postings = ivf.postings
            for centroid in centroidNeighbors {
                let cluster = Int(centroid.record_index)
                let start = Int(offsets[cluster])
                let end = Int(offsets[cluster + 1])
                for postingIndex in start..<end {
                    candidates[writeIndex] = postings[postingIndex]
                    writeIndex += 1
                }
            }

            return withUnsafeTemporaryAllocation(of: rinha_neighbor_t.self, capacity: k) { rawNeighbors in
                query.withUnsafeBufferPointer { queryBuffer in
                    rinha_topk_exact_i16_indexed(
                        queryBuffer.baseAddress,
                        index.vectors.baseAddress,
                        candidates.baseAddress,
                        numericCast(candidateCount),
                        numericCast(index.header.dim),
                        numericCast(index.header.stride),
                        numericCast(k),
                        rawNeighbors.baseAddress
                    )
                    let labels = index.labels
                    var fraudVotes = 0
                    for neighbor in UnsafeBufferPointer(start: rawNeighbors.baseAddress, count: k)
                    where labels[Int(neighbor.record_index)] == 1 {
                        fraudVotes += 1
                    }
                    return fraudVotes
                }
            }
        }
    }

    private static func withTopKRaw<T>(
        query: [Int16],
        in index: ReferencesIndex,
        k: Int,
        _ body: (UnsafeBufferPointer<rinha_neighbor_t>) -> T
    ) -> T {
        precondition(query.count == index.header.stride,
                     "query lanes must match index stride")
        precondition(k > 0, "k must be positive")

        let count = index.header.count
        let k = min(k, count)
        precondition(k > 0, "index must not be empty")

        return withUnsafeTemporaryAllocation(of: rinha_neighbor_t.self, capacity: k) { rawNeighbors in
            query.withUnsafeBufferPointer { queryBuffer in
                rinha_topk_exact_i16(
                    queryBuffer.baseAddress,
                    index.vectors.baseAddress,
                    numericCast(index.header.count),
                    numericCast(index.header.dim),
                    numericCast(index.header.stride),
                    numericCast(k),
                    rawNeighbors.baseAddress
                )
                return body(UnsafeBufferPointer(start: rawNeighbors.baseAddress, count: k))
            }
        }
    }

    /// Swift oracle kept for tests and benchmarking while the native kernel
    /// takes over production traffic.
    static func topKSwift(
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
        let k = min(k, count)
        guard k > 0 else { return [] }

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
                insertSwift(
                    Neighbor(recordIndex: record, distanceSquared: sum),
                    into: &top, capacity: k
                )
            }
        }

        return top
    }

    @inline(__always)
    private static func insertSwift(
        _ candidate: Neighbor,
        into top: inout [Neighbor],
        capacity: Int
    ) {
        if top.count < capacity {
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
    private static let responseBodies = [
        #"{"approved":true,"fraud_score":0.0}"#,
        #"{"approved":true,"fraud_score":0.2}"#,
        #"{"approved":true,"fraud_score":0.4}"#,
        #"{"approved":false,"fraud_score":0.6}"#,
        #"{"approved":false,"fraud_score":0.8}"#,
        #"{"approved":false,"fraud_score":1.0}"#
    ]

    /// Reduces `topNeighbors` to a fraud score using majority vote among
    /// labels and approves below `approvalThreshold`. Spec uses k=5, so the
    /// score is one of `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`.
    public static func score(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> ScoreResult {
        precondition(!neighbors.isEmpty, "need at least one neighbor")
        let fraudVotes = fraudVoteCount(neighbors: neighbors, index: index)
        let fraudScore = Double(fraudVotes) / Double(neighbors.count)
        let approved = fraudScore < approvalThreshold
        return ScoreResult(
            approved: approved,
            fraudScore: fraudScore,
            topNeighbors: neighbors
        )
    }

    public static func responseBody(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> String {
        responseBody(fraudVoteCount: fraudVoteCount(neighbors: neighbors, index: index))
    }

    public static func responseBody(fraudVoteCount: Int) -> String {
        responseBodies[fraudVoteCount]
    }

    @inline(__always)
    private static func fraudVoteCount(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> Int {
        precondition(!neighbors.isEmpty, "need at least one neighbor")
        let labels = index.labels
        var fraudVotes = 0
        for neighbor in neighbors where labels[neighbor.recordIndex] == 1 {
            fraudVotes += 1
        }
        return fraudVotes
    }
}
