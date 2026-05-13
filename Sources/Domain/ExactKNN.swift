import CSearch

extension KNN {
    static func fraudVoteCountExact(
        query: [Int16],
        in index: ReferencesIndex,
        k: Int
    ) -> Int {
        withTopKRaw(query: query, in: index, k: k) { rawNeighbors in
            let labels = index.labels
            var fraudVotes = 0
            for neighbor in rawNeighbors where labels[Int(neighbor.record_index)] == 1 {
                fraudVotes += 1
            }
            return fraudVotes
        }
    }

    static func withTopKRaw<T>(
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

    static func exactNeighborsInContiguousCluster(
        query: [Int16],
        orderedVectors: UnsafeBufferPointer<Int16>,
        start: Int,
        end: Int,
        stride: Int,
        dim: Int,
        k: Int
    ) -> [Neighbor] {
        let candidateCount = end - start
        guard candidateCount > 0 else { return [] }

        return withUnsafeTemporaryAllocation(of: rinha_neighbor_t.self, capacity: min(k, candidateCount)) { rawNeighbors in
            query.withUnsafeBufferPointer { queryBuffer in
                let vectorsBase = orderedVectors.baseAddress!.advanced(by: start * stride)
                rinha_topk_exact_i16_filtered(
                    queryBuffer.baseAddress,
                    vectorsBase,
                    numericCast(candidateCount),
                    numericCast(dim),
                    numericCast(stride),
                    numericCast(min(k, candidateCount)),
                    rawNeighbors.baseAddress
                )
                let rawCount = min(k, candidateCount)
                let rawBuffer = UnsafeBufferPointer(start: rawNeighbors.baseAddress, count: rawCount)
                var neighbors = [Neighbor]()
                neighbors.reserveCapacity(rawCount)
                for raw in rawBuffer where raw.record_index >= 0 {
                    neighbors.append(
                        Neighbor(
                            recordIndex: start + Int(raw.record_index),
                            distanceSquared: raw.distance_squared
                        )
                    )
                }
                return neighbors
            }
        }
    }

    static func exactNeighborsInIndexedCluster(
        query: [Int16],
        index: ReferencesIndex,
        postings: UnsafeBufferPointer<UInt32>,
        start: Int,
        end: Int,
        k: Int
    ) -> [Neighbor] {
        let candidateCount = end - start
        guard candidateCount > 0 else { return [] }

        return withUnsafeTemporaryAllocation(of: rinha_neighbor_t.self, capacity: k) { rawNeighbors in
            query.withUnsafeBufferPointer { queryBuffer in
                let candidates = postings.baseAddress!.advanced(by: start)
                rinha_topk_exact_i16_indexed_filtered(
                    queryBuffer.baseAddress,
                    index.vectors.baseAddress,
                    candidates,
                    numericCast(candidateCount),
                    numericCast(index.header.dim),
                    numericCast(index.header.stride),
                    numericCast(k),
                    rawNeighbors.baseAddress
                )
                let rawCount = min(k, candidateCount)
                let rawBuffer = UnsafeBufferPointer(start: rawNeighbors.baseAddress, count: rawCount)
                var neighbors = [Neighbor]()
                neighbors.reserveCapacity(rawCount)
                for raw in rawBuffer where raw.record_index >= 0 {
                    neighbors.append(
                        Neighbor(
                            recordIndex: Int(raw.record_index),
                            distanceSquared: raw.distance_squared
                        )
                    )
                }
                return neighbors
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
    static func insertSwift(
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
