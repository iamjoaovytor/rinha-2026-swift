import CSearch

extension KNN {
    static func fraudVoteCountIVFPQ(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        pq: IVFPQIndex,
        centroidNeighbors: [rinha_neighbor_t],
        orderedVectors: UnsafeBufferPointer<Int16>,
        orderedLabels: UnsafeBufferPointer<UInt8>,
        useBoundingBoxes: Bool,
        rerankCandidates: Int,
        metrics: UnsafeMutablePointer<SearchMetrics>?,
        k: Int
    ) -> Int {
        guard pq.header.count == ivf.header.count,
              pq.header.stride == ivf.header.stride else {
            return fraudVoteCountIVFContiguous(
                query: query,
                in: index,
                ivf: ivf,
                centroidNeighbors: centroidNeighbors,
                orderedVectors: orderedVectors,
                orderedLabels: orderedLabels,
                useBoundingBoxes: false,
                metrics: metrics,
                k: k
            )
        }

        let offsets = ivf.clusterOffsets
        let shortlistStarted = metricStart(metrics)
        var approxTop = [Neighbor]()
        approxTop.reserveCapacity(rerankCandidates)
        let lookupTables = buildPQLookupTables(query: query, pq: pq)
        let codes = pq.codes

        for centroid in centroidNeighbors {
            let cluster = Int(centroid.record_index)
            let start = Int(offsets[cluster])
            let end = Int(offsets[cluster + 1])
            guard end > start else { continue }
            for orderedIndex in start..<end {
                let codeBase = orderedIndex * pq.header.subvectorCount
                var distance: Int64 = 0
                for subvector in 0..<pq.header.subvectorCount {
                    let code = Int(codes[codeBase + subvector])
                    distance &+= lookupTables[subvector * 256 + code]
                }
                insertSwift(
                    Neighbor(recordIndex: orderedIndex, distanceSquared: distance),
                    into: &approxTop,
                    capacity: rerankCandidates
                )
            }
        }

        if approxTop.count < k {
            metrics?.pointee.exactFallbackCount &+= 1
            let result = fraudVoteCountExact(query: query, in: index, k: k)
            metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
            return result
        }

        var exactTop = [Neighbor]()
        exactTop.reserveCapacity(k)
        let stride = ivf.header.stride
        let dim = index.header.dim
        for candidate in approxTop {
            let base = candidate.recordIndex * stride
            var sum: Int64 = 0
            for lane in 0..<dim {
                let diff = Int32(query[lane]) - Int32(orderedVectors[base + lane])
                sum &+= Int64(diff &* diff)
            }
            insertSwift(
                Neighbor(recordIndex: candidate.recordIndex, distanceSquared: sum),
                into: &exactTop,
                capacity: k
            )
        }

        if useBoundingBoxes,
           exactTop.count == k,
           let extraClusters = additionalClustersToScan(
                query: query,
                ivf: ivf,
                centroidNeighbors: centroidNeighbors,
                worstDistanceSquared: exactTop[k - 1].distanceSquared,
                dim: dim
           ) {
            for extraCluster in extraClusters {
                if extraCluster.lowerBoundSquared >= exactTop[k - 1].distanceSquared {
                    break
                }
                let start = Int(offsets[extraCluster.cluster])
                let end = Int(offsets[extraCluster.cluster + 1])
                _ = withExactNeighborsInContiguousCluster(
                    query: query,
                    orderedVectors: orderedVectors,
                    start: start,
                    end: end,
                    stride: stride,
                    dim: dim,
                    k: k
                ) { rawNeighbors in
                    for raw in rawNeighbors where raw.record_index >= 0 {
                        insertSwift(
                            Neighbor(
                                recordIndex: start + Int(raw.record_index),
                                distanceSquared: raw.distance_squared
                            ),
                            into: &exactTop,
                            capacity: k
                        )
                    }
                }
            }
        }
        metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)

        if exactTop.count < k {
            metrics?.pointee.exactFallbackCount &+= 1
            return fraudVoteCountExact(query: query, in: index, k: k)
        }

        var fraudVotes = 0
        for neighbor in exactTop where orderedLabels[neighbor.recordIndex] == 1 {
            fraudVotes += 1
        }
        return fraudVotes
    }

    static func buildPQLookupTables(
        query: [Int16],
        pq: IVFPQIndex
    ) -> [Int64] {
        var lookup = [Int64](repeating: 0, count: pq.header.subvectorCount * 256)
        let codebooks = pq.codebooks
        for subvector in 0..<pq.header.subvectorCount {
            let queryBase = subvector * pq.header.subvectorWidth
            let tableBase = subvector * 256
            for code in 0..<256 {
                let codebookBase = (subvector * 256 + code) * pq.header.subvectorWidth
                var sum: Int64 = 0
                for lane in 0..<pq.header.subvectorWidth {
                    let diff = Int32(query[queryBase + lane]) - Int32(codebooks[codebookBase + lane])
                    sum &+= Int64(diff &* diff)
                }
                lookup[tableBase + code] = sum
            }
        }
        return lookup
    }
}
