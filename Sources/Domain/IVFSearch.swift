import CSearch
import Foundation

extension KNN {
    static func fraudVoteCountIVF(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        pq: IVFPQIndex?,
        config: SearchConfig,
        metrics: UnsafeMutablePointer<SearchMetrics>?,
        k: Int
    ) -> Int {
        let initialVotes = fraudVoteCountIVF(
            query: query,
            in: index,
            ivf: ivf,
            pq: pq,
            nprobe: config.initialNprobe,
            useBoundingBoxes: config.useBoundingBoxes,
            ivfpqRerankCandidates: config.ivfpqRerankCandidates,
            metrics: metrics,
            k: k
        )
        guard config.shouldExpand(after: initialVotes) else {
            return initialVotes
        }
        metrics?.pointee.adaptiveExpandCount &+= 1
        return fraudVoteCountIVF(
            query: query,
            in: index,
            ivf: ivf,
            pq: pq,
            nprobe: config.nprobe,
            useBoundingBoxes: config.useBoundingBoxes,
            ivfpqRerankCandidates: config.ivfpqRerankCandidates,
            metrics: metrics,
            k: k
        )
    }

    static func fraudVoteCountIVF(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        pq: IVFPQIndex?,
        nprobe: Int,
        useBoundingBoxes: Bool,
        ivfpqRerankCandidates: Int?,
        metrics: UnsafeMutablePointer<SearchMetrics>?,
        k: Int
    ) -> Int {
        let clusterCount = ivf.header.clusterCount
        let nprobe = min(max(1, nprobe), clusterCount)
        precondition(k > 0, "k must be positive")

        let centroidStarted = metricStart(metrics)
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
        metricRecord(centroidStarted, keyPath: \.centroidSearchNs, metrics: metrics)

        if let orderedVectors = ivf.orderedVectors,
           let orderedLabels = ivf.orderedLabels {
            if let pq, !configSupportsPQ(pq: pq, ivf: ivf, configRerankCandidates: ivfpqRerankCandidates) {
                // fall through to exact contiguous path if PQ layout mismatches
            } else if let pq, let rerankCandidates = ivfpqRerankCandidates, rerankCandidates > 5 {
                return fraudVoteCountIVFPQ(
                    query: query,
                    in: index,
                    ivf: ivf,
                    pq: pq,
                    centroidNeighbors: centroidNeighbors,
                    orderedVectors: orderedVectors,
                    orderedLabels: orderedLabels,
                    useBoundingBoxes: useBoundingBoxes,
                    rerankCandidates: rerankCandidates,
                    metrics: metrics,
                    k: k
                )
            }
            return fraudVoteCountIVFContiguous(
                query: query,
                in: index,
                ivf: ivf,
                centroidNeighbors: centroidNeighbors,
                orderedVectors: orderedVectors,
                orderedLabels: orderedLabels,
                useBoundingBoxes: useBoundingBoxes,
                metrics: metrics,
                k: k
            )
        }

        if useBoundingBoxes,
           ivf.header.hasBoundingBoxes,
           let bboxMin = ivf.bboxMin,
           let bboxMax = ivf.bboxMax {
            return fraudVoteCountIVFPruned(
                query: query,
                in: index,
                ivf: ivf,
                centroidNeighbors: centroidNeighbors,
                bboxMin: bboxMin,
                bboxMax: bboxMax,
                metrics: metrics,
                k: k
            )
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
            metrics?.pointee.exactFallbackCount &+= 1
            let shortlistStarted = metricStart(metrics)
            let result = fraudVoteCountExact(query: query, in: index, k: k)
            metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
            return result
        }

        let shortlistStarted = metricStart(metrics)
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
                    rinha_topk_exact_i16_indexed_filtered(
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
                    where neighbor.record_index >= 0 && labels[Int(neighbor.record_index)] == 1 {
                        fraudVotes += 1
                    }
                    metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
                    return fraudVotes
                }
            }
        }
    }

    static func fraudVoteCountIVFContiguous(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        centroidNeighbors: [rinha_neighbor_t],
        orderedVectors: UnsafeBufferPointer<Int16>,
        orderedLabels: UnsafeBufferPointer<UInt8>,
        useBoundingBoxes: Bool,
        metrics: UnsafeMutablePointer<SearchMetrics>?,
        k: Int
    ) -> Int {
        let offsets = ivf.clusterOffsets
        let bboxMin = useBoundingBoxes && ivf.header.hasBoundingBoxes ? ivf.bboxMin : nil
        let bboxMax = useBoundingBoxes && ivf.header.hasBoundingBoxes ? ivf.bboxMax : nil
        var top = [Neighbor]()
        top.reserveCapacity(k)
        let shortlistStarted = metricStart(metrics)

        for centroid in centroidNeighbors {
            let cluster = Int(centroid.record_index)
            if top.count == k, let bboxMin, let bboxMax {
                let lowerBound = lowerBoundSquared(
                    query: query,
                    cluster: cluster,
                    bboxMin: bboxMin,
                    bboxMax: bboxMax,
                    stride: ivf.header.stride,
                    dim: index.header.dim
                )
                if lowerBound >= top[k - 1].distanceSquared {
                    continue
                }
            }

            let start = Int(offsets[cluster])
            let end = Int(offsets[cluster + 1])
            let candidateCount = end - start
            if candidateCount <= 0 { continue }

            let clusterNeighbors = exactNeighborsInContiguousCluster(
                query: query,
                orderedVectors: orderedVectors,
                start: start,
                end: end,
                stride: ivf.header.stride,
                dim: index.header.dim,
                k: k
            )

            for neighbor in clusterNeighbors {
                insertSwift(neighbor, into: &top, capacity: k)
            }
        }

        if useBoundingBoxes,
           top.count == k,
           let extraClusters = additionalClustersToScan(
                query: query,
                ivf: ivf,
                centroidNeighbors: centroidNeighbors,
                worstDistanceSquared: top[k - 1].distanceSquared,
                dim: index.header.dim
           ) {
            for extraCluster in extraClusters {
                if extraCluster.lowerBoundSquared >= top[k - 1].distanceSquared {
                    break
                }
                let start = Int(offsets[extraCluster.cluster])
                let end = Int(offsets[extraCluster.cluster + 1])
                let clusterNeighbors = exactNeighborsInContiguousCluster(
                    query: query,
                    orderedVectors: orderedVectors,
                    start: start,
                    end: end,
                    stride: ivf.header.stride,
                    dim: index.header.dim,
                    k: k
                )
                for neighbor in clusterNeighbors {
                    insertSwift(neighbor, into: &top, capacity: k)
                }
            }
        }

        if top.count < k {
            metrics?.pointee.exactFallbackCount &+= 1
            let result = fraudVoteCountExact(query: query, in: index, k: k)
            metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
            return result
        }
        var fraudVotes = 0
        for neighbor in top where orderedLabels[neighbor.recordIndex] == 1 {
            fraudVotes += 1
        }
        metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
        return fraudVotes
    }

    static func fraudVoteCountIVFPruned(
        query: [Int16],
        in index: ReferencesIndex,
        ivf: IVFIndex,
        centroidNeighbors: [rinha_neighbor_t],
        bboxMin: UnsafeBufferPointer<Int16>,
        bboxMax: UnsafeBufferPointer<Int16>,
        metrics: UnsafeMutablePointer<SearchMetrics>?,
        k: Int
    ) -> Int {
        let offsets = ivf.clusterOffsets
        let postings = ivf.postings
        let dim = index.header.dim
        var top = [Neighbor]()
        top.reserveCapacity(k)
        let shortlistStarted = metricStart(metrics)

        for centroid in centroidNeighbors {
            let cluster = Int(centroid.record_index)
            if top.count == k {
                let lowerBound = lowerBoundSquared(
                    query: query,
                    cluster: cluster,
                    bboxMin: bboxMin,
                    bboxMax: bboxMax,
                    stride: ivf.header.stride,
                    dim: dim
                )
                if lowerBound >= top[k - 1].distanceSquared {
                    continue
                }
            }

            let start = Int(offsets[cluster])
            let end = Int(offsets[cluster + 1])
            let candidateCount = end - start
            if candidateCount <= 0 {
                continue
            }

            let clusterNeighbors = exactNeighborsInIndexedCluster(
                query: query,
                index: index,
                postings: postings,
                start: start,
                end: end,
                k: k
            )

            for neighbor in clusterNeighbors {
                insertSwift(neighbor, into: &top, capacity: k)
            }
        }

        if top.count == k,
           let extraClusters = additionalClustersToScan(
                query: query,
                ivf: ivf,
                centroidNeighbors: centroidNeighbors,
                worstDistanceSquared: top[k - 1].distanceSquared,
                dim: dim
           ) {
            for extraCluster in extraClusters {
                if extraCluster.lowerBoundSquared >= top[k - 1].distanceSquared {
                    break
                }
                let start = Int(offsets[extraCluster.cluster])
                let end = Int(offsets[extraCluster.cluster + 1])
                let clusterNeighbors = exactNeighborsInIndexedCluster(
                    query: query,
                    index: index,
                    postings: postings,
                    start: start,
                    end: end,
                    k: k
                )
                for neighbor in clusterNeighbors {
                    insertSwift(neighbor, into: &top, capacity: k)
                }
            }
        }

        if top.count < k {
            metrics?.pointee.exactFallbackCount &+= 1
            let result = fraudVoteCountExact(query: query, in: index, k: k)
            metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
            return result
        }
        let labels = index.labels
        var fraudVotes = 0
        for neighbor in top where labels[neighbor.recordIndex] == 1 {
            fraudVotes += 1
        }
        metricRecord(shortlistStarted, keyPath: \.shortlistNs, metrics: metrics)
        return fraudVotes
    }

    @inline(__always)
    static func metricStart(_ metrics: UnsafeMutablePointer<SearchMetrics>?) -> UInt64 {
        metrics == nil ? 0 : DispatchTime.now().uptimeNanoseconds
    }

    @inline(__always)
    static func metricRecord(
        _ started: UInt64,
        keyPath: WritableKeyPath<SearchMetrics, UInt64>,
        metrics: UnsafeMutablePointer<SearchMetrics>?
    ) {
        guard let metrics, started != 0 else { return }
        metrics.pointee[keyPath: keyPath] &+= DispatchTime.now().uptimeNanoseconds - started
    }

    static func lowerBoundSquared(
        query: [Int16],
        cluster: Int,
        bboxMin: UnsafeBufferPointer<Int16>,
        bboxMax: UnsafeBufferPointer<Int16>,
        stride: Int,
        dim: Int
    ) -> Int64 {
        let base = cluster * stride
        var sum: Int64 = 0
        for lane in 0..<dim {
            let q = Int32(query[lane])
            let minValue = Int32(bboxMin[base + lane])
            let maxValue = Int32(bboxMax[base + lane])
            let diff: Int32
            if q < minValue {
                diff = minValue - q
            } else if q > maxValue {
                diff = q - maxValue
            } else {
                diff = 0
            }
            sum &+= Int64(diff &* diff)
        }
        return sum
    }

    struct ClusterLowerBound {
        let cluster: Int
        let lowerBoundSquared: Int64
    }

    static func additionalClustersToScan(
        query: [Int16],
        ivf: IVFIndex,
        centroidNeighbors: [rinha_neighbor_t],
        worstDistanceSquared: Int64,
        dim: Int
    ) -> [ClusterLowerBound]? {
        guard ivf.header.hasBoundingBoxes,
              let bboxMin = ivf.bboxMin,
              let bboxMax = ivf.bboxMax else {
            return nil
        }

        var visited = [Bool](repeating: false, count: ivf.header.clusterCount)
        for centroid in centroidNeighbors {
            visited[Int(centroid.record_index)] = true
        }

        var candidates = [ClusterLowerBound]()
        candidates.reserveCapacity(max(0, ivf.header.clusterCount - centroidNeighbors.count))
        for cluster in 0..<ivf.header.clusterCount where !visited[cluster] {
            let lowerBound = lowerBoundSquared(
                query: query,
                cluster: cluster,
                bboxMin: bboxMin,
                bboxMax: bboxMax,
                stride: ivf.header.stride,
                dim: dim
            )
            if lowerBound < worstDistanceSquared {
                candidates.append(
                    ClusterLowerBound(
                        cluster: cluster,
                        lowerBoundSquared: lowerBound
                    )
                )
            }
        }

        candidates.sort { lhs, rhs in
            if lhs.lowerBoundSquared == rhs.lowerBoundSquared {
                return lhs.cluster < rhs.cluster
            }
            return lhs.lowerBoundSquared < rhs.lowerBoundSquared
        }
        return candidates
    }

    @inline(__always)
    static func configSupportsPQ(
        pq: IVFPQIndex,
        ivf: IVFIndex,
        configRerankCandidates: Int?
    ) -> Bool {
        pq.header.count == ivf.header.count &&
        pq.header.stride == ivf.header.stride &&
        (configRerankCandidates ?? 0) > 5
    }
}
