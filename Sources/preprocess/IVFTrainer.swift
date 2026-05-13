func buildIVF(
    lanes: [Int16],
    labels: [UInt8],
    count: Int,
    dim: Int,
    stride: Int,
    clusterCount requestedClusterCount: Int,
    training: IVFTrainingConfig
) -> (clusterCount: Int, centroids: [Int16], bboxMin: [Int16], bboxMax: [Int16], offsets: [UInt32], postings: [UInt32], orderedVectors: [Int16], orderedLabels: [UInt8]) {
    let clusterCount = max(1, min(requestedClusterCount, count))
    let sampleCount = min(count, max(clusterCount, training.sampleCount))

    func sampleIndices() -> [Int] {
        if sampleCount >= count {
            return Array(0..<count)
        }
        var indices = [Int]()
        indices.reserveCapacity(sampleCount)
        let step = Double(count) / Double(sampleCount)
        for i in 0..<sampleCount {
            indices.append(min(count - 1, Int(Double(i) * step)))
        }
        return indices
    }

    let sampledRecordIndices = sampleIndices()

    @inline(__always)
    func nearestCluster(for recordIndex: Int, centroids: [Int16]) -> (cluster: Int, distance: Int64) {
        let recordBase = recordIndex * stride
        var bestCluster = 0
        var bestDistance = Int64.max
        for cluster in 0..<clusterCount {
            let centroidBase = cluster * stride
            var sum: Int64 = 0
            for lane in 0..<dim {
                let diff = Int32(lanes[recordBase + lane]) - Int32(centroids[centroidBase + lane])
                sum &+= Int64(diff &* diff)
            }
            if sum < bestDistance {
                bestDistance = sum
                bestCluster = cluster
            }
        }
        return (bestCluster, bestDistance)
    }

    func initializeCentroids(seed: UInt64) -> [Int16] {
        var centroids = [Int16](repeating: 0, count: clusterCount * stride)
        if !training.useKMeansPP {
            for cluster in 0..<clusterCount {
                let recordIndex = sampledRecordIndices[cluster * sampledRecordIndices.count / clusterCount]
                let sourceBase = recordIndex * stride
                let targetBase = cluster * stride
                for lane in 0..<dim {
                    centroids[targetBase + lane] = lanes[sourceBase + lane]
                }
            }
            return centroids
        }

        var rng = SplitMix64(seed: seed)
        let firstSample = sampledRecordIndices[rng.nextInt(upperBound: sampledRecordIndices.count)]
        for lane in 0..<dim {
            centroids[lane] = lanes[firstSample * stride + lane]
        }

        var minDistances = [Int64](repeating: Int64.max, count: sampledRecordIndices.count)
        for cluster in 1..<clusterCount {
            var total: Int64 = 0
            let previousCentroidBase = (cluster - 1) * stride
            for (sampleOffset, recordIndex) in sampledRecordIndices.enumerated() {
                let recordBase = recordIndex * stride
                var sum: Int64 = 0
                for lane in 0..<dim {
                    let diff = Int32(lanes[recordBase + lane]) - Int32(centroids[previousCentroidBase + lane])
                    sum &+= Int64(diff &* diff)
                }
                if sum < minDistances[sampleOffset] {
                    minDistances[sampleOffset] = sum
                }
                total &+= max(1, minDistances[sampleOffset])
            }

            let chosenRecordIndex: Int
            if total <= 0 {
                chosenRecordIndex = sampledRecordIndices[rng.nextInt(upperBound: sampledRecordIndices.count)]
            } else {
                let target = rng.nextInt64(upperBound: total)
                var cumulative: Int64 = 0
                var chosenSampleOffset = sampledRecordIndices.count - 1
                for (sampleOffset, distance) in minDistances.enumerated() {
                    cumulative &+= max(1, distance)
                    if cumulative > target {
                        chosenSampleOffset = sampleOffset
                        break
                    }
                }
                chosenRecordIndex = sampledRecordIndices[chosenSampleOffset]
            }

            let targetBase = cluster * stride
            let sourceBase = chosenRecordIndex * stride
            for lane in 0..<dim {
                centroids[targetBase + lane] = lanes[sourceBase + lane]
            }
        }

        return centroids
    }

    func refine(
        centroids initialCentroids: [Int16],
        recordIndices: [Int],
        iterations: Int
    ) -> ([Int16], Int64) {
        var centroids = initialCentroids
        var inertia: Int64 = .max
        guard !recordIndices.isEmpty else { return (centroids, inertia) }

        for _ in 0..<iterations {
            var sums = [Int64](repeating: 0, count: clusterCount * dim)
            var counts = [Int](repeating: 0, count: clusterCount)
            inertia = 0

            for recordIndex in recordIndices {
                let nearest = nearestCluster(for: recordIndex, centroids: centroids)
                let cluster = nearest.cluster
                counts[cluster] += 1
                inertia &+= nearest.distance
                let recordBase = recordIndex * stride
                let sumBase = cluster * dim
                for lane in 0..<dim {
                    sums[sumBase + lane] += Int64(lanes[recordBase + lane])
                }
            }

            for cluster in 0..<clusterCount {
                guard counts[cluster] > 0 else { continue }
                let centroidBase = cluster * stride
                let sumBase = cluster * dim
                let divisor = Int64(counts[cluster])
                for lane in 0..<dim {
                    centroids[centroidBase + lane] = Int16(sums[sumBase + lane] / divisor)
                }
                for lane in dim..<stride {
                    centroids[centroidBase + lane] = 0
                }
            }
        }

        return (centroids, inertia)
    }

    var bestCentroids = initializeCentroids(seed: training.seed)
    var bestInertia: Int64 = .max

    for restart in 0..<max(1, training.restarts) {
        let seed = training.seed &+ UInt64(restart) &* 0x9E3779B97F4A7C15
        let initialCentroids = initializeCentroids(seed: seed)
        let refined = refine(
            centroids: initialCentroids,
            recordIndices: sampledRecordIndices,
            iterations: max(1, training.trainIterations)
        )
        if refined.1 < bestInertia {
            bestCentroids = refined.0
            bestInertia = refined.1
        }
    }

    var centroids = bestCentroids
    if training.fullRefineIterations > 0 {
        let fullIndices = Array(0..<count)
        centroids = refine(
            centroids: centroids,
            recordIndices: fullIndices,
            iterations: training.fullRefineIterations
        ).0
    }

    for lane in dim..<stride {
        for cluster in 0..<clusterCount {
            centroids[cluster * stride + lane] = 0
        }
    }

    var assignments = [UInt16](repeating: 0, count: count)
    var clusterSizes = [Int](repeating: 0, count: clusterCount)
    for recordIndex in 0..<count {
        let cluster = nearestCluster(for: recordIndex, centroids: centroids).cluster
        assignments[recordIndex] = UInt16(cluster)
        clusterSizes[cluster] += 1
    }

    var offsets = [UInt32](repeating: 0, count: clusterCount + 1)
    for cluster in 0..<clusterCount {
        offsets[cluster + 1] = offsets[cluster] + UInt32(clusterSizes[cluster])
    }

    var cursors = offsets
    var postings = [UInt32](repeating: 0, count: count)
    for recordIndex in 0..<count {
        let cluster = Int(assignments[recordIndex])
        let position = Int(cursors[cluster])
        postings[position] = UInt32(recordIndex)
        cursors[cluster] += 1
    }

    var orderedVectors = [Int16](repeating: 0, count: count * stride)
    var orderedLabels = [UInt8](repeating: 0, count: count)
    for orderedIndex in 0..<count {
        let recordIndex = Int(postings[orderedIndex])
        let sourceBase = recordIndex * stride
        let targetBase = orderedIndex * stride
        for lane in 0..<stride {
            orderedVectors[targetBase + lane] = lanes[sourceBase + lane]
        }
        orderedLabels[orderedIndex] = labels[recordIndex]
    }

    var bboxMin = [Int16](repeating: .max, count: clusterCount * stride)
    var bboxMax = [Int16](repeating: .min, count: clusterCount * stride)
    for cluster in 0..<clusterCount {
        let base = cluster * stride
        for lane in dim..<stride {
            bboxMin[base + lane] = 0
            bboxMax[base + lane] = 0
        }
    }
    for recordIndex in 0..<count {
        let cluster = Int(assignments[recordIndex])
        let clusterBase = cluster * stride
        let recordBase = recordIndex * stride
        for lane in 0..<dim {
            let value = lanes[recordBase + lane]
            if value < bboxMin[clusterBase + lane] {
                bboxMin[clusterBase + lane] = value
            }
            if value > bboxMax[clusterBase + lane] {
                bboxMax[clusterBase + lane] = value
            }
        }
    }
    for cluster in 0..<clusterCount {
        let base = cluster * stride
        if bboxMin[base] == .max {
            for lane in 0..<dim {
                let centroidValue = centroids[base + lane]
                bboxMin[base + lane] = centroidValue
                bboxMax[base + lane] = centroidValue
            }
        }
    }

    return (clusterCount, centroids, bboxMin, bboxMax, offsets, postings, orderedVectors, orderedLabels)
}
