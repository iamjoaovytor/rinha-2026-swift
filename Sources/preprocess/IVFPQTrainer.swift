func buildIVFPQ(
    orderedVectors: [Int16],
    count: Int,
    stride: Int,
    training: IVFPQTrainingConfig
) -> (subvectorCount: Int, subvectorWidth: Int, codebooks: [Int16], codes: [UInt8]) {
    precondition(training.subvectorCount > 0, "subvectorCount must be positive")
    precondition(stride % training.subvectorCount == 0, "stride must be divisible by subvectorCount")
    let subvectorCount = training.subvectorCount
    let subvectorWidth = stride / subvectorCount
    let centroidCount = 256
    let sampleCount = min(count, max(centroidCount, training.sampleCount))

    func sampleIndices() -> [Int] {
        guard count > sampleCount else {
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
    func distanceSquared(
        recordIndex: Int,
        subvector: Int,
        centroidBase: Int,
        codebooks: [Int16]
    ) -> Int64 {
        let recordBase = recordIndex * stride + subvector * subvectorWidth
        var sum: Int64 = 0
        for lane in 0..<subvectorWidth {
            let diff = Int32(orderedVectors[recordBase + lane]) - Int32(codebooks[centroidBase + lane])
            sum &+= Int64(diff &* diff)
        }
        return sum
    }

    func trainCodebook(subvector: Int) -> [Int16] {
        var codebook = [Int16](repeating: 0, count: centroidCount * subvectorWidth)
        guard !sampledRecordIndices.isEmpty else { return codebook }

        var rng = SplitMix64(seed: training.seed &+ UInt64(subvector) &* 0x9E3779B97F4A7C15)
        let firstRecordIndex = sampledRecordIndices[rng.nextInt(upperBound: sampledRecordIndices.count)]
        let firstBase = firstRecordIndex * stride + subvector * subvectorWidth
        for lane in 0..<subvectorWidth {
            codebook[lane] = orderedVectors[firstBase + lane]
        }

        var minDistances = [Int64](repeating: Int64.max, count: sampledRecordIndices.count)
        if centroidCount > 1 {
            for centroid in 1..<centroidCount {
                let previousBase = (centroid - 1) * subvectorWidth
                var total: Int64 = 0
                for (sampleOffset, recordIndex) in sampledRecordIndices.enumerated() {
                    let recordBase = recordIndex * stride + subvector * subvectorWidth
                    var sum: Int64 = 0
                    for lane in 0..<subvectorWidth {
                        let diff = Int32(orderedVectors[recordBase + lane]) - Int32(codebook[previousBase + lane])
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

                let targetBase = centroid * subvectorWidth
                let sourceBase = chosenRecordIndex * stride + subvector * subvectorWidth
                for lane in 0..<subvectorWidth {
                    codebook[targetBase + lane] = orderedVectors[sourceBase + lane]
                }
            }
        }

        let iterations = max(1, training.trainIterations)
        for _ in 0..<iterations {
            var sums = [Int64](repeating: 0, count: centroidCount * subvectorWidth)
            var counts = [Int](repeating: 0, count: centroidCount)

            for recordIndex in sampledRecordIndices {
                var bestCode = 0
                var bestDistance = Int64.max
                for code in 0..<centroidCount {
                    let centroidBase = code * subvectorWidth
                    let distance = distanceSquared(
                        recordIndex: recordIndex,
                        subvector: subvector,
                        centroidBase: centroidBase,
                        codebooks: codebook
                    )
                    if distance < bestDistance {
                        bestDistance = distance
                        bestCode = code
                    }
                }

                counts[bestCode] += 1
                let recordBase = recordIndex * stride + subvector * subvectorWidth
                let sumBase = bestCode * subvectorWidth
                for lane in 0..<subvectorWidth {
                    sums[sumBase + lane] += Int64(orderedVectors[recordBase + lane])
                }
            }

            for code in 0..<centroidCount {
                guard counts[code] > 0 else { continue }
                let centroidBase = code * subvectorWidth
                let sumBase = code * subvectorWidth
                let divisor = Int64(counts[code])
                for lane in 0..<subvectorWidth {
                    codebook[centroidBase + lane] = Int16(sums[sumBase + lane] / divisor)
                }
            }
        }

        return codebook
    }

    var codebooks = [Int16](repeating: 0, count: subvectorCount * centroidCount * subvectorWidth)
    var codes = [UInt8](repeating: 0, count: count * subvectorCount)

    for subvector in 0..<subvectorCount {
        let trained = trainCodebook(subvector: subvector)
        let codebookBase = subvector * centroidCount * subvectorWidth
        codebooks.replaceSubrange(codebookBase..<(codebookBase + trained.count), with: trained)

        for recordIndex in 0..<count {
            var bestCode = 0
            var bestDistance = Int64.max
            for code in 0..<centroidCount {
                let centroidBase = codebookBase + code * subvectorWidth
                let recordBase = recordIndex * stride + subvector * subvectorWidth
                var sum: Int64 = 0
                for lane in 0..<subvectorWidth {
                    let diff = Int32(orderedVectors[recordBase + lane]) - Int32(codebooks[centroidBase + lane])
                    sum &+= Int64(diff &* diff)
                }
                if sum < bestDistance {
                    bestDistance = sum
                    bestCode = code
                }
            }
            codes[recordIndex * subvectorCount + subvector] = UInt8(bestCode)
        }
    }

    return (subvectorCount, subvectorWidth, codebooks, codes)
}
