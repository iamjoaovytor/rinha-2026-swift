import Domain
import Foundation

// Phase 2 — preprocessor: references.json[.gz] → references.bin
//
// Pipeline (offline, runs outside the runtime container):
//   1. Read the input file. If suffixed `.gz` it is decompressed via
//      `gunzip -c` to avoid linking zlib.
//   2. Parse the top-level JSON array of `{"vector":[14 nums],
//      "label":"fraud"|"legit"}` via `JSONSerialization`.
//   3. Quantize each vector to 16-lane Int16 (scale=8192) via
//      `Domain.Vectorizer.quantize`.
//   4. Write `references.bin`: 128-byte header + page-aligned blocks of
//      labels (u8), orig_ids (u32 LE), vectors (i16 LE, AoS, stride 16).
//
// SHA-256 of the gz lives in `resources/references.sha256` alongside the
// dataset, captured via `shasum -a 256`. We keep the 32-byte slot in the
// header reserved for it but write zeros — adding a Crypto dependency for
// a one-off offline tool isn't worth the build cost.

enum Preprocess {
    static let magic: [UInt8] = [0x52, 0x4E, 0x48, 0x41] // "RNHA"
    static let formatVersion: UInt32 = 14
    static let dim: UInt32 = 14
    static let stride: UInt32 = 16
    static let layoutAoS: UInt32 = 0
    static let headerBytes = 128
    static let pageAlignment = 4096
    static let defaultIVFClusters = 1024
    static let defaultIVFTrainSample = 262_144
    static let defaultIVFTrainIterations = 8
    static let defaultIVFFullRefineIterations = 0
    static let defaultIVFRestarts = 1
    static let defaultIVFSeed: UInt64 = 42
    static let defaultIVFPQTrainSample = 131_072
    static let defaultIVFPQTrainIterations = 8
    static let defaultIVFPQSubvectorCount = 4
    static let defaultIVFPQSeed: UInt64 = 42
}

struct IVFTrainingConfig {
    let sampleCount: Int
    let trainIterations: Int
    let fullRefineIterations: Int
    let restarts: Int
    let seed: UInt64
    let useKMeansPP: Bool
}

struct IVFPQTrainingConfig {
    let enabled: Bool
    let sampleCount: Int
    let trainIterations: Int
    let subvectorCount: Int
    let seed: UInt64
}

struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func nextInt(upperBound: Int) -> Int {
        precondition(upperBound > 0)
        return Int(next() % UInt64(upperBound))
    }

    mutating func nextInt64(upperBound: Int64) -> Int64 {
        precondition(upperBound > 0)
        return Int64(next() % UInt64(upperBound))
    }
}

func die(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("preprocess: \(message)\n".utf8))
    exit(code)
}

func readInput(_ path: String) throws -> Data {
    let url = URL(fileURLWithPath: path)
    if path.hasSuffix(".gz") {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/gunzip")
        process.arguments = ["-c", path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.standardError
        try process.run()
        let payload = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        if process.terminationStatus != 0 {
            die("gunzip failed with status \(process.terminationStatus)")
        }
        return payload
    }
    return try Data(contentsOf: url)
}

extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }

    mutating func padTo(alignment: Int) {
        let rem = count % alignment
        if rem != 0 {
            append(Data(repeating: 0, count: alignment - rem))
        }
    }
}

func decodeNumber(_ value: Any) -> Double? {
    if let d = value as? Double { return d }
    if let n = value as? NSNumber { return n.doubleValue }
    if let i = value as? Int { return Double(i) }
    return nil
}

func ivfPath(for outputPath: String) -> String {
    let url = URL(fileURLWithPath: outputPath)
    return url.deletingPathExtension().appendingPathExtension("ivf").path
}

func ivfpqPath(for outputPath: String) -> String {
    let url = URL(fileURLWithPath: outputPath)
    return url.deletingPathExtension().appendingPathExtension("pq").path
}

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

func writeIVF(
    path: String,
    count: Int,
    stride: Int,
    clusterCount: Int,
    centroids: [Int16],
    bboxMin: [Int16],
    bboxMax: [Int16],
    offsets: [UInt32],
    postings: [UInt32],
    orderedVectors: [Int16],
    orderedLabels: [UInt8],
    createdUnix: Int64
) throws {
    var output = Data()
    output.append(contentsOf: [0x52, 0x49, 0x56, 0x46]) // "RIVF"
    output.appendLE(UInt32(3))
    output.appendLE(UInt64(count))
    output.appendLE(UInt32(clusterCount))
    output.appendLE(UInt32(stride))
    output.appendLE(createdUnix)
    output.padTo(alignment: IVFHeader.bytes)

    centroids.withUnsafeBufferPointer { buffer in
        let byteCount = centroids.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    bboxMin.withUnsafeBufferPointer { buffer in
        let byteCount = bboxMin.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    bboxMax.withUnsafeBufferPointer { buffer in
        let byteCount = bboxMax.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    offsets.withUnsafeBufferPointer { buffer in
        let byteCount = offsets.count * MemoryLayout<UInt32>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    postings.withUnsafeBufferPointer { buffer in
        let byteCount = postings.count * MemoryLayout<UInt32>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    orderedVectors.withUnsafeBufferPointer { buffer in
        let byteCount = orderedVectors.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFHeader.pageAlignment)

    orderedLabels.withUnsafeBufferPointer { buffer in
        output.append(buffer.baseAddress!, count: orderedLabels.count)
    }

    try output.write(to: URL(fileURLWithPath: path))
}

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

func writeIVFPQ(
    path: String,
    count: Int,
    stride: Int,
    subvectorCount: Int,
    subvectorWidth: Int,
    codebooks: [Int16],
    codes: [UInt8],
    createdUnix: Int64
) throws {
    var output = Data()
    output.append(contentsOf: [0x52, 0x56, 0x51, 0x50]) // "RVQP"
    output.appendLE(UInt32(1))
    output.appendLE(UInt64(count))
    output.appendLE(UInt32(stride))
    output.appendLE(UInt32(subvectorCount))
    output.appendLE(UInt32(subvectorWidth))
    output.appendLE(UInt32(0))
    output.appendLE(createdUnix)
    output.padTo(alignment: IVFPQHeader.bytes)

    codebooks.withUnsafeBufferPointer { buffer in
        let byteCount = codebooks.count * MemoryLayout<Int16>.size
        buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            output.append(ptr, count: byteCount)
        }
    }
    output.padTo(alignment: IVFPQHeader.pageAlignment)

    codes.withUnsafeBufferPointer { buffer in
        output.append(buffer.baseAddress!, count: codes.count)
    }

    try output.write(to: URL(fileURLWithPath: path))
}

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    die("usage: preprocess <input.json[.gz]> <output.bin>", code: 2)
}
let inputPath = arguments[1]
let outputPath = arguments[2]
let env = ProcessInfo.processInfo.environment
let ivfClusterCount = env["IVF_CLUSTERS"].flatMap(Int.init) ?? Preprocess.defaultIVFClusters
let ivfTrainingConfig = IVFTrainingConfig(
    sampleCount: env["IVF_TRAIN_SAMPLE"].flatMap(Int.init) ?? Preprocess.defaultIVFTrainSample,
    trainIterations: env["IVF_TRAIN_ITERS"].flatMap(Int.init) ?? Preprocess.defaultIVFTrainIterations,
    fullRefineIterations: env["IVF_FULL_REFINE_ITERS"].flatMap(Int.init) ?? Preprocess.defaultIVFFullRefineIterations,
    restarts: env["IVF_RESTARTS"].flatMap(Int.init) ?? Preprocess.defaultIVFRestarts,
    seed: env["IVF_SEED"].flatMap(UInt64.init) ?? Preprocess.defaultIVFSeed,
    useKMeansPP: env["IVF_KMEANSPP"].map { $0 != "0" } ?? true
)
let ivfpqTrainingConfig = IVFPQTrainingConfig(
    enabled: env["IVFPQ_BUILD"] == "1",
    sampleCount: env["IVFPQ_TRAIN_SAMPLE"].flatMap(Int.init) ?? Preprocess.defaultIVFPQTrainSample,
    trainIterations: env["IVFPQ_TRAIN_ITERS"].flatMap(Int.init) ?? Preprocess.defaultIVFPQTrainIterations,
    subvectorCount: env["IVFPQ_SUBVECTORS"].flatMap(Int.init) ?? Preprocess.defaultIVFPQSubvectorCount,
    seed: env["IVFPQ_SEED"].flatMap(UInt64.init) ?? Preprocess.defaultIVFPQSeed
)

let started = Date()
FileHandle.standardError.write(Data("preprocess: reading \(inputPath)\n".utf8))

let rawJSON = try readInput(inputPath)
let sha256 = Data(repeating: 0, count: 32)
FileHandle.standardError.write(Data("preprocess: decoded \(rawJSON.count) bytes; parsing JSON\n".utf8))

let parsed = try JSONSerialization.jsonObject(with: rawJSON, options: [])
guard let records = parsed as? [[String: Any]] else {
    die("expected JSON array of objects, got \(type(of: parsed))", code: 3)
}
let count = records.count
FileHandle.standardError.write(Data("preprocess: \(count) records\n".utf8))

let vectorizer = Vectorizer()
var labels = [UInt8]()
labels.reserveCapacity(count)
var origIds = [UInt32]()
origIds.reserveCapacity(count)
var lanes = [Int16]()
lanes.reserveCapacity(count * Int(Preprocess.stride))

var fraudCount = 0
for (index, record) in records.enumerated() {
    guard let vectorAny = record["vector"] as? [Any] else {
        die("record \(index): missing or malformed `vector`", code: 4)
    }
    guard vectorAny.count == 14 else {
        die("record \(index): vector has \(vectorAny.count) dims, expected 14", code: 4)
    }
    var doubles = [Double](repeating: 0, count: 14)
    for (i, raw) in vectorAny.enumerated() {
        guard let d = decodeNumber(raw) else {
            die("record \(index) dim \(i): not a number", code: 4)
        }
        doubles[i] = d
    }
    guard let labelString = record["label"] as? String else {
        die("record \(index): missing or malformed `label`", code: 4)
    }
    let labelByte: UInt8
    switch labelString {
    case "legit": labelByte = 0
    case "fraud": labelByte = 1; fraudCount += 1
    default: die("record \(index): unknown label `\(labelString)`", code: 4)
    }
    labels.append(labelByte)
    origIds.append(UInt32(index))
    let quantized = vectorizer.quantize(doubles)
    lanes.append(contentsOf: quantized)
}

FileHandle.standardError.write(Data("preprocess: fraud=\(fraudCount), legit=\(count - fraudCount)\n".utf8))

// Build header (128 bytes total).
var output = Data()
output.reserveCapacity(
    Preprocess.headerBytes
        + count
        + count * 4
        + count * Int(Preprocess.stride) * 2
        + Preprocess.pageAlignment * 3
)

output.append(contentsOf: Preprocess.magic)        // 0..4
output.appendLE(Preprocess.formatVersion)          // 4..8
output.appendLE(UInt64(count))                     // 8..16
output.appendLE(Preprocess.dim)                    // 16..20
output.appendLE(Preprocess.stride)                 // 20..24
output.appendLE(Int32(vectorizer.scale))           // 24..28
output.appendLE(Preprocess.layoutAoS)              // 28..32
output.append(sha256)                              // 32..64
output.appendLE(Int64(started.timeIntervalSince1970)) // 64..72
output.padTo(alignment: Preprocess.headerBytes)    // header padding to 128

precondition(output.count == Preprocess.headerBytes)

// labels[count]
labels.withUnsafeBufferPointer { buffer in
    output.append(buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: count) { $0 }, count: count)
}
output.padTo(alignment: Preprocess.pageAlignment)

// orig_ids[count] (u32 LE; host is little-endian on all targets we ship)
origIds.withUnsafeBufferPointer { buffer in
    let byteCount = count * MemoryLayout<UInt32>.size
    buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
        output.append(ptr, count: byteCount)
    }
}
output.padTo(alignment: Preprocess.pageAlignment)

// vectors[count*16] (i16 LE, AoS)
lanes.withUnsafeBufferPointer { buffer in
    let byteCount = lanes.count * MemoryLayout<Int16>.size
    buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
        output.append(ptr, count: byteCount)
    }
}

try output.write(to: URL(fileURLWithPath: outputPath))

FileHandle.standardError.write(Data(
    "preprocess: building ivf (\(ivfClusterCount) clusters, sample=\(ivfTrainingConfig.sampleCount), iters=\(ivfTrainingConfig.trainIterations), full_refine=\(ivfTrainingConfig.fullRefineIterations), restarts=\(ivfTrainingConfig.restarts), kmeanspp=\(ivfTrainingConfig.useKMeansPP ? 1 : 0))\n".utf8
))
let ivfStarted = Date()
let ivf = buildIVF(
    lanes: lanes,
    labels: labels,
    count: count,
    dim: Int(Preprocess.dim),
    stride: Int(Preprocess.stride),
    clusterCount: ivfClusterCount,
    training: ivfTrainingConfig
)
let ivfOutputPath = ivfPath(for: outputPath)
try writeIVF(
    path: ivfOutputPath,
    count: count,
    stride: Int(Preprocess.stride),
    clusterCount: ivf.clusterCount,
    centroids: ivf.centroids,
    bboxMin: ivf.bboxMin,
    bboxMax: ivf.bboxMax,
    offsets: ivf.offsets,
    postings: ivf.postings,
    orderedVectors: ivf.orderedVectors,
    orderedLabels: ivf.orderedLabels,
    createdUnix: Int64(started.timeIntervalSince1970)
)
let ivfElapsed = Date().timeIntervalSince(ivfStarted)
FileHandle.standardError.write(Data(
    "preprocess: wrote \(ivfOutputPath) (\(ivf.postings.count) postings, \(ivf.clusterCount) clusters) in \(String(format: "%.2f", ivfElapsed))s\n".utf8
))

if ivfpqTrainingConfig.enabled {
    FileHandle.standardError.write(Data(
        "preprocess: building ivfpq (subvectors=\(ivfpqTrainingConfig.subvectorCount), sample=\(ivfpqTrainingConfig.sampleCount), iters=\(ivfpqTrainingConfig.trainIterations))\n".utf8
    ))
    let ivfpqStarted = Date()
    let ivfpq = buildIVFPQ(
        orderedVectors: ivf.orderedVectors,
        count: count,
        stride: Int(Preprocess.stride),
        training: ivfpqTrainingConfig
    )
    let ivfpqOutputPath = ivfpqPath(for: outputPath)
    try writeIVFPQ(
        path: ivfpqOutputPath,
        count: count,
        stride: Int(Preprocess.stride),
        subvectorCount: ivfpq.subvectorCount,
        subvectorWidth: ivfpq.subvectorWidth,
        codebooks: ivfpq.codebooks,
        codes: ivfpq.codes,
        createdUnix: Int64(started.timeIntervalSince1970)
    )
    let ivfpqElapsed = Date().timeIntervalSince(ivfpqStarted)
    FileHandle.standardError.write(Data(
        "preprocess: wrote \(ivfpqOutputPath) (\(ivfpq.codes.count) codes, \(ivfpq.subvectorCount) subvectors) in \(String(format: "%.2f", ivfpqElapsed))s\n".utf8
    ))
}

let elapsed = Date().timeIntervalSince(started)
FileHandle.standardError.write(Data(
    "preprocess: wrote \(outputPath) (\(output.count) bytes) in \(String(format: "%.2f", elapsed))s\n".utf8
))

// Round-trip validation: reopen, read header, decode first record's vector.
let written = try Data(contentsOf: URL(fileURLWithPath: outputPath))
guard written.count >= Preprocess.headerBytes else {
    die("written file shorter than header", code: 6)
}
let countCheck = written.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self).littleEndian }
guard Int(countCheck) == count else {
    die("header count mismatch: \(countCheck) vs \(count)", code: 6)
}
let labelsBase = Preprocess.headerBytes
let labelCheck = written[labelsBase]
guard labelCheck == labels[0] else {
    die("first label mismatch", code: 6)
}
let origIdsBase = (labelsBase + count + Preprocess.pageAlignment - 1)
    / Preprocess.pageAlignment * Preprocess.pageAlignment
let firstOrigId = written.withUnsafeBytes {
    $0.load(fromByteOffset: origIdsBase, as: UInt32.self).littleEndian
}
guard firstOrigId == 0 else {
    die("first orig_id mismatch: \(firstOrigId)", code: 6)
}
let vectorsBase = (origIdsBase + count * 4 + Preprocess.pageAlignment - 1)
    / Preprocess.pageAlignment * Preprocess.pageAlignment
let firstLane = written.withUnsafeBytes {
    $0.load(fromByteOffset: vectorsBase, as: Int16.self).littleEndian
}
guard firstLane == lanes[0] else {
    die("first lane mismatch: \(firstLane) vs \(lanes[0])", code: 6)
}
FileHandle.standardError.write(Data("preprocess: round-trip OK\n".utf8))
