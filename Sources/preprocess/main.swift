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
