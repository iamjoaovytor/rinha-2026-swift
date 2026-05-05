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
    static let formatVersion: UInt32 = 1
    static let dim: UInt32 = 14
    static let stride: UInt32 = 16
    static let layoutAoS: UInt32 = 0
    static let headerBytes = 128
    static let pageAlignment = 4096
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

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    die("usage: preprocess <input.json[.gz]> <output.bin>", code: 2)
}
let inputPath = arguments[1]
let outputPath = arguments[2]

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
