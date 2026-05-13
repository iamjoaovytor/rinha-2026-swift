import Foundation

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
