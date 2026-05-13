import Domain
import Foundation

enum SearchWarmup {
    static func run(loaded: LoadedState) {
        let warmupCount = ProcessInfo.processInfo.environment["WARMUP_COUNT"].flatMap(Int.init) ?? 5_000
        guard warmupCount > 0 else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        let count = loaded.index.header.count
        let stride = loaded.index.header.stride
        let basePtr = loaded.index.vectors.baseAddress!
        var rng = SplitMix64(seed: 0xCAFE_BABE_DEAD_BEEF)
        var sink = 0
        for _ in 0..<warmupCount {
            let recIdx = Int(rng.next() % UInt64(count))
            var query = [Int16](repeating: 0, count: 16)
            for lane in 0..<stride {
                query[lane] = basePtr[recIdx * stride + lane]
            }
            query[0] = query[0] &+ 17
            query[7] = query[7] &+ 23
            sink &+= KNN.fraudVoteCount(
                query: query,
                in: loaded.index,
                ivf: loaded.ivf,
                pq: loaded.pq,
                config: loaded.searchConfig,
                k: 5
            )
        }
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        FileHandle.standardError.write(Data("warmup: \(warmupCount) queries in \(String(format: "%.1f", elapsedMs)) ms (sink=\(sink))\n".utf8))
    }

    private struct SplitMix64 {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }
}
