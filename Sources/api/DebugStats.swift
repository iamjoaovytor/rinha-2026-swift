import Foundation
import NIOConcurrencyHelpers

struct RequestPhaseMetrics: Sendable {
    var bodyCollectNs: UInt64 = 0
    var parseNs: UInt64 = 0
    var vectorizeNs: UInt64 = 0
    var searchNs: UInt64 = 0
    var searchCentroidNs: UInt64 = 0
    var searchShortlistNs: UInt64 = 0
    var searchExactFallbackCount: UInt64 = 0
    var searchAdaptiveExpandCount: UInt64 = 0
    var responseNs: UInt64 = 0
    var fastPath: Bool = false
    var fallbackPath: Bool = false
    var failed: Bool = false
}

private struct PhaseAccumulator: Sendable {
    var count: UInt64 = 0
    var totalNs: UInt64 = 0
    var maxNs: UInt64 = 0

    mutating func record(_ durationNs: UInt64) {
        count &+= 1
        totalNs &+= durationNs
        if durationNs > maxNs {
            maxNs = durationNs
        }
    }
}

final class DebugStatsCollector: @unchecked Sendable {
    private struct Snapshot: Sendable {
        var enabled: Bool
        var requests: UInt64 = 0
        var failures: UInt64 = 0
        var fastPathCount: UInt64 = 0
        var fallbackPathCount: UInt64 = 0
        var bodyCollect = PhaseAccumulator()
        var parse = PhaseAccumulator()
        var vectorize = PhaseAccumulator()
        var search = PhaseAccumulator()
        var searchCentroid = PhaseAccumulator()
        var searchShortlist = PhaseAccumulator()
        var searchExactFallbackCount: UInt64 = 0
        var searchAdaptiveExpandCount: UInt64 = 0
        var response = PhaseAccumulator()
    }

    private let enabled: Bool
    private let state: NIOLockedValueBox<Snapshot>

    init(enabled: Bool) {
        self.enabled = enabled
        self.state = NIOLockedValueBox(Snapshot(enabled: enabled))
    }

    var isEnabled: Bool { enabled }

    func record(_ metrics: RequestPhaseMetrics) {
        guard enabled else { return }
        state.withLockedValue { snapshot in
            snapshot.requests &+= 1
            if metrics.failed {
                snapshot.failures &+= 1
            }
            if metrics.fastPath {
                snapshot.fastPathCount &+= 1
            }
            if metrics.fallbackPath {
                snapshot.fallbackPathCount &+= 1
            }
            snapshot.bodyCollect.record(metrics.bodyCollectNs)
            snapshot.parse.record(metrics.parseNs)
            snapshot.vectorize.record(metrics.vectorizeNs)
            snapshot.search.record(metrics.searchNs)
            snapshot.searchCentroid.record(metrics.searchCentroidNs)
            snapshot.searchShortlist.record(metrics.searchShortlistNs)
            snapshot.searchExactFallbackCount &+= metrics.searchExactFallbackCount
            snapshot.searchAdaptiveExpandCount &+= metrics.searchAdaptiveExpandCount
            snapshot.response.record(metrics.responseNs)
        }
    }

    func reset() {
        guard enabled else { return }
        state.withLockedValue { snapshot in
            snapshot = Snapshot(enabled: true)
        }
    }

    func jsonData() throws -> Data {
        let snapshot = state.withLockedValue { $0 }
        let object: [String: Any] = [
            "enabled": snapshot.enabled,
            "requests": snapshot.requests,
            "failures": snapshot.failures,
            "fast_path_count": snapshot.fastPathCount,
            "fallback_path_count": snapshot.fallbackPathCount,
            "phases": [
                "body_collect": phaseDict(snapshot.bodyCollect),
                "parse": phaseDict(snapshot.parse),
                "vectorize": phaseDict(snapshot.vectorize),
                "search": phaseDict(snapshot.search),
                "search_centroid": phaseDict(snapshot.searchCentroid),
                "search_shortlist": phaseDict(snapshot.searchShortlist),
                "response": phaseDict(snapshot.response),
            ],
            "search_counters": [
                "exact_fallback_count": snapshot.searchExactFallbackCount,
                "adaptive_expand_count": snapshot.searchAdaptiveExpandCount,
            ],
        ]
        return try JSONSerialization.data(withJSONObject: object)
    }

    private func phaseDict(_ phase: PhaseAccumulator) -> [String: Any] {
        [
            "count": phase.count,
            "total_ns": phase.totalNs,
            "max_ns": phase.maxNs,
            "avg_ns": phase.count == 0 ? 0 : phase.totalNs / phase.count,
        ]
    }
}
