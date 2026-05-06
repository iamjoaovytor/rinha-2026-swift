import Foundation
import Testing
@testable import Domain

struct SearchTests {
    @Test func topKReturnsClosestVectorsInOrder() throws {
        // Three reference vectors with one differing lane each. Distances
        // squared should sort smallest-first.
        let records: [(vector: [Int16], label: UInt8)] = [
            (vector: [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 0),
            (vector: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 1),
            (vector: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 0)
        ]
        let url = try writeReferences(records)
        defer { try? FileManager.default.removeItem(at: url) }
        let index = try ReferencesIndex.load(path: url.path)

        let query: [Int16] = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let neighbors = KNN.topK(query: query, in: index, k: 3)

        #expect(neighbors.count == 3)
        #expect(neighbors[0].recordIndex == 2)
        #expect(neighbors[0].distanceSquared == 1)
        #expect(neighbors[1].recordIndex == 1)
        #expect(neighbors[1].distanceSquared == 16)
        #expect(neighbors[2].recordIndex == 0)
        #expect(neighbors[2].distanceSquared == 36)
    }

    @Test func topKTruncatesToK() throws {
        let records: [(vector: [Int16], label: UInt8)] = (0..<5).map { i in
            var v = [Int16](repeating: 0, count: 16)
            v[0] = Int16(i * 10)
            return (vector: v, label: 0)
        }
        let url = try writeReferences(records)
        defer { try? FileManager.default.removeItem(at: url) }
        let index = try ReferencesIndex.load(path: url.path)

        let query = [Int16](repeating: 0, count: 16)
        let neighbors = KNN.topK(query: query, in: index, k: 2)
        #expect(neighbors.count == 2)
        #expect(neighbors[0].recordIndex == 0)
        #expect(neighbors[1].recordIndex == 1)
    }

    @Test func nativeTopKMatchesSwiftOracle() throws {
        let records: [(vector: [Int16], label: UInt8)] = [
            (vector: [12, -3, 8, 0, 5, 1, 2, -7, 9, 4, 0, 3, 6, -2, 0, 0], label: 0),
            (vector: [11, -1, 9, 0, 4, 1, 3, -7, 10, 5, 0, 2, 7, -2, 0, 0], label: 1),
            (vector: [100, 40, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 0),
            (vector: [13, -2, 7, 0, 5, 1, 2, -6, 8, 4, 0, 3, 6, -1, 0, 0], label: 1),
            (vector: [-50, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0], label: 0),
        ]
        let url = try writeReferences(records)
        defer { try? FileManager.default.removeItem(at: url) }
        let index = try ReferencesIndex.load(path: url.path)

        let query: [Int16] = [12, -2, 8, 0, 5, 1, 2, -7, 9, 4, 0, 3, 6, -2, 0, 0]
        let native = KNN.topK(query: query, in: index, k: 4)
        let swift = KNN.topKSwift(query: query, in: index, k: 4)

        #expect(native == swift)
    }

    @Test func scoreFlagsMajorityFraudAndDenies() throws {
        let records: [(vector: [Int16], label: UInt8)] = [
            (vector: [Int16](repeating: 0, count: 16), label: 1),
            (vector: [Int16](repeating: 0, count: 16), label: 1),
            (vector: [Int16](repeating: 0, count: 16), label: 1),
            (vector: [Int16](repeating: 0, count: 16), label: 0),
            (vector: [Int16](repeating: 0, count: 16), label: 0)
        ]
        let url = try writeReferences(records)
        defer { try? FileManager.default.removeItem(at: url) }
        let index = try ReferencesIndex.load(path: url.path)

        let query = [Int16](repeating: 0, count: 16)
        let neighbors = KNN.topK(query: query, in: index, k: 5)
        let result = FraudScoring.score(neighbors: neighbors, index: index)
        #expect(abs(result.fraudScore - 0.6) < 1e-9)
        #expect(result.approved == false)
    }

    @Test func scoreApprovesWhenLegitMajority() throws {
        let records: [(vector: [Int16], label: UInt8)] = [
            (vector: [Int16](repeating: 0, count: 16), label: 0),
            (vector: [Int16](repeating: 0, count: 16), label: 0),
            (vector: [Int16](repeating: 0, count: 16), label: 0),
            (vector: [Int16](repeating: 0, count: 16), label: 1),
            (vector: [Int16](repeating: 0, count: 16), label: 1)
        ]
        let url = try writeReferences(records)
        defer { try? FileManager.default.removeItem(at: url) }
        let index = try ReferencesIndex.load(path: url.path)

        let query = [Int16](repeating: 0, count: 16)
        let neighbors = KNN.topK(query: query, in: index, k: 5)
        let result = FraudScoring.score(neighbors: neighbors, index: index)
        #expect(abs(result.fraudScore - 0.4) < 1e-9)
        #expect(result.approved == true)
    }

    @Test func ivfRoundTripAndFraudVoteMatchesExactWhenProbingAllClusters() throws {
        let records: [(vector: [Int16], label: UInt8)] = [
            (vector: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 0),
            (vector: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 1),
            (vector: [50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 1),
            (vector: [52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], label: 0),
        ]
        let referencesURL = try writeReferences(records)
        let ivfURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ref-\(UUID().uuidString).ivf")
        defer {
            try? FileManager.default.removeItem(at: referencesURL)
            try? FileManager.default.removeItem(at: ivfURL)
        }

        let centroids: [Int16] = [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        let bboxMin: [Int16] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        let bboxMax: [Int16] = [
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        let offsets: [UInt32] = [0, 2, 4]
        let postings: [UInt32] = [0, 1, 2, 3]
        let orderedVectors = records.flatMap(\.vector)
        let orderedLabels = records.map(\.label)
        try writeIVF(
            path: ivfURL,
            count: records.count,
            clusterCount: 2,
            centroids: centroids,
            bboxMin: bboxMin,
            bboxMax: bboxMax,
            offsets: offsets,
            postings: postings,
            orderedVectors: orderedVectors,
            orderedLabels: orderedLabels
        )

        let index = try ReferencesIndex.load(path: referencesURL.path)
        let ivf = try IVFIndex.load(path: ivfURL.path)

        #expect(ivf.header.clusterCount == 2)
        #expect(ivf.header.hasBoundingBoxes)
        #expect(ivf.header.hasClusterVectors)
        #expect(Array(ivf.bboxMin!) == bboxMin)
        #expect(Array(ivf.bboxMax!) == bboxMax)
        #expect(Array(ivf.clusterOffsets) == offsets)
        #expect(Array(ivf.postings) == postings)
        #expect(Array(ivf.orderedVectors!) == orderedVectors)
        #expect(Array(ivf.orderedLabels!) == orderedLabels)

        let query: [Int16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let exactFraudVotes = KNN.fraudVoteCount(query: query, in: index, k: 3)
        let ivfFraudVotes = KNN.fraudVoteCount(
            query: query,
            in: index,
            ivf: ivf,
            config: SearchConfig(nprobe: 2),
            k: 3
        )

        #expect(ivfFraudVotes == exactFraudVotes)
    }

    @Test func adaptiveIVFExpandsAmbiguousVotesBeforeDeciding() throws {
        let cluster0: [(vector: [Int16], label: UInt8)] = [
            ([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            ([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            ([30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
            ([40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
            ([50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ]
        let cluster1: [(vector: [Int16], label: UInt8)] = [
            ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            ([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            ([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
            ([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ]
        let records = cluster0 + cluster1
        let referencesURL = try writeReferences(records)
        let ivfURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ref-\(UUID().uuidString).ivf")
        defer {
            try? FileManager.default.removeItem(at: referencesURL)
            try? FileManager.default.removeItem(at: ivfURL)
        }

        let centroids: [Int16] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        let bboxMin = centroids
        let bboxMax: [Int16] = [
            50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        let offsets: [UInt32] = [0, 5, 10]
        let postings: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let orderedVectors = records.flatMap(\.vector)
        let orderedLabels = records.map(\.label)
        try writeIVF(
            path: ivfURL,
            count: records.count,
            clusterCount: 2,
            centroids: centroids,
            bboxMin: bboxMin,
            bboxMax: bboxMax,
            offsets: offsets,
            postings: postings,
            orderedVectors: orderedVectors,
            orderedLabels: orderedLabels
        )

        let index = try ReferencesIndex.load(path: referencesURL.path)
        let ivf = try IVFIndex.load(path: ivfURL.path)
        let query: [Int16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        let exactFraudVotes = KNN.fraudVoteCount(query: query, in: index, k: 5)
        let singleProbeVotes = KNN.fraudVoteCount(
            query: query,
            in: index,
            ivf: ivf,
            config: SearchConfig(nprobe: 1),
            k: 5
        )
        let adaptiveVotes = KNN.fraudVoteCount(
            query: query,
            in: index,
            ivf: ivf,
            config: SearchConfig(nprobe: 2, initialNprobe: 1),
            k: 5
        )

        #expect(singleProbeVotes == 2)
        #expect(exactFraudVotes == 3)
        #expect(adaptiveVotes == exactFraudVotes)
    }


    private func writeReferences(_ records: [(vector: [Int16], label: UInt8)]) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ref-\(UUID().uuidString).bin")
        var data = Data()
        let count = records.count

        data.append(contentsOf: [0x52, 0x4E, 0x48, 0x41])
        appendLE(UInt32(1), to: &data)
        appendLE(UInt64(count), to: &data)
        appendLE(UInt32(14), to: &data)
        appendLE(UInt32(16), to: &data)
        appendLE(Int32(8192), to: &data)
        appendLE(UInt32(0), to: &data)
        data.append(Data(repeating: 0, count: 32)) // sha placeholder
        appendLE(Int64(0), to: &data)
        if data.count < 128 {
            data.append(Data(repeating: 0, count: 128 - data.count))
        }

        for r in records { data.append(r.label) }
        padTo(alignment: 4096, in: &data)

        for i in 0..<count { appendLE(UInt32(i), to: &data) }
        padTo(alignment: 4096, in: &data)

        for r in records {
            precondition(r.vector.count == 16)
            for lane in r.vector { appendLE(lane, to: &data) }
        }

        try data.write(to: url)
        return url
    }

    private func writeIVF(
        path: URL,
        count: Int,
        clusterCount: Int,
        centroids: [Int16],
        bboxMin: [Int16]? = nil,
        bboxMax: [Int16]? = nil,
        offsets: [UInt32],
        postings: [UInt32],
        orderedVectors: [Int16]? = nil,
        orderedLabels: [UInt8]? = nil,
        version: UInt32 = 3
    ) throws {
        var data = Data()
        data.append(contentsOf: [0x52, 0x49, 0x56, 0x46])
        appendLE(version, to: &data)
        appendLE(UInt64(count), to: &data)
        appendLE(UInt32(clusterCount), to: &data)
        appendLE(UInt32(16), to: &data)
        appendLE(Int64(0), to: &data)
        if data.count < 64 {
            data.append(Data(repeating: 0, count: 64 - data.count))
        }

        for lane in centroids { appendLE(lane, to: &data) }
        padTo(alignment: 4096, in: &data)

        if version >= 2 {
            let bboxMin = bboxMin ?? centroids
            let bboxMax = bboxMax ?? centroids
            for lane in bboxMin { appendLE(lane, to: &data) }
            padTo(alignment: 4096, in: &data)
            for lane in bboxMax { appendLE(lane, to: &data) }
            padTo(alignment: 4096, in: &data)
        }

        for offset in offsets { appendLE(offset, to: &data) }
        padTo(alignment: 4096, in: &data)

        for posting in postings { appendLE(posting, to: &data) }
        if version >= 3 {
            padTo(alignment: 4096, in: &data)
            for lane in (orderedVectors ?? []) { appendLE(lane, to: &data) }
            padTo(alignment: 4096, in: &data)
            data.append(contentsOf: orderedLabels ?? [])
        }
        try data.write(to: path)
    }

    private func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }

    private func padTo(alignment: Int, in data: inout Data) {
        let rem = data.count % alignment
        if rem != 0 { data.append(Data(repeating: 0, count: alignment - rem)) }
    }
}
