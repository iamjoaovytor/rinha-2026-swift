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

    private func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }

    private func padTo(alignment: Int, in data: inout Data) {
        let rem = data.count % alignment
        if rem != 0 { data.append(Data(repeating: 0, count: alignment - rem)) }
    }
}
