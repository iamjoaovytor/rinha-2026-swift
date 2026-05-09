import Domain
import Foundation

struct TestData: Decodable {
    let entries: [Entry]
    struct Entry: Decodable {
        let request: FraudRequest
        let expected_approved: Bool
        let expected_fraud_score: Double
    }
}

@main
struct Evaluator {
    static func main() throws {
        let args = CommandLine.arguments
        let referencesPath = args.count > 1 ? args[1] : "resources/references.bin"
        let testDataPath = args.count > 2 ? args[2] : "test/test-data.json"
        let mccPath = args.count > 3 ? args[3] : "resources/mcc_risk.json"
        let nprobe = ProcessInfo.processInfo.environment["IVF_NPROBE"].flatMap(Int.init) ?? 12
        let initialNprobe = ProcessInfo.processInfo.environment["IVF_INITIAL_NPROBE"].flatMap(Int.init)
        let rerank = ProcessInfo.processInfo.environment["IVFPQ_RERANK_CANDIDATES"].flatMap(Int.init)
        let useBbox = ProcessInfo.processInfo.environment["IVF_USE_BBOX"] == "1"
        let listMisses = ProcessInfo.processInfo.environment["LIST_MISSES"] == "1"
        let focus = ProcessInfo.processInfo.environment["FOCUS_ID"]

        let mcc = try MccRiskTable.load(path: mccPath)
        let index = try ReferencesIndex.load(path: referencesPath)
        let ivfPath = IVFIndex.defaultPath(for: referencesPath)
        let ivf: IVFIndex? = FileManager.default.fileExists(atPath: ivfPath) ? try IVFIndex.load(path: ivfPath) : nil
        let pqPath = IVFPQIndex.defaultPath(for: referencesPath)
        let pq: IVFPQIndex? = FileManager.default.fileExists(atPath: pqPath) ? try IVFPQIndex.load(path: pqPath) : nil
        let vectorizer = Vectorizer(mccRisk: mcc)
        let config = SearchConfig(
            nprobe: nprobe,
            initialNprobe: initialNprobe,
            ivfpqRerankCandidates: rerank,
            useBoundingBoxes: useBbox
        )

        let url = URL(fileURLWithPath: testDataPath)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let testData = try decoder.decode(TestData.self, from: data)

        // Optional per-lane weights for experimentation. WEIGHTS env var: comma-separated 14 doubles.
        var weights = [Double](repeating: 1.0, count: 14)
        if let weightsEnv = ProcessInfo.processInfo.environment["WEIGHTS"] {
            let parts = weightsEnv.split(separator: ",").compactMap { Double($0) }
            for i in 0..<min(14, parts.count) { weights[i] = parts[i] }
        }
        let useWeighted = weights.contains(where: { $0 != 1.0 })
        let useL1 = ProcessInfo.processInfo.environment["USE_L1"] == "1"

        func weightedTopK(_ q: [Int16]) -> [(rec: Int, dist2: Int64, label: UInt8)] {
            let count = index.header.count
            let stride = index.header.stride
            let dim = index.header.dim
            let labels = index.labels
            let vectors = index.vectors
            let basePtr = vectors.baseAddress!
            var heap: [(rec: Int, dist2: Int64, label: UInt8)] = []
            heap.reserveCapacity(5)
            var wsq = [Double](repeating: 1.0, count: dim)
            for i in 0..<dim { wsq[i] = weights[i] * weights[i] }
            for r in 0..<count {
                let recPtr = basePtr.advanced(by: r * stride)
                var sum: Double = 0
                if useL1 {
                    for lane in 0..<dim {
                        let d = Double(abs(Int32(q[lane]) - Int32(recPtr[lane])))
                        sum += weights[lane] * d
                    }
                } else {
                    for lane in 0..<dim {
                        let d = Double(Int32(q[lane]) - Int32(recPtr[lane]))
                        sum += wsq[lane] * d * d
                    }
                }
                let dist = Int64(sum)
                let cand = (rec: r, dist2: dist, label: labels[r])
                if heap.count < 5 {
                    heap.append(cand)
                    heap.sort { $0.dist2 < $1.dist2 }
                } else if dist < heap[4].dist2 {
                    heap[4] = cand
                    heap.sort { $0.dist2 < $1.dist2 }
                }
            }
            return heap
        }

        var tp = 0, tn = 0, fp = 0, fn = 0
        var misses: [(String, Bool, Double, Bool, Double)] = []

        for entry in testData.entries {
            if let focus, entry.request.id != focus { continue }
            let raw = try vectorizer.vectorize(entry.request)
            let q = vectorizer.quantize(raw)
            let votes: Int
            if useWeighted || useL1 {
                let neighbors = weightedTopK(q)
                var v = 0
                for n in neighbors where n.label == 1 { v += 1 }
                votes = v
            } else {
                votes = KNN.fraudVoteCount(query: q, in: index, ivf: ivf, pq: pq, config: config, k: 5)
            }
            let score = Double(votes) / 5.0
            let approved = score < 0.5
            let exp = entry.expected_approved
            if exp == approved {
                if approved { tn += 1 } else { tp += 1 }
            } else {
                if approved { fn += 1 } else { fp += 1 }
                if listMisses {
                    misses.append((entry.request.id, exp, entry.expected_fraud_score, approved, score))
                }
            }
            if let focus {
                print("focus id=\(focus) expected_approved=\(exp) expected_score=\(entry.expected_fraud_score) got_score=\(score) got_approved=\(approved)")
                let queryStr = (0..<14).map { String(q[$0]) }.joined(separator: ",")
                print("  query lanes=[\(queryStr)]")
                if useWeighted || useL1 {
                    let nbrs = weightedTopK(q)
                    print("  (custom-distance top5)")
                    for n in nbrs {
                        let stride = index.header.stride
                        let base = n.rec * stride
                        let lanesStr = (0..<14).map { String(index.vectors[base + $0]) }.joined(separator: ",")
                        print("  rec=\(n.rec) dist=\(n.dist2) label=\(n.label) lanes=[\(lanesStr)]")
                    }
                } else {
                    let neighbors = KNN.topK(query: q, in: index, k: 16)
                    for n in neighbors {
                        let label = index.labels[n.recordIndex]
                        let stride = index.header.stride
                        let base = n.recordIndex * stride
                        let lanesStr = (0..<14).map { String(index.vectors[base + $0]) }.joined(separator: ",")
                        print("  rec=\(n.recordIndex) dist2=\(n.distanceSquared) label=\(label) lanes=[\(lanesStr)]")
                    }
                }
            }
        }

        print("tp=\(tp) tn=\(tn) fp=\(fp) fn=\(fn) total=\(tp+tn+fp+fn)")
        if listMisses {
            print("misses (\(misses.count)):")
            for m in misses.prefix(50) {
                print("  id=\(m.0) exp_app=\(m.1) exp_s=\(m.2) got_app=\(m.3) got_s=\(m.4)")
            }
        }
    }
}
