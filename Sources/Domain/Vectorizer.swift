import Foundation

/// MCC → risk lookup loaded from `mcc_risk.json`. Missing keys fall back to
/// `defaultValue` (0.5 per spec).
public struct MccRiskTable: Sendable {
    private let table: [String: Double]
    public let defaultValue: Double

    public init(_ table: [String: Double] = [:], defaultValue: Double = 0.5) {
        self.table = table
        self.defaultValue = defaultValue
    }

    public func risk(for mcc: String) -> Double {
        table[mcc] ?? defaultValue
    }

    public static func load(path: String) throws -> MccRiskTable {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = parsed as? [String: Any] else {
            throw NSError(
                domain: "Domain.MccRiskTable", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "expected JSON object at \(path)"]
            )
        }
        var table = [String: Double](minimumCapacity: dict.count)
        for (key, value) in dict {
            if let d = value as? Double { table[key] = d }
            else if let n = value as? NSNumber { table[key] = n.doubleValue }
            else if let i = value as? Int { table[key] = Double(i) }
        }
        return MccRiskTable(table)
    }
}

/// Produces the canonical 14-dimensional fraud feature vector and its
/// `Int16` quantized form. Indices 5 and 6 retain the `-1` sentinel when
/// `last_transaction` is null — the spec forbids substituting them.
public struct Vectorizer: Sendable {
    let constants: VectorizerConstants
    let mccRisk: MccRiskTable

    init(
        constants: VectorizerConstants = .default,
        mccRisk: MccRiskTable = MccRiskTable()
    ) {
        self.constants = constants
        self.mccRisk = mccRisk
    }

    public init() {
        self.constants = .default
        self.mccRisk = MccRiskTable()
    }

    public init(mccRisk: MccRiskTable) {
        self.constants = .default
        self.mccRisk = mccRisk
    }

    public var scale: Int16 { constants.scale }

    public func vectorize(_ request: FraudRequest) throws -> [Double] {
        let txTime = try ISO8601Fixed.parse(request.transaction.requestedAt)
        var v = [Double](repeating: 0, count: 14)

        v[0] = clamp(request.transaction.amount / constants.maxAmount)
        v[1] = clamp(Double(request.transaction.installments) / constants.maxInstallments)

        let avg = request.customer.avgAmount
        v[2] = avg > 0
            ? clamp((request.transaction.amount / avg) / constants.amountVsAvgRatio)
            : 1.0

        v[3] = Double(txTime.hour) / 23.0
        v[4] = Double(txTime.weekdayMon0) / 6.0

        if let last = request.lastTransaction {
            let lastTime = try ISO8601Fixed.parse(last.timestamp)
            let deltaSeconds = max(0, txTime.epochSeconds - lastTime.epochSeconds)
            let minutes = Double(deltaSeconds) / 60.0
            v[5] = clamp(minutes / constants.maxMinutes)
            v[6] = clamp(last.kmFromCurrent / constants.maxKm)
        } else {
            v[5] = -1
            v[6] = -1
        }

        v[7] = clamp(request.terminal.kmFromHome / constants.maxKm)
        v[8] = clamp(Double(request.customer.txCount24h) / constants.maxTxCount24h)
        v[9] = request.terminal.isOnline ? 1 : 0
        v[10] = request.terminal.cardPresent ? 1 : 0
        v[11] = request.customer.knownMerchants.contains(request.merchant.id) ? 0 : 1
        v[12] = mccRisk.risk(for: request.merchant.mcc)
        v[13] = clamp(request.merchant.avgAmount / constants.maxMerchantAvgAmount)

        return v
    }

    /// 16-lane `Int16` quantization (14 dims + 2 padding). Final two lanes
    /// stay zero so SIMD loops can read aligned 16-element strides.
    public func quantize(_ vector: [Double]) -> [Int16] {
        precondition(vector.count == 14, "Vector must have 14 dimensions")
        var lanes = [Int16](repeating: 0, count: 16)
        let scaleDouble = Double(constants.scale)
        let int16Min = Double(Int16.min)
        let int16Max = Double(Int16.max)

        for i in 0..<14 {
            let x = vector[i]
            if x == -1 {
                lanes[i] = -constants.scale
            } else {
                let scaled = (x * scaleDouble).rounded()
                lanes[i] = Int16(min(int16Max, max(int16Min, scaled)))
            }
        }
        return lanes
    }

    @inline(__always)
    private func clamp(_ x: Double) -> Double {
        min(1.0, max(0.0, x))
    }
}
