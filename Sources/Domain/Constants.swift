import Foundation

/// Normalization and quantization constants from the Rinha de Backend 2026 spec.
///
/// `scale` controls the `Int16` quantized representation: `0..1` floats map to
/// `0..scale` and the `-1` sentinel for missing `last_transaction` data maps to
/// `-scale`. `8192` is a power of two with comfortable headroom for the L2²
/// accumulator (worst per-lane diff `16384`, square `~268M`, times 14 lanes
/// `~3.76B` — fits in `Int64`).

struct VectorizerConstants: Sendable {
    var maxAmount: Double = 10_000
    var maxInstallments: Double = 12
    var amountVsAvgRatio: Double = 10
    var maxMinutes: Double = 1440
    var maxKm: Double = 1000
    var maxTxCount24h: Double = 20
    var maxMerchantAvgAmount: Double = 10_000

    var scale: Int16 = 8192

    static let `default` = VectorizerConstants()
}
