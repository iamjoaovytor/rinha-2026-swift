import Foundation
import Testing
@testable import Domain

struct VectorizerTests {
    private let tolerance = 5e-4

    @Test func legitimateExampleMatchesSpecVector() throws {
        // Spec dims: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1,
        //            0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
        let mcc = MccRiskTable(["5411": 0.15], defaultValue: 0.5)
        let vectorizer = Vectorizer(mccRisk: mcc)

        let request = FraudRequest(
            id: "tx-legit",
            transaction: .init(
                amount: 41,
                installments: 2,
                requestedAt: "2026-03-11T18:00:00Z"
            ),
            customer: .init(
                avgAmount: 82,
                txCount24h: 3,
                knownMerchants: ["MERC-001"]
            ),
            merchant: .init(id: "MERC-001", mcc: "5411", avgAmount: 60),
            terminal: .init(isOnline: false, cardPresent: true, kmFromHome: 29.2),
            lastTransaction: nil
        )

        let v = try vectorizer.vectorize(request)
        let expected: [Double] = [
            0.0041, 0.1667, 0.05, 0.7826, 0.3333,
            -1, -1,
            0.0292, 0.15, 0, 1, 0, 0.15, 0.006
        ]
        try expectVector(v, equals: expected)
    }

    @Test func fraudulentExampleMatchesSpecVector() throws {
        // Spec dims: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1,
        //            0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
        let mcc = MccRiskTable(["7995": 0.75], defaultValue: 0.5)
        let vectorizer = Vectorizer(mccRisk: mcc)

        let request = FraudRequest(
            id: "tx-fraud",
            transaction: .init(
                amount: 9506,
                installments: 10,
                requestedAt: "2026-03-14T05:00:00Z"
            ),
            customer: .init(
                avgAmount: 100,
                txCount24h: 25,
                knownMerchants: ["MERC-OTHER"]
            ),
            merchant: .init(id: "MERC-NEW", mcc: "7995", avgAmount: 55),
            terminal: .init(isOnline: false, cardPresent: true, kmFromHome: 952.3),
            lastTransaction: nil
        )

        let v = try vectorizer.vectorize(request)
        let expected: [Double] = [
            0.9506, 0.8333, 1.0, 0.2174, 0.8333,
            -1, -1,
            0.9523, 1.0, 0, 1, 1, 0.75, 0.0055
        ]
        try expectVector(v, equals: expected)
    }

    @Test func lastTransactionPopulatesDim5And6() throws {
        let vectorizer = Vectorizer(mccRisk: MccRiskTable())
        let request = FraudRequest(
            id: "tx-with-last",
            transaction: .init(
                amount: 100,
                installments: 1,
                requestedAt: "2026-03-11T18:30:00Z"
            ),
            customer: .init(avgAmount: 100, txCount24h: 0, knownMerchants: []),
            merchant: .init(id: "M", mcc: "0000", avgAmount: 100),
            terminal: .init(isOnline: true, cardPresent: true, kmFromHome: 0),
            lastTransaction: .init(
                timestamp: "2026-03-11T18:00:00Z",
                kmFromCurrent: 250
            )
        )

        let v = try vectorizer.vectorize(request)
        #expect(abs(v[5] - (30.0 / 1440.0)) < 1e-9)
        #expect(abs(v[6] - 0.25) < 1e-9)
    }

    @Test func mccRiskFallsBackToDefault() throws {
        let vectorizer = Vectorizer(mccRisk: MccRiskTable())
        let request = FraudRequest(
            id: "tx",
            transaction: .init(amount: 0, installments: 0, requestedAt: "2026-01-01T00:00:00Z"),
            customer: .init(avgAmount: 100, txCount24h: 0, knownMerchants: []),
            merchant: .init(id: "M", mcc: "9999", avgAmount: 0),
            terminal: .init(isOnline: false, cardPresent: false, kmFromHome: 0),
            lastTransaction: nil
        )
        let v = try vectorizer.vectorize(request)
        #expect(v[12] == 0.5)
    }

    @Test func quantizationMapsRangesToInt16() {
        let vectorizer = Vectorizer()
        let q = vectorizer.quantize([
            0, 1, -1, 0.5, 0.25,
            0.0041, 0.1667, 0, 0, 0,
            0, 0, 0, 0
        ])
        #expect(q.count == 16)
        #expect(q[0] == 0)
        #expect(q[1] == 8192)
        #expect(q[2] == -8192)
        #expect(q[3] == 4096)
        #expect(q[4] == 2048)
        #expect(q[5] == 34)   // 0.0041 * 8192 ≈ 33.59
        #expect(q[6] == 1366) // 0.1667 * 8192 ≈ 1365.4
        #expect(q[14] == 0)
        #expect(q[15] == 0)
    }

    @Test func quantizationKeepsMinusOneSentinel() {
        let vectorizer = Vectorizer()
        let q = vectorizer.quantize([
            0, 0, 0, 0, 0,
            -1, -1,
            0, 0, 0, 0, 0, 0, 0
        ])
        #expect(q[5] == -8192)
        #expect(q[6] == -8192)
    }

    private func expectVector(_ got: [Double], equals expected: [Double]) throws {
        #expect(got.count == expected.count)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(abs(g - e) < tolerance, "dim \(i): got \(g), want \(e)")
        }
    }
}
