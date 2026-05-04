import Foundation

struct FraudRequest: Decodable, Sendable {
    let id: String
    let transaction: TransactionPayload
    let customer: CustomerPayload
    let merchant: MerchantPayload
    let terminal: TerminalPayload
    let lastTransaction: LastTransactionPayload?

    enum CodingKeys: String, CodingKey {
        case id, transaction, customer, merchant, terminal
        case lastTransaction = "last_transaction"
    }
}

struct TransactionPayload: Decodable, Sendable {
    let amount: Double
    let installments: Int
    let requestedAt: String

    enum CodingKeys: String, CodingKey {
        case amount, installments
        case requestedAt = "requested_at"
    }
}

struct CustomerPayload: Decodable, Sendable {
    let avgAmount: Double
    let txCount24h: Int
    let knownMerchants: [String]

    enum CodingKeys: String, CodingKey {
        case avgAmount = "avg_amount"
        case txCount24h = "tx_count_24h"
        case knownMerchants = "known_merchants"
    }
}

struct MerchantPayload: Decodable, Sendable {
    let id: String
    let mcc: String
    let avgAmount: Double

    enum CodingKeys: String, CodingKey {
        case id, mcc
        case avgAmount = "avg_amount"
    }
}

struct TerminalPayload: Decodable, Sendable {
    let isOnline: Bool
    let cardPresent: Bool
    let kmFromHome: Double

    enum CodingKeys: String, CodingKey {
        case isOnline = "is_online"
        case cardPresent = "card_present"
        case kmFromHome = "km_from_home"
    }
}

struct LastTransactionPayload: Decodable, Sendable {
    let timestamp: String
    let kmFromCurrent: Double

    enum CodingKeys: String, CodingKey {
        case timestamp
        case kmFromCurrent = "km_from_current"
    }
}

struct FraudResponse: Encodable, Sendable {
    let approved: Bool
    let fraudScore: Double

    enum CodingKeys: String, CodingKey {
        case approved
        case fraudScore = "fraud_score"
    }
}
