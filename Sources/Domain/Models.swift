import Foundation

public struct FraudRequest: Decodable, Sendable {
    public let id: String
    public let transaction: TransactionPayload
    public let customer: CustomerPayload
    public let merchant: MerchantPayload
    public let terminal: TerminalPayload
    public let lastTransaction: LastTransactionPayload?

    public init(
        id: String,
        transaction: TransactionPayload,
        customer: CustomerPayload,
        merchant: MerchantPayload,
        terminal: TerminalPayload,
        lastTransaction: LastTransactionPayload?
    ) {
        self.id = id
        self.transaction = transaction
        self.customer = customer
        self.merchant = merchant
        self.terminal = terminal
        self.lastTransaction = lastTransaction
    }

    enum CodingKeys: String, CodingKey {
        case id, transaction, customer, merchant, terminal
        case lastTransaction = "last_transaction"
    }
}

public struct TransactionPayload: Decodable, Sendable {
    public let amount: Double
    public let installments: Int
    public let requestedAt: String

    public init(amount: Double, installments: Int, requestedAt: String) {
        self.amount = amount
        self.installments = installments
        self.requestedAt = requestedAt
    }

    enum CodingKeys: String, CodingKey {
        case amount, installments
        case requestedAt = "requested_at"
    }
}

public struct CustomerPayload: Decodable, Sendable {
    public let avgAmount: Double
    public let txCount24h: Int
    public let knownMerchants: [String]

    public init(avgAmount: Double, txCount24h: Int, knownMerchants: [String]) {
        self.avgAmount = avgAmount
        self.txCount24h = txCount24h
        self.knownMerchants = knownMerchants
    }

    enum CodingKeys: String, CodingKey {
        case avgAmount = "avg_amount"
        case txCount24h = "tx_count_24h"
        case knownMerchants = "known_merchants"
    }
}

public struct MerchantPayload: Decodable, Sendable {
    public let id: String
    public let mcc: String
    public let avgAmount: Double

    public init(id: String, mcc: String, avgAmount: Double) {
        self.id = id
        self.mcc = mcc
        self.avgAmount = avgAmount
    }

    enum CodingKeys: String, CodingKey {
        case id, mcc
        case avgAmount = "avg_amount"
    }
}

public struct TerminalPayload: Decodable, Sendable {
    public let isOnline: Bool
    public let cardPresent: Bool
    public let kmFromHome: Double

    public init(isOnline: Bool, cardPresent: Bool, kmFromHome: Double) {
        self.isOnline = isOnline
        self.cardPresent = cardPresent
        self.kmFromHome = kmFromHome
    }

    enum CodingKeys: String, CodingKey {
        case isOnline = "is_online"
        case cardPresent = "card_present"
        case kmFromHome = "km_from_home"
    }
}

public struct LastTransactionPayload: Decodable, Sendable {
    public let timestamp: String
    public let kmFromCurrent: Double

    public init(timestamp: String, kmFromCurrent: Double) {
        self.timestamp = timestamp
        self.kmFromCurrent = kmFromCurrent
    }

    enum CodingKeys: String, CodingKey {
        case timestamp
        case kmFromCurrent = "km_from_current"
    }
}

public struct FraudResponse: Encodable, Sendable {
    public let approved: Bool
    public let fraudScore: Double

    public init(approved: Bool, fraudScore: Double) {
        self.approved = approved
        self.fraudScore = fraudScore
    }

    enum CodingKeys: String, CodingKey {
        case approved
        case fraudScore = "fraud_score"
    }
}
