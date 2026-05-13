import Foundation

public enum FastRequestParser {
    public enum ParseError: Error, Sendable {
        case malformed
    }

    public struct ParsedQuery: Sendable {
        public let transactionAmount: Double
        public let installments: Int
        public let requestedAt: ISO8601Fixed
        public let customerAvgAmount: Double
        public let customerTxCount24h: Int
        public let knownMerchant: Bool
        public let merchantAvgAmount: Double
        public let terminalIsOnline: Bool
        public let terminalCardPresent: Bool
        public let terminalKmFromHome: Double
        public let merchantMccCode: Int
        public let lastTransaction: (timestamp: ISO8601Fixed, kmFromCurrent: Double)?
    }

    public static func parsedQuery(
        from body: UnsafeRawBufferPointer
    ) throws -> ParsedQuery {
        let bytes = body.bindMemory(to: UInt8.self)
        var cursor = FastRequestCursor(bytes: bytes)

        try cursor.consumeObjectStart()
        try cursor.consumeKey("id")
        _ = try cursor.parseString()
        try cursor.consumeComma()

        try cursor.consumeKey("transaction")
        try cursor.consumeObjectStart()
        try cursor.consumeKey("amount")
        let transactionAmount = try cursor.parseDouble()
        try cursor.consumeComma()
        try cursor.consumeKey("installments")
        let installments = try cursor.parseInt()
        try cursor.consumeComma()
        try cursor.consumeKey("requested_at")
        let requestedAt = try cursor.parseISO8601()
        try cursor.consumeObjectEnd()
        try cursor.consumeComma()

        try cursor.consumeKey("customer")
        try cursor.consumeObjectStart()
        try cursor.consumeKey("avg_amount")
        let customerAvgAmount = try cursor.parseDouble()
        try cursor.consumeComma()
        try cursor.consumeKey("tx_count_24h")
        let customerTxCount24h = try cursor.parseInt()
        try cursor.consumeComma()
        try cursor.consumeKey("known_merchants")
        let knownMerchants = try cursor.parseByteStringArray()
        try cursor.consumeObjectEnd()
        try cursor.consumeComma()

        try cursor.consumeKey("merchant")
        try cursor.consumeObjectStart()
        try cursor.consumeKey("id")
        let merchantID = try cursor.parseStringBytes()
        try cursor.consumeComma()
        try cursor.consumeKey("mcc")
        let merchantMccCode = try cursor.parseFourDigitCode()
        try cursor.consumeComma()
        try cursor.consumeKey("avg_amount")
        let merchantAvgAmount = try cursor.parseDouble()
        try cursor.consumeObjectEnd()
        try cursor.consumeComma()

        try cursor.consumeKey("terminal")
        try cursor.consumeObjectStart()
        try cursor.consumeKey("is_online")
        let terminalIsOnline = try cursor.parseBool()
        try cursor.consumeComma()
        try cursor.consumeKey("card_present")
        let terminalCardPresent = try cursor.parseBool()
        try cursor.consumeComma()
        try cursor.consumeKey("km_from_home")
        let terminalKmFromHome = try cursor.parseDouble()
        try cursor.consumeObjectEnd()
        try cursor.consumeComma()

        try cursor.consumeKey("last_transaction")
        let lastTransaction: (timestamp: ISO8601Fixed, kmFromCurrent: Double)?
        if try cursor.consumeNullIfPresent() {
            lastTransaction = nil
        } else {
            try cursor.consumeObjectStart()
            try cursor.consumeKey("timestamp")
            let timestamp = try cursor.parseISO8601()
            try cursor.consumeComma()
            try cursor.consumeKey("km_from_current")
            let kmFromCurrent = try cursor.parseDouble()
            try cursor.consumeObjectEnd()
            lastTransaction = (timestamp: timestamp, kmFromCurrent: kmFromCurrent)
        }

        try cursor.consumeObjectEnd()
        try cursor.consumeEOF()

        return ParsedQuery(
            transactionAmount: transactionAmount,
            installments: installments,
            requestedAt: requestedAt,
            customerAvgAmount: customerAvgAmount,
            customerTxCount24h: customerTxCount24h,
            knownMerchant: knownMerchants.contains { $0.elementsEqual(merchantID) },
            merchantAvgAmount: merchantAvgAmount,
            terminalIsOnline: terminalIsOnline,
            terminalCardPresent: terminalCardPresent,
            terminalKmFromHome: terminalKmFromHome,
            merchantMccCode: merchantMccCode,
            lastTransaction: lastTransaction
        )
    }

    public static func quantizedQuery(
        from body: UnsafeRawBufferPointer,
        vectorizer: Vectorizer
    ) throws -> [Int16] {
        let parsed = try parsedQuery(from: body)
        return vectorizer.quantize(
            transactionAmount: parsed.transactionAmount,
            installments: parsed.installments,
            requestedAt: parsed.requestedAt,
            customerAvgAmount: parsed.customerAvgAmount,
            customerTxCount24h: parsed.customerTxCount24h,
            knownMerchant: parsed.knownMerchant,
            merchantAvgAmount: parsed.merchantAvgAmount,
            terminalIsOnline: parsed.terminalIsOnline,
            terminalCardPresent: parsed.terminalCardPresent,
            terminalKmFromHome: parsed.terminalKmFromHome,
            merchantMccCode: parsed.merchantMccCode,
            lastTransaction: parsed.lastTransaction
        )
    }
}
