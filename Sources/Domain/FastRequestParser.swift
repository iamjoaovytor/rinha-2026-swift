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
        var cursor = Cursor(bytes: bytes)

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

private struct Cursor {
    let bytes: UnsafeBufferPointer<UInt8>
    var index: Int = 0

    mutating func skipWhitespace() {
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x0A, 0x0D, 0x09:
                index += 1
            default:
                return
            }
        }
    }

    mutating func consume(_ byte: UInt8) throws {
        skipWhitespace()
        guard index < bytes.count, bytes[index] == byte else {
            throw FastRequestParser.ParseError.malformed
        }
        index += 1
    }

    mutating func consumeObjectStart() throws { try consume(0x7B) } // {
    mutating func consumeObjectEnd() throws { try consume(0x7D) }   // }
    mutating func consumeArrayStart() throws { try consume(0x5B) }  // [
    mutating func consumeArrayEnd() throws { try consume(0x5D) }    // ]
    mutating func consumeComma() throws { try consume(0x2C) }       // ,
    mutating func consumeColon() throws { try consume(0x3A) }       // :

    mutating func consumeKey(_ key: StaticString) throws {
        skipWhitespace()
        guard index < bytes.count, bytes[index] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        let keyPointer = key.utf8Start
        index += 1
        let count = key.utf8CodeUnitCount
        guard index + count < bytes.count else {
            throw FastRequestParser.ParseError.malformed
        }
        for offset in 0..<count where bytes[index + offset] != keyPointer[offset] {
            throw FastRequestParser.ParseError.malformed
        }
        index += count
        guard bytes[index] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        index += 1
        try consumeColon()
    }

    mutating func consumeNullIfPresent() throws -> Bool {
        skipWhitespace()
        guard index + 3 < bytes.count else { return false }
        if bytes[index] == 0x6E, bytes[index + 1] == 0x75, bytes[index + 2] == 0x6C, bytes[index + 3] == 0x6C {
            index += 4
            return true
        }
        return false
    }

    mutating func consumeEOF() throws {
        skipWhitespace()
        guard index == bytes.count else {
            throw FastRequestParser.ParseError.malformed
        }
    }

    mutating func parseString() throws -> String {
        let raw = try parseStringBytes()
        return String(decoding: raw, as: UTF8.self)
    }

    mutating func parseStringBytes() throws -> [UInt8] {
        skipWhitespace()
        guard index < bytes.count, bytes[index] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        index += 1
        let start = index
        while index < bytes.count {
            let byte = bytes[index]
            if byte == 0x22 {
                let slice = Array(bytes[start..<index])
                index += 1
                return slice
            }
            // Payload strings are simple ASCII in the official dataset.
            if byte == 0x5C {
                throw FastRequestParser.ParseError.malformed
            }
            index += 1
        }
        throw FastRequestParser.ParseError.malformed
    }

    mutating func parseISO8601() throws -> ISO8601Fixed {
        skipWhitespace()
        guard index < bytes.count, bytes[index] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        index += 1
        let start = index
        let end = start + 20
        guard end < bytes.count, bytes[end] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        let value = try ISO8601Fixed.parse(bytes: UnsafeBufferPointer(rebasing: bytes[start..<end]))
        index = end + 1
        return value
    }

    mutating func parseFourDigitCode() throws -> Int {
        skipWhitespace()
        guard index < bytes.count, bytes[index] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        index += 1
        let start = index
        let end = start + 4
        guard end < bytes.count, bytes[end] == 0x22 else {
            throw FastRequestParser.ParseError.malformed
        }
        var value = 0
        for position in start..<end {
            let byte = bytes[position]
            guard byte >= 0x30, byte <= 0x39 else {
                throw FastRequestParser.ParseError.malformed
            }
            value = value * 10 + Int(byte - 0x30)
        }
        index = end + 1
        return value
    }

    mutating func parseBool() throws -> Bool {
        skipWhitespace()
        if index + 3 < bytes.count,
           bytes[index] == 0x74, bytes[index + 1] == 0x72, bytes[index + 2] == 0x75, bytes[index + 3] == 0x65 {
            index += 4
            return true
        }
        if index + 4 < bytes.count,
           bytes[index] == 0x66, bytes[index + 1] == 0x61, bytes[index + 2] == 0x6C, bytes[index + 3] == 0x73, bytes[index + 4] == 0x65 {
            index += 5
            return false
        }
        throw FastRequestParser.ParseError.malformed
    }

    mutating func parseInt() throws -> Int {
        let token = try parseNumberToken()
        guard let value = Int(token) else {
            throw FastRequestParser.ParseError.malformed
        }
        return value
    }

    mutating func parseDouble() throws -> Double {
        let token = try parseNumberToken()
        guard let value = Double(token) else {
            throw FastRequestParser.ParseError.malformed
        }
        return value
    }

    mutating func parseByteStringArray() throws -> [[UInt8]] {
        try consumeArrayStart()
        skipWhitespace()
        if index < bytes.count, bytes[index] == 0x5D {
            index += 1
            return []
        }

        var values: [[UInt8]] = []
        while true {
            values.append(try parseStringBytes())
            skipWhitespace()
            guard index < bytes.count else {
                throw FastRequestParser.ParseError.malformed
            }
            if bytes[index] == 0x2C {
                index += 1
                continue
            }
            if bytes[index] == 0x5D {
                index += 1
                return values
            }
            throw FastRequestParser.ParseError.malformed
        }
    }

    mutating func parseNumberToken() throws -> String {
        skipWhitespace()
        let start = index
        guard start < bytes.count else {
            throw FastRequestParser.ParseError.malformed
        }
        while index < bytes.count {
            switch bytes[index] {
            case 0x30...0x39, 0x2D, 0x2B, 0x2E, 0x45, 0x65:
                index += 1
            default:
                let slice = bytes[start..<index]
                guard !slice.isEmpty else {
                    throw FastRequestParser.ParseError.malformed
                }
                return String(decoding: slice, as: UTF8.self)
            }
        }
        let slice = bytes[start..<index]
        guard !slice.isEmpty else {
            throw FastRequestParser.ParseError.malformed
        }
        return String(decoding: slice, as: UTF8.self)
    }
}
