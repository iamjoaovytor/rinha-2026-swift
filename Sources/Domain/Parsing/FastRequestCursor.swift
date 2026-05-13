struct FastRequestCursor {
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

    mutating func consumeObjectStart() throws { try consume(0x7B) }
    mutating func consumeObjectEnd() throws { try consume(0x7D) }
    mutating func consumeArrayStart() throws { try consume(0x5B) }
    mutating func consumeArrayEnd() throws { try consume(0x5D) }
    mutating func consumeComma() throws { try consume(0x2C) }
    mutating func consumeColon() throws { try consume(0x3A) }

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
