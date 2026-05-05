import Foundation

/// Decomposed UTC timestamp parsed from the canonical 20-character form
/// `YYYY-MM-DDTHH:MM:SSZ`. Avoids `DateFormatter` and `Calendar` on the hot
/// path. `weekdayMon0` and `epochSeconds` are derived without Foundation
/// calendars so the type stays cheap.

struct ISO8601Fixed: Sendable, Equatable {
    let year: Int
    let month: Int
    let day: Int
    let hour: Int
    let minute: Int
    let second: Int

    /// Day of week with Monday=0..Sunday=6, matching Rinha 2026 dim 4.
    var weekdayMon0: Int {
        // Zeller's congruence: shift Jan/Feb to months 13/14 of the previous year.
        let m: Int
        let y: Int
        if month < 3 {
            m = month + 12
            y = year - 1
        } else {
            m = month
            y = year
        }
        let K = y % 100
        let J = y / 100
        // h: 0=Sat, 1=Sun, 2=Mon, ..., 6=Fri.
        let h = (day + (13 * (m + 1)) / 5 + K + K / 4 + J / 4 + 5 * J) % 7
        return (h + 5) % 7
    }

    /// Seconds since the Unix epoch (UTC) using Howard Hinnant's
    /// civil-from-days algorithm. Valid for any proleptic Gregorian date.
    var epochSeconds: Int64 {
        let yShift = month <= 2 ? year - 1 : year
        let era = (yShift >= 0 ? yShift : yShift - 399) / 400
        let yoe = Int64(yShift - era * 400)
        let mShift = month > 2 ? month - 3 : month + 9
        let doy = (153 * mShift + 2) / 5 + day - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + Int64(doy)
        let days = Int64(era) * 146_097 + doe - 719_468
        return days * 86_400
            + Int64(hour) * 3_600
            + Int64(minute) * 60
            + Int64(second)
    }

    enum ParseError: Error, Equatable, Sendable {
        case wrongLength
        case malformed
    }

    static func parse(_ string: String) throws -> ISO8601Fixed {
        var bytes = [UInt8]()
        bytes.reserveCapacity(20)
        for scalar in string.unicodeScalars {
            bytes.append(UInt8(scalar.value & 0xFF))
        }
        return try parse(bytes: bytes)
    }

    static func parse(bytes: [UInt8]) throws -> ISO8601Fixed {
        try bytes.withUnsafeBufferPointer {
            try parse(bytes: $0)
        }
    }

    static func parse(bytes: UnsafeBufferPointer<UInt8>) throws -> ISO8601Fixed {
        guard bytes.count == 20 else { throw ParseError.wrongLength }
        guard
            bytes[4] == 0x2D, bytes[7] == 0x2D, bytes[10] == 0x54,
            bytes[13] == 0x3A, bytes[16] == 0x3A, bytes[19] == 0x5A
        else { throw ParseError.malformed }

        func digit(_ index: Int) throws -> Int {
            let byte = bytes[index]
            guard byte >= 0x30, byte <= 0x39 else { throw ParseError.malformed }
            return Int(byte - 0x30)
        }

        let year = try digit(0) * 1000 + digit(1) * 100 + digit(2) * 10 + digit(3)
        let month = try digit(5) * 10 + digit(6)
        let day = try digit(8) * 10 + digit(9)
        let hour = try digit(11) * 10 + digit(12)
        let minute = try digit(14) * 10 + digit(15)
        let second = try digit(17) * 10 + digit(18)

        return ISO8601Fixed(
            year: year, month: month, day: day,
            hour: hour, minute: minute, second: second
        )
    }
}
