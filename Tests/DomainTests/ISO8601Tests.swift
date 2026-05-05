import Foundation
import Testing
@testable import Domain

struct ISO8601Tests {
    @Test func parsesCanonicalFormat() throws {
        let parsed = try ISO8601Fixed.parse("2026-03-11T20:23:35Z")
        #expect(parsed.year == 2026)
        #expect(parsed.month == 3)
        #expect(parsed.day == 11)
        #expect(parsed.hour == 20)
        #expect(parsed.minute == 23)
        #expect(parsed.second == 35)
    }

    @Test func computesWeekdayMondayAsZero() throws {
        // 2026-03-11 is Wednesday → 2.
        #expect(try ISO8601Fixed.parse("2026-03-11T00:00:00Z").weekdayMon0 == 2)
        // 2026-03-14 is Saturday → 5.
        #expect(try ISO8601Fixed.parse("2026-03-14T00:00:00Z").weekdayMon0 == 5)
        // 2026-01-04 is Sunday → 6.
        #expect(try ISO8601Fixed.parse("2026-01-04T00:00:00Z").weekdayMon0 == 6)
        // 2026-01-05 is Monday → 0.
        #expect(try ISO8601Fixed.parse("2026-01-05T00:00:00Z").weekdayMon0 == 0)
    }

    @Test func epochSecondsMatchesUnixEpoch() throws {
        #expect(try ISO8601Fixed.parse("1970-01-01T00:00:00Z").epochSeconds == 0)
        #expect(try ISO8601Fixed.parse("2000-01-01T00:00:00Z").epochSeconds == 946_684_800)
        #expect(try ISO8601Fixed.parse("2026-03-11T20:23:35Z").epochSeconds == 1_773_260_615)
    }

    @Test func epochDeltaInMinutes() throws {
        let later = try ISO8601Fixed.parse("2026-03-11T18:30:00Z")
        let earlier = try ISO8601Fixed.parse("2026-03-11T18:00:00Z")
        #expect(later.epochSeconds - earlier.epochSeconds == 1_800)
    }

    @Test func rejectsWrongLength() {
        #expect(throws: ISO8601Fixed.ParseError.wrongLength) {
            try ISO8601Fixed.parse("2026-03-11T20:23:35")
        }
    }

    @Test func rejectsMalformedSeparators() {
        #expect(throws: ISO8601Fixed.ParseError.malformed) {
            try ISO8601Fixed.parse("2026/03/11T20:23:35Z")
        }
    }

    @Test func rejectsNonDigitField() {
        #expect(throws: ISO8601Fixed.ParseError.malformed) {
            try ISO8601Fixed.parse("20X6-03-11T20:23:35Z")
        }
    }
}
