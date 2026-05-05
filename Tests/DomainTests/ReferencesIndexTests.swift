import Foundation
import Testing
@testable import Domain

struct ReferencesIndexTests {
    @Test func parsesValidHeader() throws {
        let bytes = makeHeader()
        let header = try bytes.withUnsafeBufferPointer { buffer in
            try ReferencesIndex.parseHeader(UnsafeRawBufferPointer(buffer))
        }
        #expect(header.version == 1)
        #expect(header.count == 3_000_000)
        #expect(header.dim == 14)
        #expect(header.stride == 16)
        #expect(header.scale == 8192)
        #expect(header.layout == ReferencesHeader.layoutAoS)
        #expect(header.sha256.count == 32)
        #expect(header.sha256[0] == 0xAB)
        #expect(header.createdUnix == 1_770_000_000)
    }

    @Test func rejectsBadMagic() {
        var bytes = makeHeader()
        bytes[0] = 0x00
        #expect {
            _ = try bytes.withUnsafeBufferPointer { buffer in
                try ReferencesIndex.parseHeader(UnsafeRawBufferPointer(buffer))
            }
        } throws: { error in
            if case ReferencesError.badMagic = error { return true }
            return false
        }
    }

    @Test func rejectsUnsupportedVersion() {
        var bytes = makeHeader()
        bytes[4] = 0x09
        #expect {
            _ = try bytes.withUnsafeBufferPointer { buffer in
                try ReferencesIndex.parseHeader(UnsafeRawBufferPointer(buffer))
            }
        } throws: { error in
            if case ReferencesError.unsupportedVersion = error { return true }
            return false
        }
    }

    @Test func rejectsUnexpectedDim() {
        var bytes = makeHeader()
        // dim lives at byte 16 (UInt32 LE).
        bytes[16] = 0x10
        #expect {
            _ = try bytes.withUnsafeBufferPointer { buffer in
                try ReferencesIndex.parseHeader(UnsafeRawBufferPointer(buffer))
            }
        } throws: { error in
            if case ReferencesError.unexpectedDim = error { return true }
            return false
        }
    }

    @Test func rejectsTooSmallBuffer() {
        let small = [UInt8](repeating: 0, count: 32)
        #expect {
            _ = try small.withUnsafeBufferPointer { buffer in
                try ReferencesIndex.parseHeader(UnsafeRawBufferPointer(buffer))
            }
        } throws: { error in
            if case ReferencesError.fileTooSmall = error { return true }
            return false
        }
    }

    private func makeHeader() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: ReferencesHeader.bytes)
        // magic "RNHA" little-endian.
        bytes[0] = 0x52; bytes[1] = 0x4E; bytes[2] = 0x48; bytes[3] = 0x41
        // version = 1.
        bytes[4] = 0x01
        // count = 3_000_000 (UInt64 LE).
        writeLE(UInt64(3_000_000), into: &bytes, at: 8)
        // dim = 14.
        writeLE(UInt32(14), into: &bytes, at: 16)
        // stride = 16.
        writeLE(UInt32(16), into: &bytes, at: 20)
        // scale = 8192.
        writeLE(Int32(8192), into: &bytes, at: 24)
        // layout = 0 (AoS).
        writeLE(UInt32(0), into: &bytes, at: 28)
        // sha256: marker byte at offset 32 so tests can spot-check.
        bytes[32] = 0xAB
        // createdUnix = 1_770_000_000.
        writeLE(Int64(1_770_000_000), into: &bytes, at: 64)
        return bytes
    }

    private func writeLE<T: FixedWidthInteger>(_ value: T, into bytes: inout [UInt8], at offset: Int) {
        var v = value.littleEndian
        withUnsafeBytes(of: &v) { raw in
            for (i, byte) in raw.enumerated() {
                bytes[offset + i] = byte
            }
        }
    }
}
