#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

import Foundation

/// Mirrors the on-disk layout produced by `preprocess`. The runtime parses
/// only what it needs from the 128-byte header — enough to validate the
/// blob and compute the offsets to the label, orig_id, and vector arrays.
public struct ReferencesHeader: Sendable {
    public static let magic: UInt32 = 0x41_48_4E_52 // "RNHA" little-endian
    public static let supportedVersion: UInt32 = 1
    public static let layoutAoS: UInt32 = 0
    public static let bytes = 128
    public static let pageAlignment = 4096

    public let version: UInt32
    public let count: Int
    public let dim: Int
    public let stride: Int
    public let scale: Int16
    public let layout: UInt32
    public let sha256: [UInt8]
    public let createdUnix: Int64
}

public enum ReferencesError: Error, Sendable, CustomStringConvertible {
    case fileTooSmall(size: Int)
    case badMagic(UInt32)
    case unsupportedVersion(UInt32)
    case unexpectedDim(UInt32)
    case unexpectedStride(UInt32)
    case unexpectedLayout(UInt32)
    case fileSizeMismatch(expected: Int, got: Int)
    case openFailed(path: String, errno: Int32)
    case statFailed(path: String, errno: Int32)
    case mmapFailed(errno: Int32)

    public var description: String {
        switch self {
        case .fileTooSmall(let size): return "file too small: \(size) bytes"
        case .badMagic(let m): return "bad magic: 0x\(String(m, radix: 16))"
        case .unsupportedVersion(let v): return "unsupported version: \(v)"
        case .unexpectedDim(let d): return "unexpected dim: \(d)"
        case .unexpectedStride(let s): return "unexpected stride: \(s)"
        case .unexpectedLayout(let l): return "unexpected layout: \(l)"
        case .fileSizeMismatch(let e, let g): return "size mismatch: expected \(e), got \(g)"
        case .openFailed(let p, let e): return "open(\(p)) failed: errno=\(e)"
        case .statFailed(let p, let e): return "stat(\(p)) failed: errno=\(e)"
        case .mmapFailed(let e): return "mmap failed: errno=\(e)"
        }
    }
}

@inline(__always)
func alignUp(_ value: Int, to alignment: Int) -> Int {
    (value + alignment - 1) / alignment * alignment
}

/// Read-only mmap of a `references.bin` produced by `preprocess`.
///
/// Owns the mapping for its lifetime; `munmap` runs in `deinit`.
public final class ReferencesIndex: @unchecked Sendable {
    public let header: ReferencesHeader
    public let labelsOffset: Int
    public let origIdsOffset: Int
    public let vectorsOffset: Int

    private let basePointer: UnsafeRawPointer
    private let mappedSize: Int

    public var labels: UnsafeBufferPointer<UInt8> {
        let start = basePointer.advanced(by: labelsOffset).assumingMemoryBound(to: UInt8.self)
        return UnsafeBufferPointer(start: start, count: header.count)
    }

    public var origIds: UnsafeBufferPointer<UInt32> {
        let start = basePointer.advanced(by: origIdsOffset).assumingMemoryBound(to: UInt32.self)
        return UnsafeBufferPointer(start: start, count: header.count)
    }

    /// Vector lanes in AoS order: index `i*stride + lane` is record `i` lane `lane`.
    public var vectors: UnsafeBufferPointer<Int16> {
        let start = basePointer.advanced(by: vectorsOffset).assumingMemoryBound(to: Int16.self)
        return UnsafeBufferPointer(start: start, count: header.count * header.stride)
    }

    private init(
        basePointer: UnsafeRawPointer,
        mappedSize: Int,
        header: ReferencesHeader,
        labelsOffset: Int,
        origIdsOffset: Int,
        vectorsOffset: Int
    ) {
        self.basePointer = basePointer
        self.mappedSize = mappedSize
        self.header = header
        self.labelsOffset = labelsOffset
        self.origIdsOffset = origIdsOffset
        self.vectorsOffset = vectorsOffset
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: basePointer), mappedSize)
    }

    public static func parseHeader(_ data: UnsafeRawBufferPointer) throws -> ReferencesHeader {
        guard data.count >= ReferencesHeader.bytes else {
            throw ReferencesError.fileTooSmall(size: data.count)
        }
        let magic = data.load(fromByteOffset: 0, as: UInt32.self).littleEndian
        guard magic == ReferencesHeader.magic else {
            throw ReferencesError.badMagic(magic)
        }
        let version = data.load(fromByteOffset: 4, as: UInt32.self).littleEndian
        guard version == ReferencesHeader.supportedVersion else {
            throw ReferencesError.unsupportedVersion(version)
        }
        let count = data.load(fromByteOffset: 8, as: UInt64.self).littleEndian
        let dim = data.load(fromByteOffset: 16, as: UInt32.self).littleEndian
        guard dim == 14 else { throw ReferencesError.unexpectedDim(dim) }
        let stride = data.load(fromByteOffset: 20, as: UInt32.self).littleEndian
        guard stride == 16 else { throw ReferencesError.unexpectedStride(stride) }
        let scale = data.load(fromByteOffset: 24, as: Int32.self).littleEndian
        let layout = data.load(fromByteOffset: 28, as: UInt32.self).littleEndian
        guard layout == ReferencesHeader.layoutAoS else {
            throw ReferencesError.unexpectedLayout(layout)
        }
        var sha = [UInt8](repeating: 0, count: 32)
        for i in 0..<32 {
            sha[i] = data.load(fromByteOffset: 32 + i, as: UInt8.self)
        }
        let createdUnix = data.load(fromByteOffset: 64, as: Int64.self).littleEndian
        return ReferencesHeader(
            version: version,
            count: Int(count),
            dim: Int(dim),
            stride: Int(stride),
            scale: Int16(scale),
            layout: layout,
            sha256: sha,
            createdUnix: createdUnix
        )
    }

    public static func load(path: String) throws -> ReferencesIndex {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw ReferencesError.openFailed(path: path, errno: errno)
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw ReferencesError.statFailed(path: path, errno: errno)
        }
        let fileSize = Int(st.st_size)
        guard fileSize >= ReferencesHeader.bytes else {
            throw ReferencesError.fileTooSmall(size: fileSize)
        }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: -1)
        else {
            throw ReferencesError.mmapFailed(errno: errno)
        }
        let basePointer = UnsafeRawPointer(mapped)

        do {
            let bufferPointer = UnsafeRawBufferPointer(start: basePointer, count: fileSize)
            let header = try parseHeader(bufferPointer)
            let labelsOffset = ReferencesHeader.bytes
            let origIdsOffset = alignUp(labelsOffset + header.count, to: ReferencesHeader.pageAlignment)
            let vectorsOffset = alignUp(
                origIdsOffset + header.count * MemoryLayout<UInt32>.size,
                to: ReferencesHeader.pageAlignment
            )
            let expectedSize = vectorsOffset + header.count * header.stride * MemoryLayout<Int16>.size
            guard fileSize == expectedSize else {
                munmap(mapped, fileSize)
                throw ReferencesError.fileSizeMismatch(expected: expectedSize, got: fileSize)
            }
            return ReferencesIndex(
                basePointer: basePointer,
                mappedSize: fileSize,
                header: header,
                labelsOffset: labelsOffset,
                origIdsOffset: origIdsOffset,
                vectorsOffset: vectorsOffset
            )
        } catch {
            munmap(mapped, fileSize)
            throw error
        }
    }

    /// Hint the kernel to prefetch and pin pages, then touch one byte per
    /// 4 KiB so the first request never pays a page-fault stall.
    public func warm() {
        _ = madvise(UnsafeMutableRawPointer(mutating: basePointer), mappedSize, MADV_WILLNEED)
        var sink: UInt8 = 0
        var offset = 0
        let stride = ReferencesHeader.pageAlignment
        while offset < mappedSize {
            sink ^= basePointer.load(fromByteOffset: offset, as: UInt8.self)
            offset += stride
        }
        // Reference `sink` so the loop can't be elided.
        _ = sink
    }
}
