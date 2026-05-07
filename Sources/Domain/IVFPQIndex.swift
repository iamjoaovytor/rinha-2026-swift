#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

import Foundation

public struct IVFPQHeader: Sendable {
    public static let magic: UInt32 = 0x50_51_56_52 // "RVQP" little-endian
    public static let supportedVersion: UInt32 = 1
    public static let bytes = 64
    public static let pageAlignment = 4096

    public let version: UInt32
    public let count: Int
    public let stride: Int
    public let subvectorCount: Int
    public let subvectorWidth: Int
    public let createdUnix: Int64
}

public enum IVFPQError: Error, Sendable, CustomStringConvertible {
    case fileTooSmall(size: Int)
    case badMagic(UInt32)
    case unsupportedVersion(UInt32)
    case invalidLayout(subvectorCount: UInt32, subvectorWidth: UInt32, stride: UInt32)
    case fileSizeMismatch(expected: Int, got: Int)
    case openFailed(path: String, errno: Int32)
    case statFailed(path: String, errno: Int32)
    case mmapFailed(errno: Int32)

    public var description: String {
        switch self {
        case .fileTooSmall(let size): return "ivfpq file too small: \(size) bytes"
        case .badMagic(let magic): return "ivfpq bad magic: 0x\(String(magic, radix: 16))"
        case .unsupportedVersion(let version): return "ivfpq unsupported version: \(version)"
        case .invalidLayout(let subvectorCount, let subvectorWidth, let stride):
            return "ivfpq invalid layout: subvectors=\(subvectorCount) width=\(subvectorWidth) stride=\(stride)"
        case .fileSizeMismatch(let expected, let got): return "ivfpq size mismatch: expected \(expected), got \(got)"
        case .openFailed(let path, let errno): return "ivfpq open(\(path)) failed: errno=\(errno)"
        case .statFailed(let path, let errno): return "ivfpq stat(\(path)) failed: errno=\(errno)"
        case .mmapFailed(let errno): return "ivfpq mmap failed: errno=\(errno)"
        }
    }
}

public final class IVFPQIndex: @unchecked Sendable {
    public let header: IVFPQHeader
    public let codebooksOffset: Int
    public let codesOffset: Int

    private let basePointer: UnsafeRawPointer
    private let mappedSize: Int

    public var codebooks: UnsafeBufferPointer<Int16> {
        let start = basePointer.advanced(by: codebooksOffset).assumingMemoryBound(to: Int16.self)
        return UnsafeBufferPointer(start: start, count: header.subvectorCount * 256 * header.subvectorWidth)
    }

    public var codes: UnsafeBufferPointer<UInt8> {
        let start = basePointer.advanced(by: codesOffset).assumingMemoryBound(to: UInt8.self)
        return UnsafeBufferPointer(start: start, count: header.count * header.subvectorCount)
    }

    private init(
        basePointer: UnsafeRawPointer,
        mappedSize: Int,
        header: IVFPQHeader,
        codebooksOffset: Int,
        codesOffset: Int
    ) {
        self.basePointer = basePointer
        self.mappedSize = mappedSize
        self.header = header
        self.codebooksOffset = codebooksOffset
        self.codesOffset = codesOffset
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: basePointer), mappedSize)
    }

    public static func defaultPath(for referencesPath: String) -> String {
        let url = URL(fileURLWithPath: referencesPath)
        return url.deletingPathExtension().appendingPathExtension("pq").path
    }

    public static func parseHeader(_ data: UnsafeRawBufferPointer) throws -> IVFPQHeader {
        guard data.count >= IVFPQHeader.bytes else {
            throw IVFPQError.fileTooSmall(size: data.count)
        }
        let magic = data.load(fromByteOffset: 0, as: UInt32.self).littleEndian
        guard magic == IVFPQHeader.magic else {
            throw IVFPQError.badMagic(magic)
        }
        let version = data.load(fromByteOffset: 4, as: UInt32.self).littleEndian
        guard version == IVFPQHeader.supportedVersion else {
            throw IVFPQError.unsupportedVersion(version)
        }
        let count = data.load(fromByteOffset: 8, as: UInt64.self).littleEndian
        let stride = data.load(fromByteOffset: 16, as: UInt32.self).littleEndian
        let subvectorCount = data.load(fromByteOffset: 20, as: UInt32.self).littleEndian
        let subvectorWidth = data.load(fromByteOffset: 24, as: UInt32.self).littleEndian
        let createdUnix = data.load(fromByteOffset: 32, as: Int64.self).littleEndian
        guard stride == 16,
              subvectorCount > 0,
              subvectorWidth > 0,
              subvectorCount * subvectorWidth == stride
        else {
            throw IVFPQError.invalidLayout(
                subvectorCount: subvectorCount,
                subvectorWidth: subvectorWidth,
                stride: stride
            )
        }
        return IVFPQHeader(
            version: version,
            count: Int(count),
            stride: Int(stride),
            subvectorCount: Int(subvectorCount),
            subvectorWidth: Int(subvectorWidth),
            createdUnix: createdUnix
        )
    }

    public static func load(path: String) throws -> IVFPQIndex {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw IVFPQError.openFailed(path: path, errno: errno)
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw IVFPQError.statFailed(path: path, errno: errno)
        }
        let fileSize = Int(st.st_size)
        guard fileSize >= IVFPQHeader.bytes else {
            throw IVFPQError.fileTooSmall(size: fileSize)
        }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: -1)
        else {
            throw IVFPQError.mmapFailed(errno: errno)
        }
        let basePointer = UnsafeRawPointer(mapped)

        do {
            let bufferPointer = UnsafeRawBufferPointer(start: basePointer, count: fileSize)
            let header = try parseHeader(bufferPointer)
            let codebooksOffset = IVFPQHeader.bytes
            let codebooksBytes = header.subvectorCount * 256 * header.subvectorWidth * MemoryLayout<Int16>.size
            let codesOffset = alignUp(codebooksOffset + codebooksBytes, to: IVFPQHeader.pageAlignment)
            let expectedSize = codesOffset + header.count * header.subvectorCount * MemoryLayout<UInt8>.size
            guard expectedSize == fileSize else {
                munmap(mapped, fileSize)
                throw IVFPQError.fileSizeMismatch(expected: expectedSize, got: fileSize)
            }
            return IVFPQIndex(
                basePointer: basePointer,
                mappedSize: fileSize,
                header: header,
                codebooksOffset: codebooksOffset,
                codesOffset: codesOffset
            )
        } catch {
            munmap(mapped, fileSize)
            throw error
        }
    }

    public func warm() {
        _ = madvise(UnsafeMutableRawPointer(mutating: basePointer), mappedSize, MADV_WILLNEED)
        var sink: UInt8 = 0
        var offset = 0
        while offset < mappedSize {
            sink ^= basePointer.load(fromByteOffset: offset, as: UInt8.self)
            offset += IVFPQHeader.pageAlignment
        }
        _ = sink
    }
}
