#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

import Foundation

public struct IVFHeader: Sendable {
    public static let magic: UInt32 = 0x46_56_49_52 // "RIVF" little-endian
    public static let supportedVersion: UInt32 = 1
    public static let bytes = 64
    public static let pageAlignment = 4096

    public let version: UInt32
    public let count: Int
    public let clusterCount: Int
    public let stride: Int
    public let createdUnix: Int64
}

public enum IVFError: Error, Sendable, CustomStringConvertible {
    case fileTooSmall(size: Int)
    case badMagic(UInt32)
    case unsupportedVersion(UInt32)
    case unexpectedStride(UInt32)
    case fileSizeMismatch(expected: Int, got: Int)
    case openFailed(path: String, errno: Int32)
    case statFailed(path: String, errno: Int32)
    case mmapFailed(errno: Int32)

    public var description: String {
        switch self {
        case .fileTooSmall(let size): return "ivf file too small: \(size) bytes"
        case .badMagic(let magic): return "ivf bad magic: 0x\(String(magic, radix: 16))"
        case .unsupportedVersion(let version): return "ivf unsupported version: \(version)"
        case .unexpectedStride(let stride): return "ivf unexpected stride: \(stride)"
        case .fileSizeMismatch(let expected, let got): return "ivf size mismatch: expected \(expected), got \(got)"
        case .openFailed(let path, let errno): return "ivf open(\(path)) failed: errno=\(errno)"
        case .statFailed(let path, let errno): return "ivf stat(\(path)) failed: errno=\(errno)"
        case .mmapFailed(let errno): return "ivf mmap failed: errno=\(errno)"
        }
    }
}

public final class IVFIndex: @unchecked Sendable {
    public let header: IVFHeader
    public let centroidsOffset: Int
    public let offsetsOffset: Int
    public let postingsOffset: Int

    private let basePointer: UnsafeRawPointer
    private let mappedSize: Int

    public var centroids: UnsafeBufferPointer<Int16> {
        let start = basePointer.advanced(by: centroidsOffset).assumingMemoryBound(to: Int16.self)
        return UnsafeBufferPointer(start: start, count: header.clusterCount * header.stride)
    }

    public var clusterOffsets: UnsafeBufferPointer<UInt32> {
        let start = basePointer.advanced(by: offsetsOffset).assumingMemoryBound(to: UInt32.self)
        return UnsafeBufferPointer(start: start, count: header.clusterCount + 1)
    }

    public var postings: UnsafeBufferPointer<UInt32> {
        let start = basePointer.advanced(by: postingsOffset).assumingMemoryBound(to: UInt32.self)
        return UnsafeBufferPointer(start: start, count: header.count)
    }

    private init(
        basePointer: UnsafeRawPointer,
        mappedSize: Int,
        header: IVFHeader,
        centroidsOffset: Int,
        offsetsOffset: Int,
        postingsOffset: Int
    ) {
        self.basePointer = basePointer
        self.mappedSize = mappedSize
        self.header = header
        self.centroidsOffset = centroidsOffset
        self.offsetsOffset = offsetsOffset
        self.postingsOffset = postingsOffset
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: basePointer), mappedSize)
    }

    public static func defaultPath(for referencesPath: String) -> String {
        let url = URL(fileURLWithPath: referencesPath)
        let base = url.deletingPathExtension()
        return base.appendingPathExtension("ivf").path
    }

    public static func parseHeader(_ data: UnsafeRawBufferPointer) throws -> IVFHeader {
        guard data.count >= IVFHeader.bytes else {
            throw IVFError.fileTooSmall(size: data.count)
        }
        let magic = data.load(fromByteOffset: 0, as: UInt32.self).littleEndian
        guard magic == IVFHeader.magic else {
            throw IVFError.badMagic(magic)
        }
        let version = data.load(fromByteOffset: 4, as: UInt32.self).littleEndian
        guard version == IVFHeader.supportedVersion else {
            throw IVFError.unsupportedVersion(version)
        }
        let count = data.load(fromByteOffset: 8, as: UInt64.self).littleEndian
        let clusterCount = data.load(fromByteOffset: 16, as: UInt32.self).littleEndian
        let stride = data.load(fromByteOffset: 20, as: UInt32.self).littleEndian
        guard stride == 16 else { throw IVFError.unexpectedStride(stride) }
        let createdUnix = data.load(fromByteOffset: 24, as: Int64.self).littleEndian
        return IVFHeader(
            version: version,
            count: Int(count),
            clusterCount: Int(clusterCount),
            stride: Int(stride),
            createdUnix: createdUnix
        )
    }

    public static func load(path: String) throws -> IVFIndex {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw IVFError.openFailed(path: path, errno: errno)
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw IVFError.statFailed(path: path, errno: errno)
        }
        let fileSize = Int(st.st_size)
        guard fileSize >= IVFHeader.bytes else {
            throw IVFError.fileTooSmall(size: fileSize)
        }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: -1)
        else {
            throw IVFError.mmapFailed(errno: errno)
        }
        let basePointer = UnsafeRawPointer(mapped)

        do {
            let bufferPointer = UnsafeRawBufferPointer(start: basePointer, count: fileSize)
            let header = try parseHeader(bufferPointer)
            let centroidsOffset = IVFHeader.bytes
            let offsetsOffset = alignUp(
                centroidsOffset + header.clusterCount * header.stride * MemoryLayout<Int16>.size,
                to: IVFHeader.pageAlignment
            )
            let postingsOffset = alignUp(
                offsetsOffset + (header.clusterCount + 1) * MemoryLayout<UInt32>.size,
                to: IVFHeader.pageAlignment
            )
            let expectedSize = postingsOffset + header.count * MemoryLayout<UInt32>.size
            guard expectedSize == fileSize else {
                munmap(mapped, fileSize)
                throw IVFError.fileSizeMismatch(expected: expectedSize, got: fileSize)
            }

            return IVFIndex(
                basePointer: basePointer,
                mappedSize: fileSize,
                header: header,
                centroidsOffset: centroidsOffset,
                offsetsOffset: offsetsOffset,
                postingsOffset: postingsOffset
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
            offset += IVFHeader.pageAlignment
        }
        _ = sink
    }
}
