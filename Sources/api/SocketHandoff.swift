import Foundation
import NIOCore
import NIOPosix
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

final class SocketHandoffAcceptor: @unchecked Sendable {
    private final class BootstrapBox: @unchecked Sendable {
        let bootstrap: ClientBootstrap

        init(_ bootstrap: ClientBootstrap) {
            self.bootstrap = bootstrap
        }
    }

    private let ctrlPath: String
    private let bootstrapBox: BootstrapBox
    private let logger: @Sendable (String) -> Void

    init(ctrlPath: String, bootstrap: ClientBootstrap, logger: @escaping @Sendable (String) -> Void) {
        self.ctrlPath = ctrlPath
        self.bootstrapBox = BootstrapBox(bootstrap)
        self.logger = logger
    }

    func start() throws {
        let listener = try Self.makeUnixListener(at: ctrlPath)
        let bootstrapBox = self.bootstrapBox
        Thread.detachNewThread { [bootstrapBox, ctrlPath, logger] in
            defer { _ = close(listener) }
            while true {
                let accepted = accept(listener, nil, nil)
                if accepted < 0 {
                    if errno == EINTR {
                        continue
                    }
                    logger("handoff: accept failed on \(ctrlPath): errno=\(errno)")
                    break
                }
                let connectionFD = accepted
                Thread.detachNewThread {
                    Self.handleControlConnection(fd: connectionFD, bootstrap: bootstrapBox.bootstrap, logger: logger)
                }
            }
        }
    }

    private static func handleControlConnection(
        fd controlFD: CInt,
        bootstrap: ClientBootstrap,
        logger: @escaping @Sendable (String) -> Void
    ) {
        defer { _ = close(controlFD) }
        while true {
            guard let handedOffFD = Self.receiveFD(from: controlFD) else {
                return
            }
            bootstrap.withHandedOffConnectedSocket(handedOffFD).whenFailure { error in
                logger("handoff: failed to register socket fd=\(handedOffFD): \(error)")
                _ = close(handedOffFD)
            }
        }
    }

    private static func makeUnixListener(at path: String) throws -> CInt {
        try? FileManager.default.removeItem(atPath: path)
        #if canImport(Darwin)
        let socketType = SOCK_STREAM
        #else
        let socketType = Int32(SOCK_STREAM.rawValue)
        #endif
        let fd = socket(AF_UNIX, socketType, 0)
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(path.utf8CString)
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            _ = close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(ENAMETOOLONG))
        }
        withUnsafeMutablePointer(to: &addr.sun_path.0) { sunPathPtr in
            let rawPtr = UnsafeMutableRawPointer(sunPathPtr).assumingMemoryBound(to: CChar.self)
            for (index, byte) in pathBytes.enumerated() {
                rawPtr[index] = byte
            }
        }

        let length = socklen_t(MemoryLayout.size(ofValue: addr.sun_family) + pathBytes.count)
        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, length)
            }
        }
        guard bindResult == 0 else {
            let bindErrno = errno
            _ = close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(bindErrno))
        }

        guard listen(fd, 1024) == 0 else {
            let listenErrno = errno
            _ = close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(listenErrno))
        }
        return fd
    }

    private static func receiveFD(from controlFD: CInt) -> CInt? {
        var payload: UInt8 = 0
        let controlLength = cmsgSpace(MemoryLayout<CInt>.size)
        var control = [UInt8](repeating: 0, count: controlLength)
        var message = msghdr()

        return withUnsafeMutablePointer(to: &payload) { payloadPtr in
            var io = iovec(iov_base: payloadPtr, iov_len: 1)
            message.msg_iov = withUnsafeMutablePointer(to: &io) { $0 }
            message.msg_iovlen = 1
            return control.withUnsafeMutableBytes { controlBytes in
                message.msg_control = controlBytes.baseAddress
                message.msg_controllen = .init(controlBytes.count)
                let received = recvmsg(controlFD, &message, 0)
                guard received > 0 else {
                    return nil
                }
                guard let header = cmsgFirstHeader(&message) else {
                    return nil
                }
                guard header.pointee.cmsg_level == SOL_SOCKET, header.pointee.cmsg_type == SCM_RIGHTS else {
                    return nil
                }
                return cmsgData(header).assumingMemoryBound(to: CInt.self).pointee
            }
        }
    }
}

@inline(__always)
private func cmsgAlignment() -> Int {
    MemoryLayout<size_t>.size
}

@inline(__always)
private func cmsgAlign(_ length: Int) -> Int {
    let alignment = cmsgAlignment()
    return (length + alignment - 1) & ~(alignment - 1)
}

@inline(__always)
private func cmsgSpace(_ payloadLength: Int) -> Int {
    cmsgAlign(MemoryLayout<cmsghdr>.size) + cmsgAlign(payloadLength)
}

@inline(__always)
private func cmsgData(_ header: UnsafeMutablePointer<cmsghdr>) -> UnsafeMutableRawPointer {
    UnsafeMutableRawPointer(header).advanced(by: cmsgAlign(MemoryLayout<cmsghdr>.size))
}

@inline(__always)
private func cmsgFirstHeader(_ message: UnsafeMutablePointer<msghdr>) -> UnsafeMutablePointer<cmsghdr>? {
    guard Int(message.pointee.msg_controllen) >= MemoryLayout<cmsghdr>.size,
          let control = message.pointee.msg_control else {
        return nil
    }
    return control.assumingMemoryBound(to: cmsghdr.self)
}
