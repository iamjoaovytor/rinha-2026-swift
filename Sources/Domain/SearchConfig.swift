import Foundation

public struct SearchConfig: Sendable {
    public let nprobe: Int

    public init(nprobe: Int = 4) {
        self.nprobe = max(1, nprobe)
    }
}
