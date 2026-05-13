import NIOConcurrencyHelpers

final class LoaderState: @unchecked Sendable {
    private struct State {
        var loaded: LoadedState?
        var failure: String?
    }

    private let state = NIOLockedValueBox(State())

    var current: LoadedState? {
        self.state.withLockedValue { $0.loaded }
    }

    var isReady: Bool {
        self.state.withLockedValue { $0.loaded != nil }
    }

    func install(_ loaded: LoadedState) {
        self.state.withLockedValue { state in
            state.loaded = loaded
            state.failure = nil
        }
    }

    func recordFailure(_ message: String) {
        self.state.withLockedValue { state in
            state.failure = message
            state.loaded = nil
        }
    }
}
