import Domain
import Foundation

struct APIConfig: Sendable {
    let referencesPath: String
    let mccRiskPath: String
    let ivfPath: String
    let ivfpqPath: String
    let searchConfig: SearchConfig
    let socketPath: String?
    let useSocketHandoff: Bool
    let port: Int

    init(environment env: [String: String]) {
        referencesPath = env["REFERENCES_BIN"] ?? RinhaAPI.referencesPathDefault
        mccRiskPath = env["MCC_RISK_JSON"] ?? RinhaAPI.mccRiskPathDefault
        ivfPath = env["IVF_BIN"] ?? IVFIndex.defaultPath(for: referencesPath)
        ivfpqPath = env["IVFPQ_BIN"] ?? IVFPQIndex.defaultPath(for: referencesPath)
        searchConfig = SearchConfig(
            nprobe: env["IVF_NPROBE"].flatMap(Int.init) ?? 4,
            initialNprobe: env["IVF_INITIAL_NPROBE"].flatMap(Int.init),
            adaptiveMinFraudVotes: env["IVF_ADAPTIVE_MIN_VOTES"].flatMap(Int.init) ?? 2,
            adaptiveMaxFraudVotes: env["IVF_ADAPTIVE_MAX_VOTES"].flatMap(Int.init) ?? 3,
            ivfpqRerankCandidates: env["IVFPQ_RERANK_CANDIDATES"].flatMap(Int.init),
            useBoundingBoxes: env["IVF_USE_BBOX"] == "1",
            expandOnUnanimousInitialVotes: env["IVF_ADAPTIVE_EXPAND_UNANIMOUS"] == "1",
            useBoundingBoxesOnExpandedSearch: env["IVF_ADAPTIVE_EXPANDED_USE_BBOX"] == "1"
        )
        socketPath = env["SOCKET_PATH"].flatMap { $0.isEmpty ? nil : $0 }
        useSocketHandoff = env["SOCKET_HANDOFF"] == "1"
        port = env["PORT"].flatMap(Int.init) ?? 9999
    }
}
