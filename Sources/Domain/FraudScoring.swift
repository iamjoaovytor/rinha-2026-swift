public enum FraudScoring {
    public static let approvalThreshold: Double = 0.5
    private static let responseBodies = [
        #"{"approved":true,"fraud_score":0.0}"#,
        #"{"approved":true,"fraud_score":0.2}"#,
        #"{"approved":true,"fraud_score":0.4}"#,
        #"{"approved":false,"fraud_score":0.6}"#,
        #"{"approved":false,"fraud_score":0.8}"#,
        #"{"approved":false,"fraud_score":1.0}"#
    ]

    /// Reduces `topNeighbors` to a fraud score using majority vote among
    /// labels and approves below `approvalThreshold`. Spec uses k=5, so the
    /// score is one of `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`.
    public static func score(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> ScoreResult {
        precondition(!neighbors.isEmpty, "need at least one neighbor")
        let fraudVotes = fraudVoteCount(neighbors: neighbors, index: index)
        let fraudScore = Double(fraudVotes) / Double(neighbors.count)
        let approved = fraudScore < approvalThreshold
        return ScoreResult(
            approved: approved,
            fraudScore: fraudScore,
            topNeighbors: neighbors
        )
    }

    public static func responseBody(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> String {
        responseBody(fraudVoteCount: fraudVoteCount(neighbors: neighbors, index: index))
    }

    public static func responseBody(fraudVoteCount: Int) -> String {
        responseBodies[fraudVoteCount]
    }

    @inline(__always)
    private static func fraudVoteCount(
        neighbors: [Neighbor],
        index: ReferencesIndex
    ) -> Int {
        precondition(!neighbors.isEmpty, "need at least one neighbor")
        let labels = index.labels
        var fraudVotes = 0
        for neighbor in neighbors where labels[neighbor.recordIndex] == 1 {
            fraudVotes += 1
        }
        return fraudVotes
    }
}
