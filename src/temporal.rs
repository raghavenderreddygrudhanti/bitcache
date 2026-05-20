//! Time-Aware Retrieval — prioritize recent and temporally relevant memories.
//!
//! Standard vector search ignores time. This module adds:
//! - Recency weighting (newer memories score higher)
//! - Temporal relevance (match query time context)
//! - Event evolution tracking (detect when facts change over time)

/// Compute recency weight for a memory.
/// Returns [0.0, 1.0] where 1.0 = just stored, decays over time.
///
/// Uses exponential decay: weight = exp(-lambda * days_old)
pub fn recency_weight(stored_time: f64, current_time: f64, half_life_days: f64) -> f64 {
    let days_old = (current_time - stored_time) / 86400.0;
    if days_old <= 0.0 { return 1.0; }
    let lambda = (2.0_f64).ln() / half_life_days;
    (-lambda * days_old).exp()
}

/// Combine semantic similarity with temporal relevance.
/// Returns a blended score that favors recent, relevant memories.
///
/// score = alpha * similarity + (1 - alpha) * recency
pub fn time_aware_score(
    similarity: f64,
    stored_time: f64,
    current_time: f64,
    half_life_days: f64,
    alpha: f64,  // weight for similarity vs recency [0, 1]
) -> f64 {
    let recency = recency_weight(stored_time, current_time, half_life_days);
    alpha * similarity + (1.0 - alpha) * recency
}

/// Detect if a query is asking about "current" state (uses temporal keywords).
/// Returns true if the query likely wants the most recent information.
pub fn is_current_query(query: &str) -> bool {
    let lower = query.to_lowercase();
    let current_signals = [
        "current", "now", "today", "right now", "at the moment",
        "currently", "these days", "latest", "recent",
        "what do we use", "what are we using", "what is our",
    ];
    current_signals.iter().any(|s| lower.contains(s))
}

/// Compute temporal boost for retrieval scoring.
/// If query asks about "current" state, heavily boost recent memories.
/// Otherwise, use balanced scoring.
pub fn temporal_boost(
    query: &str,
    similarity: f64,
    stored_time: f64,
    current_time: f64,
) -> f64 {
    if is_current_query(query) {
        // Strongly favor recent: alpha=0.4 (60% recency)
        time_aware_score(similarity, stored_time, current_time, 30.0, 0.4)
    } else {
        // Balanced: alpha=0.8 (80% similarity, 20% recency)
        time_aware_score(similarity, stored_time, current_time, 90.0, 0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recency_fresh() {
        let w = recency_weight(100.0, 100.0, 30.0);
        assert_eq!(w, 1.0);
    }

    #[test]
    fn test_recency_half_life() {
        // At exactly half_life days, weight should be ~0.5
        let half_life = 30.0;
        let stored = 0.0;
        let current = 30.0 * 86400.0; // 30 days in seconds
        let w = recency_weight(stored, current, half_life);
        assert!((w - 0.5).abs() < 0.01, "Expected ~0.5, got {}", w);
    }

    #[test]
    fn test_recency_decreases() {
        let w1 = recency_weight(0.0, 86400.0, 30.0);      // 1 day old
        let w2 = recency_weight(0.0, 86400.0 * 30.0, 30.0); // 30 days old
        let w3 = recency_weight(0.0, 86400.0 * 90.0, 30.0); // 90 days old
        assert!(w1 > w2);
        assert!(w2 > w3);
    }

    #[test]
    fn test_current_query_detection() {
        assert!(is_current_query("What database are we currently using?"));
        assert!(is_current_query("What do we use for caching now?"));
        assert!(is_current_query("What is our latest deployment tool?"));
        assert!(!is_current_query("Tell me about the history of our stack"));
        assert!(!is_current_query("What happened last month?"));
    }

    #[test]
    fn test_temporal_boost_current() {
        let recent_score = temporal_boost(
            "What database do we currently use?",
            0.7,  // moderate similarity
            86400.0 * 50.0,  // stored 50 days ago (recent)
            86400.0 * 60.0,  // current time = day 60
        );
        let old_score = temporal_boost(
            "What database do we currently use?",
            0.8,  // higher similarity but older
            0.0,  // stored at day 0 (old)
            86400.0 * 60.0,
        );
        // Recent should win despite lower similarity (for "current" queries)
        assert!(recent_score > old_score, "Recent {} should beat old {}", recent_score, old_score);
    }

    #[test]
    fn test_temporal_boost_history() {
        // For non-current queries, similarity dominates
        let high_sim = temporal_boost(
            "Tell me about our database history",
            0.9,  // high similarity
            0.0,  // old
            86400.0 * 60.0,
        );
        let low_sim = temporal_boost(
            "Tell me about our database history",
            0.3,  // low similarity
            86400.0 * 59.0,  // very recent
            86400.0 * 60.0,
        );
        assert!(high_sim > low_sim, "High similarity {} should beat recency {}", high_sim, low_sim);
    }
}
