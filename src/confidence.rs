//! Memory Confidence Graph — dynamic belief updating for AI agents.
//!
//! Every memory gets:
//! - confidence score (how trustworthy is this?)
//! - source tracking (where did it come from?)
//! - contradiction history (what conflicted with it?)
//! - freshness score (how recent is the information?)
//!
//! When a new memory contradicts an old one, the old one's confidence
//! drops automatically. Frequently confirmed memories gain confidence.

use std::collections::HashMap;

/// Confidence state for a single memory.
#[derive(Clone, Debug)]
pub struct ConfidenceState {
    /// Current confidence score [0.0, 1.0]
    pub confidence: f64,
    /// Source of this memory (user, system, inferred, external)
    pub source: MemorySource,
    /// Number of times this memory was confirmed by consistent retrieval
    pub confirmations: u64,
    /// Number of times this memory was contradicted
    pub contradictions: u64,
    /// Timestamp of the information (not when stored, but when the fact was true)
    pub info_timestamp: f64,
    /// IDs of memories that contradict this one
    pub contradicted_by: Vec<String>,
}

/// Where a memory came from — affects initial confidence.
#[derive(Clone, Debug, PartialEq)]
pub enum MemorySource {
    /// User explicitly stated it
    UserDirect,
    /// System observed it (logs, metrics)
    SystemObserved,
    /// Inferred from other memories
    Inferred,
    /// External source (API, document)
    External,
}

/// Configuration for confidence scoring.
#[derive(Clone, Debug)]
pub struct ConfidenceConfig {
    /// Initial confidence by source type
    pub initial_confidence: HashMap<String, f64>,
    /// How much each confirmation boosts confidence
    pub confirmation_boost: f64,
    /// How much each contradiction reduces confidence
    pub contradiction_penalty: f64,
    /// Freshness decay rate (confidence drops for old info)
    pub freshness_decay_per_day: f64,
    /// Similarity threshold to detect contradictions
    pub contradiction_threshold: f64,
}

impl ConfidenceConfig {
    pub fn default_config() -> Self {
        let mut initial = HashMap::new();
        initial.insert("user_direct".to_string(), 0.95);
        initial.insert("system_observed".to_string(), 0.90);
        initial.insert("external".to_string(), 0.80);
        initial.insert("inferred".to_string(), 0.60);

        Self {
            initial_confidence: initial,
            confirmation_boost: 0.05,
            contradiction_penalty: 0.30,
            freshness_decay_per_day: 0.005,
            contradiction_threshold: 0.50,
        }
    }
}

impl ConfidenceState {
    pub fn new(source: MemorySource, timestamp: f64) -> Self {
        let initial = match &source {
            MemorySource::UserDirect => 0.95,
            MemorySource::SystemObserved => 0.90,
            MemorySource::External => 0.80,
            MemorySource::Inferred => 0.60,
        };
        Self {
            confidence: initial,
            source,
            confirmations: 0,
            contradictions: 0,
            info_timestamp: timestamp,
            contradicted_by: Vec::new(),
        }
    }

    /// Confirm this memory (consistent with new evidence).
    pub fn confirm(&mut self, boost: f64) {
        self.confirmations += 1;
        self.confidence = (self.confidence + boost).min(1.0);
    }

    /// Contradict this memory (new evidence conflicts).
    pub fn contradict(&mut self, penalty: f64, contradicting_id: String) {
        self.contradictions += 1;
        self.confidence = (self.confidence - penalty).max(0.0);
        self.contradicted_by.push(contradicting_id);
    }

    /// Compute current effective confidence (includes freshness).
    pub fn effective_confidence(&self, current_time: f64, freshness_decay: f64) -> f64 {
        let days_old = (current_time - self.info_timestamp) / 86400.0;
        let freshness = 1.0 / (1.0 + freshness_decay * days_old.max(0.0));
        (self.confidence * freshness).max(0.0).min(1.0)
    }
}

/// Check if two memories contradict each other based on embedding similarity
/// and temporal ordering. Returns true if they likely conflict.
pub fn detect_contradiction(
    sim: f64,
    new_day: f64,
    old_day: f64,
    threshold: f64,
) -> bool {
    sim > threshold && new_day > old_day
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_confidence() {
        let state = ConfidenceState::new(MemorySource::UserDirect, 0.0);
        assert_eq!(state.confidence, 0.95);
    }

    #[test]
    fn test_confirm_boosts() {
        let mut state = ConfidenceState::new(MemorySource::External, 0.0);
        assert_eq!(state.confidence, 0.80);
        state.confirm(0.05);
        assert!((state.confidence - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_contradict_reduces() {
        let mut state = ConfidenceState::new(MemorySource::UserDirect, 0.0);
        state.contradict(0.30, "new_mem_1".to_string());
        assert!((state.confidence - 0.65).abs() < 0.001);
        assert_eq!(state.contradicted_by.len(), 1);
    }

    #[test]
    fn test_confidence_capped() {
        let mut state = ConfidenceState::new(MemorySource::UserDirect, 0.0);
        for _ in 0..20 { state.confirm(0.1); }
        assert!(state.confidence <= 1.0);
    }

    #[test]
    fn test_freshness_decay() {
        let state = ConfidenceState::new(MemorySource::UserDirect, 0.0);
        let fresh = state.effective_confidence(0.0, 0.01);
        let old = state.effective_confidence(86400.0 * 30.0, 0.01); // 30 days
        assert!(fresh > old);
    }

    #[test]
    fn test_contradiction_detection() {
        assert!(detect_contradiction(0.7, 10.0, 5.0, 0.5));
        assert!(!detect_contradiction(0.3, 10.0, 5.0, 0.5)); // low similarity
        assert!(!detect_contradiction(0.7, 5.0, 10.0, 0.5)); // old is newer
    }
}
