//! Emotional Weighting — prioritize memories with emotional intensity.
//!
//! Humans remember emotional events better. This module detects emotional
//! intensity in text and boosts memory importance accordingly.
//!
//! "Printer IP is 10.1.1.5" → low emotional weight
//! "Production outage caused customer escalation" → high emotional weight

/// Emotional intensity score [0.0, 1.0].
/// Higher = more emotionally charged = remembered better.
pub fn emotional_weight(content: &str) -> f64 {
    let lower = content.to_lowercase();
    let mut score: f64 = 0.0;

    // Urgency signals
    let urgency = ["urgent", "critical", "emergency", "asap", "immediately",
                   "outage", "down", "broken", "failed", "crash"];
    for word in &urgency {
        if lower.contains(word) { score += 0.25; }
    }

    // Frustration signals
    let frustration = ["frustrated", "annoyed", "angry", "unacceptable",
                       "keeps happening", "again", "still broken", "waste of time"];
    for phrase in &frustration {
        if lower.contains(phrase) { score += 0.20; }
    }

    // Positive emotional signals (also remembered well)
    let positive = ["amazing", "excellent", "love", "perfect", "breakthrough",
                    "promoted", "celebration", "milestone", "achievement"];
    for word in &positive {
        if lower.contains(word) { score += 0.15; }
    }

    // Escalation / impact signals
    let impact = ["customer", "escalation", "revenue", "data loss",
                  "security breach", "compliance", "legal", "deadline missed"];
    for phrase in &impact {
        if lower.contains(phrase) { score += 0.20; }
    }

    // Repetition signals (exclamation marks, caps)
    let exclamations = content.chars().filter(|c| *c == '!').count();
    if exclamations >= 2 { score += 0.10; }

    let caps_ratio = content.chars().filter(|c| c.is_uppercase()).count() as f64
        / content.len().max(1) as f64;
    if caps_ratio > 0.5 && content.len() > 5 { score += 0.10; }

    score.min(1.0)
}

/// Compute importance boost from emotional weight.
/// Returns additional importance to add (0.0 to max_boost).
pub fn emotional_importance_boost(content: &str, max_boost: f64) -> f64 {
    emotional_weight(content) * max_boost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_emotion() {
        let w = emotional_weight("Printer IP is 10.1.1.5");
        assert!(w < 0.1, "Expected low emotion, got {}", w);
    }

    #[test]
    fn test_high_urgency() {
        let w = emotional_weight("CRITICAL: Production outage, customer escalation");
        assert!(w > 0.4, "Expected high emotion, got {}", w);
    }

    #[test]
    fn test_positive_emotion() {
        let w = emotional_weight("Amazing news - I got promoted today!");
        assert!(w > 0.2, "Expected positive emotion, got {}", w);
    }

    #[test]
    fn test_frustration() {
        let w = emotional_weight("This keeps happening and I am frustrated");
        assert!(w > 0.3, "Expected frustration signal, got {}", w);
    }

    #[test]
    fn test_neutral() {
        let w = emotional_weight("The meeting is at 3pm in room 204");
        assert!(w < 0.1, "Expected neutral, got {}", w);
    }

    #[test]
    fn test_capped_at_one() {
        let w = emotional_weight("URGENT CRITICAL EMERGENCY outage customer escalation frustrated angry!!!");
        assert!(w <= 1.0);
    }

    #[test]
    fn test_boost() {
        let boost = emotional_importance_boost("Production outage caused customer escalation", 0.3);
        assert!(boost > 0.1);
        assert!(boost <= 0.3);
    }
}
