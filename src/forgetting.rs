//! Adaptive Forgetting Engine — neuroscience-inspired memory decay.
//!
//! Replaces linear decay with Ebbinghaus power-law forgetting curves.
//! Different memory categories decay at different rates:
//!   Personal (years) > Emotional (months) > Preference (months) > Factual (weeks) > Trivial (days)
//!
//! Reinforcement follows spaced repetition: accessing after a long gap
//! strengthens more than frequent short-interval access.

use std::collections::HashMap;

/// Memory categories inspired by cognitive science.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    Personal,   // identity, relationships — decays very slowly
    Emotional,  // strong reactions — decays slowly
    Factual,    // technical knowledge — medium decay
    Preference, // habits, likes — slow-medium decay
    Trivial,    // small talk, transient — decays fast
}

/// Decay profile for a category.
#[derive(Clone, Debug)]
pub struct DecayProfile {
    /// Hours until ~50% retention
    pub stability: f64,
    /// Minimum importance floor (never decays below this)
    pub floor: f64,
    /// Power-law exponent (higher = steeper drop)
    pub exponent: f64,
}

/// Spaced repetition state per memory.
#[derive(Clone, Debug)]
pub struct RepetitionState {
    pub rep_count: u64,
    pub stability_multiplier: f64,
    pub last_gap_hours: f64,
}

impl RepetitionState {
    pub fn new() -> Self {
        Self { rep_count: 0, stability_multiplier: 1.0, last_gap_hours: 0.0 }
    }
}

/// Full adaptive memory state.
#[derive(Clone, Debug)]
pub struct AdaptiveState {
    pub importance: f64,
    pub base_importance: f64,
    pub category: MemoryCategory,
    pub access_count: u64,
    pub last_accessed: f64,
    pub created_at: f64,
    pub repetition: RepetitionState,
}

/// Configuration for the forgetting engine.
#[derive(Clone, Debug)]
pub struct ForgettingConfig {
    pub profiles: HashMap<MemoryCategory, DecayProfile>,
    pub max_stability_multiplier: f64,
    pub base_reinforce_amount: f64,
}

impl ForgettingConfig {
    pub fn default_config() -> Self {
        let mut profiles = HashMap::new();

        profiles.insert(MemoryCategory::Personal, DecayProfile {
            stability: 8760.0,  // ~1 year
            floor: 0.3,
            exponent: 0.3,
        });
        profiles.insert(MemoryCategory::Emotional, DecayProfile {
            stability: 4380.0,  // ~6 months
            floor: 0.2,
            exponent: 0.4,
        });
        profiles.insert(MemoryCategory::Preference, DecayProfile {
            stability: 2160.0,  // ~90 days
            floor: 0.15,
            exponent: 0.4,
        });
        profiles.insert(MemoryCategory::Factual, DecayProfile {
            stability: 720.0,   // ~30 days
            floor: 0.05,
            exponent: 0.5,
        });
        profiles.insert(MemoryCategory::Trivial, DecayProfile {
            stability: 48.0,    // 2 days
            floor: 0.0,
            exponent: 0.8,
        });

        Self {
            profiles,
            max_stability_multiplier: 10.0,
            base_reinforce_amount: 0.1,
        }
    }
}

// ─── Core Functions ──────────────────────────────────────────────────────────

/// Ebbinghaus power-law retention: R(t) = 1 / (1 + (t/S)^e)
/// Returns value in (0, 1]. At t=0 returns 1.0. At t=S returns ~0.5.
pub fn compute_retention(elapsed_hours: f64, stability: f64, exponent: f64) -> f64 {
    if elapsed_hours <= 0.0 { return 1.0; }
    let ratio = elapsed_hours / stability.max(0.001);
    1.0 / (1.0 + ratio.powf(exponent))
}

/// Classify memory content into a category using keyword patterns.
/// No LLM calls — runs in microseconds.
pub fn classify_memory(content: &str) -> MemoryCategory {
    let lower = content.to_lowercase();

    // Personal signals
    let personal = [
        "my name", "i am", "i work", "my wife", "my husband",
        "my daughter", "my son", "my family", "my mom", "my dad",
        "i live", "i grew up", "born in",
    ];
    let personal_score: usize = personal.iter().filter(|p| lower.contains(*p)).count();

    // Emotional signals
    let emotional = [
        "love", "hate", "afraid", "angry", "happy", "sad", "excited",
        "anxious", "amazing", "terrible", "worst", "best day",
        "proud", "grief", "miss", "stressed", "frustrated",
    ];
    let emotional_score: usize = emotional.iter().filter(|p| lower.contains(*p)).count();

    // Preference signals
    let preference = [
        "i prefer", "i always", "i never", "i like", "i dislike",
        "favorite", "please always", "please never", "please remember",
        "don't like", "don't want", "allergic", "vegetarian",
    ];
    let preference_score: usize = preference.iter().filter(|p| lower.contains(*p)).count();

    // Trivial signals
    let trivial = [
        "lunch", "weather", "coffee", "snack", "printer", "wifi",
        "boring", "nothing special", "decent", "grabbed", "elevator",
        "just finished", "just now",
    ];
    let trivial_score: usize = trivial.iter().filter(|p| lower.contains(*p)).count();

    // Factual signals
    let factual = [
        "version", "api", "database", "server", "deploy", "sla",
        "uptime", "production", "migrated", "upgraded", "installed",
        "config", "monitoring",
    ];
    let factual_score: usize = factual.iter().filter(|p| lower.contains(*p)).count();

    // Return highest scoring category
    let scores = [
        (personal_score, MemoryCategory::Personal),
        (emotional_score, MemoryCategory::Emotional),
        (preference_score, MemoryCategory::Preference),
        (trivial_score, MemoryCategory::Trivial),
        (factual_score, MemoryCategory::Factual),
    ];

    scores.iter()
        .max_by_key(|(s, _)| *s)
        .filter(|(s, _)| *s > 0)
        .map(|(_, cat)| *cat)
        .unwrap_or(MemoryCategory::Factual)
}

/// Apply adaptive decay to a single memory state.
/// Returns the new importance value.
pub fn apply_decay(
    state: &AdaptiveState,
    current_time: f64,
    config: &ForgettingConfig,
) -> f64 {
    let profile = config.profiles.get(&state.category)
        .unwrap_or(&DecayProfile { stability: 720.0, floor: 0.05, exponent: 0.5 });

    let elapsed_hours = (current_time - state.last_accessed) / 3600.0;
    if elapsed_hours <= 0.0 {
        return state.base_importance;
    }

    let effective_stability = profile.stability * state.repetition.stability_multiplier;
    let retention = compute_retention(elapsed_hours, effective_stability, profile.exponent);

    (state.base_importance * retention).max(profile.floor).min(1.0)
}

/// Spaced repetition reinforcement.
/// Returns (new_importance, new_base_importance, new_stability_multiplier).
pub fn reinforce_spaced(
    state: &AdaptiveState,
    current_time: f64,
    config: &ForgettingConfig,
) -> (f64, f64, f64) {
    let gap_hours = (current_time - state.last_accessed) / 3600.0;

    // Log-scaled gap bonus: 24h gap → bonus=1.0, 7d gap → bonus≈1.6
    let gap_bonus = (1.0 + gap_hours.max(0.0)).ln() / (1.0 + 24.0_f64).ln();
    let clamped_bonus = gap_bonus.clamp(0.1, 2.0);

    let reinforce = config.base_reinforce_amount * clamped_bonus;

    let new_importance = (state.importance + reinforce).min(1.0);
    let new_base = (state.base_importance + reinforce * 0.5).min(1.0);
    let new_stability = (state.repetition.stability_multiplier + clamped_bonus * 0.2)
        .min(config.max_stability_multiplier);

    (new_importance, new_base, new_stability)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_at_zero() {
        assert_eq!(compute_retention(0.0, 100.0, 0.5), 1.0);
    }

    #[test]
    fn test_retention_at_stability() {
        // At t=S, retention should be ~0.5
        let r = compute_retention(720.0, 720.0, 1.0);
        assert!((r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_retention_decreases() {
        let r1 = compute_retention(10.0, 100.0, 0.5);
        let r2 = compute_retention(100.0, 100.0, 0.5);
        let r3 = compute_retention(1000.0, 100.0, 0.5);
        assert!(r1 > r2);
        assert!(r2 > r3);
    }

    #[test]
    fn test_higher_stability_slower_decay() {
        let r_low = compute_retention(100.0, 50.0, 0.5);
        let r_high = compute_retention(100.0, 500.0, 0.5);
        assert!(r_high > r_low);
    }

    #[test]
    fn test_classify_personal() {
        assert_eq!(classify_memory("My name is Arjun and I work at Google"), MemoryCategory::Personal);
    }

    #[test]
    fn test_classify_trivial() {
        assert_eq!(classify_memory("Had pizza for lunch, it was decent"), MemoryCategory::Trivial);
    }

    #[test]
    fn test_classify_emotional() {
        assert_eq!(classify_memory("I am so excited about the promotion!"), MemoryCategory::Emotional);
    }

    #[test]
    fn test_classify_preference() {
        assert_eq!(classify_memory("I prefer bullet-point answers always"), MemoryCategory::Preference);
    }

    #[test]
    fn test_classify_factual() {
        assert_eq!(classify_memory("We migrated the database to Aurora"), MemoryCategory::Factual);
    }

    #[test]
    fn test_personal_outlives_trivial() {
        let config = ForgettingConfig::default_config();
        let personal = AdaptiveState {
            importance: 0.8, base_importance: 0.8,
            category: MemoryCategory::Personal,
            access_count: 0, last_accessed: 0.0, created_at: 0.0,
            repetition: RepetitionState::new(),
        };
        let trivial = AdaptiveState {
            importance: 0.8, base_importance: 0.8,
            category: MemoryCategory::Trivial,
            access_count: 0, last_accessed: 0.0, created_at: 0.0,
            repetition: RepetitionState::new(),
        };

        // After 7 days (168 hours → 604800 seconds)
        let time = 604800.0;
        let p_imp = apply_decay(&personal, time, &config);
        let t_imp = apply_decay(&trivial, time, &config);

        assert!(p_imp > t_imp, "Personal ({}) should outlive Trivial ({})", p_imp, t_imp);
    }

    #[test]
    fn test_spaced_repetition_gap_bonus() {
        let config = ForgettingConfig::default_config();
        let state = AdaptiveState {
            importance: 0.5, base_importance: 0.5,
            category: MemoryCategory::Factual,
            access_count: 1, last_accessed: 0.0, created_at: 0.0,
            repetition: RepetitionState::new(),
        };

        // Short gap (1 hour)
        let (imp_short, _, _) = reinforce_spaced(&state, 3600.0, &config);
        // Long gap (7 days)
        let (imp_long, _, _) = reinforce_spaced(&state, 604800.0, &config);

        assert!(imp_long > imp_short, "Long gap ({}) should reinforce more than short ({})", imp_long, imp_short);
    }

    #[test]
    fn test_importance_capped() {
        let config = ForgettingConfig::default_config();
        let state = AdaptiveState {
            importance: 0.95, base_importance: 0.95,
            category: MemoryCategory::Personal,
            access_count: 10, last_accessed: 0.0, created_at: 0.0,
            repetition: RepetitionState { rep_count: 10, stability_multiplier: 5.0, last_gap_hours: 168.0 },
        };

        let (new_imp, new_base, _) = reinforce_spaced(&state, 604800.0, &config);
        assert!(new_imp <= 1.0);
        assert!(new_base <= 1.0);
    }
}
