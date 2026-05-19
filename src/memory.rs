//! Agent memory with prioritization, aging, and eviction.
//!
//! Memories have importance scores that decay over time. Frequently
//! accessed memories get reinforced. Low-value memories are evicted
//! when capacity is reached.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::streaming::StreamingIndex;

/// State for a single memory entry.
#[derive(Clone, Debug)]
struct MemoryState {
    importance: f64,
    access_count: u64,
    last_accessed: f64,
    created_at: f64,
}

/// A retrieved memory result.
#[derive(Clone, Debug)]
pub struct MemoryResult {
    pub id: String,
    pub content: String,
    pub importance: f64,
    pub score: f32,
    pub access_count: u64,
    pub metadata: HashMap<String, String>,
}

/// Memory system statistics.
#[derive(Clone, Debug)]
pub struct MemoryStats {
    pub total: usize,
    pub capacity: usize,
    pub mean_importance: f64,
    pub min_importance: f64,
    pub max_importance: f64,
    pub total_accesses: u64,
}

/// Prioritized agent memory with decay and eviction.
pub struct AgentMemory {
    dim: usize,
    capacity: usize,
    decay_rate: f64,
    reinforce_amount: f64,
    index: StreamingIndex,
    states: HashMap<String, MemoryState>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

impl AgentMemory {
    pub fn new(
        dim: usize,
        capacity: usize,
        decay_rate: f64,
        reinforce_amount: f64,
        rerank_factor: usize,
    ) -> Self {
        Self {
            dim,
            capacity,
            decay_rate,
            reinforce_amount,
            index: StreamingIndex::new(dim, rerank_factor),
            states: HashMap::new(),
        }
    }

    /// Store a new memory.
    pub fn save_memory(
        &mut self,
        vector: &[f32],
        content: &str,
        importance: f64,
        id: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> String {
        let importance = importance.clamp(0.0, 1.0);
        let ts = now();

        let mut meta = metadata.unwrap_or_default();
        meta.insert("content".to_string(), content.to_string());

        let mid = self.index.insert(vector, id, Some(meta));

        self.states.insert(mid.clone(), MemoryState {
            importance,
            access_count: 0,
            last_accessed: ts,
            created_at: ts,
        });

        // Evict if over capacity
        while self.index.len() > self.capacity {
            self.evict_lowest();
        }

        mid
    }

    /// Retrieve relevant memories and reinforce them.
    pub fn retrieve_memory(
        &mut self,
        query: &[f32],
        k: usize,
        min_importance: f64,
    ) -> Vec<MemoryResult> {
        self.apply_decay();

        let (scores, ids, metas) = self.index.search(query, k * 3);

        // Collect candidates first, then reinforce (avoids borrow conflict)
        let mut candidates: Vec<(f32, String, HashMap<String, String>)> = Vec::new();
        for ((score, id), meta) in scores.into_iter().zip(ids.into_iter()).zip(metas.into_iter()) {
            let state = match self.states.get(&id) {
                Some(s) => s,
                None => continue,
            };

            if state.importance < min_importance {
                continue;
            }

            candidates.push((score, id, meta));
            if candidates.len() >= k {
                break;
            }
        }

        // Now reinforce and build results
        let mut results = Vec::new();
        for (score, id, meta) in candidates {
            self.reinforce(&id);

            let state = self.states.get(&id).unwrap();
            let content = meta.get("content").cloned().unwrap_or_default();
            let filtered_meta: HashMap<String, String> = meta.into_iter()
                .filter(|(k, _)| k != "content")
                .collect();

            results.push(MemoryResult {
                id,
                content,
                importance: state.importance,
                score,
                access_count: state.access_count,
                metadata: filtered_meta,
            });
        }

        results
    }

    /// Manually reinforce a memory's importance.
    pub fn reinforce_memory(&mut self, id: &str, amount: Option<f64>) -> bool {
        self.reinforce_with_amount(id, amount)
    }

    /// Explicitly forget (delete) a memory.
    pub fn forget_memory(&mut self, id: &str) -> bool {
        self.states.remove(id);
        self.index.delete(id)
    }

    /// Get memory system statistics.
    pub fn stats(&self) -> MemoryStats {
        if self.states.is_empty() {
            return MemoryStats {
                total: 0,
                capacity: self.capacity,
                mean_importance: 0.0,
                min_importance: 0.0,
                max_importance: 0.0,
                total_accesses: 0,
            };
        }

        let importances: Vec<f64> = self.states.values().map(|s| s.importance).collect();
        let total_accesses: u64 = self.states.values().map(|s| s.access_count).sum();

        MemoryStats {
            total: self.states.len(),
            capacity: self.capacity,
            mean_importance: importances.iter().sum::<f64>() / importances.len() as f64,
            min_importance: importances.iter().cloned().fold(f64::MAX, f64::min),
            max_importance: importances.iter().cloned().fold(f64::MIN, f64::max),
            total_accesses,
        }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn reinforce(&mut self, id: &str) {
        self.reinforce_with_amount(id, None);
    }

    fn reinforce_with_amount(&mut self, id: &str, amount: Option<f64>) -> bool {
        let state = match self.states.get_mut(id) {
            Some(s) => s,
            None => return false,
        };
        let boost = amount.unwrap_or(self.reinforce_amount);
        state.importance = (state.importance + boost).min(1.0);
        state.access_count += 1;
        state.last_accessed = now();
        true
    }

    fn apply_decay(&mut self) {
        let ts = now();
        for state in self.states.values_mut() {
            let days_since_access = (ts - state.last_accessed) / 86400.0;
            let decay = self.decay_rate * days_since_access;
            state.importance = (state.importance - decay).max(0.0);
        }
    }

    fn evict_lowest(&mut self) {
        let worst_id = self.states.iter()
            .min_by(|a, b| a.1.importance.partial_cmp(&b.1.importance).unwrap())
            .map(|(id, _)| id.clone());

        if let Some(id) = worst_id {
            self.forget_memory(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_retrieve() {
        let dim = 8;
        let mut mem = AgentMemory::new(dim, 100, 0.05, 0.1, 10);

        let v = vec![1.0; 8];
        let id = mem.save_memory(&v, "test memory", 0.8, None, None);
        assert_eq!(mem.len(), 1);

        let results = mem.retrieve_memory(&v, 1, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "test memory");
    }

    #[test]
    fn test_eviction() {
        let dim = 8;
        let mut mem = AgentMemory::new(dim, 2, 0.0, 0.1, 10);

        let v1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        mem.save_memory(&v1, "low", 0.1, None, None);
        mem.save_memory(&v2, "high", 0.9, None, None);
        mem.save_memory(&v3, "medium", 0.5, None, None);

        // Capacity is 2, so lowest importance should be evicted
        assert_eq!(mem.len(), 2);
    }
}
