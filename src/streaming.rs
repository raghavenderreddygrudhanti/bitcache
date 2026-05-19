//! Streaming index: supports live inserts, updates, and deletes.
//!
//! No full rebuild needed. Vectors can be added and removed continuously,
//! making this suitable for long-running AI agent memory systems.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::quantize::{normalize, quantize};
use crate::search::{hamming_distance_one_to_many, inner_product};

/// Mutable vector index with insert, update, delete, and search.
pub struct StreamingIndex {
    dim: usize,
    n_bytes: usize,
    rerank_factor: usize,

    // Storage: parallel arrays indexed by slot
    codes: Vec<Vec<u8>>,
    vectors: Vec<Vec<f32>>,
    metadata: Vec<HashMap<String, String>>,
    timestamps: Vec<f64>,

    // ID mapping
    id_to_slot: HashMap<String, usize>,
    slot_to_id: HashMap<usize, String>,

    // Deleted slots (reusable)
    free_slots: Vec<usize>,

    next_id: u64,
}

fn now_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

impl StreamingIndex {
    pub fn new(dim: usize, rerank_factor: usize) -> Self {
        assert!(dim > 0);
        Self {
            dim,
            n_bytes: (dim + 7) / 8,
            rerank_factor,
            codes: Vec::new(),
            vectors: Vec::new(),
            metadata: Vec::new(),
            timestamps: Vec::new(),
            id_to_slot: HashMap::new(),
            slot_to_id: HashMap::new(),
            free_slots: Vec::new(),
            next_id: 0,
        }
    }

    /// Insert a single vector. Returns the assigned ID.
    pub fn insert(
        &mut self,
        vector: &[f32],
        id: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> String {
        assert_eq!(vector.len(), self.dim);

        let id = id.unwrap_or_else(|| {
            let id = format!("vec_{}", self.next_id);
            self.next_id += 1;
            id
        });

        // If ID already exists, update instead
        if self.id_to_slot.contains_key(&id) {
            self.update(&id, Some(vector), metadata);
            return id;
        }

        // Normalize
        let mut unit = vector.to_vec();
        normalize(&mut unit);

        // Quantize
        let code = quantize(&unit);

        let meta = metadata.unwrap_or_default();
        let ts = now_timestamp();

        // Find slot
        let slot = if let Some(slot) = self.free_slots.pop() {
            self.codes[slot] = code;
            self.vectors[slot] = unit;
            self.metadata[slot] = meta;
            self.timestamps[slot] = ts;
            slot
        } else {
            let slot = self.codes.len();
            self.codes.push(code);
            self.vectors.push(unit);
            self.metadata.push(meta);
            self.timestamps.push(ts);
            slot
        };

        self.id_to_slot.insert(id.clone(), slot);
        self.slot_to_id.insert(slot, id.clone());

        id
    }

    /// Update an existing vector and/or its metadata.
    pub fn update(
        &mut self,
        id: &str,
        vector: Option<&[f32]>,
        metadata: Option<HashMap<String, String>>,
    ) -> bool {
        let slot = match self.id_to_slot.get(id) {
            Some(&s) => s,
            None => return false,
        };

        if let Some(v) = vector {
            assert_eq!(v.len(), self.dim);
            let mut unit = v.to_vec();
            normalize(&mut unit);
            self.codes[slot] = quantize(&unit);
            self.vectors[slot] = unit;
        }

        if let Some(meta) = metadata {
            self.metadata[slot] = meta;
        }

        self.timestamps[slot] = now_timestamp();
        true
    }

    /// Delete a vector by ID.
    pub fn delete(&mut self, id: &str) -> bool {
        let slot = match self.id_to_slot.remove(id) {
            Some(s) => s,
            None => return false,
        };
        self.slot_to_id.remove(&slot);
        self.free_slots.push(slot);
        true
    }

    /// Search with two-stage retrieval.
    ///
    /// Returns (scores, ids, metadatas).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> (Vec<f32>, Vec<String>, Vec<HashMap<String, String>>) {
        let active_slots: Vec<usize> = self.slot_to_id.keys().copied().collect();
        if active_slots.is_empty() {
            return (vec![], vec![], vec![]);
        }

        assert_eq!(query.len(), self.dim);
        let mut query_unit = query.to_vec();
        normalize(&mut query_unit);

        // Stage 1: binary Hamming distance on active slots
        let query_code = quantize(&query_unit);
        let codes_flat: Vec<u8> = active_slots.iter()
            .flat_map(|&s| self.codes[s].iter().copied())
            .collect();
        let hamming_dists = hamming_distance_one_to_many(
            &query_code, &codes_flat, active_slots.len(), self.n_bytes
        );

        // Select candidates
        let n_candidates = (k * self.rerank_factor).min(active_slots.len());
        let mut indexed: Vec<(u32, usize)> = hamming_dists.into_iter()
            .enumerate().map(|(i, d)| (d, i)).collect();
        if n_candidates < indexed.len() {
            indexed.select_nth_unstable_by(n_candidates - 1, |a, b| a.0.cmp(&b.0));
            indexed.truncate(n_candidates);
        }

        // Stage 2: float32 rerank
        let mut scored: Vec<(f32, usize)> = indexed.iter().map(|&(_, local)| {
            let slot = active_slots[local];
            let score = inner_product(&query_unit, &self.vectors[slot]);
            (score, slot)
        }).collect();

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(k);

        let scores: Vec<f32> = scored.iter().map(|(s, _)| *s).collect();
        let ids: Vec<String> = scored.iter()
            .map(|(_, slot)| self.slot_to_id[slot].clone())
            .collect();
        let metas: Vec<HashMap<String, String>> = scored.iter()
            .map(|(_, slot)| self.metadata[*slot].clone())
            .collect();

        (scores, ids, metas)
    }

    /// Get metadata for a vector by ID.
    pub fn get(&self, id: &str) -> Option<&HashMap<String, String>> {
        let slot = self.id_to_slot.get(id)?;
        Some(&self.metadata[*slot])
    }

    pub fn len(&self) -> usize {
        self.id_to_slot.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_slot.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.id_to_slot.contains_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let dim = 8;
        let mut index = StreamingIndex::new(dim, 10);

        let v1 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let v2 = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

        let id1 = index.insert(&v1, Some("a".to_string()), None);
        let id2 = index.insert(&v2, Some("b".to_string()), None);

        assert_eq!(index.len(), 2);

        let (scores, ids, _) = index.search(&v1, 1);
        assert_eq!(ids[0], "a");
        assert!(scores[0] > 0.99);
    }

    #[test]
    fn test_delete() {
        let dim = 8;
        let mut index = StreamingIndex::new(dim, 10);

        let v = vec![1.0; 8];
        index.insert(&v, Some("x".to_string()), None);
        assert_eq!(index.len(), 1);

        assert!(index.delete("x"));
        assert_eq!(index.len(), 0);
        assert!(!index.delete("x")); // already deleted
    }
}
