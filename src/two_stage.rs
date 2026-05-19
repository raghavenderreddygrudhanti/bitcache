//! Two-stage progressive retrieval: binary coarse filter + float32 rerank.
//!
//! Stage 1: Scan all binary codes via Hamming distance (fast, cheap).
//!          Select top-N candidates (N >> k).
//! Stage 2: Rerank only those N candidates using float32 inner product (precise).
//!          Return top-k.

use crate::quantize::{normalize, normalize_batch, quantize, quantize_batch};
use crate::search::{hamming_distance_one_to_many, inner_product};

/// Two-stage retrieval: binary filter → float32 rerank.
pub struct TwoStageIndex {
    dim: usize,
    n_bytes: usize,
    n_vectors: usize,
    rerank_factor: usize,
    codes: Vec<u8>,       // binary codes: n * n_bytes
    vectors: Vec<f32>,    // unit float vectors: n * dim
}

impl TwoStageIndex {
    pub fn new(dim: usize, rerank_factor: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        Self {
            dim,
            n_bytes: (dim + 7) / 8,
            n_vectors: 0,
            rerank_factor,
            codes: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Add vectors to the index.
    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);

        let mut unit_vectors = vectors.to_vec();
        normalize_batch(&mut unit_vectors, n, self.dim);

        let codes = quantize_batch(&unit_vectors, n, self.dim);

        self.codes.extend_from_slice(&codes);
        self.vectors.extend_from_slice(&unit_vectors);
        self.n_vectors += n;
    }

    /// Two-stage search: binary filter → float32 rerank.
    ///
    /// Returns (scores, indices). Scores are float32 inner products.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
        if self.n_vectors == 0 {
            return (vec![], vec![]);
        }
        assert_eq!(query.len(), self.dim);

        let mut query_unit = query.to_vec();
        normalize(&mut query_unit);

        let query_code = quantize(&query_unit);

        // Stage 1: binary Hamming scan
        let n_candidates = k.saturating_mul(self.rerank_factor).min(self.n_vectors);
        let hamming_dists = hamming_distance_one_to_many(&query_code, &self.codes, self.n_vectors, self.n_bytes);

        // Select top-N candidates by lowest Hamming distance
        let mut indexed: Vec<(u32, usize)> = hamming_dists.into_iter().enumerate().map(|(i, d)| (d, i)).collect();
        if n_candidates < indexed.len() {
            indexed.select_nth_unstable_by(n_candidates - 1, |a, b| a.0.cmp(&b.0));
            indexed.truncate(n_candidates);
        }

        // Stage 2: float32 rerank
        let mut scored: Vec<(f32, usize)> = indexed
            .iter()
            .map(|&(_, idx)| {
                let v = &self.vectors[idx * self.dim..(idx + 1) * self.dim];
                let score = inner_product(&query_unit, v);
                (score, idx)
            })
            .collect();

        // Sort descending by score, take top-k
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(k);

        let scores: Vec<f32> = scored.iter().map(|(s, _)| *s).collect();
        let indices: Vec<usize> = scored.iter().map(|(_, i)| *i).collect();
        (scores, indices)
    }

    /// Batch search (parallel via Rayon).
    pub fn search_batch(&self, queries: &[f32], k: usize) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        use rayon::prelude::*;
        let nq = queries.len() / self.dim;

        let results: Vec<(Vec<f32>, Vec<usize>)> = (0..nq)
            .into_par_iter()
            .map(|i| {
                let q = &queries[i * self.dim..(i + 1) * self.dim];
                self.search(q, k)
            })
            .collect();

        let all_scores: Vec<Vec<f32>> = results.iter().map(|(s, _)| s.clone()).collect();
        let all_indices: Vec<Vec<usize>> = results.iter().map(|(_, i)| i.clone()).collect();
        (all_scores, all_indices)
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Total memory: binary codes + float vectors.
    pub fn memory_usage_bytes(&self) -> usize {
        self.codes.len() + self.vectors.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_stage_self_retrieval() {
        let dim = 32;
        let mut index = TwoStageIndex::new(dim, 10);

        // Random-ish vectors
        let mut vectors: Vec<f32> = (0..dim * 100)
            .map(|i| ((i * 7 + 3) % 17) as f32 - 8.0)
            .collect();

        index.add(&vectors);
        assert_eq!(index.len(), 100);

        // Query with first vector should find itself
        let query = &vectors[0..dim];
        let (scores, indices) = index.search(query, 1);
        assert_eq!(indices[0], 0);
        assert!(scores[0] > 0.99);
    }
}
