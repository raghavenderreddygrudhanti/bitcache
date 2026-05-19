//! BinaryIndex: the core flat binary vector index.
//!
//! Quantizes float vectors to binary (sign-bit) and searches via
//! Hamming distance. 32x memory compression vs float32.

use crate::quantize::{normalize_batch, quantize_batch};
use crate::search::{search_topk, hamming_distance_one_to_many};

/// Flat binary vector index with XOR + POPCOUNT search.
pub struct BinaryIndex {
    dim: usize,
    n_bytes: usize,
    n_vectors: usize,
    codes: Vec<u8>,
    norms: Vec<f32>,
}

impl BinaryIndex {
    /// Create a new empty index.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        Self {
            dim,
            n_bytes: (dim + 7) / 8,
            n_vectors: 0,
            codes: Vec::new(),
            norms: Vec::new(),
        }
    }

    /// Add vectors to the index.
    ///
    /// `vectors` is a flat f32 slice of shape (n, dim).
    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim, "vectors length must be multiple of dim");

        // Normalize
        let mut unit_vectors = vectors.to_vec();
        let norms = normalize_batch(&mut unit_vectors, n, self.dim);

        // Quantize
        let codes = quantize_batch(&unit_vectors, n, self.dim);

        self.codes.extend_from_slice(&codes);
        self.norms.extend_from_slice(&norms);
        self.n_vectors += n;
    }

    /// Search for k nearest vectors by Hamming distance.
    ///
    /// Returns (distances, indices).
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<u32>, Vec<usize>) {
        if self.n_vectors == 0 {
            return (vec![], vec![]);
        }

        assert_eq!(query.len(), self.dim);

        // Normalize query
        let mut query_unit = query.to_vec();
        crate::quantize::normalize(&mut query_unit);

        // Quantize query
        let query_code = crate::quantize::quantize(&query_unit);

        search_topk(&query_code, &self.codes, self.n_vectors, self.n_bytes, k)
    }

    /// Batch search.
    pub fn search_batch(&self, queries: &[f32], k: usize) -> (Vec<Vec<u32>>, Vec<Vec<usize>>) {
        let nq = queries.len() / self.dim;
        let mut all_dists = Vec::with_capacity(nq);
        let mut all_indices = Vec::with_capacity(nq);

        for i in 0..nq {
            let q = &queries[i * self.dim..(i + 1) * self.dim];
            let (dists, indices) = self.search(q, k);
            all_dists.push(dists);
            all_indices.push(indices);
        }

        (all_dists, all_indices)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Memory used by binary codes in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Compression ratio vs float32 storage.
    pub fn compression_ratio(&self) -> f64 {
        if self.n_vectors == 0 {
            return 0.0;
        }
        let fp32_bytes = self.n_vectors * self.dim * 4;
        fp32_bytes as f64 / self.codes.len() as f64
    }

    /// Get raw codes (for partition routing).
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn n_bytes(&self) -> usize {
        self.n_bytes
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let dim = 8;
        let mut index = BinaryIndex::new(dim);

        // Add 4 vectors
        let vectors = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  // all positive
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, // all negative
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, // alternating
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, // half-half
        ];
        index.add(&vectors);
        assert_eq!(index.len(), 4);

        // Query with all-positive should find vector 0 first
        let query = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let (dists, indices) = index.search(&query, 2);
        assert_eq!(indices[0], 0);
        assert_eq!(dists[0], 0); // exact match
    }

    #[test]
    fn test_compression_ratio() {
        let dim = 384;
        let mut index = BinaryIndex::new(dim);
        let vectors: Vec<f32> = (0..384 * 10).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        index.add(&vectors);
        let ratio = index.compression_ratio();
        assert!((ratio - 32.0).abs() < 0.1);
    }
}
