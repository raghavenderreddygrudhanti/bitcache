//! Float-space routed retrieval: semantic routing + binary filtering.
//!
//! Architecture:
//!   1. Float k-means to build partition centroids (preserves semantic neighborhoods)
//!   2. Route query to nearest centroids via float inner product
//!   3. Binary Hamming scan inside selected partitions
//!   4. Float rerank top candidates

use crate::quantize::{normalize, normalize_batch, quantize, quantize_batch};
use crate::search::{hamming_distance_one_to_many, inner_product, inner_product_one_to_many};
use rand::prelude::*;

/// Float-space routed retrieval with binary candidate filtering.
pub struct FloatRoutedIndex {
    dim: usize,
    n_bytes: usize,
    n_vectors: usize,
    n_partitions: usize,
    n_probe: usize,
    rerank_factor: usize,
    kmeans_iter: usize,
    codes: Vec<u8>,              // n * n_bytes
    vectors: Vec<f32>,           // n * dim (unit normalized)
    centroids: Vec<f32>,         // n_partitions * dim
    partition_indices: Vec<Vec<usize>>,
    partition_codes: Vec<Vec<u8>>,  // pre-grouped binary codes per partition
}

impl FloatRoutedIndex {
    pub fn new(
        dim: usize,
        n_partitions: usize,
        n_probe: usize,
        rerank_factor: usize,
        kmeans_iter: usize,
    ) -> Self {
        assert!(dim > 0);
        Self {
            dim,
            n_bytes: (dim + 7) / 8,
            n_vectors: 0,
            n_partitions,
            n_probe,
            rerank_factor,
            kmeans_iter,
            codes: Vec::new(),
            vectors: Vec::new(),
            centroids: Vec::new(),
            partition_indices: Vec::new(),
            partition_codes: Vec::new(),
        }
    }

    /// Build the float-routed index.
    pub fn build(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);

        self.n_vectors = n;

        // Normalize
        let mut unit_vectors = vectors.to_vec();
        normalize_batch(&mut unit_vectors, n, self.dim);
        self.vectors = unit_vectors.clone();

        // Binary quantize
        self.codes = quantize_batch(&unit_vectors, n, self.dim);

        // Float k-means for partition centroids
        let (centroids, assignments) = self.float_kmeans(&unit_vectors, n);
        self.centroids = centroids;

        // Build partition index
        self.partition_indices = vec![Vec::new(); self.n_partitions];
        for (i, &pid) in assignments.iter().enumerate() {
            self.partition_indices[pid].push(i);
        }

        // Pre-group binary codes per partition
        self.partition_codes = self.partition_indices.iter().map(|indices| {
            indices.iter()
                .flat_map(|&i| self.codes[i * self.n_bytes..(i + 1) * self.n_bytes].iter().copied())
                .collect()
        }).collect();
    }

    /// Float-routed search.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
        if self.n_vectors == 0 {
            return (vec![], vec![]);
        }
        assert_eq!(query.len(), self.dim);

        let mut query_unit = query.to_vec();
        normalize(&mut query_unit);

        // Route: float inner product with centroids
        let centroid_scores = inner_product_one_to_many(&query_unit, &self.centroids, self.n_partitions, self.dim);
        let mut partition_order: Vec<(f32, usize)> = centroid_scores.into_iter()
            .enumerate().map(|(i, s)| (s, i)).collect();
        partition_order.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let n_probe = self.n_probe.min(self.n_partitions);

        // Gather candidates from probed partitions
        let query_code = quantize(&query_unit);
        let mut candidate_indices: Vec<usize> = Vec::new();
        let mut candidate_dists: Vec<u32> = Vec::new();

        for &(_, pid) in partition_order.iter().take(n_probe) {
            let p_indices = &self.partition_indices[pid];
            if p_indices.is_empty() {
                continue;
            }
            let p_codes = &self.partition_codes[pid];
            let dists = hamming_distance_one_to_many(&query_code, p_codes, p_indices.len(), self.n_bytes);
            candidate_indices.extend_from_slice(p_indices);
            candidate_dists.extend_from_slice(&dists);
        }

        if candidate_indices.is_empty() {
            return (vec![], vec![]);
        }

        // Select top rf*k by Hamming distance
        let n_rerank = (k * self.rerank_factor).min(candidate_indices.len());
        let mut scored: Vec<(u32, usize)> = candidate_dists.into_iter()
            .enumerate().map(|(local, d)| (d, local)).collect();
        if n_rerank < scored.len() {
            scored.select_nth_unstable_by(n_rerank - 1, |a, b| a.0.cmp(&b.0));
            scored.truncate(n_rerank);
        }

        // Float rerank
        let mut reranked: Vec<(f32, usize)> = scored.iter().map(|&(_, local)| {
            let global = candidate_indices[local];
            let v = &self.vectors[global * self.dim..(global + 1) * self.dim];
            let score = inner_product(&query_unit, v);
            (score, global)
        }).collect();

        reranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        reranked.truncate(k);

        let scores: Vec<f32> = reranked.iter().map(|(s, _)| *s).collect();
        let indices: Vec<usize> = reranked.iter().map(|(_, i)| *i).collect();
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

    fn float_kmeans(&self, vectors: &[f32], n: usize) -> (Vec<f32>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(42);
        let k = self.n_partitions.min(n);

        // k-means++ initialization
        let mut centroids = vec![0.0f32; k * self.dim];

        // First centroid: random
        let first = rng.gen_range(0..n);
        centroids[..self.dim].copy_from_slice(&vectors[first * self.dim..(first + 1) * self.dim]);

        for c in 1..k {
            // Compute distance to nearest existing centroid
            let mut probs: Vec<f32> = Vec::with_capacity(n);
            for i in 0..n {
                let v = &vectors[i * self.dim..(i + 1) * self.dim];
                let mut max_sim = f32::MIN;
                for j in 0..c {
                    let cent = &centroids[j * self.dim..(j + 1) * self.dim];
                    let sim = inner_product(v, cent);
                    if sim > max_sim { max_sim = sim; }
                }
                probs.push((1.0 - max_sim).max(0.0));
            }
            let total: f32 = probs.iter().sum();
            if total > 1e-10 {
                for p in probs.iter_mut() { *p /= total; }
            } else {
                for p in probs.iter_mut() { *p = 1.0 / n as f32; }
            }

            // Weighted random selection
            let threshold: f32 = rng.gen();
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids[c * self.dim..(c + 1) * self.dim]
                .copy_from_slice(&vectors[chosen * self.dim..(chosen + 1) * self.dim]);
        }

        // Iterate
        let mut assignments = vec![0usize; n];
        for _ in 0..self.kmeans_iter {
            // Assign
            let mut new_assignments = vec![0usize; n];
            for i in 0..n {
                let v = &vectors[i * self.dim..(i + 1) * self.dim];
                let mut best_sim = f32::MIN;
                let mut best_c = 0;
                for c in 0..k {
                    let cent = &centroids[c * self.dim..(c + 1) * self.dim];
                    let sim = inner_product(v, cent);
                    if sim > best_sim {
                        best_sim = sim;
                        best_c = c;
                    }
                }
                new_assignments[i] = best_c;
            }

            if new_assignments == assignments {
                break;
            }
            assignments = new_assignments;

            // Update centroids
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
                if members.is_empty() {
                    let rand_idx = rng.gen_range(0..n);
                    centroids[c * self.dim..(c + 1) * self.dim]
                        .copy_from_slice(&vectors[rand_idx * self.dim..(rand_idx + 1) * self.dim]);
                } else {
                    for d in 0..self.dim {
                        let sum: f32 = members.iter().map(|&m| vectors[m * self.dim + d]).sum();
                        centroids[c * self.dim + d] = sum / members.len() as f32;
                    }
                    // Normalize centroid
                    let norm: f32 = (0..self.dim)
                        .map(|d| centroids[c * self.dim + d].powi(2))
                        .sum::<f32>()
                        .sqrt();
                    if norm > 1e-10 {
                        for d in 0..self.dim {
                            centroids[c * self.dim + d] /= norm;
                        }
                    }
                }
            }
        }

        (centroids, assignments)
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    pub fn partition_sizes(&self) -> Vec<usize> {
        self.partition_indices.iter().map(|p| p.len()).collect()
    }

    pub fn scan_percentage(&self) -> f64 {
        if self.n_partitions == 0 { return 100.0; }
        (self.n_probe as f64 / self.n_partitions as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_routed_build_and_search() {
        let dim = 32;
        let n = 500;
        let vectors: Vec<f32> = (0..dim * n)
            .map(|i| ((i * 13 + 5) % 23) as f32 - 11.0)
            .collect();

        let mut index = FloatRoutedIndex::new(dim, 16, 4, 100, 10);
        index.build(&vectors);
        assert_eq!(index.len(), n);

        let query = &vectors[0..dim];
        let (scores, indices) = index.search(query, 5);
        assert!(!indices.is_empty());
        assert!(scores[0] > 0.9);
    }
}
