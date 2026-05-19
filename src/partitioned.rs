//! Partition-aware staged retrieval using binary k-means routing.
//!
//! Clusters binary codes into P partitions at build time. At search time,
//! routes the query to the top-R most relevant partitions and scans only those.

use crate::quantize::{normalize, normalize_batch, quantize, quantize_batch};
use crate::search::{hamming_distance, hamming_distance_one_to_many, inner_product};

/// Partition-aware two-stage retrieval with binary k-means routing.
pub struct PartitionedIndex {
    dim: usize,
    n_bytes: usize,
    n_vectors: usize,
    n_partitions: usize,
    n_probe: usize,
    rerank_factor: usize,
    codes: Vec<u8>,           // n * n_bytes
    vectors: Vec<f32>,        // n * dim (unit normalized)
    centroids: Vec<u8>,       // n_partitions * n_bytes
    assignments: Vec<usize>,  // n — partition ID per vector
    partition_indices: Vec<Vec<usize>>, // partition_id -> vector indices
}

impl PartitionedIndex {
    pub fn new(dim: usize, n_partitions: usize, n_probe: usize, rerank_factor: usize) -> Self {
        assert!(dim > 0);
        Self {
            dim,
            n_bytes: (dim + 7) / 8,
            n_vectors: 0,
            n_partitions,
            n_probe,
            rerank_factor,
            codes: Vec::new(),
            vectors: Vec::new(),
            centroids: Vec::new(),
            assignments: Vec::new(),
            partition_indices: Vec::new(),
        }
    }

    /// Build the partitioned index from vectors.
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

        // Binary k-means clustering
        let (centroids, assignments) = self.binary_kmeans(&self.codes.clone(), n, 20);
        self.centroids = centroids;
        self.assignments = assignments;

        // Build partition index
        self.partition_indices = vec![Vec::new(); self.n_partitions];
        for (i, &pid) in self.assignments.iter().enumerate() {
            self.partition_indices[pid].push(i);
        }
    }

    /// Partition-routed search.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
        if self.n_vectors == 0 {
            return (vec![], vec![]);
        }
        assert_eq!(query.len(), self.dim);

        let mut query_unit = query.to_vec();
        normalize(&mut query_unit);

        let query_code = quantize(&query_unit);

        // Route: find closest partitions by Hamming to centroids
        let centroid_dists = hamming_distance_one_to_many(
            &query_code, &self.centroids, self.n_partitions, self.n_bytes
        );
        let mut partition_order: Vec<(u32, usize)> = centroid_dists.into_iter()
            .enumerate().map(|(i, d)| (d, i)).collect();
        partition_order.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let n_probe = self.n_probe.min(self.n_partitions);

        // Gather candidates from probed partitions
        let mut candidate_indices: Vec<usize> = Vec::new();
        for &(_, pid) in partition_order.iter().take(n_probe) {
            candidate_indices.extend_from_slice(&self.partition_indices[pid]);
        }

        if candidate_indices.is_empty() {
            return (vec![], vec![]);
        }

        // Binary scan within candidates
        let candidate_codes: Vec<u8> = candidate_indices.iter()
            .flat_map(|&i| self.codes[i * self.n_bytes..(i + 1) * self.n_bytes].iter().copied())
            .collect();
        let hamming_dists = hamming_distance_one_to_many(
            &query_code, &candidate_codes, candidate_indices.len(), self.n_bytes
        );

        // Select top rf*k by Hamming
        let n_rerank = (k * self.rerank_factor).min(candidate_indices.len());
        let mut scored: Vec<(u32, usize)> = hamming_dists.into_iter()
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

    fn binary_kmeans(&self, codes: &[u8], n: usize, max_iter: usize) -> (Vec<u8>, Vec<usize>) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);

        let k = self.n_partitions.min(n);

        // Initialize centroids by random selection
        let mut centroid_indices: Vec<usize> = (0..n).collect();
        centroid_indices.shuffle(&mut rng);
        centroid_indices.truncate(k);

        let mut centroids: Vec<u8> = centroid_indices.iter()
            .flat_map(|&i| codes[i * self.n_bytes..(i + 1) * self.n_bytes].iter().copied())
            .collect();

        let mut assignments = vec![0usize; n];

        for _ in 0..max_iter {
            // Assign each vector to nearest centroid
            let mut new_assignments = vec![0usize; n];
            for i in 0..n {
                let code = &codes[i * self.n_bytes..(i + 1) * self.n_bytes];
                let mut best_dist = u32::MAX;
                let mut best_c = 0;
                for c in 0..k {
                    let centroid = &centroids[c * self.n_bytes..(c + 1) * self.n_bytes];
                    let dist = hamming_distance(code, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                new_assignments[i] = best_c;
            }

            if new_assignments == assignments {
                break;
            }
            assignments = new_assignments;

            // Update centroids: majority vote per bit
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
                if members.is_empty() {
                    // Reinitialize empty cluster
                    let rand_idx = rng.gen_range(0..n);
                    centroids[c * self.n_bytes..(c + 1) * self.n_bytes]
                        .copy_from_slice(&codes[rand_idx * self.n_bytes..(rand_idx + 1) * self.n_bytes]);
                } else {
                    // Majority vote per bit
                    for byte_idx in 0..self.n_bytes {
                        let mut byte_val = 0u8;
                        for bit in 0..8 {
                            let count: usize = members.iter()
                                .filter(|&&m| codes[m * self.n_bytes + byte_idx] & (1 << (7 - bit)) != 0)
                                .count();
                            if count * 2 > members.len() {
                                byte_val |= 1 << (7 - bit);
                            }
                        }
                        centroids[c * self.n_bytes + byte_idx] = byte_val;
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
    fn test_partitioned_build_and_search() {
        let dim = 32;
        let n = 200;
        let vectors: Vec<f32> = (0..dim * n)
            .map(|i| ((i * 7 + 3) % 17) as f32 - 8.0)
            .collect();

        let mut index = PartitionedIndex::new(dim, 8, 4, 50);
        index.build(&vectors);
        assert_eq!(index.len(), n);

        let query = &vectors[0..dim];
        let (scores, indices) = index.search(query, 5);
        assert!(!indices.is_empty());
        assert!(scores[0] > 0.9);
    }
}
