//! Three-stage progressive retrieval: binary → 4-bit → float32.
//!
//! Stage 1: Binary Hamming scan (1-bit per dim). Cheap. Selects broad candidates.
//! Stage 2: 4-bit quantized inner product on candidates. Medium cost. Refines ranking.
//! Stage 3: Float32 inner product on top candidates. Precise. Final ranking.

use crate::quantize::{normalize, normalize_batch, quantize, quantize_batch};
use crate::search::{hamming_distance_one_to_many, inner_product};

/// Three-stage retrieval: binary filter → 4-bit rerank → float32 rerank.
pub struct ThreeStageIndex {
    dim: usize,
    n_bytes: usize,
    n_vectors: usize,
    stage1_factor: usize,
    stage2_factor: usize,
    n_bits: usize,
    n_levels: usize,
    binary_codes: Vec<u8>,   // n * n_bytes
    quant_codes: Vec<u8>,    // n * dim (4-bit stored as u8 per dim)
    vectors: Vec<f32>,       // n * dim
    centroids: Vec<f32>,     // n_levels * dim
    vmin: Vec<f32>,          // dim
    vrange: Vec<f32>,        // dim
}

impl ThreeStageIndex {
    pub fn new(dim: usize, stage1_factor: usize, stage2_factor: usize, n_bits: usize) -> Self {
        assert!(dim > 0);
        assert!((2..=8).contains(&n_bits));
        Self {
            dim,
            n_bytes: (dim + 7) / 8,
            n_vectors: 0,
            stage1_factor,
            stage2_factor,
            n_bits,
            n_levels: 1 << n_bits,
            binary_codes: Vec::new(),
            quant_codes: Vec::new(),
            vectors: Vec::new(),
            centroids: Vec::new(),
            vmin: vec![0.0; dim],
            vrange: vec![1.0; dim],
        }
    }

    /// Add vectors to the index.
    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);

        let mut unit_vectors = vectors.to_vec();
        normalize_batch(&mut unit_vectors, n, self.dim);

        // Stage 1: binary codes
        let binary_codes = quantize_batch(&unit_vectors, n, self.dim);

        // Compute per-dimension min/max for scalar quantization
        let mut vmin = vec![f32::MAX; self.dim];
        let mut vmax = vec![f32::MIN; self.dim];
        for i in 0..n {
            for d in 0..self.dim {
                let val = unit_vectors[i * self.dim + d];
                if val < vmin[d] { vmin[d] = val; }
                if val > vmax[d] { vmax[d] = val; }
            }
        }
        // Include existing vectors in range computation
        for i in 0..self.n_vectors {
            for d in 0..self.dim {
                let val = self.vectors[i * self.dim + d];
                if val < vmin[d] { vmin[d] = val; }
                if val > vmax[d] { vmax[d] = val; }
            }
        }

        let vrange: Vec<f32> = vmin.iter().zip(vmax.iter())
            .map(|(mn, mx)| {
                let r = mx - mn;
                if r > 1e-10 { r } else { 1.0 }
            })
            .collect();

        self.vmin = vmin;
        self.vrange = vrange;

        // Stage 2: scalar quantize all vectors (including existing)
        let total_n = self.n_vectors + n;
        let mut all_vectors = self.vectors.clone();
        all_vectors.extend_from_slice(&unit_vectors);

        let quant_codes = self.scalar_quantize(&all_vectors, total_n);

        // Compute centroids
        let centroids = self.compute_centroids();

        self.binary_codes.extend_from_slice(&binary_codes);
        self.vectors.extend_from_slice(&unit_vectors);
        self.quant_codes = quant_codes;
        self.centroids = centroids;
        self.n_vectors = total_n;
    }

    /// Three-stage search.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
        if self.n_vectors == 0 {
            return (vec![], vec![]);
        }
        assert_eq!(query.len(), self.dim);

        let mut query_unit = query.to_vec();
        normalize(&mut query_unit);

        // Stage 1: binary Hamming
        let query_binary = quantize(&query_unit);
        let n_stage1 = (k * self.stage1_factor).min(self.n_vectors);
        let hamming_dists = hamming_distance_one_to_many(&query_binary, &self.binary_codes, self.n_vectors, self.n_bytes);

        let mut stage1: Vec<(u32, usize)> = hamming_dists.into_iter().enumerate().map(|(i, d)| (d, i)).collect();
        if n_stage1 < stage1.len() {
            stage1.select_nth_unstable_by(n_stage1 - 1, |a, b| a.0.cmp(&b.0));
            stage1.truncate(n_stage1);
        }
        let stage1_indices: Vec<usize> = stage1.iter().map(|(_, i)| *i).collect();

        // Stage 2: quantized inner product
        let n_stage2 = (k * self.stage2_factor).min(stage1_indices.len());
        let quant_scores = self.score_quantized(&query_unit, &stage1_indices);

        let mut stage2: Vec<(f32, usize)> = quant_scores.into_iter()
            .zip(stage1_indices.iter())
            .map(|(s, &i)| (s, i))
            .collect();
        stage2.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        stage2.truncate(n_stage2);

        // Stage 3: float32 rerank
        let mut final_scored: Vec<(f32, usize)> = stage2
            .iter()
            .map(|&(_, idx)| {
                let v = &self.vectors[idx * self.dim..(idx + 1) * self.dim];
                let score = inner_product(&query_unit, v);
                (score, idx)
            })
            .collect();

        final_scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        final_scored.truncate(k);

        let scores: Vec<f32> = final_scored.iter().map(|(s, _)| *s).collect();
        let indices: Vec<usize> = final_scored.iter().map(|(_, i)| *i).collect();
        (scores, indices)
    }

    fn scalar_quantize(&self, vectors: &[f32], n: usize) -> Vec<u8> {
        let mut codes = vec![0u8; n * self.dim];
        for i in 0..n {
            for d in 0..self.dim {
                let val = vectors[i * self.dim + d];
                let normalized = (val - self.vmin[d]) / self.vrange[d];
                let level = (normalized * self.n_levels as f32).clamp(0.0, (self.n_levels - 1) as f32) as u8;
                codes[i * self.dim + d] = level;
            }
        }
        codes
    }

    fn compute_centroids(&self) -> Vec<f32> {
        let mut centroids = vec![0.0f32; self.n_levels * self.dim];
        for level in 0..self.n_levels {
            for d in 0..self.dim {
                centroids[level * self.dim + d] = self.vmin[d] + (level as f32 + 0.5) / self.n_levels as f32 * self.vrange[d];
            }
        }
        centroids
    }

    fn score_quantized(&self, query: &[f32], indices: &[usize]) -> Vec<f32> {
        // Build lookup table: lut[level * dim + d] = query[d] * centroid[level, d]
        let mut lut = vec![0.0f32; self.n_levels * self.dim];
        for level in 0..self.n_levels {
            for d in 0..self.dim {
                lut[level * self.dim + d] = query[d] * self.centroids[level * self.dim + d];
            }
        }

        indices.iter().map(|&idx| {
            let mut score = 0.0f32;
            for d in 0..self.dim {
                let level = self.quant_codes[idx * self.dim + d] as usize;
                score += lut[level * self.dim + d];
            }
            score
        }).collect()
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_stage_basic() {
        let dim = 16;
        let mut index = ThreeStageIndex::new(dim, 50, 10, 4);

        let vectors: Vec<f32> = (0..dim * 50)
            .map(|i| ((i * 13 + 7) % 19) as f32 - 9.0)
            .collect();

        index.add(&vectors);
        assert_eq!(index.len(), 50);

        let query = &vectors[0..dim];
        let (scores, indices) = index.search(query, 5);
        assert!(!indices.is_empty());
        assert!(scores[0] > 0.9);
    }
}
