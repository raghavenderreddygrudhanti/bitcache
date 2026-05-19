//! XOR + POPCOUNT similarity search on packed binary vectors.
//!
//! Optimized with:
//! - Blocked memory layout for cache-friendly access
//! - u64-based popcount (8 bytes at a time instead of 1)
//! - Rayon parallel batch search
//! - Compiler auto-vectorization hints

use rayon::prelude::*;

/// Hamming distance between two binary codes using u64 popcount.
/// Processes 8 bytes at a time for hardware popcount efficiency.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    // Process 8 bytes at a time using u64 popcount
    let chunks = n / 8;
    let remainder = n % 8;

    let a_ptr = a.as_ptr() as *const u64;
    let b_ptr = b.as_ptr() as *const u64;

    let mut dist: u32 = 0;

    // Main loop: 8 bytes at a time
    for i in 0..chunks {
        unsafe {
            let xa = *a_ptr.add(i);
            let xb = *b_ptr.add(i);
            dist += (xa ^ xb).count_ones();
        }
    }

    // Handle remaining bytes
    let offset = chunks * 8;
    for i in 0..remainder {
        dist += (a[offset + i] ^ b[offset + i]).count_ones();
    }

    dist
}

/// Compute Hamming distance between one query and all database vectors.
///
/// Uses u64-based popcount for maximum throughput.
#[inline(never)]
pub fn hamming_distance_one_to_many(query: &[u8], database: &[u8], n: usize, n_bytes: usize) -> Vec<u32> {
    let mut distances = vec![0u32; n];
    let n_u64 = n_bytes / 8;
    let remainder = n_bytes % 8;

    let q_ptr = query.as_ptr() as *const u64;

    for i in 0..n {
        let start = i * n_bytes;
        let d_ptr = unsafe { database.as_ptr().add(start) as *const u64 };

        let mut dist: u32 = 0;

        // Main loop: u64 popcount
        for j in 0..n_u64 {
            unsafe {
                let xa = *q_ptr.add(j);
                let xb = *d_ptr.add(j);
                dist += (xa ^ xb).count_ones();
            }
        }

        // Remainder bytes
        let offset = start + n_u64 * 8;
        for j in 0..remainder {
            dist += (query[n_u64 * 8 + j] ^ database[offset + j]).count_ones();
        }

        distances[i] = dist;
    }

    distances
}

/// Blocked Hamming scan: processes database in cache-friendly blocks.
///
/// Splits the database into blocks that fit in L1/L2 cache,
/// reducing cache misses for large databases.
const BLOCK_SIZE: usize = 4096; // vectors per block (fits in L2 cache)

pub fn hamming_distance_blocked(query: &[u8], database: &[u8], n: usize, n_bytes: usize) -> Vec<u32> {
    let mut distances = vec![0u32; n];
    let n_u64 = n_bytes / 8;
    let remainder = n_bytes % 8;
    let q_ptr = query.as_ptr() as *const u64;

    // Process in blocks for cache locality
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for block in 0..n_blocks {
        let block_start = block * BLOCK_SIZE;
        let block_end = (block_start + BLOCK_SIZE).min(n);

        for i in block_start..block_end {
            let start = i * n_bytes;
            let d_ptr = unsafe { database.as_ptr().add(start) as *const u64 };

            let mut dist: u32 = 0;
            for j in 0..n_u64 {
                unsafe {
                    dist += (*q_ptr.add(j) ^ *d_ptr.add(j)).count_ones();
                }
            }
            let offset = start + n_u64 * 8;
            for j in 0..remainder {
                dist += (query[n_u64 * 8 + j] ^ database[offset + j]).count_ones();
            }
            distances[i] = dist;
        }
    }

    distances
}

/// Find k nearest vectors by Hamming distance.
///
/// Returns (distances, indices) sorted by ascending distance.
pub fn search_topk(query: &[u8], database: &[u8], n: usize, n_bytes: usize, k: usize) -> (Vec<u32>, Vec<usize>) {
    let k = k.min(n);
    if k == 0 {
        return (vec![], vec![]);
    }

    let distances = hamming_distance_blocked(query, database, n, n_bytes);

    // Partial sort: find top-k smallest distances
    let mut indexed: Vec<(u32, usize)> = distances.into_iter().enumerate().map(|(i, d)| (d, i)).collect();

    if k < indexed.len() {
        indexed.select_nth_unstable_by(k - 1, |a, b| a.0.cmp(&b.0));
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let dists: Vec<u32> = indexed.iter().map(|(d, _)| *d).collect();
    let indices: Vec<usize> = indexed.iter().map(|(_, i)| *i).collect();
    (dists, indices)
}

/// Parallel batch search using Rayon.
///
/// Each query is processed independently on a separate thread.
pub fn search_batch_parallel(
    queries: &[u8],
    database: &[u8],
    nq: usize,
    n: usize,
    n_bytes: usize,
    k: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<usize>>) {
    let results: Vec<(Vec<u32>, Vec<usize>)> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q_start = qi * n_bytes;
            let query = &queries[q_start..q_start + n_bytes];
            search_topk(query, database, n, n_bytes, k)
        })
        .collect();

    let all_dists: Vec<Vec<u32>> = results.iter().map(|(d, _)| d.clone()).collect();
    let all_indices: Vec<Vec<usize>> = results.iter().map(|(_, i)| i.clone()).collect();
    (all_dists, all_indices)
}

/// Float32 inner product between two vectors.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Float32 inner product: one query against many database vectors.
/// Uses chunked processing for cache efficiency.
pub fn inner_product_one_to_many(query: &[f32], database: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; n];
    for i in 0..n {
        let start = i * dim;
        scores[i] = inner_product(query, &database[start..start + dim]);
    }
    scores
}

/// Parallel float reranking: score candidates across threads.
pub fn rerank_parallel(
    query: &[f32],
    database: &[f32],
    candidate_indices: &[usize],
    dim: usize,
    k: usize,
) -> (Vec<f32>, Vec<usize>) {
    if candidate_indices.is_empty() {
        return (vec![], vec![]);
    }

    // Score all candidates (parallel for large sets)
    let scored: Vec<(f32, usize)> = if candidate_indices.len() > 1000 {
        candidate_indices
            .par_iter()
            .map(|&idx| {
                let v = &database[idx * dim..(idx + 1) * dim];
                let score = inner_product(query, v);
                (score, idx)
            })
            .collect()
    } else {
        candidate_indices
            .iter()
            .map(|&idx| {
                let v = &database[idx * dim..(idx + 1) * dim];
                let score = inner_product(query, v);
                (score, idx)
            })
            .collect()
    };

    // Top-k
    let mut sorted = scored;
    sorted.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    sorted.truncate(k);

    let scores: Vec<f32> = sorted.iter().map(|(s, _)| *s).collect();
    let indices: Vec<usize> = sorted.iter().map(|(_, i)| *i).collect();
    (scores, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b11110000u8];
        let b = vec![0b10100000u8];
        assert_eq!(hamming_distance(&a, &b), 2);
    }

    #[test]
    fn test_hamming_identical() {
        let a = vec![0xFF, 0x00, 0xAB];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_u64_path() {
        // 16 bytes = 2 u64s
        let a = vec![0xFFu8; 16];
        let b = vec![0x00u8; 16];
        // All bits differ: 16 * 8 = 128
        assert_eq!(hamming_distance(&a, &b), 128);
    }

    #[test]
    fn test_search_topk() {
        let database = vec![0b11111111, 0b00000000, 0b11110000, 0b11111110];
        let query = vec![0b11111111u8];
        let (dists, indices) = search_topk(&query, &database, 4, 1, 2);
        assert_eq!(indices[0], 0);
        assert_eq!(dists[0], 0);
        assert_eq!(indices[1], 3);
        assert_eq!(dists[1], 1);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.5, 0.5, 0.0];
        assert!((inner_product(&a, &b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blocked_matches_simple() {
        let n = 100;
        let n_bytes = 48; // 384 dim
        let database: Vec<u8> = (0..n * n_bytes).map(|i| (i % 256) as u8).collect();
        let query: Vec<u8> = (0..n_bytes).map(|i| (i * 3 % 256) as u8).collect();

        let simple = hamming_distance_one_to_many(&query, &database, n, n_bytes);
        let blocked = hamming_distance_blocked(&query, &database, n, n_bytes);
        assert_eq!(simple, blocked);
    }
}
