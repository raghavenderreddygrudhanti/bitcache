//! XOR + POPCOUNT similarity search on packed binary vectors.
//!
//! Hamming distance = number of differing bits between two binary vectors.
//! Computed as: popcount(XOR(a, b)) summed across bytes.
//!
//! Uses hardware popcount intrinsics when available (x86_64, aarch64).

/// Hamming distance between two binary codes of equal length.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distance between one query and all database vectors.
///
/// `query`: single binary code of length `n_bytes`.
/// `database`: flat slice of `n * n_bytes` packed codes.
///
/// Returns a Vec of distances, one per database vector.
pub fn hamming_distance_one_to_many(query: &[u8], database: &[u8], n: usize, n_bytes: usize) -> Vec<u32> {
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * n_bytes;
        let db_code = &database[start..start + n_bytes];
        distances.push(hamming_distance(query, db_code));
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

    let distances = hamming_distance_one_to_many(query, database, n, n_bytes);

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

/// Batch search: find k nearest for each query.
///
/// `queries`: flat slice of `nq * n_bytes`.
/// Returns (distances, indices) each of length `nq * k`.
pub fn search_batch(
    queries: &[u8],
    database: &[u8],
    nq: usize,
    n: usize,
    n_bytes: usize,
    k: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<usize>>) {
    let mut all_dists = Vec::with_capacity(nq);
    let mut all_indices = Vec::with_capacity(nq);

    for qi in 0..nq {
        let q_start = qi * n_bytes;
        let query = &queries[q_start..q_start + n_bytes];
        let (dists, indices) = search_topk(query, database, n, n_bytes, k);
        all_dists.push(dists);
        all_indices.push(indices);
    }

    (all_dists, all_indices)
}

/// Float32 inner product between two vectors.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Float32 inner product: one query against many database vectors.
///
/// `query`: single vector of length `dim`.
/// `database`: flat slice of `n * dim`.
///
/// Returns scores for each database vector.
pub fn inner_product_one_to_many(query: &[f32], database: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        let score = inner_product(query, &database[start..start + dim]);
        scores.push(score);
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b11110000u8];
        let b = vec![0b10100000u8];
        // XOR = 01010000, popcount = 2
        assert_eq!(hamming_distance(&a, &b), 2);
    }

    #[test]
    fn test_hamming_identical() {
        let a = vec![0xFF, 0x00, 0xAB];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_search_topk() {
        // 4 vectors of 1 byte each
        let database = vec![0b11111111, 0b00000000, 0b11110000, 0b11111110];
        let query = vec![0b11111111u8];
        let (dists, indices) = search_topk(&query, &database, 4, 1, 2);
        assert_eq!(indices[0], 0); // exact match
        assert_eq!(dists[0], 0);
        assert_eq!(indices[1], 3); // 1 bit different
        assert_eq!(dists[1], 1);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.5, 0.5, 0.0];
        assert!((inner_product(&a, &b) - 0.5).abs() < 1e-6);
    }
}
