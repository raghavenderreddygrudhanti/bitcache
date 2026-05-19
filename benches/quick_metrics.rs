//! Quick metrics: self-retrieval recall and partition hit rate.
//! Uses queries drawn FROM the database (like the papers do).

use std::time::Instant;
use std::collections::HashSet;

use bitcache::{TwoStageIndex, FloatRoutedIndex};
use rand::prelude::*;

fn clustered_vectors(n: usize, dim: usize, n_clusters: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<f32> = (0..n_clusters * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    // Normalize centers
    let mut norm_centers = centers.clone();
    for i in 0..n_clusters {
        let start = i * dim;
        let norm: f32 = norm_centers[start..start+dim].iter().map(|x| x*x).sum::<f32>().sqrt();
        if norm > 1e-10 { for d in 0..dim { norm_centers[start+d] /= norm; } }
    }

    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        let cluster = i % n_clusters;
        let start = i * dim;
        let center_start = cluster * dim;
        for d in 0..dim {
            vecs[start + d] = norm_centers[center_start + d] + rng.gen_range(-sigma..sigma);
        }
        let norm: f32 = vecs[start..start+dim].iter().map(|x| x*x).sum::<f32>().sqrt();
        if norm > 1e-10 { for d in 0..dim { vecs[start+d] /= norm; } }
    }
    vecs
}

fn exact_topk(query: &[f32], database: &[f32], n: usize, dim: usize, k: usize) -> Vec<usize> {
    let mut scores: Vec<(f32, usize)> = (0..n).map(|i| {
        let start = i * dim;
        let score: f32 = query.iter().zip(database[start..start + dim].iter())
            .map(|(a, b)| a * b).sum();
        (score, i)
    }).collect();
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scores.iter().take(k).map(|(_, i)| *i).collect()
}

fn recall_at_k(predicted: &[usize], ground_truth: &[usize]) -> f64 {
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    let hits = predicted.iter().filter(|i| gt_set.contains(i)).count();
    hits as f64 / ground_truth.len() as f64
}

fn main() {
    let dim = 384;
    let k = 10;
    let n_db = 20_000;  // Smaller for fast k-means
    let n_queries = 500;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  BITCACHE METRICS (queries from same distribution)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Generate clustered data (simulates real sentence embeddings)
    println!("Generating clustered data (n={}, dim={}, 50 clusters, σ=0.1)...", n_db, dim);
    let db_vectors = clustered_vectors(n_db, dim, 50, 0.1, 42);

    // Use database vectors as queries (self-retrieval, like papers)
    let mut rng = StdRng::seed_from_u64(99);
    let query_indices: Vec<usize> = (0..n_queries).map(|_| rng.gen_range(0..n_db)).collect();

    // Ground truth
    println!("Computing ground truth...");
    let ground_truths: Vec<Vec<usize>> = query_indices.iter().map(|&qi| {
        let q = &db_vectors[qi * dim..(qi + 1) * dim];
        exact_topk(q, &db_vectors, n_db, dim, k)
    }).collect();

    // ─── TwoStage with various rf ───
    println!("\n─── TwoStageIndex (n={}) ───", n_db);
    for rf in [10, 50, 100, 200, 500, 1000] {
        let mut index = TwoStageIndex::new(dim, rf);
        index.add(&db_vectors);

        let t = Instant::now();
        let mut total_recall = 0.0;
        for (i, &qi) in query_indices.iter().enumerate() {
            let q = &db_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[i]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let qps = n_queries as f64 / elapsed;
        let latency_us = elapsed / n_queries as f64 * 1_000_000.0;

        println!("  rf={:<5}  Recall@10={:.4}  Latency={:.0}µs  QPS={:.0}", rf, avg_recall, latency_us, qps);
    }

    // ─── FloatRouted ───
    println!("\n─── FloatRoutedIndex (n={}) ───", n_db);
    for (p, probe) in [(32, 2), (32, 4), (64, 4), (64, 8)] {
        let t = Instant::now();
        let mut index = FloatRoutedIndex::new(dim, p, probe, 100, 5);
        index.build(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for (i, &qi) in query_indices.iter().enumerate() {
            let q = &db_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[i]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let qps = n_queries as f64 / elapsed;
        let latency_us = elapsed / n_queries as f64 * 1_000_000.0;
        let scan_pct = index.scan_percentage();

        println!("  P={:<3} probe={:<2}  Recall@10={:.4}  Latency={:.0}µs  QPS={:.0}  Scan={:.1}%  Build={:.2}s",
            p, probe, avg_recall, latency_us, qps, scan_pct, build_time);
    }

    // ─── Comparison: TwoStage exhaustive vs FloatRouted ───
    println!("\n─── Head-to-head: Exhaustive vs Routed (n={}) ───", n_db);
    {
        // Exhaustive rf=500
        let mut exhaustive = TwoStageIndex::new(dim, 500);
        exhaustive.add(&db_vectors);

        let t = Instant::now();
        let mut recall_exh = 0.0;
        for (i, &qi) in query_indices.iter().enumerate() {
            let q = &db_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = exhaustive.search(q, k);
            recall_exh += recall_at_k(&indices, &ground_truths[i]);
        }
        let time_exh = t.elapsed().as_secs_f64();
        recall_exh /= n_queries as f64;

        // Routed P=32, probe=4
        let mut routed = FloatRoutedIndex::new(dim, 32, 4, 100, 5);
        routed.build(&db_vectors);

        let t = Instant::now();
        let mut recall_rt = 0.0;
        for (i, &qi) in query_indices.iter().enumerate() {
            let q = &db_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = routed.search(q, k);
            recall_rt += recall_at_k(&indices, &ground_truths[i]);
        }
        let time_rt = t.elapsed().as_secs_f64();
        recall_rt /= n_queries as f64;

        println!("  Exhaustive (rf=500):  Recall@10={:.4}  Latency={:.0}µs  QPS={:.0}",
            recall_exh, time_exh / n_queries as f64 * 1e6, n_queries as f64 / time_exh);
        println!("  Routed (P=32,pr=4):   Recall@10={:.4}  Latency={:.0}µs  QPS={:.0}  Scan=12.5%",
            recall_rt, time_rt / n_queries as f64 * 1e6, n_queries as f64 / time_rt);
        println!("  Speedup: {:.1}x", time_exh / time_rt);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
}
