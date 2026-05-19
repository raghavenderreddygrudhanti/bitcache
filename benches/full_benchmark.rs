//! Full benchmark suite for bitcache.
//!
//! Measures: recall@10, latency, QPS, memory, build time, streaming throughput
//! across all index types and scale points.

use std::time::Instant;
use std::collections::HashSet;

use bitcache::{
    BinaryIndex, TwoStageIndex, ThreeStageIndex,
    PartitionedIndex, FloatRoutedIndex, StreamingIndex,
    AgentMemory, GraphMemory,
};

use rand::prelude::*;

/// Generate random unit-normalized vectors (flat layout).
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        let start = i * dim;
        for d in 0..dim {
            vecs[start + d] = rng.gen_range(-1.0..1.0);
        }
        // Normalize
        let norm: f32 = vecs[start..start + dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for d in 0..dim {
                vecs[start + d] /= norm;
            }
        }
    }
    vecs
}

/// Generate clustered vectors (more realistic for embeddings).
fn clustered_vectors(n: usize, dim: usize, n_clusters: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate cluster centers
    let centers = random_vectors(n_clusters, dim, seed + 1000);

    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        let cluster = i % n_clusters;
        let start = i * dim;
        let center_start = cluster * dim;
        for d in 0..dim {
            let noise: f32 = rng.gen::<f32>() * 2.0 * sigma - sigma;
            vecs[start + d] = centers[center_start + d] + noise;
        }
        // Normalize
        let norm: f32 = vecs[start..start + dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for d in 0..dim {
                vecs[start + d] /= norm;
            }
        }
    }
    vecs
}

/// Brute-force exact top-k by inner product (ground truth).
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

/// Compute recall@k.
fn recall_at_k(predicted: &[usize], ground_truth: &[usize]) -> f64 {
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    let hits = predicted.iter().filter(|i| gt_set.contains(i)).count();
    hits as f64 / ground_truth.len() as f64
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  BITCACHE BENCHMARK SUITE (Rust)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let dim = 384;
    let k = 10;

    // ─── Dataset Generation ───────────────────────────────────────────
    println!("Generating datasets...");
    let t = Instant::now();
    let n_db = 99_000;
    let n_queries = 1000;
    let db_vectors = clustered_vectors(n_db, dim, 100, 0.15, 42);
    let query_vectors = random_vectors(n_queries, dim, 99);
    println!("  {} database vectors, {} queries, dim={} [{:.2}s]\n",
        n_db, n_queries, dim, t.elapsed().as_secs_f64());

    // ─── Ground Truth ─────────────────────────────────────────────────
    println!("Computing ground truth (brute-force float32 IP)...");
    let t = Instant::now();
    let ground_truths: Vec<Vec<usize>> = (0..n_queries).map(|qi| {
        let q = &query_vectors[qi * dim..(qi + 1) * dim];
        exact_topk(q, &db_vectors, n_db, dim, k)
    }).collect();
    println!("  Done [{:.2}s]\n", t.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 1: BinaryIndex (flat binary scan)
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  1. BinaryIndex (flat binary scan, no rerank)");
    println!("───────────────────────────────────────────────────────────────");
    {
        let t = Instant::now();
        let mut index = BinaryIndex::new(dim);
        index.add(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let search_time = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let qps = n_queries as f64 / search_time;
        let latency_us = search_time / n_queries as f64 * 1_000_000.0;

        println!("  Build time:       {:.4}s", build_time);
        println!("  Recall@10:        {:.4}", avg_recall);
        println!("  Avg latency:      {:.1} µs", latency_us);
        println!("  QPS:              {:.0}", qps);
        println!("  Memory (codes):   {:.2} MB", index.memory_usage_bytes() as f64 / 1024.0 / 1024.0);
        println!("  Compression:      {:.1}x", index.compression_ratio());
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 2: TwoStageIndex (binary filter + float rerank)
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  2. TwoStageIndex (binary filter → float rerank)");
    println!("───────────────────────────────────────────────────────────────");
    for rf in [10, 50, 100, 500] {
        let t = Instant::now();
        let mut index = TwoStageIndex::new(dim, rf);
        index.add(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let search_time = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let qps = n_queries as f64 / search_time;
        let latency_us = search_time / n_queries as f64 * 1_000_000.0;

        println!("  rf={:<4}  Recall@10={:.4}  Latency={:.0}µs  QPS={:.0}",
            rf, avg_recall, latency_us, qps);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 3: ThreeStageIndex (binary → 4-bit → float)
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  3. ThreeStageIndex (binary → 4-bit → float)");
    println!("───────────────────────────────────────────────────────────────");
    {
        let t = Instant::now();
        let mut index = ThreeStageIndex::new(dim, 200, 20, 4);
        index.add(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let search_time = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let qps = n_queries as f64 / search_time;
        let latency_us = search_time / n_queries as f64 * 1_000_000.0;

        println!("  Build time:       {:.4}s", build_time);
        println!("  Recall@10:        {:.4}", avg_recall);
        println!("  Avg latency:      {:.1} µs", latency_us);
        println!("  QPS:              {:.0}", qps);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 4: FloatRoutedIndex (semantic routing)
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  4. FloatRoutedIndex (float-space semantic routing)");
    println!("───────────────────────────────────────────────────────────────");
    for (p, probe) in [(32, 4), (64, 4), (128, 8)] {
        let t = Instant::now();
        let mut index = FloatRoutedIndex::new(dim, p, probe, 100, 5);
        index.build(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let search_time = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let qps = n_queries as f64 / search_time;
        let latency_us = search_time / n_queries as f64 * 1_000_000.0;
        let scan_pct = index.scan_percentage();

        println!("  P={:<3} probe={:<2}  Recall@10={:.4}  Latency={:.0}µs  QPS={:.0}  Scan={:.1}%  Build={:.2}s",
            p, probe, avg_recall, latency_us, qps, scan_pct, build_time);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 5: Streaming throughput
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  5. StreamingIndex (insert/delete throughput)");
    println!("───────────────────────────────────────────────────────────────");
    {
        let mut index = StreamingIndex::new(dim, 10);
        let n_insert = 50_000;
        let insert_vecs = random_vectors(n_insert, dim, 77);

        let t = Instant::now();
        for i in 0..n_insert {
            let v = &insert_vecs[i * dim..(i + 1) * dim];
            index.insert(v, None, None);
        }
        let insert_time = t.elapsed().as_secs_f64();
        let insert_qps = n_insert as f64 / insert_time;

        // Search
        let t = Instant::now();
        let n_search = 1000;
        for qi in 0..n_search {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            index.search(q, k);
        }
        let search_time = t.elapsed().as_secs_f64();
        let search_qps = n_search as f64 / search_time;

        // Delete
        let t = Instant::now();
        let n_delete = 10_000;
        for i in 0..n_delete {
            index.delete(&format!("vec_{}", i));
        }
        let delete_time = t.elapsed().as_secs_f64();
        let delete_qps = n_delete as f64 / delete_time;

        println!("  Insert:  {:.0} vectors/sec ({} vectors in {:.3}s)", insert_qps, n_insert, insert_time);
        println!("  Search:  {:.0} QPS ({} queries in {:.3}s)", search_qps, n_search, search_time);
        println!("  Delete:  {:.0} ops/sec ({} deletes in {:.3}s)", delete_qps, n_delete, delete_time);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 6: AgentMemory
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  6. AgentMemory (save + retrieve + eviction)");
    println!("───────────────────────────────────────────────────────────────");
    {
        let mut mem = AgentMemory::new(dim, 5000, 0.05, 0.1, 10);
        let n_save = 10_000;
        let mem_vecs = random_vectors(n_save, dim, 55);

        let t = Instant::now();
        for i in 0..n_save {
            let v = &mem_vecs[i * dim..(i + 1) * dim];
            let importance = (i % 10) as f64 / 10.0;
            mem.save_memory(v, &format!("memory_{}", i), importance, None, None);
        }
        let save_time = t.elapsed().as_secs_f64();

        // After eviction, should be at capacity
        println!("  Saved {} memories in {:.3}s ({:.0}/sec)", n_save, save_time, n_save as f64 / save_time);
        println!("  After eviction: {} memories (capacity=5000)", mem.len());

        let t = Instant::now();
        let n_retrieve = 1000;
        for qi in 0..n_retrieve {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            mem.retrieve_memory(q, 5, 0.0);
        }
        let retrieve_time = t.elapsed().as_secs_f64();
        println!("  Retrieve: {:.0} QPS ({} queries in {:.3}s)", n_retrieve as f64 / retrieve_time, n_retrieve, retrieve_time);

        let stats = mem.stats();
        println!("  Stats: mean_importance={:.3}, min={:.3}, max={:.3}",
            stats.mean_importance, stats.min_importance, stats.max_importance);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 7: GraphMemory
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  7. GraphMemory (entity + relation + search)");
    println!("───────────────────────────────────────────────────────────────");
    {
        let mut gm = GraphMemory::new(dim, 2);
        let n_entities = 1000;
        let entity_vecs = random_vectors(n_entities, dim, 33);

        let t = Instant::now();
        for i in 0..n_entities {
            let v = &entity_vecs[i * dim..(i + 1) * dim];
            gm.add_entity(&format!("entity_{}", i), v, Some(&format!("Entity {}", i)), Some("node"));
        }
        let entity_time = t.elapsed().as_secs_f64();

        // Add relations (random graph)
        let t = Instant::now();
        let mut rng = StdRng::seed_from_u64(44);
        let n_relations = 5000;
        for _ in 0..n_relations {
            let src = rng.gen_range(0..n_entities);
            let tgt = rng.gen_range(0..n_entities);
            if src != tgt {
                gm.add_relation(&format!("entity_{}", src), "connects_to", &format!("entity_{}", tgt));
            }
        }
        let relation_time = t.elapsed().as_secs_f64();

        // Search with expansion
        let t = Instant::now();
        let n_search = 500;
        for qi in 0..n_search {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            gm.search(q, 5, true, Some(2));
        }
        let search_time = t.elapsed().as_secs_f64();

        println!("  {} entities added in {:.3}s ({:.0}/sec)", n_entities, entity_time, n_entities as f64 / entity_time);
        println!("  {} relations added in {:.3}s ({:.0}/sec)", n_relations, relation_time, n_relations as f64 / relation_time);
        println!("  Search+expand: {:.0} QPS ({} queries in {:.3}s)", n_search as f64 / search_time, n_search, search_time);
        println!("  Entities: {}, Relations: {}", gm.num_entities(), gm.num_relations());
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK 8: Scale test
    // ═══════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("  8. Scale test (TwoStage rf=100)");
    println!("───────────────────────────────────────────────────────────────");
    for n in [10_000, 50_000, 100_000] {
        let vecs = random_vectors(n, dim, 42);
        let queries = random_vectors(100, dim, 99);

        let t = Instant::now();
        let mut index = TwoStageIndex::new(dim, 100);
        index.add(&vecs);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        for qi in 0..100 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            index.search(q, k);
        }
        let search_time = t.elapsed().as_secs_f64();
        let qps = 100.0 / search_time;
        let latency_us = search_time / 100.0 * 1_000_000.0;

        println!("  n={:<7}  Build={:.3}s  Latency={:.0}µs  QPS={:.0}",
            n, build_time, latency_us, qps);
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
}
