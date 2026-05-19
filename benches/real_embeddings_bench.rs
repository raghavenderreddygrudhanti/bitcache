//! Benchmark on REAL sentence-transformer embeddings (all-MiniLM-L6-v2).
//!
//! Loads pre-generated embeddings from data/real_embeddings.npy and
//! reproduces the exact metrics from the papers.
//!
//! Prerequisites: run `python experiments/generate_real_embeddings.py` first.

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use bitcache::{
    BinaryIndex, TwoStageIndex, FloatRoutedIndex,
};
use bitcache::partitioned::PartitionedIndex;

// ─── NPY loader (minimal, handles float32 C-contiguous arrays) ───────────

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect(&format!("Cannot open {}. Run: python experiments/generate_real_embeddings.py", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    // Parse numpy .npy format
    // Magic: \x93NUMPY
    assert_eq!(&buf[0..6], b"\x93NUMPY");
    let major = buf[6];
    let _minor = buf[7];

    let header_len = if major == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    };

    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&buf[header_start..header_start + header_len]).unwrap();

    // Parse shape from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (99000, 384), }"
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let dims: Vec<usize> = shape_str.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();

    let n = dims[0];
    let dim = dims[1];

    let data_start = header_start + header_len;
    let data = &buf[data_start..];

    // Convert bytes to f32
    let n_floats = n * dim;
    let mut vectors = Vec::with_capacity(n_floats);
    for i in 0..n_floats {
        let offset = i * 4;
        let val = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
        vectors.push(val);
    }

    (vectors, n, dim)
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
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  REAL EMBEDDINGS BENCHMARK");
    println!("  all-MiniLM-L6-v2, 99K database, 1K queries, dim=384");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load embeddings
    print!("  Loading data/real_embeddings.npy... ");
    let (db_vectors, n_db, dim) = load_npy("data/real_embeddings.npy");
    println!("({} x {})", n_db, dim);

    print!("  Loading data/real_queries.npy... ");
    let (query_vectors, n_queries, _) = load_npy("data/real_queries.npy");
    println!("({} x {})", n_queries, dim);

    let k = 10;

    // Ground truth (brute-force float32 inner product)
    print!("  Computing ground truth (exact top-10)... ");
    let t = Instant::now();
    let ground_truths: Vec<Vec<usize>> = (0..n_queries).map(|qi| {
        let q = &query_vectors[qi * dim..(qi + 1) * dim];
        exact_topk(q, &db_vectors, n_db, dim, k)
    }).collect();
    println!("[{:.1}s]", t.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════════
    println!("\n───────────────────────────────────────────────────────────────────");
    println!("  PAPER 1 RESULTS: Staged Retrieval on Real Embeddings");
    println!("───────────────────────────────────────────────────────────────────\n");

    // Binary only (no rerank)
    {
        let mut index = BinaryIndex::new(dim);
        index.add(&db_vectors);

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let latency_ms = elapsed / n_queries as f64 * 1000.0;
        let qps = n_queries as f64 / elapsed;

        println!("  FAISS Binary equivalent (no rerank):");
        println!("    Recall@10:  {:.4}    (paper: 0.735)", avg_recall);
        println!("    Latency:    {:.2}ms   QPS: {:.0}", latency_ms, qps);
        println!("    Memory:     {:.2} MB (32x compression)", index.memory_usage_bytes() as f64 / 1024.0 / 1024.0);
    }

    // Two-stage with various rf
    println!("\n  Two-stage (Gen1) — recall vs rerank factor:");
    println!("  {:>6} {:>12} {:>12} {:>8}", "rf", "Recall@10", "Latency", "QPS");
    println!("  {}", "-".repeat(44));

    for rf in [10, 50, 100, 200, 500, 1000] {
        let mut index = TwoStageIndex::new(dim, rf);
        index.add(&db_vectors);

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let latency_ms = elapsed / n_queries as f64 * 1000.0;
        let qps = n_queries as f64 / elapsed;

        println!("  {:>6} {:>12.4} {:>10.2}ms {:>8.0}", rf, avg_recall, latency_ms, qps);
    }

    // ═══════════════════════════════════════════════════════════════════
    println!("\n───────────────────────────────────────────────────────────────────");
    println!("  PAPER 2 RESULTS: Float-Space Routing on Real Embeddings");
    println!("───────────────────────────────────────────────────────────────────\n");

    // Float routed (P=128, probe=8) — paper's main config
    // Using smaller P for faster k-means build
    println!("  Float-routed (Gen3):");
    println!("  {:>4} {:>6} {:>12} {:>12} {:>8} {:>8} {:>8}", "P", "probe", "Recall@10", "Latency", "QPS", "Scan%", "Build");
    println!("  {}", "-".repeat(66));

    for (p, probe) in [(32, 4), (64, 4), (64, 8)] {
        let t = Instant::now();
        let mut index = FloatRoutedIndex::new(dim, p, probe, 500, 5);
        index.build(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let latency_ms = elapsed / n_queries as f64 * 1000.0;
        let qps = n_queries as f64 / elapsed;
        let scan_pct = index.scan_percentage();

        println!("  {:>4} {:>6} {:>12.4} {:>10.2}ms {:>8.0} {:>7.1}% {:>7.1}s",
            p, probe, avg_recall, latency_ms, qps, scan_pct, build_time);
    }

    // Binary routing comparison
    println!("\n  Binary-routed (Gen2) for comparison:");
    {
        let p = 32;
        let probe = 4;
        let t = Instant::now();
        let mut index = PartitionedIndex::new(dim, p, probe, 500);
        index.build(&db_vectors);
        let build_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total_recall = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = index.search(q, k);
            total_recall += recall_at_k(&indices, &ground_truths[qi]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = total_recall / n_queries as f64;
        let latency_ms = elapsed / n_queries as f64 * 1000.0;
        let qps = n_queries as f64 / elapsed;

        println!("    Binary routed (P={}, probe={}): Recall@10={:.4}  Latency={:.2}ms  QPS={:.0}  Build={:.1}s",
            p, probe, avg_recall, latency_ms, qps, build_time);
    }

    // Head-to-head speedup
    println!("\n  Head-to-head: Exhaustive vs Float-Routed:");
    {
        let mut exhaustive = TwoStageIndex::new(dim, 500);
        exhaustive.add(&db_vectors);

        let t = Instant::now();
        let mut recall_exh = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = exhaustive.search(q, k);
            recall_exh += recall_at_k(&indices, &ground_truths[qi]);
        }
        let time_exh = t.elapsed().as_secs_f64();
        recall_exh /= n_queries as f64;

        let mut routed = FloatRoutedIndex::new(dim, 32, 4, 500, 5);
        routed.build(&db_vectors);

        let t = Instant::now();
        let mut recall_rt = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = routed.search(q, k);
            recall_rt += recall_at_k(&indices, &ground_truths[qi]);
        }
        let time_rt = t.elapsed().as_secs_f64();
        recall_rt /= n_queries as f64;

        let speedup = time_exh / time_rt;
        println!("    Exhaustive (rf=500):    Recall@10={:.4}  Latency={:.2}ms  QPS={:.0}",
            recall_exh, time_exh / n_queries as f64 * 1000.0, n_queries as f64 / time_exh);
        println!("    Float-routed (P=32,p=4): Recall@10={:.4}  Latency={:.2}ms  QPS={:.0}",
            recall_rt, time_rt / n_queries as f64 * 1000.0, n_queries as f64 / time_rt);
        println!("    Speedup: {:.1}x", speedup);
    }

    // ═══════════════════════════════════════════════════════════════════
    println!("\n───────────────────────────────────────────────────────────────────");
    println!("  COMPARISON TABLE (Paper format)");
    println!("───────────────────────────────────────────────────────────────────\n");

    // Reproduce the exact paper table
    {
        // Gen1 exhaustive rf=10
        let mut gen1 = TwoStageIndex::new(dim, 10);
        gen1.add(&db_vectors);
        let t = Instant::now();
        let mut recall_gen1 = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = gen1.search(q, k);
            recall_gen1 += recall_at_k(&indices, &ground_truths[qi]);
        }
        let time_gen1 = t.elapsed().as_secs_f64();
        recall_gen1 /= n_queries as f64;
        let latency_gen1 = time_gen1 / n_queries as f64 * 1000.0;

        // Gen3 float routed
        let mut gen3 = FloatRoutedIndex::new(dim, 32, 4, 500, 5);
        gen3.build(&db_vectors);
        let t = Instant::now();
        let mut recall_gen3 = 0.0;
        for qi in 0..n_queries {
            let q = &query_vectors[qi * dim..(qi + 1) * dim];
            let (_, indices) = gen3.search(q, k);
            recall_gen3 += recall_at_k(&indices, &ground_truths[qi]);
        }
        let time_gen3 = t.elapsed().as_secs_f64();
        recall_gen3 /= n_queries as f64;
        let latency_gen3 = time_gen3 / n_queries as f64 * 1000.0;

        let speedup = time_gen1 / time_gen3;

        println!("  {:30} {:>10} {:>10} {:>8} {:>8}", "Method", "Recall@10", "Latency", "Scan%", "Speedup");
        println!("  {}", "-".repeat(70));
        println!("  {:30} {:>10.4} {:>8.2}ms {:>8} {:>8}", "Gen1 (exhaustive, rf=10)", recall_gen1, latency_gen1, "100%", "1x");
        println!("  {:30} {:>10.4} {:>8.2}ms {:>8} {:>7.1}x", "Gen3 (float routed, P=32,p=4)", recall_gen3, latency_gen3, "12.5%", speedup);
        println!("  {:30} {:>10} {:>10} {:>8} {:>8}", "Paper: Gen1 (exhaustive)", "0.891", "8.6ms", "100%", "1x");
        println!("  {:30} {:>10} {:>10} {:>8} {:>8}", "Paper: Gen3 (float routed)", "0.892", "3.0ms", "6.2%", "3.8x");
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
}
